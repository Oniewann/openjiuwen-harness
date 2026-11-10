"""Запасной ReAct-агент openJiuwen с одним инструментом bash.

Это последняя ступень лестницы: её запускают, когда одношаговые решатели не справились
или тип задачи незнаком. Цикл «подумал → выполнил команду → посмотрел вывод» дорогой:
история диалога пересылается на каждом шаге и вся считается во вход. Поэтому:

  * системная инструкция короткая, инструмент один (описание инструментов тоже вход);
  * вывод команд обрезается (начало + конец), длинные файлы в контекст не попадают;
  * в каждом ответе инструмента агент видит, сколько токенов уже потрачено и сколько осталось;
  * число итераций и расход на запуск ограничены; потолок расхода проверяется перед каждым
    обращением к модели (см. LLMHub: агент получает ту же модель через set_llm);
  * агент работает в копии каталога задачи: неудачный запуск ничего не портит.
"""
from __future__ import annotations

import asyncio
import itertools
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .. import sandbox
from ..llm import LLMDead, LLMError, TaskCapExceeded
from ..textutil import clip, is_refusal, normalize_answer
from .base import Outcome

if TYPE_CHECKING:
    from .base import Solver

REACT_SYSTEM = (
    "You are a coding agent working in a Linux shell inside the task directory. "
    "Use the bash tool to inspect files and run python3. Be economical: few short commands, "
    "never print large files, no explanations. When you have the result, write it to answer.txt "
    "exactly as the task says, then reply with the single word FINISHED. "
    "If the task cannot be done, do not write answer.txt and reply IMPOSSIBLE."
)
_COUNTER = itertools.count(1)


def _prepare_copy(solver: "Solver") -> Path:
    work = solver.scratch / "react_ws"
    shutil.rmtree(work, ignore_errors=True)
    if solver.task.workspace.is_dir():
        shutil.copytree(solver.task.workspace, work, symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "answer.txt"))
    else:
        work.mkdir(parents=True, exist_ok=True)
    return work


async def run_react(solver: "Solver", step: str = "react") -> Outcome:
    from openjiuwen.core.foundation.tool import tool
    from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig

    task, cfg, hub = solver.task, solver.cfg, solver.env.hub
    ledger = hub.ledger
    work = await asyncio.to_thread(_prepare_copy, solver)
    spent_before = ledger.task_spent(task.task_id)
    cap = min(cfg.task_token_cap, spent_before + cfg.react_token_cap)
    uid = f"{task.task_id}_{next(_COUNTER)}"

    @tool(name="bash",
          description="Run a bash command in the task directory. Returns stdout, stderr and a non-zero exit "
                      "code; long output is truncated.",
          input_params={"type": "object",
                        "properties": {"command": {"type": "string", "description": "the command to run"}},
                        "required": ["command"]})
    async def bash(command: str) -> str:
        result = await asyncio.to_thread(sandbox.run_shell, str(command), work, 30.0)
        text = clip(result.stdout, 900, 500).rstrip()
        if result.stderr.strip():
            text += "\n[stderr]\n" + clip(result.stderr, 200, 500).rstrip()
        if result.timed_out:
            text += "\n[timed out after 30s]"
        elif result.returncode != 0:
            text += f"\n[exit code {result.returncode}]"
        used = ledger.task_spent(task.task_id) - spent_before
        return f"{text.strip() or '(no output)'}\n[tokens used {used:.0f} of {cfg.react_token_cap:.0f}]"

    agent = ReActAgent(card=AgentCard(id=f"react_{uid}", name=f"react_{uid}",
                                      description="budget-aware fallback agent"))
    agent.configure(ReActAgentConfig()
                    .configure_max_iterations(cfg.react_max_iterations)
                    .configure_prompt_template([{"role": "system", "content": REACT_SYSTEM}])
                    .configure_parallel_tool_calls(False))
    agent.set_llm(hub.model())
    agent.ability_manager.add_ability(bash.card, bash)

    final_text, note, infra = "", "", ""
    try:
        with hub.scope(task.task_id, solver.family, step, cap=cap, max_tokens=cfg.max_tokens_code,
                       temperature=0.0, timeout=cfg.call_timeout_fast, model=cfg.model_for("fast")):
            result = await asyncio.wait_for(
                agent.invoke({"query": task.instruction, "conversation_id": uid}),
                timeout=cfg.call_timeout_fast * 3 + 60)
        if isinstance(result, dict):
            final_text = str(result.get("output") or "")
    except TaskCapExceeded as exc:
        note = str(exc)
    except LLMDead:
        raise
    except LLMError as exc:
        infra = str(exc)
    except asyncio.TimeoutError:
        note = "agent run timed out"
    except Exception as exc:  # noqa: BLE001 — сбой агента не должен ронять прогон
        cause = exc.__cause__ or exc.__context__
        if isinstance(cause, LLMDead) or isinstance(exc.__context__, LLMDead):
            raise LLMDead(str(cause)) from exc
        if isinstance(cause, TaskCapExceeded):
            note = str(cause)
        else:
            note = f"agent error: {type(exc).__name__}: {str(exc)[:200]}"
    finally:
        try:
            agent.ability_manager.remove_ability("bash")
        except Exception:  # noqa: BLE001
            pass

    used = ledger.task_spent(task.task_id) - spent_before
    solver.env.estimator.observe_react(solver.family, used)
    solver.note(step, cost=round(used, 1), final=final_text[:120], note=note or infra)

    answer = ""
    try:
        answer = (work / "answer.txt").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        pass
    if answer:
        accepted = await solver.after_react(work, normalize_answer(answer) or answer)
        if accepted:
            return Outcome("answer", answer=accepted)
        return Outcome("failed", note="agent's result did not pass verification")
    if infra:
        return Outcome("infra", note=infra)
    if final_text and is_refusal(final_text):
        return Outcome("refused", note="agent says the task is impossible")
    return Outcome("failed", note=note or "agent did not write answer.txt")
