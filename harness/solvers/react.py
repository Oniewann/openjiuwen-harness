"""Запасной ReAct-агент openJiuwen с одним инструментом bash.

Это последняя ступень лестницы: её запускают, когда одношаговые решатели не справились
или тип задачи незнаком. Цикл «подумал → выполнил команду → посмотрел вывод» дорогой:
история диалога пересылается на каждом шаге и вся считается во вход. Поэтому:

  * системная инструкция короткая, инструмент один (описание инструментов тоже вход);
  * вывод команд обрезается (начало + конец), длинные файлы в контекст не попадают;
  * в каждом ответе инструмента агент видит, сколько токенов уже потрачено и сколько осталось;
  * число итераций и расход на запуск ограничены; потолок расхода проверяется перед каждым
    обращением к модели (см. LLMHub: агент получает ту же модель через set_llm);
  * перед последним возможным ходом (по счёту ходов или по остатку токенов) ответ инструмента
    требует записать answer.txt или отказаться — иначе агент тратит последний ход на разведку;
  * агент работает в копии каталога задачи: неудачный запуск ничего не портит;
  * ответ агента проходит те же бесплатные проверки, что и ответ модели в одношаговых решателях:
    невычисленное выражение в math, True/False вместо значения (ExprSolver.shape_answer) и заземление
    для search/data_analysis (ExprSolver.ungrounded) — ловится выдуманное, а не стиль записи.
"""
from __future__ import annotations

import asyncio
import itertools
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .. import sandbox
from ..llm import LLMDead, LLMError, TaskCapExceeded
from ..textutil import clip, is_refusal, normalize_answer
from .base import Outcome, is_symbol

if TYPE_CHECKING:
    from .base import Solver

REACT_SYSTEM = (
    "You are a coding agent working in a Linux shell inside the task directory. "
    "Use the bash tool to inspect files and run python3. Be economical: few short commands, "
    "never print large files, no explanations. When you have the result, write it to answer.txt "
    "exactly as the task says, then reply with the single word FINISHED. "
    "If a file, column, record or number the task needs does not exist, do not guess: "
    "do not write answer.txt and reply IMPOSSIBLE right away. Never invent values."
)
LAST_STEP_NOTE = "Last step: write answer.txt now, or reply IMPOSSIBLE if the data needed is missing."
_COUNTER = itertools.count(1)
_CHECKED_FAMILIES = ("math", "data_analysis", "search")


def _answer_problem(solver: "Solver", answer: str) -> str:
    """Замечание к ответу агента, если его нельзя принять; иначе "".

    Проверка ловит выдуманное, а не стиль записи:
      * math — невычисленное выражение с неизвестной величиной из условия («2176 - 3*X»);
      * True/False там, где спрашивают значение (shape_answer отвергает это и после переспроса);
        прочие замечания shape_answer («1024» на «Which of…», «1e-05» на «How much…») — дело вкуса,
        ExprSolver лишь переспросил бы модель, а агенту переспрашивать некогда;
      * ungrounded — слова ответа search/data_analysis не встречаются ни в данных, ни в самом вопросе
        («increase or decrease» → «increase» заземлён вопросом). У остальных семейств сверять не с чем.
    Значение shape_answer не подставляем: здесь проверка, а не переписывание ответа агента.
    Незнакомый тип задачи и задачи со словом-маркером (DONE) не проверяем: вид ответа там неизвестен.
    """
    if not answer or solver.family not in _CHECKED_FAMILIES or solver.task.sentinel:
        return ""
    question = f"{solver.task.question or ''}\n{solver.task.instruction or ''}"
    if solver.family == "math" and _unevaluated(answer, question):
        return (f"the answer {answer[:40]!r} is an expression with an unknown quantity, not a number; "
                "the problem does not give its value")
    try:
        shape = getattr(solver, "shape_answer", None)
        if callable(shape) and answer.strip() in ("True", "False"):
            shaped, problem = shape(answer)
            if shaped is None:
                return problem or "the answer does not fit the question"
        check = getattr(solver, "ungrounded", None)
        if not callable(check) or _in_question(answer, question):
            return ""
        return check(answer) or ""
    except Exception:  # noqa: BLE001 — проверка не должна ронять запуск
        return ""


def _in_question(answer: str, question: str) -> bool:
    """Все слова ответа есть в тексте вопроса: ответ выбран из предложенных («increase or decrease»)."""
    words = re.findall(r"[^\W\d_]{2,}", answer)
    q = question.lower()
    return bool(words) and all(re.search(r"(?<!\w)" + re.escape(w.lower()) + r"(?!\w)", q) for w in words)


# Ответ — само выражение: в таких вопросах буквы в ответе законны («2*x», «(x-1)*(x+1)»).
_SYMBOLIC_Q_RE = re.compile(r"simplif|factor|expand|derivativ|differentiat|integra|in terms of|expression|"
                            r"polynomial|solve for", re.IGNORECASE)


_WITH_UNIT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*[A-Za-z]+(?:\^\d)?(?:/[A-Za-z]+(?:\^\d)?)?")


def _unevaluated(answer: str, question: str) -> bool:
    """Ответ math — невычисленное выражение с неизвестной величиной из условия («2176 - 3*X», «N = 12»).

    shape_answer этого не видит, если вопрос не начинается с «How many…». Обозначение считается
    неизвестной, только если оно стоит в условии само по себе, а не единицей после числа («10 m»),
    как в absent_input. Запись 1e-05, число с единицей («5 m/s»), константа e, имена функций вроде
    sqrt(2) обозначениями не считаются; вопросы, где ответ — само выражение (simplify, factor, derivative…), не проверяются.
    """
    if not re.search(r"[-+*/^=]", answer) or _SYMBOLIC_Q_RE.search(question):
        return False
    if _WITH_UNIT_RE.fullmatch(answer.strip()):
        return False        # число с единицей: «5 m/s», «9.8 m/s^2»
    names = re.findall(r"(?<![\w.])[A-Za-z_]\w*(?![\w(])", answer)
    return any(is_symbol(n) and n != "e" and _unknown_in(n, question) for n in names)


def _unknown_in(name: str, question: str) -> bool:
    """Обозначение встречается в условии как величина, а не как единица измерения после числа."""
    for m in re.finditer(r"(?<![\w'])" + re.escape(name) + r"(?![\w'])", question):
        if not re.search(r"\d\s*$", question[:m.start()]):
            return True
    return False


def _read_answer(work: Path) -> str:
    try:
        return (work / "answer.txt").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


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
    calls_before = len(ledger.calls)
    seen_answer = [""]          # ответ в answer.txt, о котором агента уже предупредили

    def model_calls() -> list:
        return [c for c in ledger.calls[calls_before:] if c.ok and c.task_id == task.task_id and c.step == step]

    def last_step_ahead() -> bool:
        """Следующее обращение агента к модели — последнее.

        Инструмент выполняется на ходу k (k успешных вызовов модели); его вывод модель увидит на ходу k+1.
        Последний ход — либо по счёту (k+1 == react_max_iterations), либо по токенам: вызов идёт, пока
        расход меньше потолка, а каждый следующий не дешевле предыдущего (история растёт), значит после
        хода k+1 места не останется, если остаток меньше полутора последних вызовов.
        """
        calls = model_calls()
        if not calls:
            return False
        if len(calls) + 1 >= cfg.react_max_iterations:
            return True
        left = cap - ledger.task_spent(task.task_id)
        return left < 1.5 * calls[-1].cost

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
        text = text.strip() or "(no output)"
        written = await asyncio.to_thread(_read_answer, work)
        if written and written != seen_answer[0]:
            # Агент только что записал ответ: если его нет в данных (выдумка), говорим сразу,
            # пока есть ход исправить или отказаться.
            seen_answer[0] = written
            problem = _answer_problem(solver, normalize_answer(written) or written)
            if problem:
                problem = problem.split(" (there is")[0].split(". Compute")[0].rstrip(".")
                problem = problem.replace("the code evaluated to", "the answer is")
                text += (f"\n[check] answer.txt: {problem}. "
                         "Fix it using only the given data, or delete answer.txt and reply IMPOSSIBLE.")
        used = ledger.task_spent(task.task_id) - spent_before
        text += f"\n[tokens used {used:.0f} of {cfg.react_token_cap:.0f}]"
        if last_step_ahead():
            text += "\n" + LAST_STEP_NOTE
        return text

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

    answer = await asyncio.to_thread(_read_answer, work)
    refused = bool(final_text) and is_refusal(final_text)
    if answer:
        accepted = await solver.after_react(work, normalize_answer(answer) or answer)
        problem = _answer_problem(solver, accepted) if accepted else ""
        if problem:
            # Ответа нет в данных (память модели, выдумка) или он не того вида — как в ExprSolver.
            solver.note(step, rejected=problem[:160])
            if refused:
                return Outcome("refused", note="agent says the task is impossible")
            return Outcome("failed", note="agent's answer did not pass the answer check")
        if accepted:
            return Outcome("answer", answer=accepted)
        return Outcome("failed", note="agent's result did not pass verification")
    if infra:
        return Outcome("infra", note=infra)
    if refused:
        return Outcome("refused", note="agent says the task is impossible")
    return Outcome("failed", note=note or "agent did not write answer.txt")
