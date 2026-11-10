"""Общий каркас решателей.

Решатель задачи — это лестница ступеней от бесплатной к дорогой:

    навык (0 токенов) → дешёвая модель → вторая дешёвая модель → модель с рассуждениями → ReAct-агент

Решатель сам не решает, идти ли выше: он только сообщает, какая ступень следующая
(`next_step`), сколько она примерно стоит (`estimate`) и что уже известно о задаче
(`evidence`). Запускать ли её, решает планировщик — по цене, шансу и остатку бюджета.

Для задач без программной проверки (math, data_analysis, search) ответ подтверждается
голосованием источников: навык, две дешёвые модели, модель с рассуждениями. Совпали два
независимых источника — ответ считается проверенным и дальше токены не тратятся.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import Config
from ..estimator import Estimator, Evidence
from ..llm import LLMDead, LLMError, LLMHub, Reply, TaskCapExceeded
from ..pot import PotResult, looks_like_code, run_pot
from ..tasks import Task
from ..textutil import approx_tokens, extract_code, is_refusal, looks_integer, normalize_answer

# Вес источника при голосовании.
VOTE_WEIGHT = {"skill": 1.5, "fast": 1.0, "fast_alt": 1.0, "strong": 2.0, "react": 1.5}
ROLE_OF_STEP = {"fast": "fast", "fast_retry": "fast", "fast_alt": "fast_alt", "strong": "strong"}


@dataclass
class Env:
    """Общие службы, которые получает каждый решатель."""
    cfg: Config
    hub: LLMHub
    estimator: Estimator
    scratch_root: Path
    trace_text: bool = False


@dataclass
class Outcome:
    kind: str                       # answer | refused | failed | infra
    answer: Optional[str] = None
    note: str = ""


@dataclass
class StepEstimate:
    cost: float                     # ожидаемый расход, выходных токенов
    p_success: float                # шанс, что ступень принесёт баллы
    prompt_tokens: int = 0


class Solver:
    family = "generic"
    ladder: tuple[str, ...] = ("fast", "fast_alt", "strong", "react")

    def __init__(self, task: Task, env: Env):
        self.task = task
        self.env = env
        self.cfg = env.cfg
        self.status = "pending"            # pending | answered | verified | infeasible | gave_up
        self.votes: dict[str, str] = {}    # источник -> ответ
        self.best: Optional[str] = None
        self.refusals = 0
        self.failed_steps: list[str] = []
        self.done: list[str] = []
        self.p_infeasible_static: Optional[float] = None
        self.static_reason = ""
        self.skill_note = ""
        self.shaky = False                 # ответ получен не с первой попытки — повод перепроверить
        self.infra_errors = 0
        self.trace: list[dict] = []
        self.prepared = False
        self.scratch = env.scratch_root / re.sub(r"[^\w.\-]", "_", task.task_id)

    # ------------------------------------------------------------------ подготовка (0 токенов)
    def prepare(self) -> None:
        """Статический разбор задачи без обращений к модели. Переопределяется в наследниках."""
        self.prepared = True

    def mark_static_infeasible(self, probability: float, reason: str) -> None:
        self.p_infeasible_static = max(self.p_infeasible_static or 0.0, probability)
        self.static_reason = reason
        self.note("static", reason=reason, p_infeasible=round(probability, 2))

    def accept_skill(self, answer: str, note: str = "") -> None:
        """Ответ навыка: в режиме fastpath он окончательный, иначе — голос для сверки."""
        self.skill_note = note
        if self.cfg.fastpath:
            self.votes["skill"] = answer
            self.best = answer
            self.status = "verified"
            self.note("skill", answer_by="skill", detail=note)
        elif self.cfg.crosscheck:
            self.votes["skill"] = answer
            self.note("skill", role="crosscheck", detail=note)

    def note(self, step: str, **fields) -> None:
        self.trace.append({"step": step, "t": round(time.time(), 3), **fields})

    # ------------------------------------------------------------------ голосование
    def tally(self) -> list[tuple[str, float]]:
        totals: dict[str, float] = {}
        for source, answer in self.votes.items():
            totals[answer] = totals.get(answer, 0.0) + VOTE_WEIGHT.get(source, 1.0)
        return sorted(totals.items(), key=lambda kv: -kv[1])

    def decide(self) -> None:
        """Пересчитывает лучший ответ и статус после очередной ступени."""
        if self.status in ("verified", "infeasible"):
            return
        ranked = self.tally()
        llm_votes = [s for s in self.votes if s != "skill"]
        if ranked and (llm_votes or self.cfg.fastpath):
            top, weight = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            self.best = top
            self.status = "verified" if weight >= 2.0 and weight - runner_up >= 1.0 else "answered"
            return
        static_refusal = 1 if (self.p_infeasible_static or 0.0) >= 0.7 else 0
        if self.refusals + static_refusal >= 2 and "skill" not in self.votes:
            self.status = "infeasible"

    def absorb(self, step: str, outcome: Outcome) -> None:
        """Учитывает результат ступени."""
        if outcome.kind == "infra":
            self.infra_errors += 1
            return
        self.done.append(step)
        if outcome.kind == "answer" and outcome.answer:
            self.votes[step] = outcome.answer
        elif outcome.kind == "refused":
            self.refusals += 1
        else:
            self.failed_steps.append(step)
        self.decide()

    # ------------------------------------------------------------------ политика
    def wants_verification(self) -> bool:
        """Нужна ли платная перепроверка единственного непроверенного ответа."""
        mode = self.cfg.verify
        if mode == "always":
            return True
        if mode == "never":
            return False
        return self.shaky or (self.p_infeasible_static or 0.0) >= 0.5 or self.answer_looks_odd()

    def answer_looks_odd(self) -> bool:
        """Ответ не похож на то, что спрашивали (например, «how many» — а ответ не целое)."""
        if self.best is None:
            return False
        q = self.task.question.lower()
        if re.search(r"\bhow many\b|\bhow much\b|\bcount\b|\btotal\b", q) and not re.fullmatch(r"-?\d+(\.\d+)?", self.best):
            return True
        if re.search(r"\bhow many\b", q) and self.best.startswith("-"):
            return True
        if re.search(r"\bin which year\b|\bwhat year\b", q) and not re.fullmatch(r"\d{3,4}", self.best):
            return True
        return len(self.best) > 80

    def next_step(self) -> Optional[str]:
        if self.status in ("verified", "infeasible", "gave_up"):
            return None
        remaining = [s for s in self.ladder if s not in self.done]
        if not self.cfg.enable_react:
            remaining = [s for s in remaining if s != "react"]
        if not remaining:
            return None
        paid_done = [s for s in self.done if s != "skill"]
        if not paid_done:
            return remaining[0]
        if self.status == "answered":
            conflict = len(self.tally()) > 1
            arbiters = [s for s in remaining if s in ("fast_alt", "strong")]
            if conflict:
                return arbiters[0] if arbiters else None
            if self.wants_verification() and "fast_alt" in remaining:
                return "fast_alt"
            return None
        return remaining[0]  # ответа нет: отказ или сбой — следующая ступень

    def evidence(self) -> Evidence:
        return Evidence(p_infeasible_static=self.p_infeasible_static,
                        failed_steps=list(self.failed_steps), refusals=self.refusals)

    def features(self) -> dict:
        return {}

    def prompt_text(self, step: str) -> str:
        return self.task.question

    def estimate(self, step: str) -> StepEstimate:
        est = self.env.estimator
        prompt_tokens = approx_tokens(self.prompt_text(step)) + 12
        cost = est.step_cost(self.family, step, prompt_tokens, self.features())
        if self.status == "answered" and step in ("fast_alt", "strong"):
            # Ответ уже записан: ступень нужна только для перепроверки. Её ценность —
            # шанс исправить ошибку, а не шанс решить задачу с нуля.
            conflict = len(self.tally()) > 1
            p = 0.5 if conflict else (0.25 if self.shaky or self.answer_looks_odd() else 0.08)
        else:
            p = est.p_success(self.family, step, self.evidence())
        return StepEstimate(cost=cost, p_success=p, prompt_tokens=prompt_tokens)

    # ------------------------------------------------------------------ выполнение
    async def run_step(self, step: str) -> Outcome:
        raise NotImplementedError

    async def after_react(self, work: Path, answer: str) -> Optional[str]:
        """Принять ли результат ReAct-агента (он работал в копии каталога `work`).

        Возвращает ответ для записи или None, если результат не прошёл проверку.
        """
        return answer

    def last_resort(self) -> None:
        """Последняя бесплатная попытка перед закрытием нерешённой задачи (выполняется в потоке)."""

    def finalize(self) -> None:
        """Вызывается, когда планировщик решил больше не тратить токены на задачу."""
        if self.status != "pending":
            return
        if self.best is None and "skill" in self.votes:
            # Модель так и не дала ответа (сбой, отказ, конец бюджета), а навык задачу разобрал:
            # его ответ лучше пустого файла.
            self.best = self.votes["skill"]
            self.status = "answered"
            self.note("finalize", answer_by="skill_fallback")
        else:
            self.status = "gave_up"


class ExprSolver(Solver):
    """Решатель «модель пишет выражение, харнесс считает» (math, data_analysis, search)."""

    system_prompt = ""
    refusal_hint = "IMPOSSIBLE"
    max_tokens_fast: Optional[int] = None

    def user_prompt(self) -> str:
        return self.task.question

    def preload(self, code: str = "") -> str:
        """Код, который в песочнице готовит данные (rows, kb) перед вычислением выражения модели."""
        return ""

    def cwd(self) -> Path:
        return self.task.workspace

    def prompt_text(self, step: str) -> str:
        return self.system_prompt + "\n" + self.user_prompt()

    def refusal_on_missing_data(self) -> bool:
        """Считать ли KeyError/FileNotFoundError в коде модели признаком «в данных этого нет»."""
        return False

    def post_process(self, value: str) -> str:
        return value

    async def _ask(self, step: str, messages: list[dict], temperature: float = 0.0) -> Reply:
        role = ROLE_OF_STEP.get(step, "fast")
        strong = role == "strong"
        max_tokens = self.cfg.max_tokens_strong if strong else (self.max_tokens_fast or self.cfg.max_tokens_fast)
        self.env.estimator.remember_prediction(self.task.task_id, step, self.family, self.features())
        with self.env.hub.scope(self.task.task_id, self.family, step, cap=self.cfg.task_token_cap):
            return await self.env.hub.chat(messages, role=role, max_tokens=max_tokens, temperature=temperature)

    async def _evaluate(self, code: str) -> PotResult:
        return await asyncio.to_thread(run_pot, code, self.preload(code), self.cwd(), self.scratch,
                                       self.cfg.exec_timeout_s)

    async def run_step(self, step: str) -> Outcome:
        if step == "react":
            from .react import run_react

            return await run_react(self, step)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt()}]
        try:
            reply = await self._ask(step, messages, temperature=0.0 if step != "fast_alt" else 0.2)
            outcome, error = await self._interpret(step, reply)
            if outcome is None:
                # одна попытка исправления той же моделью: дешевле, чем подниматься на ступень выше
                self.shaky = True
                messages += [{"role": "assistant", "content": reply.text[:1500]},
                             {"role": "user", "content": f"That failed: {error}\nReply with the corrected code only."}]
                reply = await self._ask(step, messages, temperature=0.3)
                outcome, error = await self._interpret(step, reply)
                if outcome is None:
                    outcome = Outcome("failed", note=error)
            return outcome
        except TaskCapExceeded as exc:
            return Outcome("failed", note=str(exc))
        except LLMDead:
            raise
        except LLMError as exc:
            return Outcome("infra", note=str(exc))

    async def _interpret(self, step: str, reply: Reply) -> tuple[Optional[Outcome], str]:
        """Разбирает ответ модели. Возвращает (итог, "") либо (None, текст ошибки для исправления)."""
        text = reply.text
        self.note(step, model=reply.model, cost=round(reply.cost, 1), prompt_tokens=reply.prompt_tokens,
                  completion_tokens=reply.completion_tokens, reasoning_tokens=reply.reasoning_tokens,
                  finish=reply.finish_reason,
                  **({"reply": text[:600]} if self.env.trace_text else {}))
        if is_refusal(text):
            return Outcome("refused", note="model says the task is impossible"), ""
        if not text.strip():
            return None, "empty reply" + (" (cut off by the token limit)" if reply.truncated else "")
        code = extract_code(text)
        if not looks_like_code(code):
            # модель ответила значением, а не кодом — принимаем короткий ответ как есть
            plain = normalize_answer(text)
            if plain and len(plain) <= 60 and "\n" not in text.strip():
                return Outcome("answer", answer=self.post_process(plain)), ""
            return None, "reply is not Python code"
        result = await self._evaluate(code)
        if result.ok and result.value:
            if is_refusal(result.value):
                return Outcome("refused", note="code evaluated to a refusal"), ""
            return Outcome("answer", answer=self.post_process(result.value)), ""
        if result.missing_data and self.refusal_on_missing_data():
            return Outcome("refused", note=result.error), ""
        return None, result.error or "no result"
