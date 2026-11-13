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
from ..llm import BudgetShort, LLMDead, LLMError, LLMHub, Reply, TaskCapExceeded
from ..pot import PotResult, looks_like_code, run_pot
from ..tasks import Task
from ..textutil import (approx_tokens, extract_code, fit_number_to_question, is_refusal, looks_integer,
                        normalize_answer, strip_think)

# Вес источника при голосовании.
VOTE_WEIGHT = {"skill": 1.5, "fast": 1.0, "fast_alt": 1.0, "strong": 2.0, "react": 1.5}
ROLE_OF_STEP = {"fast": "fast", "fast_retry": "fast", "fast_alt": "fast_alt", "strong": "strong"}
# Ответ оборван по max_tokens без законченного блока кода (см. _interpret).
CUT_OFF_NOTE = "Your previous reply was cut off by the token limit. Reply with the code only, no explanation."


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
    _attempt: Optional[tuple] = None        # (код, результат) последней вычисленной попытки

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

    def shape_answer(self, value: str) -> tuple[Optional[str], str]:
        """Приводит результат выражения к виду, которого ждёт вопрос. (None, причина) — результат непригоден.

        Модель без рассуждений часто пишет выражение «почти про то»: сравнение вместо имени
        («Which company was founded earlier, A or B?» -> A_founded < B_founded -> True), год вместо
        человека, разность с обратным знаком. Тип ожидаемого ответа виден из вопроса — проверяем его
        бесплатно и возвращаем модели понятное замечание.
        """
        from ..tasks import strip_preamble

        q = strip_preamble(self.task.question).strip()
        ql = q.lower()
        v = value.strip()
        yes_no = bool(re.match(r"(?:do|does|did|is|are|was|were|has|have|had|can|could|will|would)\b", ql)) \
            or bool(re.search(r"\byes or no\b|\btrue or false\b", ql))
        if v in ("True", "False"):
            if re.search(r"\btrue or false\b", ql):
                return v, ""
            if yes_no:
                m = re.search(r"\b(yes)\b[^.?!]{0,12}\b(no)\b", q, re.IGNORECASE)
                yes, no = (m.group(1), m.group(2)) if m else ("yes", "no")
                return (yes if v == "True" else no), ""
            return None, ("the code evaluated to True/False, but the question asks for a specific value "
                          "(a name or a number). Return that value itself.")
        wants_name = re.match(r"(?:who|whose|whom|which (?:person|company|product|region|employee|city|one|of)|"
                              r"what is the name|name the)\b", ql)
        if wants_name and re.fullmatch(r"-?\d+(?:\.\d+)?", v):
            return None, f"the question asks for a name, but the code evaluated to the number {v}. Return the name."
        wants_number = re.match(r"(?:how many|how much|in (?:which|what) year|what year|how old)\b", ql)
        if wants_number and not re.fullmatch(r"-?\d+(?:[.,]\d+)?%?", v):
            return None, (f"the question asks for a number, but the code evaluated to {v[:60]!r}. "
                          "Return a single number.")
        if re.search(r"\b(?:how many years|how much) (?:older|younger)\b|\bage (?:difference|gap)\b|\byears apart\b", ql) \
                and re.fullmatch(r"-\d+(?:\.\d+)?", v):
            v = v[1:]   # «на сколько лет старше» — величина, знак зависит только от порядка вычитания
        # вид числа: ровно N знаков для «rounded to N decimal places», иначе как print()
        return fit_number_to_question(v, q, computed=getattr(self, "family", "") != "search"), ""

    def _step_max_tokens(self, step: str) -> int:
        role = ROLE_OF_STEP.get(step, "fast")
        return self.cfg.max_tokens_strong if role == "strong" else (self.max_tokens_fast or self.cfg.max_tokens_fast)

    def _cut_off_max_tokens(self, step: str) -> int:
        """Лимит повтора после обрыва: ×3 от обычного, не выше 4000 (но и не меньше обычного)."""
        usual = self._step_max_tokens(step)
        return max(usual, min(3 * usual, 4000))

    async def _ask(self, step: str, messages: list[dict], temperature: float = 0.0,
                   max_tokens: Optional[int] = None) -> Reply:
        role = ROLE_OF_STEP.get(step, "fast")
        if max_tokens is None:
            max_tokens = self._step_max_tokens(step)
        self.env.estimator.remember_prediction(self.task.task_id, step, self.family, self.features())
        with self.env.hub.scope(self.task.task_id, self.family, step, cap=self.cfg.task_token_cap):
            return await self.env.hub.chat(messages, role=role, max_tokens=max_tokens, temperature=temperature)

    async def _evaluate(self, code: str) -> PotResult:
        return await asyncio.to_thread(run_pot, code, self.preload(code), self.cwd(), self.scratch,
                                       self.cfg.exec_timeout_s)

    def refusal_word(self) -> str:
        """Слово отказа из системной инструкции решателя (search просит NOT_FOUND, остальные — IMPOSSIBLE)."""
        return "NOT_FOUND" if "NOT_FOUND" in self.system_prompt else self.refusal_hint

    def retry_prompt(self, error: str) -> str:
        """Замечание на втором ходу. Повторяет договор целиком: после «исправь код» модель без
        напоминания забывает право на отказ и подставляет выдуманное значение (N = 0, список «из файла»)."""
        return (f"That failed: {error}\nReply with the corrected code only. If a value the answer needs is not "
                f"given in the problem or in the data, reply {self.refusal_word()} instead. "
                "Never invent numbers, names or file contents.")

    async def run_step(self, step: str) -> Outcome:
        if step == "react":
            from .react import run_react

            return await run_react(self, step)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt()}]
        try:
            reply = await self._ask(step, messages, temperature=0.0 if step != "fast_alt" else 0.2)
            self._attempt = None
            outcome, error = await self._interpret(step, reply)
            if outcome is None:
                # одна попытка исправления той же моделью: дешевле, чем подниматься на ступень выше
                self.shaky = True
                first = self._attempt
                if reply.truncated:
                    # Первый ответ оборван по max_tokens: с тем же лимитом повтор оборвётся снова.
                    # Повтор с нуля (обрывок в истории только съел бы лимит) и с лимитом ×3, но не
                    # выше 4000; остаток бюджета урежет его сам (llm.py), а если на увеличенный
                    # вызов не хватает минимума — повтор с обычным лимитом.
                    messages = [messages[0],
                                {"role": "user", "content": messages[1]["content"] + "\n\n" + CUT_OFF_NOTE}]
                    try:
                        reply = await self._ask(step, messages, temperature=0.3,
                                                max_tokens=self._cut_off_max_tokens(step))
                    except BudgetShort:
                        reply = await self._ask(step, messages, temperature=0.3)
                else:
                    messages += [{"role": "assistant", "content": reply.text[:1500]},
                                 {"role": "user", "content": self.retry_prompt(error)}]
                    reply = await self._ask(step, messages, temperature=0.3)
                outcome, error = await self._interpret(step, reply, previous=first)
                if outcome is None:
                    outcome = Outcome("failed", note=error)
            return outcome
        except TaskCapExceeded as exc:
            return Outcome("failed", note=str(exc))
        except LLMDead:
            raise
        except LLMError as exc:
            return Outcome("infra", note=str(exc))

    async def _interpret(self, step: str, reply: Reply,
                         previous: Optional[tuple[str, PotResult]] = None) -> tuple[Optional[Outcome], str]:
        """Разбирает ответ модели. Возвращает (итог, "") либо (None, текст ошибки для исправления).

        `previous` — код и результат первой попытки, если это ответ на замечание.
        """
        text = reply.text
        self.note(step, model=reply.model, cost=round(reply.cost, 1), prompt_tokens=reply.prompt_tokens,
                  completion_tokens=reply.completion_tokens, reasoning_tokens=reply.reasoning_tokens,
                  finish=reply.finish_reason,
                  **({"reply": text[:600]} if self.env.trace_text else {}))
        if is_refusal(text):
            return Outcome("refused", note="model says the task is impossible"), ""
        if not text.strip():
            return None, "empty reply" + (" (cut off by the token limit)" if reply.truncated else "")
        if reply.truncated and not self._closed_code_block(text):
            # Ответ оборван по max_tokens, а законченного блока кода нет: исполнимый префикс
            # обрубка даёт промежуточное значение («price = 12*3.5\nprice» без «* 1.08»),
            # короткое значение без кода тоже могло оборваться. Только на повтор.
            self.note(step, cut_off=True)
            return None, CUT_OFF_NOTE
        code = extract_code(text)
        if not looks_like_code(code):
            # модель ответила значением, а не кодом — принимаем короткий ответ как есть,
            # но для search и data_analysis только если он есть в данных (иначе это память модели)
            plain = normalize_answer(text)
            if plain and len(plain) <= 60 and "\n" not in text.strip():
                missing = self.ungrounded(plain)
                if missing:
                    return None, missing
                return Outcome("answer", answer=self.post_process(plain)), ""
            return None, "reply is not Python code"
        result = await self._evaluate(code)
        self._attempt = (code, result)
        if result.ok and result.value:
            if is_refusal(result.value):
                return Outcome("refused", note="code evaluated to a refusal"), ""
            if previous is not None:
                invented = invented_values(code, previous[0], previous[1],
                                           self.task.question + "\n" + self.user_prompt())
                if invented:
                    # После ошибки «нет такого имени/файла/ключа» исправление вписало значения, которых
                    # нет ни в условии, ни в первой попытке: это выдумка, а не исправление.
                    self.note(step, invented=invented[:5])
                    return Outcome("failed", note=f"corrected code invents values: {', '.join(invented[:3])}"), ""
            shaped, problem = self.shape_answer(result.value)
            if shaped is None:
                return None, problem
            missing = self.ungrounded(shaped, code)
            if missing:
                return None, missing
            return Outcome("answer", answer=self.post_process(shaped)), ""
        if not result.ok:
            absent = self.absent_input(result)
            if absent:
                return Outcome("refused", note=absent), ""
        if result.missing_data and self.refusal_on_missing_data():
            return Outcome("refused", note=result.error), ""
        return None, result.error or "no result"

    @staticmethod
    def _closed_code_block(text: str) -> bool:
        """В ответе есть хотя бы один закрытый блок ``` и нет незакрытого (чётное число ограждений).
        Для оборванного ответа это значит: обрезано пояснение после кода, сам код целый."""
        fences = strip_think(text).count("```")
        return fences >= 2 and fences % 2 == 0

    # ------------------------------------------------------------------ «данных нет» и заземление
    def absent_input(self, result: PotResult) -> str:
        """Причина отказа, если ошибка в коде модели сама означает «нужного значения нет в условии».

        FileNotFoundError на файле, который упомянут в условии и которого нет в каталоге задачи;
        NameError (math) на обозначении из условия вроде «N boxes». Повтор тут бесполезен: модель
        «исправит» код, подставив выдуманное значение. Пустая строка — ошибка обычная.
        """
        question = self.task.question
        if result.etype == "FileNotFoundError":
            m = re.search(r"No such file or directory: '([^']+)'", result.error)
            name = Path(m.group(1)).name if m else ""
            if not name or not re.search(r"(?<![\w.\-])" + re.escape(name) + r"(?![\w\-])", question):
                return ""
            ws = self.task.workspace
            try:
                present = ws.is_dir() and any(p.name == name for p in ws.rglob("*"))
            except OSError:
                present = True
            return "" if present else f"file {name} mentioned in the task does not exist"
        if result.etype == "NameError" and self.family == "math":
            m = re.search(r"name '(\w+)' is not defined", result.error)
            name = m.group(1) if m else ""
            if is_symbol(name) and re.search(r"(?<![\w'])" + re.escape(name) + r"(?![\w'])", question):
                return f"the problem uses the unknown quantity {name} without giving its value"
        return ""

    def grounding_text(self) -> Optional[str]:
        """Текст данных (в нижнем регистре), в котором должен встречаться строковый ответ.

        search — тексты базы знаний, названия записей и значения полей; data_analysis — имена столбцов
        и файлы таблиц как есть. Для остальных семейств None: проверять не по чему.
        """
        if hasattr(self, "_grounding"):
            return self._grounding
        parts: list[str] = []
        try:
            if self.family == "search" and getattr(self, "kb", None) is not None:
                kb = self.kb
                parts += list(kb.texts.values()) + list(kb.entries)
                for fields in kb.entries.values():
                    parts += [str(v) for v in (fields or {}).values()]
            elif self.family == "data_analysis" and getattr(self, "tables", None):
                # значения уже загруженных строк (подходит и для xlsx/json), плюс текст csv как есть
                for table in self.tables:
                    parts += [str(c) for c in table.columns]
                    for row in table.rows[:300_000]:
                        parts += [str(v) for v in row.values() if isinstance(v, str)]
                    if table.path.suffix.lower() in (".csv", ".tsv", ".txt") and table.path.stat().st_size < 40_000_000:
                        parts.append(table.path.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001 — проверка не должна ронять решатель
            parts = []
        self._grounding = "\n".join(parts).lower() if parts else None
        return self._grounding

    def ungrounded(self, answer: str, code: Optional[str] = None) -> str:
        """Замечание, если строковый ответ search/data_analysis не встречается в данных; иначе "".

        Проверяются слова ответа (числа, да/нет, месяцы не проверяются). Для ответа, вычисленного кодом,
        — только слова, которые модель сама вписала строковым литералом: значение, взятое из rows/kb,
        заземлено по построению.
        """
        text = self.grounding_text()
        if not text:
            return ""
        typed = " ".join(code_string_literals(code)).lower() if code is not None else None

        def found(word: str) -> bool:
            w = word.lower()
            # пересказ: «robots» при «robotics» в тексте — сравниваем и по основе слова
            return w in text or (len(w) >= 6 and w[:max(5, len(w) - 2)] in text)

        missing: list[str] = []
        # Элемент перечисления выдуман, если в данных нет ни одного его слова («Alice Novak» в базе без
        # Alice и Novak). Частичное совпадение («industrial robots» при «industrial robotics») — не выдумка.
        for part in re.split(r",|;|\band\b", answer):
            if re.search(r"\d", part):
                continue            # числа и даты вычисляются, их не ищем
            words = [w for w in re.findall(r"[^\W\d_]{2,}", part) if w.lower() not in _UNCHECKED_WORDS]
            if typed is not None:
                words = [w for w in words if w.lower() in typed]
            if words and not any(found(w) for w in words):
                missing.append(words[0])
        if not missing:
            return ""
        source = "the knowledge base" if self.family == "search" else "the data file"
        return (f"{answer[:60]!r} does not occur in {source} (there is no {missing[0]!r} in it). "
                "Compute the answer from the data instead of writing it yourself, "
                f"or reply {self.refusal_word()} if the data does not contain it.")


# Слова ответа, которые не обязаны встречаться в данных: да/нет, служебные, месяцы и дни недели
# (их часто выводят из дат вида 2024-03-15).
_UNCHECKED_WORDS = frozenset(
    "yes no true false none null and or not the of in on at to by for nan inf".split()
    + "january february march april may june july august september october november december".split()
    + "jan feb mar apr jun jul aug sep sept oct nov dec".split()
    + "monday tuesday wednesday thursday friday saturday sunday".split()
)

# Числа, которые исправленный код вправе ввести сам: единицы и обычные константы перевода единиц.
_PLAIN_NUMBERS = frozenset({0, 1, 2, 10, 100, 1000, 7, 12, 24, 30, 31, 52, 60, 365, 366, 3600, 1440, 0.5})

_NUMBER_WORDS = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"dozen|half|quarter|third|twice|double|triple|thrice)\b", re.IGNORECASE)

# Ошибки первой попытки, после которых новые значения в исправлении означают выдумку.
_INVENT_AFTER = ("NameError", "FileNotFoundError", "KeyError", "IndexError", "StopIteration")


def is_symbol(name: str) -> bool:
    """Обозначение величины из условия: одна буква (N, k), две с заглавной (T1, Nx) или слово целиком
    заглавными. «a», «A», «i», «I» — обычные слова английского текста, а не обозначения."""
    return bool(re.fullmatch(r"[A-Za-z]|[A-Z][A-Za-z0-9]|[A-Z][A-Z0-9_]+", name or "")) \
        and name not in ("a", "A", "i", "I")


def code_string_literals(code: str) -> list[str]:
    """Строковые литералы кода (если код не разбирается — всё, что в кавычках)."""
    import ast

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return [a or b for a, b in re.findall(r"'([^'\n]*)'|\"([^\"\n]*)\"", code)]
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for raw in re.findall(r"\d[\d,_]*(?:\.\d+)?", text or ""):
        try:
            found.add(float(raw.replace(",", "").replace("_", "")))
        except ValueError:
            pass
    return found


def invented_values(code: str, first_code: str, first: PotResult, context: str) -> list[str]:
    """Значения, которые исправленный код ввёл после ошибки «нет такого имени/файла/ключа».

    Считаются: присваивание литерала имени, на котором упала первая попытка (N = 0), и числа, которых
    нет ни в условии и данных (`context`), ни в первой попытке, кроме обычных констант (_PLAIN_NUMBERS).
    """
    import ast

    if first.ok or first.etype not in _INVENT_AFTER:
        return []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    found: list[str] = []
    m = re.search(r"name '(\w+)' is not defined", first.error) if first.etype == "NameError" else None
    if m and is_symbol(m.group(1)) and re.search(r"(?<![\w'])" + re.escape(m.group(1)) + r"(?![\w'])", context):
        # имя из условия («N boxes»), а не забытая переменная самой модели (total = 0 — это исправление)
        name = m.group(1)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if any(isinstance(t, ast.Name) and t.id == name for t in targets) and value is not None \
                        and _is_literal(value):
                    found.append(f"{name} = {ast.unparse(value)[:40]}")
    if _NUMBER_WORDS.search(context):
        return found      # числа словами («three hundred») — новые литералы могут быть из условия
    known = _numbers_in(context) | _numbers_in(first_code)
    known |= {round(v / 100, 9) for v in known} | {round(v * 100, 9) for v in known}   # 15% -> 0.15
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            v = round(abs(float(node.value)), 9)
            if v not in known and v not in _PLAIN_NUMBERS:
                found.append(repr(node.value))
    return found


def _is_literal(node) -> bool:
    import ast

    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
