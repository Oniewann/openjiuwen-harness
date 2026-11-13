"""Решатели задач без программной проверки: math, data_analysis, search и «неизвестный тип».

Общий приём (см. pot.py): модель не «думает вслух», а пишет одно короткое выражение Python
над данными, которые харнесс уже загрузил и описал ей в двух-трёх строках. Вычисляет код.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..pot import import_harness_preamble
from ..tasks import strip_preamble
from ..textutil import approx_tokens
from .base import ExprSolver

_ONLY_CODE = "No explanations, no comments, no code fences."


def _safe(fn, *args, default=None):
    """Навык не должен ронять решатель: любая его ошибка — это просто «навык отказался»."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return default


# --------------------------------------------------------------------------- math
class MathSolver(ExprSolver):
    family = "math"
    system_prompt = (
        "Translate the word problem into Python and reply with ONLY the code: either one expression, "
        "or short statements (one per sentence of the problem, in order) ending with print(answer). "
        "Use // where the text says rounded down. " + _ONLY_CODE + " "
        "If the answer cannot be determined from the text, reply IMPOSSIBLE."
    )

    def prepare(self) -> None:
        from ..skills.mathword import answer_math_question

        self.numbers = len(re.findall(r"\d+", self.task.question))
        result = _safe(answer_math_question, self.task.question)
        if result is not None:
            if result.status == "ok" and result.answer is not None:
                self.accept_skill(result.answer, result.expression)
            elif result.status == "infeasible":
                # условие разобрано целиком, но числа для ответа в нём нет (или деление на ноль)
                self.mark_static_infeasible(0.97, result.detail)
            elif result.status == "error":
                self.mark_static_infeasible(0.8, result.detail or "arithmetic error in the problem")
        self.prepared = True

    def features(self) -> dict:
        return {"numbers": self.numbers}

    def cwd(self) -> Path:
        return self.task.workspace if self.task.has_workspace else self.scratch


# --------------------------------------------------------------------------- data_analysis
class DataSolver(ExprSolver):
    family = "data_analysis"
    system_prompt = (
        "`rows` is a list of dicts already loaded from the CSV file described below "
        "(numeric columns are numbers). Reply with ONLY one Python expression over `rows` that evaluates "
        "to the final answer. " + _ONLY_CODE + " "
        "If the data does not contain what is asked (for example, there is no such column), reply IMPOSSIBLE."
    )

    def prepare(self) -> None:
        from ..skills import tables as T

        self.tables = []
        self.main = None
        paths = _safe(T.find_tables, self.task.workspace, default=[]) if self.task.has_workspace else []
        mentioned = re.findall(r"[\w\-./]+\.(?:csv|tsv|jsonl|ndjson|json|xlsx|xls|txt|db|sqlite)\b", self.task.instruction)
        mentioned = [m for m in mentioned if Path(m).name != "answer.txt"]
        for path in paths[:8]:
            table = _safe(T.load_table, path)
            if table is not None and table.columns:
                self.tables.append(table)
        if not self.tables:
            what = f"file {mentioned[0]} is missing" if mentioned else "no data file in the workspace"
            # Данные могут лежать в формате, который навык не читает (xlsx, sqlite, свой текст): тогда это
            # не «данных нет», а работа для модели с bash. Уверенный отказ — только если такого файла нет.
            other = [p for p in self.task.workspace.rglob("*")
                     if p.is_file() and not p.name.startswith(".") and p.name != "answer.txt"
                     and (p.suffix.lower() in (".xlsx", ".xls", ".db", ".sqlite", ".sqlite3", ".parquet", ".dat", ".xml")
                          or (mentioned and p.name == Path(mentioned[0]).name and p.stat().st_size > 0))
                     ] if self.task.has_workspace else []
            self.mark_static_infeasible(0.6 if other else 0.97, what)
            if other:
                self.ladder = ("react",)
            self.prepared = True
            return
        by_name = {t.name: t for t in self.tables}
        self.main = next((by_name[Path(m).name] for m in mentioned if Path(m).name in by_name), self.tables[0])
        if mentioned and Path(mentioned[0]).name not in by_name:
            self.mark_static_infeasible(0.6, f"file {mentioned[0]} is missing (found {sorted(by_name)})")
        result = _safe(T.answer_table_question, self.main, self.task.question)
        if result is not None:
            if result.status == "ok" and result.answer is not None:
                self.accept_skill(result.answer, result.plan)
            elif result.status == "missing_column":
                # 0.97 — столбца нет ни в файле, ни в списке столбцов из самого условия, и слово не похоже
                # на другое название существующего столбца; иначе решает модель.
                # 0.9, а не 0.97: формулировки в финале будут другими, столбец могут назвать иначе.
                # Одна дешёвая попытка модели (~40 токенов) дешевле потерянной задачи; её отказ закрывает задачу.
                self.mark_static_infeasible(0.9 if result.confirmed else 0.7, result.detail)
        self.prepared = True

    def features(self) -> dict:
        q = self.task.question
        return {"filters": len(re.findall(r"'[^']+'|\"[^\"]+\"", q)) + len(re.findall(r"\bthrough\b|\bbetween\b", q))}

    def user_prompt(self) -> str:
        from ..skills.tables import table_summary

        if not self.main:
            return "(no data file was found in the task directory)\nQuestion: " + strip_preamble(self.task.question)
        parts = [table_summary(self.main, sample_rows=2)]
        others = [t for t in self.tables if t is not self.main]
        for table in others[:4]:
            parts.append(f"also available as tables[{table.name!r}]: " + table_summary(table, sample_rows=1))
        parts.append("Question: " + strip_preamble(self.task.question))
        return "\n".join(parts)

    def preload(self, code: str = "") -> str:
        if not self.main:
            return "rows = []\ntables = {}\n"
        lines = [import_harness_preamble(), "from harness.skills.tables import load_table as _load_table\n",
                 "tables = {}\n"]
        for table in self.tables:
            lines.append(f"tables[{table.name!r}] = _load_table(Path({str(table.path)!r})).rows\n")
        lines.append(f"rows = tables[{self.main.name!r}]\n")
        if re.search(r"\bdf\b|\bpd\.", code):
            lines.append(f"import pandas as pd\ndf = pd.read_csv({str(self.main.path)!r}, sep=None, engine='python')\n")
        lines.append("sys.path.pop(0)\n")
        return "".join(lines)


# --------------------------------------------------------------------------- search
class SearchSolver(ExprSolver):
    family = "search"
    system_prompt = (
        "`kb` is a dict loaded from the knowledge base: kb[title] is a dict of that entry's fields; "
        "a field value that equals another title links to that entry. Reply with ONLY one Python expression "
        "that evaluates to the final answer itself: a name, a number, or 'yes'/'no' - never True/False "
        "unless asked. You may filter, count, compare and take min/max over kb.items() (for example, "
        "the employees of a company are the entries whose Employer is that company). " + _ONLY_CODE + " "
        "Reply NOT_FOUND only if the entry or the field that the question needs is absent from kb."
    )
    direct_prompt = (
        "Answer the question using only the passages below. Reply with ONLY the answer (a name, number or "
        "short phrase), nothing else. If the passages do not contain the answer, reply NOT_FOUND."
    )

    def prepare(self) -> None:
        from ..skills import kb as K

        self.kb = None
        self.anchors: list[str] = []
        self.direct = False
        self.query = strip_preamble(self.task.question)
        kb_dir = K.find_kb_dir(self.task.workspace) if self.task.has_workspace else None
        if kb_dir is None:
            self.mark_static_infeasible(0.97, "no knowledge base directory in the workspace")
            self.prepared = True
            return
        self.kb_dir = kb_dir
        self.kb = _safe(K.load_kb, kb_dir)
        if self.kb is None or not self.kb.entries:
            self.kb = None
            self.mark_static_infeasible(0.97, "knowledge base is empty")
            self.prepared = True
            return
        self.query = _safe(K.canonicalize_question, self.kb, self.query, default=self.query)
        self.anchors = K.find_mentions(self.kb, self.query)
        self.direct = not self.kb.structured
        if not self.direct:
            result = _safe(K.answer_path_question, self.kb, self.query)
            if result is not None and result.status == "unparsed":
                # не цепочка — может быть сравнение («кто старше») или обратная ссылка («кого наставляет»)
                other = _safe(K.answer_relation_question, self.kb, self.query)
                if other is not None and other.status != "unparsed":
                    result = other
            if result is not None:
                if result.ok and result.answer:
                    self.accept_skill(result.answer, " > ".join([result.anchor or "?"] + result.chain))
                elif result.status in ("no_anchor", "broken"):
                    # Вопрос разобран полностью, но данных для ответа в базе нет: это и есть
                    # «невыполнимая задача» по определению организаторов. Токены не тратим.
                    self.mark_static_infeasible(0.96, result.detail)
        self.prepared = True

    def features(self) -> dict:
        if not self.kb:
            return {"hops": 1}
        fields = {name.lower() for name in self.kb.field_names()}
        words = re.findall(r"[A-Za-z]+", self.query.lower())
        return {"hops": max(1, sum(1 for w in words if w in fields))}

    def refusal_on_missing_data(self) -> bool:
        return True

    def _passages(self, budget_tokens: int = 1200) -> str:
        """Режим без полей: самые похожие на вопрос файлы целиком, пока не кончится лимит."""
        assert self.kb is not None
        stop = {"the", "what", "which", "who", "when", "where", "was", "were", "and", "for", "does", "did", "that",
                "with", "from", "this", "are", "has", "have", "how", "many", "much"}
        terms = {w for w in re.findall(r"[a-z0-9]+", self.query.lower()) if len(w) >= 3 and w not in stop}
        scored = []
        for title, text in self.kb.texts.items():
            low = text.lower()
            score = sum(1 for t in terms if t in low) + (5 if title in self.anchors else 0)
            scored.append((score, title, text))
        scored.sort(key=lambda item: (-item[0], item[1]))
        # Маленькую базу отдаём целиком: вопрос может идти по цепочке (человек -> компания -> город),
        # а файл компании слов вопроса не содержит.
        if approx_tokens("".join(self.kb.texts.values())) <= budget_tokens:
            return "\n---\n".join(text.strip() for _s, _t, text in scored)
        # Большая база: сначала записи из вопроса и те, на которые они ссылаются (два перехода), затем похожие.
        from ..skills.kb import find_mentions

        chain: list[str] = [t for t in self.anchors if t in self.kb.texts]
        frontier = list(chain)
        for _hop in range(2):
            nxt = []
            for title in frontier:
                for linked in find_mentions(self.kb, self.kb.texts[title]):
                    if linked not in chain and linked in self.kb.texts:
                        chain.append(linked)
                        nxt.append(linked)
            frontier = nxt[:12]
        chunks, used, taken = [], 0, set()
        for title in chain[:40]:
            text = self.kb.texts[title]
            cost = approx_tokens(text)
            if used + cost > budget_tokens and chunks:
                break
            chunks.append(text.strip())
            taken.add(title)
            used += cost
        for score, title, text in scored:
            if title in taken:
                continue
            if score <= 0 and chunks:
                break
            cost = approx_tokens(text)
            if used + cost > budget_tokens and chunks:
                break
            chunks.append(text.strip())
            used += cost
        return "\n---\n".join(chunks)

    def user_prompt(self) -> str:
        from ..skills.kb import schema_summary

        if self.kb is None:
            return "(the knowledge base directory is missing or empty)\nQuestion: " + self.query
        if self.direct:
            return self._passages() + "\n\nQuestion: " + self.query
        return schema_summary(self.kb, self.anchors[:3]) + "\nQuestion: " + self.query

    def prompt_text(self, step: str) -> str:
        return (self.direct_prompt if self.direct else self.system_prompt) + "\n" + self.user_prompt()

    async def run_step(self, step: str):
        # в режиме без полей меняется только системная инструкция; разбор ответа общий
        if self.direct:
            self.system_prompt = self.direct_prompt
        return await super().run_step(step)

    def preload(self, code: str = "") -> str:
        from ..skills.kb import kb_preload_code

        if self.kb is None:
            return "kb = {}\ntexts = {}\n"
        return kb_preload_code(self.kb_dir)


# --------------------------------------------------------------------------- неизвестный тип
class GenericSolver(ExprSolver):
    """Задача незнакомого типа: один скрипт по описанию каталога, затем ReAct-агент."""

    family = "generic"
    ladder = ("fast", "strong", "react")
    max_tokens_fast = 700
    system_prompt = (
        "You solve a task by writing Python. The code will run in the task directory (its files are listed "
        "below). Reply with ONLY Python code that prints the final answer on the last line. " + _ONLY_CODE + " "
        "If the task cannot be done, reply IMPOSSIBLE."
    )

    def prepare(self) -> None:
        self.listing = self._describe_workspace()
        if self.task.sentinel:
            # нужно что-то сделать в каталоге и отчитаться словом — это работа для агента с bash
            self.ladder = ("react",)
        self.prepared = True

    def _describe_workspace(self, max_files: int = 25, head_chars: int = 400) -> str:
        ws = self.task.workspace
        if not ws.is_dir():
            return "(the task directory has no files)"
        lines = []
        files = sorted(p for p in ws.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts))
        for path in files[:max_files]:
            rel = path.relative_to(ws)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            lines.append(f"- {rel} ({size} bytes)")
            if size <= 20000 and path.suffix.lower() in (".txt", ".md", ".csv", ".tsv", ".json", ".py", ".yaml",
                                                         ".yml", ".ini", ".cfg", ".log", ""):
                try:
                    head = path.read_text(encoding="utf-8", errors="replace")[:head_chars]
                except OSError:
                    continue
                lines.append("  " + head.strip().replace("\n", "\n  "))
        if len(files) > max_files:
            lines.append(f"... and {len(files) - max_files} more files")
        return "\n".join(lines) if lines else "(the task directory is empty)"

    def user_prompt(self) -> str:
        return f"Files:\n{self.listing}\n\nTask: {self.task.question}"


def solver_class(task_type: str):
    from .coding import CodingSolver

    return {
        "math": MathSolver,
        "data_analysis": DataSolver,
        "search": SearchSolver,
        "coding": CodingSolver,
    }.get(task_type, GenericSolver)
