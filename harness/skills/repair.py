"""Ремонт программы перебором одноточечных мутаций (generate-and-validate).

Идея: баги в задачах — обычно замена одного оператора/имени/константы. Берём функции,
которые вызывают упавшие тесты, порождаем мутанты по приоритетам (P0 — обратные
операторы, P1 — границы и off-by-one, P2 — более далёкие замены), каждый мутант
проверяем запуском тестов в отдельном процессе. Жадно принимаем мутант, который строго
уменьшает множество упавших тестов и ничего не ломает, и повторяем, пока тесты не пройдут.

Правки текстовые, по позициям узлов AST: остальной файл и форматирование не меняются.
Мутанты проверяются в копиях workspace (параллельно); сам workspace меняется только
в конце, если всё починено. Иначе исходные файлы остаются байт-в-байт. Тесты не правятся.
"""
from __future__ import annotations

import ast
import os
import queue
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from harness.skills.testrun import (
    TestReport, find_test_command, run_tests, snapshot, source_files, test_files,
)

__all__ = ["RepairResult", "mutation_repair"]

_MAX_WORKERS = 4
_MAX_COPY_BYTES = 20_000_000
_FINAL_TIMEOUT = 15.0


@dataclass
class RepairResult:
    fixed: bool                    # все тесты проходят, файлы на диске исправлены
    patched_files: dict[str, str]  # относительный путь -> новый текст
    edits: list[str]               # описания правок: "mylib.py:8 '-=' -> '+='"
    ambiguous: bool                # для какого-то теста нашлось несколько исправлений одного приоритета
    tried: int                     # сколько мутантов проверено
    seconds: float
    report: TestReport             # итоговый отчёт тестов


# ---------------------------------------------------------------- исходник и позиции

def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="surrogateescape")


def _encode(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogateescape")


class _Src:
    """Текст файла + перевод позиций AST (строка, байтовая колонка) в индексы символов."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = [m.group(0) for m in re.finditer(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$", text)]
        self.starts = []
        pos = 0
        for line in self.lines:
            self.starts.append(pos)
            pos += len(line)

    def off(self, lineno: int, col: int) -> int:
        if lineno - 1 >= len(self.lines):
            return len(self.text)
        line = self.lines[lineno - 1]
        chars = len(_encode(line)[:col].decode("utf-8", errors="ignore"))
        return self.starts[lineno - 1] + chars

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (self.off(node.lineno, node.col_offset), self.off(node.end_lineno, node.end_col_offset))

    def seg(self, node: ast.AST) -> str:
        s, e = self.span(node)
        return self.text[s:e]


@dataclass
class _Mut:
    file: str
    start: int
    end: int
    old: str
    new: str
    priority: int
    rank: int = 0
    order: int = 0

    def apply(self, text: str) -> str:
        return text[: self.start] + self.new + text[self.end:]

    def describe(self, text: str) -> str:
        line = text.count("\n", 0, self.start) + 1

        def short(s: str) -> str:
            s = " ".join(s.split())
            return s if len(s) <= 40 else s[:37] + "..."
        return f"{self.file}:{line} {short(self.old)!r} -> {short(self.new)!r}"


# ---------------------------------------------------------------- таблицы мутаций

_BIN_TXT = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
            ast.Pow: "**", ast.MatMult: "@", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
            ast.LShift: "<<", ast.RShift: ">>"}
_BIN_SWAPS = {
    ast.Add: [(ast.Sub, 0), (ast.Mult, 2)],
    ast.Sub: [(ast.Add, 0), (ast.Div, 2)],
    ast.Mult: [(ast.Div, 0), (ast.FloorDiv, 0), (ast.Add, 2)],
    ast.Div: [(ast.Mult, 0), (ast.FloorDiv, 0), (ast.Sub, 2)],
    ast.FloorDiv: [(ast.Mult, 0), (ast.Div, 0), (ast.Mod, 1)],
    ast.Mod: [(ast.FloorDiv, 1)],
}
_AUG_SWAPS = {
    ast.Add: [(ast.Sub, 0), (ast.Mult, 2)],
    ast.Sub: [(ast.Add, 0)],
    ast.Mult: [(ast.Div, 0), (ast.FloorDiv, 0), (ast.Add, 2)],
    ast.Div: [(ast.Mult, 0), (ast.FloorDiv, 0)],
    ast.FloorDiv: [(ast.Mult, 0), (ast.Div, 0)],
    ast.Mod: [(ast.FloorDiv, 1)],
}
_CMP_PAT = {ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
            ast.In: r"\bin\b", ast.NotIn: r"\bnot\s+in\b", ast.Is: r"\bis\b", ast.IsNot: r"\bis\s+not\b"}
_CMP_SWAPS = {
    ast.Lt: [(">", 0), ("<=", 1)], ast.Gt: [("<", 0), (">=", 1)],
    ast.LtE: [(">=", 0), ("<", 1)], ast.GtE: [("<=", 0), (">", 1)],
    ast.Eq: [("!=", 0)], ast.NotEq: [("==", 0)],
    ast.In: [("not in", 0)], ast.NotIn: [("in", 0)],
    ast.Is: [("is not", 0)], ast.IsNot: [("is", 0)],
}
_CALL_SWAPS = {
    "sorted": [("reversed", 0)], "reversed": [("sorted", 0)],
    "min": [("max", 0), ("sum", 2)], "max": [("min", 0), ("sum", 2)],
    "any": [("all", 0)], "all": [("any", 0)],
    "sum": [("len", 2), ("max", 2), ("min", 2)], "len": [("sum", 2)],
}
_METHOD_SWAPS = {
    "upper": [("lower", 0)], "lower": [("upper", 0)],
    "startswith": [("endswith", 0)], "endswith": [("startswith", 0)],
    "lstrip": [("rstrip", 0), ("strip", 1)], "rstrip": [("lstrip", 0), ("strip", 1)],
    "strip": [("lstrip", 1), ("rstrip", 1)],
    "append": [("extend", 0)], "extend": [("append", 0)],
    "find": [("rfind", 0)], "rfind": [("find", 0)],
    "index": [("rindex", 0)], "rindex": [("index", 0)],
    "ljust": [("rjust", 0)], "rjust": [("ljust", 0)],
    "isupper": [("islower", 0)], "islower": [("isupper", 0)],
    "keys": [("values", 0)], "values": [("keys", 0)],
    "split": [("rsplit", 0)], "rsplit": [("split", 0)],
    "removeprefix": [("removesuffix", 0)], "removesuffix": [("removeprefix", 0)],
    "title": [("capitalize", 1)], "capitalize": [("title", 1)],
    "isdigit": [("isalpha", 1)], "isalpha": [("isdigit", 1)],
}
_SIMPLE = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.Constant)


def _sym_regex(sym: str) -> str:
    """Регулярка оператора-символа, не задевающая соседние символы операторов."""
    return r"(?<![<>=!*/+\-%&|^@])" + re.escape(sym) + r"(?![<>=*/])"


def _int_const(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
            and type(node.operand.value) is int:
        return -node.operand.value
    return None


class _Gen:
    """Порождает мутанты в заданных узлах одного файла."""

    def __init__(self, rel: str, src: _Src) -> None:
        self.rel = rel
        self.src = src
        self.out: list[_Mut] = []
        self.rank = 0
        self.parents: dict[int, ast.AST] = {}

    def emit(self, start: int, end: int, new: str, prio: int) -> None:
        old = self.src.text[start:end]
        if old != new:
            self.out.append(_Mut(self.rel, start, end, old, new, prio, self.rank))

    def find_between(self, a: int, b: int, pattern: str) -> list[tuple[int, int]]:
        seg = self.src.text[a:b]
        seg = re.sub(r"#[^\n]*", lambda m: " " * len(m.group(0)), seg)  # комментарии не трогаем
        return [(a + m.start(), a + m.end()) for m in re.finditer(pattern, seg)]

    def plus(self, node: ast.AST, delta: int) -> str:
        c = _int_const(node)
        if c is not None:
            return str(c + delta)
        t = self.src.seg(node)
        if not isinstance(node, _SIMPLE):
            t = f"({t})"
        return f"{t} + {delta}" if delta > 0 else f"{t} - {-delta}"

    def off_by_one(self, node: ast.AST) -> list[str]:
        """Варианты выражения на единицу больше/меньше; «n + 1» / «n - 1» просто снимают единицу."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)) and _int_const(node.right) == 1:
            return [self.src.seg(node.left)]
        return [self.plus(node, 1), self.plus(node, -1)]

    # -- обход
    def visit_function(self, fn: ast.AST, rank: int) -> None:
        self.rank = rank
        body = fn.body if hasattr(fn, "body") and isinstance(fn.body, list) else []
        self.parents = {id(c): p for stmt in body for p in ast.walk(stmt) for c in ast.iter_child_nodes(p)}
        for stmt in body:
            for node in ast.walk(stmt):
                self.visit(node)

    def visit(self, node: ast.AST) -> None:
        try:
            if isinstance(node, ast.BinOp):
                self._binop(node)
            elif isinstance(node, ast.AugAssign):
                self._augassign(node)
            elif isinstance(node, ast.Compare):
                self._compare(node)
            elif isinstance(node, ast.BoolOp):
                self._boolop(node)
            elif isinstance(node, ast.Constant):
                self._constant(node)
            elif isinstance(node, ast.Call):
                self._call(node)
            elif isinstance(node, ast.UnaryOp):
                self._unary(node)
            elif isinstance(node, ast.Subscript):
                self._subscript(node)
            elif isinstance(node, ast.Return) and node.value is not None:
                self._return(node)
            if isinstance(node, (ast.If, ast.While, ast.IfExp)):
                self._negate(node.test)
            elif isinstance(node, ast.comprehension):
                for cond in node.ifs:
                    self._negate(cond)
        except (AttributeError, IndexError, ValueError):
            return  # узел без позиций — пропускаем

    def _binop(self, node: ast.BinOp) -> None:
        swaps = _BIN_SWAPS.get(type(node.op))
        if not swaps:
            return
        a = self.src.span(node.left)[1]
        b = self.src.span(node.right)[0]
        hits = self.find_between(a, b, _sym_regex(_BIN_TXT[type(node.op)]))
        if len(hits) != 1:
            return
        s, e = hits[0]
        for op, prio in swaps:
            self.emit(s, e, _BIN_TXT[op], prio)

    def _augassign(self, node: ast.AugAssign) -> None:
        swaps = _AUG_SWAPS.get(type(node.op))
        if not swaps:
            return
        a = self.src.span(node.target)[1]
        b = self.src.span(node.value)[0]
        hits = self.find_between(a, b, re.escape(_BIN_TXT[type(node.op)] + "="))
        if len(hits) != 1:
            return
        s, e = hits[0]
        for op, prio in swaps:
            self.emit(s, e, _BIN_TXT[op] + "=", prio)

    def _compare(self, node: ast.Compare) -> None:
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            a = self.src.span(left)[1]
            b = self.src.span(right)[0]
            pat = _CMP_PAT[type(op)]
            hits = self.find_between(a, b, pat if pat[0] == "\\" else _sym_regex(pat))
            parity = (len(node.ops) == 1 and isinstance(op, (ast.Eq, ast.NotEq))
                      and isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mod)
                      and _int_const(left.right) == 2 and _int_const(right) in (0, 1))
            if parity:
                # x % 2 == 1 -> x % 2 == 0: равносильно замене == на !=, берём замену константы
                s, e = self.src.span(right)
                self.emit(s, e, "0" if _int_const(right) == 1 else "1", 0)
            elif len(hits) == 1:
                s, e = hits[0]
                for new, prio in _CMP_SWAPS[type(op)]:
                    self.emit(s, e, new, prio)
            left = right

    def _boolop(self, node: ast.BoolOp) -> None:
        word = "and" if isinstance(node.op, ast.And) else "or"
        other = "or" if word == "and" else "and"
        hits = []
        for x, y in zip(node.values, node.values[1:]):
            h = self.find_between(self.src.span(x)[1], self.src.span(y)[0], rf"\b{word}\b")
            if len(h) != 1:
                return
            hits.append(h[0])
        s, e = hits[0][0], hits[-1][1]
        text = self.src.text[s:e]
        new = text
        for hs, he in reversed(hits):
            new = new[: hs - s] + other + new[he - s:]
        self.emit(s, e, new, 0)

    def _constant(self, node: ast.Constant) -> None:
        s, e = self.src.span(node)
        v = node.value
        if isinstance(v, bool):
            self.emit(s, e, "False" if v else "True", 0)
        elif type(v) is int:
            parent = self.parents.get(id(node))
            if v == 1 and isinstance(parent, ast.BinOp) and parent.right is node \
                    and isinstance(parent.op, (ast.Add, ast.Sub)):
                # «n + 1» -> «n», а не равносильное «n + 0»
                ps, pe = self.src.span(parent)
                self.emit(ps, pe, self.src.seg(parent.left), 1)
            elif v in (0, 1):
                self.emit(s, e, "1" if v == 0 else "0", 1)
            self.emit(s, e, str(v + 1), 2)
            self.emit(s, e, str(v - 1), 2)

    def _call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in _CALL_SWAPS:
            s, e = self.src.span(f)
            for new, prio in _CALL_SWAPS[f.id]:
                self.emit(s, e, new, prio)
        elif isinstance(f, ast.Attribute) and f.attr in _METHOD_SWAPS:
            e = self.src.span(f)[1]
            s = e - len(f.attr)
            if self.src.text[s:e] == f.attr:
                for new, prio in _METHOD_SWAPS[f.attr]:
                    self.emit(s, e, new, prio)
        if isinstance(f, ast.Name) and f.id == "range" and not node.keywords:
            args = node.args
            if len(args) == 1:
                s, e = self.src.span(args[0])
                for new in self.off_by_one(args[0]):
                    self.emit(s, e, new, 1)
                self.emit(s, e, "1, " + self.src.seg(args[0]), 1)
            elif len(args) == 2:
                for arg in args:
                    s, e = self.src.span(arg)
                    for new in self.off_by_one(arg):
                        self.emit(s, e, new, 1)
        if isinstance(f, ast.Attribute) and f.attr == "join" and len(node.args) == 1:
            self._add_reverse(node.args[0])

    def _add_reverse(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Name, ast.Call, ast.Attribute, ast.Subscript)):
            s, e = self.src.span(node)
            self.emit(s, e, self.src.text[s:e] + "[::-1]", 1)

    def _unary(self, node: ast.UnaryOp) -> None:
        s = self.src.span(node)[0]
        o = self.src.span(node.operand)[0]
        prefix = self.src.text[s:o]
        if isinstance(node.op, ast.Not):
            self.emit(s, o, re.sub(r"^not\s*", "", prefix), 1)
        elif isinstance(node.op, ast.USub) and not isinstance(node.operand, ast.Constant):
            self.emit(s, o, re.sub(r"^-\s*", "", prefix), 2)

    def _negate(self, test: ast.AST) -> None:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return  # снятие not уже порождается в _unary
        s, e = self.src.span(test)
        t = self.src.text[s:e]
        self.emit(s, e, f"not {t}" if isinstance(test, _SIMPLE) else f"not ({t})", 1)

    def _subscript(self, node: ast.Subscript) -> None:
        sl = node.slice
        if isinstance(sl, ast.Slice):
            lo, up, st = sl.lower, sl.upper, sl.step
            if st is None:
                ss, se = self.src.span(sl)
                if lo is None and up is not None:
                    self.emit(ss, se, self.src.seg(up) + ":", 1)
                elif up is None and lo is not None:
                    self.emit(ss, se, ":" + self.src.seg(lo), 1)
                for bound in (lo, up):
                    if bound is None:
                        continue
                    s, e = self.src.span(bound)
                    for new in self.off_by_one(bound):
                        self.emit(s, e, new, 1)
            elif lo is None and up is None and _int_const(st) == -1:
                s, e = self.src.span(node)
                self.emit(s, e, self.src.seg(node.value), 1)
        else:
            c = _int_const(sl)
            if c in (0, -1):
                s, e = self.src.span(sl)
                self.emit(s, e, "-1" if c == 0 else "0", 2)

    def _return(self, node: ast.Return) -> None:
        v = node.value
        self._add_reverse(v)
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub):
            return
        if isinstance(v, ast.Constant) and not (type(v.value) in (int, float)):
            return
        if isinstance(v, (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.Constant)):
            s, e = self.src.span(v)
            self.emit(s, e, "-" + self.src.text[s:e], 2)
        elif isinstance(v, ast.BinOp):
            s, e = self.src.span(v)
            self.emit(s, e, f"-({self.src.text[s:e]})", 2)


# ---------------------------------------------------------------- подозрительные функции

_FRAME_FUNC_RE = re.compile(r'File "([^"]+)", line \d+, in (\w+)')


def _parse(text: str) -> Optional[ast.Module]:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _suspects(workspace: Path, texts: dict[str, str], report: TestReport) -> list[tuple[str, ast.AST]]:
    """(файл, узел функции) по убыванию подозрительности."""
    funcs: dict[str, list[tuple[str, ast.AST]]] = {}
    classes: dict[str, list[tuple[str, ast.AST]]] = {}
    trees: dict[str, ast.Module] = {}
    for rel, text in texts.items():
        tree = _parse(text)
        if tree is None:
            continue
        trees[rel] = tree
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(node.name, []).append((rel, node))
            elif isinstance(node, ast.ClassDef):
                classes[node.name] = [(rel, b) for b in node.body
                                      if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]

    names: list[str] = []
    failed = set(report.failed)
    for tf in test_files(workspace):
        tree = _parse(tf.read_text(encoding="utf-8", errors="replace"))
        if tree is None:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.Module))]:
            members = [b for b in cls.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
            hit = [b for b in members if b.name in failed]
            if not hit:
                continue
            setup = [b for b in members if b.name in ("setUp", "setUpClass")]
            for fn in hit + setup:
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name):
                        names.append(n.id)
                    elif isinstance(n, ast.Attribute):
                        names.append(n.attr)
    src_names = {Path(r).name for r in texts}
    for m in _FRAME_FUNC_RE.finditer(report.raw):
        if Path(m.group(1)).name in src_names:
            names.append(m.group(2))

    picked: list[tuple[str, ast.AST]] = []
    seen: set[int] = set()

    def add(items: list[tuple[str, ast.AST]]) -> None:
        for rel, node in items:
            if id(node) not in seen:
                seen.add(id(node))
                picked.append((rel, node))

    for name in names:
        add(funcs.get(name, []))
        add(classes.get(name, []))
    # вызываемые из подозрительных функций (помощники)
    i = 0
    while i < len(picked):
        for n in ast.walk(picked[i][1]):
            ref = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
            if ref:
                add(funcs.get(ref, []))
        i += 1
    if not picked:
        for name in sorted(funcs):
            add(funcs[name])
    if not picked:
        add([(rel, tree) for rel, tree in sorted(trees.items())])
    return picked


def _generate(texts: dict[str, str], suspects: list[tuple[str, ast.AST]]) -> list[_Mut]:
    gens: dict[str, _Gen] = {}
    for rank, (rel, node) in enumerate(suspects):
        gen = gens.setdefault(rel, _Gen(rel, _Src(texts[rel])))
        gen.visit_function(node, rank)
    muts = [m for g in gens.values() for m in g.out]
    muts.sort(key=lambda m: (m.priority, m.rank, m.file, m.start, m.end, m.new))
    out, seen = [], set()
    for m in muts:
        new_text = m.apply(texts[m.file])
        key = (m.file, new_text)
        if key in seen:
            continue
        seen.add(key)
        try:
            compile(new_text, m.file, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            continue
        m.order = len(out)
        out.append(m)
    return out


# ---------------------------------------------------------------- проверка мутантов

class _Runner:
    """Проверяет мутанты в копиях workspace (или в самом workspace, если копировать нельзя)."""

    def __init__(self, workspace: Path, argv: list[str], test_timeout: float, deadline: float,
                 texts: dict[str, str]) -> None:
        self.workspace = workspace
        self.argv = argv
        self.test_timeout = test_timeout
        self.deadline = deadline
        self.texts = dict(texts)
        self.tmp: Optional[str] = None
        self.dirs: list[Path] = []
        size = sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file() and not p.is_symlink())
        if size <= _MAX_COPY_BYTES:
            try:
                self.tmp = tempfile.mkdtemp(prefix="repair_")
                n = max(1, min(_MAX_WORKERS, os.cpu_count() or 1))
                for k in range(n):
                    d = Path(self.tmp) / f"w{k}"
                    shutil.copytree(workspace, d, symlinks=True,
                                    ignore=shutil.ignore_patterns("__pycache__"))
                    self.dirs.append(d)
            except OSError:
                self.close()
                self.dirs = []
        if not self.dirs:
            self.dirs = [workspace]  # запасной режим: последовательно, с откатом после каждого мутанта
        self.pool: "queue.Queue[Path]" = queue.Queue()
        for d in self.dirs:
            self.pool.put(d)

    def _write(self, d: Path, rel: str, text: str) -> None:
        (d / rel).write_bytes(_encode(text))

    def sync(self, texts: dict[str, str]) -> None:
        self.texts = dict(texts)
        for d in self.dirs:
            if d != self.workspace:
                for rel, text in texts.items():
                    self._write(d, rel, text)

    def _one(self, mut: _Mut) -> Optional[TestReport]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0.2:
            return None
        d = self.pool.get()
        try:
            self._write(d, mut.file, mut.apply(self.texts[mut.file]))
            return run_tests(d, self.argv, timeout=min(self.test_timeout, remaining))
        finally:
            self._write(d, mut.file, self.texts[mut.file])
            self.pool.put(d)

    def evaluate(self, muts: list[_Mut]) -> list[Optional[TestReport]]:
        if not muts:
            return []
        if len(self.dirs) == 1:
            return [self._one(m) for m in muts]
        with ThreadPoolExecutor(max_workers=len(self.dirs)) as ex:
            return list(ex.map(self._one, muts))

    def close(self) -> None:
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None


# ---------------------------------------------------------------- основной цикл

def _restore(workspace: Path, original: dict[str, bytes]) -> None:
    for rel, data in original.items():
        p = workspace / rel
        try:
            if p.read_bytes() == data:
                continue
        except OSError:
            pass
        p.write_bytes(data)


def mutation_repair(workspace: Path, argv: Optional[list[str]] = None,
                    time_budget: float = 40.0, test_timeout: float = 5.0) -> RepairResult:
    """Чинит исходники перебором одноточечных мутаций; при неудаче оставляет файлы как были."""
    started = time.monotonic()
    deadline = started + time_budget
    workspace = Path(workspace)
    if argv is None:
        argv = find_test_command("", workspace)
    tests_before = snapshot(test_files(workspace))
    original = {str(p.relative_to(workspace)): p.read_bytes() for p in source_files(workspace)}
    texts = {rel: _decode(data) for rel, data in original.items()}

    def result(fixed: bool, report: TestReport, patched: dict, edits: list, ambiguous: bool,
               tried: int) -> RepairResult:
        if snapshot(test_files(workspace)) != tests_before:
            # тесты мы не пишем никогда; если их изменил сам прогон — результату не доверяем
            _restore(workspace, original)
            fixed = False
        return RepairResult(fixed, patched, edits, ambiguous, tried, time.monotonic() - started, report)

    base = run_tests(workspace, argv, timeout=test_timeout)
    if base.ok:
        return result(True, base, {}, [], False, 0)
    if base.error or not texts:
        return result(False, base, {}, [], False, 0)

    need_ok = base.timed_out or not base.failed   # нет имён упавших — принимаем только полный успех
    total0 = 0 if need_ok else base.total
    cur_fail = set(base.failed)
    cur_report = base
    edits: list[str] = []
    ambiguous = False
    tried = 0
    carry: list[_Mut] = []
    fully = False

    def valid(rep: Optional[TestReport]) -> bool:
        if rep is None or rep.timed_out or rep.error:
            return False
        if total0 and rep.total != total0:
            return False
        if rep.ok:
            return True
        if need_ok or not rep.failed:
            return False
        return set(rep.failed) < cur_fail

    runner = _Runner(workspace, argv, test_timeout, deadline, texts)
    try:
        while time.monotonic() < deadline:
            chosen: Optional[tuple[_Mut, TestReport]] = None
            # 1) годные мутанты прошлого раунда (второй независимый баг) — без полного перебора
            for m in carry:
                if texts[m.file][m.start:m.end] != m.old:
                    continue
                try:
                    compile(m.apply(texts[m.file]), m.file, "exec", dont_inherit=True)
                except (SyntaxError, ValueError):
                    continue
                rep = runner.evaluate([m])[0]
                tried += rep is not None
                if valid(rep):
                    chosen = (m, rep)
                    break
            carry = []
            # 2) полный перебор по приоритетам
            if chosen is None:
                muts = _generate(texts, _suspects(workspace, texts, cur_report))
                for prio in (0, 1, 2):
                    cands = [m for m in muts if m.priority == prio]
                    reps = runner.evaluate(cands)
                    tried += sum(r is not None for r in reps)
                    good = [(m, r) for m, r in zip(cands, reps) if valid(r)]
                    if good:
                        good.sort(key=lambda mr: (-len(cur_fail - set(mr[1].failed)), mr[0].order))
                        for t in cur_fail:
                            fixers = {m.apply(texts[m.file]) for m, r in good if t not in r.failed}
                            if len(fixers) > 1:
                                ambiguous = True
                        chosen = good[0]
                        carry = [m for m, _ in good[1:]]
                        break
                    if time.monotonic() >= deadline:
                        break
            if chosen is None:
                break
            m, rep = chosen
            edits.append(m.describe(texts[m.file]))
            texts[m.file] = m.apply(texts[m.file])
            delta = len(m.new) - (m.end - m.start)
            shifted = []
            for c in carry:
                if c.file != m.file or c.end <= m.start:
                    shifted.append(c)
                elif c.start >= m.end:
                    c.start += delta
                    c.end += delta
                    shifted.append(c)
            carry = shifted
            runner.sync(texts)
            cur_report, cur_fail, need_ok = rep, set(rep.failed), False
            if rep.ok:
                fully = True
                break
    finally:
        runner.close()
        _restore(workspace, original)  # на случай запасного режима или исключения

    patched = {rel: t for rel, t in texts.items() if _encode(t) != original[rel]}
    if fully:
        try:
            for rel, t in patched.items():
                (workspace / rel).write_bytes(_encode(t))
            final = run_tests(workspace, argv, timeout=max(test_timeout, _FINAL_TIMEOUT))
        except BaseException:
            _restore(workspace, original)
            raise
        if final.ok:
            return result(True, final, patched, edits, ambiguous, tried)
        _restore(workspace, original)
        return result(False, base, patched, edits, ambiguous, tried)
    return result(False, base, patched, edits, ambiguous, tried)
