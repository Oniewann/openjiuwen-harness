"""Запуск тестов задачи и разбор их вывода в компактный отчёт.

Навык детерминированный: запускает тесты через sandbox (без секретов, с таймаутом,
без __pycache__), разбирает вывод unittest (формы с точками и -v, Python 3.11–3.13)
и минимально — pytest. Отчёт нужен для промпта модели (summarize_failures),
для ремонта мутациями (repair.py) и для проверки, что тесты не изменены (snapshot).

Отдельно — contradictory_asserts: статический поиск взаимоисключающих утверждений
в тестах (признак невыполнимой задачи).
"""
from __future__ import annotations

import ast
import functools
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from harness import sandbox

__all__ = [
    "TestReport", "find_test_command", "run_tests", "summarize_failures",
    "test_files", "source_files", "snapshot", "contradictory_asserts", "unsatisfiable_tests",
    "fallback_test_command",
]

_DETAIL_LIMIT = 300
_RAW_HALF = 1500
_SKIP_DIRS = {"__pycache__", "node_modules", "venv", "site-packages"}


@dataclass
class TestReport:
    __test__ = False  # чтобы pytest/unittest не принимали класс за тест

    ok: bool                 # все тесты прошли
    total: int               # сколько тестов запущено (0, если не удалось понять)
    failed: list[str]        # короткие имена упавших тестов (FAIL и ERROR)
    details: dict[str, str]  # имя теста -> компактное описание (<= 300 символов)
    timed_out: bool = False
    error: str = ""          # непусто, если тесты вообще не запустились
    raw: str = ""            # обрезанный вывод (начало + конец)
    seconds: float = 0.0


# ---------------------------------------------------------------- файлы

def _is_test_name(name: str) -> bool:
    return name.endswith(".py") and (name.startswith("test_") or name.endswith(("_test.py", "_tests.py"))
                                     or name in ("test.py", "tests.py"))


_TESTCASE_RE = re.compile(r"^class\s+\w+\(\s*(?:unittest\.)?TestCase\s*\)\s*:", re.M)


def _has_testcase(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:200000]
    except OSError:
        return False
    return bool(_TESTCASE_RE.search(head))


def _py_files(workspace: Path) -> list[Path]:
    workspace = Path(workspace)
    out = []
    for p in sorted(workspace.rglob("*.py")):
        rel = p.relative_to(workspace).parts
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel):
            continue
        if p.is_file():
            out.append(p)
    return out


def test_files(workspace: Path) -> list[Path]:
    """test_*.py, *_test.py, test.py в workspace (рекурсивно, без скрытых и __pycache__).

    Таких нет — файлы с классом-наследником unittest.TestCase: файл проверки с необычным именем
    (check_mylib.py, run_tests.py) иначе считался бы исходником, задача объявлялась невыполнимой
    («нет тестов»), а ремонт перебором «чинил» бы сам тест. При тестах по имени содержимое не смотрим:
    исходник со своим самотестом (class _Smoke(unittest.TestCase)) остаётся исходником.
    """
    files = _py_files(workspace)
    named = [p for p in files if _is_test_name(p.name)]
    return named or [p for p in files if _has_testcase(p)]


def source_files(workspace: Path) -> list[Path]:
    """Остальные .py-файлы workspace — то, что можно править."""
    tests = set(test_files(workspace))
    return [p for p in _py_files(workspace) if p not in tests]


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    """Путь (строкой) -> sha256 содержимого; отсутствующий файл -> ""."""
    out = {}
    for p in paths:
        try:
            out[str(p)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except OSError:
            out[str(p)] = ""
    return out


# ---------------------------------------------------------------- команда

_BACKTICK_RE = re.compile(r"`([^`\n]+)`")


def find_test_command(instruction: str, workspace: Path) -> list[str]:
    """argv для запуска тестов: команда из инструкции, единственный тест-файл или discover."""
    for m in _BACKTICK_RE.finditer(instruction or ""):
        cmd = m.group(1).strip()
        head = cmd.split()[0] if cmd.split() else ""
        if head in ("python", "python3", "pytest") or head.startswith("python3."):
            argv = sandbox.test_command_argv(cmd)
            if argv:
                return argv
    tests = test_files(workspace)
    if len(tests) == 1:
        return sandbox.python_argv(str(tests[0].relative_to(workspace)))
    return sandbox.python_argv("-m", "unittest", "discover", "-v")


@functools.lru_cache(maxsize=1)
def _pytest_installed() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("pytest") is not None
    except (ImportError, ValueError):
        return False


def _wants_missing_pytest(argv: list[str]) -> bool:
    if "pytest" not in argv[1:4] or argv[argv.index("pytest") - 1] != "-m":
        return False
    return not _pytest_installed()


def _without_pytest(argv: list[str], workspace: Path) -> Optional[list[str]]:
    """Замена `python -m pytest [файлы|каталоги]` без pytest: unittest для TestCase, plainrun для функций."""
    workspace = Path(workspace)
    args = argv[argv.index("pytest") + 1:]
    named = [a.split("::")[0] for a in args if not a.startswith("-")]
    tests: list[Path] = []
    for item in named:
        target = workspace / item
        if target.is_file():
            tests.append(target)
        elif target.is_dir():
            tests.extend(p for p in test_files(workspace) if target in p.parents)
        else:
            return None                      # аргумент непонятен — оставляем команду как есть
    if not named:
        tests = test_files(workspace)
    if not tests:
        return None
    texts = []
    for path in tests:
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return None
    rels = [str(p.relative_to(workspace)) for p in tests]
    if all(_TESTCASE_RE.search(t) for t in texts):
        return sandbox.python_argv("-m", "unittest", "-v", *rels)
    if not any(_TESTCASE_RE.search(t) for t in texts) and \
            all(re.search(r"^(?:async\s+)?def test\w*\(", t, re.MULTILINE) for t in texts):
        return sandbox.python_argv(str(Path(__file__).with_name("plainrun.py")), *rels)
    return None


def fallback_test_command(workspace: Path) -> Optional[list[str]]:
    """Команда на случай, когда указанная в условии не выполнила ни одного теста.

    Так бывает, если тестовый файл сам себя не запускает: нет `unittest.main()` или тесты — голые
    функции в стиле pytest. Успешный код возврата при нуле выполненных тестов ничего не доказывает.
    """
    workspace = Path(workspace)
    tests = test_files(workspace)
    if not tests:
        return None
    texts = []
    for path in tests:
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            texts.append("")
    if any(re.search(r"\bTestCase\b", t) for t in texts):
        top = [p for p in tests if p.parent == workspace]
        if len(top) == len(tests):
            return sandbox.python_argv("-m", "unittest", "-v", *[p.stem for p in top])
        return sandbox.python_argv("-m", "unittest", "discover", "-v")
    if any(re.search(r"^(?:async\s+)?def test\w*\(", t, re.MULTILINE) for t in texts):
        runner = Path(__file__).with_name("plainrun.py")
        return sandbox.python_argv(str(runner), *[str(p.relative_to(workspace)) for p in tests])
    return None


# ---------------------------------------------------------------- разбор вывода

_FRAME_RE = re.compile(r'^\s*File "(.+?)", line (\d+)(?:, in (.+))?\s*$')
_HEADER_RE = re.compile(r"^(FAIL|ERROR|UNEXPECTED SUCCESS): (\S+)(?: \(([^)]*)\))?")
_SEP_EQ_RE = re.compile(r"^={20,}\s*$")
_SEP_DASH_RE = re.compile(r"^-{20,}\s*$")
_RAN_RE = re.compile(r"^Ran (\d+) tests? in ", re.M)
_VERBOSE_RE = re.compile(r"^(\w+) \([^)]*\).*\.\.\. (unexpected success)\s*$", re.M)
_PYTEST_FAILED_RE = re.compile(r"^(?:FAILED|ERROR) (\S+?)::(?:\S+::)*?(\w+)(?:\[[^\]]*\])?(?: - (.*))?$", re.M)
_PYTEST_SUMMARY_RE = re.compile(
    r"^=*\s*((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected),? ?)+)"
    r"(?:in [\d.]+s)?.*$", re.M)
_STDLIB_MARKERS = ("/lib/python", "site-packages", "/unittest/")


def _clip(text: str, limit: int = _DETAIL_LIMIT) -> str:
    text = " ".join(text.split()) if "\n" in text else text
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _is_stdlib(path: str) -> bool:
    return path.startswith("<") or any(m in path for m in _STDLIB_MARKERS)


@dataclass
class _Frame:
    path: str
    line: int
    func: str
    src: str

    @property
    def base(self) -> str:
        return Path(self.path).name


def _last_traceback(lines: list[str]) -> tuple[list[_Frame], str]:
    """Кадры последнего трейсбэка и первая строка исключения."""
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            start = i
    frames: list[_Frame] = []
    exc = ""
    i = start
    last_frame_end = start
    while i < len(lines):
        m = _FRAME_RE.match(lines[i])
        if m and lines[i].startswith(" "):
            src = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("    ") and not _FRAME_RE.match(lines[i + 1]):
                src = lines[i + 1].strip()
            frames.append(_Frame(m.group(1), int(m.group(2)), (m.group(3) or "").strip(), src))
            last_frame_end = i + 1
        i += 1
    for line in lines[last_frame_end:]:
        if not line.strip() or line.startswith((" ", "\t")) or line.startswith("Traceback"):
            continue
        exc = line.strip()
        break
    return frames, exc


def _detail(name: str, body: list[str]) -> str:
    frames, exc = _last_traceback(body)
    if not frames:
        return _clip(exc or " ".join(l.strip() for l in body if l.strip())[:_DETAIL_LIMIT])
    test_idx = None
    for idx, fr in enumerate(frames):
        if fr.func == name:
            test_idx = idx
    if test_idx is None:
        for idx, fr in enumerate(frames):
            if _is_test_name(fr.base):
                test_idx = idx
    if test_idx is None:
        test_idx = 0
    head = frames[test_idx].src or f"{frames[test_idx].base}:{frames[test_idx].line}"
    inner = None
    for fr in frames[test_idx + 1:]:
        if not _is_test_name(fr.base) and not _is_stdlib(fr.path):
            inner = fr
    if inner is not None:
        where = f"{inner.base}:{inner.line}" + (f" in {inner.func}" if inner.func else "")
        return _clip(f"{head} -> {where}: {exc}")
    return _clip(f"{head} -> {exc}")


def _startup_error(lines: list[str]) -> str:
    """Последняя строка трейсбэка + место (имя файла:строка)."""
    frames, exc = _last_traceback(lines)
    if not exc:
        tail = [l.strip() for l in lines if l.strip()]
        exc = tail[-1] if tail else "tests did not run"
    user = [f for f in frames if not _is_stdlib(f.path)] or frames
    if user:
        return _clip(f"{exc} ({user[-1].base}:{user[-1].line})")
    return _clip(exc)


def _strip_paths(text: str, workspace: Path) -> str:
    # Сначала самый длинный вариант пути: на macOS /var — ссылка на /private/var, и если убрать
    # короткий раньше, от полного останется обрывок «/private».
    prefixes = {str(workspace), str(Path(workspace).resolve())}
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix and prefix != "/":
            text = text.replace(prefix + "/", "")
    return text


def _truncate_raw(text: str) -> str:
    if len(text) <= 2 * _RAW_HALF:
        return text
    return text[:_RAW_HALF] + f"\n...[{len(text) - 2 * _RAW_HALF} chars omitted]...\n" + text[-_RAW_HALF:]


def _parse(output: str, returncode: int, timed_out: bool, workspace: Path) -> TestReport:
    lines = output.splitlines()
    failed: list[str] = []
    details: dict[str, str] = {}
    error = ""

    def add(name: str, detail: str) -> None:
        if name not in failed:
            failed.append(name)
            details[name] = detail

    # блоки «=====\nFAIL: name (...)\n-----\n<трейсбэк>»
    i = 0
    while i < len(lines):
        m = _HEADER_RE.match(lines[i])
        if not m or (i > 0 and not _SEP_EQ_RE.match(lines[i - 1])):
            i += 1
            continue
        name, where = m.group(2), m.group(3) or ""
        j = i + 1
        while j < len(lines) and not _SEP_DASH_RE.match(lines[j]) and not _SEP_EQ_RE.match(lines[j]):
            j += 1
        k = j + 1
        while k < len(lines) and not _SEP_DASH_RE.match(lines[k]) and not _SEP_EQ_RE.match(lines[k]):
            k += 1
        body = lines[j + 1:k]
        if "_FailedTest" in where or "ModuleImportFailure" in where:
            error = error or _startup_error(body)  # модуль тестов не импортировался
        elif m.group(1) == "UNEXPECTED SUCCESS":
            add(name, "unexpected success")
        else:
            add(name, _detail(name, body))
        i = k

    for m in _VERBOSE_RE.finditer(output):
        add(m.group(1), "unexpected success")

    total = 0
    ran = _RAN_RE.findall(output)
    started = bool(ran)
    if ran:
        total = int(ran[-1])
    else:
        # pytest
        summ = None
        for m in _PYTEST_SUMMARY_RE.finditer(output):
            summ = m.group(1)
        if summ:
            started = True
            counts = {k: int(n) for n, k in re.findall(r"(\d+) (\w+)", summ)}
            total = counts.get("passed", 0) + counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
            for m in _PYTEST_FAILED_RE.finditer(output):
                add(m.group(2), _clip((m.group(3) or "failed").strip()))
            if "error" in counts or "errors" in counts:
                if "error" in output.lower() and "ERROR collecting" in output:
                    error = error or _startup_error(lines)

    if not started and not timed_out and returncode != 0:
        error = error or _startup_error(lines)
    if started and total == 0 and returncode != 0 and not failed:
        error = error or "no tests ran"

    ok = returncode == 0 and not failed and not timed_out and not error
    return TestReport(ok=ok, total=total, failed=failed, details=details, timed_out=timed_out,
                      error=error, raw=_truncate_raw(output))


def run_tests(workspace: Path, argv: Optional[list[str]] = None, timeout: float = 60.0) -> TestReport:
    """Запускает тесты в workspace и возвращает разобранный отчёт."""
    workspace = Path(workspace)
    if argv is None:
        argv = find_test_command("", workspace)
    if _wants_missing_pytest(argv):
        # pytest в окружении нет: команда из условия всегда падала бы с «No module named pytest»,
        # и ни навык, ни модель не смогли бы подтвердить исправление. Те же тесты — без pytest.
        argv = _without_pytest(argv, workspace) or argv
    started = time.monotonic()
    res = sandbox.run_argv(argv, workspace, timeout)
    output = res.stdout
    if res.stdout and res.stderr:
        output += "\n"
    output += res.stderr
    output = _strip_paths(output, workspace)
    report = _parse(output, res.returncode, res.timed_out, workspace)
    report.seconds = time.monotonic() - started
    return report


def summarize_failures(report: TestReport, max_chars: int = 1200) -> str:
    """Текст для промпта модели: по строке на упавший тест (или ошибка запуска)."""
    if report.error:
        return _clip(f"Tests did not start: {report.error}", max_chars)
    lines = []
    if report.timed_out:
        lines.append("Tests timed out (possible infinite loop).")
    for name in report.failed:
        lines.append(f"{name}: {report.details.get(name) or 'failed'}")
    if not lines and not report.ok:
        tail = report.raw.strip().splitlines()[-5:]
        lines.append("Tests failed: " + " | ".join(t.strip() for t in tail))
    if not lines:
        return "All tests passed."
    out: list[str] = []
    used = 0
    for idx, line in enumerate(lines):
        rest = len(lines) - idx
        note = f"... {rest} more failing tests not shown"
        need = len(line) + (1 if out else 0)
        reserve = len(note) + 1 if rest > 1 else 0
        if used + need + reserve > max_chars:
            if not out:
                out.append(line[: max(0, max_chars - len(note) - 4)] + "...")
                rest -= 1
            if rest > 0:
                out.append(f"... {rest} more failing tests not shown")
            break
        out.append(line)
        used += need
    return "\n".join(out)


# ---------------------------------------------------------------- противоречия

_EQ_NAMES = {"assertEqual", "assertEquals", "failUnlessEqual"}
_NE_NAMES = {"assertNotEqual", "assertNotEquals", "failIfEqual"}
_SETUP_NAMES = {"setUp", "setUpClass", "asyncSetUp"}


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, (ast.Call, ast.Name)):
        return False
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def _func_ok(func: ast.AST) -> bool:
    node = func
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id not in ("self", "cls")


def _is_literal_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not _func_ok(node.func):
        return False
    if any(isinstance(a, ast.Starred) or not _is_literal(a) for a in node.args):
        return False
    return all(k.arg is not None and _is_literal(k.value) for k in node.keywords)


@dataclass
class _Fact:
    kind: str        # eq / ne / true / false / raises
    key: str         # ast.dump вызова
    expr: str        # исходный текст вызова
    value: object
    test: str
    scope: str       # класс (или "" для функций модуля)
    nargs: int
    file: str


def _self_assert(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
        return f.attr
    return None


def _facts_from_call(call: ast.Call, out: list) -> None:
    name = _self_assert(call)
    if name is None:
        return
    args = call.args
    if name in _EQ_NAMES | _NE_NAMES and len(args) >= 2:
        kind = "eq" if name in _EQ_NAMES else "ne"
        a, b = args[0], args[1]
        if _is_literal_call(a) and _is_literal(b):
            out.append((kind, a, ast.literal_eval(b)))
        elif _is_literal_call(b) and _is_literal(a):
            out.append((kind, b, ast.literal_eval(a)))
    elif name in ("assertTrue", "assert_") and args and _is_literal_call(args[0]):
        out.append(("true", args[0], True))
    elif name == "assertFalse" and args and _is_literal_call(args[0]):
        out.append(("false", args[0], False))
    elif name == "assertIsNone" and args and _is_literal_call(args[0]):
        out.append(("eq", args[0], None))
    elif name == "assertIsNotNone" and args and _is_literal_call(args[0]):
        out.append(("ne", args[0], None))
    elif name == "assertRaises" and len(args) >= 2:
        fn = args[1]
        if isinstance(fn, (ast.Name, ast.Attribute)):
            synth = ast.Call(func=fn, args=list(args[2:]), keywords=list(call.keywords))
            if _is_literal_call(synth):
                out.append(("raises", synth, None))


def _facts_from_stmt(stmt: ast.stmt, out: list) -> bool:
    """Собирает утверждения; False — оператор «нечистый» (метод пропускается)."""
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Constant):
            return True
        if isinstance(stmt.value, ast.Call) and (_self_assert(stmt.value) or "").startswith("assert"):
            _facts_from_call(stmt.value, out)
            return True
        return False
    if isinstance(stmt, ast.Assert):
        t = stmt.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Eq, ast.NotEq)):
            kind = "eq" if isinstance(t.ops[0], ast.Eq) else "ne"
            a, b = t.left, t.comparators[0]
            if _is_literal_call(a) and _is_literal(b):
                out.append((kind, a, ast.literal_eval(b)))
            elif _is_literal_call(b) and _is_literal(a):
                out.append((kind, b, ast.literal_eval(a)))
        elif isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) and _is_literal_call(t.operand):
            out.append(("false", t.operand, False))
        elif _is_literal_call(t):
            out.append(("true", t, True))
        return True
    if isinstance(stmt, ast.With) and len(stmt.items) == 1:
        ctx = stmt.items[0].context_expr
        if isinstance(ctx, ast.Call) and _self_assert(ctx) == "assertRaises" and len(ctx.args) == 1 \
                and stmt.items[0].optional_vars is None:
            body = [s for s in stmt.body if not isinstance(s, ast.Pass)]
            if len(body) == 1 and isinstance(body[0], ast.Expr) and _is_literal_call(body[0].value):
                out.append(("raises", body[0].value, None))
                return True
        return False
    return False


def _collect_facts(path: Path) -> list[_Fact]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return []
    facts: list[_Fact] = []

    def handle(fn: ast.AST, scope: str) -> None:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.name.startswith("test"):
            return
        raw: list = []
        for stmt in fn.body:
            if not _facts_from_stmt(stmt, raw):
                return
        for kind, node, value in raw:
            facts.append(_Fact(kind, ast.dump(node), ast.unparse(node), value, fn.name, scope,
                               len(node.args) + len(node.keywords), path.name))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            has_setup = any(isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name in _SETUP_NAMES
                            for b in node.body)
            scope = f"class:{node.name}" if has_setup else ""
            for b in node.body:
                handle(b, scope)
        else:
            handle(node, "")
    return facts


def _safe_eq(a: object, b: object) -> Optional[bool]:
    try:
        return bool(a == b)
    except Exception:
        return None


def _conflict(f: _Fact, g: _Fact) -> Optional[str]:
    kinds = {f.kind, g.kind}
    if f.kind == g.kind == "eq":
        eq = _safe_eq(f.value, g.value)
        return "different values" if eq is False else None
    if kinds == {"eq", "ne"}:
        return "equal and not equal" if _safe_eq(f.value, g.value) else None
    if kinds == {"true", "false"}:
        return "true and false"
    if "eq" in kinds and kinds & {"true", "false"} and len(kinds) == 2:
        e, t = (f, g) if f.kind == "eq" else (g, f)
        try:
            truth = bool(e.value)
        except Exception:
            return None
        return "truthiness" if truth != (t.kind == "true") else None
    if "raises" in kinds and len(kinds) == 2:
        return "raises and returns"
    return None


def _describe(f: _Fact) -> str:
    if f.kind == "eq":
        return f"{f.expr} == {f.value!r}"
    if f.kind == "ne":
        return f"{f.expr} != {f.value!r}"
    if f.kind == "true":
        return f"{f.expr} is true"
    if f.kind == "false":
        return f"{f.expr} is false"
    return f"{f.expr} raises"


def contradictory_asserts(workspace: Path) -> list[str]:
    """Пары утверждений в тестах, которые не выполнимы одновременно (статически, через ast)."""
    facts: list[_Fact] = []
    for p in test_files(workspace):
        facts.extend(_collect_facts(p))
    by_key: dict[str, list[_Fact]] = {}
    for f in facts:
        by_key.setdefault(f.key, []).append(f)
    out: list[str] = []
    for group in by_key.values():
        for i, f in enumerate(group):
            for g in group[i + 1:]:
                if f.file != g.file and (f.scope or g.scope):
                    continue
                if f.scope != g.scope and (f.scope or g.scope):
                    continue
                if f.test == g.test and f.nargs == 0:
                    continue  # повторный вызов без аргументов в одном тесте может зависеть от состояния
                if _conflict(f, g) is None:
                    continue
                msg = f"{f.file}: {f.test} expects {_describe(f)}, but {g.test} expects {_describe(g)}"
                if msg not in out:
                    out.append(msg)
                if len(out) >= 20:
                    return out
    return out


# ---------------------------------------------------------------- тесты, которые не пройдут ни при каком коде

_PURE_NODES = (ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.List, ast.Tuple, ast.Set, ast.Dict,
               ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)


def _const_value(node: ast.AST) -> tuple[bool, object]:
    """Значение выражения из одних констант (1, 2 + 2, [1, 2]); (False, None) — выражение не константное."""
    if not all(isinstance(n, _PURE_NODES) for n in ast.walk(node)):
        return False, None
    try:
        return True, eval(compile(ast.Expression(body=node), "<const>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    except Exception:  # noqa: BLE001
        return False, None


def unsatisfiable_tests(workspace: Path) -> list[str]:
    """Причины, по которым тесты не пройдут ни при какой правке исходников (править тесты нельзя):
    тестовый файл с синтаксической ошибкой; безусловное утверждение над константами («assertEqual(1, 2)»,
    «assertTrue(False)», «self.fail()»)."""
    reasons: list[str] = []
    for path in test_files(workspace):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            reasons.append(f"{path.name} has a syntax error (line {getattr(exc, 'lineno', '?')}) and tests must not be edited")
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.name.startswith("test"):
                continue
            for stmt in fn.body:          # только безусловные операторы верхнего уровня теста
                bad = None
                if isinstance(stmt, ast.Assert):
                    ok, value = _const_value(stmt.test)
                    if ok and not value:
                        bad = ast.unparse(stmt)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    name = _self_assert(call)
                    args = call.args
                    if name == "fail":
                        bad = ast.unparse(call)
                    elif name in ("assertTrue", "assert_", "assertFalse") and args:
                        ok, value = _const_value(args[0])
                        if ok and bool(value) != (name != "assertFalse"):
                            bad = ast.unparse(call)
                    elif name in _EQ_NAMES | _NE_NAMES and len(args) >= 2:
                        ok_a, a = _const_value(args[0])
                        ok_b, b = _const_value(args[1])
                        if ok_a and ok_b and _safe_eq(a, b) is not None and _safe_eq(a, b) != (name in _EQ_NAMES):
                            bad = ast.unparse(call)
                if bad:
                    reasons.append(f"{fn.name}: `{bad[:120]}` can never pass")
    return reasons
