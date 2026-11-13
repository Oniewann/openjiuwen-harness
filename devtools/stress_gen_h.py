#!/usr/bin/env python3
"""Стресс-генератор, второй круг пересказов: coding-инструкции другими словами и «новые жанры».

Отдельный от stress_gen.py набор гипотез (функции задач и проверки берутся оттуда, только чтение):
  coding — другие формулировки задания, другое слово-маркер, несколько модулей, тесты в подкаталоге,
           pytest вместо unittest, тестовый файл с необычным именем;
  math   — проценты, единицы измерения, даты и время;
  data   — json/jsonl/xlsx/tsv вместо csv;
  search — вопросы о связи двух людей.

    python devtools/stress_gen_h.py --out DIR [--per 3] [--seed 1] [--only coding] [--match substr]
    python devtools/trial.py --tasks DIR --no-llm --show-log 0 --out RES.json
    python devtools/stress_gen_h.py --recheck RES.json --meta DIR.meta.json     # обязательно для coding
    python devtools/stress_report.py --result RES.json --meta DIR.meta.json --bad --show WRONG,STATIC

evaluate.py для coding умеет только «DONE + команда из инструкции»; здесь маркер бывает другим, а тесты
лежат в подкаталоге или написаны для pytest (его в образе нет). Поэтому у coding-задач в meta есть поле
h_verify (маркер и команда проверки), а --recheck пересчитывает их строки в RES.json по рабочей копии,
которую оставил trial.py: ответ сравнивается с маркером после strip+lower (как у организаторов), тесты
запускаются на нетронутых эталонных тестовых файлах.
"""
from __future__ import annotations

import argparse
import ast
import calendar
import datetime as dt
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stress_gen as sg  # noqa: E402
from stress_gen import (CQ, F, KB_INTRO, SALES_COLUMNS, SIMPLE, TAIL_ANSWER, TAILS, _apply_bug, _pm,  # noqa: E402
                        build_kb, kb_files, run_tests, sales, test_file)

H_REG: list[dict] = []
PY = "{py}"      # в h_verify.argv заменяется на текущий интерпретатор


def h(task_type: str, level: str, kind: str = "ok"):
    def deco(fn):
        H_REG.append({"type": task_type, "level": level, "kind": kind, "name": fn.__name__, "fn": fn})
        return fn
    return deco


def retry(fn):
    """Генератор может вернуть None (задача не получилась) — пробуем ещё раз с тем же rng."""
    def wrapped(rng):
        for _ in range(60):
            spec = fn(rng)
            if spec is not None:
                return spec
        raise RuntimeError(f"cannot generate {fn.__name__}")
    wrapped.__name__ = fn.__name__
    return wrapped


# =========================================================================== CODING: сборка задач
def _names_bugs(rng, n, k):
    pool = [x for x in F if x != "binary_search"]
    names = rng.sample(pool, n)
    bugs = {}
    for b in rng.sample(names, k):
        ks = [x for x in F[b]["bugs"] if x in SIMPLE]
        if not ks:
            return None, None
        bugs[b] = rng.choice(ks)
    return names, bugs


def _bad_src(name, bugs):
    return _apply_bug(F[name]["src"], F[name]["bugs"][bugs[name]]) if name in bugs else F[name]["src"]


def _each_bug_visible(bugs):
    """Каждый баг роняет видимый тест своей функции (как у организаторов)."""
    for name in bugs:
        alone = _apply_bug(F[name]["src"], F[name]["bugs"][bugs[name]])
        if run_tests({"mylib.py": alone, "test_mylib.py": test_file([(name, list(F[name]["vis"]))])}, timeout=5):
            return False
    return True


def _eq_args(a):
    """self.assertEqual(x, y) -> (x, y) по синтаксическому дереву: в y бывают запятые ([3, 2, 1])."""
    call = ast.parse(a.strip()).body[0].value
    return tuple(ast.get_source_segment(a.strip(), arg) for arg in call.args)


def _plain_asserts(name, hidden):
    out = []
    for a in F[name]["vis"] + (F[name]["hid"] if hidden else []):
        if "assertRaises" in a:
            continue
        if a.startswith("self.assertEqual("):
            head, tail = _eq_args(a)
            out.append(f"    assert {head} == {tail}")
        elif a.startswith("self.assertTrue("):
            out.append("    assert " + a[len("self.assertTrue("):-1])
        else:
            out.append("    assert not " + a[len("self.assertFalse("):-1])
    return out


def plain_test_file(names, hidden, header="import mylib as m"):
    lines = [header, "", ""]
    for n in names:
        lines.append(f"def test_{n}():")
        lines.extend(_plain_asserts(n, hidden) or ["    pass"])
        lines += ["", ""]
    return "\n".join(lines).rstrip() + "\n"


# Прогон голых функций-тестов без pytest: для проверки в генераторе и в --recheck.
PLAIN_RUNNER = (
    "import importlib.util, os, sys\n"
    "sys.path.insert(0, os.getcwd())\n"
    "bad = 0\n"
    "for path in sys.argv[1:]:\n"
    "    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))\n"
    "    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)\n"
    "    mod = importlib.util.module_from_spec(spec)\n"
    "    spec.loader.exec_module(mod)\n"
    "    for k in sorted(vars(mod)):\n"
    "        f = getattr(mod, k)\n"
    "        if k.startswith('test') and callable(f) and not isinstance(f, type):\n"
    "            try:\n"
    "                f()\n"
    "            except Exception as exc:\n"
    "                bad += 1\n"
    "                print('FAIL', k, repr(exc))\n"
    "sys.exit(1 if bad else 0)\n"
)


def run_argv(files, argv, timeout=20):
    with tempfile.TemporaryDirectory(prefix="agH-gen-") as tmp:
        for rel, text in files.items():
            p = Path(tmp, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        cmd = [sys.executable if a == PY else a for a in argv]
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0


def coding_spec(instruction, good_files, bad_files, hidden_tests, verify_argv, marker="DONE", feasible=True):
    """Общая сборка coding-задачи с проверкой: исправная версия проходит скрытые тесты, сломанная — нет."""
    if feasible:
        if not run_argv({**good_files, **hidden_tests}, verify_argv):
            return None
        if run_argv(bad_files, verify_argv):
            return None
    fixed = {k: val for k, val in good_files.items() if k not in hidden_tests and bad_files.get(k) != val}
    return dict(instruction=instruction, expected="tests_pass" if feasible else None, files=bad_files, check="unittest",
                tests=hidden_tests, fixed=fixed, h_verify={"marker": marker, "argv": verify_argv})


def flat_task(rng, instruction, n=4, k=2, marker="DONE", tests_name="test_mylib.py", module="mylib",
              verify=None, header=None, plain=False):
    """Один модуль и один тестовый файл, разные имена и инструкции."""
    names, bugs = _names_bugs(rng, n, k)
    if not names or not _each_bug_visible(bugs):
        return None
    good = "\n\n".join(F[x]["src"] for x in names)
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    head = header or f"import {module} as m"
    if plain:
        vis, hid = plain_test_file(names, False, head), plain_test_file(names, True, head)
    else:
        vis = test_file([(x, list(F[x]["vis"])) for x in names], module)
        hid = test_file([(x, list(F[x]["vis"]) + list(F[x]["hid"])) for x in names], module)
        if header:
            vis, hid = vis.replace(f"import {module} as m", header, 1), hid.replace(f"import {module} as m", header, 1)
    src_name = f"{module}.py"
    argv = verify or ([PY, "-B", "-c", PLAIN_RUNNER, tests_name] if plain else [PY, "-B", tests_name])
    return coding_spec(instruction, {src_name: good, tests_name: hid}, {src_name: bad, tests_name: vis},
                       {tests_name: hid}, argv, marker)


DONE_TAIL = " When you are done, write ONLY the word DONE to the file answer.txt in the current directory."
GENERIC_TAIL = " Work only inside the current directory."


# --------------------------------------------------------------------------- A. другие формулировки
PHRASINGS = [
    "Fix the bug so that the tests pass. Then write DONE to answer.txt.",
    "Make test_mylib.py green. Do not modify the tests. Afterwards create answer.txt containing DONE.",
    "The tests in test_mylib.py are failing. Repair mylib.py so that `python3 test_mylib.py` succeeds, then write the word DONE into answer.txt.",
    "There are bugs in mylib.py. Find and fix them; the test suite (test_mylib.py) must pass. Finally, put DONE in a file called answer.txt.",
    "Some functions in mylib.py return wrong results. Fix them without touching test_mylib.py. Write DONE to answer.txt when all tests pass.",
    "Please debug mylib.py. Success criterion: `python3 -m unittest test_mylib` exits with code 0. Report completion by writing DONE to answer.txt.",
    "Fix mylib.py so that all unit tests pass. Output: the file answer.txt with the single word DONE.",
    "Your job: the unit tests for mylib are red. Turn them green by editing only mylib.py, then echo DONE > answer.txt.",
    "Run the tests (`python3 -m unittest -v`). Correct the implementation, not the tests. When everything passes, save DONE to answer.txt.",
]


@h("coding", "mid")
@retry
def c_h_phrasing(rng):
    return flat_task(rng, rng.choice(PHRASINGS) + GENERIC_TAIL)


@h("coding", "mid")
@retry
def c_h_phrasing_one_bug(rng):
    return flat_task(rng, rng.choice(PHRASINGS) + GENERIC_TAIL, n=3, k=1)


WRONG_VALUE = {
    "total": "an empty list", "maximum": "lists of negative numbers", "average": "some inputs", "factorial": "n = 5",
    "fib": "n = 10", "gcd": "some pairs of numbers", "power": "small exponents", "median": "unsorted input",
}


@h("coding", "mid")
@retry
def c_h_named_function(rng):
    """В условии названа одна функция — именно она и сломана."""
    names, bugs = _names_bugs(rng, rng.randint(3, 5), 1)
    if not names:
        return None
    (f,) = bugs
    what = WRONG_VALUE.get(f, "some inputs")
    instr = rng.choice([
        f"The function `{f}` in mylib.py returns the wrong value for {what}. Fix it so that `python3 test_mylib.py` passes.",
        f"`{f}()` is broken: test_mylib.py shows it returns the wrong value. Fix {f} in mylib.py (leave the tests alone).",
        f"Bug report: mylib.{f} gives wrong results. Please fix the function so that all tests in test_mylib.py pass.",
    ]) + DONE_TAIL
    if not _each_bug_visible(bugs):
        return None
    good = "\n\n".join(F[x]["src"] for x in names)
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    vis = test_file([(x, list(F[x]["vis"])) for x in names])
    hid = test_file([(x, list(F[x]["vis"]) + list(F[x]["hid"])) for x in names])
    spec = coding_spec(instr, {"mylib.py": good, "test_mylib.py": hid}, {"mylib.py": bad, "test_mylib.py": vis},
                       {"test_mylib.py": hid}, [PY, "-B", "test_mylib.py"])
    return spec


# --------------------------------------------------------------------------- B. другое слово-маркер
MARKER_FORMS = [
    ("write ONLY the word {w} to the file answer.txt", True),
    ("write the word {w} to answer.txt", False),
    ("write `{w}` to answer.txt", False),
    ('write ONLY the word "{w}" to the file answer.txt', False),
    ("write ONLY the word '{w}' to the file answer.txt", False),
    ("run `echo {w} > answer.txt`", False),
    ("answer.txt must contain exactly one word: {w}", False),
    ("write ONLY the text {w} to answer.txt", False),
    ("create answer.txt with the content {w}", False),
    ("write {w} (and nothing else) to answer.txt", False),
]
MARKERS = ["FIXED", "OK", "PASS", "SUCCESS", "COMPLETE", "ALL_TESTS_PASS", "fixed", "Done"]


def _marker_task(rng, word, form):
    instr = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
             "(do not edit the tests) until all tests pass. Work only inside the current directory. When you are done, "
             + form.format(w=word) + ".")
    return flat_task(rng, instr, marker=word)


@h("coding", "hard")
@retry
def c_h_marker_word(rng):
    form, _ = rng.choice(MARKER_FORMS)
    return _marker_task(rng, rng.choice(MARKERS), form)


@h("coding", "hard")
@retry
def c_h_marker_done_quoted(rng):
    """DONE, но в кавычках/обратных кавычках/другой форме фразы."""
    form, _ = rng.choice(MARKER_FORMS[1:])
    return _marker_task(rng, "DONE", form)


@h("coding", "hard")
@retry
def c_h_marker_lowercase_rule(rng):
    return _marker_task(rng, rng.choice(["done", "Done"]), "write only the word {w} to the file answer.txt")


# --------------------------------------------------------------------------- C. структура каталога
SYS_PATH_PARENT = ("import os\nimport sys\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
                   "import mylib as m")
SYS_PATH_SRC = ("import os\nimport sys\n\nsys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))\n"
                "import mylib as m")


@h("coding", "hard")
@retry
def c_h_two_modules(rng):
    """Два модуля, у каждого свой тестовый файл; баги в обоих."""
    names, bugs = _names_bugs(rng, 6, 0)
    if not names:
        return None
    a, b = names[:3], names[3:]
    bugs = {}
    for x in (rng.choice(a), rng.choice(b)):
        ks = [q for q in F[x]["bugs"] if q in SIMPLE]
        if not ks:
            return None
        bugs[x] = rng.choice(ks)
    if not _each_bug_visible(bugs):
        return None
    mod2 = rng.choice(["strutil", "mathutil", "helpers", "core"])
    good = {"mylib.py": "\n\n".join(F[x]["src"] for x in a), f"{mod2}.py": "\n\n".join(F[x]["src"] for x in b)}
    bad = {"mylib.py": "\n\n".join(_bad_src(x, bugs) for x in a), f"{mod2}.py": "\n\n".join(_bad_src(x, bugs) for x in b)}
    vis = {"test_mylib.py": test_file([(x, list(F[x]["vis"])) for x in a]),
           f"test_{mod2}.py": test_file([(x, list(F[x]["vis"])) for x in b], mod2)}
    hid = {"test_mylib.py": test_file([(x, list(F[x]["vis"]) + list(F[x]["hid"])) for x in a]),
           f"test_{mod2}.py": test_file([(x, list(F[x]["vis"]) + list(F[x]["hid"])) for x in b], mod2)}
    instr = rng.choice([
        "Run `python3 -m unittest discover` in the current directory; some tests fail. Fix the bugs in the source files "
        "(do not edit the tests) until all tests pass.",
        f"The project has two modules, mylib.py and {mod2}.py, with tests in test_mylib.py and test_{mod2}.py. "
        "Fix the bugs so that `python3 -m unittest discover` passes. Do not edit the tests.",
        "Some tests fail. Fix the bugs (there may be bugs in more than one file) so that all tests pass.",
    ]) + GENERIC_TAIL + DONE_TAIL
    return coding_spec(instr, {**good, **hid}, {**bad, **vis}, hid, [PY, "-B", "-m", "unittest", "discover"])


def _subdir_task(rng, layout):
    names, bugs = _names_bugs(rng, 4, 2)
    if not names or not _each_bug_visible(bugs):
        return None
    good = "\n\n".join(F[x]["src"] for x in names)
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    vis = test_file([(x, list(F[x]["vis"])) for x in names])
    hid = test_file([(x, list(F[x]["vis"]) + list(F[x]["hid"])) for x in names])
    extra = {}
    if layout == "tests_pkg":            # tests/ — пакет, запуск discover из корня
        src, tname = "mylib.py", "tests/test_mylib.py"
        extra = {"tests/__init__.py": ""}
        cmd, argv = "python3 -m unittest discover", [PY, "-B", "-m", "unittest", "discover"]
    elif layout == "tests_dir_s":        # tests/ без __init__, discover -s tests, путь к модулю — в тесте
        src, tname = "mylib.py", "tests/test_mylib.py"
        vis, hid = vis.replace("import mylib as m", SYS_PATH_PARENT, 1), hid.replace("import mylib as m", SYS_PATH_PARENT, 1)
        cmd, argv = "python3 -m unittest discover -s tests", [PY, "-B", "-m", "unittest", "discover", "-s", "tests"]
    elif layout == "tests_dir_script":   # запуск файла из подкаталога
        src, tname = "mylib.py", "tests/test_mylib.py"
        vis, hid = vis.replace("import mylib as m", SYS_PATH_PARENT, 1), hid.replace("import mylib as m", SYS_PATH_PARENT, 1)
        cmd, argv = "python3 tests/test_mylib.py", [PY, "-B", "tests/test_mylib.py"]
    elif layout == "src_layout":         # src/mylib.py и tests/test_mylib.py
        src, tname = "src/mylib.py", "tests/test_mylib.py"
        vis, hid = vis.replace("import mylib as m", SYS_PATH_SRC, 1), hid.replace("import mylib as m", SYS_PATH_SRC, 1)
        cmd, argv = "python3 -m unittest discover -s tests", [PY, "-B", "-m", "unittest", "discover", "-s", "tests"]
    else:                                # пакет: pkg/__init__.py + pkg/mylib.py, тест импортирует pkg.mylib
        pkg = rng.choice(["mypkg", "calc", "toolkit"])
        src, tname = f"{pkg}/mylib.py", "test_mylib.py"
        extra = {f"{pkg}/__init__.py": ""}
        vis, hid = (t.replace("import mylib as m", f"from {pkg} import mylib as m", 1) for t in (vis, hid))
        cmd, argv = "python3 test_mylib.py", [PY, "-B", "test_mylib.py"]
    instr = (f"Run `{cmd}` in the current directory; some tests fail. Fix the bugs in {src} (do not edit the tests) "
             "until all tests pass." + GENERIC_TAIL + DONE_TAIL)
    return coding_spec(instr, {src: good, tname: hid, **extra}, {src: bad, tname: vis, **extra}, {tname: hid}, argv)


for _lay in ("tests_pkg", "tests_dir_s", "tests_dir_script", "src_layout", "package"):
    def _mk(lay):
        def fn(rng):
            return _subdir_task(rng, lay)
        fn.__name__ = f"c_h_layout_{lay}"
        h("coding", "hard")(retry(fn))
    _mk(_lay)


# --------------------------------------------------------------------------- D. pytest вместо unittest
@h("coding", "hard")
@retry
def c_h_pytest_cmd_unittest_tests(rng):
    """Инструкция говорит про pytest, тесты — обычные unittest.TestCase (pytest их тоже запускает)."""
    instr = rng.choice([
        "Run `pytest -q` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit the tests) until all tests pass.",
        "Run `python3 -m pytest test_mylib.py`; some tests fail. Fix mylib.py until all tests pass, without editing the tests.",
        "Use pytest to run the tests. Fix the bugs in mylib.py so that every test passes.",
    ]) + GENERIC_TAIL + DONE_TAIL
    return flat_task(rng, instr, verify=[PY, "-B", "-m", "unittest", "discover"])


@h("coding", "hard")
@retry
def c_h_pytest_plain(rng):
    """Голые функции-тесты без unittest.main и без pytest в окружении."""
    instr = rng.choice([
        "Run `pytest test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit the tests) until all tests pass.",
        "Run `pytest -q`; some tests fail. Fix the bugs in mylib.py (do not edit the tests).",
        "The tests are written for pytest (test_mylib.py). Fix mylib.py so that they all pass.",
        "Run `python3 -m pytest -x` and fix the bugs in mylib.py until all tests pass. Do not modify test_mylib.py.",
    ]) + GENERIC_TAIL + DONE_TAIL
    return flat_task(rng, instr, plain=True)


@h("coding", "extreme")
@retry
def c_h_pytest_plain_subdir(rng):
    names, bugs = _names_bugs(rng, 4, 2)
    if not names or not _each_bug_visible(bugs):
        return None
    good = "\n\n".join(F[x]["src"] for x in names)
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    vis, hid = plain_test_file(names, False, SYS_PATH_PARENT), plain_test_file(names, True, SYS_PATH_PARENT)
    instr = ("Run `pytest tests/` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit the tests) "
             "until all tests pass." + GENERIC_TAIL + DONE_TAIL)
    return coding_spec(instr, {"mylib.py": good, "tests/test_mylib.py": hid}, {"mylib.py": bad, "tests/test_mylib.py": vis},
                       {"tests/test_mylib.py": hid}, [PY, "-B", "-c", PLAIN_RUNNER, "tests/test_mylib.py"])


# --------------------------------------------------------------------------- E. тестовый файл с необычным именем
@h("coding", "extreme")
@retry
def c_h_test_file_odd_name(rng):
    """test.py / tests.py / check_mylib.py / mylib_tests.py — не подходят под шаблон test_*.py."""
    tname = rng.choice(["test.py", "tests.py", "check_mylib.py", "mylib_tests.py", "run_tests.py"])
    instr = (f"Run `python3 {tname}` in the current directory; some tests fail. Fix the bugs in mylib.py "
             f"(do not edit {tname}) until all tests pass." + GENERIC_TAIL + DONE_TAIL)
    return flat_task(rng, instr, tests_name=tname)


@h("coding", "extreme")
@retry
def c_h_assert_script(rng):
    """Проверка — скрипт с assert на верхнем уровне (не unittest и не функции)."""
    names, bugs = _names_bugs(rng, 4, 2)
    if not names or not _each_bug_visible(bugs):
        return None
    good = "\n\n".join(F[x]["src"] for x in names)
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    sname = rng.choice(["check.py", "run_checks.py", "verify.py", "test_mylib.py"])

    def script(hidden):
        lines = ["import mylib as m", ""]
        for x in names:
            lines += [a[4:] for a in _plain_asserts(x, hidden)]
        return "\n".join(lines + ["", "print('all checks passed')", ""])
    instr = (f"Run `python3 {sname}`; it stops with an AssertionError. Fix the bugs in mylib.py (do not edit {sname}) "
             "until the script prints 'all checks passed'." + GENERIC_TAIL + DONE_TAIL)
    return coding_spec(instr, {"mylib.py": good, sname: script(True)}, {"mylib.py": bad, sname: script(False)},
                       {sname: script(True)}, [PY, "-B", sname])


@h("coding", "hard")
@retry
def c_h_two_test_files_no_cmd(rng):
    """Два тестовых файла на один модуль, команды в условии нет."""
    names, bugs = _names_bugs(rng, 4, 2)
    if not names or not _each_bug_visible(bugs):
        return None
    good = "\n\n".join(F[x]["src"] for x in names)
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    t1 = test_file([(x, list(F[x]["vis"])) for x in names])
    t2 = test_file([(x + "_edge", list(F[x]["hid"])) for x in names])
    instr = rng.choice(["Some tests in this directory fail. Fix mylib.py so that all tests pass (do not edit test files).",
                        "Fix the bugs in mylib.py. Tests: test_basic.py and test_edge.py."]) + GENERIC_TAIL + DONE_TAIL
    tests = {"test_basic.py": t1, "test_edge.py": t2}
    return coding_spec(instr, {"mylib.py": good, **tests}, {"mylib.py": bad, **tests}, tests,
                       [PY, "-B", "-m", "unittest", "discover", "-p", "test_*.py"])


@h("coding", "extreme")
@retry
def c_h_doctest(rng):
    """Тесты — doctest в docstring функций, отдельного тестового файла нет."""
    names, bugs = _names_bugs(rng, 3, 1)
    if not names or not _each_bug_visible(bugs):
        return None
    good_parts, bad_parts = [], []
    for x in names:
        doc = []
        for a in F[x]["vis"]:
            if a.startswith("self.assertEqual("):
                head, tail = _eq_args(a)
                doc += [f"    >>> {head.replace('m.', '', 1)}", f"    {tail}"]
        if not doc:
            return None
        ds = '    """\n' + "\n".join(doc) + '\n    """\n'
        for parts, src in ((good_parts, F[x]["src"]), (bad_parts, _bad_src(x, bugs))):
            first, _, rest = src.partition("\n")
            parts.append(first + "\n" + ds + rest)
    good, bad = "\n\n".join(good_parts), "\n\n".join(bad_parts)
    instr = ("Run `python3 -m doctest -v mylib.py`; some examples fail. Fix the code in mylib.py (do not change the "
             "examples in the docstrings) until all examples pass." + GENERIC_TAIL + DONE_TAIL)
    return coding_spec(instr, {"mylib.py": good}, {"mylib.py": bad}, {}, [PY, "-B", "-m", "doctest", "mylib.py"])


# --------------------------------------------------------------------------- F. невыполнимые в новых формах
def _contradiction_line(names):
    wrong = {"total": "m.total([1, 2, 3]) == 7", "maximum": "m.maximum([3, 9, 2]) == 3", "fib": "m.fib(10) == 56",
             "gcd": "m.gcd(12, 18) == 5", "power": "m.power(2, 5) == 31", "minimum": "m.minimum([3, 9, 2]) == 9",
             "abs_diff": "m.abs_diff(3, 10) == 8", "word_count": "m.word_count('a quick brown fox') == 5",
             "factorial": "m.factorial(5) == 121", "dot": "m.dot([1, 2, 3], [4, 5, 6]) == 33"}
    c = [n for n in names if n in wrong]
    return (c[0], wrong[c[0]]) if c else (None, None)


def _as_equal(line):
    head, _, tail = line.rpartition(" == ")
    return f"self.assertEqual({head}, {tail})"


@h("coding", "hard", "inf")
@retry
def c_h_i_subdir_contradiction(rng):
    names, bugs = _names_bugs(rng, 4, 1)
    if not names:
        return None
    t, line = _contradiction_line(names)
    if not t:
        return None
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    vis = test_file([(x, list(F[x]["vis"])) for x in names]).replace("import mylib as m", SYS_PATH_PARENT, 1)
    vis = vis.replace("\n\nif __name__", f"\n    def test_{t}_again(self):\n        {_as_equal(line)}\n\n\nif __name__", 1)
    instr = ("Run `python3 -m unittest discover -s tests`; some tests fail. Fix the bugs in mylib.py (do not edit the tests) "
             "until all tests pass." + GENERIC_TAIL + DONE_TAIL)
    return coding_spec(instr, {}, {"mylib.py": bad, "tests/test_mylib.py": vis}, {"tests/test_mylib.py": vis},
                       [PY, "-B", "-m", "unittest", "discover", "-s", "tests"], feasible=False)


@h("coding", "hard", "inf")
@retry
def c_h_i_marker_constant_false(rng):
    names, bugs = _names_bugs(rng, 4, 1)
    if not names:
        return None
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    vis = test_file([(x, list(F[x]["vis"])) for x in names])
    vis = vis.replace("\n\nif __name__", "\n    def test_sanity(self):\n        self.assertEqual(2 + 2, 5)\n\n\nif __name__", 1)
    w = rng.choice(MARKERS)
    instr = ("Run `python3 test_mylib.py`; some tests fail. Fix the bugs in mylib.py (do not edit the tests) until all tests "
             f"pass. When you are done, write the word {w} to answer.txt.")
    return coding_spec(instr, {}, {"mylib.py": bad, "test_mylib.py": vis}, {"test_mylib.py": vis},
                       [PY, "-B", "test_mylib.py"], marker=w, feasible=False)


@h("coding", "hard", "inf")
@retry
def c_h_i_pytest_contradiction(rng):
    names, bugs = _names_bugs(rng, 4, 1)
    if not names:
        return None
    t, line = _contradiction_line(names)
    if not t:
        return None
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    vis = plain_test_file(names, False) + f"\n\ndef test_{t}_again():\n    assert {line}\n"
    instr = "Run `pytest -q`; some tests fail. Fix the bugs in mylib.py (do not edit the tests)." + GENERIC_TAIL + DONE_TAIL
    return coding_spec(instr, {}, {"mylib.py": bad, "test_mylib.py": vis}, {"test_mylib.py": vis},
                       [PY, "-B", "-c", PLAIN_RUNNER, "test_mylib.py"], feasible=False)


@h("coding", "hard", "inf")
@retry
def c_h_i_odd_name_contradiction(rng):
    names, bugs = _names_bugs(rng, 4, 1)
    if not names:
        return None
    t, line = _contradiction_line(names)
    if not t:
        return None
    tname = rng.choice(["test.py", "tests.py", "check_mylib.py"])
    bad = "\n\n".join(_bad_src(x, bugs) for x in names)
    vis = test_file([(x, list(F[x]["vis"])) for x in names])
    vis = vis.replace("\n\nif __name__", f"\n    def test_{t}_again(self):\n        {_as_equal(line)}\n\n\nif __name__", 1)
    instr = (f"Run `python3 {tname}`; some tests fail. Fix the bugs in mylib.py (do not edit {tname}) until all tests pass."
             + GENERIC_TAIL + DONE_TAIL)
    return coding_spec(instr, {}, {"mylib.py": bad, tname: vis}, {tname: vis}, [PY, "-B", tname], feasible=False)


# =========================================================================== MATH: новые жанры
def MQ(q, value, check="exact_int"):
    return dict(question=q, expected=None if value is None else str(value), check=check)


@h("math", "extreme")
def m_h_percent_of(rng):
    p = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 75])
    n = rng.randint(2, 60) * 100
    val = n * p // 100
    return MQ(rng.choice([f"What is {p}% of {n}?", f"Calculate {p} percent of {n}.",
                          f"A school has {n} students; {p}% of them walk to school. How many students walk to school?",
                          f"A store had {n} items and sold {p}% of them. How many items were sold?"]), val)


@h("math", "extreme")
def m_h_percent_remaining(rng):
    p = rng.choice([10, 20, 25, 40, 50, 60])
    n = rng.randint(2, 50) * 20
    return MQ(rng.choice([f"A farmer harvested {n} apples and {p}% of them were rotten. How many apples were not rotten?",
                          f"A warehouse starts with {n} boxes. It ships {p}% of its stock. How many boxes are left?"]),
              n - n * p // 100)


@h("math", "extreme")
def m_h_percent_increase(rng):
    p = rng.choice([10, 20, 25, 50])
    n = rng.randint(2, 40) * 20
    new = n + n * p // 100
    return MQ(rng.choice([f"A price of {n} dollars is increased by {p}%. What is the new price in dollars?",
                          f"The population of a village is {n}. It grows by {p}%. What is the population now?",
                          f"A salary of {n} rises by {p} percent. What is the new salary?"]), new)


@h("math", "extreme")
def m_h_percent_change(rng):
    old = rng.randint(2, 40) * 10
    p = rng.choice([10, 20, 25, 30, 50, 75, 100])
    new = old + old * p // 100
    if old * p % 100:
        return m_h_percent_change(rng)
    return MQ(rng.choice([f"Sales grew from {old} to {new}. By what percent did sales increase?",
                          f"A price went up from {old} to {new} dollars. What is the percentage increase?"]), p)


@h("math", "extreme")
def m_h_compound_growth(rng):
    n = rng.randint(1, 30) * 400
    return MQ(f"A town has {n} residents. The population grows by 5% each year. How many residents will it have after 2 years?",
              n * 441 // 400)


@h("math", "extreme")
def m_h_units_time(rng):
    hrs, mins = rng.randint(1, 12), rng.choice([0, 15, 20, 30, 45])
    k = rng.randint(0, 3)
    if k == 0:
        return MQ(f"How many minutes are there in {hrs} hours and {mins} minutes?", hrs * 60 + mins)
    if k == 1:
        return MQ(f"How many seconds are there in {hrs} hours?", hrs * 3600)
    if k == 2:
        d = rng.randint(2, 9)
        return MQ(f"How many hours are there in {d} days?", d * 24)
    w = rng.randint(2, 12)
    return MQ(f"How many days are in {w} weeks?", w * 7)


@h("math", "extreme")
def m_h_units_metric(rng):
    k = rng.randint(0, 4)
    if k == 0:
        a = rng.randint(1, 40) / 4
        return MQ(f"Convert {a:g} km to meters.", int(a * 1000))
    if k == 1:
        a = rng.randint(1, 40) / 4
        return MQ(f"How many grams are in {a:g} kilograms?", int(a * 1000))
    if k == 2:
        m_, c = rng.randint(1, 30), rng.randint(1, 99)
        return MQ(f"How many centimeters are in {m_} meters and {c} centimeters?", m_ * 100 + c)
    if k == 3:
        v = rng.choice([18, 36, 54, 72, 90, 108])
        return MQ(f"A car travels at {v} km/h. What is its speed in meters per second?", v * 1000 // 3600)
    lit = rng.randint(1, 20) / 2
    return MQ(f"A bottle holds {lit:g} liters. How many milliliters is that?", int(lit * 1000))


@h("math", "extreme")
def m_h_units_imperial(rng):
    k = rng.randint(0, 2)
    if k == 0:
        ft = rng.randint(2, 30)
        return MQ(f"How many inches are in {ft} feet?", ft * 12)
    if k == 1:
        yd = rng.randint(2, 50)
        return MQ(f"How many feet are in {yd} yards?", yd * 3)
    lb = rng.randint(2, 20)
    return MQ(f"How many ounces are in {lb} pounds?", lb * 16)


def _date(rng, y0=2015, y1=2026):
    d0 = dt.date(y0, 1, 1)
    return d0 + dt.timedelta(days=rng.randint(0, (dt.date(y1, 12, 31) - d0).days))


@h("math", "extreme")
def m_h_days_between(rng):
    a = _date(rng)
    b = a + dt.timedelta(days=rng.randint(5, 400))
    fa, fb = rng.choice([("%Y-%m-%d", "%Y-%m-%d"), ("%B %d, %Y", "%B %d, %Y"), ("%d %B %Y", "%d %B %Y")])
    return MQ(rng.choice([f"How many days are there between {a.strftime(fa)} and {b.strftime(fb)}?",
                          f"How many days passed from {a.strftime(fa)} to {b.strftime(fb)}?"]), (b - a).days)


@h("math", "extreme")
def m_h_weekday(rng):
    a = _date(rng)
    return MQ(rng.choice([f"What day of the week was {a.isoformat()}?", f"On which weekday did {a.strftime('%B %d, %Y')} fall?"]),
              a.strftime("%A"), check="exact_str")


@h("math", "extreme")
def m_h_date_plus(rng):
    a = _date(rng)
    k = rng.randint(1, 90)
    return MQ(f"What is the date {k} days after {a.isoformat()}? Answer in YYYY-MM-DD format.",
              (a + dt.timedelta(days=k)).isoformat(), check="exact_str")


@h("math", "extreme")
def m_h_days_in_month(rng):
    y = rng.choice([1900, 2000, 2023, 2024, 2025, 2100, 2028])
    mth = rng.choice([2, 2, 4, 9, 11, 1])
    return MQ(f"How many days are in {calendar.month_name[mth]} {y}?", calendar.monthrange(y, mth)[1])


@h("math", "extreme")
def m_h_age(rng):
    born = _date(rng, 1950, 2005)
    on = _date(rng, 2020, 2026)
    age = on.year - born.year - ((on.month, on.day) < (born.month, born.day))
    return MQ(f"A person was born on {born.isoformat()}. How old (in full years) are they on {on.isoformat()}?", age)


@h("math", "extreme")
def m_h_clock_time(rng):
    h0, m0 = rng.randint(6, 15), rng.choice([0, 10, 15, 30, 45, 50])
    dh, dm = rng.randint(0, 5), rng.choice([5, 20, 35, 50])
    end = (h0 * 60 + m0 + dh * 60 + dm) % (24 * 60)
    return MQ(f"A meeting starts at {h0:02d}:{m0:02d} and lasts {dh} hours and {dm} minutes. At what time does it end? "
              "Answer in HH:MM format.", f"{end // 60:02d}:{end % 60:02d}", check="exact_str")


@h("math", "extreme")
def m_h_weekdays_in_month(rng):
    y, mth = rng.randint(2020, 2026), rng.randint(1, 12)
    n = sum(1 for d in range(1, calendar.monthrange(y, mth)[1] + 1) if dt.date(y, mth, d).weekday() < 5)
    return MQ(f"How many weekdays (Monday to Friday) are there in {calendar.month_name[mth]} {y}?", n)


@h("math", "hard")
def m_h_warehouse_units_mismatch(rng):
    """Склад как в примерах, но срок в другой единице: «per day for 2 weeks», «for 48 hours», «dozen»."""
    n = rng.randint(800, 2400)
    a = rng.randint(5, 30)
    k = rng.randint(0, 4)
    if k == 0:
        w = rng.randint(1, 3)
        return MQ(f"A warehouse starts with {n} boxes. It ships {a} boxes per day for {w} weeks. {sg.QEND}", n - a * 7 * w)
    if k == 1:
        hrs = rng.choice([24, 48, 72])
        return MQ(f"A warehouse starts with {n} boxes. It ships {a} boxes per day for {hrs} hours. {sg.QEND}", n - a * hrs // 24)
    if k == 2:
        d, dz = rng.randint(2, 5), rng.randint(2, 6)
        return MQ(f"A warehouse starts with {n} boxes. It receives {d} deliveries of {dz} dozen boxes each. {sg.QEND}",
                  n + d * dz * 12)
    if k == 3:
        d0, d1 = rng.randint(1, 10), rng.randint(12, 28)
        return MQ(f"A warehouse starts with {n} boxes. It ships {a} boxes per day from March {d0} to March {d1} inclusive. "
                  f"{sg.QEND}", n - a * (d1 - d0 + 1))
    p = rng.choice([10, 20, 25, 50])
    n = rng.randint(4, 24) * 100
    return MQ(f"A warehouse starts with {n} boxes. It ships {p}% of its boxes. It receives 2 deliveries of 50 boxes each. "
              f"{sg.QEND}", n - n * p // 100 + 100)


@h("math", "extreme", "inf")
def m_h_i_percent_no_base(rng):
    p = rng.choice([10, 15, 20, 25])
    return MQ(rng.choice([f"A store sold {p}% of its stock. How many items were sold?",
                          f"The price increased by {p}%. What is the new price?"]), None)


@h("math", "extreme", "inf")
def m_h_i_date_no_year(rng):
    a = _date(rng)
    return MQ(f"What day of the week was {a.strftime('%B %d')}?", None, check="exact_str")


# =========================================================================== DATA: другие форматы файлов
def _base_question(rng, rows):
    p, lo, hi = _pm(rng)
    return (f"How many units of product '{p}' were sold in months {lo} through {hi} inclusive?",
            sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi))


def _revenue_region(rng, rows):
    rev = {}
    for r in rows:
        rev[r["region"]] = rev.get(r["region"], 0) + r["units"] * r["price"]
    best = max(rev.values())
    if list(rev.values()).count(best) > 1:
        return None
    return ("Which region has the highest total revenue (units * price)?", max(rev, key=rev.get))


def DF(question, expected, files, check="exact_int"):
    return dict(question=question, expected=None if expected is None else str(expected), files=files, check=check)


@h("data_analysis", "extreme")
def d_h_json_wrapped(rng):
    rows = sales(rng, rng.choice([20, 50]))
    q, e = _base_question(rng, rows)
    return DF(f"The file orders.json contains an object with a key \"orders\" holding a list of orders "
              f"({len(rows)} records, fields {','.join(SALES_COLUMNS)}). {q}", e, {"orders.json": json.dumps({"orders": rows}, indent=2)})


@h("data_analysis", "extreme")
def d_h_jsonl(rng):
    rows = sales(rng, rng.choice([20, 50, 100]))
    q, e = _base_question(rng, rows)
    return DF(f"The file sales.jsonl has one JSON order per line with fields {','.join(SALES_COLUMNS)}. {q}", e,
              {"sales.jsonl": "".join(json.dumps(r) + "\n" for r in rows)})


@h("data_analysis", "extreme")
def d_h_json_columnar(rng):
    rows = sales(rng, 30)
    q, e = _base_question(rng, rows)
    cols = {c: [r[c] for r in rows] for c in SALES_COLUMNS}
    return DF(f"The file sales.json maps each column name ({','.join(SALES_COLUMNS)}) to a list of values. {q}", e,
              {"sales.json": json.dumps(cols)})


@h("data_analysis", "extreme")
def d_h_json_argmax(rng):
    rows = sales(rng, 50)
    qe = _revenue_region(rng, rows)
    if qe is None:
        return d_h_json_argmax(rng)
    return DF(f"The file sales.json contains a list of orders with fields {','.join(SALES_COLUMNS)}. {qe[0]}", qe[1],
              {"sales.json": json.dumps(rows, indent=1)}, check="exact_str")


@h("data_analysis", "extreme")
def d_h_tsv(rng):
    rows = sales(rng, rng.choice([20, 100]))
    q, e = _base_question(rng, rows)
    text = "\t".join(SALES_COLUMNS) + "\n" + "".join("\t".join(str(r[c]) for c in SALES_COLUMNS) + "\n" for r in rows)
    return DF(f"The file sales.tsv (tab-separated) has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows). {q}", e, {"sales.tsv": text})


def _xlsx(rows, sheet="Sales"):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(list(SALES_COLUMNS))
    for r in rows:
        ws.append([r[c] for c in SALES_COLUMNS])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@h("data_analysis", "extreme")
def d_h_xlsx(rng):
    rows = sales(rng, rng.choice([20, 50]))
    q, e = _base_question(rng, rows)
    return DF(f"The file sales.xlsx has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows). {q}", e, {"sales.xlsx": _xlsx(rows)})


@h("data_analysis", "extreme")
def d_h_xlsx_argmax(rng):
    rows = sales(rng, 50)
    qe = _revenue_region(rng, rows)
    if qe is None:
        return d_h_xlsx_argmax(rng)
    return DF(f"The spreadsheet sales.xlsx (sheet 'Sales') lists orders with columns {','.join(SALES_COLUMNS)}. {qe[0]}", qe[1],
              {"sales.xlsx": _xlsx(rows)}, check="exact_str")


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


@h("data_analysis", "hard")
def d_h_csv_period_words(rng):
    """csv как в примерах, период другими словами (жанр «даты»)."""
    rows = sales(rng)
    p = rng.choice(sg.PRODUCTS)
    lo, hi, label = rng.choice([(1, 6, "in the first half of the year"), (7, 12, "in the second half of the year"),
                                (6, 8, "in the summer months (June to August)"), (12, 12, "in December"),
                                (3, 5, "between March and May (inclusive)"), (1, 3, "during the first three months")])
    e = sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi)
    return DF(f"The file sales.csv has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows). "
              f"How many units of product '{p}' were sold {label}?", e, {"sales.csv": sg.csv_text(rows)})


@h("data_analysis", "hard")
def d_h_csv_month_range_dash(rng):
    rows = sales(rng)
    p, lo, hi = _pm(rng)
    e = sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi)
    q = rng.choice([f"How many units of product '{p}' were sold in months {lo}-{hi}?",
                    f"How many units of '{p}' were sold from {MONTHS[lo - 1][:3]} to {MONTHS[hi - 1][:3]} (inclusive)?",
                    f"Total units of {p} for months {lo} to {hi} inclusive?"])
    return DF(f"The file sales.csv has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows). {q}", e,
              {"sales.csv": sg.csv_text(rows)})


@h("data_analysis", "hard")
def d_h_csv_percent(rng):
    rows = sales(rng)
    p = rng.choice(sg.PRODUCTS)
    tot = sum(r["units"] for r in rows)
    part = sum(r["units"] for r in rows if r["product"] == p)
    e = round(100 * part / tot) if tot else 0
    if abs(100 * part / tot - e) > 0.45:   # спорное округление не берём
        return d_h_csv_percent(random.Random(rng.random()))
    return DF(f"The file sales.csv has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows). What percentage of all units "
              f"sold were product '{p}'? Round to the nearest integer.", e, {"sales.csv": sg.csv_text(rows)})


@h("data_analysis", "extreme", "inf")
def d_h_i_json_missing_field(rng):
    rows = sales(rng, 30)
    return DF(f"The file sales.json contains a list of orders with fields {','.join(SALES_COLUMNS)}. "
              "What is the total discount given across all orders?", None, {"sales.json": json.dumps(rows)})


@h("data_analysis", "extreme", "inf")
def d_h_i_xlsx_missing_file(rng):
    rows = sales(rng, 20)
    q, _ = _base_question(rng, rows)
    return DF(f"The file sales.xlsx has columns {','.join(SALES_COLUMNS)}. {q}", None,
              {"sales.csv": sg.csv_text(rows)})


# =========================================================================== SEARCH: связь двух людей
def SR(q, expected, people, comps):
    return dict(question=KB_INTRO + q, expected=None if expected is None else str(expected), files=kb_files(people, comps),
                check="exact_str")


def _pair_mentor(rng, people, positive):
    if positive:
        b = rng.choice(list(people))
        return people[b]["Mentor"], b
    while True:
        a, b = rng.sample(list(people), 2)
        if people[b]["Mentor"] != a:
            return a, b


@h("search", "extreme")
def s_h_is_mentor_yesno(rng):
    people, comps = build_kb(rng)
    pos = rng.random() < 0.5
    a, b = _pair_mentor(rng, people, pos)
    q = rng.choice([f"Is {a} the mentor of {b}? Answer yes or no.", f"Does {a} mentor {b}? Answer yes or no.",
                    f"Was {b} mentored by {a}? Answer yes or no."])
    return SR(q, "yes" if pos else "no", people, comps)


@h("search", "extreme")
def s_h_same_mentor_yesno(rng):
    people, comps = build_kb(rng)
    a, b = rng.sample(list(people), 2)
    if rng.random() < 0.5:
        by = {}
        for p, f in people.items():
            by.setdefault(f["Mentor"], []).append(p)
        groups = [g for g in by.values() if len(g) >= 2]
        if groups:
            a, b = rng.sample(rng.choice(groups), 2)
    same = people[a]["Mentor"] == people[b]["Mentor"]
    return SR(f"Do {a} and {b} have the same mentor? Answer yes or no.", "yes" if same else "no", people, comps)


@h("search", "extreme")
def s_h_colleagues_yesno(rng):
    people, comps = build_kb(rng)
    a, b = rng.sample(list(people), 2)
    same = people[a]["Employer"] == people[b]["Employer"]
    return SR(rng.choice([f"Are {a} and {b} colleagues (do they work at the same company)? Answer yes or no.",
                          f"Do {a} and {b} share the same employer? Answer yes or no."]), "yes" if same else "no", people, comps)


@h("search", "extreme")
def s_h_mentor_of_mentor(rng):
    people, comps = build_kb(rng)
    p = rng.choice(list(people))
    return SR(rng.choice([f"Who is the mentor of {p}'s mentor?", f"Who mentors the mentor of {p}?"]),
              people[people[p]["Mentor"]]["Mentor"], people, comps)


@h("search", "extreme")
def s_h_mentor_employer(rng):
    people, comps = build_kb(rng)
    p = rng.choice(list(people))
    return SR(rng.choice([f"Which company does the mentor of {p} work for?", f"Where does {p}'s mentor work?"]),
              people[people[p]["Mentor"]]["Employer"], people, comps)


@h("search", "extreme")
def s_h_mentor_born(rng):
    people, comps = build_kb(rng)
    p = rng.choice(list(people))
    return SR(f"In which year was the mentor of {p} born?", people[people[p]["Mentor"]]["Born"], people, comps)


@h("search", "extreme")
def s_h_mentor_same_company_yesno(rng):
    people, comps = build_kb(rng)
    p = rng.choice(list(people))
    same = people[p]["Employer"] == people[people[p]["Mentor"]]["Employer"]
    return SR(f"Does {p} work at the same company as their mentor? Answer yes or no.", "yes" if same else "no", people, comps)


@h("search", "extreme")
def s_h_older_of_pair(rng):
    people, comps = build_kb(rng)
    while True:
        a, b = rng.sample(list(people), 2)
        if people[a]["Born"] != people[b]["Born"]:
            break
    older = a if people[a]["Born"] < people[b]["Born"] else b
    return SR(rng.choice([f"Who was born earlier: {a} or {b}?", f"Which of {a} and {b} is older?"]), older, people, comps)


@h("search", "extreme")
def s_h_three_hop(rng):
    people, comps = build_kb(rng)
    p = rng.choice(list(people))
    mentor = people[p]["Mentor"]
    if rng.random() < 0.5:
        return SR(f"In which city is the headquarters of the company where {p}'s mentor works?",
                  comps[people[mentor]["Employer"]]["Headquarters"], people, comps)
    return SR(f"Who is the mentor of the mentor of {p}'s mentor?", people[people[mentor]["Mentor"]]["Mentor"], people, comps)


@h("search", "extreme")
def s_h_mentor_older_yesno(rng):
    people, comps = build_kb(rng)
    p = rng.choice(list(people))
    m_ = people[p]["Mentor"]
    if people[m_]["Born"] == people[p]["Born"]:
        return s_h_mentor_older_yesno(random.Random(rng.random()))
    return SR(f"Is {p}'s mentor older than {p}? Answer yes or no.",
              "yes" if people[m_]["Born"] < people[p]["Born"] else "no", people, comps)


@h("search", "extreme", "amb")
def s_h_relationship(rng):
    """Свободная формулировка связи: выгоднее не отвечать, чем угадывать форму ответа."""
    people, comps = build_kb(rng)
    a, b = _pair_mentor(rng, people, True)
    return SR(f"What is the relationship between {a} and {b}?", "mentor", people, comps)


@h("search", "extreme", "inf")
def s_h_i_mentor_unknown(rng):
    people, comps = build_kb(rng)
    a = rng.choice(list(people))
    ghost = next(f"{f} {l}" for f in sg.FIRST for l in sg.LAST if f"{f} {l}" not in people)
    return SR(f"Is {a} the mentor of {ghost}? Answer yes or no.", None, people, comps)


# =========================================================================== сборка
def generate(out: Path, per: int, seed: int, only: set, match: str) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    meta = {"seed": seed, "per": per, "tasks": {}}
    index = 0
    for entry in H_REG:
        if only and entry["type"] not in only:
            continue
        if match and match not in entry["name"]:
            continue
        for k in range(per):
            rng = random.Random(f"h{seed}:{entry['name']}:{k}")
            spec = entry["fn"](rng)
            kind = spec.pop("kind", entry["kind"])
            task_type = entry["type"]
            task_id = f"{task_type}_{index:04d}"
            index += 1
            instruction = spec.get("instruction") or (spec["question"] + TAILS.get(task_type, TAIL_ANSWER))
            check = spec.get("check") or {"math": "exact_int", "search": "exact_str", "data_analysis": "exact_int",
                                          "coding": "unittest"}[task_type]
            task_dir = out / task_id
            task_dir.mkdir()
            reward = rng.randint(1, 5)
            task = {"task_id": task_id, "task_type": task_type, "instruction": instruction, "reward": reward,
                    "expected": spec["expected"], "check": check}
            (task_dir / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if task_type != "math":
                (task_dir / "workspace").mkdir()
            for rel, data in spec.get("files", {}).items():
                path = task_dir / "workspace" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(data, bytes):
                    path.write_bytes(data)
                else:
                    path.write_text(data, encoding="utf-8", newline="")
            meta["tasks"][task_id] = {
                "type": task_type, "feasible": spec["expected"] is not None, "variant": entry["name"],
                "level": entry["level"], "kind": kind, "expected": spec["expected"], "check": check, "reward": reward,
                "tests": spec.get("tests"), "fixed": spec.get("fixed"), "instruction": instruction,
                **({"h_verify": spec["h_verify"]} if "h_verify" in spec else {}),
            }
    meta_path = out.with_name(out.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def recheck(result_path: Path, meta_path: Path) -> int:
    """Пересчитать строки coding-задач с h_verify по рабочей копии trial.py."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))["tasks"]
    tasks_root = Path(result["work_dir"]) / "tasks"
    changed = 0
    for row in result["rows"]:
        info = meta.get(row["task_id"], {})
        hv = info.get("h_verify")
        if not hv or not info.get("feasible") or row["answer"] is None or row.get("where") != "task_dir":
            continue
        ok, why = row["answer"].strip().lower() == hv["marker"].strip().lower(), ""
        if not ok:
            why = f"answer is not {hv['marker']}"
        else:
            ws = tasks_root / row["task_id"] / "workspace"
            with tempfile.TemporaryDirectory(prefix="agH-rc-") as tmp:
                copy = Path(tmp) / "ws"
                shutil.copytree(ws, copy, ignore=shutil.ignore_patterns("__pycache__", "answer.txt"))
                for rel, text in (info.get("tests") or {}).items():
                    (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                    (copy / rel).write_text(text, encoding="utf-8")
                cmd = [sys.executable if a == PY else a for a in hv["argv"]]
                try:
                    proc = subprocess.run(cmd, cwd=copy, capture_output=True, text=True, timeout=60)
                    ok = proc.returncode == 0
                    why = "" if ok else ((proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""])[0][:120]
                except subprocess.TimeoutExpired:
                    ok, why = False, "tests timed out"
        if ok != row["correct"] or why != row["why"]:
            changed += 1
        row["correct"], row["why"] = ok, why
    result["correct"] = sum(1 for r in result["rows"] if r["feasible"] and r["correct"])
    result["wrong"] = [r for r in result["rows"] if r["feasible"] and r["answer"] is not None and not r["correct"]]
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rechecked: {changed} rows changed; correct {result['correct']}/{result['feasible']}, wrong {len(result['wrong'])}")
    for r in result["wrong"][:30]:
        print(f"   WRONG {r['task_id']} [{r['variant']}] got={str(r['answer'])[:30]!r} {r['why']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--match", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--recheck", default="", help="RES.json от trial.py: пересчитать coding-строки (нужен --meta)")
    ap.add_argument("--meta", default="")
    args = ap.parse_args()
    if args.list:
        for e in H_REG:
            print(f"{e['type']:<14}{e['level']:<8}{e['kind']:<4}{e['name']}")
        print(f"total: {len(H_REG)} hypotheses")
        return 0
    if args.recheck:
        if not args.meta:
            ap.error("--recheck needs --meta")
        return recheck(Path(args.recheck), Path(args.meta))
    if not args.out:
        ap.error("--out is required")
    only = {t.strip() for t in args.only.split(",") if t.strip()}
    meta = generate(Path(args.out).resolve(), args.per, args.seed, only, args.match)
    by: dict[str, int] = {}
    for info in meta["tasks"].values():
        key = f"{info['type']}/{info['kind']}"
        by[key] = by.get(key, 0) + 1
    print(f"generated {len(meta['tasks'])} tasks from {len({t['variant'] for t in meta['tasks'].values()})} hypotheses: "
          + json.dumps(dict(sorted(by.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
