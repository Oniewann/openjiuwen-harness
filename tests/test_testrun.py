"""Тесты harness.skills.testrun: запуск тестов, разбор вывода, противоречия в тестах."""
from __future__ import annotations

import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from harness import sandbox
from harness.skills import testrun
from harness.skills.testrun import (
    TestReport, contradictory_asserts, find_test_command, run_tests, snapshot, source_files,
    summarize_failures, test_files,
)

TASKS = Path("/Users/roman/Desktop/huawei/task-dir")

TEST_HEADER = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
TEST_FOOTER = "\n\nif __name__ == '__main__':\n    unittest.main()\n"


def make_ws(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="testrun_"))
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return root


def tests_text(body: str) -> str:
    return TEST_HEADER + textwrap.indent(textwrap.dedent(body).strip("\n"), "    ") + "\n" + TEST_FOOTER


def copy_task(name: str) -> Path:
    src = TASKS / name / "workspace"
    if not src.is_dir():
        raise unittest.SkipTest(f"нет примера {src}")
    dst = Path(tempfile.mkdtemp(prefix="testrun_")) / "ws"
    shutil.copytree(src, dst)
    return dst


def has_pycache(root: Path) -> bool:
    return any(p.name == "__pycache__" for p in root.rglob("*"))


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.ws = make_ws({
            "mylib.py": "x = 1\n",
            "test_mylib.py": "pass\n",
            "pkg/helper.py": "y = 2\n",
            "pkg/helper_test.py": "pass\n",
            ".hidden/test_secret.py": "pass\n",
            "__pycache__/test_cached.py": "pass\n",
        })
        self.addCleanup(shutil.rmtree, self.ws, True)

    def test_split(self):
        rel = lambda ps: sorted(str(p.relative_to(self.ws)) for p in ps)
        self.assertEqual(rel(test_files(self.ws)), ["pkg/helper_test.py", "test_mylib.py"])
        self.assertEqual(rel(source_files(self.ws)), ["mylib.py", "pkg/helper.py"])

    def test_snapshot(self):
        files = test_files(self.ws)
        before = snapshot(files)
        self.assertEqual(len(before), 2)
        self.assertTrue(all(len(h) == 64 for h in before.values()))
        self.assertEqual(before, snapshot(files))
        (self.ws / "test_mylib.py").write_text("pass  # changed\n")
        self.assertNotEqual(before, snapshot(files))


class FindCommandTest(unittest.TestCase):
    def test_from_instruction(self):
        ws = Path(".")
        argv = find_test_command("Run `python3 test_mylib.py` in the current directory; fix.", ws)
        self.assertEqual(argv, sandbox.python_argv("test_mylib.py"))
        argv = find_test_command("Use `python -m unittest -v` please", ws)
        self.assertEqual(argv, sandbox.python_argv("-m", "unittest", "-v"))
        argv = find_test_command("Run `pytest -q tests`", ws)
        self.assertEqual(argv, sandbox.python_argv("-m", "pytest", "-q", "tests"))

    def test_fallbacks(self):
        ws = make_ws({"mylib.py": "", "test_a.py": ""})
        self.addCleanup(shutil.rmtree, ws, True)
        self.assertEqual(find_test_command("Fix `mylib.py`", ws), sandbox.python_argv("test_a.py"))
        (ws / "test_b.py").write_text("")
        self.assertEqual(find_test_command("", ws), sandbox.python_argv("-m", "unittest", "discover", "-v"))


class RunTestsTest(unittest.TestCase):
    def test_sample_0003(self):
        ws = copy_task("coding_0003")
        self.addCleanup(shutil.rmtree, ws.parent, True)
        rep = run_tests(ws)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.total, 6)
        self.assertEqual(sorted(rep.failed), ["test_reverse_words", "test_total"])
        self.assertEqual(rep.details["test_total"],
                         "self.assertEqual(m.total([1, 2, 3]), 6) -> AssertionError: -6 != 6")
        self.assertEqual(rep.error, "")
        self.assertNotIn(str(ws), rep.raw)
        text = summarize_failures(rep)
        self.assertIn("test_total: self.assertEqual(m.total([1, 2, 3]), 6) -> AssertionError: -6 != 6", text)
        self.assertEqual(len(text.splitlines()), 2)
        self.assertFalse(has_pycache(ws))

    def test_verbose_discover_same_result(self):
        ws = copy_task("coding_0007")
        self.addCleanup(shutil.rmtree, ws.parent, True)
        rep = run_tests(ws, sandbox.python_argv("-m", "unittest", "discover", "-v"))
        self.assertEqual(rep.total, 4)
        self.assertEqual(sorted(rep.failed), ["test_count_even", "test_maximum"])
        self.assertIn("AssertionError: 2 != 9", rep.details["test_maximum"])
        self.assertFalse(has_pycache(ws))

    def test_passing(self):
        ws = make_ws({"mylib.py": "def f(x):\n    return x\n",
                      "test_mylib.py": tests_text("def test_f(self):\n    self.assertEqual(m.f(1), 1)\n")})
        self.addCleanup(shutil.rmtree, ws, True)
        rep = run_tests(ws)
        self.assertTrue(rep.ok)
        self.assertEqual((rep.total, rep.failed, rep.error), (1, [], ""))
        self.assertEqual(summarize_failures(rep), "All tests passed.")

    def test_error_inside_module(self):
        ws = make_ws({
            "mylib.py": "def average(xs):\n    n = len(xs)\n    return sum(xs) / n\n",
            "test_mylib.py": tests_text("""
                def test_empty(self):
                    self.assertEqual(m.average([]), 0)

                def test_sub(self):
                    for i in range(2):
                        with self.subTest(i=i):
                            self.assertEqual(m.average([i]), 5)
            """),
        })
        self.addCleanup(shutil.rmtree, ws, True)
        rep = run_tests(ws)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.failed, ["test_empty", "test_sub"])
        self.assertEqual(rep.details["test_empty"],
                         "self.assertEqual(m.average([]), 0) -> mylib.py:3 in average: "
                         "ZeroDivisionError: division by zero")
        self.assertEqual(rep.total, 2)

    def test_syntax_error_in_module(self):
        ws = make_ws({"mylib.py": "def f(x):\n    return (x\n",
                      "test_mylib.py": tests_text("def test_f(self):\n    self.assertEqual(m.f(1), 1)\n")})
        self.addCleanup(shutil.rmtree, ws, True)
        for argv in (None, sandbox.python_argv("-m", "unittest", "discover", "-v")):
            rep = run_tests(ws, argv)
            self.assertFalse(rep.ok)
            self.assertIn("SyntaxError", rep.error)
            self.assertIn("mylib.py:2", rep.error)
            self.assertNotIn("/", rep.error.split("(")[-1])
            self.assertEqual(rep.failed, [])
            self.assertTrue(summarize_failures(rep).startswith("Tests did not start: SyntaxError"))
        self.assertFalse(has_pycache(ws))

    def test_missing_file(self):
        ws = make_ws({"mylib.py": ""})
        self.addCleanup(shutil.rmtree, ws, True)
        rep = run_tests(ws, sandbox.python_argv("test_nothing.py"))
        self.assertFalse(rep.ok)
        self.assertTrue(rep.error)

    def test_timeout(self):
        ws = make_ws({"mylib.py": "def f():\n    while True:\n        pass\n",
                      "test_mylib.py": tests_text("def test_f(self):\n    m.f()\n")})
        self.addCleanup(shutil.rmtree, ws, True)
        rep = run_tests(ws, timeout=1.5)
        self.assertTrue(rep.timed_out)
        self.assertFalse(rep.ok)
        self.assertLess(rep.seconds, 8)
        self.assertIn("timed out", summarize_failures(rep))


class ParseTest(unittest.TestCase):
    OLD_FORMAT = textwrap.dedent("""\
        .F.E
        ======================================================================
        ERROR: test_avg (__main__.T)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "/abs/dir/test_mylib.py", line 14, in test_avg
            self.assertEqual(m.average([]), 0)
          File "/abs/dir/mylib.py", line 12, in average
            return sum(xs) / len(xs)
        ZeroDivisionError: division by zero

        ======================================================================
        FAIL: test_total (__main__.T)
        Docstring of the test.
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "test_mylib.py", line 11, in test_total
            self.assertEqual(m.total([1, 2, 3]), 6)
        AssertionError: -6 != 6

        ----------------------------------------------------------------------
        Ran 4 tests in 0.001s

        FAILED (failures=1, errors=1)
        """)

    def test_old_header(self):
        rep = testrun._parse(self.OLD_FORMAT, 1, False, Path("/abs/dir"))
        self.assertEqual(rep.total, 4)
        self.assertEqual(rep.failed, ["test_avg", "test_total"])
        self.assertEqual(rep.details["test_avg"],
                         "self.assertEqual(m.average([]), 0) -> mylib.py:12 in average: "
                         "ZeroDivisionError: division by zero")
        self.assertEqual(rep.details["test_total"],
                         "self.assertEqual(m.total([1, 2, 3]), 6) -> AssertionError: -6 != 6")
        self.assertFalse(rep.ok)

    def test_ok_requires_returncode(self):
        rep = testrun._parse("....\n" + "-" * 70 + "\nRan 4 tests in 0.0s\n\nOK\n", 0, False, Path("."))
        self.assertTrue(rep.ok)
        rep = testrun._parse("....\n" + "-" * 70 + "\nRan 4 tests in 0.0s\n\nOK\n", 1, False, Path("."))
        self.assertFalse(rep.ok)

    def test_pytest(self):
        out = textwrap.dedent("""\
            ..F.
            =========================== short test summary info ============================
            FAILED test_mylib.py::T::test_total - AssertionError: -6 != 6
            FAILED test_mylib.py::test_free - assert 1 == 2
            ========================= 2 failed, 2 passed in 0.05s ==========================
            """)
        rep = testrun._parse(out, 1, False, Path("."))
        self.assertEqual(rep.total, 4)
        self.assertEqual(rep.failed, ["test_total", "test_free"])
        self.assertEqual(rep.details["test_total"], "AssertionError: -6 != 6")
        self.assertEqual(rep.error, "")

    def test_details_limit(self):
        long_msg = "AssertionError: " + "x" * 1000
        out = self.OLD_FORMAT.replace("AssertionError: -6 != 6", long_msg)
        rep = testrun._parse(out, 1, False, Path("."))
        self.assertTrue(all(len(v) <= 300 for v in rep.details.values()))


class SummarizeTest(unittest.TestCase):
    def test_truncation(self):
        names = [f"test_{i}" for i in range(50)]
        rep = TestReport(ok=False, total=50, failed=names,
                         details={n: "self.assertEqual(m.f(1), 2) -> AssertionError: 3 != 2" for n in names})
        text = summarize_failures(rep, max_chars=400)
        self.assertLessEqual(len(text), 400)
        self.assertRegex(text.splitlines()[-1], r"^\.\.\. \d+ more failing tests not shown$")
        shown = len(text.splitlines()) - 1
        self.assertIn(f"{50 - shown} more", text)


class ContradictionTest(unittest.TestCase):
    def ws_with(self, body: str) -> Path:
        ws = make_ws({"mylib.py": "def f(x):\n    return x\n", "test_mylib.py": tests_text(body)})
        self.addCleanup(shutil.rmtree, ws, True)
        return ws

    def test_samples_are_clean(self):
        for name in ("coding_0003", "coding_0007"):
            ws = copy_task(name)
            self.addCleanup(shutil.rmtree, ws.parent, True)
            self.assertEqual(contradictory_asserts(ws), [])

    def test_equal_different_values(self):
        ws = self.ws_with("""
            def test_a(self):
                self.assertEqual(m.f(2), 4)

            def test_b(self):
                self.assertEqual(5, m.f(2))
        """)
        found = contradictory_asserts(ws)
        self.assertEqual(len(found), 1)
        self.assertIn("m.f(2) == 4", found[0])
        self.assertIn("m.f(2) == 5", found[0])

    def test_true_false_eq_ne_raises(self):
        ws = self.ws_with("""
            def test_a(self):
                self.assertTrue(m.f('x'))

            def test_b(self):
                self.assertFalse(m.f('x'))

            def test_c(self):
                self.assertEqual(m.f([1, 2]), [1, 2])

            def test_d(self):
                self.assertNotEqual(m.f([1, 2]), [1, 2])

            def test_e(self):
                with self.assertRaises(ValueError):
                    m.f(-1)

            def test_f(self):
                self.assertEqual(m.f(-1), 0)

            def test_g(self):
                self.assertRaises(TypeError, m.f, None)

            def test_h(self):
                self.assertIsNone(m.f(None))
        """)
        found = contradictory_asserts(ws)
        joined = "\n".join(found)
        self.assertEqual(len(found), 4, joined)
        self.assertIn("test_a", joined)
        self.assertIn("test_d", joined)
        self.assertIn("m.f(-1) raises", joined)
        self.assertIn("m.f(None) raises", joined)

    def test_no_false_positives(self):
        ws = self.ws_with("""
            def test_same(self):
                self.assertEqual(m.f(2), 2)
                self.assertEqual(2, m.f(2))
                self.assertEqual(m.f(1), 1.0)
                self.assertTrue(m.f(3))
                self.assertNotEqual(m.f(3), 4)

            def test_variables(self):
                x = 2
                self.assertEqual(m.f(x), 2)

            def test_state(self):
                m.reset()
                self.assertEqual(m.f(2), 3)

            def test_counter(self):
                self.assertEqual(m.next_id(), 1)
                self.assertEqual(m.next_id(), 2)

            def test_non_literal(self):
                self.assertEqual(m.f([i for i in range(2)]), 5)
                self.assertEqual(m.f(len('ab')), 7)
        """)
        self.assertEqual(contradictory_asserts(ws), [])

    def test_setup_scopes(self):
        ws = make_ws({"mylib.py": "", "test_mylib.py": textwrap.dedent("""\
            import unittest
            import mylib as m

            class A(unittest.TestCase):
                def setUp(self):
                    m.mode = 'a'
                def test_a(self):
                    self.assertEqual(m.g(1), 1)

            class B(unittest.TestCase):
                def setUp(self):
                    m.mode = 'b'
                def test_b(self):
                    self.assertEqual(m.g(1), 2)
            """)})
        self.addCleanup(shutil.rmtree, ws, True)
        self.assertEqual(contradictory_asserts(ws), [])


if __name__ == "__main__":
    unittest.main()
