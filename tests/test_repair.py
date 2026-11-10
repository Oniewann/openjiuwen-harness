"""Тесты harness.skills.repair: ремонт мутациями на примерах задач и на синтетике."""
from __future__ import annotations

import shutil
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from harness.skills.repair import mutation_repair
from harness.skills.testrun import contradictory_asserts, snapshot, test_files

TASKS = Path("/Users/roman/Desktop/huawei/task-dir")

TEST_HEADER = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
TEST_FOOTER = "\n\nif __name__ == '__main__':\n    unittest.main()\n"


def make_ws(lib: str, tests_body: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix="repair_test_"))
    (root / "mylib.py").write_text(textwrap.dedent(lib))
    body = textwrap.indent(textwrap.dedent(tests_body).strip("\n"), "    ")
    (root / "test_mylib.py").write_text(TEST_HEADER + body + "\n" + TEST_FOOTER)
    return root


def copy_task(name: str) -> Path:
    src = TASKS / name / "workspace"
    if not src.is_dir():
        raise unittest.SkipTest(f"нет примера {src}")
    dst = Path(tempfile.mkdtemp(prefix="repair_test_")) / "ws"
    shutil.copytree(src, dst)
    return dst


def has_pycache(root: Path) -> bool:
    return any(p.name == "__pycache__" for p in root.rglob("*"))


class RepairCase(unittest.TestCase):
    def check_invariants(self, ws: Path, tests_before: dict) -> None:
        self.assertEqual(snapshot(test_files(ws)), tests_before, "тесты изменены")
        self.assertFalse(has_pycache(ws), "появился __pycache__")

    def repair(self, ws: Path, **kw):
        tests_before = snapshot(test_files(ws))
        res = mutation_repair(ws, **kw)
        self.check_invariants(ws, tests_before)
        return res


class SampleTasksTest(RepairCase):
    def test_coding_0003(self):
        ws = copy_task("coding_0003")
        self.addCleanup(shutil.rmtree, ws.parent, True)
        original = (ws / "mylib.py").read_text()
        res = self.repair(ws)
        self.assertTrue(res.fixed, res)
        self.assertTrue(res.report.ok)
        self.assertEqual(res.report.total, 6)
        self.assertFalse(res.ambiguous)
        expected = original.replace("sorted(s.split())", "reversed(s.split())").replace("s -= x", "s += x")
        self.assertEqual((ws / "mylib.py").read_text(), expected)
        self.assertEqual(res.patched_files, {"mylib.py": expected})
        self.assertEqual(sorted(res.edits), ["mylib.py:2 'sorted' -> 'reversed'", "mylib.py:8 '-=' -> '+='"])

    def test_coding_0007(self):
        ws = copy_task("coding_0007")
        self.addCleanup(shutil.rmtree, ws.parent, True)
        original = (ws / "mylib.py").read_text()
        res = self.repair(ws)
        self.assertTrue(res.fixed, res)
        self.assertFalse(res.ambiguous)
        expected = original.replace("x < best", "x > best").replace("x % 2 == 1", "x % 2 == 0")
        self.assertEqual((ws / "mylib.py").read_text(), expected)
        self.assertEqual(len(res.edits), 2)

    def test_already_passing(self):
        ws = copy_task("coding_0003")
        self.addCleanup(shutil.rmtree, ws.parent, True)
        self.assertTrue(self.repair(ws).fixed)
        again = self.repair(ws)
        self.assertTrue(again.fixed)
        self.assertEqual((again.edits, again.tried, again.patched_files), ([], 0, {}))


class SyntheticTest(RepairCase):
    def test_slice_off_by_one(self):
        ws = make_ws("""\
            def first_n(xs, n):
                # первые n элементов
                return xs[:n+1]
            """, """
            def test_first_n(self):
                self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

            def test_first_zero(self):
                self.assertEqual(m.first_n([5, 6], 0), [])
            """)
        self.addCleanup(shutil.rmtree, ws, True)
        res = self.repair(ws)
        self.assertTrue(res.fixed, res)
        self.assertEqual((ws / "mylib.py").read_text(),
                         "def first_n(xs, n):\n    # первые n элементов\n    return xs[:n]\n")
        self.assertFalse(res.ambiguous)
        self.assertEqual(res.edits, ["mylib.py:3 'n+1' -> 'n'"])

    def test_min_instead_of_max(self):
        ws = make_ws("""\
            def largest(xs):
                return min(xs)


            def smallest(xs):
                return min(xs)
            """, """
            def test_largest(self):
                self.assertEqual(m.largest([3, 9, 2]), 9)

            def test_smallest(self):
                self.assertEqual(m.smallest([3, 9, 2]), 2)
            """)
        self.addCleanup(shutil.rmtree, ws, True)
        res = self.repair(ws)
        self.assertTrue(res.fixed, res)
        text = (ws / "mylib.py").read_text()
        self.assertIn("def largest(xs):\n    return max(xs)", text)
        self.assertIn("def smallest(xs):\n    return min(xs)", text)
        self.assertEqual(res.edits, ["mylib.py:2 'min' -> 'max'"])

    def test_two_bugs_in_different_functions(self):
        lib = """\
            def area(w, h):
                \"\"\"Площадь прямоугольника.\"\"\"
                return w   +   h   # странное форматирование сохраняется


            def is_adult(age):
                return age > 18


            def greet(name):
                return 'Hello, ' + name.upper()
            """
        ws = make_ws(lib, """
            def test_area(self):
                self.assertEqual(m.area(2, 3), 6)

            def test_area_square(self):
                self.assertEqual(m.area(4, 4), 16)

            def test_adult(self):
                self.assertTrue(m.is_adult(18))
                self.assertTrue(m.is_adult(30))
                self.assertFalse(m.is_adult(17))

            def test_greet(self):
                self.assertEqual(m.greet('bob'), 'Hello, BOB')
            """)
        self.addCleanup(shutil.rmtree, ws, True)
        res = self.repair(ws)
        self.assertTrue(res.fixed, res)
        expected = textwrap.dedent(lib).replace("w   +   h", "w   *   h").replace("age > 18", "age >= 18")
        self.assertEqual((ws / "mylib.py").read_text(), expected)
        self.assertEqual(len(res.edits), 2)

    def test_looping_mutant_does_not_hang(self):
        ws = make_ws("""\
            def sum_down(n):
                s = 0
                while n > 0:
                    s = s - n
                    n -= 1
                return s
            """, """
            def test_sum_down(self):
                self.assertEqual(m.sum_down(4), 10)
            """)
        self.addCleanup(shutil.rmtree, ws, True)
        started = time.monotonic()
        res = self.repair(ws, test_timeout=1.5, time_budget=30)
        self.assertTrue(res.fixed, res)
        self.assertIn("s = s + n", (ws / "mylib.py").read_text())
        self.assertLess(time.monotonic() - started, 20)

    def test_contradictory_tests_restore_file(self):
        lib = "def double(x):\r\n    return x + x  # CRLF и комментарий\r\n\r\n\r\ndef inc(x):\r\n    return x - 1\r\n"
        ws = make_ws("", """
            def test_double(self):
                self.assertEqual(m.double(2), 4)

            def test_double_again(self):
                self.assertEqual(m.double(2), 5)

            def test_inc(self):
                self.assertEqual(m.inc(1), 2)
            """)
        self.addCleanup(shutil.rmtree, ws, True)
        (ws / "mylib.py").write_bytes(lib.encode())
        found = contradictory_asserts(ws)
        self.assertEqual(len(found), 1)
        self.assertIn("m.double(2)", found[0])
        res = self.repair(ws, time_budget=20)
        self.assertFalse(res.fixed)
        self.assertEqual((ws / "mylib.py").read_bytes(), lib.encode())
        self.assertFalse(res.report.ok)
        self.assertGreater(res.tried, 0)

    def test_time_budget(self):
        funcs = "\n\n".join(
            f"def f{i}(x):\n    return (x + {i}) * 2 - (x // 3) + {i} % 2\n" for i in range(15))
        tests = "\n".join(f"def test_f{i}(self):\n    self.assertEqual(m.f{i}(1), -999)\n" for i in range(15))
        ws = make_ws(funcs, tests)
        self.addCleanup(shutil.rmtree, ws, True)
        before = (ws / "mylib.py").read_bytes()
        started = time.monotonic()
        res = self.repair(ws, time_budget=3.0, test_timeout=2.0)
        self.assertLess(time.monotonic() - started, 8)
        self.assertFalse(res.fixed)
        self.assertEqual((ws / "mylib.py").read_bytes(), before)

    def test_syntax_error_not_repaired(self):
        ws = make_ws("def f(x):\n    return (x\n", """
            def test_f(self):
                self.assertEqual(m.f(1), 1)
            """)
        self.addCleanup(shutil.rmtree, ws, True)
        res = self.repair(ws)
        self.assertFalse(res.fixed)
        self.assertIn("SyntaxError", res.report.error)
        self.assertEqual((ws / "mylib.py").read_text(), "def f(x):\n    return (x\n")


if __name__ == "__main__":
    unittest.main()
