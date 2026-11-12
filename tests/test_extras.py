"""Дополнительные тесты: вычисление кода модели, вид ответа, невыполнимые тесты, заплатки, имена эталонов."""
import tempfile
import types
import unittest
from pathlib import Path

from harness.patching import apply_patch
from harness.pot import run_pot
from harness.skills import testrun as TR
from harness.skills.semantic import known_function
from harness.skills.testrun import unsatisfiable_tests
from harness.solvers.base import ExprSolver


class RunPot(unittest.TestCase):
    def _run(self, code: str):
        with tempfile.TemporaryDirectory() as tmp:
            return run_pot(code, "", Path(tmp), Path(tmp) / "scratch", 20)

    def _value(self, code: str) -> str:
        res = self._run(code)
        self.assertTrue(res.ok, res.error)
        return res.value

    def test_print_expression_not_lost(self):
        self.assertEqual(self._value("print(((679 // 2) * 7) + 32)"), "2405")

    def test_pow_mod(self):
        self.assertEqual(self._value("pow(2, 68, 529)"), "464")

    def test_multiline_script(self):
        self.assertEqual(self._value("x = 5\nprint(x * 2)"), "10")

    def test_print_string(self):
        self.assertEqual(self._value("print('Osaka')"), "Osaka")

    def test_list_one_item(self):
        self.assertEqual(self._value("['Ann']"), "Ann")

    def test_list_two_items(self):
        self.assertEqual(self._value("['a', 'b']"), "a, b")

    def test_error_reported(self):
        res = self._run("[1][5]")
        self.assertFalse(res.ok)
        self.assertIn("IndexError", res.error)


class ShapeAnswer(unittest.TestCase):
    def _shape(self, question: str, value: str):
        stub = types.SimpleNamespace(task=types.SimpleNamespace(question=question))
        return ExprSolver.shape_answer(stub, value)

    def test_bool_for_which_question_rejected(self):
        ans, note = self._shape("Which company was founded earlier, Dune Robotics or Iris Optics?", "True")
        self.assertIsNone(ans)
        self.assertTrue(note)

    def test_yes_no_false(self):
        q = "Do Ann Bee and Bob Cee work for the same company? Answer yes or no."
        self.assertEqual(self._shape(q, "False"), ("no", ""))

    def test_yes_no_true(self):
        q = "Do Ann Bee and Bob Cee work for the same company? Answer yes or no."
        self.assertEqual(self._shape(q, "True"), ("yes", ""))

    def test_is_question_true(self):
        self.assertEqual(self._shape("Is Ann older than Bob?", "True"), ("yes", ""))

    def test_who_with_number_rejected(self):
        ans, note = self._shape("Who was born in 1987?", "1987")
        self.assertIsNone(ans)
        self.assertTrue(note)

    def test_who_with_name(self):
        self.assertEqual(self._shape("Who was born in 1987?", "Irina Haddad"), ("Irina Haddad", ""))

    def test_years_older_negative_flipped(self):
        self.assertEqual(self._shape("How many years older is Ann Bee than Bob Cee?", "-29"), ("29", ""))

    def test_years_older_positive_kept(self):
        self.assertEqual(self._shape("How many years older is Ann Bee than Bob Cee?", "29"), ("29", ""))

    def test_year_kept(self):
        self.assertEqual(self._shape("In which year was Ann Bee born?", "1983"), ("1983", ""))

    def test_negative_kept_for_other_questions(self):
        self.assertEqual(self._shape("How many units were sold in total?", "-5"), ("-5", ""))

    def test_preamble_stripped(self):
        q = ("The directory kb/ contains a small knowledge base of markdown files. "
             "Who is the mentor of Ann Bee?")
        self.assertEqual(self._shape(q, "Bob Cee"), ("Bob Cee", ""))


class UnsatisfiableTests(unittest.TestCase):
    def _reasons(self, body: str, lib: str = "") -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            if lib:
                (ws / "mylib.py").write_text(lib, encoding="utf-8")
            (ws / "test_mylib.py").write_text(body, encoding="utf-8")
            return unsatisfiable_tests(ws)

    def _case(self, line: str, lib: str = "") -> list[str]:
        text = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                f"    def test_x(self):\n        {line}\n")
        return self._reasons(text, lib)

    def test_assert_equal_constants(self):
        self.assertTrue(self._case("self.assertEqual(1, 2)"))

    def test_assert_true_false(self):
        self.assertTrue(self._case("self.assertTrue(False)"))

    def test_fail(self):
        self.assertTrue(self._case("self.fail('x')"))

    def test_false_arithmetic(self):
        self.assertTrue(self._case("self.assertEqual(2 + 2, 5)"))

    def test_assert_statement(self):
        self.assertTrue(self._case("assert False"))

    def test_syntax_error(self):
        reasons = self._reasons("import unittest\n\nclass T(unittest.TestCase)\n    pass\n")
        self.assertTrue(reasons)
        self.assertIn("syntax", " ".join(reasons).lower())

    def test_normal_test_ok(self):
        lib = "def total(xs):\n    return sum(xs)\n"
        self.assertEqual(self._case("self.assertEqual(m.total([1, 2, 3]), 6)", lib), [])

    def test_true_constants_ok(self):
        self.assertEqual(self._case("self.assertEqual(1, 1)"), [])
        self.assertEqual(self._case("self.assertTrue(True)"), [])

    def test_conditional_branch_ignored(self):
        text = ("import unittest\n\n\nclass T(unittest.TestCase):\n"
                "    def test_x(self):\n        if False:\n            self.assertEqual(1, 2)\n")
        self.assertEqual(self._reasons(text), [])

    def test_loop_ignored(self):
        text = ("import unittest\n\n\nclass T(unittest.TestCase):\n"
                "    def test_x(self):\n        for _ in range(2):\n            self.assertTrue(False)\n")
        self.assertEqual(self._reasons(text), [])


class ApplyPatch(unittest.TestCase):
    def test_replace_one_function(self):
        original = "def a(x):\n    return x + 1\n\n\ndef b(x):\n    return x * 2\n"
        result = apply_patch(original, "def a(x):\n    return x + 100\n")
        self.assertIsNotNone(result)
        compile(result, "x", "exec")
        self.assertIn("return x + 100", result)
        self.assertNotIn("return x + 1\n", result)
        self.assertIn("def b(x):\n    return x * 2", result)

    def test_broken_original_fixed_function(self):
        original = ("def first(xs):\n    return xs[0]\n\n\n"
                    "def total(xs)\n    return sum(xs\n\n\n"
                    "def last(xs):\n    return xs[-1]\n")
        result = apply_patch(original, "def total(xs):\n    return sum(xs)\n")
        self.assertIsNotNone(result)
        compile(result, "x", "exec")
        self.assertIn("def first(xs):\n    return xs[0]\n", result)
        self.assertIn("def last(xs):\n    return xs[-1]\n", result)
        self.assertIn("return sum(xs)", result)

    def test_broken_original_unknown_function(self):
        original = "def first(xs):\n    return xs[0]\n\n\ndef total(xs)\n    return sum(xs)\n"
        self.assertIsNone(apply_patch(original, "def other(xs):\n    return 1\n"))


class KnownFunction(unittest.TestCase):
    def test_names_with_wrappers(self):
        for name, want in [("compute_total", "total"), ("get_maximum", "maximum"), ("my_average", "average"),
                           ("total_of_list", "total"), ("find_max", "maximum")]:
            with self.subTest(name=name):
                self.assertEqual(known_function(name), want)

    def test_unknown(self):
        self.assertIsNone(known_function("frobnicate"))
        self.assertIsNone(known_function("fib_x"))


BAD_LIB = "def total(xs):\n    return sum(xs) - 1\n\n\ndef maximum(xs):\n    return max(xs)\n"
GOOD_LIB = "def total(xs):\n    return sum(xs)\n\n\ndef maximum(xs):\n    return max(xs)\n"
PLAIN_TESTS = ("import mylib as m\n\n\ndef test_total():\n    assert m.total([1, 2, 3]) == 6\n\n\n"
               "def test_maximum():\n    assert m.maximum([3, 9, 2]) == 9\n")
CMD = "Run `python3 test_mylib.py`"


class FallbackRun(unittest.TestCase):
    def _ws(self, tmp: str, lib: str, tests: str) -> Path:
        ws = Path(tmp)
        (ws / "mylib.py").write_text(lib, encoding="utf-8")
        (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
        return ws

    def test_pytest_style_trap_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp, BAD_LIB, PLAIN_TESTS)
            argv = TR.find_test_command(CMD, ws)
            trap = TR.run_tests(ws, argv, 20)
            self.assertTrue(trap.ok)
            self.assertEqual(trap.total, 0)
            fb = TR.fallback_test_command(ws)
            self.assertIsNotNone(fb)
            rep = TR.run_tests(ws, fb, 20)
            self.assertFalse(rep.ok)
            self.assertEqual(rep.total, 2)
            self.assertEqual(list(rep.failed), ["test_total"])
            (ws / "mylib.py").write_text(GOOD_LIB, encoding="utf-8")
            fixed = TR.run_tests(ws, fb, 20)
            self.assertTrue(fixed.ok)
            self.assertEqual(fixed.total, 2)

    def test_pytest_style_fixed_with_original_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp, GOOD_LIB, PLAIN_TESTS)
            rep = TR.run_tests(ws, TR.find_test_command(CMD, ws), 20)
            self.assertTrue(rep.ok)

    def test_testcase_without_main(self):
        tests = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                 "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n\n"
                 "    def test_maximum(self):\n        self.assertEqual(m.maximum([3, 9, 2]), 9)\n")
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp, BAD_LIB, tests)
            fb = TR.fallback_test_command(ws)
            self.assertIsNotNone(fb)
            rep = TR.run_tests(ws, fb, 20)
            self.assertFalse(rep.ok)
            self.assertEqual(rep.total, 2)
            self.assertIn("test_total", rep.failed)

    def test_function_raising_exception(self):
        tests = "import mylib as m\n\n\ndef test_boom():\n    m.nope()\n"
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp, GOOD_LIB, tests)
            fb = TR.fallback_test_command(ws)
            self.assertIsNotNone(fb)
            rep = TR.run_tests(ws, fb, 20)
            self.assertFalse(rep.ok)
            self.assertIn("test_boom", rep.failed)

    def test_no_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "mylib.py").write_text(GOOD_LIB, encoding="utf-8")
            self.assertIsNone(TR.fallback_test_command(Path(tmp)))

    def test_regular_unittest_needs_no_fallback(self):
        tests = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                 "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n\n\n"
                 "if __name__ == '__main__':\n    unittest.main()\n")
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp, GOOD_LIB, tests)
            rep = TR.run_tests(ws, TR.find_test_command(CMD, ws), 20)
            self.assertTrue(rep.ok)
            self.assertGreater(rep.total, 0)


if __name__ == "__main__":
    unittest.main()
