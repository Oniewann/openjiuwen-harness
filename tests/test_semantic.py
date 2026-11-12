"""Смысловой ремонт (эталоны), разбор новых форматов базы знаний и таблиц, вид ответа."""
import tempfile
import unittest
from pathlib import Path

from harness.skills import kb as K
from harness.skills import tables as T
from harness.skills.mathword import answer_math_question
from harness.skills.semantic import behavior_diff, known_function, reference_repair
from harness.skills.testrun import find_test_command, run_tests

TEST_HEAD = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
TEST_TAIL = "\n\nif __name__ == '__main__':\n    unittest.main()\n"


def _ws(tmp: str, lib: str, tests: list[tuple[str, str]]) -> Path:
    ws = Path(tmp)
    (ws / "mylib.py").write_text(lib, encoding="utf-8")
    body = "\n".join(f"    def test_{name}(self):\n        {line}\n" for name, line in tests)
    (ws / "test_mylib.py").write_text(TEST_HEAD + body + TEST_TAIL, encoding="utf-8")
    return ws


class ReferenceRepair(unittest.TestCase):
    def _repair(self, lib: str, tests: list[tuple[str, str]]):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp, lib, tests)
            argv = find_test_command("Run `python3 test_mylib.py`", ws)
            base = run_tests(ws, argv, 20)
            result = reference_repair(ws, argv, base)
            for rel, text in result.texts.items():
                (ws / rel).write_text(text, encoding="utf-8")
            return result, run_tests(ws, argv, 20), (ws / "mylib.py").read_text(encoding="utf-8")

    def test_known_names(self):
        self.assertEqual(known_function("total"), "total")
        self.assertEqual(known_function("find_max"), "maximum")
        self.assertIsNone(known_function("frobnicate"))

    def test_overfit_trap_fib(self):
        # «return b» чинится и подгонкой range(n) -> range(1, n); эталон даёт верную функцию и для fib(0)
        lib = "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return b\n"
        result, report, text = self._repair(lib, [("fib", "self.assertEqual(m.fib(10), 55)")])
        self.assertTrue(report.ok)
        self.assertEqual(result.replaced, ["fib"])
        ns: dict = {}
        exec(text, ns)
        self.assertEqual([ns["fib"](i) for i in range(4)], [0, 1, 1, 2])

    def test_variant_closest_to_source(self):
        # average с целочисленным делением остаётся целочисленным: эталон выбирается по близости к исходнику
        lib = "def average(xs):\n    return sum(xs) // (len(xs) + 1)\n"
        _, report, text = self._repair(lib, [("average", "self.assertEqual(m.average([2, 4, 6]), 4)")])
        self.assertTrue(report.ok)
        self.assertIn("//", text)

    def test_passing_function_untouched(self):
        lib = ("def total(xs):\n    return sum(xs) + 0\n\n\n"
               "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n")
        result, report, text = self._repair(lib, [("total", "self.assertEqual(m.total([1, 2, 3]), 6)"),
                                                  ("maximum", "self.assertEqual(m.maximum([3, 9, 2]), 9)")])
        self.assertTrue(report.ok)
        self.assertEqual(result.replaced, ["maximum"])
        self.assertIn("sum(xs) + 0", text)

    def test_syntax_error_and_missing_function(self):
        lib = "def total(xs)\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
        _, report, _ = self._repair(lib, [("total", "self.assertEqual(m.total([1, 2, 3]), 6)"),
                                          ("is_even", "self.assertTrue(m.is_even(4))")])
        self.assertTrue(report.ok)

    def test_unknown_function_left_alone(self):
        lib = "def frobnicate(xs):\n    return len(xs) - 1\n"
        result, report, _ = self._repair(lib, [("frobnicate", "self.assertEqual(m.frobnicate([1, 2]), 2)")])
        self.assertFalse(report.ok)
        self.assertEqual(result.texts, {})

    def test_behavior_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp, "def f1(xs):\n    return xs[0]\n", [("f1", "self.assertEqual(m.f1([3, 9, 2]), 3)")])
            a = {"mylib.py": "def f1(xs):\n    return xs[0]\n"}
            self.assertEqual(behavior_diff(ws, a, {"mylib.py": "def f1(xs):\n    return list(xs)[0]\n"}), [])
            self.assertEqual(behavior_diff(ws, a, {"mylib.py": "def f1(xs):\n    return min(xs + [3])\n"}), ["f1"])


class KbFormats(unittest.TestCase):
    def _kb(self, files: dict[str, str]) -> K.KB:
        tmp = tempfile.mkdtemp()
        for name, text in files.items():
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        return K.load_kb(Path(tmp))

    def test_bold_keys(self):
        kb = self._kb({"person_a_b.md": "# Ann Bee\n\n**Born:** 1983\n**Employer**: Granite AI\n",
                       "company_granite_ai.md": "# Granite AI\n\n**Headquarters:** Osaka\n"})
        self.assertEqual(K.answer_path_question(kb, "In which year was Ann Bee born?").answer, "1983")
        self.assertEqual(K.answer_path_question(kb, "In which city is the headquarters of Ann Bee's employer?").answer,
                         "Osaka")

    def test_json_and_untitled(self):
        kb = self._kb({"person_ann_bee.json": '{"name": "Ann Bee", "Born": 1983, "Employer": "Granite AI"}',
                       "company_granite_ai.md": "Headquarters: Osaka\nFounded: 1990\n"})
        self.assertEqual(K.answer_path_question(kb, "In which city is the headquarters of Ann Bee's employer?").answer,
                         "Osaka")

    def test_full_date_and_no_value(self):
        kb = self._kb({"a.md": "# Ann Bee\n\nBorn: 1983-04-12\n", "b.md": "# Bob Cee\n\nBorn: unknown\n"})
        self.assertEqual(K.answer_path_question(kb, "In which year was Ann Bee born?").answer, "1983")
        self.assertEqual(K.answer_path_question(kb, "In which year was Bob Cee born?").status, "broken")

    def test_canonical_names(self):
        kb = self._kb({"a.md": "# Ann Bee\n\nBorn: 1983\n", "b.md": "# Bob Cee\n\nBorn: 1990\n"})
        self.assertEqual(K.canonicalize_question(kb, "In which year was Bee, Ann born?"), "In which year was Ann Bee born?")
        self.assertEqual(K.canonicalize_question(kb, "In which year was ann bee born?"), "In which year was Ann Bee born?")


class TableExtras(unittest.TestCase):
    def _table(self, name: str, text: str) -> T.Table:
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / name
        path.write_text(text, encoding="utf-8")
        return T.load_table(path)

    def test_two_values_mean_either(self):
        t = self._table("sales.csv", "product,month,units\nwidget,1,5\ngadget,1,7\ngizmo,1,9\n")
        res = T.answer_table_question(t, "How many units of products 'widget' and 'gadget' were sold in months 1 through 2 inclusive?")
        self.assertEqual((res.status, res.answer), ("ok", "12"))

    def test_month_from_date(self):
        t = self._table("sales.csv", "product,date,units\nwidget,2024-06-03,5\nwidget,2024-09-30,7\nwidget,2024-10-01,9\n")
        res = T.answer_table_question(t, "How many units of product 'widget' were sold in months 6 through 9 inclusive?")
        self.assertEqual((res.status, res.answer), ("ok", "12"))

    def test_json_table(self):
        t = self._table("sales.json", '[{"product": "widget", "month": 1, "units": 5}, {"product": "widget", "month": 3, "units": 7}]')
        self.assertEqual(t.columns, ["product", "month", "units"])
        self.assertEqual(sum(r["units"] for r in t.rows), 12)

    def test_count_entities(self):
        t = self._table("employees.csv", "employee_id,department,salary\n1,sales,10\n2,sales,20\n3,support,30\n")
        res = T.answer_table_question(t, "How many employees are in department 'sales'?")
        self.assertEqual((res.status, res.answer), ("ok", "2"))


class MathPhrasing(unittest.TestCase):
    def test_new_phrasings(self):
        cases = {
            "A warehouse starts with 100 boxes. It ships 36 boxes per day for 1 day. It receives 1 delivery of 87 boxes. "
            "How many boxes does it have at the end?": "151",
            "A warehouse starts with 100 boxes. Next, it ships 6 boxes per week for 3 weeks. Finally, it receives 2 "
            "deliveries of 10 boxes each. What is the final number of boxes?": "102",
            "A warehouse starts with 100 boxes. Then the stock is tripled by merging with two sister warehouses. Then the "
            "stock is split equally into 7 warehouses and it keeps one share (rounded up). How many boxes are in the "
            "warehouse at the end?": "43",
            "Problem: A warehouse starts with 100 boxes. It ships 6 boxes per day for 3 days. How many boxes will it have "
            "at the end? Save only the number to answer.txt.": "82",
        }
        for question, expected in cases.items():
            self.assertEqual(answer_math_question(question).answer, expected, question)

    def test_unit_must_match(self):
        res = answer_math_question("A warehouse starts with 100 boxes. It ships 6 boxes per truck for 3 days. "
                                   "How many boxes does it have at the end?")
        self.assertEqual(res.status, "unparsed")


if __name__ == "__main__":
    unittest.main()
