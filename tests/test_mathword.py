"""Тесты навыка mathword: примеры организаторов, каждое правило отдельно, отказы."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT.parent / "task-dir"


def _load(name: str):
    # Загружаем модуль по пути файла: навык не должен зависеть от остального пакета.
    spec = importlib.util.spec_from_file_location(f"_standalone_{name}", ROOT / "harness" / "skills" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mw = _load("mathword")
ask = mw.answer_math_question

START = "A warehouse starts with 100 boxes. "
END = " How many boxes does it have at the end?"


def q(*ops: str) -> str:
    return START + " ".join(ops) + END


class ExamplesTest(unittest.TestCase):
    def check_task(self, task_id):
        path = TASKS / task_id / "task.json"
        if not path.exists():
            self.skipTest(f"нет примера {path}")
        task = json.loads(path.read_text())
        res = ask(task["instruction"])
        self.assertEqual(res.status, "ok", res.detail)
        self.assertEqual(res.answer, task["expected"])
        self.assertEqual(str(eval(res.expression)), res.answer)

    def test_math_0000(self):
        self.check_task("math_0000")

    def test_math_0004(self):
        self.check_task("math_0004")

    def test_without_tail(self):
        res = ask("A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. "
                  "It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end?")
        self.assertEqual((res.status, res.answer), ("ok", "1735"))
        self.assertEqual(len(res.steps), 4)


class RulesTest(unittest.TestCase):
    def assertAnswer(self, question, expected, expression=None):
        res = ask(question)
        self.assertEqual(res.status, "ok", res.detail)
        self.assertEqual(res.answer, expected)
        self.assertEqual(str(eval(res.expression)), expected)
        if expression is not None:
            self.assertEqual(res.expression, expression)

    def test_ship_per_day(self):
        self.assertAnswer(q("It ships 7 boxes per day for 3 days."), "79", "100 - 7 * 3")

    def test_ship_per_day_singular(self):
        self.assertAnswer(q("It ships 7 boxes per day for 1 day."), "93")

    def test_receive_deliveries(self):
        self.assertAnswer(q("It receives 2 deliveries of 15 boxes each."), "130", "100 + 2 * 15")

    def test_split_keep_share(self):
        self.assertAnswer(q("Then the stock is split equally into 3 warehouses and it keeps one share (rounded down)."),
                          "33", "100 // 3")

    def test_double_merge(self):
        self.assertAnswer(q("Then the stock is doubled by merging with a sister warehouse."), "200", "100 * 2")

    def test_split_exact_divisible(self):
        self.assertAnswer(q("The stock is split equally into 4 warehouses and it keeps one share."), "25")

    def test_split_exact_not_divisible_refused(self):
        res = ask(q("The stock is split equally into 3 warehouses and it keeps one share."))
        self.assertEqual(res.status, "unparsed")

    def test_receive_per_day(self):
        self.assertAnswer(q("It receives 5 boxes per day for 4 days."), "120")

    def test_receive_n(self):
        self.assertAnswer(q("It receives 25 boxes."), "125", "100 + 25")

    def test_receive_delivery_of(self):
        self.assertAnswer(q("It receives a delivery of 40 boxes."), "140")

    def test_ship_n(self):
        self.assertAnswer(q("It ships 30 boxes."), "70")

    def test_sell_n(self):
        self.assertAnswer(q("It sells 30 boxes."), "70")

    def test_lose_n(self):
        self.assertAnswer(q("Then it loses 1 boxes."), "99")

    def test_double(self):
        self.assertAnswer(q("The stock is doubled."), "200")

    def test_triple(self):
        self.assertAnswer(q("Then the stock is tripled."), "300")

    def test_quadruple(self):
        self.assertAnswer(q("The stock is quadrupled."), "400")

    def test_double_active(self):
        self.assertAnswer(q("It doubles its stock."), "200")

    def test_triple_active(self):
        self.assertAnswer(q("It triples its stock."), "300")

    def test_halve_rounded(self):
        self.assertAnswer("A warehouse starts with 101 boxes. The stock is halved (rounded down)." + END, "50")

    def test_halve_exact(self):
        self.assertAnswer(q("The stock is halved."), "50")

    def test_halve_odd_without_rounding_refused(self):
        res = ask("A warehouse starts with 101 boxes. The stock is halved." + END)
        self.assertEqual(res.status, "unparsed")

    def test_thousands_separator(self):
        self.assertAnswer("A warehouse starts with 1,752 boxes. It receives 1,000 boxes." + END, "2752")

    def test_word_numbers(self):
        self.assertAnswer(q("It receives two deliveries of 10 boxes each."), "120")

    def test_other_items_and_subject(self):
        self.assertAnswer("A store starts with 50 crates. It sells 5 crates per day for 2 days. "
                          "How many crates does it have at the end?", "40")

    def test_chain(self):
        self.assertAnswer(q("It ships 10 boxes per day for 2 days.", "The stock is tripled.",
                            "Then the stock is split equally into 7 warehouses and it keeps one share (rounded down)."),
                          "34", "((100 - 10 * 2) * 3) // 7")

    def test_full_instruction_tail(self):
        self.assertAnswer(q("It receives 1 boxes.") + " Compute it with the bash tool (python3 -c). "
                          "Work only inside the current directory.", "101")


class VariantTest(unittest.TestCase):
    """Формулировки сверх двух шаблонов из примеров (проценты, причины, сумма по глаголу)."""

    def test_percent_rounded(self):
        self.assertEqual(ask(START.replace("100", "1659") + "It sells 50 percent of its boxes (the number sold is "
                             "rounded down). It receives 142 boxes. How many boxes does it have at the end?").answer,
                         "972")

    def test_percent_exact(self):
        self.assertEqual(ask(q("It ships 10% of its boxes.")).answer, "90")

    def test_cause_phrases(self):
        self.assertEqual(ask(q("It loses 40 boxes in a flood.", "Then the stock is tripled after a merger.")).answer,
                         "180")

    def test_total_shipped(self):
        res = ask(START + "It ships 5 boxes per day for 3 days. It receives 2 deliveries of 7 boxes each. "
                  "It ships 4 boxes per day for 2 days. How many boxes did it ship in total?")
        self.assertEqual((res.status, res.answer), ("ok", "23"))


class RefusalTest(unittest.TestCase):
    def assertStatus(self, question, status):
        res = ask(question)
        self.assertEqual(res.status, status, res)
        if status != "ok":
            self.assertIsNone(res.answer)

    def test_unknown_operation(self):
        self.assertStatus(q("Then the stock is squared."), "unparsed")

    def test_extra_sentence(self):
        self.assertStatus(q("It ships 5 boxes.", "The manager is happy."), "unparsed")

    def test_second_question(self):
        self.assertStatus(q("How many boxes did it ship?"), "unparsed")

    def test_question_about_verb_never_used(self):
        self.assertStatus(START + "It ships 5 boxes per day for 3 days. How many boxes did it receive in total?",
                          "unparsed")

    def test_question_other_items(self):
        self.assertStatus(START + "It ships 5 boxes. How many crates does it have at the end?", "infeasible")
        self.assertStatus(START + "It ships 5 boxes. How many trucks does it own at the end?", "infeasible")

    def test_vague_quantity_is_infeasible(self):
        self.assertStatus(q("It ships some boxes per day for 6 days."), "infeasible")
        self.assertStatus(q("It receives several boxes."), "infeasible")

    def test_unparsed_sentence_with_other_item_question_stays_unparsed(self):
        self.assertStatus(START + "Each pallet holds 10 boxes. How many pallets does it have at the end?", "unparsed")

    def test_percent_not_integer_without_rounding(self):
        self.assertStatus(START.replace("100", "7") + "It ships 15% of its boxes." + END, "unparsed")

    def test_fraction(self):
        self.assertStatus(q("It ships 2.5 boxes per day for 2 days."), "unparsed")

    def test_ship_items_mismatch(self):
        self.assertStatus(q("It ships 5 crates."), "unparsed")

    def test_no_start(self):
        self.assertStatus("It ships 5 boxes. How many boxes does it have at the end?", "unparsed")

    def test_no_question(self):
        self.assertStatus("A warehouse starts with 100 boxes. It ships 5 boxes.", "unparsed")

    def test_division_by_zero(self):
        self.assertStatus(q("Then the stock is split equally into 0 warehouses and it keeps one share (rounded down)."),
                          "infeasible")

    def test_negative_stock(self):
        # запас ушёл в минус: считаем арифметически (ответ не штрафуется, а отказ стоил бы токенов)
        self.assertStatus(q("It ships 50 boxes per day for 3 days."), "ok")

    def test_empty(self):
        self.assertStatus("", "unparsed")


if __name__ == "__main__":
    unittest.main()
