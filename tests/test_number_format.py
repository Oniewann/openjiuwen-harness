"""Вид числового ответа: print()-вид без указаний, ровно N знаков для «rounded to N decimal places»."""
import types
import unittest

from harness.solvers.base import ExprSolver
from harness.textutil import canonical_number, fit_number_to_question, normalize_answer

AVG = "What is the average price of product 'widget'?"
AVG2 = "What is the average price of product 'widget', rounded to 2 decimal places?"


class CanonicalNumber(unittest.TestCase):
    def test_print_like(self):
        cases = {".5": "0.5", "-.5": "-0.5", "+5": "5", "12.": "12", "12.0": "12", "389.0": "389",
                 "46.00": "46", "-0": "0", "-0.0": "0", "0.5": "0.5", "12.5": "12.5", "-3": "-3",
                 "100000000000000000000": "100000000000000000000"}
        for raw, want in cases.items():
            self.assertEqual(canonical_number(raw), want, raw)

    def test_untouched(self):
        # нули после значащих цифр, дроби, проценты, экспонента, ведущие нули, не числа — как есть
        for raw in ("0.50", "1/2", "12.5%", "1e-05", "007", "1,234", "Osaka", "v1.0", ""):
            self.assertEqual(canonical_number(raw), raw, raw)

    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("Answer: .5"), "0.5")
        self.assertEqual(normalize_answer("-0.0"), "0")
        self.assertEqual(normalize_answer("'12.'"), "12")
        self.assertEqual(normalize_answer("0.50"), "0.50")


class FitNumberToQuestion(unittest.TestCase):
    def test_rounded_places_padded(self):
        cases = {"46": "46.00", "46.0": "46.00", "23.4": "23.40", "23.45": "23.45", "23.456": "23.46",
                 "-1.5": "-1.50", "-0.001": "0.00", "1e-05": "0.00"}
        for raw, want in cases.items():
            self.assertEqual(fit_number_to_question(raw, AVG2), want, raw)

    def test_rounded_places_variants(self):
        self.assertEqual(fit_number_to_question("3.1", "Compute the mean. Round it to 3 decimal places."), "3.100")
        self.assertEqual(fit_number_to_question("7", "What is the ratio, rounded to one decimal place?"), "7.0")
        self.assertEqual(fit_number_to_question("2.5", "Give the average with 2 decimal digits."), "2.50")

    def test_rounded_places_not_about_answer(self):
        q = "How many products have an average price above 12.5, rounded to 2 decimal places?"
        self.assertEqual(fit_number_to_question("3", q), "3")
        q2 = "What is the gap, rounded to 2 decimal places, and the ratio, rounded to 3 decimal places?"
        self.assertEqual(fit_number_to_question("1.5", q2), "1.5")
        self.assertEqual(fit_number_to_question("Osaka", AVG2), "Osaka")
        self.assertEqual(fit_number_to_question("12.5%", AVG2), "12.5%")

    def test_count_anywhere_not_padded(self):
        # счёт не в начале вопроса: знаки после точки относятся к порогу или к данным, не к ответу
        for q in ("Count the rows where price, rounded to 2 decimal places, is above 10.",
                  "What is the number of products whose price, rounded to 2 decimal places, exceeds 5?",
                  "The table lists prices with 2 decimal places. How many products cost more than 10?"):
            self.assertEqual(fit_number_to_question("7", q), "7", q)
            self.assertEqual(fit_number_to_question("7.5", q), "7.50", q)   # дробное значение — не счёт
        self.assertEqual(fit_number_to_question("7", AVG2), "7.00")

    def test_no_instruction_print_like(self):
        self.assertEqual(fit_number_to_question("0.50", AVG), "0.5")
        self.assertEqual(fit_number_to_question("12.50", AVG), "12.5")
        self.assertEqual(fit_number_to_question("12", AVG), "12")
        self.assertEqual(fit_number_to_question("0.333333", AVG), "0.333333")
        self.assertEqual(fit_number_to_question("0.00001", AVG), "0.00001")
        self.assertEqual(fit_number_to_question("1e-05", AVG), "1e-05")

    def test_format_words_keep_zeros(self):
        self.assertEqual(fit_number_to_question("12.50", "What is the price, in dollars and cents?"), "12.50")
        self.assertEqual(fit_number_to_question("12.50", "What is the price rounded to the nearest cent?"), "12.50")

    def test_copied_value_kept(self):
        # search: число переписано из базы, эталон — запись как есть
        self.assertEqual(fit_number_to_question("4.50", "What is the rating of Iris Optics?", computed=False), "4.50")
        self.assertEqual(fit_number_to_question("4.5", AVG2, computed=False), "4.50")

    def test_idempotent(self):
        for q in (AVG, AVG2):
            for raw in ("46", "23.4", "0.50", "23.456"):
                once = fit_number_to_question(raw, q)
                self.assertEqual(fit_number_to_question(once, q), once)


class ShapeAnswerNumbers(unittest.TestCase):
    def _shape(self, question: str, value: str, family: str = "data_analysis"):
        stub = types.SimpleNamespace(task=types.SimpleNamespace(question=question), family=family)
        return ExprSolver.shape_answer(stub, value)

    def test_shape_pads_rounded(self):
        self.assertEqual(self._shape(AVG2, "46"), ("46.00", ""))
        self.assertEqual(self._shape(AVG2, "23.4"), ("23.40", ""))

    def test_shape_examples_unchanged(self):
        # стиль эталонов из примеров: целые и строки не меняются
        self.assertEqual(self._shape("How many units of product 'widget' were sold in months 7 through 8 inclusive?",
                                     "158"), ("158", ""))
        self.assertEqual(self._shape("In which year was Irina Smith born?", "1983", "search"), ("1983", ""))
        self.assertEqual(self._shape("In which city is the headquarters of Sergei Garcia's employer?", "Osaka",
                                     "search"), ("Osaka", ""))

    def test_shape_search_keeps_zeros(self):
        self.assertEqual(self._shape("What is the rating of Iris Optics?", "4.50", "search"), ("4.50", ""))
        self.assertEqual(self._shape("What is the rating of Iris Optics?", "4.50"), ("4.5", ""))


if __name__ == "__main__":
    unittest.main()
