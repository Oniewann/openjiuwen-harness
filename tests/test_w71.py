"""Целое среднее (стресс, зерно 10, data_analysis_1229): навык таблиц и путь модели пишут его одинаково.

Эталон стресс-генератора — repr(float) = «71.0», политика решения — «71» (12.0 -> 12, как в эталонах
exact_int). Пока политика не пересмотрена, оба пути обязаны совпадать: иначе голоса навыка и модели
расходятся на одном и том же числе."""
import tempfile
import unittest
from pathlib import Path

from harness.skills import tables as tb
from harness.textutil import fit_number_to_question

SALES = """order_id,region,product,month,units,price
1,north,gadget,4,46,18
2,north,sprocket,4,17,66
3,east,gizmo,11,38,42
4,north,sprocket,3,16,78
5,south,gizmo,7,19,51
6,south,gadget,6,16,9
7,north,sprocket,5,41,85
8,west,widget,5,27,24
9,south,gadget,6,45,97
10,south,widget,7,28,22
11,south,sprocket,10,25,79
12,west,gizmo,12,33,53
13,south,gadget,3,46,37
14,east,gizmo,2,28,20
15,east,gizmo,8,15,64
16,east,gizmo,11,18,8
17,south,widget,9,10,44
18,east,gadget,5,36,86
19,west,widget,3,40,55
20,north,sprocket,9,4,47
"""
Q = ("The file sales.csv has columns order_id,region,product,month,units,price (20 rows). "
     "What is the average price of product 'sprocket'?")


class WholeAverageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        p = Path(self._tmp.name) / "sales.csv"
        p.write_text(SALES)
        self.table = tb.load_table(p)

    def tearDown(self):
        self._tmp.cleanup()

    def test_skill_matches_model_path(self):
        # 355 / 5 = 71.0: python напечатал бы «71.0», модель тоже — её ответ проходит через fit_number_to_question
        res = tb.answer_table_question(self.table, Q)
        self.assertEqual(res.status, "ok", res.detail)
        self.assertEqual(res.answer, fit_number_to_question(repr(355 / 5), Q))

    def test_fractional_average_unchanged(self):
        # дробное среднее печатается как есть: (18 + 9 + 97 + 37 + 86) / 5 = 49.4
        res = tb.answer_table_question(self.table, Q.replace("'sprocket'", "'gadget'"))
        self.assertEqual((res.status, res.answer), ("ok", "49.4"), res.detail)


if __name__ == "__main__":
    unittest.main()
