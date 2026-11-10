"""Тесты навыка tables: примеры организаторов, загрузка, разбор вопросов и отказы."""
import importlib.util
import json
import sys
import tempfile
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


tb = _load("tables")

SALES = """order_id,region,product,month,units,price
1,west,widget,5,5,33
2,north,sprocket,6,28,28
3,north,widget,7,10,17
4,west,gadget,7,23,98
5,east,widget,8,4,10
6,west,widget,9,100,12
7,west,widget,8,6,20
"""
PREAMBLE = "The file sales.csv has columns order_id,region,product,month,units,price (7 rows). "
TAIL = (" Use bash with python3 to compute it. Work only inside the current directory. When you are done, "
        "write ONLY the final answer to the file answer.txt in the current directory (for example with bash: "
        "echo 42 > answer.txt).")


class _TmpMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, text: str, encoding="utf-8") -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding=encoding)
        return p


class ExamplesTest(unittest.TestCase):
    def check_task(self, task_id):
        d = TASKS / task_id
        if not (d / "task.json").exists():
            self.skipTest(f"нет примера {d}")
        task = json.loads((d / "task.json").read_text())
        paths = tb.find_tables(d / "workspace")
        self.assertEqual([p.name for p in paths], ["sales.csv"])
        table = tb.load_table(paths[0])
        res = tb.answer_table_question(table, task["instruction"])
        self.assertEqual(res.status, "ok", res.detail)
        self.assertEqual(res.answer, task["expected"])
        self.assertEqual(res.detail, "")
        self.assertIn("sum(units)", res.plan)

    def test_data_analysis_0002(self):
        self.check_task("data_analysis_0002")

    def test_data_analysis_0006(self):
        self.check_task("data_analysis_0006")


class LoadTest(_TmpMixin, unittest.TestCase):
    def test_find_tables(self):
        self.write("a.csv", "x\n1\n")
        self.write("sub/b.tsv", "x\n1\n")
        self.write(".hidden/c.csv", "x\n1\n")
        self.write(".d.csv", "x\n1\n")
        self.write("notes.txt", "x")
        names = [p.relative_to(self.dir).as_posix() for p in tb.find_tables(self.dir)]
        self.assertEqual(names, ["a.csv", "sub/b.tsv"])

    def test_types_and_distinct(self):
        t = tb.load_table(self.write("sales.csv", SALES))
        self.assertEqual(t.name, "sales.csv")
        self.assertEqual(t.columns, ["order_id", "region", "product", "month", "units", "price"])
        self.assertEqual(t.types["units"], "int")
        self.assertEqual(t.types["region"], "str")
        self.assertEqual(t.distinct["product"], ["gadget", "sprocket", "widget"])
        self.assertEqual(t.rows[0]["units"], 5)
        self.assertEqual(len(t.rows), 7)

    def test_semicolon_and_bom(self):
        p = self.dir / "s.csv"
        p.write_bytes("﻿name;score;ratio\na;1;0.5\nb;2;1\n".encode("utf-8"))
        t = tb.load_table(p)
        self.assertEqual(t.columns, ["name", "score", "ratio"])
        self.assertEqual(t.types, {"name": "str", "score": "int", "ratio": "float"})
        self.assertEqual(t.rows[1]["ratio"], 1.0)

    def test_tsv(self):
        t = tb.load_table(self.write("t.tsv", "a\tb\n1\tx\n2\ty\n"))
        self.assertEqual(t.columns, ["a", "b"])
        self.assertEqual(t.types["a"], "int")

    def test_empty_cells_keep_numeric(self):
        t = tb.load_table(self.write("e.csv", "k,v\na,1\nb,\nc,3\n"))
        self.assertEqual(t.types["v"], "int")
        self.assertIsNone(t.rows[1]["v"])

    def test_leading_zero_is_str(self):
        t = tb.load_table(self.write("z.csv", "code\n007\n010\n"))
        self.assertEqual(t.types["code"], "str")

    def test_negative_int(self):
        t = tb.load_table(self.write("n.csv", "v\n-3\n4\n"))
        self.assertEqual(t.types["v"], "int")
        self.assertEqual(t.rows[0]["v"], -3)

    def test_max_rows(self):
        t = tb.load_table(self.write("m.csv", "v\n" + "1\n" * 50), max_rows=10)
        self.assertEqual(len(t.rows), 10)

    def test_summary(self):
        t = tb.load_table(self.write("sales.csv", SALES))
        s = tb.table_summary(t, sample_rows=2)
        self.assertTrue(s.startswith("sales.csv: 7 rows; columns order_id:int, region:str"))
        self.assertIn("product in ['gadget', 'sprocket', 'widget']", s)
        self.assertIn("first rows: {'order_id': 1", s)
        self.assertEqual(s.count("'order_id'"), 2)


class QuestionTest(_TmpMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.t = tb.load_table(self.write("sales.csv", SALES))

    def ask(self, question):
        return tb.answer_table_question(self.t, question)

    def assertOk(self, question, answer, plan=None):
        res = self.ask(question)
        self.assertEqual(res.status, "ok", res.detail)
        self.assertEqual(res.answer, answer)
        if plan is not None:
            self.assertEqual(res.plan, plan)
        return res

    def assertStatus(self, question, status):
        res = self.ask(question)
        self.assertEqual(res.status, status, res)
        self.assertIsNone(res.answer)
        return res

    # --- шаблон из примеров
    def test_example_template_full_instruction(self):
        self.assertOk(PREAMBLE + "How many units of product 'widget' were sold in months 7 through 8 inclusive?" + TAIL,
                      "20", "sum(units) where product == 'widget' and 7 <= month <= 8")

    def test_example_template_short(self):
        self.assertOk("How many units of product 'widget' were sold in months 7 through 8 inclusive?", "20")

    def test_double_quotes(self):
        self.assertOk('How many units of product "widget" were sold in months 7 through 8 inclusive?', "20")

    # --- фильтры
    def test_two_equality_filters(self):
        self.assertOk("How many units of product 'widget' were sold in region 'west'?", "111",
                      "sum(units) where product == 'widget' and region == 'west'")

    def test_where_is(self):
        self.assertOk("How many units were sold where region is 'north'?", "38")

    def test_value_column_inferred(self):
        res = self.assertOk("How many units of 'gadget' were sold?", "23")
        self.assertIn("inferred", res.detail)

    def test_month_to(self):
        self.assertOk("How many units were sold in months 5 to 6?", "33")

    def test_between_inclusive(self):
        self.assertOk("How many units were sold between months 8 and 9 inclusive?", "110")

    def test_in_month_n(self):
        self.assertOk("How many units were sold in month 7?", "33", "sum(units) where month == 7")

    def test_comparison(self):
        self.assertOk("How many orders had units greater than 10?", "3", "count(rows) where units > 10")
        self.assertOk("How many orders had units at least 10?", "4")

    # --- меры
    def test_count_orders(self):
        self.assertOk("How many orders of product 'widget' were placed?", "5",
                      "count(rows) where product == 'widget'")

    def test_count_rows(self):
        self.assertOk("How many rows are there in region 'west'?", "4")

    def test_total_column(self):
        self.assertOk("What is the total units for product 'widget'?", "125")

    def test_total_number_of_orders(self):
        self.assertOk("What is the total number of orders in region 'west'?", "4")

    def test_sum_of(self):
        self.assertOk("What is the sum of units in month 8?", "10")

    def test_average(self):
        self.assertOk("What is the average units of product 'widget'?", "25")
        self.assertOk("What is the average price in month 7?", "57.5")

    def test_average_rounding(self):
        self.assertOk("What is the mean units in region 'west'?", "33.5")
        self.assertOk("What is the average units in month 8?", "5")

    def test_max_min(self):
        self.assertOk("What is the maximum units in region 'west'?", "100")
        self.assertOk("What is the lowest price of product 'widget'?", "10")

    # --- пустые наборы
    def test_missing_value_gives_zero(self):
        res = self.assertOk("How many units of product 'doohickey' were sold in months 7 through 8 inclusive?", "0")
        self.assertIn("not found", res.detail)

    def test_empty_range_count_zero(self):
        self.assertOk("How many orders were placed in month 12?", "0")

    def test_average_of_empty_set(self):
        res = self.ask("What is the average units of product 'doohickey'?")
        self.assertEqual(res.status, "empty")

    def test_empty_cells_skipped(self):
        t = tb.load_table(self.write("e.csv", "product,units\na,1\na,\nb,3\n"))
        res = tb.answer_table_question(t, "How many units of product 'a' were sold?")
        self.assertEqual((res.status, res.answer), ("ok", "1"))
        self.assertIn("empty", res.detail)

    def test_semicolon_table_question(self):
        t = tb.load_table(self.write("s.csv", "product;units;month\nwidget;3;1\nwidget;4;2\ngizmo;9;1\n"))
        res = tb.answer_table_question(t, "How many units of product 'widget' were sold in months 1 through 2 inclusive?")
        self.assertEqual((res.status, res.answer), ("ok", "7"))

    def test_float_sum_format(self):
        t = tb.load_table(self.write("f.csv", "k,w\na,0.1\na,0.2\nb,5\n"))
        res = tb.answer_table_question(t, "What is the total w for k 'a'?")
        self.assertEqual((res.status, res.answer), ("ok", "0.3"))

    # --- преамбула
    def test_preamble_mismatch_noted(self):
        res = self.assertOk("The file sales.csv has columns order_id,region,product,month,units (9 rows). "
                            "How many units were sold in month 7?", "33")
        self.assertIn("differ", res.detail)
        self.assertIn("9 rows", res.detail)

    def test_preamble_other_file(self):
        self.assertStatus("The file orders.csv has columns a,b (3 rows). How many units were sold in month 7?",
                          "unparsed")

    # --- невыполнимые / непонятые
    def test_missing_equality_column(self):
        res = self.assertStatus("How many units of category 'tools' were sold in months 7 through 8 inclusive?",
                                "missing_column")
        self.assertIn("category", res.detail)

    def test_missing_range_column(self):
        self.assertStatus("How many units of product 'widget' were sold in weeks 7 through 8 inclusive?",
                          "missing_column")

    def test_missing_measure_column(self):
        self.assertStatus("How many returns of product 'widget' were recorded in months 7 through 8 inclusive?",
                          "missing_column")
        self.assertStatus("What is the total discount for product 'widget'?", "missing_column")

    def test_unexplained_word(self):
        self.assertStatus("How many units of product 'widget' were sold online in months 7 through 8 inclusive?",
                          "unparsed")

    def test_revenue_refused(self):
        self.assertStatus("What is the total revenue of product 'widget'?", "missing_column")
        self.assertStatus("How much revenue did product 'widget' generate?", "unparsed")

    def test_sum_price_refused(self):
        self.assertStatus("What is the total price of product 'widget'?", "unparsed")

    def test_or_refused(self):
        self.assertStatus("How many units of product 'widget' or product 'gadget' were sold?", "unparsed")

    def test_not_refused(self):
        self.assertStatus("How many units were sold not in region 'west'?", "unparsed")

    def test_grouping_refused(self):
        self.assertStatus("Which region sold the most units of product 'widget'?", "unparsed")
        self.assertStatus("How many units were sold per region?", "unparsed")

    def test_distinct_refused(self):
        self.assertStatus("How many distinct products were sold in region 'west'?", "unparsed")
        self.assertStatus("How many products were sold in region 'west'?", "unparsed")

    def test_between_without_inclusive_refused(self):
        self.assertStatus("How many units were sold between months 5 and 8?", "unparsed")

    def test_two_questions_refused(self):
        self.assertStatus("How many units were sold in month 7? How many orders were placed?", "unparsed")

    def test_ambiguous_value_owner_refused(self):
        t = tb.load_table(self.write("amb.csv", "a,b,n\nx,y,1\ny,x,2\n"))
        res = tb.answer_table_question(t, "How many n of 'x' were there?")
        self.assertEqual(res.status, "unparsed")

    def test_numeric_value_in_quotes_for_text(self):
        self.assertStatus("How many units of month 'july' were sold?", "unparsed")


if __name__ == "__main__":
    unittest.main()
