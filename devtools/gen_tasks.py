#!/usr/bin/env python3
"""Генератор тренировочных задач в формате примеров организаторов.

Организаторы дали 8 примеров и сказали: финальный набор скрыт, 4 типа, «сложность и структура
будут похожи», 15–25% задач невыполнимы (в примерах невыполнимых нет). Тренировочный набор
участники собирают сами — этим и занимается скрипт.

    python devtools/gen_tasks.py --out DIR [--n 100] [--seed 1] [--infeasible 0.2] [--variation 0.35]
                                 [--types math,search,data_analysis,coding] [--meta FILE] [--solve-reference]

В каталоге задач лежит только то, что есть в примерах (task.json, workspace/). Всё служебное —
выполнима ли задача, шаблон, эталонный ответ, эталонные тесты и исправленный код — пишется
отдельно, в файл meta (по умолчанию DIR.meta.json рядом с каталогом). Харнесс его не читает:
он нужен только проверке (devtools/evaluate.py) и заглушке модели (devtools/mock_brain_oracle.py).

Виды невыполнимых задач — наши гипотезы по определению организаторов («нельзя точно решить
при имеющихся данных»); вид записан в meta.variant.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TAIL_ANSWER = (" Work only inside the current directory. When you are done, write ONLY the final answer to the "
               "file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
TAILS = {
    "math": " Compute it with the bash tool (python3 -c)." + TAIL_ANSWER,
    "search": " Use the file tools or bash (ls, grep, cat) to find it." + TAIL_ANSWER,
    "data_analysis": " Use bash with python3 to compute it." + TAIL_ANSWER,
}
CODING_INSTRUCTION = (
    "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
    "(do not edit the tests) until all tests pass. Work only inside the current directory. When you are done, "
    "write ONLY the word DONE to the file answer.txt in the current directory "
    "(for example with bash: echo 42 > answer.txt)."
)
TYPE_ORDER = ("math", "search", "data_analysis", "coding")   # как в примерах: номер задачи % 4
CHECKS = {"math": "exact_int", "search": "exact_str", "data_analysis": "exact_int", "coding": "unittest"}

FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Maya",
         "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Garcia", "Haddad", "Ivanova", "Kowalski", "Muller", "Novak", "Petrov", "Sato", "Smith", "Wang"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Data",
             "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods"]
CITIES = ["Accra", "Graz", "Hanoi", "Izmir", "Leeds", "Lisbon", "Medan", "Osaka", "Quito", "Tallinn", "Tbilisi",
          "Bergen", "Cusco", "Dakar", "Nagoya", "Porto"]
REGIONS = ["north", "south", "east", "west"]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]


# =========================================================================== math
def _math_base(rng: random.Random) -> dict:
    """Два подвида из примеров: «ships/receives» и «split/doubled»."""
    start = rng.randint(800, 2400)
    value = start
    sentences = [f"A warehouse starts with {start} boxes."]
    expr = str(start)
    if rng.random() < 0.5:
        # как math_0004: отгрузки по дням и поставки
        ops = rng.choice([["ship", "receive"], ["receive", "ship"], ["ship", "receive", "ship"],
                          ["receive", "ship", "receive"], ["ship", "ship", "receive"]])
        for op in ops:
            if op == "ship":
                per_day, days = rng.randint(5, 40), rng.randint(2, 6)
                while per_day * days > value:
                    per_day = max(1, per_day // 2)
                value -= per_day * days
                expr = f"({expr} - {per_day} * {days})"
                sentences.append(f"It ships {per_day} boxes per day for {days} days.")
            else:
                count, size = rng.randint(2, 6), rng.randint(20, 120)
                value += count * size
                expr = f"({expr} + {count} * {size})"
                sentences.append(f"It receives {count} deliveries of {size} boxes each.")
        variant = "base_ship_receive"
    else:
        # как math_0000: деление с округлением вниз и удвоение
        for _ in range(rng.randint(2, 6)):
            if rng.random() < 0.55:
                k = rng.randint(2, 4)
                value //= k
                expr = f"({expr} // {k})"
                sentences.append(f"Then the stock is split equally into {k} warehouses and it keeps one share "
                                 f"(rounded down).")
            else:
                value *= 2
                expr = f"({expr} * 2)"
                sentences.append("Then the stock is doubled by merging with a sister warehouse.")
        variant = "base_split_double"
    sentences.append("How many boxes does it have at the end?")
    return {"question": " ".join(sentences), "expected": str(value), "oracle": expr, "variant": variant}


def _math_variation(rng: random.Random) -> dict:
    kind = rng.choice(["mixed", "other_story", "extra_ops", "total_shipped", "percent"])
    start = rng.randint(600, 3000)
    value, expr = start, str(start)
    if kind == "mixed":
        sentences = [f"A warehouse starts with {start} boxes."]
        for _ in range(rng.randint(3, 7)):
            op = rng.choice(["ship", "receive", "split", "double"])
            if op == "ship":
                a, b = rng.randint(3, 25), rng.randint(2, 5)
                if a * b > value:
                    continue
                value -= a * b
                expr = f"({expr} - {a} * {b})"
                sentences.append(f"It ships {a} boxes per day for {b} days.")
            elif op == "receive":
                a, b = rng.randint(2, 5), rng.randint(15, 90)
                value += a * b
                expr = f"({expr} + {a} * {b})"
                sentences.append(f"It receives {a} deliveries of {b} boxes each.")
            elif op == "split":
                k = rng.randint(2, 3)
                value //= k
                expr = f"({expr} // {k})"
                sentences.append(f"Then the stock is split equally into {k} warehouses and it keeps one share "
                                 f"(rounded down).")
            else:
                value *= 2
                expr = f"({expr} * 2)"
                sentences.append("Then the stock is doubled by merging with a sister warehouse.")
        sentences.append("How many boxes does it have at the end?")
    elif kind == "other_story":
        place, item = rng.choice([("library", "books"), ("bakery", "loaves"), ("farm", "crates"), ("store", "cans")])
        sentences = [f"A {place} starts with {start} {item}."]
        a, b = rng.randint(4, 30), rng.randint(2, 6)
        value -= a * b
        expr = f"({expr} - {a} * {b})"
        sentences.append(f"It gives away {a} {item} per day for {b} days.")
        c = rng.randint(50, 400)
        value += c
        expr = f"({expr} + {c})"
        sentences.append(f"Then it buys {c} more {item}.")
        sentences.append(f"How many {item} does the {place} have now?")
    elif kind == "extra_ops":
        sentences = [f"A warehouse starts with {start} boxes."]
        lost = rng.randint(10, 200)
        value -= lost
        expr = f"({expr} - {lost})"
        sentences.append(f"It loses {lost} boxes in a flood.")
        value *= 3
        expr = f"({expr} * 3)"
        sentences.append("Then the stock is tripled after a merger.")
        got = rng.randint(10, 300)
        value += got
        expr = f"({expr} + {got})"
        sentences.append(f"It receives {got} boxes.")
        sentences.append("How many boxes does it have at the end?")
    elif kind == "total_shipped":
        sentences = [f"A warehouse starts with {start} boxes."]
        a, b = rng.randint(5, 30), rng.randint(2, 6)
        c, d = rng.randint(5, 30), rng.randint(2, 6)
        sentences.append(f"It ships {a} boxes per day for {b} days.")
        sentences.append(f"It receives {rng.randint(2, 5)} deliveries of {rng.randint(20, 90)} boxes each.")
        sentences.append(f"It ships {c} boxes per day for {d} days.")
        value = a * b + c * d
        expr = f"({a} * {b} + {c} * {d})"
        sentences.append("How many boxes did it ship in total?")
    else:
        pct = rng.choice([10, 20, 25, 50])
        sentences = [f"A warehouse starts with {start} boxes."]
        value = value - value * pct // 100
        expr = f"({expr} - {expr} * {pct} // 100)"
        sentences.append(f"It sells {pct} percent of its boxes (the number sold is rounded down).")
        add = rng.randint(20, 200)
        value += add
        expr = f"({expr} + {add})"
        sentences.append(f"It receives {add} boxes.")
        sentences.append("How many boxes does it have at the end?")
    return {"question": " ".join(sentences), "expected": str(value), "oracle": expr, "variant": f"var_{kind}"}


def _math_infeasible(rng: random.Random) -> dict:
    kind = rng.choice(["missing_number", "unknown_quantity", "zero_split"])
    start = rng.randint(800, 2400)
    if kind == "missing_number":
        question = (f"A warehouse starts with {start} boxes. It ships some boxes per day for {rng.randint(2, 6)} days. "
                    f"It receives {rng.randint(2, 5)} deliveries of {rng.randint(20, 90)} boxes each. "
                    "How many boxes does it have at the end?")
    elif kind == "unknown_quantity":
        question = (f"A warehouse starts with {start} boxes. It ships {rng.randint(5, 30)} boxes per day for "
                    f"{rng.randint(2, 6)} days. How many trucks does it own at the end?")
    else:
        question = (f"A warehouse starts with {start} boxes. Then the stock is split equally into 0 warehouses and it "
                    "keeps one share (rounded down). How many boxes does it have at the end?")
    return {"question": question, "expected": None, "oracle": "IMPOSSIBLE", "variant": f"inf_{kind}"}


def make_math(rng: random.Random, mode: str) -> dict:
    spec = {"base": _math_base, "variation": _math_variation, "infeasible": _math_infeasible}[mode](rng)
    spec["instruction"] = spec.pop("question") + TAILS["math"]
    spec["files"] = {}
    return spec


# =========================================================================== search
def _build_kb(rng: random.Random, extra_field: bool = False) -> tuple[dict, dict]:
    people_names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], rng.randint(24, 30))
    cities = rng.sample(CITIES, len(COMPANIES))
    companies = {name: {"Headquarters": cities[i], "Founded": rng.randint(1950, 2020)}
                 for i, name in enumerate(COMPANIES)}
    people = {}
    for name in people_names:
        people[name] = {"Born": rng.randint(1960, 2002), "Employer": rng.choice(COMPANIES),
                        "Mentor": rng.choice([p for p in people_names if p != name])}
        if extra_field and rng.random() < 0.6:
            people[name]["Role"] = rng.choice(["Engineer", "Analyst", "Designer", "Manager", "Researcher"])
    return people, companies


def _kb_files(people: dict, companies: dict) -> dict:
    files = {}
    for name, fields in people.items():
        body = "".join(f"{k}: {v}\n" for k, v in fields.items())
        files[f"kb/person_{name.lower().replace(' ', '_')}.md"] = f"# {name}\n\n{body}"
    for name, fields in companies.items():
        body = "".join(f"{k}: {v}\n" for k, v in fields.items())
        files[f"kb/company_{name.lower().replace(' ', '_')}.md"] = f"# {name}\n\n{body}"
    return files


KB_INTRO = "The directory kb/ contains a small knowledge base of markdown files. "


def _search_base(rng: random.Random) -> dict:
    people, companies = _build_kb(rng)
    person = rng.choice(list(people))
    if rng.random() < 0.5:
        question = f"In which year was {person} born?"
        expected = str(people[person]["Born"])
        oracle = f"kb[{person!r}]['Born']"
        variant = "base_born"
    else:
        question = f"In which city is the headquarters of {person}'s employer?"
        expected = companies[people[person]["Employer"]]["Headquarters"]
        oracle = f"kb[kb[{person!r}]['Employer']]['Headquarters']"
        variant = "base_employer_hq"
    return {"question": question, "expected": expected, "oracle": oracle, "variant": variant,
            "files": _kb_files(people, companies)}


def _search_variation(rng: random.Random) -> dict:
    kind = rng.choice(["mentor", "employer_founded", "mentor_born", "mentor_employer_hq", "of_chain",
                       "count_employees", "older", "mentee", "role"])
    people, companies = _build_kb(rng, extra_field=(kind == "role"))
    person = rng.choice(list(people))
    rec = people[person]
    if kind == "mentor":
        question, expected = f"Who is the mentor of {person}?", rec["Mentor"]
        oracle = f"kb[{person!r}]['Mentor']"
    elif kind == "employer_founded":
        question = f"In which year was {person}'s employer founded?"
        expected = str(companies[rec["Employer"]]["Founded"])
        oracle = f"kb[kb[{person!r}]['Employer']]['Founded']"
    elif kind == "mentor_born":
        question = f"In which year was {person}'s mentor born?"
        expected = str(people[rec["Mentor"]]["Born"])
        oracle = f"kb[kb[{person!r}]['Mentor']]['Born']"
    elif kind == "mentor_employer_hq":
        question = f"In which city is the headquarters of the employer of {person}'s mentor?"
        expected = companies[people[rec["Mentor"]]["Employer"]]["Headquarters"]
        oracle = f"kb[kb[kb[{person!r}]['Mentor']]['Employer']]['Headquarters']"
    elif kind == "of_chain":
        question = f"What is the headquarters of the employer of the mentor of the mentor of {person}?"
        second = people[people[rec["Mentor"]]["Mentor"]]
        expected = companies[second["Employer"]]["Headquarters"]
        oracle = f"kb[kb[kb[kb[{person!r}]['Mentor']]['Mentor']]['Employer']]['Headquarters']"
    elif kind == "count_employees":
        company = rec["Employer"]
        question = f"How many people in the knowledge base work at {company}?"
        expected = str(sum(1 for p in people.values() if p["Employer"] == company))
        oracle = f"sum(1 for v in kb.values() if v.get('Employer') == {company!r})"
    elif kind == "older":
        other = rng.choice([p for p in people if people[p]["Born"] != rec["Born"]])
        question = f"Who was born earlier, {person} or {other}?"
        expected = person if rec["Born"] < people[other]["Born"] else other
        oracle = f"min([{person!r}, {other!r}], key=lambda n: kb[n]['Born'])"
    elif kind == "mentee":
        counts: dict[str, list[str]] = {}
        for name, fields in people.items():
            counts.setdefault(fields["Mentor"], []).append(name)
        unique = [(m, ms[0]) for m, ms in counts.items() if len(ms) == 1]
        if not unique:
            return _search_variation(rng)
        mentor, mentee = rng.choice(unique)
        question, expected = f"Who is mentored by {mentor}?", mentee
        oracle = f"[n for n, v in kb.items() if v.get('Mentor') == {mentor!r}][0]"
    else:
        with_role = [p for p in people if "Role" in people[p]]
        if not with_role:
            return _search_variation(rng)
        person = rng.choice(with_role)
        question, expected = f"What is the role of {person}?", people[person]["Role"]
        oracle = f"kb[{person!r}]['Role']"
    return {"question": question, "expected": expected, "oracle": oracle, "variant": f"var_{kind}",
            "files": _kb_files(people, companies)}


def _search_infeasible(rng: random.Random) -> dict:
    kind = rng.choice(["unknown_person", "missing_company", "missing_field"])
    people, companies = _build_kb(rng)
    person = rng.choice(list(people))
    if kind == "unknown_person":
        absent = [f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}" not in people]
        ghost = rng.choice(absent)
        question = rng.choice([f"In which year was {ghost} born?",
                               f"In which city is the headquarters of {ghost}'s employer?"])
    elif kind == "missing_company":
        question = f"In which city is the headquarters of {person}'s employer?"
        del companies[people[person]["Employer"]]
    else:
        question = f"In which year was {person} born?"
        del people[person]["Born"]
    return {"question": question, "expected": None, "oracle": "NOT_FOUND", "variant": f"inf_{kind}",
            "files": _kb_files(people, companies)}


def make_search(rng: random.Random, mode: str) -> dict:
    spec = {"base": _search_base, "variation": _search_variation, "infeasible": _search_infeasible}[mode](rng)
    spec["instruction"] = KB_INTRO + spec.pop("question") + TAILS["search"]
    return spec


# =========================================================================== data_analysis
def _sales_rows(rng: random.Random, n: int) -> list[dict]:
    return [{"order_id": i + 1, "region": rng.choice(REGIONS), "product": rng.choice(PRODUCTS),
             "month": rng.randint(1, 12), "units": rng.randint(1, 50), "price": rng.randint(5, 99)}
            for i in range(n)]


def _csv_text(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[c] for c in columns])
    return buf.getvalue()


SALES_COLUMNS = ["order_id", "region", "product", "month", "units", "price"]


def _data_intro(n: int, columns=SALES_COLUMNS, name: str = "sales.csv") -> str:
    return f"The file {name} has columns {','.join(columns)} ({n} rows). "


def _data_base(rng: random.Random) -> dict:
    n = rng.choice([20, 20, 50, 100, 300, 300])
    rows = _sales_rows(rng, n)
    product = rng.choice(PRODUCTS)
    lo = rng.randint(1, 11)
    hi = min(12, lo + rng.choice([1, 1, 1, 2, 3]))
    expected = sum(r["units"] for r in rows if r["product"] == product and lo <= r["month"] <= hi)
    question = f"How many units of product '{product}' were sold in months {lo} through {hi} inclusive?"
    oracle = f"sum(r['units'] for r in rows if r['product'] == {product!r} and {lo} <= r['month'] <= {hi})"
    return {"question": _data_intro(n) + question, "expected": str(expected), "oracle": oracle,
            "variant": "base_units_product_months", "files": {"sales.csv": _csv_text(rows, SALES_COLUMNS)}}


def _data_variation(rng: random.Random) -> dict:
    kind = rng.choice(["region_months", "revenue", "count_orders", "top_product", "avg_price", "two_filters"])
    n = rng.choice([20, 60, 150, 300])
    rows = _sales_rows(rng, n)
    product, region = rng.choice(PRODUCTS), rng.choice(REGIONS)
    check = "exact_int"
    if kind == "region_months":
        lo = rng.randint(1, 10)
        hi = lo + rng.randint(1, 2)
        question = f"How many units were sold in region '{region}' in months {lo} through {hi} inclusive?"
        expected = str(sum(r["units"] for r in rows if r["region"] == region and lo <= r["month"] <= hi))
        oracle = f"sum(r['units'] for r in rows if r['region'] == {region!r} and {lo} <= r['month'] <= {hi})"
    elif kind == "revenue":
        question = f"What is the total revenue (units times price) of product '{product}'?"
        expected = str(sum(r["units"] * r["price"] for r in rows if r["product"] == product))
        oracle = f"sum(r['units'] * r['price'] for r in rows if r['product'] == {product!r})"
    elif kind == "count_orders":
        threshold = rng.randint(10, 40)
        question = f"How many orders of product '{product}' have more than {threshold} units?"
        expected = str(sum(1 for r in rows if r["product"] == product and r["units"] > threshold))
        oracle = f"sum(1 for r in rows if r['product'] == {product!r} and r['units'] > {threshold})"
    elif kind == "top_product":
        totals = {p: sum(r["units"] for r in rows if r["product"] == p) for p in PRODUCTS}
        best = max(totals.values())
        if list(totals.values()).count(best) != 1:
            return _data_variation(rng)
        question = "Which product has the highest total number of units sold?"
        expected = max(totals, key=totals.get)
        oracle = ("max(set(r['product'] for r in rows), "
                  "key=lambda p: sum(r['units'] for r in rows if r['product'] == p))")
        check = "exact_str"
    elif kind == "avg_price":
        prices = [r["price"] for r in rows if r["product"] == product]
        if not prices:
            return _data_variation(rng)
        question = f"What is the average price of product '{product}', rounded down to an integer?"
        expected = str(sum(prices) // len(prices))
        oracle = (f"(lambda p: sum(p) // len(p))([r['price'] for r in rows if r['product'] == {product!r}])")
    else:
        lo = rng.randint(1, 9)
        hi = lo + rng.randint(1, 3)
        question = (f"How many units of product '{product}' were sold in region '{region}' in months {lo} through "
                    f"{hi} inclusive?")
        expected = str(sum(r["units"] for r in rows
                           if r["product"] == product and r["region"] == region and lo <= r["month"] <= hi))
        oracle = (f"sum(r['units'] for r in rows if r['product'] == {product!r} and r['region'] == {region!r} "
                  f"and {lo} <= r['month'] <= {hi})")
    return {"question": _data_intro(n) + question, "expected": expected, "oracle": oracle, "check": check,
            "variant": f"var_{kind}", "files": {"sales.csv": _csv_text(rows, SALES_COLUMNS)}}


def _data_infeasible(rng: random.Random) -> dict:
    kind = rng.choice(["missing_column", "missing_file", "column_removed"])
    n = rng.choice([20, 100, 300])
    rows = _sales_rows(rng, n)
    product = rng.choice(PRODUCTS)
    lo = rng.randint(1, 10)
    hi = lo + 1
    if kind == "missing_column":
        question = (_data_intro(n) + f"What is the total discount given on product '{product}' in months {lo} "
                                     f"through {hi} inclusive?")
        files = {"sales.csv": _csv_text(rows, SALES_COLUMNS)}
    elif kind == "missing_file":
        question = (_data_intro(n) + f"How many units of product '{product}' were sold in months {lo} through "
                                     f"{hi} inclusive?")
        files = {"notes.txt": "The export job has not produced sales.csv yet.\n"}
    else:
        columns = [c for c in SALES_COLUMNS if c != "units"]
        question = (_data_intro(n, columns) + f"How many units of product '{product}' were sold in months {lo} "
                                              f"through {hi} inclusive?")
        files = {"sales.csv": _csv_text(rows, columns)}
    return {"question": question, "expected": None, "oracle": "IMPOSSIBLE", "variant": f"inf_{kind}", "files": files}


def make_data(rng: random.Random, mode: str) -> dict:
    spec = {"base": _data_base, "variation": _data_variation, "infeasible": _data_infeasible}[mode](rng)
    spec["instruction"] = spec.pop("question") + TAILS["data_analysis"]
    return spec


# =========================================================================== coding
# (имя, исправный код, строка проверки, варианты бага: (что заменить, на что))
FUNCTIONS = [
    ("total", "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
     "self.assertEqual(m.total([1, 2, 3]), 6)", [("s += x", "s -= x")]),
    ("maximum", "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n"
                "    return best\n",
     "self.assertEqual(m.maximum([3, 9, 2]), 9)", [("x > best", "x < best")]),
    ("minimum", "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n"
                "    return best\n",
     "self.assertEqual(m.minimum([3, 9, 2]), 2)", [("x < best", "x > best")]),
    ("average", "def average(xs):\n    return sum(xs) / len(xs)\n",
     "self.assertEqual(m.average([2, 4, 6]), 4)", [("sum(xs) / len(xs)", "sum(xs) * len(xs)"),
                                                   ("sum(xs) / len(xs)", "sum(xs) / (len(xs) + 1)")]),
    ("first_n", "def first_n(xs, n):\n    return xs[:n]\n",
     "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])", [("xs[:n]", "xs[n:]"), ("xs[:n]", "xs[:n + 1]")]),
    ("last_n", "def last_n(xs, n):\n    return xs[-n:]\n",
     "self.assertEqual(m.last_n([5, 6, 7, 8], 2), [7, 8])", [("xs[-n:]", "xs[:n]")]),
    ("count_even", "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
     "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)", [("x % 2 == 0", "x % 2 == 1")]),
    ("count_odd", "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
     "self.assertEqual(m.count_odd([1, 2, 4, 6, 7]), 2)", [("x % 2 == 1", "x % 2 == 0")]),
    ("reverse_words", "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
     "self.assertEqual(m.reverse_words('c a b'), 'b a c')", [("reversed(s.split())", "sorted(s.split())")]),
    ("is_palindrome", "def is_palindrome(s):\n    return s == s[::-1]\n",
     "self.assertTrue(m.is_palindrome('level'))", [("s == s[::-1]", "s != s[::-1]")]),
    ("factorial", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n"
                  "    return result\n",
     "self.assertEqual(m.factorial(5), 120)", [("result *= i", "result += i"), ("range(2, n + 1)", "range(2, n)")]),
    ("clamp", "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
     "self.assertEqual(m.clamp(15, 0, 10), 10)", [("max(lo, min(x, hi))", "min(lo, min(x, hi))")]),
    ("unique", "def unique(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n"
               "    return out\n",
     "self.assertEqual(m.unique([1, 2, 1, 3, 2]), [1, 2, 3])", [("if x not in out", "if x in out")]),
    ("count_vowels", "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
     "self.assertEqual(m.count_vowels('Banana'), 3)", [("if ch in 'aeiou'", "if ch not in 'aeiou'")]),
    ("range_sum", "def range_sum(a, b):\n    return sum(range(a, b + 1))\n",
     "self.assertEqual(m.range_sum(1, 4), 10)", [("range(a, b + 1)", "range(a, b)")]),
    ("capitalize_words", "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
     "self.assertEqual(m.capitalize_words('hello big world'), 'Hello Big World')",
     [("w.capitalize()", "w.upper()")]),
    ("dot", "def dot(xs, ys):\n    return sum(x * y for x, y in zip(xs, ys))\n",
     "self.assertEqual(m.dot([1, 2, 3], [4, 5, 6]), 32)", [("x * y", "x + y")]),
    ("running_total", "def running_total(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n"
                      "        out.append(s)\n    return out\n",
     "self.assertEqual(m.running_total([1, 2, 3]), [1, 3, 6])", [("s += x", "s = x")]),
    ("square_all", "def square_all(xs):\n    return [x * x for x in xs]\n",
     "self.assertEqual(m.square_all([1, 2, 3]), [1, 4, 9])", [("x * x", "x + x")]),
    ("count_positive", "def count_positive(xs):\n    return sum(1 for x in xs if x > 0)\n",
     "self.assertEqual(m.count_positive([-1, 0, 2, 5]), 2)", [("x > 0", "x >= 0"), ("x > 0", "x < 0")]),
    ("is_sorted", "def is_sorted(xs):\n    return all(a <= b for a, b in zip(xs, xs[1:]))\n",
     "self.assertTrue(m.is_sorted([1, 2, 2, 5]))", [("a <= b", "a < b"), ("all(", "any(")]),
    ("longest_word", "def longest_word(s):\n    return max(s.split(), key=len)\n",
     "self.assertEqual(m.longest_word('a quick brown fox'), 'quick')", [("max(s.split(), key=len)",
                                                                           "min(s.split(), key=len)")]),
    ("abs_diff", "def abs_diff(a, b):\n    return abs(a - b)\n",
     "self.assertEqual(m.abs_diff(3, 10), 7)", [("abs(a - b)", "abs(a + b)")]),
    ("countdown", "def countdown(n):\n    out = []\n    while n > 0:\n        out.append(n)\n        n -= 1\n"
                  "    return out\n",
     "self.assertEqual(m.countdown(3), [3, 2, 1])", [("while n > 0", "while n > 1"), ("out.append(n)",
                                                                                         "out.append(n - 1)")]),
]
SECOND_ASSERTS = {
    "total": "self.assertEqual(m.total([10, -4]), 6)",
    "maximum": "self.assertEqual(m.maximum([-5, -2, -9]), -2)",
    "minimum": "self.assertEqual(m.minimum([4, 4, 7]), 4)",
    "count_even": "self.assertEqual(m.count_even([2, 2, 3]), 2)",
    "first_n": "self.assertEqual(m.first_n([1, 2], 1), [1])",
    "factorial": "self.assertEqual(m.factorial(3), 6)",
    "range_sum": "self.assertEqual(m.range_sum(2, 3), 5)",
    "dot": "self.assertEqual(m.dot([2], [5]), 10)",
}
WRONG_VALUES = {
    "total": "self.assertEqual(m.total([1, 2, 3]), 7)",
    "maximum": "self.assertEqual(m.maximum([3, 9, 2]), 3)",
    "count_even": "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 4)",
    "factorial": "self.assertEqual(m.factorial(5), 121)",
    "is_palindrome": "self.assertFalse(m.is_palindrome('level'))",
    "first_n": "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6])",
    "dot": "self.assertEqual(m.dot([1, 2, 3], [4, 5, 6]), 33)",
    "average": "self.assertEqual(m.average([2, 4, 6]), 5)",
}


def _test_file(tests: list[tuple[str, list[str]]]) -> str:
    lines = ["import unittest", "", "import mylib as m", "", "", "class T(unittest.TestCase):"]
    for index, (name, asserts) in enumerate(tests):
        if index:
            lines.append("")
        lines.append(f"    def test_{name}(self):")
        lines.extend(f"        {a}" for a in asserts)
    lines += ["", "", "if __name__ == '__main__':", "    unittest.main()", ""]
    return "\n".join(lines)


def _run_tests(mylib: str, tests: str) -> tuple[bool, int]:
    """Запускает тесты во временном каталоге: (все прошли, сколько упало)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "mylib.py").write_text(mylib, encoding="utf-8")
        Path(tmp, "test_mylib.py").write_text(tests, encoding="utf-8")
        proc = subprocess.run([sys.executable, "-B", "test_mylib.py"], cwd=tmp, capture_output=True, text=True,
                              timeout=30)
        failed = 0
        for line in proc.stderr.splitlines():
            if line.startswith("FAILED ("):
                failed = sum(int(part.split("=")[1]) for part in line[8:-1].split(", "))
        return proc.returncode == 0, failed


def make_coding(rng: random.Random, mode: str) -> dict:
    for _ in range(50):
        chosen = rng.sample(FUNCTIONS, rng.randint(4, 6))
        bug_count = rng.choice([1, 2, 2, 2, 3]) if mode != "infeasible" else rng.choice([1, 2])
        buggy_names = {f[0] for f in rng.sample(chosen, min(bug_count, len(chosen)))}
        good_parts, bad_parts, tests = [], [], []
        fixed_functions = []
        for name, source, assertion, mutations in chosen:
            good_parts.append(source)
            if name in buggy_names:
                old, new = rng.choice(mutations)
                assert old in source, (name, old)
                bad_parts.append(source.replace(old, new, 1))
                fixed_functions.append(source.rstrip("\n"))
            else:
                bad_parts.append(source)
            asserts = [assertion]
            if mode == "variation" and name in SECOND_ASSERTS and rng.random() < 0.6:
                asserts.append(SECOND_ASSERTS[name])
            tests.append((name, asserts))
        variant = "base" if mode == "base" else "var_two_asserts"
        expected: object = "tests_pass"
        if mode == "infeasible":
            candidates = [n for n, *_ in chosen if n in WRONG_VALUES]
            if not candidates:
                continue
            target = rng.choice(candidates)
            # тот же вызов, другое ожидаемое значение: оба теста одновременно пройти не могут
            tests.append((f"{target}_again", [WRONG_VALUES[target]]))
            variant = "inf_contradictory_tests"
            expected = None
        fixed = "\n\n".join(good_parts)
        buggy = "\n\n".join(bad_parts)
        test_text = _test_file(tests)
        ok_fixed, failed_fixed = _run_tests(fixed, test_text)
        _ok_buggy, failed_buggy = _run_tests(buggy, test_text)
        if mode == "infeasible":
            if ok_fixed or failed_fixed != 1:
                continue
        elif not ok_fixed or failed_buggy != len(buggy_names):
            continue
        return {
            "instruction": CODING_INSTRUCTION, "expected": expected, "variant": variant,
            "files": {"mylib.py": buggy, "test_mylib.py": test_text},
            "oracle": "```python\n" + "\n\n\n".join(fixed_functions) + "\n```" if mode != "infeasible" else "IMPOSSIBLE",
            "fixed": {"mylib.py": fixed}, "tests": {"test_mylib.py": test_text}, "bugs": sorted(buggy_names),
        }
    raise RuntimeError("cannot generate a coding task")


MAKERS = {"math": make_math, "search": make_search, "data_analysis": make_data, "coding": make_coding}


# =========================================================================== сборка набора
def generate(out: Path, n: int, seed: int, infeasible: float, variation: float, types: list[str],
             meta_path: Path) -> dict:
    rng = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts = {t: sum(1 for i in range(n) if types[i % len(types)] == t) for t in types}
    # доля невыполнимых выдерживается по каждому типу отдельно
    modes: dict[str, list[str]] = {}
    for task_type, count in counts.items():
        n_inf = round(count * infeasible)
        n_var = round((count - n_inf) * variation)
        plan = ["infeasible"] * n_inf + ["variation"] * n_var + ["base"] * (count - n_inf - n_var)
        rng.shuffle(plan)
        modes[task_type] = plan
    meta = {"seed": seed, "n": n, "infeasible": infeasible, "variation": variation, "tasks": {}}
    for index in range(n):
        task_type = types[index % len(types)]
        mode = modes[task_type].pop()
        spec = MAKERS[task_type](rng, mode)
        task_id = f"{task_type}_{index:04d}"
        task_dir = out / task_id
        task_dir.mkdir()
        check = spec.get("check", CHECKS[task_type])
        task = {"task_id": task_id, "task_type": task_type, "instruction": spec["instruction"],
                "reward": rng.randint(1, 5), "expected": spec["expected"], "check": check}
        (task_dir / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for rel, text in spec.get("files", {}).items():
            path = task_dir / "workspace" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        if task_type != "math" and not (task_dir / "workspace").exists():
            (task_dir / "workspace").mkdir()
        meta["tasks"][task_id] = {
            "type": task_type, "feasible": spec["expected"] is not None, "variant": spec["variant"], "mode": mode,
            "expected": spec["expected"], "check": check, "reward": task["reward"], "oracle": spec.get("oracle"),
            "fixed": spec.get("fixed"), "tests": spec.get("tests"), "bugs": spec.get("bugs"),
        }
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def solve_reference(out: Path, meta: dict) -> None:
    """Эталонное решение: пишет верные ответы по meta (для проверки генератора и evaluate.py)."""
    for task_id, info in meta["tasks"].items():
        if not info["feasible"]:
            continue
        task_dir = out / task_id
        if info["type"] == "coding":
            for rel, text in info["fixed"].items():
                (task_dir / "workspace" / rel).write_text(text, encoding="utf-8")
            answer = "DONE"
        else:
            answer = info["expected"]
        (task_dir / "answer.txt").write_text(f"{answer}\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--infeasible", type=float, default=0.2)
    ap.add_argument("--variation", type=float, default=0.35)
    ap.add_argument("--types", default=",".join(TYPE_ORDER))
    ap.add_argument("--meta", default=None)
    ap.add_argument("--solve-reference", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else out.with_name(out.name + ".meta.json")
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    meta = generate(out, args.n, args.seed, args.infeasible, args.variation, types, meta_path)
    if args.solve_reference:
        solve_reference(out, meta)
    feasible = sum(1 for t in meta["tasks"].values() if t["feasible"])
    variants: dict[str, int] = {}
    for info in meta["tasks"].values():
        variants[info["variant"]] = variants.get(info["variant"], 0) + 1
    print(f"generated {len(meta['tasks'])} tasks in {out} (feasible {feasible}); meta: {meta_path}")
    print("variants:", json.dumps(dict(sorted(variants.items())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
