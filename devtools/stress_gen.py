#!/usr/bin/env python3
"""Стресс-генератор: гипотезы о скрытом наборе, которых нет в gen_tasks.py.
gen_tasks.py писался вместе с навыками и повторяет то, что они уже умеют. Этот генератор
независим: каждая функция ниже — одна гипотеза «а что, если в финале будет вот так»
(другая формулировка, другая операция, другой формат файла, другой вид невыполнимости).
    python devtools/stress_gen.py --out DIR [--per 4] [--seed 1] [--only math,search] [--match substr]
    python devtools/trial.py --tasks DIR --no-llm --show-log 0
    python devtools/stress_report.py --trial-out out.json
kind у гипотезы:
    ok  — выполнимая, эталон известен;
    inf — невыполнимая (answer.txt быть не должно; ответ не штрафуется, но токены жалко);
    amb — спорная: организаторы могли посчитать её и выполнимой, и нет. В expected лежит ответ,
          который выгоднее дать (если задача выполнима — он верный, если нет — ничего не теряем).
level: base (как в примерах) | easy | mid | hard | extreme (другой жанр задачи).
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_tasks import (CITIES, CODING_INSTRUCTION, COMPANIES, FIRST, KB_INTRO, LAST, PRODUCTS,  # noqa: E402
                       REGIONS, SALES_COLUMNS, TAILS, TAIL_ANSWER)
REG: list[dict] = []


def v(task_type: str, level: str, kind: str = "ok"):
    def deco(fn):
        REG.append({"type": task_type, "level": level, "kind": kind, "name": fn.__name__, "fn": fn})
        return fn
    return deco


# =========================================================================== MATH
QEND = "How many boxes does it have at the end?"


def _start(rng, lo=800, hi=2400):
    n = rng.randint(lo, hi)
    return n, [f"A warehouse starts with {n} boxes."]


def _ship(rng, val, pre="It"):
    a, d = rng.randint(5, 40), rng.randint(2, 6)
    while a * d > val and a > 1:
        a = max(1, a // 2)
    return f"{pre} ships {a} boxes per day for {d} days.", val - a * d, a * d


def _recv(rng, val, pre="It"):
    k, m = rng.randint(2, 6), rng.randint(20, 120)
    return f"{pre} receives {k} deliveries of {m} boxes each.", val + k * m, k * m


def _split(rng, val, k=None, pre="Then the"):
    k = k or rng.randint(2, 4)
    return f"{pre} stock is split equally into {k} warehouses and it keeps one share (rounded down).", val // k


def _double(rng, val, pre="Then the"):
    return f"{pre} stock is doubled by merging with a sister warehouse.", val * 2


def M(sents, value, q=QEND, **kw):
    return dict(question=" ".join(list(sents) + ([q] if q else [])), expected=None if value is None else str(value), **kw)


def _sd_chain(rng, val, sents, n):
    for _ in range(n):
        s, val = _split(rng, val) if rng.random() < 0.55 else _double(rng, val)
        sents.append(s)
    return val


def _sr_chain(rng, val, sents, n):
    for _ in range(n):
        s, val, _ = _ship(rng, val) if rng.random() < 0.5 else _recv(rng, val)
        sents.append(s)
    return val


@v("math", "base")
def m_base_ship_receive(rng):
    val, s = _start(rng)
    return M(s, _sr_chain(rng, val, s, rng.randint(2, 3)))


@v("math", "base")
def m_base_split_double(rng):
    val, s = _start(rng)
    return M(s, _sd_chain(rng, val, s, rng.randint(3, 6)))


@v("math", "easy")
def m_single_op(rng):
    val, s = _start(rng)
    return M(s, _sd_chain(rng, val, s, 1) if rng.random() < 0.5 else _sr_chain(rng, val, s, 1))


@v("math", "easy")
def m_no_ops(rng):
    val, s = _start(rng)
    return M(s, val)


@v("math", "mid")
def m_chain_12(rng):
    val, s = _start(rng, 5000, 90000)
    return M(s, _sd_chain(rng, val, s, 12))


@v("math", "hard")
def m_chain_40(rng):
    val, s = _start(rng, 10 ** 6, 10 ** 7)
    return M(s, _sd_chain(rng, val, s, 40))


@v("math", "hard")
def m_chain_ship_recv_25(rng):
    val, s = _start(rng, 20000, 50000)
    return M(s, _sr_chain(rng, val, s, 25))


@v("math", "mid")
def m_mixed_four_ops(rng):
    val, s = _start(rng, 2000, 9000)
    for _ in range(rng.randint(4, 8)):
        r = rng.random()
        if r < 0.25:
            t, val, _ = _ship(rng, val)
        elif r < 0.5:
            t, val, _ = _recv(rng, val)
        elif r < 0.75:
            t, val = _split(rng, val)
        else:
            t, val = _double(rng, val)
        s.append(t)
    return M(s, val)


@v("math", "mid")
def m_big_numbers(rng):
    val, s = _start(rng, 10 ** 14, 10 ** 15)
    return M(s, _sd_chain(rng, val, s, 6))


@v("math", "mid")
def m_reaches_zero(rng):
    val, s = _start(rng, 2, 9)
    t, val = _split(rng, val, 10)
    s.append(t)
    t, val = _double(rng, val)
    s.append(t)
    t, val, _ = _recv(rng, val)
    s.append(t)
    return M(s, val)


@v("math", "mid", "amb")
def m_negative_stock(rng):
    val, s = _start(rng, 50, 120)
    a, d = rng.randint(40, 60), rng.randint(4, 6)
    s.append(f"It ships {a} boxes per day for {d} days.")
    val -= a * d
    t, val, _ = _recv(rng, val)
    s.append(t)
    return M(s, val)


@v("math", "easy")
def m_singular_one(rng):
    val, s = _start(rng)
    a, m = rng.randint(5, 40), rng.randint(20, 90)
    s += [f"It ships {a} boxes per day for 1 day.", f"It receives 1 delivery of {m} boxes."]
    return M(s, val - a + m)


@v("math", "easy")
def m_naive_plural_one(rng):
    val, s = _start(rng)
    a, m = rng.randint(5, 40), rng.randint(20, 90)
    s += [f"It ships {a} boxes per day for 1 days.", f"It receives 1 deliveries of {m} boxes each."]
    return M(s, val - a + m)


@v("math", "easy")
def m_then_ship_receive(rng):
    val, s = _start(rng)
    t, val, _ = _ship(rng, val, "Then it")
    s.append(t)
    t, val, _ = _recv(rng, val, "Then it")
    s.append(t)
    return M(s, val)


@v("math", "easy")
def m_split_double_no_then(rng):
    val, s = _start(rng)
    t, val = _split(rng, val, pre="The")
    s.append(t)
    t, val = _double(rng, val, pre="The")
    s.append(t)
    return M(s, val)


@v("math", "easy")
def m_connectors(rng):
    val, s = _start(rng)
    t, val, _ = _ship(rng, val, rng.choice(["Next, it", "After that, it", "First, it", "Afterwards, it"]))
    s.append(t)
    t, val = _split(rng, val, pre=rng.choice(["Next, the", "After that, the", "Later, the"]))
    s.append(t)
    t, val, _ = _recv(rng, val, rng.choice(["Finally, it", "Lastly, it"]))
    s.append(t)
    return M(s, val)


@v("math", "easy")
def m_thousands_separator(rng):
    val = rng.randint(1000, 9999)
    s = [f"A warehouse starts with {val:,} boxes."]
    return M(s, _sr_chain(rng, val, s, 2))


@v("math", "mid")
def m_number_words(rng):
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    val, s = _start(rng)
    k = rng.randint(2, 4)
    d = rng.randint(2, 6)
    a = rng.randint(5, 30)
    s.append(f"Then the stock is split equally into {words[k]} warehouses and it keeps one share (rounded down).")
    val //= k
    s.append(f"It ships {a} boxes per day for {words[d]} days.")
    return M(s, val - a * d)


@v("math", "easy")
def m_other_entity(rng):
    place, item = rng.choice([("library", "books"), ("bakery", "loaves"), ("farm", "eggs"), ("depot", "pallets"),
                              ("shop", "items"), ("store", "crates"), ("factory", "parts")])
    val = rng.randint(800, 2400)
    a, d, k, m = rng.randint(5, 30), rng.randint(2, 6), rng.randint(2, 5), rng.randint(20, 90)
    s = [f"A {place} starts with {val} {item}.", f"It ships {a} {item} per day for {d} days.",
         f"It receives {k} deliveries of {m} {item} each."]
    return M(s, val - a * d + k * m, q=f"How many {item} does it have at the end?")


@v("math", "easy")
def m_other_entity_split(rng):
    place, item = rng.choice([("store", "crates"), ("depot", "pallets"), ("library", "books")])
    val = rng.randint(800, 2400)
    k = rng.randint(2, 4)
    s = [f"A {place} starts with {val} {item}.",
         f"Then the stock is split equally into {k} {place}s and it keeps one share (rounded down).",
         f"Then the stock is doubled by merging with a sister {place}."]
    return M(s, val // k * 2, q=f"How many {item} does it have at the end?")


@v("math", "mid")
def m_tripled_merge(rng):
    val, s = _start(rng)
    s.append("Then the stock is tripled by merging with two sister warehouses.")
    t, val2 = _split(rng, val * 3)
    s.append(t)
    return M(s, val2)


@v("math", "mid")
def m_halved(rng):
    val, s = _start(rng)
    s.append("Then the stock is halved (rounded down).")
    t, val2, _ = _recv(rng, val // 2)
    s.append(t)
    return M(s, val2)


@v("math", "mid")
def m_split_rounded_up(rng):
    val, s = _start(rng)
    k = rng.choice([3, 7, 9])
    while val % k == 0:
        val += 1
    s[0] = f"A warehouse starts with {val} boxes."
    s.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded up).")
    return M(s, -(-val // k))


@v("math", "mid")
def m_split_exact_no_rounding(rng):
    k = rng.randint(2, 5)
    val = rng.randint(200, 600) * k
    s = [f"A warehouse starts with {val} boxes.",
         f"Then the stock is split equally into {k} warehouses and it keeps one share."]
    t, val2 = _double(rng, val // k)
    s.append(t)
    return M(s, val2)


@v("math", "hard")
def m_split_keep_two_shares(rng):
    val, s = _start(rng)
    k = rng.randint(3, 5)
    s.append(f"Then the stock is split equally into {k} warehouses and it keeps two shares (each share rounded down).")
    return M(s, 2 * (val // k))


@v("math", "hard")
def m_split_keep_remainder(rng):
    val, s = _start(rng)
    k = rng.randint(3, 9)
    s.append(f"Then the stock is split equally into {k} warehouses and it keeps only the boxes left over.")
    return M(s, val % k)


@v("math", "mid")
def m_percent_sold(rng):
    val, s = _start(rng)
    p = rng.choice([10, 15, 20, 25, 30, 50])
    s.append(f"It sells {p}% of its boxes (the number sold is rounded down).")
    val -= val * p // 100
    t, val, _ = _recv(rng, val)
    s.append(t)
    return M(s, val)


@v("math", "hard")
def m_percent_growth(rng):
    val, s = _start(rng)
    p = rng.choice([10, 15, 20, 25])
    s.append(f"Then the stock grows by {p} percent (the increase is rounded down).")
    return M(s, val + val * p // 100)


@v("math", "hard")
def m_fraction_shipped(rng):
    val, s = _start(rng)
    word, k = rng.choice([("half", 2), ("a third", 3), ("a quarter", 4)])
    s.append(f"It ships {word} of its boxes (the number shipped is rounded down).")
    return M(s, val - val // k)


@v("math", "mid")
def m_compound_and(rng):
    val, s = _start(rng)
    a, d, k, m = rng.randint(5, 30), rng.randint(2, 6), rng.randint(2, 5), rng.randint(20, 90)
    s.append(f"It ships {a} boxes per day for {d} days and receives {k} deliveries of {m} boxes each.")
    return M(s, val - a * d + k * m)


@v("math", "mid")
def m_per_week(rng):
    val, s = _start(rng)
    a, d = rng.randint(5, 40), rng.randint(2, 6)
    unit = rng.choice(["week", "hour", "month"])
    s.append(f"It ships {a} boxes per {unit} for {d} {unit}s.")
    return M(s, val - a * d)


@v("math", "mid")
def m_passive_voice(rng):
    val, s = _start(rng)
    a, d, k, m = rng.randint(5, 30), rng.randint(2, 6), rng.randint(2, 5), rng.randint(20, 90)
    s += [f"{a} boxes are shipped per day for {d} days.", f"{k} deliveries of {m} boxes each arrive."]
    return M(s, val - a * d + k * m)


@v("math", "easy")
def m_plain_add_sub(rng):
    val, s = _start(rng)
    a, b = rng.randint(20, 300), rng.randint(20, 300)
    s += [f"It receives {a} boxes.", f"It ships {b} boxes."]
    return M(s, val + a - b)


@v("math", "mid")
def m_q_total_shipped(rng):
    val, s = _start(rng, 3000, 6000)
    t1, val, x1 = _ship(rng, val)
    t2, val, _ = _recv(rng, val)
    t3, val, x3 = _ship(rng, val)
    return M(s + [t1, t2, t3], x1 + x3, q="How many boxes did it ship in total?")


@v("math", "mid")
def m_q_total_received(rng):
    val, s = _start(rng, 3000, 6000)
    t1, val, x1 = _recv(rng, val)
    t2, val, _ = _ship(rng, val)
    t3, val, x3 = _recv(rng, val)
    return M(s + [t1, t2, t3], x1 + x3, q="How many boxes did it receive in total?")


@v("math", "hard")
def m_q_net_gain(rng):
    val0, s = _start(rng)
    t1, val, _ = _recv(rng, val0)
    t2, val, _ = _recv(rng, val)
    return M(s + [t1, t2], val - val0, q="How many more boxes does it have at the end than at the start?")


@v("math", "hard")
def m_q_remainder(rng):
    val, s = _start(rng)
    val = _sr_chain(rng, val, s, 2)
    k = rng.choice([7, 9, 11])
    return M(s, val % k, q=f"What is the remainder when the final number of boxes is divided by {k}?")


@v("math", "hard")
def m_q_trucks(rng):
    val, s = _start(rng)
    val = _sr_chain(rng, val, s, 2)
    c = rng.choice([30, 40, 48])
    return M(s, -(-val // c), q=f"At the end all boxes are loaded onto trucks that hold {c} boxes each. "
                                 f"How many trucks are needed?")


@v("math", "easy")
def m_q_paraphrase(rng):
    val, s = _start(rng)
    val = _sr_chain(rng, val, s, 2)
    return M(s, val, q=rng.choice([
        "How many boxes are in the warehouse at the end?", "How many boxes does the warehouse have at the end?",
        "What is the final number of boxes?", "How many boxes remain?", "How many boxes are left at the end?",
        "How many boxes does it have now?", "What is the final stock?", "How many boxes does it end up with?",
        "How many boxes will it have at the end?", "Find the number of boxes at the end.",
    ]))


@v("math", "mid")
def m_distractor_number(rng):
    val, s = _start(rng)
    s.append(f"The warehouse has {rng.randint(5, 40)} employees.")
    val = _sr_chain(rng, val, s, 2)
    return M(s, val)


@v("math", "mid")
def m_distractor_sentence(rng):
    val, s = _start(rng)
    t, val, _ = _ship(rng, val)
    s += [t, "The manager is pleased with the result."]
    t, val, _ = _recv(rng, val)
    s.append(t)
    return M(s, val)


@v("math", "hard")
def m_two_warehouses(rng):
    a, b = rng.randint(800, 2400), rng.randint(800, 2400)
    x, d = rng.randint(5, 30), rng.randint(2, 6)
    k, m = rng.randint(2, 5), rng.randint(20, 90)
    s = [f"Warehouse A starts with {a} boxes.", f"Warehouse B starts with {b} boxes.",
         f"Warehouse A ships {x} boxes per day for {d} days.", f"Warehouse B receives {k} deliveries of {m} boxes each."]
    return M(s, a - x * d + b + k * m, q="How many boxes do the two warehouses have in total at the end?")


@v("math", "easy")
def m_extra_whitespace(rng):
    val, s = _start(rng)
    val = _sr_chain(rng, val, s, 2)
    spec = M(s, val)
    spec["question"] = spec["question"].replace(". ", ".  ").replace("boxes per", "boxes  per")
    return spec


@v("math", "easy")
def m_different_tail(rng):
    val, s = _start(rng)
    val = _sd_chain(rng, val, s, 3)
    spec = M(s, val)
    spec["instruction"] = spec["question"] + rng.choice([
        " Write ONLY the final answer to the file answer.txt in the current directory.",
        " Give the answer as a single integer in answer.txt.",
        " Use python to calculate the result. Save only the number to answer.txt.",
        "",
    ])
    return spec


@v("math", "easy")
def m_leading_prefix(rng):
    val, s = _start(rng)
    val = _sr_chain(rng, val, s, 2)
    spec = M(s, val)
    spec["question"] = rng.choice(["Solve the following problem. ", "Problem: ", "Task: "]) + spec["question"]
    return spec


@v("math", "extreme")
def m_x_pure_expression(rng):
    a, b, c, d = rng.randint(100, 999), rng.randint(2, 9), rng.randint(2, 9), rng.randint(10, 99)
    return dict(question=f"What is (({a} // {b}) * {c}) + {d}?", expected=str(a // b * c + d))


@v("math", "extreme")
def m_x_sum_range(rng):
    n = rng.randint(50, 500)
    return dict(question=f"What is the sum of all integers from 1 to {n} inclusive?", expected=str(n * (n + 1) // 2))


@v("math", "extreme")
def m_x_count_primes(rng):
    n = rng.randint(50, 400)
    cnt = sum(1 for i in range(2, n) if all(i % j for j in range(2, int(i ** 0.5) + 1)))
    return dict(question=f"How many prime numbers are there below {n}?", expected=str(cnt))


@v("math", "extreme")
def m_x_gcd(rng):
    g = rng.randint(2, 40)
    a, b = g * rng.randint(10, 99), g * rng.randint(10, 99)
    return dict(question=f"What is the greatest common divisor of {a} and {b}?", expected=str(math.gcd(a, b)))


@v("math", "extreme")
def m_x_modpow(rng):
    a, b, m = rng.randint(2, 20), rng.randint(20, 90), rng.randint(100, 999)
    return dict(question=f"What is {a} to the power of {b} modulo {m}?", expected=str(pow(a, b, m)))


@v("math", "extreme")
def m_x_speed_distance(rng):
    sp, h = rng.randint(40, 120), rng.randint(2, 9)
    return dict(question=f"A truck drives at {sp} km per hour for {h} hours. How many kilometers does it travel?",
                expected=str(sp * h))


@v("math", "extreme")
def m_x_digit_sum(rng):
    n = rng.randint(10 ** 8, 10 ** 12)
    return dict(question=f"What is the sum of the digits of {n}?", expected=str(sum(map(int, str(n)))))


@v("math", "extreme")
def m_x_days_until_empty(rng):
    val, s = _start(rng)
    a = rng.randint(20, 60)
    return dict(question=f"A warehouse starts with {val} boxes. It ships {a} boxes per day. After how many full days "
                         f"does it have fewer than {a} boxes left?", expected=str(val // a))


@v("math", "extreme")
def m_x_russian(rng):
    val = rng.randint(800, 2400)
    a, d, k, m = rng.randint(5, 30), rng.randint(2, 6), rng.randint(2, 5), rng.randint(20, 90)
    return dict(question=f"На складе {val} коробок. Склад отгружает {a} коробок в день в течение {d} дней. "
                         f"Затем он получает {k} поставок по {m} коробок. Сколько коробок на складе в конце?",
                expected=str(val - a * d + k * m))


# --- math: невыполнимые


@v("math", "base", "inf")
def m_i_missing_number(rng):
    val, s = _start(rng)
    s.append(f"It ships {rng.choice(['some', 'several', 'a number of', 'an unknown number of'])} boxes per day "
             f"for {rng.randint(2, 6)} days.")
    t, _, _ = _recv(rng, val)
    return M(s + [t], None)


@v("math", "base", "inf")
def m_i_missing_days(rng):
    val, s = _start(rng)
    s.append(f"It ships {rng.randint(5, 40)} boxes per day for {rng.choice(['several', 'some', 'a few'])} days.")
    return M(s, None)


@v("math", "base", "inf")
def m_i_missing_start(rng):
    s = [f"A warehouse starts with {rng.choice(['some', 'several', 'a number of'])} boxes."]
    t, _, _ = _ship(rng, 5000)
    return M(s + [t], None)


@v("math", "mid", "inf")
def m_i_no_start_sentence(rng):
    t, _, _ = _ship(rng, 5000, "A warehouse")
    t2, _, _ = _recv(rng, 5000)
    return M([t, t2], None)


@v("math", "mid", "inf")
def m_i_variable_letter(rng):
    val, s = _start(rng)
    s.append(f"It ships {rng.choice(['X', 'N', 'k'])} boxes per day for {rng.randint(2, 6)} days.")
    return M(s, None)


@v("math", "base", "inf")
def m_i_unknown_quantity(rng):
    val, s = _start(rng)
    t, _, _ = _ship(rng, val)
    return M(s + [t], None, q=rng.choice(["How many trucks does it own at the end?",
                                         "How many employees does it have at the end?",
                                         "How many pallets does it have at the end?"]))


@v("math", "mid", "inf")
def m_i_zero_split(rng):
    val, s = _start(rng)
    s.append("Then the stock is split equally into 0 warehouses and it keeps one share (rounded down).")
    return M(s, None)


@v("math", "mid", "inf")
def m_i_unrelated_question(rng):
    val, s = _start(rng)
    t, _, _ = _ship(rng, val)
    return M(s + [t], None, q=rng.choice(["What color are the boxes?", "Who is the manager of the warehouse?",
                                         "In which city is the warehouse located?",
                                         "How much does one box weigh?"]))


@v("math", "mid", "inf")
def m_i_no_question(rng):
    val, s = _start(rng)
    t, _, _ = _ship(rng, val)
    return M(s + [t], None, q="")


@v("math", "mid", "inf")
def m_i_external_reference(rng):
    val, s = _start(rng)
    s.append("It ships as many boxes as listed in the file shipments.txt.")
    return M(s, None)


@v("math", "mid", "inf")
def m_i_unknown_split(rng):
    val, s = _start(rng)
    s.append(f"Then the stock is split equally into {rng.choice(['several', 'some'])} warehouses and it keeps "
             f"one share (rounded down).")
    return M(s, None)


@v("math", "mid", "inf")
def m_i_range(rng):
    val, s = _start(rng)
    s.append(f"It ships between 10 and 20 boxes per day for {rng.randint(2, 6)} days.")
    return M(s, None)


@v("math", "mid", "inf")
def m_i_deliveries_no_size(rng):
    val, s = _start(rng)
    s.append(f"It receives {rng.randint(2, 6)} deliveries.")
    return M(s, None)


@v("math", "mid", "amb")
def m_a_about(rng):
    val, s = _start(rng)
    s.append(f"It ships about {rng.randint(5, 40)} boxes per day for {rng.randint(2, 6)} days.")
    return M(s, None)


@v("math", "mid", "amb")
def m_a_nonint_no_rule(rng):
    val, s = _start(rng)
    while val % 7 == 0:
        val += 1
    s = [f"A warehouse starts with {val} boxes.",
         "Then the stock is split equally into 7 warehouses and it keeps one share."]
    return M(s, val // 7)


# =========================================================================== SEARCH
def build_kb(rng, n=None, companies=None):
    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], n or rng.randint(24, 30))
    comp_names = companies or COMPANIES
    cities = [rng.choice(CITIES) for _ in comp_names]
    comps = {c: {"Headquarters": cities[i], "Founded": rng.randint(1950, 2020)} for i, c in enumerate(comp_names)}
    people = {}
    for name in names:
        people[name] = {"Born": rng.randint(1960, 2002), "Employer": rng.choice(comp_names),
                        "Mentor": rng.choice([p for p in names if p != name])}
    return people, comps


def _slug(name):
    return name.lower().replace(" ", "_")


def kb_files(people, comps, fmt="md"):
    files = {}
    def body(fields, style):
        if style == "bullets":
            return "".join(f"- {k}: {val}\n" for k, val in fields.items())
        if style == "bold":
            return "".join(f"**{k}:** {val}\n" for k, val in fields.items())
        if style == "bold2":
            return "".join(f"**{k}**: {val}\n" for k, val in fields.items())
        if style == "lower":
            return "".join(f"{k.lower()}: {val}\n" for k, val in fields.items())
        if style == "equals":
            return "".join(f"{k} = {val}\n" for k, val in fields.items())
        if style == "spaced":
            return "".join(f"{k} : {val}  \n" for k, val in fields.items())
        return "".join(f"{k}: {val}\n" for k, val in fields.items())
    for kind, table in (("person", people), ("company", comps)):
        for name, fields in table.items():
            base = f"kb/{kind}_{_slug(name)}"
            if fmt == "yaml":
                files[base + ".md"] = "---\n" + f"name: {name}\n" + body(fields, "lower") + "---\n"
            elif fmt == "notitle":
                files[base + ".md"] = body(fields, "md")
            elif fmt == "txt":
                files[base + ".txt"] = f"# {name}\n\n" + body(fields, "md")
            elif fmt == "nested":
                sub = "people" if kind == "person" else "companies"
                files[f"kb/{sub}/{_slug(name)}.md"] = f"# {name}\n\n" + body(fields, "md")
            elif fmt == "json":
                files[base + ".json"] = json.dumps({"name": name, **fields}, indent=2) + "\n"
            elif fmt == "crlf":
                files[base + ".md"] = (f"# {name}\n\n" + body(fields, "md")).replace("\n", "\r\n")
            elif fmt == "h2":
                files[base + ".md"] = f"## {name}\n\n" + body(fields, "md")
            elif fmt == "name_field":
                files[base + ".md"] = f"Name: {name}\n" + body(fields, "md")
            elif fmt == "prose":
                if kind == "person":
                    files[base + ".md"] = (f"# {name}\n\n{name} was born in {fields['Born']}. {name.split()[0]} works at "
                                           f"{fields['Employer']}. The mentor of {name.split()[0]} is {fields['Mentor']}.\n")
                else:
                    files[base + ".md"] = (f"# {name}\n\n{name} is headquartered in {fields['Headquarters']}. "
                                           f"It was founded in {fields['Founded']}.\n")
            elif fmt == "synonyms":
                ren = {"Born": "Year of birth", "Employer": "Works at", "Mentor": "Mentored by",
                       "Headquarters": "HQ", "Founded": "Established"}
                files[base + ".md"] = f"# {name}\n\n" + body({ren[k]: val for k, val in fields.items()}, "md")
            elif fmt == "extra":
                extra = dict(fields)
                if kind == "person":
                    extra = {"Role": "Engineer", **fields, "Email": f"{_slug(name)}@example.com", "Nationality": "N/A"}
                else:
                    extra = {"Industry": "Technology", **fields, "Employees": 1200}
                files[base + ".md"] = f"# {name}\n\nSome notes about {name}.\n\n" + body(extra, "md") + "\nSee also: index.\n"
            else:
                files[base + ".md"] = f"# {name}\n\n" + body(fields, fmt)
    return files


def SQ(q, expected, people, comps, fmt="md", **kw):
    return dict(question=KB_INTRO + q, expected=None if expected is None else str(expected),
                files=kb_files(people, comps, fmt), **kw)


def _pick(rng, people):
    return rng.choice(list(people))


def _hq(p, people, comps):
    return comps[people[p]["Employer"]]["Headquarters"]


@v("search", "base")
def s_base_born(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    return SQ(f"In which year was {p} born?", people[p]["Born"], people, comps)


@v("search", "base")
def s_base_employer_hq(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    return SQ(f"In which city is the headquarters of {p}'s employer?", _hq(p, people, comps), people, comps)


def _one_hop(name, templates, getter, level="easy"):
    def fn(rng):
        people, comps = build_kb(rng)
        p = _pick(rng, people)
        c = people[p]["Employer"]
        return SQ(rng.choice(templates).format(p=p, c=c), getter(p, c, people, comps), people, comps)
    fn.__name__ = name
    v("search", level)(fn)

_one_hop("s_employer", ["Which company does {p} work for?", "Who is the employer of {p}?", "What is {p}'s employer?",
                        "Where does {p} work?", "Who is {p}'s employer?", "Which company employs {p}?",
                        "What company is {p} employed by?"],
         lambda p, c, P, C: P[p]["Employer"])
_one_hop("s_mentor", ["Who is the mentor of {p}?", "Who is {p}'s mentor?", "Who mentors {p}?",
                      "Who was {p} mentored by?", "What is the name of {p}'s mentor?"],
         lambda p, c, P, C: P[p]["Mentor"])
_one_hop("s_company_hq", ["In which city is the headquarters of {c}?", "Where is {c} headquartered?",
                          "In which city is {c} headquartered?", "What is the headquarters of {c}?",
                          "Where are the headquarters of {c} located?"],
         lambda p, c, P, C: C[c]["Headquarters"])
_one_hop("s_company_founded", ["In which year was {c} founded?", "When was {c} founded?",
                               "What year was {c} founded?", "What is the founding year of {c}?"],
         lambda p, c, P, C: C[c]["Founded"])
_one_hop("s_born_paraphrase", ["What year was {p} born?", "When was {p} born?", "What is the birth year of {p}?",
                               "What is {p}'s year of birth?", "{p} was born in which year?",
                               "In what year was {p} born?", "Find the year in which {p} was born."],
         lambda p, c, P, C: P[p]["Born"])
_one_hop("s_employer_hq_paraphrase",
         ["Where is {p}'s employer headquartered?", "In which city is {p}'s employer headquartered?",
          "Which city hosts the headquarters of the company {p} works for?",
          "In what city is {p}'s employer based?", "What is the headquarters city of {p}'s employer?",
          "In which city is the headquarters of the company that employs {p}?",
          "{p}'s employer has its headquarters in which city?",
          "In which city is the headquarters of the employer of {p}?"],
         lambda p, c, P, C: C[c]["Headquarters"], "mid")
_one_hop("s_employer_founded", ["In which year was {p}'s employer founded?",
                                "In which year was the employer of {p} founded?",
                                "When was {p}'s employer founded?",
                                "In which year was the company {p} works for founded?"],
         lambda p, c, P, C: C[c]["Founded"], "mid")
_one_hop("s_mentor_born", ["In which year was {p}'s mentor born?", "In which year was the mentor of {p} born?",
                           "When was {p}'s mentor born?"],
         lambda p, c, P, C: P[P[p]["Mentor"]]["Born"], "mid")
_one_hop("s_mentor_employer", ["Which company does {p}'s mentor work for?", "Who is the employer of {p}'s mentor?",
                               "What is the employer of the mentor of {p}?"],
         lambda p, c, P, C: P[P[p]["Mentor"]]["Employer"], "mid")
_one_hop("s_mentor_mentor", ["Who is the mentor of {p}'s mentor?", "Who is {p}'s mentor's mentor?",
                             "Who is the mentor of the mentor of {p}?"],
         lambda p, c, P, C: P[P[p]["Mentor"]]["Mentor"], "mid")
_one_hop("s_3hop_mentor_employer_hq",
         ["In which city is the headquarters of {p}'s mentor's employer?",
          "In which city is the headquarters of the employer of {p}'s mentor?",
          "In which city is the headquarters of the employer of the mentor of {p}?"],
         lambda p, c, P, C: C[P[P[p]["Mentor"]]["Employer"]]["Headquarters"], "hard")
_one_hop("s_3hop_mentor_employer_founded",
         ["In which year was {p}'s mentor's employer founded?",
          "In which year was the employer of {p}'s mentor founded?"],
         lambda p, c, P, C: C[P[P[p]["Mentor"]]["Employer"]]["Founded"], "hard")
_one_hop("s_3hop_mentor_mentor_born", ["In which year was {p}'s mentor's mentor born?",
                                       "In which year was the mentor of the mentor of {p} born?"],
         lambda p, c, P, C: P[P[P[p]["Mentor"]]["Mentor"]]["Born"], "hard")
_one_hop("s_4hop", ["In which city is the headquarters of {p}'s mentor's mentor's employer?",
                    "In which city is the headquarters of the employer of the mentor of the mentor of {p}?"],
         lambda p, c, P, C: C[P[P[P[p]["Mentor"]]["Mentor"]]["Employer"]]["Headquarters"], "hard")
_one_hop("s_5hop", ["In which city is the headquarters of {p}'s mentor's mentor's mentor's employer?"],
         lambda p, c, P, C: C[P[P[P[P[p]["Mentor"]]["Mentor"]]["Mentor"]]["Employer"]]["Headquarters"], "hard")
_one_hop("s_city_person_works", ["In which city does {p} work?", "In which city is {p}'s workplace headquartered?"],
         lambda p, c, P, C: C[c]["Headquarters"], "hard")


@v("search", "mid")
def s_lowercase_name_real(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    return SQ(f"In which year was {p.lower()} born?", people[p]["Born"], people, comps)


@v("search", "mid")
def s_name_reversed(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    f, l = p.split()
    return SQ(f"In which year was {l}, {f} born?", people[p]["Born"], people, comps)


@v("search", "mid")
def s_count_employees(rng):
    people, comps = build_kb(rng)
    c = rng.choice(list(comps))
    n = sum(1 for x in people.values() if x["Employer"] == c)
    return SQ(rng.choice([f"How many people work at {c}?", f"How many people in the knowledge base work for {c}?",
                          f"How many employees does {c} have in the knowledge base?"]), n, people, comps)


@v("search", "hard")
def s_count_born_before(rng):
    people, comps = build_kb(rng)
    y = rng.randint(1970, 1995)
    n = sum(1 for x in people.values() if x["Born"] < y)
    return SQ(f"How many people in the knowledge base were born before {y}?", n, people, comps)


@v("search", "hard")
def s_count_mentees(rng):
    people, comps = build_kb(rng)
    m = people[_pick(rng, people)]["Mentor"]
    n = sum(1 for x in people.values() if x["Mentor"] == m)
    return SQ(f"How many people have {m} as their mentor?", n, people, comps)


@v("search", "hard")
def s_count_companies_city(rng):
    people, comps = build_kb(rng)
    city = comps[rng.choice(list(comps))]["Headquarters"]
    n = sum(1 for x in comps.values() if x["Headquarters"] == city)
    return SQ(f"How many companies have their headquarters in {city}?", n, people, comps)


@v("search", "hard")
def s_oldest_person(rng):
    while True:
        people, comps = build_kb(rng)
        years = sorted(x["Born"] for x in people.values())
        if years[0] != years[1] and years[-1] != years[-2]:
            break
    if rng.random() < 0.5:
        return SQ("Who is the oldest person in the knowledge base?", min(people, key=lambda p: people[p]["Born"]),
                  people, comps)
    return SQ("Who is the youngest person in the knowledge base?", max(people, key=lambda p: people[p]["Born"]),
              people, comps)


@v("search", "hard")
def s_oldest_company(rng):
    while True:
        people, comps = build_kb(rng)
        years = sorted(x["Founded"] for x in comps.values())
        if years[0] != years[1]:
            break
    return SQ(rng.choice(["Which company was founded first?", "Which company in the knowledge base is the oldest?"]),
              min(comps, key=lambda c: comps[c]["Founded"]), people, comps)


@v("search", "mid")
def s_compare_born(rng):
    people, comps = build_kb(rng)
    a = _pick(rng, people)
    b = rng.choice([p for p in people if people[p]["Born"] != people[a]["Born"]])
    older = a if people[a]["Born"] < people[b]["Born"] else b
    younger = b if older == a else a
    q, e = rng.choice([(f"Who was born earlier, {a} or {b}?", older), (f"Who is older, {a} or {b}?", older),
                       (f"Who is younger, {a} or {b}?", younger), (f"Who was born later, {a} or {b}?", younger),
                       (f"Who is older: {a} or {b}?", older), (f"Between {a} and {b}, who is older?", older)])
    return SQ(q, e, people, comps)


@v("search", "hard")
def s_compare_companies(rng):
    people, comps = build_kb(rng)
    a = rng.choice(list(comps))
    b = rng.choice([c for c in comps if comps[c]["Founded"] != comps[a]["Founded"]])
    return SQ(f"Which company was founded earlier, {a} or {b}?",
              a if comps[a]["Founded"] < comps[b]["Founded"] else b, people, comps)


@v("search", "hard")
def s_inverse_mentee(rng):
    while True:
        people, comps = build_kb(rng)
        by = {}
        for n, f in people.items():
            by.setdefault(f["Mentor"], []).append(n)
        uniq = [(m, x[0]) for m, x in by.items() if len(x) == 1]
        if uniq:
            break
    m, mentee = rng.choice(uniq)
    return SQ(rng.choice([f"Who is mentored by {m}?", f"Whose mentor is {m}?", f"Who has {m} as a mentor?",
                          f"Which person is mentored by {m}?"]), mentee, people, comps)


@v("search", "hard")
def s_inverse_born_year(rng):
    while True:
        people, comps = build_kb(rng)
        years = [x["Born"] for x in people.values()]
        uniq = [p for p in people if years.count(people[p]["Born"]) == 1]
        if uniq:
            break
    p = rng.choice(uniq)
    return SQ(f"Who was born in {people[p]['Born']}?", p, people, comps)


@v("search", "hard")
def s_inverse_company_city(rng):
    while True:
        people, comps = build_kb(rng)
        cities = [x["Headquarters"] for x in comps.values()]
        uniq = [c for c in comps if cities.count(comps[c]["Headquarters"]) == 1]
        if uniq:
            break
    c = rng.choice(uniq)
    return SQ(f"Which company is headquartered in {comps[c]['Headquarters']}?", c, people, comps)


@v("search", "hard")
def s_age_difference(rng):
    people, comps = build_kb(rng)
    a = _pick(rng, people)
    b = rng.choice([p for p in people if people[p]["Born"] > people[a]["Born"]] or [a])
    if a == b:
        return s_age_difference(rng)
    return SQ(f"How many years older is {a} than {b}?", people[b]["Born"] - people[a]["Born"], people, comps)


@v("search", "hard")
def s_age_in_year(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    y = rng.randint(2010, 2024)
    return SQ(f"How old did {p} turn in {y}?", y - people[p]["Born"], people, comps)


@v("search", "extreme")
def s_x_same_employer_yesno(rng):
    people, comps = build_kb(rng)
    a, b = rng.sample(list(people), 2)
    return SQ(f"Do {a} and {b} work for the same company? Answer yes or no.",
              "yes" if people[a]["Employer"] == people[b]["Employer"] else "no", people, comps)


@v("search", "extreme")
def s_x_list_employees(rng):
    people, comps = build_kb(rng)
    c = rng.choice(list(comps))
    names = sorted(p for p in people if people[p]["Employer"] == c)
    if not names:
        return s_x_list_employees(rng)
    return SQ(f"List everyone who works at {c}, sorted alphabetically and separated by commas.", ", ".join(names),
              people, comps)


@v("search", "extreme")
def s_x_most_employees(rng):
    while True:
        people, comps = build_kb(rng)
        cnt = {c: sum(1 for x in people.values() if x["Employer"] == c) for c in comps}
        top = sorted(cnt.values())
        if top[-1] != top[-2]:
            break
    return SQ("Which company has the most employees in the knowledge base?", max(cnt, key=cnt.get), people, comps)


@v("search", "extreme")
def s_x_people_in_city(rng):
    people, comps = build_kb(rng)
    city = comps[rng.choice(list(comps))]["Headquarters"]
    n = sum(1 for p in people if _hq(p, people, comps) == city)
    return SQ(f"How many people work for a company headquartered in {city}?", n, people, comps)


def _fmt_variant(fmt, level="mid", kind="ok"):
    def fn(rng):
        people, comps = build_kb(rng)
        p = _pick(rng, people)
        if rng.random() < 0.5:
            return SQ(f"In which year was {p} born?", people[p]["Born"], people, comps, fmt)
        return SQ(f"In which city is the headquarters of {p}'s employer?", _hq(p, people, comps), people, comps, fmt)
    fn.__name__ = f"s_fmt_{fmt}"
    v("search", level, kind)(fn)

for _f, _l in (("bullets", "mid"), ("bold", "mid"), ("bold2", "mid"), ("lower", "mid"), ("equals", "hard"),
               ("spaced", "mid"), ("yaml", "hard"), ("notitle", "hard"), ("txt", "mid"), ("nested", "mid"),
               ("json", "extreme"), ("crlf", "mid"), ("h2", "mid"), ("name_field", "hard"), ("prose", "extreme"),
               ("synonyms", "extreme"), ("extra", "mid")):
    _fmt_variant(_f, _l)


@v("search", "mid")
def s_fmt_full_date(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    year = people[p]["Born"]
    for name in people:
        people[name]["Born"] = f"{people[name]['Born']}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return SQ(f"In which year was {p} born?", year, people, comps)


@v("search", "hard")
def s_big_kb(rng):
    firsts = FIRST + [f + s for f in FIRST for s in ("a", "o")]
    lasts = LAST + [l + s for l in LAST for s in ("son", "ez", "ov")]
    names = rng.sample([f"{f} {l}" for f in firsts for l in lasts], 1500)
    comp_names = COMPANIES + [f"{w} {s}" for w in ("Nova", "Orbit", "Pine", "Quartz", "Raven", "Solar", "Tidal", "Umber")
                              for s in ("Labs", "Systems", "Works", "Group")]
    comps = {c: {"Headquarters": rng.choice(CITIES), "Founded": rng.randint(1950, 2020)} for c in comp_names}
    people = {n: {"Born": rng.randint(1960, 2002), "Employer": rng.choice(comp_names), "Mentor": rng.choice(names)}
              for n in names}
    p = rng.choice(names)
    return SQ(f"In which city is the headquarters of {p}'s employer?", _hq(p, people, comps), people, comps)


@v("search", "mid")
def s_extra_files(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    spec = SQ(f"In which city is the headquarters of {p}'s employer?", _hq(p, people, comps), people, comps)
    spec["files"]["kb/README.md"] = "# Knowledge base\n\nOne file per person and per company.\n"
    spec["files"]["kb/index.md"] = "# Index\n\n" + "".join(f"- {n}\n" for n in list(people) + list(comps))
    spec["files"]["notes.txt"] = "Scratch notes. Nothing useful here.\n"
    return spec


@v("search", "hard")
def s_other_dir_name(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    spec = SQ(f"In which year was {p} born?", people[p]["Born"], people, comps)
    spec["files"] = {k.replace("kb/", "docs/", 1): val for k, val in spec["files"].items()}
    spec["question"] = spec["question"].replace("The directory kb/", "The directory docs/")
    return spec


@v("search", "hard")
def s_other_tail(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    spec = SQ(f"In which city is the headquarters of {p}'s employer?", _hq(p, people, comps), people, comps)
    spec["instruction"] = spec["question"] + " Write ONLY the answer to answer.txt."
    return spec


@v("search", "extreme")
def s_x_other_schema(rng):
    """Другая база: книги и авторы."""
    authors = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 8)
    books = {}
    files = {}
    for a in authors:
        files[f"kb/author_{_slug(a)}.md"] = f"# {a}\n\nBorn: {rng.randint(1900, 1980)}\nCountry: {rng.choice(CITIES)}\n"
    titles = rng.sample(["Silent River", "Glass Harbor", "Iron Orchard", "Paper Lantern", "Winter Atlas", "Red Meridian",
                         "Hollow Crown", "Salt Road", "Amber Tide", "Night Ferry"], 8)
    for t in titles:
        books[t] = {"Author": rng.choice(authors), "Published": rng.randint(1950, 2020)}
        files[f"kb/book_{_slug(t)}.md"] = f"# {t}\n\nAuthor: {books[t]['Author']}\nPublished: {books[t]['Published']}\n"
    t = rng.choice(titles)
    if rng.random() < 0.5:
        return dict(question=KB_INTRO + f"In which year was {t} published?", expected=str(books[t]["Published"]), files=files)
    return dict(question=KB_INTRO + f"Who is the author of {t}?", expected=books[t]["Author"], files=files)


# --- search: невыполнимые


def _ghost(rng, people):
    return rng.choice([f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}" not in people])


@v("search", "base", "inf")
def s_i_unknown_person(rng):
    people, comps = build_kb(rng)
    g = _ghost(rng, people)
    return SQ(rng.choice([f"In which year was {g} born?", f"In which city is the headquarters of {g}'s employer?",
                          f"Who is the mentor of {g}?"]), None, people, comps)


@v("search", "mid", "inf")
def s_i_unknown_person_new_name(rng):
    people, comps = build_kb(rng)
    g = rng.choice(["Zara Okafor", "Liam O'Brien", "Yuki Tanaka", "Omar", "Dr. Elena Rossi", "Jean-Luc Moreau"])
    return SQ(f"In which year was {g} born?", None, people, comps)


@v("search", "mid", "inf")
def s_i_unknown_company(rng):
    people, comps = build_kb(rng)
    g = rng.choice(["Zenith Labs", "Kestrel Dynamics", "Lumen Foods", "Vertex", "Nimbus AI"])
    return SQ(rng.choice([f"In which city is the headquarters of {g}?", f"In which year was {g} founded?"]),
              None, people, comps)


@v("search", "base", "inf")
def s_i_missing_field(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    del people[p]["Born"]
    return SQ(f"In which year was {p} born?", None, people, comps)


@v("search", "base", "inf")
def s_i_missing_employer_field(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    del people[p]["Employer"]
    return SQ(f"In which city is the headquarters of {p}'s employer?", None, people, comps)


@v("search", "base", "inf")
def s_i_missing_company_file(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    del comps[people[p]["Employer"]]
    return SQ(f"In which city is the headquarters of {p}'s employer?", None, people, comps)


@v("search", "mid", "inf")
def s_i_missing_hq_field(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    del comps[people[p]["Employer"]]["Headquarters"]
    return SQ(f"In which city is the headquarters of {p}'s employer?", None, people, comps)


@v("search", "mid", "inf")
def s_i_dangling_mentor(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    people[p]["Mentor"] = _ghost(rng, people)
    return SQ(f"In which year was {p}'s mentor born?", None, people, comps)


@v("search", "mid", "inf")
def s_i_no_such_field(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    return SQ(rng.choice([f"What is {p}'s salary?", f"What is the phone number of {p}?",
                          f"In which country was {p} born?", f"What is {p}'s favorite color?",
                          f"Who is {p}'s spouse?", f"How many employees does {p}'s employer have?",
                          f"Who is the CEO of {p}'s employer?"]), None, people, comps)


@v("search", "mid", "inf")
def s_i_empty_value(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    people[p]["Born"] = rng.choice(["unknown", "N/A", "?", "TBD"])
    return SQ(f"In which year was {p} born?", None, people, comps)


@v("search", "mid", "inf")
def s_i_blank_value(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    people[p]["Born"] = ""
    return SQ(f"In which year was {p} born?", None, people, comps)


@v("search", "hard", "inf")
def s_i_first_name_only_ambiguous(rng):
    while True:
        people, comps = build_kb(rng)
        firsts = [p.split()[0] for p in people]
        dup = [f for f in set(firsts) if firsts.count(f) >= 2]
        if dup:
            break
    return SQ(f"In which year was {rng.choice(dup)} born?", None, people, comps)


@v("search", "hard", "inf")
def s_i_conflicting_duplicates(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    spec = SQ(f"In which year was {p} born?", None, people, comps)
    spec["files"][f"kb/person_{_slug(p)}_2.md"] = (f"# {p}\n\nBorn: {people[p]['Born'] + 7}\n"
                                                   f"Employer: {people[p]['Employer']}\nMentor: {people[p]['Mentor']}\n")
    return spec


@v("search", "hard", "inf")
def s_i_conflicting_lines(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    spec = SQ(f"In which year was {p} born?", None, people, comps)
    key = f"kb/person_{_slug(p)}.md"
    spec["files"][key] += f"Born: {people[p]['Born'] + 5}\n"
    return spec


@v("search", "mid", "inf")
def s_i_empty_kb(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    return dict(question=KB_INTRO + f"In which year was {p} born?", expected=None, files={"kb/.gitkeep": ""})


@v("search", "mid", "inf")
def s_i_no_kb_dir(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    return dict(question=KB_INTRO + f"In which year was {p} born?", expected=None, files={})


@v("search", "hard", "inf")
def s_i_wrong_entity_type(rng):
    people, comps = build_kb(rng)
    c = rng.choice(list(comps))
    p = _pick(rng, people)
    return SQ(rng.choice([f"Who is the mentor of {c}?", f"In which year was {c} born?",
                          f"In which year was {p} founded?", f"In which city is the headquarters of {p}?"]),
              None, people, comps)


@v("search", "hard", "inf")
def s_i_empty_person_file(rng):
    people, comps = build_kb(rng)
    p = _pick(rng, people)
    spec = SQ(f"In which year was {p} born?", None, people, comps)
    spec["files"][f"kb/person_{_slug(p)}.md"] = ""
    return spec


@v("search", "hard", "amb")
def s_a_tie_oldest(rng):
    people, comps = build_kb(rng)
    a, b = rng.sample(list(people), 2)
    people[a]["Born"] = people[b]["Born"] = 1950
    return SQ("Who is the oldest person in the knowledge base?", None, people, comps)


@v("search", "hard", "amb")
def s_a_count_zero(rng):
    people, comps = build_kb(rng)
    c = rng.choice(list(comps))
    for x in people.values():
        if x["Employer"] == c:
            x["Employer"] = rng.choice([k for k in comps if k != c])
    return SQ(f"How many people work at {c}?", 0, people, comps)


# =========================================================================== DATA ANALYSIS
def sales(rng, n=None):
    n = n or rng.choice([20, 20, 50, 100, 300, 300])
    return [{"order_id": i + 1, "region": rng.choice(REGIONS), "product": rng.choice(PRODUCTS),
             "month": rng.randint(1, 12), "units": rng.randint(1, 50), "price": rng.randint(5, 99)} for i in range(n)]


def csv_text(rows, columns=SALES_COLUMNS, delim=",", quote=csv.QUOTE_MINIMAL, eol="\n"):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, quoting=quote, lineterminator=eol)
    w.writerow(columns)
    for r in rows:
        w.writerow([r[c] for c in columns])
    return buf.getvalue()


def DQ(q, expected, rows, columns=SALES_COLUMNS, name="sales.csv", text=None, check="exact_int", intro=True,
       nrows=None, files=None, **kw):
    pre = f"The file {name} has columns {','.join(columns)} ({len(rows) if nrows is None else nrows} rows). " if intro else ""
    f = {name: text if text is not None else csv_text(rows, columns)} if files is None else files
    return dict(question=pre + q, expected=None if expected is None else str(expected), files=f, check=check, **kw)


def _pm(rng):
    lo = rng.randint(1, 11)
    return rng.choice(PRODUCTS), lo, min(12, lo + rng.choice([1, 1, 2, 3]))


def _base_q(rng, rows):
    p, lo, hi = _pm(rng)
    return (f"How many units of product '{p}' were sold in months {lo} through {hi} inclusive?",
            sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi))


@v("data_analysis", "base")
def d_base(rng):
    rows = sales(rng)
    return DQ(*_base_q(rng, rows), rows)


@v("data_analysis", "easy")
def d_single_month(rng):
    rows = sales(rng)
    p, m = rng.choice(PRODUCTS), rng.randint(1, 12)
    q = rng.choice([f"How many units of product '{p}' were sold in month {m}?",
                    f"How many units of product '{p}' were sold in months {m} through {m} inclusive?"])
    return DQ(q, sum(r["units"] for r in rows if r["product"] == p and r["month"] == m), rows)


@v("data_analysis", "easy")
def d_product_only(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    q = rng.choice([f"How many units of product '{p}' were sold?", f"How many units of product '{p}' were sold in total?",
                    f"What is the total number of units of product '{p}' sold?"])
    return DQ(q, sum(r["units"] for r in rows if r["product"] == p), rows)


@v("data_analysis", "easy")
def d_total_units(rng):
    rows = sales(rng)
    return DQ(rng.choice(["How many units were sold in total?", "What is the total number of units sold?",
                          "What is the sum of the units column?"]), sum(r["units"] for r in rows), rows)


@v("data_analysis", "easy")
def d_region_months(rng):
    rows = sales(rng)
    _, lo, hi = _pm(rng)
    g = rng.choice(REGIONS)
    return DQ(f"How many units were sold in region '{g}' in months {lo} through {hi} inclusive?",
              sum(r["units"] for r in rows if r["region"] == g and lo <= r["month"] <= hi), rows)


@v("data_analysis", "easy")
def d_region_only(rng):
    rows = sales(rng)
    g = rng.choice(REGIONS)
    return DQ(f"How many units were sold in region '{g}'?", sum(r["units"] for r in rows if r["region"] == g), rows)


@v("data_analysis", "mid")
def d_product_region_months(rng):
    rows = sales(rng)
    p, lo, hi = _pm(rng)
    g = rng.choice(REGIONS)
    return DQ(f"How many units of product '{p}' were sold in region '{g}' in months {lo} through {hi} inclusive?",
              sum(r["units"] for r in rows if r["product"] == p and r["region"] == g and lo <= r["month"] <= hi), rows)


@v("data_analysis", "mid")
def d_months_only(rng):
    rows = sales(rng)
    _, lo, hi = _pm(rng)
    return DQ(f"How many units were sold in months {lo} through {hi} inclusive?",
              sum(r["units"] for r in rows if lo <= r["month"] <= hi), rows)


@v("data_analysis", "mid")
def d_revenue_explicit(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    q = rng.choice([f"What is the total revenue (units times price) of product '{p}'?",
                    f"What is the total revenue (units * price) for product '{p}'?",
                    f"What is the total revenue (units multiplied by price) of product '{p}'?"])
    return DQ(q, sum(r["units"] * r["price"] for r in rows if r["product"] == p), rows)


@v("data_analysis", "mid")
def d_revenue_region_months(rng):
    rows = sales(rng)
    _, lo, hi = _pm(rng)
    g = rng.choice(REGIONS)
    return DQ(f"What is the total revenue (units times price) in region '{g}' in months {lo} through {hi} inclusive?",
              sum(r["units"] * r["price"] for r in rows if r["region"] == g and lo <= r["month"] <= hi), rows)


@v("data_analysis", "hard")
def d_revenue_implicit(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    q = rng.choice([f"What is the total revenue of product '{p}'?", f"How much revenue did product '{p}' generate?",
                    f"What is the total sales value of product '{p}'?",
                    f"What is the sum of units multiplied by price for product '{p}'?"])
    return DQ(q, sum(r["units"] * r["price"] for r in rows if r["product"] == p), rows)


@v("data_analysis", "mid")
def d_count_orders(rng):
    rows = sales(rng)
    p, lo, hi = _pm(rng)
    q, e = rng.choice([
        (f"How many orders of product '{p}' are there?", sum(1 for r in rows if r["product"] == p)),
        (f"How many orders were placed for product '{p}' in months {lo} through {hi} inclusive?",
         sum(1 for r in rows if r["product"] == p and lo <= r["month"] <= hi)),
        (f"How many rows have product '{p}'?", sum(1 for r in rows if r["product"] == p)),
        (f"What is the number of orders for product '{p}'?", sum(1 for r in rows if r["product"] == p)),
    ])
    return DQ(q, e, rows)


@v("data_analysis", "mid")
def d_count_threshold(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    t = rng.randint(10, 40)
    q, e = rng.choice([
        (f"How many orders of product '{p}' have more than {t} units?",
         sum(1 for r in rows if r["product"] == p and r["units"] > t)),
        (f"How many orders have at least {t} units?", sum(1 for r in rows if r["units"] >= t)),
        (f"How many orders have a price above {t}?", sum(1 for r in rows if r["price"] > t)),
        (f"How many orders have units greater than {t}?", sum(1 for r in rows if r["units"] > t)),
    ])
    return DQ(q, e, rows)


@v("data_analysis", "mid")
def d_max_min(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    sel = [r for r in rows if r["product"] == p] or rows
    if sel is rows:
        return d_max_min(rng)
    q, e = rng.choice([
        (f"What is the maximum number of units in a single order of product '{p}'?", max(r["units"] for r in sel)),
        (f"What is the highest price of product '{p}'?", max(r["price"] for r in sel)),
        (f"What is the lowest price of product '{p}'?", min(r["price"] for r in sel)),
        (f"What is the largest units value for product '{p}'?", max(r["units"] for r in sel)),
        (f"What is the minimum units among orders of product '{p}'?", min(r["units"] for r in sel)),
    ])
    return DQ(q, e, rows)


@v("data_analysis", "mid")
def d_avg_floor(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    xs = [r["units"] for r in rows if r["product"] == p]
    if not xs:
        return d_avg_floor(rng)
    q = rng.choice([f"What is the average number of units per order of product '{p}', rounded down to an integer?",
                    f"What is the average units of product '{p}', rounded down to an integer?",
                    f"What is the mean units for product '{p}' (rounded down to the nearest integer)?"])
    return DQ(q, sum(xs) // len(xs), rows)


@v("data_analysis", "hard")
def d_avg_round2(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    xs = [r["price"] for r in rows if r["product"] == p]
    if not xs:
        return d_avg_round2(rng)
    val = f"{sum(xs) / len(xs):.2f}"
    return DQ(f"What is the average price of product '{p}', rounded to 2 decimal places?", val, rows, check="exact_str")


@v("data_analysis", "hard")
def d_avg_nearest_int(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    xs = [r["price"] for r in rows if r["product"] == p]
    if not xs or (sum(xs) * 2) % len(xs) == 0 and (sum(xs) * 2 // len(xs)) % 2 == 1:
        return d_avg_nearest_int(rng)
    return DQ(f"What is the average price of product '{p}', rounded to the nearest integer?",
              round(sum(xs) / len(xs)), rows)


@v("data_analysis", "hard")
def d_argmax_product(rng):
    while True:
        rows = sales(rng)
        tot = {p: sum(r["units"] for r in rows if r["product"] == p) for p in PRODUCTS}
        vals = sorted(tot.values())
        if vals[-1] != vals[-2] and vals[0] != vals[1]:
            break
    q, e = rng.choice([("Which product has the highest total number of units sold?", max(tot, key=tot.get)),
                       ("Which product sold the most units?", max(tot, key=tot.get)),
                       ("Which product sold the fewest units in total?", min(tot, key=tot.get)),
                       ("Which product has the largest total units?", max(tot, key=tot.get))])
    return DQ(q, e, rows, check="exact_str")


@v("data_analysis", "hard")
def d_argmax_region(rng):
    while True:
        rows = sales(rng)
        tot = {g: sum(r["units"] for r in rows if r["region"] == g) for g in REGIONS}
        rev = {g: sum(r["units"] * r["price"] for r in rows if r["region"] == g) for g in REGIONS}
        if len(set(tot.values())) == 4 and len(set(rev.values())) == 4:
            break
    q, e = rng.choice([("Which region sold the most units?", max(tot, key=tot.get)),
                       ("Which region has the highest total revenue (units times price)?", max(rev, key=rev.get)),
                       ("Which region has the lowest total units?", min(tot, key=tot.get))])
    return DQ(q, e, rows, check="exact_str")


@v("data_analysis", "hard")
def d_argmax_month(rng):
    while True:
        rows = sales(rng, rng.choice([100, 300]))
        tot = {m: sum(r["units"] for r in rows if r["month"] == m) for m in range(1, 13)}
        vals = sorted(tot.values())
        if vals[-1] != vals[-2]:
            break
    return DQ("In which month were the most units sold? Answer with the month number.", max(tot, key=tot.get), rows)


@v("data_analysis", "hard")
def d_order_id_of_max(rng):
    while True:
        rows = sales(rng)
        rev = sorted(r["units"] * r["price"] for r in rows)
        if rev[-1] != rev[-2]:
            break
    best = max(rows, key=lambda r: r["units"] * r["price"])
    return DQ("What is the order_id of the order with the highest revenue (units times price)?", best["order_id"], rows)


@v("data_analysis", "hard")
def d_distinct(rng):
    rows = sales(rng, 20)
    g = rng.choice(REGIONS)
    q, e = rng.choice([
        (f"How many distinct products were sold in region '{g}'?", len({r["product"] for r in rows if r["region"] == g})),
        ("How many distinct regions appear in the file?", len({r["region"] for r in rows})),
        (f"In how many different months was something sold in region '{g}'?", len({r["month"] for r in rows if r["region"] == g})),
    ])
    return DQ(q, e, rows)


@v("data_analysis", "hard")
def d_month_before_after(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    m = rng.randint(3, 10)
    q, e = rng.choice([
        (f"How many units of product '{p}' were sold after month {m}?", sum(r["units"] for r in rows if r["product"] == p and r["month"] > m)),
        (f"How many units of product '{p}' were sold before month {m}?", sum(r["units"] for r in rows if r["product"] == p and r["month"] < m)),
        (f"How many units of product '{p}' were sold from month {m} onwards?", sum(r["units"] for r in rows if r["product"] == p and r["month"] >= m)),
        (f"How many units of product '{p}' were sold up to and including month {m}?", sum(r["units"] for r in rows if r["product"] == p and r["month"] <= m)),
    ])
    return DQ(q, e, rows)


@v("data_analysis", "hard")
def d_month_names(rng):
    names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
    rows = sales(rng)
    p, lo, hi = _pm(rng)
    return DQ(f"How many units of product '{p}' were sold from {names[lo - 1]} through {names[hi - 1]} inclusive?",
              sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi), rows)


@v("data_analysis", "hard")
def d_quarter(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    k = rng.randint(1, 4)
    word = ["first", "second", "third", "fourth"][k - 1]
    q = rng.choice([f"How many units of product '{p}' were sold in the {word} quarter (months {3 * k - 2} through {3 * k})?",
                    f"How many units of product '{p}' were sold in the {word} quarter of the year?",
                    f"How many units of product '{p}' were sold in Q{k}?"])
    return DQ(q, sum(r["units"] for r in rows if r["product"] == p and 3 * k - 2 <= r["month"] <= 3 * k), rows)


@v("data_analysis", "hard")
def d_two_products(rng):
    rows = sales(rng)
    a, b = rng.sample(PRODUCTS, 2)
    _, lo, hi = _pm(rng)
    q = rng.choice([f"How many units of products '{a}' and '{b}' were sold in months {lo} through {hi} inclusive?",
                    f"How many units of product '{a}' or '{b}' were sold in months {lo} through {hi} inclusive?"])
    return DQ(q, sum(r["units"] for r in rows if r["product"] in (a, b) and lo <= r["month"] <= hi), rows)


@v("data_analysis", "hard")
def d_negation(rng):
    rows = sales(rng)
    g = rng.choice(REGIONS)
    p = rng.choice(PRODUCTS)
    q, e = rng.choice([
        (f"How many units of product '{p}' were sold outside region '{g}'?", sum(r["units"] for r in rows if r["product"] == p and r["region"] != g)),
        (f"How many units were sold of products other than '{p}'?", sum(r["units"] for r in rows if r["product"] != p)),
        (f"How many units of product '{p}' were sold in regions other than '{g}'?", sum(r["units"] for r in rows if r["product"] == p and r["region"] != g)),
    ])
    return DQ(q, e, rows)


@v("data_analysis", "hard")
def d_price_filter(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    t = rng.randint(30, 70)
    q, e = rng.choice([
        (f"How many units of product '{p}' were sold at a price above {t}?", sum(r["units"] for r in rows if r["product"] == p and r["price"] > t)),
        (f"How many units of product '{p}' were sold with price at least {t}?", sum(r["units"] for r in rows if r["product"] == p and r["price"] >= t)),
        (f"How many units were sold in orders with price below {t}?", sum(r["units"] for r in rows if r["price"] < t)),
    ])
    return DQ(q, e, rows)


@v("data_analysis", "hard")
def d_difference(rng):
    rows = sales(rng)
    a, b = rng.sample(REGIONS, 2)
    ta, tb = (sum(r["units"] for r in rows if r["region"] == g) for g in (a, b))
    if ta < tb:
        a, b, ta, tb = b, a, tb, ta
    return DQ(f"How many more units were sold in region '{a}' than in region '{b}'?", ta - tb, rows)


@v("data_analysis", "extreme")
def d_x_percent_share(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    tot = sum(r["units"] for r in rows)
    part = sum(r["units"] for r in rows if r["product"] == p)
    return DQ(f"What percentage of all units sold were of product '{p}'? Round down to an integer.", part * 100 // tot, rows)


@v("data_analysis", "extreme")
def d_x_median(rng):
    rows = sales(rng, rng.choice([21, 51, 101]))
    xs = sorted(r["units"] for r in rows)
    return DQ("What is the median of the units column?", xs[len(xs) // 2], rows)


@v("data_analysis", "extreme")
def d_x_top_n_sum(rng):
    rows = sales(rng)
    xs = sorted((r["units"] for r in rows), reverse=True)
    return DQ("What is the sum of units of the 3 orders with the most units?", sum(xs[:3]), rows)


@v("data_analysis", "extreme")
def d_x_group_count(rng):
    rows = sales(rng, 300)
    t = rng.randint(300, 600)
    tot = {(g, p): sum(r["units"] for r in rows if r["region"] == g and r["product"] == p) for g in REGIONS for p in PRODUCTS}
    return DQ(f"How many (region, product) pairs have total units above {t}?", sum(1 for x in tot.values() if x > t), rows)


@v("data_analysis", "easy")
def d_paraphrase(rng):
    rows = sales(rng)
    p, lo, hi = _pm(rng)
    e = sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi)
    q = rng.choice([
        f"What is the total number of units of product '{p}' sold in months {lo} through {hi} inclusive?",
        f"How many units of product '{p}' were sold between months {lo} and {hi} inclusive?",
        f"How many units of product '{p}' were sold from month {lo} to month {hi} inclusive?",
        f"How many units of the product '{p}' were sold in months {lo} to {hi} (inclusive)?",
        f"How many units of product {p} were sold in months {lo} through {hi} inclusive?",
        f"How many units of product \"{p}\" were sold in months {lo} through {hi} inclusive?",
        f"What is the sum of units for product '{p}' in months {lo}-{hi}?",
        f"Compute the total units sold of product '{p}' during months {lo} through {hi} (inclusive).",
        f"For product '{p}', how many units were sold in months {lo} through {hi} inclusive?",
        f"In months {lo} through {hi} inclusive, how many units of product '{p}' were sold?",
        f"How many '{p}' units were sold in months {lo} through {hi} inclusive?",
        f"How many units of product '{p}' were sold in months {lo} through {hi}?",
        f"How many units of product '{p.capitalize()}' were sold in months {lo} through {hi} inclusive?",
    ])
    return DQ(q, e, rows)


def _file_variant(name, level, make):
    def fn(rng):
        rows = sales(rng)
        q, e = _base_q(rng, rows)
        return make(rng, rows, q, e)
    fn.__name__ = name
    v("data_analysis", level)(fn)

_file_variant("d_f_semicolon", "mid", lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows, delim=";")))
_file_variant("d_f_tab", "mid", lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows, delim="\t")))
_file_variant("d_f_quoted_all", "mid", lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows, quote=csv.QUOTE_ALL)))
_file_variant("d_f_crlf", "easy", lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows, eol="\r\n")))
_file_variant("d_f_bom", "mid", lambda rng, rows, q, e: DQ(q, e, rows, text="\ufeff" + csv_text(rows)))
_file_variant("d_f_no_final_newline", "easy", lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows).rstrip("\n")))
_file_variant("d_f_blank_lines", "mid", lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows).replace("\n", "\n\n", 3) + "\n\n"))
_file_variant("d_f_spaces_after_comma", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows).replace(",", ", ")))
_file_variant("d_f_upper_header", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows).replace(",".join(SALES_COLUMNS), ",".join(c.upper() for c in SALES_COLUMNS), 1)))
_file_variant("d_f_title_header", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text(rows).replace(",".join(SALES_COLUMNS), ",".join(c.title() for c in SALES_COLUMNS), 1)))
_file_variant("d_f_capitalized_values", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text([dict(r, product=r["product"].capitalize(), region=r["region"].upper()) for r in rows])))
_file_variant("d_f_float_units", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text([dict(r, units=f"{r['units']}.0") for r in rows])))
_file_variant("d_f_padded_values", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, text=csv_text([dict(r, product=f" {r['product']} ") for r in rows])))
_file_variant("d_f_shuffled_columns", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, columns=["price", "units", "month", "product", "region", "order_id"]))
_file_variant("d_f_extra_columns", "mid",
              lambda rng, rows, q, e: DQ(q, e, [dict(r, customer=f"c{r['order_id'] % 7}", discount=r["price"] % 5) for r in rows],
                                         columns=SALES_COLUMNS + ["customer", "discount"]))
_file_variant("d_f_rowcount_mismatch", "mid", lambda rng, rows, q, e: DQ(q, e, rows, nrows=len(rows) + 5))
_file_variant("d_f_no_intro", "mid", lambda rng, rows, q, e: DQ("In the file sales.csv: " + q[0].lower() + q[1:], e, rows, intro=False))
_file_variant("d_f_two_files", "hard",
              lambda rng, rows, q, e: DQ(q, e, rows, files={"sales.csv": csv_text(rows), "returns.csv": csv_text(sales(rng, 15)),
                                                           "README.txt": "sales.csv holds orders; returns.csv holds returns.\n"}))
_file_variant("d_f_subdir", "hard",
              lambda rng, rows, q, e: dict(question=f"The file data/sales.csv has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows). " + q,
                                           expected=str(e), files={"data/sales.csv": csv_text(rows)}, check="exact_int"))
_file_variant("d_f_other_name", "mid",
              lambda rng, rows, q, e: DQ(q, e, rows, name=rng.choice(["orders.csv", "data.csv", "sales_2024.csv", "Sales.csv"])))
_file_variant("d_f_tsv_ext", "hard", lambda rng, rows, q, e: DQ(q, e, rows, name="sales.tsv", text=csv_text(rows, delim="\t")))
_file_variant("d_f_duplicate_rows", "mid", lambda rng, rows, q, e: DQ(q, e, rows))


@v("data_analysis", "hard")
def d_f_10k_rows(rng):
    rows = sales(rng, 10000)
    return DQ(*_base_q(rng, rows), rows)


@v("data_analysis", "hard")
def d_f_100k_rows(rng):
    rows = sales(rng, 100000)
    return DQ(*_base_q(rng, rows), rows)


@v("data_analysis", "hard")
def d_f_blank_units_cells(rng):
    rows = sales(rng, 100)
    p, lo, hi = _pm(rng)
    out = [dict(r) for r in rows]
    for r in rng.sample(out, 5):
        r["units"] = ""
    e = sum(r["units"] for r in out if r["units"] != "" and r["product"] == p and lo <= r["month"] <= hi)
    return DQ(f"How many units of product '{p}' were sold in months {lo} through {hi} inclusive?", e, rows,
              text=csv_text(out), kind="amb")


@v("data_analysis", "hard")
def d_f_negative_units(rng):
    rows = sales(rng, 100)
    for r in rng.sample(rows, 8):
        r["units"] = -r["units"]
    return DQ(*_base_q(rng, rows), rows)


@v("data_analysis", "extreme")
def d_x_other_schema(rng):
    depts = ["sales", "engineering", "support", "finance"]
    rows = [{"employee_id": i + 1, "department": rng.choice(depts), "age": rng.randint(22, 60),
             "salary": rng.randint(30, 150) * 1000} for i in range(rng.choice([20, 100]))]
    cols = ["employee_id", "department", "age", "salary"]
    d = rng.choice(depts)
    q, e = rng.choice([
        (f"What is the total salary of department '{d}'?", sum(r["salary"] for r in rows if r["department"] == d)),
        (f"How many employees are in department '{d}'?", sum(1 for r in rows if r["department"] == d)),
        (f"What is the maximum age in department '{d}'?", max([r["age"] for r in rows if r["department"] == d] or [0])),
    ])
    return DQ(q, e, rows, columns=cols, name="employees.csv")


@v("data_analysis", "extreme")
def d_x_date_column(rng):
    rows = sales(rng, 100)
    out = [dict(r, date=f"2024-{r['month']:02d}-{rng.randint(1, 28):02d}") for r in rows]
    cols = ["order_id", "region", "product", "date", "units", "price"]
    p, lo, hi = _pm(rng)
    e = sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi)
    return DQ(f"How many units of product '{p}' were sold in months {lo} through {hi} inclusive?", e, out, columns=cols)


@v("data_analysis", "extreme")
def d_x_join(rng):
    rows = sales(rng, 100)
    cat = {p: rng.choice(["tools", "toys"]) for p in PRODUCTS}
    c = rng.choice(["tools", "toys"])
    files = {"sales.csv": csv_text(rows), "products.csv": "product,category\n" + "".join(f"{p},{k}\n" for p, k in cat.items())}
    return dict(question=f"The file sales.csv has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows) and the file "
                         f"products.csv has columns product,category. How many units were sold of products in category '{c}'?",
                expected=str(sum(r["units"] for r in rows if cat[r["product"]] == c)), files=files, check="exact_int")


@v("data_analysis", "extreme")
def d_x_json_file(rng):
    rows = sales(rng, 50)
    p, lo, hi = _pm(rng)
    e = sum(r["units"] for r in rows if r["product"] == p and lo <= r["month"] <= hi)
    return dict(question=f"The file sales.json contains a list of orders with fields {','.join(SALES_COLUMNS)} ({len(rows)} records). "
                         f"How many units of product '{p}' were sold in months {lo} through {hi} inclusive?",
                expected=str(e), files={"sales.json": json.dumps(rows, indent=1)}, check="exact_int")


# --- data: невыполнимые и спорные


@v("data_analysis", "base", "inf")
def d_i_missing_column(rng):
    rows = sales(rng)
    p, lo, hi = _pm(rng)
    q = rng.choice([f"What is the total discount given on product '{p}' in months {lo} through {hi} inclusive?",
                    f"What is the total profit of product '{p}'?", f"How many returns of product '{p}' were recorded?",
                    f"Which customer bought the most units of product '{p}'?",
                    f"How many units of product '{p}' were sold in year 2023?",
                    f"What is the total weight of product '{p}' sold in months {lo} through {hi} inclusive?",
                    f"How many units were sold by salesperson 'Olga'?", f"How many units were sold in country 'France'?"])
    return DQ(q, None, rows)


@v("data_analysis", "base", "inf")
def d_i_missing_file(rng):
    rows = sales(rng)
    spec = DQ(*_base_q(rng, rows), rows)
    spec["expected"] = None
    spec["files"] = rng.choice([{}, {"notes.txt": "The export job has not produced sales.csv yet.\n"},
                                {"sales.txt": "see sales.csv\n"}])
    return spec


@v("data_analysis", "base", "inf")
def d_i_column_removed(rng):
    rows = sales(rng)
    q, _ = _base_q(rng, rows)
    drop = rng.choice(["units", "month", "product"])
    cols = [c for c in SALES_COLUMNS if c != drop]
    spec = DQ(q, None, rows, columns=cols)
    if rng.random() < 0.5:   # условие по-прежнему перечисляет все столбцы
        spec["question"] = f"The file sales.csv has columns {','.join(SALES_COLUMNS)} ({len(rows)} rows). " + q
    return spec


@v("data_analysis", "mid", "inf")
def d_i_empty_file(rng):
    rows = sales(rng)
    q, _ = _base_q(rng, rows)
    return DQ(q, None, rows, text="")


@v("data_analysis", "mid", "inf")
def d_i_non_numeric_units(rng):
    rows = sales(rng, 20)
    q, _ = _base_q(rng, rows)
    return DQ(q, None, rows, text=csv_text([dict(r, units=rng.choice(["n/a", "many", "-", "?"])) for r in rows]))


@v("data_analysis", "mid", "inf")
def d_i_binary_garbage(rng):
    rows = sales(rng)
    q, _ = _base_q(rng, rows)
    return DQ(q, None, rows, text="\x00\x01\x02PK\x03\x04 corrupted \x7f\x00\x00" * 20)


@v("data_analysis", "mid", "inf")
def d_i_other_file_asked(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    return dict(question=f"The file customers.csv has columns customer_id,name,city ({len(rows)} rows). "
                         f"How many customers live in city 'Osaka'?", expected=None,
                files={"sales.csv": csv_text(rows)}, check="exact_int")


@v("data_analysis", "mid", "inf")
def d_i_avg_of_nothing(rng):
    rows = [r for r in sales(rng, 40) if r["product"] != "gizmo"]
    return DQ("What is the average price of product 'gizmo', rounded down to an integer?", None, rows)


@v("data_analysis", "mid", "inf")
def d_i_vague_question(rng):
    rows = sales(rng)
    return DQ(rng.choice(["Which product is the best?", "Was it a good year?", "Which region should we invest in?",
                          "How many units will be sold next year?"]), None, rows)


@v("data_analysis", "mid", "amb")
def d_a_unknown_product(rng):
    rows = sales(rng)
    _, lo, hi = _pm(rng)
    return DQ(f"How many units of product '{rng.choice(['doohickey', 'gear', 'bolt'])}' were sold in months {lo} through {hi} inclusive?",
              0, rows)


@v("data_analysis", "mid", "amb")
def d_a_month_out_of_range(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    return DQ(f"How many units of product '{p}' were sold in months 13 through 14 inclusive?", 0, rows)


@v("data_analysis", "mid", "amb")
def d_a_header_only(rng):
    rows = sales(rng)
    q, _ = _base_q(rng, rows)
    return DQ(q, 0, rows, text=",".join(SALES_COLUMNS) + "\n")


@v("data_analysis", "mid", "amb")
def d_a_reversed_range(rng):
    rows = sales(rng)
    p = rng.choice(PRODUCTS)
    return DQ(f"How many units of product '{p}' were sold in months 9 through 4 inclusive?", 0, rows)


@v("data_analysis", "mid", "amb")
def d_a_some_bad_cells(rng):
    rows = sales(rng, 60)
    out = [dict(r) for r in rows]
    for r in rng.sample(out, 3):
        r["units"] = "n/a"
    p, lo, hi = _pm(rng)
    e = sum(r["units"] for r in out if r["units"] != "n/a" and r["product"] == p and lo <= r["month"] <= hi)
    return DQ(f"How many units of product '{p}' were sold in months {lo} through {hi} inclusive?", e, rows, text=csv_text(out))


# =========================================================================== CODING
# имя -> (исправный код, видимые проверки, скрытые дополнительные проверки, баги {вид: (что, на что)})
F = {}


def fn(name, src, vis, hid, bugs):
    F[name] = dict(src=src, vis=vis if isinstance(vis, list) else [vis], hid=hid, bugs=bugs)

fn("total", "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
   "self.assertEqual(m.total([1, 2, 3]), 6)", ["self.assertEqual(m.total([]), 0)", "self.assertEqual(m.total([10, -4]), 6)"],
   {"op": ("s += x", "s -= x"), "init": ("s = 0", "s = 1"), "assign": ("s += x", "s = x"), "ret": ("return s", "return x"),
    "indent": ("    return s", "        return s"), "noreturn": ("    return s\n", ""), "name": ("return s", "return sm"),
    "syntax": ("def total(xs):", "def total(xs)"), "stub": ("    s = 0\n    for x in xs:\n        s += x\n    return s\n", "    pass\n")})
fn("maximum", "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
   "self.assertEqual(m.maximum([3, 9, 2]), 9)", ["self.assertEqual(m.maximum([-5, -2, -9]), -2)", "self.assertEqual(m.maximum([7]), 7)"],
   {"cmp": ("x > best", "x < best"), "ret": ("return best", "return x"), "assign": ("best = x\n", "best = best\n"),
    "idx": ("best = xs[0]", "best = xs[1]"), "name": ("if x > best", "if x > bst")})
fn("minimum", "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
   "self.assertEqual(m.minimum([3, 9, 2]), 2)", ["self.assertEqual(m.minimum([4, 4, 7]), 4)", "self.assertEqual(m.minimum([-1, -8]), -8)"],
   {"cmp": ("x < best", "x > best")})
fn("average", "def average(xs):\n    return sum(xs) / len(xs)\n",
   "self.assertEqual(m.average([2, 4, 6]), 4)", ["self.assertEqual(m.average([1, 2]), 1.5)", "self.assertEqual(m.average([5]), 5)"],
   {"op": ("sum(xs) / len(xs)", "sum(xs) * len(xs)"), "off": ("len(xs)", "(len(xs) + 1)"), "func": ("sum(xs) /", "max(xs) /"),
    "op2": ("sum(xs) / len(xs)", "sum(xs) - len(xs)")})
fn("first_n", "def first_n(xs, n):\n    return xs[:n]\n",
   "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])", ["self.assertEqual(m.first_n([1, 2], 1), [1])", "self.assertEqual(m.first_n([1, 2, 3], 0), [])"],
   {"slice": ("xs[:n]", "xs[n:]"), "off": ("xs[:n]", "xs[:n + 1]"), "off2": ("xs[:n]", "xs[:n - 1]"), "neg": ("xs[:n]", "xs[-n:]")})
fn("last_n", "def last_n(xs, n):\n    return xs[len(xs) - n:]\n",
   "self.assertEqual(m.last_n([5, 6, 7, 8], 2), [7, 8])", ["self.assertEqual(m.last_n([1, 2, 3], 1), [3])", "self.assertEqual(m.last_n([1, 2, 3], 0), [])"],
   {"slice": ("xs[len(xs) - n:]", "xs[:len(xs) - n]"), "op": ("len(xs) - n", "len(xs) + n")})
fn("count_even", "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
   "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)", ["self.assertEqual(m.count_even([2, 2, 3]), 2)", "self.assertEqual(m.count_even([]), 0)"],
   {"const": ("x % 2 == 0", "x % 2 == 1"), "cmp": ("x % 2 == 0", "x % 2 != 0"), "mod": ("x % 2 == 0", "x % 3 == 0")})
fn("count_odd", "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
   "self.assertEqual(m.count_odd([1, 2, 4, 6, 7]), 2)", ["self.assertEqual(m.count_odd([3, 5, 8]), 2)"],
   {"const": ("x % 2 == 1", "x % 2 == 0")})
fn("reverse_words", "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
   "self.assertEqual(m.reverse_words('c a b'), 'b a c')", ["self.assertEqual(m.reverse_words('one two'), 'two one')", "self.assertEqual(m.reverse_words('x'), 'x')"],
   {"func": ("reversed(s.split())", "sorted(s.split())"), "lit": ("' '.join", "''.join"), "noop": ("reversed(s.split())", "s.split()")})
fn("is_palindrome", "def is_palindrome(s):\n    return s == s[::-1]\n",
   ["self.assertTrue(m.is_palindrome('level'))", "self.assertFalse(m.is_palindrome('hello'))"], ["self.assertTrue(m.is_palindrome(''))", "self.assertFalse(m.is_palindrome('ab'))"],
   {"cmp": ("s == s[::-1]", "s != s[::-1]"), "step": ("s[::-1]", "s[::1]")})
fn("factorial", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
   "self.assertEqual(m.factorial(5), 120)", ["self.assertEqual(m.factorial(3), 6)", "self.assertEqual(m.factorial(0), 1)", "self.assertEqual(m.factorial(1), 1)"],
   {"op": ("result *= i", "result += i"), "off": ("range(2, n + 1)", "range(2, n)"), "init": ("result = 1", "result = 0"),
    "double": ("result *= i", "result += i|range(2, n + 1)|range(2, n)")})
fn("clamp", "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
   ["self.assertEqual(m.clamp(15, 0, 10), 10)", "self.assertEqual(m.clamp(-3, 0, 10), 0)"], ["self.assertEqual(m.clamp(5, 0, 10), 5)"],
   {"func": ("max(lo, min(x, hi))", "min(lo, min(x, hi))"), "swap": ("max(lo, min(x, hi))", "max(hi, min(x, lo))")})
fn("unique", "def unique(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n",
   "self.assertEqual(m.unique([1, 2, 1, 3, 2]), [1, 2, 3])", ["self.assertEqual(m.unique([]), [])", "self.assertEqual(m.unique([3, 3, 1]), [3, 1])"],
   {"neg": ("if x not in out", "if x in out"), "delline": ("            out.append(x)\n", "            pass\n")})
fn("count_vowels", "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
   "self.assertEqual(m.count_vowels('Banana'), 3)", ["self.assertEqual(m.count_vowels('UFO'), 2)", "self.assertEqual(m.count_vowels('xyz'), 0)"],
   {"neg": ("if ch in 'aeiou'", "if ch not in 'aeiou'"), "lit": ("'aeiou'", "'eiou'")})
fn("range_sum", "def range_sum(a, b):\n    return sum(range(a, b + 1))\n",
   "self.assertEqual(m.range_sum(1, 4), 10)", ["self.assertEqual(m.range_sum(2, 3), 5)", "self.assertEqual(m.range_sum(5, 5), 5)"],
   {"off": ("range(a, b + 1)", "range(a, b)"), "off2": ("range(a, b + 1)", "range(a + 1, b + 1)")})
fn("capitalize_words", "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
   "self.assertEqual(m.capitalize_words('hello big world'), 'Hello Big World')", ["self.assertEqual(m.capitalize_words('a'), 'A')"],
   {"method": ("w.capitalize()", "w.upper()"), "method2": ("w.capitalize()", "w.lower()")})
fn("dot", "def dot(xs, ys):\n    return sum(x * y for x, y in zip(xs, ys))\n",
   "self.assertEqual(m.dot([1, 2, 3], [4, 5, 6]), 32)", ["self.assertEqual(m.dot([2], [5]), 10)", "self.assertEqual(m.dot([], []), 0)"],
   {"op": ("x * y", "x + y")})
fn("running_total", "def running_total(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n        out.append(s)\n    return out\n",
   "self.assertEqual(m.running_total([1, 2, 3]), [1, 3, 6])", ["self.assertEqual(m.running_total([]), [])", "self.assertEqual(m.running_total([5, -5]), [5, 0])"],
   {"assign": ("s += x", "s = x"), "arg": ("out.append(s)", "out.append(x)"), "delline": ("        s += x\n", "")})
fn("square_all", "def square_all(xs):\n    return [x * x for x in xs]\n",
   "self.assertEqual(m.square_all([1, 2, 3]), [1, 4, 9])", ["self.assertEqual(m.square_all([-2]), [4])", "self.assertEqual(m.square_all([0, 5]), [0, 25])"],
   {"op": ("x * x", "x + x"), "const": ("x * x", "x * 2")})
fn("count_positive", "def count_positive(xs):\n    return sum(1 for x in xs if x > 0)\n",
   "self.assertEqual(m.count_positive([-1, 0, 2, 5]), 2)", ["self.assertEqual(m.count_positive([0, 0]), 0)", "self.assertEqual(m.count_positive([-3, -1]), 0)"],
   {"cmp": ("x > 0", "x >= 0"), "cmp2": ("x > 0", "x < 0")})
fn("is_sorted", "def is_sorted(xs):\n    return all(a <= b for a, b in zip(xs, xs[1:]))\n",
   ["self.assertTrue(m.is_sorted([1, 2, 2, 5]))", "self.assertFalse(m.is_sorted([3, 1, 2]))"], ["self.assertTrue(m.is_sorted([]))", "self.assertFalse(m.is_sorted([1, 3, 2, 4]))"],
   {"cmp": ("a <= b", "a < b"), "cmp2": ("a <= b", "a >= b")})
fn("longest_word", "def longest_word(s):\n    return max(s.split(), key=len)\n",
   "self.assertEqual(m.longest_word('a quick brown fox'), 'quick')", ["self.assertEqual(m.longest_word('hi there'), 'there')"],
   {"func": ("max(s.split(), key=len)", "min(s.split(), key=len)"), "nokey": ("max(s.split(), key=len)", "max(s.split())")})
fn("abs_diff", "def abs_diff(a, b):\n    return abs(a - b)\n",
   "self.assertEqual(m.abs_diff(3, 10), 7)", ["self.assertEqual(m.abs_diff(10, 3), 7)", "self.assertEqual(m.abs_diff(-2, 2), 4)"],
   {"op": ("abs(a - b)", "abs(a + b)"), "noabs": ("abs(a - b)", "a - b")})
fn("countdown", "def countdown(n):\n    out = []\n    while n > 0:\n        out.append(n)\n        n -= 1\n    return out\n",
   "self.assertEqual(m.countdown(3), [3, 2, 1])", ["self.assertEqual(m.countdown(0), [])", "self.assertEqual(m.countdown(1), [1])"],
   {"cmp": ("while n > 0", "while n > 1"), "arg": ("out.append(n)", "out.append(n - 1)"), "loop": ("n -= 1", "n += 1")})
fn("fib", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
   "self.assertEqual(m.fib(10), 55)", ["self.assertEqual(m.fib(0), 0)", "self.assertEqual(m.fib(1), 1)", "self.assertEqual(m.fib(7), 13)"],
   {"op": ("a, b = b, a + b", "a, b = b, a - b"), "off": ("range(n)", "range(n + 1)"), "ret": ("return a", "return b")})
fn("is_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
   ["self.assertTrue(m.is_prime(13))", "self.assertFalse(m.is_prime(15))"], ["self.assertFalse(m.is_prime(1))", "self.assertTrue(m.is_prime(2))", "self.assertFalse(m.is_prime(49))"],
   {"const": ("if n % i == 0", "if n % i == 1"), "bool": ("            return False\n", "            return True\n"), "off": ("int(n ** 0.5) + 1", "int(n ** 0.5)")})
fn("gcd", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
   "self.assertEqual(m.gcd(12, 18), 6)", ["self.assertEqual(m.gcd(7, 13), 1)", "self.assertEqual(m.gcd(10, 5), 5)"],
   {"op": ("a % b", "a // b"), "ret": ("return a", "return b")})
fn("flatten", "def flatten(xss):\n    out = []\n    for xs in xss:\n        out.extend(xs)\n    return out\n",
   "self.assertEqual(m.flatten([[1, 2], [3]]), [1, 2, 3])", ["self.assertEqual(m.flatten([]), [])", "self.assertEqual(m.flatten([[], [4]]), [4])"],
   {"method": ("out.extend(xs)", "out.append(xs)")})
fn("median", "def median(xs):\n    ys = sorted(xs)\n    return ys[len(ys) // 2]\n",
   "self.assertEqual(m.median([5, 1, 3]), 3)", ["self.assertEqual(m.median([9, 2, 7, 4, 1]), 4)", "self.assertEqual(m.median([8]), 8)"],
   {"nosort": ("ys = sorted(xs)", "ys = list(xs)"), "off": ("len(ys) // 2", "len(ys) // 2 + 1")})
fn("word_count", "def word_count(s):\n    return len(s.split())\n",
   "self.assertEqual(m.word_count('a quick brown fox'), 4)", ["self.assertEqual(m.word_count(''), 0)", "self.assertEqual(m.word_count('one'), 1)"],
   {"func": ("len(s.split())", "len(s)"), "off": ("len(s.split())", "len(s.split()) + 1")})
fn("power", "def power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result\n",
   "self.assertEqual(m.power(2, 5), 32)", ["self.assertEqual(m.power(3, 0), 1)", "self.assertEqual(m.power(5, 2), 25)"],
   {"op": ("result *= base", "result += base"), "arg": ("range(exp)", "range(base)")})
fn("safe_div", "def safe_div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n",
   ["self.assertEqual(m.safe_div(6, 3), 2)", "self.assertRaises(ValueError, m.safe_div, 1, 0)"], ["self.assertEqual(m.safe_div(1, 4), 0.25)"],
   {"cmp": ("if b == 0", "if b != 0"), "exc": ("raise ValueError(", "raise TypeError("), "op": ("return a / b", "return a * b")})
fn("is_even", "def is_even(n):\n    return n % 2 == 0\n",
   ["self.assertTrue(m.is_even(4))", "self.assertFalse(m.is_even(7))"], ["self.assertTrue(m.is_even(0))"],
   {"const": ("n % 2 == 0", "n % 2 == 1")})
fn("sign", "def sign(x):\n    if x > 0:\n        return 1\n    if x < 0:\n        return -1\n    return 0\n",
   ["self.assertEqual(m.sign(5), 1)", "self.assertEqual(m.sign(-3), -1)"], ["self.assertEqual(m.sign(0), 0)"],
   {"cmp": ("if x > 0", "if x < 0|if x < 0:\n        return -1|if x > 0:\n        return -1"), "const": ("return -1", "return 1")})
fn("celsius_to_f", "def celsius_to_f(c):\n    return c * 9 / 5 + 32\n",
   "self.assertEqual(m.celsius_to_f(100), 212)", ["self.assertEqual(m.celsius_to_f(0), 32)", "self.assertEqual(m.celsius_to_f(-40), -40)"],
   {"swap2": ("c * 9 / 5", "c * 5 / 9"), "const": ("+ 32", "+ 23"), "op": ("+ 32", "- 32")})
fn("second_largest", "def second_largest(xs):\n    return sorted(set(xs))[-2]\n",
   "self.assertEqual(m.second_largest([4, 9, 7, 9]), 7)", ["self.assertEqual(m.second_largest([1, 2]), 1)"],
   {"idx": ("[-2]", "[-1]"), "idx2": ("[-2]", "[1]")})
fn("binary_search", "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
   "        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
   ["self.assertEqual(m.binary_search([1, 3, 5, 7, 9], 7), 3)", "self.assertEqual(m.binary_search([1, 3, 5], 4), -1)"],
   ["self.assertEqual(m.binary_search([], 1), -1)", "self.assertEqual(m.binary_search([2, 4, 6, 8], 2), 0)"],
   {"hang": ("lo = mid + 1", "lo = mid"), "cmp": ("if xs[mid] < target", "if xs[mid] > target"), "ret": ("            return mid\n", "            return lo + 1\n")})
fn("merge_sorted", "def merge_sorted(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n"
   "        else:\n            out.append(b[j])\n            j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n",
   "self.assertEqual(m.merge_sorted([1, 4, 6], [2, 3, 7]), [1, 2, 3, 4, 6, 7])", ["self.assertEqual(m.merge_sorted([], [1]), [1])", "self.assertEqual(m.merge_sorted([5], [1, 2]), [1, 2, 5])"],
   {"cmp": ("if a[i] <= b[j]", "if a[i] >= b[j]"), "bool": ("i < len(a) and j < len(b)", "i < len(a) or j < len(b)"), "delline": ("    out.extend(b[j:])\n", "")})
STACK_SRC = ("class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, x):\n        self.items.append(x)\n\n"
             "    def pop(self):\n        return self.items.pop()\n\n    def peek(self):\n        return self.items[-1]\n\n"
             "    def size(self):\n        return len(self.items)\n")
STACK_TEST = ("        s = m.Stack()\n        s.push(1)\n        s.push(2)\n        self.assertEqual(s.peek(), 2)\n"
              "        self.assertEqual(s.pop(), 2)\n        self.assertEqual(s.size(), 1)")


def _apply_bug(src, bug):
    """bug = (old, new) или цепочка «old|new|old2|new2» в new для двух правок сразу."""
    old, new = bug
    parts = new.split("|")
    out = src.replace(old, parts[0], 1)
    assert out != src, (old, src)
    for i in range(1, len(parts) - 1, 2):
        nxt = out.replace(parts[i], parts[i + 1], 1)
        assert nxt != out, (parts[i], out)
        out = nxt
    return out


def test_file(tests, module="mylib", extra_imports=""):
    lines = ["import unittest", "", f"import {module} as m", extra_imports, "", "class T(unittest.TestCase):"] if extra_imports else \
            ["import unittest", "", f"import {module} as m", "", "", "class T(unittest.TestCase):"]
    for i, (name, asserts) in enumerate(tests):
        if i:
            lines.append("")
        lines.append(f"    def test_{name}(self):")
        lines.extend(a if a.startswith("        ") else f"        {a}" for a in asserts)
    lines += ["", "", "if __name__ == '__main__':", "    unittest.main()", ""]
    return "\n".join(lines)


def run_tests(files, cmd=("test_mylib.py",), timeout=20):
    with tempfile.TemporaryDirectory() as tmp:
        for rel, text in files.items():
            p = Path(tmp, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-B", *cmd], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0


def CQ(rng, names, bugs, module="mylib", instruction=None, kinds=None, hidden=True, extra_tests=None,
       expect_fixable=True, mutate_lib=None, mutate_test=None, files_extra=None):
    """names — функции в файле; bugs — {имя: вид бага}. Возвращает spec или None, если задача не получилась."""
    good_parts, bad_parts, vis_tests, hid_tests = [], [], [], []
    for name in names:
        f = F[name]
        good_parts.append(f["src"])
        bad_parts.append(_apply_bug(f["src"], f["bugs"][bugs[name]]) if name in bugs else f["src"])
        vis_tests.append((name, list(f["vis"])))
        hid_tests.append((name, list(f["vis"]) + list(f["hid"])))
    good, bad = "\n\n".join(good_parts), "\n\n".join(bad_parts)
    if mutate_lib:
        bad = mutate_lib(bad)
    vis_tests += extra_tests or []
    hid_tests += extra_tests or []
    vis, hid = test_file(vis_tests, module), test_file(hid_tests if hidden else vis_tests, module)
    if mutate_test:
        vis = hid = mutate_test(vis)
    tname = f"test_{module}.py"
    files = {f"{module}.py": bad, tname: vis, **(files_extra or {})}
    if expect_fixable:
        if not run_tests({f"{module}.py": good, tname: hid, **(files_extra or {})}, (tname,)):
            return None
        if bugs and run_tests(files, (tname,)):
            return None      # баг не ловится видимыми тестами
        for name in bugs:    # каждый баг должен ронять видимый тест своей функции (как у организаторов)
            alone = _apply_bug(F[name]["src"], F[name]["bugs"][bugs[name]])
            if run_tests({"mylib.py": alone, "test_mylib.py": test_file([(name, list(F[name]["vis"]))])}, timeout=5):
                return None
    suffix = os.environ.get("STRESS_RENAME", "")
    if suffix:
        # Имена функций, которых нет в таблице эталонов навыка: проверяется путь «перебор + сверка с моделью».
        pattern = re.compile(r"\b(" + "|".join(sorted(F, key=len, reverse=True)) + r")\b(?=\()|(?<=m\.)(" + "|".join(sorted(F, key=len, reverse=True)) + r")\b")
        ren = lambda text: pattern.sub(lambda mm: (mm.group(1) or mm.group(2)) + suffix, text)  # noqa: E731
        good, hid = ren(good), ren(hid)
        files = {k: ren(val) if k.endswith(".py") else val for k, val in files.items()}
    instr = instruction or CODING_INSTRUCTION
    if module != "mylib":
        instr = instr.replace("test_mylib.py", tname).replace("mylib.py", f"{module}.py")
    return dict(instruction=instr, expected="tests_pass" if expect_fixable else None, files=files, check="unittest",
                tests={tname: hid}, fixed={f"{module}.py": good})


def _coding_variant(name, level, make, kind="ok"):
    def gen(rng):
        for _ in range(60):
            spec = make(rng)
            if spec is not None:
                return spec
        raise RuntimeError(f"cannot generate {name}")
    gen.__name__ = name
    v("coding", level, kind)(gen)


def _pick_funcs(rng, n, must=()):
    pool = [k for k in F if k not in must and k != "binary_search"]
    return list(must) + rng.sample(pool, n - len(must))


def _with_kind(kinds):
    return [(n, k) for n, f in F.items() for k in f["bugs"] if k in kinds]

SIMPLE = ("op", "cmp", "const", "func", "off", "slice", "neg", "method", "init", "assign", "ret", "arg", "idx", "lit",
          "op2", "cmp2", "off2", "method2", "idx2", "bool", "mod", "step", "exc", "noop", "nokey", "noabs", "nosort", "swap")


def _simple(n_funcs, n_bugs):
    def make(rng):
        names = _pick_funcs(rng, n_funcs)
        rng.shuffle(names)
        buggy = rng.sample(names, min(n_bugs, len(names)))
        bugs = {}
        for b in buggy:
            ks = [k for k in F[b]["bugs"] if k in SIMPLE]
            if not ks:
                return None
            bugs[b] = rng.choice(ks)
        return CQ(rng, names, bugs)
    return make

_coding_variant("c_base_2bugs", "base", _simple(5, 2))
_coding_variant("c_1bug", "easy", _simple(4, 1))
_coding_variant("c_3bugs", "mid", _simple(6, 3))
_coding_variant("c_5bugs_of_8", "hard", _simple(8, 5))
_coding_variant("c_all_buggy_6", "hard", _simple(6, 6))
_coding_variant("c_15_functions_4bugs", "hard", _simple(15, 4))
_coding_variant("c_single_function", "easy", _simple(1, 1))


def _kind_variant(kind_names, label, level):
    def make(rng):
        cands = _with_kind(kind_names)
        name, k = rng.choice(cands)
        names = _pick_funcs(rng, rng.randint(3, 5), must=(name,))
        rng.shuffle(names)
        return CQ(rng, names, {name: k})
    _coding_variant(f"c_bug_{label}", level, make)

for _lbl, _ks, _lv in (("operator", ("op", "op2"), "base"), ("comparison", ("cmp", "cmp2"), "base"),
                       ("constant", ("const", "mod", "step"), "base"), ("wrong_function", ("func", "noop", "nokey", "noabs", "nosort"), "base"),
                       ("off_by_one", ("off", "off2"), "mid"), ("slice", ("slice", "neg"), "mid"),
                       ("method", ("method", "method2"), "mid"), ("init_value", ("init",), "mid"),
                       ("assign_vs_accumulate", ("assign",), "mid"), ("wrong_return", ("ret",), "mid"),
                       ("wrong_argument", ("arg", "swap"), "mid"), ("wrong_index", ("idx", "idx2"), "mid"),
                       ("wrong_literal", ("lit",), "mid"), ("bool_op", ("bool",), "mid"), ("wrong_exception", ("exc",), "hard"),
                       ("indent_return_in_loop", ("indent",), "hard"), ("missing_return", ("noreturn",), "hard"),
                       ("name_typo", ("name",), "hard"), ("syntax_error", ("syntax",), "hard"),
                       ("stub_pass", ("stub",), "hard"), ("deleted_line", ("delline",), "hard"),
                       ("two_edits_one_function", ("double", "swap2"), "hard"), ("infinite_loop", ("hang", "loop"), "hard")):
    _kind_variant(_ks, _lbl, _lv)


def _c_missing_function(rng):
    names = _pick_funcs(rng, 4)
    victim = rng.choice(names)
    spec = CQ(rng, names, {})
    if spec is None:
        return None
    spec["files"]["mylib.py"] = "\n\n".join(F[n]["src"] for n in names if n != victim)
    return spec

_coding_variant("c_missing_function", "hard", _c_missing_function)


def _c_class(rng):
    names = _pick_funcs(rng, 2)
    good = "\n\n".join([F[n]["src"] for n in names] + [STACK_SRC])
    old, new = rng.choice([("return self.items.pop()", "return self.items.pop(0)"), ("return self.items[-1]", "return self.items[0]"),
                           ("return len(self.items)", "return len(self.items) - 1"), ("self.items.append(x)", "self.items.insert(0, x)")])
    bad = good.replace(old, new, 1)
    tests = [(n, list(F[n]["vis"])) for n in names] + [("stack", [STACK_TEST])]
    t = test_file(tests)
    if not run_tests({"mylib.py": good, "test_mylib.py": t}) or run_tests({"mylib.py": bad, "test_mylib.py": t}):
        return None
    return dict(instruction=CODING_INSTRUCTION, expected="tests_pass", files={"mylib.py": bad, "test_mylib.py": t},
                check="unittest", tests={"test_mylib.py": t}, fixed={"mylib.py": good})

_coding_variant("c_class_method_bug", "hard", _c_class)
_coding_variant("c_other_module_name", "mid", lambda rng: CQ(rng, *_nb(rng, 4, 2), module=rng.choice(["utils", "helpers", "calc", "strlib"])))


def _nb(rng, n, k):
    names = _pick_funcs(rng, n)
    bugs = {}
    for b in rng.sample(names, k):
        ks = [x for x in F[b]["bugs"] if x in SIMPLE]
        if ks:
            bugs[b] = rng.choice(ks)
    return names, bugs


def _c_no_bug(rng):
    names = _pick_funcs(rng, 4)
    return CQ(rng, names, {})

_coding_variant("c_tests_already_pass", "mid", _c_no_bug, "amb")


def _c_fmt(mut):
    def make(rng):
        names, bugs = _nb(rng, 5, 2)
        return CQ(rng, names, bugs, mutate_lib=mut) if bugs else None
    return make

_coding_variant("c_fmt_crlf", "mid", _c_fmt(lambda s: s.replace("\n", "\r\n")))
_coding_variant("c_fmt_tabs", "mid", _c_fmt(lambda s: s.replace("    ", "\t")))
_coding_variant("c_fmt_docstrings_comments", "mid",
                _c_fmt(lambda s: '"""Small utility library."""\n\n# NOTE: keep functions pure\n\n' + s.replace("):\n", '):\n    """Auto-generated docstring."""\n', 1)))
_coding_variant("c_fmt_type_hints_header", "mid", _c_fmt(lambda s: "from typing import List\nimport math\n\n\n" + s))
_coding_variant("c_fmt_main_block", "mid", _c_fmt(lambda s: s + "\n\nif __name__ == '__main__':\n    print('mylib demo')\n"))
_coding_variant("c_fmt_no_blank_lines", "mid", _c_fmt(lambda s: s.replace("\n\n", "\n")))


def _c_multi_assert(rng):
    names, bugs = _nb(rng, 5, 2)
    if not bugs:
        return None
    for n in names:
        pass
    return CQ(rng, names, bugs, hidden=False, extra_tests=[("extra_" + n, list(F[n]["hid"])) for n in names])

_coding_variant("c_many_asserts_visible", "mid", _c_multi_assert)


def _c_helper_module(rng):
    names, bugs = _nb(rng, 4, 2)
    if len(bugs) < 2:
        return None
    spec = CQ(rng, names, bugs)
    if spec is None:
        return None
    spec["files"]["helpers.py"] = "def identity(x):\n    return x\n"
    spec["files"]["README.md"] = "mylib: small helpers. Run the tests with `python3 test_mylib.py`.\n"
    spec["files"]["mylib.py"] = "from helpers import identity\n\n\n" + spec["files"]["mylib.py"]
    spec["fixed"]["mylib.py"] = "from helpers import identity\n\n\n" + spec["fixed"]["mylib.py"]
    return spec

_coding_variant("c_extra_files_and_import", "hard", _c_helper_module)


def _c_bug_in_helper(rng):
    """mylib.py в порядке, баг в helpers.py, который он импортирует."""
    good_h = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
    bad_h = good_h.replace("a + b", "a - b") if rng.random() < 0.5 else good_h.replace("a * b", "a + b")
    lib = ("from helpers import add, mul\n\n\ndef total(xs):\n    s = 0\n    for x in xs:\n        s = add(s, x)\n    return s\n\n\n"
           "def square_all(xs):\n    return [mul(x, x) for x in xs]\n")
    t = test_file([("total", ["self.assertEqual(m.total([1, 2, 3]), 6)"]), ("square_all", ["self.assertEqual(m.square_all([1, 2, 3]), [1, 4, 9])"])])
    instr = CODING_INSTRUCTION.replace("Fix the bugs in mylib.py", "Fix the bugs in the library code")
    return dict(instruction=instr, expected="tests_pass", files={"mylib.py": lib, "helpers.py": bad_h, "test_mylib.py": t},
                check="unittest", tests={"test_mylib.py": t}, fixed={"mylib.py": lib, "helpers.py": good_h})

_coding_variant("c_bug_in_imported_helper", "extreme", _c_bug_in_helper)


def _c_pytest_style(rng):
    names, bugs = _nb(rng, 4, 2)
    if len(bugs) < 2:
        return None
    spec = CQ(rng, names, bugs)
    if spec is None:
        return None
    lines = ["import mylib as m", "", ""]
    for n in names:
        lines.append(f"def test_{n}():")
        for a in F[n]["vis"] + F[n]["hid"]:
            if "assertRaises" in a:
                continue
            a = a.replace("self.assertEqual(", "assert ", 1)
            if a.startswith("assert "):
                head, _, tail = a[:-1].rpartition(", ")
                lines.append(f"    {head} == {tail}")
            elif a.startswith("self.assertTrue("):
                lines.append("    assert " + a[len("self.assertTrue("):-1])
            else:
                lines.append("    assert not " + a[len("self.assertFalse("):-1])
        lines += ["", ""]
    lines += ["if __name__ == '__main__':", "    import sys", "    failed = 0",
              "    for name, f in sorted(globals().items()):",
              "        if name.startswith('test_') and callable(f):",
              "            try:", "                f()", "            except Exception as exc:",
              "                failed += 1", "                print('FAIL', name, repr(exc))",
              "    print('failed:', failed)", "    sys.exit(1 if failed else 0)", ""]
    t = "\n".join(lines)
    if not run_tests({"mylib.py": spec["fixed"]["mylib.py"], "test_mylib.py": t}) or run_tests({"mylib.py": spec["files"]["mylib.py"], "test_mylib.py": t}):
        return None
    spec["files"]["test_mylib.py"] = t
    spec["tests"] = {"test_mylib.py": t}
    return spec

_coding_variant("c_plain_assert_tests", "extreme", _c_pytest_style)


# --- coding: невыполнимые


def _c_inf(extra_fn):
    def make(rng):
        names, bugs = _nb(rng, 4, rng.choice([1, 2]))
        spec = CQ(rng, names, bugs, expect_fixable=False)
        return extra_fn(rng, spec, names)
    return make


def _contradict(rng, spec, names):
    wrong = {"total": "self.assertEqual(m.total([1, 2, 3]), 7)", "maximum": "self.assertEqual(m.maximum([3, 9, 2]), 3)",
             "count_even": "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 4)", "factorial": "self.assertEqual(m.factorial(5), 121)",
             "first_n": "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6])", "dot": "self.assertEqual(m.dot([1, 2, 3], [4, 5, 6]), 33)",
             "fib": "self.assertEqual(m.fib(10), 56)", "gcd": "self.assertEqual(m.gcd(12, 18), 5)", "power": "self.assertEqual(m.power(2, 5), 31)",
             "minimum": "self.assertEqual(m.minimum([3, 9, 2]), 9)", "abs_diff": "self.assertEqual(m.abs_diff(3, 10), 8)",
             "is_even": "self.assertFalse(m.is_even(4))", "word_count": "self.assertEqual(m.word_count('a quick brown fox'), 5)"}
    cands = [n for n in names if n in wrong]
    if not cands:
        return None
    t = rng.choice(cands)
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace(
        "\n\nif __name__", f"\n    def test_{t}_again(self):\n        {wrong[t]}\n\n\nif __name__", 1)
    return spec

_coding_variant("c_i_contradictory_tests", "base", _c_inf(_contradict), "inf")


def _contradict_same_method(rng, spec, names):
    spec = _contradict(rng, spec, names)
    if spec is None:
        return None
    # то же самое, но противоречие внутри одного теста
    import re
    spec["files"]["test_mylib.py"] = re.sub(r"\n    def test_\w+_again\(self\):\n", "\n", spec["files"]["test_mylib.py"])
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace("\n\n\n\nif __name__", "\n\n\nif __name__")
    return spec


def _const_false(rng, spec, names):
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace(
        "\n\nif __name__", "\n    def test_sanity(self):\n        " + rng.choice(["self.assertEqual(1, 2)", "self.assertTrue(False)", "self.fail('not implemented')", "self.assertEqual(2 + 2, 5)"]) + "\n\n\nif __name__", 1)
    return spec

_coding_variant("c_i_constant_false_assert", "mid", _c_inf(_const_false), "inf")


def _test_syntax(rng, spec, names):
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace("class T(unittest.TestCase):", "class T(unittest.TestCase)", 1)
    return spec

_coding_variant("c_i_test_file_syntax_error", "mid", _c_inf(_test_syntax), "inf")


def _test_missing_import(rng, spec, names):
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace("import mylib as m", "import mylib as m\nimport " + rng.choice(["fastmath_ext", "companylib", "secret_plugin"]), 1)
    return spec

_coding_variant("c_i_test_imports_missing_module", "mid", _c_inf(_test_missing_import), "inf")


def _no_tests(rng, spec, names):
    del spec["files"]["test_mylib.py"]
    return spec

_coding_variant("c_i_test_file_missing", "mid", _c_inf(_no_tests), "inf")


def _no_lib(rng, spec, names):
    del spec["files"]["mylib.py"]
    return spec

_coding_variant("c_i_mylib_missing", "mid", _c_inf(_no_lib), "inf")


def _empty_ws(rng, spec, names):
    spec["files"] = {}
    return spec

_coding_variant("c_i_empty_workspace", "mid", _c_inf(_empty_ws), "inf")


def _needs_file(rng, spec, names):
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace(
        "\n\nif __name__", "\n    def test_config(self):\n        with open('/etc/mylib/config.ini') as fh:\n            self.assertIn('enabled', fh.read())\n\n\nif __name__", 1)
    return spec

_coding_variant("c_i_test_needs_absent_file", "hard", _c_inf(_needs_file), "inf")


def _random_expect(rng, spec, names):
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace("import unittest", "import random\nimport unittest", 1).replace(
        "\n\nif __name__", "\n    def test_lucky(self):\n        self.assertEqual(m.total([1, 2, 3]) if hasattr(m, 'total') else 6, random.randint(10**9, 10**10))\n\n\nif __name__", 1)
    return spec

_coding_variant("c_i_random_expectation", "hard", _c_inf(_random_expect), "inf")


def _bool_contradiction(rng, spec, names):
    spec["files"]["mylib.py"] += "\n\n" + F["is_even"]["src"]
    spec["files"]["test_mylib.py"] = spec["files"]["test_mylib.py"].replace(
        "\n\nif __name__", "\n    def test_is_even(self):\n        self.assertTrue(m.is_even(3))\n        self.assertFalse(m.is_even(3))\n\n\nif __name__", 1)
    return spec

_coding_variant("c_i_true_and_false_same_call", "hard", _c_inf(_bool_contradiction), "inf")


# =========================================================================== сборка
def generate(out: Path, per: int, seed: int, only: set, match: str, levels: set) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    meta = {"seed": seed, "per": per, "tasks": {}}
    index = 0
    for entry in REG:
        if only and entry["type"] not in only:
            continue
        if match and match not in entry["name"]:
            continue
        if levels and entry["level"] not in levels:
            continue
        if os.environ.get("STRESS_RENAME") and (entry["name"] in (
                "c_missing_function", "c_class_method_bug", "c_plain_assert_tests", "c_bug_in_imported_helper")
                or entry["kind"] != "ok"):
            continue    # эти гипотезы собирают файлы в обход переименования
        count = per if entry["name"] not in ("d_f_100k_rows", "s_big_kb", "d_f_10k_rows", "m_no_ops") else 1
        for k in range(count):
            rng = random.Random(f"{seed}:{entry['name']}:{k}")
            spec = entry["fn"](rng)
            kind = spec.pop("kind", entry["kind"])
            task_type = entry["type"]
            task_id = f"{task_type}_{index:04d}"
            index += 1
            instruction = spec.get("instruction") or (spec["question"] + TAILS.get(task_type, TAIL_ANSWER))
            check = spec.get("check") or {"math": "exact_int", "search": "exact_str", "data_analysis": "exact_int",
                                          "coding": "unittest"}[task_type]
            if task_type == "search" and spec["expected"] is not None and spec["expected"].lstrip("-").isdigit():
                check = spec.get("check") or "exact_str"
            task_dir = out / task_id
            task_dir.mkdir()
            reward = rng.randint(1, 5)
            task = {"task_id": task_id, "task_type": task_type, "instruction": instruction, "reward": reward,
                    "expected": spec["expected"], "check": check}
            (task_dir / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            for rel, text in spec.get("files", {}).items():
                path = task_dir / "workspace" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="")
            if task_type != "math" and "files" in spec and spec["files"] and not (task_dir / "workspace").exists():
                (task_dir / "workspace").mkdir()
            if task_type != "math" and not (task_dir / "workspace").exists() and entry["name"] not in ("s_i_no_kb_dir",):
                (task_dir / "workspace").mkdir()
            meta["tasks"][task_id] = {
                "type": task_type, "feasible": spec["expected"] is not None, "variant": entry["name"],
                "level": entry["level"], "kind": kind, "expected": spec["expected"], "check": check, "reward": reward,
                "tests": spec.get("tests"), "fixed": spec.get("fixed"), "instruction": instruction,
            }
    meta_path = out.with_name(out.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    ap.add_argument("--per", type=int, default=4, help="сколько задач на каждую гипотезу")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="", help="типы через запятую")
    ap.add_argument("--match", default="", help="подстрока имени гипотезы")
    ap.add_argument("--levels", default="", help="уровни через запятую")
    ap.add_argument("--list", action="store_true", help="только перечислить гипотезы")
    args = ap.parse_args()
    if args.list:
        for e in REG:
            print(f"{e['type']:<14}{e['level']:<8}{e['kind']:<4}{e['name']}")
        print(f"total: {len(REG)} hypotheses")
        return 0
    if not args.out:
        ap.error("--out is required")
    only = {t.strip() for t in args.only.split(",") if t.strip()}
    levels = {t.strip() for t in args.levels.split(",") if t.strip()}
    meta = generate(Path(args.out).resolve(), args.per, args.seed, only, args.match, levels)
    by: dict[str, int] = {}
    for info in meta["tasks"].values():
        key = f"{info['type']}/{info['kind']}"
        by[key] = by.get(key, 0) + 1
    print(f"generated {len(meta['tasks'])} tasks from {len({t['variant'] for t in meta['tasks'].values()})} hypotheses: "
          + json.dumps(dict(sorted(by.items()))))
    return 0

if __name__ == "__main__":
    sys.exit(main())
