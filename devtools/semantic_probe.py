#!/usr/bin/env python3
"""Сверка навыка эталонов со «скрытыми» проверками: каждая функция стресс-набора x каждый вид бага.

    python devtools/semantic_probe.py [--match maximum] [--others 2] [--seed 1] [-v]

Для каждой пары (функция, вид бага) из devtools/stress_gen.py собирается задача как у организаторов
(файл с несколькими функциями, баг роняет видимый тест), затем повторяется путь харнесса без модели:
reference_repair, при неудаче — mutation_repair, после правки — align_touched. Итог проверяется скрытыми
тестами (видимые + дополнительные). Исходы:
    ref     — починил эталон, скрытые тесты проходят;
    mut     — починил перебор, скрытые тесты проходят;
    HIDDEN  — видимые тесты проходят, скрытые падают (подгонка: в финале задача потеряна);
    fail    — без модели не починилось (уйдёт модели).
Дополнительно (--hidden-bugs): баг, который видимый тест не ловит, рядом с обычным багом в другой функции,
и исправная функция с необычной семантикой — её трогать нельзя.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))

import stress_gen as SG  # noqa: E402

from harness.skills.repair import mutation_repair  # noqa: E402
from harness.skills.semantic import reference_repair  # noqa: E402

try:
    from harness.skills.semantic import align_touched  # noqa: E402
except ImportError:  # старая версия навыка: без дочинки до эталона
    align_touched = None
from harness.skills.testrun import find_test_command, run_tests  # noqa: E402

INSTR = "Run `python3 test_mylib.py`"


def _write(ws: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (ws / rel).write_text(text, encoding="utf-8")


def solve(lib: str, vis: str, hid: str) -> tuple[str, str]:
    """(исход, итоговый текст mylib.py) — путь харнесса без модели."""
    tmp = Path(tempfile.mkdtemp(prefix="sprobe_"))
    try:
        _write(tmp, {"mylib.py": lib, "test_mylib.py": vis})
        argv = find_test_command(INSTR, tmp)
        base = run_tests(tmp, argv, 20)
        if base.ok:
            return "noop", lib
        how = "fail"
        res = reference_repair(tmp, argv, base)
        _write(tmp, res.texts)
        if res.report is not None and res.report.ok:
            how = "ref"
        else:
            mr = mutation_repair(tmp, argv, time_budget=40.0, test_timeout=5.0)
            if mr.fixed:
                how = "mut"
        final = (tmp / "mylib.py").read_text(encoding="utf-8")
        if how != "fail" and align_touched is not None:
            al = align_touched(tmp, argv, {"mylib.py": lib}, {"mylib.py": final})
            if al.texts:
                final = al.texts["mylib.py"]
                how += "+align"
        if how == "fail":
            return how, final
        _write(tmp, {"mylib.py": final, "test_mylib.py": hid})
        if not run_tests(tmp, argv, 20).ok:
            return "HIDDEN(" + how + ")", final
        return how, final
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build(names: list[str], bugs: dict[str, tuple[str, str]], hidden_extra: dict[str, list[str]] | None = None):
    lib = "\n\n".join(SG._apply_bug(SG.F[n]["src"], bugs[n]) if n in bugs else SG.F[n]["src"] for n in names)
    vis = SG.test_file([(n, list(SG.F[n]["vis"])) for n in names])
    hid = SG.test_file([(n, list(SG.F[n]["vis"]) + list(SG.F[n]["hid"]) + list((hidden_extra or {}).get(n, [])))
                        for n in names])
    return lib, vis, hid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="")
    ap.add_argument("--others", type=int, default=2, help="сколько исправных функций рядом")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hidden-bugs", action="store_true", help="также баги, которые видимый тест не ловит")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pool = [k for k in SG.F if k != "binary_search"]
    counts: dict[str, int] = {}
    bad = []
    for name, f in SG.F.items():
        if args.match and args.match not in name:
            continue
        for kind, bug in f["bugs"].items():
            others = rng.sample([k for k in pool if k != name], args.others)
            names = others + [name]
            rng.shuffle(names)
            lib, vis, hid = build(names, {name: bug})
            alone_vis = SG.test_file([(name, list(f["vis"]))])
            visible = not SG.run_tests({"mylib.py": SG._apply_bug(f["src"], bug), "test_mylib.py": alone_vis}, timeout=5)
            if not visible and not args.hidden_bugs:
                continue
            if not visible:
                # баг виден только скрытым тестам: рядом ставим обычный баг в другой функции
                helper = next((k for k in others if any(b for b in SG.F[k]["bugs"])), None)
                if helper is None:
                    continue
                hk = next(iter(SG.F[helper]["bugs"]))
                lib, vis, hid = build(names, {name: bug, helper: SG.F[helper]["bugs"][hk]})
            out, final = solve(lib, vis, hid)
            label = ("" if visible else "invisible:") + out
            counts[label] = counts.get(label, 0) + 1
            line = f"{name:16} {kind:9} {label}"
            if out.startswith("HIDDEN") or args.verbose:
                print(line)
            if out.startswith("HIDDEN"):
                bad.append((name, kind, final))
    if args.hidden_bugs:
        # исправные функции с необычной семантикой рядом с обычным багом: их трогать нельзя
        odd = [("unique", "def unique(xs):\n    return sorted(set(xs))\n"),
               ("maximum", "def maximum(xs):\n    if not xs:\n        return None\n    return max(xs)\n"),
               ("is_palindrome", "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]\n"),
               ("factorial", "def factorial(n):\n    if n < 0:\n        raise ValueError('negative')\n    result = 1\n"
                             "    for i in range(2, n + 1):\n        result *= i\n    return result\n"),
               ("clamp", "def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n"),
               ("capitalize_words", "def capitalize_words(s):\n    return ' '.join(w[0].upper() + w[1:] for w in s.split())\n"),
               ("count_odd", "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 != 0)\n"),
               ("longest_word", "def longest_word(s):\n    best = ''\n    for w in s.split():\n        if len(w) > len(best):\n"
                                "            best = w\n    return best\n"),
               ("average", "def average(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)\n"),
               ("second_largest", "def second_largest(xs):\n    ys = sorted(set(xs))\n    if len(ys) < 2:\n        return None\n"
                                  "    return ys[-2]\n"),
               ("power", "def power(base, exp):\n    return base ** exp\n"),
               ("countdown", "def countdown(n):\n    return list(range(n, 0, -1))\n"),
               ("median", "def median(xs):\n    ys = sorted(xs)\n    n = len(ys)\n    if n % 2:\n        return ys[n // 2]\n"
                          "    return (ys[n // 2 - 1] + ys[n // 2]) / 2\n")]
        for name, src in odd:
            lib = src + "\n\n" + SG._apply_bug(SG.F["total"]["src"], SG.F["total"]["bugs"]["op"])
            vis = SG.test_file([(name, list(SG.F[name]["vis"])), ("total", list(SG.F["total"]["vis"]))])
            out, final = solve(lib, vis, vis)
            kept = src.strip() in final
            label = "odd:" + ("kept" if kept else "CHANGED")
            counts[label] = counts.get(label, 0) + 1
            if not kept or args.verbose:
                print(f"{name:16} {'odd':9} {label}")
    if args.hidden_bugs:
        # проверки входа в исходнике (скрытые тесты любят пустые входы): эталон не должен их терять
        guarded = [
            ("average", "def average(xs):\n    if not xs:\n        return 0\n    return sum(xs) * len(xs)\n",
             ["self.assertEqual(m.average([]), 0)"]),
            ("second_largest", "def second_largest(xs):\n    if len(set(xs)) < 2:\n        return None\n    return sorted(set(xs))[-1]\n",
             ["self.assertIsNone(m.second_largest([5]))"]),
            ("factorial", "def factorial(n):\n    if n < 0:\n        raise ValueError('negative')\n    result = 1\n"
                          "    for i in range(2, n + 1):\n        result += i\n    return result\n",
             ["self.assertRaises(ValueError, m.factorial, -1)"]),
            ("maximum", "def maximum(xs):\n    if not xs:\n        return None\n    best = xs[0]\n    for x in xs:\n"
                        "        if x < best:\n            best = x\n    return best\n",
             ["self.assertIsNone(m.maximum([]))"]),
        ]
        for name, src, extra in guarded:
            lib = SG.F["total"]["src"] + "\n\n" + src
            vis = SG.test_file([("total", list(SG.F["total"]["vis"])), (name, list(SG.F[name]["vis"]))])
            hid = SG.test_file([("total", list(SG.F["total"]["vis"])), (name, list(SG.F[name]["vis"]) + extra)])
            out, final = solve(lib, vis, hid)
            label = "guard:" + out
            counts[label] = counts.get(label, 0) + 1
            if out.startswith("HIDDEN") or args.verbose:
                print(f"{name:16} {'guard':9} {label}")
    print("итого:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for name, kind, final in bad[:5]:
        print(f"--- {name}/{kind}\n{final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
