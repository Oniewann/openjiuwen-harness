#!/usr/bin/env python3
"""Сводка прогона стресс-набора по гипотезам.

    python devtools/trial.py --tasks DIR --no-llm --show-log 0 --out result.json
    python devtools/stress_report.py --result result.json --meta DIR.meta.json [--bad] [--show wrong,none,static] [--md FILE]

Исходы для выполнимой задачи:
    ok      — ответ верный;
    WRONG   — ответ записан и он неверный (потеря баллов: хуже всего);
    STATIC  — харнесс без модели объявил задачу невыполнимой (потеря баллов без шанса на модель);
    llm     — навык отказался, задача уйдёт модели (баллы зависят от модели, тратятся токены).
Для невыполнимой: skip (файла нет, токены не тратились), llm (уйдёт модели), answered (ответ записан — баллы не теряются).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def outcome(row: dict, info: dict, rep: dict) -> str:
    static = (rep.get("p_infeasible_static") or 0.0) >= 0.95 or rep.get("status") == "infeasible"
    if info["feasible"]:
        if row["correct"]:
            return "ok"
        if row["answer"] is not None:
            return "WRONG"
        return "STATIC" if static else "llm"
    if row["answer"] is not None:
        return "answered"
    return "skip" if static else "llm"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--bad", action="store_true", help="только гипотезы, где не всё идеально")
    ap.add_argument("--show", default="", help="показать примеры задач с этими исходами (через запятую)")
    ap.add_argument("--examples", type=int, default=1)
    ap.add_argument("--md", default="", help="записать таблицу в markdown")
    args = ap.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))["tasks"]
    report = {}
    rp = Path(result["work_dir"]) / "report" / "report.json"
    if rp.is_file():
        report = {t["task_id"]: t for t in json.loads(rp.read_text(encoding="utf-8")).get("tasks", [])}

    by: dict[str, dict] = {}
    order: list[str] = []
    for row in result["rows"]:
        info = meta[row["task_id"]]
        rep = report.get(row["task_id"], {})
        o = outcome(row, info, rep)
        key = info["variant"]
        if key not in by:
            by[key] = {"type": info["type"], "level": info["level"], "kind": info["kind"], "n": 0, "out": {}, "ex": {}}
            order.append(key)
        by[key]["n"] += 1
        by[key]["out"][o] = by[key]["out"].get(o, 0) + 1
        by[key]["ex"].setdefault(o, []).append((row, info, rep))

    show = {s.strip() for s in args.show.split(",") if s.strip()}
    totals: dict[str, dict] = {}
    lines = []
    for key in order:
        b = by[key]
        t = totals.setdefault(f"{b['type']}/{b['kind']}", {})
        for o, c in b["out"].items():
            t[o] = t.get(o, 0) + c
        good = ("ok",) if b["kind"] != "inf" else ("skip", "answered")
        perfect = all(o in good for o in b["out"])
        if args.bad and perfect:
            continue
        outs = " ".join(f"{o}={c}" for o, c in sorted(b["out"].items()))
        lines.append((b["type"], b["level"], b["kind"], key, b["n"], outs))
        print(f"{b['type']:<14}{b['level']:<8}{b['kind']:<4}{key:<36}{b['n']:>3}  {outs}")
        for o in sorted(b["out"]):
            if o in show:
                for row, info, rep in b["ex"][o][: args.examples]:
                    q = info["instruction"].split(" Compute it with")[0].split(" Use the file tools")[0].split(" Use bash with")[0]
                    static = [tr.get("reason") for tr in rep.get("trace", []) if tr.get("step") == "static"]
                    print(f"      [{o}] {row['task_id']}: {q[-230:]!r}")
                    print(f"            expected={row['expected']!r} got={row['answer']!r} {row['why']} {static or ''}")
    print("\n== totals")
    for key in sorted(totals):
        print(f"{key:<20}" + " ".join(f"{o}={c}" for o, c in sorted(totals[key].items())))
    if args.md:
        md = ["| тип | уровень | вид | гипотеза | задач | исход |", "|---|---|---|---|---|---|"]
        md += [f"| {a} | {b} | {c} | `{d}` | {e} | {f} |" for a, b, c, d, e, f in lines]
        Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
