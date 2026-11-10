#!/usr/bin/env python3
"""Локальная проверка ответов харнесса — так, как их, по нашему пониманию, проверит жюри.

    python devtools/evaluate.py --tasks DIR [--meta FILE] [--report run_report.json] [--json OUT.json] [--quiet]

Ответ берётся из DIR/<задача>/answer.txt (организаторы: «рядом с task.json»); если его там нет,
а есть workspace/answer.txt — это отмечается отдельно и НЕ засчитывается.
Проверки: exact_int, exact_str, unittest (ответ DONE и тесты проходят на эталонных тестовых файлах).
Правильные ответы берутся из meta генератора, а если его нет — из полей expected/check в task.json
(так устроены примеры организаторов). Харнесс эти поля не читает; читает только этот скрипт.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def _read_answer(task_dir: Path) -> tuple[Optional[str], str]:
    primary = task_dir / "answer.txt"
    if primary.is_file():
        return primary.read_text(encoding="utf-8", errors="replace").strip(), "task_dir"
    secondary = task_dir / "workspace" / "answer.txt"
    if secondary.is_file():
        return secondary.read_text(encoding="utf-8", errors="replace").strip(), "workspace_only"
    return None, "none"


def _check_unittest(task_dir: Path, instruction: str, pristine_tests: Optional[dict]) -> tuple[bool, str]:
    workspace = task_dir / "workspace"
    if not workspace.is_dir():
        return False, "no workspace"
    match = re.search(r"`(python3? [^`]+)`", instruction)
    command = match.group(1).split() if match else ["python3", "-m", "unittest", "discover"]
    command[0] = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "ws"
        shutil.copytree(workspace, copy, ignore=shutil.ignore_patterns("__pycache__", "answer.txt"))
        for rel, text in (pristine_tests or {}).items():   # защита от правки тестов
            (copy / rel).write_text(text, encoding="utf-8")
        try:
            proc = subprocess.run([command[0], "-B", *command[1:]], cwd=copy, capture_output=True, text=True,
                                  timeout=60)
        except subprocess.TimeoutExpired:
            return False, "tests timed out"
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
    return False, tail[0][:120]


def _check(check: str, expected, answer: str, task_dir: Path, instruction: str, tests: Optional[dict]) -> tuple[bool, str]:
    if check == "exact_int":
        try:
            return int(answer.strip()) == int(str(expected).strip()), ""
        except ValueError:
            return False, "answer is not an integer"
    if check == "exact_str":
        return answer.strip() == str(expected).strip(), ""
    if check == "unittest":
        if answer.strip() != "DONE":
            return False, "answer is not DONE"
        return _check_unittest(task_dir, instruction, tests)
    return answer.strip() == str(expected).strip(), f"unknown check {check!r}"


def _spent_by_task(report: Optional[dict]) -> tuple[Optional[float], dict]:
    if not report:
        return None, {}
    total = None
    for path in (("summary", "spent"), ("ledger", "spent"), ("spent",), ("tokens",)):
        node = report
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, (int, float)):
            total = float(node)
            break
    per_task = {}
    for item in report.get("tasks", []) if isinstance(report.get("tasks"), list) else []:
        if isinstance(item, dict) and "task_id" in item:
            per_task[item["task_id"]] = float(item.get("spent", item.get("cost", 0.0)) or 0.0)
    return total, per_task


def evaluate(tasks_dir, meta_path=None, report_path=None) -> dict:
    root = Path(tasks_dir)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))["tasks"] if meta_path else {}
    report = json.loads(Path(report_path).read_text(encoding="utf-8")) if report_path else None
    spent_total, spent_by_task = _spent_by_task(report)
    rows = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p / "task.json").is_file()):
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        task_id = task.get("task_id", task_dir.name)
        info = meta.get(task_id, {})
        expected = info.get("expected", task.get("expected"))
        check = info.get("check", task.get("check", "exact_str"))
        feasible = info.get("feasible", expected is not None)
        answer, where = _read_answer(task_dir)
        ok, why = False, ""
        if answer is not None and where == "task_dir" and feasible:
            ok, why = _check(check, expected, answer, task_dir, task.get("instruction", ""), info.get("tests"))
        elif where == "workspace_only":
            why = "answer.txt only in workspace/"
        rows.append({
            "task_id": task_id, "type": task.get("task_type", "?"), "reward": float(task.get("reward", 1)),
            "feasible": bool(feasible), "variant": info.get("variant", ""), "answer": answer, "where": where,
            "expected": expected, "correct": bool(ok), "why": why, "spent": spent_by_task.get(task_id, 0.0),
        })
    by_type: dict[str, dict] = {}
    for row in rows:
        t = by_type.setdefault(row["type"], {"tasks": 0, "feasible": 0, "correct": 0, "wrong": 0, "no_answer": 0,
                                             "answered_infeasible": 0, "score": 0.0, "max": 0.0, "spent": 0.0})
        t["tasks"] += 1
        t["spent"] += row["spent"]
        if row["feasible"]:
            t["feasible"] += 1
            t["max"] += row["reward"]
            if row["correct"]:
                t["correct"] += 1
                t["score"] += row["reward"]
            elif row["answer"] is None:
                t["no_answer"] += 1
            else:
                t["wrong"] += 1
        elif row["answer"] is not None:
            t["answered_infeasible"] += 1
    return {
        "score": sum(t["score"] for t in by_type.values()),
        "max_score": sum(t["max"] for t in by_type.values()),
        "tasks": len(rows),
        "feasible": sum(t["feasible"] for t in by_type.values()),
        "correct": sum(t["correct"] for t in by_type.values()),
        "wrong": [r for r in rows if r["feasible"] and r["answer"] is not None and not r["correct"]],
        "false_refusals": [r for r in rows if r["feasible"] and r["answer"] is None],
        "answered_infeasible": [r for r in rows if not r["feasible"] and r["answer"] is not None],
        "spent": spent_total,
        "by_type": by_type,
        "rows": rows,
    }


def print_result(result: dict) -> None:
    print(f"SCORE {result['score']:g} / {result['max_score']:g}   "
          f"correct {result['correct']}/{result['feasible']} feasible ({result['tasks']} tasks)"
          + (f"   tokens {result['spent']:.0f}" if result["spent"] is not None else ""))
    print(f"{'type':<15}{'tasks':>6}{'feas':>6}{'ok':>5}{'wrong':>6}{'none':>6}{'ans.inf':>8}{'score':>8}{'max':>6}{'tokens':>9}")
    for name, t in sorted(result["by_type"].items()):
        print(f"{name:<15}{t['tasks']:>6}{t['feasible']:>6}{t['correct']:>5}{t['wrong']:>6}{t['no_answer']:>6}"
              f"{t['answered_infeasible']:>8}{t['score']:>8g}{t['max']:>6g}{t['spent']:>9.0f}")
    for title, items in (("WRONG", result["wrong"]), ("NO ANSWER (feasible)", result["false_refusals"]),
                         ("ANSWERED INFEASIBLE", result["answered_infeasible"])):
        if items:
            print(f"-- {title}: {len(items)}")
            for r in items[:40]:
                print(f"   {r['task_id']:<22} reward {r['reward']:g} [{r['variant']}] expected={str(r['expected'])[:30]!r} "
                      f"got={str(r['answer'])[:30]!r} {r['why']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    meta = args.meta
    if meta is None:
        guess = Path(args.tasks).resolve()
        guess = guess.with_name(guess.name + ".meta.json")
        meta = str(guess) if guess.is_file() else None
    result = evaluate(args.tasks, meta, args.report)
    if not args.quiet:
        print_result(result)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
