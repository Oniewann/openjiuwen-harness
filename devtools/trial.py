#!/usr/bin/env python3
"""Пробный прогон харнесса на наборе задач с проверкой ответов.

    python devtools/trial.py --tasks ../train/mixed100                      # настоящие модели (LLM_BASE_URL, LLM_API_KEY)
    python devtools/trial.py --tasks ../train/mixed100 --mock oracle        # локальная заглушка вместо прокси
    python devtools/trial.py --tasks ../task-dir --no-llm                   # только навыки, без модели
    python devtools/trial.py --tasks ../train/mixed100 --mock oracle -e HARNESS_FASTPATH=0 -e MOCK_BAD_FAST=0.3

Что делает:
  1. копирует набор во временный каталог (оригинал не трогается) и убирает из копий task.json
     поля expected и check — как в финальном наборе;
  2. при --mock поднимает devtools/mock_llm.py с нужным «мозгом» и бюджетом;
  3. запускает run_agent.py отдельным процессом, как это сделает жюри;
  4. проверяет ответы (devtools/evaluate.py) и печатает баллы, ошибки и расход токенов.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import evaluate as ev  # noqa: E402
import mock_llm  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _copy_tasks(src: Path, dst: Path) -> dict:
    """Копия набора без правильных ответов. Возвращает «истину» для проверки, если нет meta."""
    truth = {}
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", "answer.txt"))
    for task_json in dst.glob("*/task.json"):
        task = json.loads(task_json.read_text(encoding="utf-8"))
        truth[task.get("task_id", task_json.parent.name)] = {
            "expected": task.get("expected"), "check": task.get("check", "exact_str"),
            "feasible": task.get("expected") is not None, "variant": "", "type": task.get("task_type"),
        }
        public = {k: v for k, v in task.items() if k not in ("expected", "check")}
        task_json.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="каталог набора (оригинал)")
    ap.add_argument("--meta", default=None, help="meta генератора (по умолчанию <tasks>.meta.json, если есть)")
    ap.add_argument("--mock", default=None, help="мозг заглушки: oracle | echo")
    ap.add_argument("--no-llm", action="store_true", help="не давать харнессу адрес прокси")
    ap.add_argument("--budget", type=float, default=None, help="бюджет токенов (BUDGET и лимит заглушки)")
    ap.add_argument("--latency", type=float, default=0.0)
    ap.add_argument("--reasoning-tokens", type=int, default=400)
    ap.add_argument("-e", "--env", action="append", default=[], help="KEY=VALUE для харнесса и заглушки")
    ap.add_argument("--keep", action="store_true", help="не удалять рабочую копию")
    ap.add_argument("--out", default=None, help="куда сохранить итог (json)")
    ap.add_argument("--show-log", type=int, default=12, help="сколько последних строк лога харнесса показать")
    args = ap.parse_args()

    src = Path(args.tasks).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else src.with_name(src.name + ".meta.json")
    if not meta_path.is_file():
        meta_path = None
    work = Path(tempfile.mkdtemp(prefix="trial-"))
    tasks_copy = work / "tasks"
    truth = _copy_tasks(src, tasks_copy)
    report_dir = work / "report"
    report_dir.mkdir()

    env = dict(os.environ)
    env["HARNESS_TRACE_DIR"] = str(report_dir)
    extra = dict(item.split("=", 1) for item in args.env)
    env.update(extra)
    server = state = None
    if args.no_llm:
        env.pop("LLM_BASE_URL", None)
        env.pop("LLM_API_KEY", None)
    elif args.mock:
        os.environ.update(extra)
        if meta_path:
            os.environ["MOCK_TASKS"], os.environ["MOCK_META"] = str(src), str(meta_path)
        port = _free_port()
        log_path = str(work / "mock_requests.jsonl")
        server, state = mock_llm.serve(port, args.mock, log_path, args.budget, "error", args.latency,
                                       args.reasoning_tokens)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env["LLM_BASE_URL"], env["LLM_API_KEY"] = f"http://127.0.0.1:{port}/v1", "mock-key"
        env.setdefault("HARNESS_RPM", "0")
    if args.budget is not None and "BUDGET" not in extra:
        env["BUDGET"] = str(int(args.budget))

    started = time.monotonic()
    proc = subprocess.run([sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(tasks_copy)],
                          env=env, capture_output=True, text=True)
    seconds = time.monotonic() - started
    log = (proc.stdout or "") + (proc.stderr or "")
    (work / "harness.log").write_text(log, encoding="utf-8")
    if args.show_log:
        print("\n".join(log.strip().splitlines()[-args.show_log:]))
    print(f"--- harness exit code {proc.returncode}, {seconds:.1f}s; work dir: {work}")

    report_path = report_dir / "report.json"
    if meta_path is None:
        # примеры организаторов: правильные ответы берём из их task.json
        meta_path = work / "truth.meta.json"
        meta_path.write_text(json.dumps({"tasks": truth}), encoding="utf-8")
    result = ev.evaluate(tasks_copy, meta_path, report_path if report_path.is_file() else None)
    ev.print_result(result)
    if state is not None:
        print(f"--- mock proxy: {state.calls} calls, {state.spent:.0f} tokens by the proxy's own count"
              + (f" (budget {state.budget:.0f})" if state.budget else ""))
        result["proxy_spent"], result["proxy_calls"] = state.spent, state.calls
        server.shutdown()
    leftovers = [str(p.relative_to(tasks_copy)) for p in tasks_copy.rglob("*")
                 if p.name in ("__pycache__", ".pytest_cache") or p.suffix == ".pyc"
                 or (p.name == "answer.txt" and p.parent.name == "workspace")]
    if leftovers:
        print(f"--- WARNING: leftovers in the tasks dir: {leftovers[:8]}")
    result["seconds"], result["work_dir"], result["exit_code"] = seconds, str(work), proc.returncode
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.keep and not args.out:
        pass  # каталог оставляем: в нём лог и отчёт; он лежит в системном temp и чистится сам
    return 0


if __name__ == "__main__":
    sys.exit(main())
