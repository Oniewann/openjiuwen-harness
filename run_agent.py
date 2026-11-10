#!/usr/bin/env python3
"""Точка входа харнесса.

    python /app/run_agent.py --tasks-dir /data/tasks

Переменные окружения: LLM_BASE_URL и LLM_API_KEY (адрес и ключ прокси организаторов).
В каждом каталоге задачи создаётся answer.txt — сразу после того, как задача решена.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Budget-aware coding agent harness on openJiuwen")
    parser.add_argument("--tasks-dir", default=os.environ.get("TASKS") or "/data/tasks",
                        help="каталог с задачами (подкаталоги с task.json)")
    parser.add_argument("--out", default=None, help="не используется (осталось от ранней версии регламента)")
    args, _unknown = parser.parse_known_args()

    from harness.runner import prepare_process, quiet_openjiuwen, run

    tasks_dir = str(Path(args.tasks_dir).resolve())
    scratch = prepare_process()
    quiet_openjiuwen(scratch / "logs")
    try:
        asyncio.run(run(tasks_dir, scratch=scratch))
    except KeyboardInterrupt:
        print("[harness] interrupted", flush=True)
    except Exception:  # noqa: BLE001 — ответы уже на диске; код возврата не должен их обесценить
        traceback.print_exc()
        print("[harness] finished with an internal error; answers written so far are kept", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
