"""Сборка всего вместе: настройка окружения, запуск планировщика, отчёт."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .config import Config, load_config


def prepare_process(scratch: Optional[Path] = None) -> Path:
    """Готовит процесс к работе в контейнере с файловой системой только для чтения.

    Писать можно только в /tmp и в каталог задач. Поэтому текущий каталог переносится в /tmp
    (openJiuwen и его зависимости создают служебные файлы относительно cwd и HOME), а всё,
    что могло бы полезть в сеть за пределы прокси, выключается заранее.
    """
    root = scratch or Path(tempfile.gettempdir()) / f"harness-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    for name, value in (
        ("HF_HUB_OFFLINE", "1"),               # никаких загрузок токенизаторов/моделей
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_HUB_DISABLE_TELEMETRY", "1"),
        ("OTEL_SDK_DISABLED", "true"),         # телеметрия openTelemetry не нужна и не должна ходить в сеть
        ("MPLCONFIGDIR", str(root / "mpl")),   # matplotlib пишет кеш в домашний каталог
        ("XDG_CACHE_HOME", str(root / "cache")),
    ):
        os.environ.setdefault(name, value)
    home = os.environ.get("HOME", "")
    if not home or not os.access(home, os.W_OK):
        os.environ["HOME"] = str(root)
    try:
        os.chdir(root)
    except OSError:
        pass
    return root


def quiet_openjiuwen(log_dir: Path) -> None:
    """openJiuwen по умолчанию пишет подробные логи в ./logs (а в них — тексты запросов).

    Каталог /app в контейнере доступен только для чтения, поэтому файловый вывод выключаем,
    а в консоль оставляем только критические сообщения: о сбоях вызовов харнесс сообщает сам.
    """
    try:
        from openjiuwen.core.common.logging.log_config import configure_log_config

        configure_log_config({
            "backend": "default",
            "level": "CRITICAL",
            "log_path": str(log_dir) + "/",
            "output": ["console"],
            "interface_output": ["console"],
            "performance_output": ["console"],
        })
    except Exception as exc:  # noqa: BLE001 — логирование не должно мешать запуску
        print(f"[harness] cannot configure openJiuwen logging: {exc}", flush=True)


def _log(message: str) -> None:
    print(message, flush=True)


def _write_report(path: Path, report: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


async def run(tasks_dir: str, cfg: Optional[Config] = None, scratch: Optional[Path] = None) -> dict:
    from .estimator import Estimator
    from .ledger import Ledger
    from .llm import LLMHub
    from .scheduler import Scheduler
    from .tasks import load_tasks

    started = time.monotonic()
    cfg = cfg or load_config()
    scratch = scratch or Path(tempfile.gettempdir()) / f"harness-{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.trace_dir or scratch) / "report.json"

    tasks = load_tasks(tasks_dir)
    by_type: dict[str, int] = {}
    for task in tasks:
        by_type[task.task_type] = by_type.get(task.task_type, 0) + 1
    _log(f"[harness] tasks: {len(tasks)} {by_type}; total reward {sum(t.reward for t in tasks):g}")
    cfg.scale_to(len(tasks), sum(t.reward for t in tasks))
    _log(f"[harness] budget={cfg.budget if cfg.budget is not None else 'not set'} (BUDGET={cfg.budget_raw!r}); "
         f"effective={cfg.effective_budget():.0f}; task cap={cfg.task_token_cap:.0f}; "
         f"tokens per point={cfg.tokens_per_point:.0f}")
    if cfg.budget_raw is not None and cfg.budget == cfg.unparsed_budget:
        _log(f"[harness] WARNING: cannot parse BUDGET={cfg.budget_raw!r}; assuming a cautious {cfg.unparsed_budget:.0f}")
    _log(f"[harness] models: fast={cfg.model_for('fast')} alt={cfg.model_for('fast_alt')} "
         f"strong={cfg.model_for('strong')}; fastpath={cfg.fastpath} crosscheck={cfg.crosscheck} "
         f"verify={cfg.verify}; parallel={cfg.max_parallel}; budget={cfg.effective_budget():.0f}; "
         f"proxy={'set' if cfg.base_url else 'NOT SET'}")
    if not tasks:
        _log(f"[harness] no tasks found in {tasks_dir}")
        return {"summary": {"tasks": 0}}

    ledger = Ledger(budget=cfg.budget, reasoning_in_completion=cfg.reasoning_in_completion)
    hub = LLMHub(cfg, ledger)
    estimator = Estimator()
    hub.on_call = estimator.observe_call
    scheduler = Scheduler(cfg, tasks, hub, estimator, scratch / "work", log=_log)

    last_dump = [0.0]

    def progress() -> None:
        hub.token_ratio = estimator.prompt_ratio
        if time.monotonic() - last_dump[0] > 20:
            last_dump[0] = time.monotonic()
            _write_report(report_path, scheduler.report())

    scheduler.on_progress = progress

    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()

    def on_signal() -> None:
        scheduler.stop_reason = scheduler.stop_reason or "terminated by signal"
        if main_task is not None:
            main_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, on_signal)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        await scheduler.run()
    except asyncio.CancelledError:
        _log("[harness] interrupted; answers written so far are kept")
        for solver in scheduler.solvers:
            scheduler._close(solver, "interrupted")
    finally:
        report = scheduler.report()
        _write_report(report_path, report)
        try:
            await asyncio.wait_for(hub.aclose(), timeout=5)
        except Exception:  # noqa: BLE001
            pass

    summary = report["summary"]
    _log(f"[harness] done in {time.monotonic() - started:.1f}s: answered {summary['answered']}/{summary['tasks']} "
         f"{summary['by_status']}; spent {summary['spent']:.0f} output-equivalent tokens "
         f"in {report['ledger']['calls']} calls"
         + (f"; stopped: {summary['stop_reason']}" if summary.get("stop_reason") else ""))
    led = report["ledger"]
    _log(f"[harness] budget: spent {led['spent']:.0f} of {cfg.effective_budget():.0f}; "
         f"calls {led['calls']}, failed {led.get('failed_calls', 0)}")
    _log(f"[harness] report: {report_path}")
    return report
