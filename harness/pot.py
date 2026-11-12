"""«Программа вместо рассуждения»: модель пишет короткое выражение или скрипт, считает харнесс.

Почему так. Выходные токены стоят в 10 раз дороже входных, а пошаговые рассуждения — это
именно выходные токены. Вместо «подумай и ответь» модель просят написать одно выражение Python
над уже загруженными данными (`rows` для csv, `kb` для базы знаний), а вычисляет его код.
Это дешевле (15–40 токенов ответа), точнее в арифметике и проверяемо: исключение при
вычислении — бесплатный сигнал, что ответ непригоден.

Код модели выполняется в отдельном процессе (см. sandbox.py), не в процессе харнесса.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import sandbox
from .textutil import normalize_answer

_MARK = "@@HARNESS_RESULT@@"
_ROOT = str(Path(__file__).resolve().parents[1])

# Исключения, означающие «в данных нет того, о чём спрашивают» (а не ошибку в коде).
MISSING_DATA_ERRORS = ("KeyError", "FileNotFoundError", "IndexError", "StopIteration")

_WRAPPER = r'''# -*- coding: utf-8 -*-
import sys, json, math, re, csv, os, collections, itertools, statistics, datetime, fractions, decimal, functools
from collections import Counter, defaultdict
from pathlib import Path
__PRELOAD__
__code = __CODE__
__mark = __MARKER__


def __render(v, depth=0):
    if v is None:
        return None
    item = getattr(v, "item", None)
    if callable(item) and type(v).__module__.split(".")[0] in ("numpy", "pandas"):
        try:
            v = item()
        except Exception:
            pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return str(v)
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
            return str(int(round(v)))
        return ("%.6f" % round(v, 6)).rstrip("0").rstrip(".")
    if isinstance(v, (fractions.Fraction, decimal.Decimal)):
        return __render(float(v), depth)
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple, set, frozenset)) and depth < 2:
        items = sorted(v, key=str) if isinstance(v, (set, frozenset)) else list(v)
        parts = [__render(x, depth + 1) for x in items]
        parts = [p for p in parts if p is not None]
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts) if parts else None
    return str(v).strip()


try:
    try:
        __compiled = compile(__code, "<answer>", "eval")
        __mode = "eval"
    except SyntaxError:
        __compiled = compile(__code, "<answer>", "exec")
        __mode = "exec"
    import io, contextlib
    __buf = io.StringIO()
    if __mode == "eval":
        # «print(expr)» — тоже выражение: его значение None, а ответ уходит в stdout. Перехватываем и его.
        with contextlib.redirect_stdout(__buf):
            __val = eval(__compiled, globals())
        if __val is None:
            __lines = [ln for ln in __buf.getvalue().splitlines() if ln.strip()]
            if __lines:
                __val = __lines[-1].strip()
    else:
        with contextlib.redirect_stdout(__buf):
            exec(__compiled, globals())
        __lines = [ln for ln in __buf.getvalue().splitlines() if ln.strip()]
        if __lines:
            __val = __lines[-1].strip()
        else:
            __val = globals().get("answer", globals().get("result"))
    __out = {"ok": True, "value": __render(__val), "mode": __mode}
except BaseException as __exc:
    __out = {"ok": False, "etype": type(__exc).__name__, "error": (type(__exc).__name__ + ": " + str(__exc))[:400]}
sys.stdout.write("\n" + __mark + json.dumps(__out) + "\n")
'''


@dataclass
class PotResult:
    ok: bool
    value: Optional[str] = None
    error: str = ""
    etype: str = ""
    timed_out: bool = False

    @property
    def missing_data(self) -> bool:
        return self.etype in MISSING_DATA_ERRORS


def import_harness_preamble() -> str:
    """Строки, дающие коду в песочнице доступ к загрузчикам из harness.skills."""
    return f"sys.path.insert(0, {_ROOT!r})\n"


def looks_like_code(text: str) -> bool:
    """Отсекает очевидный не-код (модель ответила фразой), чтобы не запускать процесс зря."""
    text = text.strip()
    if not text:
        return False
    if re.fullmatch(r"-?[\d.,]+", text):
        return True
    return bool(re.search(r"[()\[\]=+\-*/%]|\bprint\b|\bimport\b", text))


def run_pot(code: str, preload: str, cwd: Path, scratch: Path, timeout: float, tag: str = "pot") -> PotResult:
    """Вычисляет выражение/скрипт модели над предзагруженными данными и возвращает значение строкой."""
    scratch.mkdir(parents=True, exist_ok=True)
    script = scratch / f"{tag}_{os.getpid()}_{abs(hash(code)) % 10**8}.py"
    # код модели подставляется последним, чтобы его текст не задели остальные замены
    source = (_WRAPPER.replace("__MARKER__", repr(_MARK)).replace("__PRELOAD__", preload or "")
              .replace("__CODE__", repr(code)))
    try:
        script.write_text(source, encoding="utf-8")
    except OSError as exc:
        return PotResult(False, error=f"cannot write script: {exc}", etype="OSError")
    run_dir = cwd if cwd.is_dir() else scratch
    proc = sandbox.run_python_file(script, run_dir, timeout, isolated=True)
    try:
        script.unlink()
    except OSError:
        pass
    if proc.timed_out:
        return PotResult(False, error=f"timeout after {timeout:.0f}s", etype="Timeout", timed_out=True)
    pos = proc.stdout.rfind(_MARK)
    if pos == -1:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return PotResult(False, error=f"crashed: {tail[0][:300]}", etype="Crash")
    try:
        data = json.loads(proc.stdout[pos + len(_MARK):].strip().splitlines()[0])
    except (ValueError, IndexError):
        return PotResult(False, error="unreadable result", etype="Crash")
    if not data.get("ok"):
        return PotResult(False, error=str(data.get("error", "")), etype=str(data.get("etype", "")))
    value = data.get("value")
    if value is None or str(value).strip() == "":
        return PotResult(False, error="the code produced no value (None or empty)", etype="Empty")
    return PotResult(True, value=normalize_answer(str(value)))
