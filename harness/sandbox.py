"""Запуск кода в отдельном процессе.

Сюда попадает код, написанный моделью, и тесты из каталога задачи — то есть недоверенные данные.
Поэтому:
  * окружение дочернего процесса собирается по белому списку: ключ прокси (LLM_API_KEY)
    и адрес прокси в него не попадают;
  * у каждого запуска есть таймаут, по истечении убивается вся группа процессов;
  * вывод читается с ограничением по объёму, чтобы бесконечный print не съел память.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

_HEAD_LIMIT = 60_000
_TAIL_LIMIT = 20_000


@dataclass
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class _Capture:
    """Читает поток до конца, но хранит только начало и конец."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def drain(self, pipe) -> None:
        try:
            while True:
                chunk = pipe.read(8192)
                if not chunk:
                    break
                room = _HEAD_LIMIT - len(self.head)
                if room > 0:
                    self.head += chunk[:room]
                    chunk = chunk[room:]
                if chunk:
                    self.tail += chunk
                    if len(self.tail) > _TAIL_LIMIT:
                        self.dropped += len(self.tail) - _TAIL_LIMIT
                        del self.tail[: len(self.tail) - _TAIL_LIMIT]
        except (OSError, ValueError):
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    def text(self) -> str:
        data = bytes(self.head)
        if self.dropped:
            data += f"\n...[{self.dropped} bytes omitted]...\n".encode()
        data += bytes(self.tail)
        return data.decode("utf-8", errors="replace")


def clean_env(extra: Optional[dict] = None) -> dict:
    """Окружение дочернего процесса по белому списку (без ключей и адреса прокси)."""
    path_parts = [os.path.dirname(sys.executable)]
    path_parts += [p for p in os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin").split(os.pathsep) if p]
    env = {
        "PATH": os.pathsep.join(dict.fromkeys(path_parts)),
        "HOME": os.environ.get("HOME", "/tmp"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
    }
    if extra:
        env.update(extra)
    return env


def run_argv(argv: Sequence[str], cwd: str | Path, timeout: float, stdin_text: Optional[str] = None,
             env: Optional[dict] = None) -> ProcResult:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(argv), cwd=str(cwd), env=env or clean_env(),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as exc:
        return ProcResult(127, "", f"cannot start process: {exc}", False, time.monotonic() - started)

    out, err = _Capture(), _Capture()
    threads = [threading.Thread(target=out.drain, args=(proc.stdout,), daemon=True),
               threading.Thread(target=err.drain, args=(proc.stderr,), daemon=True)]
    for t in threads:
        t.start()
    if stdin_text is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_text.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for t in threads:
        t.join(timeout=3)
    return ProcResult(proc.returncode if proc.returncode is not None else -9,
                      out.text(), err.text(), timed_out, time.monotonic() - started)


def python_argv(*args: str, isolated: bool = False) -> list[str]:
    """Команда запуска того же интерпретатора; -B не пишет .pyc в каталог задачи."""
    argv = [sys.executable, "-B"]
    if isolated:
        argv.append("-I")
    argv.extend(args)
    return argv


def run_python_file(script: str | Path, cwd: str | Path, timeout: float, isolated: bool = True) -> ProcResult:
    return run_argv(python_argv(str(script), isolated=isolated), cwd, timeout)


def run_shell(command: str, cwd: str | Path, timeout: float) -> ProcResult:
    """Команда оболочки (для инструмента bash запасного агента)."""
    shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    return run_argv([shell, "-c", command], cwd, timeout)


def test_command_argv(command: str) -> Optional[list[str]]:
    """Превращает команду вида «python3 test_x.py» из инструкции в argv текущего интерпретатора."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    if parts[0] in ("python", "python3") or parts[0].startswith("python3."):
        return python_argv(*parts[1:])
    if parts[0] == "pytest":
        return python_argv("-m", "pytest", *parts[1:])
    return None
