"""Запись ответа.

Организаторы уточнили: answer.txt кладётся рядом с task.json (не в workspace/), а для задачи,
которую харнесс считает невыполнимой, файл не создаётся вовсе.

Запись атомарная (временный файл + rename): если прогон оборвут посреди записи,
в каталоге останется либо старый ответ целиком, либо новый целиком.
"""
from __future__ import annotations

import os
from pathlib import Path

from .tasks import Task

ANSWER_FILE = "answer.txt"


def _atomic_write(path: Path, text: str) -> bool:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
    # Запасной путь: каталог может не давать создавать новые файлы, но разрешать перезапись.
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except OSError:
        return False


def write_answer(task: Task, answer: str, also_workspace: bool = False) -> bool:
    """Пишет ответ в каталог задачи (рядом с task.json). True, если запись удалась."""
    text = answer.strip() + "\n"
    ok = _atomic_write(task.dir / ANSWER_FILE, text)
    if also_workspace and task.workspace.is_dir():
        ok = _atomic_write(task.workspace / ANSWER_FILE, text) or ok
    return ok


def read_workspace_answer(task: Task) -> str | None:
    """Ответ, который агент сам записал в workspace/answer.txt (режим ReAct)."""
    for path in (task.workspace / ANSWER_FILE, task.dir / ANSWER_FILE):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return None


def clear_answer(task: Task) -> None:
    for path in (task.dir / ANSWER_FILE, task.workspace / ANSWER_FILE):
        try:
            path.unlink()
        except OSError:
            pass
