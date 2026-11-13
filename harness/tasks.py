"""Загрузка задач.

Из task.json читаются ТОЛЬКО четыре поля: task_id, task_type, instruction, reward.
Поля expected и check (правильный ответ и способ проверки), которые есть в примерах
организаторов, харнесс не читает и не передаёт модели: белый список ниже — единственное
место, где разбирается task.json.
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ALLOWED_FIELDS = ("task_id", "task_type", "instruction", "reward")
KNOWN_TYPES = ("math", "search", "data_analysis", "coding")
_INSTRUCTION_ALIASES = ("question", "prompt", "task", "description")
_MAX_DEPTH = 3  # на сколько уровней вглубь искать каталоги с task.json

# Хвост инструкции — служебные фразы про инструменты и формат сдачи. Модели они не нужны:
# файл answer.txt пишет сам харнесс.
_TAIL_MARKERS = (
    "Compute it with",
    "Use bash with",
    "Use the file tools",
    "Use the bash tool",
    "Work only inside the current directory",
    "When you are done",
)
_SENTINEL_RE = re.compile(r"write ONLY the word (\w+)", re.IGNORECASE)
# Вводные фразы, которые повторяют то, что харнесс и так передаёт модели в сжатом виде.
_PREAMBLE_RES = (
    re.compile(r"^The directory \S+ contains a small knowledge base of markdown files\.\s*"),
    re.compile(r"^The file \S+ has columns [\w,\s]+?(?:\(\d+ rows\))?\.\s*"),
)


@dataclass
class Task:
    task_id: str
    task_type: str
    instruction: str
    reward: float
    dir: Path
    workspace: Path
    question: str = ""          # инструкция без служебного хвоста
    sentinel: Optional[str] = None  # слово, которое нужно записать вместо ответа (DONE для coding)
    notes: list = field(default_factory=list)

    @property
    def has_workspace(self) -> bool:
        return self.workspace.is_dir()


def core_question(instruction: str) -> str:
    """Убирает из инструкции служебный хвост, оставляя саму задачу."""
    cut = len(instruction)
    for marker in _TAIL_MARKERS:
        pos = instruction.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    text = instruction[:cut].strip()
    return text if len(text) >= 10 else instruction.strip()


def strip_preamble(question: str) -> str:
    """Убирает известную вводную фразу («The directory kb/ contains…», «The file … has columns…»)."""
    for pattern in _PREAMBLE_RES:
        stripped = pattern.sub("", question, count=1)
        if stripped != question and len(stripped) >= 10:
            return stripped.strip()
    return question


def infer_type(task_type: str, instruction: str, workspace: Path) -> str:
    """Тип задачи: из поля task_type, а если его нет или он незнаком — по содержимому."""
    t = (task_type or "").strip().lower()
    if t in KNOWN_TYPES:
        return t
    for known in KNOWN_TYPES:
        if known in t:
            return known
    names = []
    if workspace.is_dir():
        names = [p.name for p in workspace.iterdir()]
    low = instruction.lower()
    if any(n.startswith("test_") and n.endswith(".py") for n in names) or "tests pass" in low:
        return "coding"
    if any(n.endswith(".csv") for n in names) or ".csv" in low:
        return "data_analysis"
    if any((workspace / n).is_dir() for n in names) and ("knowledge base" in low or "kb/" in low):
        return "search"
    if not names and re.search(r"\d", instruction):
        return "math"
    return t or "unknown"


def _log(msg: str) -> None:
    print(f"[harness] {msg}", file=sys.stderr, flush=True)


def _read_json(path: Path):
    """Читает JSON с учётом BOM и нестандартной кодировки; при ошибке бросает исключение."""
    blob = path.read_bytes()
    if blob.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = blob.decode("utf-16")
    else:
        try:
            text = blob.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = blob.decode("cp1251", errors="replace")
    return json.loads(text.lstrip("\ufeff"))


def _parse_reward(value) -> float:
    """reward: число или строка с числом; всё остальное (нет, NaN, <= 0) — 1.0."""
    try:
        reward = float(str(value).strip().replace(",", ".")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(reward) or reward <= 0:
        return 1.0
    return reward


def load_task(task_dir: Path) -> Optional[Task]:
    path = task_dir / "task.json"
    if not path.is_file():
        return None
    try:
        raw = _read_json(path)
    except (OSError, ValueError) as exc:
        _log(f"{path}: task.json не разобран ({exc}); задача пропущена")
        return None
    if not isinstance(raw, dict):
        _log(f"{path}: task.json не объект; задача пропущена")
        return None
    # ключи без учёта регистра и лишних пробелов
    norm = {str(k).strip().lower(): v for k, v in raw.items()}
    data = {key: norm.get(key) for key in ALLOWED_FIELDS}  # белый список полей
    if not data["instruction"]:
        for alias in _INSTRUCTION_ALIASES:
            if isinstance(norm.get(alias), str) and norm[alias].strip():
                data["instruction"] = norm[alias]
                break

    task_id = str(data["task_id"] or "").strip() or task_dir.name
    instruction = data["instruction"] if isinstance(data["instruction"], str) else str(data["instruction"] or "")
    reward = _parse_reward(data["reward"])
    workspace = task_dir / "workspace"
    task_type = infer_type(str(data["task_type"] or ""), instruction, workspace)
    sentinel_match = _SENTINEL_RE.search(instruction)
    return Task(
        task_id=task_id,
        task_type=task_type,
        instruction=instruction,
        reward=reward,
        dir=task_dir,
        workspace=workspace,
        question=core_question(instruction),
        sentinel=sentinel_match.group(1) if sentinel_match else None,
    )


def _find_task_dirs(root: Path, depth: int) -> list[Path]:
    """Каталоги с task.json: сам root, его потомки до глубины depth; внутрь задачи не заходим."""
    if (root / "task.json").is_file():
        return [root]
    found: list[Path] = []
    if depth <= 0:
        return found
    try:
        children = sorted(root.iterdir())
    except OSError:
        return found
    for child in children:
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        try:
            if child.is_dir():
                found.extend(_find_task_dirs(child, depth - 1))
        except OSError:
            continue
    return found


def load_tasks(tasks_dir: str | Path) -> list[Task]:
    root = Path(tasks_dir)
    tasks: list[Task] = []
    if not root.is_dir():
        return tasks
    seen: set[str] = set()
    for d in _find_task_dirs(root, _MAX_DEPTH):
        try:
            task = load_task(d)
        except Exception as exc:  # одна кривая задача не должна ронять прогон
            _log(f"{d}: ошибка загрузки задачи ({exc!r}); пропущена")
            continue
        if task is None:
            continue
        if task.task_id in seen:  # одинаковые task_id: различаем по имени каталога
            task.task_id = f"{task.task_id}@{d.name}"
        seen.add(task.task_id)
        tasks.append(task)
    return tasks
