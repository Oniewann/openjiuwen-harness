"""«Мозг-оракул» для заглушки LLM: отвечает как идеальная модель, зная эталон генератора.

Нужен, чтобы без ключа и без сети прогонять ветку харнесса с моделью: промпты, разбор
ответов, эскалации, ReAct, учёт токенов. К качеству настоящих моделей отношения не имеет.

Настройка через переменные окружения:
    MOCK_TASKS       каталог с задачами (нетронутый оригинал, не копия, в которой работает харнесс)
    MOCK_META        meta-файл генератора
    MOCK_BAD_FAST    доля испорченных ответов дешёвой основной модели (0..1)
    MOCK_BAD_ALT     то же для второй дешёвой модели
    MOCK_BAD_STRONG  то же для модели с рассуждениями
    MOCK_HALLUCINATE доля невыполнимых задач, на которые дешёвая модель «выдумывает» ответ
    MOCK_SEED        зерно (порча детерминирована: зависит от задачи, модели и зерна)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

_TAIL_MARKERS = ("Compute it with", "Use bash with", "Use the file tools", "Work only inside", "When you are done")
_PREAMBLES = (re.compile(r"^The directory \S+ contains a small knowledge base of markdown files\.\s*"),
              re.compile(r"^The file \S+ has columns [\w,\s]+?(?:\(\d+ rows\))?\.\s*"))

_INDEX: list[dict] = []
_LOADED = False
_LOCK = threading.Lock()


def _core(instruction: str) -> str:
    cut = len(instruction)
    for marker in _TAIL_MARKERS:
        pos = instruction.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    text = instruction[:cut].strip()
    for pattern in _PREAMBLES:
        text = pattern.sub("", text, count=1)
    return " ".join(text.split())


def _load() -> None:
    """Строит указатель задач один раз; параллельные запросы ждут, пока он не будет готов."""
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        _build_index()
        _LOADED = True


def _build_index() -> None:
    tasks_dir, meta_path = os.environ.get("MOCK_TASKS"), os.environ.get("MOCK_META")
    if not tasks_dir or not meta_path:
        return
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))["tasks"]
    for task_id, info in meta.items():
        task_dir = Path(tasks_dir) / task_id
        try:
            task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        except OSError:
            continue
        entry = {"task_id": task_id, "info": info, "instruction": task["instruction"],
                 "core": _core(task["instruction"]), "sources": {}, "tests": info.get("tests") or {}}
        if info["type"] == "coding":
            for path in (task_dir / "workspace").glob("*.py"):
                if not path.name.startswith("test_"):
                    entry["sources"][path.name] = path.read_text(encoding="utf-8")
        _INDEX.append(entry)


def _find(text: str, family: str):
    flat = " ".join(text.split())
    best = None
    for entry in _INDEX:
        info = entry["info"]
        if family == "coding":
            if info["type"] != "coding":
                continue
            if any(src.strip() and src.strip() in text for src in entry["sources"].values()):
                return entry
            if any(t.strip() and t.strip() in text for t in entry["tests"].values()):
                best = best or entry
        elif family == "react":
            if entry["instruction"].strip() in text:
                return entry
        elif entry["core"] and entry["core"] in flat and info["type"] != "coding":
            if best is None or len(entry["core"]) > len(best["core"]):
                best = entry
    return best


def _roll(entry, model: str, salt: str) -> float:
    seed = os.environ.get("MOCK_SEED", "0")
    digest = hashlib.sha256(f"{seed}|{entry['task_id']}|{model}|{salt}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _rate(model: str) -> float:
    if model.endswith("-noreason"):
        name = "MOCK_BAD_ALT" if model.startswith("qwen3.6") else "MOCK_BAD_FAST"
        return float(os.environ.get(name, "0") or 0)
    return float(os.environ.get("MOCK_BAD_STRONG", "0") or 0)


def _corrupt(entry, oracle: str, family: str, model: str, attempt: int) -> str:
    """Испорченный ответ: неверное значение, синтаксическая ошибка, ложный отказ или болтовня."""
    kind = int(_roll(entry, model, "kind") * 4)
    if family == "coding":
        return "IMPOSSIBLE" if kind == 0 else "```python\ndef helper_that_changes_nothing():\n    return 0\n```"
    if kind == 0:
        return f"({oracle}) + 1" if entry["info"]["check"] == "exact_int" else "'Atlantis'"
    if kind == 1:
        return oracle + " + (" if attempt == 0 else oracle   # со второй попытки исправляется
    if kind == 2:
        return "IMPOSSIBLE"
    return "Sure! The answer is probably 42, but let me explain my reasoning in detail."


def respond(body: dict) -> dict:
    if not _LOADED:
        _load()
    messages = body.get("messages") or []
    model = body.get("model", "")
    system = next((str(m.get("content", "")) for m in messages if m.get("role") == "system"), "")
    users = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
    first_user = users[0] if users else ""
    attempt = sum(1 for m in messages if m.get("role") == "assistant")

    if body.get("tools"):
        return _react(body, messages, first_user)

    if "fix bugs in Python code" in system:
        family = "coding"
    elif "`kb` is a dict" in system or "using only the passages" in system:
        family = "search"
    elif "`rows` is a list of dicts" in system:
        family = "data_analysis"
    elif "Translate the word problem" in system:
        family = "math"
    else:
        family = "generic"
    entry = _find(first_user, family)
    if entry is None:
        return {"content": "IMPOSSIBLE"}
    info = entry["info"]
    oracle = info.get("oracle") or "IMPOSSIBLE"
    if not info["feasible"]:
        hallucinate = float(os.environ.get("MOCK_HALLUCINATE", "0") or 0)
        if model.endswith("-noreason") and _roll(entry, model, "hall") < hallucinate:
            return {"content": "42" if family != "coding" else "```python\ndef nothing():\n    pass\n```"}
        return {"content": oracle}
    if _roll(entry, model, "bad") < _rate(model):
        return {"content": _corrupt(entry, oracle, family, model, attempt)}
    return {"content": oracle, "reasoning": "step by step ... " * 40 if not model.endswith("-noreason") else ""}


def _react(body: dict, messages: list, first_user: str) -> dict:
    entry = _find(first_user, "react")
    turns = sum(1 for m in messages if m.get("role") == "assistant")
    name = body["tools"][0]["function"]["name"]

    def call(command: str) -> dict:
        return {"content": "", "tool_calls": [{"id": f"call_{turns}", "type": "function",
                                               "function": {"name": name,
                                                            "arguments": json.dumps({"command": command})}}]}

    if turns == 0:
        return call("ls; cat mylib.py 2>/dev/null")
    seen = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "tool")
    by_source = _find(seen, "coding")   # инструкция у всех задач coding одинакова — различаем по исходнику
    if by_source is not None:
        entry = by_source
    if entry is None or not entry["info"]["feasible"]:
        return {"content": "IMPOSSIBLE"}
    info = entry["info"]
    if turns == 1:
        if info["type"] == "coding":
            parts = []
            for rel, text in (info.get("fixed") or {}).items():
                parts.append(f"cat > {rel} <<'PYEOF'\n{text}PYEOF")
            parts.append("python3 test_mylib.py 2>&1 | tail -3; echo DONE > answer.txt")
            return call("\n".join(parts))
        return call(f"echo {json.dumps(str(info['expected']))} > answer.txt")
    return {"content": "FINISHED"}
