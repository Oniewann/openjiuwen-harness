"""Мелкие текстовые утилиты: разбор ответа модели, оценка числа токенов, нормализация ответа."""
from __future__ import annotations

import math
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-.]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", re.UNICODE)

REFUSAL_WORDS = ("IMPOSSIBLE", "NOT_FOUND", "INFEASIBLE", "CANNOT_ANSWER")


def strip_think(text: str) -> str:
    """Убирает блоки <think>…</think>, если сервер отдал рассуждения прямо в тексте ответа."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    low = text.lower()
    if "</think>" in low:  # осталась только закрывающая метка
        text = text[low.rfind("</think>") + len("</think>"):]
    elif "<think>" in low:  # рассуждение оборвано по лимиту, ответа нет
        text = text[: low.find("<think>")]
    return text.strip()


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Все блоки ```lang ...``` из ответа: список (язык, код)."""
    return [(lang.lower(), body.strip("\n")) for lang, body in _FENCE_RE.findall(text or "")]


def extract_code(text: str) -> str:
    """Код из ответа модели: первый блок в ограждении, иначе весь текст."""
    text = strip_think(text)
    blocks = extract_code_blocks(text)
    if blocks:
        for lang, body in blocks:
            if lang in ("python", "py", "python3", ""):
                return body.strip()
        return blocks[0][1].strip()
    # незакрытое ограждение (ответ оборван по max_tokens)
    if "```" in text:
        tail = text.split("```", 1)[1]
        tail = re.sub(r"^[A-Za-z0-9_+\-.]*[ \t]*\r?\n", "", tail)
        return tail.strip()
    return text.strip().strip("`").strip()


def is_refusal(text: str) -> bool:
    """Модель сказала, что задача невыполнима (IMPOSSIBLE / NOT_FOUND)."""
    cleaned = strip_think(text).strip().strip("`'\"*. ").upper().replace(" ", "_")
    if not cleaned:
        return False
    first = cleaned.splitlines()[0].strip("`'\"*. :")
    return any(first.startswith(word) for word in REFUSAL_WORDS)


def approx_tokens(text: str) -> int:
    """Грубая оценка числа токенов без токенизатора.

    Слово — токен на каждые ~5 букв, число — токен на каждые 3 цифры, знак — токен.
    Оценщик стоимости на ходу поправляет её коэффициентом по фактическому usage.
    """
    if not text:
        return 0
    total = 0
    for tok in _TOKEN_RE.findall(text):
        if tok[0].isdigit():
            total += (len(tok) + 2) // 3
        elif tok[0].isalpha() or tok[0] == "_":
            total += max(1, (len(tok) + 4) // 5)
        else:
            total += 1
    return total


def messages_tokens(messages: list[dict]) -> int:
    return sum(approx_tokens(str(m.get("content", ""))) + 4 for m in messages) + 3


def clip(text: str, head: int = 700, tail: int = 700) -> str:
    """Обрезает длинный вывод: начало + конец, середина заменяется пометкой."""
    if text is None:
        return ""
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n...[{len(text) - head - tail} chars omitted]...\n{text[-tail:]}"


def format_number(value: float) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e15:
        return str(int(round(value)))
    text = f"{round(value, 6):.6f}".rstrip("0").rstrip(".")
    return text


def normalize_answer(raw: str) -> str:
    """Приводит ответ к виду «только значение»: без кавычек, пояснений и точки в конце."""
    text = strip_think(raw or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = lines[-1] if lines else ""
    text = re.sub(r"^(final answer|answer|result)\s*[:=]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    if text.endswith(".") and not re.fullmatch(r"-?\d+\.", text) and len(text.split()) <= 6:
        text = text[:-1].strip()
    return text


def looks_integer(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", (text or "").strip()))
