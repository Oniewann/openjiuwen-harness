"""Текстовые задачи про запас на складе: разбор по предложениям и точный счёт.

Формат из примеров:

    A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses
    and it keeps one share (rounded down). ... How many boxes does it have at the end?

Первое предложение задаёт начальный запас, каждое следующее — одну операцию, последнее —
вопрос об итоговом запасе. Любое непонятое предложение → status="unparsed" (лучше отказ,
чем догадка). Только стандартная библиотека.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = ["MathAnswer", "answer_math_question"]


@dataclass
class MathAnswer:
    status: str                 # ok | unparsed | error
    answer: Optional[str] = None
    expression: str = ""        # python-выражение, дающее ответ
    steps: list[str] = field(default_factory=list)   # по строке на распознанное предложение
    detail: str = ""


# Хвост инструкции харнесса: всё, начиная с этих фраз, к задаче не относится.
_TAIL_RE = re.compile(
    r"\s*\b(?:compute it\b|use bash\b|use the bash\b|work only\b|when you are done\b|write only\b)",
    re.IGNORECASE,
)

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
# Число: 1752, 1,752 или слово one..twelve.
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+|" + "|".join(_WORD_NUMS) + r")"


def _num(tok: str) -> int:
    tok = tok.lower()
    if tok in _WORD_NUMS:
        return _WORD_NUMS[tok]
    return int(tok.replace(",", ""))


class _Unparsed(Exception):
    pass


def _split_sentences(text: str) -> list[str]:
    # Граница — «.», «?» или «!» перед пробелом/концом; «1,752» и «2.5» не режутся.
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _plural_re(word: str) -> str:
    w = re.escape(word)
    return rf"(?:{w}|{w}s|{w}es)"


def answer_math_question(question: str) -> MathAnswer:
    """Решает задачу про запас; при любом сомнении — status="unparsed"."""
    text = re.sub(r"\s+", " ", question or "").strip()
    m = _TAIL_RE.search(text)
    if m:
        text = text[: m.start()].strip()
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return MathAnswer("unparsed", detail="нужно хотя бы начальное условие и вопрос")

    # 1. Начальное значение: «A warehouse starts with N boxes.»
    first = re.fullmatch(
        r"(?:a|an|the) ([a-z]+(?: [a-z]+)?) (?:starts|begins) with " + _NUM + r" ([a-z]+)\.",
        sentences[0], re.IGNORECASE,
    )
    if not first:
        return MathAnswer("unparsed", detail=f"не понято начало: {sentences[0]!r}")
    subj = first.group(1).lower()
    value = _num(first.group(2))
    item = first.group(3).lower()
    expr = str(value)
    steps = [f"start: {value} {item}"]

    # 2. Вопрос: строго последнее предложение и строго про итоговый запас.
    q = sentences[-1]
    S = rf"(?:it|the {re.escape(subj)})"
    I = _plural_re(item)
    q_ok = re.fullmatch(
        rf"how many {I} (?:does {S} have(?: (?:at the end|in the end|now|left))?|"
        rf"(?:are|remain) left(?: at the end)?|remain(?: at the end)?)\?",
        q, re.IGNORECASE,
    )
    if not q_ok:
        return MathAnswer("unparsed", steps=steps, detail=f"не понят вопрос: {q!r}")
    if any("?" in s for s in sentences[:-1]):
        return MathAnswer("unparsed", steps=steps, detail="больше одного вопроса")

    W = _plural_re(subj.split()[-1])       # «warehouses» для разделения/слияния
    stock = rf"(?:the stock|the inventory|{S})"
    fmt = {"S": S, "I": I, "W": W, "N": _NUM, "ST": stock}

    # Правила: (имя, шаблон, функция(match, текущее) -> (новое выражение-шаблон, новое значение))
    # Шаблон выражения использует {e} как текущее выражение.
    def mul(k: int):
        return lambda m, v: (f"{{e}} * {k}", v * k)

    def floordiv_k(group: int, rounded: bool):
        def f(m, v):
            k = _num(m.group(group))
            if k == 0:
                raise ZeroDivisionError
            if not rounded and v % k:
                raise _Unparsed(f"{v} не делится на {k} нацело, а правило округления не указано")
            return (f"{{e}} // {k}", v // k)
        return f

    def halve_exact(m, v):
        if v % 2:
            raise _Unparsed(f"{v} нечётно, а правило округления не указано")
        return ("{e} // 2", v // 2)

    def sub_ab(m, v):
        a, b = _num(m.group(1)), _num(m.group(2))
        return (f"{{e}} - {a} * {b}", v - a * b)

    def add_ab(m, v):
        a, b = _num(m.group(1)), _num(m.group(2))
        return (f"{{e}} + {a} * {b}", v + a * b)

    def sub_a(m, v):
        a = _num(m.group(1))
        return (f"{{e}} - {a}", v - a)

    def add_a(m, v):
        a = _num(m.group(1))
        return (f"{{e}} + {a}", v + a)

    rd = r" \(rounded down\)"
    rules: list[tuple[str, str, Callable]] = [
        # обязательные (из примеров)
        ("ship_per_day", r"{S} (?:ships|sells|sends out) {N} {I} (?:per|each|a) day for {N} days?", sub_ab),
        ("receive_deliveries", r"{S} receives {N} deliver(?:y|ies) of {N} {I} each", add_ab),
        ("split_keep_share", r"{ST} is split equally into {N} {W} and (?:it|the \w+) keeps one share" + rd, floordiv_k(1, True)),
        ("double_merge", r"{ST} is doubled by merging with a sister \w+", mul(2)),
        # родственные однозначные формулировки
        ("split_keep_share_exact", r"{ST} is split equally into {N} {W} and (?:it|the \w+) keeps one share", floordiv_k(1, False)),
        ("receive_per_day", r"{S} receives {N} {I} (?:per|each|a) day for {N} days?", add_ab),
        ("receive_n", r"{S} (?:receives|gets|buys|gains) {N} (?:more |new |additional )?{I}", add_a),
        ("receive_delivery_of", r"{S} receives (?:a|one) delivery of {N} {I}", add_a),
        ("ship_n", r"{S} (?:ships|sells|loses|sends out|gives away|donates) {N} {I}", sub_a),
        ("double", r"{ST} is doubled", mul(2)),
        ("triple", r"{ST} is tripled", mul(3)),
        ("quadruple", r"{ST} is quadrupled", mul(4)),
        ("double_active", r"{S} doubles its (?:stock|inventory)", mul(2)),
        ("triple_active", r"{S} triples its (?:stock|inventory)", mul(3)),
        ("halve_rounded", r"{ST} is halved" + rd, lambda m, v: ("{e} // 2", v // 2)),
        ("halve_exact", r"{ST} is halved", halve_exact),
    ]
    compiled = [(name, re.compile(r"(?:then,? )?" + pat.format(**fmt) + r"\.", re.IGNORECASE), fn)
                for name, pat, fn in rules]

    for sent in sentences[1:-1]:
        for name, rx, fn in compiled:
            m = rx.fullmatch(sent)
            if not m:
                continue
            try:
                tmpl, new_value = fn(m, value)
            except ZeroDivisionError:
                return MathAnswer("error", steps=steps, expression=expr, detail=f"деление на ноль: {sent!r}")
            except _Unparsed as exc:
                return MathAnswer("unparsed", steps=steps, expression=expr, detail=str(exc))
            if new_value < 0:
                return MathAnswer("unparsed", steps=steps, expression=expr,
                                  detail=f"запас стал отрицательным после {sent!r}")
            expr = tmpl.format(e=f"({expr})" if not expr.isdigit() else expr)
            value = new_value
            steps.append(f"{name}: {sent} -> {value}")
            break
        else:
            return MathAnswer("unparsed", steps=steps, expression=expr, detail=f"не понято предложение: {sent!r}")

    # Самопроверка: выражение должно давать тот же ответ.
    if eval(expr, {"__builtins__": {}}, {}) != value:  # noqa: S307 — выражение собрано нами из чисел
        return MathAnswer("error", steps=steps, expression=expr, detail="выражение расходится со счётом")
    steps.append(f"answer: {value}")
    return MathAnswer("ok", answer=str(value), expression=expr, steps=steps)
