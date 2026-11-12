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
    status: str                 # ok | unparsed | error | infeasible (данных для ответа нет)
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
    text = re.sub(r"^(?:solve the following problem[.:]|problem:|task:|question:|word problem:)\s*", "", text,
                  flags=re.IGNORECASE)
    sentences = _split_sentences(text)
    # Служебные фразы после вопроса («Save only the number to answer.txt.») к задаче не относятся.
    while len(sentences) > 2 and not sentences[-1].endswith("?") and _SERVICE_RE.search(sentences[-1]) \
            and not re.search(r"\d", sentences[-1].replace("answer.txt", "")) \
            and any(x.endswith("?") or _IMPERATIVE_Q.match(x) for x in sentences[:-1]):
        sentences.pop()
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
    END = r"(?: (?:at the end|in the end|now|left|finally|in total|at the end of the period|after (?:all )?(?:this|that)))?"
    PLACE = rf"(?: in (?:the {re.escape(subj)}|it|stock))?"
    q_ok = re.fullmatch(
        rf"how many {I} (?:(?:does|will|would|did) {S} (?:have|hold|end up with|finish with|own){END}"
        rf"|(?:are|remain|are there|will be|will remain|are left|is)(?: left)?{PLACE}{END}"
        rf"|remain{PLACE}{END})\?"
        rf"|what is (?:the|its) final (?:number|count|amount|quantity) of {I}{PLACE}\?"
        rf"|what is (?:the|its) (?:final|ending|resulting) (?:stock|inventory)(?: of (?:the )?(?:{I}|{re.escape(subj)}))?\?"
        rf"|what is the (?:number|count) of {I}{PLACE} at the end\?"
        rf"|(?:find|compute|calculate|determine|give|state|report) (?:the )?(?:final )?(?:number|count) of {I}{PLACE}{END}\.",
        q, re.IGNORECASE,
    )
    # «How many boxes did it ship in total?» — спрашивают не запас, а сумму по одному глаголу.
    q_total = re.fullmatch(
        rf"how many {I} (?:did|has) {S} (ship|shipped|sell|sold|receive|received|lose|lost)"
        rf"(?: in total| altogether| overall)?\?",
        q, re.IGNORECASE,
    )
    asked_verb = None
    other_item = None   # вопрос про предмет, которого нет в условии («How many trucks…»)
    if q_total:
        asked_verb = _VERB_OF[q_total.group(1).lower()]
    elif not q_ok:
        qo = re.fullmatch(rf"how many ([a-z]+) (?:does|did) {S} (?:have|own|keep|hold|use)"
                          rf"(?: (?:at the end|in the end|now|left))?\?", q, re.IGNORECASE)
        if qo and not re.fullmatch(I, qo.group(1), re.IGNORECASE):
            other_item = qo.group(1).lower()
        else:
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

    def sub_ac(m, v):
        a, b = _num(m.group(1)), _num(m.group(3))
        return (f"{{e}} - {a} * {b}", v - a * b)

    def add_ac(m, v):
        a, b = _num(m.group(1)), _num(m.group(3))
        return (f"{{e}} + {a} * {b}", v + a * b)

    def ceildiv_k(m, v):
        k = _num(m.group(1))
        if k == 0:
            raise ZeroDivisionError
        return (f"-(-{{e}} // {k})", -(-v // k))

    def sub_a(m, v):
        a = _num(m.group(1))
        return (f"{{e}} - {a}", v - a)

    def sell_percent(m, v):
        pct = _num(m.group(1))
        if not 0 <= pct <= 100:
            raise _Unparsed(f"странный процент {pct}")
        rounded = m.group(2) is not None
        if not rounded and (v * pct) % 100:
            raise _Unparsed(f"{pct}% от {v} не целое, а правило округления не указано")
        return (f"{{e}} - {{e}} * {pct} // 100", v - v * pct // 100)

    def add_a(m, v):
        a = _num(m.group(1))
        return (f"{{e}} + {a}", v + a)

    rd = r" \(rounded down\)"
    # необязательная причина, не меняющая арифметику: «in a flood», «after a merger»
    CAUSE = (r"(?: (?:in|after|during|due to|because of|from) (?:a|an|the) [a-z]+(?: [a-z]+)?"
             r"| by merging with (?:a|an|one|two|three|\d+) sister \w+)?")
    rules: list[tuple[str, str, Callable]] = [
        # обязательные (из примеров)
        ("ship_per_day", r"{S} (?:ships|sells|sends out|gives away|donates|uses|loses) {N} {I} (?:per|each|a) day for {N} days?", sub_ab),
        ("receive_deliveries", r"{S} receives {N} deliver(?:y|ies) of {N} {I}(?: each)?", add_ab),
        ("ship_per_unit", r"{S} (?:ships|sells|sends out|gives away|donates|uses|loses) {N} {I} (?:per|each|a|every) ([a-z]+) for {N} \2s?", sub_ac),
        ("receive_per_unit", r"{S} receives {N} {I} (?:per|each|a|every) ([a-z]+) for {N} \2s?", add_ac),
        ("split_keep_share_up", r"{ST} is split equally into {N} {W} and (?:it|the \w+) keeps one share \(rounded up\)", ceildiv_k),
        ("split_keep_share", r"{ST} is split equally into {N} {W} and (?:it|the \w+) keeps one share" + rd, floordiv_k(1, True)),
        ("double_merge", r"{ST} is doubled by merging with a sister \w+", mul(2)),
        # родственные однозначные формулировки
        ("split_keep_share_exact", r"{ST} is split equally into {N} {W} and (?:it|the \w+) keeps one share", floordiv_k(1, False)),
        ("receive_per_day", r"{S} receives {N} {I} (?:per|each|a) day for {N} days?", add_ab),
        ("receive_n", r"{S} (?:receives|gets|buys|gains) {N} (?:more |new |additional )?{I}", add_a),
        ("receive_delivery_of", r"{S} receives (?:a|one) delivery of {N} {I}", add_a),
        ("ship_n", r"{S} (?:ships|sells|loses|sends out|gives away|donates) {N} {I}" + CAUSE, sub_a),
        ("sell_percent", r"{S} (?:sells|ships|loses|sends out|gives away) {N} ?(?:percent|%) of (?:its|the) (?:{I}|stock|inventory)"
                         r"( \((?:the number (?:sold|shipped|lost|sent out|given away) is )?rounded down\))?", sell_percent),
        ("double", r"{ST} is doubled" + CAUSE, mul(2)),
        ("triple", r"{ST} is tripled" + CAUSE, mul(3)),
        ("quadruple", r"{ST} is quadrupled" + CAUSE, mul(4)),
        ("double_active", r"{S} doubles its (?:stock|inventory)", mul(2)),
        ("triple_active", r"{S} triples its (?:stock|inventory)", mul(3)),
        ("halve_rounded", r"{ST} is halved" + rd, lambda m, v: ("{e} // 2", v // 2)),
        ("halve_exact", r"{ST} is halved", halve_exact),
    ]
    LEAD = (r"(?:(?:and )?(?:then|next|after that|after this|afterwards|afterward|later|first|second|third|finally|lastly|"
            r"subsequently|at last|in the end),? )?")
    compiled = [(name, re.compile(LEAD + pat.format(**fmt) + r"\.", re.IGNORECASE), fn)
                for name, pat, fn in rules]

    totals: dict[str, int] = {}
    total_exprs: dict[str, list[str]] = {}
    went_negative = False
    for sent in sentences[1:-1]:
        for name, rx, fn in compiled:
            m = rx.fullmatch(sent)
            if not m:
                continue
            try:
                tmpl, new_value = fn(m, value)
            except ZeroDivisionError:
                return MathAnswer("infeasible", steps=steps, expression=expr, detail=f"деление на ноль: {sent!r}")
            except _Unparsed as exc:
                return MathAnswer("unparsed", steps=steps, expression=expr, detail=str(exc))
            if new_value < 0:
                went_negative = True
            wrapped = f"({expr})" if not expr.isdigit() else expr
            expr = tmpl.format(e=wrapped)
            verb = _sentence_verb(sent)
            if verb is not None:
                totals[verb] = totals.get(verb, 0) + abs(new_value - value)
                total_exprs.setdefault(verb, []).append(f"abs(({expr}) - ({wrapped}))")
            value = new_value
            steps.append(f"{name}: {sent} -> {value}")
            break
        else:
            # «It ships some boxes per day…»: та же операция, но вместо числа — неопределённое слово.
            vague = re.sub(_VAGUE, "7", sent, count=1, flags=re.IGNORECASE)
            if vague != sent and any(rx.fullmatch(vague) for _, rx, _ in compiled):
                return MathAnswer("infeasible", steps=steps, expression=expr,
                                  detail=f"в условии нет числа: {sent!r}")
            return MathAnswer("unparsed", steps=steps, expression=expr, detail=f"не понято предложение: {sent!r}")

    if other_item is not None:
        return MathAnswer("infeasible", steps=steps, expression=expr,
                          detail=f"спрашивают про «{other_item}», а в условии только {item}")

    # Самопроверка: выражение должно давать тот же ответ.
    if eval(expr, {"__builtins__": {}}, {}) != value:  # noqa: S307 — выражение собрано нами из чисел
        return MathAnswer("error", steps=steps, expression=expr, detail="выражение расходится со счётом")
    if asked_verb is not None:
        if asked_verb not in totals:
            return MathAnswer("unparsed", steps=steps, expression=expr,
                              detail=f"спрашивают про «{asked_verb}», а таких операций нет")
        value = totals[asked_verb]
        expr = " + ".join(total_exprs[asked_verb])
        if eval(expr, {"__builtins__": {"abs": abs}}, {}) != value:  # noqa: S307
            return MathAnswer("error", steps=steps, expression=expr, detail="сумма расходится со счётом")
    steps.append(f"answer: {value}")
    return MathAnswer("ok", answer=str(value), expression=expr, steps=steps)


_SERVICE_RE = re.compile(r"answer\.txt|\bbash\b|\bpython\b|\bcurrent directory\b|\bwrite\b|\bsave\b|\bfile\b|\btool\b"
                         r"|\bcalculate\b|\bcompute\b|\binteger\b|\bnumber\b", re.IGNORECASE)
_IMPERATIVE_Q = re.compile(r"(?:find|compute|calculate|determine|give|state|report)\b", re.IGNORECASE)
_VAGUE = r"\b(?:some|several|a few|many|a number of|an unknown number of|a certain number of)\b"

_VERB_OF = {"ship": "ship", "shipped": "ship", "sell": "sell", "sold": "sell",
            "receive": "receive", "received": "receive", "lose": "lose", "lost": "lose"}
_VERB_RE = [("ship", re.compile(r"\b(?:ships|sends out)\b", re.I)), ("sell", re.compile(r"\bsells\b", re.I)),
            ("receive", re.compile(r"\breceives\b", re.I)), ("lose", re.compile(r"\bloses\b", re.I))]


def _sentence_verb(sentence: str) -> Optional[str]:
    """Глагол операции, по которому её можно сложить в «сколько отгружено всего»."""
    found = [verb for verb, rx in _VERB_RE if rx.search(sentence)]
    return found[0] if len(found) == 1 else None
