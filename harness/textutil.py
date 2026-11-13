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
    for _ in range(3):  # точка в конце и кавычки могут идти в любом порядке: «'Osaka'.», «"Osaka."»
        before = text
        text = text.strip().strip("`").strip()
        if text.endswith(".") and not re.fullmatch(r"-?\d+\.", text) and len(text.split()) <= 6:
            text = text[:-1].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1].strip()
        if text == before:
            break
    return canonical_number(text)


_PLAIN_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def canonical_number(text: str) -> str:
    """Число в записи без контекста вопроса — так, как его напечатал бы print():
    «.5» -> «0.5», «+5» -> «5», «12.» и «12.0» -> «12», «-0» -> «0».

    Целое с нулями после точки («46.00») тоже становится целым: так модель и наивный агент пишут
    целые суммы. Нули после значащих цифр («0.50») не трогаем — без вопроса не видно, просили ли
    ровно два знака (это решает fit_number_to_question)."""
    if not _PLAIN_NUM_RE.fullmatch(text or ""):
        return text
    sign = "-" if text.startswith("-") else ""
    whole, _, frac = text.lstrip("+-").partition(".")
    whole = whole or "0"
    if frac.strip("0") == "":
        frac = ""
    if not frac and whole.strip("0") == "":
        sign = ""
    return sign + whole + ("." + frac if frac else "")


_PLACES_RE = re.compile(r"\b(?:to|with)\s+(?:exactly\s+)?(\d+|one|two|three|four|five|six)\s+decimal\s+(?:places?|digits?)\b",
                        re.IGNORECASE)
# Явно «ровно N знаков» («exactly 2 decimal places», «formatted to 2 decimal places»): нули в конце
# дописываются (25.40); просто «rounded to N» — как print(round(x, N)), без нулей.
_EXACT_PLACES_RE = re.compile(r"\bexactly\s+(?:\d+|one|two|three|four|five|six)\s+decimal|"
                              r"\bformat(?:ted)?\b[^.?!]{0,40}\bdecimal", re.IGNORECASE)
_PLACES_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
# Вопрос сам говорит о виде числа — нули в дробной части не трогаем.
_FORMAT_WORDS_RE = re.compile(r"round|decimal|cent|digit|place|precision|significant|format|percent|%|fraction",
                              re.IGNORECASE)
# Ответ — счёт или год: «rounded to 2 decimal places» в таком вопросе относится к порогу, не к ответу.
_COUNT_Q_RE = re.compile(r"\s*(?:how many|in (?:which|what) year|what year|how old)\b|.*\bhow many decimal",
                         re.IGNORECASE | re.DOTALL)
# Счёт где угодно в вопросе («Count the rows…», «What is the number of…», «… How many…»): целое значение
# дополнять нулями нельзя, знаки после точки относятся к порогу или к данным.
_COUNT_ANY_RE = re.compile(r"\b(?:how many|number of|count)\b", re.IGNORECASE)
_SCI_NUM_RE = re.compile(r"-?(\d+)(?:\.(\d+))?(?:[eE][+-]?\d+)?")


def fit_number_to_question(value: str, question: str, computed: bool = True) -> str:
    """Числовой ответ в том виде, в каком его, вероятнее всего, записал автор задачи.

    «rounded to N decimal places» — как print(round(x, N)): не больше N знаков, нули в конце не
    дописываются: 23.456 -> 23.46 (модель забыла округлить), 23.40 -> 23.4, 46.00 -> 46 (целое —
    целым, как в эталонах exact_int). Без указаний о виде числа — как print():
    0.50 -> 0.5; целые остаются целыми (12.0 -> 12, как в эталонах exact_int). Нечисловые ответы
    и всё спорное возвращаются как есть.

    computed=False — значение переписано из данных (search): эталон — запись из базы как есть,
    print-вид к нему не применяется."""
    m = _SCI_NUM_RE.fullmatch(value or "")
    if not m:
        return value
    places = {int(n) if n.isdigit() else _PLACES_WORDS[n.lower()] for n in _PLACES_RE.findall(question or "")}
    if places:
        if len(places) != 1 or _COUNT_Q_RE.match(question):
            return value
        if not m.group(2) and "e" not in value.lower() and _COUNT_ANY_RE.search(question):
            return value    # целый счёт: «Count the rows where price, rounded to 2 decimal places, …» -> 7
        n = places.pop()
        if not 1 <= n <= 10:
            return value
        frac = m.group(2) or ""
        if "e" not in value.lower() and len(frac) <= n:
            if not computed:
                return value    # search: значение переписано из базы — как есть
            out = value     # знаков не больше N — не трогаем, только срезаем нули в конце (ниже)
        else:
            out = repr(round(float(value), n))
            if "e" in out.lower():
                out = f"{round(float(value), n):.{n}f}"
        if _EXACT_PLACES_RE.search(question):
            out = f"{round(float(out), n):.{n}f}"
            return out.lstrip("-") if float(out) == 0 else out
        # как print(round(x, N)): нули в конце не дописываются (25.4, а не 25.40); целое — целым
        if "." in out:
            out = out.rstrip("0")
        return canonical_number(out)
    if not computed or _FORMAT_WORDS_RE.search(question or "") or "e" in value.lower():
        return value
    if m.group(2) and value.endswith("0"):
        value = canonical_number(value.rstrip("0"))
    # 0.00001 оставляем как есть, хотя print() дал бы 1e-05: так же пишет навык таблиц (голоса
    # должны совпадать), а «1e-05» на «how much» считается странным ответом и зовёт перепроверку.
    return value


def looks_integer(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", (text or "").strip()))
