"""База знаний из markdown-файлов: загрузка, схема для модели и разбор вопросов-цепочек.

Формат файла из примеров:

    # Irina Smith

    Born: 1983
    Employer: Dune Robotics
    Mentor: Alice Smith

Загрузчик не привязан к именам полей: любые строки «Ключ: значение» становятся полями,
заголовок «# …» — названием записи. Значение, совпадающее с названием другой записи,
считается ссылкой (по ним идут многошаговые вопросы: человек → работодатель → город).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_FIELD_RE = re.compile(r"^\s*[-*]?\s*\**([A-Za-z][\w \-/]{0,40}?)\**\s*:\s*(.+?)\s*$")
_TEXT_SUFFIXES = (".md", ".txt", ".markdown")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|'s|’s|[^\sA-Za-z0-9_]")

# Слова, которые в вопросе-цепочке ничего не значат (вопросительные, служебные, тип ответа).
_FILLER = {
    "in", "which", "what", "who", "whom", "where", "when", "is", "was", "are", "were", "the", "of", "a", "an",
    "does", "did", "do", "name", "named", "year", "city", "town", "place", "company", "person", "s", "has", "have",
    "for", "to", "at", "by", "that", "and", "it", "be", "been", "located", "based", "called", "value", "tell", "me",
    "find", "give", "please", "according", "knowledge", "base", "kb",
}
# Слова, означающие, что вопрос не про путь по ссылкам (подсчёт, сравнение, отрицание).
_NON_PATH = {
    "many", "count", "number", "oldest", "youngest", "earliest", "latest", "most", "fewest", "least", "more", "less",
    "older", "younger", "before", "after", "between", "both", "all", "list", "every", "each", "average", "sum",
    "total", "same", "share", "shares", "not", "never", "except", "or", "than", "first", "last", "any", "other",
    "together", "common", "how",
}


@dataclass
class KB:
    root: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)   # название -> поля
    files: dict[str, str] = field(default_factory=dict)                # название -> файл
    texts: dict[str, str] = field(default_factory=dict)                # название -> исходный текст
    duplicates: list[str] = field(default_factory=list)
    total_chars: int = 0

    @property
    def structured(self) -> bool:
        """Почти все записи состоят из полей «Ключ: значение»."""
        if not self.entries:
            return False
        with_fields = sum(1 for fields in self.entries.values() if fields)
        return with_fields >= 0.8 * len(self.entries)

    def field_names(self) -> list[str]:
        names: dict[str, None] = {}
        for fields in self.entries.values():
            for key in fields:
                names.setdefault(key)
        return list(names)

    def groups(self) -> list[tuple[tuple[str, ...], list[str]]]:
        """Записи, сгруппированные по набору полей (в примерах это «люди» и «компании»)."""
        by_sig: dict[tuple[str, ...], list[str]] = {}
        for title, fields in self.entries.items():
            by_sig.setdefault(tuple(fields.keys()), []).append(title)
        return sorted(by_sig.items(), key=lambda kv: -len(kv[1]))


def _convert(value: str) -> Any:
    """Строку, которая без потерь читается как целое, храним числом (для сравнений и подсчётов)."""
    value = value.strip()
    if re.fullmatch(r"-?\d{1,15}", value) and str(int(value)) == value:
        return int(value)
    return value


def find_kb_dir(workspace: Path) -> Optional[Path]:
    """Каталог базы знаний: kb/, иначе подкаталог (или сам workspace) с наибольшим числом md/txt."""
    if not workspace.is_dir():
        return None
    preferred = workspace / "kb"
    if preferred.is_dir():
        return preferred
    best: tuple[int, Optional[Path]] = (0, None)
    for cand in [workspace] + [p for p in sorted(workspace.iterdir()) if p.is_dir() and not p.name.startswith(".")]:
        try:
            n = sum(1 for p in cand.iterdir() if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES)
        except OSError:
            continue
        if n > best[0]:
            best = (n, cand)
    return best[1] if best[0] >= 2 else None


def load_kb(kb_dir: Path, max_files: int = 5000) -> KB:
    kb = KB(root=kb_dir)
    paths = sorted(p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES
                   and not p.name.startswith("."))[:max_files]
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        kb.total_chars += len(text)
        title: Optional[str] = None
        fields: dict[str, Any] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if title is None and stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                continue
            m = _FIELD_RE.match(stripped)
            if m:
                fields.setdefault(m.group(1).strip(), _convert(m.group(2)))
        if not title:
            stem = re.sub(r"^(person|company|org|entity|doc)[_\-]", "", path.stem)
            title = stem.replace("_", " ").replace("-", " ").strip().title()
        if title in kb.entries:
            kb.duplicates.append(title)
            continue
        kb.entries[title] = fields
        kb.files[title] = str(path.relative_to(kb_dir))
        kb.texts[title] = text
    return kb


def find_mentions(kb: KB, text: str) -> list[str]:
    """Названия записей, встречающиеся в тексте (самые длинные совпадения, без вложенных)."""
    found: list[tuple[int, int, str]] = []
    for title in kb.entries:
        if len(title) < 2:
            continue
        for m in re.finditer(r"(?<![\w])" + re.escape(title) + r"(?![\w])", text):
            found.append((m.start(), m.end(), title))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    result: list[tuple[int, int, str]] = []
    for start, end, title in found:
        if any(start >= s and end <= e for s, e, _ in result):
            continue
        result.append((start, end, title))
    seen: dict[str, None] = {}
    for _, _, title in result:
        seen.setdefault(title)
    return list(seen)


def unknown_names(kb: KB, text: str) -> list[str]:
    """Похожие на имена собственные сочетания в вопросе, которых нет среди записей базы."""
    names = []
    for m in re.finditer(r"(?<![\w])([A-Z][a-z]+(?: [A-Z][A-Za-z]+)+)(?![\w])", text):
        name = m.group(1)
        # начало предложения («In Which ...») отсекаем: первое слово — обычное служебное
        if name.split()[0].lower() in _FILLER:
            name = " ".join(name.split()[1:])
            if len(name.split()) < 2:
                continue
        if name in kb.entries or any(name in t or t in name for t in kb.entries):
            continue
        # Имя может быть не записью, а значением поля (город, должность): тогда оно известно базе.
        low = name.lower()
        if any(isinstance(v, str) and low in v.lower() for fields in kb.entries.values() for v in fields.values()):
            continue
        names.append(name)
    return names


def schema_summary(kb: KB, anchors: list[str], max_groups: int = 6) -> str:
    """Короткое описание базы для модели: по одному примеру записи на каждый набор полей."""
    lines = []
    shown = set()
    for sig, titles in kb.groups()[:max_groups]:
        example = next((t for t in anchors if t in titles), titles[0])
        shown.add(example)
        lines.append(f"- {len(titles)} entries with fields {list(sig)}, e.g. kb[{example!r}] = {kb.entries[example]!r}")
    for title in anchors:
        if title not in shown:
            lines.append(f"- kb[{title!r}] = {kb.entries[title]!r}")
    return "\n".join(lines)


def kb_preload_code(kb_dir: Path) -> str:
    """Код, который в песочнице заново строит словарь kb (тот же разбор, что и здесь)."""
    return (
        "import sys as _sys\n"
        f"_sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
        "from harness.skills.kb import load_kb as _load_kb\n"
        "from pathlib import Path as _Path\n"
        f"_kb = _load_kb(_Path({str(kb_dir)!r}))\n"
        "kb = {t: dict(f) for t, f in _kb.entries.items()}\n"
        "texts = dict(_kb.texts)\n"
        "_sys.path.pop(0)\n"
    )


# --------------------------------------------------------------------------- разбор вопроса-цепочки
@dataclass
class PathAnswer:
    status: str                   # ok | no_anchor | broken | unparsed
    answer: Optional[str] = None
    chain: list[str] = field(default_factory=list)
    anchor: Optional[str] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _field_lookup(kb: KB) -> dict[str, str]:
    return {name.lower(): name for name in kb.field_names()}


def _related_but_different(word: str, fields: dict[str, str]) -> bool:
    """Слово похоже на имя поля, но не равно ему («mentors», «employs»): возможно обратное отношение."""
    w = word.lower()
    for low in fields:
        if w == low:
            return False
    for low in fields:
        stem = low[: max(4, len(low) - 3)]
        if len(w) >= 4 and (w.startswith(stem) or low.startswith(w[: max(4, len(w) - 3)])):
            return True
    return False


def answer_path_question(kb: KB, question: str) -> PathAnswer:
    """Отвечает на вопрос вида «поле(я) одной записи по цепочке ссылок».

    Разбор строгий: если в вопросе есть хоть одно слово, которое не объяснено
    (не служебное, не имя поля, не название записи), навык отказывается — status="unparsed".
    """
    if not kb.structured:
        return PathAnswer("unparsed", detail="kb is not structured")
    text = question.strip()
    for prefix in ("The directory kb/ contains a small knowledge base of markdown files.",):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # берём последнее предложение-вопрос
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    questions = [s for s in sentences if s.endswith("?")]
    if len(questions) != 1:
        return PathAnswer("unparsed", detail="expected exactly one question sentence")
    q = questions[0]

    anchors = find_mentions(kb, q)
    if not anchors:
        missing = unknown_names(kb, q)
        if missing:
            return PathAnswer("no_anchor", detail=f"no entry named {missing[0]!r}")
        return PathAnswer("unparsed", detail="no entry mentioned")
    if len(anchors) != 1:
        return PathAnswer("unparsed", detail="several entries mentioned")
    anchor = anchors[0]
    if unknown_names(kb, q.replace(anchor, " ")):
        return PathAnswer("unparsed", detail="unknown proper name in question")

    pos = q.find(anchor)
    left, right = q[:pos], q[pos + len(anchor):]
    fields = _field_lookup(kb)

    def tokens(s: str) -> list[str]:
        return [t for t in _WORD_RE.findall(s)]

    # правая часть: цепочка притяжательных «'s поле 's поле», затем сказуемые
    r_tokens = tokens(right)
    poss_chain: list[str] = []
    i = 0
    while i + 1 < len(r_tokens) and r_tokens[i] in ("'s", "’s") and r_tokens[i + 1].lower() in fields:
        poss_chain.append(fields[r_tokens[i + 1].lower()])
        i += 2
    tail_chain: list[str] = []
    for tok in r_tokens[i:]:
        low = tok.lower()
        if low in fields:
            tail_chain.append(fields[low])
        elif not tok[0].isalnum() or low in _FILLER:
            continue
        elif low in _NON_PATH or _related_but_different(tok, fields):
            return PathAnswer("unparsed", anchor=anchor, detail=f"word {tok!r} is not a plain path")
        else:
            return PathAnswer("unparsed", anchor=anchor, detail=f"unexplained word {tok!r}")

    # левая часть: «поле of поле of …» читается справа налево
    left_chain: list[str] = []
    for tok in tokens(left):
        low = tok.lower()
        if low in fields:
            left_chain.append(fields[low])
        elif not tok[0].isalnum() or low in _FILLER:
            continue
        elif low in _NON_PATH or _related_but_different(tok, fields):
            return PathAnswer("unparsed", anchor=anchor, detail=f"word {tok!r} is not a plain path")
        else:
            return PathAnswer("unparsed", anchor=anchor, detail=f"unexplained word {tok!r}")
    left_chain.reverse()

    chain = poss_chain + left_chain + tail_chain
    if not chain:
        return PathAnswer("unparsed", anchor=anchor, detail="no field mentioned")

    current: Any = anchor
    for hop, name in enumerate(chain):
        if not isinstance(current, str) or current not in kb.entries:
            return PathAnswer("broken", chain=chain, anchor=anchor,
                              detail=f"hop {hop}: {current!r} has no entry in the knowledge base")
        record = kb.entries[current]
        if name not in record:
            return PathAnswer("broken", chain=chain, anchor=anchor,
                              detail=f"hop {hop}: entry {current!r} has no field {name!r}")
        current = record[name]
    return PathAnswer("ok", answer=str(current).strip(), chain=chain, anchor=anchor)
