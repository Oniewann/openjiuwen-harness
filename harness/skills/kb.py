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
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_FIELD_RE = re.compile(r"^\s*[-*+]?\s*([A-Za-z][\w \-/]{0,40}?)\s*(?::|=|\|)\s*(.+?)\s*\|?\s*$")
_EMPHASIS_RE = re.compile(r"\*\*|__|`")           # **Born:** 1983, `Born`: 1983
_TEXT_SUFFIXES = (".md", ".txt", ".markdown", ".json", ".yaml", ".yml", ".rst", ".text", "")
_TITLE_KEYS = ("name", "title", "full name", "full_name", "entity", "id")
# Значения, которые означают «данных нет»
_NO_VALUE = {"unknown", "n/a", "na", "?", "tbd", "none", "-", "--", "null", "not available", "not known", "missing",
             "unspecified", "undisclosed", "not specified", "n.a."}
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
    conflicts: list[str] = field(default_factory=list)                # повторы с другим содержимым
    # файл записи -> поле -> другие значения этого поля: (значение, из того же файла ли). «Born: 1980» и ниже
    # «Born: 1985» — в том же файле; второй файл той же записи с «Born: 1987» — из другого. Совместимы ли
    # значения со значением записи, решает _values_compatible при ответе.
    field_conflicts: dict[str, dict[str, list[tuple[str, bool]]]] = field(default_factory=dict)
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
        if "\x00" in text[:2000]:
            continue                      # двоичный файл
        records: list[tuple[Optional[str], dict[str, Any]]] = []
        if path.suffix.lower() == ".json":
            records = _json_records(text)
        if not records:
            title: Optional[str] = None
            fields: dict[str, Any] = {}
            seen: dict[str, list[str]] = {}
            headings = 0
            for line in text.splitlines():
                stripped = _EMPHASIS_RE.sub("", line).strip()
                if not stripped or stripped in ("---", "..."):
                    continue
                if stripped.startswith("#"):
                    headings += 1
                if title is None and stripped.startswith("#"):
                    title = stripped.lstrip("#").strip()
                    continue
                m = _FIELD_RE.match(stripped.strip("|").strip())
                if m:
                    fields.setdefault(m.group(1).strip(), _convert(m.group(2).strip().strip('"').strip("'")))
                    seen.setdefault(m.group(1).strip(), []).append(m.group(2).strip().strip('"').strip("'").lower())
            records = [(title, fields)]
            # Повтор поля внутри одной записи («Born: 1983» и ниже «Born: 1988»). Заголовков больше одного или
            # повторяется не одно поле — в файле, скорее, несколько записей: не судим.
            repeated = [k for k, vals in seen.items() if len(vals) > 1]
            if headings <= 1 and len(repeated) == 1:
                for k in repeated:
                    vals = seen[k]
                    for v in vals[1:]:
                        kb.field_conflicts.setdefault(str(path.relative_to(kb_dir)), {}).setdefault(k, []).append((v, True))
        for title, fields in records:
            if not title:
                # заголовка нет: имя записи берём из поля name/title, иначе из имени файла
                key = next((k for k in fields if k.lower() in _TITLE_KEYS and isinstance(fields[k], str)), None)
                if key is not None:
                    title = str(fields.pop(key)).strip()
            if not title:
                stem = re.sub(r"^(person|company|org|entity|doc)[_\-]", "", path.stem)
                title = stem.replace("_", " ").replace("-", " ").strip().title()
            if title in kb.entries:
                kb.duplicates.append(title)
                if kb.entries[title] != fields:
                    kb.conflicts.append(title)
                    # одно и то же поле в двух файлах записи с разными значениями — какое верное, неизвестно;
                    # поле, которое есть только в одном файле, противоречием не считаем
                    old = kb.entries[title]
                    for k in old:
                        if k in fields and str(old[k]).lower() != str(fields[k]).lower():
                            kb.field_conflicts.setdefault(kb.files[title], {}).setdefault(k, []).append((str(fields[k]), False))
                continue
            kb.entries[title] = fields
            kb.files[title] = str(path.relative_to(kb_dir))
            kb.texts[title] = text
    _relink_titles(kb)
    return kb


def _json_records(text: str) -> list[tuple[Optional[str], dict[str, Any]]]:
    """Записи из json-файла: объект — одна запись, список объектов — несколько."""
    import json

    try:
        data = json.loads(text)
    except ValueError:
        return []
    items = data if isinstance(data, list) else [data]
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fields = {str(k): (_convert(v) if isinstance(v, str) else v) for k, v in item.items()
                  if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        key = next((k for k in fields if k.lower() in _TITLE_KEYS and isinstance(fields[k], str)), None)
        title = str(fields.pop(key)).strip() if key is not None else None
        out.append((title, fields))
    return out


def _relink_titles(kb: KB) -> None:
    """Название, взятое из имени файла («Granite Ai»), приводим к написанию из ссылок («Granite AI»)."""
    values: dict[str, str] = {}
    for fields in kb.entries.values():
        for v in fields.values():
            if isinstance(v, str):
                values.setdefault(v.lower(), v)
    for title in list(kb.entries):
        better = values.get(title.lower())
        if better and better != title and better not in kb.entries:
            kb.entries[better] = kb.entries.pop(title)
            kb.files[better] = kb.files.pop(title)
            kb.texts[better] = kb.texts.pop(title)


def _values_compatible(kb: KB, a: str, b: str) -> bool:
    """Два значения одного поля записи не противоречат друг другу: «1983» и «March 4, 1983», «Dune Robotics»
    и «[[Dune Robotics]]»/«Dune Robotics Inc.», «1983» и «Moscow». Противоречие — только годы без общего
    или ссылки на разные записи, или разные строки, ни одна из которых не входит в другую."""
    na = re.sub(r"[^a-z0-9]+", " ", fold_text(a)).strip()
    nb = re.sub(r"[^a-z0-9]+", " ", fold_text(b)).strip()
    if not na or not nb or na == nb or f" {na} " in f" {nb} " or f" {nb} " in f" {na} ":
        return True
    ya, yb = set(re.findall(r"\b(1\d{3}|20\d{2})\b", na)), set(re.findall(r"\b(1\d{3}|20\d{2})\b", nb))
    if ya or yb:
        # год против места («Born: 1983» и «Born: Moscow») — разные стороны одного факта
        return bool(ya & yb) if ya and yb else True
    ma = set(find_mentions(kb, a.replace("[[", " ").replace("]]", " ")))
    mb = set(find_mentions(kb, b.replace("[[", " ").replace("]]", " ")))
    if ma or mb:
        return bool(ma & mb) if ma and mb else True
    return False


def fold_text(s: str) -> str:
    """Нижний регистр без диакритики: «José García» -> «jose garcia», «Ångström» -> «angstrom»."""
    s = s.replace("ß", "ss").replace("Ø", "O").replace("ø", "o").replace("Ł", "L").replace("ł", "l")
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()


def fold_name(s: str) -> set[str]:
    """Варианты имени для сравнения: без диакритики, апострофов, дефисов и пробелов; ü -> u и ü -> ue.
    «Seán O'Brien» и «Sean OBrien» -> «seanobrien»; «Lena Müller» -> {«lenamuller», «lenamueller»}."""
    out = set()
    for variant in (s, re.sub(r"[äöüÄÖÜ]", lambda m: {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe",
                                                         "Ü": "Ue"}[m.group(0)], s)):
        out.add(re.sub(r"[^a-z0-9]", "", fold_text(variant)))
    return {v for v in out if v}


def near_name(kb: KB, name: str, context: str = "") -> bool:
    """Имя похоже на запись базы, хотя и не совпало: другая транслитерация («Hsu Ming-yang» при «Xu Mingyang»),
    опечатка. Тогда «такой записи нет» не утверждаем уверенно. context — вопрос: из него имя дополняется
    отрезанными частями через дефис или апостроф («Hsu Ming» -> «Hsu Ming-yang»)."""
    from difflib import SequenceMatcher

    m = re.search(r"(?<![\w])" + re.escape(name) + r"(?:[\-'’][^\W\d_]+)*", context) if context else None
    if m:
        name = m.group(0)
    q = fold_name(name)
    q_tokens = [t for t in re.split(r"[^a-z0-9]+", fold_text(name)) if t]
    # признаки транслитерации: дефис, апостроф, диакритика
    translit = bool(re.search(r"[\-'’]", name)) or fold_text(name) != name.lower()

    def close(a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() >= 0.8 or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))

    # Уменьшительное имя («Bob Lee» при «Robert Lee», «Liz Ford» при «Elizabeth Ford»): фамилия совпала ровно
    # с одной записью. Компания с другим окончанием («Helix Motor Company» при «Helix Motors»).
    if len(q_tokens) >= 2:
        same_last = [t for t in kb.entries if (fold_text(t).split() or [""])[-1] == q_tokens[-1]]
        if len(same_last) == 1:
            return True
        # «Irene Smith» при «Irina Smith», «Alex Novak» при «Alexandra Novak»: та же фамилия, имя начинается так же
        if any(fold_text(t).split()[0][:2] == q_tokens[0][:2] for t in same_last):
            return True
        if q_tokens[-1] in _COMPANY_SUFFIXES:
            for title in kb.entries:
                t_tokens = [w for w in re.split(r"[^a-z0-9]+", fold_text(title)) if w]
                if all(any(close(a, b) for b in t_tokens) for a in q_tokens[:-1]):
                    return True
    for title in kb.entries:
        if any(SequenceMatcher(None, x, y).ratio() >= 0.85 for x in q for y in fold_name(title)):
            return True
        t_tokens = [w for w in re.split(r"[^a-z0-9]+", fold_text(title)) if w]
        if q_tokens and all(any(close(a, b) for b in t_tokens) for a in q_tokens):
            return True
        if translit and any(len(a) >= 3 and any(close(a, b) for b in t_tokens) for a in q_tokens):
            return True
    return False


_COMPANY_SUFFIXES = {"company", "co", "inc", "corp", "corporation", "ltd", "limited", "llc", "group", "plc", "gmbh"}


# слово имени: апостроф внутри («O'Brien») — да, притяжательное «'s» — нет
_FOLD_SPAN_RE = re.compile(r"[^\W\d_](?:[\w\-]|['’](?!s\b)\w)*")


def _fold_canonicalize(kb: KB, text: str) -> str:
    """«Jose Garcia» -> «José García», если так (без диакритики, апострофов, дефисов) называется ровно одна
    запись; при двух подходящих записях ничего не меняем."""
    folded: dict[str, set[str]] = {}
    for title in kb.entries:
        for v in fold_name(title):
            if len(v) >= 4:
                folded.setdefault(v, set()).add(title)
    words = list(_FOLD_SPAN_RE.finditer(text))
    out, shift, i = text, 0, 0
    while i < len(words):
        done = False
        for n in range(min(4, len(words) - i), 0, -1):
            first, last = words[i], words[i + n - 1]
            if not first.group(0)[0].isupper():
                break
            span = text[first.start():last.end()]
            if not re.fullmatch(r"[\w'’\-]+(?:\s+[\w'’\-]+)*", span):
                continue
            titles = set().union(*(folded.get(v, set()) for v in fold_name(span)))
            if len(titles) == 1 and span not in titles:
                title = titles.pop()
                out = out[: first.start() + shift] + title + out[last.end() + shift:]
                shift += len(title) - len(span)
                i += n
                done = True
                break
            if titles:
                i += n
                done = True
                break
        if not done:
            i += 1
    return out


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


def canonicalize_question(kb: KB, text: str) -> str:
    """Приводит имена записей в вопросе к написанию из базы: «irina smith», «Smith, Irina» -> «Irina Smith»,
    «Jose Garcia» -> «José García»."""
    out = _fold_canonicalize(kb, text)
    low = out.lower()
    for title in sorted(kb.entries, key=len, reverse=True):
        if len(title) < 4:
            continue
        parts = title.split()
        if len(parts) == 2:
            rev = f"{parts[1]}, {parts[0]}"
            if rev.lower() in low and title.lower() not in low:
                out = re.sub(r"(?<![\w])" + re.escape(rev) + r"(?![\w])", title, out, flags=re.IGNORECASE)
                low = out.lower()
        if title.lower() in low and title not in out:
            out = re.sub(r"(?<![\w])" + re.escape(title) + r"(?![\w])", title, out, flags=re.IGNORECASE)
            low = out.lower()
    return out


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
    if len(re.findall(r"\b(?:who|whom|whose|what|which|where|when)\b", q, re.IGNORECASE)) > 1:
        # «In which year was X born, and who is the mentor?» — два вопроса в одном: не цепочка
        return PathAnswer("unparsed", detail="several question words")
    extra = [s for s in sentences if s != q and not _extra_sentence_ok(s)]
    if extra:
        # «Answer with the last two digits only.» меняет ответ: всё, кроме вопроса, должно быть объяснено
        return PathAnswer("unparsed", detail=f"unexplained sentence {extra[0]!r}")

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
        record, record_title = kb.entries[current], current
        if name in record:
            for other, same_file in kb.field_conflicts.get(kb.files.get(current, ""), {}).get(name, ()):
                if _values_compatible(kb, str(record[name]), other):
                    continue
                # Навык не отвечает, но и невыполнимость не утверждает: внутри файла это бывает историей
                # («Employer: A», ниже «Employer: B»), между файлами — пометкой («Ember Bio (former)»). Решает модель.
                where = "twice" if same_file else "in two files"
                return PathAnswer("unparsed", chain=chain, anchor=anchor,
                                  detail=f"hop {hop}: entry {current!r} lists {name!r} {where}: {record[name]!r}, {other!r}")
        if name not in record and current in kb.duplicates:
            # у записи несколько файлов: поле может быть только во втором — решает модель
            return PathAnswer("unparsed", chain=chain, anchor=anchor,
                              detail=f"hop {hop}: entry {current!r} (several files) has no field {name!r} in the first")
        if name not in record:
            return PathAnswer("broken", chain=chain, anchor=anchor,
                              detail=f"hop {hop}: entry {current!r} has no field {name!r}")
        current = record[name]
        if isinstance(current, str) and current not in kb.entries:
            # «[[Dune Robotics]]» — ссылка на запись; «Dune Robotics Inc.», а второй файл записи пишет
            # «Dune Robotics» — берём совместимое значение, которое и есть запись
            alts = [current] + [o for o, _ in kb.field_conflicts.get(kb.files.get(record_title, ""), {}).get(name, ())]
            for value in alts:
                value = re.sub(r"^\[\[([^\]|]+)(?:\|[^\]]*)?\]\]$", r"\1", value.strip()).strip()
                if value in kb.entries and _values_compatible(kb, current, value):
                    current = value
                    break
    answer = str(current).strip()
    if answer.lower() in _NO_VALUE or not answer:
        return PathAnswer("broken", chain=chain, anchor=anchor, detail=f"value of {chain[-1]!r} is {answer!r}: no data")
    wants_year = re.search(r"\b(?:year|when)\b", q, re.IGNORECASE)
    if wants_year and not re.fullmatch(r"-?\d+", answer):
        # «In which year…», а в поле полная дата: 1983-04-12, 12.04.1983, April 12, 1983
        years = set(re.findall(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)", answer))
        if len(years) != 1:
            return PathAnswer("unparsed", chain=chain, anchor=anchor, detail=f"asked for a year, field is {answer!r}")
        answer = years.pop()
    # Тип ответа должен совпадать с вопросом: «Where was X born?» или «Who founded X?» — не про год из поля
    # Born/Founded (вопросительное слово тогда осталось без объяснения).
    if not wants_year and re.fullmatch(r"-?\d+", answer) and re.search(
            r"\b(?:who|whom|where|city|town|place|company|person|name|named)\b", q, re.IGNORECASE):
        return PathAnswer("unparsed", chain=chain, anchor=anchor, detail=f"asked for a name or place, got {answer!r}")
    # «Where is X's employer located?»: спрашивают место, а цепочка кончилась на записи (компании) —
    # слово «located/where» осталось без объяснения. «Where does X work?» — это и есть работодатель.
    if answer in kb.entries and re.search(r"\b(?:where|city|town|country|place|located|location|based|situated)\b",
                                          q, re.IGNORECASE) \
            and not re.match(r"^where (?:does|did|do) \S.* work(?: for| at)?\?$", q, re.IGNORECASE):
        return PathAnswer("unparsed", chain=chain, anchor=anchor, detail=f"asked for a place, got entry {answer!r}")
    return PathAnswer("ok", answer=answer, chain=chain, anchor=anchor)


# --------------------------------------------------------------------------- сравнение и обратные ссылки

_BORN_FIELDS = ("born", "birth year", "year of birth", "birth")
_COMPARE_RE = re.compile(
    r"^(?:who|which (?:person|one|of them|of the two))\s+(?:was|is)\s+"
    r"(?:(born) (earlier|first|later|last|before|after)|(older|younger))\s*[,:]?\s*(.+?)\s+or\s+(.+?)\s*\?$",
    re.IGNORECASE)
# «Who is mentored by X?», «Who is employed by X?», «Whose mentor is X?»
_INVERSE_RE = re.compile(
    r"^(?:who|which person)\s+(?:is|was)\s+(\w+?)(?:ed)?\s+by\s+(.+?)\s*\?$|^whose\s+(\w+)\s+(?:is|was)\s+(.+?)\s*\?$",
    re.IGNORECASE)


_KB_SCOPE = r"(?:\s+(?:in|of|from|listed in|recorded in)\s+the\s+(?:knowledge\s+base|kb|database|directory))?"
_EXTREME_RE = re.compile(
    r"^who\s+is\s+the\s+(oldest|eldest|youngest)(?:\s+(?:person|individual|employee|one))?" + _KB_SCOPE + r"\s*\?$"
    r"|^which\s+(?:company|organization|firm)(?:" + _KB_SCOPE + r")?\s+(?:is|was)\s+(?:the\s+)?(?:founded\s+)?"
    r"(oldest|newest|youngest|first|last|earliest|latest)" + _KB_SCOPE + r"\s*\?$"
    r"|^which\s+is\s+the\s+(oldest|newest|youngest)\s+(?:company|organization|firm)" + _KB_SCOPE + r"\s*\?$",
    re.IGNORECASE)


def _single_question(question: str) -> Optional[str]:
    text = question.strip()
    prefix = "The directory kb/ contains a small knowledge base of markdown files."
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    questions = [s for s in sentences if s.endswith("?")]
    if len(questions) != 1 or any(s != questions[0] and not _extra_sentence_ok(s) for s in sentences):
        return None
    return questions[0]


_INTRO_RE = re.compile(r"^the (?:directory|folder) \S+ contains (?:a |an )?(?:small )?knowledge base"
                       r"(?: of (?:markdown|text|md) files)?\.$", re.IGNORECASE)
# Слова служебных фраз про сдачу ответа («Write ONLY the answer to answer.txt.»).
_SERVICE_WORDS = {
    "write", "save", "put", "store", "output", "print", "give", "use", "only", "just", "the", "a", "an", "final",
    "answer", "result", "value", "as", "to", "in", "into", "file", "answer.txt", "current", "directory", "bash",
    "tool", "tools", "it", "with", "and", "then", "please", "your", "this", "that", "is", "be", "should", "must",
    "plain", "nothing", "else",
}


def _extra_sentence_ok(sentence: str) -> bool:
    """Предложение рядом с вопросом ничего не меняет: вводная про базу или фраза про сдачу ответа."""
    if _INTRO_RE.match(sentence.strip()):
        return True
    words = re.findall(r"[a-z0-9_.]+", sentence.lower().replace("answer.txt", " answer.txt "))
    return all(w == "answer.txt" or w.strip(".") in _SERVICE_WORDS for w in words if w.strip("."))


def answer_relation_question(kb: KB, question: str) -> PathAnswer:
    """Сравнение двух записей по году рождения и обратная ссылка («кого наставляет X»).

    Строго: ровно две известные записи с целым годом и разными значениями; для обратной
    ссылки — ровно одна запись, у которой поле указывает на названную запись.
    """
    if not kb.structured:
        return PathAnswer("unparsed", detail="kb is not structured")
    q = _single_question(question)
    if q is None:
        return PathAnswer("unparsed", detail="expected exactly one question sentence")

    m = _COMPARE_RE.match(q)
    if m:
        names = [m.group(4).strip(), m.group(5).strip()]
        missing = [n for n in names if n not in kb.entries]
        if missing:
            # «…, Anna Petrov or her mentor?» — не имя, а описание: это не «записи нет», а отказ разбора
            if unknown_names(kb, missing[0]) and not find_mentions(kb, missing[0]):
                return PathAnswer("no_anchor", detail=f"no entry named {missing[0]!r}")
            return PathAnswer("unparsed", detail=f"not an exact entry name: {missing[0]!r}")
        if names[0] == names[1]:
            return PathAnswer("unparsed", detail="the same entry twice")
        born = []
        for n in names:
            key = next((k for k in kb.entries[n] if k.lower() in _BORN_FIELDS), None)
            if key is None:
                return PathAnswer("broken", anchor=n, detail=f"entry {n!r} has no birth year")
            v = kb.entries[n][key]
            if not isinstance(v, int):
                return PathAnswer("unparsed", anchor=n, detail=f"birth year of {n!r} is not a number: {v!r}")
            born.append(v)
        if born[0] == born[1]:
            return PathAnswer("broken", detail=f"both born in {born[0]}: no single answer")
        word = (m.group(2) or m.group(3)).lower()
        want_earlier = word in ("earlier", "first", "before", "older")
        pick = names[0] if (born[0] < born[1]) == want_earlier else names[1]
        return PathAnswer("ok", answer=pick, chain=["compare:" + word], anchor=" vs ".join(names))

    m = _EXTREME_RE.match(q)
    if m:
        # «Who is the youngest person in the knowledge base?» — крайнее значение года по всем записям с этим полем.
        # Считает код: модель без рассуждений путает «младший» (больший год рождения) со «старшим».
        word = (m.group(1) or m.group(2) or m.group(3)).lower()
        is_company = bool(m.group(2) or m.group(3))
        names = ("founded", "established", "founding year") if is_company else _BORN_FIELDS
        pairs = []
        for title, fields in kb.entries.items():
            key = next((k for k in fields if k.lower() in names), None)
            if key is not None and isinstance(fields[key], int):
                pairs.append((fields[key], title))
        if len(pairs) < 2:
            return PathAnswer("unparsed", detail="not enough entries with a year")
        earliest = word in ("oldest", "eldest", "first", "earliest")
        best = min(pairs)[0] if earliest else max(pairs)[0]
        winners = [t for y, t in pairs if y == best]
        if len(winners) != 1:
            return PathAnswer("unparsed", detail=f"{len(winners)} entries share the year {best}")
        return PathAnswer("ok", answer=winners[0], chain=["extreme:" + word], anchor="*")

    m = _INVERSE_RE.match(q)
    if m:
        rel = (m.group(1) or m.group(3) or "").lower()
        target = (m.group(2) or m.group(4) or "").strip()
        fields = _field_lookup(kb)
        stems = {rel, rel + "e", rel.rstrip("e"), rel + "or", rel + "er"}
        if rel == "employ":
            stems.add("employer")
        field_name = next((fields[s] for s in stems if s in fields), None)
        if field_name is None:
            return PathAnswer("unparsed", detail=f"no field for relation {rel!r}")
        if target not in kb.entries:
            if unknown_names(kb, target):
                return PathAnswer("no_anchor", detail=f"no entry named {target!r}")
            return PathAnswer("unparsed", detail=f"not an exact entry name: {target!r}")
        hits = [t for t, f in kb.entries.items() if f.get(field_name) == target]
        if len(hits) == 1:
            return PathAnswer("ok", answer=hits[0], chain=[f"inverse:{field_name}"], anchor=target)
        if not hits:
            return PathAnswer("broken", anchor=target, detail=f"nobody has {field_name} = {target!r}")
        return PathAnswer("unparsed", anchor=target, detail=f"{len(hits)} entries have {field_name} = {target!r}")
    return PathAnswer("unparsed", detail="not a comparison or inverse question")


# --------------------------------------------------------------------------- невыполнимые без модели
# Тот же приём, что в mathword: вопрос не разобран, но после одной подстановки (неизвестное имя -> имя
# записи) разбирается целиком. Плюс проверка по всему тексту базы: если ни одного слова из имени в базе
# нет вообще (и имя не похоже ни на одну запись), база ответить не может — как ни переформулируй вопрос.
_TITLE_ABBR_RE = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St)\.\s")
_NAME_RUN_RE = re.compile(r"(?<![\w'’\-])([A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)*)")


def _kb_words(kb: KB) -> Optional[tuple[set[str], str]]:
    """Все слова всех файлов каталога базы (нижний регистр) и имена файлов, плюс весь текст одной строкой
    без пробелов и знаков («Nova Tech» -> «novatech»). None — есть нечитаемый файл."""
    words: set[str] = set()
    squashed: list[str] = []
    root = kb.root
    if root is None or not Path(root).is_dir():
        return None
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            return None          # pdf, docx, картинка: что в них — не знаем
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        folded = fold_text(text + " " + path.name)
        words.update(re.findall(r"[a-z0-9]+", folded))
        squashed.append(re.sub(r"[^a-z0-9]", "", folded))
    return words, "|".join(squashed)


def _question_runs(q: str) -> list[str]:
    """Сочетания слов с заглавной буквы (кандидаты в имена), без служебных слов в начале."""
    runs = []
    for m in _NAME_RUN_RE.finditer(q):
        parts = m.group(1).split()
        while parts and parts[0].lower() in _FILLER:
            parts = parts[1:]
        if parts:
            runs.append(" ".join(parts))
    return runs


def infeasible_probe(kb: KB, question: str) -> Optional[PathAnswer]:
    """Высокоуверенные невыполнимые вопросы, которые answer_path_question не разобрал. None — не утверждаем."""
    if not kb.structured or not kb.entries:
        return None
    q = _single_question(_TITLE_ABBR_RE.sub(lambda m: m.group(1) + " ", question))
    if q is None:
        return None
    scanned = _kb_words(kb)
    if scanned is None:
        return None
    words, squashed = scanned
    anchors = find_mentions(kb, q)
    if not anchors:
        runs = _question_runs(q)
        if len(runs) != 1:
            return None
        name = runs[0]
        tokens = [t for t in re.findall(r"[a-z0-9]+", fold_text(name))
                  if len(t) >= 2 and t.lower() not in ("dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st")]
        if not tokens:
            return None
        # 1. Имени нет в базе ни целиком, ни по частям, а вопрос с именем записи на его месте разбирается.
        # и даже как часть слова («NovaTech» при «Nova Tech», «Liam» внутри «William») — иначе не утверждаем
        if not any(t in words or (len(t) >= 3 and t in squashed) for t in tokens) and not near_name(kb, name):
            for title in list(kb.entries)[:60]:
                res = answer_path_question(kb, q.replace(name, title))
                if res.ok:
                    return PathAnswer("no_anchor", detail=f"no entry named {name!r} (no part of it in the knowledge base)")
            return None
        # 2. Только имя, и так зовут нескольких: ответы для них разные — однозначного ответа нет.
        if len(tokens) == 1 and len(name.split()) == 1:
            cands = [t for t in kb.entries if name in t.split()]
            if len(cands) < 2:
                return None
            answers = set()
            for title in cands:
                res = answer_path_question(kb, re.sub(r"(?<![\w])" + re.escape(name) + r"(?![\w])", title, q))
                if not res.ok:
                    return None
                answers.add(res.answer)
            if len(answers) >= 2:
                return PathAnswer("broken", detail=f"{name!r} matches {len(cands)} entries with different answers")
        return None
    # 3. Признак, которого нет ни в одной записи («What is X's salary?»), не утверждаем: его называют синонимами
    # («Remuneration:», «Tel:», «Headquarters:») или пишут в прозе (адрес почты в тексте) — решает модель.
    return None
