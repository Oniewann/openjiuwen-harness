"""Таблицы csv/tsv: загрузка, краткая сводка для модели и строгий разбор вопросов-агрегатов.

Пример вопроса: «How many units of product 'widget' were sold in months 7 through 8 inclusive?»
→ sum(units) where product == 'widget' and 7 <= month <= 8.

Разбор строгий: каждое значимое слово вопроса должно быть объяснено (мера, фильтр, служебное
слово). Иначе status="unparsed" — задачу решит модель. Ссылка на отсутствующий столбец →
status="missing_column" (признак невыполнимой задачи). Только стандартная библиотека.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = ["Table", "TableAnswer", "find_tables", "load_table", "table_summary", "answer_table_question"]

_DISTINCT_LIMIT = 20
_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass
class Table:
    path: Path
    name: str                   # имя файла, например "sales.csv"
    columns: list[str]
    rows: list[dict]            # значения приведены к int/float, если весь столбец числовой
    types: dict[str, str]       # столбец -> "int" | "float" | "str"
    distinct: dict[str, list] = field(default_factory=dict)  # строковые столбцы с <= 20 значениями


@dataclass
class TableAnswer:
    status: str                 # ok | unparsed | missing_column | empty
    answer: Optional[str] = None
    plan: str = ""              # читаемый план вычисления
    detail: str = ""            # почему отказ / пометки


# ---------------------------------------------------------------- загрузка

def find_tables(workspace: Path) -> list[Path]:
    """Все *.csv и *.tsv внутри workspace (рекурсивно, без скрытых файлов и папок)."""
    workspace = Path(workspace)
    found = []
    for p in workspace.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".tsv"):
            continue
        rel = p.relative_to(workspace)
        if any(part.startswith(".") for part in rel.parts):
            continue
        found.append(p)
    return sorted(found)


def _detect_delimiter(sample: str, suffix: str) -> str:
    first_line = sample.splitlines()[0] if sample else ""
    try:
        d = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        if d in first_line:
            return d
    except csv.Error:
        pass
    if suffix == ".tsv" and "\t" in first_line:
        return "\t"
    counts = {d: first_line.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ("\t" if suffix == ".tsv" else ",")


def _column_type(values: list[str]) -> str:
    present = [v for v in values if v != ""]
    if not present:
        return "str"
    if all(_INT_RE.fullmatch(v) and str(int(v)) == v for v in present):
        return "int"
    # «007» и подобные — скорее коды, чем числа: оставляем строкой.
    if all(_FLOAT_RE.fullmatch(v) for v in present) and not any(
        _INT_RE.fullmatch(v) and str(int(v)) != v for v in present
    ):
        return "float"
    return "str"


def load_table(path: Path, max_rows: int = 200_000) -> Table:
    """Читает csv/tsv: определяет разделитель, убирает BOM, приводит числовые столбцы."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    delim = _detect_delimiter(text[:8192], path.suffix.lower())
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        return Table(path, path.name, [], [], {}, {})
    raw: list[list[str]] = []
    for rec in reader:
        if not rec or all(not c.strip() for c in rec):
            continue
        rec = [c.strip() for c in rec[: len(header)]]
        rec += [""] * (len(header) - len(rec))
        raw.append(rec)
        if len(raw) >= max_rows:
            break
    types: dict[str, str] = {}
    for i, col in enumerate(header):
        types[col] = _column_type([r[i] for r in raw])
    rows = []
    for rec in raw:
        row: dict[str, Any] = {}
        for i, col in enumerate(header):
            v = rec[i]
            t = types[col]
            if t == "int":
                row[col] = int(v) if v != "" else None
            elif t == "float":
                row[col] = float(v) if v != "" else None
            else:
                row[col] = v
        rows.append(row)
    distinct = {}
    for col in header:
        if types[col] == "str":
            vals = sorted({r[col] for r in rows})
            if len(vals) <= _DISTINCT_LIMIT:
                distinct[col] = vals
    return Table(path, path.name, header, rows, types, distinct)


def table_summary(table: Table, sample_rows: int = 3) -> str:
    """Компактное описание таблицы для модели."""
    lines = [f"{table.name}: {len(table.rows)} rows; columns "
             + ", ".join(f"{c}:{table.types.get(c, 'str')}" for c in table.columns)]
    if table.distinct:
        lines.append("; ".join(f"{c} in {vals!r}" for c, vals in table.distinct.items()))
    if sample_rows > 0 and table.rows:
        lines.append("first rows: " + "; ".join(repr(r) for r in table.rows[:sample_rows])
                     + ("; ..." if len(table.rows) > sample_rows else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------- разбор вопроса

_TAIL_RE = re.compile(
    r"\s*\b(?:use bash\b|use the bash\b|compute it\b|work only\b|when you are done\b|write only\b)",
    re.IGNORECASE,
)
_PREAMBLE_RE = re.compile(
    r"the file (\S+?) has columns ([^()]+?)\s*\((\d+) rows?\)\s*\.?", re.IGNORECASE)
_QUOTE_RE = re.compile(r"(?<![\w])(?:'([^']+)'|\"([^\"]+)\"|‘([^’]+)’|“([^”]+)”)(?![\w])")
_NUM = r"(-?\d+(?:\.\d+)?)"

# Слова, после которых значение в кавычках не привязано к столбцу.
_NOT_COLUMN = {
    "the", "a", "an", "of", "in", "for", "from", "by", "to", "with", "is", "as", "and", "or", "than",
    "named", "called", "at", "on", "not", "are", "was", "were", "equals", "equal", "be", "where",
    "whose", "that", "which", "only", "all", "sold", "during", "between", "through", "within",
}
# Слова, которые ничего не меняют в смысле вопроса-агрегата.
_FILLER = {
    "how", "many", "much", "what", "is", "was", "are", "were", "the", "a", "an", "of", "in", "for",
    "with", "where", "whose", "during", "from", "and", "all", "sold", "ordered", "purchased",
    "bought", "recorded", "there", "did", "do", "does", "have", "has", "had", "overall", "combined",
    "altogether", "across", "by", "total", "file", "table", "rows", "row",
    "that", "which", "had", "been", "be", "inclusive", "inclusively", "listed", "contained", "contain",
    "contains", "appear", "appears", "count", "number", "column", "it", "to",
    "on", "at", "made", "placed", "shipped", "only", "orders", "order", "records", "entries",
}
_COUNT_WORDS = {"orders", "rows", "records", "sales", "entries", "transactions", "lines", "order",
                "row", "record", "sale", "entry", "transaction", "line"}
_AGG_WORDS = {
    "total": "sum", "sum": "sum", "average": "avg", "mean": "avg", "maximum": "max", "max": "max",
    "highest": "max", "largest": "max", "minimum": "min", "min": "min", "lowest": "min", "smallest": "min",
}
# Слова после «how many», которые не являются ссылкой на столбец (не missing_column, а отказ).
_HOW_MANY_NOT_COLUMN = {
    "distinct", "different", "unique", "times", "more", "fewer", "less", "kinds", "types", "percent",
    "of", "in", "the", "items", "pieces", "products", "regions", "months", "days", "dollars",
}
# Суммировать цену почти никогда не имеет смысла (скорее имели в виду выручку) — отказ.
_PRICE_LIKE = re.compile(r"(?:^|_)(?:price|cost|rate)s?$|^unit_?price$", re.IGNORECASE)


class _Missing(Exception):
    pass


class _Unparsed(Exception):
    pass


def _key(col: str) -> str:
    return re.sub(r"\W+", "_", col.strip().lower()).strip("_")


def _fmt_number(x: Any) -> str:
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    x = float(x)
    if x.is_integer():
        return str(int(x))
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _fmt_value(v: Any) -> str:
    return repr(v) if isinstance(v, str) else _fmt_number(v)


def answer_table_question(table: Table, question: str) -> TableAnswer:
    """Строгий разбор вопроса «агрегат по столбцу с фильтрами». При сомнении — отказ."""
    try:
        return _answer(table, question)
    except _Missing as exc:
        return TableAnswer("missing_column", detail=str(exc))
    except _Unparsed as exc:
        return TableAnswer("unparsed", detail=str(exc))


def _answer(table: Table, question: str) -> TableAnswer:
    notes: list[str] = []
    text = re.sub(r"\s+", " ", question or "").strip()
    m = _TAIL_RE.search(text)
    if m:
        text = text[: m.start()].strip()

    # Преамбула «The file X has columns a,b,c (N rows).»: сверяем с файлом и убираем.
    pm = _PREAMBLE_RE.search(text)
    if pm:
        fname, cols_s, nrows = pm.group(1), pm.group(2), int(pm.group(3))
        if fname.lower() != table.name.lower():
            raise _Unparsed(f"вопрос про файл {fname}, а загружен {table.name}")
        listed = [c.strip() for c in re.split(r"[,;]", cols_s) if c.strip()]
        if [c.lower() for c in listed] != [c.lower() for c in table.columns]:
            notes.append(f"columns in instruction {listed} differ from file {table.columns}")
        if nrows != len(table.rows):
            notes.append(f"instruction says {nrows} rows, file has {len(table.rows)}")
        text = (text[: pm.start()] + " " + text[pm.end():]).strip()

    sentences = [s for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    if len(sentences) != 1 or not sentences[0].endswith("?"):
        raise _Unparsed(f"ожидался ровно один вопрос, получено: {sentences!r}")
    q = sentences[0][:-1].strip()

    cols_by_key = {_key(c): c for c in table.columns}

    def lookup(word: str) -> Optional[str]:
        w = word.lower()
        for cand in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None, w + "s"):
            if cand and cand in cols_by_key:
                return cols_by_key[cand]
        return None

    # Значения в кавычках → плейсхолдеры qvN (регистр значения сохраняется).
    quoted: list[str] = []

    def _qrepl(mm: re.Match) -> str:
        quoted.append(next(g for g in mm.groups() if g is not None))
        return f" qv{len(quoted) - 1} "

    q = _QUOTE_RE.sub(_qrepl, q)
    if "'" in q or '"' in q:
        raise _Unparsed("непарные кавычки")
    s = " " + q.lower() + " "
    # Многословные имена столбцов («unit price») → один токен.
    for c in sorted(table.columns, key=len, reverse=True):
        if re.search(r"\W", c.strip()):
            s = re.sub(r"(?<!\w)" + re.escape(c.strip().lower()) + r"(?!\w)", " " + _key(c) + " ", s)
    s = re.sub(r"\s+", " ", s)

    filters: list[tuple[str, str, Any]] = []   # (столбец, оператор, значение)

    def consume(span: tuple[int, int]) -> None:
        nonlocal s
        s = s[: span[0]] + " " + s[span[1]:]

    # --- фильтры равенства по значениям в кавычках
    for i, val in enumerate(quoted):
        mm = re.search(rf"(\w+) (?:(?:is|equals|=) )?qv{i}(?!\d)", s)
        if not mm:
            raise _Unparsed(f"значение {val!r} без столбца")
        word = mm.group(1)
        col = lookup(word) if word not in _NOT_COLUMN else None
        if col is None and word in _NOT_COLUMN:
            # «units of 'widget'»: столбец однозначно определяется значением.
            owners = [c for c in table.columns if table.types[c] == "str" and any(r[c] == val for r in table.rows)]
            if len(owners) != 1:
                raise _Unparsed(f"не ясно, к какому столбцу относится {val!r}")
            col = owners[0]
            notes.append(f"column for {val!r} inferred as {col}")
            consume((mm.end() - len(f"qv{i}"), mm.end()))
        elif col is None:
            raise _Missing(f"no column {word!r} (referenced before {val!r}); columns: {table.columns}")
        else:
            consume(mm.span())
        filters.append((col, "==", _coerce_filter_value(table, col, val, notes)))

    # --- числовые диапазоны и сравнения
    num_patterns = [
        (rf"between (\w+) {_NUM} and {_NUM} \(?inclusive\)?", "range"),
        (rf"(\w+) (?:from )?{_NUM} (?:through|thru|to|-|–) {_NUM}(?: \(?inclusive\)?)?", "range"),
        (rf"(\w+) (?:greater than or equal to|at least|>=) {_NUM}", ">="),
        (rf"(\w+) (?:less than or equal to|at most|<=) {_NUM}", "<="),
        (rf"(\w+) (?:greater than|more than|above|over|>) {_NUM}", ">"),
        (rf"(\w+) (?:less than|fewer than|below|under|<) {_NUM}", "<"),
        (rf"(\w+) (?:of |= |equal to |equals )?{_NUM}", "=="),
    ]
    for pat, op in num_patterns:
        while True:
            mm = re.search(r"(?<!\w)" + pat + r"(?!\w)", s)
            if not mm:
                break
            word = mm.group(1)
            if word in _NOT_COLUMN or word in _FILLER:
                raise _Unparsed(f"число без столбца: {mm.group(0).strip()!r}")
            col = lookup(word)
            if col is None:
                raise _Missing(f"no column {word!r} (in {mm.group(0).strip()!r}); columns: {table.columns}")
            if table.types[col] == "str":
                raise _Unparsed(f"числовой фильтр по строковому столбцу {col}")
            nums = [float(g) if "." in g else int(g) for g in mm.groups()[1:]]
            if op == "range":
                lo, hi = nums
                if lo > hi:
                    raise _Unparsed(f"пустой диапазон {lo}..{hi}")
                filters.append((col, ">=", lo))
                filters.append((col, "<=", hi))
            else:
                filters.append((col, op, nums[0]))
            consume(mm.span())

    # --- мера
    s = re.sub(r"\s+", " ", s).strip()
    agg_name, target_col, s = _parse_measure(table, s, lookup)

    # --- проверка, что всё объяснено
    leftover = [w for w in re.findall(r"[^\s,]+", s) if w not in _FILLER]
    if leftover:
        raise _Unparsed(f"необъяснённые слова: {leftover}")

    # --- вычисление
    def ok(row: dict) -> bool:
        for col, op, v in filters:
            x = row.get(col)
            if x is None:
                return False
            if op == "==" and not x == v:
                return False
            if op == ">=" and not x >= v:
                return False
            if op == "<=" and not x <= v:
                return False
            if op == ">" and not x > v:
                return False
            if op == "<" and not x < v:
                return False
        return True

    sel = [r for r in table.rows if ok(r)]
    plan = _plan_text(agg_name, target_col, filters)
    if not sel:
        notes.append("no rows match the filters")
    if agg_name == "count":
        result: Any = len(sel)
    else:
        vals = [r[target_col] for r in sel if r.get(target_col) is not None]
        if len(vals) != len(sel):
            notes.append(f"{len(sel) - len(vals)} empty {target_col} cells skipped")
        if agg_name == "sum":
            result = sum(vals)
        elif not vals:
            return TableAnswer("empty", plan=plan, detail="; ".join(notes + [f"{agg_name} of empty set"]))
        elif agg_name == "avg":
            result = sum(vals) / len(vals)
        elif agg_name == "max":
            result = max(vals)
        else:
            result = min(vals)
    return TableAnswer("ok", answer=_fmt_number(result), plan=plan, detail="; ".join(notes))


def _coerce_filter_value(table: Table, col: str, val: str, notes: list[str]) -> Any:
    t = table.types[col]
    if t in ("int", "float"):
        if not _FLOAT_RE.fullmatch(val.strip()):
            raise _Unparsed(f"нечисловое значение {val!r} для числового столбца {col}")
        return float(val) if t == "float" or "." in val else int(val)
    present = {r[col] for r in table.rows}
    if val in present:
        return val
    ci = [p for p in present if p.lower() == val.lower()]
    if len(ci) == 1:
        notes.append(f"value {val!r} matched {ci[0]!r} ignoring case")
        return ci[0]
    notes.append(f"value {val!r} not found in column {col}")
    return val


def _parse_measure(table: Table, s: str, lookup) -> tuple[str, Optional[str], str]:
    """Возвращает (агрегат, столбец или None, остаток строки без меры)."""
    def numeric_col(word: str, ctx: str) -> str:
        col = lookup(word)
        if col is None:
            raise _Missing(f"no column {word!r} for measure {ctx!r}; columns: {table.columns}")
        if table.types[col] == "str":
            raise _Unparsed(f"мера по строковому столбцу {col}")
        return col

    # «total number of orders», «number of rows»
    mm = re.search(r"(?<!\w)(?:what (?:is|was|are|were) )?(?:the )?(?:total )?number of (\w+)\b", s)
    if mm:
        word = mm.group(1)
        if word in _COUNT_WORDS and lookup(word) is None:
            return ("count", None, s[:mm.start()] + " " + s[mm.end():])
        col = lookup(word)
        if col is not None and table.types[col] != "str" and not _PRICE_LIKE.search(_key(col)):
            return ("sum", col, s[:mm.start()] + " " + s[mm.end():])
        raise _Unparsed(f"«number of {word}» — неоднозначно")

    # «How many <...>»
    mm = re.search(r"\bhow many (\w+)\b", s)
    if mm:
        word = mm.group(1)
        col = lookup(word)
        if col is not None:
            if table.types[col] == "str":
                raise _Unparsed(f"«how many {word}» по строковому столбцу — возможно, число различных")
            if _PRICE_LIKE.search(_key(col)):
                raise _Unparsed(f"сумма цены {col} — вероятно, имелась в виду выручка")
            return ("sum", col, s[:mm.start()] + " " + s[mm.end():])
        if word in _COUNT_WORDS:
            return ("count", None, s[:mm.start()] + " " + s[mm.end():])
        if word in _FILLER or word in _AGG_WORDS or word in _HOW_MANY_NOT_COLUMN:
            raise _Unparsed(f"не понята мера после «how many {word}»")
        raise _Missing(f"no column {word!r} for 'how many'; columns: {table.columns}")

    # «What is the total/average/maximum <col>», «sum of <col>»
    mm = re.search(
        r"(?<!\w)(?:(?:what|which) (?:is|was|are|were) |(?:compute|calculate|find|give|report) )?(?:the )?"
        r"(total|sum|average|mean|maximum|max|highest|largest|minimum|min|lowest|smallest)"
        r"(?: (?:of|for))?(?: the)? (\w+)\b", s)
    if mm:
        agg = _AGG_WORDS[mm.group(1)]
        word = mm.group(2)
        if word in _COUNT_WORDS and lookup(word) is None:
            if agg == "sum" and word not in ("sales", "sale"):
                return ("count", None, s[:mm.start()] + " " + s[mm.end():])
            raise _Unparsed(f"«{mm.group(1)} {word}» — неоднозначно")
        if word in ("number", "count", "amount", "value", "quantity"):
            raise _Unparsed(f"«{mm.group(1)} {word}» — неоднозначно")
        col = numeric_col(word, mm.group(0))
        if agg == "sum" and _PRICE_LIKE.search(_key(col)):
            raise _Unparsed(f"сумма цены {col} — вероятно, имелась в виду выручка")
        return (agg, col, s[:mm.start()] + " " + s[mm.end():])
    raise _Unparsed(f"не понята мера: {s!r}")


def _plan_text(agg: str, col: Optional[str], filters: list[tuple[str, str, Any]]) -> str:
    head = "count(rows)" if agg == "count" else f"{agg}({col})"
    parts: list[str] = []
    used = set()
    for i, (c, op, v) in enumerate(filters):
        if i in used:
            continue
        if op == ">=":
            j = next((k for k in range(i + 1, len(filters))
                      if k not in used and filters[k][0] == c and filters[k][1] == "<="), None)
            if j is not None:
                used.add(j)
                parts.append(f"{_fmt_value(v)} <= {c} <= {_fmt_value(filters[j][2])}")
                continue
        parts.append(f"{c} {op} {_fmt_value(v)}")
    return head + (" where " + " and ".join(parts) if parts else "")
