"""Смысловой ремонт: эталонные реализации типовых функций и сверка поведения на пробных входах.

Зачем. Перебор мутаций (repair.py) принимает первую правку, с которой проходят ВИДИМЫЕ тесты.
Если проверяющий запускает более широкие скрытые тесты, такая правка может оказаться подгонкой:
`range(n)` -> `range(1, n)` чинит fib(10) == 55, но ломает fib(0). Имя функции при этом
однозначно говорит, что она должна делать. Поэтому:

  1. для функций с известными именами (total, maximum, reverse_words, fib, ...) есть эталоны;
     поведение функции сравнивается с эталоном на пробных входах, построенных из вызовов в тестах;
  2. расходящаяся функция заменяется эталоном, если от этого строго уменьшается число упавших тестов
     (то есть она действительно была сломана, а эталон согласуется с видимыми тестами);
  3. у неоднозначных имён несколько вариантов эталона (average: `/` или `//`, range_sum: включая конец
     или нет); берётся вариант, проходящий видимые тесты и ближайший по тексту к исходной функции.

  4. проверки входа исходной функции (`if not xs: return 0`, `raise ValueError`) переносятся в эталон,
     если сами не сломаны: скрытые тесты любят пустые и отрицательные входы.

Функция, которая видимые тесты уже проходит, заменяется только в двух случаях (скрытые тесты шире видимых):
  - она расходится со всеми вариантами эталона на пробах, а текстом отличается от ближайшего варианта
    одной-двумя точечными правками — это внесённый баг, который видимый тест не поймал (`best = xs[1]`);
  - её уже правили перебором или моделью (align_touched): правка под видимые тесты могла быть подгонкой.
Иначе написанная исправная функция (свой цикл, проверка входа, другой порядок min/max) не трогается.
Только стандартная библиотека.
"""
from __future__ import annotations

import ast
import builtins
import datetime
import difflib
import io
import json
import re
import shutil
import tempfile
import textwrap
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness import sandbox
from harness.skills.testrun import TestReport, run_tests, source_files, test_files

__all__ = ["SemanticResult", "reference_repair", "align_touched", "behavior_diff", "behavior_compare",
           "known_function"]

# ---------------------------------------------------------------- эталоны
# Имя -> варианты тела. Функция в эталоне всегда называется f; параметры переименовываются под исходник.
_R: dict[str, list[str]] = {}
_ALIAS: dict[str, str] = {}


def _ref(names: str, *variants: str) -> None:
    first, *rest = names.split()
    _R[first] = [v.strip("\n") + "\n" for v in variants]
    for alias in rest:
        _ALIAS[alias] = first


_ref("total sum_list list_sum sum_all add_all sum_values sum_numbers total_sum sum_of_list sum_elements",
     "def f(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s")
_ref("maximum max_value find_max largest max_of_list get_max max_element find_largest biggest",
     "def f(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best")
_ref("minimum min_value find_min smallest min_of_list get_min min_element find_smallest",
     "def f(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best")
_ref("average mean avg arithmetic_mean",
     "def f(xs):\n    return sum(xs) / len(xs)", "def f(xs):\n    return sum(xs) // len(xs)")
_ref("first_n take head take_first first_items", "def f(xs, n):\n    return xs[:n]")
_ref("last_n tail take_last last_items",
     "def f(xs, n):\n    return xs[len(xs) - n:]", "def f(xs, n):\n    return xs[-n:]")
_ref("count_even count_evens even_count num_even", "def f(xs):\n    return sum(1 for x in xs if x % 2 == 0)")
_ref("count_odd count_odds odd_count num_odd", "def f(xs):\n    return sum(1 for x in xs if x % 2 == 1)")
_ref("reverse_words reverse_word_order", "def f(s):\n    return ' '.join(reversed(s.split()))")
_ref("reverse_string reverse reverse_str reversed_string reverse_text reverse_list reversed_list",
     "def f(s):\n    return s[::-1]")
_ref("is_palindrome palindrome",
     "def f(s):\n    return s == s[::-1]",
     "def f(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]",
     "def f(s):\n    t = str(s)\n    return t == t[::-1]")
_ref("factorial fact",
     "def f(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result")
_ref("fib fibonacci nth_fib fib_n",
     "def f(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     "def f(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a")
_ref("clamp clip limit", "def f(x, lo, hi):\n    return max(lo, min(x, hi))")
_ref("unique dedupe remove_duplicates distinct uniq deduplicate unique_items",
     "def f(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out",
     "def f(xs):\n    return sorted(set(xs))")
_ref("count_vowels vowel_count num_vowels", "def f(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')")
_ref("range_sum sum_range sum_between",
     "def f(a, b):\n    return sum(range(a, b + 1))", "def f(a, b):\n    return sum(range(a, b))")
_ref("capitalize_words title_case titlecase capitalize_each",
     "def f(s):\n    return ' '.join(w.capitalize() for w in s.split())", "def f(s):\n    return s.title()")
_ref("dot dot_product inner_product scalar_product",
     "def f(xs, ys):\n    return sum(x * y for x, y in zip(xs, ys))")
_ref("running_total cumulative_sum cumsum prefix_sums running_sum cumulative",
     "def f(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n        out.append(s)\n    return out")
_ref("square_all squares square_list squared map_square", "def f(xs):\n    return [x * x for x in xs]")
_ref("double_all doubles double_list", "def f(xs):\n    return [x * 2 for x in xs]")
_ref("count_positive positive_count count_positives", "def f(xs):\n    return sum(1 for x in xs if x > 0)")
_ref("count_negative negative_count count_negatives", "def f(xs):\n    return sum(1 for x in xs if x < 0)")
_ref("count_zeros count_zero zero_count", "def f(xs):\n    return sum(1 for x in xs if x == 0)")
_ref("is_sorted is_ascending is_ordered", "def f(xs):\n    return all(a <= b for a, b in zip(xs, xs[1:]))")
_ref("longest_word longest", "def f(s):\n    return max(s.split(), key=len)")
_ref("shortest_word shortest", "def f(s):\n    return min(s.split(), key=len)")
_ref("abs_diff absolute_difference distance difference_abs", "def f(a, b):\n    return abs(a - b)")
_ref("countdown count_down", "def f(n):\n    return list(range(n, 0, -1))")
_ref("is_prime prime",
     "def f(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n"
     "        if n % i == 0:\n            return False\n    return True")
_ref("gcd greatest_common_divisor hcf", "def f(a, b):\n    while b:\n        a, b = b, a % b\n    return a")
_ref("lcm least_common_multiple",
     "def f(a, b):\n    x, y = a, b\n    while y:\n        x, y = y, x % y\n    return a * b // x")
_ref("flatten flatten_list concat_lists", "def f(xss):\n    out = []\n    for xs in xss:\n        out.extend(xs)\n    return out")
_ref("median middle_value",
     "def f(xs):\n    ys = sorted(xs)\n    return ys[len(ys) // 2]",
     "def f(xs):\n    ys = sorted(xs)\n    n = len(ys)\n    if n % 2:\n        return ys[n // 2]\n"
     "    return (ys[n // 2 - 1] + ys[n // 2]) / 2")
_ref("word_count count_words num_words words_count", "def f(s):\n    return len(s.split())")
_ref("power pow_int exponent raise_power",
     "def f(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result")
_ref("is_even even", "def f(n):\n    return n % 2 == 0")
_ref("is_odd odd", "def f(n):\n    return n % 2 == 1")
_ref("sign signum sgn", "def f(x):\n    if x > 0:\n        return 1\n    if x < 0:\n        return -1\n    return 0")
_ref("celsius_to_f celsius_to_fahrenheit c_to_f to_fahrenheit", "def f(c):\n    return c * 9 / 5 + 32")
_ref("fahrenheit_to_celsius f_to_c fahrenheit_to_c to_celsius", "def f(x):\n    return (x - 32) * 5 / 9")
_ref("second_largest second_max second_biggest",
     "def f(xs):\n    return sorted(set(xs))[-2]", "def f(xs):\n    return sorted(xs)[-2]")
_ref("second_smallest second_min",
     "def f(xs):\n    return sorted(set(xs))[1]", "def f(xs):\n    return sorted(xs)[1]")
_ref("binary_search bsearch",
     "def f(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
     "        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            lo = mid + 1\n"
     "        else:\n            hi = mid - 1\n    return -1")
_ref("linear_search index_of find_index",
     "def f(xs, target):\n    for i, x in enumerate(xs):\n        if x == target:\n            return i\n    return -1")
_ref("merge_sorted merge merge_lists merge_sorted_lists", "def f(a, b):\n    return sorted(a + b)")
_ref("sum_of_squares sum_squares", "def f(xs):\n    return sum(x * x for x in xs)")
_ref("product prod multiply_all product_of_list",
     "def f(xs):\n    result = 1\n    for x in xs:\n        result *= x\n    return result")
_ref("add plus sum_two add_numbers", "def f(a, b):\n    return a + b")
_ref("subtract sub minus", "def f(a, b):\n    return a - b")
_ref("multiply mul times", "def f(a, b):\n    return a * b")
_ref("divide div", "def f(a, b):\n    return a / b", "def f(a, b):\n    return a // b")
_ref("square sq", "def f(x):\n    return x * x")
_ref("cube", "def f(x):\n    return x * x * x")
_ref("double twice", "def f(x):\n    return x * 2")
_ref("triple", "def f(x):\n    return x * 3")
_ref("increment inc", "def f(x):\n    return x + 1")
_ref("decrement dec", "def f(x):\n    return x - 1")
_ref("negate", "def f(x):\n    return -x")
_ref("is_positive positive", "def f(x):\n    return x > 0")
_ref("is_negative negative", "def f(x):\n    return x < 0")
_ref("max_of_two larger max2 bigger", "def f(a, b):\n    return a if a > b else b")
_ref("min_of_two smaller min2", "def f(a, b):\n    return a if a < b else b")
_ref("sum_digits digit_sum sum_of_digits digits_sum", "def f(n):\n    return sum(int(c) for c in str(abs(n)))")
_ref("count_digits num_digits digit_count", "def f(n):\n    return len(str(abs(n)))")
_ref("count_char count_occurrences count_letter count_item count_of", "def f(s, ch):\n    return s.count(ch)")
_ref("is_anagram anagram are_anagrams", "def f(a, b):\n    return sorted(a) == sorted(b)")
_ref("to_upper upper uppercase shout", "def f(s):\n    return s.upper()")
_ref("to_lower lower lowercase", "def f(s):\n    return s.lower()")
_ref("even_numbers filter_even evens keep_even only_even", "def f(xs):\n    return [x for x in xs if x % 2 == 0]")
_ref("odd_numbers filter_odd odds keep_odd only_odd", "def f(xs):\n    return [x for x in xs if x % 2 == 1]")
_ref("positives filter_positive keep_positive only_positive", "def f(xs):\n    return [x for x in xs if x > 0]")
_ref("sum_even sum_evens sum_even_numbers", "def f(xs):\n    return sum(x for x in xs if x % 2 == 0)")
_ref("sum_odd sum_odds sum_odd_numbers", "def f(xs):\n    return sum(x for x in xs if x % 2 == 1)")
_ref("sort_asc sort_list sorted_list sort_ascending sort_numbers", "def f(xs):\n    return sorted(xs)")
_ref("sort_desc sort_descending sorted_desc", "def f(xs):\n    return sorted(xs, reverse=True)")
_ref("contains has includes", "def f(xs, x):\n    return x in xs")
_ref("first first_item first_element", "def f(xs):\n    return xs[0]")
_ref("last last_item last_element", "def f(xs):\n    return xs[-1]")
_ref("length size count_items list_length string_length str_len", "def f(xs):\n    return len(xs)")
_ref("is_leap_year leap_year is_leap",
     "def f(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)")
_ref("concat concatenate join_two", "def f(a, b):\n    return a + b")
_ref("repeat repeat_string", "def f(s, n):\n    return s * n")
_ref("is_empty empty", "def f(xs):\n    return len(xs) == 0")
_ref("argmax max_index index_of_max", "def f(xs):\n    return xs.index(max(xs))")
_ref("argmin min_index index_of_min", "def f(xs):\n    return xs.index(min(xs))")
_ref("join_words join_with_space", "def f(ws):\n    return ' '.join(ws)")
_ref("split_words words tokenize", "def f(s):\n    return s.split()")
_ref("remove_spaces strip_spaces", "def f(s):\n    return s.replace(' ', '')")
_ref("last_word", "def f(s):\n    return s.split()[-1]")
_ref("first_word", "def f(s):\n    return s.split()[0]")
_ref("swap_case swapcase", "def f(s):\n    return s.swapcase()")
_ref("range_list make_range", "def f(n):\n    return list(range(n))")
_ref("sum_to sum_to_n triangular triangle_number sum_up_to", "def f(n):\n    return n * (n + 1) // 2")
_ref("count_words_starting_with", "def f(s, ch):\n    return sum(1 for w in s.split() if w.startswith(ch))")
_ref("is_multiple is_divisible divisible", "def f(a, b):\n    return a % b == 0")
_ref("remainder mod modulo", "def f(a, b):\n    return a % b")
_ref("char_count count_chars num_chars", "def f(s):\n    return len(s)")
_ref("area_rectangle rectangle_area rect_area", "def f(w, h):\n    return w * h")
_ref("perimeter_rectangle rectangle_perimeter", "def f(w, h):\n    return 2 * (w + h)")
_ref("max_diff value_range spread", "def f(xs):\n    return max(xs) - min(xs)")
_ref("zip_sum add_lists pairwise_sum", "def f(xs, ys):\n    return [x + y for x, y in zip(xs, ys)]")
_ref("count_greater count_above count_greater_than", "def f(xs, t):\n    return sum(1 for x in xs if x > t)")
_ref("count_less count_below count_less_than", "def f(xs, t):\n    return sum(1 for x in xs if x < t)")
_ref("is_substring has_substring", "def f(s, sub):\n    return sub in s")
_ref("ends_with", "def f(s, suffix):\n    return s.endswith(suffix)")
_ref("starts_with", "def f(s, prefix):\n    return s.startswith(prefix)")
_ref("drop_n skip_n drop skip", "def f(xs, n):\n    return xs[n:]")
_ref("every_other every_second", "def f(xs):\n    return xs[::2]")
_ref("chunk chunks chunked", "def f(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]")
_ref("mode most_common most_frequent", "def f(xs):\n    return max(xs, key=xs.count)")


def known_function(name: str) -> Optional[str]:
    """Каноническое имя эталона для функции с таким именем (или None)."""
    low = name.lower().strip("_")
    if low in _R:
        return low
    if low in _ALIAS:
        return _ALIAS[low]
    # compute_total, get_maximum, my_average, total_of_list, maximum_value — то же самое с обвесом
    core = re.sub(r"^(?:get|calc|calculate|compute|find|my|do|make|return|the)_", "", low)
    core = re.sub(r"_(?:of_list|of_numbers|of_values|list|values|value|numbers|nums|func|function|impl)$", "", core)
    if core != low and core:
        return core if core in _R else _ALIAS.get(core)
    return None


def _arity(variant: str) -> int:
    return len(ast.parse(variant).body[0].args.args)


class _Rename(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self.mapping.get(node.arg, node.arg)
        return node


def _instantiate(variant: str, name: str, params: Optional[list[str]]) -> Optional[str]:
    """Текст эталона с нужным именем функции и именами параметров из исходника."""
    tree = ast.parse(variant)
    fn = tree.body[0]
    own = [a.arg for a in fn.args.args]
    if params is not None:
        if len(params) != len(own):
            return None
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | set(own)
        mapping = {}
        for old, new in zip(own, params):
            if new != old and new in used:
                return variant.replace("def f(", f"def {name}(", 1)   # коллизия имён: оставляем параметры эталона
            mapping[old] = new
        _Rename(mapping).visit(fn)
    fn.name = name
    return ast.unparse(tree) + "\n"


# ---------------------------------------------------------------- блоки функций в тексте (без AST: файл может не разбираться)

_DEF_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)?")


@dataclass
class _Block:
    name: str
    start: int            # индекс первой строки
    end: int              # индекс строки после блока
    params: Optional[list[str]]


def _params(raw: str) -> Optional[list[str]]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*(?::[^=]*)?(?:=.*)?$", part)
        if not m or part.startswith("*"):
            return None
        out.append(m.group(1))
    return out


def _blocks(text: str) -> tuple[list[str], list[_Block]]:
    lines = text.splitlines(keepends=True)
    blocks: list[_Block] = []
    i = 0
    while i < len(lines):
        m = _DEF_RE.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
                break
            j += 1
        end = j
        while end > i + 1 and not lines[end - 1].strip():
            end -= 1
        blocks.append(_Block(m.group(1), i, end, _params(m.group(2))))
        i = j
    return lines, blocks


def _replace_block(text: str, block: _Block, new_src: str) -> str:
    lines, _ = _blocks(text)
    eol = "\r\n" if text.count("\r\n") > text.count("\n") / 2 else "\n"
    new_lines = [ln + eol for ln in new_src.rstrip("\n").split("\n")]
    return "".join(lines[: block.start] + new_lines + lines[block.end:])


# ---------------------------------------------------------------- пробные входы

def _literal_calls(test_text: str, name: str) -> list[tuple]:
    try:
        tree = ast.parse(test_text)
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.keywords:
            continue
        f = node.func
        fname = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        args = node.args
        if fname in ("assertRaises", "assertRaisesRegex") and len(args) >= 2:
            g = args[1]
            gname = g.attr if isinstance(g, ast.Attribute) else g.id if isinstance(g, ast.Name) else None
            if gname == name:
                fname, args = name, args[2:]
        if fname != name:
            continue
        try:
            out.append(tuple(ast.literal_eval(a) for a in args))
        except (ValueError, SyntaxError, TypeError):
            continue
    return out


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _variants(value: object) -> list:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [0, 1, 2, 3, value + 1, value - 1, -value, value * 2, 7, 10, 12]
    if isinstance(value, float):
        return [0.0, 1.5, -value, value + 1]
    if isinstance(value, str):
        if _ISO_DATE.fullmatch(value):
            # дата ISO: остальные строки дали бы одну и ту же ошибку разбора в обеих версиях
            try:
                day = datetime.date.fromisoformat(value)
                return [(day + datetime.timedelta(days=k)).isoformat() for k in range(-7, 8) if k] + [""]
            except (ValueError, OverflowError):
                pass
        return ["", "a", "ab", "abc cba", "Hello World", "one two three four", value.upper(), value[::-1],
                value + " zeta", "level", "Banana apple", "x"]
    if isinstance(value, list):
        if all(type(x) is int for x in value):
            top = max(value) if value else 0
            return [[], value[:1], value[::-1], sorted(value), value + [top + 3], [-x for x in value],
                    value + value[:1], [0], [5, 3, 8, 1], [2, 2, 2], [-4, -1, -7], [1, 2, 3, 4, 5, 6], [7],
                    [4, 9, 2, 9], [3, 1, 2]]
        if all(isinstance(x, str) for x in value):
            return [[], value[::-1], value + ["zz"], ["b", "a"], value[:1]]
        if all(isinstance(x, list) for x in value):
            return [[], [[]], value[::-1], value + [[9]], [[1], [2, 3], []]]
        return [[], value[::-1]]
    return []


def _probes(calls: list[tuple], limit: int = 70) -> list[tuple]:
    out, seen = [], set()

    def add(t: tuple) -> None:
        key = repr(t)
        if key not in seen and len(out) < limit:
            seen.add(key)
            out.append(t)

    for call in calls:
        add(call)
    for call in calls:
        for i, arg in enumerate(call):
            for alt in _variants(arg):
                add(call[:i] + (alt,) + call[i + 1:])
    return out


# ---------------------------------------------------------------- вычисление поведения в отдельном процессе

_DRIVER = r'''
import copy, json, signal, sys
sys.dont_write_bytecode = True
sys.path.insert(0, ".")

class _T(Exception):
    pass

def _alarm(*_):
    raise _T()

def norm(x):
    if x is None:
        return ("N",)
    if isinstance(x, bool):
        return ("b", x)
    if isinstance(x, (int, float)):
        try:
            return ("n", round(float(x), 9)) if abs(x) < 1e15 else ("n", repr(x))
        except (OverflowError, ValueError):
            return ("n", repr(x))
    if isinstance(x, str):
        return ("s", x)
    if isinstance(x, list):
        return ("l", [norm(i) for i in x])
    if isinstance(x, tuple):
        return ("t", [norm(i) for i in x])
    if isinstance(x, dict):
        return ("d", sorted(repr((norm(k), norm(v))) for k, v in x.items()))
    if isinstance(x, (set, frozenset)):
        return ("S", sorted(repr(norm(i)) for i in x))
    return ("o", type(x).__name__)

spec = json.load(open(sys.argv[1]))
out = {}
for case in spec["cases"]:
    ns = {"__name__": "_probe_module"}
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        exec(compile(case["source"], "<probe>", "exec"), ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        out[case["id"]] = None
        continue
    fn = ns.get(case["func"])
    if not callable(fn):
        out[case["id"]] = None
        continue
    res = []
    for args in eval(case["probes"], {"__builtins__": {}}, {}):
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            value = fn(*copy.deepcopy(args))
            signal.setitimer(signal.ITIMER_REAL, 0)
            res.append(repr(norm(value)))
        except BaseException as exc:
            signal.setitimer(signal.ITIMER_REAL, 0)
            res.append("EXC:" + type(exc).__name__)
    out[case["id"]] = res
sys.stdout = sys.__stdout__
print("\n@@SIG@@" + json.dumps(out))
'''


def _signatures(cwd: Path, cases: list[dict], timeout: float = 30.0) -> dict[str, Optional[list[str]]]:
    """cases: [{id, source, func, probes(repr списка кортежей)}] -> {id: [repr результата | 'EXC:<тип>'] | None}."""
    if not cases:
        return {}
    tmp = Path(tempfile.mkdtemp(prefix="sem_"))
    try:
        (tmp / "driver.py").write_text(_DRIVER, encoding="utf-8")
        (tmp / "spec.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
        proc = sandbox.run_argv(sandbox.python_argv(str(tmp / "driver.py"), str(tmp / "spec.json")), cwd, timeout)
        text = getattr(proc, "stdout", "") or ""
        pos = text.rfind("@@SIG@@")
        if pos == -1:
            return {}
        return json.loads(text[pos + 7:].strip().splitlines()[0])
    except Exception:  # noqa: BLE001 — сверка необязательна: без неё просто нет подтверждения
        return {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _agrees(candidate: Optional[list[str]], reference: Optional[list[str]], min_informative: int = 3) -> Optional[bool]:
    """True — совпадает с эталоном везде, где эталон вернул значение; None — сравнить не на чем."""
    if candidate is None or reference is None or len(candidate) != len(reference):
        return None
    informative = [(c, r) for c, r in zip(candidate, reference) if not r.startswith("EXC")]
    if len(informative) < min_informative:
        return None
    return all(c == r for c, r in informative)


# Исключения «по недосмотру»: так падает сломанный код (xs[1] на списке из одного элемента, зацикливание).
# Остальные (ValueError и свои классы) обычно бросаются нарочно — это проверка входа, а не баг.
_ACCIDENTAL = {"IndexError", "ZeroDivisionError", "TypeError", "KeyError", "AttributeError", "NameError",
               "UnboundLocalError", "RecursionError", "StopIteration", "_T"}


def _divergence(candidate: Optional[list[str]], reference: Optional[list[str]],
                min_informative: int = 3) -> Optional[list[int]]:
    """Номера проб, где функция заведомо расходится с эталоном: эталон вернул значение, а функция вернула
    другое или упала «по недосмотру». Нарочно брошенное исключение (проверка входа) расхождением не считается.
    None — сравнить не на чем."""
    if candidate is None or reference is None or len(candidate) != len(reference):
        return None
    out, informative = [], 0
    for i, (c, r) in enumerate(zip(candidate, reference)):
        if r.startswith("EXC"):
            continue
        informative += 1
        if c.startswith("EXC"):
            if c[4:] in _ACCIDENTAL:
                out.append(i)
        elif c != r:
            out.append(i)
    return out if informative >= min_informative else None


def _fn_tokens(src: str) -> Optional[list[str]]:
    """Токены первой функции в тексте после нормализации (ast.unparse, без docstring и комментариев)."""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except (SyntaxError, ValueError):
        return None
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return None
    body = fn.body
    if len(body) > 1 and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        fn.body = body[1:]
    fn.returns = None
    for a in fn.args.args:
        a.annotation = None
    try:
        toks = tokenize.generate_tokens(io.StringIO(ast.unparse(fn) + "\n").readline)
        return [t.string for t in toks if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                                                          tokenize.DEDENT, tokenize.ENDMARKER, tokenize.COMMENT)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


def _guards(block_src: str) -> tuple[list[str], list[ast.stmt]]:
    """Параметры функции и её начальные проверки входа: `if <условие на параметры>: return/raise ...`."""
    try:
        tree = ast.parse(textwrap.dedent(block_src))
    except (SyntaxError, ValueError):
        return [], []
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return [], []
    params = [a.arg for a in fn.args.args]
    allowed = set(params) | set(dir(builtins))
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    out = []
    for stmt in body[:-1]:
        if not (isinstance(stmt, ast.If) and not stmt.orelse and stmt.body
                and isinstance(stmt.body[-1], (ast.Return, ast.Raise)) and len(stmt.body) == 1):
            break
        names = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
        names -= {n.id for n in ast.walk(stmt.body[0]) if isinstance(n, ast.Name) and isinstance(stmt.body[0], ast.Raise)}
        if not names <= allowed:
            break
        out.append(stmt)
    return params, out


def _grafted(block_src: str, ref_src: str) -> Optional[str]:
    """Эталон с начальными проверками входа исходной функции (None — проверок нет или их не перенести)."""
    params, guards = _guards(block_src)
    if not guards:
        return None
    try:
        tree = ast.parse(ref_src)
    except (SyntaxError, ValueError):
        return None
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or [a.arg for a in fn.args.args] != params:
        return None
    have = {ast.dump(st) for st in fn.body}
    extra = [g for g in guards if ast.dump(g) not in have]
    if not extra:
        return None
    fn.body = extra + fn.body
    return ast.unparse(tree) + "\n"


def _graft_ok(graft: Optional[list[str]], reference: Optional[list[str]]) -> bool:
    """Эталон с проверками входа на области эталона ведёт себя как эталон: там, где эталон вернул значение,
    то же значение или нарочно брошенное исключение (проверка отсекла вход). Иначе сломана сама проверка."""
    if graft is None or reference is None or len(graft) != len(reference):
        return False
    for g, r in zip(graft, reference):
        if r.startswith("EXC"):
            continue
        if g.startswith("EXC"):
            if g[4:] in _ACCIDENTAL:
                return False
        elif g != r:
            return False
    return True


def _near_mutation(a_src: str, b_src: str, max_regions: int = 2, max_tokens: int = 3) -> bool:
    """Тексты функций отличаются одной-двумя точечными правками (так выглядит внесённый баг: `<` вместо `>`,
    `xs[1]` вместо `xs[0]`, `not in` вместо `in`). Иначе написанная исправная функция обычно отличается сильнее."""
    a, b = _fn_tokens(a_src), _fn_tokens(b_src)
    if a is None or b is None or a == b:
        return False
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    changed = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    return len(ops) <= max_regions and changed <= max_tokens


# ---------------------------------------------------------------- основной проход

@dataclass
class SemanticResult:
    texts: dict[str, str] = field(default_factory=dict)      # изменённые файлы: относительный путь -> текст
    edits: list[str] = field(default_factory=list)
    report: Optional[TestReport] = None                      # отчёт тестов после правок (если они были)
    confirmed: list[str] = field(default_factory=list)       # функции, совпавшие с эталоном
    replaced: list[str] = field(default_factory=list)


def _tests_text(workspace: Path) -> str:
    parts = []
    for p in test_files(workspace):
        try:
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(parts)


def _better(rep: TestReport, base: TestReport) -> bool:
    """Строго лучше: всё прошло, либо исчезла ошибка запуска, либо упавших стало строго меньше."""
    if rep.timed_out:
        return False
    if rep.ok:
        return True
    if rep.error:
        return False
    if base.error or base.timed_out:
        return True
    if base.total and rep.total != base.total:
        return False
    return bool(rep.failed) and set(rep.failed) < set(base.failed)


def reference_repair(workspace: Path, argv: Optional[list[str]], base: TestReport,
                     test_timeout: float = 10.0) -> SemanticResult:
    """Заменяет эталонами сломанные функции с известными именами. Сам workspace не меняет."""
    workspace = Path(workspace)
    result = SemanticResult()
    sources = source_files(workspace)
    if not sources or base.ok:
        return result
    texts = {}
    for p in sources:
        try:
            texts[str(p.relative_to(workspace))] = p.read_bytes().decode("utf-8", errors="surrogateescape")
        except OSError:
            continue
    tests = _tests_text(workspace)

    # известные функции и их пробы
    known: list[tuple[str, _Block, str, list[tuple]]] = []
    for rel, text in texts.items():
        for block in _blocks(text)[1]:
            canon = known_function(block.name)
            if canon is None:
                continue
            calls = _literal_calls(tests, block.name)
            calls = [c for c in calls if any(len(c) == _arity(v) for v in _R[canon])]
            if calls:
                known.append((rel, block, canon, _probes(calls)))
    # функция, которую тесты вызывают, а в исходнике её нет
    missing = sorted(set(re.findall(r"has no attribute '(\w+)'", base.raw or "")))
    if not known and not missing:
        return result

    tmp = Path(tempfile.mkdtemp(prefix="semws_"))
    work = tmp / "ws"
    try:
        shutil.copytree(workspace, work, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))

        def write(rel: str, text: str) -> None:
            (work / rel).write_bytes(text.encode("utf-8", errors="surrogateescape"))

        cases = []
        grafts: dict[tuple[int, int], str] = {}      # (функция, вариант) -> эталон с проверками входа из исходника
        for k, (rel, block, canon, probes) in enumerate(known):
            pr = repr(probes)
            cases.append({"id": f"cur{k}", "source": texts[rel], "func": block.name, "probes": pr})
            block_src = "".join(_blocks(texts[rel])[0][block.start: block.end])
            for vi, variant in enumerate(_R[canon]):
                cases.append({"id": f"ref{k}_{vi}", "source": variant, "func": "f", "probes": pr})
                inst = _instantiate(variant, block.name, block.params)
                graft = _grafted(block_src, inst) if inst is not None else None
                if graft is not None:
                    grafts[(k, vi)] = graft
                    cases.append({"id": f"gft{k}_{vi}", "source": graft, "func": block.name, "probes": pr})
        sigs = _signatures(work, cases)

        current = base
        pending: list[tuple[str, str, str]] = []     # (файл, функция, текст эталона) — расходятся, но поодиночке не помогли
        for k, (rel, block, canon, probes) in enumerate(known):
            cur_sig = sigs.get(f"cur{k}")
            verdicts = [_agrees(cur_sig, sigs.get(f"ref{k}_{vi}")) for vi in range(len(_R[canon]))]
            # Совпадение с одним из вариантов ещё не значит, что функция исправна: fib с «return b»
            # ведёт себя как fib со сдвигом на единицу. Пробуем варианты, которые ведут себя иначе;
            # принят будет только тот, от которого упавших тестов станет строго меньше.
            differing = [vi for vi, verdict in enumerate(verdicts) if verdict is not True]
            if not differing:
                result.confirmed.append(block.name)
                continue
            # блок мог сдвинуться после предыдущих замен в том же файле — ищем заново по имени
            now = next((b for b in _blocks(texts[rel])[1] if b.name == block.name), None)
            if now is None:
                continue
            original_block = "".join(_blocks(texts[rel])[0][now.start: now.end])
            options = []
            for vi in differing:
                src = _instantiate(_R[canon][vi], block.name, now.params)
                if src is None:
                    continue
                ratio = difflib.SequenceMatcher(None, original_block, src).ratio()
                options.append((ratio, 0, src))
                # Проверки входа исходника (`if not xs: return 0`, `raise ValueError`) эталон потерял бы, а скрытые
                # тесты любят пустые и отрицательные входы. Эталон с этими проверками пробуем первым — если на
                # области эталона он ведёт себя как эталон (сама проверка не сломана).
                graft = grafts.get((k, vi))
                if graft is not None and _graft_ok(sigs.get(f"gft{k}_{vi}"), sigs.get(f"ref{k}_{vi}")):
                    options.append((ratio, 1, graft))
            options.sort(key=lambda t: (-t[0], -t[1]))
            for _, _, src in options:
                new_text = _replace_block(texts[rel], now, src)
                try:
                    compile(new_text, rel, "exec", dont_inherit=True)
                except (SyntaxError, ValueError):
                    # в файле может быть ещё одна синтаксическая ошибка в другой функции — тесты покажут
                    pass
                write(rel, new_text)
                rep = run_tests(work, argv, timeout=test_timeout)
                if _better(rep, current):
                    texts[rel] = new_text
                    current = rep
                    result.texts[rel] = new_text
                    result.replaced.append(block.name)
                    result.edits.append(f"{rel}: {block.name} -> reference implementation")
                    break
                write(rel, texts[rel])
            else:
                if any(v is True for v in verdicts):
                    result.confirmed.append(block.name)
                elif options:
                    pending.append((rel, block.name, options[0][2]))
            if current.ok:
                break

        if not current.ok and not result.texts and len(pending) > 1 and not base.failed:
            # Свой формат тестов: имён упавших нет, улучшение видно только по полному успеху.
            # Тогда заменяем все расходящиеся с эталоном функции разом.
            trial = dict(texts)
            for rel, name, src in pending:
                now = next((b for b in _blocks(trial[rel])[1] if b.name == name), None)
                if now is not None:
                    trial[rel] = _replace_block(trial[rel], now, src)
            for rel, text in trial.items():
                write(rel, text)
            rep = run_tests(work, argv, timeout=test_timeout)
            if rep.ok:
                for rel, text in trial.items():
                    if text != texts[rel]:
                        result.texts[rel] = text
                texts = trial
                current = rep
                result.replaced += [name for _, name, _ in pending]
                result.edits.append("replaced together: " + ", ".join(name for _, name, _ in pending))
            else:
                for rel, text in texts.items():
                    write(rel, text)

        # после замен ошибка запуска могла смениться на «нет такой функции» — смотрим свежий отчёт
        missing = sorted(set(missing) | set(re.findall(r"has no attribute '(\w+)'", current.raw or "")))
        if not current.ok and missing:
            target = _import_target(tests, texts)
            for name in missing:
                canon = known_function(name)
                if canon is None or target is None:
                    continue
                if any(b.name == name for b in _blocks(texts[target])[1]):
                    continue
                arities = {len(c) for c in _literal_calls(tests, name)}
                for variant in _R[canon]:
                    if arities and _arity(variant) not in arities:
                        continue
                    src = _instantiate(variant, name, None)
                    sep = "" if texts[target].endswith("\n\n\n") else "\n\n" if texts[target].endswith("\n") else "\n\n\n"
                    new_text = texts[target] + sep + src
                    write(target, new_text)
                    rep = run_tests(work, argv, timeout=test_timeout)
                    if _better(rep, current):
                        texts[target] = new_text
                        current = rep
                        result.texts[target] = new_text
                        result.replaced.append(name)
                        result.edits.append(f"{target}: added {name} (reference implementation)")
                        break
                    write(target, texts[target])

        # Баг, который видимые тесты не ловят (жюри добавит проверки сверху). Функция проходит свои тесты,
        # но на пробах расходится со всеми вариантами эталона, а текстом отличается от ближайшего варианта
        # одной-двумя точечными правками: так выглядит внесённый баг (`best = xs[1]`, `not in`), а не иначе
        # написанная исправная функция — её не трогаем. Замена принимается, только если видимые тесты не хуже.
        for k, (rel, block, canon, probes) in enumerate(known):
            if block.name in result.replaced:
                continue
            refs = [sigs.get(f"ref{k}_{vi}") for vi in range(len(_R[canon]))]
            if not _diverges_from_all(sigs.get(f"cur{k}"), refs):
                continue
            now = next((b for b in _blocks(texts[rel])[1] if b.name == block.name), None)
            if now is None:
                continue
            block_src = "".join(_blocks(texts[rel])[0][now.start: now.end])
            options = []
            for variant in _R[canon]:
                src = _instantiate(variant, block.name, now.params)
                if src is not None and _near_mutation(block_src, src):
                    options.append((difflib.SequenceMatcher(None, block_src, src).ratio(), src))
            for _, src in sorted(options, key=lambda t: -t[0]):
                new_text = _replace_block(texts[rel], now, src)
                write(rel, new_text)
                rep = run_tests(work, argv, timeout=test_timeout)
                if _not_worse(rep, current):
                    texts[rel] = new_text
                    current = rep
                    result.texts[rel] = new_text
                    result.replaced.append(block.name)
                    result.edits.append(f"{rel}: {block.name} -> reference implementation (diverges on probes)")
                    break
                write(rel, texts[rel])
        if result.texts:
            result.report = current
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def _diverges_from_all(cur: Optional[list[str]], refs: list[Optional[list[str]]]) -> bool:
    """Функция заведомо расходится с каждым вариантом эталона и не держит своей проверки входа:
    нарочно брошенное исключение или значение там, где эталон падает, при замене потерялись бы."""
    if cur is None or not refs:
        return False
    for ref in refs:
        div = _divergence(cur, ref)
        if not div:          # совпадает с вариантом или сравнить не на чем
            return False
    for c, rs in zip(cur, zip(*refs)):
        if c.startswith("EXC") and c[4:] not in _ACCIDENTAL:
            return False
        if not c.startswith("EXC") and all(r.startswith("EXC") for r in rs):
            return False
    return True


def _not_worse(rep: TestReport, base: TestReport) -> bool:
    """Видимые тесты после замены не хуже: всё прошло, либо упали те же или часть из тех же."""
    if rep.timed_out or rep.error:
        return False
    if rep.ok:
        return True
    if base.ok or base.error or base.timed_out or not base.failed:
        return False
    return (not base.total or rep.total == base.total) and set(rep.failed) <= set(base.failed)


def align_touched(workspace: Path, argv: Optional[list[str]], original: dict[str, str], current: dict[str, str],
                  test_timeout: float = 10.0) -> SemanticResult:
    """Дочинка до эталона функций, которые уже правили (перебор мутаций или модель), см. P1-11.

    original и current — тексты исходников (относительный путь -> текст) до и после правки. Функция
    с известным именем, чьё поведение правка изменила, но которая на пробах всё ещё расходится со всеми
    вариантами эталона, заменяется ближайшим вариантом — если видимые тесты остаются не хуже. Мы её уже
    трогали, значит она была сломана, а правка, подобранная под видимые тесты, могла оказаться подгонкой.
    Сам workspace не меняет; изменённые файлы — в result.texts."""
    workspace = Path(workspace)
    result = SemanticResult()
    tests = _tests_text(workspace)
    texts = dict(current)
    touched: list[tuple[str, _Block, str, list[tuple]]] = []
    for rel, text in texts.items():
        before = original.get(rel)
        if before is None or before == text:
            continue
        old_blocks = {b.name: b for b in _blocks(before)[1]}
        old_lines = _blocks(before)[0]
        lines, blocks = _blocks(text)
        for block in blocks:
            canon = known_function(block.name)
            if canon is None:
                continue
            old = old_blocks.get(block.name)
            new_src = "".join(lines[block.start: block.end])
            if old is not None:
                old_src = "".join(old_lines[old.start: old.end])
                if old_src == new_src or (_fn_tokens(old_src) is not None and _fn_tokens(old_src) == _fn_tokens(new_src)):
                    continue
            calls = [c for c in _literal_calls(tests, block.name) if any(len(c) == _arity(v) for v in _R[canon])]
            if calls:
                touched.append((rel, block, canon, _probes(calls)))
    if not touched:
        return result

    tmp = Path(tempfile.mkdtemp(prefix="semal_"))
    work = tmp / "ws"
    try:
        shutil.copytree(workspace, work, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "answer.txt"))

        def write(rel: str, text: str) -> None:
            (work / rel).write_bytes(text.encode("utf-8", errors="surrogateescape"))

        for rel, text in texts.items():
            write(rel, text)
        cases = []
        for k, (rel, block, canon, probes) in enumerate(touched):
            pr = repr(probes)
            cases.append({"id": f"cur{k}", "source": texts[rel], "func": block.name, "probes": pr})
            cases.append({"id": f"old{k}", "source": original[rel], "func": block.name, "probes": pr})
            for vi, variant in enumerate(_R[canon]):
                cases.append({"id": f"ref{k}_{vi}", "source": variant, "func": "f", "probes": pr})
        sigs = _signatures(work, cases)
        if not sigs:
            return result
        report = run_tests(work, argv, timeout=test_timeout)
        for k, (rel, block, canon, probes) in enumerate(touched):
            cur_sig, old_sig = sigs.get(f"cur{k}"), sigs.get(f"old{k}")
            if cur_sig is None or cur_sig == old_sig:
                continue                     # поведение правка не изменила — функцию на деле не чинили
            refs = [sigs.get(f"ref{k}_{vi}") for vi in range(len(_R[canon]))]
            if not _diverges_from_all(cur_sig, refs):
                result.confirmed.append(block.name)
                continue
            now = next((b for b in _blocks(texts[rel])[1] if b.name == block.name), None)
            if now is None:
                continue
            block_src = "".join(_blocks(texts[rel])[0][now.start: now.end])
            options = []
            for variant in _R[canon]:
                src = _instantiate(variant, block.name, now.params)
                if src is not None:
                    options.append((difflib.SequenceMatcher(None, block_src, src).ratio(), src))
            for _, src in sorted(options, key=lambda t: -t[0]):
                new_text = _replace_block(texts[rel], now, src)
                write(rel, new_text)
                rep = run_tests(work, argv, timeout=test_timeout)
                if _not_worse(rep, report):
                    texts[rel] = new_text
                    report = rep
                    result.texts[rel] = new_text
                    result.replaced.append(block.name)
                    result.edits.append(f"{rel}: {block.name} -> reference implementation (aligned after repair)")
                    break
                write(rel, texts[rel])
        if result.texts:
            result.report = report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def _import_target(tests: str, texts: dict[str, str]) -> Optional[str]:
    """Файл исходника, который импортируют тесты (куда дописывать недостающую функцию)."""
    stems = {Path(rel).stem: rel for rel in texts}
    for m in re.finditer(r"^\s*(?:import|from)\s+([\w.]+)", tests, re.MULTILINE):
        stem = m.group(1).split(".")[0]
        if stem in stems:
            return stems[stem]
    return next(iter(texts), None)


def behavior_diff(workspace: Path, texts_a: dict[str, str], texts_b: dict[str, str]) -> Optional[list[str]]:
    """Функции, которые ведут себя по-разному в двух версиях исходников (None — сравнить не удалось)."""
    return behavior_compare(workspace, texts_a, texts_b)[0]


# столько проб, где хотя бы одна версия вернула значение, нужно каждой изменённой функции,
# чтобы совпадение ответов что-то значило
MIN_INFORMATIVE_PROBES = 3


def behavior_compare(workspace: Path, texts_a: dict[str, str],
                     texts_b: dict[str, str]) -> tuple[Optional[list[str]], bool]:
    """(различия как у behavior_diff, информативны ли пробы). Пробы неинформативны, если изменённую функцию
    пробы не вызывают (метод класса, нет вызовов в тестах) или почти на всех пробах обе версии бросают
    исключение: тогда пустой список различий не значит, что версии совпадают."""
    workspace = Path(workspace)
    tests = _tests_text(workspace)
    cases, names, changed = [], [], []
    informative = True
    for rel, a in texts_a.items():
        b = texts_b.get(rel)
        if b is None or a == b:
            continue
        lines_a, blocks_a = _blocks(a)
        lines_b, blocks_b = _blocks(b)
        src_a = {bl.name: "".join(lines_a[bl.start:bl.end]) for bl in blocks_a}
        src_b = {bl.name: "".join(lines_b[bl.start:bl.end]) for bl in blocks_b}
        rest_a = "".join(ln for i, ln in enumerate(lines_a) if not any(bl.start <= i < bl.end for bl in blocks_a))
        rest_b = "".join(ln for i, ln in enumerate(lines_b) if not any(bl.start <= i < bl.end for bl in blocks_b))
        if rest_a.strip() != rest_b.strip():
            # правка вне верхнеуровневых функций (класс, константа): пробами её не проверить
            informative = False
        for block in blocks_a:
            touched = src_a.get(block.name) != src_b.get(block.name)
            calls = _literal_calls(tests, block.name)
            if not calls:
                if touched:
                    informative = False
                continue
            pr = repr(_probes(calls))
            k = len(names)
            names.append(block.name)
            changed.append(touched)
            cases.append({"id": f"a{k}", "source": a, "func": block.name, "probes": pr})
            cases.append({"id": f"b{k}", "source": b, "func": block.name, "probes": pr})
    if not cases:
        return [], False
    sigs = _signatures(workspace, cases)
    if not sigs:
        return None, False
    diff = []
    for k, name in enumerate(names):
        sa, sb = sigs.get(f"a{k}"), sigs.get(f"b{k}")
        if sa is None or sb is None or sa != sb:
            diff.append(name)
        elif changed[k]:
            useful = sum(1 for x, y in zip(sa, sb) if not (x.startswith("EXC:") and y.startswith("EXC:")))
            if useful < MIN_INFORMATIVE_PROBES:
                informative = False
    return diff, informative
