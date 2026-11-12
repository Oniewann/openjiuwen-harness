"""Применение исправлений, присланных моделью.

Чтобы не платить выходными токенами за переписывание всего файла, модель просят прислать
только изменённые функции целиком. Здесь они вставляются на место одноимённых определений
верхнего уровня; остальной файл не меняется.
"""
from __future__ import annotations

import ast
import re
import textwrap
from typing import Optional

from .textutil import extract_code_blocks, strip_think

_FILE_TAG_RE = re.compile(r"^\s*#\s*(?:file\s*[:=]\s*)?([\w./\-]+\.py)\s*$", re.IGNORECASE)
_DEF_RE = re.compile(r"^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)", re.MULTILINE)


def _parse(code: str) -> Optional[ast.Module]:
    for candidate in (code, textwrap.dedent(code)):
        try:
            return ast.parse(candidate)
        except SyntaxError:
            continue
    return None


def _node_start(node: ast.AST) -> int:
    """Первая строка определения с учётом декораторов (нумерация с 1)."""
    decorators = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decorators])


def _named(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return None


def _assigned_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def defined_names(code: str) -> list[str]:
    """Имена функций и классов верхнего уровня (по тексту, работает и для синтаксически битого файла)."""
    return [a or b for a, b in _DEF_RE.findall(code)]


def apply_patch(original: str, patch: str) -> Optional[str]:
    """Возвращает исходник с заменёнными определениями или None, если заплатка не применима."""
    patch = textwrap.dedent(patch).strip("\n")
    patch_tree = _parse(patch)
    if patch_tree is None:
        return None
    patch_lines = patch.splitlines()
    original_tree = _parse(original)

    if original_tree is None:
        # Исходник не разбирается (баг — синтаксическая ошибка). Принимаем заплатку только как
        # полную замену файла: она должна определять почти всё, что было в оригинале.
        wanted = set(defined_names(original))
        got = {n for n in (_named(node) for node in patch_tree.body) if n}
        if wanted and len(wanted & got) >= max(1, int(0.8 * len(wanted))):
            return patch + "\n"
        return _replace_blocks_by_text(original, patch_tree, patch_lines)

    lines = original.splitlines()
    spans: dict[str, tuple[int, int]] = {}
    assigns: dict[str, tuple[int, int]] = {}
    last_import_line = 0
    for node in original_tree.body:
        name = _named(node)
        if name:
            spans[name] = (_node_start(node), node.end_lineno or node.lineno)
        target = _assigned_name(node)
        if target:
            assigns[target] = (node.lineno, node.end_lineno or node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = node.end_lineno or node.lineno

    replacements: list[tuple[int, int, list[str]]] = []
    additions: list[str] = []
    new_imports: list[str] = []
    for node in patch_tree.body:
        segment = patch_lines[_node_start(node) - 1: node.end_lineno] if hasattr(node, "lineno") else []
        name = _named(node)
        if name:
            if name in spans:
                start, end = spans[name]
                replacements.append((start, end, segment))
            else:
                additions.append("\n".join(segment))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = "\n".join(segment).strip()
            if text and text not in original:
                new_imports.append(text)
            continue
        target = _assigned_name(node)
        if target and target in assigns:
            start, end = assigns[target]
            replacements.append((start, end, segment))
    if not replacements and not additions:
        return None

    for start, end, segment in sorted(replacements, key=lambda r: -r[0]):
        lines[start - 1: end] = segment
    result = "\n".join(lines).rstrip("\n") + "\n"
    if additions:
        result += "\n\n" + "\n\n\n".join(additions) + "\n"
    if new_imports:
        result = _insert_imports(result, new_imports)
    try:
        compile(result, "<patched>", "exec")
    except (SyntaxError, ValueError):
        return None
    return result


def _replace_blocks_by_text(original: str, patch_tree: ast.Module, patch_lines: list[str]) -> Optional[str]:
    """Исходник не разбирается, а заплатка — несколько функций: заменяем их блоки по тексту
    (от строки «def имя(» до следующего оператора верхнего уровня). Годится, если после замены файл компилируется."""
    lines = original.splitlines()
    replaced = 0
    for node in patch_tree.body:
        name = _named(node)
        if not name or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        head = re.compile(r"^(?:async\s+def|def|class)\s+" + re.escape(name) + r"\b")
        start = next((i for i, line in enumerate(lines) if head.match(line)), None)
        if start is None:
            continue
        end = start + 1
        while end < len(lines):
            line = lines[end]
            if line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
                break
            end += 1
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        lines[start:end] = patch_lines[_node_start(node) - 1: node.end_lineno]
        replaced += 1
    if not replaced:
        return None
    result = "\n".join(lines).rstrip("\n") + "\n"
    try:
        compile(result, "<patched>", "exec")
    except (SyntaxError, ValueError):
        return None
    return result


def _insert_imports(source: str, imports: list[str]) -> str:
    """Добавляет импорты после последнего импорта верхнего уровня (или после docstring модуля)."""
    tree = _parse(source)
    position = 0
    if tree is not None:
        for index, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                position = node.end_lineno or node.lineno
            elif index == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                    and isinstance(node.value.value, str):
                position = max(position, node.end_lineno or node.lineno)
    lines = source.splitlines()
    lines[position:position] = imports
    return "\n".join(lines) + "\n"


def split_by_file(reply: str, files: dict[str, str]) -> dict[str, str]:
    """Раскладывает блоки кода из ответа модели по файлам: имя файла -> текст заплатки.

    Файл определяется по первой строке блока («# file: mylib.py»), иначе — по тому, в каком
    файле определены функции из блока; если файл один, всё идёт в него.
    """
    text = strip_think(reply)
    blocks = [body for lang, body in extract_code_blocks(text) if lang in ("python", "py", "python3", "")]
    if not blocks:
        cleaned = text.strip()
        if "```" in cleaned:  # незакрытое ограждение: ответ оборван по лимиту токенов
            cleaned = re.sub(r"^[A-Za-z0-9_+\-.]*[ \t]*\r?\n", "", cleaned.split("```", 1)[1])
        blocks = [cleaned]
    names = list(files)
    result: dict[str, list[str]] = {}
    for block in blocks:
        body_lines = block.strip("\n").splitlines()
        if not body_lines:
            continue
        target = None
        tag = _FILE_TAG_RE.match(body_lines[0])
        if tag:
            wanted = tag.group(1)
            target = next((n for n in names if n == wanted or n.endswith("/" + wanted)
                           or wanted.endswith("/" + n)), None)
            body_lines = body_lines[1:]
        body = "\n".join(body_lines)
        if target is None:
            defined = set(defined_names(body))
            best, best_hits = None, 0
            for name, source in files.items():
                hits = len(defined & set(defined_names(source)))
                if hits > best_hits:
                    best, best_hits = name, hits
            target = best or (names[0] if names else None)
        if target is None or not body.strip():
            continue
        result.setdefault(target, []).append(body)
    return {name: "\n\n\n".join(parts) for name, parts in result.items()}
