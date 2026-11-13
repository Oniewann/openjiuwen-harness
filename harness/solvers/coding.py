"""Решатель задач coding: починить код так, чтобы прошли тесты.

Единственный тип задач с надёжной самопроверкой — тесты можно запустить. Поэтому здесь нет
голосования: ответ DONE пишется только после того, как тесты действительно прошли.

Порядок:
  0 токенов  — запустить тесты; если уже проходят, задача решена; если тесты противоречат
               друг другу (одинаковый вызов, разные ожидания) — задача невыполнима;
  навык      — ремонт перебором мутаций (при HARNESS_FASTPATH — сразу, иначе — как бесплатная
               страховка после первой неудачи модели);
  fast       — модель видит файл и упавшие утверждения, присылает только изменённые функции;
  fast_alt   — вторая дешёвая модель: от текущего лучшего состояния, с исходником тестов
               (другая модель, а не та же ещё раз: её ошибки и отказы меньше связаны с первой);
  strong     — модель с рассуждениями;
  react      — агент openJiuwen с bash в копии каталога.
После каждой правки тесты запускаются заново; тестовые файлы защищены от изменения.
"""
from __future__ import annotations

import asyncio
import re
import shlex
import shutil
from pathlib import Path
from typing import Optional

from ..llm import BudgetShort, LLMDead, LLMError, TaskCapExceeded
from ..patching import apply_patch, split_by_file
from ..textutil import approx_tokens, is_refusal, strip_think
from .base import CUT_OFF_NOTE, ROLE_OF_STEP, Outcome, Solver

CODING_SYSTEM = (
    "You fix bugs in Python code so that the failing tests pass. Reply with ONLY one ```python code block "
    "containing the complete corrected definitions of the functions you change. Do not repeat unchanged "
    "functions, do not include tests, do not explain. "
    "If the tests contradict each other or cannot all pass, reply IMPOSSIBLE."
)
_MAX_SOURCE_CHARS = 12000


def _closed_code_block(text: str) -> bool:
    """Есть хотя бы один закрытый блок ``` и нет незакрытого (как в base.ExprSolver).
    Для оборванного ответа это значит: обрезан текст после кода, сама заплатка целая."""
    fences = strip_think(text).count("```")
    return fences >= 2 and fences % 2 == 0


def _cut_off(reply) -> bool:
    """Ответ оборван по max_tokens внутри кода. Хвост незакрытого блока проходит ast.parse, если
    обрыв пришёлся на конец строки, но последняя функция в нём укорочена (потеряна ветка
    `return 0`): видимые тесты она может пройти, скрытые — нет. Такую заплатку не применяем."""
    return bool(reply.truncated) and not _closed_code_block(reply.text)


def _cut_off_limit(usual: int) -> int:
    """Лимит повтора после обрыва: ×3 от обычного, не выше 4000 (но и не меньше обычного)."""
    return max(usual, min(3 * usual, 4000))
_MARKER_SKIP = {
    "ONLY", "NOT", "DO", "TXT", "ALL", "MUST", "NOTE", "IMPORTANT", "EXACTLY", "AND", "OR", "THE", "IMPOSSIBLE",
    # служебные слова капсом рядом с answer.txt: кодировка, формат, регистр, каталог, запасной ответ
    "CWD", "UTF", "UTF8", "JSON", "ASCII", "LF", "CRLF", "CR", "EOL", "EOF", "NEWLINE", "UPPERCASE", "LOWERCASE",
    "CAPS", "ELSE", "FAILED", "FAIL", "FAILURE", "ERROR", "TODO", "README", "API", "CLI", "BASH", "PYTHON", "IF",
    "THEN", "WHEN", "NO", "FILE", "WORD", "SINGLE", "TEXT", "PLAIN", "ANSWER", "CURRENT", "DIRECTORY",
}
_MARKER_SKIP_LOWER = {
    "the", "a", "an", "it", "this", "that", "your", "its", "only", "just", "exactly", "out", "down", "them",
    "output", "result", "results", "answer", "something", "anything", "nothing", "everything", "word", "text",
}


def answer_marker(instruction: str, sentinel: Optional[str]) -> str:
    """Слово для answer.txt. Обычная форма («write ONLY the word X») разобрана в tasks.py и главнее.
    Иначе — единственный кандидат во фразах про answer.txt (слово капсом, слово в кавычках, «the word X»,
    «echo X >»). Кандидатов нет или их несколько — DONE, как в примерах: новое слово берём, только когда
    оно однозначно."""
    if sentinel:
        return sentinel
    text = re.sub(r"\((?:for example|e\.g\.|for instance)[^)]*\)", " ", instruction or "", flags=re.IGNORECASE)
    sentences = [x.replace("answer.txt", " ") for x in re.split(r"(?<=[.!?;])\s+", text) if "answer.txt" in x]
    found: dict[str, str] = {}
    for x in sentences:
        cands = re.findall(r"\becho\s+[\"']?(\w+)[\"']?\s*>", x)
        # бэктики — код из условия (`pytest`, `total`): кандидат только слово капсом (`DONE`)
        x = re.sub(r"`([^`\n]*)`", lambda m: m.group(0) if re.fullmatch(r"[A-Z][A-Z_]+", m.group(1)) else " ", x)
        cands += re.findall(r"\bword\s+[\"'`]?(\w+)", x, re.IGNORECASE)
        # «write fixed (and nothing else) to» — слово строчными сразу после глагола
        cands += [w for w in re.findall(r"\b(?:write|put|save)\s+(\w+)\s+(?:\([^)]*\)\s+)?(?:to|in|into)\b",
                                        x, re.IGNORECASE) if w.lower() not in _MARKER_SKIP_LOWER]
        cands += re.findall(r"[\"'`](\w+)[\"'`]", x)
        cands += re.findall(r"\b[A-Z][A-Z_]+\b", x)
        for c in cands:
            if not c.isdigit() and c.upper() not in _MARKER_SKIP:
                found.setdefault(c.lower(), c)
    if len(found) != 1 or "done" in found:
        return "DONE"
    return next(iter(found.values()))


_MAX_TEST_CHARS = 4000


class CodingSolver(Solver):
    family = "coding"
    ladder = ("fast", "fast_alt", "strong", "react")

    def __init__(self, task, env):
        super().__init__(task, env)
        self.ws: Path = task.workspace
        self.argv: Optional[list[str]] = None
        self.tests: list[Path] = []
        self.sources: list[Path] = []
        self.test_bytes: dict[Path, bytes] = {}
        self.original: dict[Path, str] = {}
        self.current: dict[Path, str] = {}       # лучшее на данный момент состояние исходников
        self.report = None
        self.repair_tried = False
        self.reference_tried = False
        self.provisional = False                 # правка перебором записана, но ещё не сверена с моделью
        self.pre_mutation: Optional[tuple[dict, object]] = None   # (исходники, отчёт) до перебора мутаций
        self.partial: Optional[dict] = None      # частичная правка перебором: запасная основа, модели не показывается
        self.marker = answer_marker(task.instruction, task.sentinel)
        self.no_skills = False                   # навыки ищут тесты сами по шаблону имени — здесь им не место

    # ------------------------------------------------------------------ 0 токенов
    def prepare(self) -> None:
        from ..skills import testrun as TR

        self.prepared = True
        if not self.ws.is_dir():
            self.mark_static_infeasible(0.97, "the task has no workspace directory")
            return
        self.tests = TR.test_files(self.ws)
        self.sources = TR.source_files(self.ws)
        named = self._named_checks() if not self.tests and self.sources else None
        if named is not None:
            # Проверка названа в условии, но это не test_*.py: скрипт с assert (check.py) или doctest.
            # Не «нет тестов»: файл проверки защищаем как тест; навыки (перебор, эталоны) ищут тесты по
            # имени и «чинили» бы саму проверку, поэтому они выключены.
            self.tests = named
            self.sources = [p for p in self.sources if p not in named]
            self.no_skills = True
            self.note("named_checks", files=[self._rel(p) for p in named])
        if (not self.tests and named is None) or not self.sources:
            what = "no test files" if not self.tests else "no source files to fix"
            # Нет тестов — нечего ни запускать, ни чинить: задачу нельзя проверить, токены не тратим.
            # Нет исходников при живых тестах — агент ещё может написать модуль по тестам.
            self.mark_static_infeasible(0.97 if not self.tests else 0.6, what)
            self.ladder = ("react",)
            return
        self.argv = TR.find_test_command(self.task.instruction, self.ws)
        for path in self.tests:
            try:
                self.test_bytes[path] = path.read_bytes()
            except OSError:
                pass
        for path in self.sources:
            try:
                self.original[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        self.current = dict(self.original)
        self.report = TR.run_tests(self.ws, self.argv, self.cfg.test_timeout_s)
        if self.report.ok and self.report.total == 0:
            # Команда из условия завершилась успешно, не выполнив ни одного теста (в файле нет
            # unittest.main() или тесты — голые функции). Запускаем тесты так, чтобы они действительно шли.
            alt = TR.fallback_test_command(self.ws)
            if alt:
                again = TR.run_tests(self.ws, alt, self.cfg.test_timeout_s)
                if again.total > 0 or not again.ok:
                    self.argv, self.report = alt, again
                    self.note("tests_rerun", why="the command from the instruction ran no tests")
        self.note("tests", failed=list(self.report.failed), total=self.report.total, error=self.report.error[:120])
        if self.report.ok:
            self._accept("tests already pass")
            return
        try:
            conflicts = TR.unsatisfiable_tests(self.ws) or TR.contradictory_asserts(self.ws)
        except Exception:  # noqa: BLE001
            conflicts = []
        if conflicts:
            # Никакая реализация не пройдёт оба теста: токены тратить не на что.
            self.mark_static_infeasible(0.98, "tests cannot pass: " + conflicts[0][:200])
            self.status = "infeasible"
            return
        if self.cfg.fastpath:
            if not self._reference_repair():
                self._mutation_repair(allow_provisional=True)

    def _named_checks(self) -> Optional[list[Path]]:
        """Файлы проверки из команды условия (`python3 check.py`, `python3 -m doctest mylib.py`), если все есть.
        Для doctest — пустой список: проверка внутри исходника. None — команды нет или файла нет."""
        for m in re.finditer(r"`([^`\n]+)`", self.task.instruction or ""):
            try:
                parts = shlex.split(m.group(1))
            except ValueError:
                continue
            if not parts or not (parts[0] in ("python", "python3") or parts[0].startswith("python3.")):
                continue
            files = [self.ws / a for a in parts[1:] if a.endswith(".py")]
            if not files:
                continue                     # `python -c ...`, `python -m pytest` — ищем дальше
            if not all(p.is_file() for p in files):
                return None
            return [] if "doctest" in parts else files
        return None

    def _reference_repair(self) -> bool:
        """Сломанные функции с известными именами заменяются эталонами (0 токенов, устойчиво к скрытым тестам)."""
        if self.no_skills:
            return False
        if self.reference_tried or self.report is None or self.report.ok:
            return False
        self.reference_tried = True
        try:
            from ..skills.semantic import reference_repair

            self._write(self.current)
            result = reference_repair(self.ws, self.argv, self.report, test_timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            self.note("reference_repair", error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return False
        if not result.texts or result.report is None:
            if result.confirmed:
                self.note("reference_repair", replaced=[], confirmed=result.confirmed[:12])
            return False
        candidate = dict(self.current)
        candidate.update({self.ws / rel: text for rel, text in result.texts.items() if (self.ws / rel) in self.current})
        report = self._try(candidate)
        self.note("reference_repair", replaced=result.replaced[:12], confirmed=result.confirmed[:12], ok=report.ok,
                  still_failing=list(report.failed)[:8])
        if report.ok:
            self.current, self.report = candidate, report
            self._accept("reference implementations")
            return True
        if not report.error and not report.timed_out and (self.report.error or set(report.failed) < set(self.report.failed)):
            self.current, self.report = candidate, report
        else:
            self._write(self.current)
        return False

    def _accept(self, how: str) -> None:
        self.best = self.marker
        self.status = "verified"
        self.note("accept", how=how)

    def _mutation_repair(self, allow_provisional: bool = False) -> bool:
        """Бесплатный ремонт перебором мутаций от лучшего на данный момент состояния."""
        if self.repair_tried or self.no_skills:
            return False
        self.repair_tried = True
        try:
            from ..skills.repair import mutation_repair
        except Exception:  # noqa: BLE001 — модуль навыка недоступен
            return False
        self._write(self.current)
        self.pre_mutation = (dict(self.current), self.report)
        try:
            result = mutation_repair(self.ws, self.argv, time_budget=40.0, test_timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            self.note("repair_skill", error=str(exc)[:200])
            self._restore_tests()
            self._write(self.current)
            return False
        self._restore_tests()
        self.note("repair_skill", fixed=result.fixed, edits=result.edits[:6], tried=result.tried,
                  seconds=round(result.seconds, 1), ambiguous=result.ambiguous)
        if result.fixed:
            self.current = self._read_sources()
            self.report = result.report
            if allow_provisional and result.edits and self.cfg.code_crosscheck and self.cfg.base_url:
                # Тесты проходят, ответ уже можно записать. Но правка подобрана под видимые тесты:
                # одной дешёвой попыткой модели проверяем, не подгонка ли это (см. skills/semantic.py).
                self.provisional = True
                self.best = self.marker
                self.status = "answered"
                self.ladder = ("fast",)
                self.note("accept_provisional", how="mutation repair; waiting for the model cross-check")
            else:
                self._accept("mutation repair")
            return True
        self._write(self.current)
        self._keep_partial(result)
        return False

    def _keep_partial(self, result) -> None:
        """Перебор починил часть багов: берём это за основу, модели останется меньше работы."""
        from ..skills import testrun as TR

        partial = {self.ws / rel: text for rel, text in (result.patched_files or {}).items()
                   if (self.ws / rel) in self.original}
        if not partial or self.report is None:
            return
        candidate = dict(self.current)
        candidate.update(partial)
        report = self._try(candidate)
        better = not report.error and not report.timed_out and set(report.failed) < set(self.report.failed)
        if better and len(report.failed) < len(self.report.failed):
            # Правка подобрана под видимые тесты и может быть подгонкой, поэтому основой для модели её
            # не делаем: модель увидит исходный код и все упавшие тесты. Частичная правка пригодится,
            # только если версия модели сама тесты не пройдёт.
            self.partial = candidate
            self.note("repair_partial", still_failing=list(report.failed))
        self._write(self.current)

    # ------------------------------------------------------------------ файлы
    def _read_sources(self) -> dict[Path, str]:
        out = {}
        for path in self.sources:
            try:
                out[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                out[path] = self.current.get(path, "")
        return out

    def _write(self, texts: dict[Path, str]) -> None:
        for path, text in texts.items():
            try:
                if not path.exists() or path.read_text(encoding="utf-8", errors="replace") != text:
                    path.write_text(text, encoding="utf-8")
            except OSError:
                pass

    def _restore_tests(self) -> None:
        """Тесты менять нельзя: если что-то их задело, возвращаем исходное содержимое."""
        for path, data in self.test_bytes.items():
            try:
                if not path.exists() or path.read_bytes() != data:
                    path.write_bytes(data)
            except OSError:
                pass

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.ws))

    # ------------------------------------------------------------------ промпт
    def _sources_text(self, texts: Optional[dict] = None) -> str:
        parts = []
        for path in self.sources:
            text = (texts or self.current).get(path, "")
            if len(text) > _MAX_SOURCE_CHARS:
                text = text[:_MAX_SOURCE_CHARS] + "\n# ... (file truncated)\n"
            parts.append(f"# file: {self._rel(path)}\n{text.rstrip()}")
        return "\n\n".join(parts)

    def _tests_text(self) -> str:
        parts = []
        for path, data in self.test_bytes.items():
            text = data.decode("utf-8", errors="replace")
            if len(text) > _MAX_TEST_CHARS:
                text = text[:_MAX_TEST_CHARS] + "\n# ... (truncated)\n"
            parts.append(f"# file: {self._rel(path)} (read-only)\n{text.rstrip()}")
        return "\n\n".join(parts)

    def user_prompt(self, step: str) -> str:
        from ..skills.testrun import summarize_failures

        failures = summarize_failures(self.report, 1500) if self.report is not None else "(tests were not run)"
        text = f"{self._sources_text()}\n\nFailing tests:\n{failures}"
        with_tests = step != "fast" or (self.report is not None and (self.report.error or not self.report.failed))
        if with_tests:
            text += f"\n\nTests:\n{self._tests_text()}"
        if len(self.sources) > 1:
            text += "\n\nStart each code block with a line `# file: <name>`."
        return text

    def prompt_text(self, step: str) -> str:
        if step == "react":
            return self.task.instruction
        return CODING_SYSTEM + "\n" + self.user_prompt(step)

    def features(self) -> dict:
        return {"failing": max(1, len(self.report.failed) if self.report is not None else 1)}

    # ------------------------------------------------------------------ политика
    def absorb(self, step: str, outcome: Outcome) -> None:
        if outcome.kind == "infra":
            self.infra_errors += 1
            return
        self.done.append(step)
        if outcome.kind == "answer":
            if self.status != "verified":
                self._accept(step)
        elif outcome.kind == "refused":
            self.refusals += 1
            static = 1 if (self.p_infeasible_static or 0.0) >= 0.7 else 0
            if self.refusals + static >= 2:
                self.status = "infeasible"
        else:
            self.failed_steps.append(step)

    def next_step(self) -> Optional[str]:
        if self.status in ("verified", "infeasible", "gave_up"):
            return None
        remaining = [s for s in self.ladder if s not in self.done]
        if not self.cfg.enable_react:
            remaining = [s for s in remaining if s != "react"]
        return remaining[0] if remaining else None

    def last_resort(self) -> None:
        """Модель недоступна или не справилась, а перебор мутаций ещё не пробовали — пробуем."""
        if self.status == "pending" and self.report is not None and not self.report.ok and self.cfg.crosscheck:
            if not self._reference_repair():
                self._mutation_repair()

    def finalize(self) -> None:
        if self.status == "pending":
            self.status = "gave_up"
        if self.status != "verified":
            # оставляем лучшее состояние (не хуже исходного) и нетронутые тесты
            self._write(self.current)
            self._restore_tests()

    # ------------------------------------------------------------------ выполнение
    async def run_step(self, step: str) -> Outcome:
        if step == "react":
            from .react import run_react

            return await run_react(self, step)
        try:
            return await self._llm_fix(step)
        except TaskCapExceeded as exc:
            return Outcome("failed", note=str(exc))
        except LLMDead:
            raise
        except LLMError as exc:
            return Outcome("infra", note=str(exc))

    async def _llm_crosscheck(self, step: str) -> Outcome:
        """Правка перебором уже записана и проходит тесты. Модель чинит исходное состояние независимо:
        её версия, если проходит те же тесты, заменяет подобранную мутацию (модель понимает смысл функции
        по имени, перебор — нет). Любой сбой оставляет правку перебором."""
        from ..skills.testrun import summarize_failures

        done = Outcome("answer", answer=self.marker)
        base_texts, base_report = self.pre_mutation or (self.original, None)
        failures = summarize_failures(base_report, 1500) if base_report is not None else "(tests were not run)"
        prompt = f"{self._sources_text(base_texts)}\n\nFailing tests:\n{failures}"
        if len(self.sources) > 1:
            prompt += "\n\nStart each code block with a line `# file: <name>`."
        messages = [{"role": "system", "content": CODING_SYSTEM}, {"role": "user", "content": prompt}]
        self.env.estimator.remember_prediction(self.task.task_id, step, self.family, self.features())
        with self.env.hub.scope(self.task.task_id, self.family, step, cap=self.cfg.task_token_cap):
            reply = await self.env.hub.chat(messages, role="fast", max_tokens=self.cfg.max_tokens_code, temperature=0.0)
        self.note(step, model=reply.model, cost=round(reply.cost, 1), prompt_tokens=reply.prompt_tokens,
                  completion_tokens=reply.completion_tokens, reasoning_tokens=reply.reasoning_tokens,
                  finish=reply.finish_reason, crosscheck=True,
                  **({"reply": reply.text[:800]} if self.env.trace_text else {}))
        if is_refusal(reply.text):
            self.note("crosscheck", used="mutation", why="model refused")
            return done
        if _cut_off(reply):
            # правка перебором уже проходит тесты: обрубок не применяем и лимит ради сверки не поднимаем
            self.note("crosscheck", used="mutation", why="reply cut off by the token limit")
            return done
        files = {self._rel(p): text for p, text in base_texts.items()}
        patches = split_by_file(reply.text, files)
        candidate = dict(base_texts)
        applied = 0
        for path in self.sources:
            patch = patches.get(self._rel(path))
            if not patch:
                continue
            patched = apply_patch(base_texts.get(path, ""), patch)
            if patched is not None and patched != base_texts.get(path):
                candidate[path] = patched
                applied += 1
        if not applied:
            self.note("crosscheck", used="mutation", why="no applicable patch")
            return done
        report = await asyncio.to_thread(self._try_in_copy, candidate)
        if report is None or not report.ok:
            self.note("crosscheck", used="mutation", why="the model's version fails the tests")
            return done
        differs = None
        try:
            from ..skills.semantic import behavior_diff

            differs = await asyncio.to_thread(
                behavior_diff, self.ws, {self._rel(p): t for p, t in self.current.items()},
                {self._rel(p): t for p, t in candidate.items()})
        except Exception:  # noqa: BLE001
            pass
        if differs == []:
            self.note("crosscheck", used="mutation", agree=True)
            return done
        await asyncio.to_thread(self._write, candidate)
        self.current, self.report = candidate, report
        self.note("crosscheck", used="model", agree=False if differs else None, differs=(differs or [])[:8])
        return done

    def _try_in_copy(self, texts: dict[Path, str]):
        """Проверка кандидата в копии каталога: сам workspace не меняется (в нём лежит рабочая правка)."""
        import tempfile

        from ..skills import testrun as TR

        tmp = Path(tempfile.mkdtemp(prefix="xcheck_"))
        try:
            work = tmp / "ws"
            shutil.copytree(self.ws, work, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "answer.txt"))
            for path, text in texts.items():
                (work / path.relative_to(self.ws)).write_text(text, encoding="utf-8")
            for path, data in self.test_bytes.items():
                (work / path.relative_to(self.ws)).write_bytes(data)
            return TR.run_tests(work, self.argv, self.cfg.test_timeout_s)
        except OSError:
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def _llm_fix(self, step: str) -> Outcome:
        from ..skills import testrun as TR

        if self.provisional:
            return await self._llm_crosscheck(step)
        role = ROLE_OF_STEP.get(step, "fast")
        strong = role == "strong"
        messages = [{"role": "system", "content": CODING_SYSTEM},
                    {"role": "user", "content": self.user_prompt(step)}]
        usual = self.cfg.max_tokens_strong if strong else self.cfg.max_tokens_code
        temperature = 0.0 if step == "fast" else 0.3
        reply = await self._ask_code(step, messages, role, usual, temperature)
        if not is_refusal(reply.text) and _cut_off(reply):
            # Ответ оборван по max_tokens внутри кода: с тем же лимитом повтор оборвётся снова.
            # Повтор с нуля (обрывок в истории только съел бы лимит) и с лимитом ×3, но не выше 4000;
            # если на увеличенный вызов не хватает бюджета — с обычным лимитом (как в base.ExprSolver).
            self.note(step, cut_off=True)
            messages = [messages[0],
                        {"role": "user", "content": messages[1]["content"] + "\n\n" + CUT_OFF_NOTE}]
            try:
                try:
                    reply = await self._ask_code(step, messages, role, _cut_off_limit(usual), temperature)
                except BudgetShort:
                    reply = await self._ask_code(step, messages, role, usual, temperature)
            except TaskCapExceeded as exc:
                if await self._free_repair(step):
                    return Outcome("answer", answer=self.marker)
                return Outcome("failed", note=str(exc))
            if not is_refusal(reply.text) and _cut_off(reply):
                self.note(step, cut_off=True)
                if await self._free_repair(step):
                    return Outcome("answer", answer=self.marker)
                return Outcome("failed", note="the reply was cut off by the token limit twice")
        if is_refusal(reply.text):
            if await self._free_repair(step):
                return Outcome("answer", answer=self.marker)
            return Outcome("refused", note="model says the tests cannot all pass")

        files = {self._rel(p): text for p, text in self.current.items()}
        patches = split_by_file(reply.text, files)
        candidate = dict(self.current)
        applied = 0
        for path in self.sources:
            patch = patches.get(self._rel(path))
            if not patch:
                continue
            patched = apply_patch(self.current.get(path, ""), patch)
            if patched is not None and patched != self.current.get(path):
                candidate[path] = patched
                applied += 1
        outcome = Outcome("failed", note="the reply contained no applicable fix")
        if applied:
            report = await asyncio.to_thread(self._try, candidate)
            if report.ok:
                self.current, self.report = candidate, report
                return Outcome("answer", answer=self.marker)
            if self.partial is not None:
                # правка модели поверх частичной правки перебором: вместе они могут закрыть все тесты
                combined = dict(self.partial)
                for path in self.sources:
                    patch = patches.get(self._rel(path))
                    patched = apply_patch(combined.get(path, ""), patch) if patch else None
                    if patched is not None:
                        combined[path] = patched
                both = await asyncio.to_thread(self._try, combined)
                if both.ok:
                    self.current, self.report = combined, both
                    self.note("combined", how="model patch on top of the partial mutation repair")
                    return Outcome("answer", answer=self.marker)
            improved = (self.report is not None and not report.error and not report.timed_out
                        and set(report.failed) < set(self.report.failed))
            if improved:
                self.current, self.report = candidate, report   # частичный успех сохраняем
            else:
                await asyncio.to_thread(self._write, self.current)
            outcome = Outcome("failed", note=f"tests still fail: {', '.join(report.failed[:5]) or report.error[:80]}")
        if await self._free_repair(step):
            return Outcome("answer", answer=self.marker)
        return outcome

    async def _ask_code(self, step: str, messages: list[dict], role: str, max_tokens: int, temperature: float):
        self.env.estimator.remember_prediction(self.task.task_id, step, self.family, self.features())
        with self.env.hub.scope(self.task.task_id, self.family, step, cap=self.cfg.task_token_cap):
            reply = await self.env.hub.chat(messages, role=role, max_tokens=max_tokens, temperature=temperature)
        self.note(step, model=reply.model, cost=round(reply.cost, 1), prompt_tokens=reply.prompt_tokens,
                  completion_tokens=reply.completion_tokens, reasoning_tokens=reply.reasoning_tokens,
                  finish=reply.finish_reason, **({"reply": reply.text[:800]} if self.env.trace_text else {}))
        return reply

    async def _free_repair(self, step: str) -> bool:
        """Бесплатная страховка после неудачи модели: если баг — мутация одного оператора,
        перебор найдёт его без токенов (при fastpath он уже был сделан на подготовке)."""
        if not self.cfg.crosscheck:
            return False
        if not self.reference_tried and await asyncio.to_thread(self._reference_repair):
            return True
        if self.repair_tried:
            return False
        return await asyncio.to_thread(self._mutation_repair)

    def _try(self, texts: dict[Path, str]):
        from ..skills import testrun as TR

        self._write(texts)
        report = TR.run_tests(self.ws, self.argv, self.cfg.test_timeout_s)
        self._restore_tests()
        return report

    async def after_react(self, work: Path, answer: str) -> Optional[str]:
        """Агент чинил код в копии каталога: проверяем там же, с исходными тестами, и переносим исправление."""
        from ..skills import testrun as TR

        def sync_unverified() -> bool:
            # Тестов нет — проверить нечем: переносим то, что сделал агент, как есть.
            for src in sorted(p for p in work.rglob("*") if p.is_file()):
                rel = src.relative_to(work)
                if rel.name == "answer.txt" or any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                    continue
                try:
                    (self.ws / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, self.ws / rel)
                except OSError:
                    return False
            return True

        if not self.test_bytes:
            ok = await asyncio.to_thread(sync_unverified)
            shutil.rmtree(work, ignore_errors=True)
            return answer if ok else None

        def check() -> bool:
            for path, data in self.test_bytes.items():
                target = work / path.relative_to(self.ws)
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                except OSError:
                    return False
            report = TR.run_tests(work, self.argv, self.cfg.test_timeout_s)
            if not report.ok:
                return False
            fixed = {}
            for src in TR.source_files(work):
                rel = src.relative_to(work)
                if (self.ws / rel) in self.test_bytes:
                    continue                 # файл проверки с необычным именем: правку агента не переносим
                try:
                    fixed[self.ws / rel] = src.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return False
            for path, text in fixed.items():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
            self._write(fixed)
            final = TR.run_tests(self.ws, self.argv, self.cfg.test_timeout_s)
            self._restore_tests()
            if final.ok:
                self.current.update({p: t for p, t in fixed.items()})
                for path in fixed:
                    if path not in self.sources:
                        self.sources.append(path)
                self.report = final
                return True
            self._write(self.current)
            return False

        ok = await asyncio.to_thread(check)
        shutil.rmtree(work, ignore_errors=True)
        return self.marker if ok else None
