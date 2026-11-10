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
import shutil
from pathlib import Path
from typing import Optional

from ..llm import LLMDead, LLMError, TaskCapExceeded
from ..patching import apply_patch, split_by_file
from ..textutil import approx_tokens, is_refusal
from .base import ROLE_OF_STEP, Outcome, Solver

CODING_SYSTEM = (
    "You fix bugs in Python code so that the failing tests pass. Reply with ONLY one ```python code block "
    "containing the complete corrected definitions of the functions you change. Do not repeat unchanged "
    "functions, do not include tests, do not explain. "
    "If the tests contradict each other or cannot all pass, reply IMPOSSIBLE."
)
_MAX_SOURCE_CHARS = 12000
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

    # ------------------------------------------------------------------ 0 токенов
    def prepare(self) -> None:
        from ..skills import testrun as TR

        self.prepared = True
        if not self.ws.is_dir():
            self.mark_static_infeasible(0.97, "the task has no workspace directory")
            return
        self.tests = TR.test_files(self.ws)
        self.sources = TR.source_files(self.ws)
        if not self.tests or not self.sources:
            what = "no test files" if not self.tests else "no source files to fix"
            self.mark_static_infeasible(0.6, what)
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
        self.note("tests", failed=list(self.report.failed), total=self.report.total, error=self.report.error[:120])
        if self.report.ok:
            self._accept("tests already pass")
            return
        try:
            conflicts = TR.contradictory_asserts(self.ws)
        except Exception:  # noqa: BLE001
            conflicts = []
        if conflicts:
            # Никакая реализация не пройдёт оба теста: токены тратить не на что.
            self.mark_static_infeasible(0.98, "contradictory tests: " + conflicts[0][:200])
            self.status = "infeasible"
            return
        if self.cfg.fastpath:
            self._mutation_repair()

    def _accept(self, how: str) -> None:
        self.best = self.task.sentinel or "DONE"
        self.status = "verified"
        self.note("accept", how=how)

    def _mutation_repair(self) -> bool:
        """Бесплатный ремонт перебором мутаций от исходного состояния."""
        if self.repair_tried:
            return False
        self.repair_tried = True
        try:
            from ..skills.repair import mutation_repair
        except Exception:  # noqa: BLE001 — модуль навыка недоступен
            return False
        self._write(self.original)
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
        candidate = dict(self.original)
        candidate.update(partial)
        report = self._try(candidate)
        better = not report.error and not report.timed_out and set(report.failed) < set(self.report.failed)
        if better and len(report.failed) < len(self.report.failed):
            self.current, self.report = candidate, report
            self.note("repair_partial", still_failing=list(report.failed))
        else:
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
    def _sources_text(self) -> str:
        parts = []
        for path in self.sources:
            text = self.current.get(path, "")
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

    async def _llm_fix(self, step: str) -> Outcome:
        from ..skills import testrun as TR

        role = ROLE_OF_STEP.get(step, "fast")
        strong = role == "strong"
        messages = [{"role": "system", "content": CODING_SYSTEM},
                    {"role": "user", "content": self.user_prompt(step)}]
        self.env.estimator.remember_prediction(self.task.task_id, step, self.family, self.features())
        with self.env.hub.scope(self.task.task_id, self.family, step, cap=self.cfg.task_token_cap):
            reply = await self.env.hub.chat(
                messages, role=role,
                max_tokens=self.cfg.max_tokens_strong if strong else self.cfg.max_tokens_code,
                temperature=0.0 if step == "fast" else 0.3)
        self.note(step, model=reply.model, cost=round(reply.cost, 1), prompt_tokens=reply.prompt_tokens,
                  completion_tokens=reply.completion_tokens, reasoning_tokens=reply.reasoning_tokens,
                  finish=reply.finish_reason, **({"reply": reply.text[:800]} if self.env.trace_text else {}))
        if is_refusal(reply.text):
            if await self._free_repair(step):
                return Outcome("answer", answer=self.task.sentinel or "DONE")
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
                return Outcome("answer", answer=self.task.sentinel or "DONE")
            improved = (self.report is not None and not report.error and not report.timed_out
                        and set(report.failed) < set(self.report.failed))
            if improved:
                self.current, self.report = candidate, report   # частичный успех сохраняем
            else:
                await asyncio.to_thread(self._write, self.current)
            outcome = Outcome("failed", note=f"tests still fail: {', '.join(report.failed[:5]) or report.error[:80]}")
        if await self._free_repair(step):
            return Outcome("answer", answer=self.task.sentinel or "DONE")
        return outcome

    async def _free_repair(self, step: str) -> bool:
        """Бесплатная страховка после неудачи модели: если баг — мутация одного оператора,
        перебор найдёт его без токенов (при fastpath он уже был сделан на подготовке)."""
        if self.repair_tried or not self.cfg.crosscheck:
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
        return (self.task.sentinel or "DONE") if ok else None
