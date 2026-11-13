"""Планировщик: какой шаг какой задачи делать следующим.

Бюджет токенов харнессу не сообщают — прокси просто перестаёт отвечать, когда он кончится.
Поэтому порядок работы должен быть хорош при ЛЮБОМ бюджете: в каждый момент выполняется шаг
с наибольшей ожидаемой отдачей на токен

    плотность = шанс_успеха × баллы_задачи / ожидаемая_стоимость_шага

Отсюда само собой получается то, что нужно по регламенту:
  * сначала всё бесплатное (подготовка, навыки, «тесты уже проходят»);
  * затем дешёвые первые попытки — дорогие по баллам задачи раньше;
  * эскалации (вторая модель, модель с рассуждениями, ReAct) — только после того, как все
    дешёвые попытки сделаны, и только если шанс на успех оправдывает цену;
  * задачи, похожие на невыполнимые, уходят в конец очереди или не запускаются вовсе.

Ответ пишется в answer.txt сразу, как только он появился или изменился: обрыв прогона
в любой момент не теряет уже заработанные баллы.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .answers import write_answer
from .config import Config
from .estimator import Estimator
from .llm import LLMDead, LLMHub
from .solvers.base import Env, Outcome, Solver, StepEstimate
from .solvers.families import solver_class
from .tasks import Task

EXPENSIVE_STEPS = ("strong", "react")
WAIT = "wait for running calls"


@dataclass
class Pick:
    solver: Solver
    step: str
    estimate: StepEstimate
    density: float


class Scheduler:
    def __init__(self, cfg: Config, tasks: list[Task], hub: LLMHub, estimator: Estimator, scratch_root: Path,
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self.hub = hub
        self.ledger = hub.ledger
        self.estimator = estimator
        self.log = log
        self.env = Env(cfg=cfg, hub=hub, estimator=estimator, scratch_root=scratch_root, trace_text=cfg.trace_text)
        self.solvers: list[Solver] = [solver_class(task.task_type)(task, self.env) for task in tasks]
        self.started = time.monotonic()
        self.deadline = self.started + cfg.time_limit_s
        self.written: dict[str, str] = {}
        self.write_failed: set[str] = set()
        self.running: set[str] = set()
        self.closed: set[str] = set()      # задачи, по которым решено больше не тратить токены
        self.stop_reason = ""
        self._cond: Optional[asyncio.Condition] = None
        self.on_progress: Optional[Callable[[], None]] = None
        self.tight = False                 # бюджет мал для числа задач: работать фазами

    # ------------------------------------------------------------------ вспомогательное
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _say(self, solver: Solver, text: str) -> None:
        self.log(f"[{self.elapsed():7.1f}s] {solver.task.task_id:<22} {text}")

    def _sync_answer(self, solver: Solver) -> None:
        """Записывает ответ на диск, если он появился или изменился."""
        if solver.best is None or solver.status not in ("answered", "verified"):
            return
        task_id = solver.task.task_id
        if self.written.get(task_id) == solver.best:
            return
        if write_answer(solver.task, solver.best, also_workspace=self.cfg.answer_in_workspace):
            self.written[task_id] = solver.best
            self.write_failed.discard(task_id)
        elif task_id not in self.write_failed:
            self.write_failed.add(task_id)
            self._say(solver, "ERROR: cannot write answer.txt (permission denied?)")

    def _close(self, solver: Solver, reason: str = "") -> None:
        task_id = solver.task.task_id
        if task_id in self.closed:
            return
        self.closed.add(task_id)
        try:
            solver.finalize()
        except Exception as exc:  # noqa: BLE001
            solver.note("finalize_error", error=str(exc)[:200])
        if reason:
            solver.note("closed", reason=reason)
        self._sync_answer(solver)

    # ------------------------------------------------------------------ подготовка (0 токенов)
    async def _prepare_all(self) -> None:
        limit = asyncio.Semaphore(max(2, min(8, (os.cpu_count() or 4))))

        async def prepare(solver: Solver) -> None:
            async with limit:
                try:
                    await asyncio.wait_for(asyncio.to_thread(solver.prepare), timeout=180)
                except Exception as exc:  # noqa: BLE001 — ошибка разбора одной задачи не роняет прогон
                    solver.prepared = True
                    solver.note("prepare_error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
            self._sync_answer(solver)
            if solver.status == "verified":
                self._say(solver, f"0 tokens -> {solver.best!r} (verified)")
            elif solver.status == "infeasible":
                self._say(solver, f"0 tokens -> infeasible: {solver.static_reason[:90]}")

        ordered = sorted(self.solvers, key=lambda s: -s.task.reward)
        await asyncio.gather(*(prepare(s) for s in ordered))

    # ------------------------------------------------------------------ выбор шага
    def _gate(self, solver: Solver, step: str, est: StepEstimate) -> str:
        """Пустая строка — шаг разрешён; иначе причина отказа."""
        now = time.monotonic()
        margin = 120.0 if step in EXPENSIVE_STEPS else 30.0
        if now > self.deadline - margin:
            return "time limit"
        paid_before = any(s != "skill" for s in solver.done)
        if not paid_before and (solver.p_infeasible_static or 0.0) >= 0.95:
            return "statically infeasible"
        if step in EXPENSIVE_STEPS and est.p_success < self.cfg.min_p_escalate:
            return f"chance {est.p_success:.2f} is below the escalation threshold"
        if est.p_success * solver.task.reward * self.cfg.tokens_per_point < est.cost:
            return f"expected gain is not worth ~{est.cost:.0f} tokens (chance {est.p_success:.2f})"
        if self.ledger.task_spent(solver.task.task_id) + est.cost > self.cfg.task_token_cap:
            return "per-task token cap"
        # Бюджет из BUDGET, а если его не передали — ориентир организаторов (100 000).
        budget = self.cfg.effective_budget()
        remaining = budget - self.ledger.spent
        need = self.estimator.safe_cost(est.cost)
        if need > remaining - budget * self.cfg.budget_reserve:
            return "does not fit into the remaining budget"
        if self.ledger.has_reservations() and need > remaining - self.ledger.reserved - budget * self.cfg.budget_reserve:
            return WAIT   # не влезает только из-за резерва запущенных вызовов: подождать их
        # Дорогие ступени — только при большом запасе: при тесном бюджете их токены нужнее
        # первым попыткам других задач.
        if step == "strong" and remaining < 5 * max(est.cost, 1000.0):
            return "too little budget left for a reasoning model"
        if step == "react" and remaining < 10 * self.cfg.react_token_cap:
            return "too little budget left for the agent"
        return ""

    def _pick(self) -> Optional[Pick]:
        best: Optional[Pick] = None
        best_key: tuple = ()
        for solver in self.solvers:
            task_id = solver.task.task_id
            if task_id in self.running or task_id in self.closed:
                continue
            step = solver.next_step()
            if step is None:
                self._close(solver)
                continue
            if solver.infra_errors > 2:
                self._close(solver, "model calls keep failing")
                continue
            try:
                est = solver.estimate(step)
            except Exception as exc:  # noqa: BLE001
                solver.note("estimate_error", error=str(exc)[:200])
                est = StepEstimate(cost=200.0, p_success=0.5)
            refusal = self._gate(solver, step, est)
            if refusal == WAIT:
                continue
            if refusal:
                self._say(solver, f"skip {step}: {refusal}")
                if refusal == "statically infeasible" and solver.status == "pending" and solver.best is None:
                    solver.status = "infeasible"   # отсеяна проверкой без модели: answer.txt не создаём
                self._close(solver, f"{step}: {refusal}")
                continue
            density = est.p_success * solver.task.reward / max(est.cost, 1.0)
            # задачи, где вызовы уже срывались, чуть отодвигаем, чтобы не крутиться на одной
            density /= 1 + solver.infra_errors
            # При тесном бюджете — фазами: сначала по одной платной попытке каждой задаче,
            # потом перепроверки и эскалации.
            first = 1 if self.tight and not any(s != "skill" for s in solver.done) else 0
            key = (first, density)
            if best is None or key > best_key:
                best, best_key = Pick(solver, step, est, density), key
        return best

    # ------------------------------------------------------------------ выполнение
    def _step_timeout(self, step: str) -> float:
        cfg = self.cfg
        if step == "react":
            base = cfg.call_timeout_fast * 3 + 120
        elif step == "strong":
            base = cfg.call_timeout_strong * 2 + cfg.test_timeout_s + 30
        else:
            base = cfg.call_timeout_fast * 2 + cfg.test_timeout_s + 60
        return max(5.0, min(base, self.deadline - time.monotonic()))

    async def _run_one(self, pick: Pick) -> None:
        solver, step = pick.solver, pick.step
        spent_before = self.ledger.task_spent(solver.task.task_id)
        started = time.monotonic()
        try:
            outcome = await asyncio.wait_for(solver.run_step(step), timeout=self._step_timeout(step))
        except LLMDead as exc:
            self.stop_reason = self.stop_reason or f"model proxy is unavailable: {exc}"
            outcome = Outcome("infra", note=str(exc))
        except asyncio.TimeoutError:
            outcome = Outcome("failed", note="step timed out")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — сбой одной задачи не роняет прогон
            outcome = Outcome("failed", note=f"{type(exc).__name__}: {str(exc)[:200]}")
        cost = self.ledger.task_spent(solver.task.task_id) - spent_before
        solver.absorb(step, outcome)
        if outcome.kind != "infra":
            self.estimator.observe_outcome(solver.family, step, outcome.kind == "answer")
        solver.note("outcome", of=step, kind=outcome.kind, note=outcome.note[:200], cost=round(cost, 1),
                    predicted=round(pick.estimate.cost, 1), p=round(pick.estimate.p_success, 3),
                    seconds=round(time.monotonic() - started, 2), status=solver.status)
        self._sync_answer(solver)
        shown = f"-> {outcome.answer!r}" if outcome.kind == "answer" else f"-> {outcome.kind}: {outcome.note[:70]}"
        self._say(solver, f"{step:<10} {shown} [{solver.status}] cost={cost:.0f} (est {pick.estimate.cost:.0f}) "
                          f"total={self.ledger.spent:.0f}")
        if self.on_progress:
            try:
                self.on_progress()
            except Exception:  # noqa: BLE001
                pass

    async def _worker(self) -> None:
        assert self._cond is not None
        while True:
            async with self._cond:
                while True:
                    if self.stop_reason or self.hub.dead or time.monotonic() > self.deadline:
                        return
                    pick = self._pick()
                    if pick is not None:
                        self.running.add(pick.solver.task.task_id)
                        break
                    if not self.running:
                        self._cond.notify_all()
                        return
                    await self._cond.wait()
            try:
                await self._run_one(pick)
            finally:
                async with self._cond:
                    self.running.discard(pick.solver.task.task_id)
                    self._cond.notify_all()

    async def run(self) -> None:
        self._cond = asyncio.Condition()
        await self._prepare_all()
        free = sum(1 for s in self.solvers if s.status == "verified")
        open_tasks = sum(1 for s in self.solvers if s.status == "pending")
        self.tight = self.cfg.effective_budget() < 100.0 * open_tasks
        self.log(f"[{self.elapsed():7.1f}s] prepared {len(self.solvers)} tasks; solved without the model: {free}; "
                 f"flagged infeasible: {sum(1 for s in self.solvers if s.status == 'infeasible')}")
        if not self.cfg.base_url:
            self.hub.dead, self.hub.dead_reason = True, "LLM_BASE_URL is not set"
        workers = [asyncio.create_task(self._worker()) for _ in range(self.cfg.max_parallel)]
        try:
            remaining = max(1.0, self.deadline - time.monotonic())
            done, pending = await asyncio.wait(workers, timeout=remaining)
            for task in pending:
                task.cancel()
            if pending:
                self.stop_reason = self.stop_reason or "time limit"
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.exception() is not None:
                    self.log(f"worker error: {task.exception()!r}")
        finally:
            if self.hub.dead and not self.stop_reason:
                self.stop_reason = self.hub.dead_reason
            await self._last_resorts()
            for solver in self.solvers:
                self._close(solver, self.stop_reason and f"run stopped: {self.stop_reason}")

    async def _last_resorts(self) -> None:
        """Бесплатные последние попытки для задач, оставшихся без ответа (пока есть время)."""
        pending = [s for s in self.solvers if s.status == "pending" and s.task.task_id not in self.closed]
        if not pending or time.monotonic() > self.deadline - 20:
            return
        limit = asyncio.Semaphore(max(2, min(8, (os.cpu_count() or 4))))

        async def attempt(solver: Solver) -> None:
            async with limit:
                try:
                    await asyncio.to_thread(solver.last_resort)
                except Exception as exc:  # noqa: BLE001
                    solver.note("last_resort_error", error=str(exc)[:200])
            if solver.status == "verified":
                self._sync_answer(solver)
                self._say(solver, f"last resort -> {solver.best!r} (verified)")

        budget = max(1.0, min(120.0, self.deadline - time.monotonic() - 10))
        try:
            await asyncio.wait_for(asyncio.gather(*(attempt(s) for s in pending)), timeout=budget)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # ------------------------------------------------------------------ отчёт
    def report(self) -> dict:
        tasks = []
        counts: dict[str, int] = {}
        for solver in self.solvers:
            counts[solver.status] = counts.get(solver.status, 0) + 1
            task_id = solver.task.task_id
            tasks.append({
                "task_id": task_id,
                "type": solver.task.task_type,
                "family": solver.family,
                "reward": solver.task.reward,
                "status": solver.status,
                "answered": task_id in self.written,
                "spent": round(self.ledger.task_spent(task_id), 1),
                "steps": list(solver.done),
                "votes": sorted(solver.votes),
                "refusals": solver.refusals,
                "static": solver.static_reason,
                "p_infeasible_static": solver.p_infeasible_static,
                "trace": solver.trace,
            })
        return {
            "summary": {
                "tasks": len(self.solvers),
                "answered": len(self.written),
                "by_status": counts,
                "spent": round(self.ledger.spent, 1),
                "seconds": round(self.elapsed(), 1),
                "stop_reason": self.stop_reason,
                "fastpath": self.cfg.fastpath,
                "crosscheck": self.cfg.crosscheck,
            },
            "ledger": self.ledger.summary(),
            "estimator": self.estimator.snapshot(),
            "tasks": tasks,
            "calls": self.ledger.dump(),
        }
