"""Ответ модели, оборванный по max_tokens: обрубок кода не исполняется, повтор идёт с бОльшим лимитом."""
from __future__ import annotations

import asyncio
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.llm import BudgetShort
from harness.solvers.families import MathSolver
from harness.tasks import Task

QUESTION = "A shop sells 12 pens at 3.5 each. With 8% tax, what is the total price?"


def _reply(text: str, truncated: bool):
    return SimpleNamespace(text=text, model="m", cost=1.0, prompt_tokens=10, completion_tokens=5,
                           reasoning_tokens=0, finish_reason="length" if truncated else "stop",
                           truncated=truncated)


class _Hub:
    """Вызов модели по сценарию; запоминает max_tokens каждого вызова."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[int] = []
        self.sent: list[list[dict]] = []
        self.short_above: int | None = None   # BudgetShort для вызовов с max_tokens больше этого

    @contextlib.contextmanager
    def scope(self, *args, **kwargs):
        yield

    async def chat(self, messages, *, role="fast", max_tokens=None, temperature=0.0):
        if self.short_above is not None and max_tokens > self.short_above:
            raise BudgetShort("only ~500 tokens left in the budget")
        self.calls.append(max_tokens)
        self.sent.append([dict(m) for m in messages])
        return self.script.pop(0)


class TruncatedReplyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def solver(self, script: list) -> tuple[MathSolver, _Hub]:
        ws = self.root / "workspace"
        ws.mkdir(exist_ok=True)
        task = Task(task_id="t1", task_type="math", instruction=QUESTION, reward=1.0, dir=self.root,
                    workspace=ws, question=QUESTION)
        cfg = SimpleNamespace(fastpath=True, crosscheck=False, exec_timeout_s=10.0, task_token_cap=6000.0,
                              max_tokens_fast=400, max_tokens_strong=6000)
        hub = _Hub(script)
        estimator = SimpleNamespace(remember_prediction=lambda *a, **k: None)
        env = SimpleNamespace(cfg=cfg, hub=hub, estimator=estimator, scratch_root=self.root / "scratch",
                              trace_text=False)
        solver = MathSolver(task, env)
        solver.prepare()
        return solver, hub

    def interpret(self, text: str, truncated: bool = True):
        solver, _ = self.solver([])
        return asyncio.run(solver._interpret("fast", _reply(text, truncated)))

    def test_unclosed_fence_is_not_executed(self):
        # дальше было бы «* 1.08»: исполнимый префикс дал бы промежуточные 42
        outcome, error = self.interpret("```python\nprice = 12*3.5\nprice")
        self.assertIsNone(outcome)
        self.assertIn("cut off by the token limit", error)

    def test_unclosed_fence_without_print(self):
        # без print обёртка берёт result из globals: обрубок дал бы промежуточные 1752
        outcome, error = self.interpret("```python\nresult = 1752\nresult = result // 2\nresult = result * 2\n")
        self.assertIsNone(outcome)
        self.assertIn("cut off", error)

    def test_unfenced_code_and_short_value_go_to_retry(self):
        for text in ("price = 12*3.5\nprice", "12*3.5 *", "42"):
            outcome, error = self.interpret(text)
            self.assertIsNone(outcome, text)
            self.assertIn("cut off", error)

    def test_closed_block_with_cut_explanation_is_accepted(self):
        outcome, error = self.interpret("```python\nprint(round(12*3.5*1.08, 2))\n```\nHere the tax is appl")
        self.assertEqual(error, "")
        self.assertEqual(outcome.kind, "answer")
        self.assertEqual(outcome.answer, "45.36")

    def test_not_truncated_reply_unchanged(self):
        outcome, _ = self.interpret("```python\nprice = 12*3.5\nprice", truncated=False)
        self.assertEqual(outcome.answer, "42")
        outcome, _ = self.interpret("45.36", truncated=False)
        self.assertEqual(outcome.answer, "45.36")

    def test_retry_after_cut_gets_bigger_max_tokens_and_fresh_history(self):
        solver, hub = self.solver([_reply("```python\nprice = 12*3.5\nprice", True),
                                   _reply("```python\nprint(round(12*3.5*1.08, 2))\n```", False)])
        outcome = asyncio.run(solver.run_step("fast"))
        self.assertEqual(outcome.answer, "45.36")
        base = solver._step_max_tokens("fast")
        self.assertEqual(hub.calls, [base, min(3 * base, 4000)])
        self.assertGreater(hub.calls[1], hub.calls[0])
        # повтор с нуля: обрывок в историю не вклеивается, замечание — в том же сообщении пользователя
        retry = hub.sent[1]
        self.assertEqual([m["role"] for m in retry], ["system", "user"])
        self.assertNotIn("price = 12*3.5", retry[1]["content"])
        self.assertIn("cut off by the token limit", retry[1]["content"])

    def test_cut_off_limit_is_capped(self):
        solver, _ = self.solver([])
        solver.max_tokens_fast = 2000
        self.assertEqual(solver._cut_off_max_tokens("fast"), 4000)
        self.assertEqual(solver._cut_off_max_tokens("strong"), 6000)   # не меньше обычного

    def test_ordinary_retry_keeps_max_tokens(self):
        solver, hub = self.solver([_reply("this is not code at all, sorry\nsecond line", False),
                                   _reply("```python\nprint(round(12*3.5*1.08, 2))\n```", False)])
        asyncio.run(solver.run_step("fast"))
        base = solver._step_max_tokens("fast")
        self.assertEqual(hub.calls, [base, base])

    def test_budget_short_for_bigger_call_falls_back_to_usual_limit(self):
        solver, hub = self.solver([_reply("```python\nprice = 12*3.5\nprice", True),
                                   _reply("```python\nprint(round(12*3.5*1.08, 2))\n```", False)])
        base = solver._step_max_tokens("fast")
        hub.short_above = base
        outcome = asyncio.run(solver.run_step("fast"))
        self.assertEqual(outcome.answer, "45.36")
        self.assertEqual(hub.calls, [base, base])

    def test_second_cut_fails_instead_of_partial_answer(self):
        solver, hub = self.solver([_reply("```python\nprice = 12*3.5\nprice", True),
                                   _reply("```python\nprice = 12*3.5\nprice", True)])
        outcome = asyncio.run(solver.run_step("fast"))
        self.assertEqual(outcome.kind, "failed")
        self.assertIn("cut off", outcome.note)


if __name__ == "__main__":
    unittest.main()
