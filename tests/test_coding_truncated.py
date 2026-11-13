"""coding: ответ модели, оборванный по max_tokens внутри кода, не применяется; повтор идёт с бОльшим лимитом."""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.config import load_config
from harness.llm import BudgetShort
from harness.solvers.base import CUT_OFF_NOTE
from harness.solvers.coding import CodingSolver, _closed_code_block, _cut_off_limit
from harness.tasks import Task

LIB_BAD = ('def classify(n):\n    if n < 0:\n        return "neg"\n    if n > 0:\n        return "neg"\n'
           '    return "zero"\n')
TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
         "    def test_neg(self):\n        self.assertEqual(m.classify(-1), 'neg')\n\n"
         "    def test_pos(self):\n        self.assertEqual(m.classify(2), 'pos')\n\n\n"
         "if __name__ == '__main__':\n    unittest.main()\n")
GOOD = ('```python\ndef classify(n):\n    if n < 0:\n        return "neg"\n    if n > 0:\n        return "pos"\n'
        '    return "zero"\n```')
# обрыв на конце строки: синтаксически целый обрубок без `return "zero"` проходит видимые тесты
STUB = '```python\ndef classify(n):\n    if n < 0:\n        return "neg"\n    if n > 0:\n        return "pos"\n'
INSTRUCTION = "Run `python3 test_mylib.py`; some tests fail. Fix mylib.py, then write DONE to answer.txt."


def _reply(text: str, truncated: bool):
    return SimpleNamespace(text=text, model="m", cost=1.0, prompt_tokens=10, completion_tokens=5,
                           reasoning_tokens=0, finish_reason="length" if truncated else "stop",
                           truncated=truncated)


class _Hub:
    """Вызов модели по сценарию; запоминает max_tokens и сообщения каждого вызова."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[int] = []
        self.sent: list[list[dict]] = []
        self.short_above: int | None = None   # BudgetShort для вызовов с max_tokens больше этого
        self.free_calls = 0                   # столько первых вызовов проходит без проверки остатка

    @contextlib.contextmanager
    def scope(self, *args, **kwargs):
        yield

    async def chat(self, messages, *, role="fast", max_tokens=None, temperature=0.0):
        if self.free_calls:
            self.free_calls -= 1
        elif self.short_above is not None and max_tokens > self.short_above:
            raise BudgetShort("only ~500 tokens left in the budget")
        self.calls.append(max_tokens)
        self.sent.append([dict(m) for m in messages])
        return self.script.pop(0)


class CodingTruncatedTest(unittest.TestCase):
    def solver(self, script: list) -> tuple[CodingSolver, _Hub, Path]:
        root = Path(tempfile.mkdtemp(prefix="codetr_"))
        self.addCleanup(shutil.rmtree, root, True)
        ws = root / "workspace"
        ws.mkdir()
        (ws / "mylib.py").write_text(LIB_BAD, encoding="utf-8")
        (ws / "test_mylib.py").write_text(TESTS, encoding="utf-8")
        task = Task(task_id="t1", task_type="coding", instruction=INSTRUCTION, reward=2, dir=root,
                    workspace=ws, question=INSTRUCTION)
        cfg = dataclasses.replace(load_config(), crosscheck=False, fastpath=False, max_tokens_code=900)
        hub = _Hub(script)
        env = SimpleNamespace(cfg=cfg, hub=hub, trace_text=False, scratch_root=root / "scratch",
                              estimator=SimpleNamespace(remember_prediction=lambda *a, **k: None))
        solver = CodingSolver(task, env)
        solver.prepare()
        self.assertFalse(solver.report.ok)
        return solver, hub, ws

    def run_fast(self, solver: CodingSolver):
        return asyncio.run(solver.run_step("fast"))

    def test_cut_off_stub_is_not_applied_and_retried_bigger(self):
        solver, hub, ws = self.solver([_reply(STUB, True), _reply(GOOD, False)])
        outcome = self.run_fast(solver)
        self.assertEqual(outcome.kind, "answer")
        self.assertEqual(hub.calls, [900, 2700])
        # повтор с нуля: без обрывка в истории, с напоминанием об обрыве
        self.assertEqual([m["role"] for m in hub.sent[1]], ["system", "user"])
        self.assertTrue(hub.sent[1][1]["content"].endswith(CUT_OFF_NOTE))
        self.assertIn('return "zero"', (ws / "mylib.py").read_text(encoding="utf-8"))

    def test_cut_off_twice_fails_without_applying(self):
        solver, hub, ws = self.solver([_reply(STUB, True), _reply(STUB, True)])
        outcome = self.run_fast(solver)
        self.assertEqual(outcome.kind, "failed")
        self.assertIn("cut off", outcome.note)
        self.assertEqual(hub.calls, [900, 2700])
        self.assertEqual(solver.current[ws / "mylib.py"], LIB_BAD)
        self.assertEqual((ws / "mylib.py").read_text(encoding="utf-8"), LIB_BAD)

    def test_budget_short_on_bigger_limit_retries_usual(self):
        solver, hub, _ = self.solver([_reply(STUB, True), _reply(GOOD, False)])
        hub.short_above = 900
        outcome = self.run_fast(solver)
        self.assertEqual(outcome.kind, "answer")
        self.assertEqual(hub.calls, [900, 900])

    def test_budget_short_everywhere_fails_step(self):
        solver, hub, ws = self.solver([_reply(STUB, True)])
        hub.short_above, hub.free_calls = 100, 1   # оба повтора упираются в остаток бюджета
        outcome = self.run_fast(solver)
        self.assertEqual(outcome.kind, "failed")
        self.assertEqual(hub.calls, [900])
        self.assertEqual((ws / "mylib.py").read_text(encoding="utf-8"), LIB_BAD)

    def test_text_cut_after_closed_block_is_applied(self):
        solver, hub, ws = self.solver([_reply(GOOD + "\n\nThe bug was that the second branch ret", True)])
        outcome = self.run_fast(solver)
        self.assertEqual(outcome.kind, "answer")
        self.assertEqual(hub.calls, [900])
        self.assertIn('return "zero"', (ws / "mylib.py").read_text(encoding="utf-8"))

    def test_cut_off_without_fences_is_not_applied(self):
        bare = STUB.replace("```python\n", "")
        solver, hub, ws = self.solver([_reply(bare, True), _reply(GOOD, False)])
        outcome = self.run_fast(solver)
        self.assertEqual(outcome.kind, "answer")
        self.assertEqual(hub.calls, [900, 2700])

    def test_not_truncated_reply_is_applied_once(self):
        solver, hub, _ = self.solver([_reply(GOOD, False)])
        self.assertEqual(self.run_fast(solver).kind, "answer")
        self.assertEqual(hub.calls, [900])

    def test_helpers(self):
        self.assertTrue(_closed_code_block("```python\nx = 1\n```\ntail"))
        self.assertFalse(_closed_code_block("```python\nx = 1\n```\n```python\ny"))
        self.assertFalse(_closed_code_block("x = 1"))
        self.assertEqual(_cut_off_limit(900), 2700)
        self.assertEqual(_cut_off_limit(2000), 4000)
        self.assertEqual(_cut_off_limit(6000), 6000)


if __name__ == "__main__":
    unittest.main()
