"""ReAct: подсказка перед последним ходом и проверка ответа агента — выдумка и заземление (без модели).

Агент openJiuwen подменяется сценарием: на каждом «ходу» записываем вызов модели в учёт
(как это делает LLMHub) и выполняем команду настоящим инструментом bash из run_react.
"""
from __future__ import annotations

import asyncio
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness.ledger import CallRecord, Ledger
from harness.solvers import react as R
from harness.solvers.families import DataSolver, MathSolver
from harness.tasks import Task


class _FakeHub:
    def __init__(self):
        self.ledger = Ledger()

    @contextlib.contextmanager
    def scope(self, *args, **kwargs):
        yield

    def model(self):
        return None


def _scripted_agent(script: list, outputs: list, cost: float = 100.0):
    """Класс-подмена ReActAgent: script — список команд bash (None — ход без инструмента: конец)."""

    class Agent:
        def __init__(self, card=None):
            self.tools = {}
            self.ability_manager = SimpleNamespace(add_ability=self._add, remove_ability=lambda name: None)

        def _add(self, card, fn):
            self.tools[card.name] = fn

        def configure(self, config):
            self.max_iterations = config.max_iterations

        def set_llm(self, llm):
            pass

        async def invoke(self, inputs):
            ledger = self.hub.ledger
            for i in range(self.max_iterations):
                ledger.add(CallRecord(task_id="t1", family="x", step="react", model="m", prompt_tokens=0,
                                      completion_tokens=int(cost), reasoning_tokens=0, cost=cost, seconds=0.0,
                                      ok=True))
                command = script[i] if i < len(script) else None
                if command is None or command.startswith("FINAL:"):
                    return {"output": command[6:] if command else "FINISHED"}
                outputs.append(await self.tools["bash"].invoke({"command": command}))
            return {"output": "Max iterations reached without completion"}

    return Agent


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_agent(self, cls, question: str, files: dict, script: list, cost: float = 100.0,
                  token_cap: float = 1500.0, iterations: int = 5):
        ws = self.root / "workspace"
        for rel, text in files.items():
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            (ws / rel).write_text(text, encoding="utf-8")
        task = Task(task_id="t1", task_type=cls.family, instruction=question, reward=1.0, dir=self.root,
                    workspace=ws, question=question)
        cfg = SimpleNamespace(fastpath=True, crosscheck=False, exec_timeout_s=10.0, task_token_cap=6000.0,
                              react_token_cap=token_cap, react_max_iterations=iterations, max_tokens_code=900,
                              call_timeout_fast=30.0, model_for=lambda role: "m")
        hub = _FakeHub()
        estimator = SimpleNamespace(observe_react=lambda family, used: None)
        env = SimpleNamespace(cfg=cfg, hub=hub, estimator=estimator, scratch_root=self.root / "scratch",
                              trace_text=False)
        solver = cls(task, env)
        solver.prepare()
        outputs: list[str] = []
        agent_cls = _scripted_agent(script, outputs, cost)
        agent_cls.hub = hub
        with mock.patch("openjiuwen.core.single_agent.ReActAgent", agent_cls):
            outcome = asyncio.run(R.run_react(solver, "react"))
        return outcome, outputs


CSV = {"sales.csv": "region,amount\nnorth,10\nsouth,20\nnorth,5\n"}


class LastStepNoteTest(_Base):
    def test_note_only_before_the_last_turn(self):
        # 5 ходов: вывод инструмента на ходу 4 модель читает на ходу 5 — последнем.
        _, outputs = self.run_agent(DataSolver, "Total amount in the west region?", CSV, ["echo hi"] * 5)
        self.assertEqual(len(outputs), 5)
        for i in range(3):
            self.assertNotIn(R.LAST_STEP_NOTE, outputs[i])
        self.assertIn(R.LAST_STEP_NOTE, outputs[3])

    def test_note_when_tokens_run_out(self):
        # Потолок 1500, ход по 500: после второго хода остаётся 500 — хватит только на один вызов.
        _, outputs = self.run_agent(DataSolver, "Total amount in the west region?", CSV, ["echo hi"] * 5,
                                    cost=500.0)
        self.assertNotIn(R.LAST_STEP_NOTE, outputs[0])
        self.assertIn(R.LAST_STEP_NOTE, outputs[1])


class GroundingTest(_Base):
    def test_invented_name_rejected(self):
        outcome, outputs = self.run_agent(DataSolver, "Which region has the highest amount in column rate?", CSV,
                                          ["echo east > answer.txt", "FINAL:FINISHED"])
        self.assertEqual(outcome.kind, "failed")
        self.assertIn("[check] answer.txt", outputs[0])

    def test_invented_then_refusal_is_refusal(self):
        outcome, _ = self.run_agent(DataSolver, "Which region has the highest rate?", CSV,
                                    ["echo east > answer.txt", "FINAL:IMPOSSIBLE"])
        self.assertEqual(outcome.kind, "refused")

    def test_grounded_name_accepted(self):
        outcome, outputs = self.run_agent(DataSolver, "Which region has the highest total amount?", CSV,
                                          ["echo south > answer.txt", "FINAL:FINISHED"])
        self.assertEqual((outcome.kind, outcome.answer), ("answer", "south"))
        self.assertNotIn("[check]", outputs[0])

    def test_number_not_checked(self):
        outcome, _ = self.run_agent(DataSolver, "Total amount in the north region?", CSV,
                                    ["echo 15 > answer.txt", "FINAL:FINISHED"])
        self.assertEqual((outcome.kind, outcome.answer), ("answer", "15"))

    def test_expression_instead_of_number_rejected(self):
        outcome, outputs = self.run_agent(MathSolver, "How many boxes are left after shipping X boxes from 2176?", {},
                                          ["echo '2176-3*X' > answer.txt", "FINAL:FINISHED"])
        self.assertEqual(outcome.kind, "failed")
        self.assertIn("[check] answer.txt", outputs[0])

    def test_unevaluated_expression_rejected(self):
        q = "A warehouse starts with 2176 boxes. It ships X boxes per day for 3 days. How many boxes are left?"
        outcome, _ = self.run_agent(MathSolver, q, {}, ["echo '2176 - 3*X' > answer.txt", "FINAL:FINISHED"])
        self.assertEqual(outcome.kind, "failed")
        outcome, _ = self.run_agent(MathSolver, q, {}, ["echo 1e-05 > answer.txt", "FINAL:FINISHED"])
        self.assertEqual((outcome.kind, outcome.answer), ("answer", "1e-05"))

    def test_math_has_no_grounding(self):
        outcome, _ = self.run_agent(MathSolver, "What is 6 times 7? Answer with the word.", {},
                                    ["echo fortytwo > answer.txt", "FINAL:FINISHED"])
        self.assertEqual((outcome.kind, outcome.answer), ("answer", "fortytwo"))


class AcceptCorrectTest(_Base):
    """Проверка ловит выдуманное, а не стиль: верные ответы агента принимаются."""

    def accept(self, cls, question: str, answer: str, files: dict = None):
        outcome, outputs = self.run_agent(cls, question, files or {},
                                          [f"echo '{answer}' > answer.txt", "FINAL:FINISHED"])
        self.assertEqual(outcome.kind, "answer", (question, answer, outputs))
        self.assertFalse(any("[check]" in o for o in outputs), (question, answer, outputs))
        return outcome.answer

    def test_symbolic_math_answers(self):
        # находка 1: буквы в ответе — не обязательно неизвестная из условия
        cases = [("What is the derivative of x^2 with respect to x?", "2*x"),
                 ("Simplify (a+b)^2 - b^2.", "a^2 + 2*a*b"),
                 ("Factor x^2 - 1.", "(x-1)*(x+1)"),
                 ("Expand (x+1)^2.", "x^2 + 2*x + 1"),
                 ("What is e squared, written exactly?", "e^2"),
                 ("A car travels 100 m in 20 s. What is its speed?", "5 m/s"),
                 ("What is the speed in m/s of a car covering 100 m in 20 s?", "5 m/s")]
        for q, a in cases:
            self.accept(MathSolver, q, a)

    def test_unknown_from_question_still_rejected(self):
        q = "A box holds N apples. Two boxes are added to 5 apples. How many apples are there?"
        outcome, _ = self.run_agent(MathSolver, q, {}, ["echo '2*N + 5' > answer.txt", "FINAL:FINISHED"])
        self.assertEqual(outcome.kind, "failed")

    def test_answer_taken_from_question(self):
        # находка 2: «increase» есть в самом вопросе — не выдумка, хотя в данных этого слова нет
        q = "Did the total amount increase or decrease from north to south?"
        self.assertEqual(self.accept(DataSolver, q, "increase", CSV), "increase")
        self.assertEqual(self.accept(DataSolver, "Which of the regions has more rows, north or south?", "north",
                                     CSV), "north")
        # слово не из вопроса и не из данных по-прежнему отклоняется
        outcome, _ = self.run_agent(DataSolver, q, CSV, ["echo 'grew sharply' > answer.txt", "FINAL:FINISHED"])
        self.assertEqual(outcome.kind, "failed")

    def test_style_not_rejected(self):
        # находка 3: замечания shape_answer о стиле не отклоняют ответ агента
        self.assertEqual(self.accept(MathSolver, "Which of the numbers 2^10 and 10^3 is larger?", "1024"), "1024")
        self.assertEqual(self.accept(MathSolver, "Which one is bigger, 3/4 or 0.7? Give the value.", "0.75"), "0.75")
        self.assertEqual(self.accept(MathSolver, "How much is 1 divided by 100000?", "1e-05"), "1e-05")
        self.accept(MathSolver, "How many meters is 3 km?", "3000 m")

    def test_true_false_instead_of_value_rejected(self):
        outcome, outputs = self.run_agent(DataSolver, "Which region has the highest total amount?", CSV,
                                          ["echo True > answer.txt", "FINAL:FINISHED"])
        self.assertEqual(outcome.kind, "failed")
        self.assertIn("[check] answer.txt", outputs[0])


if __name__ == "__main__":
    unittest.main()
