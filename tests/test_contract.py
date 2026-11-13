"""Договор с моделью на втором ходу, заземление ответа, последняя строка-выражение (P1-1, P1-2, P1-3)."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.pot import PotResult, run_pot
from harness.solvers.base import invented_values
from harness.solvers.families import DataSolver, MathSolver, SearchSolver
from harness.tasks import Task


def _reply(text: str):
    return SimpleNamespace(text=text, model="m", cost=1.0, prompt_tokens=10, completion_tokens=5,
                           reasoning_tokens=0, finish_reason="stop", truncated=False)


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def solver(self, cls, question: str, files: dict | None = None):
        ws = self.root / "workspace"
        for rel, text in (files or {}).items():
            path = ws / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        task = Task(task_id="t1", task_type=cls.family, instruction=question, reward=1.0, dir=self.root,
                    workspace=ws, question=question)
        cfg = SimpleNamespace(fastpath=True, crosscheck=False, exec_timeout_s=10.0)
        env = SimpleNamespace(cfg=cfg, hub=None, estimator=None, scratch_root=self.root / "scratch", trace_text=False)
        return cls(task, env)

    @staticmethod
    def scripted(solver, replies: list[str]) -> list[list[dict]]:
        """Подменяет вызов модели готовыми ответами; возвращает список отправленных сообщений."""
        sent: list[list[dict]] = []

        async def fake_ask(step, messages, temperature=0.0):
            sent.append([dict(m) for m in messages])
            return _reply(replies[len(sent) - 1])

        solver._ask = fake_ask
        return sent


class LastExpressionTest(_Base):
    def run_code(self, code: str) -> PotResult:
        return run_pot(code, "", self.root, self.root / "scratch", 10)

    def test_last_expression_is_the_value(self):
        self.assertEqual(self.run_code("import statistics; statistics.median([3, 1, 2])").value, "2")
        self.assertEqual(self.run_code("xs = [1, 2, 3, 4]\nsum(xs) / len(xs)").value, "2.5")

    def test_print_still_wins_when_last_statement_prints(self):
        self.assertEqual(self.run_code("x = 5\nprint(x)").value, "5")
        self.assertEqual(self.run_code("def main():\n    print(42)\n    return 0\nmain()").value, "42")

    def test_answer_variable_and_none_tail(self):
        self.assertEqual(self.run_code("a = 3\nanswer = a * 4").value, "12")
        self.assertFalse(self.run_code("xs = [1]\nxs.append(2)").ok)


class RefusalContractTest(_Base):
    def test_unknown_symbol_is_a_refusal_without_retry(self):
        s = self.solver(MathSolver, "A shop has 1832 boxes. It ships N boxes per day for 3 days. How many are left?")
        sent = self.scripted(s, ["print(1832 - 3 * N)", "N = 0\nprint(1832 - 3 * N)"])
        outcome = asyncio.run(s.run_step("fast"))
        self.assertEqual(outcome.kind, "refused")
        self.assertEqual(len(sent), 1)

    def test_lowercase_letter_symbol_is_a_refusal(self):
        s = self.solver(MathSolver, "A warehouse starts with 2083 boxes. It ships k boxes per day for 2 days.")
        res = PotResult(False, error="NameError: name 'k' is not defined", etype="NameError")
        self.assertTrue(s.absent_input(res))
        res = PotResult(False, error="NameError: name 'a' is not defined", etype="NameError")
        self.assertEqual(s.absent_input(res), "")

    def test_lowercase_name_error_is_retried(self):
        s = self.solver(MathSolver, "A shop has 10 boxes and gets 5 more. What is the total?")
        sent = self.scripted(s, ["print(total)", "print(10 + 5)"])
        outcome = asyncio.run(s.run_step("fast"))
        self.assertEqual((outcome.kind, outcome.answer), ("answer", "15"))
        self.assertEqual(len(sent), 2)
        retry = sent[1][-1]["content"]
        self.assertIn("IMPOSSIBLE", retry)
        self.assertIn("Never invent", retry)

    def test_missing_mentioned_file_is_a_refusal(self):
        s = self.solver(MathSolver, "It ships as many boxes as listed in shipments.txt. How many are left of 900?")
        res = PotResult(False, error="FileNotFoundError: [Errno 2] No such file or directory: 'shipments.txt'",
                        etype="FileNotFoundError")
        self.assertIn("shipments.txt", s.absent_input(res))

    def test_present_file_is_not_a_refusal(self):
        s = self.solver(DataSolver, "Use sales.csv. What is the total?", files={"data/sales.csv": "a\n1\n"})
        res = PotResult(False, error="FileNotFoundError: [Errno 2] No such file or directory: 'sales.csv'",
                        etype="FileNotFoundError")
        self.assertEqual(s.absent_input(res), "")

    def test_search_retry_mentions_not_found(self):
        s = self.solver(SearchSolver, "Where was Ann born?")
        s.system_prompt = SearchSolver.system_prompt
        self.assertIn("NOT_FOUND", s.retry_prompt("boom"))

    def test_invented_values_after_missing_data(self):
        q = "A shop has 1832 boxes. It ships N boxes per day for 3 days. How many are left?"
        name_err = PotResult(False, error="NameError: name 'N' is not defined", etype="NameError")
        self.assertTrue(invented_values("N = 0\nprint(1832 - 3 * N)", "print(1832 - 3 * N)", name_err, q))
        file_err = PotResult(False, error="FileNotFoundError: x", etype="FileNotFoundError")
        found = invented_values("s = [100, 200, 50]\nprint(900 - sum(s))", "open('s.txt')", file_err,
                                "Subtract what is listed in s.txt from 900.")
        self.assertIn("200", found)

    def test_legitimate_fixes_are_not_invention(self):
        own = PotResult(False, error="NameError: name 'total' is not defined", etype="NameError")
        self.assertEqual(invented_values("total = 0\ntotal += 10 + 5\nprint(total)", "print(total)", own,
                                         "It has 10 boxes and gets 5 more. What is the total?"), [])
        self.assertEqual(invented_values("print(round(200 * 0.15))", "print(price * 0.15)", own,
                                         "A price of 200 grows by 15%. What is the total?"), [])
        self.assertEqual(invented_values("print(300 - 20)", "print(x)", own,
                                         "There are three hundred boxes; 20 ship. What is the total?"), [])
        type_err = PotResult(False, error="TypeError: x", etype="TypeError")
        self.assertEqual(invented_values("print(12345)", "print(x)", type_err, "What?"), [])


class GroundingTest(_Base):
    def kb_solver(self):
        s = self.solver(SearchSolver, "Who works at Acme?")
        s.kb = SimpleNamespace(texts={"Ann Lee": "Name: Ann Lee\nEmployer: Acme\nCity: Osaka"},
                               entries={"Ann Lee": {"Employer": "Acme", "City": "Osaka"}, "Acme": {"Founded": 1983}})
        s.direct = False
        return s

    def test_plain_answer_must_be_in_kb(self):
        s = self.kb_solver()
        self.assertEqual(s.ungrounded("Osaka"), "")
        self.assertEqual(s.ungrounded("osaka"), "")
        self.assertIn("Kyoto", s.ungrounded("Kyoto"))
        self.assertEqual(s.ungrounded("1983"), "")
        self.assertEqual(s.ungrounded("yes"), "")

    def test_paraphrase_is_not_a_hallucination(self):
        s = self.kb_solver()
        s.kb.texts["Acme"] = "Acme builds industrial robotics."
        self.assertEqual(s.ungrounded("industrial robots"), "")
        self.assertEqual(s.ungrounded("Ann Lee and Acme"), "")
        self.assertIn("Bob", s.ungrounded("Ann Lee, Bob Smith"))

    def test_code_answer_checks_only_typed_literals(self):
        s = self.kb_solver()
        self.assertEqual(s.ungrounded("Osaka", "kb['Ann Lee']['City']"), "")
        self.assertEqual(s.ungrounded("Bob Smith", "kb['Ann Lee']['Mentor']"), "")
        self.assertIn("Alice", s.ungrounded("Alice Novak, Bob Smith", "['Alice Novak', 'Bob Smith']"))

    def test_plain_hallucination_goes_back_to_the_model(self):
        s = self.kb_solver()
        out, err = asyncio.run(s._interpret("fast", _reply("Kyoto")))
        self.assertIsNone(out)
        self.assertIn("NOT_FOUND", err)
        out, _ = asyncio.run(s._interpret("fast", _reply("Osaka")))
        self.assertEqual(out.answer, "Osaka")

    def test_data_grounding_uses_rows_and_columns(self):
        s = self.solver(DataSolver, "Which region sold most?", files={"sales.csv": "region,units\nNorth,5\n"})
        s.tables = [SimpleNamespace(columns=["region", "units"], rows=[{"region": "North", "units": 5}],
                                    path=s.task.workspace / "sales.csv")]
        self.assertEqual(s.ungrounded("North"), "")
        self.assertEqual(s.ungrounded("region"), "")
        self.assertEqual(s.ungrounded("March"), "")
        self.assertTrue(s.ungrounded("South"))

    def test_math_is_not_grounded(self):
        s = self.solver(MathSolver, "How many?")
        self.assertEqual(s.ungrounded("Kyoto"), "")


if __name__ == "__main__":
    unittest.main()
