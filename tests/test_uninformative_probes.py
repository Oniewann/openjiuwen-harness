"""coding: сверка правки перебором с версией модели. Пробы, на которых обе версии бросают исключение
(или которые изменённую функцию вообще не вызывают), совпадения не доказывают: берётся версия модели."""
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
from harness.skills.semantic import _probes, behavior_compare, behavior_diff
from harness.solvers.coding import CodingSolver
from harness.tasks import Task

WORKDAYS = ("import datetime as _d\n\n\n"
            "# Рабочие дни (пн-пт) в полуинтервале [s1, s2), даты строками ГГГГ-ММ-ДД.\n"
            "def make_list(s1, s2):\n"
            "    a = _d.date.fromisoformat(s1)\n    b = _d.date.fromisoformat(s2)\n    n = {init}\n"
            "    while a < b:\n        if a.weekday() < {limit}:\n            n += 1\n"
            "        a += _d.timedelta(days=1)\n    return n\n")
WORKDAYS_TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                  "    def test_week(self):\n        self.assertEqual(m.make_list('2026-11-09', '2026-11-16'), 5)\n\n\n"
                  "if __name__ == '__main__':\n    unittest.main()\n")

# формат «1h30m»: любые строковые пробы, кроме вызова из теста, роняют разбор в обеих версиях
DURATION = ("import re\n\n\n"
            "def duration(s1, s2):\n"
            "    a = re.fullmatch(r'(\\d+)h(\\d+)m', s1)\n    b = re.fullmatch(r'(\\d+)h(\\d+)m', s2)\n"
            "    d = int(b[1]) * 60 + int(b[2]) - int(a[1]) * 60 - int(a[2])\n    return {ret}\n")
DURATION_TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                  "    def test_d(self):\n        self.assertEqual(m.duration('1h0m', '2h30m'), 90)\n\n\n"
                  "if __name__ == '__main__':\n    unittest.main()\n")

METHOD = ("class Acc:\n    def __init__(self):\n        self.v = 0\n\n"
          "    def add(self, x):\n        self.v += {expr}\n        return self.v\n")
METHOD_TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(m.Acc().add(2), 2)\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n")

ADD = "def add(a, b):\n    return {expr}\n"
ADD_TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
             "    def test_add(self):\n        self.assertEqual(m.add(2, 3), 5)\n\n\n"
             "if __name__ == '__main__':\n    unittest.main()\n")


def _ws(tests: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix="unp_"))
    (root / "test_mylib.py").write_text(tests, encoding="utf-8")
    return root


class _Hub:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    @contextlib.contextmanager
    def scope(self, *args, **kwargs):
        yield

    async def chat(self, messages, *, role="fast", max_tokens=None, temperature=0.0):
        self.calls += 1
        return SimpleNamespace(text=self.text, model="m", cost=1.0, prompt_tokens=10, completion_tokens=5,
                               reasoning_tokens=0, finish_reason="stop", truncated=False)


class BehaviorCompareTest(unittest.TestCase):
    def compare(self, tests: str, a: str, b: str):
        ws = _ws(tests)
        self.addCleanup(shutil.rmtree, ws, True)
        return behavior_compare(ws, {"mylib.py": a}, {"mylib.py": b})

    def test_iso_dates_are_shifted(self):
        probes = _probes([("2026-11-09", "2026-11-16")])
        self.assertIn(("2026-11-14", "2026-11-16"), probes)
        self.assertIn(("2026-11-09", "2026-11-10"), probes)

    def test_date_mutation_differs(self):
        mutated = WORKDAYS.format(init=-1, limit=6)
        model = WORKDAYS.format(init=0, limit=5)
        self.assertEqual(self.compare(WORKDAYS_TESTS, mutated, model), (["make_list"], True))

    def test_only_exceptions_is_uninformative(self):
        diff, informative = self.compare(DURATION_TESTS, DURATION.format(ret="d"),
                                         DURATION.format(ret="max(0, d)"))
        self.assertEqual(diff, [])
        self.assertFalse(informative)

    def test_method_change_is_uninformative(self):
        diff, informative = self.compare(METHOD_TESTS, METHOD.format(expr="abs(x)"), METHOD.format(expr="x"))
        self.assertEqual(diff, [])
        self.assertFalse(informative)

    def test_real_agreement_is_informative(self):
        self.assertEqual(self.compare(ADD_TESTS, ADD.format(expr="a + b"), ADD.format(expr="b + a")), ([], True))
        ws = _ws(ADD_TESTS)
        self.addCleanup(shutil.rmtree, ws, True)
        self.assertEqual(behavior_diff(ws, {"mylib.py": ADD.format(expr="a + b")},
                                       {"mylib.py": ADD.format(expr="b + a")}), [])


class CrosscheckChoiceTest(unittest.TestCase):
    def crosscheck(self, tests: str, buggy: str, mutated: str, model: str) -> tuple[str, _Hub]:
        root = Path(tempfile.mkdtemp(prefix="unpx_"))
        self.addCleanup(shutil.rmtree, root, True)
        ws = root / "workspace"
        ws.mkdir()
        (ws / "mylib.py").write_text(buggy, encoding="utf-8")
        (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
        instruction = "Run `python3 test_mylib.py`; some tests fail. Fix mylib.py, then write DONE to answer.txt."
        task = Task(task_id="t1", task_type="coding", instruction=instruction, reward=2, dir=root,
                    workspace=ws, question=instruction)
        cfg = dataclasses.replace(load_config(), crosscheck=False, fastpath=False)
        hub = _Hub(f"```python\n{model}```")
        env = SimpleNamespace(cfg=cfg, hub=hub, trace_text=False, scratch_root=root / "scratch",
                              estimator=SimpleNamespace(remember_prediction=lambda *a, **k: None))
        solver = CodingSolver(task, env)
        solver.prepare()
        self.assertFalse(solver.report.ok)
        # как после перебора: подобранная правка записана и проходит видимые тесты, ждёт сверки
        solver.pre_mutation = (dict(solver.current), solver.report)
        solver.current = dict(solver.current)
        solver.current[ws / "mylib.py"] = mutated
        solver._write(solver.current)
        solver.provisional = True
        outcome = asyncio.run(solver._llm_crosscheck("fast"))
        self.assertEqual(outcome.kind, "answer")
        return (ws / "mylib.py").read_text(encoding="utf-8"), hub

    def test_workdays_takes_model_version(self):
        model = WORKDAYS.format(init=0, limit=5)
        text, hub = self.crosscheck(WORKDAYS_TESTS, WORKDAYS.format(init=0, limit=6),
                                    WORKDAYS.format(init=-1, limit=6), model)
        self.assertEqual(hub.calls, 1)
        self.assertEqual(text, model)

    def test_uninformative_probes_take_model_version(self):
        model = DURATION.format(ret="max(0, d)")
        text, _ = self.crosscheck(DURATION_TESTS, DURATION.format(ret="d + 1"), DURATION.format(ret="d"), model)
        self.assertEqual(text, model)

    def test_informative_agreement_keeps_mutation(self):
        mutated = ADD.format(expr="a + b")
        text, _ = self.crosscheck(ADD_TESTS, ADD.format(expr="a + b + 1"), mutated, ADD.format(expr="b + a"))
        self.assertEqual(text, mutated)


if __name__ == "__main__":
    unittest.main()
