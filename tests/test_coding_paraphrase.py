"""coding в других формулировках: тестовый файл с необычным именем, pytest без pytest в окружении."""
from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.config import load_config
from harness.skills.repair import mutation_repair
from harness.skills.testrun import _without_pytest, find_test_command, run_tests, source_files, test_files
from harness.solvers.coding import CodingSolver, answer_marker
from harness.tasks import _SENTINEL_RE, Task

LIB_BAD = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"
LIB_GOOD = LIB_BAD.replace("s -= x", "s += x")
CASE_TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n    def test_total(self):\n"
              "        self.assertEqual(m.total([1, 2, 3]), 6)\n\n\nif __name__ == '__main__':\n    unittest.main()\n")
SELFTEST_LIB = LIB_BAD + "\n\nimport unittest\n\n\nclass _Smoke(unittest.TestCase):\n    def test_smoke(self):\n        pass\n"
PLAIN_TESTS = "import mylib as m\n\n\ndef test_total():\n    assert m.total([1, 2, 3]) == 6\n"
NO_PYTEST = importlib.util.find_spec("pytest") is None


class Base(unittest.TestCase):
    def ws(self, files: dict[str, str]) -> Path:
        root = Path(tempfile.mkdtemp(prefix="paraphrase_"))
        self.addCleanup(shutil.rmtree, root, True)
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text, encoding="utf-8")
        return root


class OddTestNames(Base):
    def test_testcase_file_with_any_name_is_a_test(self):
        for name in ("check_mylib.py", "run_tests.py", "test.py", "tests.py", "mylib_tests.py"):
            ws = self.ws({"mylib.py": LIB_BAD, name: CASE_TESTS})
            self.assertEqual([p.name for p in test_files(ws)], [name], name)
            self.assertEqual([p.name for p in source_files(ws)], ["mylib.py"], name)

    def test_source_is_not_a_test(self):
        ws = self.ws({"mylib.py": LIB_BAD + "\n\nclass Box:\n    pass\n", "helpers.py": "import unittest\n"})
        self.assertEqual(test_files(ws), [])

    def test_repair_does_not_touch_odd_named_test(self):
        ws = self.ws({"mylib.py": LIB_BAD, "check_mylib.py": CASE_TESTS})
        argv = find_test_command("Run `python3 check_mylib.py`; some tests fail.", ws)
        result = mutation_repair(ws, argv, time_budget=20.0, test_timeout=5.0)
        self.assertTrue(result.fixed)
        self.assertEqual((ws / "check_mylib.py").read_text(encoding="utf-8"), CASE_TESTS)
        self.assertEqual((ws / "mylib.py").read_text(encoding="utf-8"), LIB_GOOD)


class SelftestSource(Base):
    def test_source_with_selftest_stays_source(self):
        ws = self.ws({"mylib.py": SELFTEST_LIB, "test_mylib.py": CASE_TESTS})
        self.assertEqual([p.name for p in test_files(ws)], ["test_mylib.py"])
        self.assertEqual([p.name for p in source_files(ws)], ["mylib.py"])

    def test_odd_named_check_without_named_tests_is_a_test(self):
        ws = self.ws({"mylib.py": LIB_BAD, "check_mylib.py": CASE_TESTS})
        self.assertEqual([p.name for p in test_files(ws)], ["check_mylib.py"])
        self.assertEqual([p.name for p in source_files(ws)], ["mylib.py"])


# лишние слова капсом, в кавычках, в бэктиках рядом с answer.txt — слово по-прежнему DONE, навыки включены
NOISY_DONE = (
    "write DONE to answer.txt in the CWD.",
    "write DONE to answer.txt as plain text, not JSON.",
    "write DONE to answer.txt; if you cannot, write FAILED to answer.txt.",
    "write the single word DONE to answer.txt in UTF-8",
    "write DONE (in UPPERCASE) to answer.txt",
    "write DONE to answer.txt and nothing ELSE.",
    "write DONE to answer.txt with a trailing LF.",
    "write DONE to answer.txt in ASCII.",
    "Fix mylib.py so that `pytest` passes, then write DONE to answer.txt.",
    "Fix the `total` function, then write DONE to answer.txt.",
)


class Marker(Base):
    def solver(self, files: dict[str, str], instruction: str) -> CodingSolver:
        ws = self.ws(files)
        m = _SENTINEL_RE.search(instruction)
        task = Task(task_id="x", task_type="coding", instruction=instruction, reward=2, dir=ws.parent,
                    workspace=ws, question=instruction, sentinel=m.group(1) if m else None)
        env = SimpleNamespace(cfg=load_config(), hub=None, estimator=None, trace_text=False,
                              scratch_root=ws.parent / (ws.name + "_scratch"))
        self.addCleanup(shutil.rmtree, env.scratch_root, True)
        s = CodingSolver(task, env)
        s.note = lambda *a, **k: None
        return s

    def test_noisy_phrasings_give_done(self):
        for tail in NOISY_DONE:
            inst = "Run `python3 test_mylib.py`; some tests fail. " + tail
            self.assertEqual(answer_marker(inst, None), "DONE", tail)

    def test_noisy_phrasings_keep_skills(self):
        for tail in NOISY_DONE:
            inst = "Run `python3 test_mylib.py`; some tests fail. " + tail
            s = self.solver({"mylib.py": LIB_BAD, "test_mylib.py": CASE_TESTS}, inst)
            s.prepare()
            self.assertFalse(s.no_skills, tail)
            self.assertEqual((s.status, s.best), ("verified", "DONE"), tail)

    def test_unique_marker_is_used(self):
        self.assertEqual(answer_marker("Fix mylib.py. When done write the word OK to answer.txt "
                                       "(e.g. `echo OK > answer.txt`).", None), "OK")
        self.assertEqual(answer_marker("Fix mylib.py, then write `FIXED` to answer.txt in UTF-8.", None), "FIXED")
        self.assertEqual(answer_marker("Fix mylib.py. When you are done, write ONLY the word FIXED to the file "
                                       "answer.txt.", "FIXED"), "FIXED")
        self.assertEqual(answer_marker("Fix mylib.py.", None), "DONE")
        self.assertEqual(answer_marker("Fix mylib.py. When you are done, write fixed (and nothing else) to "
                                       "answer.txt.", None), "fixed")
        self.assertEqual(answer_marker("When finished, write done to answer.txt.", None), "DONE")
        self.assertEqual(answer_marker("Write the result to answer.txt.", None), "DONE")

    def test_ambiguous_marker_falls_back_to_done(self):
        self.assertEqual(answer_marker("Write OK to answer.txt. Or write YES to answer.txt.", None), "DONE")

    def test_named_checks_skip_command_without_files(self):
        inst = ("First run `python3 -c 'import mylib'`, then `python3 check.py`; it fails. Fix mylib.py. "
                "When you are done, write ONLY the word DONE to the file answer.txt.")
        check = "import mylib\nassert mylib.total([1, 2, 3]) == 6\nprint('ok')\n"
        s = self.solver({"mylib.py": LIB_BAD, "check.py": check}, inst)
        s.prepare()
        self.assertEqual([p.name for p in s.tests], ["check.py"])
        self.assertEqual([p.name for p in s.sources], ["mylib.py"])
        self.assertTrue(s.no_skills)


@unittest.skipUnless(NO_PYTEST, "pytest установлен: подмена команды не нужна")
class PytestMissing(Base):
    def check(self, files, instruction, fixed_rel="mylib.py"):
        ws = self.ws(files)
        argv = find_test_command(instruction, ws)
        self.assertNotIn("pytest", _without_pytest(argv, ws))
        bad = run_tests(ws, argv, 20)
        self.assertFalse(bad.ok)
        self.assertEqual(bad.error, "")
        self.assertTrue(bad.failed)
        (ws / fixed_rel).write_text(LIB_GOOD, encoding="utf-8")
        good = run_tests(ws, argv, 20)
        self.assertTrue(good.ok)
        self.assertGreater(good.total, 0)

    def test_plain_functions(self):
        self.check({"mylib.py": LIB_BAD, "test_mylib.py": PLAIN_TESTS}, "Run `pytest test_mylib.py`; some tests fail.")

    def test_plain_functions_no_args(self):
        self.check({"mylib.py": LIB_BAD, "test_mylib.py": PLAIN_TESTS}, "Run `pytest -q`; some tests fail.")

    def test_testcase_via_python_m(self):
        self.check({"mylib.py": LIB_BAD, "test_mylib.py": CASE_TESTS}, "Run `python3 -m pytest -x`; some tests fail.")

    def test_tests_directory(self):
        head = "import os\nimport sys\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        self.check({"mylib.py": LIB_BAD, "tests/test_mylib.py": head + PLAIN_TESTS}, "Run `pytest tests/`.")

    def test_unknown_argument_keeps_command(self):
        ws = self.ws({"mylib.py": LIB_BAD, "test_mylib.py": PLAIN_TESTS})
        argv = find_test_command("Run `pytest nowhere/`.", ws)
        self.assertIsNone(_without_pytest(argv, ws))

    def test_plain_python_command_unchanged(self):
        ws = self.ws({"mylib.py": LIB_BAD, "test_mylib.py": CASE_TESTS})
        argv = find_test_command("Run `python3 test_mylib.py` in the current directory.", ws)
        self.assertEqual(argv[-1], "test_mylib.py")


if __name__ == "__main__":
    unittest.main()
