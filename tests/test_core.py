"""Тесты ядра: загрузка задач, разбор ответов модели, песочница, заплатки, база знаний, оценщик."""
import json
import tempfile
import unittest
from pathlib import Path

from harness import patching, textutil
from harness.answers import write_answer
from harness.estimator import Estimator, Evidence
from harness.ledger import call_cost
from harness.pot import run_pot
from harness.skills import kb as K
from harness.tasks import core_question, load_tasks, strip_preamble

SAMPLES = Path(__file__).resolve().parents[2] / "task-dir"

MATH = ("A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. It receives 2 deliveries of 87 "
        "boxes each. How many boxes does it have at the end? Compute it with the bash tool (python3 -c). Work only "
        "inside the current directory. When you are done, write ONLY the final answer to the file answer.txt in the "
        "current directory (for example with bash: echo 42 > answer.txt).")


def make_kb(root: Path, drop: str = "") -> K.KB:
    files = {
        "person_irina_smith.md": "# Irina Smith\n\nBorn: 1983\nEmployer: Dune Robotics\nMentor: Alice Smith\n",
        "person_alice_smith.md": "# Alice Smith\n\nBorn: 1970\nEmployer: Granite AI\nMentor: Irina Smith\n",
        "company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Lisbon\nFounded: 1999\n",
        "company_granite_ai.md": "# Granite AI\n\nHeadquarters: Osaka\nFounded: 1992\n",
    }
    kb_dir = root / "kb"
    kb_dir.mkdir()
    for name, text in files.items():
        if name != drop:
            (kb_dir / name).write_text(text, encoding="utf-8")
    return K.load_kb(kb_dir)


class TasksTest(unittest.TestCase):
    def test_core_question_drops_tail(self):
        q = core_question(MATH)
        self.assertTrue(q.endswith("How many boxes does it have at the end?"))
        self.assertNotIn("answer.txt", q)

    def test_strip_preamble(self):
        self.assertEqual(strip_preamble("The directory kb/ contains a small knowledge base of markdown files. "
                                        "In which year was Irina Smith born?"), "In which year was Irina Smith born?")
        self.assertEqual(strip_preamble("The file sales.csv has columns a,b (20 rows). How many a?"), "How many a?")
        self.assertEqual(strip_preamble("What is 2 + 2?"), "What is 2 + 2?")

    def test_expected_is_never_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "math_0000"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(json.dumps({
                "task_id": "math_0000", "task_type": "math", "instruction": MATH, "reward": 5,
                "expected": "SECRET-ANSWER", "check": "exact_int"}), encoding="utf-8")
            (Path(tmp) / ".DS_Store").write_text("junk", encoding="utf-8")
            tasks = load_tasks(tmp)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].reward, 5)
            self.assertNotIn("SECRET-ANSWER", repr(vars(tasks[0])))

    def test_sentinel_and_answer_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "coding_0003"
            (task_dir / "workspace").mkdir(parents=True)
            (task_dir / "task.json").write_text(json.dumps({
                "task_id": "coding_0003", "task_type": "coding", "reward": 2,
                "instruction": "Fix it. When you are done, write ONLY the word DONE to the file answer.txt."}),
                encoding="utf-8")
            task = load_tasks(tmp)[0]
            self.assertEqual(task.sentinel, "DONE")
            self.assertTrue(write_answer(task, "DONE"))
            self.assertEqual((task_dir / "answer.txt").read_text(), "DONE\n")
            self.assertFalse((task_dir / "workspace" / "answer.txt").exists())   # организаторы: рядом с task.json


class TextTest(unittest.TestCase):
    def test_extract_code(self):
        self.assertEqual(textutil.extract_code("```python\nprint(1)\n```"), "print(1)")
        self.assertEqual(textutil.extract_code("<think>hmm</think>\n1 + 2"), "1 + 2")
        self.assertEqual(textutil.extract_code("```python\nx = 1\nprint(x)"), "x = 1\nprint(x)")

    def test_refusal(self):
        self.assertTrue(textutil.is_refusal("IMPOSSIBLE"))
        self.assertTrue(textutil.is_refusal("`NOT_FOUND`."))
        self.assertTrue(textutil.is_refusal("Not found"))
        self.assertFalse(textutil.is_refusal("kb['Impossible Inc']['Founded']"))
        self.assertFalse(textutil.is_refusal("1735"))

    def test_normalize_answer(self):
        self.assertEqual(textutil.normalize_answer("Answer: 'Osaka'."), "Osaka")
        self.assertEqual(textutil.normalize_answer("389.0"), "389")
        self.assertEqual(textutil.normalize_answer("12.5"), "12.5")

    def test_cost_formula(self):
        self.assertAlmostEqual(call_cost(1000, 50, 0), 150.0)
        self.assertAlmostEqual(call_cost(1000, 500, 450), 600.0)           # рассуждения уже внутри completion
        self.assertAlmostEqual(call_cost(1000, 50, 450), 600.0)            # провайдер считает их отдельно
        self.assertAlmostEqual(call_cost(1000, 500, 450, reasoning_in_completion=False), 1050.0)


class PotTest(unittest.TestCase):
    def run_code(self, code, preload=""):
        with tempfile.TemporaryDirectory() as tmp:
            return run_pot(code, preload, Path(tmp), Path(tmp) / "scratch", 20)

    def test_expression_and_script(self):
        self.assertEqual(self.run_code("((1752 // 2) * 2) // 3").value, "584")
        self.assertEqual(self.run_code("x = 1669\nx -= 36 * 3\nx += 2 * 87\nprint(x)").value, "1735")
        self.assertEqual(self.run_code("6 / 2").value, "3")
        self.assertEqual(self.run_code("answer = 'Osaka'").value, "Osaka")

    def test_errors(self):
        missing = self.run_code("kb['Nobody']['Born']", "kb = {}\n")
        self.assertFalse(missing.ok)
        self.assertTrue(missing.missing_data)
        self.assertFalse(self.run_code("1 / 0").ok)
        self.assertFalse(self.run_code("import sys\nsys.exit(0)").ok)
        self.assertFalse(self.run_code("None").ok)

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_pot("while True:\n    pass", "", Path(tmp), Path(tmp) / "s", 1.5)
        self.assertTrue(result.timed_out)


class PatchTest(unittest.TestCase):
    SOURCE = ("def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n\n\n"
              "def maximum(xs):\n    return max(xs)\n")

    def test_replaces_only_named_function(self):
        patched = patching.apply_patch(self.SOURCE, "def total(xs):\n    return sum(xs)\n")
        self.assertIn("return sum(xs)", patched)
        self.assertIn("def maximum(xs):\n    return max(xs)", patched)
        self.assertNotIn("s -= x", patched)

    def test_rejects_garbage(self):
        self.assertIsNone(patching.apply_patch(self.SOURCE, "this is not python ("))
        self.assertIsNone(patching.apply_patch(self.SOURCE, "print('no definitions here')"))

    def test_new_import_and_function(self):
        patched = patching.apply_patch(self.SOURCE, "import math\n\ndef total(xs):\n    return math.fsum(xs)\n")
        self.assertTrue(patched.startswith("import math\n"))
        compile(patched, "x", "exec")

    def test_split_by_file(self):
        reply = "Here:\n```python\ndef total(xs):\n    return sum(xs)\n```"
        self.assertEqual(patching.split_by_file(reply, {"mylib.py": self.SOURCE}),
                         {"mylib.py": "def total(xs):\n    return sum(xs)"})


class KbTest(unittest.TestCase):
    def ask(self, question, drop=""):
        with tempfile.TemporaryDirectory() as tmp:
            return K.answer_path_question(make_kb(Path(tmp), drop), question)

    def test_base_templates(self):
        self.assertEqual(self.ask("In which year was Irina Smith born?").answer, "1983")
        self.assertEqual(self.ask("In which city is the headquarters of Irina Smith's employer?").answer, "Lisbon")

    def test_chains(self):
        self.assertEqual(self.ask("Who is the mentor of Irina Smith?").answer, "Alice Smith")
        self.assertEqual(self.ask("In which year was Irina Smith's employer founded?").answer, "1999")
        self.assertEqual(self.ask("In which city is the headquarters of the employer of Irina Smith's mentor?").answer,
                         "Osaka")
        self.assertEqual(self.ask("In which year was Irina Smith's mentor born?").answer, "1970")

    def test_refuses_what_it_does_not_understand(self):
        for question in ("Who mentors Irina Smith?", "How many people work at Dune Robotics?",
                         "Who is older, Irina Smith or Alice Smith?", "Who is mentored by Alice Smith?",
                         "Where does Irina Smith work?", "Which company is headquartered in Osaka?"):
            self.assertEqual(self.ask(question).status, "unparsed", question)

    def test_missing_data_is_infeasible_signal(self):
        self.assertEqual(self.ask("In which year was Boris Novak born?").status, "no_anchor")
        self.assertEqual(self.ask("In which city is the headquarters of Irina Smith's employer?",
                                  drop="company_dune_robotics.md").status, "broken")

    def test_value_is_not_an_unknown_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = make_kb(Path(tmp))
            self.assertEqual(K.unknown_names(kb, "Who works at Dune Robotics?"), [])
            self.assertEqual(K.unknown_names(kb, "In which year was Boris Novak born?"), ["Boris Novak"])

    @unittest.skipUnless(SAMPLES.is_dir(), "нет примеров организаторов")
    def test_organizer_samples(self):
        for name in ("search_0001", "search_0005"):
            task = json.loads((SAMPLES / name / "task.json").read_text())
            kb = K.load_kb(SAMPLES / name / "workspace" / "kb")
            result = K.answer_path_question(kb, strip_preamble(core_question(task["instruction"])))
            self.assertEqual(result.answer, task["expected"], name)


class EstimatorTest(unittest.TestCase):
    def test_failures_and_refusals_lower_the_chance(self):
        est = Estimator()
        fresh = est.p_success("coding", "fast", Evidence())
        after_fail = est.p_success("coding", "strong", Evidence(failed_steps=["fast", "fast_alt"]))
        after_refusals = est.p_success("math", "strong", Evidence(refusals=2))
        self.assertGreater(fresh, 0.6)
        self.assertLess(after_fail, fresh)
        self.assertLess(after_refusals, 0.05)

    def test_cost_grows_with_prompt_and_reasoning(self):
        est = Estimator()
        cheap = est.step_cost("math", "fast", 100, {"numbers": 4})
        long_prompt = est.step_cost("math", "fast", 2000, {"numbers": 4})
        strong = est.step_cost("math", "strong", 100, {"numbers": 4})
        self.assertLess(cheap, long_prompt)
        self.assertGreater(strong, cheap * 5)


if __name__ == "__main__":
    unittest.main()
