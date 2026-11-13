"""Устойчивость загрузчика задач к неожиданному task.json."""
import json
import tempfile
import unittest
from pathlib import Path

from harness.tasks import load_tasks


def _put(root: Path, name: str, content, raw: bytes = None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    data = raw if raw is not None else json.dumps(content).encode("utf-8")
    (d / "task.json").write_bytes(data)
    return d


class RobustTasks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def one(self):
        tasks = load_tasks(self.root)
        self.assertEqual(len(tasks), 1)
        return tasks[0]

    def test_bom(self):
        _put(self.root, "a", None, b"\xef\xbb\xbf" + json.dumps(
            {"task_id": "a", "task_type": "math", "instruction": "1+2 equals what?", "reward": 2}).encode())
        t = self.one()
        self.assertEqual((t.task_id, t.reward), ("a", 2.0))

    def test_utf16(self):
        _put(self.root, "a", None, json.dumps({"task_id": "a", "instruction": "x 12"}).encode("utf-16"))
        self.assertEqual(self.one().task_id, "a")

    def test_missing_and_string_reward(self):
        _put(self.root, "a", {"task_id": "a", "instruction": "q 1"})
        _put(self.root, "b", {"task_id": "b", "instruction": "q 1", "reward": "3"})
        _put(self.root, "c", {"task_id": "c", "instruction": "q 1", "reward": -5})
        _put(self.root, "d", {"task_id": "d", "instruction": "q 1", "reward": "abc"})
        r = {t.task_id: t.reward for t in load_tasks(self.root)}
        self.assertEqual(r, {"a": 1.0, "b": 3.0, "c": 1.0, "d": 1.0})

    def test_type_case_and_unknown(self):
        _put(self.root, "a", {"task_id": "a", "task_type": "  Math ", "instruction": "q 1"})
        _put(self.root, "b", {"task_id": "b", "task_type": "weird", "instruction": "What is 3 plus 4?"})
        _put(self.root, "c", {"task_id": "c", "instruction": "x"})
        ty = {t.task_id: t.task_type for t in load_tasks(self.root)}
        self.assertEqual(ty["a"], "math")
        self.assertEqual(ty["b"], "math")  # по содержимому
        self.assertEqual(len(ty), 3)

    def test_alias_and_extra_fields(self):
        _put(self.root, "a", {"task_id": "a", "question": "What is 2 plus 2?", "expected": "4", "zzz": 1})
        t = self.one()
        self.assertIn("2 plus 2", t.instruction)

    def test_broken_json_skipped(self):
        _put(self.root, "bad", None, b"{not json")
        _put(self.root, "list", [1, 2])
        _put(self.root, "ok", {"task_id": "ok", "instruction": "q 1"})
        (self.root / "empty").mkdir()
        (self.root / "notes.txt").write_text("x")
        self.assertEqual([t.task_id for t in load_tasks(self.root)], ["ok"])

    def test_nested_and_duplicates(self):
        _put(self.root / "group", "t1", {"task_id": "same", "instruction": "q 1"})
        _put(self.root, "t2", {"task_id": "same", "instruction": "q 1"})
        tasks = load_tasks(self.root)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len({t.task_id for t in tasks}), 2)
        self.assertTrue(all((t.dir / "task.json").is_file() for t in tasks))

    def test_non_string_fields(self):
        _put(self.root, "a", {"task_id": 7, "task_type": None, "instruction": 123, "reward": None})
        t = self.one()
        self.assertEqual(t.task_id, "7")


if __name__ == "__main__":
    unittest.main()
