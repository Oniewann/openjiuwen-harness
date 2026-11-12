"""Тесты harness.skills.kb.answer_relation_question: сравнение по году и обратные ссылки."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.skills.kb import answer_relation_question, load_kb

PEOPLE = {
    "person_anna_petrov.md": "# Anna Petrov\n\nBorn: 1980\nEmployer: Dune Robotics\nMentor: Boris Sato\n",
    "person_boris_sato.md": "# Boris Sato\n\nBorn: 1975\nEmployer: Dune Robotics\nMentor: Carl Wang\n",
    "person_carl_wang.md": "# Carl Wang\n\nBorn: 1980\nEmployer: Ember Bio\nMentor: Boris Sato\n",
    "company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Osaka\nFounded: 2001\n",
    "company_ember_bio.md": "# Ember Bio\n\nHeadquarters: Lima\nFounded: 1999\n",
}


class RelationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="kb_rel_"))
        for name, text in PEOPLE.items():
            (cls.root / name).write_text(text)
        cls.kb = load_kb(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, True)

    def ask(self, q):
        return answer_relation_question(self.kb, "The directory kb/ contains a small knowledge base of markdown files. " + q)

    def test_born_earlier(self):
        res = self.ask("Who was born earlier, Anna Petrov or Boris Sato?")
        self.assertEqual((res.status, res.answer), ("ok", "Boris Sato"))

    def test_younger(self):
        res = self.ask("Who is younger, Boris Sato or Anna Petrov?")
        self.assertEqual((res.status, res.answer), ("ok", "Anna Petrov"))

    def test_same_year_has_no_answer(self):
        self.assertEqual(self.ask("Who was born earlier, Anna Petrov or Carl Wang?").status, "broken")

    def test_unknown_person_is_infeasible(self):
        self.assertEqual(self.ask("Who was born earlier, Anna Petrov or Zed Quill?").status, "no_anchor")

    def test_inverse_single(self):
        res = self.ask("Who is mentored by Carl Wang?")
        self.assertEqual((res.status, res.answer), ("ok", "Boris Sato"))

    def test_inverse_several_refused(self):
        self.assertEqual(self.ask("Who is mentored by Boris Sato?").status, "unparsed")

    def test_inverse_nobody(self):
        self.assertEqual(self.ask("Who is mentored by Anna Petrov?").status, "broken")


if __name__ == "__main__":
    unittest.main()
