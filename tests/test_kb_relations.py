"""Тесты harness.skills.kb.answer_relation_question: сравнение по году и обратные ссылки."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.skills.kb import answer_path_question, answer_relation_question, load_kb

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



class ParaphraseRefusalTest(unittest.TestCase):
    """P0-5: лишняя фраза рядом с вопросом и вопрос не того типа — отказ навыка-цепочки."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="kb_p5_"))
        for name, text in PEOPLE.items():
            (cls.root / name).write_text(text)
        cls.kb = load_kb(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, True)

    def ask(self, q):
        return answer_path_question(self.kb, "The directory kb/ contains a small knowledge base of markdown files. " + q)

    def test_plain_question_answered(self):
        res = self.ask("In which year was Anna Petrov born?")
        self.assertEqual((res.status, res.answer), ("ok", "1980"))
        res = self.ask("In which year was Anna Petrov born? Write ONLY the answer to answer.txt.")
        self.assertEqual((res.status, res.answer), ("ok", "1980"))

    def test_format_sentence_refused(self):
        for tail in (" Answer with the last two digits only.", " Give the year minus 1900."):
            self.assertEqual(self.ask("In which year was Anna Petrov born?" + tail).status, "unparsed", tail)
        self.assertIsNone(answer_relation_question(self.kb, "Who is older, Anna Petrov or Boris Sato? "
                                                   "Answer with the first name only.").answer)

    def test_two_questions_not_broken(self):
        # раньше цепочка Born -> Mentor давала broken (невыполнима) — а вопрос просто двойной
        self.assertEqual(self.ask("In which year was Anna Petrov born, and who is the mentor?").status, "unparsed")

    def test_description_instead_of_name_not_infeasible(self):
        res = answer_relation_question(self.kb, "Who was born earlier, Anna Petrov or her mentor?")
        self.assertEqual(res.status, "unparsed")

    def test_type_mismatch_refused(self):
        for q in ("Where was Anna Petrov born?", "In which city was Anna Petrov born?", "Who founded Dune Robotics?",
                  "Where is Anna Petrov's employer located?", "In which city is Anna Petrov's mentor's employer located?"):
            self.assertEqual(self.ask(q).status, "unparsed", q)


if __name__ == "__main__":
    unittest.main()
