"""Отсев невыполнимых задач без модели (P1-8): ловится только то, что разобрано целиком после одной замены.

Для каждого правила — невыполнимые формулировки (помечаются) и близкие выполнимые (не помечаются:
ложное «невыполнима» стоит задачи без шанса на модель).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.skills.kb import answer_path_question, infeasible_probe, load_kb
from harness.skills.mathword import answer_math_question
from harness.solvers.families import DataSolver, MathSolver, SearchSolver
from harness.tasks import Task

TAIL = (" Compute it with the bash tool (python3 -c). Work only inside the current directory. When you are done, "
        "write ONLY the final answer to the file answer.txt in the current directory (for example with bash: "
        "echo 42 > answer.txt).")
START = "A warehouse starts with 900 boxes. "
END = " How many boxes does it have at the end?"


def ask(text: str, files=frozenset()):
    return answer_math_question(text + TAIL, workspace_files=set(files))


class MathInfeasibleTest(unittest.TestCase):
    def assertInfeasible(self, text, files=frozenset()):
        res = ask(text, files)
        self.assertEqual(res.status, "infeasible", (text, res.detail))
        return res

    def assertNotInfeasible(self, text, files=frozenset()):
        res = ask(text, files)
        self.assertNotEqual(res.status, "infeasible", (text, res.detail))
        return res

    def test_inexact_numbers(self):
        for op in ("It ships about 31 boxes per day for 2 days.", "It ships at least 12 boxes per day for 3 days.",
                   "It receives up to 40 boxes.", "It ships 20 boxes per day for nearly 4 days.",
                   "It receives more than 50 boxes.", "It ships between 10 and 20 boxes per day for 3 days.",
                   "It ships 10-20 boxes per day for 3 days.", "It receives 30 to 40 boxes."):
            self.assertInfeasible(START + op + END)
        self.assertInfeasible("A warehouse starts with about 900 boxes. It ships 30 boxes." + END)
        self.assertInfeasible("A warehouse starts with some boxes. It ships 30 boxes per day for 2 days." + END)
        self.assertInfeasible("A warehouse starts with X boxes. It ships 30 boxes." + END)

    def test_exact_numbers_stay(self):
        res = self.assertNotInfeasible(START + "It ships 30 boxes per day for 2 days." + END)
        self.assertEqual(res.answer, "840")
        res = self.assertNotInfeasible("A warehouse starts with 1,752 boxes. It ships 52 boxes." + END)
        self.assertEqual(res.answer, "1700")
        # неточность в операции, о которой не спрашивают: «сколько отгружено» по-прежнему определено
        self.assertNotInfeasible(START + "It receives about 30 boxes. It ships 20 boxes. "
                                         "How many boxes did it ship in total?")

    def test_deliveries_without_size(self):
        self.assertInfeasible(START + "It receives 3 deliveries." + END)
        self.assertNotInfeasible(START + "It receives 3 deliveries of 20 boxes each." + END)

    def test_no_start(self):
        self.assertInfeasible("A warehouse ships 30 boxes per day for 2 days. It receives 40 boxes." + END)
        # новый или пустой склад начинает с нуля; сумма отгрузок от начала не зависит
        self.assertNotInfeasible("A new warehouse receives 2 deliveries of 50 boxes each." + END)
        self.assertNotInfeasible("An empty warehouse receives 40 boxes. It ships 10 boxes." + END)
        self.assertNotInfeasible("A warehouse ships 30 boxes. It ships 20 boxes. How many boxes did it ship in total?")
        self.assertNotInfeasible("A warehouse ships 30 boxes. It started with 900 boxes." + END)

    def test_external_file(self):
        op = "It ships as many boxes as listed in the file shipments.txt."
        self.assertInfeasible(START + op + END)
        self.assertNotInfeasible(START + op + END, files={"shipments.txt"})
        # каталог неизвестен — не утверждаем
        res = answer_math_question(START + op + END + TAIL)
        self.assertNotEqual(res.status, "infeasible")

    def test_unrelated_question(self):
        for q in ("What color are the boxes?", "Who is the manager of the warehouse?",
                  "In which city is the warehouse located?", "How much does one box weigh?"):
            self.assertInfeasible(START + "It ships 30 boxes. " + q)
        self.assertNotInfeasible(START + "It ships 30 boxes. How many boxes are left?")
        self.assertNotInfeasible(START + "It ships 30 boxes. How much does it cost if each box costs 2 dollars?")

    def test_no_question_is_not_claimed(self):
        self.assertEqual(ask(START + "It ships 30 boxes per day for 2 days.").status, "unparsed")


PEOPLE = {
    "person_anna_petrov.md": "# Anna Petrov\n\nBorn: 1980\nEmployer: Dune Robotics\nMentor: Boris Sato\n",
    "person_anna_lopez.md": "# Anna Lopez\n\nBorn: 1991\nEmployer: Ember Bio\nMentor: Boris Sato\n",
    "person_boris_sato.md": "# Boris Sato\n\nBorn: 1975\nEmployer: Dune Robotics\nMentor: Carl Wang\n",
    "person_carl_wang.md": "# Carl Wang\n\nBorn: 1975\nEmployer: Ember Bio\nMentor: Boris Sato\n",
    "person_william_ng.md": "# William Ng\n\nBorn: 1975\nEmployer: Ember Bio\nMentor: Boris Sato\n",
    "company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Osaka\nFounded: 2001\n",
    "company_ember_bio.md": "# Ember Bio\n\nHeadquarters: Lima\nFounded: 1999\n",
}
INTRO = "The directory kb/ contains a small knowledge base of markdown files. "


class SearchInfeasibleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        for name, text in PEOPLE.items():
            (root / name).write_text(text)
        cls.kb = load_kb(root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def probe(self, q):
        return infeasible_probe(self.kb, INTRO + q)

    def test_unknown_names_of_any_shape(self):
        for name in ("Omar", "Sean O'Neil", "Dr. Elena Rossi", "Jean-Luc Moreau", "Yuki Tanaka"):
            res = self.probe(f"In which year was {name} born?")
            self.assertIsNotNone(res, name)
            self.assertEqual(res.status, "no_anchor")

    def test_known_or_similar_names_are_not_flagged(self):
        # «Liam» — часть «William»: не утверждаем
        for name in ("Anna Petrov", "Petrov", "Liam", "Liam O'Brien", "Anna Petrova", "Boris"):
            self.assertIsNone(self.probe(f"In which year was {name} born?"), name)
        # вопрос не разбирается и с настоящим именем — не утверждаем
        self.assertIsNone(self.probe("How many siblings does Omar have?"))

    def test_ambiguous_first_name(self):
        res = self.probe("In which year was Anna born?")
        self.assertIsNotNone(res)
        self.assertEqual(res.status, "broken")
        # оба кандидата дают одинаковый ответ — неоднозначности нет
        self.assertIsNone(self.probe("Who is the mentor of Anna?"))

    def test_attribute_absent_from_kb_is_not_claimed(self):
        # правило «признака нет во всей базе» выключено: признак зовут синонимами («Remuneration:», «Tel:»)
        for q in ("What is Anna Petrov's salary?", "What is the phone number of Boris Sato?",
                  "Who is Carl Wang's spouse?", "What is Anna Petrov's favorite color?"):
            self.assertIsNone(self.probe(q), q)
        for q in ("What is Anna Petrov's birth year?", "Who is the CEO of Dune Robotics?",
                  "Who is Boris Sato's employer?", "What is Anna Petrov's age?"):
            self.assertIsNone(self.probe(q), q)

    def test_attribute_mentioned_somewhere_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in PEOPLE.items():
                (root / name).write_text(text)
            (root / "notes.md").write_text("# Payroll\n\nSalaries are listed in the HR system.\n")
            kb = load_kb(root)
            self.assertIsNone(infeasible_probe(kb, INTRO + "What is Anna Petrov's salary?"))

    def test_conflicting_lines_in_one_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in PEOPLE.items():
                (root / name).write_text(text)
            (root / "person_boris_sato.md").write_text(PEOPLE["person_boris_sato.md"] + "Born: 1979\n")
            # несколько записей в одном файле: повторяются все поля — это не противоречие
            (root / "people.md").write_text("Name: Ida Moss\nBorn: 1970\nEmployer: Ember Bio\n"
                                            "Name: Jon Moss\nBorn: 1972\nEmployer: Ember Bio\n")
            kb = load_kb(root)
            # внутри файла не утверждаем (бывает историей), но и сами не отвечаем — решает модель
            res = answer_path_question(kb, INTRO + "In which year was Boris Sato born?")
            self.assertEqual(res.status, "unparsed", res.detail)
            res = answer_path_question(kb, INTRO + "In which year was Anna Petrov born?")
            self.assertEqual((res.status, res.answer), ("ok", "1980"))
            # противоречие в другом поле той же записи не мешает
            res = answer_path_question(kb, INTRO + "Who is the mentor of Boris Sato?")
            self.assertEqual((res.status, res.answer), ("ok", "Carl Wang"))
            self.assertFalse(kb.field_conflicts.get("people.md"))

    def test_conflicting_duplicate_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in PEOPLE.items():
                (root / name).write_text(text)
            (root / "person_carl_wang_2.md").write_text("# Carl Wang\n\nBorn: 1982\nEmployer: Ember Bio\n")
            kb = load_kb(root)
            # противоречие между файлами: навык не отвечает и невыполнимость не утверждает — решает модель
            self.assertEqual(answer_path_question(kb, INTRO + "In which year was Carl Wang born?").status, "unparsed")
            res = answer_path_question(kb, INTRO + "In which city is the headquarters of Carl Wang's employer?")
            self.assertEqual((res.status, res.answer), ("ok", "Lima"))


class _SolverBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def solver(self, cls, question: str, files: dict | None = None):
        ws = self.root / "workspace"
        ws.mkdir(exist_ok=True)
        for rel, text in (files or {}).items():
            path = ws / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        task = Task(task_id="t1", task_type=cls.family, instruction=question, reward=1.0, dir=self.root,
                    workspace=ws, question=question)
        cfg = SimpleNamespace(fastpath=True, crosscheck=False, exec_timeout_s=10.0)
        env = SimpleNamespace(cfg=cfg, hub=None, estimator=None, scratch_root=self.root / "scratch", trace_text=False)
        s = cls(task, env)
        s.prepare()
        return s


CSV = "order_id,region,product,month,units,price\n1,east,widget,1,5,10\n2,west,gadget,2,7,20\n3,east,widget,3,2,10\n"


class SolverWiringTest(_SolverBase):
    def test_math_file_in_workspace_is_not_flagged(self):
        q = START + "It ships as many boxes as listed in the file shipments.txt." + END
        self.assertGreaterEqual(self.solver(MathSolver, q).p_infeasible_static or 0, 0.95)
        s = self.solver(MathSolver, q, files={"shipments.txt": "30\n"})
        self.assertLess(s.p_infeasible_static or 0, 0.95)

    def test_data_other_file_asked(self):
        q = "The file customers.csv has columns customer_id,name,city (3 rows). How many customers live in city 'Osaka'?"
        two = {"sales.csv": CSV, "stock.csv": "sku,qty\n1,5\n"}
        self.assertGreaterEqual(self.solver(DataSolver, q, files=two).p_infeasible_static, 0.95)
        # файл с тем же именем в другом формате — не утверждаем
        s = self.solver(DataSolver, q, files={**two, "customers.xlsx": "x"})
        self.assertLess(s.p_infeasible_static or 0, 0.95)

    def test_data_other_file_single_table(self):
        # таблица в каталоге одна — скорее всего, она и есть описанная (clients.csv с иными заголовками)
        q = "The file customers.csv has columns id,name,city,age (3 rows). What is the average age of customers in city 'Oslo'?"
        s = self.solver(DataSolver, q, files={"clients.csv": "customer_id,full_name,town,age_years\n1,a,Oslo,30\n"})
        self.assertLess(s.p_infeasible_static or 0, 0.95)

    def test_data_average_of_absent_value(self):
        q = ("The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "
             "What is the average price of product 'gizmo', rounded down to an integer?")
        # 0.9, а не выше: значение могут закодировать («E» вместо «east»)
        self.assertEqual(self.solver(DataSolver, q, files={"sales.csv": CSV}).p_infeasible_static, 0.9)
        # для суммы ответ 0 — задача выполнима
        q2 = ("The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "
              "How many units of product 'gizmo' were sold?")
        self.assertLess(self.solver(DataSolver, q2, files={"sales.csv": CSV}).p_infeasible_static or 0, 0.95)
        # значение есть в файле в другой форме — не утверждаем
        q3 = q.replace("'gizmo'", "'widgets'")
        self.assertLess(self.solver(DataSolver, q3, files={"sales.csv": CSV}).p_infeasible_static or 0, 0.95)

    def test_data_column_removed(self):
        q = ("The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "
             "How many units of product 'widget' were sold in months 1 through 3 inclusive?")
        no_month = "order_id,region,product,units,price\n1,east,widget,5,10\n2,west,gadget,7,20\n"
        self.assertEqual(self.solver(DataSolver, q, files={"sales.csv": no_month}).p_infeasible_static, 0.9)
        # столбец переименован (в файле есть столбец, которого нет в условии) — не утверждаем
        renamed = no_month.replace("order_id,region,product,units", "order_id,region,product,period,units")
        renamed = renamed.replace(",widget,", ",widget,1,").replace(",gadget,", ",gadget,2,")
        self.assertLess(self.solver(DataSolver, q, files={"sales.csv": renamed}).p_infeasible_static or 0, 0.95)

    def test_data_placeholder_column(self):
        q = ("The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "
             "How many units of product 'widget' were sold?")
        bad = CSV.replace(",5,10", ",n/a,10").replace(",7,20", ",?,20").replace(",2,10", ",many,10")
        self.assertGreaterEqual(self.solver(DataSolver, q, files={"sales.csv": bad}).p_infeasible_static, 0.95)
        # часть ячеек с числами — данные есть, решает модель
        some = CSV.replace(",5,10", ",n/a,10")
        self.assertLess(self.solver(DataSolver, q, files={"sales.csv": some}).p_infeasible_static or 0, 0.95)

    def test_data_binary_garbage(self):
        q = ("The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "
             "How many units of product 'widget' were sold?")
        ws = self.root / "workspace"
        ws.mkdir(exist_ok=True)
        (ws / "sales.csv").write_bytes(b"\x00\x01\x02PK\x03\x04 corrupted \x7f\x00\x00" * 20)
        self.assertGreaterEqual(self.solver(DataSolver, q).p_infeasible_static, 0.95)
        # тот же CSV в UTF-16 (нули через байт) — не мусор
        (ws / "sales.csv").write_bytes(CSV.encode("utf-16-le"))
        self.assertLess(self.solver(DataSolver, q).p_infeasible_static or 0, 0.95)
        # gzip под видом csv — модель может распаковать
        (ws / "sales.csv").write_bytes(b"\x1f\x8b\x08\x00" + b"\x00\x01" * 50)
        self.assertLess(self.solver(DataSolver, q).p_infeasible_static or 0, 0.95)

    def test_search_wiring(self):
        files = {"kb/" + k: v for k, v in PEOPLE.items()}
        s = self.solver(SearchSolver, INTRO + "In which year was Dr. Elena Rossi born?", files=files)
        self.assertGreaterEqual(s.p_infeasible_static, 0.95)
        s = self.solver(SearchSolver, INTRO + "In which year was Boris Sato born?", files=files)
        self.assertEqual(s.best, "1975")


class ReviewFalsePositivesTest(_SolverBase):
    """Выполнимые задачи, которые правила отсева помечали невыполнимыми (p ≥ 0.95): так быть не должно."""

    BASE = "# Irina Smith\n\nBorn: 1983\nEmployer: Dune Robotics\nMentor: Alice Smith\n"
    KB = {
        "kb/person_irina_smith.md": BASE,
        "kb/person_alice_smith.md": "# Alice Smith\n\nBorn: 1960\nEmployer: Ember Bio\n",
        "kb/company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Hanoi\nFounded: 1989\n",
        "kb/company_ember_bio.md": "# Ember Bio\n\nHeadquarters: Lima\nFounded: 2001\n",
    }

    def search(self, q, extra=None):
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        return self.solver(SearchSolver, INTRO + q, files={**self.KB, **(extra or {})})

    def assertNotClaimed(self, s, label):
        self.assertLess(s.p_infeasible_static or 0, 0.95, (label, s.static_reason))

    # 1. признак под синонимом или в прозе
    def test_attribute_synonyms(self):
        k = "kb/person_irina_smith.md"
        for q, extra in (("What is Irina Smith's email?", "Contact: irina@dune.io\n"),
                         ("What is Irina Smith's email?", "\nYou can reach her at irina@dune.io.\n"),
                         ("What is Irina Smith's phone number?", "Tel: +1 555 0100\n"),
                         ("What is Irina Smith's address?", "Residence: 12 Harbour Rd, Lisbon\n"),
                         ("What is Irina Smith's salary?", "Remuneration: 95000 USD\n"),
                         ("Who is Irina Smith's spouse?", "Significant other: Hiro Sato\n"),
                         ("What is Irina Smith's favorite color?", "Preferred colour: green\n"),
                         ("What is Irina Smith's height?", "Stature: 172 cm\n"),
                         ("What is Irina Smith's hobby?", "Interests: chess\n")):
            self.assertNotClaimed(self.search(q, {k: self.BASE + extra}), q)
        self.assertNotClaimed(self.search("What is the address of Dune Robotics?"), "address/Headquarters")

    # 2. разные записи одного значения — не противоречие; навык отвечает, как до отсева
    def test_compatible_values_are_answered(self):
        bio = "kb/notes/irina_smith_bio.md"
        year = "In which year was Irina Smith born?"
        hq = "In which city is the headquarters of Irina Smith's employer?"
        for q, extra, want in ((year, {bio: "# Irina Smith\n\nBorn: March 4, 1983\n"}, "1983"),
                               (year, {bio: "# Irina Smith\n\nBorn: 1983-03-04\n"}, "1983"),
                               (year, {bio: "# Irina Smith\n\nBorn: 1983 (Moscow)\n"}, "1983"),
                               (hq, {bio: "# Irina Smith\n\nEmployer: Dune Robotics Inc.\n"}, "Hanoi"),
                               (hq, {bio: "# Irina Smith\n\nEmployer: [[Dune Robotics]]\n"}, "Hanoi")):
            s = self.search(q, extra)
            self.assertNotClaimed(s, extra)
            self.assertEqual(s.best, want, extra)
        k = "kb/person_irina_smith.md"
        s = self.search(year, {k: self.BASE + "Born: Moscow\n"})
        self.assertEqual(s.best, "1983")
        s = self.search("In which year was Irina Smith's mentor born?", {k: self.BASE + "Mentor: Alice Smith (until 2015)\n"})
        self.assertEqual(s.best, "1960")

    def test_real_conflicts_are_not_answered(self):
        year = "In which year was Irina Smith born?"
        # второй файл записи с другим годом — противоречие, навык не отвечает
        s = self.search(year, {"kb/person_irina_smith_2.md": "# Irina Smith\n\nBorn: 1990\n"})
        self.assertIsNone(s.best)
        # то же внутри файла: не отвечаем, но и не утверждаем невыполнимость — решает модель
        s = self.search(year, {"kb/person_irina_smith.md": self.BASE + "Born: 1988\n"})
        self.assertIsNone(s.best)
        self.assertNotClaimed(s, "conflict inside one file")

    # 7. имя без диакритики / апострофа / другой транслитерацией
    def test_transliterated_names(self):
        cases = (("kb/p99.md", "# José García\n\nBorn: 1975\nEmployer: Dune Robotics\n", "Jose Garcia", "1975"),
                 ("kb/p98.md", "# Lena Müller\n\nBorn: 1990\nEmployer: Ember Bio\n", "Lena Muller", "1990"),
                 ("kb/p98.md", "# Lena Müller\n\nBorn: 1990\nEmployer: Ember Bio\n", "Lena Mueller", "1990"),
                 ("kb/p97.md", "# Zoë Ångström\n\nBorn: 1966\nEmployer: Ember Bio\n", "Zoe Angstrom", "1966"),
                 ("kb/p96.md", "# Seán O'Brien\n\nBorn: 1960\nEmployer: Ember Bio\n", "Sean OBrien", "1960"))
        for path, text, name, want in cases:
            s = self.search(f"In which year was {name} born?", {path: text})
            self.assertEqual(s.best, want, name)
        s = self.search("In which city is the headquarters of Zoe Angstrom's employer?", {"kb/p97.md": cases[3][1]})
        self.assertEqual(s.best, "Lima")
        # другая транслитерация: не находим, но и не утверждаем уверенно
        s = self.search("In which year was Hsu Ming-yang born?", {"kb/p95.md": "# Xu Mingyang\n\nBorn: 1985\n"})
        self.assertNotClaimed(s, "Hsu Ming-yang")
        # две записи с одинаковым именем без диакритики — не подставляем
        s = self.search("In which year was Jose Garcia born?", {"kb/p1.md": "# José García\n\nBorn: 1975\n",
                                                                 "kb/p2.md": "# Jose Garcia\n\nBorn: 1980\n"})
        self.assertEqual(s.best, "1980")
        # совсем чужое имя по-прежнему невыполнимо
        self.assertGreaterEqual(self.search("In which year was Zed Quill born?").p_infeasible_static or 0, 0.95)

    # 3, 4, 6 — таблицы
    def data(self, q, files):
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        return self.solver(DataSolver, q, files=files)

    def test_placeholder_column_count_and_filter(self):
        head = "The file sales.csv has columns order_id,region,product,month,units,price,{c} ({n} rows). "
        coupon = "order_id,region,product,month,units,price,coupon\n1,east,widget,1,5,10,\n2,west,gadget,2,7,20,\n3,east,widget,3,2,10,n/a\n"
        for q in ("How many orders have no coupon?", "How many orders used a coupon?"):
            self.assertNotClaimed(self.data(head.format(c="coupon", n=3) + q, {"sales.csv": coupon}), q)
        disc = coupon.replace("coupon", "discount").replace(",\n", ",-\n").replace(",n/a\n", ",-\n")
        q = head.format(c="discount", n=3) + "How many units were sold without a discount?"
        self.assertNotClaimed(self.data(q, {"sales.csv": disc}), q)
        demand = ("order_id,region,product,month,units,price,demand\n1,east,widget,1,5,10,many\n2,west,gadget,2,7,20,few\n"
                  "3,east,widget,3,2,10,n/a\n4,east,widget,3,2,10,n/a\n")
        q = head.format(c="demand", n=4) + "How many orders have demand many?"
        self.assertNotClaimed(self.data(q, {"sales.csv": demand}), q)
        # агрегация столбца из одних заглушек — по-прежнему невыполнима
        bad = coupon.replace(",5,10,", ",5,n/a,").replace(",7,20,", ",7,?,").replace(",2,10,", ",2,,")
        q = head.format(c="coupon", n=3) + "What is the average price of product 'widget'?"
        self.assertGreaterEqual(self.data(q, {"sales.csv": bad}).p_infeasible_static or 0, 0.95)

    def test_value_spelled_differently(self):
        csv = ("order_id,region,product,month,units,price\n1,east,Widget Pro,1,5,10\n2,west,gadget,2,7,20\n"
               "3,east,Widget Pro,3,2,10\n4,north,gizmo,3,1,30\n")
        head = "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). "
        for q in ("What is the average price of product 'widget_pro'?", "What is the average price of product 'widget-pro'?",
                  "What is the average price of product 'WidgetPro'?", "What is the average price in region 'Northern'?"):
            self.assertNotClaimed(self.data(head + q, {"sales.csv": csv}), q)
        q = head + "What is the average price of product 'sprocket'?"
        self.assertEqual(self.data(q, {"sales.csv": csv}).p_infeasible_static, 0.9)

    def test_other_file_with_one_table(self):
        q = "The file customers.csv has columns id,name,city,age (3 rows). What is the average age in city 'Oslo'?"
        for files in ({"clients.csv": "customer_id,full_name,town,age_years\n1,a,Oslo,30\n"},
                      {"customer_data.csv": "ID,Name,City,Age\n1,a,Oslo,30\n"}):
            self.assertNotClaimed(self.data(q, files), files)


class MathFileReferenceTest(unittest.TestCase):
    # 5. файл в условии — лишь ссылка, число есть в самом предложении
    def test_number_given_with_file_reference(self):
        for text in (START + "It ships 30 boxes, as listed in orders.csv." + END,
                     START + "It ships 30 boxes (see orders.csv)." + END,
                     START + "It ships 30 boxes, recorded in log.txt." + END,
                     START + "It ships 30 boxes. It ships the 12 boxes listed in orders.csv." + END):
            self.assertNotEqual(ask(text).status, "infeasible", text)
        res = ask(START + "It ships 30 boxes. It receives as many boxes as listed in deliveries.txt." + END)
        self.assertEqual(res.status, "infeasible")


class ReviewRound2Test(_SolverBase):
    """Второе ревью: сомнительное — не выше 0.9, модель делает попытку."""

    BASE, KB = ReviewFalsePositivesTest.BASE, ReviewFalsePositivesTest.KB
    search, data = ReviewFalsePositivesTest.search, ReviewFalsePositivesTest.data
    assertNotClaimed = ReviewFalsePositivesTest.assertNotClaimed

    def test_aggregated_column_with_formatted_numbers(self):
        head = "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). "
        base = "order_id,region,product,month,units,price\n"
        for cells in (("$10.50", "N/A", "$8.00", "N/A"), ("ten", "n/a", "twenty", "n/a"), ("10 USD", "N/A", "N/A", "N/A"),
                      ("(10)", "?", "?", "?"), ("10%", "?", "?", "?")):
            csv = base + "".join(f"{i},east,widget,1,5,{c}\n" for i, c in enumerate(cells, 1))
            self.assertNotClaimed(self.data(head + "What is the average price?", {"sales.csv": csv}), cells)

    def test_column_in_neighbour_table(self):
        q = ("The file sales.csv has columns order_id,region,product,month,units,price (2 rows). "
             "What is the average price in region 'east'?")
        no_price = "order_id,region,product,month,units\n1,east,widget,1,5\n2,west,gadget,2,7\n"
        s = self.data(q, {"sales.csv": no_price, "prices.csv": "product,price\nwidget,3\ngadget,4\n"})
        self.assertNotClaimed(s, "price in prices.csv")
        self.assertNotClaimed(self.data(q, {"sales.csv": no_price}), "price removed")

    def test_coded_filter_values(self):
        head = "The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "
        csv = "order_id,region,product,month,units,price\n1,E,widget,1,5,10\n2,W,gadget,2,7,20\n3,E,widget,3,2,10\n"
        self.assertNotClaimed(self.data(head + "What is the average price in region 'east'?", {"sales.csv": csv}), "E/W")

    def test_described_file_with_renamed_headers(self):
        q = "The file orders.csv has columns order_id,customer,amount (1 rows). What is the total amount for customer 'bob'?"
        s = self.data(q, {"data.csv": "id,client,total\n1,bob,5\n", "x.csv": "a,b\n1,2\n"})
        self.assertNotClaimed(s, "data.csv")

    def test_utf32_without_bom_is_not_garbage(self):
        head = "The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "
        ws = self.root / "workspace"
        ws.mkdir(exist_ok=True)
        (ws / "sales.csv").write_bytes(CSV.encode("utf-32-le"))
        s = self.solver(DataSolver, head + "How many units of product 'widget' were sold?")
        self.assertNotClaimed(s, "utf-32-le")

    def test_nicknames_and_company_suffix(self):
        kb = {"kb/p_rl.md": "# Robert Lee\n\nBorn: 1970\nEmployer: Ember Bio\n",
              "kb/p_wn.md": "# William Ng\n\nBorn: 1971\nEmployer: Ember Bio\n",
              "kb/p_ef.md": "# Elizabeth Ford\n\nBorn: 1972\nEmployer: Ember Bio\n",
              "kb/c_hm.md": "# Helix Motors\n\nHeadquarters: Izmir\nFounded: 1990\n"}
        for q in ("In which year was Bob Lee born?", "In which year was Bobby Lee born?", "In which year was Bill Ng born?",
                  "In which year was Liz Ford born?", "In which year was Irene Smith born?",
                  "In which city is Helix Motor Company headquartered?"):
            self.assertNotClaimed(self.search(q, kb), q)

    def test_duplicate_files(self):
        notes = {"kb/notes_irina.md": "# Irina Smith\n\nEmployer: Ember Bio (former)\n"}
        s = self.search("In which city is the headquarters of Irina Smith's employer?", notes)
        self.assertNotClaimed(s, "former")
        self.assertIsNone(s.best)
        # первый файл записи без поля, второй — с ним
        s = self.search("Who is Irina Smith's mentor?", {"kb/a_irina_notes.md": "# Irina Smith\n\nBorn: 1983\n"})
        self.assertNotClaimed(s, "field only in second file")

    def test_unrelated_question_with_how_many(self):
        text = START + "It ships 30 boxes. It receives 40 boxes. Who counts the boxes, and how many does it have at the end?"
        self.assertNotEqual(ask(text).status, "infeasible")
        self.assertEqual(ask(START + "It ships 30 boxes. Who ships the boxes?").status, "infeasible")


if __name__ == "__main__":
    unittest.main()
