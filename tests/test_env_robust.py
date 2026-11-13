"""Устойчивость запуска run_agent.py в нештатном окружении (без сети и без модели).

Проверяется: пустой/несуществующий каталог задач, каталог только для чтения, адрес прокси
в разных видах (без /v1, с /v1/, недоступный порт), SIGTERM посреди прогона.
Задачи берутся копиями из ../task-dir (эталон не трогаем); если эталона нет — тесты пропускаются.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from harness.answers import write_answer
from harness.config import load_config
from harness.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT = ROOT / "run_agent.py"
TASK_DIR = ROOT.parent / "task-dir"
# Задачи, которые навыки решают без модели (0 токенов): по одной на тип.
OFFLINE = ("math_0000", "data_analysis_0002", "search_0001")


def _dead_port() -> int:
    """Порт, на котором точно никто не слушает (занимаем и сразу освобождаем)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(base_url: str, trace: Path, **extra) -> dict:
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
    env.update({"LLM_BASE_URL": base_url, "LLM_API_KEY": "test-key", "HARNESS_TRACE_DIR": str(trace)})
    env.update(extra)
    return env


def _copy_tasks(dst: Path, names=OFFLINE) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copytree(TASK_DIR / name, dst / name)
    return dst


def _answers(root: Path) -> dict:
    return {p.parent.name: p.read_text(encoding="utf-8").strip() for p in root.glob("*/answer.txt")}


@unittest.skipUnless(TASK_DIR.is_dir(), "нет эталона ../task-dir")
class EnvRobust(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="agJ-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        for p in self.tmp.rglob("*"):  # на случай каталога только для чтения
            try:
                p.chmod(p.stat().st_mode | 0o700)
            except OSError:
                pass
        self._tmp.cleanup()

    def run_agent(self, tasks: Path, base_url: str, timeout: float = 40, **extra):
        started = time.monotonic()
        proc = subprocess.run([sys.executable, str(RUN_AGENT), "--tasks-dir", str(tasks)],
                              env=_env(base_url, self.tmp / "trace", **extra), capture_output=True,
                              text=True, timeout=timeout, cwd=str(self.tmp))
        return proc, time.monotonic() - started

    # --- каталог задач ---------------------------------------------------------------------

    def test_empty_and_missing_tasks_dir(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        for path in (empty, self.tmp / "no-such-dir"):
            proc, spent = self.run_agent(path, f"http://127.0.0.1:{_dead_port()}/v1")
            self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
            self.assertIn("no tasks found", proc.stdout)
            self.assertLess(spent, 15)

    def test_readonly_tasks_dir(self):
        tasks = _copy_tasks(self.tmp / "ro", ("math_0000",))
        for p in [*tasks.rglob("*"), tasks]:
            p.chmod(p.stat().st_mode & ~0o222)
        if os.access(tasks / "math_0000", os.W_OK):
            self.skipTest("запись не запрещается (запуск от root?)")
        proc, spent = self.run_agent(tasks, f"http://127.0.0.1:{_dead_port()}/v1")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("cannot write answer.txt", proc.stdout)  # ошибка записи видна в логе
        self.assertFalse((tasks / "math_0000" / "answer.txt").exists())
        self.assertNotIn("Traceback", proc.stderr)
        self.assertLess(spent, 15)

    def test_write_answer_readonly_returns_false(self):
        tasks = _copy_tasks(self.tmp / "ro2", ("math_0000",))
        task = load_tasks(tasks)[0]
        os.chmod(task.dir, 0o555)
        if os.access(task.dir, os.W_OK):
            self.skipTest("запись не запрещается (запуск от root?)")
        self.assertFalse(write_answer(task, "42"))
        os.chmod(task.dir, 0o755)
        self.assertTrue(write_answer(task, "42"))
        self.assertEqual((task.dir / "answer.txt").read_text(), "42\n")

    # --- адрес прокси ----------------------------------------------------------------------

    def test_base_url_variants_unreachable_proxy(self):
        """Прокси недоступен: ответы навыков должны появиться быстро, а процесс — не висеть."""
        port = _dead_port()
        for i, url in enumerate((f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port}/v1",
                                 f"http://127.0.0.1:{port}/v1/")):
            tasks = _copy_tasks(self.tmp / f"u{i}")
            proc, spent = self.run_agent(tasks, url)
            self.assertEqual(proc.returncode, 0, f"{url}: {proc.stderr[-500:]}")
            got = _answers(tasks)
            self.assertEqual(set(got), set(OFFLINE), f"{url}: {got}")
            self.assertTrue(all(got.values()))
            self.assertLess(spent, 15, url)

    def test_base_url_normalized_by_config(self):
        keep = {k: os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY")}
        try:
            for raw, want in (("http://h:1/v1/", "http://h:1/v1"), ("http://h:1", "http://h:1"),
                              ("http://h:1/v1//", "http://h:1/v1"), ("", "")):
                os.environ["LLM_BASE_URL"] = raw
                self.assertEqual(load_config().base_url, want)
            os.environ.pop("LLM_BASE_URL")
            self.assertEqual(load_config().base_url, "")  # переменной нет — не падаем
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # --- сигналы ---------------------------------------------------------------------------

    def test_sigterm_keeps_written_answers(self):
        """Лишняя задача требует модели (прокси мёртв) — процесс «занят»; SIGTERM не стирает готовые ответы."""
        tasks = _copy_tasks(self.tmp / "term")
        extra = tasks / "zz_0001"
        extra.mkdir()
        (extra / "task.json").write_text(
            '{"task_id":"zz_0001","task_type":"math","reward":1,'
            '"instruction":"Write a haiku about autumn and count its syllables."}', encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(RUN_AGENT), "--tasks-dir", str(tasks)],
                                env=_env(f"http://127.0.0.1:{_dead_port()}/v1", self.tmp / "trace"),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(self.tmp))
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and len(_answers(tasks)) < len(OFFLINE):
                time.sleep(0.1)
            self.assertEqual(len(_answers(tasks)), len(OFFLINE), "ответы навыков не появились")
            time.sleep(0.5)
            before = _answers(tasks)
            sent = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            out, _ = proc.communicate(timeout=15)
            self.assertLess(time.monotonic() - sent, 10, "процесс завершился слишком долго после SIGTERM")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(proc.returncode, 0, out[-500:])
        self.assertEqual(_answers(tasks), before)
        self.assertFalse(list(tasks.glob("*/.answer.txt.tmp*")), "остался временный файл записи")
        self.assertNotIn("Traceback", out)


if __name__ == "__main__":
    unittest.main()
