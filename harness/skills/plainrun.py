"""Запуск тестов-функций (`def test_x(): assert ...`) без pytest.

Нужен, когда тестовый файл состоит из голых функций и сам себя не запускает: команда из условия
(`python3 test_mylib.py`) тогда завершается успешно, не выполнив ни одного теста. Скрипт импортирует
тестовые модули, вызывает функции test_* и печатает отчёт в формате unittest, который уже умеет
разбирать testrun.py. Запускается отдельным процессом в каталоге задачи:

    python plainrun.py test_mylib.py [ещё файлы]
"""
import importlib.util
import os
import sys
import time
import traceback


def main(argv: list) -> int:
    sys.path.insert(0, os.getcwd())
    started = time.time()
    total, failures, errors = 0, 0, 0
    for path in argv:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except BaseException:  # noqa: BLE001 — ошибка импорта теста: отчитываемся и идём дальше
            total += 1
            errors += 1
            _report("ERROR", f"import_{name}", name)
            continue
        for attr in sorted(vars(module)):
            fn = getattr(module, attr)
            if not attr.startswith("test") or not callable(fn) or isinstance(fn, type):
                continue
            total += 1
            try:
                fn()
            except AssertionError:
                failures += 1
                _report("FAIL", attr, name)
            except BaseException:  # noqa: BLE001
                errors += 1
                _report("ERROR", attr, name)
    sys.stderr.write("-" * 70 + "\n")
    sys.stderr.write(f"Ran {total} test{'s' if total != 1 else ''} in {time.time() - started:.3f}s\n\n")
    if failures or errors:
        parts = ([f"failures={failures}"] if failures else []) + ([f"errors={errors}"] if errors else [])
        sys.stderr.write(f"FAILED ({', '.join(parts)})\n")
        return 1
    sys.stderr.write("OK\n")
    return 0


def _report(kind: str, test: str, module: str) -> None:
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f"{kind}: {test} ({module})\n")
    sys.stderr.write("-" * 70 + "\n")
    sys.stderr.write(traceback.format_exc() + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
