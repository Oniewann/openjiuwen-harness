"""Настройки харнесса.

Обязательные переменные окружения задаёт организатор: LLM_BASE_URL и LLM_API_KEY.
Всё остальное имеет разумные значения по умолчанию и может быть переопределено
переменными HARNESS_* (нужно только для экспериментов; в боевом запуске их нет).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Четыре модели, разрешённые регламентом.
MODEL_27B = "qwen3.8-27b"
MODEL_27B_NOREASON = "qwen3.8-27b-noreason"
MODEL_35B = "qwen3.6-35b-a3b"
MODEL_35B_NOREASON = "qwen3.6-35b-a3b-noreason"
ALLOWED_MODELS = (MODEL_27B, MODEL_27B_NOREASON, MODEL_35B, MODEL_35B_NOREASON)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))  # type: ignore[arg-type]
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Config:
    # --- доступ к моделям (только через прокси организаторов) ---
    base_url: str = ""
    api_key: str = ""

    # --- роли моделей ---
    model_fast: str = MODEL_27B_NOREASON        # дешёвая попытка: без рассуждений
    model_fast_alt: str = MODEL_35B_NOREASON    # второе мнение: другая модель, тоже без рассуждений
    model_strong: str = MODEL_27B               # эскалация: с рассуждениями
    single_model: Optional[str] = None          # если задано, все вызовы идут на одну модель

    # --- политика решения ---
    # fastpath: детерминированный навык отвечает сам (0 токенов), если уверенно разобрал задачу.
    #   Организаторы разрешили это явно: «Можно решить все [без LLM], если получится».
    # crosscheck: при выключенном fastpath ответ модели бесплатно сверяется с навыком;
    #   расхождение ведёт к эскалации.
    fastpath: bool = True
    crosscheck: bool = True
    # Платная перепроверка ответа второй дешёвой моделью, когда бесплатной сверки нет:
    # never | auto (только при сомнительных признаках) | always.
    # Баллы важнее токенов, а таких задач при включённых навыках единицы — проверяем всегда.
    verify: str = "always"
    enable_react: bool = True                   # запасной ReAct-агент openJiuwen

    # --- лимиты ---
    max_parallel: int = 5                       # регламент: до 5 сессий одновременно
    time_limit_s: float = 55 * 60               # жёсткий предел прогона 1 час; останавливаемся раньше
    rpm_limit: int = 55                         # запросов в минуту (у тарифа разработки лимит 60)
    budget: Optional[float] = None              # бюджет в выходных токенах из переменной BUDGET
    assumed_budget: float = 100_000.0           # ориентир организаторов, если BUDGET не передали
    budget_reserve: float = 0.03                # доля бюджета, которую не трогаем
    answer_in_workspace: bool = False           # дублировать answer.txt в workspace/ (организаторы: «рядом» с task.json)
    # Платная эскалация запускается, только если оценка шанса на успех не ниже порога.
    min_p_escalate: float = 0.15
    # «Курс обмена»: сколько выходных токенов мы готовы отдать за один ожидаемый балл.
    # Шаг запускается, если шанс × баллы задачи × курс >= ожидаемая стоимость шага.
    tokens_per_point: float = 2000.0
    task_token_cap: float = 6000.0              # потолок расхода на одну задачу (выходных токенов)
    react_token_cap: float = 3500.0             # потолок на один запуск ReAct
    react_max_iterations: int = 8

    # --- вызовы модели ---
    call_timeout_fast: float = 90.0
    call_timeout_strong: float = 300.0
    max_tokens_fast: int = 400
    max_tokens_code: int = 900
    max_tokens_strong: int = 6000
    reasoning_in_completion: bool = True        # считает ли API рассуждения внутри completion_tokens
    dead_after_failures: int = 6                # столько сбоев подряд — прокси, похоже, замолчал: ставим паузы
    give_up_after_s: float = 240.0              # столько секунд непрерывных сбоев — считаем бюджет исчерпанным

    # --- исполнение кода ---
    exec_timeout_s: float = 20.0
    test_timeout_s: float = 60.0

    # --- отладка ---
    trace_dir: Optional[str] = None             # куда писать траектории (по умолчанию /tmp)
    trace_text: bool = False                    # сохранять тексты запросов и ответов (только локально)
    verbose: bool = False

    extra: dict = field(default_factory=dict)

    def model_for(self, role: str) -> str:
        """Имя модели для роли с учётом режима одной модели."""
        if self.single_model:
            return self.single_model
        return {"fast": self.model_fast, "fast_alt": self.model_fast_alt, "strong": self.model_strong}[role]


def load_config() -> Config:
    cfg = Config()
    cfg.base_url = (_env("LLM_BASE_URL", "") or "").rstrip("/")
    cfg.api_key = _env("LLM_API_KEY", "") or ""

    cfg.model_fast = _env("HARNESS_MODEL_FAST", cfg.model_fast)  # type: ignore[assignment]
    cfg.model_fast_alt = _env("HARNESS_MODEL_FAST_ALT", cfg.model_fast_alt)  # type: ignore[assignment]
    cfg.model_strong = _env("HARNESS_MODEL_STRONG", cfg.model_strong)  # type: ignore[assignment]
    cfg.single_model = _env("HARNESS_SINGLE_MODEL")

    cfg.fastpath = _env_bool("HARNESS_FASTPATH", cfg.fastpath)
    cfg.crosscheck = _env_bool("HARNESS_CROSSCHECK", cfg.crosscheck)
    cfg.verify = (_env("HARNESS_VERIFY", cfg.verify) or cfg.verify).lower()
    cfg.enable_react = _env_bool("HARNESS_REACT", cfg.enable_react)

    cfg.max_parallel = max(1, min(5, _env_int("HARNESS_PARALLEL", cfg.max_parallel)))
    cfg.time_limit_s = float(_env_float("HARNESS_TIME_LIMIT", cfg.time_limit_s) or cfg.time_limit_s)
    cfg.rpm_limit = _env_int("HARNESS_RPM", cfg.rpm_limit)
    # Организаторы: «Бюджет будет передан в переменной окружения BUDGET — например BUDGET=100000».
    for name in ("BUDGET", "HARNESS_TOKEN_BUDGET", "TOKEN_BUDGET", "LLM_TOKEN_BUDGET", "BUDGET_TOKENS"):
        value = _env_float(name, None)
        if value and value > 0:
            cfg.budget = value
            break
    cfg.min_p_escalate = float(_env_float("HARNESS_MIN_P_ESCALATE", cfg.min_p_escalate) or 0.0)
    cfg.tokens_per_point = float(_env_float("HARNESS_TOKENS_PER_POINT", cfg.tokens_per_point) or cfg.tokens_per_point)
    cfg.task_token_cap = float(_env_float("HARNESS_TASK_CAP", cfg.task_token_cap) or cfg.task_token_cap)
    cfg.react_token_cap = float(_env_float("HARNESS_REACT_CAP", cfg.react_token_cap) or cfg.react_token_cap)
    cfg.reasoning_in_completion = _env_bool("HARNESS_REASONING_IN_COMPLETION", cfg.reasoning_in_completion)
    cfg.give_up_after_s = float(_env_float("HARNESS_GIVE_UP_AFTER", cfg.give_up_after_s) or cfg.give_up_after_s)
    cfg.call_timeout_fast = float(_env_float("HARNESS_CALL_TIMEOUT_FAST", cfg.call_timeout_fast) or cfg.call_timeout_fast)
    cfg.call_timeout_strong = float(_env_float("HARNESS_CALL_TIMEOUT_STRONG", cfg.call_timeout_strong)
                                    or cfg.call_timeout_strong)

    cfg.trace_dir = _env("HARNESS_TRACE_DIR")
    cfg.trace_text = _env_bool("HARNESS_TRACE_TEXT", False)
    cfg.verbose = _env_bool("HARNESS_VERBOSE", False)
    cfg.extra["verify_ssl"] = _env_bool("HARNESS_VERIFY_SSL", True)
    cfg.answer_in_workspace = _env_bool("HARNESS_ANSWER_IN_WORKSPACE", cfg.answer_in_workspace)
    return cfg
