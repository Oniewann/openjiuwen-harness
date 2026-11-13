"""Оценка стоимости шага и шанса на успех ДО его запуска.

Стоимость шага (в выходных токенах регламента) складывается из двух частей:

    вход / 10   — известен почти точно: промпт уже собран, остаётся оценить число токенов
                  (грубый счётчик × поправочный коэффициент, который уточняется на ходу
                  по фактическому usage первых же вызовов);
    выход       — линейная модель по типу задачи, ступени и простым признакам
                  (число чисел в условии, число упавших тестов, длина цепочки ссылок).
                  Априорные коэффициенты (DEFAULT_MODEL ниже) подобраны вручную по
                  траекториям прогонов; во время прогона оценка умножается на скользящую
                  поправку «факт / прогноз».

Модель саму себя про стоимость не спрашиваем: это стоит токенов и плохо предсказывает
(см. README, раздел про оценку расхода).

Шанс на успех — байесовская оценка. Априорно выполнимо 80% задач (регламент: 15–25%
невыполнимых). Каждая неудачная попытка и каждый отказ модели сдвигают вероятность
«задача выполнима» вниз; на эту вероятность умножается доля успехов ступени.
Планировщик не запускает дорогую эскалацию, если шанс ниже порога.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ledger import CallRecord

# Необязательный файл поправок к DEFAULT_MODEL; в репозитории его нет, используются значения ниже.
_MODEL_FILE = Path(__file__).with_name("cost_model.json")

# Априорные значения: подобраны вручную по траекториям прогонов, уточняются по факту в ходе прогона.
DEFAULT_MODEL = {
    "version": 1,
    "source": "hand-tuned priors from run trajectories",
    "prompt_ratio": 1.15,           # фактические входные токены / грубая оценка
    "feasible_prior": 0.8,
    # выходные токены ответа без рассуждений: base + сумма coef[признак] * значение
    "completion": {
        "math": {"base": 14.0, "numbers": 3.0},
        "data_analysis": {"base": 32.0, "filters": 6.0},
        "search": {"base": 10.0, "hops": 9.0},
        "coding": {"base": 20.0, "failing": 30.0},
        "generic": {"base": 80.0},
    },
    # сколько токенов рассуждений добавляет модель с reasoning
    "reasoning": {"math": 450.0, "data_analysis": 500.0, "search": 400.0, "coding": 1500.0, "generic": 900.0},
    # полная стоимость одного запуска запасного ReAct-агента
    "react": {"math": 900.0, "data_analysis": 1500.0, "search": 1800.0, "coding": 2500.0, "generic": 2500.0},
    # доля задач, которые ступень решает, если задача выполнима и предыдущие ступени не справились
    "solve": {"fast": 0.9, "fast_retry": 0.5, "fast_alt": 0.5, "strong": 0.7, "react": 0.5},
    "refuse_given_infeasible": 0.7,
    "refuse_given_feasible": 0.03,
    "safety_factor": 1.3,           # запас при проверке «влезает ли шаг в бюджет» (квантиль ~80%)
}


@dataclass
class Evidence:
    """Что уже известно о задаче к моменту оценки."""
    p_infeasible_static: Optional[float] = None   # вывод статической проверки (0 токенов)
    failed_steps: list[str] = field(default_factory=list)   # ступени, не давшие проверенного ответа
    refusals: int = 0                                        # сколько раз модель сказала «невыполнимо»


class Estimator:
    def __init__(self, model_path: Optional[Path] = None):
        self.model = json.loads(json.dumps(DEFAULT_MODEL))
        path = model_path or _MODEL_FILE
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(self.model.get(key), dict):
                    for sub, subvalue in value.items():
                        if isinstance(subvalue, dict) and isinstance(self.model[key].get(sub), dict):
                            self.model[key][sub].update(subvalue)
                        else:
                            self.model[key][sub] = subvalue
                else:
                    self.model[key] = value
        except (OSError, ValueError):
            pass
        self.prompt_ratio: float = float(self.model["prompt_ratio"])
        self._prompt_obs = 0
        self._completion_ratio: dict[tuple[str, str], float] = {}
        self._completion_obs: dict[tuple[str, str], int] = {}
        self._outcomes: dict[tuple[str, str], list[int]] = {}   # (семейство, ступень) -> [успехов, попыток]
        self._predicted: dict[tuple[str, str, str], float] = {}  # (задача, ступень, модель) -> прогноз выхода

    # ------------------------------------------------------------------ стоимость
    def _family(self, family: str) -> str:
        return family if family in self.model["completion"] else "generic"

    def _base_answer_tokens(self, family: str, step: str, features: Optional[dict] = None) -> float:
        """Прогноз длины ответа линейной моделью, без поправок, накопленных за прогон."""
        coefs = self.model["completion"][self._family(family)]
        tokens = float(coefs.get("base", 0.0))
        for name, value in (features or {}).items():
            tokens += float(coefs.get(name, 0.0)) * float(value)
        if step == "fast_retry":
            tokens *= 1.2
        return max(tokens, 1.0)

    def completion_tokens(self, family: str, step: str, features: Optional[dict] = None) -> float:
        """Ожидаемые выходные токены ступени: ответ и, для модели с рассуждениями, сами рассуждения."""
        fam = self._family(family)
        kind = "strong" if step == "strong" else "fast"
        tokens = self._base_answer_tokens(fam, step, features) * self._completion_ratio.get((fam, kind), 1.0)
        if step == "strong":
            tokens += float(self.model["reasoning"].get(fam, 800.0)) * self._completion_ratio.get((fam, "reasoning"), 1.0)
        return tokens

    def step_cost(self, family: str, step: str, prompt_tokens_approx: int, features: Optional[dict] = None,
                  calls: float = 1.0) -> float:
        """Ожидаемая стоимость ступени в выходных токенах регламента."""
        fam = self._family(family)
        if step == "react":
            return float(self.model["react"].get(fam, 2500.0)) * self._completion_ratio.get((fam, "react"), 1.0)
        prompt = prompt_tokens_approx * self.prompt_ratio
        return calls * (prompt / 10.0 + self.completion_tokens(fam, step, features))

    def safe_cost(self, cost: float) -> float:
        return cost * float(self.model["safety_factor"])

    def remember_prediction(self, task_id: str, step: str, family: str, features: Optional[dict]) -> None:
        """Запоминает прогноз перед вызовом, чтобы потом сравнить его с фактом."""
        fam = self._family(family)
        self._predicted[(task_id, step, fam)] = self._base_answer_tokens(fam, step, features)

    # ------------------------------------------------------------------ обучение на ходу
    def observe_call(self, record: CallRecord) -> None:
        """Уточняет поправки по фактическому usage вызова."""
        if not record.ok:
            return
        if record.approx_prompt_tokens > 0 and record.prompt_tokens > 0:
            ratio = record.prompt_tokens / record.approx_prompt_tokens
            if 0.3 < ratio < 4.0:
                self._prompt_obs += 1
                alpha = max(0.1, 1.0 / self._prompt_obs)
                self.prompt_ratio += alpha * (ratio - self.prompt_ratio)
        fam = self._family(record.family)
        base = self._predicted.get((record.task_id, record.step, fam))
        if not base or record.step == "react":
            return
        answer_tokens = max(record.completion_tokens - record.reasoning_tokens, 1)
        self._ema((fam, "strong" if record.step == "strong" else "fast"), answer_tokens / base)
        if record.step == "strong" and record.reasoning_tokens > 0:
            self._ema((fam, "reasoning"), record.reasoning_tokens / float(self.model["reasoning"].get(fam, 800.0)))

    def observe_react(self, family: str, cost: float) -> None:
        fam = self._family(family)
        base = float(self.model["react"].get(fam, 2500.0))
        if base > 0 and cost > 0:
            self._ema((fam, "react"), cost / base)

    def _ema(self, key: tuple[str, str], ratio: float) -> None:
        ratio = min(max(ratio, 0.2), 6.0)
        n = self._completion_obs.get(key, 0) + 1
        self._completion_obs[key] = n
        current = self._completion_ratio.get(key, 1.0)
        alpha = max(0.15, 1.0 / (n + 1))
        self._completion_ratio[key] = current + alpha * (ratio - current)

    def observe_outcome(self, family: str, step: str, success: bool) -> None:
        stat = self._outcomes.setdefault((self._family(family), step), [0, 0])
        stat[0] += 1 if success else 0
        stat[1] += 1

    # ------------------------------------------------------------------ вероятность успеха
    def solve_rate(self, family: str, step: str) -> float:
        """Доля успехов ступени на выполнимых задачах: априорное значение, уточнённое наблюдениями."""
        prior = float(self.model["solve"].get(step, 0.5))
        stat = self._outcomes.get((self._family(family), step))
        if not stat or stat[1] == 0:
            return prior
        pseudo = 6.0
        feasible_attempts = max(stat[1] * float(self.model["feasible_prior"]), stat[0])
        rate = (stat[0] + prior * pseudo) / (feasible_attempts + pseudo)
        return min(max(rate, 0.05), 0.99)

    def p_feasible(self, family: str, evidence: Evidence) -> float:
        if evidence.p_infeasible_static is not None:
            p = 1.0 - min(max(evidence.p_infeasible_static, 0.0), 1.0)
        else:
            p = float(self.model["feasible_prior"])
        like_feasible, like_infeasible = p, 1.0 - p
        for step in evidence.failed_steps:
            like_feasible *= 1.0 - self.solve_rate(family, step)
        for _ in range(evidence.refusals):
            like_feasible *= float(self.model["refuse_given_feasible"])
            like_infeasible *= float(self.model["refuse_given_infeasible"])
        total = like_feasible + like_infeasible
        return like_feasible / total if total > 0 else 0.0

    def p_success(self, family: str, step: str, evidence: Evidence) -> float:
        """Вероятность, что ступень принесёт баллы: P(выполнима | наблюдения) × доля успехов ступени."""
        return self.p_feasible(family, evidence) * self.solve_rate(family, step)

    def snapshot(self) -> dict:
        return {
            "prompt_ratio": round(self.prompt_ratio, 3),
            "completion_ratio": {f"{k[0]}/{k[1]}": round(v, 3) for k, v in self._completion_ratio.items()},
            "outcomes": {f"{k[0]}/{k[1]}": v for k, v in self._outcomes.items()},
            "source": self.model.get("source"),
        }
