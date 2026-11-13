"""Учёт расхода токенов по формуле регламента.

    расход = входные / 10 + выходные

Токены рассуждений считаются выходными. OpenAI-совместимые API обычно уже включают их
в completion_tokens, а в completion_tokens_details.reasoning_tokens дают разбивку.
Если провайдер считает их отдельно (reasoning_tokens > completion_tokens или выключен
флаг reasoning_in_completion), прибавляем сами: лучше переоценить расход, чем недооценить.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class CallRecord:
    task_id: str
    family: str
    step: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost: float
    seconds: float
    ok: bool
    finish_reason: str = ""
    error: str = ""
    approx_prompt_tokens: int = 0   # наша грубая оценка входа до вызова (для калибровки оценщика)


def call_cost(prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0,
              reasoning_in_completion: bool = True) -> float:
    output = completion_tokens
    if reasoning_tokens and (not reasoning_in_completion or reasoning_tokens > completion_tokens):
        output = completion_tokens + reasoning_tokens
    return prompt_tokens / 10.0 + output


class Ledger:
    def __init__(self, budget: Optional[float] = None, reasoning_in_completion: bool = True):
        self.budget = budget
        self.reasoning_in_completion = reasoning_in_completion
        self.calls: list[CallRecord] = []
        self.spent: float = 0.0
        self.by_task: dict[str, float] = {}
        self.reserved: float = 0.0  # оценка расхода уже запущенных, но не завершённых вызовов
        self._open = 0              # сколько резервов открыто

    def hard_limit(self) -> Optional[float]:
        """Сколько можно потратить всего: бюджет без маленького запаса на неточность учёта."""
        if self.budget is None:
            return None
        return self.budget - max(3.0, 0.01 * self.budget)

    def reserve(self, amount: float) -> None:
        self.reserved += amount
        self._open += 1

    def release(self, amount: float) -> None:
        self._open = max(0, self._open - 1)
        # Без открытых резервов обнуляем: иначе от сложений с плавающей точкой остаётся хвост
        # вроде 1e-13, и проверка «есть ли запущенные вызовы» навсегда отвечает «да».
        self.reserved = 0.0 if self._open == 0 else max(0.0, self.reserved - amount)

    def has_reservations(self) -> bool:
        return self._open > 0

    def add(self, record: CallRecord) -> None:
        self.calls.append(record)
        self.spent += record.cost
        self.by_task[record.task_id] = self.by_task.get(record.task_id, 0.0) + record.cost

    def cost(self, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0) -> float:
        return call_cost(prompt_tokens, completion_tokens, reasoning_tokens, self.reasoning_in_completion)

    def task_spent(self, task_id: str) -> float:
        return self.by_task.get(task_id, 0.0)

    def remaining(self) -> Optional[float]:
        if self.budget is None:
            return None
        return self.budget - self.spent - self.reserved

    def summary(self) -> dict:
        by_model: dict[str, dict] = {}
        for c in self.calls:
            m = by_model.setdefault(c.model, {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0, "cost": 0.0})
            m["calls"] += 1
            m["prompt"] += c.prompt_tokens
            m["completion"] += c.completion_tokens
            m["reasoning"] += c.reasoning_tokens
            m["cost"] = round(m["cost"] + c.cost, 1)
        return {
            "spent": round(self.spent, 1),
            "calls": len(self.calls),
            "failed_calls": sum(1 for c in self.calls if not c.ok),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "reasoning_tokens": sum(c.reasoning_tokens for c in self.calls),
            "by_model": by_model,
            "budget": self.budget,
        }

    def dump(self) -> list[dict]:
        return [asdict(c) for c in self.calls]
