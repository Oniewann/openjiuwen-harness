"""Вызовы моделей через openJiuwen.

Все обращения к LLM в харнессе идут через один объект openjiuwen `Model`
(OpenAI-совместимый провайдер, адрес и ключ — из LLM_BASE_URL / LLM_API_KEY).
Поверх него здесь добавлено то, что нужно для работы в бюджете:

  * учёт расхода каждого вызова в Ledger (вход/10 + выход);
  * ограничение параллельности (до 5 сессий) и частоты запросов;
  * повторы при временных сбоях (429, 5xx, таймаут) с паузами;
  * «предохранитель»: если прокси перестал отвечать (бюджет исчерпан), вызовы
    не молотят впустую, а ждут и изредка пробуют снова;
  * потолок расхода на задачу — срабатывает и для прямых вызовов, и внутри ReAct-агента,
    потому что агенту через set_llm() отдаётся эта же модель.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import ALLOWED_MODELS, Config
from .ledger import CallRecord, Ledger
from .textutil import approx_tokens, strip_think


class LLMError(Exception):
    """Вызов модели не удался (после повторов)."""


class LLMDead(LLMError):
    """Прокси не отвечает: бюджет исчерпан или доступ закрыт. Дальше звать модель бессмысленно."""


class TaskCapExceeded(LLMError):
    """Задача исчерпала свой потолок токенов."""


@dataclass
class Reply:
    text: str               # ответ без блоков рассуждений
    raw: str                # ответ как пришёл
    reasoning: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost: float
    finish_reason: str
    seconds: float

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# Кому приписать расход текущего вызова: задаётся решателем на время шага.
_CALL_SCOPE: contextvars.ContextVar[dict] = contextvars.ContextVar("harness_call_scope", default={})


class _RateLimiter:
    """Равномерно разносит старты запросов: не чаще rpm в минуту."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._next = 0.0
        self._lock: Optional[asyncio.Lock] = None

    async def wait(self) -> None:
        if not self.interval:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


def _status_code(exc: BaseException) -> Optional[int]:
    """Код HTTP из цепочки исключений (openJiuwen оборачивает ошибку SDK в свою)."""
    cur: Optional[BaseException] = exc
    for _ in range(8):
        if cur is None:
            break
        code = getattr(cur, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(getattr(cur, "response", None), "status_code", None)
        if isinstance(code, int):
            return code
        cur = cur.__cause__ or cur.__context__
    return None


def classify_error(exc: BaseException) -> str:
    """auth | model | endpoint | bad_request | transient"""
    code = _status_code(exc)
    text = str(exc).lower()
    if code in (401, 403):
        return "auth"
    if code == 404:
        return "model" if "model" in text else "endpoint"
    if code in (400, 422):
        if "model" in text and any(s in text for s in ("not found", "does not exist", "unknown", "not supported",
                                                       "invalid model", "no such")):
            return "model"
        return "bad_request"
    return "transient"  # 429, 5xx, таймаут, обрыв соединения, неизвестное


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


class LLMHub:
    def __init__(self, cfg: Config, ledger: Ledger):
        self.cfg = cfg
        self.ledger = ledger
        self.dead = False
        self.dead_reason = ""
        self._consecutive_failures = 0
        self._first_failure_at: Optional[float] = None
        self._breaker_until = 0.0
        self._auth_failures = 0
        self._sem: Optional[asyncio.Semaphore] = None
        self._rate = _RateLimiter(cfg.rpm_limit)
        self._model = None
        self._base_url = cfg.base_url
        self._alt_base_tried = False
        self.unavailable_models: set[str] = set()
        self.on_call: Optional[Callable[[CallRecord], None]] = None
        self.give_up_after_s = 240.0
        self.token_ratio = 1.0  # поправка грубой оценки токенов по фактическому usage (обновляет оценщик)

    # ------------------------------------------------------------------ модель openJiuwen
    def _build_model(self):
        from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig

        hub = self

        class MeteredModel(Model):
            """openjiuwen Model, каждый вызов которой проходит через учёт и лимиты харнесса."""

            async def invoke(self, messages, **kwargs):  # type: ignore[override]
                message, _ = await hub._guarded_invoke(messages, kwargs)
                return message

        client_cfg = ModelClientConfig(
            client_provider="OpenAI",
            api_key=self.cfg.api_key or "EMPTY",
            api_base=self._base_url,
            verify_ssl=bool(self.cfg.extra.get("verify_ssl", True)),
            timeout=max(self.cfg.call_timeout_strong, self.cfg.call_timeout_fast) + 30,
        )
        return MeteredModel(client_cfg, ModelRequestConfig(model=self.cfg.model_for("fast")))

    def model(self):
        """Общий экземпляр openjiuwen Model (его же получает ReAct-агент через set_llm)."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    async def _raw_invoke(self, messages, kwargs: dict):
        """Собственно запрос: метод invoke базового класса openjiuwen Model."""
        from openjiuwen.core.foundation.llm import Model

        return await Model.invoke(self.model(), messages, **kwargs)

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.cfg.max_parallel)
        return self._sem

    # ------------------------------------------------------------------ область учёта
    @contextlib.contextmanager
    def scope(self, task_id: str, family: str, step: str, cap: Optional[float] = None,
              max_tokens: Optional[int] = None, temperature: Optional[float] = None,
              timeout: Optional[float] = None, model: Optional[str] = None):
        """На время шага задаёт, какой задаче приписывать вызовы и какие у них лимиты.

        max_tokens, temperature, timeout и model действуют на вызовы, где они не заданы явно, —
        так настраиваются запросы ReAct-агента openJiuwen, который сам их не передаёт.
        """
        token = _CALL_SCOPE.set({"task_id": task_id, "family": family, "step": step, "cap": cap,
                                 "max_tokens": max_tokens, "temperature": temperature, "timeout": timeout,
                                 "model": model})
        try:
            yield
        finally:
            _CALL_SCOPE.reset(token)

    def resolve_model(self, name: Optional[str]) -> str:
        """Подставляет рабочую модель: режим одной модели и замена недоступных."""
        if self.cfg.single_model:
            return self.cfg.single_model
        name = name or self.cfg.model_for("fast")
        if name not in self.unavailable_models:
            return name
        reasoning = not name.endswith("-noreason")
        same_kind = [m for m in ALLOWED_MODELS if (not m.endswith("-noreason")) == reasoning]
        for candidate in same_kind + list(ALLOWED_MODELS):
            if candidate not in self.unavailable_models:
                return candidate
        return name

    # ------------------------------------------------------------------ вызов с защитой
    async def _guarded_invoke(self, messages, kwargs: dict):
        scope = _CALL_SCOPE.get()
        task_id = scope.get("task_id", "-")
        cap = scope.get("cap")
        kwargs = dict(kwargs)
        if kwargs.get("max_tokens") is None and scope.get("max_tokens"):
            kwargs["max_tokens"] = scope["max_tokens"]
        if kwargs.get("temperature") is None and scope.get("temperature") is not None:
            kwargs["temperature"] = scope["temperature"]
        if kwargs.get("timeout") is None and scope.get("timeout"):
            kwargs["timeout"] = scope["timeout"]
        if scope.get("model"):
            kwargs["model"] = scope["model"]

        attempt = 0
        while True:
            if self.dead:
                raise LLMDead(self.dead_reason or "proxy is not responding")
            if cap is not None and self.ledger.task_spent(task_id) >= cap:
                raise TaskCapExceeded(f"task {task_id} reached its token cap {cap:.0f}")
            remaining = self.ledger.remaining()
            if remaining is not None and remaining <= 0:
                self.dead, self.dead_reason = True, "known token budget is exhausted"
                raise LLMDead(self.dead_reason)

            wait = self._breaker_until - time.monotonic()
            if wait > 0:
                await asyncio.sleep(min(wait, 5.0))
                continue

            model_name = self.resolve_model(kwargs.get("model"))
            kwargs["model"] = model_name
            attempt += 1
            await self._rate.wait()
            async with self._semaphore():
                started = time.monotonic()
                try:
                    message = await self._raw_invoke(messages, kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — любая ошибка SDK или сети
                    kind = classify_error(exc)
                    self._record_failure(scope, model_name, time.monotonic() - started, kind, exc)
                    if kind == "model":
                        self.unavailable_models.add(model_name)
                        if self.cfg.single_model or len(self.unavailable_models) >= len(ALLOWED_MODELS):
                            raise LLMError(f"model {model_name} is not available: {exc}") from exc
                        attempt -= 1
                        continue  # сразу пробуем замену
                    if kind == "endpoint" and not self._alt_base_tried:
                        self._switch_base_url()
                        attempt -= 1
                        continue
                    if kind == "bad_request":
                        raise LLMError(f"bad request: {exc}") from exc
                    if kind == "auth":
                        self._auth_failures += 1
                        if self._auth_failures >= 3:
                            self.dead, self.dead_reason = True, f"access denied by proxy: {str(exc)[:200]}"
                            raise LLMDead(self.dead_reason) from exc
                    self._on_transient_failure()
                    if attempt >= 3:
                        raise LLMError(f"model call failed after {attempt} attempts: {str(exc)[:300]}") from exc
                    pause = min(2.0 * (3 ** (attempt - 1)), 20.0)
                else:
                    self._on_success()
                    record = self._record_success(scope, model_name, messages, message, time.monotonic() - started)
                    return message, record
            await asyncio.sleep(pause)

    def _switch_base_url(self) -> None:
        """Адрес прокси мог быть дан с /v1 или без: при 404 один раз пробуем второй вариант."""
        self._alt_base_tried = True
        base = self._base_url.rstrip("/")
        self._base_url = base[: -len("/v1")] if base.endswith("/v1") else base + "/v1"
        self._model = None

    def _on_success(self) -> None:
        self._consecutive_failures = 0
        self._first_failure_at = None
        self._auth_failures = 0
        self._breaker_until = 0.0

    def _on_transient_failure(self) -> None:
        now = time.monotonic()
        self._consecutive_failures += 1
        if self._first_failure_at is None:
            self._first_failure_at = now
        over = self._consecutive_failures - self.cfg.dead_after_failures
        if over >= 0:
            # Прокси молчит: скорее всего, бюджет исчерпан. Ответы уже сохранены, спешить некуда —
            # ставим паузу и изредка пробуем снова; если тишина держится долго, сдаёмся.
            self._breaker_until = now + min(15.0 * (2 ** min(over, 4)), 60.0)
            if now - self._first_failure_at >= self.give_up_after_s:
                self.dead = True
                self.dead_reason = "proxy stopped responding (token budget is probably exhausted)"

    # ------------------------------------------------------------------ учёт
    def _record_success(self, scope: dict, model_name: str, messages, message, seconds: float) -> CallRecord:
        usage = getattr(message, "usage_metadata", None)
        content = _content_text(getattr(message, "content", ""))
        reasoning = getattr(message, "reasoning_content", None) or ""
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0) if usage else 0
        items = messages if isinstance(messages, list) else [{"content": str(messages)}]
        texts = [_content_text(m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
                 for m in items]
        approx_prompt = sum(approx_tokens(t) + 4 for t in texts) + 3
        usage_missing = prompt_tokens == 0 and completion_tokens == 0
        if usage_missing:
            # Прокси не вернул usage: оцениваем сами, чтобы учёт не обнулился.
            prompt_tokens = int(approx_prompt * self.token_ratio)
            completion_tokens = approx_tokens(content) + approx_tokens(reasoning)
        if reasoning and not reasoning_tokens and not self.cfg.reasoning_in_completion:
            reasoning_tokens = approx_tokens(reasoning)
        record = CallRecord(
            task_id=scope.get("task_id", "-"), family=scope.get("family", "-"), step=scope.get("step", "-"),
            model=model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=self.ledger.cost(prompt_tokens, completion_tokens, reasoning_tokens),
            seconds=round(seconds, 3), ok=True,
            finish_reason=str(getattr(message, "finish_reason", "") or ""),
            approx_prompt_tokens=0 if usage_missing else approx_prompt,
        )
        self.ledger.add(record)
        if self.on_call:
            try:
                self.on_call(record)
            except Exception:  # noqa: BLE001 — наблюдатель не должен ломать вызов
                pass
        return record

    def _record_failure(self, scope: dict, model_name: str, seconds: float, kind: str, exc: BaseException) -> None:
        self.ledger.calls.append(CallRecord(
            task_id=scope.get("task_id", "-"), family=scope.get("family", "-"), step=scope.get("step", "-"),
            model=model_name, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, cost=0.0,
            seconds=round(seconds, 3), ok=False, error=f"{kind}: {str(exc)[:200]}",
        ))

    # ------------------------------------------------------------------ простой вызов
    def is_strong(self, model_name: str) -> bool:
        return not self.resolve_model(model_name).endswith("-noreason")

    async def chat(self, messages: list[dict], *, role: str = "fast", model: Optional[str] = None,
                   max_tokens: Optional[int] = None, temperature: Optional[float] = 0.0,
                   timeout: Optional[float] = None) -> Reply:
        """Один запрос к модели. role: fast | fast_alt | strong."""
        model_name = model or self.cfg.model_for(role)
        strong = self.is_strong(model_name)
        if max_tokens is None:
            max_tokens = self.cfg.max_tokens_strong if strong else self.cfg.max_tokens_fast
        if timeout is None:
            timeout = self.cfg.call_timeout_strong if strong else self.cfg.call_timeout_fast
        if strong and temperature is not None and temperature < 0.3:
            temperature = 0.6  # для режима рассуждений Qwen жадное декодирование не рекомендуют
        kwargs: dict[str, Any] = {"model": model_name, "max_tokens": max_tokens, "timeout": timeout}
        if temperature is not None:
            kwargs["temperature"] = temperature
        started = time.monotonic()
        message, record = await self._guarded_invoke(messages, kwargs)
        content = _content_text(message.content)
        return Reply(
            text=strip_think(content),
            raw=content,
            reasoning=message.reasoning_content or "",
            model=record.model,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            reasoning_tokens=record.reasoning_tokens,
            cost=record.cost,
            finish_reason=record.finish_reason,
            seconds=time.monotonic() - started,
        )

    async def aclose(self) -> None:
        try:
            from openjiuwen.core.foundation.llm import OpenAIModelClient

            await OpenAIModelClient.aclose()
        except Exception:  # noqa: BLE001
            pass
