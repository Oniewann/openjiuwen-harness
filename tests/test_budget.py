"""Бюджет любой величины: разбор BUDGET, масштабирование лимитов, резерв и потолок вызова."""
import asyncio
import os
import unittest
from unittest import mock

from harness.config import Config, load_config, parse_budget
from harness.ledger import Ledger
from harness.llm import BudgetShort, LLMDead, LLMHub, classify_error


class ParseBudgetTest(unittest.TestCase):
    def test_forms(self):
        cases = {
            "100000": 100000, "100000.0": 100000, "1e5": 100000, "100_000": 100000, "100 000": 100000,
            "100k": 100000, "100K": 100000, "1.5m": 1_500_000, "5k": 5000, " 300 ": 300, "0": 0,
            "'2000'": 2000, "100,000": 100000,
        }
        for raw, expected in cases.items():
            self.assertEqual(parse_budget(raw), expected, raw)

    def test_garbage(self):
        for raw in ("abc", "", "k", "nan", "inf"):
            self.assertIsNone(parse_budget(raw), raw)
        self.assertEqual(parse_budget("-5"), 0)

    def _load(self, value):
        env = {k: v for k, v in os.environ.items()
               if k not in ("BUDGET", "HARNESS_TOKEN_BUDGET", "TOKEN_BUDGET", "LLM_TOKEN_BUDGET", "BUDGET_TOKENS")}
        if value is not None:
            env["BUDGET"] = value
        with mock.patch.dict(os.environ, env, clear=True):
            return load_config()

    def test_load(self):
        self.assertEqual(self._load("5k").budget, 5000)
        self.assertEqual(self._load("0").budget, 0)
        self.assertIsNone(self._load(None).budget)
        cfg = self._load("lots")
        self.assertEqual(cfg.budget, cfg.unparsed_budget)
        self.assertEqual(cfg.budget_raw, "lots")


class ScaleTest(unittest.TestCase):
    def test_default_budget_keeps_limits(self):
        cfg = Config(budget=100_000)
        cfg.scale_to(100, 100)
        self.assertEqual(cfg.task_token_cap, 4000)
        self.assertEqual(cfg.tokens_per_point, 2000)
        self.assertEqual(cfg.react_token_cap, 1500)

    def test_small_budget_shrinks_limits(self):
        cfg = Config(budget=300)
        cfg.scale_to(2, 2)
        self.assertEqual(cfg.task_token_cap, 600)
        self.assertEqual(cfg.tokens_per_point, 500)
        self.assertLessEqual(cfg.react_token_cap, 600)


class _Msg:
    def __init__(self, content="42", out=10):
        self.content = content
        self.reasoning_content = ""
        self.finish_reason = "stop"
        self.usage_metadata = type("U", (), {"input_tokens": 100, "output_tokens": out, "reasoning_tokens": 0})()


class _Err(Exception):
    def __init__(self, code, text):
        super().__init__(text)
        self.status_code = code


class ReservationTest(unittest.TestCase):
    def test_no_float_residue(self):
        led = Ledger(budget=1000)
        amounts = [400.1, 33.7, 0.3, 912.9]
        for a in amounts:
            led.reserve(a)
        for a in amounts:
            led.release(a)
        self.assertFalse(led.has_reservations())
        self.assertEqual(led.reserved, 0.0)


class HubBudgetTest(unittest.TestCase):
    def _hub(self, budget):
        cfg = Config(budget=budget, rpm_limit=0, base_url="http://x")
        return LLMHub(cfg, Ledger(budget=budget))

    def test_max_tokens_capped_by_remaining(self):
        hub = self._hub(1000)
        seen = {}

        async def raw(messages, kwargs):
            seen.update(kwargs)
            return _Msg()

        hub._raw_invoke = raw
        hub.ledger.spent = 600
        asyncio.run(hub.chat([{"role": "user", "content": "hi"}], max_tokens=900))
        self.assertLessEqual(seen["max_tokens"], 1000 - 10 - 600)
        self.assertGreaterEqual(seen["max_tokens"], 300)
        self.assertEqual(hub.ledger.reserved, 0)

    def test_budget_short_and_dead(self):
        hub = self._hub(1000)

        async def raw(messages, kwargs):
            return _Msg()

        hub._raw_invoke = raw
        hub.ledger.spent = 960   # остаток ~30 < 60 — дешёвый вызов бесполезен
        with self.assertRaises(BudgetShort):
            asyncio.run(hub.chat([{"role": "user", "content": "hi"}]))
        hub.ledger.spent = 995
        with self.assertRaises(LLMDead):
            asyncio.run(hub.chat([{"role": "user", "content": "hi"}]))

    def test_zero_budget_never_calls(self):
        hub = self._hub(0)
        called = []

        async def raw(messages, kwargs):
            called.append(1)
            return _Msg()

        hub._raw_invoke = raw
        with self.assertRaises(LLMDead):
            asyncio.run(hub.chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(called, [])

    def test_strong_needs_room(self):
        hub = self._hub(1200)

        async def raw(messages, kwargs):
            return _Msg()

        hub._raw_invoke = raw
        with self.assertRaises(BudgetShort):
            asyncio.run(hub.chat([{"role": "user", "content": "hi"}], role="strong"))

    def test_rate_limit_waits_and_failures_charged(self):
        hub = self._hub(100_000)
        calls = []

        async def raw(messages, kwargs):
            calls.append(1)
            if len(calls) <= 2:
                raise _Err(429, "Too many concurrent requests: 3 already running")
            return _Msg()

        hub._raw_invoke = raw
        with mock.patch("harness.llm.asyncio.sleep", new=mock.AsyncMock()):
            reply = asyncio.run(hub.chat([{"role": "user", "content": "hello there"}]))
        self.assertEqual(reply.text, "42")
        self.assertEqual(len(calls), 3)
        failed = [c for c in hub.ledger.calls if not c.ok]
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(c.cost > 0 for c in failed))
        self.assertEqual(hub._gate.limit, 1)
        self.assertEqual(hub._consecutive_failures, 0)

    def test_exhausted_429_is_dead(self):
        hub = self._hub(100_000)

        async def raw(messages, kwargs):
            raise _Err(429, "token budget exhausted")

        hub._raw_invoke = raw
        with self.assertRaises(LLMDead):
            asyncio.run(hub.chat([{"role": "user", "content": "hi"}]))

    def test_classify(self):
        self.assertEqual(classify_error(_Err(429, "Слишком много одновременных запросов с аккаунта: 3 уже в работе")),
                         "rate")
        self.assertEqual(classify_error(_Err(429, "token budget exhausted")), "exhausted")
        self.assertEqual(classify_error(_Err(503, "unavailable")), "transient")


if __name__ == "__main__":
    unittest.main()
