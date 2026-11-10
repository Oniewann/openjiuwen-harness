"""Локальная заглушка LLM-прокси (OpenAI-совместимый /chat/completions).

Нужна только для разработки: проверить обвязку харнесса без ключа и без сети.
В боевом прогоне не используется.

Что умеет:
  * логировать каждый запрос в JSONL (видно, что именно openJiuwen шлёт модели);
  * считать «расход» по формуле регламента: вход/10 + выход (+ рассуждения);
  * эмулировать исчерпание бюджета (--budget): после лимита прокси перестаёт отвечать;
  * подставлять «мозг» (--brain): echo | script | oracle.

Запуск:
    python devtools/mock_llm.py --port 8089 --brain oracle --log /tmp/mock.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Грубая оценка числа токенов: слова и знаки препинания, длинные слова дробятся."""
    if not text:
        return 0
    n = 0
    for tok in _TOKEN_RE.findall(text):
        n += max(1, (len(tok) + 3) // 4)
    return n


def messages_tokens(messages: list, tools: list | None) -> int:
    n = 0
    for m in messages:
        n += 4  # служебные токены роли
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
        n += count_tokens(content or "")
        for tc in m.get("tool_calls") or []:
            n += count_tokens(json.dumps(tc, ensure_ascii=False))
    if tools:
        n += count_tokens(json.dumps(tools, ensure_ascii=False))
    return n


class State:
    def __init__(self, budget: float | None, log_path: str | None, brain, exhaust_mode: str,
                 latency: float, reasoning_tokens: int):
        self.budget = budget
        self.spent = 0.0
        self.calls = 0
        self.lock = threading.Lock()
        self.log_path = log_path
        self.brain = brain
        self.exhaust_mode = exhaust_mode
        self.latency = latency
        self.reasoning_tokens = reasoning_tokens

    def log(self, record: dict) -> None:
        if not self.log_path:
            return
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def brain_echo(body: dict) -> dict:
    return {"content": "OK"}


def load_brain(name: str):
    if name == "echo":
        return brain_echo
    here = Path(__file__).resolve().parent
    path = here / f"mock_brain_{name}.py"
    if not path.exists():
        raise SystemExit(f"нет мозга {name}: {path}")
    spec = importlib.util.spec_from_file_location(f"mock_brain_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.respond


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # тишина в консоли
            pass

        def _send_json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send_json(200, {"object": "list", "data": [
                    {"id": m, "object": "model"} for m in (
                        "qwen3.8-27b", "qwen3.8-27b-noreason",
                        "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason")]})
            elif self.path.rstrip("/").endswith("/stats"):
                self._send_json(200, {"spent": state.spent, "calls": state.calls, "budget": state.budget})
            else:
                self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": {"message": "bad json"}})
                return
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send_json(404, {"error": {"message": "not found"}})
                return

            with state.lock:
                exhausted = state.budget is not None and state.spent >= state.budget
            if exhausted:
                state.log({"t": time.time(), "event": "exhausted", "path": self.path})
                if state.exhaust_mode == "hang":
                    time.sleep(3600)
                    return
                self._send_json(429, {"error": {"message": "token budget exhausted", "type": "budget_exhausted"}})
                return

            if state.latency:
                time.sleep(state.latency)

            messages = body.get("messages") or []
            tools = body.get("tools")
            model = body.get("model", "")
            prompt_tokens = messages_tokens(messages, tools)
            try:
                reply = state.brain(body) or {}
            except Exception as exc:  # мозг не должен ронять сервер
                reply = {"content": f"MOCK_BRAIN_ERROR: {exc!r}"}

            if reply.get("http_error"):
                self._send_json(int(reply["http_error"]), {"error": {"message": reply.get("content", "mock error")}})
                return

            content = reply.get("content") or ""
            tool_calls = reply.get("tool_calls")
            reasoning = reply.get("reasoning") or ""
            is_reasoning_model = not model.endswith("-noreason")
            reasoning_tokens = 0
            if is_reasoning_model:
                reasoning_tokens = count_tokens(reasoning) if reasoning else state.reasoning_tokens
                if not reasoning:
                    reasoning = "(mock reasoning)"
            else:
                reasoning = ""
            completion_tokens = count_tokens(content) + reasoning_tokens
            if tool_calls:
                completion_tokens += count_tokens(json.dumps(tool_calls, ensure_ascii=False))
            max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
            finish_reason = "tool_calls" if tool_calls else "stop"
            if max_tokens and completion_tokens > int(max_tokens):
                completion_tokens = int(max_tokens)
                finish_reason = "length"
                if reasoning_tokens >= int(max_tokens):
                    content = ""

            cost = prompt_tokens / 10.0 + completion_tokens
            with state.lock:
                state.spent += cost
                state.calls += 1
                spent = state.spent

            message: dict = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning_content"] = reasoning
            if tool_calls:
                message["tool_calls"] = tool_calls
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            }
            state.log({"t": time.time(), "path": self.path, "model": model, "request": body,
                       "reply": message, "usage": usage, "cost": cost, "spent": spent})

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(obj) -> None:
                    payload = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")

                base = {"id": "mock", "object": "chat.completion.chunk", "model": model, "created": int(time.time())}
                delta = dict(message)
                chunk({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
                chunk({**base, "choices": [], "usage": usage})
                chunk("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                return

            self._send_json(200, {
                "id": "mock", "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": usage,
            })

    return Handler


def serve(port: int, brain_name: str = "echo", log_path: str | None = None, budget: float | None = None,
          exhaust_mode: str = "error", latency: float = 0.0, reasoning_tokens: int = 400,
          host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, State]:
    state = State(budget, log_path, load_brain(brain_name), exhaust_mode, latency, reasoning_tokens)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server, state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--brain", default="echo")
    ap.add_argument("--log", default=None)
    ap.add_argument("--budget", type=float, default=None, help="лимит в выходных токенах (вход/10 + выход)")
    ap.add_argument("--exhaust-mode", choices=["error", "hang"], default="error")
    ap.add_argument("--latency", type=float, default=0.0)
    ap.add_argument("--reasoning-tokens", type=int, default=400,
                    help="сколько токенов рассуждений приписывать reasoning-моделям")
    args = ap.parse_args()
    server, state = serve(args.port, args.brain, args.log, args.budget, args.exhaust_mode, args.latency,
                          args.reasoning_tokens, args.host)
    print(f"mock LLM on http://{args.host}:{args.port}/v1 brain={args.brain} budget={args.budget}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print(json.dumps({"spent": state.spent, "calls": state.calls}), file=sys.stderr)


if __name__ == "__main__":
    main()
