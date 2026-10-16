"""OpenAI-compatible HTTP server for a local model, using only the standard library.

    python -m superborg serve model.borg --port 8080
    curl localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Hi"}]}'

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions.
Any OpenAI client works (set base_url to http://host:port/v1; the API key is ignored unless
the server was started with one).

Extensions to the OpenAI request body:
  "reasoning": "<strategy>"   answer the last user message with superborg.reason.solve
                              (strategies: superborg.reason.STRATEGIES, e.g. "auto", "vote"; also
                              {"strategy": ...}); the response carries a "reasoning" object with the
                              strategy used, the extracted short answer and the route "auto" took.
                              Streaming sends the strategy's progress as "reasoning_content" deltas
                              and the final answer as content.
  "samples": K                samples for voting strategies (default 5)
  "context": "text"           reference text for retrieval-augmented answering ("rag")
  "enable_thinking": bool     Qwen3 thinking mode (also "chat_template_kwargs": {"enable_thinking": ...})
  "repeat_penalty": R         repetition penalty (alias "repetition_penalty")
  "lookup": N                 prompt-lookup speculative decoding with N-gram matching

The model and its KV cache are shared, so one lock serializes generation; consecutive chat
requests that extend the same conversation reuse the cached prefix.

Security: the program-aided strategies ("pal", "pal_vote", and "auto" when it picks them) run
model-written Python on this machine (superborg.tools: resource limits, not a sandbox). A
prompt can steer that code, and any web page the user visits can POST to a local server, so
code execution is off unless the server is started with allow_code=True (--allow-code); even
then it is refused for browser requests from non-local origins. Without it, "auto" routes math
questions to self-consistency voting instead.
"""
from __future__ import annotations

import contextlib
import functools
import hmac
import json
import math
import secrets
import socket
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

MAX_BODY_BYTES = 8 << 20
MAX_STOP_STRINGS = 16
MAX_STOP_CHARS = 512  # streaming holds back this much text while a stop string could still match
MAX_CHOICES = 8
SOCKET_TIMEOUT_S = 300  # a client that stops sending or reading must not pin a handler thread forever
# below this a softmax temperature is greedy in effect, and logits / T can overflow to inf (NaN probabilities)
MIN_TEMPERATURE = 1e-6

# llama.cpp-style defaults: OpenAI's temperature 1.0 is too noisy for small local models
DEFAULT_SAMPLING = {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05}

_FINISH = {"eos": "stop", "stop": "stop", "length": "length", "context": "length"}
CODE_STRATEGIES = {"pal", "pal_vote"}
# what "auto" picks for each question kind (superborg.reason.classify) when code may not run
_AUTO_WITHOUT_CODE = {"rag": "rag", "arithmetic": "calc", "math": "vote", "reasoning": "cot", "simple": "direct"}
_LOCAL_ORIGINS = ("http://localhost", "http://127.0.0.1", "http://[::1]", "https://localhost", "https://127.0.0.1")
_ROLES = {"system", "user", "assistant", "tool", "developer", "function"}


class CodeExecutionDisabled(RuntimeError):
    pass


def _refuse_code(*_a, **_k):
    raise CodeExecutionDisabled("this server does not run model-written code (start it with --allow-code)")


# The guard patches a module attribute, which is process-wide: a depth count keeps it correct
# when several servers in one process overlap (otherwise the second one could save the first
# one's refusal as "the original" and leave run_python disabled for good).
_BLOCK_LOCK = threading.Lock()
_block_depth = 0
_saved_run_python = None


@contextlib.contextmanager
def _code_execution_blocked(tools_module):
    """Make superborg.tools.run_python refuse while a request is handled (defense in depth:
    e.g. the calculator strategy falls back to a program when it cannot parse a question)."""
    global _block_depth, _saved_run_python
    with _BLOCK_LOCK:
        if _block_depth == 0:
            _saved_run_python = tools_module.run_python
            tools_module.run_python = _refuse_code
        _block_depth += 1
    try:
        yield
    finally:
        with _BLOCK_LOCK:
            _block_depth -= 1
            if _block_depth == 0:
                tools_module.run_python = _saved_run_python
                _saved_run_python = None


def _is_local_origin(origin: str | None) -> bool:
    if origin is None:
        return True  # not a browser request
    return any(origin == o or origin.startswith(o + ":") for o in _LOCAL_ORIGINS)


class APIError(Exception):
    """An error reported to the client as an OpenAI-style JSON error body."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.param, self.code = status, message, type_, param, code

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


# ---------------------------------------------------------------------------
# request parsing

def _number(body: dict, name: str, default, kind: type, lo=None, hi=None, aliases: tuple = ()):
    value = default
    for key in (name, *aliases):
        if body.get(key) is not None:
            value = body[key]
            name = key
            break
    if value is None:
        return None
    # bool is an int subclass; reject it explicitly so {"max_tokens": true} is an error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(400, f"'{name}' must be a number", param=name)
    # json.loads accepts NaN and Infinity, and NaN passes every range check below
    if isinstance(value, float) and not math.isfinite(value):
        raise APIError(400, f"'{name}' must be a finite number", param=name)
    if kind is int:
        if isinstance(value, float) and not value.is_integer():
            raise APIError(400, f"'{name}' must be an integer", param=name)
        value = int(value)
    else:
        value = float(value)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise APIError(400, f"'{name}' must be in [{lo}, {hi}]", param=name)
    return value


def _text(s: str, what: str, param: str) -> str:
    """JSON can carry lone UTF-16 surrogates ("\\ud800"), which cannot be encoded for the tokenizer."""
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as e:
        raise APIError(400, f"{what} is not valid Unicode text ({e.reason} at position {e.start})",
                       param=param) from e
    return s


def _flag(body: dict, name: str) -> bool:
    value = body.get(name)
    if value is None:
        return False
    if not isinstance(value, bool):  # "false" is truthy: refuse rather than guess
        raise APIError(400, f"'{name}' must be a boolean", param=name)
    return value


def _stop_list(body: dict) -> list[str]:
    stop = body.get("stop")
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise APIError(400, "'stop' must be a string or a list of strings", param="stop")
    if len(stop) > MAX_STOP_STRINGS:
        raise APIError(400, f"at most {MAX_STOP_STRINGS} stop strings are supported", param="stop")
    if any(len(s) > MAX_STOP_CHARS for s in stop):
        raise APIError(400, f"stop strings are limited to {MAX_STOP_CHARS} characters", param="stop")
    return [_text(s, "a stop string", "stop") for s in stop if s]


def _content_text(content: Any, where: str) -> str:
    """OpenAI content is a string or a list of parts; only text parts are understood."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _text(content, f"{where}.content", "messages")
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                out.append(str(part.get("text", "")))
            elif isinstance(part, dict):
                raise APIError(400, f"{where}: content part type '{part.get('type')}' is not supported",
                               param="messages")
        return _text("".join(out), f"{where}.content", "messages")
    raise APIError(400, f"{where}: 'content' must be a string or a list of parts", param="messages")


def parse_messages(body: dict) -> list[dict]:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise APIError(400, "'messages' must be a non-empty list", param="messages")
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise APIError(400, f"messages[{i}] must be an object with a 'role'", param="messages")
        role = m["role"]
        if role not in _ROLES:
            raise APIError(400, f"messages[{i}]: unknown role '{role}'", param="messages")
        if role == "developer":  # newer OpenAI name for the system role
            role = "system"
        out.append({"role": role, "content": _content_text(m.get("content"), f"messages[{i}]")})
    return out


def _temperature(body: dict, default: float | None) -> float | None:
    t = _number(body, "temperature", default, float, 0.0, 2.0)
    return 0.0 if t is not None and t < MIN_TEMPERATURE else t


def _sampling(body: dict, defaults: dict) -> dict:
    seed = _number(body, "seed", None, int)
    return {
        "temperature": _temperature(body, defaults["temperature"]),
        "top_p": _number(body, "top_p", defaults["top_p"], float, 0.0, 1.0),
        # the engine's top-k is a C int; anything above the vocabulary size means "off" anyway
        "top_k": _number(body, "top_k", defaults["top_k"], int, 0, 1 << 30),
        "min_p": _number(body, "min_p", defaults["min_p"], float, 0.0, 1.0),
        # the penalty divides logits: 0 (or nearly) turns them into inf/NaN
        "repeat_penalty": _number(body, "repeat_penalty", 1.0, float, 0.01, 10.0,
                                  aliases=("repetition_penalty",)),
        # a random seed per request unless the client pins one, as OpenAI does
        "seed": secrets.randbits(63) if seed is None else seed & 0xFFFFFFFFFFFFFFFF,
    }


def _include_usage(body: dict) -> bool:
    opts = body.get("stream_options")
    return isinstance(opts, dict) and bool(opts.get("include_usage"))


def _reasoning_strategy(body: dict) -> str | None:
    r = body.get("reasoning")
    if r is None or r is False or r == "":
        return None
    if isinstance(r, dict):
        # OpenAI's {"effort": ...} object has no meaning here; {"strategy": ...} selects one
        r = r.get("strategy")
        if r is None:
            return None
    if not isinstance(r, str):
        raise APIError(400, "'reasoning' must be a strategy name", param="reasoning")
    return r


def _split_thinking(text: str) -> tuple[str | None, str]:
    """Separate a leading <think>...</think> block (Qwen3 thinking mode) from the answer."""
    start = text.find("<think>")
    end = text.find("</think>")
    if start >= 0 and not text[:start].strip() and end < 0:
        return text[start + len("<think>"):].strip(), ""  # cut off (max_tokens) while still thinking
    if end < 0 or (start > 0 and text[:start].strip()):
        return None, text
    inner = text[start + len("<think>") if start >= 0 else 0:end]
    return inner.strip(), text[end + len("</think>"):].lstrip()


class _ThinkingStream:
    """Streaming counterpart of _split_thinking: text inside a leading <think>...</think> goes
    out as reasoning_content deltas, the rest as content, so a streamed reply has the same
    structure as the non-streamed one. Up to len("</think>") - 1 characters are held back
    while a tag could be split across tokens."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, out: Callable[[str, str], None], opened: bool = False):
        self.out = out  # out(field, text) with field "reasoning_content" or "content"
        # start -> think -> answer_start -> answer, or start -> answer; opened: the chat template
        # already wrote "<think>" into the prompt (some thinking models' templates do)
        self.state = "think" if opened else "start"
        self.buf = ""
        self.lead = True  # strip the whitespace that opens the reasoning, as _split_thinking does

    def _reason(self, text: str) -> None:
        if self.lead:
            text = text.lstrip()
            self.lead = not text
        if text:
            self.out("reasoning_content", text)

    def feed(self, piece: str) -> None:
        self.buf += piece
        while True:
            if self.state == "start":
                s = self.buf.lstrip()
                if s.startswith(self.OPEN):
                    self.state, self.buf = "think", s[len(self.OPEN):]
                    continue
                if self.OPEN.startswith(s):  # still undecided (empty or a partial tag)
                    return
                self.state = "answer"
                continue
            if self.state == "think":
                end = self.buf.find(self.CLOSE)
                if end < 0:
                    # hold back a possible partial tag, and whitespace that may turn out to be trailing
                    cut = len(self.buf) - (len(self.CLOSE) - 1)
                    while cut > 0 and self.buf[cut - 1].isspace():
                        cut -= 1
                    if cut > 0:
                        self._reason(self.buf[:cut])
                        self.buf = self.buf[cut:]
                    return
                self._reason(self.buf[:end].rstrip())
                self.state, self.buf = "answer_start", self.buf[end + len(self.CLOSE):]
                continue
            if self.state == "answer_start":
                self.buf = self.buf.lstrip()
                if not self.buf:
                    return
                self.state = "answer"
            if self.buf:
                self.out("content", self.buf)
                self.buf = ""
            return

    def finish(self) -> None:
        if self.state == "think":
            self._reason(self.buf.rstrip())
        elif self.state in ("start", "answer") and self.buf:
            self.out("content", self.buf)
        self.buf = ""


# ---------------------------------------------------------------------------
# the service (HTTP-independent so it can be tested directly)

class Service:
    """Turns OpenAI-style request bodies into model calls. All model access holds `lock`."""

    def __init__(self, llm, model_id: str | None = None, max_tokens_default: int = 2048,
                 sampling_defaults: dict | None = None, lookup: int = 0, allow_code: bool = False):
        self.llm = llm
        self.allow_code = allow_code
        self.model_id = model_id or llm.name
        self.max_tokens_default = max_tokens_default
        self.defaults = dict(DEFAULT_SAMPLING, **(sampling_defaults or {}))
        self.lookup = lookup
        self.lock = threading.Lock()
        self.created = int(time.time())
        self.n_requests = 0

    # -- info ---------------------------------------------------------------
    def models(self) -> dict:
        return {"object": "list", "data": [self.model_card()]}

    def model_card(self) -> dict:
        return {"id": self.model_id, "object": "model", "created": self.created, "owned_by": "superborg",
                "context_length": self.llm.n_ctx}

    def health(self) -> dict:
        return {"status": "ok", "model": self.model_id, "busy": self.lock.locked(), "n_ctx": self.llm.n_ctx,
                "requests": self.n_requests}

    # -- helpers ------------------------------------------------------------
    def _gen_kwargs(self, body: dict) -> dict:
        n = _number(body, "max_completion_tokens", None, int, 0, None, aliases=("max_tokens",))
        kw = _sampling(body, self.defaults)
        kw["max_tokens"] = self.max_tokens_default if n is None else n
        kw["max_tokens_given"] = n is not None
        kw["stop"] = _stop_list(body)
        kw["lookup"] = _number(body, "lookup", self.lookup, int, 0, 16)
        return kw

    @staticmethod
    def _n_choices(body: dict, stream: bool) -> int:
        n = _number(body, "n", 1, int, 1, MAX_CHOICES)
        if stream and n != 1:
            raise APIError(400, "streaming supports n=1 only", param="n")
        return n

    def _run(self, prompt_tokens: list[int], kw: dict, stop: list[str], on_text: Callable[[str], None] | None):
        try:
            return self.llm.generate_tokens(prompt_tokens, stream=on_text, **dict(kw, stop=stop))
        except ValueError as e:  # prompt longer than the context
            raise APIError(400, str(e), code="context_length_exceeded") from e

    @staticmethod
    def _timings(gen) -> dict:
        return {"prompt_tokens_evaluated": gen.n_prompt, "prompt_tokens_cached": gen.n_reused,
                "prompt_ms": round(gen.prompt_ms, 3), "generation_ms": round(gen.gen_ms, 3),
                "tokens_per_second": round(gen.tokens_per_second, 2),
                "drafted": gen.n_drafted, "accepted": gen.n_accepted}

    # -- chat -----------------------------------------------------------------
    def chat_completion(self, body: dict, emit: Callable[[dict], None] | None = None,
                        origin: str | None = None) -> dict | None:
        """Handle a chat completion. With `emit` (streaming) chunks are passed to it and None is returned.
        origin: the request's Origin header (browsers send one), which limits code execution."""
        stream = _flag(body, "stream")
        messages = parse_messages(body)
        kw = self._gen_kwargs(body)
        max_tokens_given = kw.pop("max_tokens_given")
        n = self._n_choices(body, stream)
        strategy = _reasoning_strategy(body)
        tmpl = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
        thinking = bool(body.get("enable_thinking", tmpl.get("enable_thinking", False)))
        cid = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        include_usage = _include_usage(body)

        def chunk(delta: dict, finish: str | None = None, extra: dict | None = None) -> dict:
            c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.model_id,
                 "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
            if extra:
                c.update(extra)
            return c

        if strategy is not None:
            if n != 1:
                raise APIError(400, "reasoning strategies return one answer: use samples to vote over several",
                               param="n")
            if not max_tokens_given:
                kw["max_tokens"] = 512  # per-sample budget; strategies run several samples
            return self._reasoned_chat(body, messages, kw, strategy, thinking, cid, created,
                                       chunk if stream else None, emit, include_usage,
                                       self.allow_code and _is_local_origin(origin))

        with self.lock:
            self.n_requests += 1
            try:
                text = self.llm.render(messages, enable_thinking=thinking)
            except Exception as e:  # noqa: BLE001 - templates reject some message orders (raise_exception)
                raise APIError(400, f"the model's chat template rejected the messages: {e}",
                               param="messages") from e
            prompt = self.llm.tokenize(text, add_bos=False)  # templates spell out BOS themselves
            stop = kw.pop("stop") + self.llm.formatter.stop_strings()
            if stream:
                emit(chunk({"role": "assistant", "content": ""}))
                on_text: Callable[[str], None] = lambda piece: emit(chunk({"content": piece}))  # noqa: E731
                splitter = None
                if thinking:  # same message structure as the non-streamed reply
                    splitter = _ThinkingStream(lambda field, t: emit(chunk({field: t})),
                                               opened=text.rstrip().endswith("<think>"))
                    on_text = splitter.feed
                gen = self._run(prompt, kw, stop, on_text)
                if splitter is not None:
                    splitter.finish()
                usage = self._usage(len(prompt), len(gen.tokens))
                emit(chunk({}, _FINISH.get(gen.stop_reason, "stop"), {"timings": self._timings(gen)}))
                if include_usage:
                    emit({"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.model_id,
                          "choices": [], "usage": usage})
                return None
            choices, n_out, last = [], 0, None
            for i in range(n):
                gen = self._run(prompt, dict(kw, seed=kw["seed"] + i), stop, None)
                n_out += len(gen.tokens)
                msg: dict[str, Any] = {"role": "assistant", "content": gen.text}
                if thinking:
                    reasoning, answer = _split_thinking(gen.text)
                    if reasoning is not None:
                        msg = {"role": "assistant", "content": answer, "reasoning_content": reasoning}
                choices.append({"index": i, "message": msg, "logprobs": None,
                                "finish_reason": _FINISH.get(gen.stop_reason, "stop")})
                last = gen
        return {"id": cid, "object": "chat.completion", "created": created, "model": self.model_id,
                "choices": choices, "usage": self._usage(len(prompt), n_out), "timings": self._timings(last),
                "system_fingerprint": None}

    def _reasoned_chat(self, body, messages, kw, strategy, thinking, cid, created, chunk, emit, include_usage,
                       code_ok: bool):
        try:
            from . import reason
        except ImportError as e:
            raise APIError(501, f"reasoning is unavailable: {e}", type_="server_error") from e
        strategies = getattr(reason, "STRATEGIES", {})
        if strategy not in strategies:
            raise APIError(400, f"unknown reasoning strategy '{strategy}'; choose from {sorted(strategies)}",
                           param="reasoning")
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if not question:
            raise APIError(400, "reasoning needs a user message", param="messages")
        context = body.get("context")
        if context is not None and not isinstance(context, str):
            raise APIError(400, "'context' must be a string", param="context")
        if context is not None:
            _text(context, "'context'", "context")
        if strategy == "rag" and not context:  # checked here so a streamed request gets a plain 400
            raise APIError(400, "strategy 'rag' needs 'context' (reference text)", param="context")
        samples = _number(body, "samples", 5, int, 1, 64)
        route = ""
        if not code_ok:
            if strategy in CODE_STRATEGIES:
                raise APIError(403, f"strategy '{strategy}' runs model-written code, which this server does not "
                                    "allow (start it with --allow-code; never for browser requests from other "
                                    "origins)", type_="permission_error", param="reasoning")
            if strategy == "auto":
                kind, why = reason.classify(question, context)
                strategy = _AUTO_WITHOUT_CODE.get(kind, "cot")
                route = f"{kind}: {why}; code execution disabled"
        extra = {"enable_thinking": thinking}
        # strategies have their own sampling defaults; override them only when the client asks
        if body.get("temperature") is not None:
            extra["temperature"] = _temperature(body, None)
        if body.get("top_p") is not None:
            extra["top_p"] = _number(body, "top_p", None, float, 0.0, 1.0)
        if chunk is not None:
            emit(chunk({"role": "assistant", "content": ""}))
            # single-pass strategies stream model tokens, sampling ones a line per candidate:
            # both are working notes, so they go out as reasoning_content like a thinking model's
            extra["stream"] = lambda piece: emit(chunk({"reasoning_content": piece}))
        with self.lock:
            self.n_requests += 1
            guard = contextlib.nullcontext() if code_ok else _code_execution_blocked(reason.tools)
            try:
                with guard:
                    ans = reason.solve(self.llm, question, strategy=strategy, samples=samples,
                                       max_tokens=kw["max_tokens"],
                                       seed=kw["seed"] if body.get("seed") is not None else 0,
                                       context=context, **extra)
                # inside the lock: shutdown frees the model once it holds the lock
                prompt_tokens = len(self.llm.tokenize(question, add_bos=False))
            except CodeExecutionDisabled as e:
                raise APIError(403, f"answering this needs code execution: {e}", type_="permission_error",
                               param="reasoning") from e
            except ValueError as e:  # e.g. "rag" without context, or a prompt longer than the context
                raise APIError(400, str(e), param="reasoning") from e
        if route:
            ans.route = route
        info = {"strategy": ans.strategy, "answer": ans.answer, "route": getattr(ans, "route", ""),
                "elapsed_s": round(float(ans.elapsed_s), 3), "tokens": ans.tokens,
                "candidates": len(ans.candidates or [])}
        usage = self._usage(prompt_tokens, int(ans.tokens or 0))
        if chunk is not None:
            # the final answer exists only once the strategy has compared its candidates
            emit(chunk({"content": ans.text}))
            emit(chunk({}, "stop", {"reasoning": info}))
            if include_usage:
                emit({"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.model_id,
                      "choices": [], "usage": usage})
            return None
        return {"id": cid, "object": "chat.completion", "created": created, "model": self.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ans.text}, "logprobs": None,
                             "finish_reason": "stop"}],
                "usage": usage, "reasoning": info, "system_fingerprint": None}

    # -- raw completion ----------------------------------------------------
    def completion(self, body: dict, emit: Callable[[dict], None] | None = None) -> dict | None:
        stream = _flag(body, "stream")
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        token_prompt = (isinstance(prompt, list) and prompt
                        and all(isinstance(t, int) and not isinstance(t, bool) for t in prompt))
        if not token_prompt and (not isinstance(prompt, str) or not prompt):
            raise APIError(400, "'prompt' must be a non-empty string (or a list of token ids)", param="prompt")
        if not token_prompt:
            _text(prompt, "'prompt'", "prompt")
        kw = self._gen_kwargs(body)
        kw.pop("max_tokens_given")
        n = self._n_choices(body, stream)
        echo = _flag(body, "echo")
        include_usage = _include_usage(body)
        cid = "cmpl-" + secrets.token_hex(12)
        created = int(time.time())

        def chunk(text: str, finish: str | None = None, extra: dict | None = None) -> dict:
            c = {"id": cid, "object": "text_completion", "created": created, "model": self.model_id,
                 "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}
            if extra:
                c.update(extra)
            return c

        with self.lock:
            self.n_requests += 1
            if token_prompt:
                tokens = list(prompt)
                if any(t < 0 or t >= self.llm.model.n_vocab for t in tokens):
                    raise APIError(400, "token id out of range", param="prompt")
                prompt_text = self.llm.detokenize(tokens)
            else:
                tokens = self.llm.tokenize(prompt)
                prompt_text = prompt
            stop = kw.pop("stop")
            if stream:
                if echo:
                    emit(chunk(prompt_text))
                gen = self._run(tokens, kw, stop, lambda piece: emit(chunk(piece)))
                emit(chunk("", _FINISH.get(gen.stop_reason, "stop"), {"timings": self._timings(gen)}))
                if include_usage:
                    emit({"id": cid, "object": "text_completion", "created": created, "model": self.model_id,
                          "choices": [], "usage": self._usage(len(tokens), len(gen.tokens))})
                return None
            choices, n_out, last = [], 0, None
            for i in range(n):
                gen = self._run(tokens, dict(kw, seed=kw["seed"] + i), stop, None)
                n_out += len(gen.tokens)
                choices.append({"index": i, "text": (prompt_text if echo else "") + gen.text, "logprobs": None,
                                "finish_reason": _FINISH.get(gen.stop_reason, "stop")})
                last = gen
        return {"id": cid, "object": "text_completion", "created": created, "model": self.model_id,
                "choices": choices, "usage": self._usage(len(tokens), n_out), "timings": self._timings(last)}

    @staticmethod
    def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}


# ---------------------------------------------------------------------------
# HTTP layer

class _ClientGone(Exception):
    """The client closed the connection or stopped reading (a write failed or timed out)."""


class Handler(BaseHTTPRequestHandler):
    server_version = "superborg/0.1"
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_TIMEOUT_S  # per blocking socket operation, not per request: generation may take longer
    service: Service  # set on the subclass made by make_server
    api_key: str | None = None
    quiet: bool = True

    # -- plumbing -------------------------------------------------------------
    def log_message(self, fmt, *args):  # noqa: D102 - stdlib signature
        if not self.quiet:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, err: APIError):
        # the request body may be unread (too large, unknown route): never reuse the connection
        self.close_connection = True
        data = json.dumps(err.body(), ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(err.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.wfile.write(data)
        except OSError:  # the client is gone (e.g. it sent a short body and hung up)
            pass

    def _authorized(self) -> bool:
        if not self.api_key:
            return True
        return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {self.api_key}")

    def _read_json(self) -> dict:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            raise APIError(411, "chunked request bodies are not supported; send Content-Length")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as e:
            raise APIError(400, "bad Content-Length") from e
        if length > MAX_BODY_BYTES:
            raise APIError(413, f"request body exceeds {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        # ValueError covers JSONDecodeError, UnicodeDecodeError and over-long integer literals;
        # RecursionError comes from deeply nested arrays/objects
        except (ValueError, RecursionError) as e:
            raise APIError(400, f"invalid JSON body: {e}") from e
        if not isinstance(body, dict):
            raise APIError(400, "request body must be a JSON object")
        return body

    # -- methods --------------------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802 - CORS preflight
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        svc = self.service
        try:
            if path in ("/health", "/v1/health"):
                return self._send_json(200, svc.health())
            if not self._authorized():
                raise APIError(401, "invalid API key", type_="authentication_error")
            if path == "/v1/models":
                return self._send_json(200, svc.models())
            if path.startswith("/v1/models/"):
                # ids like "Qwen/Qwen2.5-0.5B-Instruct" arrive percent-encoded from some clients
                mid = urllib.parse.unquote(path[len("/v1/models/"):])
                if mid != svc.model_id:
                    raise APIError(404, f"model '{mid}' not found", code="model_not_found")
                return self._send_json(200, svc.model_card())
            if path in ("/v1/chat/completions", "/v1/completions"):
                raise APIError(405, f"use POST for {path}")
            raise APIError(404, f"no route for GET {path}")
        except APIError as e:
            self._send_error(e)
        except OSError:  # client gone while the reply was written
            self.close_connection = True

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        svc = self.service
        handlers = {"/v1/chat/completions": svc.chat_completion, "/chat/completions": svc.chat_completion,
                    "/v1/completions": svc.completion, "/completions": svc.completion}
        try:
            if path not in handlers:
                raise APIError(404, f"no route for POST {path}")
            body = self._read_json()
            if not self._authorized():
                raise APIError(401, "invalid API key", type_="authentication_error")
            call = handlers[path]
            if call == svc.chat_completion:
                call = functools.partial(svc.chat_completion, origin=self.headers.get("Origin"))
            if _flag(body, "stream"):
                return self._stream(call, body)
            result = call(body)
        except APIError as e:
            return self._send_error(e)
        except Exception as e:  # noqa: BLE001 - report instead of dropping the connection
            traceback.print_exc(file=sys.stderr)
            return self._send_error(APIError(500, f"{type(e).__name__}: {e}", type_="server_error"))
        try:
            self._send_json(200, result)
        except OSError:  # the client hung up while waiting for its answer
            self.close_connection = True

    def _stream(self, call, body: dict) -> None:
        """Server-sent events. Errors before the first event are plain JSON errors; later ones go
        in-band (an "error" event), like OpenAI's. A failed write means the client is gone:
        the generation stops (the engine binding turns the callback's exception into a stop)."""
        started = []

        def emit(obj: dict):
            try:
                if not started:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self._cors()
                    self.end_headers()
                    started.append(True)
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except OSError as e:  # BrokenPipe, ConnectionReset, or a write timeout
                raise _ClientGone() from e

        self.close_connection = True  # the stream has no length; closing ends it
        try:
            try:
                call(body, emit)
            except _ClientGone:
                return
            except APIError as e:
                if not started:
                    return self._send_error(e)
                emit(e.body())  # headers are out; report in-band like OpenAI does
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                err = APIError(500, f"{type(e).__name__}: {e}", type_="server_error")
                if not started:
                    return self._send_error(err)
                emit(err.body())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (_ClientGone, OSError):
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    service: Service


class Server6(Server):
    address_family = socket.AF_INET6


def make_server(llm, host: str = "127.0.0.1", port: int = 8080, model_id: str | None = None,
                api_key: str | None = None, quiet: bool = True, **service_kw) -> Server:
    """Create (but do not start) a server around an already loaded superborg.LLM. port=0 picks a free port."""
    service = Service(llm, model_id=model_id, **service_kw)
    handler = type("BoundHandler", (Handler,), {"service": service, "api_key": api_key, "quiet": quiet})
    # the stdlib server is IPv4-only: an IPv6 literal ("::1", "::") needs an AF_INET6 socket. Names
    # stay IPv4 so that "localhost" keeps answering on 127.0.0.1.
    srv = (Server6 if ":" in host else Server)((host, port), handler)
    srv.service = service
    return srv


def serve(model_path: str, host: str = "127.0.0.1", port: int = 8080, threads: int = 0, n_ctx: int = 4096,
          draft: str | None = None, api_key: str | None = None, lookup: int = 0, kv_f16: bool = False,
          model_id: str | None = None, quiet: bool = False, allow_code: bool = False) -> None:
    """Load a model and serve it until interrupted."""
    from . import LLM

    llm = LLM(model_path, threads=threads, n_ctx=n_ctx, draft=draft, kv_f16=kv_f16)
    srv = make_server(llm, host, port, model_id=model_id, api_key=api_key, quiet=quiet, lookup=lookup,
                      allow_code=allow_code)
    h, p = srv.server_address[:2]
    h = f"[{h}]" if ":" in h else h
    print(f"superborg: serving {srv.service.model_id} (n_ctx {llm.n_ctx}) at http://{h}:{p}/v1", file=sys.stderr,
          flush=True)
    if allow_code and host not in ("127.0.0.1", "localhost", "::1"):
        print("superborg: WARNING: --allow-code on a non-local address lets any client that can reach this port "
              "run model-written code here", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        # daemon request threads may still be generating; free the model only once they are done
        if srv.service.lock.acquire(timeout=30):
            llm.close()
