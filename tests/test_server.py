"""The OpenAI-compatible HTTP server (superborg/server.py), exercised over real HTTP.

A server around smollm2-135m-q8_0.borg runs in a background thread on a free port.
Models are found like in test_engine.py; tests skip when the model file is missing.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
import types
import urllib.parse

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
THREADS = int(os.environ.get("BORG_TEST_THREADS", "4"))
MODEL = "smollm2-135m-q8_0.borg"


def model_path(name: str) -> str:
    dirs = [os.environ.get("BORG_MODELS_DIR", ""), os.path.join(ROOT, "models"),
            os.path.expanduser("~/.cache/superborg")]
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import hf_oracle  # noqa: PLC0415 - optional: this checkout's default model directory

        dirs.append(hf_oracle.BORG_MODELS_DIR)
    except Exception:  # noqa: BLE001
        pass
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    pytest.skip(f"{name} not found (set BORG_MODELS_DIR)")


@pytest.fixture(scope="module")
def llm():
    try:
        from superborg import LLM
        lm = LLM(model_path(MODEL), threads=THREADS, n_ctx=1024)
    except (OSError, FileNotFoundError) as e:
        pytest.skip(f"engine unavailable: {e}")
    yield lm
    lm.close()


def _start(llm, host: str = "127.0.0.1", **kw):
    from superborg.server import make_server
    srv = make_server(llm, host, 0, **kw)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th


@pytest.fixture(scope="module")
def server(llm):
    srv, th = _start(llm)
    yield srv
    srv.shutdown()
    srv.server_close()
    th.join(timeout=10)


def call(srv, method: str, path: str, body=None, headers: dict | None = None, raw: bytes | None = None):
    """Returns (status, headers, parsed JSON or raw bytes)."""
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=300)
    try:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        payload = r.read()
        try:
            parsed = json.loads(payload) if payload else None
        except json.JSONDecodeError:
            parsed = payload
        return r.status, dict(r.getheaders()), parsed
    finally:
        conn.close()


def stream(srv, path: str, body: dict) -> tuple[int, dict, list]:
    """POST with stream=true; returns (status, headers, list of decoded SSE events; "[DONE]" kept as str)."""
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=300)
    try:
        conn.request("POST", path, body=json.dumps(dict(body, stream=True)),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        events = []
        if r.status != 200:
            return r.status, dict(r.getheaders()), [json.loads(r.read())]
        text = r.read().decode("utf-8")  # the server closes the connection after the stream
        for block in text.split("\n\n"):
            if not block.strip():
                continue
            assert block.startswith("data: "), block
            data = block[len("data: "):]
            events.append(data if data == "[DONE]" else json.loads(data))
        return r.status, dict(r.getheaders()), events
    finally:
        conn.close()


CHAT = {"messages": [{"role": "user", "content": "Name three colors."}], "max_tokens": 24, "temperature": 0}


def warm_call(srv, path: str, req: dict):
    """The second of two identical requests (both served from a warm prompt cache)."""
    call(srv, "POST", path, req)
    return call(srv, "POST", path, req)


# ---------------------------------------------------------------------------
# info endpoints

def test_health_and_models(server):
    st, hdr, body = call(server, "GET", "/health")
    assert st == 200 and body["status"] == "ok" and body["n_ctx"] == 1024
    assert hdr.get("Access-Control-Allow-Origin") == "*"
    st, _, body = call(server, "GET", "/v1/models")
    assert st == 200 and body["object"] == "list"
    mid = body["data"][0]["id"]
    assert mid == server.service.model_id and body["data"][0]["object"] == "model"
    st, _, body = call(server, "GET", f"/v1/models/{mid}")
    assert st == 200 and body["id"] == mid
    st, _, body = call(server, "GET", "/v1/models/no-such-model")
    assert st == 404 and body["error"]["code"] == "model_not_found"
    # ids with a slash ("HuggingFaceTB/SmolLM2-135M-Instruct") may arrive percent-encoded
    st, _, body = call(server, "GET", "/v1/models/" + urllib.parse.quote(mid, safe=""))
    assert st == 200 and body["id"] == mid


def test_cors_preflight(server):
    st, hdr, _ = call(server, "OPTIONS", "/v1/chat/completions")
    assert st == 204
    assert hdr["Access-Control-Allow-Origin"] == "*"
    assert "POST" in hdr["Access-Control-Allow-Methods"]
    assert "Authorization" in hdr["Access-Control-Allow-Headers"]


# ---------------------------------------------------------------------------
# chat completions

def test_chat_completion_matches_library(server, llm):
    st, hdr, body = warm_call(server, "/v1/chat/completions", CHAT)
    assert st == 200, body
    assert hdr["Content-Type"].startswith("application/json")
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    ch = body["choices"][0]
    assert ch["index"] == 0 and ch["message"]["role"] == "assistant"
    assert ch["finish_reason"] in ("stop", "length")
    u = body["usage"]
    assert 0 < u["completion_tokens"] <= 24 and u["prompt_tokens"] > 5
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert body["timings"]["tokens_per_second"] >= 0
    # the server is a thin layer: greedy output equals the library call
    with server.service.lock:
        ref = llm.chat(CHAT["messages"], max_tokens=24, temperature=0.0)
    assert ch["message"]["content"] == ref.text
    assert u["completion_tokens"] == len(ref.tokens)


def test_chat_seed_reproducible(server):
    req = dict(CHAT, temperature=0.9, seed=1234, max_tokens=16)
    a = call(server, "POST", "/v1/chat/completions", req)[2]
    b = call(server, "POST", "/v1/chat/completions", req)[2]
    assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]
    outs = {call(server, "POST", "/v1/chat/completions", dict(req, seed=s))[2]["choices"][0]["message"]["content"]
            for s in range(4)}
    assert len(outs) > 1


def test_chat_stop_strings(server):
    ref = warm_call(server, "/v1/chat/completions", CHAT)[2]["choices"][0]["message"]["content"]
    assert len(ref) > 12, ref
    stop = ref[5:9]
    cut = ref.find(stop)
    for value in (stop, [stop, "@@nope@@"]):
        body = call(server, "POST", "/v1/chat/completions", dict(CHAT, stop=value))[2]
        assert body["choices"][0]["message"]["content"] == ref[:cut]
        assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_max_tokens_and_aliases(server):
    body = call(server, "POST", "/v1/chat/completions", dict(CHAT, max_tokens=None, max_completion_tokens=3))[2]
    assert body["usage"]["completion_tokens"] <= 3
    assert body["choices"][0]["finish_reason"] == "length"
    body = call(server, "POST", "/v1/chat/completions", dict(CHAT, max_tokens=0))[2]
    assert body["choices"][0]["message"]["content"] == "" and body["usage"]["completion_tokens"] == 0


def test_chat_stream(server):
    ref = warm_call(server, "/v1/chat/completions", CHAT)[2]
    st, hdr, events = stream(server, "/v1/chat/completions", dict(CHAT, stream_options={"include_usage": True}))
    assert st == 200 and hdr["Content-Type"] == "text/event-stream"
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == [] and usage_chunk["usage"] == ref["usage"]
    final = chunks[-2]["choices"][0]
    assert final["finish_reason"] == ref["choices"][0]["finish_reason"] and final["delta"] == {}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[:-2])
    assert text == ref["choices"][0]["message"]["content"]
    assert sum(1 for c in chunks[:-2] if c["choices"][0]["delta"].get("content")) > 1, "expected several deltas"


def test_chat_stream_with_stop(server):
    ref = warm_call(server, "/v1/chat/completions", CHAT)[2]["choices"][0]["message"]["content"]
    stop = ref[6:10]
    _, _, events = stream(server, "/v1/chat/completions", dict(CHAT, stop=stop))
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in events[:-1] if c["choices"])
    assert text == ref[:ref.find(stop)]
    assert events[-2]["choices"][0]["finish_reason"] == "stop"


def test_chat_n_choices_and_ignored_fields(server):
    req = dict(CHAT, n=2, temperature=1.0, seed=5, max_tokens=8, response_format={"type": "json_object"},
               presence_penalty=0.5, frequency_penalty=0.1, logprobs=False, user="tester", model="anything",
               tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}], top_k=20, min_p=0.1,
               top_p=0.9, repeat_penalty=1.1)
    st, _, body = call(server, "POST", "/v1/chat/completions", req)
    assert st == 200, body
    assert [c["index"] for c in body["choices"]] == [0, 1]
    assert body["usage"]["completion_tokens"] <= 16


def test_chat_message_forms(server):
    msgs = [{"role": "developer", "content": "Answer briefly."},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": " there"}]},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Bye"}]
    st, _, body = call(server, "POST", "/v1/chat/completions", {"messages": msgs, "max_tokens": 4})
    assert st == 200, body


def test_chat_reuses_prefix_cache(server):
    msgs = [{"role": "system", "content": "You are a helpful assistant. " * 8},
            {"role": "user", "content": "Say one word."}]
    call(server, "POST", "/v1/chat/completions", {"messages": msgs, "max_tokens": 2, "temperature": 0})
    msgs2 = msgs + [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Another."}]
    body = call(server, "POST", "/v1/chat/completions", {"messages": msgs2, "max_tokens": 2, "temperature": 0})[2]
    assert body["timings"]["prompt_tokens_cached"] > 40


# ---------------------------------------------------------------------------
# raw completions

def test_completion_matches_library(server, llm):
    req = {"prompt": "The capital of France is", "max_tokens": 10, "temperature": 0}
    st, _, body = warm_call(server, "/v1/completions", req)
    assert st == 200, body
    assert body["object"] == "text_completion" and body["id"].startswith("cmpl-")
    with server.service.lock:
        ref = llm.complete("The capital of France is", max_tokens=10, temperature=0.0)
    ch = body["choices"][0]
    assert ch["text"] == ref.text and ch["index"] == 0
    assert body["usage"]["completion_tokens"] == len(ref.tokens)
    assert body["usage"]["prompt_tokens"] == len(llm.tokenize("The capital of France is"))
    # echo, list-of-one prompt and token-id prompt give the same continuation
    body = call(server, "POST", "/v1/completions", dict(req, echo=True))[2]
    assert body["choices"][0]["text"] == "The capital of France is" + ref.text
    body = call(server, "POST", "/v1/completions", dict(req, prompt=["The capital of France is"]))[2]
    assert body["choices"][0]["text"] == ref.text
    toks = llm.tokenize("The capital of France is")
    body = call(server, "POST", "/v1/completions", dict(req, prompt=toks))[2]
    assert body["choices"][0]["text"] == ref.text


def test_completion_stream(server):
    req = {"prompt": "Once upon a time", "max_tokens": 12, "temperature": 0}
    ref = warm_call(server, "/v1/completions", req)[2]
    st, _, events = stream(server, "/v1/completions", req)
    assert st == 200 and events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "text_completion" for c in chunks)
    assert "".join(c["choices"][0]["text"] for c in chunks) == ref["choices"][0]["text"]
    assert chunks[-1]["choices"][0]["finish_reason"] == ref["choices"][0]["finish_reason"]


def test_completion_stream_usage(server):
    req = {"prompt": "Once upon a time", "max_tokens": 6, "temperature": 0}
    ref = warm_call(server, "/v1/completions", req)[2]
    st, _, events = stream(server, "/v1/completions", dict(req, stream_options={"include_usage": True}))
    assert st == 200 and events[-1] == "[DONE]"
    usage = events[-2]
    assert usage["choices"] == [] and usage["usage"] == ref["usage"] and usage["object"] == "text_completion"
    assert events[-3]["choices"][0]["finish_reason"] == ref["choices"][0]["finish_reason"]


def test_tiny_temperature_is_greedy(server):
    """1/T overflows float32 for T near 1e-38: the logits become inf and the probabilities NaN,
    and the engine then emits token 0 forever. Such temperatures are greedy in effect."""
    req = {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
    ref = warm_call(server, "/v1/completions", req)[2]["choices"][0]["text"]
    out = call(server, "POST", "/v1/completions", dict(req, temperature=1e-38))[2]["choices"][0]["text"]
    assert out == ref


def test_completion_lookup_decoding(server):
    text = "alpha beta gamma delta epsilon alpha beta gamma delta epsilon alpha beta"
    plain = warm_call(server, "/v1/completions", {"prompt": text, "max_tokens": 12, "temperature": 0})[2]
    spec = call(server, "POST", "/v1/completions",
                {"prompt": text, "max_tokens": 12, "temperature": 0, "lookup": 3})[2]
    assert spec["timings"]["drafted"] > 0
    assert spec["usage"] == plain["usage"]


# ---------------------------------------------------------------------------
# errors

@pytest.mark.parametrize("path,body,status,needle", [
    ("/v1/chat/completions", {}, 400, "messages"),
    ("/v1/chat/completions", {"messages": []}, 400, "messages"),
    ("/v1/chat/completions", {"messages": [{"content": "x"}]}, 400, "role"),
    ("/v1/chat/completions", {"messages": [{"role": "wizard", "content": "x"}]}, 400, "role"),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": 5}]}, 400, "content"),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, 400, "image_url"),
    # a text part's text must be a string: not str(None) or a dict's repr in the prompt, and no part is dropped
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "text", "text": None}]}]}, 400,
     "content[0]: 'text' must be a string"),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": ["a", {"type": "text", "text": {"a": 1}}]}]},
     400, "content[1]: 'text' must be a string"),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "text"}]}]}, 400, "'text'"),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": ["a", 5]}]}, 400, "content[1] must be"),
    ("/v1/chat/completions", dict(CHAT, temperature="hot"), 400, "temperature"),
    ("/v1/chat/completions", dict(CHAT, temperature=5), 400, "temperature"),
    ("/v1/chat/completions", dict(CHAT, max_tokens=-1), 400, "max_tokens"),
    ("/v1/chat/completions", dict(CHAT, max_tokens=2.5), 400, "max_tokens"),
    ("/v1/chat/completions", dict(CHAT, max_tokens=True), 400, "max_tokens"),
    ("/v1/chat/completions", dict(CHAT, top_p=1.5), 400, "top_p"),
    ("/v1/chat/completions", dict(CHAT, stop=[1, 2]), 400, "stop"),
    ("/v1/chat/completions", dict(CHAT, stop=["s"] * 40), 400, "stop"),
    ("/v1/chat/completions", dict(CHAT, n=2, stream=True), 400, "n=1"),
    ("/v1/chat/completions", dict(CHAT, n=100), 400, "'n'"),
    ("/v1/completions", {"prompt": ""}, 400, "prompt"),
    ("/v1/completions", {"prompt": 42}, 400, "prompt"),
    ("/v1/completions", {"prompt": [10 ** 9]}, 400, "range"),
    ("/v1/embeddings", {"input": "x"}, 404, "no route"),
    # json.dumps writes NaN/Infinity literals, which json.loads accepts; NaN passes range checks
    ("/v1/chat/completions", dict(CHAT, temperature=float("nan")), 400, "finite"),
    ("/v1/completions", {"prompt": "Hi", "top_p": float("nan")}, 400, "finite"),
    ("/v1/completions", {"prompt": "Hi", "min_p": float("nan")}, 400, "finite"),
    ("/v1/completions", {"prompt": "Hi", "repeat_penalty": 0}, 400, "repeat_penalty"),
    ("/v1/completions", {"prompt": "Hi", "top_k": 10 ** 12}, 400, "top_k"),
    ("/v1/completions", {"prompt": "Hi", "stream": "false"}, 400, "boolean"),
    ("/v1/completions", {"prompt": "Hi", "echo": "yes"}, 400, "boolean"),
    ("/v1/completions", {"prompt": "Hi", "stop": ["x" * 10000]}, 400, "stop"),
    # lone UTF-16 surrogates are valid JSON escapes but not encodable text
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "a\ud800b"}]}, 400, "Unicode"),
    ("/v1/completions", {"prompt": "a\udfffb", "echo": True}, 400, "Unicode"),
    ("/v1/completions", {"prompt": "Hi", "stop": ["\ud800"]}, 400, "Unicode"),
    ("/v1/chat/completions", dict(CHAT, reasoning="direct", n=2), 400, "one answer"),
])
def test_bad_requests(server, path, body, status, needle):
    st, hdr, out = call(server, "POST", path, body)
    assert st == status, out
    assert set(out["error"]) >= {"message", "type", "param", "code"}
    assert needle in out["error"]["message"]
    assert hdr.get("Access-Control-Allow-Origin") == "*"


def test_malformed_bodies(server):
    st, _, out = call(server, "POST", "/v1/chat/completions", raw=b"{not json")
    assert st == 400 and "invalid JSON" in out["error"]["message"]
    st, _, out = call(server, "POST", "/v1/chat/completions", raw=b"[1, 2]")
    assert st == 400 and "object" in out["error"]["message"]
    st, _, out = call(server, "POST", "/v1/chat/completions", raw=b"\xff\xfe")
    assert st == 400
    # parser limits are client errors too, not 500s
    st, _, out = call(server, "POST", "/v1/completions", raw=b"[" * 100000 + b"]" * 100000)
    assert st == 400 and "invalid JSON" in out["error"]["message"]
    st, _, out = call(server, "POST", "/v1/completions", raw=b'{"prompt": "Hi", "max_tokens": ' + b"9" * 5000 + b"}")
    assert st == 400 and "invalid JSON" in out["error"]["message"]
    st, _, out = call(server, "GET", "/v1/chat/completions")
    assert st == 405
    st, _, out = call(server, "GET", "/nothing/here")
    assert st == 404


def test_request_size_limit(server):
    from superborg.server import MAX_BODY_BYTES
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=60)
    try:
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        conn.endheaders()
        conn.send(b"{}")  # the server answers from the header without reading the body
        r = conn.getresponse()
        out = json.loads(r.read())
        assert r.status == 413 and "exceeds" in out["error"]["message"]
    finally:
        conn.close()


def test_prompt_too_long(server):
    long_text = "word " * 3000
    st, _, out = call(server, "POST", "/v1/chat/completions",
                      {"messages": [{"role": "user", "content": long_text}], "max_tokens": 4})
    assert st == 400 and out["error"]["code"] == "context_length_exceeded"
    st, _, out = call(server, "POST", "/v1/completions", {"prompt": long_text, "max_tokens": 4})
    assert st == 400 and out["error"]["code"] == "context_length_exceeded"
    # the server keeps working afterwards
    assert call(server, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 2})[0] == 200


def test_internal_error_is_json_500(server, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(server.service.llm, "generate_tokens", boom)
    st, _, out = call(server, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 2})
    assert st == 500 and "engine exploded" in out["error"]["message"]
    assert out["error"]["type"] == "server_error"
    # a failure before any data is sent is a plain JSON error ...
    st, _, events = stream(server, "/v1/completions", {"prompt": "Hi", "max_tokens": 2})
    assert st == 500 and "engine exploded" in events[0]["error"]["message"]
    # ... after the first chunk it is reported in-band, then the stream ends normally
    st, _, events = stream(server, "/v1/chat/completions", CHAT)
    assert st == 200 and events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "engine exploded" in events[-2]["error"]["message"]
    monkeypatch.undo()
    assert call(server, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 2})[0] == 200
    assert not server.service.lock.locked()


# ---------------------------------------------------------------------------
# concurrency, disconnects, auth

def test_concurrent_requests_serialized(server):
    # distinct first words: every request is a fresh prefill whatever order they run in
    prompts = ("One, two,", "Red, green,", "Monday, Tuesday,", "January, February,", "Alpha, beta,", "Do, re,")
    reqs = [{"prompt": p, "max_tokens": 8, "temperature": 0} for p in prompts]
    expected = [call(server, "POST", "/v1/completions", r)[2]["choices"][0]["text"] for r in reqs]
    results: dict[int, str] = {}

    def worker(i):
        results[i] = call(server, "POST", "/v1/completions", reqs[i])[2]["choices"][0]["text"]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(reqs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert [results[i] for i in range(len(reqs))] == expected


def test_client_disconnect_mid_stream(server):
    host, port = server.server_address[:2]
    sock = socket.create_connection((host, port), timeout=60)
    body = json.dumps({"prompt": "Count to one hundred:", "max_tokens": 200, "temperature": 0,
                       "stream": True}).encode()
    sock.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    got = b""
    while b"data: " not in got:  # headers and the first event may arrive separately
        part = sock.recv(4096)
        assert part, "connection closed before the first event"
        got += part
    sock.close()  # the generation stops at the next token write
    # ... well before its 200 tokens: the lock is free again within a few token times
    deadline = time.time() + 30
    while call(server, "GET", "/health")[2]["busy"] and time.time() < deadline:
        time.sleep(0.05)
    assert not call(server, "GET", "/health")[2]["busy"]
    st, _, out = call(server, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 2})
    assert st == 200, out


def test_short_body_then_hangup(server, capsys):
    """A client that announces more body than it sends and hangs up gets a 400 it never reads;
    the handler must cope (no traceback) and the server keeps serving."""
    host, port = server.server_address[:2]
    for _ in range(3):
        sock = socket.create_connection((host, port), timeout=30)
        sock.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{")
        sock.shutdown(socket.SHUT_WR)
        sock.close()
    assert call(server, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 1})[0] == 200
    time.sleep(0.3)
    assert "Traceback" not in capsys.readouterr().err


def test_handler_has_socket_timeout():
    # without one, a client that stops sending (or reading) pins a handler thread forever
    from superborg.server import Handler
    assert Handler.timeout and Handler.timeout > 0


def test_ipv6_host(llm):
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    try:
        srv, th = _start(llm, host="::1")
    except OSError as e:
        pytest.skip(f"cannot bind ::1: {e}")
    try:
        conn = http.client.HTTPConnection("::1", srv.server_address[1], timeout=30)
        conn.request("GET", "/health")
        assert conn.getresponse().status == 200
        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=10)


def test_code_guard_overlapping_servers():
    """The run_python guard patches a module attribute; overlapping guards (two servers in one
    process) must still restore the real function at the end."""
    from superborg.server import CodeExecutionDisabled, _code_execution_blocked

    def real(*a, **k):
        return "ran"

    mod = types.SimpleNamespace(run_python=real)
    a, b = _code_execution_blocked(mod), _code_execution_blocked(mod)
    a.__enter__()
    b.__enter__()
    with pytest.raises(CodeExecutionDisabled):
        mod.run_python("print(1)")
    a.__exit__(None, None, None)
    with pytest.raises(CodeExecutionDisabled):  # b is still active
        mod.run_python("print(1)")
    b.__exit__(None, None, None)
    assert mod.run_python is real


def test_api_key(llm):
    srv, th = _start(llm, api_key="s3cret")
    try:
        assert call(srv, "GET", "/health")[0] == 200  # health stays open for load balancers
        st, _, out = call(srv, "GET", "/v1/models")
        assert st == 401 and out["error"]["type"] == "authentication_error"
        st, _, _ = call(srv, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 1},
                        headers={"Authorization": "Bearer wrong"})
        assert st == 401
        st, _, _ = call(srv, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 1},
                        headers={"Authorization": "Bearer s3cret"})
        assert st == 200
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=10)


def test_keep_alive_connection(server):
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=60)
    try:
        for _ in range(3):  # several requests over one HTTP/1.1 connection
            conn.request("POST", "/v1/completions", body=json.dumps({"prompt": "Hi", "max_tokens": 1}),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            assert r.status == 200 and json.loads(r.read())["object"] == "text_completion"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# reasoning extension

def _reason_or_skip():
    try:
        from superborg import reason
    except ImportError as e:
        pytest.skip(f"superborg.reason unavailable: {e}")
    return reason


def test_reasoning_calc(server):
    _reason_or_skip()
    req = {"messages": [{"role": "user", "content": "What is 12 * 12 + 1?"}], "reasoning": "auto"}
    st, _, body = call(server, "POST", "/v1/chat/completions", req)
    assert st == 200, body
    assert body["reasoning"]["strategy"] == "calc" and body["reasoning"]["answer"] == "145"
    assert "145" in body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "stop"
    # the {"strategy": ...} object form selects the same thing
    body = call(server, "POST", "/v1/chat/completions", dict(req, reasoning={"strategy": "calc"}))[2]
    assert body["reasoning"]["answer"] == "145"
    # OpenAI's {"effort": ...} object is not a strategy and is ignored
    body = call(server, "POST", "/v1/chat/completions", dict(req, reasoning={"effort": "low"}, max_tokens=4))[2]
    assert "reasoning" not in body


def test_reasoning_model_strategy_and_stream(server):
    _reason_or_skip()
    req = {"messages": [{"role": "user", "content": "What is the capital of France?"}], "reasoning": "direct",
           "max_tokens": 16}
    st, _, body = call(server, "POST", "/v1/chat/completions", req)
    assert st == 200, body
    assert body["reasoning"]["strategy"] == "direct"
    text = body["choices"][0]["message"]["content"]
    assert text and body["usage"]["completion_tokens"] > 0
    st, _, events = stream(server, "/v1/chat/completions", req)
    assert st == 200 and events[-1] == "[DONE]"
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert content == text
    assert events[-2]["reasoning"]["strategy"] == "direct"


def test_reasoning_code_execution_policy(server, llm):
    reason = _reason_or_skip()
    word_problem = "Tom has 3 boxes with 4 apples each and eats 2 apples. How many apples are left?"
    q = {"messages": [{"role": "user", "content": word_problem}], "samples": 2, "max_tokens": 24}
    for strategy in ("pal", "pal_vote"):
        st, _, out = call(server, "POST", "/v1/chat/completions", dict(q, reasoning=strategy))
        assert st == 403 and "--allow-code" in out["error"]["message"]
    # "auto" routes math to voting instead of programs
    st, _, body = call(server, "POST", "/v1/chat/completions", dict(q, reasoning="auto"))
    assert st == 200, body
    assert body["reasoning"]["strategy"] == "vote" and "code execution disabled" in body["reasoning"]["route"]
    # the calculator's fallback to a program is blocked too, and run_python is restored afterwards
    run_python = reason.tools.run_python
    st, _, out = call(server, "POST", "/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "How tall is the tallest tree?"}],
                       "reasoning": "calc", "max_tokens": 24})
    assert st == 403 and "code execution" in out["error"]["message"]
    assert reason.tools.run_python is run_python

    srv, th = _start(llm, allow_code=True)
    try:
        st, _, out = call(srv, "POST", "/v1/chat/completions", dict(q, reasoning="pal"),
                          headers={"Origin": "https://evil.example"})
        assert st == 403, out
        st, _, body = call(srv, "POST", "/v1/chat/completions", dict(q, reasoning="pal", max_tokens=64),
                           headers={"Origin": "http://localhost:3000"})
        assert st == 200 and body["reasoning"]["strategy"] == "pal", body
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=10)


def test_reasoning_rag_and_errors(server):
    _reason_or_skip()
    q = {"role": "user", "content": "What is the wifi password?"}
    body = call(server, "POST", "/v1/chat/completions",
                {"messages": [q], "reasoning": "auto", "context": "The wifi password is tulip77.", "max_tokens": 24})[2]
    assert body["reasoning"]["strategy"] == "rag"
    st, _, out = call(server, "POST", "/v1/chat/completions", {"messages": [q], "reasoning": "rag"})
    assert st == 400 and "context" in out["error"]["message"]
    st, _, out = call(server, "POST", "/v1/chat/completions", {"messages": [q], "reasoning": "telepathy"})
    assert st == 400 and "unknown reasoning strategy" in out["error"]["message"]
    st, _, out = call(server, "POST", "/v1/chat/completions", {"messages": [q], "reasoning": 3})
    assert st == 400
    st, _, out = call(server, "POST", "/v1/chat/completions",
                      {"messages": [{"role": "system", "content": "x"}], "reasoning": "direct"})
    assert st == 400 and "user message" in out["error"]["message"]


# ---------------------------------------------------------------------------
# helpers without a model, and the official client when it is installed

def test_split_thinking():
    from superborg.server import _split_thinking
    assert _split_thinking("<think>\nstep 1\n</think>\n\nAnswer: 4") == ("step 1", "Answer: 4")
    assert _split_thinking("step 1</think>Answer") == ("step 1", "Answer")  # <think> was in the prompt
    assert _split_thinking("no thinking here") == (None, "no thinking here")
    assert _split_thinking("Sure. <think>x</think> y") == (None, "Sure. <think>x</think> y")


@pytest.mark.parametrize("text,opened", [
    ("<think>\nstep 1\nstep 2\n</think>\n\nAnswer: 4", False), ("<think>\n\n</think>\n\nHello", False),
    ("Plain answer <think> no", False), ("<think>cut off while thinking  \n", False), ("  <think>x</think>y", False),
    ("<thi", False), ("no tags", False), ("<think>a </think b</think> c", False),
    ("step 1\n</think>\n\nAnswer", True),  # the chat template wrote "<think>" into the prompt
])
def test_thinking_stream_matches_split(text, opened):
    """Streamed thinking replies carry the same reasoning/content as non-streamed ones, however
    the text is cut into pieces (tags split across tokens included)."""
    import random

    from superborg.server import _split_thinking, _ThinkingStream
    want_reasoning, want_content = _split_thinking(text)
    rng = random.Random(len(text))
    for _ in range(200):
        out: list[tuple[str, str]] = []
        st = _ThinkingStream(lambda f, t: out.append((f, t)), opened=opened)
        i = 0
        while i < len(text):
            j = i + rng.randint(1, 5)
            st.feed(text[i:j])
            i = j
        st.finish()
        reasoning = "".join(t for f, t in out if f == "reasoning_content")
        assert reasoning == (want_reasoning or "")
        assert "".join(t for f, t in out if f == "content") == want_content
        assert all(t for _, t in out), "no empty deltas"


def test_thinking_mode_stream_and_plain_agree():
    """Qwen3 thinking mode over HTTP: the think block is reasoning_content in both forms."""
    from superborg import LLM
    path = model_path("qwen3-0.6b-q8_0.borg")
    try:
        qwen3 = LLM(path, threads=THREADS, n_ctx=512)
    except OSError as e:
        pytest.skip(f"engine unavailable: {e}")
    srv, th = _start(qwen3)
    try:
        req = {"messages": [{"role": "user", "content": "What is 2 + 3?"}], "max_tokens": 160, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": True}}
        msg = warm_call(srv, "/v1/chat/completions", req)[2]["choices"][0]["message"]
        assert msg.get("reasoning_content"), msg
        assert "<think>" not in msg["content"] and "</think>" not in msg["content"]
        st, _, events = stream(srv, "/v1/chat/completions", req)
        assert st == 200 and events[-1] == "[DONE]"
        deltas = [e["choices"][0]["delta"] for e in events[:-1] if e.get("choices")]
        assert "".join(d.get("reasoning_content", "") for d in deltas) == msg["reasoning_content"]
        assert "".join(d.get("content", "") for d in deltas) == msg["content"]
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=10)
        qwen3.close()


def test_openai_client(server):
    openai = pytest.importorskip("openai")
    host, port = server.server_address[:2]
    client = openai.OpenAI(base_url=f"http://{host}:{port}/v1", api_key="unused")
    assert client.models.list().data[0].id == server.service.model_id
    r = client.chat.completions.create(model="x", messages=[{"role": "user", "content": "Hi"}], max_tokens=6,
                                       temperature=0)
    assert r.choices[0].message.role == "assistant" and r.usage.completion_tokens > 0
    parts = [c.choices[0].delta.content or "" for c in
             client.chat.completions.create(model="x", messages=[{"role": "user", "content": "Hi"}], max_tokens=6,
                                            temperature=0, stream=True) if c.choices]
    assert "".join(parts) == r.choices[0].message.content
    c = client.completions.create(model="x", prompt="Hello", max_tokens=4, temperature=0)
    assert c.choices[0].text and c.usage.prompt_tokens > 0
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="x", messages=[], max_tokens=4)
