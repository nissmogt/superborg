"""The web UI server (superborg/ui.py), exercised over real HTTP.

A UI server around smollm2-135m-q8_0.borg runs in a background thread on a free port.
Models are found like in test_server.py; tests skip when the model file is missing.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
THREADS = int(os.environ.get("BORG_TEST_THREADS", "2"))
MODEL = "smollm2-135m-q8_0.borg"
INDEX = os.path.join(ROOT, "superborg", "web", "index.html")


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
    path = model_path(MODEL)
    try:
        from superborg import LLM
        lm = LLM(path, threads=THREADS, n_ctx=1024)
    except (OSError, FileNotFoundError) as e:
        pytest.skip(f"engine unavailable: {e}")
    yield lm
    lm.close()


@pytest.fixture(scope="module")
def server(llm):
    from superborg.ui import make_ui_server
    srv = make_ui_server(llm, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.ui.cancel.set()
    srv.shutdown()
    srv.server_close()
    th.join(timeout=10)


def call(srv, method: str, path: str, body=None, headers: dict | None = None, raw: bytes | None = None):
    """Returns (status, headers, parsed JSON or raw bytes)."""
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=120)
    try:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        payload = r.read()
        try:
            parsed = json.loads(payload) if payload and "json" in (r.getheader("Content-Type") or "") else payload
        except json.JSONDecodeError:
            parsed = payload
        return r.status, {k.lower(): v for k, v in r.getheaders()}, parsed
    finally:
        conn.close()


def events_of(resp):
    """Parse a Server-Sent Events stream: one JSON event per `data:` line."""
    while True:
        line = resp.readline()
        if not line:
            return
        line = line.strip()
        if line:
            assert line.startswith(b"data: "), line
            yield json.loads(line[len(b"data: "):])


def chat(srv, body: dict):
    """POST /api/chat; returns (status, headers, events or the JSON error)."""
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=120)
    try:
        conn.request("POST", "/api/chat", body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        if r.status != 200:
            return r.status, hdrs, json.loads(r.read())
        return r.status, hdrs, list(events_of(r))
    finally:
        conn.close()


def wait_idle(srv, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not call(srv, "GET", "/api/info")[2]["busy"]:
            return True
        time.sleep(0.05)
    return False


Q = [{"role": "user", "content": "Name three colors."}]


# ---------------------------------------------------------------------------
# static content and info

def test_index_and_info(server, llm):
    st, hdr, page = call(server, "GET", "/")
    assert st == 200 and hdr["content-type"].startswith("text/html")
    html = page.decode("utf-8")
    assert '<canvas id="mapCanvas"' in html and "BORG_TRANSPORT" in html
    assert "default-src 'none'" in hdr["content-security-policy"]  # self-contained: nothing loads from elsewhere
    assert "access-control-allow-origin" not in hdr
    st, hdr, info = call(server, "GET", "/api/info")
    assert st == 200 and hdr["content-type"].startswith("application/json")
    cfg = llm.model.config
    assert info["n_layers"] == cfg["n_layers"] and info["n_heads"] == cfg["n_heads"]
    assert info["lens_layers"] == list(range(cfg["n_layers"] + 1)) and info["lens_stride"] == 1
    assert info["busy"] is False and info["quant"] == "q8_0" and info["n_ctx"] == llm.n_ctx
    assert info["max_tokens_default"] == min(512, llm.n_ctx)
    assert call(server, "GET", "/favicon.ico")[0] == 204
    st, _, body = call(server, "GET", "/nope")
    assert st == 404 and "error" in body
    assert call(server, "GET", "/api/chat")[0] == 405


def test_page_is_self_contained_and_parses(tmp_path):
    """One file, no external resources, all engine I/O behind the swappable transport."""
    html = open(INDEX, encoding="utf-8").read()
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html), "external resource"
    assert "@import" not in html
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    js = scripts[0]
    transport_end = js.index("(() => {")  # the transport comes first, the app after it
    assert js.index("window.BORG_TRANSPORT") < transport_end
    fetches = [m.start() for m in re.finditer(r"\bfetch\(", js)]
    assert fetches and all(p < transport_end for p in fetches), "fetch() outside the transport"
    node = shutil.which("node")
    if node:  # syntax check when node is available
        f = tmp_path / "page.js"
        f.write_text(js, encoding="utf-8")
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr


def test_package_data_lists_the_page():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        assert "web/*.html" in f.read()
    with open(os.path.join(ROOT, "MANIFEST.in"), encoding="utf-8") as f:
        assert "superborg/web" in f.read()
    from importlib import resources
    assert resources.files("superborg").joinpath("web", "index.html").is_file()


def test_cli_parser():
    from superborg.cli import build_parser, cmd_ui
    a = build_parser().parse_args(["ui", "m.borg", "--host", "::1", "--port", "0", "-t", "2", "-c", "512",
                                   "--lens-stride", "3", "--open"])
    assert a.func is cmd_ui and (a.host, a.port, a.threads, a.ctx, a.lens_stride, a.open) == ("::1", 0, 2, 512, 3, True)
    a = build_parser().parse_args(["ui", "m.borg"])
    assert (a.host, a.port, a.lens_stride, a.open, a.allow_host) == ("127.0.0.1", 8765, 1, False, [])
    a = build_parser().parse_args(["ui", "m.borg", "--allow-host", "mybox.lan", "--allow-host", "borg.local"])
    assert a.allow_host == ["mybox.lan", "borg.local"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ui", "m.borg", "--lens-stride", "0"])


# ---------------------------------------------------------------------------
# streaming

def test_chat_streams_prompt_tokens_done(server, llm):
    assert wait_idle(server)
    st, hdr, evs = chat(server, {"messages": Q, "max_tokens": 8, "temperature": 0})
    assert st == 200 and hdr["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in evs] == ["prompt"] + ["token"] * (len(evs) - 2) + ["done"]
    cfg = llm.model.config
    n_layers, n_heads, hidden = cfg["n_layers"], cfg["n_heads"], cfg["hidden_dim"]
    prompt, toks, done = evs[0], evs[1:-1], evs[-1]
    assert prompt["n_prompt"] == len(prompt["tokens"]) > 0 and all(isinstance(t, str) for t in prompt["tokens"])
    assert 0 <= prompt["n_reused"] < prompt["n_prompt"] and prompt["prompt_ms"] >= 0
    assert 1 <= len(toks) <= 8
    for i, t in enumerate(toks):
        assert t["i"] == i and t["pos"] == prompt["n_prompt"] - 1 + i
        assert isinstance(t["id"], int) and isinstance(t["raw"], str) and isinstance(t["text"], str)
        assert 0 <= t["prob"] <= 1 and t["entropy"] >= 0
        assert t["decode_ms"] > 0 and t["trace_ms"] >= 0  # the first token's too: the prompt's last pass
        # [text, p, token id]
        assert len(t["alternatives"]) == 5
        assert all(0 <= p <= 1 and isinstance(tid, int) for _, p, tid in t["alternatives"])
        # greedy: the chosen token is the most likely one
        assert t["alternatives"][0][0] == t["raw"] and t["alternatives"][0][2] == t["id"]
        assert [x["layer"] for x in t["lens"]] == list(range(n_layers + 1))
        for x in t["lens"]:
            assert len(x["top"]) == 3 and 0 <= x["p"] <= 1 and x["rank"] >= 0
            assert all(isinstance(s, str) and 0 <= p <= 1 and isinstance(tid, int) for s, p, tid in x["top"])
        out = t["lens"][-1]  # the lens at the last layer is the model's own output
        assert out["rank"] == 0 and abs(out["p"] - t["prob"]) < 2e-3
        assert len(t["norms"]["attn"]) == len(t["norms"]["mlp"]) == n_layers
        assert len(t["norms"]["resid"]) == n_layers + 1
        assert all(v >= 0 for k in ("attn", "mlp", "resid") for v in t["norms"][k])
        assert len(t["attn_top"]) == n_layers
        for row in t["attn_top"]:
            assert 1 <= len(row) <= 8
            assert all(0 <= p <= t["pos"] and 0 <= w <= 1 for p, w in row)
            assert [w for _, w in row] == sorted((w for _, w in row), reverse=True)
        assert len(t["attn_entropy"]) == n_layers and all(len(r) == n_heads for r in t["attn_entropy"])
        assert len(t["mlp_top"]) == n_layers
        assert all(len(r) == 6 and all(0 <= n < hidden for n, _ in r) for r in t["mlp_top"])
        # as streamed (no spaces): about 38 KB per token for this model at this context, see docs/TRACE.md
        assert len(json.dumps(t, separators=(",", ":"))) < 64_000
    assert done["n_gen"] == sum(not t["eos"] for t in toks)
    assert done["stop_reason"] in ("eos", "length", "stop")
    assert done["text"] == "".join(t["text"] for t in toks)[:len(done["text"])]
    assert done["tok_per_s"] >= 0 and done["seconds"] >= 0
    # same request again: identical tokens (greedy), and the prompt prefix comes from the cache
    st, _, again = chat(server, {"messages": Q, "max_tokens": 8, "temperature": 0})
    assert st == 200 and [e.get("id") for e in again[1:-1]] == [e["id"] for e in toks]
    assert again[0]["n_reused"] > 0


def test_lens_stride_per_request(server, llm):
    assert wait_idle(server)
    n = llm.model.config["n_layers"]
    st, _, evs = chat(server, {"messages": Q, "max_tokens": 2, "temperature": 0, "lens_stride": 4})
    assert st == 200
    want = list(range(0, n + 1, 4)) + ([n] if n % 4 else [])
    assert all([x["layer"] for x in e["lens"]] == want for e in evs if e["type"] == "token")
    assert call(server, "GET", "/api/info")[2]["lens_layers"] == list(range(n + 1))  # the default is unchanged


def test_sampling_seed_reproducible(server):
    body = {"messages": Q, "max_tokens": 6, "temperature": 1.0, "top_p": 0.9, "top_k": 20, "seed": 1234}
    a = [e["id"] for e in chat(server, body)[2] if e["type"] == "token"]
    b = [e["id"] for e in chat(server, body)[2] if e["type"] == "token"]
    assert a == b and a


# ---------------------------------------------------------------------------
# one generation at a time; stopping

def _start_long(server, max_tokens: int = 300, **extra):
    """Start a long generation in a thread; returns (thread, events list, started event)."""
    host, port = server.server_address[:2]
    got: list = []
    started = threading.Event()

    def run():
        conn = http.client.HTTPConnection(host, port, timeout=120)
        try:
            conn.request("POST", "/api/chat", body=json.dumps({
                "messages": [{"role": "user", "content": "Count from one to one hundred, in words."}],
                "max_tokens": max_tokens, "temperature": 0, **extra}), headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            got.append(r.status)
            for ev in events_of(r):
                got.append(ev)
                if ev["type"] == "token":
                    started.set()
        finally:
            started.set()
            conn.close()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th, got, started


def test_second_chat_gets_409_and_stop_cancels(server):
    assert wait_idle(server)
    th, got, started = _start_long(server)
    assert started.wait(60) and got[0] == 200
    st, hdr, body = chat(server, {"messages": Q, "max_tokens": 2})
    assert st == 409 and body["error"]["type"] == "busy"
    assert hdr["content-type"].startswith("application/json")
    st, _, body = call(server, "POST", "/api/stop", {})
    assert st == 200 and body == {"stopped": True}
    th.join(30)
    assert not th.is_alive()
    done = got[-1]
    assert done["type"] == "done" and done["stop_reason"] == "cancelled"
    assert done["n_gen"] < 300
    assert wait_idle(server, 10)
    # stopping when nothing runs is harmless and does not cancel the next generation
    assert call(server, "POST", "/api/stop", {})[2] == {"stopped": False}
    st, _, evs = chat(server, {"messages": Q, "max_tokens": 3, "temperature": 0})
    assert st == 200 and evs[-1]["type"] == "done" and evs[-1]["stop_reason"] != "cancelled"


def test_client_disconnect_stops_generation(server):
    assert wait_idle(server)
    host, port = server.server_address[:2]
    body = json.dumps({"messages": [{"role": "user", "content": "Count from one to one hundred, in words."}],
                       "max_tokens": 300, "temperature": 0}).encode()
    sock = socket.create_connection((host, port), timeout=60)
    sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                 b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    got = b""
    while b'"type":"token"' not in got:  # headers, the prompt event and a first token
        part = sock.recv(65536)
        assert part, "connection closed before the first token"
        got += part
    assert got.startswith(b"HTTP/1.1 200")
    t0 = time.time()
    sock.close()
    assert wait_idle(server, 15), "generation kept running after the client left"
    assert time.time() - t0 < 15
    st, _, evs = chat(server, {"messages": Q, "max_tokens": 2})
    assert st == 200 and evs[-1]["type"] == "done"


# ---------------------------------------------------------------------------
# bad requests

@pytest.mark.parametrize("body,needle", [
    ({}, "messages"),
    ({"messages": []}, "messages"),
    ({"messages": "hi"}, "messages"),
    ({"messages": [{"content": "hi"}]}, "role"),
    ({"messages": [{"role": "wizard", "content": "hi"}]}, "role"),
    ({"messages": [{"role": "user", "content": 5}]}, "content"),
    ({"messages": Q, "max_tokens": 0}, "max_tokens"),
    ({"messages": Q, "max_tokens": "8"}, "max_tokens"),
    ({"messages": Q, "max_tokens": True}, "max_tokens"),
    ({"messages": Q, "max_tokens": 1.5}, "max_tokens"),
    ({"messages": Q, "temperature": 3}, "temperature"),
    ({"messages": Q, "top_p": 1.5}, "top_p"),
    ({"messages": Q, "top_k": -1}, "top_k"),
    ({"messages": Q, "seed": -1}, "seed"),
    ({"messages": Q, "lens_stride": 0}, "lens_stride"),
    ({"messages": Q, "request_id": ""}, "request_id"),
    ({"messages": Q, "request_id": "x" * 65}, "request_id"),
    ({"messages": Q, "request_id": "a b"}, "request_id"),
    ({"messages": Q, "request_id": "caf\u00e9"}, "request_id"),
    ({"messages": Q, "request_id": 5}, "request_id"),
    ({"messages": Q, "enable_thinking": "yes"}, "enable_thinking"),
])
def test_bad_chat_bodies(server, body, needle):
    st, hdr, out = call(server, "POST", "/api/chat", body)
    assert st == 400, out
    assert hdr["content-type"].startswith("application/json") and needle in json.dumps(out)


def test_malformed_requests(server):
    assert wait_idle(server)
    for raw in (b"{", b"[1, 2]", b"null", b'"text"', b"\xff\xfe", b"[" * 100_000):
        st, _, out = call(server, "POST", "/api/chat", raw=raw)
        assert st == 400 and "error" in out, raw[:10]
    st, _, out = call(server, "POST", "/api/chat", raw=b"{}", headers={"Content-Type": "text/plain"})
    assert st == 415
    st, _, out = call(server, "POST", "/api/chat", {"messages": Q}, headers={"Origin": "http://evil.example"})
    assert st == 403 and "cross-origin" in out["error"]["message"]
    st, _, out = call(server, "GET", "/api/info", headers={"Host": "evil.example:80"})  # DNS rebinding
    assert st == 403
    host, port = server.server_address[:2]
    st, _, out = call(server, "POST", "/api/chat", {"messages": Q, "max_tokens": 1},
                      headers={"Origin": f"http://{host}:{port}"})  # the page's own origin is fine
    assert st == 200
    assert call(server, "POST", "/api/nope", {})[0] == 404
    assert call(server, "OPTIONS", "/api/chat")[0] == 405  # no CORS preflight
    st, _, out = call(server, "POST", "/api/chat", {"messages": [{"role": "user", "content": "word " * 3000}]})
    assert st == 400 and "context" in out["error"]["message"]
    assert wait_idle(server, 5)
    st, _, evs = chat(server, {"messages": Q, "max_tokens": 1})
    assert st == 200


def test_short_body_then_hangup(server, capsys):
    host, port = server.server_address[:2]
    for _ in range(3):
        sock = socket.create_connection((host, port), timeout=30)
        sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: 100\r\n\r\n{")
        sock.shutdown(socket.SHUT_WR)
        sock.close()
    assert wait_idle(server, 10)
    assert chat(server, {"messages": Q, "max_tokens": 1})[0] == 200
    time.sleep(0.2)
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# defaults that do not fit the model are clamped, never rejected

def test_defaults_are_clamped_to_the_model(server, llm):
    """--lens-stride above the layer count, a context under 512 tokens, or a max_tokens saved under
    a bigger context: the server's own defaults and such values are clamped, not a 400."""
    from superborg.ui import UIService
    n_layers = llm.model.config["n_layers"]
    ui = UIService(llm, lens_stride=n_layers + 34, max_tokens_default=llm.n_ctx + 1000)
    info = ui.info()
    assert info["lens_stride"] == n_layers and info["lens_layers"] == [0, n_layers]
    assert info["max_tokens_default"] == llm.n_ctx
    kw = ui.parse_chat({"messages": Q})  # what the page sends by default
    assert kw["max_tokens"] == llm.n_ctx and kw["lens_stride"] == n_layers
    kw = ui.parse_chat({"messages": Q, "max_tokens": 10 ** 9, "lens_stride": 10 ** 9})
    assert kw["max_tokens"] == llm.n_ctx and kw["lens_stride"] == n_layers
    assert wait_idle(server)
    st, _, evs = chat(server, {"messages": [{"role": "user", "content": "What is the capital of France? One word."}],
                               "max_tokens": 10 ** 6, "lens_stride": 10_000, "temperature": 0})
    assert st == 200 and evs[-1]["type"] == "done" and evs[-1]["stop_reason"] == "eos"
    assert all([x["layer"] for x in e["lens"]] == [0, n_layers] for e in evs if e["type"] == "token")


# ---------------------------------------------------------------------------
# stopping one request by its id

# a reply that runs for hundreds of tokens (the counting prompt's greedy reply ends after ten)
STORY = {"messages": [{"role": "user", "content": "Write a very long story about a dragon."}],
         "temperature": 1.0, "seed": 3}

def test_stop_by_request_id(server):
    """A stop with a request_id never touches another request's generation; one that arrives
    before its request started (still waiting on 409, or overtaken) cancels it when it arrives."""
    assert wait_idle(server)
    th, got, started = _start_long(server, request_id="gen-a", **STORY)
    assert started.wait(60) and got[0] == 200
    st, _, body = call(server, "POST", "/api/stop", {"request_id": "gen-b"})  # another tab's own request
    assert st == 200 and body == {"stopped": False, "pending": True}
    n = len(got)
    time.sleep(0.5)
    assert len(got) > n and got[-1]["type"] == "token", "gen-a must keep running"
    st, _, body = chat(server, {"messages": Q, "request_id": "gen-b"})
    assert st == 409 and body["error"]["type"] == "busy"
    assert call(server, "POST", "/api/stop", {"request_id": "gen-a"})[2] == {"stopped": True}
    th.join(30)
    assert got[-1]["type"] == "done" and got[-1]["stop_reason"] == "cancelled"
    assert wait_idle(server, 10)
    # gen-b, retried, ends at once
    st, _, evs = chat(server, {"messages": Q, "max_tokens": 5, "request_id": "gen-b"})
    assert st == 200 and [e["type"] for e in evs] == ["prompt", "done"]
    assert evs[-1]["stop_reason"] == "cancelled" and evs[-1]["n_gen"] == 0
    # that stop is used up; one for a generation that already ran is not remembered
    assert call(server, "POST", "/api/stop", {"request_id": "gen-a"})[2] == {"stopped": False}
    for rid in ("gen-a", "gen-b"):
        st, _, evs = chat(server, {"messages": Q, "max_tokens": 3, "request_id": rid})
        assert st == 200 and evs[-1]["n_gen"] == 3 and evs[-1]["stop_reason"] == "length"
    st, _, body = call(server, "POST", "/api/stop", {"request_id": "no spaces"})
    assert st == 400 and body["error"]["param"] == "request_id"


def test_chunked_stop_body_is_refused_not_ignored(server):
    """A stop whose body is chunked (no Content-Length) must not be read as {} and cancel another
    request's generation: it gets the 411 every chunked body gets."""
    import http.client
    assert wait_idle(server)
    th, got, started = _start_long(server, request_id="gen-c", **STORY)
    assert started.wait(60) and got[0] == 200
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("POST", "/api/stop", body=iter([b'{"request_id": "gen-d"}']), encode_chunked=True,
                 headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
    resp = conn.getresponse()
    assert resp.status == 411
    conn.close()
    n = len(got)
    time.sleep(0.5)
    assert len(got) > n and got[-1]["type"] == "token", "gen-c must keep running"
    assert call(server, "POST", "/api/stop", {"request_id": "gen-c"})[2] == {"stopped": True}
    th.join(30)
    assert got[-1]["stop_reason"] == "cancelled"


def test_remembered_stops_are_bounded(server, monkeypatch):
    from superborg import ui as ui_mod
    assert wait_idle(server)
    ui = server.ui

    def cancelled_at_start(rid: str) -> bool:
        assert ui.begin(rid)
        try:
            return ui.cancel.is_set()
        finally:
            ui.end()

    for i in range(ui_mod.STOP_MEMORY + 6):
        assert ui.stop(f"many-{i}") == {"stopped": False, "pending": True}
    assert not cancelled_at_start("many-0")  # the oldest are forgotten
    assert cancelled_at_start(f"many-{ui_mod.STOP_MEMORY + 5}")
    assert not cancelled_at_start("unrelated")  # and nothing leaks into other requests
    monkeypatch.setattr(ui_mod, "STOP_MEMORY_S", 0.0)
    ui.stop("expired")
    time.sleep(0.01)
    assert not cancelled_at_start("expired")


# ---------------------------------------------------------------------------
# long prompts: progress, stop and disconnect while the prompt is evaluated

def _long_messages(n_items: int = 34) -> list[dict]:  # about 530 tokens: five chunks
    text = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(n_items))
    return [{"role": "user", "content": text + "\nSummarize."}]


def test_stop_during_prefill(server, llm):
    from superborg.interp import PREFILL_CHUNK
    assert wait_idle(server)
    llm.ctx.reset()
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=120)
    try:
        conn.request("POST", "/api/chat", body=json.dumps({"messages": _long_messages(), "max_tokens": 50,
                                                           "request_id": "long-1"}),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        assert r.status == 200  # the stream opens after the first chunk, long before the prompt is done
        evs = []
        for ev in events_of(r):
            evs.append(ev)
            if ev["type"] == "prefill" and len(evs) == 1:
                assert ev == {"type": "prefill", "done": PREFILL_CHUNK, "total": ev["total"]}
                assert call(server, "POST", "/api/stop", {"request_id": "long-1"})[2] == {"stopped": True}
    finally:
        conn.close()
    types = [e["type"] for e in evs]
    assert types == ["prefill"] * (len(types) - 2) + ["prompt", "done"]
    assert evs[-1]["stop_reason"] == "cancelled" and evs[-1]["n_gen"] == 0
    assert evs[-3]["done"] < evs[-3]["total"] - PREFILL_CHUNK  # stopped well before the end of the prompt
    assert wait_idle(server, 10)


def test_disconnect_during_prefill(server, llm):
    from superborg.interp import PREFILL_CHUNK
    assert wait_idle(server)
    llm.ctx.reset()
    host, port = server.server_address[:2]
    msgs = _long_messages()
    n_prompt = len(llm.tokenize(llm.render(msgs), add_bos=False))
    body = json.dumps({"messages": msgs, "max_tokens": 50}).encode()
    sock = socket.create_connection((host, port), timeout=60)
    sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                 b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    got = b""
    while b'"type":"prefill"' not in got:
        part = sock.recv(65536)
        assert part, "connection closed before the first prefill event"
        got += part
    sock.close()
    assert wait_idle(server, 15)
    assert llm.ctx.n_past <= 3 * PREFILL_CHUNK < n_prompt, "the prompt was evaluated to the end"


# ---------------------------------------------------------------------------
# connections: half-close, high descriptor numbers, clients that stop reading

def _raw_chat(sock, body: dict) -> None:
    data = json.dumps(body).encode()
    sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                 b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)


COUNT = {"messages": [{"role": "user", "content": "Count from one to one hundred, in words."}], "max_tokens": 8,
         "temperature": 0}


def test_half_closed_client_gets_the_whole_reply(server):
    """A client may shut down its sending side after the request (nc -N does) and keep reading."""
    assert wait_idle(server)
    host, port = server.server_address[:2]
    sock = socket.create_connection((host, port), timeout=60)
    try:
        _raw_chat(sock, COUNT)
        sock.shutdown(socket.SHUT_WR)
        got = b""
        while True:
            part = sock.recv(65536)
            if not part:
                break
            got += part
    finally:
        sock.close()
    evs = [json.loads(line[6:]) for line in got.split(b"\n") if line.startswith(b"data: ")]
    assert evs[-1]["type"] == "done" and evs[-1]["stop_reason"] == "length" and evs[-1]["n_gen"] == 8


def test_connection_with_a_descriptor_above_1024(server, monkeypatch):
    """select.select cannot watch descriptors >= FD_SETSIZE (1024); such a connection must not
    look like a client that left."""
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 1200
    if soft < want:
        if hard != resource.RLIM_INFINITY and hard < want:
            pytest.skip(f"RLIMIT_NOFILE {hard} is too low")
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    seen = []
    base_setup = server.RequestHandlerClass.setup

    def setup(handler):
        base_setup(handler)
        seen.append(handler.connection.fileno())

    monkeypatch.setattr(server.RequestHandlerClass, "setup", setup)
    assert wait_idle(server)
    fds = []
    try:
        while not fds or fds[-1] < 1030:
            fds.append(os.open(os.devnull, os.O_RDONLY))
        seen.clear()
        st, _, evs = chat(server, COUNT)
    finally:
        for fd in fds:
            os.close(fd)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert seen and seen[0] >= 1024
    assert st == 200 and evs[-1]["stop_reason"] == "length" and evs[-1]["n_gen"] == 8


def test_client_that_stops_reading_does_not_keep_the_model(server, llm, monkeypatch):
    """A client that stays connected but stops reading blocks the server's write. A stop frees the
    model within STOP_GRACE_S, and without one the write gives up after STREAM_WRITE_TIMEOUT_S."""
    from superborg import ui as ui_mod
    base_setup = server.RequestHandlerClass.setup

    def setup(handler):  # small buffers: the first token event (about 10 KB) already blocks
        base_setup(handler)
        handler.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

    monkeypatch.setattr(server.RequestHandlerClass, "setup", setup)
    host, port = server.server_address[:2]
    for how in ("stop", "timeout"):
        if how == "timeout":
            monkeypatch.setattr(ui_mod, "STREAM_WRITE_TIMEOUT_S", 2)
        assert wait_idle(server)
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        sock.connect((host, port))
        try:
            _raw_chat(sock, dict(COUNT, max_tokens=300))
            deadline = time.time() + 60
            while not call(server, "GET", "/api/info")[2]["busy"]:  # on a loaded machine the request
                assert time.time() < deadline, "the chat never started"  # takes a moment to be picked up
                time.sleep(0.05)
            last, since = -1, time.time()
            while time.time() - since < 1.0:  # stalled: no forward pass for a second
                assert time.time() < deadline and call(server, "GET", "/api/info")[2]["busy"]
                if llm.ctx.n_past != last:
                    last, since = llm.ctx.n_past, time.time()
                time.sleep(0.05)
            t0 = time.time()
            if how == "stop":
                assert call(server, "POST", "/api/stop", {})[2] == {"stopped": True}
                assert wait_idle(server, ui_mod.STOP_GRACE_S + 5), "the stop said stopped but the model is held"
            else:
                assert wait_idle(server, 10)
            assert time.time() - t0 < ui_mod.STOP_GRACE_S + 3
        finally:
            sock.close()
    monkeypatch.undo()
    assert chat(server, COUNT)[0] == 200


# ---------------------------------------------------------------------------
# DNS-rebinding guard on any bind address

def test_host_guard_when_bound_to_all_interfaces(llm):
    """Bound to 0.0.0.0 (a phone, a container), the Host check stays on: local names, IP addresses
    and --allow-host names pass; any other name is a page that rebinds its DNS to this machine."""
    from superborg.ui import make_ui_server
    srv = make_ui_server(llm, "0.0.0.0", 0, allow_hosts=["MyBox.lan"])
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        conn_srv = type("S", (), {"server_address": ("127.0.0.1", port)})
        for host, ok in [(f"attacker.example:{port}", False), ("attacker.example", False), ("", False),
                         (f"127.0.0.1:{port}", True), (f"localhost:{port}", True), (f"[::1]:{port}", True),
                         (f"192.168.1.20:{port}", True), (f"[fe80::1]:{port}", True), (f"mybox.lan:{port}", True),
                         (f"MYBOX.LAN:{port}", True), (f"mybox.lan.attacker.example:{port}", False)]:
            st, _, out = call(conn_srv, "GET", "/api/info", headers={"Host": host})
            assert st == (200 if ok else 403), (host, out)
        # the rebinding page's POST, with its matching Origin, is refused too
        st, _, out = call(conn_srv, "POST", "/api/chat", {"messages": Q, "max_tokens": 1},
                          headers={"Host": f"attacker.example:{port}", "Origin": f"http://attacker.example:{port}"})
        assert st == 403 and out["error"]["type"] == "permission_error"
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=10)


# ---------------------------------------------------------------------------
# chat_template_kwargs: further variables for the model's chat template

def test_chat_template_kwargs(server, llm, monkeypatch):
    """The chat body's chat_template_kwargs reach the model's chat template in both renders (the
    400 check and stream_chat's), as superborg/server.py passes them; SmolLM2's template ignores
    them, so the prompt keeps its length; the enable_thinking field wins over the kwargs' entry;
    bad values are 400s with the server's messages."""
    assert wait_idle(server)
    seen, render = [], llm.formatter.render
    monkeypatch.setattr(llm.formatter, "render", lambda messages, **kw: (seen.append(kw), render(messages, **kw))[1])
    st, _, plain = chat(server, {"messages": Q, "max_tokens": 1, "temperature": 0})
    assert st == 200 and seen and all(kw["enable_thinking"] is False and "date_string" not in kw for kw in seen)
    seen.clear()
    st, _, extra = chat(server, {"messages": Q, "max_tokens": 1, "temperature": 0,
                                 "chat_template_kwargs": {"date_string": "16 Oct 2026", "enable_thinking": True}})
    assert st == 200 and extra[-1]["type"] == "done" and extra[0]["n_prompt"] == plain[0]["n_prompt"]
    assert len(seen) == 2 and all(kw["date_string"] == "16 Oct 2026" and kw["enable_thinking"] is True for kw in seen)
    seen.clear()
    st, _, _ = chat(server, {"messages": Q, "max_tokens": 1, "enable_thinking": False,
                             "chat_template_kwargs": {"enable_thinking": True}})
    assert st == 200 and seen and all(kw["enable_thinking"] is False for kw in seen)
    for body, needle in [({"chat_template_kwargs": [1]}, "must be an object"),
                         ({"chat_template_kwargs": {"enable_thinking": "yes"}}, "must be a boolean"),
                         ({"chat_template_kwargs": {"messages": []}}, "cannot set 'messages'"),
                         ({"chat_template_kwargs": {"add_generation_prompt": False}},
                          "cannot set 'add_generation_prompt'")]:
        st, _, out = call(server, "POST", "/api/chat", dict({"messages": Q}, **body))
        assert st == 400 and needle in out["error"]["message"], out
