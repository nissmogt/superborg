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
    assert (a.host, a.port, a.lens_stride, a.open) == ("127.0.0.1", 8765, 1, False)
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
        assert len(t["alternatives"]) == 5 and all(0 <= p <= 1 for _, p in t["alternatives"])
        # greedy: the chosen token is the most likely one
        assert t["alternatives"][0][0] == t["raw"]
        assert [x["layer"] for x in t["lens"]] == list(range(n_layers + 1))
        for x in t["lens"]:
            assert len(x["top"]) == 3 and 0 <= x["p"] <= 1 and x["rank"] >= 0
            assert all(isinstance(s, str) and 0 <= p <= 1 for s, p in x["top"])
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
        assert len(json.dumps(t)) < 16_000  # compact: about 10-12 KB per token for this model
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

def _start_long(server, max_tokens: int = 300):
    """Start a long generation in a thread; returns (thread, events list, started event)."""
    host, port = server.server_address[:2]
    got: list = []
    started = threading.Event()

    def run():
        conn = http.client.HTTPConnection(host, port, timeout=120)
        try:
            conn.request("POST", "/api/chat", body=json.dumps({
                "messages": [{"role": "user", "content": "Count from one to one hundred, in words."}],
                "max_tokens": max_tokens, "temperature": 0}), headers={"Content-Type": "application/json"})
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
    ({"messages": Q, "lens_stride": 10_000}, "lens_stride"),
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
