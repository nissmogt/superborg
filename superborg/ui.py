"""A local web UI: chat with a model and watch each reply token form inside it.

    python -m superborg ui model.borg            # then open http://127.0.0.1:8765/

The page (superborg/web/index.html, one self-contained file) shows a chat on the left and, on
the right, the trace superborg.interp.TraceSession records for every generated token: the
logit lens per layer, attention and MLP output sizes, attention patterns and the most active
MLP neurons.

Endpoints (standard library only):
  GET  /            the app
  GET  /api/info    model facts (TraceSession.info()), the defaults (lens_stride, lens_layers,
                    max_tokens_default) and whether a generation is running (busy)
  GET  /api/basis   the residual-stream basis next to the model (<model stem>.basis.json, from
                    tools/trace_basis.py), which the "xyz" trace field is projected onto; 404 without one
  GET  /api/vocab_map.bin, /api/vocab_map.json
                    the vocabulary map next to the model (<model stem>.vocab_map.bin and .json, from
                    tools/vocab_map.py), the field of the page's flow view (docs/VIEWER.md); 404 without one
  POST /api/chat    {"messages": [...], "max_tokens", "temperature", "top_p", "top_k", "seed",
                    "lens_stride", "enable_thinking", "chat_template_kwargs" (further variables for
                    the chat template, as superborg/server.py takes them), "trace_prompt",
                    "request_id"} -> Server-Sent Events, one JSON event per `data:` line: "prefill"
                    progress for a long prompt, "prompt_token" traces of the prompt positions with
                    trace_prompt (the last 256 uncached ones, each evaluated alone: about as slow as
                    a generated token), "prompt", then one "token" per generated token, then "done"
                    (or an in-band "error"); see docs/TRACE.md for the fields. max_tokens above
                    n_ctx and lens_stride above n_layers are clamped.
  POST /api/stop    cancel the running generation (it ends with a "done" event whose stop_reason
                    is "cancelled"). With {"request_id": id} only the chat sent with that id is
                    cancelled, and if it has not started yet it is cancelled as soon as it arrives
                    (within STOP_MEMORY_S): {"stopped": false, "pending": true}

One model and one KV cache serve every request, so one generation runs at a time and a second
POST /api/chat gets 409 meanwhile. A client that disconnects cancels its generation (at the next
event written to it). The server only answers requests addressed to it by a local name, an IP
address or a name given with --allow-host, and from its own page (Host and Origin checks), so
other web sites cannot drive the model through the browser (DNS rebinding needs a host name).
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import select
import socket
import sys
import threading
import time
import traceback
from collections import deque
from http.server import ThreadingHTTPServer
from importlib import resources

from .server import APIError, Handler as _BaseHandler, _flag, _number, _template_kwargs, parse_messages

__all__ = ["UIService", "make_ui_server", "serve_ui"]

DEFAULT_PORT = 8765
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
STOP_MEMORY = 64          # stops remembered for chats that have not started yet ...
STOP_MEMORY_S = 60.0      # ... and for how long
STREAM_WRITE_TIMEOUT_S = 30  # a client that stops reading the stream frees the model after this long
STOP_GRACE_S = 2.0        # after a stop, a write blocked on such a client gives up after this long
_SECURITY_HEADERS = {
    # the page is one file with inline script and style; it talks only to this server
    "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
                                "frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


def _page() -> bytes:
    """The app's HTML (read on every request, so edits show up on reload)."""
    return resources.files("superborg").joinpath("web", "index.html").read_bytes()


def _request_id(body: dict) -> str | None:
    """The optional "request_id" of a /api/chat or /api/stop body (raises APIError 400)."""
    rid = body.get("request_id")
    if rid is not None and not (isinstance(rid, str) and _REQUEST_ID.fullmatch(rid)):
        raise APIError(400, "'request_id' must be 1-64 characters from A-Z, a-z, 0-9, '_' and '-'",
                       param="request_id")
    return rid


class UIService:
    """The model, its trace session and the one-generation-at-a-time policy (HTTP-independent)."""

    def __init__(self, llm, lens_stride: int = 1, max_tokens_default: int = 512, basis_path: str | None = None):
        from .interp import TraceOptions, TraceSession

        self.llm = llm
        # a stride past the last layer traces the same layers as one equal to it
        self.default_stride = max(1, min(int(lens_stride), llm.model.config["n_layers"]))
        # the residual-stream basis (tools/trace_basis.py): <model stem>.basis.json next to the model
        self.basis_path = basis_path if basis_path is not None else os.path.splitext(llm.model.path)[0] + ".basis.json"
        self.basis = None
        if os.path.isfile(self.basis_path):
            with open(self.basis_path, "rb") as f:
                self.basis = f.read()
        self.session = TraceSession(llm, TraceOptions(lens_stride=self.default_stride),
                                    basis=json.loads(self.basis) if self.basis else None)
        # the vocabulary map (tools/vocab_map.py): <model stem>.vocab_map.bin and .json next to the model, both or neither
        self.vocab_map_stem = os.path.splitext(llm.model.path)[0] + ".vocab_map"
        self.vocab_map = {}
        for ext in ("bin", "json"):
            if os.path.isfile(f"{self.vocab_map_stem}.{ext}"):
                with open(f"{self.vocab_map_stem}.{ext}", "rb") as f:
                    self.vocab_map[ext] = f.read()
        if len(self.vocab_map) != 2:
            self.vocab_map = {}
        self.max_tokens_default = min(max_tokens_default, llm.n_ctx)
        self.lock = threading.Lock()        # held for the whole generation
        self.cancel = threading.Event()     # set by /api/stop, cleared when a generation starts
        self.n_requests = 0
        self._state = threading.Lock()      # orders begin/end/stop; guards the three fields below
        self._running = None                # (request_id, on_stop) of the generation holding `lock`
        self._started = deque(maxlen=STOP_MEMORY)  # request ids of recent generations
        self._early_stops: dict[str, float] = {}   # request id -> when it was stopped before it started

    def info(self) -> dict:
        n_layers = self.session.cfg["n_layers"]
        layers = list(range(0, n_layers + 1, self.default_stride))
        if layers[-1] != n_layers:
            layers.append(n_layers)
        return dict(self.session.info(), lens_layers=layers, lens_stride=self.default_stride,
                    busy=self.lock.locked(), max_tokens_default=self.max_tokens_default, vocab_map=bool(self.vocab_map))

    def parse_chat(self, body: dict) -> dict:
        """Validated stream_chat arguments for a /api/chat body (raises APIError 400), plus its
        lens_stride and request_id. max_tokens and lens_stride are clamped from above: the reply
        is limited by the room left in the context anyway, and the last layer is always traced."""
        n_layers = self.session.cfg["n_layers"]
        seed = _number(body, "seed", 0, int, 0, (1 << 64) - 1)
        thinking, template_kwargs = _template_kwargs(body)  # the enable_thinking field wins, as in the server
        return {
            "messages": parse_messages(body),
            "max_tokens": min(_number(body, "max_tokens", self.max_tokens_default, int, 1), self.llm.n_ctx),
            "temperature": _number(body, "temperature", 0.0, float, 0.0, 2.0),
            "top_p": _number(body, "top_p", 1.0, float, 0.0, 1.0),
            "top_k": _number(body, "top_k", 0, int, 0, 1 << 30),
            "seed": seed,
            "enable_thinking": thinking,
            "chat_template_kwargs": template_kwargs or None,
            "trace_prompt": _flag(body, "trace_prompt"),
            "lens_stride": min(_number(body, "lens_stride", self.default_stride, int, 1), n_layers),
            "request_id": _request_id(body),
        }

    def begin(self, request_id: str | None = None, on_stop=None) -> bool:
        """Take the model for one generation (False when another one holds it); end() gives it back.
        on_stop() is called when a stop reaches this generation. If request_id was stopped before
        it got here, the generation starts cancelled."""
        with self._state:
            if not self.lock.acquire(blocking=False):
                return False
            self.n_requests += 1
            self.cancel.clear()  # a stop meant for an earlier generation must not reach this one
            self._running = (request_id, on_stop)
            if request_id is not None:
                self._started.append(request_id)
                t = self._early_stops.pop(request_id, None)
                if t is not None and time.monotonic() - t <= STOP_MEMORY_S:
                    self.cancel.set()
            return True

    def end(self) -> None:
        with self._state:
            self._running = None
            self.lock.release()

    def stop(self, request_id: str | None = None) -> dict:
        """Ask the running generation to stop: any one, or only the one with request_id.
        {"stopped": True} when it was signalled. A stop for a request that has not started yet is
        remembered for it: {"stopped": False, "pending": True}."""
        with self._state:
            now = time.monotonic()
            for k in [k for k, t in self._early_stops.items() if now - t > STOP_MEMORY_S]:
                del self._early_stops[k]
            running = self._running
            if running is None or (request_id is not None and request_id != running[0]):
                if request_id is None or request_id in self._started:
                    return {"stopped": False}
                self._early_stops.pop(request_id, None)
                self._early_stops[request_id] = now
                while len(self._early_stops) > STOP_MEMORY:
                    del self._early_stops[next(iter(self._early_stops))]
                return {"stopped": False, "pending": True}
            self.cancel.set()
        if running[1] is not None:
            running[1]()
        return {"stopped": True}


class UIHandler(_BaseHandler):
    """HTTP layer. Reuses the OpenAI server's body parsing and JSON errors, but sends no CORS
    headers: only the app's own page may call the API."""

    ui: UIService  # set on the subclass made by make_ui_server
    allowed_hosts: frozenset = frozenset()  # host names accepted besides local names and IP addresses
    _writing: float | None = None           # when the event write in progress began (time.monotonic())

    def _cors(self):  # same-origin only
        pass

    def end_headers(self):
        for k, v in _SECURITY_HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    # -- request checks ---------------------------------------------------------
    def _host_ok(self) -> bool:
        """DNS-rebinding guard: a page from another site that resolves its name to this machine
        reaches this server with its own name in Host. An IP address cannot be rebound, so
        those pass (a phone opens http://<LAN address>:port/), as do local and allowed names."""
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            name = host[1:host.find("]")] if "]" in host else host
        else:
            name = host.rsplit(":", 1)[0] if ":" in host else host
        if name in _LOOPBACK or name in self.allowed_hosts:
            return True
        try:
            ipaddress.ip_address(name)
            return True
        except ValueError:
            return False

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # not a cross-site browser request (curl, tests, same-origin GETs)
        return origin == f"http://{self.headers.get('Host', '')}"

    def _check(self, post: bool) -> None:
        if not self._host_ok():
            raise APIError(403, "this server only answers requests addressed to localhost, an IP address "
                                "or a name given with --allow-host", type_="permission_error")
        if post:
            if not self._origin_ok():
                raise APIError(403, "cross-origin requests are not allowed", type_="permission_error")
            ctype = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                raise APIError(415, "send the request body as application/json")

    def _hangup_check(self):
        """A callable that is True once the connection has failed (POLLERR/POLLHUP): the client reset
        it, or closed it and refused an event written since. End of input alone does not count: a
        client may half-close after its request and still read the stream (nc -N does). None
        without select.poll (Windows) and on macOS, where a failed write is the only sign."""
        if not hasattr(select, "poll") or sys.platform == "darwin":
            # macOS emulates poll with kqueue and may report a half-close (FIN) as POLLHUP
            return None
        p = select.poll()  # unlike select.select, no limit on the descriptor number
        p.register(self.connection, select.POLLERR | select.POLLHUP)
        return lambda: bool(p.poll(0))

    def _on_stop(self) -> None:
        """A stop must free the model even if this client has stopped reading: writes from now on
        give up after STOP_GRACE_S, and a write already blocked then is cut off by a shutdown."""
        t = time.monotonic()
        try:
            self.connection.settimeout(STOP_GRACE_S)
        except OSError:
            return

        def cut():
            w = self._writing
            if w is not None and w <= t:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        timer = threading.Timer(STOP_GRACE_S, cut)
        timer.daemon = True
        timer.start()

    # -- methods ---------------------------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802 - no CORS: a preflight is refused
        self._send_error(APIError(405, "cross-origin requests are not allowed"))

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            self._check(post=False)
            if path in ("/", "/index.html"):
                data = _page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/api/info":
                return self._send_json(200, self.ui.info())
            if path == "/api/basis":
                if self.ui.basis is None:
                    raise APIError(404, f"no residual basis for this model ({self.ui.basis_path}; tools/trace_basis.py)")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(self.ui.basis)))
                self.end_headers()
                self.wfile.write(self.ui.basis)
                return
            if path in ("/api/vocab_map.bin", "/api/vocab_map.json"):
                if not self.ui.vocab_map:
                    raise APIError(404, f"no vocabulary map for this model ({self.ui.vocab_map_stem}.bin/.json; tools/vocab_map.py)")
                ext = path.rsplit(".", 1)[1]
                data = self.ui.vocab_map[ext]
                self.send_response(200)
                self.send_header("Content-Type", "application/json" if ext == "json" else "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path in ("/api/chat", "/api/stop"):
                raise APIError(405, f"use POST for {path}")
            raise APIError(404, f"no route for GET {path}")
        except APIError as e:
            self._send_error(e)
        except OSError:
            self.close_connection = True

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path not in ("/api/chat", "/api/stop"):
                raise APIError(404, f"no route for POST {path}")
            self._check(post=True)
            has_body = self.headers.get("Content-Length", "0") != "0" or "Transfer-Encoding" in self.headers
            body = self._read_json() if path == "/api/chat" or has_body else {}  # chunked: 411, never {}
            if path == "/api/stop":
                return self._send_json(200, self.ui.stop(_request_id(body)))
            kw = self.ui.parse_chat(body)
            rid = kw.pop("request_id")
            # set before begin(): a stop may lower it from then on
            self.connection.settimeout(STREAM_WRITE_TIMEOUT_S)
            if not self.ui.begin(rid, on_stop=self._on_stop):
                raise APIError(409, "a generation is already running; stop it first (POST /api/stop)",
                               type_="busy")
        except APIError as e:
            return self._send_error(e)
        except OSError:
            self.close_connection = True
            return
        try:
            self._chat(kw)
        finally:
            self.ui.end()

    def _chat(self, kw: dict) -> None:
        """Stream one generation as SSE (the lock is held). Errors before the first event are
        JSON responses; later ones are in-band {"type": "error"} events."""
        ui = self.ui
        ui.session.opt.lens_stride = kw.pop("lens_stride")
        self.close_connection = True  # the stream has no length; closing the connection ends it
        try:  # rendering is cheap; doing it here first turns a template refusal into a plain 400
            if kw["chat_template_kwargs"]:
                ui.llm.formatter.render(kw["messages"], enable_thinking=kw["enable_thinking"],
                                        **kw["chat_template_kwargs"])
            else:
                ui.llm.render(kw["messages"], enable_thinking=kw["enable_thinking"])
        except Exception as e:  # noqa: BLE001 - templates reject some message orders (raise_exception)
            return self._send_error(APIError(400, f"the model's chat template rejected the messages: {e}",
                                             param="messages"))
        hung_up = self._hangup_check()

        def should_stop() -> bool:
            return ui.cancel.is_set() or (hung_up is not None and hung_up())

        events = ui.session.stream_chat(**kw, should_stop=should_stop)
        try:
            try:
                first = next(events)
            except StopIteration:
                first = None
            except ValueError as e:  # the conversation does not fit the context, or renders empty
                return self._send_error(APIError(400, str(e), param="messages", code="context_length_exceeded"
                                                 if "context" in str(e) else None))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                return self._send_error(APIError(500, f"{type(e).__name__}: {e}", type_="server_error"))
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                if first is not None:
                    self._event(first)
                for ev in events:
                    self._event(ev)
            except OSError:  # BrokenPipe, ConnectionReset or a write timeout: the client is gone
                return
            except Exception as e:  # noqa: BLE001 - report in-band, the headers are out
                traceback.print_exc(file=sys.stderr)
                try:
                    self._event({"type": "error", "message": f"{type(e).__name__}: {e}"})
                except OSError:
                    pass
        finally:
            events.close()  # stops the generator if it is still suspended (client gone)

    def _event(self, ev: dict) -> None:
        data = b"data: " + json.dumps(ev, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"
        self._writing = time.monotonic()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        finally:
            self._writing = None


class UIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    ui: UIService


class UIServer6(UIServer):
    address_family = socket.AF_INET6


def make_ui_server(llm, host: str = "127.0.0.1", port: int = DEFAULT_PORT, lens_stride: int = 1,
                   quiet: bool = True, allow_hosts=()) -> UIServer:
    """Create (but do not start) the UI server around a loaded superborg.LLM. port=0 picks a free port.
    allow_hosts: host names to answer besides local names and IP addresses (the bound one is added)."""
    ui = UIService(llm, lens_stride=lens_stride)
    allowed = frozenset(h.strip().lower().strip("[]") for h in (*allow_hosts, host) if h.strip())
    handler = type("BoundUIHandler", (UIHandler,), {"ui": ui, "quiet": quiet, "allowed_hosts": allowed})
    srv = (UIServer6 if ":" in host else UIServer)((host, port), handler)
    srv.ui = ui
    return srv


def serve_ui(model_path: str, host: str = "127.0.0.1", port: int = DEFAULT_PORT, threads: int = 0,
             n_ctx: int = 4096, lens_stride: int = 1, open_browser: bool = False, quiet: bool = True,
             allow_hosts=()) -> None:
    """Load a model and serve the UI until interrupted."""
    from . import LLM

    llm = LLM(model_path, threads=threads, n_ctx=n_ctx)
    try:
        srv = make_ui_server(llm, host, port, lens_stride=lens_stride, quiet=quiet, allow_hosts=allow_hosts)
    except BaseException:
        llm.close()
        raise
    h, p = srv.server_address[:2]
    shown = "127.0.0.1" if h in ("0.0.0.0", "::") else h
    url = f"http://[{shown}]:{p}/" if ":" in shown else f"http://{shown}:{p}/"
    print(f"superborg: {llm.name} (n_ctx {llm.n_ctx}, lens stride {srv.ui.default_stride}) at {url}",
          file=sys.stderr, flush=True)
    if host not in _LOOPBACK:
        print("superborg: WARNING: serving on a non-local address; anyone who can reach this port can use the model "
              "(open it by IP address, or name the host with --allow-host)", file=sys.stderr, flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.ui.cancel.set()  # a running generation stops at its next token
        srv.server_close()
        deadline = time.time() + 30
        # daemon request threads may still be generating; free the model only once they are done
        if srv.ui.lock.acquire(timeout=max(0.0, deadline - time.time())):
            llm.close()
