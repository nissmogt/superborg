"""A local web UI: chat with a model and watch each reply token form inside it.

    python -m superborg ui model.borg            # then open http://127.0.0.1:8765/

The page (superborg/web/index.html, one self-contained file) shows a chat on the left and, on
the right, the trace superborg.interp.TraceSession records for every generated token: the
logit lens per layer, attention and MLP output sizes, attention patterns and the most active
MLP neurons.

Endpoints (standard library only):
  GET  /            the app
  GET  /api/info    model facts (TraceSession.info()) and whether a generation is running
  POST /api/chat    {"messages": [...], "max_tokens", "temperature", "top_p", "top_k", "seed",
                    "lens_stride", "enable_thinking"} -> Server-Sent Events, one JSON event per
                    `data:` line: "prompt", then one "token" per generated token, then "done"
                    (or an in-band "error"); see superborg.interp for the fields
  POST /api/stop    cancel the running generation (it ends with a "done" event whose
                    stop_reason is "cancelled")

One model and one KV cache serve every request, so one generation runs at a time and a second
POST /api/chat gets 409 meanwhile. A client that disconnects cancels its generation. The server
binds to localhost and only answers requests addressed to it by a local name and from its own
page (Host and Origin checks), so other web sites cannot drive the model through the browser.
"""
from __future__ import annotations

import json
import select
import socket
import sys
import threading
import time
import traceback
from http.server import ThreadingHTTPServer
from importlib import resources

from .server import APIError, Handler as _BaseHandler, _flag, _number, parse_messages

__all__ = ["UIService", "make_ui_server", "serve_ui"]

DEFAULT_PORT = 8765
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
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


class UIService:
    """The model, its trace session and the one-generation-at-a-time policy (HTTP-independent)."""

    def __init__(self, llm, lens_stride: int = 1, max_tokens_default: int = 512):
        from .interp import TraceOptions, TraceSession

        self.llm = llm
        self.default_stride = max(1, int(lens_stride))
        self.session = TraceSession(llm, TraceOptions(lens_stride=self.default_stride))
        self.max_tokens_default = max_tokens_default
        self.lock = threading.Lock()        # held for the whole generation
        self.cancel = threading.Event()     # set by /api/stop
        self.n_requests = 0

    def info(self) -> dict:
        n_layers = self.session.cfg["n_layers"]
        layers = list(range(0, n_layers + 1, self.default_stride))
        if layers[-1] != n_layers:
            layers.append(n_layers)
        return dict(self.session.info(), lens_layers=layers, lens_stride=self.default_stride,
                    busy=self.lock.locked(), max_tokens_default=self.max_tokens_default)

    def parse_chat(self, body: dict) -> dict:
        """Validated stream_chat arguments for a /api/chat body (raises APIError 400)."""
        n_layers = self.session.cfg["n_layers"]
        seed = _number(body, "seed", 0, int, 0, (1 << 64) - 1)
        return {
            "messages": parse_messages(body),
            "max_tokens": _number(body, "max_tokens", self.max_tokens_default, int, 1, self.llm.n_ctx),
            "temperature": _number(body, "temperature", 0.0, float, 0.0, 2.0),
            "top_p": _number(body, "top_p", 1.0, float, 0.0, 1.0),
            "top_k": _number(body, "top_k", 0, int, 0, 1 << 30),
            "seed": seed,
            "enable_thinking": _flag(body, "enable_thinking"),
            "lens_stride": _number(body, "lens_stride", self.default_stride, int, 1, max(1, n_layers)),
        }

    def stop(self) -> bool:
        """Ask the running generation to stop; False when nothing is running."""
        busy = self.lock.locked()
        if busy:
            self.cancel.set()
        return busy


class UIHandler(_BaseHandler):
    """HTTP layer. Reuses the OpenAI server's body parsing and JSON errors, but sends no CORS
    headers: only the app's own page may call the API."""

    ui: UIService  # set on the subclass made by make_ui_server
    loopback_only: bool = True

    def _cors(self):  # same-origin only
        pass

    def end_headers(self):
        for k, v in _SECURITY_HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    # -- request checks ---------------------------------------------------------
    def _host_ok(self) -> bool:
        """DNS-rebinding guard: a page from another site that resolves its name to 127.0.0.1
        reaches this server with its own name in Host."""
        if not self.loopback_only:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            name = host[1:host.find("]")] if "]" in host else host
        else:
            name = host.rsplit(":", 1)[0] if ":" in host else host
        return name in _LOOPBACK

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # not a cross-site browser request (curl, tests, same-origin GETs)
        return origin == f"http://{self.headers.get('Host', '')}"

    def _check(self, post: bool) -> None:
        if not self._host_ok():
            raise APIError(403, "this server only answers requests addressed to localhost",
                           type_="permission_error")
        if post:
            if not self._origin_ok():
                raise APIError(403, "cross-origin requests are not allowed", type_="permission_error")
            ctype = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                raise APIError(415, "send the request body as application/json")

    def _client_gone(self) -> bool:
        """True once the client has closed its end (it never sends anything after its request)."""
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            if not r:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

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
            body = self._read_json() if path == "/api/chat" or self.headers.get("Content-Length", "0") != "0" else {}
            if path == "/api/stop":
                return self._send_json(200, {"stopped": self.ui.stop()})
            kw = self.ui.parse_chat(body)
            if not self.ui.lock.acquire(blocking=False):
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
            self.ui.lock.release()

    def _chat(self, kw: dict) -> None:
        """Stream one generation as SSE (the lock is held). Errors before the first event are
        JSON responses; later ones are in-band {"type": "error"} events."""
        ui = self.ui
        ui.cancel.clear()
        ui.n_requests += 1
        ui.session.opt.lens_stride = kw.pop("lens_stride")
        self.close_connection = True  # the stream has no length; closing the connection ends it
        try:  # rendering is cheap; doing it here first turns a template refusal into a plain 400
            ui.llm.render(kw["messages"], enable_thinking=kw["enable_thinking"])
        except Exception as e:  # noqa: BLE001 - templates reject some message orders (raise_exception)
            return self._send_error(APIError(400, f"the model's chat template rejected the messages: {e}",
                                             param="messages"))
        gone = [False]

        def should_stop() -> bool:
            if self._client_gone():
                gone[0] = True
            return gone[0] or ui.cancel.is_set()

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
        self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                         + b"\n\n")
        self.wfile.flush()


class UIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    ui: UIService


class UIServer6(UIServer):
    address_family = socket.AF_INET6


def make_ui_server(llm, host: str = "127.0.0.1", port: int = DEFAULT_PORT, lens_stride: int = 1,
                   quiet: bool = True) -> UIServer:
    """Create (but do not start) the UI server around a loaded superborg.LLM. port=0 picks a free port."""
    ui = UIService(llm, lens_stride=lens_stride)
    handler = type("BoundUIHandler", (UIHandler,), {"ui": ui, "quiet": quiet,
                                                     "loopback_only": host in _LOOPBACK})
    srv = (UIServer6 if ":" in host else UIServer)((host, port), handler)
    srv.ui = ui
    return srv


def serve_ui(model_path: str, host: str = "127.0.0.1", port: int = DEFAULT_PORT, threads: int = 0,
             n_ctx: int = 4096, lens_stride: int = 1, open_browser: bool = False, quiet: bool = True) -> None:
    """Load a model and serve the UI until interrupted."""
    from . import LLM

    llm = LLM(model_path, threads=threads, n_ctx=n_ctx)
    try:
        srv = make_ui_server(llm, host, port, lens_stride=lens_stride, quiet=quiet)
    except BaseException:
        llm.close()
        raise
    h, p = srv.server_address[:2]
    shown = "127.0.0.1" if h in ("0.0.0.0", "::") else h
    url = f"http://[{shown}]:{p}/" if ":" in shown else f"http://{shown}:{p}/"
    print(f"superborg: {llm.name} (n_ctx {llm.n_ctx}, lens stride {srv.ui.default_stride}) at {url}",
          file=sys.stderr, flush=True)
    if host not in _LOOPBACK:
        print("superborg: WARNING: serving on a non-local address; anyone who can reach this port can use the model",
              file=sys.stderr, flush=True)
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
