"""The recorder's local web app: a page served to this computer's browser, and a JSON API.

    python -m taskrecorder                       # records into ~/Task Recordings, opens the browser
    python -m taskrecorder --dir PATH --port N    # another folder or port
    python -m taskrecorder --export [FILE]        # write every finished procedure as one Markdown file

Endpoints (standard library only; all state lives in taskrecorder.store.Store):
  GET  /                         the app (taskrecorder/web/index.html)
  GET  /api/state                {"active": task|null, "tasks": [...], "root", "focus", "version"}
  POST /api/start                {"name", "goal", "owner", "tags": [...], "track_focus"} -> task
  POST /api/event                {"kind": step|note|decision|pause|resume, "text"} -> event
  POST /api/redact               {"id"} -> event
  POST /api/capture              {"text"} -> event (takes a screenshot now)
  POST /api/file                 {"name", "data": base64, "text"} -> event
  POST /api/end                  {"outcome", "pain": 1-5, "wish"} -> task
  POST /api/discard              delete the recording in progress
  GET  /api/task/<slug>          task with its events
  GET  /api/procedure/<slug>     procedure.md (text/markdown)
  GET  /api/encyclopedia         every finished procedure in one document (text/markdown)
  GET  /files/<slug>/<path>      a capture or attached file of a recording

The server binds to 127.0.0.1 only. The page is opened through a launch link that carries a
token printed in the terminal (http://127.0.0.1:8766/?t=...); that visit sets a session cookie,
and every other request must carry the cookie. Requests whose Host is not a loopback name, whose
Sec-Fetch-Site says they come from another site, or whose Origin is not exactly this server are
refused, so other web pages, and other users of the same computer, cannot drive it.

While a task records, a background thread asks the operating system every FOCUS_PERIOD_S which
window is in front and writes a focus event when that changes (taskrecorder.probe). Nothing is
sampled when no task is recording or the task was started with track_focus off.
"""
from __future__ import annotations

import argparse
import base64
import secrets
import json
import mimetypes
import os
import sys
import threading
import time
import traceback
import webbrowser
from urllib.parse import parse_qs, quote, unquote, urlsplit
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from http import cookies as _cookies

from . import __version__, probe
from .store import KINDS, Store, StoreError

DEFAULT_PORT = 8766
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "Task Recordings")
FOCUS_PERIOD_S = 2.0
BODY_LIMIT = 80 * 1024 * 1024
_LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
                                "frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cache-Control": "no-store",
}
INLINE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")  # shown in the page; all else downloads
COOKIE = "taskrecorder"
PAGE_MARK = "Task recorder"  # focus events for our own page are not worth recording
LOCKED_PAGE = ("<!doctype html><meta charset=utf-8><title>Task recorder</title>"
               "<body style='font:16px system-ui;padding:40px;max-width:40em'><h1>Task recorder</h1>"
               "<p>This page opens only through the link printed in the terminal when the recorder started "
               "(it ends in <code>?t=…</code>). Start the recorder again if you no longer have it.</p>")


class Recorder:
    """The store plus the desktop probes and the focus-sampling thread."""

    def __init__(self, store: Store, focus: bool = True,
                 window: Optional[Callable] = probe.foreground_window,
                 shoot: Callable[[str], bool] = probe.screenshot):
        self.store = store
        self.focus_enabled = focus and window is not None
        self.window = window
        self.shoot = shoot
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.secret = secrets.token_urlsafe(32)   # the session cookie; every request but the page itself needs it
        self.token = secrets.token_urlsafe(24)    # in the launch link; the page sets the cookie only when it is given

    def start_sampling(self) -> None:
        if not self.focus_enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._loop, name="focus-sampler", daemon=True)
        self._thread.start()

    def stop_sampling(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=FOCUS_PERIOD_S + probe.TIMEOUT_S)
            self._thread = None

    def sample_once(self) -> Optional[dict]:
        if self.store.active() is None:
            return None
        got = self.window()
        if not got:
            return None
        app, title = got
        if PAGE_MARK in (title or ""):
            return None
        return self.store.note_focus(app, title)

    def _loop(self) -> None:
        while not self._stop.wait(FOCUS_PERIOD_S):
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001 - a probe failure must not stop the recorder
                traceback.print_exc()

    def state(self) -> dict:
        return {"active": self.store.active(), "tasks": self.store.list_tasks(), "root": self.store.root,
                "focus": self.focus_enabled, "version": __version__}


def _host_ok(host: str) -> bool:
    host = (host or "").strip().lower()
    if host.startswith("["):
        host = host.split("]")[0] + "]"
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return host in _LOOPBACK


def _origin_ok(origin: Optional[str], host: str) -> bool:
    """No Origin (a non-browser client) passes; a browser's Origin must be exactly this server."""
    if origin is None:
        return True
    return origin.lower() == f"http://{host.strip().lower()}"


def make_handler(rec: Recorder, page: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"taskrecorder/{__version__}"
        protocol_version = "HTTP/1.1"
        timeout = 60

        def log_message(self, fmt, *args):  # quiet by default
            if os.environ.get("TASKRECORDER_LOG"):
                super().log_message(fmt, *args)

        # -- replies -----------------------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in _HEADERS.items():
                self.send_header(k, v)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, status: int, msg: str) -> None:
            self._json({"error": msg}, status)

        def _guard(self) -> bool:
            ok = self._checks()
            if not ok:
                self.close_connection = True   # a refused POST's body is unread; do not reuse the connection
            return ok

        def _checks(self) -> bool:
            if not _host_ok(self.headers.get("Host", "")):
                self._error(HTTPStatus.FORBIDDEN, "this server only answers its own computer")
                return False
            if self.headers.get("Sec-Fetch-Site", "").lower() in ("cross-site", "same-site"):
                self._error(HTTPStatus.FORBIDDEN, "requests from other sites are refused")
                return False
            if self.command == "POST" and not _origin_ok(self.headers.get("Origin"), self.headers.get("Host", "")):
                self._error(HTTPStatus.FORBIDDEN, "cross-origin requests are refused")
                return False
            path = self.path.split("?", 1)[0]
            if path not in ("/", "/index.html") and not self._session_ok():
                self._error(HTTPStatus.FORBIDDEN, "no session: open the link printed when the recorder started")
                return False
            return True

        def _launch_token_ok(self) -> bool:
            q = parse_qs(urlsplit(self.path).query)
            t = (q.get("t") or [""])[0]
            return bool(t) and secrets.compare_digest(t, rec.token)

        def _session_ok(self) -> bool:
            jar = _cookies.SimpleCookie()
            try:
                jar.load(self.headers.get("Cookie", ""))
            except _cookies.CookieError:
                return False
            m = jar.get(COOKIE)
            return bool(m) and secrets.compare_digest(m.value, rec.secret)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > BODY_LIMIT:
                raise StoreError("request too large")
            raw = self.rfile.read(n) if n else b""
            if not raw:
                return {}
            try:
                obj = json.loads(raw.decode("utf-8"))
            except ValueError as e:
                raise StoreError(f"bad JSON: {e}") from e
            if not isinstance(obj, dict):
                raise StoreError("the body must be a JSON object")
            return obj

        # -- routes --------------------------------------------------------------------------
        def do_HEAD(self):  # noqa: N802
            self.do_GET()

        def do_GET(self):  # noqa: N802
            if not self._guard():
                return
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    if self._launch_token_ok():
                        self._send(200, page.encode("utf-8"), "text/html; charset=utf-8",
                                   {"Set-Cookie": f"{COOKIE}={rec.secret}; HttpOnly; SameSite=Strict; Path=/"})
                    elif self._session_ok():
                        self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                    else:
                        self._send(HTTPStatus.FORBIDDEN, LOCKED_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._json(rec.state())
                elif path == "/api/encyclopedia":
                    self._send(200, rec.store.encyclopedia().encode("utf-8"), "text/markdown; charset=utf-8")
                elif path.startswith("/api/task/"):
                    self._json(rec.store.get(unquote(path[len("/api/task/"):])))
                elif path.startswith("/api/procedure/"):
                    self._send(200, rec.store.procedure(unquote(path[len("/api/procedure/"):])).encode("utf-8"),
                               "text/markdown; charset=utf-8")
                elif path.startswith("/files/"):
                    self._file(path[len("/files/"):])
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
            except StoreError as e:
                self._error(HTTPStatus.NOT_FOUND, str(e))

        def _file(self, rel: str) -> None:
            parts = [p for p in unquote(rel).split("/") if p]
            if len(parts) < 2 or any(p in ("..", "") or p.startswith(".") for p in parts):
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            full = os.path.realpath(os.path.join(rec.store.root, *parts))
            if not full.startswith(os.path.realpath(rec.store.root) + os.sep) or not os.path.isfile(full):
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            extra = {"Content-Security-Policy": "sandbox; default-src 'none'"}
            if ctype not in INLINE_TYPES:
                ctype = "application/octet-stream"
                name = os.path.basename(full)
                ascii_name = name.encode("ascii", "replace").decode("ascii").replace('"', "_")
                extra["Content-Disposition"] = (f'attachment; filename="{ascii_name}"; '
                                                f"filename*=UTF-8''{quote(name)}")
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype, extra)

        def do_POST(self):  # noqa: N802
            if not self._guard():
                return
            path = self.path.split("?", 1)[0]
            try:
                b = self._body()
                if path == "/api/start":
                    tags = b.get("tags") or []
                    if isinstance(tags, str):
                        tags = [t for t in tags.replace(";", ",").split(",")]
                    task = rec.store.start(b.get("name", ""), b.get("goal", ""), b.get("owner", ""), tags,
                                           track_focus=bool(b.get("track_focus", True)) and rec.focus_enabled)
                    self._json(task)
                elif path == "/api/event":
                    kind = str(b.get("kind", "step"))
                    if kind not in KINDS or kind in ("focus", "capture", "file"):
                        raise StoreError(f"kind must be one of step, note, decision, pause, resume")
                    self._json(rec.store.add_event(kind, b.get("text", "")))
                elif path == "/api/redact":
                    self._json(rec.store.redact(int(b.get("id", 0))))
                elif path == "/api/capture":
                    self._json(rec.store.capture(rec.shoot, b.get("text", "")))
                elif path == "/api/file":
                    try:
                        data = base64.b64decode(str(b.get("data", "")), validate=True)
                    except ValueError as e:
                        raise StoreError("the file data is not valid base64") from e
                    self._json(rec.store.attach_file(str(b.get("name", "file")), data, b.get("text", "")))
                elif path == "/api/end":
                    self._json(rec.store.end(b.get("outcome", ""), b.get("pain"), b.get("wish", "")))
                elif path == "/api/discard":
                    rec.store.discard_active()
                    self._json({"ok": True})
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
            except StoreError as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    return Handler


def load_page() -> str:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
    with open(here, encoding="utf-8") as f:
        return f.read()


def make_server(rec: Recorder, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(rec, load_page()))
    srv.daemon_threads = True
    return srv


def serve(root: str, port: int = DEFAULT_PORT, open_browser: bool = True, focus: bool = True) -> int:
    rec = Recorder(Store(root), focus=focus)
    try:
        srv = make_server(rec, port)
    except OSError as e:
        print(f"could not listen on 127.0.0.1:{port}: {e}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t={rec.token}"
    print(f"Task recorder {__version__}")
    print(f"  recordings: {rec.store.root}")
    print(f"  open:       {url}")
    print("  (the link carries a key for this run; the page does not open without it)")
    print("  window tracking:", "on while a task records" if rec.focus_enabled else "off")
    print("Press Ctrl+C to stop.")
    rec.start_sampling()
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop_sampling()
        srv.server_close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="taskrecorder", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"where recordings are kept (default: {DEFAULT_DIR})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--no-focus", action="store_true", help="never record which window is in front")
    ap.add_argument("--export", nargs="?", const="-", metavar="FILE",
                    help="write every finished procedure as one Markdown document and exit")
    a = ap.parse_args(argv)
    if a.export is not None:
        text = Store(a.dir).encyclopedia()
        if a.export == "-":
            sys.stdout.write(text)
        else:
            with open(a.export, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"wrote {a.export}")
        return 0
    return serve(a.dir, a.port, open_browser=not a.no_open, focus=not a.no_focus)
