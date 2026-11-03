"""On-disk recordings: one folder per task, owned by the person who recorded it.

    <root>/
      2026-11-05-reconcile-search-terms/
        task.json       name, goal, owner, tags, started, ended, outcome, pain, wish
        events.jsonl    one event per line, in order (see Event kinds below)
        captures/       screenshots taken on request
        files/          files the user attached
        procedure.md    rendered when the task ends: a draft procedure for the encyclopedia

Event kinds: step (something the user did), note, decision (a choice and why), focus (the window
in front changed: app + title), capture (screenshot), file (attachment), pause, resume.
Every event has an integer id, an ISO timestamp `t`, a kind and a text; redact(id) blanks the
text and title of an event in place and marks it redacted.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

KINDS = ("step", "note", "decision", "focus", "capture", "file", "pause", "resume")
TEXT_LIMIT = 4000
FILE_LIMIT = 50 * 1024 * 1024
REDACTED = "[redacted]"
_SLUG = re.compile(r"[^a-z0-9]+")


class StoreError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def slugify(name: str) -> str:
    s = _SLUG.sub("-", name.lower()).strip("-")
    return (s or "task")[:60]


def _clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").strip()
    return text[:TEXT_LIMIT]


def safe_filename(name: str) -> str:
    name = os.path.basename(str(name or "")).strip() or "file"
    name = re.sub(r"[^\w.\- ]+", "_", name)
    return name[:120]


class Store:
    """One recordings root. At most one task records at a time; it survives a restart."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()
        self._active: Optional[str] = None
        self._last_focus: Optional[tuple] = None
        for slug in self._slugs():
            t = self._read_task(slug)
            if t and not t.get("ended"):
                self._active = slug
                break

    # -- paths -----------------------------------------------------------------------------
    def path(self, slug: str, *parts: str) -> str:
        return os.path.join(self.root, slug, *parts)

    def _slugs(self) -> List[str]:
        try:
            names = sorted(os.listdir(self.root))
        except OSError:
            return []
        return [n for n in names if os.path.isfile(self.path(n, "task.json"))]

    def _read_task(self, slug: str) -> Optional[dict]:
        try:
            with open(self.path(slug, "task.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write_task(self, slug: str, task: dict) -> None:
        tmp = self.path(slug, "task.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path(slug, "task.json"))

    def _read_events(self, slug: str) -> List[dict]:
        out = []
        try:
            with open(self.path(slug, "events.jsonl"), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except OSError:
            pass
        return out

    def _write_events(self, slug: str, events: List[dict]) -> None:
        tmp = self.path(slug, "events.jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path(slug, "events.jsonl"))

    # -- tasks -----------------------------------------------------------------------------
    def active(self) -> Optional[dict]:
        with self._lock:
            if not self._active:
                return None
            t = self._read_task(self._active)
            if t is None:
                self._active = None
                return None
            t["events"] = self._read_events(self._active)
            return t

    def list_tasks(self) -> List[dict]:
        out = []
        for slug in self._slugs():
            t = self._read_task(slug)
            if t:
                n = sum(1 for e in self._read_events(slug) if e["kind"] in ("step", "note", "decision"))
                out.append({"slug": slug, "name": t["name"], "owner": t.get("owner", ""),
                            "started": t["started"], "ended": t.get("ended"), "entries": n,
                            "tags": t.get("tags", [])})
        out.sort(key=lambda t: t["started"], reverse=True)
        return out

    def get(self, slug: str) -> dict:
        slug = os.path.basename(slug)
        t = self._read_task(slug)
        if t is None:
            raise StoreError(f"no recording named {slug!r}")
        t["events"] = self._read_events(slug)
        return t

    def start(self, name: str, goal: str = "", owner: str = "", tags=None, track_focus: bool = True) -> dict:
        name = _clean_text(name)
        if not name:
            raise StoreError("the task needs a name")
        with self._lock:
            if self._active:
                raise StoreError("a task is already recording; end it first")
            base = f"{datetime.now().strftime('%Y-%m-%d')}-{slugify(name)}"
            slug, n = base, 2
            while os.path.exists(self.path(slug)):
                slug, n = f"{base}-{n}", n + 1
            for d in ("captures", "files"):
                os.makedirs(self.path(slug, d), exist_ok=True)
            tags = [t.strip() for t in (tags or []) if str(t).strip()][:20]
            task = {"slug": slug, "name": name, "goal": _clean_text(goal), "owner": _clean_text(owner)[:200],
                    "tags": tags, "track_focus": bool(track_focus), "started": _now(), "ended": None,
                    "outcome": "", "pain": None, "wish": ""}
            self._write_task(slug, task)
            self._write_events(slug, [])
            self._active = slug
            self._last_focus = None
            task["events"] = []
            return task

    def _require_active(self) -> str:
        if not self._active:
            raise StoreError("no task is recording")
        return self._active

    def add_event(self, kind: str, text: str = "", **extra) -> dict:
        if kind not in KINDS:
            raise StoreError(f"unknown event kind {kind!r}")
        with self._lock:
            slug = self._require_active()
            events = self._read_events(slug)
            ev = {"id": (events[-1]["id"] + 1) if events else 1, "t": _now(), "kind": kind,
                  "text": _clean_text(text)}
            for k, v in extra.items():
                if v is not None:
                    ev[k] = v
            if kind in ("step", "note", "decision") and not ev["text"]:
                raise StoreError("the entry is empty")
            with open(self.path(slug, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            return ev

    def note_focus(self, app: str, title: str) -> Optional[dict]:
        """Record the window in front if it changed since the last focus event."""
        with self._lock:
            if not self._active:
                return None
            task = self._read_task(self._active) or {}
            if not task.get("track_focus", True):
                return None
            key = (app or "", title or "")
            if key == self._last_focus:
                return None
            self._last_focus = key
            return self.add_event("focus", "", app=(app or "")[:200], title=(title or "")[:300])

    def redact(self, event_id: int) -> dict:
        with self._lock:
            slug = self._require_active()
            events = self._read_events(slug)
            for e in events:
                if e["id"] == event_id:
                    e["text"] = REDACTED
                    if "title" in e:
                        e["title"] = REDACTED
                    if e["kind"] == "capture" and e.get("file"):
                        try:
                            os.remove(self.path(slug, e["file"]))
                        except OSError:
                            pass
                        e["file"] = None
                    e["redacted"] = True
                    self._write_events(slug, events)
                    return e
            raise StoreError(f"no event {event_id}")

    def attach_file(self, filename: str, data: bytes, text: str = "") -> dict:
        if len(data) > FILE_LIMIT:
            raise StoreError("the file is larger than 50 MB")
        with self._lock:
            slug = self._require_active()
            name = safe_filename(filename)
            dest, n = self.path(slug, "files", name), 2
            stem, ext = os.path.splitext(name)
            while os.path.exists(dest):
                dest, n = self.path(slug, "files", f"{stem}-{n}{ext}"), n + 1
            with open(dest, "wb") as f:
                f.write(data)
            rel = os.path.relpath(dest, self.path(slug)).replace(os.sep, "/")
            return self.add_event("file", text or f"attached {os.path.basename(dest)}", file=rel,
                                  size=len(data))

    def capture(self, shoot: Callable[[str], bool], text: str = "") -> dict:
        with self._lock:
            slug = self._require_active()
            name = time.strftime("%H%M%S") + ".png"
            dest = self.path(slug, "captures", name)
            n = 2
            while os.path.exists(dest):
                dest, n = self.path(slug, "captures", f"{name[:-4]}-{n}.png"), n + 1
            if not shoot(dest):
                raise StoreError("no screenshot tool worked on this computer")
            rel = os.path.relpath(dest, self.path(slug)).replace(os.sep, "/")
            return self.add_event("capture", text or "screen captured", file=rel)

    def end(self, outcome: str = "", pain=None, wish: str = "") -> dict:
        with self._lock:
            slug = self._require_active()
            task = self._read_task(slug) or {}
            task["ended"] = _now()
            task["outcome"] = _clean_text(outcome)
            task["wish"] = _clean_text(wish)
            try:
                task["pain"] = min(5, max(1, int(pain))) if pain not in (None, "") else None
            except (TypeError, ValueError):
                task["pain"] = None
            self._write_task(slug, task)
            task["events"] = self._read_events(slug)
            with open(self.path(slug, "procedure.md"), "w", encoding="utf-8") as f:
                f.write(render_procedure(task))
            self._active = None
            self._last_focus = None
            return task

    def discard_active(self) -> None:
        """Delete the recording in progress (the user changed their mind)."""
        with self._lock:
            slug = self._require_active()
            shutil.rmtree(self.path(slug), ignore_errors=True)
            self._active = None
            self._last_focus = None

    def procedure(self, slug: str) -> str:
        slug = os.path.basename(slug)
        p = self.path(slug, "procedure.md")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
        return render_procedure(self.get(slug))

    def encyclopedia(self) -> str:
        """Every finished procedure in one Markdown document, newest first, with an index."""
        tasks = [t for t in self.list_tasks() if t["ended"]]
        lines = ["# Procedures", "", f"{len(tasks)} recorded task(s). Folder: `{self.root}`", ""]
        for t in tasks:
            lines.append(f"- [{t['name']}](#{t['slug']}) · {t['started'][:10]} · {t['owner'] or 'unknown'}")
        for t in tasks:
            lines += ["", "---", "", f'<a id="{t["slug"]}"></a>', "", self.procedure(t["slug"]).rstrip(), ""]
        return "\n".join(lines)


# -- rendering -------------------------------------------------------------------------------
def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _app_time(task: dict) -> Dict[str, dict]:
    """Seconds in front, and the window titles seen, per application."""
    events = task.get("events", [])
    end = _parse(task["ended"]) if task.get("ended") else (_parse(events[-1]["t"]) if events else None)
    out: Dict[str, dict] = {}
    cur = None
    for e in events:
        if e["kind"] == "pause" and cur:
            out[cur[0]]["seconds"] += (_parse(e["t"]) - cur[1]).total_seconds()
            cur = None
        if e["kind"] != "focus":
            continue
        t = _parse(e["t"])
        if cur:
            out[cur[0]]["seconds"] += (t - cur[1]).total_seconds()
        app = e.get("app") or "unknown"
        d = out.setdefault(app, {"seconds": 0.0, "titles": []})
        title = e.get("title") or ""
        if title and title not in d["titles"] and len(d["titles"]) < 8:
            d["titles"].append(title)
        cur = (app, t)
    if cur and end:
        out[cur[0]]["seconds"] += (end - cur[1]).total_seconds()
    return out


def render_procedure(task: dict) -> str:
    events = task.get("events", [])
    start = _parse(task["started"])
    lines = [f"# {task['name']}", ""]
    meta = [f"Recorded {task['started'][:10]}"]
    if task.get("owner"):
        meta.append(f"by {task['owner']}")
    if task.get("ended"):
        meta.append(f"({_fmt_duration((_parse(task['ended']) - start).total_seconds())})")
    if task.get("tags"):
        meta.append("· " + ", ".join(task["tags"]))
    lines += [" ".join(meta), ""]
    if task.get("goal"):
        lines += [f"**Goal.** {task['goal']}", ""]

    lines += ["## Steps", ""]
    n = 0
    for e in events:
        off = _fmt_duration((_parse(e["t"]) - start).total_seconds())
        if e["kind"] == "step":
            n += 1
            lines.append(f"{n}. {e['text']}  `{off}`")
        elif e["kind"] == "decision":
            lines.append(f"   > **Decision.** {e['text']}")
        elif e["kind"] == "note":
            lines.append(f"   *Note.* {e['text']}")
        elif e["kind"] == "capture":
            lines.append(f"   ![{e['text']}]({e['file']})" if e.get("file") else f"   *Capture redacted.*")
        elif e["kind"] == "file":
            lines.append(f"   Attached: [{os.path.basename(e['file'])}]({e['file']})" if e.get("file")
                         else "   *Attachment redacted.*")
        elif e["kind"] == "pause":
            lines.append(f"   *(paused at {off})*")
    if n == 0:
        lines.append("*No steps were written down.*")
    lines.append("")

    apps = _app_time(task)
    if apps:
        lines += ["## Applications used", "", "| Application | Time in front | Windows seen |", "|---|---|---|"]
        for app, d in sorted(apps.items(), key=lambda kv: -kv[1]["seconds"]):
            titles = "; ".join(t.replace("|", "/") for t in d["titles"]) or "—"
            lines.append(f"| {app} | {_fmt_duration(d['seconds'])} | {titles} |")
        lines.append("")

    lines += ["## Outcome", ""]
    lines.append(task.get("outcome") or "*Not written down.*")
    if task.get("pain") is not None:
        lines += ["", f"**How tedious was this?** {task['pain']} / 5"]
    if task.get("wish"):
        lines += ["", f"**What should a tool do here?** {task['wish']}"]
    lines += ["", "---", "", f"Data: `task.json`, `events.jsonl` in `{task['slug']}/`. Written by the task "
              "recorder; the person who recorded it owns it.", ""]
    return "\n".join(lines)
