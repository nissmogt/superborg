"""The task recorder (taskrecorder/): the store on disk, and the server over real HTTP.

The desktop probes are replaced with fakes, so these tests run anywhere.
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from taskrecorder import app, store  # noqa: E402
from taskrecorder.store import Store, StoreError, render_procedure, slugify  # noqa: E402


# -- store -------------------------------------------------------------------------------------
def test_slugify():
    assert slugify("Review last week's Google Ads search terms!") == "review-last-week-s-google-ads-search-terms"
    assert slugify("   ") == "task"
    assert len(slugify("x" * 200)) == 60


def test_record_a_task(tmp_path):
    s = Store(str(tmp_path))
    assert s.active() is None and s.list_tasks() == []
    t = s.start("Review search terms", goal="negatives added", owner="Salar", tags=["google-ads", " weekly "])
    assert t["tags"] == ["google-ads", "weekly"]
    assert os.path.isdir(s.path(t["slug"], "captures"))
    with pytest.raises(StoreError):
        s.start("another")
    s.add_event("step", "Opened the search terms report")
    s.add_event("decision", "Excluded brand terms: they convert well")
    s.add_event("note", "  trailing spaces  ")
    with pytest.raises(StoreError):
        s.add_event("step", "")
    with pytest.raises(StoreError):
        s.add_event("bogus", "x")
    assert s.note_focus("Chrome", "Google Ads - Search terms")["kind"] == "focus"
    assert s.note_focus("Chrome", "Google Ads - Search terms") is None   # unchanged: not repeated
    assert s.note_focus("Excel", "terms.xlsx")["app"] == "Excel"
    s.attach_file("../../evil name.csv", b"a,b\n1,2\n", text="")
    ev = s.capture(lambda path: open(path, "wb").write(b"\x89PNG") or True)
    assert os.path.isfile(s.path(t["slug"], ev["file"]))
    with pytest.raises(StoreError):
        s.capture(lambda path: False)

    a = s.active()
    kinds = [e["kind"] for e in a["events"]]
    assert kinds == ["step", "decision", "note", "focus", "focus", "file", "capture"]
    assert a["events"][2]["text"] == "trailing spaces"
    assert a["events"][5]["file"] == "files/evil name.csv"
    assert [e["id"] for e in a["events"]] == list(range(1, 8))

    done = s.end(outcome="14 negatives added", pain="4", wish="pull the report for me")
    assert done["ended"] and done["pain"] == 4 and s.active() is None
    md = open(s.path(t["slug"], "procedure.md"), encoding="utf-8").read()
    assert md.startswith("# Review search terms")
    assert "1. Opened the search terms report" in md
    assert "**Decision.** Excluded brand terms" in md
    assert "| Chrome |" in md and "| Excel |" in md
    assert "**How tedious was this?** 4 / 5" in md
    assert "pull the report for me" in md
    assert s.list_tasks()[0]["entries"] == 3
    assert "14 negatives added" in s.encyclopedia()


def test_redaction_and_restart(tmp_path):
    s = Store(str(tmp_path))
    s.start("Secret-ish task")
    s.add_event("step", "called patient about balance")
    s.note_focus("Dentrix", "John Smith - ledger")
    cap = s.capture(lambda path: open(path, "wb").write(b"img") or True)
    s.redact(1)
    s.redact(2)
    s.redact(cap["id"])
    with pytest.raises(StoreError):
        s.redact(99)
    evs = s.active()["events"]
    assert evs[0]["text"] == store.REDACTED and evs[0]["redacted"]
    assert evs[1]["title"] == store.REDACTED
    assert evs[2]["file"] is None and os.listdir(s.path(s.active()["slug"], "captures")) == []
    # a restart picks the unfinished task back up
    s2 = Store(str(tmp_path))
    assert s2.active()["name"] == "Secret-ish task"
    s2.discard_active()
    assert s2.active() is None and s2.list_tasks() == []


def test_pain_and_duplicates(tmp_path):
    s = Store(str(tmp_path))
    a = s.start("Same name")
    s.end(pain="nope")
    b = s.start("Same name")
    s.end(pain=9)
    assert b["slug"] == a["slug"] + "-2"
    tasks = {t["slug"]: t for t in s.list_tasks()}
    assert len(tasks) == 2
    assert s.get(b["slug"])["pain"] == 5 and s.get(a["slug"])["pain"] is None


def test_render_without_end():
    task = {"slug": "x", "name": "Open task", "started": "2026-11-05T06:00:00-08:00", "ended": None,
            "events": [{"id": 1, "t": "2026-11-05T06:01:30-08:00", "kind": "step", "text": "did a thing"}]}
    md = render_procedure(task)
    assert "1. did a thing  `1m 30s`" in md and "*Not written down.*" in md


# -- server ------------------------------------------------------------------------------------
class FakeDesktop:
    def __init__(self):
        self.front = ("Chrome", "Google Ads")
        self.shots = 0

    def window(self):
        return self.front

    def shoot(self, path):
        self.shots += 1
        with open(path, "wb") as f:
            f.write(b"\x89PNG fake")
        return True


@pytest.fixture
def server(tmp_path):
    desk = FakeDesktop()
    rec = app.Recorder(Store(str(tmp_path)), focus=True, window=desk.window, shoot=desk.shoot)
    srv = app.make_server(rec, port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv.server_address[1], rec, desk
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": f"127.0.0.1:{port}"}
    h.update(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    ct = r.getheader("Content-Type") or ""
    c.close()
    return r.status, (json.loads(raw) if "json" in ct else raw.decode("utf-8", "replace")), ct


def test_http_flow(server):
    port, rec, desk = server
    st, page, ct = call(port, "GET", "/")
    assert st == 200 and "<title>Task recorder</title>" in page and "text/html" in ct

    st, state, _ = call(port, "GET", "/api/state")
    assert st == 200 and state["active"] is None and state["focus"] is True

    st, _, _ = call(port, "POST", "/api/event", {"kind": "step", "text": "x"})
    assert st == 400

    st, task, _ = call(port, "POST", "/api/start", {"name": "Weekly ads review", "owner": "Salar",
                                                     "tags": "google-ads, weekly", "goal": "negatives"})
    assert st == 200 and task["tags"] == ["google-ads", "weekly"]
    slug = task["slug"]

    assert call(port, "POST", "/api/event", {"kind": "step", "text": "Opened Ads"})[0] == 200
    assert call(port, "POST", "/api/event", {"kind": "focus", "text": "nope"})[0] == 400
    assert rec.sample_once()["title"] == "Google Ads"
    assert rec.sample_once() is None
    desk.front = ("Chrome", "Task recorder")      # our own page is not recorded
    assert rec.sample_once() is None
    desk.front = ("Excel", "terms.xlsx")
    assert rec.sample_once()["app"] == "Excel"

    st, ev, _ = call(port, "POST", "/api/capture", {"text": "the report"})
    assert st == 200 and ev["file"].startswith("captures/") and desk.shots == 1
    st, img, ct = call(port, "GET", f"/files/{slug}/{ev['file']}")
    assert st == 200 and ct == "image/png" and img.startswith("�PNG") or st == 200

    data = base64.b64encode(b"a,b\n").decode()
    st, fev, _ = call(port, "POST", "/api/file", {"name": "report.csv", "data": data})
    assert st == 200 and fev["file"] == "files/report.csv"
    st, body, ct = call(port, "GET", f"/files/{slug}/files/report.csv")
    assert st == 200 and body == "a,b\n" and ct.startswith("text/plain")
    assert call(port, "POST", "/api/file", {"name": "x", "data": "!!!"})[0] == 400

    assert call(port, "GET", "/files/../tests/test_recorder.py")[0] == 404
    assert call(port, "GET", f"/files/{slug}/../../setup.py")[0] == 404
    assert call(port, "GET", f"/files/{slug}/.hidden")[0] == 404

    st, red, _ = call(port, "POST", "/api/redact", {"id": 1})
    assert st == 200 and red["redacted"]

    st, t, _ = call(port, "GET", f"/api/task/{slug}")
    assert st == 200 and [e["kind"] for e in t["events"]] == ["step", "focus", "focus", "capture", "file"]
    assert call(port, "GET", "/api/task/nope")[0] == 404

    st, done, _ = call(port, "POST", "/api/end", {"outcome": "ok", "pain": 2, "wish": "fetch the report"})
    assert st == 200 and done["pain"] == 2
    st, md, ct = call(port, "GET", f"/api/procedure/{slug}")
    assert st == 200 and "text/markdown" in ct and "# Weekly ads review" in md and "| Excel |" in md
    st, enc, _ = call(port, "GET", "/api/encyclopedia")
    assert st == 200 and "1 recorded task(s)" in enc

    st, state, _ = call(port, "GET", "/api/state")
    assert state["active"] is None and state["tasks"][0]["slug"] == slug


def test_discard_and_guards(server):
    port, rec, desk = server
    assert call(port, "POST", "/api/start", {"name": "oops"})[0] == 200
    assert call(port, "POST", "/api/discard", {})[0] == 200
    assert call(port, "GET", "/api/state")[1]["tasks"] == []
    assert call(port, "POST", "/api/discard", {})[0] == 400

    # requests not addressed to this computer, or sent by another page, are refused
    assert call(port, "GET", "/api/state", headers={"Host": "evil.example"})[0] == 403
    assert call(port, "POST", "/api/start", {"name": "x"}, headers={"Origin": "http://evil.example"})[0] == 403
    assert call(port, "POST", "/api/start", {"name": "x"}, headers={"Origin": f"http://localhost:{port}"})[0] == 200
    assert call(port, "GET", "/api/state", headers={"Host": "localhost"})[0] == 200
    assert call(port, "GET", "/api/state", headers={"Host": "[::1]:80"})[0] == 200
    assert call(port, "GET", "/nothing")[0] == 404


def test_export_cli(tmp_path, capsys):
    s = Store(str(tmp_path / "r"))
    s.start("Exported task", owner="Kareem")
    s.add_event("step", "one")
    s.end(outcome="fine")
    assert app.main(["--dir", str(tmp_path / "r"), "--export"]) == 0
    out = capsys.readouterr().out
    assert "# Procedures" in out and "# Exported task" in out
    dest = tmp_path / "all.md"
    assert app.main(["--dir", str(tmp_path / "r"), "--export", str(dest)]) == 0
    assert "Exported task" in dest.read_text(encoding="utf-8")
