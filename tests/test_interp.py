"""superborg.interp.TraceSession: per-token traces of chat generation (smollm2-135m-q8_0.borg)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_ui import MODEL, THREADS, model_path  # noqa: E402

Q = [{"role": "user", "content": "What is the capital of France?"}]


@pytest.fixture(scope="module")
def session():
    path = model_path(MODEL)
    try:
        from superborg import LLM
        from superborg.interp import TraceOptions, TraceSession
        llm = LLM(path, threads=THREADS, n_ctx=512)
    except (OSError, FileNotFoundError) as e:
        pytest.skip(f"engine unavailable: {e}")
    yield TraceSession(llm, TraceOptions())
    llm.close()


def _reference_lens(ts, chosen: int, top: int = 3) -> list[dict]:
    """The logit lens as first written: a float64 softmax of every lens row."""
    from superborg.interp import _softmax
    layers = ts.lens_layers()
    lp = _softmax(ts.llm.ctx.lens(layers)[:, : ts.llm.model.n_vocab])
    out = []
    for li, row in zip(layers, lp):
        t = np.argpartition(-row, top)[:top]
        t = t[np.argsort(-row[t])]
        out.append({"layer": li, "top": [[ts.token_text(int(i)), float(row[i])] for i in t],
                    "p": float(row[chosen]), "rank": int((row > row[chosen]).sum())})
    return out


def test_lens_report_matches_float64_softmax(session):
    session.opt.lens_stride = 1
    n = 0
    for ev in session.stream_chat(Q, max_tokens=6):
        if ev["type"] != "token":
            continue
        # the generator is suspended right after the yield: the trace still holds this token's pass
        ref = _reference_lens(session, ev["id"])
        assert len(ev["lens"]) == len(ref) == session.cfg["n_layers"] + 1
        for got, want in zip(ev["lens"], ref):
            assert got["layer"] == want["layer"] and got["rank"] == want["rank"]
            assert got["p"] == pytest.approx(want["p"], abs=1e-5)
            assert [s for s, _ in got["top"]] == [s for s, _ in want["top"]]
            assert [p for _, p in got["top"]] == pytest.approx([p for _, p in want["top"]], abs=1e-4)
        n += 1
    assert n >= 2


def test_lens_stride_keeps_the_output_layer(session):
    n_layers = session.cfg["n_layers"]
    session.opt.lens_stride = 7
    try:
        assert session.lens_layers()[-1] == n_layers and session.lens_layers()[:2] == [0, 7]
        evs = [e for e in session.stream_chat(Q, max_tokens=2) if e["type"] == "token"]
        assert all([x["layer"] for x in e["lens"]] == session.lens_layers() for e in evs)
    finally:
        session.opt.lens_stride = 1


def test_should_stop_cancels_after_the_current_token(session):
    seen = []
    evs = list(session.stream_chat([{"role": "user", "content": "Count from one to fifty."}], max_tokens=40,
                                   should_stop=lambda: len(seen) >= 2 or seen.append(1)))
    toks = [e for e in evs if e["type"] == "token"]
    assert evs[-1]["type"] == "done" and evs[-1]["stop_reason"] == "cancelled"
    assert len(toks) == 3 and evs[-1]["n_gen"] == 3


def test_info(session):
    info = session.info()
    for k in ("model", "quant", "n_layers", "n_heads", "dim", "hidden_dim", "vocab", "n_ctx", "lens_layers", "threads"):
        assert k in info
    assert info["lens_layers"] == list(range(info["n_layers"] + 1))
