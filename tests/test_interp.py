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
            assert [x[0] for x in got["top"]] == [x[0] for x in want["top"]]
            assert [x[1] for x in got["top"]] == pytest.approx([x[1] for x in want["top"]], abs=1e-4)
        n += 1
    assert n >= 2


def _old_lens_summary(ts, chosen: int, k: int) -> dict:
    """The summary of lens_report's NumPy path (TraceOptions.fused_lens = False), unrounded."""
    layers = ts.lens_layers()
    z = ts.llm.ctx.lens(layers)[:, : ts.llm.model.n_vocab]
    n_vocab = z.shape[1]
    zc = z[:, chosen].copy()
    rank = (z > zc[:, None]).sum(axis=1)
    top = np.argpartition(z, n_vocab - k, axis=1)[:, n_vocab - k:]
    zt = np.take_along_axis(z, top, axis=1)
    order = np.argsort(-zt, axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    zt = np.take_along_axis(zt, order, axis=1).astype(np.float64)
    m = zt[:, :1]
    lse = m[:, 0] + np.log(np.exp(z - m.astype(np.float32)).sum(axis=1, dtype=np.float64))
    z64 = z.astype(np.float64)
    m64 = z64.max(axis=1)
    return {"top": top, "p_top": np.exp(zt - lse[:, None]), "p": np.exp(zc.astype(np.float64) - lse), "rank": rank,
            "lse": lse, "lse64": m64 + np.log(np.exp(z64 - m64[:, None]).sum(axis=1))}


PROMPTS = [Q, [{"role": "user", "content": "Write a haiku about the sea."}],
           [{"role": "user", "content": "List three prime numbers."}]]


def test_fused_lens_matches_the_numpy_path(session):
    """The summary computed inside the engine (borg_ctx_lens_summary) against the full logits
    summarized by NumPy: the same tokens and ranks, probabilities within 1e-6, and a log-sum-exp
    within 1e-5 of NumPy's float64 one. The reports of both paths agree to their rounding."""
    session.opt.lens_stride = 1
    k = session.opt.lens_top
    n = 0
    for msgs in PROMPTS:
        for ev in session.stream_chat(msgs, max_tokens=8):
            if ev["type"] != "token":
                continue
            chosen = ev["id"]
            layers = session.lens_layers()
            fused = session.llm.ctx.lens_summary(layers, k, chosen)
            old = _old_lens_summary(session, chosen, k)
            assert np.array_equal(fused["top_ids"], old["top"])
            assert np.array_equal(fused["chosen_rank"], old["rank"])
            p_fused = np.exp(fused["chosen_logit"].astype(np.float64) - fused["lse"])
            assert np.abs(p_fused - old["p"]).max() <= 1e-6
            assert np.abs(np.exp(fused["top_logits"] - fused["lse"][:, None]) - old["p_top"]).max() <= 1e-6
            assert np.abs(fused["lse"] - old["lse64"]).max() <= 1e-5
            assert fused["lse"].dtype == np.float64
            # the reports: ev["lens"] came from the fused path (the default)
            session.opt.fused_lens = False
            try:
                plain = session.lens_report(chosen)
            finally:
                session.opt.fused_lens = True
            assert len(plain) == len(ev["lens"]) == session.cfg["n_layers"] + 1
            for a, b in zip(ev["lens"], plain):
                assert a["layer"] == b["layer"] and a["rank"] == b["rank"]
                assert [x[2] for x in a["top"]] == [x[2] for x in b["top"]]
                assert [x[0] for x in a["top"]] == [x[0] for x in b["top"]]
                assert a["p"] == pytest.approx(b["p"], abs=1.1e-5)
                assert [x[1] for x in a["top"]] == pytest.approx([x[1] for x in b["top"]], abs=1.1e-4)
            n += 1
    assert n == 24


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
    evs = []
    for ev in session.stream_chat([{"role": "user", "content": "Count from one to fifty."}], max_tokens=40,
                                  should_stop=lambda: sum(e["type"] == "token" for e in evs) >= 3):
        evs.append(ev)
    toks = [e for e in evs if e["type"] == "token"]
    assert evs[-1]["type"] == "done" and evs[-1]["stop_reason"] == "cancelled"
    assert len(toks) == 3 and evs[-1]["n_gen"] == 3


def test_token_ids_with_alternatives_and_lens_tops(session):
    """[text, p, id]: byte tokens of one emoji all read U+FFFD, so only the id tells them apart."""
    for ev in session.stream_chat([{"role": "user", "content": "Reply with three emoji."}], max_tokens=8,
                                  temperature=1.0, seed=5):
        if ev["type"] != "token":
            continue
        rows = ev["alternatives"] + [r for x in ev["lens"] for r in x["top"]]
        assert all(len(r) == 3 and isinstance(r[2], int) and r[0] == session.token_text(r[2]) for r in rows)
        assert len({r[2] for r in ev["alternatives"]}) == len(ev["alternatives"])
        chosen = [r for r in ev["alternatives"] if r[2] == ev["id"]]
        assert not chosen or abs(chosen[0][1] - ev["prob"]) < 1e-4
        assert ev["lens"][-1]["top"][0][2] == ev["alternatives"][0][2]  # the last layer is the output


def test_decode_ms_on_every_token(session):
    """Including the first (the pass over the prompt's last token) and an end-of-turn token."""
    toks = [e for e in session.stream_chat(Q, max_tokens=40) if e["type"] == "token"]
    assert toks[-1]["eos"], "expected the reply to end within 40 tokens"
    assert all(e["decode_ms"] > 0 and e["trace_ms"] >= 0 for e in toks)


def _long_prompt(n_items: int) -> list[dict]:
    text = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(n_items))
    return [{"role": "user", "content": text + "\nSummarize."}]


def test_prefill_in_chunks(session, monkeypatch):
    from superborg import interp
    msgs = _long_prompt(25)  # about 330 tokens: three chunks
    session.llm.ctx.reset()
    evs = list(session.stream_chat(msgs, max_tokens=4))
    pre = [e for e in evs if e["type"] == "prefill"]
    prompt = next(e for e in evs if e["type"] == "prompt")
    assert evs.index(prompt) == len(pre) and prompt["n_reused"] == 0  # progress first, then the prompt
    total = prompt["n_prompt"]
    assert [e["done"] for e in pre] == list(range(interp.PREFILL_CHUNK, total, interp.PREFILL_CHUNK))
    assert all(e["total"] == total for e in pre) and len(pre) >= 2
    # the same tokens as a prompt evaluated in one piece
    session.llm.ctx.reset()
    monkeypatch.setattr(interp, "PREFILL_CHUNK", 1 << 20)
    one = list(session.stream_chat(msgs, max_tokens=4))
    assert [e["type"] for e in one][0] == "prompt"
    assert [e["id"] for e in one if e["type"] == "token"] == [e["id"] for e in evs if e["type"] == "token"]
    # a prompt that fits one chunk sends no progress: here only the new turn is evaluated
    again = list(session.stream_chat(msgs + [{"role": "assistant", "content": "Foxes."},
                                             {"role": "user", "content": "Shorter."}], max_tokens=1))
    assert again[0]["type"] == "prompt" and again[0]["n_reused"] > total // 2


def test_should_stop_during_prefill(session):
    """A stop between prompt chunks ends with "prompt" and an empty cancelled "done"; the chunks
    already evaluated stay in the cache."""
    from superborg import interp
    msgs = _long_prompt(25)
    session.llm.ctx.reset()
    evs = []
    for ev in session.stream_chat(msgs, max_tokens=4, should_stop=lambda: any(e["type"] == "prefill" for e in evs)):
        evs.append(ev)
    assert [e["type"] for e in evs] == ["prefill", "prompt", "done"]
    assert evs[-1]["stop_reason"] == "cancelled" and evs[-1]["n_gen"] == 0 and evs[-1]["text"] == ""
    assert evs[1]["prompt_ms"] > 0 and session.llm.ctx.n_past == interp.PREFILL_CHUNK
    # stopped before anything is evaluated: no forward pass at all
    evs = list(session.stream_chat(msgs, max_tokens=4, should_stop=lambda: True))
    assert [e["type"] for e in evs] == ["prompt", "done"] and evs[0]["n_reused"] == interp.PREFILL_CHUNK
    assert session.llm.ctx.n_past == interp.PREFILL_CHUNK


def test_request_errors_come_before_any_event(session):
    gen = session.stream_chat([{"role": "user", "content": "word " * 2000}], should_stop=lambda: True)
    with pytest.raises(ValueError, match="context"):
        next(gen)


def test_info(session):
    info = session.info()
    for k in ("model", "quant", "n_layers", "n_heads", "dim", "hidden_dim", "vocab", "n_ctx", "lens_layers", "threads"):
        assert k in info
    assert info["lens_layers"] == list(range(info["n_layers"] + 1))
