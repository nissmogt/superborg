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


def _reference_lens(ts, chosen: int, rival: int, top: int = 3) -> list[dict]:
    """The logit lens as first written: a float64 softmax of every lens row."""
    from superborg.interp import _softmax
    layers = ts.lens_layers()
    lp = _softmax(ts.llm.ctx.lens(layers)[:, : ts.llm.model.n_vocab])
    out = []
    for li, row in zip(layers, lp):
        t = np.argpartition(-row, top)[:top]
        t = t[np.argsort(-row[t])]
        out.append({"layer": li, "top": [[ts.token_text(int(i)), float(row[i])] for i in t],
                    "p": float(row[chosen]), "rank": int((row > row[chosen]).sum()),
                    "entropy": float(-(row[row > 0] * np.log(row[row > 0])).sum()),
                    "rival_p": float(row[rival]), "rival_rank": int((row > row[rival]).sum())})
    return out


def test_lens_report_matches_float64_softmax(session):
    session.opt.lens_stride = 1
    n = 0
    for ev in session.stream_chat(Q, max_tokens=6):
        if ev["type"] != "token":
            continue
        # the generator is suspended right after the yield: the trace still holds this token's pass
        ref = _reference_lens(session, ev["id"], ev["rival"][2])
        assert len(ev["lens"]) == len(ref) == session.cfg["n_layers"] + 1
        for got, want in zip(ev["lens"], ref):
            assert got["layer"] == want["layer"] and got["rank"] == want["rank"]
            assert got["p"] == pytest.approx(want["p"], abs=1e-5)
            assert [x[0] for x in got["top"]] == [x[0] for x in want["top"]]
            assert [x[1] for x in got["top"]] == pytest.approx([x[1] for x in want["top"]], abs=1e-4)
            assert got["entropy"] == pytest.approx(want["entropy"], abs=1e-4)
            assert got["rival_rank"] == want["rival_rank"] and got["rival_p"] == pytest.approx(want["rival_p"], abs=1e-5)
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
            chosen, rival = ev["id"], ev["rival"][2]
            layers = session.lens_layers()
            fused = session.llm.ctx.lens_summary(layers, k, [chosen, rival])
            old = _old_lens_summary(session, chosen, k)
            assert np.array_equal(fused["top_ids"], old["top"])
            assert np.array_equal(fused["track_rank"][:, 0], old["rank"])
            p_fused = np.exp(fused["track_logit"][:, 0].astype(np.float64) - fused["lse"])
            assert np.abs(p_fused - old["p"]).max() <= 1e-6
            assert np.abs(np.exp(fused["top_logits"] - fused["lse"][:, None]) - old["p_top"]).max() <= 1e-6
            assert np.abs(fused["lse"] - old["lse64"]).max() <= 1e-5
            assert fused["lse"].dtype == np.float64
            # the reports: ev["lens"] came from the fused path (the default)
            session.opt.fused_lens = False
            try:
                plain = session.lens_report(chosen, rival)
            finally:
                session.opt.fused_lens = True
            assert len(plain) == len(ev["lens"]) == session.cfg["n_layers"] + 1
            for a, b in zip(ev["lens"], plain):
                assert a["layer"] == b["layer"] and a["rank"] == b["rank"] and a["rival_rank"] == b["rival_rank"]
                assert [x[2] for x in a["top"]] == [x[2] for x in b["top"]]
                assert [x[0] for x in a["top"]] == [x[0] for x in b["top"]]
                assert a["p"] == pytest.approx(b["p"], abs=1.1e-5)
                assert a["rival_p"] == pytest.approx(b["rival_p"], abs=1.1e-5)
                assert a["entropy"] == pytest.approx(b["entropy"], abs=1.1e-4)
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
    assert info["basis"] is False


# ---- the attention encoding and statistics (pure functions, shared with wasm/trace-session.js) -------

def test_encode_row_round_trip_sink_and_sparsity():
    """encode_row keeps the sink (position 0) as a float, scales the 8-bit weights by the largest
    OTHER entry, keeps exactly the entries at or above 1/(2n) that quantize to at least 1, and
    decode_row gets each kept entry back within half a quantization step."""
    from superborg.interp import decode_row, encode_row
    rng = np.random.default_rng(1)
    n = 500
    row = rng.random(n) ** 6
    row[0] = 5 * row.sum()  # a sink holding most of the mass
    row = (row / row.sum()).astype(np.float32)
    enc = encode_row(row)
    assert set(enc) == {"scale", "sink", "pos", "w"}
    assert enc["sink"] == pytest.approx(float(row[0]), abs=5e-5) and enc["sink"] > 0.8
    assert enc["scale"] == pytest.approx(float(row[1:].max()), abs=1e-6)  # the sink never sets the scale
    assert enc["pos"] == sorted(enc["pos"]) and 0 not in enc["pos"] and len(enc["pos"]) == len(enc["w"])
    assert all(isinstance(p, int) and isinstance(w, int) and 1 <= w <= 255 for p, w in zip(enc["pos"], enc["w"]))
    assert 255 in enc["w"]
    kept = set(enc["pos"])
    step = enc["scale"] / 255
    for j in range(1, n):
        assert (j in kept) == (row[j] >= 1 / (2 * n) and row[j] * 255 / enc["scale"] + 0.5 >= 1), j
    assert 0 < len(kept) < n - 1  # sparse: the small entries are gone
    dec = decode_row(enc, n)
    assert dec[0] == enc["sink"] and dec.shape == (n,)
    err = np.abs(dec - row.astype(np.float64))
    assert err[sorted(kept)].max() <= step / 2 + 1e-6
    assert all(dec[j] == 0 for j in range(1, n) if j not in kept)
    # degenerate rows: one position, everything on the sink, a uniform row
    assert encode_row(np.array([1.0], np.float32)) == {"scale": 0.0, "sink": 1.0, "pos": [], "w": []}
    sink_only = np.zeros(10, np.float32)
    sink_only[0] = 1
    assert encode_row(sink_only) == {"scale": 0.0, "sink": 1.0, "pos": [], "w": []}
    uniform = encode_row(np.full(8, 1 / 8, np.float32))
    assert uniform["pos"] == list(range(1, 8)) and uniform["w"] == [255] * 7 and uniform["scale"] == pytest.approx(0.125)
    assert np.allclose(decode_row(uniform, 8), 1 / 8, atol=1e-6)


def test_head_stats_on_a_hand_built_matrix():
    """One layer, three heads, five positions (the query at 4), tokens [7, 3, 7, 9, 7]: the earlier
    copies of the query's token sit at 0 and 2, so the induction positions are 1 and 3."""
    from superborg.interp import head_stats
    attn = np.array([[[1, 0, 0, 0, 0],            # everything on the sink
                      [0, .5, 0, .5, 0],          # the induction positions only
                      [0, 0, 0, .25, .75]]], np.float32)  # the previous token and itself
    s = head_stats(attn, tokens=[7, 3, 7, 9, 7], n_prompt=3)
    assert list(s) == ["sink", "entropy_norm", "prev", "self", "lookback", "induction"]
    assert s["sink"] == [[1.0, 0.0, 0.0]]
    h2 = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert s["entropy_norm"] == [[0.0, round(np.log(2) / np.log(5), 3), round(h2 / np.log(5), 3)]]
    assert s["prev"] == [[0.0, 0.5, 0.25]]
    assert s["self"] == [[0.0, 0.0, 0.75]]
    assert s["lookback"] == [[1.0, 0.5, 0.0]]   # mass on positions 0..2
    assert s["induction"] == [[0.0, 1.0, 0.25]]
    # no earlier copy of the query's token: induction 0; a one-position row: entropy and prev 0
    assert head_stats(attn, tokens=[7, 3, 7, 9, 8], n_prompt=5)["induction"] == [[0.0, 0.0, 0.0]]
    assert head_stats(np.ones((1, 1, 1), np.float32), [7], 1) == {
        "sink": [[1.0]], "entropy_norm": [[0.0]], "prev": [[0.0]], "self": [[1.0]], "lookback": [[1.0]], "induction": [[0.0]]}


def test_effective_rows_and_attention_report_shapes():
    """The effective row weights each head's row by the value norms of its kv head and sums to 1;
    attn_top and attn_heads skip position 0; the report's shapes follow the model dims."""
    from superborg.interp import attention_report, decode_row, effective_rows
    rng = np.random.default_rng(2)
    L, H, KV, n = 2, 4, 2, 12
    attn = rng.random((L, H, n)).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)
    vnorm = rng.random((L, KV, n)).astype(np.float32) + 0.1
    eff = effective_rows(attn, vnorm, KV)
    assert eff.shape == (L, n) and np.allclose(eff.sum(axis=-1), 1.0)
    want = (attn.astype(np.float64) * vnorm.astype(np.float64)[:, [0, 0, 1, 1], :]).mean(axis=1)
    assert np.allclose(eff, want / want.sum(axis=-1, keepdims=True))
    rep = attention_report(attn, vnorm, list(range(n)), n_prompt=6, n_kv_heads=KV, attn_top=3, head_top=2)
    assert set(rep) == {"attn_rows", "attn_eff", "attn_top", "attn_heads", "head_stats", "attn_entropy"}
    assert len(rep["attn_rows"]) == len(rep["attn_eff"]) == len(rep["attn_top"]) == L
    assert all(len(x) == 3 and all(p >= 1 for p, _ in x) for x in rep["attn_top"])
    assert [p for p, _ in rep["attn_top"][0]] == [int(i) + 1 for i in np.argsort(-eff[0, 1:], kind="stable")[:3]]
    assert all(len(hl) == H and all(len(x) == 2 and all(p >= 1 for p, _ in x) for x in hl) for hl in rep["attn_heads"])
    assert np.allclose(decode_row(rep["attn_eff"][1], n)[0], eff[1, 0], atol=5e-5)
    assert np.allclose(decode_row(rep["attn_rows"][1], n), attn[1].mean(axis=0), atol=attn[1].mean(axis=0)[1:].max() / 255)
    assert np.array(rep["attn_entropy"]).shape == np.array(rep["head_stats"]["sink"]).shape == (L, H)


def test_induction_score_on_a_repeated_sequence(session):
    """A random token sequence repeated: in the second copy some heads attend to the token after
    the earlier copy of the current token, so the induction score is high for the best head (the
    median over positions of the best head's score is 0.98 on smollm2-135m-q8_0) and the mean over
    all 270 heads rises (0.04); in the first copy no token has an earlier copy and the score is 0."""
    from superborg.interp import head_stats
    rng = np.random.default_rng(3)
    seq = [int(t) for t in rng.choice(np.arange(200, 20000), size=40, replace=False)]
    toks = seq * 2
    ctx = session.llm.ctx
    ctx.reset()
    best, mean_second = [], []
    for q in range(10, 80):
        ctx.decode(toks[:q + 1] if q == 10 else [toks[q]], logits=None)
        tr = ctx.trace()
        assert tr["pos"] == q
        ind = np.array(head_stats(tr["attn"], tr["tokens"], n_prompt=80)["induction"])
        if q < 40:
            assert ind.max() == 0.0
        else:
            best.append(ind.max())
            mean_second.append(ind.mean())
    assert np.median(best) > 0.5 and np.mean(mean_second) > 0.02


def test_rival_is_the_runner_up_and_tracked_through_the_lens(session):
    """The rival is the most probable token other than the chosen one: greedy, the second
    alternative (rank 1 at the output layer); sampled, the top token whenever it was not chosen."""
    for ev in session.stream_chat(Q, max_tokens=6):
        if ev["type"] != "token":
            continue
        assert ev["rival"] == ev["alternatives"][1] and ev["rival"][2] != ev["id"]
        out = ev["lens"][-1]
        assert out["rank"] == 0 and out["rival_rank"] == 1
        assert out["rival_p"] == pytest.approx(ev["rival"][1], abs=1e-4) and out["p"] == pytest.approx(ev["prob"], abs=1e-4)
        for x in ev["lens"]:
            assert set(x) == {"layer", "top", "p", "rank", "entropy", "rival_p", "rival_rank"}
            assert x["entropy"] >= 0 and x["rival_rank"] >= 0 and 0 <= x["rival_p"] <= 1
    seen = 0
    for ev in session.stream_chat([{"role": "user", "content": "Tell me a story."}], max_tokens=20, temperature=1.5,
                                  seed=11):
        if ev["type"] == "token" and ev["id"] != ev["alternatives"][0][2]:
            assert ev["rival"] == ev["alternatives"][0] and ev["lens"][-1]["rival_rank"] == 0
            seen += 1
    assert seen > 0


def test_trace_prompt_events_and_cap(session):
    """With trace_prompt the uncached prompt positions before the last are evaluated one per pass,
    each with a prompt_token event (the trace fields, no sampling fields, the lens tracking the
    actual next token), capped at the last 256; the reply is the same as without."""
    from superborg import interp
    session.llm.ctx.reset()
    plain = [e["id"] for e in session.stream_chat(Q, max_tokens=2) if e["type"] == "token"]
    session.llm.ctx.reset()
    evs = list(session.stream_chat(Q, max_tokens=2, trace_prompt=True))
    n = next(e for e in evs if e["type"] == "prompt")["n_prompt"]
    pts = [e for e in evs if e["type"] == "prompt_token"]
    assert [e["type"] for e in evs] == ["prompt_token"] * (n - 1) + ["prompt", "token", "token", "done"]
    assert [e["id"] for e in evs if e["type"] == "token"] == plain
    tok = next(e for e in evs if e["type"] == "token")
    assert set(pts[0]) == (set(tok) - {"i", "text", "raw", "eos", "prob", "entropy", "alternatives"}) | {"next"}
    prompt_ids = session.llm.ctx.tokens[:n]
    for q, e in enumerate(pts):
        assert e["pos"] == q and e["id"] == prompt_ids[q] and e["next"] == prompt_ids[q + 1]
        assert e["decode_ms"] > 0 and e["trace_ms"] >= 0
        assert len(e["lens"]) == session.cfg["n_layers"] + 1 and e["rival"][2] != e["next"]
        assert (e["lens"][-1]["rank"] == 0) == (e["rival"] != e["lens"][-1]["top"][0][:3] and
                                                e["lens"][-1]["top"][0][2] == e["next"])
        assert all(p <= q for enc in e["attn_rows"] for p in enc["pos"])
        assert all(x == 1.0 for row in e["head_stats"]["lookback"] for x in row)  # all positions are prompt
    assert tok["pos"] == n - 1
    # the cap: a long prompt gets exactly 256 prompt_token events, for the positions before the last
    msgs = _long_prompt(30)
    session.llm.ctx.reset()
    evs = list(session.stream_chat(msgs, max_tokens=1, trace_prompt=True))
    n = next(e for e in evs if e["type"] == "prompt")["n_prompt"]
    assert n - 1 > interp.TRACE_PROMPT_MAX + interp.PREFILL_CHUNK
    pts = [e for e in evs if e["type"] == "prompt_token"]
    assert len(pts) == interp.TRACE_PROMPT_MAX and [e["pos"] for e in pts] == list(range(n - 257, n - 1))
    pre = [e for e in evs if e["type"] == "prefill"]
    assert [e["done"] for e in pre] == list(range(interp.PREFILL_CHUNK, n, interp.PREFILL_CHUNK))
    assert session.llm.ctx.n_past == n + 1
    # the cached prefix is reused: only the new turn's positions are traced
    again = list(session.stream_chat(msgs + [{"role": "assistant", "content": "Foxes."},
                                             {"role": "user", "content": "Shorter."}], max_tokens=1, trace_prompt=True))
    p2 = next(e for e in again if e["type"] == "prompt")
    pts2 = [e for e in again if e["type"] == "prompt_token"]
    assert p2["n_reused"] > n // 2 and [e["pos"] for e in pts2] == list(range(p2["n_reused"], p2["n_prompt"] - 1))
    # a stop between traced prompt tokens ends with "prompt" and a cancelled "done"
    session.llm.ctx.reset()
    evs = []
    for ev in session.stream_chat(Q, max_tokens=2, trace_prompt=True,
                                  should_stop=lambda: sum(e["type"] == "prompt_token" for e in evs) >= 3):
        evs.append(ev)
    assert [e["type"] for e in evs] == ["prompt_token"] * 3 + ["prompt", "done"] and evs[-1]["stop_reason"] == "cancelled"
    # the event sizes: a prompt_token event is a token event without the sampling fields
    assert set(evs[0]) < set(tok) | {"next"}


def test_resid_report_and_basis(session):
    """resid_report: norms, the massive-activation spike (max / median |dim|) and, with a basis
    from tools/trace_basis.py, the 3-D projection; the basis tool detects a spike dimension and
    clips it, its components are orthonormal and its explained variances sum to at most 1."""
    import importlib.util
    from superborg.interp import prepare_basis, resid_report, rms_normalize
    spec = importlib.util.spec_from_file_location("trace_basis", os.path.join(ROOT, "tools", "trace_basis.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    rng = np.random.default_rng(4)
    N, D = 300, 32
    base = rng.standard_normal((N, 2)) @ rng.standard_normal((2, D)) * 3 + rng.standard_normal((N, D)) * 0.1
    x = np.stack([base, base * 2, base + 1], axis=1)  # [N, 3 layers, D]
    x[:, 1, 5] = 1.0
    x[::7, 1, 5] = 4000.0  # a massive-activation dimension in layer 1 (the typical |value| is a few units)
    b = tb.fit_basis(x)
    assert set(b) == {"layers", "n_tokens", "clip_ratio"} and b["n_tokens"] == N and len(b["layers"]) == 3
    for lb in b["layers"]:
        assert set(lb) == {"mean", "components", "clip_dims", "clip_to", "explained"}
        comp = np.array(lb["components"])
        assert comp.shape == (3, D) and len(lb["mean"]) == D and len(lb["explained"]) == 3
        assert np.allclose(comp @ comp.T, np.eye(3), atol=1e-8)
        e = lb["explained"]
        assert e[0] >= e[1] >= e[2] >= 0 and sum(e) <= 1 + 1e-9
    assert b["layers"][0]["clip_dims"] == [] and b["layers"][1]["clip_dims"] == [5] and b["layers"][2]["clip_dims"] == []
    assert sum(b["layers"][0]["explained"][:2]) > 0.95  # the planted 2-D structure
    assert b["layers"][1]["clip_to"] == pytest.approx(100 * np.median(np.abs(x[:, 1, :])))  # raw units
    assert b["layers"][1]["clip_to"] < 4000
    # the projection, by hand: clip, RMS-normalize, center, project (a spike position too)
    basis = prepare_basis(b, n_layers=2, dim=D)
    for i in (3, 7):
        rep = resid_report(x[i].astype(np.float32), basis)
        assert set(rep) == {"norm", "spike", "xyz"} and len(rep["xyz"]) == 3 and all(len(p) == 3 for p in rep["xyz"])
        r = x[i].astype(np.float32).astype(np.float64)
        r[1, 5] = np.clip(r[1, 5], -basis["clip_to"][1], basis["clip_to"][1])
        want = np.einsum("lkd,ld->lk", basis["components"], rms_normalize(r) - basis["mean"])
        assert np.allclose(rep["xyz"], want, atol=1e-3)
    assert rep["spike"][1] > 100  # the spike position's raw residual, before the clip
    rep = resid_report(x[3].astype(np.float32), basis)
    assert rep["norm"] == pytest.approx(np.linalg.norm(x[3], axis=-1).tolist(), abs=1e-3)
    ab = np.abs(x[3])
    assert rep["spike"] == pytest.approx((ab.max(axis=-1) / np.median(ab, axis=-1)).tolist(), abs=1e-3)
    assert resid_report(x[3].astype(np.float32))["xyz"] is None
    for bad in ((3, D), (2, D + 1)):
        with pytest.raises(ValueError):
            prepare_basis(b, *bad)
    # on the live session: a basis of the model's shape turns xyz on, None turns it off
    L1, Dm = session.cfg["n_layers"] + 1, session.cfg["dim"]
    session.set_basis(tb.fit_basis(rng.standard_normal((40, L1, Dm))))
    try:
        assert session.info()["basis"] is True
        ev = next(e for e in session.stream_chat(Q, max_tokens=1) if e["type"] == "token")
        assert len(ev["resid"]["xyz"]) == L1 and all(len(p) == 3 for p in ev["resid"]["xyz"])
        assert len(ev["resid"]["norm"]) == len(ev["resid"]["spike"]) == L1 and max(ev["resid"]["spike"]) > 10
        assert ev["resid"]["norm"] == pytest.approx(ev["norms"]["resid"], rel=1e-3)
    finally:
        session.set_basis(None)
    assert session.info()["basis"] is False
    with pytest.raises(ValueError):
        session.set_basis(b)
