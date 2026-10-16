"""Tests for batched multi-sequence decoding, KV cache files, JSON mode, activation
statistics and the 4-bit formats. Models come from BORG_MODELS_DIR (skipped if absent)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from superborg import LLM
from superborg.convert import QK, dequantize, encode_tensor
from superborg.engine import Model, Pool, Sampling, generate_multi

MODELS = os.environ.get(
    "BORG_MODELS_DIR", "/tmp/claude-0/-home-user-superborg/990fbcaa-dc72-53ae-9121-cd54b34a4307/scratchpad/borg")
F32 = os.path.join(MODELS, "smollm2-135m-f32.borg")
Q8 = os.path.join(MODELS, "smollm2-135m-q8_0.borg")
QWEN = os.path.join(MODELS, "qwen2.5-0.5b-q8_0.borg")


def need(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not available")
    return path


@pytest.fixture(scope="module")
def pool():
    p = Pool(4)
    yield p
    p.close()


@pytest.fixture(scope="module")
def f32(pool):
    m = Model(need(F32), pool=pool)
    yield m
    m.close()


def test_decode_multi_matches_single(f32):
    prompts = ["The capital of France is", "def fibonacci(n):\n    ", "1, 2, 3, 4, 5,"]
    ids = [f32.tokenize(p) for p in prompts]
    singles, ctxs, nxt = [], [], []
    for t in ids:
        c = f32.context(n_ctx=128)
        tok = int(c.decode(t, "last")[: f32.n_vocab].argmax())
        singles.append(c.decode([tok], "last"))
        c.close()
        c2 = f32.context(n_ctx=128)
        c2.decode(t, None)
        ctxs.append(c2)
        nxt.append(tok)
    import ctypes
    from superborg.engine import lib
    arr = (ctypes.c_void_p * 3)(*[c._c for c in ctxs])
    out = np.zeros((3, f32.vocab_size), np.float32)
    rc = lib().borg_decode_multi(arr, 3, (ctypes.c_int32 * 3)(*nxt), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    assert rc == 0
    for i in range(3):
        assert np.array_equal(out[i], singles[i])  # same arithmetic, positions differ per row
        assert ctxs[i].n_past == len(ids[i]) + 1
    # the same context twice in one batch is rejected
    dup = (ctypes.c_void_p * 2)(ctxs[0]._c, ctxs[0]._c)
    assert lib().borg_decode_multi(dup, 2, (ctypes.c_int32 * 2)(1, 2), None) != 0


def test_generate_multi_greedy_equals_single(f32):
    ids = f32.tokenize("Once upon a time")
    c = f32.context(n_ctx=128)
    single = c.generate(ids, max_tokens=20)
    ctxs = [f32.context(n_ctx=128) for _ in range(3)]
    multi = generate_multi(ctxs, ids, max_tokens=20, sampling=Sampling(temperature=0))
    assert all(g.tokens == single.tokens and g.text == single.text for g in multi)


def test_generate_multi_seeds_and_stops(f32):
    ids = f32.tokenize("My favourite animal is the")
    ctxs = [f32.context(n_ctx=128) for _ in range(4)]
    a = generate_multi(ctxs, ids, max_tokens=12, sampling=Sampling(temperature=1.0, seed=7))
    b = generate_multi(ctxs, ids, max_tokens=12, sampling=Sampling(temperature=1.0, seed=7))
    assert [g.tokens for g in a] == [g.tokens for g in b]  # deterministic per seed
    assert len({tuple(g.tokens) for g in a}) > 1  # sequences use different seeds
    c = generate_multi(ctxs, ids, max_tokens=30, sampling=Sampling(temperature=0), stop=["."])
    for g in c:
        assert "." not in g.text
        assert g.stop_reason in ("stop", "eos", "length")


def test_llm_sample_n(pool):
    llm = LLM(need(Q8), pool=pool, n_ctx=256)
    outs = llm.sample_n(llm.tokenize("Hello"), 5, max_tokens=8, temperature=0.9, seed=3)
    assert len(outs) == 5 and all(len(o.tokens) <= 8 for o in outs)
    again = llm.sample_n(llm.tokenize("Hello"), 5, max_tokens=8, temperature=0.9, seed=3)
    assert [o.tokens for o in outs] == [o.tokens for o in again]


def test_kv_save_load(f32, tmp_path):
    ids = f32.tokenize("A system prompt that we want to cache across runs. " * 5)
    c = f32.context(n_ctx=256)
    c.decode(ids, None)
    ref = c.decode(f32.tokenize(" Next"), "last")
    c.truncate(len(ids))
    path = str(tmp_path / "kv.bin")
    c.save(path)
    c2 = f32.context(n_ctx=256)
    c2.load(path)
    assert c2.tokens == ids
    assert np.array_equal(c2.decode(f32.tokenize(" Next"), "last"), ref)
    with pytest.raises(ValueError):
        f32.context(n_ctx=16).load(path)  # does not fit
    with open(path, "rb") as f:
        data = f.read()
    (tmp_path / "bad.bin").write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError):
        c2.load(str(tmp_path / "bad.bin"))
    assert c2.n_past == 0  # a failed load leaves an empty cache, never a partial one


def test_kv_load_rejects_other_model(f32, pool, tmp_path):
    q = Model(need(Q8), pool=pool)
    c = f32.context(n_ctx=64)
    c.decode(f32.tokenize("hello"), None)
    c.save(str(tmp_path / "kv.bin"))
    with pytest.raises(ValueError):
        q.context(n_ctx=64).load(str(tmp_path / "kv.bin"))


def test_json_mode(pool):
    llm = LLM(need(QWEN), pool=pool, n_ctx=512)
    ids = llm.tokenize(llm.render([{"role": "user", "content": "Give a JSON object with keys city and country "
                                                             "for the capital of Japan."}]), add_bos=False)
    g = llm.ctx.generate(ids, max_tokens=60, json_mode=True)
    obj = json.loads(g.text)
    assert isinstance(obj, (dict, list))
    ctxs = [llm.ctx] + [llm.model.context(n_ctx=512) for _ in range(2)]
    for out in generate_multi(ctxs, ids, max_tokens=60, sampling=Sampling(temperature=1.0, seed=1), json_mode=True):
        if out.stop_reason == "eos":
            json.loads(out.text)


def test_activation_stats(f32):
    c = f32.context(n_ctx=128)
    c.enable_stats()
    toks = f32.tokenize("Calibration text for the statistics hook.")
    c.decode(toks, None)
    assert c.stats_tokens() == len(toks)
    cfg = f32.config
    widths = [cfg["dim"], cfg["n_heads"] * cfg["head_dim"], cfg["dim"], cfg["hidden_dim"]]
    for site, width in enumerate(widths):
        s = c.stats(cfg["n_layers"] - 1, site)
        assert s.shape == (width,) and (s >= 0).all() and s.sum() > 0
    with pytest.raises(ValueError):
        c.stats(cfg["n_layers"], 0)
    c.enable_stats(False)
    with pytest.raises(ValueError):
        c.stats(0, 0)


@pytest.mark.parametrize("qtype", ["q8_0", "q4_0", "q4_1"])
def test_quantizers_round_trip(qtype):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 4 * QK)).astype(np.float32)
    x[3, 5] = 9.0  # an outlier
    dt, blob = encode_tensor(x, qtype)
    y = dequantize(dt, blob, list(x.shape))
    rel = np.abs(y - x).max() / np.abs(x).max()
    assert rel < {"q8_0": 0.01, "q4_0": 0.15, "q4_1": 0.15}[qtype]
    # importance that varies inside each block moves error onto the less important columns
    if qtype != "q8_0":
        imp = np.ones(x.shape[1], np.float32)
        imp[::2] = 100.0
        _, b2 = encode_tensor(x, qtype, imp)
        y2 = dequantize(qtype, b2, list(x.shape))
        err_plain = ((y - x) ** 2 * imp).sum()
        err_weighted = ((y2 - x) ** 2 * imp).sum()
        assert err_weighted < 0.95 * err_plain


# ---- regressions from the engine review ----------------------------------------------

def test_min_p_above_one_keeps_the_best_token(f32):
    ids = f32.tokenize("The capital of France is")
    greedy = f32.context(n_ctx=64).generate(ids, max_tokens=8)
    capped = f32.context(n_ctx=64).generate(ids, max_tokens=8, sampling=Sampling(temperature=0.8, min_p=1.5))
    assert capped.tokens == greedy.tokens
    ctxs = [f32.context(n_ctx=64) for _ in range(2)]
    for g in generate_multi(ctxs, ids, max_tokens=8, sampling=Sampling(temperature=0.8, min_p=1.5)):
        assert g.tokens == greedy.tokens


def test_json_mode_with_ignore_eos(pool):
    llm = LLM(need(QWEN), pool=pool, n_ctx=512)
    ids = llm.tokenize(llm.render([{"role": "user", "content": "Reply with the JSON array [1, 2, 3]."}]),
                       add_bos=False)
    g = llm.ctx.generate(ids, max_tokens=40, json_mode=True, ignore_eos=True)
    value, _, rest = g.text.partition("<|")
    json.loads(value)  # the value stays valid; only end-of-generation markers follow it
    assert rest == "" or set(rest.replace("im_end|>", "").replace("endoftext|>", "").split("<|")) <= {""}


def test_decode_multi_sets_each_embedding(f32):
    import ctypes
    from superborg.engine import lib
    texts = ["Hello there", "Numbers: 1 2 3"]
    ctxs = [f32.context(n_ctx=64) for _ in texts]
    for c, t in zip(ctxs, texts):
        c.decode(f32.tokenize(t), None)
    arr = (ctypes.c_void_p * 2)(*[c._c for c in ctxs])
    assert lib().borg_decode_multi(arr, 2, (ctypes.c_int32 * 2)(5, 6), None) == 0
    for c, t, tok in zip(ctxs, texts, (5, 6)):
        ref = f32.context(n_ctx=64)
        ref.decode(f32.tokenize(t) + [tok], None)
        assert np.allclose(c.embedding(), ref.embedding(), atol=1e-5)


def test_generate_multi_rejects_small_batches(f32):
    ids = f32.tokenize("hi")
    ctxs = [f32.context(n_ctx=64, n_batch=8)] + [f32.context(n_ctx=64, n_batch=2) for _ in range(3)]
    with pytest.raises(RuntimeError):
        generate_multi(ctxs, ids, max_tokens=4)
