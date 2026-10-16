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


# ---- activation tracing ------------------------------------------------------------

def test_trace_matches_reference_and_own_logits(f32):
    from superborg.reference import ReferenceModel
    toks = f32.tokenize("The quick brown fox jumps over the lazy")
    c = f32.context(n_ctx=64)
    c.enable_trace()
    logits = c.decode(toks, "last")
    tr = c.trace()
    cfg = f32.config
    nl = cfg["n_layers"]
    assert tr["pos"] == len(toks) - 1
    assert tr["resid"].shape == (nl + 1, cfg["dim"]) and tr["mlp"].shape == (nl, cfg["hidden_dim"])
    assert tr["attn"].shape == (nl, cfg["n_heads"], len(toks))
    assert np.allclose(tr["attn"].sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(tr["norms"][:, 2], np.linalg.norm(tr["resid"][1:], axis=1), rtol=1e-4)
    lens = c.lens(list(range(nl + 1)))
    assert np.array_equal(lens[nl], logits)  # the last layer's lens is the model's output
    # lens(l) equals the output of the model cut after l layers (NumPy reference)
    ref = ReferenceModel(F32)
    full = list(ref.layers)
    for l in (0, 1, nl // 2, nl):
        ref.layers = full[:l]
        out = ref.forward(toks)[-1]
        assert np.abs(out - lens[l]).max() <= 1e-4 * max(1.0, float(np.abs(out).max())), l
    ref.layers = full
    # incremental decoding traces the newest token
    nxt = int(logits[: f32.n_vocab].argmax())
    c.decode([nxt], "last")
    assert c.trace()["pos"] == len(toks)
    with pytest.raises(ValueError):
        c.lens([nl + 1])
    fresh = f32.context(n_ctx=16)
    assert fresh.trace() is None


def _reference_summary(z: np.ndarray, k: int, chosen: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """top ids (logit descending, id ascending), rank (strictly greater logits) and float64 lse."""
    ids = np.lexsort((np.broadcast_to(np.arange(z.shape[1]), z.shape), -z), axis=1)[:, :k]
    rank = (z > z[:, chosen][:, None]).sum(axis=1)
    z64 = z.astype(np.float64)
    m = z64.max(axis=1)
    return ids, rank, m + np.log(np.exp(z64 - m[:, None]).sum(axis=1))


def test_lens_summary_with_ties(pool, tmp_path):
    """borg_ctx_lens_summary on a copy of smollm2-135m-q8_0 whose output rows a - 7, a + 5 and
    a + 9 are made copies of row a, the most likely token after a prompt: four tokens then tie
    for the top logit of the last layer (and tie at some rank in every lens row). The top list
    orders ties by id, a rank counts strictly greater logits only (ties share it), the chosen
    logit is the kernel's own value and the log-sum-exp is float64-accurate."""
    from superborg.convert import read_borg
    src = need(Q8)
    probe = Model(src, pool=pool)
    toks = probe.tokenize("The capital of France is")
    c = probe.context(n_ctx=64)
    a = int(c.decode(toks, "last")[: probe.n_vocab].argmax())
    c.close()
    probe.close()
    header, mm, data_start = read_borg(src)
    t = header["tensors"]["tok_embd"]  # tied embeddings: the output matrix
    assert t["dtype"] == "q8_0" and "output" not in header["tensors"]
    row_bytes = t["shape"][1] // QK * 34
    copies = [a - 7, a + 5, a + 9]
    assert a - 7 >= 0 and a + 9 < header["config"]["vocab_size"] and not set(copies + [a]) & set(toks)
    data = bytearray(bytes(mm))
    base = data_start + t["offset"]
    for r in copies:
        data[base + r * row_bytes: base + (r + 1) * row_bytes] = data[base + a * row_bytes: base + (a + 1) * row_bytes]
    path = tmp_path / "ties.borg"
    path.write_bytes(data)
    m = Model(str(path), pool=pool)
    c = m.context(n_ctx=64)
    c.enable_trace()
    c.decode(toks, "last")
    layers = list(range(m.config["n_layers"] + 1))
    z = c.lens(layers)[:, : m.n_vocab]
    tied = sorted(copies + [a])
    assert np.all(z[:, tied] == z[:, [a]]) and int(z[-1].argmax()) == tied[0]
    chosen = a + 5
    for k in (2, 4, 6):
        s = c.lens_summary(layers, k, chosen)
        ids, rank, lse = _reference_summary(z, k, chosen)
        assert np.array_equal(s["top_ids"], ids)
        assert np.array_equal(s["top_logits"], np.take_along_axis(z, ids, axis=1))
        assert np.array_equal(s["chosen_rank"], rank) and s["chosen_rank"][-1] == 0
        assert np.array_equal(s["chosen_logit"], z[:, chosen])
        assert np.abs(s["lse"] - lse).max() <= 1e-9
        assert s["top_ids"][-1].tolist()[: min(k, 4)] == tied[: min(k, 4)]  # the tie, in id order
    # rank and top list of a token tied with others in the middle of a row: the same rules
    s = c.lens_summary(layers, 4, a - 7)
    assert np.array_equal(s["chosen_rank"], (z > z[:, a - 7][:, None]).sum(axis=1))
    # every row of the output matrix (vocab_size, here the tokenizer's size) and the errors
    assert np.array_equal(c.lens_summary(layers, 4, chosen, n_vocab=0)["top_ids"], s["top_ids"])
    for kw in ({"k": 0}, {"k": m.n_vocab + 1}, {"chosen": m.n_vocab}, {"chosen": -1}, {"n_vocab": m.vocab_size + 1},
               {"layers": [m.config["n_layers"] + 1]}):
        args = dict(layers=layers, k=4, chosen=chosen)
        args.update(kw)
        with pytest.raises(ValueError):
            c.lens_summary(**args)
    with pytest.raises(ValueError, match="nothing traced"):
        m.context(n_ctx=16).lens_summary(layers, 4, chosen)
    assert c.lens_summary([], 4, chosen)["top_ids"].shape == (0, 4)
    c.close()
    m.close()


def test_lens_summary_vocab_restriction(pool):
    """Qwen2.5's output matrix has 271 rows past the tokenizer's vocabulary: by default they take
    no part; with n_vocab = 0 they do, so ranks and normalizers can only grow."""
    m = Model(need(QWEN), pool=pool)
    assert m.vocab_size > m.n_vocab
    c = m.context(n_ctx=64)
    c.enable_trace()
    c.decode(m.tokenize("Hello"), None)
    layers = [0, m.config["n_layers"]]
    chosen = int(c.lens([m.config["n_layers"]])[0, : m.n_vocab].argmax())
    within = c.lens_summary(layers, 3, chosen)
    every = c.lens_summary(layers, 3, chosen, n_vocab=0)
    assert (within["top_ids"] < m.n_vocab).all()
    assert (every["chosen_rank"] >= within["chosen_rank"]).all() and (every["lse"] >= within["lse"]).all()
    assert np.array_equal(every["chosen_logit"], within["chosen_logit"])
    c.close()
    m.close()
