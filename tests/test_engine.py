"""Tests of the Python binding (superborg.engine / superborg.LLM) against the real engine.

Models are looked up in $BORG_MODELS_DIR, ./models, ~/.cache/superborg and the default
directory of tools/hf_oracle.py; every test skips when its model file is missing.
Threads: $BORG_TEST_THREADS (default 4).

The speculative-decoding tests check the two guarantees in engine/generate.c:
greedy speculative output is token-for-token the plain greedy output, and sampled
speculative output has exactly the target model's distribution (chi-square test of the
first two tokens over many seeds against the distribution computed from the logits).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

THREADS = int(os.environ.get("BORG_TEST_THREADS", "4"))


def _model_dirs() -> list[str]:
    dirs = [os.environ.get("BORG_MODELS_DIR", ""), os.path.join(ROOT, "models"),
            os.path.expanduser("~/.cache/superborg")]
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import hf_oracle  # noqa: PLC0415 - optional: knows this checkout's default model directory

        dirs.append(hf_oracle.BORG_MODELS_DIR)
    except Exception:  # noqa: BLE001
        pass
    return [d for d in dirs if d and os.path.isdir(d)]


def model_path(name: str) -> str:
    for d in _model_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    pytest.skip(f"{name} not found (set BORG_MODELS_DIR)")


def _engine():
    try:
        from superborg import engine
        engine.lib()
    except (OSError, FileNotFoundError) as e:
        pytest.skip(f"engine library unavailable: {e}")
    return engine


@pytest.fixture(scope="module")
def pool():
    eng = _engine()
    p = eng.Pool(THREADS)
    yield p
    p.close()


@pytest.fixture(scope="module")
def smol(pool):
    eng = _engine()
    m = eng.Model(model_path("smollm2-135m-q8_0.borg"), pool=pool)
    yield m
    m.close()


@pytest.fixture(scope="module")
def qwen(pool):
    eng = _engine()
    m = eng.Model(model_path("qwen2.5-0.5b-q8_0.borg"), pool=pool)
    yield m
    m.close()


@pytest.fixture(scope="module")
def qwen_ctx(qwen):
    c = qwen.context(n_ctx=512)
    yield c
    c.close()


def greedy():
    from superborg import Sampling
    return Sampling(temperature=0.0)


# ---------------------------------------------------------------------------
# tokenizer

TEXTS = [
    "Hello, world!",
    "  leading spaces and trailing  ",
    "Numbers 1234567 and 3.14159, symbols #@$%^&*()",
    "Unicode: café, naïve, 日本語のテキスト, Привет, emoji 🎉🚀",
    "def f(x):\n    return x * 2\n\n\nprint(f(21))",
    "tabs\tand\r\nwindows newlines",
]


@pytest.mark.parametrize("name", ["smollm2-135m-q8_0.borg", "qwen2.5-0.5b-q8_0.borg", "tinyllama-1.1b-q8_0.borg"])
def test_tokenize_detokenize_roundtrip(pool, name):
    eng = _engine()
    m = eng.Model(model_path(name), pool=pool)
    try:
        for text in TEXTS:
            ids = m.tokenize(text, add_bos=False)
            assert ids and all(0 <= t < m.n_vocab for t in ids)
            assert m.detokenize(ids) == text, (name, text)
            # piece bytes concatenate to the same text (what streaming relies on); SentencePiece
            # pieces keep the prefix space that detokenize() drops at the start, like HF decode
            joined = b"".join(m.token_bytes(t) for t in ids).decode("utf-8")
            spm = not m.header["tokenizer"]["byte_level"]
            assert joined == text or (spm and joined == " " + text), (name, joined, text)
        bos = m.header["tokenizer"]["bos_id"]
        with_bos = m.tokenize("Hi", add_bos=True)
        if bos is not None and bos >= 0:
            assert with_bos[0] == bos
            assert m.detokenize(with_bos, skip_special=True) == "Hi"
        assert m.tokenize("", add_bos=False) == []
        assert m.detokenize([]) == ""
    finally:
        m.close()


def test_special_tokens(qwen):
    im_start = qwen.token_id("<|im_start|>")
    assert im_start is not None
    ids = qwen.tokenize("<|im_start|>user", add_bos=False, special=True)
    assert ids[0] == im_start
    plain = qwen.tokenize("<|im_start|>user", add_bos=False, special=False)
    assert im_start not in plain and len(plain) > 2
    assert qwen.detokenize(plain) == "<|im_start|>user"
    eos = qwen.header["tokenizer"]["eos_id"]
    assert qwen.is_eog(eos)
    assert not qwen.is_eog(qwen.tokenize("hello", add_bos=False)[0])


# ---------------------------------------------------------------------------
# decode

def test_decode_logits_shapes(smol):
    ctx = smol.context(n_ctx=64)
    try:
        toks = smol.tokenize("The quick brown fox", add_bos=False)
        assert ctx.decode(toks[:1], logits=None) is None
        assert ctx.n_past == 1
        last = ctx.decode(toks[1:2], logits="last")
        assert last.shape == (smol.vocab_size,) and last.dtype == np.float32
        allv = ctx.decode(toks[2:], logits="all")
        assert allv.shape == (len(toks) - 2, smol.vocab_size)
        assert np.isfinite(last).all() and np.isfinite(allv).all()
        assert ctx.n_past == len(toks) and ctx.tokens == toks
        emb = ctx.embedding()
        assert emb.shape == (smol.config["dim"],) and np.isfinite(emb).all()
        assert ctx.decode([], logits="last") is None
        with pytest.raises(RuntimeError, match="out of range"):
            ctx.decode([smol.vocab_size + 5], logits="last")
    finally:
        ctx.close()


@pytest.mark.parametrize("name", ["smollm2-135m-q8_0.borg", "smollm2-135m-q4_0.borg", "smollm2-135m-f32.borg",
                                  "qwen2.5-0.5b-q8_0.borg"])
def test_incremental_equals_batched(pool, name):
    """Logits of a token must not depend on how the sequence was split into decode calls.

    Exact agreement matters: speculative verification evaluates drafts in a batch and must
    reproduce token-by-token greedy decoding, and cached prefixes must not change results.
    """
    eng = _engine()
    m = eng.Model(model_path(name), pool=pool)
    ctxs = []
    try:
        toks = m.tokenize("Speculative decoding verifies several draft tokens in one batched pass.",
                          add_bos=False)
        ctxs = [m.context(n_ctx=128), m.context(n_ctx=128), m.context(n_ctx=128, n_batch=3)]
        batched = ctxs[0].decode(toks, logits="all")
        inc = np.stack([ctxs[1].decode([t], logits="last") for t in toks])
        chunked = ctxs[2].decode(toks, logits="all")  # prefill split into chunks of 3
        scale = float(np.abs(batched).max())
        worst = max(float(np.abs(x - batched).max()) for x in (inc, chunked))
        same_argmax = all((x.argmax(axis=1) == batched.argmax(axis=1)).all() for x in (inc, chunked))
        assert worst <= 1e-4 * scale, f"max |batched - incremental| = {worst}"
        assert same_argmax
    finally:
        for c in ctxs:
            c.close()
        m.close()


@pytest.mark.parametrize("name", ["smollm2-135m-q8_0.borg", "smollm2-135m-q4_0.borg"])
def test_repeated_request_same_greedy_output(pool, name):
    """Sending the same prompt twice must give the same greedy text.

    The second call finds the prompt in the cache and re-evaluates only its last token; a
    4-token prompt is otherwise evaluated by the 4-token kernels, which must sum in the same
    order as the 1-token kernels.
    """
    eng = _engine()
    m = eng.Model(model_path(name), pool=pool)
    ctx = m.context(n_ctx=128)
    try:
        prompt = m.tokenize("Once upon a time", add_bos=False)
        first = ctx.generate(prompt, max_tokens=12, sampling=greedy())
        again = ctx.generate(prompt, max_tokens=12, sampling=greedy())
        assert again.n_reused == len(prompt) - 1
        assert first.tokens == again.tokens, (first.text, again.text)
    finally:
        ctx.close()
        m.close()


def test_sync_and_prefix_reuse(qwen):
    ctx = qwen.context(n_ctx=256)
    try:
        toks = qwen.tokenize("One two three four five six seven eight nine ten", add_bos=False)
        assert ctx.sync(toks) == 0 and ctx.tokens == toks
        assert ctx.sync(toks + toks[:2]) == len(toks)
        assert ctx.n_past == len(toks) + 2
        other = toks[:4] + qwen.tokenize(" zebra", add_bos=False)
        assert ctx.sync(other) == 4 and ctx.tokens == other
        ctx.truncate(2)
        assert ctx.n_past == 2
        ctx.reset()
        assert ctx.n_past == 0

        prompt = qwen.tokenize("The three primary colors are", add_bos=False)
        g1 = ctx.generate(prompt, max_tokens=8, sampling=greedy())
        assert g1.n_reused == 0 and g1.n_prompt == len(prompt)
        g2 = ctx.generate(prompt, max_tokens=8, sampling=greedy())
        # the last prompt token is re-evaluated to obtain its logits
        assert g2.n_reused == len(prompt) - 1 and g2.n_prompt == 1
        assert g2.tokens == g1.tokens
        # a longer prompt sharing the prefix reuses it too
        g3 = ctx.generate(prompt + g1.tokens[:3], max_tokens=4, sampling=greedy())
        assert g3.n_reused >= len(prompt)
        assert g3.tokens == g1.tokens[3:7]
    finally:
        ctx.close()


def make_llm(path: str, **kw):
    # LLM.close() also frees its pool, so each LLM gets its own instead of the shared fixture
    from superborg import LLM
    return LLM(path, threads=THREADS, **kw)


def test_chat_turns_reuse_cache():
    llm = make_llm(model_path("qwen2.5-0.5b-q8_0.borg"), n_ctx=512)
    try:
        msgs = [{"role": "user", "content": "Name a fruit."}]
        g1 = llm.chat(msgs, max_tokens=12, temperature=0.0)
        assert g1.text and g1.n_reused == 0
        msgs += [{"role": "assistant", "content": g1.text}, {"role": "user", "content": "Another one?"}]
        g2 = llm.chat(msgs, max_tokens=12, temperature=0.0)
        assert g2.n_reused > 10  # system prompt and first turn come from the cache
        assert g2.n_prompt < g2.n_reused
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# generate

def test_generate_seeds(qwen_ctx, qwen):
    from superborg import Sampling
    prompt = qwen.tokenize("Write a random word:", add_bos=False)
    runs = {}
    for seed in (1, 2, 3, 4, 5):
        s = Sampling(temperature=1.0, seed=seed)
        a = qwen_ctx.generate(prompt, max_tokens=8, sampling=s)
        b = qwen_ctx.generate(prompt, max_tokens=8, sampling=s)
        assert a.tokens == b.tokens, "same seed must reproduce the same tokens"
        runs[seed] = tuple(a.tokens)
    assert len(set(runs.values())) > 1, "different seeds should give different samples at temperature 1"
    g1 = qwen_ctx.generate(prompt, max_tokens=8, sampling=greedy())
    g2 = qwen_ctx.generate(prompt, max_tokens=8, sampling=Sampling(temperature=0.0, seed=99))
    assert g1.tokens == g2.tokens, "greedy decoding ignores the seed"


def test_max_tokens_zero_and_eos(qwen_ctx, qwen):
    prompt = qwen.tokenize("Hello", add_bos=False)
    g = qwen_ctx.generate(prompt, max_tokens=0, sampling=greedy())
    assert g.tokens == [] and g.text == "" and g.stop_reason == "length"
    eos = qwen.header["tokenizer"]["eos_id"]
    # a token listed in stop_tokens ends generation like end-of-sequence and is not emitted
    first = qwen_ctx.generate(prompt, max_tokens=1, sampling=greedy()).tokens[0]
    g = qwen_ctx.generate(prompt, max_tokens=8, sampling=greedy(), stop_tokens=[first])
    assert g.tokens == [] and g.stop_reason == "eos"
    assert eos not in qwen_ctx.generate(prompt, max_tokens=8, sampling=greedy()).tokens


def test_stop_strings(qwen_ctx, qwen):
    prompt = qwen.tokenize("Count: 1, 2, 3, 4,", add_bos=False)
    ref = qwen_ctx.generate(prompt, max_tokens=24, sampling=greedy())
    assert len(ref.text) > 16, ref.text
    for lo, hi in ((6, 9), (3, 12), (len(ref.text) // 2, len(ref.text) // 2 + 1)):
        stop = ref.text[lo:hi]
        cut = ref.text.find(stop)
        pieces: list[str] = []
        g = qwen_ctx.generate(prompt, max_tokens=24, sampling=greedy(), stop=[stop, "@@never@@"],
                              stream=pieces.append)
        assert g.stop_reason == "stop"
        assert g.text == ref.text[:cut], (stop, g.text)
        assert stop not in g.text
        assert "".join(pieces) == g.text, "the stream must not leak the stop string"
        assert g.tokens == ref.tokens[:len(g.tokens)]
    g = qwen_ctx.generate(prompt, max_tokens=24, sampling=greedy(), stop=["@@never@@"])
    assert g.text == ref.text and g.stop_reason == ref.stop_reason


def test_streaming_concatenates(qwen_ctx, qwen):
    from superborg import Sampling
    for text in ("The capital of Japan is", "日本の首都は", "Emoji party: 🎉"):
        prompt = qwen.tokenize(text, add_bos=False)
        pieces: list[str] = []
        g = qwen_ctx.generate(prompt, max_tokens=20, sampling=Sampling(temperature=0.7, seed=3),
                              stream=pieces.append)
        assert "".join(pieces) == g.text
        assert g.text == qwen.detokenize(g.tokens)
        assert all(p for p in pieces)


def test_callback_exception_stops_generation(qwen_ctx, qwen):
    prompt = qwen.tokenize("Once upon a time", add_bos=False)
    seen = []

    def stream(piece):
        seen.append(piece)
        if len(seen) == 3:
            raise KeyError("stop now")

    g = qwen_ctx.generate(prompt, max_tokens=30, sampling=greedy(), stream=stream)
    assert g.stop_reason == "stop" and len(g.tokens) <= 4


def test_context_full(qwen):
    from superborg import Sampling
    ctx = qwen.context(n_ctx=32)
    try:
        assert ctx.n_ctx == 32
        prompt = qwen.tokenize("one two three four five six seven eight nine ten eleven twelve", add_bos=False)
        assert len(prompt) < 32
        g = ctx.generate(prompt, max_tokens=100, sampling=greedy(), ignore_eos=True)
        assert g.stop_reason == "context"
        assert 32 - len(prompt) <= len(g.tokens) <= 32 - len(prompt) + 1
        assert ctx.n_past <= 32
        with pytest.raises(RuntimeError, match="full"):
            ctx.decode(qwen.tokenize(" a b c d e f g h i j k l m n o p q r s t u v w x y z", add_bos=False) * 2,
                       logits="last")
        with pytest.raises(ValueError):
            ctx.generate(list(range(100, 140)), max_tokens=4)
        with pytest.raises(ValueError):
            ctx.generate([], max_tokens=4)
        # the context is still usable afterwards
        ctx.reset()
        assert ctx.generate(prompt, max_tokens=3, sampling=Sampling()).tokens
    finally:
        ctx.close()
    llm = make_llm(qwen.path, n_ctx=32)
    try:
        g = llm.complete("one two three four five six seven eight nine ten", max_tokens=100, temperature=0.0)
        assert g.stop_reason in ("length", "eos") and len(g.tokens) <= 32
        with pytest.raises(ValueError):
            llm.complete("word " * 64, max_tokens=4)
    finally:
        llm.close()


def test_score_matches_logits(qwen):
    ctx = qwen.context(n_ctx=128)
    ref = qwen.context(n_ctx=128)
    try:
        prompt = qwen.tokenize("The capital of France is", add_bos=False)
        cont = qwen.tokenize(" Paris, a city on the Seine.", add_bos=False)
        lp = ctx.score(prompt, cont)
        assert lp.shape == (len(cont),) and (lp <= 0).all()
        logits = ref.decode(prompt + cont[:-1], logits="all")[len(prompt) - 1:, : qwen.n_vocab].astype(np.float64)
        ls = logits - logits.max(axis=1, keepdims=True)
        ls -= np.log(np.exp(ls).sum(axis=1, keepdims=True))
        expect = ls[np.arange(len(cont)), cont]
        np.testing.assert_allclose(lp, expect, atol=2e-3)
        # scoring again reuses the cache and gives the same numbers
        np.testing.assert_allclose(ctx.score(prompt, cont), lp, atol=1e-4)
    finally:
        ctx.close()
        ref.close()
    llm = make_llm(qwen.path, n_ctx=128)
    try:
        good = llm.score("The capital of France is", " Paris")
        bad = llm.score("The capital of France is", " banana")
        assert good > bad
    finally:
        llm.close()


def test_kv_f16_same_greedy_text(qwen):
    prompt = qwen.tokenize("Q: What is the boiling point of water at sea level?\nA:", add_bos=False)
    out = []
    for f16 in (False, True):
        ctx = qwen.context(n_ctx=128, kv_f16=f16)
        try:
            out.append(ctx.generate(prompt, max_tokens=20, sampling=greedy()).text)
        finally:
            ctx.close()
    assert out[0] == out[1]


# ---------------------------------------------------------------------------
# KV cache persistence and copies

def test_kv_cache_save_load_copy(qwen, smol, tmp_path):
    eng = _engine()
    if not hasattr(eng.Context, "save"):
        pytest.skip("binding has no KV cache save/load")
    a, b, c = (qwen.context(n_ctx=128) for _ in range(3))
    try:
        prompt = qwen.tokenize("The quick brown fox jumps over the lazy dog because", add_bos=False)
        a.sync(prompt)
        path = str(tmp_path / "cache.kv")
        a.save(path)
        b.load(path)
        assert b.tokens == prompt and b.n_past == len(prompt)
        nxt = qwen.tokenize(" it", add_bos=False)
        la = a.decode(nxt, logits="last")
        np.testing.assert_array_equal(b.decode(nxt, logits="last"), la)
        c.copy_from(a)
        assert c.tokens == a.tokens
        more = qwen.tokenize(" was", add_bos=False)
        np.testing.assert_array_equal(c.decode(more, logits="last"), a.decode(more, logits="last"))
        # a restored cache serves as the prefix of a later generation
        g = b.generate(prompt + nxt, max_tokens=4, sampling=greedy())
        assert g.n_reused == len(prompt)
        # incompatible restores fail cleanly and leave the target usable
        small = qwen.context(n_ctx=8)
        other = smol.context(n_ctx=128)
        try:
            with pytest.raises(ValueError):
                small.load(path)
            with pytest.raises(ValueError):
                other.load(path)
            with pytest.raises(ValueError):
                other.copy_from(a)
            data = open(path, "rb").read()
            bad = tmp_path / "truncated.kv"
            bad.write_bytes(data[: len(data) // 2])
            with pytest.raises(ValueError):
                c.load(str(bad))
            with pytest.raises(ValueError):
                c.load(str(tmp_path / "missing.kv"))
            assert other.generate(smol.tokenize("Hi", add_bos=False), max_tokens=2, sampling=greedy()).tokens
        finally:
            small.close()
            other.close()
    finally:
        for x in (a, b, c):
            x.close()


# ---------------------------------------------------------------------------
# batched multi-sequence sampling

@pytest.mark.parametrize("name", ["smollm2-135m-q4_0.borg", "smollm2-135m-q8_0.borg"])
def test_generate_multi(pool, name):
    """Batched multi-sequence sampling: greedy sequences equal single greedy generation;
    sampled sequences are reproducible per seed, differ from each other, and stay inside
    the top-k support.

    (Sequence i is not token-identical to generate() with seed + i: generate() draws from
    the probability vector in vocabulary order, generate_multi from the candidate list, so
    the same random number picks different tokens; both follow the same distribution.)
    """
    eng = _engine()
    if not hasattr(eng, "generate_multi"):
        pytest.skip("binding has no generate_multi")
    from superborg import Sampling
    m = eng.Model(model_path(name), pool=pool)
    ctxs = [m.context(n_ctx=128) for _ in range(4)]
    single = m.context(n_ctx=128)
    try:
        prompt = m.tokenize("Here is a list of fruits: apple,", add_bos=False)
        pieces: dict[int, list[str]] = {i: [] for i in range(4)}
        multi = eng.generate_multi(ctxs, prompt, max_tokens=12, sampling=Sampling(temperature=0.0),
                                   stream=lambda i, t: pieces[i].append(t))
        assert len(multi) == 4
        ref = single.generate(prompt, max_tokens=12, sampling=greedy())
        for i, g in enumerate(multi):
            assert "".join(pieces[i]) == g.text
            assert g.tokens == ref.tokens and g.text == ref.text, (i, g.text, ref.text)

        samp = Sampling(temperature=0.9, top_k=5, seed=100)
        a = eng.generate_multi(ctxs, prompt, max_tokens=6, sampling=samp)
        b = eng.generate_multi(ctxs, prompt, max_tokens=6, sampling=samp)
        assert [g.tokens for g in a] == [g.tokens for g in b], "same seed, same samples"
        seqs = []
        for seed in range(0, 40, 4):
            seqs += [tuple(g.tokens) for g in eng.generate_multi(
                ctxs, prompt, max_tokens=4, sampling=Sampling(temperature=0.9, top_k=5, seed=seed))]
        single.reset()
        single.sync(prompt)
        single.truncate(len(prompt) - 1)
        top5 = set(np.argsort(-single.decode(prompt[-1:], logits="last")[: m.n_vocab])[:5].tolist())
        assert {t[0] for t in seqs if t} <= top5
        assert len(set(seqs)) > 1

        # stop strings apply per sequence
        multi = eng.generate_multi(ctxs, prompt, max_tokens=12, sampling=Sampling(temperature=0.0), stop=[","])
        assert all("," not in g.text and g.stop_reason == "stop" for g in multi)
    finally:
        for c in ctxs + [single]:
            c.close()
        m.close()


# ---------------------------------------------------------------------------
# speculative decoding: greedy equivalence

LOOKUP_PROMPT = """def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def"""


def test_lookup_greedy_identical(qwen_ctx, qwen):
    prompt = qwen.tokenize(LOOKUP_PROMPT, add_bos=False)
    plain = qwen_ctx.generate(prompt, max_tokens=48, sampling=greedy())
    qwen_ctx.reset()
    spec = qwen_ctx.generate(prompt, max_tokens=48, sampling=greedy(), lookup_ngram=3, n_draft=6)
    assert spec.n_drafted > 0 and spec.n_accepted > 0
    assert spec.n_target_evals < len(spec.tokens), "accepted drafts must save target passes"
    assert spec.tokens == plain.tokens
    assert spec.text == plain.text


def test_draft_model_greedy_identical(pool):
    eng = _engine()
    target = eng.Model(model_path("qwen2.5-1.5b-q8_0.borg"), pool=pool)
    draft = eng.Model(model_path("qwen2.5-0.5b-q8_0.borg"), pool=pool)
    tc, dc = target.context(n_ctx=256), draft.context(n_ctx=256)
    try:
        prompt = target.tokenize("<|im_start|>user\nList the planets of the solar system in order.<|im_end|>\n"
                                 "<|im_start|>assistant\n", add_bos=False)
        plain = tc.generate(prompt, max_tokens=40, sampling=greedy())
        tc.reset()
        spec = tc.generate(prompt, max_tokens=40, sampling=greedy(), draft=dc, n_draft=5)
        assert spec.n_drafted > 0 and spec.n_accepted > 0
        assert spec.tokens == plain.tokens, (plain.text, spec.text)
        # a second request reuses both caches and stays identical
        spec2 = tc.generate(prompt, max_tokens=40, sampling=greedy(), draft=dc, n_draft=3)
        assert spec2.tokens == plain.tokens
    finally:
        for o in (tc, dc, target, draft):
            o.close()


# ---------------------------------------------------------------------------
# speculative decoding: distribution equivalence

def chi2_sf(x: float, k: int) -> float:
    """Upper tail of the chi-square distribution (Wilson-Hilferty approximation)."""
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chi2_test(counts: dict, probs: dict, n: int) -> tuple[float, int, float]:
    """Goodness of fit of observed counts to probabilities; bins expecting < 5 are pooled."""
    unexpected = sum(c for key, c in counts.items() if probs.get(key, 0.0) <= 0.0)
    assert unexpected == 0, f"sampled outcomes outside the target distribution: {counts} vs {probs}"
    stat, df, pool_e, pool_o = 0.0, -1, 0.0, 0
    for key, p in probs.items():
        e, o = n * p, counts.get(key, 0)
        if e < 5:
            pool_e += e
            pool_o += o
            continue
        stat += (o - e) ** 2 / e
        df += 1
    if pool_e > 0:
        stat += (pool_o - pool_e) ** 2 / max(pool_e, 1e-9)
        df += 1
    return stat, df, chi2_sf(stat, df)


EOG = -1


def topk_probs(logits: np.ndarray, n_vocab: int, temperature: float, top_k: int) -> dict[int, float]:
    x = logits[:n_vocab].astype(np.float64) / temperature
    idx = np.argpartition(-x, top_k)[:top_k]
    p = np.exp(x[idx] - x[idx].max())
    p /= p.sum()
    return {int(i): float(v) for i, v in zip(idx, p)}


def exact_two_token_dist(model, ctx, prompt: list[int], temperature: float, top_k: int) -> dict:
    """P(first two generated tokens) from the target's logits; EOG stands for an end token.

    The logits are computed the way generate() computes them on a warm cache: the prompt
    prefilled in one chunk, its last token re-evaluated alone, then one token at a time
    (batched and single-token kernels give bit-identical results, so any batching would do).
    """
    ctx.reset()
    ctx.decode(prompt, logits=None)
    ctx.truncate(len(prompt) - 1)
    p1 = topk_probs(ctx.decode(prompt[-1:], logits="last"), model.n_vocab, temperature, top_k)
    dist: dict = {}
    for t1, a in p1.items():
        if model.is_eog(t1):
            dist[(EOG, EOG)] = dist.get((EOG, EOG), 0.0) + a
            continue
        ctx.truncate(len(prompt))
        p2 = topk_probs(ctx.decode([t1], logits="last"), model.n_vocab, temperature, top_k)
        for t2, b in p2.items():
            key = (t1, EOG if model.is_eog(t2) else t2)
            dist[key] = dist.get(key, 0.0) + a * b
    return dist


def sample_counts(ctx, prompt, n: int, temperature: float, top_k: int, **spec) -> tuple[dict, dict]:
    from superborg import Sampling
    counts: dict = {}
    stats = {"drafted": 0, "accepted": 0}
    # with max_tokens=3 the first speculative step drafts 2 tokens, so both the acceptance
    # of a draft after an accepted draft and the bonus token are exercised
    max_tokens = 3 if spec else 2
    ctx.reset()
    ctx.sync(prompt)  # warm cache: every call re-evaluates only the last prompt token
    for seed in range(n):
        g = ctx.generate(prompt, max_tokens=max_tokens, sampling=Sampling(temperature=temperature, top_k=top_k,
                                                                          seed=1000 + seed), **spec)
        t = g.tokens + [EOG, EOG]
        key = (t[0], t[1])
        counts[key] = counts.get(key, 0) + 1
        stats["drafted"] += g.n_drafted
        stats["accepted"] += g.n_accepted
    return counts, stats


N_DIST = int(os.environ.get("BORG_TEST_DIST_SAMPLES", "200"))
N_DRAFT = 5  # verification batches of up to 6 tokens exercise the 4-token kernels


@pytest.mark.parametrize("mode", ["plain", "draft", "lookup"])
def test_speculative_sampling_distribution(pool, mode):
    """Speculative sampling must leave the target distribution unchanged (Leviathan et al. 2023).

    The draft is the same model at q4_0, so its proposals are good but not exact and both the
    acceptance and the residual-resampling branches run. "plain" checks the test itself.
    """
    eng = _engine()
    target = eng.Model(model_path("qwen2.5-0.5b-q8_0.borg"), pool=pool)
    contexts = [target.context(n_ctx=128)]
    T, K = 1.0, 4
    try:
        if mode == "lookup":
            text = "red green blue red green blue red green blue red green"
        else:
            text = "My favourite colour is"
        prompt = target.tokenize(text, add_bos=False)
        probs = exact_two_token_dist(target, contexts[0], prompt, T, K)
        spec = {}
        if mode == "draft":
            draft = eng.Model(model_path("qwen2.5-0.5b-q4_0.borg"), pool=pool)
            contexts.append(draft.context(n_ctx=128))
            spec = {"draft": contexts[1], "n_draft": N_DRAFT}
        elif mode == "lookup":
            spec = {"lookup_ngram": 2, "n_draft": N_DRAFT}
        counts, stats = sample_counts(contexts[0], prompt, N_DIST, T, K, **spec)
        stat, df, p = chi2_test(counts, probs, N_DIST)
        print(f"\n    {mode}: chi2={stat:.1f} df={df} p={p:.3f} drafted={stats['drafted']} "
              f"accepted={stats['accepted']}")
        if mode != "plain":
            assert stats["drafted"] > 0
            assert 0 < stats["accepted"] < stats["drafted"], "both acceptance and rejection must occur"
        assert p > 1e-3, f"{mode}: sampled distribution differs from the target (chi2={stat:.1f}, df={df})"
    finally:
        for c in contexts:
            c.close()
        target.close()
        if mode == "draft":
            draft.close()
