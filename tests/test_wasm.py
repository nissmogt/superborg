"""The WebAssembly build (wasm/build.sh, wasm/borg-runtime.js, wasm/trace-session.js, wasm/worker.js)
against the native engine and superborg/interp.py. Needs clang with the wasm32-wasi target and
Node; skipped otherwise."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from superborg import LLM
from superborg.interp import TraceSession

ROOT = Path(__file__).resolve().parent.parent
MODELS = os.environ.get(
    "BORG_MODELS_DIR", "/tmp/claude-0/-home-user-superborg/990fbcaa-dc72-53ae-9121-cd54b34a4307/scratchpad/borg")
Q8 = os.path.join(MODELS, "smollm2-135m-q8_0.borg")
F32 = os.path.join(MODELS, "smollm2-135m-f32.borg")
Q4 = os.path.join(MODELS, "smollm2-135m-q4_0.borg")

TEXTS = ["Hello world", "  leading spaces and\ttabs\n\nnewlines", "naïve café 東京 🙂", "x = [i**2 for i in range(10)]",
         "<|im_start|>user\nhi<|im_end|>", "1234567890 3.14159"]
PROMPT = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"


def need(path: str) -> str:
    if not os.path.exists(path):
        pytest.skip(f"{path} not available")
    return path


@pytest.fixture(scope="module")
def wasm_bin(tmp_path_factory):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    out = tmp_path_factory.mktemp("wasm")
    # a relative output path is the caller's (see test_build_scripts_write_relative_paths_where_called)
    r = subprocess.run([str(ROOT / "wasm" / "build.sh"), "borg.wasm"], cwd=out, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("cannot build for wasm32-wasi: " + r.stderr.strip()[-300:])
    return out / "borg.wasm"


@pytest.fixture(scope="module")
def asm_bin(tmp_path_factory):
    """wasm/build-js.sh (wasm2js) for pages that block WebAssembly."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    if not (os.environ.get("BINARYEN_WASM2JS") or shutil.which("wasm2js")):
        pytest.skip("wasm2js not available (set BINARYEN_WASM2JS)")
    out = tmp_path_factory.mktemp("asm")
    r = subprocess.run([str(ROOT / "wasm" / "build-js.sh"), "borg-asm.js"], cwd=out, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out / "borg-asm.js"


def node_case(case: str, req: dict, tmp_path: Path, env: dict | None = None):
    """Run one case of tests/js/wasm_cases.js (env: extra environment variables for node)."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    f = tmp_path / f"{case}.json"
    f.write_text(json.dumps(req))
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "wasm_cases.js"), case, str(f)],
                       capture_output=True, text=True, timeout=600, env=dict(os.environ, **(env or {})))
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def run_wasm(wasm_bin, model, tmp_path, max_tokens=6):
    if not os.path.exists(model):
        pytest.skip(f"{model} not available")
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"texts": TEXTS, "prompt": PROMPT, "chat": "What is the capital of France?",
                               "max_tokens": max_tokens}))
    r = subprocess.run(["node", str(ROOT / "tests" / "js" / "wasm_check.js"), str(wasm_bin), model, str(req)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def wasm(wasm_bin, tmp_path_factory):
    return run_wasm(wasm_bin, Q8, tmp_path_factory.mktemp("q8"))


@pytest.fixture(scope="module")
def native():
    llm = LLM(Q8, n_ctx=256, threads=2)
    yield llm
    llm.close()


def test_simd_build(wasm):
    assert wasm["simd"] == "wasm-simd128"


def test_build_scripts_write_relative_paths_where_called(wasm_bin, asm_bin):
    for f in (wasm_bin, asm_bin):
        assert f.is_file() and not (ROOT / f.name).exists()


def test_tokenizer_matches_native(wasm, native):
    for text, toks in zip(TEXTS, wasm["tokens"]):
        assert toks == native.model.tokenize(text, add_bos=False), text


def native_logits(path):
    llm = LLM(path, n_ctx=256, threads=2)
    ctx = llm.model.context(n_ctx=256)
    ref = ctx.decode(llm.model.tokenize(PROMPT, add_bos=False), "last")[: llm.model.n_vocab]
    ctx.close()
    llm.close()
    return ref


def test_logits_match_native_f32(wasm_bin, tmp_path):
    got = np.array(run_wasm(wasm_bin, F32, tmp_path, max_tokens=1)["logits"], np.float32)
    ref = native_logits(F32)
    # float weights: only the summation order differs between instruction sets
    assert np.abs(got - ref).max() <= 1e-3 * np.abs(ref).max()


def test_logits_match_native_q8(wasm):
    got = np.array(wasm["logits"], np.float32)
    ref = native_logits(Q8)
    # 8-bit activations: a rounding that flips one quantized activation moves the logits by a few
    # percent of their range; the native AVX-512 and scalar builds differ from each other as much
    # (2.5-3.6% over five prompts; wasm against native 2.3-5.8%, with the plain and the interleaved
    # kernels alike: bench/results/prefill_after.txt, section 7)
    assert np.abs(got - ref).max() <= 8e-2 * np.abs(ref).max()
    assert int(got.argmax()) == int(ref.argmax())


def _shape(x):
    """The nesting of lists (and dicts' key sets) of a JSON value, without the numbers."""
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_shape(v) for v in x]
    return type(x).__name__ if isinstance(x, (str, bool)) or x is None else "num"


def _dense_rows(enc_rows, n):
    from superborg.interp import decode_row
    return np.array([decode_row(e, n) for e in enc_rows])


def test_trace_events_match_python_schema(wasm, native):
    """Every event of the WebAssembly session has the keys and nesting of the Python one; the
    numbers agree within what the two engines' arithmetic allows (8-bit kernels on different
    instruction sets): decoded attention rows within 0.05, per-head statistics within 0.1 and
    per-head entropies within 0.2 nats, ranks and ids equal where the distributions are well
    separated."""
    from superborg.interp import decode_row
    session = TraceSession(native)
    ref = list(session.stream_chat([{"role": "user", "content": "What is the capital of France?"}], max_tokens=6))
    got = wasm["events"]
    cfg = native.model.config
    L, H = cfg["n_layers"], cfg["n_heads"]
    assert [e["type"] for e in got] == [e["type"] for e in ref]
    for g, r in zip(got, ref):
        assert set(g) == set(r), (sorted(set(g) ^ set(r)))
    toks_g = [e for e in got if e["type"] == "token"]
    toks_r = [e for e in ref if e["type"] == "token"]
    assert [e["id"] for e in toks_g] == [e["id"] for e in toks_r]
    for g, r in zip(toks_g, toks_r):
        # the same nesting everywhere but in the sparse rows and top lists, whose lengths depend on the values
        for k in set(g) - {"attn_rows", "attn_eff", "attn_top", "attn_heads", "alternatives", "lens", "rival", "mlp_top"}:
            assert _shape(g[k]) == _shape(r[k]), k
        assert [x["layer"] for x in g["lens"]] == [x["layer"] for x in r["lens"]]
        assert all(set(x) == {"layer", "top", "p", "rank", "entropy", "rival_p", "rival_rank"} for x in g["lens"])
        assert g["lens"][-1]["rank"] == 0 == r["lens"][-1]["rank"]  # greedy: the last layer's top token is chosen
        assert g["lens"][-1]["rival_rank"] == 1 == r["lens"][-1]["rival_rank"]
        # the late layers' lens rows differ by up to 0.24 relative between the 8-bit builds
        # (bench/results/prefill_after.txt, section 7)
        np.testing.assert_allclose([x["entropy"] for x in g["lens"]], [x["entropy"] for x in r["lens"]], rtol=0.3, atol=0.1)
        assert abs(g["prob"] - r["prob"]) < 5e-2 and abs(g["entropy"] - r["entropy"]) < 0.2
        # [text, p, id] (the page tells same-text byte tokens apart by id); greedy: the top one is chosen
        assert g["alternatives"][0][2] == g["id"] == r["alternatives"][0][2]
        assert len(g["rival"]) == 3 == len(r["rival"]) and g["rival"][2] == g["alternatives"][1][2]
        assert all(len(t) == 3 for x in g["lens"] + r["lens"] for t in x["top"])
        assert g["decode_ms"] >= 0 and r["decode_ms"] >= 0
        assert len(g["attn_top"]) == len(r["attn_top"]) == L
        assert len(g["norms"]["resid"]) == L + 1
        np.testing.assert_allclose(g["norms"]["resid"], r["norms"]["resid"], rtol=5e-2, atol=1e-2)
        np.testing.assert_allclose(g["resid"]["norm"], r["resid"]["norm"], rtol=5e-2, atol=1e-2)
        np.testing.assert_allclose(g["resid"]["spike"], r["resid"]["spike"], rtol=0.15, atol=0.5)
        assert g["resid"]["xyz"] is None and r["resid"]["xyz"] is None
        n = g["pos"] + 1
        for key in ("attn_rows", "attn_eff"):
            assert all(set(e) == {"scale", "sink", "pos", "w"} for e in g[key]) and len(g[key]) == len(r[key]) == L
            dg, dr = _dense_rows(g[key], n), _dense_rows(r[key], n)
            assert np.abs(dg - dr).max() < 0.05, key
            assert all(e["pos"] == sorted(e["pos"]) and 0 not in e["pos"] for e in g[key])
        assert [len(x) for x in g["attn_top"]] == [len(x) for x in r["attn_top"]]
        assert all(p >= 1 for x in g["attn_top"] for p, _ in x)
        top_g, top_r = [x[0][0] for x in g["attn_top"]], [x[0][0] for x in r["attn_top"]]
        assert sum(a == b for a, b in zip(top_g, top_r)) >= 0.8 * L
        assert [len(hl) for hl in g["attn_heads"]] == [H] * L and all(len(x) <= 3 and all(p >= 1 for p, _ in x)
                                                                     for hl in g["attn_heads"] for x in hl)
        for stat in g["head_stats"]:  # single heads differ more between the engines than their mean does
            a, b = np.array(g["head_stats"][stat]), np.array(r["head_stats"][stat])
            assert a.shape == b.shape == (L, H) and np.abs(a - b).max() < 0.1, stat
        ent_diff = np.abs(np.array(g["attn_entropy"]) - np.array(r["attn_entropy"]))
        assert ent_diff.max() < 0.2, ent_diff.max()  # per head, in nats: as head_stats, a single head's row
        assert [len(x) for x in g["mlp_top"]] == [len(x) for x in r["mlp_top"]] == [6] * L
    assert got[-1]["text"] == ref[-1]["text"]


def test_report_functions_match_python_on_the_same_input(native, tmp_path):
    """wasm/trace-session.js's attentionReport, residReport and encodeRow against superborg/interp.py's
    attention_report, resid_report and encode_row on identical arrays (a native trace plus a synthetic
    basis): the same positions, 8-bit weights, ranks and shapes, numbers within rounding."""
    import importlib.util
    from superborg.interp import attention_report, encode_row, prepare_basis, resid_report
    spec = importlib.util.spec_from_file_location("trace_basis", ROOT / "tools" / "trace_basis.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    cfg = native.model.config
    L, H, KV, D = cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"], cfg["dim"]
    ctx = native.ctx
    ctx.reset()
    ctx.enable_trace()
    toks = native.tokenize("The cat sat on the mat. The cat sat on the mat again, and the dog watched the cat.")
    ctx.decode(toks, None)
    tr = ctx.trace()
    n = tr["pos"] + 1
    rng = np.random.default_rng(5)
    basis = tb.fit_basis(rng.standard_normal((40, L + 1, D)))
    rows = [rng.random(50).tolist(), [1.0], [1.0, 0.0, 0.0], np.full(8, 1 / 8).tolist(), [0.9] + [1e-4] * 999]
    req = {"L": L, "H": H, "KV": KV, "D": D, "n": n, "tokens": tr["tokens"].tolist(), "n_prompt": n - 5, "attn_top": 8,
           "head_top": 3, "attn": tr["attn"].ravel().tolist(), "vnorm": tr["vnorm"].ravel().tolist(),
           "resid": tr["resid"].ravel().tolist(), "basis": basis, "rows": rows}
    js = node_case("report", req, tmp_path)
    py = attention_report(tr["attn"], tr["vnorm"], tr["tokens"], n - 5, KV, 8, 3)
    py["resid"] = resid_report(tr["resid"], prepare_basis(basis, L, D))
    py["rows"] = [encode_row(np.array(r, dtype=np.float64)) for r in rows]
    assert set(js) == set(py)
    for key in ("attn_rows", "attn_eff", "rows"):
        for a, b in zip(js[key], py[key]):
            assert a["pos"] == b["pos"] and a["w"] == b["w"], key
            assert a["sink"] == pytest.approx(b["sink"], abs=1.01e-4) and a["scale"] == pytest.approx(b["scale"], abs=1.01e-6)
    for a, b in zip(js["attn_top"], py["attn_top"]):
        assert [p for p, _ in a] == [p for p, _ in b] and [w for _, w in a] == pytest.approx([w for _, w in b], abs=1.01e-4)
    for la, lb in zip(js["attn_heads"], py["attn_heads"]):
        for a, b in zip(la, lb):
            assert [p for p, _ in a] == [p for p, _ in b] and [w for _, w in a] == pytest.approx([w for _, w in b], abs=1.01e-4)
    assert set(js["head_stats"]) == set(py["head_stats"])
    for stat in py["head_stats"]:
        np.testing.assert_allclose(js["head_stats"][stat], py["head_stats"][stat], rtol=0, atol=1.01e-3)
    np.testing.assert_allclose(js["attn_entropy"], py["attn_entropy"], rtol=0, atol=1.01e-3)
    for key in ("norm", "spike", "xyz"):
        np.testing.assert_allclose(js["resid"][key], py["resid"][key], rtol=0, atol=1.01e-3)
    assert js["rows"][1] == {"scale": 0, "sink": 1, "pos": [], "w": []} and js["rows"][4]["pos"] == []


def test_javascript_build_matches_native(asm_bin, tmp_path):
    """wasm/build-js.sh (wasm2js) for pages that block WebAssembly: same tokens and logits."""
    got = run_wasm(asm_bin, Q8, tmp_path, max_tokens=2)
    assert got["simd"] == "scalar (JavaScript build)"
    llm = LLM(Q8, n_ctx=256, threads=2)
    for text, toks in zip(TEXTS, got["tokens"]):
        assert toks == llm.model.tokenize(text, add_bos=False), text
    llm.close()
    logits = np.array(got["logits"], np.float32)
    ref = native_logits(Q8)
    assert np.abs(logits - ref).max() <= 5e-2 * np.abs(ref).max()
    assert int(logits.argmax()) == int(ref.argmax())


def test_fused_lens_matches_javascript_summary(wasm_bin, tmp_path):
    """Context.lensSummary (borg_ctx_lens_summary in WebAssembly) against the full logits
    summarized in JavaScript, the trace session's old path: the same tokens and ranks,
    probabilities within 1e-6 and the same log-sum-exp; the reports agree to their rounding."""
    out = node_case("lens", {"wasm": str(wasm_bin), "model": need(Q8), "chat": "What is the capital of France?",
                             "max_tokens": 6, "k": 4}, tmp_path)
    assert out["simd"] == "wasm-simd128" and len(out["tokens"]) == 6
    for tok in out["tokens"]:
        f, o = tok["fused"], tok["old"]
        assert f["top"] == o["top"] and f["rank"] == o["rank"] and f["rival_rank"] == o["rival_rank"]
        assert len(f["top"]) == 4 * len(tok["layers"]) and len(tok["layers"]) == 31
        np.testing.assert_allclose(f["p"], o["p"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(f["rival_p"], o["rival_p"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(f["p_top"], o["p_top"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(f["lse"], o["lse"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(f["entropy"], o["entropy"], rtol=0, atol=1e-6)
        assert f["rank"][-1] == 0 and f["rival_rank"][-1] == 1  # greedy: the chosen token tops the last layer
        for a, b in zip(tok["report"], tok["plain_report"]):
            assert a["layer"] == b["layer"] and a["rank"] == b["rank"] and a["rival_rank"] == b["rival_rank"]
            assert [x[2] for x in a["top"]] == [x[2] for x in b["top"]] and len(a["top"]) == 4
            assert abs(a["p"] - b["p"]) <= 1.1e-5 and abs(a["rival_p"] - b["rival_p"]) <= 1.1e-5
            assert abs(a["entropy"] - b["entropy"]) <= 1.1e-4
            assert all(abs(x[1] - y[1]) <= 1.1e-4 for x, y in zip(a["top"], b["top"]))


def make_artifact():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_artifact", ROOT / "wasm" / "make_artifact.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_artifact_parts_reassemble(tmp_path):
    """make_artifact.py splits the model into WebAssembly-wrapped parts that rebuild the file."""
    small = need(Q4)
    page = tmp_path / "page.html"
    page.write_text("<!doctype html><html><head><title>t</title></head><body><script>1</script></body></html>")
    mod = make_artifact()
    wasm = tmp_path / "fake.wasm"
    wasm.write_bytes(b"\0asm\x01\0\0\0")
    out = tmp_path / "site"
    assert mod.main([small, str(out), "--page", str(page), "--wasm", str(wasm), "--no-asm", "--fragment",
                     "--part-size", "30000000"]) == 0
    man = json.loads((out / "model.json").read_text())
    data = b""
    for part in man["model"]["parts"]:
        b = (out / part["path"]).read_bytes()
        assert b[:8] == b"\0asm\x01\0\0\0" and b[8] == 0          # module header, custom section
        i, n, shift = 9, 0, 0
        while True:                                                 # section size (LEB128)
            c = b[i]; i += 1; n |= (c & 0x7F) << shift; shift += 7
            if not c & 0x80:
                break
        end = i + n
        name_len = b[i]; i += 1
        assert b[i:i + name_len] == b"borg-part"
        payload = b[i + name_len:end]
        assert end == len(b) and len(payload) == part["size"]
        data += payload
    assert data == Path(small).read_bytes()
    html = (out / "index.html").read_text()
    assert "<html" not in html and "<body" not in html and "BORG_TRANSPORT" in html
    assert html.index("BORG_TRANSPORT") < html.index("<script>1</script>")


# ---- runtime and engine limits ------------------------------------------------------------

def test_heap_above_2gib(wasm_bin, tmp_path):
    """Addresses from 2 GiB up reach JavaScript as negative i32 results: read as unsigned, a model
    and context placed there give the same trace as low in the heap."""
    req = {"wasm": str(wasm_bin), "model": need(Q4), "chat": "What is the capital of France?", "max_tokens": 4}
    low = node_case("chat", dict(req, high=False), tmp_path)
    high = node_case("chat", dict(req, high=True), tmp_path)
    assert low["probe"] < 2**31 <= high["probe"]
    assert high["events"] == low["events"]
    assert high["tokens"] == low["tokens"] and high["trace"] == low["trace"]


def wrapped_model(src: str, dst: Path) -> None:
    """A copy of src whose tok_embd claims f32 [2^24, dim] in 0 bytes: for dim 576 that size is
    9 * 2^32 bytes, 0 in 32-bit arithmetic."""
    data = Path(src).read_bytes()
    hlen = int.from_bytes(data[8:16], "little")
    h = json.loads(data[16:16 + hlen])
    c = h["config"]
    assert (c["dim"] << 26) % (1 << 32) == 0
    c.update(vocab_size=1 << 24, tie_embeddings=True)
    h["tensors"].pop("output", None)
    h["tensors"]["tok_embd"].update(dtype="f32", shape=[1 << 24, c["dim"]], nbytes=0)
    hb = json.dumps(h).encode()
    dst.write_bytes(b"BORG" + (1).to_bytes(4, "little") + len(hb).to_bytes(8, "little") + hb
                    + b"\0" * (-(16 + len(hb)) % 64) + data[(16 + hlen + 63) // 64 * 64:])


def test_tensor_size_that_wraps_32_bits_is_refused(wasm_bin, tmp_path):
    bad = tmp_path / "wrap.borg"
    wrapped_model(need(Q4), bad)
    out = node_case("load", {"wasm": str(wasm_bin), "model": str(bad)}, tmp_path)
    assert out["error"] == "Error: cannot load model: tensor tok_embd: size 0 does not match its shape"
    from superborg.engine import Model
    with pytest.raises(Exception, match="tok_embd: size 0 does not match its shape"):
        Model(str(bad), threads=1)


def test_context_that_wraps_32_bits_is_refused(wasm_bin, tmp_path):
    """30 layers x 192 KV values x 745655 positions is just over 2^32 elements: refused, not
    allocated as the few elements a 32-bit product wraps to."""
    out = node_case("context", {"wasm": str(wasm_bin), "model": need(Q4), "n_ctx": 745655}, tmp_path)
    assert out["error"] == "Error: cannot allocate a context (try a smaller one)"


def test_javascript_build_out_of_memory(asm_bin, tmp_path):
    """The JavaScript build stops below 2 GiB with the runtime's error (not a RangeError from
    wasm2js's memory growth), and keeps working."""
    out = node_case("malloc", {"wasm": str(asm_bin), "sizes": [1_200_000_000, 1_000_000_000, 100_000_000]}, tmp_path)
    assert out["errors"] == [None, "Error: out of WebAssembly memory allocating 1000000000 bytes", None]


# ---- sampling and prompts as in superborg/interp.py and superborg/chat.py -----------------

def test_sampler_matches_interp(tmp_path):
    """TraceSession.sample draws from what interp.py's _sample does: the whole distribution when
    top_k is 0 (not only the 256 most likely tokens), otherwise the most likely top_k tokens cut
    where the cumulative probability reaches top_p."""
    settings = [[seed, n, t, top_p, k] for seed, (n, t) in enumerate([(50, 1.0), (5000, 0.5), (5000, 2.0)], 1)
                for top_p in (1.0, 0.9, 0.5, 0.05) for k in (0, 1, 10, 300)]
    out = node_case("sample", {"flat": 1000, "draws": 4000, "cases": settings, "topk": 151665}, tmp_path)
    assert 0.68 < out["beyond"] < 0.81  # 744 of 1000 equally likely tokens lie past the first 256
    for (seed, n, t, top_p, k), case in zip(settings, out["cases"]):
        p = np.array(case["p"])
        order = np.argsort(-p, kind="stable")
        if k > 0:
            order = order[:k]
        if top_p < 1.0:
            order = order[: int(np.searchsorted(np.cumsum(p[order]), top_p) + 1)]
        if case["cand"] is None:  # every token: drawn without a sort
            assert len(order) == n, (seed, n, t, top_p, k)
        else:
            assert case["cand"] == order.tolist(), (seed, n, t, top_p, k)
    assert out["topk_sorted"] and out["topk_ms"] < 3000  # top_k = vocabulary size: a sort, not O(V k)


CHATS = [
    [{"role": "user", "content": "Hi there"}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
     {"role": "user", "content": "Tell me a joke."}],
    [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "What is 2+2?"}],
    [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"},
     {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Bye"}],
]


def test_render_chat_matches_python_formatter(tmp_path):
    """renderChat gives superborg/chat.py's prompts: the models' own templates (default system
    prompts included), and the built-in renderers for formats with no model here (mistral)."""
    pytest.importorskip("jinja2")
    from superborg.chat import ChatFormatter, formatter_for
    from superborg.engine import Model
    infos, want = [], []
    for name in ("smollm2-135m-q8_0", "qwen2.5-0.5b-q8_0", "qwen3-0.6b-q8_0", "tinyllama-1.1b-q8_0"):
        path = os.path.join(MODELS, name + ".borg")
        if os.path.exists(path):
            m = Model(path, threads=1)
            f = formatter_for(m)
            # the header plus the special tokens' text, as trace-session.js's chatInfo(model) adds it
            infos.append(dict(m.tokenizer_info, bos_token=f.bos_token, eos_token=f.eos_token, eos_tokens=f.eos_tokens))
            want.append(([f.render(c) for c in CHATS], f.stop_strings()))
            m.close()
    for fmt in ("chatml", "llama3", "zephyr", "mistral", "plain"):
        infos.append({"chat_format": fmt})
        f = ChatFormatter(infos[-1])
        want.append(([f.render(c) for c in CHATS], f.stop_strings()))
    got = node_case("render", {"infos": infos, "conversations": CHATS}, tmp_path)
    for info, (prompts, stops), g in zip(infos, want, got):
        assert g["prompts"] == prompts, info["chat_format"]
        assert g["stops"] == stops, info["chat_format"]


# ---- chat templates: wasm/vendor/hfjinja.js (@huggingface/jinja) against superborg/chat.py (jinja2)

HF_MODELS = os.environ.get(
    "HF_MODELS_DIR", "/tmp/claude-0/-home-user-superborg/990fbcaa-dc72-53ae-9121-cd54b34a4307/scratchpad/models")
# more checkpoints' tokenizer_config.json / chat_template.jinja, laid out as model directories
TEMPLATES = os.environ.get(
    "BORG_TEMPLATES_DIR", "/tmp/claude-0/-home-user-superborg/990fbcaa-dc72-53ae-9121-cd54b34a4307/scratchpad/work-templates")

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather by city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CONVERSATIONS = {
    "user": [{"role": "user", "content": "What is the capital of France?"}],
    "system_multi": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
                     {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Name a colour."}],
    "tools": [{"role": "user", "content": "Weather in Paris?"},
              {"role": "assistant", "content": "",
               "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
              {"role": "tool", "name": "get_weather", "content": "{\"temp\": 18}"},
              {"role": "assistant", "content": "It is 18 C."}, {"role": "user", "content": "And in Rome?"}],
    # OpenAI-style tool calls: content null, the arguments a dict or the JSON string clients send
    # (DeepSeek's templates concatenate them into the prompt: a TypeError on both sides for a dict)
    "tool_call_null": [{"role": "user", "content": "Weather in Paris?"},
                       {"role": "assistant", "content": None,
                        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                       {"role": "tool", "name": "get_weather", "content": "{\"temp\": 18}"},
                       {"role": "user", "content": "And in Rome?"}],
    "tool_call_json": [{"role": "user", "content": "Weather in Paris?"},
                       {"role": "assistant", "content": None,
                        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                       {"role": "tool", "name": "get_weather", "content": "{\"temp\": 18}"},
                       {"role": "user", "content": "And in Rome?"}],
    # spaces next to the newlines around the think block: Python's strip('\n') keeps them
    "think": [{"role": "user", "content": "Is 7 prime?"},
              {"role": "assistant", "content": "<think>\n  Check divisors.  \n</think>\n\n  Yes, 7 is prime.  "},
              {"role": "user", "content": "And 9?"}],
    "reasoning_field": [{"role": "user", "content": "Is 7 prime?"},
                        {"role": "assistant", "content": "Yes.", "reasoning_content": "7 has no divisors."},
                        {"role": "user", "content": "And 9?"}],
    "unicode": [{"role": "user", "content": "Café — 日本語 😀 \"quotes\" 'single' \\ backslash <tag> {{ braces }} %s \t tab"}],
    "empty": [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
}
SETTINGS = [
    {"add_generation_prompt": True},
    {"add_generation_prompt": False},
    {"add_generation_prompt": True, "enable_thinking": False},
    {"add_generation_prompt": True, "enable_thinking": True},
    {"add_generation_prompt": False, "enable_thinking": False},
    {"add_generation_prompt": False, "enable_thinking": True},
    {"add_generation_prompt": True, "extra": {"date_string": "16 Oct 2026"}},
    {"add_generation_prompt": True, "tools": True},
]
# divergences of the vendored release that are not patched: (template name part, case predicate, why)
KNOWN_DIVERGENCES = [
    ("smollm3", lambda case, got: "string" in got.get("error", "") and case.endswith("#7"),
     "SmolLM3 pipes each tool dict through `| string`: Python gives the dict's repr, @huggingface/jinja 0.5.10 "
     "has no string filter for objects (unreported upstream as of 2026-10-16)"),
]


def _special_token(value) -> str:
    return (value.get("content") if isinstance(value, dict) else value) or ""


def chat_templates() -> dict[str, dict]:
    """{name: {"chat_template", "bos_token", "eos_token"}} of every checkpoint directory in
    HF_MODELS_DIR and BORG_TEMPLATES_DIR that has a template (superborg.convert's reading)."""
    from superborg.convert import load_chat_template
    out = {}
    for base in (HF_MODELS, TEMPLATES):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            tc_path = os.path.join(d, "tokenizer_config.json")
            if not os.path.isfile(tc_path):
                continue
            with open(tc_path, encoding="utf-8") as f:
                tc = json.load(f)
            tpl = load_chat_template(d, tc)
            if tpl:
                out[name] = {"chat_template": tpl, "bos_token": _special_token(tc.get("bos_token")),
                             "eos_token": _special_token(tc.get("eos_token"))}
    return out


def python_renders(info: dict) -> dict:
    """ChatFormatter.render for every conversation x setting: {"case": {"text"} | {"error"}}."""
    from superborg.chat import ChatFormatter
    tokens = [info["bos_token"], info["eos_token"]]
    f = ChatFormatter({"chat_template": info["chat_template"], "bos_id": 0 if tokens[0] else -1,
                       "eos_id": 1 if tokens[1] else -1}, token_text=lambda i: tokens[i])
    out = {}
    for cn, conv in CONVERSATIONS.items():
        for i, s in enumerate(SETTINGS):
            kw = dict(s.get("extra", {}))
            if "enable_thinking" in s:
                kw["enable_thinking"] = s["enable_thinking"]
            if s.get("tools"):
                kw["tools"] = TOOLS
            try:
                out[f"{cn}#{i}"] = {"text": f.render(conv, add_generation_prompt=s["add_generation_prompt"], **kw)}
            except Exception as e:  # noqa: BLE001 - raise_exception, or a feature the template needs
                out[f"{cn}#{i}"] = {"error": f"{type(e).__name__}: {e}"}
    return out


@pytest.fixture(scope="module")
def template_renders(tmp_path_factory):
    """The node side of the matrix, once per locale: the vendored library under the default
    (en-US) locale and under German, where its unpatched strftime_now would localize months."""
    pytest.importorskip("jinja2")
    templates = chat_templates()
    if not templates:
        pytest.skip(f"no chat templates under {HF_MODELS} or {TEMPLATES}")
    req = {"templates": templates, "conversations": CONVERSATIONS, "settings": SETTINGS, "tools": TOOLS}
    out = {}
    for lang in ("C", "de_DE.UTF-8"):
        out[lang] = node_case("template", req, tmp_path_factory.mktemp("tpl"), env={"LANG": lang, "LC_ALL": lang})
    return templates, out


@pytest.mark.parametrize("lang", ["C", "de_DE.UTF-8"])
@pytest.mark.parametrize("name", sorted(chat_templates()) or ["(none)"])
def test_chat_template_parity(template_renders, name, lang):
    """Every template x conversation x setting renders to the same bytes in the browser engine's
    renderChat and in superborg/chat.py, or fails on both sides; the vendored library's known
    divergences (KNOWN_DIVERGENCES) are expected failures. date_string is passed explicitly in
    one setting (Llama 3.x honour it); SmolLM3 (%d %B %Y) and gpt-oss (%Y-%m-%d) call
    strftime_now themselves, which the vendored copy renders in the C locale like Python."""
    templates, renders = template_renders
    info, got = templates[name], renders[lang][name]
    want = python_renders(info)
    assert set(got) == set(want)
    bad, known = [], []
    for case, w in want.items():
        g = got[case]
        if "error" in w:
            if "error" not in g:
                bad.append((case, "python error, js text", w["error"]))
            continue
        if g.get("text") == w["text"]:
            continue
        reason = next((why for part, pred, why in KNOWN_DIVERGENCES if part in name.lower() and pred(case, g)), None)
        (known if reason else bad).append((case, reason or g, w["text"][:200] if reason is None else ""))
    assert not bad, bad[:3]
    assert sum("error" not in w for w in want.values()) >= len(CONVERSATIONS)  # the template did render
    if known:
        pytest.xfail(f"{len(known)} known divergence(s): " + known[0][1])


STRFTIME_FORMATS = ["%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%a %A %b %B %c", "%d %e %f %H %I %j %k %l %m %M %p %P %S",
                    "%U %w %W %x %X %y %Y %z %Z %%", "%C %D %F %T %R %u %G %V %g %h %s", "%n|%t|%-d|%-m|%-H|%Q|100%"]
# local midnights and odd hours around year ends and ISO week boundaries, with milliseconds
STRFTIME_TIMES = ["2026-10-16T23:05:09.123", "2026-01-01T00:00:00.000", "2026-01-04T07:05:09.000", "2026-01-05T12:00:00.500",
                  "2027-01-01T13:00:00.000", "2028-12-31T23:59:59.999", "2024-12-30T01:00:00.000", "2021-01-03T00:30:00.000",
                  "2026-03-29T02:30:00.000", "2026-12-31T11:59:59.000"]


@pytest.mark.parametrize("lang", ["C", "de_DE.UTF-8"])
def test_vendored_strftime_matches_python(tmp_path, lang):
    """strftime_now in wasm/vendor/hfjinja.js (the strftime-c-locale patch) gives Python's
    datetime.strftime for every directive, in any locale."""
    import datetime
    times = [datetime.datetime.fromisoformat(t) for t in STRFTIME_TIMES]
    got = node_case("strftime", {"times": [int(t.timestamp() * 1000) for t in times], "formats": STRFTIME_FORMATS},
                    tmp_path, env={"LANG": lang, "LC_ALL": lang})
    want = [[t.strftime(f) for f in STRFTIME_FORMATS] for t in times]
    assert got == want


def test_vendored_plus_concatenates_only_strings(tmp_path):
    """The str-concat patch: `+` with a str operand needs a str on the other side, as in Python,
    where @huggingface/jinja 0.5.10 rendered a dict as "[object Map]" (DeepSeek's templates
    concatenate tool_calls[].function.arguments into the prompt); str + str is unchanged."""
    pytest.importorskip("jinja2")
    from superborg.chat import ChatFormatter
    probe = "{{ 'args=' + messages[1].tool_calls[0].function.arguments + '|' + messages[0].content }}"
    convs = {k: CONVERSATIONS[k] for k in ("tool_call_null", "tool_call_json")}
    got = node_case("template", {"templates": {"probe": {"chat_template": probe, "bos_token": "", "eos_token": ""}},
                                 "conversations": convs, "settings": SETTINGS[:1], "tools": TOOLS}, tmp_path)["probe"]
    f = ChatFormatter({"chat_template": probe})
    with pytest.raises(TypeError, match='can only concatenate str \\(not "dict"\\)'):
        f.render(convs["tool_call_null"])
    assert got["tool_call_null#0"] == {"error": "unsupported operand types for +: StringValue and ObjectValue"}
    want = 'args={"city": "Paris"}|Weather in Paris?'
    assert f.render(convs["tool_call_json"]) == want and got["tool_call_json#0"] == {"text": want}


def test_vendored_none_prints_like_python(tmp_path):
    """The none-output patch: an expression that is None prints "None" as in Python (a null message
    content, a variable set to none, a macro's argument), where @huggingface/jinja 0.5.10 printed
    nothing; an undefined value, an inline if without else, set, macro and comment statements
    still print nothing."""
    pytest.importorskip("jinja2")
    from superborg.chat import ChatFormatter
    probe = ("{{ messages[1].content }}|{% set x = none %}{{ x }}|{{ messages[0].missing }}|{{ 'a' if false }}|"
             "{# c #}{% macro m(v) %}{{ v }}{% endmacro %}{{ m(none) }}")
    conv = {"tool_call_null": CONVERSATIONS["tool_call_null"]}
    got = node_case("template", {"templates": {"probe": {"chat_template": probe, "bos_token": "", "eos_token": ""}},
                                 "conversations": conv, "settings": SETTINGS[:1], "tools": TOOLS}, tmp_path)["probe"]
    want = "None|None|||None"
    assert ChatFormatter({"chat_template": probe}).render(conv["tool_call_null"]) == want
    assert got["tool_call_null#0"] == {"text": want}


def test_vendored_jinja_is_the_pinned_release():
    """wasm/vendor/hfjinja.js carries the version, hash, patches and license of wasm/vendor_jinja.py,
    and so do the third-party notices."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("vendor_jinja", ROOT / "wasm" / "vendor_jinja.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    js = (ROOT / "wasm" / "vendor" / "hfjinja.js").read_text()
    md = (ROOT / "wasm" / "vendor" / "THIRD_PARTY_NOTICES.md").read_text()
    head = js[:js.index("*/")]
    assert head.startswith(f"/*! {mod.NAME} {mod.VERSION} ") and mod.DIST_SHA256 in head and "MIT License" in head
    assert js.rstrip().endswith('})(typeof self !== "undefined" ? self : globalThis);')
    assert "root.HFJinja = { " + ", ".join(mod.EXPORTS) + " };" in js
    for name, why, edits in mod.PATCHES:
        assert name in head and f"**{name}**" in md and why in md
        for _old, new in edits:
            assert new in js
    for s in (mod.VERSION, mod.DIST_SHA256, mod.LICENSE_SHA256, mod.DIST_URL, "Copyright (c) 2023 Hugging Face"):
        assert s in md
    assert not re.search(r"</script|<!--|<script", js, flags=re.I)  # make_artifact.py inlines the bundle


# ---- wasm/worker.js: the browser engine's side of the page's protocol (as superborg/ui.py) -

HI = [{"role": "user", "content": "Hi"}]


@pytest.fixture(scope="module")
def site(wasm_bin, tmp_path_factory):
    """A static site from make_artifact.py: smollm2-135m-q4_0, n_ctx 1024, lens stride 3."""
    out = tmp_path_factory.mktemp("site")
    page = out / "page.html"
    page.write_text("<!doctype html><title>t</title><script>1</script>")
    assert make_artifact().main([need(Q4), str(out), "--page", str(page), "--wasm", str(wasm_bin), "--no-asm",
                                 "--n-ctx", "1024", "--name", "Smol"]) == 0
    return out


def run_worker(site, steps, tmp_path, idb=()):
    """Run the steps of tests/js/worker_check.js against wasm/worker.js serving `site`."""
    req = tmp_path / "worker.json"
    req.write_text(json.dumps({"idb": list(idb), "steps": steps}))
    # the result goes to a file, not a pipe: node exits right after writing it, and a pipe
    # keeps only the first 64 KiB of an unflushed write (a traced chat's events are larger)
    out = tmp_path / "worker.out"
    with out.open("w") as f:
        r = subprocess.run(["node", str(ROOT / "tests" / "js" / "worker_check.js"), str(site), str(req)],
                           stdout=f, stderr=subprocess.PIPE, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text())


def test_worker_info_and_part_cache(site, tmp_path):
    """Info carries the defaults the page's settings show; at init the IndexedDB cache drops the
    parts of other versions of the model, and keeps other models' parts."""
    stale, other = "Smol:0123456789abcdef:model/part-00.wasm:5", "Other:0123456789abcdef:model/part-00.wasm:5"
    init, info = run_worker(site, [["init", None], ["info", None]], tmp_path, idb=[[stale, 1], [other, 2]])
    assert init["info"] == info
    assert info["lens_stride"] == 3 and info["lens_layers"] == list(range(0, 31, 3))
    assert info["max_tokens_default"] == 512 and info["n_ctx"] == 1024 and info["busy"] is False
    assert info["model"] == "Smol" and info["backend"] == "wasm (wasm-simd128)"
    mm = json.loads((site / "model.json").read_text())["model"]
    assert sorted(init["idb"]) == sorted([other] + [f"Smol:{mm['id']}:{p['path']}:{p['size']}" for p in mm["parts"]])


def test_worker_missing_manifest(tmp_path):
    (tmp_path / "site").mkdir()
    out = run_worker(tmp_path / "site", [["init", None]], tmp_path)
    assert out == [{"fatal": "cannot download http://site.test/model.json: HTTP 404"}]


def test_worker_chat_events(site, tmp_path):
    """Token ids in alternatives and lens tops, decode_ms on every token (the eos one too);
    max_tokens and lens_stride clamped from above; without lens_stride, the default again."""
    france = [{"role": "user", "content": "What is the capital of France?"}]
    _, first, clamped, all_layers, default = run_worker(site, [
        ["init", None],
        ["chat", {"messages": france, "max_tokens": 40, "request_id": "r-1"}],
        ["stopAt", ["token", None, {"messages": HI, "max_tokens": 100000, "lens_stride": 1000}]],
        ["chat", {"messages": HI, "max_tokens": 1, "lens_stride": 1}],
        ["chat", {"messages": HI, "max_tokens": 1}],
    ], tmp_path)
    toks = [e for e in first if e["type"] == "token"]
    assert [e["type"] for e in first] == ["prompt"] + ["token"] * len(toks) + ["done"]
    assert toks[-1]["eos"] and first[-1]["stop_reason"] == "eos"
    for e in toks:
        assert e["alternatives"][0][2] == e["id"] == e["lens_top"][0][2] and all(len(a) == 3 for a in e["alternatives"])
        assert isinstance(e["decode_ms"], (int, float)) and e["decode_ms"] >= 0
    assert clamped["reply"] == {"stopped": True} and clamped["events"][1]["layers"] == [0, 30]
    assert clamped["events"][-1]["stop_reason"] == "cancelled"
    assert all_layers[1]["layers"] == list(range(31)) and default[1]["layers"] == list(range(0, 31, 3))


def test_worker_stops(site, tmp_path):
    """A stop with a request_id cancels only that chat, also during the prompt and before the chat
    arrives; it never reaches a later chat. Without an id it cancels whatever runs."""
    long = [{"role": "user", "content": "word " * 600}]
    _, early, other, prefill, after, anyone, ended, idle = run_worker(site, [
        ["init", None],
        ["early", ["e1", {"messages": HI, "max_tokens": 2}]],
        ["stopAt", ["token", "other", {"messages": HI, "max_tokens": 3, "request_id": "mine"}]],
        ["stopAt", ["prefill", "long", {"messages": long, "max_tokens": 3, "request_id": "long"}]],
        ["chat", {"messages": HI, "max_tokens": 2}],
        ["stopAt", ["token", None, {"messages": HI, "max_tokens": 50}]],
        ["stop", "mine"],
        ["stop", None],
    ], tmp_path)
    cancelled = {"type": "done", "text": "", "n_gen": 0, "stop_reason": "cancelled", "seconds": 0, "tok_per_s": 0}
    assert early["reply"] == {"stopped": False, "pending": True}
    assert [e["type"] for e in early["events"]] == ["prompt", "done"] and early["events"][1] == cancelled
    assert early["other"][-1]["stop_reason"] == "length"
    assert other["reply"] == {"stopped": False, "pending": True} and other["events"][-1]["stop_reason"] == "length"
    n = prefill["events"][1]["n_prompt"] - prefill["events"][1]["n_reused"]
    assert prefill["reply"] == {"stopped": True} and n > 64
    assert prefill["events"] == [{"type": "prefill", "done": 64, "total": n}, prefill["events"][1], cancelled]
    assert after[-1]["stop_reason"] == "length" and after[-1]["n_gen"] == 2
    assert anyone["reply"] == {"stopped": True} and anyone["events"][-1]["stop_reason"] == "cancelled"
    assert ended == {"stopped": False} and idle == {"stopped": False}


def test_worker_messages_are_checked_like_the_server(site, tmp_path):
    """Message rules of superborg/server.py parse_messages: known roles only, developer means
    system, content may be a list of text parts; enable_thinking must be a boolean; a stop with a
    malformed request_id is refused."""
    _, ok, *bad = run_worker(site, [
        ["init", None],
        ["chat", {"messages": [{"role": "developer", "content": "Be brief."},
                               {"role": "user", "content": [{"type": "text", "text": "hi"}, "!"]}], "max_tokens": 1}],
        ["chat", {"messages": [{"role": "robot", "content": "hi"}]}],
        ["chat", {"messages": [{"role": "user", "content": [{"type": "text", "text": None}]}]}],
        ["chat", {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}],
        ["chat", {"messages": HI, "enable_thinking": "yes"}],
    ], tmp_path)
    assert ok[-1]["type"] == "done" and ok[-1]["n_gen"] == 1
    assert [e for b in bad for e in b] == [
        {"type": "error", "message": "messages[0]: unknown role 'robot'"},
        {"type": "error", "message": "messages[0].content[0]: 'text' must be a string"},
        {"type": "error", "message": "messages[0]: content part type 'image_url' is not supported"},
        {"type": "error", "message": "'enable_thinking' must be true or false"},
    ]


def test_worker_chat_template_kwargs(site, tmp_path):
    """The chat body's chat_template_kwargs reach the template (SmolLM2's ignores date_string
    and enable_thinking, so the prompt stays the same length); bad values are errors, as the
    server's 400s."""
    _, plain, extra, *bad = run_worker(site, [
        ["init", None],
        ["chat", {"messages": HI, "max_tokens": 1}],
        ["chat", {"messages": HI, "max_tokens": 1,
                  "chat_template_kwargs": {"date_string": "16 Oct 2026", "enable_thinking": True}}],
        ["chat", {"messages": HI, "chat_template_kwargs": [1]}],
        ["chat", {"messages": HI, "chat_template_kwargs": {"enable_thinking": "yes"}}],
        ["chat", {"messages": HI, "chat_template_kwargs": {"messages": []}}],
        ["chat", {"messages": HI, "chat_template_kwargs": {"add_generation_prompt": False}}],
    ], tmp_path)
    assert plain[0]["n_prompt"] == extra[0]["n_prompt"] and extra[-1]["type"] == "done"
    assert [e for b in bad for e in b] == [
        {"type": "error", "message": "'chat_template_kwargs' must be an object"},
        {"type": "error", "message": "'chat_template_kwargs.enable_thinking' must be true or false"},
        {"type": "error", "message": "'chat_template_kwargs' cannot set 'messages'"},
        {"type": "error", "message": "'chat_template_kwargs' cannot set 'add_generation_prompt'"},
    ]


@pytest.fixture(scope="module")
def site_with_basis(wasm_bin, tmp_path_factory):
    """As `site`, with a (synthetic) residual basis shipped as basis.json."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trace_basis", ROOT / "tools" / "trace_basis.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    out = tmp_path_factory.mktemp("site_basis")
    page = out / "page.html"
    page.write_text("<!doctype html><title>t</title><script>1</script>")
    header = make_artifact().read_header(Path(need(Q4)))
    cfg = header["config"]
    basis = out / "synthetic.basis.json"
    basis.write_text(json.dumps(tb.fit_basis(np.random.default_rng(6).standard_normal((40, cfg["n_layers"] + 1, cfg["dim"])))))
    assert make_artifact().main([need(Q4), str(out), "--page", str(page), "--wasm", str(wasm_bin), "--no-asm",
                                 "--n-ctx", "1024", "--basis", str(basis)]) == 0
    return out


def test_worker_trace_prompt_and_basis(site, site_with_basis, tmp_path):
    """trace_prompt gives a prompt_token event per uncached prompt position before the last, with
    the trace fields (no sampling fields); a basis listed in model.json is loaded (info.basis) and
    fills the xyz projection; without one xyz is null; a bad flag is an error, as the server's 400."""
    # the traced chat goes first: a prompt the context already holds is not re-evaluated, so a
    # repeat of it would trace nothing (n_reused = n_prompt - 1)
    _, info, traced, plain, bad = run_worker(site_with_basis, [
        ["init", None],
        ["info", None],
        ["chat", {"messages": HI, "max_tokens": 1, "trace_prompt": True}],
        ["chat", {"messages": HI, "max_tokens": 1}],
        ["chat", {"messages": HI, "trace_prompt": "yes"}],
    ], tmp_path)
    assert info["basis"] is True
    man = json.loads((site_with_basis / "model.json").read_text())
    assert man["basis"] == "basis.json" and (site_with_basis / "basis.json").is_file()
    assert [e["type"] for e in plain] == ["prompt", "token", "done"]
    n = plain[0]["n_prompt"]
    pts = [e for e in traced if e["type"] == "prompt_token"]
    assert [e["type"] for e in traced] == ["prompt_token"] * (n - 1) + ["prompt", "token", "done"]
    assert [e["pos"] for e in pts] == list(range(n - 1)) and traced[-2]["id"] == plain[1]["id"]
    for e in pts:
        assert not {"prob", "entropy", "alternatives", "i", "text", "eos"} & set(e)
        for k in ("next", "rival", "lens", "norms", "resid", "attn_rows", "attn_eff", "attn_top", "attn_heads",
                  "head_stats", "attn_entropy", "mlp_top", "decode_ms", "trace_ms"):
            assert k in e, k
        assert len(e["resid"]["xyz"]) == 31 and all(len(p) == 3 for p in e["resid"]["xyz"])
        assert len(e["lens"]) == len(range(0, 31, 3)) and e["lens"][-1]["layer"] == 30
    assert bad == [{"type": "error", "message": "'trace_prompt' must be true or false"}]
    _, info, traced = run_worker(site, [["init", None], ["info", None],
                                        ["chat", {"messages": HI, "max_tokens": 1, "trace_prompt": True}]], tmp_path)
    assert info["basis"] is False and "basis" not in json.loads((site / "model.json").read_text())
    assert all(e["resid"]["xyz"] is None for e in traced if e["type"] == "prompt_token")


def test_worker_busy_and_bad_requests(site, tmp_path):
    """A second chat gets one in-band busy error (the page retries it); bad values are errors, as
    the server's 400s."""
    _, busy, *bad = run_worker(site, [
        ["init", None],
        ["busy", {"messages": HI, "max_tokens": 2}],
        ["chat", {"messages": HI, "max_tokens": 0}],
        ["chat", {"messages": HI, "lens_stride": 0}],
        ["chat", {"messages": HI, "request_id": "not an id"}],
        ["chat", {"messages": HI, "temperature": "hot"}],
        ["chat", {"messages": []}],
        ["chat", {"messages": [{"role": "user", "content": "word " * 1500}]}],
    ], tmp_path)
    assert busy["first"][-1]["stop_reason"] == "length"
    assert busy["second"] == [{"type": "error", "message": "a generation is already running", "code": "busy"}]
    assert [e for b in bad for e in b] == [
        {"type": "error", "message": "'max_tokens' must be at least 1"},
        {"type": "error", "message": "'lens_stride' must be at least 1"},
        {"type": "error", "message": "'request_id' must be 1-64 characters from A-Z, a-z, 0-9, '_' and '-'"},
        {"type": "error", "message": "'temperature' must be a finite number"},
        {"type": "error", "message": "'messages' must be a non-empty list"},
        {"type": "error", "message": "the conversation (1531 tokens) does not fit the 1024-token context",
         "code": "context_length_exceeded"},
    ]
