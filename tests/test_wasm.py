"""The WebAssembly build (wasm/build.sh, wasm/borg-runtime.js, wasm/trace-session.js) against
the native engine. Needs clang with the wasm32-wasi target and Node; skipped otherwise."""
from __future__ import annotations

import json
import os
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

TEXTS = ["Hello world", "  leading spaces and\ttabs\n\nnewlines", "naïve café 東京 🙂", "x = [i**2 for i in range(10)]",
         "<|im_start|>user\nhi<|im_end|>", "1234567890 3.14159"]
PROMPT = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture(scope="module")
def wasm_bin(tmp_path_factory):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    out = tmp_path_factory.mktemp("wasm") / "borg.wasm"
    r = subprocess.run([str(ROOT / "wasm" / "build.sh"), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("cannot build for wasm32-wasi: " + r.stderr.strip()[-300:])
    return out


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
    # percent of their range; the native AVX2 and scalar builds differ from each other as much
    assert np.abs(got - ref).max() <= 5e-2 * np.abs(ref).max()
    assert int(got.argmax()) == int(ref.argmax())


def test_trace_events_match_python_schema(wasm, native):
    session = TraceSession(native)
    ref = list(session.stream_chat([{"role": "user", "content": "What is the capital of France?"}], max_tokens=6))
    got = wasm["events"]
    assert [e["type"] for e in got] == [e["type"] for e in ref]
    for g, r in zip(got, ref):
        assert set(g) == set(r), (sorted(set(g) ^ set(r)))
    toks_g = [e for e in got if e["type"] == "token"]
    toks_r = [e for e in ref if e["type"] == "token"]
    assert [e["id"] for e in toks_g] == [e["id"] for e in toks_r]
    for g, r in zip(toks_g, toks_r):
        assert [x["layer"] for x in g["lens"]] == [x["layer"] for x in r["lens"]]
        assert g["lens"][-1]["rank"] == 0 == r["lens"][-1]["rank"]  # greedy: the last layer's top token is chosen
        assert abs(g["prob"] - r["prob"]) < 5e-2
        assert len(g["attn_top"]) == len(r["attn_top"]) == native.model.config["n_layers"]
        assert len(g["norms"]["resid"]) == native.model.config["n_layers"] + 1
        np.testing.assert_allclose(g["norms"]["resid"], r["norms"]["resid"], rtol=5e-2, atol=1e-2)
    assert got[-1]["text"] == ref[-1]["text"]


def test_javascript_build_matches_native(tmp_path):
    """wasm/build-js.sh (wasm2js) for pages that block WebAssembly: same tokens and logits."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    if not (os.environ.get("BINARYEN_WASM2JS") or shutil.which("wasm2js")):
        pytest.skip("wasm2js not available (set BINARYEN_WASM2JS)")
    if not os.path.exists(Q8):
        pytest.skip(f"{Q8} not available")
    out = tmp_path / "borg-asm.js"
    r = subprocess.run([str(ROOT / "wasm" / "build-js.sh"), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = run_wasm(out, Q8, tmp_path, max_tokens=2)
    assert got["simd"] == "scalar (JavaScript build)"
    llm = LLM(Q8, n_ctx=256, threads=2)
    for text, toks in zip(TEXTS, got["tokens"]):
        assert toks == llm.model.tokenize(text, add_bos=False), text
    llm.close()
    logits = np.array(got["logits"], np.float32)
    ref = native_logits(Q8)
    assert np.abs(logits - ref).max() <= 5e-2 * np.abs(ref).max()
    assert int(logits.argmax()) == int(ref.argmax())


def test_make_artifact_parts_reassemble(tmp_path):
    """make_artifact.py splits the model into WebAssembly-wrapped parts that rebuild the file."""
    import importlib.util
    small = os.path.join(MODELS, "smollm2-135m-q4_0.borg")
    if not os.path.exists(small):
        pytest.skip(f"{small} not available")
    page = tmp_path / "page.html"
    page.write_text("<!doctype html><html><head><title>t</title></head><body><script>1</script></body></html>")
    spec = importlib.util.spec_from_file_location("make_artifact", ROOT / "wasm" / "make_artifact.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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
