"""The NumPy reference forward pass against Hugging Face transformers, and the
converter against the original checkpoints.

Real models come from HF_MODELS_DIR (Hugging Face directories) and
BORG_MODELS_DIR (<key>-<qtype>.borg files, see tools/hf_oracle.py). Every test
skips cleanly when torch/transformers or the model files are missing; the
format-level tests at the top need only NumPy.

Run with -s to see the measured numbers.
"""
from __future__ import annotations

import gc
import json
import os
import re
import struct
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hf_oracle as ho  # noqa: E402
from superborg import convert  # noqa: E402
from superborg.reference import ReferenceModel, kl_divergence, rope_inv_freq  # noqa: E402

SPECS = {m.key: m for m in ho.MODELS}
# <key>-<qtype>.borg files checked when present; "q4" is the converter's recipe (calibrated
# q4_1 with attn_v/ffn_down at q8_0)
QTYPES = ["q8_0", "q4_0", "q4_1", "q4"]

# Quality bounds for the quantized files, measured over the prompt plus the 32-token
# greedy continuation (teacher-forced), against HF float32: (min top-1 agreement, max mean KL).
# q8_0 measures KL 4e-4..3e-3 and top-1 >= 0.96 on all four models.
# q4_0 (q8_0 embeddings, every other matrix q4_0) measures mean KL 0.27 (SmolLM2), 0.24
# (Qwen2.5), 0.27 (Qwen3) and 0.095 (TinyLlama), so a single 0.15 bound does not hold.
# The loss is inherent to plain 4-bit blocks at these sizes, not the converter:
# re-quantizing with ggml's reference Q4_0 rounding gives 0.35 (SmolLM2) and 0.24
# (Qwen2.5). Plain q4_1 measures 0.22 (SmolLM2); the "q4" recipe (calibrated q4_1,
# attn_v/ffn_down at q8_0) measures 0.071 (SmolLM2) and 0.070 (Qwen2.5). The bounds are
# ~1.5x the measurements: loose enough for float noise, tight enough to catch layout or
# scale bugs (those push KL well above 1).
QUANT_BOUNDS = {"q8_0": (0.9, 0.02), "q4_0": (0.7, 0.15), "q4_1": (0.7, 0.15), "q4": (0.75, 0.11)}
KL_BOUND = {
    ("smollm2-135m", "q4_0"): 0.4, ("qwen2.5-0.5b", "q4_0"): 0.36, ("qwen3-0.6b", "q4_0"): 0.4,
    ("tinyllama-1.1b", "q4_0"): 0.15, ("smollm2-135m", "q4_1"): 0.33,
}


def quant_bounds(key: str, qtype: str) -> tuple[float, float]:
    top1, kl = QUANT_BOUNDS[qtype]
    if qtype == "q4_1" and (key, qtype) not in KL_BOUND:
        kl = KL_BOUND.get((key, "q4_0"), kl)  # unmeasured: q4_1 is at least as precise as q4_0
    return top1, KL_BOUND.get((key, qtype), kl)


def report(msg: str) -> None:
    print(f"\n    {msg}", end="")


def need_hf():
    if not ho.hf_available():
        pytest.skip("torch/transformers not installed")


# ---------------------------------------------------------------------------
# format-level checks (NumPy only)

def _f16(b: bytes) -> float:
    return float(np.frombuffer(b, dtype="<f2")[0])


def test_q8_0_layout_matches_format_doc():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 64)).astype(np.float32)
    x[1] = 0.0  # all-zero rows must survive (scale 0)
    blob = convert.quantize_q8_0(x)
    assert len(blob) == x.size // 32 * 34
    manual = []
    for b in range(x.size // 32):  # {f16 d; int8 q[32]}; x = d * q
        blk = blob[34 * b: 34 * b + 34]
        d = _f16(blk[:2])
        manual += [d * q for q in struct.unpack("<32b", blk[2:])]
    deq = convert.dequantize("q8_0", blob, list(x.shape))
    np.testing.assert_array_equal(deq.reshape(-1), np.asarray(manual, dtype=np.float32))
    amax = np.abs(x.reshape(-1, 32)).max(axis=1, keepdims=True)
    assert np.all(np.abs(deq - x).reshape(-1, 32) <= amax / 127 * 0.5 + 1e-3 * amax / 127 + 1e-12)


def test_q4_0_layout_matches_format_doc():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 96)).astype(np.float32)
    blob = convert.quantize_q4_0(x)
    assert len(blob) == x.size // 32 * 18
    manual = []
    for b in range(x.size // 32):  # {f16 d; u8 qs[16]}: j < 16 low nibble of qs[j], j + 16 high nibble
        blk = blob[18 * b: 18 * b + 18]
        d = _f16(blk[:2])
        qs = blk[2:]
        vals = [0.0] * 32
        for j in range(16):
            vals[j] = d * ((qs[j] & 0x0F) - 8)
            vals[j + 16] = d * ((qs[j] >> 4) - 8)
        manual += vals
    deq = convert.dequantize("q4_0", blob, list(x.shape))
    np.testing.assert_array_equal(deq.reshape(-1), np.asarray(manual, dtype=np.float32))
    rel = np.sqrt(np.mean((deq - x) ** 2) / np.mean(x ** 2))
    assert rel < 0.12, rel


def test_q4_1_layout_matches_format_doc():
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((2, 64)) + 0.5).astype(np.float32)
    blob = convert.encode_tensor(x, "q4_1")[1]
    assert len(blob) == x.size // 32 * 20
    manual = []
    for b in range(x.size // 32):  # {f16 d; f16 m; u8 qs[16]}: nibbles as q4_0; x = d * nibble + m
        blk = blob[20 * b: 20 * b + 20]
        d, m = _f16(blk[:2]), _f16(blk[2:4])
        qs = blk[4:]
        vals = [0.0] * 32
        for j in range(16):
            vals[j] = d * (qs[j] & 0x0F) + m
            vals[j + 16] = d * (qs[j] >> 4) + m
        manual += vals
    deq = convert.dequantize("q4_1", blob, list(x.shape))
    np.testing.assert_allclose(deq.reshape(-1), np.asarray(manual, dtype=np.float32), rtol=1e-6, atol=1e-6)
    rel = np.sqrt(np.mean((deq - x) ** 2) / np.mean(x ** 2))
    assert rel < 0.12, rel


def test_float_encodings_round_trip():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((4, 40)) * 3).astype(np.float32)
    for dt, tol in (("f32", 0), ("f16", 1e-3), ("bf16", 8e-3)):
        got_dt, blob = convert.encode_tensor(x, dt)
        assert got_dt == dt
        y = convert.dequantize(dt, blob, list(x.shape))
        assert np.all(np.abs(y - x) <= tol * np.abs(x) + 1e-30), dt
    # rows that are not a multiple of 32 fall back to f16 for quantized types
    assert convert.encode_tensor(x, "q8_0")[0] == "f16"


def test_bf16_rounding_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((8, 64)) * 10).astype(np.float32)
    _, blob = convert.encode_tensor(x, "bf16")
    ours = np.frombuffer(blob, dtype=np.uint16)
    theirs = torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).reshape(-1)
    np.testing.assert_array_equal(ours, theirs)


def test_rope_inv_freq_matches_transformers():
    need_hf()
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cases = [
        ({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 8192}, 128, 500000.0),   # Llama 3.1
        ({"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 8192}, 64, 500000.0),    # Llama 3.2 1B
        ({"rope_type": "linear", "factor": 4.0}, 64, 10000.0),
    ]
    for rs, hd, theta in cases:
        cfg = LlamaConfig(hidden_size=hd * 4, num_attention_heads=4, head_dim=hd, rope_theta=theta,
                          rope_scaling=dict(rs), max_position_embeddings=131072)
        theirs, _ = ROPE_INIT_FUNCTIONS[rs["rope_type"]](cfg)
        ours = rope_inv_freq(hd, theta, {**rs, "type": rs.pop("rope_type")})
        # transformers works in float32; base ** x carries up to ~|x ln base| ulp of error
        np.testing.assert_allclose(ours, theirs.double().numpy(), rtol=1e-6, atol=0)
    base = rope_inv_freq(64, 10000.0, None)
    np.testing.assert_allclose(base, 10000.0 ** (-np.arange(0, 64, 2) / 64), rtol=1e-15)


# ---------------------------------------------------------------------------
# tiny random models: every config feature, no downloads

def _write_byte_tokenizer(d: str) -> None:
    """Smallest tokenizer.json the converter accepts: 256 byte tokens, no merges."""
    b2u = convert.bytes_to_unicode()
    tj = {
        "version": "1.0", "added_tokens": [], "normalizer": None, "post_processor": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": {b2u[b]: b for b in range(256)}, "merges": []},
    }
    with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)


_COMMON = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
               intermediate_size=96, vocab_size=320, max_position_embeddings=512, bos_token_id=1, eos_token_id=2)
TINY = {
    # GQA, head_dim != dim / n_heads, llama3 RoPE scaling, GeGLU, untied output, padded vocab
    "llama-llama3-gelu": ("llama", dict(_COMMON, head_dim=24, hidden_act="gelu_pytorch_tanh", tie_word_embeddings=False,
                                        rms_norm_eps=1e-5, rope_theta=5000.0,
                                        rope_scaling={"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0,
                                                      "high_freq_factor": 4.0,
                                                      "original_max_position_embeddings": 64})),
    # MHA, linear RoPE scaling, tied embeddings
    "llama-linear-tied": ("llama", dict(_COMMON, num_key_value_heads=4, tie_word_embeddings=True, rms_norm_eps=1e-6,
                                        rope_theta=10000.0, rope_scaling={"rope_type": "linear", "factor": 2.0})),
    "mistral": ("mistral", dict(_COMMON, sliding_window=None, tie_word_embeddings=False, rope_theta=100000.0)),
    # q/k/v bias
    "qwen2-bias": ("qwen2", dict(_COMMON, use_sliding_window=False, tie_word_embeddings=True, rope_theta=1e6)),
    # per-head q/k RMSNorm, head_dim (32) != dim / n_heads (16)
    "qwen3-qknorm": ("qwen3", dict(_COMMON, head_dim=32, use_sliding_window=False, tie_word_embeddings=False,
                                   rope_theta=1e6, attention_bias=False)),
}


def _make_tiny(name: str, root: str, legacy_rope_keys: bool = True) -> tuple[object, str]:
    """Random HF model saved as a checkpoint directory; returns (model, dir)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    arch, kw = TINY[name]
    cfg = AutoConfig.for_model(arch, **kw)
    torch.manual_seed(sum(map(ord, name)))
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").float().eval()
    with torch.no_grad():
        # big enough weights that logits are far from uniform; norms and biases non-trivial
        for pname, p in model.named_parameters():
            if "norm" in pname:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif pname.endswith("bias"):
                p.copy_(0.3 * torch.randn_like(p))
            else:
                p.copy_(0.15 * torch.randn_like(p))
    d = os.path.join(root, name)
    model.save_pretrained(d)
    _write_byte_tokenizer(d)
    if legacy_rope_keys:
        # transformers 5 writes only "rope_parameters"; the converter reads the classic keys
        with open(os.path.join(d, "config.json")) as f:
            cj = json.load(f)
        rp = dict(cfg.rope_parameters)
        cj["rope_theta"] = rp.pop("rope_theta")
        cj["rope_scaling"] = rp if rp.get("rope_type", "default") != "default" else None
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cj, f)
    return model, d


@pytest.mark.parametrize("name", list(TINY))
def test_tiny_model_matches_hf(name, tmp_path):
    need_hf()
    import torch
    model, d = _make_tiny(name, str(tmp_path))
    out = str(tmp_path / f"{name}.borg")
    header = convert.convert(d, out, "f32", log=None)
    c = header["config"]
    hcfg = model.config
    assert c["head_dim"] == TINY[name][1].get("head_dim", c["dim"] // c["n_heads"])
    assert c["tie_embeddings"] == bool(hcfg.tie_word_embeddings)

    rng = np.random.default_rng(len(name))
    toks = rng.integers(0, c["vocab_size"], size=80).tolist()  # positions past original_max_position_embeddings
    with torch.no_grad():
        hf = model(torch.tensor([toks])).logits[0].numpy()
    ref = ReferenceModel(out)
    ours = ref.forward(toks)
    scale = np.abs(hf).max()
    err = np.abs(ours - hf).max()
    report(f"{name}: max|dlogit| {err:.2e} (logit scale {scale:.1f}, rel {err / scale:.1e})")
    assert err / scale < 2e-5
    _assert_same_argmax(ours, hf, scale)

    # float64 mode agrees as well (HF float32 noise is what remains)
    ours64 = ReferenceModel(out, np.float64).forward(toks)
    assert np.abs(ours64 - hf).max() / scale < 2e-5

    # incremental decoding: prefill, single tokens, a chunk, then a rewind
    inc = [ref.forward(toks[:30])]
    for t in toks[30:40]:
        inc.append(ref.forward([t], start_pos=ref.n_past))
    inc.append(ref.forward(toks[40:], start_pos=40))
    np.testing.assert_allclose(np.concatenate(inc), ours, rtol=0, atol=2e-5 * scale)
    np.testing.assert_allclose(ref.forward(toks[60:], start_pos=60), ours[60:], rtol=0, atol=2e-5 * scale)
    assert ref.n_past == len(toks)
    for bad in (-1, len(toks) + 1):  # a negative start used to wipe the cache and return NaN logits
        with pytest.raises(ValueError):
            ref.forward(toks[:2], start_pos=bad)
    assert ref.n_past == len(toks)

    # lazy dequantization gives bit-identical results to dequantizing at load time
    qpath = str(tmp_path / f"{name}-q4_0.borg")
    convert.convert(d, qpath, "q4_0", log=None)
    eager, lazy = ReferenceModel(qpath, lazy=False), ReferenceModel(qpath, lazy=True)
    assert lazy.lazy and not eager.lazy
    np.testing.assert_array_equal(eager.forward(toks), lazy.forward(toks))


@pytest.mark.parametrize("name", ["llama-llama3-gelu", "qwen3-qknorm"])
def test_tiny_half_precision_files(name, tmp_path):
    """f16 and bf16 files: HF with its matrices rounded the same way gives the same logits, eager and lazy."""
    need_hf()
    import torch
    model, d = _make_tiny(name, str(tmp_path))
    toks = np.random.default_rng(1).integers(0, 320, size=60).tolist()
    for qt, tdt in (("f16", torch.float16), ("bf16", torch.bfloat16)):
        out = str(tmp_path / f"{name}-{qt}.borg")
        convert.convert(d, out, qt, log=None)
        rounded = type(model)(model.config).eval()
        rounded.load_state_dict({k: v.to(tdt).float() if v.ndim == 2 else v for k, v in model.state_dict().items()})
        with torch.no_grad():
            hf = rounded(torch.tensor([toks])).logits[0].numpy()
        scale = np.abs(hf).max()
        for lazy in (False, True):
            ours = ReferenceModel(out, lazy=lazy).forward(toks)
            assert np.abs(ours - hf).max() / scale < 2e-5, (qt, lazy)


def test_converter_deterministic(tmp_path):
    """Converting twice gives the same bytes (the stored files are reproducible)."""
    need_hf()
    _, d = _make_tiny("qwen2-bias", str(tmp_path))
    for q in ("q8_0", "q4_0", "q4_1"):
        a, b = str(tmp_path / f"a-{q}.borg"), str(tmp_path / f"b-{q}.borg")
        convert.convert(d, a, q, log=None)
        convert.convert(d, b, q, log=None)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read(), q


def test_converter_reads_rope_parameters(tmp_path):
    need_hf()
    _, d = _make_tiny("llama-llama3-gelu", str(tmp_path), legacy_rope_keys=False)
    header = convert.convert(d, str(tmp_path / "t.borg"), "f32", log=None)
    assert header["config"]["rope_theta"] == 5000.0
    assert (header["config"]["rope_scaling"] or {}).get("type") == "llama3"


# ---------------------------------------------------------------------------
# real checkpoints

def _assert_same_argmax(ours: np.ndarray, hf: np.ndarray, scale: float) -> None:
    a, b = ours.argmax(-1), hf.argmax(-1)
    for t in np.nonzero(a != b)[0]:
        top2 = np.sort(hf[t])[-2:]
        raise AssertionError(f"argmax differs at position {t}: ours {a[t]}, HF {b[t]} "
                             f"(HF top-2 margin {top2[1] - top2[0]:.2e}, scale {scale:.1f})")


@pytest.fixture(scope="module", params=list(SPECS))
def model(request):
    spec = SPECS[request.param]
    if not os.path.isfile(os.path.join(spec.hf_path, "config.json")):
        pytest.skip(f"{spec.hf_path} not found (set HF_MODELS_DIR)")
    return spec


_HF_CACHE: dict[str, dict] = {}


@pytest.fixture(scope="module")
def hf(model):
    """HF float32 eager logits over the prompt and its greedy continuation (computed once)."""
    need_hf()
    if model.key not in _HF_CACHE:
        _REF.clear()  # never hold a reference model and a torch model at once
        gc.collect()
        _HF_CACHE[model.key] = ho.hf_run(model.hf_path)
    return _HF_CACHE[model.key]


# One reference model in memory at a time: TinyLlama alone is 4.4 GB in float32.
_REF: dict[tuple, ReferenceModel] = {}


def get_ref(spec, qtype: str, dtype=np.float32) -> ReferenceModel:
    key = (spec.key, qtype, np.dtype(dtype).name)
    if key not in _REF:
        _REF.clear()
        gc.collect()
        path = spec.borg_path(qtype)
        if not os.path.isfile(path):
            pytest.skip(f"{path} not found (set BORG_MODELS_DIR)")
        _REF[key] = ReferenceModel(path, dtype)
    return _REF[key]


def teacher_forced(hf: dict) -> tuple[list[int], np.ndarray]:
    """Prompt + greedy continuation as one sequence, with HF's logits for every position.

    gen_logits[0] is the prompt's last row, so rows T.. come from gen_logits[1:].
    """
    seq = np.concatenate([hf["tokens"], hf["gen_tokens"][:-1]]).tolist()
    return seq, np.concatenate([hf["logits"], hf["gen_logits"][1:]])


_HF_TO_BORG = {
    "input_layernorm.weight": "attn_norm", "post_attention_layernorm.weight": "ffn_norm",
    "self_attn.q_proj.weight": "attn_q", "self_attn.k_proj.weight": "attn_k",
    "self_attn.v_proj.weight": "attn_v", "self_attn.o_proj.weight": "attn_o",
    "self_attn.q_proj.bias": "attn_q.bias", "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.q_norm.weight": "attn_q_norm", "self_attn.k_norm.weight": "attn_k_norm",
    "mlp.gate_proj.weight": "ffn_gate", "mlp.up_proj.weight": "ffn_up", "mlp.down_proj.weight": "ffn_down",
}


def borg_name_for(hf_name: str) -> str:
    """HF tensor name -> .borg name, written out independently of the converter's table."""
    top = {"model.embed_tokens.weight": "tok_embd", "model.norm.weight": "output_norm", "lm_head.weight": "output"}
    if hf_name in top:
        return top[hf_name]
    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", hf_name)
    assert m and m.group(2) in _HF_TO_BORG, f"unexpected HF tensor {hf_name}"
    return f"blk.{m.group(1)}.{_HF_TO_BORG[m.group(2)]}"


def hf_tensors(model_dir: str):
    """Yield (name, float32 array) for every tensor of the checkpoint, via the safetensors library."""
    import torch  # noqa: F401  (bf16 has no NumPy dtype, so read through torch)
    from safetensors import safe_open
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    for fn in files:
        with safe_open(os.path.join(model_dir, fn), framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name).float().numpy()


def _borg_or_skip(spec, qtype):
    path = spec.borg_path(qtype)
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    return convert.read_borg(path)


def test_converter_config(model):
    header, _, _ = _borg_or_skip(model, "f32")
    with open(os.path.join(model.hf_path, "config.json")) as f:
        hc = json.load(f)
    c = header["config"]
    assert c["dim"] == hc["hidden_size"]
    assert c["n_layers"] == hc["num_hidden_layers"]
    assert c["n_heads"] == hc["num_attention_heads"]
    assert c["n_kv_heads"] == hc.get("num_key_value_heads", hc["num_attention_heads"])
    assert c["head_dim"] == (hc.get("head_dim") or hc["hidden_size"] // hc["num_attention_heads"])
    assert c["hidden_dim"] == hc["intermediate_size"]
    assert c["vocab_size"] == hc["vocab_size"]
    assert c["norm_eps"] == hc["rms_norm_eps"]
    assert c["rope_theta"] == float(hc.get("rope_theta", (hc.get("rope_parameters") or {}).get("rope_theta")))
    assert c["tie_embeddings"] == hc.get("tie_word_embeddings", False)
    assert c["qk_norm"] == (hc["model_type"] == "qwen3")
    assert c["attn_bias"] == (hc["model_type"] == "qwen2")
    assert c["act"] == "silu" and c["rope_scaling"] is None


def test_converter_f32_bit_exact(model):
    """Every checkpoint tensor lands in the f32 file under the right name, bit for bit."""
    need_hf()
    header, mm, ds = _borg_or_skip(model, "f32")
    seen = set()
    for name, x in hf_tensors(model.hf_path):
        if name.endswith("rotary_emb.inv_freq"):
            continue
        bn = borg_name_for(name)
        if bn == "output" and header["config"]["tie_embeddings"]:
            np.testing.assert_array_equal(x, convert.load_tensor(header, mm, ds, "tok_embd"))
            continue
        t = header["tensors"][bn]
        assert t["dtype"] == "f32" and t["shape"] == list(x.shape), (bn, t)
        assert t["offset"] % 64 == 0
        y = convert.load_tensor(header, mm, ds, bn)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), f"{bn} differs from {name}"
        seen.add(bn)
    weights = {n for n in header["tensors"] if not n.startswith("tokenizer.")}
    assert weights == seen, f"tensors without an HF source: {sorted(weights - seen)[:5]}"
    report(f"{model.key}: {len(seen)} tensors bit-identical to the checkpoint")


@pytest.mark.parametrize("qtype", QTYPES)
def test_converter_quantized(model, qtype):
    """Quantized files: allowed dtype per tensor, 1-D tensors bit-exact, and every matrix close
    to the checkpoint. Checks the stored values, not a particular rounding procedure, so files
    from calibrated (importance-weighted) or older quantizers pass as long as they are faithful."""
    need_hf()
    header, mm, ds = _borg_or_skip(model, qtype)
    four_bit = qtype in ("q4_0", "q4_1", "q4")
    worst = {}
    for name, x in hf_tensors(model.hf_path):
        bn = borg_name_for(name)
        if bn == "output" and header["config"]["tie_embeddings"]:
            continue
        t = header["tensors"][bn]
        y = convert.load_tensor(header, mm, ds, bn)
        if x.ndim == 1:
            assert t["dtype"] == "f32"
            np.testing.assert_array_equal(x, y)
            continue
        if x.shape[-1] % 32:
            allowed = {"f16"}
        elif bn in ("tok_embd", "output") or not four_bit:
            allowed = {"q8_0"}  # 4-bit files keep embeddings/output at 8 bits
        else:
            allowed = {"q4_0", "q4_1", "q8_0"} if qtype == "q4" else {qtype, "q8_0"}  # q8_0: --keep-q8
        assert t["dtype"] in allowed, (bn, t["dtype"], allowed)
        rel = float(np.sqrt(np.mean((y - x) ** 2) / np.mean(x ** 2)))
        worst[t["dtype"]] = max(worst.get(t["dtype"], 0.0), rel)
    report(f"{model.key} {qtype} ({header.get('quant')}): worst per-tensor relative RMS error " +
           ", ".join(f"{k} {v:.4f}" for k, v in sorted(worst.items())))
    # a misplaced block, nibble order or scale shows up as an error near 1
    assert worst.get("q8_0", 0) < 0.01
    assert worst.get("q4_0", 0) < 0.2 and worst.get("q4_1", 0) < 0.2


def test_logits_match_hf(model, hf):
    ref = get_ref(model, "f32")
    ours = ref.forward(hf["tokens"].tolist())
    theirs = hf["logits"]
    scale = float(np.abs(theirs).max())
    err = np.abs(ours - theirs)
    report(f"{model.key} f32: T={len(ours)} max|dlogit| {err.max():.2e} mean {err.mean():.1e} "
           f"(logit scale {scale:.1f}, rel {err.max() / scale:.1e})")
    assert ours.shape == theirs.shape and ours.dtype == np.float32
    assert err.max() / scale < 2e-5
    _assert_same_argmax(ours, theirs, scale)


def test_incremental_matches_full(model, hf):
    ref = get_ref(model, "f32")
    seq, hf_logits = teacher_forced(hf)
    full = ref.forward(seq)
    scale = float(np.abs(full).max())
    n = len(hf["tokens"]) - 6
    parts = [ref.forward(seq[:n])]
    for i in range(n, len(seq)):
        parts.append(ref.forward([seq[i]], start_pos=i))
    inc = np.concatenate(parts)
    err = np.abs(inc - full).max()
    # positions decoded by HF itself with its KV cache: our full forward against them
    err_hf = np.abs(full - hf_logits).max()
    report(f"{model.key}: incremental vs full {err:.2e}; full vs HF (prompt + KV-cached greedy steps) "
           f"{err_hf:.2e}, scale {scale:.1f}")
    assert err / scale < 2e-5
    assert err_hf / scale < 2e-5
    _assert_same_argmax(full, hf_logits, scale)
    # rewinding: re-evaluating the tail after truncation reproduces it
    k = len(seq) - 5
    np.testing.assert_allclose(ref.forward(seq[k:], start_pos=k), full[k:], rtol=0, atol=2e-5 * scale)


def test_greedy_matches_hf(model, hf):
    ref = get_ref(model, "f32")
    out = ref.generate(hf["tokens"].tolist(), len(hf["gen_tokens"]), ignore_eos=True)
    assert out == hf["gen_tokens"].tolist()


@pytest.fixture(scope="module")
def hf_long(model):
    """HF float32 eager logits over a 200-token prompt and its KV-cached greedy continuation.
    Not kept in _HF_CACHE: at Qwen's vocabulary this is 120 MB per model."""
    need_hf()
    _REF.clear()
    gc.collect()
    return ho.hf_run(model.hf_path, ho.long_prompt_tokens(model.hf_path), n_gen=ho.LONG_N_GEN)


def test_long_prompt_matches_hf(model, hf_long):
    """The ~22-token prompt above leaves RoPE angles, the causal mask and the KV cache barely
    exercised. Here every position of a 200-token prompt is checked against HF, then HF's
    KV-cached greedy steps are reproduced by incremental decoding after the long prefill."""
    ref = get_ref(model, "f32")
    T = len(hf_long["tokens"])
    seq, theirs = teacher_forced(hf_long)
    ours = ref.forward(seq)
    scale = float(np.abs(theirs).max())
    err = np.abs(ours - theirs).max(axis=1)
    steps = [ref.forward(seq[:T])[-1:]]
    for i in range(T, len(seq)):
        steps.append(ref.forward([seq[i]], start_pos=i))
    inc = np.concatenate(steps)
    err_inc = np.abs(inc - hf_long["gen_logits"]).max()
    report(f"{model.key}: T={T} max|dlogit| {err.max():.2e} (last 20 prompt positions {err[T - 20:T].max():.2e}); "
           f"incremental after prefill vs HF KV-cached steps {err_inc:.2e}; scale {scale:.1f}")
    assert err.max() / scale < 2e-5
    _assert_same_argmax(ours, theirs, scale)
    assert err_inc / scale < 2e-5
    assert inc.argmax(-1).tolist() == hf_long["gen_tokens"].tolist()


@pytest.mark.parametrize("qtype", QTYPES)
def test_quantized_quality(model, hf, qtype):
    ref = get_ref(model, qtype)
    seq, theirs = teacher_forced(hf)
    ours = ref.forward(seq)
    T = len(hf["tokens"])
    top1 = ours.argmax(-1) == theirs.argmax(-1)
    kl = kl_divergence(theirs, ours)
    report(f"{model.key} {qtype} ({ref.header.get('quant')}): top-1 {top1.mean():.3f} (prompt {top1[:T].mean():.3f}), "
           f"KL mean {kl.mean():.4f} (prompt {kl[:T].mean():.4f}) max {kl.max():.3f} over {len(seq)} positions")
    min_top1, max_kl = quant_bounds(model.key, qtype)
    assert top1.mean() >= min_top1
    assert kl.mean() < max_kl


def test_float64_mode(model, hf):
    """High-precision mode, on the smallest model only: HF float32 noise is the same size as ours."""
    if model.key != "smollm2-135m":
        pytest.skip("float64 mode is checked on SmolLM2 only (memory)")
    ref32 = get_ref(model, "f32").forward(hf["tokens"].tolist())
    ref64 = get_ref(model, "f32", np.float64).forward(hf["tokens"].tolist())
    assert ref64.dtype == np.float64
    scale = float(np.abs(ref64).max())
    e_hf = np.abs(hf["logits"] - ref64).max()
    e_32 = np.abs(ref32 - ref64).max()
    report(f"{model.key}: vs float64 reference: HF float32 {e_hf:.2e}, reference float32 {e_32:.2e}")
    assert e_hf / scale < 2e-5 and e_32 / scale < 2e-5
    _assert_same_argmax(ref64, hf["logits"], scale)
