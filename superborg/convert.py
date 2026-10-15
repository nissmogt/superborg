"""Convert a Hugging Face checkpoint (safetensors + tokenizer.json) into a .borg file.

Only NumPy is required. The converter reads the published architecture config and
tensor layout, maps every tensor to the engine's canonical names, quantizes the
large matrices and packs the tokenizer tables. See docs/FORMAT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

MAGIC = b"BORG"
VERSION = 1
ALIGN = 64
QK = 32  # quantization block size

SUPPORTED_ARCHS = {"llama", "mistral", "qwen2", "qwen3"}

# Pre-tokenizer regexes as published in tokenizer.json files.
QWEN2_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
               r"|\s*[\r\n]+|\s+(?!\S)|\s+")
LLAMA3_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
GPT2_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


# ---------------------------------------------------------------------------
# safetensors reading (no torch)

_ST_DTYPES = {
    "F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}


class SafeTensors:
    """Lazily maps one or more .safetensors files and returns float32 arrays."""

    def __init__(self, model_dir: str):
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {model_dir}")
        self.entries: dict[str, tuple[np.memmap, dict, int]] = {}
        for fn in files:
            path = os.path.join(model_dir, fn)
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                self.entries[name] = (mm, info, 8 + n)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def keys(self):
        return self.entries.keys()

    def get(self, name: str) -> np.ndarray:
        mm, info, base = self.entries[name]
        start, end = info["data_offsets"]
        dt = info["dtype"]
        if dt not in _ST_DTYPES:
            raise ValueError(f"unsupported safetensors dtype {dt} for {name}")
        raw = np.frombuffer(mm[base + start: base + end], dtype=_ST_DTYPES[dt]).reshape(info["shape"])
        if dt == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# quantization

def quantize_q8_0(x: np.ndarray) -> bytes:
    """ggml-compatible Q8_0: per 32 values, f16 scale + 32 int8."""
    rows, cols = x.shape
    assert cols % QK == 0
    b = x.reshape(-1, QK).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.clip(np.rint(b * inv[:, None]), -127, 127).astype(np.int8)
    out = np.empty(b.shape[0], dtype=[("d", "<f2"), ("q", "i1", (QK,))])
    out["d"] = d.astype(np.float16)
    out["q"] = q
    return out.tobytes()


def _block_weights(imp: np.ndarray | None, cols: int) -> np.ndarray:
    """Per-column importance reshaped to broadcast over [rows, blocks, QK]."""
    if imp is None:
        return np.ones((1, cols // QK, QK), dtype=np.float32)
    w = np.asarray(imp, dtype=np.float32).reshape(1, cols // QK, QK)
    return np.maximum(w, w.mean() * 1e-3) + 1e-12  # keep every column at least slightly weighted


def _f16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float32)


def _safe_inv(d: np.ndarray) -> np.ndarray:
    return np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0).astype(np.float32)


ROW_CHUNK = 512  # quantize big matrices in row slices to bound temporary memory


def quantize_q4_0(x: np.ndarray, imp: np.ndarray | None = None) -> bytes:
    """Q4_0 (ggml layout): x = d * (q - 8). The scale is chosen per block from several
    candidates, each refined by a weighted least-squares fit, to minimise the squared error
    weighted by `imp` (mean squared activation per input column, if given)."""
    rows, cols = x.shape
    assert cols % QK == 0
    w = _block_weights(imp, cols)
    parts = []
    for r0 in range(0, rows, ROW_CHUNK):
        b = x[r0:r0 + ROW_CHUNK].reshape(-1, cols // QK, QK).astype(np.float32)
        idx = np.abs(b).argmax(axis=2)
        vmax = np.take_along_axis(b, idx[..., None], axis=2)[..., 0]  # signed value of largest magnitude
        best_err = np.full(b.shape[:2], np.inf, dtype=np.float32)
        best_d = np.zeros(b.shape[:2], dtype=np.float32)
        best_q = np.zeros(b.shape, dtype=np.uint8)
        for div in (-8.0, -7.75, -7.5, -7.25, -7.0, -6.75, -6.5, 7.0, 7.25, 7.5):
            d0 = _f16(vmax / div)
            q0 = np.clip(np.rint(b * _safe_inv(d0)[..., None]), -8, 7)
            num = (w * b * q0).sum(axis=2)
            den = (w * q0 * q0).sum(axis=2)
            d1 = _f16(np.where(den > 0, num / np.where(den > 0, den, 1), d0))
            q1 = np.clip(np.rint(b * _safe_inv(d1)[..., None]), -8, 7)
            for d, q in ((d0, q0), (d1, q1)):
                err = (w * (q * d[..., None] - b) ** 2).sum(axis=2)
                better = err < best_err
                best_err = np.where(better, err, best_err)
                best_d = np.where(better, d, best_d)
                best_q = np.where(better[..., None], (q + 8).astype(np.uint8), best_q)
        n = b.shape[0] * b.shape[1]
        qb = best_q.reshape(n, QK)
        out = np.empty(n, dtype=[("d", "<f2"), ("qs", "u1", (16,))])
        out["d"] = best_d.reshape(n).astype(np.float16)
        out["qs"] = (qb[:, :16] | (qb[:, 16:] << 4)).astype(np.uint8)
        parts.append(out.tobytes())
    return b"".join(parts)


def quantize_q4_1(x: np.ndarray, imp: np.ndarray | None = None) -> bytes:
    """Q4_1: x = d * q + m with q in 0..15, f16 d and m per 32 values (5 bits per weight).
    (d, m) start from the block range and are refined by alternating rounding with a
    weighted least-squares fit; the best fit seen is kept."""
    rows, cols = x.shape
    assert cols % QK == 0
    w = _block_weights(imp, cols)
    parts = []
    for r0 in range(0, rows, ROW_CHUNK):
        b = x[r0:r0 + ROW_CHUNK].reshape(-1, cols // QK, QK).astype(np.float32)
        mn, mx = b.min(axis=2), b.max(axis=2)
        d = (mx - mn) / 15.0
        m = mn.copy()
        sw = np.broadcast_to(w, b.shape).sum(axis=2)
        sx = (w * b).sum(axis=2)
        best_err = np.full(b.shape[:2], np.inf, dtype=np.float32)
        best_d = np.zeros(b.shape[:2], dtype=np.float32)
        best_m = np.zeros(b.shape[:2], dtype=np.float32)
        best_q = np.zeros(b.shape, dtype=np.uint8)
        for it in range(6):
            df, mf = _f16(d), _f16(m)
            q = np.clip(np.rint((b - mf[..., None]) * _safe_inv(df)[..., None]), 0, 15)
            err = (w * (q * df[..., None] + mf[..., None] - b) ** 2).sum(axis=2)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_d = np.where(better, df, best_d)
            best_m = np.where(better, mf, best_m)
            best_q = np.where(better[..., None], q.astype(np.uint8), best_q)
            if it == 5:
                break
            # least squares for (d, m) given q
            sq = (w * q).sum(axis=2)
            sqq = (w * q * q).sum(axis=2)
            sqx = (w * q * b).sum(axis=2)
            det = sw * sqq - sq * sq
            ok = det > 1e-30
            dn = np.where(ok, (sw * sqx - sq * sx) / np.where(ok, det, 1), d)
            mnew = np.where(ok, (sqq * sx - sq * sqx) / np.where(ok, det, 1), m)
            d = np.where(dn > 0, dn, d)
            m = np.where(dn > 0, mnew, m)
        n = b.shape[0] * b.shape[1]
        qb = best_q.reshape(n, QK)
        out = np.empty(n, dtype=[("d", "<f2"), ("m", "<f2"), ("qs", "u1", (16,))])
        out["d"] = best_d.reshape(n).astype(np.float16)
        out["m"] = best_m.reshape(n).astype(np.float16)
        out["qs"] = (qb[:, :16] | (qb[:, 16:] << 4)).astype(np.uint8)
        parts.append(out.tobytes())
    return b"".join(parts)


QUANT_BYTES = {"q8_0": 34, "q4_0": 18, "q4_1": 20}  # bytes per block of QK values
FOUR_BIT = ("q4_0", "q4_1")
# Tensors that dominate the 4-bit error: with these at 8 bits and calibrated q4_1 for the rest,
# Qwen2.5-0.5B loses 0.7% perplexity versus 11% for plain q4_0 (bench/quant_eval.py).
Q4_RECIPE_KEEP = ("attn_v", "ffn_down")


def dequantize(dtype: str, data: bytes, shape: list[int]) -> np.ndarray:
    """Inverse of the encoders above (used by tests and the reference forward pass)."""
    if dtype == "f32":
        return np.frombuffer(data, dtype=np.float32).reshape(shape)
    if dtype == "f16":
        return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "bf16":
        return (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "q8_0":
        blk = np.frombuffer(data, dtype=[("d", "<f2"), ("q", "i1", (QK,))])
        return (blk["d"].astype(np.float32)[:, None] * blk["q"].astype(np.float32)).reshape(shape)
    if dtype == "q4_0":
        blk = np.frombuffer(data, dtype=[("d", "<f2"), ("qs", "u1", (16,))])
        lo = (blk["qs"] & 0x0F).astype(np.int8) - 8
        hi = (blk["qs"] >> 4).astype(np.int8) - 8
        q = np.concatenate([lo, hi], axis=1).astype(np.float32)
        return (blk["d"].astype(np.float32)[:, None] * q).reshape(shape)
    if dtype == "q4_1":
        blk = np.frombuffer(data, dtype=[("d", "<f2"), ("m", "<f2"), ("qs", "u1", (16,))])
        q = np.concatenate([blk["qs"] & 0x0F, blk["qs"] >> 4], axis=1).astype(np.float32)
        return (blk["d"].astype(np.float32)[:, None] * q + blk["m"].astype(np.float32)[:, None]).reshape(shape)
    raise ValueError(dtype)


def encode_tensor(x: np.ndarray, dtype: str, imp: np.ndarray | None = None) -> tuple[str, bytes]:
    if dtype in QUANT_BYTES and (x.ndim != 2 or x.shape[1] % QK != 0):
        dtype = "f16"
    if dtype == "f32":
        return dtype, np.ascontiguousarray(x, dtype=np.float32).tobytes()
    if dtype == "f16":
        return dtype, np.ascontiguousarray(x, dtype=np.float16).tobytes()
    if dtype == "bf16":
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return dtype, rounded.tobytes()
    if dtype == "q8_0":
        return dtype, quantize_q8_0(x)
    if dtype == "q4_0":
        return dtype, quantize_q4_0(x, imp)
    if dtype == "q4_1":
        return dtype, quantize_q4_1(x, imp)
    raise ValueError(f"unknown dtype {dtype}")


# ---------------------------------------------------------------------------
# calibration (importance matrix)

# which activation site feeds each weight matrix (see borg_ctx_stats in engine/borg.h)
_SITE = {"attn_q": 0, "attn_k": 0, "attn_v": 0, "attn_o": 1, "ffn_gate": 2, "ffn_up": 2, "ffn_down": 3}
CALIBRATION_TEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration.txt")


def collect_importance(model_path: str, text: str, max_tokens: int = 8192, threads: int = 0,
                       log=print) -> dict[tuple[int, int], np.ndarray]:
    """Mean squared activation per input column of every matrix, measured by running the
    engine on `text`. Returns {(layer, site): array}."""
    from .engine import Model  # needs the compiled engine

    m = Model(model_path, threads=threads)
    n_ctx = min(m.max_seq_len, 2048)
    ctx = m.context(n_ctx=n_ctx)
    ctx.enable_stats()
    toks = m.tokenize(text)[:max_tokens]
    for s in range(0, len(toks), n_ctx):
        ctx.reset()
        ctx.decode(toks[s:s + n_ctx], logits=None)
    n = ctx.stats_tokens()
    out = {}
    for layer in range(m.config["n_layers"]):
        for site in range(4):
            out[(layer, site)] = ctx.stats(layer, site) / max(n, 1)
    if log:
        log(f"calibration: {n} tokens")
    ctx.close()
    m.close()
    return out


# ---------------------------------------------------------------------------
# config

def convert_config(cfg: dict) -> tuple[str, dict]:
    arch = cfg.get("model_type")
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"unsupported model_type {arch!r}; supported: {sorted(SUPPORTED_ARCHS)}")
    dim = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or dim // n_heads
    act = cfg.get("hidden_act", "silu")
    if act not in ("silu", "gelu_pytorch_tanh"):
        raise ValueError(f"unsupported activation {act}")
    rs = cfg.get("rope_scaling")
    rope_scaling = None
    if rs:
        kind = rs.get("rope_type", rs.get("type"))
        if kind == "llama3":
            rope_scaling = {"type": "llama3", "factor": float(rs["factor"]),
                            "low_freq_factor": float(rs["low_freq_factor"]),
                            "high_freq_factor": float(rs["high_freq_factor"]),
                            "original_max_position_embeddings": int(rs["original_max_position_embeddings"])}
        elif kind == "linear":
            rope_scaling = {"type": "linear", "factor": float(rs["factor"])}
        elif kind in ("default", None):
            rope_scaling = None
        else:
            raise ValueError(f"unsupported rope_scaling type {kind}")
    if cfg.get("use_sliding_window"):
        raise ValueError("sliding-window attention is not supported")
    out = {
        "dim": dim,
        "n_layers": cfg["num_hidden_layers"],
        "n_heads": n_heads,
        "n_kv_heads": cfg.get("num_key_value_heads", n_heads),
        "head_dim": head_dim,
        "hidden_dim": cfg["intermediate_size"],
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": cfg.get("max_position_embeddings", 4096),
        "norm_eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(cfg.get("rope_theta", 10000.0)),
        "rope_scaling": rope_scaling,
        "attn_bias": bool(arch == "qwen2" or cfg.get("attention_bias", False)),
        "qk_norm": arch == "qwen3",
        "tie_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "act": "silu" if act == "silu" else "gelu_tanh",
    }
    if cfg.get("mlp_bias"):
        raise ValueError("MLP bias is not supported")
    return arch, out


# ---------------------------------------------------------------------------
# tokenizer

def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _detect_pretokenizer(pt: dict | None) -> tuple[bool, str]:
    """Returns (byte_level, pretokenizer_name)."""
    if pt is None:
        return False, "none"
    kind = pt.get("type")
    if kind == "ByteLevel":
        if not pt.get("use_regex", True):
            raise ValueError("ByteLevel pre-tokenizer without regex is not supported")
        if pt.get("add_prefix_space"):
            raise ValueError("ByteLevel add_prefix_space is not supported")
        return True, "gpt2"
    if kind == "Metaspace":
        return False, "metaspace"
    if kind == "Sequence":
        items = pt["pretokenizers"]
        types = [p["type"] for p in items]
        if types == ["Digits", "ByteLevel"] and items[0].get("individual_digits") and items[1].get("use_regex", True) \
                and not items[1].get("add_prefix_space"):
            return True, "smollm"
        if types == ["Split", "ByteLevel"] and not items[1].get("use_regex", True):
            pattern = items[0]["pattern"].get("Regex")
            if items[0].get("behavior") != "Isolated" or items[0].get("invert"):
                raise ValueError("unsupported Split behaviour")
            if pattern == QWEN2_REGEX:
                return True, "qwen2"
            if pattern == LLAMA3_REGEX:
                return True, "llama3"
            if pattern == GPT2_REGEX:
                return True, "gpt2"
            raise ValueError(f"unsupported pre-tokenizer regex: {pattern!r}")
    raise ValueError(f"unsupported pre-tokenizer: {json.dumps(pt)[:300]}")


def _token_id(tok_cfg_value, vocab: dict[str, int]) -> int:
    if tok_cfg_value is None:
        return -1
    if isinstance(tok_cfg_value, dict):
        tok_cfg_value = tok_cfg_value.get("content")
    return vocab.get(tok_cfg_value, -1)


def _chat_format(template: str | None) -> str:
    if not template:
        return "plain"
    if "<|im_start|>" in template:
        return "qwen3" if "<think>" in template else "chatml"
    if "<|start_header_id|>" in template:
        return "llama3"
    if "<|user|>" in template:
        return "zephyr"
    if "[INST]" in template:
        return "mistral"
    return "plain"


def convert_tokenizer(model_dir: str, cfg: dict) -> tuple[dict, dict[str, tuple[str, list[int], bytes]]]:
    with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    tc = {}
    p = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            tc = json.load(f)
    gen = {}
    p = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            gen = json.load(f)

    model = tj["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"only BPE tokenizers are supported, got {model.get('type')}")
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise ValueError("BPE continuing_subword_prefix/end_of_word_suffix are not supported")

    byte_level, pretok = _detect_pretokenizer(tj.get("pre_tokenizer"))
    norm = tj.get("normalizer")
    nfc = False
    spm_prefix = False
    if norm is not None:
        kinds = [n["type"] for n in norm["normalizers"]] if norm["type"] == "Sequence" else [norm["type"]]
        items = norm["normalizers"] if norm["type"] == "Sequence" else [norm]
        for it in items:
            if it["type"] == "NFC":
                nfc = True
            elif it["type"] == "Prepend" and it["prepend"] == "▁":
                spm_prefix = True
            elif it["type"] == "Replace" and it["pattern"].get("String") == " " and it["content"] == "▁":
                pass
            else:
                raise ValueError(f"unsupported normalizer {kinds}")
        if not byte_level and "Replace" not in kinds:
            raise ValueError("SentencePiece tokenizer without space replacement is not supported")
    if pretok == "metaspace":
        pt = tj["pre_tokenizer"]
        if pt.get("split", False):
            raise ValueError("Metaspace with split=true is not supported")
        scheme = pt.get("prepend_scheme", "always" if pt.get("add_prefix_space", True) else "never")
        if scheme not in ("always", "first", "never"):
            raise ValueError(f"unsupported Metaspace prepend_scheme {scheme}")
        spm_prefix = scheme != "never"
        pretok = "none"
    if not byte_level and pretok != "none":
        raise ValueError("inconsistent tokenizer: SentencePiece vocabulary with a regex pre-tokenizer")

    vocab: dict[str, int] = dict(model["vocab"])
    added = tj.get("added_tokens", [])
    for a in added:
        vocab.setdefault(a["content"], a["id"])
    n_vocab = max(max(vocab.values()), max((a["id"] for a in added), default=-1)) + 1
    id_to_str: list[str | None] = [None] * n_vocab
    for s, i in vocab.items():
        if id_to_str[i] is None or s in model["vocab"]:
            id_to_str[i] = s
    added_ids = {a["id"]: a for a in added}
    for i, a in added_ids.items():
        id_to_str[i] = a["content"]

    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}
    token_bytes: list[bytes] = []
    token_type = np.zeros(n_vocab, dtype=np.uint8)
    for i, s in enumerate(id_to_str):
        if s is None:
            token_bytes.append(b"")
            token_type[i] = 4
            continue
        if i in added_ids:
            token_bytes.append(s.encode("utf-8"))
            token_type[i] = 2 if added_ids[i].get("special", False) else 3
            continue
        if byte_level:
            try:
                token_bytes.append(bytes(u2b[c] for c in s))
            except KeyError:
                # not a byte-level string (rare: tokens added to the vocab directly)
                token_bytes.append(s.encode("utf-8"))
        else:
            if len(s) == 6 and s.startswith("<0x") and s.endswith(">"):
                token_bytes.append(bytes([int(s[3:5], 16)]))
                token_type[i] = 1
            else:
                token_bytes.append(s.encode("utf-8"))

    offsets = np.zeros(n_vocab + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([len(b) for b in token_bytes])
    blob = b"".join(token_bytes)

    merges_raw = model.get("merges", [])
    merges = np.zeros((len(merges_raw), 3), dtype=np.uint32)
    kept = 0
    for m in merges_raw:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        ia, ib, iab = vocab.get(a), vocab.get(b), vocab.get(a + b)
        if ia is None or ib is None or iab is None:
            continue  # merge whose result is not in the vocabulary can never fire
        merges[kept] = (ia, ib, iab)
        kept += 1
    merges = merges[:kept]

    # BOS handling: tokenizer.json post-processor decides whether BOS is added.
    add_bos = False
    pp = tj.get("post_processor")
    bos_from_pp = -1
    if pp and pp.get("type") == "TemplateProcessing":
        single = pp.get("single", [])
        if single and "SpecialToken" in single[0]:
            add_bos = True
            bos_from_pp = vocab.get(single[0]["SpecialToken"]["id"], -1)
        if single and "SpecialToken" in single[-1]:
            raise ValueError("post-processors that append tokens are not supported")
    elif pp and pp.get("type") == "Sequence":
        for sub in pp.get("processors", []):
            if sub.get("type") == "TemplateProcessing":
                single = sub.get("single", [])
                if single and "SpecialToken" in single[0]:
                    add_bos = True
                    bos_from_pp = vocab.get(single[0]["SpecialToken"]["id"], -1)

    bos_id = bos_from_pp if bos_from_pp >= 0 else _token_id(tc.get("bos_token"), vocab)
    if bos_id < 0 and cfg.get("bos_token_id") is not None and not add_bos:
        bos_id = int(cfg["bos_token_id"])
    eos_id = _token_id(tc.get("eos_token"), vocab)
    eos_ids: list[int] = []
    for src in (gen.get("eos_token_id"), cfg.get("eos_token_id")):
        if src is None:
            continue
        for e in (src if isinstance(src, list) else [src]):
            if e not in eos_ids:
                eos_ids.append(int(e))
    if eos_id >= 0 and eos_id not in eos_ids:
        eos_ids.insert(0, eos_id)
    if eos_id < 0 and eos_ids:
        eos_id = eos_ids[0]
    pad_id = _token_id(tc.get("pad_token"), vocab)
    unk_id = vocab.get(model["unk_token"], -1) if model.get("unk_token") else -1

    template = tc.get("chat_template")
    if isinstance(template, list):  # named templates
        template = next((t["template"] for t in template if t.get("name") == "default"), template[0]["template"])

    meta = {
        "byte_level": byte_level,
        "pretokenizer": pretok,
        "nfc": nfc,
        "spm_add_prefix_space": spm_prefix,
        "byte_fallback": bool(model.get("byte_fallback", False)),
        "ignore_merges": bool(model.get("ignore_merges", False)),
        "add_bos": add_bos,
        "add_eos": False,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "eos_ids": eos_ids,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "added_tokens": sorted(added_ids),
        "chat_template": template,
        "chat_format": _chat_format(template),
        "n_vocab": n_vocab,
        "n_merges": int(merges.shape[0]),
    }
    tensors = {
        "tokenizer.vocab": ("u8", [len(blob)], blob),
        "tokenizer.vocab_offsets": ("u32", [n_vocab + 1], offsets.tobytes()),
        "tokenizer.token_type": ("u8", [n_vocab], token_type.tobytes()),
        "tokenizer.merges": ("u32", [int(merges.shape[0]), 3], merges.tobytes()),
    }
    return meta, tensors


# ---------------------------------------------------------------------------
# weights

def _layer_map(i: int) -> dict[str, str]:
    p = f"model.layers.{i}."
    b = f"blk.{i}."
    return {
        p + "input_layernorm.weight": b + "attn_norm",
        p + "self_attn.q_proj.weight": b + "attn_q",
        p + "self_attn.k_proj.weight": b + "attn_k",
        p + "self_attn.v_proj.weight": b + "attn_v",
        p + "self_attn.q_proj.bias": b + "attn_q.bias",
        p + "self_attn.k_proj.bias": b + "attn_k.bias",
        p + "self_attn.v_proj.bias": b + "attn_v.bias",
        p + "self_attn.q_norm.weight": b + "attn_q_norm",
        p + "self_attn.k_norm.weight": b + "attn_k_norm",
        p + "self_attn.o_proj.weight": b + "attn_o",
        p + "post_attention_layernorm.weight": b + "ffn_norm",
        p + "mlp.gate_proj.weight": b + "ffn_gate",
        p + "mlp.up_proj.weight": b + "ffn_up",
        p + "mlp.down_proj.weight": b + "ffn_down",
    }


def weight_plan(st: SafeTensors, config: dict) -> list[tuple[str, str]]:
    """List of (hf_name, borg_name) in file order; raises on unknown or missing tensors."""
    plan = [("model.embed_tokens.weight", "tok_embd")]
    for i in range(config["n_layers"]):
        for hf, bn in _layer_map(i).items():
            if hf in st:
                plan.append((hf, bn))
            elif not (bn.endswith(".bias") or bn.endswith(("attn_q_norm", "attn_k_norm"))):
                raise KeyError(f"missing tensor {hf}")
    plan.append(("model.norm.weight", "output_norm"))
    if "lm_head.weight" in st and not config["tie_embeddings"]:
        plan.append(("lm_head.weight", "output"))
    elif "lm_head.weight" not in st:
        config["tie_embeddings"] = True
    known = {hf for hf, _ in plan} | {"lm_head.weight"}
    unknown = [k for k in st.keys() if k not in known and not k.endswith("rotary_emb.inv_freq")]
    if unknown:
        raise KeyError(f"unexpected tensors (unsupported architecture variant?): {unknown[:5]}")
    if config["attn_bias"] and not any(bn.endswith("attn_q.bias") for _, bn in plan):
        config["attn_bias"] = False
    if config["qk_norm"] and not any(bn.endswith("attn_q_norm") for _, bn in plan):
        raise KeyError("qk_norm architecture without q_norm tensors")
    return plan


def tensor_dtype(borg_name: str, ndim: int, qtype: str, keep_q8: tuple[str, ...] = ()) -> str:
    if ndim == 1:
        return "f32"
    if qtype in FOUR_BIT and borg_name in ("tok_embd", "output"):
        return "q8_0"  # embeddings/output stay at 8 bits: small cost, large quality gain
    if qtype in FOUR_BIT and any(borg_name.endswith(k) for k in keep_q8):
        return "q8_0"
    return qtype


def convert(model_dir: str, out_path: str, qtype: str = "q8_0", source: str | None = None,
            log=print, keep_q8: tuple[str, ...] = (), calibration: str | None = None) -> dict:
    """Write `out_path`. With a 4-bit qtype and `calibration` (text, or "builtin"), the
    model is first converted to q8_0, run on the text to measure activation statistics,
    and the 4-bit rounding then minimises the error weighted by them."""
    if qtype == "q4":  # recommended 4-bit recipe
        qtype = "q4_1"
        keep_q8 = keep_q8 or Q4_RECIPE_KEEP
        calibration = calibration or "builtin"
    importance = None
    if calibration is not None and qtype in FOUR_BIT:
        text = calibration
        if calibration == "builtin":
            with open(CALIBRATION_TEXT, encoding="utf-8") as f:
                text = f.read()
        tmp_q8 = out_path + ".calib.borg"
        try:
            convert(model_dir, tmp_q8, "q8_0", source=source, log=None)
            importance = collect_importance(tmp_q8, text, log=log)
        finally:
            if os.path.exists(tmp_q8):
                os.remove(tmp_q8)
    with open(os.path.join(model_dir, "config.json")) as f:
        hf_cfg = json.load(f)
    arch, config = convert_config(hf_cfg)
    st = SafeTensors(model_dir)
    plan = weight_plan(st, config)
    tok_meta, tok_tensors = convert_tokenizer(model_dir, hf_cfg)

    # First pass: compute dtypes and sizes so the header can be written up front.
    entries = []
    offset = 0
    for hf, bn in plan:
        info = st.entries[hf][1]
        shape = list(info["shape"])
        if bn == "tok_embd" and shape[0] != config["vocab_size"]:
            config["vocab_size"] = shape[0]
        dt = tensor_dtype(bn, len(shape), qtype, keep_q8)
        if dt in QUANT_BYTES and shape[-1] % QK != 0:
            dt = "f16"
        n = int(np.prod(shape))
        nbytes = n // QK * QUANT_BYTES[dt] if dt in QUANT_BYTES else {"f32": 4 * n, "f16": 2 * n, "bf16": 2 * n}[dt]
        entries.append((hf, bn, dt, shape, offset, nbytes))
        offset = (offset + nbytes + ALIGN - 1) // ALIGN * ALIGN
    tok_entries = []
    for name, (dt, shape, blob) in tok_tensors.items():
        tok_entries.append((name, dt, shape, offset, len(blob), blob))
        offset = (offset + len(blob) + ALIGN - 1) // ALIGN * ALIGN

    header = {
        "format": "borg",
        "version": VERSION,
        "arch": arch,
        "source": source or os.path.basename(os.path.normpath(model_dir)),
        "quant": qtype + ("+imatrix" if importance is not None else ""),
        "config": config,
        "tokenizer": tok_meta,
        "tensors": {},
    }
    for hf, bn, dt, shape, off, nb in entries:
        header["tensors"][bn] = {"dtype": dt, "shape": shape, "offset": off, "nbytes": nb}
    for name, dt, shape, off, nb, _ in tok_entries:
        header["tensors"][name] = {"dtype": dt, "shape": shape, "offset": off, "nbytes": nb}

    hbytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    data_start = (16 + len(hbytes) + ALIGN - 1) // ALIGN * ALIGN
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<IQ", VERSION, len(hbytes)) + hbytes)
        f.write(b"\0" * (data_start - f.tell()))
        for k, (hf, bn, dt, shape, off, nb) in enumerate(entries):
            x = st.get(hf)
            imp = None
            if importance is not None and bn.startswith("blk.") and x.ndim == 2:
                layer, kind = int(bn.split(".")[1]), bn.split(".")[2]
                if kind in _SITE:
                    imp = importance[(layer, _SITE[kind])][: x.shape[1]]
            got_dt, blob = encode_tensor(x, dt, imp)
            assert got_dt == dt and len(blob) == nb, (bn, got_dt, dt, len(blob), nb)
            f.seek(data_start + off)
            f.write(blob)
            if log:
                log(f"[{k + 1}/{len(entries)}] {bn:24s} {dt:5s} {shape}")
        for name, dt, shape, off, nb, blob in tok_entries:
            f.seek(data_start + off)
            f.write(blob)
        f.truncate(data_start + offset)
    os.replace(tmp, out_path)
    return header


def read_borg(path: str) -> tuple[dict, np.memmap, int]:
    """Return (header, mmap, data_start) of a .borg file."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    if bytes(mm[:4]) != MAGIC:
        raise ValueError(f"{path} is not a .borg file")
    version, hlen = struct.unpack("<IQ", bytes(mm[4:16]))
    if version != VERSION:
        raise ValueError(f"unsupported .borg version {version}")
    header = json.loads(bytes(mm[16:16 + hlen]))
    data_start = (16 + hlen + ALIGN - 1) // ALIGN * ALIGN
    return header, mm, data_start


def load_tensor(header: dict, mm: np.memmap, data_start: int, name: str) -> np.ndarray:
    t = header["tensors"][name]
    raw = bytes(mm[data_start + t["offset"]: data_start + t["offset"] + t["nbytes"]])
    if t["dtype"] in ("u8", "u32", "i32"):
        return np.frombuffer(raw, dtype={"u8": np.uint8, "u32": np.uint32, "i32": np.int32}[t["dtype"]]).reshape(t["shape"])
    return dequantize(t["dtype"], raw, t["shape"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert a Hugging Face model directory to .borg")
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--qtype", default="q8_0", choices=["f32", "f16", "bf16", "q8_0", "q4", "q4_0", "q4_1"],
                    help="q8_0: near-lossless (default). q4: recommended 4-bit recipe = q4_1 with calibrated "
                         "rounding and attn_v/ffn_down kept at q8_0. q4_0/q4_1: plain 4-bit formats")
    ap.add_argument("--calibrate", nargs="?", const="builtin", default=None, metavar="TEXT_FILE",
                    help="4-bit only: weight rounding errors by activation statistics measured on a text "
                         "(the built-in sample when no file is given); needs the compiled engine")
    ap.add_argument("--source", default=None, help="model id recorded in the header")
    ap.add_argument("--keep-q8", default="", help="4-bit only: comma-separated tensor name suffixes "
                    "kept at q8_0, e.g. ffn_down,attn_v")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    keep = tuple(k.strip() for k in a.keep_q8.split(",") if k.strip())
    calib = a.calibrate
    if calib not in (None, "builtin"):
        with open(calib, encoding="utf-8") as f:
            calib = f.read()
    h = convert(a.model_dir, a.out, a.qtype, a.source, log=None if a.quiet else print, keep_q8=keep,
                calibration=calib)
    size = os.path.getsize(a.out)
    print(f"wrote {a.out}: {h['arch']} {h['config']['n_layers']} layers, {a.qtype}, {size / 1e6:.1f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
