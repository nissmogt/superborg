"""NumPy reference forward pass for .borg models: the test oracle for the C engine.

This file is a direct, readable transcription of the math the engine must
reproduce. It favours clarity over speed: every weight is dequantized to the
compute dtype (at load time, or at each use in lazy mode), so a quantized file is
evaluated exactly as its stored values say (no activation quantization, no fused
kernels).

One decoder layer, for a sequence of hidden states x [T, dim] at positions p:

    h  = rms_norm(x, attn_norm)
    q  = h @ Wq.T (+ bq)            -> [T, n_heads,    head_dim]
    k  = h @ Wk.T (+ bk)            -> [T, n_kv_heads, head_dim]
    v  = h @ Wv.T (+ bv)            -> [T, n_kv_heads, head_dim]
    q, k = rms_norm(q, q_norm), rms_norm(k, k_norm)      (qk_norm only, per head)
    q, k = rope(q, p), rope(k, p)                         (rotate-half / NeoX form)
    K, V = cache + k, cache + v                           (keys are cached after RoPE)
    a  = softmax(q K^T / sqrt(head_dim) + causal mask) V  (query head h uses kv head
                                                           h // (n_heads / n_kv_heads))
    x  = x + a @ Wo.T
    h  = rms_norm(x, ffn_norm)
    x  = x + (act(h @ Wgate.T) * (h @ Wup.T)) @ Wdown.T

and logits = rms_norm(x, output_norm) @ output.T, where output is tok_embd when
the embeddings are tied.

Usage:
    m = ReferenceModel("model.borg")
    logits = m.forward(tokens)                     # fresh sequence, [T, vocab_size]
    step = m.forward([next_tok], start_pos=m.n_past)  # incremental decoding
    out = m.generate(prompt_tokens, 32)            # greedy
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from .convert import dequantize, load_tensor, read_borg


# ---------------------------------------------------------------------------
# building blocks

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """x * rsqrt(mean(x^2) + eps) * w over the last axis.

    transformers upcasts to float32 for this; the reference simply computes in its
    own dtype (float32 or float64). The weight multiplies the normalized value, it
    is not folded into the scale, which matches the rounding order of HF.
    """
    var = np.mean(x * x, axis=-1, keepdims=True)
    return w * (x * (1.0 / np.sqrt(var + eps)))


def silu(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):  # exp(-x) overflows to inf for x << 0; x / inf == 0 is right
        return x / (1.0 + np.exp(-x))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation (torch gelu(approximate="tanh"), HF "gelu_pytorch_tanh")."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


ACTIVATIONS = {"silu": silu, "gelu_tanh": gelu_tanh}


def rope_inv_freq(head_dim: int, theta: float, scaling: dict | None) -> np.ndarray:
    """Inverse frequencies of the rotary embedding, float64 [head_dim // 2].

    inv_freq[i] = theta ** (-2i / head_dim), then optionally rescaled:
      linear: every frequency is divided by factor (positions are stretched).
      llama3: follows transformers' _compute_llama3_parameters: wavelengths longer
              than original_max_position_embeddings / low_freq_factor are divided
              by factor, those shorter than .../high_freq_factor are kept, and the
              band between is interpolated smoothly.
    transformers evaluates these in float32, where theta ** x is off by up to ~5e-7
    relative for theta = 1e6; the reference keeps float64 (the effect on logits is
    far below the float32 noise of the rest of the network).
    """
    inv_freq = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    if not scaling:
        return inv_freq
    kind = scaling.get("type", scaling.get("rope_type"))
    factor = float(scaling["factor"])
    if kind == "linear":
        return inv_freq / factor
    if kind == "llama3":
        low_freq_factor = float(scaling["low_freq_factor"])
        high_freq_factor = float(scaling["high_freq_factor"])
        old_context_len = float(scaling["original_max_position_embeddings"])
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: keep; wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # in between: interpolate with a smooth factor
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
        is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        return np.where(is_medium, smoothed, inv_freq_llama)
    raise ValueError(f"unsupported rope scaling {kind!r}")


def rope_cos_sin(positions: np.ndarray, inv_freq: np.ndarray, dtype) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables [T, head_dim // 2] for the given absolute positions.

    The angle position * inv_freq is formed in float64: in float32 it loses about
    position * 6e-8 radians, which starts to matter at long contexts.
    """
    ang = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate-half (NeoX) RoPE on x [T, heads, head_dim]: dimension i pairs with i + head_dim/2.

        out[i]          = x[i] * cos[i] - x[i + half] * sin[i]
        out[i + half]   = x[i + half] * cos[i] + x[i] * sin[i]
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# model

class _LazyMatrix:
    """A weight matrix kept in its file encoding and dequantized each time it is used.

    Lazy mode trades speed for memory: TinyLlama-1.1B dequantized to float32 is
    4.4 GB. It supports the two uses the forward pass makes of a matrix: W.T for
    a product and W[rows] for an embedding lookup.
    """

    def __init__(self, dtype_name: str, raw: np.ndarray, shape: list[int], out_dtype):
        self.dtype_name, self.raw, self.shape, self.out_dtype = dtype_name, raw, tuple(shape), out_dtype

    def dense(self) -> np.ndarray:
        return dequantize(self.dtype_name, self.raw, list(self.shape)).astype(self.out_dtype, copy=False)

    @property
    def T(self) -> np.ndarray:
        return self.dense().T

    def __getitem__(self, rows) -> np.ndarray:
        rows = np.asarray(rows).reshape(-1)
        picked = self.raw.reshape(self.shape[0], -1)[rows]  # whole rows of blocks
        return dequantize(self.dtype_name, picked.tobytes(), [len(rows), self.shape[1]]).astype(self.out_dtype,
                                                                                               copy=False)


class _Layer:
    __slots__ = ("attn_norm", "wq", "wk", "wv", "wo", "bq", "bk", "bv", "q_norm", "k_norm",
                 "ffn_norm", "w_gate", "w_up", "w_down")


class ReferenceModel:
    """Loads a .borg file and evaluates it with NumPy.

    dtype: np.float32 (default, the precision the engine targets) or np.float64
    (high-precision mode: weights are upcast, so memory doubles).
    lazy: keep f16/bf16/q8_0/q4_0/q4_1 matrices encoded and dequantize them at every use
    (same results, far less memory, slower). None picks lazy mode when the
    dequantized matrices would exceed LAZY_AUTO_BYTES.

    f32 tensors are read-only views of the memory-mapped file in float32 mode, so a
    float32 model costs page cache, not process memory.
    """

    LAZY_AUTO_BYTES = 3 << 30

    def __init__(self, path: str, dtype=np.float32, lazy: bool | None = None):
        self.path = path
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError("dtype must be float32 or float64")
        header, mm, data_start = read_borg(path)
        self.header = header
        self.config = c = header["config"]
        self.tokenizer_meta = header.get("tokenizer", {})

        encoded = [t for n, t in header["tensors"].items()
                   if not n.startswith("tokenizer.") and t["dtype"] not in ("f32", "u8", "u32", "i32")]
        if lazy is None:
            lazy = sum(int(np.prod(t["shape"])) for t in encoded) * self.dtype.itemsize > self.LAZY_AUTO_BYTES
        self.lazy = bool(lazy)

        def get(name: str):
            t = header["tensors"][name]
            start = data_start + t["offset"]
            if t["dtype"] == "f32" and self.dtype == np.float32:
                return np.frombuffer(mm, dtype=np.float32, count=int(np.prod(t["shape"])),
                                     offset=start).reshape(t["shape"])
            if self.lazy and len(t["shape"]) == 2 and t["dtype"] != "f32":
                return _LazyMatrix(t["dtype"], mm[start: start + t["nbytes"]], t["shape"], self.dtype)
            # load_tensor dequantizes every storage type to float32; float64 upcasts once more
            return load_tensor(header, mm, data_start, name).astype(self.dtype, copy=False)

        def opt(name: str):
            return get(name) if name in header["tensors"] else None

        self.dim = c["dim"]
        self.n_layers = c["n_layers"]
        self.n_heads = c["n_heads"]
        self.n_kv_heads = c.get("n_kv_heads", self.n_heads)
        self.head_dim = c.get("head_dim") or self.dim // self.n_heads
        self.hidden_dim = c["hidden_dim"]
        self.vocab_size = c["vocab_size"]
        self.max_seq_len = c.get("max_seq_len", 4096)
        self.eps = float(c.get("norm_eps", 1e-5))
        self.act = ACTIVATIONS[c.get("act", "silu")]
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")

        hd, nq, nkv = self.head_dim, self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        self.tok_embd = get("tok_embd")
        self.output_norm = get("output_norm")
        self.output = self.tok_embd if c.get("tie_embeddings") else get("output")
        expect = {"tok_embd": (self.vocab_size, self.dim), "output": (self.vocab_size, self.dim),
                  "output_norm": (self.dim,)}
        for name, shape in expect.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")

        self.layers: list[_Layer] = []
        for i in range(self.n_layers):
            p = f"blk.{i}."
            L = _Layer()
            L.attn_norm = get(p + "attn_norm")
            L.wq, L.wk, L.wv, L.wo = get(p + "attn_q"), get(p + "attn_k"), get(p + "attn_v"), get(p + "attn_o")
            L.bq, L.bk, L.bv = opt(p + "attn_q.bias"), opt(p + "attn_k.bias"), opt(p + "attn_v.bias")
            if c.get("attn_bias") and L.bq is None:
                raise ValueError(f"attn_bias is set but {p}attn_q.bias is missing")
            L.q_norm = L.k_norm = None
            if c.get("qk_norm"):
                L.q_norm, L.k_norm = get(p + "attn_q_norm"), get(p + "attn_k_norm")
            L.ffn_norm = get(p + "ffn_norm")
            L.w_gate, L.w_up, L.w_down = get(p + "ffn_gate"), get(p + "ffn_up"), get(p + "ffn_down")
            shapes = {"wq": (nq, self.dim), "wk": (nkv, self.dim), "wv": (nkv, self.dim), "wo": (self.dim, nq),
                      "w_gate": (self.hidden_dim, self.dim), "w_up": (self.hidden_dim, self.dim),
                      "w_down": (self.dim, self.hidden_dim)}
            for name, shape in shapes.items():
                if tuple(getattr(L, name).shape) != shape:
                    raise ValueError(f"layer {i} {name} has shape {getattr(L, name).shape}, expected {shape}")
            self.layers.append(L)
        self._mm = mm  # the views above point into it

        self.inv_freq = rope_inv_freq(hd, float(c.get("rope_theta", 10000.0)), c.get("rope_scaling"))
        self.last_hidden: np.ndarray | None = None
        self.reset()

    # ---- KV cache ----------------------------------------------------------
    def reset(self) -> None:
        """Empty the KV cache."""
        empty = np.zeros((0, self.n_kv_heads, self.head_dim), dtype=self.dtype)
        self.k_cache = [empty] * self.n_layers  # per layer: [n_past, n_kv_heads, head_dim], after RoPE
        self.v_cache = [empty] * self.n_layers
        self.tokens: list[int] = []

    @property
    def n_past(self) -> int:
        return len(self.tokens)

    def truncate(self, n_keep: int) -> None:
        """Drop cached positions >= n_keep (like borg_ctx_truncate)."""
        n_keep = max(0, min(n_keep, self.n_past))
        self.k_cache = [k[:n_keep] for k in self.k_cache]
        self.v_cache = [v[:n_keep] for v in self.v_cache]
        self.tokens = self.tokens[:n_keep]

    # ---- forward -----------------------------------------------------------
    def forward(self, tokens, start_pos: int | None = None) -> np.ndarray:
        """Logits [len(tokens), vocab_size] in the compute dtype.

        start_pos None: a fresh sequence; the cache is cleared and tokens take
        positions 0..T-1. Otherwise the cache is truncated to start_pos (which must
        not exceed n_past) and tokens are appended at positions start_pos.. .
        Either way the tokens stay in the cache for later calls.
        """
        toks = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if toks.size == 0:
            return np.zeros((0, self.vocab_size), dtype=self.dtype)
        if toks.min() < 0 or toks.max() >= self.vocab_size:
            raise ValueError("token id out of range")
        if start_pos is None:
            self.reset()
            start_pos = 0
        elif not 0 <= start_pos <= self.n_past:
            raise ValueError(f"start_pos {start_pos} is outside the cached length 0..{self.n_past}")
        else:
            self.truncate(start_pos)

        T = toks.size
        H, KV, hd = self.n_heads, self.n_kv_heads, self.head_dim
        G = H // KV  # query heads per kv head
        positions = np.arange(start_pos, start_pos + T)
        cos, sin = rope_cos_sin(positions, self.inv_freq, self.dtype)
        S = start_pos + T  # keys visible after this call
        # causal mask: query t (position start_pos + t) sees keys 0..start_pos + t
        mask = np.arange(S)[None, :] > positions[:, None]  # [T, S], True = hidden
        scale = hd ** -0.5  # same Python double as transformers, rounded once when it meets the scores

        x = self.tok_embd[toks]  # [T, dim] (fancy indexing copies)
        for li, L in enumerate(self.layers):
            # --- attention
            h = rms_norm(x, L.attn_norm, self.eps)
            q = h @ L.wq.T
            k = h @ L.wk.T
            v = h @ L.wv.T
            if L.bq is not None:
                q, k, v = q + L.bq, k + L.bk, v + L.bv
            q = q.reshape(T, H, hd)
            k = k.reshape(T, KV, hd)
            v = v.reshape(T, KV, hd)
            if L.q_norm is not None:
                q = rms_norm(q, L.q_norm, self.eps)
                k = rms_norm(k, L.k_norm, self.eps)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            K = np.concatenate([self.k_cache[li], k], axis=0)  # [S, KV, hd]
            V = np.concatenate([self.v_cache[li], v], axis=0)
            self.k_cache[li], self.v_cache[li] = K, V

            qg = q.reshape(T, KV, G, hd).transpose(1, 2, 0, 3)          # [KV, G, T, hd]
            scores = (qg @ K.transpose(1, 2, 0)[:, None]) * scale         # [KV, G, T, S]
            scores = np.where(mask, -np.inf, scores)
            probs = softmax(scores, axis=-1)
            att = probs @ V.transpose(1, 0, 2)[:, None]                   # [KV, G, T, hd]
            att = att.transpose(2, 0, 1, 3).reshape(T, H * hd)            # head h = kv * G + g
            x = x + att @ L.wo.T

            # --- feed-forward (gated)
            h = rms_norm(x, L.ffn_norm, self.eps)
            x = x + (self.act(h @ L.w_gate.T) * (h @ L.w_up.T)) @ L.w_down.T

        self.tokens.extend(int(t) for t in toks)
        h = rms_norm(x, self.output_norm, self.eps)
        self.last_hidden = h
        logits = h @ self.output.T
        assert logits.dtype == self.dtype, logits.dtype  # nothing silently promoted to float64
        return logits

    __call__ = forward

    # ---- generation --------------------------------------------------------
    def eos_ids(self) -> list[int]:
        t = self.tokenizer_meta
        ids = list(t.get("eos_ids") or [])
        if not ids and t.get("eos_id", -1) >= 0:
            ids = [t["eos_id"]]
        return ids

    def generate(self, prompt, max_new_tokens: int, eos_ids=None, ignore_eos: bool = False) -> list[int]:
        """Greedy decoding with the KV cache: prefill the prompt, then one token per step.

        Stops after an end-of-generation token (included in the output) unless
        ignore_eos. eos_ids defaults to the tokenizer's eos_ids. Ties in argmax go
        to the lowest id, as in torch.argmax.
        """
        prompt = list(prompt)
        if not prompt:
            raise ValueError("generate needs at least one prompt token")
        stop = set(self.eos_ids() if eos_ids is None else eos_ids)
        logits = self.forward(prompt)
        out: list[int] = []
        for _ in range(max_new_tokens):
            tok = int(np.argmax(logits[-1]))
            out.append(tok)
            if not ignore_eos and tok in stop:
                break
            if len(out) == max_new_tokens:
                break
            logits = self.forward([tok], start_pos=self.n_past)
        return out


# ---------------------------------------------------------------------------
# metrics shared by the tests and fixture tools

def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """KL(P || Q) per row, in nats, computed in float64 from logits."""
    lp, lq = log_softmax(p_logits), log_softmax(q_logits)
    return np.sum(np.exp(lp) * (lp - lq), axis=-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the NumPy reference model on token ids")
    ap.add_argument("model")
    ap.add_argument("tokens", help="comma-separated prompt token ids")
    ap.add_argument("-n", "--max-new", type=int, default=16)
    ap.add_argument("--f64", action="store_true", help="compute in float64")
    ap.add_argument("--ignore-eos", action="store_true")
    a = ap.parse_args(argv)
    m = ReferenceModel(a.model, np.float64 if a.f64 else np.float32)
    prompt = [int(t) for t in a.tokens.split(",") if t.strip()]
    print(",".join(str(t) for t in m.generate(prompt, a.max_new, ignore_eos=a.ignore_eos)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
