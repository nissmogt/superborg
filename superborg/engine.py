"""ctypes binding to the borg C engine (engine/borg.h)."""
from __future__ import annotations

import codecs
import ctypes
import json
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _lib_names() -> list[str]:
    if sys.platform == "darwin":
        return ["libborg.dylib"]
    if sys.platform == "win32":
        return ["borg.dll", "libborg.dll"]
    return ["libborg.so"]


def find_library() -> str:
    env = os.environ.get("BORG_LIB")
    if env:
        return env
    for d in (os.path.join(_HERE, "_lib"), os.path.join(_REPO, "build")):
        for n in _lib_names():
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        "libborg not found. Build it with `make` in the repository root, or set BORG_LIB to its path.")


class _CtxParams(ctypes.Structure):
    _fields_ = [("n_ctx", ctypes.c_int), ("n_batch", ctypes.c_int), ("kv_f16", ctypes.c_int)]


class _Sampling(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_float), ("top_k", ctypes.c_int), ("top_p", ctypes.c_float),
                ("min_p", ctypes.c_float), ("repeat_penalty", ctypes.c_float), ("repeat_last_n", ctypes.c_int),
                ("seed", ctypes.c_uint64)]


class _GenParams(ctypes.Structure):
    _fields_ = [("max_tokens", ctypes.c_int), ("sampling", _Sampling), ("draft", ctypes.c_void_p),
                ("n_draft", ctypes.c_int), ("lookup_ngram", ctypes.c_int), ("ignore_eos", ctypes.c_int),
                ("stop_tokens", ctypes.POINTER(ctypes.c_int32)), ("n_stop_tokens", ctypes.c_int),
                ("json_mode", ctypes.c_int)]


class _GenStats(ctypes.Structure):
    _fields_ = [("n_prompt", ctypes.c_int), ("n_reused", ctypes.c_int), ("n_gen", ctypes.c_int),
                ("n_drafted", ctypes.c_int), ("n_accepted", ctypes.c_int), ("n_target_evals", ctypes.c_int),
                ("t_prompt_ms", ctypes.c_double), ("t_gen_ms", ctypes.c_double), ("stop_reason", ctypes.c_int)]


class _Config(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ("dim", "n_layers", "n_heads", "n_kv_heads", "head_dim", "hidden_dim",
                                            "vocab_size", "max_seq_len")] + \
               [("norm_eps", ctypes.c_float), ("rope_theta", ctypes.c_float)] + \
               [(n, ctypes.c_int) for n in ("attn_bias", "qk_norm", "tie_embeddings", "n_vocab", "bos_id", "eos_id",
                                            "add_bos")] + [("weight_bytes", ctypes.c_size_t)]


_TOKEN_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_void_p)
_MULTI_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int,
                             ctypes.c_void_p)

_lib = None


def lib():
    global _lib
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(find_library())
    vp, i32p, fp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
    sigs = {
        "borg_version": (ctypes.c_char_p, []),
        "borg_simd": (ctypes.c_char_p, []),
        "borg_pool_new": (vp, [ctypes.c_int]),
        "borg_pool_free": (None, [vp]),
        "borg_pool_size": (ctypes.c_int, [vp]),
        "borg_model_load": (vp, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
        "borg_model_free": (None, [vp]),
        "borg_model_config": (None, [vp, ctypes.POINTER(_Config)]),
        "borg_model_header": (ctypes.c_char_p, [vp]),
        "borg_tokenize": (ctypes.c_int, [vp, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, i32p,
                                         ctypes.c_int]),
        "borg_token_bytes": (ctypes.c_int, [vp, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int]),
        "borg_detokenize": (ctypes.c_int, [vp, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]),
        "borg_token_type": (ctypes.c_int, [vp, ctypes.c_int32]),
        "borg_is_eog": (ctypes.c_int, [vp, ctypes.c_int32]),
        "borg_ctx_new": (vp, [vp, vp, ctypes.POINTER(_CtxParams)]),
        "borg_ctx_free": (None, [vp]),
        "borg_ctx_n_ctx": (ctypes.c_int, [vp]),
        "borg_ctx_n_past": (ctypes.c_int, [vp]),
        "borg_ctx_tokens": (i32p, [vp]),
        "borg_ctx_truncate": (None, [vp, ctypes.c_int]),
        "borg_decode": (ctypes.c_int, [vp, i32p, ctypes.c_int, fp, ctypes.c_int]),
        "borg_ctx_embedding": (fp, [vp]),
        "borg_decode_multi": (ctypes.c_int, [ctypes.POINTER(vp), ctypes.c_int, i32p, fp]),
        "borg_ctx_copy": (ctypes.c_int, [vp, vp]),
        "borg_ctx_enable_trace": (ctypes.c_int, [vp, ctypes.c_int]),
        "borg_ctx_trace_pos": (ctypes.c_int, [vp]),
        "borg_ctx_trace_resid": (fp, [vp, ctypes.c_int]),
        "borg_ctx_trace_attn": (fp, [vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]),
        "borg_ctx_trace_mlp": (fp, [vp, ctypes.c_int]),
        "borg_ctx_trace_norms": (fp, [vp]),
        "borg_ctx_lens": (ctypes.c_int, [vp, ctypes.POINTER(ctypes.c_int), ctypes.c_int, fp]),
        "borg_ctx_save": (ctypes.c_int, [vp, ctypes.c_char_p]),
        "borg_ctx_load": (ctypes.c_int, [vp, ctypes.c_char_p]),
        "borg_generate_multi": (ctypes.c_int, [ctypes.POINTER(vp), ctypes.c_int, i32p, ctypes.c_int,
                                               ctypes.POINTER(_GenParams), _MULTI_CB, vp,
                                               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                               ctypes.POINTER(_GenStats)]),
        "borg_ctx_enable_stats": (ctypes.c_int, [vp, ctypes.c_int]),
        "borg_ctx_stats": (ctypes.POINTER(ctypes.c_double), [vp, ctypes.c_int, ctypes.c_int,
                                                              ctypes.POINTER(ctypes.c_int)]),
        "borg_ctx_stats_tokens": (ctypes.c_int64, [vp]),
        "borg_sampling_default": (_Sampling, []),
        "borg_gen_default_params": (_GenParams, []),
        "borg_generate": (ctypes.c_int, [vp, i32p, ctypes.c_int, ctypes.POINTER(_GenParams), _TOKEN_CB, vp, i32p,
                                         ctypes.c_int, ctypes.POINTER(_GenStats)]),
    }
    for name, (res, args) in sigs.items():
        f = getattr(L, name)
        f.restype = res
        f.argtypes = args
    _lib = L
    return L


def simd() -> str:
    return lib().borg_simd().decode()


class Pool:
    """Worker threads shared by contexts that run one after another."""

    def __init__(self, n_threads: int = 0):
        self._p = lib().borg_pool_new(int(n_threads))
        if not self._p:
            raise RuntimeError("cannot create thread pool")

    @property
    def size(self) -> int:
        return lib().borg_pool_size(self._p)

    def close(self):
        if self._p:
            lib().borg_pool_free(self._p)
            self._p = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


@dataclass
class Sampling:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    seed: int = 0x5EED

    def _c(self) -> _Sampling:
        return _Sampling(self.temperature, self.top_k, self.top_p, self.min_p, self.repeat_penalty,
                         self.repeat_last_n, self.seed & 0xFFFFFFFFFFFFFFFF)


@dataclass
class Generation:
    text: str
    tokens: list[int]
    stop_reason: str  # "length", "eos", "stop", "context"
    n_prompt: int = 0
    n_reused: int = 0
    n_drafted: int = 0
    n_accepted: int = 0
    n_target_evals: int = 0
    prompt_ms: float = 0.0
    gen_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) * 1000.0 / self.gen_ms if self.gen_ms > 0 else 0.0


_STOP_REASONS = {0: "length", 1: "eos", 2: "stop", 3: "context"}


class Model:
    """A loaded .borg model (read-only, memory-mapped)."""

    def __init__(self, path: str, threads: int = 0, pool: Pool | None = None):
        err = ctypes.create_string_buffer(1024)
        self._m = lib().borg_model_load(os.fsencode(path), err, len(err))
        if not self._m:
            raise RuntimeError(f"cannot load {path}: {err.value.decode(errors='replace')}")
        self.path = path
        self.header = json.loads(lib().borg_model_header(self._m).decode("utf-8"))
        cfg = _Config()
        lib().borg_model_config(self._m, ctypes.byref(cfg))
        self._cfg = cfg
        self.config = self.header["config"]
        self.tokenizer_info = self.header["tokenizer"]
        self.n_vocab = cfg.n_vocab
        self.vocab_size = cfg.vocab_size
        self.max_seq_len = cfg.max_seq_len
        self.weight_bytes = cfg.weight_bytes
        self.pool = pool or Pool(threads)
        self._nfc = bool(self.tokenizer_info.get("nfc"))
        self._piece_cache: dict[int, bytes] = {}
        self._added_re = None  # splits text on added tokens (built lazily, for NFC models)

    def close(self):
        if getattr(self, "_m", None):
            lib().borg_model_free(self._m)
            self._m = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def name(self) -> str:
        return self.header.get("source") or os.path.basename(self.path)

    # ---- tokenizer -------------------------------------------------------
    def _normalize(self, text: str, special: bool) -> str:
        """NFC-normalize like Hugging Face: only the text between added tokens, so a combining
        mark after "<|im_end|>" cannot merge into the token's last character."""
        if not special:
            return unicodedata.normalize("NFC", text)
        if self._added_re is None:
            import re
            pieces = sorted({self.token_bytes(i).decode("utf-8", "replace")
                             for i in self.tokenizer_info.get("added_tokens", [])} - {""}, key=len, reverse=True)
            self._added_re = re.compile("(" + "|".join(map(re.escape, pieces)) + ")") if pieces else False
        if not self._added_re:
            return unicodedata.normalize("NFC", text)
        parts = self._added_re.split(text)
        return "".join(p if i % 2 else unicodedata.normalize("NFC", p) for i, p in enumerate(parts))

    def tokenize(self, text: str, add_bos: bool | None = None, special: bool = True) -> list[int]:
        if self._nfc:
            text = self._normalize(text, special)
        data = text.encode("utf-8")
        cap = len(data) + 16
        bos = -1 if add_bos is None else int(bool(add_bos))
        while True:
            buf = (ctypes.c_int32 * cap)()
            n = lib().borg_tokenize(self._m, data, len(data), bos, int(special), buf, cap)
            if n >= 0:
                return list(buf[:n])
            cap = -n

    def token_bytes(self, tok: int) -> bytes:
        b = self._piece_cache.get(tok)
        if b is None:
            buf = ctypes.create_string_buffer(256)
            n = lib().borg_token_bytes(self._m, tok, buf, len(buf))
            if n > len(buf):
                buf = ctypes.create_string_buffer(n)
                n = lib().borg_token_bytes(self._m, tok, buf, len(buf))
            b = buf.raw[:n]
            self._piece_cache[tok] = b
        return b

    def detokenize(self, tokens: Sequence[int], skip_special: bool = False) -> str:
        if not tokens:
            return ""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        cap = 64 + 16 * len(tokens)
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = lib().borg_detokenize(self._m, arr, len(tokens), int(skip_special), buf, cap)
            if n <= cap:
                return buf.raw[:n].decode("utf-8", errors="replace")
            cap = n

    def token_type(self, tok: int) -> int:
        return lib().borg_token_type(self._m, tok)

    def is_eog(self, tok: int) -> bool:
        return bool(lib().borg_is_eog(self._m, tok))

    def token_id(self, text: str) -> int | None:
        """Id of a single token whose text is exactly `text` (e.g. a special token), else None."""
        ids = self.tokenize(text, add_bos=False, special=True)
        return ids[0] if len(ids) == 1 else None

    # ---- contexts --------------------------------------------------------
    def context(self, n_ctx: int = 0, n_batch: int = 0, kv_f16: bool = False) -> "Context":
        return Context(self, n_ctx=n_ctx, n_batch=n_batch, kv_f16=kv_f16)


class Context:
    """KV cache and scratch for one sequence of a model."""

    def __init__(self, model: Model, n_ctx: int = 0, n_batch: int = 0, kv_f16: bool = False):
        self.model = model
        params = _CtxParams(int(n_ctx), int(n_batch), int(kv_f16))
        self._c = lib().borg_ctx_new(model._m, model.pool._p, ctypes.byref(params))
        if not self._c:
            raise MemoryError("cannot allocate context (reduce n_ctx)")
        self.n_ctx = lib().borg_ctx_n_ctx(self._c)

    def close(self):
        if getattr(self, "_c", None):
            lib().borg_ctx_free(self._c)
            self._c = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def n_past(self) -> int:
        return lib().borg_ctx_n_past(self._c)

    @property
    def tokens(self) -> list[int]:
        n = self.n_past
        p = lib().borg_ctx_tokens(self._c)
        return [p[i] for i in range(n)]

    def truncate(self, n_keep: int) -> None:
        lib().borg_ctx_truncate(self._c, int(n_keep))

    def reset(self) -> None:
        self.truncate(0)

    def decode(self, tokens: Sequence[int], logits: str | None = "last") -> np.ndarray | None:
        """Append tokens to the cache. logits: None, "last" -> [vocab], "all" -> [n, vocab]."""
        n = len(tokens)
        if n == 0:
            return None
        arr = (ctypes.c_int32 * n)(*tokens)
        V = self.model.vocab_size
        mode = {None: 0, "last": 1, "all": 2}[logits]
        out = None
        ptr = None
        if mode == 1:
            out = np.empty(V, dtype=np.float32)
        elif mode == 2:
            out = np.empty((n, V), dtype=np.float32)
        if out is not None:
            ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = lib().borg_decode(self._c, arr, n, ptr, mode)
        if rc != 0:
            raise RuntimeError({-1: "context is full", -3: "token id out of range"}.get(rc, f"decode failed ({rc})"))
        return out

    def sync(self, tokens: Sequence[int]) -> int:
        """Make the cache hold exactly `tokens` (reusing the shared prefix). Returns tokens reused."""
        cached = self.tokens
        common = 0
        for a, b in zip(cached, tokens):
            if a != b:
                break
            common += 1
        self.truncate(common)
        if common < len(tokens):
            self.decode(tokens[common:], logits=None)
        return common

    def copy_from(self, src: "Context") -> None:
        """Make this context hold the same cached tokens as src (same model)."""
        if lib().borg_ctx_copy(self._c, src._c) != 0:
            raise ValueError("contexts are incompatible (model, kv type or size)")

    def save(self, path: str) -> None:
        """Write the cached tokens and keys/values to `path`."""
        if lib().borg_ctx_save(self._c, os.fsencode(path)) != 0:
            raise OSError(f"cannot write {path}")

    def load(self, path: str) -> None:
        """Restore a cache written by save() for a model of the same shape."""
        rc = lib().borg_ctx_load(self._c, os.fsencode(path))
        if rc != 0:
            raise ValueError(f"cannot load KV cache {path}: " + {-1: "cannot open", -2: "incompatible model or "
                             "context too small", -3: "truncated or corrupt file"}.get(rc, str(rc)))

    # ---- interpretability ----------------------------------------------------
    def enable_trace(self, enable: bool = True) -> None:
        """Record the internals of the last token of every forward pass (see trace())."""
        if lib().borg_ctx_enable_trace(self._c, int(enable)) != 0:
            raise MemoryError("cannot allocate trace buffers")

    def trace(self, attention: bool = True, mlp: bool = True) -> dict | None:
        """Internals of the last traced token as NumPy arrays (copies), or None.

        pos: its position; resid [n_layers + 1, dim]: embeddings then each layer's output;
        norms [n_layers, 3]: |attention output|, |MLP output|, |residual after|;
        attn [n_layers, n_heads, pos + 1]: attention weights; mlp [n_layers, hidden_dim]."""
        L = lib()
        pos = L.borg_ctx_trace_pos(self._c)
        if pos < 0:
            return None
        cfg = self.model.config
        nl, d, nh, f = cfg["n_layers"], cfg["dim"], cfg["n_heads"], cfg["hidden_dim"]
        resid = np.ctypeslib.as_array(L.borg_ctx_trace_resid(self._c, 0), shape=((nl + 1) * d,))
        out = {"pos": pos, "resid": resid.reshape(nl + 1, d).copy(),
               "norms": np.ctypeslib.as_array(L.borg_ctx_trace_norms(self._c), shape=(nl * 3,)).reshape(nl, 3).copy()}
        if attention:
            n = ctypes.c_int(0)
            base = L.borg_ctx_trace_attn(self._c, 0, 0, ctypes.byref(n))
            full = np.ctypeslib.as_array(base, shape=(nl * nh * self.n_ctx,)).reshape(nl, nh, self.n_ctx)
            out["attn"] = full[:, :, : n.value].copy()
        if mlp:
            out["mlp"] = np.ctypeslib.as_array(L.borg_ctx_trace_mlp(self._c, 0), shape=(nl * f,)).reshape(nl, f).copy()
        return out

    def lens(self, layers: Sequence[int]) -> np.ndarray:
        """Logit lens of the traced token: final norm + output projection applied to the
        residual after each listed layer (0 = embeddings, n_layers = the model's own output)."""
        layers = list(layers)
        if not layers:
            return np.zeros((0, self.model.vocab_size), dtype=np.float32)
        arr = (ctypes.c_int * len(layers))(*layers)
        out = np.empty((len(layers), self.model.vocab_size), dtype=np.float32)
        rc = lib().borg_ctx_lens(self._c, arr, len(layers), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise ValueError({-1: "nothing traced yet (enable_trace, then decode)",
                              -3: "layer out of range"}.get(rc, f"lens failed ({rc})"))
        return out

    def embedding(self) -> np.ndarray:
        p = lib().borg_ctx_embedding(self._c)
        return np.ctypeslib.as_array(p, shape=(self.model.config["dim"],)).copy()

    def enable_stats(self, enable: bool = True) -> None:
        """Accumulate per-column x^2 of every matrix input (for calibrated quantization)."""
        if lib().borg_ctx_enable_stats(self._c, int(enable)) != 0:
            raise MemoryError("cannot allocate statistics buffers")

    def stats(self, layer: int, site: int) -> np.ndarray:
        """Sums of x^2 per input column at a site (0 attn in, 1 attn out, 2 ffn in, 3 ffn hidden)."""
        n = ctypes.c_int(0)
        p = lib().borg_ctx_stats(self._c, layer, site, ctypes.byref(n))
        if not p:
            raise ValueError("statistics are not enabled or the layer/site is out of range")
        return np.ctypeslib.as_array(p, shape=(n.value,)).copy()

    def stats_tokens(self) -> int:
        return int(lib().borg_ctx_stats_tokens(self._c))

    def score(self, prompt: Sequence[int], continuation: Sequence[int]) -> np.ndarray:
        """Log-probabilities of each continuation token given the prompt (and earlier continuation)."""
        if not continuation:
            return np.zeros(0, dtype=np.float32)
        if not prompt:
            raise ValueError("score() needs at least one prompt token")
        prompt = list(prompt)
        self.sync(prompt[:-1])
        logits = self.decode([prompt[-1]] + list(continuation[:-1]), logits="all")
        logits = logits[:, : self.model.n_vocab].astype(np.float64)
        mx = logits.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
        idx = np.asarray(continuation, dtype=np.int64)
        return (logits[np.arange(len(idx)), idx] - lse).astype(np.float32)

    def generate(self, prompt: Sequence[int], max_tokens: int = 256, sampling: Sampling | None = None,
                 stop: Iterable[str] = (), stream: Callable[[str], None] | None = None,
                 draft: "Context | None" = None, n_draft: int = 6, lookup_ngram: int = 0,
                 ignore_eos: bool = False, stop_tokens: Sequence[int] = (), json_mode: bool = False) -> Generation:
        """Generate a continuation of `prompt` (token ids).

        stop: strings that end generation (the stop string is not included in the text).
        stream: called with each new piece of text as it is produced.
        draft / lookup_ngram: speculative decoding with a draft model context or prompt lookup.
        """
        sampling = sampling or Sampling()
        stops = [s for s in stop if s]
        gp = lib().borg_gen_default_params()
        gp.max_tokens = int(max_tokens)
        gp.sampling = sampling._c()
        gp.draft = draft._c if draft is not None else None
        gp.n_draft = int(n_draft)
        gp.lookup_ngram = int(lookup_ngram)
        gp.ignore_eos = int(ignore_eos)
        gp.json_mode = int(json_mode)
        stop_arr = None
        if stop_tokens:
            stop_arr = (ctypes.c_int32 * len(stop_tokens))(*stop_tokens)
            gp.stop_tokens = stop_arr
            gp.n_stop_tokens = len(stop_tokens)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        state = {"text": "", "emitted": 0, "stopped": False}
        hold = max((len(s) for s in stops), default=0)

        def on_token(tok, piece, n, user):
            try:
                state["text"] += decoder.decode(ctypes.string_at(piece, n))
                text = state["text"]
                if stops:
                    # only search the region that could contain a new match
                    lo = max(0, state["emitted"] - hold)
                    cut = min((i for i in (text.find(s, lo) for s in stops) if i >= 0), default=-1)
                    if cut >= 0:
                        state["text"] = text[:cut]
                        state["stopped"] = True
                        if stream and cut > state["emitted"]:
                            stream(text[state["emitted"]:cut])
                        state["emitted"] = cut
                        return 1
                if stream:
                    safe = len(text) - (hold - 1 if hold > 1 else 0)
                    if safe > state["emitted"]:
                        stream(text[state["emitted"]:safe])
                        state["emitted"] = safe
                return 0
            except Exception:  # noqa: BLE001 - never let exceptions cross the C boundary
                state["stopped"] = True
                return 1

        cb = _TOKEN_CB(on_token)
        prompt = list(prompt)
        if not prompt:
            raise ValueError("empty prompt")
        if len(prompt) >= self.n_ctx:
            raise ValueError(f"prompt has {len(prompt)} tokens; context holds {self.n_ctx}")
        parr = (ctypes.c_int32 * len(prompt))(*prompt)
        cap = max(1, int(max_tokens))
        out = (ctypes.c_int32 * cap)()
        st = _GenStats()
        n = lib().borg_generate(self._c, parr, len(prompt), ctypes.byref(gp), cb, None, out, cap, ctypes.byref(st))
        if n < 0:
            raise RuntimeError(f"generation failed ({n})")
        state["text"] += decoder.decode(b"", final=True)
        if stream and not state["stopped"] and len(state["text"]) > state["emitted"]:
            stream(state["text"][state["emitted"]:])
        reason = "stop" if state["stopped"] else _STOP_REASONS.get(st.stop_reason, "length")
        return Generation(text=state["text"], tokens=list(out[:min(n, cap)]), stop_reason=reason,
                          n_prompt=st.n_prompt, n_reused=st.n_reused, n_drafted=st.n_drafted,
                          n_accepted=st.n_accepted, n_target_evals=st.n_target_evals,
                          prompt_ms=st.t_prompt_ms, gen_ms=st.t_gen_ms)


class _StopTracker:
    """Incremental UTF-8 decoding and stop-string detection for one sequence."""

    def __init__(self, stops: Sequence[str], stream: Callable[[str], None] | None):
        self.stops = [x for x in stops if x]
        self.hold = max((len(x) for x in self.stops), default=0)
        self.stream = stream
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.text = ""
        self.emitted = 0
        self.stopped = False

    def feed(self, data: bytes) -> bool:
        """Add token bytes; returns True when a stop string was found."""
        self.text += self.decoder.decode(data)
        if self.stops:
            lo = max(0, self.emitted - self.hold)
            cut = min((i for i in (self.text.find(x, lo) for x in self.stops) if i >= 0), default=-1)
            if cut >= 0:
                self.text = self.text[:cut]
                self.stopped = True
                self._flush(cut)
                return True
        self._flush(len(self.text) - (self.hold - 1 if self.hold > 1 else 0))
        return False

    def _flush(self, upto: int) -> None:
        if self.stream and upto > self.emitted:
            self.stream(self.text[self.emitted:upto])
        self.emitted = max(self.emitted, upto)

    def finish(self) -> None:
        self.text += self.decoder.decode(b"", final=True)
        if not self.stopped:
            self._flush(len(self.text))


def generate_multi(contexts: Sequence[Context], prompt: Sequence[int], max_tokens: int = 256,
                   sampling: Sampling | None = None, stop: Iterable[str] = (),
                   stream: Callable[[int, str], None] | None = None, ignore_eos: bool = False,
                   stop_tokens: Sequence[int] = (), json_mode: bool = False) -> list[Generation]:
    """Sample len(contexts) continuations of `prompt` in lockstep with batched decoding.

    Sequence i uses seed sampling.seed + i. The prompt is evaluated once (in contexts[0],
    reusing its cached prefix) and copied to the other contexts. stream(i, text) receives
    the text pieces of sequence i.
    """
    n = len(contexts)
    if n == 0:
        return []
    sampling = sampling or Sampling(temperature=0.7)
    stops = list(stop)
    trackers = [_StopTracker(stops, (lambda t, i=i: stream(i, t)) if stream else None) for i in range(n)]
    toks: list[list[int]] = [[] for _ in range(n)]
    gp = lib().borg_gen_default_params()
    gp.max_tokens = int(max_tokens)
    gp.sampling = sampling._c()
    gp.ignore_eos = int(ignore_eos)
    gp.json_mode = int(json_mode)
    stop_arr = None
    if stop_tokens:
        stop_arr = (ctypes.c_int32 * len(stop_tokens))(*stop_tokens)
        gp.stop_tokens = stop_arr
        gp.n_stop_tokens = len(stop_tokens)

    def on_token(seq, tok, piece, nbytes, user):
        try:
            toks[seq].append(tok)
            return 1 if trackers[seq].feed(ctypes.string_at(piece, nbytes)) else 0
        except Exception:  # noqa: BLE001 - never let exceptions cross the C boundary
            trackers[seq].stopped = True
            return 1

    cb = _MULTI_CB(on_token)
    prompt = list(prompt)
    if not prompt:
        raise ValueError("empty prompt")
    if len(prompt) >= contexts[0].n_ctx:
        raise ValueError(f"prompt has {len(prompt)} tokens; context holds {contexts[0].n_ctx}")
    carr = (ctypes.c_void_p * n)(*[c._c for c in contexts])
    parr = (ctypes.c_int32 * len(prompt))(*prompt)
    ngen = (ctypes.c_int * n)()
    reasons = (ctypes.c_int * n)()
    st = _GenStats()
    rc = lib().borg_generate_multi(carr, n, parr, len(prompt), ctypes.byref(gp), cb, None, ngen, reasons,
                                   ctypes.byref(st))
    if rc < 0:
        raise RuntimeError(f"generation failed ({rc})")
    out = []
    for i in range(n):
        trackers[i].finish()
        reason = "stop" if trackers[i].stopped else _STOP_REASONS.get(reasons[i], "length")
        out.append(Generation(text=trackers[i].text, tokens=toks[i], stop_reason=reason,
                              n_prompt=st.n_prompt, n_reused=st.n_reused, prompt_ms=st.t_prompt_ms,
                              gen_ms=st.t_gen_ms, n_target_evals=st.n_target_evals))
    return out
