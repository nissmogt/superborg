"""Live interpretability traces for chat generation.

TraceSession generates a reply token by token and, for every token, reports what happened
inside the model while it produced that token:

  * logit lens: the final norm and unembedding applied to the residual stream after each
    layer, i.e. what the model "would say" if it stopped at that depth. The probability
    and rank of the token finally chosen, per layer, show where the answer emerges.
  * contributions: the size of each layer's attention output and MLP output, and of the
    residual stream after the layer.
  * attention: for every layer, the head-mean attention row over the context (sparse, 8-bit),
    the same row weighted by the value norms (the effective attention, which ranks the
    positions a layer really reads from above the attention sink), the positions each head
    attends to most, and per-head statistics (sink mass, normalized entropy, previous-token,
    self, lookback and induction scores).
  * residual stream: its norm per layer, the massive-activation spike (max / median |dim|)
    and, with a basis file (tools/trace_basis.py), its 3-D projection per layer.
  * MLP neurons: the most strongly activated hidden units of every layer.

Every field is documented in docs/TRACE.md; wasm/trace-session.js emits the same events for
the browser engine. All of it comes from one forward pass per token (borg_ctx_enable_trace)
plus one batched projection of the per-layer residuals through the output matrix, summarized
inside the engine (borg_ctx_lens_summary) so that the per-layer logits never leave C.
"""
from __future__ import annotations

import codecs
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

__all__ = ["TraceSession", "TraceOptions", "encode_row", "decode_row", "head_stats", "effective_rows",
           "attention_report", "resid_report", "prepare_basis", "rms_normalize"]

PREFILL_CHUNK = 128  # prompt tokens per forward pass while prefilling; should_stop is checked in between
TRACE_PROMPT_MAX = 256  # prompt positions traced one per pass with trace_prompt: the last ones before the final


@dataclass
class TraceOptions:
    lens_stride: int = 1   # logit lens on every n-th layer (the last layer is always included)
    lens_top: int = 3      # top tokens reported per lens layer
    attn_top: int = 8      # most attended positions (by effective attention) reported per layer
    head_top: int = 3      # most attended positions reported per head
    mlp_top: int = 6       # most active neurons reported per layer
    final_top: int = 5     # alternatives reported from the output distribution
    fused_lens: bool = True  # summarize the lens in C (borg_ctx_lens_summary); False: full logits + NumPy


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _r(x, d: int) -> float:
    return round(float(x), d)


def _rl(x, d: int) -> list:
    """An array rounded to d decimals as nested lists of floats, in one vectorized step (a report
    has thousands of values; np.round is the rint(x * 10^d) / 10^d of the JavaScript port)."""
    return np.round(np.asarray(x, dtype=np.float64), d).tolist()


def _top_rows(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k largest entries along the last axis: (ids, values), largest first, ties by lower id."""
    n = x.shape[-1]
    k = min(k, n)
    if k <= 0:
        shape = x.shape[:-1] + (0,)
        return np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=x.dtype)
    part = np.sort(np.argpartition(-x, k - 1, axis=-1)[..., :k], axis=-1)
    vals = np.take_along_axis(x, part, axis=-1)
    order = np.argsort(-vals, axis=-1, kind="stable")  # candidates are in id order: a tie keeps the lower id first
    return np.take_along_axis(part, order, axis=-1), np.take_along_axis(vals, order, axis=-1)


# ---- attention rows and statistics (the same arithmetic as wasm/trace-session.js) ----------------
def encode_row(row: np.ndarray) -> dict:
    """An attention row over positions 0..n-1 as {"scale", "sink", "pos", "w"}: the mass on
    position 0 (the sink) kept apart as a float, the rest quantized to 8 bits relative to the
    row's maximum excluding position 0 (scale; an entry is w * scale / 255), and sparse: entries
    below 1/(2n) and entries that would quantize to 0 are dropped. Positions are ascending."""
    n = row.shape[0]
    out = {"scale": 0.0, "sink": _r(row[0], 4), "pos": [], "w": []}
    if n < 2:
        return out
    rest = row[1:].astype(np.float64)
    m = float(rest.max())
    if m <= 0:
        return out
    q = np.floor(rest * 255.0 / m + 0.5)
    keep = (rest >= 1.0 / (2 * n)) & (q >= 1)
    out["scale"] = round(m, 6)
    out["pos"] = (np.nonzero(keep)[0] + 1).tolist()
    out["w"] = q[keep].astype(int).tolist()
    return out


def decode_row(enc: dict, n: int) -> np.ndarray:
    """The dense row an encode_row result stands for (position 0 the sink, the rest dequantized)."""
    row = np.zeros(n, dtype=np.float64)
    row[0] = enc["sink"]
    for p, w in zip(enc["pos"], enc["w"]):
        row[p] = w * enc["scale"] / 255.0
    return row


def effective_rows(attn: np.ndarray, vnorm: np.ndarray, n_kv_heads: int) -> np.ndarray:
    """The effective attention of each layer: mean over heads of alpha_hj * |v_{kv(h), j}|,
    normalized to sum 1 over positions (Kobayashi et al. 2020). attn [L, H, n], vnorm [L, KV, n]."""
    L, H, n = attn.shape
    kv = np.arange(H) // (H // n_kv_heads)
    eff = attn.astype(np.float64) * vnorm.astype(np.float64)[:, kv, :]
    row = eff.sum(axis=1) / H
    s = row.sum(axis=-1, keepdims=True)
    return np.where(s > 0, row / np.where(s > 0, s, 1.0), row)


def head_stats(attn: np.ndarray, tokens, n_prompt: int) -> dict:
    """Per-head scalars of one query, [L][H] lists (3 decimals): sink (mass on position 0),
    entropy_norm (entropy / log(n_pos)), prev (weight on position q - 1), self (weight on q),
    lookback (mass on the prompt positions / total) and induction (sum of the weights on j + 1
    over the earlier positions j whose token equals the query's token; HeadVis's definition).
    tokens: the context's ids up to the query position q = n_pos - 1."""
    L, H, n = attn.shape
    q = n - 1
    a = attn.astype(np.float64)
    tot = a.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(a * np.log(np.where(a > 0, a, 1.0))).sum(axis=-1)
    zeros = np.zeros((L, H))
    toks = np.asarray(tokens)[:n]
    match = np.nonzero(toks[:q] == toks[q])[0]
    stats = {
        "sink": a[:, :, 0],
        "entropy_norm": ent / np.log(n) if n > 1 else zeros,
        "prev": a[:, :, q - 1] if q >= 1 else zeros,
        "self": a[:, :, q],
        "lookback": a[:, :, : max(0, min(int(n_prompt), n))].sum(axis=-1) / np.where(tot > 0, tot, 1.0),
        "induction": a[:, :, match + 1].sum(axis=-1) if match.size else zeros,
    }
    return {k: _rl(v, 3) for k, v in stats.items()}


def attention_report(attn: np.ndarray, vnorm: np.ndarray, tokens, n_prompt: int, n_kv_heads: int,
                     attn_top: int = 8, head_top: int = 3) -> dict:
    """The attention fields of a token event from the traced rows: attn_rows (the head-mean row
    per layer, encode_row), attn_eff (the effective row, encode_row), attn_top (the attn_top
    largest entries of the effective row, position 0 excluded, [[pos, w]...] per layer),
    attn_heads (per layer and head the head_top largest raw weights, position 0 excluded),
    head_stats and attn_entropy ([L][H] entropies in nats, 3 decimals)."""
    L, H, n = attn.shape
    a = attn.astype(np.float64)
    head_mean = a.sum(axis=1) / H
    eff = effective_rows(attn, vnorm, n_kv_heads)
    ids, vals = _top_rows(eff[:, 1:], attn_top)
    hid, hval = _top_rows(a[:, :, 1:], head_top)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(a * np.log(np.where(a > 0, a, 1.0))).sum(axis=-1)
    top_pos, top_w = (ids + 1).tolist(), _rl(vals, 4)
    head_pos, head_w = (hid + 1).tolist(), _rl(hval, 4)
    return {
        "attn_rows": [encode_row(row) for row in head_mean],
        "attn_eff": [encode_row(row) for row in eff],
        "attn_top": [[[p, w] for p, w in zip(top_pos[l], top_w[l])] for l in range(L)],
        "attn_heads": [[[[p, w] for p, w in zip(head_pos[l][h], head_w[l][h])] for h in range(H)] for l in range(L)],
        "head_stats": head_stats(attn, tokens, n_prompt),
        "attn_entropy": _rl(ent, 3),
    }


# ---- residual stream ------------------------------------------------------------------------------
def rms_normalize(x: np.ndarray) -> np.ndarray:
    """x / sqrt(mean(x^2)) along the last axis (rows of zeros stay zero)."""
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True))
    return x / np.where(rms > 0, rms, 1.0)


def prepare_basis(basis: dict, n_layers: int, dim: int) -> dict:
    """A <model>.basis.json (tools/trace_basis.py) as arrays for resid_report: mean [L + 1, D],
    components [L + 1, 3, D], clip_to [L + 1] and clip_dims (a list of index arrays). Raises
    ValueError when the shapes do not fit the model."""
    layers = basis.get("layers")
    if not isinstance(layers, list) or len(layers) != n_layers + 1:
        raise ValueError(f"the basis has {len(layers) if isinstance(layers, list) else 'no'} layers; the model needs "
                         f"{n_layers + 1}")
    mean = np.array([l["mean"] for l in layers], dtype=np.float64)
    comp = np.array([l["components"] for l in layers], dtype=np.float64)
    if mean.shape != (n_layers + 1, dim) or comp.shape != (n_layers + 1, 3, dim):
        raise ValueError(f"the basis is for dim {mean.shape[-1]}; the model has {dim}")
    return {"mean": mean, "components": comp,
            "clip_to": np.array([float(l.get("clip_to", np.inf)) for l in layers]),
            "clip_dims": [np.asarray(l.get("clip_dims", []), dtype=np.int64) for l in layers]}


def resid_report(resid: np.ndarray, basis: dict | None = None) -> dict:
    """Per-layer diagnostics of the residual stream [L + 1, D]: norm, spike (max |dim| over the
    median |dim|: massive activations) and xyz, its projection onto the fixed basis (massive dims
    clipped, RMS-normalized, centered, 3 components; null without a basis)."""
    r = resid.astype(np.float64)
    ab = np.abs(r)
    med = np.median(ab, axis=-1)
    out = {"norm": _rl(np.sqrt((r * r).sum(axis=-1)), 3),
           "spike": _rl(ab.max(axis=-1) / np.where(med > 0, med, 1.0), 3),
           "xyz": None}
    if basis is not None:
        x = r.copy()
        for l, dims in enumerate(basis["clip_dims"]):
            if dims.size:
                x[l, dims] = np.clip(x[l, dims], -basis["clip_to"][l], basis["clip_to"][l])
        y = rms_normalize(x) - basis["mean"]
        xyz = np.einsum("lkd,ld->lk", basis["components"], y)
        out["xyz"] = _rl(xyz, 3)
    return out


class TraceSession:
    """Chat generation with per-token activation traces (one model, one context)."""

    def __init__(self, llm, options: TraceOptions | None = None, basis: dict | None = None):
        self.llm = llm
        self.opt = options or TraceOptions()
        self.cfg = llm.model.config
        self._texts: dict[int, str] = {}
        self.basis = None
        self.n_prompt = 0  # prompt positions of the current generation (head_stats' lookback)
        self.set_basis(basis)
        llm.ctx.enable_trace(True)

    # ---- helpers ---------------------------------------------------------------
    def token_text(self, tok: int) -> str:
        t = self._texts.get(tok)
        if t is None:
            t = self.llm.model.token_bytes(tok).decode("utf-8", errors="replace")
            self._texts[tok] = t
        return t

    def set_basis(self, basis: dict | None) -> None:
        """Use a <model>.basis.json (tools/trace_basis.py) for the "xyz" residual projection, or none."""
        self.basis = None if basis is None else prepare_basis(basis, self.cfg["n_layers"], self.cfg["dim"])

    def info(self) -> dict:
        m = self.llm.model
        c = self.cfg
        return {
            "model": m.name, "arch": m.header.get("arch"), "quant": m.header.get("quant"),
            "n_layers": c["n_layers"], "n_heads": c["n_heads"], "n_kv_heads": c["n_kv_heads"],
            "dim": c["dim"], "hidden_dim": c["hidden_dim"], "vocab": m.vocab_size, "n_ctx": self.llm.n_ctx,
            "chat_format": m.tokenizer_info.get("chat_format"), "threads": self.llm.pool.size,
            "lens_layers": self.lens_layers(), "basis": self.basis is not None,
        }

    def lens_layers(self) -> list[int]:
        n = self.cfg["n_layers"]
        stride = max(1, int(self.opt.lens_stride))
        layers = list(range(0, n + 1, stride))
        if layers[-1] != n:
            layers.append(n)
        return layers

    def _sample(self, logits: np.ndarray, rng: np.random.Generator, temperature: float, top_p: float,
                top_k: int) -> tuple[int, np.ndarray]:
        logits = logits[: self.llm.model.n_vocab]
        probs = _softmax(logits)
        if temperature <= 1e-4:
            return int(np.argmax(logits)), probs
        p = _softmax(logits / temperature)
        order = np.argsort(-p)
        if top_k > 0:
            order = order[:top_k]
        if top_p < 1.0:
            cum = np.cumsum(p[order])
            order = order[: int(np.searchsorted(cum, top_p) + 1)]
        q = p[order] / p[order].sum()
        return int(order[rng.choice(len(order), p=q)]), probs

    # ---- per-token payload --------------------------------------------------------
    def rival_of(self, chosen: int, probs: np.ndarray) -> int | None:
        """The most probable token of the output distribution other than `chosen` (its rival, as
        in a logit difference), or None when the vocabulary has one token."""
        n = min(len(probs), self.llm.model.n_vocab)
        if n < 2:
            return None
        ids, _ = _top_rows(probs[:n], 2)
        return int(ids[0]) if int(ids[0]) != chosen else int(ids[1])

    def lens_report(self, chosen: int, rival: int | None = None) -> list[dict]:
        """Logit lens of the traced token: per lens layer, the top tokens, the probability and
        rank of `chosen`, the row's entropy (nats) and the probability and rank of `rival`.

        Only a few entries of each row's softmax are reported, so the full distribution is never
        materialized: a log-sum-exp per row gives the normalizer, and ranks compare logits. With
        fused_lens the engine computes the top tokens, the ranks, the log-sum-exp and the entropy
        itself while it projects the residuals (borg_ctx_lens_summary), so the [n_lens, vocab]
        logits are never written out at all; otherwise they are, and NumPy summarizes them
        (float32 exponentials, float64 sums). Both give the same tokens and ranks, and
        probabilities within 1e-6."""
        o = self.opt
        layers = self.lens_layers()
        n_vocab = self.llm.model.n_vocab
        k = max(1, min(o.lens_top, n_vocab))
        track = [chosen] if rival is None else [chosen, rival]
        if o.fused_lens:
            s = self.llm.ctx.lens_summary(layers, k, track, n_vocab)
            lse = s["lse"]
            p_top = np.exp(s["top_logits"].astype(np.float64) - lse[:, None])
            p_track = np.exp(s["track_logit"].astype(np.float64) - lse[:, None])
            rank, entropy = s["track_rank"], s["entropy"]
        else:
            z = self.llm.ctx.lens(layers)[:, :n_vocab]   # [n_lens, V] float32 logits
            zt = z[:, track]
            rank = (z[:, :, None] > zt[:, None, :]).sum(axis=1)
            top, zk = _top_rows(z, k)
            zk = zk.astype(np.float64)
            m = zk[:, :1]                                                  # row maxima
            lse = m[:, 0] + np.log(np.exp(z - m.astype(np.float32)).sum(axis=1, dtype=np.float64))
            p_top = np.exp(zk - lse[:, None])
            p_track = np.exp(zt.astype(np.float64) - lse[:, None])
            p = _softmax(z)
            entropy = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)
            s = {"top_ids": top}
        out = []
        for j, li in enumerate(layers):
            e = {"layer": li,
                 "top": [[self.token_text(int(t)), _r(p, 4), int(t)] for t, p in zip(s["top_ids"][j], p_top[j])],
                 "p": _r(p_track[j, 0], 5), "rank": int(rank[j, 0]), "entropy": _r(entropy[j], 4),
                 "rival_p": None if rival is None else _r(p_track[j, 1], 5),
                 "rival_rank": None if rival is None else int(rank[j, 1])}
            out.append(e)
        return out

    def token_report(self, chosen: int, probs: np.ndarray, n_prompt: int | None = None, sampling: bool = True) -> dict:
        """Everything the trace says about the step whose output distribution is `probs` and whose
        token of interest is `chosen` (the sampled token, or the actual next token of a traced
        prompt position). n_prompt: prompt positions for head_stats' lookback (default: the
        current generation's). sampling=False leaves out the fields that describe the sampling
        distribution (prob, entropy, alternatives), as prompt_token events do."""
        o = self.opt
        c = self.cfg
        tr = self.llm.ctx.trace()
        rival = self.rival_of(chosen, probs)
        lens = self.lens_report(chosen, rival)
        norms = tr["norms"]
        if n_prompt is None:
            n_prompt = self.n_prompt or tr["pos"] + 1
        mid, mval = _top_rows(np.abs(tr["mlp"]), o.mlp_top)
        rep = {"pos": int(tr["pos"])}
        if sampling:
            fid, fval = _top_rows(probs[: self.llm.model.n_vocab], o.final_top)
            p = probs[probs > 0]
            rep["prob"] = _r(probs[chosen], 5)
            rep["entropy"] = _r(-(p * np.log(p)).sum(), 4)
            # [text, p, id]: different byte tokens can decode to the same text (U+FFFD)
            rep["alternatives"] = [[self.token_text(int(t)), _r(v, 4), int(t)] for t, v in zip(fid, fval)]
        rep["rival"] = None if rival is None else [self.token_text(rival), _r(probs[rival], 4), rival]
        rep["lens"] = lens
        rep["norms"] = {"attn": _rl(norms[:, 0], 3), "mlp": _rl(norms[:, 1], 3),
                        "resid": _rl(np.linalg.norm(tr["resid"], axis=1), 3)}
        rep["resid"] = resid_report(tr["resid"], self.basis)
        rep.update(attention_report(tr["attn"], tr["vnorm"], tr["tokens"], n_prompt, c["n_kv_heads"],
                                    o.attn_top, o.head_top))
        mpos, mval = mid.tolist(), _rl(np.take_along_axis(tr["mlp"], mid, axis=1), 3)
        rep["mlp_top"] = [[[p, v] for p, v in zip(mpos[l], mval[l])] for l in range(len(mid))]
        return rep

    # ---- generation ----------------------------------------------------------------
    def stream_chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0,
                    top_p: float = 1.0, top_k: int = 0, seed: int = 0, enable_thinking: bool = False,
                    should_stop=None, chat_template_kwargs: dict | None = None,
                    trace_prompt: bool = False) -> Iterator[dict]:
        """Yield {"type": "prefill" | "prompt_token" | "prompt" | "token" | "done", ...} events while
        generating a reply.

        The uncached part of the prompt is evaluated PREFILL_CHUNK tokens at a time; when that takes
        more than one chunk, a {"type": "prefill", "done", "total"} event follows every chunk but the
        last. With trace_prompt, the last TRACE_PROMPT_MAX uncached prompt positions before the final
        one are evaluated one per pass instead, each followed by a "prompt_token" event with the
        trace fields of a token event (no sampling fields; "id" is the token at "pos", "next" the
        token that follows it, the one the lens tracks). The final prompt position is traced by the
        first token event. Every traced prompt position costs a one-token forward pass plus the trace
        (about as much as one generated token) instead of its share of a batched chunk. Errors in
        the request itself (ValueError: empty, or too long for the context) are raised before the
        first event.

        should_stop: optional callable checked before each prompt chunk, before each traced prompt
        token and between tokens (client disconnected, stop pressed). A stop during the prompt ends
        with "prompt" then a "done" event with no tokens.

        chat_template_kwargs: further variables for the model's chat template (date_string, ...),
        as superborg/server.py passes them."""
        llm = self.llm
        ctx = llm.ctx
        text = (llm.formatter.render(messages, enable_thinking=enable_thinking, **chat_template_kwargs)
                if chat_template_kwargs else llm.render(messages, enable_thinking=enable_thinking))
        prompt = llm.tokenize(text, add_bos=False)
        if not prompt:
            raise ValueError("empty prompt")
        room = ctx.n_ctx - len(prompt) - 1
        if room <= 0:
            raise ValueError(f"the conversation ({len(prompt)} tokens) does not fit the {ctx.n_ctx}-token context")
        max_tokens = max(1, min(int(max_tokens), room))
        t0 = time.perf_counter()
        reused = 0  # the cached prefix, short of the last prompt token (its logits are needed)
        for a, b in zip(ctx.tokens, prompt[:-1]):
            if a != b:
                break
            reused += 1
        ctx.truncate(reused)
        n, total = len(prompt), len(prompt) - reused
        self.n_prompt = n
        n_vocab = llm.model.n_vocab
        # prompt positions evaluated one per pass and traced: [first_traced, n - 1)
        first_traced = max(reused, n - 1 - TRACE_PROMPT_MAX) if trace_prompt else n - 1

        def prompt_event() -> dict:
            return {"type": "prompt", "tokens": [self.token_text(t) for t in prompt], "n_prompt": n,
                    "n_reused": reused, "prompt_ms": round((time.perf_counter() - t0) * 1000, 1)}

        def cancelled() -> dict:
            return {"type": "done", "text": "", "n_gen": 0, "stop_reason": "cancelled", "seconds": 0.0,
                    "tok_per_s": 0.0}

        for start in range(reused, n, PREFILL_CHUNK):
            if start > reused:
                yield {"type": "prefill", "done": start - reused, "total": total}
            if should_stop and should_stop():
                yield prompt_event()
                yield cancelled()
                return
            end = min(start + PREFILL_CHUNK, n)
            last = end - 1 if end == n else end  # the final prompt token goes alone (its logits are needed)
            plain_end = min(last, max(first_traced, start))
            if plain_end > start:
                ctx.decode(prompt[start:plain_end], logits=None)
            for q in range(max(plain_end, start), last):  # traced prompt positions, one per pass
                if should_stop and should_stop():
                    yield prompt_event()
                    yield cancelled()
                    return
                t1 = time.perf_counter()
                logits = ctx.decode([prompt[q]], "last")
                t2 = time.perf_counter()
                report = self.token_report(prompt[q + 1], _softmax(logits[:n_vocab]), n_prompt=n, sampling=False)
                yield {"type": "prompt_token", "pos": q, "id": prompt[q], "next": prompt[q + 1],
                       "decode_ms": round((t2 - t1) * 1000, 1), "trace_ms": round((time.perf_counter() - t2) * 1000, 1),
                       **report}
            if end == n:
                t1 = time.perf_counter()  # the last token alone: its pass gives the first reply token's decode_ms
                logits = ctx.decode([prompt[-1]], "last")
                t_decode = time.perf_counter() - t1
        yield prompt_event()
        rng = np.random.default_rng(seed)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        stops = llm.formatter.stop_strings()
        out_text, n_gen, stop_reason = "", 0, "length"
        t_gen = time.perf_counter()
        for i in range(max_tokens):
            tok, probs = self._sample(logits, rng, temperature, top_p, top_k)
            t1 = time.perf_counter()
            report = self.token_report(tok, probs, n_prompt=n)
            t_trace = time.perf_counter() - t1
            # decode_ms: the forward pass that produced this token's logits
            times = {"decode_ms": round(t_decode * 1000, 1), "trace_ms": round(t_trace * 1000, 1)}
            if llm.model.is_eog(tok):
                stop_reason = "eos"
                yield {"type": "token", "i": i, "id": tok, "text": "", "raw": self.token_text(tok), "eos": True,
                       **times, **report}
                break
            piece = decoder.decode(llm.model.token_bytes(tok))
            out_text += piece
            n_gen += 1
            yield {"type": "token", "i": i, "id": tok, "text": piece, "raw": self.token_text(tok), "eos": False,
                   **times, **report}
            if any(s in out_text for s in stops):
                stop_reason = "stop"
                break
            if should_stop and should_stop():
                stop_reason = "cancelled"
                break
            t2 = time.perf_counter()
            logits = ctx.decode([tok], "last")
            t_decode = time.perf_counter() - t2
        else:
            stop_reason = "length"
        elapsed = time.perf_counter() - t_gen
        for s in stops:
            out_text = out_text.split(s)[0]
        yield {"type": "done", "text": out_text, "n_gen": n_gen, "stop_reason": stop_reason,
               "seconds": round(elapsed, 2), "tok_per_s": round(n_gen / elapsed, 2) if elapsed > 0 else 0.0}
