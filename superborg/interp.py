"""Live interpretability traces for chat generation.

TraceSession generates a reply token by token and, for every token, reports what happened
inside the model while it produced that token:

  * logit lens: the final norm and unembedding applied to the residual stream after each
    layer, i.e. what the model "would say" if it stopped at that depth. The probability
    and rank of the token finally chosen, per layer, show where the answer emerges.
  * contributions: the size of each layer's attention output and MLP output, and of the
    residual stream after the layer.
  * attention: for every layer, the context tokens the current query attends to most
    (averaged over heads), and the entropy of every head.
  * MLP neurons: the most strongly activated hidden units of every layer.

All of it comes from one forward pass per token (borg_ctx_enable_trace) plus one batched
projection of the per-layer residuals through the output matrix, summarized inside the engine
(borg_ctx_lens_summary) so that the per-layer logits never leave C.
"""
from __future__ import annotations

import codecs
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

__all__ = ["TraceSession", "TraceOptions"]

PREFILL_CHUNK = 128  # prompt tokens per forward pass while prefilling; should_stop is checked in between


@dataclass
class TraceOptions:
    lens_stride: int = 1   # logit lens on every n-th layer (the last layer is always included)
    lens_top: int = 3      # top tokens reported per lens layer
    attn_top: int = 8      # most attended positions reported per layer
    mlp_top: int = 6       # most active neurons reported per layer
    final_top: int = 5     # alternatives reported from the output distribution
    fused_lens: bool = True  # summarize the lens in C (borg_ctx_lens_summary); False: full logits + NumPy


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class TraceSession:
    """Chat generation with per-token activation traces (one model, one context)."""

    def __init__(self, llm, options: TraceOptions | None = None):
        self.llm = llm
        self.opt = options or TraceOptions()
        self.cfg = llm.model.config
        self._texts: dict[int, str] = {}
        llm.ctx.enable_trace(True)

    # ---- helpers ---------------------------------------------------------------
    def token_text(self, tok: int) -> str:
        t = self._texts.get(tok)
        if t is None:
            t = self.llm.model.token_bytes(tok).decode("utf-8", errors="replace")
            self._texts[tok] = t
        return t

    def info(self) -> dict:
        m = self.llm.model
        c = self.cfg
        return {
            "model": m.name, "arch": m.header.get("arch"), "quant": m.header.get("quant"),
            "n_layers": c["n_layers"], "n_heads": c["n_heads"], "n_kv_heads": c["n_kv_heads"],
            "dim": c["dim"], "hidden_dim": c["hidden_dim"], "vocab": m.vocab_size, "n_ctx": self.llm.n_ctx,
            "chat_format": m.tokenizer_info.get("chat_format"), "threads": self.llm.pool.size,
            "lens_layers": self.lens_layers(),
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
    def lens_report(self, chosen: int) -> list[dict]:
        """Logit lens of the traced token: per lens layer, the top tokens, and the probability
        and rank of `chosen`.

        Only a few entries of each row's softmax are reported, so the full distribution is never
        materialized: a log-sum-exp per row gives the normalizer, and ranks compare logits. With
        fused_lens the engine computes the top tokens, the rank and the log-sum-exp itself while
        it projects the residuals (borg_ctx_lens_summary), so the [n_lens, vocab] logits are never
        written out at all; otherwise they are, and NumPy summarizes them (float32 exponentials,
        float64 sums). Both give the same tokens and ranks, and probabilities within 1e-6."""
        o = self.opt
        layers = self.lens_layers()
        n_vocab = self.llm.model.n_vocab
        k = max(1, min(o.lens_top, n_vocab))
        if o.fused_lens:
            s = self.llm.ctx.lens_summary(layers, k, chosen, n_vocab)
            lse = s["lse"]
            p_top = np.exp(s["top_logits"].astype(np.float64) - lse[:, None])
            p_chosen = np.exp(s["chosen_logit"].astype(np.float64) - lse)
            return [{"layer": li,
                     "top": [[self.token_text(int(t)), round(float(p), 4), int(t)]
                             for t, p in zip(s["top_ids"][j], p_top[j])],
                     "p": round(float(p_chosen[j]), 5),
                     "rank": int(s["chosen_rank"][j])}
                    for j, li in enumerate(layers)]
        z = self.llm.ctx.lens(layers)[:, :n_vocab]   # [n_lens, V] float32 logits
        zc = z[:, chosen].copy()
        rank = (z > zc[:, None]).sum(axis=1)
        top = np.argpartition(z, n_vocab - k, axis=1)[:, n_vocab - k:]
        zt = np.take_along_axis(z, top, axis=1)
        order = np.argsort(-zt, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        zt = np.take_along_axis(zt, order, axis=1).astype(np.float64)
        m = zt[:, :1]                                                  # row maxima
        lse = m[:, 0] + np.log(np.exp(z - m.astype(np.float32)).sum(axis=1, dtype=np.float64))
        p_top = np.exp(zt - lse[:, None])
        p_chosen = np.exp(zc.astype(np.float64) - lse)
        return [{"layer": li,
                 "top": [[self.token_text(int(t)), round(float(p), 4), int(t)] for t, p in zip(top[j], p_top[j])],
                 "p": round(float(p_chosen[j]), 5),
                 "rank": int(rank[j])}
                for j, li in enumerate(layers)]

    def token_report(self, chosen: int, probs: np.ndarray) -> dict:
        """Everything the trace says about the step that produced `chosen`."""
        o = self.opt
        tr = self.llm.ctx.trace()
        lens = self.lens_report(chosen)
        norms = tr["norms"]
        resid_norm = np.linalg.norm(tr["resid"], axis=1)
        attn = tr["attn"]                      # [L, H, n_pos]
        head_mean = attn.mean(axis=1)          # [L, n_pos]
        k = min(o.attn_top, head_mean.shape[1])
        attn_top = []
        for row in head_mean:
            idx = np.argpartition(-row, k - 1)[:k]
            idx = idx[np.argsort(-row[idx])]
            attn_top.append([[int(i), round(float(row[i]), 4)] for i in idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(attn * np.log(np.where(attn > 0, attn, 1.0))).sum(axis=-1)  # [L, H]
        mlp = tr["mlp"]
        mk = min(o.mlp_top, mlp.shape[1])
        mlp_top = []
        for row in mlp:
            idx = np.argpartition(-np.abs(row), mk - 1)[:mk]
            idx = idx[np.argsort(-np.abs(row[idx]))]
            mlp_top.append([[int(i), round(float(row[i]), 3)] for i in idx])
        final_top = np.argpartition(-probs, o.final_top)[: o.final_top]
        final_top = final_top[np.argsort(-probs[final_top])]
        p = probs[probs > 0]
        return {
            "pos": int(tr["pos"]),
            "prob": round(float(probs[chosen]), 5),
            "entropy": round(float(-(p * np.log(p)).sum()), 4),
            # [text, p, id]: different byte tokens can decode to the same text (U+FFFD)
            "alternatives": [[self.token_text(int(t)), round(float(probs[t]), 4), int(t)] for t in final_top],
            "lens": lens,
            "norms": {"attn": [round(float(x), 3) for x in norms[:, 0]],
                      "mlp": [round(float(x), 3) for x in norms[:, 1]],
                      "resid": [round(float(x), 3) for x in resid_norm]},
            "attn_top": attn_top,
            "attn_entropy": [[round(float(x), 3) for x in row] for row in ent],
            "mlp_top": mlp_top,
        }

    # ---- generation ----------------------------------------------------------------
    def stream_chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0,
                    top_p: float = 1.0, top_k: int = 0, seed: int = 0, enable_thinking: bool = False,
                    should_stop=None) -> Iterator[dict]:
        """Yield {"type": "prefill" | "prompt" | "token" | "done", ...} events while generating a reply.

        The uncached part of the prompt is evaluated PREFILL_CHUNK tokens at a time; when that takes
        more than one chunk, a {"type": "prefill", "done", "total"} event follows every chunk but the
        last. Errors in the request itself (ValueError: empty, or too long for the context) are
        raised before the first event.

        should_stop: optional callable checked before each prompt chunk and between tokens (client
        disconnected, stop pressed). A stop during the prompt ends with "prompt" then a "done" event
        with no tokens."""
        llm = self.llm
        ctx = llm.ctx
        text = llm.render(messages, enable_thinking=enable_thinking)
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

        def prompt_event() -> dict:
            return {"type": "prompt", "tokens": [self.token_text(t) for t in prompt], "n_prompt": n,
                    "n_reused": reused, "prompt_ms": round((time.perf_counter() - t0) * 1000, 1)}

        for start in range(reused, n, PREFILL_CHUNK):
            if start > reused:
                yield {"type": "prefill", "done": start - reused, "total": total}
            if should_stop and should_stop():
                yield prompt_event()
                yield {"type": "done", "text": "", "n_gen": 0, "stop_reason": "cancelled", "seconds": 0.0,
                       "tok_per_s": 0.0}
                return
            end = min(start + PREFILL_CHUNK, n)
            if end < n:
                ctx.decode(prompt[start:end], logits=None)
                continue
            if end - 1 > start:
                ctx.decode(prompt[start:end - 1], logits=None)
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
            report = self.token_report(tok, probs)
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
