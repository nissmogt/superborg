"""Measure quantization loss with the engine alone.

The f32 .borg model reproduces Hugging Face logits to ~5e-6 relative error, so it
serves as the reference: for each quantized file we report the mean KL divergence
KL(f32 || quant) per token, top-1 agreement and perplexity on a text sample.

    python bench/quant_eval.py --ref model-f32.borg model-q8_0.borg model-q4_0.borg --text file.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superborg.engine import Model, Pool  # noqa: E402


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def all_logits(model: Model, tokens: list[int]) -> np.ndarray:
    ctx = model.context(n_ctx=len(tokens) + 8)
    out = ctx.decode(tokens, logits="all")[:, : model.n_vocab]
    ctx.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="f32 .borg used as the reference")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    pool = Pool(a.threads)
    with open(a.text, encoding="utf-8") as f:
        text = f.read()
    ref = Model(a.ref, pool=pool)
    toks = ref.tokenize(text)[: a.tokens]
    lp_ref = log_softmax(all_logits(ref, toks))
    p_ref = np.exp(lp_ref)
    tgt = np.asarray(toks[1:])

    def ppl(lp):
        return float(np.exp(-lp[np.arange(len(tgt)), tgt].mean()))

    print(f"{'model':40s} {'size MB':>8s} {'KL':>9s} {'top1':>6s} {'ppl':>8s}")
    print(f"{os.path.basename(a.ref):40s} {os.path.getsize(a.ref) / 1e6:8.1f} {0:9.5f} {1:6.3f} {ppl(lp_ref[:-1]):8.3f}")
    ref.close()
    for path in a.models:
        m = Model(path, pool=pool)
        lp = log_softmax(all_logits(m, toks))
        kl = (p_ref * (lp_ref - lp)).sum(axis=-1).mean()
        top1 = (lp.argmax(-1) == lp_ref.argmax(-1)).mean()
        print(f"{os.path.basename(path):40s} {os.path.getsize(path) / 1e6:8.1f} {kl:9.5f} {top1:6.3f} {ppl(lp[:-1]):8.3f}")
        m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
