"""Fit the per-layer residual-stream basis the trace's "xyz" field is projected onto.

    python tools/trace_basis.py model.borg calibration.txt [-o model.basis.json] [--n-tokens 4096]
                                [--chunk 512] [--threads 4]

The model runs with tracing over the calibration text (plain text, or a .jsonl whose records'
string values are joined) in chunks of --chunk tokens, one token per forward pass so that every
position is traced. Per layer (the embedding and the output of each block) the residuals are
collected, excluding position 0 of each chunk and the chunk's first delimiter token (period or
newline: the positions that carry massive activations, Sun et al. 2024); their massive-activation
dimensions are detected on the raw values (any dimension whose |value| somewhere in the corpus
exceeds 100 times the median |value| of the layer) and clipped to that bound, the vectors are
then RMS-normalized (after the clip, so a massive dimension cannot squash the rest of the vector),
and a 3-component PCA is fitted. The output, next to the model as <model stem>.basis.json by
default, is what superborg/ui.py serves at /api/basis and wasm/make_artifact.py ships as
basis.json:

  {"layers": [{"mean": [D], "components": [[D], [D], [D]], "clip_dims": [...], "clip_to": c,
               "explained": [3]}, ...], "n_tokens": N, "model": name, "clip_ratio": 100}

clip_to is in the residual's own units; explained is the fraction of the (clipped, normalized,
centered) variance each component carries. superborg.interp.resid_report and
wasm/trace-session.js apply the same steps to a traced residual: clip the listed dims to
+-clip_to, RMS-normalize, subtract mean, dot with the components.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from superborg.interp import rms_normalize  # noqa: E402

CLIP_RATIO = 100.0  # a dimension is massive when its |value| exceeds this times the layer's median |value|
DELIMITERS = (".", "\n")


def read_text(path: str) -> str:
    """The calibration text: a .jsonl's string values joined by blank lines, or the file as is."""
    with open(path, encoding="utf-8", errors="replace") as f:
        if not path.endswith(".jsonl"):
            return f.read()
        parts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                parts.extend(v for v in rec.values() if isinstance(v, str))
            elif isinstance(rec, str):
                parts.append(rec)
        return "\n\n".join(parts)


def fit_layer(x: np.ndarray, clip_ratio: float = CLIP_RATIO) -> dict:
    """The basis of one layer from its residual vectors x [N, D] (raw): detect and clip the
    massive dimensions, RMS-normalize, center, PCA with 3 components (SVD)."""
    x = np.array(x, dtype=np.float64)
    ab = np.abs(x)
    clip_to = clip_ratio * float(np.median(ab))
    clip_dims = np.nonzero(ab.max(axis=0) > clip_to)[0]
    if clip_dims.size:
        x[:, clip_dims] = np.clip(x[:, clip_dims], -clip_to, clip_to)
    x = rms_normalize(x)
    mean = x.mean(axis=0)
    y = x - mean
    # the principal axes from the D x D scatter matrix (an SVD of the tall matrix is much slower
    # with numpy's bundled LAPACK); eigh gives ascending eigenvalues
    var, vecs = np.linalg.eigh(y.T @ y)
    order = np.argsort(var)[::-1]
    var, vecs = np.maximum(var[order], 0.0), vecs[:, order]
    total = float(var.sum())
    k = min(3, vecs.shape[1])
    comp = np.zeros((3, x.shape[1]))
    comp[:k] = vecs[:, :k].T
    explained = np.zeros(3)
    if total > 0:
        explained[:k] = var[:k] / total
    return {"mean": mean.tolist(), "components": comp.tolist(), "clip_dims": clip_dims.tolist(),
            "clip_to": clip_to, "explained": explained.tolist()}


def fit_basis(resid: np.ndarray, clip_ratio: float = CLIP_RATIO) -> dict:
    """The basis for residuals [N, L + 1, D] (N traced positions): {"layers": [...], "n_tokens": N}."""
    n, n_layers1, _ = resid.shape
    return {"layers": [fit_layer(resid[:, l, :], clip_ratio) for l in range(n_layers1)], "n_tokens": int(n),
            "clip_ratio": clip_ratio}


def delimiter_ids(model) -> set[int]:
    """Token ids whose text is a period or a newline (the sentence delimiters that take the sink's role)."""
    out = set()
    for tok in range(model.n_vocab):
        if model.token_bytes(tok).decode("utf-8", errors="replace").strip(" ") in DELIMITERS:
            out.add(tok)
    return out


def collect(model, tokens: list[int], chunk: int, n_ctx: int, log=None) -> np.ndarray:
    """Residuals [N, L + 1, D] of the traced positions: the text in chunks of `chunk` tokens, each
    token its own forward pass, skipping position 0 and the first delimiter of every chunk."""
    cfg = model.config
    delim = delimiter_ids(model)
    ctx = model.context(n_ctx=max(chunk, 1) if n_ctx <= 0 else n_ctx)
    ctx.enable_trace()
    rows = []
    t0 = time.time()
    for start in range(0, len(tokens), chunk):
        ctx.reset()
        piece = tokens[start:start + chunk]
        first_delim = next((i for i, t in enumerate(piece) if t in delim), -1)
        for i, tok in enumerate(piece):
            ctx.decode([tok], logits=None)
            if i == 0 or i == first_delim:
                continue
            rows.append(ctx.trace(attention=False, mlp=False)["resid"])
        if log:
            log(f"  {min(start + chunk, len(tokens))}/{len(tokens)} tokens, {len(rows)} kept, {time.time() - t0:.0f} s")
    ctx.close()
    if not rows:
        raise SystemExit("no residuals collected (the text is too short)")
    return np.stack(rows).astype(np.float64).reshape(len(rows), cfg["n_layers"] + 1, cfg["dim"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("text", help="calibration text (.txt, or .jsonl of records with string fields)")
    ap.add_argument("-o", "--out", help="output file (default: <model stem>.basis.json next to the model)")
    ap.add_argument("--n-tokens", type=int, default=4096, help="calibration tokens to run (default 4096)")
    ap.add_argument("--chunk", type=int, default=512, help="tokens per context window (default 512)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--clip-ratio", type=float, default=CLIP_RATIO)
    a = ap.parse_args(argv)

    from superborg.engine import Model
    model = Model(a.model, threads=a.threads)
    text = read_text(a.text)
    tokens = model.tokenize(text, add_bos=False)[: a.n_tokens]
    if len(tokens) < a.n_tokens:
        print(f"note: the text has {len(tokens)} tokens, fewer than --n-tokens {a.n_tokens}", file=sys.stderr)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    log(f"{model.name}: {len(tokens)} tokens in chunks of {a.chunk}")
    resid = collect(model, tokens, a.chunk, a.chunk, log)
    basis = fit_basis(resid, a.clip_ratio)
    basis["model"] = model.name
    out = a.out or os.path.splitext(a.model)[0] + ".basis.json"
    with open(out, "w") as f:
        json.dump(basis, f)
    for l, b in enumerate(basis["layers"]):
        log(f"  layer {l:2d}: explained {' '.join(f'{e:.3f}' for e in b['explained'])}  clip_dims {b['clip_dims']}")
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {basis['n_tokens']} positions)")
    model.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
