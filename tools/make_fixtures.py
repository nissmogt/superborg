"""Write golden fixtures for the C engine's tests, so they can run without torch.

    python tools/make_fixtures.py OUT_DIR [--models smollm2-135m,qwen3-0.6b] [--no-ref] [--long 200]

For every model available under HF_MODELS_DIR this writes OUT_DIR/<key>.npz with

    prompt          int32 [T]        prompt token ids (HF tokenizer, BOS included if the model adds one)
    logits          float32 [T, V]   HF transformers float32 (eager attention) logits at every prompt position
    gen_tokens      int32 [N]        greedy continuation, N = 32, EOS does not stop it
    gen_logits      float32 [N, V]   HF's raw logits at each greedy step (KV-cached decoding);
                                     gen_logits[0] is the same position as logits[-1]
    ref_<q>_logits  float32 [T, V]   NumPy reference (superborg.reference) on <key>-<q>.borg for each
                                     quantized file found under BORG_MODELS_DIR: the exact math of the
                                     stored weights, which is what the engine should match for that file
    ref_<q>_gen_tokens int32 [N]     the reference's greedy continuation on that file

Long context (the short prompt leaves RoPE, the causal mask and the KV cache barely used; an
attention kernel that tiles keys can be wrong only past its first tile). L = 200 by default, the
first L tokens of superborg/data/calibration.txt; full [L, V] logits would be 120 MB for Qwen, so
only the tail is kept:

    long_prompt     int32 [L]        prompt token ids
    long_logits     float32 [K, V]   HF logits at the last K = 8 prompt positions (L-K .. L-1)
    long_argmax     int32 [L]        HF argmax at every prompt position
    long_margin     float32 [L]      HF top-1 minus top-2 logit (skip near-ties when comparing argmax)
    long_gen_tokens int32 [8]        greedy continuation after the long prompt (KV-cached decoding)
    long_gen_logits float32 [8, V]   HF's raw logits at each of those steps
    ref_<q>_long_logits float32 [K, V]  NumPy reference on <key>-<q>.borg at the last K prompt positions

    meta            str (JSON)       model key, HF directory, prompt text, library versions, metrics,
                                     and per quantized file its size and sha256 (ref_* arrays are
                                     stale once the file is reconverted)

Engine test guidance (measured, see tests/test_reference.py): the reference matches HF to
~1e-4 absolute (~3e-6 of the logit scale) on f32 files with identical argmax everywhere, on the
short prompt and at all 200 long-prompt positions alike.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import hf_oracle as ho  # noqa: E402
from superborg.reference import ReferenceModel, kl_divergence  # noqa: E402

REF_QTYPES = ["q8_0", "q4_0", "q4_1", "q4"]  # whichever <key>-<q>.borg files exist
LONG_TAIL = 8  # long-prompt positions whose full logits are stored


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_fixture(spec: ho.ModelSpec, out_dir: str, with_ref: bool = True, n_gen: int = ho.N_GEN,
                 n_long: int = ho.LONG_PROMPT_TOKENS) -> str:
    import torch
    import transformers

    r = ho.hf_run(spec.hf_path, n_gen=n_gen)  # loads, runs and frees the torch model
    arrays = {"prompt": r["tokens"], "logits": r["logits"], "gen_tokens": r["gen_tokens"],
              "gen_logits": r["gen_logits"]}
    meta = {"key": spec.key, "hf_dir": spec.hf_dir, "prompt_text": ho.PROMPT_TEXT, "n_gen": n_gen,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "hf_settings": "float32 weights, attn_implementation=eager, greedy, EOS ignored", "ref": {}}
    long_toks = None
    if n_long:
        long_toks = ho.long_prompt_tokens(spec.hf_path, n_long)
        rl = ho.hf_run(spec.hf_path, long_toks, n_gen=ho.LONG_N_GEN)
        top2 = np.partition(rl["logits"], -2, axis=1)[:, -2:]
        arrays.update(long_prompt=rl["tokens"], long_logits=rl["logits"][-LONG_TAIL:],
                      long_argmax=rl["logits"].argmax(-1).astype(np.int32),
                      long_margin=np.abs(top2[:, 1] - top2[:, 0]).astype(np.float32),
                      long_gen_tokens=rl["gen_tokens"], long_gen_logits=rl["gen_logits"])
        meta["long"] = {"n_prompt": n_long, "n_tail": LONG_TAIL, "n_gen": ho.LONG_N_GEN,
                        "source": "superborg/data/calibration.txt"}
        del rl
    if with_ref:
        prompt = r["tokens"].tolist()
        for q in REF_QTYPES:
            path = spec.borg_path(q)
            if not os.path.isfile(path):
                continue
            m = ReferenceModel(path)
            if long_toks is not None:
                arrays[f"ref_{q}_long_logits"] = m.forward(long_toks)[-LONG_TAIL:].astype(np.float32)
            lg = m.forward(prompt)
            gen = m.generate(prompt, n_gen, ignore_eos=True)
            del m
            gc.collect()
            arrays[f"ref_{q}_logits"] = lg.astype(np.float32)
            arrays[f"ref_{q}_gen_tokens"] = np.asarray(gen, dtype=np.int32)
            kl = kl_divergence(r["logits"], lg)
            meta["ref"][q] = {"file": os.path.basename(path), "bytes": os.path.getsize(path),
                              "sha256": sha256_file(path),
                              "top1_vs_hf": float((lg.argmax(-1) == r["logits"].argmax(-1)).mean()),
                              "kl_vs_hf_mean": float(kl.mean()),
                              "greedy_prefix_match": int(next((i for i, (a, b) in enumerate(zip(gen, r["gen_tokens"]))
                                                               if a != b), n_gen))}
    arrays["meta"] = np.array(json.dumps(meta))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{spec.key}.npz")
    np.savez(out + ".tmp.npz", **arrays)
    os.replace(out + ".tmp.npz", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--models", default=",".join(m.key for m in ho.MODELS), help="comma-separated model keys")
    ap.add_argument("--no-ref", action="store_true", help="skip the reference logits of quantized files")
    ap.add_argument("--n-gen", type=int, default=ho.N_GEN)
    ap.add_argument("--long", type=int, default=ho.LONG_PROMPT_TOKENS,
                    help="long-prompt length for the long_* arrays (0: none)")
    a = ap.parse_args(argv)
    if not ho.hf_available():
        print("torch and transformers are required", file=sys.stderr)
        return 1
    specs = {m.key: m for m in ho.MODELS}
    for key in a.models.split(","):
        spec = specs[key]
        if not os.path.isfile(os.path.join(spec.hf_path, "config.json")):
            print(f"skip {key}: {spec.hf_path} not found", file=sys.stderr)
            continue
        path = make_fixture(spec, a.out_dir, not a.no_ref, a.n_gen, a.long)
        meta = json.loads(str(np.load(path)["meta"]))
        print(f"wrote {path}: {len(np.load(path)['prompt'])} prompt tokens, refs {json.dumps(meta['ref'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
