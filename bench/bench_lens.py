"""Cost of the logit lens per generated token: the old path (borg_ctx_lens writes every
layer's logits, NumPy summarizes them) against the fused one (borg_ctx_lens_summary keeps the
logits in C), next to the decode step itself.

  bench/bench_lens.py [--models-dir DIR] [--tokens 32] [--runs 3] [--repeat 2] [--wasm build/borg.wasm]

For smollm2-135m-q8_0 and qwen2.5-0.5b-q8_0, at lens stride 1 and 3, with 1 and 4 threads: a
chat prompt is prefilled, then `tokens` greedy tokens are decoded and both lens paths are timed
on each; a run's figure is the mean per token, a cell's figure the median of `runs` runs. The
whole grid is measured `repeat` times and the minimum of each cell is reported, which is the
figure least affected by other work on the machine. The same for the WebAssembly engine under
node (smollm2-135m-q4_0: the old path with the JavaScript summary against the fused one), through
tests/js/wasm_cases.js lensBench.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from superborg import LLM  # noqa: E402
from superborg.interp import TraceOptions, TraceSession  # noqa: E402

MODELS_DIR = os.environ.get("BORG_MODELS_DIR", os.path.join(ROOT, "models"))
NATIVE = ["smollm2-135m-q8_0", "qwen2.5-0.5b-q8_0"]
WASM = "smollm2-135m-q4_0"
CHAT = [{"role": "user", "content": "Explain in two paragraphs why the sky is blue."}]


def one_run(llm, session: TraceSession, prompt: list[int], tokens: int) -> dict:
    """Mean milliseconds per token of decode, the old lens path and the fused one."""
    ctx = llm.ctx
    ctx.reset()
    logits = ctx.decode(prompt, "last")
    t = {"decode": 0.0, "old": 0.0, "fused": 0.0}
    for _ in range(tokens):
        tok = int(logits[: llm.model.n_vocab].argmax())
        t0 = time.perf_counter()
        session.opt.fused_lens = False
        session.lens_report(tok)
        t1 = time.perf_counter()
        session.opt.fused_lens = True
        session.lens_report(tok)
        t2 = time.perf_counter()
        logits = ctx.decode([tok], "last")
        t3 = time.perf_counter()
        t["old"] += t1 - t0
        t["fused"] += t2 - t1
        t["decode"] += t3 - t2
    return {k: v * 1000 / tokens for k, v in t.items()}


def native_grid(models_dir: str, tokens: int, runs: int, repeat: int) -> list[dict]:
    cells = []
    for name in NATIVE:
        path = os.path.join(models_dir, name + ".borg")
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found", file=sys.stderr)
            continue
        for threads in (1, 4):
            llm = LLM(path, threads=threads, n_ctx=512)
            prompt = llm.tokenize(llm.render(CHAT), add_bos=False)
            for stride in (1, 3):
                session = TraceSession(llm, TraceOptions(lens_stride=stride))
                best = None
                for _ in range(repeat):
                    rs = [one_run(llm, session, prompt, tokens) for _ in range(runs)]
                    med = {k: statistics.median(r[k] for r in rs) for k in rs[0]}
                    best = med if best is None else {k: min(best[k], med[k]) for k in med}
                cells.append({"model": name, "threads": threads, "stride": stride, "layers": len(session.lens_layers()),
                              **best})
                print(fmt(cells[-1]), flush=True)
            llm.close()
    return cells


def wasm_grid(models_dir: str, wasm: str, tokens: int, runs: int, repeat: int) -> list[dict]:
    path = os.path.join(models_dir, WASM + ".borg")
    if not os.path.exists(path) or not os.path.exists(wasm):
        print(f"skipping WebAssembly: {path} or {wasm} not found", file=sys.stderr)
        return []
    best = {}
    for _ in range(repeat):
        with tempfile.TemporaryDirectory() as d:
            req = os.path.join(d, "req.json")
            with open(req, "w") as f:
                json.dump({"wasm": wasm, "model": path, "chat": CHAT[0]["content"], "tokens": tokens, "runs": runs,
                           "strides": [1, 3]}, f)
            r = subprocess.run(["node", os.path.join(ROOT, "tests", "js", "wasm_cases.js"), "lensBench", req],
                               capture_output=True, text=True, check=True)
        out = json.loads(r.stdout)
        for stride, runs_ in out["strides"].items():
            med = {k[:-3]: statistics.median(v) for k, v in runs_.items() if k.endswith("_ms")}
            cell = best.setdefault(stride, {"model": WASM + " (wasm, node)", "threads": 1, "stride": int(stride),
                                            "layers": runs_["layers"], **med})
            for k in med:
                cell[k] = min(cell[k], med[k])
    cells = [best[s] for s in sorted(best, key=int)]
    for c in cells:
        print(fmt(c))
    return cells


def fmt(c: dict) -> str:
    return (f"{c['model']:28s} threads {c['threads']}  stride {c['stride']} ({c['layers']:2d} rows)  "
            f"decode {c['decode']:7.1f} ms  old lens {c['old']:7.1f} ms  fused {c['fused']:7.1f} ms  "
            f"({c['old'] / c['fused']:.2f}x, fused = {100 * c['fused'] / c['decode']:.0f}% of decode)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--wasm", default=os.path.join(ROOT, "build", "borg.wasm"))
    ap.add_argument("--no-wasm", action="store_true")
    ap.add_argument("--json", help="also write the cells to this file")
    a = ap.parse_args(argv)
    print(f"{a.tokens} tokens per run, median of {a.runs} runs, minimum over {a.repeat} repeats; ms per token")
    cells = native_grid(a.models_dir, a.tokens, a.runs, a.repeat)
    if not a.no_wasm:
        cells += wasm_grid(a.models_dir, a.wasm, a.tokens, a.runs, a.repeat)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(cells, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
