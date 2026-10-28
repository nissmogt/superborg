"""Prefill throughput against the chunk size (bench/bench_nbatch.py MODEL.borg [THREADS] [PROMPT] [REPS]).

A prompt is evaluated in chunks of n_batch tokens; each chunk streams every weight once, so
larger chunks trade activation-panel cache residency for fewer passes over the weights. Prints
the best of REPS runs for n_batch in 32, 64, 128, 256 (the benchmark machine is shared)."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superborg import engine  # noqa: E402


def main() -> int:
    path = sys.argv[1]
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_prompt = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    pool = engine.Pool(threads)
    m = engine.Model(path, pool=pool)
    toks = [(i * 7919 + 13) % m.n_vocab for i in range(n_prompt)]
    print(f"{os.path.basename(path)} threads={threads} prompt={n_prompt} (best of {reps}, load {os.getloadavg()[0]:.1f})")
    for n_batch in (32, 64, 128, 256):
        best = float("inf")
        for _ in range(reps):
            ctx = m.context(n_ctx=n_prompt + 8, n_batch=n_batch)
            ctx.decode(toks[:8], logits=None)  # warm up
            ctx.reset()
            t0 = time.perf_counter()
            ctx.decode(toks, logits="last")
            best = min(best, time.perf_counter() - t0)
            ctx.close()
        print(f"  n_batch={n_batch:3d}: {n_prompt / best:7.1f} tok/s ({best * 1000:.0f} ms)")
    m.close()
    pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
