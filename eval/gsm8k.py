"""GSM8K accuracy of the reasoning strategies on a local .borg model.

    python -m eval.gsm8k --model qwen2.5-0.5b-q8_0.borg --data gsm8k_test.jsonl --n 40 \\
        --strategies direct,cot,vote,pal,pal_vote --samples 5 --threads 4 --out results.json

Prints accuracy, seconds/question and generated tokens/question per strategy and saves
every per-question record (with each candidate's answer, and its program for PAL).
Runs are deterministic for a given --seed and resumable: re-running with the same --out
skips (problem, strategy) pairs that are already recorded. A results file is only resumed
when it was written with the same model, data file and settings and its problems match
the data; otherwise the run refuses to start (use another --out). The summary covers the
problems of the current run only.
The data file is the GSM8K test split as JSON lines with "question" and "answer"; the gold
answer is the number after "####".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

from superborg.reason import STRATEGIES, answers_match, normalize_number, solve

DEFAULT_DATA = os.environ.get("GSM8K_PATH", os.path.join("data", "gsm8k_test.jsonl"))


def load_problems(path: str, offset: int = 0, n: int | None = None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset or not line.strip():
                continue
            if n is not None and len(out) >= n:
                break
            d = json.loads(line)
            gold = d["answer"].rsplit("####", 1)[-1].strip()
            out.append({"index": i, "question": d["question"], "gold": normalize_number(gold) or gold})
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (honest for small n)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(wins: int, losses: int) -> float:
    """Exact two-sided McNemar test: p value of `wins` against `losses` discordant pairs
    (problems only one of two strategies solved) under the null of equal accuracy."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(records: list[dict], strategies: list[str]) -> dict:
    out = {}
    by = {s: {r["index"]: r for r in records if r["strategy"] == s} for s in strategies}
    base = strategies[0] if strategies else None
    for s in strategies:
        rs = list(by[s].values())
        if not rs:
            continue
        k = sum(r["correct"] for r in rs)
        lo, hi = wilson(k, len(rs))
        row = {"n": len(rs), "correct": k, "accuracy": k / len(rs), "ci95": [lo, hi],
               "sec_per_q": sum(r["elapsed_s"] for r in rs) / len(rs),
               "tokens_per_q": sum(r["tokens"] for r in rs) / len(rs),
               "samples_per_q": sum(r["n_candidates"] for r in rs) / len(rs)}
        if base and s != base:
            common = [i for i in by[s] if i in by[base]]
            row["wins_vs_" + base] = sum(by[s][i]["correct"] and not by[base][i]["correct"] for i in common)
            row["losses_vs_" + base] = sum(by[base][i]["correct"] and not by[s][i]["correct"] for i in common)
            row["p_vs_" + base] = mcnemar(row["wins_vs_" + base], row["losses_vs_" + base])
        out[s] = row
    return out


def format_table(summary: dict, base: str | None = None) -> str:
    lines = [f"{'strategy':10s} {'acc':>6s} {'correct':>8s} {'95% CI':>13s} {'s/q':>7s} {'tok/q':>7s} "
             f"{'samples':>7s} {'vs ' + base if base else '':>9s} {'p' if base else '':>6s}",
             "-" * 81]
    for s, r in summary.items():
        vs = p = ""
        if base and s != base:
            vs = f"+{r['wins_vs_' + base]}/-{r['losses_vs_' + base]}"
            if "p_vs_" + base in r:
                p = f"{r['p_vs_' + base]:.3f}"
        lines.append(f"{s:10s} {100 * r['accuracy']:5.1f}% {r['correct']:>4d}/{r['n']:<3d} "
                     f"{100 * r['ci95'][0]:5.1f}-{100 * r['ci95'][1]:4.1f}% {r['sec_per_q']:7.1f} "
                     f"{r['tokens_per_q']:7.0f} {r['samples_per_q']:7.1f} {vs:>9s} {p:>6s}")
    return "\n".join(lines)


def _trim_candidate(c: dict) -> dict:
    keep = ("kind", "answer", "tokens", "seconds", "temperature", "seed", "stop_reason", "ok", "error", "value",
            "logprob", "p_yes", "tiebreaker")
    d = {k: c[k] for k in keep if k in c}
    if "code" in c:
        d["code"] = c["code"][:2000]
        if len(c.get("attempts", [])) > 1:
            d["repaired"] = True
    if c.get("kind") in ("cot", "direct"):
        d["text"] = c.get("text", "")[:2500]
    return d


def _save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


# config entries that change the answers (besides evaluate()'s own arguments)
RESUME_CONFIG_KEYS = ("model", "data", "n_ctx")


def run_settings(config: dict | None, samples: int, max_tokens: int, seed: int, solve_kw: dict) -> dict:
    """What a results file must have been produced with to be resumed."""
    s = {k: (config or {}).get(k) for k in RESUME_CONFIG_KEYS}
    s.update(samples=samples, max_tokens=max_tokens, seed=seed)
    for k in sorted(solve_kw):
        if solve_kw[k] is None or isinstance(solve_kw[k], (bool, int, float, str)):
            s[k] = solve_kw[k]
    return s


def _qid(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def resumable(out: str | None, settings: dict, problems: list[dict], strategies: list[str]
              ) -> tuple[list[dict], list[dict]]:
    """(records of `out` this run reuses, other records kept in the file untouched).

    Raises ValueError when `out` was written with other settings or for other problems:
    its records would be reported as this run's.
    """
    if not out or not os.path.exists(out):
        return [], []
    with open(out, encoding="utf-8") as f:
        prev = json.load(f)
    old = prev.get("settings") if isinstance(prev, dict) else None
    if old != settings:
        if not isinstance(old, dict):
            why = "no recorded settings"
        else:
            why = ", ".join(f"{k}: {old.get(k)!r} -> {settings.get(k)!r}" for k in sorted(set(old) | set(settings))
                            if old.get(k) != settings.get(k))
        raise ValueError(f"{out} holds results of another run ({why}); use another --out or delete it")
    qids = {p["index"]: _qid(p["question"]) for p in problems}
    reuse: list[dict] = []
    other: list[dict] = []
    for r in prev.get("records", []):
        i = r.get("index")
        if i in qids and r.get("qid") != qids[i]:
            raise ValueError(f"{out}: problem #{i} differs from the one in the data file; use another --out")
        (reuse if i in qids and r.get("strategy") in strategies else other).append(r)
    return reuse, other


def evaluate(llm, problems: list[dict], strategies: list[str], samples: int = 5, max_tokens: int = 512,
             seed: int = 0, out: str | None = None, config: dict | None = None, verbose: bool = True,
             **solve_kw) -> dict:
    """Run every strategy on every problem (problem-major, so partial runs stay comparable).

    With `out`, records are saved after every problem and an existing file is resumed (see
    resumable(); ValueError if it belongs to another run). The summary covers `problems` only.
    """
    settings = run_settings(config, samples, max_tokens, seed, solve_kw)
    records, other = resumable(out, settings, problems, strategies)
    if verbose and records:
        print(f"resuming: {len(records)} records already in {out}", file=sys.stderr)
    done = {(r["index"], r["strategy"]) for r in records}
    t_start = time.time()
    for pi, p in enumerate(problems):
        for s in strategies:
            if (p["index"], s) in done:
                continue
            # start from an empty KV cache: prefill chunking changes float rounding, so reusing
            # whatever the previous question left behind would make greedy runs order-dependent
            ctx = getattr(llm, "ctx", None)
            if ctx is not None and hasattr(ctx, "reset"):
                ctx.reset()
            ans = solve(llm, p["question"], strategy=s, samples=samples, max_tokens=max_tokens,
                        seed=seed + p["index"], numeric=True, **solve_kw)
            ok = answers_match(ans.answer, p["gold"])
            rec = {"index": p["index"], "qid": _qid(p["question"]), "strategy": s, "gold": p["gold"],
                   "pred": ans.answer, "correct": ok,
                   "elapsed_s": round(ans.elapsed_s, 3), "tokens": ans.tokens,
                   "n_candidates": len(ans.candidates), "resolved": ans.strategy,
                   "candidates": [_trim_candidate(c) for c in ans.candidates]}
            records.append(rec)
            if verbose:
                mark = "ok " if ok else "BAD"
                print(f"[{pi + 1:3d}/{len(problems)}] #{p['index']:<5d} {s:9s} {mark} pred={ans.answer!s:>10.10s} "
                      f"gold={p['gold']:>8s} {ans.elapsed_s:6.1f}s {ans.tokens:5d} tok "
                      f"({len(ans.candidates)} cand)", file=sys.stderr, flush=True)
        if out:
            _save(out, {"config": config or {}, "settings": settings, "summary": summarize(records, strategies),
                        "records": other + records})
    summary = summarize(records, strategies)
    if verbose:
        print(f"done in {time.time() - t_start:.0f}s", file=sys.stderr)
    data = {"config": config or {}, "settings": settings, "summary": summary, "records": records}
    if out:
        _save(out, dict(data, records=other + records))
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".borg model file")
    ap.add_argument("--data", default=DEFAULT_DATA, help="GSM8K test JSONL (default: $GSM8K_PATH or %(default)s)")
    ap.add_argument("--n", type=int, default=40, help="number of problems")
    ap.add_argument("--offset", type=int, default=0, help="index of the first problem")
    ap.add_argument("--strategies", default="direct,cot,pal,pal_vote",
                    help=f"comma-separated, from: {', '.join(s for s in STRATEGIES if s not in ('rag', 'calc'))}")
    ap.add_argument("--samples", type=int, default=5, help="samples for vote/pal_vote/verify")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-early-stop", action="store_true", help="always draw all samples")
    ap.add_argument("--thinking", action="store_true", help="enable thinking mode (Qwen3)")
    ap.add_argument("--out", help="JSON file for per-question records (resumes if it exists)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in STRATEGIES or s in ("rag",)]
    if bad:
        ap.error(f"unknown strategies: {bad}")
    problems = load_problems(a.data, a.offset, a.n)
    import superborg
    config = {"model": os.path.basename(a.model), "data": os.path.basename(a.data), "offset": a.offset,
              "n": len(problems), "strategies": strategies, "samples": a.samples, "max_tokens": a.max_tokens,
              "seed": a.seed, "temperature": a.temperature, "early_stop": not a.no_early_stop,
              "thinking": a.thinking, "n_ctx": a.n_ctx, "threads": a.threads, "simd": superborg.simd()}
    solve_kw = {"temperature": a.temperature, "early_stop": not a.no_early_stop, "enable_thinking": a.thinking}
    try:  # refuse a results file of another run before spending time on loading the model
        resumable(a.out, run_settings(config, a.samples, a.max_tokens, a.seed, solve_kw), problems, strategies)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    llm = superborg.LLM(a.model, threads=a.threads, n_ctx=a.n_ctx)
    try:
        data = evaluate(llm, problems, strategies, samples=a.samples, max_tokens=a.max_tokens, seed=a.seed,
                        out=a.out, config=config, verbose=not a.quiet, **solve_kw)
    finally:
        llm.close()
    print(f"GSM8K test #{a.offset}..#{a.offset + len(problems) - 1} | {config['model']} | samples={a.samples}")
    print(format_table(data["summary"], strategies[0] if strategies else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
