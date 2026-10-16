"""Speculative decoding and batched sampling speedups (bench/speedups.py TARGET.borg DRAFT.borg).

Prints tokens/second for plain greedy decoding, prompt lookup and a draft model on three
prompt types, and for sampling N answers sequentially versus in one batch."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import superborg  # noqa: E402

CODE = '''Add type hints and a docstring to every function. Output only the code.

def mean(xs):
    return sum(xs) / len(xs)

def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)

def zscores(xs):
    m = mean(xs)
    s = variance(xs) ** 0.5
    return [(x - m) / s for x in xs]
'''
DOC = ("The Hoover Dam is a concrete arch-gravity dam in the Black Canyon of the Colorado River, on the border "
       "between Nevada and Arizona. It was constructed between 1931 and 1936 during the Great Depression and was "
       "dedicated on September 30, 1935, by President Franklin D. Roosevelt. Its construction was the result of a "
       "massive effort involving thousands of workers, and cost over one hundred lives. The dam impounds Lake Mead, "
       "the largest reservoir in the United States by volume when it is full.\n\nUsing only the text above, when "
       "was the dam dedicated, by whom, and what reservoir does it impound? Quote the relevant sentences.")
OPEN = "Explain in two paragraphs why the sky is blue."


def main() -> int:
    target, draft = sys.argv[1], sys.argv[2]
    llm = superborg.LLM(target, threads=4, n_ctx=2048, draft=draft, n_draft=5)
    print(f"target {os.path.basename(target)}, draft {os.path.basename(draft)}")
    for name, prompt in (("code edit", CODE), ("document QA", DOC), ("open-ended", OPEN)):
        ids = llm.tokenize(llm.render([{"role": "user", "content": prompt}]), add_bos=False)
        res = {}
        for mode in ("plain", "lookup", "draft"):
            llm.ctx.reset()
            if llm.draft_ctx is not None:
                llm.draft_ctx.reset()
            kw = {"max_tokens": 160}
            if mode == "plain":
                kw["speculative"] = False
            elif mode == "lookup":
                kw.update(lookup=3)
                llm.draft_ctx, saved = None, llm.draft_ctx
            g = llm.generate_tokens(ids, **kw)
            if mode == "lookup":
                llm.draft_ctx = saved
            res[mode] = g
        base = res["plain"].tokens_per_second
        same = all(r.tokens == res["plain"].tokens for r in res.values())
        print(f"  {name:12s} plain {base:5.1f} tok/s | lookup {res['lookup'].tokens_per_second:5.1f} "
              f"(x{res['lookup'].tokens_per_second / base:.2f}) | draft {res['draft'].tokens_per_second:5.1f} "
              f"(x{res['draft'].tokens_per_second / base:.2f}) | identical output: {same}")
    msgs = [{"role": "user", "content": "A shop sells pens at 3 for $2. How much do 12 pens cost? Think step by step."}]
    for n in (4, 8):
        llm.ctx.reset()
        t = time.time()
        seq = [llm.chat(msgs, max_tokens=128, temperature=0.7, seed=i, speculative=False) for i in range(n)]
        t_seq = time.time() - t
        t = time.time()
        bat = llm.chat_n(msgs, n, max_tokens=128, temperature=0.7, seed=0)
        t_bat = time.time() - t
        ntok_s, ntok_b = sum(len(g.tokens) for g in seq), sum(len(g.tokens) for g in bat)
        print(f"  sample {n}: sequential {t_seq:5.1f}s ({ntok_s / t_seq:5.1f} tok/s) | batched {t_bat:5.1f}s "
              f"({ntok_b / t_bat:5.1f} tok/s) | x{(ntok_b / t_bat) / (ntok_s / t_seq):.2f} throughput")
    return 0


if __name__ == "__main__":
    sys.exit(main())
