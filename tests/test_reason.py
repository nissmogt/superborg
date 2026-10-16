"""Tests for the reasoning layer (superborg.reason, superborg.agent, eval.gsm8k).

Unit tests drive the strategies with a scripted fake LLM, so they need no model and run
in seconds. Integration tests at the bottom use a real model and are skipped when it is
missing (set BORG_TEST_MODEL to point at another .borg file).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields

import pytest

from superborg import reason
from superborg.agent import run_agent
from superborg.reason import (STRATEGIES, Answer, answer_key, answers_match, classify, extract_answer,
                              majority_vote, normalize_number, solve)


# =========================================================================================
# a scripted stand-in for superborg.LLM

@dataclass
class FakeGeneration:
    text: str
    tokens: list
    stop_reason: str


class FakeFormatter:
    def stop_strings(self):
        return ["<|im_end|>"]


class FakeLLM:
    """Implements the LLM surface the reasoning layer uses. `responder(call)` returns the
    model output for a call; stop strings and max_tokens (one token per character) apply."""

    def __init__(self, responder, scorer=None):
        self.responder = responder
        self.scorer = scorer
        self.calls: list[dict] = []
        self.scores: list[tuple[str, str]] = []
        self.formatter = FakeFormatter()

    def render(self, messages, enable_thinking=False, add_generation_prompt=True):
        return "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages) + "<|assistant|>"

    def tokenize(self, text, add_bos=None, special=True):
        return list(text.encode("utf-8"))

    def generate_tokens(self, prompt_tokens, max_tokens=256, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0,
                        repeat_penalty=1.0, seed=0x5EED, stop=(), stream=None, lookup=0, speculative=True):
        prompt = bytes(prompt_tokens).decode("utf-8")
        call = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                "seed": seed, "stop": list(stop), "lookup": lookup, "index": len(self.calls)}
        self.calls.append(call)
        out = self.responder(call)
        reason_ = "eos"
        cuts = [i for i in (out.find(s) for s in stop if s) if i >= 0]
        if cuts:
            out, reason_ = out[:min(cuts)], "stop"
        if len(out) > max_tokens:
            out, reason_ = out[:max_tokens], "length"
        if stream:
            stream(out)
        return FakeGeneration(text=out, tokens=list(range(len(out))), stop_reason=reason_)

    def score(self, prompt, continuation, add_bos=None):
        self.scores.append((prompt, continuation))
        return self.scorer(prompt, continuation) if self.scorer else -float(len(continuation))


class FakeBatchLLM(FakeLLM):
    """Adds sample_n (batched sampling); each sequence goes through the responder."""

    def __init__(self, responder, scorer=None):
        super().__init__(responder, scorer)
        self.batches: list[dict] = []

    def sample_n(self, prompt_tokens, n, max_tokens=256, temperature=0.7, top_k=0, top_p=1.0, min_p=0.0,
                 repeat_penalty=1.0, seed=0x5EED, stop=(), stream=None):
        self.batches.append({"n": n, "temperature": temperature, "seed": seed, "max_tokens": max_tokens})
        return [self.generate_tokens(prompt_tokens, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                     seed=seed + i, stop=stop) for i in range(n)]


def is_pal(call):
    return call["prompt"].endswith("```python\n")


def is_cot(call):
    return "step by step" in call["prompt"]


GSM_Q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with "
         "four. She sells the remainder for $2 per egg. How much in dollars does she make every day?")


# =========================================================================================
# extraction and normalization

@pytest.mark.parametrize("text,want", [
    ("so she makes 9 * 2 = 18 dollars.\n#### 18", "18"),
    ("The total is\nAnswer: $1,234.50", "1234.5"),
    ("Therefore the answer is \\boxed{42}.", "42"),
    ("\\boxed{\\frac{3}{4}}", "0.75"),
    ("\\boxed{\\$18.00}", "18"),
    ("\\boxed{1{,}000}", "1000"),
    ("\\boxed{18 \\text{ dollars}}", "18"),
    ("The answer is 18 dollars.", "18"),
    ("Answer: 16 - 3 - 4 = 9", "9"),
    ("Answer: 5 apples", "5"),
    ("**Answer:** 7", "7"),
    ("Final answer: 25%", "25"),
    ("Answer: -12", "-12"),
    ("Answer: 3/4", "0.75"),
    ("Answer: .5", "0.5"),
    ("She has 3 cats and 4 dogs, so 7 pets in total.", "7"),
    ("1,000,000 people", "1000000"),
    ("<think>maybe it is 3</think>Answer: 4", "4"),
    ("Answer: 1.333333333", "1.333333"),
    ("It costs 2.50 per pound and 10 pounds, so 25.", "25"),
    ("no numbers here", None),
    ("", None),
])
def test_extract_numeric(text, want):
    assert extract_answer(text, numeric=True) == want


def test_extract_prefers_last_answer_marker_and_boxed():
    assert extract_answer("Answer: 3\nWait, recheck.\nAnswer: 5", numeric=True) == "5"
    assert extract_answer("I think 3. \\boxed{8} then 99 later", numeric=True) == "8"


@pytest.mark.parametrize("text,want", [
    ("The capital is Paris.\nAnswer: Paris.", "Paris"),
    ("Answer: **Blue whale**", "Blue whale"),
    ("The answer is Mount Everest.", "Mount Everest"),
    ("Some prose without a marker", None),
])
def test_extract_text(text, want):
    assert extract_answer(text, numeric=False) == want


def test_extract_auto_kind():
    assert extract_answer("Answer: $18", numeric=None) == "18"
    assert extract_answer("Answer: World War 2", numeric=None) == "World War 2"
    assert extract_answer("so 41 + 1 = 42", numeric=None) == "42"


@pytest.mark.parametrize("value,want", [
    (18, "18"), (18.0, "18"), (18.000000001, "18"), (0.1 + 0.2, "0.3"), (2.5, "2.5"), (-3, "-3"),
    ("$1,234.50", "1234.5"), ("3/4", "0.75"), ("25%", "25"), ("1e3", "1000"), ("2.5e-1", "0.25"), ("x", None), (None, None),
    (True, None), (float("inf"), None), ([7], "7"), ("1/0", None),
])
def test_normalize_number(value, want):
    assert normalize_number(value) == want


def test_answers_match_and_keys():
    assert answers_match("18", "18.0") and answers_match("$1,000", "1000")
    assert not answers_match("17", "18") and not answers_match(None, "1")
    assert answers_match("The Beatles", "beatles")
    assert answer_key("18.0") == answer_key("18") == "18"
    assert answer_key("Paris!") == answer_key("paris") == "paris"
    assert answer_key(None) is None


# =========================================================================================
# voting and routing

def _c(key, lp=None):
    return {"key": key, "logprob": lp}


def test_majority_vote_basic_and_none():
    key, info = majority_vote([_c("3"), _c("5"), _c("5"), _c(None)])
    assert key == "5" and info["counts"] == {"3": 1, "5": 2} and not info["tie"]
    assert info["winner_index"] == 1
    assert majority_vote([_c(None)])[0] is None


def test_majority_vote_tie_breaks():
    # earliest group wins without log-probs (sample 0 is the greedy one)
    assert majority_vote([_c("3"), _c("5"), _c("5"), _c("3")])[0] == "3"
    # higher mean log-prob wins when every tied group has them
    cands = [_c("3", -0.9), _c("5", -0.2), _c("5", -0.4), _c("3", -0.7)]
    assert majority_vote(cands)[0] == "5"


def test_decided_early_stop_rule():
    assert reason._decided([_c("1"), _c("1"), _c("1")], 5)
    assert not reason._decided([_c("1"), _c("1"), _c("2")], 5)
    assert reason._decided([_c("1"), _c("1")], 3)
    assert not reason._decided([_c(None), _c(None)], 3)


@pytest.mark.parametrize("q,kind", [
    ("What is 17*23?", "arithmetic"),
    ("calculate (2+3)^2 / 5", "arithmetic"),
    ("15% of 80", "arithmetic"),
    (GSM_Q, "math"),
    ("A train travels 60 miles per hour for 3 hours. How far does it go?", "math"),
    ("Why is the sky blue?", "reasoning"),
    ("Hi there!", "simple"),
    ("What is the capital of France?", "simple"),
])
def test_classify(q, kind):
    assert classify(q)[0] == kind


def test_classify_context_means_rag():
    assert classify("anything", context="[1] text")[0] == "rag"


def test_strategies_contract():
    for name in ("direct", "cot", "vote", "pal", "pal_vote", "mix", "verify", "auto", "rag"):
        assert name in STRATEGIES and isinstance(STRATEGIES[name], str)
    names = [f.name for f in fields(Answer)]
    assert names[:6] == ["text", "answer", "strategy", "candidates", "elapsed_s", "tokens"]


# =========================================================================================
# strategies with the fake model

def test_direct_and_limits():
    llm = FakeLLM(lambda c: "She makes 18 dollars.\nAnswer: 18<|im_end|>ignored")
    seen = []
    ans = solve(llm, GSM_Q, strategy="direct", max_tokens=77, stream=seen.append)
    assert ans.answer == "18" and ans.strategy == "direct" and ans.tokens == len("She makes 18 dollars.\nAnswer: 18")
    assert llm.calls[0]["max_tokens"] == 77 and "<|im_end|>" in llm.calls[0]["stop"]
    assert llm.calls[0]["temperature"] == 0.0 and seen and ans.elapsed_s >= 0
    assert GSM_Q in llm.calls[0]["prompt"]


def test_max_tokens_truncates_output():
    llm = FakeLLM(lambda c: "x" * 1000 + " Answer: 3")
    ans = solve(llm, "q", strategy="cot", max_tokens=50)
    assert ans.tokens == 50 and ans.candidates[0]["stop_reason"] == "length"


def test_cot_prompt_and_answer():
    llm = FakeLLM(lambda c: "16-3-4=9 eggs, 9*2=18.\nAnswer: 18")
    ans = solve(llm, GSM_Q, strategy="cot")
    assert ans.answer == "18" and is_cot(llm.calls[0]) and "Answer:" in llm.calls[0]["prompt"]
    assert ans.candidates[0]["kind"] == "cot" and "prompt" not in ans.candidates[0]


def test_vote_majority_seeds_and_temperatures():
    answers = ["7", "9", "9", "7", "9"]
    llm = FakeLLM(lambda c: f"reasoning...\nAnswer: {answers[c['index']]}")
    ans = solve(llm, GSM_Q, strategy="vote", samples=5, early_stop=False, seed=3)
    assert ans.answer == "9" and ans.strategy == "vote" and len(ans.candidates) == 5
    temps = [c["temperature"] for c in llm.calls]
    assert temps[0] == 0.0 and all(t == 0.7 for t in temps[1:])
    sampled_seeds = [c["seed"] for c in llm.calls[1:]]
    assert len(set(sampled_seeds)) == 4
    assert "3 of 5" in ans.text
    # same seed -> same calls (deterministic)
    llm2 = FakeLLM(lambda c: f"Answer: {answers[c['index']]}")
    solve(llm2, GSM_Q, strategy="vote", samples=5, early_stop=False, seed=3)
    assert [c["seed"] for c in llm2.calls] == [c["seed"] for c in llm.calls]


def test_vote_batched_sampling():
    answers = ["7", "9", "9", "7", "9"]
    llm = FakeBatchLLM(lambda c: f"Answer: {answers[c['index']]}")
    ans = solve(llm, "q", strategy="vote", samples=5, max_tokens=99)
    assert ans.answer == "9" and len(ans.candidates) == 5
    assert llm.batches == [{"n": 4, "temperature": 0.7, "seed": llm.batches[0]["seed"], "max_tokens": 99}]
    assert llm.calls[0]["temperature"] == 0.0  # greedy first, outside the batch
    assert [c.get("batched") for c in ans.candidates] == [None, 4, 4, 4, 4]
    # batch=False falls back to one-by-one sampling with early stopping
    llm2 = FakeBatchLLM(lambda c: "Answer: 1")
    solve(llm2, "q", strategy="vote", samples=5, batch=False)
    assert llm2.batches == [] and len(llm2.calls) == 3


def test_pal_vote_batched():
    vals = ["12", "13", "12", "12"]
    llm = FakeBatchLLM(lambda c: f"answer = {vals[c['index']]}\n```")
    ans = solve(llm, "q", strategy="pal_vote", samples=4)
    assert ans.answer == "12" and llm.batches[0]["n"] == 3
    assert all(c["kind"] == "program" and c["ok"] for c in ans.candidates)


def test_vote_stops_early_when_decided():
    llm = FakeLLM(lambda c: "Answer: 4")
    ans = solve(llm, "q", strategy="vote", samples=5)
    assert len(llm.calls) == 3 and ans.answer == "4" and len(ans.candidates) == 3


def test_vote_tie_uses_logprob_when_greedy_is_not_tied():
    answers = ["3", "5", "7", "5", "7"]
    llm = FakeLLM(lambda c: f"Answer: {answers[c['index']]}",
                  scorer=lambda p, cont: -1.0 if cont.endswith("7") else -50.0)
    ans = solve(llm, "q", strategy="vote", samples=5)
    assert ans.answer == "7" and len(llm.scores) == 4
    assert [c.get("logprob") is not None for c in ans.candidates] == [False, True, True, True, True]


def test_vote_tie_with_greedy_needs_no_scoring():
    answers = ["3", "5"]
    llm = FakeLLM(lambda c: f"Answer: {answers[c['index']]}", scorer=lambda p, cont: 0.0)
    ans = solve(llm, "q", strategy="vote", samples=2)
    assert ans.answer == "3" and llm.scores == []


def test_pal_runs_program():
    llm = FakeLLM(lambda c: "eggs = 16 - 3 - 4\nanswer = eggs * 2\n```\nExplanation that is cut off")
    ans = solve(llm, GSM_Q, strategy="pal")
    assert ans.answer == "18" and ans.strategy == "pal"
    call = llm.calls[0]
    assert is_pal(call) and "```" in call["stop"] and "`answer`" in call["prompt"]
    c = ans.candidates[0]
    assert c["kind"] == "program" and c["ok"] and c["value"] == 18 and "answer = eggs * 2" in c["code"]
    assert "Answer: 18" in ans.text


def test_pal_uses_printed_output():
    llm = FakeLLM(lambda c: "print('total', 6 * 7)\n```")
    assert solve(llm, "q", strategy="pal").answer == "42"


def test_pal_repairs_once_with_error_feedback():
    def responder(c):
        if "The program failed" in c["prompt"]:
            assert "ZeroDivisionError" in c["prompt"]
            return "answer = 10\n```"
        return "answer = 1 / 0\n```"
    llm = FakeLLM(responder)
    ans = solve(llm, "q", strategy="pal")
    assert ans.answer == "10" and len(llm.calls) == 2
    assert len(ans.candidates[0]["attempts"]) == 2 and not ans.candidates[0]["attempts"][0]["ok"]


def test_pal_falls_back_to_cot():
    llm = FakeLLM(lambda c: "Answer: 6" if is_cot(c) else "raise ValueError('nope')\n```")
    ans = solve(llm, "q", strategy="pal")
    assert ans.answer == "6" and [x["kind"] for x in ans.candidates] == ["program", "cot"]


def test_pal_vote_majority_and_early_stop():
    vals = ["12", "12", "13"]
    llm = FakeLLM(lambda c: f"answer = {vals[c['index']]}\n```")
    ans = solve(llm, "q", strategy="pal_vote", samples=3)
    assert ans.answer == "12" and len(llm.calls) == 2  # 2 of 3 agree -> third sample skipped
    assert "2 of 2" in ans.text


def test_pal_vote_tie_broken_by_cot():
    vals = ["1", "2"]

    def responder(c):
        return "so... Answer: 2" if is_cot(c) else f"answer = {vals[c['index']]}\n```"
    llm = FakeLLM(responder)
    ans = solve(llm, "q", strategy="pal_vote", samples=2)
    assert ans.answer == "2"
    assert ans.candidates[-1]["kind"] == "cot" and ans.candidates[-1]["tiebreaker"]


def test_pal_vote_all_programs_fail():
    llm = FakeLLM(lambda c: "Answer: 5" if is_cot(c) else "undefined_name\n```")
    ans = solve(llm, "q", strategy="pal_vote", samples=2)
    assert ans.answer == "5" and ans.strategy == "pal_vote"


def test_plausibility_prior():
    from superborg.reason import plausibility
    assert plausibility("18", GSM_Q) == 1.0 and plausibility("2.5") == 0.6
    assert plausibility("-65000", GSM_Q) == 0.25 and plausibility("14.333333") == 0.3
    assert plausibility("-3", "The temperature fell from 5 to -3 degrees. What is it now?") == 1.0
    assert plausibility("Paris") == 1.0 and plausibility(None) == 0.0


def test_weighted_vote():
    cands = [{"key": "-5", "weight": 0.25}] * 3 + [{"key": "7", "weight": 1.0}]
    key, info = majority_vote(cands)
    assert key == "7" and info["counts"] == {"-5": 3, "7": 1} and info["scores"]["-5"] == 0.75


def test_mix_stops_when_program_and_solution_agree():
    llm = FakeBatchLLM(lambda c: "answer = 18\n```" if is_pal(c) else "16-3-4=9, 9*2=18\nAnswer: 18")
    ans = solve(llm, GSM_Q, strategy="mix", samples=5)
    assert ans.answer == "18" and ans.strategy == "mix" and len(llm.calls) == 2 and llm.batches == []
    assert "Answer: 18" in ans.text and "1 of 1 programs and 1 of 1 written" in ans.text


def test_mix_weighted_vote_over_both_formats():
    def responder(c):
        if is_pal(c):
            return "answer = -5\n```"
        return "Answer: 7" if c["index"] % 2 else "Answer: 9"
    llm = FakeBatchLLM(responder)
    ans = solve(llm, GSM_Q, strategy="mix", samples=3)
    # programs: three votes for -5 at weight 0.25; solutions: 7 and 9 split
    assert [b["n"] for b in llm.batches] == [2, 2]
    kinds = [c["kind"] for c in ans.candidates]
    assert kinds.count("program") == 3 and kinds.count("cot") == 3
    assert ans.answer in ("7", "9") and ans.answer != "-5"
    # without batching: one by one, same candidates
    llm2 = FakeLLM(responder)
    ans2 = solve(llm2, GSM_Q, strategy="mix", samples=3, early_stop=False)
    assert len(ans2.candidates) == 6


def test_verify_prefers_verified_minority():
    answers = ["6", "5", "6"]

    def scorer(prompt, cont):
        good = "Answer: 5" in prompt
        return {"Yes": -0.1 if good else -6.0, "No": -6.0 if good else -0.1}[cont]
    llm = FakeLLM(lambda c: f"work\nAnswer: {answers[c['index']]}", scorer=scorer)
    ans = solve(llm, "q", strategy="verify", samples=3, early_stop=False)
    assert ans.answer == "5" and ans.strategy == "verify"
    assert {c["answer"]: c.get("p_yes") for c in ans.candidates if "p_yes" in c}.keys() == {"5", "6"}


def test_verify_unanimous_skips_scoring():
    llm = FakeLLM(lambda c: "Answer: 2")
    ans = solve(llm, "q", strategy="verify", samples=3)
    assert ans.answer == "2" and llm.scores == []


def test_rag_uses_context_lookup_and_citations():
    ctx = "[1] (notes.md)\nThe launch code is 4721.\n\n[2] (other.md)\nUnrelated."
    llm = FakeLLM(lambda c: "The code is 4721 [1].")
    ans = solve(llm, "What is the launch code?", context=ctx)
    assert ans.strategy == "rag" and "4721" in ans.text and ans.route.startswith("rag")
    call = llm.calls[0]
    assert call["lookup"] == 3 and "The launch code is 4721." in call["prompt"] and "Cite" in call["prompt"]
    assert ans.candidates[0]["citations"] == ["1"]
    with pytest.raises(ValueError):
        solve(llm, "q", strategy="rag")


def test_rag_wraps_unnumbered_context():
    llm = FakeLLM(lambda c: "It is blue [1].")
    solve(llm, "color?", strategy="rag", context="The sky is blue.")
    assert "[1]\nThe sky is blue." in llm.calls[0]["prompt"]


def test_rag_trims_context_to_fit_window():
    llm = FakeLLM(lambda c: "It is 7 [1].")
    llm.n_ctx = 600
    ctx = "[1] (a.txt)\nThe number is 7.\n\n" + "\n\n".join(f"[{i}] (b.txt)\n" + "filler " * 40 for i in range(2, 30))
    ans = solve(llm, "What is the number?", context=ctx, max_tokens=100)
    assert ans.answer == "7"
    prompt = llm.calls[0]["prompt"]
    assert len(prompt) + 100 <= 600 and "The number is 7." in prompt


def test_auto_routes():
    llm = FakeLLM(lambda c: "answer = 18\n```" if is_pal(c) else "Hello! Answer: hi")
    a = solve(llm, "What is 12*(3+4)?")
    assert a.strategy == "calc" and a.answer == "84" and a.tokens == 0 and llm.calls == []
    a = solve(llm, GSM_Q, samples=3)
    assert a.strategy == "pal_vote" and a.answer == "18" and a.route.startswith("math")
    a = solve(llm, GSM_Q, samples=1)
    assert a.strategy == "pal"
    a = solve(llm, "Hi there!")
    assert a.strategy == "direct"
    a = solve(llm, "Why do cats purr?")
    assert a.strategy == "cot"


def test_unknown_strategy():
    with pytest.raises(ValueError):
        solve(FakeLLM(lambda c: ""), "q", strategy="magic")


def test_thinking_is_stripped():
    llm = FakeLLM(lambda c: "<think>it might be 3</think>\nAnswer: 4")
    ans = solve(llm, "q", strategy="cot")
    assert ans.answer == "4" and "<think>" not in ans.text


def test_stream_progress_for_sampling():
    llm = FakeLLM(lambda c: "Answer: 1")
    lines = []
    solve(llm, "q", strategy="vote", samples=3, stream=lines.append)
    assert any("sample 1/3" in x for x in lines)


# =========================================================================================
# agent loop

def test_agent_runs_code_then_answers():
    def responder(c):
        if "Output:" in c["prompt"]:
            assert "42" in c["prompt"]
            return "The product is 42.\nFinal Answer: 42"
        return "Let me compute.\n```python\nprint(6 * 7)\n```\nOutput: 41 (hallucinated)"
    llm = FakeLLM(responder)
    res = run_agent(llm, "What is 6 times 7?")
    assert res.answer == "42" and res.stop_reason == "final"
    assert len(res.steps) == 2 and res.steps[0].code == "print(6 * 7)" and "42" in res.steps[0].observation
    assert "hallucinated" not in llm.calls[1]["prompt"]


def test_agent_handles_bare_fence_and_max_steps():
    def responder(c):
        if c["prompt"].endswith("Sure:\n```\n"):
            return "print(1 + 1)\n```\n"
        if "Final Answer: <answer>" in c["prompt"] and c["prompt"].rstrip().endswith("<|assistant|>") \
                and "Now give" in c["prompt"]:
            return "Final Answer: 2"
        return "Sure:\n```\nprint(1 + 1)\n```\n"
    llm = FakeLLM(responder)
    res = run_agent(llm, "1+1?", max_steps=2)
    assert all(s.code == "print(1 + 1)" for s in res.steps[:2])
    assert res.answer == "2" and res.stop_reason == "max_steps"


def test_agent_without_tool_call():
    llm = FakeLLM(lambda c: "Paris is the capital.\nFinal Answer: Paris")
    res = run_agent(llm, "Capital of France?", numeric=False)
    assert res.answer == "Paris" and len(res.steps) == 1 and res.steps[0].code is None


# =========================================================================================
# evaluation harness

def test_gsm8k_harness_with_fake_model(tmp_path):
    from eval import gsm8k
    data = tmp_path / "gsm.jsonl"
    rows = [{"question": "Q one: 2 apples plus 3 apples?", "answer": "2+3=5\n#### 5"},
            {"question": "Q two: 1,000 minus 1?", "answer": "#### 999"},
            {"question": "Q three", "answer": "#### 1,200"}]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    probs = gsm8k.load_problems(str(data), offset=1, n=2)
    assert [p["gold"] for p in probs] == ["999", "1200"] and probs[0]["index"] == 1
    llm = FakeLLM(lambda c: "answer = 999\n```" if is_pal(c) else "Answer: 1200")
    out = tmp_path / "res.json"
    res = gsm8k.evaluate(llm, probs, ["cot", "pal"], out=str(out), verbose=False)
    s = res["summary"]
    assert s["cot"]["correct"] == 1 and s["pal"]["correct"] == 1
    assert s["pal"]["wins_vs_cot"] == 1 and s["pal"]["losses_vs_cot"] == 1
    saved = json.loads(out.read_text())
    assert len(saved["records"]) == 4
    # resume: nothing is recomputed
    n_calls = len(llm.calls)
    gsm8k.evaluate(llm, probs, ["cot", "pal"], out=str(out), verbose=False)
    assert len(llm.calls) == n_calls
    table = gsm8k.format_table(s, "cot")
    assert "cot" in table and "50.0%" in table
    lo, hi = gsm8k.wilson(20, 40)
    assert 0.34 < lo < 0.36 and 0.64 < hi < 0.66


# =========================================================================================
# integration with a real model (skipped when it is not available)

MODEL = os.environ.get("BORG_TEST_MODEL",
                       "/tmp/claude-0/-home-user-superborg/990fbcaa-dc72-53ae-9121-cd54b34a4307/scratchpad/borg/"
                       "qwen2.5-0.5b-q8_0.borg")
needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model not found: {MODEL}")


@pytest.fixture(scope="module")
def llm():
    if not os.path.exists(MODEL):
        pytest.skip("no model")
    try:
        import superborg
        m = superborg.LLM(MODEL, threads=int(os.environ.get("BORG_TEST_THREADS", "4")), n_ctx=1024)
    except (OSError, RuntimeError, FileNotFoundError) as e:  # library not built
        pytest.skip(f"cannot load model: {e}")
    yield m
    m.close()


@needs_model
def test_integration_pal(llm):
    ans = solve(llm, "Tom has 3 boxes with 12 pencils in each box. He gives away 5 pencils. "
                     "How many pencils does he have left?", strategy="pal", max_tokens=200)
    assert ans.answer == "31", ans.candidates


@needs_model
def test_integration_cot_and_vote_early_stop(llm):
    q = "What is 25 plus 17?"
    ans = solve(llm, q, strategy="cot", max_tokens=200)
    assert ans.answer == "42"
    ans = solve(llm, q, strategy="vote", samples=3, max_tokens=200)
    assert ans.answer == "42" and len(ans.candidates) <= 3


@needs_model
def test_integration_rag(llm):
    ctx = "[1] (facts.txt)\nThe office wifi password is 'tangerine-42'.\n\n[2] (menu.txt)\nLunch is at noon."
    ans = solve(llm, "What is the office wifi password?", context=ctx, max_tokens=60)
    assert ans.strategy == "rag" and "tangerine-42" in ans.text


@needs_model
def test_integration_agent(llm):
    res = run_agent(llm, "Use Python to compute 123456789 * 987654321.", max_steps=3, max_tokens=200)
    assert any(s.code for s in res.steps)
    assert res.answer == "121932631112635269", (res.text, [s.observation for s in res.steps])
