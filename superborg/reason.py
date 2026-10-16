"""Test-time reasoning: make a small local model answer better by spending more compute.

    import superborg
    from superborg.reason import solve
    llm = superborg.LLM("qwen2.5-0.5b-q8_0.borg", threads=4)
    ans = solve(llm, "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                     "How many bolts in total does it take?", strategy="auto")
    print(ans.answer, ans.strategy, ans.elapsed_s)

Strategies (see STRATEGIES): plain answering, chain of thought, self-consistency voting
(Wang et al. 2022), program-aided reasoning (Gao et al. 2022) with execution in a
resource-limited subprocess, voting over programs, model self-verification scored with
log-probabilities, and grounded answering over retrieved text. Sampling strategies stop
early once the remaining samples cannot change the majority (adaptive consistency), so
easy questions cost little more than one sample.
"""
from __future__ import annotations

import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Callable

from . import tools

__all__ = ["Answer", "solve", "STRATEGIES", "extract_answer", "normalize_number", "normalize_text",
           "answer_key", "majority_vote", "classify", "answers_match"]

STRATEGIES: dict[str, str] = {
    "auto": "pick a strategy from the question: arithmetic -> exact calculator, math word problem -> "
            "pal_vote, other reasoning -> cot, short questions -> direct, given context -> rag",
    "direct": "one greedy answer to the question as asked (the model's default behaviour)",
    "cot": "one greedy step-by-step solution ending in 'Answer: ...'",
    "vote": "self-consistency: several sampled step-by-step solutions, majority vote on the final answer",
    "pal": "program-aided: the model writes Python that computes the answer; it runs in a sandboxed "
           "subprocess, with one repair attempt on error",
    "pal_vote": "several sampled programs are executed and their results majority-voted",
    "mix": "programs and written step-by-step solutions vote together, weighted by answer plausibility; "
           "stops early when the greedy program and the greedy solution agree",
    "verify": "sampled solutions are checked by the model itself (log-prob of 'Yes'), combined with the vote",
    "rag": "answer grounded in the given context (retrieved text) with [n] citations; prompt-lookup decoding",
    "calc": "evaluate a bare arithmetic expression exactly, without the model",
}


@dataclass
class Answer:
    text: str                  # final user-facing answer
    answer: str | None         # extracted short answer (a canonical number or a short phrase)
    strategy: str              # strategy actually used ("auto" resolves to one of the others)
    candidates: list = field(default_factory=list)  # per-sample details (dicts)
    elapsed_s: float = 0.0
    tokens: int = 0            # generated tokens over all samples
    route: str = ""            # why auto chose the strategy (empty when chosen explicitly)


# =========================================================================================
# answer extraction and normalization

_NUM_RE = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
                     r"(?:/\d+(?:\.\d+)?)?(?![\w])|(?<![\w.\d])[-+−]?\.\d+")
_ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*(?:is)?\s*\**\s*[:：]\s*\**\s*(.+)", re.I)
_ANSWER_IS_RE = re.compile(r"(?:final answer|the answer|answer)\s+is\s*:?\s*(.+)", re.I)


def _latex_to_text(s: str) -> str:
    """Strip the LaTeX that small models put around numbers (\\$, \\text{}, \\frac{}{}, {,})."""
    s = s.replace("{,}", ",").replace("\\!", "").replace("\\,", "").replace("\\;", " ").replace("\\ ", " ")
    s = s.replace("\\$", "$").replace("\\%", "%").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    for _ in range(3):
        s = re.sub(r"\\[dt]?frac\s*\{\s*(\d+(?:\.\d+)?)\s*\}\s*\{\s*(\d+(?:\.\d+)?)\s*\}", r"\1/\2", s)
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\(?:text|textbf|mathrm|mbox|textit|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\^\s*\{?\\circ\}?", "", s)
    return s


def _boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} (nested braces allowed)."""
    i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        m = re.match(r"\\boxed\s+(\S+)", text[i:])
        return m.group(1) if m else None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return text[j + 1:]


def _fmt_decimal(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    q = d.quantize(Decimal("1e-6")).normalize()
    s = format(q, "f")
    return "0" if s in ("-0", "0") else s


def normalize_number(s: Any) -> str | None:
    """Canonical string for a numeric answer: '$1,234.50' -> '1234.5', '3/4' -> '0.75',
    '18.0' -> '18', '25%' -> '25'. Uses the first number in the string; None if there is none."""
    if s is None or isinstance(s, bool):
        return None
    if isinstance(s, (int, Fraction)):
        f = Fraction(s)
        return _fmt_decimal(Decimal(f.numerator) / Decimal(f.denominator))
    if isinstance(s, float):
        if not math.isfinite(s):
            return None
        return _fmt_decimal(Decimal(repr(s)))
    if isinstance(s, (list, tuple)):
        return normalize_number(s[0]) if len(s) == 1 else None
    text = _latex_to_text(str(s))
    m = _NUM_RE.search(text)
    if not m:
        return None
    tok = m.group(0).replace(",", "").replace("−", "-").lstrip("+")
    try:
        if "/" in tok:
            a, b = tok.split("/", 1)
            if Decimal(b) == 0:
                return None
            d = Decimal(a) / Decimal(b)
        else:
            d = Decimal(tok)
    except (InvalidOperation, ValueError):
        return None
    return _fmt_decimal(d)


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)


def normalize_text(s: str) -> str:
    """Loose text answer normal form (SQuAD style): lowercase, no punctuation/articles/extra spaces."""
    s = _latex_to_text(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:  # unfinished thought: nothing final to extract
        return text.split("<think>", 1)[0]
    return text


def _clean_short(seg: str) -> str:
    seg = seg.strip().strip("*").strip()
    seg = re.sub(r"^\$+|\$+$", "", seg).strip()
    seg = seg.split("\n", 1)[0].strip()
    return seg.rstrip(".").strip()


def _number_in_segment(seg: str) -> str | None:
    """Number in a final-answer segment: after the last '=' if it is an equation, else the first."""
    seg = _latex_to_text(seg)
    if "=" in seg:
        tail = seg.rsplit("=", 1)[1]
        n = normalize_number(tail)
        if n is not None:
            return n
    return normalize_number(seg)


def extract_answer(text: str, numeric: bool | None = None) -> str | None:
    """The final answer stated in a model's reply.

    Looks, in order, for '\\boxed{..}', '#### ..', 'Answer: ..' (last occurrence),
    'the answer is ..', and finally (numeric only) the last number in the text.
    numeric=True returns a canonical number, False a cleaned phrase, None decides by
    whether the stated answer starts with a number.
    """
    if not text:
        return None
    text = _strip_thinking(text)
    seg = None
    b = _boxed(text)
    if b is not None and b.strip():
        seg = b
    if seg is None:
        m = re.findall(r"####\s*(.+)", text)
        if m:
            seg = m[-1]
    if seg is None:
        m = _ANSWER_RE.findall(text)
        m = [x for x in m if x.strip(" *")]
        if m:
            seg = m[-1]
            # "Answer: \boxed{3}" or "Answer: **$18**" are handled by the generic cleanup below
    if seg is None:
        m = _ANSWER_IS_RE.findall(text)
        if m:
            seg = m[-1]
    if numeric is None:
        if seg is not None:
            numeric = bool(re.match(r"\s*\**\s*(?:\\?\$|€|£|¥)?\s*[-+−]?\.?\d", _latex_to_text(seg)))
        else:
            numeric = True
    if numeric:
        if seg is not None:
            n = _number_in_segment(seg)
            if n is not None:
                return n
        nums = _NUM_RE.findall(_latex_to_text(text))
        return normalize_number(nums[-1]) if nums else None
    if seg is None:
        return None
    seg = _clean_short(_latex_to_text(seg))
    return seg or None


def answer_key(ans: str | None) -> str | None:
    """Key under which answers are counted as equal when voting."""
    if ans is None:
        return None
    n = normalize_number(ans) if re.fullmatch(r"\s*[-+]?[\d.,/]+\s*", ans) else None
    return n if n is not None else normalize_text(ans) or None


def answers_match(pred: str | None, gold: str | None, tol: float = 1e-6) -> bool:
    """Numeric comparison with a small relative tolerance; text comparison otherwise."""
    if pred is None or gold is None:
        return False
    a, b = normalize_number(pred), normalize_number(gold)
    if a is not None and b is not None:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= tol * max(1.0, abs(fb))
    return normalize_text(pred) == normalize_text(gold)


_NEG_CUES = re.compile(r"(?<![\w])-\d|\b(?:negative|below zero|minus|loss|lost|lose|debt|owe[sd]?|deficit|"
                       r"temperature|degrees|decrease|difference|change)\b", re.I)


def plausibility(answer: str | None, question: str = "") -> float:
    """Prior weight of a numeric answer to a word problem, used to weigh votes.

    Answers to everyday quantity questions are almost always non-negative and whole (or
    money with cents), and a negative or 14.333333 result usually means a misread problem.
    Such answers are down-weighted, never excluded: they still win when nothing else does.
    Non-numeric answers weigh 1.
    """
    if answer is None:
        return 0.0
    if not re.fullmatch(r"\s*-?[\d.]+\s*", answer):
        return 1.0
    w = 1.0
    if answer.strip().startswith("-") and not _NEG_CUES.search(question):
        w *= 0.25
    if "." in answer:
        w *= 0.6 if len(answer.split(".", 1)[1].strip()) <= 2 else 0.3
    return w


def majority_vote(cands: list[dict]) -> tuple[str | None, dict]:
    """Answer key with the largest total vote weight (candidate "weight", default 1).

    Ties go to the higher mean log-probability when every tied group has one, else to
    the group whose first member came earliest (sample 0 is the greedy one).
    Returns (key, info) with info = {"counts": {key: n}, "scores": {key: weight},
    "tie": bool, "winner_index": i}.
    """
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, c in enumerate(cands):
        k = c.get("key")
        if k is not None:
            groups.setdefault(k, []).append(i)
    if not groups:
        return None, {"counts": {}, "scores": {}, "tie": False, "winner_index": None}
    counts = {k: len(v) for k, v in groups.items()}
    scores = {k: sum(float(cands[i].get("weight", 1.0)) for i in v) for k, v in groups.items()}
    top = max(scores.values())
    tied = [k for k, w in scores.items() if w >= top - 1e-9]

    def mean_lp(k):
        lps = [cands[i].get("logprob") for i in groups[k]]
        lps = [x for x in lps if x is not None]
        return sum(lps) / len(lps) if lps else None

    best = tied[0]
    if len(tied) > 1:
        lps = {k: mean_lp(k) for k in tied}
        if all(v is not None for v in lps.values()):
            best = max(tied, key=lambda k: (lps[k], -groups[k][0]))
        else:
            best = min(tied, key=lambda k: groups[k][0])
    return best, {"counts": counts, "scores": {k: round(v, 4) for k, v in scores.items()},
                  "tie": len(tied) > 1, "winner_index": groups[best][0]}


def _decided(cands: list[dict], total: int) -> bool:
    """True when the leader cannot be overtaken (or tied) by the samples still to come
    (each can add a weight of at most 1)."""
    scores: dict[str, float] = {}
    for c in cands:
        if c.get("key") is not None:
            scores[c["key"]] = scores.get(c["key"], 0.0) + float(c.get("weight", 1.0))
    if not scores:
        return False
    ranked = sorted(scores.values(), reverse=True)
    second = ranked[1] if len(ranked) > 1 else 0.0
    return ranked[0] > second + (total - len(cands)) + 1e-9


# =========================================================================================
# question routing

_WORD_NUMS = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|" \
             r"hundred|thousand|million|half|twice|double|triple|dozen|quarter|third)\b"
_MATH_CUES = re.compile(r"\b(?:how (?:many|much|long|far|old|fast)|what (?:is|was|are) the (?:total|sum|difference|"
                        r"product|average|number|cost|price|area|perimeter|value|percentage|ratio|amount)|"
                        r"calculate|compute|percent|average|total|profit|cost|price|per|each|remaining|left|"
                        r"more than|less than|times|altogether|in all|solve|equation|sum|product|ratio)\b|[$%€£]",
                        re.I)
_EXPR_RE = re.compile(r"^[\d\s.,+\-*/×÷^()%]+$|^(?:[\d\s.+\-*/×÷^()]|sqrt|log|ln|exp|sin|cos|tan|pi|factorial)+$",
                      re.I)


def _as_expression(q: str) -> str | None:
    """A bare arithmetic expression in the question ("what is 17*23?"), else None."""
    s = q.strip()
    s = re.sub(r"^(?:please\s+)?(?:what\s+is|what's|calculate|compute|evaluate|solve)\s*:?\s*", "", s, flags=re.I)
    s = s.rstrip(" ?.!=").replace(",", "")
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s+of\s+", r"\1/100*", s, flags=re.I)  # "15% of 80"
    if not s or not re.search(r"\d", s) or not re.search(r"[-+*/×÷^%]|sqrt|log|exp|sin|cos|tan|factorial", s):
        return None
    if not _EXPR_RE.match(s):
        return None
    return s


def classify(question: str, context: str | None = None) -> tuple[str, str]:
    """(kind, reason) with kind in {"rag", "arithmetic", "math", "reasoning", "simple"}."""
    if context:
        return "rag", "context was given"
    if _as_expression(question) is not None:
        return "arithmetic", "the question is a bare arithmetic expression"
    q = question.lower()
    n_numbers = len(re.findall(r"\d+(?:\.\d+)?", q)) + len(re.findall(_WORD_NUMS, q))
    cues = len(_MATH_CUES.findall(q))
    if n_numbers >= 2 and cues >= 1 or n_numbers >= 1 and cues >= 2:
        return "math", f"{n_numbers} quantities and {cues} quantitative cues"
    if len(question) > 200 or re.search(r"\b(?:why|explain|prove|puzzle|riddle|if .* then|which of|step|reason|"
                                        r"logic|deduce|compare)\b", q):
        return "reasoning", "long or reasoning-style question"
    return "simple", "short question"


# =========================================================================================
# prompts (kept short: small models degrade with long instructions)

COT_PROMPT = "{question}\n\nSolve this step by step. End with a final line of the form \"Answer: <answer>\"."
PAL_PROMPT = ("{question}\n\nWrite a short Python program that computes the answer to this problem. "
              "Use clear variable names and store the final result in a variable named `answer`.")
PAL_REPAIR = "The program failed:\n{error}\n\nWrite the corrected program. Store the final result in `answer`."
VERIFY_PROMPT = ("Question: {question}\n\nProposed solution:\n{solution}\n\n"
                 "Is the proposed solution correct? Reply with Yes or No.")
RAG_SYSTEM = ("Answer the question using only the numbered context passages. Cite the passages you used "
              "like [1]. If the context does not contain the answer, say that you don't know.")
CODE_FENCE = "```python\n"

# Optional worked examples, given as earlier chat turns: (question, assistant reply).
# For PAL the reply is the program body (it is wrapped in a code fence).
COT_EXAMPLES: list[tuple[str, str]] = []
PAL_EXAMPLES: list[tuple[str, str]] = []


def _with_examples(template: str, question: str, examples: list[tuple[str, str]], fence: bool = False) -> list[dict]:
    msgs: list[dict] = []
    for ex_q, ex_a in examples:
        msgs.append({"role": "user", "content": template.format(question=ex_q)})
        msgs.append({"role": "assistant", "content": f"{CODE_FENCE}{ex_a.strip()}\n```" if fence else ex_a})
    msgs.append({"role": "user", "content": template.format(question=question)})
    return msgs


# =========================================================================================
# generation helper

class _Session:
    """Wraps an LLM for one solve() call: renders chat prompts, counts tokens, records timing."""

    def __init__(self, llm, max_tokens: int, enable_thinking: bool, stream: Callable[[str], None] | None,
                 batch: bool = True):
        self.llm = llm
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.stream = stream
        self.tokens = 0
        # decode several samples in lockstep when the LLM supports it (weights are read once per step)
        self.batch = batch and callable(getattr(llm, "sample_n", None))
        self.question = ""
        self.prior = True  # weigh numeric votes by plausibility()

    def weight(self, answer: str | None) -> float:
        return plausibility(answer, self.question) if self.prior else (0.0 if answer is None else 1.0)

    def prompt(self, messages: list[dict], prefix: str = "") -> str:
        return self.llm.render(messages, enable_thinking=self.enable_thinking) + prefix

    def gen(self, messages: list[dict], prefix: str = "", temperature: float = 0.0, top_p: float = 1.0,
            seed: int = 0, stop: tuple[str, ...] = (), max_tokens: int | None = None, stream: bool = False,
            lookup: int = 0) -> dict:
        text = self.prompt(messages, prefix)
        toks = self.llm.tokenize(text, add_bos=False)
        stops = list(stop) + list(self.llm.formatter.stop_strings())
        t0 = time.perf_counter()
        g = self.llm.generate_tokens(toks, max_tokens=max_tokens or self.max_tokens, temperature=temperature,
                                     top_p=top_p, seed=seed, stop=stops,
                                     stream=self.stream if stream else None, lookup=lookup)
        self.tokens += len(g.tokens)
        return {"text": g.text, "tokens": len(g.tokens), "stop_reason": g.stop_reason, "prompt": text,
                "seconds": round(time.perf_counter() - t0, 3), "temperature": temperature, "seed": seed}

    def gen_many(self, messages: list[dict], n: int, prefix: str = "", temperature: float = 0.7,
                 top_p: float = 1.0, seed: int = 0, stop: tuple[str, ...] = ()) -> list[dict]:
        """n sampled generations of one prompt in a single batched pass (requires self.batch)."""
        text = self.prompt(messages, prefix)
        toks = self.llm.tokenize(text, add_bos=False)
        stops = list(stop) + list(self.llm.formatter.stop_strings())
        t0 = time.perf_counter()
        gens = self.llm.sample_n(toks, n, max_tokens=self.max_tokens, temperature=temperature, top_p=top_p,
                                 seed=seed, stop=stops)
        dt = round((time.perf_counter() - t0) / max(1, len(gens)), 3)
        out = []
        for i, g in enumerate(gens):
            self.tokens += len(g.tokens)
            out.append({"text": g.text, "tokens": len(g.tokens), "stop_reason": g.stop_reason, "prompt": text,
                        "seconds": dt, "temperature": temperature, "seed": seed + i, "batched": len(gens)})
        return out

    def note(self, msg: str) -> None:
        if self.stream:
            self.stream(msg)

    def logprob(self, prompt: str, continuation: str) -> float | None:
        """Total log-probability of continuation after prompt (None if the LLM cannot score)."""
        try:
            return float(self.llm.score(prompt, continuation, add_bos=False))
        except Exception:  # noqa: BLE001 - scoring is an optional refinement
            return None


def _seed(base: int, i: int) -> int:
    return (base * 1_000_003 + i * 7_919 + 0x5EED) & 0xFFFFFFFF


def _messages(content: str, system: str | None = None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]


# =========================================================================================
# strategies

def _direct(s: _Session, q: str, numeric: bool | None) -> Answer:
    g = s.gen(_messages(q), stream=True)
    ans = extract_answer(g["text"], numeric)
    g.update(kind="direct", answer=ans, key=answer_key(ans))
    return Answer(text=_strip_thinking(g["text"]).strip(), answer=ans, strategy="direct", candidates=[_slim(g)])


def _cot_finish(s: _Session, g: dict, numeric: bool | None) -> dict:
    ans = extract_answer(g["text"], numeric)
    g.update(kind="cot", answer=ans, key=answer_key(ans), weight=s.weight(ans))
    return g


def _cot_sample(s: _Session, q: str, numeric: bool | None, temperature: float, top_p: float, seed: int,
                stream: bool = False) -> dict:
    g = s.gen(_with_examples(COT_PROMPT, q, COT_EXAMPLES), temperature=temperature, top_p=top_p, seed=seed,
              stream=stream)
    return _cot_finish(s, g, numeric)


def _cot_batch(s: _Session, q: str, numeric: bool | None, k: int, temperature: float, top_p: float,
               seed: int) -> list[dict]:
    gens = s.gen_many(_with_examples(COT_PROMPT, q, COT_EXAMPLES), k, temperature=temperature, top_p=top_p,
                      seed=seed)
    return [_cot_finish(s, g, numeric) for g in gens]


def _cot(s: _Session, q: str, numeric: bool | None) -> Answer:
    g = _cot_sample(s, q, numeric, 0.0, 1.0, 0, stream=True)
    return Answer(text=_strip_thinking(g["text"]).strip(), answer=g["answer"], strategy="cot", candidates=[_slim(g)])


def _sample_loop(s: _Session, n: int, make: Callable[[int], dict], early_stop: bool, label: str,
                 make_batch: Callable[[int, int], list[dict]] | None = None) -> list[dict]:
    """Candidate 0 is greedy; the rest are sampled, in one batched pass when the LLM can
    (all at once: lockstep decoding makes extra samples cheap), else one by one with
    early stopping once the vote is decided."""
    cands: list[dict] = [make(0)]
    s.note(f"[{label} 1/{n}] answer: {cands[0].get('answer')}\n")
    if n > 1 and s.batch and make_batch is not None:
        for j, c in enumerate(make_batch(n - 1, _seed(cands[0].get("seed", 0), 1))):
            cands.append(c)
            s.note(f"[{label} {j + 2}/{n}] answer: {c.get('answer')}\n")
        return cands
    for i in range(1, n):
        if early_stop and _decided(cands, n):
            break
        c = make(i)
        cands.append(c)
        s.note(f"[{label} {i + 1}/{n}] answer: {c.get('answer')}\n")
    return cands


def _tiebreak_logprobs(s: _Session, cands: list[dict]) -> None:
    """Fill mean token log-probs for candidates in tied top groups (only those are needed)."""
    _, info = majority_vote(cands)
    if not info["tie"]:
        return
    top = max(info["counts"].values())
    tied = {k for k, v in info["counts"].items() if v == top}
    if cands and cands[0].get("temperature") == 0.0 and cands[0].get("key") in tied:
        return  # the greedy chain has the highest mean log-prob anyway; earliest-wins picks it for free
    for c in cands:
        if c.get("key") in tied and c.get("logprob") is None and c.get("prompt"):
            lp = s.logprob(c["prompt"], c["text"])
            if lp is not None:
                c["logprob"] = lp / max(1, c.get("tokens") or 1)


def _vote(s: _Session, q: str, numeric: bool | None, n: int, temperature: float, top_p: float, seed: int,
          early_stop: bool) -> Answer:
    # sample 0 is greedy: the single most likely chain is a strong first vote and breaks ties
    cands = _sample_loop(s, n, lambda i: _cot_sample(s, q, numeric, 0.0 if i == 0 else temperature, top_p,
                                                     _seed(seed, i)), early_stop, "sample",
                         lambda k, sd: _cot_batch(s, q, numeric, k, temperature, top_p, sd))
    _tiebreak_logprobs(s, cands)
    key, info = majority_vote(cands)
    if key is None:
        best = cands[0]
    else:
        best = cands[info["winner_index"]]
    n_agree = info["counts"].get(key, 0) if key else 0
    text = _strip_thinking(best["text"]).strip()
    if len(cands) > 1:
        text += f"\n\n({n_agree} of {len(cands)} sampled solutions agree.)"
    return Answer(text=text, answer=best.get("answer"), strategy="vote", candidates=[_slim(c) for c in cands])


def _program_value(res: tools.ToolResult) -> str | None:
    """Numeric answer of an executed program: `answer`/solution() value, else the last printed number."""
    if res.value is not None:
        n = normalize_number(res.value)
        if n is not None:
            return n
    if res.stdout.strip():
        nums = _NUM_RE.findall(res.stdout)
        if nums:
            return normalize_number(nums[-1])
    return None


def _run_program(s: _Session, g: dict, msgs: list[dict], repair: bool, timeout: float, stream: bool = False) -> dict:
    """Execute the program in generation g; on failure optionally ask once for a fix."""
    code = tools.extract_code(g["text"])
    res = tools.run_python(code, timeout=timeout)
    ans = _program_value(res) if res.ok else None
    attempts = [{"code": code, "ok": res.ok, "error": res.error, "stdout": res.stdout[:500], "value": res.value}]
    if repair and ans is None:
        err = res.error or "the program did not set `answer`"
        if res.ok and res.value is None and not res.stdout.strip():
            err = "the program ran but did not set `answer`"
        fix = msgs + [{"role": "assistant", "content": CODE_FENCE + code + "\n```"},
                      {"role": "user", "content": PAL_REPAIR.format(error=err)}]
        g2 = s.gen(fix, prefix=CODE_FENCE, temperature=g["temperature"], top_p=1.0, seed=g["seed"] + 1,
                   stop=("```",), stream=stream)
        code = tools.extract_code(g2["text"])
        res = tools.run_python(code, timeout=timeout)
        ans = _program_value(res) if res.ok else None
        attempts.append({"code": code, "ok": res.ok, "error": res.error, "stdout": res.stdout[:500],
                         "value": res.value})
        g["tokens"] += g2["tokens"]
        g["seconds"] = round(g["seconds"] + g2["seconds"], 3)
    g.update(kind="program", code=code, ok=res.ok, error=res.error, stdout=res.stdout[:500], value=res.value,
             answer=ans, key=answer_key(ans), weight=s.weight(ans), attempts=attempts,
             text=CODE_FENCE + code + "\n```")
    return g


def _pal_sample(s: _Session, q: str, temperature: float, top_p: float, seed: int, repair: bool,
                timeout: float, stream: bool = False) -> dict:
    msgs = _with_examples(PAL_PROMPT, q, PAL_EXAMPLES, fence=True)
    g = s.gen(msgs, prefix=CODE_FENCE, temperature=temperature, top_p=top_p, seed=seed, stop=("```",),
              stream=stream)
    return _run_program(s, g, msgs, repair, timeout, stream)


def _pal_batch(s: _Session, q: str, k: int, temperature: float, top_p: float, seed: int,
               timeout: float) -> list[dict]:
    msgs = _with_examples(PAL_PROMPT, q, PAL_EXAMPLES, fence=True)
    gens = s.gen_many(msgs, k, prefix=CODE_FENCE, temperature=temperature, top_p=top_p, seed=seed, stop=("```",))
    return [_run_program(s, g, msgs, False, timeout) for g in gens]


def _pal_text(c: dict) -> str:
    out = c["text"]
    shown = c.get("stdout", "").strip()
    if shown:
        out += f"\n\nOutput:\n{shown[:500]}"
    if c.get("answer") is not None:
        out += f"\n\nAnswer: {c['answer']}"
    elif c.get("error"):
        out += f"\n\nThe program failed: {c['error']}"
    return out


def _pal(s: _Session, q: str, numeric: bool | None, timeout: float) -> Answer:
    c = _pal_sample(s, q, 0.0, 1.0, 0, repair=True, timeout=timeout, stream=True)
    cands = [c]
    if c["answer"] is None:  # no runnable program: fall back to reasoning in text
        s.note("[pal] program failed; falling back to step-by-step reasoning\n")
        g = _cot_sample(s, q, numeric, 0.0, 1.0, 0)
        cands.append(g)
        return Answer(text=_strip_thinking(g["text"]).strip(), answer=g["answer"], strategy="pal",
                      candidates=[_slim(x) for x in cands])
    return Answer(text=_pal_text(c), answer=c["answer"], strategy="pal", candidates=[_slim(c)])


def _pal_vote(s: _Session, q: str, numeric: bool | None, n: int, temperature: float, top_p: float, seed: int,
              early_stop: bool, timeout: float) -> Answer:
    cands = _sample_loop(s, n, lambda i: _pal_sample(s, q, 0.0 if i == 0 else temperature, top_p, _seed(seed, i),
                                                     repair=(i == 0), timeout=timeout), early_stop, "program",
                         lambda k, sd: _pal_batch(s, q, k, temperature, top_p, sd, timeout))
    key, info = majority_vote(cands)
    if key is None or info["tie"]:
        # no program worked, or a tie: a step-by-step solution decides (or answers outright)
        g = _cot_sample(s, q, numeric, 0.0, 1.0, 0)
        g["tiebreaker"] = True
        g["weight"] = 0.0  # it only breaks the tie; it does not vote
        if key is not None and info["tie"]:
            top = max(info["scores"].values())
            tied = [k for k, v in info["scores"].items() if v >= top - 1e-9]
            if g["key"] in tied:
                key = g["key"]
                info["winner_index"] = next(i for i, c in enumerate(cands) if c.get("key") == key)
        cands.append(g)
        if key is None:
            return Answer(text=_strip_thinking(g["text"]).strip(), answer=g["answer"], strategy="pal_vote",
                          candidates=[_slim(c) for c in cands])
    best = cands[info["winner_index"]]
    n_prog = sum(1 for c in cands if c.get("kind") == "program")
    text = _pal_text(best) + f"\n\n({info['counts'][key]} of {n_prog} sampled programs agree.)"
    return Answer(text=text, answer=best["answer"], strategy="pal_vote", candidates=[_slim(c) for c in cands])


def _mix(s: _Session, q: str, numeric: bool | None, n: int, temperature: float, top_p: float, seed: int,
         early_stop: bool, timeout: float) -> Answer:
    """Programs and written step-by-step solutions vote together.

    The two formats fail differently (programs slip on reading the problem, prose on
    arithmetic), so agreement between them is strong evidence: when the greedy program
    and the greedy solution agree, sampling is skipped.
    """
    p0 = _pal_sample(s, q, 0.0, 1.0, _seed(seed, 0), repair=True, timeout=timeout)
    s.note(f"[program 1/{n}] answer: {p0.get('answer')}\n")
    c0 = _cot_sample(s, q, numeric, 0.0, 1.0, _seed(seed, 0))
    s.note(f"[solution 1/{n}] answer: {c0.get('answer')}\n")
    cands = [p0, c0]
    agree = p0.get("key") is not None and p0.get("key") == c0.get("key")
    if n > 1 and not (early_stop and agree):
        if s.batch:
            cands += _pal_batch(s, q, n - 1, temperature, top_p, _seed(seed, 1), timeout)
            cands += _cot_batch(s, q, numeric, n - 1, temperature, top_p, _seed(seed, 2))
        else:
            for i in range(1, n):
                for make in (lambda: _pal_sample(s, q, temperature, top_p, _seed(seed, i), False, timeout),
                             lambda: _cot_sample(s, q, numeric, temperature, top_p, _seed(seed, i))):
                    if early_stop and _decided(cands, 2 * n):
                        break
                    cands.append(make())
        for c in cands[2:]:
            s.note(f"[{'program' if c['kind'] == 'program' else 'solution'}] answer: {c.get('answer')}\n")
    key, info = majority_vote(cands)
    if key is None:
        return Answer(text=_strip_thinking(c0["text"]).strip(), answer=c0.get("answer"), strategy="mix",
                      candidates=[_slim(c) for c in cands])
    progs = [c for c in cands if c["kind"] == "program"]
    cots = [c for c in cands if c["kind"] == "cot"]
    # show a written solution for the winning answer when there is one (easier to read than code)
    best = next((c for c in cots if c.get("key") == key), None)
    best = best or cands[info["winner_index"]]
    body = _strip_thinking(best["text"]).strip() if best["kind"] == "cot" else _pal_text(best)
    agree_p = sum(c.get("key") == key for c in progs)
    agree_c = sum(c.get("key") == key for c in cots)
    text = body + (f"\n\n({agree_p} of {len(progs)} programs and {agree_c} of {len(cots)} written solutions "
                   f"agree.)")
    return Answer(text=text, answer=best.get("answer"), strategy="mix", candidates=[_slim(c) for c in cands])


def _verify(s: _Session, q: str, numeric: bool | None, n: int, temperature: float, top_p: float, seed: int,
            early_stop: bool) -> Answer:
    cands = _sample_loop(s, n, lambda i: _cot_sample(s, q, numeric, 0.0 if i == 0 else temperature, top_p,
                                                     _seed(seed, i)), early_stop, "sample",
                         lambda k, sd: _cot_batch(s, q, numeric, k, temperature, top_p, sd))
    key, info = majority_vote(cands)
    counts = info["counts"]
    if key is None or len(counts) == 1:
        best = cands[info["winner_index"]] if key is not None else cands[0]
        return Answer(text=_strip_thinking(best["text"]).strip(), answer=best.get("answer"), strategy="verify",
                      candidates=[_slim(c) for c in cands])
    # check one representative solution per distinct answer (most voted first, at most 4)
    scores: dict[str, float] = {}
    total = sum(counts.values())
    for k in sorted(counts, key=lambda k: -counts[k])[:4]:
        i = next(i for i, c in enumerate(cands) if c.get("key") == k)
        sol = _strip_thinking(cands[i]["text"]).strip()[-1500:]
        prompt = s.prompt(_messages(VERIFY_PROMPT.format(question=q, solution=sol)))
        yes, no = s.logprob(prompt, "Yes"), s.logprob(prompt, "No")
        if yes is None or no is None:
            p_yes = 0.5
        else:
            p_yes = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, no - yes))))
        cands[i]["p_yes"] = round(p_yes, 4)
        scores[k] = counts[k] / total + p_yes  # votes and the verifier weigh equally
        s.note(f"[verify] answer {cands[i].get('answer')}: votes {counts[k]}/{total}, P(correct) {p_yes:.2f}\n")
    best_key = max(scores, key=lambda k: (scores[k], counts[k]))
    best = next(c for c in cands if c.get("key") == best_key)
    text = _strip_thinking(best["text"]).strip() + \
        f"\n\n(Chosen from {len(counts)} distinct answers by vote and self-verification.)"
    return Answer(text=text, answer=best.get("answer"), strategy="verify", candidates=[_slim(c) for c in cands])


def _rag(s: _Session, q: str, context: str, numeric: bool | None) -> Answer:
    ctx = context.strip()
    if not re.search(r"^\[\d+\]", ctx, flags=re.M):
        ctx = "[1]\n" + ctx
    msgs = _messages(f"Context:\n{ctx}\n\nQuestion: {q}", system=RAG_SYSTEM)
    # keep room for the answer: trim the context's tail (the lowest-ranked passages) to fit
    n_ctx = getattr(s.llm, "n_ctx", None)
    for _ in range(4):
        if not n_ctx:
            break
        over = len(s.llm.tokenize(s.prompt(msgs), add_bos=False)) + min(s.max_tokens, 256) + 8 - n_ctx
        if over <= 0:
            break
        ctx = ctx[:max(0, int(len(ctx) * (1 - over / max(1, len(s.llm.tokenize(ctx, add_bos=False)))) - 64))]
        msgs = _messages(f"Context:\n{ctx}\n\nQuestion: {q}", system=RAG_SYSTEM)
    # answers copy spans of the context, which prompt-lookup decoding drafts for free
    g = s.gen(msgs, stream=True, lookup=3)
    text = _strip_thinking(g["text"]).strip()
    plain = re.sub(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]", "", text)  # citation markers are not answers
    ans = extract_answer(plain, numeric)
    if ans is None and len(plain) <= 120:
        ans = _clean_short(plain)
    g.update(kind="rag", answer=ans, key=answer_key(ans), citations=sorted(set(re.findall(r"\[(\d+)\]", text))))
    return Answer(text=text, answer=ans, strategy="rag", candidates=[_slim(g)])


def _calc(q: str) -> Answer | None:
    expr = _as_expression(q)
    if expr is None:
        return None
    try:
        v = tools.calculator(expr)
    except tools.CalculatorError:
        return None
    ans = normalize_number(v) or str(v)
    return Answer(text=f"{expr} = {ans}", answer=ans, strategy="calc",
                  candidates=[{"kind": "calculator", "expression": expr, "value": v, "answer": ans}])


def _slim(c: dict) -> dict:
    """Candidate record without the bulky rendered prompt."""
    return {k: v for k, v in c.items() if k != "prompt"}


# =========================================================================================

def solve(llm, question: str, strategy: str = "auto", samples: int = 5, max_tokens: int = 512, seed: int = 0,
          stream: Callable[[str], None] | None = None, context: str | None = None, *,
          temperature: float = 0.7, top_p: float = 0.95, early_stop: bool = True, enable_thinking: bool = False,
          numeric: bool | None = None, timeout: float = 10.0, batch: bool = True, prior: bool = True) -> Answer:
    """Answer `question` with test-time reasoning.

    llm: a superborg.LLM (anything with render/tokenize/generate_tokens/score/formatter).
    strategy: a key of STRATEGIES. samples: candidates for the sampling strategies.
    max_tokens: limit for each generation. seed: base seed (sampling is deterministic).
    stream: called with text as it is produced; single-pass strategies stream the model's
    tokens, sampling strategies one progress line per candidate.
    context: reference text (e.g. retrieval.format_context(hits)); "auto" then answers with "rag".
    numeric: force numeric (True) or text (False) answer extraction; None guesses.
    batch: sample candidates together with llm.sample_n when available (otherwise one at a
    time, stopping early once the majority is decided).
    prior: weigh numeric votes by plausibility() (negative / long fractional answers count less).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
    t0 = time.perf_counter()
    samples = max(1, int(samples))
    s = _Session(llm, max_tokens=max(1, int(max_tokens)), enable_thinking=enable_thinking, stream=stream,
                 batch=batch)
    s.question, s.prior = question, prior
    route = ""
    kind = None
    if strategy == "auto":
        kind, why = classify(question, context)
        strategy = {"rag": "rag", "arithmetic": "calc", "math": "pal_vote" if samples > 1 else "pal",
                    "reasoning": "cot", "simple": "direct"}[kind]
        route = f"{kind}: {why}"
    if numeric is None and kind in ("math", "arithmetic"):
        numeric = True

    ans: Answer | None = None
    if strategy == "calc":
        ans = _calc(question)
        if ans is None:  # not an expression after all
            strategy, route = "pal", (route + "; calculator could not parse it").lstrip("; ")
    if ans is None:
        if strategy == "direct":
            ans = _direct(s, question, numeric)
        elif strategy == "cot":
            ans = _cot(s, question, numeric)
        elif strategy == "vote":
            ans = _vote(s, question, numeric, samples, temperature, top_p, seed, early_stop)
        elif strategy == "pal":
            ans = _pal(s, question, numeric, timeout)
        elif strategy == "pal_vote":
            ans = _pal_vote(s, question, numeric, samples, temperature, top_p, seed, early_stop, timeout)
        elif strategy == "mix":
            ans = _mix(s, question, numeric, samples, temperature, top_p, seed, early_stop, timeout)
        elif strategy == "verify":
            ans = _verify(s, question, numeric, samples, temperature, top_p, seed, early_stop)
        elif strategy == "rag":
            if not context:
                raise ValueError("strategy 'rag' needs context")
            ans = _rag(s, question, context, numeric)
        else:  # pragma: no cover - STRATEGIES and this dispatch are kept in sync
            raise ValueError(strategy)
    ans.route = route
    ans.tokens = s.tokens
    ans.elapsed_s = time.perf_counter() - t0
    return ans
