"""Hugging Face transformers side of the reference tests and fixtures.

Shared by tests/test_reference.py and tools/make_fixtures.py so both use the same
models, prompt and HF settings. Needs torch and transformers (not part of the
superborg package, which depends on NumPy only).
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import numpy as np

_SCRATCH = "/tmp/claude-0/-home-user-superborg/990fbcaa-dc72-53ae-9121-cd54b34a4307/scratchpad"
HF_MODELS_DIR = os.environ.get("HF_MODELS_DIR", os.path.join(_SCRATCH, "models"))
BORG_MODELS_DIR = os.environ.get("BORG_MODELS_DIR", os.path.join(_SCRATCH, "borg"))

# ~24 tokens: mixes words, digits and punctuation, and ends where the continuation is predictable.
PROMPT_TEXT = "In 1969, Neil Armstrong became the first person to walk on the Moon. He said"
N_GEN = 32
# Long-context check: the first LONG_PROMPT_TOKENS tokens of the converter's calibration text
# (prose, code, numbers), so RoPE, the causal mask and the KV cache run well past the short prompt.
LONG_PROMPT_TOKENS = 200
LONG_N_GEN = 8
LONG_PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "superborg", "data", "calibration.txt")


@dataclass(frozen=True)
class ModelSpec:
    key: str     # .borg files are <key>-<qtype>.borg
    hf_dir: str  # directory name under HF_MODELS_DIR

    @property
    def hf_path(self) -> str:
        return os.path.join(HF_MODELS_DIR, self.hf_dir)

    def borg_path(self, qtype: str) -> str:
        return os.path.join(BORG_MODELS_DIR, f"{self.key}-{qtype}.borg")


MODELS = [
    ModelSpec("smollm2-135m", "HuggingFaceTB_SmolLM2-135M-Instruct"),
    ModelSpec("qwen2.5-0.5b", "Qwen_Qwen2.5-0.5B-Instruct"),
    ModelSpec("qwen3-0.6b", "Qwen_Qwen3-0.6B"),
    ModelSpec("tinyllama-1.1b", "TinyLlama_TinyLlama-1.1B-Chat-v1.0"),
]


def hf_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def load_hf_model(model_dir: str):
    """float32 weights and the eager attention path: the plainest HF computation."""
    import torch
    from transformers import AutoModelForCausalLM
    # Few threads by default: on a shared machine torch's spin-waiting threads get
    # descheduled and a 135M-parameter generate slows down 10x.
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", min(2, os.cpu_count() or 1))))
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    except TypeError:  # transformers < 4.56 only knows torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32,
                                                     attn_implementation="eager")
    model.eval()
    return model


def hf_prompt_tokens(model_dir: str, text: str = PROMPT_TEXT) -> list[int]:
    """Prompt ids exactly as the HF tokenizer produces them (BOS included when the model adds one)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    return [int(t) for t in tok(text)["input_ids"]]


def long_prompt_tokens(model_dir: str, n: int = LONG_PROMPT_TOKENS) -> list[int]:
    """The first n prompt ids of LONG_PROMPT_FILE (BOS included when the model adds one)."""
    with open(LONG_PROMPT_FILE, encoding="utf-8") as f:
        toks = hf_prompt_tokens(model_dir, f.read())
    if len(toks) < n:
        raise ValueError(f"{LONG_PROMPT_FILE} gives only {len(toks)} tokens, need {n}")
    return toks[:n]


_CHILD = r"""
import io, json, os, sys
out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)  # anything the libraries print goes to stderr; fd 1's copy carries the result
sys.path.insert(0, %r)
import numpy as np
import hf_oracle as ho
a = json.loads(sys.argv[1])
buf = io.BytesIO()
np.savez(buf, **ho.hf_run(a["model_dir"], a["tokens"], a["n_gen"], isolate=False))
out.write(buf.getvalue())
out.close()
"""


def _run_in_child(model_dir: str, tokens, n_gen: int) -> dict:
    """hf_run in a separate interpreter. When it exits every byte of the torch model goes
    back to the OS; freeing the model in-process leaves allocator arenas that later NumPy
    work stacks on top of (the 1.1B model is 4.4 GB in float32)."""
    import io
    import json
    import subprocess
    import sys
    args = json.dumps({"model_dir": model_dir, "tokens": tokens, "n_gen": n_gen})
    proc = subprocess.run([sys.executable, "-c", _CHILD % os.path.dirname(os.path.abspath(__file__)), args],
                          stdout=subprocess.PIPE, check=True)
    with np.load(io.BytesIO(proc.stdout)) as z:
        return {k: z[k] for k in z.files}


def hf_run(model_dir: str, tokens: list[int] | None = None, n_gen: int = N_GEN, isolate: bool = True) -> dict:
    """Run HF once: full-prompt logits plus a greedy continuation.

    Returns numpy arrays: tokens [T] int32, logits [T, vocab] float32,
    gen_tokens [n_gen] int32, gen_logits [n_gen, vocab] float32 (the raw logits
    that chose each generated token; gen_logits[0] == logits[-1]). EOS does not
    stop generation, so exactly n_gen tokens come back. With isolate (default) the
    model lives in a short-lived child process.

    The continuation is the greedy loop of model.generate() written out: prefill
    with a KV cache, then one token per forward. generate() itself is not used
    because transformers >= 5 fills every unset field of a passed GenerationConfig
    from the checkpoint's generation_config.json, which brings back EOS stopping
    and, for Qwen2.5, repetition_penalty=1.1.
    """
    if isolate:
        return _run_in_child(model_dir, tokens, n_gen)

    import torch

    if tokens is None:
        tokens = hf_prompt_tokens(model_dir)
    model = load_hf_model(model_dir)
    try:
        with torch.no_grad():
            out = model(torch.tensor([tokens], dtype=torch.long), use_cache=True)
            logits = out.logits[0].float().numpy().copy()
            past, last = out.past_key_values, out.logits[0, -1]
            gen, gen_logits = [], []
            for _ in range(n_gen):
                tok = int(torch.argmax(last))  # first index on ties, like np.argmax
                gen.append(tok)
                gen_logits.append(last.float().numpy().copy())
                if len(gen) == n_gen:
                    break
                out = model(torch.tensor([[tok]], dtype=torch.long), past_key_values=past, use_cache=True)
                past, last = out.past_key_values, out.logits[0, -1]
        return {"tokens": np.asarray(tokens, dtype=np.int32), "logits": logits.astype(np.float32),
                "gen_tokens": np.asarray(gen, dtype=np.int32),
                "gen_logits": np.stack(gen_logits).astype(np.float32)}
    finally:
        del model
        gc.collect()
