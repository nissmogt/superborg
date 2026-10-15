"""superborg: fast local inference for open-weight language models, plus a reasoning layer.

    import superborg
    llm = superborg.LLM("qwen2.5-1.5b-q8_0.borg")
    print(llm.chat([{"role": "user", "content": "Hello!"}]).text)
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .chat import ChatFormatter, formatter_for
from .engine import Context, Generation, Model, Pool, Sampling, simd

__all__ = ["LLM", "Model", "Context", "Sampling", "Generation", "Pool", "ChatFormatter", "simd"]
__version__ = "0.1.0"


class LLM:
    """A model, one KV-cache context and the model's chat format.

    Consecutive calls reuse the cached prefix they share (for example earlier chat
    turns), so only new tokens are processed. Not thread-safe: use one LLM per thread
    or serialize calls.
    """

    def __init__(self, path: str, threads: int = 0, n_ctx: int = 4096, kv_f16: bool = False,
                 draft: str | None = None, n_draft: int = 6, pool: Pool | None = None):
        self.pool = pool or Pool(threads)
        self.model = Model(path, pool=self.pool)
        n_ctx = min(n_ctx, self.model.max_seq_len) if n_ctx > 0 else 0
        self.ctx = self.model.context(n_ctx=n_ctx, kv_f16=kv_f16)
        self.formatter = formatter_for(self.model)
        self.draft_model = Model(draft, pool=self.pool) if draft else None
        self.draft_ctx = self.draft_model.context(n_ctx=self.ctx.n_ctx, kv_f16=kv_f16) if draft else None
        self.n_draft = n_draft

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def n_ctx(self) -> int:
        return self.ctx.n_ctx

    def tokenize(self, text: str, add_bos: bool | None = None, special: bool = True) -> list[int]:
        return self.model.tokenize(text, add_bos=add_bos, special=special)

    def detokenize(self, tokens: Sequence[int], skip_special: bool = False) -> str:
        return self.model.detokenize(tokens, skip_special=skip_special)

    def render(self, messages: list[dict], enable_thinking: bool = False, add_generation_prompt: bool = True) -> str:
        return self.formatter.render(messages, add_generation_prompt=add_generation_prompt,
                                     enable_thinking=enable_thinking)

    def generate_tokens(self, prompt_tokens: Sequence[int], max_tokens: int = 256, temperature: float = 0.0,
                        top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, repeat_penalty: float = 1.0,
                        seed: int = 0x5EED, stop: Iterable[str] = (), stream: Callable[[str], None] | None = None,
                        lookup: int = 0, speculative: bool = True) -> Generation:
        sampling = Sampling(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                            repeat_penalty=repeat_penalty, seed=seed)
        prompt_tokens = list(prompt_tokens)
        room = self.ctx.n_ctx - len(prompt_tokens) - 1
        if room <= 0:
            raise ValueError(f"prompt ({len(prompt_tokens)} tokens) does not fit the {self.ctx.n_ctx}-token context")
        max_tokens = max(0, min(max_tokens, room))
        draft = self.draft_ctx if (speculative and self.draft_ctx is not None) else None
        return self.ctx.generate(prompt_tokens, max_tokens=max_tokens, sampling=sampling, stop=stop, stream=stream,
                                 draft=draft, n_draft=self.n_draft, lookup_ngram=lookup if speculative else 0)

    def complete(self, prompt: str, max_tokens: int = 256, add_bos: bool | None = None, **kw) -> Generation:
        """Continue raw text."""
        return self.generate_tokens(self.tokenize(prompt, add_bos=add_bos), max_tokens=max_tokens, **kw)

    def chat(self, messages: list[dict], max_tokens: int = 512, enable_thinking: bool = False, **kw) -> Generation:
        """Answer the last message of a conversation using the model's chat template."""
        text = self.render(messages, enable_thinking=enable_thinking)
        stop = list(kw.pop("stop", ())) + self.formatter.stop_strings()
        # chat templates spell out BOS themselves; transformers tokenizes them without adding one
        return self.generate_tokens(self.tokenize(text, add_bos=False), max_tokens=max_tokens, stop=stop, **kw)

    def ask(self, question: str, system: str | None = None, **kw) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
        return self.chat(msgs, **kw).text

    def score(self, prompt: str, continuation: str, add_bos: bool | None = None) -> float:
        """Total log-probability of `continuation` following `prompt`."""
        p = self.tokenize(prompt, add_bos=add_bos)
        full = self.tokenize(prompt + continuation, add_bos=add_bos)
        # tokenization of the joint string decides the continuation tokens
        k = 0
        while k < min(len(p), len(full)) and p[k] == full[k]:
            k += 1
        if k == 0:
            raise ValueError("prompt must produce at least one token")
        return float(self.ctx.score(full[:k], full[k:]).sum())

    def close(self):
        for obj in (self.draft_ctx, self.draft_model, self.ctx, self.model):
            if obj is not None:
                obj.close()
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
