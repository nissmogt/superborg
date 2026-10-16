"""Chat prompt formatting.

Renders the model's own Jinja chat template (copied from the checkpoint into the .borg header
by superborg/convert.py) the way transformers' apply_chat_template does, when jinja2 is
installed. Otherwise falls back to built-in renderers for the common formats.
wasm/trace-session.js renders the same template in the browser with @huggingface/jinja;
tests/test_wasm.py checks that the two give the same bytes.
"""
from __future__ import annotations

import datetime
import functools
import json
from typing import Any

Message = dict[str, Any]


@functools.lru_cache(maxsize=1)
def _environment():
    """transformers' template environment (utils/chat_template_utils.py): a sandbox with
    trim_blocks and lstrip_blocks, loop controls, its tojson, raise_exception and strftime_now,
    and the {% generation %} tag (transformers marks assistant tokens with it; here the body
    just renders, as @huggingface/jinja does)."""
    import jinja2
    from jinja2 import nodes
    from jinja2.ext import Extension, loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class GenerationTag(Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

        def _render(self, caller):
            return caller()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationTag, loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


@functools.lru_cache(maxsize=16)
def _compiled(template: str):
    return _environment().from_string(template)


def _jinja_render(template: str, messages: list[Message], add_generation_prompt: bool, **kwargs) -> str:
    return _compiled(template).render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)


def _builtin_render(fmt: str, messages: list[Message], add_generation_prompt: bool, enable_thinking: bool = True,
                    default_system: str | None = None) -> str:
    out = []
    msgs = list(messages)
    if default_system and (not msgs or msgs[0]["role"] != "system"):
        msgs.insert(0, {"role": "system", "content": default_system})
    if fmt in ("chatml", "qwen3"):
        for m in msgs:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
            if fmt == "qwen3" and not enable_thinking:
                out.append("<think>\n\n</think>\n\n")
    elif fmt == "llama3":
        out.append("<|begin_of_text|>")
        for m in msgs:
            out.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>")
        if add_generation_prompt:
            out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif fmt == "zephyr":
        for m in msgs:
            out.append(f"<|{m['role']}|>\n{m['content']}</s>\n")
        if add_generation_prompt:
            out.append("<|assistant|>\n")
    elif fmt == "mistral":
        sys_txt = ""
        for m in msgs:
            if m["role"] == "system":
                sys_txt = m["content"] + "\n\n"
            elif m["role"] == "user":
                out.append(f"[INST] {sys_txt}{m['content']} [/INST]")
                sys_txt = ""
            else:
                out.append(f"{m['content']}</s>")
    else:
        for m in msgs:
            out.append(f"{m['role']}: {m['content']}\n")
        if add_generation_prompt:
            out.append("assistant:")
    return "".join(out)


class ChatFormatter:
    """Turns a list of {"role", "content"} messages into prompt text for one model."""

    def __init__(self, tokenizer_info: dict, token_text=None):
        self.template = tokenizer_info.get("chat_template")
        self.fmt = tokenizer_info.get("chat_format", "plain")
        self.token_text = token_text or (lambda i: "")
        bos, eos = tokenizer_info.get("bos_id", -1), tokenizer_info.get("eos_id", -1)
        self.bos_token = self._text(bos)
        self.eos_token = self._text(eos)
        # the end-of-turn tokens' text (the header's eos_ids), in order, without repeats
        self.eos_tokens: list[str] = []
        for i in tokenizer_info.get("eos_ids") or ([eos] if eos is not None and eos >= 0 else []):
            t = self._text(i)
            if t and t not in self.eos_tokens:
                self.eos_tokens.append(t)
        self.add_bos = bool(tokenizer_info.get("add_bos"))
        try:
            import jinja2  # noqa: F401
            self.have_jinja = True
        except ImportError:
            self.have_jinja = False

    def _text(self, token_id) -> str:
        return self.token_text(token_id) if token_id is not None and token_id >= 0 else ""

    @property
    def supports_thinking(self) -> bool:
        return bool(self.template and "enable_thinking" in self.template) or self.fmt == "qwen3"

    def render(self, messages: list[Message], add_generation_prompt: bool = True, enable_thinking: bool | None = None,
               tools: list | None = None, **chat_template_kwargs) -> str:
        """The prompt for `messages`. enable_thinking None leaves the variable out, so each
        template's own default applies (thinking on for Qwen3), as transformers does when the
        caller passes none; chat_template_kwargs (date_string, documents, ...) reach the template
        as variables, as apply_chat_template's extra keyword arguments do."""
        if self.template and self.have_jinja:
            kwargs: dict[str, Any] = {"tools": tools, "documents": None}
            # a special token the header lacks stays undefined, as transformers' special_tokens_map
            # leaves it (None would render as "None")
            if self.bos_token:
                kwargs["bos_token"] = self.bos_token
            if self.eos_token:
                kwargs["eos_token"] = self.eos_token
            kwargs.update(chat_template_kwargs)
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            return _jinja_render(self.template, messages, add_generation_prompt, **kwargs)
        default_system = None
        if self.template and "You are a helpful AI assistant named SmolLM" in self.template:
            default_system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
        return _builtin_render(self.fmt, messages, add_generation_prompt, bool(enable_thinking), default_system)

    def stop_strings(self) -> list[str]:
        """The end-of-turn tokens' text: what the template ends turns with, whatever its format."""
        return list(self.eos_tokens)


def formatter_for(model) -> ChatFormatter:
    def token_text(i: int) -> str:
        try:
            return model.token_bytes(i).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
    return ChatFormatter(model.tokenizer_info, token_text)
