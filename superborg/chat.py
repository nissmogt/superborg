"""Chat prompt formatting.

Renders the model's own Jinja chat template (copied from tokenizer_config.json into the
.borg header) the way transformers' apply_chat_template does, when jinja2 is installed.
Otherwise falls back to built-in renderers for the common formats.
"""
from __future__ import annotations

import datetime
import json
from typing import Any

Message = dict[str, Any]


def _jinja_render(template: str, messages: list[Message], add_generation_prompt: bool, **kwargs) -> str:
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env.from_string(template).render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)


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
        self.bos_token = self.token_text(bos) if bos is not None and bos >= 0 else ""
        self.eos_token = self.token_text(eos) if eos is not None and eos >= 0 else ""
        self.add_bos = bool(tokenizer_info.get("add_bos"))
        try:
            import jinja2  # noqa: F401
            self.have_jinja = True
        except ImportError:
            self.have_jinja = False

    @property
    def supports_thinking(self) -> bool:
        return bool(self.template and "enable_thinking" in self.template) or self.fmt == "qwen3"

    def render(self, messages: list[Message], add_generation_prompt: bool = True, enable_thinking: bool = False,
               tools: list | None = None) -> str:
        if self.template and self.have_jinja:
            kwargs = dict(bos_token=self.bos_token, eos_token=self.eos_token, enable_thinking=enable_thinking)
            if tools:
                kwargs["tools"] = tools
            return _jinja_render(self.template, messages, add_generation_prompt, **kwargs)
        default_system = None
        if self.template and "You are a helpful AI assistant named SmolLM" in self.template:
            default_system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
        return _builtin_render(self.fmt, messages, add_generation_prompt, enable_thinking, default_system)

    def stop_strings(self) -> list[str]:
        return {"chatml": ["<|im_end|>"], "qwen3": ["<|im_end|>"], "llama3": ["<|eot_id|>"],
                "zephyr": ["</s>"], "mistral": ["</s>"]}.get(self.fmt, [])


def formatter_for(model) -> ChatFormatter:
    def token_text(i: int) -> str:
        try:
            return model.token_bytes(i).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
    return ChatFormatter(model.tokenizer_info, token_text)
