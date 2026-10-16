"""superborg/chat.py (ChatFormatter), the converter's reading of chat templates and the servers'
chat_template_kwargs, without a model: the variables a template sees are those transformers'
apply_chat_template passes."""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("jinja2")

from superborg import convert  # noqa: E402
from superborg.chat import ChatFormatter  # noqa: E402

HI = [{"role": "user", "content": "hi"}]
PROBE = ("{% if enable_thinking is defined %}T={{ enable_thinking }}{% else %}T=undef{% endif %}|"
         "B={% if bos_token is defined %}[{{ bos_token }}]{% else %}undef{% endif %}|"
         "E={% if eos_token is defined %}[{{ eos_token }}]{% else %}undef{% endif %}|"
         "tools={{ tools }}|docs={{ documents }}|date={{ date_string }}|"
         "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}gen={{ add_generation_prompt }}")


def formatter(template: str, bos: str = "", eos: str = "", **info) -> ChatFormatter:
    tokens = {0: bos, 1: eos}
    return ChatFormatter(dict({"chat_template": template, "bos_id": 0 if bos else -1, "eos_id": 1 if eos else -1}, **info),
                         token_text=lambda i: tokens.get(i, ""))


def test_enable_thinking_is_passed_only_as_a_boolean():
    """None leaves the variable undefined (the template's own default, as transformers when the
    caller passes nothing); a boolean reaches the template; a boolean wins over chat_template_kwargs."""
    f = formatter(PROBE)
    assert f.render(HI).startswith("T=undef|")
    assert f.render(HI, enable_thinking=False).startswith("T=False|")
    assert f.render(HI, enable_thinking=True).startswith("T=True|")
    assert f.render(HI, **{"enable_thinking": True}).startswith("T=True|")  # as a chat_template_kwargs entry


def test_special_tokens_absent_rather_than_none():
    """A token the header lacks is not passed (transformers' special_tokens_map drops None tokens):
    the template never prints "None"."""
    assert "|B=undef|E=undef|" in formatter(PROBE).render(HI)
    assert "|B=[<s>]|E=[</s>]|" in formatter(PROBE, bos="<s>", eos="</s>").render(HI)
    assert "|B=undef|E=[<|im_end|>]|" in formatter(PROBE, eos="<|im_end|>").render(HI)


def test_tools_documents_and_extra_kwargs():
    """tools and documents are None when absent (defined, like transformers passes them); extra
    keyword arguments are template variables."""
    out = formatter(PROBE).render(HI, date_string="16 Oct 2026")
    assert "|tools=None|docs=None|date=16 Oct 2026|user:hi;gen=True" in out
    out = formatter(PROBE).render(HI, add_generation_prompt=False, tools=[{"type": "function"}], documents=[1])
    assert "|tools=[{'type': 'function'}]|docs=[1]|date=|user:hi;gen=False" in out


def test_generation_tag_and_transformers_globals():
    """{% generation %} (transformers' assistant-token marker, in SmolLM3's template) renders its
    body; tojson, raise_exception, strftime_now and loop controls are there."""
    assert formatter("a{% generation %}b{{ messages[0].content }}{% endgeneration %}c").render(HI) == "abhic"
    assert formatter("{{ {'k': [1, 'é']} | tojson }}").render(HI) == '{"k": [1, "é"]}'
    assert formatter("{% for m in messages %}{% if m.role == 'user' %}{% break %}{% endif %}x{% endfor %}y").render(HI) == "y"
    with pytest.raises(Exception, match="no system role"):
        formatter("{{ raise_exception('no system role') }}").render(HI)
    assert len(formatter("{{ strftime_now('%Y-%m-%d') }}").render(HI)) == 10


def test_compiled_templates_are_cached():
    from superborg import chat
    chat._compiled.cache_clear()
    f = formatter(PROBE)
    f.render(HI)
    f.render(HI, enable_thinking=True)
    assert chat._compiled.cache_info().hits == 1 and chat._compiled.cache_info().misses == 1


def test_stop_strings_are_the_eos_tokens():
    """The text of eos_ids (the header's end-of-turn tokens), in order and without repeats; eos_id
    alone when there is no list; nothing for a bare format label."""
    texts = {1: "<|im_end|>", 2: "<|endoftext|>", 3: ""}
    f = ChatFormatter({"chat_format": "chatml", "eos_id": 1, "eos_ids": [1, 2, 3, 1]}, token_text=lambda i: texts[i])
    assert f.stop_strings() == ["<|im_end|>", "<|endoftext|>"]
    assert ChatFormatter({"chat_format": "chatml", "eos_id": 2}, token_text=lambda i: texts[i]).stop_strings() == ["<|endoftext|>"]
    assert ChatFormatter({"chat_format": "chatml"}).stop_strings() == []
    assert ChatFormatter({"chat_format": "llama3", "eos_id": -1, "eos_ids": []}).stop_strings() == []


def test_builtin_formats_without_a_template():
    f = ChatFormatter({"chat_format": "qwen3"})
    assert f.render(HI).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # thinking off unless asked
    assert f.render(HI, enable_thinking=True).endswith("<|im_start|>assistant\n")
    assert ChatFormatter({"chat_format": "llama3"}).render(HI, add_generation_prompt=False) == \
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
    assert ChatFormatter({}).render(HI) == "user: hi\nassistant:"


# ---- superborg/convert.py: where the template comes from -----------------------------------

def _repo(tmp_path, name, tc: dict | None = None, jinja: str | None = None, extra: dict | None = None):
    d = tmp_path / name
    d.mkdir()
    if tc is not None:
        (d / "tokenizer_config.json").write_text(json.dumps(tc), encoding="utf-8")
    if jinja is not None:
        (d / "chat_template.jinja").write_text(jinja, encoding="utf-8")
    for rel, text in (extra or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8")
    return str(d)


def test_load_chat_template_precedence(tmp_path):
    """chat_template.jinja (what transformers >= 5 saves) over additional_chat_templates/default.jinja
    over tokenizer_config.json's field, which may be a string, a named list or a dict."""
    load = convert.load_chat_template
    assert load(_repo(tmp_path, "both", {"chat_template": "CFG"}, "FILE {{ x }}\n")) == "FILE {{ x }}\n"
    assert load(_repo(tmp_path, "file_only", {"bos_token": None}, "FILE")) == "FILE"
    assert load(_repo(tmp_path, "no_config", None, "FILE")) == "FILE"
    assert load(_repo(tmp_path, "named_file", {"chat_template": "CFG"},
                      extra={"additional_chat_templates/default.jinja": "NAMED",
                             "additional_chat_templates/tool_use.jinja": "TOOLS"})) == "NAMED"
    assert load(_repo(tmp_path, "tool_use_only", {"chat_template": "CFG"},
                      extra={"additional_chat_templates/tool_use.jinja": "TOOLS"})) == "CFG"
    assert load(_repo(tmp_path, "field", {"chat_template": "CFG"})) == "CFG"
    assert load(_repo(tmp_path, "named_list", {"chat_template": [{"name": "tool_use", "template": "T"},
                                                                 {"name": "default", "template": "D"}]})) == "D"
    assert load(_repo(tmp_path, "list_no_default", {"chat_template": [{"name": "rag", "template": "R"}]})) == "R"
    assert load(_repo(tmp_path, "dict", {"chat_template": {"tool_use": "T", "default": "D"}})) == "D"
    assert load(_repo(tmp_path, "none", {"chat_template": None})) is None
    assert load(_repo(tmp_path, "empty", {"chat_template": ""})) is None
    assert load(_repo(tmp_path, "nothing", {})) is None
    assert load(_repo(tmp_path, "no_files")) is None


def _byte_tokenizer(d: str) -> None:
    """The smallest tokenizer.json the converter accepts (as tests/test_reference.py)."""
    b2u = convert.bytes_to_unicode()
    tj = {"version": "1.0", "added_tokens": [{"id": 256, "content": "<|im_end|>", "special": True}],
          "normalizer": None, "post_processor": None,
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
          "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
          "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                    "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                    "vocab": {b2u[b]: b for b in range(256)}, "merges": []}}
    with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)


def test_convert_tokenizer_reads_chat_template_jinja(tmp_path):
    """A checkpoint laid out like SmolLM3 or Gemma 3 (tokenizer_config.json without chat_template,
    the template in chat_template.jinja) converts with that template in the header, and the
    format label is detected from it; without any template the header says plain."""
    tpl = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
           "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    d = _repo(tmp_path, "smol3", {"bos_token": None, "eos_token": "<|im_end|>", "model_max_length": 65536}, tpl)
    _byte_tokenizer(d)
    meta, _tensors = convert.convert_tokenizer(d, {"eos_token_id": 256})
    assert meta["chat_template"] == tpl and meta["chat_format"] == "chatml"
    assert meta["eos_id"] == 256 and meta["eos_ids"] == [256] and meta["bos_id"] == -1
    f = ChatFormatter(meta, token_text=lambda i: "<|im_end|>" if i == 256 else "")
    assert f.render(HI) == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert f.stop_strings() == ["<|im_end|>"]
    d = _repo(tmp_path, "plain", {"eos_token": "<|im_end|>"})
    _byte_tokenizer(d)
    meta, _tensors = convert.convert_tokenizer(d, {})
    assert meta["chat_template"] is None and meta["chat_format"] == "plain"


# ---- superborg/server.py: chat_template_kwargs ---------------------------------------------

class _Gen:
    def __init__(self, text="ok"):
        self.text, self.tokens, self.stop_reason = text, [1, 2], "eos"
        self.n_prompt = self.n_reused = 1
        self.prompt_ms = self.gen_ms = self.tokens_per_second = 0.0
        self.n_drafted = self.n_accepted = 0


class _FakeLLM:
    """What Service.chat_completion touches: rendering through a ChatFormatter and a generation
    that records the prompt text."""
    name, n_ctx = "fake", 1024

    def __init__(self, template):
        self.formatter = formatter(template, eos="<|im_end|>")
        self.prompts = []

    def render(self, messages, enable_thinking=False, add_generation_prompt=True):
        return self.formatter.render(messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)

    def tokenize(self, text, add_bos=None, special=True):
        self.prompts.append(text)
        return [1]

    def generate_tokens(self, prompt_tokens, stream=None, **kw):
        return _Gen()


def test_server_chat_template_kwargs():
    from superborg.server import APIError, Service
    llm = _FakeLLM(PROBE)
    svc = Service(llm)
    body = {"messages": HI, "chat_template_kwargs": {"date_string": "16 Oct 2026", "reasoning_effort": "low"}}
    assert svc.chat_completion(body)["choices"][0]["message"]["content"] == "ok"
    assert llm.prompts[-1].startswith("T=False|") and "|date=16 Oct 2026|" in llm.prompts[-1]
    # enable_thinking: the field, else chat_template_kwargs', else off
    svc.chat_completion({"messages": HI, "chat_template_kwargs": {"enable_thinking": True, "date_string": "x"}})
    assert llm.prompts[-1].startswith("T=True|") and "|date=x|" in llm.prompts[-1]
    svc.chat_completion({"messages": HI, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": True}})
    assert llm.prompts[-1].startswith("T=False|")
    svc.chat_completion({"messages": HI, "enable_thinking": True})
    assert llm.prompts[-1].startswith("T=True|")
    svc.chat_completion({"messages": HI})
    assert llm.prompts[-1].startswith("T=False|") and "|date=|" in llm.prompts[-1]
    for bad, needle in [({"chat_template_kwargs": [1]}, "must be an object"),
                        ({"chat_template_kwargs": {"messages": []}}, "cannot set 'messages'"),
                        ({"chat_template_kwargs": {"add_generation_prompt": False}}, "cannot set 'add_generation_prompt'"),
                        ({"chat_template_kwargs": {"enable_thinking": "yes"}}, "must be a boolean"),
                        ({"enable_thinking": "yes"}, "must be a boolean")]:
        with pytest.raises(APIError, match=needle) as e:
            svc.chat_completion(dict({"messages": HI}, **bad))
        assert e.value.status == 400
