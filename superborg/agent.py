"""A minimal tool-using agent loop (ReAct style) for small local models.

The protocol is plain text that even sub-1B instruction models follow:
  * to use Python, the model writes one fenced block  ```python ... ```
  * the harness runs it (superborg.tools.run_python) and replies with its output
  * when done, the model writes a line "Final Answer: ..."
Generation stops right after a closing code fence, so the model never invents tool output.

    from superborg.agent import run_agent
    res = run_agent(llm, "What is the 20th Fibonacci number times 3?")
    print(res.answer, len(res.steps))

Code runs in an isolation-light subprocess (see superborg.tools): only use this with
your own questions on your own machine.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from . import tools
from .reason import _strip_thinking, extract_answer

__all__ = ["AgentResult", "Step", "run_agent", "AGENT_SYSTEM"]

AGENT_SYSTEM = (
    "You solve problems step by step and can run Python. To run code, write it in a block:\n"
    "```python\nprint(2 + 2)\n```\n"
    "You will then see its output. Use print() to show results. Run code for any calculation. "
    "When you know the answer, write a last line: Final Answer: <answer>"
)
_OBSERVATION = "Output:\n```\n{output}\n```"
_FINAL_RE = re.compile(r"final answer\s*[:：]\s*(.+)", re.I)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n", re.I)


@dataclass
class Step:
    thought: str                  # model text of this step (including any code block)
    code: str | None = None
    observation: str | None = None
    ok: bool | None = None


@dataclass
class AgentResult:
    answer: str | None            # extracted final answer (number or short phrase)
    text: str                     # the model's final message
    steps: list[Step] = field(default_factory=list)
    tokens: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = ""         # "final", "no_tool_call", "max_steps", "length"


def _pending_code(text: str) -> str | None:
    """Code of a python block opened in `text` (closed or cut off by the stop string)."""
    starts = list(_OPEN_FENCE.finditer(text))
    if not starts:
        return None
    body = text[starts[-1].end():]
    body = body.split("```", 1)[0]
    return body.strip("\n") or None


def run_agent(llm, question: str, max_steps: int = 5, max_tokens: int = 512, system: str = AGENT_SYSTEM,
              temperature: float = 0.0, seed: int = 0, timeout: float = 10.0,
              stream: Callable[[str], None] | None = None, enable_thinking: bool = False,
              numeric: bool | None = None) -> AgentResult:
    """Alternate model turns and Python executions until a final answer or max_steps.

    max_tokens bounds each model turn; observations are truncated so the conversation
    stays well inside the context window of small models.
    """
    t0 = time.perf_counter()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    steps: list[Step] = []
    tokens = 0
    stops = ["\n```\n", "Output:\n```"] + list(llm.formatter.stop_strings())
    reason = "max_steps"
    text = ""
    for i in range(max_steps):
        prompt = llm.render(msgs, enable_thinking=enable_thinking)
        raw = ""
        budget = max_tokens
        for _ in range(3):
            g = llm.generate_tokens(llm.tokenize(prompt + raw, add_bos=False), max_tokens=budget,
                                    temperature=temperature, seed=seed + i, stop=stops, stream=stream)
            tokens += len(g.tokens)
            budget -= len(g.tokens)
            raw += g.text
            if budget <= 0:
                break
            # a bare ``` that *opens* a block also matches the stop string: keep generating
            # (end-of-turn tokens end generation with "eos", so "stop" means one of our strings)
            if (g.stop_reason == "stop" and raw.count("```") % 2 == 0 and not raw.rstrip().endswith("```")
                    and not raw.rstrip().endswith("Output:")):
                raw += "\n```\n"
                continue
            break
        text = _strip_thinking(raw)
        code = _pending_code(text)
        final = _FINAL_RE.search(text)
        if code is None or (final and final.start() > text.rfind("```")):
            reason = "final" if final else ("length" if g.stop_reason == "length" else "no_tool_call")
            steps.append(Step(thought=text.strip()))
            break
        res = tools.run_python(code, timeout=timeout)
        obs = res.observation(limit=1200)
        steps.append(Step(thought=text.strip(), code=code, observation=obs, ok=res.ok))
        if stream:
            stream(f"\n```\n{_OBSERVATION.format(output=obs)}\n")
        body = text.rstrip()
        if not body.endswith("```"):
            body += "\n```"
        msgs.append({"role": "assistant", "content": body})
        msgs.append({"role": "user", "content": _OBSERVATION.format(output=obs)})
    else:
        # out of steps: ask once more for the answer without tools
        msgs.append({"role": "user", "content": "Now give the final answer as: Final Answer: <answer>"})
        prompt = llm.render(msgs, enable_thinking=enable_thinking)
        g = llm.generate_tokens(llm.tokenize(prompt, add_bos=False), max_tokens=min(128, max_tokens),
                                temperature=0.0, stop=list(llm.formatter.stop_strings()), stream=stream)
        tokens += len(g.tokens)
        text = _strip_thinking(g.text)
        steps.append(Step(thought=text.strip()))
    m = _FINAL_RE.findall(text)
    if m:
        answer = extract_answer("Answer: " + m[-1].strip(), numeric)
    else:
        answer = extract_answer(text, numeric)
        if answer is None:  # last resort: what the last program printed
            obs = next((s.observation for s in reversed(steps) if s.observation and s.ok), None)
            answer = extract_answer(obs, numeric) if obs else None
    return AgentResult(answer=answer, text=text.strip(), steps=steps, tokens=tokens,
                       elapsed_s=time.perf_counter() - t0, stop_reason=reason)
