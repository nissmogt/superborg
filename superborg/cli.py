"""Command-line interface: python -m superborg <command> (or the `superborg` script).

    pull REPO                 download a Hugging Face model and convert it to .borg
    convert HF_DIR OUT        convert a local Hugging Face checkpoint
    info MODEL                architecture, size, tokenizer and chat format of a .borg file
    run MODEL -p PROMPT       generate text (streams to stdout)
    chat MODEL                interactive chat with history
    ask MODEL QUESTION        answer with test-time reasoning (voting, programs, verification, retrieval)
    serve MODEL               OpenAI-compatible HTTP server
    ui MODEL                  local web UI: chat, and watch each reply token form inside the model
    bench MODEL               prefill/decode speed and speculative decoding statistics

MODEL is a path to a .borg file, or a name looked up in $SUPERBORG_MODELS (default
~/.cache/superborg), where `pull` stores its conversions.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass

DEFAULT_HOME = os.path.expanduser(os.environ.get("SUPERBORG_MODELS", "~/.cache/superborg"))
# what superborg.convert.convert accepts; checked before `pull` spends minutes downloading
QTYPES = ("q8_0", "q4", "q4_0", "q4_1", "f16", "bf16", "f32")
QTYPE_HELP = ("weight type: q8_0 (default, near-lossless), q4 (recommended 4-bit recipe: calibrated q4_1 with "
              "attn_v/ffn_down at q8_0), q4_0, q4_1, f16, bf16 or f32")


def _err(msg: str) -> None:
    print(f"superborg: {msg}", file=sys.stderr)


class CLIError(Exception):
    pass


# ---------------------------------------------------------------------------
# helpers

def resolve_model(name: str) -> str:
    """A path as given, or a model name found in the models directory (`pull` output).

    A unique prefix is enough ("qwen2.5-1.5b" for qwen2.5-1.5b-q8_0.borg); an ambiguous one is
    an error rather than a silent pick (the alphabetically first file is often f32, the slowest)."""
    if os.path.isfile(name):
        return name
    for d in (DEFAULT_HOME, os.path.join(DEFAULT_HOME, "models")):
        for cand in (name, name + ".borg"):
            p = os.path.join(d, cand)
            if os.path.isfile(p):
                return p
        if os.path.isdir(d):
            hits = sorted(f for f in os.listdir(d) if f.endswith(".borg") and f.startswith(name))
            if len(hits) == 1:
                return os.path.join(d, hits[0])
            if hits:
                raise CLIError(f"model name {name!r} is ambiguous in {d}: " + ", ".join(hits))
    raise CLIError(f"model not found: {name} (give a .borg path or `superborg pull` it first)")


def _load(a, draft: str | None = None):
    from . import LLM

    path = resolve_model(a.model)
    draft = getattr(a, "draft", None) if draft is None else draft
    return LLM(path, threads=a.threads, n_ctx=a.ctx, kv_f16=getattr(a, "kv_f16", False),
               draft=resolve_model(draft) if draft else None, n_draft=getattr(a, "n_draft", 6))


def _sampling_kwargs(a) -> dict:
    return {"temperature": a.temp, "top_k": a.top_k, "top_p": a.top_p, "min_p": a.min_p,
            "repeat_penalty": a.repeat_penalty, "seed": a.seed}


class _CtrlC:
    """While active, Ctrl-C ends the current generation instead of killing the program.

    A KeyboardInterrupt raised inside the engine's token callback would be swallowed by
    ctypes, so the handler only sets a flag that the stream callback turns into a stop.
    """

    def __init__(self):
        self.hit = False
        self._old = None

    def __enter__(self):
        self.hit = False
        try:
            self._old = signal.signal(signal.SIGINT, self._handle)
        except ValueError:  # not the main thread
            self._old = None
        return self

    def _handle(self, *_):
        self.hit = True

    def __exit__(self, *exc):
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
        return False


class _StopGeneration(Exception):
    pass


def _printer(intr: _CtrlC | None = None, lstrip: bool = False):
    """Stream callback writing to stdout. lstrip drops leading whitespace of a reply
    (SentencePiece models start their first token with a space)."""
    started = [not lstrip]

    def out(piece: str) -> None:
        if not started[0]:
            piece = piece.lstrip()
            started[0] = bool(piece)
        sys.stdout.write(piece)
        sys.stdout.flush()
        if intr is not None and intr.hit:
            raise _StopGeneration()  # the binding turns callback exceptions into a stop
    return out


def _stats_line(g) -> str:
    pre = g.n_prompt * 1000.0 / g.prompt_ms if g.prompt_ms > 0 else 0.0
    s = (f"prompt: {g.n_prompt} tokens evaluated, {g.n_reused} cached, {g.prompt_ms:.0f} ms ({pre:.1f} tok/s) | "
         f"generated: {len(g.tokens)} tokens, {g.gen_ms:.0f} ms ({g.tokens_per_second:.1f} tok/s) | "
         f"stop: {g.stop_reason}")
    if g.n_drafted:
        s += (f" | speculative: {g.n_accepted}/{g.n_drafted} drafts accepted "
              f"({100.0 * g.n_accepted / g.n_drafted:.0f}%), {g.n_target_evals} target passes")
    return s


def _read_text_arg(value: str | None, path: str | None) -> str | None:
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    if value == "-":
        return sys.stdin.read()
    return value


# ---------------------------------------------------------------------------
# pull / convert / info

def cmd_pull(a) -> int:
    from . import convert, hub

    dest = os.path.abspath(os.path.expanduser(a.dest))
    os.makedirs(dest, exist_ok=True)
    name = a.name or a.repo.rstrip("/").split("/")[-1].lower()
    out = os.path.join(dest, f"{name}-{a.qtype}.borg")
    if os.path.exists(out) and not a.force:
        _err(f"{out} exists (use --force to convert again)")
        print(out)
        return 0
    extra = _convert_extra(a)  # read the calibration file now: a typo must not cost a download
    hf_root = os.path.join(dest, "hf")
    hf_dir = hub.pull(a.repo, hf_root, quiet=a.quiet)
    try:
        convert.convert(hf_dir, out, a.qtype, source=a.repo, log=print if a.verbose else None, **extra)
    finally:
        if not a.keep_hf:
            shutil.rmtree(hf_dir, ignore_errors=True)
            with contextlib.suppress(OSError):
                os.rmdir(hf_root)  # only if empty
    _err(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    print(out)
    return 0


def _convert_extra(a) -> dict:
    """Optional converter features, passed only when used (older converters lack them)."""
    extra = {}
    if a.calibrate is not None:
        if a.calibrate == "builtin":
            extra["calibration"] = "builtin"
        else:
            with open(a.calibrate, encoding="utf-8") as f:
                extra["calibration"] = f.read()
    if a.keep_q8:
        extra["keep_q8"] = tuple(x.strip() for x in a.keep_q8.split(",") if x.strip())
    return extra


def cmd_convert(a) -> int:
    from . import convert

    if not os.path.isfile(os.path.join(a.hf_dir, "config.json")):
        raise CLIError(f"{a.hf_dir} is not a Hugging Face model directory (no config.json)")
    h = convert.convert(a.hf_dir, a.out, a.qtype, source=a.source, log=print if a.verbose else None,
                        **_convert_extra(a))
    _err(f"wrote {a.out}: {h['arch']}, {h['config']['n_layers']} layers, {a.qtype}, "
         f"{os.path.getsize(a.out) / 1e6:.1f} MB")
    return 0


def model_info(path: str) -> dict:
    """Facts about a .borg file, from its header and the engine."""
    from . import engine
    from .chat import formatter_for

    m = engine.Model(path, pool=engine.Pool(1))
    try:
        h = m.header
        cfg = h["config"]
        tok = h["tokenizer"]
        n_params = 0
        for name, t in h["tensors"].items():
            if not name.startswith("tokenizer."):
                n_params += math.prod(t["shape"])
        dtypes = Counter(t["dtype"] for n, t in h["tensors"].items() if not n.startswith("tokenizer."))
        fmt = formatter_for(m)
        return {
            "path": os.path.abspath(path),
            "source": h.get("source"),
            "arch": h.get("arch"),
            "quant": h.get("quant"),
            "file_bytes": os.path.getsize(path),
            "weight_bytes": m.weight_bytes,
            "parameters": n_params,
            "tensor_dtypes": dict(dtypes),
            "config": cfg,
            "tokenizer": {
                "type": "byte-level BPE" if tok.get("byte_level") else "SentencePiece BPE",
                "pretokenizer": tok.get("pretokenizer"),
                "n_vocab": m.n_vocab,
                "bos": tok.get("bos_id"), "eos": tok.get("eos_id"), "eos_ids": tok.get("eos_ids"),
                "add_bos": tok.get("add_bos"),
                "added_tokens": len(tok.get("added_tokens") or []),
            },
            "chat": {
                "format": tok.get("chat_format"),
                "jinja_template": bool(tok.get("chat_template")),
                "renderer": "jinja2" if (fmt.template and fmt.have_jinja) else f"built-in {fmt.fmt}",
                "thinking": fmt.supports_thinking,
                "stop_strings": fmt.stop_strings(),
            },
            "engine": {"version": engine.lib().borg_version().decode(), "simd": engine.simd()},
        }
    finally:
        m.close()


def cmd_info(a) -> int:
    info = model_info(resolve_model(a.model))
    if a.json:
        print(json.dumps(info, indent=2))
        return 0
    c = info["config"]
    t = info["tokenizer"]
    ch = info["chat"]
    rope = c.get("rope_scaling")
    lines = [
        ("file", f"{info['path']} ({info['file_bytes'] / 1e6:.1f} MB)"),
        ("source", info["source"]),
        ("architecture", f"{info['arch']}: {c['n_layers']} layers, dim {c['dim']}, ffn {c['hidden_dim']}, "
                         f"{c['n_heads']} heads / {c['n_kv_heads']} kv heads, head_dim {c['head_dim']}"),
        ("parameters", f"{info['parameters'] / 1e6:.1f} M"),
        ("weights", f"{info['quant']} ({', '.join(f'{k}: {v}' for k, v in sorted(info['tensor_dtypes'].items()))}"
                    f" tensors), {info['weight_bytes'] / 1e6:.1f} MB"),
        ("context", f"{c['max_seq_len']} tokens, rope theta {c['rope_theta']:g}"
                    + (f", scaling {rope.get('type')} x{rope.get('factor')}" if rope else "")),
        ("features", ", ".join(k for k in ("attn_bias", "qk_norm", "tie_embeddings") if c.get(k)) or "-"),
        ("activation", c.get("act", "silu")),
        ("tokenizer", f"{t['type']} ({t['pretokenizer']}), {t['n_vocab']} tokens, bos {t['bos']}, eos {t['eos']}"
                      f"{' (added)' if t['add_bos'] else ''}, eos ids {t['eos_ids']}"),
        ("chat format", f"{ch['format']} via {ch['renderer']}"
                        + (", thinking mode" if ch["thinking"] else "")),
        ("engine", f"borg {info['engine']['version']}, simd {info['engine']['simd']}"),
    ]
    w = max(len(k) for k, _ in lines)
    for k, v in lines:
        print(f"{k:<{w}}  {v}")
    return 0


# ---------------------------------------------------------------------------
# run / chat / ask

def cmd_run(a) -> int:
    prompt = _read_text_arg(a.prompt, a.file)
    if not prompt:
        raise CLIError("give a prompt with -p TEXT, -p - (stdin) or -f FILE")
    with _load(a) as llm:
        kw = dict(_sampling_kwargs(a), max_tokens=a.n, stop=a.stop or (), lookup=a.lookup)
        with _CtrlC() as intr:
            out = _printer(intr, lstrip=a.chat)
            if a.chat:
                msgs = ([{"role": "system", "content": a.system}] if a.system else []) + \
                       [{"role": "user", "content": prompt}]
                g = llm.chat(msgs, enable_thinking=a.thinking, stream=out, **kw)
            else:
                if a.echo:
                    out(prompt)
                g = llm.complete(prompt, stream=out, **kw)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if a.verbose:
            _err(_stats_line(g))
    return 0


def cmd_chat(a) -> int:
    interactive = sys.stdin.isatty()
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401 - line editing and history for input()
    with _load(a) as llm:
        system = a.system
        history: list[dict] = []
        verbose = a.verbose
        if interactive:
            _err(f"chatting with {llm.name} (n_ctx {llm.n_ctx}). Commands: /reset, /system TEXT, /stats, /exit")
        while True:
            try:
                line = input("> " if interactive else "")
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break
            if not line.strip():
                continue
            if line.startswith("/"):
                cmd, _, arg = line.strip().partition(" ")
                if cmd in ("/exit", "/quit", "/bye"):
                    break
                if cmd == "/reset":
                    history.clear()
                    print("(conversation cleared)")
                elif cmd == "/system":
                    system = arg.strip() or None
                    history.clear()
                    print("(system prompt set; conversation cleared)" if system else "(system prompt removed)")
                elif cmd == "/stats":
                    verbose = not verbose
                    print(f"(stats {'on' if verbose else 'off'})")
                elif cmd == "/help":
                    print("/reset  clear the conversation\n/system TEXT  set the system prompt\n"
                          "/stats  toggle timing statistics\n/exit  quit")
                else:
                    print(f"(unknown command {cmd}; try /help)")
                sys.stdout.flush()
                continue
            history.append({"role": "user", "content": line})
            g = None
            with _CtrlC() as intr:
                while g is None:
                    msgs = ([{"role": "system", "content": system}] if system else []) + history
                    try:
                        g = llm.chat(msgs, max_tokens=a.n, enable_thinking=a.thinking,
                                     stream=_printer(intr, lstrip=True), **_sampling_kwargs(a))
                    except ValueError as e:  # conversation no longer fits: forget the oldest exchange
                        if len(history) <= 1:
                            print(f"(message too long: {e})")
                            history.pop()
                            break
                        del history[:2]
            print()
            sys.stdout.flush()
            if g is not None:
                history.append({"role": "assistant", "content": g.text.strip()})
                if verbose:
                    _err(_stats_line(g))
    return 0


def retrieve_context(directory: str, question: str, k: int = 4, verbose: bool = False) -> str:
    """Top-k BM25 passages from the files under `directory`, formatted as reasoning context."""
    from . import retrieval

    if not os.path.isdir(directory):
        raise CLIError(f"--context-dir {directory} is not a directory")
    idx = retrieval.Index()
    n = idx.add_path(directory)
    hits = idx.search(question, k=k)
    if verbose:
        _err(f"indexed {n} chunks from {directory}; using {len(hits)}: "
             + ", ".join(f"{os.path.basename(h.source)}:{h.line} ({h.score:.1f})" for h in hits))
    return retrieval.format_context(hits)


def cmd_ask(a) -> int:
    try:
        from . import reason
    except ImportError as e:
        raise CLIError(f"the reasoning module is unavailable: {e}") from e
    question = _read_text_arg(a.question, None)
    if not question or not question.strip():
        raise CLIError("empty question")
    strategies = getattr(reason, "STRATEGIES", {})
    if a.strategy not in strategies:
        names = "\n".join(f"  {k:10s} {v}" for k, v in strategies.items())
        raise CLIError(f"unknown strategy '{a.strategy}'. Available:\n{names}")
    context = retrieve_context(a.context_dir, question, a.top_docs, a.verbose) if a.context_dir else None
    with _load(a) as llm:
        t0 = time.time()
        # progress (the model's tokens, or one line per sampled candidate) goes to stderr
        progress = (lambda piece: (sys.stderr.write(piece), sys.stderr.flush())) if a.verbose else None
        try:
            ans = reason.solve(llm, question, strategy=a.strategy, samples=a.samples, max_tokens=a.n, seed=a.seed,
                               context=context or None, stream=progress, enable_thinking=a.thinking)
        except ValueError as e:
            raise CLIError(str(e)) from e
        if a.verbose:
            sys.stderr.write("\n")
        if a.json:
            d = asdict(ans) if is_dataclass(ans) else dict(vars(ans))
            print(json.dumps(d, indent=2, default=str))
        else:
            print(ans.text)
        if a.verbose:
            route = getattr(ans, "route", "")
            _err(f"strategy: {ans.strategy}{f' ({route})' if route else ''} | answer: {ans.answer!r} | "
                 f"candidates: {len(ans.candidates or [])} | tokens: {ans.tokens} | {ans.elapsed_s:.1f} s "
                 f"(wall {time.time() - t0:.1f} s)")
    return 0


# ---------------------------------------------------------------------------
# serve / bench

def cmd_serve(a) -> int:
    from .server import serve

    serve(resolve_model(a.model), host=a.host, port=a.port, threads=a.threads, n_ctx=a.ctx,
          draft=resolve_model(a.draft) if a.draft else None, api_key=a.api_key, lookup=a.lookup,
          kv_f16=a.kv_f16, model_id=a.model_id, quiet=not a.verbose, allow_code=a.allow_code)
    return 0


def cmd_ui(a) -> int:
    from .ui import serve_ui

    serve_ui(resolve_model(a.model), host=a.host, port=a.port, threads=a.threads, n_ctx=a.ctx,
             lens_stride=a.lens_stride, open_browser=a.open, quiet=not a.verbose)
    return 0


_BENCH_TEXT = (
    "The history of computing is a story of abstraction. Early machines were programmed by rewiring "
    "panels; later, stored programs let the same hardware run many tasks. Compilers translated readable "
    "code into machine instructions, operating systems shared processors among users, and networks "
    "connected distant computers. Each layer hid the details beneath it, so that people could think "
    "about problems instead of circuits. Today a language model running on a laptop is built from the "
    "same ideas: matrix multiplications, caches, careful memory layout and a great deal of software. "
)


def _bench_tokens(llm, n: int, path: str | None) -> list[int]:
    text = open(path, encoding="utf-8").read() if path else _BENCH_TEXT
    toks = llm.tokenize(text)
    if not toks:
        raise CLIError("benchmark prompt is empty")
    body = llm.tokenize(text, add_bos=False)
    while len(toks) < n:
        toks += body
    return toks[:n]


def cmd_bench(a) -> int:
    from . import Sampling

    a.ctx = max(a.ctx, a.prompt + a.gen + 8)
    with _load(a) as llm:
        prompt = _bench_tokens(llm, a.prompt, a.prompt_file)
        greedy = Sampling(temperature=0.0)

        def run(spec: str):
            best = None
            for _ in range(a.repeat):
                llm.ctx.reset()
                kw = {}
                if spec == "draft":
                    llm.draft_ctx.reset()
                    llm.draft_ctx.sync(prompt)  # the draft's own prefill is not part of decode speed
                    kw = {"draft": llm.draft_ctx, "n_draft": a.n_draft}
                elif spec == "lookup":
                    kw = {"lookup_ngram": a.lookup, "n_draft": a.n_draft}
                g = llm.ctx.generate(prompt, max_tokens=a.gen, sampling=greedy, ignore_eos=True, **kw)
                if best is None or g.gen_ms < best.gen_ms:
                    best = g
            return best

        results = {}
        modes = ["plain"] + (["draft"] if a.draft else []) + (["lookup"] if a.lookup else [])
        for mode in modes:
            g = run(mode)
            results[mode] = {
                "prompt_tokens": g.n_prompt, "prompt_ms": g.prompt_ms,
                "prefill_tok_s": g.n_prompt * 1000.0 / g.prompt_ms if g.prompt_ms > 0 else 0.0,
                "gen_tokens": len(g.tokens), "gen_ms": g.gen_ms, "decode_tok_s": g.tokens_per_second,
                "drafted": g.n_drafted, "accepted": g.n_accepted, "target_evals": g.n_target_evals,
                "tokens": g.tokens,
            }
        base = results["plain"]
        for mode in modes[1:]:
            r = results[mode]
            r["speedup"] = r["decode_tok_s"] / base["decode_tok_s"] if base["decode_tok_s"] else 0.0
            # greedy speculative decoding must reproduce plain greedy decoding exactly
            r["matches_plain"] = r["tokens"] == base["tokens"]
        from . import engine
        summary = {"model": llm.name, "simd": engine.simd(), "threads": llm.pool.size, "n_ctx": llm.n_ctx,
                   "results": {k: {kk: vv for kk, vv in v.items() if kk != "tokens"} for k, v in results.items()}}
    if a.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"model {summary['model']} | simd {summary['simd']} | threads {summary['threads']} | "
          f"best of {a.repeat}")
    for mode, r in summary["results"].items():
        line = (f"{mode:7s} prefill {r['prompt_tokens']:5d} tok {r['prefill_tok_s']:8.1f} tok/s | "
                f"decode {r['gen_tokens']:4d} tok {r['decode_tok_s']:7.1f} tok/s")
        if mode != "plain":
            rate = 100.0 * r["accepted"] / r["drafted"] if r["drafted"] else 0.0
            per = r["gen_tokens"] / r["target_evals"] if r["target_evals"] else 0.0
            line += (f" | accepted {r['accepted']}/{r['drafted']} ({rate:.0f}%), {per:.2f} tok/target pass, "
                     f"speedup x{r['speedup']:.2f}, identical to plain: {'yes' if r['matches_plain'] else 'NO'}")
        print(line)
    return 0


# ---------------------------------------------------------------------------
# argument parsing

def _positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def _add_model_opts(p, ctx: int = 4096):
    p.add_argument("model", help="path to a .borg file (or a name in $SUPERBORG_MODELS)")
    p.add_argument("-t", "--threads", type=int, default=0, help="worker threads (default: all cores)")
    p.add_argument("-c", "--ctx", type=int, default=ctx, help=f"context length in tokens (default {ctx})")
    p.add_argument("--kv-f16", action="store_true", help="store the KV cache as float16")


def _add_sampling_opts(p, temp: float):
    p.add_argument("--temp", type=float, default=temp, help=f"temperature (0 = greedy, default {temp})")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repeat-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0x5EED)


def _add_quant_opts(p):
    p.add_argument("--calibrate", nargs="?", const="builtin", default=None, metavar="TEXT_FILE",
                   help="4-bit only: weight rounding errors by activation statistics measured on a text "
                        "(a built-in sample when no file is given)")
    p.add_argument("--keep-q8", default="", help="4-bit only: comma-separated tensor name suffixes kept at q8_0")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="superborg", description="Fast local inference for open-weight models.")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("pull", help="download a Hugging Face model and convert it to .borg")
    p.add_argument("repo", help="Hugging Face repo id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dest", default=DEFAULT_HOME, help=f"output directory (default {DEFAULT_HOME})")
    p.add_argument("--qtype", default="q8_0", choices=QTYPES, metavar="QTYPE", help=QTYPE_HELP)
    _add_quant_opts(p)
    p.add_argument("--name", help="output file stem (default: lowercase repo name)")
    p.add_argument("--keep-hf", action="store_true", help="keep the downloaded Hugging Face files")
    p.add_argument("--force", action="store_true", help="convert even if the output exists")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("convert", help="convert a Hugging Face model directory to .borg")
    p.add_argument("hf_dir")
    p.add_argument("out")
    p.add_argument("--qtype", default="q8_0", choices=QTYPES, metavar="QTYPE", help=QTYPE_HELP)
    _add_quant_opts(p)
    p.add_argument("--source", help="model id recorded in the header")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("info", help="show a model's architecture, tokenizer and chat format")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("run", help="generate text from a prompt (streams to stdout)")
    _add_model_opts(p)
    p.add_argument("-p", "--prompt", help="prompt text ('-' reads stdin)")
    p.add_argument("-f", "--file", help="read the prompt from a file")
    p.add_argument("-n", type=int, default=256, help="maximum tokens to generate (default 256)")
    p.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
    p.add_argument("--system", help="system message (with --chat)")
    p.add_argument("--thinking", action="store_true", help="enable thinking mode (Qwen3, with --chat)")
    p.add_argument("--stop", action="append", help="stop string (repeatable)")
    p.add_argument("--echo", action="store_true", help="print the prompt before the continuation")
    p.add_argument("--lookup", type=int, default=0, help="prompt-lookup speculative decoding with N-grams")
    p.add_argument("--draft", help="draft model for speculative decoding (same tokenizer)")
    p.add_argument("--n-draft", type=_positive, default=6, help="tokens proposed per speculative step")
    _add_sampling_opts(p, 0.0)
    p.add_argument("-v", "--verbose", action="store_true", help="print timing statistics to stderr")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("chat", help="interactive chat (/reset, /system TEXT, /stats, /exit)")
    _add_model_opts(p)
    p.add_argument("--system", help="system message")
    p.add_argument("--thinking", action="store_true", help="enable thinking mode (Qwen3)")
    p.add_argument("-n", type=int, default=1024, help="maximum tokens per reply")
    p.add_argument("--draft", help="draft model for speculative decoding")
    p.add_argument("--n-draft", type=_positive, default=6)
    _add_sampling_opts(p, 0.7)
    p.set_defaults(min_p=0.05)
    p.add_argument("-v", "--verbose", action="store_true", help="print timing statistics after each reply")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ask", help="answer a question with test-time reasoning strategies")
    _add_model_opts(p)
    p.add_argument("question", help="the question ('-' reads stdin)")
    p.add_argument("--strategy", default="auto", help="reasoning strategy (default auto; see superborg.reason)")
    p.add_argument("--samples", type=_positive, default=5, help="samples for voting strategies (default 5)")
    p.add_argument("--context-dir", help="answer from the text files in this directory (retrieval)")
    p.add_argument("--top-docs", type=_positive, default=4, help="passages retrieved with --context-dir")
    p.add_argument("-n", type=int, default=512, help="maximum tokens per sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", action="store_true", help="enable thinking mode (Qwen3)")
    p.add_argument("--draft", help="draft model for speculative decoding")
    p.add_argument("--json", action="store_true", help="print the full Answer as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    _add_model_opts(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--draft", help="draft model for speculative decoding")
    p.add_argument("--lookup", type=int, default=0, help="prompt-lookup speculative decoding by default")
    p.add_argument("--api-key", default=os.environ.get("SUPERBORG_API_KEY"), help="require this bearer token")
    p.add_argument("--model-id", help="model name reported by /v1/models")
    p.add_argument("--allow-code", action="store_true",
                   help="let reasoning strategies run model-written Python (pal, pal_vote) for non-browser and "
                        "local-origin requests; off by default because prompts can steer that code")
    p.add_argument("-v", "--verbose", action="store_true", help="log requests")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("ui", help="local web UI: chat and watch each reply token form inside the model")
    p.add_argument("model", help="path to a .borg file (or a name in $SUPERBORG_MODELS)")
    p.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1: this machine only)")
    p.add_argument("--port", type=int, default=8765, help="port (default 8765; 0 picks a free one)")
    p.add_argument("-t", "--threads", type=int, default=0, help="worker threads (default: all cores)")
    p.add_argument("-c", "--ctx", type=int, default=4096, help="context length in tokens (default 4096)")
    p.add_argument("--lens-stride", type=_positive, default=1, metavar="N",
                   help="logit lens on every N-th layer (default 1: all; larger is faster per token)")
    p.add_argument("--open", action="store_true", help="open the UI in a web browser")
    p.add_argument("-v", "--verbose", action="store_true", help="log requests")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("bench", help="measure prefill and decode speed")
    _add_model_opts(p, ctx=0)
    p.add_argument("--prompt", type=_positive, default=256, help="prompt tokens (default 256)")
    p.add_argument("--gen", type=_positive, default=64, help="generated tokens (default 64)")
    p.add_argument("--prompt-file", help="use this text as the prompt")
    p.add_argument("--draft", help="draft model: also benchmark speculative decoding with it")
    p.add_argument("--lookup", type=int, default=0, help="also benchmark prompt lookup with N-grams")
    p.add_argument("--n-draft", type=_positive, default=6)
    p.add_argument("--repeat", type=_positive, default=1, help="runs per mode; the fastest is reported")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_bench)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except CLIError as e:
        _err(str(e))
        return 2
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # e.g. piped into `head`
        with contextlib.suppress(Exception):
            sys.stdout = open(os.devnull, "w")
        return 0
    # OSError: unreadable prompt files, network failures in `pull`, a port in use for `serve`
    except (OSError, RuntimeError, MemoryError, ValueError) as e:
        _err(f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
