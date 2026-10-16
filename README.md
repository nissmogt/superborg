# superborg

superborg runs open-weight language models on a CPU, without a GPU, framework or
internet connection, and adds a reasoning layer that spends extra local computation on
checking answers.

It has three parts:

* **borg**, an inference engine written from scratch in C11 (about 5,000 lines, no
  dependencies). It loads a model with one `mmap` call and runs on x86-64 (scalar, AVX2,
  AVX-512 VNNI) and ARM64 (NEON, dot-product), on Linux, macOS and Windows (MinGW).
* **A Python package**: a converter from Hugging Face checkpoints, a model downloader,
  chat templates, an OpenAI-compatible HTTP server and a command-line tool.
* **A reasoning layer**: self-consistency voting, program-aided reasoning with a
  sandboxed Python runner, answer verification, retrieval over local files and a
  tool-using agent.

## Reimplementing instead of distilling

Distillation trains a smaller student model to imitate a larger one: the student is
cheaper but weaker, and training it needs data and GPUs. superborg keeps the original
model's weights and reimplements its computation from the published architecture
(configuration, tensor layout, tokenizer rules). The goal is the same output as the
reference implementation, produced faster and on more devices.

Checked against Hugging Face `transformers` and `tokenizers`:

| check | result |
|-------|--------|
| logits, float32 weights (SmolLM2-135M, Qwen2.5-0.5B, Qwen3-0.6B, TinyLlama-1.1B) | max relative error ≤ 5.4e-6; same argmax at every position |
| logits on 200-token prompts (NumPy reference, `tests/test_reference.py`) | relative error ≤ 3e-6; identical greedy continuations |
| tokenizer, about 25,000 adversarial strings per model (scripts, emoji, whitespace, digits, special tokens) | 0 mismatches |
| batched vs token-by-token evaluation, speculative vs plain greedy decoding | bit-identical |

Speed comes from the runtime: quantized integer kernels, a thread pool built for shared
machines, batched evaluation, speculative decoding and parallel sampling.

## Measured results

All numbers come from a 4-core Intel Xeon VM (2.8 GHz, AVX-512 VNNI), 4 threads; the
logs are in `bench/results/`.

### Speed

| model | file size | prompt (tok/s) | generation (tok/s) |
|-------|----------:|---------------:|-------------------:|
| SmolLM2-135M, q8_0 | 138 MB | 535 | 130 |
| SmolLM2-135M, q4 | 106 MB | 601 | 154 |
| Qwen2.5-0.5B, q8_0 | 505 MB | 223 | 45 |
| Qwen2.5-0.5B, q4 | 400 MB | 212 | 51 |
| Qwen3-0.6B, q8_0 | 608 MB | 172 | 33 |
| TinyLlama-1.1B, q8_0 | 1,116 MB | 93 | 23 |
| Qwen2.5-1.5B, q8_0 | 1,569 MB | 69 | 16 |
| Qwen2.5-1.5B, q4 | 1,187 MB | 67 | 19 |

Compared with llama.cpp (build 59d5fad, Qwen2.5-0.5B-Instruct Q8_0, 128-token prompt,
3 alternating runs; `bench/compare_llamacpp.sh`):

| condition | engine | prompt (tok/s) | generation (tok/s) |
|-----------|--------|---------------:|-------------------:|
| idle machine | borg | 190–233 | 45–47 |
| | llama.cpp | 299–342 | 41–43 |
| two other busy processes | borg | 124–130 | 25–27 |
| | llama.cpp | 57–81 | 7–11 |

llama.cpp evaluates prompts faster; borg generates about 10% faster on an idle machine
and 2–4× faster when other programs compete for the cores. borg's thread pool finishes a
step when its work items are done instead of waiting for every worker thread to be
scheduled (see `docs/ARCHITECTURE.md`).

### Speculative decoding and parallel sampling

Qwen2.5-1.5B q4 (`bench/speedups.py`); the output is token-for-token identical to plain
greedy decoding in every case.

| prompt | prompt lookup | Qwen2.5-0.5B as draft model |
|--------|--------------:|----------------------------:|
| rewrite code | 1.29× | 1.09× |
| answer from a document | 1.74× | 0.82× |
| open-ended writing | 0.98× | 0.82× |

Prompt lookup drafts tokens by copying from earlier in the context and helps whenever
the answer quotes the input. A draft model pays off only when it is much smaller than
the target: here it costs a third of a target step per drafted token.

Sampling several answers to one prompt (for voting) decodes them together, reading the
weights once per step for all of them: 4 samples run 2.4× and 8 samples 2.6× faster than
one after another.

### 4-bit quantization

Qwen2.5-0.5B, 512 tokens of text, compared with the float32 model (`bench/quant_eval.py`):

| file | size | KL divergence | perplexity |
|------|-----:|--------------:|-----------:|
| f32 | 1,980 MB | 0 | 3.831 |
| q8_0 | 529 MB | 0.0026 | 3.844 |
| q4_0 | 350 MB | 0.189 | 4.258 |
| `q4` recipe | 419 MB | 0.052 | 3.858 |

The `q4` recipe stores most matrices as 4-bit blocks with a scale and an offset, picks
each block's rounding to minimise the error weighted by activation statistics measured on
a calibration text, and keeps `attn_v` and `ffn_down` at 8 bits.

### Reasoning strategies on GSM8K

Qwen2.5-1.5B-Instruct at 4 bits on 100 held-out problems: 68% when asked the question as
is, 76% with a step-by-step prompt, 80% with a majority vote over 5 step-by-step solutions.
Details in [Reasoning results](#reasoning-results) below.

## Quick start

Requirements: a C compiler, Python 3.10+ and NumPy (`jinja2` is optional and gives exact
chat templates).

```sh
make                     # builds build/libborg.so and the build/borg command-line tool
pip install .            # optional: installs the Python package and compiles the engine

# download and convert a model (q8_0: near lossless; q4: about 25% smaller)
python -m superborg pull Qwen/Qwen2.5-1.5B-Instruct --qtype q4

python -m superborg run qwen2.5-1.5b-instruct-q4 --chat -p "Explain RAID 5 in two sentences."
python -m superborg chat qwen2.5-1.5b-instruct-q4
python -m superborg ask qwen2.5-1.5b-instruct-q4 "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes?"
python -m superborg ask qwen2.5-1.5b-instruct-q4 "What is our refund policy?" --context-dir ./docs
python -m superborg serve qwen2.5-1.5b-instruct-q4 --port 8080      # OpenAI-compatible API
python -m superborg bench qwen2.5-1.5b-instruct-q4 --lookup 3
```

The C tool needs no Python at all:

```sh
./build/borg -m model.borg --chat -p "Hello" -n 128
./build/borg -m model.borg --chat --json -p "List three primary colors as JSON."
./build/borg -m model.borg --bench 128 64
```

From Python:

```python
import superborg

llm = superborg.LLM("qwen2.5-1.5b-instruct-q4.borg", n_ctx=4096)
print(llm.ask("What causes the seasons?"))

from superborg.reason import solve
ans = solve(llm, "If 3 pencils cost $0.75, how much do 20 pencils cost?", strategy="auto")
print(ans.answer, ans.strategy)
```

## In the browser

The same C engine compiles to WebAssembly (`wasm/build.sh`, clang with the wasm32-wasi
target) and runs in a Web Worker. `wasm/make_artifact.py` packages the chat and activation
UI with the engine and a model as a static site:

```sh
wasm/build.sh                    # build/borg.wasm (SIMD128, single thread)
wasm/build-js.sh                 # optional build/borg-asm.js for pages that block WebAssembly
python wasm/make_artifact.py smollm2-135m-q4.borg site/
cd site && python -m http.server # open http://localhost:8000/
```

Measured on this machine in headless Chromium with SmolLM2-135M q4 (110.8 MB, downloaded
in 8 parts and cached in IndexedDB): about 32 ms per decoded token, and 6.6–9.4 tok/s for
chat with the full trace (logit lens on every third layer, attention, MLP, norms) recorded
for every token. The lens costs more than the decode step itself (11 rows through the
49,152 × 576 output matrix, about 43 ms). The wasm2js fallback runs the same chat at
1.6 tok/s. `tests/test_wasm.py` checks the WebAssembly build against the native engine:
identical tokenization, logits within 4e-4 of the native f32 result, the same trace events.

## Supported models

Decoder-only models with these Hugging Face `model_type` values: `llama` (Llama 2/3,
SmolLM, TinyLlama), `mistral` (contexts up to its sliding window), `qwen2` (Qwen2,
Qwen2.5), `qwen3`. Tokenizers: byte-level BPE (GPT-2, Llama 3, Qwen2, SmolLM rules) and
SentencePiece-style BPE with a prepend normalizer. The converter refuses checkpoints it
cannot reproduce exactly (other architectures, sliding-window attention, Metaspace
tokenizers, added tokens with strip rules) instead of converting them approximately.

## Reasoning results

GSM8K test problems 200–299, Qwen2.5-1.5B-Instruct with the `q4` recipe, 2048-token
context, at most 512 generated tokens per candidate, sampled candidates at temperature 0.7
and top-p 0.95, seed 0. The prompts and answer extraction were developed on problems 0–39
and 100–128 with Qwen2.5-0.5B; one problem of this range (#281) also appears in a test of
the question router, which these runs do not use.

```sh
python -m eval.gsm8k --model qwen2.5-1.5b-q4.borg --data gsm8k_test.jsonl --offset 200 --n 100 \
    --strategies direct,cot,vote,pal_vote --samples 5 --threads 4 --out results.json
```

| strategy | what it does | accuracy | 95% CI | solved where direct failed / failed where direct solved | p vs direct | tokens per problem |
|---|---|---|---|---|---|---|
| `direct` | one greedy answer to the question as asked (the instruct model still writes out its working) | 68% | 58–76% | – | – | 288 |
| `cot` | one greedy solution to "reason step by step, and put your final answer within \boxed{}" | 76% | 67–83% | 14 / 6 | 0.12 | 298 |
| `vote` | the `cot` solution plus 4 sampled ones, majority vote on the final answer | 80% | 71–87% | 16 / 4 | 0.012 | 1,463 |
| `pal_vote` | 5 Python programs (greedy with one repair attempt, plus 4 sampled) executed and their results voted; a greedy step-by-step solution decides ties | 73% | 64–81% | 18 / 13 | 0.47 | 475 |

The p values are exact two-sided McNemar tests on the paired per-problem results and the
intervals are Wilson intervals, both computed by `eval/gsm8k.py`. Per-problem records with
every candidate are in `eval/results/gsm8k_qwen2.5-1.5b-q4_200-299.json`; running the
command above with `--out` pointing at that file prints the table again without generating.

* Most of the gain comes from the step-by-step prompt: 68% to 76% (p = 0.12). Voting adds
  4 points over the same greedy solution (6 problems gained, 2 lost against `cot`,
  p = 0.29). Only the combined gain over `direct` is unlikely to be chance (p = 0.012, or
  0.035 after a Bonferroni correction for the three comparisons). Voting generates 5.1
  times as many tokens.
* Programs solve a different subset: `pal_vote` fixes 18 problems the direct answer gets
  wrong and breaks 13 it gets right. 13 of its 500 programs produced no answer (syntax and
  name errors, timeouts, one cut off at the token limit). The written tiebreaker ran on 14
  problems and changed the outcome on 5 (4 to correct, 1 to wrong).
* 90 problems are solved by at least one of the four strategies and 10 by none, so choosing
  the best of these four runs for each problem would reach at most 90%. On 3 of the 10, a
  correct candidate lost the vote or the tie-break.
* The run shared a 4-core machine with other jobs (tests, builds) for much of the time, so
  its wall-clock times (21–60 s per problem on average, depending on the strategy) are not
  a speed measurement; see [Speed](#speed) for those.

## Limitations

* The quality ceiling is the base model. Step-by-step prompting with voting lifts a 1.5B
  model from 68% to 80% on grade-school math above, but it is still a 1.5B model.
* Prompt evaluation is slower than llama.cpp's on an idle machine (see above).
* CPU only. MSVC builds are single-threaded (MinGW and clang builds use threads).
* `pal`, `pal_vote` and the agent run model-written Python in a subprocess with resource
  limits but no full sandbox; the HTTP server refuses them unless started with
  `--allow-code`.

## Layout

```
engine/              C engine (borg.h is the public API)
superborg/           Python package: engine binding, converter, chat, reasoning, server, CLI
docs/FORMAT.md       the .borg file format
docs/ARCHITECTURE.md design notes
tests/               pytest suites (tokenizer parity, reference, engine, server, reasoning)
tests/c/             C kernel and JSON-mode tests (make test-c)
bench/               benchmarks and their logs
eval/gsm8k.py        GSM8K evaluation of the reasoning strategies
wasm/                WebAssembly build, browser runtime, Web Worker and static-site packager
tools/               Unicode table generator, Hugging Face oracle, fixture builder
```

## Tests

```sh
make test-c
pytest tests/          # model-dependent tests skip when BORG_MODELS_DIR has no models
```

Tokenizer and reference tests compare against `tokenizers` / `transformers` when they are
installed (`HF_MODELS_DIR` points at the original checkpoints).
