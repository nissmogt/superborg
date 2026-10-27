# Architecture

superborg has three layers:

1. **borg**, an inference engine in C11 (`engine/`, about 4,900 lines, no dependencies).
2. **The Python package** (`superborg/`): a ctypes binding, chat templates, a converter
   from Hugging Face checkpoints, and a reference implementation in NumPy.
3. **The reasoning layer** (`superborg/reason.py`, `tools.py`, `retrieval.py`, `agent.py`):
   test-time strategies that trade extra local computation for better answers.

## Reimplementing models instead of distilling them

Distillation trains a new, smaller student model to imitate a large one. The student is
cheaper to run, but it is a different and weaker model, and training it needs data and GPUs.

superborg keeps the original open-weight model and rebuilds its inference path:

* The converter reads the published `config.json`, `tokenizer.json` and safetensors
  tensors and maps them to one memory-mappable file (`docs/FORMAT.md`).
* The engine reimplements the forward pass of each supported architecture from the
  published equations: RMSNorm, rotary embeddings (with Llama 3 and linear scaling),
  grouped-query attention, SwiGLU, Qwen2 attention biases, and Qwen3 per-head q/k norms.
* The tokenizer reimplements byte-level BPE (GPT-2, Llama 3, Qwen2 and SmolLM
  pre-tokenizers, written as hand-coded matchers instead of a regex engine) and
  SentencePiece-style BPE with byte fallback.

Each piece is checked against the reference implementation, not just eyeballed:

| check | method | result |
|-------|--------|--------|
| forward pass | logits vs. `transformers` (float32, eager attention) on 30 positions | max relative error ≤ 5.4e-6, argmax identical at every position, for SmolLM2-135M, Qwen2.5-0.5B, Qwen3-0.6B, TinyLlama-1.1B |
| incremental decoding | token-by-token vs. batched prefill | bit-identical |
| batched multi-sequence decoding | vs. single-sequence decoding | bit-identical (f32) |
| tokenizer | `tokenizers.Tokenizer.from_file` on an adversarial corpus | see `tests/test_tokenizer.py` |
| kernels | every SIMD path vs. a double-precision reference; the interleaved-layout kernels vs. their scalar contract, bit for bit, for every token count | `make test-c`, run on AVX-512 VNNI, AVX2, AVX-VNNI, scalar x86, NEON and NEON+dotprod (under QEMU), WebAssembly SIMD128 and scalar (under Node) |
| speculative decoding | greedy output vs. plain greedy | identical tokens |

Speed and memory come from the runtime and from quantization, not from changing the model.

## Engine

### Memory

The `.borg` file is memory-mapped and every tensor starts on a 64-byte boundary. Float
tensors are used in place. Quantized matrices are repacked at load time into the
interleaved layout of the batched kernels (below): a mapped file gets a malloc'ed copy and
the mapped pages are released with `madvise(MADV_DONTNEED)`, so resident memory stays one
model size (547 MB after loading and decoding with Qwen2.5-0.5B q8_0, against 546 MB
without the repack); a buffer the engine read itself, or the caller's buffer given to
`borg_model_load_memory` (the WebAssembly path), is converted in place through a one-group
scratch, so that buffer is modified and must not be read back as a `.borg` file. Loading
therefore reads and converts every quantized tensor instead of mapping the file lazily:
about 0.55 s for the 525 MB Qwen2.5-0.5B q8_0 file and 1.2 s for the 1.24 GB Qwen2.5-1.5B q4
file from the page cache (`bench/results/prefill_after.txt`; the conversion itself runs at
2–3.6 GB/s, the rest is first-touch page faults of the new copy), against 25–40 ms before.
`BORG_NO_REPACK=1` keeps the plain layout and the per-row kernels. The KV cache is
allocated per context, in float32 or float16, laid out `[layer][kv_head][position][head_dim]`
so each attention head reads contiguous memory.

### Quantization

| type   | bits/weight | layout |
|--------|------------:|--------|
| `q8_0` | 8.5 | 32 values per block, int8 + f16 scale |
| `q4_0` | 4.5 | 32 values per block, 4-bit + f16 scale |
| `q4_1` | 5.0 | 32 values per block, 4-bit + f16 scale and offset |

Activations are quantized to int8 per 32-value block on the fly, so the inner loops are
integer dot products (`vpdpbusd` on AVX-512 VNNI, `maddubs` on AVX2, `sdot` on ARMv8.2).

The converter's `q4` recipe is q4_1 with **calibrated rounding**: the engine runs the
model on a calibration text and records the mean squared activation of every input
column (`borg_ctx_enable_stats`); each 4-bit block then picks the scale and offset that
minimise the error weighted by those statistics. `attn_v` and `ffn_down`, which carry most
of the remaining error, stay at 8 bits. On Qwen2.5-0.5B (512 tokens, `bench/quant_eval.py`):

| file | size | KL vs f32 | perplexity |
|------|-----:|----------:|-----------:|
| f32 | 1980 MB | 0 | 3.831 |
| q8_0 | 529 MB | 0.0026 | 3.844 |
| q4_0 | 350 MB | 0.189 | 4.258 |
| q4_1 + calibration | 372 MB | 0.085 | 4.059 |
| `q4` recipe | 419 MB | 0.052 | 3.858 |

### Threads

`pool.c` runs data-parallel loops. Work is handed out in chunks through an atomic counter
and the caller works too. A dispatch completes when its *items* are done; it does not wait
for every worker to check in. On a loaded machine (a phone, or a laptop running other
programs), waiting for a descheduled worker on each of the ~150 dispatches per token made
decoding 50× slower in an early version; the current design keeps working at full speed
with whichever threads are running. Workers join a dispatch with a compare-and-swap on a
word holding (generation, closed flag, joined count), so a late worker can never touch a
finished dispatch. Idle workers spin briefly, then yield, then sleep.

### Forward pass

Prompt tokens are processed in chunks (64 by default). The same batched path serves
speculative verification and multi-sequence decoding: each row of a chunk carries its own
KV cache and position, so rows may belong to different sequences.

**Matrix products.** Quantized matrices are stored in groups of 8 rows: per 32-element
block, the 8 rows' fp16 scales (and offsets for q4_1), then the rows' quant bytes
interleaved 4 bytes at a time, so one 32-byte load is 8 columns × 4 elements and a token's
4 activation bytes are broadcast against it (`vpbroadcastd`, `vdupq_lane`); q8_0 bytes are
stored as `w + 127` so the unsigned-by-signed dot instructions apply, and the offset comes
out as `127 × Σx` in integer arithmetic (ik_llama.cpp's `_R8` recipe, oneDNN's s8s8
compensation). A micro-kernel computes a tile of columns × tokens: 16 × 8 on AVX-512
(two groups per zmm, 8 int32 and 8 fp32 accumulators), 8 × 4 on AVX2 (`vpdpbusd` with
AVX-VNNI or AVX-512 VL, otherwise `maddubs` on sign-corrected weights), 8 × 4 on NEON
(`sdot` by lane, or widening multiplies) and 8 × 4 on WebAssembly SIMD128 (`i32x4.dot_i16x8`
on widened chunks). Per block it accumulates the exact int32 dot and then does one convert,
one multiply and one add per token for the whole tile, instead of one per (row, token). The
token loop is unrolled with macros so the accumulators stay in registers (left to the
optimizer, the same kernel ran 3× slower). An 8 × 8 AVX2 tile was measured against 8 × 4
on the benchmark machine with the 16-column path disabled: with `vpdpbusd` it spills (16
ymm registers) and runs q8_0 at 41–43 GMAC/s against 70–71 for 8 × 4, and with `maddubs`
the two are within noise of each other, so AVX2 keeps 8 × 4. Larger forward chunks were
measured too (`bench/bench_nbatch.py`): a 256-token prompt in chunks of 32/64/128/256
tokens ran at 337/335/340/382 tok/s for Qwen2.5-0.5B q8_0 and 506/610/667/712 tok/s for
SmolLM2-135M q4 on 4 threads, so the activation panel of a chunk does not fall out of cache
at these sizes and no token-panel blocking is needed (a larger default `n_batch` would gain
10–15%; it stays at 64 here).

The numerical contract is the same for every kernel and every token count: exact int32 per
block, then `acc = acc + (d_w · d_x) · float(dot)` (plus `m_w · s_x` for q4_1) in block
order, each operation rounded to float and never fused (WebAssembly has no FMA, and the
scalar, AVX2, AVX-512, NEON and WebAssembly kernels must agree bit for bit; kernels.c is
compiled with `-ffp-contract=off`). A token's logits therefore do not depend on how it was
batched, which speculative decoding relies on, and the same instantiations serve decoding
(one token), small batches and prefill. `tests/c/test_kernels.c` checks every kernel against
the scalar contract for T ∈ {1, 2, 3, 4, 5, 8, 16, 17, 64} on every build, and
`tests/test_engine.py` checks whole-model logits across chunk sizes.

On the benchmark machine (Emerald Rapids, AVX-512 VNNI; shared with other jobs, minimum of
7 runs) the q8_0 kernel runs at 82/110/93 GMAC/s per core for T = 4/16/64 and 24 GMAC/s at
T = 1, against 20/17/18 and 13 for the previous per-row tiles (`bench/bench_kernels.c`,
2048 × 2048); a 64-token prefill of Qwen2.5-0.5B q8_0 on one thread went from 1084 to 388 ms,
with the matrix products now 79% of it instead of 95% and the activation function,
attention and activation quantization the next largest phases (`bench/results/prefill_after.txt`).
The next step on this class of machine would be an AMX tile kernel (`tdpbusd`, 16×16×64 per
instruction, about 5× the VNNI peak per core) for T ≥ 32, keeping the VNNI kernels for
decoding and for machines without AMX; it is not done here. The chunking of `matmul()` in
`forward.c` hands out row ranges that are not multiples of 16, so the partial groups at the
ends of a range are computed in full and stored in part.

### Generation

* **Prefix reuse**: a context remembers which tokens its cache holds; `borg_generate`
  evaluates only what follows the shared prefix (chat turns, repeated system prompts).
* **Speculative decoding**: a draft model with the same tokenizer, or *prompt lookup*
  (continuations of earlier n-gram matches in the context), proposes up to K tokens; the
  target scores them in one batched pass. Greedy output is identical to plain greedy
  decoding; with sampling, the rejection rule of Leviathan et al. (2023) preserves the
  target distribution. The draft length adapts: each proposal is at most one token longer
  than the last accepted run, so text where drafts keep failing costs 2-token passes
  instead of K+1. Prompt lookup needs a match of at least 2 tokens.
* **Batched multi-sequence sampling**: `borg_generate_multi` evaluates a prompt once,
  copies its KV cache to N contexts, and decodes all N continuations in lockstep. Weights
  are read once per step for all sequences, which is what self-consistency voting needs.
* **JSON mode**: a byte-level pushdown recognizer masks tokens that would make the output
  invalid JSON.
* **KV cache files**: `borg_ctx_save` / `borg_ctx_load` persist a processed prompt, keyed
  by a hash of the model header.

### Portability

Plain C11 plus pthreads. Tested on x86-64 (scalar, AVX2, AVX-512 VNNI) and AArch64 (NEON,
dotprod; under QEMU); cross-compiles for Windows with MinGW (MSVC builds run
single-threaded). `make PORTABLE=1` produces a baseline-ISA build; `make THREADS=0`
removes pthreads (for example for WebAssembly).

## Reasoning layer

A small model answers better when it can spend computation on checking itself.
`superborg.reason.solve` implements:

* **direct** / **cot**: one greedy answer, without or with step-by-step reasoning.
* **vote**: self-consistency (Wang et al., 2022): several sampled reasoning paths, majority
  vote on the extracted answers.
* **pal** / **pal_vote**: program-aided reasoning (Gao et al., 2022): the model writes a
  Python program, a sandboxed subprocess runs it, and the printed result is the answer;
  `pal_vote` votes over several programs.
* **verify**: the model checks candidate answers; the engine's log-probabilities score the
  verdicts.
* **rag**: answers grounded in passages retrieved by BM25 from local files, decoded with
  prompt-lookup speculation because answers copy from the context.
* **auto**: routes by question type.

`superborg.agent` runs a tool loop (Python execution, calculator, local search) for
multi-step tasks.
