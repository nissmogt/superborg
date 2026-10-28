# Trace events

The chat UI (`superborg/ui.py`, `POST /api/chat`) and the browser engine (`wasm/worker.js`,
`{op: "chat"}`) stream one JSON event per generated token with what the model did inside
while producing it. `superborg/interp.py` (`TraceSession`) emits the events natively and
`wasm/trace-session.js` emits the same events from the WebAssembly engine: the same names,
shapes and rounding, so a page can consume either. `tests/test_wasm.py` compares the two
event by event, and `tests/test_interp.py` checks the arithmetic against NumPy.

Everything comes from one traced forward pass per token (`borg_ctx_enable_trace`: the
residual after each layer, each head's attention row, the MLP activations, the layer norms and
the value norms of the cache) plus one batched projection of the per-layer residuals through
the output matrix, summarized inside the engine (`borg_ctx_lens_summary`) so that the
`[n_lens, vocab]` logits never leave C.

## The stream

| event | when |
|-------|------|
| `prefill` | after every prompt chunk but the last, when the uncached prompt takes more than one chunk (`PREFILL_CHUNK` = 128 native, 64 in the browser): `{"done", "total"}` |
| `prompt_token` | with `"trace_prompt": true`, one per traced prompt position (below) |
| `prompt` | once the prompt is evaluated: `{"tokens": [text...], "n_prompt", "n_reused", "prompt_ms"}` |
| `token` | one per generated token: `{"i", "id", "text", "raw", "eos", "decode_ms", "trace_ms"}` plus the report fields below |
| `done` | `{"text", "n_gen", "stop_reason": "eos" \| "stop" \| "length" \| "cancelled", "seconds", "tok_per_s"}` |

`decode_ms` is the forward pass that produced the token's logits and `trace_ms` the time spent
building the report (the lens projection included). Errors are an in-band `{"type": "error",
"message"}` event.

## The report fields

All floats are rounded as noted (`r3`, `r4`, `r5` = 3, 4, 5 decimals); positions and ids are
integers.
`L` is the number of layers, `H` the heads, `KV` the key/value heads, `n` the number of
context positions seen by the traced token (`pos + 1`), `q = pos` the traced position.

### Sampling

| field | contents |
|-------|----------|
| `pos` | the traced position (`prompt_token`: the prompt position, `token`: `n_prompt + i`) |
| `prob` | probability of the chosen token in the sampling distribution (r5) |
| `entropy` | Shannon entropy of the sampling distribution in nats (r4) |
| `alternatives` | the `final_top` (5) most probable tokens, each `[text, p (r4), id]` (distinct byte tokens can share a text, hence the id) |
| `rival` | `[text, p (r4), id]` of the runner-up: the most probable token of the output distribution other than the chosen one (`null` for a one-token vocabulary). Greedy: `alternatives[1]`; sampled: the top token whenever it was not chosen. Its logit difference to the chosen token is what direct logit attribution scores. |

`prob`, `entropy` and `alternatives` describe the sampling distribution (temperature, top-k,
top-p applied) and are absent from `prompt_token` events; `rival` is of the plain softmax.

### Logit lens: `lens`

One entry per lens layer (`lens_layers`: every `lens_stride`-th layer from 0, the embeddings,
to `L`, the model's own output, which is always included): the final norm and the output
projection applied to the residual after that layer.

| field | contents |
|-------|----------|
| `layer` | 0 (embeddings) .. L |
| `top` | the `lens_top` (3) most likely tokens, `[text, p (r4), id]`, logit descending then id ascending |
| `p`, `rank` | probability (r5) and rank (the number of strictly greater logits: 0 = top) of the chosen token (`prompt_token`: of the actual next token, `next`) |
| `entropy` | Shannon entropy of the row's softmax in nats (r4): where in depth the model is still undecided |
| `rival_p`, `rival_rank` | the same for `rival` (`null` when there is no rival) |

The engine computes every row's top list, the logits and ranks of the tracked tokens (the
chosen one and the rival, one call), the log-sum-exp and the entropy in one pass over the
output matrix in double precision, merged in a fixed block order so the numbers are
bit-identical whatever the thread count (`borg_ctx_lens_summary` in `engine/borg.h`). Only the
tokenizer's `n_vocab` rows take part (Qwen2.5's output matrix has rows past the vocabulary).

### Contributions: `norms`

`{"attn": [L], "mlp": [L], "resid": [L + 1]}` (r3): the norm of each layer's attention output,
of its MLP output, and of the residual stream after the embeddings and after each layer.

### Residual stream: `resid`

| field | contents |
|-------|----------|
| `norm` | `[L + 1]` the residual's norm per layer (r3; the same numbers as `norms.resid`) |
| `spike` | `[L + 1]` max \|dim\| / median \|dim\| of the residual (r3): the massive-activation indicator. A few fixed dimensions of pre-norm models are 100-10,000x larger than the rest at the first position and at delimiter tokens (Sun et al. 2024); a spike far above the usual value marks a position that acts as an attention sink. |
| `xyz` | `[L + 1][3]` (r3) the residual projected onto the model's fixed per-layer basis (below), or `null` when none is loaded |

The projection: clip the basis's `clip_dims` to `+-clip_to`, RMS-normalize the vector,
subtract the basis mean, dot with the three components. The basis is fitted once per model by
`tools/trace_basis.py` (see "The residual basis").

### Attention

Positions are context indices (0 = the first token of the context). Every list that ranks
positions skips position 0, the attention sink, which otherwise tops every list without
telling where the layer reads from (HeadVis's reductions skip row 0 and column 0 the same way).

| field | contents |
|-------|----------|
| `attn_rows` | `[L]` the head-mean attention row `mean_h alpha_hj` over all `n` positions, encoded as below |
| `attn_eff` | `[L]` the effective attention row: `mean_h (alpha_hj * \|v_{kv(h),j}\|)` normalized to sum 1 over positions, the same encoding. Weighting each head's row by the value norm of the position (Kobayashi et al. 2020; `\|v_j\|` per kv head is the cheap proxy for `\|alpha f(x_j)\|`) removes the sink and the delimiters, whose values are nearly zero, and ranks the positions a layer actually copies from. |
| `attn_top` | `[L][<= attn_top (8)]` `[pos, w (r4)]` the largest entries of the effective row (`attn_eff`, dequantized values are not used: `w` is the exact effective weight), position 0 excluded, descending |
| `attn_heads` | `[L][H][<= head_top (3)]` `[pos, w (r4)]` per head the largest raw weights, position 0 excluded, descending |
| `head_stats` | per-head scalars, each `[L][H]` (r3), below |
| `attn_entropy` | `[L][H]` entropy of each head's row in nats (r3) |

The row encoding (`encode_row` / `encodeRow`), `{"scale", "sink", "pos", "w"}`:

* `sink` (r4): the row's mass on position 0, kept apart as a float because it is often most
  of the row and a scale that included it would waste the 8 bits on it.
* `scale` (6 decimals): the row's maximum over positions 1..n-1.
* `pos`, `w`: the kept entries, ascending positions and their weights quantized to
  `round(255 * a_j / scale)` (1..255); an entry is `w * scale / 255`. Entries below
  `1 / (2n)` and entries that would quantize to 0 are dropped (over 95% of a row, H2O), so a
  row costs a few dozen pairs. `decode_row` / `decodeRow` give the dense row back, exact at
  position 0 and within half a step (`scale / 510`) elsewhere.

`head_stats` (HeadVis's definitions where it has one, so the live numbers compare with the
head dashboards on Neuronpedia):

| key | per head |
|-----|----------|
| `sink` | the weight on position 0 |
| `entropy_norm` | entropy / log(n): 1 = uniform over the context, 0 = one position (0 when n = 1) |
| `prev` | the weight on position q - 1 (the previous token; 0 when q = 0) |
| `self` | the weight on position q (the token itself) |
| `lookback` | mass on the prompt positions (0..n_prompt-1) / total: how much the head reads from the prompt rather than from what the model wrote (Lookback Lens); 1 while the prompt is being traced |
| `induction` | sum of the weights on j + 1 over every earlier position j whose token equals the query's token (`token[j] == token[q]`): the induction pattern, 0 when the query's token has not occurred before |

### MLP: `mlp_top`

`[L][<= mlp_top (6)]` `[neuron, activation (r3)]`: the hidden units of each layer with the
largest `|act(gate) * up|`, descending by magnitude, with their signed value.

## Prompt tracing: `"trace_prompt": true`

The prompt's positions are normally evaluated in batched chunks, and only the last one is
traced (by the first `token` event). With `trace_prompt` the uncached prompt positions before
the last one are evaluated one per forward pass instead, each followed by a `prompt_token`
event with the report fields of a `token` event without the sampling fields (`prob`,
`entropy`, `alternatives`), plus:

| field | contents |
|-------|----------|
| `pos`, `id` | the prompt position and its token |
| `next` | the token that follows it in the prompt: the token `lens` tracks (`p`, `rank`) and whose runner-up `rival` is |
| `decode_ms`, `trace_ms` | the one-token pass and the report |

At most the last `TRACE_PROMPT_MAX` = 256 uncached positions before the final one are traced;
earlier ones go through the batched chunks (with `prefill` events) as usual. A prompt that
the context already holds (`n_reused`) is not re-evaluated, so a follow-up turn traces only
its new positions.

Cost: a traced position costs a one-token forward pass plus the report, about as much as one
generated token (the `token` rows in the measurements below), instead of its share of a
batched chunk; 256 traced positions are therefore 256 token-times before the first reply token.
`prompt_token` events are a little smaller than `token` events (no sampling fields).

Both engines take the option: the chat body of `POST /api/chat` and of the worker's
`{op: "chat"}` message, `stream_chat(trace_prompt=True)` and `streamChat({tracePrompt: true})`.

## The residual basis

`tools/trace_basis.py model.borg calibration.txt` runs the model with tracing over a
calibration text in chunks of 512 tokens, one token per pass, and collects every position's
residual after each layer, excluding position 0 of each chunk and the chunk's first delimiter
token (period or newline), the positions that carry the massive activations. Per layer it
detects the massive-activation dimensions (any dimension whose |value| somewhere in the corpus
exceeds `clip_ratio` = 100 times the layer's median |value|), clips them to that bound,
RMS-normalizes the vectors, centers them and fits a 3-component PCA. Without the clip the
first component of every layer would be the massive-activation axis and every ordinary token
would collapse onto one line.

The output, `<model stem>.basis.json` next to the model:

```
{"layers": [{"mean": [D], "components": [[D], [D], [D]], "clip_dims": [...], "clip_to": c,
             "explained": [3]}, ...  (L + 1 entries)],
 "n_tokens": N, "model": name, "clip_ratio": 100}
```

`clip_to` is in the residual's own units; `explained` is the fraction of the clipped,
normalized, centered variance each component carries. `superborg/ui.py` serves the file at
`GET /api/basis` (404 without one) and reports `"basis": true` in `/api/info`;
`wasm/make_artifact.py` copies it into the site as `basis.json`, lists it in `model.json`
(`"basis": "basis.json"`), and `worker.js` loads it when present (a file that does not fit the
model is logged and ignored: `xyz` stays `null`). `TraceSession.set_basis` /
`session.setBasis` take the parsed JSON directly.

## Sizes per token

SmolLM2-135M (30 layers, 9 heads, 3 kv heads, dim 576, lens on every layer, basis loaded) at
500 context: JSON without spaces, UTF-8 bytes, means over the 10 `token` events at positions
490..499 of one chat (smollm2-135m-q8_0; the sparse rows and the lens texts vary with the
content).

| field | bytes | what sets the size |
|-------|------:|--------------------|
| `attn_eff` | 18,900 | 30 rows, about 95 kept entries each of 499 positions (the effective row is flatter than the raw one) |
| `attn_rows` | 14,600 | 30 rows, about 70 kept entries each |
| `attn_heads` | 10,900 | 30 x 9 x 3 `[pos, w]` pairs |
| `head_stats` | 9,600 | 6 x 270 scalars |
| `lens` | 4,500 | 31 rows with 3 token texts each |
| `attn_top` | 3,100 | 30 x 8 pairs |
| `mlp_top` | 2,400 | 30 x 6 pairs |
| `attn_entropy` | 1,700 | 270 scalars |
| `resid` | 1,100 | `norm`, `spike` (31 each) and `xyz` (31 x 3) |
| `norms` | 700 | 91 scalars |
| the rest (`alternatives`, `rival`, sampling and timing fields) | 300 | |
| **token event** | **67,700** | |
| `prompt_token` event | 64,200 | the same without the sampling fields |

The two sparse rows and the per-head fields are three quarters of it; they grow with the
context (the rows) and with layers x heads (the per-head fields), the rest is fixed per model.
At a short context (a 40-token prompt) a token event is about 38 KB; before these fields it
was 11 KB at 200 context.

The cost (`trace_ms`, the whole report with the lens projection, SmolLM2-135M at 200 context;
before and after interleaved in two passes, the minimum over 5 runs natively and 3 under node,
on a shared 4-core machine whose load average was 8-11 during the runs, so the spread between
the passes is noise): natively (q8_0, 4 threads, the same kernels before and after) 24-32 ms
before and 35-47 ms after, so the attention and residual summaries add about 10 ms per token
to a lens projection of about 25-30 ms (16 ms for the whole report with the engine's
interleaved kernels, which speed up the projection); under node (q4, one thread) 112-138 ms
before and 88-92 ms after, where the rebuilt engine's faster kernels outweigh the JavaScript
summaries. A `prompt_token` event costs the same as a `token` event minus the sampling.
