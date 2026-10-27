# The viewer

The chat page (`superborg/web/index.html`, served by `python -m superborg ui` and packaged as a
static site by `wasm/make_artifact.py`) shows, for every reply word, what the model did inside while
producing it. The events it reads are described in `docs/TRACE.md`. The page is one file with no
external scripts; all engine I/O goes through `window.BORG_TRANSPORT`, so the same page runs
against the HTTP server, the in-browser engine (in a Worker, or on the page's own thread on phones
that cannot start one) and a recorded trace.

## Two views

The right-hand side has a view switch (top right of the panel bar).

**Flow** (the default) tells one word's story on the model's word map:

* **The conversation strip.** Every word of the context is a span: the question, the reply so far,
  the chat template folded behind a chip. Tap a word for its story. Words the selected word *reads
  from* carry an aqua wash whose strength is the effective attention weight (`attn_eff`, the
  head-mean row weighted by each position's value norm). At rest (the output layer) the wash is
  the mean over all blocks, "what did this word read overall"; while scrubbing it is the single
  block's row. Position 0 (the attention sink) is stated in words and never washed; a position
  whose residual spike (`resid.spike`) exceeds 100x *and* stands 5x above the median of the traced
  positions at that layer is marked as sink-like (dotted underline), named in the narration and
  left out of the wash scaling the same way. (The absolute rule alone would flag ordinary words:
  in the recordings many reach 110-150x at layer 1 or 2, where every position spikes.)
* **The field** ("word space"). Every word piece of the vocabulary is a dim dot at its position on
  the vocabulary map (`tools/vocab_map.py`: a UMAP of the standardized, PCA-64 unembedding rows,
  framed to its 0.5..99.5 percentile box). Nothing else is drawn until a word is picked. Then, for
  the current layer, the lens top-3 glow at their map positions (blue), the word the model chose
  glows orange, earlier layers remain as an afterimage, a thin orange ring and label mark the
  chosen word's own place, a hollow ink ring the runner-up's (labelled when its probability is at
  least 3%), and a faint dot stays at every reply word's place (the constellation of the reply so
  far). Glow size is the lens entropy of the layer (`lens[l].entropy / ln V`: a settled readout is
  a point, an undecided one a cloud), brightness the probability; layers below 0.2 L are dimmed
  (early readouts are reflexes). No map position is ever brighter than the current layer's top-1.
  Under the story lies the **pigment**: every reply word deposits its output-row glow (the chosen
  word in orange, the runner-up in blue at half the light) at its own map spot, and each deposit
  keeps 0.975 of its light per later word (a half-life of about 27 words), never per second, so the
  calm field shows where the reply was written and live, replay and screenshots agree. The deposits
  are rebuilt from the reply itself whenever a word arrives (one pass, only then), the light at one
  spot is clamped (a long reply written in one region cannot saturate), and a word whose story was
  skipped by the pacing rule still deposits.
  The header line states the map's honesty: how many pieces, how many sit off-frame on the border,
  and the map's neighbourhood stability across seeds (`stability_knn15`). The region labels are the
  map json's readable landmarks when it carries them (`clusters[i].landmark`: the region's commonest
  whole words in a reference text, the `--label-text` rule of `tools/vocab_map.py`), else the
  region's three commonest pieces joined with middle dots (`stud · bec · includ`), so the reader
  sees that regions are made of word pieces. The nine largest regions are labelled (five when the
  stage is under 420 px wide, as on a phone), each label kept inside the stage by its measured width,
  in muted ink at 60% on paper and 45% in the dark theme (`--region-alpha`). A label beside a mark
  (the chosen word's, the top guess's, the runner-up's) is kept between the header line and the
  caption, and a region label it would cover is hidden while it is there.
* **The scrubber** under the field moves through the model's depth: layer 0 is the word's own
  embedding, the last layer the output. It is a native range input (arrow keys, Home, End). A
  sparkline shows the chosen word's lens probability per layer with the entropy per layer as grey
  bars and an orange tick where the word first takes the lead. Speed buttons (½×, 1×, 3×) set the
  sweep's pace; *Step ›* moves one layer up (and from the output back to layer 0) for a thumb that
  wants single steps; *Walk me through this word* plays the story slowly (~110 ms per layer) with
  captions at layer 0, the lead layer, the midpoint if the word has not taken the lead yet, and the
  output.
* **The story**: narration generated from the fields only, and four cards that are the text twins
  of the drawing: *Reads from*, *Candidates* (the lens top-3 at this layer), *Final choice* (the
  sampling distribution) and *Inner state* (`resid.xyz` of every traced word at this layer, with
  the per-layer basis aligned in sign and order to the previous layer so scrubbing does not flip
  the cloud; nothing without a residual basis). While a story's sweep plays, everything that names
  a layer (the label, the sparkline's cursor, the marks, the wash, the *Reads from*, *Candidates*
  and *Inner state* cards) follows each layer step, and the narration waits dimmed with its layer
  paragraphs withheld until the sweep lands, so the page never names two layers at once. Under
  the cards, a collapsed **Layer by layer** table (layer, lens top-1, p(chosen), rank, entropy;
  the lead layer marked, the current row highlighted) is the full-depth twin of the sparkline; a
  row click scrubs to its layer. The narration also states each layer's MLP write relative to the
  stream (`norms.mlp[k-1] / norms.resid[k]`) and the sink's share of the attention row.

Following live, each new word's story plays as it arrives and the afterimage decays with time
(`exp(-age / 0.5 s)`), so streaming reads as a living field. Words can arrive faster than their
stories play (25 words/s natively), so the queue is paced: with up to two words waiting a story
sweeps in 0.9 s; past two, the sweep takes `max(4, 28 / (waiting - 1))` ms per layer; past six,
the queue is dropped and the newest word plays. A skipped word still deposits its pigment and
stays tappable in the strip. The next story starts in the frame after the previous one lands, so
a landing and a start never share a frame.
Tapping a word or scrubbing turns following off and makes the afterimage a function of the layer
alone, so scrubbing back and forth is deterministic.

**Details** keeps the earlier panels as the overview: the emergence map (layers × reply tokens,
colour = the lens probability of the chosen token), the layer-by-layer readout, the output
distribution, the contribution norms, attention over the context with head entropies, and the
MLP neurons. The two views share the selection: a token picked in either, and the layer scrubbed
in either, is shown by both ("all layers" in Details is the flow's rest at the output).

## First run

Three cards over the field explain the map, a word's story and the layers; the last offers the
guided walk. Dismissing them is remembered in `localStorage` (a convenience: the page works
without it) and the `?` button brings them back.

## Rendering

WebGL2 draws the field as points and the story as instanced gaussian sprites, both additive into
float render targets (`RGBA16F`; `RGBA8` when the browser has no float targets), with a quarter-size
two-pass blur as a light bloom, and the pigment into a half-size float target that is redrawn only
when a deposit is pending (a word arrived, a resize, the theme); one composite pass turns the
accumulated light into "glow on black" (dark theme) or "ink on paper" (light theme), clamping the
pigment by its brightest channel (the hue is kept) and, in the light theme, passing the story's
light through a soft knee (`1.6 · (1 − exp(−0.65 L / 1.6))`) so a guess held for many layers at
one place piles up into a deep wash, not a block. Without WebGL2 (or with `?renderer=2d`) a Canvas2D
path rasterizes the field once and draws the sprites as pre-rendered radial gradients with
`lighter` / `multiply` compositing and a `filter: blur()` bloom where the browser has it. Both
honour `prefers-reduced-motion` (no sweep: a story lands at once with the layer-indexed afterimage;
no drift, no landing pulse; the frame loop idles). When
the engine runs on the page's own thread (`BORG_TRANSPORT.capabilities().engine === "main-thread"`,
or a transport's status hook reporting `decoding`), the loop caps itself at 30 fps, pauses the
field's drift and the bloom and keeps each sprite pass under 200 records while a reply is being
generated. `index.html?perf` makes the renderer badge in the field's header show the frame times
live, for a check on a real phone (SwiftShader numbers bound the main thread's submission cost,
not GPU time).

The animation is driven by `requestAnimationFrame` from a queue of token reports, so it stays
smooth at any decode rate. Per-frame main-thread time is measured with `performance.now()` and
exposed as `window.__superborg.flowStats` (`renderer`, `frames`, `ms` of the last frame, `mean`,
`max`, `p95` over the last 600 frames, `particles` drawn, `pigment` deposits drawn, `points` on
the map, the ingest cost per token, and `phase`, the cost of a layer step, a landing, the queue
and the draw). Measured on the "capital of France" recording (64 reply words, 30 layers, the
49,152-point SmolLM2 map) replayed with follow-live in headless Chromium under SwiftShader, so GPU
time is not in the numbers and frames are few (every frame carries a layer step), on a shared
4-core machine whose 1-minute load was 6–8.5 during every run; three repeats, the minimum of each
statistic reported, no console errors in any run:

| replay | view | renderer | frames | mean ms | p95 ms | max ms | layer step mean / max ms | ingest per word mean / max ms |
|---|---|---|---:|---:|---:|---:|---|---|
| 6 words/s | desktop 1440×900, dpr 1 | webgl2 | 94–97 | 2.32 | 6.0 | 13.4 | 1.64 / 5.0 | 0.70 / 1.8 |
| 6 words/s | phone 390×844, dpr 3 | webgl2 | 243–257 | 1.24 | 4.2 | 9.6 | 1.44 / 4.6 | 0.72 / 2.5 |
| 25 words/s | desktop | webgl2 | 53–67 | 1.54 | 5.4 | 7.8 | 1.41 / 2.5 | 0.59 / 1.7 |
| 25 words/s | phone | webgl2 | 160–179 | 0.63 | 2.9 | 12.0 | 1.32 / 2.0 | 0.65 / 2.0 |
| 6 words/s | desktop | canvas2d | 123–138 | 2.24 | 5.7 | 26.4 | 1.37 / 3.4 | 0.72 / 1.4 |
| 25 words/s | phone | canvas2d | 146–201 | 1.02 | 4.2 | 20.9 | 1.43 / 2.8 | 0.70 / 2.5 |

The means sit well under the 4 ms budget; the frames above it are the layer steps that rebuild
the three per-layer cards (scrubbing every layer by hand costs 0.6–0.7 ms a step on the desktop
and 0.9–1.5 ms on the phone, of which the cards are about a third) and, in Canvas2D, the one-time
raster of the 49k-point field (its 21–26 ms maximum). A frame with nothing to animate costs the
GL submission alone (`phase.draw`: 0.3–0.8 ms mean). Following live at 30 words/s (the same
recording, load 3–5), a story's landing and the next story's start are in separate frames and
the landing frame refreshes the layer once (the step to the last layer and the landing used to
refresh it twice in one frame); attributing each frame's time to its phases then puts the medians
at 1.6–1.7 ms for a layer step, 1.9–2.3 ms for a landing and 1.6–2.3 ms for a start, and the
6–14 % of frames still over 4 ms split about evenly between a GL submission over 2.5 ms
(SwiftShader's tail; its median is 0.4 ms) and a JS phase taking two to five times its median
(the shared machine), not a fixed cost.

## The vocabulary map

`tools/vocab_map.py MODEL.borg` writes `<model stem>.vocab_map.bin` (n × 2 float16) and
`.vocab_map.json` (the parameters, the stability metrics and 24 labelled regions) next to the model.
The UI server serves them at `GET /api/vocab_map.json` and `GET /api/vocab_map.bin` (404 without
them) and reports `"vocab_map": true` in `/api/info`; `wasm/make_artifact.py` copies them into the
site as `vocab_map.json` and `vocab_map.bin` and lists them in `model.json` (`"vocab_map"`), which
the page reads first, so a site without a map makes no request for one. A transport may instead
provide `vocabMap()` returning `{meta, bin}`. Without a map the field
becomes a layer ladder: the model's guesses laid out by rank across and by layer upwards, with
the same size, brightness and hue encodings; everything else is unchanged.

## Replay

`index.html?replay=<url of a recorded .jsonl trace>` replaces the engine with the recording: the
recorded question is sent and its events are streamed as if generated, one reply token every
`1000 / replay_speed` ms (`&replay_speed=6` is the default; `0` plays everything at once). The
file is one JSON object per line: an optional first `{"type": "request", "model", "body": {...}}`
line (the model name and the chat request, as the recording tools write it) followed by the
events of `docs/TRACE.md` (`prompt_token`, `prompt`, `token`, `done`). The URL must be fetchable
from the page (same origin, or CORS-enabled); browsers refuse `file:` URLs, so serve the directory
over HTTP. The model's sizes are unknown to a replay, so the header shows only what the trace
carries. Fields absent from an older recording degrade gracefully: without `lens[].entropy` the
glow's spread comes from the top probability; without `attn_eff` the wash uses `attn_top`, then
raw `attn_rows` (with a note that the sink and punctuation are overstated); without `resid.xyz`
the inner-state card says so.

## Tests

`tests/test_page.py` drives the page in headless Chromium (Playwright for node): the event
contract, the settings, and the flow view (sprites and pigment deposits via `flowStats`, the
no-map ladder, reduced motion, the strip's wrapping at phone width, the scrubber's keys, the
first-run cards, the cards following a sweep while the narration waits, the layer table and the
Step button, readable region landmarks, the follow-live queue policy, `?replay=` of a synthetic
trace, and of a real recording when `SUPERBORG_TRACE=<path>` is set); its `test_vocab_map_routes`
checks the server's `/api/vocab_map.json` and `.bin` routes and the `vocab_map` flag of
`/api/info` with and without the two files. `tests/test_ui.py` covers the server, including
that the page is one file whose only `fetch()` calls are in the transport section.
