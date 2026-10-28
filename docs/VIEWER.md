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
  block's row. Position 0 is always the attention sink: stated in words, never washed. Any other
  traced position gets the dotted underline (and is named in the narration and left out of the
  wash scaling the same way) when, at some layer from the third on (index 2), its residual spike
  (`resid.spike`: the largest |dimension| over the median |dimension|) is above 100x *and* above
  5x the median spike of the traced positions at that layer (`SINK_SPIKE`, `SINK_RATIO`; with
  fewer than four traced positions, above 300x instead, since a median of so few says little). The
  absolute rule alone would flag ordinary words: in the recordings many reach 110-150x at layer 1
  or 2, where every position spikes; the positions that act as sinks sit 14-50x above the rest for
  a run of layers. The flags are recomputed as more positions arrive, since the medians move.
* **The field** ("word space"). Every word piece of the vocabulary is a dim dot at its position on
  the vocabulary map (`tools/vocab_map.py`, below: clusters of pieces that are near each other in
  the model's output space, laid out as discs so that related clusters are near each other too,
  then refined by a t-SNE of every piece started from those discs, which keeps their arrangement
  but moves each piece next to its own nearest pieces; framed to its 0.5..99.5 percentile box).
  Nothing else is drawn until a word is picked. Then, for
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
  map json's readable landmarks when it carries them (`clusters[i].landmark`, written by
  `tools/vocab_map.py`: up to three whole words of the region, `water · light · energy`, the
  commonest by token id among those in the region's inner half around its density peak, or, for a
  region without words, the kind of piece it is made of: `fragments`, `punctuation`, `numbers`,
  `code`), else the region's three commonest pieces joined with middle dots (`stud · bec · includ`),
  so the reader sees that regions are made of word pieces. The nine largest regions are labelled (five when the
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
`.vocab_map.json` (the parameters, the metrics, 24 labelled regions and the fine clusters) next to
the model.

**What the layout means.** The unembedding rows are standardized per dimension and reduced to 64
dimensions by PCA, then scaled to unit length, so "near" means cosine in that space (where the
neighbourhoods are semantic: the five nearest to ` Paris` are ` London`, ` Berlin`, ` Boston`,
`Paris`, ` Philadelphia`; to ` dog`: ` cat`, ` post`, ` Cat`, ` pet`). The map is hierarchical
(`--method hierarchical`, the default; `"method"` in the json):

* **Islands are clusters.** The unit rows are grouped by MiniBatchKMeans into n / 120 fine
  clusters (at least 200, at most 1200: 409 for SmolLM2, 1,199 for Qwen2.5-0.5B, one of its 1,200
  coming out empty), each drawn as a disc whose area is proportional to its size. Where a disc
  sits is a UMAP of the cluster centroids (cosine, 15 neighbours, min_dist 0.3), so neighbouring
  islands hold related clusters; discs that overlapped are pushed apart by pairwise repulsion
  until no pair overlaps by more than 1%, or for 400 iterations (the pushes are local, the global
  arrangement is kept: SmolLM2's centroids keep 0.86 of their 15 nearest centroids through the
  packing). The discs are listed in the json as `fine_clusters` (`x`, `y`, `r` in map units, `n`,
  and a `landmark` by the rule below).
* **Position within an island** is the cluster's own local layout: the two principal axes of its
  members' unit vectors (`--within pca`) or, the default, a small t-SNE of the members started
  from those axes (`--within tsne`: exact gradients, perplexity min(30, (size − 1) / 3)), centred,
  turned so that the directions of the neighbouring clusters in the cluster's principal plane
  match their directions on the map (the side of an island facing a neighbour holds the members
  nearest to it), and scaled so the members' 95th-percentile radius is the disc radius (the outer
  5% are squeezed into a 10% rim; duplicate rows get a tiny jitter).
* **Then a refinement** (`--refine`, 1000 iterations by default; 0 keeps the discs as placed).
  One t-SNE of every piece, started from the placed discs scaled down to scikit-learn's own
  initial spread, with its early exaggeration (`--refine-start small`, `--refine-exaggeration
  12`, the defaults; Barnes-Hut, perplexity 30; each piece attracted by its exact 92 nearest
  pieces by cosine, found by an exact chunked search). The refinement moves every piece towards
  its nearest pieces wherever they fell, so the result is a t-SNE of the whole vocabulary whose
  arrangement comes from the islands: the cluster centres keep 0.36 (SmolLM2) / 0.27 (Qwen)
  of their 15 nearest centres through it and the passes below (`refine_centroid_knn15`), but
  the islands themselves dissolve into the field: only 12% (SmolLM2) / 11% (Qwen) of the pieces
  end nearest their own cluster's centre (`in_own_disc`). The json's `fine_clusters` are the
  clusters as finally laid out: each cluster's mean position and 95th-percentile radius, which
  overlap.
  The neighbour graph's hubs are capped first (`--hub-cap 30`): a hub is a piece among the 15
  nearest of far more pieces than a plane can put it next to (on Qwen `和` is among the 15
  nearest of 405 others, ` one` of 357, Thai and CJK pieces of 300; on SmolLM2 the worst sits in
  485 lists; a point of a 2-D layout is in about 15), and a t-SNE that tries to honour all those
  attractions pulls hubs in between the frequent pieces: without the cap Qwen's `1` and `2` had
  `评价`, `いただ` and `高频` between them. With the cap a hub keeps only its 30 nearest
  followers (the other entries are set to the largest cosine distance, 2, so their weight in the
  perplexity-calibrated neighbourhoods is nil): the digits sit together again, and the 15-NN kept
  is unchanged (0.193 against 0.193 on SmolLM2, 0.209 against 0.208 on Qwen, measured on the same
  placement). `--refine-start placed` instead keeps the placement at its own scale (standardized
  to a spread of 30, about that of a t-SNE of this many points) with `--refine-exaggeration 1`,
  so the islands hold and each piece moves towards its nearest pieces from where its island put
  it: 0.179 / 0.180 (without / with the hub cap) of the 15-NN kept on SmolLM2 and 0.198 / 0.199
  on Qwen, a little less than the small start, with 10 of the 12 probes semantic on SmolLM2
  (against 9) and the same 7 on Qwen; the small start is the default for the higher number.
* **Then neighbour passes** (`--passes`, at most 12 by default; 0 for none), a local search for
  the very number `knn15_preserved` measures. In each pass every piece, in a random order, is
  offered spots beside each of its 20 nearest pieces in the 64-D space (three close to the piece,
  at 0.35, 0.8 and 1.25 of its own nearest distance, two inside its neighbourhood, at 0.3 and 0.7
  of its 15th-nearest distance) and eight spots around its own position, all rounded to float16,
  the coordinates the `.bin` holds; it takes the best spot when the move raises the whole
  vocabulary's 15-NN overlap, counted exactly: its own 15 nearest at the spot, the lists it drops
  out of (each loses it and gains its 16th) and the lists it enters (each gains it and loses its
  15th). After the first pass only the pieces a move touched are scored again, and the passes
  end when one gains less than 0.0002. The moves are short, a piece goes to where its nearest
  pieces already are, so the field keeps its arrangement; the first pass moves two fifths of the
  pieces and lifts the 15-NN kept from 0.19 to 0.24 on SmolLM2 (0.20 to 0.24 on Qwen), the next
  ones to 0.25 (0.27), where a piece's best spot beside any of its nearest pieces, side effects
  ignored, would keep 0.27-0.28: the t-SNE's arrangement is the ceiling, not the search (only a
  third of a piece's 15 nearest lie among its 100 nearest on the map; a t-SNE at perplexity 8
  instead of 30 gave 0.20 before the passes and 0.244 after them against 0.242 with the same
  passes, and no hub cap the same numbers, so neither was changed). A pass that moved every piece
  by its own count alone, ignoring the lists it displaces, lowered the number (0.19 to 0.18) and
  piled pieces onto the hubs, which is why the exact count is used.
* **Regions** (the 24 `clusters` of the json, the labels on the field) are unions of whole fine
  clusters: a KMeans of the final disc centres (the clusters' means after the passes) weighted by
  their sizes.

The old flat layout (one UMAP of every row, cosine, 30 neighbours, min_dist 0.1) stays behind
`--method flat`; its neighbourhoods were not semantic (` Paris` next to ` directory`, ` needle`,
` Georgia`), which is why the map became hierarchical.

**Metrics** (in the json, printed by the tool; the measurements below are from the shipped maps,
run with `--n-jobs 1` on a shared 4-core machine):

* `knn15_preserved`: for a random sample of 12,000 tokens (`knn_sample`, seed 0), the mean overlap
  of each one's 15 nearest tokens on the map with its exact 15 nearest by cosine in the 64-D space,
  both neighbourhoods taken among the whole vocabulary. `knn15_preserved_within_sample` is the
  same overlap with both neighbourhoods taken among the sample alone (looser: a sample's
  neighbours are farther apart). Both are measured on the float16 coordinates the `.bin` holds
  (the probes too), and the json also carries the overlap over every token rather than a sample,
  before and after the passes (`knn15_all_before_passes`, `knn15_all_after_passes`). The ceiling
  is low because the 64-D neighbourhoods are intrinsically high-dimensional: a token's exact 15
  nearest share its fine cluster only 0.27 of the time on SmolLM2 (0.44 with 50 clusters, 0.24
  with 800), only 0.26 of its 15-NN relations are mutual (a 2-D layout's are nearly all mutual),
  and a t-SNE of the whole vocabulary plateaus at 0.20 whatever its settings: 0.196-0.202 on
  SmolLM2 with perplexity 8, 15 or 30, 1000 or 2000 iterations, started from the t-SNE or the
  plain-PCA placement (a UMAP refinement, 15 neighbours, min_dist 0, reaches 0.07; a dedicated
  t-SNE of the 12,000 sampled tokens alone 0.20 within the sample). The refinement lifts the map
  from the placement's 0.14 to that plateau, and the neighbour passes from the plateau to 0.25
  on SmolLM2 (0.255 on the sample, 0.253 and 0.252 with two other sample seeds, 0.252 over all
  tokens) and 0.27 on Qwen; on SmolLM2 the passes' own ceiling is near: a token's best spot
  beside any of its 15 (or 15 × 15) nearest, side effects ignored, would keep 0.275 (0.28).
* `stability_knn15`: the mean 15-NN overlap of the token coordinates across the four centroid
  seeds, measured on the placements before the refinement and the passes (which run once, on
  the kept seed's placement, so the final coordinates' stability across seeds is not what this
  number measures; the Flow view's heading therefore shows `knn15_preserved`, not this);
  `centroid_stability_knn15` is the same on the centroid layouts, which is where the seed is
  picked: the one most central by that overlap). High for the hierarchical map because a seed
  only moves the islands, not the members within them.

| map | method | knn15_preserved (within sample) | stability_knn15 (centroids) | time |
|---|---|---:|---:|---:|
| SmolLM2-135M (49,152 × 576) | flat UMAP (before) | 0.039 (0.068) | 0.10 | |
| SmolLM2-135M | hierarchical, PCA within | 0.082 | | |
| SmolLM2-135M | hierarchical, t-SNE within, `--refine 0` | 0.140 (0.134) | 0.98 (0.75) | about 4 min |
| SmolLM2-135M | refined, `--hub-cap 0` | 0.193 (0.145) | 0.98 (0.76) | about 13 min |
| SmolLM2-135M | refined, `--refine-start placed`, exaggeration 1, hub cap 0 / 30 | 0.179 (0.139) / 0.180 (0.137) | | |
| SmolLM2-135M | refined, hub cap 30 | 0.193 (0.145) | 0.98 (0.76) | about 13 min |
| SmolLM2-135M | refined, hub cap 30, 9 neighbour passes (shipped) | 0.255 (0.146); 0.252 over all tokens | 0.98 (0.76) | about 24 min |
| Qwen2.5-0.5B (151,665 × 896) | flat UMAP (before) | 0.029 | 0.05 | |
| Qwen2.5-0.5B | hierarchical, t-SNE within, `--refine 0` | 0.152 (0.127) | 0.98 (0.57) | about 16 min |
| Qwen2.5-0.5B | refined, `--hub-cap 0` | 0.208 (0.145) | 0.98 (0.63) | about 60 min |
| Qwen2.5-0.5B | refined, `--refine-start placed`, exaggeration 1, hub cap 0 / 30 | 0.198 (0.144) / 0.199 (0.145) | | |
| Qwen2.5-0.5B | refined, hub cap 30 | 0.210 (0.146) | 0.98 (0.63) | about 65 min |
| Qwen2.5-0.5B | refined, hub cap 30, 11 neighbour passes (shipped) | 0.281 (0.137); 0.274 over all tokens | 0.98 (0.63) | about 2 h |

(The rows without a time are the tool's refinement run by itself on the placement of the
`--refine 0` row, from a cache of the exact neighbours. The shipped maps: peak RSS 0.9 GB for
SmolLM2 and 1.8 GB for Qwen, both with `--n-jobs 1`, on a shared 4-core machine at load 3-7, the
two running at once: of SmolLM2's 24 minutes 8 are the refinement, 10 the 9 neighbour passes (the
first three 100 s each, the later ones shorter, since fewer pieces need scoring) and 3.5 the 409
local t-SNEs; of Qwen's 117 minutes 55 are the refinement (3 of them its exact neighbour search),
40 the 11 passes (the first three 340 s each) and 11 the 1,199 local t-SNEs. Before
the refinement, Qwen's discs stopped at the 400-iteration cap with 10% of the target distance
still overlapping, inside the 15% margin, so no two discs touched; SmolLM2's settled below 1% in
377 iterations. Two runs of the tool under the same conditions gave the same map, but runs under
different machine load have not: the fine clusters depend on the last digits of the PCA, which
differ with the BLAS threading, so the regions and the probe neighbourhoods below can vary from
run to run while `knn15_preserved` stays within 0.01.)

Probe neighbourhoods on the shipped SmolLM2 map (the tool prints them; the five nearest on the map):
* `' Paris'`: ' Berlin', ' Rome', 'London', 'Paris', ' London'
* `' France'`: ' Japan', ' India', ' Russia', ' China', ' Germany'
* `' city'`: ' towns', ' town', ' cities', ' City', ' hex'
* `' dog'`: ' Cat', ' Cow', ' Post', ' cat', ' cow'
* `' cat'`: ' Post', ' Dog', 'Dog', ' Cat', ' dog'
* `'1'`: '4', '3', '6', '8', '5'
* `'2'`: '-', '0', 'y', '3', '1'
* `' def'`: ' reign', ' prop', ' dist', ' adj', ' ast'
* `' return'`: 'Sus', ' returned', ' returning', ' returns', ' Returning'
* `'.'`: '\n\n', ';', ',', ':', "'s"
* `','`: ';', '.', ' authority', "'s", '\n\n'
* `' the'`: ' that', ' here', ' is', ' being', ' an'

11 of the 12 have three or more plainly related neighbours (` def` has none; before the passes the
refined map had 9: ` city` sat by ` tree`, ` table`, ` file` and `,` by ` he`, ` my`, ` own`). The
refinement and the passes follow the 64-D neighbourhoods, for better and worse: before the
refinement `.` and `,` sat among punctuation (`.,`, `;`, `!`), which is where a reader expects
them, their nearest pieces in the model's space are `,`, `:`, ` and`, ` to`, ` on`, and the shipped
map has them by `;`, `:`, `'s` and `\n\n`.

On the shipped Qwen map ` Paris` sits by ` Houston`, ` London`, ` Washington`, ` husband`,
` Texas`, ` France` by ` Germany`, ` Switzerland`, ` China`, ` Spain`, ` Canada`, ` cat` by `猫`,
`Cat`, `_cat`, ` CAT`, `.cat`, `1` by `0`, `4`, `7`, `9`, `3`, `2` by `3`, `5`, `6`, `A`, `a`,
` return` by `.return`, ` returns`, ` blow`, `Returns`, `_returns`, `.` by `:`, `)`, `(`, `?`, `/`,
`,` by `;`, `!`, `(`, `?`, `.` and ` the` by ` one`, ` it`, ` this`, ` them`, ` now`, but ` dog` by
` gun`, ` Moon`, ` cow`, `받`, ` moon`, ` city` by ` line`, ` path`, ` group`, ` version`,
` constant` and ` def` by ` mis`, ` §`, ` pre`, ` des`, ` sub`: 9 of the 12 probes have three or
more plainly related neighbours (8 without the function-word reading of ` the`; 11 on SmolLM2,
all but ` def`). Before the passes 7 passed on Qwen (` cat` sat by ` pred`, ` бес`, ` pod` and
` return` by ` setup`, ` finite`, ` called`). Qwen's 64-D neighbourhoods are weaker to begin with
(` dog`: ` Dog`, ` bug`, ` dogs`, ` gun`; ` cat`: ` Cat`, ` car`, ` ac`; ` city`: ` City`, ` time`,
` first`; ` def`: ` f`, ` de`, ` v`), which bounds any layout of that space, and its 151k
vocabulary is full of hubs (above); without the hub cap the digits had `评价`, `いただ` and `高频`
between them and only 5 probes passed, before the refinement (`--refine 0`) 6-7 did on Qwen and 10
on SmolLM2, for 0.15 and 0.14 of the exact 15-NN kept instead of 0.21 and 0.19.

Each region carries a `label` (its three lowest-id printable pieces, kept for older pages) and a
`landmark`: whole words are the word-initial, alphabetic pieces of 3-12 letters that are not stop
words, that stand alone rather than being the start of the vocabulary's longer pieces (a fragment
such as ` includ` is outweighed by ` include`, ` included`, ` includes` and ` including`, measured on
the ids as a frequency proxy, since BPE merges are ordered by frequency) and that the vocabulary
holds in both cases (` water` and ` Water`); a region whose words are fewer than one piece in twenty
is named by its kind instead (SmolLM2's regions of fragments have up to 5% "words", mostly fragments
the gates let through; its regions of words 11-61%). `--relabel` recomputes the regions and
landmarks of an existing map from its `.bin` in a few seconds, without the layout (on a hierarchical
map each token is counted with the disc whose centre is nearest in units of that disc's radius).
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
trace, and of a real recording: `tests/fixtures/qwen2.5-0.5b-q8_0-insects.jsonl`, a complete
four-token recording with every token's trace, or the one `SUPERBORG_TRACE=<path>` names in its
place; without either the case is a pytest skip); its `test_vocab_map_routes`
checks the server's `/api/vocab_map.json` and `.bin` routes and the `vocab_map` flag of
`/api/info` with and without the two files. `tests/test_ui.py` covers the server, including
that the page is one file whose only `fetch()` calls are in the transport section.
