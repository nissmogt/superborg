// JavaScript port of superborg/interp.py: chat generation with per-token activation traces.
// Emits the same events as TraceSession.stream_chat ("prefill", "prompt", "token", "done"), so one
// UI renders either the local Python server or the in-browser WebAssembly engine.

/* global self, globalThis, performance */
(function (root) {
  "use strict";

  // Chat prompts as superborg/chat.py's ChatFormatter renders them: the model's own Jinja template
  // (the .borg header's chat_template) through @huggingface/jinja (wasm/vendor/hfjinja.js, bundled
  // ahead of this file by wasm/make_artifact.py) with the variables transformers' apply_chat_template
  // passes, or the built-in renderers for headers without a template.
  const templates = new Map();  // template string -> parsed HFJinja.Template

  function jinja() {
    if (!root.HFJinja && typeof require === "function") require("./vendor/hfjinja.js");  // Node
    if (!root.HFJinja) throw new Error("chat templates need wasm/vendor/hfjinja.js (python wasm/vendor_jinja.py)");
    return root.HFJinja;
  }

  // A model's tokenizer info plus the text of its special tokens (bos_token, eos_token and
  // eos_tokens, the end-of-turn tokens of eos_ids), which the template and stopStrings need.
  const chatInfos = new WeakMap();
  function chatInfo(model) {
    let info = chatInfos.get(model);
    if (!info) {
      const t = model.tokenizerInfo, text = (i) => (i != null && i >= 0 ? model.tokenText(i) : "");
      const ids = t.eos_ids && t.eos_ids.length ? t.eos_ids : t.eos_id >= 0 ? [t.eos_id] : [];
      info = Object.assign({}, t, { bos_token: text(t.bos_id), eos_token: text(t.eos_id), eos_tokens: ids.map(text) });
      chatInfos.set(model, info);
    }
    return info;
  }

  // info: chatInfo(model), or any object with chat_template / chat_format and the token texts;
  // enableThinking reaches the template only as a boolean (null: the template's own default, as
  // transformers when the caller passes nothing); extra: further template variables (the chat
  // body's chat_template_kwargs: date_string, tools, ...), merged in after the special tokens.
  function renderChat(info, messages, addGenerationPrompt = true, enableThinking = null, extra = null) {
    if (info.chat_template) {
      let tpl = templates.get(info.chat_template);
      if (!tpl) {
        if (templates.size >= 8) templates.clear();
        templates.set(info.chat_template, tpl = new (jinja().Template)(info.chat_template));
      }
      const vars = { messages, add_generation_prompt: !!addGenerationPrompt, tools: null, documents: null };
      // a special token the header lacks stays undefined, never null (Python would print "None")
      if (info.bos_token) vars.bos_token = info.bos_token;
      if (info.eos_token) vars.eos_token = info.eos_token;
      if (extra) for (const k of Object.keys(extra)) if (extra[k] !== undefined) vars[k] = extra[k];
      if (typeof enableThinking === "boolean") vars.enable_thinking = enableThinking;
      return tpl.render(vars);
    }
    const fmt = info.chat_format || "plain";
    let out = "";
    if (fmt === "chatml" || fmt === "qwen3") {
      for (const m of messages) out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
      if (addGenerationPrompt)
        out += "<|im_start|>assistant\n" + (fmt === "qwen3" && !enableThinking ? "<think>\n\n</think>\n\n" : "");
    } else if (fmt === "llama3") {
      out += "<|begin_of_text|>";
      for (const m of messages) out += `<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content.trim()}<|eot_id|>`;
      if (addGenerationPrompt) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (fmt === "zephyr") {
      for (const m of messages) out += `<|${m.role}|>\n${m.content}</s>\n`;
      if (addGenerationPrompt) out += "<|assistant|>\n";
    } else if (fmt === "mistral") {
      let pending = "";  // a system message goes in front of the next user turn
      for (const m of messages) {
        if (m.role === "system") pending = m.content + "\n\n";
        else if (m.role === "user") { out += `[INST] ${pending}${m.content} [/INST]`; pending = ""; }
        else out += `${m.content}</s>`;
      }
    } else {
      for (const m of messages) out += `${m.role}: ${m.content}\n`;
      if (addGenerationPrompt) out += "assistant:";
    }
    return out;
  }

  // The text of the end-of-turn tokens (the header's eos_ids), as ChatFormatter.stop_strings:
  // what a template ends turns with, whatever its format.
  function stopStrings(info) {
    const out = [];
    for (const t of info.eos_tokens || (info.eos_token ? [info.eos_token] : [])) if (t && !out.includes(t)) out.push(t);
    return out;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // softmax of row r of a [rows, stride] array, first n entries; returns {probs, max, sum}
  function softmaxRow(arr, r, stride, n) {
    const off = r * stride;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (arr[off + i] > mx) mx = arr[off + i];
    const p = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { const e = Math.exp(arr[off + i] - mx); p[i] = e; s += e; }
    for (let i = 0; i < n; i++) p[i] /= s;
    return p;
  }

  // indices of the k largest values, largest first (ties: lower index first)
  function topK(values, k) {
    const n = values.length;
    if (k > 4096) {  // a sort of all indices: the insertion buffer below is O(n k) at worst
      const ord = new Uint32Array(n);
      for (let i = 0; i < n; i++) ord[i] = i;
      ord.sort((a, b) => values[b] - values[a] || a - b);
      return Array.from(ord.subarray(0, Math.min(k, n)));
    }
    const idx = [], val = [];
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (idx.length < k || v > val[val.length - 1]) {
        let j = val.length;
        while (j > 0 && val[j - 1] < v) j--;
        idx.splice(j, 0, i); val.splice(j, 0, v);
        if (idx.length > k) { idx.pop(); val.pop(); }
      }
    }
    return idx;
  }

  const r4 = (x) => Math.round(x * 1e4) / 1e4;
  const r5 = (x) => Math.round(x * 1e5) / 1e5;
  const r3 = (x) => Math.round(x * 1e3) / 1e3;

  const TRACE_PROMPT_MAX = 256;  // prompt positions traced one per pass with tracePrompt (interp.TRACE_PROMPT_MAX)

  // ---- attention rows and statistics (the same arithmetic as superborg/interp.py) -------------
  // An attention row over positions 0..n-1 as {scale, sink, pos, w}: the mass on position 0 (the
  // sink) kept apart as a float, the rest quantized to 8 bits relative to the row's maximum
  // excluding position 0 (scale; an entry is w * scale / 255), and sparse: entries below 1/(2n) and
  // entries that would quantize to 0 are dropped. Positions ascending.
  function encodeRow(row, n) {
    const out = { scale: 0, sink: r4(row[0]), pos: [], w: [] };
    if (n < 2) return out;
    let m = 0;
    for (let j = 1; j < n; j++) if (row[j] > m) m = row[j];
    if (m <= 0) return out;
    const thr = 1 / (2 * n);
    for (let j = 1; j < n; j++) {
      const a = row[j], q = Math.floor(a * 255 / m + 0.5);
      if (a >= thr && q >= 1) { out.pos.push(j); out.w.push(q); }
    }
    out.scale = Math.round(m * 1e6) / 1e6;
    return out;
  }

  // the dense row an encodeRow result stands for
  function decodeRow(enc, n) {
    const row = new Float64Array(n);
    row[0] = enc.sink;
    enc.pos.forEach((p, i) => { row[p] = enc.w[i] * enc.scale / 255; });
    return row;
  }

  // The effective attention of each layer, Float64Array [L * n]: mean over heads of
  // alpha_hj * |v_{kv(h), j}|, normalized to sum 1 over positions (Kobayashi et al. 2020).
  function effectiveRows(attn, vnorm, L, H, KV, n) {
    const g = H / KV, out = new Float64Array(L * n);
    for (let l = 0; l < L; l++) {
      const row = out.subarray(l * n, (l + 1) * n);
      for (let h = 0; h < H; h++) {
        const a = (l * H + h) * n, v = (l * KV + Math.floor(h / g)) * n;
        for (let j = 0; j < n; j++) row[j] += attn[a + j] * vnorm[v + j];
      }
      let s = 0;
      for (let j = 0; j < n; j++) { row[j] /= H; s += row[j]; }
      if (s > 0) for (let j = 0; j < n; j++) row[j] /= s;
    }
    return out;
  }

  // Per-head scalars of one query, [L][H] (3 decimals): sink (mass on position 0), entropy_norm
  // (entropy / log(n)), prev (weight on q - 1), self (weight on q), lookback (mass on the prompt
  // positions / total) and induction (the weights on j + 1 over the earlier positions j whose
  // token equals the query's; HeadVis's definition). tokens: the context's ids up to q = n - 1.
  function headStats(attn, L, H, n, tokens, nPrompt) {
    const q = n - 1, logn = n > 1 ? Math.log(n) : 0, np = Math.max(0, Math.min(nPrompt, n));
    const out = { sink: [], entropy_norm: [], prev: [], self: [], lookback: [], induction: [] };
    const match = [];
    for (let j = 0; j < q; j++) if (tokens[j] === tokens[q]) match.push(j + 1);
    for (let l = 0; l < L; l++) {
      const S = { sink: [], entropy_norm: [], prev: [], self: [], lookback: [], induction: [] };
      for (let h = 0; h < H; h++) {
        const b = (l * H + h) * n;
        let tot = 0, e = 0, look = 0, ind = 0;
        for (let j = 0; j < n; j++) {
          const a = attn[b + j];
          tot += a;
          if (a > 0) e -= a * Math.log(a);
          if (j < np) look += a;
        }
        for (const j of match) ind += attn[b + j];
        S.sink.push(r3(attn[b]));
        S.entropy_norm.push(r3(n > 1 ? e / logn : 0));
        S.prev.push(r3(q >= 1 ? attn[b + q - 1] : 0));
        S.self.push(r3(attn[b + q]));
        S.lookback.push(r3(look / (tot > 0 ? tot : 1)));
        S.induction.push(r3(ind));
      }
      for (const k of Object.keys(S)) out[k].push(S[k]);
    }
    return out;
  }

  // The attention fields of a token event (attention_report in superborg/interp.py): attn_rows
  // (the head-mean row per layer), attn_eff (the effective row), attn_top (its largest entries,
  // position 0 excluded, [[pos, w]...]), attn_heads (per head the largest raw weights, position 0
  // excluded), head_stats and attn_entropy ([L][H] nats, 3 decimals).
  function attentionReport(attn, vnorm, L, H, KV, n, tokens, nPrompt, attnTop, headTop) {
    const mean = new Float64Array(n), rows = [], effRows = [], top = [], heads = [], ent = [];
    const eff = effectiveRows(attn, vnorm, L, H, KV, n);
    for (let l = 0; l < L; l++) {
      mean.fill(0);
      for (let h = 0; h < H; h++) {
        const b = (l * H + h) * n;
        for (let j = 0; j < n; j++) mean[j] += attn[b + j];
      }
      for (let j = 0; j < n; j++) mean[j] /= H;
      rows.push(encodeRow(mean, n));
      const er = eff.subarray(l * n, (l + 1) * n);
      effRows.push(encodeRow(er, n));
      top.push(topK(er.subarray(1), attnTop).map((j) => [j + 1, r4(er[j + 1])]));
      const hl = [], el = [];
      for (let h = 0; h < H; h++) {
        const row = attn.subarray((l * H + h) * n, (l * H + h + 1) * n);
        hl.push(topK(row.subarray(1), headTop).map((j) => [j + 1, r4(row[j + 1])]));
        let e = 0;
        for (let j = 0; j < n; j++) if (row[j] > 0) e -= row[j] * Math.log(row[j]);
        el.push(r3(e));
      }
      heads.push(hl);
      ent.push(el);
    }
    return { attn_rows: rows, attn_eff: effRows, attn_top: top, attn_heads: heads,
             head_stats: headStats(attn, L, H, n, tokens, nPrompt), attn_entropy: ent };
  }

  // ---- residual stream ----------------------------------------------------------------------
  // A <model>.basis.json (tools/trace_basis.py) checked against the model: {layers: [{mean,
  // components, clip_dims, clip_to}]}. Throws when the shapes do not fit.
  function prepareBasis(basis, L, D) {
    const layers = basis && basis.layers;
    if (!Array.isArray(layers) || layers.length !== L + 1)
      throw new Error(`the basis has ${Array.isArray(layers) ? layers.length : "no"} layers; the model needs ${L + 1}`);
    for (const b of layers)
      if (!b.mean || b.mean.length !== D || !b.components || b.components.length !== 3 || b.components.some((c) => c.length !== D))
        throw new Error(`the basis is for dim ${b.mean ? b.mean.length : "?"}; the model has ${D}`);
    return { layers: layers.map((b) => ({ mean: b.mean, components: b.components, clip_dims: b.clip_dims || [],
                                          clip_to: b.clip_to == null ? Infinity : b.clip_to })) };
  }

  // Per-layer diagnostics of the residual stream [(L + 1) * D] (resid_report in interp.py): norm,
  // spike (max |dim| / median |dim|) and xyz, the projection onto the basis (massive dims
  // clipped, RMS-normalized, centered, 3 components; null without a basis).
  function residReport(resid, L, D, basis) {
    const norm = [], spike = [], xyz = basis ? [] : null, ab = new Float64Array(D);
    for (let l = 0; l <= L; l++) {
      const r = resid.subarray(l * D, (l + 1) * D);
      let ss = 0, mx = 0;
      for (let i = 0; i < D; i++) {
        const v = r[i], a = Math.abs(v);
        ss += v * v;
        ab[i] = a;
        if (a > mx) mx = a;
      }
      norm.push(r3(Math.sqrt(ss)));
      const sorted = Float64Array.from(ab).sort();
      const med = D % 2 ? sorted[(D - 1) / 2] : (sorted[D / 2 - 1] + sorted[D / 2]) / 2;
      spike.push(r3(mx / (med > 0 ? med : 1)));
      if (basis) {
        const x = Float64Array.from(r), b = basis.layers[l], p = [];
        for (const d of b.clip_dims) x[d] = Math.max(-b.clip_to, Math.min(b.clip_to, x[d]));
        let s2 = 0;
        for (let i = 0; i < D; i++) s2 += x[i] * x[i];
        const rms = Math.sqrt(s2 / D);
        if (rms > 0) for (let i = 0; i < D; i++) x[i] /= rms;
        for (let k = 0; k < 3; k++) {
          const c = b.components[k];
          let s = 0;
          for (let i = 0; i < D; i++) s += c[i] * (x[i] - b.mean[i]);
          p.push(r3(s));
        }
        xyz.push(p);
      }
    }
    return { norm, spike, xyz };
  }

  class TraceSession {
    constructor(model, ctx, options = {}, basis = null) {
      this.model = model;
      this.ctx = ctx;
      // fusedLens: the lens summarized inside the engine (borg_ctx_lens_summary); false reads the
      // full per-layer logits and summarizes them here (the same tokens and ranks, p within 1e-6)
      this.opt = Object.assign({ lensStride: 1, lensTop: 3, attnTop: 8, headTop: 3, mlpTop: 6, finalTop: 5, fusedLens: true },
                               options);
      this.cfg = model.config;
      this.basis = null;
      this.nPrompt = 0;  // prompt positions of the current generation (head_stats' lookback)
      this.setBasis(basis);
      ctx.enableTrace(true);
    }

    /** Use a <model>.basis.json (tools/trace_basis.py) for the "xyz" residual projection, or none. */
    setBasis(basis) { this.basis = basis ? prepareBasis(basis, this.cfg.n_layers, this.cfg.dim) : null; }

    lensLayers(stride = this.opt.lensStride) {
      const n = this.cfg.n_layers, s = Math.max(1, stride | 0), out = [];
      for (let l = 0; l <= n; l += s) out.push(l);
      if (out[out.length - 1] !== n) out.push(n);
      return out;
    }

    info() {
      const m = this.model, c = this.cfg;
      return {
        model: m.name, arch: m.header.arch, quant: m.header.quant, n_layers: c.n_layers, n_heads: c.n_heads,
        n_kv_heads: c.n_kv_heads, dim: c.dim, hidden_dim: c.hidden_dim, vocab: m.vocabSize, n_ctx: this.ctx.nCtx,
        chat_format: m.tokenizerInfo.chat_format, threads: 1, lens_layers: this.lensLayers(), basis: this.basis !== null,
        lens_stride: Math.min(Math.max(1, this.opt.lensStride | 0), c.n_layers), backend: "wasm (" + m.b.simd + ")",
      };
    }

    sample(logits, rand, temperature, topP, topKN) {
      const n = this.model.nVocab;
      const probs = softmaxRow(logits, 0, logits.length, n);
      if (temperature <= 1e-4) {
        let best = 0;
        for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return { tok: best, probs };
      }
      const scaled = new Float32Array(n);
      for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;
      const p = softmaxRow(scaled, 0, n, n);
      const cand = this.candidates(p, topP, topKN);
      if (!cand) {  // the whole distribution: a draw needs no sort
        let r = rand(), last = 0;
        for (let i = 0; i < n; i++) if (p[i] > 0) { last = i; r -= p[i]; if (r < 0) return { tok: i, probs }; }
        return { tok: last, probs };
      }
      let tot = 0;
      for (const i of cand) tot += p[i];
      let r = rand() * tot;
      for (const i of cand) { r -= p[i]; if (r <= 0) return { tok: i, probs }; }
      return { tok: cand[cand.length - 1], probs };
    }

    /** The tokens interp.py's _sample draws from, most probable first: at most top_k of them, cut
     *  where the cumulative probability reaches top_p (null: every token, no truncation). */
    candidates(p, topP, topKN) {
      const n = p.length, k = topKN > 0 ? Math.min(topKN, n) : n;
      if (k === n && topP >= 1) return null;
      let cand = null;
      if (topP < 1) {
        // tokens below t hold at most (1 - top_p) / 2 of the mass together, so the cut falls before
        // them: sort only the rest
        const t = (1 - topP) / (2 * n);
        let mass = 0;
        cand = [];
        for (let i = 0; i < n; i++) if (p[i] >= t) { cand.push(i); mass += p[i]; }
        if (mass >= topP) cand.sort((a, b) => p[b] - p[a] || a - b);
        else cand = null;  // (rounding)
      }
      cand = cand ? cand.slice(0, k) : topK(p, k);
      if (topP < 1) {
        let cum = 0, cut = cand.length;
        for (let i = 0; i < cand.length; i++) { cum += p[cand[i]]; if (cum >= topP) { cut = i + 1; break; } }
        cand = cand.slice(0, cut);
      }
      return cand;
    }

    /** The most probable token of the output distribution other than `chosen` (its rival, as in
     *  a logit difference), or null when the vocabulary has one token. */
    rivalOf(chosen, probs) {
      const n = Math.min(probs.length, this.model.nVocab);
      if (n < 2) return null;
      const t = topK(n < probs.length ? probs.subarray(0, n) : probs, 2);
      return t[0] !== chosen ? t[0] : t[1];
    }

    /** The logit lens rows of the report (as lens_report in superborg/interp.py): per lens layer
     *  the top tokens as [text, p, id], the probability and rank of `chosen`, the row's entropy
     *  (nats) and the probability and rank of `rival`. */
    lensReport(chosen, rival = null) {
      const o = this.opt, m = this.model, n = m.nVocab, k = Math.max(1, Math.min(o.lensTop, n));
      const layers = this.lensLayers(), track = rival === null ? [chosen] : [chosen, rival], nt = track.length;
      if (o.fusedLens) {
        const s = this.ctx.lensSummary(layers, k, track, n);
        return layers.map((li, r) => {
          const lse = s.lse[r], top = [];
          for (let i = 0; i < k; i++) {
            const t = s.topIds[r * k + i];
            top.push([m.tokenText(t), r4(Math.exp(s.topLogits[r * k + i] - lse)), t]);
          }
          return { layer: li, top, p: r5(Math.exp(s.trackLogit[r * nt] - lse)), rank: s.trackRank[r * nt],
                   entropy: r4(s.entropy[r]),
                   rival_p: rival === null ? null : r5(Math.exp(s.trackLogit[r * nt + 1] - lse)),
                   rival_rank: rival === null ? null : s.trackRank[r * nt + 1] };
        });
      }
      // the full per-layer logits, then per row: log-sum-exp for the normalizer, ranks and top
      // tokens from the logits, so no full softmax is materialized; H = log S - E[z - max]
      const V = m.vocabSize, lensLogits = this.ctx.lens(layers);
      return layers.map((li, r) => {
        const z = lensLogits.subarray(r * V, r * V + n);
        const zc = z[chosen], zr = rival === null ? Infinity : z[rival];
        let mx = -Infinity, rank = 0, rrank = 0;
        for (let i = 0; i < n; i++) { const v = z[i]; if (v > mx) mx = v; if (v > zc) rank++; if (v > zr) rrank++; }
        let s = 0, w = 0;
        for (let i = 0; i < n; i++) { const e = Math.exp(z[i] - mx); s += e; w += (z[i] - mx) * e; }
        const lse = mx + Math.log(s);
        return { layer: li, top: topK(z, k).map((t) => [m.tokenText(t), r4(Math.exp(z[t] - lse)), t]),
                 p: r5(Math.exp(zc - lse)), rank, entropy: r4(Math.log(s) - w / s),
                 rival_p: rival === null ? null : r5(Math.exp(zr - lse)), rival_rank: rival === null ? null : rrank };
      });
    }

    /** Everything the trace says about the step whose output distribution is `probs` and whose
     *  token of interest is `chosen` (the sampled token, or the actual next token of a traced
     *  prompt position). nPrompt: prompt positions for head_stats' lookback (null: the current
     *  generation's). sampling=false leaves out prob, entropy and alternatives, as prompt_token
     *  events do. The fields are those of interp.py's token_report (docs/TRACE.md). */
    tokenReport(chosen, probs, nPrompt = null, sampling = true) {
      const o = this.opt, m = this.model, c = this.cfg;
      const L = c.n_layers, H = c.n_heads, KV = c.n_kv_heads, F = c.hidden_dim, D = c.dim;
      const tr = this.ctx.trace();
      const rival = this.rivalOf(chosen, probs);
      const lens = this.lensReport(chosen, rival);
      if (nPrompt === null) nPrompt = this.nPrompt || tr.nPos;
      const rep = { pos: tr.pos };
      if (sampling) {
        let entropy = 0;
        for (let i = 0; i < probs.length; i++) if (probs[i] > 0) entropy -= probs[i] * Math.log(probs[i]);
        rep.prob = r5(probs[chosen]);
        rep.entropy = r4(entropy);
        rep.alternatives = topK(probs, o.finalTop).map((t) => [m.tokenText(t), r4(probs[t]), t]);
      }
      rep.rival = rival === null ? null : [m.tokenText(rival), r4(probs[rival]), rival];
      rep.lens = lens;
      const attnN = [], mlpN = [], resid = [];
      for (let l = 0; l < L; l++) { attnN.push(r3(tr.norms[l * 3])); mlpN.push(r3(tr.norms[l * 3 + 1])); }
      for (let l = 0; l <= L; l++) {
        let s = 0;
        for (let i = 0; i < D; i++) { const v = tr.resid[l * D + i]; s += v * v; }
        resid.push(r3(Math.sqrt(s)));
      }
      rep.norms = { attn: attnN, mlp: mlpN, resid };
      rep.resid = residReport(tr.resid, L, D, this.basis);
      Object.assign(rep, attentionReport(tr.attn, tr.vnorm, L, H, KV, tr.nPos, tr.tokens, nPrompt, o.attnTop, o.headTop));
      const mlpTop = [];
      for (let l = 0; l < L; l++) {
        const row = tr.mlp.subarray(l * F, (l + 1) * F);
        const abs = Float32Array.from(row, Math.abs);
        mlpTop.push(topK(abs, Math.min(o.mlpTop, F)).map((i) => [i, r3(row[i])]));
      }
      rep.mlp_top = mlpTop;
      return rep;
    }

    /** Generator of events; `shouldStop()` is polled before each chunk of `prefillChunk` prompt tokens
     *  (a "prefill" event follows every chunk but the last), before each traced prompt token and
     *  between tokens. tracePrompt: the last TRACE_PROMPT_MAX uncached prompt positions before the
     *  final one are evaluated one per pass, each followed by a "prompt_token" event (the trace
     *  fields of a token event, no sampling fields; "id" the token at "pos", "next" the one after
     *  it, which the lens tracks); each costs a one-token pass plus the trace, as a generated token. */
    * streamChat(messages, { maxTokens = 256, temperature = 0, topP = 1, topK: topKN = 0, seed = 0, enableThinking = null,
                             chatTemplateKwargs = null, tracePrompt = false, shouldStop, prefillChunk = 64 } = {}) {
      const m = this.model, ctx = this.ctx, info = chatInfo(m);
      const text = renderChat(info, messages, true, enableThinking, chatTemplateKwargs);
      const prompt = m.tokenize(text, false, true);
      if (!prompt.length) throw new Error("empty prompt");
      const room = ctx.nCtx - prompt.length - 1;
      if (room <= 0)
        throw Object.assign(new Error(`the conversation (${prompt.length} tokens) does not fit the ${ctx.nCtx}-token context`),
                            { code: "context_length_exceeded" });
      maxTokens = Math.max(1, Math.min(maxTokens | 0, room));
      const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
      const t0 = now();
      // keep the cached prefix, short of the last prompt token (its logits are needed), and evaluate
      // the rest in chunks, as superborg/interp.py does
      const n = prompt.length, cached = ctx.tokens();
      let reused = 0;
      while (reused < cached.length && reused < n - 1 && cached[reused] === prompt[reused]) reused++;
      ctx.truncate(reused);
      const total = n - reused;
      this.nPrompt = n;
      // prompt positions evaluated one per pass and traced: [firstTraced, n - 1)
      const firstTraced = tracePrompt ? Math.max(reused, n - 1 - TRACE_PROMPT_MAX) : n - 1;
      const promptEvent = () => ({ type: "prompt", tokens: prompt.map((t) => m.tokenText(t)), n_prompt: n,
                                   n_reused: reused, prompt_ms: Math.round(now() - t0) });
      const cancelled = () => ({ type: "done", text: "", n_gen: 0, stop_reason: "cancelled", seconds: 0, tok_per_s: 0 });
      let logits, tDecode;  // decode_ms: the forward pass that produced the token's logits
      for (let s = reused; s < n; s += prefillChunk) {
        if (s > reused) yield { type: "prefill", done: s - reused, total };
        if (shouldStop && shouldStop()) { yield promptEvent(); yield cancelled(); return; }
        const e = Math.min(s + prefillChunk, n);
        const last = e === n ? e - 1 : e;  // the final prompt token goes alone (its logits are needed)
        const plainEnd = Math.min(last, Math.max(firstTraced, s));
        if (plainEnd > s) ctx.decode(prompt.slice(s, plainEnd), null);
        for (let q = Math.max(plainEnd, s); q < last; q++) {  // traced prompt positions, one per pass
          if (shouldStop && shouldStop()) { yield promptEvent(); yield cancelled(); return; }
          const t1 = now();
          const lg = ctx.decode([prompt[q]], "last");
          const t2 = now();
          const report = this.tokenReport(prompt[q + 1], softmaxRow(lg, 0, lg.length, m.nVocab), n, false);
          yield Object.assign({ type: "prompt_token", pos: q, id: prompt[q], next: prompt[q + 1],
                                decode_ms: Math.round(t2 - t1), trace_ms: Math.round(now() - t2) }, report);
        }
        if (e === n) {
          const tLast = now();  // the last token alone: its pass gives the first reply token's decode_ms
          logits = ctx.decode([prompt[n - 1]], "last");
          tDecode = now() - tLast;
        }
      }
      yield promptEvent();
      const rand = mulberry32(seed);
      const dec = new TextDecoder("utf-8");
      const stops = stopStrings(info);
      let out = "", nGen = 0, stop = "length";
      const tGen = now();
      for (let i = 0; i < maxTokens; i++) {
        const { tok, probs } = this.sample(logits, rand, temperature, topP, topKN);
        const t1 = now();
        const report = this.tokenReport(tok, probs, n);
        const traceMs = Math.round(now() - t1);
        if (m.isEog(tok)) {
          stop = "eos";
          yield Object.assign({ type: "token", i, id: tok, text: "", raw: m.tokenText(tok), eos: true,
                                decode_ms: Math.round(tDecode), trace_ms: traceMs }, report);
          break;
        }
        const piece = dec.decode(m.tokenBytes(tok), { stream: true });
        out += piece;
        nGen++;
        yield Object.assign({ type: "token", i, id: tok, text: piece, raw: m.tokenText(tok), eos: false,
                              decode_ms: Math.round(tDecode), trace_ms: traceMs }, report);
        if (stops.some((s) => out.includes(s))) { stop = "stop"; break; }
        if (shouldStop && shouldStop()) { stop = "cancelled"; break; }
        if (i === maxTokens - 1) break;
        const t2 = now();
        logits = ctx.decode([tok], "last");
        tDecode = now() - t2;
      }
      for (const s of stops) out = out.split(s)[0];
      const secs = (now() - tGen) / 1000;
      yield { type: "done", text: out, n_gen: nGen, stop_reason: stop, seconds: Math.round(secs * 100) / 100,
              tok_per_s: secs > 0 ? Math.round((nGen / secs) * 100) / 100 : 0 };
    }
  }

  root.TraceSession = TraceSession;
  root.renderChat = renderChat;
  if (typeof module !== "undefined" && module.exports)
    module.exports = { TraceSession, renderChat, stopStrings, chatInfo, topK, encodeRow, decodeRow, effectiveRows,
                       headStats, attentionReport, prepareBasis, residReport };
})(typeof self !== "undefined" ? self : globalThis);
