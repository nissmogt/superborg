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

  class TraceSession {
    constructor(model, ctx, options = {}) {
      this.model = model;
      this.ctx = ctx;
      // fusedLens: the lens summarized inside the engine (borg_ctx_lens_summary); false reads the
      // full per-layer logits and summarizes them here (the same tokens and ranks, p within 1e-6)
      this.opt = Object.assign({ lensStride: 1, lensTop: 3, attnTop: 8, mlpTop: 6, finalTop: 5, fusedLens: true },
                               options);
      this.cfg = model.config;
      ctx.enableTrace(true);
    }

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
        chat_format: m.tokenizerInfo.chat_format, threads: 1, lens_layers: this.lensLayers(),
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

    /** The logit lens rows of the report (as lens_report in superborg/interp.py): per lens layer
     *  the top tokens as [text, p, id], and the probability and rank of `chosen`. */
    lensReport(chosen) {
      const o = this.opt, m = this.model, n = m.nVocab, k = Math.max(1, Math.min(o.lensTop, n));
      const layers = this.lensLayers();
      if (o.fusedLens) {
        const s = this.ctx.lensSummary(layers, k, chosen, n);
        return layers.map((li, r) => {
          const lse = s.lse[r], top = [];
          for (let i = 0; i < k; i++) {
            const t = s.topIds[r * k + i];
            top.push([m.tokenText(t), r4(Math.exp(s.topLogits[r * k + i] - lse)), t]);
          }
          return { layer: li, top, p: r5(Math.exp(s.chosenLogit[r] - lse)), rank: s.chosenRank[r] };
        });
      }
      // the full per-layer logits, then per row: log-sum-exp for the normalizer, ranks and top
      // tokens from the logits, so no full softmax is materialized
      const V = m.vocabSize, lensLogits = this.ctx.lens(layers);
      return layers.map((li, r) => {
        const z = lensLogits.subarray(r * V, r * V + n);
        const zc = z[chosen];
        let mx = -Infinity, rank = 0;
        for (let i = 0; i < n; i++) { const v = z[i]; if (v > mx) mx = v; if (v > zc) rank++; }
        let s = 0;
        for (let i = 0; i < n; i++) s += Math.exp(z[i] - mx);
        const lse = mx + Math.log(s);
        return { layer: li, top: topK(z, k).map((t) => [m.tokenText(t), r4(Math.exp(z[t] - lse)), t]),
                 p: r5(Math.exp(zc - lse)), rank };
      });
    }

    tokenReport(chosen, probs) {
      const o = this.opt, m = this.model, c = this.cfg;
      const L = c.n_layers, H = c.n_heads, F = c.hidden_dim, D = c.dim;
      const tr = this.ctx.trace();
      const lens = this.lensReport(chosen);
      const attnN = [], mlpN = [], resid = [];
      for (let l = 0; l < L; l++) { attnN.push(r3(tr.norms[l * 3])); mlpN.push(r3(tr.norms[l * 3 + 1])); }
      for (let l = 0; l <= L; l++) {
        let s = 0;
        for (let i = 0; i < D; i++) { const v = tr.resid[l * D + i]; s += v * v; }
        resid.push(r3(Math.sqrt(s)));
      }
      const np = tr.nPos, k = Math.min(o.attnTop, np);
      const attnTop = [], attnEnt = [];
      for (let l = 0; l < L; l++) {
        const mean = new Float64Array(np), ent = [];
        for (let h = 0; h < H; h++) {
          let e = 0;
          const base = (l * H + h) * np;
          for (let j = 0; j < np; j++) {
            const a = tr.attn[base + j];
            mean[j] += a / H;
            if (a > 0) e -= a * Math.log(a);
          }
          ent.push(r3(e));
        }
        attnTop.push(topK(mean, k).map((j) => [j, r4(mean[j])]));
        attnEnt.push(ent);
      }
      const mlpTop = [];
      for (let l = 0; l < L; l++) {
        const row = tr.mlp.subarray(l * F, (l + 1) * F);
        const abs = Float32Array.from(row, Math.abs);
        mlpTop.push(topK(abs, Math.min(o.mlpTop, F)).map((i) => [i, r3(row[i])]));
      }
      let entropy = 0;
      for (let i = 0; i < probs.length; i++) if (probs[i] > 0) entropy -= probs[i] * Math.log(probs[i]);
      return {
        pos: tr.pos, prob: r5(probs[chosen]), entropy: r4(entropy),
        alternatives: topK(probs, o.finalTop).map((t) => [m.tokenText(t), r4(probs[t]), t]),
        lens, norms: { attn: attnN, mlp: mlpN, resid }, attn_top: attnTop, attn_entropy: attnEnt, mlp_top: mlpTop,
      };
    }

    /** Generator of events; `shouldStop()` is polled before each chunk of `prefillChunk` prompt tokens
     *  (a "prefill" event follows every chunk but the last) and between tokens. */
    * streamChat(messages, { maxTokens = 256, temperature = 0, topP = 1, topK: topKN = 0, seed = 0, enableThinking = null,
                             chatTemplateKwargs = null, shouldStop, prefillChunk = 64 } = {}) {
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
      const promptEvent = () => ({ type: "prompt", tokens: prompt.map((t) => m.tokenText(t)), n_prompt: n,
                                   n_reused: reused, prompt_ms: Math.round(now() - t0) });
      let logits, tDecode;  // decode_ms: the forward pass that produced the token's logits
      for (let s = reused; s < n; s += prefillChunk) {
        if (s > reused) yield { type: "prefill", done: s - reused, total };
        if (shouldStop && shouldStop()) {
          yield promptEvent();
          yield { type: "done", text: "", n_gen: 0, stop_reason: "cancelled", seconds: 0, tok_per_s: 0 };
          return;
        }
        const e = Math.min(s + prefillChunk, n);
        if (e < n) { ctx.decode(prompt.slice(s, e), null); continue; }
        if (e - 1 > s) ctx.decode(prompt.slice(s, e - 1), null);
        const tLast = now();  // the last token alone: its pass gives the first reply token's decode_ms
        logits = ctx.decode([prompt[n - 1]], "last");
        tDecode = now() - tLast;
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
        const report = this.tokenReport(tok, probs);
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
    module.exports = { TraceSession, renderChat, stopStrings, chatInfo, topK };
})(typeof self !== "undefined" ? self : globalThis);
