// JavaScript port of superborg/interp.py: chat generation with per-token activation traces.
// Emits the same events as TraceSession.stream_chat ("prompt", "token", "done"), so one UI
// renders either the local Python server or the in-browser WebAssembly engine.

/* global self, globalThis, performance */
(function (root) {
  "use strict";

  function renderChat(info, messages, addGenerationPrompt = true) {
    const fmt = info.chat_format || "plain";
    const tpl = info.chat_template || "";
    const msgs = messages.slice();
    if (tpl.includes("You are a helpful AI assistant named SmolLM") && (!msgs.length || msgs[0].role !== "system"))
      msgs.unshift({ role: "system", content: "You are a helpful AI assistant named SmolLM, trained by Hugging Face" });
    let out = "";
    if (fmt === "chatml" || fmt === "qwen3") {
      for (const m of msgs) out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
      if (addGenerationPrompt) out += "<|im_start|>assistant\n" + (fmt === "qwen3" ? "<think>\n\n</think>\n\n" : "");
    } else if (fmt === "llama3") {
      out += "<|begin_of_text|>";
      for (const m of msgs) out += `<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content.trim()}<|eot_id|>`;
      if (addGenerationPrompt) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (fmt === "zephyr") {
      for (const m of msgs) out += `<|${m.role}|>\n${m.content}</s>\n`;
      if (addGenerationPrompt) out += "<|assistant|>\n";
    } else {
      for (const m of msgs) out += `${m.role}: ${m.content}\n`;
      if (addGenerationPrompt) out += "assistant:";
    }
    return out;
  }

  function stopStrings(info) {
    return { chatml: ["<|im_end|>"], qwen3: ["<|im_end|>"], llama3: ["<|eot_id|>"], zephyr: ["</s>"] }[info.chat_format] || [];
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

  function topK(values, k) {
    // indices of the k largest values (k is small: a simple insertion buffer)
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
      this.opt = Object.assign({ lensStride: 1, lensTop: 3, attnTop: 8, mlpTop: 6, finalTop: 5 }, options);
      this.cfg = model.config;
      ctx.enableTrace(true);
    }

    lensLayers() {
      const n = this.cfg.n_layers, s = Math.max(1, this.opt.lensStride | 0), out = [];
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
        backend: "wasm (" + m.b.simd + ")",
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
      let cand = topK(p, topKN > 0 ? Math.min(topKN, n) : 256);
      if (topP < 1) {
        let cum = 0, cut = cand.length;
        for (let i = 0; i < cand.length; i++) { cum += p[cand[i]]; if (cum >= topP) { cut = i + 1; break; } }
        cand = cand.slice(0, cut);
      }
      let tot = 0;
      for (const i of cand) tot += p[i];
      let r = rand() * tot;
      for (const i of cand) { r -= p[i]; if (r <= 0) return { tok: i, probs }; }
      return { tok: cand[cand.length - 1], probs };
    }

    tokenReport(chosen, probs) {
      const o = this.opt, m = this.model, c = this.cfg;
      const L = c.n_layers, H = c.n_heads, F = c.hidden_dim, D = c.dim, V = m.vocabSize, n = m.nVocab;
      const tr = this.ctx.trace();
      const layers = this.lensLayers();
      const lensLogits = this.ctx.lens(layers);
      // per lens row: log-sum-exp for the normalizer, ranks and top tokens from the logits, so
      // no full softmax is materialized (as lens_report in superborg/interp.py)
      const lens = layers.map((li, r) => {
        const z = lensLogits.subarray(r * V, r * V + n);
        const zc = z[chosen];
        let mx = -Infinity, rank = 0;
        for (let i = 0; i < n; i++) { const v = z[i]; if (v > mx) mx = v; if (v > zc) rank++; }
        let s = 0;
        for (let i = 0; i < n; i++) s += Math.exp(z[i] - mx);
        const lse = mx + Math.log(s);
        return { layer: li, top: topK(z, o.lensTop).map((t) => [m.tokenText(t), r4(Math.exp(z[t] - lse))]),
                 p: r5(Math.exp(zc - lse)), rank };
      });
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
        alternatives: topK(probs, o.finalTop).map((t) => [m.tokenText(t), r4(probs[t])]),
        lens, norms: { attn: attnN, mlp: mlpN, resid }, attn_top: attnTop, attn_entropy: attnEnt, mlp_top: mlpTop,
      };
    }

    /** Generator of events; `shouldStop()` is polled between tokens. */
    * streamChat(messages, { maxTokens = 256, temperature = 0, topP = 1, topK: topKN = 0, seed = 0, shouldStop } = {}) {
      const m = this.model, ctx = this.ctx, info = m.tokenizerInfo;
      const text = renderChat(info, messages);
      const prompt = m.tokenize(text, false, true);
      const room = ctx.nCtx - prompt.length - 1;
      if (room <= 0) throw new Error(`the conversation (${prompt.length} tokens) does not fit the ${ctx.nCtx}-token context`);
      maxTokens = Math.max(1, Math.min(maxTokens | 0, room));
      const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
      let t0 = now();
      const reused = ctx.sync(prompt.slice(0, -1));
      let logits = ctx.decode([prompt[prompt.length - 1]], "last");
      yield { type: "prompt", tokens: prompt.map((t) => m.tokenText(t)), n_prompt: prompt.length, n_reused: reused,
              prompt_ms: Math.round(now() - t0) };
      const rand = mulberry32(seed);
      const dec = new TextDecoder("utf-8");
      const stops = stopStrings(info);
      let out = "", nGen = 0, stop = "length", tDecode = 0;
      const tGen = now();
      for (let i = 0; i < maxTokens; i++) {
        const { tok, probs } = this.sample(logits, rand, temperature, topP, topKN);
        const t1 = now();
        const report = this.tokenReport(tok, probs);
        const traceMs = Math.round(now() - t1);
        if (m.isEog(tok)) {
          stop = "eos";
          yield Object.assign({ type: "token", i, id: tok, text: "", raw: m.tokenText(tok), eos: true, trace_ms: traceMs }, report);
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
  if (typeof module !== "undefined" && module.exports) module.exports = { TraceSession, renderChat };
})(typeof self !== "undefined" ? self : globalThis);
