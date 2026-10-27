// Driven by tests/test_wasm.py: one check of the WebAssembly runtime per run, printed as JSON.
//   node wasm_cases.js CASE request.json
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..", "..");
const { Borg } = require(path.join(root, "wasm", "borg-runtime.js"));
const { TraceSession, renderChat, stopStrings, topK, attentionReport, residReport, prepareBasis, encodeRow } =
  require(path.join(root, "wasm", "trace-session.js"));

const create = (wasmPath) => (wasmPath.endsWith(".js")
  ? Borg.create(null, { asm: require(path.resolve(wasmPath)).borgAsmInstantiate, log: () => {} })
  : Borg.create(fs.readFileSync(wasmPath), { log: () => {} }));
const error = (f) => { try { f(); return null; } catch (e) { return `${e.constructor.name}: ${e.message}`; } };

const cases = {
  // a short traced chat, with the heap first pushed past 2 GiB when `high`
  async chat(req) {
    const borg = await create(req.wasm);
    const hold = req.high ? [borg.malloc(1.1e9), borg.malloc(1.1e9)] : [];
    const probe = borg.malloc(16);
    borg.free(probe);
    const model = borg.loadModel(fs.readFileSync(req.model));
    const ctx = model.context({ nCtx: 256 });
    const s = new TraceSession(model, ctx, { lensStride: 1 });
    const events = [...s.streamChat([{ role: "user", content: req.chat }], { maxTokens: req.max_tokens })];
    for (const ev of events) for (const k of ["prompt_ms", "decode_ms", "trace_ms", "seconds", "tok_per_s"]) delete ev[k];
    return { probe, held: hold.length, heap: borg.x.memory.buffer.byteLength, tokens: ctx.tokens(), trace: Array.from(ctx.trace().resid.slice(0, 8)),
             events };
  },

  async load(req) {
    const borg = await create(req.wasm);
    return { error: error(() => borg.loadModel(fs.readFileSync(req.model))) };
  },

  async context(req) {
    const borg = await create(req.wasm);
    const model = borg.loadModel(fs.readFileSync(req.model));
    return { error: error(() => model.context({ nCtx: req.n_ctx })) };
  },

  // the JavaScript build stops growing below 2 GiB: malloc then fails with the runtime's own error
  async malloc(req) {
    const borg = await create(req.wasm);
    return { errors: req.sizes.map((n) => error(() => borg.malloc(n))), heap: borg.x.memory.buffer.byteLength };
  },

  // TraceSession.sample / candidates on given logits (no model: the methods only need nVocab)
  async sample(req) {
    const s = Object.create(TraceSession.prototype);
    const out = {};
    if (req.flat) {  // a flat distribution over n tokens: how often are tokens past the 256 most likely drawn?
      const n = req.flat, logits = new Float32Array(n);
      s.model = { nVocab: n };
      let rng = 1, beyond = 0;
      const rand = () => ((rng = (rng * 48271) % 2147483647) / 2147483647);
      for (let i = 0; i < req.draws; i++) if (s.sample(logits, rand, 1, 1, 0).tok >= 256) beyond++;
      out.beyond = beyond / req.draws;
    }
    if (req.cases) {
      out.cases = req.cases.map(([seed, n, temperature, topP, k]) => {
        let x = seed;
        const rand = () => ((x = (x * 48271) % 2147483647) / 2147483647);
        const p = new Float64Array(n);
        let sum = 0;
        for (let i = 0; i < n; i++) { p[i] = Math.exp((rand() * 12) / temperature); sum += p[i]; }
        for (let i = 0; i < n; i++) p[i] /= sum;
        const cand = s.candidates(p, topP, k);
        return { p: Array.from(p), cand: cand && Array.from(cand) };
      });
    }
    if (req.topk) {  // topK with k = n on a large vocabulary (was O(n k))
      const n = req.topk, v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = Math.sin(i * 12.9898) * 43758.5453 % 1;
      const t = Date.now();
      const got = topK(v, n);
      out.topk_ms = Date.now() - t;
      out.topk_sorted = got.length === n && got.every((j, i) => i === 0 || v[got[i - 1]] >= v[j]);
    }
    return out;
  },

  async render(req) {
    return req.infos.map((info) => ({
      prompts: req.conversations.map((c) => renderChat(info, c)),
      stops: stopStrings(info),
    }));
  },

  // chat template parity with superborg/chat.py: every template x conversation x setting through
  // renderChat (the vendored @huggingface/jinja), the rendered text or the error's message
  async template(req) {
    const out = {};
    for (const [name, info] of Object.entries(req.templates)) {
      out[name] = {};
      for (const [cn, conv] of Object.entries(req.conversations)) {
        req.settings.forEach((s, i) => {
          const extra = Object.assign({}, s.extra || {});
          if (s.tools) extra.tools = req.tools;
          let r;
          try {
            r = { text: renderChat(info, conv, s.add_generation_prompt, "enable_thinking" in s ? s.enable_thinking : null,
                                   extra) };
          } catch (e) { r = { error: e.message }; }
          out[name][`${cn}#${i}`] = r;
        });
      }
    }
    return out;
  },

  // the vendored library's strftime_now (patched to Python's C-locale directives) at given times:
  // Date without arguments is pinned to each time, as the library creates it
  async strftime(req) {
    require(path.join(root, "wasm", "vendor", "hfjinja.js"));
    const tpl = new HFJinja.Template("{{ strftime_now(fmt) }}"), RealDate = Date, out = [];
    for (const ms of req.times) {
      globalThis.Date = class extends RealDate { constructor(...a) { super(...(a.length ? a : [ms])); } };
      try { out.push(req.formats.map((fmt) => tpl.render({ fmt }))); } finally { globalThis.Date = RealDate; }
    }
    return out;
  },

  // the lens summarized in the engine (ctx.lensSummary) against the full logits summarized in
  // JavaScript, for every token of a short chat: unrounded, and as TraceSession's reports
  async lens(req) {
    const borg = await create(req.wasm);
    const model = borg.loadModel(fs.readFileSync(req.model));
    const ctx = model.context({ nCtx: 256 });
    const s = new TraceSession(model, ctx, { lensStride: req.stride || 1, lensTop: req.k });
    const k = req.k, n = model.nVocab, V = model.vocabSize, out = [];
    for (const ev of s.streamChat([{ role: "user", content: req.chat }], { maxTokens: req.max_tokens })) {
      if (ev.type !== "token") continue;
      // the generator is suspended after the yield: the trace still holds this token's pass
      const layers = s.lensLayers(), chosen = ev.id, rival = ev.rival[2];
      const f = ctx.lensSummary(layers, k, [chosen, rival]);
      const z = ctx.lens(layers);
      const old = layers.map((li, r) => {
        const row = z.subarray(r * V, r * V + n), zc = row[chosen], zr = row[rival];
        let mx = -Infinity, rank = 0, rrank = 0;
        for (let i = 0; i < n; i++) { if (row[i] > mx) mx = row[i]; if (row[i] > zc) rank++; if (row[i] > zr) rrank++; }
        let sum = 0;
        for (let i = 0; i < n; i++) sum += Math.exp(row[i] - mx);
        const lse = mx + Math.log(sum), top = topK(row, k);
        let ent = 0;  // -sum p log p over the whole row
        for (let i = 0; i < n; i++) { const p = Math.exp(row[i] - lse); if (p > 0) ent -= p * Math.log(p); }
        return { top, p_top: top.map((t) => Math.exp(row[t] - lse)), p: Math.exp(zc - lse), rank, lse, entropy: ent,
                 rival_p: Math.exp(zr - lse), rival_rank: rrank };
      });
      s.opt.fusedLens = false;
      const plainReport = s.lensReport(chosen, rival);
      s.opt.fusedLens = true;
      out.push({
        id: chosen, rival, layers,
        fused: { top: Array.from(f.topIds), rank: Array.from(f.trackRank).filter((_, i) => i % 2 === 0),
                 rival_rank: Array.from(f.trackRank).filter((_, i) => i % 2 === 1), lse: Array.from(f.lse),
                 entropy: Array.from(f.entropy),
                 p: layers.map((_, r) => Math.exp(f.trackLogit[2 * r] - f.lse[r])),
                 rival_p: layers.map((_, r) => Math.exp(f.trackLogit[2 * r + 1] - f.lse[r])),
                 p_top: Array.from(f.topLogits, (v, i) => Math.exp(v - f.lse[Math.floor(i / k)])) },
        old: { top: old.flatMap((o) => o.top), rank: old.map((o) => o.rank), rival_rank: old.map((o) => o.rival_rank),
               lse: old.map((o) => o.lse), entropy: old.map((o) => o.entropy), p: old.map((o) => o.p),
               rival_p: old.map((o) => o.rival_p), p_top: old.flatMap((o) => o.p_top) },
        report: ev.lens, plain_report: plainReport,
      });
    }
    return { simd: borg.simd, tokens: out };
  },

  // the attention and residual summaries (attentionReport, residReport, encodeRow) on given arrays:
  // tests/test_wasm.py runs superborg/interp.py's functions on the same input and compares
  async report(req) {
    const { L, H, KV, D, n, tokens, n_prompt, attn_top, head_top } = req;
    const out = attentionReport(Float32Array.from(req.attn), Float32Array.from(req.vnorm), L, H, KV, n, tokens, n_prompt,
                                attn_top, head_top);
    out.resid = residReport(Float32Array.from(req.resid), L, D, req.basis ? prepareBasis(req.basis, L, D) : null);
    out.rows = req.rows.map((r) => encodeRow(Float64Array.from(r), r.length));
    return out;
  },

  // per-token milliseconds of decoding, of the old lens path (full logits + JavaScript summary)
  // and of the fused one, for bench/bench_lens.py: per run, the mean over the tokens of one chat
  async lensBench(req) {
    const borg = await create(req.wasm);
    const model = borg.loadModel(fs.readFileSync(req.model));
    const ctx = model.context({ nCtx: 512 });
    const s = new TraceSession(model, ctx, { lensStride: 1 });
    const prompt = model.tokenize(renderChat(model.tokenizerInfo, [{ role: "user", content: req.chat }]), false, true);
    const out = {};
    for (const stride of req.strides) {
      s.opt.lensStride = stride;
      const runs = { decode_ms: [], old_ms: [], fused_ms: [], layers: s.lensLayers().length };
      for (let run = 0; run < req.runs; run++) {
        ctx.truncate(0);
        let logits = ctx.decode(prompt, "last"), t = { decode: 0, old: 0, fused: 0 };
        for (let i = 0; i < req.tokens; i++) {
          let best = 0;
          for (let j = 1; j < model.nVocab; j++) if (logits[j] > logits[best]) best = j;
          const t0 = performance.now();
          s.opt.fusedLens = false;
          s.lensReport(best);
          const t1 = performance.now();
          s.opt.fusedLens = true;
          s.lensReport(best);
          const t2 = performance.now();
          logits = ctx.decode([best], "last");
          t.old += t1 - t0; t.fused += t2 - t1; t.decode += performance.now() - t2;
        }
        runs.decode_ms.push(t.decode / req.tokens);
        runs.old_ms.push(t.old / req.tokens);
        runs.fused_ms.push(t.fused / req.tokens);
      }
      out[stride] = runs;
    }
    return { simd: borg.simd, strides: out };
  },
};

(async () => {
  const [name, reqPath] = process.argv.slice(2);
  const out = await cases[name](JSON.parse(fs.readFileSync(reqPath, "utf8")));
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
