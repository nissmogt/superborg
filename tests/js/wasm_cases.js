// Driven by tests/test_wasm.py: one check of the WebAssembly runtime per run, printed as JSON.
//   node wasm_cases.js CASE request.json
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..", "..");
const { Borg } = require(path.join(root, "wasm", "borg-runtime.js"));
const { TraceSession, renderChat, stopStrings, topK } = require(path.join(root, "wasm", "trace-session.js"));

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
};

(async () => {
  const [name, reqPath] = process.argv.slice(2);
  const out = await cases[name](JSON.parse(fs.readFileSync(reqPath, "utf8")));
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
