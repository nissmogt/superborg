// Driven by tests/test_wasm.py: load a model in the WebAssembly engine and print results as JSON.
//   node wasm_check.js borg.wasm|borg-asm.js model.borg request.json
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..", "..");
const { Borg } = require(path.join(root, "wasm", "borg-runtime.js"));
const { TraceSession } = require(path.join(root, "wasm", "trace-session.js"));

(async () => {
  const [wasmPath, modelPath, reqPath] = process.argv.slice(2);
  const req = JSON.parse(fs.readFileSync(reqPath, "utf8"));
  // a .js path is the wasm2js build (wasm/build-js.sh)
  const borg = wasmPath.endsWith(".js")
    ? await Borg.create(null, { asm: require(path.resolve(wasmPath)).borgAsmInstantiate, log: () => {} })
    : await Borg.create(fs.readFileSync(wasmPath), { log: () => {} });
  // load in 3 parts, as the browser worker does
  const file = fs.readFileSync(modelPath);
  const third = Math.floor(file.length / 3);
  const model = borg.loadModel([file.subarray(0, third), file.subarray(third, 2 * third), file.subarray(2 * third)]);
  const out = { simd: borg.simd, tokens: req.texts.map((t) => model.tokenize(t, false, true)) };
  const ctx = model.context({ nCtx: 256 });
  const logits = ctx.decode(model.tokenize(req.prompt, false, true), "last");
  out.logits = Array.from(logits.subarray(0, model.nVocab));
  ctx.truncate(0);
  const s = new TraceSession(model, ctx, { lensStride: 1 });
  out.events = [];
  for (const ev of s.streamChat([{ role: "user", content: req.chat }], { maxTokens: req.max_tokens }))
    out.events.push(ev);
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
