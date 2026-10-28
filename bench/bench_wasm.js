// Prefill and decode throughput of the WebAssembly build under Node, best of several runs.
//   node bench/bench_wasm.js build/borg.wasm MODEL.borg [PROMPT=64] [GEN=16] [REPS=3]
const fs = require("fs");
const path = require("path");
const { Borg } = require(path.join(__dirname, "..", "wasm", "borg-runtime.js"));

(async () => {
  const [wasmPath, modelPath] = process.argv.slice(2);
  const P = parseInt(process.argv[4] || "64", 10), G = parseInt(process.argv[5] || "16", 10);
  const reps = parseInt(process.argv[6] || "3", 10);
  const borg = await (wasmPath.endsWith(".js")
    ? Borg.create(null, { asm: require(path.resolve(wasmPath)).borgAsmInstantiate, log: () => {} })
    : Borg.create(fs.readFileSync(wasmPath), { log: () => {} }));
  const t0 = performance.now();
  const model = borg.loadModel(fs.readFileSync(modelPath));
  const loadMs = performance.now() - t0;
  const toks = Array.from({ length: P }, (_, i) => (i * 7919 + 13) % model.nVocab);
  let bestPrefill = Infinity, bestDecode = Infinity;
  for (let r = 0; r < reps; r++) {
    const ctx = model.context({ nCtx: P + G + 8, kvF16: false });
    ctx.decode(toks.slice(0, 8), "last"); // warm up
    ctx.sync([]);
    let t = performance.now();
    let logits = ctx.decode(toks, "last");
    bestPrefill = Math.min(bestPrefill, performance.now() - t);
    t = performance.now();
    for (let i = 0; i < G; i++) {
      let best = 0;
      for (let j = 1; j < model.nVocab; j++) if (logits[j] > logits[best]) best = j;
      logits = ctx.decode([best], "last");
    }
    bestDecode = Math.min(bestDecode, performance.now() - t);
  }
  console.log(`${path.basename(wasmPath)} ${path.basename(modelPath)}: load ${loadMs.toFixed(0)} ms, ` +
    `prefill ${P} tokens ${bestPrefill.toFixed(0)} ms = ${(P * 1000 / bestPrefill).toFixed(1)} tok/s, ` +
    `decode ${G} tokens ${bestDecode.toFixed(0)} ms = ${(G * 1000 / bestDecode).toFixed(1)} tok/s (best of ${reps})`);
})();
