// Web Worker that runs the borg engine (WebAssembly) and streams chat traces to the page.
//
// Messages from the page:
//   {op: "init", base?}                 download the engine and model listed in model.json
//   {op: "info", id}                    -> {id, info}
//   {op: "chat", id, body}              -> {id, event} for every TraceSession event, then {id, end: true}
//   {op: "stop"}                        cancel the running generation (it ends with stop_reason "cancelled")
//   {op: "reset"}                       drop the KV cache
// Messages to the page without an id: {status: {text, progress}}, {ready: info}, {fatal: message}.
//
// model.json: {"wasm": "borg.wasm", "model": {"name", "size", "parts": [{"path", "size"}...]}, "n_ctx"}
// Model parts are cached in IndexedDB when the browser allows it, so a second visit skips the download.
// The worker expects Borg (borg-runtime.js) and TraceSession (trace-session.js) as globals: either
// concatenated into one file (wasm/make_artifact.py does that) or loaded with importScripts.

/* global self, Borg, TraceSession, importScripts, indexedDB */
"use strict";

if (typeof Borg === "undefined") importScripts("borg-runtime.js", "trace-session.js");

let session = null, model = null, ctx = null, busy = false, stopFlag = false, nCtx = 2048;

const post = (m) => self.postMessage(m);
const status = (text, progress = null) => post({ status: { text, progress } });

// Let queued messages (stop) run between tokens: a MessageChannel round trip is not clamped like setTimeout.
const tick = () => new Promise((resolve) => {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => resolve();
  ch.port2.postMessage(0);
});

// ---- IndexedDB cache of model parts (best effort) ----------------------------------------
function openCache() {
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open("borg-model-cache", 1);
      req.onupgradeneeded = () => req.result.createObjectStore("parts");
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch (e) { resolve(null); }
  });
}
function cacheGet(db, key) {
  if (!db) return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const r = db.transaction("parts").objectStore("parts").get(key);
      r.onsuccess = () => resolve(r.result || null);
      r.onerror = () => resolve(null);
    } catch (e) { resolve(null); }
  });
}
function cachePut(db, key, value) {
  if (!db) return Promise.resolve();
  return new Promise((resolve) => {
    try {
      const tx = db.transaction("parts", "readwrite");
      tx.objectStore("parts").put(value, key);
      tx.oncomplete = tx.onerror = tx.onabort = () => resolve();
    } catch (e) { resolve(); }
  });
}

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`cannot download ${url}: HTTP ${res.status}`);
  if (!res.body || !res.body.getReader) {
    const b = new Uint8Array(await res.arrayBuffer());
    onProgress(b.length);
    return b;
  }
  const reader = res.body.getReader(), chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onProgress(n);
  }
  const out = new Uint8Array(n);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function init(base) {
  const url = (p) => new URL(p, base || self.location.href).href;
  status("Loading the engine");
  const manifest = await (await fetch(url("model.json"))).json();
  nCtx = manifest.n_ctx || 2048;
  const wasmBytes = await fetchBytes(url(manifest.wasm || "borg.wasm"), () => {});
  let borg;
  try {
    borg = await Borg.create(wasmBytes, { log: (s) => console.warn("[borg]", s) });
  } catch (e) {
    throw new Error("this browser or page does not allow WebAssembly here (" + e.message + ")");
  }
  const mm = manifest.model, total = mm.size;
  const reservation = borg.reserveModel(total);
  const db = await openCache();
  let done = 0, off = 0, fromCache = 0;
  const pct = (x) => Math.round((100 * x) / total);
  for (const part of mm.parts) {
    const key = `${mm.name}:${mm.id || ""}:${part.path}:${part.size}`;
    let bytes = await cacheGet(db, key);
    if (bytes && bytes.byteLength === part.size) {
      bytes = new Uint8Array(bytes);
      fromCache += bytes.length;
    } else {
      bytes = await fetchBytes(url(part.path), (n) =>
        status(`Downloading ${mm.name} (${pct(done + n)}% of ${(total / 1e6).toFixed(0)} MB)`, (done + n) / total));
      if (bytes.length !== part.size) throw new Error(`${part.path}: expected ${part.size} bytes, got ${bytes.length}`);
      await cachePut(db, key, bytes.buffer);
    }
    reservation.write(bytes, off);
    off += bytes.length;
    done += part.size;
    status(fromCache === done ? `Loading ${mm.name} from the browser cache` : `Downloading ${mm.name} (${pct(done)}%)`,
           done / total);
  }
  status("Preparing the model", 1);
  await tick();
  model = reservation.load();
  ctx = model.context({ nCtx, kvF16: true });
  session = new TraceSession(model, ctx, { lensStride: manifest.lens_stride || 3 });
  const info = session.info();
  info.model = mm.name || info.model;
  return info;
}

function info() {
  const i = session.info();
  i.busy = busy;
  return i;
}

async function chat(id, body) {
  if (busy) { post({ id, event: { type: "error", error: "a generation is already running" } }); post({ id, end: true }); return; }
  busy = true;
  stopFlag = false;
  try {
    if (body.lens_stride) session.opt.lensStride = Math.max(1, body.lens_stride | 0);
    const gen = session.streamChat(body.messages || [], {
      maxTokens: body.max_tokens ?? 256,
      temperature: body.temperature ?? 0,
      topP: body.top_p ?? 1,
      topK: body.top_k ?? 0,
      seed: body.seed ?? 0,
      shouldStop: () => stopFlag,
    });
    for (const ev of gen) {
      post({ id, event: ev });
      await tick(); // let a stop message in
    }
  } catch (e) {
    post({ id, event: { type: "error", error: String(e && e.message ? e.message : e) } });
  } finally {
    busy = false;
    post({ id, end: true });
  }
}

let ready = null;
self.onmessage = async (e) => {
  const m = e.data || {};
  try {
    if (m.op === "init") {
      if (!ready) ready = init(m.base);
      post({ ready: await ready });
    } else if (m.op === "stop") {
      stopFlag = true;
    } else if (m.op === "info") {
      await ready;
      post({ id: m.id, info: info() });
    } else if (m.op === "reset") {
      if (!busy && ctx) ctx.truncate(0);
    } else if (m.op === "chat") {
      await ready;
      await chat(m.id, m.body || {});
    }
  } catch (err) {
    post({ fatal: String(err && err.message ? err.message : err) });
  }
};
