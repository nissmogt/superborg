// Web Worker that runs the borg engine (WebAssembly) and streams chat traces to the page.
//
// Messages from the page:
//   {op: "init", base?, fetchVia?}      download the engine and model listed in model.json; fetchVia
//                                       "page" (the page answers {fetch} requests) or "worker+page"
//                                       (try here first)
//   {op: "info", id}                    -> {id, info}
//   {op: "chat", id, body}              -> {id, event} for every TraceSession event, then {id, end: true};
//                                       body as POST /api/chat (superborg/ui.py), with "request_id"?
//   {op: "stop", id?, request_id?}      cancel the running generation (it ends with stop_reason
//                                       "cancelled"), or only the one with that request_id; a request_id
//                                       whose chat has not started yet is remembered, and that chat
//                                       ends at once -> {id, stop: {stopped} | {stopped: false, pending: true}}
//   {op: "reset"}                       drop the KV cache
// Messages to the page without an id: {status: {text, progress}}, {ready: info}, {fatal: message}.
// A chat while another runs gets the one event {type: "error", message, code: "busy"}.
//
// model.json: {"wasm": "borg.wasm", "n_ctx", "lens_stride",
//              "model": {"name", "id", "size", "wrap"?, "parts": [{"path", "size"}...]}}
// Model parts are cached in IndexedDB when the browser allows it, so a second visit skips the download.
// The worker expects Borg (borg-runtime.js) and TraceSession (trace-session.js) as globals: either
// concatenated into one file (wasm/make_artifact.py does that) or loaded with importScripts.
// When the bundle also holds borgAsmInstantiate (wasm/build-js.sh), pages that block WebAssembly
// run that JavaScript build instead.

/* global self, Borg, TraceSession, borgAsmInstantiate, importScripts, indexedDB, WebAssembly */
"use strict";

if (typeof Borg === "undefined") importScripts("borg-runtime.js", "trace-session.js");

let session = null, model = null, ctx = null, nCtx = 2048, modelName = "", defaultStride = 3;
let running = null;               // {requestId, stopped} of the generation in progress
const started = [];               // request ids of the last 64 generations
const earlyStops = new Map();     // request_id -> time, for stops that came before their chat

// Inside a Worker, messages go through postMessage. Loaded on the page's main thread instead
// (a web view that refuses Workers), the page drives handleMessage directly and receives
// replies through self.borgEngine.onmessage; the engine then runs between the page's frames.
const IN_WORKER = typeof WorkerGlobalScope !== "undefined" ? self instanceof WorkerGlobalScope
                                                          : typeof document === "undefined";
const post = IN_WORKER ? (m) => self.postMessage(m)
                       : (m) => { const h = self.borgEngine && self.borgEngine.onmessage; if (h) h({ data: m }); };
const status = (text, progress = null) => post({ status: { text, progress } });

// Let queued messages (stop) run between tokens: a MessageChannel round trip is not clamped like setTimeout.
const tick = () => new Promise((resolve) => {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => { ch.port1.close(); resolve(); };
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
// drop the parts of other versions of this model (another id, part size or split)
function cachePrune(db, prefix, keep) {
  if (!db) return Promise.resolve();
  return new Promise((resolve) => {
    try {
      const tx = db.transaction("parts", "readwrite"), st = tx.objectStore("parts");
      const r = st.getAllKeys();
      r.onsuccess = () => { for (const k of r.result) if (String(k).startsWith(prefix) && !keep.has(k)) st.delete(k); };
      tx.oncomplete = tx.onerror = tx.onabort = () => resolve();
    } catch (e) { resolve(); }
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

// make_artifact.py stores each model part as the custom section of a WebAssembly module
// (hosts serve .wasm files but not arbitrary binary types): return the section's payload.
function unwrapPart(b) {
  let i = 8;
  const leb = () => { let v = 0, shift = 0, c; do { c = b[i++]; v += (c & 0x7f) * 2 ** shift; shift += 7; } while (c & 0x80); return v; };
  if (b.length < 10 || b[0] !== 0 || b[1] !== 0x61 || b[2] !== 0x73 || b[3] !== 0x6d || b[8] !== 0)
    throw new Error("model part is not a wrapped WebAssembly module");
  i = 9;
  const end = leb() + i;
  const nameLen = leb();
  i += nameLen;
  return b.subarray(i, end);
}

async function readBody(res, onProgress) {
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

// Downloads can go through the page instead ({fetch: {id, url}} to it, {fetchProgress} and
// {fetchResult} back): a worker started from a blob: URL sends no Referer, and a host may
// refuse such requests for the page's own files while the page's fetches succeed.
let fetchVia = "worker";          // "page" once the page says it answers fetches
const pageFetches = new Map();    // id -> {resolve, reject, onProgress}
let nextFetch = 1;
function pageFetch(url, onProgress) {
  return new Promise((resolve, reject) => {
    const id = nextFetch++;
    pageFetches.set(id, { resolve, reject, onProgress });
    post({ fetch: { id, url } });
  });
}
function onPageFetch(m) {
  const f = pageFetches.get(m.id);
  if (!f) return;
  if (m.loaded != null && !m.done) { f.onProgress(m.loaded); return; }
  pageFetches.delete(m.id);
  if (m.error) f.reject(new Error(`cannot download ${m.url || ""}: ${m.error}`));
  else { f.onProgress(m.buffer.byteLength); f.resolve(new Uint8Array(m.buffer)); }
}

async function fetchBytes(url, onProgress) {
  if (fetchVia === "page") return pageFetch(url, onProgress);
  let res;
  try {
    res = await fetch(url);
  } catch (e) {
    if (fetchVia !== "worker+page") throw new Error(`cannot fetch ${url}: ${e.message}`);
    res = null;
  }
  if (res && res.ok) return readBody(res, onProgress);
  if (fetchVia !== "worker+page") throw new Error(`cannot download ${url}: HTTP ${res.status}`);
  console.warn("[borg] the worker cannot fetch", url, res ? "HTTP " + res.status : "", "- asking the page");
  fetchVia = "page";
  return pageFetch(url, onProgress);
}

async function init(base, via) {
  const url = (p) => new URL(p, base || self.location.href).href;
  if (via === "page" || via === "worker+page") fetchVia = via;
  status("Loading the engine");
  let manifest;
  try {
    manifest = JSON.parse(new TextDecoder().decode(await fetchBytes(url("model.json"), () => {})));
  } catch (e) {
    throw new Error(e instanceof SyntaxError ? "model.json is not valid JSON" : e.message);
  }
  nCtx = manifest.n_ctx || 2048;
  const log = (s) => console.warn("[borg]", s);
  let borg, slow = false;
  try {
    if (typeof WebAssembly === "undefined") throw new Error("WebAssembly is not available");
    borg = await Borg.create(await fetchBytes(url(manifest.wasm || "borg.wasm"), () => {}), { log });
  } catch (e) {
    // e.g. a Content-Security-Policy without 'wasm-unsafe-eval': fall back to the wasm2js build
    if (typeof borgAsmInstantiate === "undefined")
      throw new Error("this page cannot run WebAssembly (" + e.message + ")");
    console.warn("[borg] WebAssembly unavailable, using the JavaScript build:", e.message);
    borg = await Borg.create(null, { asm: borgAsmInstantiate, log });
    slow = true;
  }
  const mm = manifest.model, total = mm.size;
  const reservation = borg.reserveModel(total);
  const db = await openCache();
  const keyOf = (part) => `${mm.name}:${mm.id || ""}:${part.path}:${part.size}`;
  await cachePrune(db, `${mm.name}:`, new Set(mm.parts.map(keyOf)));  // before downloading: room for this one
  let done = 0, off = 0, fromCache = 0;
  const pct = (x) => Math.round((100 * x) / total);
  for (const part of mm.parts) {
    const key = keyOf(part);
    let bytes = await cacheGet(db, key);
    if (bytes && bytes.byteLength === part.size) {
      bytes = new Uint8Array(bytes);
      fromCache += bytes.length;
    } else {
      bytes = await fetchBytes(url(part.path), (n) =>
        status(`Downloading ${mm.name} (${pct(done + n)}% of ${(total / 1e6).toFixed(0)} MB)`, (done + n) / total));
      if (mm.wrap === "wasm-custom-section") bytes = unwrapPart(bytes);
      if (bytes.length !== part.size) throw new Error(`${part.path}: expected ${part.size} bytes, got ${bytes.length}`);
      await cachePut(db, key, bytes.slice().buffer);
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
  // the JavaScript build is about 4x slower: trace fewer lens layers by default
  defaultStride = Math.min(Math.max(1, (manifest.lens_stride || 3) * (slow ? 2 : 1)), model.config.n_layers);
  session = new TraceSession(model, ctx, { lensStride: defaultStride });
  modelName = mm.name;
  return info();
}

// as GET /api/info of superborg/ui.py
function info() {
  const i = session.info();
  i.model = modelName || i.model;
  i.lens_stride = defaultStride;
  i.lens_layers = session.lensLayers(defaultStride);
  i.max_tokens_default = Math.min(512, ctx.nCtx);
  i.busy = !!running;
  return i;
}

const ROLES = new Set(["system", "user", "assistant", "tool", "developer", "function"]);
const validRequestId = (v) => typeof v === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(v);

// OpenAI content: a string or a list of text parts (superborg/server.py _content_text)
function contentText(c, where) {
  if (c == null) return "";
  if (typeof c === "string") return c;
  if (!Array.isArray(c)) throw new Error(`${where}: 'content' must be a string or a list of parts`);
  return c.map((part, j) => {
    if (typeof part === "string") return part;
    if (part && typeof part === "object" && (part.type === "text" || part.type === "input_text")) {
      if (typeof part.text !== "string") throw new Error(`${where}.content[${j}]: 'text' must be a string`);
      return part.text;
    }
    if (part && typeof part === "object") throw new Error(`${where}: content part type '${part.type}' is not supported`);
    throw new Error(`${where}.content[${j}] must be a string or an object`);
  }).join("");
}

// streamChat options for a chat body, checked like superborg/ui.py's parse_chat (throws on a bad value)
function parseChat(body) {
  const num = (name, dflt, lo, hi, int = true) => {
    const v = body[name] ?? dflt;
    if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`'${name}' must be a finite number`);
    if (int && !Number.isInteger(v)) throw new Error(`'${name}' must be an integer`);
    if (v < lo || v > hi)
      throw new Error(hi === Infinity ? `'${name}' must be at least ${lo}` : `'${name}' must be in [${lo}, ${hi}]`);
    return v;
  };
  const msgs = body.messages;
  if (!Array.isArray(msgs) || !msgs.length) throw new Error("'messages' must be a non-empty list");
  const messages = msgs.map((m, i) => {
    if (!m || typeof m !== "object" || typeof m.role !== "string")
      throw new Error(`messages[${i}] must be an object with a 'role'`);
    if (!ROLES.has(m.role)) throw new Error(`messages[${i}]: unknown role '${m.role}'`);
    return { role: m.role === "developer" ? "system" : m.role, content: contentText(m.content, `messages[${i}]`) };
  });
  const rid = body.request_id;
  if (rid != null && !validRequestId(rid))
    throw new Error("'request_id' must be 1-64 characters from A-Z, a-z, 0-9, '_' and '-'");
  if (body.enable_thinking != null && typeof body.enable_thinking !== "boolean")
    throw new Error("'enable_thinking' must be true or false");
  return {
    requestId: rid ?? null,
    messages,
    // above the context or the layer count: the most there is
    maxTokens: Math.min(num("max_tokens", Math.min(512, ctx.nCtx), 1, Infinity), ctx.nCtx),
    lensStride: Math.min(num("lens_stride", defaultStride, 1, Infinity), model.config.n_layers),
    temperature: num("temperature", 0, 0, 2, false),
    topP: num("top_p", 1, 0, 1, false),
    topK: num("top_k", 0, 0, 2 ** 30),
    seed: num("seed", 0, 0, 2 ** 64 - 1),
    enableThinking: body.enable_thinking === true,
  };
}

function stop(requestId) {
  if (requestId == null || (running && running.requestId === requestId)) {
    if (running) running.stopped = true;
    return { stopped: !!running };
  }
  if (started.includes(requestId)) return { stopped: false };  // that one has ended already
  // its chat has not started (it may still be on its way): remember the id, the last 64 for 60 s
  earlyStops.delete(requestId);
  earlyStops.set(requestId, Date.now());
  if (earlyStops.size > 64) earlyStops.delete(earlyStops.keys().next().value);
  return { stopped: false, pending: true };
}

// a generation with this id starts: was it stopped already?
function stoppedEarly(requestId) {
  for (const [k, t] of earlyStops) if (Date.now() - t > 60000) earlyStops.delete(k);
  if (requestId == null) return false;
  started.push(requestId);
  if (started.length > 64) started.shift();
  return earlyStops.delete(requestId);
}

async function chat(id, body) {
  let opt;
  try {
    opt = parseChat(body);
  } catch (e) {
    post({ id, event: { type: "error", message: e.message } });
    post({ id, end: true });
    return;
  }
  if (running) {
    post({ id, event: { type: "error", message: "a generation is already running", code: "busy" } });
    post({ id, end: true });
    return;
  }
  const gen = running = { requestId: opt.requestId, stopped: stoppedEarly(opt.requestId) };
  try {
    session.opt.lensStride = opt.lensStride;
    const events = session.streamChat(opt.messages, Object.assign({}, opt, { shouldStop: () => gen.stopped }));
    for (const ev of events) {
      post({ id, event: ev });
      await tick(); // let a stop message in (between tokens and between prompt chunks)
    }
  } catch (e) {
    const ev = { type: "error", message: String(e && e.message ? e.message : e) };
    if (e && e.code) ev.code = e.code;
    post({ id, event: ev });
  } finally {
    running = null;
    post({ id, end: true });
  }
}

let ready = null;
async function handleMessage(m) {
  if (m.fetchResult) return onPageFetch(Object.assign({ done: true }, m.fetchResult));
  if (m.fetchProgress) return onPageFetch(m.fetchProgress);
  try {
    if (m.op === "init") {
      if (!ready) ready = init(m.base, m.fetchVia);
      post({ ready: await ready });
    } else if (m.op === "stop") {
      const rid = m.request_id ?? null;
      const r = rid === null || validRequestId(rid) ? stop(rid) : { stopped: false, error: "bad request_id" };
      if (m.id != null) post({ id: m.id, stop: r });
    } else if (m.op === "info") {
      await ready;
      post({ id: m.id, info: info() });
    } else if (m.op === "reset") {
      if (!running && ctx) ctx.truncate(0);
    } else if (m.op === "chat") {
      await ready;
      await chat(m.id, m.body || {});
    }
  } catch (err) {
    post({ fatal: String(err && err.message ? err.message : err) });
  }
}
if (IN_WORKER) self.onmessage = (e) => handleMessage(e.data || {});
else self.borgEngine = { send: (m) => handleMessage(m || {}), onmessage: null };
