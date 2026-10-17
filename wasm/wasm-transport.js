// Page-side transport for the in-browser engine: runs wasm/worker.js in a Web Worker and exposes
// the interface the chat UI (superborg/web/index.html) expects from window.BORG_TRANSPORT:
//   info() -> Promise<info>, chat(body, onEvent) -> Promise, stop(requestId?) -> Promise<{stopped, pending?}>,
//   onStatus(cb)
// Load this before the UI's main script. The worker file is "worker.js" next to the page.

/* global window, Worker, Blob, URL, fetch, location */
(function () {
  "use strict";
  const WORKER = (window.BORG_WASM && window.BORG_WASM.worker) || "worker.js";
  let worker = null, nextId = 1, statusCbs = [], lastStatus = { text: "Starting the engine", progress: null };
  const pending = new Map();   // id -> {resolve, reject, onEvent}
  let readyResolve, readyReject;
  const ready = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });
  ready.catch(() => {});

  function setStatus(s) {
    lastStatus = s;
    for (const cb of statusCbs) { try { cb(s); } catch (e) { /* ignore */ } }
  }

  // Fetch on the worker's behalf (see fetchVia in worker.js): the page's own requests carry
  // its referer and credentials. Progress goes back every 256 KB, the body as one transfer.
  async function fetchForWorker(w, { id, url }) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      let buffer;
      if (res.body && res.body.getReader) {
        const reader = res.body.getReader(), chunks = [];
        let n = 0, told = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          n += value.length;
          if (n - told >= 262144) { told = n; w.postMessage({ fetchProgress: { id, loaded: n } }); }
        }
        const out = new Uint8Array(n);
        let off = 0;
        for (const c of chunks) { out.set(c, off); off += c.length; }
        buffer = out.buffer;
      } else {
        buffer = await res.arrayBuffer();
      }
      w.postMessage({ fetchResult: { id, url, buffer } }, [buffer]);
    } catch (e) {
      w.postMessage({ fetchResult: { id, url, error: String(e.message || e) } });
    }
  }

  function onMessage(e) {
    const m = e.data || {};
    if (m.fetch) return void fetchForWorker(worker, m.fetch);
    if (m.status) return setStatus(m.status);
    if (m.ready) { setStatus({ text: "", progress: null, ready: true }); return readyResolve(m.ready); }
    if (m.fatal) {
      setStatus({ text: m.fatal, progress: null, error: true, diag: m.diag || diag() });
      readyReject(new Error(m.fatal));
      for (const p of pending.values()) p.reject(new Error(m.fatal));
      pending.clear();
      return;
    }
    const p = pending.get(m.id);
    if (!p) return;
    if (m.info) { pending.delete(m.id); p.resolve(m.info); }
    else if (m.stop) { pending.delete(m.id); p.resolve(m.stop); }
    else if (m.event) p.onEvent && p.onEvent(m.event);
    else if (m.end) { pending.delete(m.id); p.resolve(); }
  }

  // The worker's script comes from (in order of preference) a <script type="text/plain"
  // id="borg-worker-src"> block in the page, the worker.js file next to the page, or that file
  // fetched and started as a blob: (for hosts that serve the page's files but do not let a Worker
  // load them). Relative names resolve against the document's base URL, as the page's own
  // fetches do (a host may set <base>).
  const base = (typeof document !== "undefined" && document.baseURI) || location.href;
  const params = new URLSearchParams(location.search);
  const FORCE = params.get("borg") || (window.BORG_WASM && window.BORG_WASM.engine) || "";  // "main": no Worker

  // What this browser offers, for the failure report (the page shows it when the engine cannot start).
  function capabilities() {
    const c = { ua: navigator.userAgent, origin: location.origin, base, worker: typeof Worker !== "undefined",
                wasm: typeof WebAssembly !== "undefined", simd: false, mem4g: false, indexedDB: typeof indexedDB !== "undefined",
                deviceMemory: navigator.deviceMemory || null, isolated: !!window.crossOriginIsolated };
    try {  // a module with one v128 instruction
      c.simd = c.wasm && WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]));
    } catch (e) { c.simd = "error: " + e.message; }
    try { if (c.wasm) { new WebAssembly.Memory({ initial: 1, maximum: 65536 }); c.mem4g = true; } }
    catch (e) { c.mem4g = "no: " + e.message; }
    return c;
  }
  const caps = capabilities();
  const diag = () => Object.entries(caps).map(([k, v]) => `${k}: ${v}`).join("\n");
  const fail = (text) => onMessage({ data: { fatal: text, diag: diag() } });
  const blobWorker = (src) => new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
  let alive = false, how = "";

  let silence = null;   // a worker that never reports in is replaced by the main-thread engine
  function attach(w) {
    worker = w;
    w.onmessage = (e) => { alive = true; if (silence) { clearTimeout(silence); silence = null; } onMessage(e); };
    w.onerror = (e) => {
      if (!alive && how === "file") {    // its script did not load: try the same file as a blob
        w.terminate();
        startFetched().catch((err) => fail(String(err.message || err)));
        return;
      }
      if (!alive && how !== "main-thread") { w.terminate(); startMainThread(e && e.message); return; }
      const what = e && e.message ? e.message : "its script did not load";
      fail(`the engine's worker (${how}) failed: ${what}`);
    };
    if (how !== "main-thread")
      silence = setTimeout(() => { if (!alive) { w.terminate(); startMainThread("no reply from the worker in 8 s"); } }, 8000);
    // a blob: worker sends no Referer, which some hosts require even for the page's own files:
    // let it fall back to fetching through the page
    w.postMessage({ op: "init", base, fetchVia: how === "file" ? "worker+page" : "page" });
  }

  // Last resort: run the engine's script on the page itself. It yields between tokens, so the
  // page stays usable, but each token's work blocks it for its duration.
  function startMainThread(reason) {
    const inline = typeof document !== "undefined" && document.getElementById("borg-worker-src");
    if (!inline || !inline.textContent.trim()) { fail(`the engine's worker failed (${reason || "unknown"}) and the page has no embedded engine`); return; }
    if (window.borgEngine) return;
    how = "main-thread";
    console.warn("[borg] running the engine on the main thread:", reason);
    const sc = document.createElement("script");
    sc.textContent = inline.textContent;
    document.head.appendChild(sc);
    if (!window.borgEngine) { fail("the embedded engine did not start on the page (" + (reason || "") + ")"); return; }
    const w = { postMessage: (m) => window.borgEngine.send(m), terminate() {}, onmessage: null, onerror: null };
    window.borgEngine.onmessage = (e) => w.onmessage && w.onmessage(e);
    attach(w);
    setStatus({ text: "Starting the engine on the page (no worker: " + (reason || "") + ")", progress: null });
  }

  async function startFetched() {
    how = "fetched";
    setStatus({ text: "Loading the engine script", progress: null });
    const res = await fetch(new URL(WORKER, base).href);
    if (!res.ok) throw new Error(`cannot load ${WORKER}: HTTP ${res.status}`);
    attach(blobWorker(await res.text()));
  }

  async function start() {
    const inline = typeof document !== "undefined" && document.getElementById("borg-worker-src");
    if (FORCE === "main" || typeof Worker === "undefined") { startMainThread(FORCE === "main" ? "requested" : "no Worker in this browser"); return; }
    if (inline && inline.textContent.trim()) {
      how = "inline";
      try { attach(blobWorker(inline.textContent)); } catch (e) { startMainThread("cannot start a worker: " + e.message); }
      return;
    }
    try {
      how = "file";
      attach(new Worker(new URL(WORKER, base).href));
    } catch (e) {
      await startFetched();
    }
  }
  start().catch((e) => fail(String(e.message || e)));

  function post(op, extra, onEvent) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject, onEvent });
      worker.postMessage(Object.assign({ op, id }, extra));
    });
  }
  const request = (op, extra, onEvent) => ready.then(() => post(op, extra, onEvent));

  window.BORG_TRANSPORT = {
    kind: "wasm",
    ready,
    info: () => request("info", {}),
    chat: (body, onEvent) => request("chat", { body }, onEvent),
    // not queued behind the engine's start: the worker remembers a stop for a chat it has not seen yet
    stop: async (requestId) => (worker ? post("stop", requestId ? { request_id: requestId } : {})
      .catch(() => ({ stopped: false })) : { stopped: false }),
    reset: async () => { if (worker) worker.postMessage({ op: "reset" }); },
    onStatus: (cb) => { statusCbs.push(cb); cb(lastStatus); },
    capabilities: () => Object.assign({ engine: how }, caps),
  };
})();
