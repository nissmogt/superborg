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
      setStatus({ text: m.fatal, progress: null, error: true });
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
  const blobWorker = (src) => new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
  let alive = false, how = "";

  function attach(w) {
    worker = w;
    w.onmessage = (e) => { alive = true; onMessage(e); };
    w.onerror = (e) => {
      if (!alive && how === "file") {    // its script did not load: try the same file as a blob
        w.terminate();
        startFetched().catch((err) => onMessage({ data: { fatal: String(err.message || err) } }));
        return;
      }
      const what = e && e.message ? e.message : "its script did not load";
      onMessage({ data: { fatal: `the engine's worker (${how}) failed: ${what}` } });
    };
    // a blob: worker sends no Referer, which some hosts require even for the page's own files:
    // let it fall back to fetching through the page
    w.postMessage({ op: "init", base, fetchVia: how === "file" ? "worker+page" : "page" });
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
    if (inline && inline.textContent.trim()) {
      how = "inline";
      attach(blobWorker(inline.textContent));
      return;
    }
    try {
      how = "file";
      attach(new Worker(new URL(WORKER, base).href));
    } catch (e) {
      await startFetched();
    }
  }
  start().catch((e) => onMessage({ data: { fatal: String(e.message || e) } }));

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
  };
})();
