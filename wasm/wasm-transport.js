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

  function onMessage(e) {
    const m = e.data || {};
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

  async function start() {
    try {
      worker = new Worker(WORKER);
    } catch (e) {
      // a page that may not start workers from its own files may still start them from a blob
      const src = await (await fetch(WORKER)).text();
      worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    }
    worker.onmessage = onMessage;
    worker.onerror = (e) => onMessage({ data: { fatal: "the engine stopped: " + (e.message || "worker error") } });
    worker.postMessage({ op: "init", base: location.href });
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
