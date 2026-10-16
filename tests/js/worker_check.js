// Driven by tests/test_wasm.py: run wasm/worker.js in Node with the globals of a Web Worker (self,
// postMessage, fetch from a site directory, an in-memory IndexedDB) and print what each step saw.
//   node worker_check.js site_dir request.json
// request.json: {"idb": [[key, value]...] (the cache before init), "steps": [[name, arg]...]}
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const root = path.resolve(__dirname, "..", "..");
const [site, reqPath] = process.argv.slice(2);
const req = JSON.parse(fs.readFileSync(reqPath, "utf8"));

// a static host: files of the site directory, an HTML page for anything missing
globalThis.fetch = async (u) => {
  const f = path.join(site, decodeURIComponent(new URL(u).pathname));
  if (!fs.existsSync(f)) return new Response("<!DOCTYPE html><title>Not found</title>", { status: 404 });
  return new Response(fs.readFileSync(f));
};

// the part of IndexedDB the worker uses: one database, one object store, async requests
const store = new Map(req.idb || []);
const later = (f) => setTimeout(f, 0);
const db = {
  createObjectStore() {},
  transaction() {
    let n = 0;
    const tx = {};
    const op = (f) => {
      const r = {};
      n++;
      later(() => {
        r.result = f();
        if (r.onsuccess) r.onsuccess();
        if (--n === 0) later(() => tx.oncomplete && tx.oncomplete());
      });
      return r;
    };
    tx.objectStore = () => ({
      get: (k) => op(() => store.get(k)),
      put: (v, k) => op(() => { store.set(k, v); }),
      delete: (k) => op(() => { store.delete(k); }),
      getAllKeys: () => op(() => [...store.keys()]),
    });
    return tx;
  },
};
globalThis.indexedDB = { open() { const r = {}; later(() => { r.result = db; r.onupgradeneeded && r.onupgradeneeded(); r.onsuccess(); }); return r; } };

globalThis.self = globalThis;
self.location = { href: "http://site.test/index.html" };
const waiting = new Map();  // id -> {events, onEvent, resolve}
let ready = null;
self.postMessage = (m) => {
  if (m.ready || m.fatal) return ready && ready(m);
  const w = waiting.get(m.id);
  if (!w) return;
  if (m.event) { w.events.push(m.event); if (w.onEvent) w.onEvent(m.event); }
  else { waiting.delete(m.id); w.resolve(m.end ? w.events : m.info || m.stop); }
};
Object.assign(globalThis, require(path.join(root, "wasm", "borg-runtime.js")),
              require(path.join(root, "wasm", "trace-session.js")));
vm.runInThisContext(fs.readFileSync(path.join(root, "wasm", "worker.js"), "utf8"), { filename: "worker.js" });

let nextId = 1;
const send = (m, onEvent) => new Promise((resolve) => {
  const id = nextId++;
  waiting.set(id, { events: [], onEvent, resolve });
  self.onmessage({ data: Object.assign({ id }, m) });
});
const chat = (body, onEvent) => send({ op: "chat", body }, onEvent);
const stop = (requestId) => send(requestId == null ? { op: "stop" } : { op: "stop", request_id: requestId });
const user = (content) => [{ role: "user", content }];
// what a chat's events say, without timings and traces
const brief = (evs) => evs.map((e) => (e.type === "token"
  ? { type: e.type, id: e.id, eos: e.eos, layers: e.lens.map((x) => x.layer), alternatives: e.alternatives,
      lens_top: e.lens[e.lens.length - 1].top, decode_ms: e.decode_ms }
  : e.type === "prompt" ? { type: e.type, n_prompt: e.n_prompt, n_reused: e.n_reused } : e));

const steps = {
  async init() {
    const m = await new Promise((resolve) => { ready = resolve; self.onmessage({ data: { op: "init" } }); });
    return m.ready ? { info: m.ready, idb: [...store.keys()] } : { fatal: m.fatal };
  },
  info: () => send({ op: "info" }),
  stop: (id) => stop(id),
  async chat(body) { return brief(await chat(body)); },
  // two chats at once: the second is refused
  async busy(body) {
    const a = chat(body), b = chat(body);
    return { first: brief(await a), second: await b };
  },
  // a stop for a chat that has not started, then that chat, then one with another id
  async early([id, body]) {
    const reply = await stop(id);
    const evs = brief(await chat(Object.assign({ request_id: id }, body)));
    const other = brief(await chat(Object.assign({ request_id: id + "x" }, body)));
    return { reply, events: evs, other };
  },
  // stop with `id` (null: no id) as soon as the chat emits an event of type `at`
  async stopAt([at, id, body]) {
    let reply = null;
    const evs = await chat(body, (e) => { if (e.type === at && !reply) reply = stop(id); });
    return { reply: await reply, events: brief(evs) };
  },
};

(async () => {
  const out = [];
  for (const [name, arg] of req.steps) out.push(await steps[name](arg));
  process.stdout.write(JSON.stringify(out));
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
