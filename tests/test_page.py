"""The chat page (superborg/web/index.html) in headless Chromium, driven through a scripted
BORG_TRANSPORT: the event contract it shares with superborg/interp.py and wasm/trace-session.js
(token ids, "prefill" progress, error events, busy retries, request ids on chat and stop), the
settings it sends, and the flow view (the word map: sprites and the pigment via window.__superborg.flowStats, the
no-map fallback, reduced motion, the strip's wrapping at phone width, the scrubber's keys, the first-run cards, the cards
following a sweep, the layer table and Step button, readable region landmarks, the follow-live queue policy, the Canvas2D
fallback, ?replay= of a synthetic trace and of a real recording: tests/fixtures/qwen2.5-0.5b-q8_0-insects.jsonl, or the trace
SUPERBORG_TRACE=<path to a recorded .jsonl> names in its place; that case is a pytest skip when neither is at hand). Skipped
without node, Playwright for node, or a browser it can launch. test_vocab_map_routes covers the UI server's GET /api/vocab_map.json and
.bin, where the page fetches its word map over HTTP (skipped without the engine and a model, like tests/test_ui.py)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "superborg", "web", "index.html")
# a complete recording (request, prompt, four tokens, done) of Qwen2.5-0.5B answering "What is the single word for a
# person who studies insects?" with the full trace of every token, 170 KB; SUPERBORG_TRACE=<path> replays another instead
RECORDED_TRACE = os.path.join(ROOT, "tests", "fixtures", "qwen2.5-0.5b-q8_0-insects.jsonl")

# A transport whose chat() plays one plan of window.__plans per call. A step is an event, {wait: ms}
# or {untilStop: true} (until stop() is called with this chat's request_id). Stops are logged.
TRANSPORT = r"""
(function () {
  const NL = 4, NH = 2;
  window.__calls = []; window.__plans = []; window.__stopLog = [];
  window.__INFO = Object.assign({ model: "Scripted", arch: "x", quant: "f32", n_layers: NL, n_heads: NH, n_kv_heads: 1, dim: 8,
    hidden_dim: 16, vocab: 100, n_ctx: 4096, chat_format: "chatml", threads: 1, lens_layers: [0, 1, 2, 3, 4], lens_stride: 1,
    max_tokens_default: 512, busy: false }, window.__INFO_OVERRIDE || {});
  window.mkTok = (i, pos, o = {}) => {
    const raw = o.raw != null ? o.raw : o.text, id = o.id != null ? o.id : 10 + i, lens = [];
    for (const l of o.lensLayers || [0, 1, 2, 3, 4])
      lens.push({ layer: l, top: o.top || [[raw, 0.5, id], ["b", 0.1, 2], ["c", 0.05, 3]], p: 0.5, rank: o.rank ? o.rank(l) : l >= 2 ? 0 : 3 });
    const attn_top = [], attn_entropy = [], mlp_top = [];
    for (let l = 0; l < NL; l++) {
      attn_top.push([[0, 0.4], [pos, 0.3]]);
      attn_entropy.push(Array.from({ length: NH }, (_, h) => 0.1 * (h + 1) + 0.01 * l));
      mlp_top.push(o.mlp ? o.mlp(l) : [[1, 2.5], [2, -1.5]]);
    }
    return { type: "token", i, id, text: o.eos ? "" : o.text, raw, eos: !!o.eos, pos, prob: o.prob ?? 0.5, entropy: 1.2,
             alternatives: o.alts || [[raw, o.prob ?? 0.5, id], ["zz", 0.1, 99]], lens,
             norms: { attn: [1, 2, 3, 4], mlp: [2, 3, 4, 5], resid: [3, 4, 5, 6, 7] },
             attn_top, attn_entropy, mlp_top, decode_ms: 1, trace_ms: 1 };
  };
  window.mkPrompt = (tokens) => ({ type: "prompt", tokens, n_prompt: tokens.length, n_reused: 0, prompt_ms: 1 });
  window.mkDone = (text = "", n = 1, reason = "length") => ({ type: "done", text, n_gen: n, stop_reason: reason, seconds: 1, tok_per_s: n });
  window.BORG_TRANSPORT = {
    name: "scripted",
    async info() { return JSON.parse(JSON.stringify(window.__INFO)); },
    async chat(body, onEvent) {
      window.__calls.push(JSON.parse(JSON.stringify(body)));
      const plan = window.__plans.shift() || [];
      for (const step of plan) {
        if (step.wait) await new Promise((r) => setTimeout(r, step.wait));
        else if (step.untilStop) await new Promise((r) => { const iv = setInterval(() => { if (window.__stopLog.includes(body.request_id)) { clearInterval(iv); r(); } }, 10); });
        else onEvent(step);
      }
    },
    async stop(id) { window.__stopLog.push(id == null ? null : id); return { stopped: true }; },
    // the flow view's vocabulary map: 100 word pieces on a spiral (ids 0..99 cover every token the plans use); none with __NO_MAP
    async vocabMap() {
      if (window.__NO_MAP) return null;
      const n = 100, u = new Uint16Array(2 * n);
      const f16 = (v) => { const s = v < 0 ? 0x8000 : 0; v = Math.abs(v); if (v < 1e-6) return s; const e = Math.floor(Math.log2(v)); return s | ((e + 15) << 10) | Math.min(1023, Math.round((v / 2 ** e - 1) * 1024)); };
      for (let i = 0; i < n; i++) { const a = i * 0.7, r = 0.1 + 0.85 * i / n; u[2 * i] = f16(r * Math.cos(a)); u[2 * i + 1] = f16(r * Math.sin(a)); }
      const cluster = { label: "b c zz", x: 0.2, y: 0.1, n, tokens: [["b", 2], ["c", 3], ["zz", 99]] };
      if (window.__LANDMARK) cluster.landmark = "paris · city · france";  // a readable region name (the landmark rule of tools/vocab_map.py)
      const clusters = [cluster];  // __CLUSTERS = k: k - 1 smaller regions spread over the map (the label-count case)
      for (let i = 1; i < (window.__CLUSTERS || 1); i++) clusters.push({ label: "r" + i, x: 0.8 * Math.cos(i * 2.4), y: 0.8 * Math.sin(i * 2.4), n: n - i, tokens: [["r" + i, i]] });
      return { meta: { n, stability_knn15: 0.5, clusters }, bin: u.buffer };
    },
  };
})();
"""

DRIVER = r"""
const { chromium } = require("playwright");
const fs = require("fs"), http = require("http"), url = require("url");
const [PAGE, transport] = [process.argv[2], fs.readFileSync(process.argv[3], "utf8")];
const RECORDED = process.argv[4] ? fs.readFileSync(process.argv[4]) : null;  // a recorded .jsonl trace for flow_replay_recorded, or none
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = {};
const check = (name, ok, info) => { results[name] = { ok: !!ok, info }; };
// opts: prefs (saved view prefs; the cases written for the panels open the Details view), noMap (no vocabulary
// map: the flow view's ladder fallback), reducedMotion, coach (keep the first-run cards), url (instead of PAGE), viewport
async function open(browser, info = {}, saved = null, opts = {}) {
  const ctx = await browser.newContext({ viewport: opts.viewport || { width: 1280, height: 900 }, reducedMotion: opts.reducedMotion || "no-preference" });
  let pre = `window.__INFO_OVERRIDE = ${JSON.stringify(info)};\n`;
  if (saved) pre += `localStorage.setItem("superborg.ui.settings", ${JSON.stringify(JSON.stringify(saved))});\n`;
  const prefs = "prefs" in opts ? opts.prefs : { view: "details" };
  if (prefs) pre += `localStorage.setItem("superborg.ui.prefs", ${JSON.stringify(JSON.stringify(prefs))});\n`;
  if (!opts.coach) pre += `localStorage.setItem("superborg.ui.coach", "done");\n`;
  if (opts.noMap) pre += "window.__NO_MAP = true;\n";
  if (opts.landmark) pre += "window.__LANDMARK = true;\n";
  if (opts.clusters) pre += `window.__CLUSTERS = ${opts.clusters};\n`;
  await ctx.addInitScript(pre + transport);
  const page = await ctx.newPage();
  page.errs = []; page.conErrs = [];
  page.on("pageerror", (e) => page.errs.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") page.conErrs.push(m.text()); });
  await page.goto(opts.url || PAGE);
  await page.waitForFunction(() => window.__superborg && window.__superborg.ready);
  return { ctx, page };
}
const ask = async (page, q) => { await page.fill("#input", q); await page.keyboard.press("Enter"); };
const idle = async (page) => { await page.waitForFunction(() => !window.__superborg.generating, null, { timeout: 15000 }); await sleep(60); };
// the flow view: the story of token `sel` has landed on the output layer and its narration (rebuilt in idle time) is there.
// The narration must name the word: after a tap the previous story's "4 of 4" and narration stay in the DOM until the new
// story starts (40-90 ms later), so the label alone can match the stale state.
const landed = async (page, sel) => {
  await page.waitForFunction((k) => {
    const S = window.__superborg, r = S.toks[k], narr = document.getElementById("fnarr"), text = narr.textContent;
    return S.sel === k && !!r && !narr.classList.contains("dim") && document.getElementById("flayerLabel").textContent === "4 of 4"
      && /Layer 4 · /.test(text) && text.startsWith(r.kind === "reply" ? `Word ${r.i + 1} of the reply` : "Reading ");
  }, sel, { timeout: 15000 });
  await sleep(100);
};
// a site without a vocabulary map answers 404 to the page's probe for one; the browser logs that as an error
const realErrors = (page) => page.conErrs.filter((m) => !/vocab_map|404/.test(m));
const flowState = (page) => page.evaluate(() => { const S = window.__superborg, fs = S.flowStats; return {
  renderer: fs.renderer, frames: fs.frames, particles: fs.particles, trail: fs.trail, pigment: fs.pigment, points: fs.points, mapLoaded: fs.mapLoaded, mean: +fs.mean.toFixed(2), max: +fs.max.toFixed(1),
  reduced: fs.reduced, animating: fs.animating, sel: S.sel, layer: S.layer, flowShown: !document.getElementById("flow").hidden, detailsHidden: document.getElementById("details").hidden,
  layerLabel: document.getElementById("flayerLabel").textContent, narr: document.getElementById("fnarr").textContent, hud: document.getElementById("fhudMap").textContent,
  labels: document.getElementById("flabels").textContent, marks: document.getElementById("fmarks").children.length, washed: document.querySelectorAll("#fwords .fw.read").length,
  cur: (document.querySelector("#fwords .fw.cur") || {}).textContent, reads: document.getElementById("freadsLayer").textContent, cands: document.getElementById("fcandsLayer").textContent,
  webgl2: !!document.createElement("canvas").getContext("webgl2") }; });
// a static server for the replay case: the page and a recorded trace (fetch() refuses file: URLs)
function serveFiles(files) {
  return new Promise((res) => {
    const srv = http.createServer((req, r) => { const f = files[url.parse(req.url).pathname]; if (!f) { r.writeHead(404); r.end(); return; } r.writeHead(200, { "Content-Type": f.type }); r.end(f.body); });
    srv.listen(0, "127.0.0.1", () => res(srv));
  });
}
function syntheticTrace() {  // a trace in the recorded format (docs/VIEWER.md): a request line, a traced prompt position, the prompt, three tokens, done
  const NL = 4;
  const lens = (raw, id, p) => Array.from({ length: NL + 1 }, (_, l) => ({ layer: l, top: [[raw, 0.5, id], ["b", 0.1, 2], ["c", 0.05, 3]], p, rank: l >= 2 ? 0 : 3, entropy: 2.0 - 0.4 * l, rival_p: 0.1, rival_rank: 1 }));
  const rows = (n) => Array.from({ length: NL }, () => ({ scale: 0.5, sink: 0.3, pos: Array.from({ length: n - 1 }, (_, j) => j + 1), w: Array.from({ length: n - 1 }, (_, j) => 255 - 40 * j) }));
  const tok = (i, pos, text, id) => ({ type: "token", i, id, text, raw: text, eos: false, pos, prob: 0.6, entropy: 1.1, alternatives: [[text, 0.6, id], ["zz", 0.2, 99]], rival: ["zz", 0.2, 99],
    lens: lens(text, id, 0.6), norms: { attn: [1, 2, 3, 4], mlp: [2, 3, 4, 5], resid: [3, 4, 5, 6, 7] },
    resid: { norm: [3, 4, 5, 6, 7], spike: [8, 40, 50, 45, 30], xyz: [[0.1 * i, 0.2, 0.3], [0.2, 0.1 * i, 0.3], [0.3, 0.2, 0.1 * i], [0.1, 0.1, 0.1], [0.2 * i, 0.2, 0.2]] },
    attn_eff: rows(pos + 1), attn_rows: rows(pos + 1), attn_top: Array.from({ length: NL }, () => [[1, 0.4], [pos, 0.3]]), attn_entropy: Array.from({ length: NL }, () => [0.3, 0.5]),
    mlp_top: Array.from({ length: NL }, () => [[1, 2.5], [2, -1.5]]), decode_ms: 1, trace_ms: 1 });
  const ptok = Object.assign(tok(0, 1, "", 11), { type: "prompt_token", next: 12 });
  for (const k of ["prob", "entropy", "alternatives", "i", "text", "raw"]) delete ptok[k];
  const evs = [{ type: "request", model: "scripted", body: { messages: [{ role: "user", content: "replayed question" }] } }, ptok,
    { type: "prompt", tokens: ["<s>", "hello", "world"], n_prompt: 3, n_reused: 0, prompt_ms: 1 }, tok(0, 2, "x", 10), tok(1, 3, "y", 11), tok(2, 4, "z", 12),
    { type: "done", text: "xyz", n_gen: 3, stop_reason: "eos", seconds: 1, tok_per_s: 3 }];
  return evs.map((e) => JSON.stringify(e)).join("\n") + "\n";
}
const outRows = (page) => page.evaluate(() => [...document.querySelectorAll("#outRows .orow")].filter((r) => !r.hidden)
  .map((r) => [r.querySelector(".v").textContent, r.classList.contains("chosen")]));

const cases = {
  async chosen_by_id(page) {
    // two byte tokens with the same text U+FFFD: the chosen one (id 7) is the second alternative
    await page.evaluate(() => __plans.push([mkPrompt(["a", "b"]),
      mkTok(0, 1, { text: "", raw: "�", id: 7, prob: 0.3, alts: [["�", 0.4, 5], ["�", 0.3, 7], ["x", 0.1, 8]] }),
      mkTok(1, 2, { text: "q", id: 20, prob: 0.001, alts: [["�", 0.5, 5], ["q", 0.2, 21]] }), mkDone("q", 2)]));
    await ask(page, "x"); await idle(page);
    await page.click(".msg.asst .t >> nth=0"); await sleep(60);
    const a = await outRows(page);
    await page.click(".msg.asst .t >> nth=1"); await sleep(60);
    const b = await outRows(page);
    check("chosen_by_id", JSON.stringify(a) === JSON.stringify([["0.400", false], ["0.300", true], ["0.100", false]])
      && b.length === 3 && b[2][1] && !b[1][1], { a, b });
  },
  async error_events(page) {
    await page.evaluate(() => __plans.push([{ type: "error", error: "the conversation does not fit the context" }]));
    await ask(page, "a"); await idle(page);
    const e = await page.evaluate(() => [...document.querySelectorAll(".msg.user .err")].map((x) => x.textContent));
    await page.fill("#input", "");
    await page.evaluate(() => {
      __plans.push([{ type: "error", message: "a generation is already running", code: "busy" }]);
      __plans.push([{ type: "error", message: "a generation is already running; stop it first", status: 409 }]);
      __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "ok" }), mkDone("ok", 1)]);
    });
    await ask(page, "b"); await idle(page);
    const r = await page.evaluate(() => ({ calls: __calls.length, body: document.querySelector(".msg.asst .body").textContent }));
    check("error_events", e[0] === "Error: the conversation does not fit the context" && r.calls === 4 && r.body === "ok", { e, r });
  },
  async request_ids_and_stop(page) {
    await page.evaluate(() => {
      __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]);
      __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "y" }), { untilStop: true }, mkDone("y", 1, "cancelled")]);
    });
    await ask(page, "1"); await idle(page);
    await ask(page, "2"); await sleep(150);
    await page.click("#sendBtn");  // Stop
    await idle(page);
    const r = await page.evaluate(() => ({ ids: __calls.map((c) => c.request_id), stops: __stopLog }));
    check("request_ids_and_stop", r.ids.length === 2 && r.ids[0] !== r.ids[1] && r.ids.every((x) => /^[A-Za-z0-9_-]{1,64}$/.test(x))
      && JSON.stringify(r.stops) === JSON.stringify([r.ids[1]]), r);
  },
  async stop_while_waiting_for_busy_engine(page) {
    await page.evaluate(() => { for (let i = 0; i < 60; i++) __plans.push([{ type: "error", message: "busy", status: 409, code: "busy" }]); });
    await ask(page, "d"); await sleep(600);
    const status = await page.textContent("#statusText");
    await page.click("#sendBtn");  // Stop
    await idle(page);
    const n = await page.evaluate(() => __calls.length);
    await sleep(500);
    const r = await page.evaluate(() => ({ calls: __calls.length, ids: __calls.map((c) => c.request_id), stops: __stopLog,
      note: document.querySelector(".msg.user .meta") && document.querySelector(".msg.user .meta").textContent, input: document.getElementById("input").value }));
    check("stop_while_waiting_for_busy_engine", /waiting for the engine/.test(status) && r.calls === n && r.stops.length === 1
      && r.stops[0] === r.ids[0] && r.note === "Stopped before the reply started." && r.input === "d", Object.assign(r, { status }));
  },
  async prefill_then_new_chat(page) {
    await page.evaluate(() => {
      __plans.push([{ wait: 150 }, { type: "prefill", done: 128, total: 512 }, { wait: 150 }, { type: "prefill", done: 256, total: 512 },
                    { untilStop: true }, { wait: 300 }, mkPrompt(["a"]), mkDone("", 0, "cancelled")]);
      __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "4" }), mkDone("4", 1)]);
    });
    await ask(page, "long");
    await page.waitForFunction(() => /256 of 512/.test(document.getElementById("statusText").textContent));
    const pend = await page.textContent(".msg.asst .pending");
    await page.click("#newBtn");
    await ask(page, "2+2?");
    await sleep(100);
    const waiting = await page.textContent("#statusText");
    const callsWhileWaiting = await page.evaluate(() => __calls.length);
    await idle(page);
    const r = await page.evaluate(() => ({ body: document.querySelector(".msg.asst .body").textContent, ids: __calls.map((c) => c.request_id), stops: __stopLog,
      err: document.querySelectorAll(".err").length }));
    check("prefill_then_new_chat", pend === "reading the prompt… 50%" && /previous reply/.test(waiting) && callsWhileWaiting === 1
      && r.body === "4" && !r.err && JSON.stringify(r.stops) === JSON.stringify([r.ids[0]]), Object.assign(r, { pend, waiting }));
  },
  async stride_default_not_pinned(page) {
    await page.click("#setBtn");
    const shown = await page.inputValue("#sStride");
    await page.fill("#sTemp", "0.5");
    await page.keyboard.press("Escape");
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "s"); await idle(page);
    const body = await page.evaluate(() => __calls[0]);
    await page.click("#setBtn");
    await page.fill("#sStride", "2");
    const s2 = await page.evaluate(() => JSON.parse(localStorage.getItem("superborg.ui.settings")).lens_stride);
    await page.fill("#sStride", "3");
    const s3 = await page.evaluate(() => JSON.parse(localStorage.getItem("superborg.ui.settings")).lens_stride);
    check("stride_default_not_pinned", shown === "3" && !("lens_stride" in body) && !("max_tokens" in body) && body.temperature === 0.5
      && s2 === 2 && s3 === 0, { shown, keys: Object.keys(body), s2, s3 });
  },
  async saved_values_clamped(page) {
    await page.click("#setBtn");
    const shown = await page.inputValue("#sMax");
    await page.keyboard.press("Escape");
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "m"); await idle(page);
    const b = await page.evaluate(() => __calls[0]);
    const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("superborg.ui.settings")));
    check("saved_values_clamped", shown === "2048" && b.max_tokens === 2048 && b.lens_stride === 4 && saved.max_tokens === 4096,
      { shown, max: b.max_tokens, stride: b.lens_stride, saved });
  },
  async selection_labels(page) {
    // prompt [p0 p1 p2], reply A B; lens at layers 0, 3, 4 only
    await page.evaluate(() => __plans.push([mkPrompt(["p0", "p1", "p2"]),
      mkTok(0, 2, { text: "A", lensLayers: [0, 3, 4], rank: (l) => (l >= 3 ? 0 : 5) }),
      mkTok(1, 3, { text: "B", lensLayers: [0, 3, 4], rank: (l) => (l >= 4 ? 0 : 2) }), mkDone("AB", 2)]));
    await ask(page, "x"); await idle(page);
    await page.click(".msg.asst .t >> nth=1"); await sleep(60);
    const r = await page.evaluate(() => ({ what: document.getElementById("selWhat").textContent, key: document.getElementById("emergeKey").textContent,
      out: document.querySelector("#chips .chip.out").dataset.p, lsel: document.querySelectorAll("#readRows .rrow.lsel").length,
      mlp: [...document.querySelectorAll("#mlpRows .mrow")].filter((x) => !x.hidden).length }));
    check("selection_labels", /pos 5/.test(r.what) && r.out === "4" && /top-1 by layer 4/.test(r.what) && /^first traced layer/.test(r.key)
      && r.lsel === 0 && r.mlp === 2, r);
  },
  async mlp_all_layers(page) {
    await page.evaluate(() => __plans.push([mkPrompt(["a"]),
      mkTok(0, 0, { text: "x", mlp: (l) => Array.from({ length: 6 }, (_, j) => [100 * l + j, (l === 3 ? 50 : 10) - j]) }), mkDone("x", 1)]));
    await ask(page, "x"); await idle(page);
    const r = await page.evaluate(() => [...document.querySelectorAll("#mlpRows .mrow")].filter((x) => !x.hidden).map((x) => x.textContent));
    check("mlp_all_layers", r.length === 6 && r.every((s) => s.startsWith("L4 #")), r);
  },
  async zero_width_tokens_clickable(page) {
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "", raw: "�" }), mkTok(1, 1, { text: "❤" }),
      mkTok(2, 2, { text: "️" }), mkTok(3, 3, { text: "\n" }), mkTok(4, 4, { text: "ok" }), mkDone("", 5)]));
    await ask(page, "x"); await idle(page);
    const sel = [];
    for (const k of [0, 2, 3]) { await page.click(`.msg.asst .t[data-k="${k}"]`, { timeout: 2000 }); sel.push(await page.evaluate(() => window.__superborg.sel)); }
    const body = await page.evaluate(() => document.querySelector(".msg.asst .body").textContent);
    check("zero_width_tokens_clickable", JSON.stringify(sel) === "[0,2,3]" && body === "❤️\nok", { sel, body });
  },
  async stop_during_prefill(page) {
    // the engine ends a stop while the prompt is read with "prompt" then a 0-token "done": no reply
    // exists, so the turn is dropped and the text handed back, as for a stop before the start
    await page.evaluate(() => __plans.push([{ type: "prefill", done: 64, total: 512 }, { untilStop: true },
      mkPrompt(["a"]), mkDone("", 0, "cancelled")]));
    await ask(page, "a long question");
    await page.waitForFunction(() => /64 of 512/.test(document.getElementById("statusText").textContent));
    await page.click("#sendBtn");
    await idle(page);
    const r = await page.evaluate(() => ({ asst: document.querySelectorAll(".msg.asst").length, hist: __superborg.history.length,
      input: document.getElementById("input").value, note: (document.querySelector(".msg.user .meta") || {}).textContent }));
    check("stop_during_prefill", r.asst === 0 && r.hist === 0 && r.input === "a long question"
      && r.note === "Stopped before the reply started.", r);
  },
  async layer_values_and_announce(page) {
    await page.evaluate(() => __plans.push([mkPrompt(["a", "b"]), mkTok(0, 1, { text: "x" }), mkTok(1, 2, { text: "y" }), mkDone("xy", 2)]));
    await ask(page, "x"); await idle(page);
    await page.focus("#mapScroll");
    await page.keyboard.press("ArrowLeft");
    const a1 = await page.textContent("#announce");
    await page.keyboard.press("ArrowUp"); await page.keyboard.press("ArrowUp"); await sleep(60);
    const r = await page.evaluate(() => ({ a2: document.getElementById("announce").textContent,
      c: document.getElementById("contribVals").textContent, e: document.getElementById("entVals").textContent }));
    check("layer_values_and_announce", a1 === "x, reply 1 token 1, p 0.500, all layers" && /; layer 2: chosen token p 0\.500, rank 1; lens top-1 x$/.test(r.a2)
      && r.c === "layer 2: attention 2.00 · MLP 3.00 · residual 5.00" && /: 0\.11 0\.21$/.test(r.e), Object.assign(r, { a1 }));
  },
  // ---- the flow view (the default): the word map, the story, the scrubber, replay
  async flow_particles(page) {
    // two reply tokens: the newest one's story plays (a sweep to the output layer) and leaves sprites on the field
    await page.evaluate(() => __plans.push([mkPrompt(["a", "b"]), mkTok(0, 1, { text: "x" }), mkTok(1, 2, { text: "y" }), mkDone("xy", 2)]));
    await ask(page, "q"); await idle(page); await landed(page, 1);
    const r = await flowState(page);
    check("flow_particles", r.flowShown && r.detailsHidden && r.mapLoaded && r.points === 100 && r.particles > 0 && r.trail > 0 && r.frames > 0
      && (!r.webgl2 || r.renderer === "webgl2") && /Word 2 of the reply/.test(r.narr) && r.cur === "y" && r.washed > 0 && /100 word pieces/.test(r.hud)
      && /all layers/.test(r.reads) && r.marks >= 2 && r.layer === 0, r);
  },
  async flow_map_absent(page) {
    // no vocabulary map: the candidates are laid out by rank and layer, everything else is unchanged
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "q"); await idle(page); await landed(page, 0);
    const r = await flowState(page);
    check("flow_map_absent", !r.mapLoaded && r.points === 0 && /no word map/.test(r.hud) && /1st guess/.test(r.labels) && r.particles > 0 && r.trail > 0
      && /Word 1 of the reply/.test(r.narr) && r.marks >= 1, r);
  },
  async flow_reduced_motion(page) {
    // prefers-reduced-motion: no sweep (the story lands at once), no field drift (the frame loop idles)
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "q"); await idle(page);
    await page.waitForFunction(() => window.__superborg.sel === 0 && window.__superborg.flowStats.particles > 0, null, { timeout: 10000 });
    const first = await flowState(page);
    await sleep(500);
    const a = await flowState(page); await sleep(500); const b = await flowState(page);
    check("flow_reduced_motion", first.reduced && first.layerLabel === "4 of 4" && first.trail > 0 && !a.animating && a.frames === b.frames && b.particles > 0, { first, a, b });
  },
  async flow_strip_wraps(page) {
    // a phone-wide strip: the reply's words (their spaces are glyphs) wrap into lines; none is laid out past the strip's right edge
    await page.evaluate(() => { const plan = [mkPrompt(["a", "b"])]; for (let i = 0; i < 40; i++) plan.push(mkTok(i, i + 1, { text: " word" + i })); plan.push(mkDone("", 40)); __plans.push(plan); });
    await ask(page, "q"); await idle(page); await landed(page, 39);
    const r = await page.evaluate(() => {
      const strip = document.getElementById("fstrip"), box = strip.getBoundingClientRect(), spans = [...document.querySelectorAll("#freply .fw")].map((s) => s.getBoundingClientRect());
      return { n: spans.length, inside: spans.filter((b) => b.left >= box.left - 1 && b.right <= box.right + 1).length, lines: new Set(spans.map((b) => Math.round(b.top))).size,
        scrollW: strip.scrollWidth, clientW: strip.clientWidth, vw: innerWidth };
    });
    check("flow_strip_wraps", r.vw === 390 && r.n === 40 && r.inside === 40 && r.lines > 1 && r.scrollW <= r.clientW, r);
  },
  async flow_scrubber_keys(page) {
    // the layer scrubber is a native range input: arrows, Home and End move the layer; the details follow; the strip has keys too
    await page.evaluate(() => __plans.push([mkPrompt(["a", "b"]), mkTok(0, 1, { text: "x" }), mkTok(1, 2, { text: "y" }), mkDone("xy", 2)]));
    await ask(page, "q"); await idle(page); await landed(page, 1);
    await page.focus("#fscrub");
    await page.keyboard.press("ArrowLeft"); await page.keyboard.press("ArrowLeft"); await sleep(350);
    const two = await flowState(page);
    await page.keyboard.press("Home"); await sleep(60); const zero = await flowState(page);
    await page.keyboard.press("End"); await sleep(60); const end = await flowState(page);
    await page.focus("#fstrip"); await page.keyboard.press("ArrowDown"); await sleep(60); const three = await flowState(page);
    await page.keyboard.press("ArrowLeft"); await landed(page, 0); const prev = await flowState(page);
    check("flow_scrubber_keys", two.layerLabel === "2 of 4" && two.layer === 2 && two.cands === "· layer 2" && two.reads === "· layer 2" && /Layer 2 · /.test(two.narr)
      && zero.layerLabel === "0 of 4" && end.layerLabel === "4 of 4" && end.layer === 0 && three.layerLabel === "3 of 4" && three.layer === 3
      && prev.sel === 0 && prev.cur === "x", { two, zero, end, three, prev });
  },
  async flow_first_run_coach(page) {
    // the first-run guidance: three cards, dismissed state remembered; the ? button brings it back
    const step = () => page.evaluate(() => { const c = document.getElementById("fcoach"); return c.hidden ? null : c.querySelector(".step").textContent; });
    await page.waitForFunction(() => !document.getElementById("fcoach").hidden, null, { timeout: 10000 });  // shown once the flow view has started
    const s1 = await step();
    await page.click("#fcoach .btn.primary"); const s2 = await step();
    await page.click("#fcoach .btn.primary"); const s3 = await step();
    const last = await page.textContent("#fcoach .btn.primary");
    await page.click("#fcoach .btn.primary"); const s4 = await step();
    const stored = await page.evaluate(() => localStorage.getItem("superborg.ui.coach"));
    await page.click("#fhelpBtn"); const s5 = await step();
    check("flow_first_run_coach", /^1 of 3/.test(s1) && /^2 of 3/.test(s2) && /^3 of 3/.test(s3) && last === "Got it" && s4 === null && stored === "done" && /^1 of 3/.test(s5),
      { s1, s2, s3, last, s4, stored, s5 });
  },
  async flow_canvas2d(page) {
    // ?renderer=2d: the Canvas2D fallback draws the same story and pigment (no WebGL2 needed)
    await page.evaluate(() => __plans.push([mkPrompt(["a", "b"]), Object.assign(mkTok(0, 1, { text: "x" }), { rival: ["zz", 0.2, 99] }), mkTok(1, 2, { text: "y" }), mkDone("xy", 2)]));
    await ask(page, "q"); await idle(page); await landed(page, 1);
    const r = await flowState(page);
    check("flow_canvas2d", r.renderer === "canvas2d" && r.mapLoaded && r.particles > 0 && r.trail > 0 && r.pigment === 3 && /Word 2 of the reply/.test(r.narr) && r.marks >= 2, r);
  },
  async flow_replay_done(page) {
    // ?replay=<trace url>: the recorded chat is sent and streamed to "done" with no console errors; a traced prompt word has a story
    await page.waitForFunction(() => window.__superborg.turns.length && window.__superborg.turns[0].done, null, { timeout: 30000 });
    await landed(page, 2);
    const r = await flowState(page);
    const t = await page.evaluate(() => { const S = window.__superborg; return { stop: S.turns[0].done.stop_reason, toks: S.toks.length, question: document.querySelector(".msg.user .bubble").textContent,
      backend: S.info.backend, promptStory: !!document.querySelector('#fwords .fw[data-pos="1"]:not(.nostory)'), nostory: !!document.querySelector('#fwords .fw[data-pos="2"].nostory') }; });
    check("flow_replay_done", t.stop === "eos" && t.toks === 3 && t.question === "replayed question" && /replay/.test(t.backend) && t.promptStory && t.nostory
      && r.particles > 0 && /Word 3 of the reply/.test(r.narr) && !realErrors(page).length, Object.assign(t, { conErrs: page.conErrs, flow: r }));
  },
  async flow_pigment(page) {
    // every reply word deposits its chosen (and runner-up) glow at its map spot; the deposits are rebuilt from the reply itself
    await page.evaluate(() => __plans.push([mkPrompt(["a", "b"]), Object.assign(mkTok(0, 1, { text: "x" }), { rival: ["zz", 0.2, 99] }),
      Object.assign(mkTok(1, 2, { text: "y" }), { rival: ["zz", 0.2, 99] }), mkDone("xy", 2)]));
    await ask(page, "q"); await idle(page); await landed(page, 1);
    const r = await flowState(page);
    await page.click("#themeBtn"); await sleep(150);  // the theme changes the colours: the pigment is redrawn, the same deposits
    const after = await flowState(page);
    const key = await page.textContent("#fkey");
    check("flow_pigment", r.pigment === 4 && after.pigment === 4 && r.particles > 0 && /pigment/.test(key) && r.labels === "b · c · zz", { r, after });
  },
  async flow_sweep_cards(page) {
    // while a story's sweep plays, the per-layer cards follow the scrubber and the narration waits, dimmed and naming no layer;
    // when it lands the full narration names the output layer (prefs.speed 0.5: a 2.8 s sweep, so a mid-sweep layer can be caught)
    await page.evaluate(() => __plans.push([mkPrompt(["a", "b"]), mkTok(0, 1, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "q");
    const mid = await page.evaluate(() => new Promise((res) => {
      const t0 = performance.now();
      (function poll() {
        const m = /^([123]) of 4/.exec(document.getElementById("flayerLabel").textContent), n = document.getElementById("fnarr");
        if (m) { res({ k: +m[1], dim: n.classList.contains("dim"), narr: n.textContent, cands: document.getElementById("fcandsLayer").textContent,
          reads: document.getElementById("freadsLayer").textContent, state: document.getElementById("fstateLayer").textContent }); return; }
        if (performance.now() - t0 > 8000) { res(null); return; }
        requestAnimationFrame(poll);
      })();
    }));
    await idle(page); await landed(page, 0);
    const end = await page.evaluate(() => ({ dim: document.getElementById("fnarr").classList.contains("dim"), narr: document.getElementById("fnarr").textContent, cands: document.getElementById("fcandsLayer").textContent }));
    check("flow_sweep_cards", mid && mid.dim && !/Layer \d/.test(mid.narr) && /Word 1 of the reply/.test(mid.narr) && mid.cands === `· layer ${mid.k}`
      && (mid.k === 0 || mid.reads === `· layer ${mid.k}`) && mid.state === `· layer ${mid.k}` && !end.dim && /Layer 4 · /.test(end.narr) && end.cands === "· layer 4", { mid, end });
  },
  async flow_layer_table(page) {
    // the collapsed layer-by-layer table: one row per layer, the lead layer marked, a click scrubs; the Step button moves one layer up and wraps
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "q"); await idle(page); await landed(page, 0);
    const rows = await page.evaluate(() => { const t = document.getElementById("ftable"); t.open = true; return [...document.querySelectorAll("#ftableRows tr")].map((r) => [r.dataset.l, r.className.trim(), r.children[1].textContent, r.children[2].textContent]); });
    await page.click('#ftableRows tr[data-l="1"]'); await sleep(200);
    const one = await flowState(page);
    const cur = await page.evaluate(() => (document.querySelector("#ftableRows tr.cur") || {}).dataset?.l);
    await page.click("#fstepBtn"); await sleep(60); const two = await flowState(page);
    await page.focus("#fscrub"); await page.keyboard.press("End"); await sleep(60);
    await page.click("#fstepBtn"); await sleep(60); const zero = await flowState(page);
    check("flow_layer_table", rows.length === 5 && rows[2][1] === "lead" && rows[0][1] === "" && /^x 50%$/.test(rows[4][2]) && rows[4][3] === "50%"
      && one.layerLabel === "1 of 4" && /Layer 1 · /.test(one.narr) && cur === "1" && two.layerLabel === "2 of 4" && zero.layerLabel === "0 of 4", { rows, one, cur, two, zero });
  },
  async flow_region_landmark(page) {
    // a map json whose regions carry readable landmarks: the label is the landmark, its tooltip says what it is
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "q"); await idle(page); await landed(page, 0);
    const r = await page.evaluate(() => { const l = document.querySelector("#flabels .rl"); return { text: l && l.textContent, title: l && l.title, inside: l && l.getBoundingClientRect().left >= document.getElementById("fstage").getBoundingClientRect().left }; });
    check("flow_region_landmark", r.text === "paris · city · france" && /commonest whole words: paris · city · france/.test(r.title) && r.inside, r);
  },
  async flow_label_count(page) {
    // the largest regions are labelled: nine on a desktop stage (474 px wide at 1440x900), five on a phone's; every label inside the stage
    await page.evaluate(() => __plans.push([mkPrompt(["a"]), mkTok(0, 0, { text: "x" }), mkDone("x", 1)]));
    await ask(page, "q"); await idle(page); await landed(page, 0);
    const labels = () => page.evaluate(() => { const st = document.getElementById("fstage").getBoundingClientRect(), ls = [...document.querySelectorAll("#flabels .rl")];
      return { w: Math.round(st.width), n: ls.length, inside: ls.every((l) => { const r = l.getBoundingClientRect(); return r.left >= st.left && r.right <= st.right; }) }; });
    const desk = await labels();
    await page.setViewportSize({ width: 390, height: 844 }); await sleep(300);  // the stage's ResizeObserver lays the labels out again
    const phone = await labels();
    check("flow_label_count", desk.n === 9 && desk.inside && desk.w >= 420 && phone.n === 5 && phone.inside && phone.w < 420, { desk, phone });
  },
  async flow_queue_policy(page) {
    // twelve words at once: past six waiting the queue jumps to the newest; the skipped words still deposit pigment and stay tappable
    await page.evaluate(() => { const plan = [mkPrompt(["a"])]; for (let i = 0; i < 12; i++) plan.push(Object.assign(mkTok(i, i, { text: " w" + i }), { rival: ["zz", 0.2, 99] })); plan.push(mkDone("", 12)); __plans.push(plan); });
    await ask(page, "q"); await idle(page); await landed(page, 11);
    const r = await flowState(page);
    const strip = await page.evaluate(() => ({ tappable: document.querySelectorAll("#freply .fw:not(.nostory)").length, cur: document.querySelector("#freply .fw.cur").dataset.pos }));
    await page.click('#freply .fw[data-pos="3"]'); await landed(page, 2);
    const back = await flowState(page);
    check("flow_queue_policy", r.sel === 11 && r.pigment === 24 && /Word 12 of the reply/.test(r.narr) && strip.tappable === 12 && strip.cur === "12"
      && back.sel === 2 && /w2$/.test(back.cur) && back.pigment === 24, { r, strip, back });  // the strip shows the leading space as a glyph
  },
  async flow_replay_recorded(page) {
    // a real recording (RECORDED: the fixture, or SUPERBORG_TRACE): the recorded question is sent, every recorded token
    // arrives with its story and the turn ends with the recorded stop reason, no console errors
    if (!RECORDED) { results.flow_replay_recorded = { ok: false, skipped: "no recorded trace: SUPERBORG_TRACE is not set and tests/fixtures has none" }; return; }
    const evs = RECORDED.toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const want = { toks: evs.filter((e) => e.type === "token").length, stop: evs.find((e) => e.type === "done").stop_reason,
                   question: evs[0].body.messages.slice(-1)[0].content, nL: evs.find((e) => e.type === "token").lens.length - 1 };
    await page.waitForFunction(() => window.__superborg.turns.length && window.__superborg.turns[0].done, null, { timeout: 120000 });
    // the stories play one after another (a frame apart, so "not animating" alone is also the gap between two): wait for the last
    // word's story to land and its narration to be written
    await page.waitForFunction((n) => { const S = window.__superborg; return S.sel === n - 1 && !S.flowStats.animating && !document.getElementById("fnarr").classList.contains("dim"); },
                               want.toks, { timeout: 60000 });
    await sleep(400);
    const r = await flowState(page);
    const t = await page.evaluate(() => { const S = window.__superborg; return { stop: S.turns[0].done.stop_reason, toks: S.toks.length, nL: S.nL,
      question: document.querySelector(".msg.user .bubble").textContent, stories: document.querySelectorAll("#freply .fw:not(.nostory)").length }; });
    check("flow_replay_recorded", t.toks === want.toks && t.stop === want.stop && t.question === want.question && t.nL === want.nL && t.stories === want.toks
      && r.particles > 0 && new RegExp(`Word ${want.toks} of the reply`).test(r.narr) && !realErrors(page).length, Object.assign(t, { want, conErrs: page.conErrs, flow: r }));
  },
};
const setup = {
  stride_default_not_pinned: [{ lens_stride: 3, lens_layers: [0, 3, 4] }],
  saved_values_clamped: [{ n_ctx: 2048 }, { temperature: 0, top_p: 1, max_tokens: 4096, seed: 0, lens_stride: 9 }],
  flow_particles: [{}, null, { prefs: null }],
  flow_map_absent: [{}, null, { prefs: null, noMap: true }],
  flow_reduced_motion: [{}, null, { prefs: null, reducedMotion: "reduce" }],
  flow_strip_wraps: [{}, null, { prefs: null, viewport: { width: 390, height: 844 } }],
  flow_scrubber_keys: [{}, null, { prefs: null }],
  flow_first_run_coach: [{}, null, { prefs: null, coach: true }],
  flow_pigment: [{}, null, { prefs: null }],
  flow_sweep_cards: [{}, null, { prefs: { speed: 0.5 } }],
  flow_layer_table: [{}, null, { prefs: null }],
  flow_region_landmark: [{}, null, { prefs: null, landmark: true }],
  flow_label_count: [{}, null, { prefs: null, clusters: 12, viewport: { width: 1440, height: 900 } }],
  flow_queue_policy: [{}, null, { prefs: null }],
  flow_canvas2d: [{}, null, { prefs: null, url: PAGE + "?renderer=2d" }],
  flow_replay_done: [{}, null, { prefs: null, replay: "/trace.jsonl" }],
  flow_replay_recorded: [{}, null, { prefs: null, replay: "/recorded.jsonl" }],
};
(async () => {
  let browser;
  try { browser = await chromium.launch(); } catch (e) { console.log("LAUNCH-FAILED " + e.message.split("\n")[0]); return; }
  const files = { "/index.html": { type: "text/html; charset=utf-8", body: fs.readFileSync(url.fileURLToPath(PAGE)) }, "/trace.jsonl": { type: "application/x-ndjson", body: syntheticTrace() } };
  if (RECORDED) files["/recorded.jsonl"] = { type: "application/x-ndjson", body: RECORDED };
  const srv = await serveFiles(files), port = srv.address().port;
  for (const [name, fn] of Object.entries(cases)) {
    const [info, saved, opts] = setup[name] || [];
    const o = Object.assign({}, opts || {});
    if (o.replay) { if (name === "flow_replay_recorded" && !RECORDED) { await fn(null); continue; } o.url = `http://127.0.0.1:${port}/index.html?replay=${o.replay}&replay_speed=0`; }
    const { ctx, page } = await open(browser, info, saved, o);
    try { await fn(page); } catch (e) { check(name, false, "threw: " + e.message.split("\n")[0]); }
    if (page.errs.length) check(name, false, { pageErrors: page.errs });
    await ctx.close();
  }
  srv.close();
  await browser.close();
  console.log("RESULTS " + JSON.stringify(results));
})();
"""

CASES = ["chosen_by_id", "error_events", "request_ids_and_stop", "stop_while_waiting_for_busy_engine", "prefill_then_new_chat",
         "stride_default_not_pinned", "saved_values_clamped", "selection_labels", "mlp_all_layers",
         "zero_width_tokens_clickable", "stop_during_prefill", "layer_values_and_announce",
         "flow_particles", "flow_map_absent", "flow_reduced_motion", "flow_strip_wraps", "flow_scrubber_keys", "flow_first_run_coach",
         "flow_pigment", "flow_sweep_cards", "flow_layer_table", "flow_region_landmark", "flow_label_count", "flow_queue_policy", "flow_canvas2d",
         "flow_replay_done", "flow_replay_recorded"]


def _node_env():
    paths = [os.environ.get("NODE_PATH", "")]
    npm = shutil.which("npm")
    if npm:
        r = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, timeout=60)
        paths.append(r.stdout.strip())
    return dict(os.environ, NODE_PATH=os.pathsep.join(p for p in paths if p))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    env = _node_env()
    if subprocess.run([node, "-e", "require('playwright')"], env=env, capture_output=True, timeout=60).returncode:
        pytest.skip("Playwright for node is not installed")
    d = tmp_path_factory.mktemp("page")
    (d / "transport.js").write_text(TRANSPORT, encoding="utf-8")
    (d / "driver.js").write_text(DRIVER, encoding="utf-8")
    recorded = os.environ.get("SUPERBORG_TRACE") or (RECORDED_TRACE if os.path.isfile(RECORDED_TRACE) else "")
    r = subprocess.run([node, str(d / "driver.js"), "file://" + INDEX, str(d / "transport.js")] + ([recorded] if recorded else []),
                       env=env, capture_output=True, text=True, timeout=300)
    if "LAUNCH-FAILED" in r.stdout:
        pytest.skip("no browser for Playwright: " + r.stdout.strip())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULTS ")]
    assert lines, f"driver failed:\n{r.stdout}\n{r.stderr}"
    return json.loads(lines[-1][len("RESULTS "):])


@pytest.mark.parametrize("case", CASES)
def test_page(results, case):
    res = results.get(case)
    assert res is not None, f"{case} did not run"
    if res.get("skipped"):
        pytest.skip(res["skipped"])
    assert res["ok"], json.dumps(res["info"], ensure_ascii=False)


def _model_path(name: str) -> str | None:
    for d in (os.environ.get("BORG_MODELS_DIR", ""), os.path.join(ROOT, "models"), os.path.expanduser("~/.cache/superborg")):
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def test_vocab_map_routes(tmp_path):
    """GET /api/vocab_map.json and .bin (where the page's flow view fetches its word map over HTTP) serve the two files
    next to the model byte for byte and /api/info says "vocab_map": true; without them both routes 404 with the JSON
    error body and the flag is false. The model is reached through a symlink in a directory with or without the files."""
    import http.client
    import threading

    real = _model_path("smollm2-135m-q8_0.borg")
    if not real:
        pytest.skip("smollm2-135m-q8_0.borg not found (set BORG_MODELS_DIR)")
    try:
        from superborg import LLM
        from superborg.ui import make_ui_server
    except (OSError, ImportError) as e:
        pytest.skip(f"engine unavailable: {e}")

    def get(srv, path):
        conn = http.client.HTTPConnection(*srv.server_address[:2], timeout=60)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, r.getheader("Content-Type"), r.read()
        finally:
            conn.close()

    files = {"json": b'{"clusters": []}', "bin": bytes(range(256)) * 4}
    for with_map in (True, False):
        d = tmp_path / ("with" if with_map else "without")
        d.mkdir()
        os.symlink(real, d / "m.borg")
        if with_map:
            for ext, data in files.items():
                (d / f"m.vocab_map.{ext}").write_bytes(data)
        try:
            llm = LLM(str(d / "m.borg"), threads=1, n_ctx=512)
        except (OSError, FileNotFoundError) as e:
            pytest.skip(f"engine unavailable: {e}")
        srv = make_ui_server(llm, "127.0.0.1", 0)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            st, _, body = get(srv, "/api/info")
            assert st == 200 and json.loads(body)["vocab_map"] is with_map
            for ext, data in files.items():
                st, ctype, body = get(srv, f"/api/vocab_map.{ext}")
                if with_map:
                    assert st == 200 and body == data
                    assert ctype == ("application/json" if ext == "json" else "application/octet-stream")
                else:
                    assert st == 404 and "vocab" in json.loads(body)["error"]["message"]
            assert get(srv, "/api/vocab_map")[0] == 404  # only the two files, by their names
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=10)
            llm.close()
