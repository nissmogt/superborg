"""The chat page (superborg/web/index.html) in headless Chromium, driven through a scripted
BORG_TRANSPORT: the event contract it shares with superborg/interp.py and wasm/trace-session.js
(token ids, "prefill" progress, error events, busy retries, request ids on chat and stop) and the
settings it sends. Skipped without node, Playwright for node, or a browser it can launch."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "superborg", "web", "index.html")

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
  };
})();
"""

DRIVER = r"""
const { chromium } = require("playwright");
const fs = require("fs");
const [PAGE, transport] = [process.argv[2], fs.readFileSync(process.argv[3], "utf8")];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = {};
const check = (name, ok, info) => { results[name] = { ok: !!ok, info }; };
async function open(browser, info = {}, saved = null) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  let pre = `window.__INFO_OVERRIDE = ${JSON.stringify(info)};\n`;
  if (saved) pre += `localStorage.setItem("superborg.ui.settings", ${JSON.stringify(JSON.stringify(saved))});\n`;
  await ctx.addInitScript(pre + transport);
  const page = await ctx.newPage();
  page.errs = [];
  page.on("pageerror", (e) => page.errs.push(e.message));
  await page.goto(PAGE);
  await page.waitForFunction(() => window.__superborg && window.__superborg.ready);
  return { ctx, page };
}
const ask = async (page, q) => { await page.fill("#input", q); await page.keyboard.press("Enter"); };
const idle = async (page) => { await page.waitForFunction(() => !window.__superborg.generating, null, { timeout: 15000 }); await sleep(60); };
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
};
const setup = {
  stride_default_not_pinned: [{ lens_stride: 3, lens_layers: [0, 3, 4] }],
  saved_values_clamped: [{ n_ctx: 2048 }, { temperature: 0, top_p: 1, max_tokens: 4096, seed: 0, lens_stride: 9 }],
};
(async () => {
  let browser;
  try { browser = await chromium.launch(); } catch (e) { console.log("LAUNCH-FAILED " + e.message.split("\n")[0]); return; }
  for (const [name, fn] of Object.entries(cases)) {
    const [info, saved] = setup[name] || [];
    const { ctx, page } = await open(browser, info, saved);
    try { await fn(page); } catch (e) { check(name, false, "threw: " + e.message.split("\n")[0]); }
    if (page.errs.length) check(name, false, { pageErrors: page.errs });
    await ctx.close();
  }
  await browser.close();
  console.log("RESULTS " + JSON.stringify(results));
})();
"""

CASES = ["chosen_by_id", "error_events", "request_ids_and_stop", "stop_while_waiting_for_busy_engine", "prefill_then_new_chat",
         "stride_default_not_pinned", "saved_values_clamped", "selection_labels", "mlp_all_layers",
         "zero_width_tokens_clickable", "stop_during_prefill", "layer_values_and_announce"]


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
    r = subprocess.run([node, str(d / "driver.js"), "file://" + INDEX, str(d / "transport.js")], env=env,
                       capture_output=True, text=True, timeout=300)
    if "LAUNCH-FAILED" in r.stdout:
        pytest.skip("no browser for Playwright: " + r.stdout.strip())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULTS ")]
    assert lines, f"driver failed:\n{r.stdout}\n{r.stderr}"
    return json.loads(lines[-1][len("RESULTS "):])


@pytest.mark.parametrize("case", CASES)
def test_page(results, case):
    res = results.get(case)
    assert res is not None, f"{case} did not run"
    assert res["ok"], json.dumps(res["info"], ensure_ascii=False)
