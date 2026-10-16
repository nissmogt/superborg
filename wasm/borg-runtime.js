// The borg engine compiled to WebAssembly (see wasm/build.sh), wrapped for JavaScript.
// Works in a browser (main thread or Web Worker) and in Node >= 18 (for tests).
//
//   const borg = await Borg.create(wasmBytes);
//   const model = borg.loadModel(fileBytes);          // Uint8Array of a .borg file
//   const ctx = model.context({ nCtx: 2048 });
//   const logits = ctx.decode(model.tokenize("Hello"), "last");
//
// Memory views are re-created on every access because WebAssembly memory growth detaches
// the old ArrayBuffer. Addresses are unsigned: a heap past 2 GiB gives i32 results that JS reads
// as negative numbers, so every pointer from the module goes through `>>> 0` (or `>>> 2` to
// index 32-bit views).

/* global WebAssembly, TextDecoder, TextEncoder, performance */
(function (root) {
  "use strict";

  const ENOSYS = 52, EBADF = 8;

  function wasiShim(getMemory, log) {
    const dec = new TextDecoder();
    const view = () => new DataView(getMemory().buffer);
    return {
      clock_time_get(id, precision, out) {
        const ns = BigInt(Math.round((typeof performance !== "undefined" ? performance.now() : Date.now()) * 1e6));
        view().setBigUint64(out >>> 0, ns, true);
        return 0;
      },
      fd_write(fd, iovs, iovsLen, nwritten) {
        const v = view();
        let n = 0, text = "";
        iovs >>>= 0;
        for (let i = 0; i < iovsLen; i++) {
          const p = v.getUint32(iovs + i * 8, true), l = v.getUint32(iovs + i * 8 + 4, true);
          text += dec.decode(new Uint8Array(getMemory().buffer, p, l));
          n += l;
        }
        if (text.trim()) log(text.replace(/\n$/, ""));
        v.setUint32(nwritten >>> 0, n, true);
        return 0;
      },
      fd_close: () => 0,
      fd_seek: () => ENOSYS,
      fd_read: () => ENOSYS,
      fd_fdstat_get: () => EBADF,
      fd_fdstat_set_flags: () => ENOSYS,
      fd_filestat_get: () => ENOSYS,
      fd_prestat_get: () => EBADF,       // no preopened directories
      fd_prestat_dir_name: () => EBADF,
      path_open: () => ENOSYS,
      proc_exit(code) { throw new Error("borg exited with code " + code); },
    };
  }

  class Borg {
    /** `wasmBytes`: borg.wasm. With `options.asm` (borgAsmInstantiate from build-js.sh) and no
     *  bytes, run the JavaScript build instead, for pages that may not compile WebAssembly. */
    static async create(wasmBytes, options = {}) {
      let memory = null;
      const log = options.log || ((s) => console.warn("[borg]", s));
      const wasi = wasiShim(() => memory, log);
      let exports;
      if (wasmBytes) {
        const { instance } = await WebAssembly.instantiate(wasmBytes, { wasi_snapshot_preview1: wasi });
        exports = instance.exports;
      } else {
        // wasm2js splits 64-bit arguments in two 32-bit halves ("legalized" imports)
        const legal = Object.assign({}, wasi, {
          clock_time_get: (id, precLo, precHi, out) => wasi.clock_time_get(id, 0n, out),
          fd_seek: () => ENOSYS,
          path_open: () => ENOSYS,
        });
        exports = options.asm({ wasi_snapshot_preview1: legal, env: { setTempRet0: () => {} } });
      }
      memory = exports.memory;
      if (exports._initialize) exports._initialize();
      return new Borg(exports, wasmBytes ? "" : " (JavaScript build)");
    }

    constructor(exports, note = "") {
      this.x = exports;
      this.enc = new TextEncoder();
      this.dec = new TextDecoder("utf-8", { fatal: false });
      this.simd = this.str(this.x.borg_simd()) + note;
      this.version = this.str(this.x.borg_version());
    }

    get u8() { return new Uint8Array(this.x.memory.buffer); }
    get i32() { return new Int32Array(this.x.memory.buffer); }
    get f32() { return new Float32Array(this.x.memory.buffer); }
    get f64() { return new Float64Array(this.x.memory.buffer); }

    malloc(n) {
      const p = this.x.malloc(n) >>> 0;
      if (!p) throw new Error(`out of WebAssembly memory allocating ${n} bytes`);
      return p;
    }
    free(p) { if (p) this.x.free(p); }

    str(ptr, len) {
      ptr >>>= 0;
      const u8 = this.u8;
      if (len === undefined) { len = 0; while (u8[ptr + len]) len++; }
      return this.dec.decode(u8.subarray(ptr, ptr + len));
    }

    /** Copy a .borg file into WebAssembly memory (64-byte aligned) and load it.
     *  `chunks` is a Uint8Array or an array of Uint8Array parts of the file. */
    loadModel(chunks) {
      if (chunks instanceof Uint8Array) chunks = [chunks];
      const r = this.reserveModel(chunks.reduce((s, c) => s + c.length, 0));
      let off = 0;
      for (const c of chunks) { r.write(c, off); off += c.length; }
      return r.load();
    }

    /** Reserve memory for a .borg file of `size` bytes so its parts can be copied in as they
     *  arrive (no second copy of the whole file): write(bytes, offset) each part, then load(). */
    reserveModel(size) {
      const raw = this.malloc(size + 64);
      const base = raw + ((64 - (raw % 64)) % 64);
      return {
        base, size,
        write: (bytes, offset) => {
          if (offset < 0 || offset + bytes.length > size) throw new Error("model part out of range");
          this.u8.set(bytes, base + offset);
        },
        load: () => {
          const err = this.malloc(512);
          const m = this.x.borg_model_load_memory(base, size, err, 512) >>> 0;
          const msg = this.str(err);
          this.free(err);
          if (!m) { this.free(raw); throw new Error("cannot load model: " + msg); }
          return new Model(this, m, raw, size);
        },
        release: () => this.free(raw),
      };
    }

    pool(nThreads = 1) { return this.x.borg_pool_new(nThreads) >>> 0; }
  }

  class Model {
    constructor(borg, ptr, raw, size) {
      this.b = borg;
      this.ptr = ptr;
      this.raw = raw;
      this.size = size;
      this.header = JSON.parse(borg.str(borg.x.borg_model_header(ptr)));
      this.config = this.header.config;
      this.tokenizerInfo = this.header.tokenizer;
      this.vocabSize = this.config.vocab_size;
      this.nVocab = this.tokenizerInfo.n_vocab || this.vocabSize;
      this.pool = borg.pool(1);
      this._pieces = new Map();
      this._added = null;
    }

    get name() { return this.header.source || "model"; }

    tokenize(text, addBos = false, special = true) {
      const b = this.b;
      if (this.tokenizerInfo.nfc) text = this.nfc(text, special);
      const bytes = b.enc.encode(text);
      const tp = b.malloc(bytes.length + 1);
      b.u8.set(bytes, tp);
      let cap = bytes.length + 16;
      for (;;) {
        const out = b.malloc(cap * 4);
        const n = b.x.borg_tokenize(this.ptr, tp, bytes.length, addBos ? 1 : 0, special ? 1 : 0, out, cap);
        if (n >= 0) {
          const toks = Array.from(b.i32.subarray(out >>> 2, (out >>> 2) + n));
          b.free(out); b.free(tp);
          return toks;
        }
        b.free(out);
        cap = -n;
      }
    }

    // NFC only between added tokens, as Hugging Face normalizes (see superborg/engine.py)
    nfc(text, special) {
      if (!special) return text.normalize("NFC");
      if (this._added === null) {
        const pieces = [...new Set((this.tokenizerInfo.added_tokens || []).map((t) => this.tokenText(t)))]
          .filter(Boolean).sort((a, b) => b.length - a.length);
        const esc = pieces.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
        this._added = esc.length ? new RegExp("(" + esc.join("|") + ")") : false;
      }
      if (!this._added) return text.normalize("NFC");
      return text.split(this._added).map((p, i) => (i % 2 ? p : p.normalize("NFC"))).join("");
    }

    tokenBytes(tok) {
      let bytes = this._pieces.get(tok);
      if (!bytes) {
        const b = this.b;
        const buf = b.malloc(512);
        let n = b.x.borg_token_bytes(this.ptr, tok, buf, 512);
        if (n > 512) n = 512;
        bytes = b.u8.slice(buf, buf + n);
        b.free(buf);
        this._pieces.set(tok, bytes);
      }
      return bytes;
    }

    tokenText(tok) { return this.b.dec.decode(this.tokenBytes(tok)); }
    isEog(tok) { return !!this.b.x.borg_is_eog(this.ptr, tok); }

    context({ nCtx = 2048, nBatch = 0, kvF16 = true } = {}) {
      const b = this.b;
      const p = b.malloc(12);
      b.i32.set([nCtx, nBatch, kvF16 ? 1 : 0], p >>> 2);
      const c = b.x.borg_ctx_new(this.ptr, this.pool, p) >>> 0;
      b.free(p);
      if (!c) throw new Error("cannot allocate a context (try a smaller one)");
      return new Context(this, c);
    }
  }

  class Context {
    constructor(model, ptr) {
      this.m = model;
      this.b = model.b;
      this.ptr = ptr;
      this.nCtx = this.b.x.borg_ctx_n_ctx(ptr);
      this._logits = 0;
      this._logitsCap = 0;
    }

    get nPast() { return this.b.x.borg_ctx_n_past(this.ptr); }
    tokens() {
      const n = this.nPast, p = this.b.x.borg_ctx_tokens(this.ptr) >>> 2;
      return Array.from(this.b.i32.subarray(p, p + n));
    }
    truncate(n) { this.b.x.borg_ctx_truncate(this.ptr, n); }

    _logitsBuf(n) {
      const need = n * this.m.vocabSize * 4;
      if (need > this._logitsCap) {
        this.b.free(this._logits);
        this._logits = this.b.malloc(need);
        this._logitsCap = need;
      }
      return this._logits;
    }

    /** Append tokens; returns a Float32Array copy of the last logits (mode "last") or null. */
    decode(tokens, mode = "last") {
      const b = this.b, n = tokens.length;
      if (!n) return null;
      const tp = b.malloc(n * 4);
      b.i32.set(tokens, tp >>> 2);
      const lp = mode ? this._logitsBuf(1) : 0;
      const rc = b.x.borg_decode(this.ptr, tp, n, lp, mode ? 1 : 0);
      b.free(tp);
      if (rc !== 0) throw new Error(rc === -1 ? "the context is full" : "decode failed (" + rc + ")");
      return mode ? b.f32.slice(lp >>> 2, (lp >>> 2) + this.m.vocabSize) : null;
    }

    /** Keep the cached prefix shared with `tokens`, evaluate the rest. Returns tokens reused. */
    sync(tokens) {
      const cached = this.tokens();
      let common = 0;
      while (common < cached.length && common < tokens.length && cached[common] === tokens[common]) common++;
      this.truncate(common);
      if (common < tokens.length) this.decode(tokens.slice(common), null);
      return common;
    }

    enableTrace(on = true) {
      if (this.b.x.borg_ctx_enable_trace(this.ptr, on ? 1 : 0) !== 0) throw new Error("cannot allocate trace buffers");
    }

    /** Copies of the trace arrays (see borg_ctx_trace_* in engine/borg.h). */
    trace() {
      const b = this.b, x = b.x, c = this.m.config;
      const pos = x.borg_ctx_trace_pos(this.ptr);
      if (pos < 0) return null;
      const L = c.n_layers, D = c.dim, H = c.n_heads, F = c.hidden_dim;
      const f = b.f32;
      const rp = x.borg_ctx_trace_resid(this.ptr, 0) >>> 2;
      const np = x.borg_ctx_trace_norms(this.ptr) >>> 2;
      const npos = pos + 1;
      const ap = x.borg_ctx_trace_attn(this.ptr, 0, 0, 0) >>> 2;
      const attn = new Float32Array(L * H * npos);
      for (let l = 0; l < L; l++)
        for (let h = 0; h < H; h++) {
          const src = ap + (l * H + h) * this.nCtx;
          attn.set(f.subarray(src, src + npos), (l * H + h) * npos);
        }
      const mp = x.borg_ctx_trace_mlp(this.ptr, 0) >>> 2;
      return {
        pos, nPos: npos,
        resid: f.slice(rp, rp + (L + 1) * D),
        norms: f.slice(np, np + L * 3),
        attn,
        mlp: f.slice(mp, mp + L * F),
      };
    }

    /** Logit lens rows for the listed layers (Float32Array of layers.length * vocab). */
    lens(layers) {
      const b = this.b, n = layers.length;
      const lp = b.malloc(n * 4);
      b.i32.set(layers, lp >>> 2);
      const out = this._logitsBuf(n);
      const rc = b.x.borg_ctx_lens(this.ptr, lp, n, out);
      b.free(lp);
      if (rc !== 0) throw new Error("lens failed (" + rc + ")");
      return b.f32.slice(out >>> 2, (out >>> 2) + n * this.m.vocabSize);
    }

    /** The logit lens summarized inside the engine (borg_ctx_lens_summary): per listed layer the k
     *  most likely tokens (topIds / topLogits, [n * k], logit descending then id ascending), the
     *  logit and rank (strictly greater logits) of `chosen`, and the row's log-sum-exp (lse,
     *  Float64Array), so a probability is exp(logit - lse). Only the first nVocab logits take part
     *  (default: the tokenizer's vocabulary). The [n, vocab] logits never leave the engine. */
    lensSummary(layers, k, chosen, nVocab = this.m.nVocab) {
      const b = this.b, n = layers.length;
      const lp = b.malloc(n * 4), ids = b.malloc(n * k * 4), vals = b.malloc(n * k * 4);
      const zc = b.malloc(n * 4), rank = b.malloc(n * 4), lse = b.malloc(n * 8);
      b.i32.set(layers, lp >>> 2);
      const rc = b.x.borg_ctx_lens_summary(this.ptr, lp, n, nVocab, k, chosen, ids, vals, zc, rank, lse);
      const out = rc === 0 ? {
        topIds: b.i32.slice(ids >>> 2, (ids >>> 2) + n * k), topLogits: b.f32.slice(vals >>> 2, (vals >>> 2) + n * k),
        chosenLogit: b.f32.slice(zc >>> 2, (zc >>> 2) + n), chosenRank: b.i32.slice(rank >>> 2, (rank >>> 2) + n),
        lse: b.f64.slice(lse >>> 3, (lse >>> 3) + n),
      } : null;
      for (const p of [lp, ids, vals, zc, rank, lse]) b.free(p);
      if (!out) throw new Error("lens summary failed (" + rc + ")");
      return out;
    }
  }

  root.Borg = Borg;
  if (typeof module !== "undefined" && module.exports) module.exports = { Borg };
})(typeof self !== "undefined" ? self : globalThis);
