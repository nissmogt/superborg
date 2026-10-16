"""Package the chat + activation UI with the WebAssembly engine as a static site.

    python wasm/make_artifact.py model.borg out_dir [--page superborg/web/index.html]

Writes to out_dir:
  index.html      the UI page with the WebAssembly transport inlined (no server needed); with
                  --fragment, without <!doctype>/<html>/<head>/<body>, for hosts that wrap pages,
                  and (also with --inline-worker) with worker.js embedded, started from a blob: URL
  worker.js       borg-runtime.js + trace-session.js + worker.js in one file, plus the
                  JavaScript build (build/borg-asm.js from wasm/build-js.sh) when it exists
  borg.wasm       the engine (built with wasm/build.sh if missing)
  model.json      manifest: engine file, model parts and sizes, context length
  model/part-NN.wasm the .borg file split into parts of at most --part-size bytes, each stored as
                  the custom section "borg-part" of an otherwise empty WebAssembly module, so
                  hosts that serve only web file types (.wasm, not .bin) serve the parts

Serve the directory over HTTP (python -m http.server) and open index.html. Browsers do not run
WebAssembly from file:// URLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WASM_DIR = ROOT / "wasm"


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        head = f.read(16)
        if head[:4] != b"BORG":
            raise SystemExit(f"{path}: not a .borg file")
        n = int.from_bytes(head[8:16], "little")
        return json.loads(f.read(n))


def leb128(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def wasm_wrap(payload: bytes, name: bytes = b"borg-part") -> bytes:
    """A valid WebAssembly module whose only content is a custom section holding `payload`."""
    body = leb128(len(name)) + name + payload
    return b"\0asm" + (1).to_bytes(4, "little") + b"\x00" + leb128(len(body)) + body


def bundle_worker(asm: Path | None) -> str:
    parts = []
    for name in ("borg-runtime.js", "trace-session.js", "worker.js"):
        text = (WASM_DIR / name).read_text()
        if name == "worker.js":  # its own scope: its top-level names must not clash with the page's
            text = "(function () {\n" + text + "\n})();"
        parts.append(f"// ---- {name} ----\n" + text)
    if asm is not None:
        parts.insert(0, f"// ---- {asm.name} ----\n" + asm.read_text())
    return "\n".join(parts)


def page_for_static(page: str, transport: str, fragment: bool, worker_src: str | None = None) -> str:
    """Inline the WebAssembly transport ahead of the page's own scripts. With fragment=True, also
    drop the document skeleton (the host supplies doctype, charset and viewport). With worker_src,
    the worker's source goes into the page as an unexecuted script block, so the transport can
    start it from a blob: URL without the host serving (or allowing) a separate script file."""
    if fragment:
        page = re.sub(r"<!doctype[^>]*>", "", page, flags=re.I)
        page = re.sub(r"</?(html|head|body)\b[^>]*>", "", page, flags=re.I)
        page = re.sub(r"<meta\s+charset[^>]*>", "", page, flags=re.I)
        page = re.sub(r'<meta\s+name="viewport"[^>]*>', "", page, flags=re.I)
        page = re.sub(r'<link\s+rel="icon"[^>]*>', "", page, flags=re.I)
    m = re.search(r"<script\b", page, flags=re.I)
    if not m:
        raise SystemExit("the page has no <script>")
    inline = ""
    if worker_src is not None:
        if re.search(r"</script|<!--|<script", worker_src, flags=re.I):
            raise SystemExit("the worker bundle contains text the HTML parser would act on inside <script>")
        inline += '<script type="text/plain" id="borg-worker-src">\n' + worker_src + "\n</script>\n"
    inline += "<script>\n" + transport + "\n</script>\n"
    return page[: m.start()] + inline + page[m.start():]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--page", type=Path, default=ROOT / "superborg" / "web" / "index.html")
    ap.add_argument("--wasm", type=Path, default=ROOT / "build" / "borg.wasm")
    ap.add_argument("--asm", type=Path, default=ROOT / "build" / "borg-asm.js",
                    help="JavaScript build for pages that block WebAssembly (wasm/build-js.sh); bundled if it exists")
    ap.add_argument("--no-asm", action="store_true", help="do not bundle the JavaScript build")
    ap.add_argument("--part-size", type=int, default=14_500_000)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--lens-stride", type=int, default=3)
    ap.add_argument("--name", help="model name shown in the UI (default: the header's source)")
    ap.add_argument("--fragment", action="store_true", help="omit the document skeleton from index.html")
    ap.add_argument("--inline-worker", action="store_true",
                    help="embed worker.js in index.html (implied by --fragment): hosts that do not serve "
                         "the page's own scripts to a Worker still run the engine")
    a = ap.parse_args(argv)

    header = read_header(a.model)
    if not a.wasm.exists():
        subprocess.run([str(WASM_DIR / "build.sh"), str(a.wasm)], check=True)
    out = a.out
    (out / "model").mkdir(parents=True, exist_ok=True)
    for old in (out / "model").glob("part-*.*"):
        old.unlink()

    size = a.model.stat().st_size
    digest = hashlib.sha256()
    parts = []
    with open(a.model, "rb") as f:
        i = 0
        while True:
            chunk = f.read(a.part_size)
            if not chunk:
                break
            digest.update(chunk)
            rel = f"model/part-{i:02d}.wasm"
            (out / rel).write_bytes(wasm_wrap(chunk))
            parts.append({"path": rel, "size": len(chunk)})
            i += 1
    manifest = {
        "wasm": "borg.wasm",
        "n_ctx": a.n_ctx,
        "lens_stride": a.lens_stride,
        "model": {"name": a.name or header.get("source") or a.model.stem, "id": digest.hexdigest()[:16],
                  "size": size, "quant": header.get("quant"), "wrap": "wasm-custom-section", "parts": parts},
    }
    (out / "model.json").write_text(json.dumps(manifest, indent=1) + "\n")
    shutil.copyfile(a.wasm, out / "borg.wasm")
    asm = None if a.no_asm or not a.asm.exists() else a.asm
    bundle = bundle_worker(asm)
    (out / "worker.js").write_text(bundle)
    transport = (WASM_DIR / "wasm-transport.js").read_text()
    inline = bundle if a.inline_worker or a.fragment else None
    (out / "index.html").write_text(page_for_static(a.page.read_text(), transport, a.fragment, inline))
    print(f"wrote {out}: {len(parts)} model parts, {size / 1e6:.1f} MB"
          + ("" if asm else " (no JavaScript fallback: run wasm/build-js.sh)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
