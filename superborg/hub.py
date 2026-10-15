"""Download model files from the Hugging Face Hub with parallel HTTP range requests.

Uses only the standard library. Large files are split into chunks that download
concurrently; each chunk retries independently, so one slow connection does not
stall the whole transfer.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
DEFAULT_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
CHUNK = 8 << 20


def _request(url: str, headers: dict | None = None, timeout: float = 60.0):
    req = urllib.request.Request(url, headers=headers or {})
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith(HF_ENDPOINT):
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "superborg/0.1")
    return urllib.request.urlopen(req, timeout=timeout)


def _resolve(url: str) -> tuple[str, int]:
    """Follow redirects and return (final_url, size) using a 1-byte range probe."""
    with _request(url, {"Range": "bytes=0-0"}) as r:
        final = r.geturl()
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return final, int(cr.rsplit("/", 1)[1])
        length = r.headers.get("Content-Length")
        return final, int(length) if length else -1


def _fetch_range(url: str, start: int, end: int, path: str, retries: int = 8) -> None:
    for attempt in range(retries):
        try:
            with _request(url, {"Range": f"bytes={start}-{end}"}, timeout=30) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(data)
            return
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404) or attempt == retries - 1:
                raise
        except Exception:  # noqa: BLE001 - retry any transport error
            if attempt == retries - 1:
                raise
        time.sleep(min(2 ** attempt, 16))


def download_file(repo: str, filename: str, dest_dir: str, workers: int = 8, revision: str = "main",
                  quiet: bool = False) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, filename)
    url = f"{HF_ENDPOINT}/{repo}/resolve/{revision}/{filename}"
    final, size = _resolve(url)
    if os.path.exists(path) and size >= 0 and os.path.getsize(path) == size:
        return path
    tmp = path + ".part"
    if size < 0 or size <= CHUNK:
        with _request(final) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
        return path
    with open(tmp, "wb") as f:
        f.truncate(size)
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_range, final, s, e, tmp) for s, e in ranges]
        for fut in cf.as_completed(futs):
            fut.result()
            done += 1
            if not quiet:
                got = min(done * CHUNK, size)
                rate = got / 1e6 / max(time.time() - t0, 1e-6)
                print(f"\r{filename}: {got / 1e6:.0f}/{size / 1e6:.0f} MB ({rate:.1f} MB/s)",
                      end="", file=sys.stderr, flush=True)
    if not quiet:
        print(file=sys.stderr)
    os.replace(tmp, path)
    return path


def list_files(repo: str, revision: str = "main") -> list[str]:
    with _request(f"{HF_ENDPOINT}/api/models/{repo}/revision/{revision}") as r:
        info = json.load(r)
    return [s["rfilename"] for s in info.get("siblings", [])]


def pull(repo: str, dest_root: str, workers: int = 8, quiet: bool = False) -> str:
    """Download config, tokenizer and safetensors weights of `repo` into dest_root/<owner>_<name>."""
    dest = os.path.join(dest_root, repo.replace("/", "_"))
    files = list_files(repo)
    wanted = [f for f in files if f in DEFAULT_FILES or (f.endswith(".safetensors") and "/" not in f)
              or f == "model.safetensors.index.json"]
    if not any(f.endswith(".safetensors") for f in wanted):
        raise RuntimeError(f"{repo} has no top-level .safetensors weights")
    for f in wanted:
        download_file(repo, f, dest, workers=workers, quiet=quiet)
    return dest


if __name__ == "__main__":
    for repo in sys.argv[2:]:
        print(pull(repo, sys.argv[1]))
