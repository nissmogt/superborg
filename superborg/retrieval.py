"""Local retrieval: Okapi BM25 over chunks of the user's own files (pure Python).

    idx = Index()
    idx.add_path("~/notes")                  # .txt/.md/.py/... files, recursively
    for hit in idx.search("vacation policy", k=3):
        print(hit.score, hit.source, hit.text[:80])
    idx.save("notes.idx.json"); idx = Index.load("notes.idx.json")

BM25 needs no model and no embeddings, is fast enough for tens of thousands of chunks,
and is strong at the exact-term matching that small models are worst at recalling.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

__all__ = ["Hit", "Chunk", "Index", "tokenize", "chunk_text", "format_context", "DEFAULT_EXTS"]

DEFAULT_EXTS = (".txt", ".md", ".markdown", ".rst", ".py", ".c", ".h", ".cc", ".cpp", ".hpp", ".js", ".ts",
                ".java", ".go", ".rs", ".rb", ".sh", ".toml", ".yaml", ".yml", ".json", ".csv", ".tex",
                ".html", ".htm", ".css", ".sql", ".org", ".ini", ".cfg")
_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
              ".pytest_cache", "build", "dist", ".tox", ".idea", ".vscode"}

# Unicode letters/digits; underscores split identifiers so snake_case parts match words.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased runs of unicode letters and digits."""
    return _WORD.findall(text.lower())


@dataclass
class Hit:
    text: str
    source: str
    score: float
    line: int = 0     # 1-based line where the chunk starts in its source (0 if unknown)
    chunk: int = -1   # chunk id inside the index


@dataclass
class Chunk:
    text: str
    source: str
    line: int = 0


def _split_long(par: str, size: int) -> list[str]:
    """Split one oversized paragraph at sentence/line boundaries, hard-cutting as a last resort."""
    pieces = re.split(r"(?<=[.!?。])\s+|\n", par)
    out, cur = [], ""
    for p in pieces:
        while len(p) > size:  # e.g. minified code or one huge sentence
            if cur:
                out.append(cur)
                cur = ""
            cut = p.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            out.append(p[:cut])
            p = p[cut:].lstrip()
        if cur and len(cur) + 1 + len(p) > size:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[tuple[str, int]]:
    """Paragraph-aware chunks of about `size` characters as (text, start_line) pairs.

    Paragraphs (blank-line separated) are packed greedily; a chunk repeats up to `overlap`
    characters of trailing paragraphs from the previous chunk so facts that straddle a
    boundary stay retrievable.
    """
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"chunk size must be a positive int, got {size!r}")  # size 0 never ends
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras: list[tuple[str, int]] = []
    line = 1
    for block in re.split(r"(\n(?:[ \t]*\n)+)", text):
        if block.strip():
            lead = len(block) - len(block.lstrip("\n"))
            body = block.strip("\n")
            if len(body) > size:
                sub_line = line + lead
                for piece in _split_long(body, size):
                    paras.append((piece, sub_line))
            else:
                paras.append((body, line + lead))
        line += block.count("\n")
    chunks: list[tuple[str, int]] = []
    cur: list[tuple[str, int]] = []
    cur_len = 0
    for p, ln in paras:
        if cur and cur_len + 2 + len(p) > size:
            chunks.append(("\n\n".join(x for x, _ in cur), cur[0][1]))
            # carry trailing paragraphs (up to `overlap` chars) into the next chunk
            carry: list[tuple[str, int]] = []
            n = 0
            for q in reversed(cur):
                if n + len(q[0]) > overlap:
                    break
                carry.insert(0, q)
                n += len(q[0]) + 2
            while carry and n + len(p) > size:  # the overlap must not push a chunk past `size`
                n -= len(carry.pop(0)[0]) + 2
            cur, cur_len = carry, n
        cur.append((p, ln))
        cur_len += len(p) + 2
    if cur:
        chunks.append(("\n\n".join(x for x, _ in cur), cur[0][1]))
    return chunks


class Index:
    """An in-memory BM25 (Okapi) index over text chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, chunk_size: int = 800, overlap: int = 150):
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
            raise ValueError(f"overlap must be a non-negative int, got {overlap!r}")
        for name, v in (("k1", k1), ("b", b)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {v!r}")
        self.k1, self.b = k1, b
        self.chunk_size, self.overlap = chunk_size, overlap
        self.chunks: list[Chunk] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(chunk, tf)]
        self._lengths: list[int] = []
        self._total_len = 0

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- building ---------------------------------------------------------------------
    def _add_chunk(self, text: str, source: str, line: int) -> None:
        cid = len(self.chunks)
        self.chunks.append(Chunk(text, source, line))
        terms = Counter(tokenize(text))
        for t, tf in terms.items():
            self._postings[t].append((cid, tf))
        n = sum(terms.values())
        self._lengths.append(n)
        self._total_len += n

    def add_text(self, text: str, source: str = "<text>") -> int:
        """Chunk and index `text`. Returns the number of chunks added."""
        parts = chunk_text(text, self.chunk_size, self.overlap)
        for t, ln in parts:
            self._add_chunk(t, source, ln)
        return len(parts)

    def add_path(self, path: str, exts: Iterable[str] = DEFAULT_EXTS, max_bytes: int = 2_000_000) -> int:
        """Index a file or every matching file under a directory. Returns chunks added."""
        path = os.path.expanduser(path)
        exts = tuple(e.lower() for e in exts)
        files: list[str] = []
        if os.path.isfile(path):
            files = [path]
        else:
            for root, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
                files.extend(os.path.join(root, n) for n in sorted(names)
                             if n.lower().endswith(exts) and not n.startswith("."))
        added = 0
        for f in files:
            raw = _read_regular(f, max_bytes)
            if raw is None:
                continue
            if b"\0" in raw[:4096]:  # binary
                continue
            added += self.add_text(raw.decode("utf-8", errors="replace"), source=f)
        return added

    # ---- querying ---------------------------------------------------------------------
    def idf(self, term: str) -> float:
        """BM25 idf with the +1 inside the log (never negative, as in Lucene)."""
        df = len(self._postings.get(term, ()))
        n = len(self.chunks)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """The k best chunks for `query`, best first (chunks without any query term are omitted)."""
        if not self.chunks or k <= 0:
            return []
        avg = self._total_len / max(1, len(self.chunks))
        k1, b = self.k1, self.b
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in Counter(tokenize(query)).items():
            posting = self._postings.get(term)
            if not posting:
                continue
            w = self.idf(term) * qtf
            for cid, tf in posting:
                norm = k1 * (1.0 - b + b * self._lengths[cid] / avg)
                scores[cid] += w * tf * (k1 + 1.0) / (tf + norm)
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(self.chunks[c].text, self.chunks[c].source, s, self.chunks[c].line, c) for c, s in best]

    # ---- persistence ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Write the chunks and settings as JSON (postings are rebuilt on load)."""
        data = {"format": "superborg-bm25", "version": 1, "k1": self.k1, "b": self.b,
                "chunk_size": self.chunk_size, "overlap": self.overlap,
                "chunks": [asdict(c) for c in self.chunks]}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Index":
        """Read an index written by save(). Anything else raises ValueError."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("format") != "superborg-bm25":
            raise ValueError(f"{path} is not a superborg BM25 index")
        try:
            idx = cls(k1=data["k1"], b=data["b"], chunk_size=data["chunk_size"], overlap=data["overlap"])
            chunks = data["chunks"]
            if not isinstance(chunks, list):
                raise TypeError("chunks is not a list")
            for c in chunks:
                if not isinstance(c, dict) or not isinstance(c.get("text"), str) or \
                        not isinstance(c.get("source"), str):
                    raise TypeError(f"bad chunk {str(c)[:80]}")
                line = c.get("line", 0)
                if isinstance(line, bool) or not isinstance(line, int):
                    raise TypeError(f"bad line number {line!r}")
                idx._add_chunk(c["text"], c["source"], line)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path} is not a valid superborg BM25 index: {e}") from e
        return idx


def _read_regular(path: str, max_bytes: int) -> bytes | None:
    """Contents of a regular file of at most max_bytes, else None. FIFOs and devices (a
    symlink to /dev/zero reports size 0) would block or never end, so they are skipped;
    the open does not block and the read is bounded even if the file changes meanwhile."""
    nonblock = getattr(os, "O_NONBLOCK", 0)  # opening a FIFO without a writer would block
    try:
        fd = os.open(path, os.O_RDONLY | nonblock | getattr(os, "O_BINARY", 0))
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            return None
        if nonblock:
            os.set_blocking(fd, True)
        with os.fdopen(fd, "rb") as fh:
            fd = -1  # owned by fh now
            raw = fh.read(max_bytes + 1)
    except OSError:
        return None
    finally:
        if fd >= 0:
            os.close(fd)
    return raw if len(raw) <= max_bytes else None


def format_context(hits: list[Hit], max_chars: int = 3000) -> str:
    """Numbered context block for a prompt: "[1] (source:line)\\ntext" per hit, within max_chars."""
    out, used = [], 0
    for i, h in enumerate(hits, 1):
        src = os.path.basename(h.source) if os.sep in h.source else h.source
        if h.line:
            src += f":{h.line}"
        block = f"[{i}] ({src})\n{h.text.strip()}"
        if used + len(block) > max_chars:
            room = max_chars - used
            if room < 200:
                break
            block = block[:room].rstrip() + " ..."
        out.append(block)
        used += len(block) + 2
    return "\n\n".join(out)
