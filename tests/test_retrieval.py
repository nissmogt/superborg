"""Tests for superborg.retrieval: tokenizer, chunking and BM25 ranking."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap

import pytest

from superborg.retrieval import Hit, Index, chunk_text, format_context, tokenize


def test_tokenize():
    assert tokenize("Hello, Wörld! snake_case v2.5 東京") == ["hello", "wörld", "snake", "case", "v2", "5", "東京"]
    assert tokenize("") == []


def test_chunks_respect_size_and_paragraphs():
    paras = [f"Paragraph {i} " + "word " * 30 for i in range(12)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, size=400, overlap=0)
    assert len(chunks) > 1
    for c, _ in chunks:
        assert len(c) <= 400
        # no paragraph is cut in the middle when paragraphs fit
        for piece in c.split("\n\n"):
            assert piece.strip() in [p.strip() for p in paras]
    assert "".join(c for c, _ in chunks).count("Paragraph") == 12


def test_chunk_overlap_repeats_trailing_paragraph():
    text = "\n\n".join(f"P{i} " + "x" * 90 for i in range(10))
    chunks = chunk_text(text, size=300, overlap=120)
    assert len(chunks) >= 3
    for (a, _), (b, _) in zip(chunks, chunks[1:]):
        last_par = a.split("\n\n")[-1]
        assert b.startswith(last_par)
        assert len(b) <= 300


def test_long_paragraph_is_split():
    text = "Sentence number one is here. " * 100
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) >= 8
    assert all(len(c) <= 300 for c, _ in chunks)


def test_chunk_line_numbers():
    text = "alpha\n\nbeta line\nmore\n\n\n\ngamma"
    idx = Index(chunk_size=8, overlap=0)
    idx.add_text(text, "f.txt")
    lines = {c.text.split()[0]: c.line for c in idx.chunks}
    assert lines == {"alpha": 1, "beta": 3, "line": 3, "more": 3, "gamma": 8}  # split pieces keep the paragraph start


def _toy_index() -> Index:
    idx = Index()
    idx.add_text("The cat sat on the mat. Cats like warm places.", "cats.txt")
    idx.add_text("Dogs are loyal. The dog barked at the mailman.", "dogs.txt")
    idx.add_text("Python is a programming language. The interpreter runs code.", "python.md")
    idx.add_text("The the the the the and and of of of a a a.", "stop.txt")
    return idx


def test_bm25_ranks_relevant_document_first():
    idx = _toy_index()
    hits = idx.search("dog barked", k=3)
    assert hits[0].source == "dogs.txt"
    assert all(isinstance(h, Hit) for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)
    assert idx.search("programming language")[0].source == "python.md"


def test_bm25_rare_terms_weigh_more_than_common_ones():
    idx = _toy_index()
    # "the" is everywhere, "mailman" in one document only
    assert idx.idf("mailman") > idx.idf("the") > 0
    assert idx.search("the mailman", k=1)[0].source == "dogs.txt"


def test_bm25_matches_reference_formula():
    idx = Index(k1=1.5, b=0.75)
    docs = ["apple banana apple", "banana cherry", "cherry cherry date elderberry"]
    for i, d in enumerate(docs):
        idx.add_text(d, f"d{i}")
    lens = [len(d.split()) for d in docs]
    avg = sum(lens) / len(lens)

    def ref(query, i):
        s = 0.0
        terms = docs[i].split()
        for q in query.split():
            df = sum(q in d.split() for d in docs)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = terms.count(q)
            s += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * lens[i] / avg))
        return s
    for q in ["apple", "banana cherry", "date", "cherry"]:
        got = {h.source: h.score for h in idx.search(q, k=10)}
        for i in range(3):
            want = ref(q, i)
            assert got.get(f"d{i}", 0.0) == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_length_normalization_prefers_focused_chunk():
    idx = Index()
    idx.add_text("solar panel " + "filler words here " * 40, "long")
    idx.add_text("solar panel efficiency", "short")
    assert idx.search("solar panel", k=1)[0].source == "short"


def test_no_match_and_empty():
    idx = _toy_index()
    assert idx.search("zebra quantum") == []
    assert Index().search("anything") == []
    assert idx.search("cat", k=0) == []


def test_add_path_dir_skips_binary_hidden_and_other_types(tmp_path):
    (tmp_path / "notes.md").write_text("# Trip\n\nThe flight to Lisbon leaves at 9am.", encoding="utf-8")
    (tmp_path / "code.py").write_text("def lisbon_weather():\n    return 'sunny'\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\x00\x00 lisbon")
    (tmp_path / "data.txt").write_bytes(b"lisbon\x00\x01binary")
    (tmp_path / ".hidden.md").write_text("lisbon secret", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "more.txt").write_text("Lisbon is the capital of Portugal.", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.txt").write_text("lisbon in git", encoding="utf-8")
    idx = Index()
    n = idx.add_path(str(tmp_path))
    assert n == 3
    sources = {os.path.basename(h.source) for h in idx.search("lisbon", k=10)}
    assert sources == {"notes.md", "code.py", "more.txt"}
    assert idx.search("capital of portugal", k=1)[0].source.endswith("more.txt")
    assert Index().add_path(str(sub / "more.txt")) == 1


def test_save_load_roundtrip(tmp_path):
    idx = _toy_index()
    p = str(tmp_path / "idx.json")
    idx.save(p)
    idx2 = Index.load(p)
    assert len(idx2) == len(idx)
    for q in ["cat mat", "dog", "interpreter code"]:
        a, b = idx.search(q, k=3), idx2.search(q, k=3)
        assert [(h.source, round(h.score, 9)) for h in a] == [(h.source, round(h.score, 9)) for h in b]
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        Index.load(str(tmp_path / "bad.json"))


# ---- regressions from the adversarial review -------------------------------------------

_RETRIEVAL_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superborg", "retrieval.py")


def _in_child(body: str, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """Run `body` with superborg.retrieval loaded as `m`, under a 1 GiB address-space limit and
    a timeout: the defects these tests guard against hang or eat memory without bound."""
    script = textwrap.dedent("""
        import importlib.util, os, sys
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        except (ImportError, ValueError, OSError):
            pass
        spec = importlib.util.spec_from_file_location("borg_retrieval", sys.argv[1])
        m = importlib.util.module_from_spec(spec); sys.modules["borg_retrieval"] = m; spec.loader.exec_module(m)
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script, _RETRIEVAL_PY, *args], capture_output=True, text=True,
                          timeout=timeout)


@pytest.mark.skipif(not hasattr(os, "mkfifo") or not os.path.exists("/dev/zero"), reason="needs FIFOs, /dev/zero")
def test_add_path_skips_fifos_and_devices(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta", encoding="utf-8")
    os.mkfifo(tmp_path / "inbox.txt")  # nobody writes: opening it for reading would block forever
    os.symlink("/dev/zero", tmp_path / "notes.txt")  # reports size 0, never ends
    os.mkfifo(tmp_path / "zfeed.txt")  # fed with more than max_bytes
    out = _in_child("""
        import threading
        d = sys.argv[2]
        def feed():
            try:
                with open(os.path.join(d, "zfeed.txt"), "w") as f:
                    f.write("gamma " * 1000)
            except OSError:
                pass
        threading.Thread(target=feed, daemon=True).start()
        idx = m.Index()
        n = idx.add_path(d, max_bytes=1000)
        print(n, sorted({os.path.basename(c.source) for c in idx.chunks}))
    """, str(tmp_path))
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "1 ['a.txt']"


def test_add_path_reads_at_most_max_bytes(tmp_path):
    (tmp_path / "big.txt").write_text("word " * 1000, encoding="utf-8")
    (tmp_path / "small.txt").write_text("word", encoding="utf-8")
    assert Index().add_path(str(tmp_path), max_bytes=100) == 1


def test_invalid_chunk_sizes_are_rejected():
    for kw in ({"chunk_size": 0}, {"chunk_size": -5}, {"chunk_size": 1.5}, {"overlap": -1}, {"k1": float("nan")},
               {"b": -1}):
        with pytest.raises(ValueError):
            Index(**kw)
    for size in (0, -3):
        with pytest.raises(ValueError):
            chunk_text("hello world", size)
    assert Index(chunk_size=1).add_text("hi yo") > 0  # the smallest valid size still works


def test_chunk_size_zero_cannot_loop_forever(tmp_path):
    p = tmp_path / "zero.json"
    p.write_text(json.dumps({"format": "superborg-bm25", "version": 1, "k1": 1.5, "b": 0.75, "chunk_size": 0,
                             "overlap": 0, "chunks": []}), encoding="utf-8")
    out = _in_child("""
        results = []
        for f in (lambda: m.Index.load(sys.argv[2]).add_text("hello world"),
                  lambda: m.Index(chunk_size=0).add_text("hello world"),
                  lambda: m.chunk_text("hello world", 0)):
            try:
                f()
                results.append("ok")
            except ValueError:
                results.append("ValueError")
        print(results)
    """, str(p), timeout=20)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "['ValueError', 'ValueError', 'ValueError']"


@pytest.mark.parametrize("content", [
    "[]", "null", "\"x\"", '{"format": "superborg-bm25"}',
    '{"format": "superborg-bm25", "k1": 1.5, "b": 0.75, "chunk_size": 800, "overlap": 150, "chunks": [{"text": 5, '
    '"source": "a"}]}',
    '{"format": "superborg-bm25", "k1": 1.5, "b": 0.75, "chunk_size": 800, "overlap": 150, "chunks": [7]}',
    '{"format": "superborg-bm25", "k1": 1.5, "b": 0.75, "chunk_size": 800, "overlap": 150, "chunks": {}}',
    '{"format": "superborg-bm25", "k1": "x", "b": 0.75, "chunk_size": 800, "overlap": 150, "chunks": []}',
    '{"format": "superborg-bm25", "k1": 1.5, "b": 0.75, "chunk_size": 800, "overlap": 150, '
    '"chunks": [{"text": "t", "source": "s", "line": "x"}]}',
])
def test_load_rejects_malformed_files_with_value_error(tmp_path, content):
    p = tmp_path / "bad.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        Index.load(str(p))


def test_format_context_numbers_and_limits():
    hits = [Hit("first passage", "/a/b/notes.md", 3.0, line=4), Hit("second passage " * 50, "x.txt", 2.0)]
    ctx = format_context(hits, max_chars=400)
    assert ctx.startswith("[1] (notes.md:4)\nfirst passage")
    assert "[2] (x.txt)" in ctx
    assert len(ctx) <= 420
