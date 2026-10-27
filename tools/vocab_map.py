"""Vocabulary map: a fixed 2D layout of a model's output vocabulary for the activation viewer.

    python tools/vocab_map.py MODEL.borg [--out-dir DIR] [--seeds 4] [--n-jobs 2]

The viewer wants a stable map of "word space" on which it can draw each layer's logit-lens
tokens. This tool computes that map offline, once per model:

  1. the output (unembedding) matrix as float32: `output`, or `tok_embd` when the embeddings
     are tied; only the n_vocab rows that are tokens (padding rows are dropped)
  2. per-dimension standardization: a few "rogue" dimensions otherwise dominate cosine
     distances between rows (Timkey & van Schijndel 2021)
  3. PCA to --pca dims (64); fit on a random subset of at most --pca-fit rows (40k) when the
     vocabulary is larger, then every row is transformed
  4. UMAP to 2D (cosine metric, n_neighbors 30, min_dist 0.1), once per seed
  5. stability: for every pair of seeds, the mean overlap of each token's 15 nearest
     neighbours on the two maps (1.0 = identical neighbourhoods); the layout kept is the one
     whose mean overlap with the other seeds is highest
  6. coordinates centred and scaled to [-1, 1] with one scale for both axes (shapes are kept;
     the longer axis spans the full range)
  7. --clusters (24) KMeans clusters of the 2D points, each labelled by its three most
     frequent-looking tokens: the lowest ids among its printable normal tokens, since BPE ids
     follow merge order, which tracks frequency
  8. a readable landmark per cluster, for the viewer's region labels: up to three whole words
     among its members (word-initial pieces, alphabetic, 3-12 letters, not stop words, standing
     alone rather than being the start of the vocabulary's longer pieces, see word_share(), and
     held by the vocabulary in both cases, see whole_words()), the lowest ids among those in the
     cluster's inner half around its density peak; a cluster without words (punctuation, digits,
     code, word fragments) is named by the kind of piece it is made of

Written next to the model (or in --out-dir):
  <stem>.vocab_map.bin   n x 2 float16, little-endian, row i = (x, y) of token i
  <stem>.vocab_map.json  {"n", "stability_knn15", "seed", "clusters": [{"label", "landmark",
                          "landmark_tokens", "x", "y", "n", "tokens"}...], "knn15_preserved"
                          (high-dim 15-NN kept on the map), "extent", "landmark_rule", the
                          parameters and the per-seed overlaps}

--relabel keeps an existing .bin and recomputes only the clusters and landmarks into the .json
(seconds, no UMAP), for maps made before the landmark rule or with other cluster settings.

It also prints the 5 nearest map neighbours of a few probe tokens (and, for comparison, their
nearest neighbours in the PCA space by cosine) so a human can judge whether neighbourhoods
are semantic.

Needs numpy, scikit-learn and umap-learn. Memory: the float32 matrix is freed once the PCA
projection exists; each UMAP run of a 152k x 896 vocabulary then peaks at about 2 GB (its
NN-descent phase), so the process tree peaks at about 4.2 GB with --n-jobs 2 and 1.7 GB with
--n-jobs 1 (a 49k x 576 vocabulary: 0.9 GB per run). Four seeds of the 152k vocabulary take
about 19 minutes with --n-jobs 2 on a shared 4-core machine.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import resource
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FORMAT = "vocab_map/1"
PROBES = [" Paris", " France", " city", " dog", " cat", " 1", " 2", " def", " return", ".", ",", " the"]
TOKEN_NORMAL, TOKEN_BYTE, TOKEN_SPECIAL, TOKEN_ADDED, TOKEN_UNUSED = 0, 1, 2, 3, 4


# ---- model -------------------------------------------------------------------------
def token_texts(header: dict, mm, data_start: int) -> tuple[list[str], np.ndarray]:
    """The text of every token (U+FFFD for bytes that are not UTF-8) and the token types."""
    from superborg.convert import load_tensor

    vocab = load_tensor(header, mm, data_start, "tokenizer.vocab").tobytes()
    off = load_tensor(header, mm, data_start, "tokenizer.vocab_offsets")
    types = load_tensor(header, mm, data_start, "tokenizer.token_type").astype(np.uint8)
    spm = not header["tokenizer"].get("byte_level", True)
    texts = []
    for i in range(len(off) - 1):
        b = vocab[off[i]:off[i + 1]]
        if spm and types[i] == TOKEN_BYTE:
            texts.append(f"<0x{b[0]:02X}>" if b else "")
            continue
        t = b.decode("utf-8", errors="replace")
        texts.append(t.replace("▁", " ") if spm else t)
    return texts, types


def load_unembedding(path: str) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """(W float32 [n_vocab, dim], token texts, token types, header) of a .borg model."""
    from superborg.convert import load_tensor, read_borg

    header, mm, data_start = read_borg(path)
    name = "tok_embd" if header["config"].get("tie_embeddings") or "output" not in header["tensors"] else "output"
    texts, types = token_texts(header, mm, data_start)
    n = len(texts)
    w = load_tensor(header, mm, data_start, name)
    if w.ndim != 2 or w.shape[0] < n:
        raise ValueError(f"{name} has shape {list(w.shape)}, expected [>= {n}, dim]")
    w = np.ascontiguousarray(w[:n], dtype=np.float32)
    return w, texts, types, header


# ---- projection ----------------------------------------------------------------------
def standardize(w: np.ndarray) -> np.ndarray:
    """Subtract each column's mean and divide by its standard deviation, in place (float32)."""
    mean = w.mean(axis=0, dtype=np.float64)
    std = w.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    w -= mean.astype(w.dtype)
    w /= std.astype(w.dtype)
    return w


def pca_reduce(w: np.ndarray, n_components: int = 64, fit_rows: int = 40000, seed: int = 0,
               chunk: int = 16384) -> tuple[np.ndarray, float]:
    """Project the rows of w onto their first n_components principal directions (float32).
    The basis is fitted on a random subset of fit_rows rows when w has more (0: all rows).
    Returns (projection, fraction of the fitted rows' variance the components explain)."""
    from sklearn.decomposition import PCA

    n = w.shape[0]
    k = min(n_components, w.shape[1], n)
    if fit_rows and n > fit_rows:
        idx = np.sort(np.random.default_rng(seed).choice(n, fit_rows, replace=False))
        fit = w[idx]
    else:
        fit = w
    pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(fit)
    del fit
    comps = pca.components_.T.astype(np.float32)   # [dim, k]
    mean = pca.mean_.astype(np.float32)
    out = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        out[s:s + chunk] = (w[s:s + chunk] - mean) @ comps
    return out, float(pca.explained_variance_ratio_.sum())


def _drop_self(idx: np.ndarray, k: int) -> np.ndarray:
    """From [n, >= k+1] neighbour lists, remove each row's own index (if present) and keep k."""
    n = idx.shape[0]
    mask = idx != np.arange(n)[:, None]
    keep = mask & (np.cumsum(mask, axis=1) <= k)
    return idx[keep].reshape(n, k)


def knn2d(xy: np.ndarray, k: int = 15) -> np.ndarray:
    """Indices [n, k] of each point's k nearest other points (Euclidean, 2D)."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k = min(k, n - 1)
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        if n > 8000:
            raise
        d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        idx = np.argsort(d, axis=1, kind="stable")[:, :k + 1]
        return _drop_self(idx, k)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xy)
    return _drop_self(nn.kneighbors(xy, return_distance=False), k)


def knn_cosine(y: np.ndarray, k: int = 15, chunk: int = 2048) -> np.ndarray:
    """Exact cosine k nearest neighbours of every row of y, [n, k] (O(n^2): small inputs only)."""
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    k = min(k, n - 1)
    u = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    out = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, chunk):
        sims = u[s:s + chunk] @ u.T
        part = np.argpartition(-sims, k, axis=1)[:, :k + 1]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1, kind="stable")
        out[s:s + chunk] = _drop_self(np.take_along_axis(part, order, axis=1), k)
    return out


def knn_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over rows of |a[i] ∩ b[i]| / k for two [n, k] neighbour-index arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    n, k = a.shape
    total = 0.0
    for s in range(0, n, 16384):  # [rows, k, k] booleans per chunk
        hit = (a[s:s + 16384, :, None] == b[s:s + 16384, None, :]).any(axis=2)
        total += float(hit.sum())
    return total / (n * k)


def stability(layouts: list[np.ndarray], k: int = 15) -> tuple[float, list[float], int]:
    """k-NN overlap between 2D layouts of the same points.
    Returns (mean over all pairs, each layout's mean overlap with the others, index of the
    layout with the highest mean). One layout: (1.0, [1.0], 0)."""
    m = len(layouts)
    if m == 0:
        raise ValueError("no layouts")
    if m == 1:
        return 1.0, [1.0], 0
    knn = [knn2d(xy, k) for xy in layouts]
    pair = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            pair[i, j] = pair[j, i] = knn_overlap(knn[i], knn[j])
    per = [float(pair[i].sum() / (m - 1)) for i in range(m)]
    mean = float(pair[np.triu_indices(m, 1)].mean())
    return mean, per, int(np.argmax(per))


def scale_unit(xy: np.ndarray) -> np.ndarray:
    """Centre the bounding box on the origin and scale both axes by the same factor so the
    longer axis spans [-1, 1] (float32)."""
    xy = np.asarray(xy, dtype=np.float64)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    centred = xy - (lo + hi) / 2
    half = float(np.max(np.abs(centred))) if len(xy) else 0.0
    if half > 0:
        centred /= half
    return np.clip(centred, -1.0, 1.0).astype(np.float32)


def umap_layout(y: np.ndarray, seed: int, n_neighbors: int = 30, min_dist: float = 0.1,
                knn_k: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """One UMAP run (cosine) on the PCA rows. Returns (xy float32 [n, 2], the high-dimensional
    knn_k nearest neighbours [n, knn_k] UMAP worked from, for the preservation metric)."""
    import warnings

    import numba
    import umap

    numba.set_num_threads(1)  # a seeded UMAP is single-threaded anyway; keep the machine shared
    n = len(y)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1")
        reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=min(n_neighbors, n - 1),
                            min_dist=min_dist, random_state=seed, low_memory=True, n_jobs=1)
        xy = reducer.fit_transform(y).astype(np.float32)
    hi = getattr(reducer, "_knn_indices", None)
    if hi is not None and hi.shape[1] > knn_k and not (hi < 0).any():
        hi = _drop_self(np.asarray(hi[:, :knn_k + 1], dtype=np.int64), knn_k)
    else:  # small inputs (n < 4096): UMAP used a full distance matrix and kept no kNN lists
        hi = knn_cosine(y, knn_k)
    return xy, hi


def _umap_worker(args):
    y, seed, n_neighbors, min_dist, knn_k = args
    t0 = time.perf_counter()
    xy, hi = umap_layout(y, seed, n_neighbors, min_dist, knn_k)
    return seed, xy, hi, time.perf_counter() - t0


def run_seeds(y: np.ndarray, seeds: list[int], n_neighbors: int = 30, min_dist: float = 0.1,
              n_jobs: int = 1, knn_k: int = 15, log=print) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """UMAP layouts for several seeds, up to n_jobs of them in parallel processes."""
    tasks = [(y, s, n_neighbors, min_dist, knn_k) for s in seeds]
    out = {}
    if n_jobs <= 1 or len(seeds) == 1 or __name__ != "__main__":
        results = map(_umap_worker, tasks)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=min(n_jobs, len(seeds)), mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(_umap_worker, tasks)
    for seed, xy, hi, dt in results:
        log(f"  seed {seed}: UMAP done in {dt:.1f}s")
        out[seed] = (xy, hi)
    return out


# ---- clusters and probes ------------------------------------------------------------------
def _printable(t: str) -> bool:
    return bool(t) and t.isprintable() and "�" not in t


def _label_text(t: str) -> str:
    s = t.strip()
    return s if s else repr(t)


# ---- landmarks ----------------------------------------------------------------------------
# Function words a region is never named after: the lowest ids of any region of words are these.
STOP_WORDS = frozenset("""
a about above after again against all along also although am among an and another any are around as at away
be became because become been before being below between both but by came can cannot come could did do does
doing done down during each either else enough ever every few for from further had has have having he hence her
here hers herself him himself his how however i if in into is it its itself just least less let many may me
might more most much must my myself neither never no nor not nothing now of off often on once one only onto or
other others otherwise ought our ours ourselves out over own per rather same shall she should since so some
something still such than that the their theirs them themselves then there therefore these they this those
though through thus to too toward towards under until unto up upon us very was we were what whatever when
whenever where whereas wherever whether which whichever while who whoever whom whose why will with within
without would yes yet you your yours yourself yourselves
""".split())
WORD_SHARE_MIN = 0.1      # a piece standing alone for less than a tenth of its weight is a fragment of longer pieces
WORD_FRACTION_MIN = 0.05  # a region is named by its words only when at least one member in twenty is a word
LANDMARK_RULE = ("up to three whole words (word-initial, alphabetic, 3-12 letters, no stop word, word_share >= 0.1, held in both "
                 "cases), the lowest ids among those in the inner half around the density peak; else the kind of piece the region is made of")


def readable_word(text: str) -> str | None:
    """The word a token spells when it is readable: word-initial (the tokenizer's space marker first),
    alphabetic after it, lowercase or Capitalized (no ALL CAPS, no camelCase), 3-12 letters long and
    not a stop word. Else None."""
    w = text[1:] if text[:1] == " " else ""
    if not w.isalpha() or not 3 <= len(w) <= 12 or not w[1:].islower() or w.lower() in STOP_WORDS:
        return None
    return w


def word_share(texts: list[str], ids) -> dict[int, float]:
    """For each token id in ids, the share of its weight that stands alone: 1 - (id + 1) * the sum of
    1 / (j + 1) over the tokens j that extend it with letters (" includ": " include", " included",
    " includes", " including"). BPE ids follow merge order, so 1 / (id + 1) is a Zipf-like proxy of a
    piece's frequency and the sum stands for the occurrences merged on into longer pieces: a fragment,
    nearly always continued, comes out near or below 0 (" includ": -0.6 in SmolLM2's vocabulary,
    " stud": -0.9, " toget": 0.0), a whole word, extended only by its rarer derivations, well above
    (" water": 0.44 with " watered", " waterfall"...; " people": 0.91; " energy": 1.0)."""
    order = sorted(range(len(texts)), key=texts.__getitem__)
    stexts = [texts[i] for i in order]
    out = {}
    for i in ids:
        t = texts[i]
        j = bisect.bisect_right(stexts, t)
        s = 0.0
        while j < len(stexts) and stexts[j].startswith(t):
            if stexts[j][len(t):].isalpha():
                s += 1.0 / (order[j] + 1)
            j += 1
        out[int(i)] = 1.0 - (int(i) + 1) * s
    return out


def whole_words(texts: list[str], types) -> dict[int, float]:
    """{id: word_share()} of the vocabulary's whole words: the readable_word() candidates among the
    normal tokens whose share is at least WORD_SHARE_MIN and which the vocabulary also holds in the
    other case. A lowercase word has a capitalized token from sentence starts (" water" and " Water");
    a fragment rarely has one (" appet", " inadequ"), even when the share gate lets it through because
    most of its continuations are separate tokens. A capitalized candidate passes when its lowercase
    form is absent (a name: " Europe") or is itself a word (" Muslim"), not when that form is a
    fragment (" Anthrop": " anthrop", " anthropology")."""
    types = np.asarray(types)
    cands = [i for i in range(len(texts)) if types[i] == TOKEN_NORMAL and readable_word(texts[i])]
    share = word_share(texts, cands)
    ids = {}
    for i in cands:
        ids.setdefault(texts[i], i)
    have = set(texts)
    out = {}
    for i in cands:
        if share[i] < WORD_SHARE_MIN:
            continue
        w = texts[i][1:]
        if w[0].isupper():
            j = ids.get(" " + w.lower())
            ok = (" " + w.lower()) not in have if j is None else share[j] >= WORD_SHARE_MIN
        else:
            ok = (" " + w[0].upper() + w[1:]) in have
        if ok:
            out[i] = share[i]
    return out


def piece_kind(text: str) -> str:
    """What a token is made of: "numbers" (digits), "punctuation" (no letter or digit), "fragments"
    (letters only, with or without the space marker) or "code" (anything mixed, whitespace runs, bytes)."""
    s = text.strip()
    if not s:
        return "code"
    if s.isdigit():
        return "numbers"
    if s.isalpha():
        return "fragments"
    if not any(ch.isalnum() for ch in s):
        return "punctuation"
    return "code"


def density_peak(pts: np.ndarray, k: int = 30) -> np.ndarray:
    """The point of pts whose k nearest other points are closest on average: the mode of the cloud."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts[0] if len(pts) else np.zeros(2)
    idx = knn2d(pts, min(k, len(pts) - 1))
    return pts[int(np.argmin(np.linalg.norm(pts[idx] - pts[:, None, :], axis=2).mean(axis=1)))]


def landmark(xy: np.ndarray, members, texts: list[str], words: dict[int, float]) -> tuple[str, list]:
    """A readable name for a region of the map: up to three of its whole words (words: whole_words()
    of the vocabulary), the lowest ids among those in the region's inner half (closer to its density
    peak than the median member), or among all of its words when fewer than three sit there. A region
    whose words are fewer than WORD_FRACTION_MIN of its members is named by the kind of piece it is
    made of instead (piece_kind(), the commonest): the few words of a region of fragments are mostly
    fragments the gates let through (" gamb", " hed": the SmolLM2 region of " stud", " bec", " includ"
    has 4% "words"; its regions of words have 11-61%). Returns (name, [[text, id], ...]); the list is
    empty for a kind name."""
    members = np.asarray(members)
    words = [int(i) for i in members if int(i) in words]
    if words and len(words) >= WORD_FRACTION_MIN * len(members):
        pts = np.asarray(xy[members], dtype=np.float64)
        peak = density_peak(pts)
        inner = np.linalg.norm(xy[words] - peak, axis=1) <= np.median(np.linalg.norm(pts - peak, axis=1))
        near = [i for i, ok in zip(words, inner) if ok]
        pick = sorted(near if len(near) >= 3 else words)[:3]
        return " · ".join(texts[i].strip() for i in pick), [[texts[i], i] for i in pick]
    kinds = Counter(piece_kind(texts[i]) for i in members)
    return max(kinds, key=kinds.get), []


def cluster_summary(xy: np.ndarray, texts: list[str], types: np.ndarray, n_clusters: int = 24,
                    seed: int = 0) -> list[dict]:
    """KMeans clusters of the 2D points, each with a "label" (its 3 lowest-id printable normal tokens,
    a frequency proxy, kept for older pages) and a "landmark" (landmark(): its whole words nearest its
    density peak, or the kind of piece it is made of)."""
    from sklearn.cluster import KMeans

    n = len(xy)
    k = max(1, min(n_clusters, n))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(xy)
    labels = km.labels_
    types = np.asarray(types)
    words = whole_words(texts, types)
    clusters = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        if len(members) == 0:
            continue
        good = [i for i in members if types[i] in (TOKEN_NORMAL, TOKEN_BYTE) and _printable(texts[i])]
        pick = (good if len(good) >= 3 else list(members))[:3]  # members are id-sorted already
        centre = xy[members].mean(axis=0)
        name, picked = landmark(xy, members, texts, words)
        clusters.append({
            "label": " ".join(_label_text(texts[i]) for i in pick), "landmark": name, "landmark_tokens": picked,
            "x": round(float(centre[0]), 4), "y": round(float(centre[1]), 4), "n": len(members),
            "tokens": [[texts[i], int(i)] for i in pick],
        })
    clusters.sort(key=lambda c: -c["n"])
    return clusters


def format_clusters(clusters: list[dict]) -> str:
    return "\n".join(f"  n={c['n']:6d} at ({c['x']:+.2f}, {c['y']:+.2f}): {c['landmark']}  [{c['label']}]" for c in clusters)


def find_token(texts: list[str], text: str, types=None) -> int | None:
    """Id of the token whose text is exactly `text` (normal tokens preferred), or of its
    variant without the leading space, else None."""
    candidates = [text]
    if text.startswith(" ") and len(text) > 1:
        candidates.append(text[1:])
    for cand in candidates:
        found = [i for i, t in enumerate(texts) if t == cand]
        if types is not None:
            normal = [i for i in found if types[i] == TOKEN_NORMAL]
            found = normal or found
        if found:
            return found[0]
    return None


def probe_neighbours(xy: np.ndarray, y: np.ndarray | None, texts: list[str], types=None,
                     probes: list[str] = PROBES, k: int = 5) -> list[dict]:
    """For each probe token: its k nearest tokens on the map and (when y is given) in the
    PCA space by cosine. Missing probes are reported with "id": None."""
    xy = np.asarray(xy, dtype=np.float64)
    if y is not None:
        u = np.asarray(y, dtype=np.float32)
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    out = []
    for p in probes:
        i = find_token(texts, p, types)
        if i is None:
            out.append({"probe": p, "id": None, "map": [], "pca": []})
            continue
        d = ((xy - xy[i]) ** 2).sum(axis=1)
        d[i] = np.inf
        near = np.argsort(d, kind="stable")[:k]
        row = {"probe": p, "id": int(i), "token": texts[i],
               "map": [[texts[j], int(j), round(float(np.sqrt(d[j])), 4)] for j in near]}
        if y is not None:
            s = u @ u[i]
            s[i] = -np.inf
            near = np.argsort(-s, kind="stable")[:k]
            row["pca"] = [[texts[j], int(j), round(float(s[j]), 3)] for j in near]
        out.append(row)
    return out


def format_probes(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        if r["id"] is None:
            lines.append(f"  {r['probe']!r}: not a single token")
            continue
        head = f"  {r['token']!r} (id {r['id']})"
        lines.append(f"{head}: map -> " + ", ".join(repr(t) for t, _, _ in r["map"]))
        if r.get("pca"):
            lines.append(" " * len(head) + "  pca -> " + ", ".join(repr(t) for t, _, _ in r["pca"]))
    return "\n".join(lines)


# ---- files -------------------------------------------------------------------------------
def write_map(stem: str, xy: np.ndarray, meta: dict) -> tuple[str, str]:
    """Write <stem>.vocab_map.bin (n x 2 float16 LE) and <stem>.vocab_map.json."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be [n, 2], got {list(xy.shape)}")
    if len(xy) and (np.abs(xy) > 1.0).any():
        raise ValueError("coordinates must lie in [-1, 1]")
    bin_path, json_path = stem + ".vocab_map.bin", stem + ".vocab_map.json"
    with open(bin_path, "wb") as f:
        f.write(np.ascontiguousarray(xy, dtype="<f2").tobytes())
    write_meta(json_path, dict(meta, format=FORMAT, n=int(len(xy)), bin=os.path.basename(bin_path)))
    return bin_path, json_path


def write_meta(json_path: str, meta: dict) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")


def relabel(model: str, stem: str, n_clusters: int = 24, log=print) -> dict:
    """Recompute the clusters and landmarks of the existing map <stem>.vocab_map.bin/.json from the
    .bin and the model's vocabulary, rewrite the .json (the .bin is kept) and return the new meta."""
    from superborg.convert import read_borg

    header, mm, data_start = read_borg(model)
    texts, types = token_texts(header, mm, data_start)
    xy = read_map(stem + ".vocab_map.bin")
    if len(xy) != len(texts):
        raise ValueError(f"{stem}.vocab_map.bin has {len(xy)} rows, the model {len(texts)} tokens")
    with open(stem + ".vocab_map.json", encoding="utf-8") as f:
        meta = json.load(f)
    t0 = time.perf_counter()
    meta["clusters"] = cluster_summary(xy, texts, types, n_clusters)
    meta["landmark_rule"] = LANDMARK_RULE
    write_meta(stem + ".vocab_map.json", meta)
    log(f"relabelled {stem}.vocab_map.json: {len(meta['clusters'])} clusters in {time.perf_counter() - t0:.1f}s")
    log(format_clusters(meta["clusters"]))
    return meta


def read_map(bin_path: str) -> np.ndarray:
    """The n x 2 float32 coordinates stored in a .vocab_map.bin file."""
    data = np.fromfile(bin_path, dtype="<f2")
    if data.size % 2:
        raise ValueError(f"{bin_path}: odd number of values")
    return data.reshape(-1, 2).astype(np.float32)


# ---- pipeline ------------------------------------------------------------------------------
def project(w: np.ndarray, pca_dims: int = 64, pca_fit: int = 40000, log=print) -> tuple[np.ndarray, float]:
    """Standardize w in place and reduce it with PCA: (float32 [n, pca_dims], explained variance)."""
    n, dim = w.shape
    t0 = time.perf_counter()
    standardize(w)
    y, explained = pca_reduce(w, pca_dims, pca_fit)
    log(f"  standardized {n} x {dim}, PCA to {y.shape[1]} dims ({explained:.1%} of variance) "
        f"in {time.perf_counter() - t0:.1f}s")
    return y, explained


def build_map(w: np.ndarray, texts: list[str], types: np.ndarray, seeds: list[int] = (0, 1, 2, 3),
              pca_dims: int = 64, pca_fit: int = 40000, n_neighbors: int = 30, min_dist: float = 0.1,
              n_clusters: int = 24, n_jobs: int = 1, knn_k: int = 15, probes: list[str] = PROBES,
              log=print) -> dict:
    """The whole computation on an in-memory matrix (standardized in place): project(), then
    layout_map(). The caller keeps w alive meanwhile; main() frees it between the two steps."""
    y, explained = project(w, pca_dims, pca_fit, log)
    return layout_map(y, texts, types, seeds, n_neighbors, min_dist, n_clusters, n_jobs, knn_k, probes,
                      explained=explained, pca_fit=pca_fit, dim=w.shape[1], log=log)


def layout_map(y: np.ndarray, texts: list[str], types: np.ndarray, seeds: list[int] = (0, 1, 2, 3),
               n_neighbors: int = 30, min_dist: float = 0.1, n_clusters: int = 24, n_jobs: int = 1,
               knn_k: int = 15, probes: list[str] = PROBES, explained: float = float("nan"),
               pca_fit: int = 0, dim: int | None = None, log=print) -> dict:
    """UMAP layouts of the PCA rows y for every seed, the stability pick, scaling, clusters and
    probes. Returns {"xy": scaled coordinates [n, 2], "meta": the JSON fields, "probes": probe
    neighbourhoods, "layouts": {seed: raw UMAP xy}}."""
    seeds = list(seeds)
    n = len(y)
    layouts = run_seeds(y, seeds, n_neighbors, min_dist, n_jobs, knn_k, log=log)
    xys = [layouts[s][0] for s in seeds]
    mean_overlap, per_seed, best = stability(xys, knn_k)
    seed = seeds[best]
    xy_best, knn_hi = layouts[seed]
    preserved = knn_overlap(knn2d(xy_best, knn_k), knn_hi)
    log(f"  stability (mean {knn_k}-NN overlap over {len(seeds)} seeds): {mean_overlap:.4f}; "
        f"per seed {dict(zip(seeds, [round(p, 4) for p in per_seed]))}; keeping seed {seed}; "
        f"high-dim {knn_k}-NN preserved on the map: {preserved:.4f}")
    xy = scale_unit(xy_best)
    clusters = cluster_summary(xy, texts, types, n_clusters)
    rows = probe_neighbours(xy, y, texts, types, probes)
    meta = {
        "n": n, "dim": int(dim if dim is not None else y.shape[1]),
        "stability_knn15": round(mean_overlap, 4), "seed": int(seed),
        "seeds": seeds, "seed_overlap": {str(s): round(p, 4) for s, p in zip(seeds, per_seed)},
        "knn15_preserved": round(preserved, 4), "knn_k": knn_k,
        "pca_dims": int(y.shape[1]), "pca_fit_rows": int(min(pca_fit, n) if pca_fit else n),
        "pca_explained": round(explained, 4), "n_neighbors": n_neighbors, "min_dist": min_dist,
        "metric": "cosine", "standardized": True,
        "extent": [round(float(v), 4) for v in (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())],
        "landmark_rule": LANDMARK_RULE, "clusters": clusters,
    }
    return {"xy": xy, "meta": meta, "probes": rows, "layouts": {s: layouts[s][0] for s in seeds}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("model", help="path to a .borg file")
    ap.add_argument("--out-dir", default=None, help="where to write the map (default: next to the model)")
    ap.add_argument("--seeds", type=int, default=4, help="number of UMAP seeds, 0..N-1 (default 4)")
    ap.add_argument("--n-jobs", type=int, default=2, help="seeds run in parallel processes (default 2)")
    ap.add_argument("--pca", type=int, default=64, help="PCA dimensions before UMAP (default 64)")
    ap.add_argument("--pca-fit", type=int, default=40000, help="rows the PCA is fitted on; 0 = all (default 40000)")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--clusters", type=int, default=24, help="KMeans clusters to label (default 24)")
    ap.add_argument("--probes", default=None, help="comma-separated probe tokens (default: a built-in list)")
    ap.add_argument("--relabel", action="store_true",
                    help="keep the existing .bin and recompute only the clusters and landmarks into the .json (no UMAP)")
    a = ap.parse_args(argv)

    stem = os.path.splitext(os.path.basename(a.model))[0]
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.model))
    if a.relabel:
        relabel(a.model, os.path.join(out_dir, stem), a.clusters)
        return 0
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(max(1, a.n_jobs))  # BLAS threads of the PCA
    except ImportError:
        pass
    probes = [p for p in a.probes.split(",")] if a.probes else PROBES
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    w, texts, types, header = load_unembedding(a.model)
    src = header.get("source") or stem
    print(f"{a.model}: {src}, output matrix {w.shape[0]} x {w.shape[1]} ({header['tensors'].get('output', header['tensors']['tok_embd'])['dtype']})", flush=True)
    def log(s: str) -> None:
        print(s, flush=True)

    dim = w.shape[1]
    y, explained = project(w, a.pca, a.pca_fit, log)
    del w  # the float32 matrix is the largest allocation; the workers only need y
    res = layout_map(y, texts, types, seeds=list(range(a.seeds)), n_neighbors=a.n_neighbors, min_dist=a.min_dist,
                     n_clusters=a.clusters, n_jobs=a.n_jobs, probes=probes, explained=explained, pca_fit=a.pca_fit,
                     dim=dim, log=log)
    meta = dict(res["meta"], model=stem, source=src, arch=header.get("arch"), quant=header.get("quant"))
    bin_path, json_path = write_map(os.path.join(out_dir, stem), res["xy"], meta)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"wrote {bin_path} ({os.path.getsize(bin_path)} bytes) and {json_path}")
    print(f"clusters ({len(meta['clusters'])}), landmark [label]:")
    print(format_clusters(meta["clusters"]))
    print("probe neighbourhoods (map: 5 nearest on the 2D map; pca: 5 nearest by cosine in the PCA space):")
    print(format_probes(res["probes"]))
    print(f"total {time.perf_counter() - t0:.1f}s, peak RSS {rss:.0f} MB (largest worker {rss_children:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
