"""Vocabulary map: a fixed 2D layout of a model's output vocabulary for the activation viewer.

    python tools/vocab_map.py MODEL.borg [--out-dir DIR] [--method hierarchical|flat] [--seeds 4] [--n-jobs N]

The viewer wants a stable map of "word space" on which it can draw each layer's logit-lens
tokens, with neighbourhoods that mean something (' Paris' beside ' London'). This tool computes
that map offline, once per model:

  1. the output (unembedding) matrix as float32: `output`, or `tok_embd` when the embeddings
     are tied; only the n_vocab rows that are tokens (padding rows are dropped)
  2. per-dimension standardization: a few "rogue" dimensions otherwise dominate cosine
     distances between rows (Timkey & van Schijndel 2021)
  3. PCA to --pca dims (64); fit on a random subset of at most --pca-fit rows (40k) when the
     vocabulary is larger, then every row is transformed; the rows are then scaled to unit
     length, so Euclidean and cosine agree
  4. the layout, by --method:
     hierarchical (the default): the unit rows are clustered with MiniBatchKMeans (fixed seed)
     into K = n / 120 fine clusters, at least 200 and at most 1200 (--fine-k overrides); the K
     unit centroids are laid out in 2D by UMAP (cosine, n_neighbors 15, min_dist 0.3), once per
     seed; each cluster becomes a disc of radius proportional to sqrt(size) (scaled so that
     three quarters of the nearest pairs of centres overlap, disc_radii()), and overlapping discs
     are pushed apart by iterations of pairwise repulsion (pack_discs(): the pushes are local,
     so the global arrangement is kept; it stops when no pair overlaps by more than 1%); each
     member is placed inside its disc by a local layout of the cluster's unit vectors: their 2D
     PCA (--within pca) or, the default, a t-SNE of them started from that PCA (--within tsne:
     exact gradients, perplexity min(30, (size-1)/3)), centred, turned so that the neighbouring
     clusters' directions in the cluster's PCA plane match their directions on the map
     (orient_rotation()), scaled so the cluster's 95th-percentile radius equals the disc radius
     (the outer 5% squeezed into a 10% rim, so discs stay apart), with a tiny jitter for
     duplicate rows. Measured on SmolLM2, the plain-PCA placement keeps 0.08 of the exact 15-NN
     (below) and the t-SNE placement 0.14; the flat UMAP kept 0.04. The kept seed's placement is
     then refined by one t-SNE of every token started from it (refine_layout(): Barnes-Hut,
     perplexity 30, --refine iterations, 1000; each token attracted by its exact 92 nearest rows
     by cosine, found by exact_knn()). The start is the placement scaled down to the spread
     scikit-learn gives its own PCA initialization, with its early exaggeration (--refine-start
     small, --refine-exaggeration 12): a t-SNE of the whole vocabulary whose arrangement comes from
     the discs, where the neighbours that fell into different clusters meet (SmolLM2 keeps 0.19 of
     the exact 15-NN after it, the same with perplexity 8, 15 or 30, 1000 or 2000 iterations, and
     from the plain-PCA placement; a UMAP refinement, 15 neighbours, min_dist 0, keeps 0.07). The
     hubs of the neighbour graph, rows among the 15 nearest of hundreds of others, are capped first
     (cap_hubs(), --hub-cap 30): a hub stays in the lists of its 30 nearest rows only, so a t-SNE
     does not pull unrelated frequent pieces between the digits or the countries (on Qwen2.5 it
     did: '1' next to '评价' and 'いただ'; with the cap the digits sit together again, and the
     15-NN kept is unchanged, 0.193 against 0.193 on SmolLM2, 0.209 against 0.208 on Qwen).
     --refine-start placed keeps the placement at its own scale (with --refine-exaggeration 1),
     so the islands hold and each token moves towards its nearest rows from where its island put
     it: 0.18 kept on SmolLM2 and 0.20 on Qwen (0.199), a little less than the small start,
     with the same number of probes semantic on Qwen (7 of 12) and more on SmolLM2 (10 against 9).
     Then the neighbour passes (neighbour_pass(), at most --passes, 12): a local search for the very
     number knn15_preserved measures, in which each token may move beside one of its 20 nearest
     rows (or jiggle where it is) when the move raises the whole vocabulary's exact 15-NN overlap,
     counted exactly with the lists it leaves and enters; the first pass moves two fifths of the
     tokens and takes SmolLM2 from 0.19 to 0.24, the rest to 0.25 (Qwen 0.20 to 0.27), where the
     t-SNE's arrangement is the ceiling (a token's best spot beside any of its nearest rows, side
     effects ignored, would keep 0.27-0.28). The discs reported after the refinement and the passes
     are the clusters' means and 95th-percentile radii on the final layout (cluster_discs());
     --refine 0 --passes 0 keeps the discs as placed.
     flat: UMAP of all the rows straight to 2D (cosine, n_neighbors 30, min_dist 0.1), once per
     seed (the method of maps before vocab_map/1's "method" field)
  5. stability: for every pair of seeds, the mean overlap of each point's 15 nearest neighbours
     on the two layouts (1.0 = identical neighbourhoods); the seed kept is the one whose mean
     overlap with the others is highest. The hierarchical method picks the seed on the centroid
     layouts (the only part a seed changes) and reports "stability_knn15" on the token placements
     of all seeds before the refinement and the neighbour passes, which run once, on the kept
     seed's placement; the flat method does both on the token layouts
  6. coordinates centred and scaled to [-1, 1] with one scale for both axes (shapes are kept;
     the longer axis spans the full range)
  7. neighbourhood preservation, "knn15_preserved": for a random sample of --knn-sample (12000)
     tokens, the overlap of each one's 15 nearest tokens on the map with its exact 15 nearest by
     cosine in the PCA space, both among the whole vocabulary; "knn15_preserved_within_sample"
     is the same overlap with both neighbourhoods taken among the sample alone (a looser number:
     the sample's neighbours are farther apart)
  8. --clusters (24) regions for the viewer's labels: KMeans of the fine clusters' centres,
     each weighted by its size (hierarchical), or of the 2D points (flat); each labelled by its
     three most frequent-looking tokens: the lowest ids among its printable normal tokens, since
     BPE ids follow merge order, which tracks frequency
  9. a readable landmark per region (and per fine cluster), for the viewer's region labels: up
     to three whole words among its members (word-initial pieces, alphabetic, 3-12 letters, not
     stop words, standing alone rather than being the start of the vocabulary's longer pieces,
     see word_share(), and held by the vocabulary in both cases, see whole_words()), the lowest
     ids among those in the region's inner half around its density peak; a region without words
     (punctuation, digits, code, word fragments) is named by the kind of piece it is made of

Written next to the model (or in --out-dir):
  <stem>.vocab_map.bin   n x 2 float16, little-endian, row i = (x, y) of token i
  <stem>.vocab_map.json  {"n", "method", "stability_knn15", "seed", "clusters": [{"label",
                          "landmark", "landmark_tokens", "x", "y", "n", "tokens"}...],
                          "fine_clusters": [{"x", "y", "r", "n", "landmark"}...] (hierarchical:
                          the discs, in map units), "refine" (its iterations, 0 for none),
                          "refine_perplexity", "refine_start", "refine_exaggeration",
                          "refine_hub_cap", "in_own_disc" (the share of tokens nearest their
                          own cluster's centre on the final layout), "refine_centroid_knn15"
                          (the 15-NN overlap of the cluster centres as packed and as final),
                          "neighbour_passes" (the passes run), "neighbour_pass_moved" (the share
                          of tokens they moved), "knn15_all_before_passes" and
                          "knn15_all_after_passes" (the exact 15-NN overlap over all tokens, not
                          the sample, before and after them), "knn15_preserved",
                          "knn15_preserved_within_sample", "extent", "landmark_rule", the
                          parameters and the per-seed overlaps}

--relabel keeps an existing .bin and recomputes only the regions and landmarks into the .json
(seconds, no layout), for maps made before the landmark rule or with other cluster settings; on
a hierarchical map it regroups the json's discs, each token counted with the disc whose centre is
nearest in units of that disc's radius (an unrefined map, --refine 0: the discs are disjoint, so
this is nearly the membership) or simply nearest (a refined map: "in_own_disc" says how much of the
membership that recovers).

It also prints the 5 nearest map neighbours of a few probe tokens (and, for comparison, their
nearest neighbours in the PCA space by cosine) so a human can judge whether neighbourhoods
are semantic.

Needs numpy, scipy, scikit-learn and umap-learn. Memory: the float32 matrix is freed once the PCA
projection exists. The hierarchical method peaks at 1.8 GB on a 152k x 896 vocabulary (the load
and PCA; the layout itself needs a few hundred MB, the exact neighbour search 0.6 GB, the passes
0.9 GB) and took about 2 hours (117 minutes) with --n-jobs 1 on a shared 4-core machine (load
3-7), 11 of them the 1199 small t-SNEs, 55 the refinement and 40 the 11 neighbour passes (a 49k x
576 vocabulary: 0.9 GB, about 24 minutes, 8 of them the refinement and 10 the 9 passes; --within
pca takes seconds instead of the small t-SNEs, --refine 0 skips the refinement, --passes 0 the
passes). The flat method's
UMAP of the 152k rows peaks at about 2 GB per seed (its NN-descent phase), so the process tree
peaks at about 4.2 GB with --n-jobs 2 and 1.7 GB with --n-jobs 1, and four seeds take about 19
minutes with --n-jobs 2.
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


def _drop_self(idx: np.ndarray, k: int, own: np.ndarray | None = None) -> np.ndarray:
    """From [m, >= k+1] neighbour lists, remove each row's own index (own[i], default i; if
    present) and keep k."""
    m = idx.shape[0]
    mask = idx != (np.arange(m) if own is None else np.asarray(own))[:, None]
    keep = mask & (np.cumsum(mask, axis=1) <= k)
    return idx[keep].reshape(m, k)


def knn2d(xy: np.ndarray, k: int = 15, queries: np.ndarray | None = None) -> np.ndarray:
    """Indices [m, k] of the k nearest other points (Euclidean, 2D) of every point, or of the
    points `queries` (indices into xy) alone."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k = min(k, n - 1)
    own = np.arange(n) if queries is None else np.asarray(queries)
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        if n > 8000:
            raise
        d = ((xy[own][:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        idx = np.argsort(d, axis=1, kind="stable")[:, :k + 1]
        return _drop_self(idx, k, own)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xy)
    return _drop_self(nn.kneighbors(xy[own], return_distance=False), k, own)


def unit_rows(y: np.ndarray) -> np.ndarray:
    """The rows of y scaled to unit length (float32): cosine becomes a dot product and
    Euclidean k-means groups by cosine."""
    y = np.asarray(y, dtype=np.float32)
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def _top(sims: np.ndarray, k: int) -> np.ndarray:
    """Indices [rows, k] of the k largest entries of each row of sims, largest first."""
    part = np.argpartition(sims, sims.shape[1] - k, axis=1)[:, -k:]
    order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def knn_preserved(xy: np.ndarray, y: np.ndarray, k: int = 15, sample: int = 12000, seed: int = 0,
                  chunk: int = 256) -> tuple[float, float]:
    """How much of the high-dimensional neighbourhood the map keeps. For a random sample of
    the tokens (all of them when there are fewer than `sample`), the mean overlap of each one's
    k nearest tokens on the map with its exact k nearest by cosine among the rows of y. Returns
    (the overlap with both neighbourhoods taken among the whole vocabulary, the overlap with both
    taken among the sample alone: the looser number a layout of the sample by itself would be
    judged by, since the sample's neighbours are farther apart)."""
    u = unit_rows(y)
    n = len(u)
    if n < 3:
        return 1.0, 1.0
    q = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False)) if n > sample else np.arange(n)
    k_full, k_sub = min(k, n - 1), min(k, len(q) - 1)
    hi_full = np.empty((len(q), k_full), dtype=np.int64)
    hi_sub = np.empty((len(q), k_sub), dtype=np.int64)
    for s in range(0, len(q), chunk):
        qi = q[s:s + chunk]
        sims = u[qi] @ u.T
        sims[np.arange(len(qi)), qi] = -np.inf
        hi_full[s:s + chunk] = _top(sims, k_full)
        hi_sub[s:s + chunk] = _top(sims[:, q], k_sub)
    full = knn_overlap(knn2d(xy, k_full, queries=q), hi_full)
    within = knn_overlap(knn2d(np.asarray(xy)[q], k_sub), hi_sub)
    return full, within


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


def unit_transform(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """(centre, half) of scale_unit(): the bounding box's centre and half its longer side (1.0
    for a degenerate box), so that other points and lengths can be put in the same map units."""
    xy = np.asarray(xy, dtype=np.float64)
    if not len(xy):
        return np.zeros(2), 1.0
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    centre = (lo + hi) / 2
    half = float(np.max(np.abs(xy - centre)))
    return centre, half if half > 0 else 1.0


def scale_unit(xy: np.ndarray, centre: np.ndarray | None = None, half: float | None = None) -> np.ndarray:
    """Centre the bounding box on the origin and scale both axes by the same factor so the
    longer axis spans [-1, 1] (float32); with centre and half given, that transform instead."""
    xy = np.asarray(xy, dtype=np.float64)
    if centre is None or half is None:
        centre, half = unit_transform(xy)
    return np.clip((xy - centre) / half, -1.0, 1.0).astype(np.float32)


def umap_layout(y: np.ndarray, seed: int, n_neighbors: int = 30, min_dist: float = 0.1) -> np.ndarray:
    """One UMAP run (cosine) on the rows of y: xy float32 [n, 2]."""
    import warnings

    import numba
    import umap

    numba.set_num_threads(1)  # a seeded UMAP is single-threaded anyway; keep the machine shared
    n = len(y)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1")
        reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=min(n_neighbors, n - 1),
                            min_dist=min_dist, random_state=seed, low_memory=True, n_jobs=1)
        return reducer.fit_transform(y).astype(np.float32)


def _umap_worker(args):
    y, seed, n_neighbors, min_dist = args
    t0 = time.perf_counter()
    xy = umap_layout(y, seed, n_neighbors, min_dist)
    return seed, xy, time.perf_counter() - t0


def run_seeds(y: np.ndarray, seeds: list[int], n_neighbors: int = 30, min_dist: float = 0.1,
              n_jobs: int = 1, log=print) -> dict[int, np.ndarray]:
    """UMAP layouts for several seeds, up to n_jobs of them in parallel processes."""
    tasks = [(y, s, n_neighbors, min_dist) for s in seeds]
    out = {}
    if n_jobs <= 1 or len(seeds) == 1 or __name__ != "__main__":
        results = map(_umap_worker, tasks)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=min(n_jobs, len(seeds)), mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(_umap_worker, tasks)
    for seed, xy, dt in results:
        log(f"  seed {seed}: UMAP of {len(y)} points done in {dt:.1f}s")
        out[seed] = xy
    return out


# ---- the hierarchical layout -------------------------------------------------------------------
def fine_count(n: int, per: int = 120, lo: int = 200, hi: int = 1200) -> int:
    """The number of fine clusters for n tokens: n / per within [lo, hi], and at most n."""
    return max(1, min(n, int(np.clip(n // per, lo, hi))))


def fine_kmeans(u: np.ndarray, k: int, seed: int = 0, batch: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MiniBatchKMeans of the unit rows u into k clusters (Euclidean on unit vectors orders pairs
    like cosine). Returns (labels [n] in 0..K-1, empty clusters dropped; unit centroids [K, d],
    the exact means of the members; sizes [K])."""
    from sklearn.cluster import MiniBatchKMeans

    n = len(u)
    k = max(1, min(k, n))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(batch, n), n_init=3, max_iter=200,
                         reassignment_ratio=0.01, max_no_improvement=20).fit(u)
    labels = km.labels_.astype(np.int64)
    keep = np.flatnonzero(np.bincount(labels, minlength=k) > 0)
    remap = np.full(k, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    labels = remap[labels]
    sizes = np.bincount(labels, minlength=len(keep))
    cent = np.zeros((len(keep), u.shape[1]), dtype=np.float64)
    np.add.at(cent, labels, u)
    return labels, unit_rows(cent / sizes[:, None]), sizes


def members_of(labels: np.ndarray, k: int) -> list[np.ndarray]:
    """The sorted member indices of each of the k clusters."""
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(k + 1))
    return [order[bounds[c]:bounds[c + 1]] for c in range(k)]


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The centred rows of x on their first two principal directions: (P [n, 2], basis [2, d];
    zero rows where x has fewer directions)."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    basis, p = np.zeros((2, d)), np.zeros((n, 2))
    if n >= 2:
        xc = x - x.mean(axis=0)
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        vt = vt[:2][s[:2] > 1e-9 * max(float(s[0]), 1e-30)]
        basis[:len(vt)] = vt
        p = xc @ basis.T
    return p, basis


def local_layout(x: np.ndarray, within: str = "tsne", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A centred 2D layout of the rows of x (one fine cluster's unit vectors): their 2D PCA
    (within "pca"), or that PCA refined by a t-SNE started from it (within "tsne": exact
    gradients, cosine, perplexity min(30, (n-1)/3); rows fewer than 10, or without any spread,
    keep the PCA). Returns (P [n, 2], basis [2, d]: the PCA directions, for orient_rotation())."""
    p, basis = pca2(x)
    n = len(p)
    if within == "tsne" and n >= 10 and float(p[:, 0].std()) > 0:
        from sklearn.manifold import TSNE

        init = p / float(p[:, 0].std()) * 1e-4  # the scale sklearn gives its own PCA initialization
        p = TSNE(n_components=2, perplexity=min(30.0, (n - 1) / 3), metric="cosine", init=init, random_state=seed,
                 method="exact" if n <= 2000 else "barnes_hut", n_jobs=1).fit_transform(np.asarray(x, dtype=np.float64))
        p = np.asarray(p, dtype=np.float64)
        p -= p.mean(axis=0)
    elif within not in ("tsne", "pca"):
        raise ValueError(f"unknown local layout {within!r}")
    return p, basis


def disc_radii(cxy: np.ndarray, sizes: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    """Disc radii proportional to sqrt(size), scaled so that, of the nearest pairs of centres,
    the one at `quantile` just touches: below it the pairs overlap and pack_discs() pushes them
    apart. Measured on SmolLM2 (409 discs): the median pair leaves the discs covering 14% of
    their bounding box, a field of scattered dots, and keeps 0.95 of the centroids' 15-NN through
    the packing; the 0.75 quantile covers 29% and keeps 0.86; the 0.9 quantile covers 43% but
    keeps 0.74 and does not settle in 300 iterations. A single disc: radius 1."""
    sq = np.sqrt(np.asarray(sizes, dtype=np.float64))
    if len(sq) < 2:
        return np.ones_like(sq)
    nb = knn2d(cxy, 1)[:, 0]
    d = np.linalg.norm(np.asarray(cxy, dtype=np.float64) - np.asarray(cxy, dtype=np.float64)[nb], axis=1)
    c = float(np.quantile(d / (sq + sq[nb]), quantile))
    return (c if c > 0 else 1.0) * sq


def pack_discs(cxy: np.ndarray, r: np.ndarray, margin: float = 0.15, iters: int = 400,
               tol: float = 0.01) -> tuple[np.ndarray, int, float]:
    """Push overlapping discs apart: in each iteration every pair of discs closer than the sum of
    their radii (plus `margin` of it, a gap) moves apart along the line between the centres by
    half the overlap, split in proportion to the other's area, so a big disc hardly moves for a
    small one; the pushes are local, so the global arrangement is kept. Stops when no pair
    overlaps by more than tol of its target distance. Returns (centres, iterations, the largest
    remaining relative overlap)."""
    xy = np.array(cxy, dtype=np.float64, copy=True)
    r = np.asarray(r, dtype=np.float64)
    if len(xy) < 2:
        return xy, 0, 0.0
    target = (r[:, None] + r[None, :]) * (1 + margin)
    mass = r ** 2
    share = mass[None, :] / (mass[:, None] + mass[None, :])
    rng = np.random.default_rng(0)
    done, rel = 0, 0.0
    for done in range(iters + 1):
        d = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((d ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        rel = float((np.maximum(target - dist, 0.0) / target).max())
        if rel < tol or done == iters:
            break
        same = dist < 1e-12  # coincident centres: a random direction apart
        if same.any():
            d[same] = rng.normal(size=(int(same.sum()), 2))
            dist[same] = np.linalg.norm(d[same], axis=1)
        step = np.maximum(target - dist, 0.0) * share * 0.5
        xy += (d / dist[..., None] * step[..., None]).sum(axis=1)
    return xy, done, rel


def orient_rotation(basis: np.ndarray, cent: np.ndarray, c: int, nb: np.ndarray, cxy: np.ndarray) -> np.ndarray:
    """The 2 x 2 orthogonal matrix that turns fine cluster c's local frame (basis: its two PCA
    directions in the high-dimensional space) so that the directions of its neighbouring
    clusters nb in that frame match their directions on the map (orthogonal Procrustes, each
    neighbour weighted by 1 / its map distance): the side of a disc that faces a neighbour then
    holds the members nearest to it."""
    a = (cent[nb] - cent[c]) @ np.asarray(basis).T
    b = np.asarray(cxy, dtype=np.float64)[nb] - np.asarray(cxy, dtype=np.float64)[c]
    w = 1.0 / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    uu, _, vt = np.linalg.svd((a * w[:, None]).T @ (b * w[:, None]))
    return uu @ vt


def place_members(cxy: np.ndarray, r: np.ndarray, members: list[np.ndarray], layouts: list[tuple[np.ndarray, np.ndarray]],
                  cent: np.ndarray | None = None, jitter: float = 0.02, seed: int = 0) -> np.ndarray:
    """Every token's map position: its fine cluster's disc centre plus its local coordinates
    (layouts[c] = local_layout() of the cluster's rows), turned by orient_rotation() towards the
    8 nearest discs when the centroids cent are given, scaled so the cluster's 95th-percentile
    radius equals the disc radius with the outer 5% squeezed into a 10% rim (neighbouring discs
    stay apart), and coincident members (duplicate rows) jittered by `jitter` of the radius."""
    cxy = np.asarray(cxy, dtype=np.float64)
    k = len(cxy)
    n = sum(len(m) for m in members)
    xy = np.zeros((n, 2))
    nb = knn2d(cxy, min(8, k - 1)) if cent is not None and k > 1 else None
    rng = np.random.default_rng(seed)
    for c, m in enumerate(members):
        p, basis = layouts[c]
        p = np.array(p, dtype=np.float64, copy=True)
        if nb is not None and len(m) > 1:
            p = p @ orient_rotation(basis, cent, c, nb[c], cxy)
        rad = np.linalg.norm(p, axis=1)
        p95 = float(np.percentile(rad, 95)) if len(m) > 1 else 0.0
        if p95 > 0:
            p *= r[c] / p95
            rad *= r[c] / p95
            out = rad > r[c]
            if out.any():
                p[out] *= (r[c] * (1 + 0.1 * np.tanh((rad[out] - r[c]) / (0.1 * r[c]))) / rad[out])[:, None]
        if len(m) > 1:
            key = np.round(p / max(float(r[c]), 1e-12) * 1e4).astype(np.int64)
            _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
            dup = cnt[np.ravel(inv)] > 1
            if dup.any():
                p[dup] += rng.normal(scale=jitter * r[c], size=(int(dup.sum()), 2))
        xy[m] = cxy[c] + p
    return xy


def exact_knn(u: np.ndarray, k: int = 15, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """The exact k nearest other rows of every row of the unit rows u, by cosine: (indices int32
    [n, k], nearest first; cosine distances float32 [n, k]). By chunks of `chunk` rows against
    every row, so the working memory is chunk x n similarities and as many int64 indices (0.5 GB
    for 256 x 152k); 49k rows take about 20 s, 152k rows a few minutes."""
    u = np.asarray(u, dtype=np.float32)
    n = len(u)
    k = max(0, min(k, n - 1))
    idx = np.empty((n, k), dtype=np.int32)
    dist = np.empty((n, k), dtype=np.float32)
    if k == 0:
        return idx, dist
    for s in range(0, n, chunk):
        sims = u[s:s + chunk] @ u.T
        rows = np.arange(sims.shape[0])
        sims[rows, s + rows] = -np.inf
        top = _top(sims, k)
        idx[s:s + chunk] = top
        dist[s:s + chunk] = np.maximum(1.0 - np.take_along_axis(sims, top, axis=1), 0.0)
    return idx, dist


def cap_hubs(idx: np.ndarray, dist: np.ndarray, cap: int, k: int = 15) -> tuple[np.ndarray, int, float]:
    """Hubs are rows that sit among the first k neighbours of far more rows than a 2D layout can
    put them next to (on Qwen2.5's 151k rows, '和' is among the 15 nearest of 405 others and ' one'
    of 357; a point of a 2D layout is among the 15 nearest of about 15). In the neighbour lists
    (idx, dist) [n, >= k] from exact_knn(), keep every row among the first k neighbours of at most
    `cap` other rows, its nearest ones, and push it to the largest cosine distance, 2, in the other
    rows' lists (the entry stays, so each row keeps its count of neighbours; its weight in a
    perplexity-calibrated neighbourhood becomes nil). In place; returns (dist, the number of hubs,
    the share of the first k columns pushed away)."""
    n, kk = idx.shape[0], min(k, idx.shape[1])
    if cap <= 0 or n == 0:
        return dist, 0, 0.0
    cols = idx[:, :kk].ravel()
    indeg = np.bincount(cols, minlength=n)
    order = np.lexsort((dist[:, :kk].ravel(), cols))   # by column, then by distance: a hub's nearest rows first
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order)) - np.repeat(np.concatenate([[0], np.cumsum(indeg)[:-1]]), indeg)
    far = rank >= cap
    dist[np.repeat(np.arange(n), kk)[far], np.tile(np.arange(kk), n)[far]] = 2.0
    return dist, int((indeg > cap).sum()), float(far.mean())


def refine_layout(xy: np.ndarray, u: np.ndarray, perplexity: float = 30.0, iters: int = 1000, seed: int = 0,
                  n_jobs: int = 1, exaggeration: float = 12.0, start: str = "small", hub_cap: int = 30,
                  knn: tuple[np.ndarray, np.ndarray] | None = None, log=None) -> np.ndarray:
    """A t-SNE of every token started from the layout xy (scikit-learn's Barnes-Hut t-SNE, its
    gradient on n_jobs threads), each token attracted by its exact 3 * perplexity + 2 nearest rows
    of u by cosine (exact_knn(), handed over as a precomputed sparse distance graph: scikit-learn's
    own neighbour search would take 1.7 GB on 49k rows, this takes 0.4 GB; with hub_cap > 0 the
    lists first go through cap_hubs()). The start: "placed" keeps the layout at its own scale
    (standardized to a spread of 30, about that of a t-SNE of this many points) with `exaggeration`
    (1: none) throughout, so the islands of a hierarchical map hold and each token moves towards
    its nearest rows from where its island put it (on SmolLM2: 0.18 of the exact 15-NN kept against
    0.14 before, 10 of the 12 probes with 3 or more related neighbours); "small" scales the layout
    down to the 1e-4 spread scikit-learn gives its own PCA initialization and runs the usual early
    exaggeration (12 for 250 iterations), which makes a t-SNE of the whole vocabulary with the
    islands' arrangement (0.19-0.20 kept whatever the perplexity, 8 to 30, or iterations, 1000 or
    2000, but the islands dissolve and hubs pull unrelated pieces between the frequent ones; see the
    module docstring). `knn`: the exact neighbour lists (indices, cosine distances) of u from
    exact_knn() when the caller has them, with at least 3 * perplexity + 2 columns; else they are
    searched here. Returns the raw t-SNE coordinates (float64 [n, 2]); xy itself when iters is 0
    or there are fewer than 10 rows."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if iters <= 0 or n < 10:
        return xy
    if start not in ("placed", "small"):
        raise ValueError(f"unknown refinement start {start!r}")
    from scipy.sparse import csr_matrix
    from sklearn.manifold import TSNE

    perplexity = min(float(perplexity), (n - 4) / 3)  # scikit-learn wants 3 * perplexity + 2 other rows per row
    k = min(int(3 * perplexity) + 2, n - 1)
    if knn is not None and knn[0].shape[1] >= k:
        idx, dist = np.ascontiguousarray(knn[0][:, :k]), np.array(knn[1][:, :k], dtype=np.float32)  # a copy: cap_hubs() writes
    else:
        idx, dist = exact_knn(u, k)
    if hub_cap:
        dist, hubs, pushed = cap_hubs(idx, dist, hub_cap)
        if log:
            log(f"  hub cap {hub_cap}: {hubs} pieces among the 15 nearest of more than {hub_cap} others; "
                f"{pushed:.1%} of the 15-NN entries pushed away")
    graph = csr_matrix((dist.ravel().astype(np.float64), idx.ravel().astype(np.int64), np.arange(0, n * k + 1, k)),
                       shape=(n, n))
    init = xy - xy.mean(axis=0)
    sd = float(init[:, 0].std())
    init = init / sd * (30.0 if start == "placed" else 1e-4) if sd > 0 else init
    ts = TSNE(n_components=2, perplexity=perplexity, metric="precomputed", init=init, random_state=seed,
              early_exaggeration=max(1.0, float(exaggeration)),
              method="barnes_hut", n_jobs=max(1, n_jobs), max_iter=max(250, iters), n_iter_without_progress=max(250, iters))
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Precomputed sparse input was not sorted")  # cap_hubs() unsorts rows
        try:  # scikit-learn's Barnes-Hut gradient runs on every OpenMP thread regardless of n_jobs, and its barriers
            from threadpoolctl import threadpool_limits  # crawl on a loaded machine (300 points: 21 s against 1.2 s)
        except ImportError:
            return np.asarray(ts.fit_transform(graph), dtype=np.float64)
        with threadpool_limits(max(1, n_jobs), user_api="openmp"):
            return np.asarray(ts.fit_transform(graph), dtype=np.float64)


def neighbour_pass(xy: np.ndarray, nbr: np.ndarray, k: int = 15, passes: int = 12, anchors: np.ndarray | None = None,
                   n_anchors: int = 20, per_anchor: int = 3, far: int = 2, jiggle: int = 8, batch: int = 0,
                   window: int = 64, min_gain: int = 1, stop: float = 0.0002, offset: float = 0.35,
                   offset_floor: float = 6e-4, quantize: bool = True, seed: int = 0, log=None) -> tuple[np.ndarray, dict]:
    """A local search on the layout xy (map units) for the objective knn15_preserved measures: the
    mean over all tokens of how many of a token's k nearest points on the map are among its exact k
    nearest rows, `nbr` ([n, >= k] from exact_knn(), nearest first). In each pass every token, in a
    random order by batches, is offered candidate spots: beside each of its `anchors` rows ([n, >= 1],
    default the first n_anchors columns of nbr, its 20 nearest when nbr is that wide; per_anchor
    spots per row, at 0.35, 0.8, 1.25... of that row's own nearest distance in a random direction, at
    least offset_floor, the float16 step of the .bin at the map's edge), `far` more per row at 0.3,
    0.7, ... of that row's kth-nearest distance (inside its neighbourhood, not on top of it) and
    `jiggle` spots around its own position (0.3 to 2 nearest distances away); with `quantize` the
    spots are rounded to float16 first, the coordinates the .bin holds, so the objective is the
    file's (the rounding ties points under the float16 step, which otherwise costs 0.002 of the
    15-NN kept). A move is scored by the exact change of the whole objective: the token's own count
    at the spot, the counts of the tokens whose k nearest it leaves (each loses it and gains its
    (k+1)th) and of those whose k nearest it enters (found among the `window` points nearest the
    spot: each gains it and loses its kth); the best spot is taken when it gains at least min_gain.
    A batch is scored against the layout as it stands at the batch's start (the next batch sees
    the moves), so a pass's gain can fall short of its estimate; after the first pass only the
    tokens a move touched are scored again (those that moved, those with a moved anchor, those
    whose k nearest changed), and the passes end when one gains less than `stop` or moves
    nothing. On SmolLM2's refined map the first pass moves two fifths of the tokens and takes the
    exact 15-NN kept from 0.19 to 0.24, the next ones to 0.25 (the moves are short: a token goes
    to where its nearest rows already are; the ceiling is the arrangement's: the best spot beside
    any exact neighbour, side effects ignored, would keep 0.27-0.28, and a third of a token's 15
    nearest lie beyond its 100 nearest on the map). Returns (xy float64 [n, 2], {"passes": run,
    "moved": the share of tokens moved at least once, "before", "after": the objective})."""
    from scipy.spatial import cKDTree

    xy = np.array(xy, dtype=np.float64)
    n = len(xy)
    lists = np.asarray(nbr)
    nbr = np.ascontiguousarray(lists[:, :k], dtype=np.int32)
    anchors = np.ascontiguousarray(lists[:, :max(k, n_anchors)] if anchors is None else anchors, dtype=np.int32)
    stats = {"passes": 0, "moved": 0.0, "before": None, "after": None}
    if passes <= 0 or k < 1 or n < k + 3 or anchors.shape[1] < 1:
        return xy, stats
    if quantize:
        xy = xy.astype(np.float16).astype(np.float64)
    window = min(window, n - 1)
    batch = batch or max(1024, n // 48)   # tokens scored together: a pass of a 152k vocabulary rebuilds its lists 48 times
    ka, pa = anchors.shape[1], per_anchor + far
    c = ka * pa + jiggle                         # candidate spots per token
    rng = np.random.default_rng(seed)
    fac = np.concatenate([np.tile(np.concatenate([offset + 0.45 * np.arange(per_anchor), np.zeros(far)]), ka), np.zeros(jiggle)])
    fac_k = np.concatenate([np.tile(np.concatenate([np.zeros(per_anchor), 0.3 + 0.4 * np.arange(far)]), ka), np.zeros(jiggle)])

    def state(xy):
        tree = cKDTree(xy)
        d, idx = tree.query(xy, k + 2)
        keep = idx != np.arange(n)[:, None]
        keep &= np.cumsum(keep, axis=1) <= k + 1
        knn = idx[keep].reshape(n, k + 1).astype(np.int32)      # each point's k+1 nearest others
        dk = d[keep].reshape(n, k + 1)
        hit = (nbr[:, None, :] == knn[:, :, None]).any(axis=2)   # [n, k+1]: which of them are exact neighbours
        cols = knn[:, :k].ravel()
        order = np.argsort(cols, kind="stable")                  # the lists each point is in: rev[start[i]:start[i+1]]
        return tree, knn, dk, hit, (order // k).astype(np.int32), np.searchsorted(cols[order], np.arange(n + 1))

    tree, knn, dk, hit, rev, start = state(xy)
    before = last = float(hit[:, :k].sum() / (n * k))
    ever = np.zeros(n, dtype=bool)
    dirty = np.ones(n, dtype=bool)   # the tokens worth scoring: all at first, then those a move touched
    for p in range(passes):
        t0 = time.perf_counter()
        order = rng.permutation(np.flatnonzero(dirty))
        knn0, moved_now = knn[:, :k].copy(), np.zeros(n, dtype=bool)
        for s in range(0, len(order), batch):
            if s:
                tree, knn, dk, hit, rev, start = state(xy)
            b = order[s:s + batch]
            m = len(b)
            own = hit[:, :k].sum(axis=1)
            hit_k, hit_k1, d_k = hit[:, k - 1], hit[:, k], dk[:, k - 1]
            loss = np.zeros(m)           # the lists the token leaves: each loses it and gains its (k+1)th
            for q, i in enumerate(b):
                ts = rev[start[i]:start[i + 1]]
                if len(ts):
                    loss[q] = (hit_k1[ts].astype(int) - (nbr[ts] == i).any(axis=1).astype(int)).sum()
            at = np.concatenate([np.repeat(anchors[b], pa, axis=1), np.repeat(b[:, None], jiggle, axis=1)], axis=1).ravel()
            f = np.tile(fac, m)
            if jiggle:
                f.reshape(m, c)[:, ka * pa:] = 0.3 + 1.7 * rng.random((m, jiggle))
            off = np.maximum(f * dk[at, 0] + np.tile(fac_k, m) * dk[at, k - 1], offset_floor)
            ang = rng.uniform(0, 2 * np.pi, size=m * c)
            cand = xy[at] + off[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=1)
            if quantize:
                cand = cand.astype(np.float16).astype(np.float64)
            mover, nbr_b = np.repeat(b, c), np.repeat(nbr[b], c, axis=0)
            delta = np.repeat(loss - own[b], c)
            for s0 in range(0, m * c, 8192):                      # by chunks of spots: [8192, window, k] comparisons
                sl = slice(s0, s0 + 8192)
                wd, widx = tree.query(cand[sl], window)
                widx, mv = widx.astype(np.int32), mover[sl]
                other = widx != mv[:, None]
                first = other & (np.cumsum(other, axis=1) <= k)  # the token's k nearest at the spot
                near = widx[first].reshape(-1, k)
                own_at = (near[:, :, None] == nbr_b[sl][:, None, :]).any(axis=2).sum(axis=1)
                enters = other & (wd < d_k[widx])                               # the lists the spot is inside
                in_list = (knn[widx, :k] == mv[:, None, None]).any(axis=2)     # those the token is in already
                is_nbr = (nbr[widx] == mv[:, None, None]).any(axis=2)          # those it belongs in
                gain = np.where(enters & ~in_list, is_nbr.astype(int) - hit_k[widx].astype(int), 0).sum(axis=1)
                stay = np.where(enters & in_list, is_nbr.astype(int) - hit_k1[widx].astype(int), 0).sum(axis=1)  # no loss in lists it stays in
                delta[sl] += own_at + gain + stay
            delta = delta.reshape(m, c)
            best = delta.argmax(axis=1)
            go = delta[np.arange(m), best] >= min_gain
            xy[b[go]] = cand.reshape(m, c, 2)[np.arange(m), best][go]
            moved_now[b[go]] = True
        tree, knn, dk, hit, rev, start = state(xy)
        now = float(hit[:, :k].sum() / (n * k))
        moved = int(moved_now.sum())
        ever |= moved_now
        # next time, score the tokens that moved, those with a moved anchor and those whose k nearest changed
        dirty = moved_now | moved_now[anchors].any(axis=1) | (knn[:, :k] != knn0).any(axis=1)
        stats["passes"] = p + 1
        if log:
            log(f"  neighbour pass {p + 1}: {len(order)} tokens scored, {moved} moved ({moved / n:.1%}); exact {k}-NN kept "
                f"over all tokens {now:.4f} ({time.perf_counter() - t0:.0f}s)")
        gained, last = now - last, now
        if gained < stop or not moved:
            break
    stats.update(moved=float(ever.mean()), before=round(before, 4), after=round(last, 4))
    return xy, stats


def cluster_discs(xy: np.ndarray, members: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The discs of the clusters `members` on the layout xy: (centres [k, 2]: the members' means;
    radii [k]: their 95th-percentile distances from the centre, the rule of place_members(); a
    cluster of one member, or of coincident members, gets the smallest radius per sqrt(size) of
    the others, 0 without any)."""
    xy = np.asarray(xy, dtype=np.float64)
    cxy = np.array([xy[m].mean(axis=0) if len(m) else np.zeros(2) for m in members])
    r = np.array([float(np.percentile(np.linalg.norm(xy[m] - cxy[c], axis=1), 95)) if len(m) > 1 else 0.0
                  for c, m in enumerate(members)])
    per = [r[c] / np.sqrt(len(m)) for c, m in enumerate(members) if len(m) > 1 and r[c] > 0]
    r[r == 0] = min(per) if per else 0.0
    return cxy, r


def nearest_disc(xy: np.ndarray, cxy: np.ndarray, r: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """For each point of xy, the disc (centres cxy) whose centre is nearest: in units of each
    disc's radius when the radii r are given (the membership --relabel assigns on a map of packed
    discs, where a big disc's rim is far from its centre), else by plain distance (a refined map,
    whose clusters' 95th-percentile radii can span the whole map; the share of tokens nearest
    their own cluster's centre, "in_own_disc")."""
    xy = np.asarray(xy, dtype=np.float64)
    cxy = np.asarray(cxy, dtype=np.float64)
    fr = np.ones(len(cxy)) if r is None else np.maximum(np.asarray(r, dtype=np.float64), 1e-9)
    return np.concatenate([np.argmin(np.linalg.norm(xy[s:s + chunk, None, :] - cxy[None, :, :], axis=2) / fr, axis=1)
                           for s in range(0, len(xy), chunk)]) if len(xy) else np.zeros(0, dtype=np.int64)


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
# Words a region is never named by: a label is read by everyone who opens the viewer, and these
# turned up as landmarks on the Qwen map (its vocabulary carries plenty of web text).
EXCLUDED_WORDS = frozenset("""
anal ass bitch boobs cock cum dick erotic erotica escort fuck fucking horny incontri masturbation milf naked
nude penis porn porno pornos pussy rape rencontre sex sexe sesso sexo sexual sexuelle sexy shit slut tits
vagina whore xxx assault viagra cialis prostitutas prostitute prostitution casino poker
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
    if w.lower() in EXCLUDED_WORDS:
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
    """region_summary() of the KMeans clusters of the 2D points (the flat method's regions)."""
    from sklearn.cluster import KMeans

    k = max(1, min(n_clusters, len(xy)))
    return region_summary(xy, texts, types, KMeans(n_clusters=k, n_init=4, random_state=seed).fit(xy).labels_)


def region_labels_weighted(fxy: np.ndarray, sizes: np.ndarray, n_clusters: int = 24, seed: int = 0) -> np.ndarray:
    """The region of each fine cluster: KMeans of the fine clusters' centres fxy into n_clusters
    regions, each centre weighted by its cluster's size (the hierarchical method's regions)."""
    from sklearn.cluster import KMeans

    k = max(1, min(n_clusters, len(fxy)))
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit(fxy, sample_weight=np.asarray(sizes, dtype=np.float64)).labels_


def fine_summary(xy: np.ndarray, fxy: np.ndarray, fr: np.ndarray, members: list[np.ndarray], texts: list[str],
                 words: dict[int, float]) -> list[dict]:
    """The fine clusters for the json: each disc's centre and radius (map units), size and landmark()."""
    out = []
    for c, m in enumerate(members):
        name = landmark(xy, m, texts, words)[0] if len(m) else ""
        out.append({"x": round(float(fxy[c, 0]), 4), "y": round(float(fxy[c, 1]), 4), "r": round(float(fr[c]), 4),
                    "n": int(len(m)), "landmark": name})
    return out


def region_summary(xy: np.ndarray, texts: list[str], types: np.ndarray, labels: np.ndarray,
                   words: dict[int, float] | None = None) -> list[dict]:
    """The regions of the map given each token's region label: each with a "label" (its 3 lowest-id
    printable normal tokens, a frequency proxy, kept for older pages) and a "landmark" (landmark():
    its whole words nearest its density peak, or the kind of piece it is made of), largest first."""
    labels = np.asarray(labels)
    types = np.asarray(types)
    if words is None:
        words = whole_words(texts, types)
    clusters = []
    for members in members_of(labels, int(labels.max()) + 1 if len(labels) else 0):
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
    if meta.get("fine_clusters"):  # a hierarchical map: regroup its discs, each token with the disc it is nearest to in radii
        fine = meta["fine_clusters"]
        fxy = np.array([[c["x"], c["y"]] for c in fine], dtype=np.float64)
        fr = np.array([c["r"] for c in fine], dtype=np.float64)
        labels = nearest_disc(xy, fxy, None if meta.get("refine") or meta.get("neighbour_passes") else fr)
        members = members_of(labels, len(fine))
        words = whole_words(texts, types)
        meta["fine_clusters"] = fine_summary(xy, fxy, fr, members, texts, words)
        region = region_labels_weighted(fxy, [len(m) for m in members], n_clusters)
        meta["clusters"] = region_summary(xy, texts, types, region[labels], words)
    else:
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
              pca_dims: int = 64, pca_fit: int = 40000, n_neighbors: int | None = None, min_dist: float | None = None,
              n_clusters: int = 24, n_jobs: int = 1, knn_k: int = 15, probes: list[str] = PROBES,
              method: str = "hierarchical", fine_k: int = 0, within: str = "tsne", refine: int = 1000,
              refine_exaggeration: float = 12.0, refine_start: str = "small", hub_cap: int = 30, passes: int = 12,
              knn_sample: int = 12000, log=print) -> dict:
    """The whole computation on an in-memory matrix (standardized in place): project(), then
    hierarchical_map() or layout_map() by `method`. The caller keeps w alive meanwhile; main()
    frees it between the two steps."""
    y, explained = project(w, pca_dims, pca_fit, log)
    common = dict(n_clusters=n_clusters, knn_k=knn_k, probes=probes, knn_sample=knn_sample, explained=explained,
                  pca_fit=pca_fit, dim=w.shape[1], log=log)
    if method == "hierarchical":
        return hierarchical_map(y, texts, types, seeds, fine_k=fine_k, n_neighbors=15 if n_neighbors is None else n_neighbors,
                                min_dist=0.3 if min_dist is None else min_dist, within=within, refine=refine,
                                refine_exaggeration=refine_exaggeration, refine_start=refine_start, hub_cap=hub_cap,
                                passes=passes, n_jobs=n_jobs, **common)
    if method == "flat":
        return layout_map(y, texts, types, seeds, n_neighbors=30 if n_neighbors is None else n_neighbors,
                          min_dist=0.1 if min_dist is None else min_dist, n_jobs=n_jobs, **common)
    raise ValueError(f"unknown method {method!r}")


def _meta(y: np.ndarray, xy: np.ndarray, seeds: list[int], seed: int, stability_mean: float, per_seed: list[float],
          preserved: tuple[float, float], knn_k: int, knn_sample: int, n_neighbors: int, min_dist: float,
          explained: float, pca_fit: int, dim: int | None, clusters: list[dict]) -> dict:
    """The JSON fields both methods share."""
    n = len(y)
    return {
        "n": n, "dim": int(dim if dim is not None else y.shape[1]),
        "stability_knn15": round(stability_mean, 4), "seed": int(seed),
        "seeds": seeds, "seed_overlap": {str(s): round(p, 4) for s, p in zip(seeds, per_seed)},
        "knn15_preserved": round(preserved[0], 4), "knn15_preserved_within_sample": round(preserved[1], 4),
        "knn_k": knn_k, "knn_sample": int(min(knn_sample, n)), "knn_sample_seed": 0,
        "pca_dims": int(y.shape[1]), "pca_fit_rows": int(min(pca_fit, n) if pca_fit else n),
        "pca_explained": round(explained, 4), "n_neighbors": n_neighbors, "min_dist": min_dist,
        "metric": "cosine", "standardized": True,
        "extent": [round(float(v), 4) for v in (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())],
        "landmark_rule": LANDMARK_RULE, "clusters": clusters,
    }


def layout_map(y: np.ndarray, texts: list[str], types: np.ndarray, seeds: list[int] = (0, 1, 2, 3),
               n_neighbors: int = 30, min_dist: float = 0.1, n_clusters: int = 24, n_jobs: int = 1,
               knn_k: int = 15, probes: list[str] = PROBES, knn_sample: int = 12000, explained: float = float("nan"),
               pca_fit: int = 0, dim: int | None = None, log=print) -> dict:
    """The flat method: UMAP layouts of the PCA rows y for every seed, the stability pick, scaling,
    regions and probes. Returns {"xy": scaled coordinates [n, 2], "meta": the JSON fields,
    "probes": probe neighbourhoods, "layouts": {seed: raw UMAP xy}}."""
    seeds = list(seeds)
    layouts = run_seeds(y, seeds, n_neighbors, min_dist, n_jobs, log=log)
    mean_overlap, per_seed, best = stability([layouts[s] for s in seeds], knn_k)
    seed = seeds[best]
    xy = scale_unit(layouts[seed]).astype(np.float16).astype(np.float32)  # as the .bin holds them
    preserved = knn_preserved(xy, y, knn_k, knn_sample)
    log(f"  stability (mean {knn_k}-NN overlap over {len(seeds)} seeds): {mean_overlap:.4f}; "
        f"per seed {dict(zip(seeds, [round(p, 4) for p in per_seed]))}; keeping seed {seed}; "
        f"exact {knn_k}-NN preserved on the map: {preserved[0]:.4f} ({preserved[1]:.4f} within the sample)")
    clusters = cluster_summary(xy, texts, types, n_clusters)
    rows = probe_neighbours(xy, y, texts, types, probes)
    meta = dict(_meta(y, xy, seeds, seed, mean_overlap, per_seed, preserved, knn_k, knn_sample, n_neighbors, min_dist,
                      explained, pca_fit, dim, clusters), method="flat")
    return {"xy": xy, "meta": meta, "probes": rows, "layouts": {s: layouts[s] for s in seeds}}


def hierarchical_map(y: np.ndarray, texts: list[str], types: np.ndarray, seeds: list[int] = (0, 1, 2, 3),
                     fine_k: int = 0, n_neighbors: int = 15, min_dist: float = 0.3, within: str = "tsne",
                     refine: int = 1000, refine_perplexity: float = 30.0, refine_exaggeration: float = 12.0,
                     refine_start: str = "small", hub_cap: int = 30, passes: int = 12, n_jobs: int = 1,
                     n_clusters: int = 24, knn_k: int = 15, probes: list[str] = PROBES, knn_sample: int = 12000,
                     explained: float = float("nan"), pca_fit: int = 0, dim: int | None = None, log=print) -> dict:
    """The hierarchical method (the module docstring, step 4): fine clusters of the unit PCA rows,
    their centroids laid out by UMAP for every seed (the stability pick is made on those), discs
    packed, members placed inside by their local layout, the kept seed's placement refined by a
    t-SNE of all tokens started from it (refine_layout(), `refine` iterations, from the placement
    at its own scale or scaled down by refine_start, with refine_exaggeration, the hub_cap; 0
    iterations keep the discs as placed), then up to `passes` neighbour passes (neighbour_pass():
    each token may move beside one of its exact knn_k nearest rows when that raises the whole
    vocabulary's knn_k-NN overlap; 0: none), scaling, the metrics, regions and probes. Returns
    {"xy": scaled coordinates [n, 2], "meta": the JSON fields, "probes": probe neighbourhoods,
    "layouts": {seed: raw centroid xy}, "labels": each token's fine cluster}."""
    seeds = list(seeds)
    n = len(y)
    u = unit_rows(y)
    t0 = time.perf_counter()
    labels, cent, sizes = fine_kmeans(u, fine_k or fine_count(n))
    k = len(sizes)
    members = members_of(labels, k)
    log(f"  {k} fine clusters of the unit PCA rows (MiniBatchKMeans; sizes {sizes.min()}..{sizes.max()}, "
        f"median {np.median(sizes):.0f}) in {time.perf_counter() - t0:.1f}s")
    if k > 16:
        layouts = run_seeds(cent, seeds, n_neighbors, min_dist, 1, log=log)
    else:  # too few centroids for UMAP: their 2D PCA, the same for every seed
        layouts = {s: pca2(cent)[0] for s in seeds}
        log(f"  {k} centroids laid out by PCA (too few for UMAP)")
    if k >= 3:
        c_mean, c_per, best = stability([layouts[s] for s in seeds], knn_k)
    else:
        c_mean, c_per, best = 1.0, [1.0] * len(seeds), 0
    seed = seeds[best]
    t0 = time.perf_counter()
    local = [local_layout(u[m], within) for m in members]
    log(f"  local {within} layouts of the {k} clusters in {time.perf_counter() - t0:.1f}s")
    finals = {}
    for s in seeds:
        r = disc_radii(layouts[s], sizes)
        pxy, iters, rel = pack_discs(layouts[s], r)
        finals[s] = (place_members(pxy, r, members, local, cent), pxy, r, iters, rel)
    f_mean, f_per, _ = stability([finals[s][0] for s in seeds], knn_k)
    xy_raw, pxy, r, iters, rel = finals[seed]
    centre, half = unit_transform(xy_raw)
    fxy, fr = (pxy - centre) / half, r / half
    fill = float(np.pi * (fr ** 2).sum() / max(float(np.prod(np.ptp(fxy, axis=0) + 2 * fr.mean())), 1e-12))
    log(f"  discs: {iters} packing iterations, largest remaining overlap {rel:.3f} of the target distance, "
        f"radii {fr.min():.4f}..{fr.max():.4f} map units, filling {fill:.0%} of their bounding box")
    own, kept = 1.0, 1.0
    pass_stats = {"passes": 0, "moved": 0.0, "before": None, "after": None}
    knn = None
    if refine or passes:  # the exact neighbour lists, once for the refinement and the passes
        t0 = time.perf_counter()
        kk = max(knn_k, min(int(3 * refine_perplexity) + 2, n - 1)) if refine else knn_k
        knn = exact_knn(u, kk)
        log(f"  exact {knn[0].shape[1]}-NN of the {n} unit rows by cosine in {time.perf_counter() - t0:.1f}s")
    if refine:
        t0 = time.perf_counter()
        xy_raw = refine_layout(xy_raw, u, refine_perplexity, refine, n_jobs=n_jobs, exaggeration=refine_exaggeration,
                               start=refine_start, hub_cap=hub_cap, knn=knn, log=log)
        log(f"  refined by a t-SNE of all {n} tokens started from the discs ({refine_start}, perplexity {refine_perplexity:g}, "
            f"exaggeration {refine_exaggeration:g}, {refine} iterations) in {time.perf_counter() - t0:.1f}s")
    if passes:
        t0 = time.perf_counter()
        xy_raw, pass_stats = neighbour_pass(scale_unit(xy_raw), knn[0], knn_k, passes, log=log)
        log(f"  {pass_stats['passes']} neighbour passes moved {pass_stats['moved']:.1%} of the tokens in "
            f"{time.perf_counter() - t0:.1f}s: exact {knn_k}-NN kept over all tokens {pass_stats['before']:.4f} -> "
            f"{pass_stats['after']:.4f}")
    if refine or passes:
        packed, (pxy, r) = pxy, cluster_discs(xy_raw, members)
        # the passes' spots are float16 map coordinates already: kept as they are (a new centring would round them again)
        centre, half = (np.zeros(2), 1.0) if passes else unit_transform(xy_raw)
        fxy, fr = (pxy - centre) / half, r / half
        own = float(np.mean(nearest_disc(xy_raw, pxy) == labels))
        if k > 2:  # how much of the islands' arrangement survives: the centres' 15-NN before and after
            kept = knn_overlap(knn2d(packed, min(knn_k, k - 1)), knn2d(pxy, min(knn_k, k - 1)))
        log(f"  the json's discs are now the clusters' means and 95th-percentile radii on the final layout (radii "
            f"{fr.min():.4f}..{fr.max():.4f} map units); the centres keep {kept:.3f} of their {knn_k}-NN from the packing, "
            f"and {own:.1%} of the tokens are nearest their own cluster's centre")
    xy = scale_unit(xy_raw, centre, half).astype(np.float16).astype(np.float32)  # as the .bin holds them: the metrics are the file's
    preserved = knn_preserved(xy, y, knn_k, knn_sample)
    log(f"  centroid stability (mean {knn_k}-NN overlap over {len(seeds)} seeds): {c_mean:.4f}; per seed "
        f"{dict(zip(seeds, [round(p, 4) for p in c_per]))}; keeping seed {seed}; token stability {f_mean:.4f}"
        f"{' (of the placements before the refinement)' if refine else ''}; "
        f"exact {knn_k}-NN preserved on the map: {preserved[0]:.4f} ({preserved[1]:.4f} within the sample)")
    words = whole_words(texts, types)
    fine = fine_summary(xy, fxy, fr, members, texts, words)
    region = region_labels_weighted(fxy, sizes, n_clusters)
    clusters = region_summary(xy, texts, types, region[labels], words)
    rows = probe_neighbours(xy, y, texts, types, probes)
    meta = dict(_meta(y, xy, seeds, seed, f_mean, c_per, preserved, knn_k, knn_sample, n_neighbors, min_dist,
                      explained, pca_fit, dim, clusters),
                method="hierarchical", within=within, fine_k=int(k), centroid_stability_knn15=round(c_mean, 4),
                pack_iterations=int(iters), pack_overlap=round(rel, 4), refine=int(refine),
                refine_perplexity=float(refine_perplexity) if refine else None,
                refine_exaggeration=float(refine_exaggeration) if refine else None, refine_start=refine_start if refine else None,
                refine_hub_cap=int(hub_cap) if refine else None, in_own_disc=round(own, 4),
                refine_centroid_knn15=round(kept, 4), neighbour_passes=int(pass_stats["passes"]),
                neighbour_pass_moved=round(pass_stats["moved"], 4), knn15_all_before_passes=pass_stats["before"],
                knn15_all_after_passes=pass_stats["after"], fine_clusters=fine)
    return {"xy": xy, "meta": meta, "probes": rows, "layouts": {s: layouts[s] for s in seeds}, "labels": labels}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("model", help="path to a .borg file")
    ap.add_argument("--out-dir", default=None, help="where to write the map (default: next to the model)")
    ap.add_argument("--method", choices=("hierarchical", "flat"), default="hierarchical",
                    help="hierarchical: fine clusters as packed discs laid out by UMAP, members inside by a local layout "
                         "(default); flat: one UMAP of every row")
    ap.add_argument("--seeds", type=int, default=4, help="number of UMAP seeds, 0..N-1 (default 4)")
    ap.add_argument("--n-jobs", type=int, default=2, help="flat: seeds run in parallel processes; both: BLAS threads (default 2)")
    ap.add_argument("--pca", type=int, default=64, help="PCA dimensions before the layout (default 64)")
    ap.add_argument("--pca-fit", type=int, default=40000, help="rows the PCA is fitted on; 0 = all (default 40000)")
    ap.add_argument("--fine-k", type=int, default=0, help="hierarchical: fine clusters (default 0: n / 120 within 200..1200)")
    ap.add_argument("--within", choices=("tsne", "pca"), default="tsne",
                    help="hierarchical: the layout of a cluster's members inside its disc (default tsne, started from the pca)")
    ap.add_argument("--refine", type=int, default=1000,
                    help="hierarchical: iterations of the t-SNE of all tokens started from the placed discs (default 1000; 0: none)")
    ap.add_argument("--refine-start", choices=("small", "placed"), default="small",
                    help="hierarchical: start the refinement from the placement scaled down to scikit-learn's initial spread "
                         "(default small: a t-SNE of the whole vocabulary arranged by the islands, which dissolve) or at its own "
                         "scale (placed: the islands hold; measured with --refine-exaggeration 1)")
    ap.add_argument("--refine-exaggeration", type=float, default=12.0,
                    help="hierarchical: the attraction's exaggeration for the first 250 refinement iterations (default 12, "
                         "scikit-learn's; 1: none)")
    ap.add_argument("--hub-cap", type=int, default=30,
                    help="hierarchical: in the refinement, keep each piece among the 15 nearest of at most this many others, its "
                         "nearest (default 30; 0: no cap)")
    ap.add_argument("--passes", type=int, default=12,
                    help="hierarchical: at most this many neighbour passes after the refinement, each moving tokens beside "
                         "their exact nearest rows where that raises the 15-NN kept (default 12; they stop earlier when one "
                         "gains less than 0.0002; 0: none)")
    ap.add_argument("--n-neighbors", type=int, default=None, help="UMAP n_neighbors (default 15 hierarchical, 30 flat)")
    ap.add_argument("--min-dist", type=float, default=None, help="UMAP min_dist (default 0.3 hierarchical, 0.1 flat)")
    ap.add_argument("--knn-sample", type=int, default=12000, help="tokens sampled for the knn15_preserved metric (default 12000)")
    ap.add_argument("--clusters", type=int, default=24, help="regions to label (default 24)")
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
    del w  # the float32 matrix is the largest allocation; the layout only needs y
    common = dict(seeds=list(range(a.seeds)), n_clusters=a.clusters, probes=probes, knn_sample=a.knn_sample,
                  explained=explained, pca_fit=a.pca_fit, dim=dim, log=log)
    if a.method == "hierarchical":
        res = hierarchical_map(y, texts, types, fine_k=a.fine_k, within=a.within, refine=a.refine, n_jobs=a.n_jobs,
                               refine_exaggeration=a.refine_exaggeration, refine_start=a.refine_start, hub_cap=a.hub_cap,
                               passes=a.passes,
                               n_neighbors=15 if a.n_neighbors is None else a.n_neighbors,
                               min_dist=0.3 if a.min_dist is None else a.min_dist, **common)
    else:
        res = layout_map(y, texts, types, n_jobs=a.n_jobs, n_neighbors=30 if a.n_neighbors is None else a.n_neighbors,
                         min_dist=0.1 if a.min_dist is None else a.min_dist, **common)
    meta = dict(res["meta"], model=stem, source=src, arch=header.get("arch"), quant=header.get("quant"))
    bin_path, json_path = write_map(os.path.join(out_dir, stem), res["xy"], meta)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"wrote {bin_path} ({os.path.getsize(bin_path)} bytes) and {json_path}")
    if meta.get("fine_clusters"):
        print(f"{len(meta['fine_clusters'])} fine clusters; knn15_preserved {meta['knn15_preserved']} "
              f"({meta['knn15_preserved_within_sample']} within the sample); stability_knn15 {meta['stability_knn15']} "
              f"(centroids {meta['centroid_stability_knn15']}); {meta['neighbour_passes']} neighbour passes "
              f"({meta['knn15_all_before_passes']} -> {meta['knn15_all_after_passes']} over all tokens)")
    print(f"clusters ({len(meta['clusters'])}), landmark [label]:")
    print(format_clusters(meta["clusters"]))
    print("probe neighbourhoods (map: 5 nearest on the 2D map; pca: 5 nearest by cosine in the PCA space):")
    print(format_probes(res["probes"]))
    print(f"total {time.perf_counter() - t0:.1f}s, peak RSS {rss:.0f} MB (largest worker {rss_children:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
