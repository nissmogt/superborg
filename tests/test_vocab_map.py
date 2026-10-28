"""tools/vocab_map.py: the offline 2D vocabulary map for the activation viewer.

The numpy-only parts (standardization, scaling, the k-NN overlap stability metric, the
neighbourhood-preservation metric, disc packing, member placement, the .bin/.json files) are
tested on small synthetic matrices; the hierarchical pipeline (MiniBatchKMeans, discs, local
t-SNE) runs on a few well separated 64-D clusters and the flat UMAP pipeline on a tiny
clustered matrix when umap-learn and scikit-learn are installed, and both are skipped otherwise.
No model file is needed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def vm():
    spec = importlib.util.spec_from_file_location("vocab_map", os.path.join(ROOT, "tools", "vocab_map.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _blobs(n_per: int = 40, dim: int = 12, k: int = 3, seed: int = 0) -> np.ndarray:
    """k well separated Gaussian blobs, n_per rows each."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim)) * 6
    return np.concatenate([centres[i] + rng.normal(size=(n_per, dim)) for i in range(k)]).astype(np.float32)


def _planar_blobs(n_per: int = 40, dim: int = 64, k: int = 6, seed: int = 0) -> np.ndarray:
    """k well separated clusters in dim dimensions, n_per rows each, every cluster a flat 2-D
    sheet (a random plane through its centre, Gaussian spread, a little noise off the plane), so
    that a good local 2-D layout can keep its neighbourhoods."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim)) * 6
    rows = []
    for i in range(k):
        plane = np.linalg.qr(rng.normal(size=(dim, 2)))[0]            # an orthonormal basis of a random plane
        rows.append(centres[i] + rng.normal(size=(n_per, 2)) @ plane.T * 1.5 + rng.normal(size=(n_per, dim)) * 0.03)
    return np.concatenate(rows).astype(np.float32)


# ---- numpy-only helpers ------------------------------------------------------------------
def test_standardize_centres_and_scales_columns(vm):
    w = (np.random.default_rng(1).normal(size=(200, 5)) * [1, 10, 100, 0.01, 3] + [5, -2, 0, 1, 0]).astype(np.float32)
    out = vm.standardize(w)
    assert out is w  # in place
    assert np.allclose(w.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(w.std(axis=0), 1, atol=1e-3)
    const = np.ones((10, 3), dtype=np.float32)
    vm.standardize(const)  # a constant column must not divide by zero
    assert np.isfinite(const).all() and np.allclose(const, 0)


def test_scale_unit_range_and_aspect(vm):
    xy = np.array([[0, 0], [4, 1], [2, -1], [4, 0.5]], dtype=np.float64)  # x spans 4, y spans 2
    s = vm.scale_unit(xy)
    assert s.dtype == np.float32 and s.shape == (4, 2)
    assert s.min() >= -1 and s.max() <= 1
    assert np.isclose(s[:, 0].min(), -1) and np.isclose(s[:, 0].max(), 1)  # the longer axis fills the range
    assert np.isclose(s[:, 1].max() - s[:, 1].min(), 1.0)  # the shorter one keeps its aspect (2/4)
    assert np.allclose((s[:, 0].min() + s[:, 0].max()) / 2, 0)  # centred on the bounding box
    assert vm.scale_unit(np.zeros((3, 2))).tolist() == [[0, 0]] * 3  # degenerate: no division by zero


def test_knn2d_excludes_self_and_handles_duplicates(vm):
    pts = np.array([[0, 0], [0, 0], [1, 0], [5, 5], [5, 5.1]], dtype=np.float64)
    idx = vm.knn2d(pts, k=2)
    assert idx.shape == (5, 2)
    assert all(i not in idx[i] for i in range(5))
    assert set(idx[0]) == {1, 2}       # its duplicate and the next closest
    assert set(idx[3]) == {4, 2}


def test_knn_overlap_counts_shared_neighbours(vm):
    a = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    assert vm.knn_overlap(a, a) == 1.0
    b = np.array([[1, 2, 9], [0, 8, 9], [7, 8, 9], [0, 1, 2]])  # 2/3, 1/3, 0, 1 shared
    assert np.isclose(vm.knn_overlap(a, b), (2 / 3 + 1 / 3 + 0 + 1) / 4)
    assert np.isclose(vm.knn_overlap(a, b[:, ::-1]), vm.knn_overlap(a, b))  # order inside a row is irrelevant
    with pytest.raises(ValueError):
        vm.knn_overlap(a, a[:, :2])


def test_stability_metric_on_synthetic_layouts(vm):
    rng = np.random.default_rng(3)
    xy = rng.normal(size=(300, 2))
    rotated = xy @ np.array([[0, -1], [1, 0]]) * 3 + 7      # a rigid transform keeps every neighbourhood
    mirrored = xy * [-1, 1]
    shuffled = xy[rng.permutation(300)]                       # the same cloud with the ids scrambled
    mean, per, best = vm.stability([xy, rotated, mirrored], k=15)
    assert mean == 1.0 and per == [1.0, 1.0, 1.0] and best == 0
    mean2, per2, best2 = vm.stability([xy, rotated, shuffled], k=15)
    assert 0.3 < mean2 < 0.45                                  # pairs: 1.0, ~0.05, ~0.05
    assert per2[2] < 0.2 and best2 in (0, 1)                   # the odd one out is never the pick
    assert per2[0] == per2[1] > 0.5
    assert vm.stability([xy], k=15) == (1.0, [1.0], 0)
    with pytest.raises(ValueError):
        vm.stability([])


def test_stability_scores_a_jittered_layout_between_identical_and_random(vm):
    rng = np.random.default_rng(5)
    xy = rng.uniform(size=(500, 2))
    jitter = xy + rng.normal(scale=0.003, size=xy.shape)
    mean, per, _best = vm.stability([xy, jitter], k=15)
    assert 0.7 < mean < 1.0 and per == [mean, mean]


def test_scale_unit_with_a_given_transform(vm):
    xy = np.array([[0, 0], [4, 1], [2, -1], [4, 0.5]], dtype=np.float64)
    centre, half = vm.unit_transform(xy)
    assert centre.tolist() == [2, 0] and half == 2
    assert np.allclose(vm.scale_unit(xy, centre, half), vm.scale_unit(xy))
    assert vm.scale_unit(np.array([[3, 0]]), centre, half).tolist() == [[0.5, 0]]   # other points in the same map units
    assert vm.unit_transform(np.zeros((0, 2)))[1] == 1.0 and vm.unit_transform(np.ones((3, 2)))[1] == 1.0   # degenerate boxes


def test_knn_preserved_on_a_planar_cloud_and_its_shuffle(vm):
    rng = np.random.default_rng(2)
    angle = rng.uniform(0, 2 * np.pi, size=400)
    xy = np.stack([np.cos(angle), np.sin(angle)], axis=1)        # on the unit circle, the Euclidean order is the cosine order
    y = np.concatenate([xy, np.zeros((400, 6))], axis=1)       # the cloud itself, padded: the map is exact
    full, within = vm.knn_preserved(xy, y, k=5, sample=400)
    assert full == 1.0 and within == 1.0
    full_s, within_s = vm.knn_preserved(xy, y, k=5, sample=100, seed=1)  # a sample of the tokens: still exact
    assert full_s == 1.0 and within_s == 1.0
    shuffled = xy[rng.permutation(400)]
    full_r, within_r = vm.knn_preserved(shuffled, y, k=5, sample=400)
    assert full_r < 0.1 and within_r < 0.1
    assert vm.knn_preserved(xy[:2], y[:2], k=5) == (1.0, 1.0)   # too few points to have neighbours
    assert vm.fine_count(49152) == 409 and vm.fine_count(151665) == 1200 and vm.fine_count(1000) == 200 and vm.fine_count(100) == 100


def test_pack_discs_separates_overlapping_discs_and_keeps_the_arrangement(vm):
    centres = np.array([[0, 0], [1, 0], [2, 0.2], [0.5, 1], [10, 10]], dtype=np.float64)   # four crowded discs and one far away
    r = np.array([1.0, 0.5, 0.5, 0.8, 1.0])
    packed, iters, rel = vm.pack_discs(centres, r, margin=0.15, tol=0.01)
    assert 0 < iters <= 400 and rel < 0.01
    d = np.linalg.norm(packed[:, None, :] - packed[None, :, :], axis=2)
    target = (r[:, None] + r[None, :]) * 1.15
    off = ~np.eye(5, dtype=bool)
    assert (d[off] >= target[off] * 0.99).all()                 # no pair closer than its target distance
    assert np.allclose(packed[4], centres[4])                   # the lone disc overlapped nothing and never moved
    order = np.argsort(packed[:4, 0])
    assert order.tolist() == [0, 1, 2, 3] or order.tolist() == [0, 3, 1, 2]   # the crowd keeps its left-to-right order
    assert np.linalg.norm(packed[0] - centres[0]) < np.linalg.norm(packed[1] - centres[1])   # the big disc moved less than the small one
    same, it2, _ = vm.pack_discs(np.zeros((2, 2)), np.ones(2))  # coincident centres get a direction
    assert it2 > 0 and np.linalg.norm(same[0] - same[1]) >= 2.3 * 0.99
    one, it1, rel1 = vm.pack_discs(np.array([[1.0, 1.0]]), np.array([3.0]))
    assert one.tolist() == [[1, 1]] and it1 == 0 and rel1 == 0


def test_place_members_fits_the_disc_and_jitters_duplicates(vm):
    rng = np.random.default_rng(4)
    p = rng.normal(size=(200, 2))
    p[10:15] = p[9]                                              # six coincident members
    layouts = [(p, np.zeros((2, 8))), (np.zeros((1, 2)), np.zeros((2, 8)))]
    members = [np.arange(200), np.array([200])]
    xy = vm.place_members(np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([2.0, 0.5]), members, layouts)
    rad = np.linalg.norm(xy[:200], axis=1)
    assert np.isclose(np.percentile(rad, 95), 2.0, atol=0.05)   # the 95th-percentile radius is the disc radius
    assert rad.max() <= 2.0 * 1.1 + 0.05                         # the outer 5% squeezed into the rim
    assert len({tuple(np.round(v, 6)) for v in xy[9:15]}) == 6   # duplicates jittered apart
    assert np.linalg.norm(xy[9:15] - xy[9], axis=1).max() < 0.2  # but kept together
    assert xy[200].tolist() == [5, 5]                            # a single member sits at its disc's centre
    p2, basis = vm.local_layout(np.concatenate([p, np.zeros((200, 6))], axis=1), "pca")
    assert basis.shape == (2, 8) and np.allclose(basis @ basis.T, np.eye(2), atol=1e-6) and np.allclose(basis[:, 2:], 0)
    assert np.allclose(np.linalg.norm(p2, axis=1), np.linalg.norm(p - p.mean(axis=0), axis=1))   # the PCA of a planar cloud is the cloud
    few, _ = vm.local_layout(p[:5], "tsne")                      # fewer than 10 rows: the PCA
    assert np.allclose(few, p[:5] - p[:5].mean(axis=0)) or np.allclose(np.linalg.norm(few, axis=1), np.linalg.norm(p[:5] - p[:5].mean(axis=0), axis=1))
    with pytest.raises(ValueError):
        vm.local_layout(p, "spiral")


def test_write_and_read_map_roundtrip(vm, tmp_path):
    xy = vm.scale_unit(np.random.default_rng(0).normal(size=(50, 2)))
    stem = str(tmp_path / "tiny-model")
    bin_path, json_path = vm.write_map(stem, xy, {"stability_knn15": 0.9, "seed": 1, "clusters": []})
    assert bin_path.endswith("tiny-model.vocab_map.bin") and json_path.endswith("tiny-model.vocab_map.json")
    assert os.path.getsize(bin_path) == 50 * 2 * 2                       # V x 2 float16
    with open(bin_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype="<f2").reshape(50, 2)
    assert np.allclose(raw, xy, atol=2e-3)                               # float16 precision in [-1, 1]
    back = vm.read_map(bin_path)
    assert back.shape == (50, 2) and back.dtype == np.float32 and np.abs(back).max() <= 1
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["n"] == 50 and meta["format"] == vm.FORMAT and meta["bin"] == "tiny-model.vocab_map.bin"
    assert meta["stability_knn15"] == 0.9 and meta["seed"] == 1 and meta["clusters"] == []
    with pytest.raises(ValueError):
        vm.write_map(stem, np.array([[1.5, 0.0]]), {})                   # out of [-1, 1]
    with pytest.raises(ValueError):
        vm.write_map(stem, np.zeros((3, 3)), {})                         # not [n, 2]


def test_find_token_and_probe_formatting(vm):
    texts = ["<s>", " Paris", "Paris", " cat", "x"]
    types = np.array([2, 0, 0, 0, 0], dtype=np.uint8)
    assert vm.find_token(texts, " Paris", types) == 1
    assert vm.find_token(texts, " x", types) == 4          # falls back to the space-less token
    assert vm.find_token(texts, " dog", types) is None
    xy = np.array([[0, 0], [0.1, 0], [0.15, 0], [0.9, 0.9], [0.95, 0.9]])  # from ' Paris': 'Paris' then '<s>'
    rows = vm.probe_neighbours(xy, None, texts, types, probes=[" Paris", " dog"], k=2)
    assert rows[0]["id"] == 1 and [t for t, _, _ in rows[0]["map"]] == ["Paris", "<s>"]
    assert rows[1]["id"] is None
    text = vm.format_probes(rows)
    assert "' Paris' (id 1)" in text and "not a single token" in text


# ---- landmarks -------------------------------------------------------------------------------
def test_readable_word_and_piece_kind(vm):
    assert vm.readable_word(" Paris") == "Paris" and vm.readable_word(" city") == "city"
    assert vm.readable_word("Paris") is None                                  # not word-initial: no space marker
    assert vm.readable_word(" the") is None and vm.readable_word(" The") is None and vm.readable_word(" would") is None
    assert vm.readable_word(" ab") is None and vm.readable_word(" " + "a" * 13) is None   # 3-12 letters
    assert vm.readable_word(" " + "a" * 12) == "a" * 12 and vm.readable_word(" abc") == "abc"
    assert vm.readable_word(" don't") is None and vm.readable_word(" 123") is None and vm.readable_word(" x1") is None
    assert vm.readable_word(" LGBT") is None and vm.readable_word(" QtWidgets") is None and vm.readable_word(" iPhone") is None
    assert vm.readable_word("") is None and vm.readable_word(" ") is None and vm.readable_word("  abc") is None
    assert vm.piece_kind("12") == "numbers" == vm.piece_kind(" 12")
    assert vm.piece_kind("!") == "punctuation" == vm.piece_kind("--") == vm.piece_kind(" ...")
    assert vm.piece_kind("ing") == "fragments" == vm.piece_kind(" stud")
    assert vm.piece_kind("foo_bar") == "code" == vm.piece_kind("    ") == vm.piece_kind("x1") == vm.piece_kind("<div>") == vm.piece_kind("\n")


def test_word_share_separates_words_from_fragments(vm):
    # ids follow merge order: a fragment's extensions come right after it, a word's derivations much later
    texts = [" includ", " include", " included", " includes", " water", " city"] + [f"f{i}" for i in range(40)] + [" watered", " citywide", "waterfall"]
    share = vm.word_share(texts, range(len(texts)))
    assert share[0] < 0                                            # " includ": 1 - 1 * (1/2 + 1/3 + 1/4)
    assert np.isclose(share[0], 1 - (1 / 2 + 1 / 3 + 1 / 4))
    assert share[1] < vm.WORD_SHARE_MIN                             # " include" is followed at once by " included", " includes"
    assert share[2] == 1.0 and share[3] == 1.0                      # nothing extends them
    assert np.isclose(share[4], 1 - 5 / 47) and share[4] > vm.WORD_SHARE_MIN   # " water": " watered" at id 46; "waterfall" (no space) is not an extension
    assert np.isclose(share[5], 1 - 6 / 48)
    assert set(share) == set(range(len(texts)))
    assert vm.word_share(texts, []) == {}


def _synthetic_vocabulary():
    """Six tight regions on the map: words (with stop words, a fragment and its extensions, an added token,
    a far-off word, a name, a lowercase word without a capitalized twin and a capitalized fragment), a
    second word region with one word outside the inner half, and regions of digits, punctuation, code
    and word fragments. Returns (texts, types, xy, {region name: centre})."""
    rng = np.random.default_rng(7)
    texts, types, xy = [], [], []
    centres = {"words": (0.6, 0.6), "numbers": (-0.6, 0.6), "punctuation": (-0.6, -0.6), "code": (0.6, -0.6),
               "fragments": (0.0, 0.0), "words2": (0.0, 0.8)}

    def put(region, pieces, kind=0, offset=(0.0, 0.0), spread=0.0):
        for t in pieces:
            texts.append(t)
            types.append(kind)
            xy.append(np.array(centres[region]) + offset + rng.normal(scale=spread, size=2))

    put("words", [" the", " and"])                     # ids 0, 1: stop words, never a landmark
    put("words", [" Special"], kind=3)                 # id 2: an added token, not a candidate
    put("words", [" qwerty"])                          # id 3: lowercase without " Qwerty" in the vocabulary: not a word
    put("words", [" zebra"], offset=(0.25, 0.0))       # id 4: a word, but outside the region's inner half
    put("words", [" paris", " city", " france"])       # ids 5-7: the landmark
    put("words", [" includ", " include", " included", " includes", " water"])   # ids 8-12: fragment, near-fragment, words
    put("words2", [" alpha", " beta"])                 # ids 13, 14
    put("words2", [" gamma"], offset=(0.0, -0.25))     # id 15: outside the inner half, but fewer than three words sit inside
    put("words", [" Europe"])                          # id 16: a name, no lowercase form: a word
    put("words", [" Anthrop", " anthrop", " anthropology"])   # ids 17-19: a capitalized fragment whose lowercase form is one
    put("words", [" QtWidgets", " LGBT"])              # ids 20, 21: not readable
    put("words", [f"wa{i}" for i in range(40)], spread=0.02)
    put("numbers", [str(d) for d in range(10)] + [" 10", " 2024", "100"], spread=0.02)
    put("punctuation", ["!", "?", ",", ".", ";", ":", "--", "(", ")", "...", '"', "'"], spread=0.02)
    put("code", ["foo_bar", "x1", "    ", "<div>", "self.", "();", "__init__", "arr[0]", "\n", "\t"], spread=0.02)
    put("fragments", ["ing", "tion", "ness", "ment", "able", "ous", "ly", "ed"], spread=0.02)
    put("words2", [f"wb{i}" for i in range(20)], spread=0.02)
    put("words", [" Zebra", " Paris", " City", " France", " Included", " Water", " Alpha", " Beta", " Gamma"])   # the capitalized twins, at high ids
    return texts, np.array(types, dtype=np.uint8), np.array(xy, dtype=np.float32), centres


def test_landmark_rule_on_a_synthetic_vocabulary(vm):
    texts, types, xy, centres = _synthetic_vocabulary()
    cands = [i for i in range(len(texts)) if types[i] == vm.TOKEN_NORMAL and vm.readable_word(texts[i])]
    assert 2 not in cands and 0 not in cands and 5 in cands and 20 not in cands and 21 not in cands
    share = vm.word_share(texts, cands)
    assert share[8] < vm.WORD_SHARE_MIN and share[10] >= vm.WORD_SHARE_MIN   # " includ" is a fragment, " included" a word
    assert share[18] < vm.WORD_SHARE_MIN                                      # " anthrop": " anthropology" follows at once
    words = vm.whole_words(texts, types)
    twins = [i for i, t in enumerate(texts) if t in (" Zebra", " Paris", " City", " France", " Included", " Water", " Alpha", " Beta", " Gamma")]
    assert set(words) == {4, 5, 6, 7, 10, 12, 13, 14, 15, 16} | set(twins)
    assert all(v >= vm.WORD_SHARE_MIN for v in words.values())
    region = lambda name: [i for i in range(len(texts)) if np.linalg.norm(xy[i] - centres[name]) < 0.3]
    assert vm.landmark(xy, region("words"), texts, words) == ("paris · city · france", [[" paris", 5], [" city", 6], [" france", 7]])
    assert vm.landmark(xy, region("words2"), texts, words) == ("alpha · beta · gamma", [[" alpha", 13], [" beta", 14], [" gamma", 15]])
    for name in ("numbers", "punctuation", "code", "fragments"):
        assert vm.landmark(xy, region(name), texts, words) == (name, [])
    # a lone word among many pieces does not name the region: fewer than one member in twenty
    many = region("numbers") + region("fragments") + region("code") + [5]
    assert vm.landmark(xy, many, texts, words) == ("numbers", [])
    assert vm.landmark(xy, region("code") + [5, 6, 7], texts, words)[0] == "paris · city · france"   # 3 of 13: words
    peak = vm.density_peak(xy[region("words")])
    assert np.linalg.norm(peak - centres["words"]) < 0.05 and vm.density_peak(np.zeros((1, 2))).tolist() == [0, 0]


def test_cluster_summary_carries_landmarks(vm):
    pytest.importorskip("sklearn")
    texts, types, xy, centres = _synthetic_vocabulary()
    clusters = vm.cluster_summary(xy, texts, types, n_clusters=len(centres))
    assert len(clusters) == len(centres) and sum(c["n"] for c in clusters) == len(texts)
    by_region = {min(centres, key=lambda k: np.hypot(c["x"] - centres[k][0], c["y"] - centres[k][1])): c for c in clusters}
    assert set(by_region) == set(centres)
    assert by_region["words"]["landmark"] == "paris · city · france" and by_region["words"]["landmark_tokens"] == [[" paris", 5], [" city", 6], [" france", 7]]
    assert by_region["words2"]["landmark"] == "alpha · beta · gamma"
    for name in ("numbers", "punctuation", "code", "fragments"):
        assert by_region[name]["landmark"] == name and by_region[name]["landmark_tokens"] == []
    for c in clusters:  # the compatibility label is still the three lowest printable ids
        assert c["label"] == " ".join(vm._label_text(t) for t, _ in c["tokens"]) and len(c["tokens"]) == 3
    assert by_region["words"]["label"] == "the and qwerty"   # the added token " Special" is skipped, as before


# ---- the UMAP + KMeans pipeline on a tiny matrix --------------------------------------------
def test_build_map_on_synthetic_clusters(vm):
    pytest.importorskip("umap")
    pytest.importorskip("sklearn")
    n_per, k = 40, 3
    w = _blobs(n_per=n_per, dim=12, k=k)
    n = len(w)
    texts = [f"tok{i}" for i in range(n)]
    types = np.zeros(n, dtype=np.uint8)
    res = vm.build_map(w, texts, types, seeds=[0, 1], pca_dims=8, pca_fit=0, n_neighbors=10, min_dist=0.1,
                       n_clusters=k, n_jobs=1, knn_k=5, probes=["tok0", "tok50", "missing"], method="flat", log=lambda s: None)
    xy, meta = res["xy"], res["meta"]
    assert xy.shape == (n, 2) and xy.dtype == np.float32
    assert np.abs(xy).max() <= 1 and np.isclose(np.abs(xy).max(), 1)
    assert meta["n"] == n and meta["dim"] == 12 and meta["pca_dims"] == 8 and meta["method"] == "flat"
    assert "fine_clusters" not in meta and meta["n_neighbors"] == 10 and meta["min_dist"] == 0.1
    assert meta["seed"] in (0, 1) and meta["seeds"] == [0, 1]
    assert 0 <= meta["stability_knn15"] <= 1 and 0 <= meta["knn15_preserved"] <= 1
    assert 0 <= meta["knn15_preserved_within_sample"] <= 1 and meta["knn_sample"] == n
    assert set(meta["seed_overlap"]) == {"0", "1"}
    assert len(meta["extent"]) == 4
    assert len(meta["clusters"]) == k and sum(c["n"] for c in meta["clusters"]) == n
    for c in meta["clusters"]:
        assert set(c) >= {"label", "landmark", "landmark_tokens", "x", "y", "n", "tokens"} and len(c["tokens"]) == 3
        assert -1 <= c["x"] <= 1 and -1 <= c["y"] <= 1
        assert c["label"] == " ".join(t for t, _ in c["tokens"])
        assert c["landmark"] == "code" and c["landmark_tokens"] == []   # "tokN" pieces (letters and digits): no whole word anywhere
    assert meta["landmark_rule"] == vm.LANDMARK_RULE
    # the three blobs stay apart on the map: each token's 5 map neighbours come from its own blob
    blob = np.arange(n) // n_per
    near = vm.knn2d(xy, k=5)
    assert (blob[near] == blob[:, None]).mean() > 0.95
    assert [r["id"] for r in res["probes"]] == [0, 50, None]
    assert len(res["probes"][0]["map"]) == 5 and len(res["probes"][0]["pca"]) == 5
    # clusters are labelled by their lowest ids, and the UMAP coordinates themselves were kept
    assert all(c["tokens"][0][1] <= c["tokens"][1][1] <= c["tokens"][2][1] for c in meta["clusters"])
    assert set(res["layouts"]) == {0, 1}


# ---- the hierarchical pipeline on a few well separated 64-D clusters --------------------------
def _check_hierarchical(vm, res, n_per, k, n_clusters):
    xy, meta = res["xy"], res["meta"]
    n = n_per * k
    assert xy.shape == (n, 2) and xy.dtype == np.float32
    assert np.abs(xy).max() <= 1 and np.abs(xy).max() > 0.95   # scaled to [-1, 1]; the passes may move the outermost points inwards
    assert np.array_equal(xy, xy.astype(np.float16).astype(np.float32))                           # the coordinates the .bin holds
    assert meta["method"] == "hierarchical" and meta["fine_k"] == k and meta["n"] == n
    fine = meta["fine_clusters"]
    assert len(fine) == k and all(set(c) == {"x", "y", "r", "n", "landmark"} for c in fine)
    assert sorted(c["n"] for c in fine) == [n_per] * k and all(c["r"] > 0 for c in fine)
    # the fine clusters are the blobs, and every member sits nearer its own disc's centre than any other's
    labels, blob = res["labels"], np.arange(n) // n_per
    assert all(len(set(labels[blob == b])) == 1 for b in range(k)) and len(set(labels)) == k
    centres = np.array([[c["x"], c["y"]] for c in fine])
    nearest = np.argmin(np.linalg.norm(xy[:, None, :] - centres[None, :, :], axis=2), axis=1)
    assert (nearest == labels).all()
    # the metrics are there, in range, and the sheets' neighbourhoods are kept
    for key in ("knn15_preserved", "knn15_preserved_within_sample", "stability_knn15", "centroid_stability_knn15"):
        assert 0 <= meta[key] <= 1, key
    assert meta["knn_sample"] == n and meta["knn_k"] == 5 and meta["pack_overlap"] < 0.01
    assert meta["seed"] in meta["seeds"] and set(meta["seed_overlap"]) == {str(s) for s in meta["seeds"]}
    assert len(meta["extent"]) == 4 and meta["landmark_rule"] == vm.LANDMARK_RULE
    assert len(meta["clusters"]) == n_clusters and sum(c["n"] for c in meta["clusters"]) == n
    for c in meta["clusters"]:
        assert set(c) >= {"label", "landmark", "landmark_tokens", "x", "y", "n", "tokens"} and len(c["tokens"]) == 3
        assert c["n"] % n_per == 0   # a region is a union of whole fine clusters
    return meta


def test_hierarchical_map_on_synthetic_clusters(vm):
    pytest.importorskip("sklearn")
    n_per, k = 40, 6
    w = _planar_blobs(n_per=n_per, dim=64, k=k)
    n = len(w)
    texts = [f"tok{i}" for i in range(n)]
    types = np.zeros(n, dtype=np.uint8)
    res = vm.build_map(w.copy(), texts, types, seeds=[0, 1], pca_dims=64, pca_fit=0, n_clusters=3, knn_k=5, refine=0,
                       method="hierarchical", fine_k=k, within="tsne", probes=["tok0", "missing"], log=lambda s: None)
    meta = _check_hierarchical(vm, res, n_per, k, 3)
    assert meta["within"] == "tsne" and meta["n_neighbors"] == 15 and meta["min_dist"] == 0.3
    assert meta["refine"] == 0 and meta["refine_perplexity"] is None
    assert meta["knn15_preserved"] > 0.5 and meta["knn15_preserved_within_sample"] > 0.5   # flat sheets: the local t-SNE keeps them
    assert meta["stability_knn15"] > 0.9                                                     # the seeds only move the discs
    assert all(c["landmark"] == "code" for c in meta["fine_clusters"])                        # "tokN": no whole words anywhere
    assert [r["id"] for r in res["probes"]] == [0, None] and len(res["probes"][0]["map"]) == 5
    # the plain PCA placement keeps the sheets exactly
    res_pca = vm.build_map(w.copy(), texts, types, seeds=[0], pca_dims=64, pca_fit=0, n_clusters=3, knn_k=5, refine=0,
                           method="hierarchical", fine_k=k, within="pca", probes=[], log=lambda s: None)
    meta_pca = _check_hierarchical(vm, res_pca, n_per, k, 3)
    assert meta_pca["within"] == "pca" and meta_pca["knn15_preserved"] > 0.8 and meta_pca["stability_knn15"] == 1.0
    with pytest.raises(ValueError):
        vm.build_map(w.copy(), texts, types, method="spiral", log=lambda s: None)


def test_hierarchical_map_lays_many_centroids_out_with_umap(vm):
    pytest.importorskip("umap")
    pytest.importorskip("sklearn")
    n_per, k = 15, 24   # more than 16 fine clusters: their centroids go through UMAP, once per seed
    w = _planar_blobs(n_per=n_per, dim=64, k=k, seed=3)
    n = len(w)
    texts = [f"tok{i}" for i in range(n)]
    res = vm.build_map(w, texts, np.zeros(n, dtype=np.uint8), seeds=[0, 1], pca_dims=32, pca_fit=0, n_clusters=4, knn_k=5,
                       method="hierarchical", fine_k=k, within="pca", refine=0, probes=[], log=lambda s: None)
    meta = _check_hierarchical(vm, res, n_per, k, 4)
    assert meta["pca_dims"] == 32 and set(res["layouts"]) == {0, 1} and res["layouts"][0].shape == (k, 2)
    assert meta["knn15_preserved"] > 0.8


# ---- the refinement: a t-SNE of every token started from the placed discs ---------------------
def test_exact_knn_matches_a_brute_force_search(vm):
    rng = np.random.default_rng(4)
    u = vm.unit_rows(rng.normal(size=(50, 8)))
    idx, dist = vm.exact_knn(u, 5, chunk=7)   # several chunks, the last one short
    sims = u @ u.T
    np.fill_diagonal(sims, -np.inf)
    ref = np.argsort(-sims, axis=1)[:, :5]
    assert idx.shape == (50, 5) and idx.dtype == np.int32 and (idx == ref).all()
    assert np.allclose(dist, 1 - np.take_along_axis(sims, ref, axis=1), atol=1e-5) and (dist >= 0).all()
    assert vm.exact_knn(u[:1], 5)[0].shape == (1, 0)   # a single row has no other rows


def test_refine_layout_recovers_a_sheet_from_a_random_start(vm):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(5)
    n = 300
    plane = np.linalg.qr(rng.normal(size=(64, 2)))[0]
    sheet = rng.normal(size=(n, 2))
    u = vm.unit_rows(sheet @ plane.T * 1.5 + rng.normal(size=(n, 64)) * 0.03 + 5 * rng.normal(size=64))
    start = rng.normal(size=(n, 2))
    assert vm.knn_preserved(start, u, 15)[0] < 0.15                        # the start knows nothing
    small = vm.refine_layout(start, u, perplexity=30, iters=500)           # the default: scikit-learn's small start, early exaggeration
    assert small.shape == (n, 2) and small.dtype == np.float64
    assert vm.knn_preserved(small, u, 15)[0] > 0.6                         # the sheet's neighbourhoods come back
    rough = sheet + rng.normal(scale=0.35, size=sheet.shape)               # the sheet, roughly placed
    before = vm.knn_preserved(rough, u, 15)[0]
    assert 0.2 < before < 0.55
    placed = vm.refine_layout(rough, u, perplexity=30, iters=500, start="placed", exaggeration=1.0)   # from the placement, at its scale
    assert vm.knn_preserved(placed, u, 15)[0] > max(0.6, before + 0.1)     # the neighbourhoods are recovered from where they were
    assert np.array_equal(vm.refine_layout(start, u, iters=0), start)                   # no iterations: kept as is
    assert np.array_equal(vm.refine_layout(start[:9], u[:9], iters=500), start[:9])   # too few rows: kept as is
    assert vm.refine_layout(start[:12], u[:12], iters=250).shape == (12, 2)          # the perplexity is clamped
    with pytest.raises(ValueError):
        vm.refine_layout(start, u, iters=250, start="spiral")


def test_cap_hubs_keeps_a_hubs_nearest_rows(vm):
    idx = np.array([[5, 1], [5, 2], [5, 3], [5, 4], [5, 0], [0, 1]])   # token 5 is among the 2 nearest of five rows
    dist = np.array([[0.1, 0.5], [0.3, 0.5], [0.2, 0.5], [0.4, 0.5], [0.05, 0.5], [0.1, 0.2]], dtype=np.float32)
    out, hubs, pushed = vm.cap_hubs(idx, dist.copy(), cap=2, k=2)
    assert hubs == 1 and np.isclose(pushed, 3 / 12)
    assert out[4, 0] == 0.05 and out[0, 0] == 0.1                           # its two nearest rows keep it
    assert out[1, 0] == out[2, 0] == out[3, 0] == 2.0                       # the others see it at the largest cosine distance
    assert np.array_equal(out[:, 1], dist[:, 1]) and out[5, 0] == 0.1       # nothing else touched
    same, hubs0, pushed0 = vm.cap_hubs(idx, dist.copy(), cap=0, k=2)
    assert hubs0 == 0 and pushed0 == 0.0 and np.array_equal(same, dist)     # no cap: nothing changes
    wide, _, _ = vm.cap_hubs(idx, dist.copy(), cap=2, k=5)                  # k beyond the lists' width: all columns count
    assert wide[1, 0] == 2.0


def test_neighbour_pass_brings_strays_back_to_their_neighbours(vm):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(8)
    n = 400
    plane = np.linalg.qr(rng.normal(size=(64, 2)))[0]
    sheet = rng.normal(size=(n, 2))
    u = vm.unit_rows(sheet @ plane.T * 1.5 + rng.normal(size=(n, 64)) * 0.03 + 5 * rng.normal(size=64))
    nbr, _ = vm.exact_knn(u, 15)
    xy = sheet.copy()
    strays = rng.choice(n, 60, replace=False)                 # 15% of the points dropped somewhere else on the sheet
    xy[strays] = rng.normal(size=(60, 2))
    before = vm.knn_overlap(vm.knn2d(xy, 15), nbr)
    ceiling = vm.knn_overlap(vm.knn2d(sheet, 15), nbr)         # what the sheet itself keeps (its noise costs a little)
    assert 0.6 < before < 0.8 < 0.9 < ceiling
    out, stats = vm.neighbour_pass(xy, nbr, 15, passes=6, batch=64, offset_floor=1e-6)
    assert out.shape == (n, 2) and out.dtype == np.float64 and not np.shares_memory(out, xy)
    after = vm.knn_overlap(vm.knn2d(out, 15), nbr)
    assert np.isclose(stats["before"], before, atol=1e-4) and np.isclose(stats["after"], after, atol=1e-4)
    assert after > max(ceiling - 0.03, before + 0.15) and 1 <= stats["passes"] <= 6 and 0.15 < stats["moved"] < 0.6
    # the strays went to their neighbours: each now sits nearer the mean of its exact 15 than it did
    far = np.linalg.norm(xy[strays] - sheet[nbr[strays]].mean(axis=1), axis=1)
    near = np.linalg.norm(out[strays] - out[nbr[strays]].mean(axis=1), axis=1)
    assert (near < far).mean() > 0.8
    assert (np.linalg.norm(out - xy, axis=1)[np.setdiff1d(np.arange(n), strays)] < 0.5).mean() > 0.9   # the others barely move
    kept, none = vm.neighbour_pass(xy, nbr, 15, passes=0)
    assert np.array_equal(kept, xy) and none == {"passes": 0, "moved": 0.0, "before": None, "after": None}
    assert vm.neighbour_pass(xy[:10], nbr[:10], 15, passes=3)[1]["passes"] == 0   # too few points: left alone


def test_cluster_discs_are_means_and_95th_percentile_radii(vm):
    rng = np.random.default_rng(6)
    xy = np.concatenate([rng.normal(size=(40, 2)), [[10.0, 10.0]] * 20, [[-5.0, 3.0]]])
    members = [np.arange(40), np.arange(40, 60), np.array([60])]
    cxy, r = vm.cluster_discs(xy, members)
    assert np.allclose(cxy[0], xy[:40].mean(axis=0)) and np.allclose(cxy[1], [10, 10]) and np.allclose(cxy[2], [-5, 3])
    assert np.isclose(r[0], np.percentile(np.linalg.norm(xy[:40] - cxy[0], axis=1), 95))
    assert r[1] == r[2] == r[0] / np.sqrt(40)   # coincident members, a single member: the smallest radius per sqrt(size) of the others
    assert vm.cluster_discs(xy[:5], [np.arange(5)])[1][0] > 0 and vm.cluster_discs(xy[:1], [np.arange(1)])[1][0] == 0
    # nearest_disc: by plain distance, or in units of each disc's radius (a big disc claims a point nearer a small one's centre)
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 10.0]])
    assert vm.nearest_disc(pts, cxy).tolist() == [0, 0, 1]
    assert vm.nearest_disc(pts, np.array([[0.0, 0.0], [4.0, 0.0]]), np.array([10.0, 1.0]), chunk=2).tolist() == [0, 0, 0]
    assert vm.nearest_disc(pts, np.array([[0.0, 0.0], [4.0, 0.0]])).tolist() == [0, 1, 1]
    assert vm.nearest_disc(pts[:0], cxy).shape == (0,)


def test_hierarchical_map_refined_keeps_its_clusters(vm):
    pytest.importorskip("sklearn")
    n_per, k = 40, 6
    w = _planar_blobs(n_per=n_per, dim=64, k=k, seed=7)
    n = len(w)
    texts = [f"tok{i}" for i in range(n)]
    res = vm.build_map(w, texts, np.zeros(n, dtype=np.uint8), seeds=[0], pca_dims=64, pca_fit=0, n_clusters=3, knn_k=5,
                       method="hierarchical", fine_k=k, within="pca", refine=500, probes=["tok0"], log=lambda s: None)
    meta = _check_hierarchical(vm, res, n_per, k, 3)   # every member still nearer its own disc's centre; fields present
    assert meta["refine"] == 500 and meta["refine_perplexity"] == 30.0
    assert meta["refine_start"] == "small" and meta["refine_exaggeration"] == 12.0 and meta["refine_hub_cap"] == 30
    assert meta["in_own_disc"] == 1.0 and 0 <= meta["refine_centroid_knn15"] <= 1
    assert 1 <= meta["neighbour_passes"] <= 12 and 0 <= meta["neighbour_pass_moved"] <= 1     # the passes ran after the refinement
    assert meta["knn15_all_after_passes"] >= meta["knn15_all_before_passes"] > 0.5
    assert meta["knn15_preserved"] > 0.5 and meta["knn15_preserved_within_sample"] > 0.5
    # the json's discs are the clusters as refined: each the mean of its members' final positions
    centres = np.array([[c["x"], c["y"]] for c in meta["fine_clusters"]])
    for c in range(k):
        assert np.allclose(res["xy"][res["labels"] == c].mean(axis=0), centres[c], atol=1e-3)
    # the other refinement options: the start from the placement at its own scale, no exaggeration, no hub cap
    res2 = vm.build_map(w.copy(), texts, np.zeros(n, dtype=np.uint8), seeds=[0], pca_dims=64, pca_fit=0, n_clusters=3, knn_k=5,
                        method="hierarchical", fine_k=k, within="pca", refine=300, refine_start="placed", refine_exaggeration=1.0,
                        hub_cap=0, passes=0, probes=[], log=lambda s: None)
    meta2 = res2["meta"]
    assert meta2["refine_start"] == "placed" and meta2["refine_exaggeration"] == 1.0 and meta2["refine_hub_cap"] == 0
    assert meta2["neighbour_passes"] == 0 and meta2["neighbour_pass_moved"] == 0.0 and meta2["knn15_all_after_passes"] is None
    assert meta2["in_own_disc"] > 0.9 and meta2["knn15_preserved"] > 0.5 and np.abs(res2["xy"]).max() <= 1
    unrefined = vm.build_map(w.copy(), texts, np.zeros(n, dtype=np.uint8), seeds=[0], pca_dims=64, pca_fit=0, n_clusters=3,
                             method="hierarchical", fine_k=k, within="pca", refine=0, probes=[], log=lambda s: None)["meta"]
    assert unrefined["refine"] == 0 and unrefined["refine_start"] is None and unrefined["refine_hub_cap"] is None
