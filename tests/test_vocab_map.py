"""tools/vocab_map.py: the offline 2D vocabulary map for the activation viewer.

The numpy-only parts (standardization, scaling, the k-NN overlap stability metric, the
.bin/.json files) are tested on small synthetic matrices; the UMAP/KMeans pipeline runs on a
tiny clustered matrix when umap-learn and scikit-learn are installed and is skipped otherwise.
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
                       n_clusters=k, n_jobs=1, knn_k=5, probes=["tok0", "tok50", "missing"], log=lambda s: None)
    xy, meta = res["xy"], res["meta"]
    assert xy.shape == (n, 2) and xy.dtype == np.float32
    assert np.abs(xy).max() <= 1 and np.isclose(np.abs(xy).max(), 1)
    assert meta["n"] == n and meta["dim"] == 12 and meta["pca_dims"] == 8
    assert meta["seed"] in (0, 1) and meta["seeds"] == [0, 1]
    assert 0 <= meta["stability_knn15"] <= 1 and 0 <= meta["knn15_preserved"] <= 1
    assert set(meta["seed_overlap"]) == {"0", "1"}
    assert len(meta["extent"]) == 4
    assert len(meta["clusters"]) == k and sum(c["n"] for c in meta["clusters"]) == n
    for c in meta["clusters"]:
        assert set(c) >= {"label", "x", "y", "n", "tokens"} and len(c["tokens"]) == 3
        assert -1 <= c["x"] <= 1 and -1 <= c["y"] <= 1
        assert c["label"] == " ".join(t for t, _ in c["tokens"])
    # the three blobs stay apart on the map: each token's 5 map neighbours come from its own blob
    blob = np.arange(n) // n_per
    near = vm.knn2d(xy, k=5)
    assert (blob[near] == blob[:, None]).mean() > 0.95
    assert [r["id"] for r in res["probes"]] == [0, 50, None]
    assert len(res["probes"][0]["map"]) == 5 and len(res["probes"][0]["pca"]) == 5
    # clusters are labelled by their lowest ids, and the UMAP coordinates themselves were kept
    assert all(c["tokens"][0][1] <= c["tokens"][1][1] <= c["tokens"][2][1] for c in meta["clusters"])
    assert set(res["layouts"]) == {0, 1}
