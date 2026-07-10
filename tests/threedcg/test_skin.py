from __future__ import annotations

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.skin import skin_metrics


def _skinned(weights: np.ndarray) -> Asset:
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.skin_weights = weights
    box.bones = [f"b{i}" for i in range(weights.shape[1])]
    return box


def test_no_skin_returns_none():
    out = skin_metrics(asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1))))
    assert out["has_skin"] == 0.0
    assert out["weight_normalization_error"] is None
    assert out["weight_smoothness"] is None


def test_normalized_weights():
    weights = np.array([[0.5, 0.5], [1.0, 0.0], [0.3, 0.7]])
    out = skin_metrics(_skinned(weights))
    assert out["has_skin"] == 1.0
    assert abs(out["weight_normalization_error"]) < 1e-9
    assert out["max_influences"] == 2.0
    assert out["isolated_weight_ratio"] == 0.0
    assert out["weight_smoothness"] is not None
    assert out["weight_smoothness"] >= 0.0


def test_smooth_weights_have_lower_laplacian_than_noisy():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    n = len(box.vertices)
    smooth = np.zeros((n, 2))
    smooth[:, 0] = 1.0
    noisy = np.random.default_rng(0).random((n, 2))
    noisy = noisy / noisy.sum(axis=1, keepdims=True)
    box.bones = ["a", "b"]
    box.skin_weights = smooth
    s_smooth = skin_metrics(box)["weight_smoothness"]
    box.skin_weights = noisy
    s_noisy = skin_metrics(box)["weight_smoothness"]
    assert s_smooth is not None and s_noisy is not None
    assert s_smooth < s_noisy


def test_unnormalized_and_isolated():
    weights = np.array([[0.5, 0.2], [0.0, 0.0]])
    out = skin_metrics(_skinned(weights))
    assert out["weight_normalization_error"] > 0.1
    assert out["isolated_weight_ratio"] == 0.5
