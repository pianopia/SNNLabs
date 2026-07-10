"""Skinning / weight-paint quality metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .asset import Asset


def weight_field_smoothness(candidate: Asset) -> Optional[float]:
    """Mean absolute Laplacian of skin weights over mesh edges (lower is smoother)."""
    weights = candidate.skin_weights
    if weights is None or weights.size == 0:
        return None
    weights = np.asarray(weights, dtype=np.float64)
    n = min(len(candidate.vertices), weights.shape[0])
    if n == 0:
        return None
    # Build undirected edge adjacency from triangles, limited to weight rows.
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for face in candidate.faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        if max(a, b, c) >= n:
            continue
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    lap_abs = []
    for i in range(n):
        nbrs = [j for j in neighbors[i] if j < n]
        if not nbrs:
            continue
        mean_n = weights[nbrs].mean(axis=0)
        lap_abs.append(float(np.abs(weights[i] - mean_n).mean()))
    if not lap_abs:
        # No mesh connectivity for the weighted vertices — fall back to
        # neighbor differences along the linear vertex order.
        if n < 2:
            return 0.0
        return float(np.abs(weights[1:n] - weights[0:n - 1]).mean())
    return float(np.mean(lap_abs))


def skin_metrics(candidate: Asset) -> dict[str, Optional[float]]:
    weights = candidate.skin_weights
    if weights is None or weights.size == 0:
        return {
            "has_skin": 0.0,
            "weight_normalization_error": None,
            "max_influences": None,
            "mean_influences": None,
            "isolated_weight_ratio": None,
            "weight_smoothness": None,
        }
    weights = np.asarray(weights, dtype=np.float64)
    row_sums = weights.sum(axis=1)
    influences = (weights > 1e-6).sum(axis=1)
    isolated = (row_sums <= 1e-6).sum()
    return {
        "has_skin": 1.0,
        "weight_normalization_error": float(np.abs(row_sums - 1.0).mean()),
        "max_influences": float(influences.max()),
        "mean_influences": float(influences.mean()),
        "isolated_weight_ratio": float(isolated / len(weights)),
        "weight_smoothness": weight_field_smoothness(candidate),
    }
