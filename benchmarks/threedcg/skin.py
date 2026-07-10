"""Skinning / weight-paint quality metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .asset import Asset


def skin_metrics(candidate: Asset) -> dict[str, Optional[float]]:
    weights = candidate.skin_weights
    if weights is None or weights.size == 0:
        return {
            "has_skin": 0.0,
            "weight_normalization_error": None,
            "max_influences": None,
            "mean_influences": None,
            "isolated_weight_ratio": None,
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
    }
