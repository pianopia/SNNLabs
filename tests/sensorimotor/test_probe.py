from __future__ import annotations

import numpy as np

from src.dst_snn.sensorimotor.probe import (
    cluster_purity,
    linear_probe_accuracy,
    nearest_centroid_accuracy,
)


def test_linear_probe_perfectly_separable():
    # Two well-separated blobs → probe should be perfect on hold-out.
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.1, size=(40, 4))
    b = rng.normal(loc=3.0, scale=0.1, size=(40, 4))
    features = np.vstack([a, b])
    labels = np.array([0] * 40 + [1] * 40)
    result = linear_probe_accuracy(features, labels, seed=0, train_frac=0.7)
    assert result["accuracy"] >= 0.9
    assert result["n_test"] >= 1


def test_cluster_purity_and_centroid():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, scale=0.05, size=(30, 3))
    b = rng.normal(loc=5.0, scale=0.05, size=(30, 3))
    features = np.vstack([a, b])
    labels = np.array([0] * 30 + [1] * 30)
    purity = cluster_purity(features, labels, seed=0)
    assert purity["purity"] >= 0.9
    centroid = nearest_centroid_accuracy(features, labels)
    assert centroid["accuracy"] >= 0.9


def test_probe_handles_too_few_samples():
    features = np.array([[0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1])
    result = linear_probe_accuracy(features, labels)
    assert "accuracy" in result
