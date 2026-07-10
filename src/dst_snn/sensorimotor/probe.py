"""Representation probes for sensorimotor latents (design B-7).

Used when a synthetic closed loop has known discrete state labels
(e.g. phase bins). Metrics measure whether the latent codes carry that
structure — not end-task classification accuracy claims.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_2d_features(features: np.ndarray | Sequence) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:
        # [N, T, D] → mean over time
        arr = arr.mean(axis=1)
    if arr.ndim != 2:
        raise ValueError("features must be [N, D] or [N, T, D]")
    return arr


def _as_labels(labels: np.ndarray | Sequence) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be 1-D")
    return arr.astype(np.int64, copy=False)


def linear_probe_accuracy(
    features: np.ndarray | Sequence,
    labels: np.ndarray | Sequence,
    *,
    train_frac: float = 0.7,
    seed: int = 0,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Hold-out multi-class linear probe via ridge least-squares one-vs-rest.

    Returns accuracy on the hold-out split plus train size metadata.
    With fewer than 2 classes or too few samples, returns 0.0 accuracy.
    """
    x = _as_2d_features(features)
    y = _as_labels(labels)
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and labels length mismatch")
    n = int(x.shape[0])
    classes = np.unique(y)
    if n < 4 or classes.size < 2:
        return {"accuracy": 0.0, "n_train": 0.0, "n_test": float(n), "n_classes": float(classes.size)}

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = max(1, min(n - 1, int(round(n * train_frac))))
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    if test_idx.size == 0:
        test_idx = order[-1:]
        train_idx = order[:-1]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    # Center features for stability.
    mean = x_train.mean(axis=0, keepdims=True)
    x_train = x_train - mean
    x_test = x_test - mean

    # One-hot targets for classes present in train.
    train_classes = np.unique(y_train)
    class_to_col = {int(c): i for i, c in enumerate(train_classes)}
    y_oh = np.zeros((x_train.shape[0], train_classes.size), dtype=np.float64)
    for i, label in enumerate(y_train):
        col = class_to_col.get(int(label))
        if col is not None:
            y_oh[i, col] = 1.0

    # Ridge: W = (X'X + λI)^{-1} X'Y
    d = x_train.shape[1]
    xtx = x_train.T @ x_train + ridge * np.eye(d)
    xty = x_train.T @ y_oh
    try:
        weights = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(xtx) @ xty
    scores = x_test @ weights
    pred_cols = scores.argmax(axis=1)
    pred = train_classes[pred_cols]
    accuracy = float((pred == y_test).mean()) if y_test.size else 0.0
    return {
        "accuracy": accuracy,
        "n_train": float(train_idx.size),
        "n_test": float(test_idx.size),
        "n_classes": float(classes.size),
    }


def nearest_centroid_accuracy(
    features: np.ndarray | Sequence,
    labels: np.ndarray | Sequence,
) -> dict[str, float]:
    """Assign each sample to the nearest class-mean centroid (leave-in)."""
    x = _as_2d_features(features)
    y = _as_labels(labels)
    classes = np.unique(y)
    if x.shape[0] == 0 or classes.size == 0:
        return {"accuracy": 0.0, "n_classes": 0.0}
    centroids = []
    for c in classes:
        centroids.append(x[y == c].mean(axis=0))
    centers = np.stack(centroids, axis=0)
    # [N, K] squared distances
    diffs = x[:, None, :] - centers[None, :, :]
    dist2 = (diffs * diffs).sum(axis=-1)
    pred = classes[dist2.argmin(axis=1)]
    return {"accuracy": float((pred == y).mean()), "n_classes": float(classes.size)}


def _kmeans(
    x: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    max_iter: int = 30,
) -> np.ndarray:
    """Simple Lloyd k-means; returns cluster assignment of shape [N]."""
    n = x.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        diffs = x[:, None, :] - centers[None, :, :]
        dist2 = (diffs * diffs).sum(axis=-1)
        new_labels = dist2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
            else:
                centers[j] = x[rng.integers(0, n)]
    return labels


def cluster_purity(
    features: np.ndarray | Sequence,
    labels: np.ndarray | Sequence,
    *,
    n_clusters: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Unsupervised k-means purity against ground-truth labels.

    purity = (1/N) * sum_k max_c |cluster_k ∩ class_c|
    """
    x = _as_2d_features(features)
    y = _as_labels(labels)
    if x.shape[0] == 0:
        return {"purity": 0.0, "n_clusters": 0.0, "n_classes": 0.0}
    classes = np.unique(y)
    k = int(n_clusters) if n_clusters is not None else int(max(1, classes.size))
    cluster_ids = _kmeans(x, k, seed=seed)
    n = x.shape[0]
    total = 0
    for j in range(k):
        mask = cluster_ids == j
        if not mask.any():
            continue
        # majority class count in this cluster
        vals, counts = np.unique(y[mask], return_counts=True)
        total += int(counts.max())
    purity = float(total) / float(n)
    return {"purity": purity, "n_clusters": float(k), "n_classes": float(classes.size)}
