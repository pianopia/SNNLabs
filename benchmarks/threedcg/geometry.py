"""Geometry similarity metrics between a candidate and reference asset."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .asset import Asset


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points
    centroid = points.mean(axis=0)
    centered = points - centroid
    extent = np.abs(centered).max()
    if extent < 1e-9:
        return centered
    return centered / extent


def _mesh(asset: Asset) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)


def _sample_surface(asset: Asset, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = _mesh(asset)
    rng = np.random.default_rng(seed)
    points, face_index = trimesh.sample.sample_surface(mesh, samples, seed=int(rng.integers(1 << 31)))
    normals = mesh.face_normals[face_index]
    return normalize_points(np.asarray(points)), np.asarray(normals)


def chamfer_distance(a: Asset, b: Asset, *, samples: int = 2000) -> float:
    if np.array_equal(a.faces, b.faces) and np.allclose(a.vertices, b.vertices):
        return 0.0
    pa, _ = _sample_surface(a, samples, seed=0)
    pb, _ = _sample_surface(b, samples, seed=1)
    tree_a = cKDTree(pa)
    tree_b = cKDTree(pb)
    dist_ab, _ = tree_b.query(pa)
    dist_ba, _ = tree_a.query(pb)
    return float((dist_ab.mean() + dist_ba.mean()) / 2.0)


def volumetric_iou(a: Asset, b: Asset, *, resolution: int = 24) -> float:
    grid = np.linspace(-1.0, 1.0, resolution)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    query = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def _occupancy(asset: Asset) -> np.ndarray:
        mesh = _mesh(asset).copy()
        mesh.vertices = normalize_points(mesh.vertices)
        try:
            return mesh.contains(query)
        except Exception:
            mins = mesh.vertices.min(axis=0)
            maxs = mesh.vertices.max(axis=0)
            return np.all((query >= mins) & (query <= maxs), axis=1)

    occ_a = _occupancy(a)
    occ_b = _occupancy(b)
    union = np.logical_or(occ_a, occ_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(occ_a, occ_b).sum() / union)


def normal_consistency(a: Asset, b: Asset, *, samples: int = 2000) -> float:
    pa, na = _sample_surface(a, samples, seed=0)
    pb, nb = _sample_surface(b, samples, seed=1)
    tree_b = cKDTree(pb)
    _, idx = tree_b.query(pa)
    matched = nb[idx]
    na_unit = na / (np.linalg.norm(na, axis=1, keepdims=True) + 1e-9)
    nb_unit = matched / (np.linalg.norm(matched, axis=1, keepdims=True) + 1e-9)
    return float(np.abs((na_unit * nb_unit).sum(axis=1)).mean())


def geometry_metrics(candidate: Asset, reference: Asset) -> dict[str, float]:
    return {
        "chamfer": chamfer_distance(candidate, reference),
        "volume_iou": volumetric_iou(candidate, reference),
        "normal_consistency": normal_consistency(candidate, reference),
    }
