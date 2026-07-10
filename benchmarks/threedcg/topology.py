"""Topology metrics for a candidate asset relative to a reference budget."""

from __future__ import annotations

import trimesh

from .asset import Asset


def _mesh(asset: Asset) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)


def topology_metrics(candidate: Asset, reference: Asset) -> dict[str, float]:
    cand_faces = int(len(candidate.faces))
    ref_faces = int(len(reference.faces))
    cand_verts = int(len(candidate.vertices))
    ref_verts = int(len(reference.vertices))
    mesh = _mesh(candidate)
    return {
        "poly_count_ratio": cand_faces / ref_faces if ref_faces else 0.0,
        "vertex_count_ratio": cand_verts / ref_verts if ref_verts else 0.0,
        "is_watertight": 1.0 if mesh.is_watertight else 0.0,
        "is_manifold": 1.0 if mesh.is_winding_consistent else 0.0,
        "ngon_ratio": 0.0,
    }
