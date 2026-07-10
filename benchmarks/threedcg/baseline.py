"""Trivial convex-hull baseline generator to validate the 3DCG scorer."""

from __future__ import annotations

import trimesh

from .asset import Asset, asset_from_trimesh
from .scorer import score_to_result
from src.dst_snn.eval.result import RunResult


def convex_hull_candidate(reference: Asset) -> Asset:
    mesh = trimesh.Trimesh(vertices=reference.vertices, faces=reference.faces, process=False)
    return asset_from_trimesh(mesh.convex_hull)


def run_baseline(reference: Asset, *, asset_id: str) -> RunResult:
    candidate = convex_hull_candidate(reference)
    return score_to_result(candidate, reference, asset_id=asset_id)
