"""Rig/skeleton structure metrics."""

from __future__ import annotations

from typing import Optional

from .asset import Asset


def hierarchy_depths(bone_parents: list[int]) -> list[int]:
    depths = [0] * len(bone_parents)
    for i in range(len(bone_parents)):
        depth = 0
        cursor = bone_parents[i]
        guard = 0
        while cursor is not None and cursor >= 0 and guard <= len(bone_parents):
            depth += 1
            cursor = bone_parents[cursor]
            guard += 1
        depths[i] = depth
    return depths


def rig_metrics(candidate: Asset, reference: Asset) -> dict[str, Optional[float]]:
    has_rig = 1.0 if candidate.bones else 0.0
    if not reference.bones:
        return {"has_rig": has_rig, "bone_count_ratio": None, "hierarchy_depth_diff": None}
    ref_depth = max(hierarchy_depths(reference.bone_parents), default=0)
    cand_depth = max(hierarchy_depths(candidate.bone_parents), default=0)
    return {
        "has_rig": has_rig,
        "bone_count_ratio": len(candidate.bones) / len(reference.bones) if reference.bones else None,
        "hierarchy_depth_diff": float(abs(cand_depth - ref_depth)),
    }
