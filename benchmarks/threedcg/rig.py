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


def _parent_name_pairs(bones: list[str], parents: list[int]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for i, parent in enumerate(parents):
        child = bones[i] if i < len(bones) else f"bone_{i}"
        if parent is None or parent < 0:
            pairs.add(("<root>", child))
        else:
            pname = bones[parent] if parent < len(bones) else f"bone_{parent}"
            pairs.add((pname, child))
    return pairs


def hierarchy_edit_distance(candidate: Asset, reference: Asset) -> Optional[float]:
    """Symmetric difference of parent→child name edges (Jaccard distance).

    Uses bone names for correspondence. When names do not overlap, distance is 1.
    Lower is better (0 = identical edge sets).
    """
    if not reference.bones:
        return None
    if not candidate.bones:
        return 1.0
    ref_edges = _parent_name_pairs(reference.bones, reference.bone_parents)
    cand_edges = _parent_name_pairs(candidate.bones, candidate.bone_parents)
    union = ref_edges | cand_edges
    if not union:
        return 0.0
    inter = ref_edges & cand_edges
    return float(1.0 - (len(inter) / len(union)))


def rig_metrics(candidate: Asset, reference: Asset) -> dict[str, Optional[float]]:
    has_rig = 1.0 if candidate.bones else 0.0
    if not reference.bones:
        return {
            "has_rig": has_rig,
            "bone_count_ratio": None,
            "hierarchy_depth_diff": None,
            "hierarchy_edit_distance": None,
        }
    ref_depth = max(hierarchy_depths(reference.bone_parents), default=0)
    cand_depth = max(hierarchy_depths(candidate.bone_parents), default=0)
    return {
        "has_rig": has_rig,
        "bone_count_ratio": len(candidate.bones) / len(reference.bones) if reference.bones else None,
        "hierarchy_depth_diff": float(abs(cand_depth - ref_depth)),
        "hierarchy_edit_distance": hierarchy_edit_distance(candidate, reference),
    }
