"""Texturing / PBR material metrics."""

from __future__ import annotations

from typing import Optional

from .asset import Asset

_CHANNELS = ("has_albedo", "has_normal", "has_roughness", "has_metallic")


def texture_metrics(candidate: Asset) -> dict[str, Optional[float]]:
    materials = candidate.materials
    if not materials:
        return {
            "has_material": 0.0,
            "pbr_channel_completeness": None,
            "max_texture_resolution": None,
        }
    completeness_values = []
    max_res: Optional[float] = None
    for material in materials:
        present = sum(1 for channel in _CHANNELS if material.get(channel))
        completeness_values.append(present / len(_CHANNELS))
        for width, height in material.get("texture_sizes", []):
            edge = float(max(width, height))
            max_res = edge if max_res is None else max(max_res, edge)
    return {
        "has_material": 1.0,
        "pbr_channel_completeness": float(sum(completeness_values) / len(completeness_values)),
        "max_texture_resolution": max_res,
    }
