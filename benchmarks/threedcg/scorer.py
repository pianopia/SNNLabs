"""Aggregate 3DCG metric families into a composite quality and RunResult."""

from __future__ import annotations

from benchmarks.threedcg.asset import Asset
from benchmarks.threedcg.geometry import geometry_metrics
from benchmarks.threedcg.rig import rig_metrics
from benchmarks.threedcg.skin import skin_metrics
from benchmarks.threedcg.texture import texture_metrics
from benchmarks.threedcg.topology import topology_metrics
from benchmarks.threedcg.uv import uv_metrics
from src.dst_snn.eval.result import MetricSet, RunResult


def score_assets(candidate: Asset, reference: Asset) -> dict[str, dict]:
    return {
        "geometry": geometry_metrics(candidate, reference),
        "topology": topology_metrics(candidate, reference),
        "uv": uv_metrics(candidate),
        "rig": rig_metrics(candidate, reference),
        "skin": skin_metrics(candidate),
        "texture": texture_metrics(candidate),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ratio_score(ratio) -> float:
    if ratio is None or ratio <= 0:
        return 0.0
    return _clamp01(1.0 - abs(1.0 - float(ratio)))


def aggregate_quality(scores: dict[str, dict]) -> float:
    """Map heterogeneous sub-metrics to higher-is-better [0, 1] and average."""
    parts: list[float] = []

    geo = scores["geometry"]
    parts.append(_clamp01(1.0 - geo["chamfer"]))
    parts.append(_clamp01(geo["volume_iou"]))
    parts.append(_clamp01(geo["normal_consistency"]))

    topo = scores["topology"]
    parts.append(_ratio_score(topo["poly_count_ratio"]))
    parts.append(_clamp01(topo["is_watertight"]))
    parts.append(_clamp01(topo["is_manifold"]))

    uv = scores["uv"]
    if uv["has_uv"]:
        parts.append(_clamp01(uv["uv_coverage"]))
        parts.append(_clamp01(1.0 - uv["uv_overlap_ratio"]))
        if uv["uv_stretch"] is not None:
            parts.append(_clamp01(1.0 - uv["uv_stretch"]))

    rig = scores["rig"]
    if rig["bone_count_ratio"] is not None:
        parts.append(_ratio_score(rig["bone_count_ratio"]))

    skin = scores["skin"]
    if skin["weight_normalization_error"] is not None:
        parts.append(_clamp01(1.0 - skin["weight_normalization_error"]))
        parts.append(_clamp01(1.0 - skin["isolated_weight_ratio"]))

    tex = scores["texture"]
    if tex["pbr_channel_completeness"] is not None:
        parts.append(_clamp01(tex["pbr_channel_completeness"]))

    return float(sum(parts) / len(parts)) if parts else 0.0


def score_to_result(
    candidate: Asset,
    reference: Asset,
    *,
    asset_id: str,
    build_latency_ms: float = 0.0,
) -> RunResult:
    scores = score_assets(candidate, reference)
    quality = aggregate_quality(scores)
    metrics = MetricSet(
        quality=quality,
        quality_metric="3dcg_composite",
        latency_ms_p50=build_latency_ms,
        latency_ms_p95=build_latency_ms,
        spikes_per_inference=0.0,
        active_neuron_fraction=0.0,
        energy_pj=0.0,
        energy_source="n/a (offline scorer)",
        param_count=0,
        model_bytes=0,
        extra={"scores": scores},
    )
    return RunResult(
        benchmark="eden14-image-to-3d",
        model="snn-3dcg",
        metrics=metrics,
        baseline=None,
        meta={"asset_id": asset_id},
    )
