"""End-to-end image → 3D candidate → harness score."""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Union
import time

import numpy as np

from benchmarks.threedcg.asset import Asset
from benchmarks.threedcg.scorer import score_to_result
from src.dst_snn.eval.result import RunResult

from .image_spikes import image_to_spikes, load_image_array
from .mesh_backend import execute_ops_backend
from .track1_policy import decode_ops_from_spikes
from .track2_occupancy import track2_from_spikes

TrackName = Literal[
    "track1",
    "track2",
    "track1_scripted",
    "track2_occupancy",
    "track1_trained",
    "track2_trained",
    "track1_sequence",
    "track2_sdf",
]
ImageLike = Union[np.ndarray, str]


def _reference_extents(reference: Optional[Asset]) -> Optional[list[float]]:
    if reference is None or reference.vertices is None:
        return None
    verts = np.asarray(reference.vertices, dtype=np.float64)
    if verts.size == 0:
        return None
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (hi - lo).tolist()


def _reference_bounds(reference: Optional[Asset]) -> tuple[Optional[list[float]], Optional[list[float]]]:
    if reference is None or reference.vertices is None:
        return None, None
    verts = np.asarray(reference.vertices, dtype=np.float64)
    if verts.size == 0:
        return None, None
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return lo.tolist(), (hi - lo).tolist()


def generate_from_image(
    image: ImageLike,
    *,
    track: TrackName = "track1",
    reference: Optional[Asset] = None,
    time_bins: int = 8,
    seed: int = 0,
    resolution: int = 8,
    shape: Literal["box", "sphere", "cylinder"] = "box",
    track1_checkpoint: Optional[str] = None,
    track2_checkpoint: Optional[str] = None,
    mesh_backend: str = "trimesh",
    **spike_kwargs: Any,
) -> Asset:
    """Build a candidate Asset from an image via Track 1 or Track 2."""
    spikes = image_to_spikes(image, time_bins=time_bins, seed=seed, **spike_kwargs)
    track_key = (
        track.replace("track1_scripted", "track1")
        .replace("track2_occupancy", "track2")
        .replace("track1_trained", "track1_trained")
        .replace("track2_trained", "track2_trained")
    )
    if track_key == "track1_sequence":
        from src.dst_snn.threedcg.train import load_track1_sequence_head

        from pathlib import Path as _P

        ckpt = track1_checkpoint or "artifacts/threedcg/checkpoints/track1_seq_quality.pt"
        if not _P(ckpt).is_file():
            ckpt = track1_checkpoint or "artifacts/threedcg/checkpoints/track1_seq.pt"
        seq_head = load_track1_sequence_head(ckpt)
        ops = seq_head.decode(spikes)
        return execute_ops_backend(ops, backend=mesh_backend)
    if track_key in {"track1", "track1_trained"}:
        head = None
        mode = "scripted"
        if track_key == "track1_trained" or track1_checkpoint:
            from src.dst_snn.threedcg.train import load_track1_head

            ckpt = track1_checkpoint or "artifacts/threedcg/checkpoints/track1_quality.pt"
            from pathlib import Path as _P

            if not _P(ckpt).is_file():
                ckpt = "artifacts/threedcg/checkpoints/track1.pt"
            head = load_track1_head(ckpt)
            mode = "torch"
        extents = _reference_extents(reference) if mode == "scripted" else None
        ops = decode_ops_from_spikes(
            spikes,
            mode=mode,
            shape=shape,
            extents_hint=extents,
            head=head,
        )
        return execute_ops_backend(ops, backend=mesh_backend)
    if track_key == "track2_sdf":
        from src.dst_snn.threedcg.train import load_track2_sdf_head

        ckpt = track2_checkpoint or "artifacts/threedcg/checkpoints/track2_sdf.pt"
        sdf_head = load_track2_sdf_head(ckpt)
        origin, extents = _reference_bounds(reference)
        return sdf_head.predict_mesh(spikes, origin=origin, extents=extents)
    if track_key in {"track2", "track2_trained"}:
        head = None
        if track_key == "track2_trained" or track2_checkpoint:
            from src.dst_snn.threedcg.train import load_track2_head
            from pathlib import Path as _P

            ckpt = track2_checkpoint or "artifacts/threedcg/checkpoints/track2_quality.pt"
            if not _P(ckpt).is_file():
                ckpt = "artifacts/threedcg/checkpoints/track2.pt"
            head = load_track2_head(ckpt)
            resolution = head.resolution
        origin, extents = _reference_bounds(reference)
        return track2_from_spikes(
            spikes,
            resolution=resolution,
            origin=origin,
            extents=extents,
            head=head,
        )
    raise ValueError(f"unknown track: {track!r}")


def run_pipeline_score(
    image: ImageLike,
    reference: Asset,
    *,
    track: TrackName = "track1",
    asset_id: str = "asset",
    time_bins: int = 8,
    seed: int = 0,
    resolution: int = 8,
    shape: Literal["box", "sphere", "cylinder"] = "box",
    track1_checkpoint: Optional[str] = None,
    track2_checkpoint: Optional[str] = None,
    mesh_backend: str = "trimesh",
) -> RunResult:
    """Generate candidate and score against reference (shared RunResult schema)."""
    start = time.perf_counter()
    candidate = generate_from_image(
        image,
        track=track,
        reference=reference,
        time_bins=time_bins,
        seed=seed,
        resolution=resolution,
        shape=shape,
        track1_checkpoint=track1_checkpoint,
        track2_checkpoint=track2_checkpoint,
        mesh_backend=mesh_backend,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    result = score_to_result(candidate, reference, asset_id=asset_id, build_latency_ms=latency_ms)
    result.model = f"snn-3dcg:{track}"
    result.meta["track"] = track
    result.meta["seed"] = seed
    result.meta["time_bins"] = time_bins
    result.meta["mesh_backend"] = mesh_backend
    if track1_checkpoint:
        result.meta["track1_checkpoint"] = track1_checkpoint
    if track2_checkpoint:
        result.meta["track2_checkpoint"] = track2_checkpoint
    spikes = image_to_spikes(image, time_bins=time_bins, seed=seed)
    result.metrics.spikes_per_inference = float(spikes.sum() / max(1, spikes.shape[0]))
    result.metrics.active_neuron_fraction = float((spikes.sum(axis=0) > 0).mean())
    result.metrics.energy_source = "spike proxy from image encoder (not full 3DCG model energy)"
    return result


def synthetic_box_image(
    *,
    size: int = 32,
    box_value: float = 0.9,
    bg: float = 0.1,
) -> np.ndarray:
    """Utility for tests: bright square on dark background (HxWx3)."""
    img = np.full((size, size, 3), bg, dtype=np.float64)
    lo, hi = size // 4, 3 * size // 4
    img[lo:hi, lo:hi, :] = box_value
    return img
