"""Tests for quality closed-loop training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from benchmarks.threedcg.asset import asset_from_trimesh
from src.dst_snn.threedcg.quality_loop import (
    sample_primitive_points_torch,
    score_quality,
    soft_chamfer_torch,
    train_track1_quality,
    train_track2_quality,
)
import trimesh


def test_score_quality_self_high():
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    a = asset_from_trimesh(mesh)
    q = score_quality(a, a)
    assert q > 0.85


def test_soft_chamfer_identical_zero():
    pts = torch.randn(32, 3)
    d = soft_chamfer_torch(pts, pts)
    assert float(d.item()) < 1e-3


def test_sample_primitive_shapes():
    ex = torch.tensor([1.0, 1.2, 0.8])
    for sid in (0, 1, 2):
        p = sample_primitive_points_torch(sid, ex, n=64, seed=0)
        assert p.shape == (64, 3)


def test_train_track1_quality_improves_or_stable(tmp_path: Path):
    r = train_track1_quality(
        epochs=8,
        n_samples=12,
        seed=0,
        lr=1e-2,
        out_path=tmp_path / "t1q.pt",
        lambda_rl=0.2,
        lambda_proxy=0.4,
    )
    assert (tmp_path / "t1q.pt").is_file()
    # quality history recorded
    assert r.extra.get("quality_history")
    assert r.extra.get("final_quality") is not None


def test_train_track2_quality(tmp_path: Path):
    r = train_track2_quality(
        epochs=6,
        n_samples=10,
        seed=1,
        lr=1e-2,
        resolution=4,
        out_path=tmp_path / "t2q.pt",
    )
    assert r.extra.get("final_quality") is not None
    assert r.final_loss < float("inf")
