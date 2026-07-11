"""Supervised training smoke: loss decreases; checkpoints load."""

from __future__ import annotations

from pathlib import Path

from src.dst_snn.threedcg.dataset import make_batch, make_sample
from src.dst_snn.threedcg.pipeline import generate_from_image, synthetic_box_image
from src.dst_snn.threedcg.train import load_track1_head, train_track1, train_track2


def test_dataset_sample_shapes():
    s = make_sample(shape="box", extents=(1.0, 1.2, 0.8), seed=0, resolution=4)
    assert s.spikes.ndim == 2
    assert s.occupancy.shape == (4, 4, 4)
    assert s.shape_id == 0
    batch = make_batch(6, seed=1, resolution=4)
    assert len(batch) == 6


def test_train_track1_loss_drops(tmp_path: Path):
    out = tmp_path / "track1.pt"
    result = train_track1(
        epochs=15,
        n_samples=36,
        seed=0,
        lr=2e-2,
        out_path=out,
        image_size=16,
        time_bins=4,
    )
    assert out.is_file()
    assert result.final_loss < result.extra["first_loss"]
    head = load_track1_head(out)
    spikes = make_sample(shape="sphere", extents=(1, 1, 1), seed=3, time_bins=4, image_size=16).spikes
    ops = head.decode(spikes)
    assert ops[0].name in {"ADD_BOX", "ADD_SPHERE", "ADD_CYLINDER"}


def test_train_track2_and_pipeline(tmp_path: Path):
    out = tmp_path / "track2.pt"
    result = train_track2(
        epochs=12,
        n_samples=24,
        seed=1,
        lr=2e-2,
        resolution=4,
        out_path=out,
        image_size=16,
        time_bins=4,
    )
    assert result.final_loss <= result.extra["first_loss"] * 1.05  # non-increasing-ish
    img = synthetic_box_image(size=16)
    ref = make_sample(shape="box", extents=(1, 1, 1), seed=0).asset
    asset = generate_from_image(
        img,
        track="track2_trained",
        reference=ref,
        track2_checkpoint=str(out),
        seed=0,
        time_bins=4,
    )
    assert asset.vertices is not None
    assert len(asset.vertices) > 0
