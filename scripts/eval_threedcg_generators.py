#!/usr/bin/env python3
"""Train (optional) and evaluate 3DCG generator tracks with the shared scorer.

Produces quality / loss-oriented JSON+Markdown under artifacts/threedcg/eval/.
Does not require Blender or network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.threedcg.asset import asset_from_trimesh, load_asset
from benchmarks.threedcg.scorer import score_to_result
from src.dst_snn.threedcg.dataset import make_batch
from src.dst_snn.threedcg.pipeline import generate_from_image
from src.dst_snn.threedcg.train import (
    train_track1,
    train_track1_sequence,
    train_track2,
    train_track2_sdf,
)


def _score_pair(image, reference, *, track: str, **kwargs) -> dict:
    t0 = time.perf_counter()
    cand = generate_from_image(image, track=track, reference=reference, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = score_to_result(cand, reference, asset_id="eval", build_latency_ms=latency_ms)
    geo = (result.metrics.extra.get("scores") or {}).get("geometry") or {}
    return {
        "track": track,
        "quality": float(result.metrics.quality),
        "chamfer": geo.get("chamfer"),
        "volume_iou": geo.get("volume_iou"),
        "latency_ms": latency_ms,
        "kwargs": {k: str(v) for k, v in kwargs.items() if v is not None},
    }


def evaluate_on_batch(
    samples,
    *,
    track: str,
    **kwargs,
) -> dict:
    rows = []
    for s in samples:
        rows.append(
            _score_pair(
                s.image,
                s.asset,
                track=track,
                time_bins=8,
                seed=0,
                **kwargs,
            )
        )
    qualities = [r["quality"] for r in rows]
    return {
        "track": track,
        "n": len(rows),
        "quality_mean": statistics.fmean(qualities) if qualities else 0.0,
        "quality_std": statistics.pstdev(qualities) if len(qualities) > 1 else 0.0,
        "quality_min": min(qualities) if qualities else 0.0,
        "quality_max": max(qualities) if qualities else 0.0,
        "latency_ms_mean": statistics.fmean([r["latency_ms"] for r in rows]) if rows else 0.0,
        "rows": rows,
        "kwargs": {k: str(v) for k, v in kwargs.items() if v is not None},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "threedcg" / "eval")
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "artifacts" / "threedcg" / "checkpoints")
    p.add_argument("--n-eval", type=int, default=12)
    p.add_argument("--n-train", type=int, default=48)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--unit-box", action="store_true", help="Also score against data/threedcg/unit-box")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_meta = {}
    if not args.skip_train:
        print("training heads...", flush=True)
        train_meta["track1"] = train_track1(
            epochs=args.epochs,
            n_samples=args.n_train,
            seed=args.seed,
            out_path=args.ckpt_dir / "track1.pt",
            image_size=24,
            time_bins=6,
        ).__dict__
        train_meta["track2"] = train_track2(
            epochs=args.epochs,
            n_samples=max(24, args.n_train // 2),
            seed=args.seed,
            resolution=6,
            out_path=args.ckpt_dir / "track2.pt",
            image_size=24,
            time_bins=6,
        ).__dict__
        train_meta["track1_seq"] = train_track1_sequence(
            epochs=args.epochs,
            n_samples=args.n_train,
            seed=args.seed,
            out_path=args.ckpt_dir / "track1_seq.pt",
            image_size=24,
            time_bins=6,
        ).__dict__
        train_meta["track2_sdf"] = train_track2_sdf(
            epochs=max(12, args.epochs // 2),
            n_samples=max(16, args.n_train // 3),
            seed=args.seed,
            resolution=8,
            out_path=args.ckpt_dir / "track2_sdf.pt",
            image_size=24,
            time_bins=6,
        ).__dict__
        # JSON-serialize train results
        for k, v in list(train_meta.items()):
            train_meta[k] = {
                "track": v.get("track"),
                "final_loss": v.get("final_loss"),
                "first_loss": (v.get("extra") or {}).get("first_loss"),
                "checkpoint": v.get("checkpoint"),
                "epochs": v.get("epochs"),
            }

    print("evaluating on held-out synthetic batch...", flush=True)
    # Different seed than train
    eval_samples = make_batch(args.n_eval, seed=args.seed + 999, time_bins=6, image_size=24, resolution=6)

    summaries = []
    # Baselines without training signal
    summaries.append(evaluate_on_batch(eval_samples, track="track1", mesh_backend="trimesh"))
    summaries.append(evaluate_on_batch(eval_samples, track="track2", resolution=6))

    t1 = args.ckpt_dir / "track1.pt"
    t2 = args.ckpt_dir / "track2.pt"
    t1s = args.ckpt_dir / "track1_seq.pt"
    t2s = args.ckpt_dir / "track2_sdf.pt"
    if t1.is_file():
        summaries.append(
            evaluate_on_batch(
                eval_samples,
                track="track1_trained",
                track1_checkpoint=str(t1),
                mesh_backend="trimesh",
            )
        )
    if t2.is_file():
        summaries.append(
            evaluate_on_batch(
                eval_samples,
                track="track2_trained",
                track2_checkpoint=str(t2),
            )
        )
    if t1s.is_file():
        summaries.append(
            evaluate_on_batch(
                eval_samples,
                track="track1_sequence",
                track1_checkpoint=str(t1s),
                mesh_backend="trimesh",
            )
        )
    if t2s.is_file():
        summaries.append(
            evaluate_on_batch(
                eval_samples,
                track="track2_sdf",
                track2_checkpoint=str(t2s),
            )
        )

    unit_box = None
    unit_path = ROOT / "data" / "threedcg" / "unit-box" / "reference.glb"
    if args.unit_box and unit_path.is_file():
        import numpy as np
        from src.dst_snn.threedcg.pipeline import synthetic_box_image

        ref = load_asset(str(unit_path))
        img = synthetic_box_image(size=32)
        unit_box = {}
        for track, kw in [
            ("track1", {}),
            ("track1_trained", {"track1_checkpoint": str(t1)} if t1.is_file() else None),
            ("track1_sequence", {"track1_checkpoint": str(t1s)} if t1s.is_file() else None),
            ("track2", {"resolution": 8}),
            ("track2_trained", {"track2_checkpoint": str(t2)} if t2.is_file() else None),
            ("track2_sdf", {"track2_checkpoint": str(t2s)} if t2s.is_file() else None),
        ]:
            if kw is None:
                continue
            unit_box[track] = _score_pair(img, ref, track=track, **kw)

    report = {
        "train": train_meta,
        "eval_summaries": [
            {k: v for k, v in s.items() if k != "rows"} for s in summaries
        ],
        "unit_box": unit_box,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 3DCG generator evaluation",
        "",
        "## Train losses",
        "",
    ]
    if train_meta:
        lines.append("| track | first_loss | final_loss |")
        lines.append("|---|---:|---:|")
        for name, meta in train_meta.items():
            lines.append(
                f"| {name} | {meta.get('first_loss')} | {meta.get('final_loss'):.4f} |"
                if meta.get("final_loss") is not None
                else f"| {name} | {meta.get('first_loss')} | n/a |"
            )
    else:
        lines.append("_skipped train_")

    lines.extend(["", "## Scorer quality on held-out synthetic (higher is better)", ""])
    lines.append("| track | n | quality mean±std | min | max | latency ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s['track']} | {s['n']} | "
            f"{s['quality_mean']:.4f}±{s['quality_std']:.4f} | "
            f"{s['quality_min']:.4f} | {s['quality_max']:.4f} | "
            f"{s['latency_ms_mean']:.1f} |"
        )
    if unit_box:
        lines.extend(["", "## unit-box reference", ""])
        lines.append("| track | quality | chamfer | volume_iou |")
        lines.append("|---|---:|---:|---:|")
        for track, row in unit_box.items():
            lines.append(
                f"| {track} | {row['quality']:.4f} | {row.get('chamfer')} | {row.get('volume_iou')} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Quality is scorer composite (not training loss).",
            "- Synthetic eval uses a different seed than training.",
            "- track1_sequence includes UV/rig/material ops which scorer may partially score.",
            "",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
