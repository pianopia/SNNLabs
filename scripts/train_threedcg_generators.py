#!/usr/bin/env python3
"""Train Track1 / Track2 3DCG generator heads (original Phase-1 path).

Examples:
  python scripts/train_threedcg_generators.py --track track1 --epochs 30
  python scripts/train_threedcg_generators.py --track both --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dst_snn.threedcg.train import (
    train_track1,
    train_track1_sequence,
    train_track2,
    train_track2_sdf,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised training for SNN 3DCG generators")
    p.add_argument(
        "--track",
        choices=["track1", "track2", "track1_seq", "track2_sdf", "both", "all"],
        default="both",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--resolution", type=int, default=6, help="Track2 occupancy/SDF grid size")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "threedcg" / "checkpoints",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.track in {"track1", "both", "all"}:
        r1 = train_track1(
            epochs=args.epochs,
            n_samples=args.n_samples,
            seed=args.seed,
            lr=args.lr,
            out_path=args.out_dir / "track1.pt",
        )
        print(
            f"track1 final_loss={r1.final_loss:.4f} "
            f"first={r1.extra.get('first_loss')} -> {r1.checkpoint}"
        )
    if args.track in {"track2", "both", "all"}:
        r2 = train_track2(
            epochs=args.epochs,
            n_samples=max(32, args.n_samples // 2),
            seed=args.seed,
            lr=args.lr,
            resolution=args.resolution,
            out_path=args.out_dir / "track2.pt",
        )
        print(
            f"track2 final_loss={r2.final_loss:.4f} "
            f"first={r2.extra.get('first_loss')} -> {r2.checkpoint}"
        )
    if args.track in {"track1_seq", "all"}:
        rs = train_track1_sequence(
            epochs=args.epochs,
            n_samples=args.n_samples,
            seed=args.seed,
            lr=args.lr,
            out_path=args.out_dir / "track1_seq.pt",
        )
        print(
            f"track1_seq final_loss={rs.final_loss:.4f} "
            f"first={rs.extra.get('first_loss')} -> {rs.checkpoint}"
        )
    if args.track in {"track2_sdf", "all"}:
        rd = train_track2_sdf(
            epochs=max(15, args.epochs // 2),
            n_samples=max(24, args.n_samples // 3),
            seed=args.seed,
            lr=args.lr,
            resolution=max(6, args.resolution),
            out_path=args.out_dir / "track2_sdf.pt",
        )
        print(
            f"track2_sdf final_loss={rd.final_loss:.4f} "
            f"first={rd.extra.get('first_loss')} -> {rd.checkpoint}"
        )


if __name__ == "__main__":
    main()
