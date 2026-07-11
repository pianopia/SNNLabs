#!/usr/bin/env python3
"""Quality closed-loop fine-tuning for 3DCG generators.

Uses scorer quality (+ soft Chamfer / REINFORCE) so models optimize what we measure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dst_snn.threedcg.quality_loop import (
    train_track1_quality,
    train_track1_sequence_quality,
    train_track2_quality,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--track",
        choices=["track1", "track1_seq", "track2", "all"],
        default="all",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--n-samples", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=ROOT / "artifacts" / "threedcg" / "checkpoints",
    )
    p.add_argument("--init-from-supervised", action="store_true", default=True)
    p.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT / "data" / "threedcg",
        help="External/local mesh corpus root (data/threedcg layout)",
    )
    p.add_argument(
        "--mix-synthetic",
        type=float,
        default=0.25,
        help="Fraction of batch from synthetic families (rest from corpus)",
    )
    p.add_argument(
        "--no-corpus",
        action="store_true",
        help="Ignore corpus; train on synthetic families only",
    )
    args = p.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    corpus_root = None if args.no_corpus else args.corpus_root
    if corpus_root is not None and corpus_root.is_dir():
        n_corpus = sum(1 for p in corpus_root.iterdir() if p.is_dir() and (p / "reference.glb").is_file())
        print(f"corpus: {corpus_root} ({n_corpus} assets) mix_synthetic={args.mix_synthetic}")
    else:
        print("corpus: none (synthetic-only)")
        corpus_root = None

    def init(name: str) -> Path | None:
        if not args.init_from_supervised:
            return None
        path = args.ckpt_dir / name
        return path if path.is_file() else None

    if args.track in {"track1", "all"}:
        r = train_track1_quality(
            epochs=args.epochs,
            n_samples=args.n_samples,
            seed=args.seed,
            lr=args.lr,
            out_path=args.ckpt_dir / "track1_quality.pt",
            init_checkpoint=init("track1.pt"),
            corpus_root=corpus_root,
            mix_synthetic=args.mix_synthetic,
        )
        print(
            f"track1_quality loss {r.extra.get('first_loss')} → {r.final_loss:.4f} | "
            f"quality {r.extra.get('first_quality')} → {r.extra.get('final_quality')}"
        )
    if args.track in {"track1_seq", "all"}:
        r = train_track1_sequence_quality(
            epochs=args.epochs,
            n_samples=max(16, args.n_samples // 2),
            seed=args.seed,
            lr=args.lr,
            out_path=args.ckpt_dir / "track1_seq_quality.pt",
            init_checkpoint=init("track1_seq.pt"),
            corpus_root=corpus_root,
            mix_synthetic=args.mix_synthetic,
        )
        print(
            f"track1_seq_quality loss {r.extra.get('first_loss')} → {r.final_loss:.4f} | "
            f"quality {r.extra.get('first_quality')} → {r.extra.get('final_quality')}"
        )
    if args.track in {"track2", "all"}:
        r = train_track2_quality(
            epochs=args.epochs,
            n_samples=max(16, args.n_samples // 2),
            seed=args.seed,
            lr=args.lr,
            out_path=args.ckpt_dir / "track2_quality.pt",
            init_checkpoint=init("track2.pt"),
            corpus_root=corpus_root,
            mix_synthetic=args.mix_synthetic,
        )
        print(
            f"track2_quality loss {r.extra.get('first_loss')} → {r.final_loss:.4f} | "
            f"quality {r.extra.get('first_quality')} → {r.extra.get('final_quality')}"
        )


if __name__ == "__main__":
    main()
