#!/usr/bin/env python3
"""Run SNN image→3D Track 1/2 generator and score against a reference.

Examples:
  python benchmarks/threedcg/run_generate.py \\
    --reference data/threedcg/unit-box/reference.glb \\
    --image data/threedcg/unit-box/input.png \\
    --track track1 --asset-id unit-box
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.threedcg.asset import load_asset
from src.dst_snn.eval import run_benchmarks
from src.dst_snn.eval.result import RunResult
from src.dst_snn.threedcg.pipeline import run_pipeline_score, synthetic_box_image


class GenerateScoreRunner:
    name = "eden14-image-to-3d-generate"

    def __init__(
        self,
        reference: str,
        image: str | None,
        *,
        track: str = "track1",
        asset_id: str = "asset",
        seed: int = 0,
        resolution: int = 8,
        synthetic: bool = False,
        mesh_backend: str = "trimesh",
        track1_checkpoint: str | None = None,
        track2_checkpoint: str | None = None,
    ) -> None:
        self.reference = reference
        self.image = image
        self.track = track
        self.asset_id = asset_id
        self.seed = seed
        self.resolution = resolution
        self.synthetic = synthetic
        self.mesh_backend = mesh_backend
        self.track1_checkpoint = track1_checkpoint
        self.track2_checkpoint = track2_checkpoint
        self._result: RunResult | None = None

    def prepare(self) -> None:
        ref = load_asset(self.reference)
        if self.synthetic or not self.image:
            image = synthetic_box_image(size=32)
        else:
            image = self.image
        self._result = run_pipeline_score(
            image,
            ref,
            track=self.track,  # type: ignore[arg-type]
            asset_id=self.asset_id,
            seed=self.seed,
            resolution=self.resolution,
            mesh_backend=self.mesh_backend,
            track1_checkpoint=self.track1_checkpoint,
            track2_checkpoint=self.track2_checkpoint,
        )

    def run(self) -> RunResult:
        assert self._result is not None
        return self._result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SNN image→3D Track1/Track2 generator + scorer")
    p.add_argument("--reference", required=True)
    p.add_argument("--image", default=None, help="Input PNG; omit with --synthetic")
    p.add_argument("--synthetic", action="store_true", help="Use built-in synthetic box image")
    p.add_argument(
        "--track",
        choices=[
            "track1",
            "track2",
            "track1_scripted",
            "track2_occupancy",
            "track1_trained",
            "track2_trained",
            "track1_sequence",
            "track2_sdf",
        ],
        default="track1",
    )
    p.add_argument("--asset-id", default="asset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=8, help="Track2 occupancy resolution")
    p.add_argument(
        "--backend",
        choices=["trimesh", "bpy", "auto", "mock"],
        default="trimesh",
        help="Track1 mesh executor: trimesh (CI default), bpy (Blender), auto, mock",
    )
    p.add_argument("--track1-checkpoint", default=None)
    p.add_argument("--track2-checkpoint", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks/threedcg-generate"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Sensible default checkpoints for trained/sequence/sdf tracks
    t1 = args.track1_checkpoint
    t2 = args.track2_checkpoint
    ckpt = ROOT / "artifacts" / "threedcg" / "checkpoints"
    if t1 is None and args.track == "track1_trained" and (ckpt / "track1.pt").is_file():
        t1 = str(ckpt / "track1.pt")
    if t1 is None and args.track == "track1_sequence" and (ckpt / "track1_seq.pt").is_file():
        t1 = str(ckpt / "track1_seq.pt")
    if t2 is None and args.track == "track2_trained" and (ckpt / "track2.pt").is_file():
        t2 = str(ckpt / "track2.pt")
    if t2 is None and args.track == "track2_sdf" and (ckpt / "track2_sdf.pt").is_file():
        t2 = str(ckpt / "track2_sdf.pt")

    runner = GenerateScoreRunner(
        args.reference,
        args.image,
        track=args.track,
        asset_id=args.asset_id,
        seed=args.seed,
        resolution=args.resolution,
        synthetic=args.synthetic,
        mesh_backend=args.backend,
        track1_checkpoint=t1,
        track2_checkpoint=t2,
    )
    results = run_benchmarks([runner], args.out_dir)
    print(results[0].to_json())


if __name__ == "__main__":
    main()
