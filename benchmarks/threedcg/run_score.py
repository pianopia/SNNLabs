#!/usr/bin/env python3
"""Score a candidate glTF/GLB against a reference using the 3DCG scorer.

Usage:
    python benchmarks/threedcg/run_score.py \\
        --reference data/threedcg/unit-box/reference.glb \\
        --candidate data/threedcg/unit-box/reference.glb \\
        --asset-id unit-box
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.threedcg.asset import load_asset
from benchmarks.threedcg.baseline import convex_hull_candidate
from benchmarks.threedcg.scorer import score_to_result
from src.dst_snn.eval import run_benchmarks
from src.dst_snn.eval.result import RunResult


class ThreedcgScoreRunner:
    name = "eden14-image-to-3d"

    def __init__(
        self,
        reference: str,
        candidate: str | None = None,
        *,
        asset_id: str = "asset",
        use_convex_hull: bool = False,
    ) -> None:
        self.reference = reference
        self.candidate = candidate
        self.asset_id = asset_id
        self.use_convex_hull = use_convex_hull
        self._result: RunResult | None = None

    def prepare(self) -> None:
        ref = load_asset(self.reference)
        start = time.perf_counter()
        if self.use_convex_hull or not self.candidate:
            cand = convex_hull_candidate(ref)
        else:
            cand = load_asset(self.candidate)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._result = score_to_result(cand, ref, asset_id=self.asset_id, build_latency_ms=latency_ms)

    def run(self) -> RunResult:
        assert self._result is not None
        return self._result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--asset-id", default="asset")
    parser.add_argument("--convex-hull", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks/threedcg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ThreedcgScoreRunner(
        args.reference,
        args.candidate,
        asset_id=args.asset_id,
        use_convex_hull=args.convex_hull,
    )
    results = run_benchmarks([runner], args.out_dir)
    print(results[0].to_json())


if __name__ == "__main__":
    main()
