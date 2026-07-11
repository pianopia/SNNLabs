#!/usr/bin/env python3
"""Run Phase 2 next-token and image-text retrieval smoke benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dst_snn.foundation.phase2_benchmarks import (  # noqa: E402
    run_image_text_retrieval_benchmark,
    run_text_next_token_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = run_text_next_token_benchmark(
        seed=args.seed,
        samples=48 if args.quick else 96,
        teacher_epochs=12 if args.quick else 35,
        student_epochs=16 if args.quick else 45,
    )
    retrieval = run_image_text_retrieval_benchmark(
        seed=args.seed,
        pairs=8 if args.quick else 16,
        epochs=20 if args.quick else 60,
    )
    payload = {
        "scope": "synthetic_phase2_smoke_not_capability_claim",
        "results": [text.to_dict(), retrieval.to_dict()],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
