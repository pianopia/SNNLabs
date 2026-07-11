#!/usr/bin/env python3
"""Offline smoke benchmark for the multimodal streaming SNN runtime contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dst_snn.foundation import (  # noqa: E402
    DEFAULT_MODALITY_SPECS,
    SignedEventEncoder,
    StreamingSpikingSSM,
    align_and_fuse_modalities,
    streaming_efficiency_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.features < 1 or args.state_size < 1:
        parser.error("steps must be >=2 and sizes must be positive")

    rng = np.random.default_rng(args.seed)
    time = np.linspace(0.0, 2.0 * np.pi, args.steps, dtype=np.float32)
    streams: dict[str, np.ndarray] = {}
    for offset, modality in enumerate(("text", "image", "audio", "sensor")):
        values = np.stack(
            [np.sin(time * (1.0 + index / 4.0) + offset) for index in range(args.features)],
            axis=1,
        )
        values += rng.normal(0.0, 0.01, values.shape).astype(np.float32)
        spec = DEFAULT_MODALITY_SPECS[modality]
        encoder = SignedEventEncoder(
            args.features,
            threshold=spec.threshold,
            max_event_level=spec.max_event_level,
        )
        streams[modality] = encoder.encode(values)

    fused, slices = align_and_fuse_modalities(streams)
    model = StreamingSpikingSSM(fused.shape[1], args.state_size, seed=args.seed)
    output = model.run(fused)
    report = streaming_efficiency_report(fused, state_size=args.state_size).to_dict()
    report.update(
        {
            "modalities": list(streams),
            "modality_slices": {
                name: [section.start, section.stop] for name, section in slices.items()
            },
            "output_nonzero_events": int(np.count_nonzero(output)),
            "runtime_state_bytes": model.state_bytes,
        }
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
