#!/usr/bin/env python3
"""Run a fixed set of DVS Gesture multi-seed configurations and compare them.

Writes under ``artifacts/benchmarks/dvs-multi-seed/``:
  - one directory per config with per-seed JSON + summary
  - ``comparison.json`` / ``comparison.md`` across configs
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT_ROOT = ROOT / "artifacts" / "benchmarks" / "dvs-multi-seed"

# Shared smoke protocol: stratified subsets of official test split.
COMMON = [
    "--benchmark",
    "dvs-gesture",
    "--root",
    "data/dvs-gesture",
    "--seeds",
    "0,1,2,13,42",
    "--smoke-from-test",
    "--epochs",
    "5",
    "--limit-train",
    "96",
    "--limit-test",
    "64",
    "--time-bins",
    "16",
    "--downsample",
    "8",
    "--batch-size",
    "8",
    "--device",
    "cpu",
]

CONFIGS = [
    {
        "tag": "A_direct_spike_count",
        "args": [
            "--no-chrono",
            "--hidden-features",
            "0",
            "--threshold",
            "0.5",
            "--readout",
            "spike_count",
        ],
    },
    {
        "tag": "B_hidden64_lowthr",
        "args": [
            "--no-chrono",
            "--hidden-features",
            "64",
            "--hidden-threshold",
            "0.1",
            "--hidden-output",
            "spikes",
            "--threshold",
            "0.1",
            "--readout",
            "spike_count",
        ],
    },
    {
        "tag": "C_temporal_hidden64",
        "args": [
            "--no-chrono",
            "--use-temporal-features",
            "--temporal-project-to",
            "128",
            "--hidden-features",
            "64",
            "--hidden-threshold",
            "0.1",
            "--hidden-output",
            "spikes",
            "--threshold",
            "0.1",
            "--readout",
            "spike_count",
        ],
    },
    {
        "tag": "D_chrono_temporal",
        "args": [
            "--use-chrono",
            "--chrono-hidden",
            "64",
            "--use-temporal-features",
            "--temporal-project-to",
            "128",
            "--hidden-features",
            "32",
            "--threshold",
            "0.3",
            "--readout",
            "max_membrane",
        ],
    },
]


def run_config(cfg: dict) -> dict:
    out_dir = OUT_ROOT / cfg["tag"]
    cmd = [
        PYTHON,
        str(ROOT / "benchmarks" / "neuromorphic" / "run_multi_seed.py"),
        *COMMON,
        "--out-dir",
        str(out_dir),
        "--tag",
        cfg["tag"],
        *cfg["args"],
    ]
    print("\n===", cfg["tag"], "===\n", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    summary["wall_time_s_total"] = time.perf_counter() - t0
    return summary


def write_comparison(summaries: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "dataset": "DVS Gesture (smoke-from-test stratified subsets)",
            "seeds": [0, 1, 2, 13, 42],
            "limit_train": 96,
            "limit_test": 64,
            "epochs": 5,
            "time_bins": 16,
            "downsample": 8,
        },
        "configs": summaries,
    }
    (OUT_ROOT / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# DVS Gesture multi-seed comparison",
        "",
        "Protocol: smoke-from-test, stratified 96/64, 5 epochs, 5 seeds (0,1,2,13,42), downsample=8, time_bins=16.",
        "",
        "| config | quality mean±std | min | max | majority | above maj | margin | decision_lat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        maj = s.get("majority_mean")
        maj_s = f"{maj:.4f}" if maj is not None else ""
        margin = s.get("margin_over_majority_mean")
        margin_s = f"{margin:+.4f}" if margin is not None else ""
        above = s.get("seeds_above_majority")
        above_s = f"{above}/{s['n_seeds']}" if above is not None else ""
        dec = s.get("decision_latency_fraction_mean")
        dec_s = f"{dec:.4f}" if dec is not None else ""
        tag = s.get("config", {}).get("tag", "?")
        lines.append(
            f"| {tag} | {s['quality_mean']:.4f}±{s['quality_std']:.4f} | "
            f"{s['quality_min']:.4f} | {s['quality_max']:.4f} | {maj_s} | {above_s} | {margin_s} | {dec_s} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- **majority**: constant class predictor on the evaluation split.",
            "- **above maj**: how many seeds beat that baseline.",
            "- **margin**: mean(quality) − mean(majority). Positive ⇒ above chance-of-majority on average.",
            "- Smoke subsets are small; treat as directional evidence, not a published SOTA claim.",
            "",
        ]
    )
    (OUT_ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cfg in CONFIGS:
        summaries.append(run_config(cfg))
    write_comparison(summaries)
    print(f"\nWrote {OUT_ROOT / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
