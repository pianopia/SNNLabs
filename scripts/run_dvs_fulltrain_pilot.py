#!/usr/bin/env python3
"""DVS Gesture pilot: Conv-PLIF vs matched Frame-CNN.

Uses official train/test when ``ibmGestureTrain`` is present (Zenodo 8060604
mirror recommended). Otherwise falls back to a large stratified smoke-from-test
split of the official test set.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT = ROOT / "artifacts" / "benchmarks" / "dvs-fulltrain-pilot"
TRAIN_TAR = ROOT / "data" / "dvs-gesture" / "DVSGesture" / "ibmGestureTrain.tar.gz"
TRAIN_DIR = ROOT / "data" / "dvs-gesture" / "DVSGesture" / "ibmGestureTrain"


def train_available() -> bool:
    if TRAIN_DIR.is_dir() and any(TRAIN_DIR.iterdir()):
        return True
    return TRAIN_TAR.is_file() and TRAIN_TAR.stat().st_size > 100_000_000


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    has_train = train_available()
    mode = "full-train" if has_train else "smoke-large"
    seeds = [0, 1, 2]
    epochs = 10 if has_train else 12
    print(f"protocol_mode={mode} train_available={has_train}", flush=True)

    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        out_dir = OUT / f"seed{seed}"
        cmd = [
            PYTHON,
            str(ROOT / "benchmarks" / "neuromorphic" / "run_dvs_gesture.py"),
            "--root",
            "data/dvs-gesture",
            "--backbone",
            "conv-plif",
            "--with-ann-baseline",
            "--threshold",
            "1.0",
            "--readout",
            "spike_count",
            "--time-bins",
            "16",
            "--downsample",
            "8",
            "--batch-size",
            "16",
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--lr",
            "1e-3",
            "--out-dir",
            str(out_dir),
        ]
        if has_train:
            cmd += ["--limit-train", "0", "--limit-test", "0"]
        else:
            cmd += [
                "--smoke-from-test",
                "--limit-train",
                "200",
                "--limit-test",
                "64",
            ]
        run(cmd)
        result = json.loads((out_dir / "dvs-gesture.json").read_text(encoding="utf-8"))
        baseline = result.get("baseline") or {}
        rows.append(
            {
                "seed": seed,
                "snn_quality": result["metrics"]["quality"],
                "cnn_quality": baseline.get("quality"),
                "majority": (baseline.get("extra") or {}).get("majority_class_accuracy"),
                "snn_energy_pj": result["metrics"]["energy_pj"],
                "cnn_energy_pj": baseline.get("energy_pj"),
                "decision_latency_fraction": result["metrics"]["extra"].get(
                    "decision_latency_fraction"
                ),
                "param_count": result["metrics"]["param_count"],
                "cnn_param_count": baseline.get("param_count"),
                "energy_ratio_cnn_over_snn": (baseline.get("extra") or {}).get(
                    "energy_ratio_cnn_over_snn"
                ),
            }
        )
        print(
            f"seed={seed} snn={rows[-1]['snn_quality']:.4f} "
            f"cnn={rows[-1]['cnn_quality']:.4f} maj={rows[-1]['majority']}",
            flush=True,
        )

    snn_q = [r["snn_quality"] for r in rows]
    cnn_q = [float(r["cnn_quality"]) for r in rows if r["cnn_quality"] is not None]
    maj = [float(r["majority"]) for r in rows if r["majority"] is not None]
    report = {
        "protocol_mode": mode,
        "train_available": has_train,
        "epochs": epochs,
        "seeds": seeds,
        "wall_time_s": time.perf_counter() - t0,
        "rows": rows,
        "summary": {
            "snn_mean": statistics.fmean(snn_q),
            "snn_std": statistics.pstdev(snn_q) if len(snn_q) > 1 else 0.0,
            "cnn_mean": statistics.fmean(cnn_q) if cnn_q else None,
            "cnn_std": statistics.pstdev(cnn_q) if len(cnn_q) > 1 else 0.0,
            "majority_mean": statistics.fmean(maj) if maj else None,
            "snn_above_majority": sum(
                1 for r in rows if r["majority"] is not None and r["snn_quality"] > float(r["majority"])
            ),
            "cnn_above_majority": sum(
                1
                for r in rows
                if r["majority"] is not None
                and r["cnn_quality"] is not None
                and float(r["cnn_quality"]) > float(r["majority"])
            ),
            "gap_cnn_minus_snn": (
                statistics.fmean(cnn_q) - statistics.fmean(snn_q) if cnn_q else None
            ),
        },
    }
    (OUT / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    lines = [
        "# DVS Conv-PLIF vs Frame-CNN pilot",
        "",
        f"- mode: **{mode}** (train archive available: `{has_train}`)",
        f"- seeds: `{seeds}`, epochs: `{epochs}`",
        f"- wall time: `{report['wall_time_s']:.1f}s`",
        "",
        f"- SNN mean±std: **{s['snn_mean']:.4f} ± {s['snn_std']:.4f}** "
        f"(above majority {s['snn_above_majority']}/{len(seeds)})",
        f"- CNN mean±std: **{s['cnn_mean']:.4f} ± {s['cnn_std']:.4f}** "
        f"(above majority {s['cnn_above_majority']}/{len(seeds)})"
        if s["cnn_mean"] is not None
        else "- CNN: n/a",
        f"- gap (CNN − SNN): **{s['gap_cnn_minus_snn']:+.4f}**"
        if s["gap_cnn_minus_snn"] is not None
        else "",
        "",
        "| seed | SNN | CNN | majority | SNN pJ | CNN pJ | CNN/SNN energy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio = r.get("energy_ratio_cnn_over_snn")
        ratio_s = f"{float(ratio):.1f}" if ratio is not None else ""
        lines.append(
            f"| {r['seed']} | {r['snn_quality']:.4f} | {float(r['cnn_quality'] or 0):.4f} | "
            f"{float(r['majority'] or 0):.4f} | {r['snn_energy_pj']:.1f} | "
            f"{float(r['cnn_energy_pj'] or 0):.1f} | {ratio_s} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Frame-CNN matches Conv-PLIF channel widths (32-64-64) with ReLU instead of PLIF.",
            "- Full train requires `data/dvs-gesture/DVSGesture/ibmGestureTrain.tar.gz` "
            "(Zenodo record `8060604`, ~2.3GB).",
            "- Smoke-large uses stratified subsets of the official test split only.",
            "",
        ]
    )
    (OUT / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
