#!/usr/bin/env python3
"""Full-train (or large-smoke) pilot: Conv-PLIF, SEW-PLIF, and matched Frame-CNN.

Requires ``ibmGestureTrain.tar.gz`` under data/dvs-gesture/DVSGesture/ for
full-train mode (Zenodo 8060604, ~2.28GB). Tonic extracts on first load.
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
OUT = ROOT / "artifacts" / "benchmarks" / "dvs-fulltrain-sew"
TRAIN_TAR = ROOT / "data" / "dvs-gesture" / "DVSGesture" / "ibmGestureTrain.tar.gz"
TRAIN_DIR = ROOT / "data" / "dvs-gesture" / "DVSGesture" / "ibmGestureTrain"
EXPECTED_MIN_BYTES = 2_200_000_000


def train_ready() -> bool:
    if TRAIN_DIR.is_dir() and any(TRAIN_DIR.iterdir()):
        return True
    return TRAIN_TAR.is_file() and TRAIN_TAR.stat().st_size >= EXPECTED_MIN_BYTES


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_backbone(backbone: str, *, has_train: bool, seeds: list[int], epochs: int) -> list[dict]:
    rows = []
    for seed in seeds:
        out_dir = OUT / f"{backbone}_seed{seed}"
        cmd = [
            PYTHON,
            str(ROOT / "benchmarks" / "neuromorphic" / "run_dvs_gesture.py"),
            "--root",
            "data/dvs-gesture",
            "--backbone",
            backbone,
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
            "16" if backbone == "conv-plif" else "8",
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--lr",
            "1e-3",
            "--sew-width",
            "32",
            "--sew-blocks",
            "2",
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
        row = {
            "backbone": backbone,
            "seed": seed,
            "snn_quality": result["metrics"]["quality"],
            "cnn_quality": baseline.get("quality"),
            "majority": (baseline.get("extra") or {}).get("majority_class_accuracy"),
            "snn_energy_pj": result["metrics"]["energy_pj"],
            "cnn_energy_pj": baseline.get("energy_pj"),
            "decision_latency_fraction": result["metrics"]["extra"].get("decision_latency_fraction"),
            "param_count": result["metrics"]["param_count"],
            "model": result.get("model"),
        }
        rows.append(row)
        print(
            f"{backbone} seed={seed} snn={row['snn_quality']:.4f} "
            f"cnn={row['cnn_quality']} maj={row['majority']}",
            flush=True,
        )
    return rows


def summarize(rows: list[dict], backbone: str) -> dict:
    subset = [r for r in rows if r["backbone"] == backbone]
    snn = [r["snn_quality"] for r in subset]
    cnn = [float(r["cnn_quality"]) for r in subset if r["cnn_quality"] is not None]
    maj = [float(r["majority"]) for r in subset if r["majority"] is not None]
    return {
        "backbone": backbone,
        "n": len(subset),
        "snn_mean": statistics.fmean(snn) if snn else 0.0,
        "snn_std": statistics.pstdev(snn) if len(snn) > 1 else 0.0,
        "snn_min": min(snn) if snn else 0.0,
        "snn_max": max(snn) if snn else 0.0,
        "cnn_mean": statistics.fmean(cnn) if cnn else None,
        "cnn_std": statistics.pstdev(cnn) if len(cnn) > 1 else 0.0,
        "majority_mean": statistics.fmean(maj) if maj else None,
        "snn_above_majority": sum(
            1 for r in subset if r["majority"] is not None and r["snn_quality"] > float(r["majority"])
        ),
        "params_mean": statistics.fmean([r["param_count"] for r in subset]) if subset else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    has_train = train_ready()
    mode = "full-train" if has_train else "smoke-large"
    seeds = [0, 1, 2]
    epochs = 12 if has_train else 12
    print(
        f"mode={mode} train_ready={has_train} tar_size="
        f"{TRAIN_TAR.stat().st_size if TRAIN_TAR.exists() else 0}",
        flush=True,
    )

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    for backbone in ("conv-plif", "sew-plif"):
        all_rows.extend(run_backbone(backbone, has_train=has_train, seeds=seeds, epochs=epochs))

    summaries = [summarize(all_rows, b) for b in ("conv-plif", "sew-plif")]
    report = {
        "protocol_mode": mode,
        "train_ready": has_train,
        "epochs": epochs,
        "seeds": seeds,
        "wall_time_s": time.perf_counter() - t0,
        "rows": all_rows,
        "summaries": summaries,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# DVS full-train / SEW pilot",
        "",
        f"- mode: **{mode}**",
        f"- train ready: `{has_train}`",
        f"- seeds: `{seeds}`, epochs: `{epochs}`",
        f"- wall time: `{report['wall_time_s']:.1f}s`",
        "",
        "| backbone | SNN mean±std | range | CNN mean | above maj | params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        cnn = f"{s['cnn_mean']:.4f}" if s["cnn_mean"] is not None else "n/a"
        lines.append(
            f"| {s['backbone']} | {s['snn_mean']:.4f}±{s['snn_std']:.4f} | "
            f"[{s['snn_min']:.4f},{s['snn_max']:.4f}] | {cnn} | "
            f"{s['snn_above_majority']}/{s['n']} | {int(s['params_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed",
            "",
            "| backbone | seed | SNN | CNN | majority | decision_lat |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in all_rows:
        lines.append(
            f"| {r['backbone']} | {r['seed']} | {r['snn_quality']:.4f} | "
            f"{float(r['cnn_quality'] or 0):.4f} | {float(r['majority'] or 0):.4f} | "
            f"{float(r['decision_latency_fraction'] or 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `sew-plif` = stem + SEW residual stages (Fang et al. SEW-ADD).",
            "- Full train needs complete `ibmGestureTrain.tar.gz` (~2.28GB, Zenodo 8060604).",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
