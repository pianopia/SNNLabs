#!/usr/bin/env python3
"""Freeze hires-ds4 multi-seed full-train numbers (Phase 0 remainder).

Uses recipe ``hires-ds4`` (downsample=4 ≈32×32, cosine LR, 12 epochs).
Default: conv-plif + matched Frame-CNN, seeds 0–2. SEW optional via --sew.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_OUT = ROOT / "artifacts" / "benchmarks" / "dvs-hires-fulltrain"
TRAIN_DIR = ROOT / "data" / "dvs-gesture" / "DVSGesture" / "ibmGestureTrain"
TRAIN_TAR = ROOT / "data" / "dvs-gesture" / "DVSGesture" / "ibmGestureTrain.tar.gz"


def train_ready() -> bool:
    if TRAIN_DIR.is_dir() and any(TRAIN_DIR.iterdir()):
        return True
    return TRAIN_TAR.is_file() and TRAIN_TAR.stat().st_size >= 2_200_000_000


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_one(backbone: str, seed: int, out: Path, *, has_train: bool, epochs: int | None) -> dict:
    out_dir = out / f"{backbone}_seed{seed}"
    cmd = [
        PYTHON,
        str(ROOT / "benchmarks" / "neuromorphic" / "run_dvs_gesture.py"),
        "--root",
        "data/dvs-gesture",
        "--recipe",
        "hires-ds4",
        "--backbone",
        backbone,
        "--with-ann-baseline",
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
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
    extra = result["metrics"].get("extra") or {}
    return {
        "backbone": backbone,
        "seed": seed,
        "snn_quality": result["metrics"]["quality"],
        "cnn_quality": baseline.get("quality"),
        "majority": extra.get("majority_class_accuracy")
        or (baseline.get("extra") or {}).get("majority_class_accuracy"),
        "snn_energy_pj": result["metrics"]["energy_pj"],
        "cnn_energy_pj": baseline.get("energy_pj"),
        "decision_latency_fraction": extra.get("decision_latency_fraction"),
        "param_count": result["metrics"]["param_count"],
        "lr_schedule": extra.get("lr_schedule"),
        "energy_accounting": extra.get("energy_accounting"),
        "recipe": result.get("meta", {}).get("recipe", "hires-ds4"),
    }


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


def write_reports(out: Path, report: dict) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# DVS hires-ds4 full-train freeze",
        "",
        f"- mode: **{report['protocol_mode']}**",
        f"- recipe: `hires-ds4` (downsample=4, cosine LR)",
        f"- train ready: `{report['train_ready']}`",
        f"- seeds: `{report['seeds']}`, epochs: `{report['epochs']}`",
        f"- wall time: `{report['wall_time_s']:.1f}s`",
        "",
        "| backbone | SNN mean±std | range | CNN mean | above maj | params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in report["summaries"]:
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
    for r in report["rows"]:
        lines.append(
            f"| {r['backbone']} | {r['seed']} | {r['snn_quality']:.4f} | "
            f"{float(r['cnn_quality'] or 0):.4f} | {float(r['majority'] or 0):.4f} | "
            f"{float(r['decision_latency_fraction'] or 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim language",
            "",
            "- Valid: hires-ds4 controlled multi-seed under this recipe.",
            "- Not SOTA; compare to parity-ds8 freeze, not literature ~97%.",
            "- Energy: read `energy_accounting` (shared_spatial_mac_proxy_v1 for conv-plif).",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "INTERPRETATION.md").write_text(
        "\n".join(
            [
                "# Interpretation — hires-ds4 freeze",
                "",
                "See `report.md` / `report.json` in this directory.",
                "Milestone parity freeze remains at `dvs-fulltrain-sew/` (ds=8).",
                "This directory freezes the **higher-resolution** controlled push.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=None, help="Override recipe epochs")
    parser.add_argument("--sew", action="store_true", help="Also run sew-plif")
    parser.add_argument("--backbone", default="conv-plif", choices=["conv-plif", "sew-plif"])
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    has_train = train_ready()
    mode = "full-train" if has_train else "smoke-large"
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    backbones = [args.backbone]
    if args.sew and "sew-plif" not in backbones:
        backbones.append("sew-plif")

    print(f"mode={mode} train_ready={has_train} seeds={seeds} backbones={backbones}", flush=True)
    t0 = time.perf_counter()
    rows: list[dict] = []
    for backbone in backbones:
        for seed in seeds:
            rows.append(run_one(backbone, seed, out, has_train=has_train, epochs=args.epochs))

    report = {
        "protocol_mode": mode,
        "train_ready": has_train,
        "recipe": "hires-ds4",
        "epochs": args.epochs if args.epochs is not None else 12,
        "seeds": seeds,
        "wall_time_s": time.perf_counter() - t0,
        "rows": rows,
        "summaries": [summarize(rows, b) for b in backbones],
    }
    write_reports(out, report)


if __name__ == "__main__":
    main()
