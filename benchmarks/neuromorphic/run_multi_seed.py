#!/usr/bin/env python3
"""Run N-MNIST or DVS Gesture across multiple seeds and summarize.

Usage:
    python benchmarks/neuromorphic/run_multi_seed.py \\
      --benchmark dvs-gesture --seeds 0,1,2,13,42 --smoke-from-test \\
      --limit-train 96 --limit-test 64 --epochs 5 \\
      --hidden-features 64 --no-chrono --threshold 0.1 \\
      --use-temporal-features --temporal-project-to 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.neuromorphic.run_dvs_gesture import DvsGestureRunner
from benchmarks.neuromorphic.run_nmnist import NmnistRunner
from src.dst_snn.eval.result import RunResult, write_report


def _parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _summarize(results: list[RunResult], *, config: dict) -> dict:
    qualities = [r.metrics.quality for r in results]
    energies = [r.metrics.energy_pj for r in results]
    lat_p50 = [r.metrics.latency_ms_p50 for r in results]
    ratios = [float(r.metrics.extra.get("energy_ratio_dense_over_snn", 0.0)) for r in results]
    decision = [
        float(r.metrics.extra["decision_latency_fraction"])
        for r in results
        if "decision_latency_fraction" in r.metrics.extra
    ]
    majority = []
    for r in results:
        if r.baseline is not None and r.baseline.quality_metric in {
            "majority_class_accuracy",
            "ann_mlp_accuracy",
        }:
            # Prefer majority stored in extra when ANN baseline replaced quality.
            maj = r.baseline.extra.get("majority_class_accuracy")
            if maj is not None:
                majority.append(float(maj))
            elif r.baseline.quality_metric == "majority_class_accuracy":
                majority.append(float(r.baseline.quality))
        elif r.baseline is not None:
            maj = r.baseline.extra.get("majority_class_accuracy")
            if maj is not None:
                majority.append(float(maj))
    # Uniform chance for DVS=11 / N-MNIST=10 if present.
    chance = []
    for r in results:
        if r.baseline is not None and "uniform_chance_accuracy" in r.baseline.extra:
            chance.append(float(r.baseline.extra["uniform_chance_accuracy"]))

    maj_mean = statistics.fmean(majority) if majority else None
    beats_majority = (
        sum(1 for q, m in zip(qualities, majority) if q > m) if majority else None
    )
    summary = {
        "n_seeds": len(results),
        "config": config,
        "quality_mean": statistics.fmean(qualities) if qualities else 0.0,
        "quality_std": statistics.pstdev(qualities) if len(qualities) > 1 else 0.0,
        "quality_min": min(qualities) if qualities else 0.0,
        "quality_max": max(qualities) if qualities else 0.0,
        "per_seed_quality": {
            str(r.meta.get("seed", i)): r.metrics.quality for i, r in enumerate(results)
        },
        "energy_pj_mean": statistics.fmean(energies) if energies else 0.0,
        "latency_ms_p50_mean": statistics.fmean(lat_p50) if lat_p50 else 0.0,
        "energy_ratio_mean": statistics.fmean(ratios) if ratios else 0.0,
        "majority_mean": maj_mean,
        "seeds_above_majority": beats_majority,
        "margin_over_majority_mean": (
            statistics.fmean(qualities) - maj_mean if maj_mean is not None and qualities else None
        ),
        "uniform_chance": statistics.fmean(chance) if chance else None,
    }
    if decision:
        summary["decision_latency_fraction_mean"] = statistics.fmean(decision)
        summary["decision_latency_fraction_std"] = (
            statistics.pstdev(decision) if len(decision) > 1 else 0.0
        )
    return summary


def _write_summary_md(path: Path, summary: dict, results: list[RunResult]) -> None:
    lines = [
        "# Multi-seed summary",
        "",
        f"- seeds: `{summary['n_seeds']}`",
        f"- quality mean±std: **{summary['quality_mean']:.4f} ± {summary['quality_std']:.4f}**",
        f"- quality range: [{summary['quality_min']:.4f}, {summary['quality_max']:.4f}]",
    ]
    if summary.get("majority_mean") is not None:
        lines.append(f"- majority baseline mean: **{summary['majority_mean']:.4f}**")
        lines.append(
            f"- seeds above majority: **{summary['seeds_above_majority']}/{summary['n_seeds']}**"
        )
        if summary.get("margin_over_majority_mean") is not None:
            lines.append(
                f"- mean margin over majority: **{summary['margin_over_majority_mean']:+.4f}**"
            )
    if summary.get("decision_latency_fraction_mean") is not None:
        lines.append(
            f"- decision latency fraction mean: **{summary['decision_latency_fraction_mean']:.4f}**"
        )
    lines.extend(["", "## Per seed", "", "| seed | quality | majority | decision_lat |", "|---|---:|---:|---:|"])
    for r in results:
        seed = r.meta.get("seed", "?")
        maj = ""
        if r.baseline is not None:
            maj_v = r.baseline.extra.get("majority_class_accuracy", r.baseline.quality)
            maj = f"{float(maj_v):.4f}"
        dec = r.metrics.extra.get("decision_latency_fraction")
        dec_s = f"{float(dec):.4f}" if dec is not None else ""
        lines.append(f"| {seed} | {r.metrics.quality:.4f} | {maj} | {dec_s} |")
    lines.extend(["", "## Config", "", "```json", json.dumps(summary.get("config", {}), indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["n-mnist", "dvs-gesture"], default="n-mnist")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks/multi-seed"))
    p.add_argument("--root", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--time-bins", type=int, default=12)
    p.add_argument("--limit-train", type=int, default=256)
    p.add_argument("--limit-test", type=int, default=128)
    p.add_argument("--smoke-from-test", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hidden-features", type=int, default=0)
    p.add_argument("--hidden-threshold", type=float, default=None)
    p.add_argument("--hidden-output", choices=["spikes", "membrane"], default="spikes")
    p.add_argument("--readout", default="max_membrane")
    p.add_argument("--use-temporal-features", action="store_true")
    p.add_argument("--temporal-project-to", type=int, default=128)
    p.add_argument("--temporal-alpha", type=float, default=0.25)
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument("--use-chrono", dest="use_chrono", action="store_true", default=False)
    p.add_argument("--no-chrono", dest="use_chrono", action="store_false")
    p.add_argument("--chrono-hidden", type=int, default=64)
    p.add_argument("--num-branches", type=int, default=16)
    p.add_argument("--max-delay", type=int, default=16)
    p.add_argument("--backbone", choices=["dendritic", "conv-plif", "sew-plif"], default="dendritic")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sew-width", type=int, default=32)
    p.add_argument("--sew-blocks", type=int, default=2)
    p.add_argument("--tag", default="", help="Optional label stored in summary config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    config = {
        "tag": args.tag or out_dir.name,
        "benchmark": args.benchmark,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "time_bins": args.time_bins,
        "limit_train": args.limit_train,
        "limit_test": args.limit_test,
        "smoke_from_test": args.smoke_from_test,
        "threshold": args.threshold,
        "hidden_features": args.hidden_features,
        "hidden_threshold": args.hidden_threshold,
        "hidden_output": args.hidden_output,
        "readout": args.readout,
        "use_temporal_features": args.use_temporal_features,
        "temporal_project_to": args.temporal_project_to,
        "use_chrono": args.use_chrono,
        "chrono_hidden": args.chrono_hidden,
        "downsample": args.downsample if args.benchmark == "dvs-gesture" else None,
        "backbone": args.backbone if args.benchmark == "dvs-gesture" else "dendritic",
        "lr": args.lr,
        "seeds": seeds,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    for seed in seeds:
        seed_t0 = time.perf_counter()
        if args.benchmark == "n-mnist":
            root = args.root or "data/nmnist"
            runner = NmnistRunner(
                root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                time_bins=args.time_bins,
                device=args.device,
                limit_train=args.limit_train,
                limit_test=args.limit_test,
                smoke_from_test=args.smoke_from_test,
                seed=seed,
                threshold=args.threshold,
                num_branches=args.num_branches,
                max_delay=args.max_delay,
                readout=args.readout,
                use_chrono=args.use_chrono,
                chrono_hidden=args.chrono_hidden,
                hidden_features=args.hidden_features,
                hidden_threshold=args.hidden_threshold,
                hidden_output=args.hidden_output,
                use_temporal_features=args.use_temporal_features,
                temporal_project_to=args.temporal_project_to,
                temporal_alpha=args.temporal_alpha,
            )
        else:
            root = args.root or "data/dvs-gesture"
            runner = DvsGestureRunner(
                root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                time_bins=args.time_bins,
                downsample=args.downsample,
                device=args.device,
                limit_train=args.limit_train,
                limit_test=args.limit_test,
                smoke_from_test=args.smoke_from_test,
                seed=seed,
                threshold=args.threshold,
                num_branches=args.num_branches,
                max_delay=args.max_delay,
                readout=args.readout,
                use_chrono=args.use_chrono,
                chrono_hidden=args.chrono_hidden,
                hidden_features=args.hidden_features,
                hidden_threshold=args.hidden_threshold,
                hidden_output=args.hidden_output,
                use_temporal_features=args.use_temporal_features,
                temporal_project_to=args.temporal_project_to,
                temporal_alpha=args.temporal_alpha,
                backbone=args.backbone,
                lr=args.lr,
                sew_width=args.sew_width,
                sew_blocks=args.sew_blocks,
            )
        runner.prepare()
        result = runner.run()
        # Ensure seed is always present for summarizer.
        result.meta["seed"] = seed
        (out_dir / f"{runner.name}-seed{seed}.json").write_text(result.to_json(), encoding="utf-8")
        results.append(result)
        maj = None
        if result.baseline is not None:
            maj = result.baseline.extra.get("majority_class_accuracy", result.baseline.quality)
        print(
            f"seed={seed} quality={result.metrics.quality:.4f}"
            + (f" majority={float(maj):.4f}" if maj is not None else "")
            + f" elapsed={time.perf_counter() - seed_t0:.1f}s",
            flush=True,
        )

    summary = _summarize(results, config=config)
    summary["wall_time_s"] = time.perf_counter() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(results, out_dir / "report.md")
    _write_summary_md(out_dir / "summary.md", summary, results)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
