#!/usr/bin/env python3
"""Multi-seed LLM baseline freeze (scripted + optional HTTP sample).

Default: scripted majority on a stratified smoke split — offline, free, CI-safe.
Optional: ``--http`` with ``OPENAI_API_KEY`` runs a small capped HTTP sample and
records quality/latency/token energy (not AC/MAC comparable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from benchmarks.neuromorphic.datasets import dataset_targets, load_nmnist_test_only
from benchmarks.neuromorphic.llm_baseline_util import nmnist_class_names, run_llm_baseline
from src.dst_snn.eval.result import MetricSet


def stratified_indices(dataset, n: int, seed: int) -> list[int]:
    targets = dataset_targets(dataset)
    if targets is None:
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(len(dataset), generator=g)[:n].tolist()
    per_class: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        per_class.setdefault(int(t), []).append(i)
    g = torch.Generator().manual_seed(seed)
    picked: list[int] = []
    labels = sorted(per_class)
    per = max(1, n // max(1, len(labels)))
    for label in labels:
        idxs = per_class[label]
        order = torch.randperm(len(idxs), generator=g).tolist()
        for j in order[:per]:
            picked.append(idxs[j])
            if len(picked) >= n:
                return picked[:n]
    return picked[:n]


def run_seed(
    *,
    seed: int,
    backend: str,
    n_samples: int,
    root: str,
    time_bins: int,
) -> dict:
    dataset, _ = load_nmnist_test_only(root, time_bins=time_bins)
    indices = stratified_indices(dataset, n_samples, seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=8)
    # majority class of this subset
    ys = []
    for _, y in loader:
        ys.extend(y.tolist())
    majority = max(set(ys), key=ys.count) if ys else 0
    t0 = time.perf_counter()
    metrics: MetricSet = run_llm_baseline(
        loader,
        num_classes=10,
        class_names=nmnist_class_names(),
        backend_kind=backend,
        majority_class=majority,
        max_samples=n_samples,
    )
    wall = time.perf_counter() - t0
    return {
        "seed": seed,
        "backend": backend,
        "n_samples": len(ys),
        "majority_class": majority,
        "quality": metrics.quality,
        "latency_ms_p50": metrics.latency_ms_p50,
        "latency_ms_p95": metrics.latency_ms_p95,
        "energy_pj": metrics.energy_pj,
        "energy_source": metrics.energy_source,
        "energy_accounting": metrics.extra.get("energy_accounting"),
        "prompt_tokens": metrics.extra.get("prompt_tokens"),
        "completion_tokens": metrics.extra.get("completion_tokens"),
        "parse_failures": metrics.extra.get("parse_failures"),
        "wall_time_s": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "benchmarks" / "llm-baseline-multiseed")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--time-bins", type=int, default=8)
    parser.add_argument("--root", default="data/nmnist")
    parser.add_argument("--http", action="store_true", help="Also run a tiny HTTP sample (needs API key)")
    parser.add_argument("--http-samples", type=int, default=8)
    parser.add_argument("--http-seeds", default="0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        row = run_seed(
            seed=seed,
            backend="scripted",
            n_samples=args.n_samples,
            root=args.root,
            time_bins=args.time_bins,
        )
        rows.append(row)
        print(
            f"scripted seed={seed} quality={row['quality']:.4f} "
            f"p50={row['latency_ms_p50']:.3f}ms tokens={row['prompt_tokens']}",
            flush=True,
        )

    http_rows = []
    if args.http:
        http_seeds = [int(s.strip()) for s in args.http_seeds.split(",") if s.strip()]
        for seed in http_seeds:
            try:
                row = run_seed(
                    seed=seed,
                    backend="http",
                    n_samples=args.http_samples,
                    root=args.root,
                    time_bins=args.time_bins,
                )
                http_rows.append(row)
                print(
                    f"http seed={seed} quality={row['quality']:.4f} "
                    f"p50={row['latency_ms_p50']:.1f}ms tokens={row['prompt_tokens']}",
                    flush=True,
                )
            except Exception as exc:
                http_rows.append({"seed": seed, "backend": "http", "error": str(exc)})
                print(f"http seed={seed} FAILED: {exc}", flush=True)

    def summarize(part: list[dict]) -> dict | None:
        ok = [r for r in part if "quality" in r]
        if not ok:
            return None
        q = [r["quality"] for r in ok]
        return {
            "n": len(ok),
            "quality_mean": statistics.fmean(q),
            "quality_std": statistics.pstdev(q) if len(q) > 1 else 0.0,
            "quality_min": min(q),
            "quality_max": max(q),
            "latency_ms_p50_mean": statistics.fmean([r["latency_ms_p50"] for r in ok]),
        }

    report = {
        "benchmark": "n-mnist-llm-baseline",
        "scripted_rows": rows,
        "scripted_summary": summarize(rows),
        "http_rows": http_rows,
        "http_summary": summarize(http_rows),
        "notes": [
            "Scripted backend emits majority class (weak offline baseline).",
            "HTTP energy uses llm_token_proxy_v1 — not comparable to SNN AC/MAC.",
            "Not a product path; optional Phase 0 eval interface only.",
        ],
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# LLM baseline multi-seed",
        "",
        "## Scripted (majority)",
        "",
        f"- seeds: `{seeds}`, n_samples/seed: `{args.n_samples}`",
    ]
    if report["scripted_summary"]:
        s = report["scripted_summary"]
        lines.append(
            f"- quality: **{s['quality_mean']:.4f}±{s['quality_std']:.4f}** "
            f"[{s['quality_min']:.4f},{s['quality_max']:.4f}]"
        )
    if http_rows:
        lines.extend(["", "## HTTP (optional)", ""])
        if report["http_summary"]:
            s = report["http_summary"]
            lines.append(
                f"- quality: **{s['quality_mean']:.4f}±{s['quality_std']:.4f}** "
                f"(n={s['n']}, samples/seed={args.http_samples})"
            )
        for r in http_rows:
            if "error" in r:
                lines.append(f"- seed {r['seed']}: ERROR {r['error']}")
            else:
                lines.append(
                    f"- seed {r['seed']}: quality={r['quality']:.4f} "
                    f"p50={r['latency_ms_p50']:.1f}ms"
                )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {n}" for n in report["notes"])
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
