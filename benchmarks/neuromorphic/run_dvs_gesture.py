#!/usr/bin/env python3
"""Train and evaluate the DST-SNN on DVS128 Gesture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from benchmarks.neuromorphic.classifier import SnnClassifier
from benchmarks.neuromorphic.datasets import load_dvs_gesture
from benchmarks.neuromorphic.decision_latency import decision_latency_fraction
from src.dst_snn.eval import (
    EnergyModel,
    MetricSet,
    RunResult,
    accuracy,
    latency_percentiles,
    model_size,
    run_benchmarks,
    snn_energy_pj,
    spike_stats,
)

NUM_CLASSES = 11


def _maybe_subset(dataset, limit: int):
    if limit and limit < len(dataset):
        return Subset(dataset, list(range(limit)))
    return dataset


class DvsGestureRunner:
    name = "dvs-gesture"

    def __init__(
        self,
        root: str,
        *,
        epochs: int = 3,
        batch_size: int = 32,
        time_bins: int = 32,
        downsample: int = 4,
        device: str = "cpu",
        limit_train: int = 0,
        limit_test: int = 0,
    ) -> None:
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_bins = time_bins
        self.downsample = downsample
        self.device = torch.device(device)
        self.limit_train = limit_train
        self.limit_test = limit_test
        self.model: SnnClassifier | None = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def prepare(self) -> None:
        train, test, in_features = load_dvs_gesture(self.root, time_bins=self.time_bins, downsample=self.downsample)
        self.train_loader = DataLoader(_maybe_subset(train, self.limit_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(_maybe_subset(test, self.limit_test), batch_size=self.batch_size)
        self.model = SnnClassifier(in_features, NUM_CLASSES).to(self.device)

    def run(self) -> RunResult:
        assert self.model is not None and self.train_loader is not None and self.test_loader is not None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for _ in range(self.epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = nn.functional.cross_entropy(out["logits"], y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.model.eval()
        preds_all, targets_all = [], []
        latencies_ms: list[float] = []
        latency_fracs: list[float] = []
        spike_total, spike_batches = 0.0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                start = time.perf_counter()
                out = self.model(x)
                latencies_ms.append((time.perf_counter() - start) * 1000.0 / max(1, x.shape[0]))
                preds_all.append(out["logits"].argmax(dim=-1))
                targets_all.append(y)
                latency_fracs.append(decision_latency_fraction(out["spikes"], y, confirm_window=3))
                spike_total += spike_stats(out["spikes"])["spikes_per_inference"]
                spike_batches += 1

        acc = accuracy(torch.cat(preds_all), torch.cat(targets_all))
        lat = latency_percentiles(latencies_ms)
        spikes_per_inf = spike_total / max(1, spike_batches)
        decision_latency = sum(latency_fracs) / max(1, len(latency_fracs))
        energy_model = EnergyModel()
        fanout = NUM_CLASSES
        size = model_size(self.model)
        return RunResult(
            benchmark=self.name,
            model="dst-snn",
            metrics=MetricSet(
                quality=acc,
                quality_metric="accuracy",
                latency_ms_p50=lat["p50"],
                latency_ms_p95=lat["p95"],
                spikes_per_inference=spikes_per_inf,
                active_neuron_fraction=0.0,
                energy_pj=snn_energy_pj(spikes_per_inf, fanout, energy_model),
                energy_source=energy_model.source,
                param_count=size["param_count"],
                model_bytes=size["model_bytes"],
                extra={"epochs": self.epochs, "fanout": fanout, "decision_latency_fraction": decision_latency},
            ),
            baseline=None,
            meta={"downsample": self.downsample},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/dvs-gesture")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--time-bins", type=int, default=32)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = DvsGestureRunner(
        args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        time_bins=args.time_bins,
        downsample=args.downsample,
        device=args.device,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    print(run_benchmarks([runner], args.out_dir)[0].to_json())


if __name__ == "__main__":
    main()
