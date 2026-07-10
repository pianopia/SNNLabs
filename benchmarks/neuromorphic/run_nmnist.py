#!/usr/bin/env python3
"""Train and evaluate the DST-SNN on N-MNIST, emitting a harness RunResult."""

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
from benchmarks.neuromorphic.datasets import load_nmnist, load_nmnist_test_only
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

NUM_CLASSES = 10


def _maybe_subset(dataset, limit: int):
    if limit and limit < len(dataset):
        return Subset(dataset, list(range(limit)))
    return dataset


def _random_split_from_dataset(dataset, train_limit: int, test_limit: int, seed: int) -> tuple[Subset, Subset]:
    total = min(len(dataset), train_limit + test_limit)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:total].tolist()
    return Subset(dataset, indices[:train_limit]), Subset(dataset, indices[train_limit:total])


class NmnistRunner:
    name = "n-mnist"

    def __init__(
        self,
        root: str,
        *,
        epochs: int = 3,
        batch_size: int = 64,
        time_bins: int = 24,
        device: str = "cpu",
        limit_train: int = 0,
        limit_test: int = 0,
        smoke_from_test: bool = False,
        seed: int = 0,
        threshold: float = 0.85,
        num_branches: int = 16,
        max_delay: int = 16,
        readout: str = "max_membrane",
        use_chrono: bool = False,
        chrono_hidden: int = 128,
    ) -> None:
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_bins = time_bins
        self.device = torch.device(device)
        self.limit_train = limit_train
        self.limit_test = limit_test
        self.smoke_from_test = smoke_from_test
        self.seed = seed
        self.threshold = threshold
        self.num_branches = num_branches
        self.max_delay = max_delay
        self.readout = readout
        self.use_chrono = use_chrono
        self.chrono_hidden = chrono_hidden
        self.model: SnnClassifier | None = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def prepare(self) -> None:
        if self.smoke_from_test:
            test_only, in_features = load_nmnist_test_only(self.root, time_bins=self.time_bins)
            train_limit = self.limit_train or min(128, len(test_only) // 2)
            test_limit = self.limit_test or min(128, len(test_only) - train_limit)
            train, test = _random_split_from_dataset(test_only, train_limit, test_limit, self.seed)
        else:
            train, test, in_features = load_nmnist(self.root, time_bins=self.time_bins)
            train = _maybe_subset(train, self.limit_train)
            test = _maybe_subset(test, self.limit_test)
        generator = torch.Generator().manual_seed(self.seed)
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, generator=generator)
        self.test_loader = DataLoader(test, batch_size=self.batch_size)
        self.model = SnnClassifier(
            in_features,
            NUM_CLASSES,
            num_branches=self.num_branches,
            max_delay=self.max_delay,
            threshold=self.threshold,
            readout=self.readout,
            use_chrono=self.use_chrono,
            chrono_hidden=self.chrono_hidden,
        ).to(self.device)

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
        spike_total, spike_batches = 0.0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                start = time.perf_counter()
                out = self.model(x)
                latencies_ms.append((time.perf_counter() - start) * 1000.0 / max(1, x.shape[0]))
                preds_all.append(out["logits"].argmax(dim=-1))
                targets_all.append(y)
                spike_total += spike_stats(out["spikes"])["spikes_per_inference"]
                spike_batches += 1

        acc = accuracy(torch.cat(preds_all), torch.cat(targets_all))
        lat = latency_percentiles(latencies_ms)
        spikes_per_inf = spike_total / max(1, spike_batches)
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
                extra={
                    "epochs": self.epochs,
                    "fanout": fanout,
                    "threshold": self.threshold,
                    "num_branches": self.num_branches,
                    "max_delay": self.max_delay,
                    "readout": self.readout,
                    "use_chrono": self.use_chrono,
                    "chrono_hidden": self.chrono_hidden,
                },
            ),
            baseline=None,
            meta={"smoke_from_test": self.smoke_from_test, "seed": self.seed},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/nmnist")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-bins", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--smoke-from-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--num-branches", type=int, default=16)
    parser.add_argument("--max-delay", type=int, default=16)
    parser.add_argument("--readout", choices=["spike_count", "max_membrane", "mean_membrane"], default="max_membrane")
    parser.add_argument("--use-chrono", action="store_true")
    parser.add_argument("--chrono-hidden", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = NmnistRunner(
        args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        time_bins=args.time_bins,
        device=args.device,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        smoke_from_test=args.smoke_from_test,
        seed=args.seed,
        threshold=args.threshold,
        num_branches=args.num_branches,
        max_delay=args.max_delay,
        readout=args.readout,
        use_chrono=args.use_chrono,
        chrono_hidden=args.chrono_hidden,
    )
    print(run_benchmarks([runner], args.out_dir)[0].to_json())


if __name__ == "__main__":
    main()
