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
from benchmarks.neuromorphic.datasets import dataset_targets, load_nmnist, load_nmnist_test_only
from benchmarks.neuromorphic.energy_report import pack_snn_energy
from src.dst_snn.eval import (
    MetricSet,
    RunResult,
    accuracy,
    latency_percentiles,
    majority_class_accuracy,
    model_size,
    run_benchmarks,
    spike_stats,
)
from src.dst_snn.eval.baselines import DenseAnnClassifier, train_ann_classifier
from src.dst_snn.eval.energy import dense_mac_energy_pj, EnergyModel

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


def _stratified_split_from_dataset(dataset, train_limit: int, test_limit: int, seed: int) -> tuple[Subset, Subset]:
    targets = dataset_targets(dataset)
    if targets is None:
        return _random_split_from_dataset(dataset, train_limit, test_limit, seed)

    generator = torch.Generator().manual_seed(seed)
    per_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        per_class.setdefault(target, []).append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(per_class):
        indices = per_class[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in order]
        train_take = min(len(shuffled), max(1, round(train_limit * len(shuffled) / len(targets))))
        test_take = min(len(shuffled) - train_take, max(1, round(test_limit * len(shuffled) / len(targets))))
        train_indices.extend(shuffled[:train_take])
        test_indices.extend(shuffled[train_take:train_take + test_take])

    train_set = set(train_indices[:train_limit])
    test_indices = [idx for idx in test_indices if idx not in train_set]
    return Subset(dataset, train_indices[:train_limit]), Subset(dataset, test_indices[:test_limit])


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
        hidden_features: int = 0,
        hidden_threshold: float | None = None,
        hidden_output: str = "spikes",
        with_ann_baseline: bool = False,
        ann_hidden: int = 128,
        with_llm_baseline: bool = False,
        llm_backend: str = "scripted",
        llm_max_samples: int = 0,
        use_temporal_features: bool = False,
        temporal_project_to: int = 0,
        temporal_alpha: float = 0.25,
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
        self.hidden_features = hidden_features
        self.hidden_threshold = hidden_threshold
        self.hidden_output = hidden_output
        self.with_ann_baseline = with_ann_baseline
        self.ann_hidden = ann_hidden
        self.with_llm_baseline = with_llm_baseline
        self.llm_backend = llm_backend
        self.llm_max_samples = llm_max_samples
        self.use_temporal_features = use_temporal_features
        self.temporal_project_to = temporal_project_to
        self.temporal_alpha = temporal_alpha
        self.model: SnnClassifier | None = None
        self.ann_model: DenseAnnClassifier | None = None
        self.in_features: int = 0
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def prepare(self) -> None:
        torch.manual_seed(self.seed)
        if self.smoke_from_test:
            test_only, in_features = load_nmnist_test_only(self.root, time_bins=self.time_bins)
            train_limit = self.limit_train or min(128, len(test_only) // 2)
            test_limit = self.limit_test or min(128, len(test_only) - train_limit)
            train, test = _stratified_split_from_dataset(test_only, train_limit, test_limit, self.seed)
        else:
            train, test, in_features = load_nmnist(self.root, time_bins=self.time_bins)
            train = _maybe_subset(train, self.limit_train)
            test = _maybe_subset(test, self.limit_test)
        self.in_features = in_features
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
            hidden_features=self.hidden_features,
            hidden_threshold=self.hidden_threshold,
            hidden_output=self.hidden_output,
            use_temporal_features=self.use_temporal_features,
            temporal_project_to=self.temporal_project_to,
            temporal_alpha=self.temporal_alpha,
        ).to(self.device)
        if self.with_ann_baseline:
            self.ann_model = DenseAnnClassifier(in_features, NUM_CLASSES, hidden=self.ann_hidden).to(self.device)

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
        spike_total, active_total, spike_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                start = time.perf_counter()
                out = self.model(x)
                latencies_ms.append((time.perf_counter() - start) * 1000.0 / max(1, x.shape[0]))
                preds_all.append(out["logits"].argmax(dim=-1))
                targets_all.append(y)
                stats = spike_stats(out["spikes"])
                spike_total += stats["spikes_per_inference"]
                active_total += stats["active_neuron_fraction"]
                spike_batches += 1

        acc = accuracy(torch.cat(preds_all), torch.cat(targets_all))
        targets = torch.cat(targets_all)
        majority_acc = majority_class_accuracy(targets, NUM_CLASSES)
        lat = latency_percentiles(latencies_ms)
        spikes_per_inf = spike_total / max(1, spike_batches)
        active_fraction = active_total / max(1, spike_batches)
        energy = pack_snn_energy(
            in_features=self.in_features,
            num_classes=NUM_CLASSES,
            time_bins=self.time_bins,
            spikes_per_inference=spikes_per_inf,
            hidden_features=self.hidden_features,
            chrono_hidden=self.chrono_hidden if self.use_chrono else 0,
        )
        size = model_size(self.model)

        baseline_quality = majority_acc
        baseline_metric = "majority_class_accuracy"
        baseline_energy_pj = float(energy["dense_energy_pj"])
        baseline_source = "dense-mac proxy (same widths × time bins)"
        baseline_params = 0
        baseline_bytes = 0
        baseline_extra: dict = {
            "uniform_chance_accuracy": 1.0 / NUM_CLASSES,
            "dense_mac_ops": energy["dense_mac_ops"],
            "energy_ratio_dense_over_snn": energy["energy_ratio_dense_over_snn"],
        }
        if self.ann_model is not None and self.train_loader is not None and self.test_loader is not None:
            train_ann_classifier(self.ann_model, self.train_loader, epochs=self.epochs, device=self.device)
            self.ann_model.eval()
            ann_preds, ann_targets = [], []
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    ann_preds.append(self.ann_model(x).argmax(dim=-1))
                    ann_targets.append(y)
            baseline_quality = accuracy(torch.cat(ann_preds), torch.cat(ann_targets))
            baseline_metric = "ann_mlp_accuracy"
            ann_size = model_size(self.ann_model)
            baseline_params = ann_size["param_count"]
            baseline_bytes = ann_size["model_bytes"]
            ann_macs = self.ann_model.mac_ops_per_inference(self.time_bins)
            baseline_energy_pj = dense_mac_energy_pj(ann_macs, EnergyModel())
            baseline_source = "dense ANN MLP (mean-pool + 2-layer)"
            baseline_extra["ann_mac_ops"] = ann_macs
            baseline_extra["ann_hidden"] = self.ann_hidden
            baseline_extra["energy_ratio_ann_over_snn"] = (
                baseline_energy_pj / float(energy["energy_pj"])
                if float(energy["energy_pj"]) > 0
                else float("inf")
            )

        baseline = MetricSet(
            quality=baseline_quality,
            quality_metric=baseline_metric,
            latency_ms_p50=0.0,
            latency_ms_p95=0.0,
            spikes_per_inference=0.0,
            active_neuron_fraction=0.0,
            energy_pj=baseline_energy_pj,
            energy_source=baseline_source,
            param_count=baseline_params,
            model_bytes=baseline_bytes,
            extra=baseline_extra,
        )
        metrics_extra: dict = {
            "epochs": self.epochs,
            "fanout": energy["fanout"],
            "dense_mac_ops": energy["dense_mac_ops"],
            "dense_energy_pj": energy["dense_energy_pj"],
            "energy_ratio_dense_over_snn": energy["energy_ratio_dense_over_snn"],
            "layer_count": energy["layer_count"],
            "threshold": self.threshold,
            "num_branches": self.num_branches,
            "max_delay": self.max_delay,
            "readout": self.readout,
            "use_chrono": self.use_chrono,
            "chrono_hidden": self.chrono_hidden,
            "hidden_features": self.hidden_features,
            "hidden_threshold": self.hidden_threshold,
            "hidden_output": self.hidden_output,
            "use_temporal_features": self.use_temporal_features,
            "temporal_project_to": self.temporal_project_to,
        }
        metrics_extra["majority_class_accuracy"] = majority_acc
        if self.with_llm_baseline and self.test_loader is not None:
            from benchmarks.neuromorphic.llm_baseline_util import (
                attach_llm_to_result,
                nmnist_class_names,
                run_llm_baseline,
            )

            majority_class = int(targets.mode().values.item()) if targets.numel() else 0
            llm_metrics = run_llm_baseline(
                self.test_loader,
                num_classes=NUM_CLASSES,
                class_names=nmnist_class_names(),
                backend_kind=self.llm_backend,
                majority_class=majority_class,
                max_samples=self.llm_max_samples,
            )
            baseline, llm_patch = attach_llm_to_result(
                llm_metrics=llm_metrics,
                primary_baseline=baseline,
                had_ann_or_cnn=self.ann_model is not None,
            )
            metrics_extra.update(llm_patch)

        return RunResult(
            benchmark=self.name,
            model="dst-snn",
            metrics=MetricSet(
                quality=acc,
                quality_metric="accuracy",
                latency_ms_p50=lat["p50"],
                latency_ms_p95=lat["p95"],
                spikes_per_inference=spikes_per_inf,
                active_neuron_fraction=active_fraction,
                energy_pj=float(energy["energy_pj"]),
                energy_source=str(energy["energy_source"]),
                param_count=size["param_count"],
                model_bytes=size["model_bytes"],
                extra=metrics_extra,
            ),
            baseline=baseline,
            meta={
                "smoke_from_test": self.smoke_from_test,
                "seed": self.seed,
                "with_llm_baseline": self.with_llm_baseline,
                "llm_backend": self.llm_backend if self.with_llm_baseline else None,
            },
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
    parser.add_argument("--hidden-features", type=int, default=0)
    parser.add_argument("--hidden-threshold", type=float, default=None)
    parser.add_argument("--hidden-output", choices=["spikes", "membrane"], default="spikes")
    parser.add_argument("--with-ann-baseline", action="store_true")
    parser.add_argument("--ann-hidden", type=int, default=128)
    parser.add_argument(
        "--with-llm-baseline",
        action="store_true",
        help="Optional LLM classification baseline (scripted offline default; not product path).",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["scripted", "majority", "http"],
        default="scripted",
        help="scripted/majority = offline weak baseline; http = OpenAI-compatible API.",
    )
    parser.add_argument(
        "--llm-max-samples",
        type=int,
        default=0,
        help="Cap LLM baseline samples (0 = full test set).",
    )
    parser.add_argument("--use-temporal-features", action="store_true")
    parser.add_argument("--temporal-project-to", type=int, default=0)
    parser.add_argument("--temporal-alpha", type=float, default=0.25)
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
        hidden_features=args.hidden_features,
        hidden_threshold=args.hidden_threshold,
        hidden_output=args.hidden_output,
        with_ann_baseline=args.with_ann_baseline,
        ann_hidden=args.ann_hidden,
        with_llm_baseline=args.with_llm_baseline,
        llm_backend=args.llm_backend,
        llm_max_samples=args.llm_max_samples,
        use_temporal_features=args.use_temporal_features,
        temporal_project_to=args.temporal_project_to,
        temporal_alpha=args.temporal_alpha,
    )
    print(run_benchmarks([runner], args.out_dir)[0].to_json())


if __name__ == "__main__":
    main()
