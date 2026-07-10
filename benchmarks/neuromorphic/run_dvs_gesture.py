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
from benchmarks.neuromorphic.conv_snn import ConvPLIFClassifier, SewConvPLIFClassifier
from benchmarks.neuromorphic.datasets import dataset_targets, load_dvs_gesture, load_dvs_gesture_test_only
from benchmarks.neuromorphic.decision_latency import decision_latency_fraction
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
from src.dst_snn.eval.baselines import (
    DenseAnnClassifier,
    FrameCnnClassifier,
    train_ann_classifier,
    train_frame_cnn,
)
from src.dst_snn.eval.energy import EnergyModel, dense_mac_energy_pj, energy_ratio, snn_energy_pj

NUM_CLASSES = 11


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
        smoke_from_test: bool = False,
        seed: int = 0,
        threshold: float = 0.85,
        num_branches: int = 16,
        max_delay: int = 16,
        readout: str = "max_membrane",
        use_chrono: bool = True,
        chrono_hidden: int = 128,
        hidden_features: int = 0,
        hidden_threshold: float | None = None,
        hidden_output: str = "spikes",
        with_ann_baseline: bool = False,
        ann_hidden: int = 128,
        use_temporal_features: bool = False,
        temporal_project_to: int = 0,
        temporal_alpha: float = 0.25,
        backbone: str = "dendritic",
        lr: float = 1e-3,
        plif_channels: tuple[int, int, int] = (32, 64, 64),
        sew_width: int = 32,
        sew_blocks: int = 2,
    ) -> None:
        if backbone not in {"dendritic", "conv-plif", "sew-plif"}:
            raise ValueError("backbone must be 'dendritic', 'conv-plif', or 'sew-plif'")
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_bins = time_bins
        self.downsample = downsample
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
        self.use_temporal_features = use_temporal_features
        self.temporal_project_to = temporal_project_to
        self.temporal_alpha = temporal_alpha
        self.backbone = backbone
        self.lr = lr
        self.plif_channels = plif_channels
        self.sew_width = sew_width
        self.sew_blocks = sew_blocks
        self.model: nn.Module | None = None
        self.ann_model: DenseAnnClassifier | None = None
        self.cnn_model: FrameCnnClassifier | None = None
        self.in_features: int = 0
        self.spatial_hw: tuple[int, int] = (1, 1)
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def prepare(self) -> None:
        torch.manual_seed(self.seed)
        mode = "frames" if self.backbone in {"conv-plif", "sew-plif"} else "flat"
        if self.smoke_from_test:
            test_only, in_features = load_dvs_gesture_test_only(
                self.root,
                time_bins=self.time_bins,
                downsample=self.downsample,
                mode=mode,
            )
            train_limit = self.limit_train or min(64, len(test_only) // 2)
            test_limit = self.limit_test or min(64, len(test_only) - train_limit)
            train, test = _stratified_split_from_dataset(test_only, train_limit, test_limit, self.seed)
        else:
            train, test, in_features = load_dvs_gesture(
                self.root,
                time_bins=self.time_bins,
                downsample=self.downsample,
                mode=mode,
            )
            train = _maybe_subset(train, self.limit_train)
            test = _maybe_subset(test, self.limit_test)
        self.in_features = in_features
        side = max(1, int(round((in_features / 2) ** 0.5)))
        self.spatial_hw = (side, side)
        generator = torch.Generator().manual_seed(self.seed)
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, generator=generator)
        self.test_loader = DataLoader(test, batch_size=self.batch_size)
        if self.backbone == "conv-plif":
            # Conv-BN-PLIF keeps polarity×H×W geometry (SEW-ResNet style).
            self.model = ConvPLIFClassifier(
                in_channels=2,
                num_classes=NUM_CLASSES,
                channels=self.plif_channels,
                threshold=self.threshold,
                readout=self.readout,
            ).to(self.device)
            if self.with_ann_baseline:
                self.cnn_model = FrameCnnClassifier(
                    in_channels=2,
                    num_classes=NUM_CLASSES,
                    channels=self.plif_channels,
                ).to(self.device)
        elif self.backbone == "sew-plif":
            self.model = SewConvPLIFClassifier(
                in_channels=2,
                num_classes=NUM_CLASSES,
                width=self.sew_width,
                blocks_per_stage=self.sew_blocks,
                threshold=self.threshold,
                readout=self.readout,
            ).to(self.device)
            if self.with_ann_baseline:
                # Approximate matched CNN width for energy/quality reference.
                w = self.sew_width
                self.cnn_model = FrameCnnClassifier(
                    in_channels=2,
                    num_classes=NUM_CLASSES,
                    channels=(w, w, w * 2),
                ).to(self.device)
        else:
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        spike_total, active_total, spike_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                start = time.perf_counter()
                out = self.model(x)
                latencies_ms.append((time.perf_counter() - start) * 1000.0 / max(1, x.shape[0]))
                preds_all.append(out["logits"].argmax(dim=-1))
                targets_all.append(y)
                latency_fracs.append(decision_latency_fraction(out["spikes"], y, confirm_window=3))
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
        decision_latency = sum(latency_fracs) / max(1, len(latency_fracs))
        if self.backbone in {"conv-plif", "sew-plif"}:
            # Rough AC proxy: feature spikes × channel fanout of the readout.
            energy_model = EnergyModel()
            fanout = int(getattr(self.model, "out_channels", NUM_CLASSES))
            snn_pj = snn_energy_pj(spikes_per_inf, fanout, energy_model)
            # Dense MAC proxy: order-of-magnitude over spatial frames × T.
            side = max(1, int((self.in_features / 2) ** 0.5))
            if self.backbone == "sew-plif":
                w = self.sew_width
                dense_macs = float(self.time_bins) * (
                    2 * w * 9 * side * side
                    + w * w * 9 * side * side * self.sew_blocks * 2
                    + w * w * 9 * (side // 2) * (side // 2) * self.sew_blocks * 2
                    + w * (2 * w) * 9 * (side // 4) * (side // 4)
                    + (2 * w) * (2 * w) * 9 * (side // 4) * (side // 4) * self.sew_blocks * 2
                    + (2 * w) * NUM_CLASSES
                )
                layer_count = float(1 + self.sew_blocks * 3 + 1)
            else:
                dense_macs = float(self.time_bins) * (
                    2 * 32 * 9 * side * side
                    + 32 * 64 * 9 * (side // 2) * (side // 2)
                    + 64 * 64 * 9 * (side // 4) * (side // 4)
                    + 64 * NUM_CLASSES
                )
                layer_count = 3.0
            dense_pj = dense_mac_energy_pj(dense_macs, energy_model)
            energy = {
                "energy_pj": snn_pj,
                "energy_source": energy_model.source,
                "fanout": float(fanout),
                "dense_mac_ops": dense_macs,
                "dense_energy_pj": dense_pj,
                "energy_ratio_dense_over_snn": energy_ratio(snn_pj, dense_pj),
                "layer_count": layer_count,
            }
        else:
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
            "majority_class_accuracy": majority_acc,
        }
        if self.ann_model is not None and self.train_loader is not None and self.test_loader is not None:
            train_ann_classifier(self.ann_model, self.train_loader, epochs=self.epochs, device=self.device, lr=self.lr)
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
        if self.cnn_model is not None and self.train_loader is not None and self.test_loader is not None:
            train_frame_cnn(self.cnn_model, self.train_loader, epochs=self.epochs, device=self.device, lr=self.lr)
            self.cnn_model.eval()
            cnn_preds, cnn_targets = [], []
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    cnn_preds.append(self.cnn_model(x).argmax(dim=-1))
                    cnn_targets.append(y)
            baseline_quality = accuracy(torch.cat(cnn_preds), torch.cat(cnn_targets))
            baseline_metric = "frame_cnn_accuracy"
            cnn_size = model_size(self.cnn_model)
            baseline_params = cnn_size["param_count"]
            baseline_bytes = cnn_size["model_bytes"]
            h, w = self.spatial_hw
            cnn_macs = self.cnn_model.mac_ops_per_inference(self.time_bins, h, w)
            baseline_energy_pj = dense_mac_energy_pj(cnn_macs, EnergyModel())
            baseline_source = "frame CNN (matched Conv-PLIF topology, ReLU)"
            baseline_extra["cnn_mac_ops"] = cnn_macs
            baseline_extra["cnn_channels"] = list(self.cnn_model.channels)
            baseline_extra["majority_class_accuracy"] = majority_acc
            baseline_extra["energy_ratio_cnn_over_snn"] = (
                baseline_energy_pj / float(energy["energy_pj"])
                if float(energy["energy_pj"]) > 0
                else float("inf")
            )

        return RunResult(
            benchmark=self.name,
            model={
                "conv-plif": "conv-plif",
                "sew-plif": "sew-plif",
            }.get(self.backbone, "dst-snn"),
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
                extra={
                    "epochs": self.epochs,
                    "backbone": self.backbone,
                    "fanout": energy["fanout"],
                    "dense_mac_ops": energy["dense_mac_ops"],
                    "dense_energy_pj": energy["dense_energy_pj"],
                    "energy_ratio_dense_over_snn": energy["energy_ratio_dense_over_snn"],
                    "layer_count": energy["layer_count"],
                    "decision_latency_fraction": decision_latency,
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
                    "lr": self.lr,
                    "plif_channels": list(self.plif_channels),
                    "sew_width": self.sew_width,
                    "sew_blocks": self.sew_blocks,
                },
            ),
            baseline=MetricSet(
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
            ),
            meta={
                "downsample": self.downsample,
                "smoke_from_test": self.smoke_from_test,
                "seed": self.seed,
                "backbone": self.backbone,
            },
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
    parser.add_argument("--smoke-from-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--num-branches", type=int, default=16)
    parser.add_argument("--max-delay", type=int, default=16)
    parser.add_argument("--readout", choices=["spike_count", "max_membrane", "mean_membrane"], default="max_membrane")
    parser.add_argument("--use-chrono", dest="use_chrono", action="store_true", default=True)
    parser.add_argument("--no-chrono", dest="use_chrono", action="store_false")
    parser.add_argument("--chrono-hidden", type=int, default=128)
    parser.add_argument("--hidden-features", type=int, default=0)
    parser.add_argument("--hidden-threshold", type=float, default=None)
    parser.add_argument("--hidden-output", choices=["spikes", "membrane"], default="spikes")
    parser.add_argument("--with-ann-baseline", action="store_true")
    parser.add_argument("--ann-hidden", type=int, default=128)
    parser.add_argument("--use-temporal-features", action="store_true")
    parser.add_argument("--temporal-project-to", type=int, default=128)
    parser.add_argument("--temporal-alpha", type=float, default=0.25)
    parser.add_argument("--backbone", choices=["dendritic", "conv-plif", "sew-plif"], default="dendritic")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sew-width", type=int, default=32)
    parser.add_argument("--sew-blocks", type=int, default=2)
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
        use_temporal_features=args.use_temporal_features,
        temporal_project_to=args.temporal_project_to,
        temporal_alpha=args.temporal_alpha,
        backbone=args.backbone,
        lr=args.lr,
        sew_width=args.sew_width,
        sew_blocks=args.sew_blocks,
    )
    print(run_benchmarks([runner], args.out_dir)[0].to_json())


if __name__ == "__main__":
    main()
