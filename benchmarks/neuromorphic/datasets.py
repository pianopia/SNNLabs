"""Neuromorphic dataset wrappers producing DST-SNN spike tensors via tonic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from .events import bin_events_to_frames, frames_to_spike_tensor


@dataclass(frozen=True)
class SpikeDatasetConfig:
    time_bins: int
    sensor_size: tuple[int, int]


def events_to_tensor(events, config: SpikeDatasetConfig) -> np.ndarray:
    width, height = config.sensor_size
    frames = bin_events_to_frames(
        events["x"],
        events["y"],
        events["t"],
        events["p"],
        width=width,
        height=height,
        time_bins=config.time_bins,
    )
    return frames_to_spike_tensor(frames)


class _MappedDataset(Dataset):
    def __init__(self, base, config: SpikeDatasetConfig) -> None:
        self.base = base
        self.config = config

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        events, label = self.base[index]
        tensor = events_to_tensor(events, self.config)
        return torch.from_numpy(tensor).float(), int(label)


def load_nmnist(root: str, *, time_bins: int = 24):
    import tonic

    sensor = tonic.datasets.NMNIST.sensor_size
    config = SpikeDatasetConfig(time_bins=time_bins, sensor_size=(sensor[0], sensor[1]))
    train = tonic.datasets.NMNIST(save_to=root, train=True)
    test = tonic.datasets.NMNIST(save_to=root, train=False)
    in_features = sensor[0] * sensor[1] * 2
    return _MappedDataset(train, config), _MappedDataset(test, config), in_features


def load_dvs_gesture(root: str, *, time_bins: int = 32, downsample: int = 4):
    import tonic

    sensor = tonic.datasets.DVSGesture.sensor_size
    width = sensor[0] // downsample
    height = sensor[1] // downsample
    config = SpikeDatasetConfig(time_bins=time_bins, sensor_size=(width, height))
    transform = tonic.transforms.Downsample(spatial_factor=1.0 / downsample)
    train = tonic.datasets.DVSGesture(save_to=root, train=True, transform=transform)
    test = tonic.datasets.DVSGesture(save_to=root, train=False, transform=transform)
    in_features = width * height * 2
    return _MappedDataset(train, config), _MappedDataset(test, config), in_features
