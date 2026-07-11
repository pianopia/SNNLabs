"""Teacher adapters and portable cached intermediate-feature datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


@dataclass
class TeacherBatch:
    logits: Tensor
    features: dict[str, Tensor]


class TeacherAdapter(Protocol):
    def infer(self, inputs: Tensor) -> TeacherBatch: ...


class TorchTeacherAdapter:
    """Freeze a Torch teacher and capture named module outputs with hooks."""

    def __init__(self, model: nn.Module, feature_layers: Sequence[str] = ()) -> None:
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        modules = dict(self.model.named_modules())
        missing = [name for name in feature_layers if name not in modules]
        if missing:
            raise ValueError(f"unknown teacher feature layers: {missing}")
        self.feature_layers = tuple(feature_layers)
        self._features: dict[str, Tensor] = {}
        self._hooks = [
            modules[name].register_forward_hook(self._capture(name)) for name in self.feature_layers
        ]

    def _capture(self, name: str):
        def hook(_module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, Tensor):
                raise TypeError(f"teacher feature {name} is not a Tensor")
            self._features[name] = tensor.detach()

        return hook

    @torch.no_grad()
    def infer(self, inputs: Tensor) -> TeacherBatch:
        self._features = {}
        output = self.model(inputs)
        logits = output[0] if isinstance(output, tuple) else output
        if isinstance(logits, Mapping):
            logits = logits["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("teacher output must be Tensor, tuple, or logits mapping")
        return TeacherBatch(logits.detach(), dict(self._features))

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class FeatureCacheDataset(Dataset):
    """In-memory view of a deterministic, torch-saved teacher cache."""

    def __init__(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.inputs: Tensor = payload["inputs"]
        self.targets: Tensor = payload["targets"]
        self.teacher_logits: Tensor = payload["teacher_logits"]
        self.teacher_features: dict[str, Tensor] = payload.get("teacher_features", {})
        size = int(self.inputs.shape[0])
        if any(int(item.shape[0]) != size for item in (self.targets, self.teacher_logits)):
            raise ValueError("cache tensors must have the same leading dimension")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor | dict[str, Tensor]]:
        return {
            "inputs": self.inputs[index],
            "targets": self.targets[index],
            "teacher_logits": self.teacher_logits[index],
            "teacher_features": {
                name: values[index] for name, values in self.teacher_features.items()
            },
        }


def cache_teacher_features(
    batches: Iterable[tuple[Tensor, Tensor]],
    adapter: TeacherAdapter,
    path: str | Path,
) -> Path:
    inputs_all = []
    targets_all = []
    logits_all = []
    features_all: dict[str, list[Tensor]] = {}
    for inputs, targets in batches:
        teacher = adapter.infer(inputs)
        inputs_all.append(inputs.detach().cpu())
        targets_all.append(targets.detach().cpu())
        logits_all.append(teacher.logits.detach().cpu())
        for name, values in teacher.features.items():
            features_all.setdefault(name, []).append(values.detach().cpu())
    if not inputs_all:
        raise ValueError("cannot cache an empty batch iterable")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dst_snn_teacher_cache_v1",
            "inputs": torch.cat(inputs_all),
            "targets": torch.cat(targets_all),
            "teacher_logits": torch.cat(logits_all),
            "teacher_features": {
                name: torch.cat(values) for name, values in features_all.items()
            },
        },
        destination,
    )
    return destination
