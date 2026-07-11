"""Supervised training loops for Track1 / Track2 generator heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from src.dst_snn.threedcg.dataset import ID_TO_SHAPE, make_batch
from src.dst_snn.threedcg.sdf import Track2SdfHead, mesh_to_sdf
from src.dst_snn.threedcg.sequence import (
    SEQ_LEN,
    Track1SequenceHead,
    program_to_ids,
    template_program,
)
from src.dst_snn.threedcg.track1_policy import Track1OpHead
from src.dst_snn.threedcg.track2_occupancy import Track2OccupancyHead


@dataclass
class TrainResult:
    track: str
    epochs: int
    final_loss: float
    history: list[float]
    checkpoint: str
    in_features: int
    extra: dict[str, Any]


def _mean_pool(spikes: np.ndarray, in_features: int) -> "object":
    import torch

    x = torch.as_tensor(np.asarray(spikes, dtype=np.float32).mean(axis=0), dtype=torch.float32)
    if x.numel() != in_features:
        buf = torch.zeros(in_features, dtype=torch.float32)
        n = min(in_features, int(x.numel()))
        buf[:n] = x.reshape(-1)[:n]
        return buf
    return x


def train_track1(
    *,
    epochs: int = 40,
    n_samples: int = 96,
    seed: int = 0,
    lr: float = 1e-2,
    time_bins: int = 8,
    image_size: int = 32,
    out_path: Path | str = "artifacts/threedcg/checkpoints/track1.pt",
) -> TrainResult:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install PyTorch to train Track1.") from exc

    samples = make_batch(n_samples, seed=seed, time_bins=time_bins, image_size=image_size)
    in_features = int(samples[0].spikes.shape[1])
    head = Track1OpHead(in_features, seed=seed)
    optimizer = torch.optim.Adam(head.net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history: list[float] = []

    for _ in range(epochs):
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        # mini full-batch for simplicity / CPU friendliness
        class_logits = []
        extent_preds = []
        class_targets = []
        extent_targets = []
        for s in samples:
            x = _mean_pool(s.spikes, in_features)
            out = head.net(x)
            class_logits.append(out[:3])
            extent_preds.append(torch.nn.functional.softplus(out[3:6]) + 0.2)
            class_targets.append(s.shape_id)
            extent_targets.append(torch.tensor(s.extents, dtype=torch.float32))
        logits = torch.stack(class_logits)
        targets = torch.tensor(class_targets, dtype=torch.long)
        pred_e = torch.stack(extent_preds)
        true_e = torch.stack(extent_targets)
        loss = ce(logits, targets) + 0.5 * nn.functional.mse_loss(pred_e, true_e)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))
        total = history[-1]
        del total

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "track": "track1",
        "in_features": in_features,
        "state_dict": head.net.state_dict(),
        "history": history,
        "meta": {"epochs": epochs, "n_samples": n_samples, "seed": seed},
    }
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "track": "track1",
                "in_features": in_features,
                "final_loss": history[-1] if history else None,
                "epochs": epochs,
                "checkpoint": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        track="track1",
        epochs=epochs,
        final_loss=history[-1] if history else float("inf"),
        history=history,
        checkpoint=str(path),
        in_features=in_features,
        extra={"first_loss": history[0] if history else None},
    )


def train_track2(
    *,
    epochs: int = 50,
    n_samples: int = 64,
    seed: int = 0,
    lr: float = 1e-2,
    time_bins: int = 8,
    resolution: int = 6,
    image_size: int = 32,
    out_path: Path | str = "artifacts/threedcg/checkpoints/track2.pt",
) -> TrainResult:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install PyTorch to train Track2.") from exc

    samples = make_batch(
        n_samples, seed=seed, time_bins=time_bins, resolution=resolution, image_size=image_size
    )
    in_features = int(samples[0].spikes.shape[1])
    head = Track2OccupancyHead(in_features, resolution=resolution, seed=seed)
    optimizer = torch.optim.Adam(head.net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    history: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for s in samples:
            x = _mean_pool(s.spikes, in_features)
            logits = head.forward_logits(x)
            target = torch.as_tensor(s.occupancy.reshape(-1), dtype=torch.float32)
            losses.append(bce(logits, target))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "track": "track2",
            "in_features": in_features,
            "resolution": resolution,
            "state_dict": head.net.state_dict(),
            "history": history,
            "meta": {"epochs": epochs, "n_samples": n_samples, "seed": seed},
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "track": "track2",
                "in_features": in_features,
                "resolution": resolution,
                "final_loss": history[-1] if history else None,
                "epochs": epochs,
                "checkpoint": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        track="track2",
        epochs=epochs,
        final_loss=history[-1] if history else float("inf"),
        history=history,
        checkpoint=str(path),
        in_features=in_features,
        extra={"resolution": resolution, "first_loss": history[0] if history else None},
    )


def load_track1_head(path: Path | str) -> Track1OpHead:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    head = Track1OpHead(int(data["in_features"]), seed=0)
    head.net.load_state_dict(data["state_dict"])
    head.net.eval()
    return head


def load_track2_head(path: Path | str) -> Track2OccupancyHead:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    head = Track2OccupancyHead(
        int(data["in_features"]),
        resolution=int(data.get("resolution", 8)),
        seed=0,
    )
    head.net.load_state_dict(data["state_dict"])
    head.net.eval()
    return head


def train_track2_sdf(
    *,
    epochs: int = 40,
    n_samples: int = 48,
    seed: int = 0,
    lr: float = 1e-2,
    time_bins: int = 8,
    resolution: int = 8,
    image_size: int = 32,
    out_path: Path | str = "artifacts/threedcg/checkpoints/track2_sdf.pt",
) -> TrainResult:
    try:
        import torch
        from torch import nn
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install PyTorch to train Track2 SDF.") from exc

    samples = make_batch(
        n_samples, seed=seed, time_bins=time_bins, resolution=resolution, image_size=image_size
    )
    in_features = int(samples[0].spikes.shape[1])
    head = Track2SdfHead(in_features, resolution=resolution, seed=seed)
    optimizer = torch.optim.Adam(head.net.parameters(), lr=lr)
    history: list[float] = []

    # Precompute SDF targets (slow-ish but n small)
    targets = []
    for s in samples:
        mesh = trimesh.Trimesh(vertices=s.asset.vertices, faces=s.asset.faces, process=False)
        sdf, _, _ = mesh_to_sdf(mesh, resolution=resolution)
        # normalize by grid extent scale for stable MSE
        scale = max(float(np.std(sdf)), 1e-3)
        targets.append(sdf.reshape(-1) / scale)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for s, tgt in zip(samples, targets):
            x = _mean_pool(s.spikes, in_features)
            pred = head.forward_sdf(x)
            t = torch.as_tensor(tgt, dtype=torch.float32)
            losses.append(nn.functional.mse_loss(pred, t))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "track": "track2_sdf",
            "in_features": in_features,
            "resolution": resolution,
            "state_dict": head.net.state_dict(),
            "history": history,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "track": "track2_sdf",
                "final_loss": history[-1] if history else None,
                "epochs": epochs,
                "checkpoint": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        track="track2_sdf",
        epochs=epochs,
        final_loss=history[-1] if history else float("inf"),
        history=history,
        checkpoint=str(path),
        in_features=in_features,
        extra={"resolution": resolution, "first_loss": history[0] if history else None},
    )


def load_track2_sdf_head(path: Path | str) -> Track2SdfHead:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    head = Track2SdfHead(int(data["in_features"]), resolution=int(data.get("resolution", 8)), seed=0)
    head.net.load_state_dict(data["state_dict"])
    head.net.eval()
    return head


def train_track1_sequence(
    *,
    epochs: int = 40,
    n_samples: int = 64,
    seed: int = 0,
    lr: float = 1e-2,
    time_bins: int = 8,
    image_size: int = 32,
    out_path: Path | str = "artifacts/threedcg/checkpoints/track1_seq.pt",
) -> TrainResult:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install PyTorch to train sequence head.") from exc

    samples = make_batch(n_samples, seed=seed, time_bins=time_bins, image_size=image_size)
    in_features = int(samples[0].spikes.shape[1])
    head = Track1SequenceHead(in_features, seq_len=SEQ_LEN, seed=seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history: list[float] = []

    labels = []
    for s in samples:
        shape = ID_TO_SHAPE[s.shape_id]
        prog = template_program(shape, s.extents)
        labels.append(program_to_ids(prog, seq_len=SEQ_LEN))

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for s, lab in zip(samples, labels):
            x = _mean_pool(s.spikes, in_features)
            op_logits, extents = head.forward(x)
            target_ids = torch.as_tensor(lab, dtype=torch.long)
            # CE per step
            step_loss = ce(op_logits, target_ids)
            ext_t = torch.tensor(s.extents, dtype=torch.float32)
            ext_loss = nn.functional.mse_loss(extents, ext_t)
            losses.append(step_loss + 0.4 * ext_loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "track": "track1_seq",
            "in_features": in_features,
            "seq_len": SEQ_LEN,
            "state_dict": head.state_dict(),
            "history": history,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "track": "track1_seq",
                "final_loss": history[-1] if history else None,
                "epochs": epochs,
                "checkpoint": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        track="track1_seq",
        epochs=epochs,
        final_loss=history[-1] if history else float("inf"),
        history=history,
        checkpoint=str(path),
        in_features=in_features,
        extra={"first_loss": history[0] if history else None, "seq_len": SEQ_LEN},
    )


def load_track1_sequence_head(path: Path | str) -> Track1SequenceHead:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    head = Track1SequenceHead(int(data["in_features"]), seq_len=int(data.get("seq_len", SEQ_LEN)), seed=0)
    head.load_state_dict(data["state_dict"])
    head.backbone.eval()
    head.op_head.eval()
    head.extent_head.eval()
    return head
