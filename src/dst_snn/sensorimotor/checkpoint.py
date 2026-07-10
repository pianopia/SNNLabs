"""Checkpoint helpers for predictive sensorimotor world models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from .world_model import LearningProgress, PredictiveWorldModel


def save_world_model_checkpoint(
    path: Path,
    model: PredictiveWorldModel,
    optimizer: torch.optim.Optimizer | None = None,
    progress: LearningProgress | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {
            "sensory_size": model.sensory_size,
            "motor_size": model.motor_size,
            "latent_size": model.latent_size,
        },
        "model": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if progress is not None:
        payload["progress"] = {"ema_loss": progress.ema_loss, "alpha": progress.alpha}
    torch.save(payload, path)


def load_world_model_checkpoint(
    path: Path,
    *,
    device: str = "cpu",
    with_optimizer: bool = False,
    lr: float = 1e-3,
) -> tuple[PredictiveWorldModel, torch.optim.Optimizer | None, LearningProgress | None, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    config = payload["config"]
    model = PredictiveWorldModel(
        sensory_size=int(config["sensory_size"]),
        motor_size=int(config["motor_size"]),
        latent_size=int(config["latent_size"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    optimizer = None
    if with_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
    progress = None
    if "progress" in payload:
        state = payload["progress"]
        progress = LearningProgress(
            ema_loss=state.get("ema_loss"),
            alpha=float(state.get("alpha", 0.1)),
        )
    return model, optimizer, progress, dict(payload.get("extra", {}))
