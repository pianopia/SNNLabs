"""Predictive SNN world model for sensorimotor observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from src.dst_snn import ChronoPlasticLIFLayer

from .homeostasis import HomeostasisController


@dataclass
class LearningProgress:
    """EMA-based intrinsic reward from prediction-loss improvement."""

    ema_loss: float | None = None
    alpha: float = 0.1

    def update(self, loss: float) -> dict[str, float]:
        if self.ema_loss is None:
            self.ema_loss = float(loss)
            return {"ema_loss": self.ema_loss, "learning_progress": 0.0, "intrinsic_reward": 0.0}
        previous = self.ema_loss
        self.ema_loss = (1.0 - self.alpha) * self.ema_loss + self.alpha * float(loss)
        progress = max(0.0, previous - self.ema_loss)
        reward = max(0.0, min(1.0, progress / max(1e-6, previous)))
        return {"ema_loss": self.ema_loss, "learning_progress": progress, "intrinsic_reward": reward}


class PredictiveWorldModel(nn.Module):
    """Predict the next sensory spike vector from current sensory and motor state.

    Homeostatic threshold offsets (from ``HomeostasisController``) are applied
    inside the ChronoPlastic encoder so firing rates self-regulate without
    changing synaptic weights.
    """

    def __init__(
        self,
        sensory_size: int,
        motor_size: int,
        latent_size: int = 64,
        *,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.sensory_size = sensory_size
        self.motor_size = motor_size
        self.latent_size = latent_size
        self.base_threshold = float(threshold)
        self.encoder = ChronoPlasticLIFLayer(
            sensory_size + motor_size,
            latent_size,
            threshold=threshold,
        )
        self.predictor = nn.Linear(latent_size, sensory_size)
        self.motor_head = nn.Linear(latent_size, motor_size)

    def forward(
        self,
        sensory: Tensor,
        motor: Tensor | None = None,
        *,
        threshold_offset: Tensor | None = None,
        homeostasis: HomeostasisController | None = None,
        update_homeostasis: bool = False,
    ) -> dict[str, Tensor]:
        if sensory.ndim != 3:
            raise ValueError("sensory must have shape [batch, time, sensory_size]")
        if motor is None:
            motor = torch.zeros(sensory.shape[0], sensory.shape[1], self.motor_size, device=sensory.device)
        if motor.shape[:2] != sensory.shape[:2] or motor.shape[-1] != self.motor_size:
            raise ValueError("motor must have shape [batch, time, motor_size]")

        offset = threshold_offset
        if offset is None and homeostasis is not None:
            offset = homeostasis.tensor_offsets(
                self.latent_size,
                device=sensory.device,
                dtype=sensory.dtype,
            )

        encoded = self.encoder(
            torch.cat([sensory, motor], dim=-1),
            threshold_offset=offset,
        )
        spikes = encoded["spikes"]
        prediction = self.predictor(spikes)
        motor_logits = self.motor_head(spikes)
        if update_homeostasis and homeostasis is not None:
            with torch.no_grad():
                homeostasis.update(spikes.detach())
        out: dict[str, Tensor] = {
            "prediction": prediction,
            "motor_logits": motor_logits,
            "spikes": spikes,
            "membrane": encoded["membrane"],
        }
        if offset is not None:
            out["threshold_offset"] = offset
        return out

    def prediction_loss(
        self,
        sensory: Tensor,
        motor: Tensor | None = None,
        *,
        threshold_offset: Tensor | None = None,
        homeostasis: HomeostasisController | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        out = self.forward(
            sensory,
            motor,
            threshold_offset=threshold_offset,
            homeostasis=homeostasis,
            update_homeostasis=False,
        )
        target = torch.zeros_like(sensory)
        target[:, :-1] = sensory[:, 1:]
        loss = F.binary_cross_entropy_with_logits(out["prediction"], target)
        return loss, out


def train_world_model_step(
    model: PredictiveWorldModel,
    optimizer: torch.optim.Optimizer,
    sensory: Tensor,
    motor: Tensor | None = None,
    progress: LearningProgress | None = None,
    *,
    homeostasis: HomeostasisController | None = None,
    threshold_offset: Tensor | None = None,
) -> dict[str, float]:
    """One supervised predictive step, optionally with homeostatic thresholds.

    Order:
    1. Build offsets from the controller's *previous* rates (or explicit tensor).
    2. Forward + loss with those offsets applied inside ChronoPlastic.
    3. Optimizer step.
    4. Update the controller from the new spike rates.
    """
    loss, out = model.prediction_loss(
        sensory,
        motor,
        threshold_offset=threshold_offset,
        homeostasis=homeostasis,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    metrics: dict[str, float] = {
        "prediction_loss": loss_value,
        "latent_spike_rate": float(out["spikes"].detach().mean().item()),
    }
    if progress is not None:
        metrics.update(progress.update(loss_value))
    if homeostasis is not None:
        with torch.no_grad():
            homeo_stats = homeostasis.update(out["spikes"].detach())
        metrics.update({f"homeo_{k}": float(v) for k, v in homeo_stats.items()})
    if "threshold_offset" in out and isinstance(out["threshold_offset"], Tensor):
        metrics["applied_offset_mean"] = float(out["threshold_offset"].detach().mean().item())
    return metrics
