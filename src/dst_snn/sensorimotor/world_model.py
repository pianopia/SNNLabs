"""Predictive SNN world model for sensorimotor observations."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from src.dst_snn import ChronoPlasticLIFLayer


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
    """Predict the next sensory spike vector from current sensory and motor state."""

    def __init__(self, sensory_size: int, motor_size: int, latent_size: int = 64) -> None:
        super().__init__()
        self.sensory_size = sensory_size
        self.motor_size = motor_size
        self.latent_size = latent_size
        self.encoder = ChronoPlasticLIFLayer(sensory_size + motor_size, latent_size)
        self.predictor = nn.Linear(latent_size, sensory_size)
        self.motor_head = nn.Linear(latent_size, motor_size)

    def forward(self, sensory: Tensor, motor: Tensor | None = None) -> dict[str, Tensor]:
        if sensory.ndim != 3:
            raise ValueError("sensory must have shape [batch, time, sensory_size]")
        if motor is None:
            motor = torch.zeros(sensory.shape[0], sensory.shape[1], self.motor_size, device=sensory.device)
        if motor.shape[:2] != sensory.shape[:2] or motor.shape[-1] != self.motor_size:
            raise ValueError("motor must have shape [batch, time, motor_size]")
        encoded = self.encoder(torch.cat([sensory, motor], dim=-1))
        spikes = encoded["spikes"]
        prediction = self.predictor(spikes)
        motor_logits = self.motor_head(spikes)
        return {"prediction": prediction, "motor_logits": motor_logits, "spikes": spikes}

    def prediction_loss(self, sensory: Tensor, motor: Tensor | None = None) -> Tensor:
        out = self.forward(sensory, motor)
        target = torch.zeros_like(sensory)
        target[:, :-1] = sensory[:, 1:]
        return F.binary_cross_entropy_with_logits(out["prediction"], target)


def train_world_model_step(
    model: PredictiveWorldModel,
    optimizer: torch.optim.Optimizer,
    sensory: Tensor,
    motor: Tensor | None = None,
    progress: LearningProgress | None = None,
) -> dict[str, float]:
    loss = model.prediction_loss(sensory, motor)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    metrics = {"prediction_loss": loss_value}
    if progress is not None:
        metrics.update(progress.update(loss_value))
    return metrics
