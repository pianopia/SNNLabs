"""Dense ANN next-step predictor baseline for sensorimotor world models.

Matches the PredictiveWorldModel task: map [sensory, motor] over time to a
next-step sensory prediction with BCE-with-logits loss. Used as a quality /
latency / energy reference on the same closed-loop stream — not as an SOTA
sequence model.
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


class DenseAnnPredictor(nn.Module):
    """Per-timestep MLP: concat(sensory, motor) → sensory logits."""

    def __init__(
        self,
        sensory_size: int,
        motor_size: int,
        *,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.sensory_size = int(sensory_size)
        self.motor_size = int(motor_size)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.sensory_size + self.motor_size, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.sensory_size),
        )

    def forward(self, sensory: Tensor, motor: Tensor) -> Tensor:
        if sensory.ndim != 3 or motor.ndim != 3:
            raise ValueError("sensory and motor must be [batch, time, features]")
        if sensory.shape[:2] != motor.shape[:2]:
            raise ValueError("sensory and motor batch/time dims must match")
        x = torch.cat([sensory, motor], dim=-1)
        return self.net(x)

    def prediction_loss(self, sensory: Tensor, motor: Tensor) -> Tensor:
        pred = self.forward(sensory, motor)
        target = torch.zeros_like(sensory)
        target[:, :-1] = sensory[:, 1:]
        return F.binary_cross_entropy_with_logits(pred, target)

    def mac_ops_per_inference(self, time_steps: int) -> float:
        """Dense MAC ops for one sample over ``time_steps`` (both linear layers)."""
        t = float(time_steps)
        in_dim = float(self.sensory_size + self.motor_size)
        first = in_dim * float(self.hidden) * t
        second = float(self.hidden) * float(self.sensory_size) * t
        return first + second


def train_ann_predictor_step(
    model: DenseAnnPredictor,
    optimizer: torch.optim.Optimizer,
    sensory: Tensor,
    motor: Tensor,
) -> float:
    loss = model.prediction_loss(sensory, motor)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
