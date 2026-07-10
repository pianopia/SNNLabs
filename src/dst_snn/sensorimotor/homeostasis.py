"""Homeostatic plasticity and sleep-replay consolidation helpers.

These keep latent activity from saturating and gently re-train the world model
on high-salience recent experiences (a lightweight sleep-replay stand-in).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


@dataclass
class HomeostasisController:
    """Track per-neuron firing rates and produce threshold offsets for LIF cells.

    Positive offset → harder to fire (for neurons above ``target_rate``).
    Negative offset → easier to fire (for silent neurons). Offsets are applied
    by ``ChronoPlasticLIFCell`` / ``PredictiveWorldModel`` as
    ``V_th' = V_th + offset``.
    """

    target_rate: float = 0.05
    ema_alpha: float = 0.05
    max_offset: float = 0.5
    gain: float = 2.0
    ema_rates: Optional[Tensor] = None
    last_applied_mean_rate: float = 0.0

    def update(self, spikes: Tensor) -> dict[str, float]:
        """Update rate EMAs from ``spikes`` shaped ``[batch, time, neurons]``."""
        if spikes.ndim != 3:
            raise ValueError("spikes must have shape [batch, time, neurons]")
        rates = spikes.mean(dim=(0, 1)).detach()
        if self.ema_rates is None or self.ema_rates.shape != rates.shape:
            self.ema_rates = rates.clone()
        else:
            self.ema_rates = (1.0 - self.ema_alpha) * self.ema_rates + self.ema_alpha * rates
        self.last_applied_mean_rate = float(rates.mean().item())
        mean_rate = float(self.ema_rates.mean().item())
        deficit = float((self.target_rate - self.ema_rates).clamp(min=0).mean().item())
        excess = float((self.ema_rates - self.target_rate).clamp(min=0).mean().item())
        offsets = self.threshold_offsets()
        return {
            "mean_rate": mean_rate,
            "instant_rate": self.last_applied_mean_rate,
            "excess_rate": excess,
            "deficit_rate": deficit,
            "threshold_offset": float(offsets.mean().item()),
            "threshold_offset_max": float(offsets.max().item()),
            "threshold_offset_min": float(offsets.min().item()),
            "rate_error": float((mean_rate - self.target_rate)),
        }

    def threshold_offsets(self) -> Tensor:
        if self.ema_rates is None:
            return torch.zeros(1)
        # Neurons firing above target get a positive threshold offset (harder to fire).
        offset = (self.ema_rates - self.target_rate) * self.gain
        return offset.clamp(-self.max_offset, self.max_offset)

    def tensor_offsets(
        self,
        n_neurons: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Offsets ready to pass into ChronoPlastic as ``threshold_offset``."""
        if self.ema_rates is None:
            return torch.zeros(n_neurons, device=device, dtype=dtype or torch.float32)
        offsets = self.threshold_offsets()
        if offsets.numel() != n_neurons:
            # Resize conservatively if latent size changed (e.g. checkpoint restore).
            if offsets.numel() < n_neurons:
                pad = torch.zeros(n_neurons - offsets.numel(), dtype=offsets.dtype)
                offsets = torch.cat([offsets, pad], dim=0)
            else:
                offsets = offsets[:n_neurons]
        if device is not None or dtype is not None:
            offsets = offsets.to(device=device or offsets.device, dtype=dtype or offsets.dtype)
        return offsets

    def effective_thresholds(self, base_threshold: float = 1.0, **kwargs) -> Tensor:
        return base_threshold + self.tensor_offsets(self.ema_rates.numel() if self.ema_rates is not None else 1, **kwargs)

    def state_dict(self) -> dict:
        return {
            "target_rate": self.target_rate,
            "ema_alpha": self.ema_alpha,
            "max_offset": self.max_offset,
            "gain": self.gain,
            "ema_rates": None if self.ema_rates is None else self.ema_rates.detach().cpu().tolist(),
            "last_applied_mean_rate": self.last_applied_mean_rate,
        }

    def load_state_dict(self, state: dict) -> None:
        self.target_rate = float(state.get("target_rate", self.target_rate))
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.max_offset = float(state.get("max_offset", self.max_offset))
        self.gain = float(state.get("gain", self.gain))
        self.last_applied_mean_rate = float(state.get("last_applied_mean_rate", 0.0))
        rates = state.get("ema_rates")
        self.ema_rates = None if rates is None else torch.tensor(rates, dtype=torch.float32)


@dataclass
class ExperienceBuffer:
    """Ring buffer of recent (sensory, motor, salience) tuples for replay."""

    capacity: int = 64
    _items: Deque[tuple[Tensor, Tensor, float]] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.capacity)

    def add(self, sensory: Tensor, motor: Tensor, salience: float) -> None:
        self._items.append((sensory.detach().cpu(), motor.detach().cpu(), float(salience)))

    def __len__(self) -> int:
        return len(self._items)

    def high_salience_batch(self, k: int = 8) -> Optional[tuple[Tensor, Tensor]]:
        if not self._items:
            return None
        ranked = sorted(self._items, key=lambda item: item[2], reverse=True)
        take = ranked[: min(k, len(ranked))]
        sensory = torch.cat([item[0] for item in take], dim=0)
        motor = torch.cat([item[1] for item in take], dim=0)
        return sensory, motor


def sleep_replay(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ExperienceBuffer,
    *,
    steps: int = 4,
    batch_k: int = 8,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Re-train on high-salience memories for a few consolidation steps."""
    from .world_model import PredictiveWorldModel, train_world_model_step

    if not isinstance(model, PredictiveWorldModel):
        raise TypeError("sleep_replay expects a PredictiveWorldModel")
    if len(buffer) == 0 or steps <= 0:
        return {"replay_steps": 0.0, "replay_loss": 0.0}

    total = 0.0
    used = 0
    for _ in range(steps):
        batch = buffer.high_salience_batch(batch_k)
        if batch is None:
            break
        sensory, motor = batch
        if device is not None:
            sensory = sensory.to(device)
            motor = motor.to(device)
        metrics = train_world_model_step(
            model,
            optimizer,
            sensory,
            motor,
            homeostasis=None,  # replay consolidates weights; leave thresholds as-is
        )
        total += metrics["prediction_loss"]
        used += 1
    return {
        "replay_steps": float(used),
        "replay_loss": total / max(1, used),
    }


def representation_stability(latents: list[Tensor]) -> dict[str, float]:
    """Measure how stable latent spike codes are across consecutive steps.

    Higher cosine similarity and lower mean absolute delta ⇒ more stable codes.
    """
    if len(latents) < 2:
        return {"mean_cosine": 1.0, "mean_abs_delta": 0.0, "stability": 1.0}
    cosines: list[float] = []
    deltas: list[float] = []
    for prev, curr in zip(latents[:-1], latents[1:]):
        a = prev.detach().float().reshape(prev.shape[0], -1)
        b = curr.detach().float().reshape(curr.shape[0], -1)
        # Match batch sizes if needed by truncating.
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        a_n = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
        b_n = torch.nn.functional.normalize(b, dim=-1, eps=1e-8)
        cos = (a_n * b_n).sum(dim=-1).mean()
        delta = (a - b).abs().mean()
        cosines.append(float(cos.item()))
        deltas.append(float(delta.item()))
    mean_cos = sum(cosines) / len(cosines)
    mean_delta = sum(deltas) / len(deltas)
    stability = max(0.0, min(1.0, 0.5 * (mean_cos + 1.0) * (1.0 / (1.0 + mean_delta))))
    return {
        "mean_cosine": mean_cos,
        "mean_abs_delta": mean_delta,
        "stability": stability,
    }
