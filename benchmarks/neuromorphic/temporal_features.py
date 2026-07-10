"""Temporal feature front-ends for event spike tensors.

These operate on ``[batch, time, features]`` binary (or rate) tensors and
produce richer temporal codes without requiring network access in tests.
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


def causal_ema(x: Tensor, alpha: float = 0.25) -> Tensor:
    """Causal exponential moving average over the time axis."""
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, features]")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    out = torch.zeros_like(x)
    state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
    for t in range(x.shape[1]):
        state = (1.0 - alpha) * state + alpha * x[:, t]
        out[:, t] = state
    return out


def temporal_difference(x: Tensor) -> Tensor:
    """First-order difference along time (prepend zeros)."""
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, features]")
    diff = torch.zeros_like(x)
    diff[:, 1:] = x[:, 1:] - x[:, :-1]
    return diff


class TemporalFeatureFrontEnd(nn.Module):
    """Stack raw spikes with EMA rate and temporal difference channels.

    Output features = ``3 * in_features`` unless ``project_to`` is set, in which
    case a learned linear map compresses the stacked code.
    """

    def __init__(
        self,
        in_features: int,
        *,
        alpha: float = 0.25,
        project_to: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha = alpha
        self.project_to = project_to
        stacked = in_features * 3
        if project_to > 0:
            self.proj: nn.Module | None = nn.Linear(stacked, project_to)
            self.out_features = project_to
        else:
            self.proj = None
            self.out_features = stacked

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, features]")
        rate = causal_ema(x, alpha=self.alpha)
        delta = temporal_difference(x)
        stacked = torch.cat([x, rate, delta], dim=-1)
        if self.proj is None:
            return stacked
        # Apply linear map at each time step.
        b, t, f = stacked.shape
        return self.proj(stacked.reshape(b * t, f)).reshape(b, t, self.out_features)
