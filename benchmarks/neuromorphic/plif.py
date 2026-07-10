"""Parametric LIF (PLIF) neuron with arctan surrogate gradients.

PLIF (Fang et al., 2021) makes the membrane time-constant learnable, which is a
standard building block in modern directly trained SNNs for event vision
(SEW-ResNet / SpikingJelly tutorials). Implemented without external SNN deps.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


class ArctanSurrogate(torch.autograd.Function):
    """Heaviside forward, arctan-shaped surrogate backward (Fang et al.)."""

    @staticmethod
    def forward(ctx, membrane_minus_threshold: Tensor, alpha: float = 2.0) -> Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.alpha = float(alpha)
        return (membrane_minus_threshold >= 0).to(dtype=membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # d/dx arctan(alpha * x) / pi  ≈ alpha / (pi * (1 + (alpha x)^2))
        grad = grad_output * (alpha / math.pi) / (1.0 + (alpha * x).pow(2))
        return grad, None


def spike_fn(membrane_minus_threshold: Tensor, alpha: float = 2.0) -> Tensor:
    return ArctanSurrogate.apply(membrane_minus_threshold, alpha)


class PLIF(nn.Module):
    """Parametric leaky integrate-and-fire with soft/hard reset.

    Decay uses learnable ``tau = 1 + exp(w)`` so the effective leak stays in (0,1).
    """

    def __init__(
        self,
        *,
        init_tau: float = 2.0,
        threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_alpha: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        if init_tau <= 1.0:
            raise ValueError("init_tau must be > 1")
        # tau = 1 + exp(w) ⇒ w = log(tau - 1)
        self.w = nn.Parameter(torch.tensor(math.log(init_tau - 1.0)))
        self.threshold = float(threshold)
        self.v_reset = float(v_reset)
        self.surrogate_alpha = float(surrogate_alpha)
        self.detach_reset = bool(detach_reset)

    @property
    def tau(self) -> Tensor:
        return 1.0 + torch.exp(self.w)

    def forward(self, x: Tensor, v: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Integrate ``x`` into membrane ``v`` and emit spikes.

        Shapes of ``x``/``v`` are arbitrary as long as they match (broadcast-free).
        """
        if v is None:
            v = torch.zeros_like(x)
        # V ← V + (X − V) / tau  ≡  (1 − 1/tau) V + X/tau
        decay = 1.0 / self.tau
        h = v + (x - v) * decay
        s = spike_fn(h - self.threshold, self.surrogate_alpha)
        # Soft reset toward v_reset; optional detach of spike for reset path.
        s_reset = s.detach() if self.detach_reset else s
        v_next = h * (1.0 - s_reset) + self.v_reset * s_reset
        return s, v_next
