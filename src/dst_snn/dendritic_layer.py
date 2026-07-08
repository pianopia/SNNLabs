"""Core prototype for Dendritic Spatio-Temporal SNN (DST-SNN).

Input shape is ``[batch, time, in_features]`` and output spikes are
``[batch, time, out_features]``. The layer factorizes computation into:

1. spatial synapses: W maps input neurons to output neurons;
2. temporal dendrites: branch-specific delay buffers select X(t - tau_k);
3. soma: a lightweight integrate-and-fire threshold on instantaneous current.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - import-time guard for non-torch envs
    raise ImportError(
        "DST-SNN requires PyTorch. Install with `pip install torch` before running."
    ) from exc


class SurrogateSpike(torch.autograd.Function):
    """Heaviside spike with sigmoid-derivative surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane_minus_threshold: Tensor, beta: float = 10.0) -> Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.beta = beta
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        sigmoid = torch.sigmoid(beta * x)
        surrogate_grad = beta * sigmoid * (1.0 - sigmoid)
        return grad_output * surrogate_grad, None


def _round_robin_branches(in_features: int, num_branches: int, device=None) -> Tensor:
    return torch.arange(in_features, device=device, dtype=torch.long) % num_branches


class DendriticLayer(nn.Module):
    """Factorized spatial/temporal dendritic SNN layer.

    Args:
        in_features: number of input neurons.
        out_features: number of soma/output neurons.
        num_branches: number of dendritic memory groups G_k.
        max_delay: maximum delay tau in time steps.
        branch_index: optional LongTensor[in_features] assigning each input to G_k.
        learnable_delay: if true, branch delays are differentiable soft kernels over
            ``0..max_delay``. If false, integer tau values are used.
        threshold: soma threshold V_th.
        reset: reset value is kept for API clarity; soma is instantaneous IF, so
            long leaky state is intentionally not retained.
        surrogate_beta: slope for the sigmoid surrogate gradient.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_branches: int,
        max_delay: int,
        branch_index: Optional[Tensor] = None,
        learnable_delay: bool = False,
        threshold: float = 1.0,
        reset: float = 0.0,
        surrogate_beta: float = 10.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")

        self.in_features = in_features
        self.out_features = out_features
        self.num_branches = num_branches
        self.max_delay = max_delay
        self.learnable_delay = learnable_delay
        self.threshold = threshold
        self.reset = reset
        self.surrogate_beta = surrogate_beta

        weight = torch.empty(in_features, out_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

        if branch_index is None:
            branch_index = _round_robin_branches(in_features, num_branches)
        if branch_index.shape != (in_features,):
            raise ValueError("branch_index must have shape [in_features]")
        if branch_index.min().item() < 0 or branch_index.max().item() >= num_branches:
            raise ValueError("branch_index values must be in [0, num_branches)")
        self.register_buffer("branch_index", branch_index.to(dtype=torch.long))

        if learnable_delay:
            self.delay_logits = nn.Parameter(torch.zeros(num_branches, max_delay + 1))
            with torch.no_grad():
                self.delay_logits[:, 0] = 1.0
        else:
            fixed = torch.arange(num_branches, dtype=torch.long) % (max_delay + 1)
            self.register_buffer("delays", fixed)

    def _time_windows(self, x: Tensor) -> Tensor:
        # x: [B, T, I] -> windows in delay order [B, T, I, D+1],
        # where last dim index 0 means tau=0, index d means x(t-d).
        x_channels = x.transpose(1, 2)  # [B, I, T]
        padded = F.pad(x_channels, (self.max_delay, 0))
        windows = padded.unfold(dimension=2, size=self.max_delay + 1, step=1)
        return windows.flip(-1).permute(0, 2, 1, 3).contiguous()

    def delayed_inputs(self, x: Tensor) -> Tensor:
        """Return X_i(t - tau_{branch(i)}) as [B, T, I]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, in_features]")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected in_features={self.in_features}, got {x.shape[-1]}")

        windows = self._time_windows(x)
        if self.learnable_delay:
            branch_kernel = torch.softmax(self.delay_logits, dim=-1)
            input_kernel = branch_kernel[self.branch_index]  # [I, D+1]
            return torch.einsum("btid,id->bti", windows, input_kernel)

        input_delay = self.delays[self.branch_index]
        gather_index = input_delay.view(1, 1, self.in_features, 1).expand(x.shape[0], x.shape[1], -1, 1)
        return windows.gather(dim=-1, index=gather_index).squeeze(-1)

    def branch_currents(self, delayed_x: Tensor) -> Tensor:
        """Compute per-branch spatial currents [B, T, K, O]."""
        branch_one_hot = F.one_hot(self.branch_index, num_classes=self.num_branches).to(delayed_x.dtype)
        branch_mask = branch_one_hot.transpose(0, 1)  # [K, I]
        return torch.einsum("bti,ki,io->btko", delayed_x, branch_mask, self.weight)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        delayed_x = self.delayed_inputs(x)
        branch_current = self.branch_currents(delayed_x)
        membrane = branch_current.sum(dim=2) + self.bias
        spikes = SurrogateSpike.apply(membrane - self.threshold, self.surrogate_beta)
        return spikes, membrane, branch_current


class DendriticSNN(nn.Module):
    """Minimal DST-SNN classifier/regressor head built from one dendritic layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_branches: int = 8,
        max_delay: int = 12,
        learnable_delay: bool = False,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.dendrite = DendriticLayer(
            in_features=in_features,
            out_features=out_features,
            num_branches=num_branches,
            max_delay=max_delay,
            learnable_delay=learnable_delay,
            threshold=threshold,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        spikes, membrane, branch_current = self.dendrite(x)
        return {
            "spikes": spikes,
            "membrane": membrane,
            "branch_current": branch_current,
            "spike_count": spikes.sum(dim=1),
        }
