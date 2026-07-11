"""Trainable signed-integer spiking state-space blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _SignedIntegerSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized: Tensor, max_level: int, surrogate_width: float) -> Tensor:
        ctx.save_for_backward(normalized)
        ctx.max_level = int(max_level)
        ctx.surrogate_width = float(surrogate_width)
        return torch.clamp(torch.round(normalized), -max_level, max_level)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (normalized,) = ctx.saved_tensors
        # Smooth straight-through estimator. Gradients fade outside the valid
        # integer range rather than disappearing at every rounding boundary.
        width = max(ctx.surrogate_width, 1e-6)
        distance = (normalized.abs() - float(ctx.max_level)).clamp_min(0.0)
        surrogate = torch.exp(-distance / width)
        return grad_output * surrogate, None, None


def signed_integer_spike(
    normalized: Tensor,
    *,
    max_level: int = 3,
    surrogate_width: float = 1.0,
) -> Tensor:
    if max_level < 1:
        raise ValueError("max_level must be positive")
    return _SignedIntegerSpike.apply(normalized, int(max_level), float(surrogate_width))


@dataclass
class SpikingSSMOutput:
    hidden: Tensor
    events: Tensor
    membrane: Tensor
    final_state: Tensor

    @property
    def spike_rate(self) -> Tensor:
        return (self.events != 0).to(dtype=self.hidden.dtype).mean()


class SignedSpikingSSMBlock(nn.Module):
    """Diagonal SSM recurrence with signed multi-level event activations.

    Input/output shape is ``[batch, steps, dim]``. Only ``[batch, state_dim]``
    recurrent state is required at inference, independent of context length.
    """

    def __init__(
        self,
        dim: int,
        *,
        state_dim: int | None = None,
        max_level: int = 3,
        min_threshold: float = 0.05,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.state_dim = int(state_dim or dim)
        self.max_level = int(max_level)
        self.min_threshold = float(min_threshold)
        self.norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, self.state_dim)
        self.output_projection = nn.Linear(self.state_dim, dim)
        self.decay_logit = nn.Parameter(torch.full((self.state_dim,), 2.0))
        self.threshold_raw = nn.Parameter(torch.zeros(self.state_dim))
        self.gate = nn.Linear(dim, self.state_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.threshold_raw) + self.min_threshold

    def forward(self, x: Tensor, state: Tensor | None = None) -> SpikingSSMOutput:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"expected [batch, steps, {self.dim}], got {tuple(x.shape)}")
        batch, steps, _ = x.shape
        membrane = (
            x.new_zeros((batch, self.state_dim)) if state is None else state.to(x)
        )
        if membrane.shape != (batch, self.state_dim):
            raise ValueError(f"state must have shape {(batch, self.state_dim)}")
        decay = torch.sigmoid(self.decay_logit)
        threshold = self.threshold
        event_steps = []
        membrane_steps = []
        normalized_input = self.norm(x)
        for step in range(steps):
            current = normalized_input[:, step]
            update = self.input_projection(current)
            input_gate = torch.sigmoid(self.gate(current))
            membrane = decay * membrane + input_gate * update
            events = signed_integer_spike(membrane / threshold, max_level=self.max_level)
            membrane = membrane - events * threshold
            event_steps.append(events)
            membrane_steps.append(membrane)
        event_tensor = torch.stack(event_steps, dim=1)
        membrane_tensor = torch.stack(membrane_steps, dim=1)
        hidden = x + self.dropout(self.output_projection(event_tensor))
        return SpikingSSMOutput(hidden, event_tensor, membrane_tensor, membrane)


class SpikingSSMBackbone(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        state_dim: int | None = None,
        max_level: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.dim = int(dim)
        self.blocks = nn.ModuleList(
            [
                SignedSpikingSSMBlock(
                    dim,
                    state_dim=state_dim,
                    max_level=max_level,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        states: list[Tensor | None] | None = None,
    ) -> tuple[Tensor, list[SpikingSSMOutput]]:
        layer_states = states or [None] * len(self.blocks)
        if len(layer_states) != len(self.blocks):
            raise ValueError("states length must match depth")
        outputs = []
        hidden = x
        for block, state in zip(self.blocks, layer_states):
            output = block(hidden, state)
            outputs.append(output)
            hidden = output.hidden
        return self.final_norm(hidden), outputs
