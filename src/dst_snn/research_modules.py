"""Composable 2025-2026 SNN research modules.

The classes in this module are PyTorch-first building blocks that turn recent
SNN paper concepts into reusable components:

* Threshold Guarding Optimization (TGO): membrane-threshold margin loss and
  noisy probabilistic spiking during training.
* Max-Former / high-frequency Spiking Transformer: max-pool high-pass token
  mixing and depth-wise convolution in place of low-pass token mixing.
* ChronoPlastic SNNs (CPSNNs): dynamic leak modulation conditioned on local
  membrane, spike, and temporal trace state.
* CaRe-BN: confidence-adaptive BN statistics and exact recalibration.

The implementation intentionally stays dependency-light and uses only PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dendritic_layer import SurrogateSpike


def _as_threshold(threshold: float | Tensor, like: Tensor) -> Tensor:
    return torch.as_tensor(threshold, dtype=like.dtype, device=like.device)


class ThresholdGuardingLoss(nn.Module):
    """TGO membrane-margin regularizer.

    Paper concept: Threshold Guarding Optimization constrains neurons so their
    membrane potentials move away from the firing threshold. The margin reduces
    threshold-neighboring neurons that are prone to adversarial state flips.

    Args:
        margin: safe distance around the threshold.
        weight: multiplier applied to the returned penalty.
        power: 1 for hinge, 2 for squared hinge.
        reduction: "mean", "sum", or "none".
    """

    def __init__(
        self,
        margin: float = 0.2,
        weight: float = 1.0,
        power: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.margin = float(margin)
        self.weight = float(weight)
        self.power = float(power)
        self.reduction = reduction

    def forward(
        self,
        membrane: Tensor | Sequence[Tensor],
        threshold: float | Tensor = 1.0,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if isinstance(membrane, (list, tuple)):
            losses = [self.forward(item, threshold=threshold, mask=mask) for item in membrane if item is not None]
            if not losses:
                raise ValueError("membrane sequence must contain at least one tensor")
            return torch.stack(losses).mean()

        threshold_tensor = _as_threshold(threshold, membrane)
        distance = (membrane - threshold_tensor).abs()
        penalty = F.relu(self.margin - distance)
        if self.power != 1.0:
            penalty = penalty.pow(self.power)
        if mask is not None:
            penalty = penalty * mask.to(dtype=penalty.dtype, device=penalty.device)

        if self.reduction == "sum":
            return penalty.sum() * self.weight
        if self.reduction == "none":
            return penalty * self.weight
        if mask is not None:
            denom = mask.to(dtype=penalty.dtype, device=penalty.device).sum().clamp_min(1.0)
            return penalty.sum() / denom * self.weight
        return penalty.mean() * self.weight


class NoisySpikingActivation(nn.Module):
    """Surrogate spiking activation with TGO-style training noise.

    Paper concept: TGO transitions deterministic thresholding into probabilistic
    spiking during training by injecting noise around the membrane-threshold
    boundary, reducing state-flipping sensitivity to small perturbations.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        noise_std: float = 0.05,
        surrogate_beta: float = 10.0,
        training_only: bool = True,
        return_noisy_membrane: bool = False,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.noise_std = float(noise_std)
        self.surrogate_beta = float(surrogate_beta)
        self.training_only = bool(training_only)
        self.return_noisy_membrane = bool(return_noisy_membrane)

    def forward(
        self,
        membrane: Tensor,
        threshold: float | Tensor | None = None,
        noise_std: Optional[float] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        threshold_tensor = _as_threshold(self.threshold if threshold is None else threshold, membrane)
        delta = membrane - threshold_tensor
        std = self.noise_std if noise_std is None else float(noise_std)
        if std > 0 and (self.training or not self.training_only):
            delta = delta + torch.randn_like(delta) * std
        spikes = SurrogateSpike.apply(delta, self.surrogate_beta)
        if self.return_noisy_membrane:
            return spikes, delta + threshold_tensor
        return spikes


class MaxPoolPatchEmbed(nn.Module):
    """Patch embedding with an explicit Max-Pool high-frequency pre-path.

    Paper concept: Max-Former restores high-frequency information by adding
    Max-Pool in patch embedding. The module accepts either images
    ``[B, C, H, W]`` or spike/video tensors ``[B, T, C, H, W]`` and returns
    token sequences ``[B, T, N, D]``.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 4,
        high_frequency_pool: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.high_frequency_pool = high_frequency_pool
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        has_time = x.ndim == 5
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError("x must have shape [B,C,H,W] or [B,T,C,H,W]")
        batch, time, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")

        x2d = x.reshape(batch * time, channels, height, width)
        if self.high_frequency_pool:
            x2d = self.max_pool(x2d)
        patches = self.proj(x2d)
        grid = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, time, tokens.shape[1], self.embed_dim)
        if not has_time:
            tokens = tokens[:, :1]
        return tokens, grid


class HighFrequencyTokenMixer(nn.Module):
    """High-frequency token mixer for Spiking Transformers.

    Paper concept: Spiking Transformers Need High-Frequency Information /
    Max-Former replaces low-pass token mixing such as Avg-Pooling with
    high-frequency preserving operators. This mixer uses:

    1. Max-Pool minus Avg-Pool as an explicit local high-pass component.
    2. Depth-wise convolution as a lightweight early-stage attention substitute.

    Input and output shape: ``[B, T, N, C]``.
    """

    def __init__(
        self,
        dim: int,
        grid_size: Optional[tuple[int, int]] = None,
        kernel_size: int = 3,
        highpass_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.dim = dim
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(dim)
        self.highpass_scale = nn.Parameter(torch.tensor(float(highpass_scale)))
        self.depthwise = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def _grid(self, tokens: int, grid_size: Optional[tuple[int, int]]) -> tuple[int, int]:
        grid = grid_size or self.grid_size
        if grid is not None:
            if grid[0] * grid[1] != tokens:
                raise ValueError(f"grid_size {grid} does not match token count {tokens}")
            return grid
        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError("grid_size is required when token count is not a square")
        return side, side

    def forward(self, x: Tensor, grid_size: Optional[tuple[int, int]] = None) -> Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,C]")
        batch, time, tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {channels}")
        height, width = self._grid(tokens, grid_size)

        y = self.norm(x).reshape(batch * time, tokens, channels).transpose(1, 2)
        y = y.reshape(batch * time, channels, height, width)
        max_path = F.max_pool2d(y, kernel_size=3, stride=1, padding=1)
        avg_path = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        high_frequency = max_path - avg_path
        mixed = y + self.highpass_scale * high_frequency
        mixed = self.pointwise(self.depthwise(mixed))
        mixed = mixed.flatten(2).transpose(1, 2).reshape(batch, time, tokens, channels)
        return x + self.dropout(mixed)


@dataclass
class ChronoPlasticState:
    """State container for ChronoPlastic LIF dynamics."""

    membrane: Tensor
    spike: Tensor
    traces: Tensor


class ChronoPlasticLIFCell(nn.Module):
    """LIF cell with state-conditioned dynamic leak.

    Paper concept: ChronoPlastic SNNs dynamically modulate synaptic decay rates
    from internal network state rather than using fixed membrane/synaptic time
    constants. This cell keeps multiple local temporal traces and learns a
    continuous time-warping gate that decides when to preserve or forget state.
    """

    def __init__(
        self,
        features: int,
        num_traces: int = 3,
        min_leak: float = 0.05,
        max_leak: float = 0.98,
        threshold: float = 1.0,
        reset: float = 0.0,
        noise_std: float = 0.03,
        surrogate_beta: float = 10.0,
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        if num_traces <= 0:
            raise ValueError("num_traces must be positive")
        if not 0 <= min_leak < max_leak < 1:
            raise ValueError("require 0 <= min_leak < max_leak < 1")
        self.features = features
        self.num_traces = num_traces
        self.min_leak = min_leak
        self.max_leak = max_leak
        self.threshold = threshold
        self.reset = reset
        self.spike_fn = NoisySpikingActivation(threshold, noise_std, surrogate_beta)

        base = torch.linspace(min_leak, max_leak, steps=num_traces)
        self.trace_decay_logits = nn.Parameter(torch.logit(base.clamp(1e-4, 1 - 1e-4)))
        self.trace_readout = nn.Linear(num_traces, 1)
        self.leak_gate = nn.Sequential(
            nn.Linear(features * 4, features),
            nn.SiLU(),
            nn.Linear(features, features),
        )
        self.input_scale = nn.Parameter(torch.ones(features))
        self.trace_scale = nn.Parameter(torch.full((features,), 0.1))

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> ChronoPlasticState:
        shape = (batch_size, self.features)
        return ChronoPlasticState(
            membrane=torch.zeros(shape, device=device, dtype=dtype),
            spike=torch.zeros(shape, device=device, dtype=dtype),
            traces=torch.zeros(batch_size, self.features, self.num_traces, device=device, dtype=dtype),
        )

    def effective_threshold(self, threshold_offset: Optional[Tensor] = None) -> float | Tensor:
        """Base threshold plus optional per-neuron / per-batch offset."""
        if threshold_offset is None:
            return self.threshold
        return self.threshold + threshold_offset

    def forward(
        self,
        input_current: Tensor,
        state: Optional[ChronoPlasticState] = None,
        *,
        threshold_offset: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, ChronoPlasticState, Tensor]:
        if input_current.ndim != 2 or input_current.shape[-1] != self.features:
            raise ValueError(f"input_current must have shape [B,{self.features}]")
        if state is None:
            state = self.initial_state(input_current.shape[0], input_current.device, input_current.dtype)
        if threshold_offset is not None:
            if threshold_offset.ndim == 1 and threshold_offset.shape[0] != self.features:
                raise ValueError(f"threshold_offset must have shape [{self.features}] or [B,{self.features}]")
            if threshold_offset.ndim == 2 and (
                threshold_offset.shape[0] != input_current.shape[0]
                or threshold_offset.shape[1] != self.features
            ):
                raise ValueError(f"threshold_offset must have shape [{self.features}] or [B,{self.features}]")

        trace_decays = torch.sigmoid(self.trace_decay_logits).to(dtype=input_current.dtype, device=input_current.device)
        traces = state.traces * trace_decays.view(1, 1, -1) + input_current.unsqueeze(-1)
        trace_summary = self.trace_readout(traces).squeeze(-1)
        trace_mean = traces.mean(dim=-1)
        leak_condition = torch.cat([input_current, state.membrane, state.spike, trace_mean], dim=-1)
        leak_unit = torch.sigmoid(self.leak_gate(leak_condition))
        leak = self.min_leak + (self.max_leak - self.min_leak) * leak_unit

        membrane = (
            leak * state.membrane
            + self.input_scale * input_current
            + self.trace_scale * trace_summary
        )
        threshold = self.effective_threshold(threshold_offset)
        if threshold_offset is not None and isinstance(threshold, Tensor):
            threshold = threshold.to(device=membrane.device, dtype=membrane.dtype)
        spikes = self.spike_fn(membrane, threshold=threshold if threshold_offset is not None else None)
        membrane = torch.where(spikes.bool(), torch.full_like(membrane, self.reset), membrane)
        next_state = ChronoPlasticState(membrane=membrane, spike=spikes, traces=traces)
        return spikes, membrane, next_state, leak


class ChronoPlasticLIFLayer(nn.Module):
    """Sequence layer wrapping ``ChronoPlasticLIFCell``.

    Input shape is ``[B, T, input_size]``. The internal linear projection can be
    interpreted as synaptic current generation, while the cell handles adaptive
    temporal credit assignment with state-conditioned leak.

    ``threshold_offset`` (optional ``[H]`` or ``[B, H]``) raises or lowers the
    per-neuron spike threshold for homeostatic control without changing weights.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_traces: int = 3,
        **cell_kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input = nn.Linear(input_size, hidden_size)
        self.cell = ChronoPlasticLIFCell(hidden_size, num_traces=num_traces, **cell_kwargs)

    @property
    def threshold(self) -> float:
        return float(self.cell.threshold)

    def forward(
        self,
        x: Tensor,
        state: Optional[ChronoPlasticState] = None,
        *,
        threshold_offset: Optional[Tensor] = None,
    ) -> dict[str, Tensor | ChronoPlasticState]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,T,F]")
        spikes = []
        membranes = []
        leaks = []
        for step in range(x.shape[1]):
            current = self.input(x[:, step])
            spike, membrane, state, leak = self.cell(
                current,
                state,
                threshold_offset=threshold_offset,
            )
            spikes.append(spike)
            membranes.append(membrane)
            leaks.append(leak)
        return {
            "spikes": torch.stack(spikes, dim=1),
            "membrane": torch.stack(membranes, dim=1),
            "leak": torch.stack(leaks, dim=1),
            "state": state,
            "threshold_offset": threshold_offset,
        }


class ConfidenceAwareBatchNorm1d(nn.Module):
    """Confidence-adaptive and recalibratable BatchNorm.

    Paper concept: CaRe-BN stabilizes SNN reinforcement learning by updating BN
    moving statistics according to sample confidence and by recalibrating moving
    statistics from buffered/replay data. Inference remains standard BN with
    frozen running statistics.

    The default ``channels_last=True`` matches SNN sequence tensors:
    ``[B, T, C]`` or ``[B, T, N, C]``.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        channels_last: bool = True,
        min_update_scale: float = 0.05,
        max_update_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.channels_last = channels_last
        self.min_update_scale = min_update_scale
        self.max_update_scale = max_update_scale
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("running_confidence", torch.tensor(1.0))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("running_confidence", None)
            self.register_buffer("num_batches_tracked", None)

    def _to_channel_first(self, x: Tensor) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        if x.ndim < 2:
            raise ValueError("BatchNorm input must have at least 2 dimensions")
        if self.channels_last and x.ndim > 2:
            x_cf = x.movedim(-1, 1)

            def restore(y: Tensor) -> Tensor:
                return y.movedim(1, -1)

            return x_cf, restore

        def identity(y: Tensor) -> Tensor:
            return y

        return x, identity

    def _confidence_to_channel_first(self, confidence: Optional[Tensor], x_cf: Tensor) -> Optional[Tensor]:
        if confidence is None:
            return None
        conf = confidence.to(dtype=x_cf.dtype, device=x_cf.device).clamp(0.0, 1.0)
        batch = x_cf.shape[0]
        spatial_rank = x_cf.ndim - 2
        if conf.ndim == 1:
            conf = conf.view(batch, 1, *([1] * spatial_rank))
        elif conf.ndim == x_cf.ndim:
            if conf.shape[1] == x_cf.shape[1]:
                conf = conf.mean(dim=1, keepdim=True)
            elif conf.shape[1] != 1:
                raise ValueError("confidence with channel dimension must have C or 1 channels")
        elif conf.ndim >= 2 and conf.shape[0] == batch:
            missing = spatial_rank - (conf.ndim - 1)
            if missing < 0:
                raise ValueError("confidence has too many spatial dimensions")
            conf = conf.reshape(batch, 1, *conf.shape[1:], *([1] * missing))
        else:
            raise ValueError("confidence must be shaped as [B], [B,...], or [B,1/C,...]")
        return conf.expand_as(x_cf)

    def _weighted_stats(self, x_cf: Tensor, confidence: Optional[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        reduce_dims = (0, *range(2, x_cf.ndim))
        weights = self._confidence_to_channel_first(confidence, x_cf)
        if weights is None:
            mean = x_cf.mean(dim=reduce_dims)
            var = x_cf.var(dim=reduce_dims, unbiased=False)
            confidence_mean = x_cf.new_tensor(1.0)
            return mean, var, confidence_mean
        sum_w = weights.sum(dim=reduce_dims).clamp_min(self.eps)
        mean = (x_cf * weights).sum(dim=reduce_dims) / sum_w
        view_shape = (1, -1, *([1] * (x_cf.ndim - 2)))
        var = ((x_cf - mean.view(view_shape)).pow(2) * weights).sum(dim=reduce_dims) / sum_w
        confidence_mean = weights.mean().detach()
        return mean, var, confidence_mean

    def _normalize(self, x_cf: Tensor, mean: Tensor, var: Tensor) -> Tensor:
        view_shape = (1, -1, *([1] * (x_cf.ndim - 2)))
        y = (x_cf - mean.view(view_shape)) * torch.rsqrt(var.view(view_shape) + self.eps)
        if self.affine:
            y = y * self.weight.view(view_shape) + self.bias.view(view_shape)
        return y

    def forward(self, x: Tensor, confidence: Optional[Tensor] = None) -> Tensor:
        x_cf, restore = self._to_channel_first(x)
        if x_cf.shape[1] != self.num_features:
            raise ValueError(f"expected {self.num_features} channels, got {x_cf.shape[1]}")
        if self.training or not self.track_running_stats:
            mean, var, confidence_mean = self._weighted_stats(x_cf, confidence)
            if self.track_running_stats:
                scale = confidence_mean.clamp(self.min_update_scale, self.max_update_scale)
                adaptive_momentum = float(self.momentum) * float(scale.detach().cpu())
                self.running_mean.lerp_(mean.detach(), adaptive_momentum)
                self.running_var.lerp_(var.detach(), adaptive_momentum)
                self.running_confidence.lerp_(confidence_mean, adaptive_momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean, var = self.running_mean, self.running_var
        return restore(self._normalize(x_cf, mean, var))

    @torch.no_grad()
    def reset_running_stats(self) -> None:
        if not self.track_running_stats:
            return
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.running_confidence.fill_(1.0)
        self.num_batches_tracked.zero_()

    @torch.no_grad()
    def recalibrate(
        self,
        batches: Iterable[Any],
        input_getter: Optional[Callable[[Any], Tensor]] = None,
        confidence_getter: Optional[Callable[[Any], Optional[Tensor]]] = None,
        max_batches: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        """Recompute exact weighted running statistics from replay/buffer data."""
        if not self.track_running_stats:
            raise RuntimeError("recalibrate requires track_running_stats=True")
        device = torch.device(device) if device is not None else self.running_mean.device
        sum_w = torch.zeros(self.num_features, device=device)
        sum_x = torch.zeros(self.num_features, device=device)
        sum_x2 = torch.zeros(self.num_features, device=device)
        confidence_total = torch.tensor(0.0, device=device)
        confidence_count = torch.tensor(0.0, device=device)

        for index, batch in enumerate(batches):
            if max_batches is not None and index >= max_batches:
                break
            x = self._extract_input(batch, input_getter).to(device)
            confidence = self._extract_confidence(batch, confidence_getter)
            if confidence is not None:
                confidence = confidence.to(device)
            x_cf, _ = self._to_channel_first(x)
            weights = self._confidence_to_channel_first(confidence, x_cf)
            if weights is None:
                weights = torch.ones_like(x_cf)
            reduce_dims = (0, *range(2, x_cf.ndim))
            batch_sum_w = weights.sum(dim=reduce_dims)
            sum_w += batch_sum_w
            sum_x += (x_cf * weights).sum(dim=reduce_dims)
            sum_x2 += (x_cf.pow(2) * weights).sum(dim=reduce_dims)
            confidence_total += weights.mean()
            confidence_count += 1

        valid = sum_w > 0
        mean = torch.where(valid, sum_x / sum_w.clamp_min(self.eps), self.running_mean.to(device))
        var = torch.where(valid, sum_x2 / sum_w.clamp_min(self.eps) - mean.pow(2), self.running_var.to(device))
        self.running_mean.copy_(mean.to(self.running_mean.device))
        self.running_var.copy_(var.clamp_min(self.eps).to(self.running_var.device))
        if confidence_count > 0:
            self.running_confidence.copy_((confidence_total / confidence_count).to(self.running_confidence.device))
        self.num_batches_tracked.fill_(int(confidence_count.item()))

    @staticmethod
    def _extract_input(batch: Any, input_getter: Optional[Callable[[Any], Tensor]]) -> Tensor:
        if input_getter is not None:
            return input_getter(batch)
        if isinstance(batch, Mapping):
            for key in ("x", "input", "inputs", "obs", "observation"):
                if key in batch:
                    return batch[key]
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    @staticmethod
    def _extract_confidence(
        batch: Any,
        confidence_getter: Optional[Callable[[Any], Optional[Tensor]]],
    ) -> Optional[Tensor]:
        if confidence_getter is not None:
            return confidence_getter(batch)
        if isinstance(batch, Mapping):
            for key in ("confidence", "conf", "critic_confidence"):
                if key in batch:
                    return batch[key]
        return None


class HighFrequencySpikingTransformerBlock(nn.Module):
    """Composable block combining CaRe-BN, high-frequency mixing, and CPSNN."""

    def __init__(
        self,
        dim: int,
        grid_size: Optional[tuple[int, int]] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        tgo_noise_std: float = 0.03,
    ) -> None:
        super().__init__()
        self.mixer = HighFrequencyTokenMixer(dim, grid_size=grid_size, dropout=dropout)
        self.bn = ConfidenceAwareBatchNorm1d(dim, channels_last=True)
        self.chrono_lif = ChronoPlasticLIFLayer(
            dim,
            dim,
            threshold=1.0,
            noise_std=tgo_noise_std,
        )
        hidden = int(dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        grid_size: Optional[tuple[int, int]] = None,
        confidence: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        mixed = self.mixer(x, grid_size=grid_size)
        mixed = self.bn(mixed, confidence=confidence)
        batch, time, tokens, channels = mixed.shape
        lif_input = mixed.permute(0, 2, 1, 3).reshape(batch * tokens, time, channels)
        lif_out = self.chrono_lif(lif_input)
        spikes = lif_out["spikes"].reshape(batch, tokens, time, channels).permute(0, 2, 1, 3)
        membrane = lif_out["membrane"].reshape(batch, tokens, time, channels).permute(0, 2, 1, 3)
        leak = lif_out["leak"].reshape(batch, tokens, time, channels).permute(0, 2, 1, 3)
        x = x + spikes
        x = x + self.mlp(self.mlp_norm(x))
        return {"x": x, "spikes": spikes, "membrane": membrane, "leak": leak}


class ResearchSpikingTransformerSNN(nn.Module):
    """Integrated example architecture using all four requested mechanisms."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        image_size: int | tuple[int, int] = 32,
        patch_size: int = 4,
        embed_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
        tgo_noise_std: float = 0.03,
    ) -> None:
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        grid = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.threshold = 1.0
        self.patch_embed = MaxPoolPatchEmbed(in_channels, embed_dim, patch_size=patch_size)
        self.blocks = nn.ModuleList([
            HighFrequencySpikingTransformerBlock(
                embed_dim,
                grid_size=grid,
                dropout=dropout,
                tgo_noise_std=tgo_noise_std,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor, confidence: Optional[Tensor] = None) -> dict[str, Any]:
        tokens, grid = self.patch_embed(x)
        membranes = []
        spikes = []
        leaks = []
        for block in self.blocks:
            out = block(tokens, grid_size=grid, confidence=confidence)
            tokens = out["x"]
            membranes.append(out["membrane"])
            spikes.append(out["spikes"])
            leaks.append(out["leak"])
        pooled = self.norm(tokens).mean(dim=(1, 2))
        logits = self.head(pooled)
        return {
            "logits": logits,
            "tokens": tokens,
            "membranes": membranes,
            "spikes": spikes,
            "leaks": leaks,
            "grid": grid,
        }


def research_snn_training_step(
    model: nn.Module,
    batch: Mapping[str, Tensor] | Sequence[Tensor],
    optimizer: torch.optim.Optimizer,
    tgo_loss: Optional[ThresholdGuardingLoss] = None,
    task_loss_fn: Optional[nn.Module] = None,
    spike_l1: float = 1e-4,
    tgo_weight: float = 0.05,
    grad_clip: Optional[float] = 1.0,
) -> dict[str, float]:
    """One training step wiring task loss, TGO loss, and spike-rate penalty.

    Expected batch forms:
    ``{"x": input, "y": target, "confidence": optional_confidence}`` or
    ``(input, target[, confidence])``.
    """

    if isinstance(batch, Mapping):
        x = batch["x"]
        y = batch["y"]
        confidence = batch.get("confidence")
    else:
        x = batch[0]
        y = batch[1]
        confidence = batch[2] if len(batch) > 2 else None

    model.train()
    out = model(x, confidence=confidence)
    logits = out["logits"]
    if task_loss_fn is None:
        task_loss = F.cross_entropy(logits, y)
    else:
        task_loss = task_loss_fn(logits, y)

    if tgo_loss is None:
        tgo_loss = ThresholdGuardingLoss(weight=1.0)
    threshold = getattr(model, "threshold", 1.0)
    guard_loss = tgo_loss(out["membranes"], threshold=threshold) * tgo_weight
    if out["spikes"]:
        spike_rate = torch.stack([item.float().mean() for item in out["spikes"]]).mean()
    else:
        spike_rate = logits.new_tensor(0.0)
    loss = task_loss + guard_loss + spike_l1 * spike_rate

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "task_loss": float(task_loss.detach().cpu()),
        "tgo_loss": float(guard_loss.detach().cpu()),
        "spike_rate": float(spike_rate.detach().cpu()),
    }
