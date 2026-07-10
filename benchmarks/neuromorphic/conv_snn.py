"""Spatial Conv+PLIF SNN for event-frame classification.

Research context
----------------
SOTA DVS Gesture SNNs (SEW-ResNet, SpikingJelly tutorials) keep the 2D event
geometry and stack Conv-BN-PLIF blocks rather than flattening polarity frames
into dense vectors. This module is a compact, dependency-light version of that
pattern, plus Spike-Element-Wise (SEW) residual blocks (Fang et al., NeurIPS 2021).
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from benchmarks.neuromorphic.plif import PLIF


class ConvPLIFBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        threshold: float = 1.0,
        init_tau: float = 2.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = PLIF(init_tau=init_tau, threshold=threshold)

    def forward(self, x: Tensor, v: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        return self.lif(self.bn(self.conv(x)), v)


class SEWResidualBlock(nn.Module):
    """Spike-Element-Wise residual block (ADD connect).

    Identity path is pure spikes (no second LIF on the shortcut), so identity
    mapping is easy and gradients do not vanish through deep stacks. Downsample
    uses a stride-2 1×1 conv + PLIF when dimensions change.
    """

    def __init__(
        self,
        channels: int,
        *,
        stride: int = 1,
        threshold: float = 1.0,
        init_tau: float = 2.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.lif1 = PLIF(init_tau=init_tau, threshold=threshold)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.lif2 = PLIF(init_tau=init_tau, threshold=threshold)
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.downsample_lif = PLIF(init_tau=init_tau, threshold=threshold)
        else:
            self.downsample = None
            self.downsample_lif = None

    def forward(
        self,
        x: Tensor,
        state: Optional[tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]] = None,
    ) -> tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]]:
        v1 = v2 = vd = None
        if state is not None:
            v1, v2, vd = state
        identity = x
        s1, v1 = self.lif1(self.bn1(self.conv1(x)), v1)
        s2, v2 = self.lif2(self.bn2(self.conv2(s1)), v2)
        if self.downsample is not None and self.downsample_lif is not None:
            identity, vd = self.downsample_lif(self.downsample(identity), vd)
        # SEW-ADD: element-wise sum of spike tensors (can exceed 1 — intentional).
        out = s2 + identity
        return out, (v1, v2, vd)


class ConvPLIFClassifier(nn.Module):
    """Lightweight Conv-BN-PLIF stack for ``[B, T, C, H, W]`` event frames.

    Default topology (inspired by SEW / SpikingJelly DVS demos, scaled down):
    C→32 (s1) → 64 (s2) → 64 (s2) → GAP → Linear(num_classes).
    Logits are accumulated membrane or spike counts over time.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 11,
        *,
        channels: tuple[int, int, int] = (32, 64, 64),
        threshold: float = 1.0,
        init_tau: float = 2.0,
        dropout: float = 0.0,
        readout: str = "spike_count",
    ) -> None:
        super().__init__()
        if readout not in {"spike_count", "max_membrane", "mean_membrane"}:
            raise ValueError("readout must be one of: spike_count, max_membrane, mean_membrane")
        c1, c2, c3 = channels
        self.readout = readout
        self.block1 = ConvPLIFBlock(in_channels, c1, stride=1, threshold=threshold, init_tau=init_tau)
        self.block2 = ConvPLIFBlock(c1, c2, stride=2, threshold=threshold, init_tau=init_tau)
        self.block3 = ConvPLIFBlock(c2, c3, stride=2, threshold=threshold, init_tau=init_tau)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c3, num_classes)
        self.out_channels = c3

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # x: [B, T, C, H, W]
        if x.ndim != 5:
            raise ValueError("ConvPLIFClassifier expects [batch, time, channels, height, width]")
        batch, time_steps, _, _, _ = x.shape
        v1 = v2 = v3 = None
        spikes_t: list[Tensor] = []
        membrane_t: list[Tensor] = []
        for t in range(time_steps):
            s1, v1 = self.block1(x[:, t], v1)
            s2, v2 = self.block2(s1, v2)
            s3, v3 = self.block3(s2, v3)
            # Global average pool of spikes and membrane at this step.
            pooled_s = s3.mean(dim=(-2, -1))  # [B, C]
            pooled_m = v3.mean(dim=(-2, -1))
            spikes_t.append(pooled_s)
            membrane_t.append(pooled_m)
        spikes = torch.stack(spikes_t, dim=1)  # [B, T, C]
        membrane = torch.stack(membrane_t, dim=1)
        if self.readout == "max_membrane":
            features = membrane.amax(dim=1)
        elif self.readout == "mean_membrane":
            features = membrane.mean(dim=1)
        else:
            features = spikes.sum(dim=1)
        logits = self.fc(self.dropout(features))
        # Project channel features to class-dim spikes for decision-latency hooks:
        # use logits-as-proxy by repeating fc weights on per-step pooled spikes.
        class_spikes = torch.relu(spikes @ self.fc.weight.t())  # [B,T,num_classes]
        return {
            "logits": logits,
            "spikes": class_spikes,
            "membrane": membrane,
            "feature_spikes": spikes,
            "feature_membrane": membrane,
        }


class SewConvPLIFClassifier(nn.Module):
    """Deeper Conv-PLIF with SEW residual stages.

    Stem: 2→width, then N residual blocks at width, downsample+blocks at 2×width,
    then GAP + FC. Membrane state is carried across time steps inside each PLIF.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 11,
        *,
        width: int = 32,
        blocks_per_stage: int = 2,
        threshold: float = 1.0,
        init_tau: float = 2.0,
        dropout: float = 0.0,
        readout: str = "spike_count",
    ) -> None:
        super().__init__()
        if readout not in {"spike_count", "max_membrane", "mean_membrane"}:
            raise ValueError("readout must be one of: spike_count, max_membrane, mean_membrane")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1")
        self.readout = readout
        self.stem = ConvPLIFBlock(in_channels, width, stride=1, threshold=threshold, init_tau=init_tau)
        stage1 = [SEWResidualBlock(width, stride=1, threshold=threshold, init_tau=init_tau) for _ in range(blocks_per_stage)]
        # First block of stage 2 downsamples.
        stage2 = [SEWResidualBlock(width, stride=2, threshold=threshold, init_tau=init_tau)]
        stage2 += [
            SEWResidualBlock(width, stride=1, threshold=threshold, init_tau=init_tau)
            for _ in range(blocks_per_stage - 1)
        ]
        # Expand channels via a stride-2 conv PLIF into 2*width, then residuals.
        self.expand = ConvPLIFBlock(width, width * 2, stride=2, threshold=threshold, init_tau=init_tau)
        stage3 = [
            SEWResidualBlock(width * 2, stride=1, threshold=threshold, init_tau=init_tau)
            for _ in range(blocks_per_stage)
        ]
        self.stage1 = nn.ModuleList(stage1)
        self.stage2 = nn.ModuleList(stage2)
        self.stage3 = nn.ModuleList(stage3)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(width * 2, num_classes)
        self.out_channels = width * 2
        self.width = width
        self.blocks_per_stage = blocks_per_stage

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 5:
            raise ValueError("SewConvPLIFClassifier expects [batch, time, channels, height, width]")
        time_steps = x.shape[1]
        v_stem = None
        st1 = [None] * len(self.stage1)
        st2 = [None] * len(self.stage2)
        st3 = [None] * len(self.stage3)
        v_exp = None
        spikes_t: list[Tensor] = []
        membrane_t: list[Tensor] = []
        for t in range(time_steps):
            h, v_stem = self.stem(x[:, t], v_stem)
            for i, block in enumerate(self.stage1):
                h, st1[i] = block(h, st1[i])
            for i, block in enumerate(self.stage2):
                h, st2[i] = block(h, st2[i])
            h, v_exp = self.expand(h, v_exp)
            for i, block in enumerate(self.stage3):
                h, st3[i] = block(h, st3[i])
            # After SEW-ADD, h may be multi-valued spikes; use as activity map.
            pooled = h.mean(dim=(-2, -1))
            spikes_t.append(pooled)
            # No single membrane at residual output; reuse activity as membrane proxy.
            membrane_t.append(pooled)
        spikes = torch.stack(spikes_t, dim=1)
        membrane = torch.stack(membrane_t, dim=1)
        if self.readout == "max_membrane":
            features = membrane.amax(dim=1)
        elif self.readout == "mean_membrane":
            features = membrane.mean(dim=1)
        else:
            features = spikes.sum(dim=1)
        logits = self.fc(self.dropout(features))
        class_spikes = torch.relu(spikes @ self.fc.weight.t())
        return {
            "logits": logits,
            "spikes": class_spikes,
            "membrane": membrane,
            "feature_spikes": spikes,
            "feature_membrane": membrane,
        }
