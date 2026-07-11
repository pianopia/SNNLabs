"""Track 1: decode spikes into a mesh-op sequence.

``scripted`` mode is fully offline and deterministic. Optional torch head is a
trainable scaffold (untrained logits → fallback to scripted).
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np

from .ops import (
    OP_ADD_BOX,
    OP_ADD_CYLINDER,
    OP_ADD_SPHERE,
    OP_FINISH,
    OP_SCALE,
    MeshOp,
)


def _mean_rate(spikes: np.ndarray) -> float:
    arr = np.asarray(spikes, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def scripted_box_policy(
    spikes: np.ndarray,
    *,
    extents_hint: Optional[Sequence[float]] = None,
) -> list[MeshOp]:
    """Heuristic policy: single scaled box from mean spike activity.

    Higher mean rate → larger extents (clamped). ``extents_hint`` overrides
    when provided (e.g. reference AABB for teacher-forced eval).
    """
    if extents_hint is not None:
        extents = [float(x) for x in extents_hint]
    else:
        rate = _mean_rate(spikes)
        scale = 0.5 + 1.5 * rate  # ~0.5–2.0
        extents = [scale, scale, scale]
    return [
        MeshOp(OP_ADD_BOX, {"extents": extents}),
        MeshOp(OP_SCALE, {"factors": [1.0, 1.0, 1.0]}),
        MeshOp(OP_FINISH, {}),
    ]


def scripted_shape_policy(
    spikes: np.ndarray,
    *,
    shape: Literal["box", "sphere", "cylinder"] = "box",
    extents_hint: Optional[Sequence[float]] = None,
) -> list[MeshOp]:
    rate = _mean_rate(spikes)
    if shape == "box":
        return scripted_box_policy(spikes, extents_hint=extents_hint)
    if shape == "sphere":
        if extents_hint is not None:
            radius = 0.5 * float(max(extents_hint))
        else:
            radius = 0.25 + rate
        return [MeshOp(OP_ADD_SPHERE, {"radius": radius}), MeshOp(OP_FINISH, {})]
    # cylinder
    if extents_hint is not None:
        height = float(extents_hint[1] if len(extents_hint) > 1 else extents_hint[0])
        radius = 0.5 * float(max(extents_hint[0], extents_hint[-1]))
    else:
        height = 0.5 + rate
        radius = 0.2 + 0.3 * rate
    return [
        MeshOp(OP_ADD_CYLINDER, {"radius": radius, "height": height}),
        MeshOp(OP_FINISH, {}),
    ]


def decode_ops_from_spikes(
    spikes: np.ndarray,
    *,
    mode: str = "scripted",
    shape: Literal["box", "sphere", "cylinder"] = "box",
    extents_hint: Optional[Sequence[float]] = None,
    head: Any = None,
) -> list[MeshOp]:
    """Decode ops. ``mode`` is ``scripted`` or ``torch`` (uses head if trained-ish)."""
    mode = (mode or "scripted").lower()
    if mode in {"scripted", "heuristic"}:
        return scripted_shape_policy(spikes, shape=shape, extents_hint=extents_hint)
    if mode == "torch":
        if head is None:
            # Fall back rather than fail — untrained scaffold.
            return scripted_shape_policy(spikes, shape=shape, extents_hint=extents_hint)
        return head.decode(spikes)
    raise ValueError(f"unknown track1 mode: {mode!r}")


class Track1OpHead:
    """Tiny torch head: mean-pool spikes → shape class + XYZ extents.

    Output layout: [logit_box, logit_sphere, logit_cylinder, e_x, e_y, e_z]
    where extents use softplus-ish mapping to (0.2, ~2.0).
    """

    def __init__(self, in_features: int, *, seed: int = 0) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install PyTorch for Track1OpHead.") from exc

        self._torch = torch
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        with torch.no_grad():
            for p in self.net.parameters():
                p.normal_(0.0, 0.02, generator=g)
        self.in_features = in_features

    def _features(self, spikes: np.ndarray):
        torch = self._torch
        x = torch.as_tensor(np.asarray(spikes, dtype=np.float32).mean(axis=0), dtype=torch.float32)
        if x.numel() != self.in_features:
            buf = torch.zeros(self.in_features, dtype=torch.float32)
            n = min(self.in_features, int(x.numel()))
            buf[:n] = x.reshape(-1)[:n]
            return buf
        return x

    def decode(self, spikes: np.ndarray) -> list[MeshOp]:
        torch = self._torch
        x = self._features(spikes)
        with torch.no_grad():
            out = self.net(x)
        cls = int(out[:3].argmax().item())
        extents = (torch.nn.functional.softplus(out[3:6]) + 0.2).tolist()
        shape = ("box", "sphere", "cylinder")[cls]
        return scripted_shape_policy(
            spikes,
            shape=shape,  # type: ignore[arg-type]
            extents_hint=extents,
        )
