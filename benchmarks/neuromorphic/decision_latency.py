"""Decision-latency metric: how early in an event stream the SNN commits."""

from __future__ import annotations

try:
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


def running_predictions(spikes: Tensor) -> Tensor:
    if spikes.ndim != 3:
        raise ValueError("spikes must have shape [batch, time, classes]")
    cumulative = spikes.cumsum(dim=1)
    return cumulative.argmax(dim=-1)


def decision_latency_fraction(spikes: Tensor, targets: Tensor, *, confirm_window: int = 3) -> float:
    if spikes.ndim != 3:
        raise ValueError("spikes must have shape [batch, time, classes]")
    if confirm_window <= 0:
        raise ValueError("confirm_window must be positive")
    batch, time_steps, _ = spikes.shape
    preds = running_predictions(spikes)
    fractions: list[float] = []
    for b in range(batch):
        target = int(targets[b].item())
        latency = 1.0
        for step in range(time_steps):
            window_end = min(time_steps, step + confirm_window)
            if all(int(preds[b, tt].item()) == target for tt in range(step, window_end)):
                latency = (step + 1) / time_steps
                break
        fractions.append(latency)
    return sum(fractions) / max(1, len(fractions))
