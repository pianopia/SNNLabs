"""Benchmark-agnostic metric functions for the SNN eval harness."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


def accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Fraction of exactly matching integer class predictions."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.numel() == 0:
        return 0.0
    return float((predictions == targets).float().mean().item())


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    rank = fraction * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Return p50, p95, and mean of per-inference latencies in milliseconds."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(float(v) for v in latencies_ms)
    return {
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "mean": sum(ordered) / len(ordered),
    }


def spike_stats(spikes: Tensor) -> dict[str, float]:
    """Sparsity stats from a ``[batch, time, neurons]`` spike tensor."""
    if spikes.ndim != 3:
        raise ValueError("spikes must have shape [batch, time, neurons]")
    batch = spikes.shape[0]
    spikes_per_inference = float(spikes.sum().item()) / max(1, batch)
    fired = (spikes.sum(dim=1) > 0).float()
    return {
        "spikes_per_inference": spikes_per_inference,
        "active_neuron_fraction": float(fired.mean().item()),
    }


def model_size(module: nn.Module) -> dict[str, int]:
    """Parameter count and byte size of a module."""
    param_count = 0
    model_bytes = 0
    for param in module.parameters():
        param_count += param.numel()
        model_bytes += param.numel() * param.element_size()
    return {"param_count": int(param_count), "model_bytes": int(model_bytes)}
