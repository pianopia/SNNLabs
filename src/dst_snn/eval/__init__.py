"""Benchmark-agnostic SNN evaluation harness."""

from .energy import EnergyModel, dense_mac_energy_pj, energy_ratio, snn_energy_pj
from .result import MetricSet, RunResult, write_report

__all__ = [
    "BenchmarkRunner",
    "EnergyModel",
    "MetricSet",
    "RunResult",
    "accuracy",
    "dense_mac_energy_pj",
    "energy_ratio",
    "latency_percentiles",
    "model_size",
    "run_benchmarks",
    "snn_energy_pj",
    "spike_stats",
    "write_report",
]


def __getattr__(name: str):
    if name in {"accuracy", "latency_percentiles", "model_size", "spike_stats"}:
        from .metrics import accuracy, latency_percentiles, model_size, spike_stats

        return {
            "accuracy": accuracy,
            "latency_percentiles": latency_percentiles,
            "model_size": model_size,
            "spike_stats": spike_stats,
        }[name]
    if name in {"BenchmarkRunner", "run_benchmarks"}:
        from .runner import BenchmarkRunner, run_benchmarks

        return {
            "BenchmarkRunner": BenchmarkRunner,
            "run_benchmarks": run_benchmarks,
        }[name]
    raise AttributeError(name)
