"""JSON result schema and Markdown report for benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class MetricSet:
    quality: float
    quality_metric: str
    latency_ms_p50: float
    latency_ms_p95: float
    spikes_per_inference: float
    active_neuron_fraction: float
    energy_pj: float
    energy_source: str
    param_count: int
    model_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    benchmark: str
    model: str
    metrics: MetricSet
    baseline: Optional[MetricSet] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(text: str) -> "RunResult":
        data = json.loads(text)
        baseline = data.get("baseline")
        return RunResult(
            benchmark=data["benchmark"],
            model=data["model"],
            metrics=MetricSet(**data["metrics"]),
            baseline=MetricSet(**baseline) if baseline is not None else None,
            meta=data.get("meta", {}),
        )


def write_report(results: list[RunResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| Benchmark | Model | Quality | Metric | Lat p50 (ms) | Lat p95 (ms) "
        "| Spikes/inf | Active frac | Energy (pJ) | Params |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for result in results:
        for label, metrics in (("", result.metrics), ("baseline", result.baseline)):
            if metrics is None:
                continue
            model_name = f"{result.model} ({label})" if label else result.model
            rows.append(
                f"| {result.benchmark} | {model_name} | {metrics.quality:.4f} "
                f"| {metrics.quality_metric} | {metrics.latency_ms_p50:.3f} "
                f"| {metrics.latency_ms_p95:.3f} | {metrics.spikes_per_inference:.1f} "
                f"| {metrics.active_neuron_fraction:.4f} | {metrics.energy_pj:.1f} "
                f"| {metrics.param_count} |"
            )
    path.write_text("# Benchmark Report\n\n" + header + "\n".join(rows) + "\n", encoding="utf-8")
