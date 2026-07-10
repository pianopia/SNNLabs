from __future__ import annotations

from pathlib import Path

from src.dst_snn.eval.result import MetricSet, RunResult, write_report


def _metrics(quality: float) -> MetricSet:
    return MetricSet(
        quality=quality,
        quality_metric="accuracy",
        latency_ms_p50=1.0,
        latency_ms_p95=2.0,
        spikes_per_inference=10.0,
        active_neuron_fraction=0.1,
        energy_pj=100.0,
        energy_source="test",
        param_count=42,
        model_bytes=168,
        extra={},
    )


def test_run_result_json_roundtrip():
    result = RunResult(
        benchmark="n-mnist",
        model="dst-snn",
        metrics=_metrics(0.9),
        baseline=_metrics(0.95),
        meta={"epochs": 3},
    )
    restored = RunResult.from_json(result.to_json())
    assert restored.benchmark == "n-mnist"
    assert restored.metrics.quality == 0.9
    assert restored.baseline is not None
    assert restored.baseline.quality == 0.95
    assert restored.meta["epochs"] == 3


def test_write_report(tmp_path: Path):
    result = RunResult(
        benchmark="n-mnist",
        model="dst-snn",
        metrics=_metrics(0.9),
        baseline=None,
        meta={},
    )
    out = tmp_path / "report.md"
    write_report([result], out)
    text = out.read_text(encoding="utf-8")
    assert "n-mnist" in text
    assert "accuracy" in text
    assert "| Benchmark |" in text
