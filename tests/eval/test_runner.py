from __future__ import annotations

from pathlib import Path

from src.dst_snn.eval.result import MetricSet, RunResult
from src.dst_snn.eval.runner import run_benchmarks


class _FakeRunner:
    name = "fake"

    def __init__(self):
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True

    def run(self) -> RunResult:
        assert self.prepared, "prepare must be called before run"
        return RunResult(
            benchmark="fake",
            model="dst-snn",
            metrics=MetricSet(
                quality=1.0,
                quality_metric="accuracy",
                latency_ms_p50=0.5,
                latency_ms_p95=0.9,
                spikes_per_inference=3.0,
                active_neuron_fraction=0.2,
                energy_pj=10.0,
                energy_source="test",
                param_count=5,
                model_bytes=20,
                extra={},
            ),
            baseline=None,
            meta={},
        )


def test_run_benchmarks_writes_outputs(tmp_path: Path):
    results = run_benchmarks([_FakeRunner()], tmp_path)
    assert len(results) == 1
    assert results[0].benchmark == "fake"
    assert (tmp_path / "fake.json").exists()
    assert (tmp_path / "report.md").exists()
    assert "fake" in (tmp_path / "report.md").read_text(encoding="utf-8")
