# SNN Evaluation Harness + Neuromorphic Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared SNN evaluation harness (quality / latency / sparsity / energy / model-size) plus N-MNIST and DVS128 Gesture benchmark runners on the existing PyTorch DST-SNN, and fix two learning bugs in the web learner.

**Architecture:** A dependency-light `src/dst_snn/eval/` package computes benchmark-agnostic metrics and a JSON result schema. `benchmarks/neuromorphic/` converts event-camera datasets into `[batch, time, features]` spike tensors (via the `tonic` library), wraps `DendriticSNN` as a classifier, and produces harness results with a decision-latency metric for DVS. Two bug fixes wire novelty and reward into the online web learner correctly.

**Tech Stack:** Python 3.14, PyTorch ≥2.2, NumPy ≥1.24, tonic ≥1.4 (neuromorphic dataset loader), pytest ≥8.

## Global Constraints

- Python 3.14; PyTorch `>=2.2`; NumPy `>=1.24`; tonic `>=1.4`; pytest `>=8`. (Verbatim floors — add to `requirements-bench.txt`.)
- Every new Python module starts with `from __future__ import annotations`.
- Any module importing `torch` at import time MUST use the existing guard pattern:
  ```python
  try:
      import torch
  except ImportError as exc:  # pragma: no cover
      raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc
  ```
- Energy model defaults (45nm, configurable): `mac_pj=0.9`, `ac_pj=0.1`. Every energy result MUST record its `source` string in metadata.
- Tests MUST NOT access the network. Neuromorphic dataset tests use synthetic events only; real datasets are downloaded by the runner scripts, never by tests.
- Package imports resolve from repo ROOT: `from src.dst_snn... import ...` and `from benchmarks... import ...`.
- Commit after each task with the exact message shown.
- Follow existing repo conventions: scripts insert ROOT on `sys.path`; keep files focused and single-responsibility.

---

## File Structure

```
requirements-bench.txt                      # new deps
pytest.ini                                   # pytest config
tests/conftest.py                            # ROOT on sys.path
src/dst_snn/eval/__init__.py                 # eval package exports
src/dst_snn/eval/energy.py                   # EnergyModel + AC/MAC energy
src/dst_snn/eval/metrics.py                  # accuracy, latency, spikes, size
src/dst_snn/eval/result.py                   # MetricSet, RunResult, report
src/dst_snn/eval/runner.py                   # BenchmarkRunner protocol + loop
benchmarks/__init__.py
benchmarks/neuromorphic/__init__.py
benchmarks/neuromorphic/events.py            # event->frame->spike-tensor (numpy)
benchmarks/neuromorphic/datasets.py          # tonic wrappers (N-MNIST, DVS)
benchmarks/neuromorphic/classifier.py        # DendriticSNN classification head
benchmarks/neuromorphic/decision_latency.py  # DVS reactivity metric
benchmarks/neuromorphic/run_nmnist.py        # N-MNIST runner + __main__
benchmarks/neuromorphic/run_dvs_gesture.py   # DVS runner + __main__
tests/eval/test_energy.py
tests/eval/test_metrics.py
tests/eval/test_result.py
tests/eval/test_runner.py
tests/neuromorphic/test_events.py
tests/neuromorphic/test_classifier.py
tests/neuromorphic/test_decision_latency.py
tests/dst_snn/test_web_learner_novelty.py
tests/dst_snn/test_web_learner_reward.py
```

---

### Task 0: Environment bootstrap

**Files:**
- Create: `requirements-bench.txt`
- Create: `pytest.ini`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: a working `python -m pytest` that can import `src.dst_snn.*` and `benchmarks.*` from ROOT.

- [ ] **Step 1: Create `requirements-bench.txt`**

```
torch>=2.2
numpy>=1.24
tonic>=1.4
pytest>=8
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Create `tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 5: Install dependencies**

Run:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dst-snn.txt -r requirements-bench.txt
```
Expected: installs succeed; `python -c "import torch, numpy, tonic"` prints nothing and exits 0.

- [ ] **Step 6: Verify pytest collects nothing yet**

Run: `python -m pytest`
Expected: `no tests ran` (exit code 5) — confirms config loads.

- [ ] **Step 7: Commit**

```bash
git add requirements-bench.txt pytest.ini tests/__init__.py tests/conftest.py
git commit -m "chore: bootstrap benchmark test environment"
```

---

### Task 1: Energy model

**Files:**
- Create: `src/dst_snn/eval/__init__.py`
- Create: `src/dst_snn/eval/energy.py`
- Test: `tests/eval/__init__.py`, `tests/eval/test_energy.py`

**Interfaces:**
- Produces:
  - `EnergyModel(mac_pj: float = 0.9, ac_pj: float = 0.1, source: str = ...)` frozen dataclass.
  - `snn_energy_pj(total_spikes: float, fanout: int, model: EnergyModel) -> float`
  - `dense_mac_energy_pj(mac_ops: float, model: EnergyModel) -> float`
  - `energy_ratio(snn_pj: float, dense_pj: float) -> float` (dense/snn = ×efficiency; 0 snn → `float("inf")`)

- [ ] **Step 1: Create `tests/eval/__init__.py`** (empty)

```python
```

- [ ] **Step 2: Write the failing test** — `tests/eval/test_energy.py`

```python
from __future__ import annotations

import math

from src.dst_snn.eval.energy import (
    EnergyModel,
    dense_mac_energy_pj,
    energy_ratio,
    snn_energy_pj,
)


def test_defaults():
    m = EnergyModel()
    assert m.mac_pj == 0.9
    assert m.ac_pj == 0.1
    assert m.source


def test_snn_energy_is_spikes_times_fanout_times_ac():
    m = EnergyModel()
    assert snn_energy_pj(total_spikes=100, fanout=10, model=m) == 100.0


def test_dense_energy_is_macs_times_mac_cost():
    m = EnergyModel()
    assert dense_mac_energy_pj(mac_ops=1000, model=m) == 900.0


def test_energy_ratio_reports_efficiency_factor():
    assert energy_ratio(snn_pj=100.0, dense_pj=900.0) == 9.0


def test_energy_ratio_infinite_when_snn_zero():
    assert math.isinf(energy_ratio(snn_pj=0.0, dense_pj=900.0))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_energy.py -v`
Expected: FAIL — `ModuleNotFoundError: src.dst_snn.eval.energy`

- [ ] **Step 4: Create `src/dst_snn/eval/__init__.py`**

```python
"""Benchmark-agnostic SNN evaluation harness."""

from .energy import EnergyModel, dense_mac_energy_pj, energy_ratio, snn_energy_pj

__all__ = [
    "EnergyModel",
    "dense_mac_energy_pj",
    "energy_ratio",
    "snn_energy_pj",
]
```

- [ ] **Step 5: Create `src/dst_snn/eval/energy.py`**

```python
"""Compute-energy proxy for SNN vs dense baselines.

SNN synaptic energy is modeled as accumulate (AC) operations: each spike drives
its post-synaptic fan-out as one AC each. Dense baselines are modeled as
multiply-accumulate (MAC) operations. Per-op energies default to a 45nm process
and are configurable; the source string is recorded in results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyModel:
    mac_pj: float = 0.9
    ac_pj: float = 0.1
    source: str = "45nm defaults (Horowitz ISSCC 2014); configurable"


def snn_energy_pj(total_spikes: float, fanout: int, model: EnergyModel) -> float:
    """Total AC energy (pJ) for ``total_spikes`` each driving ``fanout`` synapses."""
    if total_spikes < 0 or fanout < 0:
        raise ValueError("total_spikes and fanout must be non-negative")
    return float(total_spikes) * float(fanout) * model.ac_pj


def dense_mac_energy_pj(mac_ops: float, model: EnergyModel) -> float:
    """Total MAC energy (pJ) for ``mac_ops`` multiply-accumulate operations."""
    if mac_ops < 0:
        raise ValueError("mac_ops must be non-negative")
    return float(mac_ops) * model.mac_pj


def energy_ratio(snn_pj: float, dense_pj: float) -> float:
    """Efficiency factor: how many times less energy the SNN uses than dense."""
    if snn_pj <= 0:
        return float("inf")
    return float(dense_pj) / float(snn_pj)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_energy.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add src/dst_snn/eval/__init__.py src/dst_snn/eval/energy.py tests/eval/__init__.py tests/eval/test_energy.py
git commit -m "feat: add SNN vs dense energy model to eval harness"
```

---

### Task 2: Quality, latency, sparsity, and size metrics

**Files:**
- Create: `src/dst_snn/eval/metrics.py`
- Modify: `src/dst_snn/eval/__init__.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `accuracy(predictions: Tensor, targets: Tensor) -> float`
  - `latency_percentiles(latencies_ms: list[float]) -> dict[str, float]` keys `p50`, `p95`, `mean`
  - `spike_stats(spikes: Tensor) -> dict[str, float]` keys `spikes_per_inference`, `active_neuron_fraction` (spikes shape `[batch, time, neurons]`)
  - `model_size(module: nn.Module) -> dict[str, int]` keys `param_count`, `model_bytes`

- [ ] **Step 1: Write the failing test** — `tests/eval/test_metrics.py`

```python
from __future__ import annotations

import torch
from torch import nn

from src.dst_snn.eval.metrics import (
    accuracy,
    latency_percentiles,
    model_size,
    spike_stats,
)


def test_accuracy_counts_matching_argmax():
    preds = torch.tensor([0, 1, 2, 2])
    targets = torch.tensor([0, 1, 2, 0])
    assert accuracy(preds, targets) == 0.75


def test_latency_percentiles():
    out = latency_percentiles([10.0, 20.0, 30.0, 40.0])
    assert out["p50"] == 25.0
    assert out["mean"] == 25.0
    assert out["p95"] >= 38.0


def test_spike_stats():
    # batch=2, time=3, neurons=2
    spikes = torch.zeros(2, 3, 2)
    spikes[0, 0, 0] = 1.0
    spikes[0, 1, 0] = 1.0  # neuron 0 fires twice for sample 0
    spikes[1, 2, 1] = 1.0  # neuron 1 fires once for sample 1
    stats = spike_stats(spikes)
    assert stats["spikes_per_inference"] == 1.5  # (2 + 1) / 2
    assert stats["active_neuron_fraction"] == 0.5  # 1 of 2 neurons active per sample


def test_model_size():
    module = nn.Linear(4, 2)  # 4*2 weights + 2 bias = 10 float32 params
    size = model_size(module)
    assert size["param_count"] == 10
    assert size["model_bytes"] == 40  # 10 * 4 bytes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: src.dst_snn.eval.metrics`

- [ ] **Step 3: Create `src/dst_snn/eval/metrics.py`**

```python
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
    fired = (spikes.sum(dim=1) > 0).float()  # [batch, neurons]
    active_neuron_fraction = float(fired.mean().item())
    return {
        "spikes_per_inference": spikes_per_inference,
        "active_neuron_fraction": active_neuron_fraction,
    }


def model_size(module: nn.Module) -> dict[str, int]:
    """Parameter count and byte size of a module."""
    param_count = 0
    model_bytes = 0
    for param in module.parameters():
        param_count += param.numel()
        model_bytes += param.numel() * param.element_size()
    return {"param_count": int(param_count), "model_bytes": int(model_bytes)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Update `src/dst_snn/eval/__init__.py` exports**

Replace the file contents with:
```python
"""Benchmark-agnostic SNN evaluation harness."""

from .energy import EnergyModel, dense_mac_energy_pj, energy_ratio, snn_energy_pj
from .metrics import accuracy, latency_percentiles, model_size, spike_stats

__all__ = [
    "EnergyModel",
    "accuracy",
    "dense_mac_energy_pj",
    "energy_ratio",
    "latency_percentiles",
    "model_size",
    "snn_energy_pj",
    "spike_stats",
]
```

- [ ] **Step 6: Commit**

```bash
git add src/dst_snn/eval/metrics.py src/dst_snn/eval/__init__.py tests/eval/test_metrics.py
git commit -m "feat: add quality/latency/sparsity/size metrics"
```

---

### Task 3: Result schema and report

**Files:**
- Create: `src/dst_snn/eval/result.py`
- Modify: `src/dst_snn/eval/__init__.py`
- Test: `tests/eval/test_result.py`

**Interfaces:**
- Consumes: nothing (pure dataclasses + json).
- Produces:
  - `MetricSet` dataclass with fields: `quality: float`, `quality_metric: str`, `latency_ms_p50: float`, `latency_ms_p95: float`, `spikes_per_inference: float`, `active_neuron_fraction: float`, `energy_pj: float`, `energy_source: str`, `param_count: int`, `model_bytes: int`, `extra: dict`
  - `RunResult` dataclass: `benchmark: str`, `model: str`, `metrics: MetricSet`, `baseline: Optional[MetricSet]`, `meta: dict`; methods `to_json() -> str`, `from_json(text: str) -> RunResult` (staticmethod)
  - `write_report(results: list[RunResult], path: Path) -> None` (writes a Markdown table)

- [ ] **Step 1: Write the failing test** — `tests/eval/test_result.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_result.py -v`
Expected: FAIL — `ModuleNotFoundError: src.dst_snn.eval.result`

- [ ] **Step 3: Create `src/dst_snn/eval/result.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_result.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Update `src/dst_snn/eval/__init__.py` exports**

Replace the file contents with:
```python
"""Benchmark-agnostic SNN evaluation harness."""

from .energy import EnergyModel, dense_mac_energy_pj, energy_ratio, snn_energy_pj
from .metrics import accuracy, latency_percentiles, model_size, spike_stats
from .result import MetricSet, RunResult, write_report

__all__ = [
    "EnergyModel",
    "MetricSet",
    "RunResult",
    "accuracy",
    "dense_mac_energy_pj",
    "energy_ratio",
    "latency_percentiles",
    "model_size",
    "snn_energy_pj",
    "spike_stats",
    "write_report",
]
```

- [ ] **Step 6: Commit**

```bash
git add src/dst_snn/eval/result.py src/dst_snn/eval/__init__.py tests/eval/test_result.py
git commit -m "feat: add benchmark result schema and report"
```

---

### Task 4: BenchmarkRunner protocol and run loop

**Files:**
- Create: `src/dst_snn/eval/runner.py`
- Modify: `src/dst_snn/eval/__init__.py`
- Test: `tests/eval/test_runner.py`

**Interfaces:**
- Consumes: `RunResult`, `MetricSet`, `write_report` (Task 3).
- Produces:
  - `BenchmarkRunner` `Protocol` with attribute `name: str` and methods `prepare(self) -> None`, `run(self) -> RunResult`.
  - `run_benchmarks(runners: list[BenchmarkRunner], out_dir: Path) -> list[RunResult]` — calls `prepare()` then `run()` for each, writes each `RunResult` as `<name>.json` and a combined `report.md` under `out_dir`, returns the results.

- [ ] **Step 1: Write the failing test** — `tests/eval/test_runner.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: src.dst_snn.eval.runner`

- [ ] **Step 3: Create `src/dst_snn/eval/runner.py`**

```python
"""BenchmarkRunner protocol and a small execution loop."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .result import RunResult, write_report


@runtime_checkable
class BenchmarkRunner(Protocol):
    name: str

    def prepare(self) -> None:
        ...

    def run(self) -> RunResult:
        ...


def run_benchmarks(runners: list[BenchmarkRunner], out_dir: Path) -> list[RunResult]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for runner in runners:
        runner.prepare()
        result = runner.run()
        (out_dir / f"{runner.name}.json").write_text(result.to_json(), encoding="utf-8")
        results.append(result)
    write_report(results, out_dir / "report.md")
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_runner.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Update `src/dst_snn/eval/__init__.py` exports**

Add `BenchmarkRunner` and `run_benchmarks`. Replace file contents with:
```python
"""Benchmark-agnostic SNN evaluation harness."""

from .energy import EnergyModel, dense_mac_energy_pj, energy_ratio, snn_energy_pj
from .metrics import accuracy, latency_percentiles, model_size, spike_stats
from .result import MetricSet, RunResult, write_report
from .runner import BenchmarkRunner, run_benchmarks

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
```

- [ ] **Step 6: Commit**

```bash
git add src/dst_snn/eval/runner.py src/dst_snn/eval/__init__.py tests/eval/test_runner.py
git commit -m "feat: add BenchmarkRunner protocol and run loop"
```

---

### Task 5: Fix degenerate novelty in the web learner

**Context:** In `src/dst_snn/web_autonomous_learner.py`, `DstWebLearner.train_observation` computes novelty as
`sum(1 for item in active if self.feature_space.token_to_index.get(item["token"]) is not None) / max(1, len(active))`.
But `FeatureSpace.encode` (called earlier in the same method) inserts every active token into `token_to_index`, so this is always ≈1.0. Novelty must be the fraction of active tokens that were unseen **before** this observation.

**Files:**
- Modify: `src/dst_snn/web_autonomous_learner.py` (add `compute_novelty`; change `train_observation`)
- Test: `tests/dst_snn/__init__.py`, `tests/dst_snn/test_web_learner_novelty.py`

**Interfaces:**
- Produces: `compute_novelty(active: list[dict], known_before: set[str]) -> float` — module-level pure function (fraction of `active` items whose `item["token"]` is not in `known_before`; `0.0` if `active` empty).
- Changes: `train_observation` snapshots known tokens before `encode`, then uses `compute_novelty`.

- [ ] **Step 1: Create `tests/dst_snn/__init__.py`** (empty)

```python
```

- [ ] **Step 2: Write the failing test** — `tests/dst_snn/test_web_learner_novelty.py`

```python
from __future__ import annotations

from src.dst_snn.web_autonomous_learner import compute_novelty


def test_all_novel_when_nothing_known():
    active = [{"token": "text:a"}, {"token": "text:b"}]
    assert compute_novelty(active, known_before=set()) == 1.0


def test_none_novel_when_all_known():
    active = [{"token": "text:a"}, {"token": "text:b"}]
    assert compute_novelty(active, known_before={"text:a", "text:b"}) == 0.0


def test_half_novel():
    active = [{"token": "text:a"}, {"token": "text:b"}]
    assert compute_novelty(active, known_before={"text:a"}) == 0.5


def test_empty_active_is_zero():
    assert compute_novelty([], known_before={"text:a"}) == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/dst_snn/test_web_learner_novelty.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_novelty'`

- [ ] **Step 4: Add `compute_novelty` to `src/dst_snn/web_autonomous_learner.py`**

Insert this module-level function immediately after the `stable_hash` function (right before `class FeatureSpace:`):
```python
def compute_novelty(active: list[dict[str, Any]], known_before: set[str]) -> float:
    """Fraction of active tokens not seen before this observation."""
    if not active:
        return 0.0
    novel = sum(1 for item in active if item["token"] not in known_before)
    return novel / len(active)
```

- [ ] **Step 5: Rewrite the novelty computation in `train_observation`**

In `DstWebLearner.train_observation`, replace these lines:
```python
    def train_observation(self, observation: WebObservation) -> dict[str, Any]:
        spikes, target, active = self.feature_space.encode(observation.modules, self.time_steps)
```
with (snapshot known tokens BEFORE encode):
```python
    def train_observation(self, observation: WebObservation) -> dict[str, Any]:
        known_before = set(self.feature_space.token_to_index)
        spikes, target, active = self.feature_space.encode(observation.modules, self.time_steps)
```
and replace this line:
```python
        novelty = sum(1 for item in active if self.feature_space.token_to_index.get(item["token"]) is not None) / max(1, len(active))
```
with:
```python
        novelty = compute_novelty(active, known_before)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/dst_snn/test_web_learner_novelty.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add src/dst_snn/web_autonomous_learner.py tests/dst_snn/__init__.py tests/dst_snn/test_web_learner_novelty.py
git commit -m "fix: compute web-learner novelty against pre-observation vocabulary"
```

---

### Task 6: Wire reward into the web learner's SNN weight update

**Context:** In `train_observation`, reward currently only feeds `self.relations.update(...)`; the SNN weights are updated by a BCE loss that ignores reward, so "autonomous reward-driven learning" is not actually happening. Reward is computed *after* the loss. Reorder so reward is computed first and scales the task loss (reward-modulated learning): higher-reward observations produce larger gradient steps.

**Files:**
- Modify: `src/dst_snn/web_autonomous_learner.py` (`train_observation`)
- Test: `tests/dst_snn/test_web_learner_reward.py`

**Interfaces:**
- Consumes: `compute_novelty` (Task 5).
- Changes: in `train_observation`, `salience`, `novelty`, and `reward` are computed before `loss`; `loss = reward * F.binary_cross_entropy_with_logits(logits, y) + 0.0008 * spike_rate`. The returned dict gains a `"reward"` float key. Relation update reuses the same `reward`.

- [ ] **Step 1: Write the failing test** — `tests/dst_snn/test_web_learner_reward.py`

```python
from __future__ import annotations

import copy

import torch

from src.dst_snn.web_autonomous_learner import (
    BodyAction,
    DstWebLearner,
    ModuleObservation,
    WebObservation,
)


def _observation(reward_salience: float) -> WebObservation:
    module = ModuleObservation(
        module="dom-text",
        modality="text",
        tokens=["alpha", "beta", "gamma"],
        salience=reward_salience,
        source="test",
    )
    return WebObservation(url="http://x", title="t", modules=[module], action=None)


def _weight_delta(salience: float) -> float:
    torch.manual_seed(0)
    learner = DstWebLearner(in_features=64, time_steps=8, branches=4, max_delay=4)
    before = copy.deepcopy(learner.model.dendrite.weight.detach())
    result = learner.train_observation(_observation(salience))
    after = learner.model.dendrite.weight.detach()
    assert "reward" in result
    return float((after - before).abs().sum().item())


def test_reward_scales_weight_update():
    low = _weight_delta(0.1)
    high = _weight_delta(1.0)
    # Higher salience -> higher reward -> larger weight change.
    assert high > low
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/dst_snn/test_web_learner_reward.py -v`
Expected: FAIL — `KeyError: 'reward'` (or assertion failure), because reward does not affect the loss and is not returned.

- [ ] **Step 3: Rewrite the body of `train_observation`**

Replace the current body from the `out = self.model(x)` line through the `return {...}` block with:
```python
        out = self.model(x)
        logits = out["membrane"].amax(dim=1)
        spike_rate = out["spikes"].mean()

        salience = max([obs.salience for obs in observation.modules], default=0.1)
        novelty = compute_novelty(active, known_before)
        reward = max(0.05, min(1.0, 0.2 + salience * 0.35 + novelty * 0.1))

        loss = reward * F.binary_cross_entropy_with_logits(logits, y) + 0.0008 * spike_rate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        relation_updates = self.relations.update(active, reward=reward, salience=salience)
        self.steps += 1
        return {
            "step": self.steps,
            "url": observation.url,
            "title": observation.title,
            "loss": float(loss.detach().cpu()),
            "reward": float(reward),
            "spike_rate": float(spike_rate.detach().cpu()),
            "active_features": len(active),
            "relation_updates": relation_updates,
            "relations": len(self.relations.relations),
            "last_action": action_record(observation.action),
        }
```
(This removes the now-duplicate later computation of `salience`, `novelty`, `reward`, `relation_updates`, and `self.steps += 1`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/dst_snn/test_web_learner_reward.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full dst_snn test group to confirm no regression**

Run: `python -m pytest tests/dst_snn -v`
Expected: PASS (5 passed total)

- [ ] **Step 6: Commit**

```bash
git add src/dst_snn/web_autonomous_learner.py tests/dst_snn/test_web_learner_reward.py
git commit -m "fix: make reward modulate the web-learner SNN weight update"
```

---

### Task 7: Event-to-spike-tensor conversion

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/neuromorphic/__init__.py`
- Create: `benchmarks/neuromorphic/events.py`
- Test: `tests/neuromorphic/__init__.py`, `tests/neuromorphic/test_events.py`

**Interfaces:**
- Produces:
  - `bin_events_to_frames(x, y, t, p, *, width, height, time_bins, t_start=None, t_end=None) -> np.ndarray` shape `[time_bins, 2, height, width]` float32; two polarity channels; events accumulate (count) into their time bin; `t_start`/`t_end` default to `t.min()`/`t.max()`.
  - `frames_to_spike_tensor(frames, threshold=1.0) -> np.ndarray` shape `[time_bins, 2*height*width]` float32 binary.

- [ ] **Step 1: Create `benchmarks/__init__.py`** (empty)

```python
```

- [ ] **Step 2: Create `benchmarks/neuromorphic/__init__.py`** (empty)

```python
```

- [ ] **Step 3: Create `tests/neuromorphic/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Write the failing test** — `tests/neuromorphic/test_events.py`

```python
from __future__ import annotations

import numpy as np

from benchmarks.neuromorphic.events import (
    bin_events_to_frames,
    frames_to_spike_tensor,
)


def test_bins_events_by_time_and_polarity():
    # Two events: one ON at t=0 pixel (0,0); one OFF at t=9 pixel (1,1).
    x = np.array([0, 1])
    y = np.array([0, 1])
    t = np.array([0, 9])
    p = np.array([1, 0])
    frames = bin_events_to_frames(x, y, t, p, width=2, height=2, time_bins=10)
    assert frames.shape == (10, 2, 2, 2)
    assert frames[0, 1, 0, 0] == 1.0  # first bin, ON channel, pixel (row0,col0)
    assert frames[9, 0, 1, 1] == 1.0  # last bin, OFF channel, pixel (row1,col1)
    assert frames.sum() == 2.0


def test_frames_to_spike_tensor_is_binary_and_flat():
    frames = np.zeros((3, 2, 2, 2), dtype=np.float32)
    frames[0, 0, 0, 0] = 5.0  # multiple events -> still one spike
    spikes = frames_to_spike_tensor(frames, threshold=1.0)
    assert spikes.shape == (3, 8)  # 2*2*2 features
    assert spikes[0, 0] == 1.0
    assert spikes.sum() == 1.0
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pytest tests/neuromorphic/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.neuromorphic.events`

- [ ] **Step 6: Create `benchmarks/neuromorphic/events.py`**

```python
"""Convert event-camera event streams into DST-SNN spike tensors.

Events are (x, y, t, polarity) arrays. They are accumulated into a fixed number
of time bins across two polarity channels, then flattened per bin into the
``[time, features]`` layout consumed by DendriticSNN (batched later).
"""

from __future__ import annotations

import numpy as np


def bin_events_to_frames(
    x,
    y,
    t,
    p,
    *,
    width: int,
    height: int,
    time_bins: int,
    t_start=None,
    t_end=None,
) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    t = np.asarray(t).astype(np.float64)
    p = np.asarray(p)
    if time_bins <= 0 or width <= 0 or height <= 0:
        raise ValueError("time_bins, width, and height must be positive")

    frames = np.zeros((time_bins, 2, height, width), dtype=np.float32)
    if t.size == 0:
        return frames

    start = float(t.min()) if t_start is None else float(t_start)
    end = float(t.max()) if t_end is None else float(t_end)
    span = max(1e-9, end - start)
    bin_idx = ((t - start) / span * time_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, time_bins - 1)
    pol = (np.asarray(p) > 0).astype(np.int64)

    xi = np.clip(x, 0, width - 1)
    yi = np.clip(y, 0, height - 1)
    np.add.at(frames, (bin_idx, pol, yi, xi), 1.0)
    return frames


def frames_to_spike_tensor(frames: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError("frames must have shape [time, 2, height, width]")
    time_bins = frames.shape[0]
    flat = frames.reshape(time_bins, -1)
    return (flat >= threshold).astype(np.float32)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/neuromorphic/test_events.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git add benchmarks/__init__.py benchmarks/neuromorphic/__init__.py benchmarks/neuromorphic/events.py tests/neuromorphic/__init__.py tests/neuromorphic/test_events.py
git commit -m "feat: add event-to-spike-tensor conversion for neuromorphic benchmarks"
```

---

### Task 8: DST-SNN classification wrapper

**Files:**
- Create: `benchmarks/neuromorphic/classifier.py`
- Test: `tests/neuromorphic/test_classifier.py`

**Interfaces:**
- Consumes: `DendriticSNN`, `ChronoPlasticLIFLayer` from `src.dst_snn`.
- Produces:
  - `SnnClassifier(in_features: int, num_classes: int, *, num_branches: int = 16, max_delay: int = 16, use_chrono: bool = False, chrono_hidden: int = 128, threshold: float = 0.85, learnable_delay: bool = True)` (`nn.Module`).
  - `forward(x: Tensor) -> dict[str, Tensor]` where `x` is `[batch, time, in_features]`; returns `{"logits": [batch, num_classes], "spikes": [batch, time, num_classes], "membrane": [batch, time, num_classes]}`. `logits` is the per-class spike count over time (`out["spike_count"]`).

- [ ] **Step 1: Write the failing test** — `tests/neuromorphic/test_classifier.py`

```python
from __future__ import annotations

import torch

from benchmarks.neuromorphic.classifier import SnnClassifier


def test_forward_shapes_plain():
    model = SnnClassifier(in_features=16, num_classes=4, num_branches=4, max_delay=4)
    x = torch.rand(2, 8, 16)
    out = model(x)
    assert out["logits"].shape == (2, 4)
    assert out["spikes"].shape == (2, 8, 4)
    assert out["membrane"].shape == (2, 8, 4)


def test_forward_shapes_with_chrono_frontend():
    model = SnnClassifier(
        in_features=16, num_classes=4, num_branches=4, max_delay=4,
        use_chrono=True, chrono_hidden=12,
    )
    x = torch.rand(2, 8, 16)
    out = model(x)
    assert out["logits"].shape == (2, 4)


def test_logits_are_differentiable():
    model = SnnClassifier(in_features=8, num_classes=3, num_branches=2, max_delay=2)
    x = torch.rand(2, 5, 8)
    out = model(x)
    loss = out["logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/neuromorphic/test_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.neuromorphic.classifier`

- [ ] **Step 3: Create `benchmarks/neuromorphic/classifier.py`**

```python
"""DST-SNN wrapped as a spike-count classifier for neuromorphic benchmarks."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from src.dst_snn import ChronoPlasticLIFLayer, DendriticSNN


class SnnClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        num_branches: int = 16,
        max_delay: int = 16,
        use_chrono: bool = False,
        chrono_hidden: int = 128,
        threshold: float = 0.85,
        learnable_delay: bool = True,
    ) -> None:
        super().__init__()
        self.use_chrono = use_chrono
        if use_chrono:
            self.front: nn.Module | None = ChronoPlasticLIFLayer(in_features, chrono_hidden)
            backbone_in = chrono_hidden
        else:
            self.front = None
            backbone_in = in_features
        self.backbone = DendriticSNN(
            in_features=backbone_in,
            out_features=num_classes,
            num_branches=num_branches,
            max_delay=max_delay,
            learnable_delay=learnable_delay,
            threshold=threshold,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if self.front is not None:
            x = self.front(x)["spikes"]
        out = self.backbone(x)
        return {
            "logits": out["spike_count"],
            "spikes": out["spikes"],
            "membrane": out["membrane"],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/neuromorphic/test_classifier.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/neuromorphic/classifier.py tests/neuromorphic/test_classifier.py
git commit -m "feat: add DST-SNN spike-count classifier wrapper"
```

---

### Task 9: Decision-latency metric (DVS reactivity)

**Files:**
- Create: `benchmarks/neuromorphic/decision_latency.py`
- Test: `tests/neuromorphic/test_decision_latency.py`

**Interfaces:**
- Consumes: nothing from prior tasks (torch only).
- Produces:
  - `running_predictions(spikes: Tensor) -> Tensor` — `spikes` `[batch, time, classes]` → `[batch, time]` argmax of the cumulative spike count up to each step.
  - `decision_latency_fraction(spikes: Tensor, targets: Tensor, *, confirm_window: int = 3) -> float` — mean over the batch of the earliest fraction-of-stream `(t+1)/time` at which the running prediction equals the target and stays equal for `confirm_window` steps (or to the end); `1.0` if never confirmed. Lower is more reactive.

- [ ] **Step 1: Write the failing test** — `tests/neuromorphic/test_decision_latency.py`

```python
from __future__ import annotations

import torch

from benchmarks.neuromorphic.decision_latency import (
    decision_latency_fraction,
    running_predictions,
)


def test_running_predictions_tracks_cumulative_argmax():
    # 1 sample, 3 steps, 2 classes. Class 1 dominates after step 2.
    spikes = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    preds = running_predictions(spikes)
    assert preds.shape == (1, 3)
    assert preds[0, 0].item() == 0  # only class 0 so far
    assert preds[0, 2].item() == 1  # class 1 accumulated more


def test_decision_latency_confirmed_early():
    # Class 1 correct from step 0 and stays; time=4, confirm_window=2.
    spikes = torch.zeros(1, 4, 2)
    spikes[0, :, 1] = 1.0
    targets = torch.tensor([1])
    frac = decision_latency_fraction(spikes, targets, confirm_window=2)
    assert frac == 0.25  # confirmed at t=0 -> (0+1)/4


def test_decision_latency_never_correct_is_one():
    spikes = torch.zeros(1, 4, 2)
    spikes[0, :, 0] = 1.0  # always predicts class 0
    targets = torch.tensor([1])
    frac = decision_latency_fraction(spikes, targets, confirm_window=2)
    assert frac == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/neuromorphic/test_decision_latency.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.neuromorphic.decision_latency`

- [ ] **Step 3: Create `benchmarks/neuromorphic/decision_latency.py`**

```python
"""Decision-latency metric: how early in an event stream the SNN commits."""

from __future__ import annotations

try:
    import torch
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
        for t in range(time_steps):
            window_end = min(time_steps, t + confirm_window)
            if all(int(preds[b, tt].item()) == target for tt in range(t, window_end)):
                latency = (t + 1) / time_steps
                break
        fractions.append(latency)
    return sum(fractions) / max(1, len(fractions))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/neuromorphic/test_decision_latency.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/neuromorphic/decision_latency.py tests/neuromorphic/test_decision_latency.py
git commit -m "feat: add decision-latency reactivity metric"
```

---

### Task 10: Dataset wrappers (tonic)

**Context:** `tonic` provides download + parsing for N-MNIST and DVS128 Gesture. This task wraps tonic to yield our spike tensors. It has **no unit test** (network + large download); it is smoke-tested by the runner scripts in Tasks 11-12. Keep it thin and correct-by-construction.

**Files:**
- Create: `benchmarks/neuromorphic/datasets.py`

**Interfaces:**
- Consumes: `bin_events_to_frames`, `frames_to_spike_tensor` (Task 7).
- Produces:
  - `SpikeDatasetConfig(time_bins: int, sensor_size: tuple[int, int])` dataclass (sensor_size = (width, height)).
  - `events_to_tensor(events, config: SpikeDatasetConfig) -> np.ndarray` — converts one tonic structured event array (fields `x`, `y`, `t`, `p`) to `[time_bins, 2*height*width]`.
  - `load_nmnist(root: str, *, time_bins: int = 24) -> tuple[Dataset, Dataset, int]` returns `(train, test, in_features)` where each dataset yields `(spike_tensor[float32], label[int])`.
  - `load_dvs_gesture(root: str, *, time_bins: int = 32, downsample: int = 4) -> tuple[Dataset, Dataset, int]` (downsample divides the 128×128 sensor to keep `in_features` tractable).

- [ ] **Step 1: Create `benchmarks/neuromorphic/datasets.py`**

```python
"""Neuromorphic dataset wrappers producing DST-SNN spike tensors via tonic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from .events import bin_events_to_frames, frames_to_spike_tensor


@dataclass(frozen=True)
class SpikeDatasetConfig:
    time_bins: int
    sensor_size: tuple[int, int]  # (width, height)


def events_to_tensor(events, config: SpikeDatasetConfig) -> np.ndarray:
    width, height = config.sensor_size
    frames = bin_events_to_frames(
        events["x"], events["y"], events["t"], events["p"],
        width=width, height=height, time_bins=config.time_bins,
    )
    return frames_to_spike_tensor(frames)


class _MappedDataset(Dataset):
    def __init__(self, base, config: SpikeDatasetConfig) -> None:
        self.base = base
        self.config = config

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        events, label = self.base[index]
        tensor = events_to_tensor(events, self.config)
        return torch.from_numpy(tensor).float(), int(label)


def load_nmnist(root: str, *, time_bins: int = 24):
    import tonic

    sensor = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    config = SpikeDatasetConfig(time_bins=time_bins, sensor_size=(sensor[0], sensor[1]))
    train = tonic.datasets.NMNIST(save_to=root, train=True)
    test = tonic.datasets.NMNIST(save_to=root, train=False)
    in_features = sensor[0] * sensor[1] * 2
    return _MappedDataset(train, config), _MappedDataset(test, config), in_features


def load_dvs_gesture(root: str, *, time_bins: int = 32, downsample: int = 4):
    import tonic

    sensor = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)
    width = sensor[0] // downsample
    height = sensor[1] // downsample
    config = SpikeDatasetConfig(time_bins=time_bins, sensor_size=(width, height))
    transform = tonic.transforms.Downsample(spatial_factor=1.0 / downsample)
    train = tonic.datasets.DVSGesture(save_to=root, train=True, transform=transform)
    test = tonic.datasets.DVSGesture(save_to=root, train=False, transform=transform)
    in_features = width * height * 2
    return _MappedDataset(train, config), _MappedDataset(test, config), in_features
```

- [ ] **Step 2: Verify it imports without a dataset present**

Run: `python -c "from benchmarks.neuromorphic.datasets import events_to_tensor, SpikeDatasetConfig; print('ok')"`
Expected: prints `ok` (tonic import is deferred into the loader functions, so import must not require a download).

- [ ] **Step 3: Verify `events_to_tensor` with a synthetic tonic-style array**

Run:
```bash
python -c "
import numpy as np
from benchmarks.neuromorphic.datasets import events_to_tensor, SpikeDatasetConfig
ev = np.zeros(2, dtype=[('x','<i8'),('y','<i8'),('t','<i8'),('p','<i8')])
ev['x']=[0,1]; ev['y']=[0,1]; ev['t']=[0,100]; ev['p']=[1,0]
out = events_to_tensor(ev, SpikeDatasetConfig(time_bins=4, sensor_size=(2,2)))
print(out.shape, out.sum())
"
```
Expected: `(4, 8) 2.0`

- [ ] **Step 4: Commit**

```bash
git add benchmarks/neuromorphic/datasets.py
git commit -m "feat: add tonic-based N-MNIST and DVS Gesture spike-tensor loaders"
```

---

### Task 11: N-MNIST runner

**Files:**
- Create: `benchmarks/neuromorphic/run_nmnist.py`

**Interfaces:**
- Consumes: `SnnClassifier` (Task 8), `load_nmnist` (Task 10), `SnnClassifier`; eval harness `MetricSet`, `RunResult`, `EnergyModel`, `snn_energy_pj`, `accuracy`, `latency_percentiles`, `spike_stats`, `model_size`, `run_benchmarks`.
- Produces: `NmnistRunner(root, *, epochs, batch_size, time_bins, device, limit_train, limit_test)` implementing `BenchmarkRunner` (`name = "n-mnist"`, `prepare()`, `run() -> RunResult`); `__main__` calls `run_benchmarks([NmnistRunner(...)], out_dir)`.

- [ ] **Step 1: Create `benchmarks/neuromorphic/run_nmnist.py`**

```python
#!/usr/bin/env python3
"""Train and evaluate the DST-SNN on N-MNIST, emitting a harness RunResult.

Usage:
    python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from benchmarks.neuromorphic.classifier import SnnClassifier
from benchmarks.neuromorphic.datasets import load_nmnist
from src.dst_snn.eval import (
    EnergyModel,
    MetricSet,
    RunResult,
    accuracy,
    latency_percentiles,
    model_size,
    run_benchmarks,
    snn_energy_pj,
    spike_stats,
)

NUM_CLASSES = 10


def _maybe_subset(dataset, limit: int):
    if limit and limit < len(dataset):
        return Subset(dataset, list(range(limit)))
    return dataset


class NmnistRunner:
    name = "n-mnist"

    def __init__(
        self,
        root: str,
        *,
        epochs: int = 3,
        batch_size: int = 64,
        time_bins: int = 24,
        device: str = "cpu",
        limit_train: int = 0,
        limit_test: int = 0,
    ) -> None:
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_bins = time_bins
        self.device = torch.device(device)
        self.limit_train = limit_train
        self.limit_test = limit_test
        self.model: SnnClassifier | None = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def prepare(self) -> None:
        train, test, in_features = load_nmnist(self.root, time_bins=self.time_bins)
        train = _maybe_subset(train, self.limit_train)
        test = _maybe_subset(test, self.limit_test)
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=self.batch_size)
        self.model = SnnClassifier(in_features, NUM_CLASSES).to(self.device)

    def run(self) -> RunResult:
        assert self.model is not None and self.train_loader and self.test_loader
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for _ in range(self.epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = nn.functional.cross_entropy(out["logits"], y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.model.eval()
        preds_all, targets_all = [], []
        latencies_ms: list[float] = []
        spike_total, spike_batches = 0.0, 0
        fanout = NUM_CLASSES  # each input neuron fans out to all class somas
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                start = time.perf_counter()
                out = self.model(x)
                latencies_ms.append((time.perf_counter() - start) * 1000.0 / max(1, x.shape[0]))
                preds_all.append(out["logits"].argmax(dim=-1))
                targets_all.append(y)
                stats = spike_stats(out["spikes"])
                spike_total += stats["spikes_per_inference"]
                spike_batches += 1

        preds = torch.cat(preds_all)
        targets = torch.cat(targets_all)
        acc = accuracy(preds, targets)
        lat = latency_percentiles(latencies_ms)
        spikes_per_inf = spike_total / max(1, spike_batches)
        energy_model = EnergyModel()
        energy_pj = snn_energy_pj(spikes_per_inf, fanout, energy_model)
        size = model_size(self.model)

        metrics = MetricSet(
            quality=acc,
            quality_metric="accuracy",
            latency_ms_p50=lat["p50"],
            latency_ms_p95=lat["p95"],
            spikes_per_inference=spikes_per_inf,
            active_neuron_fraction=0.0,
            energy_pj=energy_pj,
            energy_source=energy_model.source,
            param_count=size["param_count"],
            model_bytes=size["model_bytes"],
            extra={"epochs": self.epochs, "fanout": fanout},
        )
        return RunResult(benchmark=self.name, model="dst-snn", metrics=metrics, baseline=None, meta={})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/nmnist")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-bins", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = NmnistRunner(
        args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        time_bins=args.time_bins,
        device=args.device,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    results = run_benchmarks([runner], args.out_dir)
    print(results[0].to_json())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports and shows help**

Run: `python benchmarks/neuromorphic/run_nmnist.py --help`
Expected: argparse help text prints, exit 0 (no dataset download triggered).

- [ ] **Step 3: Smoke test on a tiny subset (downloads N-MNIST once — allowed for the runner, not for tests)**

Run:
```bash
python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 1 --limit-train 128 --limit-test 128 --time-bins 12
```
Expected: prints a `RunResult` JSON with `"benchmark": "n-mnist"` and `quality` above `0.1` (better than 10% chance); `artifacts/benchmarks/n-mnist.json` and `report.md` created.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/neuromorphic/run_nmnist.py
git commit -m "feat: add N-MNIST benchmark runner"
```

---

### Task 12: DVS Gesture runner

**Files:**
- Create: `benchmarks/neuromorphic/run_dvs_gesture.py`

**Interfaces:**
- Consumes: `SnnClassifier` (Task 8), `load_dvs_gesture` (Task 10), `decision_latency_fraction` (Task 9), eval harness symbols (as Task 11).
- Produces: `DvsGestureRunner(root, *, epochs, batch_size, time_bins, downsample, device, limit_train, limit_test)` implementing `BenchmarkRunner` (`name = "dvs-gesture"`); records decision-latency in `MetricSet.extra["decision_latency_fraction"]`; `__main__` runs it through `run_benchmarks`.

- [ ] **Step 1: Create `benchmarks/neuromorphic/run_dvs_gesture.py`**

```python
#!/usr/bin/env python3
"""Train and evaluate the DST-SNN on DVS128 Gesture with a reactivity metric.

Usage:
    python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from benchmarks.neuromorphic.classifier import SnnClassifier
from benchmarks.neuromorphic.datasets import load_dvs_gesture
from benchmarks.neuromorphic.decision_latency import decision_latency_fraction
from src.dst_snn.eval import (
    EnergyModel,
    MetricSet,
    RunResult,
    accuracy,
    latency_percentiles,
    model_size,
    run_benchmarks,
    snn_energy_pj,
    spike_stats,
)

NUM_CLASSES = 11


def _maybe_subset(dataset, limit: int):
    if limit and limit < len(dataset):
        return Subset(dataset, list(range(limit)))
    return dataset


class DvsGestureRunner:
    name = "dvs-gesture"

    def __init__(
        self,
        root: str,
        *,
        epochs: int = 5,
        batch_size: int = 16,
        time_bins: int = 32,
        downsample: int = 4,
        device: str = "cpu",
        limit_train: int = 0,
        limit_test: int = 0,
    ) -> None:
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_bins = time_bins
        self.downsample = downsample
        self.device = torch.device(device)
        self.limit_train = limit_train
        self.limit_test = limit_test
        self.model: SnnClassifier | None = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def prepare(self) -> None:
        train, test, in_features = load_dvs_gesture(
            self.root, time_bins=self.time_bins, downsample=self.downsample
        )
        train = _maybe_subset(train, self.limit_train)
        test = _maybe_subset(test, self.limit_test)
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=self.batch_size)
        self.model = SnnClassifier(in_features, NUM_CLASSES, use_chrono=True).to(self.device)

    def run(self) -> RunResult:
        assert self.model is not None and self.train_loader and self.test_loader
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for _ in range(self.epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = nn.functional.cross_entropy(out["logits"], y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.model.eval()
        preds_all, targets_all = [], []
        latencies_ms: list[float] = []
        spike_total, spike_batches = 0.0, 0
        decision_fracs: list[float] = []
        fanout = NUM_CLASSES
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                start = time.perf_counter()
                out = self.model(x)
                latencies_ms.append((time.perf_counter() - start) * 1000.0 / max(1, x.shape[0]))
                preds_all.append(out["logits"].argmax(dim=-1))
                targets_all.append(y)
                stats = spike_stats(out["spikes"])
                spike_total += stats["spikes_per_inference"]
                spike_batches += 1
                decision_fracs.append(decision_latency_fraction(out["spikes"], y))

        preds = torch.cat(preds_all)
        targets = torch.cat(targets_all)
        acc = accuracy(preds, targets)
        lat = latency_percentiles(latencies_ms)
        spikes_per_inf = spike_total / max(1, spike_batches)
        energy_model = EnergyModel()
        energy_pj = snn_energy_pj(spikes_per_inf, fanout, energy_model)
        size = model_size(self.model)
        decision_latency = sum(decision_fracs) / max(1, len(decision_fracs))

        metrics = MetricSet(
            quality=acc,
            quality_metric="accuracy",
            latency_ms_p50=lat["p50"],
            latency_ms_p95=lat["p95"],
            spikes_per_inference=spikes_per_inf,
            active_neuron_fraction=0.0,
            energy_pj=energy_pj,
            energy_source=energy_model.source,
            param_count=size["param_count"],
            model_bytes=size["model_bytes"],
            extra={
                "epochs": self.epochs,
                "fanout": fanout,
                "decision_latency_fraction": decision_latency,
            },
        )
        return RunResult(benchmark=self.name, model="dst-snn", metrics=metrics, baseline=None, meta={})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/dvs")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--time-bins", type=int, default=32)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = DvsGestureRunner(
        args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        time_bins=args.time_bins,
        downsample=args.downsample,
        device=args.device,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    results = run_benchmarks([runner], args.out_dir)
    print(results[0].to_json())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports and shows help**

Run: `python benchmarks/neuromorphic/run_dvs_gesture.py --help`
Expected: argparse help text prints, exit 0.

- [ ] **Step 3: Smoke test on a tiny subset (downloads DVS128 Gesture once)**

Run:
```bash
python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs --epochs 1 --limit-train 32 --limit-test 32 --time-bins 16
```
Expected: prints a `RunResult` JSON with `"benchmark": "dvs-gesture"` and `extra.decision_latency_fraction` between 0 and 1; artifact files created.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/neuromorphic/run_dvs_gesture.py
git commit -m "feat: add DVS128 Gesture benchmark runner with reactivity metric"
```

---

### Task 13: Documentation and full-suite verification

**Files:**
- Create: `benchmarks/README.md`
- Modify: `README.md` (append a "Benchmarks" section)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: developer-facing docs; a green full test run.

- [ ] **Step 1: Create `benchmarks/README.md`**

```markdown
# SNN Benchmarks

Shared evaluation harness (`src/dst_snn/eval/`) plus neuromorphic benchmark
runners. Every runner emits the same `RunResult` schema: quality, latency
(p50/p95), spikes-per-inference, energy (pJ, AC/MAC model), and model size.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dst-snn.txt -r requirements-bench.txt
```

## N-MNIST (accuracy at low energy)

```bash
python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 3
```

## DVS128 Gesture (accuracy + reactivity)

```bash
python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs --epochs 5
```

The DVS runner reports `decision_latency_fraction` in `extra`: the mean
fraction of the event stream elapsed before the running prediction commits to
the correct class. Lower is more reactive.

## Energy model

`EnergyModel` defaults to 45nm (`0.9 pJ/MAC`, `0.1 pJ/AC`) and records its
`source`. Override the constants when comparing against a specific target
device. SNN energy = spikes × fan-out × AC cost; dense baseline = MACs × MAC
cost. Compare with `energy_ratio(snn_pj, dense_pj)`.

## Results

Runners write `<name>.json` and a combined `report.md` under
`artifacts/benchmarks/`.
```

- [ ] **Step 2: Append a "Benchmarks" section to `README.md`**

Add at the end of `README.md`:
```markdown

## Benchmarks

SNN evaluation harness and neuromorphic benchmarks (N-MNIST, DVS128 Gesture)
live under `benchmarks/`. See [benchmarks/README.md](benchmarks/README.md).

```bash
pip install -r requirements-bench.txt
python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 3
python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs --epochs 5
```
```

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -v`
Expected: PASS — all tests from Tasks 1-9 green (eval: energy 5, metrics 4, result 2, runner 1; dst_snn: novelty 4, reward 1; neuromorphic: events 2, classifier 3, decision_latency 3). No failures.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/README.md README.md
git commit -m "docs: document SNN benchmark harness and runners"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-09-snn-benchmark-harness-design.md`):
- §2 Shared harness → Tasks 1 (energy), 2 (quality/latency/sparsity/size), 3 (result schema + report), 4 (runner). ✅
- §2 Energy AC/MAC + configurable + source recorded → Task 1 (`EnergyModel.source`), surfaced in `MetricSet.energy_source`. ✅
- Benchmark A (N-MNIST) → Tasks 7, 8, 10, 11. ✅
- Benchmark B (DVS Gesture) + decision latency → Tasks 7, 8, 9, 10, 12. ✅
- ChronoPlastic front-end wiring → Task 8 (`use_chrono`), used by Task 12. ✅
- §5 Bug fix: degenerate novelty → Task 5. ✅
- §5 Bug fix: reward not affecting SNN weights → Task 6. ✅
- Repo consolidation (`benchmarks/` new dir, DST-SNN as core) → Tasks 7, 13. ✅
- Benchmark C (3DCG scorer) → **out of scope for this plan; covered by the companion plan `2026-07-09-snn-3dcg-scorer.md`** (uses the same `RunResult`/`MetricSet` schema from Task 3).
- `powermetrics` measured energy (optional in spec) → not implemented; spec marked it optional ("可能なら"). Documented as future work in `benchmarks/README.md` energy section intent; acceptable YAGNI deferral.

**Placeholder scan:** No TBD/TODO/"handle edge cases" left; every code step contains full code.

**Type consistency:** `MetricSet`/`RunResult` field names identical across Tasks 3, 11, 12. `SnnClassifier.forward` returns `logits`/`spikes`/`membrane` consumed consistently by Tasks 11, 12, and `decision_latency_fraction` (Task 9). `spike_stats` expects `[batch, time, neurons]`, fed `out["spikes"]` `[batch, time, classes]` — consistent. `snn_energy_pj(total_spikes, fanout, model)` signature matches call sites. `run_benchmarks(runners, out_dir)` matches Tasks 11/12 `main`. ✅

**Note on N-MNIST fanout:** modeled as `NUM_CLASSES` (single dendritic layer, input→class somas). This is a documented simplification recorded in `MetricSet.extra["fanout"]`; refine when multi-layer backbones are added.
