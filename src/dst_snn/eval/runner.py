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
