"""Comparable efficiency accounting for streaming sparse and dense paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.dst_snn.eval.energy import EnergyModel, dense_mac_energy_pj, snn_energy_pj


@dataclass(frozen=True)
class StreamingEfficiencyReport:
    steps: int
    features: int
    state_size: int
    nonzero_events: int
    event_sparsity: float
    estimated_snn_ac_ops: float
    estimated_dense_mac_ops: float
    estimated_snn_energy_pj: float
    estimated_dense_energy_pj: float
    estimated_energy_ratio: float
    state_bytes: int
    accounting: str = "estimated_ac_vs_mac_v1; hardware measurement required for claims"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def streaming_efficiency_report(
    events: np.ndarray,
    *,
    state_size: int,
    energy_model: EnergyModel | None = None,
) -> StreamingEfficiencyReport:
    source = np.asarray(events)
    if source.ndim != 2:
        raise ValueError("events must have shape [steps, features]")
    if state_size < 1:
        raise ValueError("state_size must be positive")
    steps, features = source.shape
    nonzero = int(np.count_nonzero(source))
    total = max(1, int(source.size))
    ac_ops = float(nonzero * state_size)
    dense_ops = float(steps * features * state_size)
    model = energy_model or EnergyModel()
    snn_pj = snn_energy_pj(nonzero, state_size, model)
    dense_pj = dense_mac_energy_pj(dense_ops, model)
    ratio = float("inf") if snn_pj == 0 else dense_pj / snn_pj
    return StreamingEfficiencyReport(
        steps=int(steps),
        features=int(features),
        state_size=int(state_size),
        nonzero_events=nonzero,
        event_sparsity=1.0 - nonzero / total,
        estimated_snn_ac_ops=ac_ops,
        estimated_dense_mac_ops=dense_ops,
        estimated_snn_energy_pj=snn_pj,
        estimated_dense_energy_pj=dense_pj,
        estimated_energy_ratio=ratio,
        state_bytes=int(state_size * np.dtype(np.float32).itemsize),
    )
