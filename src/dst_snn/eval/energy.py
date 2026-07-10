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
