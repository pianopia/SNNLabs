"""Compute-energy proxy for SNN vs dense baselines.

SNN synaptic energy is modeled as accumulate (AC) operations: each spike drives
its post-synaptic fan-out as one AC each. Dense baselines are modeled as
multiply-accumulate (MAC) operations. Per-op energies default to a 45nm process
and are configurable; the source string is recorded in results.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Union


@dataclass(frozen=True)
class EnergyModel:
    mac_pj: float = 0.9
    ac_pj: float = 0.1
    source: str = "45nm defaults (Horowitz ISSCC 2014); configurable"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EnergyModel":
        """Build from a dict (e.g. JSON). Unknown keys are ignored."""
        mac = float(data["mac_pj"]) if "mac_pj" in data else 0.9
        ac = float(data["ac_pj"]) if "ac_pj" in data else 0.1
        source = str(data["source"]) if "source" in data else (
            f"config override (mac_pj={mac}, ac_pj={ac})"
        )
        if mac < 0 or ac < 0:
            raise ValueError("mac_pj and ac_pj must be non-negative")
        return cls(mac_pj=mac, ac_pj=ac, source=source)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "EnergyModel":
        """Load energy constants from a JSON file (design: config-file override)."""
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("energy config JSON must be an object")
        model = cls.from_mapping(data)
        # Prefer file path in source if caller did not set one.
        if "source" not in data:
            return cls(
                mac_pj=model.mac_pj,
                ac_pj=model.ac_pj,
                source=f"config file: {Path(path).as_posix()}",
            )
        return model


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


def dense_linear_mac_ops(in_features: int, out_features: int, *, time_steps: int = 1) -> float:
    """MAC ops for a dense linear map applied once per time step."""
    if in_features < 0 or out_features < 0 or time_steps < 0:
        raise ValueError("sizes must be non-negative")
    return float(in_features) * float(out_features) * float(time_steps)


def estimate_snn_classifier_ops(
    *,
    in_features: int,
    num_classes: int,
    time_bins: int,
    hidden_features: int = 0,
    chrono_hidden: int = 0,
    spikes_per_inference: float = 0.0,
) -> dict[str, float]:
    """Estimate AC/MAC ops and effective fanout for a layered SNN classifier.

    Dense baseline MACs model a non-spiking MLP with the same layer widths,
    evaluated at every time bin (worst-case dense temporal compute).

    SNN energy uses spikes × effective fanout. Effective fanout averages the
    post-synaptic widths of each layer so multi-layer models are not
    under-counted as ``num_classes`` alone.
    """
    layers: list[tuple[int, int]] = []
    width = in_features
    if chrono_hidden > 0:
        layers.append((width, chrono_hidden))
        width = chrono_hidden
    if hidden_features > 0:
        layers.append((width, hidden_features))
        width = hidden_features
    layers.append((width, num_classes))

    dense_macs = 0.0
    fanout_weights = 0.0
    fanout_total = 0.0
    for src, dst in layers:
        dense_macs += dense_linear_mac_ops(src, dst, time_steps=time_bins)
        fanout_weights += float(dst) * float(dst)
        fanout_total += float(dst)
    effective_fanout = fanout_weights / max(1.0, fanout_total)
    # Spikes are typically counted on the readout layer; scale by layer count so
    # hidden activity is not ignored when estimating AC cost.
    layer_scale = max(1.0, float(len(layers)))
    snn_spike_events = float(spikes_per_inference) * layer_scale
    return {
        "dense_mac_ops": dense_macs,
        "effective_fanout": effective_fanout,
        "snn_spike_events": snn_spike_events,
        "layer_count": float(len(layers)),
    }
