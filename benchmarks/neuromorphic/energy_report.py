"""Shared energy packing helpers for neuromorphic runners."""

from __future__ import annotations

from src.dst_snn.eval.energy import (
    EnergyModel,
    dense_mac_energy_pj,
    energy_ratio,
    estimate_snn_classifier_ops,
    snn_energy_pj,
)


def pack_snn_energy(
    *,
    in_features: int,
    num_classes: int,
    time_bins: int,
    spikes_per_inference: float,
    hidden_features: int = 0,
    chrono_hidden: int = 0,
    energy_model: EnergyModel | None = None,
) -> dict[str, float | str]:
    model = energy_model or EnergyModel()
    ops = estimate_snn_classifier_ops(
        in_features=in_features,
        num_classes=num_classes,
        time_bins=time_bins,
        hidden_features=hidden_features,
        chrono_hidden=chrono_hidden if chrono_hidden else 0,
        spikes_per_inference=spikes_per_inference,
    )
    snn_pj = snn_energy_pj(ops["snn_spike_events"], int(round(ops["effective_fanout"])), model)
    dense_pj = dense_mac_energy_pj(ops["dense_mac_ops"], model)
    return {
        "energy_pj": snn_pj,
        "energy_source": model.source,
        "fanout": ops["effective_fanout"],
        "dense_mac_ops": ops["dense_mac_ops"],
        "dense_energy_pj": dense_pj,
        "energy_ratio_dense_over_snn": energy_ratio(snn_pj, dense_pj),
        "layer_count": ops["layer_count"],
    }
