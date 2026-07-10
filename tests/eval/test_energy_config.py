from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dst_snn.eval.energy import EnergyModel, dense_mac_energy_pj, snn_energy_pj


def test_from_mapping_override():
    m = EnergyModel.from_mapping({"mac_pj": 1.0, "ac_pj": 0.2, "source": "unit-test"})
    assert m.mac_pj == 1.0
    assert m.ac_pj == 0.2
    assert m.source == "unit-test"
    assert snn_energy_pj(10, 2, m) == 4.0
    assert dense_mac_energy_pj(10, m) == 10.0


def test_from_json_file(tmp_path: Path):
    path = tmp_path / "energy.json"
    path.write_text(json.dumps({"mac_pj": 0.5, "ac_pj": 0.05}), encoding="utf-8")
    m = EnergyModel.from_json_file(path)
    assert m.mac_pj == 0.5
    assert m.ac_pj == 0.05
    assert "energy.json" in m.source or "config file" in m.source


def test_from_mapping_rejects_negative():
    with pytest.raises(ValueError):
        EnergyModel.from_mapping({"mac_pj": -1.0})
