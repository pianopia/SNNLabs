from __future__ import annotations

from src.dst_snn.eval.powermetrics import (
    _parse_energy_uj,
    powermetrics_available,
    sample_process_window,
)


def test_parse_energy_from_sample_text():
    text = "CPU Power: 1200 mW\nGPU Power: 300 mW\n"
    uj = _parse_energy_uj(text)
    assert uj == 1_200_000.0


def test_powermetrics_available_is_bool():
    assert isinstance(powermetrics_available(), bool)


def test_sample_returns_none_or_powersample():
    # Safe call: either unavailable, failed privilege, or a sample object.
    result = sample_process_window(0.1) if powermetrics_available() else None
    if result is not None:
        assert result.source == "powermetrics"
        assert result.duration_s > 0
