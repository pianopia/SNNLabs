import numpy as np
import pytest

from src.dst_snn.foundation import (
    EarlyExitController,
    StreamingSpikingSSM,
    streaming_efficiency_report,
)


def test_streaming_chunks_match_single_run_and_state_is_constant_size():
    events = np.array([[1, 0], [0, 1], [-1, 0], [0, 0]], dtype=np.int8)
    whole = StreamingSpikingSSM(2, 4, seed=3)
    expected = whole.run(events)
    chunked = StreamingSpikingSSM(2, 4, seed=3)
    actual = np.concatenate([chunked.run(events[:2]), chunked.run(events[2:])])
    np.testing.assert_array_equal(actual, expected)
    assert chunked.state_bytes == 16


def test_early_exit_requires_confidence_and_stability():
    policy = EarlyExitController(confidence=0.8, patience=2)
    assert not policy.update(np.array([0.1, 0.9]))
    assert policy.update(np.array([0.15, 0.85]))
    policy.reset()
    assert not policy.update(np.array([0.9, 0.1]))
    assert not policy.update(np.array([0.1, 0.9]))


def test_efficiency_report_is_explicitly_estimated_and_sparse():
    events = np.array([[1, 0, 0], [0, 0, -1]], dtype=np.int8)
    report = streaming_efficiency_report(events, state_size=4)
    assert report.nonzero_events == 2
    assert report.event_sparsity == pytest.approx(2 / 3)
    assert report.estimated_dense_mac_ops == 24
    assert report.estimated_energy_ratio > 1
    assert "hardware measurement required" in report.accounting
