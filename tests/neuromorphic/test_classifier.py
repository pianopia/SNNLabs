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
        in_features=16,
        num_classes=4,
        num_branches=4,
        max_delay=4,
        use_chrono=True,
        chrono_hidden=12,
    )
    x = torch.rand(2, 8, 16)
    out = model(x)
    assert out["logits"].shape == (2, 4)


def test_forward_shapes_with_hidden_spiking_layer():
    model = SnnClassifier(
        in_features=16,
        num_classes=4,
        num_branches=4,
        max_delay=4,
        hidden_features=10,
        hidden_threshold=0.4,
    )
    x = torch.rand(2, 8, 16)
    out = model(x)
    assert out["logits"].shape == (2, 4)
    assert out["spikes"].shape == (2, 8, 4)
    assert out["hidden_spikes"] is not None
    assert out["hidden_spikes"].shape == (2, 8, 10)


def test_forward_shapes_with_hidden_membrane_path():
    model = SnnClassifier(
        in_features=16,
        num_classes=4,
        num_branches=4,
        max_delay=4,
        hidden_features=10,
        hidden_output="membrane",
    )
    x = torch.rand(2, 8, 16)
    out = model(x)
    assert out["logits"].shape == (2, 4)
    assert out["hidden_membrane"] is not None
    assert out["hidden_membrane"].shape == (2, 8, 10)


def test_logits_are_differentiable():
    model = SnnClassifier(in_features=8, num_classes=3, num_branches=2, max_delay=2)
    x = torch.rand(2, 5, 8)
    out = model(x)
    loss = out["logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_membrane_readout_modes():
    x = torch.rand(2, 5, 8)
    for readout in ["max_membrane", "mean_membrane", "spike_count"]:
        model = SnnClassifier(in_features=8, num_classes=3, num_branches=2, max_delay=2, readout=readout)
        out = model(x)
        assert out["logits"].shape == (2, 3)


def test_rejects_unknown_readout():
    try:
        SnnClassifier(in_features=8, num_classes=3, readout="unknown")
    except ValueError as exc:
        assert "readout" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_rejects_negative_hidden_features():
    try:
        SnnClassifier(in_features=8, num_classes=3, hidden_features=-1)
    except ValueError as exc:
        assert "hidden_features" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_rejects_unknown_hidden_output():
    try:
        SnnClassifier(in_features=8, num_classes=3, hidden_features=4, hidden_output="unknown")
    except ValueError as exc:
        assert "hidden_output" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
