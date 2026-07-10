from __future__ import annotations

import torch

from benchmarks.neuromorphic.conv_snn import ConvPLIFClassifier, SEWResidualBlock, SewConvPLIFClassifier
from benchmarks.neuromorphic.plif import PLIF, spike_fn


def test_plif_emits_spikes_and_is_differentiable():
    lif = PLIF(init_tau=2.0, threshold=0.5)
    x = torch.ones(2, 4, 8, 8) * 0.8
    s, v = lif(x)
    assert s.shape == x.shape
    assert v.shape == x.shape
    loss = s.sum() + v.sum()
    loss.backward()
    assert lif.w.grad is not None


def test_surrogate_spike_binary_forward():
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = spike_fn(x)
    assert y.tolist() == [0.0, 1.0, 1.0]


def test_conv_plif_forward_shapes_and_grad():
    model = ConvPLIFClassifier(in_channels=2, num_classes=5, channels=(8, 16, 16), threshold=0.5)
    x = torch.rand(2, 6, 2, 32, 32)
    out = model(x)
    assert out["logits"].shape == (2, 5)
    assert out["spikes"].shape == (2, 6, 5)
    assert out["feature_spikes"].shape[0] == 2
    loss = out["logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_sew_residual_add_and_grad():
    block = SEWResidualBlock(8, stride=1, threshold=0.5)
    x = torch.rand(2, 8, 16, 16)
    y, state = block(x)
    assert y.shape == x.shape
    assert state[0] is not None
    y.sum().backward()
    assert any(p.grad is not None for p in block.parameters())


def test_sew_classifier_forward_and_learn():
    torch.manual_seed(0)
    model = SewConvPLIFClassifier(
        in_channels=2,
        num_classes=2,
        width=8,
        blocks_per_stage=1,
        threshold=0.5,
        readout="spike_count",
    )
    x = torch.rand(2, 4, 2, 16, 16)
    out = model(x)
    assert out["logits"].shape == (2, 2)
    out["logits"].sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_conv_plif_learns_trivial_spatial_pattern():
    torch.manual_seed(0)
    model = ConvPLIFClassifier(in_channels=2, num_classes=2, channels=(8, 16, 16), threshold=0.5, readout="spike_count")
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    # Class 0: activity in top-left; class 1: bottom-right.
    for _ in range(40):
        x = torch.zeros(8, 4, 2, 16, 16)
        y = torch.zeros(8, dtype=torch.long)
        for i in range(8):
            label = i % 2
            y[i] = label
            if label == 0:
                x[i, :, 0, 0:4, 0:4] = 1.0
            else:
                x[i, :, 1, 12:16, 12:16] = 1.0
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out["logits"], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        x = torch.zeros(4, 4, 2, 16, 16)
        y = torch.tensor([0, 1, 0, 1])
        x[0, :, 0, 0:4, 0:4] = 1.0
        x[1, :, 1, 12:16, 12:16] = 1.0
        x[2, :, 0, 0:4, 0:4] = 1.0
        x[3, :, 1, 12:16, 12:16] = 1.0
        preds = model(x)["logits"].argmax(dim=-1)
    assert (preds == y).float().mean().item() >= 0.75
