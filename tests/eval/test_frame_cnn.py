from __future__ import annotations

import torch

from src.dst_snn.eval.baselines import FrameCnnClassifier, train_frame_cnn
from src.dst_snn.eval.metrics import accuracy
from torch.utils.data import DataLoader, TensorDataset


def test_frame_cnn_forward_and_macs():
    model = FrameCnnClassifier(in_channels=2, num_classes=4, channels=(8, 16, 16))
    x = torch.rand(2, 5, 2, 16, 16)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert model.mac_ops_per_inference(5, 16, 16) > 0


def test_frame_cnn_learns_trivial_pattern():
    torch.manual_seed(0)
    model = FrameCnnClassifier(in_channels=2, num_classes=2, channels=(8, 16, 16))
    xs, ys = [], []
    for i in range(32):
        label = i % 2
        x = torch.zeros(4, 2, 16, 16)
        if label == 0:
            x[:, 0, 0:4, 0:4] = 1.0
        else:
            x[:, 1, 12:16, 12:16] = 1.0
        xs.append(x)
        ys.append(label)
    x = torch.stack(xs)
    y = torch.tensor(ys)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    train_frame_cnn(model, loader, epochs=25, device=torch.device("cpu"), lr=5e-3)
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    assert accuracy(preds, y) >= 0.85
