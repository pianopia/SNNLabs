from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dst_snn.eval.baselines import DenseAnnClassifier, train_ann_classifier
from src.dst_snn.eval.metrics import accuracy, model_size


def test_ann_forward_and_train_step():
    model = DenseAnnClassifier(in_features=8, num_classes=3, hidden=16)
    x = torch.rand(4, 5, 8)
    y = torch.tensor([0, 1, 2, 1])
    logits = model(x)
    assert logits.shape == (4, 3)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2)
    train_ann_classifier(model, loader, epochs=1, device=torch.device("cpu"))
    size = model_size(model)
    assert size["param_count"] > 0
    assert model.mac_ops_per_inference(time_bins=5) > 0


def test_ann_can_fit_trivial_pattern():
    torch.manual_seed(0)
    model = DenseAnnClassifier(in_features=4, num_classes=2, hidden=32)
    # Strong class cue on feature 0: class 0 ≈ 0.1, class 1 ≈ 0.9.
    xs, ys = [], []
    for i in range(64):
        label = i % 2
        x = torch.rand(6, 4) * 0.05
        x[:, 0] = 0.1 + 0.8 * label
        xs.append(x)
        ys.append(label)
    x = torch.stack(xs)
    y = torch.tensor(ys)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    train_ann_classifier(model, loader, epochs=30, device=torch.device("cpu"), lr=5e-3)
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    assert accuracy(preds, y) >= 0.9
