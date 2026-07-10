"""Small dense ANN baseline for neuromorphic spike-tensor classification.

Operates on the same ``[batch, time, features]`` tensors as the SNN runners by
mean-pooling over time, then applying a 2-layer MLP. Used as an energy/quality
reference, not as a competitive event-vision SOTA model.
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


class DenseAnnClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden = hidden

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, time, features] → mean-pool time → logits
        if x.ndim != 3:
            raise ValueError("expected [batch, time, features]")
        pooled = x.mean(dim=1)
        return self.net(pooled)

    def mac_ops_per_inference(self, time_bins: int) -> float:
        # Mean-pool is free in this proxy; both linear layers run once per sample.
        # Count MACs as if features were dense at each step for a fair temporal
        # compute upper bound comparable to the SNN energy model.
        first = float(self.in_features) * float(self.hidden) * float(time_bins)
        second = float(self.hidden) * float(self.num_classes)
        return first + second


def train_ann_classifier(
    model: DenseAnnClassifier,
    loader,
    *,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
