"""Frame CNN baseline matched to Conv-PLIF topology (ReLU, no spikes).

Operates on ``[batch, time, channels, height, width]`` event frames. Processes
each time step independently through shared conv weights, mean-pools over time,
then classifies. Used as a dense ANN reference for energy/quality comparison
against Conv-PLIF under the same spatial input.
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc


class FrameCnnClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 11,
        *,
        channels: tuple[int, int, int] = (32, 64, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c3, num_classes)
        self.out_channels = c3
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError("expected [batch, time, channels, height, width]")
        batch, time_steps, c, h, w = x.shape
        flat = x.reshape(batch * time_steps, c, h, w)
        feats = self.features(flat)
        pooled = feats.mean(dim=(-2, -1))  # [B*T, C]
        pooled = pooled.view(batch, time_steps, -1).mean(dim=1)  # [B, C]
        return self.fc(self.dropout(pooled))

    def mac_ops_per_inference(self, time_bins: int, height: int, width: int) -> float:
        """Order-of-magnitude dense MAC count for the three convs + FC."""
        c1, c2, c3 = self.channels
        h2, w2 = max(1, height // 2), max(1, width // 2)
        h3, w3 = max(1, height // 4), max(1, width // 4)
        per_step = (
            float(self.in_channels * c1 * 9 * height * width)
            + float(c1 * c2 * 9 * h2 * w2)
            + float(c2 * c3 * 9 * h3 * w3)
            + float(c3 * self.num_classes)
        )
        return per_step * float(time_bins)


def train_frame_cnn(
    model: FrameCnnClassifier,
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
