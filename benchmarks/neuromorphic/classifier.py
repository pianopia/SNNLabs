"""DST-SNN wrapped as a spike-count classifier for neuromorphic benchmarks."""

from __future__ import annotations

try:
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from src.dst_snn import ChronoPlasticLIFLayer, DendriticSNN


class SnnClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        num_branches: int = 16,
        max_delay: int = 16,
        use_chrono: bool = False,
        chrono_hidden: int = 128,
        threshold: float = 0.85,
        learnable_delay: bool = True,
    ) -> None:
        super().__init__()
        self.use_chrono = use_chrono
        if use_chrono:
            self.front: nn.Module | None = ChronoPlasticLIFLayer(in_features, chrono_hidden)
            backbone_in = chrono_hidden
        else:
            self.front = None
            backbone_in = in_features
        self.backbone = DendriticSNN(
            in_features=backbone_in,
            out_features=num_classes,
            num_branches=num_branches,
            max_delay=max_delay,
            learnable_delay=learnable_delay,
            threshold=threshold,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if self.front is not None:
            x = self.front(x)["spikes"]
        out = self.backbone(x)
        return {
            "logits": out["spike_count"],
            "spikes": out["spikes"],
            "membrane": out["membrane"],
        }
