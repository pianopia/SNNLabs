"""DST-SNN wrapped as a spike-count classifier for neuromorphic benchmarks."""

from __future__ import annotations

try:
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from benchmarks.neuromorphic.temporal_features import TemporalFeatureFrontEnd
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
        readout: str = "spike_count",
        hidden_features: int = 0,
        hidden_threshold: float | None = None,
        hidden_output: str = "spikes",
        use_temporal_features: bool = False,
        temporal_project_to: int = 0,
        temporal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        if readout not in {"spike_count", "max_membrane", "mean_membrane"}:
            raise ValueError("readout must be one of: spike_count, max_membrane, mean_membrane")
        if hidden_output not in {"spikes", "membrane"}:
            raise ValueError("hidden_output must be one of: spikes, membrane")
        if hidden_features < 0:
            raise ValueError("hidden_features must be non-negative")
        self.use_chrono = use_chrono
        self.readout = readout
        self.hidden_features = hidden_features
        self.hidden_output = hidden_output
        self.use_temporal_features = use_temporal_features
        backbone_in = in_features
        if use_temporal_features:
            self.temporal: nn.Module | None = TemporalFeatureFrontEnd(
                in_features,
                alpha=temporal_alpha,
                project_to=temporal_project_to,
            )
            backbone_in = self.temporal.out_features
        else:
            self.temporal = None
        if use_chrono:
            self.front: nn.Module | None = ChronoPlasticLIFLayer(backbone_in, chrono_hidden)
            backbone_in = chrono_hidden
        else:
            self.front = None
        if hidden_features:
            self.hidden: nn.Module | None = DendriticSNN(
                in_features=backbone_in,
                out_features=hidden_features,
                num_branches=num_branches,
                max_delay=max_delay,
                learnable_delay=learnable_delay,
                threshold=threshold if hidden_threshold is None else hidden_threshold,
            )
            backbone_in = hidden_features
        else:
            self.hidden = None
        self.backbone = DendriticSNN(
            in_features=backbone_in,
            out_features=num_classes,
            num_branches=num_branches,
            max_delay=max_delay,
            learnable_delay=learnable_delay,
            threshold=threshold,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if self.temporal is not None:
            x = self.temporal(x)
        if self.front is not None:
            x = self.front(x)["spikes"]
        hidden_out = None
        if self.hidden is not None:
            hidden_out = self.hidden(x)
            x = hidden_out[self.hidden_output]
        out = self.backbone(x)
        if self.readout == "max_membrane":
            logits = out["membrane"].amax(dim=1)
        elif self.readout == "mean_membrane":
            logits = out["membrane"].mean(dim=1)
        else:
            logits = out["spike_count"]
        return {
            "logits": logits,
            "spikes": out["spikes"],
            "membrane": out["membrane"],
            "hidden_spikes": hidden_out["spikes"] if hidden_out is not None else None,
            "hidden_membrane": hidden_out["membrane"] if hidden_out is not None else None,
        }
