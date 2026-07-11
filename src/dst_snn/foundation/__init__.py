"""Research-aligned primitives for a streaming multimodal SNN foundation model."""

from .events import (
    DEFAULT_MODALITY_SPECS,
    ModalitySpec,
    SignedEventEncoder,
    align_and_fuse_modalities,
    compressed_time_steps,
    temporal_compress,
)
from .streaming import EarlyExitController, StreamingSpikingSSM
from .metrics import StreamingEfficiencyReport, streaming_efficiency_report
try:  # Keep the NumPy streaming contract importable without PyTorch.
    from .torch_ssm import (
        SignedSpikingSSMBlock,
        SpikingSSMBackbone,
        SpikingSSMOutput,
        signed_integer_spike,
    )
    from .losses import FoundationLossOutput, FoundationLossWeights, foundation_loss
    from .distillation import (
        FeatureCacheDataset,
        TeacherBatch,
        TorchTeacherAdapter,
        cache_teacher_features,
    )
    from .replacement import (
        BlockwiseReplacementTrainer,
        ProgressiveBlockReplacement,
        ReplacementStepResult,
    )
except ImportError:  # pragma: no cover - optional lightweight installation
    pass

__all__ = [
    "DEFAULT_MODALITY_SPECS",
    "EarlyExitController",
    "ModalitySpec",
    "SignedEventEncoder",
    "StreamingEfficiencyReport",
    "StreamingSpikingSSM",
    "SignedSpikingSSMBlock",
    "SpikingSSMBackbone",
    "SpikingSSMOutput",
    "signed_integer_spike",
    "FoundationLossOutput",
    "FoundationLossWeights",
    "foundation_loss",
    "FeatureCacheDataset",
    "TeacherBatch",
    "TorchTeacherAdapter",
    "cache_teacher_features",
    "BlockwiseReplacementTrainer",
    "ProgressiveBlockReplacement",
    "ReplacementStepResult",
    "align_and_fuse_modalities",
    "compressed_time_steps",
    "streaming_efficiency_report",
    "temporal_compress",
]
