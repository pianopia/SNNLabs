"""Optional non-SNN baselines for harness comparisons."""

from .ann_classifier import DenseAnnClassifier, train_ann_classifier
from .frame_cnn import FrameCnnClassifier, train_frame_cnn

__all__ = [
    "DenseAnnClassifier",
    "FrameCnnClassifier",
    "train_ann_classifier",
    "train_frame_cnn",
]
