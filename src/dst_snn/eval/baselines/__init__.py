"""Optional non-SNN baselines for harness comparisons."""

from .ann_classifier import DenseAnnClassifier, train_ann_classifier
from .ann_predictor import DenseAnnPredictor, train_ann_predictor_step
from .frame_cnn import FrameCnnClassifier, train_frame_cnn
from .llm_backend import (
    HttpChatLlmBackend,
    LlmBackend,
    LlmCompletion,
    ScriptedLlmBackend,
    make_llm_backend,
    parse_class_id,
)
from .llm_classifier import (
    LlmClassifierConfig,
    LlmClassifyStats,
    default_dvs_gesture_class_names,
    default_nmnist_class_names,
    evaluate_llm_classifier,
    evaluate_llm_on_loader,
    majority_scripted_backend,
    metric_set_to_dict,
)
from .spatial_ops import conv2d_mac_ops, estimate_sew_macs, estimate_three_stage_conv_macs

__all__ = [
    "DenseAnnClassifier",
    "DenseAnnPredictor",
    "FrameCnnClassifier",
    "HttpChatLlmBackend",
    "LlmBackend",
    "LlmClassifierConfig",
    "LlmClassifyStats",
    "LlmCompletion",
    "ScriptedLlmBackend",
    "conv2d_mac_ops",
    "default_dvs_gesture_class_names",
    "default_nmnist_class_names",
    "estimate_sew_macs",
    "estimate_three_stage_conv_macs",
    "evaluate_llm_classifier",
    "evaluate_llm_on_loader",
    "majority_scripted_backend",
    "make_llm_backend",
    "metric_set_to_dict",
    "parse_class_id",
    "train_ann_classifier",
    "train_ann_predictor_step",
    "train_frame_cnn",
]
