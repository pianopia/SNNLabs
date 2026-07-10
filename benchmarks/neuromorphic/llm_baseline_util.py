"""Shared wiring for optional LLM baselines on neuromorphic runners."""

from __future__ import annotations

from typing import Any

from src.dst_snn.eval.baselines import (
    LlmClassifierConfig,
    default_dvs_gesture_class_names,
    default_nmnist_class_names,
    evaluate_llm_on_loader,
    majority_scripted_backend,
    make_llm_backend,
    metric_set_to_dict,
)
from src.dst_snn.eval.result import MetricSet


def build_llm_backend(
    *,
    kind: str = "scripted",
    majority_class: int = 0,
    fixed_reply: str | None = None,
):
    """Construct a backend. Scripted default is majority-class (weak offline baseline)."""
    kind = (kind or "scripted").lower()
    if kind in {"scripted", "mock", "offline", "majority"}:
        if fixed_reply is not None:
            return make_llm_backend("scripted", fixed_reply=str(fixed_reply))
        return majority_scripted_backend(majority_class)
    return make_llm_backend(kind)


def run_llm_baseline(
    loader,
    *,
    num_classes: int,
    class_names: tuple[str, ...],
    backend_kind: str = "scripted",
    majority_class: int = 0,
    max_samples: int = 0,
    pj_per_token: float | None = None,
) -> MetricSet:
    backend = build_llm_backend(kind=backend_kind, majority_class=majority_class)
    kwargs: dict[str, Any] = {
        "num_classes": num_classes,
        "class_names": class_names,
        "max_samples": max_samples,
        "fallback_class": majority_class,
    }
    if pj_per_token is not None:
        kwargs["pj_per_token"] = pj_per_token
    config = LlmClassifierConfig(**kwargs)
    stats = evaluate_llm_on_loader(backend, loader, config=config)
    return stats.to_metric_set(config=config, backend_name=getattr(backend, "name", backend_kind))


def attach_llm_to_result(
    *,
    llm_metrics: MetricSet,
    primary_baseline: MetricSet | None,
    had_ann_or_cnn: bool,
) -> tuple[MetricSet, dict[str, Any]]:
    """Place LLM metrics: fill ``baseline`` if empty, else nest under extra.

    Returns ``(baseline_metric_set, metrics_extra_patch)``.
    """
    llm_dict = metric_set_to_dict(llm_metrics)
    if not had_ann_or_cnn or primary_baseline is None:
        return llm_metrics, {"llm_baseline": llm_dict}
    # Keep ANN/CNN as the primary baseline; attach LLM alongside.
    extra_patch = {"llm_baseline": llm_dict}
    return primary_baseline, extra_patch


def nmnist_class_names() -> tuple[str, ...]:
    return default_nmnist_class_names()


def dvs_class_names() -> tuple[str, ...]:
    return default_dvs_gesture_class_names()
