"""Offline tests for the LLM baseline interface (no network)."""

from __future__ import annotations

from src.dst_snn.eval.baselines.llm_backend import (
    ScriptedLlmBackend,
    estimate_tokens,
    make_llm_backend,
    parse_class_id,
)
from src.dst_snn.eval.baselines.llm_classifier import (
    LlmClassifierConfig,
    build_classification_prompt,
    classify_sample,
    evaluate_llm_classifier,
    majority_scripted_backend,
    summarize_sample,
)


def test_estimate_tokens_and_parse_class():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert parse_class_id("class 3 please", num_classes=11) == 3
    assert parse_class_id("The answer is 99", num_classes=11) is None
    assert parse_class_id("no number", num_classes=5) is None


def test_scripted_backend_fixed_reply():
    backend = ScriptedLlmBackend(fixed_reply="7", artificial_latency_ms=1.5)
    out = backend.complete("anything")
    assert out.text == "7"
    assert out.latency_ms >= 1.5
    assert out.backend == "scripted"
    assert out.prompt_tokens > 0


def test_scripted_responder():
    backend = ScriptedLlmBackend(responder=lambda prompt: "2" if "mean_activity" in prompt else "0")
    assert backend.complete("mean_activity=0.1").text == "2"
    assert backend.complete("other").text == "0"


def test_make_llm_backend_scripted():
    backend = make_llm_backend("scripted", fixed_reply="1")
    assert backend.complete("x").text == "1"


def test_summarize_flat_sample():
    # [time=2, features=3]
    x = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
    # list-of-lists not supported by shape attr — use a tiny fake
    class _T:
        shape = (2, 3)

        def __iter__(self):
            return iter(x)

        def tolist(self):
            return x

    summary = summarize_sample(_T())
    assert summary["layout"] == "flat"
    assert summary["time_bins"] == 2
    assert summary["features"] == 3
    assert "mean_activity" in summary


def test_summarize_frame_sample():
    # [T=1, C=2, H=2, W=2]
    class _F:
        shape = (1, 2, 2, 2)

        def tolist(self):
            # ch0 all 1s, ch1 all 0s
            return [[[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]]

    summary = summarize_sample(_F())
    assert summary["layout"] == "frames"
    assert summary["channels"] == 2
    assert summary["channel_means"][0] == 1.0
    assert summary["channel_means"][1] == 0.0


def test_build_prompt_and_classify():
    summary = {"layout": "flat", "mean_activity": 0.5, "time_bins": 4, "features": 8}
    prompt = build_classification_prompt(
        summary, class_names=("a", "b", "c"), num_classes=3
    )
    assert "0: a" in prompt
    assert "mean_activity" in prompt

    backend = ScriptedLlmBackend(fixed_reply="The label is 1.")
    config = LlmClassifierConfig(num_classes=3, class_names=("a", "b", "c"))
    pred, completion = classify_sample(backend, _flat_tensor(), config=config)
    assert pred == 1
    assert completion.latency_ms >= 0.0


class _flat_tensor:
    shape = (2, 4)

    def tolist(self):
        return [[0.1, 0.0, 0.0, 0.2], [0.0, 0.3, 0.0, 0.0]]


def test_evaluate_llm_classifier_accuracy():
    # Oracle scripted backend: always returns the ground-truth embedded in a marker
    # we don't have in prompt — instead use responder that cycles from targets via
    # a closure over call index.
    targets = [0, 1, 2, 1]
    replies = iter(str(t) for t in targets)
    backend = ScriptedLlmBackend(responder=lambda _p: next(replies))
    samples = [(_flat_tensor(), y) for y in targets]
    config = LlmClassifierConfig(num_classes=3, class_names=("a", "b", "c"))
    stats = evaluate_llm_classifier(backend, samples, config=config)
    assert stats.accuracy == 1.0
    assert stats.parse_failures == 0
    metrics = stats.to_metric_set(config=config, backend_name=backend.name)
    assert metrics.quality == 1.0
    assert metrics.quality_metric == "llm_classification_accuracy"
    assert metrics.extra["energy_accounting"] == "llm_api_external_v1"
    assert "NOT comparable" in metrics.energy_source
    assert metrics.energy_pj > 0.0


def test_majority_scripted_is_weak():
    backend = majority_scripted_backend(0)
    samples = [(_flat_tensor(), y) for y in (0, 1, 1, 1)]
    config = LlmClassifierConfig(num_classes=2)
    stats = evaluate_llm_classifier(backend, samples, config=config)
    assert stats.accuracy == 0.25
    assert all(p == 0 for p in stats.predictions)


def test_parse_failure_uses_fallback():
    backend = ScriptedLlmBackend(fixed_reply="not a class")
    config = LlmClassifierConfig(num_classes=5, fallback_class=3)
    pred, _ = classify_sample(backend, _flat_tensor(), config=config)
    assert pred == 3


def test_attach_llm_to_result_nested_when_ann_present():
    from src.dst_snn.eval.baselines.llm_classifier import metric_set_to_dict
    from src.dst_snn.eval.result import MetricSet
    from benchmarks.neuromorphic.llm_baseline_util import attach_llm_to_result

    ann = MetricSet(
        quality=0.5,
        quality_metric="ann_mlp_accuracy",
        latency_ms_p50=0.0,
        latency_ms_p95=0.0,
        spikes_per_inference=0.0,
        active_neuron_fraction=0.0,
        energy_pj=1.0,
        energy_source="ann",
        param_count=10,
        model_bytes=40,
    )
    llm = MetricSet(
        quality=0.1,
        quality_metric="llm_classification_accuracy",
        latency_ms_p50=5.0,
        latency_ms_p95=9.0,
        spikes_per_inference=0.0,
        active_neuron_fraction=0.0,
        energy_pj=1e12,
        energy_source="llm_token_proxy_v1",
        param_count=0,
        model_bytes=0,
        extra={"energy_accounting": "llm_api_external_v1"},
    )
    primary, patch = attach_llm_to_result(
        llm_metrics=llm, primary_baseline=ann, had_ann_or_cnn=True
    )
    assert primary.quality_metric == "ann_mlp_accuracy"
    assert "llm_baseline" in patch
    assert patch["llm_baseline"]["quality"] == 0.1

    only_llm, patch2 = attach_llm_to_result(
        llm_metrics=llm, primary_baseline=ann, had_ann_or_cnn=False
    )
    assert only_llm.quality_metric == "llm_classification_accuracy"
    assert metric_set_to_dict(only_llm)["quality"] == patch2["llm_baseline"]["quality"]
