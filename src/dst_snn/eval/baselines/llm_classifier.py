"""Rasterized-frame → text → LLM classification baseline for neuromorphic tasks.

Design (Phase 0 harness): A/B benchmarks feed compact event-frame summaries to
an LLM and score quality / latency / a *non-AC/MAC* token energy proxy.

This is an **optional eval interface**, not a product path. Real API calls
require ``HttpChatLlmBackend``; tests and CI use ``ScriptedLlmBackend``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.dst_snn.eval.baselines.llm_backend import (
    LlmBackend,
    LlmCompletion,
    ScriptedLlmBackend,
    estimate_tokens,
    parse_class_id,
)
from src.dst_snn.eval.result import MetricSet

# Token energy proxy: intentionally *not* comparable to SNN AC / dense MAC pJ.
# Documented in energy_source / energy_accounting. Override via constructor.
DEFAULT_PJ_PER_TOKEN = 1.0e9  # 1 mJ/token ballpark cloud-API proxy


SYSTEM_PROMPT = (
    "You classify neuromorphic event-camera samples from compact numeric "
    "summaries. Reply with a single integer class id and nothing else."
)


def _as_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            if isinstance(item, list):
                out.extend(_as_list(item))
            else:
                out.append(float(item))
        return out
    raise TypeError(f"unsupported tensor-like type: {type(value)!r}")


def summarize_sample(x: Any, *, max_stats: int = 16) -> dict[str, Any]:
    """Build a JSON-serializable summary of one sample.

    Accepts:
      - flat spikes ``[time, features]``
      - event frames ``[time, channels, height, width]``
    """
    # Prefer torch-less path when possible.
    shape = tuple(getattr(x, "shape", ()))
    if len(shape) == 2:
        # [T, F]
        t_len, feat = int(shape[0]), int(shape[1])
        vals = _as_list(x)
        mean_act = sum(vals) / max(1, len(vals))
        # Per-time mean of features
        per_t = []
        for ti in range(t_len):
            row = vals[ti * feat : (ti + 1) * feat]
            per_t.append(sum(row) / max(1, feat))
        return {
            "layout": "flat",
            "time_bins": t_len,
            "features": feat,
            "mean_activity": round(mean_act, 6),
            "per_time_mean": [round(v, 6) for v in per_t[:max_stats]],
            "activity_std": round(_std(vals), 6),
        }
    if len(shape) == 4:
        # [T, C, H, W]
        t_len, ch, h, w = (int(v) for v in shape)
        vals = _as_list(x)
        plane = ch * h * w
        mean_act = sum(vals) / max(1, len(vals))
        # Channel means over all time/space
        ch_means = []
        for c in range(ch):
            s = 0.0
            n = 0
            for ti in range(t_len):
                base = ti * plane + c * h * w
                chunk = vals[base : base + h * w]
                s += sum(chunk)
                n += h * w
            ch_means.append(round(s / max(1, n), 6))
        # Spatial centroid of absolute activity (last time bin, all channels summed)
        last = vals[(t_len - 1) * plane :]
        mass = 0.0
        cx = cy = 0.0
        for c in range(ch):
            for yi in range(h):
                for xi in range(w):
                    a = abs(last[c * h * w + yi * w + xi])
                    mass += a
                    cx += a * xi
                    cy += a * yi
        if mass > 0:
            cx /= mass
            cy /= mass
        else:
            cx = (w - 1) / 2.0
            cy = (h - 1) / 2.0
        return {
            "layout": "frames",
            "time_bins": t_len,
            "channels": ch,
            "height": h,
            "width": w,
            "mean_activity": round(mean_act, 6),
            "channel_means": ch_means,
            "centroid_xy": [round(cx, 4), round(cy, 4)],
            "activity_std": round(_std(vals), 6),
        }
    # Fallback: flatten whatever we got
    vals = _as_list(x)
    return {
        "layout": "unknown",
        "shape": list(shape),
        "mean_activity": round(sum(vals) / max(1, len(vals)), 6),
        "n": len(vals),
    }


def _std(vals: Sequence[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return var ** 0.5


def build_classification_prompt(
    summary: dict[str, Any],
    *,
    class_names: Sequence[str],
    num_classes: int,
) -> str:
    names = list(class_names) if class_names else [str(i) for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError("class_names length must equal num_classes")
    class_lines = "\n".join(f"  {i}: {names[i]}" for i in range(num_classes))
    return (
        f"Task: classify this event-camera sample into exactly one class.\n"
        f"Classes (id: name):\n{class_lines}\n"
        f"Sample summary (JSON):\n{json_dumps(summary)}\n"
        f"Respond with only the integer class id (0-{num_classes - 1})."
    )


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass
class LlmClassifierConfig:
    num_classes: int
    class_names: tuple[str, ...] = ()
    pj_per_token: float = DEFAULT_PJ_PER_TOKEN
    system_prompt: str = SYSTEM_PROMPT
    max_samples: int = 0  # 0 = all
    fallback_class: int = 0


@dataclass
class LlmClassifyStats:
    predictions: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parse_failures: int = 0
    completions: list[LlmCompletion] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.targets:
            return 0.0
        correct = sum(int(p == t) for p, t in zip(self.predictions, self.targets))
        return correct / len(self.targets)

    def to_metric_set(self, *, config: LlmClassifierConfig, backend_name: str) -> MetricSet:
        from src.dst_snn.eval.metrics import latency_percentiles

        lat = latency_percentiles(self.latencies_ms) if self.latencies_ms else {"p50": 0.0, "p95": 0.0}
        total_tokens = self.prompt_tokens + self.completion_tokens
        energy_pj = float(total_tokens) * float(config.pj_per_token)
        return MetricSet(
            quality=self.accuracy,
            quality_metric="llm_classification_accuracy",
            latency_ms_p50=float(lat["p50"]),
            latency_ms_p95=float(lat["p95"]),
            spikes_per_inference=0.0,
            active_neuron_fraction=0.0,
            energy_pj=energy_pj,
            energy_source=(
                f"llm_token_proxy_v1 ({config.pj_per_token:g} pJ/token; "
                "NOT comparable to SNN AC / dense MAC)"
            ),
            param_count=0,
            model_bytes=0,
            extra={
                "backend": backend_name,
                "n_samples": len(self.targets),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "parse_failures": self.parse_failures,
                "pj_per_token": config.pj_per_token,
                "energy_accounting": "llm_api_external_v1",
                "class_names": list(config.class_names) if config.class_names else None,
            },
        )


def classify_sample(
    backend: LlmBackend,
    x: Any,
    *,
    config: LlmClassifierConfig,
) -> tuple[int, LlmCompletion]:
    names = config.class_names or tuple(str(i) for i in range(config.num_classes))
    summary = summarize_sample(x)
    prompt = build_classification_prompt(
        summary, class_names=names, num_classes=config.num_classes
    )
    completion = backend.complete(prompt, system=config.system_prompt)
    parsed = parse_class_id(completion.text, num_classes=config.num_classes)
    if parsed is None:
        return config.fallback_class, completion
    return parsed, completion


def evaluate_llm_classifier(
    backend: LlmBackend,
    samples: Sequence[tuple[Any, int]],
    *,
    config: LlmClassifierConfig,
) -> LlmClassifyStats:
    """Classify a list of ``(x, y)`` pairs. No network unless backend uses HTTP."""
    stats = LlmClassifyStats()
    limit = config.max_samples if config.max_samples > 0 else len(samples)
    for x, y in list(samples)[:limit]:
        pred, completion = classify_sample(backend, x, config=config)
        if parse_class_id(completion.text, num_classes=config.num_classes) is None:
            stats.parse_failures += 1
        stats.predictions.append(pred)
        stats.targets.append(int(y))
        stats.latencies_ms.append(completion.latency_ms)
        stats.prompt_tokens += completion.prompt_tokens or estimate_tokens("")
        stats.completion_tokens += completion.completion_tokens
        stats.completions.append(completion)
    return stats


def evaluate_llm_on_loader(
    backend: LlmBackend,
    loader,
    *,
    config: LlmClassifierConfig,
    device: str = "cpu",
) -> LlmClassifyStats:
    """Iterate a DataLoader of ``(batch_x, batch_y)`` tensors (offline-safe)."""
    samples: list[tuple[Any, int]] = []
    for batch_x, batch_y in loader:
        # Support torch tensors without requiring torch at import time of callers.
        if hasattr(batch_x, "detach"):
            batch_x = batch_x.detach().cpu()
            batch_y = batch_y.detach().cpu()
        n = int(batch_x.shape[0])
        for i in range(n):
            samples.append((batch_x[i], int(batch_y[i].item() if hasattr(batch_y[i], "item") else batch_y[i])))
            if config.max_samples > 0 and len(samples) >= config.max_samples:
                return evaluate_llm_classifier(backend, samples, config=config)
    return evaluate_llm_classifier(backend, samples, config=config)


def metric_set_to_dict(metrics: MetricSet) -> dict[str, Any]:
    return asdict(metrics)


def default_nmnist_class_names() -> tuple[str, ...]:
    return tuple(str(i) for i in range(10))


def default_dvs_gesture_class_names() -> tuple[str, ...]:
    # IBM DVS128 Gesture class order used by tonic (11 classes).
    return (
        "hand_clapping",
        "right_hand_wave",
        "left_hand_wave",
        "right_arm_clockwise",
        "right_arm_counter_clockwise",
        "left_arm_clockwise",
        "left_arm_counter_clockwise",
        "arm_roll",
        "air_drums",
        "air_guitar",
        "other_gestures",
    )


def majority_scripted_backend(majority_class: int = 0) -> ScriptedLlmBackend:
    """Weak baseline that always emits the majority class id."""
    return ScriptedLlmBackend(name="scripted-majority", fixed_reply=str(int(majority_class)))
