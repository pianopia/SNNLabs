"""Controlled DVS training recipes (Phase 0 closeout).

Named presets keep accuracy pushes comparable instead of ad-hoc flag soup.
``parity-ds8`` freezes the 2026-07-10 full-train spatial recipe (downsample=8).
``hires-ds4`` is the higher spatial-resolution preset from the closeout plan goal
(downsample=4 ≈ 32×32 frames, cosine LR).

CLI explicit flags always win over recipe defaults when present on argv.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional
import sys


@dataclass(frozen=True)
class DvsRecipe:
    """Hyper-parameters that define a controlled DVS training recipe."""

    name: str
    description: str
    downsample: int
    time_bins: int
    epochs: int
    batch_size: int
    threshold: float
    readout: str
    lr: float
    lr_schedule: str
    # Optional limits (0 = full split). Smoke recipes set these.
    limit_train: int = 0
    limit_test: int = 0
    smoke_from_test: bool = False
    # Suggested backbone is documentation only; runners keep --backbone free.
    suggested_backbone: str = "conv-plif"

    def as_runner_kwargs(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("name")
        data.pop("description")
        data.pop("suggested_backbone")
        return data


# Spatial note: DVS128 sensor is 128×128. downsample=k → floor(128/k)×floor(128/k).
# ds8 → 16×16 (frozen full-train). ds4 → 32×32 (hires preset).
_RECIPES: dict[str, DvsRecipe] = {
    "parity-ds8": DvsRecipe(
        name="parity-ds8",
        description=(
            "Matches 2026-07-10 full-train freeze spatial recipe: "
            "downsample=8 (~16×16), time_bins=16, epochs=12, spike_count, constant LR."
        ),
        downsample=8,
        time_bins=16,
        epochs=12,
        batch_size=16,
        threshold=1.0,
        readout="spike_count",
        lr=1e-3,
        lr_schedule="constant",
        suggested_backbone="sew-plif",
    ),
    "hires-ds4": DvsRecipe(
        name="hires-ds4",
        description=(
            "Higher spatial resolution preset (closeout plan): "
            "downsample=4 (~32×32), time_bins=16, epochs=12, spike_count, cosine LR."
        ),
        downsample=4,
        time_bins=16,
        epochs=12,
        batch_size=8,
        threshold=1.0,
        readout="spike_count",
        lr=1e-3,
        lr_schedule="cosine",
        suggested_backbone="conv-plif",
    ),
    "hires-smoke": DvsRecipe(
        name="hires-smoke",
        description=(
            "CI-scale higher-res smoke: downsample=4, short epochs, stratified smoke split."
        ),
        downsample=4,
        time_bins=8,
        epochs=2,
        batch_size=4,
        threshold=1.0,
        readout="spike_count",
        lr=1e-3,
        lr_schedule="cosine",
        limit_train=48,
        limit_test=24,
        smoke_from_test=True,
        suggested_backbone="conv-plif",
    ),
    "smoke-spatial": DvsRecipe(
        name="smoke-spatial",
        description="Fast spatial smoke at freeze resolution (ds8).",
        downsample=8,
        time_bins=8,
        epochs=2,
        batch_size=8,
        threshold=1.0,
        readout="spike_count",
        lr=1e-3,
        lr_schedule="constant",
        limit_train=64,
        limit_test=32,
        smoke_from_test=True,
        suggested_backbone="conv-plif",
    ),
}


def list_recipes() -> dict[str, DvsRecipe]:
    return dict(_RECIPES)


def get_recipe(name: str) -> DvsRecipe:
    key = name.strip().lower()
    if key not in _RECIPES:
        known = ", ".join(sorted(_RECIPES))
        raise KeyError(f"Unknown DVS recipe '{name}'. Known: {known}")
    return _RECIPES[key]


def recipe_names() -> list[str]:
    return sorted(_RECIPES)


# CLI flag name (with --) → DvsRecipe field name
_FLAG_TO_FIELD: dict[str, str] = {
    "--downsample": "downsample",
    "--time-bins": "time_bins",
    "--epochs": "epochs",
    "--batch-size": "batch_size",
    "--threshold": "threshold",
    "--readout": "readout",
    "--lr": "lr",
    "--lr-schedule": "lr_schedule",
    "--limit-train": "limit_train",
    "--limit-test": "limit_test",
    "--smoke-from-test": "smoke_from_test",
}


def _argv_overrides(argv: Optional[list[str]] = None) -> set[str]:
    """Return recipe field names explicitly set on the command line."""
    args = list(sys.argv[1:] if argv is None else argv)
    overridden: set[str] = set()
    i = 0
    while i < len(args):
        token = args[i]
        if token in _FLAG_TO_FIELD:
            overridden.add(_FLAG_TO_FIELD[token])
            # boolean store_true has no value; others may be next token or --flag=value
            if token == "--smoke-from-test":
                i += 1
                continue
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if "=" in token:
            flag, _ = token.split("=", 1)
            if flag in _FLAG_TO_FIELD:
                overridden.add(_FLAG_TO_FIELD[flag])
        i += 1
    return overridden


def merge_recipe_with_cli(
    recipe_name: str | None,
    cli: Mapping[str, Any],
    *,
    argv: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Merge recipe defaults with CLI values.

    Explicit CLI flags always win. Unspecified recipe-controlled fields take
    recipe values when a recipe is selected; otherwise the CLI/default values
    from ``cli`` are kept.
    """
    out = dict(cli)
    if not recipe_name:
        out["recipe"] = None
        return out
    recipe = get_recipe(recipe_name)
    overridden = _argv_overrides(argv)
    for field in fields(DvsRecipe):
        if field.name in {"name", "description", "suggested_backbone"}:
            continue
        if field.name in overridden:
            continue
        out[field.name] = getattr(recipe, field.name)
    out["recipe"] = recipe.name
    out["recipe_description"] = recipe.description
    out["recipe_suggested_backbone"] = recipe.suggested_backbone
    return out
