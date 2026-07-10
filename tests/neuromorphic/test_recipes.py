from __future__ import annotations

import pytest

from benchmarks.neuromorphic.recipes import (
    get_recipe,
    list_recipes,
    merge_recipe_with_cli,
    recipe_names,
)


def test_recipe_catalog_contains_closeout_presets():
    names = recipe_names()
    assert "parity-ds8" in names
    assert "hires-ds4" in names
    assert "hires-smoke" in names
    hires = get_recipe("hires-ds4")
    assert hires.downsample == 4
    assert hires.lr_schedule == "cosine"
    parity = get_recipe("parity-ds8")
    assert parity.downsample == 8
    assert parity.lr_schedule == "constant"


def test_unknown_recipe_raises():
    with pytest.raises(KeyError):
        get_recipe("not-a-real-recipe")


def test_merge_recipe_fills_fields_when_not_on_argv():
    cli = {
        "downsample": 99,
        "time_bins": 99,
        "epochs": 99,
        "batch_size": 99,
        "threshold": 0.1,
        "readout": "max_membrane",
        "lr": 0.5,
        "lr_schedule": "constant",
        "limit_train": 1,
        "limit_test": 1,
        "smoke_from_test": False,
    }
    # argv empty → recipe wins over cli defaults
    merged = merge_recipe_with_cli("hires-ds4", cli, argv=[])
    assert merged["downsample"] == 4
    assert merged["lr_schedule"] == "cosine"
    assert merged["epochs"] == 12
    assert merged["recipe"] == "hires-ds4"


def test_merge_cli_override_wins():
    cli = {
        "downsample": 2,
        "time_bins": 16,
        "epochs": 3,
        "batch_size": 8,
        "threshold": 1.0,
        "readout": "spike_count",
        "lr": 1e-3,
        "lr_schedule": "constant",
        "limit_train": 0,
        "limit_test": 0,
        "smoke_from_test": False,
    }
    merged = merge_recipe_with_cli(
        "hires-ds4",
        cli,
        argv=["--downsample", "2", "--epochs", "3"],
    )
    assert merged["downsample"] == 2  # explicit CLI
    assert merged["epochs"] == 3  # explicit CLI
    assert merged["lr_schedule"] == "cosine"  # from recipe
    assert merged["time_bins"] == 16


def test_no_recipe_passthrough():
    cli = {"downsample": 8, "epochs": 5}
    merged = merge_recipe_with_cli(None, cli, argv=[])
    assert merged["downsample"] == 8
    assert merged["recipe"] is None


def test_list_recipes_nonempty():
    assert len(list_recipes()) >= 3
