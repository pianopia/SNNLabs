from __future__ import annotations

import copy

import torch

from src.dst_snn.web_autonomous_learner import (
    DstWebLearner,
    ModuleObservation,
    WebObservation,
)


def _observation(reward_salience: float) -> WebObservation:
    module = ModuleObservation(
        module="dom-text",
        modality="text",
        tokens=["alpha", "beta", "gamma"],
        salience=reward_salience,
        source="test",
    )
    return WebObservation(url="http://x", title="t", modules=[module], action=None)


def _weight_delta(salience: float) -> float:
    torch.manual_seed(0)
    learner = DstWebLearner(in_features=64, time_steps=8, branches=4, max_delay=4)
    before = copy.deepcopy(learner.model.dendrite.weight.detach())
    result = learner.train_observation(_observation(salience))
    after = learner.model.dendrite.weight.detach()
    assert "reward" in result
    return float((after - before).abs().sum().item())


def test_reward_scales_weight_update():
    low = _weight_delta(0.1)
    high = _weight_delta(1.0)
    assert high > low
