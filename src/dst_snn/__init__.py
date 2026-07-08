"""Dendritic Spatio-Temporal SNN prototype modules."""

from .dendritic_layer import DendriticLayer, DendriticSNN, SurrogateSpike
from .research_modules import (
    ChronoPlasticLIFCell,
    ChronoPlasticLIFLayer,
    ConfidenceAwareBatchNorm1d,
    HighFrequencySpikingTransformerBlock,
    HighFrequencyTokenMixer,
    MaxPoolPatchEmbed,
    NoisySpikingActivation,
    ResearchSpikingTransformerSNN,
    ThresholdGuardingLoss,
    research_snn_training_step,
)

__all__ = [
    "ChronoPlasticLIFCell",
    "ChronoPlasticLIFLayer",
    "ConfidenceAwareBatchNorm1d",
    "DendriticLayer",
    "DendriticSNN",
    "HighFrequencySpikingTransformerBlock",
    "HighFrequencyTokenMixer",
    "MaxPoolPatchEmbed",
    "NoisySpikingActivation",
    "ResearchSpikingTransformerSNN",
    "SurrogateSpike",
    "ThresholdGuardingLoss",
    "DstWebLearner",
    "PlaywrightWebTrainer",
    "research_snn_training_step",
]


def __getattr__(name: str):
    if name in {"DstWebLearner", "PlaywrightWebTrainer"}:
        from .web_autonomous_learner import DstWebLearner, PlaywrightWebTrainer

        return {
            "DstWebLearner": DstWebLearner,
            "PlaywrightWebTrainer": PlaywrightWebTrainer,
        }[name]
    raise AttributeError(name)
