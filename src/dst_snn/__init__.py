"""Dendritic Spatio-Temporal SNN prototype modules.

Torch-heavy symbols are imported lazily so lightweight subpackages such as
``src.dst_snn.eval.result`` remain usable in scorer-only environments.
"""

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
    if name in {"DendriticLayer", "DendriticSNN", "SurrogateSpike"}:
        from .dendritic_layer import DendriticLayer, DendriticSNN, SurrogateSpike

        return {
            "DendriticLayer": DendriticLayer,
            "DendriticSNN": DendriticSNN,
            "SurrogateSpike": SurrogateSpike,
        }[name]
    if name in {
        "ChronoPlasticLIFCell",
        "ChronoPlasticLIFLayer",
        "ConfidenceAwareBatchNorm1d",
        "HighFrequencySpikingTransformerBlock",
        "HighFrequencyTokenMixer",
        "MaxPoolPatchEmbed",
        "NoisySpikingActivation",
        "ResearchSpikingTransformerSNN",
        "ThresholdGuardingLoss",
        "research_snn_training_step",
    }:
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

        return {
            "ChronoPlasticLIFCell": ChronoPlasticLIFCell,
            "ChronoPlasticLIFLayer": ChronoPlasticLIFLayer,
            "ConfidenceAwareBatchNorm1d": ConfidenceAwareBatchNorm1d,
            "HighFrequencySpikingTransformerBlock": HighFrequencySpikingTransformerBlock,
            "HighFrequencyTokenMixer": HighFrequencyTokenMixer,
            "MaxPoolPatchEmbed": MaxPoolPatchEmbed,
            "NoisySpikingActivation": NoisySpikingActivation,
            "ResearchSpikingTransformerSNN": ResearchSpikingTransformerSNN,
            "ThresholdGuardingLoss": ThresholdGuardingLoss,
            "research_snn_training_step": research_snn_training_step,
        }[name]
    if name in {"DstWebLearner", "PlaywrightWebTrainer"}:
        from .web_autonomous_learner import DstWebLearner, PlaywrightWebTrainer

        return {
            "DstWebLearner": DstWebLearner,
            "PlaywrightWebTrainer": PlaywrightWebTrainer,
        }[name]
    raise AttributeError(name)
