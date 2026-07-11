"""Constant-state streaming spiking mixer and stable early-exit policy."""

from __future__ import annotations

import numpy as np


class StreamingSpikingSSM:
    """Small reference SSM with signed threshold events and bounded state.

    This is the executable runtime contract for later trainable Torch and
    hardware backends. Runtime state is O(state_size), independent of history.
    """

    def __init__(
        self,
        input_size: int,
        state_size: int,
        *,
        decay: float = 0.9,
        threshold: float = 1.0,
        seed: int = 0,
    ):
        if input_size < 1 or state_size < 1:
            raise ValueError("input_size and state_size must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.input_size = int(input_size)
        self.state_size = int(state_size)
        self.decay = float(decay)
        self.threshold = float(threshold)
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(float(input_size))
        self.input_projection = rng.normal(0.0, scale, (input_size, state_size)).astype(np.float32)
        self.state = np.zeros(state_size, dtype=np.float32)

    @property
    def state_bytes(self) -> int:
        return int(self.state.nbytes)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, events: np.ndarray) -> np.ndarray:
        row = np.asarray(events, dtype=np.float32).reshape(-1)
        if row.shape[0] != self.input_size:
            raise ValueError(f"expected {self.input_size} input features")
        self.state = self.decay * self.state + row @ self.input_projection
        spikes = np.zeros(self.state_size, dtype=np.int8)
        spikes[self.state >= self.threshold] = 1
        spikes[self.state <= -self.threshold] = -1
        self.state -= spikes.astype(np.float32) * self.threshold
        return spikes

    def run(self, events: np.ndarray) -> np.ndarray:
        source = np.asarray(events)
        if source.ndim != 2 or source.shape[1] != self.input_size:
            raise ValueError(f"expected [steps, {self.input_size}] events")
        return np.stack([self.step(row) for row in source], axis=0)


class EarlyExitController:
    """Exit only when confidence is high and the prediction is stable."""

    def __init__(self, *, confidence: float = 0.9, patience: int = 2, min_steps: int = 1):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if patience < 1 or min_steps < 1:
            raise ValueError("patience and min_steps must be positive")
        self.confidence = float(confidence)
        self.patience = int(patience)
        self.min_steps = int(min_steps)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.stable_steps = 0
        self.last_prediction: int | None = None

    def update(self, probabilities: np.ndarray) -> bool:
        probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if probs.size == 0 or not np.all(np.isfinite(probs)):
            raise ValueError("probabilities must be finite and non-empty")
        prediction = int(np.argmax(probs))
        score = float(probs[prediction])
        self.steps += 1
        if prediction == self.last_prediction and score >= self.confidence:
            self.stable_steps += 1
        elif score >= self.confidence:
            self.stable_steps = 1
        else:
            self.stable_steps = 0
        self.last_prediction = prediction
        return self.steps >= self.min_steps and self.stable_steps >= self.patience
