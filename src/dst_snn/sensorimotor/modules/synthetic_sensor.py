"""Deterministic synthetic sensor for headless closed-loop tests."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..protocol import SensorimotorMessage, register_message


@dataclass
class SyntheticSensor:
    """Sine-wave sensor whose phase can be shifted by motor commands.

    Ground-truth discrete state is ``phase_bin`` in the observation payload,
    enabling linear-probe / cluster-purity metrics on learned latents.
    """

    id: str = "synthetic-sensor"
    phase: float = 0.0
    phase_shift: float = 0.0
    n_phase_bins: int = 16
    step_rate: float = 0.25
    left_delta: float = -0.15
    right_delta: float = 0.15

    def register(self) -> SensorimotorMessage:
        return register_message(module_id=self.id, role="sensor", modality="synthetic", shape=[2])

    def apply_motor(self, command: str) -> None:
        """Apply an actuator command effect (closes the synthetic loop)."""
        cmd = str(command).lower()
        if cmd == "left":
            self.phase_shift += self.left_delta
        elif cmd == "right":
            self.phase_shift += self.right_delta
        # "stop" and unknown commands leave phase_shift unchanged.

    def effective_phase(self, step: int) -> float:
        return self.phase + self.phase_shift + float(step) * self.step_rate

    def phase_bin_at(self, step: int) -> int:
        phi = self.effective_phase(step) % (2.0 * math.pi)
        bin_id = int(phi / (2.0 * math.pi) * self.n_phase_bins) % self.n_phase_bins
        return bin_id

    def observe(self, step: int) -> SensorimotorMessage:
        value = 0.5 + 0.5 * math.sin(self.effective_phase(step))
        phase_bin = self.phase_bin_at(step)
        return SensorimotorMessage(
            type="observation",
            id=self.id,
            payload={
                "signal": value,
                "step": step % 16,
                "phase_bin": phase_bin,
                "phase_shift": self.phase_shift,
            },
        )
