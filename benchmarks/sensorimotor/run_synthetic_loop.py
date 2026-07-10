#!/usr/bin/env python3
"""Synthetic closed-loop benchmark for the sensorimotor runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyTorch with `pip install -r requirements-dst-snn.txt`.") from exc

from src.dst_snn.eval import (
    EnergyModel,
    MetricSet,
    RunResult,
    latency_percentiles,
    model_size,
    run_benchmarks,
    snn_energy_pj,
)
from src.dst_snn.sensorimotor.modules import MockActuator, SyntheticSensor
from src.dst_snn.sensorimotor.policy import IntrinsicMotorPolicy
from src.dst_snn.sensorimotor.registry import ModuleRegistry
from src.dst_snn.sensorimotor.runtime import SensorimotorRuntime
from src.dst_snn.sensorimotor.world_model import (
    LearningProgress,
    PredictiveWorldModel,
    train_world_model_step,
)


def _loss_reduction(losses: list[float]) -> float:
    if len(losses) < 2:
        return 0.0
    first_window = losses[: max(1, len(losses) // 4)]
    last_window = losses[-max(1, len(losses) // 4):]
    first = sum(first_window) / len(first_window)
    last = sum(last_window) / len(last_window)
    if first <= 0:
        return 0.0
    return max(0.0, min(1.0, (first - last) / first))


class SyntheticSensorimotorRunner:
    name = "synthetic-sensorimotor"

    def __init__(
        self,
        *,
        steps: int = 32,
        feature_size: int = 64,
        motor_size: int = 16,
        time_steps: int = 8,
        latent_size: int = 32,
        lr: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.steps = steps
        self.feature_size = feature_size
        self.motor_size = motor_size
        self.time_steps = time_steps
        self.latent_size = latent_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.runtime: SensorimotorRuntime | None = None
        self.model: PredictiveWorldModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.policy: IntrinsicMotorPolicy | None = None
        self.sensor = SyntheticSensor()
        self.actuator = MockActuator()

    def prepare(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        registry = ModuleRegistry(feature_size=self.feature_size, motor_size=self.motor_size)
        runtime = SensorimotorRuntime(registry, time_steps=self.time_steps)
        runtime.ingest(self.sensor.register())
        runtime.ingest(self.actuator.register())
        self.runtime = runtime
        self.model = PredictiveWorldModel(
            sensory_size=self.feature_size,
            motor_size=self.motor_size,
            latent_size=self.latent_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.policy = IntrinsicMotorPolicy(epsilon=0.25, temperature=0.5, seed=self.seed)

    def run(self) -> RunResult:
        assert (
            self.runtime is not None
            and self.model is not None
            and self.optimizer is not None
            and self.policy is not None
        )
        progress = LearningProgress(alpha=0.2)
        losses: list[float] = []
        rewards: list[float] = []
        latencies_ms: list[float] = []
        spike_counts: list[float] = []
        active_fracs: list[float] = []
        selected_commands: list[list[dict[str, str]]] = []

        for step in range(self.steps):
            self.runtime.ingest(self.sensor.observe(step))
            motor_activity, selected = self.policy.activity(self.runtime.registry)
            start = time.perf_counter()
            tick = self.runtime.tick(motor_activity)
            sensory = torch.from_numpy(tick["spikes"]).float().unsqueeze(0).to(self.device)
            motor = torch.from_numpy(np.tile(motor_activity, (self.time_steps, 1))).float().unsqueeze(0).to(self.device)
            metrics = train_world_model_step(self.model, self.optimizer, sensory, motor, progress)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            losses.append(metrics["prediction_loss"])
            reward = metrics.get("intrinsic_reward", 0.0)
            rewards.append(reward)
            self.policy.update(selected, reward)
            selected_commands.append(selected)
            spike_counts.append(float(tick["spikes"].sum()))
            active_fracs.append(float((tick["spikes"] > 0).mean()))

        quality = _loss_reduction(losses)
        lat = latency_percentiles(latencies_ms)
        size = model_size(self.model)
        energy_model = EnergyModel()
        spikes_per_inf = sum(spike_counts) / max(1, len(spike_counts))
        energy_pj = snn_energy_pj(spikes_per_inf, self.latent_size, energy_model)
        return RunResult(
            benchmark=self.name,
            model="predictive-world-model",
            metrics=MetricSet(
                quality=quality,
                quality_metric="prediction_loss_reduction",
                latency_ms_p50=lat["p50"],
                latency_ms_p95=lat["p95"],
                spikes_per_inference=spikes_per_inf,
                active_neuron_fraction=sum(active_fracs) / max(1, len(active_fracs)),
                energy_pj=energy_pj,
                energy_source=energy_model.source,
                param_count=size["param_count"],
                model_bytes=size["model_bytes"],
                extra={
                    "steps": self.steps,
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "mean_intrinsic_reward": sum(rewards) / max(1, len(rewards)),
                    "losses": losses,
                    "policy_scores": self.policy.command_scores,
                    "selected_commands": selected_commands,
                },
            ),
            baseline=None,
            meta={
                "feature_size": self.feature_size,
                "motor_size": self.motor_size,
                "time_steps": self.time_steps,
                "latent_size": self.latent_size,
                "seed": self.seed,
            },
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--motor-size", type=int, default=16)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = SyntheticSensorimotorRunner(
        steps=args.steps,
        feature_size=args.feature_size,
        motor_size=args.motor_size,
        time_steps=args.time_steps,
        latent_size=args.latent_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    print(run_benchmarks([runner], args.out_dir)[0].to_json())


if __name__ == "__main__":
    main()
