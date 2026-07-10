#!/usr/bin/env python3
"""Synthetic closed-loop benchmark for the sensorimotor runtime.

Closes the loop by applying motor commands to the synthetic sensor phase,
records representation probes (linear probe / cluster purity) against known
phase bins, and optionally trains a dense ANN predictor baseline for
quality / latency / energy comparison on the same stream.
"""

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
    dense_mac_energy_pj,
    energy_ratio,
    latency_percentiles,
    model_size,
    run_benchmarks,
    snn_energy_pj,
)
from src.dst_snn.eval.baselines import DenseAnnPredictor, train_ann_predictor_step
from src.dst_snn.sensorimotor.homeostasis import (
    ExperienceBuffer,
    HomeostasisController,
    representation_stability,
    sleep_replay,
)
from src.dst_snn.sensorimotor.modules import MockActuator, SyntheticSensor
from src.dst_snn.sensorimotor.policy import IntrinsicMotorPolicy
from src.dst_snn.sensorimotor.probe import (
    cluster_purity,
    linear_probe_accuracy,
    nearest_centroid_accuracy,
)
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
    last_window = losses[-max(1, len(losses) // 4) :]
    first = sum(first_window) / len(first_window)
    last = sum(last_window) / len(last_window)
    if first <= 0:
        return 0.0
    return max(0.0, min(1.0, (first - last) / first))


def _snn_dense_mac_ops(
    *,
    sensory_size: int,
    motor_size: int,
    latent_size: int,
    time_steps: int,
) -> float:
    """Dense MAC proxy matching encoder Linear + predictor Linear over time."""
    in_dim = float(sensory_size + motor_size)
    t = float(time_steps)
    encoder = in_dim * float(latent_size) * t
    predictor = float(latent_size) * float(sensory_size) * t
    motor_head = float(latent_size) * float(motor_size) * t
    return encoder + predictor + motor_head


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
        replay_every: int = 8,
        replay_steps: int = 2,
        with_ann_baseline: bool = False,
        closed_loop: bool = True,
    ) -> None:
        self.steps = steps
        self.feature_size = feature_size
        self.motor_size = motor_size
        self.time_steps = time_steps
        self.latent_size = latent_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.replay_every = replay_every
        self.replay_steps = replay_steps
        self.with_ann_baseline = with_ann_baseline
        self.closed_loop = closed_loop
        self.runtime: SensorimotorRuntime | None = None
        self.model: PredictiveWorldModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.policy: IntrinsicMotorPolicy | None = None
        self.homeostasis: HomeostasisController | None = None
        self.buffer: ExperienceBuffer | None = None
        self.sensor = SyntheticSensor()
        self.actuator = MockActuator(
            on_command=self.sensor.apply_motor if closed_loop else None,
        )
        self.ann: DenseAnnPredictor | None = None
        self.ann_optimizer: torch.optim.Optimizer | None = None

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
            # Lower than default 1.0 so latent codes are not silent on sparse
            # hashed sensory spikes; homeostasis then regulates from there.
            threshold=0.45,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.policy = IntrinsicMotorPolicy(epsilon=0.25, temperature=0.5, seed=self.seed)
        self.homeostasis = HomeostasisController(target_rate=0.08, ema_alpha=0.1, gain=2.0)
        self.buffer = ExperienceBuffer(capacity=64)
        if self.with_ann_baseline:
            self.ann = DenseAnnPredictor(
                self.feature_size,
                self.motor_size,
                hidden=self.latent_size,
            ).to(self.device)
            self.ann_optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr)

    def run(self) -> RunResult:
        assert (
            self.runtime is not None
            and self.model is not None
            and self.optimizer is not None
            and self.policy is not None
            and self.homeostasis is not None
            and self.buffer is not None
        )
        progress = LearningProgress(alpha=0.2)
        losses: list[float] = []
        rewards: list[float] = []
        latencies_ms: list[float] = []
        spike_counts: list[float] = []
        active_fracs: list[float] = []
        selected_commands: list[list[dict[str, str]]] = []
        latent_trace: list[torch.Tensor] = []
        homeo_stats: list[dict[str, float]] = []
        replay_stats: list[dict[str, float]] = []
        latent_rates: list[float] = []
        phase_labels: list[int] = []
        latent_vectors: list[np.ndarray] = []
        ann_losses: list[float] = []
        ann_latencies_ms: list[float] = []
        phase_shift_trace: list[float] = []

        for step in range(self.steps):
            obs = self.sensor.observe(step)
            self.runtime.ingest(obs)
            phase_labels.append(int(obs.payload.get("phase_bin", step % 16)))
            phase_shift_trace.append(float(self.sensor.phase_shift))
            motor_activity, selected = self.policy.activity(self.runtime.registry)
            start = time.perf_counter()
            tick = self.runtime.tick(motor_activity)
            # Deliver decoded actions to actuator (sensor phase shifts via on_command).
            for action_msg in tick["actions"]:
                self.actuator.handle(action_msg)
            sensory = torch.from_numpy(tick["spikes"]).float().unsqueeze(0).to(self.device)
            motor = (
                torch.from_numpy(np.tile(motor_activity, (self.time_steps, 1)))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
            # Offsets from *previous* rates are applied inside ChronoPlastic this step.
            metrics = train_world_model_step(
                self.model,
                self.optimizer,
                sensory,
                motor,
                progress,
                homeostasis=self.homeostasis,
            )
            with torch.no_grad():
                # Snapshot latent with the same applied offsets for stability tracking.
                latent = self.model(
                    sensory,
                    motor,
                    homeostasis=self.homeostasis,
                    update_homeostasis=False,
                )["spikes"]
            latent_trace.append(latent.detach().cpu())
            latent_vectors.append(latent.detach().cpu().mean(dim=1).squeeze(0).numpy())
            latent_rates.append(float(metrics.get("latent_spike_rate", 0.0)))
            homeo_stats.append(
                {
                    "mean_rate": metrics.get("homeo_mean_rate", 0.0),
                    "instant_rate": metrics.get("homeo_instant_rate", 0.0),
                    "excess_rate": metrics.get("homeo_excess_rate", 0.0),
                    "threshold_offset": metrics.get("homeo_threshold_offset", 0.0),
                    "applied_offset_mean": metrics.get("applied_offset_mean", 0.0),
                }
            )
            salience = float(metrics.get("intrinsic_reward", 0.0)) + float(
                tick["global_signal"].get("novelty", 0.0)
            )
            self.buffer.add(sensory, motor, salience)
            if self.replay_every > 0 and (step + 1) % self.replay_every == 0:
                replay_stats.append(
                    sleep_replay(
                        self.model,
                        self.optimizer,
                        self.buffer,
                        steps=self.replay_steps,
                        device=self.device,
                    )
                )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            losses.append(metrics["prediction_loss"])
            reward = metrics.get("intrinsic_reward", 0.0)
            # Fatigue still gates policy; threshold homeostasis already acts on spikes.
            fatigue = float(tick["global_signal"].get("fatigue", 0.0))
            homeo_penalty = homeo_stats[-1]["excess_rate"]
            gated_reward = reward * (1.0 - 0.5 * fatigue) * (1.0 - min(0.5, homeo_penalty))
            rewards.append(gated_reward)
            self.policy.update(selected, gated_reward)
            selected_commands.append(selected)
            spike_counts.append(float(tick["spikes"].sum()))
            active_fracs.append(float((tick["spikes"] > 0).mean()))

            if self.ann is not None and self.ann_optimizer is not None:
                ann_start = time.perf_counter()
                ann_loss = train_ann_predictor_step(
                    self.ann, self.ann_optimizer, sensory, motor
                )
                ann_latencies_ms.append((time.perf_counter() - ann_start) * 1000.0)
                ann_losses.append(ann_loss)

        quality = _loss_reduction(losses)
        stability = representation_stability(latent_trace)
        lat = latency_percentiles(latencies_ms)
        size = model_size(self.model)
        energy_model = EnergyModel()
        spikes_per_inf = sum(spike_counts) / max(1, len(spike_counts))
        # Fanout ≈ latent→sensory predictor + motor head average width.
        fanout = max(1, (self.feature_size + self.motor_size) // 2)
        energy_pj = snn_energy_pj(spikes_per_inf, fanout, energy_model)
        dense_macs = _snn_dense_mac_ops(
            sensory_size=self.feature_size,
            motor_size=self.motor_size,
            latent_size=self.latent_size,
            time_steps=self.time_steps,
        )
        dense_energy = dense_mac_energy_pj(dense_macs, energy_model)

        feat = np.stack(latent_vectors, axis=0) if latent_vectors else np.zeros((0, self.latent_size))
        labels = np.asarray(phase_labels, dtype=np.int64)
        probe = linear_probe_accuracy(feat, labels, seed=self.seed)
        centroid = nearest_centroid_accuracy(feat, labels)
        purity = cluster_purity(feat, labels, seed=self.seed)

        baseline: MetricSet | None = None
        if self.ann is not None and ann_losses:
            ann_quality = _loss_reduction(ann_losses)
            ann_lat = latency_percentiles(ann_latencies_ms)
            ann_size = model_size(self.ann)
            ann_macs = self.ann.mac_ops_per_inference(self.time_steps)
            ann_energy = dense_mac_energy_pj(ann_macs, energy_model)
            baseline = MetricSet(
                quality=ann_quality,
                quality_metric="prediction_loss_reduction",
                latency_ms_p50=ann_lat["p50"],
                latency_ms_p95=ann_lat["p95"],
                spikes_per_inference=0.0,
                active_neuron_fraction=1.0,
                energy_pj=ann_energy,
                energy_source=energy_model.source,
                param_count=ann_size["param_count"],
                model_bytes=ann_size["model_bytes"],
                extra={
                    "initial_loss": ann_losses[0],
                    "final_loss": ann_losses[-1],
                    "losses": ann_losses,
                    "dense_mac_ops": ann_macs,
                    "energy_accounting": "sensorimotor_ann_mac_proxy_v1",
                    "hidden": self.latent_size,
                },
            )

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
                    "representation_stability": stability,
                    "mean_threshold_offset": (
                        sum(s["threshold_offset"] for s in homeo_stats) / max(1, len(homeo_stats))
                    ),
                    "final_threshold_offset": homeo_stats[-1]["threshold_offset"] if homeo_stats else 0.0,
                    "mean_latent_spike_rate": sum(latent_rates) / max(1, len(latent_rates)),
                    "final_latent_spike_rate": latent_rates[-1] if latent_rates else 0.0,
                    "homeostasis_target_rate": self.homeostasis.target_rate,
                    "replay_events": len(replay_stats),
                    "mean_replay_loss": (
                        sum(s["replay_loss"] for s in replay_stats) / max(1, len(replay_stats))
                        if replay_stats
                        else 0.0
                    ),
                    "final_fatigue": float(self.runtime.global_signal.get("fatigue", 0.0)),
                    "homeostasis_wired_to_threshold": True,
                    "closed_loop": self.closed_loop,
                    "final_phase_shift": float(self.sensor.phase_shift),
                    "phase_shift_range": (
                        float(max(phase_shift_trace) - min(phase_shift_trace))
                        if phase_shift_trace
                        else 0.0
                    ),
                    "actuator_commands_received": len(self.actuator.received),
                    "linear_probe_accuracy": probe["accuracy"],
                    "linear_probe_n_test": probe["n_test"],
                    "nearest_centroid_accuracy": centroid["accuracy"],
                    "cluster_purity": purity["purity"],
                    "dense_mac_ops": dense_macs,
                    "dense_energy_pj": dense_energy,
                    "energy_ratio_dense_over_snn": energy_ratio(energy_pj, dense_energy),
                    "energy_accounting": "sensorimotor_snn_ac_vs_dense_mac_v1",
                    "mean_step_latency_ms": float(sum(latencies_ms) / max(1, len(latencies_ms))),
                },
            ),
            baseline=baseline,
            meta={
                "feature_size": self.feature_size,
                "motor_size": self.motor_size,
                "time_steps": self.time_steps,
                "latent_size": self.latent_size,
                "seed": self.seed,
                "replay_every": self.replay_every,
                "closed_loop": self.closed_loop,
                "with_ann_baseline": self.with_ann_baseline,
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
    parser.add_argument("--replay-every", type=int, default=8)
    parser.add_argument("--replay-steps", type=int, default=2)
    parser.add_argument(
        "--with-ann-baseline",
        action="store_true",
        help="Train a dense ANN predictor on the same closed-loop stream.",
    )
    parser.add_argument(
        "--no-closed-loop",
        action="store_true",
        help="Disable actuator→sensor phase coupling (open-loop sine only).",
    )
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
        replay_every=args.replay_every,
        replay_steps=args.replay_steps,
        with_ann_baseline=args.with_ann_baseline,
        closed_loop=not args.no_closed_loop,
    )
    print(run_benchmarks([runner], args.out_dir)[0].to_json())


if __name__ == "__main__":
    main()
