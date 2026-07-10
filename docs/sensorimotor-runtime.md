# Sensorimotor Runtime

This runtime is the first implementation increment for the embodied
sensorimotor design in `docs/superpowers/specs/2026-07-09-embodied-sensorimotor-runtime-design.md`.

## What Is Implemented

- JSON message protocol:
  - `register`
  - `deregister`
  - `observation`
  - `action`
  - `global_signal`
  - `trace`
- Dynamic module registry with fixed-size hashed sensory and motor spaces.
- Observation encoder that converts arbitrary JSON-like payloads into sparse
  spike tensors.
- Action decoder that maps motor activity into actuator commands or axes.
- Predictive world model using `ChronoPlasticLIFLayer`.
- EMA learning-progress tracker for intrinsic reward.
- Intrinsic-reward motor policy that updates actuator command scores.
- In-process runtime loop that emits action, global-signal, and trace messages.
- JSONL transport helpers for deterministic replay.
- Runtime checkpoint save/load.
- Predictive world-model checkpoint save/load.
- Synthetic sensor and mock actuator for headless tests.

## Minimal Example

```python
import numpy as np

from src.dst_snn.sensorimotor.modules import MockActuator, SyntheticSensor
from src.dst_snn.sensorimotor.registry import ModuleRegistry
from src.dst_snn.sensorimotor.runtime import SensorimotorRuntime

registry = ModuleRegistry(feature_size=64, motor_size=16)
runtime = SensorimotorRuntime(registry, time_steps=8)

sensor = SyntheticSensor()
actuator = MockActuator()

runtime.ingest(sensor.register())
runtime.ingest(actuator.register())
runtime.ingest(sensor.observe(step=0))

activity = np.zeros(16, dtype=np.float32)
activity[registry.motor_index(actuator.id, "left") % 16] = 1.0
result = runtime.tick(activity)

for message in result["messages"]:
    print(message.type, message.id, message.payload)
```

## JSONL Replay

```python
from src.dst_snn.sensorimotor.registry import ModuleRegistry
from src.dst_snn.sensorimotor.runtime import SensorimotorRuntime
from src.dst_snn.sensorimotor.transport import replay_jsonl

runtime = SensorimotorRuntime(ModuleRegistry())
results = replay_jsonl(runtime, "data/sensorimotor/session.jsonl")
```

The replay helper ingests every message and runs one tick after each
`observation`, making module streams deterministic and easy to test offline.

## Checkpoints

```python
runtime.save("artifacts/sensorimotor/runtime.json")
runtime = SensorimotorRuntime.load("artifacts/sensorimotor/runtime.json")
```

Checkpoints currently cover runtime step count, registry specs, latest
observations, and global signals. Predictive model checkpoints are separate:

```python
from src.dst_snn.sensorimotor.checkpoint import (
    load_world_model_checkpoint,
    save_world_model_checkpoint,
)

save_world_model_checkpoint("artifacts/sensorimotor/world-model.pt", model, optimizer, progress)
model, optimizer, progress, extra = load_world_model_checkpoint(
    "artifacts/sensorimotor/world-model.pt",
    with_optimizer=True,
)
```

## Intrinsic Motor Policy

```python
from src.dst_snn.sensorimotor.policy import IntrinsicMotorPolicy

policy = IntrinsicMotorPolicy(epsilon=0.25, seed=0)
motor_activity, selected = policy.activity(registry)
policy.update(selected, intrinsic_reward=0.4)
```

The policy is intentionally simple: it samples actuator commands with a
softmax over learned command scores and updates selected command scores toward
the latest intrinsic reward.

## Still Deferred

- WebSocket server/client transport.
- Real hardware bridges such as serial/USB motor arms and tactile sensors.
- Long-running autonomous policy loop that directly couples world-model
  predictions to richer motor exploration.
