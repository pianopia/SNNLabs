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
- Serial / USB motor + tactile bridges (`SerialMotorBridge`, `SerialTactileSensor`)
  with `MockSerialPort` for offline tests (optional `pyserial` for real ports).
- EDEN ↔ Python bridge (`eden_bridge.py` + `EDEN/src/snn/sensorimotorBridge.ts`)
  mapping body / global_signal / spike events onto the shared protocol.

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

## WebSocket transport

```python
from src.dst_snn.sensorimotor.websocket_transport import (
    LocalMessageHub,
    serve_runtime,
    websockets_available,
)

# In-process fan-out (no websockets package required):
hub = LocalMessageHub()

# Real WebSocket server (requires `pip install websockets`):
# await serve_runtime(runtime, host="127.0.0.1", port=8766)
```

Each WS frame is one JSON sensorimotor message. Observation frames trigger a
runtime tick; action / global_signal / trace messages are broadcast to clients.

## Webcam sensor

```python
from src.dst_snn.sensorimotor.modules import WebcamSensor

sensor = WebcamSensor(use_camera=False)  # synthetic frame stream for CI
# sensor = WebcamSensor(use_camera=True, camera_index=0)  # needs OpenCV
runtime.ingest(sensor.register())
runtime.ingest(sensor.observe())
```

## Serial / USB hardware bridges

Offline by default via `MockSerialPort`. Real ports need `pip install pyserial`.

```python
from src.dst_snn.sensorimotor.modules import (
    MockSerialPort,
    SerialMotorBridge,
    SerialTactileSensor,
)
from src.dst_snn.sensorimotor.protocol import SensorimotorMessage

port = MockSerialPort()
motor = SerialMotorBridge(n_channels=4)
motor.attach(port)
motor.on_action(SensorimotorMessage(type="action", id="core", payload={"values": [0.1, 0, 0, 0]}))

tactile = SerialTactileSensor(n_taxels=8)
tactile.attach(port)
tactile.inject_raw_json({"type": "tactile", "values": [0.0] * 8})
obs = tactile.poll()  # → observation message
```

JSONL wire format for motors: `{"type":"motor","channels":[...],"ts":...}`.
Binary mode (`binary=True`) packs little-endian float32 channels.

## EDEN ↔ Python bridge

Python:

```python
from src.dst_snn.sensorimotor.eden_bridge import EdenBridgeSession

session = EdenBridgeSession()
runtime.ingest(session.register())
for msg in session.ingest_events([{"kind": "body", "meta": {"gait_drive": 0.4}}]):
    runtime.ingest(msg)
```

TypeScript client (`EDEN/src/snn/sensorimotorBridge.ts`):

```ts
import { SensorimotorBridge } from './snn/sensorimotorBridge';

const bridge = new SensorimotorBridge('ws://127.0.0.1:8766');
bridge.connect();
bridge.sendEvents([{ kind: 'body', meta: { gait_drive: 0.4 } }]);
```

Serve the Python core with `serve_runtime(runtime, port=8766)` (requires
`websockets`).

## EDEN autonomous biotope (generated bodies)

Design: `docs/superpowers/specs/2026-07-11-eden-autonomous-generated-body-design.md`

In the EDEN client, SNN Life is **on by default**. On spawn each creature gets a
deterministic procedural body (seed → GLB blob URL) with no setup UI:

- `EDEN/src/snn/proceduralBody.ts` — multi-part body mesh from seed
- `EDEN/src/snn/bodyGlb.ts` — minimal glTF binary encoder
- `EDEN/src/snn/generatedBodyRegistry.ts` — seed persistence + object URLs

Learning continues via the existing embodied SNN tick (reward-modulated STDP)
while creatures walk; users do not need to configure generation parameters.
Python mutual training remains a later slice.

### Vision → morph → external construct

Default learning goal `imitateAndConstruct`:

1. **Vision (coarse):** nearby object size/shape → `visionWidth/Height/Depth/Salience`
2. **Morph reward:** improve body extents match to the seen shape (`motor:imitate_shape`)
3. **Construct:** `motor:construct_object` places a **new world primitive** (not self-mesh)
   with cooldown + energy cost + construct reward

Helpers: `EDEN/src/snn/visionShape.ts`, `constructAction.ts`. Real RGB vision / Track1
Blender ops remain later upgrades.

## Homeostasis and sleep replay

Homeostatic offsets are **wired into ChronoPlastic thresholds**:

```
V_th_i' = V_th + clamp(gain * (rate_i - target), -max_offset, max_offset)
```

```python
from src.dst_snn.sensorimotor.homeostasis import (
    ExperienceBuffer,
    HomeostasisController,
    representation_stability,
    sleep_replay,
)
from src.dst_snn.sensorimotor.world_model import PredictiveWorldModel, train_world_model_step

homeo = HomeostasisController(target_rate=0.05, gain=2.0)
model = PredictiveWorldModel(sensory_size=64, motor_size=16, latent_size=32)

# train_world_model_step applies previous offsets inside the encoder, then
# updates EMA rates from the new spikes.
metrics = train_world_model_step(model, optimizer, sensory, motor, homeostasis=homeo)
# metrics["latent_spike_rate"], metrics["homeo_threshold_offset"], ...

buffer = ExperienceBuffer(capacity=64)
buffer.add(sensory, motor, salience=0.8)
sleep_replay(model, optimizer, buffer, steps=4)
```

You can also pass an explicit offset tensor:

```python
offset = homeo.tensor_offsets(model.latent_size, device=x.device)
out = model(sensory, motor, threshold_offset=offset)
```

The synthetic sensorimotor benchmark records representation stability, latent
spike rate, applied threshold offset, replay events, and fatigue in
`MetricSet.extra` (`homeostasis_wired_to_threshold: true`).

## Closed-loop synthetic world

`SyntheticSensor` exposes a discrete `phase_bin` ground-truth label. Motor
commands from `MockActuator` shift the sensor phase when wired with
`on_command=sensor.apply_motor` (the synthetic runner does this by default).

```python
from src.dst_snn.sensorimotor.modules import MockActuator, SyntheticSensor

sensor = SyntheticSensor()
actuator = MockActuator(on_command=sensor.apply_motor)
# after runtime.tick(...): actuator.handle(action_message)
```

## Representation probes (design B-7)

```python
from src.dst_snn.sensorimotor.probe import (
    cluster_purity,
    linear_probe_accuracy,
    nearest_centroid_accuracy,
)

probe = linear_probe_accuracy(latent_vectors, phase_bins, seed=0)
purity = cluster_purity(latent_vectors, phase_bins, seed=0)
```

The synthetic runner records these under `MetricSet.extra` together with
dense-MAC energy proxies and optional ANN predictor baseline metrics.

## Still Deferred

- Real hardware bridges such as serial/USB motor arms and tactile sensors.
- Long-running autonomous policy loop with continuous WebSocket modules.
