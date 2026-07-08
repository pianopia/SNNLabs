import std/[json, strformat, strutils]

import core

type
  BodyState = object
    x: float
    targetX: float
    energy: float

proc buildBodyNetwork(): Network =
  result = initNetwork(dtMs = 1.0)
  result.setLearningRate(0.7)

  result.addNeuron initNeuron(0, "sensor:target_left", threshold = 0.8, tauMs = 14.0)
  result.addNeuron initNeuron(1, "sensor:target_right", threshold = 0.8, tauMs = 14.0)
  result.addNeuron initNeuron(2, "interneuron:approach_left", threshold = 0.9, tauMs = 18.0)
  result.addNeuron initNeuron(3, "interneuron:approach_right", threshold = 0.9, tauMs = 18.0)
  result.addNeuron initNeuron(4, "motor:left", threshold = 0.95, tauMs = 16.0)
  result.addNeuron initNeuron(5, "motor:right", threshold = 0.95, tauMs = 16.0)

  result.addSynapse initSynapse(0, pre = 0, post = 2, w = 0.42)
  result.addSynapse initSynapse(1, pre = 1, post = 3, w = 0.42)
  result.addSynapse initSynapse(2, pre = 2, post = 4, w = 0.52)
  result.addSynapse initSynapse(3, pre = 3, post = 5, w = 0.52)
  result.addSynapse initSynapse(4, pre = 4, post = 2, w = 0.16, dPre = 0.006, dPost = -0.007)
  result.addSynapse initSynapse(5, pre = 5, post = 3, w = 0.16, dPre = 0.006, dPost = -0.007)

proc sense(body: BodyState, neuronCount: int): seq[float] =
  result = newSeq[float](neuronCount)
  let dx = body.targetX - body.x
  if dx < -0.05:
    result[0] = min(1.4, abs(dx) * 2.0)
  elif dx > 0.05:
    result[1] = min(1.4, abs(dx) * 2.0)

proc summarizeAction(events: openArray[TraceEvent]): JsonNode =
  var left = 0
  var right = 0
  for event in events:
    if event.kind == "spike":
      if event.label == "motor:left":
        inc left
      elif event.label == "motor:right":
        inc right
  %*{"left": left, "right": right, "dx": float(right - left) * 0.08}

when isMainModule:
  var body = BodyState(x: 0.35, targetX: -0.45, energy: 1.0)
  var net = buildBodyNetwork()

  for _ in 0 ..< 120:
    let beforeDistance = abs(body.targetX - body.x)
    net.step(sense(body, net.neurons.len))

    let events = net.drainTrace()
    let action = summarizeAction(events)
    body.x += action["dx"].getFloat()
    body.energy = max(0.0, body.energy - 0.003)

    let afterDistance = abs(body.targetX - body.x)
    let reward = if afterDistance < beforeDistance: 1.35 else: 0.45
    net.setRewardSignal(reward * body.energy)

    echo $(%*{
      "t_ms": net.tMs,
      "kind": "body",
      "x": body.x,
      "target_x": body.targetX,
      "energy": body.energy,
      "reward": net.rewardSignal,
      "action": action
    })

    for event in events:
      if event.kind in ["spike", "weight", "global_signal"]:
        echo event.toJsonLine()

  echo &"# final_x={body.x.formatFloat(ffDecimal, 4)} target_x={body.targetX.formatFloat(ffDecimal, 4)}"
