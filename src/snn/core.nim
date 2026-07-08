import std/[json, math]

type
  Neuron* = object
    id*: int
    label*: string
    v*: float
    rest*: float
    reset*: float
    threshold*: float
    tauMs*: float
    refractoryMs*: float
    refractoryLeftMs*: float
    lastSpikeMs*: float

  Synapse* = object
    id*: int
    pre*: int
    post*: int
    w*: float
    minW*: float
    maxW*: float
    delayMs*: float
    aPre*: float
    aPost*: float
    tauPreMs*: float
    tauPostMs*: float
    dPre*: float
    dPost*: float

  SpikeDelivery = object
    deliverAtMs: float
    post: int
    synapse: int
    amount: float

  TraceEvent* = object
    tMs*: float
    kind*: string
    neuron*: int
    synapse*: int
    value*: float
    label*: string
    meta*: JsonNode

  Network* = object
    tMs*: float
    dtMs*: float
    learningEnabled*: bool
    learningRate*: float
    rewardSignal*: float
    neurons*: seq[Neuron]
    synapses*: seq[Synapse]
    pending: seq[SpikeDelivery]
    trace*: seq[TraceEvent]

proc initNeuron*(
    id: int,
    label: string,
    threshold = 1.0,
    rest = 0.0,
    reset = 0.0,
    tauMs = 20.0,
    refractoryMs = 2.0
): Neuron =
  Neuron(
    id: id,
    label: label,
    v: rest,
    rest: rest,
    reset: reset,
    threshold: threshold,
    tauMs: tauMs,
    refractoryMs: refractoryMs,
    refractoryLeftMs: 0.0,
    lastSpikeMs: -Inf
  )

proc initSynapse*(
    id: int,
    pre: int,
    post: int,
    w = 0.25,
    minW = 0.0,
    maxW = 1.0,
    delayMs = 1.0,
    tauPreMs = 20.0,
    tauPostMs = 20.0,
    dPre = 0.015,
    dPost = -0.016
): Synapse =
  Synapse(
    id: id,
    pre: pre,
    post: post,
    w: w,
    minW: minW,
    maxW: maxW,
    delayMs: delayMs,
    aPre: 0.0,
    aPost: 0.0,
    tauPreMs: tauPreMs,
    tauPostMs: tauPostMs,
    dPre: dPre,
    dPost: dPost
  )

proc initNetwork*(dtMs = 1.0): Network =
  Network(
    tMs: 0.0,
    dtMs: dtMs,
    learningEnabled: true,
    learningRate: 1.0,
    rewardSignal: 1.0,
    neurons: @[],
    synapses: @[],
    pending: @[],
    trace: @[]
  )

proc addNeuron*(net: var Network, neuron: Neuron) =
  net.neurons.add neuron

proc addSynapse*(net: var Network, synapse: Synapse) =
  net.synapses.add synapse

proc clampWeight(s: Synapse, w: float): float =
  min(max(w, s.minW), s.maxW)

proc plasticityScale(net: Network): float =
  if net.learningEnabled:
    net.learningRate * net.rewardSignal
  else:
    0.0

proc record(
    net: var Network,
    kind: string,
    neuron = -1,
    synapse = -1,
    value = 0.0,
    label = "",
    meta: JsonNode = newJObject()
) =
  net.trace.add TraceEvent(
    tMs: net.tMs,
    kind: kind,
    neuron: neuron,
    synapse: synapse,
    value: value,
    label: label,
    meta: meta
  )

proc toJson*(event: TraceEvent): JsonNode =
  result = %*{
    "t_ms": event.tMs,
    "kind": event.kind,
    "value": event.value
  }
  if event.neuron >= 0:
    result["neuron"] = %event.neuron
  if event.synapse >= 0:
    result["synapse"] = %event.synapse
  if event.label.len > 0:
    result["label"] = %event.label
  if event.meta.kind != JNull and event.meta.len > 0:
    result["meta"] = event.meta

proc toJsonLine*(event: TraceEvent): string =
  $event.toJson()

proc setLearningEnabled*(net: var Network, enabled: bool) =
  net.learningEnabled = enabled
  net.record("learning", value = (if enabled: 1.0 else: 0.0), label = "enabled")

proc setLearningRate*(net: var Network, learningRate: float) =
  net.learningRate = learningRate
  net.record("learning", value = learningRate, label = "learning_rate")

proc setRewardSignal*(net: var Network, reward: float) =
  net.rewardSignal = reward
  net.record("global_signal", value = reward, label = "reward")

proc decayTraces(net: var Network) =
  for synapse in net.synapses.mitems:
    synapse.aPre *= exp(-net.dtMs / synapse.tauPreMs)
    synapse.aPost *= exp(-net.dtMs / synapse.tauPostMs)

proc deliverPending(net: var Network, currents: var seq[float]) =
  var remaining: seq[SpikeDelivery] = @[]
  for delivery in net.pending:
    if delivery.deliverAtMs <= net.tMs:
      currents[delivery.post] += delivery.amount
      net.record(
        "input",
        neuron = delivery.post,
        synapse = delivery.synapse,
        value = delivery.amount,
        label = net.neurons[delivery.post].label
      )
    else:
      remaining.add delivery
  net.pending = remaining

proc applyPreSpike(net: var Network, pre: int) =
  let scale = net.plasticityScale()
  for synapse in net.synapses.mitems:
    if synapse.pre == pre:
      synapse.aPre += synapse.dPre
      synapse.w = synapse.clampWeight(synapse.w + synapse.aPost * scale)
      net.pending.add SpikeDelivery(
        deliverAtMs: net.tMs + synapse.delayMs,
        post: synapse.post,
        synapse: synapse.id,
        amount: synapse.w
      )
      net.record(
        "weight",
        synapse = synapse.id,
        value = synapse.w,
        meta = %*{
          "pre": synapse.pre,
          "post": synapse.post,
          "rule": "on_pre",
          "plasticity_scale": scale
        }
      )

proc applyPostSpike(net: var Network, post: int) =
  let scale = net.plasticityScale()
  for synapse in net.synapses.mitems:
    if synapse.post == post:
      synapse.aPost += synapse.dPost
      synapse.w = synapse.clampWeight(synapse.w + synapse.aPre * scale)
      net.record(
        "weight",
        synapse = synapse.id,
        value = synapse.w,
        meta = %*{
          "pre": synapse.pre,
          "post": synapse.post,
          "rule": "on_post",
          "plasticity_scale": scale
        }
      )

proc step*(net: var Network, externalCurrents: openArray[float]) =
  if externalCurrents.len != net.neurons.len:
    raise newException(ValueError, "externalCurrents length must match neuron count")

  net.decayTraces()

  var currents = newSeq[float](net.neurons.len)
  for i, value in externalCurrents:
    currents[i] = value
    if abs(value) > 1e-12:
      net.record("input", neuron = i, value = value, label = net.neurons[i].label)
  net.deliverPending(currents)

  var spiked: seq[int] = @[]
  for i, neuron in net.neurons.mpairs:
    if neuron.refractoryLeftMs > 0.0:
      neuron.refractoryLeftMs = max(0.0, neuron.refractoryLeftMs - net.dtMs)
      net.record("membrane", neuron = i, value = neuron.v, label = neuron.label)
      continue

    let dv = (neuron.rest - neuron.v) * (net.dtMs / neuron.tauMs) + currents[i]
    neuron.v += dv
    net.record("membrane", neuron = i, value = neuron.v, label = neuron.label)

    if neuron.v >= neuron.threshold:
      neuron.v = neuron.reset
      neuron.refractoryLeftMs = neuron.refractoryMs
      neuron.lastSpikeMs = net.tMs
      spiked.add i
      net.record("spike", neuron = i, value = 1.0, label = neuron.label)

  for neuronId in spiked:
    net.applyPreSpike(neuronId)
  for neuronId in spiked:
    net.applyPostSpike(neuronId)

  net.tMs += net.dtMs

proc run*(net: var Network, steps: int, stimulus: proc(tMs: float, neuronCount: int): seq[float]) =
  for _ in 0 ..< steps:
    net.step(stimulus(net.tMs, net.neurons.len))

proc drainTrace*(net: var Network): seq[TraceEvent] =
  result = net.trace
  net.trace = @[]

proc traceAsJsonl*(events: openArray[TraceEvent]): string =
  for event in events:
    result.add event.toJsonLine()
    result.add "\n"
