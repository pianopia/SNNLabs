import std/[strformat, strutils]

import core

proc buildDemoNetwork(): Network =
  result = initNetwork(dtMs = 1.0)

  result.addNeuron initNeuron(0, "sensor:light", threshold = 0.7, tauMs = 12.0)
  result.addNeuron initNeuron(1, "sensor:sound", threshold = 0.7, tauMs = 12.0)
  result.addNeuron initNeuron(2, "concept:presence", threshold = 0.9, tauMs = 18.0)
  result.addNeuron initNeuron(3, "concept:motion", threshold = 0.9, tauMs = 18.0)
  result.addNeuron initNeuron(4, "attractor:attention", threshold = 1.0, tauMs = 24.0)

  result.addSynapse initSynapse(0, pre = 0, post = 2, w = 0.46)
  result.addSynapse initSynapse(1, pre = 1, post = 3, w = 0.46)
  result.addSynapse initSynapse(2, pre = 2, post = 4, w = 0.38)
  result.addSynapse initSynapse(3, pre = 3, post = 4, w = 0.38)
  result.addSynapse initSynapse(4, pre = 4, post = 2, w = 0.18, dPre = 0.008, dPost = -0.009)
  result.addSynapse initSynapse(5, pre = 4, post = 3, w = 0.18, dPre = 0.008, dPost = -0.009)

proc stimulus(tMs: float, neuronCount: int): seq[float] =
  result = newSeq[float](neuronCount)

  if tMs in 5.0 .. 18.0:
    result[0] = 1.15
  if tMs in 18.0 .. 32.0:
    result[1] = 1.10
  if tMs in 52.0 .. 68.0:
    result[0] = 1.20
    result[1] = 1.08

when isMainModule:
  var net = buildDemoNetwork()
  net.run(steps = 90, stimulus = stimulus)

  let events = net.drainTrace()
  for event in events:
    if event.kind in ["spike", "weight", "input"]:
      echo event.toJsonLine()

  echo &"# neurons={net.neurons.len} synapses={net.synapses.len} events={events.len}"
  for synapse in net.synapses:
    echo &"# synapse {synapse.id}: {synapse.pre}->{synapse.post} w={synapse.w.formatFloat(ffDecimal, 4)}"
