import { useMemo, useState, type CSSProperties } from 'react';
import { Link } from 'react-router-dom';
import { decodeEdenSnnModelFile, type EdenSnnModelEntry, type EdenSnnModelFile } from '../snn/modelFile';
import type { BrowserLanguageSnnSnapshot } from '../snn/languageLearner';

const panel: CSSProperties = {
  background: 'rgba(12, 16, 20, 0.82)',
  border: '1px solid rgba(150, 190, 220, 0.18)',
  borderRadius: 8,
  padding: 16,
};

const tableStyle: CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: 12,
};

const cellStyle: CSSProperties = {
  borderBottom: '1px solid rgba(255,255,255,0.08)',
  padding: '7px 8px',
  textAlign: 'left',
};

const numberCell: CSSProperties = {
  ...cellStyle,
  fontVariantNumeric: 'tabular-nums',
  textAlign: 'right',
};

const format = (value: number) => Number.isFinite(value) ? value.toFixed(4) : '-';

const average = (values: number[]) => (
  values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length
);

const isEmbodiedSnapshot = (snapshot: EdenSnnModelEntry['snapshot']): snapshot is Extract<EdenSnnModelEntry['snapshot'], { network: unknown }> => (
  typeof snapshot === 'object' && snapshot !== null && 'network' in snapshot
);

const isBrowserLanguageSnapshot = (snapshot: EdenSnnModelEntry['snapshot']): snapshot is BrowserLanguageSnnSnapshot => (
  typeof snapshot === 'object' && snapshot !== null && (snapshot as BrowserLanguageSnnSnapshot).domain === 'browser-language'
);

const analyzeModel = (model: EdenSnnModelEntry | null) => {
  if (!model) return null;
  if (!isEmbodiedSnapshot(model.snapshot)) return null;
  const neurons = model.snapshot.network.neurons;
  const synapses = model.snapshot.network.synapses;
  const spikeStats = model.snapshot.network.spikeStats ?? {
    ticks: 0,
    positiveSpikes: 0,
    negativeSpikes: 0,
    absoluteSpikeMass: 0,
    sparseAcOps: 0,
    denseMacOps: 0,
  };
  const weights = synapses.map((synapse) => synapse.w);
  const traces = synapses.flatMap((synapse) => [Math.abs(synapse.aPre), Math.abs(synapse.aPost)]);
  const activeNeurons = neurons.filter((neuron) => Math.abs(neuron.v) > 0.01).length;
  const refractoryNeurons = neurons.filter((neuron) => neuron.refractoryLeftMs > 0).length;
  const strongestSynapses = [...synapses].sort((a, b) => b.w - a.w).slice(0, 12);
  const denseMacOps = Math.max(1, spikeStats.denseMacOps);
  const sparseAcOps = spikeStats.sparseAcOps;
  const fireRate = spikeStats.ticks > 0 && neurons.length > 0
    ? spikeStats.absoluteSpikeMass / (spikeStats.ticks * neurons.length)
    : 0;
  const sparseRatio = sparseAcOps / denseMacOps;
  const energyRatio = sparseAcOps > 0 ? denseMacOps / sparseAcOps : 0;
  const signedMass = spikeStats.positiveSpikes + spikeStats.negativeSpikes;
  const negativeSpikeShare = signedMass > 0 ? spikeStats.negativeSpikes / signedMass : 0;
  const weightBuckets = Array.from({ length: 10 }, (_, index) => {
    const min = index / 10;
    const max = (index + 1) / 10;
    return {
      min,
      max,
      count: weights.filter((weight) => index === 9 ? weight >= min && weight <= max : weight >= min && weight < max).length,
    };
  });

  return {
    neurons,
    synapses,
    activeNeurons,
    refractoryNeurons,
    strongestSynapses,
    weightBuckets,
    spikeStats,
    fireRate,
    sparseRatio,
    energyRatio,
    negativeSpikeShare,
    neuronType: model.snapshot.network.neuronType ?? 'lif',
    spikeRangeD: model.snapshot.network.spikeRangeD ?? 1,
    weightMin: weights.length > 0 ? Math.min(...weights) : 0,
    weightMax: weights.length > 0 ? Math.max(...weights) : 0,
    weightAvg: average(weights),
    traceAvg: average(traces),
  };
};

export default function SnnDashboard() {
  const [modelFile, setModelFile] = useState<EdenSnnModelFile | null>(null);
  const [selectedCreatureId, setSelectedCreatureId] = useState('');
  const [error, setError] = useState('');

  const selectedModel = useMemo(() => {
    if (!modelFile) return null;
    return modelFile.models.find((model) => model.creatureId === selectedCreatureId) ?? modelFile.models[0] ?? null;
  }, [modelFile, selectedCreatureId]);

  const analysis = useMemo(() => analyzeModel(selectedModel), [selectedModel]);
  const languageSnapshot = selectedModel && isBrowserLanguageSnapshot(selectedModel.snapshot)
    ? selectedModel.snapshot
    : null;

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    setError('');
    try {
      const decoded = await decodeEdenSnnModelFile(file);
      setModelFile(decoded);
      setSelectedCreatureId(decoded.models[0]?.creatureId ?? '');
    } catch (err) {
      setModelFile(null);
      setSelectedCreatureId('');
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const maxBucket = Math.max(1, ...(analysis?.weightBuckets.map((bucket) => bucket.count) ?? [1]));

  return (
    <div style={{
      minHeight: '100vh',
      background: '#05070a',
      color: '#e8f2ff',
      fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      padding: 24,
      boxSizing: 'border-box',
    }}>
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, marginBottom: 18 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, letterSpacing: 0 }}>SNN Model Dashboard</h1>
          <div style={{ marginTop: 4, fontSize: 13, color: '#93abc2' }}>
            Import `.edensnn` files and inspect neuron state, STDP traces, and synaptic weights.
          </div>
        </div>
        <Link to="/" style={{ color: '#9fd3ff', textDecoration: 'none', fontSize: 13 }}>
          Back to EDEN
        </Link>
      </header>

      <section style={{ ...panel, display: 'grid', gap: 12, marginBottom: 16 }}>
        <input
          type="file"
          aria-label="Import EDEN SNN model file"
          accept=".edensnn,application/x-edensnn"
          onChange={(event) => void handleFile(event.target.files?.[0])}
          style={{
            color: '#dfefff',
            background: '#0c1118',
            border: '1px solid rgba(150,190,220,0.28)',
            borderRadius: 6,
            padding: 10,
          }}
        />
        {error && <div style={{ color: '#ff9f9f', fontSize: 13 }}>{error}</div>}
        {modelFile && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, alignItems: 'end' }}>
            <label style={{ display: 'grid', gap: 6, fontSize: 12, color: '#a9c2d9' }}>
              Model
              <select
                value={selectedModel?.creatureId ?? ''}
                onChange={(event) => setSelectedCreatureId(event.target.value)}
                style={{
                  background: '#080c12',
                  color: '#e8f2ff',
                  border: '1px solid rgba(150,190,220,0.28)',
                  borderRadius: 6,
                  padding: 8,
                }}
              >
                {modelFile.models.map((model) => (
                  <option key={model.creatureId} value={model.creatureId}>{model.creatureId}</option>
                ))}
              </select>
            </label>
            <div style={{ fontSize: 12, color: '#93abc2' }}>Exported: {modelFile.exportedAt}</div>
            <div style={{ fontSize: 12, color: '#93abc2' }}>Container v{modelFile.version}</div>
            <div style={{ fontSize: 12, color: '#93abc2' }}>Models in file: {modelFile.modelCount}</div>
            <div style={{ fontSize: 12, color: '#93abc2' }}>Goal: {modelFile.learningGoal ?? '-'}</div>
          </div>
        )}
      </section>

      {languageSnapshot && (
        <>
          <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
            {[
              ['Domain', languageSnapshot.domain],
              ['Neuron type', languageSnapshot.config.neuronType],
              ['Spike range D', languageSnapshot.config.spikeRangeD],
              ['Vocabulary', languageSnapshot.vocabulary.length],
              ['Associations', languageSnapshot.associations.length],
              ['Observations', languageSnapshot.stats.observations],
              ['Positive spikes', format(languageSnapshot.stats.positiveSpikes)],
              ['Negative spikes', format(languageSnapshot.stats.negativeSpikes)],
              ['Energy benefit est.', languageSnapshot.stats.sparseAcOps > 0 ? `${format(languageSnapshot.stats.denseMacOps / languageSnapshot.stats.sparseAcOps)}x` : '-'],
            ].map(([label, value]) => (
              <div key={label} style={panel}>
                <div style={{ fontSize: 12, color: '#91a7bd' }}>{label}</div>
                <div style={{ marginTop: 6, fontSize: 20, fontWeight: 700 }}>{value}</div>
              </div>
            ))}
          </section>

          <section style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 16 }}>
            <div style={panel}>
              <h2 style={{ margin: '0 0 12px', fontSize: 16 }}>Language Token Neurons</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={cellStyle}>ID</th>
                      <th style={cellStyle}>Token</th>
                      <th style={numberCell}>Count</th>
                      <th style={numberCell}>+Spike</th>
                      <th style={numberCell}>-Spike</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...languageSnapshot.vocabulary].sort((a, b) => b.count - a.count).slice(0, 40).map((neuron) => (
                      <tr key={neuron.id}>
                        <td style={cellStyle}>{neuron.id}</td>
                        <td style={cellStyle}>{neuron.token}</td>
                        <td style={numberCell}>{neuron.count}</td>
                        <td style={numberCell}>{format(neuron.positiveSpikeMass)}</td>
                        <td style={numberCell}>{format(neuron.negativeSpikeMass)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div style={panel}>
              <h2 style={{ margin: '0 0 12px', fontSize: 16 }}>Language Associations</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={cellStyle}>Pre {'->'} Post</th>
                      <th style={numberCell}>Weight</th>
                      <th style={numberCell}>aPre</th>
                      <th style={numberCell}>aPost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...languageSnapshot.associations].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 40).map((association) => {
                      const pre = languageSnapshot.vocabulary.find((neuron) => neuron.id === association.pre)?.token ?? association.pre;
                      const post = languageSnapshot.vocabulary.find((neuron) => neuron.id === association.post)?.token ?? association.post;
                      return (
                        <tr key={association.id}>
                          <td style={cellStyle}>{pre} {'->'} {post}</td>
                          <td style={numberCell}>{format(association.w)}</td>
                          <td style={numberCell}>{format(association.aPre)}</td>
                          <td style={numberCell}>{format(association.aPost)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </>
      )}

      {analysis && selectedModel && (
        <>
          <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
            {[
              ['Neurons', analysis.neurons.length],
              ['Synapses', analysis.synapses.length],
              ['Active neurons', analysis.activeNeurons],
              ['Refractory neurons', analysis.refractoryNeurons],
              ['Neuron type', analysis.neuronType],
              ['Spike range D', analysis.spikeRangeD],
              ['Fire rate', format(analysis.fireRate)],
              ['Negative spike share', `${(analysis.negativeSpikeShare * 100).toFixed(1)}%`],
              ['Sparse AC / dense MAC', `${(analysis.sparseRatio * 100).toFixed(1)}%`],
              ['Energy benefit est.', analysis.energyRatio > 0 ? `${format(analysis.energyRatio)}x` : '-'],
              ['Avg weight', format(analysis.weightAvg)],
              ['Avg STDP trace', format(analysis.traceAvg)],
            ].map(([label, value]) => (
              <div key={label} style={panel}>
                <div style={{ fontSize: 12, color: '#91a7bd' }}>{label}</div>
                <div style={{ marginTop: 6, fontSize: 22, fontWeight: 700 }}>{value}</div>
              </div>
            ))}
          </section>

          <section style={{ ...panel, marginBottom: 16 }}>
            <h2 style={{ margin: '0 0 12px', fontSize: 16 }}>Spike Efficiency</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, fontSize: 12, color: '#b8cbe0' }}>
              <div>Ticks: {analysis.spikeStats.ticks}</div>
              <div>Positive spike mass: {format(analysis.spikeStats.positiveSpikes)}</div>
              <div>Negative spike mass: {format(analysis.spikeStats.negativeSpikes)}</div>
              <div>Absolute spike mass: {format(analysis.spikeStats.absoluteSpikeMass)}</div>
              <div>Sparse AC ops: {format(analysis.spikeStats.sparseAcOps)}</div>
              <div>Dense MAC baseline: {format(analysis.spikeStats.denseMacOps)}</div>
            </div>
          </section>

          <section style={{ ...panel, marginBottom: 16 }}>
            <h2 style={{ margin: '0 0 12px', fontSize: 16 }}>Weight Distribution</h2>
            <div style={{ display: 'grid', gap: 7 }}>
              {analysis.weightBuckets.map((bucket) => (
                <div key={bucket.min} style={{ display: 'grid', gridTemplateColumns: '88px 1fr 36px', gap: 10, alignItems: 'center', fontSize: 12 }}>
                  <span>{bucket.min.toFixed(1)}-{bucket.max.toFixed(1)}</span>
                  <div style={{ height: 10, background: '#101822', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ width: `${(bucket.count / maxBucket) * 100}%`, height: '100%', background: '#6ee7b7' }} />
                  </div>
                  <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{bucket.count}</span>
                </div>
              ))}
            </div>
          </section>

          <section style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 16 }}>
            <div style={panel}>
              <h2 style={{ margin: '0 0 12px', fontSize: 16 }}>Neurons</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={cellStyle}>ID</th>
                      <th style={cellStyle}>Label</th>
                      <th style={numberCell}>Membrane</th>
                      <th style={numberCell}>Refractory ms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.neurons.map((neuron) => (
                      <tr key={neuron.id}>
                        <td style={cellStyle}>{neuron.id}</td>
                        <td style={cellStyle}>{neuron.label}</td>
                        <td style={numberCell}>{format(neuron.v)}</td>
                        <td style={numberCell}>{format(neuron.refractoryLeftMs)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div style={panel}>
              <h2 style={{ margin: '0 0 12px', fontSize: 16 }}>Strongest Synapses</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={cellStyle}>ID</th>
                      <th style={cellStyle}>Pre {'->'} Post</th>
                      <th style={numberCell}>Weight</th>
                      <th style={numberCell}>aPre</th>
                      <th style={numberCell}>aPost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.strongestSynapses.map((synapse) => (
                      <tr key={synapse.id}>
                        <td style={cellStyle}>{synapse.id}</td>
                        <td style={cellStyle}>{synapse.pre} {'->'} {synapse.post}</td>
                        <td style={numberCell}>{format(synapse.w)}</td>
                        <td style={numberCell}>{format(synapse.aPre)}</td>
                        <td style={numberCell}>{format(synapse.aPost)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
