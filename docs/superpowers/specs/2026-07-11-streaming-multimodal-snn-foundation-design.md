# Streaming Multimodal SNN Foundation Design

**Status (2026-07-11):** Research-backed redesign; Phases 1–2 implemented.

## Objective

Build toward one local system that can understand and generate text, image,
audio, video, 3D and actions while using materially less memory and measured
power than dense multimodal baselines, with streaming responses instead of a
long request/response stall.

This is a research program, not a current capability claim. No published pure
SNN yet demonstrates the complete capability and quality envelope of frontier
multimodal LLMs. Success therefore requires staged parity, not replacing every
dense component at once.

## Research findings and missing ingredients

1. **Scaling needs conversion/distillation.** SpikeLLM scales spiking activation
   methods to 7–70B parameter LLMs, but does so in a quantization/conversion
   pipeline and explicitly identifies long BPTT and long spike trains as scaling
   barriers. A random-initialized pure LIF language model is not the practical
   starting point. Source: [SpikeLLM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/510e7d39fce008a3e31de54b8f5be9ac-Paper-Conference.pdf).
2. **Long context needs linear/recurrent state.** SpikingSSMs reports competitive
   long-sequence performance at about 90% sparsity and better WikiText-103
   results than earlier spiking LMs with one-third the model size. SpikingBrain
   combines linear/hybrid-linear attention, adaptive spiking neurons, conversion
   training and system-specific kernels; it reports partially constant inference
   memory and 69.15% sparsity. Sources: [SpikingSSMs](https://arxiv.org/abs/2408.14909),
   [SpikingBrain v4](https://arxiv.org/abs/2509.05276).
3. **One spike clock is wrong for multimodality.** SpikeMLLM identifies uniform
   encoding and visual timestep expansion as core problems. Its modality-specific
   temporal scales and compressed LIF keep the reported gap to FP16 baselines near
   1% in tested models, while its dedicated accelerator—not ordinary GPU sparsity
   alone—provides the large power/throughput gain. Source:
   [SpikeMLLM](https://arxiv.org/abs/2604.18610).
4. **Binary spikes discard important information.** Signed/multi-level neurons
   are needed for semantic polarity and outlier channels. Pure binary rate coding
   raises latency because precision is represented by extra timesteps.
5. **Feature-level teacher alignment matters.** ANN-guided block-wise replacement
   improves SNN training by aligning intermediate rate features and avoids the
   linear time/memory growth of conventional BPTT at larger timestep counts.
   Source: [CVPR 2025 ANN-guided distillation](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Efficient_ANN-Guided_Distillation_Aligning_Rate-based_Features_of_Spiking_Neural_Networks_CVPR_2025_paper.pdf).
6. **Operation-count estimates are not power measurements.** Sparse tensors on a
   GPU may still execute dense kernels. Energy claims require wall power, latency,
   memory traffic and accuracy on the same hardware, plus a neuromorphic or custom
   event-driven backend for the intended advantage.

## Redesigned architecture

```text
modality tokenizer/teacher encoder
  -> signed multi-level delta events (per-modality clock)
  -> temporal compression
  -> sparse modality router / shared latent event space
  -> streaming SpikingSSM core (bounded recurrent state)
  -> specialist sparse experts + external episodic memory
  -> text/image/audio/video/3D/action decoders
  -> confidence/stability early exit and continuous output
```

Training proceeds by progressive block replacement:

1. Freeze a capable open multimodal teacher and cache tokenizer/intermediate
   targets. Teacher use is a training cost and must not be hidden in efficiency
   claims.
2. Distil modality encoders and shared latent features, preserving ANN escape
   paths until each replaced block meets parity.
3. Train signed-integer spiking SSM blocks with task loss, feature alignment,
   spike-rate budget, latency budget and state-stability losses.
4. Distil output heads separately. Keep dense generation heads where quality
   loss exceeds the gate, then replace them incrementally.
5. Add continual local plasticity only to bounded adapters/memory; protect the
   foundation weights from catastrophic forgetting.

## Capability and efficiency gates

Every stage must report paired teacher/student results on identical examples.

- Quality: first parity gate is no more than 1% relative degradation per task;
  the final “exceeds” claim requires statistically significant wins across the
  agreed multimodal suite, not one benchmark.
- Latency: sensor/action p95 below 20 ms; first semantic response below 100 ms
  on the declared target device; generation streams immediately thereafter.
- Memory: bounded recurrent context state; peak RSS/VRAM and model bytes at most
  25% of the matched dense local baseline for the deployment tier.
- Energy: at least 10x lower measured joules per successful task at matched
  quality. AC/MAC proxy remains diagnostic only.
- Sparsity: at least 80% zero events after fusion without violating quality.
- Robustness: calibration, OOD, adversarial/noise and continual-learning
  regression gates are mandatory.

## Phase 1 implemented in this increment

- Modality-specific signed sigma-delta event encoder.
- Logarithmic temporal compression and aligned multimodal event fusion.
- Constant-state streaming spiking SSM reference runtime.
- Confidence-and-stability early exit.
- Explicit estimated AC/MAC, sparsity and state-memory report that refuses to
  present proxy energy as measured hardware power.

The reference runtime defines backend semantics. It is not yet a trained
foundation model.

## Phase 2 implemented

- Trainable diagonal signed-integer Spiking SSM with surrogate gradients,
  learnable decay, threshold and input gate.
- Frozen Torch teacher adapter with named intermediate-feature hooks and a
  portable cached-feature dataset.
- Progressive prefix block replacement; gradients pass through the frozen ANN
  suffix into replaced SNN blocks.
- Joint task, logit distillation, cosine feature alignment, spike-budget and
  early-exit losses.
- Synthetic text next-token and image-text retrieval smoke benchmarks. Both
  reach teacher parity on their tiny deterministic tasks, while event rates
  remain above the eventual sparsity target.

This validates the Phase 2 training path, not multimodal LLM capability.
