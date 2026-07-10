# Elfentier

## DST-SNN Prototype

`src/dst_snn/` contains a PyTorch prototype for Dendritic Spatio-Temporal SNN.
It factorizes spatial synapses, dendritic delay buffers, and instantaneous soma
spiking.

Install:

```bash
pip install -r requirements-dst-snn.txt
```

Run the minimal delayed-pattern experiment:

```bash
python scripts/train_dst_snn_pattern.py
```

Run autonomous web observations through Playwright. By default this keeps
learning until you stop it with Ctrl-C:

```bash
playwright install chromium
python scripts/train_dst_snn_web.py https://example.com --steps-per-url 8
```

Run one finite pass:

```bash
python scripts/train_dst_snn_web.py https://example.com --steps-per-url 8 --once
```

Optional actions:

```bash
python scripts/train_dst_snn_web.py https://example.com \
  --allow-clicks --allow-inputs --allow-navigation
```

When autonomous actions are enabled, the trainer emits an `event: "action"`
JSON line after each action. Link navigation, button clicks, input, scrolls,
URL changes, title changes, text deltas, and visible error text are fed back
into the next training observation as body/action features.

Run continuously like the Chrome extension until you stop it:

```bash
python scripts/train_dst_snn_web.py https://note.com \
  --allow-clicks --allow-inputs --allow-navigation \
  --resume
```

For a bounded long run:

```bash
python scripts/train_dst_snn_web.py https://note.com \
  --allow-clicks --allow-inputs --allow-navigation \
  --endless --max-total-steps 1000 --save-every 20 --max-navigations 48
```

The web learner observes text, visual screenshots/images, audio/video DOM state,
and browser-body actions as separate modules. It trains the DST-SNN online and
writes checkpoints and cross-modal relation memory under `artifacts/`:

Text learning uses visible semantic page text first (`main`, `article`,
paragraphs, lists, headings) and removes code blocks, SVG/canvas/script/style
content, URLs, HTML tag names, URL fragments, hashes, and short numeric noise
before creating text spikes. Media modules only learn human-facing labels, not
raw `src` URLs.

- `dst-web-learner.pt`: PyTorch checkpoint for continued training.
- `dst-web-learner.relations.json`: cross-modal relation memory.
- `dst-web-learner.chat.json`: browser-readable model for `snn-chat-lab`.

Open the learned `.pt` directly in `snn-chat-lab` through the local converter:

```bash
python scripts/serve_snn_chat_lab.py
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) and import the `.pt`.
For static use, convert an existing checkpoint first:

```bash
python scripts/export_dst_chat_model.py artifacts/dst-web-learner.pt
```

## Benchmarks

SNN evaluation harness and benchmark runners live under `benchmarks/`.
See [benchmarks/README.md](benchmarks/README.md) for full usage.

```bash
pip install -r requirements-bench.txt
# Optional 3DCG scorer deps
pip install -r requirements-3dcg.txt

# Neuromorphic (N-MNIST / DVS Gesture)
python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 3
python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs-gesture --epochs 5

# With dense energy + ANN quality baseline on a smoke subset
python benchmarks/neuromorphic/run_nmnist.py \
  --root data/nmnist --epochs 1 --limit-train 128 --limit-test 64 \
  --smoke-from-test --with-ann-baseline

# 3DCG scorer (offline unit corpus)
python scripts/build_threedcg_unit_corpus.py
python benchmarks/threedcg/run_score.py \
  --reference data/threedcg/unit-box/reference.glb --convex-hull --asset-id unit-box

# Synthetic sensorimotor loop
python benchmarks/sensorimotor/run_synthetic_loop.py --steps 32
```
