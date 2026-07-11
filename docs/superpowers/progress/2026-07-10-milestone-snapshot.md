# Milestone Snapshot — 2026-07-10

凍結時点の作り・性能・解釈を記録する。以降の比較・回帰の基準点。

## 1. Phase 0 位置づけ

| 設計項目 (Phase 0) | 状態 |
|---|---|
| 共通評価ハーネス `src/dst_snn/eval/` | 完了 |
| N-MNIST / DVS128 Gesture ランナー | 完了（空間 SNN 含む） |
| 3DCG スコアラ + 凸包ベースライン | 完了（合成 unit-box） |
| Web 学習器 novelty / reward 修正 | 完了 |
| Sensorimotor ランタイム（初回増分） | 完了〜拡張済み |
| 対 LLM ベースライン | 完了（optional eval interface: scripted/HTTP; 実 API 数値は未凍結） |
| 3DCG 生成器本体 | 初回増分済み（Track1 スクリプト + Track2 occupancy scaffold） |

対応 plan:
- `docs/superpowers/plans/2026-07-09-snn-eval-harness-neuromorphic.md` — 実装済み
- `docs/superpowers/plans/2026-07-09-snn-3dcg-scorer.md` — 実装済み

## 2. 現在のアーキテクチャ（コードマップ）

### 評価ハーネス
- `src/dst_snn/eval/{energy,metrics,result,runner}.py`
- ベースライン: `baselines/ann_classifier.py` (flat MLP), `baselines/frame_cnn.py` (空間 CNN), `baselines/llm_*.py` (optional LLM)

### ニューロモルフィック
- イベント: `benchmarks/neuromorphic/events.py`, `datasets.py` (`flat` / `frames`)
- バックボーン:
  - `dendritic` — 平坦化 + `SnnClassifier` (DST-SNN / Chrono / hidden)
  - `conv-plif` — 空間 Conv-BN-PLIF（`conv_snn.ConvPLIFClassifier`）
  - `sew-plif` — SEW 残差付き深い Conv-PLIF（`SewConvPLIFClassifier`）
- ランナー: `run_nmnist.py`, `run_dvs_gesture.py`, `run_multi_seed.py`
- パイロット: `scripts/run_dvs_fulltrain_sew_pilot.py`

### 3DCG
- `benchmarks/threedcg/*` — asset / geometry / topology / UV / rig / skin / texture / scorer / baseline / render_similarity
- glTF skin_weights 抽出、hierarchy_edit_distance、weight_smoothness 済み

### Sensorimotor
- `src/dst_snn/sensorimotor/` — protocol, registry, codec, runtime, world_model, homeostasis (閾値配線), sleep_replay, policy, websocket_transport, webcam_sensor
- ベンチ: `benchmarks/sensorimotor/run_synthetic_loop.py`

## 3. 性能記録（凍結値）

### 3.1 DVS Gesture — 公式 full-train parity-ds8（2026-07-10）

出典: `artifacts/benchmarks/dvs-fulltrain-sew/report.md`  
条件: train 1077 / test 264, epochs 12, seeds 0–2, time_bins 16, downsample 8, threshold 1.0, readout spike_count, CPU

| backbone | SNN mean±std | SNN range | Frame-CNN mean | majority | seeds > maj | params |
|---|---:|---:|---:|---:|---:|---:|
| conv-plif | **0.447±0.020** | 0.424–0.474 | 0.434 | 0.091 | 3/3 | 57k |
| sew-plif | **0.490±0.020** | 0.466–0.515 | 0.489 | 0.091 | 3/3 | 243k |

解釈:
- dense dendritic 平坦化経路（majority 張り付き）は棄却済み。
- 空間 SNN は **安定学習**し、**同幅 Frame-CNN（ReLU ANN）と精度は並ぶ**。
- **精度で CNN を明確に超えたとは言えない**（SEW は +0.001、seed 依存で CNN が勝つ回あり）。
- エネルギー proxy は SNN が桁違いに小さいが、AC スパイク proxy vs full MAC で定義が非対称。
- 文献 SEW-ResNet ~97% には遠い（浅いネット・短い学習・粗い空間解像度）。

### 3.1b DVS Gesture — hires-ds4 full-train freeze（2026-07-10）

出典: `artifacts/benchmarks/dvs-hires-fulltrain/report.md`  
条件: full train, recipe `hires-ds4` (downsample=4 ≈32×32, cosine LR, epochs 12), seeds 0–2, conv-plif + Frame-CNN

| backbone | SNN mean±std | SNN range | Frame-CNN mean | majority | seeds > maj | params |
|---|---:|---:|---:|---:|---:|---:|
| conv-plif | **0.537±0.039** | 0.485–0.580 | 0.342 | 0.091 | 3/3 | 57k |

解釈:
- hires は parity-ds8 の conv-plif **0.447 → 0.537** と明確に改善。
- 本条件では SNN mean **>** CNN mean（CNN は seed1 で崩壊気味 0.186；SNN は全 seed で安定）。
- 依然 SOTA (~97%) ではない。主張は「制御された高解像度レシピで学習が伸びた」まで。

### 3.2 DVS Gesture — 経路別（smoke 時、参考）

| 構成 | mean | above maj | メモ |
|---|---:|---:|---|
| dendritic direct | ~0.09 | 1/5 | 崩壊 |
| dendritic + hidden | ~0.10 | 1/5 | seed 依存 |
| conv-plif smoke 168/96 | 0.333 | 5/5 | 初の安定学習 |
| sew-plif smoke 200/64 | 0.432 | 3/3 | residual 有効 |

### 3.3 N-MNIST（smoke 記録、フル train 未主張）

- stratified smoke 1024/512, max_membrane: accuracy ~0.50 vs majority ~0.11（過去ログ）。
- 本マイルストーンでは N-MNIST フル train は未凍結。

### 3.4 3DCG / Sensorimotor

- 3DCG: 合成 unit-box で scorer E2E 可。実 SketchFab コーパス未整備。
- Sensorimotor: 合成**真の**閉ループ（actuator → sensor phase）、予測損失低下、homeostasis 閾値配線、sleep-replay、線形プローブ/クラスタ純度、ANN 予測ベースライン（`--with-ann-baseline`）、AC vs dense MAC energy を記録。実 HW 未接続。

## 4. 既知の制限

1. エネルギー比較の非対称（SNN AC proxy vs CNN full MAC）
2. DVS downsample=8 で空間解像度が低い（16×16 相当）
3. 学習は Adam 固定 lr、長い schedule / augment なし
4. LLM ベースライン: scripted multi-seed 凍結済み; HTTP は小サンプル（token proxy は AC/MAC 非比較）
5. 3DCG は合成 multi-asset カタログ（licensed SketchFab GLB はドロップイン待ち）
6. figshare DVS 直ダウンロードは WAF で失敗 → Zenodo MD5 一致アーカイブを使用
7. 実 HW デバイスは未接続（serial bridge は MockSerialPort で検証済み）

## 5. 再現コマンド（主要）

```bash
# フル train SEW / Conv + CNN ベースライン
python scripts/run_dvs_fulltrain_sew_pilot.py

# 単発
python benchmarks/neuromorphic/run_dvs_gesture.py \
  --backbone sew-plif --with-ann-baseline \
  --epochs 12 --time-bins 16 --downsample 8 --seed 0

# テスト
. .venv/bin/activate && python -m pytest -q
```

## 6. 次フェーズへの引き継ぎ

- Phase 0 計測基盤は **運用可能な水準で凍結**。
- 精度主張は「CNN 並みの浅い空間 SNN がフル train で安定」まで。SOTA 未達。
- Closeout plan: `docs/superpowers/plans/2026-07-10-phase0-closeout-dvs-training.md` (implemented)
- LLM baseline plan: `docs/superpowers/plans/2026-07-10-llm-baseline-interface.md` (implemented)
- Remainder closeout: hires freeze script, HW serial bridges, synthetic multi-asset corpus,
  3DCG generators, EDEN bridge, LLM multi-seed report (see progress log).
