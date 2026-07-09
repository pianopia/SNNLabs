# Embodied Sensorimotor Runtime with Predictive Unsupervised Learning — Design

作成日: 2026-07-09
状態: 承認済み(brainstorming フェーズ完了、実装計画へ移行)

## 背景と目的

Elfentier / SNNLabs の最終目標は「LLMより高精度・能動的・自律的で、空間把握・
瞬発性・身体操作性を持つ、低消費電力・低リソースの AI モデル」。本スペックは、
**センサー/アクチュエータのモジュールを動的に増減でき、その閉ループのフィード
バックからラベルなしで教師なし学習する身体性ランタイム**を設計する。

カメラはモジュールの一例にすぎない。モーターアーム、触覚センサーなど任意の
デバイスをモジュールとして着脱でき、コアはそれらの入出力から自律的に学習する。

### 既存資産との関係

構想の骨格は既にリポジトリに断片的に存在する。本設計はこれらを統合・実体化する。
- `web_autonomous_learner.py`: `ObservationModule` プロトコル + `add_module()`、
  固定次元スパース `FeatureSpace`、`--endless`/`--resume` の継続学習ループ。
- biotope 研究ノート Phase 3「Micro Module Registry」: 疎結合モジュール +
  グローバル信号(arousal/reward/novelty/fatigue)。
- EDEN 身体SNN + `modules.ts`: trace イベント(`body`/`global_signal`/`spike`)
  による疎結合、恒常性可塑性、睡眠リプレイ風固定化、eligibility trace。
- PyTorch DST-SNN コア + `ChronoPlasticLIFLayer`(学習正典)。

## 設計上の決定(brainstorming で確定)

1. **接続方式**: 外部プロセス・メッセージプロトコル(WebSocket / JSONL)。
   各モジュールは独立プロセス/デバイスとして中央コアに接続し、実行中にホット
   プラグで増減する。言語非依存。
2. **学習則**: 予測符号化・世界モデル(自己教師)。予測誤差が学習信号かつ内発
   好奇心報酬。active inference 風。
3. **初回増分スコープ**: プロトコル + コア + 模擬モジュール。実ハードウェアは
   同プロトコルを再利用する後続増分。

## アーキテクチャ

中央の SNN コア(1プロセス)に任意個のモジュールプロセス(sensor / actuator /
both)がメッセージプロトコルで接続。コアは予測世界モデル SNN を回し、毎 tick
「潜在状態 + 選択行動」から次の感覚入力を予測する。予測誤差が教師なし学習信号
であり、同時に内発好奇心報酬になる。アクチュエータは復号行動を受け取り、その
効果がセンサー経由で戻って閉ループを閉じる。

```
modules --observation--> core --encode--> SNN --predict--> 予測誤差
   ^                                                          |
   |                                                       --learn--
   |                                                          |
sensors <--(物理/模擬効果)-- actuator <--action-- decode <----+
```
`global_signal`(arousal/reward/novelty/fatigue)と `trace`(spikes/membrane/
weights/prediction_error)は外部へブロードキャスト。

## コンポーネント

### B-1. メッセージプロトコル `src/dst_snn/sensorimotor/protocol.py`

言語非依存イベント型(JSONL over WebSocket)。全メッセージにタイムスタンプ。
- `register`: `id`, `role`(sensor|actuator|both), `modality`, `shape`,
  `action_space`(actuator の場合)
- `deregister`: `id`
- `observation`: module → core、感覚値
- `action`: core → actuator、復号行動
- `global_signal`: core → 全体、arousal/reward/novelty/fatigue のブロードキャスト
- `trace`: core → 観測者、可視化用

### B-2. モジュールレジストリ(コア側)

接続モジュールの動的テーブル。**モデルをリサイズせずに**増減対応するため、
`FeatureSpace`(固定次元スパース空間へハッシュ)方式を流用。新モダリティは既存
固定入力空間へハッシュされ、アクチュエータの行動空間も固定モーターニューロン
プールへ写像。モジュール切断は deregister して継続。

### B-3. エンコーダ / デコーダ

センサー観測 → スパイク列(FeatureSpace + ストリーム用の時間ビン化を流用)。
モーターニューロン発火 → 行動メッセージ。モジュール仕様ごとに解決し、未知
モダリティは既定の数値/スパイクエンコーダ。

### B-4. 予測世界モデル SNN コア `src/dst_snn/sensorimotor/world_model.py`

- 潜在状態は `ChronoPlasticLIFLayer`(時間文脈 + 適応リーク)。
- (潜在, 行動)から次ステップ感覚スパイクを予測。
- 学習はサロゲート勾配による自己教師的予測損失。ラベル不要。
- **自律性のための必須要素:**
  - **行動方策を内発報酬で駆動**: 好奇心(予測誤差の期待低減 = learning
    progress)でモーターニューロンの探索を駆動する。行動はランダムではなく
    内発目標に従う。
  - **自己調整(恒常性)**: 発火飽和・表現崩壊を防ぐため、恒常性可塑性(発火
    しすぎるニューロンの閾値上げ)、疲労、睡眠リプレイ風固定化を組み込む。
  - **継続学習**: チェックポイント保存/再開(`web_autonomous_learner` の
    `--endless`/`--resume` と同方針)で、停止しても学び続けられる。
- 予測誤差 → 内発好奇心/新奇性 → 神経修飾グローバル信号として可塑性をゲート。

### B-5. ランタイムループ `src/dst_snn/sensorimotor/runtime.py`

非同期。1 tick = 全センサーの最新観測収集 → エンコード → SNN 前進 → 予測 →
予測誤差算出 → 教師なし更新 → 行動復号 → アクチュエータ送信 → global_signal /
trace ブロードキャスト。ループ途中のホットプラグ対応。遅延観測はモジュール
ごとに最新値を使用。

### B-6. 模擬モジュール(初回増分)`src/dst_snn/sensorimotor/modules/`

- `synthetic_sensor`: 決定論的信号生成、テスト用。
- `webcam_sensor`: RGB を cv2 で取得、フレーム差分 → イベント化。前回のカメラ
  要望はこの1モジュールに包含。
- `mock_actuator`: 行動をログし、合成センサーへ効果を戻してテスト可能な閉ループ
  を形成。
- EDEN ブリッジは将来コネクタとして文書化(TS側は既に trace/observation 相当を出力)。

### B-7. 教師なし学習の評価

既存ハーネス `MetricSet.extra` に接続。指標: 予測誤差の時間推移(減少すべき)、
内発報酬、表現安定性、正解が既知の合成閉ループでの線形プローブ/クラスタ純度。
精度主張ではなく学習進行を測る。

### B-8. 性能予算 provision(M3 MacBook 実行性)

- 設定可能パラメータ: `tick_hz`(既定 5〜20Hz)、`time_steps`(小)、`latent_size`
  (小)、`device`(既定 `cpu`、`mps` 任意)。
- **既定 CPU、MPS 任意**: PyTorch の Apple Silicon MPS バックエンドを許容。ただし
  `ChronoPlasticLIFLayer` は時間ステップを Python ループで回すため、モデル小・
  time_steps 小・低 tick レートを既定にしてリアルタイム性を確保。MPS は一部 op で
  CPU フォールバックし得る(unfold/gather/カスタム autograd 周り)。`SurrogateSpike`
  は要素演算のため全デバイスで動作。
- **合成ヘッドレスモード**: webカメラ無し・GPU 非依存で CI 実行可能。
- M3 CPU での実測 step 時間を `MetricSet.extra` に記録(前フェーズの latency/
  energy 指標と一貫)。
- 固定次元スパース状態で継続学習中もメモリ有界。CUDA 不要。

## エラー処理

- ループ途中のモジュール切断: deregister してレジストリから除外、継続。
- 不正メッセージ: ログ + スキップ(web学習器の per-module try/except と同方針)。
- 遅延観測: モジュールごとに最新値を使用。全メッセージにタイムスタンプ。

## テスト方針

- プロトコルのシリアライズ往復。
- レジストリ add/remove。
- エンコーダ・デコーダの形状テスト。
- world-model の前進・予測・学習(繰り返し合成パターンで損失減少)。
- **閉ループスモークテスト**: 行動が決定論的にセンサー信号をずらす合成系で、
  予測誤差が tick とともに減少 = 教師なし学習が働くことを検証。
- **自律性検証**: 無人で N tick 連続実行し、予測誤差が人手介入なしで低下し続け、
  恒常性で発火率が発散/崩壊しないことを確認。
- すべてオフライン・ハードウェア不要。テストは in-process transport または
  localhost ループバック。

## スコープと分解

一貫したサブプロジェクト = 1スペック → 1計画。初回増分はプロトコル + コア +
模擬モジュール。実ハードウェア(モーターアーム/触覚)は同プロトコルを再利用する
後続増分(別スペック)。

- PyTorch DST-SNN コア + `ChronoPlasticLIFLayer` を再利用。
- 評価は既存ハーネス(`2026-07-09-snn-benchmark-harness-design.md` の `RunResult`/
  `MetricSet`)を再利用。
- biotope Phase 3(モジュールレジストリ)+ Phase 1(目標/好奇心)を統合ランタイム
  として実体化。

## 環境上の注意

- 本環境に PyTorch/OpenCV は未インストール。実装は correct-by-construction を旨と
  し、torch/cv2 依存部は実データ検証を別途要する旨をコード/README に明記する。

## 未解決事項

- 予測誤差から行動方策を駆動する具体アルゴリズム(好奇心の定式化: prediction-error
  vs learning-progress)は実装計画時に1つに確定する。既定は予測誤差の指数移動
  平均の減少量(learning progress)を内発報酬とする。
- 実ハードウェアモジュールのトランスポート詳細(シリアル/USB ブリッジ)は後続
  スペックで詰める。
