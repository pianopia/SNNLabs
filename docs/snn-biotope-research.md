# SNN Biotope Research Notes

作成日: 2026-06-19

## 目的

Elfentier のLPデザインは維持したまま、今後のサービスを「SNNの働きを可視化してトレースできる実験基盤」および「SNNモデルを利用できるサービス」に拡張するための実装知識を整理する。

この段階では「超知性」を直接実装対象にせず、検証可能な最小単位であるマイクロSNNモジュールを作る。各モジュールは発火、膜電位、シナプス重み、STDPトレースをイベントとして出力し、後続のUIや3D環境がそれを購読できるようにする。

## 調査メモ

### 既存SNN基盤から得る設計判断

- Brian 2 は、スパイクニューロン、シナプス、任意の微分方程式、実験プロトコルを高水準に書き、低水準コードへ生成する設計を採る。サービス実装では、Nimコアも「モデル定義」と「実行器」を分けると拡張しやすい。
- Brian 2 のSTDP例では、pre/postイベントごとに `Apre` と `Apost` を指数減衰させ、pre発火時に `w += Apost`、post発火時に `w += Apre` で重みをクリップする。今回の試作はこのイベント駆動形式を採用する。
- Nengo SPA は、概念を高次元ベクトルの Semantic Pointer として扱い、superposition、binding、unbinding で構造表現を作る。SNNビオトープでは「概念細胞」や「LLM知識ブロックのルーティングキー」を、最終的にSemantic Pointer的な表現へ寄せる。
- Norse と snnTorch は PyTorch上でSNNを扱う実用系ライブラリで、勾配学習や既存MLパイプラインとの接続に強い。Elfentier側はリアルタイム可視化と軽量実行を優先し、学習済みSNNやANN-to-SNN変換の取り込みは後続フェーズで外部ツール連携として扱う。
- Izhikevich の比較論文は、ニューロンモデル選択が大規模シミュレーションの妥当性を左右すると述べる。初期実装はLIFで軽く始め、バースト、適応閾値、樹状突起相当のマルチコンパートメントは別モデルとして追加する。

### 参照元

- Brian 2 STDP example: https://brian2.readthedocs.io/en/latest/examples/synapses.STDP.html
- Brian 2 paper: https://elifesciences.org/articles/47314
- Nengo SPA introduction: https://www.nengo.ai/nengo-spa/v1.1.0/user-guide/spa-intro.html
- Nengo SPA repository: https://github.com/nengo/nengo-spa
- Norse repository: https://github.com/electronicvisions/norse
- snnTorch documentation: https://snntorch.readthedocs.io/
- Izhikevich, "Which Model to Use for Cortical Spiking Neurons?": https://www.izhikevich.org/publications/whichmod.pdf

## サービス設計への落とし込み

### 1. SNN Core

責務:

- LIFなどのニューロンモデルをステップ実行する。
- 外部入力、内部シナプス入力、発火、リセット、不応期を扱う。
- STDPでシナプス重みをオンライン更新する。
- 全状態変化をトレースイベントとして出力する。

最小イベント:

- `membrane`: ニューロンごとの膜電位。
- `spike`: 発火したニューロン。
- `weight`: STDPで更新されたシナプス重み。
- `input`: 外部またはシナプスから流入した電流。

### 2. Trace API

UIが直接SNN内部表現に依存しないよう、JSONLまたはWebSocketでイベントを流す。

```json
{"t":12.0,"kind":"spike","neuron":3,"label":"concept:edge","v":1.0}
{"t":12.0,"kind":"weight","synapse":7,"pre":1,"post":3,"w":0.42}
```

可視化側はこのイベントを時系列で再生すればよい。初期UIでは膜電位グラフ、スパイクラスタ、重みヒートマップ、ネットワークグラフを優先する。

### 3. Biotope Runtime

複数のマイクロSNNを疎結合にし、モジュール間はスパイクイベントと低頻度のグローバル信号で連携する。

- モジュール: reflex、memory、concept、motor、world-model など。
- グローバル信号: arousal、reward、novelty、fatigue など。
- 通信: 初期はWebSocket、負荷が増えたらNATS、Redis Streams、または専用バイナリプロトコルを検討する。

### 4. LLMとの接続

Frozen LLM Core は最初から重み更新対象にしない。SNN側が出す発火パターンを「知識ブロック検索のルーティング信号」として使う。

初期実装案:

1. SNNがセンサ入力を受け、概念細胞を発火させる。
2. 発火ラベルを検索キーへ変換する。
3. RAGまたはLLM tool callで知識ブロックを取得する。
4. 結果を再びSNNへ報酬、抑制、または概念入力として戻す。

## ロードマップ

### Phase 1: Nim SNN Core

- LIFニューロン。
- イベント駆動STDP。
- JSONLトレース。
- 小さなアトラクタ風ネットワークのデモ。

### Phase 2: Visual Trace UI

- Three.js またはCanvasでニューロンとシナプスを表示。
- WebSocketでNimプロセスからイベント購読。
- スパイク、膜電位、重み変化を時間軸で再生。

### Phase 3: Micro Module Registry

- SNNモジュール定義ファイルを読み込む。
- センサ、概念、運動などのモジュールを疎結合にする。
- グローバル信号を全モジュールに配信する。

### Phase 4: LLM/SNN Hybrid

- SNNの発火パターンをLLM/RAGルーティングへ接続。
- SPA風の高次元概念表現を導入。
- World Modelは別プロセスで夢・予測イベントを生成する。

## 現時点の実装方針

今回のNim試作は、依存を追加せず標準ライブラリだけで実装する。LPのページやレイアウトは変更しない。

追加するもの:

- `src/snn/core.nim`: LIF、STDP、トレースイベントの最小コア。
- `src/snn/demo.nim`: 2入力、2概念ニューロン、1アトラクタ風ニューロンのデモ。

実行例:

```sh
nim c -r src/snn/demo.nim
```

出力はJSONLなので、後続のWebSocketサーバや可視化UIにそのまま流用できる。
