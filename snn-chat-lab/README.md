# SNN Neuron Inspector

学習済み `.edensnn` / DST-SNN `.pt` モデルのニューロン出力を確認するための独立ツール。

Chrome拡張や `browser-snn-lab/` からエクスポートした `browser-language` モデルを読み込み、語彙ニューロン、結合重み、発火状態を確認する。
`scripts/train_dst_snn_web.py` が保存したPyTorch checkpointは、チャット用の特徴語・相互関係モデルへ変換して読み込む。
チャット欄は自然文生成ではなく、読み込んだSNNモデルのニューロン出力だけを表示する検査面として扱う。

## 使い方

1. `index.html` をブラウザで開く。
2. `Import Model` から学習済みモデルを読み込む。
3. 右下の入力欄から質問する。
4. 左側で活性化したニューロンと強い結合を確認する。
5. `Spike Threshold` で、本文に出すニューロン値の閾値を調整する。
6. `Neuron Render` から、直近返答の発火状態を `Audio WAV` または `Video WebM` として出力できる。

`.edensnn` と `.chat.json` は静的ファイルだけで動くため、サーバーは必須ではない。

DST-SNNの `.pt` を直接アップロードする場合は、PyTorch checkpointをブラウザで安全に復元できないため、ローカル変換サーバから起動する。

```bash
python scripts/serve_snn_chat_lab.py
```

その後、[http://127.0.0.1:8765](http://127.0.0.1:8765) を開いて `.pt` を読み込む。

事前にチャット用JSONへ変換する場合:

```bash
python scripts/export_dst_chat_model.py artifacts/dst-web-learner.pt
```

生成される `artifacts/dst-web-learner.chat.json` は `index.html` へ直接読み込める。

## チャット出力の仕組み

外部LLMには接続しない。読み込んだSNNモデル内の情報だけを使う。

1. 質問文を日本語/英語の単語へ分割する。
2. 接続詞、助詞、冠詞などの構造語は低い重みで扱い、頻出だけで支配的にならないようにする。
3. 一致する語彙ニューロンを初期発火させる。
4. チャットに打った意味語は、既存の発火ニューロンとのシナプスを強める。
5. 正のシナプス結合を2段まで伝播し、関連ニューロンを活性化する。
6. 活性化したニューロンをスコア順に並べる。
7. `Spike Threshold` 未満のニューロンをマスクする。
8. マスクされていない `Neuron` カラム値だけを区切りなしで連結し、`SNN_NEURON_TEXT` として表示する。
9. 表にはマスク済みニューロンも残し、発火ベクトルとシナプス経路を検査できるようにする。
10. 同じ内容を `snn_neuron_output` のRaw vectorとして表示する。

観測履歴の全文、再整理済み記憶文、テンプレート文、外部AIによる自然文は返答に使わない。
本文に出る全文は、入力に対してSNNが発火させたニューロンの `token` 値のみで構成される。
Raw vectorには `spikeEvents` と `moduleTriggers` も含めるため、後付けモジュールやスキルの発火判定に流用できる。
チャットで強化されたモデルは `Export Updated` から `.edensnn` として保存できる。

## ニューロンによる音声/動画返答

音声や動画ストリームから学んだ記憶がある場合も、外部生成AIには渡さず、反応したSNNニューロンを直接レンダリングする。

- `Audio WAV`: 活性化ニューロンを周波数、スコアを振幅、結合量を音の厚みに変換する。
- `Video WebM`: 活性化ニューロンを円形グラフに配置し、シナプス結合と発火パルスをフレームとして描く。

これは元の動画や音声を復元するものではない。SNNモデル内に形成されたニューロン反応を、音声/映像ファイルとして描き出す可視化返答である。

## 対応モデル

- `.edensnn`
- `.pt` (`scripts/serve_snn_chat_lab.py` から起動した場合)
- `.chat.json` (`scripts/export_dst_chat_model.py` またはWeb learnerの保存時に生成)
- `modelKind: "browser-language"`
- `snapshot.domain: "browser-language"`
- `snapshot.domain: "dst-web"`

DST-SNN由来モデルでは、画像、音声、動画、テキスト、ブラウザ身体操作の特徴語とクロスモーダル関係もニューロン出力として扱う。

## ファイル

- `index.html`: チャットUI
- `styles.css`: UIスタイル
- `app.js`: `.edensnn` / `.pt` / `.chat.json` デコード、活性化伝播、ニューロン出力表示
