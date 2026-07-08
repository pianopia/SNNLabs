# Browser SNN Language Lab

EDENとは独立した、自律ブラウザ操作・言語SNN学習環境です。

## 目的

- SNNエージェントが内部ブラウザ環境でマウス移動、クリック、スクロール、読解を自律実行する。
- 操作によって可視領域に現れた言語をSNNへ入力する。
- 報酬 `-1..1` で単語ニューロンと隣接関連シナプスを逐次更新する。
- 学習済みモデルを `.edensnn` としてエクスポートする。

## 起動

静的HTMLなので、以下のどちらでも動きます。

```sh
open browser-snn-lab/index.html
```

または任意の静的サーバーで配信します。

```sh
python3 -m http.server 5174 --directory browser-snn-lab
```

URL:

```text
http://127.0.0.1:5174/
```

## 制約

初期版は安全に制御できる内部Webページ群を操作対象にしています。実サイトの完全自律操作はCORS、iframe、認証、利用規約、セキュリティ境界の設計が必要なため、別段階でブラウザ拡張またはPlaywrightベースの実行環境として分離します。

## モデル

エクスポート形式はEDENと互換の `.edensnn` です。

- container: `EDENSNN1`
- modelKind: `browser-language`
- snapshot.domain: `browser-language`
- neuron type: `si-lif`
