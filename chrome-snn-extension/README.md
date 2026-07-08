# Elfentier Browser SNN Learner

Chrome拡張として動く、EDEN非依存のブラウザ内SNN学習環境。

通常のWeb閲覧中にページ内の可視テキスト、クリック、スクロール、入力、選択範囲を観測し、ブラウザ言語SNNモデルを逐次学習する。学習済みモデルは `.edensnn` としてエクスポートできる。

## 目的

- EDENのThree.js環境とは別に、ブラウザそのものを身体化された環境として扱う。
- ユーザーのマウス操作、スクロール、入力、選択を感覚刺激として記録する。
- 再生中の動画・音声ストリーミングを時間変化する感覚刺激として記録する。
- ページの可視言語を語彙ニューロンとして成長させる。
- 操作と文脈の共起からトークン間結合を報酬変調STDP風に更新する。
- 新奇性、注意、情動価、疲労、睡眠リプレイ風の固定化、恒常性閾値調整を使い、単純な共起より脳型の逐次学習へ寄せる。
- EDENのSNNダッシュボードで読める `.edensnn` モデルファイルを生成する。

## インストール

1. Chromeで `chrome://extensions` を開く。
2. `Developer mode` を有効にする。
3. `Load unpacked` を押す。
4. このリポジトリの `chrome-snn-extension/` ディレクトリを選ぶ。
5. ツールバーの拡張アイコンを押すとサイドパネルが開く。

## 使い方

1. 通常通りWebページを閲覧する。
2. ページ上でクリック、スクロール、入力、テキスト選択を行う。
3. サイドパネルの `Model` プルダウンで、このタブが学習するSNNモデルを選ぶ。別タブで同じモデルを選べば、同じモデルへ追記学習できる。
4. サイドパネルの `START Learning` / `STOP Learning` で、現在のタブだけ学習を切り替える（初期状態は STOP）。
5. `START Explore` / `STOP Explore` で、現在のタブだけ自律的なブラウザ探索を切り替える。
6. `Learning Tabs` で、学習中または探索中のタブと割り当てモデルを見る。
7. `Sleep Consolidate` を押すと、現在選択中のモデルを睡眠リプレイ風に整理できる。
8. サイドパネルの `Learning Settings` で入力値やモラル評価の扱いを調整する。
9. サイドパネルで語彙ニューロン、結合、観測履歴、操作トレースを見る。
10. `Import Model` で学習済みの `.edensnn`、JSON、`.chat.json`、`.pt` を読み込み、現在のモデルへ追加学習用にマージするか、新しいモデルとして追加する。
11. `Export .edensnn` で選択中モデルを保存する。
12. EDENの `/snn-dashboard` に読み込むとニューロン分析に使える。

## 観測する刺激

- `page_view`: ページ表示時の可視テキスト。
- `click`: クリック位置、要素ラベル、可視テキスト。
- `scroll`: スクロール位置、現在の可視テキスト。
- `input`: 入力対象のラベルと値の一部。
- `selection`: ユーザーが選択したテキスト。
- `media_detected`: 再生済みまたは表示中の動画・音声要素。
- `media_play`: 動画・音声の再生開始。
- `media_pause`: 一時停止。
- `media_sample`: 再生中の定期サンプル。
- `media_seek`: シーク操作。
- `media_rate`: 再生速度変更。
- `media_volume`: 音量またはミュート変更。
- `media_ended`: 再生終了。
- `visual_scan`: 表示中の画像・動画フレームを視覚特徴として観測。
- `text_visual`: ページ上のテキスト要素の表示矩形をタブキャプチャから切り出し、その見た目と本文語彙を同一ステップで結合。
- `text_visual_click`: クリックしたテキスト要素の表示矩形と語彙を結合。
- `text_visual_selection`: ユーザーが選択したテキスト範囲の見た目と語彙を結合。
- `text_visual_caption`: 動画の字幕 DOM または字幕帯の見た目と cue 語彙を結合。
- `autonomous_visual`: 自律探索中の画像・動画フレーム観測。
- `autonomous_scan`: 自律探索中のページ観測。
- `autonomous_scroll`: 自律探索中のスクロール。
- `autonomous_link`: 自律探索中のリンク遷移。
- `autonomous_button`: 自律探索中のボタンクリック。
- `autonomous_input`: 自律探索中のテキスト入力。
- `click_change` / `autonomous_change`: クリックや自律操作後、同一ページ上に現れた追加/削除テキスト、URL/title/scroll変化。

拡張は観測データを `chrome.storage.local` に保存する。外部サーバーへ送信しない。

メディア刺激では、種別、再生時間、進捗、音量、ミュート、再生速度、ready/network state、動画サイズ、メディア要素周辺のテキスト、取得可能な字幕cueをSNNトークン化する。
YouTubeでは標準のTextTrackに加え、表示中キャプションDOMとトランスクリプトパネル/説明欄も `mediaCueText` / `mediaTranscriptText` として同時に入力する。
音声そのものの波形解析や自前文字起こしはまだ行わないが、再生状態・音量・速度・進捗などの音声/動画状態トークンと、字幕/文字起こし語彙は同じステップで結合される。

## 視覚野モデル

画像や動画フレームは全ピクセルをモデルへ保存しない。ブラウザ上の `img` / `video` を16x16に低解像度サンプリングし、明暗、彩度、エッジ量、支配色、表示サイズ、画面内位置、可視率を `visual:*` トークンへ変換する。外部画像などでCanvas読み取りがCORSにより失敗する場合は、ピクセル特徴は `visual:pixels:unreadable` とし、サイズ、位置、可視率、alt/周辺テキストだけを学習する。

**テキスト領域の視覚結合**が視覚野の中心機能である。Web上のテキストを読む際、そのテキストが実際に描画されている矩形だけを `chrome.tabs.captureVisibleTab` で切り出し、16x16 の視覚特徴（`visual:kind:text-region`）として抽出する。SNN では視覚特徴トークンの直後に `source:text-region` アンカー経由で、その矩形の本文語彙トークンへ Hebbian 結合する。つまり「この見た目の文字列」と「この語彙」が同じステップで結び付く。動画では字幕 DOM（YouTube キャプション等）または動画下部の字幕帯を同様に切り出し、`text_visual_caption` として cue 語彙と結合する。

`img` / `video` 全体のスキャン（`visual_scan`）に加え、ページ表示・スクロール・定期サンプル・クリック・選択・動画再生中サンプルで `text_visual*` イベントが発行される。

この層は将来のカメラ入力にも流用できる。カメラ映像は `video` フレームとして扱い、`visual:kind:camera-or-stream` から同じ特徴抽出パイプラインに入る。SNN側では `visual:cortex:v1` を持つ特徴スパイクとして膜電位と結合を更新し、言語・操作・報酬との共起で視覚特徴を意味づける。画像/動画の alt・aria-label・周辺キャプションなどのメタデータテキストは `source:visual-text`、画面上に描画されたテキスト本体は `source:text-region` 経由で語彙トークンへ結合される。`img` / `video` クリック時も同一イベント内で視覚特徴とテキストが同時入力される。

## クロスモーダル関係値

通常の隣接トークン結合 `associations` に加えて、画像・音声・動画・テキストの異なるモダリティ間には `crossModalRelations` を持つ。同じ観測イベント内で共起したニューロンを `image`, `audio`, `video`, `text` に分類し、異なるモダリティ同士の対称的な関係重み `w` と共活動量 `coactivity` を更新する。

例:

- 画像特徴 `visual:hue:blue` と周辺テキスト `sky` の関係。
- 動画フレーム特徴 `visual:edge:high` と字幕語彙の関係。
- 音声/動画状態 `media:audio` / `media:video` とトランスクリプト語彙の関係。

サイドパネルの `Cross-modal Relations` で、強いモダリティ間ニューロン関係を確認できる。

## 自律探索

`START Explore` を有効にすると、content script がページ内で周期的に教師なし探索を行う。学習が OFF でも探索操作（スクロール、クリック、入力、リンク遷移）は実行される。SNN モデルへの反映は `START Learning` が ON のときだけ行われる。

- 周期ごとにページを観測し、`autonomous_scan` としてSNNへ入力する。
- 表示中の画像・動画フレームを `autonomous_visual` として視覚野へ入力する。
- ページをゆっくりスクロールし、見える言語とレイアウト変化を `autonomous_scroll` として学習する。
- 一定確率で、現在表示されているリンクへ遷移し、`autonomous_link` として学習する。外部サイトへの遷移も許可する。
- 一定確率で、表示中の `button`, `role="button"`, button系input, `summary` をクリックし、`autonomous_button` として学習する。
- 一定確率で、表示中の `input`, `textarea`, `contenteditable` に短い探索用テキストを入力し、`autonomous_input` として学習する。
- `mailto:`, `tel:`, `javascript:`, downloadリンク、ログアウト、削除、購入、支払い系のラベルは避ける。
- 送信/購入/支払い/削除/ログアウト系に見えるボタンは避ける。passwordやsecret系に見える欄への自動入力は避ける。
- 操作後に同一ページ内のエラー、成功メッセージ、DOM更新などが出た場合は、約0.9秒後に差分を `autonomous_change` または `click_change` として追加学習する。
- `Explore interval seconds` で探索周期、`Explore max same-site navigations` で自律リンク遷移の最大回数を調整できる。

探索はChromeが開いているページ上でのみ動く。バックグラウンドで新しいタブを勝手に生成するものではない。

## 脳型学習の近似

人間の脳そのものではないが、以下の学習機構をSNNモデルに入れている。

- 報酬予測風の `dopamine`: 正/負の経験で可塑性を増減させる。
- 注意/新奇性風の `acetylcholine`: 新しい語彙や刺激を強く学習する。
- 覚醒/驚き風の `norepinephrine`: クリック、シーク、メディア刺激などの顕著な入力を強める。
- 気分安定風の `serotonin`: 正の経験を蓄積し、負の経験で下がる。
- `fatigue`: 長時間の連続学習で可塑性を弱める。
- 恒常性可塑性: 発火しすぎるニューロンは閾値を上げ、発火しにくいニューロンは閾値を下げる。
- 睡眠リプレイ風固定化: 一定ステップごとに高報酬・高顕著性の観測を再評価し、強いシナプスを安定化する。
- 構造化入力: イベント種別、同一オリジン、viewport、スクロール位置、対象タグを `context:*` トークンとして入れ、単なるホワイトノイズ的な語彙列ではなく文脈つき刺激として扱う。
- アナログ決断: スパイク回数だけでなく、膜電位を閾値で割った興奮度と現在のシナプス重みを `analogDrive` として結合更新に使う。発火が0でも膜電位が高い場合は弱い学習信号を残す。
- 大脳基底核風のGo/No-Go: 各シナプスに `d1Go` と `d2NoGo` を持たせ、良い予測誤差ではGoを強め、悪い予測誤差ではNo-Goを強める。
- RPE: シナプスごとの `rewardPrediction` と実報酬の差を使い、予想外に良い/悪い経験ほど強く更新する。
- 側抑制: 睡眠整理時に、リプレイされた強い経路の周辺にある弱い競合経路を軽く抑制する。
- シナプススケーリング: 1ニューロンから出る結合強度の総量が膨らみすぎないよう正規化する。
- 構造語の抑制: 接続詞、助詞、冠詞などは削除せず、低い `importance` の構造ニューロンとして残し、頻出だけで強結合化しないよう更新量を下げる。
- 記憶保護: 安定化したニューロンやシナプスは、新しい語彙に上書きされにくくなる。
- 遅延報酬: 最近発火したシナプスに eligibility trace を残し、後から発生した成果イベントを過去の原因候補へ割り当てる。
- モラル評価: 危害、プライバシー、欺瞞、同意、支援/安全を内的価値信号として評価し、報酬と可塑性を調整する。

## 発見報酬（新知識・新単語・新経験）

未知の語彙やページ、操作経験を正の報酬として扱う。`context:*` や `visual:*` などの構造トークンに薄められないよう、意味語彙（semantic token）中心に評価する。

- **新単語**: 語彙に未登録の意味トークン1語あたり最大 +0.038（上限 +0.45）、semantic novelty 比率 × 0.48 も加算。
- **新ページ**: 初めて訪れる origin で +0.22（`page_view` 遅延信号にも +0.18）。
- **新経験**: 同一ページで初めてのイベント種別（例: 初クリック）で +0.10。
- **新知識の出現**: `click_change` 等で追加された未知語彙ごとに遅延報酬（最大 +0.35）。
- **神経調質**: semantic novelty はアセチルコリン（注意・学習ゲート）を直接上げる。
- **確認**: trace の `discovery`（`novel_words`, `new_page`, `new_experience`）、メトリクスの `Discovery rewards`, `Novel words rewarded`, `New page discoveries`。

## 語彙の部分一致（lexical bridge）

`wiki` と `Wikipedia` のように表記が異なり部分一致する語は、単独トークンだけでは隣接 Hebbian 結合が張れない。そこで各意味語の直後に `lex:*` ブリッジトークンを挿入する。

- `lex:prefix:wiki`: 語頭プレフィックスの共有ハブ（`wiki` と `wikipedia` が同じニューロンへ接続）
- `lex:form:wiki` / `lex:form:wikipedia`: 語の正規形アンカー
- `lex:alias:wiki` / `lex:extends:wikipedia`: 語彙に両方登録済みのとき、短語↔長語を明示的に橋渡し
- 同一イベント内に短語と長語が共存するときは `lexical_link` trace で直接結合も強化

視覚テキスト結合（`source:text-region`）でも、語彙トークンの直後に上記ブリッジが入るため、視覚特徴 → 長語 → `lex:prefix:wiki` → 短語側の経路が形成される。

## 睡眠整理アクション

`Sleep Consolidate` は、学習中に溜まったSNNモデルを明示的に整理する手動アクションである。

- 強い結合、安定した結合、D1/Goが強い結合、遅延報酬の痕跡が残る結合を優先してリプレイする。
- D2/No-Goが強い結合は重みの伸びを抑え、誤った行動選択に相当する経路を弱める。
- リプレイされた経路の周辺にある競合経路へ側抑制をかける。
- 出力結合の総量を `maxOutgoingWeight` 以下にスケーリングし、過発火を防ぐ。
- ニューロンごとの発火率が `targetFiringRate` を超える場合は閾値を上げ、低い場合は少し下げる。
- `fatigue` を下げ、`serotonin` を少し回復させ、次の学習に入りやすい状態へ戻す。

サイドパネルでは `Sleep cycles`, `Sleep replayed`, `Sleep scaled`, `Sleep inhibited`, `RPE` を確認できる。

## 学習停止対策

既存ユーザーの保存済みモデルが古い内部形式でも、読み込み時に `vocabulary` / `associations` を現在の配列形式へ正規化してから学習する。Chrome Storage の容量上限で保存に失敗した場合は、観測本文、イベントtrace、弱い結合を圧縮してリトライする。Manifestには `unlimitedStorage` を付与し、長時間のWeb操作でも保存失敗で学習ループが止まりにくいようにしている。

## スパイクtraceとモジュール接続

発火したニューロンは `token_spike` traceとして記録され、`meta.routeKey` に `neuron:<token>` を持つ。
`module:`, `skill:`, `action:`, `tool:`, `media:`, `moral:`, `instinct:`, `source:` で始まるトークンは `meta.moduleTrigger` にも入り、後付けモジュールやスキルの発火条件として使える。

## 遅延報酬の設計

即時報酬だけでは「動画をしばらく見た後に最後まで見た」「スクロール後にクリックした」「読んだ後にテキスト選択した」といった時間差のある成果を学習できない。そのため、SNNは各シナプス発火に短期の資格痕跡を残す。

- `eligibility.synapses`: 最近活動したシナプス、痕跡量、最終ステップを保存する。
- `delayedRewardWindow`: 何ステップ前まで報酬を戻すかを決める。
- `eligibilityDecay`: 古い痕跡ほど弱くする。
- `delayedLearningRate`: 遅延報酬で重みを更新する専用学習率。
- 成果信号: `click`, `selection`, `input`, `media_ended`, 長く視聴した後の `media_pause`, 既知ページへの再訪など。
- 本能層の遅延危機: `click_change`, `autonomous_change` で追加された警告文や不審 URL、`page_view` で判明した危険ページ。直前操作への帰属は `instinctDelayedWindow` が通常の `delayedRewardWindow` より長い。

成果信号が来ると、まだ残っている eligibility trace に対して報酬を配分し、関連シナプスの重みを後から更新する。サイドパネルでは `Delayed rewards`, `Credit assigns`, 各結合の `delayed` が確認できる。

## モラル評価の設計

SNN自体に大規模な倫理推論能力を持たせるのではなく、機械倫理や安全な強化学習で使われる reward shaping / constraint の考え方を、SNNの報酬系へ写像する。

- 危害リスク: 暴力、自傷、脅迫、虐待などに関する刺激を負の内的信号にする。
- プライバシー尊重: パスワード、APIキー、秘密鍵、メール、電話番号、カード番号などを検出し、保存前に `[redacted-sensitive]` へ置換する。
- 欺瞞リスク: 詐欺、フィッシング、なりすまし、偽装などを負の信号にする。
- 同意/許可: 無断、許可、同意などをモラル文脈として扱う。
- 向社会性: 支援、安全、保護、説明、尊重などを弱い正の信号にする。

モラル評価は `moral:*` トークンとしても入力されるが、危険語や個人情報そのものを強く記憶させるのではなく、`moral:privacy-risk` や `moral:harm-risk` のような抽象ラベルを学習させる。サイドパネルでは `Moral events`, `Moral penalties`, `Redactions` を確認できる。

## 本能層と危機回避

モラル評価が社会的・倫理的な reward shaping であるのに対し、本能層はより原始的な危機回避（amygdala 的な脅威検出）を SNN へ写像する。ページ上の警告文、フィッシング語彙、マルウェア警告、エラーページ、不審 URL（HTTP 直アクセス、IP 直打ち、punycode など）を `instinct:*` トークンとして入力し、回避衝動を負の報酬と no-go 経路へ結び付ける。

- 即時評価: 現在のページ本文・URL・タイトルから `instinct:threat-active`, `instinct:crisis:*`, `instinct:escape-urge` などを生成する。
- 遅延評価: クリックや自律操作の後に現れる `click_change` / `autonomous_change` の追加テキスト、URL 遷移先、危険な `page_view` を後から評価する。直前の操作は `instinct.actionTrace` に残し、資格痕跡（eligibility trace）へ `instinctDelayedWindow`（既定 48 ステップ）の範囲で危機信号を戻す。
- ノルアドレナリン: 危機の緊急度に応じて上昇し、探索・学習の覚醒度を上げる。
- 設定: `Instinct learning`（Off / Observe / Shape reward / Constraint）、`Crisis sensitivity`。
- 統計: `Instinct events`, `Instinct avoidances`, `Instinct delayed` をサイドパネルで確認できる。

`instinct:*` トークンも `meta.moduleTrigger` に入り、後付けモジュールの危機回避スキル発火条件として使える。

## 学習設定

サイドパネルの `Learning Settings` で、学習の扱いを変更できる。

- `START Learning` / `STOP Learning`
  - 現在のタブだけON/OFFする。
  - 新しいタブの初期状態は STOP（学習OFF）。
  - OFFのタブはブラウザイベントを受けてもモデル、語彙、シナプスは更新しない。
- `START Explore` / `STOP Explore`
  - 現在のタブだけON/OFFする。
  - 学習 OFF でもブラウザ操作（スクロール、クリック、入力、リンク遷移）は実行される。
  - SNN への学習反映は `START Learning` が ON のときのみ。
- `Model`
  - 現在のタブが学習するモデルを選ぶ。
  - 複数タブで同じモデルを選ぶと、同一SNNへイベントを集約できる。
  - `New Model` で別モデルを作成できる。
- `Learning Tabs`
  - 学習中または探索中のタブ、現在のモデル、URLを一覧表示する。

- `Input capture`
  - `Metadata only`: 入力欄は `[input:text]` のような種別だけ学習する。
  - `Text values`: 通常テキスト入力は値も学習する。password型は値を送らない。
  - `Full values`: password型を含む入力値を学習対象にする。
- `Sensitive text`
  - `Redact`: メール、電話、カード、APIキー等を `[redacted-sensitive]` に置換する。
  - `Abstract`: 具体値は保存せず `sensitive-value` として抽象化して学習する。
  - `Full`: 具体値を保持して学習する。
- `Moral learning`
  - `Off`: モラル評価を学習に使わない。
  - `Observe`: 評価値は記録するが報酬には効かせない。
  - `Shape reward`: モラル評価を報酬へ加減算する。
  - `Constraint`: 高リスク時に報酬上限と可塑性も制限する。
- `Privacy sensitivity`: 個人情報や秘密情報をどれだけ強い価値信号として扱うか。
- `Sensitive reward`: 個人情報やパスワード等を「価値の高い情報」としてどれだけ正の学習信号にするか。
- `Explore interval seconds`: 自律探索の実行間隔。
- `Explore max same-site navigations`: 同一サイト内で自律遷移する最大回数。0ならスクロール観測のみ。
- `Compute backend`
  - `Auto`: 通常はCPU、語彙数や処理時間が増えたらWebGPUを試す。
  - `CPU only`: CPUだけで学習する。
  - `WebGPU preferred`: WebGPUが使える場合は補助計算に使う。
- `Performance limit`
  - 100%に近いほど学習イベントとWebGPU補助計算を多く使う。
  - 低くすると高頻度イベントを間引き、WebGPU呼び出し間隔を空け、GPUへ渡すトークン数を減らす。
  - 学習速度とブラウザ負荷のトレードオフ用の上限であり、クリック、入力、ページ表示などの優先イベントはできるだけ残す。

## 性能計測とWebGPU

サイドパネルには以下を表示する。

- `Model size`: 現在の学習済みSNNモデルをJSON化した概算サイズ。
- `CPU load`: 1学習ステップの平均処理時間から推定した拡張内の負荷。OS全体のCPU使用率ではない。
- `Step time`: 直近の学習ステップ処理時間。
- `Avg step`: 移動平均の学習ステップ処理時間。
- `Backend`: 直近ステップで使った `cpu` / `webgpu` / `cpu-throttled`。
- `WebGPU`: WebGPUが利用可能か。
- `Perf limit`: 現在の負荷上限。
- `Learn throttle`: 負荷上限に応じた学習イベント間引き間隔。
- `GPU token cap`: WebGPU補助計算へ渡す最大トークン数。
- `GPU throttle`: WebGPU補助計算の最小実行間隔。

CPU側は語彙とシナプス検索をMapインデックス化している。WebGPUが使える場合は、アクティブ語彙ニューロンの膜電位更新をcompute shaderで補助する。Chromeや環境によってWebGPUが拡張service workerで使えない場合は自動的にCPUへ戻る。

## モデルインポートと追加学習

サイドパネルの `Import Model` は以下の形式を受け付ける。

- `.edensnn`: 拡張機能またはEDEN互換の `EDENSNN1` コンテナ。ブラウザ内だけで復元できる。
- `.json` / `.chat.json`: `scripts/export_dst_chat_model.py` や `scripts/serve_snn_chat_lab.py` が出力した会話用モデル。HTMLタグなどの生入力ではなく、抽出済み語彙、関連、クロスモーダル関係を取り込む。
- `.pt`: PyTorch checkpoint。Chrome拡張はpickle/torch tensorを直接安全に解釈できないため、ローカル変換サーバー経由で `.chat.json` 相当へ変換してから取り込む。

`.pt` をインポートする場合は、先にリポジトリルートで以下を起動する。

```bash
python scripts/serve_snn_chat_lab.py
```

その状態で拡張機能の `Import Model` から `.pt` を選ぶと、`http://127.0.0.1:8765/api/convert-pt` に一度だけ送られ、変換後の語彙・シナプス・クロスモーダル関係が拡張機能のモデルへ正規化される。取り込み時の確認ダイアログで `OK` を押すと現在選択中モデルへマージし、`Cancel` を押すと新しいモデルとして追加する。どちらの場合も、その後のWeb操作からの追加学習はChrome内のSNNで継続され、PyTorchは不要になる。

ブラウザ内学習はPyTorchのautogradやoptimizer状態をそのまま再現するものではない。取り込む対象は、学習済みチェックポイントから会話/探索に使える語彙ニューロン、重みつき関連、モダリティ関係へ射影した状態である。実行時の高速化は現在 `CPU` / `WebGPU preferred` で行い、Nim/WASMなどの追加バックエンドはこの正規化済みモデル表現に接続する形で拡張できる。

## モデル形式

エクスポートされるファイルは `.edensnn`。先頭に `EDENSNN1` マジックヘッダを持つEDEN互換コンテナで、payload内に `modelKind: "browser-language"` のSNNモデルを含む。

## 実装メモ

- `manifest.json`: Manifest V3拡張定義。
- `src/contentScript.js`: 実ページDOMとユーザー操作を観測するcontent script。
- `src/background.js`: イベントを受け取り、SNNモデルを更新して永続化するservice worker。
- `src/snnCore.js`: ブラウザ言語SNNの逐次学習ロジック。
- `src/sidepanel.*`: 学習状態の可視化と `.edensnn` エクスポートUI。

## 注意

`host_permissions` とcontent scriptの `matches` は `<all_urls>` にしている。ブラウザ全体を学習環境にするための設定だが、配布用には対象ドメインを絞るか、ユーザーが明示的に学習対象を選ぶUIを追加するべきである。
