# EDEN 自律生成ボディ + 闊歩学習 — Design

作成日: 2026-07-11  
状態: 承認済み方針（D + 最小 UI + 自律学習）

## 背景と目的

ユーザー目標:
1. 実際に使える 3D 体を速く用意する
2. EDEN 内でその体で闊歩させる
3. 環境との相互フィードバックで SNN が勝手に学習し続ける

制約（今回明示）:
- **UI 設定項目は極力ゼロ**（見られない／触りたくない前提）
- 利便性優先: 起動したら **自動で体が付き、動き、学習する**
- 学習はユーザー操作なしで継続

## スコープ

### 含む（スライス D 改）
- ブラウザ内プロシージャル体 → 最小 GLB（blob URL）
- SNN Life スポーン時に自動で体を付与
- **ワールド入場時に自律バイオトープを自動開始**（設定画面に深いオプションを増やさない）
- 既存 `stepEmbodiedCreature` による移動・報酬・STDP をそのまま駆動
- seed を localStorage に保存し、リロード後も同じ体を再生成

### 含まない
- 体型スライダ、生成パラメータ UI、手動「生成して闊歩」必須フロー
- Python 相互学習本番接続（次スライス）
- 本格リグ歩行アニメ / SOTA image→3D 学習

## 自律ループ（ユーザー操作なし）

```
EDEN mount / ワールド ready
  → ensureAutonomousBiotope()
      → 既定 N 体（例: 2）の SNN Life がいなければスポーン
      → 各体: seed → procedural GLB → glbUrl 付き entity
  → 毎フレーム: 環境刺激 → stepEmbodiedCreature → 位置更新 + STDP
  → 永続: snapshot + body seed
```

学習の「勝手さ」:
- 既存の wander / seekStimulus / avoidOverload と報酬変調 STDP を **常時 ON**
- 追加 UI トグルは作らない（将来デバッグ用フラグはコード定数のみ）

## アーキテクチャ

| 部品 | 役割 |
|---|---|
| `proceduralBody.ts` | seed → メッシュ部品（胴・頭・肢） |
| `bodyGlb.ts` | メッシュ → glTF binary → blob URL |
| `generatedBodyRegistry.ts` | creatureId ↔ seed / bounds（localStorage） |
| `Game.tsx` 改修 | 自動スポーン、entity に glbUrl、フォールバック geometry |

## UI 方針

- **設定項目を増やさない**
- 既存の SNN Life 表示・トレースはそのまま（見るだけ）
- 手動スポーン UI が既にある場合は残してよいが、**主経路は自動**
- 定数（コード内）: `AUTONOMOUS_SNN_COUNT = 2`, `AUTONOMOUS_ENABLED = true`

## エラー処理

- GLB 生成失敗 → 従来 custom geometry で闊歩継続
- blob URL リーク防止: 再生成前に revoke、アンマウント時クリーンアップ
- WS 未接続時: ローカル players マップ上でも描画できる経路を優先（既存ローカル SNN 表示と整合）

## 成功条件

1. EDEN を開くだけで SNN 生命体が **生成ボディ付きで** 動き出す
2. ユーザーがパラメータを触らなくても STDP / 報酬が回り trace が流れる
3. リロード後も seed から体が復元される
4. 設定 UI に新しい必須項目が増えていない

## 次スライス（記載のみ）

- Python 生成 / 学習正典との WebSocket 相互接続
- 生成品質の学習（op 系列・occupancy）を Python 側で回し GLB を差し替え
