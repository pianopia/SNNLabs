# Vision Shape Inspiration + Morph + External Construct — Design

作成日: 2026-07-11  
状態: 実装済み（初回増分）

## 目的

EDEN 身体 SNN に欠けていた次を埋める。

1. **視覚的形状インスピレーション**（現実に近い「見た形」の粗特徴）
2. **自己ジオメトリを変える必然・報酬**（見た形に近づく）
3. **自分以外のオブジェクトを作る行動と報酬**

## 方針（軽量）

- RGB CNN は使わない。近傍エンティティの **size/shape/位置** をシルエット／AABB の代理視覚とする。
- 追加 UI なし。既定ゴール `imitateAndConstruct`、参照 props 自動配置。
- 計算量は O(近傍オブジェクト) で既存刺激収集と同程度。

## コンポーネント

| 部品 | 役割 |
|---|---|
| `visionShape.ts` | size→特徴、match スコア、inspiration 選択 |
| `constructAction.ts` | 配置リクエストの形・サイズ決定ヘルパ |
| `lif.ts` | sensor vision_* / motor imitate / construct、morph 報酬、pendingConstruct |
| `Game.tsx` | 刺激に vision を流し込み、construct で createEntity |

## 報酬

- **morph**: 視覚 shape との match 改善 + 到達 match
- **construct**: 配置成功（クールダウン付き）+ novelty
- 既存移動・衝突報酬と加算

## 成功条件

- 世界に参照オブジェクトがあるとき vision 電流が非ゼロ
- imitate で body 寸法が参照に寄る
- construct が外物を生成し、チャットまたは players に現れる
- `tsc` / build / スモーク PASS
