# Cosmos Policy 拡散メカニズム検証レポート（インデックス）

このファイルは検証結果を整理したドキュメント群のインデックスです。  
各検証内容は以下の個別ファイルに詳細を記載しています。

---

## ドキュメント一覧

| ファイル | 内容 |
|---------|------|
| [index.md](../index.md) | **全体インデックス**: 全検証の主要発見・実験コード・結果ディレクトリ一覧 |
| [results_01_action_denoising.md](results_01_action_denoising.md) | **テーマ1 (アクション)**: 実験設定・デノイジング過程解析・層別再検証 |
| [results_02_action_features.md](results_02_action_features.md) | **テーマ2 (アクション)**: DiT 中間特徴量・線形プロービング・言語クロスアテンション |
| [results_03_image_generation.md](results_03_image_generation.md) | **画像生成**: 将来画像ラテット解析・アクションとの比較 |
| [results_04_self_attention.md](results_04_self_attention.md) | **自己注意**: 入力画像・proprio への注意パターン（モダリティ対応型注意） |

---

## 全検証を通じた主要発見（要約）

### 1. Confidence-to-Commitment（確信度から確定へ）

デノイジングステップ間の変化量が単調増加するパターン（k=0→1 < k=1→2 < k=2→3 < k=3→4）は、  
アクション・画像・全 7 DiT 層にわたって例外なく成立する普遍的法則。  
EDM の高 σ 段階では x̂₀ が平均値付近（underdispersed）に留まり、低 σ 段階で確定的予測へ大きく収束する。

### 2. DiT の 4 フェーズ処理構造

| フェーズ | Block | 特徴 |
|---------|-------|------|
| 汎用前処理 | 0 | 分散 ≈ 1.3、全 policy call でほぼ同一出力 |
| 安定中間表現 | 4–13 | CKA ≈ 0.98〜1.00、分散 ≈ 1,800–2,000 |
| 高次セマンティック | 18–22 | 分散 ≈ 8,600–14,000、タスク状態に最も敏感 |
| 最終確定出力 | 27 | 分散 ≈ 212,000、Commitment の主要な座 |

### 3. 3 種の Commitment メカニズム

| 層グループ | 型 | 指標 |
|-----------|---|------|
| Block 4–13 | 方向変化型 | cos sim k=3→4: ≈ 0.90 |
| Block 18–22 | スケール増大型 | ノルム +12〜26% |
| Block 27 | 収縮型 | ノルム −7%、cos sim ≈ 0.9996 |

### 4. モダリティ対応型自己注意

各出力トークンが対応する入力トークンに最も強く注目する構造:
- `future_wrist → curr_wrist`、`future_primary → curr_primary`（同一視点参照）
- `action → proprio`（固有感覚から制御指令を生成）
- アクション生成は proprio + curr_wrist（グリッパー領域）に集中型（CV ≈ 1.0）
- 将来画像生成は対応する現在画像全体を均等参照（CV ≈ 0.29〜0.36）

### 5. 言語の意味的処理の層別分業

Block-9 → "from" (ソース位置) → Block-18 → "[object]" (把持対象) → Block-27 → "cabinet + </s>" (目的地)  
注目パターンはデノイジングステップ・スキルフェーズに対してほぼ不変（変化 < 1%）。

---

## 検証設定（共通）

- **モデル**: Cosmos-Policy-RoboCasa-Predict2-2B（EDM ベース 2B-parameter DiT）
- **タスク**: PnPCounterToCab (RoboCasa) — Pick-and-Place: Counter → Cabinet
- **成功率**: 70.0%（10 エピソード）
- **σ スケジュール**: [80.0, 42.3, 21.0, 9.6, 4.0]
- **プローブ層**: Block-0, 4, 9, 13, 18, 22, 27
