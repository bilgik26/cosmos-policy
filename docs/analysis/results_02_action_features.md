# DiT 中間特徴量・スキル表現解析

**テーマ2: アクション生成の内部表現検証**  
**日付**: 2026-06-28 | **タスク**: PnPCounterToCab | **成功率**: 70.0% (7/10)  
**データ**: 218 policy calls × 7 層 × 5 デノイジングステップ × 2048 次元

---

## 1. 実験設定

| 項目 | 内容 |
|------|------|
| プローブ層 | Block-0, 4, 9, 13, 18, 22, 27（28 ブロック中 7 箇所） |
| 特徴量 | action token (T=5) の block 出力 (B, T, H, W, D) を H×W 空間平均した 2048 次元ベクトル |
| 総サンプル数 | 218 policy calls × 7 層 × 5 ステップ = 7630 本の特徴ベクトル |
| スクリプト | `theme2_analysis.py` (特徴量収集) + `theme2_plot.py` + `theme2_linear_probe.py` + `theme2_crossattn.py` |

---

## 2. テーマ 2-1: 特徴量の分散（層ごとの表現の多様性）

action token 特徴ベクトルの全次元分散の合計（218 policy calls 間の variability）:

| 層 | 説明 | 分散合計 (k=0) | 分散合計 (k=4) |
|----|------|----------------|----------------|
| Block-0 | Shallowest | 1.36 | 1.32 |
| Block-4 | Early | 1,827 | 2,020 |
| Block-9 | Early-Mid | 1,843 | 2,030 |
| Block-13 | Mid | 1,954 | 2,200 |
| Block-18 | Late-Mid | 8,694 | 8,612 |
| Block-22 | Deep | 14,225 | 13,972 |
| Block-27 | Deepest | **211,784** | **214,929** |

**所見**: 深い層ほど特徴量の分散が劇的に増大（Block-0 対比 Block-27 は **15 万倍以上**）。
- Block-0: 全 218 calls でほぼ同一の出力（分散 ≈ 1.3）→ 汎用前処理・固定特徴抽出器
- Block-27: call ごとの状況（フェーズ・観測）を高精度で弁別する表現 → タスク特化

→ プロット: `theme2_results/theme2_feature_variance.png`

---

## 3. テーマ 2-2: デノイジングステップ間の特徴量変化量（層別）

k=0 と k=4 の特徴ベクトルの L2 距離（218 calls 平均）:

| 層 | 変化量 (mean ± std) |
|----|---------------------|
| Block-0 | **3.67 ± 0.05**（最小） |
| Block-4 | 115.3 ± 3.6 |
| Block-9 | 105.7 ± 1.9 |
| Block-13 | 115.1 ± 2.2 |
| Block-18 | 119.7 ± 4.4 |
| Block-22 | 110.7 ± 9.6 |
| Block-27 | **329.0 ± 29.9**（最大） |

**所見**:
- Block-0 は k=0→k=4 でほぼ変化しない（L2 ≈ 3.67）→ デノイジングから切り離された固定前処理器
- Block-4〜22 は中程度・類似した変化量（L2 ≈ 105〜120）→ 程よく更新される中間表現
- Block-27 は他の層の約 3 倍（L2 ≈ 329）→ **最深部のみで最終的な確定的予測への大きな移行が起きる**

テーマ 1-A の「k=3→4 での x̂₀ 変化量急増」は、Block-27 の特徴量が最終ステップで最も大きく変化するという内部メカニズムに対応する。

→ プロット: `theme2_results/theme2_feature_change_by_layer.png`, `theme2_step_change_per_layer.png`

---

## 4. テーマ 2-3: 層間表現類似度（Linear CKA）

各層ペアの Linear CKA（k=4、値域 [0,1]、1 = 同一表現）:

|  | Block-0 | Block-4 | Block-9 | Block-13 | Block-18 | Block-22 | Block-27 |
|--|---------|---------|---------|----------|----------|----------|----------|
| **Block-0** | 1.00 | **0.70** | **0.71** | **0.74** | 0.55 | 0.52 | 0.54 |
| **Block-4** | | 1.00 | **0.999** | **0.981** | 0.52 | 0.53 | **0.844** |
| **Block-9** | | | 1.00 | **0.984** | 0.52 | 0.54 | **0.845** |
| **Block-13** | | | | 1.00 | 0.58 | 0.60 | **0.841** |
| **Block-18** | | | | | 1.00 | **0.815** | 0.55 |
| **Block-22** | | | | | | 1.00 | 0.61 |
| **Block-27** | | | | | | | 1.00 |

**CKA クラスター構造**:

1. **Block-4, 9, 13 クラスター**: CKA ≈ 0.98〜1.00（ほぼ同一の表現）→「収束ゾーン」
2. **Block-18, 22 クラスター**: CKA ≈ 0.81（高類似）。Block-4〜13 とは明確に分離（CKA ≈ 0.52〜0.60）
3. **Block-0（孤立）**: 他の全層との CKA が 0.52〜0.74（独自の前処理モード）
4. **Block-27（橋渡し）**: Block-4〜13 との CKA が 0.84 で比較的高い → 中間部の表現を参照しながら最終出力を生成

**4 フェーズ構造**:
```
Block-0       → Blocks 4–13           → Blocks 18–22      → Block-27
前処理（汎用）  安定中間表現（収束ゾーン）  高次セマンティック処理  最終確定出力
分散 ≈ 1.3    分散 ≈ 1,800–2,000       分散 ≈ 8,600–14,000  分散 ≈ 212,000
```

→ プロット: `theme2_results/theme2_cka_matrix.png`

---

## 5. テーマ 2-4: PCA による可視化

各層の特徴量を PCA で 2 次元に投影（k=0 および k=4、call index で色付け）:

- **Block-0**: 全 218 点がほぼ 1 点に集中。call index との相関なし。
- **Block-4〜13**: 分散が広がり始め、異なる episode 間で適度な分離が現れる。
- **Block-18〜22**: 特徴空間が広がり、同一 episode 内の call 進行と PC1 に緩やかな相関が見られる場合がある。
- **Block-27**: 最も広い特徴空間、episode 間の分離が明確。

→ プロット: `theme2_results/theme2_pca_k4.png`, `theme2_pca_k0.png`, `theme2_pc1_timeseries_k4.png`

---

## 6. テーマ 2-5: 線形プロービング（スキルフェーズ予測）

**手法**:
- Global PCA で 2048→50 次元に削減
- Ridge 回帰（λ=1.0）+ one-hot → argmax で予測
- LOEO（Leave-One-Episode-Out）10-fold 交差検証
- ラベル種別:
  - `progress_3`: エピソード内進行度 → 3 クラス [early/mid/late]（chance = 34.4%）
  - `gripper_3`: k=4 の平均グリッパー値 → 3 クラス（chance = 33.5%）
  - `gripper_2`: グリッパー開/閉 バイナリ（chance = 57.8%）

**結果**:

| ラベル種別 | k | Block-0 | Block-4 | Block-9 | Block-13 | Block-18 | Block-22 | Block-27 | Chance |
|-----------|---|---------|---------|---------|----------|----------|----------|----------|--------|
| **progress_3** | 0 | 37% | 30% | 31% | 36% | 50% | 62% | **70%** | 34% |
| **progress_3** | 4 | 45% | 34% | 37% | 44% | 49% | 61% | **66%** | 34% |
| **gripper_3** | 0 | 46% | 44% | 47% | 52% | 78% | 85% | **88%** | 34% |
| **gripper_3** | 4 | 70% | 71% | 71% | 75% | 77% | 85% | **87%** | 34% |
| **gripper_2** | 0 | 62% | 51% | 55% | 62% | **91%** | **95%** | **99%** | 58% |
| **gripper_2** | 4 | **93%** | **94%** | **96%** | **94%** | 93% | 96% | **98%** | 58% |

**主要な発見**:

1. **グリッパー開閉は最深部で超早期確立**: Block-27 k=0 で gripper_2 が 99% → 最大ノイズ段階でも把持意図が確立されている。「グリッパーは最も早く最も確実に Commit される次元」。

2. **k=4 での浅い層の急改善（+31 ポイント）**: Block-0 の gripper_2 が k=0: 62% → k=4: 93%。最終ステップではノイズ入力 x_t が真のアクションに近くなり、グリッパー意図が入力信号に直接反映される。モデルの「賢さ」ではなく入力品質の向上による結果。

3. **タスク進行（progress_3）の弱い線形分離性**: 最高でも Block-27 k=0 の 70%（chance から +36%）。gripper_2 の最高 99% と対比して「今タスクの何フェーズ目か」は特徴ベクトル上で explicit に表現されていない → 視覚観測への implicit な依存で成立している可能性。

4. **深さ依存のグラジェント**: progress_3 での k=0 精度が Block-4〜9 ではほぼ chance レベル → 中間収束ゾーンよりも最深部がタスク時間的文脈を保持。

→ プロット: `theme2_linear_probe_results/probe_accuracy_by_layer.png`, `probe_k0_vs_k4.png`, `probe_accuracy_heatmap.png`

---

## 7. テーマ 2-6: 言語クロスアテンション解析

**手法**: 新規 5 エピソード（160 policy calls）で `block.cross_attn` に forward hook → Q, K を再計算 → `softmax(QK^T/√d)` で attention weights 取得 → action token (T=5) の位置に限定して平均化

**タスク説明のトークン例**:
`['pick', 'the', 'canned', 'food', 'from', 'the', 'counter', 'and', 'place', 'it', 'in', 'the', 'cabinet', '</s>']` (14 real tokens)

**層別の注目トークンパターン (k=4、real tokens 内で正規化)**:

| 層 | Top-1 | Top-2 | Top-3 | 解釈 |
|----|-------|-------|-------|------|
| Block-0 | 'from' (11%) | '</s>' (11%) | 'and' (10%) | 拡散的・ほぼ一様 |
| Block-4 | 'from' (14%) | 'and' (12%) | 'counter' (11%) | ソース位置への傾き |
| Block-9 | **'from' (25%)** | 'food' (16%) | 'and' (10%) | ソースに強集中 |
| Block-13 | 'from' (13%) | 'food' (10%) | 'and' (10%) | 集中が緩和 |
| Block-18 | **'canned' (18%)** | **'food' (16%)** | 'from' (12%) | オブジェクト名詞へシフト |
| Block-22 | '</s>' (19%) | **'cabinet' (14%)** | 'and' (11%) | 目的地・文末へシフト |
| Block-27 | **'</s>' (44%)** | **'cabinet' (30%)** | 'and' (4%) | 目的地に超集中 |

**主要な発見**:

1. **「ソース → オブジェクト → 目的地」の意味的階層**: 層が深まるにつれて注目トークンが変化。Block-9 → "from" (ピックアップ元) → Block-18 → "[object] food" (把持対象) → Block-27 → "cabinet" (配置先)。PnP タスクの 3 フェーズ（リーチ→把持→配置）と対応する意味的処理の深化。

2. **デノイジングステップ不変性**: k=0〜4 の間で注目パターンはほぼ変化しない（上位トークン attention weight の変化 < 1%）。Cross-attention は固定的な言語的文脈読み取りに特化。

3. **スキルフェーズ不変性**: early/mid/late フェーズ間でも変化は小さい（Block-27 の '</s>' への attention が late で 1.7% 増加のみ）。

4. **Block-27 の '</s>' への超集中（44%）**: T5 の `</s>` は文全体のグローバル要約を符号化する特別なトークン。最深部での注目は「タスクの目標（配置先）」という高レベルな言語意味から最終アクション出力が直接決定されることを示す。

5. **注目エントロピー**: 全 512 トークンへの attention の entropy H ≈ 6.16〜6.24（log(512) ≈ 6.24）。Padding 含む全体では広く分散、real tokens 内での相対的集中がパターンを形成。Block-27 の H=6.15 が最低値（最も選択的）。

→ プロット: `theme2_crossattn_results/crossattn_layer_token_heatmap_k4.png`, `crossattn_by_denoise_step.png`, `crossattn_by_skill_phase.png`, `crossattn_top8_tokens_k4.png`, `crossattn_entropy.png`

---

## 8. 総合サマリー

| 検証項目 | 結果 | 意義 |
|---------|------|------|
| Block 分散比 (0→27) | **1.36 → 211,784（15 万倍）** | 浅い=汎用処理、深い=タスク特化表現の明確な分業 |
| Block-27 デノイズ変化量 | **L2=329（中間層の 3 倍）** | 最終確定出力は最深部でのみ大きく変化（Commitment の座） |
| CKA クラスター構造 | **4–13=収束ゾーン (CKA≈1.0)、18–22=セマンティック** | 28 層が 4 つの処理フェーズに自然分割 |
| 線形プロービング (progress_3) | **Block-27 最高 70%（chance +36%）** | 時間的進行の線形分離は深い層に限定 |
| 線形プロービング (gripper_2) | **Block-27 k=0: 99%、Block-0 k=4: 93%** | グリッパー開閉は最深部で完全分離、最終ステップでは浅い層でも高精度 |
| Cross-Attention 層別注目 | **Blk-9: 'from' 25%、Blk-18: 'canned'+'food' 34%、Blk-27: '</s>'+'cabinet' 74%** | 層深度に沿って「ソース→対象→目的地」と意味的に変化、k・フェーズには不変 |

---

## 9. 考察

1. **DiT 内部の 4 フェーズ構造が定量的に確認された**。Block-0 が固定前処理器として全 call に共通する処理を実行し、Block-4〜13 が「残差接続による安定収束ゾーン」を形成し、Block-18〜22 が観測からタスク状態への高次意味理解を担い、Block-27 が最終確定出力を生成する。

2. **「グリッパーの超早期確立」は重要な新発見**。把持アクションの決定が拡散プロセスの第 1 ステップ（σ=80）の時点で最深部に確立されているという事実は、Cosmos Policy が観測から即座に把持意図を抽出できることを示す。

3. **「タスク時間的進行の弱い線形表現」**は、スキル軌跡の追跡が視覚観測への implicit な依存によって成立しており、特徴ベクトル上での explicit な状態管理として機能していないことを示唆する。

4. **言語の意味的処理の分業**は、各層が PnP タスクの各フェーズに対応する言語要素を処理するという、言語とアクションの深い統合メカニズムを示す。

---

## 10. 出力ファイル一覧

| ファイル | 内容 |
|---------|------|
| `theme2_results/theme2_features.npz` | 全 policy call × 全ステップ × 7 層の特徴ベクトル（再解析用） |
| `theme2_results/theme2_pca_k4.png` | 各層の PCA 可視化（k=4） |
| `theme2_results/theme2_pca_k0.png` | 各層の PCA 可視化（k=0） |
| `theme2_results/theme2_pc1_timeseries_k4.png` | PC1 時系列（call 進行との対応） |
| `theme2_results/theme2_feature_variance.png` | 各層の特徴量分散 |
| `theme2_results/theme2_feature_change_by_layer.png` | k=0→k=4 の層別変化量 |
| `theme2_results/theme2_step_change_per_layer.png` | ステップごとの層別変化量 |
| `theme2_results/theme2_cka_matrix.png` | 7×7 層間 Linear CKA ヒートマップ |
| `theme2_results/theme2_stats.json` | テーマ 2 の全統計量 |
| `theme2_linear_probe_results/probe_accuracy_by_layer.png` | 層別・ラベル種別のプロービング精度 |
| `theme2_linear_probe_results/probe_k0_vs_k4.png` | k=0 vs k=4 の精度比較 |
| `theme2_linear_probe_results/probe_accuracy_heatmap.png` | 層 × ステップ × ラベルの精度ヒートマップ |
| `theme2_linear_probe_results/probe_label_distribution.png` | 3 種ラベルの分布確認 |
| `theme2_linear_probe_results/probe_delta_over_chance.png` | chance からの上昇量 |
| `theme2_linear_probe_results/theme2_probe_stats.json` | プロービング精度の全統計量 |
| `theme2_crossattn_results/crossattn_layer_token_heatmap_k4.png` | 層 × トークンの attention ヒートマップ |
| `theme2_crossattn_results/crossattn_by_denoise_step.png` | デノイジングステップ別 token attention |
| `theme2_crossattn_results/crossattn_by_skill_phase.png` | スキルフェーズ別 token attention |
| `theme2_crossattn_results/crossattn_top8_tokens_k4.png` | 各層上位 8 トークン（バーチャート） |
| `theme2_crossattn_results/crossattn_entropy.png` | 全層 × 全 k の attention entropy |
| `theme2_crossattn_results/theme2_crossattn.npz` | 全 160 policy call の attention weights |
| `theme2_crossattn_results/theme2_crossattn_meta.json` | 実験メタデータ |
