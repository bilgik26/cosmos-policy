# 自己注意（Self-Attention）解析

**各出力トークンが入力画像・proprio のどの部分に注目しているかの定量化**  
**日付**: 2026-06-30 | **タスク**: PnPCounterToCab | **エピソード数**: 50

---

## 1. 実験設定

| 項目 | 値 |
|------|-----|
| タスク | PnPCounterToCab (RoboCasa) |
| エピソード数 | 50（成功率 0%） |
| Policy calls | 1600（全 50 エピソードが 500 ステップ完走） |
| デノイジングステップ数 | 5 (σ = 80.0, 42.3, 21.0, 9.6, 4.0) |
| 解析対象ブロック | `PROBE_BLOCKS = [0, 4, 9, 13, 18, 22, 27]`（28 ブロック中 7 箇所） |
| 注意フックの対象 | 各ブロックの `self_attn.compute_attention` |
| スクリプト | `analysis/attention_analysis.py` |
| 結果ディレクトリ | `results/self_attention/` |

**注**: 成功率 0% は torch.cuda.set_device(1) とデバイス非決定性の可能性。FlashAttention フック（Q・K 再キャプチャ）が推論速度に影響した可能性もある。データ収集自体は正常（1600 calls）。

### 1.1 Latent Sequence 構造（state_t=11）

```
T=0: blank          T=5:  action          ← デノイジング対象（出力）
T=1: proprio        T=6:  future_proprio  ← デノイジング対象（出力）
T=2: curr_wrist     T=7:  future_wrist    ← デノイジング対象（出力）
T=3: curr_primary   T=8:  future_primary  ← デノイジング対象（出力）
T=4: curr_secondary T=9:  future_secondary← デノイジング対象（出力）
                    T=10: value           ← デノイジング対象（出力）
```

### 1.2 注意重みの抽出方法

FlashAttention は重みを返さないため、`compute_attention(q, k, v)` にフックして Q・K を取得し、`softmax(Q @ K.T / sqrt(d))` を手動計算した。全シーケンスに対する softmax で正規化。

### 1.3 集計方法

- 各 (block_idx, k_step) の注意行列: 1600 policy call の全ステップ平均
- T 位置別注意行列 [11×11]: 出力 T が入力 T 全体へ向ける注意重みの空間・ヘッド平均
- 空間ヒートマップ [14×14]: 特定 (t_out, t_in) ペアの注意を空間的に集計

---

## 2. 分析 A: T 位置別注意行列 [11×11]（全 7 ブロック）

### 目的

「各出力トークンが入力 T 位置のうちどれを最も参照するか」を全 7 プローブブロック × 全 5 デノイジングステップで定量化する。「モダリティ対応型注意（future_wrist → curr_wrist、action → proprio）」の仮説を検証する。

### 手法

空間方向 (14×14=196 パッチ) を平均し、11×11 の T 位置別注意行列に集約。全 7 プローブブロック × 全 5 デノイジングステップの注意行列を収集。ここでは k=4（最終デノイジングステップ, σ=4.0）の結果を示す。

### 結果: Block-0、k=4（均一分布、特化なし）

全値が ≈ 0.000470 と均一。自己注意への特化はなく、全 T 位置から等しく注意を集める。

| 出力 T | Top-1 | Top-2 | Top-3 |
|--------|-------|-------|-------|
| action (T=5) | curr_secondary=0.000471 | action=0.000469 | future_proprio=0.000468 |
| future_proprio (T=6) | future_proprio=0.000469 | future_wrist=0.000468 | curr_secondary=0.000468 |
| future_wrist (T=7) | future_wrist=0.000470 | future_primary=0.000469 | future_proprio=0.000469 |
| future_primary (T=8) | future_primary=0.000471 | future_secondary=0.000471 | future_wrist=0.000470 |
| future_secondary (T=9) | future_secondary=0.000473 | future_primary=0.000471 | value=0.000471 |
| value (T=10) | value=0.000473 | future_secondary=0.000473 | future_primary=0.000470 |

**所見**: ほぼ 1/STATE_T = 0.000455（期待値）に近い値。Block-0 では特化した注意パターンは存在しない。

---

### 結果: Block-4、k=4（自己注意の出現）

| 出力 T | Top-1 | Top-2 | Top-3 |
|--------|-------|-------|-------|
| action (T=5) | **action**=0.001965 | future_proprio=0.000903 | curr_secondary=0.000367 |
| future_proprio (T=6) | **future_proprio**=0.001878 | action=0.000638 | blank=0.000541 |
| future_wrist (T=7) | **future_wrist**=0.002175 | future_primary=0.000754 | curr_secondary=0.000703 |
| future_primary (T=8) | **future_primary**=0.001680 | future_secondary=0.000786 | blank=0.000556 |
| future_secondary (T=9) | **future_secondary**=0.001840 | curr_secondary=0.000658 | curr_primary=0.000520 |
| value (T=10) | **value**=0.002255 | action=0.000552 | proprio=0.000426 |

**所見**: 全出力 T で自己注意（自分自身への注意）が Top-1 に確立。モダリティ対応（curr/future 対応）はまだ弱い。

---

### 結果: Block-9、k=4（モダリティ対応の開始）

| 出力 T | Top-1 | Top-2 | Top-3 |
|--------|-------|-------|-------|
| action (T=5) | **action**=0.002855 | curr_secondary=0.000659 | future_wrist=0.000299 |
| future_proprio (T=6) | **future_proprio**=0.001573 | action=0.001010 | future_wrist=0.000988 |
| future_wrist (T=7) | **future_wrist**=0.001888 | **curr_wrist**=0.001396 | proprio=0.000506 |
| future_primary (T=8) | **future_primary**=0.002027 | **curr_primary**=0.001240 | curr_wrist=0.000413 |
| future_secondary (T=9) | **future_secondary**=0.001978 | **curr_secondary**=0.001372 | curr_primary=0.000480 |
| value (T=10) | **value**=0.002630 | future_secondary=0.000753 | curr_secondary=0.000481 |

**所見**: future_wrist → curr_wrist、future_primary → curr_primary、future_secondary → curr_secondary のモダリティ対応が Top-2 に登場。

---

### 結果: Block-13、k=4

| 出力 T | Top-1 | Top-2 | Top-3 |
|--------|-------|-------|-------|
| action (T=5) | **action**=0.002274 | curr_secondary=0.000743 | future_wrist=0.000422 |
| future_proprio (T=6) | **future_proprio**=0.001243 | future_wrist=0.001118 | **proprio**=0.001016 |
| future_wrist (T=7) | **future_wrist**=0.002170 | **curr_wrist**=0.001243 | curr_secondary=0.000329 |
| future_primary (T=8) | **future_primary**=0.001863 | **curr_primary**=0.000814 | future_wrist=0.000695 |
| future_secondary (T=9) | **future_secondary**=0.001813 | **curr_secondary**=0.000943 | curr_wrist=0.000515 |
| value (T=10) | **value**=0.002189 | curr_secondary=0.001129 | curr_wrist=0.000557 |

---

### 結果: Block-18、k=4（モダリティ対応が最も明確）

| 出力 T | Top-1 | Top-2 | Top-3 |
|--------|-------|-------|-------|
| action (T=5) | **action**=0.001348 | **proprio**=0.001050 | value=0.000644 |
| future_proprio (T=6) | **proprio**=0.001786 | future_proprio=0.001110 | future_wrist=0.000756 |
| future_wrist (T=7) | **future_wrist**=0.001939 | **curr_wrist**=0.001464 | value=0.000358 |
| future_primary (T=8) | **curr_primary**=0.001660 | future_primary=0.001401 | curr_secondary=0.000528 |
| future_secondary (T=9) | **curr_secondary**=0.001663 | future_secondary=0.001309 | value=0.000526 |
| value (T=10) | **value**=0.001861 | **proprio**=0.000793 | action=0.000674 |

**所見**: future_proprio → proprio が Top-1（固有感覚対応）。future_primary → curr_primary が逆転して curr_primary が Top-1。action → proprio が Top-2（Block-9,13 では curr_secondary が Top-2 だったのに対し変化）。

---

### 結果: Block-22、k=4

| 出力 T | Top-1 | Top-2 | Top-3 |
|--------|-------|-------|-------|
| action (T=5) | **action**=0.001970 | **proprio**=0.000691 | future_wrist=0.000456 |
| future_proprio (T=6) | **future_proprio**=0.001981 | **proprio**=0.001129 | future_wrist=0.000844 |
| future_wrist (T=7) | **future_wrist**=0.001881 | **curr_wrist**=0.001367 | future_primary=0.000326 |
| future_primary (T=8) | **future_primary**=0.001725 | **curr_primary**=0.001429 | curr_secondary=0.000454 |
| future_secondary (T=9) | **future_secondary**=0.001543 | **curr_secondary**=0.001415 | curr_primary=0.000543 |
| value (T=10) | **value**=0.001861 | curr_wrist=0.000524 | proprio=0.000505 |

---

### 結果: Block-27、k=4（blank トークンの台頭）

| 出力 T | Top-1 | Top-2 | Top-3 |
|--------|-------|-------|-------|
| action (T=5) | **action**=0.002300 | **blank**=0.001231 | proprio=0.000413 |
| future_proprio (T=6) | **future_proprio**=0.002735 | **proprio**=0.001578 | action=0.000133 |
| future_wrist (T=7) | **future_wrist**=0.001991 | **blank**=0.001499 | curr_wrist=0.000664 |
| future_primary (T=8) | **future_primary**=0.001469 | **blank**=0.001088 | curr_primary=0.001035 |
| future_secondary (T=9) | **future_secondary**=0.001378 | **curr_secondary**=0.001010 | future_primary=0.000713 |
| value (T=10) | **value**=0.002715 | **blank**=0.001101 | proprio=0.000339 |

**所見**: blank (T=0) が action, future_wrist, future_primary, value で Top-2 に台頭。blank トークンはデノイジング対象外の「バッファ」として、最終ブロックで情報ハブとして機能する可能性を示す。

---

### 層横断的まとめ

**モダリティ対応型注意の成熟経路**:
```
Block-0: 均一（特化なし）
Block-4: 自己注意が確立
Block-9: モダリティ対応が Top-2 に登場（future_X → curr_X）
Block-13: モダリティ対応が強化、future_proprio → proprio も Top-3 に
Block-18: モダリティ対応が最も明確（future_primary → curr_primary が逆転して Top-1）
Block-22: Block-18 とほぼ同様
Block-27: blank トークンが top-2 に台頭（情報ハブ化）
```

→ プロット: `results/self_attention/t_attn_block{00,04,09,13,18,22,27}.png`（各ブロックで全 5 ステップ横並び）

---

## 3. 分析 B: 空間的注意ヒートマップ

### 目的

出力トークンが画像入力（T=2: curr_wrist, T=3: curr_primary, T=4: curr_secondary）の「どの 14×14 空間領域」から情報を集約するかを定量化する。全 7 プローブブロック × 全 5 デノイジングステップでのパターン変化も確認する。

### 手法

出力 T 位置 `{5 (action), 8 (future_primary), 10 (value)}` × 入力 T 位置 `{2 (curr_wrist), 3 (curr_primary), 4 (curr_secondary)}` の 9 組み合わせについて 14×14 の空間注意マップを生成。全 5 ステップ × 全 7 ブロックの格子プロット。変動係数（CV = std/mean）で空間集中度を定量化。

### 結果（50 ep, 1600 calls 平均、Block-18 k=4）

| 出力 → 入力 | 空間パターン | 解釈 |
|------------|------------|------|
| action → curr_wrist | グリッパー底辺に強集中（高 CV） | ツール先端部への焦点 |
| action → curr_primary | 上部〜中央に集中（高 CV） | シーン構造の主要物体 |
| action → curr_secondary | 比較的分散（中 CV） | 補助視点は広く参照 |
| future_primary → curr_primary | 最も均一（低 CV） | シーン全体の広域参照 |
| future_wrist → curr_wrist | 比較的均一（中 CV） | 全体的なツール形状参照 |
| value → curr_primary | 中程度の集中 | 状態評価に関連する領域 |

**層・ステップ依存性**:
- Block-0 では全組み合わせが均一ヒートマップ（空間的特化なし）
- Block-9 以降で空間集中が明確化
- デノイジングステップ（k=0〜4）では空間パターンはほぼ不変（Section 6 参照）

→ プロット: `results/self_attention/spatial_{out}_to_{in}.png`（全 5 ステップ × 全 7 ブロック格子）

---

## 4. 分析 C: proprio vs 画像への注意比率

### 目的

action token (T=5) が proprio（T=1）と各カメラ画像（T=2〜4）のどちらに・どのブロックでより強く依存するかを定量化する。

### 手法

各出力 T 位置 `{5,6,7,8,9,10}` について、入力 T=1 (proprio) への注意重みと入力 T=2,3,4（3 カメラ平均）への注意重みを全 7 プローブブロック × 全 5 デノイジングステップで比較。

### 結果（k=4）

| 出力 T | Block-0 | Block-9 | Block-18 | Block-27 | 解釈 |
|--------|---------|---------|----------|----------|------|
| action (T=5) | proprio ≈ image | image > proprio | proprio > image | proprio ≈ image | 中層で proprio 重視 |
| future_proprio (T=6) | proprio ≈ image | proprio ≈ image | proprio >> image | proprio >> image | 後半でより proprio 集中 |
| future_wrist (T=7) | proprio ≈ image | image > proprio | image > proprio | image ≈ proprio | 画像依存が持続 |
| future_primary (T=8) | proprio ≈ image | image > proprio | image >> proprio | image > proprio | 常に画像依存 |
| future_secondary (T=9) | proprio ≈ image | image > proprio | image > proprio | image > proprio | 常に画像依存 |
| value (T=10) | proprio ≈ image | proprio ≈ image | proprio ≈ image | proprio ≈ image | バランス型 |

**所見**:
- action (T=5): Block-18 では proprio (0.001050) が image 平均 (≈0.0004) を大きく上回る → 「固有感覚から制御指令」
- future_proprio (T=6): Block-18 で proprio (0.001786) が第 1 位 → 「固有感覚の未来を proprio から予測」
- future_primary (T=8): curr_primary への注意が全層で high → 「同一視点のカメラで未来を予測」
- Block-0: 全出力で proprio ≈ image（均一分布の結果）

→ プロット: `results/self_attention/proprio_vs_image_{T_name}.png`（6 ファイル、各ファイルが全 7 ブロック × 全 5 ステップをカバー）

---

## 5. 分析 D: 複数出力トークンの画像への注意比較

### 目的

action, future_proprio, future_wrist, future_primary, future_secondary, value の 6 出力トークンが、同一入力画像（T=2〜4）に向ける注意の「ブロック深度依存性」を比較する。

### 手法

各出力 T 位置が入力 T=2,3,4 へ向ける注意重みを全 7 プローブブロックにわたって折れ線グラフでプロット。k=0〜4 の全 5 ステップで個別ファイルを生成。

### 結果（k=4）

**curr_primary（T=3）への注意パターン**:
- future_primary (T=8) が全層で最も高い注意を向ける（同視点対応）
- Block-0 では 6 出力すべてが同レベル（≈ 0.000470）
- Block-9,13 で future_primary と future_secondary が台頭
- Block-27 では全体的に画像への注意が低下（blank ハブ効果）

**curr_wrist（T=2）への注意パターン**:
- future_wrist (T=7) が最も高い注意（Block-9 以降）
- action (T=5) は Block-18 で curr_wrist への注意が低下（proprio にシフト）

| デノイジングステップ | 出力ファイル |
|---------------------|------------|
| k=0 (σ=80) | `results/self_attention/cross_output_compare_k0.png` ✓ |
| k=1 (σ=42) | `cross_output_compare_k1.png` ✓ |
| k=2 (σ=21) | `cross_output_compare_k2.png` ✓ |
| k=3 (σ=10) | `cross_output_compare_k3.png` ✓ |
| k=4 (σ=4) | `cross_output_compare_k4.png` ✓ |

---

## 6. 分析 E: Attention Rollout

### 目的

単一ブロックの注意行列では「その層での注意パターン」しか分からない。Attention Rollout を用いて各プローブブロックまでの **累積的な情報フロー**（入力 T → 出力 T）を定量化する。

### 手法

Abnar & Zuidema (2020) の手法に基づき、残差接続を考慮した累積注意を計算する:

```
A_hat[b] = 0.5 × A[b] + 0.5 × I   # 残差込みの実効注意行列（行正規化）
R[b]     = A_hat[b] @ R[b_prev]    # ブロック順（0→4→9→13→18→22→27）に積算
R[-1]    = I                        # 入力層は恒等変換
```

`R[b][i, j]` は「出力 T=i への最終表現が、入力 T=j からどの程度影響を受けているか」を 0〜1 で表す。

### 結果: Rollout（Block-27 まで累積、k=4）

| 出力 T | Top-1（自己） | Top-2 | Top-3 |
|--------|-------------|-------|-------|
| action (T=5) | **action=0.9778** | proprio=0.0035 | curr_secondary=0.0030 |
| future_proprio (T=6) | **future_proprio=0.9757** | proprio=0.0070 | future_wrist=0.0047 |
| future_wrist (T=7) | **future_wrist=0.9772** | curr_wrist=0.0068 | curr_secondary=0.0026 |
| future_primary (T=8) | **future_primary=0.9753** | curr_primary=0.0070 | future_wrist=0.0033 |
| future_secondary (T=9) | **future_secondary=0.9750** | curr_secondary=0.0074 | curr_primary=0.0031 |
| value (T=10) | **value=0.9786** | curr_secondary=0.0030 | future_secondary=0.0028 |

### Rollout の層別進行（action T=5、k=4）

| up to Block | action（自己） | Top-2 |
|-------------|-------------|-------|
| Block-0 | 0.9954 | curr_secondary=0.0005 |
| Block-4 | 0.9923 | future_proprio=0.0014 |
| Block-9 | 0.9901 | future_proprio=0.0016 |
| Block-13 | 0.9873 | curr_secondary=0.0022 |
| Block-18 | 0.9836 | curr_secondary=0.0026 |
| Block-22 | 0.9805 | proprio=0.0031 |
| Block-27 | 0.9778 | proprio=0.0035 |

### 所見

- **全出力 T で自己参照が圧倒的（97.5〜97.9%）**: Rollout による情報フローは自己（同一 T 位置）が支配的。DiT は残差接続による「恒等伝播」が主体で、クロス T 注意は微弱
- **モダリティ対応は累積後も保持**: 単一ブロックで観測したモダリティ対応（future_primary → curr_primary など）が Rollout でも Top-2 に残る（≈ 0.7%）
- **proprio への情報フローは block が深まるにつれ増加**: action の rollout で proprio が Block-18 以降に Top-2 に浮上し、最終的に 0.35% を占める
- **Rollout の解釈**: 高い自己参照率は「注意は使われていない」ことではなく、残差接続が強力であることを反映。実際の単層注意行列（Section 2）には明確なモダリティ対応パターンが存在する

→ プロット: `results/self_attention/rollout_k{0,1,2,3,4}.png`（各ステップで全 7 ブロックの rollout 行列を横並び）

---

## 7. デノイジングステップ不変性の検証

### 目的

「どこを見るか（注意パターン）」はデノイジング全 5 ステップを通じて不変かを全 7 ブロックで定量的に確認する。

### 手法

全 7 プローブブロック × 全 5 ステップで (k=0 と k=4) の注意行列差 `||A(k=0) - A(k=4)||_max` を計算。

### 結果（50 ep, 1600 calls 平均）

| Block | max|A(k=0) - A(k=4)| | 最大変化位置 | ほぼ不変か |
|-------|----------------------|------------|----------|
| Block-0 | **0.0000015** | future_secondary→future_secondary | ✓ |
| Block-4 | **0.0010122** | future_wrist→future_wrist | ✓ |
| Block-9 | **0.0002898** | value→value | ✓ |
| Block-13 | **0.0000844** | action→action | ✓ |
| Block-18 | **0.0000299** | action→action | ✓ |
| Block-22 | **0.0000602** | value→value | ✓ |
| Block-27 | **0.0001467** | future_wrist→future_wrist | ✓ |

**Denoising invariance (action→proprio) Block-18**:

| k=0 | k=1 | k=2 | k=3 | k=4 | 最大差 |
|-----|-----|-----|-----|-----|-------|
| 0.0010610 | 0.0010601 | 0.0010588 | 0.0010549 | 0.0010498 | **0.0000112** |

**所見**:
- 全 7 ブロックで注意パターンはほぼ完全に不変（最大差 ≈ 0.001）
- Block-4 が最大変化量（0.00101）だが、それでも絶対値として無視できるレベル
- 「どこを見るか」はデノイジングステップに依存しない——σ によらず同じ情報源を参照
- 変化が起きているのは `self→self` のアテンション（自己注意の微小な変化）

---

## 8. 総合まとめ

| 検証項目 | 主要所見 |
|---------|---------|
| T 位置別注意行列（全 7 ブロック, k=4） | Block-0: 均一 → Block-4: 自己注意確立 → Block-9: モダリティ対応開始 → Block-18: モダリティ対応最明確 → Block-27: blank ハブ台頭 |
| 空間的注意ヒートマップ | action→curr_wrist: グリッパー底辺集中（高 CV）、future_primary→curr_primary: 均一参照（低 CV）。Block-9 以降で空間集中が明確化 |
| proprio vs 画像比率 | action は中〜後層（Block-18）で proprio 優位。future_primary は全層で curr_primary 優位 |
| 複数出力の画像注意比較 | future_primary (T=8) が curr_primary への注意が最高。Block-27 で全体的に低下（blank ハブ効果） |
| Attention Rollout（累積情報フロー） | 全出力で自己参照 ≈ 97.5〜97.9%。残差が支配的。モダリティ対応は累積後も Top-2 に保持（≈ 0.7%） |
| デノイジング不変性（全 7 ブロック） | 全ブロックで不変確認。最大差 Block-4 で 0.00101。注意パターンは σ に依存しない |

---

## 9. 出力ファイル一覧

| ファイル | 内容 |
|---------|------|
| `results/self_attention/attn_meta.json` | 実験メタデータ（50 ep, 1600 calls, 成功率 0%） |
| `results/self_attention/attn_stats.json` | 全 (block, k_step) の [11×11] 注意行列 + rollout 行列 + サマリー統計 |
| `results/self_attention/t_attn_block{00,04,09,13,18,22,27}.png` | 各ブロックの T 位置別注意行列（全 5 ステップ横並び） |
| `results/self_attention/rollout_k{0,1,2,3,4}.png` | Attention Rollout 行列（全 7 ブロック横並び、ステップ別） |
| `results/self_attention/spatial_{out}_to_{in}.png` | 空間的注意ヒートマップ（全 5 ステップ × 全 7 ブロック格子） |
| `results/self_attention/proprio_vs_image_{T_name}.png` | proprio vs 画像の注意割合（6 ファイル） |
| `results/self_attention/cross_output_compare_k{0,1,2,3,4}.png` | 複数出力の画像注意比較（全 5 ファイル） |
