# Cosmos Policy ドキュメント インデックス

**モデル**: Cosmos-Policy-RoboCasa-Predict2-2B (EDM ベース 2B-parameter DiT)  
**タスク**: PnPCounterToCab (RoboCasa) — Pick-and-Place: Counter → Cabinet  
**実行環境**: Singularity CE (`--nv`), GPU 1 (RTX 4090), Python 3.10, uv 管理 venv

---

## ドキュメント一覧

### 環境構築 (`setup/`)

| ドキュメント | 内容 |
|-------------|------|
| [setup_guide.md](setup/setup_guide.md) | 一般セットアップ手順 |
| [setup_guide_new_robocasa.md](setup/setup_guide_new_robocasa.md) | RoboCasa 環境セットアップ（最新版） |
| [armstrong_setup_guide.md](setup/armstrong_setup_guide.md) | Armstrong クラスタ向けセットアップ |

### Policy・学習・評価 (`policy/`)

| ドキュメント | 内容 |
|-------------|------|
| [cosmos-policy_fpo.md](policy/cosmos-policy_fpo.md) | FPO (Flow Policy Optimization) ノート |
| [fpo_robocasa_vs_finetune_comparison.md](policy/fpo_robocasa_vs_finetune_comparison.md) | FPO vs ファインチューン比較 |
| [eval_comment.md](policy/eval_comment.md) | 評価コメント・観察メモ |
| [robocasa_target_datasets.md](policy/robocasa_target_datasets.md) | RoboCasa ターゲットデータセット情報 |

### 検証解析 (`analysis/`)

| ドキュメント | 内容 |
|-------------|------|
| [mechanism_eval.md](analysis/mechanism_eval.md) | 検証計画・手法の提案（オリジナル） |
| [mechanism_eval_results.md](analysis/mechanism_eval_results.md) | 検証結果インデックス・主要発見サマリー |
| [results_01_action_denoising.md](analysis/results_01_action_denoising.md) | **デノイジング過程 (アクション)**: x̂₀ 変化量・FFT・スコアノルム・層別特徴量解析 |
| [results_02_action_features.md](analysis/results_02_action_features.md) | **特徴量解析 (アクション)**: DiT 中間特徴量・線形プロービング・言語クロスアテンション |
| [results_03_image_generation.md](analysis/results_03_image_generation.md) | **画像生成**: 将来画像ラテント解析、アクションとの比較 |
| [results_04_self_attention.md](analysis/results_04_self_attention.md) | **自己注意**: 入力画像・proprio への注意パターン解析 |

---

## 全検証を通じた主要な発見

### 発見 1: Confidence-to-Commitment（確信度から確定へ）
連続デノイジングステップ間の変化量が全ステップで単調増加（k=0→1 < ⋯ < k=3→4）。  
この法則は **アクション・画像・全 7 DiT 層** にわたって例外なく成立する。  
EDM の高 σ 段階では x̂₀ 予測が平均値付近に underdispersed に留まり、  
低 σ 段階で観測信号を強く使った確定的予測へと大きく収束する。

### 発見 2: DiT の 4 フェーズ処理構造
```
Block-0          → Blocks 4–13           → Blocks 18–22       → Block-27
汎用前処理        安定中間表現（収束ゾーン）  高次セマンティック処理   最終確定出力
分散 ≈ 1.3       分散 ≈ 1,800–2,000       分散 ≈ 8,600–14,000  分散 ≈ 212,000
CKA≈1（隣接層）   CKA≈0.98–1.00            CKA≈0.81（内部）     Commitment の座
```

### 発見 3: 3 種の Commitment メカニズム（層深度依存）
| 層グループ | Commitment の型 | 指標 |
|-----------|----------------|------|
| Block 4–13 | 方向変化型（特徴ベクトルが回転して収束）| cos sim k=3→4: ≈ 0.90 |
| Block 18–22 | スケール増大型（同方向のまま大きくなる）| ノルム +12〜26% |
| Block 27 | 収縮型（方向を変えずにノルムが縮む）| ノルム −7%、cos sim ≈ 0.9996 |

### 発見 4: モダリティ対応型自己注意（Modality-Matched Attention）
各出力トークンは対応する入力トークンに最も強く注目する：
- `future_wrist → curr_wrist`、`future_primary → curr_primary`（同一視点参照）
- `action → proprio`（固有感覚から制御指令を生成）
- `value → proprio`（固有感覚から状態価値を評価）

### 発見 5: 言語の意味的処理の層別分業
Block-9: "from"（ソース位置）→ Block-18: "[object] food"（把持対象）→ Block-27: "cabinet + </s>"（目的地）  
言語的注目はデノイジングステップ・スキルフェーズに対して不変（変化 < 1%）。

---

## 実験コード

検証スクリプトは `cosmos_policy/experiments/robot/robocasa/analysis/` に格納。

| スクリプト | 内容 |
|-----------|------|
| `analysis_shared.py` | **共通定数・ユーティリティ**（全スクリプト共有） |
| `mechanism_analysis.py` | アクションのデノイジング解析（x̂₀ 変化量・FFT・スコアノルム） |
| `feature_analysis.py` | DiT 特徴量収集（全 7 層 × 全 5 ステップ） |
| `layer_analysis.py` | 層別解析: 既存特徴量 npz から変化量・有効ランク・ノルムを再解析 |
| `linear_probe.py` | 線形プロービング（クラス均衡ラベル × 層 × ステップ） |
| `crossattn_analysis.py` | 言語クロスアテンション解析（real-token entropy 含む） |
| `feature_plot.py` | 特徴量の可視化ユーティリティ |
| `feature_replot.py` | 既存 features.npz からプロットのみ再実行 |
| `crossattn_replot.py` | 既存 crossattn.npz から全プロット（H_real 含む）を再生成 |
| `attention_replot.py` | 既存 t_matrices.npy から Attention Rollout を計算・プロット |
| `image_analysis.py` | 画像生成解析（ラテント変化量・CKA・線形プロービング） |
| `attention_analysis.py` | 自己注意解析（全モダリティ・Attention Rollout 含む） |

実行スクリプト（リポジトリルートに配置）:

| スクリプト | 対応解析 | 備考 |
|-----------|---------|------|
| `run_mechanism_analysis.sh` | `mechanism_analysis.py` | Singularity + EGL |
| `run_feature_analysis.sh` | `feature_analysis.py` | Singularity + EGL |
| `run_crossattn_analysis.sh` | `crossattn_analysis.py` | Singularity + EGL |
| `run_attention_analysis.sh` | `attention_analysis.py` | Singularity + EGL |
| `run_image_analysis.sh` | `image_analysis.py` | Singularity + EGL |
| `run_offline_analysis.sh` | `layer_analysis.py` + `linear_probe.py` + `dimension_analysis.py` | オフライン（GPU不要） |
| `run_feature_replot.sh` | `feature_replot.py` | オフライン（プロット再実行） |
| `run_all_analysis_host.sh` | 上記全スクリプトを順次実行 | ホスト側マスタースクリプト |

## 結果ディレクトリ

`cosmos_policy/experiments/robot/robocasa/analysis/results/` 以下に格納。

| ディレクトリ | 内容 |
|------------|------|
| `action_denoising/` | アクションのデノイジング解析の PNG・JSON・NPZ |
| `action_layer/` | 層別特徴量解析の PNG・JSON |
| `action_features/` | DiT 特徴量 PCA・CKA・分散の PNG・NPZ |
| `action_probe/` | 線形プロービング精度の PNG・JSON |
| `action_crossattn/` | 言語クロスアテンションの PNG・JSON・NPZ |
| `image_generation/` | 画像生成解析の PNG・JSON |
| `self_attention/` | 自己注意解析の PNG・JSON・NPY |
