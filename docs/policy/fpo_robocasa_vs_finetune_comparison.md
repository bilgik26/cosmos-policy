# FPO 実装比較

## 対象ファイル

| 略称 | パス |
|------|------|
| **FPO-Cosmos** | `cosmos-policy_rl/cosmos_policy/experiments/robot/robocasa/train_fpo_robocasa.py`<br>`cosmos-policy_rl/cosmos_policy/experiments/robot/robocasa/fpo_buffer.py`<br>`cosmos-policy_rl/cosmos_policy/experiments/robot/cosmos_fpo_model.py` |
| **FPO-Manip** | `fpo-control/manipulation_experiments/finetune_online_rl.py` |
| **Cosmos-IL** | `cosmos-policy_rl/cosmos_policy/scripts/train.py`  |

---

## 1. Actor アーキテクチャ

| 項目 | FPO-Cosmos | FPO-Manip | Cosmos-IL |
|------|-----------|-----------|-----------|
| **ベースモデル** | Cosmos DiT 2B (Video2World) | FlowMatchingPolicy | Cosmos DiT 2B (Video2World) |
| **Fine-tune 手法** | `finetune_mode` で選択<br>**`"lora"`**（デフォルト）: LoRA のみ (rank=8, α=16, q/k/v/output_proj) — base frozen<br>**`"full_dit"`**: DiT 全パラメータ更新、VAE frozen | 全パラメータ更新（vision encoder は freeze 可） | 全パラメータ更新 |
| **ノイズスケジュール** | EDM (Karras et al. 2022) | Flow Matching (CFM) t∈[0,1] | EDM |
| **σ サンプリング分布** | `model.sde.sample_t()` を使用<br>HybridEDMSDE: 70% log-normal (p_mean=1.386, p_std=1.2) + 30% uniform [1.0, 85.0] | CFM t を一様サンプル | 同左（HybridEDMSDE） |
| **Denoising ステップ数（推論）** | 5 ステップ | config 依存 | — |
| **Actor への入力** | VAE latent `(B, 16, 11, 28, 28)` | raw image + state → vision encoder → global cond vector | VAE latent |

---

## 2. 価値関数

| 項目 | FPO-Cosmos | FPO-Manip | Cosmos-IL |
|------|-----------|-----------|-----------|
| **方式** | **モデル内蔵の value token (idx 10)** を直接利用。独立したモジュールなし | 独立 Critic MLP: `global_obs_dim → 512 → 256 → 1` | value token (idx 10) として EDM 学習 |
| **V(s) の取得（ロールアウト）** | `generated[:, :, 10, :, :]` を `_latent_frame_to_scalar` でmean pool + unnormalize → [0,1] | `critic(encode_observations(obs))` を毎ステップ | — |
| **更新時の V(s)** | `compute_value_and_future_loss` 内で value token を denoise して scalar 化、GAE returns に MSE | `critic(obs_cond)` を毎ステップ再エンコード、MSE | EDM 損失（dataset の実 returns をターゲット） |
| **value ターゲット形式** | GAE returns (scalar) を `(C=16, H=28, W=28)` に均一 broadcast **← Cosmos-IL と同形式** | MSE の target は returns scalar | スカラーを `(C, H, W)` 全体に broadcast |
| **価値クリッピング** | なし | オプション (`clip_vloss`) | なし |

---

## 3. Optimizer

| 項目 | FPO-Cosmos | FPO-Manip | Cosmos-IL |
|------|-----------|-----------|-----------|
| **Optimizer 数** | **1つ**（trainable params のみ） | **2つ**（actor / critic 独立） | 1つ（全パラメータ） |
| **lr** | `lr_lora=1e-5`（LoRA・full_dit 共用） | actor: 1e-5 / critic: 1e-4 | 1e-4（config 依存） |
| **weight_decay** | 0.0 | 1e-6 | config 依存 |
| **勾配クリップ** | trainable params を 1 つの `clip_grad_norm` (max=1.0) | actor / critic を独立に clip | あり |

---

## 4. LR スケジューラ

| 項目 | FPO-Cosmos | FPO-Manip | Cosmos-IL |
|------|-----------|-----------|-----------|
| **実装** | `diffusers.get_scheduler` | 同左 | Imaginaire `LambdaWarmUpCosineScheduler` |
| **デフォルト種別** | `"constant"` (warmup 後一定) | `"constant"` | 2サイクル cosine 減衰 |
| **Warmup** | 5 iter 線形 | actor: 5 iter / critic: 1 iter | 1,000 ステップ線形 |
| **減衰** | なし | なし | サイクル1: cosine (peak→30%) → サイクル2: 6% 固定 |
| **ステップ単位** | イテレーション毎 | イテレーション毎 | gradient step 毎 |

---

## 5. ロールアウト収集

| 項目 | FPO-Cosmos | FPO-Manip |
|------|-----------|-----------|
| **チャンク生成** | `generate_samples_from_batch()` で EDM を 5 ステップ denoise | FlowMatching でサンプリング |
| **open-loop ステップ数** | n_open_loop=16 | n_action_steps（config 依存） |
| **V(s) の更新タイミング** | チャンク境界のみ（同一 cond_latent を 16 ステップ再利用） | **毎ステップ** `curr_obs` を encode_observations で再エンコード |
| **将来観測の保存** | チャンク境界 s の実観測 = step s+n_open_loop の `cond_latent_new` を retroactive に格納（リセット時は無効） | なし |
| **FPO 用 CFM データ保存** | チャンク境界のみ (`old_cfm_loss`, `sigmas`, `epsilons`, `x0_latent`, `cond_latent`) | チャンクごとに `cfm_losses` 等を保存 |

---

## 6. GAE / アドバンテージ

| 項目 | FPO-Cosmos | FPO-Manip |
|------|-----------|-----------|
| **V(s) の精度** | チャンク内（16ステップ）は `V(s_t) = V(s_0)` で固定。TD bootstrap がチャンク内で消失し、**16ステップ Monte Carlo + チャンク境界 bootstrap** になる | 毎ステップ真の `V(s_t)` → 正規の GAE |
| **アドバンテージ正規化** | 全バッファでグローバル正規化 | ミニバッチ単位（DDP 時はグローバル） |
| **γ / λ** | 0.99 / 0.95 | 0.99 / 0.95 |

---

## 7. 損失関数の構成

### FPO-Cosmos

```
loss = pg_loss  +  vf_coef × vf_loss  +  aux_coef × aux_loss
                    (0.5)                    (1.0)
```

| 損失項 | 対象トークン | σ | forward pass 数 | 説明 |
|--------|------------|---|----------------|------|
| `pg_loss` (FPO++) | idx 5 (action) | **N=16 個の独立 σ** | N 回 | PPO/SPO/ASPO clip で policy 更新 |
| `vf_loss` | idx 10 (value) scalar | — | 0 (aux 内で共有) | `MSE(v_pred, GAE_returns)` |
| `aux_loss` | idx 10 + idx 6-9 | **共有 σ 1 個** | **1 回** | EDM 再構成 (value + future) の平均 |

`vf_loss` と `aux_loss` は **`compute_value_and_future_loss` 内の 1 回の `model.denoise()` 呼び出し**で同時計算。

- **σ 分布**: `model.sde.sample_t()` → HybridEDMSDE（Cosmos-IL と同一）
- **value target 形式**: `GAE_return` を `(C=16, H=28, W=28)` に均一 broadcast → Cosmos-IL の injection と同形式
- **future target**: 実際の環境観測（step s+n_open_loop の `cond_latent_new`）

### FPO-Manip

```
loss = pg_loss + vf_coef × vf_loss    (vf_coef=1.0)
```

| 損失項 | 説明 |
|--------|------|
| `pg_loss` | PPO/SPO/ASPO clip |
| `vf_loss` | `0.5 × MSE(newvalue, returns)` — 全ステップ、done 後マスクあり |

### Cosmos-IL（模倣学習）

```
loss = EDM_weighted_MSE( 全 prediction トークン )  ← 全トークン共有 σ 1 個で 1 forward pass
     = Σ  w(σ) × MSE(x0_pred_token, x0_target_token)
        token ∈ {action=5, future_proprio=6, ..., future_right=9, value=10}
```

target はデータセットの実観測（chunk_size=32 ステップ先）。value target はスカラーを `(C, H, W)` に broadcast。

---

## 8. FPO Policy Gradient の詳細

| 項目 | FPO-Cosmos | FPO-Manip |
|------|-----------|-----------|
| **確率比の計算** | `ratio = exp(L_old - L_new)` — EDM 損失差を log 確率比の代理に使用 | 同じ |
| **サンプル数 N** | **N=16** (σ, ε) ペアで Monte Carlo 積分を近似 | N=16 |
| **N が必要な理由** | 連続拡散モデルでは log π(a\|s) を解析的に計算不可 → σ 全体の積分を Monte Carlo で近似するため | 同じ |
| **更新対象ステップ** | `has_future_cond=True` のチャンク境界のみ | `do_chunk_level_ppo=True` でチャンク先頭のみ |
| **Trust region** | PPO / SPO / ASPO (`clip_coef=0.01`) | 同じ |

---

## 9. Eval ループ

| 項目 | FPO-Cosmos | FPO-Manip | Cosmos-IL |
|------|-----------|-----------|-----------|
| **実装** | `run_eval()` — 同一環境を再利用 | `eval_all_ranks()` + `_run_rollouts()` | なし |
| **実行タイミング** | `eval_rollout_freq` iter 毎（デフォルト 10） | `rollout_freq` iter 毎（デフォルト 10） | — |
| **エピソード数** | `eval_num_episodes`（デフォルト 10） | `eval_num_episodes`（デフォルト 10） | — |
| **eval 後の状態処理** | `envs.reset()` + `policy.reset_buffers()` | 独立した `eval_actor`（deepcopy）を使用 | — |
| **DDP 対応** | なし | あり（全ランクで実行し集約） | — |

---

## 10. ロギング

| 項目 | FPO-Cosmos | FPO-Manip | Cosmos-IL |
|------|-----------|-----------|-----------|
| **ツール** | **W&B** | W&B | 内部ロギング |
| **主要メトリクス** | `losses/policy`, `losses/value`, `losses/aux`, `fpo/ratio`, `train/lr`, `eval/success_rate` | `losses/policy_loss`, `values/values`, `eval/success_rate_zero_sampling` | action/future state/value の L1 損失 |

---

## 11. その他の実装差異（FPO-Cosmos vs FPO-Manip）

| 項目 | FPO-Cosmos | FPO-Manip |
|------|-----------|-----------|
| **分散学習** | 未対応 | DDP 対応 |
| **チェックポイント** | `trainable_state_dict`（LoRA・full_dit 共用キー）+ optimizer を `.pt` | `save_pretrained` (safetensors) + `ppo_state.pt` |
| **EMA** | なし | あり（オプション） |
| **Gradient Accumulation** | なし | あり |
| **エピソードリセット** | イテレーション跨ぎで継続（eval 後のみリセット） | `reset_every_iteration=True` でイテレーション頭にリセット |

---

## 12. Cosmos-IL（模倣学習）との主な差異

| 項目 | FPO-Cosmos (RL) | Cosmos-IL (模倣学習) |
|------|----------------|---------------------|
| **学習シグナル** | 環境報酬 (RL) | データセットの実演（模倣） |
| **action の損失** | FPO++ policy gradient (PPO clip on ratio) | EDM weighted MSE — 再構成損失 |
| **future の target** | 実行後に環境から取得した観測（n_open_loop=16 ステップ先） | データセット内の将来観測（chunk_size=32 ステップ先） |
| **value の target 出所** | GAE returns（RL 報酬から計算） | データセットの precomputed returns |
| **value target 形式** | スカラーを `(C, H, W)` に均一 broadcast **← 同じ** | 同左 |
| **σ の共有範囲** | action は独立 N 個; value + future は共有 1 個（2 forward pass 合計） | **全トークン共有 1 個**（1 forward pass） |
| **Fine-tune 対象** | `finetune_mode="lora"`（デフォルト）or `"full_dit"`（VAE frozen、DiT 全パラメータ） | 全パラメータ |
| **分散学習** | 未対応 | DDP (torchrun `--nproc_per_node=8`) |
| **LR スケジューラ** | constant + short warmup | 2サイクル cosine 減衰 |

---

## 13. FPO-Cosmos コード構造

リファクタリングにより `train_fpo_robocasa.py` 内の `train()` 関数（358行）を小関数に分割。

| ファイル | 内容 |
|---------|------|
| `train_fpo_robocasa.py` | `TrainConfig`, `_EvalCfg`, GAE/FPO 損失関数, `run_eval`, checkpoint, 学習サブ関数群, `train()` オーケストレーター |
| `fpo_buffer.py` | `RolloutBuffer` クラス（rollout データの蓄積・mini-batch 生成） |
| `cosmos_fpo_model.py` | `CosmosFPOPolicy`, LoRA/full_dit 適用関数, EDM ヘルパー群 |

`train_fpo_robocasa.py` 内の主なサブ関数：

| 関数 | 役割 |
|------|------|
| `_setup_model_and_policy` | モデルロード・fine-tune 設定・policy 構築 |
| `_setup_optimizer_and_scheduler` | AdamW + LR スケジューラ生成 |
| `_collect_rollout` | 1 イテレーション分の環境ステップ収集 |
| `_compute_normalized_gae` | GAE + アドバンテージ正規化 |
| `_fpo_update` | FPO++ 更新ループ（epochs × mini-batches） |
| `_log_iteration` | W&B ログ + コンソール出力 |

`cosmos_fpo_model.py` に追加されたヘルパー関数：

| 関数 | 役割 | 使用箇所 |
|------|------|---------|
| `_build_sigma_B_T` | per-frame sigma テンソル構築 | `_edm_loss_on_action_token`, `compute_value_and_future_loss` |
| `_latent_frame_to_scalar` | latent frame → [0,1] スカラー | `select_action`, `compute_value_and_future_loss` |

---

## 14. 共通設計

| 項目 | 共通内容 |
|------|---------|
| **EDM loss weight** | `w(σ) = (σ² + σ_data²) / (σ·σ_data)²` |
| **σ 分布** | HybridEDMSDE (`model.sde.sample_t()`) — FPO-Cosmos と Cosmos-IL で共通 |
| **FPO ratio** | `ratio = exp(L_old - L_new)` |
| **Trust region の選択肢** | ppo / spo / aspo の 3 モード |
| **clip_coef** | 0.01（拡散モデルの ratio は小さく変動するため通常 PPO より小さい） |
| **アクションチャンク** | チャンク先頭の観測から全アクションを生成し n_open_loop ステップを実行 |
| **CFM/EDM サンプル数 N=16** | 1 観測に対して 16 個の (σ, ε) をサンプルして確率比の期待値を取る |
| **GAE パラメータ** | γ=0.99, λ=0.95 |
| **W&B ロギング** | FPO-Cosmos・FPO-Manip で共通 |
