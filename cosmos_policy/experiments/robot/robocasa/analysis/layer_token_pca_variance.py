"""
layer_token_pca_variance.py — 層ごとの潜在表現(全トークン)のPCA分散説明率解析

feature_analysis.py の PCA (plot_pca_per_layer) は action token (T=5) のみを
H×W で空間平均した (D,) ベクトル1個/callを対象にしていた。本スクリプトは
プーリングを一切行わず、各 probe block の出力テンソル (B, T, H, W, D) の
「全トークン」(T=11 token種別 × H×W=14×14=196 空間位置 = 最大2156トークン/call)
を個別サンプルとして捕捉し、層ごとに PCA を実行して累積寄与率
(cumulative explained variance ratio) を可視化する。
  「いくつの主成分で分散をどのくらい説明できるか」= スクリー・累積寄与率プロット。

トークン種別 (analysis_shared.T_NAMES): blank, proprio, curr_wrist,
curr_primary, curr_secondary, action, future_proprio, future_wrist,
future_primary, future_secondary, value。image系(curr/future primary/
secondary/wrist)は 14×14 の空間パッチ、それ以外は同じ (H,W) 形状に
展開された非空間トークン(モデルアーキテクチャ上、全T位置が同じテンソル
形状で処理されるための形式)。

スコープ・限界 (事前に明記):
  - 最終デノイジングステップ (k = NUM_DENOISE_STEPS-1、収束後の表現) のみを対象。
    全kでの変化は未検証 (future work)。
  - メモリ制約のため、1 call・1層あたり最大 --tokens_per_call 個をランダム
    サブサンプルする (全2156トークンではない)。デフォルト128。
  - 「全トークン」には性質の大きく異なる種別 (画像/proprio/action/value等) が
    混在するため、全トークンPCAの主成分はトークン種別間の分散(モダリティ間の
    分離)に支配されている可能性が高い。種別ごとの寄与率を breakdown プロットで
    別途確認し、この解釈を検証する。
  - PCA は sklearn randomized SVD (n_components = min(--max_components, N-1, D))
    で打ち切って計算する。真の累積寄与率が100%に達する成分数を直接示すもの
    ではなく、打ち切り時点(--max_components)での到達率を明記する。
  - 複数タスクをプールして「Cosmos Policyの表現空間」の一般的傾向として報告する
    (タスク固有の主張ではない)。

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.layer_token_pca_variance \
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
      --config_file cosmos_policy/config/config.py \
      --use_wrist_image True --num_wrist_images 1 \
      --use_proprio True --normalize_proprio True --unnormalize_actions True \
      --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
      --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
      --trained_with_image_aug True \
      --chunk_size 32 --num_open_loop_steps 16 \
      --seed 195 --randomize_seed False --deterministic True \
      --use_variance_scale False --use_jpeg_compression True --flip_images True \
      --num_denoising_steps_action 5 \
      --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
      --data_collection False \
      --tasks "PnPCounterToCab,CloseDrawer,CoffeePressButton,TurnOnStove" \
      --n_episodes_per_task 15 \
      --tokens_per_call 128 \
      --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/layer_token_pca
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS,
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.experiments.robot.robocasa.analysis.feature_analysis import (
    get_action_with_features,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    PROBE_LAYERS,
    PROBE_LAYER_SHORT,
    NUM_DENOISE_STEPS,
    STATE_T,
    T_NAMES,
)

DEFAULT_TASKS = "PnPCounterToCab,CloseDrawer,CoffeePressButton,TurnOnStove"
VARIANCE_TARGETS = [0.50, 0.80, 0.90, 0.95, 0.99]


# ── 全トークン捕捉 ────────────────────────────────────────────────────────────

class AllTokenCapture:
    """
    Forward-hook で各ブロックの出力 (B,T,H,W,D) を、プーリングせず全トークン
    位置を個別サンプルとして (サブサンプル込みで) 蓄積する。
    最終デノイジングステップのみを対象とする。
    """

    def __init__(self, probe_layers: List[int], tokens_per_call: int, target_step: int):
        self.probe_layers = probe_layers
        self.tokens_per_call = tokens_per_call
        self.target_step = target_step
        self._current_step = -1
        self._handles: List = []
        # {layer_idx: [ (n_tok_i, D) numpy chunks ]}
        self.layer_tokens: Dict[int, List[np.ndarray]] = {l: [] for l in probe_layers}
        # {layer_idx: [ (n_tok_i,) T-index chunks ]}
        self.layer_t_idx: Dict[int, List[np.ndarray]] = {l: [] for l in probe_layers}
        self.n_calls_captured = 0

    def register(self, model):
        net = model.net
        for layer_idx in self.probe_layers:
            block = net.blocks[layer_idx]
            h = block.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset_for_policy_call(self):
        self._current_step = -1

    def before_denoise_step(self):
        self._current_step += 1
        if self._current_step == self.target_step:
            self.n_calls_captured += 1

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if self._current_step != self.target_step:
                return
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return
            B, T, H, W, D = output.shape
            x = output[0].reshape(T * H * W, D)
            t_idx = torch.arange(T, device=x.device).repeat_interleave(H * W)
            n_tok = x.shape[0]
            if n_tok > self.tokens_per_call:
                perm = torch.randperm(n_tok, device=x.device)[: self.tokens_per_call]
                x = x[perm]
                t_idx = t_idx[perm]
            self.layer_tokens[layer_idx].append(x.float().detach().cpu().numpy())
            self.layer_t_idx[layer_idx].append(t_idx.detach().cpu().numpy())
        return hook

    def stacked(self, layer_idx: int):
        toks = self.layer_tokens[layer_idx]
        t_idxs = self.layer_t_idx[layer_idx]
        if not toks:
            return np.zeros((0, 1)), np.zeros((0,), dtype=int)
        return np.concatenate(toks, axis=0), np.concatenate(t_idxs, axis=0)


# ── PCA 解析 ──────────────────────────────────────────────────────────────────

def run_pca_variance(X: np.ndarray, max_components: int, pca_max_tokens: int, seed: int = 0):
    """
    X: (N, D)。N が pca_max_tokens を超える場合はランダムサブサンプル。
    Returns: cum_evr (np.ndarray, 累積寄与率), n_used (実際にPCAに使ったサンプル数)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N > pca_max_tokens:
        idx = rng.choice(N, size=pca_max_tokens, replace=False)
        X = X[idx]
        N = pca_max_tokens
    D = X.shape[1]
    n_comp = min(max_components, N - 1, D)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    pca.fit(X)
    cum_evr = np.cumsum(pca.explained_variance_ratio_)
    return cum_evr, N


def components_for_target(cum_evr: np.ndarray, target: float):
    """target 寄与率に到達する最小の主成分数。到達しなければ None を返す。"""
    idx = np.searchsorted(cum_evr, target)
    if idx >= len(cum_evr):
        return None
    return int(idx + 1)


# ── プロット ──────────────────────────────────────────────────────────────────

def plot_cumulative_variance(layer_cum_evr: Dict[int, np.ndarray], probe_layers: List[int],
                              out_dir: Path, max_components: int, tasks: List[str]):
    layer_colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(probe_layers)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Layer-wise PCA of Full-Token Latent Representations (all T×H×W tokens, final denoise step)\n"
        f"Tasks: {', '.join(tasks)}",
        fontsize=11,
    )

    # Panel 0: cumulative explained variance curve per layer
    for li, l in enumerate(probe_layers):
        cum_evr = layer_cum_evr.get(l)
        if cum_evr is None or len(cum_evr) == 0:
            continue
        x = np.arange(1, len(cum_evr) + 1)
        axes[0].plot(x, cum_evr, color=layer_colors[li], linewidth=2,
                     label=PROBE_LAYER_SHORT.get(l, f"Blk-{l}"))
    for target in VARIANCE_TARGETS:
        axes[0].axhline(target, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
        axes[0].text(max_components * 0.99, target, f"{target:.0%}", fontsize=7,
                     ha="right", va="bottom", color="gray")
    axes[0].set_xlabel("Number of principal components")
    axes[0].set_ylabel("Cumulative explained variance ratio")
    axes[0].set_title("Cumulative Variance Explained per Layer")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Panel 1: #components needed to reach 90% variance, per layer (bar chart)
    target = 0.90
    comps = []
    hit_cap = []
    for l in probe_layers:
        cum_evr = layer_cum_evr.get(l)
        if cum_evr is None or len(cum_evr) == 0:
            comps.append(0)
            hit_cap.append(False)
            continue
        n = components_for_target(cum_evr, target)
        if n is None:
            comps.append(len(cum_evr))
            hit_cap.append(True)  # capped: true value >= this bar
        else:
            comps.append(n)
            hit_cap.append(False)
    x = np.arange(len(probe_layers))
    bar_colors = ["tomato" if hc else "royalblue" for hc in hit_cap]
    axes[1].bar(x, comps, color=bar_colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([PROBE_LAYER_SHORT.get(l, f"Blk-{l}") for l in probe_layers], fontsize=8)
    axes[1].set_ylabel(f"# components for {target:.0%} variance")
    axes[1].set_title(f"Components Needed for {target:.0%} Variance per Layer\n"
                       f"(red = did not reach {target:.0%} within cap={max_components}; bar = cap value, true # is larger)")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, (c, hc) in enumerate(zip(comps, hit_cap)):
        label = f"{c}{'+' if hc else ''}"
        axes[1].text(i, c + max(comps) * 0.01, label, ha="center", fontsize=8)

    plt.tight_layout()
    p = out_dir / "layer_pca_cumulative_variance.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")


def plot_token_type_breakdown(capture: AllTokenCapture, probe_layers_for_breakdown: List[int],
                               out_dir: Path, max_components: int, pca_max_tokens: int):
    """
    選択した層について、トークン種別ごとの累積寄与率カーブを「全トークン」カーブと
    重ねて表示する。「全トークンPCAの主成分がモダリティ間分散に支配されているか」
    を直接確認するための補助プロット。
    """
    n = len(probe_layers_for_breakdown)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Token-Type Breakdown of Cumulative Explained Variance\n"
                 "(checks whether 'all tokens' PCA is dominated by cross-modality variance)",
                 fontsize=10)

    t_colors = plt.cm.tab20(np.linspace(0, 1, STATE_T))
    summary = {}

    for ax, l in zip(axes, probe_layers_for_breakdown):
        X_all, t_idx_all = capture.stacked(l)
        if X_all.shape[0] < 20:
            ax.set_title(f"{PROBE_LAYER_SHORT.get(l, f'Blk-{l}')}: insufficient data")
            continue

        cum_evr_all, n_used_all = run_pca_variance(X_all, max_components, pca_max_tokens)
        x = np.arange(1, len(cum_evr_all) + 1)
        ax.plot(x, cum_evr_all, color="black", linewidth=2.5, label=f"all tokens (N={n_used_all})")

        layer_summary = {"all_tokens": {"n_used": int(n_used_all),
                                         "n_components_90pct": components_for_target(cum_evr_all, 0.90)}}

        for t in range(STATE_T):
            mask = t_idx_all == t
            X_t = X_all[mask]
            if X_t.shape[0] < 20:
                continue
            max_comp_t = min(max_components, X_t.shape[0] - 1)
            cum_evr_t, n_used_t = run_pca_variance(X_t, max_comp_t, pca_max_tokens)
            xt = np.arange(1, len(cum_evr_t) + 1)
            ax.plot(xt, cum_evr_t, color=t_colors[t], linewidth=1.2, alpha=0.85,
                    label=T_NAMES.get(t, f"T{t}"))
            layer_summary[T_NAMES.get(t, f"T{t}")] = {
                "n_used": int(n_used_t),
                "n_components_90pct": components_for_target(cum_evr_t, 0.90),
            }

        summary[PROBE_LAYER_SHORT.get(l, f"Blk-{l}")] = layer_summary

        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative explained variance ratio")
        ax.set_title(PROBE_LAYER_SHORT.get(l, f"Blk-{l}"))
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "layer_pca_token_type_breakdown.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

@dataclass
class PCAConfig(PolicyEvalConfig):
    tasks: str = DEFAULT_TASKS
    n_episodes_per_task: int = 15
    tokens_per_call: int = 128
    max_components: int = 512
    pca_max_tokens: int = 30000
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/layer_token_pca"


def main():
    import draccus
    cfg: PCAConfig = draccus.parse(PCAConfig)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in cfg.tasks.split(",") if t.strip()]

    log_message("=== Layer-wise PCA of Full-Token Latent Representations ===")
    log_message(f"Tasks: {tasks}  Episodes/task: {cfg.n_episodes_per_task}")
    log_message(f"Probe layers: {PROBE_LAYERS}")
    log_message(f"tokens_per_call={cfg.tokens_per_call}  max_components={cfg.max_components}"
                f"  pca_max_tokens={cfg.pca_max_tokens}")

    log_message("Loading model...")
    model, cosmos_config = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    log_message("Model loaded.")

    num_blocks = len(model.net.blocks)
    actual_probe = [l for l in PROBE_LAYERS if l < num_blocks]
    log_message(f"Effective probe layers: {actual_probe}")

    target_step = cfg.num_denoising_steps_action - 1
    capture = AllTokenCapture(actual_probe, cfg.tokens_per_call, target_step)
    capture.register(model)
    log_message(f"Forward hooks registered (capturing final step k={target_step} only).")

    episode_counter = 0
    for task_name in tasks:
        cfg.task_name = task_name
        for ep_idx in range(cfg.n_episodes_per_task):
            log_message(f"\n--- Task {task_name}  Episode {ep_idx + 1}/{cfg.n_episodes_per_task} "
                        f"(global {episode_counter}) ---")
            env, _ = create_robocasa_env(cfg, seed=cfg.seed + ep_idx, episode_idx=ep_idx)
            obs = env.reset()

            for _ in range(10):
                dummy = np.zeros(env.action_spec[0].shape)
                obs, _, _, _ = env.step(dummy)

            task_description = env.get_ep_meta().get("lang", task_name)

            from collections import deque
            action_queue = deque()
            success = False
            max_steps = TASK_MAX_STEPS.get(task_name, 500)
            call_idx_in_ep = 0

            for t in range(max_steps):
                observation = prepare_observation(obs, cfg.flip_images)
                if len(action_queue) == 0:
                    try:
                        result = get_action_with_features(
                            cfg=cfg,
                            model=model,
                            dataset_stats=dataset_stats,
                            observation=observation,
                            task_description=task_description,
                            capture=capture,
                            seed=cfg.seed + episode_counter + t,
                            num_denoising_steps=cfg.num_denoising_steps_action,
                        )
                        actions = result["actions"]
                        call_idx_in_ep += 1
                        for i in range(min(cfg.num_open_loop_steps, len(actions))):
                            action_queue.append(actions[i])
                    except Exception as e:
                        log_message(f"  Error at t={t}: {e}")
                        import traceback; traceback.print_exc()
                        break

                if action_queue:
                    action = action_queue.popleft()
                    if action.shape[-1] == 7 and env.action_dim == 12:
                        action = np.concatenate([action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    obs, _, _, _ = env.step(action)
                    if env._check_success():
                        success = True
                        break

            env.close()
            log_message(f"  Result: {'SUCCESS' if success else 'FAIL'}  calls={call_idx_in_ep}")
            episode_counter += 1

    capture.remove()
    log_message(f"\nTotal calls captured at target step: {capture.n_calls_captured}")

    # ── PCA per layer (全トークンプール) ──────────────────────────────────────
    log_message("\nRunning PCA per layer (all tokens pooled)...")
    layer_cum_evr = {}
    layer_stats = {}
    for l in actual_probe:
        X, _ = capture.stacked(l)
        log_message(f"  Layer {l}: {X.shape[0]} tokens x {X.shape[1] if X.ndim==2 else 0} dims")
        if X.shape[0] < 20:
            log_message(f"  Layer {l}: insufficient tokens, skipping.")
            continue
        cum_evr, n_used = run_pca_variance(X, cfg.max_components, cfg.pca_max_tokens, seed=cfg.seed)
        layer_cum_evr[l] = cum_evr
        layer_stats[PROBE_LAYER_SHORT.get(l, f"Blk-{l}")] = {
            "n_tokens_total": int(X.shape[0]),
            "n_tokens_used_for_pca": int(n_used),
            "n_components_for_target": {
                f"{t:.0%}": components_for_target(cum_evr, t) for t in VARIANCE_TARGETS
            },
            "cum_evr_at_cap": float(cum_evr[-1]) if len(cum_evr) else None,
        }

    plot_cumulative_variance(layer_cum_evr, actual_probe, out_dir, cfg.max_components, tasks)

    # ── トークン種別breakdown (最浅層・最深層のみ、計算コスト抑制) ───────────────
    log_message("\nRunning token-type breakdown (shallowest & deepest probe layer)...")
    breakdown_layers = [actual_probe[0], actual_probe[-1]] if len(actual_probe) >= 2 else actual_probe
    breakdown_summary = plot_token_type_breakdown(
        capture, breakdown_layers, out_dir, cfg.max_components, cfg.pca_max_tokens
    )

    # ── Save JSON ──────────────────────────────────────────────────────────────
    stats = {
        "tasks": tasks,
        "n_episodes_per_task": cfg.n_episodes_per_task,
        "total_calls_captured": capture.n_calls_captured,
        "target_denoise_step": target_step,
        "tokens_per_call_cap": cfg.tokens_per_call,
        "max_components_cap": cfg.max_components,
        "pca_max_tokens_cap": cfg.pca_max_tokens,
        "per_layer": layer_stats,
        "token_type_breakdown": breakdown_summary,
    }
    stats_path = out_dir / "layer_token_pca_variance.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log_message(f"\nSaved: {stats_path}")

    log_message("\n=== Summary: components needed for 90% variance ===")
    for name, s in layer_stats.items():
        n90 = s["n_components_for_target"]["90%"]
        log_message(f"  {name}: {n90 if n90 is not None else f'>{cfg.max_components}'} "
                    f"components  (N={s['n_tokens_used_for_pca']})")

    log_message(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
