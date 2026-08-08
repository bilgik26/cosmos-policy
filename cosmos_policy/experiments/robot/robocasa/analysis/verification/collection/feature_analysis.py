"""
Cosmos Policy DiT特徴量収集スクリプト
スキル表現 (DiT 中間特徴量の収集と可視化) の解析

アクションの時間方向に抽象化したスキル表現の検証
  2-1. 中間特徴量の時間軸クラスタリング (Unsupervised Skill Discovery)
       - 浅い層から深い層まで 7 Block 出力を抽出
       - Policy call ごとの特徴量を PCA / t-SNE で可視化
       - タスク進行 (call_idx_in_episode) との対応を分析
  2-2. デノイジングステップ間の特徴量変化量 (層別)
       - 各層で k=0 と k=4 の特徴ベクトルの L2 距離を比較
       - 浅い層 vs 深い層の変化パターン
  2-3. 層間表現類似度 (Linear CKA)
       - 各層ペアの Linear CKA ヒートマップ

実行方法:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.collection.feature_analysis \\
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \\
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \\
      --config_file cosmos_policy/config/config.py \\
      --use_wrist_image True --num_wrist_images 1 \\
      --use_proprio True --normalize_proprio True --unnormalize_actions True \\
      --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \\
      --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \\
      --trained_with_image_aug True \\
      --chunk_size 32 --num_open_loop_steps 16 \\
      --task_name PnPCounterToCab \\
      --num_trials_per_task 10 \\
      --seed 195 --randomize_seed False --deterministic True \\
      --use_variance_scale False --use_jpeg_compression True --flip_images True \\
      --num_denoising_steps_action 5 \\
      --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \\
      --data_collection False \\
      --num_analysis_episodes 10 \\
      --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_features
"""

import json
import os
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    get_t5_embedding_from_cache,
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
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    CHUNK_SIZE,
    ACTION_LATENT_IDX_ROBOCASA,
    PROBE_LAYERS,
    PROBE_LAYER_LABELS,
    NUM_DENOISE_STEPS,
)


# ── Feature Capture ──────────────────────────────────────────────────────────

class FeatureCapture:
    """
    Forward-hook ベースで各ブロックの出力特徴量を収集するクラス。
    block.forward の出力 x_B_T_H_W_D から action token (T=5) を抽出し、
    H×W 空間平均した D 次元ベクトルを保存する。

    構造:
        self.call_records: List[Dict]  ← policy call ごとのレコード
            {
              "episode":   int,
              "call_idx":  int,
              "step_feats": {
                  k (int): {layer_idx (int): np.array (D,)}
              }
            }
    """

    def __init__(self, probe_layers: List[int]):
        self.probe_layers = probe_layers
        self.call_records: List[Dict] = []
        self._current_step = -1       # 現在のデノイジングステップ
        self._current_feats: Dict = {}  # {k: {layer_idx: feat}}
        self._handles: List = []

    # ── Hook 登録 / 解除 ──────────────────────────────────────────────────

    def register(self, model):
        """model.net.blocks の指定層にフックを登録"""
        net = model.net
        for layer_idx in self.probe_layers:
            block = net.blocks[layer_idx]
            h = block.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ── Policy call ライフサイクル ────────────────────────────────────────

    def reset_for_policy_call(self):
        """1 回の policy call の前にリセット"""
        self._current_step = -1
        self._current_feats = {}

    def before_denoise_step(self):
        """x0_fn が呼ばれる直前に呼ぶ（デノイジングステップカウンタ更新）"""
        self._current_step += 1
        self._current_feats.setdefault(self._current_step, {})

    def finalize_policy_call(self, episode_idx: int, call_idx: int):
        """policy call が完了した後に呼び、記録を保存"""
        self.call_records.append({
            "episode": episode_idx,
            "call_idx": call_idx,
            "step_feats": {
                k: dict(lf) for k, lf in self._current_feats.items()
            },
        })

    # ── Hook 本体 ──────────────────────────────────────────────────────────

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if self._current_step < 0:
                return
            k = self._current_step
            # output: (B, T, H, W, D) - B=1 for inference
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return
            B, T, H, W, D = output.shape
            if T <= ACTION_LATENT_IDX_ROBOCASA:
                return
            # Action token at T=5, average over H×W
            feat = output[0, ACTION_LATENT_IDX_ROBOCASA]  # (H, W, D)
            feat_vec = feat.float().mean(dim=(0, 1))       # (D,)
            self._current_feats.setdefault(k, {})[layer_idx] = (
                feat_vec.detach().cpu().numpy()
            )
        return hook


# ── Policy call with Feature Capture ─────────────────────────────────────────

def get_action_with_features(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    capture: FeatureCapture,
    seed: int,
    num_denoising_steps: int,
):
    """特徴量キャプチャフックを有効にしてアクションを生成"""
    capture.reset_for_policy_call()

    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
            return wrapped, extra
        else:
            x0_fn_raw = result
            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
            return wrapped

    model.get_x0_fn_from_batch = patched_get_x0_fn
    try:
        result = get_action(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=observation,
            task_label_or_embedding=task_description,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = original_get_x0_fn

    return result


# ── 解析ユーティリティ ────────────────────────────────────────────────────

def _collect_layer_features(call_records: List[Dict], denoise_step: int, layer_idx: int) -> np.ndarray:
    """
    指定のデノイジングステップと層の特徴ベクトルを (N, D) 配列で返す。
    feat が存在しない call は除外。
    """
    vecs = []
    for rec in call_records:
        step_feats = rec["step_feats"]
        if denoise_step in step_feats and layer_idx in step_feats[denoise_step]:
            vecs.append(step_feats[denoise_step][layer_idx])
    return np.stack(vecs) if vecs else np.zeros((0, 1))


def _get_call_meta(call_records: List[Dict]) -> tuple:
    """episode / call_idx 配列を返す"""
    episodes = np.array([r["episode"] for r in call_records])
    call_idxs = np.array([r["call_idx"] for r in call_records])
    return episodes, call_idxs


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two feature matrices X (N, d1) and Y (N, d2).
    Returns value in [0, 1] where 1 means identical representations.
    """
    N = X.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    K = H @ (X @ X.T) @ H
    L = H @ (Y @ Y.T) @ H
    hsic_xy = np.trace(K @ L)
    hsic_xx = np.linalg.norm(K, "fro")
    hsic_yy = np.linalg.norm(L, "fro")
    denom = hsic_xx * hsic_yy
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ── プロット ──────────────────────────────────────────────────────────────────

def plot_pca_per_layer(
    call_records: List[Dict],
    probe_layers: List[int],
    out_dir: Path,
    task_name: str,
    success_rate: float,
    denoise_step: int = 4,
    use_tsne: bool = True,
):
    """
    各層の特徴量を PCA で 2D 可視化。
    色 = call_idx_in_episode (タスク進行) / マーカー = episode
    全デノイジングステップ (k=0〜4) で呼び出される。
    """
    from sklearn.decomposition import PCA

    episodes, call_idxs = _get_call_meta(call_records)
    n_episodes = int(episodes.max()) + 1 if len(episodes) > 0 else 1

    n_layers = len(probe_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle(
        f"PCA of Block Output Features (k={denoise_step}, action token)\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  "
        f"Color = call index within episode",
        fontsize=11,
    )

    cmap = plt.cm.viridis
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]

    for ax, layer_idx in zip(axes, probe_layers):
        feats = _collect_layer_features(call_records, denoise_step, layer_idx)
        if feats.shape[0] < 5:
            ax.set_title(PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}"))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # PCA to 2D
        pca = PCA(n_components=min(2, feats.shape[1], feats.shape[0] - 1))
        proj = pca.fit_transform(feats)  # (N, 2)
        var_ratio = pca.explained_variance_ratio_

        # Color by call index within episode; marker by episode
        # Re-compute call_idx per-layer (may differ if some calls had no feat)
        valid_call_idxs = []
        valid_episodes = []
        for rec in call_records:
            if (denoise_step in rec["step_feats"]
                    and layer_idx in rec["step_feats"][denoise_step]):
                valid_call_idxs.append(rec["call_idx"])
                valid_episodes.append(rec["episode"])
        valid_call_idxs = np.array(valid_call_idxs)
        valid_episodes = np.array(valid_episodes)

        max_call = max(valid_call_idxs.max(), 1)
        for ep in range(n_episodes):
            mask = valid_episodes == ep
            if not mask.any():
                continue
            sc = ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=valid_call_idxs[mask],
                cmap=cmap,
                vmin=0,
                vmax=max_call,
                marker=markers[ep % len(markers)],
                s=30,
                alpha=0.7,
                label=f"Ep{ep}",
            )

        ax.set_title(
            PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}"),
            fontsize=9,
        )
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontsize=7)
        if proj.shape[1] > 1:
            ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=max(call_idxs.max(), 1) if len(call_idxs) else 1),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, label="call index within episode")

    path = out_dir / f"pca_per_layer_k{denoise_step}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path}")


def plot_tsne_per_layer(
    call_records: List[Dict],
    probe_layers: List[int],
    out_dir: Path,
    task_name: str,
    success_rate: float,
    denoise_step: int = 4,
):
    """t-SNE 可視化 (n>=10 の場合のみ)。全デノイジングステップ (k=0〜4) で呼び出される。"""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        log_message("sklearn not available, skipping t-SNE.")
        return

    n_layers = len(probe_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle(
        f"t-SNE of Block Output Features (k={denoise_step}, action token)\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  "
        f"Color = call index within episode",
        fontsize=11,
    )

    cmap = plt.cm.plasma
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]

    all_max_call = 1
    for rec in call_records:
        all_max_call = max(all_max_call, rec["call_idx"])

    for ax, layer_idx in zip(axes, probe_layers):
        feats = _collect_layer_features(call_records, denoise_step, layer_idx)
        if feats.shape[0] < 10:
            ax.set_title(PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}"))
            ax.text(0.5, 0.5, "Too few data", ha="center", va="center", transform=ax.transAxes)
            continue

        # PCA to 50 before t-SNE
        n_comp = min(50, feats.shape[1], feats.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(feats)

        perplexity = min(30, feats.shape[0] // 3)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
        proj = tsne.fit_transform(reduced)

        valid_call_idxs = []
        valid_episodes = []
        for rec in call_records:
            if (denoise_step in rec["step_feats"]
                    and layer_idx in rec["step_feats"][denoise_step]):
                valid_call_idxs.append(rec["call_idx"])
                valid_episodes.append(rec["episode"])
        valid_call_idxs = np.array(valid_call_idxs)
        valid_episodes = np.array(valid_episodes)
        n_episodes = int(valid_episodes.max()) + 1 if len(valid_episodes) > 0 else 1

        for ep in range(n_episodes):
            mask = valid_episodes == ep
            if not mask.any():
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=valid_call_idxs[mask],
                cmap=cmap,
                vmin=0,
                vmax=all_max_call,
                marker=markers[ep % len(markers)],
                s=30,
                alpha=0.7,
                label=f"Ep{ep}",
            )

        ax.set_title(PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}"), fontsize=9)
        ax.set_xlabel("t-SNE 1", fontsize=7)
        ax.set_ylabel("t-SNE 2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=all_max_call))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, label="call index within episode")

    path = out_dir / f"tsne_per_layer_k{denoise_step}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path}")


def plot_feature_change_by_layer(
    call_records: List[Dict],
    probe_layers: List[int],
    out_dir: Path,
    task_name: str,
    success_rate: float,
):
    """
    各層における k=0 → k=4 の特徴ベクトル L2 変化量を棒グラフで表示。
    層が深くなるほど変化量が大きいかを確認 (Confidence-to-Commitment の層別版)。
    """
    k_first = 0
    k_last = NUM_DENOISE_STEPS - 1

    layer_means = []
    layer_stds = []
    valid_layers = []

    for layer_idx in probe_layers:
        feats_first = _collect_layer_features(call_records, k_first, layer_idx)
        feats_last = _collect_layer_features(call_records, k_last, layer_idx)

        # Align lengths (both should be same N unless some calls had issues)
        n = min(len(feats_first), len(feats_last))
        if n < 2:
            continue
        diffs = np.linalg.norm(feats_last[:n] - feats_first[:n], axis=1)  # (N,)
        layer_means.append(float(diffs.mean()))
        layer_stds.append(float(diffs.std()))
        valid_layers.append(layer_idx)

    if not valid_layers:
        log_message("No data for feature change plot.")
        return {}

    labels = [PROBE_LAYER_LABELS.get(l, f"Block-{l}").replace("\n", " ") for l in valid_layers]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(valid_layers)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(valid_layers)), layer_means, yerr=layer_stds,
                  color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(valid_layers)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean ||feat(k=4) - feat(k=0)||₂", fontsize=10)
    ax.set_title(
        f"Feature Change (k=0→k=4) by Layer Depth\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate values
    for i, (m, s) in enumerate(zip(layer_means, layer_stds)):
        ax.text(i, m + s + 0.01 * max(layer_means), f"{m:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = out_dir / "feature_change_by_layer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path}")

    return {
        "layer_indices": valid_layers,
        "feature_change_mean": layer_means,
        "feature_change_std": layer_stds,
    }


def plot_cka_matrix(
    call_records: List[Dict],
    probe_layers: List[int],
    out_dir: Path,
    task_name: str,
    success_rate: float,
    denoise_step: int = 4,
):
    """
    各層ペアの Linear CKA ヒートマップ。
    高い値 = 類似した表現; 低い値 = 異なる表現。
    全デノイジングステップ (k=0〜4) で呼び出される。
    """
    n = len(probe_layers)
    feats_list = []
    for layer_idx in probe_layers:
        f = _collect_layer_features(call_records, denoise_step, layer_idx)
        feats_list.append(f)

    # Check we have data
    min_n = min(f.shape[0] for f in feats_list)
    if min_n < 3:
        log_message("Not enough data for CKA matrix.")
        return {}

    # Truncate to common length
    feats_list = [f[:min_n] for f in feats_list]

    cka_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_mat[i, j] = linear_cka(feats_list[i], feats_list[j])

    labels = [PROBE_LAYER_LABELS.get(l, f"Block-{l}").replace("\n", " ") for l in probe_layers]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cka_mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Linear CKA")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cka_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if cka_mat[i, j] < 0.7 else "white")

    ax.set_title(
        f"Inter-Layer Linear CKA (k={denoise_step}, action token)\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )
    plt.tight_layout()
    path = out_dir / f"cka_matrix_k{denoise_step}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path}")

    return {"cka_matrix": cka_mat.tolist(), "layer_indices": probe_layers}


def plot_denoising_step_change_per_layer(
    call_records: List[Dict],
    probe_layers: List[int],
    out_dir: Path,
    task_name: str,
    success_rate: float,
):
    """
    各ステップ間 (k-1 → k) の L2 変化量を層別に折れ線グラフで表示。
    浅い層と深い層で変化のパターンが異なるかを分析。
    """
    k_values = list(range(NUM_DENOISE_STEPS))
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(probe_layers)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Feature Change per Denoising Step by Layer Depth\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    layer_step_means = {}
    for layer_idx in probe_layers:
        step_means = []
        step_stds = []
        valid_ks = []
        for k in k_values[1:]:  # k=1..4 (consecutive differences)
            feats_prev = _collect_layer_features(call_records, k - 1, layer_idx)
            feats_curr = _collect_layer_features(call_records, k, layer_idx)
            n = min(len(feats_prev), len(feats_curr))
            if n < 2:
                continue
            diffs = np.linalg.norm(feats_curr[:n] - feats_prev[:n], axis=1)
            step_means.append(float(diffs.mean()))
            step_stds.append(float(diffs.std()))
            valid_ks.append(k)
        layer_step_means[layer_idx] = (valid_ks, step_means, step_stds)

    # Left: absolute change per step
    for (layer_idx, color) in zip(probe_layers, colors):
        ks, means, stds = layer_step_means[layer_idx]
        if not ks:
            continue
        lbl = PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}").replace("\n", " ")
        axes[0].plot(ks, means, "o-", color=color, linewidth=2, label=lbl)
        axes[0].fill_between(
            ks,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.12, color=color,
        )
    axes[0].set_xlabel("Denoising step k", fontsize=10)
    axes[0].set_ylabel("Mean ||feat(k) - feat(k-1)||₂", fontsize=9)
    axes[0].set_title("Absolute Feature Change by Layer", fontsize=10)
    axes[0].set_xticks(k_values[1:])
    axes[0].set_xticklabels([f"k={k-1}→{k}" for k in k_values[1:]])
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Right: normalized (relative to k=0→1 change for each layer)
    for (layer_idx, color) in zip(probe_layers, colors):
        ks, means, stds = layer_step_means[layer_idx]
        if not ks or means[0] < 1e-8:
            continue
        norm_means = [m / means[0] for m in means]
        lbl = PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}").replace("\n", " ")
        axes[1].plot(ks, norm_means, "o-", color=color, linewidth=2, label=lbl)
    axes[1].set_xlabel("Denoising step k", fontsize=10)
    axes[1].set_ylabel("Feature change (normalized to k=0→1)", fontsize=9)
    axes[1].set_title("Normalized Feature Change by Layer", fontsize=10)
    axes[1].set_xticks(k_values[1:])
    axes[1].set_xticklabels([f"k={k-1}→{k}" for k in k_values[1:]])
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "step_change_per_layer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path}")

    return layer_step_means


def save_features_npz(
    call_records: List[Dict],
    probe_layers: List[int],
    out_dir: Path,
    episode_success: Optional[Dict[int, bool]] = None,
    filename: str = "features.npz",
):
    """
    特徴量を npz として保存（再解析用）。
    キー: feat_k{k}_layer{l} → array (N_calls, D)
    メタキー: episode_labels, call_idx_labels → array (N_calls,)
    episode_success が与えられた場合、call毎の成功フラグ (episode_success_per_call,
    call_idx基準でその call が属する episode の成否) と、episode番号→成否の対応表
    (episode_success_index, episode_success_flag) も保存する。
    """
    npz_data = {}
    for k in range(NUM_DENOISE_STEPS):
        for layer_idx in probe_layers:
            feats = _collect_layer_features(call_records, k, layer_idx)
            if feats.shape[0] > 0:
                npz_data[f"feat_k{k}_layer{layer_idx}"] = feats

    episodes = np.array([r["episode"] for r in call_records])
    call_idxs = np.array([r["call_idx"] for r in call_records])
    npz_data["episode_labels"] = episodes
    npz_data["call_idx_labels"] = call_idxs

    if episode_success is not None:
        ep_ids_sorted = np.array(sorted(episode_success.keys()))
        ep_flags_sorted = np.array([bool(episode_success[e]) for e in ep_ids_sorted])
        npz_data["episode_success_index"] = ep_ids_sorted
        npz_data["episode_success_flag"] = ep_flags_sorted
        npz_data["episode_success_per_call"] = np.array(
            [bool(episode_success.get(int(e), False)) for e in episodes]
        )

    path = out_dir / filename
    np.savez(path, **npz_data)
    log_message(f"Saved: {path}")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig(PolicyEvalConfig):
    num_analysis_episodes: int = 50
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_features"
    obj_instance_split: Optional[str] = None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import draccus
    cfg: FeatureConfig = draccus.parse(FeatureConfig)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message("=== Feature Analysis: Skill Representation in DiT Layers ===")
    log_message(f"Task: {cfg.task_name}")
    log_message(f"Episodes: {cfg.num_analysis_episodes}")
    log_message(f"Probe layers: {PROBE_LAYERS}")

    # Load model
    log_message("Loading model...")
    model, cosmos_config = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    log_message("Model loaded.")

    # Verify num_blocks
    num_blocks = len(model.net.blocks)
    log_message(f"model.net.blocks count: {num_blocks}")
    actual_probe = [l for l in PROBE_LAYERS if l < num_blocks]
    log_message(f"Effective probe layers: {actual_probe}")

    # Register hooks
    capture = FeatureCapture(actual_probe)
    capture.register(model)
    log_message("Forward hooks registered.")

    call_records_all: List[Dict] = []
    success_count = 0
    episode_success: Dict[int, bool] = {}
    total_episodes = cfg.num_analysis_episodes

    for ep_idx in range(total_episodes):
        log_message(f"\n--- Episode {ep_idx + 1}/{total_episodes} ---")
        env, _ = create_robocasa_env(cfg, seed=cfg.seed + ep_idx, episode_idx=ep_idx)
        obs = env.reset()

        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)

        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        log_message(f"  Task: {task_description!r}")

        action_queue = deque()
        success = False
        max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
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
                        seed=cfg.seed + ep_idx + t,
                        num_denoising_steps=cfg.num_denoising_steps_action,
                    )
                    # Record features for this policy call
                    capture.finalize_policy_call(
                        episode_idx=ep_idx,
                        call_idx=call_idx_in_ep,
                    )
                    call_records_all.append(capture.call_records[-1])
                    log_message(
                        f"  t={t}: call_idx={call_idx_in_ep} "
                        f"steps_captured={capture._current_step + 1}"
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
        if success:
            success_count += 1
        episode_success[ep_idx] = success
        log_message(f"  Result: {'SUCCESS' if success else 'FAIL'}")

    capture.remove()

    success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
    log_message(f"\n=== Results ===")
    log_message(f"Success: {success_count}/{total_episodes} = {success_rate:.1%}")
    log_message(f"Total policy calls: {len(call_records_all)}")

    # Save raw features
    save_features_npz(call_records_all, actual_probe, out_dir, episode_success=episode_success)

    # Analysis & plots
    log_message("\nRunning analysis...")

    stats = {
        "task": cfg.task_name,
        "success_rate": success_rate,
        "success_count": success_count,
        "total_episodes": total_episodes,
        "total_policy_calls": len(call_records_all),
        "probe_layers": actual_probe,
    }

    # 2-1: PCA visualization — 全デノイジングステップ (k=0〜4) で実行
    for k in range(NUM_DENOISE_STEPS):
        plot_pca_per_layer(call_records_all, actual_probe, out_dir, cfg.task_name, success_rate, denoise_step=k)

    # t-SNE — 全デノイジングステップ (k=0〜4) で実行（低速なのでステップ数が多い場合は注意）
    for k in range(NUM_DENOISE_STEPS):
        plot_tsne_per_layer(call_records_all, actual_probe, out_dir, cfg.task_name, success_rate, denoise_step=k)

    # 2-2: Feature change by layer
    change_stats = plot_feature_change_by_layer(
        call_records_all, actual_probe, out_dir, cfg.task_name, success_rate
    )
    stats.update(change_stats)

    # 2-2 extension: per-step change by layer
    plot_denoising_step_change_per_layer(
        call_records_all, actual_probe, out_dir, cfg.task_name, success_rate
    )

    # 2-3: CKA — 全デノイジングステップ (k=0〜4) で実行
    for k in range(NUM_DENOISE_STEPS):
        cka_stats = plot_cka_matrix(
            call_records_all, actual_probe, out_dir, cfg.task_name, success_rate, denoise_step=k
        )
        if k == NUM_DENOISE_STEPS - 1:
            stats.update(cka_stats)

    # Save stats
    stats_path = out_dir / "feature_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log_message(f"Stats saved: {stats_path}")

    log_message("Feature analysis complete!")
    return stats


if __name__ == "__main__":
    main()
