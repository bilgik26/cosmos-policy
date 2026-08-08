"""
Cosmos Policy DiT 層別解析スクリプト

features.npz に保存された各 DiT 層の中間特徴量を使い、
デノイジング過程の層別解析を実施する。

【解析内容】
  A (層別): 連続ステップ間の特徴量変化量 ||feat(k) - feat(k-1)||₂
            「単調増加」パターンが全層で成立するか？浅い層と深い層で異なるか？
  B (層別): 特徴空間の固有値スペクトル解析
            各ステップで特徴空間の「実効次元数（有効ランク）」がどう変化するか
            ＝デノイジングで表現が収縮（特化）するか拡張（多様）するか
  C (層別): 特徴ノルム ||feat(k)||₂ の層別・ステップ別推移
            深い層ほどノルムが変化するか

加えて以下の補足解析も実施:
  - ステップ間コサイン類似度 (feat(k) · feat(k+1) の変化)
  - 全ステップ結合の "denoising trajectory" in feature space (PCA で 2D 可視化)

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.layer_analysis \\
      --npz_path cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz \\
      --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_layer \\
      --task_name PnPCounterToCab \\
      --success_rate 0.70
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    PROBE_LAYERS,
    PROBE_LAYER_LABELS,
    PROBE_LAYER_SHORT,
    NUM_DENOISE_STEPS,
    pca_numpy,
    effective_rank,
)


# ── データロード ──────────────────────────────────────────────────────────

def load_features(npz_path: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    feats = {}
    for k in range(NUM_DENOISE_STEPS):
        feats[k] = {}
        for l in PROBE_LAYERS:
            key = f"feat_k{k}_layer{l}"
            feats[k][l] = data[key] if key in data else None
    return feats, data["episode_labels"], data["call_idx_labels"]


def feature_stats(F: np.ndarray):
    """F: (N, D)  →  (mean_norm, std_norm, mean_cosine_sim_against_mean)"""
    norms = np.linalg.norm(F, axis=1)  # (N,)
    mean_f = F.mean(axis=0)
    mean_norm = float(norms.mean())
    std_norm = float(norms.std())
    mean_f_norm = np.linalg.norm(mean_f)
    if mean_f_norm < 1e-12:
        mean_cosim = 0.0
    else:
        cosim = (F @ mean_f) / (norms * mean_f_norm + 1e-12)
        mean_cosim = float(cosim.mean())
    return mean_norm, std_norm, mean_cosim


# ── A (層別): 連続ステップ間変化量 ──────────────────────────────────────

def compute_step_deltas(feats: Dict) -> Dict:
    """
    各 (layer, transition k-1→k) の L2 変化量を返す。
    Returns: deltas[layer][k] = (mean, std, array_of_norms)
    """
    deltas = {}
    for l in PROBE_LAYERS:
        deltas[l] = {}
        for k in range(1, NUM_DENOISE_STEPS):
            f_prev = feats[k - 1][l]
            f_curr = feats[k][l]
            if f_prev is None or f_curr is None:
                continue
            n = min(len(f_prev), len(f_curr))
            d = np.linalg.norm(f_curr[:n] - f_prev[:n], axis=1)
            deltas[l][k] = (float(d.mean()), float(d.std()), d)
    return deltas


def plot_layer_change(deltas: Dict, out_dir: Path, task_name: str, success_rate: float):
    """
    A (層別): 2 種のプロット
    (a) 各層ごとの遷移 bar chart（7層 × 4遷移）
    (b) 全層を重ね書きした折れ線グラフ + 正規化版
    """
    n_layers = len(PROBE_LAYERS)
    transitions = list(range(1, NUM_DENOISE_STEPS))  # [1,2,3,4]
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))

    # ── (a) 7-panel bar chart ────────────────────────────────────────
    fig = plt.figure(figsize=(5 * n_layers, 5))
    gs = gridspec.GridSpec(1, n_layers, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_layers)]

    fig.suptitle(
        f"Layer-wise Consecutive Feature Change ||feat(k) - feat(k-1)||₂\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  "
        f"(color = denoising progression)",
        fontsize=11,
    )

    trans_cmap = plt.cm.RdYlGn_r
    trans_colors = trans_cmap(np.linspace(0.1, 0.9, len(transitions)))

    layer_means_dict = {}
    for ax, (l, lc) in zip(axes, zip(PROBE_LAYERS, layer_colors)):
        means, stds = [], []
        for k in transitions:
            if k in deltas[l]:
                m, s, _ = deltas[l][k]
                means.append(m)
                stds.append(s)
            else:
                means.append(0)
                stds.append(0)

        layer_means_dict[l] = means
        bars = ax.bar(range(len(transitions)), means, yerr=stds,
                      color=trans_colors, alpha=0.85, capsize=4, ecolor="gray")
        ax.set_title(
            PROBE_LAYER_LABELS.get(l, f"Block-{l}"), fontsize=8, pad=3
        )
        ax.set_xticks(range(len(transitions)))
        ax.set_xticklabels([f"k={k-1}→{k}" for k in transitions], fontsize=6.5, rotation=30)
        ax.set_ylabel("Mean L2", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        max_idx = int(np.argmax(means))
        ax.text(max_idx, means[max_idx] + stds[max_idx] * 0.5,
                f"↑{means[max_idx]:.1f}", ha="center", fontsize=6.5, color="red")

    path = out_dir / "layer_change_barchart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── (b) 折れ線 + 正規化版 ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Layer-wise Feature Change Pattern\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    for l, lc in zip(PROBE_LAYERS, layer_colors):
        means = layer_means_dict[l]
        stds_list = []
        for k in transitions:
            s = deltas[l][k][1] if k in deltas[l] else 0
            stds_list.append(s)

        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        axes[0].plot(transitions, means, "o-", color=lc, linewidth=2, markersize=5, label=lbl)
        axes[0].fill_between(transitions,
                              [m - s for m, s in zip(means, stds_list)],
                              [m + s for m, s in zip(means, stds_list)],
                              alpha=0.12, color=lc)

        base = means[0] if means[0] > 1e-8 else 1.0
        norm_means = [m / base for m in means]
        axes[1].plot(transitions, norm_means, "o-", color=lc, linewidth=2, markersize=5, label=lbl)

    for i, ax in enumerate(axes):
        ax.set_xlabel("Denoising step k", fontsize=10)
        ax.set_xticks(transitions)
        ax.set_xticklabels([f"k={k-1}→{k}" for k in transitions])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean ||feat(k) - feat(k-1)||₂", fontsize=9)
    axes[0].set_title("Absolute Feature Change by Layer", fontsize=10)
    axes[1].set_ylabel("Normalized change (relative to k=0→1)", fontsize=9)
    axes[1].set_title("Normalized Pattern (base = k=0→1)", fontsize=10)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    path = out_dir / "layer_change_lineplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── (c) Heatmap ─────────────────────────────────────────
    heatmap = np.zeros((n_layers, len(transitions)))
    for li, l in enumerate(PROBE_LAYERS):
        for ti, k in enumerate(transitions):
            if k in deltas[l]:
                heatmap[li, ti] = deltas[l][k][0]

    row_max = heatmap.max(axis=1, keepdims=True) + 1e-12
    heatmap_norm = heatmap / row_max

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Feature Change Heatmap (Layer × Transition)\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    y_labels = [PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS]
    x_labels = [f"k={k-1}→{k}" for k in transitions]

    for ax, data, title, cmap in zip(
        axes,
        [heatmap, heatmap_norm],
        ["Absolute L2 Change", "Row-normalized (pattern)"],
        ["YlOrRd", "RdYlGn_r"],
    ):
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(transitions)))
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n_layers):
            for j in range(len(transitions)):
                ax.text(j, i, f"{data[i,j]:.2f}" if data.max() < 500 else f"{data[i,j]:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if data[i, j] > 0.7 * data.max() else "black")

    plt.tight_layout()
    path = out_dir / "layer_change_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return layer_means_dict


# ── B (層別): 特徴空間の固有値スペクトル ──────────────────────────────

def compute_spectral_stats(feats: Dict) -> Dict:
    """
    各 (layer, step k) で特徴行列 (N, D) の SVD を計算し固有値スペクトルを得る。
    Returns: spec[layer][k] = {"singular_values": array, "eff_rank": float, "top5_var": array}
    """
    spec = {}
    for l in PROBE_LAYERS:
        spec[l] = {}
        for k in range(NUM_DENOISE_STEPS):
            F = feats[k][l]
            if F is None or len(F) < 3:
                continue
            F_c = F - F.mean(axis=0)
            _, S, _ = np.linalg.svd(F_c, full_matrices=False)
            var = S**2 / (F.shape[0] - 1)
            total_var = var.sum() + 1e-12
            evr = var / total_var
            er = effective_rank(S)
            spec[l][k] = {
                "singular_values": S,
                "eff_rank": er,
                "top5_var": evr[:5].tolist(),
                "total_var": float(total_var),
            }
    return spec


def plot_layer_effrank(spec: Dict, out_dir: Path, task_name: str, success_rate: float):
    """
    B (層別):
    (a) 有効ランク (Participation Ratio) の推移 — 各層 × 各ステップ
    (b) 上位 10 固有値の推移 — 全デノイジングステップ (k=0〜4) の比較
    (c) 分散説明率の累積曲線 — 全 k=0〜4 で各層を比較
    """
    n_layers = len(PROBE_LAYERS)
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))
    k_values = list(range(NUM_DENOISE_STEPS))

    # ── (a) 有効ランクの推移 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Layer-wise Feature Space Dimensionality across Denoising Steps\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    eff_rank_matrix = np.zeros((n_layers, NUM_DENOISE_STEPS))

    for l, lc in zip(PROBE_LAYERS, layer_colors):
        er_vals, valid_ks = [], []
        for k in k_values:
            if k in spec[l]:
                er_vals.append(spec[l][k]["eff_rank"])
                eff_rank_matrix[PROBE_LAYERS.index(l), k] = spec[l][k]["eff_rank"]
                valid_ks.append(k)
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        if er_vals:
            axes[0].plot(valid_ks, er_vals, "o-", color=lc, linewidth=2, markersize=5, label=lbl)

    axes[0].set_xlabel("Denoising step k (0 = highest noise)", fontsize=10)
    axes[0].set_ylabel("Effective Rank (Participation Ratio)", fontsize=9)
    axes[0].set_title("Feature Space Dimensionality by Layer", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_values)
    axes[0].set_xticklabels([f"k={k}\n(σ≈{[80,42.3,21.0,9.6,4.0][k]:.0f})" for k in k_values],
                              fontsize=7)

    im = axes[1].imshow(eff_rank_matrix, aspect="auto", cmap="viridis")
    axes[1].set_xticks(k_values)
    axes[1].set_xticklabels([f"k={k}" for k in k_values], fontsize=9)
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels([PROBE_LAYER_SHORT.get(l, f"B-{l}") for l in PROBE_LAYERS], fontsize=9)
    axes[1].set_title("Effective Rank Heatmap (Layer × Step)", fontsize=10)
    plt.colorbar(im, ax=axes[1], label="Effective Rank")
    for i in range(n_layers):
        for j in k_values:
            axes[1].text(j, i, f"{eff_rank_matrix[i,j]:.0f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if eff_rank_matrix[i, j] < eff_rank_matrix.max() * 0.5 else "black")

    plt.tight_layout()
    path = out_dir / "layer_effrank.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── (b) 上位10固有値の比較 ────────────────────────────────────────
    n_top = 10
    step_colors_b = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, NUM_DENOISE_STEPS))
    step_sigma = [80.0, 42.3, 21.0, 9.6, 4.0]
    markers_b = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4.5), sharey=False)
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(
        f"Top-10 Singular Value Spectra (k=0〜4) by Layer\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    for ax, l in zip(axes, PROBE_LAYERS):
        for k in k_values:
            if k not in spec[l]:
                continue
            Sk = spec[l][k]["singular_values"][:n_top]
            ax.plot(range(n_top), Sk, markers_b[k] + "-", color=step_colors_b[k],
                    linewidth=1.5, markersize=4, label=f"k={k}(σ≈{step_sigma[k]:.0f})")
        ax.set_title(PROBE_LAYER_LABELS.get(l, f"Block-{l}"), fontsize=8)
        ax.set_xlabel("Rank", fontsize=7)
        ax.set_ylabel("Singular Value", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    path = out_dir / "layer_spectra.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── (c) 累積分散説明率 ───────────────────────────────────────────
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4.5), sharey=True)
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(
        f"Cumulative Variance Explained (k=0〜4) by Layer\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    for ax, l in zip(axes, PROBE_LAYERS):
        for k in k_values:
            if k not in spec[l]:
                continue
            Sk = spec[l][k]["singular_values"]
            vark = (Sk**2) / ((Sk**2).sum() + 1e-12)
            cumk = np.cumsum(vark)
            max_show = min(50, len(cumk))
            linestyle = "-" if k == 0 else "--" if k == 4 else ":"
            ax.plot(range(max_show), cumk[:max_show], linestyle,
                    color=step_colors_b[k], linewidth=1.5,
                    label=f"k={k}(σ≈{step_sigma[k]:.0f})")
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_title(PROBE_LAYER_LABELS.get(l, f"Block-{l}"), fontsize=8)
        ax.set_xlabel("Number of PCs", fontsize=7)
        ax.set_ylabel("Cumulative variance explained", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    path = out_dir / "layer_cumvar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return spec


# ── C (層別): 特徴ノルム ────────────────────────────────────────────────

def plot_layer_norm(feats: Dict, out_dir: Path, task_name: str, success_rate: float):
    """
    C (層別): 特徴ノルム ||feat(k)||₂ と call 間コサイン類似度の推移
    """
    k_values = list(range(NUM_DENOISE_STEPS))
    n_layers = len(PROBE_LAYERS)
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))

    norm_matrix = np.zeros((n_layers, NUM_DENOISE_STEPS))
    std_matrix = np.zeros((n_layers, NUM_DENOISE_STEPS))
    cosim_matrix = np.zeros((n_layers, NUM_DENOISE_STEPS))

    for li, l in enumerate(PROBE_LAYERS):
        for k in k_values:
            F = feats[k][l]
            if F is None:
                continue
            mn, sn, mc = feature_stats(F)
            norm_matrix[li, k] = mn
            std_matrix[li, k] = sn
            cosim_matrix[li, k] = mc

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Layer-wise Feature Norm & Concentration across Denoising Steps\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    k_labels = [f"k={k}\n(σ≈{[80,42.3,21.0,9.6,4.0][k]:.0f})" for k in k_values]

    for li, (l, lc) in enumerate(zip(PROBE_LAYERS, layer_colors)):
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        axes[0].plot(k_values, norm_matrix[li], "o-", color=lc, linewidth=2, markersize=5, label=lbl)
        axes[0].fill_between(k_values,
                              norm_matrix[li] - std_matrix[li],
                              norm_matrix[li] + std_matrix[li],
                              alpha=0.10, color=lc)

    axes[0].set_xlabel("Denoising step k", fontsize=10)
    axes[0].set_ylabel("Mean ||feat(k)||₂", fontsize=9)
    axes[0].set_title("Feature Norm by Layer", fontsize=10)
    axes[0].set_xticks(k_values)
    axes[0].set_xticklabels(k_labels, fontsize=7)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for li, (l, lc) in enumerate(zip(PROBE_LAYERS, layer_colors)):
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        axes[1].plot(k_values, cosim_matrix[li], "o-", color=lc, linewidth=2, markersize=5, label=lbl)

    axes[1].set_xlabel("Denoising step k", fontsize=10)
    axes[1].set_ylabel("Mean cosine similarity to group mean", fontsize=9)
    axes[1].set_title("Feature Concentration\n(↑ = more similar across calls)", fontsize=10)
    axes[1].set_xticks(k_values)
    axes[1].set_xticklabels(k_labels, fontsize=7)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    im = axes[2].imshow(norm_matrix, aspect="auto", cmap="plasma")
    axes[2].set_xticks(k_values)
    axes[2].set_xticklabels([f"k={k}" for k in k_values], fontsize=9)
    axes[2].set_yticks(range(n_layers))
    axes[2].set_yticklabels([PROBE_LAYER_SHORT.get(l, f"B-{l}") for l in PROBE_LAYERS], fontsize=9)
    axes[2].set_title("Feature Norm Heatmap", fontsize=10)
    plt.colorbar(im, ax=axes[2], label="||feat||₂")
    for i in range(n_layers):
        for j in k_values:
            axes[2].text(j, i, f"{norm_matrix[i, j]:.0f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if norm_matrix[i, j] > norm_matrix.max() * 0.6 else "black")

    plt.tight_layout()
    path = out_dir / "layer_norm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return {"norm_matrix": norm_matrix.tolist(), "cosim_matrix": cosim_matrix.tolist()}


# ── 補足: デノイジング軌跡の特徴空間 PCA 可視化 ──────────────────────

def plot_denoising_trajectory_pca(feats: Dict, out_dir: Path, task_name: str, success_rate: float):
    """
    各層で、全 policy call の全ステップ特徴量を混ぜて PCA し、
    k=0 から k=4 へのデノイジング「軌跡」を特徴空間上に描く。
    """
    n_layers = len(PROBE_LAYERS)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle(
        f"Denoising Trajectory in Feature Space (PCA)\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  "
        f"Color = denoising step k (blue=k=0 → red=k=4)",
        fontsize=11,
    )

    step_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, NUM_DENOISE_STEPS))

    for ax, l in zip(axes, PROBE_LAYERS):
        F_all, step_labels, call_labels = [], [], []
        n_calls = None
        for k in range(NUM_DENOISE_STEPS):
            F = feats[k][l]
            if F is None:
                continue
            F_all.append(F)
            step_labels.extend([k] * len(F))
            call_labels.extend(list(range(len(F))))
            n_calls = len(F)
        if not F_all:
            ax.set_title(PROBE_LAYER_SHORT.get(l, f"B-{l}"))
            continue

        F_stack = np.vstack(F_all)  # (K*N, D)
        step_arr = np.array(step_labels)
        call_arr = np.array(call_labels)

        n_comp = min(2, F_stack.shape[0] - 1, F_stack.shape[1])
        scores, evr, _ = pca_numpy(F_stack, n_components=n_comp)

        n_show = min(30, n_calls or 30)
        show_calls = np.linspace(0, (n_calls or 1) - 1, n_show, dtype=int)

        for ci in show_calls:
            traj_x, traj_y = [], []
            for k in range(NUM_DENOISE_STEPS):
                mask = (step_arr == k) & (call_arr == ci)
                if mask.any():
                    traj_x.append(float(scores[mask, 0]))
                    traj_y.append(float(scores[mask, 1]) if scores.shape[1] > 1 else 0.0)
            if len(traj_x) > 1:
                ax.plot(traj_x, traj_y, "-", color="gray", alpha=0.15, linewidth=0.7, zorder=1)

        for k in range(NUM_DENOISE_STEPS):
            mask = step_arr == k
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1] if scores.shape[1] > 1 else np.zeros(mask.sum()),
                color=step_colors[k],
                s=15,
                alpha=0.5,
                zorder=2,
                label=f"k={k} (σ≈{[80,42.3,21.0,9.6,4.0][k]:.0f})",
            )

        ax.set_title(PROBE_LAYER_LABELS.get(l, f"Block-{l}"), fontsize=8)
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})", fontsize=7)
        if scores.shape[1] > 1:
            ax.set_ylabel(f"PC2 ({evr[1]:.1%})", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
        if l == PROBE_LAYERS[0]:
            ax.legend(fontsize=6, loc="best")

    plt.tight_layout()
    path = out_dir / "denoising_trajectory_pca.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 補足: ステップ間コサイン類似度 ─────────────────────────────────────

def plot_step_cosine_similarity(feats: Dict, out_dir: Path, task_name: str, success_rate: float):
    """
    各層・各遷移 (k-1→k) の特徴ベクトル間コサイン類似度。
    - mean cos_sim(feat(k-1)_i, feat(k)_i) — 同一 call の連続ステップ間
    高い値 = 方向がほぼ変わらない（magnitude の調整のみ）
    低い値 = 方向が大きく変わる（質的な変化）
    """
    k_values = list(range(NUM_DENOISE_STEPS))
    transitions = k_values[1:]
    n_layers = len(PROBE_LAYERS)
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))

    cosim_data = {}
    for l in PROBE_LAYERS:
        cosim_data[l] = {}
        for k in transitions:
            F_prev = feats[k - 1][l]
            F_curr = feats[k][l]
            if F_prev is None or F_curr is None:
                continue
            n = min(len(F_prev), len(F_curr))
            norms_prev = np.linalg.norm(F_prev[:n], axis=1, keepdims=True) + 1e-12
            norms_curr = np.linalg.norm(F_curr[:n], axis=1, keepdims=True) + 1e-12
            cosim = ((F_prev[:n] / norms_prev) * (F_curr[:n] / norms_curr)).sum(axis=1)
            cosim_data[l][k] = (float(cosim.mean()), float(cosim.std()), cosim)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Step-to-Step Cosine Similarity of Features\n"
        f"Task: {task_name}  Success: {success_rate:.1%}\n"
        f"↑ close to 1 = feature direction barely changes (magnitude shift only)",
        fontsize=11,
    )

    cosim_matrix = np.zeros((n_layers, len(transitions)))
    for li, l in enumerate(PROBE_LAYERS):
        vals, stds = [], []
        for ti, k in enumerate(transitions):
            if k in cosim_data[l]:
                m, s, _ = cosim_data[l][k]
                vals.append(m)
                stds.append(s)
                cosim_matrix[li, ti] = m
            else:
                vals.append(0)
                stds.append(0)
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        lc = layer_colors[li]
        axes[0].plot(transitions, vals, "o-", color=lc, linewidth=2, markersize=5, label=lbl)
        axes[0].fill_between(transitions,
                              [m - s for m, s in zip(vals, stds)],
                              [m + s for m, s in zip(vals, stds)],
                              alpha=0.10, color=lc)

    axes[0].set_xlabel("Denoising step k", fontsize=10)
    axes[0].set_ylabel("Mean cosine similarity (same call, k-1 vs k)", fontsize=9)
    axes[0].set_title("Feature Direction Preservation per Step", fontsize=10)
    axes[0].set_xticks(transitions)
    axes[0].set_xticklabels([f"k={k-1}→{k}" for k in transitions])
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=0.7)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    im = axes[1].imshow(cosim_matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axes[1].set_xticks(range(len(transitions)))
    axes[1].set_xticklabels([f"k={k-1}→{k}" for k in transitions], fontsize=9)
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels([PROBE_LAYER_SHORT.get(l, f"B-{l}") for l in PROBE_LAYERS], fontsize=9)
    axes[1].set_title("Cosine Similarity Heatmap", fontsize=10)
    plt.colorbar(im, ax=axes[1], label="Mean cosine similarity")
    for i in range(n_layers):
        for j in range(len(transitions)):
            axes[1].text(j, i, f"{cosim_matrix[i,j]:.3f}",
                         ha="center", va="center", fontsize=7,
                         color="black" if cosim_matrix[i, j] > 0.7 else "white")

    plt.tight_layout()
    path = out_dir / "step_cosine_similarity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return cosim_data


# ── 総合サマリーグラフ ────────────────────────────────────────────────────

def plot_summary(deltas: Dict, spec: Dict, norm_stats: Dict, out_dir: Path,
                 task_name: str, success_rate: float):
    """4 パネルの総合サマリー"""
    n_layers = len(PROBE_LAYERS)
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))
    k_values = list(range(NUM_DENOISE_STEPS))
    transitions = k_values[1:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Layer-wise Denoising Analysis Summary\nTask: {task_name}  Success: {success_rate:.1%}",
        fontsize=13,
    )

    for l, lc in zip(PROBE_LAYERS, layer_colors):
        means = [deltas[l][k][0] if k in deltas[l] else 0 for k in transitions]
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        axes[0, 0].plot(transitions, means, "o-", color=lc, linewidth=2, markersize=5, label=lbl)
    axes[0, 0].set_title("A: Feature Change ||Δfeat||₂ per Layer")
    axes[0, 0].set_xlabel("Denoising step k")
    axes[0, 0].set_ylabel("Mean L2 change")
    axes[0, 0].set_xticks(transitions)
    axes[0, 0].set_xticklabels([f"k={k-1}→{k}" for k in transitions])
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    for li, (l, lc) in enumerate(zip(PROBE_LAYERS, layer_colors)):
        er_vals = [spec[l][k]["eff_rank"] if k in spec[l] else 0 for k in k_values]
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        axes[0, 1].plot(k_values, er_vals, "o-", color=lc, linewidth=2, markersize=5, label=lbl)
    axes[0, 1].set_title("B: Effective Rank (Feature Space Dimensionality)")
    axes[0, 1].set_xlabel("Denoising step k")
    axes[0, 1].set_ylabel("Effective Rank (Participation Ratio)")
    axes[0, 1].set_xticks(k_values)
    axes[0, 1].set_xticklabels([f"k={k}" for k in k_values])
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    norm_mat = np.array(norm_stats["norm_matrix"])
    cosim_mat = np.array(norm_stats["cosim_matrix"])
    for li, (l, lc) in enumerate(zip(PROBE_LAYERS, layer_colors)):
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        axes[1, 0].plot(k_values, norm_mat[li], "o-", color=lc, linewidth=2, markersize=5, label=lbl)
    axes[1, 0].set_title("C: Feature Norm per Layer")
    axes[1, 0].set_xlabel("Denoising step k")
    axes[1, 0].set_ylabel("Mean ||feat||₂")
    axes[1, 0].set_xticks(k_values)
    axes[1, 0].set_xticklabels([f"k={k}" for k in k_values])
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    for li, (l, lc) in enumerate(zip(PROBE_LAYERS, layer_colors)):
        lbl = PROBE_LAYER_SHORT.get(l, f"B-{l}")
        axes[1, 1].plot(k_values, cosim_mat[li], "o-", color=lc, linewidth=2, markersize=5, label=lbl)
    axes[1, 1].set_title("Feature Concentration (cosine sim to mean)")
    axes[1, 1].set_xlabel("Denoising step k")
    axes[1, 1].set_ylabel("Mean cosine similarity to group mean")
    axes[1, 1].set_xticks(k_values)
    axes[1, 1].set_xticklabels([f"k={k}" for k in k_values])
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "layer_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_name", default="PnPCounterToCab")
    parser.add_argument("--success_rate", type=float, default=0.70)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.npz_path}")
    feats, ep_labels, call_labels = load_features(args.npz_path)

    N = len(ep_labels)
    print(f"Calls: {N}, Episodes: {int(ep_labels.max())+1}")
    for k in range(NUM_DENOISE_STEPS):
        for l in PROBE_LAYERS:
            F = feats[k][l]
            if F is not None:
                print(f"  k={k}, layer={l}: {F.shape}")

    print("\nRunning A (layer-wise): feature change...")
    deltas = compute_step_deltas(feats)
    layer_means = plot_layer_change(deltas, out_dir, args.task_name, args.success_rate)

    print("\nRunning B (layer-wise): spectral / effective rank...")
    spec = compute_spectral_stats(feats)
    plot_layer_effrank(spec, out_dir, args.task_name, args.success_rate)

    print("\nRunning C (layer-wise): feature norm...")
    norm_stats = plot_layer_norm(feats, out_dir, args.task_name, args.success_rate)

    print("\nRunning supplementary: denoising trajectory PCA...")
    plot_denoising_trajectory_pca(feats, out_dir, args.task_name, args.success_rate)

    print("\nRunning supplementary: step cosine similarity...")
    cosim_data = plot_step_cosine_similarity(feats, out_dir, args.task_name, args.success_rate)

    print("\nGenerating summary plot...")
    plot_summary(deltas, spec, norm_stats, out_dir, args.task_name, args.success_rate)

    stats = {
        "task": args.task_name,
        "success_rate": args.success_rate,
        "probe_layers": PROBE_LAYERS,
        "total_calls": int(N),
        "layer_change_mean": {
            str(l): {str(k): deltas[l][k][0] for k in deltas[l]}
            for l in PROBE_LAYERS
        },
        "effective_rank": {
            str(l): {str(k): spec[l][k]["eff_rank"] for k in spec[l]}
            for l in PROBE_LAYERS
        },
        "feature_norm": norm_stats,
        "cosine_similarity": {
            str(l): {str(k): cosim_data[l][k][0] if k in cosim_data[l] else None
                     for k in range(1, NUM_DENOISE_STEPS)}
            for l in PROBE_LAYERS
        },
    }

    stats_path = out_dir / "layer_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved: {stats_path}")
    print("Layer-wise analysis complete!")


if __name__ == "__main__":
    main()
