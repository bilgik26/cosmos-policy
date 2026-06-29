"""
Theme2 特徴量の解析・可視化スクリプト (sklearn 不要版)
theme2_features.npz を読み込んで numpy のみで PCA・CKA・変化量分析を行う

実行:
  python cosmos_policy/experiments/robot/robocasa/theme2_plot.py \
      --npz_path cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/theme2_features.npz \
      --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_features \
      --task_name PnPCounterToCab \
      --success_rate 0.70
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    PROBE_LAYERS,
    PROBE_LAYER_LABELS,
    NUM_DENOISE_STEPS,
    pca_numpy,
    linear_cka,
)


# ── データロード ──────────────────────────────────────────────────────────

def load_features(npz_path: str, probe_layers: List[int], num_steps: int):
    """
    npz から特徴量を読み込み
    Returns: feats[k][layer] = np.array (N, D)
             episode_labels: (N,)
             call_idx_labels: (N,)
    """
    data = np.load(npz_path)
    episode_labels = data["episode_labels"]   # (N,)
    call_idx_labels = data["call_idx_labels"] # (N,)

    feats = {}
    for k in range(num_steps):
        feats[k] = {}
        for layer in probe_layers:
            key = f"feat_k{k}_layer{layer}"
            if key in data:
                feats[k][layer] = data[key]  # (N, D)
            else:
                feats[k][layer] = None
    return feats, episode_labels, call_idx_labels


# ── Plot 1: PCA per layer (k=4) ──────────────────────────────────────────

def plot_pca_per_layer(feats, episode_labels, call_idx_labels, probe_layers,
                       out_dir, task_name, success_rate, denoise_step=4):
    n_layers = len(probe_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle(
        f"Theme 2-1: PCA of Block Output Features (k={denoise_step}, action token)\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  "
        f"Color = call index within episode",
        fontsize=11,
    )

    cmap = plt.cm.viridis
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    n_episodes = int(episode_labels.max()) + 1

    max_call_idx = max(call_idx_labels.max(), 1)

    for ax, layer_idx in zip(axes, probe_layers):
        f = feats[denoise_step].get(layer_idx)
        if f is None or len(f) < 5:
            ax.set_title(PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}"))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_comp = min(2, f.shape[0] - 1)
        scores, evr, _ = pca_numpy(f, n_components=n_comp)

        for ep in range(n_episodes):
            mask = episode_labels == ep
            if not mask.any():
                continue
            sc = ax.scatter(
                scores[mask, 0],
                scores[mask, 1] if scores.shape[1] > 1 else np.zeros(mask.sum()),
                c=call_idx_labels[mask],
                cmap=cmap,
                vmin=0,
                vmax=max_call_idx,
                marker=markers[ep % len(markers)],
                s=35,
                alpha=0.75,
                label=f"Ep{ep}",
            )

        ax.set_title(PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}"), fontsize=9)
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})", fontsize=7)
        if n_comp > 1:
            ax.set_ylabel(f"PC2 ({evr[1]:.1%})", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_call_idx))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, label="call index within episode")

    path = out_dir / f"theme2_pca_k{denoise_step}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: Feature change k=0→k=4 by layer ─────────────────────────────

def plot_feature_change_by_layer(feats, probe_layers, out_dir, task_name, success_rate):
    k_first = 0
    k_last = NUM_DENOISE_STEPS - 1

    layer_means, layer_stds, valid_layers = [], [], []

    for layer_idx in probe_layers:
        f0 = feats[k_first].get(layer_idx)
        f4 = feats[k_last].get(layer_idx)
        if f0 is None or f4 is None:
            continue
        n = min(len(f0), len(f4))
        if n < 2:
            continue
        diffs = np.linalg.norm(f4[:n] - f0[:n], axis=1)
        layer_means.append(float(diffs.mean()))
        layer_stds.append(float(diffs.std()))
        valid_layers.append(layer_idx)

    if not valid_layers:
        print("No data for feature change plot.")
        return {}

    labels = [PROBE_LAYER_LABELS.get(l, f"B-{l}").replace("\n", " ") for l in valid_layers]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(valid_layers)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(valid_layers)), layer_means, yerr=layer_stds,
           color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(valid_layers)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean ||feat(k=4) - feat(k=0)||₂", fontsize=10)
    ax.set_title(
        f"Theme 2-2: Feature Change (k=0→k=4) by Layer Depth\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="y")
    for i, (m, s) in enumerate(zip(layer_means, layer_stds)):
        ax.text(i, m + s + 0.005 * max(layer_means), f"{m:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = out_dir / "theme2_feature_change_by_layer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return {"layer_indices": valid_layers, "feature_change_mean": layer_means, "feature_change_std": layer_stds}


# ── Plot 3: CKA matrix ──────────────────────────────────────────────────

def plot_cka_matrix(feats, probe_layers, out_dir, task_name, success_rate, denoise_step=4):
    n = len(probe_layers)
    feats_list = []
    for layer_idx in probe_layers:
        f = feats[denoise_step].get(layer_idx)
        if f is None:
            f = np.zeros((1, 1))
        feats_list.append(f)

    min_n = min(f.shape[0] for f in feats_list)
    if min_n < 3:
        print("Not enough data for CKA matrix.")
        return {}

    feats_list = [f[:min_n] for f in feats_list]
    cka_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_mat[i, j] = linear_cka(feats_list[i], feats_list[j])

    labels = [PROBE_LAYER_LABELS.get(l, f"B-{l}").replace("\n", " ") for l in probe_layers]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cka_mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Linear CKA")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cka_mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if cka_mat[i, j] < 0.7 else "white")

    ax.set_title(
        f"Theme 2-3: Inter-Layer Linear CKA (k={denoise_step}, action token)\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )
    plt.tight_layout()
    path = out_dir / "theme2_cka_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return {"cka_matrix": cka_mat.tolist(), "layer_indices": probe_layers}


# ── Plot 4: Denoising step change per layer ──────────────────────────────

def plot_step_change_per_layer(feats, probe_layers, out_dir, task_name, success_rate):
    k_values = list(range(NUM_DENOISE_STEPS))
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(probe_layers)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Theme 2-2 Extension: Feature Change per Denoising Step by Layer Depth\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    layer_results = {}
    for layer_idx, color in zip(probe_layers, colors):
        ks, means, stds = [], [], []
        for k in k_values[1:]:
            f_prev = feats[k - 1].get(layer_idx)
            f_curr = feats[k].get(layer_idx)
            if f_prev is None or f_curr is None:
                continue
            n = min(len(f_prev), len(f_curr))
            if n < 2:
                continue
            diffs = np.linalg.norm(f_curr[:n] - f_prev[:n], axis=1)
            ks.append(k)
            means.append(float(diffs.mean()))
            stds.append(float(diffs.std()))
        layer_results[layer_idx] = (ks, means, stds)

        if not ks:
            continue
        lbl = PROBE_LAYER_LABELS.get(layer_idx, f"B-{layer_idx}").replace("\n", " ")
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

    # Normalized
    for layer_idx, color in zip(probe_layers, colors):
        ks, means, _ = layer_results.get(layer_idx, ([], [], []))
        if not ks or means[0] < 1e-8:
            continue
        norm_means = [m / means[0] for m in means]
        lbl = PROBE_LAYER_LABELS.get(layer_idx, f"B-{layer_idx}").replace("\n", " ")
        axes[1].plot(ks, norm_means, "o-", color=color, linewidth=2, label=lbl)

    axes[1].set_xlabel("Denoising step k", fontsize=10)
    axes[1].set_ylabel("Normalized change (rel. to k=0→1)", fontsize=9)
    axes[1].set_title("Normalized Feature Change by Layer", fontsize=10)
    axes[1].set_xticks(k_values[1:])
    axes[1].set_xticklabels([f"k={k-1}→{k}" for k in k_values[1:]])
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "theme2_step_change_per_layer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return layer_results


# ── Plot 5: PC1 progression over episode time ────────────────────────────

def plot_pc1_time_series(feats, episode_labels, call_idx_labels, probe_layers,
                          out_dir, task_name, success_rate, denoise_step=4):
    """
    各層の PC1 の値を policy call の時間順でプロット。
    タスクの進行に応じて PC1 が単調変化するかを確認。
    """
    n_layers = len(probe_layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 2.5 * n_layers), sharex=False)
    if n_layers == 1:
        axes = [axes]

    fig.suptitle(
        f"Theme 2-1: PC1 Time Series by Layer (k={denoise_step})\n"
        f"Task: {task_name}  X = call index in episode (task progression)",
        fontsize=11,
    )

    n_episodes = int(episode_labels.max()) + 1
    ep_colors = plt.cm.tab10(np.linspace(0, 1, n_episodes))

    for ax, layer_idx in zip(axes, probe_layers):
        f = feats[denoise_step].get(layer_idx)
        if f is None or len(f) < 5:
            ax.set_title(PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}"))
            continue

        scores, evr, _ = pca_numpy(f, n_components=1)
        pc1 = scores[:, 0]

        for ep in range(n_episodes):
            mask = episode_labels == ep
            if not mask.any():
                continue
            ep_calls = call_idx_labels[mask]
            ep_pc1 = pc1[mask]
            # Sort by call_idx
            sort_idx = np.argsort(ep_calls)
            ax.plot(ep_calls[sort_idx], ep_pc1[sort_idx],
                    "o-", color=ep_colors[ep], alpha=0.7, linewidth=1.5,
                    markersize=4, label=f"Ep{ep}")

        ax.set_title(
            PROBE_LAYER_LABELS.get(layer_idx, f"Block-{layer_idx}").replace("\n", " ")
            + f"  (PC1 var={evr[0]:.1%})",
            fontsize=9,
        )
        ax.set_xlabel("Call index within episode", fontsize=7)
        ax.set_ylabel("PC1", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right", ncol=5)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = out_dir / f"theme2_pc1_timeseries_k{denoise_step}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 6: Feature variance across layers at k=0 vs k=4 ─────────────────

def plot_feature_variance_by_layer(feats, probe_layers, out_dir, task_name, success_rate):
    """
    各層の特徴量の分散（トレース）を k=0 と k=4 で比較。
    深い層ほど分散が大きい（多様な表現）かを確認。
    """
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Theme 2-1 Extension: Feature Variance by Layer\n"
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=11,
    )

    for k_idx, (denoise_step, label, color) in enumerate(
        [(0, "k=0 (σ=80)", "royalblue"), (4, "k=4 (σ=4)", "tomato")]
    ):
        variances = []
        valid_layers = []
        for layer_idx in probe_layers:
            f = feats[denoise_step].get(layer_idx)
            if f is None or len(f) < 2:
                continue
            # Total variance = sum of per-dim variances
            var = float(np.var(f, axis=0).sum())
            variances.append(var)
            valid_layers.append(layer_idx)

        results[f"k{denoise_step}"] = {"layers": valid_layers, "variances": variances}

        layer_labels = [
            PROBE_LAYER_LABELS.get(l, f"B-{l}").replace("\n", " ")
            for l in valid_layers
        ]
        axes[0].plot(range(len(valid_layers)), variances, "o-", color=color,
                     linewidth=2, markersize=6, label=label)

    axes[0].set_xlabel("Layer (shallow → deep)", fontsize=9)
    axes[0].set_ylabel("Feature variance (Σ per-dim var)", fontsize=9)
    axes[0].set_title("Total Feature Variance by Layer", fontsize=10)
    valid_labels = [
        PROBE_LAYER_LABELS.get(l, f"B-{l}").replace("\n", " ")
        for l in probe_layers
    ]
    axes[0].set_xticks(range(len(probe_layers)))
    axes[0].set_xticklabels(valid_labels, fontsize=7, rotation=20)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Mean cosine similarity across policy calls per layer
    for denoise_step, label, color in [(0, "k=0", "royalblue"), (4, "k=4", "tomato")]:
        cosine_sims = []
        valid_layers = []
        for layer_idx in probe_layers:
            f = feats[denoise_step].get(layer_idx)
            if f is None or len(f) < 2:
                continue
            # Mean pairwise cosine similarity (sample 100 pairs)
            rng = np.random.default_rng(42)
            n = min(len(f), 100)
            idx = rng.choice(len(f), size=n, replace=False)
            f_sub = f[idx]
            norms = np.linalg.norm(f_sub, axis=1, keepdims=True) + 1e-8
            f_norm = f_sub / norms
            sim_mat = f_norm @ f_norm.T
            # Mean of off-diagonal
            mask = ~np.eye(n, dtype=bool)
            cosine_sims.append(float(sim_mat[mask].mean()))
            valid_layers.append(layer_idx)

        axes[1].plot(range(len(valid_layers)), cosine_sims, "o-", color=color,
                     linewidth=2, markersize=6, label=label)

    axes[1].set_xlabel("Layer (shallow → deep)", fontsize=9)
    axes[1].set_ylabel("Mean pairwise cosine similarity", fontsize=9)
    axes[1].set_title("Feature Similarity across Policy Calls", fontsize=10)
    axes[1].set_xticks(range(len(probe_layers)))
    axes[1].set_xticklabels(valid_labels, fontsize=7, rotation=20)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "theme2_feature_variance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return results


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
    feats, episode_labels, call_idx_labels = load_features(
        args.npz_path, PROBE_LAYERS, NUM_DENOISE_STEPS
    )

    # Verify data
    for k in range(NUM_DENOISE_STEPS):
        for layer_idx in PROBE_LAYERS:
            f = feats[k].get(layer_idx)
            if f is not None:
                print(f"  k={k}, layer={layer_idx}: shape={f.shape}")

    print(f"\nEpisodes: {int(episode_labels.max())+1}, Total calls: {len(episode_labels)}")

    print("\nRunning analysis...")
    stats = {}

    # 2-1: PCA per layer at k=4
    plot_pca_per_layer(feats, episode_labels, call_idx_labels,
                       PROBE_LAYERS, out_dir, args.task_name, args.success_rate,
                       denoise_step=4)
    # Also k=0 for comparison
    plot_pca_per_layer(feats, episode_labels, call_idx_labels,
                       PROBE_LAYERS, out_dir, args.task_name, args.success_rate,
                       denoise_step=0)

    # 2-1: PC1 time series
    plot_pc1_time_series(feats, episode_labels, call_idx_labels,
                         PROBE_LAYERS, out_dir, args.task_name, args.success_rate,
                         denoise_step=4)

    # 2-1 extension: feature variance
    var_stats = plot_feature_variance_by_layer(
        feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate
    )
    stats.update({"feature_variance": var_stats})

    # 2-2: Feature change k=0→k=4 by layer
    change_stats = plot_feature_change_by_layer(
        feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate
    )
    stats.update(change_stats)

    # 2-2 extension: per-step change
    step_results = plot_step_change_per_layer(
        feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate
    )

    # 2-3: CKA matrix
    cka_stats = plot_cka_matrix(
        feats, PROBE_LAYERS, out_dir, args.task_name, args.success_rate, denoise_step=4
    )
    stats.update(cka_stats)

    # Save stats JSON
    stats["task"] = args.task_name
    stats["success_rate"] = args.success_rate
    stats["probe_layers"] = PROBE_LAYERS
    stats["total_calls"] = int(len(episode_labels))

    stats_path = out_dir / "theme2_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved: {stats_path}")
    print("Theme 2 analysis complete!")


if __name__ == "__main__":
    main()
