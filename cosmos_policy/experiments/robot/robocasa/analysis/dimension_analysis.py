"""
eval_comment.md への回答: 次元ごとの分解分析

1. L2 変化量をアクション群（位置/姿勢/グリッパー）ごとに分割
2. k=0 と k=4 の空間的オフセット（次元ごとの定数シフト）を確認

step_actions.npz から読み込むため、シミュレーション再実行不要。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("cosmos_policy/experiments/robot/robocasa/analysis/results/action_denoising")
OUT_DIR = RESULTS_DIR  # 同じディレクトリに出力

# ── RoboCasa (Panda EEF delta control) のアクション次元マッピング
# unnormalize_actions=True なので policy output が実際の EEF デルタ制御値
# Panda + PandaMobile (7D manipulation action):
DIM_GROUPS = {
    "position (XYZ)": [0, 1, 2],
    "orientation (AxisAngle)": [3, 4, 5],
    "gripper": [6],
}
ACTION_DIM = 7


def load_step_actions(path: Path) -> dict:
    data = np.load(path)
    return {int(k): data[k] for k in data.files}  # {step_idx: (N, T, D)}


def compute_per_dim_deltas(step_actions: dict) -> dict:
    """
    連続するステップ間の ||x̂₀(k) - x̂₀(k-1)||₂ を
    アクション群ごとに分割して計算する。

    Returns:
        {group_name: {step_k: list of per-sample L2}}
    """
    sorted_steps = sorted(step_actions.keys())
    result = {g: {} for g in DIM_GROUPS}

    for i in range(1, len(sorted_steps)):
        k_prev = sorted_steps[i - 1]
        k_curr = sorted_steps[i]
        acts_prev = step_actions[k_prev]  # (N, T, D)
        acts_curr = step_actions[k_curr]

        diff = acts_curr - acts_prev  # (N, T, D)

        for group, dims in DIM_GROUPS.items():
            diff_g = diff[:, :, dims]           # (N, T, len(dims))
            # L2 over T and dim axes, per sample
            l2_per_sample = np.linalg.norm(diff_g.reshape(diff_g.shape[0], -1), axis=1)  # (N,)
            result[group][k_curr] = l2_per_sample

    return result


def compute_spatial_offsets(step_actions: dict) -> dict:
    """
    k=0 と k=last の各次元における定数オフセット: mean_over_samples_and_time(x̂₀_k4 - x̂₀_k0)

    Returns:
        offsets: (D,) mean offset per dimension
        stds: (D,) std across samples
    """
    sorted_steps = sorted(step_actions.keys())
    k_first = sorted_steps[0]
    k_last = sorted_steps[-1]

    acts_first = step_actions[k_first]  # (N, T, D)
    acts_last = step_actions[k_last]

    diff = acts_last - acts_first  # (N, T, D)
    # 各次元の「全サンプル・全タイムステップ」にわたる平均オフセット
    mean_offset = diff.mean(axis=(0, 1))  # (D,)
    std_offset = diff.std(axis=(0, 1))    # (D,)
    # 各サンプルの時間平均値の間の差（軌跡の平均値レベルのシフト）
    mean_per_sample_first = acts_first.mean(axis=1)  # (N, D)
    mean_per_sample_last = acts_last.mean(axis=1)
    sample_offset = mean_per_sample_last - mean_per_sample_first  # (N, D)

    return {
        "mean_offset": mean_offset,
        "std_offset": std_offset,
        "sample_offset": sample_offset,  # (N, D) for distribution analysis
        "k_first": k_first,
        "k_last": k_last,
    }


def plot_per_dim_delta(per_dim_deltas, step_actions, task_name="PnPCounterToCab"):
    sorted_steps = sorted(step_actions.keys())
    transitions = sorted(next(iter(per_dim_deltas.values())).keys())
    tick_labels = [f"k={k-1}→{k}" for k in transitions]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Dimension-wise L2 Change (k-1 → k)  |  Task: {task_name}\n"
        f"N={len(step_actions[sorted_steps[0]])} samples per step",
        fontsize=11,
    )

    group_colors = {
        "position (XYZ)": "royalblue",
        "orientation (AxisAngle)": "forestgreen",
        "gripper": "tomato",
    }

    # Panel 1: per-group bars
    x = np.arange(len(transitions))
    width = 0.25
    for i, (group, color) in enumerate(group_colors.items()):
        means = [per_dim_deltas[group][k].mean() for k in transitions]
        stds = [per_dim_deltas[group][k].std() for k in transitions]
        axes[0].bar(x + i * width, means, width, yerr=stds,
                    label=group, color=color, alpha=0.8, capsize=3)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(tick_labels)
    axes[0].set_ylabel("||Δx̂₀||₂ per group")
    axes[0].set_title("L2 Change by Action Group")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: per-dim bars (all 7 dims)
    dim_colors = plt.cm.tab10(np.linspace(0, 1, ACTION_DIM))
    x_dim = np.arange(len(transitions))
    w = 1.0 / (ACTION_DIM + 1)
    for d in range(ACTION_DIM):
        acts_all_steps = [step_actions[sorted_steps[i]] for i in range(len(sorted_steps))]
        means_d = []
        stds_d = []
        for ki, k in enumerate(transitions):
            k_prev = sorted_steps[ki]   # ki corresponds to k-1
            k_curr = sorted_steps[ki + 1]
            diff_d = (step_actions[k_curr] - step_actions[k_prev])[:, :, d]  # (N, T)
            l2 = np.linalg.norm(diff_d, axis=1)  # (N,)
            means_d.append(l2.mean())
            stds_d.append(l2.std())
        axes[1].bar(x_dim + d * w, means_d, w, yerr=stds_d,
                    label=f"dim {d}", color=dim_colors[d], alpha=0.8, capsize=2)
    axes[1].set_xticks(x_dim + (ACTION_DIM - 1) * w / 2)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_ylabel("||Δx̂₀||₂ per dim")
    axes[1].set_title("L2 Change per Individual Dimension")
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: proportion of total L2 change at k=3→4 by group
    k_last_trans = transitions[-1]
    group_means_last = {g: per_dim_deltas[g][k_last_trans].mean() for g in DIM_GROUPS}
    total = sum(group_means_last.values())
    labels_pie = [f"{g}\n{v:.3f} ({v/total*100:.1f}%)" for g, v in group_means_last.items()]
    colors_pie = [group_colors[g] for g in DIM_GROUPS]
    axes[2].pie(group_means_last.values(), labels=labels_pie, colors=colors_pie,
                autopct=None, startangle=90, wedgeprops=dict(alpha=0.8))
    axes[2].set_title(f"L2 Contribution at {tick_labels[-1]}")

    plt.tight_layout()
    path = OUT_DIR / "dim_analysis_l2_change.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_spatial_offset(offset_info, task_name="PnPCounterToCab"):
    mean_offset = offset_info["mean_offset"]   # (D,)
    std_offset = offset_info["std_offset"]
    sample_offset = offset_info["sample_offset"]  # (N, D)
    k_first = offset_info["k_first"]
    k_last = offset_info["k_last"]

    D = len(mean_offset)
    dims = np.arange(D)
    dim_labels = [f"dim{d}" for d in range(D)]
    for g, idxs in DIM_GROUPS.items():
        for i in idxs:
            short = g.split(" ")[0]
            dim_labels[i] = f"d{i}\n({short})"

    group_colors_flat = []
    gc = {"position": "royalblue", "orientation": "forestgreen", "gripper": "tomato"}
    for d in range(D):
        for g, idxs in DIM_GROUPS.items():
            if d in idxs:
                group_colors_flat.append(gc[g.split(" ")[0]])
                break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Spatial Offset (k={k_first} → k={k_last})  |  Task: {task_name}\n"
        f"mean(x̂₀_k{k_last}) - mean(x̂₀_k{k_first}) per dimension",
        fontsize=11,
    )

    # Panel 1: mean ± std bar chart per dimension
    axes[0].bar(dims, mean_offset, yerr=std_offset, color=group_colors_flat, alpha=0.8, capsize=4)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(dims)
    axes[0].set_xticklabels(dim_labels, fontsize=8)
    axes[0].set_ylabel("Mean offset (k=last minus k=first)")
    axes[0].set_title("Mean Spatial Offset per Dimension")
    axes[0].grid(True, alpha=0.3)
    for d in range(D):
        axes[0].text(d, mean_offset[d] + std_offset[d] * 0.15,
                     f"{mean_offset[d]:.4f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: distribution of per-sample offset (violin or box) for each dim
    bp = axes[1].boxplot(
        [sample_offset[:, d] for d in range(D)],
        positions=dims,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], group_colors_flat):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(dims)
    axes[1].set_xticklabels(dim_labels, fontsize=8)
    axes[1].set_ylabel("Per-sample offset distribution")
    axes[1].set_title("Sample-wise Offset Distribution per Dimension")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "dim_analysis_spatial_offset.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_gripper_trajectory(step_actions, task_name="PnPCounterToCab"):
    """グリッパー次元（dim 6）の k=0〜4 の軌跡を詳細に可視化"""
    sorted_steps = sorted(step_actions.keys())
    gripper_dim = 6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gripper Dimension (dim 6) Trajectories per Denoising Step  |  Task: {task_name}",
        fontsize=11,
    )

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sorted_steps)))

    # Panel 1: mean trajectory per denoising step
    T = step_actions[sorted_steps[0]].shape[1]
    t = np.arange(T)
    for i, k in enumerate(sorted_steps):
        acts = step_actions[k][:, :, gripper_dim]  # (N, T)
        mean_g = acts.mean(axis=0)
        std_g = acts.std(axis=0)
        axes[0].fill_between(t, mean_g - std_g, mean_g + std_g, alpha=0.1, color=colors[i])
        axes[0].plot(t, mean_g, color=colors[i], linewidth=2, label=f"k={k}")
    axes[0].set_xlabel("Chunk timestep")
    axes[0].set_ylabel("Gripper value (dim 6)")
    axes[0].set_title("Gripper Mean Trajectory per Step (with ±1σ band)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: distribution of gripper values (histogram) at k=0 vs k=last
    k_first = sorted_steps[0]
    k_last = sorted_steps[-1]
    g_first = step_actions[k_first][:, :, gripper_dim].flatten()
    g_last = step_actions[k_last][:, :, gripper_dim].flatten()
    bins = np.linspace(min(g_first.min(), g_last.min()), max(g_first.max(), g_last.max()), 40)
    axes[1].hist(g_first, bins=bins, alpha=0.5, color="royalblue", label=f"k={k_first} (σ=80)")
    axes[1].hist(g_last, bins=bins, alpha=0.5, color="tomato", label=f"k={k_last} (σ=4)")
    axes[1].set_xlabel("Gripper value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Gripper Value Distribution: k=first vs k=last")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "dim_analysis_gripper.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    npz_path = RESULTS_DIR / "step_actions.npz"
    print(f"Loading {npz_path} ...")
    step_actions = load_step_actions(npz_path)
    sorted_steps = sorted(step_actions.keys())
    print(f"Steps: {sorted_steps}  |  Shape per step: {step_actions[sorted_steps[0]].shape}")

    # ── 1. 次元ごとの L2 変化量 ──────────────────────────────────────────
    print("\n=== Per-dimension L2 Changes ===")
    per_dim_deltas = compute_per_dim_deltas(step_actions)
    for group, group_data in per_dim_deltas.items():
        print(f"\n  [{group}]")
        for k, vals in sorted(group_data.items()):
            k_prev = sorted_steps[sorted_steps.index(k) - 1]
            print(f"    k={k_prev}→{k}: mean={vals.mean():.4f}  std={vals.std():.4f}")

    # ── 2. 空間的オフセット ───────────────────────────────────────────────
    print("\n=== Spatial Offsets (k=first → k=last) ===")
    offset_info = compute_spatial_offsets(step_actions)
    for d in range(ACTION_DIM):
        group = next(g for g, idxs in DIM_GROUPS.items() if d in idxs)
        print(f"  dim {d} ({group}): mean_offset = {offset_info['mean_offset'][d]:.5f}"
              f"  std = {offset_info['std_offset'][d]:.5f}")

    # ── プロット ──────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_per_dim_delta(per_dim_deltas, step_actions)
    plot_spatial_offset(offset_info)
    plot_gripper_trajectory(step_actions)

    # ── 3. 数値サマリー ───────────────────────────────────────────────────
    import json
    transitions = sorted(next(iter(per_dim_deltas.values())).keys())
    summary = {
        "per_group_l2_change_mean": {
            g: {str(k): float(per_dim_deltas[g][k].mean()) for k in transitions}
            for g in DIM_GROUPS
        },
        "per_group_l2_change_std": {
            g: {str(k): float(per_dim_deltas[g][k].std()) for k in transitions}
            for g in DIM_GROUPS
        },
        "spatial_offset_mean": {str(d): float(offset_info["mean_offset"][d]) for d in range(ACTION_DIM)},
        "spatial_offset_std": {str(d): float(offset_info["std_offset"][d]) for d in range(ACTION_DIM)},
    }
    out_json = RESULTS_DIR / "dim_analysis_stats.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
