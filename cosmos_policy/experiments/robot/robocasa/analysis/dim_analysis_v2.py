"""
dim_analysis_v2.py — 設計書 §3.F 準拠の次元分解解析

§3.F 次元別アクション変化量:
  - step_actions.npz: shape (N=1108, T=32, D=7) per k=0..4
  - 次元グループ: XYZ (dims 0-2), Rotation (dims 3-5), Gripper (dim 6)
  - Per-call RMS change across T timesteps
  - グリッパーの大std → 二値切替によるものかを確認
    (切替イベント除外後の conditional std を計算)
  - Episode bootstrap 95% CI

成果物:
  dim_change_perdim.png — 各次元 RMS の k×次元 ヒートマップ + グループ折れ線
  gripper_switch_analysis.png — グリッパー切替条件付き std 分析
  dim_offset_ci.json

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.dim_analysis_v2 \
      --actions_npz results/action_denoising/step_actions.npz \
      --out_dir     results/action_dim_analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIGMA_SCHEDULE = [80.0, 42.3, 21.0, 9.6, 4.0]
DIM_GROUPS = {
    "XYZ":      [0, 1, 2],
    "Rotation": [3, 4, 5],
    "Gripper":  [6],
}
DIM_LABELS = ["X", "Y", "Z", "Rx", "Ry", "Rz", "Grip"]


def load_actions(npz_path: str, feat_npz_path: str = None):
    """
    step_actions.npz: keys は '0'..'4'、各 (N, T, D)。
    ep_labels は feat_npz_path から読み込む (step_actions.npz には存在しない)。
    """
    data = np.load(npz_path)
    ep_labels = data["episode_labels"] if "episode_labels" in data else None

    acts = {}
    for k in range(5):
        key = str(k)
        if key in data:
            acts[k] = data[key].astype(np.float32)

    if not acts:
        if "step_actions" in data:
            sa = data["step_actions"]
            for k in range(sa.shape[1]):
                acts[k] = sa[:, k, :, :]

    # ep_labels が step_actions.npz にない場合は features.npz から取得
    if ep_labels is None and feat_npz_path is not None:
        feat_data = np.load(feat_npz_path)
        if "episode_labels" in feat_data:
            ep_labels = feat_data["episode_labels"]

    return acts, ep_labels


def compute_rms_change(acts_k: np.ndarray) -> np.ndarray:
    """
    Per-call, per-dim RMS of (x_{t+1} - x_t) across T timesteps.
    acts_k: (N, T, D) → return: (N, D)
    """
    deltas = np.diff(acts_k, axis=1)  # (N, T-1, D)
    rms = np.sqrt((deltas**2).mean(axis=1))  # (N, D)
    return rms


def bootstrap_episode_ci(values: np.ndarray, ep_labels: np.ndarray,
                          n_boot: int = 1000) -> tuple:
    """
    values: (N, D) per-call values
    Returns: (means: D, ci_lo: D, ci_hi: D)
    """
    episodes = np.unique(ep_labels)
    # Per-episode means
    ep_means = np.array([values[ep_labels == ep].mean(axis=0) for ep in episodes])  # (E, D)
    n_ep = len(episodes)
    boots = np.array([ep_means[np.random.randint(0, n_ep, n_ep)].mean(axis=0)
                      for _ in range(n_boot)])  # (n_boot, D)
    return (float_arr(ep_means.mean(axis=0)),
            float_arr(np.percentile(boots, 2.5, axis=0)),
            float_arr(np.percentile(boots, 97.5, axis=0)))


def float_arr(x: np.ndarray):
    return [float(v) for v in x]


def analyze_gripper_switch(acts_k: np.ndarray, ep_labels: np.ndarray,
                            k: int, GRIPPER_THRESHOLD: float = 0.0) -> dict:
    """
    グリッパー次元 (dim=6) の switch イベントを検出し、
    切替を含むタイムステップを除いた conditional std を計算。
    """
    grip_vals = acts_k[:, :, 6]  # (N, T)
    grip_rms_all = np.sqrt(((np.diff(grip_vals, axis=1))**2).mean(axis=1))  # (N,)

    # 切替イベント: |grip[t+1] - grip[t]| > threshold の timestep があるか
    grip_diffs = np.abs(np.diff(grip_vals, axis=1))  # (N, T-1)
    has_switch = (grip_diffs > 0.3).any(axis=1)  # (N,) — 0.3 は二値切替の閾値

    grip_rms_no_switch = grip_rms_all[~has_switch]
    grip_rms_switch    = grip_rms_all[has_switch]

    return {
        f"k{k}_grip_rms_all_mean":       float(grip_rms_all.mean()),
        f"k{k}_grip_rms_all_std":        float(grip_rms_all.std()),
        f"k{k}_n_calls_with_switch":     int(has_switch.sum()),
        f"k{k}_n_calls_no_switch":       int((~has_switch).sum()),
        f"k{k}_grip_rms_no_switch_mean": float(grip_rms_no_switch.mean()) if len(grip_rms_no_switch) > 0 else None,
        f"k{k}_grip_rms_no_switch_std":  float(grip_rms_no_switch.std()) if len(grip_rms_no_switch) > 0 else None,
        f"k{k}_grip_rms_switch_mean":    float(grip_rms_switch.mean()) if len(grip_rms_switch) > 0 else None,
        f"k{k}_grip_rms_switch_std":     float(grip_rms_switch.std()) if len(grip_rms_switch) > 0 else None,
    }


def plot_dim_change(all_means: dict, all_lo: dict, all_hi: dict, out_dir: Path, task_name: str):
    """§3.F メイン図: 次元別 RMS 変化量の k×dim 全体像。"""
    n_k = 5
    D = 7
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"§3.F Per-Dimension Action Change (RMS across T=32)\nTask: {task_name}", fontsize=11)

    # (0) Heatmap: k × dim
    heat = np.array([[float(all_means[k][d]) if all_means.get(k) is not None else np.nan
                      for d in range(D)] for k in range(n_k)])  # (5, 7)
    im = axes[0].imshow(heat, aspect="auto", cmap="YlOrRd")
    axes[0].set_xticks(range(D))
    axes[0].set_xticklabels(DIM_LABELS, fontsize=9)
    axes[0].set_yticks(range(n_k))
    axes[0].set_yticklabels([f"k={k}\nσ≈{int(SIGMA_SCHEDULE[k])}" for k in range(n_k)], fontsize=8)
    axes[0].set_title("Per-Dim RMS Change (episode mean)")
    plt.colorbar(im, ax=axes[0], shrink=0.8)
    for ki in range(n_k):
        for di in range(D):
            v = heat[ki, di]
            if not np.isnan(v):
                axes[0].text(di, ki, f"{v:.3f}", ha="center", va="center", fontsize=7,
                             color="black" if v < heat.max() * 0.7 else "white")

    # (1) Group line plot: XYZ / Rotation / Gripper vs k
    k_vals = range(n_k)
    group_colors = {"XYZ": "steelblue", "Rotation": "darkorange", "Gripper": "green"}
    for gname, gdims in DIM_GROUPS.items():
        g_means = [np.mean([all_means[k][d] for d in gdims if all_means.get(k) is not None])
                   for k in k_vals]
        g_los   = [np.mean([all_lo[k][d] for d in gdims if all_lo.get(k) is not None])
                   for k in k_vals]
        g_his   = [np.mean([all_hi[k][d] for d in gdims if all_hi.get(k) is not None])
                   for k in k_vals]
        axes[1].plot(k_vals, g_means, "o-", color=group_colors[gname],
                     linewidth=2, markersize=7, label=gname)
        axes[1].fill_between(k_vals, g_los, g_his, alpha=0.15, color=group_colors[gname])

    axes[1].set_xlabel("Denoising step k  (σ decreases →)")
    axes[1].set_xticks(list(k_vals))
    axes[1].set_xticklabels([f"k={k}\nσ≈{int(SIGMA_SCHEDULE[k])}" for k in k_vals], fontsize=8)
    axes[1].set_ylabel("Group-mean per-dim RMS (episode 95%CI)")
    axes[1].set_title("Per-Group RMS Change vs Denoising Step")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "dim_change_perdim.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_gripper_switch(gripper_stats: dict, out_dir: Path, task_name: str):
    """グリッパー切替条件付き分析プロット。"""
    k_vals = list(range(5))
    rms_all = [gripper_stats.get(f"k{k}_grip_rms_all_mean", np.nan) for k in k_vals]
    rms_nosw = [gripper_stats.get(f"k{k}_grip_rms_no_switch_mean", np.nan) for k in k_vals]
    rms_sw   = [gripper_stats.get(f"k{k}_grip_rms_switch_mean", np.nan) for k in k_vals]
    n_switch = [gripper_stats.get(f"k{k}_n_calls_with_switch", 0) for k in k_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"§3.F Gripper Dimension Analysis\nTask: {task_name}", fontsize=11)

    axes[0].plot(k_vals, rms_all,   "o-", color="black",   label="All calls", linewidth=2, markersize=7)
    axes[0].plot(k_vals, rms_nosw,  "s-", color="steelblue", label="No-switch calls", linewidth=2, markersize=7)
    axes[0].plot(k_vals, rms_sw,    "^-", color="tomato",   label="Switch calls", linewidth=2, markersize=7)
    axes[0].set_xlabel("Denoising step k")
    axes[0].set_ylabel("Gripper dim RMS change")
    axes[0].set_title("Gripper RMS: All vs Switch vs No-Switch")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_vals)
    axes[0].set_xticklabels([f"k={k}" for k in k_vals])

    axes[1].bar(k_vals, n_switch, color="tomato", alpha=0.8)
    axes[1].set_xlabel("Denoising step k")
    axes[1].set_ylabel("N calls with gripper switch")
    axes[1].set_title("Gripper Switch Event Count per Step\n(|Δgrip| > 0.3 in any timestep)")
    axes[1].set_xticks(k_vals)
    axes[1].set_xticklabels([f"k={k}" for k in k_vals])
    axes[1].grid(True, alpha=0.3, axis="y")
    for ki, n in enumerate(n_switch):
        axes[1].text(ki, n + 1, str(n), ha="center", fontsize=10)

    plt.tight_layout()
    p = out_dir / "gripper_switch_analysis.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions_npz", required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--task_name",   default="PnPCounterToCab")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print(f"Loading actions: {args.actions_npz}")
    feat_npz = args.actions_npz.replace("action_denoising/step_actions.npz",
                                        "action_features/features.npz")
    acts, ep_labels = load_actions(args.actions_npz, feat_npz_path=feat_npz)

    if not acts:
        print("ERROR: No step_actions found in npz!")
        return

    k_sample = next(iter(acts.keys()))
    N, T, D = acts[k_sample].shape
    print(f"N={N}, T={T}, D={D}, n_steps={len(acts)}")
    print(f"n_episodes={len(np.unique(ep_labels)) if ep_labels is not None else 'N/A'}")

    if ep_labels is None:
        # フォールバック: 全サンプルを単一エピソードとして扱う
        ep_labels = np.zeros(N, dtype=int)

    all_means: dict = {}
    all_lo:    dict = {}
    all_hi:    dict = {}
    gripper_stats = {}

    for k in sorted(acts.keys()):
        print(f"\n--- k={k} (σ≈{SIGMA_SCHEDULE[k]}) ---")
        acts_k = acts[k]  # (N, T, D)
        rms_k  = compute_rms_change(acts_k)  # (N, D)
        means_k, lo_k, hi_k = bootstrap_episode_ci(rms_k, ep_labels, n_boot=1000)
        all_means[k] = means_k
        all_lo[k]    = lo_k
        all_hi[k]    = hi_k
        print(f"  Per-dim RMS means: " + " ".join(f"{DIM_LABELS[d]}={means_k[d]:.4f}" for d in range(D)))
        gripper_stats.update(analyze_gripper_switch(acts_k, ep_labels, k))

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_dim_change(all_means, all_lo, all_hi, out_dir, args.task_name)
    plot_gripper_switch(gripper_stats, out_dir, args.task_name)

    # ── JSON ──────────────────────────────────────────────────────────────────
    stats = {
        "task": args.task_name,
        "N_calls": int(N),
        "T_timesteps": int(T),
        "D_dims": int(D),
        "dim_labels": DIM_LABELS,
        "per_k_per_dim": {
            str(k): {
                "rms_mean": all_means[k],
                "rms_ci_lo": all_lo[k],
                "rms_ci_hi": all_hi[k],
            }
            for k in sorted(acts.keys())
        },
        "gripper_switch_analysis": gripper_stats,
    }
    jpath = out_dir / "dim_offset_ci.json"
    with open(jpath, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {jpath}")
    print(f"\ndim_analysis_v2 complete! Output: {out_dir}")


if __name__ == "__main__":
    main()
