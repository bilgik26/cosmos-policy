"""
skill_activation_heatmap.py — attractor_verification_design.md §7 (時間的特徴) 準拠

各クラスタ i が主にどの τ/フェーズ位置で最大確率になるか (活性化ヒートマップ)、
所属確率エントロピーのピーク鋭さ、相空間速度のピークを計算する。

  - GMM (K=consensus_k, skill_count.py と同じ前処理) の事後確率で「ソフト」クラスタ所属を得る。
  - 進行度 (call_idx / episode内最大call_idx) を10 binに離散化し、bin×クラスタの平均所属確率
    ヒートマップを描画。
  - 所属確率エントロピー H(call) = -Σp_i log p_i を進行度binごとに平均。
    Null: 進行度ラベルをepisode内でシャッフルし、bin間エントロピー分散の帰無分布を作る。
    実測のbin間分散がnullを有意に超える → 活性化は進行度にロックしている(鋭い遷移がある)。
  - 相空間速度 ‖z_t − z_{t-1}‖ (call間, episode内) のピークが phase_labeling.py の
    transition_flag (gripper状態変化near-window) と一致するかを検証。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import (
    LAYER, K_STEP, load_task_seed_data, scene_residualize, preprocess,
)

N_BINS = 10


def compute_progress(episode, call_idx):
    progress = np.zeros(len(episode), dtype=float)
    for e in np.unique(episode):
        mask = episode == e
        max_c = call_idx[mask].max()
        progress[mask] = call_idx[mask] / max(max_c, 1)
    return progress


def entropy(p, eps=1e-12):
    return -np.sum(p * np.log(p + eps), axis=1)


def phase_space_velocity(Xp, episode, call_idx):
    n = len(episode)
    vel = np.zeros(n)
    order = np.lexsort((call_idx, episode))
    for pos in range(len(order)):
        i = order[pos]
        if pos == 0 or episode[order[pos - 1]] != episode[i]:
            vel[i] = np.nan
        else:
            prev = order[pos - 1]
            vel[i] = np.linalg.norm(Xp[i] - Xp[prev])
    # backfill first-of-episode from next
    for pos in range(len(order)):
        i = order[pos]
        if np.isnan(vel[i]):
            if pos + 1 < len(order) and episode[order[pos + 1]] == episode[i]:
                vel[i] = vel[order[pos + 1]]
            else:
                vel[i] = 0.0
    return vel


def analyze_task(collect_dir, task, fnames_by_seed, consensus_k, out_dir):
    result = {"task": task, "consensus_k": consensus_k, "per_seed_series": {}}
    for seed_series, fname in fnames_by_seed.items():
        d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
        X_raw, episode, call_idx = d["feats"], d["episode"], d["call_idx"]
        phase_path = collect_dir / fname.replace(".npz", "_phases.npz")
        keep_idx = np.load(collect_dir / fname)[f"feat_k{K_STEP}_layer{LAYER}_idx"]
        transition_flag = np.load(phase_path)["transition_flag"][keep_idx]

        Xr = scene_residualize(X_raw, episode)
        Xp = preprocess(Xr, seed=0).astype(np.float64)
        k = max(consensus_k, 2)

        reg = max(1e-6, float(np.var(Xp)) * 1e-3)
        gmm = GaussianMixture(n_components=k, random_state=0, n_init=3,
                               covariance_type="diag", reg_covar=reg).fit(Xp)
        proba = gmm.predict_proba(Xp)
        ent = entropy(proba)
        progress = compute_progress(episode, call_idx)
        bins = np.clip((progress * N_BINS).astype(int), 0, N_BINS - 1)

        heatmap = np.zeros((k, N_BINS))
        for b in range(N_BINS):
            mask = bins == b
            if mask.any():
                heatmap[:, b] = proba[mask].mean(axis=0)

        bin_entropy_mean = np.array([ent[bins == b].mean() if (bins == b).any() else np.nan
                                      for b in range(N_BINS)])
        real_var = float(np.nanvar(bin_entropy_mean))

        rng = np.random.RandomState(5)
        null_vars = []
        for _ in range(100):
            bins_shuf = bins.copy()
            for e in np.unique(episode):
                mask = episode == e
                bins_shuf[mask] = rng.permutation(bins[mask])
            be = np.array([ent[bins_shuf == b].mean() if (bins_shuf == b).any() else np.nan
                           for b in range(N_BINS)])
            null_vars.append(np.nanvar(be))
        null_vars = np.array(null_vars)
        p_val = float((np.sum(null_vars >= real_var) + 1) / (len(null_vars) + 1))

        vel = phase_space_velocity(Xp, episode, call_idx)
        vel_thresh = np.percentile(vel, 90)
        high_vel_mask = vel >= vel_thresh
        frac_transition_in_high_vel = float(transition_flag[high_vel_mask].mean()) if high_vel_mask.any() else float("nan")
        frac_transition_overall = float(transition_flag.mean())

        # plot heatmap
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0, vmax=heatmap.max())
        ax.set_xlabel("progress bin (0=episode start, 9=episode end)")
        ax.set_ylabel("cluster")
        ax.set_title(f"{task} seed={seed_series}: cluster activation vs progress")
        plt.colorbar(im, ax=ax, label="mean posterior prob.")
        fig.tight_layout()
        fig_path = out_dir / f"activation_heatmap_{task}_seed{seed_series}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

        result["per_seed_series"][str(seed_series)] = {
            "heatmap": heatmap.tolist(),
            "bin_entropy_mean": bin_entropy_mean.tolist(),
            "bin_entropy_variance_real": real_var,
            "bin_entropy_variance_null_mean": float(null_vars.mean()),
            "bin_entropy_variance_null_std": float(null_vars.std()),
            "p_value_progress_locked": p_val,
            "progress_locked_supported": bool(p_val < 0.05),
            "velocity_p90_threshold": float(vel_thresh),
            "frac_transition_flag_in_high_velocity_calls": frac_transition_in_high_vel,
            "frac_transition_flag_overall": frac_transition_overall,
            "velocity_peaks_match_gripper_transitions": bool(
                not np.isnan(frac_transition_in_high_vel)
                and frac_transition_in_high_vel > 1.5 * frac_transition_overall
            ),
            "heatmap_png": str(fig_path.name),
        }
        log_message(
            f"[heatmap {task} seed={seed_series}] entropy_var real={real_var:.4f} "
            f"null={null_vars.mean():.4f}±{null_vars.std():.4f} p={p_val:.3f} | "
            f"high-vel transition-frac={frac_transition_in_high_vel:.2f} "
            f"vs overall={frac_transition_overall:.2f}"
        )
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--skill_count_json", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())
    skill_count = json.loads(Path(args.skill_count_json).read_text())

    files_by_task = {}
    for fname, info in manifest["files"].items():
        files_by_task.setdefault(info["task"], {})[info["seed_base"]] = fname

    results = {}
    for task, fnames_by_seed in files_by_task.items():
        ks = skill_count["level1"][task]["consensus_k_by_seed_series"]
        consensus_k = int(round(np.mean(list(ks.values())))) if ks else 2
        results[task] = analyze_task(collect_dir, task, fnames_by_seed, consensus_k, out_dir)

    with open(out_dir / "skill_activation_heatmap.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'skill_activation_heatmap.json'}")


if __name__ == "__main__":
    main()
