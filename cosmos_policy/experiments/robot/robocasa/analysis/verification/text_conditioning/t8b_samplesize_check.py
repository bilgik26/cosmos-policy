"""Verify whether T8b's ~20pp lower Blk-18/22 baseline probe accuracy (N=360, 15 episodes)
vs. the original probe_acc_ci.json (N=1108, 50 episodes) is explained by sample size alone.

Method: subsample the GOOD features.npz to 15 random episodes (matching T8b's episode
count), using the EXACT SAME pipeline as linear_probe_v2.py (fold-internal PCA n=30,
LogisticRegression, LOEO) that generated probe_acc_ci.json, with the SAME label definition
T8b used (gripper_2_median: median-threshold of the k=4 action gripper dim).
Repeat over many random 15-episode draws to get a distribution, and compare to:
  (a) T8b's actual observed values (Blk-18=0.747, Blk-22=0.775, Blk-27=0.891)
  (b) the full N=1108/50ep reference (gripper_2_median: Blk-18=0.948, Blk-22=0.957, Blk-27=0.969)
If subsampled 15-episode accuracy lands close to (a), sample size alone explains the gap.
If it stays close to (b), something else about T8b's live rollout is responsible.
"""
import sys
sys.path.insert(0, "/home/bilgehan.sakai/cosmos-policy")
import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import PROBE_LAYERS, NUM_DENOISE_STEPS
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.linear_probe_v2 import (
    labels_gripper_median, precompute_pca_projections, run_loeo_lr,
)

FEAT = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz"
ACTIONS = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_denoising/step_actions.npz"
K = 0  # match T8b's k=0 baseline
N_EPISODES_TARGET = 15
N_RESAMPLES = 20
N_PCA = 30
SEED = 0


def main():
    raw = np.load(FEAT)
    ep_arr_full = raw["episode_labels"]
    act_raw = np.load(ACTIONS)
    actions = {int(k): act_raw[k] for k in act_raw.files}
    labels_full = labels_gripper_median(actions)
    n_classes = 2

    all_episodes = np.unique(ep_arr_full)
    rng = np.random.default_rng(SEED)

    results = {l: [] for l in PROBE_LAYERS}

    for trial in range(N_RESAMPLES):
        chosen_eps = rng.choice(all_episodes, size=N_EPISODES_TARGET, replace=False)
        mask = np.isin(ep_arr_full, chosen_eps)
        ep_arr = ep_arr_full[mask]
        labels = labels_full[mask]
        n_calls = mask.sum()

        feats_sub = {k: {l: None for l in PROBE_LAYERS} for k in range(NUM_DENOISE_STEPS)}
        for l in PROBE_LAYERS:
            feats_sub[K][l] = raw[f"feat_k{K}_layer{l}"][mask].astype(np.float32)

        pca_cache = precompute_pca_projections(feats_sub, ep_arr, n_pca=N_PCA)
        for l in PROBE_LAYERS:
            mean_acc, fold_accs, _ = run_loeo_lr(pca_cache, labels, ep_arr, l, K, n_classes)
            results[l].append(mean_acc)
        print(f"trial {trial+1}/{N_RESAMPLES}: n_calls={n_calls}, "
              f"Blk18={results[18][-1]:.3f} Blk22={results[22][-1]:.3f} Blk27={results[27][-1]:.3f}")

    print("\n=== Summary: 15-episode subsample of GOOD features.npz, gripper_2_median, k=0 ===")
    print(f"{'Layer':<8}{'mean':<10}{'std':<10}{'min':<10}{'max':<10}")
    for l in PROBE_LAYERS:
        arr = np.array(results[l])
        print(f"Blk-{l:<5}{arr.mean():<10.3f}{arr.std():<10.3f}{arr.min():<10.3f}{arr.max():<10.3f}")

    print("\n=== Reference values ===")
    print("T8b observed (N=360,15ep, live rollout): Blk-18=0.747 Blk-22=0.775 Blk-27=0.891")
    print("Full N=1108/50ep (gripper_2_median):     Blk-18=0.948 Blk-22=0.957 Blk-27=0.969")


if __name__ == "__main__":
    main()
