"""
arm_ablation_test.py — review_report.md (latent_dynamics_verification 査読) MUST-1 対応。

report.md §3.4 / report_v2.md §4.4 の中心的な肯定的知見(「D空間でPnPCounterToCabのDMD力学
シグナルが顕在化する」)は、`X_p`空間(遅延埋め込みなし・行動ブロックなし)とD空間(遅延埋め込み
あり・行動ブロックあり)という、**最低3要因**(遅延埋め込みの有無・行動ブロックの有無・次元数と
分散構造)で同時に異なる2条件の比較から導かれており、どの要因が効果を生んでいるのか単離できて
いなかった(review_report.md §1 D-1)。本スクリプトは同一サンプル(collect_actions/、
action_dspace_test.pyが使う行動チャンク付きデータ)上で4アームを構築し、同一のDMD検定・進行度
逸脱検定を適用することでこれを単離する:

  A0 [Xp]                       (10dim) : 遅延埋め込みなし(=既存のXp空間そのもの)。基準。
  A1 [Xp, V]                    (20dim) : 遅延埋め込みあり(tau=3, PCA(8))、行動ブロックなし。
                                           遅延埋め込み単独の効果を測る。
  A2 [Xp, V, A_PCA_shuffled]    (25dim) : 遅延埋め込みあり、行動PCA特徴をエピソード間・時間方向に
                                           シャッフルした代理変数(周辺分布・分散スケールのみ保存)。
                                           「次元を追加するだけ」のプラセボ効果を測る。
  A3 [Xp, V, A_PCA]             (25dim) : 遅延埋め込みあり、真の行動ブロックあり(=action_dspace_
                                           test.pyの(3)群そのもの)。結合の効果(本検証の主張)。

判定基準(review_report.md記載): A1で既にp~0.02が出るなら(R)の主張(結合表現が構造を顕在化させる)
は崩壊する。A2で改善するなら次元追加自体のアーチファクトである。A3 > A2 ~ A1 ~ A0 が示せて初めて
「感覚運動の結合」自体の効果と言える。

全アームで同一のwindowed DMD(dmd_jacobian_stability_test.windowed_dmd, WINDOW=8)・進行度逸脱
検定(dynamics_embedding_test.deviation_mannwhitney)を適用する。A0はD_dim=10で遅延埋め込み
PCAを経由しない(素のXpをそのまま検定対象とする)点のみ他と異なる。
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import scene_residualize
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.manifold_trajectory_test import episode_progress
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.success_failure_trajectory_test import MIN_FAIL_EPISODES
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.dmd_jacobian_stability_test import windowed_dmd, WINDOW
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamics_embedding_test import (
    PROGRESS_PCA_DIM, WINDOW_TAU, D_DIM, build_delay_embedding_input, deviation_mannwhitney,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.action_dspace_test import (
    A_DIM, load_pooled_task_data_actions,
)

N_SHUFFLE_SEEDS = 20   # A2's shuffled-action surrogate is randomized; report the median arm across seeds
                        # plus the full per-seed distribution (a single shuffle draw is itself a random
                        # variable and could look better/worse than typical by chance)


def dmd_test(D, episode, call_idx, success, progress):
    episodes = np.unique(episode)
    ep_success = {e: bool(success[episode == e][0]) for e in episodes}
    success_episodes = np.array([e for e in episodes if ep_success[e]])
    fail_episodes = np.array([e for e in episodes if not ep_success[e]])
    if len(fail_episodes) < MIN_FAIL_EPISODES:
        return None

    def episode_traj(e):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        return D[order], progress[order]

    succ_max, fail_max = [], []
    succ_n_windows, fail_n_windows = [], []
    succ_crossed = fail_crossed = 0
    for e in success_episodes:
        Z, prog = episode_traj(e)
        if len(Z) < WINDOW:
            continue
        windows = windowed_dmd(Z, prog)
        if not windows:
            continue
        lmax = np.array([w["lambda_max"] for w in windows])
        succ_max.append(lmax.max())
        succ_n_windows.append(len(windows))
        succ_crossed += int((lmax > 1.0).any())
    for e in fail_episodes:
        Z, prog = episode_traj(e)
        if len(Z) < WINDOW:
            continue
        windows = windowed_dmd(Z, prog)
        if not windows:
            continue
        lmax = np.array([w["lambda_max"] for w in windows])
        fail_max.append(lmax.max())
        fail_n_windows.append(len(windows))
        fail_crossed += int((lmax > 1.0).any())

    if not (len(succ_max) and len(fail_max)):
        return None
    u_stat, p = mannwhitneyu(succ_max, fail_max, alternative="less")
    table = [[succ_crossed, len(succ_max) - succ_crossed], [fail_crossed, len(fail_max) - fail_crossed]]
    _, fisher_p = fisher_exact(table, alternative="less")
    return {
        "n_success_episodes_used": len(succ_max), "n_fail_episodes_used": len(fail_max),
        "episode_max_lambda_mannwhitney_success_lt_fail_p": float(p),
        "frac_episodes_crossing_1": {"success": f"{succ_crossed}/{len(succ_max)}",
                                      "fail": f"{fail_crossed}/{len(fail_max)}"},
        "fisher_crossing_success_lt_fail_p": float(fisher_p),
        "mean_n_windows_per_episode": {"success": float(np.mean(succ_n_windows)),
                                        "fail": float(np.mean(fail_n_windows))},
        "auc_max_lambda": float(u_stat / (len(succ_max) * len(fail_max))),
    }


def build_arm_D(c_block, episode, call_idx, d_dim, seed=0):
    """Standardize -> tau=3 backward delay-embed -> linear PCA(d_dim). Same pipeline as
    dynamics_embedding_test.py / action_dspace_test.py (incl. the §7 bug#1 standardize-before-PCA
    fix), applied here to an arbitrary c_t block so all arms share identical machinery."""
    c_scaler = StandardScaler().fit(c_block)
    c = c_scaler.transform(c_block)
    S = build_delay_embedding_input(c, episode, call_idx, tau=WINDOW_TAU)
    encoder_pca = PCA(n_components=min(d_dim, S.shape[0] - 1, S.shape[1]), random_state=seed)
    D = encoder_pca.fit_transform(S)
    return D, float(encoder_pca.explained_variance_ratio_.sum())


def shuffle_action_pca(A_pca, seed):
    """Surrogate for A2: permute rows of A_PCA across the ENTIRE pooled (episode x call) axis,
    destroying any true state-action temporal correlation while exactly preserving each column's
    marginal distribution and variance (a row permutation cannot change per-column statistics)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(A_pca))
    return A_pca[perm]


def analyze_task(collect_dir, manifest, task, out_dir, seed=0):
    pd_ = load_pooled_task_data_actions(collect_dir, manifest, task)
    X_raw, episode, call_idx = pd_["feats"], pd_["episode"], pd_["call_idx"]
    success_call, action_chunk_flat = pd_["success"], pd_["action_chunk_flat"]

    Xr = scene_residualize(X_raw, episode)
    ambient_scaler = StandardScaler().fit(Xr)
    Xs = ambient_scaler.transform(Xr)
    prog_pca = PCA(n_components=min(PROGRESS_PCA_DIM, Xs.shape[0] - 1, Xs.shape[1]), random_state=seed)
    Xp = prog_pca.fit_transform(Xs)
    progress = episode_progress(episode, call_idx)

    action_scaler = StandardScaler().fit(action_chunk_flat)
    action_pca = PCA(n_components=min(A_DIM, action_chunk_flat.shape[0] - 1, action_chunk_flat.shape[1]),
                      random_state=seed)
    A_pca = action_pca.fit_transform(action_scaler.transform(action_chunk_flat))

    V = np.zeros_like(Xp)
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        if len(order) < 2:
            continue
        V[order[1:]] = Xp[order[1:]] - Xp[order[:-1]]

    results = {}

    # ── A0: raw Xp, no delay embedding, no action block (baseline) ──
    dev_A0 = deviation_mannwhitney(Xp, episode, call_idx, success_call, seed=seed)
    dmd_A0 = dmd_test(Xp, episode, call_idx, success_call, progress)
    results["A0_Xp_only"] = {"dim": Xp.shape[1], "encoder_explained_var": None,
                              "deviation_test": dev_A0, "dmd_test": dmd_A0}

    # ── A1: [Xp, V], delay-embedded, no action block ──
    c_A1 = np.concatenate([Xp, V], axis=1)
    D_A1, var_A1 = build_arm_D(c_A1, episode, call_idx, D_DIM, seed=seed)
    dev_A1 = deviation_mannwhitney(D_A1, episode, call_idx, success_call, seed=seed)
    dmd_A1 = dmd_test(D_A1, episode, call_idx, success_call, progress)
    results["A1_Xp_V_delay_embedded"] = {"dim": c_A1.shape[1], "encoder_explained_var": var_A1,
                                          "deviation_test": dev_A1, "dmd_test": dmd_A1}

    # ── A2: [Xp, V, A_PCA_shuffled], delay-embedded (placebo, N_SHUFFLE_SEEDS draws) ──
    a2_draws = []
    for shuffle_seed in range(N_SHUFFLE_SEEDS):
        A_shuf = shuffle_action_pca(A_pca, seed=shuffle_seed)
        c_A2 = np.concatenate([Xp, V, A_shuf], axis=1)
        D_A2, var_A2 = build_arm_D(c_A2, episode, call_idx, D_DIM, seed=seed)
        dev_A2 = deviation_mannwhitney(D_A2, episode, call_idx, success_call, seed=seed)
        dmd_A2 = dmd_test(D_A2, episode, call_idx, success_call, progress)
        a2_draws.append({"shuffle_seed": shuffle_seed, "encoder_explained_var": var_A2,
                          "deviation_test": dev_A2, "dmd_test": dmd_A2})
    dmd_ps = [d["dmd_test"]["episode_max_lambda_mannwhitney_success_lt_fail_p"] for d in a2_draws if d["dmd_test"]]
    dmd_aucs = [d["dmd_test"]["auc_max_lambda"] for d in a2_draws if d["dmd_test"]]
    results["A2_Xp_V_shuffled_action_placebo"] = {
        "dim": c_A1.shape[1] + A_DIM, "n_shuffle_draws": N_SHUFFLE_SEEDS,
        "dmd_p_median": float(np.median(dmd_ps)) if dmd_ps else None,
        "dmd_p_min": float(np.min(dmd_ps)) if dmd_ps else None,
        "dmd_p_max": float(np.max(dmd_ps)) if dmd_ps else None,
        "dmd_auc_median": float(np.median(dmd_aucs)) if dmd_aucs else None,
        "per_draw": a2_draws,
    }

    # ── A3: [Xp, V, A_PCA], delay-embedded (true action block = action_dspace_test.py's group (3)) ──
    c_A3 = np.concatenate([Xp, V, A_pca], axis=1)
    D_A3, var_A3 = build_arm_D(c_A3, episode, call_idx, D_DIM, seed=seed)
    dev_A3 = deviation_mannwhitney(D_A3, episode, call_idx, success_call, seed=seed)
    dmd_A3 = dmd_test(D_A3, episode, call_idx, success_call, progress)
    results["A3_Xp_V_action_true"] = {"dim": c_A3.shape[1], "encoder_explained_var": var_A3,
                                       "deviation_test": dev_A3, "dmd_test": dmd_A3}

    def _p(arm_key):
        r = results[arm_key].get("dmd_test")
        return r["episode_max_lambda_mannwhitney_success_lt_fail_p"] if r else float("nan")

    log_message(f"[arm_ablation {task}] DMD p: A0={_p('A0_Xp_only'):.4g} A1={_p('A1_Xp_V_delay_embedded'):.4g} "
                f"A2(median of {N_SHUFFLE_SEEDS})={results['A2_Xp_V_shuffled_action_placebo']['dmd_p_median']} "
                f"A3={_p('A3_Xp_V_action_true'):.4g}")

    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True,
                    help="collect_action_chunks.py's output dir (needs action_chunk + episode_success)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())
    tasks = sorted({info["task"] for info in manifest["files"].values()})

    results = {
        "method_note": (
            "MUST-1 4-arm ablation isolating delay-embedding vs action-block effects on the DMD "
            "self-stabilization/destabilization signal (report.md §3.3/§3.4). A0=Xp only "
            "(no delay embedding), A1=[Xp,V] delay-embedded (no action block), "
            "A2=[Xp,V,A_PCA_shuffled] delay-embedded placebo (N=20 shuffle draws), "
            "A3=[Xp,V,A_PCA] delay-embedded true (= action_dspace_test.py's group (3))."
        ),
        "n_shuffle_seeds": N_SHUFFLE_SEEDS, "tasks": {},
    }
    for task in tasks:
        results["tasks"][task] = analyze_task(collect_dir, manifest, task, out_dir, seed=0)

    with open(out_dir / "arm_ablation_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'arm_ablation_test.json'}")


if __name__ == "__main__":
    main()
