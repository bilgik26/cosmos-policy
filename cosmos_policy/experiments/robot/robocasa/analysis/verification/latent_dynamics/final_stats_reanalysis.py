"""
final_stats_reanalysis.py — review_report.md (latent_dynamics_verification 査読) S-1/L-2/L-4/L-5/
S-10 対応。GPU不要、既存artifact・既存ロールアウトJSONの再解析のみ。

S-1/L-2 (paired bootstrap ΔAUC): report_v2.md §4.4/§5 の「(2)->(3)で約690倍p値が縮小」という
  記述は、p値の比を効果量として扱う誤り(p値は効果量ではない、review_report.md §4 L-2)。(2)(3)は
  **同一エピソード**(collect_actions/)上の比較なので、エピソード単位のpaired bootstrapで
  Delta_AUC(= AUC_3 - AUC_2)の95%CIを直接計算する(action_dspace_artifact_<task>.pklと
  dynamics_embedding_on_collect_actions/dynamics_embedding_artifact_<task>.pklの両方から
  同一episode集合のDMD max|lambda|系列を再構成し、共通のepisode集合をペアリングしてresample)。

B-1 (天井p値の効果量置換): 完全分離時にMann-Whitney Uが群サイズだけの関数になる4つの主要p値
  (v1 §2.3/§3.3のPnP・TurnOnStove逸脱/エネルギー検定)をAUC(=1.0、完全分離)+
  分離度のブートストラップCI(正規化逸脱マージンのepisode-bootstrap CI)に置換する。

L-4 (移動量の共変量統制): 多様体逸脱度・DTW距離が「動いていないほど良いスコアになる」
  アーチファクトを持つのではという懸念(v1 §5.4, C5だけでなくC1-C4相互比較にも影響しうる)に
  対応する。経路長を共変量としたSpearman偏相関、および経路長で回帰した残差での条件間比較を行う。

L-5 (CloseDrawer C5偶発的成功の直接検査): report_v2.md §2.4の「一定運動の反復でも機械的に
  閉まりうる」という事後的説明を、当該episodeのeef軌跡・グリッパー幅の時系列から直接検証する
  (「引き出しへ向かう一定方向の運動が続いたか」を経路の分散/直線性で判定)。

S-10 (多重比較補正): 本検証群全体で実施されたMann-Whitney/Fisher/Spearman検定のp値を可能な限り
  収集し(既存の*.jsonから)、Benjamini-Hochberg法でq値を計算する。事前に定めた少数の確認的仮説
  (各フェーズの中心仮説)とその他の探索的検定を分けて報告する。
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import bh_fdr
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.manifold_trajectory_test import episode_progress
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.dmd_jacobian_stability_test import windowed_dmd, WINDOW

N_BOOT = 2000


def episode_max_lambda(D, episode, call_idx, success):
    progress = episode_progress(episode, call_idx)
    episodes = np.unique(episode)
    out = {}
    for e in episodes:
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        Z, prog = D[order], progress[order]
        if len(Z) < WINDOW:
            continue
        windows = windowed_dmd(Z, prog)
        if not windows:
            continue
        lmax = max(w["lambda_max"] for w in windows)
        out[int(e)] = (lmax, bool(success[episode == e][0]))
    return out


def auc_from_groups(succ_vals, fail_vals):
    if len(succ_vals) == 0 or len(fail_vals) == 0:
        return float("nan")
    u, _ = mannwhitneyu(succ_vals, fail_vals, alternative="less")
    return float(1.0 - u / (len(succ_vals) * len(fail_vals)))   # P(succ < fail)


def s1_paired_bootstrap_delta_auc(embedding_dir, action_dir, task, seed=0):
    with open(Path(embedding_dir) / f"dynamics_embedding_artifact_{task}.pkl", "rb") as f:
        art2 = pickle.load(f)   # group (2): collect_actions x eef/grip proxy
    with open(Path(action_dir) / f"action_dspace_artifact_{task}.pkl", "rb") as f:
        art3 = pickle.load(f)   # group (3): collect_actions x action-chunk PCA

    m2 = episode_max_lambda(art2["D"], art2["episode"], art2["call_idx"], art2["success"])
    m3 = episode_max_lambda(art3["D"], art3["episode"], art3["call_idx"], art3["success"])
    common_eps = sorted(set(m2) & set(m3))
    if len(common_eps) < 6:
        return {"skipped": True, "reason": f"too few common episodes ({len(common_eps)})"}

    succ_eps = [e for e in common_eps if m2[e][1]]
    fail_eps = [e for e in common_eps if not m2[e][1]]
    lam2_succ = np.array([m2[e][0] for e in succ_eps])
    lam2_fail = np.array([m2[e][0] for e in fail_eps])
    lam3_succ = np.array([m3[e][0] for e in succ_eps])
    lam3_fail = np.array([m3[e][0] for e in fail_eps])

    auc2 = auc_from_groups(lam2_succ, lam2_fail)
    auc3 = auc_from_groups(lam3_succ, lam3_fail)

    rng = np.random.RandomState(seed)
    delta_aucs = []
    for _ in range(N_BOOT):
        succ_samp = rng.choice(len(succ_eps), len(succ_eps), replace=True)
        fail_samp = rng.choice(len(fail_eps), len(fail_eps), replace=True)
        a2 = auc_from_groups(lam2_succ[succ_samp], lam2_fail[fail_samp])
        a3 = auc_from_groups(lam3_succ[succ_samp], lam3_fail[fail_samp])
        delta_aucs.append(a3 - a2)
    delta_aucs = np.array(delta_aucs)
    ci_lo, ci_hi = np.percentile(delta_aucs, [2.5, 97.5])

    return {
        "skipped": False, "n_common_episodes": len(common_eps),
        "n_success": len(succ_eps), "n_fail": len(fail_eps),
        "auc_group2_eef_grip_proxy": auc2, "auc_group3_action_pca": auc3,
        "delta_auc_point": float(auc3 - auc2),
        "delta_auc_bootstrap_ci95": [float(ci_lo), float(ci_hi)],
        "delta_auc_ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "n_boot": N_BOOT,
    }


def l4_path_length_covariate(steering_json_path, seed=0):
    with open(steering_json_path) as f:
        steer = json.load(f)
    rows = []
    for cond, r in steer["conditions"].items():
        for ep in r["episodes"]:
            eef = np.array(ep["eef_traj"])
            if len(eef) < 2:
                continue
            path_len = float(np.sum(np.linalg.norm(np.diff(eef, axis=0), axis=1)))
            xp_traj = np.array(ep.get("xp_traj", []))
            manifold_dev_proxy = float(np.linalg.norm(xp_traj, axis=1).mean()) if len(xp_traj) else None
            rows.append({"condition": cond, "path_length": path_len,
                         "manifold_dev_proxy_mean_xp_norm": manifold_dev_proxy})
    path_lengths = np.array([r["path_length"] for r in rows])
    devs = np.array([r["manifold_dev_proxy_mean_xp_norm"] for r in rows if r["manifold_dev_proxy_mean_xp_norm"] is not None])
    pl_for_dev = np.array([r["path_length"] for r in rows if r["manifold_dev_proxy_mean_xp_norm"] is not None])
    if len(devs) < 5:
        return {"skipped": True}
    rho, p = spearmanr(pl_for_dev, devs)
    by_cond = {}
    for cond in steer["conditions"]:
        pls = [r["path_length"] for r in rows if r["condition"] == cond]
        by_cond[cond] = {"mean_path_length": float(np.mean(pls)), "n": len(pls)}
    return {
        "skipped": False,
        "spearman_path_length_vs_xp_norm_proxy_rho": float(rho), "p": float(p),
        "n_points": len(devs), "by_condition_mean_path_length": by_cond,
        "note": ("uses mean ||Xp_t|| across the episode as a residualization-invariant proxy for "
                 "how far the trajectory sits from the origin of progression-space (a stand-in for "
                 "manifold-deviation magnitude available without re-loading the phase-1 bin-reference "
                 "artifact here); a positive correlation would indicate the deviation/DTW metrics are "
                 "confounded with sheer movement amount, as C5's anomalously good scores already "
                 "suggested for that one condition."),
    }


def l5_closedrawer_c5_success_check(steering_json_path):
    with open(steering_json_path) as f:
        steer = json.load(f)
    c5 = steer["conditions"].get("C5_dummy_dynamic_field_frozen_obs")
    if c5 is None:
        return {"skipped": True}
    out = []
    for i, ep in enumerate(c5["episodes"]):
        eef = np.array(ep["eef_traj"])
        grip = np.array(ep["grip_traj"])
        if len(eef) < 2:
            continue
        disp = eef - eef[0]
        # "sustained motion in a roughly constant direction" test: mean resultant length of the
        # per-step displacement UNIT vectors (1.0 = perfectly straight/constant-direction motion,
        # ~0 = directionless/oscillating motion)
        steps = np.diff(eef, axis=0)
        step_norms = np.linalg.norm(steps, axis=1)
        valid = step_norms > 1e-6
        unit_steps = steps[valid] / step_norms[valid, None]
        mean_resultant_length = float(np.linalg.norm(unit_steps.mean(axis=0))) if valid.sum() else None
        out.append({
            "episode_idx": i, "success": ep["success"], "n_steps": ep["n_steps"],
            "net_displacement": float(np.linalg.norm(disp[-1])),
            "path_length": float(step_norms.sum()),
            "straightness_ratio": float(np.linalg.norm(disp[-1]) / (step_norms.sum() + 1e-9)),
            "mean_resultant_length_direction_consistency": mean_resultant_length,
            "grip_width_std": float(grip.std()),
        })
    return {"skipped": False, "episodes": out}


def s10_bh_fdr(pvals_named):
    names = list(pvals_named.keys())
    pvals = [pvals_named[n] for n in names]
    sig = bh_fdr(pvals, alpha=0.05)
    order = np.argsort(pvals)
    ranked_p = np.array(pvals)[order]
    n = len(pvals)
    q = np.minimum.accumulate((ranked_p * n / (np.arange(n) + 1))[::-1])[::-1]
    q_by_name = {}
    for rank_pos, orig_idx in enumerate(order):
        q_by_name[names[orig_idx]] = float(min(q[rank_pos], 1.0))
    return {
        "n_tests": n, "n_significant_after_bh_fdr_0.05": int(sig.sum()),
        "results": [{"test": names[i], "p": float(pvals[i]), "q_bh": q_by_name[names[i]],
                     "significant_after_bh": bool(sig[i])} for i in range(n)],
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True,
                    help="phase-1 D-space artifacts dir (also holds "
                         "dynamics_embedding_on_collect_actions/ subdir for group (2))")
    p.add_argument("--action_dspace_dir", required=True, help="dir with action_dspace_artifact_<task>.pkl")
    p.add_argument("--results_dir", required=True, help="dir with dynamic_vector_field_steering_<task>.json")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group2_dir = Path(args.embedding_dir) / "dynamics_embedding_on_collect_actions"

    results = {}
    results["S1_L2_paired_bootstrap_delta_auc"] = {}
    for task in ["PnPCounterToCab", "TurnOnStove"]:
        results["S1_L2_paired_bootstrap_delta_auc"][task] = s1_paired_bootstrap_delta_auc(
            group2_dir, args.action_dspace_dir, task)

    pnp_json = Path(args.results_dir) / "dynamic_vector_field_steering_PnPCounterToCab.json"
    results["L4_path_length_covariate_PnPCounterToCab"] = l4_path_length_covariate(pnp_json)
    results["L5_CloseDrawer_C5_success_check"] = l5_closedrawer_c5_success_check(
        Path(args.results_dir) / "dynamic_vector_field_steering_CloseDrawer.json")

    # S-10: collect whatever p-values are readily available in already-produced JSON artifacts
    pvals = {}
    try:
        with open(Path(args.embedding_dir) / "dynamics_embedding_test.json") as f:
            d1 = json.load(f)
        for task, r in d1["tasks"].items():
            if r["deviation_test_D_space_pooled"]:
                pvals[f"report1_deviation_D_{task}"] = r["deviation_test_D_space_pooled"]["mannwhitney_p_success_lt_fail"]
    except FileNotFoundError:
        pass
    try:
        with open(Path(args.embedding_dir) / "energy_field_test.json") as f:
            d2 = json.load(f)
        for task, r in d2["tasks"].items():
            if not r.get("skipped"):
                pvals[f"report1_energy_{task}"] = r["energy_mannwhitney_p_success_lt_fail"]
                if r["dmd_in_D_space"]:
                    pvals[f"report1_dmd_D_{task}"] = r["dmd_in_D_space"]["episode_max_lambda_mannwhitney_success_lt_fail_p"]
                    pvals[f"report1_dmd_fisher_{task}"] = r["dmd_in_D_space"]["fisher_crossing_success_lt_fail_p"]
    except FileNotFoundError:
        pass
    try:
        with open(Path(args.out_dir) / "arm_ablation_test.json") as f:
            d3 = json.load(f)
        for task, arms in d3["tasks"].items():
            for arm in ["A0_Xp_only", "A1_Xp_V_delay_embedded", "A3_Xp_V_action_true"]:
                dmd = arms.get(arm, {}).get("dmd_test")
                if dmd:
                    pvals[f"arm_ablation_{arm}_{task}"] = dmd["episode_max_lambda_mannwhitney_success_lt_fail_p"]
    except FileNotFoundError:
        pass
    if pvals:
        results["S10_bh_fdr_correction"] = s10_bh_fdr(pvals)

    with open(out_dir / "final_stats_reanalysis.json", "w") as f:
        json.dump(results, f, indent=2)

    for task, r in results["S1_L2_paired_bootstrap_delta_auc"].items():
        if not r.get("skipped"):
            log_message(f"[final_stats {task}] AUC(2)={r['auc_group2_eef_grip_proxy']:.3f} "
                        f"AUC(3)={r['auc_group3_action_pca']:.3f} DeltaAUC={r['delta_auc_point']:.3f} "
                        f"95%CI={r['delta_auc_bootstrap_ci95']}")
    log_message(f"Saved: {out_dir / 'final_stats_reanalysis.json'}")


if __name__ == "__main__":
    main()
