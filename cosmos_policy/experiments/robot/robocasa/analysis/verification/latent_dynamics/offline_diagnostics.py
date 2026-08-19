"""
offline_diagnostics.py — review_report.md (latent_dynamics_verification 査読) S-2/S-3/S-4/S-5/M-4
対応。フェーズ3(dynamic_vector_field_steering.py)が既に保存済みのロールアウトログ
(`dynamic_vector_field_steering_<task>.json`: 各episodeのxp_traj/d_traj/success)と、フェーズ1の
D空間artifact(`dynamics_embedding_artifact_<task>.pkl`)だけから、GPU不要で4つの診断を行う
(いずれも既存の保存済み数値の再解析であり、新規ロールアウトは不要)。

S-2 (オンライン/オフライン多様体逸脱度の矛盾診断): 「オフラインでは完全分離する逸脱量検定が、
  オンラインではC0の成功/失敗を全く区別できない」というreport.md §5.3の内部矛盾を診断する。
  C0の8episodeを成功/失敗で分割し、オンラインxp_trajに対して§2.3と同一の逸脱量検定を再実行する。
  分離しなければ、featurization(残差化方式の違い: エピソード平均 vs 累積平均)か進行度定義の
  破綻が疑われる(§7-B-2)。

S-3 (注入ベクトルの成分集中度診断): Delta_X_rawの participation ratio・top-1/top-5エネルギー占有率
  を計算し、2048次元一様ランダム方向の理論値(有効次元)と比較する。C2(動的フィールド)全episode・
  全callについて field.query()+xp_direction_to_rawをオフラインで再計算する(オンライン実行時と
  厳密に同一の関数、xp_traj/d_trajは保存済みなのでGPU再ロールアウト不要)。

S-4/M-1 (steering方向の時間変動診断): cos(v_t, v_{t+1})・cos(v_t, mean_t(v_t))を計算し、C2が
  実質的に静的(C3相当)になっていないかを検証する。あわせてC2とC3(固定v_steer)の注入方向間の
  平均コサイン類似度も報告する。

S-5/M-4 (kNNライブラリに対するOOD診断): 各条件・各callのD_tについて、DynamicFlowFieldのライブラリ
  点集合に対するkNN距離を計算し、ライブラリ自身の内部最近傍距離分布の分位点と比較する。
  C0(実プロンプト)とC1〜C5(ダミープロンプト系)でこの分布がどれだけ異なるかを見る。
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.manifold_trajectory_test import episode_progress
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.success_failure_trajectory_test import (
    fit_bin_reference, deviation_from_reference, held_out_success_deviation, episode_mean,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DynamicFlowField, xp_direction_to_raw,
)


def load_steering_json(results_dir, task):
    with open(Path(results_dir) / f"dynamic_vector_field_steering_{task}.json") as f:
        return json.load(f)


def load_dyn_artifact(results_dir, task):
    with open(Path(results_dir) / f"dynamics_embedding_artifact_{task}.pkl", "rb") as f:
        return pickle.load(f)


# ─────────────────────────── S-2 ───────────────────────────

def s2_online_offline_deviation(results_dir, task, seed=0):
    art = load_dyn_artifact(results_dir, task)
    steer = load_steering_json(results_dir, task)
    c0 = steer["conditions"].get("C0_real_prompt_no_steer")
    if c0 is None:
        return None
    succ_idx = [i for i, e in enumerate(c0["episodes"]) if e["success"]]
    fail_idx = [i for i, e in enumerate(c0["episodes"]) if not e["success"]]
    if len(succ_idx) < 2 or len(fail_idx) < 2:
        return {"skipped": True, "reason": f"too few per group in C0 (succ={len(succ_idx)}, fail={len(fail_idx)})"}

    # build a pseudo-episode/call_idx/Xp array over ALL C0 online xp_traj points
    Xp_all, ep_all, call_all, succ_flag_all = [], [], [], []
    for i, ep in enumerate(c0["episodes"]):
        xp = np.array(ep["xp_traj"])
        if len(xp) == 0:
            continue
        Xp_all.append(xp)
        ep_all.append(np.full(len(xp), i))
        call_all.append(np.array(ep["xp_call_idx"]))
        succ_flag_all.append(np.full(len(xp), ep["success"]))
    Xp_all = np.concatenate(Xp_all)
    ep_all = np.concatenate(ep_all)
    call_all = np.concatenate(call_all)
    succ_flag_all = np.concatenate(succ_flag_all)
    progress = episode_progress(ep_all, call_all)

    succ_eps = np.unique(ep_all[succ_flag_all])
    fail_eps = np.unique(ep_all[~succ_flag_all])
    dev_succ_heldout = held_out_success_deviation(Xp_all, ep_all, progress, succ_eps, seed=seed)
    mask_succ = np.isin(ep_all, succ_eps)
    centers, variances = fit_bin_reference(Xp_all[mask_succ], progress[mask_succ])
    mask_fail = np.isin(ep_all, fail_eps)
    dev_fail = np.full(len(ep_all), np.nan)
    dev_fail[mask_fail] = deviation_from_reference(Xp_all[mask_fail], progress[mask_fail], centers, variances)

    succ_vals = np.array(list(episode_mean(dev_succ_heldout, ep_all, succ_eps).values()))
    fail_vals = np.array(list(episode_mean(dev_fail, ep_all, fail_eps).values()))
    u, p = mannwhitneyu(succ_vals, fail_vals, alternative="less") if len(succ_vals) and len(fail_vals) else (np.nan, np.nan)

    # also: per-dimension mean/var comparison of the ONLINE Xp vs the OFFLINE (training) Xp
    offline_Xp = art["Xp"]
    online_mean, online_std = Xp_all.mean(axis=0), Xp_all.std(axis=0)
    offline_mean, offline_std = offline_Xp.mean(axis=0), offline_Xp.std(axis=0)

    return {
        "skipped": False,
        "n_success_episodes": len(succ_eps), "n_fail_episodes": len(fail_eps),
        "online_C0_deviation_mannwhitney_p_success_lt_fail": float(p),
        "online_succ_dev_mean": float(succ_vals.mean()), "online_fail_dev_mean": float(fail_vals.mean()),
        "per_dim_mean_online": online_mean.tolist(), "per_dim_mean_offline_training": offline_mean.tolist(),
        "per_dim_std_online": online_std.tolist(), "per_dim_std_offline_training": offline_std.tolist(),
        "per_dim_mean_abs_diff": float(np.mean(np.abs(online_mean - offline_mean))),
        "per_dim_std_ratio_online_over_offline": (online_std / (offline_std + 1e-12)).tolist(),
    }


# ─────────────────────────── S-3 ───────────────────────────

def concentration_stats(vec):
    v2 = vec ** 2
    pr = float((v2.sum() ** 2) / (v2 ** 2).sum())   # participation ratio (effective # of active dims)
    order = np.argsort(-v2)
    total = v2.sum()
    top1 = float(v2[order[0]] / total)
    top5 = float(v2[order[:5]].sum() / total)
    return pr, top1, top5


def s3_injection_concentration(results_dir, task):
    art = load_dyn_artifact(results_dir, task)
    steer = load_steering_json(results_dir, task)
    c2 = steer["conditions"]["C2_dummy_dynamic_field"]
    field = DynamicFlowField(art, randomize=False, seed=0)

    prs, top1s, top5s = [], [], []
    for ep in c2["episodes"]:
        xp_traj, d_traj = np.array(ep["xp_traj"]), np.array(ep["d_traj"])
        for D_t, Xp_t in zip(d_traj, xp_traj):
            v_xp = field.query(D_t, Xp_t)
            raw_dir = xp_direction_to_raw(v_xp, art)
            pr, top1, top5 = concentration_stats(raw_dir)
            prs.append(pr)
            top1s.append(top1)
            top5s.append(top5)
    prs, top1s, top5s = np.array(prs), np.array(top1s), np.array(top5s)

    ambient_scale = art["ambient_scaler"].scale_
    scale_ratio = float(ambient_scale.max() / np.median(ambient_scale))

    n_dims = art["prog_pca"].components_.shape[1]
    return {
        "n_calls": len(prs), "raw_dir_dim": n_dims,
        "participation_ratio_mean": float(prs.mean()), "participation_ratio_median": float(np.median(prs)),
        "uniform_random_direction_expected_participation_ratio": float(n_dims),
        "top1_energy_share_mean": float(top1s.mean()), "top5_energy_share_mean": float(top5s.mean()),
        "uniform_random_direction_expected_top1_energy_share": float(1.0 / n_dims),
        "ambient_scaler_scale_max_over_median": scale_ratio,
    }


# ─────────────────────────── S-4 / M-1 ───────────────────────────

def s4_temporal_variation(results_dir, task):
    art = load_dyn_artifact(results_dir, task)
    steer = load_steering_json(results_dir, task)
    field = DynamicFlowField(art, randomize=False, seed=0)
    c2 = steer["conditions"]["C2_dummy_dynamic_field"]

    cos_consecutive_all, cos_to_mean_all = [], []
    c2_dirs_by_ep = []
    for ep in c2["episodes"]:
        xp_traj, d_traj = np.array(ep["xp_traj"]), np.array(ep["d_traj"])
        dirs = []
        for D_t, Xp_t in zip(d_traj, xp_traj):
            v_xp = field.query(D_t, Xp_t)
            raw_dir = xp_direction_to_raw(v_xp, art)
            dirs.append(raw_dir / (np.linalg.norm(raw_dir) + 1e-12))
        dirs = np.array(dirs)
        c2_dirs_by_ep.append(dirs)
        if len(dirs) >= 2:
            cos_consecutive_all.extend((dirs[:-1] * dirs[1:]).sum(axis=1).tolist())
        mean_dir = dirs.mean(axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
        cos_to_mean_all.extend((dirs * mean_dir[None, :]).sum(axis=1).tolist())

    # C2 vs C3 (static v_steer) direction similarity, if C3 vec is recoverable: C3 uses a FIXED
    # unit vector for the whole episode (static_vec_unit in dynamic_vector_field_steering.main());
    # we cannot recover that exact vector from the saved JSON (it's not logged per-call), so we
    # report only the C2-internal temporal-variation diagnostics here, which alone are sufficient
    # to answer "is C2 effectively static" (M-1's core question).
    return {
        "n_calls_with_consecutive_pair": len(cos_consecutive_all),
        "cos_consecutive_mean": float(np.mean(cos_consecutive_all)) if cos_consecutive_all else None,
        "cos_consecutive_median": float(np.median(cos_consecutive_all)) if cos_consecutive_all else None,
        "cos_consecutive_q10": float(np.percentile(cos_consecutive_all, 10)) if cos_consecutive_all else None,
        "cos_to_episode_mean_direction_mean": float(np.mean(cos_to_mean_all)) if cos_to_mean_all else None,
        "note": ("C3's fixed v_steer direction is not logged per-call in the saved rollout JSON "
                 "(it is set once per episode and never re-read), so a direct C2-vs-C3 vector "
                 "comparison could not be reconstructed post-hoc from existing artifacts; the "
                 "within-C2 temporal-variation statistics above directly answer M-1's question of "
                 "whether C2 is effectively static (mean(cos_consecutive) > 0.95 would indicate so)."),
    }


# ─────────────────────────── S-5 / M-4 ───────────────────────────

def s5_knn_ood(results_dir, task):
    art = load_dyn_artifact(results_dir, task)
    steer = load_steering_json(results_dir, task)
    field = DynamicFlowField(art, randomize=False, seed=0)

    # library's own internal nearest-neighbor distance distribution (leave-one-out 1-NN)
    D_pts = field.D_pts
    n = len(D_pts)
    sub_n = min(n, 800)
    rng = np.random.RandomState(0)
    sub_idx = rng.choice(n, sub_n, replace=False)
    internal_nn = []
    for i in sub_idx:
        d = np.linalg.norm(D_pts - D_pts[i][None, :], axis=1)
        d[i] = np.inf
        internal_nn.append(d.min())
    internal_nn = np.array(internal_nn)

    out = {"library_internal_1nn_distance": {
        "mean": float(internal_nn.mean()), "median": float(np.median(internal_nn)),
        "q90": float(np.percentile(internal_nn, 90)),
    }, "conditions": {}}

    for cond, r in steer["conditions"].items():
        dists = []
        for ep in r["episodes"]:
            for D_t in ep["d_traj"]:
                D_t = np.array(D_t)
                d = np.linalg.norm(D_pts - D_t[None, :], axis=1)
                dists.append(d.min())
        if not dists:
            continue
        dists = np.array(dists)
        out["conditions"][cond] = {
            "n_calls": len(dists), "mean_1nn_to_library": float(dists.mean()),
            "median_1nn_to_library": float(np.median(dists)),
            "q90_1nn_to_library": float(np.percentile(dists, 90)),
            "frac_beyond_library_internal_q90": float((dists > np.percentile(internal_nn, 90)).mean()),
        }
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                    help="latent_dynamics_verification results dir (has both "
                         "dynamics_embedding_artifact_<task>.pkl and "
                         "dynamic_vector_field_steering_<task>.json)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=["PnPCounterToCab"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"tasks": {}}
    for task in args.tasks:
        results["tasks"][task] = {
            "S2_online_offline_deviation_contradiction": s2_online_offline_deviation(args.results_dir, task),
            "S3_injection_vector_concentration": s3_injection_concentration(args.results_dir, task),
            "S4_temporal_variation_of_steering_direction": s4_temporal_variation(args.results_dir, task),
            "S5_M4_knn_ood_diagnostic": s5_knn_ood(args.results_dir, task),
        }
        s2 = results["tasks"][task]["S2_online_offline_deviation_contradiction"]
        s3 = results["tasks"][task]["S3_injection_vector_concentration"]
        s4 = results["tasks"][task]["S4_temporal_variation_of_steering_direction"]
        log_message(f"[offline_diagnostics {task}] S2 online C0 deviation p="
                    f"{s2.get('online_C0_deviation_mannwhitney_p_success_lt_fail') if s2 and not s2.get('skipped') else 'skipped'} | "
                    f"S3 participation_ratio_mean={s3['participation_ratio_mean']:.2f} (uniform-random expected={s3['uniform_random_direction_expected_participation_ratio']:.0f}) "
                    f"top1_energy={s3['top1_energy_share_mean']:.3f} | "
                    f"S4 cos_consecutive_mean={s4['cos_consecutive_mean']:.3f}")

    with open(out_dir / "offline_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'offline_diagnostics.json'}")


if __name__ == "__main__":
    main()
