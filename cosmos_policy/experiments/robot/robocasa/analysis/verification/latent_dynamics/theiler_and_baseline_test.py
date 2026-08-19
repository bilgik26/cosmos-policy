"""
theiler_and_baseline_test.py — review_report.md (latent_dynamics_verification 査読) M-6/S-7・M-7
対応。GPU不要、フェーズ1・2の既存artifactの再解析のみ。

M-6/S-7 (相関次元のTheiler window補正): §2.3の相関次元推定(Grassberger-Procaccia)は、時間的に
  近接したペアを除外するTheiler windowを設けていなかった。連続するcallの点は自明に近接しており
  (特にtau=3の遅延埋め込みで連続するD_tが成分を共有するため、この近接性はさらに強い)、
  これを除外しないと相関積分が小さいepsilon側で人工的に膨らみ、次元が系統的に過小推定される。
  本スクリプトは全点対距離(ランダム部分サンプルではなく)から、同一エピソード内で
  |call_idx_i - call_idx_j| <= window のペアを除外して相関積分を再計算し、
  window in {0(補正なし,元の推定に相当), tau, 2*tau, 5, 10} で感度分析する。

M-7 (KDEエネルギー場 vs 1-NN距離ベースラインの比較): D空間のKDEエネルギーによる成功/失敗分離が、
  自明なベースライン(失敗episodeの各callから成功エピソード群への最近傍ユークリッド距離)を
  AUCで上回るかを検定する。上回らなければ「エネルギー場」という概念装置は本検証固有の付加価値を
  持たないことになる(§3.4での自己言及的懸念「§5.6の逸脱量検定と本質的に同じ情報を測っている
  可能性」への直接対応)。
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamics_embedding_test import WINDOW_TAU
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.energy_field_test import GaussianKDEField
from sklearn.preprocessing import StandardScaler

THEILER_WINDOWS = [0, WINDOW_TAU, 2 * WINDOW_TAU, 5, 10]
MAX_POINTS = 900   # cap for O(N^2) distance matrix memory/time


def correlation_dimension_theiler(X, episode, call_idx, window, seed=0):
    Xs = StandardScaler().fit_transform(X)
    n = len(Xs)
    rng = np.random.RandomState(seed)
    if n > MAX_POINTS:
        idx = rng.choice(n, MAX_POINTS, replace=False)
        Xs, episode, call_idx = Xs[idx], episode[idx], call_idx[idx]
        n = MAX_POINTS

    diff = Xs[:, None, :] - Xs[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    same_ep = episode[:, None] == episode[None, :]
    call_gap = np.abs(call_idx[:, None] - call_idx[None, :])
    exclude = same_ep & (call_gap <= window)
    iu = np.triu_indices(n, k=1)
    keep = ~exclude[iu]
    d = dist[iu][keep]
    d = d[d > 1e-12]
    if len(d) < 50:
        return {"skipped": True, "reason": "too few surviving pairs after Theiler exclusion"}

    r = np.logspace(np.log10(np.percentile(d, 5)), np.log10(np.percentile(d, 50)), 12)
    C = np.array([(d < ri).mean() for ri in r])
    valid = C > 0
    logr, logC = np.log(r[valid]), np.log(C[valid])
    if len(logr) < 3:
        return {"skipped": True, "reason": "too few valid radii"}
    slope, intercept = np.polyfit(logr, logC, 1)
    pred = slope * logr + intercept
    ss_res = np.sum((logC - pred) ** 2)
    ss_tot = np.sum((logC - logC.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {
        "skipped": False, "theiler_window": window, "n_points_used": n,
        "n_surviving_pairs": int(keep.sum()), "frac_pairs_excluded": float(1 - keep.mean()),
        "correlation_dimension": float(slope), "fit_r2": r2,
    }


def m7_kde_vs_1nn_baseline(D, episode, call_idx, success, seed=0):
    from sklearn.model_selection import GroupKFold
    episodes = np.unique(episode)
    ep_success = {e: bool(success[episode == e][0]) for e in episodes}
    success_episodes = np.array([e for e in episodes if ep_success[e]])
    fail_episodes = np.array([e for e in episodes if not ep_success[e]])
    if len(fail_episodes) < 3:
        return {"skipped": True}

    mask_succ = np.isin(episode, success_episodes)
    D_s, ep_s = D[mask_succ], episode[mask_succ]
    idx_map = np.where(mask_succ)[0]
    n_splits = min(5, len(success_episodes))
    gkf = GroupKFold(n_splits=n_splits)
    energy_out = np.full(len(episode), np.nan)
    nn1_out = np.full(len(episode), np.nan)
    for train_idx, test_idx in gkf.split(D_s, groups=ep_s):
        kde = GaussianKDEField(D_s[train_idx])
        energy_out[idx_map[test_idx]] = kde.energy(D_s[test_idx])
        train_pts = D_s[train_idx]
        test_pts = D_s[test_idx]
        d = np.linalg.norm(test_pts[:, None, :] - train_pts[None, :, :], axis=2)
        nn1_out[idx_map[test_idx]] = d.min(axis=1)

    mask_fail = np.isin(episode, fail_episodes)
    kde_full = GaussianKDEField(D[mask_succ])
    energy_out[mask_fail] = kde_full.energy(D[mask_fail])
    nn1_out[mask_fail] = np.linalg.norm(
        D[mask_fail][:, None, :] - D[mask_succ][None, :, :], axis=2).min(axis=1)

    succ_e, fail_e = energy_out[mask_succ], energy_out[mask_fail]
    succ_n, fail_n = nn1_out[mask_succ], nn1_out[mask_fail]
    u_e, p_e = mannwhitneyu(succ_e, fail_e, alternative="less")
    u_n, p_n = mannwhitneyu(succ_n, fail_n, alternative="less")
    # scipy's mannwhitneyu(succ, fail, alternative="less") returns U = U_succ, i.e. a
    # count-based estimator of P(succ_score > fail_score). Separation AUC in the direction we
    # actually care about (succ score LOWER than fail, i.e. P(succ < fail)) is its complement.
    auc_kde = float(1.0 - u_e / (len(succ_e) * len(fail_e)))
    auc_1nn = float(1.0 - u_n / (len(succ_n) * len(fail_n)))

    rng = np.random.RandomState(seed)
    n_perm = 2000
    labels = np.concatenate([np.ones(len(succ_e)), np.zeros(len(fail_e))])
    all_e = np.concatenate([succ_e, fail_e])
    null_aucs = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        s = all_e[perm_labels == 1]
        fa = all_e[perm_labels == 0]
        u, _ = mannwhitneyu(s, fa, alternative="less")
        null_aucs.append(u / (len(s) * len(fa)))
    null_aucs = np.array(null_aucs)
    perm_p_kde = float((np.sum(null_aucs <= auc_kde) + 1) / (n_perm + 1))

    return {
        "skipped": False, "n_success_calls": len(succ_e), "n_fail_calls": len(fail_e),
        "kde_energy_mannwhitney_p": float(p_e), "kde_energy_auc": auc_kde,
        "one_nn_distance_mannwhitney_p": float(p_n), "one_nn_distance_auc": auc_1nn,
        "kde_beats_1nn_baseline": bool(auc_kde > auc_1nn),
        "kde_auc_vs_baseline_diff": float(auc_kde - auc_1nn),
        "kde_auc_permutation_null_p": perm_p_kde,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"theiler_windows_tested": THEILER_WINDOWS, "tasks": {}}
    for art_path in sorted(embedding_dir.glob("dynamics_embedding_artifact_*.pkl")):
        task = art_path.stem.replace("dynamics_embedding_artifact_", "")
        with open(art_path, "rb") as f:
            art = pickle.load(f)
        D, Xp, episode, call_idx, success = art["D"], art["Xp"], art["episode"], art["call_idx"], art["success"]

        theiler_D = [correlation_dimension_theiler(D, episode, call_idx, w) for w in THEILER_WINDOWS]
        theiler_Xp = [correlation_dimension_theiler(Xp, episode, call_idx, w) for w in THEILER_WINDOWS]
        m7 = m7_kde_vs_1nn_baseline(D, episode, call_idx, success)

        results["tasks"][task] = {
            "correlation_dimension_theiler_sensitivity_D_space": theiler_D,
            "correlation_dimension_theiler_sensitivity_Xp_space": theiler_Xp,
            "M7_kde_energy_vs_1nn_baseline": m7,
        }
        d0 = theiler_D[0].get("correlation_dimension") if not theiler_D[0].get("skipped") else float("nan")
        dmax = theiler_D[-1].get("correlation_dimension") if not theiler_D[-1].get("skipped") else float("nan")
        log_message(f"[theiler_baseline {task}] corr_dim D-space: window=0 -> {d0:.2f}, "
                    f"window={THEILER_WINDOWS[-1]} -> {dmax:.2f} | "
                    f"M7 KDE AUC={m7.get('kde_energy_auc') if not m7.get('skipped') else 'NA'} "
                    f"vs 1NN AUC={m7.get('one_nn_distance_auc') if not m7.get('skipped') else 'NA'}")

    with open(out_dir / "theiler_and_baseline_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'theiler_and_baseline_test.json'}")


if __name__ == "__main__":
    main()
