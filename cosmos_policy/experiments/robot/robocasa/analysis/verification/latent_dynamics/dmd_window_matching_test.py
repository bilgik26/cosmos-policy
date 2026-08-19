"""
dmd_window_matching_test.py — review_report.md (latent_dynamics_verification 査読) MUST-5 対応。

review_report.md §3 B-5 / §2 M-9 が指摘した2つの交絡へ対応する:

  B-5 (窓数交絡): windowed DMDの `max|lambda|` はエピソード内の全窓にわたる最大値であり、
    極値統計として窓数が多いほど系統的に大きくなる。RoboCasaはタスク成功時点でエピソードを
    即終了するため、成功エピソードは短く(窓数少)・失敗エピソードはタイムアウト上限まで走る
    (窓数多)— つまり「失敗の方がmax|lambda|が大きい」という中心的知見は、力学的不安定性を
    一切仮定せずとも極値統計だけで生じうる。

  M-9 (遅延埋め込みのシフト演算子構造): D_tがtau=3の遅延窓であるため連続するD_tは成分を共有し、
    windowed DMDが推定する線形写像Aの一部は真の力学ではなく遅延埋め込み自体が持つシフト演算子
    構造を推定している可能性がある。シフト演算子の固有値は単位円上にあるため、|lambda|~1近傍への
    系統的バイアスが生じうる。

対応:
  (1) 窓数マッチング: 成功エピソードの窓数分布に合わせ、失敗エピソードから連続窓の部分列を
      繰り返しサブサンプル(N_REPLICATES回)し、その都度Mann-Whitney検定をやり直して p値の分布
      (中央値・四分位範囲・p<0.05の頻度)を報告する。単一のp値ではなく分布全体を報告することで、
      「窓数を揃えたら消える効果なのか、揃えても残る効果なのか」を定量化する。
  (2) シフト演算子帰無分布: 真の力学的構造を一切持たない代理データ(各次元が独立な標準正規分布
      駆動のランダムウォーク、実データと同じエピソード長分布・同じtau・同じD_dim)に対して同一の
      windowed DMDパイプラインを適用し、`max|lambda|`の帰無分布・`|lambda|>1`到達率を得る。
      これは「遅延埋め込みの窓構造それ自体が、真の力学なしにどれだけの見かけの不安定性を生むか」
      の直接的な定量化であり、固定閾値1.0の代わりにこの帰無分布の分位点を使うべきという査読の
      要求に対応する。

対象空間: report.md §3.3のD空間(dynamics_embedding_artifact, eef/gripper代理指標版)と、
arm_ablation_test.pyのA0(Xp)/A1(delay-embed only)/A3(true action block)の3アーム
(collect_actions上、MUST-1と同一サンプル)。A1で観測される強い有意性がどこまでシフト演算子
アーチファクトで説明されるかを直接検証できる。
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import scene_residualize
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.manifold_trajectory_test import episode_progress
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.success_failure_trajectory_test import MIN_FAIL_EPISODES
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.dmd_jacobian_stability_test import windowed_dmd, WINDOW
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamics_embedding_test import (
    PROGRESS_PCA_DIM, WINDOW_TAU, D_DIM, build_delay_embedding_input,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.action_dspace_test import (
    A_DIM, load_pooled_task_data_actions,
)

N_REPLICATES = 500
N_NULL_EPISODES = 60   # per (task-like) synthetic null batch


def episode_windows(D, episode, call_idx, progress):
    """Return {episode_id: [{'lambda_max':..., ...}, ...]} windowed-DMD dicts per episode."""
    out = {}
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        Z, prog = D[order], progress[order]
        if len(Z) < WINDOW:
            continue
        out[int(e)] = windowed_dmd(Z, prog)
    return out


def window_count_matched_test(D, episode, call_idx, success, progress, seed=0):
    episodes = np.unique(episode)
    ep_success = {e: bool(success[episode == e][0]) for e in episodes}
    success_eps = [e for e in episodes if ep_success[e]]
    fail_eps = [e for e in episodes if not ep_success[e]]
    if len(fail_eps) < MIN_FAIL_EPISODES:
        return None

    win_by_ep = episode_windows(D, episode, call_idx, progress)
    succ_windows = {e: win_by_ep[e] for e in success_eps if e in win_by_ep}
    fail_windows = {e: win_by_ep[e] for e in fail_eps if e in win_by_ep}
    if not succ_windows or not fail_windows:
        return None

    succ_n = np.array([len(w) for w in succ_windows.values()])
    fail_n = np.array([len(w) for w in fail_windows.values()])
    target_n = max(1, int(round(np.median(succ_n))))

    succ_max_unmatched = np.array([max(w["lambda_max"] for w in ws) for ws in succ_windows.values()])
    u_orig, p_orig = mannwhitneyu(
        succ_max_unmatched, [max(w["lambda_max"] for w in ws) for ws in fail_windows.values()],
        alternative="less")

    rng = np.random.RandomState(seed)
    p_dist, auc_dist = [], []
    for _ in range(N_REPLICATES):
        fail_max_matched = []
        for ws in fail_windows.values():
            lambdas = np.array([w["lambda_max"] for w in ws])
            if len(lambdas) <= target_n:
                fail_max_matched.append(lambdas.max())
            else:
                start = rng.randint(0, len(lambdas) - target_n + 1)
                fail_max_matched.append(lambdas[start:start + target_n].max())
        u, p = mannwhitneyu(succ_max_unmatched, fail_max_matched, alternative="less")
        p_dist.append(p)
        auc_dist.append(u / (len(succ_max_unmatched) * len(fail_max_matched)))
    p_dist, auc_dist = np.array(p_dist), np.array(auc_dist)

    return {
        "n_success_episodes": len(succ_windows), "n_fail_episodes": len(fail_windows),
        "mean_n_windows_success": float(succ_n.mean()), "mean_n_windows_fail": float(fail_n.mean()),
        "target_n_windows_matched_to": target_n,
        "unmatched_mannwhitney_p": float(p_orig),
        "matched_p_median": float(np.median(p_dist)), "matched_p_q25": float(np.percentile(p_dist, 25)),
        "matched_p_q75": float(np.percentile(p_dist, 75)),
        "matched_frac_p_lt_0.05": float((p_dist < 0.05).mean()),
        "matched_auc_median": float(np.median(auc_dist)),
        "n_replicates": N_REPLICATES,
    }


def shift_operator_null(episode_lengths, c_dim, tau, d_dim, seed=0):
    """Surrogate D-space with NO genuine dynamical structure: c_dim-dim independent standard
    Gaussian random walks (one per synthetic episode, length drawn from the REAL episode-length
    distribution so window-count statistics match), delay-embedded with the SAME tau, then
    projected through a PCA(d_dim) fit fresh on the pooled surrogate windows (mirrors the real
    pipeline exactly except the input is pure noise instead of real Blk-13-derived features).
    Returns the pooled max|lambda| null distribution and crossing rate."""
    rng = np.random.RandomState(seed)
    c_list, ep_list, call_list = [], [], []
    for e in range(N_NULL_EPISODES):
        T = int(rng.choice(episode_lengths))
        T = max(T, WINDOW + 1)
        steps = rng.randn(T, c_dim)
        c_ep = np.cumsum(steps, axis=0)   # pure random walk, no self-stabilization/destabilization
        c_list.append(c_ep)
        ep_list.append(np.full(T, e))
        call_list.append(np.arange(T))
    c = np.concatenate(c_list)
    episode = np.concatenate(ep_list)
    call_idx = np.concatenate(call_list)
    c = StandardScaler().fit_transform(c)
    S = build_delay_embedding_input(c, episode, call_idx, tau=tau)
    pca = PCA(n_components=min(d_dim, S.shape[0] - 1, S.shape[1]), random_state=seed)
    D = pca.fit_transform(S)
    progress = episode_progress(episode, call_idx)

    all_lmax, n_crossed, n_eps_used = [], 0, 0
    for e in range(N_NULL_EPISODES):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        Z, prog = D[order], progress[order]
        if len(Z) < WINDOW:
            continue
        windows = windowed_dmd(Z, prog)
        if not windows:
            continue
        lmax = np.array([w["lambda_max"] for w in windows])
        all_lmax.extend(lmax.tolist())
        n_crossed += int((lmax > 1.0).any())
        n_eps_used += 1
    all_lmax = np.array(all_lmax)
    return {
        "n_null_episodes_used": n_eps_used, "n_null_windows": len(all_lmax),
        "null_max_lambda_mean": float(all_lmax.mean()), "null_max_lambda_median": float(np.median(all_lmax)),
        "null_max_lambda_q90": float(np.percentile(all_lmax, 90)),
        "null_max_lambda_q99": float(np.percentile(all_lmax, 99)),
        "null_frac_windows_gt_1": float((all_lmax > 1.0).mean()),
        "null_frac_episodes_ever_crossing_1": float(n_crossed / max(n_eps_used, 1)),
    }


def analyze_space(D, episode, call_idx, success, progress, c_dim_for_null, tau, d_dim, seed=0):
    matched = window_count_matched_test(D, episode, call_idx, success, progress, seed=seed)
    if matched is None:
        return {"skipped": True}
    episodes = np.unique(episode)
    ep_lengths = np.array([np.sum(episode == e) for e in episodes])
    null = shift_operator_null(ep_lengths, c_dim_for_null, tau, d_dim, seed=seed)
    return {"skipped": False, "window_count_matched_test": matched, "shift_operator_surrogate_null": null}


def analyze_task_report1_dspace(embedding_dir, task, seed=0):
    """Original report.md D-space (eef/gripper proxy, collect_v2-derived artifact)."""
    art_path = Path(embedding_dir) / f"dynamics_embedding_artifact_{task}.pkl"
    if not art_path.exists():
        return None
    with open(art_path, "rb") as f:
        art = pickle.load(f)
    D, episode, call_idx, success = art["D"], art["episode"], art["call_idx"], art["success"]
    progress = episode_progress(episode, call_idx)
    c_dim = 24   # [Xp(10), V(10), Delta_eef(3), Delta_grip(1)]
    return analyze_space(D, episode, call_idx, success, progress, c_dim, art["window_tau"],
                          D_DIM, seed=seed)


def analyze_task_arms(collect_dir, manifest, task, seed=0):
    """A0 (Xp, no delay embed)/A1 (Xp+V delay-embedded)/A3 (Xp+V+A_PCA true) on collect_actions,
    matching MUST-1's arm_ablation_test.py construction exactly (same sample as MUST-1 so the two
    analyses are directly comparable)."""
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

    out = {}
    out["A0_Xp_only"] = analyze_space(Xp, episode, call_idx, success_call, progress, 10, 1, 10, seed=seed)

    def build_D(c_block, d_dim):
        c_scaler = StandardScaler().fit(c_block)
        c = c_scaler.transform(c_block)
        S = build_delay_embedding_input(c, episode, call_idx, tau=WINDOW_TAU)
        pca = PCA(n_components=min(d_dim, S.shape[0] - 1, S.shape[1]), random_state=seed)
        return pca.fit_transform(S)

    c_A1 = np.concatenate([Xp, V], axis=1)
    D_A1 = build_D(c_A1, D_DIM)
    out["A1_Xp_V_delay_embedded"] = analyze_space(D_A1, episode, call_idx, success_call, progress,
                                                    c_A1.shape[1], WINDOW_TAU, D_DIM, seed=seed)

    c_A3 = np.concatenate([Xp, V, A_pca], axis=1)
    D_A3 = build_D(c_A3, D_DIM)
    out["A3_Xp_V_action_true"] = analyze_space(D_A3, episode, call_idx, success_call, progress,
                                                 c_A3.shape[1], WINDOW_TAU, D_DIM, seed=seed)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True, help="report.md's phase-1 D-space artifacts dir")
    p.add_argument("--collect_dir", required=True, help="collect_action_chunks.py output (for the arm spaces)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.collect_dir) / "multitask_manifest.json").read_text())
    tasks = sorted({info["task"] for info in manifest["files"].values()})

    results = {
        "method_note": (
            "MUST-5: window-count-matched DMD reanalysis (subsample matching windows from "
            f"long/timeout-length fail episodes, N={N_REPLICATES} replicates, report p-value "
            "distribution not a point estimate) + shift-operator surrogate null (pure random-walk "
            f"surrogates through the identical delay-embedding/PCA/windowed-DMD pipeline, "
            f"N={N_NULL_EPISODES} synthetic episodes) quantifying how much apparent |lambda|>1 "
            "crossing is explained by the delay-embedding's own shift-operator structure alone, "
            "with no genuine dynamical content."
        ),
        "n_replicates": N_REPLICATES, "n_null_episodes": N_NULL_EPISODES,
        "report1_dspace_eef_grip_proxy": {}, "arm_ablation_spaces": {},
    }
    for task in tasks:
        r1 = analyze_task_report1_dspace(args.embedding_dir, task)
        if r1 is not None:
            results["report1_dspace_eef_grip_proxy"][task] = r1
        results["arm_ablation_spaces"][task] = analyze_task_arms(Path(args.collect_dir), manifest, task)

    for task, r in results["report1_dspace_eef_grip_proxy"].items():
        if not r.get("skipped"):
            m = r["window_count_matched_test"]
            log_message(f"[dmd_window_matching report1-Dspace {task}] unmatched_p={m['unmatched_mannwhitney_p']:.4g} "
                        f"matched_p_median={m['matched_p_median']:.4g} matched_frac_sig={m['matched_frac_p_lt_0.05']:.2f} "
                        f"null_frac_windows>1={r['shift_operator_surrogate_null']['null_frac_windows_gt_1']:.3f}")

    with open(out_dir / "dmd_window_matching_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'dmd_window_matching_test.json'}")


if __name__ == "__main__":
    main()
