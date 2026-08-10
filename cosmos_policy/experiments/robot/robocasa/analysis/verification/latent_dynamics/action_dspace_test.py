"""
action_dspace_test.py — ldv_design_v2.md フェーズ7 対応。

latent_dynamics_verification_report.md §2.6/§9(・ldv_design_v2.md §フェーズ7)が指摘した
代理指標の限界 -- フェーズ1の結合ベクトル c_t が「行動の結果」(eef_pos/gripper_qposのcall間差分)
であって「行動そのもの」(モデルが実際に生成した行動チャンク X_hat_0)ではなかった -- を解消する。
collect_action_chunks.py が新規収集した action_chunk (32 timesteps x 7 dims, call単位) を使い、

  c_t = [ Xp_t (10dim, §5.6/§5.7と同一の進行多様体),
          V_t = Xp_t - Xp_{t-1} (10dim, backward difference, dynamics_embedding_test.pyと同一),
          A_PCA_t (A_DIM=5dim, その call の生成行動チャンクをタスク内でPCA圧縮したもの) ]

という新しい結合ベクトル(ldv_design_v2.md §フェーズ7実装指示3の定義に従う)でD空間を再構築し、
(a) §2.3と同じ進行度逸脱検定でこのD空間が退化していないかを確認したうえで、
(b) §3.3のDMD再評価(windowed DMD, WINDOW=8)をこの新しいD空間に再適用し、旧D空間(eef/gripper
    代理指標版, PnPCounterToCab: p=0.0228, TurnOnStove: p=8.55e-08)と比較する。

A_PCA_t は「行動が環境に及ぼした効果」ではなく「モデルが生成した運動指令そのもの」であり、Delta_eef/
Delta_grip とは異なりcall間差分ではない(その call 単体の生成内容)。dynamics_embedding_test.pyの
build_combined_tensor()が計算するDelta_eef/Delta_gripは本結合ベクトルには含めない
(design.mdの指示通り、代理指標を置き換えるのが目的であり、両方を混ぜると何を検定しているか
不明瞭になるため)。

窓padding・backward-only設計・タスク単位プーリング等、フェーズ1からの意図的逸脱は全て
dynamics_embedding_test.pyのものをそのまま引き継ぐ(このスクリプトはオフライン専用の事後解析であり、
フェーズ3のようなオンライン再利用はしないため、V_tを再度forward differenceにする選択肢もあったが、
フェーズ1との直接比較を優先しbackward差分のまま統一した)。
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import LAYER, K_STEP, scene_residualize
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.manifold_trajectory_test import episode_progress
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.success_failure_trajectory_test import (
    load_task_seed_data_with_success, MIN_FAIL_EPISODES,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.dmd_jacobian_stability_test import (
    windowed_dmd, WINDOW,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamics_embedding_test import (
    PROGRESS_PCA_DIM, WINDOW_TAU, D_DIM, build_delay_embedding_input, deviation_mannwhitney,
    intrinsic_dim_quick, plot_embedding,
)

A_DIM = 5   # PCA-compressed dimensionality of the flattened (chunk_size x 7) generated action chunk


def load_pooled_task_data_actions(collect_dir: Path, manifest, task: str):
    feats, episode, call_idx, success, action_chunk, seed_series_arr = [], [], [], [], [], []
    ep_offset = 0
    for fname, info in manifest["files"].items():
        if info["task"] != task:
            continue
        d = load_task_seed_data_with_success(collect_dir, fname, LAYER, K_STEP)
        fd = np.load(collect_dir / fname)
        key_idx = fd[f"feat_k{K_STEP}_layer{LAYER}_idx"]
        ac = fd["action_chunk"][key_idx]   # (n_calls, chunk_size, 7)
        feats.append(d["feats"])
        episode.append(d["episode"] + ep_offset)
        call_idx.append(d["call_idx"])
        success.append(d["success"])
        action_chunk.append(ac.reshape(ac.shape[0], -1))
        seed_series_arr.append(np.full(len(d["episode"]), info["seed_base"]))
        ep_offset += int(d["episode"].max()) + 1
    return {
        "feats": np.concatenate(feats), "episode": np.concatenate(episode),
        "call_idx": np.concatenate(call_idx), "success": np.concatenate(success),
        "action_chunk_flat": np.concatenate(action_chunk),
        "seed_series": np.concatenate(seed_series_arr),
    }


def build_combined_tensor_action(Xp, episode, call_idx, A_pca):
    """c_t = [Xp_t (10), V_t=Xp_t-Xp_{t-1} (10, backward diff), A_PCA_t (A_DIM)].
    Unlike dynamics_embedding_test.build_combined_tensor, A_PCA_t is NOT a call-to-call
    difference -- it IS the (PCA-compressed) generated action content of call t itself, which is
    what ldv_design_v2.md's phase-7 instruction asks to substitute for the Delta_eef/Delta_grip
    environmental-effect proxy."""
    V = np.zeros_like(Xp)
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        if len(order) < 2:
            continue
        V[order[1:]] = Xp[order[1:]] - Xp[order[:-1]]
    return np.concatenate([Xp, V, A_pca], axis=1)


def dmd_reevaluation(D, episode, call_idx, success, progress, seed=0):
    """Mirrors energy_field_test.py's §3.3 DMD-in-D-space block exactly (windowed DMD, same
    WINDOW/rank settings), but without the KDE-energy cross-comparison (out of scope for phase 7,
    which asks specifically to re-run the DMD stability re-evaluation on the new D-space)."""
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
        fail_crossed += int((lmax > 1.0).any())

    if not (len(succ_max) and len(fail_max)):
        return None
    u_stat, p = mannwhitneyu(succ_max, fail_max, alternative="less")
    table = [[succ_crossed, len(succ_max) - succ_crossed], [fail_crossed, len(fail_max) - fail_crossed]]
    fisher_odds, fisher_p = fisher_exact(table, alternative="less")
    return {
        "n_success_episodes_used": len(succ_max), "n_fail_episodes_used": len(fail_max),
        "episode_max_lambda_mannwhitney_success_lt_fail_p": float(p),
        "frac_episodes_crossing_1": {"success": f"{succ_crossed}/{len(succ_max)}",
                                      "fail": f"{fail_crossed}/{len(fail_max)}"},
        "fisher_crossing_success_lt_fail_p": float(fisher_p),
    }


def analyze_task(collect_dir, manifest, task, out_dir, seed=0):
    pd_ = load_pooled_task_data_actions(collect_dir, manifest, task)
    X_raw, episode, call_idx = pd_["feats"], pd_["episode"], pd_["call_idx"]
    success_call, action_chunk_flat, seed_series = pd_["success"], pd_["action_chunk_flat"], pd_["seed_series"]

    Xr = scene_residualize(X_raw, episode)
    ambient_scaler = StandardScaler().fit(Xr)
    Xs = ambient_scaler.transform(Xr)
    prog_pca = PCA(n_components=min(PROGRESS_PCA_DIM, Xs.shape[0] - 1, Xs.shape[1]), random_state=seed)
    Xp = prog_pca.fit_transform(Xs)

    action_scaler = StandardScaler().fit(action_chunk_flat)
    action_pca = PCA(n_components=min(A_DIM, action_chunk_flat.shape[0] - 1, action_chunk_flat.shape[1]),
                      random_state=seed)
    A_pca = action_pca.fit_transform(action_scaler.transform(action_chunk_flat))

    c_raw = build_combined_tensor_action(Xp, episode, call_idx, A_pca)
    c_scaler = StandardScaler().fit(c_raw)   # same standardize-before-delay-PCA fix as phase 1 (§7バグ#1)
    c = c_scaler.transform(c_raw)
    S = build_delay_embedding_input(c, episode, call_idx, tau=WINDOW_TAU)
    encoder_pca = PCA(n_components=min(D_DIM, S.shape[0] - 1, S.shape[1]), random_state=seed)
    D = encoder_pca.fit_transform(S)

    n_episodes = len(np.unique(episode))
    n_success_episodes = int(sum(bool(success_call[episode == e][0]) for e in np.unique(episode)))

    dev_D_all = deviation_mannwhitney(D, episode, call_idx, success_call, seed=seed)
    idim = intrinsic_dim_quick(D, seed=seed)

    progress = episode_progress(episode, call_idx)
    dmd_result = dmd_reevaluation(D, episode, call_idx, success_call, progress, seed=seed)

    png = plot_embedding(D[:, :2], episode, call_idx, success_call, task + "_action", out_dir)

    def _fmt(dev):
        if dev is None:
            return "p=nan"
        return (f"p={dev['mannwhitney_p_success_lt_fail']:.2e} "
                f"margin_norm={dev['separation_margin_normalized_by_success_spread']:.2f}")

    log_message(f"[action_dspace {task}] n_episodes={n_episodes} n_success={n_success_episodes} "
                f"action_pca_explained_var={action_pca.explained_variance_ratio_.sum():.3f} "
                f"encoder_explained_var={encoder_pca.explained_variance_ratio_.sum():.3f} "
                f"D_dev=({_fmt(dev_D_all)}) "
                f"DMD_p={dmd_result['episode_max_lambda_mannwhitney_success_lt_fail_p'] if dmd_result else 'NA'}")

    with open(out_dir / f"action_dspace_artifact_{task}.pkl", "wb") as f:
        pickle.dump({
            "encoder_pca": encoder_pca, "prog_pca": prog_pca, "ambient_scaler": ambient_scaler,
            "c_scaler": c_scaler, "action_scaler": action_scaler, "action_pca": action_pca,
            "D": D, "Xp": Xp, "episode": episode, "call_idx": call_idx, "success": success_call,
            "seed_series": seed_series, "task": task,
        }, f)

    return {
        "n_episodes": n_episodes, "n_success_episodes": n_success_episodes,
        "n_fail_episodes": n_episodes - n_success_episodes,
        "action_pca_explained_variance_ratio_sum": float(action_pca.explained_variance_ratio_.sum()),
        "encoder_explained_variance_ratio_sum": float(encoder_pca.explained_variance_ratio_.sum()),
        "deviation_test_D_space_pooled": dev_D_all,
        "D_space_intrinsic_dim": idim,
        "dmd_in_action_D_space": dmd_result,
        "plot_png": png,
    }


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
            f"c_t=[Xp_t(10), V_t(10,backward diff), A_PCA_t({A_DIM}, PCA of the call's own "
            f"generated 32x7 action chunk, task-pooled)]={20+A_DIM}dim, "
            f"delay-embedded (tau={WINDOW_TAU}, backward-only) then PCA(D_dim<={D_DIM}) per task. "
            f"Replaces phase-1's Delta_eef/Delta_grip environmental-effect proxy with the model's "
            f"own generated action content (ldv_design_v2.md フェーズ7)."
        ),
        "action_pca_dim": A_DIM, "window_tau": WINDOW_TAU, "d_dim_max": D_DIM,
        "progress_pca_dim": PROGRESS_PCA_DIM, "tasks": {},
    }
    for task in tasks:
        results["tasks"][task] = analyze_task(collect_dir, manifest, task, out_dir, seed=0)

    with open(out_dir / "action_dspace_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'action_dspace_test.json'}")


if __name__ == "__main__":
    main()
