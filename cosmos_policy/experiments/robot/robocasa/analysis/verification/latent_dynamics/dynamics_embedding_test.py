"""
dynamics_embedding_test.py — latent_dynamics_verification_design.md フェーズ1 対応。

「感覚運動の結合動態(Dynamics)の空間埋め込み」を構築する。設計書は状態 Z_t 単体ではなく、
系列 S_t = [Z_{t-tau},...,Z_t,...,Z_{t+tau}] を単一の動態ベクトル D_t に写像する遅延座標埋め込み
(Takens' delay embedding)を要求している。本スクリプトはこれを、既存の attractor_verification
基盤 (collect_v2/, LAYER=13, K_STEP=4, scene_residualize, PCA(10)進行多様体, §5.6/§5.7で確立
済みの手法)の上に構築する。

**設計からの3つの意図的な逸脱(いずれも本docstring末尾の「開示事項」で明記)**:

1. **因果的(backward-only)窓を採用する**: 設計書の式は S_t に未来 Z_{t+tau} を含む対称窓だが、
   フェーズ3(推論時オンラインsteering)はこの埋め込みをロールアウト実行中にリアルタイムで参照する
   必要があり、未来のZは原理的に参照できない。フェーズ1・3で共通の埋め込み関数を使うことを優先し、
   S_t = [Z_{t-2tau},...,Z_{t-tau},...,Z_t] のbackward-only窓を採用する。

2. **行動出力の変化 Delta A_t の代替**: 生成された行動チャンク X_hat_0 (32x7) は
   attractor_verification_report.md §1.7の通り本プロジェクトでは未捕捉(スコープ外)である。
   代わりに collect_v2/ に保存済みの物理量 eef_pos・gripper_qpos の call間差分を
   「行動が環境に及ぼした効果」の代理指標として使う。視覚特徴の勾配(attention mapの時間微分)も
   未捕捉のため本埋め込みでは使用しない。

3. **タスク単位でのプーリング**: 進行多様体PCA(10)・遅延埋め込みエンコーダPCAは、同一タスクの
   2 seed系列(195/196)をエピソードIDをオフセットして結合した上で1回だけ学習する(compute_
   steering_vectors()がタスク単位で主データをプールする既存の慣行に倣う)。タスクをまたいだ
   プーリングは行わない — 各タスクのPCA(10)進行多様体は互いに独立な基底であり、そのまま連結する
   ことは無意味な座標系混合になるため(実装中に気づいた設計ミス、§バグ一覧参照)。

手法:
  1. 対象task内(2 seed系列プール)で、scene_residualize後の特徴(Blk-13, k=4)をStandardScaler+
     PCA(10)に還元し、進行多様体 Xp (§5.6/§5.7と同一パイプライン)を得る。
  2. 潜在フロー V_t = Xp[t+1] - Xp[t] (10次元)、エンドエフェクタ速度 Delta eef_t (3次元)、
     グリッパー開閉速度 Delta grip_t (1次元)を計算し、結合ベクトル c_t = [Xp_t, V_t, Delta eef_t,
     Delta grip_t] (24次元)を構成する。
  3. 各エピソード内で、backward窓 tau=WINDOW_TAU (デフォルト3) の c_{t-2},c_{t-1},c_t を連結した
     系列ベクトル S_t (72次元)を構成する(エピソード先頭でtau未満の場合は最初の値でpadding)。
  4. タスク内でプールした S_t 全体に対して線形PCA(D_DIM次元、デフォルト8)を学習し、
     エンコーダ E: S_t -> D_t とする(design.md §2実装指示3「線形PCA、または自己符号化器」の
     線形PCA版を採用。非線形自己符号化器は今回のスコープでは実施しない、今後の課題参照)。
  5. 妥当性検証:
     (a) D空間で§5.6と同一の「進行度ビン参照分布からの逸脱量」検定を再実行し、生のXp空間
         (§5.6の元の結果)と比較する。結合動態表現が成功/失敗の分離を強める/弱めるかを見る。
     (b) D空間の内在次元数を(§5.4の4手法バッテリーの簡易版として)PCA累積寄与率と相関次元で
         簡易チェックする。
     (c) 2次元PCA可視化(進行度で彩色、成功/失敗を線種で区別)。

開示事項(スコープの完全開示、attractor_verification_report.md §1.7に倣う):
  - 視覚特徴の勾配(attention mapの時間微分)は未捕捉のため結合テンソルから省略。
  - 生成行動チャンク X_hat_0 は未捕捉のため、eef_pos/gripper_qposの物理量差分で代替。
  - 埋め込みはbackward-onlyであり、design.md記載の対称窓(未来を含む)ではない。
  - エンコーダは線形PCAのみ(非線形自己符号化器は未実施)。
  - プーリング単位はタスク(2 seed系列結合)であり、design.md自体はプーリング粒度を規定していない
    ための実装判断。
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import LAYER, K_STEP, scene_residualize
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.manifold_trajectory_test import episode_progress
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.success_failure_trajectory_test import (
    load_task_seed_data_with_success, fit_bin_reference, deviation_from_reference,
    held_out_success_deviation, episode_mean, MIN_FAIL_EPISODES,
)

PROGRESS_PCA_DIM = 10
WINDOW_TAU = 3          # number of consecutive calls stacked into S_t (causal, backward-only)
D_DIM = 8                # dimensionality of the dynamics-embedding space D


def load_physical(collect_dir: Path, fname: str):
    fd = np.load(collect_dir / fname)
    key = f"feat_k{K_STEP}_layer{LAYER}"
    idx_key = key + "_idx"
    keep_idx = fd[idx_key]
    gripper_qpos = fd["gripper_qpos"][keep_idx]
    gripper_width = np.abs(gripper_qpos).sum(axis=1)
    eef_pos = fd["eef_pos"][keep_idx]
    return gripper_width, eef_pos


def load_pooled_task_data(collect_dir: Path, manifest, task: str):
    """Pool both seed series of one task, offsetting episode ids so they stay unique
    (mirrors compute_steering_vectors()'s ep_offset pattern in steering_intervention.py)."""
    feats, episode, call_idx, success, gripper_width, eef_pos, seed_series_arr = [], [], [], [], [], [], []
    ep_offset = 0
    for fname, info in manifest["files"].items():
        if info["task"] != task:
            continue
        d = load_task_seed_data_with_success(collect_dir, fname, LAYER, K_STEP)
        gw, ep = load_physical(collect_dir, fname)
        feats.append(d["feats"])
        episode.append(d["episode"] + ep_offset)
        call_idx.append(d["call_idx"])
        success.append(d["success"])
        gripper_width.append(gw)
        eef_pos.append(ep)
        seed_series_arr.append(np.full(len(d["episode"]), info["seed_base"]))
        ep_offset += int(d["episode"].max()) + 1
    return {
        "feats": np.concatenate(feats), "episode": np.concatenate(episode),
        "call_idx": np.concatenate(call_idx), "success": np.concatenate(success),
        "gripper_width": np.concatenate(gripper_width), "eef_pos": np.concatenate(eef_pos),
        "seed_series": np.concatenate(seed_series_arr),
    }


def build_combined_tensor(Xp, episode, call_idx, gripper_width, eef_pos):
    """c_t = [Xp_t (10), V_t=Xp_t-Xp_{t-1} (10), Delta_eef_t=eef_t-eef_{t-1} (3),
    Delta_grip_t=grip_t-grip_{t-1} (1)] per call, ordered within each episode by call_idx.

    NOTE (bug found & fixed during this verification, see report bug list): an earlier version
    used the FORWARD difference V_t=Xp_{t+1}-Xp_t. That is fine for a purely offline/retrospective
    analysis of already-collected rollouts, but this same c_t construction is reused verbatim by
    the phase-3 online steering hook to locate the live rollout's current position in D-space --
    and at call t during a real rollout, Xp_{t+1} (this call's OWN outcome) does not exist yet.
    Using a forward difference here would silently bake non-causal (look-ahead) information into
    a function that is later called online, which is a much more serious bug than it looks: the
    field would seem to "know" where the trajectory is about to go. Backward differences
    (this call vs. the previous one) are the causal analogue: available in real time, and for a
    smooth trajectory they carry essentially the same local-dynamics information as the forward
    difference did. The first call of each episode (t=0) has no t-1, so it is padded with zero
    flow (no motion has "happened yet"), NOT with the last call as before.
    """
    V = np.zeros_like(Xp)
    d_eef = np.zeros((len(episode), 3))
    d_grip = np.zeros((len(episode), 1))
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        if len(order) < 2:
            continue
        V[order[1:]] = Xp[order[1:]] - Xp[order[:-1]]
        d_eef[order[1:]] = eef_pos[order[1:]] - eef_pos[order[:-1]]
        d_grip[order[1:], 0] = gripper_width[order[1:]] - gripper_width[order[:-1]]
    return np.concatenate([Xp, V, d_eef, d_grip], axis=1)


def build_delay_embedding_input(c, episode, call_idx, tau=WINDOW_TAU):
    """S_t = concat(c_{t-tau+1}, ..., c_t) per call, causal (backward-only), within-episode.
    Episode-start calls with fewer than tau history steps are padded by repeating c_t."""
    n, dc = c.shape
    S = np.zeros((n, dc * tau))
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        for pos, gi in enumerate(order):
            hist = []
            for back in range(tau - 1, -1, -1):
                src_pos = max(pos - back, 0)
                hist.append(c[order[src_pos]])
            S[gi] = np.concatenate(hist)
    return S


def intrinsic_dim_quick(X, seed=0):
    """Lightweight two-statistic check (cumulative-variance PCA-PR + correlation dimension),
    a reduced version of §5.4's 4-method battery -- scoped down deliberately (see module
    docstring's disclosure) since the primary goal here is embedding sanity, not a rigorous
    dimensionality claim."""
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(random_state=seed).fit(Xs)
    var_ratio = pca.explained_variance_ratio_
    n90 = int(np.searchsorted(np.cumsum(var_ratio), 0.90) + 1)
    pr = float((np.sum(pca.explained_variance_) ** 2) / np.sum(pca.explained_variance_ ** 2))

    rng = np.random.RandomState(seed)
    n = min(len(Xs), 400)
    sub = Xs[rng.choice(len(Xs), n, replace=False)] if len(Xs) > n else Xs
    dists = pdist(sub)
    dists = dists[dists > 1e-12]
    r = np.logspace(np.log10(np.percentile(dists, 5)), np.log10(np.percentile(dists, 50)), 12)
    C = np.array([(dists < ri).mean() for ri in r])
    valid = C > 0
    logr, logC = np.log(r[valid]), np.log(C[valid])
    if len(logr) >= 3:
        slope, intercept = np.polyfit(logr, logC, 1)
        pred = slope * logr + intercept
        ss_res = np.sum((logC - pred) ** 2)
        ss_tot = np.sum((logC - logC.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    else:
        slope, r2 = float("nan"), float("nan")
    return {"n90_variance": n90, "pca_participation_ratio": pr, "correlation_dimension": float(slope),
            "correlation_dimension_fit_r2": r2}


def deviation_mannwhitney(X, episode, call_idx, success_call, seed=0):
    """Re-run the §5.6 progress-bin-reference deviation test on an arbitrary feature space X
    (either Xp or D), returning the Mann-Whitney p(success<fail) and effect summary. Mirrors
    success_failure_trajectory_test.analyze_task_seed but takes a pre-built feature matrix."""
    episodes = np.unique(episode)
    ep_success = {e: bool(success_call[episode == e][0]) for e in episodes}
    success_episodes = np.array([e for e in episodes if ep_success[e]])
    fail_episodes = np.array([e for e in episodes if not ep_success[e]])
    if len(fail_episodes) < MIN_FAIL_EPISODES:
        return None
    progress = episode_progress(episode, call_idx)
    dev_success_heldout = held_out_success_deviation(X, episode, progress, success_episodes, seed=seed)
    mask_succ_all = np.isin(episode, success_episodes)
    centers_full, variances_full = fit_bin_reference(X[mask_succ_all], progress[mask_succ_all])
    mask_fail_all = np.isin(episode, fail_episodes)
    dev_fail = np.full(len(episode), np.nan)
    dev_fail[mask_fail_all] = deviation_from_reference(X[mask_fail_all], progress[mask_fail_all],
                                                         centers_full, variances_full)
    succ_vals = np.array(list(episode_mean(dev_success_heldout, episode, success_episodes).values()))
    fail_vals = np.array(list(episode_mean(dev_fail, episode, fail_episodes).values()))
    succ_ep_means = episode_mean(dev_success_heldout, episode, success_episodes)
    fail_ep_means = episode_mean(dev_fail, episode, fail_episodes)
    u_stat, u_p = mannwhitneyu(succ_vals, fail_vals, alternative="less")
    # NOTE (found during this verification, see report bug list): when success/fail deviation
    # distributions are *completely* non-overlapping (as they already were for the raw Xp space
    # per §5.6, and remain so for D), Mann-Whitney U saturates at its maximum (n_s*n_f) and p
    # becomes a function of sample size alone -- identical p across two genuinely different
    # feature spaces is then expected, not evidence of a computation bug. Report an unsaturating
    # effect-size (separation margin, normalized by the success group's own spread) alongside p
    # so embeddings can still be compared once the test statistic has hit its ceiling.
    separation_margin = float(fail_vals.min() - succ_vals.max())
    succ_spread = float(succ_vals.std() + 1e-8)
    return {
        "n_success_episodes": int(len(success_episodes)), "n_fail_episodes": int(len(fail_episodes)),
        "completely_separated": bool(fail_vals.min() > succ_vals.max()),
        "separation_margin_raw": separation_margin,
        "separation_margin_normalized_by_success_spread": separation_margin / succ_spread,
        "mean_ratio_fail_over_success": float(fail_vals.mean() / (succ_vals.mean() + 1e-8)),
        "mean_dev_success_heldout": float(np.mean(list(succ_ep_means.values()))),
        "mean_dev_fail": float(np.mean(list(fail_ep_means.values()))),
        "mannwhitney_p_success_lt_fail": float(u_p),
    }


def plot_embedding(D2, episode, call_idx, success_call, task, out_dir, max_each=8):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = plt.get_cmap("viridis")
    succ_eps = np.unique(episode[success_call.astype(bool)])[:max_each]
    fail_eps = np.unique(episode[~success_call.astype(bool)])[:max_each]
    sc = None
    for e in succ_eps:
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        traj = D2[order]
        prog = np.arange(len(order)) / max(len(order) - 1, 1)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="green", alpha=0.5, linewidth=1.2, zorder=1)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=prog, cmap=cmap, vmin=0, vmax=1, s=12, zorder=2, marker="o")
    for e in fail_eps:
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        traj = D2[order]
        prog = np.arange(len(order)) / max(len(order) - 1, 1)
        ax.plot(traj[:, 0], traj[:, 1], "--", color="red", alpha=0.5, linewidth=1.2, zorder=1)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=prog, cmap=cmap, vmin=0, vmax=1, s=12, zorder=2, marker="x")
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="within-episode progress")
    ax.set_xlabel("D-PC1")
    ax.set_ylabel("D-PC2")
    ax.set_title(f"{task}: dynamics-embedding D space (both seed series pooled)\n"
                 f"(success solid/o green, failure dashed/x red)")
    fig.tight_layout()
    fig_path = out_dir / f"dynamics_embedding_{task}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def analyze_task(collect_dir, manifest, task, out_dir, seed=0):
    pd_ = load_pooled_task_data(collect_dir, manifest, task)
    X_raw, episode, call_idx = pd_["feats"], pd_["episode"], pd_["call_idx"]
    success_call, gripper_width, eef_pos = pd_["success"], pd_["gripper_width"], pd_["eef_pos"]
    seed_series = pd_["seed_series"]

    Xr = scene_residualize(X_raw, episode)
    ambient_scaler = StandardScaler().fit(Xr)
    Xs = ambient_scaler.transform(Xr)
    prog_pca = PCA(n_components=min(PROGRESS_PCA_DIM, Xs.shape[0] - 1, Xs.shape[1]), random_state=seed)
    Xp = prog_pca.fit_transform(Xs)

    c_raw = build_combined_tensor(Xp, episode, call_idx, gripper_width, eef_pos)
    # NOTE (bug found & fixed during this verification, see report bug list): c_raw's 4 blocks
    # live on wildly different physical scales (Xp_t/V_t are PCA-score units with std~O(5-20),
    # Delta_eef_t is in meters with std~O(0.01), Delta_grip_t similarly small). Feeding c_raw
    # unstandardized into the delay-embedding PCA lets the PCA variance be dominated entirely by
    # Xp_t/V_t, so D collapses to (numerically) just a rotation of the progression manifold Xp and
    # silently discards the eef/gripper action-effect information -- defeating the whole point of
    # a *joint* sensorimotor embedding. Standardize each column of c before stacking/PCA so all
    # 4 blocks contribute comparably.
    c_scaler = StandardScaler().fit(c_raw)
    c = c_scaler.transform(c_raw)
    S = build_delay_embedding_input(c, episode, call_idx, tau=WINDOW_TAU)
    encoder_pca = PCA(n_components=min(D_DIM, S.shape[0] - 1, S.shape[1]), random_state=seed)
    D = encoder_pca.fit_transform(S)

    n_episodes = len(np.unique(episode))
    n_success_episodes = int(sum(bool(success_call[episode == e][0]) for e in np.unique(episode)))

    dev_D_all = deviation_mannwhitney(D, episode, call_idx, success_call, seed=seed)
    dev_Xp_all = deviation_mannwhitney(Xp, episode, call_idx, success_call, seed=seed)
    idim = intrinsic_dim_quick(D, seed=seed)

    by_seed_series = {}
    for ss in np.unique(seed_series):
        mask = seed_series == ss
        dev_D_ss = deviation_mannwhitney(D[mask], episode[mask], call_idx[mask], success_call[mask], seed=seed)
        by_seed_series[str(ss)] = {"deviation_test_D_space": dev_D_ss}

    png = plot_embedding(D[:, :2], episode, call_idx, success_call, task, out_dir)

    def _fmt(dev):
        if dev is None:
            return "p=nan"
        return (f"p={dev['mannwhitney_p_success_lt_fail']:.2e} "
                f"margin_norm={dev['separation_margin_normalized_by_success_spread']:.2f} "
                f"ratio={dev['mean_ratio_fail_over_success']:.2f}")

    log_message(f"[dynamics_embedding {task}] n_episodes={n_episodes} n_success={n_success_episodes} "
                f"encoder_explained_var={encoder_pca.explained_variance_ratio_.sum():.3f} "
                f"D_dev=({_fmt(dev_D_all)}) Xp_dev=({_fmt(dev_Xp_all)}) "
                f"D_n90var={idim['n90_variance']} D_corrdim={idim['correlation_dimension']:.2f}")

    artifact = {
        "encoder_pca": encoder_pca, "prog_pca": prog_pca, "ambient_scaler": ambient_scaler,
        "c_scaler": c_scaler,
        "window_tau": WINDOW_TAU, "progress_pca_dim": PROGRESS_PCA_DIM, "d_dim": encoder_pca.n_components_,
        "task": task,
        # for phase-3 online use: store the fitted D-space per call, plus enough metadata
        # to build a nearest-neighbour flow-field lookup (see energy_field_test.py / phase 3 script).
        # ambient_scaler+prog_pca reproduce Xp from a live (scene-residualized) feature vector;
        # scene_residualize itself is per-episode centering and has no causal online equivalent
        # (see phase-3 script for the running-mean substitute used there, documented there).
        "D": D, "Xp": Xp, "episode": episode, "call_idx": call_idx, "success": success_call,
        "seed_series": seed_series, "c": c,
    }
    with open(out_dir / f"dynamics_embedding_artifact_{task}.pkl", "wb") as f:
        pickle.dump(artifact, f)

    return {
        "n_episodes": n_episodes, "n_success_episodes": n_success_episodes,
        "n_fail_episodes": n_episodes - n_success_episodes,
        "encoder_explained_variance_ratio_sum": float(encoder_pca.explained_variance_ratio_.sum()),
        "deviation_test_D_space_pooled": dev_D_all,
        "deviation_test_progress_manifold_Xp_space_pooled": dev_Xp_all,
        "deviation_test_D_space_by_seed_series": by_seed_series,
        "D_space_intrinsic_dim": idim,
        "plot_png": png,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())

    tasks = sorted({info["task"] for info in manifest["files"].values()})

    results = {
        "method_note": (
            f"S_t = backward窓(tau={WINDOW_TAU})で結合ベクトル c_t=[Xp_t(10), V_t(10), "
            f"Delta_eef_t(3), Delta_grip_t(1)]=24次元をtau回スタックした{WINDOW_TAU*24}次元ベクトル。"
            f"線形PCA(D_dim<={D_DIM})をタスク単位(2 seed系列プール)で学習しエンコーダとする"
            f"(design.md §2実装指示3の線形PCA版)。"
        ),
        "window_tau": WINDOW_TAU, "d_dim_max": D_DIM, "progress_pca_dim": PROGRESS_PCA_DIM,
        "tasks": {},
    }
    for task in tasks:
        results["tasks"][task] = analyze_task(collect_dir, manifest, task, out_dir, seed=0)

    with open(out_dir / "dynamics_embedding_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'dynamics_embedding_test.json'}")


if __name__ == "__main__":
    main()
