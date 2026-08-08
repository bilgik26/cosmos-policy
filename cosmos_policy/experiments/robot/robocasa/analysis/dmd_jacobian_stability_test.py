"""
dmd_jacobian_stability_test.py — attractor_verification_report.md への追加検証 (§4.1.6)。

ユーザー提案「ロールアウト軌跡のヤコビアン固有値を追跡し、自己安定化/分岐を検証する」への対応。
厳密なヤコビアン（自動微分でのモデル全体の線形化）は計算コストと拡散サンプリングの
非決定的な内部ステップ構造のため実行しない。代わりに **Dynamic Mode Decomposition (DMD)**
（Tu et al. 2014, exact DMD）を用い、観測された特徴量軌跡 z_t (§4.1.5と同じPCA(10)空間、
`skill_count.py`のscene_residualize後、LAYER=13, K_STEP=4) から、局所的な線形化演算子
A（z_{t+1} ≈ A z_t）をスライディングウィンドウで data-driven に推定し、その固有値をヤコビアンの
近似として扱う。

方法:
  1. §4.1.5と同一パイプラインでPCA(10)空間の軌跡 Xp を構築（collect_v2/使用、シーン残差化、
     成功エピソードのみで進行度5分位ビン参照分布をGroupKFold不要で全成功エピソードに全体fit
     — これは§4.1.5の`fit_bin_reference`をそのまま再利用し、失敗エピソードの「逸脱ベクトル」
     （§4.1.5は逸脱量=スカラー距離のみ算出していたが、本検証では方向情報を残した生ベクトル
     Xp[i]-center[bin]を使う）を得るためのみに用いる。循環性は§4.1.5と同様に生じない
     （失敗エピソードは参照分布の学習に一切使われない）。
  2. 各エピソードの軌跡（call_idx順）に窓幅WINDOW=8のスライディングウィンドウ（step=1）を
     適用し、各窓でexact DMDを実行: X=[z_t,...,z_{t+6}], X'=[z_{t+1},...,z_{t+7}]、SVD階数
     打ち切り（累積エネルギー90%、上限RANK_MAX=4 — §4.1.3で確認済みの低次元構造と整合）で
     簡約演算子Atildeを求め、その固有値・固有ベクトル（exact DMD modeとして元のPCA(10)空間に
     引き戻す）を各窓の「局所ヤコビアン近似」とする。
  3. 検証1（成功=自己安定化）: 成功エピソードの全窓から得た固有値を複素平面にプロットし、
     単位円内に収まる割合を報告する。
  4. 検証2（失敗の予兆）: 各エピソードの最大固有値絶対値|λ_max|を進行度に対してプロットし、
     成功/失敗エピソード群でエピソード内最大値・平均値・「初めて|λ|>1を超える進行度」を
     比較する（Mann-Whitney U、Fisher exact）。
  5. 検証3（不安定固有ベクトルと逸脱方向の一致）: |λ|>1となった窓の最大固有値に対応する
     固有ベクトル（実部を正規化）と、同時刻の§4.1.5式「逸脱ベクトル」の絶対コサイン類似度を
     計算する。**2種類の対照群**と比較する: (a) permutation null — 逸脱ベクトルの対応付けを
     シャッフルした帰無分布との比較、(b) 同一窓内の最小|λ|固有ベクトル（安定方向）を同じ
     逸脱ベクトルと比較するpaired negative control（Wilcoxon符号順位検定）。

既知の限界（事前に開示）:
  - **重大な交絡: エピsoード長そのものが成功/失敗とほぼ完全に相関している**。RoboCasa環境は
    タスク成功時点で即座にエピソードを終了するため、成功エピソードは常に短く（本データでは
    5〜19ステップ）、失敗エピソードは常にタイムアウト上限（32ステップ）まで走る（後述の実測値
    参照）。したがって固定ステップ数の窓（WINDOW=8）は、短い成功エピソードでは「エピソード
    のほぼ全体」を覆う一方、長い失敗エピソードでは「一部分」しか覆わない——両条件で窓が捉える
    力学の「局所性」の度合いが異なる。これはDMD法自体の欠陥ではなく環境の終了規則に由来する
    構造的交絡であり、本データのみからは成功/失敗の比較からこの交絡を完全に分離できない。
    WINDOW=8は「ほぼ全ての成功エピソードが最低1窓を持つ」という条件を満たす最小限の値として
    選んだが（成功エピソード最短長=5のCoffeePressButtonは元々失敗数不足でスキップ対象）、
    この限界自体は解消されない。検証2（不安定化の先行性）の結果はこの交絡を踏まえ、確定的な
    主張ではなく示唆として扱う。
  - 個々のepisodeで「物理的に失敗が顕在化する正確な瞬間」（例: 物を落とした時刻）を直接
    記録したログはcollect_v2/に存在しない（feat_k*_layer*, episode, call_idx,
    episode終端のsuccessフラグのみ）。よって「失敗の予兆」は「エピソード終端(progress=1、
    成功フラグ確定)よりも有意に早い進行度で不安定化が検出される」という、利用可能なデータで
    検証可能な最も厳密な形に操作的に定義する。「ロボットが物を落とす瞬間より前」という文字通りの
    主張はできない。
  - DMDは窓内で局所線形とみなす近似であり、窓幅WINDOW=10・階数打ち切りという設計選択に依存する。
    頑健性は合成データでの事前検証（`synthetic_dmd_validation()`、既知の固有値・固有ベクトルを
    注入し正しく回収できることを確認）でのみ担保しており、別ハイパーパラメータでの網羅的な
    感度分析は未実施。
  - 固有ベクトルは複素数になりうるため実部のみを使う簡略化をしている（回転を伴う不安定化の
    場合、実部だけでは方向の一部しか捉えられない可能性がある）。
  - §4.1.5と同じくCloseDrawer・CoffeePressButtonは失敗エピソードが不足するためスキップする。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import (
    LAYER, K_STEP, scene_residualize,
)
from cosmos_policy.experiments.robot.robocasa.analysis.manifold_trajectory_test import episode_progress
from cosmos_policy.experiments.robot.robocasa.analysis.success_failure_trajectory_test import (
    load_task_seed_data_with_success, fit_bin_reference, progress_bin,
    MIN_FAIL_EPISODES, N_PROGRESS_BINS, PCA_DIM,
)

WINDOW = 8
WINDOW_STEP = 1
RANK_MAX = 4
RANK_ENERGY_THRESHOLD = 0.90
N_PERM = 2000


# ─────────────────────────── core DMD machinery ───────────────────────────

def dmd_eigendecompose(window, rank_energy=RANK_ENERGY_THRESHOLD, rank_max=RANK_MAX):
    """Exact DMD (Tu et al. 2014) on one window of snapshots. window: (W, D)."""
    X = window[:-1].T   # (D, W-1)
    Xp = window[1:].T   # (D, W-1)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    r = int(np.searchsorted(energy, rank_energy) + 1)
    r = max(1, min(r, rank_max, len(S)))
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r, :].T
    Sr_safe = np.maximum(Sr, 1e-8 * max(S[0], 1e-12))
    Atilde = Ur.T @ Xp @ Vr @ np.diag(1.0 / Sr_safe)
    eigvals, W_eig = np.linalg.eig(Atilde)
    Phi = Xp @ Vr @ np.diag(1.0 / Sr_safe) @ W_eig  # exact DMD modes, (D, r), complex
    return eigvals, Phi, r


def windowed_dmd(Z, progress, window=WINDOW, step=WINDOW_STEP):
    """Slide a window over trajectory Z (T,D); return list of per-window dicts."""
    out = []
    T = len(Z)
    for start in range(0, T - window + 1, step):
        end_idx = start + window - 1
        eigvals, Phi, r = dmd_eigendecompose(Z[start:start + window])
        mags = np.abs(eigvals)
        i_max, i_min = int(np.argmax(mags)), int(np.argmin(mags))
        out.append({
            "start": start, "end_idx": end_idx, "progress": float(progress[end_idx]),
            "eigvals": eigvals, "lambda_max": float(mags[i_max]), "lambda_min": float(mags[i_min]),
            "top_eigvec": Phi[:, i_max], "stable_eigvec": Phi[:, i_min], "rank": r,
        })
    return out


def unit_real(v):
    v = np.real(v)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ─────────────────────────── synthetic validation ───────────────────────────

def synthetic_dmd_validation():
    """Inject known stable/unstable linear dynamics; verify DMD recovers them.
    Mirrors this project's established practice (e.g. §4.1.5) of validating a new
    statistical method on synthetic data with a known ground truth before trusting it
    on real, noisier, nonlinear features."""
    rng = np.random.RandomState(42)
    D, T, switch_t = 10, 48, 24
    Q, _ = np.linalg.qr(rng.randn(D, D))
    stable_vals = np.linspace(0.75, 0.88, D)
    A_stable = Q @ np.diag(stable_vals) @ Q.T
    u_true = Q[:, 0]  # ground-truth unstable direction
    unstable_vals = stable_vals.copy()
    unstable_vals[0] = 1.20
    A_unstable = Q @ np.diag(unstable_vals) @ Q.T

    z = np.zeros((T, D))
    z[0] = rng.randn(D) * 0.5
    noise_sigma = 0.02
    for t in range(T - 1):
        A = A_stable if t < switch_t else A_unstable
        z[t + 1] = A @ z[t] + rng.randn(D) * noise_sigma

    progress = np.arange(T) / (T - 1)
    windows = windowed_dmd(z, progress)

    stable_windows = [w for w in windows if w["end_idx"] < switch_t]
    unstable_windows = [w for w in windows if w["start"] >= switch_t]

    stable_lmax = np.array([w["lambda_max"] for w in stable_windows])
    unstable_lmax = np.array([w["lambda_max"] for w in unstable_windows])

    assert len(stable_windows) >= 3 and len(unstable_windows) >= 3, (
        f"synthetic check: too few fully-stable/unstable windows "
        f"({len(stable_windows)}/{len(unstable_windows)}) — adjust T/WINDOW/switch_t"
    )
    assert stable_lmax.mean() < 1.0, f"stable-regime windows should have mean lambda_max<1, got {stable_lmax.mean():.3f}"
    assert (stable_lmax < 1.0).mean() >= 0.8, f"stable-regime windows should mostly have lambda_max<1, got frac={((stable_lmax < 1.0).mean()):.2f}"
    assert unstable_lmax.mean() > 1.0, f"unstable-regime windows should have mean lambda_max>1, got {unstable_lmax.mean():.3f}"
    assert (unstable_lmax > 1.0).mean() >= 0.8, f"unstable-regime windows should mostly have lambda_max>1, got frac={((unstable_lmax > 1.0).mean()):.2f}"

    cos_top = [abs(np.dot(unit_real(w["top_eigvec"]), u_true)) for w in unstable_windows if w["rank"] >= 2]
    cos_stable = [abs(np.dot(unit_real(w["stable_eigvec"]), u_true)) for w in unstable_windows if w["rank"] >= 2]
    assert len(cos_top) >= 3, "synthetic check: too few rank>=2 unstable windows for eigenvector check"
    assert np.mean(cos_top) > 0.7, f"top (unstable) eigenvector should align with known direction u, got mean|cos|={np.mean(cos_top):.3f}"
    assert np.mean(cos_top) > np.mean(cos_stable) + 0.15, (
        f"top eigenvector should align with u more than the stable eigenvector does: "
        f"top={np.mean(cos_top):.3f} stable={np.mean(cos_stable):.3f}"
    )
    log_message(
        f"[synthetic DMD validation] PASSED: stable mean|lambda_max|={stable_lmax.mean():.3f} "
        f"(frac<1: {(stable_lmax < 1.0).mean():.2f}), unstable mean|lambda_max|={unstable_lmax.mean():.3f} "
        f"(frac>1: {(unstable_lmax > 1.0).mean():.2f}), eigvec cos|top vs u|={np.mean(cos_top):.3f} "
        f"vs cos|stable vs u|={np.mean(cos_stable):.3f}"
    )
    return {
        "stable_mean_lambda_max": float(stable_lmax.mean()),
        "stable_frac_lt_1": float((stable_lmax < 1.0).mean()),
        "unstable_mean_lambda_max": float(unstable_lmax.mean()),
        "unstable_frac_gt_1": float((unstable_lmax > 1.0).mean()),
        "eigvec_cos_top_vs_true": float(np.mean(cos_top)),
        "eigvec_cos_stable_vs_true": float(np.mean(cos_stable)),
    }


# ─────────────────────────── plotting ───────────────────────────

def plot_eigenvalue_spectrum(eigvals_pool, frac_inside, task, seed_series, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, label="unit circle")
    re, im = np.real(eigvals_pool), np.imag(eigvals_pool)
    ax.scatter(re, im, s=8, alpha=0.35, color="#2ca02c")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_aspect("equal")
    ax.set_title(f"{task} seed={seed_series}: DMD eigenvalue spectrum (success episodes only)\n"
                 f"n={len(eigvals_pool)} eigenvalues from all windows, {frac_inside*100:.1f}% inside |λ|<1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / f"dmd_spectrum_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def plot_lambda_max_vs_progress(success_series, fail_series, task, seed_series, out_dir, max_each=6):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle=":", label="|λ|=1")
    for i, (ep, prog, lmax) in enumerate(success_series[:max_each]):
        ax.plot(prog, lmax, "-", color="#2ca02c", alpha=0.6, linewidth=1.2,
                label="success" if i == 0 else None)
    for i, (ep, prog, lmax) in enumerate(fail_series[:max_each]):
        ax.plot(prog, lmax, "-", color="#d62728", alpha=0.6, linewidth=1.2,
                label="failure" if i == 0 else None)
    ax.set_xlabel("within-episode progress (window end)")
    ax.set_ylabel("|λ_max| (windowed DMD)")
    ax.set_title(f"{task} seed={seed_series}: max DMD eigenvalue magnitude vs. progress")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / f"dmd_lambda_max_vs_progress_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def plot_eigenvector_alignment_hist(cos_top, cos_stable, null_means, obs_mean, task, seed_series, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.hist(cos_top, bins=15, alpha=0.6, color="#d62728", label=f"unstable eigvec (n={len(cos_top)})")
    ax.hist(cos_stable, bins=15, alpha=0.6, color="#1f77b4", label=f"stable eigvec (n={len(cos_stable)})")
    ax.set_xlabel("|cos(eigenvector, §4.1.5 deviation vector)|")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("paired comparison (same call, same deviation vector)")

    ax = axes[1]
    ax.hist(null_means, bins=30, alpha=0.6, color="gray", label="permutation null (shuffled pairing)")
    ax.axvline(obs_mean, color="#d62728", linewidth=2, label=f"observed mean={obs_mean:.3f}")
    ax.set_xlabel("mean |cos| across pairs")
    ax.set_ylabel("permutation count")
    ax.legend(fontsize=8)
    ax.set_title("permutation null for unstable-eigenvector alignment")

    fig.suptitle(f"{task} seed={seed_series}: unstable-eigenvector vs. deviation-direction alignment")
    fig.tight_layout()
    fig_path = out_dir / f"dmd_eigenvector_alignment_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


# ─────────────────────────── per task/seed analysis ───────────────────────────

def analyze_task_seed(collect_dir, task, seed_series, fname, out_dir, seed=0):
    d = load_task_seed_data_with_success(collect_dir, fname, LAYER, K_STEP)
    X_raw, episode, call_idx, success_call = d["feats"], d["episode"], d["call_idx"], d["success"]

    episodes = np.unique(episode)
    ep_success = {e: bool(success_call[episode == e][0]) for e in episodes}
    success_episodes = np.array([e for e in episodes if ep_success[e]])
    fail_episodes = np.array([e for e in episodes if not ep_success[e]])
    n_success, n_fail = len(success_episodes), len(fail_episodes)

    log_message(f"[dmd {task} seed={seed_series}] n_episodes={len(episodes)} success={n_success} fail={n_fail}")

    if n_fail < MIN_FAIL_EPISODES:
        return {
            "n_episodes": int(len(episodes)), "n_success_episodes": int(n_success),
            "n_fail_episodes": int(n_fail), "skipped": True,
            "skip_reason": f"insufficient failure episodes (n={n_fail} < {MIN_FAIL_EPISODES})",
        }

    Xr = scene_residualize(X_raw, episode)
    Xs = StandardScaler().fit_transform(Xr)
    pca = PCA(n_components=min(PCA_DIM, Xs.shape[0] - 1, Xs.shape[1]), random_state=seed)
    Xp = pca.fit_transform(Xs)
    progress = episode_progress(episode, call_idx)

    # success-only reference (no circularity: same construction as §4.1.5's fit on success episodes)
    mask_succ_all = np.isin(episode, success_episodes)
    centers_full, variances_full = fit_bin_reference(Xp[mask_succ_all], progress[mask_succ_all])

    def episode_traj(e):
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        return order, Xp[order], progress[order]

    # ── part 1 + 2: windowed DMD per episode ──
    success_eigvals_pool = []
    success_series, fail_series = [], []
    ep_summary = {}
    for e in success_episodes:
        order, Z, prog = episode_traj(e)
        if len(Z) < WINDOW:
            continue
        windows = windowed_dmd(Z, prog)
        for w in windows:
            success_eigvals_pool.extend(list(w["eigvals"]))
        lmax_series = np.array([w["lambda_max"] for w in windows])
        prog_series = np.array([w["progress"] for w in windows])
        success_series.append((e, prog_series, lmax_series))
        crossing = prog_series[lmax_series > 1.0]
        ep_summary[("success", int(e))] = {
            "max_lambda_max": float(lmax_series.max()), "mean_lambda_max": float(lmax_series.mean()),
            "first_crossing_progress": float(crossing[0]) if len(crossing) else None,
        }

    fail_eigvec_pairs = []  # (top_eigvec, stable_eigvec, deviation_vec) for |lambda|>1 windows, rank>=2
    for e in fail_episodes:
        order, Z, prog = episode_traj(e)
        if len(Z) < WINDOW:
            continue
        windows = windowed_dmd(Z, prog)
        lmax_series = np.array([w["lambda_max"] for w in windows])
        prog_series = np.array([w["progress"] for w in windows])
        fail_series.append((e, prog_series, lmax_series))
        crossing = prog_series[lmax_series > 1.0]
        ep_summary[("fail", int(e))] = {
            "max_lambda_max": float(lmax_series.max()), "mean_lambda_max": float(lmax_series.mean()),
            "first_crossing_progress": float(crossing[0]) if len(crossing) else None,
        }
        for w in windows:
            if w["lambda_max"] > 1.0 and w["rank"] >= 2:
                call_pos_in_traj = w["end_idx"]
                global_i = order[call_pos_in_traj]
                b = progress_bin(np.array([progress[global_i]]))[0]
                dev_vec = Xp[global_i] - centers_full[b]
                if np.linalg.norm(dev_vec) < 1e-8:
                    continue
                fail_eigvec_pairs.append((unit_real(w["top_eigvec"]), unit_real(w["stable_eigvec"]), dev_vec / np.linalg.norm(dev_vec)))

    eigvals_pool_arr = np.array(success_eigvals_pool)
    frac_inside = float((np.abs(eigvals_pool_arr) < 1.0).mean()) if len(eigvals_pool_arr) else float("nan")

    # ── stats: episode-level max/mean lambda_max, success vs fail ──
    succ_max = np.array([ep_summary[("success", int(e))]["max_lambda_max"] for e in success_episodes if ("success", int(e)) in ep_summary])
    fail_max = np.array([ep_summary[("fail", int(e))]["max_lambda_max"] for e in fail_episodes if ("fail", int(e)) in ep_summary])
    succ_mean = np.array([ep_summary[("success", int(e))]["mean_lambda_max"] for e in success_episodes if ("success", int(e)) in ep_summary])
    fail_mean = np.array([ep_summary[("fail", int(e))]["mean_lambda_max"] for e in fail_episodes if ("fail", int(e)) in ep_summary])

    u_max_stat, u_max_p = mannwhitneyu(succ_max, fail_max, alternative="less") if len(succ_max) and len(fail_max) else (float("nan"), float("nan"))
    u_mean_stat, u_mean_p = mannwhitneyu(succ_mean, fail_mean, alternative="less") if len(succ_mean) and len(fail_mean) else (float("nan"), float("nan"))

    succ_crossed = sum(1 for e in success_episodes if ("success", int(e)) in ep_summary and ep_summary[("success", int(e))]["first_crossing_progress"] is not None)
    fail_crossed = sum(1 for e in fail_episodes if ("fail", int(e)) in ep_summary and ep_summary[("fail", int(e))]["first_crossing_progress"] is not None)
    succ_n_used = sum(1 for e in success_episodes if ("success", int(e)) in ep_summary)
    fail_n_used = sum(1 for e in fail_episodes if ("fail", int(e)) in ep_summary)
    table = [[succ_crossed, succ_n_used - succ_crossed], [fail_crossed, fail_n_used - fail_crossed]]
    fisher_odds, fisher_p = fisher_exact(table, alternative="less") if succ_n_used and fail_n_used else (float("nan"), float("nan"))

    # ── part 3: eigenvector-deviation alignment ──
    eigvec_result = None
    if len(fail_eigvec_pairs) >= 5:
        cos_top = np.array([abs(np.dot(top, dev)) for top, stab, dev in fail_eigvec_pairs])
        cos_stable = np.array([abs(np.dot(stab, dev)) for top, stab, dev in fail_eigvec_pairs])
        obs_mean = float(cos_top.mean())

        rng = np.random.RandomState(seed + 2000)
        devs = np.stack([dev for top, stab, dev in fail_eigvec_pairs])
        tops = np.stack([top for top, stab, dev in fail_eigvec_pairs])
        null_means = []
        for _ in range(N_PERM):
            perm = rng.permutation(len(devs))
            null_means.append(float(np.mean(np.abs(np.sum(tops * devs[perm], axis=1)))))
        null_means = np.array(null_means)
        perm_p = float((np.sum(null_means >= obs_mean) + 1) / (N_PERM + 1))

        w_stat, w_p = wilcoxon(cos_top - cos_stable, alternative="greater")

        png_align = plot_eigenvector_alignment_hist(cos_top, cos_stable, null_means, obs_mean, task, seed_series, out_dir)
        eigvec_result = {
            "n_pairs": len(fail_eigvec_pairs),
            "mean_abscos_unstable_vs_deviation": obs_mean,
            "mean_abscos_stable_vs_deviation": float(cos_stable.mean()),
            "permutation_null_p": perm_p,
            "paired_wilcoxon_unstable_gt_stable_p": float(w_p),
            "plot_png": png_align,
        }
        log_message(
            f"[dmd {task} seed={seed_series}] eigvec alignment: n_pairs={len(fail_eigvec_pairs)} "
            f"mean|cos|(unstable,dev)={obs_mean:.3f} mean|cos|(stable,dev)={cos_stable.mean():.3f} "
            f"perm_p={perm_p:.4f} paired_wilcoxon_p={w_p:.4f}"
        )
    else:
        log_message(f"[dmd {task} seed={seed_series}] too few |λ|>1,rank>=2 windows for eigenvector alignment (n={len(fail_eigvec_pairs)}) — skipped")

    png_spectrum = plot_eigenvalue_spectrum(eigvals_pool_arr, frac_inside, task, seed_series, out_dir) if len(eigvals_pool_arr) else None
    png_lmax = plot_lambda_max_vs_progress(success_series, fail_series, task, seed_series, out_dir)

    log_message(
        f"[dmd {task} seed={seed_series}] spectrum: {frac_inside*100:.1f}% of {len(eigvals_pool_arr)} success-episode "
        f"eigenvalues inside |λ|<1 | episode max_lambda_max: success={succ_max.mean():.3f} fail={fail_max.mean():.3f} "
        f"MannWhitney(success<fail) p={u_max_p:.4f} | crossed 1.0 at least once: success={succ_crossed}/{succ_n_used} "
        f"fail={fail_crossed}/{fail_n_used} Fisher p={fisher_p:.4f}"
    )

    return {
        "n_episodes": int(len(episodes)), "n_success_episodes": int(n_success), "n_fail_episodes": int(n_fail),
        "skipped": False,
        "spectrum_frac_success_eigs_inside_unit_circle": frac_inside,
        "spectrum_n_eigenvalues": int(len(eigvals_pool_arr)),
        "episode_max_lambda_max": {"success": succ_max.tolist(), "fail": fail_max.tolist()},
        "episode_mean_lambda_max": {"success": succ_mean.tolist(), "fail": fail_mean.tolist()},
        "mannwhitney_max_lambda_success_lt_fail_p": float(u_max_p),
        "mannwhitney_mean_lambda_success_lt_fail_p": float(u_mean_p),
        "frac_episodes_ever_crossing_1": {
            "success": f"{succ_crossed}/{succ_n_used}", "fail": f"{fail_crossed}/{fail_n_used}",
        },
        "fisher_crossing_success_lt_fail_p": float(fisher_p),
        "eigenvector_deviation_alignment": eigvec_result,
        "spectrum_plot_png": png_spectrum,
        "lambda_max_vs_progress_plot_png": png_lmax,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    synthetic_result = synthetic_dmd_validation()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())

    files_by_task = {}
    for fname, info in manifest["files"].items():
        files_by_task.setdefault(info["task"], {})[info["seed_base"]] = fname

    results = {
        "method_note": (
            "厳密ヤコビアンではなくExact DMD (Tu et al. 2014)による局所線形化演算子の固有値を"
            "ヤコビアン固有値の近似として用いる。窓幅=10（ステップ=1でスライド）、SVD階数打ち切り"
            "（累積エネルギー90%、上限4）。§4.1.5と同一のPCA(10)空間・scene_residualize後の"
            "特徴（LAYER=13, K_STEP=4）を使用。"
        ),
        "window": WINDOW, "window_step": WINDOW_STEP, "rank_max": RANK_MAX,
        "rank_energy_threshold": RANK_ENERGY_THRESHOLD, "n_permutation": N_PERM,
        "min_fail_episodes_threshold": MIN_FAIL_EPISODES,
        "synthetic_validation": synthetic_result,
        "tasks": {},
    }
    for task, fnames_by_seed in files_by_task.items():
        results["tasks"][task] = {}
        for seed_series, fname in fnames_by_seed.items():
            results["tasks"][task][str(seed_series)] = analyze_task_seed(
                collect_dir, task, seed_series, fname, out_dir, seed=0
            )

    # ── pooled-across-task/seed eigenvector alignment summary (extra power, mirrors §9.2's N=30 pooling spirit) ──
    all_perm_inputs = []
    for task, by_seed in results["tasks"].items():
        for seed_series, r in by_seed.items():
            ev = r.get("eigenvector_deviation_alignment")
            if ev is not None:
                all_perm_inputs.append((task, seed_series, ev["n_pairs"], ev["mean_abscos_unstable_vs_deviation"],
                                         ev["mean_abscos_stable_vs_deviation"], ev["permutation_null_p"],
                                         ev["paired_wilcoxon_unstable_gt_stable_p"]))
    results["eigenvector_alignment_summary_across_task_seed"] = [
        {"task": t, "seed_series": s, "n_pairs": n, "mean_abscos_unstable": mu, "mean_abscos_stable": ms,
         "perm_p": pp, "paired_wilcoxon_p": wp}
        for t, s, n, mu, ms, pp, wp in all_perm_inputs
    ]

    with open(out_dir / "dmd_jacobian_stability_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'dmd_jacobian_stability_test.json'}")


if __name__ == "__main__":
    main()
