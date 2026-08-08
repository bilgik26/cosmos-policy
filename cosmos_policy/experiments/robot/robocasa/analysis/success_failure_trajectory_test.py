"""
success_failure_trajectory_test.py — attractor_verification_report.md への追加検証 (課題A)。

ユーザー提案「成功エピソードと失敗エピソードの軌跡の分岐分析」への対応。
collect_multitask.py の episode_success 未保存バグ(§10 バグ#9)を修正した上で再収集した
collect_v2/ (成功フラグ付き)を使い、「軌跡が進行多様体を滑らかにたどれば成功し、
そこから逸脱すると失敗する」という仮説を検定する。

方法 (循環性回避 P5/P6 準拠):
  1. シーン残差化後の特徴をPCA(10次元)に還元。
  2. 成功エピソードのみを使い、進行度(call_idx/max_call_idx)を5分位ビンに区切った各ビンでの
     PCA空間中心・(縮小推定)分散を「成功多様体の基準」として定義する。
     GroupKFold(group=episode, 成功エピソードのみ)でheld-out成功エピソードの逸脱量
     (正規化ユークリッド距離)を測定し、"成功エピソード自身が基準からどれだけばらつくか"の
     ベースラインとする。
  3. 全成功エピソードで学習した最終モデル(失敗エピソードは学習に一切使われないため
     このモデルを失敗エピソードに適用しても循環性の問題はない)を失敗エピソードに適用し、
     逸脱量を測定する。
  4. 統計検定はエピソード単位(各エピソードの平均逸脱量)で行う(Mann-Whitney U、
     コール単位でのpseudo-replicationを避けるため、GroupKFold(group=episode)を全解析で
     一貫させている本プロジェクトの方針に従う)。
  5. 「失敗エピソード内で時間とともに逸脱が拡大するか」をWilcoxon符号順位検定
     (エピソードごとのSpearman(progress, deviation)の中央値が0より大きいか)で検定する。
  6. 副次検定として、成功群/失敗群別に progress regression R² と trajectory smoothness
     (turn-cosine) を比較する (manifold_trajectory_test.py の関数を再利用)。

限界:
  - CloseDrawer/CoffeePressButtonは元データで成功率95-100%と非常に高く、失敗エピソードが
    ほぼ存在しない(再収集後の実際のnは実行結果依存)。失敗episode数が閾値未満のtask/seedは
    「データ不足によりスキップ」と明記し、無理に検定しない。
  - 逸脱量の基準(進行度ビン中心)は5分位という粗い離散化であり、真の連続的な進行度基準に
    比べると解像度が低い。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import (
    LAYER, K_STEP, scene_residualize,
)
from cosmos_policy.experiments.robot.robocasa.analysis.manifold_trajectory_test import (
    episode_progress, progress_regression_probe, trajectory_smoothness,
)

MIN_FAIL_EPISODES = 3
N_PROGRESS_BINS = 5
PCA_DIM = 10
SHRINKAGE_PSEUDOCOUNT = 5.0


def load_task_seed_data_with_success(collect_dir: Path, fname: str, layer: int, k: int):
    fd = np.load(collect_dir / fname)
    key = f"feat_k{k}_layer{layer}"
    idx_key = key + "_idx"
    feats = fd[key]
    keep_idx = fd[idx_key]
    return {
        "feats": feats,
        "episode": fd["episode"][keep_idx],
        "call_idx": fd["call_idx"][keep_idx],
        "success": fd["success"][keep_idx],
    }


def progress_bin(progress, n_bins=N_PROGRESS_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(progress, edges[1:-1], right=True), 0, n_bins - 1)
    return b


def fit_bin_reference(Xp, progress, n_bins=N_PROGRESS_BINS, pseudocount=SHRINKAGE_PSEUDOCOUNT):
    bins = progress_bin(progress, n_bins)
    global_var = Xp.var(axis=0) + 1e-8
    centers, variances = [], []
    for b in range(n_bins):
        mask = bins == b
        n_b = mask.sum()
        if n_b == 0:
            centers.append(Xp.mean(axis=0))
            variances.append(global_var)
            continue
        local_mean = Xp[mask].mean(axis=0)
        local_var = Xp[mask].var(axis=0) if n_b > 1 else global_var
        alpha = n_b / (n_b + pseudocount)
        centers.append(local_mean)
        variances.append(alpha * local_var + (1 - alpha) * global_var)
    return np.stack(centers), np.stack(variances)


def deviation_from_reference(Xp, progress, centers, variances, n_bins=N_PROGRESS_BINS):
    bins = progress_bin(progress, n_bins)
    dev = np.zeros(len(Xp))
    for i in range(len(Xp)):
        b = bins[i]
        dev[i] = np.sqrt(np.sum((Xp[i] - centers[b]) ** 2 / variances[b]))
    return dev


def held_out_success_deviation(Xp, episode, progress, success_episodes, seed=0):
    n_splits = min(5, len(success_episodes))
    if n_splits < 2:
        return None
    mask_succ = np.isin(episode, success_episodes)
    Xp_s, ep_s, prog_s = Xp[mask_succ], episode[mask_succ], progress[mask_succ]
    idx_map = np.where(mask_succ)[0]

    gkf = GroupKFold(n_splits=n_splits)
    dev_out = np.full(len(episode), np.nan)
    for train_idx, test_idx in gkf.split(Xp_s, groups=ep_s):
        centers, variances = fit_bin_reference(Xp_s[train_idx], prog_s[train_idx])
        dev_test = deviation_from_reference(Xp_s[test_idx], prog_s[test_idx], centers, variances)
        dev_out[idx_map[test_idx]] = dev_test
    return dev_out


def episode_mean(values, episode, episodes_subset):
    means = {}
    for e in episodes_subset:
        mask = episode == e
        v = values[mask]
        v = v[~np.isnan(v)]
        if len(v) > 0:
            means[e] = float(np.mean(v))
    return means


def per_episode_progress_deviation_trend(deviation, episode, progress, episodes_subset, min_calls=5):
    rhos = []
    for e in episodes_subset:
        mask = episode == e
        if mask.sum() < min_calls:
            continue
        d, p = deviation[mask], progress[mask]
        valid = ~np.isnan(d)
        if valid.sum() < min_calls:
            continue
        rho, _ = spearmanr(p[valid], d[valid])
        if not np.isnan(rho):
            rhos.append(rho)
    return rhos


def plot_success_fail_trajectories(Xp2, episode, call_idx, success, task, seed_series, out_dir, max_each=5):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = plt.get_cmap("viridis")
    success_eps = np.unique(episode[success])[:max_each]
    fail_eps = np.unique(episode[~success])[:max_each]
    sc = None
    for e in success_eps:
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        traj = Xp2[order]
        prog = np.arange(len(order)) / max(len(order) - 1, 1)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="green", alpha=0.5, linewidth=1.3, zorder=1)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=prog, cmap=cmap, vmin=0, vmax=1, s=12, zorder=2, marker="o")
    for e in fail_eps:
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        traj = Xp2[order]
        prog = np.arange(len(order)) / max(len(order) - 1, 1)
        ax.plot(traj[:, 0], traj[:, 1], "--", color="red", alpha=0.5, linewidth=1.3, zorder=1)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=prog, cmap=cmap, vmin=0, vmax=1, s=12, zorder=2, marker="x")
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="within-episode progress (0=start,1=end)")
    ax.set_xlabel("PC1 (scene-residualized)")
    ax.set_ylabel("PC2 (scene-residualized)")
    ax.set_title(f"{task} seed={seed_series}: success(solid,o,green) vs failure(dashed,x,red) trajectories\n"
                 f"({len(success_eps)} success / {len(fail_eps)} failure episodes shown)")
    fig.tight_layout()
    fig_path = out_dir / f"success_fail_trajectory_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def plot_deviation_boxplot(succ_ep_means, fail_ep_means, u_p, task, seed_series, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    succ_v = list(succ_ep_means.values())
    fail_v = list(fail_ep_means.values())
    bp = ax.boxplot([succ_v, fail_v], tick_labels=[f"success\n(held-out)\nn={len(succ_v)}",
                                                    f"failure\nn={len(fail_v)}"],
                     patch_artist=True, widths=0.5, showmeans=False)
    for patch, color in zip(bp["boxes"], ["#2ca02c", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    rng = np.random.RandomState(0)
    for x0, vals, color in [(1, succ_v, "#2ca02c"), (2, fail_v, "#d62728")]:
        jitter = rng.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), x0) + jitter, vals, color=color, edgecolor="black",
                   linewidth=0.5, s=35, zorder=3)
    ax.set_ylabel("episode-mean deviation from success-progress reference")
    ax.set_title(f"{task} seed={seed_series}\nMann-Whitney p(success<failure)={u_p:.4g}", fontsize=11)
    fig.tight_layout()
    fig_path = out_dir / f"deviation_boxplot_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def plot_deviation_vs_progress(dev_success_heldout, dev_fail, progress, episode, success_episodes,
                                fail_episodes, task, seed_series, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    mask_s = np.isin(episode, success_episodes) & ~np.isnan(dev_success_heldout)
    mask_f = np.isin(episode, fail_episodes) & ~np.isnan(dev_fail)
    ax.scatter(progress[mask_s], dev_success_heldout[mask_s], s=10, alpha=0.5, color="#2ca02c",
               label=f"success (held-out), n_calls={mask_s.sum()}")
    ax.scatter(progress[mask_f], dev_fail[mask_f], s=10, alpha=0.5, color="#d62728",
               label=f"failure, n_calls={mask_f.sum()}")

    def binned_mean(prog, dev, mask, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        centers, means = [], []
        for i in range(n_bins):
            b = mask & (prog >= edges[i]) & (prog <= edges[i + 1] if i == n_bins - 1 else prog < edges[i + 1])
            if b.sum() > 0:
                centers.append((edges[i] + edges[i + 1]) / 2)
                means.append(dev[b].mean())
        return np.array(centers), np.array(means)

    cs, ms = binned_mean(progress, dev_success_heldout, mask_s)
    cf, mf = binned_mean(progress, dev_fail, mask_f)
    ax.plot(cs, ms, "-o", color="#1a6b1a", linewidth=2, markersize=5, label="success bin-mean")
    ax.plot(cf, mf, "-o", color="#8f1414", linewidth=2, markersize=5, label="failure bin-mean")
    ax.set_xlabel("within-episode progress (0=start, 1=end)")
    ax.set_ylabel("deviation from success-progress reference")
    ax.set_title(f"{task} seed={seed_series}: deviation vs. progress")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / f"deviation_vs_progress_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def analyze_task_seed(collect_dir, task, seed_series, fname, out_dir, seed=0):
    d = load_task_seed_data_with_success(collect_dir, fname, LAYER, K_STEP)
    X_raw, episode, call_idx, success_call = d["feats"], d["episode"], d["call_idx"], d["success"]

    episodes = np.unique(episode)
    ep_success = {}
    for e in episodes:
        ep_success[e] = bool(success_call[episode == e][0])
    success_episodes = np.array([e for e in episodes if ep_success[e]])
    fail_episodes = np.array([e for e in episodes if not ep_success[e]])

    n_success, n_fail = len(success_episodes), len(fail_episodes)
    log_message(f"[success-fail {task} seed={seed_series}] n_episodes={len(episodes)} "
                f"success={n_success} fail={n_fail}")

    if n_fail < MIN_FAIL_EPISODES:
        return {
            "n_episodes": int(len(episodes)), "n_success_episodes": int(n_success),
            "n_fail_episodes": int(n_fail),
            "skipped": True,
            "skip_reason": f"insufficient failure episodes (n={n_fail} < {MIN_FAIL_EPISODES})",
        }

    Xr = scene_residualize(X_raw, episode)
    Xs = StandardScaler().fit_transform(Xr)
    pca = PCA(n_components=min(PCA_DIM, Xs.shape[0] - 1, Xs.shape[1]), random_state=seed)
    Xp = pca.fit_transform(Xs)
    progress = episode_progress(episode, call_idx)
    success_mask = success_call.astype(bool)

    # ── held-out success deviation baseline ──
    dev_success_heldout = held_out_success_deviation(Xp, episode, progress, success_episodes, seed=seed)

    # ── full success-only model applied to failure episodes (no circularity: fail never trains it) ──
    mask_succ_all = np.isin(episode, success_episodes)
    centers_full, variances_full = fit_bin_reference(Xp[mask_succ_all], progress[mask_succ_all])
    mask_fail_all = np.isin(episode, fail_episodes)
    dev_fail = np.full(len(episode), np.nan)
    dev_fail[mask_fail_all] = deviation_from_reference(
        Xp[mask_fail_all], progress[mask_fail_all], centers_full, variances_full
    )

    succ_ep_means = episode_mean(dev_success_heldout, episode, success_episodes)
    fail_ep_means = episode_mean(dev_fail, episode, fail_episodes)

    u_stat, u_p = mannwhitneyu(list(succ_ep_means.values()), list(fail_ep_means.values()),
                                alternative="less")

    # ── does deviation grow over time within failure episodes? ──
    fail_rhos = per_episode_progress_deviation_trend(dev_fail, episode, progress, fail_episodes)
    succ_rhos = per_episode_progress_deviation_trend(dev_success_heldout, episode, progress, success_episodes)
    if len(fail_rhos) >= 3:
        w_stat, w_p = wilcoxon(np.array(fail_rhos) - 0.0, alternative="greater")
    else:
        w_stat, w_p = float("nan"), float("nan")

    # ── secondary: progress-regression / smoothness split by success vs failure ──
    def subset_result(ep_subset):
        mask = np.isin(episode, ep_subset)
        if mask.sum() < 10 or len(ep_subset) < 2:
            return None
        reg = progress_regression_probe(Xp[mask], episode[mask], progress[mask], seed=seed)
        smooth = trajectory_smoothness(Xp[mask][:, :3], episode[mask], call_idx[mask], seed=seed)
        return {"progress_regression": reg, "trajectory_smoothness": smooth}

    secondary_success = subset_result(success_episodes)
    secondary_fail = subset_result(fail_episodes)

    png_name = plot_success_fail_trajectories(Xp[:, :2], episode, call_idx, success_mask, task, seed_series, out_dir)
    box_png_name = plot_deviation_boxplot(succ_ep_means, fail_ep_means, u_p, task, seed_series, out_dir)
    devprog_png_name = plot_deviation_vs_progress(
        dev_success_heldout, dev_fail, progress, episode, success_episodes, fail_episodes,
        task, seed_series, out_dir,
    )

    log_message(
        f"[success-fail {task} seed={seed_series}] held-out-success mean_dev="
        f"{np.mean(list(succ_ep_means.values())):.3f} (n={len(succ_ep_means)}) vs "
        f"fail mean_dev={np.mean(list(fail_ep_means.values())):.3f} (n={len(fail_ep_means)}) "
        f"MannWhitneyU p(success<fail)={u_p:.4f} | "
        f"fail within-episode progress-deviation Spearman median={np.median(fail_rhos) if fail_rhos else float('nan'):.3f} "
        f"Wilcoxon p(>0)={w_p:.4f} (n_fail_eps_used={len(fail_rhos)})"
    )

    return {
        "n_episodes": int(len(episodes)),
        "n_success_episodes": int(n_success),
        "n_fail_episodes": int(n_fail),
        "skipped": False,
        "deviation_success_heldout_episode_means": {str(k): v for k, v in succ_ep_means.items()},
        "deviation_fail_episode_means": {str(k): v for k, v in fail_ep_means.items()},
        "deviation_success_lt_fail_mannwhitney_p": float(u_p),
        "fail_within_episode_progress_deviation_spearman_rhos": fail_rhos,
        "fail_within_episode_progress_deviation_spearman_median": float(np.median(fail_rhos)) if fail_rhos else float("nan"),
        "success_within_episode_progress_deviation_spearman_rhos": succ_rhos,
        "success_within_episode_progress_deviation_spearman_median": float(np.median(succ_rhos)) if succ_rhos else float("nan"),
        "fail_deviation_grows_with_progress_wilcoxon_p": float(w_p),
        "secondary_progress_regression_smoothness": {
            "success_episodes": secondary_success,
            "fail_episodes": secondary_fail,
        },
        "trajectory_plot_png": png_name,
        "deviation_boxplot_png": box_png_name,
        "deviation_vs_progress_plot_png": devprog_png_name,
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

    files_by_task = {}
    for fname, info in manifest["files"].items():
        files_by_task.setdefault(info["task"], {})[info["seed_base"]] = fname

    results = {
        "method_note": (
            "成功エピソードのみで進行度5分位ビンの参照分布(縮小推定分散)を"
            "GroupKFold(held-out)で学習し、held-out成功/失敗エピソードの逸脱量(正規化距離)を"
            "エピソード単位でMann-Whitney U検定 (循環性回避: 失敗エピソードは一切参照分布の"
            "学習に使われない)。"
        ),
        "min_fail_episodes_threshold": MIN_FAIL_EPISODES,
        "tasks": {},
    }
    for task, fnames_by_seed in files_by_task.items():
        results["tasks"][task] = {}
        for seed_series, fname in fnames_by_seed.items():
            results["tasks"][task][str(seed_series)] = analyze_task_seed(
                collect_dir, task, seed_series, fname, out_dir, seed=0
            )

    with open(out_dir / "success_failure_trajectory_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'success_failure_trajectory_test.json'}")


if __name__ == "__main__":
    main()
