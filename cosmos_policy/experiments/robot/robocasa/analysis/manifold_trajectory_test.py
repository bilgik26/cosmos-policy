"""
manifold_trajectory_test.py — attractor_verification_report.md への追加検証。

ユーザー提案「離散クラスタではなく連続多様体・軌跡としての表現空間の検証」への対応。
背景: skill_count.py の教師なしクラスタリングは0/4タスクで棄却され、
gradient_structure_test.py はフェーズの線形射影(GroupKFold held-out)が単峰的(勾配的)
であることを直接示した(§4.1.1)。本スクリプトはさらに2つの角度から「連続多様体/軌跡」
解釈を検証する:

  (A) 進行度(progress)の回帰プローブ: フェーズを分類ではなく、エピソード内の
      正規化進行度 (call_idx / max_call_idx ∈ [0,1]) への回帰として捉え直し、
      GroupKFold(group=episode) held-out R² / Spearman を評価する (P5/P6準拠:
      fold内でスケーラ・回帰を学習、テストfoldは一切学習に使わない)。
      Null: 各episode内でprogress値を独立にシャッフルしてから同じGroupKFold回帰を
      繰り返し(n_null回)、実測R²がnull分布を有意に超えるか検定する
      (scene grouping自体は不変、progressの"意味"だけを壊すnullなのでGate2circularityとは別の検定)。

  (B) 軌跡の滑らかさ(trajectory smoothness): 各エピソードをcall_idx順にPCA空間(シーン残差化後)
      へ射影し、連続する変位ベクトル間のターニング角のコサイン平均を計算する
      (1に近い=直線的/滑らか、0=ランダムな折れ線、負=後戻り)。
      Null: 同一episode内のcall順をランダムに並べ替えたときの同じ統計量。
      観測順序が「単なる点群」ではなく真に時間的な滑らかな軌跡を成すかを検定する。

限界の明記:
  - 各episodeの成功/失敗ラベルは collect_multitask.py の実装上保存されていない
    (manifest["episode_success"]は宣言のみで値が入らない未使用フィールド、
    run_task_seed()が返すep_successのper-episode配列はnpzに書き出されず破棄される)。
    そのため「成功/失敗エピソードで軌跡がどう分岐するか」は本データでは検証不可能
    (再収集が必要、future work)。
  - 可視化用PCAは (A)(B) とも scene_residualize 後の特徴を使う。生特徴でのPCAは
    シーン(物体配置・照明等)の差がトップ主成分を支配し、進行度による軌跡が埋もれるため
    (skill_count.py Level1と同じ統制)。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import (
    LAYER, K_STEP, load_task_seed_data, scene_residualize,
)

N_NULL = 20


def episode_progress(episode, call_idx):
    progress = np.zeros(len(episode), dtype=np.float64)
    for e in np.unique(episode):
        mask = episode == e
        max_ci = call_idx[mask].max()
        progress[mask] = call_idx[mask] / max(max_ci, 1)
    return progress


def _gkf_r2(Xp, y, episode, seed=0):
    n_groups = len(np.unique(episode))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in gkf.split(Xp, y, groups=episode):
        scaler = StandardScaler().fit(Xp[train_idx])
        reg = Ridge(alpha=1.0, random_state=seed)
        reg.fit(scaler.transform(Xp[train_idx]), y[train_idx])
        pred[test_idx] = reg.predict(scaler.transform(Xp[test_idx]))
    valid = ~np.isnan(pred)
    y_v, pred_v = y[valid], pred[valid]
    ss_res = np.sum((y_v - pred_v) ** 2)
    ss_tot = np.sum((y_v - y_v.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return r2, pred, valid


def progress_regression_probe(Xp, episode, progress, seed=0, n_null=N_NULL):
    r2_real, pred, valid = _gkf_r2(Xp, progress, episode, seed=seed)
    rho, rho_p = spearmanr(progress[valid], pred[valid])

    rng = np.random.RandomState(seed + 1000)
    null_r2s = []
    for i in range(n_null):
        prog_shuf = progress.copy()
        for e in np.unique(episode):
            mask = episode == e
            prog_shuf[mask] = rng.permutation(progress[mask])
        r2_null, _, _ = _gkf_r2(Xp, prog_shuf, episode, seed=seed)
        null_r2s.append(r2_null)
    null_r2s = np.array(null_r2s)
    p_val = float((np.sum(null_r2s >= r2_real) + 1) / (len(null_r2s) + 1))

    return {
        "n_holdout": int(valid.sum()),
        "r2_real": r2_real,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "within_episode_progress_shuffle_null_r2_mean": float(null_r2s.mean()),
        "within_episode_progress_shuffle_null_r2_std": float(null_r2s.std()),
        "p_value_real_exceeds_null": p_val,
        "progress_decodable_beyond_null": bool(p_val < 0.05),
    }


def trajectory_smoothness(pca_scores, episode, call_idx, seed=0, n_null=N_NULL):
    def mean_turn_cos(order_fn):
        cos_list = []
        for e in np.unique(episode):
            idx = np.where(episode == e)[0]
            if len(idx) < 3:
                continue
            ordered = order_fn(idx)
            traj = pca_scores[ordered]
            deltas = np.diff(traj, axis=0)
            norms = np.linalg.norm(deltas, axis=1)
            for i in range(len(deltas) - 1):
                if norms[i] < 1e-9 or norms[i + 1] < 1e-9:
                    continue
                cos_list.append(float(np.dot(deltas[i], deltas[i + 1]) / (norms[i] * norms[i + 1])))
        return float(np.mean(cos_list)) if cos_list else float("nan")

    real_val = mean_turn_cos(lambda idx: idx[np.argsort(call_idx[idx])])

    rng = np.random.RandomState(seed + 2000)
    null_vals = np.array([mean_turn_cos(lambda idx: rng.permutation(idx)) for _ in range(n_null)])
    p_val = float((np.sum(null_vals >= real_val) + 1) / (len(null_vals) + 1))

    return {
        "real_mean_turn_cosine": real_val,
        "call_order_shuffle_null_mean": float(np.nanmean(null_vals)),
        "call_order_shuffle_null_std": float(np.nanstd(null_vals)),
        "p_value_real_smoother_than_null": p_val,
        "trajectory_smoother_than_shuffled_null": bool(p_val < 0.05),
    }


def plot_trajectory(pca_scores, episode, call_idx, progress, task, seed_series, out_dir, max_episodes=8):
    fig, ax = plt.subplots(figsize=(6, 6))
    eps = np.unique(episode)[:max_episodes]
    cmap = plt.get_cmap("viridis")
    for e in eps:
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        traj = pca_scores[order]
        prog = progress[order]
        ax.plot(traj[:, 0], traj[:, 1], "-", color="gray", alpha=0.4, linewidth=1, zorder=1)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=prog, cmap=cmap, vmin=0, vmax=1, s=14, zorder=2)
        ax.scatter(traj[0, 0], traj[0, 1], marker="^", color="black", s=40, zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", color="red", s=40, zorder=3)
    plt.colorbar(sc, ax=ax, label="within-episode progress (0=start,1=end)")
    ax.set_xlabel("PC1 (scene-residualized)")
    ax.set_ylabel("PC2 (scene-residualized)")
    ax.set_title(f"{task} seed={seed_series}: episode trajectories in PCA space\n"
                 f"(▲=start, ■=end, {len(eps)}/{len(np.unique(episode))} episodes shown)")
    fig.tight_layout()
    fig_path = out_dir / f"trajectory_pca_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def analyze_task_seed(collect_dir, task, seed_series, fname, out_dir, seed=0):
    d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
    X_raw, episode, call_idx = d["feats"], d["episode"], d["call_idx"]

    Xr = scene_residualize(X_raw, episode)
    Xs = StandardScaler().fit_transform(Xr)
    pca = PCA(n_components=min(10, Xs.shape[0] - 1, Xs.shape[1]), random_state=seed)
    Xp_full = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    progress = episode_progress(episode, call_idx)

    reg_result = progress_regression_probe(Xp_full, episode, progress, seed=seed)
    smooth_result = trajectory_smoothness(Xp_full[:, :3], episode, call_idx, seed=seed)

    png_name = plot_trajectory(Xp_full[:, :2], episode, call_idx, progress, task, seed_series, out_dir)

    log_message(
        f"[manifold {task} seed={seed_series}] progress-regression: R2={reg_result['r2_real']:.3f} "
        f"(null={reg_result['within_episode_progress_shuffle_null_r2_mean']:.3f}±"
        f"{reg_result['within_episode_progress_shuffle_null_r2_std']:.3f}, p={reg_result['p_value_real_exceeds_null']:.3f}) "
        f"spearman={reg_result['spearman_rho']:.3f} | "
        f"trajectory-smoothness: turn_cos={smooth_result['real_mean_turn_cosine']:.3f} "
        f"(shuffle_null={smooth_result['call_order_shuffle_null_mean']:.3f}±"
        f"{smooth_result['call_order_shuffle_null_std']:.3f}, p={smooth_result['p_value_real_smoother_than_null']:.3f})"
    )

    return {
        "n_calls": int(len(episode)),
        "n_episodes": int(len(np.unique(episode))),
        "pca_explained_variance_ratio_top10": evr.tolist(),
        "progress_regression_probe": reg_result,
        "trajectory_smoothness": smooth_result,
        "trajectory_plot_png": png_name,
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
        "limitation_note": (
            "各episodeの成功/失敗フラグはcollect_multitask.pyの実装上npzに保存されていない"
            "(manifest['episode_success']は宣言のみで未使用、per-episode成功配列は破棄される)。"
            "そのため成功/失敗エピソードでの軌跡分岐比較は本データでは実施不可能(要再収集、future work)。"
        ),
        "tasks": {},
    }
    for task, fnames_by_seed in files_by_task.items():
        results["tasks"][task] = {}
        for seed_series, fname in fnames_by_seed.items():
            results["tasks"][task][str(seed_series)] = analyze_task_seed(
                collect_dir, task, seed_series, fname, out_dir, seed=0
            )

    with open(out_dir / "manifold_trajectory_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'manifold_trajectory_test.json'}")


if __name__ == "__main__":
    main()
