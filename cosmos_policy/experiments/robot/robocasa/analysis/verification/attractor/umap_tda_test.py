"""
umap_tda_test.py — attractor_verification_report.md への追加検証 (課題B-2)。

ユーザー提案「UMAP/TDA(位相的データ解析)の導入」への対応。
  (A) UMAP: シーン残差化後の表現Zを2次元にUMAP埋め込みし、進行度(0-1)で色付けする
      (manifold_trajectory_test.py のPCA版trajectory plotのUMAP版)。PCAは線形部分空間しか
      見ないため、UMAPで非線形構造を見ても同じ「進行度に沿った連続的な経路」が保たれるかを
      補強的に確認する。
  (B) TDA (persistent homology, ripser使用): 各task/seedの全エピソードをプールした点群
      (シーン残差化後の特徴をPCAで次元圧縮)についてH0/H1の持続ホモロジーを計算し、
      「軌跡全体が輪(ループ)を持たない単一のストリームである」という主張を検証する。

方法論的注意 (厳格レビュー対策):
  - TDAは「有意な穴が無い」ことを主張するのが最も難しい(何をもって「無い」と言えるかの
    基準が必要)。そこで正の対照(positive control)として、既知のリング構造を持つ合成データ
    (2次元円+ノイズ、実データと同じ点数・同程度のノイズ幅)に同一パイプラインを適用し、
    「本パイプラインは実際にループがあれば検出できる」ことをまず確認した上で、実データの
    H1持続時間分布(最大値・上位バーとその他バーとのギャップ)を正の対照と比較する。
  - TDA入力はPCA上位k次元(intrinsic_dimension_test.pyの実測値を参考にk=6に固定)に圧縮した
    点群。1024/2048次元の生空間に直接Rips複体を作ると距離集中により無意味になるため。
  - 点数が多いtask(PnPCounterToCab/TurnOnStove, N>450)は計算量抑制のため乱数部分抽出
    (N=300, 固定seed)を行う。正の対照もN=300で揃え、直接比較可能にする。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import (
    LAYER, K_STEP, load_task_seed_data, scene_residualize,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.manifold_trajectory_test import episode_progress

TDA_PCA_DIM = 6
TDA_MAX_POINTS = 300
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


def synthetic_ring_positive_control(n=TDA_MAX_POINTS, noise_std=0.05, radius=1.0,
                                     ambient_dim=TDA_PCA_DIM, seed=0):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    if ambient_dim > 2:
        extra = rng.normal(0, noise_std, size=(n, ambient_dim - 2))
        ring = np.concatenate([ring, extra], axis=1)
    ring = ring + rng.normal(0, noise_std, size=ring.shape)
    return ring


def persistence_summary(dgms, label, out):
    h0, h1 = dgms[0], dgms[1]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    diameter = float(h0_finite[:, 1].max()) if len(h0_finite) else float("nan")
    h1_pers = np.sort((h1[:, 1] - h1[:, 0]))[::-1] if len(h1) else np.array([])
    top5 = h1_pers[:5].tolist()
    top1 = float(h1_pers[0]) if len(h1_pers) else 0.0
    top2 = float(h1_pers[1]) if len(h1_pers) > 1 else 0.0
    gap_ratio = float(top1 / top2) if top2 > 1e-12 else float("inf") if top1 > 1e-12 else float("nan")
    log_message(
        f"[TDA {label}] n_H1_bars={len(h1_pers)} diameter(H0 full merge)={diameter:.4f} "
        f"top1_H1_persistence={top1:.4f} ({100*top1/diameter:.1f}% of diameter) "
        f"top1/top2_gap_ratio={gap_ratio:.2f}"
    )
    return {
        "n_h1_bars": int(len(h1_pers)),
        "diameter_h0_full_merge": diameter,
        "top5_h1_persistence": top5,
        "top1_h1_persistence": top1,
        "top1_h1_persistence_pct_of_diameter": float(100 * top1 / diameter) if diameter > 1e-12 else float("nan"),
        "top1_over_top2_gap_ratio": gap_ratio,
    }


def plot_umap_trajectory(emb, episode, call_idx, progress, task, seed_series, out_dir, max_episodes=8):
    fig, ax = plt.subplots(figsize=(6, 6))
    eps = np.unique(episode)[:max_episodes]
    cmap = plt.get_cmap("viridis")
    sc = None
    for e in eps:
        idx = np.where(episode == e)[0]
        order = idx[np.argsort(call_idx[idx])]
        traj = emb[order]
        prog = progress[order]
        ax.plot(traj[:, 0], traj[:, 1], "-", color="gray", alpha=0.4, linewidth=1, zorder=1)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=prog, cmap=cmap, vmin=0, vmax=1, s=14, zorder=2)
        ax.scatter(traj[0, 0], traj[0, 1], marker="^", color="black", s=40, zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", color="red", s=40, zorder=3)
    plt.colorbar(sc, ax=ax, label="within-episode progress (0=start,1=end)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{task} seed={seed_series}: UMAP embedding of episode trajectories\n"
                 f"(▲=start, ■=end, {len(eps)}/{len(np.unique(episode))} episodes shown)")
    fig.tight_layout()
    fig_path = out_dir / f"umap_trajectory_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def analyze_task_seed(collect_dir, task, seed_series, fname, out_dir, ring_summary, seed=0):
    d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
    X_raw, episode, call_idx = d["feats"], d["episode"], d["call_idx"]

    Xr = scene_residualize(X_raw, episode)
    Xs = StandardScaler().fit_transform(Xr)
    progress = episode_progress(episode, call_idx)

    # ── (A) UMAP visualization ──
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                         n_components=2, random_state=seed)
    emb = reducer.fit_transform(Xs)
    png_name = plot_umap_trajectory(emb, episode, call_idx, progress, task, seed_series, out_dir)

    # ── (B) TDA on PCA-reduced pooled point cloud ──
    pca = PCA(n_components=min(TDA_PCA_DIM, Xs.shape[0] - 1, Xs.shape[1]), random_state=seed)
    Xp = pca.fit_transform(Xs)

    rng = np.random.RandomState(seed + 4000)
    if Xp.shape[0] > TDA_MAX_POINTS:
        sub_idx = rng.choice(Xp.shape[0], TDA_MAX_POINTS, replace=False)
        Xp_sub = Xp[sub_idx]
    else:
        Xp_sub = Xp

    dgms = ripser(Xp_sub, maxdim=1)["dgms"]
    tda_result = persistence_summary(dgms, f"{task} seed={seed_series}", out_dir)
    tda_result["n_points_used"] = int(Xp_sub.shape[0])
    tda_result["ring_positive_control_top1_h1_persistence"] = ring_summary["top1_h1_persistence"]
    tda_result["real_top1_over_ring_top1_ratio"] = (
        float(tda_result["top1_h1_persistence"] / ring_summary["top1_h1_persistence"])
        if ring_summary["top1_h1_persistence"] > 1e-12 else float("nan")
    )

    return {
        "umap_trajectory_plot_png": png_name,
        "tda": tda_result,
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

    # ── positive control: synthetic ring, same N as most task/seed subsamples ──
    ring = synthetic_ring_positive_control(n=TDA_MAX_POINTS, seed=0)
    ring_dgms = ripser(ring, maxdim=1)["dgms"]
    ring_summary = persistence_summary(ring_dgms, "SYNTHETIC RING (positive control)", out_dir)

    files_by_task = {}
    for fname, info in manifest["files"].items():
        files_by_task.setdefault(info["task"], {})[info["seed_base"]] = fname

    results = {
        "method_note": (
            "UMAP(2D, n_neighbors=15, min_dist=0.1)埋め込み + PCA上位6次元でのRips持続ホモロジー"
            "(ripser, maxdim=1)。正の対照として同じ点数の合成リング(2D円+ノイズ)に同一TDA"
            "パイプラインを適用し、パイプラインの検出感度を確認した上で実データのH1持続性と比較。"
        ),
        "synthetic_ring_positive_control": ring_summary,
        "tasks": {},
    }
    for task, fnames_by_seed in files_by_task.items():
        results["tasks"][task] = {}
        for seed_series, fname in fnames_by_seed.items():
            results["tasks"][task][str(seed_series)] = analyze_task_seed(
                collect_dir, task, seed_series, fname, out_dir, ring_summary, seed=0
            )

    with open(out_dir / "umap_tda_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'umap_tda_test.json'}")


if __name__ == "__main__":
    main()
