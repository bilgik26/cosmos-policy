"""
attractor_geometry.py — attractor_verification_design.md §7 (空間的特徴) 準拠

skill_count.py が Level1 (フェーズ, タスク内) で得た consensus クラスタ (k-means, K=consensus_k)
について、各クラスタ i の空間的特徴を計算する:
  - centroid μ_i, 共分散 Σ_i (異方性 = 最大/最小固有値比)
  - local PR (クラスタ内 participation ratio, シーン残差化済み特徴上で計算)
  - 分離度: Fisher比 (between-cluster distance / within-cluster spread) と
    held-out 線形分類精度 (5-fold, フェーズラベルではなくクラスタラベル自身の
    自己整合性チェック) の scene-shuffle null 比較

**basin幅** (μ_iから摂動してなお i に収束する最大α) は steering_intervention.py の
dose-response 掃引で測定するため、本スクリプトでは扱わない (§7脚注、クロスリファレンス)。
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.skill_count import (
    LAYER, K_STEP, load_task_seed_data, scene_residualize, preprocess,
)
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import effective_rank


def local_participation_ratio(X_cluster):
    if X_cluster.shape[0] < 3:
        return float("nan")
    Xc = X_cluster - X_cluster.mean(axis=0, keepdims=True)
    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    return effective_rank(sv)


def fisher_ratio(X, labels):
    centroids = {l: X[labels == l].mean(axis=0) for l in np.unique(labels)}
    grand_mean = X.mean(axis=0)
    between = np.mean([np.sum((c - grand_mean) ** 2) for c in centroids.values()])
    within = np.mean([np.mean(np.sum((X[labels == l] - centroids[l]) ** 2, axis=1))
                       for l in np.unique(labels)])
    return float(between / (within + 1e-12))


def held_out_cluster_separability(X, labels, groups, n_splits=5, seed=0):
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=k)
    accs = []
    for train_idx, test_idx in gkf.split(X, labels, groups):
        if len(np.unique(labels[train_idx])) < 2:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train_idx]), labels[train_idx])
        accs.append(clf.score(scaler.transform(X[test_idx]), labels[test_idx]))
    return float(np.mean(accs)) if accs else float("nan")


def analyze_task(collect_dir, task, fnames_by_seed, consensus_k):
    result = {"task": task, "consensus_k": consensus_k, "per_seed_series": {}}
    for seed_series, fname in fnames_by_seed.items():
        d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
        X_raw, episode = d["feats"], d["episode"]
        Xr = scene_residualize(X_raw, episode)
        Xp = preprocess(Xr, seed=0)

        k = max(consensus_k, 2)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xp)
        labels = km.labels_

        clusters = {}
        for c in np.unique(labels):
            mask = labels == c
            Xc = Xp[mask]
            cov = np.cov(Xc, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-12, None)
            anisotropy = float(eigvals.max() / eigvals.min())
            clusters[int(c)] = {
                "n_members": int(mask.sum()),
                "centroid_norm": float(np.linalg.norm(Xc.mean(axis=0))),
                "covariance_trace": float(np.trace(cov)),
                "anisotropy_ratio": anisotropy,
                "local_participation_ratio": local_participation_ratio(Xc),
            }

        fisher = fisher_ratio(Xp, labels)
        sep_acc = held_out_cluster_separability(Xp, labels, episode)

        result["per_seed_series"][str(seed_series)] = {
            "n_pca_dims": int(Xp.shape[1]),
            "clusters": clusters,
            "fisher_ratio": fisher,
            "held_out_linear_separability_acc": sep_acc,
            "chance_level": 1.0 / k,
        }
        log_message(
            f"[geometry {task} seed={seed_series}] K={k} fisher={fisher:.2f} "
            f"held_out_acc={sep_acc:.3f} (chance={1/k:.3f})"
        )
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--skill_count_json", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())
    skill_count = json.loads(Path(args.skill_count_json).read_text())

    files_by_task = {}
    for fname, info in manifest["files"].items():
        files_by_task.setdefault(info["task"], {})[info["seed_base"]] = fname

    results = {}
    for task, fnames_by_seed in files_by_task.items():
        ks = skill_count["level1"][task]["consensus_k_by_seed_series"]
        consensus_k = int(round(np.mean(list(ks.values())))) if ks else 2
        results[task] = analyze_task(collect_dir, task, fnames_by_seed, consensus_k)

    with open(out_dir / "attractor_geometry.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'attractor_geometry.json'}")


if __name__ == "__main__":
    main()
