"""
T6 交絡分解 — レビュー §3.2 対応

verification_report.md §3.G の T6 照合表は「旧 (Global PCA + Ridge, n_comp=50)」と
「新 (fold-PCA + LogisticRegression, n_comp=30)」を比較しているが、これは
(i) PCA が Global か fold内か, (ii) 分類器が Ridge か LogisticRegression か,
(iii) n_comp が 50 か 30 か、という**3つの変更を同時に**行っており、
中間層 (Blk-4/9/13) で観測された +11〜+21pp の改善が何に起因するか分解できていない
（レビュー: 「リーク除去は通常精度を下げるはずなのに、中間層は逆に上がった」ため
 分類器変更が主因である疑いが強いと指摘）。

本スクリプトは同一の LOEO split・同一の特徴量キャッシュを用いて、
  {PCA: Global, Fold} x {Classifier: Ridge, LogisticRegression} x {n_comp: 30, 50}
の 2x2x2 = 8 セルを progress_3 ラベル @ k=4, 中間層 (Blk-4,9,13) + 対照層 (Blk-0,18,22,27)
について計算し、各要因の主効果を分解する。

「Global PCA」= 全 N サンプル (train+test 混在) で PCA を fit する旧方式 (リークあり)。
「Fold PCA」  = train split のみで PCA を fit する新方式 (リークなし、linear_probe_v2.pca_gram を再利用)。

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.causal_gates.t6_decompose_confound \
      --feat_npz cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz \
      --actions_npz cosmos_policy/experiments/robot/robocasa/analysis/results/action_denoising/step_actions.npz \
      --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/t6_decompose
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import PROBE_LAYERS
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.linear_probe_v2 import (
    labels_progress_3, lr_fit_predict, ridge_fit_predict,
)

K_STEP = 4  # progress_3 @ k=4, matching the report's reconciliation table
LAYERS = PROBE_LAYERS  # [0, 4, 9, 13, 18, 22, 27]


def pca_fold_internal(X_tr: np.ndarray, X_te: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """fold内PCA（linear_probe_v2.pca_gram と同一の Gram 行列法）。train のみで fit。"""
    mu = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    X_tr_s = (X_tr - mu) / std
    X_te_s = (X_te - mu) / std
    n_comp = min(n_components, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
    G = X_tr_s @ X_tr_s.T
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1][:n_comp]
    U_p = eigenvectors[:, idx]
    sigma_p = np.sqrt(np.maximum(eigenvalues[idx], 1e-12))
    V = (X_tr_s.T @ U_p) / sigma_p[np.newaxis, :]
    X_tr_pca = U_p * sigma_p[np.newaxis, :]
    X_te_pca = X_te_s @ V
    return X_tr_pca.astype(np.float32), X_te_pca.astype(np.float32)


def pca_global_leaked_fit(X_all: np.ndarray, n_components: int) -> np.ndarray:
    """
    旧方式の再現: PCA を train+test 全体 (Global) で1回だけ fit する（意図的にリークを含む）。
    linear_probe.py の Global PCA 前処理と同じ標準化+SVD。
    fold ごとに再計算する必要はない（全 fold で同一の fit のため、1層あたり1回で十分）。
    """
    mu = X_all.mean(axis=0)
    std = X_all.std(axis=0) + 1e-8
    X_s = (X_all - mu) / std
    U, S, Vt = np.linalg.svd(X_s, full_matrices=False)
    n_comp = min(n_components, Vt.shape[0])
    proj = X_s @ Vt[:n_comp].T
    return proj.astype(np.float32)


def run_loeo(X_by_ep: Dict, labels: np.ndarray, ep_arr: np.ndarray, n_classes: int,
             classifier: str) -> Tuple[float, list]:
    episodes = np.unique(ep_arr)
    accs = []
    for ep in episodes:
        fd = X_by_ep.get(ep)
        if fd is None:
            continue
        y_tr = labels[fd["train_mask"]]
        y_te = labels[fd["test_mask"]]
        if len(np.unique(y_tr)) < n_classes:
            continue
        if classifier == "lr":
            acc, _ = lr_fit_predict(fd["X_tr"], y_tr, fd["X_te"], y_te, n_classes)
        elif classifier == "ridge":
            acc = ridge_fit_predict(fd["X_tr"], y_tr, fd["X_te"], y_te, n_classes, lam=1.0)
        else:
            raise ValueError(classifier)
        accs.append(acc)
    return (float(np.mean(accs)) if accs else 0.0), accs


def bootstrap_ci(accs, n_boot=1000, seed=0):
    if not accs:
        return 0.0, 0.0
    arr = np.array(accs)
    rng = np.random.default_rng(seed)
    boot = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", required=True)
    ap.add_argument("--actions_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.feat_npz)
    ep_arr = raw["episode_labels"]
    ci_arr = raw["call_idx_labels"]
    labels = labels_progress_3(ep_arr, ci_arr)
    n_classes = 3
    episodes = np.unique(ep_arr)

    results = {}
    for layer in LAYERS:
        key = f"feat_k{K_STEP}_layer{layer}"
        X_all = raw[key].astype(np.float32)
        cell_results = {}
        for pca_mode in ["global", "fold"]:
            for n_comp in [30, 50]:
                # Global PCA fit is fold-independent -> compute once per (layer, n_comp)
                global_proj = pca_global_leaked_fit(X_all, n_comp) if pca_mode == "global" else None
                # Pre-compute per-episode-fold projections for this (pca_mode, n_comp)
                fold_cache = {}
                for ep in episodes:
                    train_mask = ep_arr != ep
                    test_mask = ep_arr == ep
                    if train_mask.sum() < 2 or test_mask.sum() < 1:
                        continue
                    if pca_mode == "fold":
                        X_tr, X_te = pca_fold_internal(X_all[train_mask], X_all[test_mask], n_comp)
                    else:
                        X_tr, X_te = global_proj[train_mask], global_proj[test_mask]
                    fold_cache[ep] = {"X_tr": X_tr, "X_te": X_te,
                                       "train_mask": train_mask, "test_mask": test_mask}
                for clf in ["ridge", "lr"]:
                    mean_acc, accs = run_loeo(fold_cache, labels, ep_arr, n_classes, clf)
                    ci_lo, ci_hi = bootstrap_ci(accs)
                    cell_name = f"pca={pca_mode}_ncomp={n_comp}_clf={clf}"
                    cell_results[cell_name] = {
                        "mean_acc": mean_acc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_folds": len(accs),
                    }
                    print(f"  Blk-{layer:2d} {cell_name:35s} acc={mean_acc:.4f} [{ci_lo:.3f},{ci_hi:.3f}]")
        results[f"Blk-{layer}"] = cell_results

    # ── Main-effect decomposition at n_comp=30 (matches new pipeline's n_comp) ──
    decomposition = {}
    for layer in LAYERS:
        r = results[f"Blk-{layer}"]
        gr = r["pca=global_ncomp=30_clf=ridge"]["mean_acc"]
        gl = r["pca=global_ncomp=30_clf=lr"]["mean_acc"]
        fr = r["pca=fold_ncomp=30_clf=ridge"]["mean_acc"]
        fl = r["pca=fold_ncomp=30_clf=lr"]["mean_acc"]
        total_delta = fl - gr
        pca_effect_holding_ridge = fr - gr        # PCA leak removal alone (classifier=ridge fixed)
        clf_effect_holding_global = gl - gr        # classifier swap alone (PCA=global fixed)
        pca_effect_holding_lr = fl - gl            # PCA leak removal alone (classifier=LR fixed)
        clf_effect_holding_fold = fl - fr          # classifier swap alone (PCA=fold fixed)
        decomposition[f"Blk-{layer}"] = {
            "old_pipeline_global_ridge_ncomp30": gr,
            "new_pipeline_fold_lr_ncomp30": fl,
            "total_delta_pp": round(100 * total_delta, 2),
            "pca_leak_removal_effect_pp (holding clf=ridge)": round(100 * pca_effect_holding_ridge, 2),
            "classifier_swap_effect_pp (holding pca=global)": round(100 * clf_effect_holding_global, 2),
            "pca_leak_removal_effect_pp (holding clf=lr)": round(100 * pca_effect_holding_lr, 2),
            "classifier_swap_effect_pp (holding pca=fold)": round(100 * clf_effect_holding_fold, 2),
            "old_report_ncomp50_ridge_global_for_reference": r["pca=global_ncomp=50_clf=ridge"]["mean_acc"],
            "ncomp_50_vs_30_effect_pp (fold+lr)": round(
                100 * (r["pca=fold_ncomp=50_clf=lr"]["mean_acc"] - r["pca=fold_ncomp=30_clf=lr"]["mean_acc"]), 2),
        }

    out = {
        "description": "T6 confound decomposition: PCA(global/fold) x classifier(ridge/LR) x n_comp(30/50), progress_3 @ k=4",
        "label": "progress_3", "k_step": K_STEP, "n_classes": n_classes,
        "full_grid": results,
        "main_effect_decomposition": decomposition,
    }
    with open(out_dir / "t6_decomposition.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir / 't6_decomposition.json'}")


if __name__ == "__main__":
    main()
