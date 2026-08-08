"""
linear_probe_v2.py — 設計書 §3.G 準拠の改訂版線形プローブ

改善点 (vs linear_probe.py):
  - Global PCA廃止 → fold内PCA (P3: テストセット情報リークゼロ)
  - Ridge回帰 → LogisticRegression (sklearn, L2, class_weight='balanced')
  - Permutationテスト (episodeレベルラベルシャッフル, n_perm=100)
  - Bootstrap 95%CI (fold精度のリサンプル, 1000反復)
  - gripper_2: 物理閾値0.0 (primary) + 中央値 (secondary)
  - Benjamini-Hochberg FDR補正 (7層×5step×3ラベル=105テスト)

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.common.linear_probe_v2 \
      --feat_npz results/action_features/features.npz \
      --actions_npz results/action_denoising/step_actions.npz \
      --out_dir results/action_probe_v2 \
      --task_name PnPCounterToCab \
      --success_rate 0.60
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    PROBE_LAYERS,
    PROBE_LAYER_SHORT,
    NUM_DENOISE_STEPS,
)

GRIPPER_PHYSICAL_THRESHOLD = 0.0  # gripper dim6: -1=open, +1=closed


# ── PCA (fold-internal, Gram matrix法) ───────────────────────────────────────

def pca_gram(X_tr: np.ndarray, X_te: np.ndarray, n_components: int = 30
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    fold内PCA (リークゼロ): X_tr のみで PCA を推定し X_te に適用。
    Gram行列法 (N<D 時に高速): O(N²D) の行列積 + O(N³) の固有値分解。
    """
    mu = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    X_tr_s = (X_tr - mu) / std
    X_te_s = (X_te - mu) / std

    n_comp = min(n_components, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
    G = X_tr_s @ X_tr_s.T                           # (N_tr, N_tr)
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1][:n_comp]
    U_p = eigenvectors[:, idx]                       # (N_tr, n_comp)
    sigma_p = np.sqrt(np.maximum(eigenvalues[idx], 1e-12))
    # Data-space PCA components: V[d,j] = (X_tr_s.T @ U_p)[d,j] / sigma_p[j]
    V = (X_tr_s.T @ U_p) / sigma_p[np.newaxis, :]   # (D, n_comp)
    X_tr_pca = U_p * sigma_p[np.newaxis, :]          # (N_tr, n_comp)
    X_te_pca = X_te_s @ V                            # (N_te, n_comp)
    return X_tr_pca.astype(np.float32), X_te_pca.astype(np.float32)


def precompute_pca_projections(feats: Dict, ep_arr: np.ndarray,
                                n_pca: int = 30) -> Dict:
    """
    全 (layer, step, fold) の fold内PCA を事前計算してキャッシュ。
    ラベルに依存しないため、3ラベル種で共有できる。
    """
    episodes = np.unique(ep_arr)
    cache = {}
    n_cells = len(PROBE_LAYERS) * NUM_DENOISE_STEPS
    done = 0
    for step in range(NUM_DENOISE_STEPS):
        for layer in PROBE_LAYERS:
            F = feats[step][layer]  # (N, D)
            if F is None:
                continue
            for ep in episodes:
                train_mask = ep_arr != ep
                test_mask = ep_arr == ep
                if train_mask.sum() < 2 or test_mask.sum() < 1:
                    continue
                X_tr_pca, X_te_pca = pca_gram(F[train_mask], F[test_mask], n_pca)
                cache[(layer, step, ep)] = {
                    "X_tr": X_tr_pca,
                    "X_te": X_te_pca,
                    "train_mask": train_mask,
                    "test_mask": test_mask,
                }
            done += 1
            print(f"\r  PCA precompute: {done}/{n_cells} cells done", end="", flush=True)
    print()
    return cache


# ── ラベル生成 ────────────────────────────────────────────────────────────────

def labels_progress_3(ep_arr: np.ndarray, ci_arr: np.ndarray) -> np.ndarray:
    """quantile分割: 全サンプルの正規化進行度で3等分 (バランス保証)。"""
    N = len(ep_arr)
    progress = np.zeros(N)
    for ep in np.unique(ep_arr):
        mask = ep_arr == ep
        max_ci = ci_arr[mask].max()
        progress[mask] = ci_arr[mask] / max(max_ci, 1)
    q1, q2 = np.quantile(progress, [1/3, 2/3])
    labels = np.zeros(N, dtype=int)
    labels[progress > q1] = 1
    labels[progress > q2] = 2
    return labels


def labels_gripper_physical(actions: Dict, k: int = 4,
                              threshold: float = GRIPPER_PHYSICAL_THRESHOLD) -> np.ndarray:
    """物理閾値 (開/閉境界 = 0.0) でグリッパーを2値化 (primary)。"""
    gripper = actions[k][:, :, 6].mean(axis=1)  # (N,)
    return (gripper > threshold).astype(int)


def labels_gripper_median(actions: Dict, k: int = 4) -> np.ndarray:
    """中央値閾値でグリッパーを2値化 (secondary, balanced)。"""
    gripper = actions[k][:, :, 6].mean(axis=1)
    return (gripper > float(np.median(gripper))).astype(int)


def labels_gripper_3(actions: Dict, k: int = 4) -> np.ndarray:
    """quantile分割でグリッパーを3値化。"""
    gripper = actions[k][:, :, 6].mean(axis=1)
    q1, q2 = np.quantile(gripper, [1/3, 2/3])
    labels = np.zeros(len(gripper), dtype=int)
    labels[gripper > q1] = 1
    labels[gripper > q2] = 2
    return labels


# ── 分類器 ───────────────────────────────────────────────────────────────────

def lr_fit_predict(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray,
                   n_classes: int) -> Tuple[float, np.ndarray]:
    """LogisticRegression (sklearn) による分類。"""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs",
        max_iter=300, random_state=42,
    )
    clf.fit(X_tr, y_tr.astype(int))
    y_pred = clf.predict(X_te)
    return float((y_pred == y_te.astype(int)).mean()), y_pred


def ridge_fit_predict(X_tr: np.ndarray, y_tr: np.ndarray,
                      X_te: np.ndarray, y_te: np.ndarray,
                      n_classes: int, lam: float = 1.0) -> float:
    """クラス重み付きRidge回帰による分類 (permutation用高速版)。"""
    N, D = X_tr.shape
    X_tr_b = np.hstack([X_tr, np.ones((N, 1))])
    X_te_b = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
    counts = np.bincount(y_tr.astype(int), minlength=n_classes).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    W_samp = 1.0 / counts[y_tr.astype(int)]
    W_samp /= W_samp.mean()
    Y_oh = np.zeros((N, n_classes))
    Y_oh[np.arange(N), y_tr.astype(int)] = 1.0
    WD = np.diag(W_samp)
    A = X_tr_b.T @ WD @ X_tr_b + lam * np.eye(D + 1)
    b_rhs = X_tr_b.T @ WD @ Y_oh
    try:
        W = np.linalg.solve(A, b_rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, b_rhs, rcond=None)[0]
    y_pred = (X_te_b @ W).argmax(axis=1)
    return float((y_pred == y_te.astype(int)).mean())


# ── LOEO交差検証 ──────────────────────────────────────────────────────────────

def run_loeo_lr(cache: Dict, labels: np.ndarray, ep_arr: np.ndarray,
                layer: int, step: int, n_classes: int,
                ) -> Tuple[float, List[float], np.ndarray]:
    """LogisticRegression LOEO CV (fold内PCA済みキャッシュを使用)。"""
    episodes = np.unique(ep_arr)
    accs = []
    all_preds = np.full(len(labels), -1, dtype=int)
    for ep in episodes:
        fd = cache.get((layer, step, ep))
        if fd is None:
            continue
        y_tr = labels[fd["train_mask"]]
        y_te = labels[fd["test_mask"]]
        if len(np.unique(y_tr)) < n_classes:
            continue
        acc, preds = lr_fit_predict(fd["X_tr"], y_tr, fd["X_te"], y_te, n_classes)
        accs.append(acc)
        all_preds[fd["test_mask"]] = preds
    mean_acc = float(np.mean(accs)) if accs else 0.0
    return mean_acc, accs, all_preds


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(fold_accs: List[float], n_boot: int = 1000,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """fold精度をリサンプルし (1-alpha) 信頼区間を返す。"""
    if not fold_accs:
        return 0.0, 0.0
    arr = np.array(fold_accs)
    boot_means = [arr[np.random.randint(0, len(arr), len(arr))].mean()
                  for _ in range(n_boot)]
    return float(np.percentile(boot_means, 100 * alpha / 2)), \
           float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


# ── Permutationテスト ─────────────────────────────────────────────────────────

def permute_episode_labels(labels: np.ndarray, ep_arr: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """episodeブロック単位でラベルをシャッフル。"""
    episodes = np.unique(ep_arr)
    ep_labels = {ep: labels[ep_arr == ep].copy() for ep in episodes}
    shuffled_eps = rng.permutation(episodes)
    new_labels = labels.copy()
    for orig_ep, src_ep in zip(episodes, shuffled_eps):
        mask = ep_arr == orig_ep
        n = mask.sum()
        src = ep_labels[src_ep]
        new_labels[mask] = np.resize(src, n)
    return new_labels


def run_permutation_test(cache: Dict, labels: np.ndarray, ep_arr: np.ndarray,
                          layer: int, step: int, n_classes: int,
                          real_acc: float, n_perm: int = 100,
                          seed: int = 0) -> Tuple[float, np.ndarray]:
    """
    episodeレベルのpermutationテスト (Ridgeを使用、高速)。
    p_value = P(perm_acc >= real_acc)
    """
    rng = np.random.default_rng(seed)
    episodes = np.unique(ep_arr)
    perm_accs = []
    for _ in range(n_perm):
        perm_labels = permute_episode_labels(labels, ep_arr, rng)
        accs = []
        for ep in episodes:
            fd = cache.get((layer, step, ep))
            if fd is None:
                continue
            y_tr = perm_labels[fd["train_mask"]]
            y_te = perm_labels[fd["test_mask"]]
            if len(np.unique(y_tr)) < n_classes:
                continue
            acc = ridge_fit_predict(fd["X_tr"], y_tr, fd["X_te"], y_te, n_classes)
            accs.append(acc)
        if accs:
            perm_accs.append(float(np.mean(accs)))
    perm_arr = np.array(perm_accs)
    p_val = float((perm_arr >= real_acc).sum() + 1) / (len(perm_arr) + 1)
    return p_val, perm_arr


# ── Benjamini-Hochberg FDR補正 ────────────────────────────────────────────────

def bh_correction(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray:
    """p値配列にBH FDR補正を適用し、有意かどうかのboolean配列を返す。"""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * fdr
    passing = sorted_p <= thresholds
    if not passing.any():
        return np.zeros(n, dtype=bool)
    k = int(np.where(passing)[0].max())
    result = np.zeros(n, dtype=bool)
    result[sorted_idx[: k + 1]] = True
    return result


# ── プロット ──────────────────────────────────────────────────────────────────

def plot_accuracy_heatmap(results: Dict, label_type: str, n_classes: int,
                           out_dir: Path, task_name: str, significant: np.ndarray):
    """層×ステップのAccuracyヒートマップ (BH有意セルに★印)。"""
    n_layers = len(PROBE_LAYERS)
    hmap = np.zeros((n_layers, NUM_DENOISE_STEPS))
    ci_lo = np.zeros_like(hmap)
    ci_hi = np.zeros_like(hmap)
    cell_idx = 0
    for ni, l in enumerate(PROBE_LAYERS):
        for k in range(NUM_DENOISE_STEPS):
            cell = results[label_type][k][l]
            hmap[ni, k] = cell["mean_acc"]
            ci_lo[ni, k] = cell["ci_lo"]
            ci_hi[ni, k] = cell["ci_hi"]

    chance = 1.0 / n_classes
    fig, ax = plt.subplots(figsize=(NUM_DENOISE_STEPS * 1.8 + 1, n_layers * 1.3 + 1.5))
    im = ax.imshow(hmap, aspect="auto", cmap="RdYlGn", vmin=max(0.0, chance - 0.1), vmax=1.0)
    ax.set_xticks(range(NUM_DENOISE_STEPS))
    ax.set_xticklabels([f"k={k}" for k in range(NUM_DENOISE_STEPS)], fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS], fontsize=9)
    ax.set_xlabel("Denoising step k", fontsize=9)
    ax.set_ylabel("Probe layer", fontsize=9)
    ax.set_title(
        f"Linear Probe Accuracy — {label_type} ({n_classes} classes)\n"
        f"Task: {task_name}  |  fold-PCA+LR+LOEO  |  ★=BH-significant (FDR=0.05)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="Accuracy")
    sig_idx = 0
    for ni in range(n_layers):
        for k in range(NUM_DENOISE_STEPS):
            is_sig = significant[ni * NUM_DENOISE_STEPS + k] if sig_idx < len(significant) else False
            val = hmap[ni, k]
            text = f"{val:.2f}"
            if is_sig:
                text += "★"
            ax.text(k, ni, text, ha="center", va="center", fontsize=7.5,
                    color="white" if val < chance + 0.1 else "black")
    plt.tight_layout()
    p = out_dir / f"probe_heatmap_v2_{label_type}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_perm_null(perm_nulls: Dict, real_accs: Dict,
                   label_type: str, out_dir: Path, task_name: str):
    """permutationヌル分布と実測精度の比較プロット (k=0, k=4, 各層)。"""
    ks_to_plot = [0, NUM_DENOISE_STEPS - 1]
    n_layers = len(PROBE_LAYERS)
    fig, axes = plt.subplots(len(ks_to_plot), n_layers,
                              figsize=(n_layers * 2.5, len(ks_to_plot) * 2.5))
    if len(ks_to_plot) == 1:
        axes = axes.reshape(1, -1)
    if n_layers == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(f"Permutation Null vs Real Accuracy — {label_type}  |  {task_name}", fontsize=10)
    for ri, k in enumerate(ks_to_plot):
        for ci, l in enumerate(PROBE_LAYERS):
            ax = axes[ri, ci]
            null = perm_nulls.get((k, l))
            real = real_accs.get((k, l), 0.0)
            if null is not None and len(null) > 0:
                ax.hist(null, bins=20, color="steelblue", alpha=0.7, density=True)
                ax.axvline(real, color="tomato", linewidth=2, label=f"Real={real:.2f}")
                ax.legend(fontsize=7)
            ax.set_title(f"k={k}, Blk-{l}", fontsize=7)
            ax.tick_params(labelsize=6)
            if ci == 0:
                ax.set_ylabel(f"k={k}", fontsize=8)
    plt.tight_layout()
    p = out_dir / f"probe_perm_null_{label_type}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_npz", required=True)
    parser.add_argument("--actions_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_name", default="PnPCounterToCab")
    parser.add_argument("--success_rate", type=float, default=0.60)
    parser.add_argument("--n_pca", type=int, default=30)
    parser.add_argument("--n_perm", type=int, default=100)
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading features: {args.feat_npz}")
    raw = np.load(args.feat_npz)
    feats: Dict = {}
    for k in range(NUM_DENOISE_STEPS):
        feats[k] = {}
        for l in PROBE_LAYERS:
            key = f"feat_k{k}_layer{l}"
            feats[k][l] = raw[key].astype(np.float32) if key in raw else None
    ep_arr = raw["episode_labels"]
    ci_arr = raw["call_idx_labels"]
    N = len(ep_arr)
    n_episodes = len(np.unique(ep_arr))
    print(f"N={N}, episodes={n_episodes}")

    print(f"Loading actions: {args.actions_npz}")
    act_raw = np.load(args.actions_npz)
    actions = {int(k): act_raw[k] for k in act_raw.files}

    # ── Label configurations ──────────────────────────────────────────────────
    prog3 = labels_progress_3(ep_arr, ci_arr)
    grp_phys = labels_gripper_physical(actions)
    grp_med  = labels_gripper_median(actions)
    grp3     = labels_gripper_3(actions)

    label_configs = {
        "progress_3":       (prog3,    3, "quantile-based"),
        "gripper_2_phys":   (grp_phys, 2, "physical threshold 0.0 (primary)"),
        "gripper_2_median": (grp_med,  2, "median threshold (secondary/balanced)"),
    }
    for ltype, (lbl, nc, desc) in label_configs.items():
        cnts = np.bincount(lbl.astype(int), minlength=nc)
        print(f"  Label '{ltype}' ({nc} cls): {cnts}  [{desc}]")

    # ── Pre-compute fold-internal PCA ─────────────────────────────────────────
    print(f"\nPre-computing fold-internal PCA (n_pca={args.n_pca})…")
    print(f"  Total cells: {len(PROBE_LAYERS) * NUM_DENOISE_STEPS}, folds: {n_episodes}")
    pca_cache = precompute_pca_projections(feats, ep_arr, n_pca=args.n_pca)
    print(f"  Cache entries: {len(pca_cache)}")

    # ── LOEO with LogisticRegression ──────────────────────────────────────────
    np.random.seed(42)
    results: Dict = {}
    for ltype, (labels, n_classes, _) in label_configs.items():
        results[ltype] = {}
        chance = float(np.bincount(labels, minlength=n_classes).max()) / N
        print(f"\n=== {ltype} (n_classes={n_classes}, chance={chance:.1%}) ===")
        for k in range(NUM_DENOISE_STEPS):
            results[ltype][k] = {}
            print(f"  k={k}: ", end="", flush=True)
            for l in PROBE_LAYERS:
                mean_acc, fold_accs, all_preds = run_loeo_lr(
                    pca_cache, labels, ep_arr, l, k, n_classes
                )
                ci_lo, ci_hi = bootstrap_ci(fold_accs, n_boot=args.n_boot)
                results[ltype][k][l] = {
                    "mean_acc": mean_acc,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "fold_accs": fold_accs,
                    "all_preds": all_preds,
                }
                print(f"Blk{l}:{mean_acc:.2f}[{ci_lo:.2f},{ci_hi:.2f}] ", end="", flush=True)
            print()
        results[ltype]["meta"] = {
            "n_classes": n_classes,
            "chance_level": chance,
            "all_labels": labels,
        }

    # ── Permutation test ──────────────────────────────────────────────────────
    print(f"\nPermutation tests (n_perm={args.n_perm})…")
    all_p_values = {}
    perm_nulls: Dict = {}
    for ltype, (labels, n_classes, _) in label_configs.items():
        print(f"  {ltype}:")
        for k in range(NUM_DENOISE_STEPS):
            for li, l in enumerate(PROBE_LAYERS):
                real_acc = results[ltype][k][l]["mean_acc"]
                p_val, null_dist = run_permutation_test(
                    pca_cache, labels, ep_arr, l, k, n_classes,
                    real_acc, n_perm=args.n_perm,
                    seed=li * NUM_DENOISE_STEPS + k,
                )
                results[ltype][k][l]["p_value"] = p_val
                perm_nulls[(ltype, k, l)] = null_dist
            print(f"    k={k} done", flush=True)

    # ── BH FDR correction ─────────────────────────────────────────────────────
    print("\nBH FDR correction (FDR=0.05)…")
    for ltype in label_configs:
        p_flat = np.array([
            results[ltype][k][l]["p_value"]
            for l in PROBE_LAYERS for k in range(NUM_DENOISE_STEPS)
        ])
        sig_flat = bh_correction(p_flat, fdr=0.05)
        idx = 0
        for l in PROBE_LAYERS:
            for k in range(NUM_DENOISE_STEPS):
                results[ltype][k][l]["bh_significant"] = bool(sig_flat[idx])
                idx += 1
        n_sig = sig_flat.sum()
        print(f"  {ltype}: {n_sig}/{len(p_flat)} cells BH-significant")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    for ltype, (labels, n_classes, _) in label_configs.items():
        sig_arr = np.array([
            results[ltype][k][l].get("bh_significant", False)
            for l in PROBE_LAYERS for k in range(NUM_DENOISE_STEPS)
        ])
        # Reorder to [layer_idx * NUM_STEPS + k] for heatmap
        sig_reordered = np.zeros(len(sig_arr), dtype=bool)
        cell_idx = 0
        for ni, l in enumerate(PROBE_LAYERS):
            for k in range(NUM_DENOISE_STEPS):
                sig_reordered[ni * NUM_DENOISE_STEPS + k] = results[ltype][k][l].get("bh_significant", False)
        plot_accuracy_heatmap(results, ltype, n_classes, out_dir,
                               args.task_name, sig_reordered)
        real_accs_dict = {(k, l): results[ltype][k][l]["mean_acc"]
                          for k in range(NUM_DENOISE_STEPS) for l in PROBE_LAYERS}
        null_dict = {(k, l): perm_nulls.get((ltype, k, l), np.array([]))
                     for k in range(NUM_DENOISE_STEPS) for l in PROBE_LAYERS}
        plot_perm_null(null_dict, real_accs_dict, ltype, out_dir, args.task_name)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def _safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(kk): _safe(vv) for kk, vv in obj.items()
                    if kk not in ("all_preds", "all_labels", "fold_accs")}
        if isinstance(obj, (list, tuple)):
            return [_safe(x) for x in obj]
        return obj

    stats_out = {
        "task": args.task_name,
        "success_rate": args.success_rate,
        "N_calls": int(N),
        "n_episodes": int(n_episodes),
        "n_pca": args.n_pca,
        "n_perm": args.n_perm,
        "n_boot": args.n_boot,
        "probe_layers": PROBE_LAYERS,
        "method": "fold-internal PCA + LogisticRegression (L2, balanced) + LOEO",
        "perm_method": "episode-block label shuffle + Ridge",
        "correction": "Benjamini-Hochberg FDR=0.05",
        "gripper_physical_threshold": GRIPPER_PHYSICAL_THRESHOLD,
        "results": _safe(results),
    }
    stats_path = out_dir / "probe_acc_ci.json"
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Summary (k=4) ===")
    print(f"{'Layer':<12}", end="")
    for ltype in label_configs:
        print(f"  {ltype:<30}", end="")
    print()
    for l in PROBE_LAYERS:
        print(f"{PROBE_LAYER_SHORT.get(l, f'B{l}'):<12}", end="")
        for ltype, (labels, n_classes, _) in label_configs.items():
            r = results[ltype][NUM_DENOISE_STEPS - 1][l]
            sig = "★" if r.get("bh_significant") else " "
            print(f"  {r['mean_acc']:.2f}[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]{sig}(p={r['p_value']:.3f})   ",
                  end="")
        print()
    print(f"\nOutput: {out_dir}")
    print("linear_probe_v2 complete!")


if __name__ == "__main__":
    main()
