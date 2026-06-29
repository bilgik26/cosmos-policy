"""
Cosmos Policy テーマ2 線形プロービング解析スクリプト
既存の theme2_features.npz と step_actions.npz を使用（再シミュレーション不要）。

【目的】
  各 DiT 層の中間特徴量が「スキルフェーズ（リーチング/グラスピング/プレーシング）」を
  どの程度線形分離可能な形で保持しているかを層ごとに定量化する。

【手法】
  - スキルラベル
      A) 進行度ベース: エピソード内の call 進行を 3 分割（早期/中期/後期）
      B) グリッパーベース: k=4 の予測グリッパー値で閾値分類（開/遷移/閉）
      C) 2 ラベル: グリッパー開閉の 2 値分類（最も明確な物理的スキル差）
  - 分類器: Ridge 回帰（numpy のみ、解析解）
      W = (X^T X + λI)^{-1} X^T Y_onehot → 予測 = argmax(X W)
  - 評価: Leave-One-Episode-Out (LOEO) 交差検証

実行:
  python cosmos_policy/experiments/robot/robocasa/theme2_linear_probe.py \\
      --feat_npz cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/theme2_features.npz \\
      --actions_npz cosmos_policy/experiments/robot/robocasa/analysis/results/action_denoising/step_actions.npz \\
      --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_probe \\
      --task_name PnPCounterToCab \\
      --success_rate 0.70
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    PROBE_LAYERS,
    PROBE_LAYER_SHORT,
    NUM_DENOISE_STEPS,
)


# ── データロード ──────────────────────────────────────────────────────────

def load_features(npz_path: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    feats = {}
    for k in range(NUM_DENOISE_STEPS):
        feats[k] = {}
        for l in PROBE_LAYERS:
            key = f"feat_k{k}_layer{l}"
            feats[k][l] = data[key] if key in data else None
    return feats, data["episode_labels"], data["call_idx_labels"]


def load_actions(npz_path: str) -> Dict:
    """step_actions.npz: keys '0'-'4', each (N, 32, 7)."""
    data = np.load(npz_path)
    return {int(k): data[k] for k in data.files}


# ── スキルラベル生成 ──────────────────────────────────────────────────────

def labels_from_progress(ep_arr: np.ndarray, ci_arr: np.ndarray,
                          n_classes: int = 3) -> np.ndarray:
    """
    エピソード内の正規化進行度から n_classes クラスのラベルを生成。
    classes: 0=early(reach), 1=mid(grasp), 2=late(place) for n_classes=3
    """
    N = len(ep_arr)
    progress = np.zeros(N)
    for ep in np.unique(ep_arr):
        mask = ep_arr == ep
        max_ci = ci_arr[mask].max()
        progress[mask] = ci_arr[mask] / max(max_ci, 1)

    labels = np.zeros(N, dtype=int)
    bins = np.linspace(0, 1, n_classes + 1)[1:-1]  # [0.33, 0.67] for n=3
    for i, thr in enumerate(bins):
        labels[progress > thr] = i + 1
    return labels


def labels_from_gripper(step_actions: Dict, k: int = 4,
                         n_classes: int = 3) -> np.ndarray:
    """
    k=4 (最終デノイジング) での予測グリッパー値（次元 6）の
    32-step チャンク平均から n_classes クラスのラベルを生成。
    gripper ∈ [-1, 1]: -1=open(reach), +1=closed(grasp/place)
    """
    gripper = step_actions[k][:, :, 6].mean(axis=1)  # (N,)
    if n_classes == 2:
        threshold = 0.0
        return (gripper > threshold).astype(int)
    else:
        # 3 classes by quantile
        q33, q67 = np.quantile(gripper, [1/3, 2/3])
        labels = np.zeros(len(gripper), dtype=int)
        labels[gripper > q33] = 1
        labels[gripper > q67] = 2
        return labels


# ── 線形プローブ (Ridge Regression, numpy のみ) ───────────────────────────

def pca_project(X_all: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    全データで PCA を計算し n_components 次元に射影する（グローバル PCA）。
    2048 次元 → 50 次元 にして ridge の行列を小さくする。
    """
    X_c = X_all - X_all.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components]          # (n_components, D)
    return X_c @ components.T              # (N, n_components)


def ridge_probe(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                n_classes: int, lambda_reg: float = 1.0) -> float:
    """
    Ridge regression の解析解でマルチクラス線形分類。
    X はすでに PCA 済みの低次元特徴量を想定（~50 dim）。
    W = (X^T X + λI)^{-1} X^T Y_onehot
    """
    N_tr, D = X_train.shape
    # Normalize per-fold (fitted on train)
    mu = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8
    X_tr_n = (X_train - mu) / sig
    X_te_n = (X_test - mu) / sig

    # Add bias column
    X_tr_b = np.hstack([X_tr_n, np.ones((N_tr, 1))])
    X_te_b = np.hstack([X_te_n, np.ones((X_test.shape[0], 1))])

    # One-hot targets
    Y_oh = np.zeros((N_tr, n_classes))
    Y_oh[np.arange(N_tr), y_train.astype(int)] = 1.0

    # Solve ridge: (D+1) × (D+1) matrix — small because D is ~50
    A = X_tr_b.T @ X_tr_b + lambda_reg * np.eye(D + 1)
    b = X_tr_b.T @ Y_oh
    try:
        W = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, b, rcond=None)[0]

    logits = X_te_b @ W
    y_pred = logits.argmax(axis=1)
    return float((y_pred == y_test.astype(int)).mean())


def loeo_cross_validate(features: np.ndarray, labels: np.ndarray,
                        ep_arr: np.ndarray, n_classes: int,
                        lambda_reg: float = 1.0,
                        n_pca: int = 50) -> Tuple[float, float, List[float]]:
    """
    Leave-One-Episode-Out (LOEO) 交差検証。
    features はグローバル PCA で n_pca 次元に削減してから使用する。
    Returns: (mean_acc, std_acc, per_fold_accs)
    """
    # Global PCA: fit on ALL data to get stable components
    features_pca = pca_project(features, n_components=min(n_pca, features.shape[0] - 1, features.shape[1]))

    episodes = np.unique(ep_arr)
    accs = []
    for test_ep in episodes:
        test_mask = ep_arr == test_ep
        train_mask = ~test_mask

        if train_mask.sum() < n_classes or test_mask.sum() < 1:
            continue
        if len(np.unique(labels[train_mask])) < n_classes:
            continue

        acc = ridge_probe(
            features_pca[train_mask], labels[train_mask],
            features_pca[test_mask], labels[test_mask],
            n_classes=n_classes,
            lambda_reg=lambda_reg,
        )
        accs.append(acc)

    if not accs:
        return 0.0, 0.0, []
    return float(np.mean(accs)), float(np.std(accs)), accs


def chance_level(n_classes: int, labels: np.ndarray) -> float:
    """Majority class baseline."""
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    return float(counts.max() / counts.sum())


# ── プロット ─────────────────────────────────────────────────────────────

def plot_probe_results(results: Dict, out_dir: Path, task_name: str,
                       success_rate: float):
    """
    線形プロービング結果の全プロット。
    results[label_type][k]: {layer: (mean_acc, std_acc)}
    """
    n_layers = len(PROBE_LAYERS)
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))
    k_markers = ["o", "s", "^", "D", "v"]
    k_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, NUM_DENOISE_STEPS))

    label_types = list(results.keys())

    # ── (1) Accuracy by layer for each label type ──────────────────────────
    fig, axes = plt.subplots(1, len(label_types), figsize=(6 * len(label_types), 5))
    if len(label_types) == 1:
        axes = [axes]
    fig.suptitle(
        f"Theme 2 Linear Probe: Accuracy by Layer\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  "
        f"(LOEO cross-validation)",
        fontsize=11,
    )
    for ax, ltype in zip(axes, label_types):
        meta = results[ltype]["meta"]
        n_cls = meta["n_classes"]
        chance = meta["chance_level"]

        for k in range(NUM_DENOISE_STEPS):
            means = [results[ltype][k][l]["mean_acc"] for l in PROBE_LAYERS]
            stds = [results[ltype][k][l]["std_acc"] for l in PROBE_LAYERS]
            ax.plot(range(n_layers), means, marker=k_markers[k], color=k_colors[k],
                    linewidth=2, markersize=6, label=f"k={k}(σ≈{[80,42,21,10,4][k]})")
            ax.fill_between(range(n_layers),
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.08, color=k_colors[k])

        ax.axhline(chance, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Chance ({chance:.1%})")
        ax.axhline(1.0 / n_cls, color="lightgray", linestyle=":", linewidth=0.8,
                   label=f"Random ({1/n_cls:.1%})")

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS],
                           fontsize=8)
        ax.set_xlabel("Probe Layer")
        ax.set_ylabel("Accuracy (LOEO CV)")
        ax.set_title(f"Label: {ltype}\n({n_cls} classes)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    p = out_dir / "probe_accuracy_by_layer.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # ── (2) Best k comparison across layers ────────────────────────────────
    fig, axes = plt.subplots(1, len(label_types), figsize=(5 * len(label_types), 5))
    if len(label_types) == 1:
        axes = [axes]
    fig.suptitle(
        f"Theme 2 Linear Probe: k=0 vs k=4 Comparison\n"
        f"Task: {task_name}  (LOEO)",
        fontsize=11,
    )
    for ax, ltype in zip(axes, label_types):
        meta = results[ltype]["meta"]
        chance = meta["chance_level"]
        n_cls = meta["n_classes"]

        means_k0 = [results[ltype][0][l]["mean_acc"] for l in PROBE_LAYERS]
        stds_k0 = [results[ltype][0][l]["std_acc"] for l in PROBE_LAYERS]
        means_k4 = [results[ltype][4][l]["mean_acc"] for l in PROBE_LAYERS]
        stds_k4 = [results[ltype][4][l]["std_acc"] for l in PROBE_LAYERS]

        x = np.arange(n_layers)
        width = 0.38
        bars0 = ax.bar(x - width/2, means_k0, width, yerr=stds_k0,
                       label="k=0 (σ=80)", color="royalblue", alpha=0.75, capsize=4)
        bars4 = ax.bar(x + width/2, means_k4, width, yerr=stds_k4,
                       label="k=4 (σ=4)", color="tomato", alpha=0.75, capsize=4)
        ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label=f"Chance")
        ax.set_xticks(x)
        ax.set_xticklabels([PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS],
                           fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Label: {ltype} ({n_cls} classes)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    p = out_dir / "probe_k0_vs_k4.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # ── (3) Heatmap: accuracy (label_type × layer), k=4 ─────────────────
    if len(label_types) > 1:
        fig, ax = plt.subplots(figsize=(n_layers * 1.3, len(label_types) * 1.3 + 1))
        hmap = np.zeros((len(label_types), n_layers))
        for li, ltype in enumerate(label_types):
            for ni, l in enumerate(PROBE_LAYERS):
                hmap[li, ni] = results[ltype][4][l]["mean_acc"]
        im = ax.imshow(hmap, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS], fontsize=9)
        ax.set_yticks(range(len(label_types)))
        ax.set_yticklabels(label_types, fontsize=9)
        ax.set_title(f"Linear Probe Accuracy (k=4) — Task: {task_name}", fontsize=10)
        plt.colorbar(im, ax=ax, label="Accuracy")
        for i in range(len(label_types)):
            for j in range(n_layers):
                ax.text(j, i, f"{hmap[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if hmap[i,j] < 0.6 else "black")
        plt.tight_layout()
        p = out_dir / "probe_accuracy_heatmap.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")

    # ── (4) Label distribution visualization ─────────────────────────────
    n_label_types = len(label_types)
    fig, axes = plt.subplots(1, n_label_types, figsize=(5 * n_label_types, 4))
    if n_label_types == 1:
        axes = [axes]
    fig.suptitle("Skill Label Distributions", fontsize=11)
    for ax, ltype in zip(axes, label_types):
        lbls = results[ltype]["meta"]["all_labels"]
        n_cls = results[ltype]["meta"]["n_classes"]
        counts = np.bincount(lbls.astype(int), minlength=n_cls)
        class_names = {
            "progress_3": ["Early\n(reach)", "Mid\n(grasp)", "Late\n(place)"],
            "gripper_3": ["Open\n(reach)", "Mid\n(trans)", "Closed\n(grasp)"],
            "gripper_2": ["Open\n(reach)", "Closed\n(grasp)"],
        }.get(ltype, [f"Class {i}" for i in range(n_cls)])
        colors = plt.cm.Set2(np.linspace(0, 1, n_cls))
        ax.bar(range(n_cls), counts, color=colors, alpha=0.8)
        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_ylabel("Number of policy calls")
        ax.set_title(f"Label: {ltype}", fontsize=9)
        for i, c in enumerate(counts):
            ax.text(i, c + 1, str(c), ha="center", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = out_dir / "probe_label_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # ── (5) Accuracy improvement over chance (delta) ─────────────────────
    fig, axes = plt.subplots(1, len(label_types), figsize=(6 * len(label_types), 5))
    if len(label_types) == 1:
        axes = [axes]
    fig.suptitle(
        f"Theme 2 Linear Probe: Accuracy Gain Over Chance (Δ%)\n"
        f"Task: {task_name}  (k=4)",
        fontsize=11,
    )
    for ax, ltype in zip(axes, label_types):
        chance = results[ltype]["meta"]["chance_level"]
        deltas = [results[ltype][4][l]["mean_acc"] - chance for l in PROBE_LAYERS]
        colors = ["tomato" if d > 0 else "royalblue" for d in deltas]
        ax.bar(range(n_layers), [d * 100 for d in deltas], color=colors, alpha=0.8)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS], fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Δ Accuracy over chance (%)")
        ax.set_title(f"Label: {ltype}", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        for i, d in enumerate(deltas):
            ax.text(i, d * 100 + (1 if d >= 0 else -3), f"{d*100:+.1f}%",
                    ha="center", fontsize=7)
    plt.tight_layout()
    p = out_dir / "probe_delta_over_chance.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_npz", required=True)
    parser.add_argument("--actions_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_name", default="PnPCounterToCab")
    parser.add_argument("--success_rate", type=float, default=0.70)
    parser.add_argument("--lambda_reg", type=float, default=1.0,
                        help="Ridge regression regularization coefficient")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.feat_npz}")
    feats, ep_arr, ci_arr = load_features(args.feat_npz)
    N = len(ep_arr)
    print(f"N={N}, episodes={np.unique(ep_arr)}")

    print(f"Loading actions from {args.actions_npz}")
    actions = load_actions(args.actions_npz)
    print(f"Action keys: {list(actions.keys())}, shape: {actions[0].shape}")

    # ── Generate skill labels ──────────────────────────────────────────────
    label_configs = {
        "progress_3": {
            "labels": labels_from_progress(ep_arr, ci_arr, n_classes=3),
            "n_classes": 3,
            "class_names": ["Early(reach)", "Mid(grasp)", "Late(place)"],
        },
        "gripper_3": {
            "labels": labels_from_gripper(actions, k=4, n_classes=3),
            "n_classes": 3,
            "class_names": ["Open(reach)", "Mid(trans)", "Closed(grasp)"],
        },
        "gripper_2": {
            "labels": labels_from_gripper(actions, k=4, n_classes=2),
            "n_classes": 2,
            "class_names": ["Open(reach)", "Closed(grasp)"],
        },
    }

    for ltype, lcfg in label_configs.items():
        lbls = lcfg["labels"]
        counts = np.bincount(lbls.astype(int), minlength=lcfg["n_classes"])
        print(f"\nLabel '{ltype}': {lcfg['n_classes']} classes, "
              f"distribution: {counts} (total {N})")

    # ── Run linear probing ────────────────────────────────────────────────
    results = {}
    for ltype, lcfg in label_configs.items():
        labels = lcfg["labels"]
        n_cls = lcfg["n_classes"]
        chance = chance_level(n_cls, labels)
        print(f"\n=== Label: {ltype} (chance={chance:.1%}) ===")

        results[ltype] = {
            "meta": {
                "n_classes": n_cls,
                "chance_level": chance,
                "all_labels": labels,
                "class_names": lcfg["class_names"],
            }
        }

        for k in range(NUM_DENOISE_STEPS):
            results[ltype][k] = {}
            print(f"  k={k}: ", end="", flush=True)
            for l in PROBE_LAYERS:
                F = feats[k][l]
                if F is None or len(F) != N:
                    results[ltype][k][l] = {"mean_acc": 0.0, "std_acc": 0.0, "per_fold": []}
                    continue

                mean_acc, std_acc, per_fold = loeo_cross_validate(
                    F, labels, ep_arr, n_cls,
                    lambda_reg=args.lambda_reg, n_pca=50,
                )
                results[ltype][k][l] = {
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "per_fold": per_fold,
                }
                print(f"Blk-{l}:{mean_acc:.2f}±{std_acc:.2f} ", end="", flush=True)
            print()

    # ── Serialize results (convert to JSON-safe types) ──────────────────
    def to_json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items() if k != "all_labels"}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    stats = {
        "task": args.task_name,
        "success_rate": args.success_rate,
        "N_calls": int(N),
        "probe_layers": PROBE_LAYERS,
        "lambda_reg": args.lambda_reg,
        "results": {
            ltype: {
                "meta": {
                    "n_classes": v["meta"]["n_classes"],
                    "chance_level": v["meta"]["chance_level"],
                    "class_names": v["meta"]["class_names"],
                },
                **{
                    str(k): {
                        str(l): {kk: vv for kk, vv in v[k][l].items() if kk != "per_fold"}
                        for l in PROBE_LAYERS
                    }
                    for k in range(NUM_DENOISE_STEPS)
                }
            }
            for ltype, v in results.items()
        },
    }

    stats_path = out_dir / "theme2_probe_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_probe_results(results, out_dir, args.task_name, args.success_rate)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n=== Linear Probe Accuracy Summary (k=4) ===")
    print(f"{'Layer':<10}", end="")
    for ltype in label_configs:
        print(f"  {ltype:<14}", end="")
    print()

    for l in PROBE_LAYERS:
        print(f"{PROBE_LAYER_SHORT.get(l, f'B{l}'):<10}", end="")
        for ltype in label_configs:
            r = results[ltype][4][l]
            print(f"  {r['mean_acc']:.2f}±{r['std_acc']:.2f}      ", end="")
        print()

    chance_str = "  ".join([f"{results[lt]['meta']['chance_level']:.2f}           " for lt in label_configs])
    print(f"{'Chance':<10}  {chance_str}")
    print(f"\nOutput: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
