"""
DiT 層別線形プロービング解析スクリプト

【目的】
  各 DiT 層の中間特徴量が「スキルフェーズ（リーチング/グラスピング/プレーシング）」を
  どの程度線形分離可能な形で保持しているかを層ごとに定量化する。

【手法】
  - スキルラベル
      A) 進行度ベース: エピソード内の call 進行を 3 等分（quantile 分割）
      B) グリッパー 2 値: 中央値で閾値分類（balanced）
      C) グリッパー 3 クラス: quantile 分割（1/3, 2/3）
  - 特徴量の前処理: Global PCA で 50 次元に削減（詳細は下記）
  - 分類器: クラス重み付き Ridge 回帰（numpy のみ、解析解）
  - 評価: Leave-One-Episode-Out (LOEO) 交差検証

【Global PCA について】
  PCA は LOEO 全フォールドに共通の（全データで計算した）固有ベクトルを使用する。
  フォールドごとに PCA を計算すると、各フォールドで固有空間が異なり、
  「分類器の精度」と「PCA の変動」が混在した評価になってしまう。
  Global PCA により全フォールドが同じ特徴空間上で比較可能になる。
  ただし、テストデータの情報が PCA 計算に含まれる（data leakage）ため、
  精度の絶対値よりも層間・ステップ間の相対比較に意味がある。

【分類器の学習単位について】
  各サンプルは「1 policy call」= 1 回の推論（32 step チャンク）に対応する。
  入力特徴量は action token embedding（2048 次元、各 policy call で 1 ベクトル）。
  エピソード内のタイムステップ（0〜499 step）方向に分類器を分けてはいない。
  スキルフェーズラベルは policy call 全体の「エピソード進行度」を表すラベルであり、
  各 policy call が「早期/中期/後期」のどのフェーズに属するかを示す。

実行:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.linear_probe \\
      --feat_npz cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz \\
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


# ── スキルラベル生成（クラスバランス保証） ────────────────────────────────

def labels_from_progress(ep_arr: np.ndarray, ci_arr: np.ndarray,
                          n_classes: int = 3) -> np.ndarray:
    """
    エピソード内の正規化進行度から n_classes クラスのラベルを生成。
    quantile 分割でクラスバランスを保証する。
    """
    N = len(ep_arr)
    progress = np.zeros(N)
    for ep in np.unique(ep_arr):
        mask = ep_arr == ep
        max_ci = ci_arr[mask].max()
        progress[mask] = ci_arr[mask] / max(max_ci, 1)

    quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]  # e.g. [1/3, 2/3]
    thresholds = np.quantile(progress, quantiles)
    labels = np.zeros(N, dtype=int)
    for i, thr in enumerate(thresholds):
        labels[progress > thr] = i + 1
    return labels, progress


def labels_from_gripper(step_actions: Dict, k: int = 4,
                         n_classes: int = 3) -> np.ndarray:
    """
    k=4 (最終デノイジング) での予測グリッパー値（次元 6）の
    32-step チャンク平均から n_classes クラスのラベルを生成。
    2 クラスの場合は中央値で分割してバランスを保証する。
    """
    gripper = step_actions[k][:, :, 6].mean(axis=1)  # (N,)
    if n_classes == 2:
        threshold = float(np.median(gripper))
        return (gripper > threshold).astype(int)
    else:
        q33, q67 = np.quantile(gripper, [1/3, 2/3])
        labels = np.zeros(len(gripper), dtype=int)
        labels[gripper > q33] = 1
        labels[gripper > q67] = 2
        return labels


# ── 線形プローブ (クラス重み付き Ridge Regression) ────────────────────────

def pca_project(X_all: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    全データで PCA を計算し n_components 次元に射影する（Global PCA）。
    2048 次元 → 50 次元 にして ridge の行列を小さくする。
    全データで計算することで LOEO 全フォールド共通の固有空間を使用する。
    """
    X_c = X_all - X_all.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components]          # (n_components, D)
    return X_c @ components.T              # (N, n_components)


def ridge_probe(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                n_classes: int, lambda_reg: float = 1.0) -> Tuple[float, np.ndarray]:
    """
    クラス重み付き Ridge regression でマルチクラス線形分類。
    各クラスの重みを 1/count に設定してクラス不均衡を補正する。
    Returns: (accuracy, predictions for test set)
    """
    N_tr, D = X_train.shape
    mu = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8
    X_tr_n = (X_train - mu) / sig
    X_te_n = (X_test - mu) / sig

    X_tr_b = np.hstack([X_tr_n, np.ones((N_tr, 1))])
    X_te_b = np.hstack([X_te_n, np.ones((X_test.shape[0], 1))])

    # クラス重み (1/count で各クラスを等重視)
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    sample_weights = 1.0 / counts[y_train.astype(int)]
    sample_weights = sample_weights / sample_weights.mean()  # 正規化
    W_diag = np.diag(sample_weights)

    Y_oh = np.zeros((N_tr, n_classes))
    Y_oh[np.arange(N_tr), y_train.astype(int)] = 1.0

    A = X_tr_b.T @ W_diag @ X_tr_b + lambda_reg * np.eye(D + 1)
    b = X_tr_b.T @ W_diag @ Y_oh
    try:
        W = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, b, rcond=None)[0]

    logits = X_te_b @ W
    y_pred = logits.argmax(axis=1)
    acc = float((y_pred == y_test.astype(int)).mean())
    return acc, y_pred


def loeo_cross_validate(features: np.ndarray, labels: np.ndarray,
                        ep_arr: np.ndarray, n_classes: int,
                        lambda_reg: float = 1.0,
                        n_pca: int = 50) -> Tuple[float, float, List[float], np.ndarray]:
    """
    Leave-One-Episode-Out (LOEO) 交差検証。
    Global PCA で n_pca 次元に削減してから使用する。
    Returns: (mean_acc, std_acc, per_fold_accs, all_predictions)
    all_predictions[i] は i 番目のサンプルに対する予測クラス (-1 = 評価未実施)
    """
    features_pca = pca_project(
        features, n_components=min(n_pca, features.shape[0] - 1, features.shape[1])
    )

    episodes = np.unique(ep_arr)
    accs = []
    all_preds = np.full(len(labels), -1, dtype=int)

    for test_ep in episodes:
        test_mask = ep_arr == test_ep
        train_mask = ~test_mask

        if train_mask.sum() < n_classes or test_mask.sum() < 1:
            continue
        if len(np.unique(labels[train_mask])) < n_classes:
            continue

        acc, preds = ridge_probe(
            features_pca[train_mask], labels[train_mask],
            features_pca[test_mask], labels[test_mask],
            n_classes=n_classes,
            lambda_reg=lambda_reg,
        )
        accs.append(acc)
        all_preds[test_mask] = preds

    if not accs:
        return 0.0, 0.0, [], all_preds
    return float(np.mean(accs)), float(np.std(accs)), accs, all_preds


def chance_level(n_classes: int, labels: np.ndarray) -> float:
    """Majority class baseline."""
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    return float(counts.max() / counts.sum())


# ── プロット ─────────────────────────────────────────────────────────────

def plot_probe_results(results: Dict, out_dir: Path, task_name: str,
                       success_rate: float):
    n_layers = len(PROBE_LAYERS)
    layer_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_layers))
    k_markers = ["o", "s", "^", "D", "v"]
    k_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, NUM_DENOISE_STEPS))
    label_types = list(results.keys())

    # ── (1) Accuracy by layer ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(label_types), figsize=(6 * len(label_types), 5))
    if len(label_types) == 1:
        axes = [axes]
    fig.suptitle(
        f"Linear Probe: Accuracy by Layer\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  (LOEO, class-weighted)",
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

    # ── (2) k=0 vs k=4 comparison ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(label_types), figsize=(5 * len(label_types), 5))
    if len(label_types) == 1:
        axes = [axes]
    fig.suptitle(
        f"Linear Probe: k=0 vs k=4 Comparison\nTask: {task_name}  (LOEO)",
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
        ax.bar(x - width/2, means_k0, width, yerr=stds_k0,
               label="k=0 (σ=80)", color="royalblue", alpha=0.75, capsize=4)
        ax.bar(x + width/2, means_k4, width, yerr=stds_k4,
               label="k=4 (σ=4)", color="tomato", alpha=0.75, capsize=4)
        ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label="Chance")
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

    # ── (3) Heatmap: accuracy (layer × step) per label_type ───────────────
    k_labels_short = [f"k={k}" for k in range(NUM_DENOISE_STEPS)]
    layer_labels_short = [PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS]
    for ltype in label_types:
        fig, ax = plt.subplots(figsize=(NUM_DENOISE_STEPS * 1.6 + 1, n_layers * 1.2 + 1))
        hmap = np.zeros((n_layers, NUM_DENOISE_STEPS))
        for ni, l in enumerate(PROBE_LAYERS):
            for k in range(NUM_DENOISE_STEPS):
                if k in results[ltype] and l in results[ltype][k]:
                    hmap[ni, k] = results[ltype][k][l]["mean_acc"]
        im = ax.imshow(hmap, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
        ax.set_xticks(range(NUM_DENOISE_STEPS))
        ax.set_xticklabels(k_labels_short, fontsize=9)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_labels_short, fontsize=9)
        ax.set_xlabel("Denoising step k", fontsize=9)
        ax.set_ylabel("Probe layer", fontsize=9)
        ax.set_title(
            f"Linear Probe Accuracy (layer × step) — {ltype}\nTask: {task_name}", fontsize=10
        )
        plt.colorbar(im, ax=ax, label="Accuracy")
        for i in range(n_layers):
            for j in range(NUM_DENOISE_STEPS):
                ax.text(j, i, f"{hmap[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if hmap[i,j] < 0.6 else "black")
        plt.tight_layout()
        p = out_dir / f"probe_accuracy_heatmap_{ltype}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")

    # ── (4) Label distribution visualization ─────────────────────────────
    n_label_types = len(label_types)
    fig, axes = plt.subplots(1, n_label_types, figsize=(5 * n_label_types, 4))
    if n_label_types == 1:
        axes = [axes]
    fig.suptitle("Skill Label Distributions (balanced)", fontsize=11)
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

    # ── (5) Accuracy gain over chance ────────────────────────────────────
    fig, axes = plt.subplots(1, len(label_types), figsize=(6 * len(label_types), 5))
    if len(label_types) == 1:
        axes = [axes]
    fig.suptitle(
        f"Linear Probe: Accuracy Gain Over Chance (Δ%)\nTask: {task_name}  (k=4)",
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


def plot_trajectory_comparison(results: Dict, ep_arr: np.ndarray, ci_arr: np.ndarray,
                                out_dir: Path, task_name: str,
                                n_show_episodes: int = 5,
                                probe_layer: int = 27, k_step: int = 4):
    """
    数エピソード分の正解ラベルと予測ラベルの time-series 比較プロット。
    x 軸 = エピソード内の policy call index（時系列順）
    y 軸 = クラスラベル（0, 1, 2 など）
    各エピソードで正解（solid）vs 予測（dashed）を重ね描きする。

    probe_layer と k_step: 最も精度が高い組み合わせを想定（デフォルト Block-27, k=4）。
    """
    label_types = list(results.keys())
    n_label_types = len(label_types)

    episodes = np.unique(ep_arr)
    show_eps = episodes[:n_show_episodes]

    fig, axes = plt.subplots(n_label_types, len(show_eps),
                             figsize=(4 * len(show_eps), 3 * n_label_types),
                             sharey="row")
    if n_label_types == 1:
        axes = axes.reshape(1, -1)
    if len(show_eps) == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f"Predicted vs Ground-Truth Label Trajectory\n"
        f"Probe: Block-{probe_layer}, k={k_step}  |  Task: {task_name}\n"
        f"Solid = Ground Truth, Dashed = Predicted",
        fontsize=11,
    )

    for row, ltype in enumerate(label_types):
        meta = results[ltype]["meta"]
        n_cls = meta["n_classes"]
        all_labels = meta["all_labels"]
        # per_fold predictions are stored per (ltype, k_step, layer)
        all_preds = results[ltype][k_step][probe_layer].get("all_preds", None)
        class_names_short = {
            "progress_3": ["Early", "Mid", "Late"],
            "gripper_3": ["Open", "Mid", "Closed"],
            "gripper_2": ["Open", "Closed"],
        }.get(ltype, [f"C{i}" for i in range(n_cls)])
        colors = plt.cm.Set1(np.linspace(0, 0.6, n_cls))

        for col, ep in enumerate(show_eps):
            ax = axes[row, col]
            ep_mask = ep_arr == ep
            ci_ep = ci_arr[ep_mask]
            sort_order = np.argsort(ci_ep)
            ci_sorted = ci_ep[sort_order]
            true_sorted = all_labels[ep_mask][sort_order]

            # Ground truth
            ax.step(ci_sorted, true_sorted, where="post",
                    linewidth=2, color="steelblue", label="GT")
            ax.fill_between(ci_sorted, true_sorted, step="post", alpha=0.15, color="steelblue")

            if all_preds is not None:
                pred_ep = all_preds[ep_mask][sort_order]
                valid = pred_ep >= 0
                if valid.any():
                    ax.step(ci_sorted[valid], pred_ep[valid], where="post",
                            linewidth=1.5, color="tomato", linestyle="--", label="Pred")

            ax.set_yticks(range(n_cls))
            ax.set_yticklabels(class_names_short, fontsize=7)
            ax.set_ylim(-0.3, n_cls - 0.7)
            ax.set_xlabel("Policy call index", fontsize=7)
            ax.set_title(f"Ep {ep}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(ltype, fontsize=8)
                ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    p = out_dir / f"probe_trajectory_comparison_blk{probe_layer}_k{k_step}.png"
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
    parser.add_argument("--lambda_reg", type=float, default=1.0)
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

    # ── Generate balanced skill labels ────────────────────────────────────
    progress_labels, progress_values = labels_from_progress(ep_arr, ci_arr, n_classes=3)
    label_configs = {
        "progress_3": {
            "labels": progress_labels,
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
        ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
        print(f"\nLabel '{ltype}': {lcfg['n_classes']} classes, "
              f"distribution: {counts} (ratio={ratio:.2f}, total={N})")

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
                    results[ltype][k][l] = {
                        "mean_acc": 0.0, "std_acc": 0.0, "per_fold": [], "all_preds": None
                    }
                    continue

                mean_acc, std_acc, per_fold, all_preds = loeo_cross_validate(
                    F, labels, ep_arr, n_cls,
                    lambda_reg=args.lambda_reg, n_pca=50,
                )
                results[ltype][k][l] = {
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "per_fold": per_fold,
                    "all_preds": all_preds,
                }
                print(f"Blk-{l}:{mean_acc:.2f}±{std_acc:.2f} ", end="", flush=True)
            print()

    # ── Save stats JSON ───────────────────────────────────────────────────
    def to_json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()
                    if k not in ("all_labels", "all_preds")}
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
        "label_method": {
            "progress_3": "quantile-based (balanced)",
            "gripper_2": "median threshold (balanced)",
            "gripper_3": "quantile-based (balanced)",
        },
        "results": {
            ltype: {
                "meta": {
                    "n_classes": v["meta"]["n_classes"],
                    "chance_level": v["meta"]["chance_level"],
                    "class_names": v["meta"]["class_names"],
                },
                **{
                    str(k): {
                        str(l): {kk: vv for kk, vv in v[k][l].items()
                                 if kk not in ("per_fold", "all_preds")}
                        for l in PROBE_LAYERS
                    }
                    for k in range(NUM_DENOISE_STEPS)
                }
            }
            for ltype, v in results.items()
        },
    }

    stats_path = out_dir / "probe_stats.json"
    with open(stats_path, "w") as f:
        json.dump(to_json_safe(stats), f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_probe_results(results, out_dir, args.task_name, args.success_rate)

    # Trajectory comparison (Block-27, k=4 が最も精度が高い)
    plot_trajectory_comparison(
        results, ep_arr, ci_arr, out_dir, args.task_name,
        n_show_episodes=5, probe_layer=27, k_step=4,
    )

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
    chance_str = "  ".join(
        [f"{results[lt]['meta']['chance_level']:.2f}           " for lt in label_configs]
    )
    print(f"{'Chance':<10}  {chance_str}")
    print(f"\nOutput: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
