"""
intrinsic_dimension_test.py — attractor_verification_report.md への追加検証 (課題B-1)。

ユーザー提案「連続多様体仮説の幾何学的裏付け: 本質的次元数(Intrinsic Dimension)の定量化」
への対応。§4.1.2 (manifold_trajectory_test.py) は「進行度に沿った滑らかな軌跡」を示したが、
その軌跡が埋め込まれている多様体の次元数そのものは未測定だった。本スクリプトは
シーン残差化後の表現 Z (ambient D=2048) が実質何次元の多様体に拘束されているかを、
3つの独立な内在次元推定法で測定する:

  1. Two-NN (Facco et al. 2017): 各点の最近傍2点までの距離比 r2/r1 の分布から、
     局所一様密度を仮定した最尤推定。ノイズに強く次元選択が不要。
  2. Levina-Bickel MLE (2004): k近傍距離の対数比からの最尤推定。k=5..20の平均を採用
     (単一kの分散低減、原論文の推奨)。
  3. 相関次元 (Grassberger-Procaccia): 対数-対数プロット log C(r) vs log r の
     スケーリング領域の傾き。診断用log-logプロットも保存する。
  4. (参考) PCA participation ratio と累積寄与率90%/95%到達成分数
     — 上記3手法より粗いが解釈しやすい「実効次元」の目安。

方法論的注意 (厳格レビュー対策):
  - 単一の推定法だけでは「小さい数値が出た」ことの意味は保証できない
    (推定法自体のバイアス・有限サンプルの影響で常に小さめに出る可能性があるため)。
    そこで各task/seedについて **列シャッフルnull** (各特徴次元を独立にシャッフルし、
    周辺分布は保ったまま次元間の共分散構造だけ破壊したデータ) にも同じ4手法を適用し、
    真データの推定値がnullを大きく下回ることをもって「低次元構造が実在する」ことの
    証拠とする。nullでの推定値がambient Dに近ければパイプライン自体は妥当に機能している
    ことの確認にもなる。
  - シーン残差化 (scene_residualize) 後の特徴を使う。生特徴でのPCAはシーン差が
    支配的になるため (skill_count.py Level1と同じ統制)。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import (
    LAYER, K_STEP, load_task_seed_data, scene_residualize,
)

MLE_K_MIN, MLE_K_MAX = 5, 20
TWONN_DISCARD_FRACTION = 0.1
CORRDIM_QUANTILE_RANGE = (0.05, 0.4)
CORRDIM_N_R = 25


def twonn_dim(X, discard_fraction=TWONN_DISCARD_FRACTION):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nbrs.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    valid = r1 > 1e-12
    mu = np.sort(r2[valid] / r1[valid])
    n_valid = len(mu)
    keep = max(int(np.floor(n_valid * (1 - discard_fraction))), 2)
    mu_keep = mu[:keep]
    F = np.arange(1, keep + 1) / n_valid
    x = np.log(mu_keep)
    y = -np.log(1 - F)
    d_hat = float(np.sum(x * y) / np.sum(x * x))
    return d_hat, n_valid


def mle_dim(X, k_min=MLE_K_MIN, k_max=MLE_K_MAX):
    n = X.shape[0]
    k_max = min(k_max, n - 2)
    if k_max < k_min:
        return float("nan"), []
    nbrs = NearestNeighbors(n_neighbors=k_max + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    dist = dist[:, 1:]
    m_hat_per_k = []
    for k in range(k_min, k_max + 1):
        Tk = dist[:, k - 1]
        Tj = dist[:, : k - 1]
        logs = np.log(np.clip(Tk[:, None], 1e-12, None) / np.clip(Tj, 1e-12, None))
        m_k_i = 1.0 / np.clip(np.mean(logs, axis=1), 1e-12, None)
        m_hat_k = 1.0 / np.mean(1.0 / np.clip(m_k_i, 1e-6, None))
        m_hat_per_k.append(float(m_hat_k))
    return float(np.mean(m_hat_per_k)), m_hat_per_k


def correlation_dim(X, quantile_range=CORRDIM_QUANTILE_RANGE, n_r=CORRDIM_N_R):
    d = pdist(X)
    d = d[d > 1e-12]
    r_lo, r_hi = np.quantile(d, quantile_range)
    rs = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), n_r))
    Cs = np.array([np.mean(d < r) for r in rs])
    valid = Cs > 1e-8
    x, y = np.log(rs[valid]), np.log(Cs[valid])
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_fit = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return float(slope), r2_fit, rs, Cs, x, y, y_pred


def pca_participation_ratio(X):
    n_comp = min(X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp).fit(X)
    lam = pca.explained_variance_
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    return pr, n90, n95


def column_shuffle_null(X, seed=0):
    rng = np.random.RandomState(seed)
    Xn = X.copy()
    for j in range(Xn.shape[1]):
        rng.shuffle(Xn[:, j])
    return Xn


def run_all_estimators(X, seed=0):
    twonn, twonn_n = twonn_dim(X)
    mle, mle_curve = mle_dim(X)
    corrdim, corrdim_r2, rs, Cs, x, y, y_pred = correlation_dim(X)
    pr, n90, n95 = pca_participation_ratio(X)
    return {
        "twonn_dim": twonn,
        "twonn_n_valid_pairs": twonn_n,
        "mle_dim": mle,
        "mle_dim_curve_k5to20": mle_curve,
        "correlation_dim": corrdim,
        "correlation_dim_loglog_fit_r2": corrdim_r2,
        "pca_participation_ratio": pr,
        "pca_n_components_for_90pct_var": n90,
        "pca_n_components_for_95pct_var": n95,
        "ambient_dim": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
    }, (rs, Cs, x, y, y_pred)


def plot_corrdim(real_curve, null_curve, task, seed_series, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (rs, Cs, x, y, y_pred), color in [
        ("real (scene-residualized Z)", real_curve, "C0"),
        ("column-shuffle null", null_curve, "C3"),
    ]:
        ax.plot(x, y, "o", color=color, markersize=3, alpha=0.6, label=f"{label} (data)")
        ax.plot(x, y_pred, "-", color=color, linewidth=1.5, alpha=0.9)
    ax.set_xlabel("log r")
    ax.set_ylabel("log C(r)")
    ax.set_title(f"{task} seed={seed_series}: correlation-integral scaling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / f"corrdim_loglog_{task}_seed{seed_series}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path.name


def analyze_task_seed(collect_dir, task, seed_series, fname, out_dir, seed=0):
    d = load_task_seed_data(collect_dir, fname, LAYER, K_STEP)
    X_raw, episode = d["feats"], d["episode"]

    Xr = scene_residualize(X_raw, episode)
    Xs = StandardScaler().fit_transform(Xr)

    real_result, real_curve = run_all_estimators(Xs, seed=seed)

    Xnull = column_shuffle_null(Xs, seed=seed + 3000)
    null_result, null_curve = run_all_estimators(Xnull, seed=seed)

    png_name = plot_corrdim(real_curve, null_curve, task, seed_series, out_dir)

    log_message(
        f"[intrinsic-dim {task} seed={seed_series}] "
        f"TwoNN={real_result['twonn_dim']:.2f} (null={null_result['twonn_dim']:.2f}) | "
        f"MLE={real_result['mle_dim']:.2f} (null={null_result['mle_dim']:.2f}) | "
        f"CorrDim={real_result['correlation_dim']:.2f} (null={null_result['correlation_dim']:.2f}, "
        f"R2fit={real_result['correlation_dim_loglog_fit_r2']:.3f}) | "
        f"PCA-PR={real_result['pca_participation_ratio']:.2f} (null={null_result['pca_participation_ratio']:.2f}) | "
        f"n90%={real_result['pca_n_components_for_90pct_var']} (null={null_result['pca_n_components_for_90pct_var']}) "
        f"N={real_result['n_samples']} D={real_result['ambient_dim']}"
    )

    return {
        "real": real_result,
        "column_shuffle_null": null_result,
        "corrdim_loglog_plot_png": png_name,
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
            "Two-NN(Facco2017)/Levina-Bickel MLE(k=5-20平均)/相関次元(Grassberger-Procaccia)/"
            "PCA participation ratioの4手法。各task/seedについて列シャッフルnull"
            "(周辺分布保持・次元間共分散破壊)との対比で低次元構造の実在を検証。"
        ),
        "tasks": {},
    }
    for task, fnames_by_seed in files_by_task.items():
        results["tasks"][task] = {}
        for seed_series, fname in fnames_by_seed.items():
            results["tasks"][task][str(seed_series)] = analyze_task_seed(
                collect_dir, task, seed_series, fname, out_dir, seed=0
            )

    with open(out_dir / "intrinsic_dimension_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'intrinsic_dimension_test.json'}")


if __name__ == "__main__":
    main()
