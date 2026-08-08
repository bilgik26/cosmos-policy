"""
v4改訂（2026-08-06、3seed成功epのみデータへの統一 + §8厳格化）で使う
統計処理の共通ライブラリ。verification_report_v3.md の既存節（§2, §3, §5）が
確立した手法をそのまま踏襲し、§8（画像ラテント）にも同水準で適用できるように
汎用化したもの。

主な機能:
  - episode単位bootstrap CI（call単位ではなくepisode単位でリサンプル）
  - episode単位符号検定
  - B_B置換検定（FFTスペクトル重心のk依存性）
  - 線形ガウス最適デノイザ ヌルモデル（Tweedie公式）
  - Participation Ratio（有効ランク）
  - Linear CKA
  - ステップ間コサイン類似度
  - Cohen's d（paired, episode単位）
  - Benjamini-Hochberg FDR補正
  - fold内標準化PCA + LogisticRegression + LOEO + permutationテスト（プロービング）
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut


# ── 基本ユーティリティ ──────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    N = X.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    K = H @ (X @ X.T) @ H
    L = H @ (Y @ Y.T) @ H
    hsic_xy = np.trace(K @ L)
    denom = np.linalg.norm(K, "fro") * np.linalg.norm(L, "fro")
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0


def participation_ratio(X: np.ndarray) -> float:
    """PR = (Σσᵢ²)² / Σσᵢ⁴ 。Xは (N, D)。Gram行列の固有値からSVD相当を求める
    （N << D のときGram行列を使う方が高速、本プロジェクトのN規模なら
    np.linalg.svd でも十分高速）。"""
    Xc = X - X.mean(axis=0)
    if Xc.shape[0] <= Xc.shape[1]:
        G = Xc @ Xc.T
        eigs = np.linalg.eigvalsh(G)
        sv2 = np.maximum(eigs, 0.0)
    else:
        S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
        sv2 = S ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1.0
    return float(total ** 2 / (sv2 ** 2).sum())


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else 0.0


def bh_fdr(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR補正。有意判定のbool配列を返す（pvalsと同じ順序）。"""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k = np.max(np.where(below)[0])
    sig = np.zeros(n, dtype=bool)
    sig[order[: max_k + 1]] = True
    return sig


# ── episode単位の集計・検定 ──────────────────────────────────────────────────

def per_episode_mean(values: np.ndarray, episode_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """call単位の値をepisode単位の平均に集約する。Returns (unique_eps, means)。"""
    eps = np.unique(episode_ids)
    means = np.array([values[episode_ids == e].mean() for e in eps])
    return eps, means


def episode_bootstrap_ci(
    values: np.ndarray, episode_ids: np.ndarray, n_boot: int = 1000,
    alpha: float = 0.05, seed: int = 0, agg: Callable = np.mean,
) -> Dict[str, float]:
    """episode単位のブートストラップ（各episodeをリサンプル単位とする）で
    call値全体の集約統計量の95%CIを求める。"""
    rng = np.random.RandomState(seed)
    eps = np.unique(episode_ids)
    n_eps = len(eps)
    point = float(agg(values))
    boot_stats = []
    ep_to_idx = {e: np.where(episode_ids == e)[0] for e in eps}
    for _ in range(n_boot):
        sampled_eps = rng.choice(eps, size=n_eps, replace=True)
        idxs = np.concatenate([ep_to_idx[e] for e in sampled_eps])
        boot_stats.append(agg(values[idxs]))
    boot_stats = np.array(boot_stats)
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_episodes": n_eps}


def episode_sign_test(
    values_a: np.ndarray, values_b: np.ndarray, episode_ids: np.ndarray,
) -> Dict[str, float]:
    """各episode内でvalues_bがvalues_aを上回るか(2値化)、二項検定(p=0.5)。
    values_a, values_bは同じcall順に対応する同じ長さの配列。"""
    from scipy.stats import binomtest
    eps = np.unique(episode_ids)
    n_gt = 0
    n_total = 0
    for e in eps:
        mask = episode_ids == e
        ma, mb = values_a[mask].mean(), values_b[mask].mean()
        if ma == mb:
            continue
        n_total += 1
        if mb > ma:
            n_gt += 1
    if n_total == 0:
        return {"n_episodes": 0, "n_greater": 0, "p_value": 1.0}
    res = binomtest(n_gt, n_total, 0.5)
    return {"n_episodes": n_total, "n_greater": n_gt, "p_value": float(res.pvalue)}


def bb_permutation_test(
    per_call_step_values: np.ndarray, call_ids: np.ndarray, n_perm: int = 1000, seed: int = 0,
) -> Dict[str, float]:
    """B_B置換検定: per_call_step_values は (N_calls, N_steps) 、各callにおいて
    ステップラベル(列)をシャッフルしたときの (max-min) 分布を帰無分布とし、
    実測の (max-min of per-step mean) がどの位置にあるかでp値を求める。"""
    rng = np.random.RandomState(seed)
    N, K = per_call_step_values.shape
    observed_range = per_call_step_values.mean(axis=0).max() - per_call_step_values.mean(axis=0).min()
    null_ranges = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = np.array([rng.permutation(row) for row in per_call_step_values])
        m = shuffled.mean(axis=0)
        null_ranges[i] = m.max() - m.min()
    p = float((null_ranges >= observed_range).mean())
    return {"observed_range": float(observed_range), "p_value": p, "n_perm": n_perm}


# ── 線形ガウス最適デノイザ ヌルモデル（Tweedie公式） ─────────────────────────

def gaussian_null_delta(sigma_schedule: Sequence[float], sigma_data: float, D: int,
                         n_mc: Optional[int] = None, seed: int = 0,
                         max_bytes: int = 1_500_000_000) -> np.ndarray:
    """
    x̂₀(x_t,σ) = c_σ * x_t, c_σ = sigma_data^2/(sigma_data^2+σ^2) の下での
    E[||Δx̂₀||] をモンテカルロで計算する。

    重要: 単一の生成トラジェクトリ内では x_σ = x_target + σ・n という関係を
    「同一のノイズ方向 n」が全σレベルで共有される（EDMの確率流ODEの、
    ガウス分布データに対する厳密解に対応）。σごとに独立な n を引くのは誤り
    （実測値の約2倍になり、report §2.1 の公表ヌル値
    [0.0325, 0.0708, 0.1660, 0.4246](raw, sigma_data=0.443, D=224)を再現できない
    ことをMCで確認済み。n を全σで共有すると正しく再現される）。

    n_mc未指定時はmax_bytes(既定1.5GB)に収まるよう D から自動決定する
    （行動D=224なら数十万サンプル可能だが、画像ラテントD=12544では
    同じn_mcだと必要メモリが56倍になりOOMする。実際にD=12544×n_mc=200000×
    float64で développementプロセスRSSが数十秒で30GB超に達し強制終了する
    事故が発生したため、float32化+自動スケーリングで対応した）。
    x0_hatは5ステップ分を同時に保持せず、隣接ステップのみ保持して逐次計算し
    ピークメモリをさらに抑える。

    Returns: (len(sigma_schedule)-1,) 各遷移のヌル期待値（raw、schednorm前）。
    """
    if n_mc is None:
        # 必要な同時生存バッファ: x_target, n, x0_prev, x0_cur (4本) x 4bytes(float32)
        n_mc = max(2000, min(200000, max_bytes // (4 * D * 4)))
    rng = np.random.RandomState(seed)
    x_target = (rng.randn(n_mc, D) * sigma_data).astype(np.float32)
    n = rng.randn(n_mc, D).astype(np.float32)  # 全σレベルで共有する単一のノイズ方向
    sigmas = np.asarray(sigma_schedule)
    c = sigma_data ** 2 / (sigma_data ** 2 + sigmas ** 2)

    deltas = []
    x0_prev = (c[0] * (x_target + sigmas[0] * n)).astype(np.float32)
    for i in range(1, len(sigmas)):
        x0_cur = (c[i] * (x_target + sigmas[i] * n)).astype(np.float32)
        d = np.linalg.norm(x0_cur - x0_prev, axis=1)
        deltas.append(float(d.mean()))
        x0_prev = x0_cur
    return np.array(deltas)


def estimate_sigma_data(x0_final: np.ndarray) -> float:
    """最終ステップのx̂₀ population標準偏差(次元平均)からsigma_dataを推定。"""
    return float(x0_final.std(axis=0).mean())


# ── プロービング（fold内標準化PCA + LogisticRegression + LOEO） ─────────────

def _fold_pca_projections(X: np.ndarray, groups: np.ndarray, n_components: int) -> List[Tuple]:
    """
    LOEOの各foldについて、train側で fold内標準化PCA(§4.1で確定した正しい
    前処理: 中心化でなく標準化してからPCA)を計算し、(train_idx, test_idx,
    X_train_p, X_test_p) のリストを返す。PCAはラベルyに依存しないため、
    permutationテストで観測値・全null反復を通じて使い回せる（これをしないと
    n_perm回分のSVDを毎回計算することになり計算コストが約100倍になる）。
    """
    logo = LeaveOneGroupOut()
    folds = []
    for train_idx, test_idx in logo.split(X, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        X_train_s = (X_train - mu) / sd
        X_test_s = (X_test - mu) / sd
        Vt = _robust_svd_vt(X_train_s)
        n_comp = min(n_components, Vt.shape[0])
        comps = Vt[:n_comp].T
        X_train_p = X_train_s @ comps
        X_test_p = X_test_s @ comps
        folds.append((train_idx, test_idx, X_train_p, X_test_p))
    return folds


def _robust_svd_vt(X: np.ndarray) -> np.ndarray:
    """np.linalg.svd (LAPACK gesdd) は特定の入力で "SVD did not converge" を
    起こすことがある（実データで発生を確認: fold内標準化後の特定の
    episode分割で発生）。gesvd driver（遅いが安定）にフォールバックし、
    それでも失敗する場合は共分散行列の固有値分解（数学的に同値、SVDの
    収束問題を回避できる）にフォールバックする。"""
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt
    except np.linalg.LinAlgError:
        pass
    try:
        import scipy.linalg
        _, _, Vt = scipy.linalg.svd(X, full_matrices=False, lapack_driver="gesvd")
        return Vt
    except Exception:
        pass
    # 固有値分解フォールバック（X^T X の固有ベクトル = 右特異ベクトル）
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order].T


def _loeo_probe_with_folds(folds: List[Tuple], y: np.ndarray) -> Tuple[float, np.ndarray]:
    accs = []
    for train_idx, test_idx, X_train_p, X_test_p in folds:
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train_p, y_train)
        accs.append(clf.score(X_test_p, y_test))
    accs = np.array(accs)
    return float(accs.mean()) if len(accs) else float("nan"), accs


def loeo_probe(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_components: int = 30,
) -> Tuple[float, np.ndarray]:
    """fold内標準化PCA + LogisticRegression、LOEO(Leave-One-Episode-Out)交差検証。
    Returns: (mean_accuracy, per_fold_accuracies)"""
    folds = _fold_pca_projections(X, groups, n_components)
    return _loeo_probe_with_folds(folds, y)


def loeo_probe_permutation_test(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_components: int = 30,
    n_perm: int = 100, seed: int = 0,
) -> Dict[str, float]:
    """各foldでepisodeブロック単位でラベルをシャッフルするpermutationテスト。
    fold-PCAは観測値・全null反復で共有する（labelに依存しないため）。"""
    rng = np.random.RandomState(seed)
    folds = _fold_pca_projections(X, groups, n_components)
    observed, _ = _loeo_probe_with_folds(folds, y)
    unique_groups = np.unique(groups)
    group_labels = {g: y[groups == g] for g in unique_groups}
    null_accs = []
    for _ in range(n_perm):
        shuffled_order = rng.permutation(unique_groups)
        new_y = np.empty_like(y)
        for g_orig, g_new in zip(unique_groups, shuffled_order):
            mask_orig = groups == g_orig
            src = group_labels[g_new]
            if len(src) == mask_orig.sum():
                new_y[mask_orig] = src
            else:
                new_y[mask_orig] = rng.choice(src, size=mask_orig.sum(), replace=True)
        acc, _ = _loeo_probe_with_folds(folds, new_y)
        null_accs.append(acc)
    null_accs = np.array(null_accs)
    p = float((null_accs >= observed).mean())
    return {"observed_acc": observed, "null_mean": float(np.nanmean(null_accs)), "p_value": p}
