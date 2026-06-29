"""
Cosmos Policy 検証スクリプト共通モジュール

各解析スクリプト間で共有する定数・ユーティリティをまとめたモジュール。
"""
from typing import Tuple

import numpy as np

# ── RoboCasa Latent Sequence T-Index 定数 ─────────────────────────────────────
# state_t=11 の構造:
#   T=0: blank (placeholder)    T=5: action        T=10: value
#   T=1: proprio                T=6: future_proprio
#   T=2: curr_wrist             T=7: future_wrist
#   T=3: curr_primary           T=8: future_primary
#   T=4: curr_secondary         T=9: future_secondary

STATE_T = 11

BLANK_T_IDX = 0
PROPRIO_T_IDX = 1
CURR_WRIST_T_IDX = 2
CURR_PRIMARY_T_IDX = 3
CURR_SECONDARY_T_IDX = 4
ACTION_T_IDX = 5
ACTION_LATENT_IDX = 5          # backward compat alias
ACTION_LATENT_IDX_ROBOCASA = 5  # backward compat alias
FUTURE_PROPRIO_T_IDX = 6
FUTURE_WRIST_T_IDX = 7
FUTURE_PRIMARY_T_IDX = 8
FUTURE_SECONDARY_T_IDX = 9
VALUE_T_IDX = 10

T_NAMES = {
    BLANK_T_IDX: "blank",
    PROPRIO_T_IDX: "proprio",
    CURR_WRIST_T_IDX: "curr_wrist",
    CURR_PRIMARY_T_IDX: "curr_primary",
    CURR_SECONDARY_T_IDX: "curr_secondary",
    ACTION_T_IDX: "action",
    FUTURE_PROPRIO_T_IDX: "future_proprio",
    FUTURE_WRIST_T_IDX: "future_wrist",
    FUTURE_PRIMARY_T_IDX: "future_primary",
    FUTURE_SECONDARY_T_IDX: "future_secondary",
    VALUE_T_IDX: "value",
}

INPUT_T_IDXS = [BLANK_T_IDX, PROPRIO_T_IDX, CURR_WRIST_T_IDX, CURR_PRIMARY_T_IDX, CURR_SECONDARY_T_IDX]
OUTPUT_T_IDXS = [ACTION_T_IDX, FUTURE_PROPRIO_T_IDX, FUTURE_WRIST_T_IDX, FUTURE_PRIMARY_T_IDX, FUTURE_SECONDARY_T_IDX, VALUE_T_IDX]
IMAGE_INPUT_T_IDXS = [CURR_WRIST_T_IDX, CURR_PRIMARY_T_IDX, CURR_SECONDARY_T_IDX]
IMAGE_OUTPUT_T_IDXS = [FUTURE_WRIST_T_IDX, FUTURE_PRIMARY_T_IDX, FUTURE_SECONDARY_T_IDX]
FUTURE_IMAGE_T_IDXS = IMAGE_OUTPUT_T_IDXS  # alias
FUTURE_IMAGE_TIDXS = IMAGE_OUTPUT_T_IDXS   # backward compat alias

FUTURE_IMAGE_NAMES = {
    FUTURE_WRIST_T_IDX: "future_wrist",
    FUTURE_PRIMARY_T_IDX: "future_primary",
    FUTURE_SECONDARY_T_IDX: "future_secondary",
}

# ── 自己注意の空間定数 ────────────────────────────────────────────────────────
# patch_spatial=2 → H_p = W_p = 14 → 各 T 位置 = 14×14 = 196 トークン
PATCHES_PER_T = 196
SPATIAL_H = 14
SPATIAL_W = 14
TOTAL_SEQ_LEN = STATE_T * PATCHES_PER_T  # 2156

# ── プローブ層定数 ─────────────────────────────────────────────────────────────
# 28 ブロック (2B DiT) 中 7 箇所を等間隔にサンプル
PROBE_LAYERS = [0, 4, 9, 13, 18, 22, 27]
PROBE_BLOCKS = PROBE_LAYERS  # alias used in attention scripts

PROBE_LAYER_LABELS = {
    0:  "Block-0\n(Shallowest)",
    4:  "Block-4\n(Early)",
    9:  "Block-9\n(Early-Mid)",
    13: "Block-13\n(Mid)",
    18: "Block-18\n(Late-Mid)",
    22: "Block-22\n(Deep)",
    27: "Block-27\n(Deepest)",
}
PROBE_LAYER_SHORT = {
    0: "Blk-0", 4: "Blk-4", 9: "Blk-9", 13: "Blk-13",
    18: "Blk-18", 22: "Blk-22", 27: "Blk-27",
}

# ── デノイジング定数 ──────────────────────────────────────────────────────────
CHUNK_SIZE = 32
NUM_DENOISE_STEPS = 5
SIGMA_SCHEDULE = [80.0, 42.3, 21.0, 9.6, 4.0]


# ── 解析ユーティリティ ────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA 類似度 [0, 1]。値が大きいほど表現が類似。"""
    N = X.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    K = H @ (X @ X.T) @ H
    L = H @ (Y @ Y.T) @ H
    hsic_xy = np.trace(K @ L)
    denom = np.linalg.norm(K, "fro") * np.linalg.norm(L, "fro")
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def pca_numpy(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD による PCA。
    Returns: (scores: (N, n), explained_variance_ratio: (n,), singular_values: (min(N,D),))
    """
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    scores = X_c @ Vt[:n_components].T
    var = S**2 / max(X.shape[0] - 1, 1)
    evr = var[:n_components] / (var.sum() + 1e-12)
    return scores, evr, S


def pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pca_numpy の 2D 版。(scores, evr) のみ返す (pca_numpy の旧 API 互換)。"""
    scores, evr, _ = pca_numpy(X, n_components=2)
    return scores, evr


def effective_rank(singular_values: np.ndarray) -> float:
    """
    Participation Ratio = (Σσᵢ²)² / Σσᵢ⁴。
    完全一様分布で最大 (= N)、1 点集中で最小 (= 1)。
    """
    sv2 = singular_values**2
    total = sv2.sum()
    if total < 1e-12:
        return 1.0
    return float(total**2 / (sv2**2).sum())


def skill_phase_labels(ep_arr: np.ndarray, ci_arr: np.ndarray) -> np.ndarray:
    """
    エピソード内の正規化進行度から 3 クラスのフェーズラベルを付与。
    0=early (reach), 1=mid (grasp), 2=late (place)
    """
    N = len(ep_arr)
    progress = np.zeros(N)
    for ep in np.unique(ep_arr):
        mask = ep_arr == ep
        if mask.any():
            max_ci = ci_arr[mask].max()
            progress[mask] = ci_arr[mask] / max(max_ci, 1)
    labels = np.zeros(N, dtype=int)
    labels[progress > 1 / 3] = 1
    labels[progress > 2 / 3] = 2
    return labels
