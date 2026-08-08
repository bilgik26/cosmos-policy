"""
Cosmos Policy 画像生成解析スクリプト

【目的】
  Cosmos Policy は アクションと同時に「future image」(将来の観測) を生成する。
  このスクリプトでは、アクション生成解析と同様の手法を
  「画像生成の側」に適用し、次の2つの問いに答える:

  [ラテント解析] デノイジング過程で future image latent はどのように変化するか?
    ラテント変化量 (||Δx̂₀_img||₂) vs アクション
    周波数解析 (2D-FFT) — 低周波成分が先に安定するか?
    ノルム推移 (layer ごと、step ごと)

  [隠れ状態解析] DiT 内部表現で image token はどのような情報を保持するか?
    PCA 可視化 (タスク進行との対応)
    Linear CKA (層別)
    線形プロービング — image token 特徴量でスキルフェーズを予測できるか?

【アーキテクチャ理解】
  RoboCasa の latent sequence (state_t=11):
    T=0:  blank (placeholder)
    T=1:  curr proprio
    T=2:  curr wrist image    ← 現在の観測 (入力として固定)
    T=3:  curr primary image  ← 現在の観測 (入力として固定)
    T=4:  curr secondary image← 現在の観測 (入力として固定)
    T=5:  action chunk        ← アクション生成 (拡散)
    T=6:  future proprio
    T=7:  future wrist image  ← 将来の観測 (画像生成の対象)
    T=8:  future primary image← 将来の観測 (画像生成の対象) ★メイン解析対象
    T=9:  future secondary img← 将来の観測 (画像生成の対象)
    T=10: value

  Latent shape: (B, C'=16, T'=11, H'=28, W'=28)
  VAE decode: model.decode(latent) → (B, 3, T_raw, H_raw, W_raw)

実行例:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.image_latent.image_analysis \\
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \\
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \\
      --config_file cosmos_policy/config/config.py \\
      --use_wrist_image True --num_wrist_images 1 \\
      --use_proprio True --normalize_proprio True --unnormalize_actions True \\
      --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \\
      --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \\
      --trained_with_image_aug True \\
      --chunk_size 32 --num_open_loop_steps 16 \\
      --task_name PnPCounterToCab \\
      --seed 195 --randomize_seed False --deterministic True \\
      --use_variance_scale False --use_jpeg_compression True --flip_images True \\
      --num_denoising_steps_action 5 \\
      --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \\
      --data_collection False \\
      --num_image_episodes 5 \\
      --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/image_generation
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

# NOTE: Do NOT call torch.cuda.set_device() at module level.
# When set_device(1) is called, the NVIDIA EGL driver enumerates only GPU 1
# and fails (renderD129 has permission denied in Singularity).
# Instead we call set_device(1) in main(), AFTER the robosuite env (EGL) is created.
import cosmos_policy.experiments.robot.cosmos_utils as _cosmos_utils

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS,
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.constants import ACTION_DIM

from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    STATE_T,
    ACTION_T_IDX,
    FUTURE_WRIST_T_IDX,
    FUTURE_PRIMARY_T_IDX,
    FUTURE_SECONDARY_T_IDX,
    CURR_PRIMARY_T_IDX,
    CURR_WRIST_T_IDX,
    VALUE_T_IDX,
    FUTURE_IMAGE_TIDXS,
    FUTURE_IMAGE_NAMES,
    PROBE_LAYERS,
    PROBE_LAYER_SHORT,
    NUM_DENOISE_STEPS,
    SIGMA_SCHEDULE,
    linear_cka,
    pca_2d,
    skill_phase_labels,
)


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class ImageAnalysisConfig(PolicyEvalConfig):
    num_image_episodes: int = 50
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/image_generation"


# ── Latent Capture ────────────────────────────────────────────────────────────
class LatentCapture:
    """
    x0_fn をラップして各デノイジングステップの predicted clean latent x̂₀ を収集。

    収集するもの:
      - x̂₀ の各 T 位置の L2 norm
      - x̂₀ の各 T 位置の flattened vector (FFT 解析用)
    """

    def __init__(self):
        self.call_records: List[Dict] = []
        self._current_step = -1
        self._current_step_data: Dict = {}
        self._prev_x0: Optional[torch.Tensor] = None  # for Δx̂₀

    def reset_for_policy_call(self):
        self._current_step = -1
        self._current_step_data = {}
        self._prev_x0 = None

    def before_denoise_step(self):
        self._current_step += 1

    def capture_x0(self, x0_hat: torch.Tensor):
        """x0_fn の戻り値テンソル (B, C', T', H', W') を受け取って記録"""
        if self._current_step < 0:
            return
        k = self._current_step
        # x0_hat: (B=1, C'=16, T'=11, H'=28, W'=28)
        x0 = x0_hat.detach().float().cpu()  # (1, 16, 11, 28, 28)

        step_data = {}
        t_indices = [
            ACTION_T_IDX,
            CURR_PRIMARY_T_IDX,
            FUTURE_WRIST_T_IDX,
            FUTURE_PRIMARY_T_IDX,
            FUTURE_SECONDARY_T_IDX,
        ]
        for t_idx in t_indices:
            frame = x0[0, :, t_idx, :, :]  # (16, 28, 28)
            flat = frame.numpy().ravel()     # (16*28*28 = 12544,)
            norm = float(np.linalg.norm(flat))
            step_data[t_idx] = {
                "norm": norm,
                "flat": flat.astype(np.float32),
            }

        # Δx̂₀ from previous step
        x0_full = x0[0].numpy()  # (16, 11, 28, 28)
        if self._prev_x0 is not None:
            for t_idx in t_indices:
                f_cur = x0_full[:, t_idx, :, :].ravel()
                f_pre = self._prev_x0[:, t_idx, :, :].ravel()
                step_data[t_idx]["delta_norm"] = float(np.linalg.norm(f_cur - f_pre))
        else:
            for t_idx in t_indices:
                step_data[t_idx]["delta_norm"] = 0.0

        self._prev_x0 = x0_full
        self._current_step_data[k] = step_data

    def finalize_policy_call(self, episode_idx: int, call_idx: int):
        self.call_records.append({
            "episode": episode_idx,
            "call_idx": call_idx,
            "step_data": {k: dict(v) for k, v in self._current_step_data.items()},
        })


# ── Block Hidden State Capture ────────────────────────────────────────────────
class ImageTokenCapture:
    """
    DiT block 出力の image token 位置の hidden state を収集する。

    各ブロック出力は (B, T', H', W', D) 形状。
    action token (T=5) と future_primary image token (T=8) の
    HW 平均ベクトル (D,) を保存する。
    """

    def __init__(self, probe_layers: List[int]):
        self.probe_layers = probe_layers
        self._handles: List = []
        self._current_step = -1
        self._current_feats: Dict = {}  # {k: {layer: {"action": (D,), "future_primary": (D,), "curr_primary": (D,)}}}
        self.call_records: List[Dict] = []

    def register(self, model):
        net = model.net
        for layer_idx in self.probe_layers:
            block = net.blocks[layer_idx]
            h = block.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset_for_policy_call(self):
        self._current_step = -1
        self._current_feats = {}

    def before_denoise_step(self):
        self._current_step += 1
        self._current_feats.setdefault(self._current_step, {})

    def finalize_policy_call(self, episode_idx: int, call_idx: int):
        self.call_records.append({
            "episode": episode_idx,
            "call_idx": call_idx,
            "step_feats": {k: dict(ld) for k, ld in self._current_feats.items()},
        })

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if self._current_step < 0:
                return
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return
            k = self._current_step
            B, T, H, W, D = output.shape
            if T < STATE_T:
                return
            rec = {}
            for (name, t_idx) in [
                ("action", ACTION_T_IDX),
                ("future_primary", FUTURE_PRIMARY_T_IDX),
                ("curr_primary", CURR_PRIMARY_T_IDX),
                ("future_wrist", FUTURE_WRIST_T_IDX),
                ("future_secondary", FUTURE_SECONDARY_T_IDX),
            ]:
                if t_idx < T:
                    feat = output[0, t_idx].float().mean(dim=(0, 1))  # (D,)
                    rec[name] = feat.detach().cpu().numpy()
            self._current_feats.setdefault(k, {})[layer_idx] = rec
        return hook


# ── Wrapped inference ─────────────────────────────────────────────────────────
def get_action_with_captures(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    latent_cap: LatentCapture,
    token_cap: ImageTokenCapture,
    seed: int,
    num_denoising_steps: int,
):
    """LatentCapture + ImageTokenCapture を有効にしてアクションを生成"""
    latent_cap.reset_for_policy_call()
    token_cap.reset_for_policy_call()
    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            def wrapped(x_t, sigma):
                latent_cap.before_denoise_step()
                token_cap.before_denoise_step()
                x0_hat = x0_fn_raw(x_t, sigma)
                latent_cap.capture_x0(x0_hat)
                return x0_hat
            return wrapped, extra
        else:
            x0_fn_raw = result
            def wrapped(x_t, sigma):
                latent_cap.before_denoise_step()
                token_cap.before_denoise_step()
                x0_hat = x0_fn_raw(x_t, sigma)
                latent_cap.capture_x0(x0_hat)
                return x0_hat
            return wrapped

    model.get_x0_fn_from_batch = patched_get_x0_fn
    try:
        result = get_action(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=observation,
            task_label_or_embedding=task_description,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = original_get_x0_fn
    return result


# skill_phase_labels / linear_cka / pca_2d は analysis_shared からインポート済み


def compute_fft_amplitude(flat: np.ndarray, spatial_h: int = 28, spatial_w: int = 28) -> np.ndarray:
    """
    (C*H*W,) latent を C 枚の 2D FFT に変換し、
    各チャンネルの amplitude spectrum を radial 平均で 1D に圧縮して返す。
    Returns: (n_radial_bins,) averaged radial amplitude
    """
    C = flat.shape[0] // (spatial_h * spatial_w)
    frames = flat.reshape(C, spatial_h, spatial_w)
    freq_limit = min(spatial_h, spatial_w) // 2
    radial_bins = freq_limit
    radial_accum = np.zeros(radial_bins)

    ys, xs = np.meshgrid(np.arange(spatial_h), np.arange(spatial_w), indexing="ij")
    cy, cx = spatial_h // 2, spatial_w // 2
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    for c in range(C):
        f = np.fft.fftshift(np.fft.fft2(frames[c]))
        amp = np.abs(f)
        for b in range(radial_bins):
            mask = (r >= b) & (r < b + 1)
            if mask.any():
                radial_accum[b] += amp[mask].mean()

    return radial_accum / C


_T_NAME_MAP = {
    ACTION_T_IDX: "action",
    CURR_PRIMARY_T_IDX: "curr_primary",
    FUTURE_WRIST_T_IDX: "future_wrist",
    FUTURE_PRIMARY_T_IDX: "future_primary",
    FUTURE_SECONDARY_T_IDX: "future_secondary",
}


def save_image_features_npz(
    latent_cap: "LatentCapture",
    token_cap: "ImageTokenCapture",
    probe_layers: List[int],
    out_dir: Path,
    episode_success: Optional[Dict[int, bool]] = None,
    filename: str = "image_features.npz",
):
    """
    画像ラテント特徴量を npz として保存（§8 の全解析はこのファイルから行う）。

    保存内容:
      - norm_{token}_k{k}, delta_norm_{token}_k{k}: (N_calls,) — x̂₀ の L2 ノルムと
        前ステップからの変化量。token ∈ {action, curr_primary, future_wrist,
        future_primary, future_secondary}。
      - fft_{token}_k{k}: (N_calls, 14) — 2D FFT radial 平均振幅スペクトル
        （28x28 空間マップ、min(28,28)//2=14 bins）。
      - sigma_data_est_names / sigma_data_est_values: 線形ガウスヌル構成用に、
        最終デノイジングステップでの x̂₀ の per-component 標準偏差（全次元平均）。
      - hidfeat_{token}_k{k}_layer{l}: (N_calls, D) — DiT hidden state（token別）。
      - episode_labels, call_idx_labels: 上記 norm/fft/hidfeat 系列に対応する
        call 単位のメタデータ（latent_cap 由来）。
      - hidden_episode_labels, hidden_call_idx_labels: hidfeat 系列に対応する
        call 単位のメタデータ（token_cap 由来。latent_capと同じ順序のはずだが
        念のため別名で保存し、解析側で一致を確認できるようにする）。
      - episode_success_index/flag/per_call: 成功epフィルタ用。
    """
    npz_data: Dict[str, np.ndarray] = {}

    call_records = latent_cap.call_records
    episodes = np.array([r["episode"] for r in call_records])
    call_idxs = np.array([r["call_idx"] for r in call_records])
    npz_data["episode_labels"] = episodes
    npz_data["call_idx_labels"] = call_idxs

    t_indices = [
        ACTION_T_IDX, CURR_PRIMARY_T_IDX, FUTURE_WRIST_T_IDX,
        FUTURE_PRIMARY_T_IDX, FUTURE_SECONDARY_T_IDX,
    ]

    for t_idx in t_indices:
        name = _T_NAME_MAP[t_idx]
        for k in range(NUM_DENOISE_STEPS):
            norms = np.full(len(call_records), np.nan, dtype=np.float32)
            deltas = np.full(len(call_records), np.nan, dtype=np.float32)
            specs = np.full((len(call_records), 14), np.nan, dtype=np.float32)
            for i, r in enumerate(call_records):
                d = r["step_data"].get(k, {}).get(t_idx, None)
                if d is None:
                    continue
                norms[i] = d.get("norm", np.nan)
                deltas[i] = d.get("delta_norm", np.nan)
                if "flat" in d:
                    specs[i] = compute_fft_amplitude(d["flat"])
            npz_data[f"norm_{name}_k{k}"] = norms
            npz_data[f"delta_norm_{name}_k{k}"] = deltas
            npz_data[f"fft_{name}_k{k}"] = specs

    # sigma_data 推定（最終ステップ x̂₀ の population per-component std、線形ガウスヌル用）。
    # 成功epのみで計算する（episode_successが与えられていればそれでフィルタ）。
    # per-dimension mean/std も保存し、複数seed(run)を後でpooled varianceで
    # 正しく合成できるようにする（scalarのσ_data推定値だけだと単純平均しかできない）。
    kf = NUM_DENOISE_STEPS - 1
    success_ep_set = None
    if episode_success is not None:
        success_ep_set = {e for e, ok in episode_success.items() if ok}
    sigma_names, sigma_values = [], []
    for t_idx in [ACTION_T_IDX, FUTURE_WRIST_T_IDX, FUTURE_PRIMARY_T_IDX, FUTURE_SECONDARY_T_IDX]:
        name = _T_NAME_MAP[t_idx]
        flats = [
            r["step_data"].get(kf, {}).get(t_idx, {}).get("flat", None)
            for r in call_records
            if (success_ep_set is None or r["episode"] in success_ep_set)
        ]
        flats = [f for f in flats if f is not None]
        sigma_names.append(name)
        if len(flats) >= 2:
            mat = np.stack(flats, axis=0)
            per_dim_mean = mat.mean(axis=0)
            per_dim_std = mat.std(axis=0)
            sigma_values.append(float(per_dim_std.mean()))
            npz_data[f"sigma_data_per_dim_mean_{name}"] = per_dim_mean.astype(np.float32)
            npz_data[f"sigma_data_per_dim_std_{name}"] = per_dim_std.astype(np.float32)
            npz_data[f"sigma_data_n_{name}"] = np.array([len(flats)])
        else:
            sigma_values.append(float("nan"))
    npz_data["sigma_data_est_names"] = np.array(sigma_names)
    npz_data["sigma_data_est_values"] = np.array(sigma_values, dtype=np.float32)
    npz_data["sigma_data_success_only"] = np.array([success_ep_set is not None])
    npz_data["latent_dim_per_token"] = np.array([16 * 28 * 28])

    # Hidden-state features (DiT block outputs)
    tok_records = token_cap.call_records
    tok_episodes = np.array([r["episode"] for r in tok_records])
    tok_call_idxs = np.array([r["call_idx"] for r in tok_records])
    npz_data["hidden_episode_labels"] = tok_episodes
    npz_data["hidden_call_idx_labels"] = tok_call_idxs

    token_names = ["action", "curr_primary", "future_wrist", "future_primary", "future_secondary"]
    for k in range(NUM_DENOISE_STEPS):
        for layer_idx in probe_layers:
            for name in token_names:
                feats = []
                ok = True
                for r in tok_records:
                    v = r["step_feats"].get(k, {}).get(layer_idx, {}).get(name, None)
                    if v is None:
                        ok = False
                        break
                    feats.append(v)
                if ok and len(feats) > 0:
                    npz_data[f"hidfeat_{name}_k{k}_layer{layer_idx}"] = np.stack(feats, axis=0).astype(np.float32)

    if episode_success is not None:
        ep_ids_sorted = np.array(sorted(episode_success.keys()))
        ep_flags_sorted = np.array([bool(episode_success[e]) for e in ep_ids_sorted])
        npz_data["episode_success_index"] = ep_ids_sorted
        npz_data["episode_success_flag"] = ep_flags_sorted
        npz_data["episode_success_per_call"] = np.array(
            [bool(episode_success.get(int(e), False)) for e in episodes]
        )

    path = out_dir / filename
    np.savez_compressed(path, **npz_data)
    log_message(f"Saved: {path}")


# ── Plots — Latent denoising analysis ─────────────────────────────────────────
def plot_latent_denoising(records: List[Dict], out_dir: Path, task_name: str):
    """
    ラテントデノイジング解析: future image latent の変化量・ノルム・FFT

    1-A: 連続ステップ間変化量 (Δ norm) — future_primary vs action
    1-B: 各ステップでのラテントノルム推移 (T 位置別)
    1-C: FFT 周波数解析 — 全 k=0〜4 の周波数成分比較
    """
    N = len(records)
    if N == 0:
        return

    # Aggregate per-step data
    t_targets = {
        ACTION_T_IDX: "action",
        FUTURE_PRIMARY_T_IDX: "future_primary",
        FUTURE_WRIST_T_IDX: "future_wrist",
        FUTURE_SECONDARY_T_IDX: "future_secondary",
        CURR_PRIMARY_T_IDX: "curr_primary",
    }

    # delta_norms[t_idx][k] = list of delta norms across calls (k≥1)
    delta_norms = {t: [[] for _ in range(NUM_DENOISE_STEPS)] for t in t_targets}
    # latent_norms[t_idx][k] = list of latent norms
    latent_norms = {t: [[] for _ in range(NUM_DENOISE_STEPS)] for t in t_targets}
    # fft_spectra[t_idx][k] = list of radial spectra
    fft_spectra = {t: [[] for _ in range(NUM_DENOISE_STEPS)] for t in t_targets}

    for rec in records:
        sd = rec["step_data"]
        for k_str, step_d in sd.items():
            k = int(k_str)
            if k >= NUM_DENOISE_STEPS:
                continue
            for t_idx in t_targets:
                if t_idx in step_d:
                    d = step_d[t_idx]
                    latent_norms[t_idx][k].append(d["norm"])
                    if k > 0:
                        delta_norms[t_idx][k].append(d["delta_norm"])
                    spectrum = compute_fft_amplitude(d["flat"])
                    fft_spectra[t_idx][k].append(spectrum)

    def mean_std(lst):
        if not lst:
            return 0.0, 0.0
        a = np.array(lst)
        return float(a.mean()), float(a.std())

    # ── Plot 1-A: Δnorm per step (future_primary vs action) ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"(Image): Latent Change ||Δx̂₀||₂ Latent Change ||Δx̂₀||₂ per Denoising Step\n"
        f"Task: {task_name}  N={N} policy calls",
        fontsize=10,
    )
    colors_map = {
        ACTION_T_IDX: ("steelblue", "action (T=5)"),
        FUTURE_PRIMARY_T_IDX: ("tomato", "future_primary (T=8)"),
        FUTURE_WRIST_T_IDX: ("goldenrod", "future_wrist (T=7)"),
        FUTURE_SECONDARY_T_IDX: ("mediumpurple", "future_secondary (T=9)"),
        CURR_PRIMARY_T_IDX: ("gray", "curr_primary (T=3) [fixed input]"),
    }
    x_steps = list(range(1, NUM_DENOISE_STEPS))
    x_labels = [f"k={k}\n(σ≈{SIGMA_SCHEDULE[k]:.0f})" for k in x_steps]

    for t_idx, (color, label) in colors_map.items():
        means = []
        stds = []
        for k in x_steps:
            m, s = mean_std(delta_norms[t_idx][k])
            means.append(m)
            stds.append(s)
        means, stds = np.array(means), np.array(stds)
        axes[0].plot(x_steps, means, "o-", color=color, linewidth=2, markersize=5, label=label)
        axes[0].fill_between(x_steps, means - stds, means + stds, alpha=0.15, color=color)

    axes[0].set_xlabel("Denoising step k")
    axes[0].set_ylabel("||x̂₀(k) - x̂₀(k-1)||₂")
    axes[0].set_title("Δ latent norm per step (all T positions)")
    axes[0].set_xticks(x_steps)
    axes[0].set_xticklabels(x_labels, fontsize=7)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Normalized version (ratio to final step)
    for t_idx, (color, label) in colors_map.items():
        means = []
        for k in x_steps:
            m, _ = mean_std(delta_norms[t_idx][k])
            means.append(m)
        total = sum(means) + 1e-9
        ratios = [m / total for m in means]
        axes[1].plot(x_steps, ratios, "o-", color=color, linewidth=2, markersize=5, label=label)

    axes[1].set_xlabel("Denoising step k")
    axes[1].set_ylabel("Fraction of total Δ norm")
    axes[1].set_title("Δ norm fraction (normalized to sum=1)")
    axes[1].set_xticks(x_steps)
    axes[1].set_xticklabels(x_labels, fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "image_delta_norm.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 1-B: Latent norm per step ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"(Image): x̂₀ Latent Norm x̂₀ Latent Norm per Denoising Step\n"
        f"Task: {task_name}  N={N} policy calls",
        fontsize=10,
    )
    k_steps = list(range(NUM_DENOISE_STEPS))
    k_labels = [f"k={k}\n(σ≈{SIGMA_SCHEDULE[k]:.0f})" for k in k_steps]

    for t_idx, (color, label) in colors_map.items():
        means = []
        stds = []
        for k in k_steps:
            m, s = mean_std(latent_norms[t_idx][k])
            means.append(m)
            stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(k_steps, means, "o-", color=color, linewidth=2, markersize=5, label=label)
        ax.fill_between(k_steps, means - stds, means + stds, alpha=0.1, color=color)

    ax.set_xlabel("Denoising step k")
    ax.set_ylabel("||x̂₀||₂ (mean ± std)")
    ax.set_xticks(k_steps)
    ax.set_xticklabels(k_labels, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "image_latent_norm.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 1-C: FFT 周波数解析 (全デノイジングステップ k=0〜4) ─────────────
    # 各ステップで action / future_primary / future_wrist の周波数スペクトルを比較
    focus_t_idxs = [ACTION_T_IDX, FUTURE_PRIMARY_T_IDX, FUTURE_WRIST_T_IDX]
    focus_names = {
        ACTION_T_IDX: ("steelblue", "action (T=5)"),
        FUTURE_PRIMARY_T_IDX: ("tomato", "future_primary (T=8)"),
        FUTURE_WRIST_T_IDX: ("goldenrod", "future_wrist (T=7)"),
    }
    fig, axes = plt.subplots(1, NUM_DENOISE_STEPS, figsize=(6 * NUM_DENOISE_STEPS, 5))
    if NUM_DENOISE_STEPS == 1:
        axes = [axes]
    fig.suptitle(
        f"(Image): Radial FFT Spectrum Radial FFT Spectrum of x̂₀ (全ステップ k=0〜4)\n"
        f"Task: {task_name}  N={N} policy calls",
        fontsize=10,
    )

    for ax, k_focus in zip(axes, range(NUM_DENOISE_STEPS)):
        for t_idx in focus_t_idxs:
            spectra = fft_spectra[t_idx][k_focus]
            if not spectra:
                continue
            sp_arr = np.stack(spectra, axis=0)  # (N, R)
            m = sp_arr.mean(axis=0)
            s = sp_arr.std(axis=0)
            color, label = focus_names[t_idx]
            r = np.arange(len(m))
            ax.semilogy(r, m, "-", color=color, linewidth=1.8, label=label)
            ax.fill_between(r, np.maximum(m - s, 1e-6), m + s, alpha=0.15, color=color)
        ax.set_xlabel("Radial frequency (pixels⁻¹)")
        ax.set_ylabel("Mean amplitude (log scale)")
        ax.set_title(f"k={k_focus} (σ≈{SIGMA_SCHEDULE[k_focus]:.0f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    p = out_dir / "image_fft_spectrum.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = {}
    for t_idx, (_, label) in colors_map.items():
        entry = {}
        for k in range(NUM_DENOISE_STEPS):
            m, s = mean_std(latent_norms[t_idx][k])
            entry[f"norm_k{k}_mean"] = round(m, 3)
            entry[f"norm_k{k}_std"] = round(s, 3)
        for k in range(1, NUM_DENOISE_STEPS):
            m, s = mean_std(delta_norms[t_idx][k])
            entry[f"delta_k{k}_mean"] = round(m, 3)
            entry[f"delta_k{k}_std"] = round(s, 3)
        summary[label] = entry

    with open(out_dir / "image_latent_stats.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log_message(f"Saved: {out_dir / 'image_latent_stats.json'}")

    return summary


# ── Plots — Hidden state analysis ─────────────────────────────────────────────
def plot_hidden_states(records: List[Dict], out_dir: Path, task_name: str):
    """
    隠れ状態解析: image token の層別特徴量解析

    2-1: future_primary image token の PCA (7 層 × 全ステップ k=0〜4)
    2-2: action token vs future_primary image token の Linear CKA (層 × ステップ)
    線形プロービング (Ridge) — image token でスキルフェーズを予測
    """
    N = len(records)
    if N == 0:
        return

    ep_arr = np.array([r["episode"] for r in records])
    ci_arr = np.array([r["call_idx"] for r in records])
    phase_labels = skill_phase_labels(ep_arr, ci_arr)

    # token_names: probe_layers × denoise_step × token_type
    token_types = ["action", "future_primary", "curr_primary", "future_wrist"]
    token_colors = {
        "action": "steelblue",
        "future_primary": "tomato",
        "curr_primary": "gray",
        "future_wrist": "goldenrod",
    }

    # Collect feature matrices: feat[layer][k][token_type] = (N, D)
    D = None
    feat = {}
    for l in PROBE_LAYERS:
        feat[l] = {}
        for k in range(NUM_DENOISE_STEPS):
            feat[l][k] = {tt: [] for tt in token_types}

    for rec in records:
        sf = rec["step_feats"]
        for k_str, ld in sf.items():
            k = int(k_str)
            if k >= NUM_DENOISE_STEPS:
                continue
            for l_str, type_feats in ld.items():
                l = int(l_str)
                if l not in feat:
                    continue
                for tt in token_types:
                    if tt in type_feats and type_feats[tt] is not None:
                        v = type_feats[tt]
                        feat[l][k][tt].append(v)
                        if D is None:
                            D = v.shape[0]

    def get_mat(l, k, tt):
        lst = feat[l][k][tt]
        if len(lst) == 0:
            return None
        m = np.stack(lst, axis=0)  # (N, D)
        if m.shape[0] != N:
            return None
        return m

    # ── Plot 2-1: PCA of future_primary image token (全ステップ k=0〜4, 全層) ─
    n_layers = len(PROBE_LAYERS)
    n_tt = 2  # action + future_primary
    phase_colors_map = {0: "royalblue", 1: "goldenrod", 2: "tomato"}
    phase_names_map = {0: "Early (reach)", 1: "Mid (grasp)", 2: "Late (place)"}

    for k_focus in range(NUM_DENOISE_STEPS):
        fig, axes = plt.subplots(n_tt, n_layers, figsize=(3.8 * n_layers, 4 * n_tt))
        fig.suptitle(
            f"(Image): PCA of Block Hidden States PCA of Block Hidden States (k={k_focus}, σ≈{SIGMA_SCHEDULE[k_focus]:.0f})\n"
            f"Task: {task_name}  Color = skill phase (blue=early, gold=mid, red=late)",
            fontsize=10,
        )

        for ti, tt in enumerate(["action", "future_primary"]):
            for li, l in enumerate(PROBE_LAYERS):
                ax = axes[ti, li]
                M = get_mat(l, k_focus, tt)
                if M is None or M.shape[0] < 4:
                    ax.set_visible(False)
                    continue
                try:
                    scores, evr = pca_2d(M)
                except Exception:
                    ax.set_visible(False)
                    continue
                for ph in [0, 1, 2]:
                    mask = phase_labels == ph
                    if mask.any():
                        ax.scatter(
                            scores[mask, 0], scores[mask, 1],
                            c=phase_colors_map[ph], s=12, alpha=0.65,
                            label=phase_names_map[ph] if li == 0 else "",
                        )
                ax.set_title(
                    f"{PROBE_LAYER_SHORT.get(l, f'B{l}')}\n({tt})\n"
                    f"EVR: {evr[0]:.1%}/{evr[1]:.1%}",
                    fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if li == 0:
                    ax.legend(fontsize=6, loc="upper left")

        plt.tight_layout()
        p = out_dir / f"image_pca_k{k_focus}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        log_message(f"Saved: {p}")

    # ── Plot 2-2: Linear CKA — action vs future_primary (layer × step) ───────
    cka_matrix = np.zeros((n_layers, NUM_DENOISE_STEPS))
    cka_curr_matrix = np.zeros((n_layers, NUM_DENOISE_STEPS))

    for li, l in enumerate(PROBE_LAYERS):
        for k in range(NUM_DENOISE_STEPS):
            M_act = get_mat(l, k, "action")
            M_img = get_mat(l, k, "future_primary")
            M_cur = get_mat(l, k, "curr_primary")
            if M_act is not None and M_img is not None:
                cka_matrix[li, k] = linear_cka(M_act, M_img)
            if M_act is not None and M_cur is not None:
                cka_curr_matrix[li, k] = linear_cka(M_act, M_cur)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"(Image): Linear CKA Linear CKA (action ↔ image token)\n"
        f"Task: {task_name}  Higher = more similar representations",
        fontsize=10,
    )
    layer_labels = [PROBE_LAYER_SHORT.get(l, f"B{l}") for l in PROBE_LAYERS]
    step_labels = [f"k={k}\n(σ≈{SIGMA_SCHEDULE[k]:.0f})" for k in range(NUM_DENOISE_STEPS)]

    for ax, mat, title in zip(
        axes,
        [cka_matrix, cka_curr_matrix],
        ["action ↔ future_primary", "action ↔ curr_primary"],
    ):
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_DENOISE_STEPS))
        ax.set_xticklabels(step_labels, fontsize=7)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_labels, fontsize=8)
        ax.set_title(f"CKA: {title}", fontsize=9)
        plt.colorbar(im, ax=ax, label="CKA similarity")
        for i in range(n_layers):
            for j in range(NUM_DENOISE_STEPS):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] < 0.5 else "black")

    plt.tight_layout()
    p = out_dir / "image_cka.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 2-3: Linear probing (image token → skill phase) ──────────────────
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    probe_accs = {}  # {tt: {layer: {k: accuracy}}}
    for tt in ["action", "future_primary", "curr_primary"]:
        probe_accs[tt] = {}
        for l in PROBE_LAYERS:
            probe_accs[tt][l] = {}
            for k in range(NUM_DENOISE_STEPS):
                M = get_mat(l, k, tt)
                if M is None or M.shape[0] < 10:
                    probe_accs[tt][l][k] = 0.0
                    continue
                # LOEO cross-validation (Leave-One-Episode-Out)
                scaler = StandardScaler()
                M_s = scaler.fit_transform(M)
                y = phase_labels[:M_s.shape[0]]
                unique_eps = np.unique(ep_arr[:M_s.shape[0]])
                correct = 0
                total = 0
                for ep in unique_eps:
                    test_mask = ep_arr[:M_s.shape[0]] == ep
                    train_mask = ~test_mask
                    if train_mask.sum() < 3 or test_mask.sum() == 0:
                        continue
                    clf = Ridge(alpha=1.0)
                    clf.fit(M_s[train_mask], y[train_mask])
                    preds = clf.predict(M_s[test_mask]).round().clip(0, 2).astype(int)
                    correct += (preds == y[test_mask]).sum()
                    total += test_mask.sum()
                acc = correct / total if total > 0 else 0.0
                probe_accs[tt][l][k] = float(acc)

    # Heatmap: layer × step for each token type
    n_tt_probe = 3
    fig, axes = plt.subplots(1, n_tt_probe, figsize=(6.5 * n_tt_probe, 5))
    fig.suptitle(
        f"(Image): Linear Probing Accuracy Linear Probing Accuracy (skill phase, 3-class)\n"
        f"Task: {task_name}  LOEO CV  Chance = 33%",
        fontsize=10,
    )
    tt_labels = {
        "action": "Action token (T=5)",
        "future_primary": "Future Primary Image token (T=8)",
        "curr_primary": "Curr Primary Image token (T=3, fixed input)",
    }
    for ax, tt in zip(axes, ["action", "future_primary", "curr_primary"]):
        mat = np.zeros((n_layers, NUM_DENOISE_STEPS))
        for li, l in enumerate(PROBE_LAYERS):
            for k in range(NUM_DENOISE_STEPS):
                mat[li, k] = probe_accs[tt][l][k]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
        ax.set_xticks(range(NUM_DENOISE_STEPS))
        ax.set_xticklabels([f"k={k}" for k in range(NUM_DENOISE_STEPS)], fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_labels, fontsize=8)
        ax.set_title(tt_labels[tt], fontsize=8)
        plt.colorbar(im, ax=ax, label="Accuracy")
        for i in range(n_layers):
            for j in range(NUM_DENOISE_STEPS):
                ax.text(j, i, f"{mat[i,j]:.0%}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.tight_layout()
    p = out_dir / "image_linear_probe.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    # ── Plot 2-4: Feature change magnitude (action vs image token, by layer) ──
    # For each layer, plot ||feat(k+1) - feat(k)||₂ averaged over calls
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"(Image): Hidden State Change per Denoising Step (||Δfeat||₂)\n"
        f"Task: {task_name}  action vs future_primary image token",
        fontsize=10,
    )
    layer_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_layers))

    # Compute and store hidden state delta stats for JSON saving
    hidden_delta_stats = {}
    for tt in ["action", "future_primary"]:
        hidden_delta_stats[tt] = {}
        for li, l in enumerate(PROBE_LAYERS):
            hidden_delta_stats[tt][PROBE_LAYER_SHORT.get(l, f"B{l}")] = {}

    for ax, tt in zip(axes, ["action", "future_primary"]):
        for li, l in enumerate(PROBE_LAYERS):
            delta_means = []
            for k in range(1, NUM_DENOISE_STEPS):
                M_prev = get_mat(l, k - 1, tt)
                M_curr = get_mat(l, k, tt)
                if M_prev is None or M_curr is None:
                    delta_means.append(0.0)
                    continue
                n = min(M_prev.shape[0], M_curr.shape[0])
                deltas = np.linalg.norm(M_curr[:n] - M_prev[:n], axis=1)
                val = float(deltas.mean())
                delta_means.append(val)
                hidden_delta_stats[tt][PROBE_LAYER_SHORT.get(l, f"B{l}")][f"k{k-1}→k{k}"] = round(val, 3)
            x = list(range(1, NUM_DENOISE_STEPS))
            ax.plot(x, delta_means, "o-", color=layer_colors[li], linewidth=2,
                    markersize=5, label=PROBE_LAYER_SHORT.get(l, f"B{l}"))
        ax.set_title(f"Hidden state Δ norm: {tt}", fontsize=9)
        ax.set_xlabel("Denoising step k")
        ax.set_ylabel("||feat(k) - feat(k-1)||₂")
        ax.set_xticks(range(1, NUM_DENOISE_STEPS))
        ax.set_xticklabels([f"k={k}\n(σ≈{SIGMA_SCHEDULE[k]:.0f})" for k in range(1, NUM_DENOISE_STEPS)], fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "image_hidden_delta.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {p}")

    with open(out_dir / "image_hidden_delta_stats.json", "w") as f:
        json.dump(hidden_delta_stats, f, indent=2, ensure_ascii=False)
    log_message(f"Saved: {out_dir / 'image_hidden_delta_stats.json'}")

    # Save probe accuracy stats
    probe_summary = {}
    for tt in ["action", "future_primary", "curr_primary"]:
        probe_summary[tt] = {}
        for l in PROBE_LAYERS:
            probe_summary[tt][str(l)] = {
                f"k{k}": round(probe_accs[tt][l][k], 4)
                for k in range(NUM_DENOISE_STEPS)
            }
    with open(out_dir / "image_probe_stats.json", "w") as f:
        json.dump(probe_summary, f, indent=2, ensure_ascii=False)
    log_message(f"Saved: {out_dir / 'image_probe_stats.json'}")

    # Save CKA stats
    cka_summary = {
        "action_vs_future_primary": {
            PROBE_LAYER_SHORT.get(l, f"B{l}"): {
                f"k{k}": round(float(cka_matrix[li, k]), 4)
                for k in range(NUM_DENOISE_STEPS)
            }
            for li, l in enumerate(PROBE_LAYERS)
        },
        "action_vs_curr_primary": {
            PROBE_LAYER_SHORT.get(l, f"B{l}"): {
                f"k{k}": round(float(cka_curr_matrix[li, k]), 4)
                for k in range(NUM_DENOISE_STEPS)
            }
            for li, l in enumerate(PROBE_LAYERS)
        },
    }
    with open(out_dir / "image_cka_stats.json", "w") as f:
        json.dump(cka_summary, f, indent=2, ensure_ascii=False)
    log_message(f"Saved: {out_dir / 'image_cka_stats.json'}")

    return probe_accs, cka_matrix


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import draccus
    cfg: ImageAnalysisConfig = draccus.parse(ImageAnalysisConfig)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message("=== Cosmos Policy Image Generation Analysis ===")
    log_message(f"Task: {cfg.task_name}")
    log_message(f"Episodes: {cfg.num_image_episodes}")
    log_message(f"Probe layers: {PROBE_LAYERS}")

    # Create env FIRST (before any CUDA init) so EGL can enumerate all GPU devices.
    # If set_device(1) is called first, NVIDIA EGL driver sees only GPU 1
    # which lacks DRI render permission in Singularity (renderD129 denied).
    log_message("Creating environment (EGL must init before CUDA)...")
    env, _ = create_robocasa_env(cfg)
    task_description = env.get_ep_meta().get("lang", cfg.task_name)
    log_message(f"Environment created. Task: {task_description}")

    # Now redirect CUDA to GPU 1 before loading the 2B model
    # (GPU 0 is occupied by qwen3-vllm with ~21 GiB)
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        _cosmos_utils.DEVICE = torch.device("cuda:1")
        log_message("CUDA device set to GPU 1 for model loading.")

    # Load model
    log_message("Loading model...")
    model, cosmos_config = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    # GPU0のみ使用する運用に変更。CUDA_VISIBLE_DEVICES=0で1枚しか見えない前提のため
    # worker_id=0（可視デバイス内の論理indexは常に0）。
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=0)
    log_message("Model loaded.")

    num_blocks = len(model.net.blocks)
    actual_probe = [l for l in PROBE_LAYERS if l < num_blocks]
    log_message(f"Effective probe layers: {actual_probe}")

    # Initialize captures
    latent_cap = LatentCapture()
    token_cap = ImageTokenCapture(actual_probe)
    token_cap.register(model)
    log_message("Hooks registered.")

    # Initialize T5 cache
    from cosmos_policy.experiments.robot.cosmos_utils import get_t5_embedding_from_cache
    get_t5_embedding_from_cache(task_description)

    all_ep_idxs = []
    all_call_idxs = []
    n_success = 0
    episode_success: Dict[int, bool] = {}

    for ep_idx in range(cfg.num_image_episodes):
        log_message(f"\n--- Episode {ep_idx + 1}/{cfg.num_image_episodes} ---")
        obs = env.reset()
        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        done = False
        step_count = 0
        call_idx_in_ep = 0
        action_queue = deque()
        max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
        success = False

        while not done and step_count < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg)
                result = get_action_with_captures(
                    cfg=cfg,
                    model=model,
                    dataset_stats=dataset_stats,
                    observation=observation,
                    task_description=task_description,
                    latent_cap=latent_cap,
                    token_cap=token_cap,
                    seed=cfg.seed + ep_idx * 1000 + call_idx_in_ep,
                    num_denoising_steps=cfg.num_denoising_steps_action,
                )
                latent_cap.finalize_policy_call(ep_idx, call_idx_in_ep)
                token_cap.finalize_policy_call(ep_idx, call_idx_in_ep)
                all_ep_idxs.append(ep_idx)
                all_call_idxs.append(call_idx_in_ep)
                call_idx_in_ep += 1

                actions = result["actions"]
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a_step = actions[i]
                    if a_step.shape[-1] == 7 and env.action_dim == 12:
                        a_step = np.concatenate([a_step, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    action_queue.append(a_step)

            action = action_queue.popleft()
            obs, reward, done, info = env.step(action)
            step_count += 1
            # env.step()の返すdone/info["success"]はignore_done設定下では常に
            # False/未設定のため、他スクリプト(feature_analysis.py等)と同様に
            # env._check_success()で直接判定する（元のinfo.get("success")依存の
            # 実装は常にFalseを返すバグだった: seed195で50episode中0成功という
            # 明らかに他スクリプト(55-60%成功)と矛盾する結果で発覚）。
            if env._check_success():
                success = True
                done = True

        if success:
            n_success += 1
        episode_success[ep_idx] = success
        log_message(f"Episode {ep_idx + 1}: {'SUCCESS' if success else 'FAIL'} ({step_count} steps, {call_idx_in_ep} policy calls)")

    env.close()

    log_message(f"\nSuccess rate: {n_success}/{cfg.num_image_episodes}")
    log_message(f"Total policy calls: {len(latent_cap.call_records)}")

    # Run analyses
    log_message("\n=== Running Latent Denoising Analysis ===")
    plot_latent_denoising(latent_cap.call_records, out_dir, cfg.task_name)

    log_message("\n=== Running Hidden State Analysis ===")
    plot_hidden_states(token_cap.call_records, out_dir, cfg.task_name)

    # Save metadata
    meta = {
        "task_name": cfg.task_name,
        "n_episodes": cfg.num_image_episodes,
        "success_rate": n_success / max(cfg.num_image_episodes, 1),
        "total_policy_calls": len(latent_cap.call_records),
        "probe_layers": actual_probe,
        "state_t": STATE_T,
        "action_t_idx": ACTION_T_IDX,
        "future_primary_t_idx": FUTURE_PRIMARY_T_IDX,
        "future_wrist_t_idx": FUTURE_WRIST_T_IDX,
        "future_secondary_t_idx": FUTURE_SECONDARY_T_IDX,
        "curr_primary_t_idx": CURR_PRIMARY_T_IDX,
        "sigma_schedule": SIGMA_SCHEDULE,
    }
    with open(out_dir / "image_analysis_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log_message(f"\nSaved: {out_dir / 'image_analysis_meta.json'}")

    # Save raw data (§8 の全解析はここで保存する image_features.npz から行う)
    log_message("Saving raw latent/hidden-state data...")
    save_image_features_npz(latent_cap, token_cap, actual_probe, out_dir, episode_success=episode_success)

    log_message("\n=== Image Generation Analysis Complete ===")
    token_cap.remove()


if __name__ == "__main__":
    main()
