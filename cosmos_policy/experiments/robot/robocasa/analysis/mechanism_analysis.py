"""
Cosmos Policy 拡散モデルのメカニズム解析スクリプト
mechanism_eval.md の検証計画に基づく実装

【テーマ1】デノイジング過程の解析
  1-A. 各ステップの予測軌跡 (x_hat_0) の変化と定量化
  1-B. 周波数領域（FFT）解析
  1-C. スコア関数（ノイズ予測）のノルム解析

実行方法（Singularity コンテナ内で）:
  python -m cosmos_policy.experiments.robot.robocasa.mechanism_analysis \
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
      --config_file cosmos_policy/config/config.py \
      --use_wrist_image True --num_wrist_images 1 \
      --use_proprio True --normalize_proprio True --unnormalize_actions True \
      --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
      --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
      --trained_with_image_aug True \
      --chunk_size 32 --num_open_loop_steps 16 \
      --task_name PnPCounterToCab \
      --num_trials_per_task 10 \
      --seed 195 --randomize_seed False --deterministic True \
      --use_variance_scale False --use_jpeg_compression True --flip_images True \
      --num_denoising_steps_action 5 \
      --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
      --data_collection False \
      --num_analysis_episodes 10 \
      --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/action_denoising
"""

import ast
import json
import os
import pickle
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Cosmos Policy imports ────────────────────────────────────────────────────
os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    extract_action_chunk_from_latent_sequence,
    get_action,
    get_model,
    get_t5_embedding_from_cache,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    CONTROLLER_CONFIGS_PATH,
    TASK_MAX_STEPS,
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    CHUNK_SIZE,
    ACTION_LATENT_IDX_ROBOCASA,
)


# ── Hook: x0_fn をラップして中間予測を収集 ────────────────────────────────

class DenoiseCapture:
    """x0_fn を包んで各デノイジングステップの予測を記録するクラス"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_records: List[Dict] = []

    def wrap_x0_fn(self, x0_fn):
        capture = self

        def wrapped(x_t, sigma):
            x0 = x0_fn(x_t, sigma)
            sigma_val = float(sigma.float().mean().item())
            # noise_pred = (x_t - x0) / sigma
            noise_pred = (x_t.float() - x0.float()) / (sigma_val + 1e-8)
            capture.step_records.append(
                {
                    "sigma": sigma_val,
                    "x0_latent": x0.float().detach().cpu(),
                    "noise_pred_norm": float(noise_pred.norm().item()),
                }
            )
            return x0

        return wrapped


# ── アクション抽出 ────────────────────────────────────────────────────────

def extract_action_from_x0(x0_latent: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> Optional[np.ndarray]:
    """x0_latent (B, C, T, H, W) から action チャンクを (chunk_size, ACTION_DIM) で返す"""
    try:
        action_indices = torch.tensor([ACTION_LATENT_IDX_ROBOCASA], dtype=torch.int64)
        actions = extract_action_chunk_from_latent_sequence(
            x0_latent, action_shape=(chunk_size, ACTION_DIM), action_indices=action_indices
        )
        return actions[0].numpy()  # (chunk_size, ACTION_DIM)
    except Exception as e:
        print(f"  [extract_action] failed: {e}")
        return None


# ── get_action のラッパー（denoising ステップをキャプチャ付き） ────────────

def get_action_with_capture(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    capture: DenoiseCapture,
    seed: int,
    num_denoising_steps: int,
) -> Dict[str, Any]:
    """中間予測を記録しながらアクションを生成する"""
    capture.reset()

    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            return capture.wrap_x0_fn(x0_fn_raw), extra
        else:
            return capture.wrap_x0_fn(result)

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


# ── 解析関数 ──────────────────────────────────────────────────────────────

def compute_per_step_stats(all_records: List[List[Dict]], chunk_size: int = CHUNK_SIZE):
    """
    全 policy call の各デノイジングステップの統計を計算する。
    Returns:
        step_actions: {step_idx: list of (chunk_size, ACTION_DIM)}
        step_norms: {step_idx: list of noise_pred_norm}
        step_sigmas: {step_idx: list of sigma}
        step_fft_low: {step_idx: list of low-freq power}
        step_fft_high: {step_idx: list of high-freq power}
    """
    step_actions = defaultdict(list)
    step_norms = defaultdict(list)
    step_sigmas = defaultdict(list)
    step_fft_low = defaultdict(list)
    step_fft_high = defaultdict(list)

    for ep_records in all_records:
        for step_idx, rec in enumerate(ep_records):
            step_norms[step_idx].append(rec["noise_pred_norm"])
            step_sigmas[step_idx].append(rec["sigma"])

            act = extract_action_from_x0(rec["x0_latent"], chunk_size)
            if act is not None:
                step_actions[step_idx].append(act)
                # FFT over time axis for each action dim
                for dim in range(ACTION_DIM):
                    series = act[:, dim]
                    fft_mag = np.abs(np.fft.rfft(series))
                    freqs = np.fft.rfftfreq(len(series))
                    low_mask = freqs <= 0.25
                    high_mask = freqs > 0.25
                    step_fft_low[step_idx].append(fft_mag[low_mask].mean() if low_mask.any() else 0.0)
                    step_fft_high[step_idx].append(fft_mag[high_mask].mean() if high_mask.any() else 0.0)

    return step_actions, step_norms, step_sigmas, step_fft_low, step_fft_high


def compute_prediction_deltas(all_records: List[List[Dict]], chunk_size: int = CHUNK_SIZE):
    """連続するステップ間の予測変化量 ||x̂₀(k) - x̂₀(k-1)||₂ を計算"""
    transition_deltas = defaultdict(list)  # (step_i-1 -> step_i) -> list of L2 norms

    for ep_records in all_records:
        prev_act = None
        for step_idx, rec in enumerate(ep_records):
            act = extract_action_from_x0(rec["x0_latent"], chunk_size)
            if act is not None and prev_act is not None:
                delta = float(np.linalg.norm(act - prev_act))
                transition_deltas[step_idx].append(delta)
            prev_act = act

    return transition_deltas


# ── 追加解析関数 ───────────────────────────────────────────────────────────

def plot_trajectory_overlay(step_actions, out_dir, task_name, success_rate, sorted_steps, mean_sigmas):
    """k=0 と k=last の予測軌跡を同じグラフにオーバーレイ（全アクション次元）"""
    k_first = sorted_steps[0]
    k_last = sorted_steps[-1]
    if k_first not in step_actions or k_last not in step_actions:
        return
    if not step_actions[k_first] or not step_actions[k_last]:
        return

    sigma_first = mean_sigmas[0]
    sigma_last = mean_sigmas[-1]

    acts_first = np.stack(step_actions[k_first])  # (N, chunk_size, ACTION_DIM)
    acts_last = np.stack(step_actions[k_last])

    N, T, D = acts_first.shape
    t = np.arange(T)
    mean_first = acts_first.mean(axis=0)  # (T, D)
    mean_last = acts_last.mean(axis=0)
    std_first = acts_first.std(axis=0)
    std_last = acts_last.std(axis=0)

    fig, axes = plt.subplots(D, 1, figsize=(12, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]
    fig.suptitle(
        f"Theme 1-A Extension: Trajectory Overlay\n"
        f"Dashed = k={k_first} (σ≈{sigma_first:.1f})  Solid = k={k_last} (σ≈{sigma_last:.1f})\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  N={N} samples",
        fontsize=10,
    )

    for d, ax in enumerate(axes):
        for i in range(min(30, N)):
            ax.plot(t, acts_first[i, :, d], color="royalblue", alpha=0.04, linewidth=0.7)
            ax.plot(t, acts_last[i, :, d], color="tomato", alpha=0.04, linewidth=0.7)
        ax.fill_between(t,
                        mean_first[:, d] - std_first[:, d],
                        mean_first[:, d] + std_first[:, d],
                        alpha=0.15, color="royalblue")
        ax.fill_between(t,
                        mean_last[:, d] - std_last[:, d],
                        mean_last[:, d] + std_last[:, d],
                        alpha=0.15, color="tomato")
        ax.plot(t, mean_first[:, d], "--", color="royalblue", linewidth=2,
                label=f"k={k_first} (σ≈{sigma_first:.1f})")
        ax.plot(t, mean_last[:, d], "-", color="tomato", linewidth=2,
                label=f"k={k_last} (σ≈{sigma_last:.1f})")
        ax.set_ylabel(f"Dim {d}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Action chunk timestep (0 to chunk_size-1)")
    plt.tight_layout()
    path = out_dir / "theme1a_trajectory_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path}")


def plot_phase_analysis(step_actions, out_dir, task_name, success_rate, sorted_steps, mean_sigmas):
    """FFT 位相解析: k=0 と k=last の振幅スペクトル・位相差・時間シフト推定"""
    k_first = sorted_steps[0]
    k_last = sorted_steps[-1]
    if k_first not in step_actions or k_last not in step_actions:
        return {}
    if not step_actions[k_first] or not step_actions[k_last]:
        return {}

    sigma_first = mean_sigmas[0]
    sigma_last = mean_sigmas[-1]

    acts_first = np.stack(step_actions[k_first])  # (N, T, D)
    acts_last = np.stack(step_actions[k_last])
    N, T, D = acts_first.shape
    freqs = np.fft.rfftfreq(T)

    fft_first = np.fft.rfft(acts_first, axis=1)  # (N, F, D)
    fft_last = np.fft.rfft(acts_last, axis=1)

    # Phase difference wrapped to [-π, π]
    phase_diff = np.angle(fft_last) - np.angle(fft_first)  # (N, F, D)
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    mean_phase_diff = phase_diff.mean(axis=(0, 2))   # (F,)
    std_phase_diff = phase_diff.std(axis=(0, 2))

    # Amplitude spectra averaged over samples and dims
    amp_first = np.abs(fft_first).mean(axis=(0, 2))  # (F,)
    amp_last = np.abs(fft_last).mean(axis=(0, 2))

    # Linear fit to phase_diff vs freq (weighted by amplitude) → slope*2π = -time_shift
    valid = (freqs > 0) & (freqs < 0.5)
    time_shift = 0.0
    slope = 0.0
    fit_line = np.zeros_like(freqs)
    if valid.sum() > 2:
        w = amp_first[valid]
        coeffs = np.polyfit(freqs[valid], mean_phase_diff[valid], deg=1, w=w)
        slope = float(coeffs[0])
        time_shift = float(-slope / (2 * np.pi))
        fit_line = np.polyval(coeffs, freqs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Theme 1-B Extension: FFT Phase Analysis\n"
        f"k={k_first} (σ≈{sigma_first:.1f})  vs  k={k_last} (σ≈{sigma_last:.1f})  |  "
        f"Task: {task_name}  Success: {success_rate:.1%}",
        fontsize=10,
    )

    # Panel 1: Amplitude spectra
    axes[0].plot(freqs, amp_first, "o-", color="royalblue", alpha=0.8, markersize=4,
                 label=f"k={k_first} (σ≈{sigma_first:.1f})")
    axes[0].plot(freqs, amp_last, "s-", color="tomato", alpha=0.8, markersize=4,
                 label=f"k={k_last} (σ≈{sigma_last:.1f})")
    axes[0].set_xlabel("Frequency (cycles / timestep)")
    axes[0].set_ylabel("Mean FFT amplitude")
    axes[0].set_title("Amplitude Spectrum")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Phase difference + linear fit
    axes[1].errorbar(freqs, mean_phase_diff, yerr=std_phase_diff, fmt="o-",
                     color="purple", alpha=0.7, capsize=3, markersize=4,
                     label="Mean phase diff ± std")
    axes[1].plot(freqs, fit_line, "--", color="gray", linewidth=1.5,
                 label=f"Linear fit (slope={slope:.2f})\n→ time shift ≈ {time_shift:.3f} steps")
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle=":")
    axes[1].set_xlabel("Frequency (cycles / timestep)")
    axes[1].set_ylabel("Phase diff [rad]")
    axes[1].set_title(f"Phase Difference (k={k_first}→k={k_last})")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Phase difference per action dim
    phase_diff_per_dim = phase_diff.mean(axis=0)  # (F, D)
    for d in range(D):
        axes[2].plot(freqs, phase_diff_per_dim[:, d], alpha=0.75, linewidth=1.2, label=f"Dim {d}")
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle=":")
    axes[2].set_xlabel("Frequency (cycles / timestep)")
    axes[2].set_ylabel("Phase difference [rad]")
    axes[2].set_title("Phase Difference per Action Dim")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "theme1b_phase_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path}")
    log_message(f"  Estimated time shift k={k_first}→k={k_last}: {time_shift:.4f} timesteps")

    return {
        "phase_time_shift_steps": float(time_shift),
        "phase_linear_fit_slope": float(slope),
    }


# ── プロット ───────────────────────────────────────────────────────────────

def plot_all(all_records, out_dir: Path, task_name: str, success_rate: float, chunk_size: int = CHUNK_SIZE):
    out_dir.mkdir(parents=True, exist_ok=True)

    step_actions, step_norms, step_sigmas, step_fft_low, step_fft_high = compute_per_step_stats(all_records, chunk_size)
    transition_deltas = compute_prediction_deltas(all_records, chunk_size)

    sorted_steps = sorted(step_norms.keys())
    mean_norms = [np.mean(step_norms[s]) for s in sorted_steps]
    std_norms = [np.std(step_norms[s]) for s in sorted_steps]
    mean_sigmas = [np.mean(step_sigmas[s]) for s in sorted_steps]

    # ── Figure 1: デノイジング軌跡 (Theme 1-A) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Theme 1-A: x̂₀ Prediction Trajectory\nTask: {task_name}  Success: {success_rate:.1%}  "
        f"({len(all_records)} policy calls)",
        fontsize=12,
    )

    # 変化量バープロット
    if transition_deltas:
        sorted_trans = sorted(transition_deltas.keys())
        t_means = [np.mean(transition_deltas[t]) for t in sorted_trans]
        t_stds = [np.std(transition_deltas[t]) for t in sorted_trans]
        axes[0].bar(range(len(sorted_trans)), t_means, yerr=t_stds, color="steelblue", alpha=0.8, capsize=4)
        axes[0].set_xlabel("Denoising step transition")
        axes[0].set_ylabel("||x̂₀(k) - x̂₀(k-1)||₂")
        axes[0].set_title("Action Prediction Change (consecutive steps)")
        axes[0].set_xticks(range(len(sorted_trans)))
        axes[0].set_xticklabels([f"k={t-1}→{t}" for t in sorted_trans])
        axes[0].grid(True, alpha=0.3)

    # 各ステップの action トレース（Dim 0、平均）
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_steps)))
    for i, step_idx in enumerate(sorted_steps):
        if step_actions[step_idx]:
            acts = np.stack(step_actions[step_idx])  # (N, chunk_size, ACTION_DIM)
            mean_traj = acts[:, :, 0].mean(axis=0)
            sigma_str = f"{mean_sigmas[i]:.2f}" if i < len(mean_sigmas) else "?"
            axes[1].plot(mean_traj, color=colors[i], alpha=0.9, label=f"k={step_idx} (σ≈{sigma_str})")
    axes[1].set_xlabel("Action chunk timestep")
    axes[1].set_ylabel("Action dim 0 (mean over policy calls)")
    axes[1].set_title("Predicted Action Trajectory per Denoising Step")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path_1a = out_dir / "theme1a_denoising_trajectory.png"
    plt.savefig(path_1a, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path_1a}")

    # ── Figure 2: FFT 解析 (Theme 1-B) ──────────────────────────────────
    if step_fft_low:
        fig, ax = plt.subplots(figsize=(10, 5))
        low_means = [np.mean(step_fft_low[s]) for s in sorted_steps]
        high_means = [np.mean(step_fft_high[s]) for s in sorted_steps]
        ax.plot(sorted_steps, low_means, "o-", color="royalblue", label="Low-freq power (f ≤ 0.25)")
        ax.plot(sorted_steps, high_means, "s-", color="tomato", label="High-freq power (f > 0.25)")
        ax.set_xlabel("Denoising step index (0 = highest noise)")
        ax.set_ylabel("Mean FFT magnitude")
        ax.set_title(
            f"Theme 1-B: Spectral Analysis across Denoising Steps\n"
            f"Task: {task_name}  Success: {success_rate:.1%}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path_1b = out_dir / "theme1b_fft_analysis.png"
        plt.savefig(path_1b, dpi=150, bbox_inches="tight")
        plt.close()
        log_message(f"Saved: {path_1b}")

    # ── Figure 1-A Extension: 軌跡オーバーレイ ──────────────────────────
    plot_trajectory_overlay(step_actions, out_dir, task_name, success_rate, sorted_steps, mean_sigmas)

    # ── Figure 1-B Extension: 位相解析 ───────────────────────────────────
    plot_phase_analysis(step_actions, out_dir, task_name, success_rate, sorted_steps, mean_sigmas)

    # ── step_actions を npz 保存（再解析用） ─────────────────────────────
    npz_data = {str(k): np.stack(v) for k, v in step_actions.items() if v}
    np.savez(out_dir / "step_actions.npz", **npz_data)
    log_message(f"Saved: {out_dir / 'step_actions.npz'}")

    # ── Figure 3: スコア関数ノルム (Theme 1-C) ──────────────────────────
    if step_norms:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Theme 1-C: Score Function (Noise Prediction) Norm\n"
            f"Task: {task_name}  Success: {success_rate:.1%}",
            fontsize=12,
        )

        axes[0].errorbar(sorted_steps, mean_norms, yerr=std_norms, fmt="o-", color="darkorange", capsize=4)
        axes[0].set_xlabel("Denoising step index")
        axes[0].set_ylabel("||ε_θ(x_k, σ)||₂")
        axes[0].set_title("Score Function Norm")
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(sorted_steps, mean_sigmas, "s-", color="purple")
        axes[1].set_xlabel("Denoising step index")
        axes[1].set_ylabel("σ (noise level, log scale)")
        axes[1].set_title("Noise Level Schedule")
        axes[1].grid(True, alpha=0.3)

        # score norm vs sigma (scatter)
        axes[2].scatter(mean_sigmas, mean_norms, color="green", s=60, zorder=5)
        for i, (sig, nrm) in enumerate(zip(mean_sigmas, mean_norms)):
            axes[2].annotate(f"k={sorted_steps[i]}", (sig, nrm), fontsize=7, textcoords="offset points", xytext=(4, 2))
        axes[2].set_xlabel("σ (noise level)")
        axes[2].set_ylabel("||ε_θ||₂")
        axes[2].set_title("Score Norm vs Sigma")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path_1c = out_dir / "theme1c_score_norm.png"
        plt.savefig(path_1c, dpi=150, bbox_inches="tight")
        plt.close()
        log_message(f"Saved: {path_1c}")

    # ── Figure 4: サマリー ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Cosmos Policy Denoising Mechanism Analysis\n"
        f"Task: {task_name}  Success: {success_rate:.1%}  Calls: {len(all_records)}",
        fontsize=13,
    )

    # (0,0) Score norm
    axes[0, 0].errorbar(sorted_steps, mean_norms, yerr=std_norms, fmt="o-", color="darkorange", capsize=4)
    axes[0, 0].set_title("Score Function Norm per Step")
    axes[0, 0].set_xlabel("Step index k")
    axes[0, 0].set_ylabel("||ε_θ||₂")
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Sigma schedule
    axes[0, 1].semilogy(sorted_steps, mean_sigmas, "s-", color="purple")
    axes[0, 1].set_title("Noise Level σ Schedule")
    axes[0, 1].set_xlabel("Step index k")
    axes[0, 1].set_ylabel("σ")
    axes[0, 1].grid(True, alpha=0.3)

    # (1,0) Prediction delta
    if transition_deltas:
        sorted_trans = sorted(transition_deltas.keys())
        t_means = [np.mean(transition_deltas[t]) for t in sorted_trans]
        t_stds = [np.std(transition_deltas[t]) for t in sorted_trans]
        axes[1, 0].bar(range(len(sorted_trans)), t_means, yerr=t_stds, color="steelblue", alpha=0.8, capsize=4)
        axes[1, 0].set_title("Prediction Change ||Δx̂₀||₂ (k-1 → k)")
        axes[1, 0].set_xlabel("Step transition")
        axes[1, 0].set_ylabel("||x̂₀(k) - x̂₀(k-1)||₂")
        axes[1, 0].set_xticks(range(len(sorted_trans)))
        axes[1, 0].set_xticklabels([f"k={t-1}→{t}" for t in sorted_trans])
        axes[1, 0].grid(True, alpha=0.3)

    # (1,1) FFT low vs high power
    if step_fft_low:
        low_means_plot = [np.mean(step_fft_low[s]) for s in sorted_steps]
        high_means_plot = [np.mean(step_fft_high[s]) for s in sorted_steps]
        axes[1, 1].plot(sorted_steps, low_means_plot, "o-", color="royalblue", label="Low-freq")
        axes[1, 1].plot(sorted_steps, high_means_plot, "s-", color="tomato", label="High-freq")
        axes[1, 1].set_title("Spectral Power by Denoising Step")
        axes[1, 1].set_xlabel("Step index k")
        axes[1, 1].set_ylabel("FFT magnitude")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path_summary = out_dir / "summary_mechanism_analysis.png"
    plt.savefig(path_summary, dpi=150, bbox_inches="tight")
    plt.close()
    log_message(f"Saved: {path_summary}")


# ── メイン ────────────────────────────────────────────────────────────────

@dataclass
class MechAnalysisConfig(PolicyEvalConfig):
    num_analysis_episodes: int = 10
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_denoising"
    obj_instance_split: Optional[str] = None  # use all available objects (not held-out B-split)


def main():
    import draccus

    cfg: MechAnalysisConfig = draccus.parse(MechAnalysisConfig)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_message(f"=== Mechanism Analysis ===")
    log_message(f"Task: {cfg.task_name}")
    log_message(f"Episodes: {cfg.num_analysis_episodes}")
    log_message(f"Denoising steps: {cfg.num_denoising_steps_action}")

    # ---- Load model & data ----
    log_message("Loading model...")
    model, cosmos_config = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    log_message("Model loaded.")

    capture = DenoiseCapture()
    all_denoising_records: List[List[Dict]] = []  # per policy call
    success_count = 0
    total_episodes = cfg.num_analysis_episodes

    for ep_idx in range(total_episodes):
        log_message(f"\n--- Episode {ep_idx + 1}/{total_episodes} ---")
        env, _ = create_robocasa_env(cfg, seed=cfg.seed + ep_idx, episode_idx=ep_idx)
        obs = env.reset()

        # Wait for stabilization
        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)

        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        log_message(f"  Task description: {task_description!r}")

        action_queue = deque()
        success = False
        max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)

        for t in range(max_steps):
            observation = prepare_observation(obs, cfg.flip_images)

            if len(action_queue) == 0:
                try:
                    result = get_action_with_capture(
                        cfg=cfg,
                        model=model,
                        dataset_stats=dataset_stats,
                        observation=observation,
                        task_description=task_description,
                        capture=capture,
                        seed=cfg.seed + ep_idx + t,
                        num_denoising_steps=cfg.num_denoising_steps_action,
                    )
                    actions = result["actions"]
                    all_denoising_records.append(list(capture.step_records))
                    log_message(
                        f"  t={t}: {len(capture.step_records)} denoising steps captured"
                    )
                    for i in range(min(cfg.num_open_loop_steps, len(actions))):
                        action_queue.append(actions[i])
                except Exception as e:
                    log_message(f"  Policy call error at t={t}: {e}")
                    import traceback; traceback.print_exc()
                    break

            if action_queue:
                action = action_queue.popleft()
                if action.shape[-1] == 7 and env.action_dim == 12:
                    action = np.concatenate([action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                obs, _, _, _ = env.step(action)
                if env._check_success():
                    success = True
                    break

        env.close()
        if success:
            success_count += 1
        log_message(f"  Result: {'SUCCESS' if success else 'FAIL'}")

    success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
    log_message(f"\n=== Final Results ===")
    log_message(f"Task: {cfg.task_name}")
    log_message(f"Success: {success_count}/{total_episodes} = {success_rate:.1%}")
    log_message(f"Total policy calls captured: {len(all_denoising_records)}")

    # ---- Save light records ----
    light = [
        [{"sigma": r["sigma"], "noise_pred_norm": r["noise_pred_norm"]} for r in ep]
        for ep in all_denoising_records
    ]
    with open(out_dir / "denoising_light_records.json", "w") as f:
        json.dump(light, f, indent=2)

    # ---- Compute and save stats ----
    step_actions, step_norms, step_sigmas, step_fft_low, step_fft_high = compute_per_step_stats(
        all_denoising_records, cfg.chunk_size
    )
    transition_deltas = compute_prediction_deltas(all_denoising_records, cfg.chunk_size)

    sorted_steps = sorted(step_norms.keys())
    stats = {
        "task": cfg.task_name,
        "success_rate": success_rate,
        "success_count": success_count,
        "total_episodes": total_episodes,
        "total_policy_calls": len(all_denoising_records),
        "num_denoising_steps": cfg.num_denoising_steps_action,
        "per_step_score_norm_mean": {str(k): float(np.mean(step_norms[k])) for k in sorted_steps},
        "per_step_score_norm_std": {str(k): float(np.std(step_norms[k])) for k in sorted_steps},
        "per_step_sigma_mean": {str(k): float(np.mean(step_sigmas[k])) for k in sorted_steps},
        "per_step_fft_low_power": {str(k): float(np.mean(step_fft_low[k])) for k in sorted_steps if step_fft_low[k]},
        "per_step_fft_high_power": {str(k): float(np.mean(step_fft_high[k])) for k in sorted_steps if step_fft_high[k]},
        "prediction_change_mean": {
            str(k): float(np.mean(transition_deltas[k])) for k in sorted(transition_deltas.keys())
        },
        "prediction_change_std": {
            str(k): float(np.std(transition_deltas[k])) for k in sorted(transition_deltas.keys())
        },
    }
    with open(out_dir / "analysis_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log_message(f"Stats saved: {out_dir / 'analysis_stats.json'}")

    # ---- Plots ----
    if all_denoising_records:
        plot_all(all_denoising_records, out_dir, cfg.task_name, success_rate, cfg.chunk_size)

    log_message("Analysis complete!")
    return stats


if __name__ == "__main__":
    main()
