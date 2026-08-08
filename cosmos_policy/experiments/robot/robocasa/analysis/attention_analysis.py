"""
Cosmos Policy 自己注意（Self-Attention）解析スクリプト

【目的】
  Cosmos Policy DiT がアクション・画像・proprio・状態価値を生成する際、
  入力画像と入力 proprio のどの部分に注目しているかを定量化する。

  [分析 A] T 位置別注意行列 [11 × 11]
    - 出力トークン (T=5: action, T=6: future_proprio, T=7-9: future images, T=10: value)
      が入力トークン (T=1: proprio, T=2: wrist, T=3: primary, T=4: secondary) へ
      向ける平均注意重みを計算。
    - 複数 DiT ブロック × 複数デノイジングステップで集計。

  [分析 B] 空間的注意ヒートマップ [14 × 14]
    - 出力トークン種別ごとに、入力画像 (T=2,3,4) の
      どの空間領域に注目するかを可視化。

  [分析 C] Proprio vs Image 注意比率
    - 各出力ポジションが proprio (T=1) vs 画像 (T=2/3/4) に向ける
      注意割合を比較。

  [分析 D] Attention Rollout
    - 各プローブブロックまでの累積注意伝播行列 [11 × 11] を計算。
    - A_hat[b] = 0.5 * A[b] + 0.5 * I (残差接続考慮)
    - R[b] = A_hat[b] @ R[b_prev] (ブロック順に積算)
    - 入力 T 位置から出力 T 位置への情報フロー全体を可視化。

【アーキテクチャ】
  - 2B DiT: 28 ブロック, 16 heads, head_dim=128
  - patch_spatial=2 → 各 T 位置: 14×14 = 196 トークン
  - 全シーケンス長: S = 11 × 196 = 2156 トークン
  - latent sequence (state_t=11):
    T=0: blank       T=5: action       T=10: value
    T=1: proprio     T=6: future_proprio
    T=2: curr_wrist  T=7: future_wrist
    T=3: curr_primary  T=8: future_primary
    T=4: curr_secondary T=9: future_secondary

実行例（Singularity コンテナ内）:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.attention_analysis \\
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
      --num_attn_episodes 50 \\
      --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention
"""

import json
import os
from collections import defaultdict
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
# EGL must initialize before CUDA set_device to avoid permission conflicts.
import cosmos_policy.experiments.robot.cosmos_utils as _cosmos_utils

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS,  # dict: task_name -> max_steps
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.constants import ACTION_DIM

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    STATE_T,
    T_NAMES,
    INPUT_T_IDXS,
    OUTPUT_T_IDXS,
    IMAGE_INPUT_T_IDXS,
    IMAGE_OUTPUT_T_IDXS,
    PROBE_BLOCKS,
    NUM_DENOISE_STEPS,
    SIGMA_SCHEDULE,
    PATCHES_PER_T,
    SPATIAL_H,
    SPATIAL_W,
    TOTAL_SEQ_LEN,
)


# ── Attention Capture ─────────────────────────────────────────────────────────
class SelfAttentionCapture:
    """
    DiT ブロックの自己注意 Q, K をフックし、
    T 位置別注意行列 [11×11] と空間ヒートマップ [14×14] を収集する。

    patching strategy:
      Attention.compute_attention(q, k, v) の直前に q, k を横取りし、
      softmax(q @ k^T / sqrt(d)) を手動計算する。
      実際の forward pass には影響を与えない。
    """

    def __init__(self, probe_blocks: List[int]):
        self.probe_blocks = probe_blocks
        # Accumulated over all policy calls: sum and count for averaging
        # key: (block_idx, k_step) → np.array [11, 11]
        self._t_mat_sum: Dict[Tuple[int, int], np.ndarray] = {}
        self._t_mat_count: Dict[Tuple[int, int], int] = {}
        # key: (block_idx, k_step, t_out, t_in) → np.array [14, 14]
        self._spatial_sum: Dict[Tuple, np.ndarray] = {}
        self._spatial_count: Dict[Tuple, int] = {}

        # Per-block call counters for tracking denoising step
        self._block_call_cnt: Dict[int, int] = {}
        self._current_step = -1
        self._handles: List = []

    def reset_for_policy_call(self):
        self._block_call_cnt = {b: 0 for b in self.probe_blocks}
        self._current_step = -1

    def before_denoise_step(self):
        self._current_step += 1
        self._block_call_cnt = {b: 0 for b in self.probe_blocks}

    def install_hooks(self, model):
        """model.net.blocks[] の self_attn に compute_attention をパッチ"""
        net = model.net
        for block_idx in self.probe_blocks:
            block = net.blocks[block_idx]
            attn = block.self_attn
            self._patch_attn(block_idx, attn)

    def _patch_attn(self, block_idx: int, attn_module):
        original_fn = attn_module.compute_attention
        capture = self

        def patched_compute_attention(q, k, v, **kwargs):
            # Always call the real attention first
            result = original_fn(q, k, v, **kwargs)
            k_step = capture._current_step
            if k_step < 0:
                return result
            # q, k: [B, S, H_attn, D_head] in bshd format
            if q.dim() != 4 or q.shape[1] != TOTAL_SEQ_LEN:
                return result
            capture._compute_and_accumulate(block_idx, k_step, q, k)
            return result

        attn_module.compute_attention = patched_compute_attention

    @torch.no_grad()
    def _compute_and_accumulate(self, block_idx: int, k_step: int, q: torch.Tensor, k: torch.Tensor):
        """
        Q: [1, S, H, D]  S=2156, H=16, D=128
        K: [1, S, H, D]
        → T位置別注意行列 [11, 11] + 空間ヒートマップ [14, 14]
        """
        B, S, H, D = q.shape
        scale = float(D) ** -0.5
        n = PATCHES_PER_T  # 196

        # fp32 for stability
        q_f = q[0].float()  # [S, H, D]
        k_f = k[0].float()  # [S, H, D]

        t_mat = np.zeros((STATE_T, STATE_T), dtype=np.float32)

        for t_out in range(STATE_T):
            q_slice = q_f[t_out * n:(t_out + 1) * n]   # [196, H, D]
            # Compute scores against all keys: [H, 196, S]
            # q_slice: [196, H, D] → [H, 196, D]
            q_t = q_slice.permute(1, 0, 2)   # [H, 196, D]
            k_t = k_f.permute(1, 0, 2)        # [H, S, D]
            scores = torch.bmm(q_t, k_t.transpose(1, 2)) * scale  # [H, 196, S]
            attn = torch.softmax(scores, dim=-1)  # [H, 196, S]

            for t_in in range(STATE_T):
                block = attn[:, :, t_in * n:(t_in + 1) * n]  # [H, 196, 196]
                t_mat[t_out, t_in] = float(block.mean().item())

                # Spatial heatmaps for image input positions
                if t_in in IMAGE_INPUT_T_IDXS:
                    # Average over output tokens and heads: [196]
                    heatmap = block.mean(dim=(0, 1))  # [196]
                    heatmap_2d = heatmap.reshape(SPATIAL_H, SPATIAL_W).cpu().numpy()
                    key = (block_idx, k_step, t_out, t_in)
                    if key not in self._spatial_sum:
                        self._spatial_sum[key] = np.zeros((SPATIAL_H, SPATIAL_W), dtype=np.float32)
                        self._spatial_count[key] = 0
                    self._spatial_sum[key] += heatmap_2d
                    self._spatial_count[key] += 1

        key2 = (block_idx, k_step)
        if key2 not in self._t_mat_sum:
            self._t_mat_sum[key2] = np.zeros((STATE_T, STATE_T), dtype=np.float32)
            self._t_mat_count[key2] = 0
        self._t_mat_sum[key2] += t_mat
        self._t_mat_count[key2] += 1

    def get_averaged_t_matrices(self) -> Dict[Tuple[int, int], np.ndarray]:
        """(block_idx, k_step) → [11, 11] 平均注意行列"""
        result = {}
        for key, s in self._t_mat_sum.items():
            result[key] = s / max(self._t_mat_count[key], 1)
        return result

    def get_averaged_spatial_maps(self) -> Dict[Tuple, np.ndarray]:
        """(block_idx, k_step, t_out, t_in) → [14, 14] 平均ヒートマップ"""
        result = {}
        for key, s in self._spatial_sum.items():
            result[key] = s / max(self._spatial_count[key], 1)
        return result


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class AttentionAnalysisConfig(PolicyEvalConfig):
    num_attn_episodes: int = 50
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention"


# ── Wrapped inference ─────────────────────────────────────────────────────────
def get_action_with_attn_capture(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    attn_cap: SelfAttentionCapture,
    seed: int,
    num_denoising_steps: int,
):
    """注意フック有効化状態でアクションを生成"""
    attn_cap.reset_for_policy_call()
    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            def wrapped(x_t, sigma):
                attn_cap.before_denoise_step()
                x0_hat = x0_fn_raw(x_t, sigma)
                return x0_hat
            return wrapped, extra
        else:
            x0_fn_raw = result
            def wrapped(x_t, sigma):
                attn_cap.before_denoise_step()
                x0_hat = x0_fn_raw(x_t, sigma)
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


# ── Visualization ─────────────────────────────────────────────────────────────
T_LABELS = [T_NAMES[i] for i in range(STATE_T)]
T_SHORT = ["blank", "proprio", "wrist", "primary", "sec", "action",
           "f_prop", "f_wrist", "f_prim", "f_sec", "value"]


def plot_t_attention_matrices(t_matrices: Dict, output_dir: Path):
    """各ブロック・ステップの T 位置別注意行列をヒートマップとして保存"""
    block_list = sorted(set(k[0] for k in t_matrices))
    step_list = sorted(set(k[1] for k in t_matrices))

    for blk in block_list:
        fig, axes = plt.subplots(1, len(step_list), figsize=(5 * len(step_list), 5))
        if len(step_list) == 1:
            axes = [axes]
        for ax, k_step in zip(axes, step_list):
            mat = t_matrices.get((blk, k_step), np.zeros((STATE_T, STATE_T)))
            im = ax.imshow(mat, aspect="auto", cmap="viridis",
                           vmin=0, vmax=mat.max())
            ax.set_title(f"Block {blk}, σ={SIGMA_SCHEDULE[k_step]:.1f} (k={k_step})", fontsize=9)
            ax.set_xticks(range(STATE_T))
            ax.set_xticklabels(T_SHORT, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(STATE_T))
            ax.set_yticklabels(T_SHORT, fontsize=7)
            ax.set_xlabel("Key (input T)")
            ax.set_ylabel("Query (output T)")
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Self-Attention T-Position Matrix: Block {blk}", fontsize=11)
        plt.tight_layout()
        path = output_dir / f"t_attn_block{blk:02d}.png"
        plt.savefig(path, dpi=100)
        plt.close(fig)
        log_message(f"Saved {path}")


def plot_spatial_heatmaps(spatial_maps: Dict, output_dir: Path):
    """
    output token type ごとに、入力画像 (T=2,3,4) の空間注意ヒートマップを保存する。
    全 7 プローブブロック × 全 5 デノイジングステップ の組み合わせをカバーする。
    """
    t_outs_plot = [5, 8, 10]
    t_ins_plot = [2, 3, 4]
    steps_plot = list(range(NUM_DENOISE_STEPS))   # 全 5 ステップ (k=0〜4)
    blocks_plot = sorted(PROBE_BLOCKS)             # 全プローブブロック (7 層)

    for t_out in t_outs_plot:
        for t_in in t_ins_plot:
            n_blk = len(blocks_plot)
            n_step = len(steps_plot)
            fig, axes = plt.subplots(n_blk, n_step, figsize=(4 * n_step, 3.5 * n_blk))
            if n_blk == 1:
                axes = axes[np.newaxis, :]
            if n_step == 1:
                axes = axes[:, np.newaxis]

            for bi, blk in enumerate(blocks_plot):
                for si, k_step in enumerate(steps_plot):
                    ax = axes[bi, si]
                    key = (blk, k_step, t_out, t_in)
                    heatmap = spatial_maps.get(key)
                    if heatmap is not None:
                        im = ax.imshow(heatmap, cmap="hot", aspect="equal")
                        plt.colorbar(im, ax=ax, shrink=0.8)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes)
                    σ = SIGMA_SCHEDULE[k_step] if k_step < len(SIGMA_SCHEDULE) else "?"
                    ax.set_title(f"Blk{blk} σ={σ:.0f}", fontsize=8)
                    ax.set_xlabel("W patches")
                    ax.set_ylabel("H patches")

            out_name = T_NAMES[t_out]
            in_name = T_NAMES[t_in]
            fig.suptitle(
                f"Spatial Attention: [{out_name}] → [{in_name}]", fontsize=11
            )
            plt.tight_layout()
            path = output_dir / f"spatial_{out_name}_to_{in_name}.png"
            plt.savefig(path, dpi=100)
            plt.close(fig)
            log_message(f"Saved {path}")


def plot_proprio_vs_image_ratio(t_matrices: Dict, output_dir: Path):
    """
    各出力位置での proprio (T=1) vs 画像平均 (T=2+3+4 mean) への
    注意割合を、ブロック別・ステップ別に折れ線グラフで可視化。
    """
    block_list = sorted(set(k[0] for k in t_matrices))
    step_list = sorted(set(k[1] for k in t_matrices))
    t_outs_plot = [5, 6, 7, 8, 9, 10]

    for t_out in t_outs_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Left: vary by block (at k=4)
        ax = axes[0]
        k_step = min(4, max(step_list))
        proprio_vals = []
        image_vals = []
        for blk in block_list:
            mat = t_matrices.get((blk, k_step))
            if mat is None:
                proprio_vals.append(np.nan)
                image_vals.append(np.nan)
            else:
                proprio_vals.append(float(mat[t_out, 1]))
                image_vals.append(float(np.mean([mat[t_out, 2], mat[t_out, 3], mat[t_out, 4]])))
        ax.plot(block_list, proprio_vals, "b-o", label="proprio (T=1)")
        ax.plot(block_list, image_vals, "r-s", label="images (T=2-4 mean)")
        ax.set_xlabel("Block index")
        ax.set_ylabel("Mean attention weight")
        ax.set_title(f"{T_NAMES[t_out]} → input (k=4)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: vary by step (at block=13)
        ax = axes[1]
        blk = 13
        proprio_vals = []
        image_vals = []
        for k_step in step_list:
            mat = t_matrices.get((blk, k_step))
            if mat is None:
                proprio_vals.append(np.nan)
                image_vals.append(np.nan)
            else:
                proprio_vals.append(float(mat[t_out, 1]))
                image_vals.append(float(np.mean([mat[t_out, 2], mat[t_out, 3], mat[t_out, 4]])))
        sigmas = [SIGMA_SCHEDULE[s] for s in step_list]
        ax.plot(sigmas, proprio_vals, "b-o", label="proprio (T=1)")
        ax.plot(sigmas, image_vals, "r-s", label="images (T=2-4 mean)")
        ax.set_xlabel("σ (denoising step)")
        ax.set_xscale("log")
        ax.set_ylabel("Mean attention weight")
        ax.set_title(f"{T_NAMES[t_out]} → input (Block {blk})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Proprio vs Image Attention: output={T_NAMES[t_out]}", fontsize=11)
        plt.tight_layout()
        path = output_dir / f"proprio_vs_image_{T_NAMES[t_out]}.png"
        plt.savefig(path, dpi=100)
        plt.close(fig)
        log_message(f"Saved {path}")


def plot_cross_output_comparison(t_matrices: Dict, output_dir: Path):
    """
    複数の出力トークン種別が入力画像に向ける注意の強さを、ブロック深度の関数として比較。
    全デノイジングステップ (k=0〜4) について別ファイルで生成する。
    """
    block_list = sorted(set(k[0] for k in t_matrices))
    t_outs_compare = [5, 6, 7, 8, 9, 10]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]

    for k_step in range(NUM_DENOISE_STEPS):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, t_in in zip(axes, [2, 3, 4]):  # wrist, primary, secondary
            for t_out, color in zip(t_outs_compare, colors):
                vals = []
                for blk in block_list:
                    mat = t_matrices.get((blk, k_step))
                    vals.append(float(mat[t_out, t_in]) if mat is not None else np.nan)
                ax.plot(block_list, vals, "-o", color=color, label=T_NAMES[t_out], markersize=5)
            ax.set_xlabel("Block index")
            ax.set_ylabel("Mean attention weight")
            ax.set_title(f"→ {T_NAMES[t_in]} (σ={SIGMA_SCHEDULE[k_step]:.1f})")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Output-Token Attention to Image Inputs (k={k_step})", fontsize=11)
        plt.tight_layout()
        path = output_dir / f"cross_output_compare_k{k_step}.png"
        plt.savefig(path, dpi=100)
        plt.close(fig)
        log_message(f"Saved {path}")


# ── Attention Rollout ─────────────────────────────────────────────────────────
def compute_attention_rollout(
    t_matrices: Dict[Tuple[int, int], np.ndarray],
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Attention Rollout: 各プローブブロックまでの累積注意伝播行列 [11×11] を計算。

    残差接続を考慮した rollout (Abnar & Zuidema, 2020):
      A_hat[b] = 0.5 * A[b] + 0.5 * I   (residual 込みの実効注意)
      R[b]     = A_hat[b] @ R[b_prev]    (ブロック順に左から積算)
      R[-1]    = I                        (入力層は恒等変換)

    行正規化: A_hat の各行和を 1 に揃えてから積算する。

    返り値: (block_idx, k_step) → [11×11] rollout matrix
      - [i, j] = 出力 T=i が入力 T=j から受け取る累積的な情報量（0〜1）
    """
    block_list = sorted(set(k[0] for k in t_matrices))
    step_list = sorted(set(k[1] for k in t_matrices))
    rollout: Dict[Tuple[int, int], np.ndarray] = {}

    for k_step in step_list:
        R = np.eye(STATE_T, dtype=np.float32)
        for blk in block_list:
            A = t_matrices.get((blk, k_step))
            if A is None:
                continue
            A_hat = 0.5 * A + 0.5 * np.eye(STATE_T, dtype=np.float32)
            row_sums = A_hat.sum(axis=-1, keepdims=True)
            A_hat = A_hat / np.maximum(row_sums, 1e-8)
            R = A_hat @ R
            rollout[(blk, k_step)] = R.copy()

    return rollout


def plot_rollout_matrices(
    rollout: Dict[Tuple[int, int], np.ndarray],
    output_dir: Path,
):
    """
    各デノイジングステップについて、ブロック別の Attention Rollout 行列をプロット。
    各 PNG: 1行 × 7列（全 7 プローブブロック）の rollout ヒートマップ。
    """
    block_list = sorted(set(k[0] for k in rollout))
    step_list = sorted(set(k[1] for k in rollout))

    for k_step in step_list:
        fig, axes = plt.subplots(1, len(block_list), figsize=(5 * len(block_list), 5))
        if len(block_list) == 1:
            axes = [axes]
        for ax, blk in zip(axes, block_list):
            mat = rollout.get((blk, k_step), np.zeros((STATE_T, STATE_T)))
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=mat.max())
            ax.set_title(f"up to Block {blk}", fontsize=9)
            ax.set_xticks(range(STATE_T))
            ax.set_xticklabels(T_SHORT, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(STATE_T))
            ax.set_yticklabels(T_SHORT, fontsize=7)
            ax.set_xlabel("Source T (input)")
            ax.set_ylabel("Target T (output)")
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(
            f"Attention Rollout (σ={SIGMA_SCHEDULE[k_step]:.1f}, k={k_step}): "
            f"Cumulative information flow from input T to output T",
            fontsize=10,
        )
        plt.tight_layout()
        path = output_dir / f"rollout_k{k_step}.png"
        plt.savefig(path, dpi=100)
        plt.close(fig)
        log_message(f"Saved {path}")


def save_json_stats(
    t_matrices: Dict,
    rollout: Dict[Tuple[int, int], np.ndarray],
    output_dir: Path,
):
    """数値結果を JSON に保存"""
    block_list = sorted(set(k[0] for k in t_matrices))
    step_list = sorted(set(k[1] for k in t_matrices))

    stats = {
        "probe_blocks": block_list,
        "sigma_schedule": SIGMA_SCHEDULE,
        "t_names": T_NAMES,
        "t_matrices": {},
        "rollout_matrices": {},
        "input_attention_summary": {},
        "rollout_summary": {},
    }

    for (blk, k_step), mat in t_matrices.items():
        key = f"block{blk}_step{k_step}"
        stats["t_matrices"][key] = mat.tolist()

    for (blk, k_step), mat in rollout.items():
        key = f"block{blk}_step{k_step}"
        stats["rollout_matrices"][key] = mat.tolist()

    # Summary: per-output T, fraction of attention to each input type
    # averaged over blocks [9,13,18] (middle layers) and step k=4 (final)
    mid_blocks = [b for b in block_list if 9 <= b <= 22]
    final_step = max(step_list)
    for t_out in range(STATE_T):
        row = {}
        mats = [t_matrices[(b, final_step)] for b in mid_blocks if (b, final_step) in t_matrices]
        if mats:
            avg_mat = np.mean(mats, axis=0)
            for t_in in range(STATE_T):
                row[T_NAMES[t_in]] = float(avg_mat[t_out, t_in])
        stats["input_attention_summary"][T_NAMES[t_out]] = row

    # Rollout summary: final block (Block-27), k=4
    last_blk = max(block_list)
    r_mat = rollout.get((last_blk, final_step))
    if r_mat is not None:
        for t_out in range(STATE_T):
            row = {}
            for t_in in range(STATE_T):
                row[T_NAMES[t_in]] = float(r_mat[t_out, t_in])
            stats["rollout_summary"][T_NAMES[t_out]] = row

    path = output_dir / "attn_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    log_message(f"Saved {path}")

    # Summary table to console
    log_message("\n=== Input Attention Summary (mid blocks, k=4) ===")
    header = f"{'output':>14}" + "".join(f" {T_NAMES[t]:>10}" for t in INPUT_T_IDXS)
    log_message(header)
    log_message("-" * (14 + 11 * len(INPUT_T_IDXS)))
    for t_out in OUTPUT_T_IDXS:
        row = stats["input_attention_summary"].get(T_NAMES[t_out], {})
        vals = "".join(f" {row.get(T_NAMES[t], 0):.6f}" for t in INPUT_T_IDXS)
        log_message(f"{T_NAMES[t_out]:>14}{vals}")

    if r_mat is not None:
        log_message(f"\n=== Rollout Summary (Block-{last_blk}, k=4) ===")
        log_message(header)
        log_message("-" * (14 + 11 * len(INPUT_T_IDXS)))
        for t_out in OUTPUT_T_IDXS:
            row = stats["rollout_summary"].get(T_NAMES[t_out], {})
            vals = "".join(f" {row.get(T_NAMES[t], 0):.6f}" for t in INPUT_T_IDXS)
            log_message(f"{T_NAMES[t_out]:>14}{vals}")

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()

    # PolicyEvalConfig fields
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="cosmos_policy/config/config.py")
    parser.add_argument("--use_wrist_image", type=lambda x: x == "True", default=True)
    parser.add_argument("--num_wrist_images", type=int, default=1)
    parser.add_argument("--use_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--normalize_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--unnormalize_actions", type=lambda x: x == "True", default=True)
    parser.add_argument("--dataset_stats_path", type=str, default="")
    parser.add_argument("--t5_text_embeddings_path", type=str, default="")
    parser.add_argument("--trained_with_image_aug", type=lambda x: x == "True", default=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--num_open_loop_steps", type=int, default=16)
    parser.add_argument("--task_name", type=str, default="PnPCounterToCab")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--randomize_seed", type=lambda x: x == "True", default=False)
    parser.add_argument("--deterministic", type=lambda x: x == "True", default=True)
    parser.add_argument("--use_variance_scale", type=lambda x: x == "True", default=False)
    parser.add_argument("--use_jpeg_compression", type=lambda x: x == "True", default=True)
    parser.add_argument("--flip_images", type=lambda x: x == "True", default=True)
    parser.add_argument("--num_denoising_steps_action", type=int, default=5)
    parser.add_argument("--num_denoising_steps_future_state", type=int, default=1)
    parser.add_argument("--num_denoising_steps_value", type=int, default=1)
    parser.add_argument("--data_collection", type=lambda x: x == "True", default=False)
    parser.add_argument("--num_attn_episodes", type=int, default=50)
    parser.add_argument("--output_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Env first (EGL before CUDA) ──────────────────────────────────────────
    cfg = AttentionAnalysisConfig(
        config=args.config,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        use_wrist_image=args.use_wrist_image,
        num_wrist_images=args.num_wrist_images,
        use_proprio=args.use_proprio,
        normalize_proprio=args.normalize_proprio,
        unnormalize_actions=args.unnormalize_actions,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        trained_with_image_aug=args.trained_with_image_aug,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        task_name=args.task_name,
        seed=args.seed,
        randomize_seed=args.randomize_seed,
        deterministic=args.deterministic,
        use_variance_scale=args.use_variance_scale,
        use_jpeg_compression=args.use_jpeg_compression,
        flip_images=args.flip_images,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=args.num_denoising_steps_value,
        data_collection=args.data_collection,
        num_attn_episodes=args.num_attn_episodes,
        output_dir=args.output_dir,
    )

    log_message(f"Creating env for task: {cfg.task_name}")
    env, _ = create_robocasa_env(cfg)

    # ── GPU setup ────────────────────────────────────────────────────────────
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")

    _cosmos_utils.DEVICE = device
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cu.DEVICE = device

    # ── Model loading ────────────────────────────────────────────────────────
    log_message("Loading model...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=1)
    model, cosmos_config = get_model(cfg)
    model.eval()

    # ── Install attention hooks ───────────────────────────────────────────────
    log_message(f"Installing attention hooks on blocks: {PROBE_BLOCKS}")
    attn_cap = SelfAttentionCapture(probe_blocks=PROBE_BLOCKS)
    attn_cap.install_hooks(model)

    set_seed_everywhere(cfg.seed)

    from collections import deque

    # ── Episode loop ─────────────────────────────────────────────────────────
    episode_results = []
    total_policy_calls = 0
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)

    for ep in range(cfg.num_attn_episodes):
        log_message(f"\n{'='*60}")
        log_message(f"Episode {ep+1}/{cfg.num_attn_episodes}")
        obs = env.reset()
        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        step_count = 0
        done = False
        success = False
        action_queue = deque()

        while not done and step_count < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                result = get_action_with_attn_capture(
                    cfg=cfg,
                    model=model,
                    dataset_stats=dataset_stats,
                    observation=observation,
                    task_description=task_description,
                    attn_cap=attn_cap,
                    seed=cfg.seed,
                    num_denoising_steps=cfg.num_denoising_steps_action,
                )
                total_policy_calls += 1
                log_message(f"  Policy call {total_policy_calls} (ep {ep+1}, step {step_count})")

                actions = result["actions"]
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a_step = actions[i]
                    if a_step.shape[-1] == 7 and env.action_dim == 12:
                        a_step = np.concatenate([a_step, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    action_queue.append(a_step)

            action = action_queue.popleft()
            obs, reward, done, info = env.step(action)
            step_count += 1
            if done or (isinstance(info, dict) and info.get("success", False)):
                success = True
                done = True

        episode_results.append({"episode": ep, "success": success, "steps": step_count})
        log_message(f"Episode {ep+1}: success={success}, steps={step_count}")

    log_message(f"\nTotal policy calls: {total_policy_calls}")
    log_message(f"Success rate: {sum(r['success'] for r in episode_results)}/{cfg.num_attn_episodes}")

    # ── Aggregate and save ───────────────────────────────────────────────────
    log_message("\nAggregating results...")
    t_matrices = attn_cap.get_averaged_t_matrices()
    spatial_maps = attn_cap.get_averaged_spatial_maps()

    log_message(f"Captured T-matrices for {len(t_matrices)} (block, step) pairs")
    log_message(f"Captured spatial maps for {len(spatial_maps)} (block, step, t_out, t_in) tuples")

    # Compute Attention Rollout
    log_message("\nComputing Attention Rollout...")
    rollout = compute_attention_rollout(t_matrices)
    log_message(f"Computed rollout for {len(rollout)} (block, step) pairs")

    # Save raw data
    np.save(output_dir / "t_matrices.npy", {
        str(k): v for k, v in t_matrices.items()
    })
    np.save(output_dir / "spatial_maps.npy", {
        str(k): v for k, v in spatial_maps.items()
    })

    # Save JSON stats (includes rollout)
    stats = save_json_stats(t_matrices, rollout, output_dir)

    # Save meta
    meta = {
        "task_name": cfg.task_name,
        "n_episodes": cfg.num_attn_episodes,
        "total_policy_calls": total_policy_calls,
        "probe_blocks": PROBE_BLOCKS,
        "sigma_schedule": SIGMA_SCHEDULE,
        "state_t": STATE_T,
        "tokens_per_t": PATCHES_PER_T,
        "spatial_h": SPATIAL_H,
        "spatial_w": SPATIAL_W,
        "episode_results": episode_results,
    }
    with open(output_dir / "attn_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────────────
    log_message("\nGenerating plots...")
    plot_t_attention_matrices(t_matrices, output_dir)
    plot_spatial_heatmaps(spatial_maps, output_dir)
    plot_proprio_vs_image_ratio(t_matrices, output_dir)
    plot_cross_output_comparison(t_matrices, output_dir)
    plot_rollout_matrices(rollout, output_dir)

    log_message(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
