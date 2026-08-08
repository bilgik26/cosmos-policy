"""
Cosmos Policy 自己注意解析 v2 — §2.1 T1 / T2 / T3 + フック修正版

v1 からの変更点 (設計書 §2.1 P1 準拠):
  [BUG FIX] _compute_and_accumulate で Q/K を CUDA のまま bmm → VRAM 過消費 → 0% 成功率
    修正: q[0].detach().cpu().float() で CPU 転送してから計算
  [T1] フックあり/なし出力一致テスト (episode ループ前に実行)
  [T2] 手計算 softmax vs SDPA 一致テスト (オフライン)
  追加: capture 内に try-except → キャプチャ失敗でもポリシーを止めない

実行例:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.attention.attention_analysis_v2 \\
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \\
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \\
      --config_file cosmos_policy/config/config.py \\
      ...  (run_attention_analysis_v2.sh を参照)
"""

import hashlib
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

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

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
    STATE_T, T_NAMES, INPUT_T_IDXS, OUTPUT_T_IDXS,
    IMAGE_INPUT_T_IDXS, IMAGE_OUTPUT_T_IDXS,
    PROBE_BLOCKS, NUM_DENOISE_STEPS, SIGMA_SCHEDULE,
    PATCHES_PER_T, SPATIAL_H, SPATIAL_W, TOTAL_SEQ_LEN,
)

# Re-export plot/util functions from v1 to avoid duplication
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attention.attention_analysis import (
    plot_t_attention_matrices,
    plot_spatial_heatmaps,
    plot_proprio_vs_image_ratio,
    plot_cross_output_comparison,
    compute_attention_rollout,
    plot_rollout_matrices,
    save_json_stats,
)

T_SHORT = ["blank", "proprio", "wrist", "primary", "sec", "action",
           "f_prop", "f_wrist", "f_prim", "f_sec", "value"]


# ── T2: softmax sanity check (offline, no model needed) ──────────────────────

def run_t2_softmax_sanity(seed: int = 0) -> dict:
    """
    T2: 手計算 softmax(QK^T/√d) vs SDPA 一致テスト。
    対象: 短系列・pad なし・因果マスクなし の小ケース。
    合否基準: 最大絶対差 < 1e-4 (bf16 相当の丸め誤差)
    """
    log_message("\n=== T2: softmax sanity (manual vs SDPA) ===")
    torch.manual_seed(seed)
    B, H, S, D = 1, 4, 16, 32  # 小規模ケース
    scale = D ** -0.5
    dtype = torch.float32

    q = torch.randn(B, H, S, D, dtype=dtype)
    k = torch.randn(B, H, S, D, dtype=dtype)
    v = torch.randn(B, H, S, D, dtype=dtype)

    # 手計算
    attn_manual = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    out_manual = attn_manual @ v  # [B, H, S, D]

    # SDPA
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
        )

    max_diff = float((out_manual - out_sdpa).abs().max().item())
    passed = max_diff < 1e-4
    result = {
        "test": "T2_softmax_sanity",
        "max_abs_diff": max_diff,
        "threshold": 1e-4,
        "passed": passed,
        "shape": {"B": B, "H": H, "S": S, "D": D},
    }
    status = "PASS" if passed else "FAIL"
    log_message(f"  max|manual - SDPA| = {max_diff:.2e}  [{status}]")
    return result


# ── Attention Capture v2 (CPU-based, observe-only) ───────────────────────────

class SelfAttentionCaptureV2:
    """
    v1 との差分:
      - _compute_and_accumulate: Q/K を CPU に転送してから bmm/softmax → VRAM 消費ゼロ
      - try-except: キャプチャ失敗でもポリシー出力に影響なし
    """

    def __init__(self, probe_blocks: List[int]):
        self.probe_blocks = probe_blocks
        self._t_mat_sum: Dict[Tuple[int, int], np.ndarray] = {}
        self._t_mat_count: Dict[Tuple[int, int], int] = {}
        self._spatial_sum: Dict[Tuple, np.ndarray] = {}
        self._spatial_count: Dict[Tuple, int] = {}
        self._capture_errors: int = 0
        self._current_step = -1
        self._current_episode = -1
        self._handles: List = []
        # episode毎の内訳（成功epのみでの再平均化に必須。全体平均のみだと
        # 事後的な成功epフィルタが不可能になるため、call毎にepisode_idxで
        # 分けて別途保持する）。
        self._t_mat_sum_ep: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
        self._t_mat_count_ep: Dict[int, Dict[Tuple[int, int], int]] = {}

    def set_episode(self, episode_idx: int):
        self._current_episode = episode_idx

    def reset_for_policy_call(self):
        self._current_step = -1

    def before_denoise_step(self):
        self._current_step += 1

    def install_hooks(self, model):
        """model.net.blocks[].self_attn.compute_attention をパッチ (v1 互換)"""
        net = model.net
        for block_idx in self.probe_blocks:
            block = net.blocks[block_idx]
            attn = block.self_attn
            self._patch_attn(block_idx, attn)

    def _patch_attn(self, block_idx: int, attn_module):
        original_fn = attn_module.compute_attention
        capture = self

        def patched_compute_attention(q, k, v, **kwargs):
            result = original_fn(q, k, v, **kwargs)  # always call real attention FIRST
            k_step = capture._current_step
            if k_step < 0:
                return result
            if q.dim() != 4 or q.shape[1] != TOTAL_SEQ_LEN:
                return result
            capture._compute_and_accumulate(block_idx, k_step, q, k)
            return result

        attn_module.compute_attention = patched_compute_attention

    @torch.no_grad()
    def _compute_and_accumulate(self, block_idx: int, k_step: int,
                                 q: torch.Tensor, k: torch.Tensor):
        """
        [FIX v2] Q/K を CPU 転送してから attention 計算 → VRAM 消費なし。
        v1 では q[0].float() (CUDA のまま) → VRAM 過消費 → 0% 成功率。
        """
        try:
            # CPU 転送 (KEY FIX: was q[0].float() which kept tensors on CUDA)
            q_f = q[0].detach().cpu().float()  # [S, H, D]
            k_f = k[0].detach().cpu().float()  # [S, H, D]

            S, H, D = q_f.shape
            scale = float(D) ** -0.5
            n = PATCHES_PER_T  # 196

            t_mat = np.zeros((STATE_T, STATE_T), dtype=np.float32)

            for t_out in range(STATE_T):
                q_slice = q_f[t_out * n:(t_out + 1) * n]   # [196, H, D]
                q_t = q_slice.permute(1, 0, 2)              # [H, 196, D]
                k_t = k_f.permute(1, 0, 2)                  # [H, S, D]
                scores = torch.bmm(q_t, k_t.transpose(1, 2)) * scale  # [H, 196, S] on CPU
                attn = torch.softmax(scores, dim=-1)         # [H, 196, S] on CPU

                for t_in in range(STATE_T):
                    block = attn[:, :, t_in * n:(t_in + 1) * n]  # [H, 196, 196]
                    t_mat[t_out, t_in] = float(block.mean().item())

                    if t_in in IMAGE_INPUT_T_IDXS:
                        heatmap = block.mean(dim=(0, 1))  # [196]
                        heatmap_2d = heatmap.reshape(SPATIAL_H, SPATIAL_W).numpy()
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

            ep = self._current_episode
            ep_sum = self._t_mat_sum_ep.setdefault(ep, {})
            ep_cnt = self._t_mat_count_ep.setdefault(ep, {})
            if key2 not in ep_sum:
                ep_sum[key2] = np.zeros((STATE_T, STATE_T), dtype=np.float32)
                ep_cnt[key2] = 0
            ep_sum[key2] += t_mat
            ep_cnt[key2] += 1

        except Exception as e:
            # Never let capture errors affect the policy
            self._capture_errors += 1

    def get_averaged_t_matrices(self) -> Dict[Tuple[int, int], np.ndarray]:
        return {k: s / max(self._t_mat_count[k], 1) for k, s in self._t_mat_sum.items()}

    def get_averaged_spatial_maps(self) -> Dict[Tuple, np.ndarray]:
        return {k: s / max(self._spatial_count[k], 1) for k, s in self._spatial_sum.items()}

    def get_per_episode_t_matrices(self) -> Dict[int, Dict[Tuple[int, int], np.ndarray]]:
        """episode毎の平均T-matrix（成功epフィルタ後の再平均化用）。"""
        out: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
        for ep, sums in self._t_mat_sum_ep.items():
            cnts = self._t_mat_count_ep[ep]
            out[ep] = {k: s / max(cnts[k], 1) for k, s in sums.items()}
        return out

    def get_averaged_t_matrices_filtered(
        self, episode_success: Dict[int, bool]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """成功epのみでの平均T-matrix（episode単位で均等加重: 各epの平均を
        さらにepisode間で平均する。call数の多いepisodeに引きずられないよう、
        まずcall平均→episode平均の2段階にする）。"""
        per_ep = self.get_per_episode_t_matrices()
        succ_eps = [e for e, ok in episode_success.items() if ok and e in per_ep]
        if not succ_eps:
            return {}
        all_keys = set()
        for e in succ_eps:
            all_keys.update(per_ep[e].keys())
        result: Dict[Tuple[int, int], np.ndarray] = {}
        for key in all_keys:
            mats = [per_ep[e][key] for e in succ_eps if key in per_ep[e]]
            if mats:
                result[key] = np.mean(np.stack(mats, axis=0), axis=0)
        return result


# ── T1: hook invariance test ──────────────────────────────────────────────────

def run_t1_hook_invariance_test(
    cfg, model, dataset_stats, env, attn_cap: SelfAttentionCaptureV2,
    seed: int, num_denoising_steps: int, output_dir: Path,
) -> dict:
    """
    T1: フックあり/なしで x̂₀ (生成アクション) が一致するかテスト。
    合否基準: max|Δ| < 1e-4 (bf16 丸め相当)
    注意: 実行には env の reset が必要。
    """
    log_message("\n=== T1: Hook Invariance Test ===")

    # 固定入力を得るため env を一度 reset
    set_seed_everywhere(seed)
    obs = env.reset()
    task_desc = env.get_ep_meta().get("lang", cfg.task_name)
    observation = prepare_observation(obs, cfg.flip_images)

    # --- Run WITHOUT hooks (use regular get_action) ---
    set_seed_everywhere(seed)
    with torch.no_grad():
        result_no_hook = get_action(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=observation,
            task_label_or_embedding=task_desc,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    actions_no_hook = np.array(result_no_hook["actions"])  # (T, 7)
    fp_no_hook = hashlib.sha256(actions_no_hook.tobytes()).hexdigest()[:16]

    # --- Run WITH hooks ---
    set_seed_everywhere(seed)
    attn_cap.reset_for_policy_call()

    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn_t1(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            def wrapped(x_t, sigma):
                attn_cap.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
            return wrapped, extra
        else:
            x0_fn_raw = result
            def wrapped(x_t, sigma):
                attn_cap.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
            return wrapped

    model.get_x0_fn_from_batch = patched_get_x0_fn_t1
    try:
        with torch.no_grad():
            result_with_hook = get_action(
                cfg=cfg,
                model=model,
                dataset_stats=dataset_stats,
                obs=observation,
                task_label_or_embedding=task_desc,
                seed=seed,
                randomize_seed=False,
                num_denoising_steps_action=num_denoising_steps,
                generate_future_state_and_value_in_parallel=False,
            )
    finally:
        model.get_x0_fn_from_batch = original_get_x0_fn

    actions_with_hook = np.array(result_with_hook["actions"])
    fp_with_hook = hashlib.sha256(actions_with_hook.tobytes()).hexdigest()[:16]

    # --- Compare ---
    diff = np.abs(actions_with_hook - actions_no_hook)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    passed = max_diff < 1e-4

    result = {
        "test": "T1_hook_invariance",
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "threshold": 1e-4,
        "passed": passed,
        "fingerprint_no_hook": fp_no_hook,
        "fingerprint_with_hook": fp_with_hook,
        "fingerprints_match": fp_no_hook == fp_with_hook,
        "capture_errors_during_test": attn_cap._capture_errors,
    }
    status = "PASS" if passed else "FAIL"
    log_message(f"  max|Δ_action| = {max_diff:.2e}  (threshold=1e-4)  [{status}]")
    log_message(f"  mean|Δ_action| = {mean_diff:.2e}")
    log_message(f"  Fingerprints: no_hook={fp_no_hook}  with_hook={fp_with_hook}")
    if not passed:
        log_message(f"  [FAIL] Hook is modifying policy output! max|Δ|={max_diff:.2e} >= 1e-4")
        log_message(f"  Capture errors during test: {attn_cap._capture_errors}")
    return result


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AttentionAnalysisV2Config(PolicyEvalConfig):
    num_attn_episodes: int = 50
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention_v2"
    skip_t1_on_fail: bool = False  # if True, continue even if T1 fails


# ── Wrapped inference ─────────────────────────────────────────────────────────

def get_action_with_attn_capture_v2(
    cfg, model, dataset_stats, observation, task_description,
    attn_cap: SelfAttentionCaptureV2, seed: int, num_denoising_steps: int,
):
    """フック有効化状態でアクション生成 (v2: CPU capture)"""
    attn_cap.reset_for_policy_call()
    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            def wrapped(x_t, sigma):
                attn_cap.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
            return wrapped, extra
        else:
            x0_fn_raw = result
            def wrapped(x_t, sigma):
                attn_cap.before_denoise_step()
                return x0_fn_raw(x_t, sigma)
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--skip_t1_on_fail", type=lambda x: x == "True", default=False)
    parser.add_argument("--output_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/self_attention_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = AttentionAnalysisV2Config(
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
        skip_t1_on_fail=args.skip_t1_on_fail,
    )

    log_message(f"Creating env for task: {cfg.task_name}")
    env, _ = create_robocasa_env(cfg)

    # With CUDA_VISIBLE_DEVICES=1, physical GPU 1 is remapped to cuda:0.
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    _cosmos_utils.DEVICE = device
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cu.DEVICE = device

    log_message("Loading model...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=0)
    model, cosmos_config = get_model(cfg)
    model.eval()

    # ── T2 (offline, before hook installation) ───────────────────────────────
    t2_result = run_t2_softmax_sanity(seed=cfg.seed)
    with open(output_dir / "t2_softmax_sanity.json", "w") as f:
        json.dump(t2_result, f, indent=2)
    if not t2_result["passed"]:
        log_message("[WARN] T2 FAILED — softmax implementation mismatch.")

    # ── Install hooks (v2: CPU-based capture) ────────────────────────────────
    log_message(f"\nInstalling v2 hooks (CPU capture) on blocks: {PROBE_BLOCKS}")
    attn_cap = SelfAttentionCaptureV2(probe_blocks=PROBE_BLOCKS)
    attn_cap.install_hooks(model)

    # ── T1 (hook invariance, with fixed env observation) ─────────────────────
    t1_result = run_t1_hook_invariance_test(
        cfg=cfg, model=model, dataset_stats=dataset_stats,
        env=env, attn_cap=attn_cap,
        seed=cfg.seed, num_denoising_steps=cfg.num_denoising_steps_action,
        output_dir=output_dir,
    )
    with open(output_dir / "t1_hook_invariance.json", "w") as f:
        json.dump(t1_result, f, indent=2)

    if not t1_result["passed"] and not cfg.skip_t1_on_fail:
        log_message("[ABORT] T1 FAILED — hook is modifying outputs. "
                    "Use --skip_t1_on_fail True to proceed anyway.")
        return

    set_seed_everywhere(cfg.seed)

    # ── Episode loop ─────────────────────────────────────────────────────────
    episode_results = []
    total_policy_calls = 0
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)

    for ep in range(cfg.num_attn_episodes):
        log_message(f"\n{'='*60}")
        log_message(f"Episode {ep+1}/{cfg.num_attn_episodes}")
        attn_cap.set_episode(ep)
        obs = env.reset()
        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        step_count = 0
        done = False
        success = False
        action_queue = deque()

        while not done and step_count < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                result = get_action_with_attn_capture_v2(
                    cfg=cfg, model=model, dataset_stats=dataset_stats,
                    observation=observation, task_description=task_description,
                    attn_cap=attn_cap, seed=cfg.seed + ep * 1000 + step_count,
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
            # ignore_done=True → done is always False; use _check_success() directly
            if env._check_success():
                success = True
                done = True

        episode_results.append({"episode": ep, "success": success, "steps": step_count})
        log_message(f"Episode {ep+1}: success={success}, steps={step_count}")

    n_success = sum(r["success"] for r in episode_results)
    success_rate = n_success / cfg.num_attn_episodes
    log_message(f"\nTotal policy calls: {total_policy_calls}")
    log_message(f"Success rate: {n_success}/{cfg.num_attn_episodes} = {success_rate:.1%}")
    log_message(f"Capture errors: {attn_cap._capture_errors}")

    # ── Aggregate and save ───────────────────────────────────────────────────
    t_matrices = attn_cap.get_averaged_t_matrices()
    spatial_maps = attn_cap.get_averaged_spatial_maps()
    log_message(f"Captured T-matrices for {len(t_matrices)} (block, step) pairs")

    rollout = compute_attention_rollout(t_matrices)
    stats = save_json_stats(t_matrices, rollout, output_dir)

    # episode毎のT-matrix（成功epのみでの再平均化・再解析用）
    per_ep = attn_cap.get_per_episode_t_matrices()
    per_ep_npz = {}
    for ep_idx, mats in per_ep.items():
        for (block_idx, k_step), mat in mats.items():
            per_ep_npz[f"tmat_ep{ep_idx}_block{block_idx}_k{k_step}"] = mat
    per_ep_npz["episode_indices"] = np.array(sorted(per_ep.keys()))
    ep_success_index = np.array([r["episode"] for r in episode_results])
    ep_success_flag = np.array([bool(r["success"]) for r in episode_results])
    per_ep_npz["episode_success_index"] = ep_success_index
    per_ep_npz["episode_success_flag"] = ep_success_flag
    np.savez_compressed(output_dir / "per_episode_t_matrices.npz", **per_ep_npz)
    log_message(f"Saved: {output_dir / 'per_episode_t_matrices.npz'}")

    meta = {
        "task_name": cfg.task_name,
        "hook_mode": "observe_only_cpu",
        "n_episodes": cfg.num_attn_episodes,
        "total_policy_calls": total_policy_calls,
        "success_rate": success_rate,
        "capture_errors": attn_cap._capture_errors,
        "probe_blocks": PROBE_BLOCKS,
        "sigma_schedule": SIGMA_SCHEDULE,
        "state_t": STATE_T,
        "tokens_per_t": PATCHES_PER_T,
        "spatial_h": SPATIAL_H,
        "spatial_w": SPATIAL_W,
        "t1_passed": t1_result["passed"],
        "t1_max_diff": t1_result["max_abs_diff"],
        "t2_passed": t2_result["passed"],
        "episode_results": episode_results,
    }
    with open(output_dir / "attn_meta_v2.json", "w") as f:
        json.dump(meta, f, indent=2)

    log_message("\nGenerating plots...")
    plot_t_attention_matrices(t_matrices, output_dir)
    plot_spatial_heatmaps(spatial_maps, output_dir)
    plot_proprio_vs_image_ratio(t_matrices, output_dir)
    plot_cross_output_comparison(t_matrices, output_dir)
    plot_rollout_matrices(rollout, output_dir)

    log_message(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
