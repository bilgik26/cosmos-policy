"""
拡張ステップ解析 — レビュー §3.3 (PR崖の中間σ挿入) + §3.4 (J-b をアテンション/残差比に置換)

背景:
  §3.D の PR は k=0-3 でほぼ平坦 (55.8) → k=4 で崖 (18.4)。設計書 §3.D は
  「中間σを挿入して崖か滑らかな関数かを切り分ける」ことを要求していたが未実施だった。
  §3.J の J-b 指標 (‖feat(l+1)-feat(l)‖/‖feat(l)‖) は実際にはブロック間の
  ノルム成長 (Blk-0: ノルム≈16.7 → Blk-4: ノルム≈146) を測っているだけで、
  アテンションの寄与を分離できていない、という指摘がある。

本スクリプトは1回のロールアウトで両方を解決する:
  1. num_denoising_steps_action を 5 → 9 に増やし (同一 σ_max=80, σ_min=4 の
     Karras rho スケジュールがより密になる)、action-token 特徴量を全 probe
     block・全ステップで捕捉 → PR を σ の関数として高分解能で追跡し、
     k=3→4 の「崖」が σ=9.6→4.0 間の離散化アーティファクトなのか、
     それとも σ が十分細かくなっても残る急峻な遷移なのかを判定する。
  2. 各 block の self_attn モジュールに forward hook、Block 本体に
     forward_pre_hook を張り、action-token 位置での
     ‖attn_raw_output‖ / ‖residual_input‖ 比 (=J-b 代替指標: アテンション経路の
     残差ストリームへの相対寄与) を計算する。これはノルム成長そのものではなく
     「そのブロックで attention が残差にどれだけ寄与したか」を直接測る。

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.causal_gates.extended_step_effrank_jb \
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
      --config_file cosmos_policy/config/config.py \
      ... --num_denoising_steps_action 9 --num_analysis_episodes 20
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    PROBE_LAYERS, PATCHES_PER_T, effective_rank,
)

ACTION_T_IDX = 5


class ExtendedCapture:
    """block 出力特徴 (PR用) + self_attn 生出力/残差ノルム (J-b用) を同時捕捉。"""

    def __init__(self, probe_layers: List[int]):
        self.probe_layers = probe_layers
        self._current_step = -1
        self._handles = []
        # call_records[call_idx] = {"episode":.., "call_idx":.., "sigma_by_step": [...],
        #    "feat": {k: {layer: vec}}, "attn_ratio": {k: {layer: ratio}}}
        self.call_records: List[Dict] = []
        self._cur_feat: Dict = {}
        self._cur_attn_ratio: Dict = {}
        self._cur_residual_norm: Dict = {}  # temp storage per (layer) during current step

    def reset_for_policy_call(self):
        self._current_step = -1
        self._cur_feat = {}
        self._cur_attn_ratio = {}
        self._cur_residual_norm = {}

    def before_denoise_step(self):
        self._current_step += 1
        self._cur_feat.setdefault(self._current_step, {})
        self._cur_attn_ratio.setdefault(self._current_step, {})

    def finalize_policy_call(self, episode_idx: int, call_idx: int, sigma_schedule: list):
        self.call_records.append({
            "episode": episode_idx, "call_idx": call_idx, "sigma_schedule": sigma_schedule,
            "feat": {k: dict(v) for k, v in self._cur_feat.items()},
            "attn_ratio": {k: dict(v) for k, v in self._cur_attn_ratio.items()},
        })

    def register(self, model):
        net = model.net
        for layer_idx in self.probe_layers:
            block = net.blocks[layer_idx]
            h1 = block.register_forward_pre_hook(self._make_pre_hook(layer_idx))
            h2 = block.self_attn.register_forward_hook(self._make_attn_hook(layer_idx))
            h3 = block.register_forward_hook(self._make_block_out_hook(layer_idx))
            self._handles += [h1, h2, h3]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_pre_hook(self, layer_idx: int):
        def pre_hook(module, args):
            if self._current_step < 0:
                return
            x = args[0]
            if not (isinstance(x, torch.Tensor) and x.dim() == 5):
                return
            B, T, H, W, D = x.shape
            if T <= ACTION_T_IDX:
                return
            residual = x[0, ACTION_T_IDX].float().mean(dim=(0, 1))  # (D,)
            self._cur_residual_norm[layer_idx] = float(residual.norm().item())
        return pre_hook

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, inp, output):
            if self._current_step < 0:
                return
            out = output
            # self_attn's raw output is (B, S, D) with S = STATE_T * PATCHES_PER_T,
            # flattened over (t h w) -- it is rearranged back to (B,T,H,W,D) by the
            # *caller* (Block.forward), so the hook here always sees the 3D form.
            if not (isinstance(out, torch.Tensor) and out.dim() == 3):
                return
            B, S, D = out.shape
            start = ACTION_T_IDX * PATCHES_PER_T
            end = start + PATCHES_PER_T
            if S < end:
                return
            attn_out = out[0, start:end].float().mean(dim=0)  # (D,) -- mean over the 196 patches
            attn_norm = float(attn_out.norm().item())
            resid_norm = self._cur_residual_norm.get(layer_idx, None)
            if resid_norm is not None and resid_norm > 1e-8:
                ratio = attn_norm / resid_norm
                self._cur_attn_ratio.setdefault(self._current_step, {})[layer_idx] = ratio
        return hook

    def _make_block_out_hook(self, layer_idx: int):
        def hook(module, inp, output):
            if self._current_step < 0:
                return
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return
            B, T, H, W, D = output.shape
            if T <= ACTION_T_IDX:
                return
            feat = output[0, ACTION_T_IDX].float().mean(dim=(0, 1))
            self._cur_feat.setdefault(self._current_step, {})[layer_idx] = feat.detach().cpu().numpy()
        return hook


def get_action_with_capture(cfg, model, dataset_stats, observation, task_description,
                             capture: ExtendedCapture, seed: int, num_denoising_steps: int):
    capture.reset_for_policy_call()
    original_get_x0_fn = model.get_x0_fn_from_batch
    sigmas_seen = []

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result

            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                try:
                    sigmas_seen.append(float(sigma.flatten()[0].item()))
                except Exception:
                    pass
                return x0_fn_raw(x_t, sigma)

            return wrapped, extra
        else:
            x0_fn_raw = result

            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                try:
                    sigmas_seen.append(float(sigma.flatten()[0].item()))
                except Exception:
                    pass
                return x0_fn_raw(x_t, sigma)

            return wrapped

    model.get_x0_fn_from_batch = patched_get_x0_fn
    try:
        result = get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
            task_label_or_embedding=task_description, seed=seed, randomize_seed=False,
            num_denoising_steps_action=num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = original_get_x0_fn
    return result, sigmas_seen


@dataclass
class ExtStepConfig(PolicyEvalConfig):
    num_analysis_episodes: int = 20
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/extended_step_effrank_jb"


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
    parser.add_argument("--num_denoising_steps_action", type=int, default=9)
    parser.add_argument("--num_denoising_steps_future_state", type=int, default=1)
    parser.add_argument("--num_denoising_steps_value", type=int, default=1)
    parser.add_argument("--data_collection", type=lambda x: x == "True", default=False)
    parser.add_argument("--num_analysis_episodes", type=int, default=20)
    parser.add_argument("--output_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/extended_step_effrank_jb")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExtStepConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        use_wrist_image=args.use_wrist_image, num_wrist_images=args.num_wrist_images,
        use_proprio=args.use_proprio, normalize_proprio=args.normalize_proprio,
        unnormalize_actions=args.unnormalize_actions, dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        trained_with_image_aug=args.trained_with_image_aug, chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps, task_name=args.task_name, seed=args.seed,
        randomize_seed=args.randomize_seed, deterministic=args.deterministic,
        use_variance_scale=args.use_variance_scale, use_jpeg_compression=args.use_jpeg_compression,
        flip_images=args.flip_images, num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=args.num_denoising_steps_value, data_collection=args.data_collection,
        num_analysis_episodes=args.num_analysis_episodes, output_dir=args.output_dir,
    )

    log_message(f"Creating env for task: {cfg.task_name}")
    env, _ = create_robocasa_env(cfg)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cu.DEVICE = device

    log_message("Loading model...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=0)
    model, cosmos_config = get_model(cfg)
    model.eval()

    capture = ExtendedCapture(PROBE_LAYERS)
    capture.register(model)

    set_seed_everywhere(cfg.seed)
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    n_steps_action = cfg.num_denoising_steps_action

    total_calls = 0
    episode_results = []
    all_sigma_schedules = []

    for ep in range(cfg.num_analysis_episodes):
        log_message(f"\nEpisode {ep+1}/{cfg.num_analysis_episodes}")
        obs = env.reset()
        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        step_count = 0
        done = False
        success = False
        from collections import deque
        action_queue = deque()

        while not done and step_count < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                # IMPORTANT: vary the noise seed per call (matches feature_analysis.py's
                # seed=cfg.seed+ep_idx+t convention). A constant seed=cfg.seed here was a
                # real bug: it froze the initial diffusion noise x_T identically across
                # every single policy call in the run, collapsing the action-token feature
                # PR to ~2-4 regardless of episode/scene diversity (confirmed via a clean
                # frozen-vs-varying replay test on identical observations: mid-layer PR
                # jumped from ~2-4 to ~50-54 when only this seed was varied).
                call_seed = cfg.seed + total_calls
                result, sigmas = get_action_with_capture(
                    cfg=cfg, model=model, dataset_stats=dataset_stats,
                    observation=observation, task_description=task_description,
                    capture=capture, seed=call_seed, num_denoising_steps=n_steps_action,
                )
                capture.finalize_policy_call(ep, total_calls, sigmas)
                all_sigma_schedules.append(sigmas)
                total_calls += 1
                if total_calls % 10 == 0:
                    log_message(f"  Policy call {total_calls}")

                actions = result["actions"]
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a_step = actions[i]
                    if a_step.shape[-1] == 7 and env.action_dim == 12:
                        a_step = np.concatenate([a_step, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    action_queue.append(a_step)

            action = action_queue.popleft()
            obs, reward, done, info = env.step(action)
            step_count += 1
            if env._check_success():
                success = True
                done = True

        episode_results.append({"episode": ep, "success": success, "steps": step_count})
        log_message(f"Episode {ep+1}: success={success}, steps={step_count}")

    n_success = sum(r["success"] for r in episode_results)
    success_rate = n_success / cfg.num_analysis_episodes
    log_message(f"\nTotal policy calls: {total_calls}, success_rate={success_rate:.1%}")

    # ── Aggregate PR per (k, layer), using the *consensus* sigma at each step index ──
    # (sigma should be identical across calls for a fixed num_steps schedule; verify + record.)
    sigma_arr = np.array(all_sigma_schedules)  # (n_calls, n_steps) if consistent lengths
    sigma_consistent = len(set(tuple(round(s, 4) for s in row) for row in all_sigma_schedules)) == 1
    sigma_schedule_used = all_sigma_schedules[0] if all_sigma_schedules else []

    pr_by_layer_k = {}
    attn_ratio_by_layer_k = {}
    for layer in PROBE_LAYERS:
        pr_by_layer_k[layer] = []
        attn_ratio_by_layer_k[layer] = []
        for k in range(n_steps_action):
            feats = [rec["feat"].get(k, {}).get(layer) for rec in capture.call_records]
            feats = np.stack([f for f in feats if f is not None]) if feats else np.zeros((0, 1))
            if feats.shape[0] >= 5:
                _, sv, _ = np.linalg.svd(feats - feats.mean(axis=0), full_matrices=False)
                pr = effective_rank(sv)
            else:
                pr = None
            pr_by_layer_k[layer].append(pr)

            ratios = [rec["attn_ratio"].get(k, {}).get(layer) for rec in capture.call_records]
            ratios = [r for r in ratios if r is not None]
            attn_ratio_by_layer_k[layer].append(
                {"mean": float(np.mean(ratios)), "std": float(np.std(ratios)), "n": len(ratios)}
                if ratios else None
            )

    out = {
        "task_name": cfg.task_name, "n_episodes": cfg.num_analysis_episodes,
        "total_policy_calls": total_calls, "success_rate": success_rate,
        "num_denoising_steps_action": n_steps_action,
        "sigma_schedule_consistent_across_calls": sigma_consistent,
        "sigma_schedule": sigma_schedule_used,
        "probe_layers": PROBE_LAYERS,
        "pr_by_layer_k": pr_by_layer_k,
        "attn_output_over_residual_ratio_by_layer_k": attn_ratio_by_layer_k,
    }
    with open(output_dir / "extended_step_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log_message(f"Saved: {output_dir / 'extended_step_results.json'}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for layer in PROBE_LAYERS:
        prs = pr_by_layer_k[layer]
        axes[0].plot(sigma_schedule_used, prs, marker="o", label=f"Blk-{layer}")
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("σ (log scale, decreasing →)")
    axes[0].set_ylabel("Participation Ratio")
    axes[0].set_title(f"PR vs σ at {n_steps_action}-step resolution (N={cfg.num_analysis_episodes}ep)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for layer in PROBE_LAYERS:
        vals = [r["mean"] if r else np.nan for r in attn_ratio_by_layer_k[layer]]
        axes[1].plot(sigma_schedule_used, vals, marker="o", label=f"Blk-{layer}")
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("σ (log scale, decreasing →)")
    axes[1].set_ylabel("‖attn_raw_output‖ / ‖residual_input‖")
    axes[1].set_title("J-b replacement: attention contribution to residual stream")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "extended_step_pr_and_jb.png", dpi=150)
    plt.close()
    log_message(f"Saved: {output_dir / 'extended_step_pr_and_jb.png'}")

    capture.remove()
    log_message("\nExtended-step analysis complete.")


if __name__ == "__main__":
    main()
