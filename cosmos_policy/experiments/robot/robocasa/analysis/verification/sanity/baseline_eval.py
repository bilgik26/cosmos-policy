"""
§1.2 成功率ゲート G1 — フックなしベースライン評価

目的:
  フックを一切使わない純粋なベースライン評価で成功率 p0 を取得。
  G2 (フックあり) との差を Fisher 検定で比較して G3 の合否を判定。

出力:
  results/baseline_eval/baseline_meta.json  — p0, episode 結果, G3 判定
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

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

import torch


@dataclass
class BaselineEvalConfig(PolicyEvalConfig):
    num_baseline_episodes: int = 50
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/baseline_eval"
    # G2 hook-run success rate for G3 comparison (optional, set from known results)
    p1_hook_run: float = 0.60   # feature_analysis run success rate (from prior run)
    n1_hook_run: int = 50


def run_g3_gate(p0: float, n0: int, p1: float, n1: int) -> dict:
    """
    G3: 2 群比率検定 (Fisher exact) で p0 vs p1 の有意差を検定。
    合否基準: 有意差なし (p>0.05) かつ |p0-p1| < 0.05
    """
    from scipy.stats import fisher_exact
    s0 = int(round(p0 * n0))
    f0 = n0 - s0
    s1 = int(round(p1 * n1))
    f1 = n1 - s1
    table = [[s0, f0], [s1, f1]]
    odds, p_val = fisher_exact(table, alternative="two-sided")

    abs_diff = abs(p0 - p1)
    passed = (p_val > 0.05) and (abs_diff < 0.05)

    return {
        "p0_baseline": p0,
        "n0": n0,
        "p1_hook": p1,
        "n1": n1,
        "abs_diff": abs_diff,
        "fisher_p": float(p_val),
        "fisher_odds": float(odds),
        "passed": passed,
        "interpretation": (
            f"|p0−p1|={abs_diff:.3f}, Fisher p={p_val:.3f}. "
            + ("PASS: フックは成功率に有意差なし。" if passed
               else "FAIL: フックが成功率に影響している可能性あり。")
        ),
    }


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
    parser.add_argument("--num_baseline_episodes", type=int, default=50)
    parser.add_argument("--output_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/baseline_eval")
    parser.add_argument("--p1_hook_run", type=float, default=0.60,
                        help="Success rate of the feature hook run (G2 reference)")
    parser.add_argument("--n1_hook_run", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = BaselineEvalConfig(
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
        num_baseline_episodes=args.num_baseline_episodes,
        output_dir=args.output_dir,
        p1_hook_run=args.p1_hook_run,
        n1_hook_run=args.n1_hook_run,
    )

    log_message(f"G1 Baseline Eval — task: {cfg.task_name}, n_episodes: {cfg.num_baseline_episodes}")
    log_message("Creating env (NO HOOKS — pure baseline)...")
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
    model, _ = get_model(cfg)
    model.eval()

    set_seed_everywhere(cfg.seed)

    episode_results = []
    total_calls = 0
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)

    for ep in range(cfg.num_baseline_episodes):
        log_message(f"\nEpisode {ep+1}/{cfg.num_baseline_episodes}")
        obs = env.reset()
        task_desc = env.get_ep_meta().get("lang", cfg.task_name)
        step_count = 0
        done = False
        success = False
        action_queue = deque()

        while not done and step_count < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                result = get_action(
                    cfg=cfg,
                    model=model,
                    dataset_stats=dataset_stats,
                    obs=observation,
                    task_label_or_embedding=task_desc,
                    seed=cfg.seed,
                    randomize_seed=False,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False,
                )
                total_calls += 1
                actions = result["actions"]
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    action_queue.append(a)

            action = action_queue.popleft()
            obs, reward, done, info = env.step(action)
            step_count += 1
            # ignore_done=True → done is always False; use _check_success() directly
            if env._check_success():
                success = True
                done = True

        episode_results.append({"episode": ep, "success": success, "steps": step_count})
        log_message(f"  Episode {ep+1}: success={success}, steps={step_count}")

    n_success = sum(r["success"] for r in episode_results)
    p0 = n_success / cfg.num_baseline_episodes
    log_message(f"\nG1 result: {n_success}/{cfg.num_baseline_episodes} = {p0:.1%}")

    # G3: compare with hook run
    g3 = run_g3_gate(p0=p0, n0=cfg.num_baseline_episodes,
                     p1=cfg.p1_hook_run, n1=cfg.n1_hook_run)
    log_message(f"\nG3 gate: {g3['interpretation']}")

    meta = {
        "gate": "G1_G3_success_rate",
        "hook_mode": "none",
        "task_name": cfg.task_name,
        "seed": cfg.seed,
        "n_episodes": cfg.num_baseline_episodes,
        "n_success": n_success,
        "p0_baseline": p0,
        "total_policy_calls": total_calls,
        "g3_gate": g3,
        "episode_results": episode_results,
    }
    with open(output_dir / "baseline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log_message(f"Saved: {output_dir / 'baseline_meta.json'}")


if __name__ == "__main__":
    main()
