"""
§3.C F_θ Analysis — EDM preconditioning: F_θ vs D_θ vs score norms

設計書 §3.C の要求:
  測る対象を3つ分離:
    1. F_θ (前処理前の生ネット出力) ノルム
    2. D_θ = x̂₀ (後処理済み) ノルム
    3. スコア s = (x_t − D_θ) / σ

  ‖·‖/√d を報告し、CV を計算して「入力依存性テスト」を行う。
  CV ≈ 0 は preconditioningの強力さではなく「測度集中」であることを式で示す。

F_θ 計算:
  EDMScaling (sigma_data=0.5) では:
    c_skip = σ_d² / (σ² + σ_d²)
    c_out  = σ × σ_d / √(σ² + σ_d²)
  ゆえに F_θ = (D_θ − c_skip × x_t) / c_out

入力: simulator run (50 episodes)
出力: results/precond_ftheta/
  - precond_norms_Ftheta_Dtheta_score.png
  - precond_normalized_by_sqrt_d.json
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

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

SIGMA_DATA = 0.5  # EDMScaling default (text2world_model.py line 97)
SIGMA_SCHEDULE = [80.0, 42.29, 20.97, 9.62, 4.0]


def edm_scaling(sigma: float) -> tuple:
    """Compute c_skip, c_out for EDMScaling(sigma_data=0.5)."""
    sd = SIGMA_DATA
    c_skip = sd**2 / (sigma**2 + sd**2)
    c_out = sigma * sd / (sigma**2 + sd**2) ** 0.5
    return c_skip, c_out


class FThetaCapture:
    """Wraps x0_fn to capture F_θ, D_θ, x_t, σ norms per denoising step."""

    def __init__(self):
        self.records: List[Dict] = []

    def reset(self):
        self.records = []

    def wrap_x0_fn(self, x0_fn):
        cap = self

        def wrapped(x_t, sigma):
            x0 = x0_fn(x_t, sigma)  # D_θ
            sigma_val = float(sigma.float().mean().item())

            c_skip, c_out = edm_scaling(sigma_val)

            # F_θ = (D_θ − c_skip × x_t) / c_out
            f_theta = (x0.float() - c_skip * x_t.float()) / (c_out + 1e-12)

            # score s = (x_t − D_θ) / σ
            score = (x_t.float() - x0.float()) / (sigma_val + 1e-8)

            d = x0.numel()
            sqrt_d = d ** 0.5

            cap.records.append({
                "sigma": sigma_val,
                "c_skip": c_skip,
                "c_out": c_out,
                "f_theta_norm": float(f_theta.norm().item()),
                "dtheta_norm": float(x0.float().norm().item()),
                "xt_norm": float(x_t.float().norm().item()),
                "score_norm": float(score.norm().item()),
                "f_theta_norm_normed": float(f_theta.norm().item()) / sqrt_d,
                "dtheta_norm_normed": float(x0.float().norm().item()) / sqrt_d,
                "xt_norm_normed": float(x_t.float().norm().item()) / sqrt_d,
                "score_norm_normed": float(score.norm().item()) / sqrt_d,
                "sqrt_d": sqrt_d,
                "d": d,
            })
            return x0

        return wrapped


@dataclass
class PrecondConfig(PolicyEvalConfig):
    num_episodes: int = 50
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/precond_ftheta"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--config_file", default="cosmos_policy/config/config.py")
    parser.add_argument("--use_wrist_image", type=lambda x: x == "True", default=True)
    parser.add_argument("--num_wrist_images", type=int, default=1)
    parser.add_argument("--use_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--normalize_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--unnormalize_actions", type=lambda x: x == "True", default=True)
    parser.add_argument("--dataset_stats_path", default="")
    parser.add_argument("--t5_text_embeddings_path", default="")
    parser.add_argument("--trained_with_image_aug", type=lambda x: x == "True", default=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--num_open_loop_steps", type=int, default=16)
    parser.add_argument("--task_name", default="PnPCounterToCab")
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
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--output_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/precond_ftheta")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PrecondConfig(
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
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    _cosmos_utils.DEVICE = device
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cu.DEVICE = device

    log_message("=== §3.C F_θ Preconditioning Analysis ===")
    log_message(f"sigma_data = {SIGMA_DATA}, EDMScaling")

    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=0)
    model, _ = get_model(cfg)
    model.eval()

    env, _ = create_robocasa_env(cfg)
    set_seed_everywhere(cfg.seed)

    cap = FThetaCapture()
    all_records: List[List[Dict]] = []  # per policy call: list of 5 step records
    episode_results = []
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    total_calls = 0

    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        cap.reset()
        if isinstance(result, tuple):
            x0_fn_raw, extra = result
            return cap.wrap_x0_fn(x0_fn_raw), extra
        else:
            return cap.wrap_x0_fn(result)

    model.get_x0_fn_from_batch = patched_get_x0_fn

    try:
        for ep in range(cfg.num_episodes):
            log_message(f"\nEpisode {ep+1}/{cfg.num_episodes}")
            obs = env.reset()
            task_desc = env.get_ep_meta().get("lang", cfg.task_name)
            action_queue = deque()
            step_count = 0
            done = False
            success = False

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
                    if cap.records:
                        all_records.append(list(cap.records))
                        cap.reset()

                    actions = result["actions"]
                    for i in range(min(cfg.num_open_loop_steps, len(actions))):
                        a = actions[i]
                        if a.shape[-1] == 7 and env.action_dim == 12:
                            a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                        action_queue.append(a)

                action = action_queue.popleft()
                obs, reward, done, info = env.step(action)
                step_count += 1
                if env._check_success():
                    success = True
                    done = True

            episode_results.append({"episode": ep, "success": success, "steps": step_count})
            log_message(f"  Episode {ep+1}: success={success}, steps={step_count}")
    finally:
        model.get_x0_fn_from_batch = original_get_x0_fn

    n_success = sum(r["success"] for r in episode_results)
    p_success = n_success / cfg.num_episodes
    log_message(f"\nSuccess rate: {n_success}/{cfg.num_episodes} = {p_success:.1%}")
    log_message(f"Total policy calls: {total_calls}")
    log_message(f"Calls with F_θ captured: {len(all_records)}")

    # Aggregate per sigma-step
    per_sigma: Dict[str, Dict[str, list]] = {}
    for sigma_val in SIGMA_SCHEDULE:
        key = f"sigma_{sigma_val:.2f}"
        per_sigma[key] = {
            "f_theta_norm": [], "dtheta_norm": [], "xt_norm": [], "score_norm": [],
            "f_theta_norm_normed": [], "dtheta_norm_normed": [], "score_norm_normed": [],
        }

    for call_records in all_records:
        if len(call_records) != 5:
            continue
        for rec in call_records:
            sigma_val = rec["sigma"]
            # Match to nearest schedule value
            nearest = min(SIGMA_SCHEDULE, key=lambda s: abs(s - sigma_val))
            key = f"sigma_{nearest:.2f}"
            for field in ["f_theta_norm", "dtheta_norm", "xt_norm", "score_norm",
                          "f_theta_norm_normed", "dtheta_norm_normed", "score_norm_normed"]:
                per_sigma[key][field].append(rec[field])

    # Summary stats
    sqrt_d_ref = all_records[0][0]["sqrt_d"] if all_records else 1.0
    d_ref = all_records[0][0]["d"] if all_records else 1

    summary = {
        "section": "3C_ftheta_preconditioning",
        "sigma_data": SIGMA_DATA,
        "sqrt_d": sqrt_d_ref,
        "d": d_ref,
        "n_calls": len(all_records),
        "success_rate": p_success,
        "n_sigma_steps": 5,
        "sigma_schedule": SIGMA_SCHEDULE,
        "per_sigma": {},
    }

    log_message("\n§3.C F_θ Results:")
    log_message(f"  √d = {sqrt_d_ref:.1f}, d = {d_ref}")
    log_message(f"  {'σ':>8} | {'‖F_θ‖/√d':>10} | {'CV%':>6} | {'‖D_θ‖/√d':>10} | {'CV%':>6} | {'‖score‖/√d':>10}")
    log_message(f"  {'-'*70}")

    for sigma_val in SIGMA_SCHEDULE:
        key = f"sigma_{sigma_val:.2f}"
        d_s = per_sigma[key]
        fn = np.array(d_s["f_theta_norm_normed"])
        dn = np.array(d_s["dtheta_norm_normed"])
        sn = np.array(d_s["score_norm_normed"])
        fn_cv = float(fn.std() / fn.mean() * 100) if len(fn) > 1 and fn.mean() > 0 else 0.0
        dn_cv = float(dn.std() / dn.mean() * 100) if len(dn) > 1 and dn.mean() > 0 else 0.0

        summary["per_sigma"][key] = {
            "sigma": sigma_val,
            "c_skip": edm_scaling(sigma_val)[0],
            "c_out": edm_scaling(sigma_val)[1],
            "n": len(fn),
            "f_theta_norm_normed_mean": float(fn.mean()),
            "f_theta_norm_normed_std": float(fn.std()),
            "f_theta_cv_pct": fn_cv,
            "dtheta_norm_normed_mean": float(dn.mean()),
            "dtheta_norm_normed_std": float(dn.std()),
            "dtheta_cv_pct": dn_cv,
            "score_norm_normed_mean": float(sn.mean()),
        }
        log_message(f"  σ={sigma_val:>7.2f} | {fn.mean():>10.4f} | {fn_cv:>6.3f} | {dn.mean():>10.4f} | {dn_cv:>6.3f} | {sn.mean():>10.4f}")

    # Measure concentration: for Gaussian F_θ ~ N(0, I_d), ‖F_θ‖/√d → 1
    # CV ≈ 1/(2√d) for Gaussian (Berry-Esseen bound)
    gaussian_cv_pct = 100.0 / (2.0 * sqrt_d_ref)
    summary["gaussian_concentration_cv_pct"] = gaussian_cv_pct
    log_message(f"\n  Gaussian concentration bound: CV ≈ 1/(2√d) = {gaussian_cv_pct:.4f}%")
    log_message("  (If measured CV ≈ this bound, D_θ/F_θ variance is measurement-concentration, not architecture)")

    with open(out_dir / "precond_normalized_by_sqrt_d.json", "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"\nSaved: {out_dir}/precond_normalized_by_sqrt_d.json")

    _plot_precond(summary, out_dir)


def _plot_precond(summary: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = summary["sigma_schedule"]
    per_sigma = summary["per_sigma"]
    sqrt_d = summary["sqrt_d"]
    gaussian_cv = summary["gaussian_concentration_cv_pct"]

    fn_means = [per_sigma[f"sigma_{s:.2f}"]["f_theta_norm_normed_mean"] for s in sigmas]
    fn_stds  = [per_sigma[f"sigma_{s:.2f}"]["f_theta_norm_normed_std"] for s in sigmas]
    fn_cvs   = [per_sigma[f"sigma_{s:.2f}"]["f_theta_cv_pct"] for s in sigmas]
    dn_means = [per_sigma[f"sigma_{s:.2f}"]["dtheta_norm_normed_mean"] for s in sigmas]
    dn_cvs   = [per_sigma[f"sigma_{s:.2f}"]["dtheta_cv_pct"] for s in sigmas]
    sn_means = [per_sigma[f"sigma_{s:.2f}"]["score_norm_normed_mean"] for s in sigmas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = range(len(sigmas))
    xlabels = [f"k={k}\nσ={s:.1f}" for k, s in enumerate(sigmas)]

    # Panel 1: ‖F_θ‖/√d and ‖D_θ‖/√d
    axes[0].errorbar(x, fn_means, yerr=fn_stds, marker='o', label='‖F_θ‖/√d', color='royalblue', capsize=4)
    axes[0].plot(x, dn_means, marker='s', label='‖D_θ‖/√d', color='tomato')
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Gaussian N(0,I)')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(xlabels, fontsize=8)
    axes[0].set_ylabel("Norm / √d")
    axes[0].set_title("§3.C: ‖F_θ‖ vs ‖D_θ‖ (normed by √d)\nσ_data=0.5, EDMScaling")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: CV%
    axes[1].bar([xi - 0.2 for xi in x], fn_cvs, width=0.4, label='F_θ CV%', color='royalblue', alpha=0.8)
    axes[1].bar([xi + 0.2 for xi in x], dn_cvs, width=0.4, label='D_θ CV%', color='tomato', alpha=0.8)
    axes[1].axhline(gaussian_cv, color='green', linestyle='--',
                    label=f'Gaussian bound 1/(2√d)={gaussian_cv:.3f}%')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(xlabels, fontsize=8)
    axes[1].set_ylabel("CV (%)")
    axes[1].set_title("§3.C: CV% per σ\n(≈Gaussian bound → concentration, not architecture)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # Panel 3: score norm
    axes[2].plot(x, sn_means, marker='o', color='purple', label='‖score‖/√d = ‖(x_t-D_θ)/σ‖/√d')
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(xlabels, fontsize=8)
    axes[2].set_ylabel("Score norm / √d")
    axes[2].set_title("§3.C: Score norm (D_θ diverges from x_t as σ→0)")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"§3.C Preconditioning: F_θ / D_θ / Score  |  √d={sqrt_d:.0f}  |  N_calls={summary['n_calls']}",
                 fontsize=11)
    plt.tight_layout()
    out = out_dir / "precond_norms_Ftheta_Dtheta_score.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
