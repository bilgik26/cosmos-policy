"""
FPO++ online RL training of Cosmos Policy on RoboCasa (Phase 2).

Structure mirrors manipulation_experiments/finetune_online_rl.py, adapted for:
  - Cosmos's EDM noise schedule (instead of pure flow matching)
  - RoboCasa multi-camera observations
  - 2B-parameter DiT with LoRA (or full DiT) fine-tuning
  - Action-chunk execution (chunk=32, open-loop=16)

Usage (inside Docker container):
    cd ~/cosmos-policy
    python -m cosmos_policy.experiments.robot.robocasa.train_fpo_robocasa \\
        --task_name TurnOffMicrowave \\
        --num_envs 4 \\
        --total_timesteps 100000 \\
        --lora_rank 8 \\
        --log_dir runs/fpo_TurnOffMicrowave

Value estimation
----------------
No separate Critic or ValueHead.  V(s) is read directly from the value token
(index 10) that the Cosmos DiT generates together with the action chunk, using
the same extraction + unnormalization as run_robocasa_eval.py.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from diffusers.optimization import get_scheduler

from cosmos_policy.experiments.robot.robocasa.fpo_buffer import RolloutBuffer

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Environment ──────────────────────────────────────────────────────
    task_name: str = "TurnOffMicrowave"
    num_envs: int = 4
    img_res: int = 224
    obj_instance_split: str = "B"

    # ── Model / Fine-tuning ───────────────────────────────────────────────
    ckpt_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
    dataset_stats_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json"
    t5_embeddings_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl"
    config_file: str = "cosmos_policy/config/config.py"
    cosmos_config: str = "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"
    finetune_mode: str = "lora"      # "lora" | "full_dit"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "q_proj,k_proj,v_proj,output_proj"

    # ── Action chunking ───────────────────────────────────────────────────
    chunk_size: int = 32
    n_open_loop: int = 16

    # ── Rollout ───────────────────────────────────────────────────────────
    steps_per_iter: int = 96
    n_cfm_samples: int = 16

    # ── GAE ───────────────────────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── PPO / FPO ─────────────────────────────────────────────────────────
    update_epochs: int = 4
    num_mini_batches: int = 4
    clip_coef: float = 0.01
    vf_coef: float = 0.5
    aux_coef: float = 1.0
    max_grad_norm: float = 1.0
    trust_region_mode: str = "ppo"   # "ppo" | "spo" | "aspo"

    # ── Optimiser ────────────────────────────────────────────────────────
    lr_lora: float = 1e-5
    adam_eps: float = 1e-5
    weight_decay: float = 0.0

    # ── LR scheduler ─────────────────────────────────────────────────────
    lr_scheduler_name: str = "constant"
    lr_scheduler_warmup_steps: int = 5

    # ── Training schedule ─────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    seed: int = 42

    # ── Eval ─────────────────────────────────────────────────────────────
    eval_rollout_freq: int = 10
    eval_num_episodes: int = 10

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_enable: bool = True
    wandb_project: str = "fpo-cosmos-robocasa"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # ── Logging ───────────────────────────────────────────────────────────
    log_dir: str = "runs/fpo_robocasa"
    save_interval: int = 10
    log_interval: int = 1


@dataclass
class _EvalCfg:
    """Thin config forwarded to CosmosFPOPolicy / data_batch builder."""
    suite: str = "robocasa"
    config: str = ""
    ckpt_path: str = ""
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_third_person_image: bool = True
    num_third_person_images: int = 2
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    chunk_size: int = 32
    num_open_loop_steps: int = 16
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# GAE
# ──────────────────────────────────────────────────────────────────────────────

def calculate_advantage(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation.  Returns (advantages, returns), both (T, B)."""
    T, B = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(B, dtype=np.float32)

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


# ──────────────────────────────────────────────────────────────────────────────
# FPO++ surrogate loss
# ──────────────────────────────────────────────────────────────────────────────

def fpo_surrogate_loss(
    old_cfm_loss: torch.Tensor,
    new_cfm_loss: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    trust_region_mode: str = "ppo",
) -> torch.Tensor:
    """FPO++ policy gradient loss.  ratio_i = exp(L_old_i − L_new_i) per sample i."""
    ratio = torch.exp(old_cfm_loss - new_cfm_loss)   # (B, N)
    adv   = advantages.expand_as(ratio)               # broadcast (B,1)→(B,N)

    if trust_region_mode == "ppo":
        loss = torch.max(
            -adv * ratio,
            -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
        ).mean()
    elif trust_region_mode == "spo":
        loss = -(adv * ratio - adv.abs() / (2.0 * clip_coef) * (ratio - 1.0) ** 2).mean()
    elif trust_region_mode == "aspo":
        ppo = torch.max(-adv * ratio,
                        -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef))
        spo = -(adv * ratio - adv.abs() / (2.0 * clip_coef) * (ratio - 1.0) ** 2)
        loss = torch.where(adv > 0, ppo, spo).mean()
    else:
        raise ValueError(f"Unknown trust_region_mode: {trust_region_mode!r}")

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(policy, envs, num_episodes: int) -> float:
    """Run evaluation episodes on the shared env and return success rate."""
    was_training = policy.model.training
    policy.model.eval()
    policy.reset_buffers()

    env_list = list(range(envs.num_envs))
    obs_list, _ = envs.reset()
    success_count = episode_count = 0

    while episode_count < num_episodes:
        actions_np, _, _, _ = policy.select_action(obs_list, env_indices=env_list)
        obs_list, _, dones, truncateds, infos = envs.step(actions_np)

        for i in np.where(dones | truncateds)[0]:
            if episode_count < num_episodes:
                success_count += int(infos[i].get("success", False))
                episode_count += 1

        done_indices = np.where(dones | truncateds)[0]
        if len(done_indices) > 0:
            policy.reset_buffers(env_indices=done_indices.tolist())

    if was_training:
        policy.model.train()
    return success_count / max(1, episode_count)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, iteration: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "iteration": iteration,
        "trainable_state_dict": {k: v for k, v in model.net.named_parameters()
                                 if v.requires_grad},
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(path: str, model, optimizer) -> int:
    ckpt = torch.load(path, weights_only=False)
    # Support legacy "lora_state_dict" key from earlier runs
    key = "trainable_state_dict" if "trainable_state_dict" in ckpt else "lora_state_dict"
    for name, param in model.net.named_parameters():
        if name in ckpt[key]:
            param.data.copy_(ckpt[key][name])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"]


# ──────────────────────────────────────────────────────────────────────────────
# Training sub-functions (extracted from train())
# ──────────────────────────────────────────────────────────────────────────────

def _build_eval_cfg(cfg: TrainConfig) -> _EvalCfg:
    return _EvalCfg(
        config=cfg.cosmos_config,
        ckpt_path=cfg.ckpt_path,
        config_file=cfg.config_file,
        dataset_stats_path=cfg.dataset_stats_path,
        t5_text_embeddings_path=cfg.t5_embeddings_path,
        chunk_size=cfg.chunk_size,
        num_open_loop_steps=cfg.n_open_loop,
    )


def _setup_model_and_policy(cfg: TrainConfig, eval_cfg: _EvalCfg):
    """Load Cosmos model, apply fine-tuning strategy, wrap in CosmosFPOPolicy."""
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, load_dataset_stats, init_t5_text_embeddings_cache,
    )
    from cosmos_policy.experiments.robot.cosmos_fpo_model import (
        apply_lora_to_cosmos, unfreeze_dit_for_finetuning, CosmosFPOPolicy,
    )

    print("[init] Loading Cosmos Policy …")
    init_t5_text_embeddings_cache(cfg.t5_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(eval_cfg)

    if cfg.finetune_mode == "lora":
        print("[init] Applying LoRA …")
        model = apply_lora_to_cosmos(
            model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets.split(","),
        )
    elif cfg.finetune_mode == "full_dit":
        print("[init] Unfreezing full DiT (VAE frozen) …")
        model = unfreeze_dit_for_finetuning(model)
    else:
        raise ValueError(f"Unknown finetune_mode: {cfg.finetune_mode!r}")

    policy = CosmosFPOPolicy(model, dataset_stats, eval_cfg,
                             n_cfm_samples=cfg.n_cfm_samples)
    return model, policy


def _setup_optimizer_and_scheduler(cfg: TrainConfig, model, num_iterations: int):
    """Create AdamW and diffusers LR scheduler for trainable DiT params."""
    trainable_params = [p for p in model.net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg.lr_lora,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler_warmup_steps,
        num_training_steps=num_iterations,
    )
    return trainable_params, optimizer, lr_scheduler


def _setup_envs(cfg: TrainConfig):
    from cosmos_policy.experiments.robot.robocasa.robocasa_gym_wrapper import (
        VectorizedRoboCasaEnv,
    )
    print(f"[init] Starting {cfg.num_envs} RoboCasa environments …")
    return VectorizedRoboCasaEnv(
        task_name=cfg.task_name,
        num_envs=cfg.num_envs,
        base_seed=cfg.seed,
        img_res=cfg.img_res,
        obj_instance_split=cfg.obj_instance_split,
    )


def _create_buffer(cfg: TrainConfig) -> RolloutBuffer:
    return RolloutBuffer(
        steps=cfg.steps_per_iter,
        num_envs=cfg.num_envs,
        n_cfm=cfg.n_cfm_samples,
        latent_c=16, latent_h=28, latent_w=28, latent_t=11,
    )


def _collect_rollout(
    cfg: TrainConfig,
    policy,
    envs,
    buf: RolloutBuffer,
    obs_list: list,
    total_steps: int,
    success_buffer: deque,
    ep_rew_buffer: deque,
    ep_len_buffer: deque,
    cur_ep_rew: np.ndarray,
    cur_ep_len: np.ndarray,
) -> Tuple[list, int]:
    """Collect cfg.steps_per_iter environment steps into buf.

    Returns updated (obs_list, total_steps).
    """
    policy.model.eval()
    env_list = list(range(cfg.num_envs))

    last_data_batch: Optional[dict]       = None
    last_old_loss:   Optional[np.ndarray] = None
    last_sigmas:     Optional[np.ndarray] = None
    last_epsilons:   Optional[np.ndarray] = None

    for step in range(cfg.steps_per_iter):
        actions_np, x0_new, cond_latent_new, data_batch_new = policy.select_action(
            obs_list, env_indices=env_list
        )

        full_chunk = x0_new is not None and x0_new.shape[0] == cfg.num_envs
        if full_chunk:
            old_loss, sigmas, epsilons = policy.compute_cfm_loss_for_storage(
                data_batch_new, x0_new, cond_latent_new
            )
            last_data_batch = data_batch_new
            last_old_loss   = old_loss.numpy()
            last_sigmas     = sigmas.numpy()
            last_epsilons   = epsilons.numpy()

            # Retroactively store actual future obs for the previous chunk.
            prev_step = step - cfg.n_open_loop
            if prev_step >= 0 and prev_step in buf.data_batches:
                if not buf.dones[prev_step:step].any():
                    buf.set_future_cond(prev_step, cond_latent_new.cpu().numpy())

        v = policy.get_env_values(env_list) or np.zeros(cfg.num_envs, dtype=np.float32)

        obs_list, rewards, dones, truncateds, infos = envs.step(actions_np)

        reset_envs = np.where(dones | truncateds)[0]
        if len(reset_envs) > 0:
            policy.reset_buffers(env_indices=reset_envs.tolist())
            for i in reset_envs:
                success_buffer.append(float(infos[i].get("success", False)))
                ep_rew_buffer.append(cur_ep_rew[i])
                ep_len_buffer.append(cur_ep_len[i])
                cur_ep_rew[i] = 0.0
                cur_ep_len[i] = 0.0

        cur_ep_rew += rewards
        cur_ep_len += 1.0
        total_steps += cfg.num_envs

        buf_x0   = policy.get_env_x0_latents(env_list, device="cpu") if full_chunk else None
        buf_cond = policy.get_env_cond_latents(env_list, device="cpu") if full_chunk else None
        buf.add(
            step_idx=step,
            rewards=rewards,
            dones=dones | truncateds,
            values=v,
            old_cfm_loss=last_old_loss if full_chunk else None,
            sigmas=last_sigmas        if full_chunk else None,
            epsilons=last_epsilons    if full_chunk else None,
            x0_latent=buf_x0,
            cond_latent=buf_cond,
            data_batch=last_data_batch if full_chunk else None,
        )

    return obs_list, total_steps


def _compute_normalized_gae(cfg: TrainConfig, policy, buf: RolloutBuffer):
    """Bootstrap V(s_T), run GAE, normalise advantages globally."""
    last_v = policy.get_env_values(list(range(cfg.num_envs)))
    if last_v is None:
        last_v = np.zeros(cfg.num_envs, dtype=np.float32)

    advantages, returns = calculate_advantage(
        buf.values, buf.rewards, buf.dones, last_v,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def _fpo_update(
    cfg: TrainConfig,
    policy,
    optimizer,
    buf: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    trainable_params: list,
) -> Dict[str, float]:
    """Run FPO++ update epochs.  Returns dict of mean losses."""
    policy.model.train()
    total = {"pg": 0.0, "vf": 0.0, "aux": 0.0, "ratio": 0.0}
    n_updates = 0

    for _ in range(cfg.update_epochs):
        for (adv_mb, ret_mb, old_loss_mb, sigmas_mb, eps_mb,
             x0_mb, cond_mb, future_cond_mb, db_mb) in buf.get_mini_batches(
                cfg.num_mini_batches, advantages, returns):

            adv_mb         = adv_mb.cuda()
            ret_mb         = ret_mb.cuda()
            old_loss_mb    = old_loss_mb.cuda()
            sigmas_mb      = sigmas_mb.cuda()
            eps_mb         = eps_mb.cuda()
            x0_mb          = x0_mb.cuda()
            cond_mb        = cond_mb.cuda()
            future_cond_mb = future_cond_mb.cuda()

            new_cfm_loss = policy.compute_cfm_loss(
                db_mb, x0_mb.detach(), cond_mb, sigmas_mb, eps_mb
            )
            pg_loss = fpo_surrogate_loss(
                old_cfm_loss=old_loss_mb,
                new_cfm_loss=new_cfm_loss,
                advantages=adv_mb,
                clip_coef=cfg.clip_coef,
                trust_region_mode=cfg.trust_region_mode,
            )

            v_pred, aux_loss = policy.compute_value_and_future_loss(
                db_mb, ret_mb, cond_mb, future_cond_mb
            )
            vf_loss = ((v_pred - ret_mb.squeeze(-1)) ** 2).mean()

            loss = pg_loss + cfg.vf_coef * vf_loss + cfg.aux_coef * aux_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                total["ratio"] += torch.exp(old_loss_mb - new_cfm_loss).mean().item()
            total["pg"]  += pg_loss.item()
            total["vf"]  += vf_loss.item()
            total["aux"] += aux_loss.item()
            n_updates += 1

    if n_updates > 0:
        for k in total:
            total[k] /= n_updates
    return total


def _log_iteration(
    cfg: TrainConfig,
    iteration: int,
    num_iterations: int,
    losses: Dict[str, float],
    optimizer,
    fps: float,
    iter_time: float,
    total_steps: int,
    success_buffer: deque,
    ep_rew_buffer: deque,
    ep_len_buffer: deque,
) -> None:
    log_dict = {
        "losses/policy":  losses["pg"],
        "losses/value":   losses["vf"],
        "losses/aux":     losses["aux"],
        "fpo/ratio":      losses["ratio"],
        "train/lr":       optimizer.param_groups[0]["lr"],
        "perf/fps":       fps,
        "perf/iter_time": iter_time,
    }
    if success_buffer:
        log_dict["train/success_rate"]   = float(np.mean(success_buffer))
        log_dict["train/mean_ep_reward"] = float(np.mean(ep_rew_buffer))
        log_dict["train/mean_ep_length"] = float(np.mean(ep_len_buffer))

    if cfg.wandb_enable:
        wandb.log(log_dict, step=total_steps)

    succ_str = f"succ={np.mean(success_buffer):.3f}" if success_buffer else "succ=n/a"
    print(
        f"Iter {iteration:4d}/{num_iterations}  steps={total_steps:,}  fps={fps:5.0f}  "
        f"pg={losses['pg']:.4f}  vf={losses['vf']:.4f}  aux={losses['aux']:.4f}  "
        f"ratio={losses['ratio']:.4f}  {succ_str}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)

    if cfg.wandb_enable:
        run_name = cfg.wandb_run_name or f"{cfg.task_name}_lora{cfg.lora_rank}"
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity or None,
                   name=run_name, config=vars(cfg), dir=cfg.log_dir)
        print(f"[wandb] Run: {wandb.run.name}  url: {wandb.run.url}")

    eval_cfg = _build_eval_cfg(cfg)
    model, policy = _setup_model_and_policy(cfg, eval_cfg)

    num_iterations = cfg.total_timesteps // (cfg.steps_per_iter * cfg.num_envs)
    trainable_params, optimizer, lr_scheduler = _setup_optimizer_and_scheduler(
        cfg, model, num_iterations
    )
    envs = _setup_envs(cfg)
    buf  = _create_buffer(cfg)

    obs_list, _ = envs.reset()
    policy.reset_buffers()
    total_steps = 0
    success_buffer = deque(maxlen=100)
    ep_rew_buffer  = deque(maxlen=100)
    ep_len_buffer  = deque(maxlen=100)
    cur_ep_rew     = np.zeros(cfg.num_envs, dtype=np.float32)
    cur_ep_len     = np.zeros(cfg.num_envs, dtype=np.float32)

    print(f"[train] Starting {num_iterations} iterations × "
          f"{cfg.steps_per_iter} steps × {cfg.num_envs} envs")

    for iteration in range(1, num_iterations + 1):
        t_start = time.perf_counter()

        obs_list, total_steps = _collect_rollout(
            cfg, policy, envs, buf, obs_list, total_steps,
            success_buffer, ep_rew_buffer, ep_len_buffer, cur_ep_rew, cur_ep_len,
        )
        advantages, returns = _compute_normalized_gae(cfg, policy, buf)
        losses = _fpo_update(cfg, policy, optimizer, buf, advantages, returns, trainable_params)

        lr_scheduler.step()
        iter_time = time.perf_counter() - t_start
        fps = cfg.steps_per_iter * cfg.num_envs / iter_time

        if iteration % cfg.log_interval == 0:
            _log_iteration(cfg, iteration, num_iterations, losses, optimizer,
                           fps, iter_time, total_steps,
                           success_buffer, ep_rew_buffer, ep_len_buffer)

        if iteration % cfg.save_interval == 0:
            save_checkpoint(os.path.join(cfg.log_dir, f"ckpt_{iteration:05d}.pt"),
                            model, optimizer, iteration)

        if (cfg.eval_rollout_freq > 0
                and (iteration % cfg.eval_rollout_freq == 0
                     or iteration == num_iterations)):
            print(f"[eval] Running {cfg.eval_num_episodes} episodes …")
            eval_sr = run_eval(policy, envs, cfg.eval_num_episodes)
            print(f"[eval] iter={iteration}  success_rate={eval_sr:.3f}")
            if cfg.wandb_enable:
                wandb.log({"eval/success_rate": eval_sr,
                           "eval/episodes": cfg.eval_num_episodes}, step=total_steps)
            obs_list, _ = envs.reset()
            policy.reset_buffers()

        buf.ptr = 0

    save_checkpoint(os.path.join(cfg.log_dir, "ckpt_final.pt"),
                    model, optimizer, iteration)
    envs.close()
    if cfg.wandb_enable:
        wandb.finish()
    print("Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="FPO++ Cosmos Policy on RoboCasa")
    cfg = TrainConfig()
    for f in cfg.__dataclass_fields__:
        default = getattr(cfg, f)
        p.add_argument(f"--{f}", type=type(default), default=default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    train(_parse_args())
