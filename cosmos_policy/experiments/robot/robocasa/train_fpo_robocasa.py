"""
FPO++ online RL training of Cosmos Policy on RoboCasa (Phase 2).

Structure mirrors manipulation_experiments/finetune_online_rl.py, adapted for:
  - Cosmos's EDM noise schedule (instead of pure flow matching)
  - RoboCasa multi-camera observations
  - 2B-parameter DiT with LoRA (not full fine-tune)
  - Action-chunk execution (chunk=32, open-loop=16)

Usage (inside Docker container):
    cd ~/cosmos-policy
    python -m cosmos_policy.experiments.robot.robocasa.train_fpo_robocasa \\
        --task_name TurnOffMicrowave \\
        --num_envs 4 \\
        --total_timesteps 100000 \\
        --lora_rank 8 \\
        --log_dir runs/fpo_TurnOffMicrowave

Phase 3 extension point
-----------------------
Replace the scalar Critic with a flow-based value head (Value Flows approach):
  1. Remove Critic from cosmos_fpo_model.py.
  2. Add a value latent decoder that reads the value token (index 10) from the
     generated full latent and maps it to a scalar via a learned linear head.
  3. This lets the value prediction benefit from the world model's context.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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

    # ── Model / LoRA ──────────────────────────────────────────────────────
    ckpt_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
    dataset_stats_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json"
    t5_embeddings_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl"
    config_file: str = "cosmos_policy/config/config.py"
    cosmos_config: str = "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "q_proj,k_proj,v_proj,output_proj"

    # ── Action chunking ───────────────────────────────────────────────────
    chunk_size: int = 32
    n_open_loop: int = 16          # actions executed per chunk before re-query

    # ── Rollout ───────────────────────────────────────────────────────────
    steps_per_iter: int = 96       # environment steps collected per iteration
    n_cfm_samples: int = 16        # (σ, ε) samples per step for FPO ratio

    # ── GAE ───────────────────────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── PPO / FPO ─────────────────────────────────────────────────────────
    update_epochs: int = 4
    num_mini_batches: int = 4
    clip_coef: float = 0.01        # PPO clip ε
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    trust_region_mode: str = "ppo" # "ppo" | "spo" | "aspo"

    # ── Optimiser ────────────────────────────────────────────────────────
    lr_lora: float = 1e-5
    lr_critic: float = 1e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.0

    # ── Training schedule ─────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    seed: int = 42

    # ── Logging ───────────────────────────────────────────────────────────
    log_dir: str = "runs/fpo_robocasa"
    save_interval: int = 10        # save checkpoint every N iterations
    log_interval: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# Inline eval config (passed to CosmosFPOPolicy / data_batch builder)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _EvalCfg:
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
    values: np.ndarray,      # (T, B)
    rewards: np.ndarray,     # (T, B)
    dones: np.ndarray,       # (T, B)
    last_value: np.ndarray,  # (B,)
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation.

    Returns:
        advantages: (T, B)
        returns:    (T, B)
    """
    T, B = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(B, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        next_non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ──────────────────────────────────────────────────────────────────────────────
# FPO++ surrogate loss
# ──────────────────────────────────────────────────────────────────────────────

def fpo_surrogate_loss(
    old_cfm_loss: torch.Tensor,   # (B, N)
    new_cfm_loss: torch.Tensor,   # (B, N)
    advantages: torch.Tensor,     # (B, 1)
    clip_coef: float,
    trust_region_mode: str = "ppo",
) -> torch.Tensor:
    """
    FPO++ policy gradient loss.

    ratio_i = exp(L_old_i - L_new_i)   per CFM sample i
    Then the N ratios are averaged for the PPO / SPO objective.
    """
    log_ratio = old_cfm_loss - new_cfm_loss          # (B, N)
    ratio = torch.exp(log_ratio)                      # (B, N)

    # Broadcast advantages: (B, 1) → (B, N)
    adv = advantages.expand_as(ratio)

    if trust_region_mode == "ppo":
        s1 = -adv * ratio
        s2 = -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        loss = torch.max(s1, s2).mean()
    elif trust_region_mode == "spo":
        loss = -(adv * ratio - adv.abs() / (2.0 * clip_coef) * (ratio - 1.0) ** 2).mean()
    elif trust_region_mode == "aspo":
        ppo = torch.max(-adv * ratio,
                        -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef))
        spo = -(adv * ratio - adv.abs() / (2.0 * clip_coef) * (ratio - 1.0) ** 2)
        loss = torch.where(adv > 0, ppo, spo).mean()
    else:
        raise ValueError(f"Unknown trust_region_mode: {trust_region_mode}")

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Rollout storage
# ──────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one iteration's worth of transitions for FPO++ update."""

    def __init__(self, steps: int, num_envs: int, n_cfm: int,
                 latent_c: int, latent_h: int, latent_w: int, latent_t: int):
        self.steps = steps
        self.B = num_envs
        self.N = n_cfm
        C, H, W, T = latent_c, latent_h, latent_w, latent_t

        # RL scalars
        self.rewards   = np.zeros((steps, num_envs), dtype=np.float32)
        self.dones     = np.zeros((steps, num_envs), dtype=bool)
        self.values    = np.zeros((steps, num_envs), dtype=np.float32)

        # FPO++ tensors
        self.old_cfm_loss = np.zeros((steps, num_envs, n_cfm), dtype=np.float32)
        self.sigmas       = np.zeros((steps, num_envs, n_cfm), dtype=np.float32)
        self.epsilons     = np.zeros((steps, num_envs, n_cfm, C, H, W), dtype=np.float32)
        self.x0_latent    = np.zeros((steps, num_envs, C, T, H, W), dtype=np.float32)

        # Conditional latent (full 11-frame clean latent from rollout) — per step
        self.cond_latent = np.zeros((steps, num_envs, C, T, H, W), dtype=np.float32)

        # data_batch is stored as a list (one per chunk boundary, not per step)
        self.data_batches: Dict[int, dict] = {}

        self.ptr = 0

    def add(
        self,
        step_idx: int,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        old_cfm_loss: Optional[np.ndarray],
        sigmas: Optional[np.ndarray],
        epsilons: Optional[np.ndarray],
        x0_latent: Optional[torch.Tensor],
        cond_latent: Optional[torch.Tensor],
        data_batch: Optional[dict],
    ):
        self.rewards[step_idx] = rewards
        self.dones[step_idx] = dones
        self.values[step_idx] = values

        if old_cfm_loss is not None:
            self.old_cfm_loss[step_idx] = old_cfm_loss
            self.sigmas[step_idx] = sigmas
            self.epsilons[step_idx] = epsilons
            self.x0_latent[step_idx] = x0_latent.cpu().numpy()
            self.cond_latent[step_idx] = cond_latent.cpu().numpy()  # full (B, C, T, H, W)
            self.data_batches[step_idx] = data_batch

    def get_mini_batches(self, num_mini_batches: int, advantages: np.ndarray, returns: np.ndarray):
        """Yield (advantages, returns, old_cfm_loss, sigmas, epsilons, x0_latent, cond_latent, data_batch)."""
        # Collect steps that have CFM data (chunk boundaries)
        chunk_steps = sorted(self.data_batches.keys())
        if not chunk_steps:
            return

        indices = np.random.permutation(len(chunk_steps))
        mb_size = max(1, len(indices) // num_mini_batches)

        for start in range(0, len(indices), mb_size):
            batch_steps = [chunk_steps[i] for i in indices[start: start + mb_size]]

            # Concatenate across (time_steps × envs) in this mini-batch.
            # advantages[s] shape: (num_envs,) → reshape to (num_envs, 1)
            adv_mb    = torch.tensor(
                np.concatenate([advantages[s].reshape(-1, 1) for s in batch_steps], axis=0),
                dtype=torch.float32,
            )  # (mb*B, 1)

            ret_mb    = torch.tensor(
                np.concatenate([returns[s].reshape(-1, 1) for s in batch_steps], axis=0),
                dtype=torch.float32,
            )  # (mb*B, 1)

            old_loss  = torch.tensor(
                np.concatenate([self.old_cfm_loss[s] for s in batch_steps], axis=0),
                dtype=torch.float32,
            )  # (mb*B, N)

            sigmas_mb = torch.tensor(
                np.concatenate([self.sigmas[s] for s in batch_steps], axis=0),
                dtype=torch.float32,
            )  # (mb*B, N)

            eps_mb    = torch.tensor(
                np.concatenate([self.epsilons[s] for s in batch_steps], axis=0),
                dtype=torch.float32,
            )  # (mb*B, N, C, H, W)

            x0_mb     = torch.tensor(
                np.concatenate([self.x0_latent[s] for s in batch_steps], axis=0),
                dtype=torch.float32,
            )  # (mb*B, C, T, H, W)

            cond_mb   = torch.tensor(
                np.concatenate([self.cond_latent[s] for s in batch_steps], axis=0),
                dtype=torch.float32,
            )  # (mb*B, C, T=11, H, W)

            # data_batch: merge first step's data_batch (text embeddings are constant)
            db_mb = self.data_batches[batch_steps[0]]

            yield adv_mb, ret_mb, old_loss, sigmas_mb, eps_mb, x0_mb, cond_mb, db_mb


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, critic, optimizer, iteration: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "iteration": iteration,
        "lora_state_dict": {k: v for k, v in model.net.named_parameters() if v.requires_grad},
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(path: str, model, critic, optimizer):
    ckpt = torch.load(path, weights_only=False)
    # Load LoRA params
    for name, param in model.net.named_parameters():
        if name in ckpt["lora_state_dict"]:
            param.data.copy_(ckpt["lora_state_dict"][name])
    critic.load_state_dict(ckpt["critic_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"]


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── 1. Build eval config ──────────────────────────────────────────────
    eval_cfg = _EvalCfg(
        config=cfg.cosmos_config,
        ckpt_path=cfg.ckpt_path,
        config_file=cfg.config_file,
        dataset_stats_path=cfg.dataset_stats_path,
        t5_text_embeddings_path=cfg.t5_embeddings_path,
        chunk_size=cfg.chunk_size,
        num_open_loop_steps=cfg.n_open_loop,
    )

    # ── 2. Load model ─────────────────────────────────────────────────────
    print("[init] Loading Cosmos Policy …")
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, load_dataset_stats, init_t5_text_embeddings_cache,
    )
    from cosmos_policy.experiments.robot.cosmos_fpo_model import (
        apply_lora_to_cosmos, CosmosFPOPolicy, Critic,
    )

    init_t5_text_embeddings_cache(cfg.t5_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(eval_cfg)

    # ── 3. Apply LoRA ─────────────────────────────────────────────────────
    print("[init] Applying LoRA …")
    model = apply_lora_to_cosmos(
        model,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets.split(","),
    )

    # ── 4. Build Critic ───────────────────────────────────────────────────
    critic = Critic().cuda()

    # ── 5. Build policy wrapper ───────────────────────────────────────────
    policy = CosmosFPOPolicy(model, critic, dataset_stats, eval_cfg,
                             n_cfm_samples=cfg.n_cfm_samples)

    # ── 6. Optimiser (LoRA params + Critic) ───────────────────────────────
    lora_params  = [p for p in model.net.parameters() if p.requires_grad]
    critic_params = list(critic.parameters())
    optimizer = optim.AdamW(
        [{"params": lora_params,   "lr": cfg.lr_lora},
         {"params": critic_params, "lr": cfg.lr_critic}],
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    # ── 7. Vectorised environment ─────────────────────────────────────────
    print(f"[init] Starting {cfg.num_envs} RoboCasa environments …")
    from cosmos_policy.experiments.robot.robocasa.robocasa_gym_wrapper import (
        VectorizedRoboCasaEnv,
    )
    envs = VectorizedRoboCasaEnv(
        task_name=cfg.task_name,
        num_envs=cfg.num_envs,
        base_seed=cfg.seed,
        img_res=cfg.img_res,
        obj_instance_split=cfg.obj_instance_split,
    )

    # ── 8. Training loop ──────────────────────────────────────────────────
    # Latent shape constants (RoboCasa: C=16, T=11, H=28, W=28)
    LAT_C, LAT_T, LAT_H, LAT_W = 16, 11, 28, 28

    buf = RolloutBuffer(
        steps=cfg.steps_per_iter,
        num_envs=cfg.num_envs,
        n_cfm=cfg.n_cfm_samples,
        latent_c=LAT_C, latent_h=LAT_H, latent_w=LAT_W, latent_t=LAT_T,
    )

    obs_list, _ = envs.reset()
    policy.reset_buffers()

    total_steps = 0
    iteration = 0
    num_iterations = cfg.total_timesteps // (cfg.steps_per_iter * cfg.num_envs)

    success_buffer = deque(maxlen=100)
    ep_len_buffer  = deque(maxlen=100)
    ep_rew_buffer  = deque(maxlen=100)

    cur_ep_len = np.zeros(cfg.num_envs, dtype=np.float32)
    cur_ep_rew = np.zeros(cfg.num_envs, dtype=np.float32)

    print(f"[train] Starting {num_iterations} iterations × "
          f"{cfg.steps_per_iter} steps × {cfg.num_envs} envs")

    for iteration in range(1, num_iterations + 1):
        t_iter_start = time.perf_counter()

        # ── 8a. Rollout collection ────────────────────────────────────────
        model.eval()
        critic.eval()
        # No-grad is handled inside select_action / compute_cfm_loss_for_storage

        # Keep track of last full-chunk CFM data (refreshed at full-env chunk boundaries)
        last_data_batch:  Optional[dict]       = None
        last_old_loss:    Optional[np.ndarray] = None
        last_sigmas:      Optional[np.ndarray] = None
        last_epsilons:    Optional[np.ndarray] = None

        env_list = list(range(cfg.num_envs))

        for step in range(cfg.steps_per_iter):
            # Select action (refills buffer when empty)
            # Returns: (actions, x0_latent, cond_latent_frames, data_batch)
            # x0_latent / cond_latent_frames are None when no env needed a new chunk.
            actions_np, x0_new, cond_latent_new, data_batch_new = policy.select_action(
                obs_list, env_indices=env_list
            )

            # Recompute CFM loss at full-env chunk boundaries only.
            # x0_new / cond_latent_new cover needs_chunk envs (sub_B ≤ num_envs).
            # We only write to the buffer when all envs generated together (sub_B == num_envs),
            # which is the common case: iteration start and every n_open_loop steps without
            # mid-episode resets. Partial boundaries (episode resets causing desync) are skipped.
            full_chunk = x0_new is not None and x0_new.shape[0] == cfg.num_envs
            if full_chunk:
                old_loss, sigmas, epsilons = policy.compute_cfm_loss_for_storage(
                    data_batch_new, x0_new, cond_latent_new
                )
                last_data_batch  = data_batch_new
                last_old_loss    = old_loss.numpy()
                last_sigmas      = sigmas.numpy()
                last_epsilons    = epsilons.numpy()

            # Critic value estimate — use per-env stored latents for correctness
            with torch.no_grad():
                all_cond = policy.get_env_cond_latents(env_list)
                if all_cond is not None:
                    v = policy.get_value(all_cond).cpu().squeeze(-1).numpy()  # (B,)
                else:
                    v = np.zeros(cfg.num_envs, dtype=np.float32)

            # Step environments
            obs_list, rewards, dones, truncateds, infos = envs.step(actions_np)

            # Handle episode resets
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

            # Store CFM data in buffer only at full-env chunk boundaries.
            if full_chunk:
                buf_x0   = policy.get_env_x0_latents(env_list, device="cpu")
                buf_cond = policy.get_env_cond_latents(env_list, device="cpu")
            else:
                buf_x0 = buf_cond = None

            buf.add(
                step_idx=step,
                rewards=rewards,
                dones=dones | truncateds,
                values=v,
                old_cfm_loss=last_old_loss if full_chunk else None,
                sigmas=last_sigmas if full_chunk else None,
                epsilons=last_epsilons if full_chunk else None,
                x0_latent=buf_x0,
                cond_latent=buf_cond,
                data_batch=last_data_batch if full_chunk else None,
            )

        # Bootstrap last value using per-env stored cond latents
        with torch.no_grad():
            all_cond = policy.get_env_cond_latents(env_list)
            if all_cond is not None:
                last_v = policy.get_value(all_cond).cpu().squeeze(-1).numpy()
            else:
                last_v = np.zeros(cfg.num_envs, dtype=np.float32)

        advantages, returns = calculate_advantage(
            buf.values, buf.rewards, buf.dones, last_v,
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        )

        # Normalise advantages globally
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ── 8b. FPO++ update ──────────────────────────────────────────────
        model.train()
        critic.train()

        mean_pg_loss   = 0.0
        mean_vf_loss   = 0.0
        mean_ratio     = 0.0
        n_updates      = 0

        for epoch in range(cfg.update_epochs):
            for (adv_mb, ret_mb, old_loss_mb, sigmas_mb, eps_mb,
                 x0_mb, cond_mb, db_mb) in buf.get_mini_batches(
                    cfg.num_mini_batches, advantages, returns
            ):
                adv_mb      = adv_mb.cuda()         # (mb*B, 1)
                ret_mb      = ret_mb.cuda()         # (mb*B, 1)
                old_loss_mb = old_loss_mb.cuda()    # (mb*B, N)
                sigmas_mb   = sigmas_mb.cuda()      # (mb*B, N)
                eps_mb      = eps_mb.cuda()         # (mb*B, N, C, H, W)
                x0_mb       = x0_mb.cuda()          # (mb*B, C, T=11, H, W)
                cond_mb     = cond_mb.cuda()        # (mb*B, C, T=11, H, W)

                # --- FPO surrogate loss ---
                new_cfm_loss = policy.compute_cfm_loss(
                    db_mb, x0_mb.detach(), cond_mb, sigmas_mb, eps_mb
                )  # (B_mb, N)  — has grad through LoRA

                pg_loss = fpo_surrogate_loss(
                    old_cfm_loss=old_loss_mb,
                    new_cfm_loss=new_cfm_loss,
                    advantages=adv_mb,
                    clip_coef=cfg.clip_coef,
                    trust_region_mode=cfg.trust_region_mode,
                )

                # --- Value loss ---
                v_pred = policy.get_value(cond_mb)  # (B_mb, 1)
                vf_loss = ((v_pred - ret_mb) ** 2).mean()

                # --- Combined loss ---
                loss = pg_loss + cfg.vf_coef * vf_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(lora_params + critic_params, cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    ratio = torch.exp(old_loss_mb - new_cfm_loss)
                    mean_ratio += ratio.mean().item()

                mean_pg_loss += pg_loss.item()
                mean_vf_loss += vf_loss.item()
                n_updates += 1

        if n_updates > 0:
            mean_pg_loss /= n_updates
            mean_vf_loss /= n_updates
            mean_ratio   /= n_updates

        # ── 8c. Logging ───────────────────────────────────────────────────
        t_iter_end = time.perf_counter()
        iter_time  = t_iter_end - t_iter_start
        fps = (cfg.steps_per_iter * cfg.num_envs) / iter_time

        if iteration % cfg.log_interval == 0:
            writer.add_scalar("Loss/policy",    mean_pg_loss, total_steps)
            writer.add_scalar("Loss/value",     mean_vf_loss, total_steps)
            writer.add_scalar("FPO/ratio",      mean_ratio,   total_steps)
            writer.add_scalar("Perf/fps",       fps,          total_steps)
            writer.add_scalar("Perf/iter_time", iter_time,    total_steps)

            if success_buffer:
                writer.add_scalar("Train/success_rate",      np.mean(success_buffer), total_steps)
                writer.add_scalar("Train/mean_ep_reward",    np.mean(ep_rew_buffer),  total_steps)
                writer.add_scalar("Train/mean_ep_length",    np.mean(ep_len_buffer),  total_steps)

            print(
                f"Iter {iteration:4d}/{num_iterations}  "
                f"steps={total_steps:,}  "
                f"fps={fps:5.0f}  "
                f"pg={mean_pg_loss:.4f}  vf={mean_vf_loss:.4f}  "
                f"ratio={mean_ratio:.4f}  "
                + (f"succ={np.mean(success_buffer):.3f}" if success_buffer else "succ=n/a")
            )

        # ── 8d. Checkpoint ────────────────────────────────────────────────
        if iteration % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.log_dir, f"ckpt_{iteration:05d}.pt")
            save_checkpoint(ckpt_path, model, critic, optimizer, iteration)

        # Reset buffer pointer
        buf.ptr = 0

    # ── 9. Final checkpoint ───────────────────────────────────────────────
    save_checkpoint(os.path.join(cfg.log_dir, "ckpt_final.pt"),
                    model, critic, optimizer, iteration)
    envs.close()
    writer.close()
    print("Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="FPO++ Cosmos Policy on RoboCasa")
    cfg = TrainConfig()

    for f in cfg.__dataclass_fields__:
        default = getattr(cfg, f)
        t = type(default)
        p.add_argument(f"--{f}", type=t, default=default)

    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    train(_parse_args())
