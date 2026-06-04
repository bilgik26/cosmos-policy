"""
Cosmos Policy wrapper for FPO++ online RL.

Provides:
  - apply_lora_to_cosmos():   attaches LoRA adapters to DiT attention layers.
  - CosmosFPOPolicy:
      prepare_data_batch()           build data_batch from a raw obs dict
      select_action()                action-chunked inference with buffer
      compute_cfm_loss_for_storage() sample (sigma, eps) + compute old loss
      compute_cfm_loss()             recompute loss at stored (sigma, eps)
      get_env_values()               return per-env V(s) from last chunk's value token

Value estimation
----------------
No separate Critic or ValueHead.  V(s) is read directly from the value token
(index 10) that Cosmos generates together with the action chunk, using the same
extraction and unnormalization as run_robocasa_eval.py:

    raw = mean( generated[:, :, 10, :, :].flatten() )   # (B,)
    V(s) = clamp( (raw + 1) / 2, 0, 1 )                 # mapped to [0, 1]

Design notes
------------
Cosmos uses an EDM noise schedule (Karras et al. 2022).
FPO++ treats the EDM reconstruction loss as a pseudo-log-likelihood:

    L(a; s, σ, ε) = w(σ) · ‖x0_pred(x_t, σ, s) − x0_action‖²

where:
  x_t     = orig_clean_latent_frames with the action token (index 5) replaced
             by   x0_action + σ·ε
  x0_action  = generated_latent_with_action[:,:,5,:,:]  (the denoised action)
  w(σ)   = EDM loss weight

orig_clean_latent_frames (stored from rollout) serves as condition.gt_frames:
  frames 0-4 = VAE-encoded state (proprio injected by model during rollout)
  frame  5   = blank/zero action placeholder
  frames 6-10= blank future / value placeholders

The policy model's denoise() replaces frames 0-4 with gt_frames / sigma_data
via the condition mask.  Frame 5 is kept as the noisy action sample.
Only the loss on frame 5 is used for the FPO++ update.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# ──────────────────────────────────────────────────────────────────────────────
# Constants (match RoboCasa model config)
# ──────────────────────────────────────────────────────────────────────────────

# Temporal structure for RoboCasa (state_t = 11, min_num_conditional_frames = 5)
# Index: 0=blank, 1=proprio, 2=wrist, 3=agentview_left, 4=agentview_right
#        5=action, 6=future_proprio, 7=future_wrist, 8=future_left,
#        9=future_right, 10=value
ROBOCASA_NUM_COND_FRAMES = 5
ROBOCASA_ACTION_LATENT_IDX  = 5   # == num_cond_frames
ROBOCASA_VALUE_LATENT_IDX   = 10  # value token in the generated sequence
# Future state tokens: future_proprio, future_wrist, future_left, future_right
ROBOCASA_FUTURE_LATENT_INDICES = (6, 7, 8, 9)

# Mapping from future prediction index → current-state index in the next chunk's
# cond_latent_frames.  At the next chunk boundary, cond_latent[:, :, src, :, :]
# is the actual ground-truth for the predicted future token at dst.
FUTURE_TARGET_MAPPING = {6: 1, 7: 2, 8: 3, 9: 4}  # dst → src

# VAE / tokenizer constants
COSMOS_IMAGE_SIZE = 224
COSMOS_TEMPORAL_COMPRESSION = 4  # 1 latent frame ≡ 4 raw frames

# Sigma is sampled via model.sde.sample_t() (HybridEDMSDE) — see constants below.


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning setup: LoRA or full DiT
# ──────────────────────────────────────────────────────────────────────────────

def apply_lora_to_cosmos(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Attach LoRA adapters to the Cosmos DiT (model.net) and freeze everything else.

    Args:
        model:          Full Cosmos policy model (has .net attribute = DiT).
        lora_rank:      LoRA rank r.
        lora_alpha:     LoRA scaling α.
        lora_dropout:   Dropout on the LoRA path.
        target_modules: Linear layer name suffixes to adapt.

    Returns:
        The model with LoRA applied in-place to model.net.
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]

    # 1. Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Attach LoRA to the DiT
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.net = get_peft_model(model.net, lora_cfg)

    # 3. Keep LoRA adapters in bfloat16 to match the base model
    model.net = model.net.to(torch.bfloat16)
    torch.cuda.empty_cache()

    # 4. Invalidate torch.compile cache — the LoRA-wrapped q/k/v projections change
    # the module graph, so any previously compiled graphs must be recompiled.
    torch._dynamo.reset()

    n_trainable = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.net.parameters())
    print(
        f"[LoRA] trainable: {n_trainable:,} / {n_total:,} "
        f"({100*n_trainable/n_total:.3f}%)"
    )
    return model


def unfreeze_dit_for_finetuning(model: nn.Module) -> nn.Module:
    """
    Freeze everything except the DiT (model.net) for full parameter fine-tuning.

    The VAE tokenizer (encoder/decoder) remains frozen — only the transformer
    weights are updated, matching the scope of Cosmos-IL supervised training.

    Args:
        model: Full Cosmos policy model (has .net attribute = DiT).

    Returns:
        The model with model.net fully unfrozen and all other weights frozen.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the DiT in full
    for param in model.net.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.net.parameters())
    n_total_model = sum(p.numel() for p in model.parameters())
    print(
        f"[Full DiT] trainable: {n_trainable:,} / {n_total_model:,} "
        f"({100*n_trainable/n_total_model:.3f}%)"
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# EDM loss weight
# ──────────────────────────────────────────────────────────────────────────────
# Sigma is sampled via model.sde.sample_t() so that the distribution matches
# supervised training exactly (HybridEDMSDE: 70% log-normal p_mean=1.386,
# p_std=1.2  + 30% uniform [1.0, 85.0]).


def _edm_loss_weight(sigma: torch.Tensor) -> torch.Tensor:
    sigma_data = 0.5
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def _build_sigma_B_T(
    B: int,
    T: int,
    dev,
    sigma: torch.Tensor,
    sigma_conditional: float,
    prediction_indices: List[int],
) -> torch.Tensor:
    """Build per-frame sigma tensor used by model.denoise().

    Conditional frames (0..ROBOCASA_NUM_COND_FRAMES-1) receive sigma_conditional;
    prediction_indices receive the sampled sigma; all other frames remain zero.
    """
    sigma_B_T = torch.zeros(B, T, device=dev, dtype=sigma.dtype)
    sigma_B_T[:, :ROBOCASA_NUM_COND_FRAMES] = sigma_conditional
    for idx in prediction_indices:
        sigma_B_T[:, idx] = sigma
    return sigma_B_T


def _latent_frame_to_scalar(latent_frame: torch.Tensor) -> torch.Tensor:
    """Mean-pool a single latent frame and unnormalize from [-1,1] to [0,1].

    Args:
        latent_frame: (B, C, H, W) float32
    Returns:
        (B,) float32 in [0, 1]
    """
    B   = latent_frame.shape[0]
    raw = latent_frame.float().reshape(B, -1).mean(dim=1)
    return torch.clamp((raw + 1.0) / 2.0, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Data-batch preparation
# ──────────────────────────────────────────────────────────────────────────────

def _build_data_batch_from_obs(
    obs_batch: Dict,
    model,
    dataset_stats: dict,
    cfg,
) -> dict:
    """
    Convert a batch of canonical obs dicts into the data_batch format expected
    by model.generate_samples_from_batch() / model.conditioner().

    Args:
        obs_batch: dict with keys:
            primary_image   (H,W,3) or (B,H,W,3) uint8
            secondary_image (H,W,3) or (B,H,W,3) uint8
            wrist_image     (H,W,3) or (B,H,W,3) uint8
            proprio         (D,)    or (B,D)      float32
            task_description  str  or list[str]   -- MUST be in T5 cache

    WARNING: task_description must be a key already in the T5 embedding cache.
    If the key is missing, the T5 encoder will be loaded (~9GB) causing OOM.
    """
    from cosmos_policy.experiments.robot.cosmos_utils import (
        COSMOS_TEMPORAL_COMPRESSION_FACTOR,
        prepare_images_for_model,
        rescale_proprio,
        get_t5_embedding_from_cache,
    )
    from cosmos_policy.utils.utils import duplicate_array

    # Normalise task_description to list[str]
    task_descs = obs_batch["task_description"]
    if isinstance(task_descs, str):
        task_descs = [task_descs]

    # Normalise images to (B, H, W, C)
    if isinstance(obs_batch["primary_image"], np.ndarray) and obs_batch["primary_image"].ndim == 3:
        obs_batch = {k: (np.expand_dims(v, 0) if isinstance(v, np.ndarray) else v)
                     for k, v in obs_batch.items()}

    B = len(task_descs)

    all_raw_sequences = []
    proprio_tensors = []
    t5_embs = []

    for b in range(B):
        images = prepare_images_for_model(
            [obs_batch["wrist_image"][b],
             obs_batch["primary_image"][b],
             obs_batch["secondary_image"][b]],
            cfg,
        )
        blank = np.zeros_like(np.array(images[1]))

        def dup(arr):
            return duplicate_array(arr, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)

        blank_dup = dup(blank)
        seq = [
            np.expand_dims(np.zeros_like(blank), 0),  # blank (1 frame)
            blank_dup.copy(),                           # proprio placeholder
            dup(np.array(images[0])),                   # wrist
            dup(np.array(images[1])),                   # agentview_left
            dup(np.array(images[2])),                   # agentview_right
            blank_dup.copy(),                           # action placeholder
            blank_dup.copy(),                           # future proprio
            dup(np.array(images[0])).copy(),            # future wrist
            dup(np.array(images[1])).copy(),            # future primary
            dup(np.array(images[2])).copy(),            # future secondary
            blank_dup.copy(),                           # value placeholder
        ]

        raw_seq = np.concatenate(seq, axis=0)    # (T_raw, H, W, C)
        raw_seq = raw_seq.transpose(3, 0, 1, 2)  # (C, T_raw, H, W)
        all_raw_sequences.append(raw_seq)

        if cfg.use_proprio:
            proprio = obs_batch["proprio"][b]
            if cfg.normalize_proprio:
                proprio = rescale_proprio(proprio, dataset_stats, scale_multiplier=1.0)
            proprio_tensors.append(torch.tensor(proprio, dtype=torch.bfloat16))

        t5_embs.append(get_t5_embedding_from_cache(task_descs[b]))

    raw_video = np.stack(all_raw_sequences, axis=0)  # (B, C, T_raw, H, W)
    video_t = torch.from_numpy(raw_video).to(dtype=torch.uint8).cuda()
    # Each T5 embedding has shape (1, N_tokens, D). Concatenate along batch dim.
    t5_emb_t = torch.cat(t5_embs, dim=0).to(dtype=torch.bfloat16).cuda()  # (B, N, D)

    def _idx(v):
        return torch.tensor([v] * B, dtype=torch.int64).cuda()

    data_batch = {
        "dataset_name": "video_data",
        "video": video_t,
        "t5_text_embeddings": t5_emb_t,
        "t5_text_mask": torch.ones(B, t5_emb_t.shape[1], dtype=torch.bfloat16).cuda(),
        "fps": torch.tensor([16] * B, dtype=torch.bfloat16).cuda(),
        "padding_mask": torch.zeros(
            B, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE, dtype=torch.bfloat16
        ).cuda(),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": torch.stack(proprio_tensors, 0).cuda() if cfg.use_proprio else None,
        # Latent sequence indices (fixed for RoboCasa)
        "current_proprio_latent_idx":        _idx(1),
        "current_wrist_image_latent_idx":    _idx(2),
        "current_wrist_image2_latent_idx":   _idx(-1),
        "current_image_latent_idx":          _idx(3),
        "current_image2_latent_idx":         _idx(4),
        "action_latent_idx":                 _idx(5),
        "future_proprio_latent_idx":         _idx(6),
        "future_wrist_image_latent_idx":     _idx(7),
        "future_wrist_image2_latent_idx":    _idx(-1),
        "future_image_latent_idx":           _idx(8),
        "future_image2_latent_idx":          _idx(9),
        "value_latent_idx":                  _idx(10),
    }
    return data_batch


# ──────────────────────────────────────────────────────────────────────────────
# Main policy wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CosmosFPOPolicy:
    """
    Wraps a LoRA-adapted Cosmos model and exposes the FPO++ RL interface.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset_stats: dict,
        cfg,
        n_cfm_samples: int = 16,
    ):
        self.model = model
        self.dataset_stats = dataset_stats
        self.cfg = cfg
        self.n_cfm_samples = n_cfm_samples
        self.chunk_size = cfg.chunk_size
        self.n_open_loop = cfg.num_open_loop_steps
        self._action_buffers: Dict[int, deque] = {}
        # Per-env latents (CPU tensors, shape (1, 16, 11, H, W)), updated at chunk boundaries
        self._env_cond:  Dict[int, torch.Tensor] = {}   # orig_clean_latent_frames
        self._env_x0:    Dict[int, torch.Tensor] = {}   # generated_latent (full sequence)
        self._env_value: Dict[int, float] = {}           # V(s) from model's value token

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        obs_batch: Dict,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict]]:
        """
        Return next actions.  On chunk boundary, also returns:
          x0_latent:           (sub_B, 16, 11, 28, 28) generated clean latent
          cond_latent_frames:  (sub_B, 16, 11, 28, 28) clean input latent (gt_frames)
          data_batch:          dict for CFM loss recomputation
        """
        from cosmos_policy.experiments.robot.cosmos_utils import (
            extract_action_chunk_from_latent_sequence,
            unnormalize_actions,
        )
        from cosmos_policy.constants import ACTION_DIM

        B = len(obs_batch["task_description"])
        if env_indices is None:
            env_indices = list(range(B))

        for i in env_indices:
            if i not in self._action_buffers or len(self._action_buffers[i]) == 0:
                self._action_buffers[i] = deque()

        needs_chunk = [i for i in env_indices if len(self._action_buffers[i]) == 0]
        x0_latent_out = None
        cond_latent_out = None
        data_batch_out = None

        if needs_chunk:
            sub_obs = {
                k: ([obs_batch[k][i] for i in needs_chunk]
                    if isinstance(obs_batch[k], list)
                    else obs_batch[k][needs_chunk])
                for k in obs_batch
            }
            data_batch = _build_data_batch_from_obs(
                sub_obs, self.model, self.dataset_stats, self.cfg
            )

            generated, orig_clean = self.model.generate_samples_from_batch(
                data_batch,
                n_sample=len(needs_chunk),
                num_steps=self.cfg.num_denoising_steps_action,
                seed=1,
                is_negative_prompt=False,
                use_variance_scale=self.cfg.use_variance_scale,
                return_orig_clean_latent_frames=True,
            )
            # generated:   (sub_B, 16, 11, 28, 28) — clean generated sequence
            # orig_clean:  (sub_B, 16, 11, 28, 28) — VAE-encoded input + proprio inj.

            action_indices = torch.full(
                (len(needs_chunk),), ROBOCASA_ACTION_LATENT_IDX,
                dtype=torch.int64, device=generated.device,
            )
            actions_raw = (
                extract_action_chunk_from_latent_sequence(
                    generated,
                    action_shape=(self.cfg.chunk_size, ACTION_DIM),
                    action_indices=action_indices,
                )
                .float().cpu().numpy()
            )  # (sub_B, chunk_size, 7)
            if self.cfg.unnormalize_actions:
                actions_raw = unnormalize_actions(actions_raw, self.dataset_stats)

            # Extract V(s) from the model's value token (index 10, last frame).
            value_latent = generated[:, :, ROBOCASA_VALUE_LATENT_IDX, :, :].float()
            env_values   = _latent_frame_to_scalar(value_latent)  # (sub_B,) in [0,1]

            for k, env_i in enumerate(needs_chunk):
                for s in range(self.n_open_loop):
                    self._action_buffers[env_i].append(actions_raw[k, s])
                # Store per-env latents on CPU for CFM loss recomputation
                self._env_cond[env_i]  = orig_clean[k : k + 1].float().cpu()
                self._env_x0[env_i]    = generated[k : k + 1].float().cpu()
                self._env_value[env_i] = env_values[k].item()  # Python float

            x0_latent_out = generated.float()
            cond_latent_out = orig_clean.float()
            data_batch_out = data_batch

        actions = np.stack(
            [self._action_buffers[i].popleft() for i in env_indices], axis=0
        )
        return actions, x0_latent_out, cond_latent_out, data_batch_out

    def reset_buffers(self, env_indices: Optional[List[int]] = None):
        if env_indices is None:
            self._action_buffers.clear()
            self._env_cond.clear()
            self._env_x0.clear()
            self._env_value.clear()
        else:
            for i in env_indices:
                self._action_buffers.pop(i, None)
                self._env_cond.pop(i, None)
                self._env_x0.pop(i, None)
                self._env_value.pop(i, None)

    def get_env_cond_latents(
        self, env_indices: List[int], device: str = "cuda"
    ) -> Optional[torch.Tensor]:
        """
        Return stacked cond latents for the given env indices.
        Returns None if any env has no latent yet.
        Shape: (len(env_indices), 16, 11, H, W)
        """
        if not all(i in self._env_cond for i in env_indices):
            return None
        return torch.cat([self._env_cond[i] for i in env_indices], dim=0).to(device)

    def get_env_x0_latents(
        self, env_indices: List[int], device: str = "cuda"
    ) -> Optional[torch.Tensor]:
        """
        Return stacked x0 (generated action) latents for the given env indices.
        Returns None if any env has no latent yet.
        Shape: (len(env_indices), 16, 11, H, W)
        """
        if not all(i in self._env_x0 for i in env_indices):
            return None
        return torch.cat([self._env_x0[i] for i in env_indices], dim=0).to(device)

    # ──────────────────────────────────────────
    # Value access
    # ──────────────────────────────────────────

    def get_env_values(self, env_indices: List[int]) -> Optional[np.ndarray]:
        """Return V(s) for each env from the last chunk's value token.

        Returns None if any env has not generated a chunk yet.
        Shape: (len(env_indices),) float32  range [0, 1].
        """
        if not all(i in self._env_value for i in env_indices):
            return None
        return np.array([self._env_value[i] for i in env_indices], dtype=np.float32)

    # ──────────────────────────────────────────
    # Condition builder (no VAE re-encoding)
    # ──────────────────────────────────────────

    def _build_condition(
        self,
        data_batch: dict,
        cond_latent_frames: torch.Tensor,
    ):
        """
        Build a Video2WorldCondition from stored data without VAE re-encoding.

        Args:
            data_batch:         dict with T5 embeddings + metadata
            cond_latent_frames: (B, 16, 11, H, W) clean latent from rollout
                                (= orig_clean_latent_frames, serves as gt_frames)
        """
        # 1. Get text + misc conditioning (cheap — just packages embeddings)
        text_cond = self.model.conditioner(data_batch)

        # 2. Add video conditioning using the stored clean latent
        full_cond = text_cond.set_video_condition(
            gt_frames=cond_latent_frames.to(**self.model.tensor_kwargs),
            random_min_num_conditional_frames=self.model.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.model.config.max_num_conditional_frames,
            num_conditional_frames=ROBOCASA_NUM_COND_FRAMES,
        )
        return full_cond

    # ──────────────────────────────────────────
    # CFM loss helpers
    # ──────────────────────────────────────────

    def _edm_loss_on_action_token(
        self,
        cond_latent_frames: torch.Tensor,  # (B, 16, 11, H, W) clean input latent
        x0_action: torch.Tensor,           # (B, 16, H, W) clean generated action
        sigma: torch.Tensor,               # (B,)
        eps: torch.Tensor,                 # (B, 16, H, W)
        condition,
    ) -> torch.Tensor:
        """
        Single EDM forward pass; returns weighted MSE on the action token.

        Noise is applied ONLY to the action token (index 5).
        Conditional frames (0-4) are replaced by condition.gt_frames / sigma_data
        inside model.denoise() via the condition mask.
        """
        B = cond_latent_frames.shape[0]
        T = cond_latent_frames.shape[2]

        xt_full  = cond_latent_frames.clone()
        sigma_hw = rearrange(sigma, "b -> b 1 1 1")
        xt_full[:, :, ROBOCASA_ACTION_LATENT_IDX, :, :] = x0_action + sigma_hw * eps

        sigma_B_T = _build_sigma_B_T(B, T, sigma.device, sigma,
                                     self.model.config.sigma_conditional,
                                     [ROBOCASA_ACTION_LATENT_IDX])

        denoised = self.model.denoise(xt_full, sigma_B_T, condition)
        x0_pred_action = denoised.x0[:, :, ROBOCASA_ACTION_LATENT_IDX, :, :]

        mse = ((x0_pred_action - x0_action.to(x0_pred_action.dtype)) ** 2).mean(dim=[1, 2, 3])
        weight = _edm_loss_weight(sigma)
        return weight * mse

    # ──────────────────────────────────────────
    # CFM loss for rollout storage
    # ──────────────────────────────────────────

    @torch.no_grad()
    def compute_cfm_loss_for_storage(
        self,
        data_batch: dict,
        x0_full_latent: torch.Tensor,       # generated_latent_with_action
        cond_latent_frames: torch.Tensor,   # orig_clean_latent_frames
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (σ, ε) and compute old EDM loss on the action token.

        Returns:
            old_loss:  (B, N) — cpu
            sigmas:    (B, N) — cpu
            epsilons:  (B, N, 16, H, W) — cpu
        """
        N = n_samples or self.n_cfm_samples
        B = x0_full_latent.shape[0]
        dev = x0_full_latent.device

        x0_action = x0_full_latent[:, :, ROBOCASA_ACTION_LATENT_IDX, :, :]
        # Use model.sde.sample_t() for the same HybridEDMSDE distribution as supervised training
        sigmas = self.model.sde.sample_t(B * N).reshape(B, N)
        epsilons = torch.randn(B, N, *x0_action.shape[1:], device=dev)

        condition = self._build_condition(data_batch, cond_latent_frames)

        losses = torch.zeros(B, N, device=dev)
        for n in range(N):
            losses[:, n] = self._edm_loss_on_action_token(
                cond_latent_frames, x0_action, sigmas[:, n], epsilons[:, n], condition
            )

        return losses.cpu(), sigmas.cpu(), epsilons.cpu()

    # ──────────────────────────────────────────
    # CFM loss for FPO update (needs grad)
    # ──────────────────────────────────────────

    def compute_cfm_loss(
        self,
        data_batch: dict,
        x0_full_latent: torch.Tensor,
        cond_latent_frames: torch.Tensor,
        sigmas: torch.Tensor,
        epsilons: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recompute EDM loss at stored (σ, ε) — with grad through LoRA params.

        Returns:
            new_loss: (B, N)
        """
        B, N = sigmas.shape
        dev = x0_full_latent.device
        sigmas = sigmas.to(dev)
        epsilons = epsilons.to(dev)
        cond_latent_frames = cond_latent_frames.to(dev)

        x0_action = x0_full_latent[:, :, ROBOCASA_ACTION_LATENT_IDX, :, :]
        condition = self._build_condition(data_batch, cond_latent_frames)

        losses = []
        for n in range(N):
            loss_n = self._edm_loss_on_action_token(
                cond_latent_frames, x0_action, sigmas[:, n], epsilons[:, n], condition
            )
            losses.append(loss_n)

        return torch.stack(losses, dim=1)  # (B, N)

    # ──────────────────────────────────────────────────────────────────────────
    # Combined value + future prediction (shared sigma, single forward pass)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_value_and_future_loss(
        self,
        data_batch: dict,
        returns: torch.Tensor,             # (B, 1) or (B,)  GAE returns in [0, 1]
        cond_latent_frames: torch.Tensor,  # (B, C, T, H, W) conditioning from rollout
        future_cond_latent: torch.Tensor,  # (B, C, T, H, W) VAE-encoded actual future obs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single EDM forward pass for value (idx 10) and future state tokens (idx 6-9),
        sharing ONE sigma per batch item — matching supervised training exactly.

        Value token target format (Cosmos-IL style):
          The scalar GAE return is renormalized [0,1] → [-1,1] and broadcast uniformly
          across (C, H, W), identical to how Cosmos-IL injects value_function_return:
            x0_value = ret_normalized.reshape(B,1,1,1).expand(B, C, H, W)
          EDM reconstruction loss is computed at the latent level (same as Cosmos-IL).

        Future token targets (FUTURE_TARGET_MAPPING):
          idx 6-9 ← future_cond_latent[:, :, 1-4, :, :]  (actual future obs)

        Returns:
            v_pred:   (B,) float32 in [0, 1]  — scalar value from denoised token
            aux_loss: scalar — EDM weighted MSE averaged over value + future tokens
        """
        B   = cond_latent_frames.shape[0]
        T   = cond_latent_frames.shape[2]
        C   = cond_latent_frames.shape[1]
        H   = cond_latent_frames.shape[3]
        W   = cond_latent_frames.shape[4]
        dev = cond_latent_frames.device

        future_cond_latent = future_cond_latent.to(dev)

        # Renormalize returns [0,1] → [-1,1] to match Cosmos-IL injection scale,
        # then broadcast scalar uniformly to fill (C, H, W) — same format as:
        #   x0[b, :, value_idx, :, :] = scalar.reshape(-1,1,1,1).expand(-1, C, H, W)
        ret_normalized = (returns.view(B) * 2.0 - 1.0).clamp(-1.0, 1.0)  # (B,)
        x0_value = ret_normalized.view(B, 1, 1, 1).expand(B, C, H, W).float().contiguous()

        # ONE sigma per batch item, shared across value + all future tokens.
        sigma    = self.model.sde.sample_t(B).to(dev)   # (B,)
        sigma_hw = rearrange(sigma, "b -> b 1 1 1")

        # Build noisy input: start from clean conditioning, replace prediction tokens
        xt_full = cond_latent_frames.clone()
        xt_full[:, :, ROBOCASA_VALUE_LATENT_IDX, :, :] = (
            x0_value + sigma_hw * torch.randn_like(x0_value)
        )
        for dst_idx, src_idx in FUTURE_TARGET_MAPPING.items():
            x0_future = future_cond_latent[:, :, src_idx, :, :].float()
            xt_full[:, :, dst_idx, :, :] = x0_future + sigma_hw * torch.randn_like(x0_future)

        sigma_B_T = _build_sigma_B_T(
            B, T, dev, sigma,
            self.model.config.sigma_conditional,
            [ROBOCASA_VALUE_LATENT_IDX] + list(FUTURE_TARGET_MAPPING.keys()),
        )

        condition = self._build_condition(data_batch, cond_latent_frames)
        denoised  = self.model.denoise(xt_full, sigma_B_T, condition)
        weight    = _edm_loss_weight(sigma)  # (B,)

        # ── Value scalar prediction ─────────────────────────────────────────
        x0_pred_value = denoised.x0[:, :, ROBOCASA_VALUE_LATENT_IDX, :, :]  # (B, C, H, W)
        v_pred = _latent_frame_to_scalar(x0_pred_value)                      # (B,) [0,1]

        # ── EDM reconstruction losses (value + future, averaged over 5 tokens) ──
        total_loss = torch.zeros(B, device=dev)

        # Value token: latent-level reconstruction against uniform broadcast target
        mse_v = ((x0_pred_value - x0_value.to(x0_pred_value.dtype)) ** 2).mean(dim=[1, 2, 3])
        total_loss = total_loss + weight * mse_v

        # Future tokens: reconstruction against actual future observations
        for dst_idx, src_idx in FUTURE_TARGET_MAPPING.items():
            x0_pred_f   = denoised.x0[:, :, dst_idx, :, :]
            x0_target_f = future_cond_latent[:, :, src_idx, :, :].float()
            mse_f = ((x0_pred_f - x0_target_f.to(x0_pred_f.dtype)) ** 2).mean(dim=[1, 2, 3])
            total_loss = total_loss + weight * mse_f

        aux_loss = (total_loss / (1 + len(FUTURE_TARGET_MAPPING))).mean()
        return v_pred, aux_loss
