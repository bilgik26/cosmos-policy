"""
Unit test: LoRA application + single inference pass on Cosmos Policy.

Run inside the Docker container:
    cd ~/cosmos-policy
    python -m cosmos_policy.experiments.robot.robocasa.test_lora_inference

What this test checks
---------------------
1. Model loads without error.
2. LoRA is applied: trainable params << total params.
3. Non-LoRA params are frozen (requires_grad = False).
4. A single forward pass (generate_samples_from_batch) completes and
   returns an action latent of the expected shape.
5. A single compute_cfm_loss_for_storage() call returns tensors of
   the expected shapes.
6. Critic forward pass returns (B, 1) scalar.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Minimal eval config (mirrors PolicyEvalConfig)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _TestCfg:
    suite: str = "robocasa"
    config: str = "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"
    ckpt_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    dataset_stats_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json"
    t5_text_embeddings_path: str = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl"
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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_cached_task_description() -> str:
    """Return a task description that is guaranteed to be in the T5 embedding cache."""
    import pickle, glob
    pattern = (
        "/home/ubuntu/.cache/huggingface/models--nvidia--Cosmos-Policy-RoboCasa-Predict2-2B"
        "/snapshots/*/robocasa_t5_embeddings.pkl"
    )
    pkl_files = glob.glob(pattern)
    if pkl_files:
        data = pickle.load(open(pkl_files[0], "rb"))
        return list(data.keys())[0]
    return "pick the cucumber from the cabinet and place it on the counter"


def _make_dummy_obs(h: int = 224, w: int = 224) -> dict:
    """Create a minimal obs dict that get_action() expects."""
    rng = np.random.default_rng(42)
    return {
        "primary_image":   rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "secondary_image": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "wrist_image":     rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "proprio":         rng.random(9).astype(np.float32),
        "task_description": [_get_cached_task_description()],
    }


def _assert(condition: bool, msg: str):
    if not condition:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  PASS: {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_lora_inference():
    print("\n" + "=" * 70)
    print("Cosmos Policy LoRA Inference Unit Test")
    print("=" * 70)

    # ── 1. Imports ────────────────────────────────────────────────────────
    print("\n[1] Importing modules …")
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        load_dataset_stats,
        init_t5_text_embeddings_cache,
    )
    from cosmos_policy.experiments.robot.cosmos_fpo_model import (
        apply_lora_to_cosmos,
        CosmosFPOPolicy,
        Critic,
        _build_data_batch_from_obs,
    )
    print("  OK")

    cfg = _TestCfg()

    # ── 2. Load model ─────────────────────────────────────────────────────
    print("\n[2] Loading Cosmos Policy checkpoint …")
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)
    print(f"  model type: {type(model).__name__}")
    print(f"  net type:   {type(model.net).__name__}")

    total_before = sum(p.numel() for p in model.parameters())
    print(f"  total params before LoRA: {total_before:,}")

    # Put model in eval mode before applying LoRA (disables dropout, etc.)
    model.eval()

    # ── 3. Apply LoRA ─────────────────────────────────────────────────────
    print("\n[3] Applying LoRA (rank=8) …")
    model = apply_lora_to_cosmos(model, lora_rank=8, lora_alpha=16)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    ratio = n_trainable / n_total

    _assert(n_trainable > 0,        "trainable params > 0 after LoRA")
    _assert(ratio < 0.05,           f"LoRA trainable fraction < 5% (got {ratio:.4%})")

    # Check backbone is frozen (spot-check a few non-LoRA params)
    frozen_params = [
        p for name, p in model.named_parameters()
        if "lora_" not in name and p.requires_grad
    ]
    _assert(len(frozen_params) == 0, "all non-LoRA params are frozen")

    # ── 4. Build Critic ───────────────────────────────────────────────────
    print("\n[4] Building Critic …")
    critic = Critic().cuda()
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"  critic params: {n_critic:,}")
    _assert(n_critic > 0, "critic has parameters")

    # ── 5. Build data_batch and condition ────────────────────────────────
    print("\n[5] Building data_batch from dummy obs …")
    obs = _make_dummy_obs()
    data_batch = _build_data_batch_from_obs(obs, model, dataset_stats, cfg)
    _assert("video" in data_batch,              "'video' key present")
    _assert("t5_text_embeddings" in data_batch, "'t5_text_embeddings' key present")
    _assert(data_batch["video"].dtype == torch.uint8, "video is uint8")
    print(f"  video shape: {tuple(data_batch['video'].shape)}")

    # Build the full condition (text + video conditioning) using stored latent.
    # model.conditioner() gives text; set_video_condition() adds gt_frames + mask.
    print("\n[5.5] Building full Video2WorldCondition …")
    B, C, T, H, W = 1, 16, 11, 28, 28
    x0_synth = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)

    # Build a dummy CosmosFPOPolicy just to reuse _build_condition
    from cosmos_policy.experiments.robot.cosmos_fpo_model import CosmosFPOPolicy, Critic as Critic2
    _policy_tmp = CosmosFPOPolicy(model, critic, dataset_stats, cfg, n_cfm_samples=4)

    with torch.no_grad():
        condition = _policy_tmp._build_condition(data_batch, x0_synth.float())
    print(f"  condition type: {type(condition).__name__}")
    print(f"  condition.gt_frames is None: {condition.gt_frames is None}")
    _assert(condition.gt_frames is not None, "condition.gt_frames is populated")

    # ── 6. model.denoise() forward pass ──────────────────────────────────
    # Use model.train() so gradient checkpointing + torch.no_grad() interact
    # correctly (eval mode + checkpoint can fail with some backends).
    print("\n[6] model.denoise() forward pass (train mode + no_grad) …")
    model.train()

    # Noise only the action token (idx 5); conditional frames stay clean
    eps_synth = torch.randn_like(x0_synth)
    xt_synth  = x0_synth.clone()
    sigma_val = 1.0
    xt_synth[:, :, 5, :, :] = x0_synth[:, :, 5, :, :] + sigma_val * eps_synth[:, :, 5, :, :]

    # Per-frame sigma: near-zero for conditional, sigma_val for action token
    sigma_BT = torch.full((B, T), model.config.sigma_conditional,
                          device="cuda", dtype=torch.bfloat16)
    sigma_BT[:, 5] = sigma_val

    with torch.no_grad():
        denoised = model.denoise(xt_synth, sigma_BT, condition)

    print(f"  x0_pred shape: {tuple(denoised.x0.shape)}")
    _assert(denoised.x0.shape == (B, C, T, H, W), "x0_pred has correct shape")

    # ── 7. CFM loss on synthetic latent ──────────────────────────────────
    # x0_synth = generated latent (action at index 5)
    # cond_synth = "orig_clean" latent (same shape, used as gt_frames)
    # For the unit test they're the same random tensor — just verifying shapes.
    print("\n[7] compute_cfm_loss_for_storage on synthetic latent …")
    policy = CosmosFPOPolicy(model, critic, dataset_stats, cfg, n_cfm_samples=4)

    cond_synth = x0_synth.clone()  # simulated orig_clean_latent_frames

    old_loss, sigmas, epsilons = policy.compute_cfm_loss_for_storage(
        data_batch, x0_synth.float(), cond_synth.float(), n_samples=4
    )
    print(f"  old_loss shape: {tuple(old_loss.shape)}  (expect (1, 4))")
    print(f"  sigmas shape:   {tuple(sigmas.shape)}")
    print(f"  epsilons shape: {tuple(epsilons.shape)}")

    _assert(old_loss.shape == (1, 4),  "old_loss is (B=1, N=4)")
    _assert(sigmas.shape == (1, 4),    "sigmas is (B=1, N=4)")
    _assert((old_loss >= 0).all(),     "all losses non-negative")

    # ── 8. Critic forward ─────────────────────────────────────────────────
    print("\n[8] Critic forward pass …")
    value = policy.get_value(cond_synth.float())
    print(f"  value shape: {tuple(value.shape)}  (expect (1, 1))")
    _assert(value.shape == (1, 1), "critic returns (B=1, 1)")

    # ── 9. Grad-enabled CFM loss + backward ──────────────────────────────
    print("\n[9] compute_cfm_loss with backward …")
    torch.cuda.empty_cache()

    new_loss = policy.compute_cfm_loss(
        data_batch,
        x0_synth.float().detach(),
        cond_synth.float().detach(),
        sigmas.cuda(),
        epsilons.cuda(),
    )
    print(f"  new_loss shape: {tuple(new_loss.shape)}  (expect (1, 4))")
    _assert(new_loss.shape == (1, 4), "new_loss is (B=1, N=4)")

    new_loss.mean().backward()
    lora_grads = [
        p.grad for p in model.net.parameters()
        if p.requires_grad and p.grad is not None
    ]
    _assert(len(lora_grads) > 0, "LoRA params have gradients after backward()")

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("All tests PASSED.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_lora_inference()
