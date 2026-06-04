"""RolloutBuffer for FPO++ online RL training of Cosmos Policy."""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Stores one iteration's worth of transitions for FPO++ update.

    Only chunk-boundary steps that have a valid actual future observation target
    (has_future_cond == True) are yielded by get_mini_batches().
    """

    def __init__(
        self,
        steps: int,
        num_envs: int,
        n_cfm: int,
        latent_c: int,
        latent_h: int,
        latent_w: int,
        latent_t: int,
    ):
        self.steps = steps
        self.B     = num_envs
        self.N     = n_cfm
        C, H, W, T = latent_c, latent_h, latent_w, latent_t

        # RL scalars
        self.rewards = np.zeros((steps, num_envs), dtype=np.float32)
        self.dones   = np.zeros((steps, num_envs), dtype=bool)
        self.values  = np.zeros((steps, num_envs), dtype=np.float32)

        # FPO++ tensors — stored only at chunk boundaries
        self.old_cfm_loss = np.zeros((steps, num_envs, n_cfm),          dtype=np.float32)
        self.sigmas       = np.zeros((steps, num_envs, n_cfm),          dtype=np.float32)
        self.epsilons     = np.zeros((steps, num_envs, n_cfm, C, H, W), dtype=np.float32)
        self.x0_latent    = np.zeros((steps, num_envs, C, T, H, W),     dtype=np.float32)
        self.cond_latent  = np.zeros((steps, num_envs, C, T, H, W),     dtype=np.float32)

        # Actual future obs latent — filled retroactively at the next chunk boundary.
        # has_future_cond[s] = True when no episode reset occurred during n_open_loop
        # steps following step s.
        self.future_cond_latent = np.zeros((steps, num_envs, C, T, H, W), dtype=np.float32)
        self.has_future_cond    = np.zeros(steps, dtype=bool)

        # data_batch dict per chunk boundary step (for text embeddings etc.)
        self.data_batches: Dict[int, dict] = {}

        self.ptr = 0

    # ── Write ─────────────────────────────────────────────────────────────────

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
    ) -> None:
        self.rewards[step_idx] = rewards
        self.dones[step_idx]   = dones
        self.values[step_idx]  = values

        if old_cfm_loss is not None:
            self.old_cfm_loss[step_idx] = old_cfm_loss
            self.sigmas[step_idx]       = sigmas
            self.epsilons[step_idx]     = epsilons
            self.x0_latent[step_idx]    = x0_latent.cpu().numpy()
            self.cond_latent[step_idx]  = cond_latent.cpu().numpy()
            self.data_batches[step_idx] = data_batch

    def set_future_cond(self, step_idx: int, future_cond: np.ndarray) -> None:
        """Store actual future observation latent for chunk at step_idx.

        future_cond: VAE-encoded obs n_open_loop steps later, shape (B, C, T, H, W)
        """
        self.future_cond_latent[step_idx] = future_cond
        self.has_future_cond[step_idx]    = True

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_mini_batches(
        self,
        num_mini_batches: int,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Yield mini-batches for the FPO++ update.

        Only chunk-boundary steps with a valid future observation target are
        included.  Yields:
            adv, ret, old_cfm_loss, sigmas, epsilons,
            x0_latent, cond_latent, future_cond_latent, data_batch
        """
        chunk_steps = [s for s in sorted(self.data_batches.keys())
                       if self.has_future_cond[s]]
        if not chunk_steps:
            return

        indices = np.random.permutation(len(chunk_steps))
        mb_size = max(1, len(indices) // num_mini_batches)

        def _cat(arrays: list) -> torch.Tensor:
            return torch.tensor(np.concatenate(arrays, axis=0), dtype=torch.float32)

        for start in range(0, len(indices), mb_size):
            batch_steps = [chunk_steps[i] for i in indices[start: start + mb_size]]

            yield (
                _cat([advantages[s].reshape(-1, 1)         for s in batch_steps]),  # (mb*B, 1)
                _cat([returns[s].reshape(-1, 1)            for s in batch_steps]),  # (mb*B, 1)
                _cat([self.old_cfm_loss[s]                 for s in batch_steps]),  # (mb*B, N)
                _cat([self.sigmas[s]                       for s in batch_steps]),  # (mb*B, N)
                _cat([self.epsilons[s]                     for s in batch_steps]),  # (mb*B, N, C, H, W)
                _cat([self.x0_latent[s]                    for s in batch_steps]),  # (mb*B, C, T, H, W)
                _cat([self.cond_latent[s]                  for s in batch_steps]),  # (mb*B, C, T, H, W)
                _cat([self.future_cond_latent[s]           for s in batch_steps]),  # (mb*B, C, T, H, W)
                self.data_batches[batch_steps[0]],                                  # data_batch
            )
