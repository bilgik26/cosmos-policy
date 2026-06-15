# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset for new robocasa (bilgik26/robocasa v1.0.1) using LeRobot format.

Expected LeRobot dataset features:
  - observation.images.robot0_agentview_left   (video)
  - observation.images.robot0_agentview_right  (video)
  - observation.images.robot0_eye_in_hand      (video)
  - observation.state                           (9-dim: gripper_qpos(2)+eef_pos(3)+eef_quat(4))
  - action                                      (12-dim, first ACTION_DIM=7 used)

Dataset is stored as LeRobot v2.1 format:
  {ds_path}/
    meta/info.json         - dataset info (fps, features, total_episodes, ...)
    meta/tasks.jsonl       - task_index -> task description
    meta/episodes.jsonl    - per-episode metadata
    data/chunk-NNN/episode_NNNNNN.parquet  - scalar data per episode
    videos/chunk-NNN/{feature_key}/episode_NNNNNN.mp4  - video per camera per episode

Usage:
    ds_metas = [get_ds_meta(task=t, split="pretrain", source="human")
                for t in ATOMIC_TASK_DATASETS if ...]
    dataset = NewRoboCasaDataset(ds_metas=ds_metas, t5_text_embeddings_path="...")

    # Generate T5 embeddings first:
    python -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings \\
        --output_path /path/to/t5_embeddings.pkl
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.utils.utils import duplicate_array

# LeRobot feature key names (may need adjustment if actual dataset differs)
LEFT_IMG_KEY = "observation.images.robot0_agentview_left"
RIGHT_IMG_KEY = "observation.images.robot0_agentview_right"
WRIST_IMG_KEY = "observation.images.robot0_eye_in_hand"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

ACTION_DIM = 7    # First 7 dims of 12-dim action space (arm only, not mobile base)
PROPRIO_DIM = 9   # gripper_qpos(2) + eef_pos(3) + eef_quat(4)
FPS = 20          # Control frequency of new robocasa


def _read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_lerobot_meta(ds_path: Path):
    """Load LeRobot metadata: info, tasks, episodes."""
    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)
    tasks_list = _read_jsonl(ds_path / "meta" / "tasks.jsonl")
    tasks = {r["task_index"]: r["task"] for r in tasks_list}
    episodes = _read_jsonl(ds_path / "meta" / "episodes.jsonl")
    return info, tasks, episodes


def _get_video_path(ds_path: Path, info: dict, ep_idx: int, feature_key: str) -> Path:
    chunks_size = info.get("chunks_size", 1000)
    chunk = ep_idx // chunks_size
    return ds_path / "videos" / f"chunk-{chunk:03d}" / feature_key / f"episode_{ep_idx:06d}.mp4"


def _get_parquet_path(ds_path: Path, info: dict, ep_idx: int) -> Path:
    chunks_size = info.get("chunks_size", 1000)
    chunk = ep_idx // chunks_size
    return ds_path / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"


def _decode_video_frame(video_path: Path, timestamp: float) -> np.ndarray:
    """Decode a single video frame at the given timestamp.

    Returns: (H, W, 3) uint8 numpy array.
    """
    from lerobot.datasets.video_utils import decode_video_frames

    frames = decode_video_frames(str(video_path), [timestamp], tolerance_s=0.08, backend="pyav")
    # frames: (1, C, H, W) float32 [0,1]
    frame = frames[0]  # (C, H, W)
    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return frame_np


class NewRoboCasaDataset(Dataset):
    """
    PyTorch Dataset for the new robocasa (bilgik26/robocasa v1.0.1).

    Reads LeRobot-format datasets (parquet + MP4) and returns batches compatible
    with CosmosPolicyVideo2WorldModel, matching the output format of RoboCasaDataset.

    Key differences from the old RoboCasaDataset:
    - Reads from LeRobot format (parquet + MP4) instead of HDF5
    - Supports all 65 new atomic tasks (new task name conventions)
    - No rollout data support (demos only, for initial training)
    """

    def __init__(
        self,
        ds_metas: list,
        t5_text_embeddings_path: str = "",
        chunk_size: int = 32,
        final_image_size: int = 224,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
        stats_save_path: str = "",
        skip_missing_datasets: bool = True,
        # Accept (and ignore) params that bleed through from parent experiment configs
        # (e.g. LIBERODataset/RoboCasaDataset params merged in via Hydra defaults inheritance)
        normalize_images: bool = False,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        treat_demos_as_success_rollouts: bool = False,
        data_dir: str = "",
        use_jpeg_for_rollouts: bool = False,
        **kwargs,
    ):
        """
        Args:
            ds_metas: List of dataset metadata dicts from robocasa.utils.dataset_registry_utils.get_ds_meta().
                      Each dict has keys: "path", "task", "split", "source", "filter_key", "horizon".
            t5_text_embeddings_path: Path to pre-computed T5 embeddings pickle file.
                      Generate with: python -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings
            chunk_size: Number of actions in each action chunk.
            final_image_size: Target image size (square).
            use_image_aug: Apply image augmentations.
            use_stronger_image_aug: Apply stronger image augmentations.
            use_wrist_images: Include wrist camera images.
            use_third_person_images: Include left/right agent-view images.
            use_proprio: Include proprioceptive state.
            normalize_actions: Normalize actions to [-1, 1].
            normalize_proprio: Normalize proprio to [-1, 1].
            num_duplicates_per_image: Number of times to duplicate each image frame
                                      (must match tokenizer: WAN2.1 uses 4 images per latent frame).
            return_value_function_returns: If True, return value function returns. Not supported yet.
            gamma: Discount factor for value function returns.
            stats_save_path: Path to save/load dataset statistics JSON. Falls back to first dataset dir.
            skip_missing_datasets: If True, skip datasets whose path doesn't exist. If False, raise.
        """
        assert use_wrist_images or use_third_person_images, "Must use at least one camera type."

        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma

        # Each element: {
        #   "task": str,
        #   "actions": np.array (T, ACTION_DIM) float32,
        #   "proprio": np.array (T, PROPRIO_DIM) float32,
        #   "timestamps": list[float],
        #   "num_steps": int,
        #   "video_paths": {key: Path},
        # }
        self.episodes = []
        self.unique_commands = set()
        self.num_steps = 0

        # Raw data for statistics computation (kept in memory, actions + proprio only)
        _all_actions_raw = []
        _all_proprio_raw = []

        for ds_meta in tqdm(ds_metas, desc="Loading dataset metadata"):
            ds_path = Path(ds_meta["path"])
            if not ds_path.exists():
                if skip_missing_datasets:
                    print(f"[NewRoboCasaDataset] Warning: {ds_path} not found, skipping.")
                    continue
                raise FileNotFoundError(f"Dataset not found: {ds_path}")

            try:
                info, tasks_map, episodes_meta = _load_lerobot_meta(ds_path)
            except Exception as e:
                if skip_missing_datasets:
                    print(f"[NewRoboCasaDataset] Warning: failed to load {ds_path}: {e}, skipping.")
                    continue
                raise

            # Limit episodes by filter_key ("100_demos" → 100 episodes)
            max_eps = None
            filter_key = ds_meta.get("filter_key")
            if filter_key is not None:
                try:
                    max_eps = int(filter_key.split("_")[0])
                except (ValueError, IndexError):
                    pass

            num_eps_in_ds = len(episodes_meta)
            if max_eps is not None:
                num_eps_in_ds = min(num_eps_in_ds, max_eps)

            for ep_meta in episodes_meta[:num_eps_in_ds]:
                ep_idx = ep_meta["episode_index"]
                parquet_path = _get_parquet_path(ds_path, info, ep_idx)

                if not parquet_path.exists():
                    if skip_missing_datasets:
                        continue
                    raise FileNotFoundError(f"Parquet not found: {parquet_path}")

                try:
                    table = pq.read_table(str(parquet_path))
                except Exception as e:
                    if skip_missing_datasets:
                        print(f"[NewRoboCasaDataset] Warning: could not read {parquet_path}: {e}")
                        continue
                    raise

                col_names = table.schema.names

                # Actions: (T, full_action_dim) → take first ACTION_DIM
                actions_raw = np.array(table[ACTION_KEY].to_pylist(), dtype=np.float32)
                if actions_raw.ndim == 1:
                    actions_raw = actions_raw.reshape(1, -1)
                actions = actions_raw[:, :ACTION_DIM]

                # Proprio state: (T, full_state_dim) → take first PROPRIO_DIM
                state_raw = np.array(table[STATE_KEY].to_pylist(), dtype=np.float32)
                if state_raw.ndim == 1:
                    state_raw = state_raw.reshape(1, -1)
                proprio = state_raw[:, :PROPRIO_DIM]

                # Timestamps: (T,) in seconds (per-episode, starts from 0)
                timestamps = [float(t) for t in table["timestamp"].to_pylist()]

                # Task description from task_index
                task_indices = [int(i) for i in table["task_index"].to_pylist()]
                task_desc = tasks_map.get(task_indices[0], f"task_{task_indices[0]}")
                self.unique_commands.add(task_desc)

                num_steps = len(timestamps)
                if num_steps < 2:
                    continue

                # Video paths (may not exist yet if download is in progress)
                video_paths = {
                    "left": _get_video_path(ds_path, info, ep_idx, LEFT_IMG_KEY),
                    "right": _get_video_path(ds_path, info, ep_idx, RIGHT_IMG_KEY),
                    "wrist": _get_video_path(ds_path, info, ep_idx, WRIST_IMG_KEY),
                }

                self.episodes.append({
                    "task": task_desc,
                    "actions": actions,
                    "proprio": proprio,
                    "timestamps": timestamps,
                    "num_steps": num_steps,
                    "video_paths": video_paths,
                })
                self.num_steps += num_steps
                _all_actions_raw.append(actions)
                _all_proprio_raw.append(proprio)

        if len(self.episodes) == 0:
            raise RuntimeError(
                "No episodes loaded. Check that datasets are downloaded with: "
                "python -m robocasa.scripts.download_datasets --split pretrain --task_type atomic --source human"
            )

        print(f"[NewRoboCasaDataset] Loaded {len(self.episodes)} episodes, "
              f"{self.num_steps} steps, {len(self.unique_commands)} unique tasks.")

        # Build step-to-episode mapping
        self._build_step_index_mapping()

        # Compute/load dataset statistics
        stats_path = self._resolve_stats_path(stats_save_path, ds_metas)
        self.dataset_stats = self._load_or_compute_stats(
            stats_path, _all_actions_raw, _all_proprio_raw
        )

        # Normalize in-place
        if self.normalize_actions or self.normalize_proprio:
            for ep in self.episodes:
                if self.normalize_actions:
                    ep["actions"] = self._normalize(ep["actions"], "actions")
                if self.normalize_proprio:
                    ep["proprio"] = self._normalize(ep["proprio"], "proprio")

        # Load T5 embeddings
        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as f:
                self.t5_text_embeddings = pickle.load(f)
        else:
            self.t5_text_embeddings = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_step_index_mapping(self):
        self._step_to_episode = []  # index → (episode_idx, step_within_episode)
        for ep_idx, ep in enumerate(self.episodes):
            for s in range(ep["num_steps"]):
                self._step_to_episode.append((ep_idx, s))

    def _resolve_stats_path(self, stats_save_path: str, ds_metas: list) -> Path:
        if stats_save_path:
            return Path(stats_save_path)
        # Fall back to first available dataset path
        for ds_meta in ds_metas:
            p = Path(ds_meta["path"])
            if p.exists():
                return p / "new_robocasa_dataset_statistics.json"
        return Path("new_robocasa_dataset_statistics.json")

    def _load_or_compute_stats(
        self,
        stats_path: Path,
        all_actions: list,
        all_proprio: list,
    ) -> dict:
        if stats_path.exists():
            with open(stats_path) as f:
                raw = json.load(f)
            return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

        print("[NewRoboCasaDataset] Computing dataset statistics...")
        actions_cat = np.concatenate(all_actions, axis=0)
        proprio_cat = np.concatenate(all_proprio, axis=0)

        stats = {
            "actions_min": actions_cat.min(axis=0),
            "actions_max": actions_cat.max(axis=0),
            "actions_mean": actions_cat.mean(axis=0),
            "actions_std": actions_cat.std(axis=0),
            "actions_median": np.median(actions_cat, axis=0),
            "proprio_min": proprio_cat.min(axis=0),
            "proprio_max": proprio_cat.max(axis=0),
            "proprio_mean": proprio_cat.mean(axis=0),
            "proprio_std": proprio_cat.std(axis=0),
            "proprio_median": np.median(proprio_cat, axis=0),
        }

        # Save for next run
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=2)
            print(f"[NewRoboCasaDataset] Saved statistics to {stats_path}")
        except Exception as e:
            print(f"[NewRoboCasaDataset] Warning: could not save statistics: {e}")

        return stats

    def _normalize(self, arr: np.ndarray, key: str) -> np.ndarray:
        """Normalize arr to [-1, 1] using stored min/max statistics."""
        lo = self.dataset_stats[f"{key}_min"]
        hi = self.dataset_stats[f"{key}_max"]
        denom = hi - lo
        denom = np.where(denom == 0, 1.0, denom)  # avoid div-by-zero
        return (2.0 * (arr - lo) / denom - 1.0).astype(np.float32)

    def _decode_frame(self, ep: dict, camera: str, step_idx: int) -> np.ndarray:
        """Decode a single video frame. Returns (H, W, 3) uint8."""
        video_path = ep["video_paths"][camera]
        timestamp = ep["timestamps"][step_idx]
        return _decode_video_frame(video_path, timestamp)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> dict:
        ep_idx, step_idx = self._step_to_episode[idx % self.num_steps]
        ep = self.episodes[ep_idx]
        num_steps = ep["num_steps"]

        # Future frame index (clamped to episode boundary)
        future_step_idx = min(step_idx + self.chunk_size, num_steps - 1)

        # ---- Build action chunk ----
        remaining = num_steps - step_idx
        if remaining >= self.chunk_size:
            action_chunk = ep["actions"][step_idx : step_idx + self.chunk_size].copy()
        else:
            available = ep["actions"][step_idx:]
            padding = np.tile(ep["actions"][-1], (self.chunk_size - remaining, 1))
            action_chunk = np.concatenate([available, padding], axis=0)

        # ---- Proprio ----
        proprio = ep["proprio"][step_idx].copy()
        future_proprio = ep["proprio"][future_step_idx].copy()

        # ---- Images ----
        # Decode current and future frames from video
        curr_left = self._decode_frame(ep, "left", step_idx)
        curr_right = self._decode_frame(ep, "right", step_idx)
        curr_wrist = self._decode_frame(ep, "wrist", step_idx)
        fut_left = self._decode_frame(ep, "left", future_step_idx)
        fut_right = self._decode_frame(ep, "right", future_step_idx)
        fut_wrist = self._decode_frame(ep, "wrist", future_step_idx)

        # ---- Assemble image sequence (same layout as RoboCasaDataset) ----
        # Layout (matches cosmos_predict2_2b_480p_robocasa config state_t=11):
        # 0: blank, 1: curr proprio, 2: curr wrist, 3: curr left, 4: curr right,
        # 5: action, 6: future proprio, 7: future wrist, 8: future left, 9: future right, 10: value
        image_list = []
        latent_indices = {}
        seq_idx = 0

        # Slot 0: blank (1 frame, not duplicated)
        blank = np.zeros_like(curr_left)
        image_list.append(np.expand_dims(blank, 0))
        seq_idx += 1

        # Slot 1: current proprio (blank placeholder, num_duplicates frames)
        if self.use_proprio:
            image_list.append(duplicate_array(np.zeros_like(curr_left), self.num_duplicates_per_image))
            latent_indices["current_proprio_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 2: current wrist image
        if self.use_wrist_images:
            image_list.append(duplicate_array(curr_wrist, self.num_duplicates_per_image))
            latent_indices["current_wrist_image_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 3: current left (primary)
        if self.use_third_person_images:
            image_list.append(duplicate_array(curr_left, self.num_duplicates_per_image))
            latent_indices["current_image_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 4: current right (secondary)
        if self.use_third_person_images:
            image_list.append(duplicate_array(curr_right, self.num_duplicates_per_image))
            latent_indices["current_image2_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 5: action (blank placeholder)
        image_list.append(duplicate_array(np.zeros_like(curr_left), self.num_duplicates_per_image))
        latent_indices["action_latent_idx"] = seq_idx
        seq_idx += 1

        # Slot 6: future proprio (blank placeholder)
        if self.use_proprio:
            image_list.append(duplicate_array(np.zeros_like(curr_left), self.num_duplicates_per_image))
            latent_indices["future_proprio_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 7: future wrist
        if self.use_wrist_images:
            image_list.append(duplicate_array(fut_wrist, self.num_duplicates_per_image))
            latent_indices["future_wrist_image_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 8: future left
        if self.use_third_person_images:
            image_list.append(duplicate_array(fut_left, self.num_duplicates_per_image))
            latent_indices["future_image_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 9: future right
        if self.use_third_person_images:
            image_list.append(duplicate_array(fut_right, self.num_duplicates_per_image))
            latent_indices["future_image2_latent_idx"] = seq_idx
            seq_idx += 1

        # Slot 10: value (blank placeholder) — kept even without rollout data for model compatibility
        value_image = np.zeros_like(curr_left)
        image_list.append(duplicate_array(value_image, self.num_duplicates_per_image))
        latent_indices["value_latent_idx"] = seq_idx
        seq_idx += 1

        # Stack and preprocess: (T, H, W, 3) uint8 → (C, T, H, W) uint8/float32
        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=False,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # ---- T5 embeddings ----
        task_desc = ep["task"]
        if self.t5_text_embeddings is not None:
            # Try exact match first; fall back to lowercased no-period variant
            t5_emb = self.t5_text_embeddings.get(task_desc)
            if t5_emb is None:
                t5_emb = self.t5_text_embeddings.get(task_desc.lower().rstrip("."))
            if t5_emb is None:
                raise KeyError(
                    f"T5 embedding not found for task: {task_desc!r}. "
                    "Re-generate embeddings with save_new_robocasa_t5_text_embeddings.py."
                )
            t5_emb = torch.squeeze(t5_emb)
        else:
            raise ValueError(
                "t5_text_embeddings_path is required. "
                "Generate with: python -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings"
            )

        return {
            "video": images,
            "command": task_desc,
            "actions": action_chunk,
            "t5_text_embeddings": t5_emb,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": FPS,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio if self.use_proprio else np.zeros(PROPRIO_DIM, dtype=np.float32),
            "future_proprio": future_proprio if self.use_proprio else np.zeros(PROPRIO_DIM, dtype=np.float32),
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            # Latent sequence indices for the model
            "action_latent_idx": latent_indices["action_latent_idx"],
            "value_latent_idx": latent_indices["value_latent_idx"],
            "current_proprio_latent_idx": latent_indices.get("current_proprio_latent_idx", -1),
            "current_wrist_image_latent_idx": latent_indices.get("current_wrist_image_latent_idx", -1),
            "current_image_latent_idx": latent_indices.get("current_image_latent_idx", -1),
            "current_image2_latent_idx": latent_indices.get("current_image2_latent_idx", -1),
            "future_proprio_latent_idx": latent_indices.get("future_proprio_latent_idx", -1),
            "future_wrist_image_latent_idx": latent_indices.get("future_wrist_image_latent_idx", -1),
            "future_image_latent_idx": latent_indices.get("future_image_latent_idx", -1),
            "future_image2_latent_idx": latent_indices.get("future_image2_latent_idx", -1),
            "value_function_return": float("-100"),  # Placeholder; no rollout data
        }
