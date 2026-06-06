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
Experiment configs for training Cosmos Policy on new robocasa (bilgik26/robocasa v1.0.1).

Dataset: LeRobot format (parquet + MP4), 65 atomic tasks, pretrain/target splits.

Quick start:

  # 1. Download datasets (inside Docker container with venv activated):
  python -m robocasa.scripts.download_datasets --split pretrain --task_type atomic --source human

  # 2. Pre-compute T5 text embeddings:
  python -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings \\
    --output_path /workspace/robocasa/datasets/new_robocasa_t5_embeddings.pkl

  # 3. Train (8-GPU):
  torchrun --nproc_per_node=8 -m cosmos_policy.scripts.train \\
    --config=cosmos_policy/config/config_v2.py \\
    -- experiment="cosmos_predict2_2b_480p_new_robocasa_pretrain_human" \\
    trainer.grad_accum_iter=4

Environment variables:
  NEW_ROBOCASA_T5_EMBEDDINGS_PATH  Path to the T5 embeddings pickle (default: see below)
  NEW_ROBOCASA_DATASET_STATS_PATH  Path to dataset statistics JSON (default: auto-computed)
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.new_robocasa_dataset import NewRoboCasaDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Default dataset base path matches robocasa.macros.DATASET_BASE_PATH fallback:
#   /workspace/robocasa/robocasa/../datasets/v1.0  →  /workspace/robocasa/datasets/v1.0
_DEFAULT_DS_BASE = "/workspace/robocasa/datasets"

NEW_ROBOCASA_DS_BASE = os.environ.get("NEW_ROBOCASA_DS_BASE", _DEFAULT_DS_BASE)
NEW_ROBOCASA_T5_EMBEDDINGS_PATH = os.environ.get(
    "NEW_ROBOCASA_T5_EMBEDDINGS_PATH",
    os.path.join(NEW_ROBOCASA_DS_BASE, "new_robocasa_t5_embeddings.pkl"),
)
NEW_ROBOCASA_DATASET_STATS_PATH = os.environ.get(
    "NEW_ROBOCASA_DATASET_STATS_PATH",
    os.path.join(NEW_ROBOCASA_DS_BASE, "new_robocasa_dataset_statistics.json"),
)


def _get_all_atomic_pretrain_human_ds_metas():
    """Return ds_metas for all available atomic pretrain/human datasets."""
    try:
        from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
        from robocasa.utils.dataset_registry_utils import get_ds_meta
    except ImportError:
        return []

    metas = []
    for task in ATOMIC_TASK_DATASETS.keys():
        meta = get_ds_meta(task=task, split="pretrain", source="human")
        if meta is not None:
            metas.append(meta)
    return metas


# ---------------------------------------------------------------------------
# Dataset LazyCall
# ---------------------------------------------------------------------------
new_robocasa_all_atomic_pretrain_human_dataset = L(NewRoboCasaDataset)(
    ds_metas=_get_all_atomic_pretrain_human_ds_metas(),
    t5_text_embeddings_path=NEW_ROBOCASA_T5_EMBEDDINGS_PATH,
    chunk_size=32,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_actions=True,
    normalize_proprio=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    return_value_function_returns=False,
    gamma=0.99,
    stats_save_path=NEW_ROBOCASA_DATASET_STATS_PATH,
    skip_missing_datasets=True,
)

# ---------------------------------------------------------------------------
# Training experiment config
#
# Model layout (state_t=11, matches cosmos_predict2_2b_480p_robocasa_50_demos_per_task):
#   Latent frame 0:  blank (1 frame)
#   Latent frame 1:  current proprio   (4 images)
#   Latent frame 2:  current wrist     (4 images)
#   Latent frame 3:  current left      (4 images)
#   Latent frame 4:  current right     (4 images)
#   Latent frame 5:  action            (4 images)
#   Latent frame 6:  future proprio    (4 images)
#   Latent frame 7:  future wrist      (4 images)
#   Latent frame 8:  future left       (4 images)
#   Latent frame 9:  future right      (4 images)
#   Latent frame 10: value             (4 images)
#
# chunk_duration = 1 blank + 10 × 4 images = 41
# min/max_num_conditional_frames = 5  (blank + 4 conditioning slots)
# ---------------------------------------------------------------------------
cosmos_predict2_2b_480p_new_robocasa_pretrain_human = LazyDict(
    dict(
        defaults=[
            # Inherit base Cosmos Policy config (libero has the full default set)
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                    ),
                ),
                # 11 latent temporal slots: blank + proprio/wrist/left/right (cur+fut) + action + value
                state_t=11,
                min_num_conditional_frames=5,  # blank + 4 conditioning (proprio, wrist, left, right)
                max_num_conditional_frames=5,
                sigma_conditional=0.0,
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    # 1 blank + 40 images (4×10 slots)
                    chunk_duration=41,
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            cycle_lengths=[30000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"
            ),
            load_training_state=False,
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            # forkserver avoids the fork-after-CUDA deadlock: workers are spawned
            # by a pre-fork server process that has no CUDA context, so they can
            # safely initialize CUDA/pyarrow/PyAV independently.
            multiprocessing_context="spawn",
            persistent_workers=True,
            pin_memory=True,
            dataset=new_robocasa_all_atomic_pretrain_human_dataset,
            sampler=L(DistributedSampler)(
                dataset=new_robocasa_all_atomic_pretrain_human_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        upload_reproducible_setup=False,
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_new_robocasa_pretrain_human",
        ),
    )
)

# Inference-only variant (tighter sigma range, no training noise)
cosmos_predict2_2b_480p_new_robocasa_pretrain_human__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_new_robocasa_pretrain_human",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_new_robocasa_pretrain_human__inference",
        ),
    )
)

# ---------------------------------------------------------------------------
# Hydra ConfigStore registration
# This runs automatically when import_all_modules_from_package() imports this file.
# ---------------------------------------------------------------------------
_cs = ConfigStore.instance()
for _cfg in [
    cosmos_predict2_2b_480p_new_robocasa_pretrain_human,
    cosmos_predict2_2b_480p_new_robocasa_pretrain_human__inference,
]:
    _experiment_name = _cfg["job"]["name"]
    log.info(f"Registering experiment: {_experiment_name}")
    _cs.store(
        group="experiment",
        package="_global_",
        name=_experiment_name,
        node=_cfg,
    )
