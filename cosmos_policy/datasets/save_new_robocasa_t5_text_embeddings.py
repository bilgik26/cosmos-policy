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
Pre-compute T5 text embeddings for all new robocasa atomic tasks (LeRobot format).

Usage:
    # All atomic pretrain tasks (human source):
    python -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings \\
        --output_path /workspace/robocasa/datasets/new_robocasa_t5_embeddings.pkl

    # Dry-run: print all unique task descriptions without computing embeddings:
    python -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings --dryrun

How task descriptions are collected:
    For each downloaded dataset, we read meta/tasks.jsonl and collect all unique
    task description strings. This handles the new robocasa "First letter capital + period"
    format (e.g. "Put the can in the cabinet.") as used in the LeRobot datasets.

The generated pickle maps: str (task description) -> torch.Tensor (1, 512, 1024) bfloat16
"""

import argparse
import json
import os
import pickle
from pathlib import Path

from tqdm import tqdm


def _collect_task_descriptions(ds_metas: list, verbose: bool = True) -> list:
    """Scan downloaded datasets and collect all unique task description strings."""
    unique_descriptions = set()
    missing = []

    for ds_meta in tqdm(ds_metas, desc="Scanning datasets", disable=not verbose):
        ds_path = Path(ds_meta["path"])
        tasks_jsonl = ds_path / "meta" / "tasks.jsonl"
        if not tasks_jsonl.exists():
            missing.append(str(ds_path))
            continue
        with open(tasks_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    unique_descriptions.add(record["task"])

    if missing and verbose:
        print(f"Warning: {len(missing)} datasets not found (not yet downloaded?):")
        for p in missing[:5]:
            print(f"  {p}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    return sorted(unique_descriptions)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute T5 embeddings for new robocasa tasks.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/robocasa/datasets/new_robocasa_t5_embeddings.pkl",
        help="Output path for the T5 embeddings pickle file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Dataset split to collect task descriptions from.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="human",
        choices=["human", "mg"],
        help="Dataset source to collect task descriptions from.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="atomic",
        choices=["atomic", "composite", "all"],
        help="Task type to collect.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print unique task descriptions without computing embeddings.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip computing embeddings for commands already in the output file.",
    )
    args = parser.parse_args()

    # Import robocasa task registry
    from robocasa.utils.dataset_registry import (
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
    )
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    if args.task_type == "atomic":
        task_names = list(ATOMIC_TASK_DATASETS.keys())
    elif args.task_type == "composite":
        task_names = list(COMPOSITE_TASK_DATASETS.keys())
    else:
        task_names = list(ATOMIC_TASK_DATASETS.keys()) + list(COMPOSITE_TASK_DATASETS.keys())

    # Build ds_metas list
    ds_metas = []
    for task in task_names:
        meta = get_ds_meta(task=task, split=args.split, source=args.source)
        if meta is not None:
            ds_metas.append(meta)

    print(f"Found {len(ds_metas)} registered datasets for {args.split}/{args.source}/{args.task_type}.")

    # Collect unique task descriptions from downloaded datasets
    unique_descriptions = _collect_task_descriptions(ds_metas, verbose=True)
    print(f"\nFound {len(unique_descriptions)} unique task descriptions:")
    for desc in unique_descriptions:
        print(f"  {desc!r}")

    if args.dryrun:
        print("\nDry-run mode: not computing embeddings.")
        return

    if not unique_descriptions:
        print("No task descriptions found. Download datasets first:")
        print("  python -m robocasa.scripts.download_datasets --split pretrain --task_type atomic --source human")
        return

    # Load existing embeddings if output file exists and skip_existing
    existing_embeddings = {}
    output_path = Path(args.output_path)
    if args.skip_existing and output_path.exists():
        with open(output_path, "rb") as f:
            existing_embeddings = pickle.load(f)
        print(f"\nLoaded {len(existing_embeddings)} existing embeddings from {output_path}")

    # Filter out already-computed commands
    to_compute = [d for d in unique_descriptions if d not in existing_embeddings]
    print(f"Computing embeddings for {len(to_compute)} new commands...")

    if to_compute:
        from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings
        new_embeddings = generate_t5_embeddings(to_compute)
        existing_embeddings.update(new_embeddings)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(existing_embeddings, f)
    print(f"\nSaved {len(existing_embeddings)} T5 embeddings to {output_path}")


if __name__ == "__main__":
    main()
