#!/bin/bash
# Download 10 representative atomic tasks, generate T5 embeddings, and start training.
set -e

VENV="/workspace/.venv/bin/python"
DS_BASE="/workspace/robocasa/datasets"
T5_PATH="${DS_BASE}/new_robocasa_t5_embeddings.pkl"
STATS_PATH="${DS_BASE}/new_robocasa_dataset_statistics.json"

TASKS=(
  PickPlaceCounterToCabinet
  PickPlaceCabinetToCounter
  TurnOffMicrowave
  TurnOnMicrowave
  OpenCabinet
  CloseCabinet
  TurnOnSinkFaucet
  TurnOffSinkFaucet
  OpenFridge
  CloseFridge
)

echo "=== Step 1: Download datasets ==="
for TASK in "${TASKS[@]}"; do
  echo "--- Downloading ${TASK} ---"
  $VENV -c "
from robocasa.scripts.download_datasets import download_datasets
download_datasets(split=['pretrain'], tasks=['${TASK}'], source=['human'], overwrite=False)
" || echo "WARNING: ${TASK} download failed or skipped"
done

echo ""
echo "=== Step 2: Generate T5 embeddings ==="
$VENV -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings \
  --output_path "${T5_PATH}" \
  --split pretrain \
  --source human \
  --task_type atomic

echo ""
echo "=== Step 3: Start training ==="
export NEW_ROBOCASA_T5_EMBEDDINGS_PATH="${T5_PATH}"
export NEW_ROBOCASA_DATASET_STATS_PATH="${STATS_PATH}"
export WANDB_PROJECT="cosmos-policy"

torchrun --nproc_per_node=1 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py \
  -- \
  experiment="cosmos_predict2_2b_480p_new_robocasa_pretrain_human" \
  trainer.max_iter=2000 \
  trainer.logging_iter=10 \
  trainer.grad_accum_iter=4 \
  dataloader_train.batch_size=2 \
  dataloader_train.num_workers=4 \
  checkpoint.save_iter=500 \
  job.name="new_robocasa_10tasks_2000iter"
