#!/bin/bash
# Shared environment setup for verification/*/run_*.sh wrapper scripts.
#
# Usage (from a run_*.sh script, any suite depth):
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
#
# Sets up GPU/EGL rendering (RoboCasa/MuJoCo headless render), HF cache, and
# activates the project venv. Works both inside the Singularity container and
# on the bare host. Individual scripts may still export extra vars (e.g.
# CUDA_VISIBLE_DEVICES for a different GPU) *after* sourcing this file.

PROJECT_ROOT="/home/bilgehan.sakai/cosmos-policy"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}"

EGL_ICD_DIR="/tmp/singularity_egl_icd"
mkdir -p "$EGL_ICD_DIR"
if [ -f "/.singularity.d/libs/libEGL_nvidia.so.0" ]; then
    EGL_LIB="/.singularity.d/libs/libEGL_nvidia.so.0"
else
    EGL_LIB="/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"
fi
cat > "$EGL_ICD_DIR/10_nvidia.json" << JSONEOF
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "$EGL_LIB"
    }
}
JSONEOF
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_ICD_DIR/10_nvidia.json"

export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface}"
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"

cd "$PROJECT_ROOT"
source .venv/bin/activate
