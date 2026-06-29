# Cosmos Policy セットアップガイド（Singularity版）

このサーバーでは Docker が使用できないため、Singularity コンテナを使用してセットアップする。

## 前提条件

- Singularity CE 4.x がインストール済み
- NVIDIA GPU + CUDA 12.8 ドライバ
- `~/singularity/` スクリプト群が利用可能
- HuggingFace アカウント＋以下のモデルへのアクセス申請・承認済み
  - `nvidia/Cosmos-Predict2-2B-Video2World`
  - `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`

---

## Step 1: Singularity コンテナの作成

`~/singularity/cosmos_policy.def` を使って SIF ファイルをビルドする。
ベースイメージは Dockerfile (`docker/Dockerfile`) と同等の `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`。

```bash
singularity build --fakeroot \
    /mnt/data/bilgehan.sakai/singularity/sif/cosmos-policy.sif \
    ~/singularity/cosmos_policy.def
```

- ビルド所要時間: 約10〜20分（Docker Hub からのイメージ取得を含む）
- 出力先: `/mnt/data/bilgehan.sakai/singularity/sif/cosmos-policy.sif`（約5.7GB）
- `~/singularity/project_env.sh` はプロジェクト名と同名の SIF（例: `cosmos-policy.sif`）が存在すれば自動的にそちらを使用する

コンテナに含まれるもの:
- Ubuntu 24.04 + CUDA 12.8.1 + cuDNN
- cmake, ffmpeg, git, git-lfs, libgl1, libegl1-mesa-dev 等
- uv 0.11+, just 1.42.4

---

## Step 2: Python 仮想環境のセットアップ

```bash
cd ~/
bash singularity/setup_uv_project.sh cosmos-policy "" 3.10
```

- 仮想環境の作成先: `~/cosmos-policy/.venv`
- キャッシュ: `/mnt/data/bilgehan.sakai/cosmos-policy/cache/`
- `pyproject.toml` が既に存在するため `uv init` はスキップされる

---

## Step 3: 依存関係のインストール

以降のコマンドはすべてコンテナ内で実行する。
`run_singularity` は以下の singularity exec ラッパーを指す（後述の「コンテナ実行関数」参照）。

```bash
# コンテナ内で実行
cd ~/cosmos-policy
source .venv/bin/activate
uv sync --extra cu128 --group robocasa --python 3.10
```

インストールされる主なパッケージ:
- `torch==2.7.0+cu128`
- `flash-attn==2.7.3`, `transformer-engine==2.2.0`, `natten==0.21.0`
- `robosuite==1.5.1`, `mujoco==3.2.6`, `draccus`

**注意**: `uv pip install -e robocasa-cosmos-policy` 実行後に numba が 0.56.4 にダウングレードされ、NumPy 2.x と非互換になる。後述の Step 4 で修正する。

---

## Step 4: RoboCasa のインストールとアセットのダウンロード

```bash
# コンテナ内で実行（~/cosmos-policy ディレクトリ）
source .venv/bin/activate

# RoboCasa リポジトリのクローンとインストール
git clone https://github.com/moojink/robocasa-cosmos-policy.git
uv pip install -e robocasa-cosmos-policy

# numba を NumPy 2.x 互換バージョンにアップグレード（重要）
uv pip install "numba>=0.61.0"

# Kitchen アセットのダウンロード（約5GB、確認プロンプトに "y" を入力）
echo "y" | python robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py

# プライベートマクロファイルのセットアップ
python robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
```

- ダウンロード内容: fixtures (~472MB), objaverse objects (~2.1GB), textures, kitchen layouts
- ダウンロード先: `robocasa-cosmos-policy/robocasa/models/assets/`
- マクロファイル: `robocasa-cosmos-policy/robocasa/macros_private.py` が生成される

---

## Step 5: 評価の実行

### コンテナ実行の基本コマンド

```bash
GLOBAL_PYTHONUSERBASE="/mnt/data/bilgehan.sakai/cache/python_local"
SIF_PATH="/mnt/data/bilgehan.sakai/singularity/sif/cosmos-policy.sif"
PROJECT_ROOT="$HOME/cosmos-policy"
UV_PROJECT_DIR="$HOME/cosmos-policy/.venv"
HOST_CACHE_ROOT="/mnt/data/bilgehan.sakai/cosmos-policy/cache"

singularity exec --nv \
    --bind "${HOST_CACHE_ROOT}:${HOST_CACHE_ROOT}" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --bind "${GLOBAL_PYTHONUSERBASE}:${GLOBAL_PYTHONUSERBASE}" \
    --env "PYTHONUSERBASE=${GLOBAL_PYTHONUSERBASE}" \
    --env "HF_HOME=${HOST_CACHE_ROOT}/huggingface" \
    --env "HF_TOKEN=<YOUR_HF_TOKEN>" \
    --env "MUJOCO_GL=egl" \
    --env "PYOPENGL_PLATFORM=egl" \
    --env "__EGL_VENDOR_LIBRARY_FILENAMES=${PROJECT_ROOT}/cosmos_policy/experiments/robot/libero/10_nvidia.json" \
    "$SIF_PATH" \
    bash -c "cd '$PROJECT_ROOT' && source '$UV_PROJECT_DIR/bin/activate' && <COMMAND>"
```

**重要な環境変数:**
- `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl`: ヘッドレス EGL レンダリングを有効化
- `__EGL_VENDOR_LIBRARY_FILENAMES`: NVIDIA EGL ドライバを明示的に指定（これなしだと `Cannot initialize a EGL device display` エラーが発生する）
- `--nv`: NVIDIA GPU ライブラリをコンテナ内にマウント（`--fakeroot` は使わない）

### RoboCasa 評価コマンド

```bash
singularity exec --nv \
    --bind "/mnt/data/bilgehan.sakai/cosmos-policy/cache:/mnt/data/bilgehan.sakai/cosmos-policy/cache" \
    --bind "$HOME/cosmos-policy:$HOME/cosmos-policy" \
    --bind "$HOME:$HOME" \
    --bind "/mnt/data/bilgehan.sakai/cache/python_local:/mnt/data/bilgehan.sakai/cache/python_local" \
    --env "PYTHONUSERBASE=/mnt/data/bilgehan.sakai/cache/python_local" \
    --env "HF_HOME=/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface" \
    --env "HF_TOKEN=<YOUR_HF_TOKEN>" \
    --env "MUJOCO_GL=egl" \
    --env "PYOPENGL_PLATFORM=egl" \
    --env "__EGL_VENDOR_LIBRARY_FILENAMES=$HOME/cosmos-policy/cosmos_policy/experiments/robot/libero/10_nvidia.json" \
    /mnt/data/bilgehan.sakai/singularity/sif/cosmos-policy.sif \
    bash -c "
        cd '$HOME/cosmos-policy'
        source '.venv/bin/activate'
        python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
            --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
            --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
            --config_file cosmos_policy/config/config.py \
            --use_wrist_image True \
            --num_wrist_images 1 \
            --use_proprio True \
            --normalize_proprio True \
            --unnormalize_actions True \
            --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
            --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
            --trained_with_image_aug True \
            --chunk_size 32 \
            --num_open_loop_steps 16 \
            --task_name TurnOffMicrowave \
            --num_trials_per_task 50 \
            --run_id_note chkpt45000--5stepAct--seed195--deterministic \
            --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
            --seed 195 \
            --randomize_seed False \
            --deterministic True \
            --use_variance_scale False \
            --use_jpeg_compression True \
            --flip_images True \
            --num_denoising_steps_action 5 \
            --num_denoising_steps_future_state 1 \
            --num_denoising_steps_value 1 \
            --data_collection False
    "
```

初回実行時はチェックポイント（`nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`）が HuggingFace から自動ダウンロードされる。
ダウンロード先: `/mnt/data/bilgehan.sakai/cosmos-policy/cache/huggingface/`

---

## 評価設定の説明

### 主要な引数

| 引数 | 値 | 説明 |
|---|---|---|
| `--config` | `cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference` | モデル設定名（後述） |
| `--ckpt_path` | `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B` | HuggingFace のチェックポイント |
| `--task_name` | `TurnOffMicrowave` 等 | 実行するタスク名 |
| `--num_trials_per_task` | `50` | 1タスクあたりの試行回数 |
| `--seed` | `195` | 再現性のためのシード（論文では 195, 196, 197 を使用） |
| `--deterministic` | `True` | 決定論的な実行（再現性確保） |
| `--num_denoising_steps_action` | `5` | アクションのデノイジングステップ数 |
| `--chunk_size` | `32` | アクションチャンクサイズ |
| `--num_open_loop_steps` | `16` | オープンループ実行ステップ数 |

### 設定ファイルの場所

**モデル・学習設定:**
`cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`

- `cosmos_predict2_2b_480p_robocasa_50_demos_per_task`（L.228）: 学習設定
  - ベースモデル構造、データローダー、バッチサイズ (`25`)、学習データパス等
- `cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference`（L.266）: 推論設定
  - 上記学習設定を継承し、推論用 SDE パラメータ（`sigma_max=80, sigma_min=4`）を設定
  - `--config` 引数で指定する名前がこれに対応
- `robocasa_50_demos_per_task_dataset`（L.206）: データセット設定
  - `chunk_size=32`, `use_image_aug=True`, `gamma=0.99` 等

**評価パラメータのデータクラス:**
`cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py`

- `PolicyEvalConfig`（L.149）: 全評価パラメータの定義
  - `task_name: str = "PnPCounterToCab"` がデフォルト
  - `--task_name TurnOffMicrowave` のようにコマンドライン引数で上書き可能
  - `draccus` ライブラリがデータクラスのフィールドを自動的に CLI 引数に変換する

### 利用可能なタスク一覧

`TASK_MAX_STEPS`（run_robocasa_eval.py L.113〜）に定義されている24タスク:
`TurnOffStove`, `TurnOffSinkFaucet`, `TurnOffMicrowave`, `PnPCounterToCab`,
`CoffeeSetupMug`, `CoffeeServeMug`, `CoffeePressButton`, `TurnOnStove` など

---

## 評価結果の確認

ログと動画は `cosmos_policy/experiments/robot/robocasa/logs/` に保存される:

```
logs/
├── ENV_EVAL-<TASK>-cosmos-<DATE>--<RUN_ID_NOTE>.txt   # 集計結果
└── rollout_data/
    └── <TASK>--<DATE>/
        ├── <DATE>--episode=0--success=True--task=<...>.mp4
        ├── <DATE>--with_future_img--episode=0--success=True--task=<...>.mp4
        └── ...
```

各エピソードについて通常映像と future prediction 付き映像の2種類が保存される。

---

## 実績

| タスク | 成功率 | 試行数 | 平均ステップ数 |
|---|---|---|---|
| TurnOffMicrowave | 100% (50/50) | 50 | 224.8 |

論文の結果（seed 195, 196, 197 の平均）と一致。

---

## トラブルシューティング

### `Cannot initialize a EGL device display`

`__EGL_VENDOR_LIBRARY_FILENAMES` が設定されていない場合に発生。
`cosmos_policy/experiments/robot/libero/10_nvidia.json` を指定すること（`--nv` フラグで `/singularity.d/libs/libEGL_nvidia.so.0` がマウントされる）。

### `numba.core.pythonapi ... _ARRAY_API not found`

`uv pip install -e robocasa-cosmos-policy` 後に numba が 0.56.4 にダウングレードされ NumPy 2.x と非互換になる。
`uv pip install "numba>=0.61.0"` で解決。

### `GatedRepoError: 401 Client Error`

`HF_TOKEN` 環境変数が未設定、またはモデルへのアクセス申請が未承認。
HuggingFace で `nvidia/Cosmos-Predict2-2B-Video2World` と `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B` へのアクセスをリクエストし、承認後にトークンを設定すること。

### `--fakeroot` 使用時の EGL エラー

`--fakeroot` を付けると NVIDIA EGL の初期化が失敗する。
評価実行時は `--fakeroot` を外し `--nv` のみを使用すること。
（`setup_uv_project.sh` 等の環境構築ステップは `--fakeroot` を使用して問題ない。）
