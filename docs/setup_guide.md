# Cosmos Policy セットアップガイド

## 前提条件

- NVIDIA GPU + CUDA 12.8 ドライバ
- Docker（NVIDIA Container Toolkit 込み）
- HuggingFace アカウント＋以下のモデルへのアクセス申請・承認済み
  - `nvidia/Cosmos-Predict2-2B-Video2World`
  - `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`

---

## Step 1: Docker イメージのビルド

プロジェクトルートから実行する。

```bash
cd ~/cosmos-policy
docker build -t cosmos-policy docker
```

- ビルド所要時間: 約5〜10分
- ベースイメージ: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
- `uv 0.8.12`、`just 1.42.4` が含まれる

---

## Step 2: Python 仮想環境のセットアップ

**コンテナ起動コマンドの説明（Step 3 以降で共通）:**

```bash
docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  --gpus all \
  --ipc=host \
  --rm \
  -w /workspace \
  cosmos-policy \
  bash -c "source .venv/bin/activate && <COMMAND>"
```

**重要なマウント:**
- `-v $HOME/.local:/home/ubuntu/.local`: **必須**。`.venv/bin/python` が `~/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/bin/python3.10` へのシンボリックリンクになるため、このマウントがないとコンテナ再起動時に venv が破損する。
- `-v $HOME/.cache:/home/ubuntu/.cache`: HuggingFace キャッシュ等の再利用に使用。

**重要な環境変数:**
- `HOST_USER_NAME=ubuntu`: コンテナ内ユーザーをホストと同じ `ubuntu` に統一する。省略するとデフォルトの `cosmos` ユーザーが作成され、`/home/cosmos` への `.local` マウントが競合してエラーになる。
- `HOST_USER_ID` / `HOST_GROUP_ID`: ファイルの所有権をホストユーザーに合わせる。

**仮想環境のセットアップ（初回のみ）:**

```bash
docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  --gpus all --ipc=host --rm -w /workspace \
  cosmos-policy \
  bash -c "uv sync --extra cu128 --group robocasa --python 3.10"
```

インストールされる主なパッケージ:
- `torch==2.7.0+cu128`
- `flash-attn==2.7.3`, `transformer-engine==2.2.0+cu128.torch27`, `natten==0.21.0+cu128.torch27`
- `robosuite==1.5.1`, `mujoco==3.2.6`, `draccus`

---

## Step 3: RoboCasa のインストールとアセットのダウンロード

```bash
# リポジトリのクローン（ホスト上で実行）
cd ~/cosmos-policy
git clone https://github.com/moojink/robocasa-cosmos-policy.git
```

```bash
# コンテナ内で実行
docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  --gpus all --ipc=host --rm -w /workspace \
  cosmos-policy \
  bash -c "
    source .venv/bin/activate

    # robocasa-cosmos-policy のインストール
    uv pip install -e robocasa-cosmos-policy

    # numba を NumPy 2.x 互換バージョンにアップグレード（重要）
    # robocasa のインストールで numba が 0.56.4 にダウングレードされ NumPy 2.x と非互換になるため
    uv pip install 'numba>=0.61.0'

    # Kitchen アセットのダウンロード（約5GB）
    echo 'y' | python robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py

    # プライベートマクロファイルのセットアップ
    python robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
  "
```

ダウンロード内容と所要時間（目安）:
- `textures.zip`: ~538MB
- `fixtures.zip`: ~472MB
- `objaverse.zip`: ~2.1GB
- `generative_textures.zip`: ~593MB

ダウンロード先: `robocasa-cosmos-policy/robocasa/models/assets/`  
マクロファイル: `robocasa-cosmos-policy/robocasa/macros_private.py` が生成される

**動作確認:**

```bash
docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  --gpus all --ipc=host --rm -w /workspace \
  cosmos-policy \
  bash -c "
    source .venv/bin/activate
    python -c '
import torch, numba, numpy, robosuite, mujoco
print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())
print(\"numba:\", numba.__version__)
print(\"numpy:\", numpy.__version__)
print(\"robosuite:\", robosuite.__version__)
print(\"mujoco:\", mujoco.__version__)
'
    python -c 'import cosmos_policy; print(\"cosmos_policy: OK\")'
  "
```

期待される出力:
```
torch: 2.7.0+cu128 cuda: True
numba: 0.65.1
numpy: 2.2.6
robosuite: 1.5.1
mujoco: 3.2.6
cosmos_policy: OK
```

---

## Step 4: 評価の実行

### RoboCasa 評価コマンド

```bash
docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  -e HF_HOME=$HOME/.cache/huggingface \
  -e HF_TOKEN=<YOUR_HF_TOKEN> \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/cosmos_policy/experiments/robot/libero/10_nvidia.json \
  --gpus all --ipc=host --rm -w /workspace \
  cosmos-policy \
  bash -c "
    source .venv/bin/activate
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
ダウンロード先: `$HOME/.cache/huggingface/`

**重要な環境変数:**
- `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl`: ヘッドレス EGL レンダリングを有効化
- `__EGL_VENDOR_LIBRARY_FILENAMES`: NVIDIA EGL ドライバを明示的に指定（これなしだと `Cannot initialize a EGL device display` エラーが発生する）

---

## 評価設定の説明

### 主要な引数

| 引数 | 値 | 説明 |
|---|---|---|
| `--config` | `cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference` | モデル設定名 |
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

### `.venv/bin/python` のシンボリックリンクが切れる

コンテナ再起動後に `Broken symlink at .venv/bin/python3` エラーが出る場合、`$HOME/.local` がマウントされていない。
`-v $HOME/.local:/home/ubuntu/.local` を必ず付けること。

`.venv/bin/python` は `~/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/bin/python3.10` へのシンボリックリンクになっており、このディレクトリがコンテナ外に永続化されていないと起動ごとに venv が壊れる。

### `usermod: directory /home/cosmos exists` でコンテナが起動しない

`HOST_USER_NAME` を指定せずに `-v $HOME/.local:/home/cosmos/.local` を渡すと、Docker がマウント先ディレクトリ `/home/cosmos` を先に作成し、エントリポイントの `usermod` が失敗する。
`-e HOST_USER_NAME=ubuntu` を必ず指定すること。

### `numba.core.pythonapi ... _ARRAY_API not found`

`uv pip install -e robocasa-cosmos-policy` 後に numba が 0.56.4 にダウングレードされ NumPy 2.x と非互換になる。
`uv pip install "numba>=0.61.0"` で解決（0.65.1 以上に更新される）。

### `Cannot initialize a EGL device display`

`__EGL_VENDOR_LIBRARY_FILENAMES` が設定されていない場合に発生。
`-e __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/cosmos_policy/experiments/robot/libero/10_nvidia.json` を指定すること。

### `GatedRepoError: 401 Client Error`

`HF_TOKEN` 環境変数が未設定、またはモデルへのアクセス申請が未承認。
HuggingFace で `nvidia/Cosmos-Predict2-2B-Video2World` と `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B` へのアクセスをリクエストし、承認後に `-e HF_TOKEN=<YOUR_HF_TOKEN>` を設定すること。
