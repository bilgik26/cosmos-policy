# Cosmos Policy セットアップガイド（新 RoboCasa 対応版）

> **対象リポジトリ:** `bilgik26/robocasa`（v1.0.1）  
> 旧リポジトリ（`moojink/robocasa-cosmos-policy`）向けの手順は `docs/setup_guide.md` を参照。  
> Docker イメージのビルド（Step 1）と Python 仮想環境のセットアップ（Step 2）は旧ガイドと共通。

---

## 前提条件

- NVIDIA GPU + CUDA 12.8 ドライバ
- Docker（NVIDIA Container Toolkit 込み）
- HuggingFace アカウント＋以下のモデルへのアクセス申請・承認済み
  - `nvidia/Cosmos-Predict2-2B-Video2World`
  - `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`
- `sudo` 権限（ユーザーが `docker` グループに属していない場合）

---

## Step 1: Docker イメージのビルド

`docs/setup_guide.md` の Step 1 と同じ。以下を実行する。

```bash
cd ~/cosmos-policy
sudo docker build -t cosmos-policy docker
```

ビルド所要時間: 約5〜10分。

> **`sudo` について:** ユーザーが `docker` グループに属していない場合、全 `docker` コマンドに `sudo` が必要。以降のコマンドでは `sudo docker` と表記する。

---

## Step 2: Python 仮想環境のセットアップ

`docs/setup_guide.md` の Step 2 と同じ。以下を実行する。

```bash
sudo docker run \
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

インストールされる主なパッケージ（Step 2 終了時点）:
- `torch==2.7.0+cu128`
- `robosuite==1.5.1`（後の Step 3 で GitHub 版に差し替える）
- `mujoco==3.2.6`（後の Step 3 で 3.3.1 に更新される）
- `numba==0.61.2`（後の Step 3 で 0.63.1 に差し替える）

---

## Step 3: 新 RoboCasa のインストールとアセットのダウンロード

### 3-1. リポジトリのクローン（ホスト上で実行）

```bash
cd ~/cosmos-policy
git clone -b dev https://github.com/bilgik26/robocasa.git new_robocasa
```

クローン先: `~/cosmos-policy/robocasa/`

> 旧リポジトリを参照用に残す場合（推奨）:
> ```bash
> # 旧リポジトリも別名でクローン（参照用）
> git clone https://github.com/moojink/robocasa-cosmos-policy.git ex_robocasa
> ```

### 3-2. 新 RoboCasa のインストール（コンテナ内で実行）

新 RoboCasa は旧版と比べてパッケージの依存関係が変わっており、**複数のパッチが必要**。以下を順番に実行する。

```bash
sudo docker run \
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

    # 新 robocasa のインストール（編集可能モード）
    uv pip install -e robocasa

    # robosuite を GitHub 版（main）に差し替え
    # 新 robocasa の kitchen.py が load_model_on_init を使用しており
    # PyPI 版 robosuite 1.5.1 / 1.5.2 には含まれていないため
    uv pip install 'robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git'

    # numba を安定バージョンに固定
    # 新 robocasa の setup.py が numba==0.61.2 をインストールするが
    # このバージョンは当環境の LLVM で SIGSEGV クラッシュを起こすため
    uv pip install 'numba==0.63.1' 'llvmlite==0.46.0'
  "
```

インストール後のパッケージバージョン（目標値）:

| パッケージ | バージョン |
|---|---|
| robocasa | 1.0.1 |
| robosuite | 1.5.2（GitHub main） |
| mujoco | 3.3.1（robocasa の依存で自動更新） |
| numba | 0.63.1 |
| llvmlite | 0.46.0 |
| numpy | 2.2.6 |

### 3-3. パッチの適用

新 RoboCasa インストール後、**2つのファイルを手動でパッチ**する必要がある。

#### パッチ1: editable インストールの名前空間パッケージバグ修正

**問題:** `/workspace` がカレントディレクトリとして `sys.path` に入るため、Python が `robocasa` パッケージを実際のパッケージディレクトリ（`/workspace/robocasa/robocasa`）ではなく名前空間パッケージ（`/workspace/robocasa`）として誤認識する。

**修正:** `.venv/lib/python3.10/site-packages/__editable___robocasa_1_0_1_finder.py` の `install()` 関数末尾を編集する。

```python
# 変更前
sys.meta_path.append(_EditableFinder)

# 変更後
sys.meta_path.insert(0, _EditableFinder)  # PathFinder より先に登録
```

```bash
# ホスト上で直接編集（sed で一発修正）
sed -i 's/sys.meta_path.append(_EditableFinder)/sys.meta_path.insert(0, _EditableFinder)/' \
  ~/cosmos-policy/.venv/lib/python3.10/site-packages/__editable___robocasa_1_0_1_finder.py
```

#### パッチ2: NumPy バージョンアサーション修正

**問題:** `robocasa/__init__.py` が `numpy.__version__ in ["2.2.5"]` とアサートしているが、インストールされる numpy は 2.2.6。

**修正:** `~/cosmos-policy/robocasa/robocasa/__init__.py` のアサーションに `"2.2.6"` を追加する。

```python
# 変更前
assert numpy.__version__ in [
    "2.2.5",
], "numpy version must be 2.2.5 or 2.2.6. Please install this version."

# 変更後
assert numpy.__version__ in [
    "2.2.5",
    "2.2.6",
], "numpy version must be 2.2.5 or 2.2.6. Please install this version."
```

```bash
# ホスト上で直接編集（sed で一発修正）
# "2.2.5", の行の直後に "2.2.6", を挿入する
sed -i 's/    "2\.2\.5",/    "2.2.5",\n    "2.2.6",/' \
  ~/cosmos-policy/robocasa/robocasa/__init__.py
```

### 3-4. Kitchen アセットのダウンロード

```bash
sudo docker run \
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

    # Kitchen アセットのダウンロード（約10GB、時間がかかる）
    echo 'y' | python robocasa/robocasa/scripts/download_kitchen_assets.py

    # プライベートマクロファイルのセットアップ
    python robocasa/robocasa/scripts/setup_macros.py
  "
```

ダウンロード内容と所要時間（目安）:

| ファイル | サイズ | 内容 |
|---|---|---|
| `textures.zip` | ~538MB | キッチンテクスチャ |
| `fixtures.zip` | ~472MB | キッチン器具モデル |
| `objaverse.zip` | ~2.1GB | 操作対象オブジェクト |
| `generative_textures.zip` | ~593MB | 生成テクスチャ |

ダウンロード先: `robocasa/robocasa/models/assets/`  
マクロファイル: `robocasa/robocasa/macros_private.py` が生成される

### 3-5. 動作確認

```bash
sudo docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -e MUJOCO_GL=osmesa \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  --gpus all --ipc=host --rm -w /workspace \
  cosmos-policy \
  bash -c "
    source .venv/bin/activate
    python -c '
import torch, numba, numpy, robosuite, mujoco, robocasa
print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())
print(\"numba:\", numba.__version__)
print(\"numpy:\", numpy.__version__)
print(\"robosuite:\", robosuite.__version__)
print(\"mujoco:\", mujoco.__version__)
print(\"robocasa:\", robocasa.__version__)
print(\"robocasa path:\", robocasa.__file__)
'
  "
```

期待される出力:
```
torch: 2.7.0+cu128 cuda: True
numba: 0.63.1
numpy: 2.2.6
robosuite: 1.5.2
mujoco: 3.3.1
robocasa: 1.0.1
robocasa path: /workspace/robocasa/robocasa/__init__.py
```

> **注意:** `robocasa path` が `/workspace/robocasa/robocasa/__init__.py` になっていることを確認する。`/workspace/robocasa/__init__.py` と表示される場合はパッチ1が正しく適用されていない。

---

## Step 4: 評価の実行

### 評価スクリプトについて

新 RoboCasa 向けの評価スクリプトは旧スクリプト（`run_robocasa_eval.py`）を元に新しく作成した:

```
cosmos_policy/experiments/robot/robocasa/run_robocasa_eval_new.py
```

旧スクリプトは**修正せず**そのまま残してある（旧 robocasa での参照用）。

#### 旧スクリプトとの主な差異

| 項目 | 旧 (`run_robocasa_eval.py`) | 新 (`run_robocasa_eval_new.py`) |
|---|---|---|
| データセットレジストリ | `SINGLE_STAGE_TASK_DATASETS`, `MULTI_STAGE_TASK_DATASETS` | `ATOMIC_TASK_DATASETS`, `COMPOSITE_TASK_DATASETS` |
| オブジェクト分割の値 | `"A"`（訓練）/ `"B"`（テスト） | `"pretrain"` / `"target"` |
| タスク名（PnP系） | `PnPCounterToCab` 等 | `PickPlaceCounterToCabinet` 等 |
| タスク名（ドア系） | `OpenSingleDoor`, `CloseSingleDoor` | `OpenCabinet`, `CloseCabinet` |
| タスク名（コーヒー） | `CoffeePressButton` | `StartCoffeeMachine` |

新スクリプトは新 robocasa の名前体系のみを受け付ける。旧タスク名・旧 split 値（`"A"`, `"B"`）は使用不可。

### 4-1. 評価コマンド

```bash
sudo docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -e HF_TOKEN=<YOUR_HF_TOKEN> \
  -e HF_HOME=/home/ubuntu/.cache/huggingface \
  -e MUJOCO_GL=osmesa \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  --gpus all --ipc=host --rm -w /workspace \
  cosmos-policy \
  bash -c "
    source .venv/bin/activate
    python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval_new \
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
      --obj_instance_split target \
      --num_trials_per_task 1 \
      --run_id_note my_eval \
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

初回実行時はチェックポイント（`nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`、約3.9GB）が HuggingFace から自動ダウンロードされる。  
ダウンロード先: `$HOME/.cache/huggingface/`

**重要な環境変数:**
- `MUJOCO_GL=osmesa`: CPU ソフトウェアレンダリングを使用（EGL GPU ドライバがない環境向け。詳細はトラブルシューティング参照）
- `HF_HOME`: HuggingFace キャッシュの保存先（コンテナ再起動をまたいで再利用するため `-v $HOME/.cache:/home/ubuntu/.cache` と組み合わせて指定）
- `HF_TOKEN`: チェックポイントへのアクセスに必要

### 4-2. 主要な引数

| 引数 | 値 | 説明 |
|---|---|---|
| `--config` | `cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference` | モデル設定名 |
| `--ckpt_path` | `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B` | HuggingFace のチェックポイント |
| `--task_name` | `TurnOffMicrowave` 等 | 実行するタスク名（旧名・新名どちらでも可） |
| `--num_trials_per_task` | `50` | 1タスクあたりの試行回数（論文と比較する場合は 50） |
| `--seed` | `195` | 再現性のためのシード |
| `--deterministic` | `True` | 決定論的な実行（再現性確保） |
| `--num_denoising_steps_action` | `5` | アクションのデノイジングステップ数 |
| `--chunk_size` | `32` | アクションチャンクサイズ |
| `--num_open_loop_steps` | `16` | オープンループ実行ステップ数 |
| `--obj_instance_split` | `"target"` | オブジェクト分割（`"target"` = テスト用、`"pretrain"` = 訓練用） |

---

## Step 5: 学習用データセットのダウンロード

新 RoboCasa のデータセットは HuggingFace Hub からダウンロードする。全 65 タスクをダウンロードすると 100 GB 超になるため、まず代表的な 10 タスクで動作確認することを推奨する。

### 5-1. 永続コンテナの起動

学習は数時間〜数日かかるため、`--rm` なしの**永続コンテナ**を使用する。

```bash
sudo docker run \
  -u root \
  -e HOST_USER_NAME=ubuntu \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -e HF_HOME=/home/ubuntu/.cache/huggingface \
  -v $HOME/.cache:/home/ubuntu/.cache \
  -v $HOME/.local:/home/ubuntu/.local \
  -v ~/cosmos-policy:/workspace \
  --gpus all \
  --ipc=host \
  --name cosmos_train \
  -w /workspace \
  --entrypoint bash \
  -d \
  cosmos-policy \
  -c "sleep infinity"
```

> **`-v $HOME/.local:/home/ubuntu/.local` について:** `.venv/bin/python` のシンボリックリンクは `uv` が管理する Python インタープリタ（`/home/ubuntu/.local/share/uv/python/.../bin/python3.10`）を参照する。このマウントがないと `.venv/bin/python` が「存在しないパス」を指して `No such file or directory` になる。

> **同名コンテナが既に存在する場合:** `sudo docker rm -f cosmos_train` で削除してから再実行する。

### 5-2. データセットのダウンロード

以下の 10 タスクをダウンロードする（学習の動作確認用）:

| タスク名 | カテゴリ |
|---|---|
| `CloseFridge` / `OpenFridge` | 冷蔵庫 |
| `TurnOnSinkFaucet` / `TurnOffSinkFaucet` | シンク |
| `TurnOnMicrowave` / `TurnOffMicrowave` | 電子レンジ |
| `OpenCabinet` / `CloseCabinet` | キャビネット |
| `PickPlaceCounterToCabinet` / `PickPlaceCabinetToCounter` | 物体操作 |

```bash
sudo docker exec cosmos_train bash -c "
  source /workspace/.venv/bin/activate
  for TASK in CloseFridge OpenFridge TurnOnSinkFaucet TurnOffSinkFaucet \
              TurnOnMicrowave TurnOffMicrowave OpenCabinet CloseCabinet \
              PickPlaceCounterToCabinet PickPlaceCabinetToCounter; do
    echo \"=== Downloading: \$TASK ===\"
    python -c \"
from robocasa.scripts.download_datasets import download_datasets
download_datasets(split=['pretrain'], tasks=['\$TASK'], source=['human'], overwrite=False)
\"
  done
"
```

ダウンロード先: `/workspace/robocasa/datasets/v1.0/pretrain/atomic/<TaskName>/`

各タスクのデータ形式（LeRobot 形式）:
```
datasets/v1.0/pretrain/atomic/<TaskName>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # アクション・固有感覚データ
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.agentview_image/<episode>.mp4
        ├── observation.images.robot0_eye_in_hand_image/<episode>.mp4
        └── ...
```

10 タスクダウンロード後の統計:
- エピソード数: 約 1,000
- 総ステップ数: 約 220,000
- ユニークタスク説明: 約 137 件

---

## Step 6: T5テキスト埋め込みの生成

Cosmos Policy はテキストコンディショニングに T5 エンコーダを使用する。実行時にオンザフライでエンコードするとメモリ不足（OOM）になるため、事前に全タスク説明をエンコードして `.pkl` ファイルに保存する。

```bash
sudo docker exec cosmos_train bash -c "
  source /workspace/.venv/bin/activate
  HF_TOKEN=<YOUR_HF_TOKEN> \
  python -m cosmos_policy.datasets.save_new_robocasa_t5_text_embeddings \
    --output_path /workspace/robocasa/datasets/new_robocasa_t5_embeddings.pkl
"
```

出力:
- ファイル: `/workspace/robocasa/datasets/new_robocasa_t5_embeddings.pkl`
- サイズ: 約 154 MB
- 内容: ユニークタスク説明 153 件の T5 埋め込み（shape: `[seq_len=512, dim=1024]`）

HuggingFace からダウンロードしたデータセットに含まれる全タスク説明を自動収集してエンコードするため、ダウンロード済みタスクが増えた場合は再実行すること。

---

## Step 7: 学習の実行

### 7-1. 学習関連ファイルの概要

新 RoboCasa 対応のために新規作成・修正したファイル:

| ファイル | 種別 | 概要 |
|---|---|---|
| `cosmos_policy/datasets/new_robocasa_dataset.py` | 新規 | LeRobot 形式（parquet + MP4）→ PyTorch Dataset |
| `cosmos_policy/datasets/save_new_robocasa_t5_text_embeddings.py` | 新規 | T5 埋め込みの事前計算スクリプト |
| `cosmos_policy/config/experiment/new_robocasa_experiment_configs.py` | 新規 | 実験設定（モデル・データローダー・最適化パラメータ） |
| `cosmos_policy/scripts/train.py` | 修正 | DataLoader に `multiprocessing_context` を渡すよう修正（後述） |

### 7-2. 学習コマンド（1-GPU）

```bash
sudo docker exec -it cosmos_train bash -c "
  source /workspace/.venv/bin/activate
  cd /workspace
  WANDB_API_KEY=<YOUR_WANDB_API_KEY> \
  HF_TOKEN=<YOUR_HF_TOKEN> \
  torchrun --nproc_per_node=1 \
    -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py \
    -- \
    experiment='cosmos_predict2_2b_480p_new_robocasa_pretrain_human' \
    trainer.max_iter=2000 \
    trainer.logging_iter=10 \
    trainer.grad_accum_iter=4 \
    dataloader_train.batch_size=2 \
    checkpoint.save_iter=500 \
    job.name='new_robocasa_2000iter'
"
```

主要引数:

| 引数 | 値 | 説明 |
|---|---|---|
| `experiment` | `cosmos_predict2_2b_480p_new_robocasa_pretrain_human` | 実験設定名（`new_robocasa_experiment_configs.py` で定義） |
| `trainer.max_iter` | `2000` | 学習イテレーション数 |
| `trainer.grad_accum_iter` | `4` | 勾配累積ステップ数（実効バッチサイズ = `batch_size × grad_accum_iter = 8`） |
| `dataloader_train.batch_size` | `2` | 1-GPU (A100-40GB) でのバッチサイズ上限 |
| `checkpoint.save_iter` | `500` | チェックポイント保存間隔 |

### 7-3. 期待される出力と性能

- 学習速度: 約 4.4 秒 / イテレーション（A100-40GB、1-GPU）
- GPU 使用率: 約 98%、VRAM 使用量: 約 36 GB
- 2000 イテレーション所要時間: 約 2.7 時間

損失値の推移（10 タスク、ランダム重みから学習開始）:

```
[iter   10]  loss: ~9100  (学習開始直後)
[iter  500]  loss: ~3-4
[iter 1000]  loss: ~1.5-2
[iter 2000]  loss: ~1.2-1.3
```

### 7-4. チェックポイントの保存先

チェックポイントは以下に保存される:

```
/tmp/experiments/cosmos_v2_finetune/<job.name>/
├── config.yaml
├── 0000000500/    # iter=500
├── 0001000000/    # iter=1000
└── 0002000000/    # iter=2000
```

> `/tmp` は Docker コンテナ再起動で消えるため、重要なチェックポイントはホストにコピーすること:
> ```bash
> sudo docker cp cosmos_train:/tmp/experiments ~/cosmos-policy/checkpoints/
> ```

### 7-5. wandb でのモニタリング

`WANDB_API_KEY` を設定して起動すると、学習の進捗を wandb でリアルタイムに確認できる。ログは `https://wandb.ai/<YOUR_USERNAME>/cosmos_policy/` に自動で記録される。

---

## 新 RoboCasa のタスク一覧

### ATOMIC タスク（65タスク）

以下は `ATOMIC_TASK_DATASETS` に含まれる代表的なタスクと最大ステップ数:

| タスク名（新） | 最大ステップ数 | 旧タスク名 |
|---|---|---|
| `TurnOffMicrowave` | 300 | — |
| `TurnOnMicrowave` | 450 | — |
| `TurnOffSinkFaucet` | 300 | — |
| `TurnOnSinkFaucet` | 600 | — |
| `TurnOffStove` | 750 | — |
| `TurnOnStove` | 450 | — |
| `TurnSinkSpout` | 300 | — |
| `OpenDrawer` | 750 | — |
| `CloseDrawer` | 450 | — |
| `OpenCabinet` | 1050 | `OpenSingleDoor` |
| `CloseCabinet` | 750 | `CloseSingleDoor` |
| `CoffeeSetupMug` | 600 | — |
| `CoffeeServeMug` | 450 | — |
| `StartCoffeeMachine` | 300 | `CoffeePressButton` |
| `PickPlaceCounterToCabinet` | 750 | `PnPCounterToCab` |
| `PickPlaceCabinetToCounter` | 450 | `PnPCabToCounter` |
| `PickPlaceCounterToSink` | 600 | `PnPCounterToSink` |
| `PickPlaceSinkToCounter` | 900 | `PnPSinkToCounter` |
| `PickPlaceCounterToMicrowave` | 1050 | `PnPCounterToMicrowave` |
| `PickPlaceMicrowaveToCounter` | 750 | `PnPMicrowaveToCounter` |
| `PickPlaceCounterToStove` | 600 | `PnPCounterToStove` |
| `PickPlaceStoveToCounter` | 450 | `PnPStoveToCounter` |

### COMPOSITE タスク（252タスク）

`COMPOSITE_TASK_DATASETS` には複数のサブタスクを組み合わせた複合タスクが含まれる。

---

## 評価結果の確認

ログと動画は `cosmos_policy/experiments/robot/robocasa/logs/` に保存される:

```
logs/
├── ENV_EVAL-<TASK>-cosmos-<DATE>--<RUN_ID_NOTE>.txt   # 集計結果・エピソードログ
└── rollout_data/
    └── <TASK>--<DATE>/
        ├── <DATE>--episode=0--success=True--task=<...>.mp4   # ロールアウト動画
        ├── <DATE>--with_future_img--episode=0--success=True--task=<...>.mp4   # 未来画像予測付き動画
        └── ...
```

---

## トラブルシューティング

### `robocasa.__file__` が `/workspace/robocasa/__init__.py` を指す（名前空間パッケージ誤認識）

**症状:** `robocasa.__path__[0]` が `/workspace/robocasa` になり、`robocasa.environments` 等のサブモジュールが見つからない。

**原因:** `uv pip install -e robocasa` で生成される editable インストールのファインダー（`__editable___robocasa_1_0_1_finder.py`）が `sys.meta_path.append()` で登録されるため、`/workspace` を走査する `PathFinder` がより先に実行され、`/workspace/robocasa/` が名前空間パッケージとして誤認識される。

**修正:** `.venv/lib/python3.10/site-packages/__editable___robocasa_1_0_1_finder.py` の末尾を変更する（Step 3-3 パッチ1 参照）。

```python
# 変更前
sys.meta_path.append(_EditableFinder)
# 変更後
sys.meta_path.insert(0, _EditableFinder)
```

---

### `AssertionError: numpy version must be 2.2.5`

**症状:** `import robocasa` 時に AssertionError。

**原因:** `robocasa/__init__.py` が `numpy.__version__ in ["2.2.5"]` とアサートしているが、インストールされる numpy は 2.2.6。

**修正:** `robocasa/robocasa/__init__.py` のアサーションに `"2.2.6"` を追加する（Step 3-3 パッチ2 参照）。

---

### `TypeError: ManipulationEnv.__init__() got an unexpected keyword argument 'load_model_on_init'`

**症状:** 環境作成時に `TypeError`。

**原因:** 新 RoboCasa の `kitchen.py` が `load_model_on_init=False` を `ManipulationEnv` に渡しているが、PyPI 版 `robosuite==1.5.1` / `1.5.2` にはこの引数が存在しない。

**修正:** robosuite を GitHub main 版に差し替える（Step 3-2 参照）。

```bash
source .venv/bin/activate
uv pip install 'robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git'
```

---

### SIGSEGV クラッシュ（`llvmlite.binding.passmanagers`）

**症状:** `placement_samplers.py` の numba JIT コンパイル中に SIGSEGV でプロセスが落ちる。

**原因:** 新 RoboCasa の `setup.py` が `numba==0.61.2` + `llvmlite==0.44.0` をインストールするが、このバージョンは当環境の LLVM 最適化パスで SIGSEGV を起こす。

**修正:** 安定バージョンに固定する（Step 3-2 参照）。

```bash
source .venv/bin/activate
uv pip install 'numba==0.63.1' 'llvmlite==0.46.0'
```

---

### `ValueError: Invalid split value: B`（または `"A"`）

**症状:** 環境作成時に `kitchen_object_utils.py` で `ValueError`。

**原因:** 新 RoboCasa はオブジェクト分割値として `"pretrain"` / `"target"` のみを受け付ける。旧形式の `"A"` / `"B"` は無効。

**修正:** `--obj_instance_split target`（テスト用）または `--obj_instance_split pretrain`（訓練用）を指定すること。

---

### `torch.OutOfMemoryError: CUDA out of memory`（T5 テキストエンコーダ読み込み時）

**症状:** アクション推論時に T5 テキストエンコーダを GPU に読み込もうとして OOM。

**原因:** `--t5_text_embeddings_path` で指定したキャッシュファイルに、実行中のタスク説明が登録されていない場合にオンザフライで T5 モデルを GPU にロードしようとする。新 RoboCasa のタスク説明は先頭大文字・末尾ピリオドありの形式（例: `"Press the stop button on the microwave."`）だが、事前計算済みキャッシュは小文字・末尾ピリオドなし形式（例: `"press the stop button on the microwave"`）で登録されているため、キャッシュミスが発生する。

**修正:** `run_robocasa_eval_new.py` はタスク説明を `.lower().rstrip(".")` で正規化してからキャッシュを検索する（`run_task()` 関数内）。旧スクリプトを使っている場合は手動で正規化するか、T5 推論用 GPU メモリを確保すること。

---

### DataLoader が固まって学習が進まない（`futex_wait_queue` デッドロック）

**症状:** wandb 初期化後、最初のデータ取得（`next(dataloader_train_iter)`）でプロセスが無限にブロックされ、GPU 使用率が 0% のまま何時間経過しても最初のイテレーションが完了しない。

**原因:** `train.py` が DataLoader を `multiprocessing_context` 未指定で生成するため、デフォルト（`fork`）が使用される。FSDP・wandb・PyTorch インダクタのコンパイルワーカー（計 100 スレッド以上）が起動した後に `fork` すると、GIL を保持したまま fork されたスレッドが子プロセス内で `futex_wait_queue` にブロックされ、全 DataLoader ワーカーがデッドロックする。

**修正（既に適用済み）:**

1. `cosmos_policy/scripts/train.py` の DataLoader 生成部分（行 71〜82）に `multiprocessing_context` を追加:
   ```python
   dataloader_train = DataLoader(
       ...
       multiprocessing_context=getattr(config.dataloader_train, "multiprocessing_context", None),
   )
   ```

2. `cosmos_policy/config/experiment/new_robocasa_experiment_configs.py` の `dataloader_train` に設定を追加:
   ```python
   dataloader_train=L(DataLoader)(
       num_workers=4,
       multiprocessing_context="spawn",   # fork-after-multithread デッドロックを回避
       persistent_workers=True,
       ...
   )
   ```

`"spawn"` は既存スレッドを引き継がない新しい Python インタープリタを起動するため、fork 起因のデッドロックが発生しない。

**応急処置（`train.py` を修正できない場合）:** `dataloader_train.num_workers=0` を学習コマンドに追加するとシングルスレッドで動作する（速度は約 8.6 秒/iter と低下）:
```bash
torchrun ... -- ... dataloader_train.num_workers=0
```

---

### `Cannot initialize a EGL device display`

Compute-only ドライバ環境（`libEGL_nvidia.so.0` が存在しない）では EGL レンダリングは使用できない。`MUJOCO_GL=osmesa` に切り替えること。

```bash
# EGL 関連変数（削除）:
# -e MUJOCO_GL=egl
# -e PYOPENGL_PLATFORM=egl
# -e __EGL_VENDOR_LIBRARY_FILENAMES=...

# 代わりに（追加）:
-e MUJOCO_GL=osmesa
```

OSMesa は CPU でのソフトウェアレンダリングになるが、MuJoCo の物理演算と推論は引き続き GPU で実行される。

確認方法:
```bash
find /usr -name "libEGL_nvidia*" 2>/dev/null
# 何も表示されない場合は compute-only ドライバ環境
```

---

## 旧 RoboCasa との差異まとめ

| 項目 | 旧 (`moojink/robocasa-cosmos-policy`) | 新 (`bilgik26/robocasa`) |
|---|---|---|
| バージョン | v0.1 | v1.0.1 |
| robosuite | 1.5.1（PyPI） | 1.5.2（GitHub main） |
| mujoco | 3.2.6 | 3.3.1 |
| numba | 0.61.x | 0.63.1（要手動固定） |
| データセットレジストリ | `SINGLE/MULTI_STAGE_TASK_DATASETS` | `ATOMIC/COMPOSITE_TASK_DATASETS` |
| ATOMIC タスク数 | 24 | 65 |
| COMPOSITE タスク数 | — | 252 |
| オブジェクト分割値 | `"A"` / `"B"` | `"pretrain"` / `"target"` |
| タスク説明の形式 | 小文字・末尾ピリオドなし | 先頭大文字・末尾ピリオドあり |
| 評価スクリプト | `run_robocasa_eval.py` | `run_robocasa_eval_new.py` |
| インストールパス | `robocasa-cosmos-policy/robocasa/` | `robocasa/robocasa/` |
