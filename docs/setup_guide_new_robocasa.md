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
git clone https://github.com/bilgik26/robocasa.git new_robocasa
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
      --num_trials_per_task 50 \
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
