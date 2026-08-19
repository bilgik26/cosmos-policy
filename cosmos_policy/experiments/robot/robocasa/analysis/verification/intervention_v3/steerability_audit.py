"""
steerability_audit.py — design_v3.md Stage 0 準拠 (steerability 監査)

WA-LQR (arXiv:2607.14943) §4 の手法を Cosmos-Policy/RoboCasa に適用する。
5種類の matched contrastive pair 集合について、(層 ℓ) × (デノイジング step k) の
全格子で「PCA(top-3) + 線形SVM の交差検証ヒンジ損失」を分離性スコアとして計算する。

対比集合 (design_v3.md §3 Stage 0 の表):
  L (言語)       : 実プロンプト vs ダミープロンプト、同一シーン・同一初期ノイズ seed
  T (タスク特異性): "pick and place" 系プロンプト vs 他タスクのプロンプト、同一シーン・同一seed
  P (フェーズ)    : grasp フェーズ (gripper closed) vs approach フェーズ (gripper open+moving)
  G (グリッパー)  : gripper closed vs open
  S (成否)       : 成功エピソードの call vs 失敗エピソードの call

L・T は「ロールアウト不要」(design_v3.md の特筆): 各シードで env をリセットして単一の
観測を取得し、そこから条件を variable のみ変えた2回の get_action 呼び出しでペアを作る。
P・G・S は attractor/collect_multitask.py が既に収集した held-out データ
(results/attractor_verification/collect/) を再利用する — 新規GPU計算不要。

出力: --out_dir 下に
  steerability_grid.npz       (5 pair_type × 7 layer × 5 step のヒンジ損失/精度/p値/BH-FDR)
  steerability_heatmap_<pair_type>.png
  steerability_summary.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    ACTION_LATENT_IDX_ROBOCASA,
    NUM_DENOISE_STEPS,
    PROBE_LAYERS,
    SIGMA_SCHEDULE,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import bh_fdr
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    preload_dummy_prompt_embedding,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_COLLECT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_steerability_audit"
)

DUMMY_PROMPT = "The weather today is sunny."
# T (task-specificity) pair "other task" candidates for the online collector. Their
# ACTUAL env.get_ep_meta()["lang"] string (fetched once via get_real_task_lang(), not
# a hand-written paraphrase) is used as the minus-condition prompt — a hand-written
# prompt not present in the precomputed t5_text_embeddings_path cache triggers an
# on-the-fly T5-11b encode, which OOMs a 24GB GPU alongside the already-loaded 2B
# policy DiT (hit during development — see report_v3.md bug list).
TASK_PROMPTS_FOR_T_PAIR = ["PnPCounterToCab", "TurnOnStove", "CoffeePressButton", "CloseDrawer"]


# ── オンライン収集 (L, T): ロールアウト不要、単一観測 + ペア forward pass ──

class PairCapture:
    """1回の get_action 呼び出し内の (denoise step k) x (probe layer) 活性を捕捉する。
    attractor/collect_multitask.py の MultiTaskCapture と同一パターン。"""

    def __init__(self, probe_layers):
        self.probe_layers = probe_layers
        self._current_step = -1
        self._current_feats = {}
        self._handles = []

    def register(self, model):
        net = model.net
        for layer_idx in self.probe_layers:
            block = net.blocks[layer_idx]
            self._handles.append(block.register_forward_hook(self._make_hook(layer_idx)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        self._current_step = -1
        self._current_feats = {}

    def before_denoise_step(self):
        self._current_step += 1
        self._current_feats.setdefault(self._current_step, {})

    def _make_hook(self, layer_idx):
        def hook(module, inp, output):
            if self._current_step < 0:
                return
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return
            _, T, _, _, _ = output.shape
            if T <= ACTION_LATENT_IDX_ROBOCASA:
                return
            feat = output[0, ACTION_LATENT_IDX_ROBOCASA]
            self._current_feats.setdefault(self._current_step, {})[layer_idx] = (
                feat.float().mean(dim=(0, 1)).detach().cpu().numpy()
            )
        return hook


def get_action_with_capture(cfg, model, dataset_stats, obs, task_desc, capture, seed):
    capture.reset()
    orig = model.get_x0_fn_from_batch

    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result

            def w(x, s):
                capture.before_denoise_step()
                return fn(x, s)

            return w, extra
        else:
            def w(x, s):
                capture.before_denoise_step()
                return result(x, s)

            return w

    model.get_x0_fn_from_batch = patched
    try:
        get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
            task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = orig
    # returns dict[k][layer] -> (D,) vector
    return {k: dict(lf) for k, lf in capture._current_feats.items()}


_TASK_LANG_CACHE = {}


def get_real_task_lang(cfg, task_name):
    """その task の実際の言語指示文字列を1回だけ取得してキャッシュする。t5_text_embeddings_path
    の事前計算キャッシュに含まれる正確な文字列でなければならない (自作の言い換え文は
    on-the-fly T5 エンコードを誘発し、2B policy と同一GPU上でCUDA OOMする — 開発時に実際に
    ヒットしたバグ、report_v3.md バグリスト参照)。"""
    if task_name in _TASK_LANG_CACHE:
        return _TASK_LANG_CACHE[task_name]
    prev_task = cfg.task_name
    cfg.task_name = task_name
    env, _ = create_robocasa_env(cfg, seed=1000, episode_idx=0)
    env.reset()
    lang = env.get_ep_meta().get("lang", task_name)
    env.close()
    cfg.task_name = prev_task
    _TASK_LANG_CACHE[task_name] = lang
    return lang


def collect_online_pairs(cfg, model, dataset_stats, pair_type, n_pairs, base_seed, probe_layers, task_for_env):
    """L, T ペア: 同一シーン (env reset, 単一観測)・同一 get_action seed、プロンプトのみ変える。
    ロールアウトはしない (env.step は観測を安定させるための no-op warmup のみ)。
    Returns: feats_plus[k][layer] -> list of (D,) vectors, feats_minus[...], 同形式。
    """
    capture = PairCapture(probe_layers)
    capture.register(model)
    feats_plus = {k: {l: [] for l in probe_layers} for k in range(NUM_DENOISE_STEPS)}
    feats_minus = {k: {l: [] for l in probe_layers} for k in range(NUM_DENOISE_STEPS)}
    n_ok = 0
    if pair_type == "T":
        other_task_langs = [get_real_task_lang(cfg, t) for t in TASK_PROMPTS_FOR_T_PAIR if t != task_for_env]
    try:
        for i in range(n_pairs):
            seed = base_seed + i
            cfg.task_name = task_for_env
            env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=i)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
            observation = prepare_observation(obs, cfg.flip_images)
            real_task_desc = env.get_ep_meta().get("lang", task_for_env)
            env.close()

            if pair_type == "L":
                desc_plus, desc_minus = real_task_desc, DUMMY_PROMPT
            elif pair_type == "T":
                desc_plus = real_task_desc
                desc_minus = other_task_langs[i % len(other_task_langs)]
            else:
                raise ValueError(pair_type)

            fp = get_action_with_capture(cfg, model, dataset_stats, observation, desc_plus, capture, seed=seed)
            fm = get_action_with_capture(cfg, model, dataset_stats, observation, desc_minus, capture, seed=seed)
            for k in range(NUM_DENOISE_STEPS):
                for l in probe_layers:
                    if l in fp.get(k, {}) and l in fm.get(k, {}):
                        feats_plus[k][l].append(fp[k][l])
                        feats_minus[k][l].append(fm[k][l])
            n_ok += 1
            if (i + 1) % 5 == 0:
                log_message(f"  [{pair_type}] {i + 1}/{n_pairs} pairs collected")
    finally:
        capture.remove()
    log_message(f"[{pair_type}] collected {n_ok}/{n_pairs} pairs")
    return feats_plus, feats_minus


# ── オフライン収集 (P, G, S): collect_multitask.py の held-out データを再利用 ──

def load_offline_pairs(collect_dir: Path, pair_type: str, probe_layers, tasks=None, success_collect_dir: Path = None):
    """P, G, S ペア: attractor/collect_multitask.py + phase_labeling.py の出力から
    ラベル + episode id を読み、(k,layer) ごとに X(plus/minus), group(episode) を返す。

    NOTE: --collect_dir (results/attractor_verification/collect/, デフォルト) はP/Gに使う
    _phases.npz を持つが、この収集run自体は episode success を記録していない (report_v3.md
    バグリスト参照 — collect_multitask.py の初期の実行では ep_success が空だった)。
    S (成否) ペアのみ、success ラベルが実際に入っている collect_v2/ を success_collect_dir
    として別途参照する。

    Returns: feats[k][layer] -> dict(X: (N,D), y: (N,) in {+1,-1}, groups: (N,))
    """
    src_dir = success_collect_dir if (pair_type == "S" and success_collect_dir is not None) else collect_dir
    manifest = json.loads((src_dir / "multitask_manifest.json").read_text())
    files = sorted(manifest["files"].keys())
    if tasks is not None:
        files = [f for f in files if manifest["files"][f]["task"] in tasks]

    h_threshold = None
    if pair_type == "H":
        from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.phase_labeling import (
            two_means_1d,
        )
        pooled_z = np.concatenate([np.load(src_dir / f)["eef_pos"][:, 2] for f in files])
        h_threshold, _ = two_means_1d(pooled_z)
        log_message(f"[H pair] global EE-height threshold (2-means on eef_pos[:,2]) = {h_threshold:.4f}")

    feats = {k: {l: {"X": [], "y": [], "groups": []} for l in probe_layers} for k in range(NUM_DENOISE_STEPS)}
    ep_offset = 0
    for fname in files:
        fd = np.load(src_dir / fname)
        n_calls_total = len(fd["episode"])
        ep_ids_full = fd["episode"] + ep_offset

        if pair_type == "G":
            pd = np.load(src_dir / fname.replace(".npz", "_phases.npz"))
            label_full = pd["gripper_state"]  # 1=closed(+), 0=open(-)
            keep_mask_full = np.ones(n_calls_total, dtype=bool)
        elif pair_type == "P":
            pd = np.load(src_dir / fname.replace(".npz", "_phases.npz"))
            # grasp (closed, phase_label in {2,3}) = +1  vs  approach (open+moving, phase_label==1) = -1
            phase = pd["phase_label"]
            keep_mask_full = np.isin(phase, [1, 2, 3])
            label_full = np.where(np.isin(phase, [2, 3]), 1, 0)
        elif pair_type == "S":
            label_full = fd["success"].astype(int)  # 1=success(+), 0=fail(-)
            keep_mask_full = np.ones(n_calls_total, dtype=bool)
        elif pair_type == "V":
            # EE velocity (motion) pair, reusing phase_labeling.py's motion_state (call-to-call
            # eef_pos displacement norm, globally two-means-thresholded in log1p space).
            # 1=moving(+), 0=still(-). Added for report_v3.md §6.2 follow-up E-4 (EE velocity
            # reproduction of the Stage 3-A gripper setpoint result).
            pd = np.load(src_dir / fname.replace(".npz", "_phases.npz"))
            label_full = pd["motion_state"]
            keep_mask_full = np.ones(n_calls_total, dtype=bool)
        elif pair_type == "H":
            # EE height pair: raw eef_pos z-component, globally two-means-thresholded (same
            # methodology as phase_labeling.py's gripper/motion binarization, computed once
            # below across all files before this per-file loop runs — see height_threshold
            # closure). 1=high(+), 0=low(-). Added for report_v3.md §6.2 follow-up E-4.
            label_full = (fd["eef_pos"][:, 2] > h_threshold).astype(int)
            keep_mask_full = np.ones(n_calls_total, dtype=bool)
        else:
            raise ValueError(pair_type)

        for k in range(NUM_DENOISE_STEPS):
            for l in probe_layers:
                key, idx_key = f"feat_k{k}_layer{l}", f"feat_k{k}_layer{l}_idx"
                if key not in fd:
                    continue
                X = fd[key]
                keep_idx = fd[idx_key]  # row index into the per-call meta arrays
                row_mask = keep_mask_full[keep_idx]
                if not row_mask.any():
                    continue
                feats[k][l]["X"].append(X[row_mask])
                feats[k][l]["y"].append(np.where(label_full[keep_idx][row_mask] == 1, 1, -1))
                feats[k][l]["groups"].append(ep_ids_full[keep_idx][row_mask])
        ep_offset += int(fd["episode"].max()) + 1

    out = {k: {} for k in range(NUM_DENOISE_STEPS)}
    for k in range(NUM_DENOISE_STEPS):
        for l in probe_layers:
            d = feats[k][l]
            if not d["X"]:
                continue
            out[k][l] = {
                "X": np.concatenate(d["X"]),
                "y": np.concatenate(d["y"]),
                "groups": np.concatenate(d["groups"]),
            }
    return out


# ── PCA(top-3) + 線形SVM ヒンジ損失 (グループ付き交差検証 + permutation null) ──
#
# 効率上の注意: fold毎の厳密SVD (np.linalg.svd, full_matrices=False) は offline (P/G/S)
# データが数千行×D=2048のとき O(N・D・min(N,D)) で非常に遅い (smoketestで1セルが
# 300秒のタイムアウト内に完了しないことを実測した — 本節末「バグ」参照)。
# 上位3成分のみで良いので randomized_svd (sklearn.utils.extmath) を使い、さらに
# グループ単位でサブサンプルして1セルあたりの行列サイズに上限を設ける。

MAX_ROWS_PER_CELL = 1200  # グループ単位のサブサンプル上限 (offline P/G/Sのみ影響)


def _subsample_by_group(X, y, groups, cap, seed):
    if len(y) <= cap:
        return X, y, groups
    rng = np.random.RandomState(seed)
    uniq_groups = np.unique(groups)
    rng.shuffle(uniq_groups)
    keep_mask = np.zeros(len(y), dtype=bool)
    n_kept = 0
    for g in uniq_groups:
        gmask = groups == g
        keep_mask |= gmask
        n_kept += gmask.sum()
        if n_kept >= cap:
            break
    return X[keep_mask], y[keep_mask], groups[keep_mask]


def pca3_svm_hinge_cv(X, y, groups, n_splits=5, seed=0, max_rows=MAX_ROWS_PER_CELL):
    """fold内標準化 -> fold内 randomized PCA(top-3) -> 線形SVM。held-outでのヒンジ損失と
    精度を返す。0=完全分離(margin>=1で全て正しい側), 1=ランダム分離相当、が近似的な目安に
    なるようSVMのCはデフォルト(C=1)に固定して全fold・全permutationで統一する。"""
    from sklearn.model_selection import GroupKFold
    from sklearn.svm import LinearSVC
    from sklearn.utils.extmath import randomized_svd

    X, y, groups = _subsample_by_group(X, y, groups, max_rows, seed)
    uniq_groups = np.unique(groups)
    n_splits_eff = min(n_splits, len(uniq_groups))
    if n_splits_eff < 2:
        return None
    gkf = GroupKFold(n_splits=n_splits_eff)
    hinge_losses, accs = [], []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        X_train_s, X_test_s = (X_train - mu) / sd, (X_test - mu) / sd
        n_comp = min(3, min(X_train_s.shape) - 1)
        if n_comp < 1:
            continue
        _, _, Vt = randomized_svd(X_train_s, n_components=n_comp, random_state=seed)
        comps = Vt.T
        X_train_p, X_test_p = X_train_s @ comps, X_test_s @ comps

        clf = LinearSVC(C=1.0, max_iter=5000, dual="auto")
        clf.fit(X_train_p, y_train)
        scores = clf.decision_function(X_test_p)
        hinge = np.maximum(0.0, 1.0 - y_test * scores).mean()
        hinge_losses.append(hinge)
        accs.append(clf.score(X_test_p, y_test))
    if not hinge_losses:
        return None
    return {"hinge": float(np.mean(hinge_losses)), "acc": float(np.mean(accs)), "n_folds": len(hinge_losses)}


def permutation_null(X, y, groups, observed_hinge, n_perm=100, seed=0, n_splits=5, max_rows=MAX_ROWS_PER_CELL):
    """グループ単位でラベルをシャッフルした帰無分布に対するp値 (低いほど有意=分離している)。
    低ヒンジ損失が「分離している」方向なので、p = P(null_hinge <= observed_hinge)。
    観測値と同じサブサンプルを固定して使い、公平な比較にする。"""
    X, y, groups = _subsample_by_group(X, y, groups, max_rows, seed)
    rng = np.random.RandomState(seed)
    uniq_groups = np.unique(groups)
    group_label = {}
    for g in uniq_groups:
        vals = y[groups == g]
        group_label[g] = vals[0] if len(np.unique(vals)) == 1 else rng.choice(vals)
    null_hinges = []
    for i in range(n_perm):
        shuffled = rng.permutation(uniq_groups)
        mapping = dict(zip(uniq_groups, shuffled))
        y_perm = np.array([group_label.get(mapping[g], y[j]) if g in mapping else y[j]
                            for j, g in enumerate(groups)])
        # groups whose label wasn't well-defined (mixed) keep original y; otherwise use shuffled label
        res = pca3_svm_hinge_cv(X, y_perm, groups, n_splits=n_splits, seed=seed + i + 1, max_rows=max_rows)
        if res is not None:
            null_hinges.append(res["hinge"])
    if not null_hinges:
        return 1.0
    null_hinges = np.array(null_hinges)
    return float((null_hinges <= observed_hinge).mean())


def run_pair_type_grid(feats_plus_or_dict, feats_minus, probe_layers, pair_type, n_perm, seed):
    """feats_plus/feats_minus (online) または offline dict のいずれかから grid を計算。"""
    grid_hinge = np.full((len(probe_layers), NUM_DENOISE_STEPS), np.nan)
    grid_acc = np.full((len(probe_layers), NUM_DENOISE_STEPS), np.nan)
    grid_p = np.full((len(probe_layers), NUM_DENOISE_STEPS), np.nan)
    grid_n = np.zeros((len(probe_layers), NUM_DENOISE_STEPS), dtype=int)

    for li, l in enumerate(probe_layers):
        for k in range(NUM_DENOISE_STEPS):
            if feats_minus is not None:  # online pair format
                xp = np.array(feats_plus_or_dict[k][l])
                xm = np.array(feats_minus[k][l])
                if len(xp) < 6 or len(xm) < 6:
                    continue
                X = np.concatenate([xp, xm])
                y = np.concatenate([np.ones(len(xp)), -np.ones(len(xm))])
                # group by pair index i so plus/minus of the SAME scene/seed never split
                # across train/test (avoids leakage of the shared visual scene).
                groups = np.concatenate([np.arange(len(xp)), np.arange(len(xm))])
            else:  # offline dict format keyed [k][l] -> {X,y,groups}
                if l not in feats_plus_or_dict.get(k, {}):
                    continue
                d = feats_plus_or_dict[k][l]
                X, y, groups = d["X"], d["y"], d["groups"]
                if len(np.unique(y)) < 2:
                    continue

            res = pca3_svm_hinge_cv(X, y, groups, seed=seed)
            if res is None:
                continue
            grid_hinge[li, k] = res["hinge"]
            grid_acc[li, k] = res["acc"]
            grid_n[li, k] = len(y)
            grid_p[li, k] = permutation_null(X, y, groups, res["hinge"], n_perm=n_perm, seed=seed)
        log_message(f"  [{pair_type}] layer {l} done "
                    f"(hinge row={np.round(grid_hinge[li], 3).tolist()})")
    return grid_hinge, grid_acc, grid_p, grid_n


def plot_heatmap(grid, probe_layers, title, out_path, vmin=0, vmax=1, cmap="RdYlGn_r", fmt="{:.2f}"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(NUM_DENOISE_STEPS))
    ax.set_xticklabels([f"k={k}\n(σ={SIGMA_SCHEDULE[k]})" for k in range(NUM_DENOISE_STEPS)])
    ax.set_yticks(range(len(probe_layers)))
    ax.set_yticklabels([f"Blk-{l}" for l in probe_layers])
    ax.set_xlabel("denoising step")
    ax.set_ylabel("DiT block")
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--collect_dir", default=str(DEFAULT_COLLECT_DIR))
    p.add_argument("--success_collect_dir", default=str(DEFAULT_COLLECT_DIR.parent / "collect_v2"),
                    help="used only for pair_type=S: --collect_dir's run did not record "
                         "episode success (see report_v3.md bug list)")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--pair_types", nargs="+", default=["L", "T", "P", "G", "S"])
    p.add_argument("--n_pairs_online", type=int, default=20)
    p.add_argument("--online_task", default="PnPCounterToCab")
    p.add_argument("--offline_tasks", nargs="+", default=None,
                    help="restrict P/G/S to these tasks; default = all in manifest")
    p.add_argument("--n_perm", type=int, default=100)
    p.add_argument("--seed", type=int, default=195)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_dir = Path(args.collect_dir)
    probe_layers = PROBE_LAYERS

    need_online = any(pt in args.pair_types for pt in ("L", "T"))
    model, dataset_stats, cfg = None, None, None
    if need_online:
        cfg = PolicyEvalConfig(
            config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
            dataset_stats_path=args.dataset_stats_path, t5_text_embeddings_path=args.t5_text_embeddings_path,
            task_name=args.online_task, seed=args.seed,
        )
        model, _ = get_model(cfg)
        model.eval()
        dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
        preload_dummy_prompt_embedding(device="cuda:0")

    all_results = {}
    for pair_type in args.pair_types:
        t0 = time.time()
        log_message(f"=== Stage 0 steerability audit: pair_type={pair_type} ===")
        if pair_type in ("L", "T"):
            fp, fm = collect_online_pairs(
                cfg, model, dataset_stats, pair_type, args.n_pairs_online,
                base_seed=args.seed, probe_layers=probe_layers, task_for_env=args.online_task,
            )
            grid_hinge, grid_acc, grid_p, grid_n = run_pair_type_grid(fp, fm, probe_layers, pair_type, args.n_perm, args.seed)
        else:
            offline = load_offline_pairs(collect_dir, pair_type, probe_layers, tasks=args.offline_tasks,
                                          success_collect_dir=Path(args.success_collect_dir))
            grid_hinge, grid_acc, grid_p, grid_n = run_pair_type_grid(offline, None, probe_layers, pair_type, args.n_perm, args.seed)

        valid = ~np.isnan(grid_p)
        sig = np.zeros_like(grid_p, dtype=bool)
        if valid.any():
            sig_flat = bh_fdr(grid_p[valid].tolist())
            sig[valid] = sig_flat

        plot_heatmap(grid_hinge, probe_layers, f"Stage0 [{pair_type}]: PCA(3)+SVM hinge loss (CV)",
                     out_dir / f"steerability_heatmap_{pair_type}_hinge.png", vmin=0, vmax=1.2)
        plot_heatmap(grid_p, probe_layers, f"Stage0 [{pair_type}]: permutation p-value",
                     out_dir / f"steerability_heatmap_{pair_type}_pval.png", vmin=0, vmax=1, cmap="viridis")

        all_results[pair_type] = {
            "hinge": grid_hinge, "acc": grid_acc, "pval": grid_p, "n": grid_n, "sig_bhfdr": sig,
        }
        best_cell = np.unravel_index(np.nanargmin(grid_hinge), grid_hinge.shape) if valid.any() else None
        log_message(
            f"[{pair_type}] done in {time.time() - t0:.1f}s. "
            f"best (layer,k)={None if best_cell is None else (probe_layers[best_cell[0]], best_cell[1])} "
            f"min_hinge={np.nanmin(grid_hinge) if valid.any() else float('nan'):.3f}"
        )
        # incremental save: a later pair_type crashing must not lose already-computed results
        save_results(out_dir, probe_layers, args, all_results)

    save_results(out_dir, probe_layers, args, all_results)


def save_results(out_dir, probe_layers, args, all_results):
    npz_payload = {}
    for pt, d in all_results.items():
        for key in ("hinge", "acc", "pval", "n", "sig_bhfdr"):
            npz_payload[f"{pt}_{key}"] = d[key]
    np.savez(out_dir / "steerability_grid.npz", probe_layers=np.array(probe_layers), **npz_payload)

    summary = {
        "probe_layers": probe_layers,
        "sigma_schedule": SIGMA_SCHEDULE,
        "pair_types": args.pair_types,
        "n_pairs_online": args.n_pairs_online,
        "online_task": args.online_task,
        "n_perm": args.n_perm,
        "gate": {},
    }
    for pt, d in all_results.items():
        valid = ~np.isnan(d["hinge"])
        if not valid.any():
            summary["gate"][pt] = {"status": "no_valid_cells"}
            continue
        min_idx = np.unravel_index(np.nanargmin(d["hinge"]), d["hinge"].shape)
        summary["gate"][pt] = {
            "min_hinge": float(np.nanmin(d["hinge"])),
            "at_layer": probe_layers[min_idx[0]],
            "at_step_k": int(min_idx[1]),
            "acc_at_min_hinge": float(d["acc"][min_idx]),
            "pval_at_min_hinge": float(d["pval"][min_idx]),
            "n_sig_cells_bhfdr": int(d["sig_bhfdr"].sum()),
            "n_total_cells": int(valid.sum()),
            "gate_pass": bool(d["sig_bhfdr"][min_idx]) if not np.isnan(d["pval"][min_idx]) else False,
        }
    with open(out_dir / "steerability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"Saved summary to {out_dir / 'steerability_summary.json'}")


if __name__ == "__main__":
    main()
