"""
basin_convergence.py — attractor_verification_design.md §5.A 準拠

confirm_seed_rootcause.py の「固定観測・seed掃引」パターンをアトラクタ測定に転用する。

手順:
  1. 各タスクで短いロールアウト (3 episodes) を行い、(obs, task_desc) の pool を集める
     (この過程で各 call は default 1-seed で Z_pool[k,layer] も記録する → これが
     「条件付けランダム化」nullの代わりに使う実データ: 多数の異なる条件付け×1 seed)。
  2. pool から各タスク代表点を2つ選択 (進行度 25%, 75%)し、その (obs, task_desc) を
     **固定**して M=48 seed で replay、Z_pool[k,layer] を record。
  3. 各 k で Var_seed[固定条件付け] (M=48 の分散) を計算。
  4. Null: 主収集済みデータ (collect_multitask.py の出力 = 多様な条件付け×1 seed) から
     N=48 をランダム抽出したときの分散を 200 回ブートストラップし、Var_condition[k] の
     分布を作る。固定条件付けの Var_seed[k] がこの分布の下側に有意に位置するか検定。
  5. 最終ステップ (k=last) での固定条件付けサンプルのモード数を GMM (BIC選択, 1-3成分)で推定。

判定 (反証条件, 設計書§3):
  - Var_seed が k とともに減らない、または conditioning-null と有意差がない
    → 「basin構造なし」と報告 (反証)。
  - Var_seed(k=last) が null 分布より有意に小さい かつ 単調減少
    → 「conditioning-driven basin convergence」を支持。
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import PROBE_LAYERS

BASIN_LAYERS = [4, 9, 13]  # mid layers per design emphasis
N_SEED_SWEEP = 48
POOL_EPISODES_PER_TASK = 3


class SweepCapture:
    def __init__(self, layers):
        self.layers = layers
        self._step = -1
        self._cur = {}
        self.records = []

    def register(self, model):
        for l in self.layers:
            model.net.blocks[l].register_forward_hook(self._hook(l))

    def reset(self):
        self._step = -1
        self._cur = {}

    def before(self):
        self._step += 1
        self._cur.setdefault(self._step, {})

    def finalize(self):
        self.records.append({k: dict(v) for k, v in self._cur.items()})

    def _hook(self, l):
        def hook(module, inp, out):
            if self._step < 0 or not (isinstance(out, torch.Tensor) and out.dim() == 5):
                return
            feat = out[0, 5].float().mean(dim=(0, 1))
            self._cur.setdefault(self._step, {})[l] = feat.detach().cpu().numpy()
        return hook


def get_action_capture(cfg, model, dataset_stats, obs, task_desc, cap, call_seed, n_steps):
    cap.reset()
    orig = model.get_x0_fn_from_batch
    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result
            def w(x, s):
                cap.before()
                return fn(x, s)
            return w, extra
        else:
            def w(x, s):
                cap.before()
                return result(x, s)
            return w
    model.get_x0_fn_from_batch = patched
    try:
        res = get_action(cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
                          task_label_or_embedding=task_desc, seed=call_seed, randomize_seed=False,
                          num_denoising_steps_action=n_steps, generate_future_state_and_value_in_parallel=False)
    finally:
        model.get_x0_fn_from_batch = orig
    return res


def gather_pool_with_capture(cfg, model, dataset_stats, env, task_name, n_episodes, cap, seed_base):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    pool = []  # list of (obs, task_desc, progress_bin_placeholder)
    for ep in range(n_episodes):
        obs = env.reset()
        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)
        task_desc = env.get_ep_meta().get("lang", task_name)
        step_count = 0
        done = False
        q = deque()
        call_idx = 0
        ep_calls = []
        while not done and step_count < max_steps:
            if len(q) == 0:
                o = prepare_observation(obs, cfg.flip_images)
                call_seed = seed_base + ep * 97 + call_idx
                res = get_action_capture(cfg, model, dataset_stats, o, task_desc, cap,
                                          call_seed, cfg.num_denoising_steps_action)
                cap.finalize()
                ep_calls.append((o, task_desc))
                actions = res["actions"]
                call_idx += 1
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0., 0., 0., 0., -1.])])
                    q.append(a)
            action = q.popleft()
            obs, r, done, info = env.step(action)
            step_count += 1
            if env._check_success():
                done = True
        n_calls = len(ep_calls)
        for i, (o, td) in enumerate(ep_calls):
            progress = i / max(n_calls - 1, 1)
            pool.append({"obs": o, "task_desc": td, "progress": progress, "episode": ep})
        log_message(f"  [basin-pool {task_name}] ep {ep} done, {n_calls} calls, total pool={len(pool)}")
    return pool


def var_by_k(records, layers):
    """records: list of {k: {layer: vec}}. Returns {layer: [var_k0, var_k1, ...]}"""
    n_steps = max(max(r.keys()) for r in records if r) + 1
    out = {}
    for l in layers:
        vs = []
        for k in range(n_steps):
            feats = [r.get(k, {}).get(l) for r in records]
            feats = np.stack([f for f in feats if f is not None])
            if feats.shape[0] < 2:
                vs.append(np.nan)
                continue
            # total variance = sum of per-dim variances (trace of covariance)
            vs.append(float(feats.var(axis=0, ddof=1).sum()))
        out[l] = vs
    return out


def gmm_mode_count(X, max_k=3, seed=0):
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    X = X.astype(np.float64)
    # cap components well below N (M=48 seed-sweep samples split across up to
    # max_k GMM components can leave near-singleton covariances); a fixed-conditioning
    # seed sweep is EXPECTED to be tightly concentrated (that is the basin hypothesis
    # itself), so reg_covar must be generous rather than tuned per-run.
    n_comp = min(5, X.shape[0] // 4, X.shape[1])
    Xp = PCA(n_components=max(n_comp, 1), random_state=seed).fit_transform(X)
    # scale-invariant regularization floor (reg_covar is added to the diagonal in
    # the ORIGINAL units of Xp, so it must track Xp's own variance scale)
    reg = max(1e-6, float(np.var(Xp)) * 1e-3)
    bics = []
    for k in range(1, max_k + 1):
        try:
            gm = GaussianMixture(n_components=k, random_state=seed, n_init=5,
                                  reg_covar=reg, covariance_type="diag").fit(Xp)
            bics.append(gm.bic(Xp))
        except Exception:
            bics.append(np.inf)
    if all(np.isinf(b) for b in bics):
        return 1, [float(b) for b in bics]
    best_k = int(np.nanargmin(bics)) + 1
    return best_k, [float(b) for b in bics]


def load_null_pool_variance(collect_dir: Path, layer: int, k: int, manifest):
    """主収集データから (k,layer) の全call特徴を集め、null分散のブートストラップ元にする。"""
    all_feats = []
    for fname in manifest["files"].keys():
        d = np.load(collect_dir / fname)
        key = f"feat_k{k}_layer{layer}"
        if key in d:
            all_feats.append(d[key])
    return np.concatenate(all_feats) if all_feats else np.zeros((0, 1))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--collect_dir", required=True, help="main collection dir (for conditioning-varies null)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tasks", default="PnPCounterToCab,CloseDrawer,TurnOnStove,CoffeePressButton")
    p.add_argument("--seed", type=int, default=195)
    args = p.parse_args()

    tasks = args.tasks.split(",")
    collect_dir = Path(args.collect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        use_wrist_image=True, num_wrist_images=1, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path, trained_with_image_aug=True,
        chunk_size=32, num_open_loop_steps=16, task_name=tasks[0], seed=args.seed,
        randomize_seed=False, deterministic=True, use_variance_scale=False,
        use_jpeg_compression=True, flip_images=True, num_denoising_steps_action=5,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1, data_collection=False,
    )
    set_seed_everywhere(args.seed)

    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    cap = SweepCapture(BASIN_LAYERS)
    cap.register(model)

    results = {}
    for task_name in tasks:
        cfg.task_name = task_name
        env, _ = create_robocasa_env(cfg, seed=args.seed + 5000, episode_idx=0)

        cap.records = []
        pool = gather_pool_with_capture(
            cfg, model, dataset_stats, env, task_name, POOL_EPISODES_PER_TASK, cap, args.seed
        )
        pool_records = cap.records  # 1 record per pool element, aligned by index
        env.close()

        # pick representative fixed points: progress ~0.25 and ~0.75
        progresses = np.array([e["progress"] for e in pool])
        idx_25 = int(np.argmin(np.abs(progresses - 0.25)))
        idx_75 = int(np.argmin(np.abs(progresses - 0.75)))

        task_result = {"pool_size": len(pool), "fixed_points": {}}

        for tag, idx in [("progress_25pct", idx_25), ("progress_75pct", idx_75)]:
            fixed_obs = pool[idx]["obs"]
            fixed_task_desc = pool[idx]["task_desc"]

            cap.records = []
            for i in range(N_SEED_SWEEP):
                seed_i = args.seed + 90000 + i
                get_action_capture(cfg, model, dataset_stats, fixed_obs, fixed_task_desc,
                                    cap, seed_i, cfg.num_denoising_steps_action)
                cap.finalize()
            sweep_records = cap.records

            var_seed = var_by_k(sweep_records, BASIN_LAYERS)

            # mode count at final k for each layer (concat layers for a joint PCA/GMM view)
            n_steps = max(max(r.keys()) for r in sweep_records) + 1
            k_last = n_steps - 1
            joint_final = np.concatenate(
                [np.stack([r[k_last][l] for r in sweep_records]) for l in BASIN_LAYERS], axis=1
            )
            mode_k, bics = gmm_mode_count(joint_final)

            # null comparison at k_last, per layer: bootstrap N=48 draws from main collection
            null_stats = {}
            for l in BASIN_LAYERS:
                null_pool = load_null_pool_variance(collect_dir, l, k_last, manifest)
                if null_pool.shape[0] < N_SEED_SWEEP:
                    null_stats[l] = {"note": "insufficient null pool size"}
                    continue
                rng = np.random.RandomState(0)
                boot_vars = []
                for _ in range(200):
                    sub = null_pool[rng.choice(null_pool.shape[0], N_SEED_SWEEP, replace=False)]
                    boot_vars.append(float(sub.var(axis=0, ddof=1).sum()))
                boot_vars = np.array(boot_vars)
                real_var = var_seed[l][k_last]
                p_val = float((np.sum(boot_vars <= real_var) + 1) / (len(boot_vars) + 1))
                null_stats[l] = {
                    "null_boot_mean": float(boot_vars.mean()),
                    "null_boot_std": float(boot_vars.std()),
                    "real_var_seed_k_last": real_var,
                    "p_value_real_lt_null": p_val,
                    "basin_supported": bool(p_val < 0.05 and real_var < np.percentile(boot_vars, 5)),
                }

            task_result["fixed_points"][tag] = {
                "pool_progress": float(pool[idx]["progress"]),
                "var_seed_by_k": var_seed,
                "n_denoise_steps": n_steps,
                "mode_count_k_last": mode_k,
                "gmm_bics": bics,
                "null_comparison_k_last": null_stats,
                "monotonic_decrease": {
                    str(l): bool(np.all(np.diff([v for v in var_seed[l] if not np.isnan(v)]) <= 1e-9))
                    for l in BASIN_LAYERS
                },
            }
            log_message(
                f"[{task_name}/{tag}] var_seed(k=last) Blk-13={var_seed.get(13, [np.nan]*5)[k_last]:.4f} "
                f"mode_count={mode_k}"
            )

        results[task_name] = task_result

    with open(out_dir / "basin_convergence.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'basin_convergence.json'}")


if __name__ == "__main__":
    main()
