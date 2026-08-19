"""
observer_minimal_norm_stage3a.py — design_v3.md Stage 3-A 準拠 (観測器ベースの最小ノルム介入)

Buurmeijer "Observing and Controlling" (arXiv:2603.05487) の枠組みを、Stage 0
(steerability_audit.py) が (Blk-13, k=3) を G (グリッパー) 対比集合の最良セルと同定した
結果の上に実装する。design_v3.md §3 Stage 3-A の指示通り、**まず低次特徴（グリッパー開閉）
で再現を取る**——これが再現できなければ実装自体を疑うべき、という設計上の位置づけである。

手順:
  1. `attractor/collect_multitask.py` が既に収集した held-out データ (results/
     attractor_verification/collect/) から、Blk-13 の各 denoising step で事前計算済みの
     pooled 活性 (`feat_k{k}_layer13`) と phase_labeling.py 由来の gripper_state ラベルを
     読み (load_offline_pairs を再利用、新規GPU計算不要)、線形観測器 f(x)=W^Tx+b
     (標準化ロジスティック回帰) を学習する。交差検証で最良の denoising step を確認する
     (Stage 0 の (Blk-13, k=3) と一致するか)。
  2. Remark 1 (Buurmeijer): プローブのロバスト性を確認する。held-out活性に相対摂動を加え、
     ζ=f(x) の変化量・分類の安定性を測る。
  3. 閉形式の最小ノルムsetpoint介入: ζ_ℓ = W_raw・pooled_feat + b_raw を計算し、目標区間
     [ζ_min, ζ_max] の外なら u_ℓ = (ζ_target − ζ_ℓ)·W_raw/‖W_raw‖² を action-token出力
     (全空間位置に均一加算 — これは pooled ζ の目標達成に関して full-token空間でも
     minimal-normであることが Lagrange 条件から従う、本ファイル docstring 末尾参照) に加える。
     Blk-13・k=3(Stage0最良セル)でのみ介入する。
  4. 閉ループロールアウトで **制約充足率**（介入直後にζが目標区間に入ったか）と
     **実際の物理グリッパー開閉度**（`robot0_gripper_qpos`、観測器が依拠するラベルの
     元になった生の物理量）を同一の実行で同時に測定する（design_v3.md §5 作法1・2の遵守）。
     ダミープロンプト下で3条件（介入なし／open方向へ強制／closed方向へ強制）を比較する。

**最小ノルム性の補足**: 観測器はpooled特徴 (H×W空間平均) の上で学習されているため、
「pooled ζ をtargetへ動かす」という制約だけを見たとき、token全体 (H,W,2048) の空間への
最小L2ノルム補正は「補正を全空間位置に一様に加える」ことで達成される
(Σ_{h,w}‖u_hw‖² を mean_{h,w}(u_hw)=u_target の制約下で最小化すると u_hw=u_target ∀h,w、
ラグランジュ未定乗数法より自明)。したがって既存のsteeringフック群 (DynamicFieldHook等) と
同じ「action-tokenの全空間位置に均一加算」という注入方式は、pooled観測器に対しては
文字通りの最小ノルム解になっている。
"""

import argparse
import json
from collections import deque
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
    TASK_MAX_STEPS,
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import ACTION_T_IDX
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import episode_bootstrap_ci
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT,
    preload_dummy_prompt_embedding,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.steerability_audit import (
    load_offline_pairs,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_COLLECT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_observer_minimal_norm"
)


# ────────────────────────────── 1. observer (linear probe) fitting ──────────────────────────────

def fit_probe(X, y01, groups, n_splits=5, seed=0):
    """標準化ロジスティック回帰。raw-space (W_raw, b_raw) と CV精度を返す。
    ζ(x) = W_raw・x + b_raw が、標準化空間の分類器 W_std・((x-mu)/sd) + b_std と
    数学的に同一になるように変換する。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    uniq_groups = np.unique(groups)
    n_splits_eff = min(n_splits, len(uniq_groups))
    accs = []
    if n_splits_eff >= 2:
        gkf = GroupKFold(n_splits=n_splits_eff)
        for train_idx, test_idx in gkf.split(X, y01, groups=groups):
            if len(np.unique(y01[train_idx])) < 2 or len(np.unique(y01[test_idx])) < 2:
                continue
            mu, sd = X[train_idx].mean(axis=0), X[train_idx].std(axis=0) + 1e-8
            Xtr, Xte = (X[train_idx] - mu) / sd, (X[test_idx] - mu) / sd
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y01[train_idx])
            accs.append(clf.score(Xte, y01[test_idx]))
    cv_acc = float(np.mean(accs)) if accs else float("nan")

    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xs = (X - mu) / sd
    clf_full = LogisticRegression(max_iter=2000, C=1.0)
    clf_full.fit(Xs, y01)
    W_std = clf_full.coef_[0]
    b_std = float(clf_full.intercept_[0])
    W_raw = W_std / sd
    b_raw = float(b_std - np.sum(W_std * mu / sd))
    zeta = X @ W_raw + b_raw
    return {
        "W_raw": W_raw, "b_raw": b_raw, "cv_acc": cv_acc, "n_folds": len(accs),
        "zeta": zeta, "zeta_closed_median": float(np.median(zeta[y01 == 1])),
        "zeta_open_median": float(np.median(zeta[y01 == 0])),
        "n_samples": len(y01), "n_groups": len(uniq_groups),
    }


def select_best_layer_step(collect_dir, target_layer, seed=0):
    """target_layer の全5 denoising stepでプローブを学習し、CV精度最良のkを選ぶ。
    Stage 0 (steerability_audit.py) が (Blk-13, k=3) を最良セルと報告したことの独立な追試。"""
    offline = load_offline_pairs(collect_dir, "G", probe_layers=[target_layer])
    per_k = {}
    for k in range(5):
        if target_layer not in offline.get(k, {}):
            continue
        d = offline[k][target_layer]
        y01 = (d["y"] == 1).astype(int)
        res = fit_probe(d["X"], y01, d["groups"], seed=seed)
        per_k[k] = res
        log_message(f"  [probe] layer={target_layer} k={k} cv_acc={res['cv_acc']:.3f} "
                    f"n={res['n_samples']} n_groups={res['n_groups']}")
    best_k = max(per_k.keys(), key=lambda k: (per_k[k]["cv_acc"] if not np.isnan(per_k[k]["cv_acc"]) else -1))
    return best_k, per_k


# ────────────────────────────── 2. robustness check (Buurmeijer Remark 1) ──────────────────────────────

def robustness_check(X, W_raw, b_raw, eps_rel_list, n_trials=200, seed=0):
    """held-out活性に相対摂動 (‖delta‖ = eps_rel*‖x‖, ランダム方向) を加え、
    ζ の変化量と決定境界の安定性 (フリップ率) を測る。線形写像なので |Δζ|<=‖W_raw‖‖delta‖
    が理論的に保証されること自体は自明だが、実測して確認する。"""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n_trials, len(X)), replace=False)
    Xs = X[idx]
    zeta0 = Xs @ W_raw + b_raw
    W_norm = float(np.linalg.norm(W_raw))
    out = {}
    for eps_rel in eps_rel_list:
        deltas = rng.normal(size=Xs.shape)
        deltas = deltas / (np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-12)
        deltas = deltas * (eps_rel * np.linalg.norm(Xs, axis=1, keepdims=True))
        zeta1 = (Xs + deltas) @ W_raw + b_raw
        d_zeta = zeta1 - zeta0
        theoretical_bound = W_norm * eps_rel * np.linalg.norm(Xs, axis=1)
        flip = (np.sign(zeta0) != np.sign(zeta1)).mean()
        out[eps_rel] = {
            "mean_abs_d_zeta": float(np.mean(np.abs(d_zeta))),
            "mean_theoretical_bound": float(np.mean(theoretical_bound)),
            "bound_satisfied_frac": float(np.mean(np.abs(d_zeta) <= theoretical_bound + 1e-6)),
            "sign_flip_frac": float(flip),
        }
    return out, W_norm


# ────────────────────────────── 3. minimal-norm setpoint intervention hook ──────────────────────────────

class SetpointHook:
    """target_layer の action-token出力を、target_k のdenoising stepでのみ観測器ζ=W_raw・x+b_raw
    を目標区間へ最小ノルムで補正する (Buurmeijer closed-form)。mode='off'なら常にno-op。"""

    def __init__(self, layer, target_k, W_raw, b_raw, zeta_min_closed, zeta_max_open):
        self.layer = layer
        self.target_k = target_k
        self.W_raw = W_raw
        self.b_raw = b_raw
        self.zeta_min_closed = zeta_min_closed
        self.zeta_max_open = zeta_max_open
        self.mode = "off"  # "off" | "force_closed" | "force_open"
        self.step = -1
        self.handle = None
        self.last_log = None  # dict(zeta_before, zeta_after, u_norm, x_norm)

    def register(self, model):
        self.handle = model.net.blocks[self.layer].register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

    def reset_step(self):
        self.step = -1
        self.last_log = None

    def before_step(self):
        self.step += 1

    def _hook(self, module, inp, output):
        if not (isinstance(output, torch.Tensor) and output.dim() == 5):
            return output
        if self.mode == "off" or self.step != self.target_k:
            return output
        feat = output[0, ACTION_T_IDX].float().mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)
        zeta = float(self.W_raw @ feat + self.b_raw)
        W_norm_sq = float(self.W_raw @ self.W_raw)
        if self.mode == "force_closed" and zeta < self.zeta_min_closed:
            target = self.zeta_min_closed
        elif self.mode == "force_open" and zeta > self.zeta_max_open:
            target = self.zeta_max_open
        else:
            self.last_log = {"zeta_before": zeta, "zeta_after": zeta, "u_norm": 0.0,
                              "x_norm": float(np.linalg.norm(feat)), "intervened": False}
            return output
        u = (target - zeta) * self.W_raw / W_norm_sq
        feat_after = feat + u
        zeta_after = float(self.W_raw @ feat_after + self.b_raw)
        self.last_log = {"zeta_before": zeta, "zeta_after": zeta_after, "zeta_target": target,
                          "u_norm": float(np.linalg.norm(u)), "x_norm": float(np.linalg.norm(feat)),
                          "intervened": True}
        u_t = torch.tensor(u, dtype=output.dtype, device=output.device)
        output = output.clone()
        output[0, ACTION_T_IDX] = output[0, ACTION_T_IDX] + u_t
        return output


def get_action_with_hook(cfg, model, dataset_stats, obs, task_desc, hook, seed):
    hook.reset_step()
    orig = model.get_x0_fn_from_batch

    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result

            def w(x, s):
                hook.before_step()
                return fn(x, s)

            return w, extra
        else:
            def w(x, s):
                hook.before_step()
                return result(x, s)

            return w

    model.get_x0_fn_from_batch = patched
    try:
        r = get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
            task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = orig
    return r


def t1_noop_check(cfg, model, dataset_stats, hook, task_name, base_seed):
    env, _ = create_robocasa_env(cfg, seed=base_seed + 9999, episode_idx=0)
    obs = env.reset()
    task_desc = env.get_ep_meta().get("lang", task_name)
    observation = prepare_observation(obs, cfg.flip_images)
    env.close()

    hook.mode = "off"
    res_with = get_action_with_hook(cfg, model, dataset_stats, observation, task_desc, hook, base_seed + 12345)
    res_without = get_action(
        cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
        task_label_or_embedding=task_desc, seed=base_seed + 12345, randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
    )
    return float(np.max(np.abs(np.stack(res_with["actions"]) - np.stack(res_without["actions"]))))


# ────────────────────────────── 4. closed-loop eval ──────────────────────────────

def run_condition(cfg, model, dataset_stats, hook, task_name, condition_name, mode, n_episodes, base_seed, max_call):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_logs = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        cfg.task_name = task_name
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep)
        obs = env.reset()
        hook.mode = mode
        action_queue = deque()
        success = False
        call_idx = 0
        t = 0
        grip_traj, intervene_logs = [], []
        for t in range(max_steps):
            if len(action_queue) == 0 and call_idx < max_call:
                observation = prepare_observation(obs, cfg.flip_images)
                call_seed = seed + call_idx * 131
                r = get_action_with_hook(cfg, model, dataset_stats, observation, DUMMY_PROMPT, hook, call_seed)
                if hook.last_log is not None:
                    intervene_logs.append(hook.last_log)
                actions = r["actions"]
                call_idx += 1
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    action_queue.append(a)
            if not action_queue:
                break
            action = action_queue.popleft()
            obs, _, _, _ = env.step(action)
            grip_traj.append(float(np.abs(obs["robot0_gripper_qpos"]).sum()))
            if env._check_success():
                success = True
                break
        env.close()
        constraint_sat = (
            float(np.mean([abs(l["zeta_after"] - l.get("zeta_target", l["zeta_after"])) < 1e-3
                            for l in intervene_logs if l["intervened"]]))
            if any(l["intervened"] for l in intervene_logs) else float("nan")
        )
        u_over_x = [l["u_norm"] / (l["x_norm"] + 1e-8) for l in intervene_logs if l["intervened"]]
        ep_logs.append({
            "episode": ep, "success": bool(success), "n_calls": call_idx, "n_steps": t + 1,
            "mean_gripper_width": float(np.mean(grip_traj)) if grip_traj else float("nan"),
            "final_gripper_width": float(grip_traj[-1]) if grip_traj else float("nan"),
            "n_calls_intervened": int(sum(l["intervened"] for l in intervene_logs)),
            "constraint_satisfaction_rate": constraint_sat,
            "mean_u_over_x": float(np.mean(u_over_x)) if u_over_x else float("nan"),
        })
        log_message(f"  [{condition_name}] ep {ep + 1}/{n_episodes} "
                    f"{'SUCCESS' if success else 'FAIL'} calls={call_idx} "
                    f"mean_grip={ep_logs[-1]['mean_gripper_width']:.3f} "
                    f"constraint_sat={constraint_sat if not np.isnan(constraint_sat) else float('nan'):.3f} "
                    f"n_intervened={ep_logs[-1]['n_calls_intervened']}")
    return ep_logs


# ────────────────────────────── main ──────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--collect_dir", default=str(DEFAULT_COLLECT_DIR))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--target_layer", type=int, default=13)
    p.add_argument("--target_k", type=int, default=None, help="None = auto-select best CV-acc step")
    p.add_argument("--eps_rel_list", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--n_episodes_eval", type=int, default=8)
    p.add_argument("--max_call_eval", type=int, default=40)
    p.add_argument("--conditions", nargs="+", default=["off", "force_open", "force_closed"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_dir = Path(args.collect_dir)

    log_message("=== Stage 3-A step 1: fitting observer (linear probe) on G-pair offline data ===")
    best_k, per_k = select_best_layer_step(collect_dir, args.target_layer, seed=args.seed)
    target_k = args.target_k if args.target_k is not None else best_k
    probe = per_k[target_k]
    log_message(f"[probe] using layer={args.target_layer} k={target_k} (auto-selected best={best_k}) "
                f"cv_acc={probe['cv_acc']:.3f}")

    log_message("=== Stage 3-A step 2: robustness check (Remark 1) ===")
    offline_full = load_offline_pairs(collect_dir, "G", probe_layers=[args.target_layer])
    X_all = offline_full[target_k][args.target_layer]["X"]
    robustness, W_norm = robustness_check(X_all, probe["W_raw"], probe["b_raw"], args.eps_rel_list, seed=args.seed)
    log_message(f"[robustness] ||W_raw||={W_norm:.4f} results={json.dumps(robustness, indent=2)}")

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path, t5_text_embeddings_path=args.t5_text_embeddings_path,
        task_name=args.task_name, seed=args.seed,
    )
    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    preload_dummy_prompt_embedding(device="cuda:0")

    hook = SetpointHook(args.target_layer, target_k, probe["W_raw"], probe["b_raw"],
                         probe["zeta_closed_median"], probe["zeta_open_median"])
    hook.register(model)

    log_message("=== Stage 3-A step 3: T1 no-op check (mode=off) ===")
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, args.task_name, args.seed)
    log_message(f"T1 max|delta| (mode=off vs no hook) = {t1_max_diff:.2e}")
    assert t1_max_diff < 1e-4, f"T1 FAIL: mode=off hook is not a no-op (max_diff={t1_max_diff})"

    log_message("=== Stage 3-A step 4: closed-loop eval ===")
    mode_map = {"off": "off", "force_open": "force_open", "force_closed": "force_closed"}
    all_results = {}
    for cond in args.conditions:
        cname = f"E_{cond}"
        log_message(f"=== condition {cname} ===")
        res = run_condition(cfg, model, dataset_stats, hook, args.task_name, cname, mode_map[cond],
                             args.n_episodes_eval, args.seed, args.max_call_eval)
        all_results[cname] = res

    summary = {
        "target_layer": args.target_layer, "target_k": target_k, "auto_best_k": best_k,
        "probe_cv_acc": probe["cv_acc"], "probe_n_samples": probe["n_samples"],
        "probe_n_groups": probe["n_groups"], "W_norm": W_norm,
        "zeta_closed_median": probe["zeta_closed_median"], "zeta_open_median": probe["zeta_open_median"],
        "per_k_cv_acc": {str(k): v["cv_acc"] for k, v in per_k.items()},
        "robustness": {str(e): v for e, v in robustness.items()},
        "t1_noop_max_diff": t1_max_diff,
        "conditions": {},
    }
    for cname, res in all_results.items():
        successes = np.array([r["success"] for r in res], dtype=float)
        episode_ids = np.arange(len(res))
        ci = episode_bootstrap_ci(successes, episode_ids, n_boot=2000, seed=0, agg=np.mean)
        grip_widths = np.array([r["mean_gripper_width"] for r in res])
        summary["conditions"][cname] = {
            "success_rate": float(successes.mean()), "n_success": int(successes.sum()),
            "n_episodes": len(res), "bootstrap_ci95": [ci["ci_lo"], ci["ci_hi"]],
            "mean_gripper_width": float(np.nanmean(grip_widths)),
            "std_gripper_width": float(np.nanstd(grip_widths)),
            "mean_constraint_satisfaction_rate": float(np.nanmean(
                [r["constraint_satisfaction_rate"] for r in res])),
            "mean_u_over_x": float(np.nanmean([r["mean_u_over_x"] for r in res])),
        }

    if "E_off" in all_results:
        from scipy.stats import mannwhitneyu
        base_grip = np.array([r["mean_gripper_width"] for r in all_results["E_off"]])
        for cname, res in all_results.items():
            if cname == "E_off":
                continue
            cond_grip = np.array([r["mean_gripper_width"] for r in res])
            try:
                stat, pval = mannwhitneyu(cond_grip, base_grip, alternative="two-sided")
                summary["conditions"][cname]["mannwhitney_gripper_vs_off_p"] = float(pval)
            except ValueError:
                summary["conditions"][cname]["mannwhitney_gripper_vs_off_p"] = None

    per_k_probe_json = {
        str(k): {
            "cv_acc": v["cv_acc"], "n_folds": int(v["n_folds"]), "n_samples": int(v["n_samples"]),
            "n_groups": int(v["n_groups"]), "zeta_closed_median": v["zeta_closed_median"],
            "zeta_open_median": v["zeta_open_median"], "b_raw": v["b_raw"],
        }
        for k, v in per_k.items()
    }
    with open(out_dir / f"observer_minimal_norm_{args.task_name}.json", "w") as f:
        json.dump({
            "per_k_probe": per_k_probe_json,
            "robustness": robustness, "summary": summary, "episode_results": all_results,
        }, f, indent=2)
    log_message(f"Saved to {out_dir / f'observer_minimal_norm_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
