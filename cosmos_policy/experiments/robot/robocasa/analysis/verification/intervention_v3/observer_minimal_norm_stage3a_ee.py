"""
observer_minimal_norm_stage3a_ee.py — report_v3.md §6.2 フォローアップ E-4

Stage 3-A (observer_minimal_norm_stage3a.py) はグリッパー開閉 (低次特徴) でBuurmeijer型
観測器＋最小ノルム介入を検証した。design_v3.md §3 Stage3-Aは「まず低次特徴（グリッパー開閉、
EE 高さ、EE 速度）で再現を取る」と明記しており、本ファイルはEE高さ・EE速度への拡張を行う
(第6.2節「今後の検証案」: 「EE高さ・EE速度での再現」)。

3-Aのfit_probe/SetpointHook/get_action_with_hook/t1_noop_checkをそのまま再利用し、
- feature_type="H" (EE高さ): steerability_audit.load_offline_pairsの新規"H"ペア
  (eef_pos z成分の全データプール2-means二値化) で観測器を学習
- feature_type="V" (EE速度): 新規"V"ペア (phase_labeling.py由来のmotion_state、call間
  eef_pos変位の2-means二値化) で観測器を学習
のいずれかを選び、閉ループロールアウトで実際の物理量 (obs["robot0_eef_pos"]) への因果効果を
測定する。collect_multitask.pyがeef_posを各callの直前(get_action呼び出し前)に記録している
のと同じ粒度で、評価時もcall単位でeef_posを記録する。
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

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
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.v4_stats_lib import episode_bootstrap_ci
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT,
    preload_dummy_prompt_embedding,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.steerability_audit import (
    load_offline_pairs,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.intervention_v3.observer_minimal_norm_stage3a import (
    SetpointHook,
    fit_probe,
    get_action_with_hook,
    t1_noop_check,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_COLLECT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_observer_minimal_norm_ee"
)


def select_best_layer_step(collect_dir, pair_type, target_layer, seed=0):
    offline = load_offline_pairs(collect_dir, pair_type, probe_layers=[target_layer])
    per_k = {}
    for k in range(5):
        if target_layer not in offline.get(k, {}):
            continue
        d = offline[k][target_layer]
        y01 = (d["y"] == 1).astype(int)
        res = fit_probe(d["X"], y01, d["groups"], seed=seed)
        per_k[k] = res
        log_message(f"  [probe:{pair_type}] layer={target_layer} k={k} cv_acc={res['cv_acc']:.3f} "
                    f"n={res['n_samples']} n_groups={res['n_groups']}")
    best_k = max(per_k.keys(), key=lambda k: (per_k[k]["cv_acc"] if not np.isnan(per_k[k]["cv_acc"]) else -1))
    return best_k, per_k


def run_condition(cfg, model, dataset_stats, hook, task_name, feature_type, condition_name, mode,
                   n_episodes, base_seed, max_call):
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
        height_traj, speed_traj, intervene_logs = [], [], []
        prev_eef = None
        for t in range(max_steps):
            if len(action_queue) == 0 and call_idx < max_call:
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                height_traj.append(float(eef_pos[2]))
                if prev_eef is not None:
                    speed_traj.append(float(np.linalg.norm(eef_pos - prev_eef)))
                prev_eef = eef_pos

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
            "mean_height": float(np.mean(height_traj)) if height_traj else float("nan"),
            "final_height": float(height_traj[-1]) if height_traj else float("nan"),
            "mean_speed": float(np.mean(speed_traj)) if speed_traj else float("nan"),
            "n_calls_intervened": int(sum(l["intervened"] for l in intervene_logs)),
            "constraint_satisfaction_rate": constraint_sat,
            "mean_u_over_x": float(np.mean(u_over_x)) if u_over_x else float("nan"),
        })
        log_message(f"  [{condition_name}] ep {ep + 1}/{n_episodes} "
                    f"{'SUCCESS' if success else 'FAIL'} calls={call_idx} "
                    f"mean_height={ep_logs[-1]['mean_height']:.3f} mean_speed={ep_logs[-1]['mean_speed']:.4f} "
                    f"constraint_sat={constraint_sat if not np.isnan(constraint_sat) else float('nan'):.3f}")
    return ep_logs


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
    p.add_argument("--feature_type", choices=["H", "V"], required=True,
                    help="H=EE height (eef_pos z), V=EE velocity (motion_state)")
    p.add_argument("--target_layer", type=int, default=13)
    p.add_argument("--target_k", type=int, default=None, help="None = auto-select best CV-acc step")
    p.add_argument("--n_episodes_eval", type=int, default=8)
    p.add_argument("--max_call_eval", type=int, default=40)
    p.add_argument("--conditions", nargs="+", default=["off", "force_open", "force_closed"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_dir = Path(args.collect_dir)

    log_message(f"=== Stage 3-A/EE step 1: fitting observer ({args.feature_type}-pair) ===")
    best_k, per_k = select_best_layer_step(collect_dir, args.feature_type, args.target_layer, seed=args.seed)
    target_k = args.target_k if args.target_k is not None else best_k
    probe = per_k[target_k]
    log_message(f"[probe] feature={args.feature_type} layer={args.target_layer} k={target_k} "
                f"(auto-selected best={best_k}) cv_acc={probe['cv_acc']:.3f}")

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

    log_message("=== step 2: T1 no-op check (mode=off) ===")
    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, args.task_name, args.seed)
    log_message(f"T1 max|delta| (mode=off vs no hook) = {t1_max_diff:.2e}")
    assert t1_max_diff < 1e-4, f"T1 FAIL: mode=off hook is not a no-op (max_diff={t1_max_diff})"

    log_message("=== step 3: closed-loop eval ===")
    mode_map = {"off": "off", "force_open": "force_open", "force_closed": "force_closed"}
    all_results = {}
    for cond in args.conditions:
        cname = f"E_{args.feature_type}_{cond}"
        log_message(f"=== condition {cname} ===")
        res = run_condition(cfg, model, dataset_stats, hook, args.task_name, args.feature_type,
                             cname, mode_map[cond], args.n_episodes_eval, args.seed, args.max_call_eval)
        all_results[cname] = res

    summary = {
        "feature_type": args.feature_type, "target_layer": args.target_layer,
        "target_k": target_k, "auto_best_k": best_k,
        "probe_cv_acc": probe["cv_acc"], "probe_n_samples": probe["n_samples"],
        "probe_n_groups": probe["n_groups"],
        "zeta_closed_median": probe["zeta_closed_median"], "zeta_open_median": probe["zeta_open_median"],
        "per_k_cv_acc": {str(k): v["cv_acc"] for k, v in per_k.items()},
        "t1_noop_max_diff": t1_max_diff,
        "conditions": {},
    }
    for cname, res in all_results.items():
        successes = np.array([r["success"] for r in res], dtype=float)
        episode_ids = np.arange(len(res))
        ci = episode_bootstrap_ci(successes, episode_ids, n_boot=2000, seed=0, agg=np.mean)
        heights = np.array([r["mean_height"] for r in res])
        speeds = np.array([r["mean_speed"] for r in res])
        summary["conditions"][cname] = {
            "success_rate": float(successes.mean()), "n_success": int(successes.sum()),
            "n_episodes": len(res), "bootstrap_ci95": [ci["ci_lo"], ci["ci_hi"]],
            "mean_height": float(np.nanmean(heights)), "std_height": float(np.nanstd(heights)),
            "mean_speed": float(np.nanmean(speeds)), "std_speed": float(np.nanstd(speeds)),
            "mean_constraint_satisfaction_rate": float(np.nanmean(
                [r["constraint_satisfaction_rate"] for r in res])),
            "mean_u_over_x": float(np.nanmean([r["mean_u_over_x"] for r in res])),
        }

    off_key = f"E_{args.feature_type}_off"
    if off_key in all_results:
        from scipy.stats import mannwhitneyu
        metric_key = "mean_height" if args.feature_type == "H" else "mean_speed"
        base_vals = np.array([r[metric_key] for r in all_results[off_key]])
        for cname, res in all_results.items():
            if cname == off_key:
                continue
            cond_vals = np.array([r[metric_key] for r in res])
            try:
                stat, pval = mannwhitneyu(cond_vals, base_vals, alternative="two-sided")
                summary["conditions"][cname][f"mannwhitney_{metric_key}_vs_off_p"] = float(pval)
            except ValueError:
                summary["conditions"][cname][f"mannwhitney_{metric_key}_vs_off_p"] = None

    with open(out_dir / f"observer_minimal_norm_ee_{args.feature_type}_{args.task_name}.json", "w") as f:
        json.dump({"summary": summary, "episode_results": all_results}, f, indent=2)
    log_message(f"Saved to {out_dir / f'observer_minimal_norm_ee_{args.feature_type}_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
