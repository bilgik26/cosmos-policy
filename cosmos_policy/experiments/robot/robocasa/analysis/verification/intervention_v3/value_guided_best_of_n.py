"""
value_guided_best_of_n.py — design_v3.md §4.2 準拠 (value ガイド付き best-of-N ベースライン)

「ダミープロンプト下で行動チャンクを N 本サンプルし、value ヘッドで選択する」だけの
最も安価なベースライン。前回レポート群 (report.md/report_v2.md) が繰り返し観測した
「ダミープロンプト条件は success_rate が 0.00 に張り付く」という床効果を、活性への
介入なしに (test-time compute のみで) 打開できるかを検定する。

条件:
  C0_real_promptN1  : 実プロンプト、N=1  (上限ベースライン)
  C1_dummy_N1        : ダミープロンプト、N=1、選択なし (前回レポートの床、下限)
  C2_dummy_bestofN    : ダミープロンプト、call毎に N 本サンプルし value_prediction
                         (get_action の "generate_future_state_and_value_in_parallel" 経路、
                         cosmos_utils.extract_value_from_latent_sequence) の argmax を実行

matched design: 全条件で同一の base_seed+episode_idx を使う (env/シーンを揃える)。
value 選択に使う各クエリの sampling seed は条件・callをまたいで衝突しないよう
決定的に振る (base_seed, episode, call_idx, query_idx から導出)。
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_value_from_latent_sequence,
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
    preload_dummy_prompt_embedding,
)

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_value_best_of_n"
)
DUMMY_PROMPT = "The weather today is sunny."


def query_seed(base_seed, ep_idx, call_idx, query_idx):
    return int(base_seed) * 1_000_000 + int(ep_idx) * 10_000 + int(call_idx) * 100 + int(query_idx)


def run_condition(cfg, model, dataset_stats, task_name, condition_name, task_desc, best_of_n,
                   n_episodes, base_seed, max_call=200):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    episode_results = []
    for ep in range(n_episodes):
        cfg.task_name = task_name
        env, _ = create_robocasa_env(cfg, seed=base_seed + ep, episode_idx=ep)
        obs = env.reset()
        action_queue = deque()
        success = False
        call_idx = 0
        n_gen = best_of_n  # need value_prediction whenever we do selection
        value_trace = []
        t0 = time.time()
        t = 0
        for t in range(max_steps):
            if len(action_queue) == 0 and call_idx < max_call:
                observation = prepare_observation(obs, cfg.flip_images)
                best_value, best_actions = -1.0, None
                for q in range(n_gen):
                    seed_q = query_seed(base_seed, ep, call_idx, q)
                    # NOTE: generate_future_state_and_value_in_parallel=True also VAE-decodes
                    # 3 future images per query, which OOMs a 24GB GPU once n_gen>1 (see
                    # report_v3.md bug list). We only need the value scalar, which is already
                    # computable from `generated_latent` (always returned) via
                    # extract_value_from_latent_sequence directly, matching get_action's own
                    # internal convention (value_indices=-1) without the expensive decode.
                    r = get_action(
                        cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                        task_label_or_embedding=task_desc, seed=seed_q, randomize_seed=False,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=False,
                    )
                    value_indices = torch.full(
                        (1,), -1, dtype=torch.int64, device=r["generated_latent"].device
                    )
                    with torch.inference_mode():
                        v_raw = extract_value_from_latent_sequence(r["generated_latent"], value_indices)
                    v = float(torch.clamp((v_raw + 1) / 2, 0, 1).item())
                    if v > best_value:
                        best_value, best_actions = v, r["actions"]
                actions = best_actions
                value_trace.append(best_value)
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
        dt = time.time() - t0
        episode_results.append({
            "episode": ep, "success": bool(success), "n_calls": call_idx, "n_steps": t + 1,
            "mean_selected_value": float(np.mean(value_trace)) if value_trace else float("nan"),
            "elapsed_sec": dt,
        })
        log_message(
            f"  [{condition_name}] ep {ep + 1}/{n_episodes} "
            f"{'SUCCESS' if success else 'FAIL'} calls={call_idx} steps={t + 1} "
            f"mean_val={episode_results[-1]['mean_selected_value']:.3f} ({dt:.1f}s)"
        )
    return episode_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=8)
    p.add_argument("--best_of_n_values", type=int, nargs="+", default=[1, 8])
    p.add_argument("--conditions", nargs="+",
                    default=["C0_real_promptN1", "C1_dummy_N1", "C2_dummy_bestofN"])
    p.add_argument("--max_call", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # env-only pass to get the real task description (matches suite convention)
    tmp_env, _ = create_robocasa_env(cfg, seed=args.seed, episode_idx=0)
    real_task_desc = tmp_env.get_ep_meta().get("lang", args.task_name)
    tmp_env.close()
    log_message(f"real_task_desc = {real_task_desc!r}")

    all_results = {}
    for cond in args.conditions:
        if cond == "C0_real_promptN1":
            task_desc, best_of_n = real_task_desc, 1
        elif cond == "C1_dummy_N1":
            task_desc, best_of_n = DUMMY_PROMPT, 1
        elif cond == "C2_dummy_bestofN":
            for n in args.best_of_n_values:
                if n == 1:
                    continue
                cname = f"C2_dummy_bestof{n}"
                log_message(f"=== Condition {cname} (task={args.task_name}) ===")
                res = run_condition(cfg, model, dataset_stats, args.task_name, cname, DUMMY_PROMPT,
                                     n, args.n_episodes, args.seed, max_call=args.max_call)
                all_results[cname] = res
            continue
        else:
            raise ValueError(cond)
        log_message(f"=== Condition {cond} (task={args.task_name}) ===")
        res = run_condition(cfg, model, dataset_stats, args.task_name, cond, task_desc,
                             best_of_n, args.n_episodes, args.seed, max_call=args.max_call)
        all_results[cond] = res

    summary = {"task_name": args.task_name, "n_episodes": args.n_episodes,
               "best_of_n_values": args.best_of_n_values, "conditions": {}}
    for cname, res in all_results.items():
        successes = np.array([r["success"] for r in res], dtype=float)
        episode_ids = np.arange(len(res))
        ci = episode_bootstrap_ci(successes, episode_ids, n_boot=2000, seed=0, agg=np.mean)
        summary["conditions"][cname] = {
            "success_rate": float(successes.mean()),
            "n_success": int(successes.sum()),
            "n_episodes": len(res),
            "bootstrap_ci95": [ci["ci_lo"], ci["ci_hi"]],
            "mean_n_calls": float(np.mean([r["n_calls"] for r in res])),
            "mean_selected_value": float(np.nanmean([r["mean_selected_value"] for r in res])),
        }

    # Fisher exact test: each best-of-N condition vs C1_dummy_N1
    if "C1_dummy_N1" in all_results:
        from scipy.stats import fisher_exact
        base = all_results["C1_dummy_N1"]
        base_succ = int(sum(r["success"] for r in base))
        base_n = len(base)
        for cname, res in all_results.items():
            if cname == "C1_dummy_N1":
                continue
            succ = int(sum(r["success"] for r in res))
            n = len(res)
            table = [[succ, n - succ], [base_succ, base_n - base_succ]]
            _, pval = fisher_exact(table, alternative="greater")
            summary["conditions"][cname]["fisher_vs_C1_dummy_N1_p_greater"] = float(pval)

    with open(out_dir / f"value_best_of_n_{args.task_name}.json", "w") as f:
        json.dump({"episode_results": all_results, "summary": summary}, f, indent=2)
    log_message(f"Saved to {out_dir / f'value_best_of_n_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
