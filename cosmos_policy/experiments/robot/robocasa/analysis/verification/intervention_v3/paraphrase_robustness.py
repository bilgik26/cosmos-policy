"""
paraphrase_robustness.py — 実プロンプトへの依存度合いの検定

design_v3.md Stage 0の L（言語）対比集合は「実プロンプト vs ダミープロンプト」を対比し、
言語条件付けが活性へ与える差（オラクル方向）を測った。しかし「実プロンプトの正確な文字列に
方策が丸暗記的に依存しているのか、それとも意味を理解して汎化しているのか」は別の問いであり、
本スクリプトはこれを直接検定する：**同じ意味を持つが表現の異なるparaphrase文**（訓練時のT5
埋め込みキャッシュには存在しない文字列）を与えたとき、success_rateが実プロンプト（verbatim）
からどれだけ低下するかを測る。

3条件（matched design、episodeごとに同一seedでenv/シーンを揃える）:
  C0_real_verbatim  : 実プロンプト（訓練時キャッシュに存在する文字列そのまま）— 上限
  C1_paraphrase     : 意味的に同一だが表現の異なるparaphrase（動詞・前置詞を変更）
  C2_dummy          : ダミープロンプト（"The weather today is sunny."）— 言語なしの床、参照点

paraphrase文のT5埋め込みは`precompute_paraphrase_embeddings.py`が別プロセス（policyモデルを
ロードしない、T5-11bのみbf16）で事前計算したものを読み込み、`t5_text_embeddings_cache`へ
直接注入する（`preload_dummy_prompt_embedding`と同一パターン）——評価実行中にT5を一切ロード
しないことで、2B policy DiTと同一GPUメモリ上でのOOMを避ける。
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
import cosmos_policy.experiments.robot.cosmos_utils as cosmos_utils
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

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_EMBEDDINGS_PATH = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/"
    "intervention_v3_paraphrase_robustness/paraphrase_embeddings_PnPCounterToCab.pt"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/intervention_v3_paraphrase_robustness"
)


def run_single_rollout(cfg, model, dataset_stats, task_name, seed, ep_idx, task_desc, max_call):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    cfg.task_name = task_name
    env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep_idx)
    obs = env.reset()
    action_queue = deque()
    success = False
    call_idx = 0
    t = 0
    for t in range(max_steps):
        if len(action_queue) == 0 and call_idx < max_call:
            observation = prepare_observation(obs, cfg.flip_images)
            call_seed = seed + call_idx * 131
            r = get_action(
                cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
                task_label_or_embedding=task_desc, seed=call_seed, randomize_seed=False,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=False,
            )
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
    return success, call_idx, t + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--embeddings_path", default=str(DEFAULT_EMBEDDINGS_PATH))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--max_call", type=int, default=40)
    p.add_argument("--conditions", nargs="+",
                    default=["C0_real_verbatim", "C1_paraphrase", "C2_dummy"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.embeddings_path, map_location="cpu")
    episodes = payload["episodes"]
    paraphrase_embeddings = payload["paraphrase_embeddings"]
    log_message(f"Loaded {len(episodes)} episode records and "
                f"{len(paraphrase_embeddings)} unique paraphrase embeddings from {args.embeddings_path}")

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path, t5_text_embeddings_path=args.t5_text_embeddings_path,
        task_name=args.task_name, seed=episodes[0]["seed"],
    )
    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    preload_dummy_prompt_embedding(device="cuda:0")

    device = next(model.parameters()).device
    for phrase, emb in paraphrase_embeddings.items():
        cosmos_utils.t5_text_embeddings_cache[phrase] = emb.to(device)
    log_message(f"Injected {len(paraphrase_embeddings)} paraphrase embedding(s) into t5_text_embeddings_cache "
                f"(no on-the-fly T5 load will occur).")

    cond_to_prompt_key = {
        "C0_real_verbatim": "real_prompt", "C1_paraphrase": "paraphrase_prompt", "C2_dummy": None,
    }

    all_results = {c: [] for c in args.conditions}
    for entry in episodes:
        for cond in args.conditions:
            key = cond_to_prompt_key[cond]
            task_desc = DUMMY_PROMPT if key is None else entry[key]
            success, n_calls, n_steps = run_single_rollout(
                cfg, model, dataset_stats, args.task_name, entry["seed"], entry["episode"], task_desc, args.max_call
            )
            all_results[cond].append({
                "episode": entry["episode"], "seed": entry["seed"], "obj": entry["obj"],
                "task_desc": task_desc, "success": bool(success), "n_calls": n_calls, "n_steps": n_steps,
            })
            log_message(f"  [{cond}] ep={entry['episode']} obj={entry['obj']!r} "
                        f"{'SUCCESS' if success else 'FAIL'} calls={n_calls} steps={n_steps}")

    summary = {"task_name": args.task_name, "n_episodes": len(episodes), "conditions": {}}
    for cname, res in all_results.items():
        successes = np.array([r["success"] for r in res], dtype=float)
        episode_ids = np.arange(len(res))
        ci = episode_bootstrap_ci(successes, episode_ids, n_boot=2000, seed=0, agg=np.mean)
        summary["conditions"][cname] = {
            "success_rate": float(successes.mean()), "n_success": int(successes.sum()),
            "n_episodes": len(res), "bootstrap_ci95": [ci["ci_lo"], ci["ci_hi"]],
        }

    from scipy.stats import fisher_exact

    def fisher(a_name, b_name, alternative):
        a = all_results[a_name]
        b = all_results[b_name]
        a_succ, a_n = int(sum(r["success"] for r in a)), len(a)
        b_succ, b_n = int(sum(r["success"] for r in b)), len(b)
        _, pval = fisher_exact([[a_succ, a_n - a_succ], [b_succ, b_n - b_succ]], alternative=alternative)
        return float(pval)

    pairwise = {}
    if "C1_paraphrase" in all_results and "C0_real_verbatim" in all_results:
        pairwise["C1_paraphrase_vs_C0_real_verbatim_p_less"] = fisher(
            "C1_paraphrase", "C0_real_verbatim", "less")
    if "C1_paraphrase" in all_results and "C2_dummy" in all_results:
        pairwise["C1_paraphrase_vs_C2_dummy_p_greater"] = fisher(
            "C1_paraphrase", "C2_dummy", "greater")
    if "C0_real_verbatim" in all_results and "C2_dummy" in all_results:
        pairwise["C0_real_verbatim_vs_C2_dummy_p_greater"] = fisher(
            "C0_real_verbatim", "C2_dummy", "greater")
    summary["pairwise_fisher_exact"] = pairwise

    with open(out_dir / f"paraphrase_robustness_{args.task_name}.json", "w") as f:
        json.dump({"episodes": episodes, "summary": summary, "episode_results": all_results}, f, indent=2)
    log_message(f"Saved to {out_dir / f'paraphrase_robustness_{args.task_name}.json'}")
    log_message(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
