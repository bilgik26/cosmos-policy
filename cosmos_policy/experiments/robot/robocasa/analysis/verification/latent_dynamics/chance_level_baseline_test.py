"""
chance_level_baseline_test.py — review_report.md (latent_dynamics_verification 査読) D-6/S-9/L-5
対応。ポリシー(2Bモデル)を一切ロードせず、RoboCasa環境に対してスクリプト化した行動を直接
適用するのみ(GPU上のEGLレンダリングのみ必要、拡散モデル推論は不要のため既存のGPU実験と比べて
桁違いに軽い)。

D-6/S-9 (chance-level baseline): 「言語なしでアクションを駆動する」問題設定に対し、chance level
  (=もっとも単純な非知的方策でどの程度成功するか)が一度も測られていなかった(review_report.md
  §1 D-6)。ゼロ行動(全ステップaction=0)とランダム定行動(episode開始時に1回サンプルした
  固定行動ベクトルを全ステップ適用)の2種を、C1〜C5と同じ2タスク(PnPCounterToCab・CloseDrawer)
  でN=20 episode実施する。

L-5 (CloseDrawer C5偶発的成功の検証): 「引き出しへ向かう一定方向の運動の反復で機械的に閉まりうる」
  というreport_v2.md §2.4の事後的仮説を、ランダム定行動ベースラインのCloseDrawer成功率で直接
  検証する。有意な成功率が出れば「一定運動の反復閉扉」仮説を支持する追加証拠になる。
"""

import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env

N_EPISODES = 20


def run_scripted_condition(cfg, task_name, condition_name, action_fn, n_episodes, base_seed):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_logs = []
    for ep in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=base_seed + ep, episode_idx=ep)
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
        rng = np.random.RandomState(base_seed + ep + 5000)
        const_action = action_fn(env, rng)
        success, t_step = False, 0
        eef_traj = []
        while t_step < max_steps and not success:
            obs, _, _, _ = env.step(const_action)
            t_step += 1
            eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())
            if env._check_success():
                success = True
                break
        env.close()
        ep_logs.append({"success": success, "n_steps": t_step})
        log_message(f"  [{condition_name} ep{ep}] success={success} n_steps={t_step}")
    sr = float(np.mean([e["success"] for e in ep_logs]))
    log_message(f"[{condition_name}] success_rate={sr:.3f} (n={n_episodes})")
    from scipy.stats import binomtest
    ci = binomtest(int(sum(e["success"] for e in ep_logs)), n_episodes).proportion_ci(confidence_level=0.95)
    return {"success_rate": sr, "n_episodes": n_episodes, "n_success": int(sum(e["success"] for e in ep_logs)),
            "clopper_pearson_ci95": [float(ci.low), float(ci.high)], "episodes": ep_logs}


def zero_action(env, rng):
    return np.zeros(env.action_spec[0].shape)


def random_constant_action(env, rng):
    low, high = env.action_spec
    a = rng.uniform(low, high)
    return a


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=["PnPCounterToCab", "CloseDrawer"])
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=N_EPISODES)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"n_episodes": args.n_episodes, "tasks": {}}
    for task in args.tasks:
        cfg = PolicyEvalConfig(
            config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
            use_wrist_image=True, num_wrist_images=1, use_proprio=True, normalize_proprio=True,
            unnormalize_actions=True, dataset_stats_path=args.dataset_stats_path,
            t5_text_embeddings_path=args.t5_text_embeddings_path, trained_with_image_aug=True,
            chunk_size=32, num_open_loop_steps=16, task_name=task, seed=args.seed,
            randomize_seed=False, deterministic=True, use_variance_scale=False,
            use_jpeg_compression=True, flip_images=True, num_denoising_steps_action=5,
            num_denoising_steps_future_state=1, num_denoising_steps_value=1, data_collection=False,
        )
        log_message(f"=== Task: {task} ===")
        r_zero = run_scripted_condition(cfg, task, f"{task}_zero_action", zero_action, args.n_episodes, args.seed)
        r_rand = run_scripted_condition(cfg, task, f"{task}_random_constant_action", random_constant_action,
                                         args.n_episodes, args.seed)
        results["tasks"][task] = {"zero_action": r_zero, "random_constant_action": r_rand}
        with open(out_dir / "chance_level_baseline_test.json", "w") as f:
            json.dump(results, f, indent=2)

    with open(out_dir / "chance_level_baseline_test.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / 'chance_level_baseline_test.json'}")


if __name__ == "__main__":
    main()
