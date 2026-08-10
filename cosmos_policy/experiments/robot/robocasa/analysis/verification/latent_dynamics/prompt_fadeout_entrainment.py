"""
prompt_fadeout_entrainment.py — ldv_design_v2.md フェーズ6 対応。

§4(dynamic_vector_field_steering.py)は「言語プロンプトを完全に排除した状態」でのベクトル場
steeringを検定し、反証された(§4.5)。ldv_design_v2.mdはこの反証を受け、仮説を「言語の完全排除」
から「言語の役割の再定位」へと修正する: 言語プロンプトは行動系列全体を持続的に記述・制御する
表象ではなく、力学系を特定のアトラクタ盆(初期状態)へ引き込む(entrainment)ための一過性のトリガー
に過ぎず、一度引き込まれれば、その後の行動生成は感覚運動の結合動態(§4のベクトル場)が自律的に
引き受けられるのではないか、という仮説である。

本スクリプトはこれを、エピソード内でのプロンプト動的切り替え("fadeout")により検定する:

  C_full            : エピソード完了まで実プロンプトを与え続ける(steeringなし)。基準条件。
  C_fadeout_only    : 最初の fadeout_calls call のみ実プロンプトを与え、以降はダミープロンプト
                      に切り替える(steeringなし)。「言語を打ち切ると運動が崩壊するか」の基準。
  C_fadeout_dynamic : C_fadeout_only と同じ切り替えタイミングだが、ダミープロンプトに切り替わった
                      直後のcallから dynamic_vector_field_steering.py の動的フィールドsteering
                      (DynamicFieldHook/DynamicFlowField, §4と同一パラメータ)をONにする。

検証の狙い(ldv_design_v2.md §2フェーズ6より): C_fadeout_only で運動が崩壊するにもかかわらず、
C_fadeout_dynamic でタスク完遂(または成功率の有意な上昇)が見られれば、「言語は初期のアトラクタ
選択にのみ必要で、その後の行動生成は言語的表象に依らない」という仮説の具体的な支持になる。
反対に C_fadeout_dynamic も C_fadeout_only と同程度にしか成功しなければ、その仮説はこのタスク・
このパラメータ範囲では支持されない。

実装は dynamic_vector_field_steering.py のオンライン因果推定器・フローライブラリ・steeringフック
(CausalStateEstimator/DynamicFlowField/DynamicFieldHook/FinalFeatCapture)をそのまま再利用し、
"どのcallでどちらのプロンプトを使うか"という制御ロジックのみを新規実装する。T1(alpha=0 no-op)
チェックも同モジュールの t1_noop_check をそのまま再利用する(P6/P7の遵守)。

fadeout_calls のデフォルト(5): ldv_design_v2.md §フェーズ6実装指示2「最初のNステップ(例: 5〜10
steps、対象物へ向かい始める初期フェーズ)」に従い、call粒度(1 call = num_open_loop_steps=16
env-step、design.mdの"ステップ"はこの粒度と解釈した — 個々のenv-stepでは物理的にほぼ何も進行しない
ため)でその範囲の下限を採用した。網羅的なスイープは実施していない(§9参照)。
"""

import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import LAYER
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.steering_intervention import (
    sanitize_actions,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT, preload_dummy_prompt_embedding,
    CausalStateEstimator, DynamicFlowField, DynamicFieldHook, FinalFeatCapture,
    get_action_with_dynamic_hook, xp_direction_to_raw, t1_noop_check,
)

DEFAULT_FADEOUT_CALLS = 5
MAX_CALL = 40


def run_condition(cfg, model, dataset_stats, hook, capture, estimator, field, task_name,
                   condition_name, fadeout_calls, use_dynamic_steering, alpha, n_episodes, base_seed):
    """use_dynamic_steering=False -> C_full (fadeout_calls=10**9, i.e. never switches) or
    C_fadeout_only (switches prompt but never engages the hook). use_dynamic_steering=True ->
    C_fadeout_dynamic (switches prompt AND engages the hook from the same call onward)."""
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_logs = []
    for ep in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=base_seed + ep, episode_idx=ep)
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
        real_desc = env.get_ep_meta().get("lang", task_name)

        estimator.reset_episode()
        hook.vec = None
        hook.alpha = 0.0

        action_queue = deque()
        success = False
        call_idx = 0
        t_step = 0
        eef_traj, grip_traj, action_log, prompt_schedule = [], [], [], []
        xp_traj, d_traj, xp_call_idx = [], [], []
        max_delta_this_ep = 0.0
        nan_detected = False

        while call_idx < MAX_CALL and t_step < max_steps and not success:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                this_desc = real_desc if call_idx < fadeout_calls else DUMMY_PROMPT
                prompt_schedule.append("real" if call_idx < fadeout_calls else "dummy")
                call_seed = base_seed + ep * 131 + call_idx
                result = get_action_with_dynamic_hook(
                    cfg, model, dataset_stats, observation, this_desc, hook, capture,
                    call_seed, cfg.num_denoising_steps_action,
                )
                max_delta_this_ep = max(max_delta_this_ep, hook.last_delta_max)

                gripper_width = float(np.abs(obs["robot0_gripper_qpos"]).sum())
                eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                if capture._final is not None:
                    D_t, Xp_t = estimator.update(capture._final, eef_pos, gripper_width)
                    xp_traj.append(Xp_t.tolist())
                    d_traj.append(D_t.tolist())
                    xp_call_idx.append(call_idx)
                    # NEXT call (call_idx+1) determines whether the hook should be armed: the
                    # hook always affects the call AFTER the one whose features we just captured
                    # (same one-call delay documented in dynamic_vector_field_steering.py §4.2.5).
                    if use_dynamic_steering and (call_idx + 1) >= fadeout_calls:
                        v_xp = field.query(D_t, Xp_t)
                        raw_dir = xp_direction_to_raw(v_xp, field.dyn_artifact)
                        raw_dir_unit = raw_dir / (np.linalg.norm(raw_dir) + 1e-12)
                        hook.vec = raw_dir_unit
                        hook.alpha = alpha
                    else:
                        hook.vec = None
                        hook.alpha = 0.0

                actions, had_nan = sanitize_actions(result["actions"])
                nan_detected = nan_detected or had_nan
                call_idx += 1
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0., 0., 0., 0., -1.])])
                    action_queue.append(a)

            if action_queue:
                action = action_queue.popleft()
                obs, _, _, _ = env.step(action)
                t_step += 1
                eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())
                grip_traj.append(float(np.abs(obs["robot0_gripper_qpos"]).sum()))
                action_log.append(np.asarray(action, dtype=np.float64).tolist())
                if env._check_success():
                    success = True
                    break

        env.close()
        ep_logs.append({
            "success": success, "n_calls": call_idx, "n_steps": t_step,
            "eef_traj": eef_traj, "grip_traj": grip_traj, "action_log": action_log,
            "xp_traj": xp_traj, "d_traj": d_traj, "xp_call_idx": xp_call_idx,
            "prompt_schedule": prompt_schedule,
            "max_hook_delta": max_delta_this_ep, "nan_detected": nan_detected,
        })
        log_message(f"  [{condition_name} ep{ep}] success={success} n_steps={t_step} n_calls={call_idx} "
                    f"max_delta={max_delta_this_ep:.4f} nan={nan_detected}")

    success_rate = float(np.mean([r["success"] for r in ep_logs]))
    log_message(f"[{condition_name}] success_rate={success_rate:.2f} (n={n_episodes})")
    return {"success_rate": success_rate, "episodes": ep_logs}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=8)
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--fadeout_calls", type=int, default=DEFAULT_FADEOUT_CALLS)
    p.add_argument("--k_range", type=int, nargs=2, default=[0, 1])
    p.add_argument("--conditions", nargs="+",
                    default=["C_full", "C_fadeout_only", "C_fadeout_dynamic"])
    args = p.parse_args()

    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(embedding_dir / f"dynamics_embedding_artifact_{args.task_name}.pkl", "rb") as f:
        dyn_artifact = pickle.load(f)

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        use_wrist_image=True, num_wrist_images=1, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path, trained_with_image_aug=True,
        chunk_size=32, num_open_loop_steps=16, task_name=args.task_name, seed=args.seed,
        randomize_seed=False, deterministic=True, use_variance_scale=False,
        use_jpeg_compression=True, flip_images=True, num_denoising_steps_action=5,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1, data_collection=False,
    )
    set_seed_everywhere(args.seed)

    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    preload_dummy_prompt_embedding()

    hook = DynamicFieldHook(LAYER, k_range=tuple(args.k_range))
    hook.register(model)
    capture = FinalFeatCapture(LAYER, cfg.num_denoising_steps_action)
    capture.register(model)

    t1_max_diff = t1_noop_check(cfg, model, dataset_stats, hook, capture, args.task_name, args.seed)
    log_message(f"=== T1 check: max|delta| with alpha=0 hook vs no hook = {t1_max_diff:.2e} ===")
    assert t1_max_diff < 1e-4, f"T1 FAIL: alpha=0 hook is not a no-op (max_diff={t1_max_diff})"

    estimator = CausalStateEstimator(dyn_artifact)
    field = DynamicFlowField(dyn_artifact, randomize=False, seed=0)

    results = {"method_note": (
        f"Phase6: C_full=real prompt whole episode (no steering). C_fadeout_only=real prompt for "
        f"the first fadeout_calls={args.fadeout_calls} calls, dummy prompt afterward (no steering) "
        f"-- tests whether motion collapses once language is withdrawn. C_fadeout_dynamic=same "
        f"prompt schedule as C_fadeout_only, but the dynamic-field steering hook (same field/alpha/"
        f"k_range as dynamic_vector_field_steering.py's C2) is armed starting from the SAME call the "
        f"prompt switches to dummy -- tests whether the flow field can carry the trajectory onward "
        f"once language's initial entrainment has occurred."
    ), "task": args.task_name, "alpha": args.alpha, "fadeout_calls": args.fadeout_calls,
        "k_range": args.k_range, "t1_noop_max_diff": t1_max_diff, "conditions": {}}

    for cond in args.conditions:
        log_message(f"=== Condition: {cond} ===")
        if cond == "C_full":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, fadeout_calls=MAX_CALL + 1,
                               use_dynamic_steering=False, alpha=0.0,
                               n_episodes=args.n_episodes, base_seed=args.seed)
        elif cond == "C_fadeout_only":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, fadeout_calls=args.fadeout_calls,
                               use_dynamic_steering=False, alpha=0.0,
                               n_episodes=args.n_episodes, base_seed=args.seed)
        elif cond == "C_fadeout_dynamic":
            r = run_condition(cfg, model, dataset_stats, hook, capture, estimator, field,
                               args.task_name, cond, fadeout_calls=args.fadeout_calls,
                               use_dynamic_steering=True, alpha=args.alpha,
                               n_episodes=args.n_episodes, base_seed=args.seed)
        else:
            raise ValueError(f"unknown condition {cond}")
        results["conditions"][cond] = r

    capture.remove()
    hook.remove()

    with open(out_dir / f"prompt_fadeout_entrainment_{args.task_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / f'prompt_fadeout_entrainment_{args.task_name}.json'}")


if __name__ == "__main__":
    main()
