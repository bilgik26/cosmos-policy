"""
graded_dv_rerun.py — review_report.md (latent_dynamics_verification 査読) MUST-3 対応。

C1〜C5(ダミープロンプト条件)の主従属変数がsuccess_rate(二値、n=8)であり、対照条件が軒並み0/8
だったため、本検証は「成功率30%未満の効果」を検出する検出力を構造的に持たない(review_report.md
§1 D-3、0/8のClopper-Pearson片側95%上限=0.312)。既存のロールアウトログ(eef_traj/grip_traj)には
対象物体の位置が保存されていないため、床のないgraded指標(対象物体までの最小距離・接触イベント数・
物体変位量)は事後解析では計算できず、再走が必要(§7-D-3 MUST-3「既存ログ再解析+一部再走」の
「一部再走」部分)。

dynamic_vector_field_steering.run_condition と全く同じ条件設計・同じseed・同じalpha・同じ
k_rangeでPnPCounterToCabの6条件(C0〜C5)を再実行し、以下を追加でログする:
  - object_dist_traj: 各env-stepでの ||eef_pos - obj_pos|| (対象物体="obj"、PnP.pyの命名規則)
  - min_object_dist: エピソード内最小値(「対象物体にどれだけ近づいたか」の連続指標)
  - gripper_contact_events: env.check_contact(gripper, obj)がFalse->Trueに遷移した回数
    (把持試行回数の直接的な代理指標、grip_trajのような間接推定ではない)
  - object_displacement: ||obj_pos_end - obj_pos_start||(対象物体が実際にどれだけ動かされたか)

success_rateとT1チェックは元のrun_conditionと完全に同一の実装(dynamic_vector_field_steering.py
からrun_conditionをそのまま再利用、この再走に新規性はない)。唯一の追加はobs取得直後に
env.sim.data.body_xpos[env.obj_body_id["obj"]]とenv.check_contact()を呼ぶラッパーのみ。
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation
from cosmos_policy.experiments.robot.cosmos_utils import get_model, load_dataset_stats, init_t5_text_embeddings_cache
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.skill_count import LAYER, K_STEP
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.steering_intervention import (
    compute_steering_vectors, sanitize_actions,
)
import pickle
from cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamic_vector_field_steering import (
    DUMMY_PROMPT, CausalStateEstimator, DynamicFlowField, DynamicFieldHook, FinalFeatCapture,
    get_action_with_dynamic_hook, xp_direction_to_raw, t1_noop_check, preload_dummy_prompt_embedding,
)

TARGET_OBJ_KEY = "obj"


def run_condition_graded(cfg, model, dataset_stats, hook, capture, estimator, field, task_name,
                          condition_name, task_desc, use_steering, alpha, n_episodes, base_seed,
                          static_vec_unit=None, max_call=40, frozen_obs=False):
    max_steps = TASK_MAX_STEPS.get(task_name, 500)
    ep_logs = []
    for ep in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=base_seed + ep, episode_idx=ep)
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
        this_task_desc = task_desc if task_desc is not None else env.get_ep_meta().get("lang", task_name)

        has_obj = TARGET_OBJ_KEY in getattr(env, "obj_body_id", {})
        obj_pos_start = (np.array(env.sim.data.body_xpos[env.obj_body_id[TARGET_OBJ_KEY]]).copy()
                          if has_obj else None)

        estimator.reset_episode()
        hook.vec, hook.alpha = None, 0.0
        if static_vec_unit is not None:
            hook.vec, hook.alpha = static_vec_unit, alpha

        action_queue = deque()
        success, call_idx, t_step = False, 0, 0
        eef_traj, grip_traj, action_log = [], [], []
        xp_traj, d_traj, xp_call_idx = [], [], []
        object_dist_traj, contact_flags = [], []
        max_delta_this_ep, nan_detected = 0.0, False
        frozen_observation = None

        while call_idx < max_call and t_step < max_steps and not success:
            if len(action_queue) == 0:
                if frozen_obs and frozen_observation is not None:
                    observation = frozen_observation
                else:
                    observation = prepare_observation(obs, cfg.flip_images)
                    if frozen_obs:
                        frozen_observation = observation
                call_seed = base_seed + ep * 131 + call_idx
                result = get_action_with_dynamic_hook(cfg, model, dataset_stats, observation, this_task_desc,
                                                        hook, capture, call_seed, cfg.num_denoising_steps_action)
                max_delta_this_ep = max(max_delta_this_ep, hook.last_delta_max)

                gripper_width = float(np.abs(obs["robot0_gripper_qpos"]).sum())
                eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                if capture._final is not None:
                    D_t, Xp_t = estimator.update(capture._final, eef_pos, gripper_width)
                    xp_traj.append(Xp_t.tolist())
                    d_traj.append(D_t.tolist())
                    xp_call_idx.append(call_idx)
                    if use_steering and static_vec_unit is None:
                        v_xp = field.query(D_t, Xp_t)
                        raw_dir = xp_direction_to_raw(v_xp, field.dyn_artifact)
                        raw_dir_unit = raw_dir / (np.linalg.norm(raw_dir) + 1e-12)
                        hook.vec, hook.alpha = raw_dir_unit, alpha

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
                eef_now = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                eef_traj.append(eef_now.tolist())
                grip_traj.append(float(np.abs(obs["robot0_gripper_qpos"]).sum()))
                action_log.append(np.asarray(action, dtype=np.float64).tolist())
                if has_obj:
                    obj_pos_now = np.array(env.sim.data.body_xpos[env.obj_body_id[TARGET_OBJ_KEY]])
                    object_dist_traj.append(float(np.linalg.norm(eef_now - obj_pos_now)))
                    try:
                        contact = bool(env.check_contact(env.robots[0].gripper, env.objects[TARGET_OBJ_KEY]))
                    except Exception:
                        contact = False
                    contact_flags.append(contact)
                if env._check_success():
                    success = True
                    break

        obj_pos_end = (np.array(env.sim.data.body_xpos[env.obj_body_id[TARGET_OBJ_KEY]]).copy()
                        if has_obj else None)
        env.close()

        contact_events = 0
        prev = False
        for c in contact_flags:
            if c and not prev:
                contact_events += 1
            prev = c

        ep_logs.append({
            "success": success, "n_calls": call_idx, "n_steps": t_step,
            "eef_traj": eef_traj, "grip_traj": grip_traj,
            "min_object_dist": float(np.min(object_dist_traj)) if object_dist_traj else None,
            "mean_object_dist": float(np.mean(object_dist_traj)) if object_dist_traj else None,
            "gripper_contact_events": contact_events,
            "object_displacement": (float(np.linalg.norm(obj_pos_end - obj_pos_start))
                                     if has_obj else None),
            "max_hook_delta": max_delta_this_ep, "nan_detected": nan_detected,
        })
        log_message(f"  [{condition_name} ep{ep}] success={success} n_steps={t_step} "
                    f"min_obj_dist={ep_logs[-1]['min_object_dist']} contacts={contact_events} "
                    f"obj_displacement={ep_logs[-1]['object_displacement']}")

    success_rate = float(np.mean([r["success"] for r in ep_logs]))
    min_dists = [r["min_object_dist"] for r in ep_logs if r["min_object_dist"] is not None]
    log_message(f"[{condition_name}] success_rate={success_rate:.2f} (n={n_episodes}) "
                f"mean(min_object_dist)={np.mean(min_dists) if min_dists else float('nan'):.4f}")
    return {"success_rate": success_rate, "episodes": ep_logs}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--collect_dir", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=8)
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--k_range", type=int, nargs=2, default=[0, 1])
    args = p.parse_args()

    collect_dir = Path(args.collect_dir)
    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(embedding_dir / f"dynamics_embedding_artifact_{args.task_name}.pkl", "rb") as f:
        dyn_artifact = pickle.load(f)
    manifest = json.loads((collect_dir / "multitask_manifest.json").read_text())

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
    assert t1_max_diff < 1e-4, f"T1 FAIL: {t1_max_diff}"
    log_message(f"=== T1 check passed: {t1_max_diff:.2e} ===")

    estimator = CausalStateEstimator(dyn_artifact)
    field = DynamicFlowField(dyn_artifact, randomize=False, seed=0)
    random_field = DynamicFlowField(dyn_artifact, randomize=True, seed=1)
    static_vecs = compute_steering_vectors(collect_dir, manifest, args.task_name, LAYER, K_STEP, seed=0)
    static_vec_unit = static_vecs["v_steer"] / np.linalg.norm(static_vecs["v_steer"])

    results = {"method_note": "MUST-3 graded-DV rerun (min object distance, gripper-object contact "
                               "events, object displacement) of the original 6 phase-3 conditions, "
                               "identical seeds/alpha/k_range to dynamic_vector_field_steering.py.",
               "task": args.task_name, "alpha": args.alpha, "k_range": args.k_range,
               "t1_noop_max_diff": t1_max_diff, "conditions": {}}

    conditions = [
        ("C0_real_prompt_no_steer", None, False, 0.0, None, False),
        ("C1_dummy_no_steer", DUMMY_PROMPT, False, 0.0, None, False),
        ("C2_dummy_dynamic_field", DUMMY_PROMPT, True, args.alpha, None, False),
        ("C3_dummy_static_v_steer", DUMMY_PROMPT, True, args.alpha, static_vec_unit, False),
        ("C4_dummy_random_field", DUMMY_PROMPT, True, args.alpha, None, False),
        ("C5_dummy_dynamic_field_frozen_obs", DUMMY_PROMPT, True, args.alpha, None, True),
    ]
    for cond, task_desc, use_steering, alpha, static_vec, frozen_obs in conditions:
        log_message(f"=== Condition: {cond} ===")
        this_field = random_field if cond == "C4_dummy_random_field" else field
        r = run_condition_graded(cfg, model, dataset_stats, hook, capture, estimator, this_field,
                                  args.task_name, cond, task_desc, use_steering, alpha, args.n_episodes,
                                  args.seed, static_vec_unit=static_vec, frozen_obs=frozen_obs)
        results["conditions"][cond] = r
        with open(out_dir / f"graded_dv_rerun_{args.task_name}.json", "w") as f:
            json.dump(results, f, indent=2)

    capture.remove()
    hook.remove()
    with open(out_dir / f"graded_dv_rerun_{args.task_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {out_dir / f'graded_dv_rerun_{args.task_name}.json'}")


if __name__ == "__main__":
    main()
