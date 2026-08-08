"""Lightweight diagnostic (no policy/model needed): does calling env.reset() repeatedly on a
SINGLE persistent env (the pattern used by extended_step_effrank_jb.py / t8 / t8b) actually
change the kitchen scene layout, or does it silently reuse the same layout every time?

Compares against feature_analysis.py's pattern of creating a NEW env per episode with
episode_idx=ep_idx and seed=cfg.seed+ep_idx.
"""
import os
os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import PolicyEvalConfig, create_robocasa_env

cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
    ckpt_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    task_name="PnPCounterToCab", seed=195, randomize_seed=False, deterministic=True,
)

print("=== Pattern A: single env, repeated reset() (extended_step/t8/t8b style) ===")
env, _ = create_robocasa_env(cfg)
for i in range(6):
    obs = env.reset()
    meta = env.get_ep_meta()
    layout_id = meta.get("layout_id", "?")
    style_id = meta.get("style_id", "?")
    print(f"  reset #{i}: layout_id={layout_id} style_id={style_id} "
          f"obj_pos_sample={obs.get('robot0_eef_pos', 'NA')[:3] if 'robot0_eef_pos' in obs else 'NA'}")

print("=== Pattern B: new env per episode, episode_idx=ep, seed=cfg.seed+ep (feature_analysis.py style) ===")
for ep in range(6):
    env_b, _ = create_robocasa_env(cfg, seed=cfg.seed + ep, episode_idx=ep)
    obs = env_b.reset()
    meta = env_b.get_ep_meta()
    layout_id = meta.get("layout_id", "?")
    style_id = meta.get("style_id", "?")
    print(f"  ep #{ep}: layout_id={layout_id} style_id={style_id} "
          f"obj_pos_sample={obs.get('robot0_eef_pos', 'NA')[:3] if 'robot0_eef_pos' in obs else 'NA'}")
