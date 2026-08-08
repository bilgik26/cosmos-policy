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

print("=== Pattern A: single env, repeated reset() (t8b style) ===")
env, _ = create_robocasa_env(cfg)
for i in range(15):
    obs = env.reset()
    meta = env.get_ep_meta()
    lang = meta.get("lang", "<MISSING>")
    print(f"  reset #{i}: lang={lang!r}")

print("=== Pattern B: new env per episode, episode_idx=ep, seed=cfg.seed+ep (feature_analysis.py style) ===")
for ep in range(15):
    env_b, _ = create_robocasa_env(cfg, seed=cfg.seed + ep, episode_idx=ep)
    obs = env_b.reset()
    meta = env_b.get_ep_meta()
    lang = meta.get("lang", "<MISSING>")
    print(f"  ep #{ep}: lang={lang!r}")
