import os
os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig, create_robocasa_env, prepare_observation,
)

cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
    ckpt_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    task_name="PnPCounterToCab", seed=195, randomize_seed=False, deterministic=True,
)

print("=== Pattern A: single env, repeated reset() ===")
env, _ = create_robocasa_env(cfg)
imgs_a = []
for i in range(8):
    obs_raw = env.reset()
    obs = prepare_observation(obs_raw, cfg.flip_images)
    imgs_a.append(obs["primary_image"].astype(np.float32))

print("=== Pattern B: new env per episode, episode_idx=ep, seed=cfg.seed+ep ===")
imgs_b = []
for ep in range(8):
    env_b, _ = create_robocasa_env(cfg, seed=cfg.seed + ep, episode_idx=ep)
    obs_raw = env_b.reset()
    obs = prepare_observation(obs_raw, cfg.flip_images)
    imgs_b.append(obs["primary_image"].astype(np.float32))

def pairwise_diffs(imgs, label):
    print(f"--- {label} pairwise mean abs pixel diff ---")
    n = len(imgs)
    diffs = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.abs(imgs[i]-imgs[j]).mean()
            diffs.append(d)
    diffs = np.array(diffs)
    print(f"  mean={diffs.mean():.3f} std={diffs.std():.3f} min={diffs.min():.3f} max={diffs.max():.3f}")
    return diffs

pairwise_diffs(imgs_a, "Pattern A")
pairwise_diffs(imgs_b, "Pattern B")
