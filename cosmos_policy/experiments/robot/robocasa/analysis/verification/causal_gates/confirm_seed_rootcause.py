"""Confirm root cause: extended_step_effrank_jb.py/t8/t8b pass a CONSTANT seed=cfg.seed
to every policy call (freezing the initial diffusion noise x_T across all calls), while
feature_analysis.py (the good run) varies seed=cfg.seed+ep_idx+t per call.

Clean design: gather a FIXED set of observations once (closed-loop rollout across
n_episodes), then feed this IDENTICAL observation set through the model TWICE --
once with a frozen seed (bug-reproducing) and once with a varying seed (good-run
convention) -- and compare PR. This isolates the seed effect with zero scene-sampling
confound (same observations in both conditions).
"""
import os
os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")
from collections import deque
import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS, PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import PROBE_LAYERS, effective_rank

ACTION_T_IDX = 5


class SimpleCapture:
    def __init__(self, layers):
        self.layers = layers
        self._step = -1
        self.records = []
        self._cur = {}

    def register(self, model):
        for l in self.layers:
            model.net.blocks[l].register_forward_hook(self._hook(l))

    def reset(self):
        self._step = -1
        self._cur = {}

    def before(self):
        self._step += 1
        self._cur.setdefault(self._step, {})

    def finalize(self):
        self.records.append({k: dict(v) for k, v in self._cur.items()})

    def _hook(self, l):
        def hook(module, inp, out):
            if self._step < 0 or not (isinstance(out, torch.Tensor) and out.dim() == 5):
                return
            feat = out[0, ACTION_T_IDX].float().mean(dim=(0, 1))
            self._cur.setdefault(self._step, {})[l] = feat.detach().cpu().numpy()
        return hook


def get_action_capture(cfg, model, dataset_stats, obs, task_desc, cap, call_seed, n_steps):
    cap.reset()
    orig = model.get_x0_fn_from_batch
    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result
            def w(x, s):
                cap.before()
                return fn(x, s)
            return w, extra
        else:
            def w(x, s):
                cap.before()
                return result(x, s)
            return w
    model.get_x0_fn_from_batch = patched
    try:
        res = get_action(cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
                          task_label_or_embedding=task_desc, seed=call_seed, randomize_seed=False,
                          num_denoising_steps_action=n_steps, generate_future_state_and_value_in_parallel=False)
    finally:
        model.get_x0_fn_from_batch = orig
    return res


def gather_pool(cfg, model, dataset_stats, env, n_episodes):
    """Just gather (obs, task_desc) pairs via a real closed-loop rollout -- no capture yet."""
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    pool = []
    for ep in range(n_episodes):
        obs = env.reset()
        task_desc = env.get_ep_meta().get("lang", cfg.task_name)
        step_count = 0
        done = False
        q = deque()
        while not done and step_count < max_steps:
            if len(q) == 0:
                o = prepare_observation(obs, cfg.flip_images)
                pool.append((o, task_desc))
                call_seed = cfg.seed + len(pool)
                res = get_action(cfg=cfg, model=model, dataset_stats=dataset_stats, obs=o,
                                  task_label_or_embedding=task_desc, seed=call_seed, randomize_seed=False,
                                  num_denoising_steps_action=cfg.num_denoising_steps_action,
                                  generate_future_state_and_value_in_parallel=False)
                actions = res["actions"]
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0., 0., 0., 0., -1.])])
                    q.append(a)
            action = q.popleft()
            obs, r, done, info = env.step(action)
            step_count += 1
            if env._check_success():
                done = True
        log_message(f"pool-gather ep {ep} done, total calls so far {len(pool)}")
    return pool


def report_pr(cap, n_steps, label):
    log_message(f"=== PR report: {label} (N calls={len(cap.records)}) ===")
    for l in PROBE_LAYERS:
        for k in range(n_steps):
            feats = [rec.get(k, {}).get(l) for rec in cap.records]
            feats = np.stack([f for f in feats if f is not None])
            if feats.shape[0] < 5:
                continue
            _, sv, _ = np.linalg.svd(feats - feats.mean(axis=0), full_matrices=False)
            pr = effective_rank(sv)
            if k == 0:
                log_message(f"  Blk-{l:2d} k={k} N={feats.shape[0]} PR={pr:.2f}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--n_steps", type=int, default=5)
    p.add_argument("--n_episodes", type=int, default=10)
    args = p.parse_args()

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        use_wrist_image=True, num_wrist_images=1, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path, trained_with_image_aug=True,
        chunk_size=32, num_open_loop_steps=16, task_name="PnPCounterToCab", seed=195,
        randomize_seed=False, deterministic=True, use_variance_scale=False,
        use_jpeg_compression=True, flip_images=True, num_denoising_steps_action=args.n_steps,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1, data_collection=False,
    )
    env, _ = create_robocasa_env(cfg)
    torch.cuda.set_device(0)
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cu.DEVICE = torch.device("cuda:0")

    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=0)
    model, _ = get_model(cfg)
    model.eval()

    set_seed_everywhere(cfg.seed)
    log_message("=== Phase 1: gather fixed observation pool (single closed-loop rollout) ===")
    pool = gather_pool(cfg, model, dataset_stats, env, args.n_episodes)
    log_message(f"Pool size: {len(pool)} calls")

    cap = SimpleCapture(PROBE_LAYERS)
    cap.register(model)

    log_message("=== Phase 2: replay pool with FROZEN seed (bug-reproducing) ===")
    cap.records = []
    for (o, task_desc) in pool:
        get_action_capture(cfg, model, dataset_stats, o, task_desc, cap, cfg.seed, args.n_steps)
        cap.finalize()
    report_pr(cap, args.n_steps, "FROZEN seed (matches extended_step_effrank_jb.py bug)")

    log_message("=== Phase 3: replay SAME pool with VARYING seed (feature_analysis.py convention) ===")
    cap.records = []
    for i, (o, task_desc) in enumerate(pool):
        get_action_capture(cfg, model, dataset_stats, o, task_desc, cap, cfg.seed + i, args.n_steps)
        cap.finalize()
    report_pr(cap, args.n_steps, "VARYING seed (matches feature_analysis.py convention)")


if __name__ == "__main__":
    main()
