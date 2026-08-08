"""
collect_multitask.py — attractor_verification_design.md §4 準拠のマルチタスク収集スクリプト

設計書の要求 (縮小スコープ版, ユーザー承認済み):
  - タスク集合 (4種、異種スキル): PnPCounterToCab (pick-place), CloseDrawer (push-close),
    TurnOnStove (rotate knob), CoffeePressButton (press)
  - 各タスク N_ep エピソード (縮小: 18, design原案は≥25)
  - 2 seed系列で独立に反復 (縮小: 2, design原案は3)
  - P1: call毎 seed = seed_base + ep_idx + t (feature_analysis.py と同じ規約)
  - P2: 単一 run・単一 manifest。全 npz に run_id/sha256 を記録
  - フェーズラベル付け用の物理量 (robot0_gripper_qpos, robot0_eef_pos, robot0_eef_quat) を
    call 時点の raw obs から捕捉 (潜在非依存、循環回避 §4.3)
  - τ軸 (Z_tau) の捕捉は本収集ではスコープ外 (時間予算の都合で明示的に縮退。
    k軸 (basin) は別スクリプト basin_convergence.py で専用収集する)

出力:
  results/attractor_verification/collect/<task>_seed{S}.npz
  results/attractor_verification/collect/multitask_manifest.json
"""

import hashlib
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

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
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    ACTION_LATENT_IDX_ROBOCASA,
    NUM_DENOISE_STEPS,
    PROBE_LAYERS,
)

DEFAULT_TASKS = ["PnPCounterToCab", "CloseDrawer", "TurnOnStove", "CoffeePressButton"]


# ── Feature + physical-state capture ─────────────────────────────────────────

class MultiTaskCapture:
    """feature_analysis.FeatureCapture 相当 + 物理量記録を追加。"""

    def __init__(self, probe_layers: List[int]):
        self.probe_layers = probe_layers
        self.call_records: List[Dict] = []
        self._current_step = -1
        self._current_feats: Dict = {}
        self._handles: List = []

    def register(self, model):
        net = model.net
        for layer_idx in self.probe_layers:
            block = net.blocks[layer_idx]
            h = block.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset_for_policy_call(self):
        self._current_step = -1
        self._current_feats = {}

    def before_denoise_step(self):
        self._current_step += 1
        self._current_feats.setdefault(self._current_step, {})

    def finalize_policy_call(self, meta: Dict):
        rec = {"step_feats": {k: dict(lf) for k, lf in self._current_feats.items()}}
        rec.update(meta)
        self.call_records.append(rec)

    def _make_hook(self, layer_idx: int):
        def hook(module, inp, output):
            if self._current_step < 0:
                return
            k = self._current_step
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return
            B, T, H, W, D = output.shape
            if T <= ACTION_LATENT_IDX_ROBOCASA:
                return
            feat = output[0, ACTION_LATENT_IDX_ROBOCASA]
            feat_vec = feat.float().mean(dim=(0, 1))
            self._current_feats.setdefault(k, {})[layer_idx] = feat_vec.detach().cpu().numpy()
        return hook


def get_action_with_capture(cfg, model, dataset_stats, obs, task_desc, capture, seed, n_steps):
    capture.reset_for_policy_call()
    orig = model.get_x0_fn_from_batch

    def patched(data_batch, guidance, **kwargs):
        result = orig(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            fn, extra = result
            def w(x, s):
                capture.before_denoise_step()
                return fn(x, s)
            return w, extra
        else:
            def w(x, s):
                capture.before_denoise_step()
                return result(x, s)
            return w

    model.get_x0_fn_from_batch = patched
    try:
        res = get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats, obs=obs,
            task_label_or_embedding=task_desc, seed=seed, randomize_seed=False,
            num_denoising_steps_action=n_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = orig
    return res


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(1 << 20)
            if not data:
                break
            h.update(data)
    return "sha256:" + h.hexdigest()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[7]
        ).decode().strip()
    except Exception:
        return "unknown"


# ── One (task, seed_series) rollout ──────────────────────────────────────────

def run_task_seed(cfg, model, dataset_stats, capture, task_name, seed_base, n_episodes,
                   global_episode_offset, seed_series_idx, task_idx):
    """1つの (task, seed系列) 組み合わせについて closed-loop rollout を実行し、
    call_records (capture.call_records に蓄積) と episode-level meta を返す。
    """
    cfg.task_name = task_name
    max_steps = TASK_MAX_STEPS.get(task_name, 500)

    ep_success = []
    seen_seeds_this_run = []

    for ep_idx in range(n_episodes):
        env, _ = create_robocasa_env(cfg, seed=seed_base + ep_idx, episode_idx=ep_idx)
        obs = env.reset()
        for _ in range(10):
            dummy = np.zeros(env.action_spec[0].shape)
            obs, _, _, _ = env.step(dummy)

        task_desc = env.get_ep_meta().get("lang", task_name)
        global_ep_id = global_episode_offset + ep_idx

        action_queue = deque()
        success = False
        call_idx_in_ep = 0
        t0 = time.time()

        for t in range(max_steps):
            observation = prepare_observation(obs, cfg.flip_images)

            if len(action_queue) == 0:
                # ── raw 物理量を potential-call 時点で捕捉 (潜在非依存, §4.3) ──
                gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)

                call_seed = cfg.seed + ep_idx + t
                seen_seeds_this_run.append(call_seed)

                try:
                    result = get_action_with_capture(
                        cfg, model, dataset_stats, observation, task_desc,
                        capture, call_seed, cfg.num_denoising_steps_action,
                    )
                except Exception as e:
                    log_message(f"  [{task_name} seed{seed_base}] Error at ep{ep_idx} t={t}: {e}")
                    import traceback; traceback.print_exc()
                    break

                meta = {
                    "task_name": task_name,
                    "task_idx": task_idx,
                    "seed_series_idx": seed_series_idx,
                    "seed_base": seed_base,
                    "episode": global_ep_id,
                    "episode_in_task_seed": ep_idx,
                    "call_idx": call_idx_in_ep,
                    "sim_step_t": t,
                    "call_seed": call_seed,
                    "gripper_qpos": gripper_qpos,
                    "eef_pos": eef_pos,
                    "eef_quat": eef_quat,
                }
                capture.finalize_policy_call(meta)

                actions = result["actions"]
                call_idx_in_ep += 1
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                    action_queue.append(a)

            if action_queue:
                action = action_queue.popleft()
                obs, _, _, _ = env.step(action)
                if env._check_success():
                    success = True
                    break

        env.close()
        ep_success.append(success)
        dt = time.time() - t0
        log_message(
            f"  [{task_name} seed{seed_base}] ep {ep_idx+1}/{n_episodes} "
            f"{'SUCCESS' if success else 'FAIL'} calls={call_idx_in_ep} steps={t+1} ({dt:.1f}s)"
        )

    # A2 sanity: within this run, consecutive calls must not share a seed (P1 규율)
    consec_dupes = sum(1 for i in range(1, len(seen_seeds_this_run))
                        if seen_seeds_this_run[i] == seen_seeds_this_run[i - 1])
    assert consec_dupes == 0, (
        f"A2 FAIL: {consec_dupes} consecutive-call seed duplicates in "
        f"{task_name}/seed{seed_base} — seed-per-call discipline violated (P1)."
    )

    return ep_success


def save_records_npz(call_records: List[Dict], probe_layers: List[int], out_path: Path,
                      ep_success: Optional[List[bool]] = None):
    npz_data = {}
    for k in range(NUM_DENOISE_STEPS):
        for layer_idx in probe_layers:
            vecs = []
            keep_idx = []
            for i, rec in enumerate(call_records):
                sf = rec["step_feats"]
                if k in sf and layer_idx in sf[k]:
                    vecs.append(sf[k][layer_idx])
                    keep_idx.append(i)
            if vecs:
                npz_data[f"feat_k{k}_layer{layer_idx}"] = np.stack(vecs)
                npz_data[f"feat_k{k}_layer{layer_idx}_idx"] = np.array(keep_idx)

    for key in ["task_name", "task_idx", "seed_series_idx", "seed_base", "episode",
                "episode_in_task_seed", "call_idx", "sim_step_t", "call_seed"]:
        npz_data[key] = np.array([r[key] for r in call_records])
    for key in ["gripper_qpos", "eef_pos", "eef_quat"]:
        npz_data[key] = np.stack([r[key] for r in call_records])

    if ep_success is not None:
        # per-call success flag: the outcome of the episode this call belongs to
        # (episode_in_task_seed indexes ep_success, which is 0-indexed per (task, seed) run).
        ep_success_arr = np.array(ep_success, dtype=bool)
        npz_data["episode_success"] = ep_success_arr
        npz_data["success"] = np.array(
            [ep_success_arr[r["episode_in_task_seed"]] for r in call_records], dtype=bool
        )

    np.savez(out_path, **npz_data)


@dataclass
class MultiTaskConfig(PolicyEvalConfig):
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/attractor_verification/collect"
    tasks: str = ",".join(DEFAULT_TASKS)
    seed_series: str = "195,196"
    n_episodes_per_task: int = 18


def main():
    import draccus
    cfg: MultiTaskConfig = draccus.parse(MultiTaskConfig)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    tasks = cfg.tasks.split(",")
    seed_series = [int(s) for s in cfg.seed_series.split(",")]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message("=== collect_multitask.py: マルチタスク・マルチシード収集 ===")
    log_message(f"Tasks: {tasks}")
    log_message(f"Seed series: {seed_series}")
    log_message(f"Episodes/task/seed: {cfg.n_episodes_per_task}")

    model, _ = get_model(cfg)
    model.eval()
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    num_blocks = len(model.net.blocks)
    actual_probe = [l for l in PROBE_LAYERS if l < num_blocks]
    capture = MultiTaskCapture(actual_probe)
    capture.register(model)

    manifest = {
        "run_id": f"attractor_verification_collect_{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit_hash(),
        "tasks": tasks,
        "seed_series": seed_series,
        "n_episodes_per_task": cfg.n_episodes_per_task,
        "probe_layers": actual_probe,
        "num_denoising_steps_action": cfg.num_denoising_steps_action,
        "chunk_size": cfg.chunk_size,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "scope_reduction_note": (
            "縮小スコープ (ユーザー承認): タスク4種(design原案6-8)、seed系列2(design原案3)、"
            "episode数18/task/seed(design原案≥25)。τ軸(Z_tau)捕捉は本収集ではスコープ外。"
        ),
        "files": {},
        "episode_success": {},
    }

    global_ep_offset = 0
    seed_success_details = []

    for seed_idx, seed_base in enumerate(seed_series):
        set_seed_everywhere(seed_base)
        for task_idx, task_name in enumerate(tasks):
            set_seed_everywhere(seed_base)
            cfg.seed = seed_base
            capture.call_records = []
            t_start = time.time()
            ep_success = run_task_seed(
                cfg, model, dataset_stats, capture, task_name, seed_base,
                cfg.n_episodes_per_task, global_ep_offset, seed_idx, task_idx,
            )
            global_ep_offset += cfg.n_episodes_per_task
            elapsed = time.time() - t_start

            fname = f"{task_name}_seed{seed_base}.npz"
            fpath = out_dir / fname
            save_records_npz(capture.call_records, actual_probe, fpath, ep_success=ep_success)
            sha = sha256_file(str(fpath))

            manifest["files"][fname] = {
                "task": task_name, "seed_base": seed_base,
                "n_episodes": cfg.n_episodes_per_task,
                "n_calls": len(capture.call_records),
                "success_rate": float(np.mean(ep_success)),
                "success_count": int(np.sum(ep_success)),
                "elapsed_sec": elapsed,
                "sha256": sha,
                "size_bytes": fpath.stat().st_size,
            }
            manifest["episode_success"][fname] = [bool(s) for s in ep_success]
            log_message(
                f"=== Done {task_name}/seed{seed_base}: "
                f"{len(capture.call_records)} calls, "
                f"success={np.mean(ep_success):.1%}, {elapsed:.1f}s ==="
            )

            # Persist manifest incrementally so partial progress survives crashes.
            with open(out_dir / "multitask_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    capture.remove()
    log_message("=== collect_multitask.py complete ===")
    log_message(f"Manifest: {out_dir / 'multitask_manifest.json'}")


if __name__ == "__main__":
    main()
