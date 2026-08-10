"""
collect_action_chunks.py — ldv_design_v2.md フェーズ7 対応、新規データ収集。

latent_dynamics_verification_report.md §2.6/§9で開示した通り、フェーズ1のD空間構築は
「行動が環境に及ぼした効果」の代理指標(eef_pos/gripper_qposのcall間差分)を使っており、
モデルが実際に生成した行動チャンク X_hat_0 (32 timesteps x 7 dims) 自体は未捕捉だった
(attractor_verification/collect・collect_v2のいずれのnpzにも `actions` は保存されていない)。

本スクリプトは attractor/collect_multitask.py を土台に、各policy callで
`get_action()` が返す生成済み行動チャンク `result["actions"]` を追加で捕捉する新規収集を行う。
それ以外(捕捉するBlk特徴・物理量・タスク集合・seed系列・エピソード数・P1のcall毎seed規約)は
collect_multitask.pyと完全に同一にしてある — 新規に収集する理由は「actionsフィールドを含む
npzが存在しないから」のみであり、それ以外の収集方法論を変える意図はない。

出力:
  results/latent_dynamics_verification/collect_actions/<task>_seed{S}.npz
    (collect_multitask.pyのnpzと同じフィールド + action_chunk (n_calls, chunk_size, 7))
  results/latent_dynamics_verification/collect_actions/multitask_manifest.json
"""

import hashlib
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
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
from cosmos_policy.experiments.robot.robocasa.analysis.verification.attractor.collect_multitask import (
    MultiTaskCapture, get_action_with_capture, sha256_file, git_commit_hash, DEFAULT_TASKS,
)

CHUNK_SIZE = 32


def run_task_seed(cfg, model, dataset_stats, capture, task_name, seed_base, n_episodes,
                   global_episode_offset, seed_series_idx, task_idx):
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

                actions = result["actions"]
                # X_hat_0: the model's own generated action chunk for THIS call, unpadded (7-dim
                # arm+gripper -- padding to env.action_dim==12 for a mobile base happens only
                # below, at env.step() time, and is not part of what the model generated).
                action_chunk = np.asarray(actions, dtype=np.float32)

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
                    "action_chunk": action_chunk,
                }
                capture.finalize_policy_call(meta)

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

    consec_dupes = sum(1 for i in range(1, len(seen_seeds_this_run))
                        if seen_seeds_this_run[i] == seen_seeds_this_run[i - 1])
    assert consec_dupes == 0, (
        f"A2 FAIL: {consec_dupes} consecutive-call seed duplicates in "
        f"{task_name}/seed{seed_base} -- seed-per-call discipline violated (P1)."
    )

    return ep_success


def save_records_npz(call_records: List[Dict], probe_layers: List[int], out_path: Path,
                      ep_success: Optional[List[bool]] = None):
    npz_data = {}
    for k in range(NUM_DENOISE_STEPS):
        for layer_idx in probe_layers:
            vecs, keep_idx = [], []
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
    # action_chunk shape can vary call-to-call only in principle (it never does in practice --
    # chunk_size is fixed by cfg.chunk_size=32 -- but pad defensively so np.stack never breaks a
    # long unattended collection run over a single malformed call).
    max_len = max(r["action_chunk"].shape[0] for r in call_records)
    action_dim = call_records[0]["action_chunk"].shape[1]
    chunks = np.zeros((len(call_records), max_len, action_dim), dtype=np.float32)
    for i, r in enumerate(call_records):
        ac = r["action_chunk"]
        chunks[i, :ac.shape[0]] = ac
    npz_data["action_chunk"] = chunks

    if ep_success is not None:
        ep_success_arr = np.array(ep_success, dtype=bool)
        npz_data["episode_success"] = ep_success_arr
        npz_data["success"] = np.array(
            [ep_success_arr[r["episode_in_task_seed"]] for r in call_records], dtype=bool
        )

    np.savez(out_path, **npz_data)


@dataclass
class ActionCollectConfig(PolicyEvalConfig):
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/collect_actions"
    tasks: str = ",".join(DEFAULT_TASKS)
    seed_series: str = "195,196"
    n_episodes_per_task: int = 20


def main():
    import draccus
    cfg: ActionCollectConfig = draccus.parse(ActionCollectConfig)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    tasks = cfg.tasks.split(",")
    seed_series = [int(s) for s in cfg.seed_series.split(",")]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_message("=== collect_action_chunks.py: フェーズ7用、action_chunk付きマルチタスク収集 ===")
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
        "run_id": f"ldv_phase7_collect_actions_{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit_hash(),
        "tasks": tasks,
        "seed_series": seed_series,
        "n_episodes_per_task": cfg.n_episodes_per_task,
        "probe_layers": actual_probe,
        "num_denoising_steps_action": cfg.num_denoising_steps_action,
        "chunk_size": cfg.chunk_size,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "note": (
            "attractor_verification/collect_multitask.pyと完全に同一のcollection methodology "
            "(タスク・seed系列・episode数・call毎seed規約)。唯一の違いはaction_chunk "
            "(生成された行動チャンク X_hat_0) を追加で保存すること -- ldv_design_v2.md フェーズ7 "
            "のD空間構築(真の生成行動を結合ベクトルに含める)のための新規収集。"
        ),
        "files": {},
        "episode_success": {},
    }

    global_ep_offset = 0

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

            with open(out_dir / "multitask_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    capture.remove()
    log_message("=== collect_action_chunks.py complete ===")
    log_message(f"Manifest: {out_dir / 'multitask_manifest.json'}")


if __name__ == "__main__":
    main()
