"""
T8b: gripper プローブの情報源テスト（review_verification_report_2.md §3 対応）

背景:
  T8 (t8_fixed_input_control.py) は「条件付け(画像/proprio)の差し替えが action-token の
  内部特徴量を変化させる」ことは示したが、レビューはこれだけでは
  「§3.G の gripper プローブ(96-97%)が条件付け経由の情報を読んでいる」ことの証拠には
  ならないと指摘した。理由: 条件付け全差し替えでも gripper 出力(action次元6)の変化は
  わずか0.0018（[-1,1]スケールで無視できる）に留まり、feature-Δ(最大0.20)との
  乖離が大きい。

  レビューが要求した本当のテスト:
    「gripper状態が異なる2つの入力の条件付けを入れ替えたとき、学習済みgripperプローブの
     予測が、差し替え先(source)のgripper状態に追随するか、元(base)のままか」
  追随すれば conditioning-driven、そうでなければ action-latent driven と判定できる。

方法:
  1. 通常のclosed-loopロールアウト(自身の行動で実際に環境を進める)をN episode行い、
     各 policy call で (a) 実際にモデルに入力された観測(proprio含む)、
     (b) proprio内の robot0_gripper_qpos の平均値(真のgripper開閉状態、モデル出力ではなく
     生の観測から直接読む)、(c) baseline (差し替えなし) の action-token 特徴量 (k=0, 全層)
     をキャッシュする。
  2. プール内の robot0_gripper_qpos 平均値の中央値で2値ラベル(open/closed)を作る。
  3. 各層について、LOEO (episode単位) で ridge プローブを学習し、
     「k=0の特徴量から『現在の』真のgripper状態を予測できるか」の精度を測る
     (§3.G の元プローブはeventual/action-basedラベルだったのに対し、本テストは
     現在の入力proprioそのものをラベルにする点が異なることに注意)。
  4. 真のgripper状態が異なる2 call (i, j) のペアを全プール内から抽出し、
     call i の観測の proprio (または全条件付け) を call j のもので差し替えて
     (call iのseedのまま) 再度モデルに通し、action-token特徴量(k=0)を取得する。
     このとき「call iを除いた残りのプール」で学習したプローブを使って予測し、
     予測が call i の真のラベル(base, 変えていない)に留まるか、call j の真のラベル
     (source, 差し替えた側)に追随するかを判定する（2値なので必ずどちらか一方）。

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.t8b_info_source_test \\
      --config ... --ckpt_path ... (run_t8b_info_source_test.sh 参照) \\
      --num_pool_episodes 15 --max_pairs 60
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import PROBE_LAYERS
from cosmos_policy.experiments.robot.robocasa.analysis.t8_fixed_input_control import (
    SingleCallCapture,
    get_action_with_capture,
    make_variant_obs,
)

SWAP_VARIANTS = ["proprio_swap", "img_swap", "all_swap"]


def ridge_fit_predict(X_tr, y_tr, X_te, n_classes=2, lam=1.0):
    """Class-weighted ridge regression classifier (same recipe as linear_probe.py/v2)."""
    mu = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-8
    X_tr_n = (X_tr - mu) / sig
    X_te_n = (X_te - mu) / sig
    N, D = X_tr_n.shape
    X_tr_b = np.hstack([X_tr_n, np.ones((N, 1))])
    X_te_b = np.hstack([X_te_n, np.ones((X_te_n.shape[0], 1))])
    counts = np.bincount(y_tr.astype(int), minlength=n_classes).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    w_samp = 1.0 / counts[y_tr.astype(int)]
    w_samp /= w_samp.mean()
    Y_oh = np.zeros((N, n_classes))
    Y_oh[np.arange(N), y_tr.astype(int)] = 1.0
    WD = np.diag(w_samp)
    A = X_tr_b.T @ WD @ X_tr_b + lam * np.eye(D + 1)
    b_rhs = X_tr_b.T @ WD @ Y_oh
    try:
        W = np.linalg.solve(A, b_rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, b_rhs, rcond=None)[0]
    logits = X_te_b @ W
    return logits.argmax(axis=1)


@dataclass
class T8bConfig(PolicyEvalConfig):
    num_pool_episodes: int = 15
    max_pairs: int = 60
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/t8b_info_source"


def gather_pool(cfg, model, dataset_stats, env, capture: SingleCallCapture):
    """Closed-loop rollout across num_pool_episodes; cache per-call obs, true gripper state,
    baseline (unswapped) k=0 action-token features."""
    from collections import deque

    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    pool: List[Dict] = []
    call_seed_counter = 0

    for ep in range(cfg.num_pool_episodes):
        obs_raw = env.reset()
        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        step_count = 0
        done = False
        q = deque()
        while not done and step_count < max_steps:
            if len(q) == 0:
                obs = prepare_observation(obs_raw, cfg.flip_images)
                true_gripper_qpos_mean = float(np.mean(obs["proprio"][:2]))
                call_seed = cfg.seed * 1000 + call_seed_counter
                call_seed_counter += 1
                set_seed_everywhere(call_seed)
                result = get_action_with_capture(
                    cfg=cfg, model=model, dataset_stats=dataset_stats,
                    observation=obs, task_description=task_description,
                    capture=capture, seed=call_seed,
                    num_denoising_steps=cfg.num_denoising_steps_action,
                )
                feats_k0 = dict(capture.feats.get(0, {}))
                actions = result["actions"]
                action_gripper_mean = float(np.mean(np.array(actions)[:, 6]))
                pool.append({
                    "episode": ep,
                    "obs": obs,
                    "task_description": task_description,
                    "call_seed": call_seed,
                    "true_gripper_qpos_mean": true_gripper_qpos_mean,
                    "action_gripper_mean": action_gripper_mean,
                    "feat_k0": {l: v for l, v in feats_k0.items()},
                })
                for i in range(min(cfg.num_open_loop_steps, len(actions))):
                    a = actions[i]
                    if a.shape[-1] == 7 and env.action_dim == 12:
                        a = np.concatenate([a, np.array([0., 0., 0., 0., -1.])])
                    q.append(a)
            action = q.popleft()
            obs_raw, r, done, info = env.step(action)
            step_count += 1
            if env._check_success():
                done = True
        log_message(f"pool episode {ep} done, total calls so far {len(pool)}")
    return pool


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="cosmos_policy/config/config.py")
    parser.add_argument("--use_wrist_image", type=lambda x: x == "True", default=True)
    parser.add_argument("--num_wrist_images", type=int, default=1)
    parser.add_argument("--use_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--normalize_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--unnormalize_actions", type=lambda x: x == "True", default=True)
    parser.add_argument("--dataset_stats_path", type=str, default="")
    parser.add_argument("--t5_text_embeddings_path", type=str, default="")
    parser.add_argument("--trained_with_image_aug", type=lambda x: x == "True", default=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--num_open_loop_steps", type=int, default=16)
    parser.add_argument("--task_name", type=str, default="PnPCounterToCab")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--randomize_seed", type=lambda x: x == "True", default=False)
    parser.add_argument("--deterministic", type=lambda x: x == "True", default=True)
    parser.add_argument("--use_variance_scale", type=lambda x: x == "True", default=False)
    parser.add_argument("--use_jpeg_compression", type=lambda x: x == "True", default=True)
    parser.add_argument("--flip_images", type=lambda x: x == "True", default=True)
    parser.add_argument("--num_denoising_steps_action", type=int, default=5)
    parser.add_argument("--num_denoising_steps_future_state", type=int, default=1)
    parser.add_argument("--num_denoising_steps_value", type=int, default=1)
    parser.add_argument("--data_collection", type=lambda x: x == "True", default=False)
    parser.add_argument("--num_pool_episodes", type=int, default=15)
    parser.add_argument("--max_pairs", type=int, default=60)
    parser.add_argument("--label_source", type=str, default="current_proprio",
                        choices=["current_proprio", "action_gripper"],
                        help="current_proprio: ground-truth robot0_gripper_qpos at input time. "
                             "action_gripper: eventual predicted action gripper dim (k=final), "
                             "matching the original gripper_2 probe's label definition.")
    parser.add_argument("--output_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/t8b_info_source")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = T8bConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        use_wrist_image=args.use_wrist_image, num_wrist_images=args.num_wrist_images,
        use_proprio=args.use_proprio, normalize_proprio=args.normalize_proprio,
        unnormalize_actions=args.unnormalize_actions, dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        trained_with_image_aug=args.trained_with_image_aug, chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps, task_name=args.task_name, seed=args.seed,
        randomize_seed=args.randomize_seed, deterministic=args.deterministic,
        use_variance_scale=args.use_variance_scale, use_jpeg_compression=args.use_jpeg_compression,
        flip_images=args.flip_images, num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=args.num_denoising_steps_value, data_collection=args.data_collection,
        num_pool_episodes=args.num_pool_episodes, max_pairs=args.max_pairs, output_dir=args.output_dir,
    )

    log_message(f"Creating env for task: {cfg.task_name}")
    env, _ = create_robocasa_env(cfg)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cu.DEVICE = device

    log_message("Loading model...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=0)
    model, _ = get_model(cfg)
    model.eval()

    capture = SingleCallCapture(PROBE_LAYERS)
    capture.register(model)

    set_seed_everywhere(cfg.seed)

    log_message("=== Phase 1: gathering rollout pool ===")
    pool = gather_pool(cfg, model, dataset_stats, env, capture)
    log_message(f"Pool size: {len(pool)} calls across {cfg.num_pool_episodes} episodes")

    label_key = "action_gripper_mean" if args.label_source == "action_gripper" else "true_gripper_qpos_mean"
    gripper_vals = np.array([p[label_key] for p in pool])
    median_thr = float(np.median(gripper_vals))
    true_labels = (gripper_vals > median_thr).astype(int)  # 0=more-closed, 1=more-open (relative)
    for i, p in enumerate(pool):
        p["true_label"] = int(true_labels[i])
    ep_of = np.array([p["episode"] for p in pool])

    log_message(f"label_source={args.label_source} ({label_key}): min={gripper_vals.min():.4f} "
                f"max={gripper_vals.max():.4f} median={median_thr:.4f} label_balance={np.bincount(true_labels)}")

    # ── Phase 2: LOEO probe accuracy on CURRENT true gripper label from k=0 baseline features ──
    baseline_probe_acc = {}
    episodes = np.unique(ep_of)
    for layer in PROBE_LAYERS:
        X = np.stack([p["feat_k0"].get(layer, np.zeros(2048, dtype=np.float32)) for p in pool])
        accs = []
        for test_ep in episodes:
            test_mask = ep_of == test_ep
            train_mask = ~test_mask
            if train_mask.sum() < 2 or test_mask.sum() < 1:
                continue
            y_tr, y_te = true_labels[train_mask], true_labels[test_mask]
            if len(np.unique(y_tr)) < 2:
                continue
            y_pred = ridge_fit_predict(X[train_mask], y_tr, X[test_mask])
            accs.append(float((y_pred == y_te).mean()))
        baseline_probe_acc[f"Blk-{layer}"] = float(np.mean(accs)) if accs else None
        log_message(f"Blk-{layer:2d} LOEO acc (label_source={args.label_source}, k=0): "
                    f"{baseline_probe_acc[f'Blk-{layer}']}")

    # ── Phase 3: build discriminating pairs (true_label(i) != true_label(j), different episodes) ──
    rng = np.random.default_rng(0)
    idx_by_label = {0: np.where(true_labels == 0)[0], 1: np.where(true_labels == 1)[0]}
    candidate_pairs = []
    for i in idx_by_label[0]:
        for j in idx_by_label[1]:
            if ep_of[i] != ep_of[j]:
                candidate_pairs.append((i, j))
    for i in idx_by_label[1]:
        for j in idx_by_label[0]:
            if ep_of[i] != ep_of[j]:
                candidate_pairs.append((i, j))
    rng.shuffle(candidate_pairs)
    pairs = candidate_pairs[:cfg.max_pairs]
    log_message(f"Discriminating candidate pairs: {len(candidate_pairs)}, using {len(pairs)}")

    # ── Phase 4: for each pair & swap variant, re-run inference with i's conditioning replaced
    #    by j's, capture k=0 features, and classify with a probe trained excluding both episodes ──
    results_by_variant = {v: {f"Blk-{l}": {"n": 0, "follows_source": 0, "stays_base": 0}
                               for l in PROBE_LAYERS} for v in SWAP_VARIANTS}
    baseline_selfcheck = {f"Blk-{l}": {"n": 0, "correct": 0} for l in PROBE_LAYERS}

    X_all = {layer: np.stack([p["feat_k0"].get(layer, np.zeros(2048, dtype=np.float32)) for p in pool])
             for layer in PROBE_LAYERS}

    for pair_idx, (i, j) in enumerate(pairs):
        base = pool[i]
        source = pool[j]
        exclude_eps = {base["episode"], source["episode"]}
        train_mask = np.array([ep not in exclude_eps for ep in ep_of])
        if train_mask.sum() < 4 or len(np.unique(true_labels[train_mask])) < 2:
            continue

        for variant in SWAP_VARIANTS:
            obs_variant = make_variant_obs(base["obs"], source["obs"], variant)
            set_seed_everywhere(base["call_seed"])
            get_action_with_capture(
                cfg=cfg, model=model, dataset_stats=dataset_stats,
                observation=obs_variant, task_description=base["task_description"],
                capture=capture, seed=base["call_seed"],
                num_denoising_steps=cfg.num_denoising_steps_action,
            )
            feats_swap = dict(capture.feats.get(0, {}))

            for layer in PROBE_LAYERS:
                if layer not in feats_swap:
                    continue
                X_tr = X_all[layer][train_mask]
                y_tr = true_labels[train_mask]
                x_te = feats_swap[layer][None, :]
                y_pred = int(ridge_fit_predict(X_tr, y_tr, x_te)[0])
                r = results_by_variant[variant][f"Blk-{layer}"]
                r["n"] += 1
                if y_pred == source["true_label"]:
                    r["follows_source"] += 1
                elif y_pred == base["true_label"]:
                    r["stays_base"] += 1

                if variant == SWAP_VARIANTS[0]:
                    # baseline self-check (probe applied to base's OWN unswapped k=0 feature,
                    # same excluded-episode training set) -- sanity floor
                    x_base = base["feat_k0"][layer][None, :]
                    y_pred_base = int(ridge_fit_predict(X_tr, y_tr, x_base)[0])
                    bc = baseline_selfcheck[f"Blk-{layer}"]
                    bc["n"] += 1
                    bc["correct"] += int(y_pred_base == base["true_label"])
        if (pair_idx + 1) % 10 == 0:
            log_message(f"  processed {pair_idx+1}/{len(pairs)} pairs")

    summary = {}
    for variant in SWAP_VARIANTS:
        summary[variant] = {}
        for layer in PROBE_LAYERS:
            r = results_by_variant[variant][f"Blk-{layer}"]
            n = r["n"]
            summary[variant][f"Blk-{layer}"] = {
                "n_pairs": n,
                "frac_follows_source": r["follows_source"] / n if n else None,
                "frac_stays_base": r["stays_base"] / n if n else None,
            }
    baseline_selfcheck_summary = {
        k: (v["correct"] / v["n"] if v["n"] else None) for k, v in baseline_selfcheck.items()
    }

    out = {
        "test": "T8b_information_source_test",
        "label_source": args.label_source,
        "task_name": cfg.task_name,
        "num_pool_episodes": cfg.num_pool_episodes,
        "pool_size_calls": len(pool),
        "num_denoising_steps": cfg.num_denoising_steps_action,
        "label_median_threshold": median_thr,
        "label_balance": {"0": int((true_labels == 0).sum()), "1": int((true_labels == 1).sum())},
        "probe_layers": PROBE_LAYERS,
        f"baseline_loeo_probe_acc_{args.label_source}_k0": baseline_probe_acc,
        "baseline_selfcheck_acc_excluding_both_episodes": baseline_selfcheck_summary,
        "n_discriminating_pairs_used": len(pairs),
        "swap_variant_results": summary,
        "interpretation_note": (
            "For each discriminating pair (true label of i != true label of j, different episodes), "
            "we swap i's conditioning for j's, keep i's noise seed fixed, and ask a probe trained "
            "WITHOUT either episode's data whether its prediction on the swapped features matches "
            "j (frac_follows_source -> conditioning-driven) or i (frac_stays_base -> action-latent-driven). "
            "Since this is a binary label, follows_source + stays_base should sum to ~1 (up to n)."
        ),
    }
    out_name = f"t8b_results_{args.label_source}.json"
    with open(output_dir / out_name, "w") as f:
        json.dump(out, f, indent=2)
    log_message(f"Saved: {output_dir / out_name}")
    log_message("T8b complete.")


if __name__ == "__main__":
    main()
