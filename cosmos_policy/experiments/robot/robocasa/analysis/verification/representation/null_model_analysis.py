"""
§5 ヌルモデル — ランダム初期化 DiT での特徴捕捉

目的:
  学習済みモデルで観測された特徴構造 (PR低下, CKA パターン) が
  「学習で獲得した構造」なのか「アーキテクチャの帰結」なのかを切り分ける。

  同一 DiT アーキテクチャのパラメータをすべてランダム初期化し、
  同じタスク + 同一エピソードを 50 エピソード実行してアクショントークン特徴を捕捉。

比較:
  trained:  results/action_features/features.npz
  null:     results/null_model_random_init/features.npz

出力: results/null_model_random_init/
  - features.npz          (trained と同形式)
  - null_vs_trained_pr.json
  - null_vs_trained_pr.png
  - null_vs_trained_cka.png
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

import cosmos_policy.experiments.robot.cosmos_utils as _cosmos_utils
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
from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    CHUNK_SIZE,
    ACTION_LATENT_IDX_ROBOCASA,
    PROBE_LAYERS,
    PROBE_LAYER_LABELS,
    NUM_DENOISE_STEPS,
)
from cosmos_policy.experiments.robot.robocasa.analysis.verification.collection.feature_analysis import (
    FeatureCapture,
    get_action_with_features,
    save_features_npz,
    _collect_layer_features,
    linear_cka,
)


RNG_SEED = 195
N_BOOTSTRAP = 1000


def random_init_model(model, seed: int = 42):
    """Replace all learned parameters with random Gaussian initialization.

    LayerNorm weights (gamma) are set to 1.0 so LN remains functional.
    All other 1D params (biases, LN beta) are zeroed.
    2D+ params (weight matrices) get N(0, 0.02²).
    """
    torch.manual_seed(seed)
    n_params = 0
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p.data, mean=0.0, std=0.02)
        elif "weight" in name and "norm" in name.lower():
            # LayerNorm / GroupNorm gamma → ones so normalization stays functional
            torch.nn.init.ones_(p.data)
        else:
            torch.nn.init.zeros_(p.data)
        n_params += p.numel()
    log_message(f"  Random init: {n_params:,} parameters reset (seed={seed})")
    return model


def compute_pr_gram(X: np.ndarray) -> float:
    """PR via Gram matrix eigvalsh (N×N, fast when N < D)."""
    Xc = X - X.mean(axis=0)
    if Xc.shape[0] < 2:
        return 1.0
    if Xc.shape[0] <= 500:
        G = Xc @ Xc.T
        ev = np.linalg.eigvalsh(G)
        ev = ev[ev > 1e-10]
    else:
        from sklearn.utils.extmath import randomized_svd
        _, s, _ = randomized_svd(Xc, n_components=min(100, Xc.shape[0] - 1), random_state=42)
        ev = s ** 2
    if len(ev) == 0:
        return 1.0
    ev = ev / ev.sum()
    return float(1.0 / (ev ** 2).sum())


def compute_pr_from_npz(npz_path: Path) -> Dict:
    """Load features.npz and compute PR per (k, layer)."""
    d = np.load(npz_path)
    ep_labels = d["episode_labels"]
    results = {}
    for k in range(NUM_DENOISE_STEPS):
        for l in PROBE_LAYERS:
            key = f"feat_k{k}_layer{l}"
            if key not in d:
                continue
            F = d[key]  # (N, D)
            global_pr = compute_pr_gram(F)

            # Per-episode PR
            eps = np.unique(ep_labels)
            ep_prs = [compute_pr_gram(F[ep_labels == ep]) for ep in eps
                      if (ep_labels == ep).sum() >= 3]
            ep_prs = np.array(ep_prs)

            boot = [np.random.choice(ep_prs, len(ep_prs), replace=True).mean()
                    for _ in range(N_BOOTSTRAP)]
            results[(k, l)] = {
                "global_pr": global_pr,
                "ep_pr_mean": float(ep_prs.mean()),
                "ci_lo": float(np.percentile(boot, 2.5)),
                "ci_hi": float(np.percentile(boot, 97.5)),
            }
    return results


def compare_pr(trained_pr: Dict, null_pr: Dict) -> Dict:
    comparison = {}
    for key in trained_pr:
        if key not in null_pr:
            continue
        k, l = key
        comparison[f"k{k}_layer{l}"] = {
            "k": k, "layer": l,
            "trained_pr": trained_pr[key]["global_pr"],
            "null_pr": null_pr[key]["global_pr"],
            "trained_pr_ci": [trained_pr[key]["ci_lo"], trained_pr[key]["ci_hi"]],
            "null_pr_ci": [null_pr[key]["ci_lo"], null_pr[key]["ci_hi"]],
            "null_higher": null_pr[key]["global_pr"] > trained_pr[key]["global_pr"],
        }
    return comparison


def compute_cka_matrix(npz_path: Path, k: int) -> Optional[np.ndarray]:
    """CKA matrix for a specific k step."""
    d = np.load(npz_path)
    feats = []
    valid_layers = []
    for l in PROBE_LAYERS:
        key = f"feat_k{k}_layer{l}"
        if key in d:
            feats.append(d[key])
            valid_layers.append(l)
    n = len(feats)
    if n < 2:
        return None, valid_layers
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = linear_cka(feats[i], feats[j])
    return mat, valid_layers


def plot_comparison(trained_npz: Path, null_npz: Path, pr_comparison: Dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PR comparison plot
    k_vals = sorted(set(v["k"] for v in pr_comparison.values()))
    n_k = len(k_vals)
    n_layers = len(PROBE_LAYERS)

    fig, axes = plt.subplots(1, n_k, figsize=(4 * n_k, 5), sharey=True)
    if n_k == 1:
        axes = [axes]

    for ki, k in enumerate(k_vals):
        ax = axes[ki]
        layer_indices = sorted(set(v["layer"] for v in pr_comparison.values() if v["k"] == k))
        layers_x = list(range(len(layer_indices)))
        trained_prs = [pr_comparison[f"k{k}_layer{l}"]["trained_pr"] for l in layer_indices]
        null_prs = [pr_comparison[f"k{k}_layer{l}"]["null_pr"] for l in layer_indices]

        ax.plot(layers_x, trained_prs, 'o-', color='steelblue', label='Trained', linewidth=2)
        ax.plot(layers_x, null_prs, 's--', color='tomato', label='Random init', linewidth=2)
        xlabels = [PROBE_LAYER_LABELS.get(l, f"L{l}").replace("\n", " ") for l in layer_indices]
        ax.set_xticks(layers_x)
        ax.set_xticklabels(xlabels, rotation=30, fontsize=7)
        ax.set_title(f"k={k}")
        ax.set_xlabel("Layer")
        if ki == 0:
            ax.set_ylabel("Participation Ratio (PR)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("§5 Null Model: PR Comparison\nTrained vs Random Init DiT",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "null_vs_trained_pr.png", dpi=150)
    plt.close()
    log_message(f"Saved: {out_dir}/null_vs_trained_pr.png")

    # CKA comparison (k=4, side by side)
    for k in [0, 4]:
        mat_t, layers_t = compute_cka_matrix(trained_npz, k)
        mat_n, layers_n = compute_cka_matrix(null_npz, k)
        if mat_t is None or mat_n is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        xlabels = [PROBE_LAYER_LABELS.get(l, f"L{l}").replace("\n", " ") for l in layers_t]

        for ax, mat, title in zip(axes, [mat_t, mat_n], ["Trained", "Random Init"]):
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(xlabels)))
            ax.set_yticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=30, fontsize=7)
            ax.set_yticklabels(xlabels, fontsize=7)
            ax.set_title(f"{title} (k={k})")
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"§5 Null Model: Linear CKA  k={k}", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / f"null_vs_trained_cka_k{k}.png", dpi=150)
        plt.close()
        log_message(f"Saved: {out_dir}/null_vs_trained_cka_k{k}.png")


@dataclass
class NullModelConfig(PolicyEvalConfig):
    num_episodes: int = 50
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/null_model_random_init"
    trained_features_path: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz"
    random_init_seed: int = 42


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--config_file", default="cosmos_policy/config/config.py")
    parser.add_argument("--use_wrist_image", type=lambda x: x == "True", default=True)
    parser.add_argument("--num_wrist_images", type=int, default=1)
    parser.add_argument("--use_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--normalize_proprio", type=lambda x: x == "True", default=True)
    parser.add_argument("--unnormalize_actions", type=lambda x: x == "True", default=True)
    parser.add_argument("--dataset_stats_path", default="")
    parser.add_argument("--t5_text_embeddings_path", default="")
    parser.add_argument("--trained_with_image_aug", type=lambda x: x == "True", default=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--num_open_loop_steps", type=int, default=16)
    parser.add_argument("--task_name", default="PnPCounterToCab")
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
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--output_dir", default="cosmos_policy/experiments/robot/robocasa/analysis/results/null_model_random_init")
    parser.add_argument("--trained_features_path",
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/action_features/features.npz")
    parser.add_argument("--random_init_seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = NullModelConfig(
        config=args.config,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        use_wrist_image=args.use_wrist_image,
        num_wrist_images=args.num_wrist_images,
        use_proprio=args.use_proprio,
        normalize_proprio=args.normalize_proprio,
        unnormalize_actions=args.unnormalize_actions,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        trained_with_image_aug=args.trained_with_image_aug,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        task_name=args.task_name,
        seed=args.seed,
        randomize_seed=args.randomize_seed,
        deterministic=args.deterministic,
        use_variance_scale=args.use_variance_scale,
        use_jpeg_compression=args.use_jpeg_compression,
        flip_images=args.flip_images,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=args.num_denoising_steps_value,
        data_collection=args.data_collection,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        trained_features_path=args.trained_features_path,
        random_init_seed=args.random_init_seed,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    _cosmos_utils.DEVICE = device
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cu.DEVICE = device

    log_message("=== §5 Null Model: Random Init DiT ===")

    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path, worker_id=0)
    model, _ = get_model(cfg)

    log_message("Reinitializing model weights randomly...")
    model = random_init_model(model, seed=cfg.random_init_seed)
    model.eval()

    num_blocks = len(model.net.blocks)
    actual_probe = [l for l in PROBE_LAYERS if l < num_blocks]
    log_message(f"Probe layers: {actual_probe}")

    env, _ = create_robocasa_env(cfg)
    set_seed_everywhere(cfg.seed)

    capture = FeatureCapture(actual_probe)
    capture.register(model)
    log_message("Forward hooks registered.")

    call_records_all = []
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)

    for ep in range(cfg.num_episodes):
        log_message(f"\nEpisode {ep+1}/{cfg.num_episodes}")
        obs = env.reset()
        task_desc = env.get_ep_meta().get("lang", cfg.task_name)
        action_queue = deque()
        step_count = 0
        done = False
        call_idx = 0

        while not done and step_count < max_steps:
            if len(action_queue) == 0:
                observation = prepare_observation(obs, cfg.flip_images)
                try:
                    result = get_action_with_features(
                        cfg=cfg,
                        model=model,
                        dataset_stats=dataset_stats,
                        observation=observation,
                        task_description=task_desc,
                        capture=capture,
                        seed=cfg.seed + ep + step_count,
                        num_denoising_steps=cfg.num_denoising_steps_action,
                    )
                    capture.finalize_policy_call(episode_idx=ep, call_idx=call_idx)
                    call_records_all.append(capture.call_records[-1])
                    call_idx += 1

                    actions = result["actions"]
                    for i in range(min(cfg.num_open_loop_steps, len(actions))):
                        a = actions[i]
                        if a.shape[-1] == 7 and env.action_dim == 12:
                            a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
                        action_queue.append(a)
                except Exception as e:
                    log_message(f"  Error at step {step_count}: {e}")
                    import traceback; traceback.print_exc()
                    done = True
                    break

            action = action_queue.popleft()
            obs, reward, done, info = env.step(action)
            step_count += 1
            # Random init model won't succeed, but check anyway
            if env._check_success():
                done = True

        log_message(f"  Episode {ep+1}: steps={step_count}, calls={call_idx}")

    capture.remove()
    log_message(f"\nTotal policy calls captured: {len(call_records_all)}")

    null_npz = out_dir / "features.npz"
    save_features_npz(call_records_all, actual_probe, out_dir)

    # PR comparison
    trained_npz = Path(args.trained_features_path)
    if trained_npz.exists():
        log_message("\nComputing PR comparison: trained vs null...")
        np.random.seed(RNG_SEED)
        trained_pr = compute_pr_from_npz(trained_npz)
        null_pr = compute_pr_from_npz(null_npz)
        pr_comparison = compare_pr(trained_pr, null_pr)

        with open(out_dir / "null_vs_trained_pr.json", "w") as f:
            json.dump({
                "section": "5_null_model_random_init",
                "comparison": pr_comparison,
                "note": (
                    "PR(null) > PR(trained) indicates trained model learns a more structured "
                    "(lower-dimensional) representation than random init. "
                    "PR(null) ~ d (max) for Gaussian random features."
                ),
            }, f, indent=2)
        log_message(f"Saved: {out_dir}/null_vs_trained_pr.json")

        log_message("\n§5 PR summary (k=4, last denoise step):")
        log_message(f"  {'Layer':>7} | {'Trained PR':>12} | {'Null PR':>10} | {'Null>Trained':>12}")
        for l in actual_probe:
            key = f"k4_layer{l}"
            if key in pr_comparison:
                v = pr_comparison[key]
                log_message(f"  Layer {l:>2}  | {v['trained_pr']:>12.1f} | {v['null_pr']:>10.1f} | {str(v['null_higher']):>12}")

        plot_comparison(trained_npz, null_npz, pr_comparison, out_dir)
    else:
        log_message(f"WARNING: trained_features_path not found: {trained_npz}")
        log_message("Skipping comparison plots.")

    log_message("\n§5 null model analysis complete.")


if __name__ == "__main__":
    main()
