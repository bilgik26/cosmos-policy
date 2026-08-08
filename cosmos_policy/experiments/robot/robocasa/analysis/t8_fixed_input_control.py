"""
T8: 固定入力コントロール（設計書 revised_verification_design.md §6 T8 / レビュー §3.1）

目的:
  §3.G の線形プローブで gripper_2_phys が「Blk-0 k=0 では chance 水準(60%)だが
  Blk-18/22/27 は k=0（σ=80、行動トークンはほぼ純ノイズ）でも既に 96-97%」という
  結果が出ている（probe_acc_ci.json で確認済み）。これは「デノイジングによる
  行動デコード」ではなく「固定の観測入力（条件付け: 画像 or proprio）からの
  漏洩」で説明できてしまう可能性がある。

  T8 はこれを検証する唯一の統制テスト:
    同一の action-token 特徴量抽出パイプラインで、観測入力の一部だけを
    別エピソードのものに差し替えたとき、action-token 特徴量 (層 x step) が
    有意に変化する (Δ ≠ 0) ことを確認する。
    - 変化が「ある」ことは、その入力チャネルが表現に因果的な影響を持つことの証拠。
    - 特に k=0（ほぼ純ノイズ入力）で深層の特徴量が画像/proprioの差し替えに
      敏感に反応するなら、k=0 での高精度プローブは「条件付けの通過」である
      可能性が高いことを支持する。

  4 バリアント（同一 obs_base・同一 seed・同一 task_description で比較）:
    - repeat:       obs_base をそのまま2回 (数値の床/決定性チェック; Δ≈0 のはず)
    - img_swap:     primary_image のみ別エピソードのものに差し替え
    - proprio_swap: proprio のみ別エピソードのものに差し替え
    - all_swap:     primary_image + secondary_image + wrist_image + proprio を全差し替え

実行例:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.t8_fixed_input_control \\
      --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \\
      --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \\
      --config_file cosmos_policy/config/config.py \\
      ... (run_t8_fixed_input_control.sh を参照) \\
      --num_pairs 30
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import PROBE_LAYERS


SWAP_VARIANTS = ["img_swap", "proprio_swap", "all_swap"]
ALL_ROWS = ["repeat"] + SWAP_VARIANTS  # 'repeat' = floor from two independent baseline calls


class SingleCallCapture:
    """1回の policy call 分だけ (k, layer) -> action-token 特徴ベクトルを保持するフック集合。"""

    def __init__(self, probe_layers: List[int]):
        self.probe_layers = probe_layers
        self._current_step = -1
        self.feats: Dict[int, Dict[int, np.ndarray]] = {}
        self._handles = []

    def register(self, model):
        net = model.net
        for layer_idx in self.probe_layers:
            h = net.blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset_for_policy_call(self):
        self._current_step = -1
        self.feats = {}

    def before_denoise_step(self):
        self._current_step += 1
        self.feats.setdefault(self._current_step, {})

    def _make_hook(self, layer_idx: int):
        ACTION_T_IDX = 5

        def hook(module, inp, output):
            if self._current_step < 0:
                return
            if not (isinstance(output, torch.Tensor) and output.dim() == 5):
                return
            B, T, H, W, D = output.shape
            if T <= ACTION_T_IDX:
                return
            feat = output[0, ACTION_T_IDX].float().mean(dim=(0, 1))  # (D,)
            self.feats.setdefault(self._current_step, {})[layer_idx] = feat.detach().cpu().numpy()

        return hook


def get_action_with_capture(cfg, model, dataset_stats, observation, task_description,
                             capture: SingleCallCapture, seed: int, num_denoising_steps: int):
    capture.reset_for_policy_call()
    original_get_x0_fn = model.get_x0_fn_from_batch

    def patched_get_x0_fn(data_batch, guidance, **kwargs):
        result = original_get_x0_fn(data_batch, guidance, **kwargs)
        if isinstance(result, tuple):
            x0_fn_raw, extra = result

            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                return x0_fn_raw(x_t, sigma)

            return wrapped, extra
        else:
            x0_fn_raw = result

            def wrapped(x_t, sigma):
                capture.before_denoise_step()
                return x0_fn_raw(x_t, sigma)

            return wrapped

    model.get_x0_fn_from_batch = patched_get_x0_fn
    try:
        result = get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats, obs=observation,
            task_label_or_embedding=task_description, seed=seed, randomize_seed=False,
            num_denoising_steps_action=num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        model.get_x0_fn_from_batch = original_get_x0_fn
    return result


def make_variant_obs(obs_base: dict, obs_source: dict, variant: str) -> dict:
    obs = dict(obs_base)
    if variant == "repeat":
        pass
    elif variant == "img_swap":
        obs["primary_image"] = obs_source["primary_image"]
    elif variant == "proprio_swap":
        obs["proprio"] = obs_source["proprio"]
    elif variant == "all_swap":
        obs["primary_image"] = obs_source["primary_image"]
        obs["secondary_image"] = obs_source["secondary_image"]
        obs["wrist_image"] = obs_source["wrist_image"]
        obs["proprio"] = obs_source["proprio"]
    else:
        raise ValueError(variant)
    return obs


@dataclass
class T8Config(PolicyEvalConfig):
    num_pairs: int = 30
    output_dir: str = "cosmos_policy/experiments/robot/robocasa/analysis/results/t8_fixed_input"


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
    parser.add_argument("--num_pairs", type=int, default=30)
    parser.add_argument("--output_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/t8_fixed_input")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = T8Config(
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
        num_pairs=args.num_pairs, output_dir=args.output_dir,
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
    model, cosmos_config = get_model(cfg)
    model.eval()

    capture = SingleCallCapture(PROBE_LAYERS)
    capture.register(model)

    set_seed_everywhere(cfg.seed)

    # records[row][pair_idx][k][layer] = feat vector; also action outputs
    # 'baseline1'/'baseline2' = two independent calls on the *same* obs_base (determinism floor)
    records: Dict[str, List[Dict]] = {v: [] for v in (["baseline1", "baseline2"] + SWAP_VARIANTS)}
    action_records: Dict[str, List[np.ndarray]] = {v: [] for v in (["baseline1", "baseline2"] + SWAP_VARIANTS)}

    n_steps = cfg.num_denoising_steps_action

    for pair_idx in range(cfg.num_pairs):
        log_message(f"\nPair {pair_idx+1}/{cfg.num_pairs}")
        obs_raw_base = env.reset()
        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        obs_base = prepare_observation(obs_raw_base, cfg.flip_images)

        obs_raw_source = env.reset()
        obs_source = prepare_observation(obs_raw_source, cfg.flip_images)

        call_seed = cfg.seed + pair_idx  # vary noise across pairs, fixed across rows within a pair
        rows_to_run = {
            "baseline1": obs_base,
            "baseline2": obs_base,
            "img_swap": make_variant_obs(obs_base, obs_source, "img_swap"),
            "proprio_swap": make_variant_obs(obs_base, obs_source, "proprio_swap"),
            "all_swap": make_variant_obs(obs_base, obs_source, "all_swap"),
        }
        for row, obs_variant in rows_to_run.items():
            set_seed_everywhere(call_seed)
            result = get_action_with_capture(
                cfg=cfg, model=model, dataset_stats=dataset_stats,
                observation=obs_variant, task_description=task_description,
                capture=capture, seed=call_seed, num_denoising_steps=n_steps,
            )
            feats_copy = {k: dict(v) for k, v in capture.feats.items()}
            records[row].append(feats_copy)
            action_records[row].append(np.array(result["actions"]))  # (T, 7)

    # ── Aggregate: Δfeat relative to 'repeat' baseline (same obs, same seed twice) ──
    def rel_delta(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))

    stats = {}
    for row in ALL_ROWS:
        compare_row = "baseline2" if row == "repeat" else row
        stats[row] = {}
        for k in range(n_steps):
            stats[row][k] = {}
            for layer in PROBE_LAYERS:
                deltas = []
                for pair_idx in range(cfg.num_pairs):
                    orig_feats = records["baseline1"][pair_idx]
                    var_feats = records[compare_row][pair_idx]
                    if k in orig_feats and layer in orig_feats[k] and k in var_feats and layer in var_feats[k]:
                        deltas.append(rel_delta(var_feats[k][layer], orig_feats[k][layer]))
                deltas = np.array(deltas)
                if len(deltas) == 0:
                    continue
                # bootstrap CI over pairs
                n_boot = 1000
                rng = np.random.default_rng(0)
                boot_means = [deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n_boot)]
                ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
                stats[row][k][layer] = {
                    "mean_rel_delta": float(deltas.mean()),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "n": int(len(deltas)),
                }

    # Action-level delta (gripper dim = index 6), relative to baseline1
    action_stats = {}
    for row in ALL_ROWS:
        compare_row = "baseline2" if row == "repeat" else row
        deltas_grip = []
        deltas_all = []
        for pair_idx in range(cfg.num_pairs):
            a_orig = action_records["baseline1"][pair_idx]
            a_var = action_records[compare_row][pair_idx]
            deltas_grip.append(float(np.abs(a_var[:, 6] - a_orig[:, 6]).mean()))
            deltas_all.append(float(np.linalg.norm(a_var - a_orig, axis=-1).mean()))
        action_stats[row] = {
            "mean_abs_delta_gripper_dim": float(np.mean(deltas_grip)),
            "mean_l2_delta_all_dims": float(np.mean(deltas_all)),
        }

    out = {
        "test": "T8_fixed_input_control",
        "task_name": cfg.task_name,
        "num_pairs": cfg.num_pairs,
        "num_denoising_steps": n_steps,
        "sigma_schedule_note": "k=0 is sigma_max (near-pure noise action latent); k=4 is final (sigma_min)",
        "probe_layers": PROBE_LAYERS,
        "variants": ALL_ROWS,
        "variant_definitions": {
            "repeat": "same obs_base run twice with identical seed (numerical floor / determinism check)",
            "img_swap": "primary_image replaced with a different episode's primary_image; proprio/secondary/wrist/task unchanged",
            "proprio_swap": "proprio replaced with a different episode's proprio; images/task unchanged",
            "all_swap": "primary_image+secondary_image+wrist_image+proprio all replaced with a different episode's values",
        },
        "feature_delta_by_variant_k_layer": stats,
        "action_delta_by_variant": action_stats,
    }
    with open(output_dir / "t8_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log_message(f"Saved: {output_dir / 't8_results.json'}")

    # ── Plot: relative Δfeat at k=0 across layers, per variant ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    layers = PROBE_LAYERS
    for variant in SWAP_VARIANTS:
        ys0 = [stats[variant][0].get(l, {}).get("mean_rel_delta", np.nan) for l in layers]
        ys4 = [stats[variant][n_steps - 1].get(l, {}).get("mean_rel_delta", np.nan) for l in layers]
        axes[0].plot(layers, ys0, marker="o", label=variant)
        axes[1].plot(layers, ys4, marker="o", label=variant)
    floor0 = [stats["repeat"][0].get(l, {}).get("mean_rel_delta", np.nan) for l in layers]
    floor4 = [stats["repeat"][n_steps - 1].get(l, {}).get("mean_rel_delta", np.nan) for l in layers]
    axes[0].plot(layers, floor0, marker="x", linestyle="--", color="gray", label="repeat (floor)")
    axes[1].plot(layers, floor4, marker="x", linestyle="--", color="gray", label="repeat (floor)")
    axes[0].set_title(f"k=0 (σ≈{80.0:.0f}, near-pure noise)")
    axes[1].set_title(f"k={n_steps-1} (final step)")
    for ax in axes:
        ax.set_xlabel("Block")
        ax.set_ylabel("relative ‖Δfeat‖ vs repeat-baseline")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"T8: action-token feature sensitivity to conditioning swap ({cfg.task_name}, N={cfg.num_pairs} pairs)")
    plt.tight_layout()
    plt.savefig(output_dir / "t8_feature_sensitivity.png", dpi=150)
    plt.close()
    log_message(f"Saved: {output_dir / 't8_feature_sensitivity.png'}")

    capture.remove()
    log_message("\nT8 complete.")


if __name__ == "__main__":
    main()
