"""
Pad マスク・アテンション寄与の直接検証 — レビュー §3.7 対応

背景（コード調査で判明した事実、本スクリプトはこれを実データで直接確認する）:
  1. `Attention.attn_op = DotProductAttention(..., attn_mask_type="no_mask")` — cross_attn は
     アーキテクチャ上マスクを一切適用しない（フックが見落とした隠れマスクは存在しない）。
  2. T5 embedding 計算 (`get_t5_emb.py::encode_prompts`) は実配列長を超える位置を
     明示的に 0 埋めしてからキャッシュしている（`encoded_text[b][lengths[b]:] = 0`）。
     実際に使用中のキャッシュ (`robocasa_t5_embeddings.pkl`) でも pad 位置のノルムが
     厳密に 0.0 であることを直接確認済み。
  3. `k_proj`/`v_proj` は `bias=False`（実チェックポイントにも bias パラメータが存在しない
     ことを確認済み）。よって K_pad = k_norm(k_proj(0)) = 0、V_pad = v_proj(0) = 0 が
     厳密に成り立つ（RMSNorm(0)=0 なので k_norm を通しても 0 のまま）。
  4. したがって pad 位置に softmax 確率質量の 97% が乗ろうと、対応する V が厳密に 0 の
     ため、cross-attention 出力への寄与は理論上 0 のはず。

本スクリプトはこれを「フックで Q,K,V を再構成した attention·V」と「モデルの実出力
（cross_attn モジュールの実際の forward 戻り値）」を直接数値照合することで実証する
（レビューが要求した検証そのもの）。あわせて出力を real-token 由来項と pad-token
由来項に分解し、pad 由来項のノルムを定量化する。

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.t_pad_attn_verify \\
      --config ... --ckpt_path ... (run_t_pad_attn_verify.sh を参照)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ROBOT_PLATFORM", "ROBOCASA")

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import log_message
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig, create_robocasa_env, prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import PROBE_LAYERS

ACTION_T_IDX = 5
PATCHES_PER_T = 196


class PadVerifyCapture:
    """cross_attn の compute_qkv 入出力と実際の forward 出力を同時に捕捉する。"""

    def __init__(self, probe_layers):
        self.probe_layers = probe_layers
        self._step = -1
        self.records = []  # list of dict per (layer, step) captured during this call
        self._handles = []

    def register(self, model):
        net = model.net
        for l in self.probe_layers:
            block = net.blocks[l]
            orig_compute_qkv = block.cross_attn.compute_qkv
            capture = self
            layer_idx = l

            def make_wrapped_qkv(orig_fn, layer_idx):
                def wrapped(x, context=None, rope_emb=None):
                    q, k, v = orig_fn(x, context, rope_emb=rope_emb)
                    if capture._step >= 0:
                        capture._pending = capture._pending if hasattr(capture, "_pending") else {}
                        capture._pending[layer_idx] = {
                            "q": q.detach().float().cpu(),
                            "k": k.detach().float().cpu(),
                            "v": v.detach().float().cpu(),
                        }
                    return q, k, v
                return wrapped

            def make_attn_op_hook(layer_idx):
                # Hook the attn_op submodule directly to capture its RAW output
                # (i.e. attn·V BEFORE output_proj -- compute_attention's return value is
                # post-output_proj, which is a different quantity from the manual attn·V
                # reconstruction and must not be compared against it directly).
                def hook(module, inp, output):
                    if capture._step >= 0 and hasattr(capture, "_pending") and layer_idx in capture._pending:
                        entry = capture._pending[layer_idx]
                        entry["real_output_pre_proj"] = output.detach().float().cpu()
                        capture.records.append({
                            "layer": layer_idx, "step": capture._step, **entry,
                        })
                        del capture._pending[layer_idx]
                return hook

            block.cross_attn.compute_qkv = make_wrapped_qkv(orig_compute_qkv, layer_idx)
            block.cross_attn.attn_op.register_forward_hook(make_attn_op_hook(layer_idx))

    def reset_for_policy_call(self):
        self._step = -1
        self.records = []

    def before_denoise_step(self):
        self._step += 1


def get_action_with_capture(cfg, model, dataset_stats, observation, task_description,
                             capture, seed, num_denoising_steps):
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


def analyze_record(rec, n_real: int, n_heads: int, head_dim: int):
    """
    q,k,v: (B, S_context_or_query, H, D) after compute_qkv (cross-attn: q over full seq len
    (T*H*W), k/v over context length Sk=512).
    real_output_pre_proj: (B, S, H*D) raw attn_op(q,k,v) output -- BEFORE output_proj (captured via a hook on
    attn_op itself so it is directly comparable to a manual attn·V reconstruction).
    Reconstructs manual softmax(QK^T/sqrt(d)) @ V restricted to the action-token query rows,
    decomposes into real (0..n_real-1) vs pad (n_real..511) contributions, and compares the
    manually reconstructed action-token output rows against the model's real_output rows.
    """
    q, k, v = rec["q"][0], rec["k"][0], rec["v"][0]  # (Sq,H,D), (Sk,H,D), (Sk,H,D)
    real_out_raw = rec["real_output_pre_proj"][0]  # attn_op raw output (pre-output_proj), shape (Sq, H*D)
    Sq_check = real_out_raw.shape[0]
    real_out = real_out_raw.reshape(Sq_check, n_heads, head_dim)  # (Sq, H, D)

    Sq = q.shape[0]
    start = ACTION_T_IDX * PATCHES_PER_T
    end = start + PATCHES_PER_T
    q_action = q[start:end]  # (196, H, D)

    scale = head_dim ** -0.5
    # (196, H, D) x (Sk, H, D) -> (H, 196, Sk)
    q_ = q_action.permute(1, 0, 2)         # (H,196,D)
    k_ = k.permute(1, 0, 2)                 # (H,Sk,D)
    v_ = v.permute(1, 0, 2)                 # (H,Sk,D)
    scores = torch.bmm(q_, k_.transpose(1, 2)) * scale  # (H,196,Sk)
    attn = torch.softmax(scores, dim=-1)                 # (H,196,Sk)

    out_full = torch.bmm(attn, v_)                       # (H,196,D) -- full (real+pad) reconstruction
    out_real = torch.bmm(attn[:, :, :n_real], v_[:, :n_real])   # real-only contribution
    out_pad = torch.bmm(attn[:, :, n_real:], v_[:, n_real:])    # pad-only contribution

    real_out_action = real_out[start:end].permute(1, 0, 2)  # (H,196,D) -- model's actual output for action rows

    max_diff = (out_full - real_out_action).abs().max().item()
    weight_on_pad = attn[:, :, n_real:].sum(dim=-1).mean().item()
    weight_on_real = attn[:, :, :n_real].sum(dim=-1).mean().item()
    norm_out_full = out_full.norm(dim=-1).mean().item()
    norm_out_real = out_real.norm(dim=-1).mean().item()
    norm_out_pad = out_pad.norm(dim=-1).mean().item()
    norm_v_real = v_[:, :n_real].norm(dim=-1).mean().item()
    norm_v_pad = v_[:, n_real:].norm(dim=-1).mean().item()

    return {
        "max_abs_diff_reconstruction_vs_real": max_diff,
        "attn_weight_on_real": weight_on_real,
        "attn_weight_on_pad": weight_on_pad,
        "output_norm_full_reconstruction": norm_out_full,
        "output_norm_real_only_term": norm_out_real,
        "output_norm_pad_only_term": norm_out_pad,
        "v_norm_real_mean": norm_v_real,
        "v_norm_pad_mean": norm_v_pad,
    }


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
    parser.add_argument("--num_calls", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default="cosmos_policy/experiments/robot/robocasa/analysis/results/pad_attn_verify")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PolicyEvalConfig(
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

    n_heads = model.net.blocks[0].cross_attn.n_heads
    head_dim = model.net.blocks[0].cross_attn.head_dim
    log_message(f"n_heads={n_heads} head_dim={head_dim}")

    capture = PadVerifyCapture(PROBE_LAYERS)
    capture.register(model)

    set_seed_everywhere(cfg.seed)
    all_results = []

    for call_idx in range(args.num_calls):
        obs = env.reset()
        task_description = env.get_ep_meta().get("lang", cfg.task_name)
        observation = prepare_observation(obs, cfg.flip_images)

        # determine n_real from the actual T5 embedding used for this task description
        from cosmos_policy.experiments.robot.cosmos_utils import get_t5_embedding_from_cache
        t5_emb = get_t5_embedding_from_cache(task_description)
        t5_emb_np = t5_emb.float().cpu().numpy()[0]
        n_real = int((np.linalg.norm(t5_emb_np, axis=-1) > 1e-6).sum())

        with torch.no_grad():
            get_action_with_capture(
                cfg=cfg, model=model, dataset_stats=dataset_stats,
                observation=observation, task_description=task_description,
                capture=capture, seed=cfg.seed, num_denoising_steps=cfg.num_denoising_steps_action,
            )

        for rec in capture.records:
            stats = analyze_record(rec, n_real, n_heads, head_dim)
            stats["call_idx"] = call_idx
            stats["layer"] = rec["layer"]
            stats["step"] = rec["step"]
            stats["n_real"] = n_real
            stats["task_description"] = task_description
            all_results.append(stats)
        log_message(f"call {call_idx}: task='{task_description}' n_real={n_real} "
                    f"captured {len(capture.records)} (layer,step) records")

    with open(output_dir / "pad_attn_verify.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log_message(f"Saved: {output_dir / 'pad_attn_verify.json'}")

    # Summary
    max_diffs = [r["max_abs_diff_reconstruction_vs_real"] for r in all_results]
    pad_out_norms = [r["output_norm_pad_only_term"] for r in all_results]
    real_out_norms = [r["output_norm_real_only_term"] for r in all_results]
    v_pad_norms = [r["v_norm_pad_mean"] for r in all_results]
    weight_on_pad = [r["attn_weight_on_pad"] for r in all_results]

    log_message(f"\n=== SUMMARY (N={len(all_results)} layer x step x call records) ===")
    log_message(f"max|reconstruction - real_output| : max={max(max_diffs):.3e} mean={np.mean(max_diffs):.3e}")
    log_message(f"mean attn weight on pad positions : {np.mean(weight_on_pad):.4f}")
    log_message(f"mean ‖V_pad‖ (should be ~0)        : {np.mean(v_pad_norms):.3e}")
    log_message(f"mean output norm from pad term    : {np.mean(pad_out_norms):.3e}")
    log_message(f"mean output norm from real term   : {np.mean(real_out_norms):.3e}")

    summary = {
        "n_records": len(all_results),
        "max_reconstruction_vs_real_diff": max(max_diffs),
        "mean_reconstruction_vs_real_diff": float(np.mean(max_diffs)),
        "mean_attn_weight_on_pad": float(np.mean(weight_on_pad)),
        "mean_v_norm_pad": float(np.mean(v_pad_norms)),
        "mean_output_norm_from_pad_term": float(np.mean(pad_out_norms)),
        "mean_output_norm_from_real_term": float(np.mean(real_out_norms)),
    }
    with open(output_dir / "pad_attn_verify_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
