"""
run_manifest_gen.py — 設計書 §1.1 準拠の Run Manifest 生成スクリプト

既存の解析結果ファイルから run_manifest.json を生成し、
provenance (どの run のどのテンソルを使ったか) を記録する。

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.sanity.run_manifest_gen \
      --results_root cosmos_policy/experiments/robot/robocasa/analysis/results \
      --out_path cosmos_policy/experiments/robot/robocasa/analysis/results/run_manifest.json \
      --task_name PnPCounterToCab \
      --n_episodes 50 \
      --seed 195 \
      --success_rate 0.60 \
      --success_count 30 \
      --n_policy_calls 1108
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

# ファイルの SHA256 を計算する (大きいファイルはチャンク読み取り)
def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                h.update(data)
        return "sha256:" + h.hexdigest()
    except Exception as e:
        return f"error:{e}"


def file_entry(path: Path, root: Path) -> dict:
    rel = str(path.relative_to(root))
    size = path.stat().st_size if path.exists() else -1
    return {
        "path": rel,
        "exists": path.exists(),
        "size_bytes": size,
        "sha256": sha256_file(str(path)) if path.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root",    required=True)
    parser.add_argument("--out_path",        required=True)
    parser.add_argument("--task_name",       default="PnPCounterToCab")
    parser.add_argument("--n_episodes",      type=int,   default=50)
    parser.add_argument("--seed",            type=int,   default=195)
    parser.add_argument("--success_rate",    type=float, default=0.60)
    parser.add_argument("--success_count",   type=int,   default=30)
    parser.add_argument("--n_policy_calls",  type=int,   default=1108)
    parser.add_argument("--hook_mode",       default="unknown")
    parser.add_argument("--git_commit",      default="unknown")
    args = parser.parse_args()

    root = Path(args.results_root)

    # ── 既知のファイルリスト ────────────────────────────────────────────────
    tracked_files = {
        "action_denoising": [
            "step_actions.npz",
            "denoising_light_records.json",
            "analysis_stats.json",
            "summary_mechanism_analysis.png",
            "fft_analysis.png",
            "score_norm.png",
        ],
        "action_features": [
            "features.npz",
            "feature_stats.json",
        ],
        "action_crossattn": [
            "crossattn.npz",
            "crossattn_meta.json",
        ],
        "self_attention": [
            "t_matrices.npy",
            "spatial_maps.npy",
            "attn_meta.json",
            "attn_stats.json",
        ],
        "action_probe": [
            "probe_stats.json",
        ],
    }

    fingerprints: dict = {}
    for subdir, files in tracked_files.items():
        for fname in files:
            fpath = root / subdir / fname
            entry = file_entry(fpath, root)
            key = f"{subdir}/{fname}"
            fingerprints[key] = entry
            status = "✓" if entry["exists"] else "✗ MISSING"
            print(f"  {status}  {key}  ({entry['size_bytes']} bytes)")

    # ── 動的に見つかるファイル ─────────────────────────────────────────────
    for subdir in ["action_features", "action_probe_v2", "action_denoising_reanalysis"]:
        dpath = root / subdir
        if dpath.exists():
            for fpath in sorted(dpath.glob("*.png")) + sorted(dpath.glob("*.json")):
                key = str(fpath.relative_to(root))
                if key not in fingerprints:
                    fingerprints[key] = file_entry(fpath, root)

    manifest = {
        "run_id": f"{args.task_name}_{args.n_episodes}ep_seed{args.seed}",
        "generated_at": datetime.now().isoformat(),
        "task": args.task_name,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "hook_mode": args.hook_mode,
        "success_rate": args.success_rate,
        "success_count": args.success_count,
        "n_policy_calls": args.n_policy_calls,
        "sigma_schedule": [80.0, 42.3, 21.0, 9.6, 4.0],
        "git_commit": args.git_commit,
        "capture_fingerprints": fingerprints,
        "p9_note": (
            "本マニフェストが参照する全ファイルは capture_fingerprints に記録。"
            "レポートで参照する図表ファイルは exists=true のもののみ使用可。"
        ),
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved run_manifest.json: {out_path}")

    # ── 存在チェックサマリー ────────────────────────────────────────────────
    missing = [k for k, v in fingerprints.items() if not v["exists"]]
    present = [k for k, v in fingerprints.items() if v["exists"]]
    print(f"\n  Present: {len(present)} files")
    print(f"  Missing: {len(missing)} files")
    if missing:
        print("  Missing files:")
        for m in missing:
            print(f"    ✗ {m}")


if __name__ == "__main__":
    main()
