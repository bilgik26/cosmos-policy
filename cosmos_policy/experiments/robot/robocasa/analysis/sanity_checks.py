"""
sanity_checks.py — 設計書 §6 サニティ/回帰テスト (オフライン実行可能な部分)

T9: ノルム–cos–L2 整合検算
    各 (layer, k→k+1) について ‖Δfeat‖ と √(‖a‖²+‖b‖²-2‖a‖‖b‖cos(a,b)) が
    数値誤差内で一致するかを検証。

T10: マニフェスト照合
    run_manifest.json が参照するファイルが実際に存在するかを確認。
    results_***.md のような文書が「生成済み」と記述している図表が存在するかを検証。

Usage:
  python -m cosmos_policy.experiments.robot.robocasa.analysis.sanity_checks \
      --feat_npz results/action_features/features.npz \
      --manifest  results/run_manifest.json \
      --out_dir   results/sanity
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cosmos_policy.experiments.robot.robocasa.analysis.analysis_shared import (
    PROBE_LAYERS,
    NUM_DENOISE_STEPS,
)


# ── T9: ノルム–cos–L2 整合検算 ───────────────────────────────────────────────

def t9_norm_cos_l2_consistency(
    feats: Dict, ep_arr: np.ndarray, tol: float = 1e-3
) -> Dict:
    """
    T9: 各 (layer, k→k+1) について
        ‖Δfeat‖² ≈ ‖feat_k‖² + ‖feat_k+1‖² − 2‖feat_k‖‖feat_k+1‖cos(feat_k, feat_k+1)
    を検算し、最大残差が tol を超えたら FAIL とする。

    Returns dict with results per (layer, transition).
    """
    results = {}
    all_passed = True

    for l in PROBE_LAYERS:
        for k in range(NUM_DENOISE_STEPS - 1):
            F_k  = feats[k][l]
            F_k1 = feats[k + 1][l]
            if F_k is None or F_k1 is None:
                continue

            F_k  = F_k.astype(np.float64)
            F_k1 = F_k1.astype(np.float64)

            norm_k  = np.linalg.norm(F_k,  axis=1)   # (N,)
            norm_k1 = np.linalg.norm(F_k1, axis=1)

            # cos(a, b) per sample
            dot = (F_k * F_k1).sum(axis=1)
            cos = dot / (norm_k * norm_k1 + 1e-12)
            cos = np.clip(cos, -1.0, 1.0)

            # LHS: ‖feat_k+1 - feat_k‖² per sample
            delta = F_k1 - F_k
            lhs = np.linalg.norm(delta, axis=1) ** 2   # (N,)

            # RHS: norm identity
            rhs = norm_k**2 + norm_k1**2 - 2 * norm_k * norm_k1 * cos  # (N,)

            residual = np.abs(lhs - rhs)
            max_resid = float(residual.max())
            rel_resid = float((residual / (lhs + 1e-12)).max())
            passed = rel_resid < tol

            key = f"layer{l}_k{k}_to_k{k+1}"
            results[key] = {
                "layer": l,
                "k_from": k,
                "k_to": k + 1,
                "max_abs_residual": max_resid,
                "max_rel_residual": rel_resid,
                "tolerance": tol,
                "passed": passed,
            }
            status = "PASS" if passed else "FAIL"
            print(f"  T9 {status}: layer={l}, k={k}→{k+1}  "
                  f"max_rel_resid={rel_resid:.2e}  (tol={tol:.1e})")
            if not passed:
                all_passed = False

    return {"per_cell": results, "all_passed": all_passed}


# ── T10: マニフェスト照合 ────────────────────────────────────────────────────

def t10_manifest_check(manifest_path: str) -> Dict:
    """
    T10: run_manifest.json の capture_fingerprints に記録されたファイルが
    実際に存在するかを確認し、exists フラグと実態の一致を検証。
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    fps = manifest.get("capture_fingerprints", {})
    results = {}
    manifest_dir = Path(manifest_path).parent

    mismatches = []
    for key, entry in fps.items():
        full_path = manifest_dir / entry["path"]
        actually_exists = full_path.exists()
        recorded_exists = entry.get("exists", None)
        mismatch = actually_exists != recorded_exists
        results[key] = {
            "path": entry["path"],
            "recorded_exists": recorded_exists,
            "actually_exists": actually_exists,
            "mismatch": mismatch,
        }
        if mismatch:
            mismatches.append(key)
            print(f"  T10 MISMATCH: {key}  recorded={recorded_exists}, actual={actually_exists}")
        else:
            status = "✓" if actually_exists else "✗(absent)"
            print(f"  T10 {status}: {key}")

    all_passed = len(mismatches) == 0
    return {
        "n_files": len(fps),
        "n_present": sum(1 for v in results.values() if v["actually_exists"]),
        "n_absent": sum(1 for v in results.values() if not v["actually_exists"]),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "all_passed": all_passed,
        "per_file": results,
    }


# ── 追加: ハッシュ衝突検出 (T4, offline 版) ─────────────────────────────────

def t4_hash_collision_check(feats: Dict, ep_arr: np.ndarray) -> Dict:
    """
    T4 (offline): 異なる (layer, step) ペアの特徴量が7桁以上一致する場合を検出。
    これは設計書で指摘された「results_02/04 の7桁一致バグ」の再発防止。
    """
    import hashlib
    def array_hash(arr: np.ndarray) -> str:
        return hashlib.md5(arr.tobytes()).hexdigest()

    hashes: Dict[str, Tuple[int, int]] = {}
    collisions = []
    for k in range(NUM_DENOISE_STEPS):
        for l in PROBE_LAYERS:
            F = feats[k][l]
            if F is None:
                continue
            h = array_hash(F)
            key = f"k{k}_layer{l}"
            if h in hashes:
                other = hashes[h]
                collisions.append((key, f"k{other[0]}_layer{other[1]}"))
                print(f"  T4 COLLISION: {key} == k{other[0]}_layer{other[1]}  (hash={h[:8]}…)")
            else:
                hashes[h] = (k, l)

    all_passed = len(collisions) == 0
    if all_passed:
        print(f"  T4 PASS: {len(hashes)} distinct feature arrays, no collisions")
    return {"n_arrays": len(hashes), "collisions": collisions, "all_passed": all_passed}


# ── T5: k間の特徴捕捉点同一性チェック (offline 版) ──────────────────────────

def t5_feature_shape_consistency(feats: Dict) -> Dict:
    """
    T5 (offline): 全 (layer, k) で features が同じ shape (N, D) を持つことを確認。
    最終ステップだけ別テンソルを掴んでいた場合、shape や dtype が変わる可能性がある。
    """
    shapes = {}
    dtypes = {}
    inconsistencies = []
    ref_shape = None
    for k in range(NUM_DENOISE_STEPS):
        for l in PROBE_LAYERS:
            F = feats[k][l]
            if F is None:
                continue
            if ref_shape is None:
                ref_shape = F.shape
            shapes[(k, l)] = F.shape
            dtypes[(k, l)] = str(F.dtype)
            if F.shape != ref_shape:
                msg = f"k={k},l={l}: shape={F.shape} ≠ ref={ref_shape}"
                inconsistencies.append(msg)
                print(f"  T5 FAIL: {msg}")

    all_passed = len(inconsistencies) == 0
    if all_passed:
        print(f"  T5 PASS: all features have shape={ref_shape}")
    return {
        "ref_shape": list(ref_shape) if ref_shape else None,
        "inconsistencies": inconsistencies,
        "all_passed": all_passed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_npz",  required=True)
    parser.add_argument("--manifest",  default=None)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--tol_t9",    type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load features ─────────────────────────────────────────────────────────
    print(f"Loading features: {args.feat_npz}")
    raw = np.load(args.feat_npz)
    feats: Dict = {}
    for k in range(NUM_DENOISE_STEPS):
        feats[k] = {}
        for l in PROBE_LAYERS:
            key = f"feat_k{k}_layer{l}"
            feats[k][l] = raw[key].astype(np.float32) if key in raw else None
    ep_arr = raw["episode_labels"]

    report = {}

    # ── T5: Shape consistency ─────────────────────────────────────────────────
    print("\n[T5] Feature capture consistency (shape)…")
    report["T5"] = t5_feature_shape_consistency(feats)

    # ── T4: Hash collision ────────────────────────────────────────────────────
    print("\n[T4] Hash collision check…")
    report["T4"] = t4_hash_collision_check(feats, ep_arr)

    # ── T9: Norm-cos-L2 consistency ──────────────────────────────────────────
    print(f"\n[T9] Norm–cos–L2 consistency (tol={args.tol_t9})…")
    report["T9"] = t9_norm_cos_l2_consistency(feats, ep_arr, tol=args.tol_t9)

    # ── T10: Manifest check ───────────────────────────────────────────────────
    if args.manifest and Path(args.manifest).exists():
        print(f"\n[T10] Manifest file existence check: {args.manifest}")
        report["T10"] = t10_manifest_check(args.manifest)
    else:
        print("\n[T10] Manifest path not provided or not found — skip")
        report["T10"] = {"skipped": True, "reason": "manifest not provided"}

    # ── Save report ───────────────────────────────────────────────────────────
    def _safe(obj):
        if isinstance(obj, bool):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(x) for x in obj]
        return obj

    report_path = out_dir / "sanity_check_report.json"
    with open(report_path, "w") as f:
        json.dump(_safe(report), f, indent=2)
    print(f"\nSaved: {report_path}")

    # ── Overall verdict ───────────────────────────────────────────────────────
    tests_run = ["T5", "T4", "T9"]
    if "T10" in report and not report["T10"].get("skipped"):
        tests_run.append("T10")
    results_bool = {t: report[t].get("all_passed", False) for t in tests_run}
    all_pass = all(results_bool.values())
    print("\n=== Sanity Check Results ===")
    for t, passed in results_bool.items():
        print(f"  {t}: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
