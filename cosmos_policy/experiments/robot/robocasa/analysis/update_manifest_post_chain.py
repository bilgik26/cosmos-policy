"""
Update run_manifest.json with files created by run_analysis_chain.sh.
Run after the chain completes.
"""
import hashlib
import json
from pathlib import Path


RESULTS_BASE = Path("cosmos_policy/experiments/robot/robocasa/analysis/results")
MANIFEST_PATH = RESULTS_BASE / "run_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def add_file(manifest: dict, rel_path: str, section: str, description: str):
    path = RESULTS_BASE / rel_path
    if not path.exists():
        print(f"  SKIP (not found): {rel_path}")
        return
    sha = sha256_file(path)
    size = path.stat().st_size
    entry = {
        "path": str(path),
        "sha256": sha,
        "size_bytes": size,
        "section": section,
        "description": description,
    }
    key = str(path)
    if key in manifest.get("files", {}):
        print(f"  UPDATE: {rel_path}")
    else:
        print(f"  ADD: {rel_path}")
    manifest.setdefault("files", {})[key] = entry


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    old_count = len(manifest.get("files", {}))
    print(f"Current manifest: {old_count} files")

    # Scan self_attention_v2 directory
    attn_dir = RESULTS_BASE / "self_attention_v2"
    if attn_dir.exists():
        for p in sorted(attn_dir.iterdir()):
            if p.suffix in (".json", ".png", ".npz"):
                rel = str(p.relative_to(RESULTS_BASE))
                if "t1_hook" in p.name:
                    desc = "T1 hook invariance test result"
                    section = "§3.I/T1"
                elif "t2_softmax" in p.name:
                    desc = "T2 softmax sanity test result"
                    section = "§3.I/T2"
                elif "attn_stats" in p.name:
                    desc = "Self-attention T matrices + rollout stats"
                    section = "§3.I"
                elif "attn_meta" in p.name:
                    desc = "Run metadata: success_rate, T1/T2 results, episode summary"
                    section = "§3.I"
                elif "selfattn_Tmatrix_rownorm" in p.name:
                    desc = "Row-normalized T matrix (per block × k step)"
                    section = "§3.I"
                elif "selfattn_modality_effectsize" in p.name:
                    desc = "Effect sizes vs uniform: action→input type attention fractions"
                    section = "§3.I"
                elif "selfattn_action_profile" in p.name:
                    desc = "Action token attention to input types per block/k"
                    section = "§3.I"
                elif "t_attn_block" in p.name:
                    desc = f"T-matrix heatmap {p.stem}"
                    section = "§3.I"
                elif "rollout" in p.name:
                    desc = f"Attention rollout {p.stem}"
                    section = "§3.I"
                elif "spatial" in p.name:
                    desc = f"Spatial attention heatmap {p.stem}"
                    section = "§3.I"
                elif "proprio_vs_image" in p.name:
                    desc = f"Proprio vs image attention {p.stem}"
                    section = "§3.I"
                elif "cross_output" in p.name:
                    desc = f"Cross-output attention comparison {p.stem}"
                    section = "§3.I"
                else:
                    desc = f"Self-attention v2 output: {p.name}"
                    section = "§3.I"
                add_file(manifest, rel, section, desc)

    # Specific §3.C F_θ files
    precond_files = [
        ("precond_ftheta/precond_normalized_by_sqrt_d.json", "§3.C", "F_θ norms per sigma step: ‖F_θ‖/√d, CV%, Gaussian bound"),
        ("precond_ftheta/precond_norms_Ftheta_Dtheta_score.png", "§3.C", "F_θ vs D_θ vs score norms plot"),
    ]
    for rel_path, section, desc in precond_files:
        add_file(manifest, rel_path, section, desc)

    # §5 null model files
    null_files = [
        ("null_model_random_init/features.npz", "§5", "Action token features: random-init DiT (null model)"),
        ("null_model_random_init/null_vs_trained_pr.json", "§5", "PR comparison: trained vs random-init null model"),
        ("null_model_random_init/null_vs_trained_pr.png", "§5", "PR comparison plot"),
        ("null_model_random_init/null_vs_trained_cka_k0.png", "§5", "CKA matrix comparison k=0"),
        ("null_model_random_init/null_vs_trained_cka_k4.png", "§5", "CKA matrix comparison k=4"),
    ]
    for rel_path, section, desc in null_files:
        add_file(manifest, rel_path, section, desc)

    new_count = len(manifest.get("files", {}))
    manifest["total_files"] = new_count
    print(f"\nUpdated manifest: {new_count} files (+{new_count - old_count})")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
