# Cosmos Policy — Verification Suite

This package holds the implementation behind the three verification reports in
`docs/analysis/`:

| Report | Suites it draws on |
|---|---|
| [`verification_report_v3.md`](../../../../../../docs/analysis/verification_report_v3.md) | `collection`, `mechanism`, `representation`, `causal_gates`, `probing`, `attention`, `text_conditioning`, `image_latent`, `sanity` |
| [`attractor_verification_report.md`](../../../../../../docs/analysis/attractor_verification_report.md) | `attractor` (+ `common/phase_labeling.py`, `common/analysis_shared.py`) |
| [`latent_dynamics_verification_report.md`](../../../../../../docs/analysis/latent_dynamics_verification_report.md) | `latent_dynamics` (reuses several `attractor` functions directly) |
| [`latent_dynamics_verification/report_v3.md`](../../../../../../docs/analysis/latent_dynamics_verification/report_v3.md) | `intervention_v3` (reuses `attractor/collect_multitask.py` output + `common/phase_labeling.py`, `common/v4_stats_lib.py`, and the EDM sampler internals under `cosmos_policy/_src/imaginaire/functional/{multi_step,runge_kutta}.py`) |

Read this file before adding a new verification or visualization — it exists
so that both humans and Claude Code can extend the suite without having to
rediscover these conventions from scratch each time.

## Layout

```
verification/
  README.md              <- you are here
  common/                 <- shared library code, imported across suites
    env.sh                <- canonical GPU/EGL/venv setup, sourced by run_*.sh
    analysis_shared.py    <- PROBE_LAYERS, T_NAMES, STATE_T, effective_rank(), linear_cka(), ...
    v4_stats_lib.py       <- bootstrap CI, permutation tests, BH-FDR, fold-PCA probing, Cohen's d
    linear_probe_v2.py    <- fold-internal PCA + LogisticRegression + LOEO probing pipeline
    phase_labeling.py     <- physics-based (latent-independent) skill-phase labels
  collection/             <- rollout data collection & multi-seed merge
  mechanism/              <- denoising mechanism: Δx̂0, FFT spectrum, EDM preconditioning
  representation/         <- layer-wise representation geometry: PR, norm/cos, CKA, null model
  causal_gates/           <- T6 confound-ablation gates, seed/diversity diagnostics
  probing/                <- linear probing of action-token features
  attention/               <- self-/cross-attention analysis, pad-token contribution
  text_conditioning/      <- T8/T8b language-conditioning gates
  image_latent/           <- future-image-latent analysis
  attractor/               <- attractor-geometry battery (Themes A-G, DMD, steering, critical boundary)
  latent_dynamics/         <- delay-embedding / energy-field / dynamic-steering pipeline
  intervention_v3/         <- steerability audit (PCA+SVM separability grid), value-guided
                              best-of-N baseline, EDM-sampler noise-inversion positive control,
                              observation-conditioned noise actor (SFT on inverted-noise
                              targets), observer-based minimal-norm setpoint intervention
                              (gripper + EE-height/EE-velocity variants, early-denoising-step
                              sweep, real-vs-dummy-prompt zeta distribution check), Gaussian
                              Monge-map distribution-transport steering (DiMaS-style, mean-shift
                              vs full-transport ablation), multi-layer chunk-index-decayed
                              minimal-norm control (WA-LQR-style scalar layer-chain LQR),
                              paraphrase-robustness eval (verbatim vs semantically-equivalent
                              prompt vs dummy prompt; T5 paraphrase embeddings are precomputed
                              in a separate policy-free process to avoid OOM)
  sanity/                  <- T9/T10 sanity checks, G1 baseline-eval gate, run-manifest generator
  run_offline_analysis.sh  <- cross-suite orchestrator (representation + probing + mechanism)
  run_all_analysis_host.sh <- cross-suite orchestrator (Singularity host wrapper)
```

Each suite directory is a normal Python package (`__init__.py` present) plus
its `run_*.sh` wrapper script(s), co-located so a suite is self-contained. A
script that spans more than one suite (e.g. `run_offline_analysis.sh`) lives
at the `verification/` root instead of being forced into one suite.

Experiment *output data* (`results/`, currently ~11 GB) intentionally stays
at `cosmos_policy/experiments/robot/robocasa/analysis/results/` — one level
above `verification/` — and is `.gitignore`d. Every script's `--output_dir`
default is that absolute-from-repo-root path string, so it is independent of
where the script itself lives; do not move `results/` when you reorganize
code.

## Conventions

**Imports are absolute, never relative.** Every cross-module import in this
package uses the full dotted path:

```python
from cosmos_policy.experiments.robot.robocasa.analysis.verification.common.analysis_shared import (
    PROBE_LAYERS,
    effective_rank,
)
```

When you add a new script that needs something from another suite (this is
normal — e.g. `latent_dynamics/energy_field_test.py` imports from
`attractor/dmd_jacobian_stability_test.py`), just import it by its full path.
Don't add `sys.path` hacks for internal imports — `cosmos_policy` is on the
Python path already (installed editable via `uv`).

**Never hardcode directory depth from `__file__`.** A couple of legacy
scripts compute the repo root via `Path(__file__).resolve().parents[N]` or a
fixed count of `"../.."`. This breaks silently (wrong directory, not an
error) whenever a file moves to a different nesting depth — it already bit
this refactor once (`attractor/collect_multitask.py`,
`mechanism/effect_size_power.py`, both had to be corrected from depth 5→7 and
4→6 `..`s respectively when suites were introduced). Prefer walking up to a
recognizable anchor instead, e.g.:

```python
REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent
```

or simply hardcode the absolute repo path the way most scripts already do
(`/home/bilgehan.sakai/cosmos-policy`) — this codebase is single-machine and
several scripts already rely on that assumption (see `common/env.sh`).

**Every script is a standalone CLI**, `argparse` or a `pydantic`/dataclass
config parsed via `--flag value` pairs, runnable as:

```
python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.<suite>.<script> --output_dir ... [other flags]
```

Output goes under `cosmos_policy/experiments/robot/robocasa/analysis/results/<name>/`
via an `--output_dir`/`--out_dir` flag with that as its default. Pick a new,
distinct `<name>` per script/experiment — don't overwrite another script's
results directory.

**`run_*.sh` wrappers** hold the exact CLI invocation used to produce a
report's numbers (config name, checkpoint id, seeds, episode counts). This is
the reproducibility record — keep one wrapper per invocation that actually
produced reported results, not per code change.

## Adding a new verification

1. Decide which suite it belongs to (see table above), or create a new suite
   directory (`mkdir` + empty `__init__.py`) if it doesn't fit an existing
   theme — this is intentionally cheap.
2. Write `verification/<suite>/<name>.py`:
   - Pull shared constants/helpers from `common/analysis_shared.py` (probe
     layers, timestep names, `effective_rank`, `linear_cka`, ...) rather than
     redefining them.
   - For statistical claims (CI, permutation tests, effect sizes, multiple-
     comparison correction), use `common/v4_stats_lib.py` rather than
     hand-rolling — the existing verification reports lean on it heavily and
     a reviewer will expect the same rigor (bootstrap CI + permutation +
     BH-FDR, not a bare t-test).
   - Default `--output_dir` to
     `cosmos_policy/experiments/robot/robocasa/analysis/results/<name>`.
3. Write `verification/<suite>/run_<name>.sh`:
   ```bash
   #!/bin/bash
   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../common/env.sh"
   # (adjust the relative `cd`/path count above to your actual suite depth)

   python -m cosmos_policy.experiments.robot.robocasa.analysis.verification.<suite>.<name> \
       --config ... \
       --output_dir cosmos_policy/experiments/robot/robocasa/analysis/results/<name>
   ```
   `common/env.sh` sets up GPU/EGL rendering, HF cache, and activates the
   venv — new scripts should source it rather than re-pasting the
   Singularity/EGL boilerplate (existing scripts predate `env.sh` and were
   left as-is rather than mechanically rewritten, since their exact
   boilerplate is part of what was actually run for the reports).
4. Run it, then add a short section to the relevant report (or a new report)
   describing what was tested and the result — a script without a
   corresponding write-up is exactly the kind of file this refactor removed.

## Adding a new visualization

Two established patterns, pick based on cost of the underlying data:

- **Inline**: if the analysis script already holds the data in memory, plot
  directly at the end of `main()` (see `mechanism/mechanism_analysis.py`'s
  `plot_all()` for the fullest example — several PNGs per run, all written
  under `out_dir`).
- **Replot from saved artifacts**: if regenerating the data requires a GPU
  rollout (expensive) but you only want a different figure, write a small
  script that loads the saved `.npz`/`.json`/`.pkl` from an existing
  `results/<name>/` directory and only does plotting — no model, no
  simulator import. This lets you iterate on a figure in seconds instead of
  minutes. (This pattern existed here before as `*_replot.py` scripts; they
  were removed in this refactor only because they were never cited by a
  final report, not because the pattern is discouraged — recreate it next to
  the analysis script it re-plots, e.g. `attention/replot_<name>.py`.)

## Deleted / superseded

Round-2 scripts superseded by a later, more rigorous round-3 rewrite
(bootstrap/permutation/fold-PCA methodology) were removed in this refactor:
`dim_analysis_v2`, `layer_stats_v2`, `mechanism_null_v2`,
`compute_selfattn_rownorm`, `rollout_jb`, plus two pure offline replotters
(`attention_replot`, `crossattn_replot`, `feature_replot`, `feature_plot`)
and `layer_token_pca_variance` / `update_manifest_post_chain`, none of which
are cited by any of the three current reports. Their history is in git
(`checkpoint: snapshot verification scripts and reports before suite
refactor`) if you need to recover one.
