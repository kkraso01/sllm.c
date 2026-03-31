# LOCAL_VALIDATION.md

Validation date: 2026-03-31 (UTC)

This file records commands executed in this environment to validate the local reproducibility path.

## Commands run and outcomes

### Build
- `./scripts/local/build_all.sh`
  - Result: ✅ success.
  - Built:
    - `train_gpt2`
    - `test_gpt2`
    - `experiments/alc_publication_suite`
    - `dev/test/alc_smoke`
    - `dev/test/tiny_alc_e2e`
    - `dev/test/alc_hardening`

### Artifact-independent tests
- `./scripts/local/run_tests.sh`
  - Result: ✅ success.
  - Sub-tests executed:
    - `./dev/test/alc_smoke`
    - `./dev/test/tiny_alc_e2e`
    - `./dev/test/alc_hardening`

### Tiny local ALC demo
- `./scripts/local/run_alc_demo.sh`
  - Result: ✅ success.
  - Output file:
    - `artifacts/local_demo/metrics_demo_seed42.csv`

### Primary benchmark (SessionKV-DR)
- `./scripts/local/run_sessionkv.sh`
  - Result: ✅ success.
  - Output files:
    - `artifacts/sessionkv/metrics_sessionkv_seed42.csv`
    - `artifacts/sessionkv/sessionkv_summary_seed42.json`
  - Key printed comparison (`delayed_acc_delay_6_facts_4`):
    - baseline: `0.125000`
    - alc_no_write: `0.125000`
    - alc: `0.718750`

### Persistence (save/load) demo
- `./scripts/local/run_persistence_demo.sh`
  - Result: ✅ success.
  - Confirmed sidecar state files present:
    - `artifacts/persistence_demo/sessionkv_state_baseline.bin`
    - `artifacts/persistence_demo/sessionkv_state_nowrite.bin`
    - `artifacts/persistence_demo/sessionkv_state_alc.bin`
  - Persistence metrics confirmed from CSV:
    - save/load ok flags were `1.0`
    - `alc` persistence recall was `1.0`

### Main paper benchmark suite
- `./scripts/run_alc_publication_suite.sh`
  - Result: ✅ success with one optional-tool warning.
  - Generated:
    - `paper/results/*.csv` and `paper/results/*.json`
    - `paper/tables/*.md` and `paper/tables/*.tex`
  - Figure generation:
    - ⚠️ skipped because `matplotlib` is not installed in this environment.
    - Warning file written by script behavior.

## Environment-dependent gaps

1. `matplotlib` is optional and was unavailable here; figure PNG generation was skipped.
2. LaTeX toolchain (`pdflatex`, `bibtex`) was not part of this validation pass.
3. CUDA/GPU path was not required for this local CPU reproducibility pass.
