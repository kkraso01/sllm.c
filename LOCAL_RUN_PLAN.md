# Local reproducibility audit and execution plan

This file captures the **pre-edit audit** of the current repository run path and what will be fixed in this local usability pass.

## 1) Required binaries/tests for local reproducibility

### Core ALC publication/benchmark binary
- `experiments/alc_publication_suite` (compiled from `experiments/alc_publication_suite.c`)
- Current one-liner build command used in repo scripts:
  - `cc -O2 -fopenmp experiments/alc_publication_suite.c -lm -lgomp -o experiments/alc_publication_suite`

### ALC-focused tests (artifact-independent)
These tests compile against local C sources and do not require model checkpoints:
- `dev/test/alc_smoke.c`
- `dev/test/tiny_alc_e2e.c`
- `dev/test/alc_hardening.c`

## 2) Existing scripts and docs already present

### Existing run scripts
- `scripts/run_alc_publication_suite.sh`:
  - builds `experiments/alc_publication_suite`
  - runs a single-seed paper metrics run
  - runs multiseed aggregation
  - generates tables/summary artifacts
- `scripts/run_alc_multiseed.py`
- `scripts/make_paper_artifacts.py`

### Existing docs
- `paper/README.md` has a short reproduction pointer.
- `docs/adaptive_learning_component.md` documents ALC behavior and env vars.

## 3) Artifacts expected/produced today

### Inputs
- For ALC publication path: no large external model artifact is required.
- Tiny QA benchmark uses local file: `experiments/tiny_qa_dataset.txt`.

### Outputs
- Metrics CSVs in `paper/results/` (single-seed and multiseed).
- Summary JSONs in `paper/results/`.
- Tables in `paper/tables/`.
- Figures (or warning file) in `paper/figures/`.
- SessionKV persistence sidecar state binaries in `paper/results/`.

## 4) Current blockers for a clean laptop run

1. No top-level, laptop-oriented runbook (README coverage is broad but not ALC-local focused).
2. No dedicated `scripts/local/*` convenience wrappers for build/test/demo/sessionkv/persistence.
3. No single place that explains ALC env vars + expected defaults + persistence flags.
4. No explicit first-run path that walks baseline vs `alc_no_write` vs `alc` with small commands.
5. Output locations and failure troubleshooting are spread across files.

## 5) What this pass will fix

1. Add a top-level human guide: `RUN_LOCAL.md`.
2. Add local convenience scripts under `scripts/local/`:
   - `build_all.sh`
   - `run_tests.sh`
   - `run_alc_demo.sh`
   - `run_sessionkv.sh`
   - `run_persistence_demo.sh`
3. Add a centralized env-var reference doc for ALC knobs.
4. Add a reproducible baseline vs `alc_no_write` vs `alc` comparison path for SessionKV-DR.
5. Add explicit save/load persistence demo path and artifact locations.
6. Add `LOCAL_VALIDATION.md` documenting what commands were executed in this environment and outcomes.
