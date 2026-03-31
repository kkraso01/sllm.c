# Final submission checklist (ALC mechanism paper)

Date: 2026-03-31

## 1) Files needed for submission

- Paper source:
  - `paper/main.tex`
  - `paper/sections/*.tex`
  - `paper/references.bib`
- Core results artifacts:
  - `paper/results/metrics.csv`
  - `paper/results/summary.json`
  - `paper/results/multiseed_summary.csv`
  - `paper/results/multiseed_summary.json`
  - `paper/results/sessionkv_results.csv`
  - `paper/results/sessionkv_summary.json`
- Tables:
  - `paper/tables/main_results.tex`
  - `paper/tables/sessionkv_main.tex`
  - `paper/tables/sessionkv_overwrite.tex`
  - `paper/tables/sessionkv_capacity.tex`
  - `paper/tables/tinyqa_results.tex`
- Repro scripts:
  - `scripts/run_alc_publication_suite.sh`
  - `scripts/run_alc_multiseed.py`
  - `scripts/make_paper_artifacts.py`
  - `experiments/alc_publication_suite.c`

## 2) Claims currently supported

- SessionKV-DR directly measures write/retain/retrieve/overwrite/persistence behavior.
- Full ALC can be compared against baseline and no-write on all primary SessionKV-DR sub-tasks.
- Persistence save/load behavior is explicitly tested with reload-time recall.
- TinyQA remains as supporting semi-realistic evidence.

## 3) Evidence mapping (claim -> artifact)

- Delayed recall, overwrite, persistence, capacity/interference:
  - `paper/results/sessionkv_results.csv`
  - `paper/results/sessionkv_summary.json`
  - `paper/tables/sessionkv_main.tex`
  - `paper/tables/sessionkv_overwrite.tex`
  - `paper/tables/sessionkv_capacity.tex`
- Multi-seed aggregate stats:
  - `paper/results/multiseed_summary.csv`
  - `paper/results/multiseed_summary.json`
- Supporting TinyQA:
  - `paper/results/tinyqa_results.csv`
  - `paper/results/tinyqa_summary.json`
  - `paper/tables/tinyqa_results.tex`

## 4) Remaining weak points

- Primary benchmark is targeted synthetic memory evaluation, not broad downstream NLP.
- No-write competitiveness in some settings indicates remaining write interference risk.
- Efficiency remains lightweight/tiny setup only.

## 5) Realistic venue readiness now

- **arXiv/workshop mechanism track:** reasonable with conservative claims.
- **Main-track breadth claims:** still needs larger standard benchmarks and stronger external baselines.
