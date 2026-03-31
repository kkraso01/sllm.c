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
- Tables:
  - `paper/tables/main_results.tex`
  - `paper/tables/ablation_top10.tex`
  - `paper/tables/ablation_results.csv`
- Repro scripts:
  - `scripts/run_alc_publication_suite.sh`
  - `scripts/run_alc_multiseed.py`
  - `scripts/make_paper_artifacts.py`
  - `experiments/alc_publication_suite.c`

## 2) Claims currently supported

- ALC improves over stale-memory baseline on the core delayed-recall probe (multi-seed mean).
- ALC remains numerically stable in this stress setup (NaN rate 0 across 10k-update runs, all seeds).
- Persistence save/load is exact in the tested reference configuration.
- Interface training reliably reduces retrieval loss (multi-seed).
- ALC also improves over stale-memory baseline on the added language-shaped delayed-recall benchmark.

## 3) Evidence mapping (claim -> artifact)

- Core adaptation improvement:
  - `paper/results/multiseed_summary.csv` rows for `core_adaptation`.
- Stability:
  - `paper/results/multiseed_summary.csv` rows for `stability` (`nan_rate`, norm stats).
- Persistence correctness:
  - `paper/results/metrics.csv` rows for `persistence`.
- Trainability:
  - `paper/results/multiseed_summary.csv` rows for `trainability` (`loss_before`, `loss_after`).
- Language-shaped benchmark:
  - `paper/results/multiseed_summary.csv` rows for `language_benchmark`.

## 4) Remaining weak points

- Tasks remain synthetic/semi-synthetic; no standard public NLP benchmark yet.
- Stronger external retrieval/memory baselines are not included yet.
- No-write baseline is competitive and can outperform full ALC on some metrics, indicating unresolved write interference.
- Persistence and efficiency are still single-reference-run checks in this pass.

## 5) Realistic venue readiness now

- **arXiv:** ready (mechanism paper with clear limitations).
- **Workshop:** realistic now, especially for systems/mechanism-focused tracks.
- **Main track:** not yet competitive without broader benchmarks and stronger comparisons.

## 6) What is needed for stronger main-conference version

- Add at least one recognized public benchmark suite with clear task protocol.
- Add stronger external-memory/retrieval baselines.
- Expand statistical analysis to include significance testing for key pairwise comparisons.
- Improve write-policy robustness analysis (why no-write can be stronger).
- Include polished figures and full reproducibility appendix with hardware/runtime details.
