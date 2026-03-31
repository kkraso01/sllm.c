# Submission Variant Plan

## 1) Canonical content from `paper/`

The canonical technical core is the current `paper/` package:

- Core method and mechanism definition in `paper/sections/method.tex`.
- Experimental protocol and artifact mapping in `paper/sections/experimental_setup.tex`.
- Primary evidence tables for SessionKV-DR (`paper/tables/sessionkv_*.tex`) and supporting TinyQA (`paper/tables/tinyqa_results.tex`).
- Claims discipline and evidence boundaries already present in `paper/sections/limitations.tex` and `paper/sections/conclusion.tex`.
- Reproducibility artifacts under `paper/results/` and generation entrypoint `./scripts/run_alc_publication_suite.sh`.

## 2) Shared content across venue versions

Shared across all submission folders:

- Same ALC mechanism description and equations (method section unchanged in substance).
- Same experiment definitions and reported metrics/tables from the existing artifacts.
- Same references file (`references.bib`) and table assets.
- Same central contribution statement: structured in-model adaptive memory with inference-time updates and mechanism-level evidence on delayed recall, persistence, and bounded adaptation behavior.
- Same conservative baseline framing: baseline vs ALC no-write vs full ALC.

## 3) Content adapted per venue

Adapted for each venue package:

- Title and abstract emphasis.
- Introduction framing (mechanism novelty vs empirical/systems polish vs main-track caution).
- Results framing language (what is primary evidence and how strongly to interpret it).
- Related-work emphasis and novelty boundary statements.
- Limitations and conclusion strength calibrated to venue expectations.
- Appendix depth (expanded most in arXiv package).
- README and VENUE_NOTES with venue-specific evidence mapping and risk disclosure.

## 4) Safe claims across all versions

Safe, consistently supported claims:

- ALC supports stable inference-time state updates without changing base transformer weights.
- In this controlled setup, ALC improves delayed session-specific recall on SessionKV-DR relative to baseline controls.
- ALC supports overwrite/update and save/reload persistence behavior in the provided implementation.
- Evidence is mechanism-level with TinyQA retained as supporting semi-realistic evidence.

## 5) Claims to soften in main-track versions

For ICLR/NeurIPS main-track packages, soften or avoid:

- Any implication of broad downstream NLP gains.
- Any suggestion of lifelong-learning resolution.
- Any claim of universal superiority over retrieval, external memory, or weight-update test-time adaptation.
- Any overgeneralization from SessionKV-DR/TinyQA to broad long-context competence.

Main-track language should explicitly state current evidence scope and identify required follow-up (larger public benchmarks, stronger external baselines, broader scale tests).

## 6) Appendix movement/expansion for arXiv

The arXiv version should include the most complete appendix by:

- Explicitly listing all core artifacts and where they live.
- Expanding implementation details for read/write/persistence paths.
- Detailing multi-seed aggregation and confidence interval reporting conventions.
- Adding a short reproducibility checklist and known environment constraints (e.g., figure generation warning).
- Keeping all statements tied to generated artifacts, with no new fabricated metrics.
