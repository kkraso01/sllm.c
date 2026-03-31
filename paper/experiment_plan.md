# ALC experiment plan (final strengthening pass)

## Scope guard

This pass is for **pre-submission strengthening only**. The core ALC mechanism is fixed; changes focus on evidence quality, robustness, and packaging.

## Main hypothesis

ALC can provide useful inference-time adaptation with controllable mutable state and persistence semantics, while remaining numerically stable under repeated updates.

## Strengthening priorities executed

1. Multi-seed robustness (5 seeds)
2. One stronger semi-realistic benchmark
3. Stronger baseline comparison beyond plain baseline
4. Paper/table/appendix tightening

## Experiments

### A) Core adaptation (multi-seed)
- Delayed insert-and-recall episodes.
- Compare:
  - baseline (query before writing current fact)
  - ALC full
  - ALC no-write (read active, writes off)
- Metrics: recall MSE, threshold recall accuracy.

### B) Stability (multi-seed)
- 10k repeated updates per seed.
- Metrics: NaN rate, slot norm init/final/max.

### C) Persistence (reference run)
- Save/load sidecar state after adaptation.
- Metrics: save/load success, slot max diff, behavior max diff.

### D) Trainability (multi-seed)
- Fixed memory bank; optimize read interface only.
- Metrics: loss before vs after training.

### E) Language-shaped delayed recall benchmark (multi-seed)
- Fact insertion + distractor updates + delayed query.
- Synthetic but language-structured key/value setup.
- Same baseline triplet as core adaptation.

### F) Ablations + efficiency
- Existing ablation sweep retained (single reference run).
- Efficiency retained in tiny CPU setup (single reference run).

## Outputs

- `paper/results/metrics.csv`
- `paper/results/summary.json`
- `paper/results/multiseed_raw.csv`
- `paper/results/multiseed_summary.csv`
- `paper/results/multiseed_summary.json`
- `paper/tables/main_results.md`
- `paper/tables/main_results.tex`
- `paper/tables/ablation_results.csv`
- `paper/tables/ablation_top10.tex`

## Claim discipline

- Primary claims are mechanism-level.
- Explicitly avoid claims of broad lifelong-memory or long-context superiority.
- Explicitly report where no-write baseline remains competitive/stronger.
