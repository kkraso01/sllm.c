# Final strengthening plan (pre-submission pass)

Date: 2026-03-31

## 1) Claims already supported by existing package

Supported at mechanism level by existing artifacts:
- ALC improves delayed synthetic recall over a plain no-memory baseline in the current core task.
- ALC state updates remained numerically stable in a long synthetic update run (no NaN events observed in the current single-seed run).
- ALC persistence save/load round-trip preserved state and behavior in the tested setup.
- Trainable read/fusion interface can reduce supervised retrieval loss versus its initial setting.
- Runtime overhead is measurable but moderate in the tiny CPU profile.

## 2) Claims currently weak / under-supported

Main weaknesses before this pass:
- Most results are single-seed and lack uncertainty reporting.
- Evidence is dominated by synthetic mechanism probes with limited language-shaped evaluation.
- Baselines are limited (mostly plain transformer / no-memory).
- Figure production is fragile and currently under-polished.
- Narrative still needs tighter evidence-to-claim discipline for submission.

## 3) Exact experiments to run in this strengthening pass

Highest-priority additions (in execution order):

1. **Multi-seed robustness (mandatory)**
   - Re-run core adaptation, stability summary, and trainability with 5 seeds (fallback to 3 if runtime issues).
   - Aggregate mean, standard deviation, and 95% confidence interval where practical.
   - Export:
     - `paper/results/multiseed_summary.csv`
     - `paper/results/multiseed_summary.json`

2. **One stronger semi-realistic benchmark (mandatory)**
   - Add a language-shaped delayed fact insertion/retrieval task with distractors and temporal gap.
   - Compare at least:
     - plain baseline
     - full ALC
     - stronger non-trivial baseline (no-write memory variant)
   - Report explicit metric (MSE and threshold accuracy).

3. **Stronger baseline comparison (mandatory)**
   - Add `ALC no-write` baseline (read path active, writes disabled) to isolate the value of online writes.
   - Keep baseline implementation simple and reproducible.

4. **Key ablation robustness (optional-if-feasible)**
   - Multi-seed a small subset of key ablations (not full combinatorial sweep).

5. **Figure/table/paper polish (mandatory)**
   - Update main and ablation tables with multi-seed statistics where available.
   - Generate at least one main figure; if plotting unavailable, provide clear fallback artifacts.
   - Tighten abstract/intro/results/limitations/conclusion language to conservative scope.

## 4) Mandatory vs optional additions

### Mandatory for this pass
- Multi-seed aggregation for core adaptation, stability summary, trainability.
- One stronger semi-realistic benchmark with baseline vs ALC.
- One stronger baseline beyond plain transformer.
- Updated paper text and appendix for reproducibility and claims discipline.
- Final submission checklist mapping claims to concrete evidence.

### Optional if runtime allows
- Multi-seed on selected ablations.
- Additional polished figures beyond one core plot.
- Expanded stress-test length beyond prior 10k setting.

## 5) Out of scope for this submission

Explicitly out of scope in this pass:
- New core ALC architecture features.
- Large external benchmark suites requiring major data/infra buildout.
- Large-scale model training or SOTA comparative campaigns.
- Claims of broad lifelong memory or general long-context superiority.

## 6) Success criteria for this pass

- Reproducible multi-seed artifacts exist and are consumed by paper tables.
- Stronger benchmark + stronger baseline are implemented and reported.
- Paper claims are narrowed to what is directly supported.
- Submission checklist clearly states supported evidence and residual risks.
