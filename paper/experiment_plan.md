# ALC experiment plan (aligned benchmark pass)

## Scope guard

This pass keeps the ALC mechanism fixed and aligns evaluation to the actual contribution: **structured inference-time adaptive memory**.

## Main hypothesis

ALC can write session-specific information during inference, retain it through delay/interference, retrieve it later, update stale entries, and preserve memory state across save/reload.

## Benchmark structure

### Primary evidence: SessionKV-DR (Session Key-Value Delayed Recall)

Variants for every primary result:
- baseline
- alc_no_write
- alc

Sub-evaluations:
1. Delayed key-value recall sweep (delay and inserted fact count)
2. Overwrite/update recall (+ stale confusion)
3. Persistence across runs (save/reload recall + state/behavior equality)
4. Capacity/interference sweep (facts, delays, distractors)

### Supporting evidence

- TinyQA (semi-realistic support; not the primary claim test)
- Existing stability/trainability/efficiency/ablation checks retained

## Metrics

Primary metrics:
- exact delayed recall accuracy
- overwrite accuracy
- stale-value confusion rate
- persistence recall accuracy
- accuracy vs delay / fact count / distractor load

## Outputs

- `paper/results/sessionkv_results.csv`
- `paper/results/sessionkv_summary.json`
- `paper/tables/sessionkv_main.tex`
- `paper/tables/sessionkv_overwrite.tex`
- `paper/tables/sessionkv_capacity.tex`
- Existing `paper/results/multiseed_summary.csv` for aggregate stats and CIs

## Multi-seed policy

5 seeds (42, 1337, 2027, 31415, 27182) for SessionKV-DR and TinyQA metrics that appear in primary tables.

## Claim discipline

Use conservative claims:
- improves delayed session-specific recall under inference-time updates
- supports adaptive overwrite and persistence in structured in-model memory
- mechanism-level evidence only (not lifelong-memory claims)
