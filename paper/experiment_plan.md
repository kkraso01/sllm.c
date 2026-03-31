# ALC experiment plan (publication pass)

## 1) Main hypothesis

A structured adaptive memory component (ALC) can improve inference-time adaptation (store/use newly seen information) while remaining numerically stable and controllable, without modifying core transformer weights during deployment.

## 2) Primary claims to support

1. **Adaptation efficacy**: ALC improves delayed recall after support-time insertion vs no-ALC baseline.
2. **Stability**: repeated writes remain bounded and finite (no NaNs, controlled norm growth).
3. **Persistence correctness**: saved ALC state reloads exactly and preserves behavior.
4. **Learned interface value**: training the read/fusion interface improves retrieval quality over untrained interface.
5. **Practical overhead**: ALC incurs measurable but moderate inference overhead.

## 3) Exact experiments

### A. Core adaptation
- Synthetic associative insert-and-recall episodes.
- Support phase writes key/value into ALC state during inference.
- Query phase asks delayed recall by key.
- Metrics: recall MSE, recall accuracy under threshold.

### B. Stability
- 10k repeated random updates.
- Track NaN rate, final slot norm, max slot norm.

### C. Persistence
- Save/load sidecar state after adaptation.
- Compare slot/key tensors and query behavior before/after reload.
- Metrics: max absolute diff over slots and outputs.

### D. Trainability / learned-interface
- Fixed memory bank and supervised query->value mapping.
- Compare untrained interface loss vs loss after interface optimization.

### E. Ablations
- Slot count: {4, 8, 16}
- Update rate: {0.05, 0.2, 0.5}
- Routing mode: {hard-top1, softmax, topk-softmax}
- Fusion mode: {additive, gated}
- Persistence on/off assessed via persistence experiment.

### F. Efficiency
- Mean forward-pass latency (tiny GPT synthetic setup), with/without ALC.
- Report slowdown ratio and extra ALC parameters.

## 4) Baselines

- Plain baseline (no ALC updates / no memory adaptation signal).
- Full ALC.
- ALC variants from ablations (routing/fusion/update-rate/slots).
- Untrained interface variant in trainability experiment.

## 5) Metrics

- Adaptation: recall MSE, recall accuracy.
- Stability: NaN rate, slot norm statistics.
- Persistence: state max-abs-diff, behavior max-abs-diff.
- Trainability: pre/post optimization loss.
- Efficiency: forward ms, slowdown ratio, parameter overhead.

## 6) Planned plots/tables

- Table 1: main results summary (adaptation/stability/persistence/trainability/efficiency).
- Figure 1: core adaptation MSE bar chart (baseline vs ALC).
- Figure 2: inference latency bar chart.
- Appendix table: ablation sweep CSV.

## 7) Compute/runtime notes

- All experiments are synthetic CPU runs from finalized ALC implementation.
- Intended runtime: a few seconds locally.
- Deterministic seed for reproducibility (`srand(42)`).

## 8) Risk notes

- Synthetic tasks validate mechanism but are weaker than standard NLP benchmarks.
- Novelty overlap risk with prior memory-transformer and inference-time adaptation work.
- Current run is single-seed; confidence intervals not yet established.
