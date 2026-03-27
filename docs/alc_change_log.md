# ALC Change Log

## Pass: mathematical correctness + numerical stability hardening

### `train_gpt2.c`
- Added explicit ALC runtime invariant helpers and `[ALC]` failure paths for:
  - config/runtime mode/range checks
  - selected slot bounds checks in forward/write/backward
  - finite checks on routing scores, gate logits, fused hidden, write projections, slot/key updates, and loaded persisted tensors.
- Tightened forward math and stability behavior:
  - routing score scaling by `1/sqrt(key_dim)`
  - gate-logit clamp to `[-40, 40]` in gated fusion
  - additive fusion norm-ratio guard to prevent runaway retrieved contribution
- Tightened write/update stability behavior:
  - high-cap write-vector and slot/key norm guards (`1e6`) to avoid catastrophic drift.
- Updated backward gated derivative path to match forward clamp semantics.

### `dev/test/alc_hardening.c` (new)
- Added artifact-independent hardening suite covering:
  - finite-difference gradient checks (representative subset) for `slot_to_hidden`, `gate_h`, `gate_a`, `gate_b`
  - analytic gradient finiteness + non-zero gate-gradient presence checks
  - EMA write semantics for `eta=0`, `eta=1`, and repeated small-eta convergence
  - non-selected slot immutability assertions
  - persistence save/load identity into a fresh model with exact tensor/logit comparisons
  - long-run stress loop with norm/gate/index/hit-histogram monitoring and explicit thresholds.

### `docs/adaptive_learning_component.md`
- Reworked as a correctness/stability design contract with sections for:
  - Mathematical correctness audit (implemented equations only)
  - Numerical stability decisions with rationale
  - Gradient-check coverage and tolerances
  - EMA stability analysis
  - Persistence test contract
  - Long-run stress summary and thresholds
  - Proven invariants and remaining mathematical limitations.

### `README.md`
- Added artifact-independent hardening test build/run commands and clarified what the hardening suite proves.
