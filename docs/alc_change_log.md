# ALC Change Log

## 2026-03-27 — Differentiable routing redesign

### `train_gpt2.c`

- Replaced hard argmax-only routing with configurable routing modes:
  - `hard_top1`
  - `softmax`
  - `topk_softmax` (default)
- Added new ALC config fields and validation:
  - `alc_routing_mode`
  - `alc_topk`
  - `alc_temperature`
- Implemented numerically stable routing softmax (`max` subtraction + temperature).
- Replaced single-slot retrieval with routing-weighted retrieval:
  - `a = sum_j p_j v_j`
- Added per-token routing probability traces for backward and write paths.
- Updated ALC backward path to propagate differentiable routing gradients into `query_proj` for soft routing modes.
- Added optimizer/gradient buffers and AdamW updates for `query_proj` (soft routing modes).
- Reworked write semantics from hard selected-slot EMA to routing-weighted EMA updates for slots and keys.
- Preserved existing fusion behavior and safety guards (gate clamp, finite checks, norm caps).

### `dev/test/alc_smoke.c`

- Added routing normalization and top-k activity checks (`sum(p)=1`, active slots `<= topk`).

### `dev/test/tiny_alc_e2e.c`

- Switched tiny e2e ALC configuration to differentiable top-k routing.
- Added assertion that `query_proj` receives non-zero gradient signal.

### `dev/test/alc_hardening.c`

- Added explicit routing normalization and top-k masking assertions.
- Added non-zero/finite `query_proj` gradient checks.
- Updated EMA write-rule tests to verify routing-weighted write math.

### `docs/adaptive_learning_component.md`

- Added full redesign section:
  - old hard routing vs new differentiable routing math
  - read/write equations now implemented
  - differentiable vs stateful tensor semantics
  - resolved hard-routing limitation and remaining constraints

### `README.md`

- Updated ALC section to describe differentiable routing modes and new env vars.
- Updated training semantics to mark `query_proj` as gradient-trained in soft routing modes.
