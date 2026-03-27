# ALC Change Log

## Pass: Training integration + hardening + merge-readiness

### `train_gpt2.c`
- Added real ALC gradient integration for differentiable fusion tensors:
  - per-layer ALC traces (`hidden_pre_layers`, `retrieved_layers`, `selected_slots_layers`)
  - backward hook `alc_backward_fuse_and_accumulate(...)`
  - ALC-local grad buffers + zeroing + AdamW optimizer path (`alc_ensure_grad_buffers`, `alc_zero_param_grads`, `alc_adamw_update`)
  - integrated into main lifecycle (`gpt2_zero_grad`, `gpt2_backward`, `gpt2_update`).
- Explicitly retained hybrid boundary:
  - `slot_to_hidden` and gated fusion weights are gradient-trained.
  - `slots`/`slot_keys` remain EMA memory state.
  - `query_proj`/`write_proj` remain non-backprop in this pass.
- Strengthened startup observability with explicit `[ALC] training semantics: ...` banner.
- Hardened sidecar persistence contract:
  - extended header with `endian_marker` and `header_bytes`
  - strict load validation for endian/header-size compatibility
  - trailing-bytes rejection to avoid partial/extra payload ambiguity.

### `dev/test/tiny_alc_e2e.c`
- Strengthened tiny artifact-independent integration test to assert:
  - ALC-off vs ALC-on logits differ under same synthetic setup.
  - ALC write/update changes subsequent behavior deterministically.
  - ALC gradient tensors are allocated and receive non-zero gradient signal.
  - ALC gradient-trained parameters update after optimizer step.
  - GPT core parameters still update in normal path.
  - Sidecar save/load round-trip restores tensors exactly.
  - Forward pass succeeds after reload.

### `docs/adaptive_learning_component.md`
- Added explicit gradient-feasibility audit section and final path choice.
- Documented exact training semantics split (gradient vs EMA vs non-backprop).
- Added explicit persistence contract details (header checks + exact tensor ordering).
- Consolidated all ALC env vars into one place.
- Expanded validation matrix and future-work notes.

### `README.md`
- Updated ALC section with explicit training behavior split.
- Consolidated env vars and clarified state sidecar behavior.
- Updated tiny validation command notes to reflect stronger integration checks.
