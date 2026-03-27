# ALC Change Log

## 2026-03-27

## Audit summary (follow-up pass)

### What was already implemented before this pass
- ALC existed as a native component in `train_gpt2.c` with config/state structs, lazy initialization, forward read/fuse, optional write path, and feature gating.
- Documentation and README sections existed.

### What was incomplete before this pass
- `alc_slot_dim` field was missing from config despite being required by the integration checklist.
- Slot vectors were effectively tied to hidden dimension; no explicit slot-dim decoupling/projection.
- No checkpoint-independent smoke command existed to exercise ALC read/write behavior.
- Docs needed synchronization to reflect final config and equations.

### `train_gpt2.c`
- **Purpose:** Complete ALC config and state integration end-to-end, including explicit slot dimension support.
- **Symbols added/changed:**
  - Changed `ALCConfig`: added `alc_slot_dim`
  - Changed `ALCState`: `write_proj` now `(D, C)`, added `slot_to_hidden` `(C, D)`, `slots` now `(S, D)`
  - Changed validation: `gpt2_validate_alc_config` now checks `alc_slot_dim > 0`
  - Changed init: alloc/init for `slot_to_hidden`; shape updates for `write_proj` and `slots`
  - Changed read/fuse path: retrieved slot now projected through `slot_to_hidden` before fusion
  - Changed write path: EMA updates write into slot dimension `D`
  - Changed defaults/env/logs/free: wired `alc_slot_dim` through defaults, env var parsing, enable log, and cleanup
- **Behavior change under `use_alc`:**
  - Baseline behavior unchanged when `use_alc=0`.
  - Under `use_alc=1`, slot memory now uses configurable dimension `D`, with explicit projection back to hidden dim.

### `dev/test/alc_smoke.c`
- **Purpose:** Add minimal smoke-testable ALC path independent of checkpoint/tokenizer assets.
- **Symbols added/changed:**
  - Added standalone smoke test covering `gpt2_init_alc_state`, `alc_forward_read_and_fuse`, `alc_write_update`, and assertions on slot selection/update.
- **Behavior change under `use_alc`:** Test-only file; no runtime change to main training binary.

### `docs/adaptive_learning_component.md`
- **Purpose:** Add audit section and synchronize design doc with final implementation.
- **Symbols added/changed:**
  - Added audit findings/resolutions
  - Added `alc_slot_dim` to config section
  - Updated equations to include slot projection `a_h = W_s a`
  - Added smoke-test compile/run commands
- **Behavior change under `use_alc`:** Documentation only.

### `docs/alc_change_log.md`
- **Purpose:** Record audit and all follow-up completion changes.
- **Symbols added/changed:** N/A (documentation file).
- **Behavior change under `use_alc`:** Documentation only.

### `README.md`
- **Purpose:** Keep user-facing env-var list aligned with implemented config fields.
- **Symbols added/changed:** Added `LLMC_ALC_SLOT_DIM` mention.
- **Behavior change under `use_alc`:** Documentation only; runtime defaults unchanged.
