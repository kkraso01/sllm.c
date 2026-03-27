# ALC Change Log

## Pass: Native-component completion (serialization, semantics, observability, tiny e2e)

### `train_gpt2.c`
- Added explicit ALC sidecar persistence:
  - `gpt2_save_alc_state(...)`
  - `gpt2_load_alc_state(...)`
  - binary header/version + strict compatibility checks (`C/S/D/K/fusion/update`).
- Added clearer ALC config validation with human-readable failure messages (replacing assert-only behavior for ALC config errors).
- Improved ALC init lifecycle:
  - supports scratch buffer resize when `B*T` changes
  - added slot hit histogram allocation for observability.
- Added ALC observability counters + debug summaries gated by `LLMC_ALC_DEBUG=1`.
- Added runtime sidecar wiring in `main`:
  - load via `LLMC_ALC_STATE_IN`
  - save via `LLMC_ALC_STATE_OUT`.
- Added synthetic-model constructor `gpt2_build_from_synthetic(...)` for artifact-independent model instantiation use cases.
- Clarified/logged fusion/update mode names in enable banner.
- Added explicit block-level counters to show how often ALC is considered/applied.

### `dev/test/tiny_alc_e2e.c`
- Added new artifact-independent tiny end-to-end executable that:
  - builds a synthetic GPT-2 model,
  - runs baseline forward/backward/update,
  - runs ALC-enabled forward/backward/update through the real block path,
  - validates ALC sidecar save/load round trip.

### `docs/adaptive_learning_component.md`
- Added required **Native-component completeness audit** section.
- Added explicit **ALC training semantics** section (hybrid gradient + EMA semantics).
- Documented exact ALC serialization scope, compatibility checks, and env usage.
- Documented debug observability controls and emitted metrics.
- Added artifact-independent validation commands and updated run guidance.
- Clarified future block ordering for FFN/MoE/Engram/ALC coexistence.
