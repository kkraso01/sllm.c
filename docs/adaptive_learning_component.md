# Adaptive Learning Component (ALC) in `llm.c` CPU GPT-2 Path

## Native-component completeness audit

### What is fully native today vs partially native today

**Fully native in-model (integrated in runtime lifecycle):**
- Config/env parsing + validation (`gpt2_set_alc_config`, `gpt2_validate_alc_config`).
- Forward-path application in transformer block (`gpt2_forward` -> `alc_forward_read_and_fuse`).
- Stateful write path (`alc_write_update`) with mode control (`off` / `train_only` / `always`).
- Sidecar persistence (`gpt2_save_alc_state`, `gpt2_load_alc_state`).
- Observability/logging (`[ALC]` banner + `[ALC][debug]` summaries).

**Partially native (intentional hybrid semantics):**
- Differentiable ALC fusion tensors are only **partially** gradient-trained (details below).
- Discrete routing and memory-state tensors are still non-backprop stateful memory.

## Training semantics

### Gradient-trained in this pass

These tensors now receive explicit gradients during `gpt2_backward` and are updated in `gpt2_update` via an ALC-local AdamW path that runs in the same step lifecycle as GPT-2 core updates:

- `slot_to_hidden` `(C, D)`
- `gate_h`, `gate_a`, `gate_b` `(C)` when `alc_fusion_mode=gated`

Implementation points:
- Backward hook per layer: `alc_backward_fuse_and_accumulate(...)`.
- Gradient buffer init: `alc_ensure_grad_buffers(...)`.
- Gradient reset: `alc_zero_param_grads(...)` from `gpt2_zero_grad(...)`.
- Optimizer step: `alc_adamw_update(...)` called from `gpt2_update(...)`.

### Stateful/non-gradient tensors

These remain explicit memory/runtime state, not AdamW-trained tensors:
- `slots` `(S, D)`
- `slot_keys` `(S, K)`

They are updated by EMA write-back (`alc_write_update`) according to `alc_update_mode` and `alc_update_rate`.

### Not gradient-trained (by design in this pass)

- `query_proj` `(K, C)`
- `write_proj` `(D, C)`

Reason: current routing uses hard argmax slot selection (`selected_slots`) and state mutation through selected-slot writes. In this CPU reference path, no surrogate gradient / soft routing was introduced, to keep baseline architecture stable and deterministic.

## ALC gradient-integration feasibility audit

### 1) Differentiable path membership (current math)

From `gpt2_forward` block order:

```text
... residual3 -> ALC read/fuse -> next layer / final LN
```

Inside `alc_forward_read_and_fuse`:
- `slot_to_hidden` always participates in differentiable affine map `retrieved = W_slot_to_hidden * slot[selected]`.
- `gate_h/gate_a/gate_b` participate in differentiable gated fusion when gated mode is enabled.
- `query_proj` contributes to query vector for slot scoring/selection (hard argmax), making gradient path through routing non-smooth/discrete.
- `write_proj` is used in post-fuse write update into memory state; effects couple across layers via mutable memory, but through selected slots and state updates, not currently represented in dense autograd buffers.

### 2) Backward support needed

To gradient-train differentiable fusion tensors with minimal intrusion, backward needed:
- Per-layer cached traces of ALC pre-fusion hidden, retrieved vector, and selected slot index.
- Layer-local backward transform from `d(hidden_out)` to `d(hidden_in)` at ALC boundary.
- Parameter gradient accumulation for `slot_to_hidden` and gated weights.

### 3) Minimal invasive path

Implemented path:
- Add ALC-local per-layer trace buffers (outside global `ActivationTensors`, to avoid destabilizing baseline tensor layout).
- Inject a single backward call before existing `residual_backward(...)` in each layer.
- Add ALC-local AdamW updates in `gpt2_update(...)` after baseline GPT parameter update loop.

### 4) Feasibility/stability assessment

This is cleanly feasible for `slot_to_hidden` + gated weights, and was implemented.

`query_proj` / `write_proj` full gradient integration would require introducing soft routing and/or surrogate-gradient semantics (or full differentiable memory dynamics) and would be materially more invasive for this pass.

## Final chosen training design

**Chosen path: Path A (partial real gradient integration + explicit hybrid boundary).**

- Real gradients and AdamW updates are now present for the differentiable fusion parameters (`slot_to_hidden`, gated weights).
- Memory-state tensors remain EMA-managed adaptive state.
- Routing/write projections are explicitly documented as non-backprop in this pass.

This avoids the previous ambiguous middle state by making trained-vs-stateful boundaries explicit in code and docs.

## Persistence contract

ALC sidecar format is a binary `ALC1` contract (`version=1`) with strict compatibility checks.

### Header checks

On load, runtime validates:
- `magic == ALC1`
- `version == 1`
- `endian_marker == 0x01020304`
- `header_bytes == sizeof(ALCCheckpointHeader)`
- exact config match for `channels`, `num_slots`, `slot_dim`, `key_dim`, `fusion_mode`, `update_mode`
- no trailing bytes after expected payload

### Serialized tensor order (exact)

1. `query_proj` `(K, C)`
2. `write_proj` `(D, C)`
3. `slot_to_hidden` `(C, D)`
4. `gate_h` `(C)`
5. `gate_a` `(C)`
6. `gate_b` `(C)`
7. `slot_keys` `(S, K)`
8. `slots` `(S, D)`

This sidecar saves both ALC parameters and ALC memory state.

## Validation matrix

### Artifact-independent tests

Build + run tiny integrated path:

```bash
cc -Ofast -fopenmp dev/test/tiny_alc_e2e.c -lm -lgomp -o dev/test/tiny_alc_e2e
./dev/test/tiny_alc_e2e
```

What it validates:
- ALC-off vs ALC-on behavioral divergence (`logits` differ).
- ALC stateful writes change subsequent behavior deterministically.
- ALC fusion parameters receive non-zero gradients and update after optimizer step.
- Core GPT parameters still update in same training loop.
- Save/load round-trip preserves persisted tensors exactly.
- Loaded model can immediately run a successful forward pass.

Additional smoke path:

```bash
cc -Ofast -fopenmp dev/test/alc_smoke.c -lm -lgomp -o dev/test/alc_smoke
./dev/test/alc_smoke
```

## Environment variables (single place)

- `LLMC_USE_ALC`
- `LLMC_ALC_NUM_SLOTS`
- `LLMC_ALC_SLOT_DIM`
- `LLMC_ALC_KEY_DIM`
- `LLMC_ALC_UPDATE_RATE`
- `LLMC_ALC_FUSION_MODE` (`0` additive, `1` gated)
- `LLMC_ALC_UPDATE_MODE` (`0` off, `1` train_only, `2` always)
- `LLMC_ALC_APPLY_EVERY_N_LAYERS`
- `LLMC_ALC_ADDITIVE_SCALE`
- `LLMC_ALC_STATE_IN`
- `LLMC_ALC_STATE_OUT`
- `LLMC_ALC_DEBUG`

## Forward ordering note (for future MoE/Engram)

Current intended extension region ordering remains:

```text
attention -> FFN/MoE -> Engram(optional) -> ALC(optional) -> residual continuation
```

ALC remains the post-FFN/post-Engram fusion point in this CPU path.

## Future work

- Soft routing / temperature-controlled top-k routing for differentiable query/key learning.
- Optional surrogate-gradient mode for hard routing.
- Decision whether to fold selected ALC tensors into checkpoint-native GPT tensor layouts.
- Broader gradient coverage for memory-write dynamics (`write_proj`) once routing semantics are formalized.
