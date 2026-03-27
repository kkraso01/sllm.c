# Adaptive Learning Component (ALC) in `llm.c` CPU GPT-2 Path

This document records the **implemented** ALC behavior in `train_gpt2.c`, with an explicit correctness/stability audit and test contracts.

## Training semantics (implemented)

Gradient-trained tensors (updated by `alc_adamw_update` in `gpt2_update`):
- `slot_to_hidden` `(C, D)`
- `gate_h`, `gate_a`, `gate_b` `(C)` when `alc_fusion_mode=gated`

State-updated tensors (updated by EMA write rule in `alc_write_update`):
- `slots` `(S, D)`
- `slot_keys` `(S, K)`

Not gradient-trained in this pass:
- `query_proj` `(K, C)`
- `write_proj` `(D, C)`

## Mathematical correctness audit

Notation (for a token-position `bt`):
- hidden input: `h_in ∈ R^C`
- query projection: `W_q = query_proj ∈ R^{K×C}`
- keys: `K_s = slot_keys[s] ∈ R^K`
- slot memory: `m_s = slots[s] ∈ R^D`
- slot-to-hidden map: `W_sh = slot_to_hidden ∈ R^{C×D}`
- write projection: `W_w = write_proj ∈ R^{D×C}`
- update rate: `η = alc_update_rate ∈ [0,1]`

### Forward equations: routing and retrieval

1. Query:

`q = W_q h_in`

2. Routing score per slot (implemented with scale):

`score_s = (q · K_s) / sqrt(K)`

3. Slot selection:

`s* = argmax_s score_s`

4. Retrieved hidden vector:

`r = W_sh m_{s*}`

### Forward equations: fusion modes

#### Additive mode (`alc_fusion_mode=additive`)

Implemented base equation:

`h_out = h_in + α r`, where `α = alc_additive_scale`

Implemented stability guard:
- if `||r||_2 > 100 * (||h_in||_2 + 1e-6)`, scale `α` down so that the effective retrieved norm ratio is capped at 100.

#### Gated mode (`alc_fusion_mode=gated`)

Per channel `c`:

`z_c_raw = gate_h[c] * h_in[c] + gate_a[c] * r[c] + gate_b[c]`

`z_c = clamp(z_c_raw, -40, 40)`

`g_c = sigmoid(z_c)`

`h_out[c] = g_c * h_in[c] + (1 - g_c) * r[c]`

### Write/update equations

Writes happen only when the layer is ALC-enabled **and** update-mode policy allows writes:
- `update_mode=off`: never
- `update_mode=train_only`: only when `targets != NULL` (training forward)
- `update_mode=always`: always

Within `alc_write_update`, for each active token `bt` with selected slot `s*`:

`w = W_w h_out`

`m_{s*} <- (1-η) m_{s*} + η w`

`K_{s*} <- (1-η) K_{s*} + η q`

Non-selected slots/keys are unchanged for that token update.

Additional safety caps:
- write vector magnitude soft-capped (very loose cap: `1e6`)
- post-update slot/key vector norms capped at `1e6`

### Backward equations for gradient-trained tensors

Implemented in `alc_backward_fuse_and_accumulate` using cached per-layer traces (`h_in`, `r`, selected slot id):

- additive path: `∂L/∂r = α * ∂L/∂h_out`
- gated path uses channelwise chain rule from
  `h_out = g*h_in + (1-g)*r`, `g=sigmoid(clamp(z_raw,-40,40))`
- for clamp saturation (`|z_raw| > 40`), derivative through clamp is zero
- `slot_to_hidden` gradient accumulates as outer product with selected slot state

### Correctness reconciliation notes

The design doc now matches implementation details that were previously implicit:
- routing includes `/sqrt(K)` scaling
- gated forward/backward both include gate-logit clamp at `[-40, 40]`
- additive path includes explicit norm-ratio safety scaling
- write updates include explicit high-cap norm guards

## Numerical stability decisions

1. **Routing score scaling** (`/sqrt(K)`):
   - keeps score magnitude stable as key dimension grows.

2. **Gate logit clamp** (`[-40,40]`) before sigmoid:
   - avoids exp overflow/underflow and stabilizes gradient behavior.

3. **Additive fusion norm-ratio guard**:
   - prevents rare runaway when retrieved vector is disproportionately large vs hidden state.

4. **EMA write/state high-cap norm guards** (`1e6`):
   - fail-safe prevention of long-run catastrophic drift without changing typical behavior.

5. **Finite checks with `[ALC]` errors**:
   - routing scores, gate logits, fused hidden, slot/key updates, and loaded tensors are checked for NaN/Inf.

## Persistence contract

ALC sidecar format is a binary `ALC1` (`version=1`) with strict checks:
- magic/version/endian/header-size
- exact config match (`C,S,D,K,fusion_mode,update_mode`)
- no trailing bytes
- loaded tensor finiteness verification

Serialized tensor order:
1. `query_proj` `(K,C)`
2. `write_proj` `(D,C)`
3. `slot_to_hidden` `(C,D)`
4. `gate_h` `(C)`
5. `gate_a` `(C)`
6. `gate_b` `(C)`
7. `slot_keys` `(S,K)`
8. `slots` `(S,D)`

## Validation matrix

### Existing tiny integration + smoke

```bash
cc -Ofast -fopenmp dev/test/alc_smoke.c -lm -lgomp -o dev/test/alc_smoke
./dev/test/alc_smoke

cc -Ofast -fopenmp dev/test/tiny_alc_e2e.c -lm -lgomp -o dev/test/tiny_alc_e2e
./dev/test/tiny_alc_e2e
```

### Hardening test suite (artifact-independent)

```bash
cc -Ofast -fopenmp dev/test/alc_hardening.c -lm -lgomp -o dev/test/alc_hardening
./dev/test/alc_hardening
```

Coverage in `alc_hardening.c`:

1. **Gradient-check coverage** (representative subset):
   - tensors: `slot_to_hidden`, `gate_h`, `gate_a`, `gate_b`
   - checks finite analytic grads
   - checks non-zero gate gradient presence under synthetic setup
   - checks finite-difference agreement on representative indices
   - tolerances: `abs_tol=2e-3`, `rel_tol=2e-2`

2. **EMA stability/correctness analysis**:
   - `η=0` => no change
   - `η=1` => selected slot/key exact overwrite
   - small `η` repeated writes converge toward write vector
   - non-selected slots remain unchanged

3. **Persistence test contract**:
   - save from model ALC state
   - load into fresh model with matching config
   - assert exact tensor equality for all persisted tensors
   - assert exact logits equality on same input post-load

4. **Long-run stability stress summary**:
   - 800 repeated ALC-enabled forwards with writes
   - tracks max slot norm, max key norm, max fused hidden norm, gate min/max/mean, slot-hit histogram summary
   - fails on NaN/Inf, invalid slot index, or norm threshold breaches
   - thresholds: `1e4` for slot/key/fused norms (far below hard cap, chosen as practical explosion detector)

## Proven invariants

The following are now enforced by code and/or tests:

- Config invariants: valid dimensions, valid fusion/update modes, `alc_update_rate ∈ [0,1]`.
- Runtime invariants: selected slot indices always in `[0, S-1]`.
- Finite-value invariants at key stages (routing, gating, fusion, writes, load).
- EMA semantics invariants for `η ∈ {0,1}` and non-selected slot immutability.
- Persistence invariants: strict contract checks + exact round-trip identity under matching config.
- Long-run boundedness invariants: no NaN/Inf and norms remain under stress thresholds.

## Remaining mathematically known limitations

- Routing is hard argmax and non-differentiable; `query_proj` is intentionally not trained by backprop in this pass.
- Write dynamics are stateful EMA updates external to dense autograd; `write_proj` is not backprop-trained.
- Finite-difference gradient checks cover representative subsets, not every parameter element.
- Additive and write-state guards are high-level safety protections, not formal Lyapunov guarantees.

## Environment variables

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
