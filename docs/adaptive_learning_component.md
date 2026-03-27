# Adaptive Learning Component (ALC) in `llm.c` CPU GPT-2 Path

This document describes the **implemented** ALC math and runtime behavior in `train_gpt2.c` after the differentiable routing redesign.

## Differentiable routing redesign

### Old design (replaced)

Previous routing was hard top-1:

- `q = W_q h`
- `s_j = (q · k_j) / sqrt(d_k)`
- `j* = argmax_j s_j`
- `a = v_{j*}`

This was non-differentiable in slot selection and blocked end-to-end gradient flow to `query_proj` through routing.

### New design (implemented)

ALC now supports configurable routing modes:

- `hard_top1` (legacy fallback)
- `softmax`
- `topk_softmax` (**default**)

Forward read path now uses:

- `q = W_q h`
- `s_j = (q · k_j) / sqrt(d_k)`
- stable logits: subtract `s_max` before exponentiation
- for soft modes:
  - full softmax: `p_j = exp((s_j - s_max)/tau) / sum_l exp((s_l - s_max)/tau)`
  - top-k softmax: same normalization but only over `TopK(s, k)` and `p_j = 0` outside the active set
- adaptive memory read:
  - `a = sum_j p_j v_j`
- projection to hidden:
  - `a_h = W_s a`

Fusion is unchanged structurally:

- additive: `h' = h + alpha * a_h` (with existing norm-ratio guard)
- gated:
  - `z = W_h ⊙ h + W_a ⊙ a_h + b`
  - `g = sigmoid(clamp(z, -40, 40))`
  - `h' = g ⊙ h + (1-g) ⊙ a_h`

### Why this is more correct

Soft routing is a standard differentiable relaxation of argmax selection, so `query_proj` now receives read-path gradients (for softmax/topk_softmax modes).

## Training semantics (implemented)

### Gradient-trained tensors

Updated by `alc_adamw_update` in `gpt2_update`:

- `query_proj` `(K, C)` when routing mode is `softmax` or `topk_softmax`
- `slot_to_hidden` `(C, D)`
- `gate_h`, `gate_a`, `gate_b` `(C)` in gated fusion mode

### Stateful EMA-updated tensors

Updated in `alc_write_update`:

- `slots` `(S, D)`
- `slot_keys` `(S, K)`

### Not backprop-trained

- `write_proj` `(D, C)` remains state-write-only in this pass

## Read path backward details

For soft routing modes, backward includes:

1. `dL/dW_s` from `a`
2. `dL/da`
3. `dL/dp_j = dL/da · v_j`
4. softmax Jacobian-vector contraction:
   - `dL/ds_j = p_j * (dL/dp_j - sum_l p_l dL/dp_l) / tau`
5. `dL/dq = sum_j (dL/ds_j) * k_j / sqrt(d_k)`
6. `dL/dW_q = (dL/dq) h^T`

Implemented choice for keys is **Option B**:

- slot keys are not optimizer-updated by backprop
- but routing still differentiates through current keys to train `query_proj`

## Write path math (coherent with routing)

Writes are now routing-weighted EMA updates:

- `w = W_w h`
- `v_j <- (1 - eta p_j) v_j + eta p_j w`
- `k_j <- (1 - eta p_j) k_j + eta p_j q`

This is applied for all routed slots (`p_j > 0`), not only a single selected slot.

## Numerical stability decisions

Implemented guards:

- score scale `1/sqrt(d_k)`
- stable softmax with max subtraction
- routing config validation: temperature `> 0`, `topk in [1, S]`, valid routing mode
- gate-logit clamp `[-40, 40]`
- additive fusion norm-ratio guard
- write vector and slot/key norm caps (`1e6`)
- finite checks on routing/fusion/write/load paths

## Persistence contract

ALC sidecar remains `ALC1` (`version=1`) and persists:

1. `query_proj`
2. `write_proj`
3. `slot_to_hidden`
4. `gate_h`
5. `gate_a`
6. `gate_b`
7. `slot_keys`
8. `slots`

## What hard routing limitation is now resolved vs what remains

### Resolved

- Hard argmax read-path non-differentiability is resolved for `softmax` / `topk_softmax`.
- `query_proj` is now trainable through routing in those modes.
- Write semantics now align with routing weights instead of hard one-slot writes.

### Remaining

- `write_proj` remains outside dense backprop (EMA/state update path).
- `slot_keys`/`slots` are stateful EMA memory, not optimizer parameters.
- `hard_top1` mode remains available and non-differentiable by design.

## Validation matrix

```bash
cc -Ofast -fopenmp dev/test/alc_smoke.c -lm -lgomp -o dev/test/alc_smoke
./dev/test/alc_smoke

cc -Ofast -fopenmp dev/test/tiny_alc_e2e.c -lm -lgomp -o dev/test/tiny_alc_e2e
./dev/test/tiny_alc_e2e

cc -Ofast -fopenmp dev/test/alc_hardening.c -lm -lgomp -o dev/test/alc_hardening
./dev/test/alc_hardening
```

Hardening coverage includes:

- routing probability normalization checks
- top-k mask sparsity checks
- non-zero `query_proj` gradient in differentiable routing path
- weighted write-rule correctness checks
- persistence identity checks
- long-run stability stress checks

## Environment variables

- `LLMC_USE_ALC`
- `LLMC_ALC_NUM_SLOTS`
- `LLMC_ALC_SLOT_DIM`
- `LLMC_ALC_KEY_DIM`
- `LLMC_ALC_UPDATE_RATE`
- `LLMC_ALC_FUSION_MODE` (`0` additive, `1` gated)
- `LLMC_ALC_UPDATE_MODE` (`0` off, `1` train_only, `2` always)
- `LLMC_ALC_ROUTING_MODE` (`0` hard_top1, `1` softmax, `2` topk_softmax)
- `LLMC_ALC_TOPK`
- `LLMC_ALC_TEMPERATURE`
- `LLMC_ALC_APPLY_EVERY_N_LAYERS`
- `LLMC_ALC_ADDITIVE_SCALE`
- `LLMC_ALC_STATE_IN`
- `LLMC_ALC_STATE_OUT`
- `LLMC_ALC_DEBUG`
