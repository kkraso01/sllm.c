# MoE Expert-Local Memory Design (Experimental Path)

## Stage 1 audit of `train_gpt2_moe_experimental.c` (before this training pass)

This audit captures what existed immediately before enabling MoE backward/training:

- **Fail-fast location:** `gpt2_backward` exited when it detected an active MoE layer (`moe_layer_enabled(model, l)`), so no MoE gradients were computed/applied.
- **Forward caches already available:** per-token router logits/probs, selected expert ids, selected expert weights, and per-selected-expert memory routing probs were already stored in `MoEState` scratch buffers.
- **Missing gradient ownership:** MoE parameters (`router_*`, `expert_fc*`) had no dedicated gradient buffers or optimizer moments.
- **Optimizer gap:** `gpt2_update` updated dense GPT-2 params (+ optional ALC tensors), but not MoE router/expert tensors.

The goal of this pass was to replace that fail-fast with minimum-correct top-2 MoE training support.

---

## Preserved forward architecture

For MoE-enabled layers:

1. LN1 -> attention -> residual
2. LN2
3. Router logits over experts
4. Top-k selected experts (`k=2` default)
5. For each selected expert: FFN -> expert-local memory read/write -> fuse
6. Weighted combine of selected expert outputs
7. Residual add
8. Optional global ALC path remains separate (unchanged)

No architectural expansion was added here.

---

## Backward math now implemented

### 1) Weighted top-2 combine backward

Forward:
\[
o = \sum_{i\in\mathcal{E}} \alpha_i o_i
\]

Backward:
\[
\frac{\partial L}{\partial o_i} = \alpha_i\frac{\partial L}{\partial o}
\]
\[
\frac{\partial L}{\partial \alpha_i} = \left\langle \frac{\partial L}{\partial o}, o_i \right\rangle
\]

Implemented explicitly for selected experts only.

### 2) Selected-softmax router backward

Selected router weights:
\[
\alpha_i = \frac{\exp(r_i/\tau)}{\sum_{j\in\mathcal{E}}\exp(r_j/\tau)}
\]

For selected experts:
\[
\frac{\partial L}{\partial r_i} = \alpha_i\left(\frac{\partial L}{\partial \alpha_i} - \sum_{j\in\mathcal{E}}\alpha_j\frac{\partial L}{\partial \alpha_j}\right)\frac{1}{\tau}
\]

Then router param grads:
\[
\frac{\partial L}{\partial w_i^{router}} = \frac{\partial L}{\partial r_i} h,
\quad
\frac{\partial L}{\partial b_i^{router}} = \frac{\partial L}{\partial r_i}
\]

### 3) Selected expert FFN backward

For each selected expert only, gradients are computed for:

- `expert_fcw`
- `expert_fcb`
- `expert_fcprojw`
- `expert_fcprojb`

Non-selected experts receive zero MoE FFN gradients for that token.

### 4) Memory-read gradient participation

Memory writes remain **stateful EMA updates** (not differentiated through write updates).

Memory **read** contribution in expert output still contributes to gradient flow into selected expert `u_i` through local query->memory-attention->read path (with memory state treated as fixed state for this step).

---

## Trainable now vs stateful/fixed now

### Trainable in this pass

- Router params: `router_w`, `router_b`
- Selected expert FFN params: `expert_fcw`, `expert_fcb`, `expert_fcprojw`, `expert_fcprojb`

These now have:

- dedicated gradient buffers
- AdamW moment buffers
- update integration in `gpt2_update`

### Stateful EMA (not backpropagated through write updates)

- `expert_memory`
- `expert_memory_keys`

### Fixed (this pass)

- `memory_query_proj`, `memory_write_proj`, `memory_slot_to_hidden` remain unchanged by optimizer in this milestone.

---

## Why training support comes before load balancing

Load balancing is only meaningful once the routed path is demonstrably learnable.

This pass establishes that:

- selected experts receive gradients and updates,
- non-selected experts remain isolated,
- router parameters receive gradients and update,
- forward behavior changes after optimizer steps.

With those fundamentals in place, auxiliary balancing losses can be added on a correct training substrate instead of masking routing/backward bugs.

---

## Current status (implemented now)

- MoE-enabled layers no longer fail-fast in `gpt2_backward`.
- Top-2 weighted-combine backward is implemented.
- Selected-softmax router backward is implemented.
- Selected expert FFN backward is implemented with isolation.
- MoE router + expert FFN params are integrated into AdamW updates.
- Artifact-independent tests now cover gradient/update isolation and post-step behavior change.

## Deferred (explicitly not in this pass)

- Load-balancing / auxiliary routing loss
- Distributed MoE
- Production checkpointing of MoE tensors
- Pulling global ALC into MoE internals
- Full differentiation through stateful EMA writes

---

## Experimental test build/run commands

```bash
gcc -Ofast -g -I. dev/test/moe_expert_memory.c -lm -o dev/test/moe_expert_memory
./dev/test/moe_expert_memory
```
