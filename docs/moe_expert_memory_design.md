# MoE Expert-Local Memory Design (Experimental Path)

## Stage 1 audit of router training path for load balancing (this pass)

Audit findings in `train_gpt2_moe_experimental.c` before this load-balancing milestone:

- **Router gradient path existed only through selected-topk weights.** `moe_backward_topk_expert_local` already propagated `dL/dlogit` from weighted expert output combine, but no auxiliary balancing term was present.
- **Routing scratch was sufficient for insertion.** Per-token full router probabilities (`router_probs`) and selected experts (`selected_expert`) were already materialized in forward and reused in backward.
- **Clean collection point:** usage statistics can be collected directly in `moe_forward_topk_expert_local` immediately after router softmax and top-k selection, without changing top-2 semantics.
- **Minimal-disruption insertion point:** auxiliary loss gradients can be added in `moe_backward_topk_expert_local` as an extra `dlogit` contribution from full softmax probabilities, then accumulated into `d_router_w/d_router_b` exactly where router grads are already handled.

This pass implements that plan end-to-end.

---

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

## Load-balancing math implemented in this pass

Per MoE-enabled layer, over active MoE tokens \(N\) and experts \(E\):

\[
\text{importance}_e=\frac{1}{N}\sum_{t=1}^{N} p_{t,e}
\]
\[
\text{load}_e=\frac{1}{N}\sum_{t=1}^{N}\mathbf{1}[e\in\text{TopK}(t)]
\]
\[
L_{\text{balance,layer}}=\lambda\cdot E\sum_{e=1}^{E}\text{importance}_e\cdot\text{load}_e
\]

Implementation details:

- `importance` is computed from full router softmax probabilities (`router_probs`).
- `load` is computed from actual top-k selections (`selected_expert`), so it reflects discrete traffic.
- Layer losses are summed into `model->moe_aux_loss`, and `model->mean_loss = cross_entropy_mean + moe_aux_loss` when targets are present.

### Auxiliary gradient path into router logits

`load` is treated as fixed statistics for the current batch/layer (standard MoE practice with non-differentiable top-k indicators). Thus:

\[
\frac{\partial L_{\text{balance}}}{\partial p_{t,e}}=\lambda\cdot E\cdot \frac{\text{load}_e}{N}
\]

With router temperature \(\tau\), logits \(z\), and softmax \(p\):

\[
\frac{\partial L}{\partial z_{t,e}}=\frac{1}{\tau}p_{t,e}\left(g_e-\sum_j p_{t,j}g_j\right)
\]

where \(g_e=\lambda\cdot E\cdot \frac{\text{load}_e}{N}\).

This `dlogit` is added to the existing router gradient path in `moe_backward_topk_expert_local`, and therefore updates `router_w/router_b` via the existing AdamW integration.

## Stats and observability added

`MoEState` now keeps per-layer batch stats:

- `layer_importance_sum[L,E]`
- `layer_load_count[L,E]`
- `layer_active_tokens[L]`
- `layer_balance_loss[L]`

When `LLMC_MOE_DEBUG=1`, `[MOE]` lines report per-layer:

- active token count,
- tokens per expert,
- average importance and load per expert,
- number of zero-traffic experts,
- balancing loss value.

## Config knobs and defaults

- `moe_load_balance_coef` (default `0.01`) and env override `LLMC_MOE_LOAD_BALANCE_COEF`.
- `moe_load_balance_coef=0` disables auxiliary balancing.
- Auxiliary loss remains zero when MoE is disabled.

## Why load balancing is the next step after top-2 training

Now that top-2 routing, selected-expert training, and router updates are wired correctly, expert-collapse is the dominant failure mode. Load balancing is the minimal next mechanism that directly regularizes router usage without redesigning the MoE architecture.

## Limitations

- `load` uses hard top-k indicators; gradients from the balancing term flow only through `importance` (softmax probabilities), not through discrete selection decisions.
- Balancing is currently global over active tokens in each MoE-enabled layer and does not include per-sequence or per-head constraints.
- No distributed/global-expert balancing is included.
- Stateful EMA memory writes remain non-differentiable by design.

---

## Experimental test build/run commands

```bash
gcc -Ofast -g -I. dev/test/moe_expert_memory.c -lm -o dev/test/moe_expert_memory
./dev/test/moe_expert_memory
```
