# MoE Expert-Local Memory Design (Experimental Path)

## Stage 1 audit of `train_gpt2_moe_experimental.c` (before this pass)

- Router existed as top-1 over experts, but only as a scaffold.
- FFN compute was still shared dense GPT-2 FFN (`fcw/fcb/fcprojw/fcprojb`) for all tokens.
- MoE memory state had per-expert slots/keys, but memory read used post-dense FFN output and did not include expert-owned query/write/slot-to-hidden projections.
- Forward integration called scaffold memory after dense FFN, not true `router -> expert_i FFN -> expert_i memory` ownership.
- Backward path was still dense-only (no expert FFN gradients).

This milestone upgrades the experimental path to real top-1 routed expert FFN + expert-local memory.

## Exact block placement (implemented now)

Within MoE-enabled transformer layers in the experimental file:

1. LN1 -> attention -> residual (unchanged)
2. LN2
3. Router (top-1 expert per token)
4. Selected expert FFN (`FFN_i`)
5. Selected expert local memory read + write (`Memory_i`)
6. Fuse `u_i + scale * m_i^h`
7. Residual add
8. Optional global ALC (unchanged) runs after residual if enabled

## Router semantics (implemented now)

- Per token `(b,t)`, compute logits:
  \[
  r_{e} = w^{router}_{e}\cdot h + b^{router}_{e}
  \]
- Softmax probabilities are stored for diagnostics.
- Top-1 index is selected with argmax and validated with bounds assertions.
- Only selected expert contributes output in this milestone.

## Expert FFN ownership (implemented now)

Each expert owns distinct FFN parameters in `MoEState`:

- `expert_fcw` shape `(E, 4C, C)`
- `expert_fcb` shape `(E, 4C)`
- `expert_fcprojw` shape `(E, C, 4C)`
- `expert_fcprojb` shape `(E, C)`

For selected expert \(i\):
\[
f_i = \text{GELU}(W^{in}_i h + b^{in}_i),\quad
u_i = W^{out}_i f_i + b^{out}_i
\]

## Expert-local memory ownership (implemented now)

Each expert owns:

- slots `expert_memory` `(E,S,D)`
- slot keys `expert_memory_keys` `(E,S,D)`
- query projection `memory_query_proj` `(E,D,C)`
- write projection `memory_write_proj` `(E,D,C)`
- slot-to-hidden projection `memory_slot_to_hidden` `(E,C,D)`

No cross-expert memory bank is used in this path.

## Read equations (implemented now)

For selected expert \(i\):

\[
q_i = W^q_i u_i,\quad
s_{ij} = \frac{q_i \cdot k_{ij}}{\sqrt{D}},\quad
p_{ij} = \text{softmax}_j(s_{ij})
\]
\[
m_i = \sum_j p_{ij} v_{ij},\quad
m^h_i = W^m_i m_i,\quad
o_i = u_i + \lambda m^h_i
\]

where \(\lambda = \texttt{moe\_memory\_fusion\_scale}\).

## Write equations (implemented now)

Only selected expert \(i\) is updated:

\[
w_i = W^w_i u_i
\]
\[
v_{ij} \leftarrow (1-\alpha) v_{ij} + \alpha\, p_{ij}\, w_i
\]
\[
k_{ij} \leftarrow (1-\alpha) k_{ij} + \alpha\, p_{ij}\, q_i
\]

where \(\alpha = \texttt{moe\_memory\_update\_rate}\).

## Why memory is after FFN

Memory is intentionally applied after expert FFN so lookup/write occur in the expert-transformed space \(u_i\), not raw shared LN2 space \(h\). This keeps specialization local to each expert’s representation.

## Why expert-local memory instead of shared memory

- Prevents interference between unrelated experts.
- Aligns sparse routing with sparse memory access/update.
- Makes expert identity meaningful: `Expert_i = FFN_i + Memory_i`.

## Implemented now vs future work

### Implemented now

- Top-1 per-token router.
- True per-expert FFN parameters.
- True per-expert memory parameters + state.
- Selected-expert-only read/write behavior.
- Forward integration in block path for MoE-enabled layers.
- Unit tests for routing validity, expert/memory isolation, read/write activation, repeated-write isolation, and finite outputs.

### Future work (not in this pass)

- Top-k expert routing/combination.
- Load balancing auxiliary losses.
- MoE backward/optimizer support for expert parameters.
- Checkpoint serialization of expert tensors.
- Distributed expert parallelism.

## Known limitations of this first top-1 milestone

- `gpt2_backward` explicitly rejects active MoE layers (forward-only milestone for now).
- Router and expert weights are runtime-initialized in-memory and not checkpoint-loaded.
- Single selected expert only (no expert combine).

## Experimental test build/run commands

```bash
gcc -Ofast -g -I. dev/test/moe_expert_memory.c -lm -o dev/test/moe_expert_memory
./dev/test/moe_expert_memory
```
