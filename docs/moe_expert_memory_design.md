# MoE Expert-Local Memory Design (Experimental Path)

## Stage 1 audit of `train_gpt2_moe_experimental.c` (before this top-2 pass)

Before this pass, the experimental MoE path already had a real top-1 expert-local forward path:

- Router computed per-expert logits and picked a single argmax expert.
- Selected expert owned its own FFN parameters and local memory tensors.
- Selected expert read/wrote only its own memory bank.
- Non-selected experts were untouched.
- Backward/training for active MoE layers was intentionally blocked in `gpt2_backward`.

What needed to change for this milestone:

- replace top-1 selection with top-2 routing and normalized selected-expert weights
- run both selected experts each token
- weighted-combine selected expert outputs
- scale each selected expert memory write coherently by its router weight
- strengthen tests for top-2 validity, contribution, and isolation

---

## Implemented top-2 block order (current experimental state)

For MoE-enabled layers in `train_gpt2_moe_experimental.c`:

1. LN1 -> attention -> residual
2. LN2
3. Router logits over experts
4. Select top-k experts (`k=moe_topk`, top-2 by default)
5. For each selected expert `i`:
   - `FFN_i(h) -> u_i`
   - local memory read in expert `i`
   - fuse expert compute + expert memory
   - local memory write in expert `i`
6. Weighted combine selected expert outputs
7. Residual add
8. Optional ALC remains after residual (unchanged)

No global ALC was moved into the MoE path in this pass.

---

## Router math (implemented now)

Given token hidden state `h` and expert logits:

\[
r_i = w_i^{router} \cdot h + b_i^{router}
\]

Select top-k set \(\mathcal{E}=\text{TopK}(r,k)\) (`k=2` default).

Router weights over selected experts use stable selected-softmax with temperature \(\tau>0\):

\[
\alpha_i =
\begin{cases}
\frac{\exp((r_i-r_{\max})/\tau)}{\sum_{j\in\mathcal{E}}\exp((r_j-r_{\max})/\tau)} & i\in\mathcal{E} \\
0 & i\notin\mathcal{E}
\end{cases}
\]

Notes:

- `moe_topk` is validated with `1 <= moe_topk <= moe_num_experts`.
- `moe_router_temperature` is validated with `> 0`.
- Full router softmax over all experts is still stored in `router_probs` for diagnostics.
- Selected indices and selected weights are stored per token in `(B*T, K)` buffers.

---

## Selected-expert compute + combine math (implemented now)

For each selected expert \(i\in\mathcal{E}\):

\[
u_i = \text{FFN}_i(h)
\]

\[
m_i = \text{ExpertMemoryRead}_i(u_i)
\]

\[
o_i = u_i + \lambda\,m_i^h
\]

where \(\lambda = \texttt{moe\_memory\_fusion\_scale}\).

Final MoE output:

\[
o = \sum_{i\in\mathcal{E}} \alpha_i o_i
\]

This is the actual forward path now used in MoE-enabled layers.

---

## Expert-local memory ownership and write semantics (implemented now)

Each expert owns distinct local tensors/state:

- FFN params (`expert_fcw`, `expert_fcb`, `expert_fcprojw`, `expert_fcprojb`)
- memory state (`expert_memory`, `expert_memory_keys`)
- memory projections (`memory_query_proj`, `memory_write_proj`, `memory_slot_to_hidden`)

For selected expert \(i\), slot routing stays local and softmax-based over that expert’s slots:

\[
q_i = W_i^q u_i,\quad
w_i = W_i^w u_i
\]

\[
p_{is} = \text{softmax}_s\left(\frac{q_i\cdot k_{is}}{\sqrt{D}}\right)
\]

\[
m_i = \sum_s p_{is} v_{is}
\]

Weighted write semantics for top-2 are implemented via an effective update rate
\(\alpha_i^{eff} = \alpha\,\alpha_i\):

\[
v_{is} \leftarrow (1-\alpha_i^{eff})v_{is} + \alpha_i^{eff} p_{is} w_i
\]
\[
k_{is} \leftarrow (1-\alpha_i^{eff})k_{is} + \alpha_i^{eff} p_{is} q_i
\]

This preserves:

- selected experts write only to their own local memory
- non-selected experts are not touched
- write magnitude scales with router selection weight

---

## Why top-2 before global ALC integration

Top-2 is the correct next milestone because it validates core sparse-MoE behavior first:

- sparse multi-expert routing correctness
- weighted expert output mixing
- concurrent expert-local memory updates with isolation

Without this, adding global ALC would blur ownership boundaries and make debugging attribution harder. Top-2 expert-local behavior must be correct and testable before introducing cross-expert/global memory interactions.

---

## Implemented now vs deferred

### Implemented now

- Top-k router path with top-2 default in experimental MoE config.
- Stable selected-expert softmax weighting with router temperature.
- Two-expert forward execution and weighted output combine.
- Expert-local weighted writes for selected experts only.
- Extended artifact-independent tests for routing validity, normalization, output contribution, write isolation, and finite forward stability.

### Deferred (explicitly not in this pass)

- MoE backward/optimizer support for expert params and router params.
- Load-balancing / auxiliary routing losses.
- Distributed MoE and production checkpointing of expert tensors.
- Global ALC integration into MoE internals.

Backward status remains unchanged: active MoE layers intentionally fail fast in `gpt2_backward`.

---

## Experimental test build/run commands

```bash
gcc -Ofast -g -I. dev/test/moe_expert_memory.c -lm -o dev/test/moe_expert_memory
./dev/test/moe_expert_memory
```
