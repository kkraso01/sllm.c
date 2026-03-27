# MoE + Expert-Local Memory Design (Experimental Path)

## Current baseline summary (preserved)

Baseline is the current hardened ALC CPU training path in `train_gpt2.c`.

- Transformer block is currently:
  1. LN1 -> attention -> projection -> residual
  2. LN2 -> FFN (`fcw/fcb` -> GELU -> `fcprojw/fcprojb`) -> residual
  3. Optional ALC read/fuse/write after FFN residual (when enabled)
- ALC is global across experts because experts do not exist yet; it has one slot bank + key bank per model, with routing controls and fusion/update modes.

This baseline remains untouched and recoverable.

## Preserved baseline vs experimental target

- **Preserved baseline file:** `train_gpt2_alc_hardened_baseline.c` (verbatim copy of current hardened ALC implementation).
- **Experimental target file:** `train_gpt2_moe_experimental.c` (separate architecture pass file).

Rationale: isolate risk, keep stable hardened ALC behavior reviewable, and allow aggressive iteration without hidden regressions in the baseline.

## Where FFN currently lives

Inside `gpt2_forward` per layer:

- `matmul(l_fch, l_ln2, l_fcw, l_fcb)`
- `gelu(l_fch_gelu, l_fch)`
- `matmul(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb)`
- `residual(l_residual3, l_residual2, l_fcproj)`

In backward, the exact inverse path mirrors those tensors.

## Architecture options considered

### Option A — Full per-expert duplication (recommended long-term)

Replace FFN weights with per-expert FFN parameter sets and add per-expert memory banks.

- Each expert owns:
  - FFN in/out projections
  - local memory keys/slots
  - local read/write behavior
- Router picks top-k experts per token.

**Pros:** strongest expert specialization; closest to target concept.
**Cons:** highest parameter and runtime growth; invasive backward/optimizer/checkpoint changes.

### Option B — Shared FFN, per-expert memory (recommended short-term stepping stone)

Keep current dense FFN weights unchanged; add router + per-expert memory as post-FFN experimental path.

- Experts are represented first by router + local memory, while FFN remains shared.

**Pros:** minimal invasive changes; preserves current training/backward flow; validates expert-local memory behavior quickly.
**Cons:** not true MoE FFN capacity increase yet.

### Option C — Shared memory with expert-conditioned access (not preferred by default)

Maintain one global memory bank and condition reads/writes on expert id.

**Pros:** lower memory footprint.
**Cons:** violates desired locality and expert specialization intent; more interference between experts.

## Recommended design

Use a two-stage plan:

1. **Now (minimal implementation): Option B scaffold in experimental file**
   - Keep dense FFN compute path intact.
   - Add MoE router and expert-local memory read/write scaffold after FFN projection and before residual add.
   - Maintain baseline + ALC behavior availability.

2. **Next (major pass): evolve to Option A full MoE FFN**
   - Introduce per-expert FFN weights and top-k dispatch/combine.
   - Keep expert-local memory attached to each expert output path.

## Proposed insertion point and ordering

Recommended ordering inside each transformer block:

1. Attention path unchanged.
2. LN2 output -> MoE routing.
3. Expert FFN compute (or shared FFN in scaffold stage).
4. Expert-local memory read/write on expert output.
5. Fuse memory-enhanced expert output.
6. Residual add.
7. ALC application (optional), after FFN/MoE residual output.

So the intended ordering is:

`attention -> MoE route -> expert FFN -> expert-local memory -> fusion -> residual -> ALC(optional)`

## ALC interaction decision

Primary recommendation: **per-expert memory bank** for MoE path (default), while keeping ALC as a separate optional global post-block mechanism.

- Near term: keep existing ALC unchanged to avoid destabilizing hardened work.
- Future option: add a config to disable global ALC when per-expert memory is active if redundancy is observed.

## Config fields needed

Added/needed MoE config (experimental):

- `use_moe`
- `moe_num_experts`
- `moe_topk`
- `moe_apply_every_n_layers`
- `moe_router_mode`
- `moe_expert_memory_slots`
- `moe_expert_memory_dim`
- `moe_memory_update_rate`
- `moe_memory_fusion_scale`

Environment plumbing mirrors these via `LLMC_MOE_*` variables in experimental path.

## Parameter/state growth

Let C=channels, E=experts, S=memory slots per expert, D=memory dim.

Scaffold growth (current experimental pass):

- Router params: `E*C + E`
- Expert memory state: `E*S*D` values + `E*S*D` keys
- Scratch: `B*T*E` logits/probs + `B*T` selected expert + `B*T*C` retrieval buffer

Full MoE FFN (future) adds approximately (for GPT-2 MLP style):

- Per-expert FFN weights: `E*(4*C*C + 4*C + 4*C*C + C)`
- Optional factorized variants can reduce this if needed.

## Least invasive implementation plan

1. Preserve baseline file untouched.
2. Create experimental file copy.
3. Add MoE config + runtime state structs in experimental file.
4. Add router + expert-local memory scaffold in forward only.
5. Keep default `use_moe=0` to preserve behavior unless explicitly enabled.
6. Keep backward and optimizer baseline-compatible for now.

## Recommended implementation stages

1. **Stage 0 (done):** file separation + design doc.
2. **Stage 1 (done):** config/state plumbing and forward scaffold with expert-local memory.
3. **Stage 2:** add deterministic unit test for expert routing + memory read/write invariants.
4. **Stage 3:** replace shared FFN with true per-expert FFN compute (top-1 first, then top-k).
5. **Stage 4:** add backward support for router and expert FFN params.
6. **Stage 5:** checkpoint format updates for MoE params/state.

## Risks and tradeoffs

- Current scaffold is intentionally minimal and not a complete production MoE training path.
- Forward memory fusion changes outputs when enabled; backward does not yet model MoE-memory gradients.
- Full MoE conversion will require activation layout updates, optimizer state expansion, and careful checkpoint versioning.
- Keeping baseline isolated avoids accidental regressions while this is under iteration.
