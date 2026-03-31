# Related work notes for ALC paper

## Positioning summary

Working framing: **ALC is a structured in-model adaptive memory that updates state at inference time without changing core transformer weights**. It is closest to memory-augmented transformer lines and test-time adaptation lines, but differs by combining (i) explicit slot/key state, (ii) routing-weighted EMA updates, and (iii) persistence as a first-class mechanism in the runtime implementation.

## Candidate papers and comparison map

> NOTE: Citation keys are placeholders for `references.bib` entries.

### Segment-level and recurrent memory transformers

1. **Transformer-XL** (`dai2019transformerxl`)
   - Summary: recurrence over hidden states + relative positional encoding for longer context reuse.
   - Difference vs ALC: Transformer-XL reuses prior activations; ALC writes into explicit slots/keys with controllable update-rate and routing.
   - Overlap risk: both target long-horizon dependencies; novelty claim should avoid implying ALC is the first recurrent memory mechanism.

2. **Compressive Transformer** (`rae2020compressive`)
   - Summary: extends memory with compressed older memories.
   - Difference vs ALC: compression hierarchy over activations vs learned slot memory with keyed reads and explicit persistence.
   - Overlap risk: if claiming superior long-context handling without direct comparison.

3. **Recurrent Memory Transformer (RMT)** (`bulatov2022rmt`)
   - Summary: inserts trainable memory tokens propagated between segments.
   - Difference vs ALC: RMT uses token-level recurrent carriers; ALC uses non-token externalized in-model state with write/read operators.
   - Overlap risk: both are “in-model memory”. Be explicit that ALC contribution is update rule + persistence + interface in this codebase.

4. **Infini-attention / infinite-context lines** (`munkhdalai2024infini`)
   - Summary: compressive recurrent memory for unbounded context windows.
   - Difference vs ALC: target context extension in attention mechanism; ALC is a modular adaptation component with explicit mutable memory state.
   - Overlap risk: avoid broad claim that ALC “solves long context”.

### Retrieval-augmented and external memory

5. **kNN-LM** (`khandelwal2020knnlm`)
   - Summary: retrieves nearest neighbors from datastore at inference.
   - Difference vs ALC: external datastore retrieval vs internal state updates in model memory.
   - Overlap risk: do not claim novelty for “inference-time adaptation” alone.

6. **RAG** (`lewis2020rag`)
   - Summary: retrieval-augmented generation with differentiable retriever + generator.
   - Difference vs ALC: dependence on corpus/index + retrieval pipeline, while ALC adapts with internal writable slots.
   - Overlap risk: both increase factual adaptability at inference; ALC claim should emphasize *self-contained* stateful adaptation.

7. **RETRO** (`borgeaud2022retro`)
   - Summary: retrieval from huge external database for stronger LM performance.
   - Difference vs ALC: external non-parametric memory scaling vs compact in-model slot memory.
   - Overlap risk: ALC is not a replacement for retrieval at web-scale knowledge.

8. **Memorizing Transformers** (`wu2022memorizing`)
   - Summary: integrates approximate nearest-neighbor memory into transformer.
   - Difference vs ALC: ANN memory bank keyed over past representations; ALC uses fixed-size slot/key state with EMA writes and persistence semantics.
   - Overlap risk: closest conceptual overlap in “memory-augmented inference”, so novelty claim must be narrow.

### Test-time adaptation / inference-time learning

9. **Test-Time Training (TTT)** (`sun2020ttt`)
   - Summary: updates model parameters at test time using self-supervised objective.
   - Difference vs ALC: ALC adapts state, not core model weights (except interface training during offline training).
   - Overlap risk: avoid claiming first inference-time adaptation method.

10. **Test-Time Prompt Tuning / TTA variants** (`shu2022tpt`, `wang2021tent`)
   - Summary: adapt prompts or normalization parameters at inference/test time.
   - Difference vs ALC: they adjust parameter-like objects; ALC writes structured memory states with load/save.
   - Overlap risk: overlapping “no full finetune” claim.

11. **Fast weights / linear attention as memory** (`schlag2021fastweights`)
   - Summary: interpret linear transformers as implicit fast-weight memory updates.
   - Difference vs ALC: implicit matrix-state dynamics vs explicit slot/key bank and discrete persistence interface.
   - Overlap risk: conceptual prior art for stateful updates; claim should not imply fundamental novelty in principle.

### KV-cache alternatives and long-context efficiency

12. **Mamba / state-space model alternatives** (`gu2023mamba`)
   - Summary: recurrent state-space dynamics can replace attention/KV cache at scale.
   - Difference vs ALC: architecture-level replacement; ALC is an add-on component to transformer blocks.
   - Overlap risk: do not frame as global replacement of cache/attention paradigms.

13. **StreamingLLM / sink token methods** (`xiao2023streamingllm`)
   - Summary: maintain stable generation with limited cache via token strategies.
   - Difference vs ALC: cache management heuristic vs writable learned memory module.
   - Overlap risk: both involve long-running inference stability.

## Strongest honest novelty claim

Most defensible novelty claim for this submission:

1. **Engineering + method integration claim**: a *fully implemented, differentiable, routing-based adaptive memory module* integrated into a GPT-style runtime, with explicit write/read math, stability hardening, and persistent state round-tripping.
2. **Empirical claim scope**: on controlled adaptation tasks, ALC improves delayed recall over no-memory baselines while maintaining stable bounded state updates and exact persistence identity.

Not defensible without more evidence:
- “state-of-the-art long-context modeling”
- “lifelong memory”
- broad claims over retrieval-augmented methods on real knowledge-intensive benchmarks.

## Novelty-overlap risk notes (explicit)

- **High overlap risk**: memory-augmented transformer family (RMT, Memorizing Transformers).
- **Medium overlap risk**: test-time adaptation framing (TTT, TENT/TPT).
- **Lower overlap risk**: persistence-first runtime contract and weighted-EMA in-slot updates with differentiable routing in this compact C implementation.

## Gaps for camera-ready novelty confidence

- Add at least one public benchmark with stronger ecological validity (e.g., long-context recall benchmark with delayed facts).
- Include direct comparison to one retrieval-free recurrent memory baseline beyond plain transformer.
- Expand robustness: multi-seed intervals and longer stress runs with failure probabilities.
