# Aligned benchmark plan for ALC (structured inference-time adaptive memory)

## Why TinyQA is not sufficient as the primary benchmark

TinyQA is useful as a semi-realistic supporting task, but it is not a direct test of ALC's core claim. TinyQA performance can be influenced by tokenization heuristics, lexical overlap, and generic retrieval proxies, so success/failure does not isolate whether ALC actually **writes new information during inference**, retains it under interference, and later retrieves it to affect predictions.

As a result, TinyQA should remain in the suite, but only as supporting evidence.

## Why the new benchmark matches ALC's contribution

The new primary benchmark is designed so that success requires session-local write/read behavior. It directly tests the claim:

> ALC can write new information during inference, preserve it across delay/interference, and retrieve it later to affect predictions.

The benchmark uses explicit session key-value writes, delayed queries, overwrite updates, and persistence reload checks under controlled distractor load and deterministic seeds.

## Benchmark definition: SessionKV-DR (Session Key-Value Delayed Recall)

Primary benchmark name (paper-facing): **SessionKV-DR**.

### Task 1: Delayed key-value recall
- Insert one or more session facts `key -> value` during inference.
- Apply distractor updates/delay steps.
- Query each inserted key later.
- Report exact recall accuracy as functions of delay, inserted fact count, and distractor count.

### Task 2: Overwrite/update recall
- Insert `key -> value_old`.
- Later update same key with `key -> value_new`.
- After additional distractors, query key.
- Report final-value accuracy and stale-value confusion rate.

### Task 3: Persistence across runs
- Insert facts, save ALC state, reload into fresh process/model state.
- Query without reinserting facts.
- Report post-reload recall accuracy and state/behavior equality diagnostics.

### Task 4: Capacity/interference sweep
- Sweep inserted fact count, delay length, and distractor load.
- Report recall degradation curves.

## Required variants

All SessionKV-DR primary results compare:
1. `baseline` (no ALC retrieval/write effect)
2. `alc_no_write` (ALC read path active, inference-time writes disabled)
3. `alc` (full ALC)

## Metrics

Required:
- exact recall accuracy
- delayed recall accuracy
- overwrite accuracy
- persistence recall accuracy
- accuracy vs delay
- accuracy vs inserted fact count
- accuracy vs distractor count

Included when straightforward:
- stale-value confusion rate
- persistence slot max-abs diff and behavior max-abs diff

## Decision criteria

Primary support pattern sought:

- `alc` significantly outperforms `alc_no_write` and `baseline` on delayed recall, overwrite, and persistence recall.
- Curves show controlled degradation (not collapse) under increased delay/interference.

Interpretation discipline:
- If `alc_no_write` is close to `alc`, report this as weak write benefit.
- If overwrite or persistence fail, state that adaptive memory claim is only partially supported.

## Negative outcomes and meaning

- **No `alc` > `alc_no_write` gap:** current write policy may add interference or insufficiently target updates.
- **Poor overwrite accuracy with high stale confusion:** memory update/selection is not reliably adaptive.
- **Poor reload recall or non-trivial behavior mismatch:** persistence semantics are not robust enough for claim scope.
- **Sharp collapse under moderate distractors:** capacity/interference robustness remains limited.

These outcomes are informative and must be reported directly in Results/Limitations.
