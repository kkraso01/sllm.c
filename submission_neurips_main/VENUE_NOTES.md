# Venue Notes: submission_neurips_main

## Title used

**Structured Runtime Memory via ALC: Controlled Mechanism Evidence for Inference-Time Adaptation**

## Venue fit rationale

This variant keeps the same underlying method and numbers while adapting framing to expected reviewer priorities for this venue.

## Abstract rationale

The abstract emphasizes controlled mechanism evidence, avoids broad downstream claims, and states evidence boundaries explicitly.

## Key framing adjustments

- Introduction and results framing are tailored to this venue's expected balance of novelty vs empirical caution.
- Related work emphasis is calibrated to position ALC against memory, retrieval, and test-time adaptation without overclaiming.
- Limitations and conclusion are adjusted to keep contribution scope conservative and evidence-aligned.

## What is emphasized

- Structured in-model adaptive memory.
- Stable inference-time updates without weight changes.
- SessionKV-DR as central mechanism-aligned evidence.
- Persistence and overwrite behavior as bounded adaptation signals.

## What is de-emphasized

- Broad downstream NLP generalization claims.
- Universal superiority over external retrieval or parameter-update adaptation.

## Risks / weaknesses for this venue

- Evidence breadth may be viewed as limited because the strongest benchmark is controlled and mechanism-targeted.
- No-write competitiveness in some slices may be interpreted as incomplete policy robustness.
- Additional large-scale public benchmarks would strengthen external validity.
