# Submission readiness assessment (ALC)

## What claims are currently supported

- **Supported (mechanism-level):** ALC improves delayed synthetic recall versus baseline in this suite.
- **Supported:** state updates are stable in 10k-step stress (no NaNs; bounded slot norms).
- **Supported:** persistence round-trip is exact in tested settings (state and behavior diff = 0).
- **Supported:** trained interface reduces retrieval loss relative to untrained initialization.
- **Supported:** ALC incurs moderate measured overhead (~1.14x forward time in tiny CPU setup).

## Realistic venue fit (current package)

- **Most realistic now:** workshop track (efficient adaptation, memory mechanisms, systems-for-LLMs workshops).
- **Borderline for main conference:** without standard benchmark breadth and stronger baseline comparisons.

## What is still needed for stronger submission

1. Multi-seed runs with confidence intervals.
2. At least one public benchmark beyond synthetic tasks.
3. Direct comparison to one strong memory/retrieval-adaptation baseline.
4. Expanded efficiency results (larger model scales, throughput/latency curves).
5. Optional qualitative examples demonstrating useful persistence in realistic prompts.

## Workshop-ready checklist status

- [x] Clear method description
- [x] Reproducible scripts and raw outputs
- [x] Main tables and ablations
- [x] Honest limitations and scoped claims
- [ ] Standard external benchmark coverage
- [ ] Multi-seed statistical significance
