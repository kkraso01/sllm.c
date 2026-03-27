# Adaptive Learning Component (ALC) in `llm.c` CPU GPT-2 Path

## Native-component completeness audit

### Tensor classification (current implementation)

**Trainable via GPT-2 gradients (AdamW path):**
- None of the ALC tensors are currently connected to `gpt2_backward` gradient buffers.

**Model-coupled ALC tensors (persistable sidecar):**
- `query_proj` `(K, C)`
- `write_proj` `(D, C)`
- `slot_to_hidden` `(C, D)`
- `gate_h`, `gate_a`, `gate_b` `(C)`
- `slot_keys` `(S, K)`
- `slots` `(S, D)`

**Mutable runtime state (reinitialized each process):**
- `query_buffer` `(B*T, K)`
- `retrieved_buffer` `(B*T, C)`
- `selected_slots` `(B*T)`
- debug counters/hit histograms

### Persistence semantics

- Base GPT-2 checkpoint (`gpt2_124M.bin`) still stores only legacy GPT-2 core tensors.
- ALC now has explicit sidecar persistence via `LLMC_ALC_STATE_IN` / `LLMC_ALC_STATE_OUT`.
- Sidecar load enforces exact compatibility for:
  - `channels`, `num_slots`, `slot_dim`, `key_dim`, `fusion_mode`, `update_mode`.
- No silent reshape/truncate/fallback is performed.

### Update semantics today

- Forward uses ALC read/fuse when enabled and layer-selected.
- Write/update path uses EMA only (`slot_keys`, `slots`) depending on `alc_update_mode`.
- ALC projection/gate tensors are *used* in forward but *not gradient-trained* in this CPU reference path.

## Placement and block ordering

Actual current block flow in `train_gpt2.c` is:

```text
attention -> FFN -> ALC(optional) -> residual continuation
```

Future coexistence intent inside the same insertion region is:

```text
attention -> FFN/MoE -> Engram(optional) -> ALC(optional) -> residual continuation
```

No MoE/Engram implementation is added in this pass; only ordering/annotations were tightened.

## ALC training semantics

### Trainable tensors

- GPT-2 native tensors in `ParameterTensors` receive gradients and AdamW updates.
- ALC tensors are currently outside `ParameterTensors` and do not receive gradients in this CPU path.

### Stateful tensors

- `slots` and `slot_keys` are online memory state and are updated through EMA write-back.
- Update rules:
  - `slot_i <- (1-eta) slot_i + eta * (W_w h')`
  - `key_i  <- (1-eta) key_i  + eta * q`

### Runtime update policy

Controlled by `alc_update_mode`:
- `off`: never write
- `train_only`: write only when targets are provided
- `always`: write on train + inference forwards

This is an explicit **hybrid design** in current CPU `llm.c` path: gradient-trained core GPT-2 + EMA-updated adaptive memory.

## Serialization/checkpoint integration

ALC persistence uses a dedicated binary sidecar format (`ALC1`, version `1`).

### Serialized tensors

- `query_proj`
- `write_proj`
- `slot_to_hidden`
- `gate_h`, `gate_a`, `gate_b`
- `slot_keys`
- `slots`

### CLI/env integration

- Load before training/inference: `LLMC_ALC_STATE_IN=<path>`
- Save after run: `LLMC_ALC_STATE_OUT=<path>`

### Validation behavior

- Fails fast on header mismatch, version mismatch, or dimension/config incompatibility.
- Does not silently skip enabled ALC load requests.

## Observability / debug mode

Enable lightweight ALC runtime diagnostics:

```bash
LLMC_ALC_DEBUG=1 ...
```

Debug output includes periodic summaries:
- forward call count
- layers applied vs checked
- write call count and token-write count
- active fusion/update mode
- top selected slot and hit count

Default behavior remains non-spammy (`LLMC_ALC_DEBUG=0`).

## Artifact-independent validation path

A new tiny end-to-end test binary exercises integrated GPT block path (forward + backward + update + ALC persistence) without external GPT-2 artifacts.

### Build

```bash
cc -Ofast -fopenmp dev/test/tiny_alc_e2e.c -lm -lgomp -o dev/test/tiny_alc_e2e
```

### Run baseline tiny path

```bash
./dev/test/tiny_alc_e2e
```

(The binary runs baseline then ALC-enabled phases in one execution.)

## Full-run commands

### Baseline (normal artifacts)

```bash
LLMC_USE_ALC=0 ./train_gpt2
```

### ALC enabled (normal artifacts)

```bash
LLMC_USE_ALC=1 LLMC_ALC_STATE_OUT=alc_state.bin ./train_gpt2
```

### ALC reload (normal artifacts)

```bash
LLMC_USE_ALC=1 LLMC_ALC_STATE_IN=alc_state.bin LLMC_ALC_STATE_OUT=alc_state.bin ./train_gpt2
```

## Notes on synthetic tiny model mode in `train_gpt2`

For quick local-only model construction (still requires tokenizer/data loader paths for the full training main loop), optional synthetic weight init is available:

```bash
LLMC_USE_SYNTHETIC_MODEL=1 ./train_gpt2
```

Primary artifact-independent verification should use `dev/test/tiny_alc_e2e`.
