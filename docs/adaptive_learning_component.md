# Adaptive Learning Component (ALC) in `llm.c` CPU GPT-2 Path

## Audit update (this pass)

The prior patch was audited for end-to-end completeness. Gaps found and fixed in this pass:

- Added missing `alc_slot_dim` config field and wired it through env parsing, validation, logging, docs, and runtime behavior.
- Decoupled slot-storage dimension from hidden dimension by introducing explicit slot projection:
  - store slots in `(S, D)` (`D = alc_slot_dim`)
  - project retrieved slot back into hidden space with `slot_to_hidden` `(C, D)`.
- Added checkpoint-independent smoke validation (`dev/test/alc_smoke.c`) so ALC read/write behavior can be exercised without starter-pack model files.
- Updated docs and change log to reflect exact current implementation state.

## Stage 0: Repository mapping and insertion decision

### Baseline architecture summary (as implemented in `train_gpt2.c`)
The baseline CPU path in this repository implements GPT-2 with:
1. token + position embedding (`encoder_forward`)
2. per-layer transformer block:
   - LN1 -> QKV projection -> causal self-attention -> attention projection -> residual add
   - LN2 -> FFN (`fcw` + GELU + `fcprojw`) -> residual add
3. final layer norm
4. LM head (`wte` reused as classifier projection)
5. softmax and optional cross entropy loss

The model state is represented by:
- `GPT2Config`
- `ParameterTensors` + contiguous parameter memory
- `ActivationTensors` + contiguous activation memory
- explicit forward/backward/update functions for train/infer in one file

### Source map used for ALC integration
- `GPT2Config`, `ParameterTensors`, `ActivationTensors`, `GPT2` model struct: `train_gpt2.c`
- model loading and initialization: `gpt2_build_from_checkpoint`
- inference/training forward path: `gpt2_forward`
- backward and optimizer: `gpt2_backward`, `gpt2_update`
- runtime/training entry point + generation loop: `main`

### ALC placement decision
**Chosen insertion point:** after the FFN residual in each transformer layer (i.e., after `residual3` is formed), optionally every N layers.

**Rationale:**
- least invasive insertion point in existing block topology
- naturally composes as `attention + FFN + ALC` sub-stack inside each layer
- keeps baseline path unchanged when disabled (`use_alc=0`)
- allows both read/fusion and optional write updates with current layer hidden state

Conceptual block with ALC enabled:

```text
LN -> Attn -> Residual -> LN -> FFN -> Residual -> ALC(read/fuse [+optional write])
```

## Stage 1: Config plumbing
ALC config is represented with a dedicated `ALCConfig` struct (stored inside `GPT2`) and includes:

- `use_alc`
- `alc_num_slots`
- `alc_slot_dim`
- `alc_key_dim`
- `alc_update_rate`
- `alc_fusion_mode` (`additive`, `gated`)
- `alc_update_mode` (`off`, `train_only`, `always`)
- `alc_apply_every_n_layers`
- `alc_additive_scale`

Defaults preserve baseline behavior by keeping `use_alc=0`.

Runtime override is exposed via environment variables in `main`:
- `LLMC_USE_ALC`
- `LLMC_ALC_NUM_SLOTS`
- `LLMC_ALC_SLOT_DIM`
- `LLMC_ALC_KEY_DIM`
- `LLMC_ALC_UPDATE_RATE`
- `LLMC_ALC_FUSION_MODE`
- `LLMC_ALC_UPDATE_MODE`
- `LLMC_ALC_APPLY_EVERY_N_LAYERS`
- `LLMC_ALC_ADDITIVE_SCALE`

Validation asserts reject invalid config when ALC is enabled.

## Stage 2: Native ALC structs/state
ALC is implemented as first-class in-model state (`ALCState` in `GPT2`) with:

### ALC parameters
- `query_proj` `(K, C)`
- `write_proj` `(D, C)`
- `slot_to_hidden` `(C, D)`
- `gate_h`, `gate_a`, `gate_b` `(C)`

### Adaptive memory/state
- `slot_keys` `(S, K)`
- `slots` `(S, D)`

### Runtime traces/scratch
- `query_buffer` `(B*T, K)`
- `retrieved_buffer` `(B*T, C)`
- `selected_slots` `(B*T)`

ALC memory is lazily initialized on first forward pass with ALC enabled.

## Stage 3: ALC forward/read path
Given hidden state vector `h` at token position:

1. **key projection**: `q = W_q h`
2. **slot lookup**: `i = argmax_s <q, key_s>`
3. **retrieval**: `a = slot_i`
4. **slot projection back to hidden**: `a_h = W_s a`
5. **fusion**:
   - additive mode: `h' = h + alpha * a_h`
   - gated mode: `g = sigmoid(w_h ⊙ h + w_a ⊙ a_h + b)`, then
     `h' = g ⊙ h + (1-g) ⊙ a_h`

This is wired directly into per-layer forward after FFN residual.

## Stage 4: ALC write/update path
Optional write path performs deterministic EMA update on selected slot(s):

- `slot_i <- (1-eta) slot_i + eta * (W_w h')`
- `key_i <- (1-eta) key_i + eta * q`

where `eta = alc_update_rate`.

Update execution policy:
- `off`: never write
- `train_only`: write only when targets are present (training-style forward)
- `always`: write in both train and inference forwards

## Stage 5: End-to-end transformer integration
ALC is called inside the layer loop in `gpt2_forward` if both are true:
- layer selected by `alc_apply_every_n_layers`
- `use_alc=1`

The model logs an activation banner once per process when ALC first runs.

## Stage 6: Interaction with MoE / Engram and stack intent
This codebase's CPU GPT-2 path does not currently implement MoE or Engram modules in `train_gpt2.c`. ALC is integrated at the exact architectural insertion point where those modules would compose as peer in-model components.

Intended conceptual stack in this prototype:

```text
embeddings + attention + FFN (+future MoE) + (future Engram) + ALC + LM head
```

## Stage 7: Limitations and future extension notes
Current prototype limitations:
- ALC projection/gate tensors are runtime-initialized and not checkpoint-serialized yet.
- ALC is implemented in forward/update path; there is no explicit ALC gradient/backward pass.
- Lookup is hard top-1 by dot-product; no kNN or soft addressing.
- State is process-local and resets on model reload.

Future extensions:
- checkpoint save/load for ALC parameter/state tensors
- optional differentiable addressing path
- per-layer ALC state banks (instead of shared global bank)
- ALC-specific regularizers and consolidation policies
- integration points for MoE/Engram once merged in this repository path

## Build and run instructions
### Compile baseline CPU path
```bash
make train_gpt2
```

### Run baseline (ALC disabled by default)
```bash
./train_gpt2
```

### Run with ALC enabled
```bash
LLMC_USE_ALC=1 ./train_gpt2
```

### Run with ALC gated fusion
```bash
LLMC_USE_ALC=1 LLMC_ALC_FUSION_MODE=1 ./train_gpt2
```

### Run with ALC custom slot dimensions
```bash
LLMC_USE_ALC=1 LLMC_ALC_NUM_SLOTS=128 LLMC_ALC_SLOT_DIM=128 LLMC_ALC_KEY_DIM=64 ./train_gpt2
```

### Run with ALC train-only update policy
```bash
LLMC_USE_ALC=1 LLMC_ALC_UPDATE_MODE=1 LLMC_ALC_UPDATE_RATE=0.05 ./train_gpt2
```

### ALC with MoE / Engram
MoE/Engram toggles are **not present in `train_gpt2.c` today**. Once available, run with both feature flags enabled along with `LLMC_USE_ALC=1`, keeping ALC after FFN/MoE and after Engram fusion by design.

## Minimal validation checklist
- compile with `make train_gpt2`
- compile regression test binary with `make test_gpt2`
- compile ALC smoke test with `cc -Ofast -fopenmp dev/test/alc_smoke.c -lm -lgomp -o dev/test/alc_smoke`
- run ALC smoke test with `./dev/test/alc_smoke`
- baseline run with `LLMC_USE_ALC=0` (requires model/tokenizer artifacts)
- ALC run with `LLMC_USE_ALC=1` (requires model/tokenizer artifacts)
- observe one-time `[ALC] enabled:` log line
- verify assertion failures for intentionally invalid configs (e.g. `LLMC_USE_ALC=1 LLMC_ALC_NUM_SLOTS=0`)
