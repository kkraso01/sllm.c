# ALC environment variable reference (local runs)

This reference centralizes ALC-related runtime knobs used by `train_gpt2.c`.

## Core on/off and persistence

- `LLMC_USE_ALC`
  - `0` disable ALC, `1` enable ALC.
- `LLMC_ALC_STATE_IN`
  - Optional path to load an existing ALC state sidecar at startup.
- `LLMC_ALC_STATE_OUT`
  - Optional path to save ALC state sidecar at shutdown.

## Memory geometry

- `LLMC_ALC_NUM_SLOTS` (default from compiled config)
- `LLMC_ALC_SLOT_DIM`
- `LLMC_ALC_KEY_DIM`

## Update and fusion behavior

- `LLMC_ALC_UPDATE_RATE` (EMA update rate)
- `LLMC_ALC_FUSION_MODE`
  - `0` additive
  - `1` gated
- `LLMC_ALC_UPDATE_MODE`
  - `0` off (equivalent to no write updates)
  - `1` train_only
  - `2` always
- `LLMC_ALC_APPLY_EVERY_N_LAYERS`
- `LLMC_ALC_ADDITIVE_SCALE`

## Routing behavior

- `LLMC_ALC_ROUTING_MODE`
  - `0` hard_top1
  - `1` softmax
  - `2` topk_softmax
- `LLMC_ALC_TOPK`
- `LLMC_ALC_TEMPERATURE`

## Debug

- `LLMC_ALC_DEBUG`
  - `0` off, nonzero enables extra ALC debug logging.

## Useful local presets

### Baseline (no ALC)
```bash
export LLMC_USE_ALC=0
```

### ALC interface enabled but writes disabled (`alc_no_write` behavior)
```bash
export LLMC_USE_ALC=1
export LLMC_ALC_UPDATE_MODE=0
```

### Full ALC
```bash
export LLMC_USE_ALC=1
export LLMC_ALC_UPDATE_MODE=2
```

## Notes about non-ALC knobs

`train_gpt2.c` also has synthetic-model/debug env vars such as `LLMC_USE_SYNTHETIC_MODEL` and `LLMC_SYNTH_*`. Those are optional for testing and are not required for the local ALC publication/demo flow documented in `RUN_LOCAL.md`.
