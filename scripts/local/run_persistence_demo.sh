#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/persistence_demo"
OUT_CSV="$OUT_DIR/metrics_persistence_seed42.csv"

mkdir -p "$OUT_DIR"
cd "$ROOT_DIR"

echo "[run_persistence_demo] Building experiments/alc_publication_suite"
cc -O2 -fopenmp experiments/alc_publication_suite.c -lm -lgomp -o experiments/alc_publication_suite

echo "[run_persistence_demo] Run A: create memory and save state sidecars"
./experiments/alc_publication_suite "$OUT_CSV" "$OUT_DIR" 42

echo "[run_persistence_demo] Run B: verify saved-state metrics from output CSV"
python3 - "$OUT_CSV" "$OUT_DIR" <<'PY'
import csv
import os
import sys

csv_path, out_dir = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
metrics = {}
for r in rows:
    if r['experiment'] == 'sessionkv' and r['variant'] in {'baseline', 'alc_no_write', 'alc'} and r['metric'] in {
        'persistence_recall_acc',
        'persistence_save_ok',
        'persistence_load_ok',
        'persistence_slot_max_abs_diff',
        'persistence_behavior_max_abs_diff',
    }:
        metrics[(r['variant'], r['metric'])] = float(r['value'])

for variant in ('baseline', 'alc_no_write', 'alc'):
    save_ok = metrics.get((variant, 'persistence_save_ok'))
    load_ok = metrics.get((variant, 'persistence_load_ok'))
    recall = metrics.get((variant, 'persistence_recall_acc'))
    print(f"{variant}: save_ok={save_ok}, load_ok={load_ok}, persistence_recall_acc={recall}")

print('alc state identity checks:')
print('  slot_max_abs_diff:', metrics.get(('alc', 'persistence_slot_max_abs_diff')))
print('  behavior_max_abs_diff:', metrics.get(('alc', 'persistence_behavior_max_abs_diff')))

for fn in ('sessionkv_state_baseline.bin', 'sessionkv_state_nowrite.bin', 'sessionkv_state_alc.bin'):
    p = os.path.join(out_dir, fn)
    print(f"state file {fn}: {'present' if os.path.exists(p) else 'missing'}")
PY

echo "[run_persistence_demo] Done. Artifacts are in: $OUT_DIR"
