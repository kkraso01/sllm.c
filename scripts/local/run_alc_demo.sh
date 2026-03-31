#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/local_demo"
OUT_CSV="$OUT_DIR/metrics_demo_seed42.csv"

mkdir -p "$OUT_DIR"
cd "$ROOT_DIR"

echo "[run_alc_demo] Building experiments/alc_publication_suite"
cc -O2 -fopenmp experiments/alc_publication_suite.c -lm -lgomp -o experiments/alc_publication_suite

echo "[run_alc_demo] Running tiny local publication-style demo (seed=42)"
./experiments/alc_publication_suite "$OUT_CSV" "$OUT_DIR" 42

echo "[run_alc_demo] Summarizing core adaptation + language benchmark (baseline vs alc_no_write vs alc)"
python3 - "$OUT_CSV" <<'PY'
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
by = defaultdict(dict)
for r in rows:
    if r['experiment'] in {'core_adaptation', 'language_benchmark'} and r['variant'] in {'baseline', 'alc_no_write', 'alc'}:
        by[(r['experiment'], r['metric'])][r['variant']] = float(r['value'])

for (exp, metric), vals in sorted(by.items()):
    if {'baseline', 'alc_no_write', 'alc'} <= set(vals):
        print(f"{exp}/{metric}: baseline={vals['baseline']:.6f}, alc_no_write={vals['alc_no_write']:.6f}, alc={vals['alc']:.6f}")
PY

echo "[run_alc_demo] Wrote metrics + state artifacts under: $OUT_DIR"
