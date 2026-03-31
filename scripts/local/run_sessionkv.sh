#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/sessionkv"
OUT_CSV="$OUT_DIR/metrics_sessionkv_seed42.csv"
SUMMARY_JSON="$OUT_DIR/sessionkv_summary_seed42.json"

mkdir -p "$OUT_DIR"
cd "$ROOT_DIR"

echo "[run_sessionkv] Building experiments/alc_publication_suite"
cc -O2 -fopenmp experiments/alc_publication_suite.c -lm -lgomp -o experiments/alc_publication_suite

echo "[run_sessionkv] Running benchmark suite (seed=42)"
./experiments/alc_publication_suite "$OUT_CSV" "$OUT_DIR" 42

echo "[run_sessionkv] Extracting SessionKV-DR metrics"
python3 - "$OUT_CSV" "$SUMMARY_JSON" <<'PY'
import csv, json, sys
from collections import defaultdict

csv_path, out_json = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
summary = defaultdict(dict)
for r in rows:
    if r['experiment'] == 'sessionkv' and r['variant'] in {'baseline', 'alc_no_write', 'alc'}:
        summary[r['variant']][r['metric']] = float(r['value'])

with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)

key = 'delayed_acc_delay_6_facts_4'
if all(key in summary[v] for v in ('baseline', 'alc_no_write', 'alc')):
    print('SessionKV-DR delayed recall (delay=6,facts=4):')
    print(f"  baseline     : {summary['baseline'][key]:.6f}")
    print(f"  alc_no_write : {summary['alc_no_write'][key]:.6f}")
    print(f"  alc          : {summary['alc'][key]:.6f}")
else:
    print('Warning: key delayed_acc_delay_6_facts_4 missing from one or more variants')
print(f'Wrote {out_json}')
PY

echo "[run_sessionkv] Outputs:"
echo "  - metrics csv: $OUT_CSV"
echo "  - sessionkv summary json: $SUMMARY_JSON"
