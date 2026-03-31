#!/usr/bin/env python3
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'paper' / 'results'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1337, 2027, 31415, 27182]


def run_seed(seed: int):
    out_csv = OUT / f'metrics_seed{seed}.csv'
    cmd = ['./experiments/alc_publication_suite', str(out_csv), str(OUT), str(seed)]
    subprocess.run(cmd, check=True)
    with out_csv.open() as f:
        return list(csv.DictReader(f))


def mean(xs):
    return sum(xs) / len(xs)


def std_sample(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    if len(xs) < 2:
        return 0.0
    return 1.96 * std_sample(xs) / math.sqrt(len(xs))


all_rows = []
for seed in SEEDS:
    rows = run_seed(seed)
    for r in rows:
        rr = dict(r)
        rr['seed'] = seed
        all_rows.append(rr)

with (OUT / 'multiseed_raw.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed', 'experiment', 'variant', 'metric', 'value'])
    w.writeheader()
    for r in all_rows:
        w.writerow({'seed': r['seed'], 'experiment': r['experiment'], 'variant': r['variant'], 'metric': r['metric'], 'value': r['value']})

by_key = defaultdict(list)
for r in all_rows:
    by_key[(r['experiment'], r['variant'], r['metric'])].append(float(r['value']))

summary_rows = []
summary_json = {}
for key, vals in sorted(by_key.items()):
    exp, variant, metric = key
    m = mean(vals)
    s = std_sample(vals)
    c = ci95(vals)
    summary_rows.append({
        'experiment': exp,
        'variant': variant,
        'metric': metric,
        'n_seeds': len(vals),
        'mean': m,
        'std': s,
        'ci95': c,
    })
    summary_json.setdefault(exp, {})[f'{variant}::{metric}'] = {
        'n_seeds': len(vals),
        'mean': m,
        'std': s,
        'ci95': c,
        'values': vals,
    }

with (OUT / 'multiseed_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['experiment', 'variant', 'metric', 'n_seeds', 'mean', 'std', 'ci95'])
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)

(OUT / 'multiseed_summary.json').write_text(json.dumps(summary_json, indent=2))

# TinyQA-focused exports for the paper section/table.
tinyqa_rows = []
for r in all_rows:
    if r['experiment'] != 'tinyqa':
        continue
    metric = r['metric']
    if not metric.startswith('qa_acc_delay_'):
        continue
    delay = int(metric.split('_')[-1])
    tinyqa_rows.append({
        'seed': int(r['seed']),
        'variant': r['variant'],
        'delay': delay,
        'qa_acc': float(r['value']),
    })

with (OUT / 'tinyqa_results.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed', 'variant', 'delay', 'qa_acc'])
    w.writeheader()
    for r in sorted(tinyqa_rows, key=lambda x: (x['variant'], x['delay'], x['seed'])):
        w.writerow(r)

tinyqa_by_key = defaultdict(list)
for r in tinyqa_rows:
    tinyqa_by_key[(r['variant'], r['delay'])].append(r['qa_acc'])

tinyqa_summary = {}
for (variant, delay), vals in sorted(tinyqa_by_key.items()):
    tinyqa_summary.setdefault(variant, {})[f'delay_{delay}'] = {
        'n_seeds': len(vals),
        'mean': mean(vals),
        'std': std_sample(vals),
        'values': vals,
    }

(OUT / 'tinyqa_summary.json').write_text(json.dumps(tinyqa_summary, indent=2))
print(f'Wrote multi-seed outputs for {len(SEEDS)} seeds to {OUT}')
