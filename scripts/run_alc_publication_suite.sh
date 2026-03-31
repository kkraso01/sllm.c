#!/usr/bin/env bash
set -euo pipefail
mkdir -p paper/results paper/tables paper/figures
cc -O2 -fopenmp experiments/alc_publication_suite.c -lm -lgomp -o experiments/alc_publication_suite
./experiments/alc_publication_suite paper/results/metrics.csv paper/results
python3 scripts/make_paper_artifacts.py
