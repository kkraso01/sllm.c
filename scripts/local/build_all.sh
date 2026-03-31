#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[build_all] Building core CPU binaries via Makefile"
make train_gpt2 test_gpt2

echo "[build_all] Building publication benchmark binary"
cc -O2 -fopenmp experiments/alc_publication_suite.c -lm -lgomp -o experiments/alc_publication_suite

echo "[build_all] Building artifact-independent ALC tests"
cc -Ofast -fopenmp dev/test/alc_smoke.c -lm -lgomp -o dev/test/alc_smoke
cc -Ofast -fopenmp dev/test/tiny_alc_e2e.c -lm -lgomp -o dev/test/tiny_alc_e2e
cc -Ofast -fopenmp dev/test/alc_hardening.c -lm -lgomp -o dev/test/alc_hardening

echo "[build_all] Done"
