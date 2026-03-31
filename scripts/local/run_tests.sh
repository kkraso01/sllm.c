#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[run_tests] Ensuring ALC-focused test binaries are built"
cc -Ofast -fopenmp dev/test/alc_smoke.c -lm -lgomp -o dev/test/alc_smoke
cc -Ofast -fopenmp dev/test/tiny_alc_e2e.c -lm -lgomp -o dev/test/tiny_alc_e2e
cc -Ofast -fopenmp dev/test/alc_hardening.c -lm -lgomp -o dev/test/alc_hardening

echo "[run_tests] Running dev/test/alc_smoke"
./dev/test/alc_smoke

echo "[run_tests] Running dev/test/tiny_alc_e2e"
./dev/test/tiny_alc_e2e

echo "[run_tests] Running dev/test/alc_hardening"
./dev/test/alc_hardening

echo "[run_tests] All selected artifact-independent tests passed"
