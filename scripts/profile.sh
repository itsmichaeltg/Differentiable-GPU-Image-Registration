#!/usr/bin/env bash
set -euo pipefail

size="${1:-1024}"
iterations="${2:-20}"

cmake -S . -B build
cmake --build build

./build/registration_bench "$size" "$iterations"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found; benchmark completed without Nsight Compute capture" >&2
  exit 0
fi

mkdir -p profiling
ncu --set full \
  --target-processes all \
  -o "profiling/texture_vs_global_${size}" \
  ./build/registration_bench "$size" "$iterations"
