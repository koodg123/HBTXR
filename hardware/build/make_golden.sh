#!/bin/sh
# Regenerate the integer goldens the V1 testbenches compare against (S1).
#
# The output is generated, not tracked — hardware/workspace/golden/ is gitignored and the
# seed makes it reproducible, so committing 9 MB of integers would buy nothing. Run this
# once after a clone, and again whenever algorithm/quantization changes.
#
#   sh hardware/build/make_golden.sh              # all six presets
#   sh hardware/build/make_golden.sh tiny-4       # just one
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python

presets="$*"
[ -n "$presets" ] || presets="search-4 search-8 track-4 track-8 tiny-4 tiny-8"

for preset in $presets; do
  "$PY" hardware/tools/export_hls_golden.py --mode "${preset%-*}" --bits "${preset##*-}"
done
