#!/bin/sh
# Regenerate the integer goldens the V1 testbenches compare against (S1).
#
# The output is generated, not tracked — hardware/workspace/golden/ is gitignored and the
# seed makes it reproducible, so committing 9 MB of integers would buy nothing. Run this
# once after a clone, and again whenever algorithm/quantization changes.
#
#   sh hardware/build/make_golden.sh              # everything
#   sh hardware/build/make_golden.sh tiny-4       # one block preset
#   sh hardware/build/make_golden.sh model        # the whole model, paper bit-widths
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
GOLDEN="$PY hardware/tools/export_hls_golden.py"

presets="$*"
[ -n "$presets" ] || presets="requant search-4 search-8 track-4 track-8 tiny-4 tiny-8 model model-tiny"

for preset in $presets; do
  case "$preset" in
    # direct cases for the requant primitive, over its whole domain.
    requant)    $GOLDEN --scope requant ;;
    # model scope: two stems, ONE shared block stack, two heads, at the paper's widths
    # (4-bit MHA/MLP, 4-bit residual stream, 8-bit final norm + head).
    model)      $GOLDEN --scope model --model hbtxr ;;
    model-tiny) $GOLDEN --scope model --model tiny ;;
    # block scope: one block + the stage vectors S2-S5 compare against.
    *)          $GOLDEN --mode "${preset%-*}" --bits "${preset##*-}" ;;
  esac
done
