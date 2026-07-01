#!/usr/bin/env bash
# Download Grounded-SAM-2 weights (SAM 2.1 + Grounding DINO) INTO THIS FOLDER.
#
#   bash download_weights.sh          # defaults: SAM2.1-large + GDINO Swin-T  (~1.5 GB)
#   bash download_weights.sh all      # + tiny/small/base_plus + GDINO Swin-B (~4+ GB)
#   bash download_weights.sh fast     # SAM2.1-tiny + GDINO Swin-T (smaller/faster)
#
# URLs verified from IDEA-Research/Grounded-SAM-2 (checkpoints & gdino_checkpoints
# download_ckpts.sh). Config .yaml/.py files are NOT weights — they ship with the
# `sam2` and `groundingdino` pip packages (see README.md next to this script).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if command -v wget >/dev/null 2>&1; then DL(){ wget -c -O "$2" "$1"; }
elif command -v curl >/dev/null 2>&1; then DL(){ curl -L -C - -o "$2" "$1"; }
else echo "Please install wget or curl."; exit 1; fi

SAM="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
GD="https://github.com/IDEA-Research/GroundingDINO/releases/download"
MODE="${1:-default}"

echo "== Grounding DINO (Swin-T) =="
DL "$GD/v0.1.0-alpha/groundingdino_swint_ogc.pth" groundingdino_swint_ogc.pth

if [ "$MODE" = "fast" ]; then
  echo "== SAM 2.1 (tiny) =="
  DL "$SAM/sam2.1_hiera_tiny.pt" sam2.1_hiera_tiny.pt
else
  echo "== SAM 2.1 (large) =="
  DL "$SAM/sam2.1_hiera_large.pt" sam2.1_hiera_large.pt
fi

if [ "$MODE" = "all" ]; then
  echo "== extra SAM 2.1 variants =="
  DL "$SAM/sam2.1_hiera_tiny.pt"      sam2.1_hiera_tiny.pt
  DL "$SAM/sam2.1_hiera_small.pt"     sam2.1_hiera_small.pt
  DL "$SAM/sam2.1_hiera_base_plus.pt" sam2.1_hiera_base_plus.pt
  echo "== Grounding DINO (Swin-B) =="
  DL "$GD/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" groundingdino_swinb_cogcoor.pth
fi

echo ""
echo "Done. Files in: $DIR"
ls -lh "$DIR"/*.pt "$DIR"/*.pth 2>/dev/null || true
