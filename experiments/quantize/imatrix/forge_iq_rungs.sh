#!/usr/bin/env bash
# Ornith-1.5-9B-MTP — imatrix (i-quant) rungs for small-VRAM cards.
#
# Requested on the hub: protoLabsAI/Ornith-1.5-9B-MTP-GGUF discussion #2,
# "can u make soemthing like iq4xxs or in iq3 for 6 gb vram people".
#
# Mirrors the ladder we shipped for Ornith-1.0-9B-MTP-GGUF: IQ4_XS / IQ3_M / IQ2_M,
# bundled (trunk + nextn head in one file), with the MTP head PINNED TO Q8_0 so draft
# acceptance does not fall off with the trunk. The head is only 15 tensors (~0.5 GB in
# bf16), so pinning it costs almost nothing in file size and is the difference between
# a usable and a useless draft on a 3-bit trunk.
set -euo pipefail

FORGE=/mnt/data/gguf-forge/Ornith-1.5-9B-MTP
BIN=/home/ava/dev/llama.cpp/build-cuda/bin/llama-quantize
SRC="$FORGE/out/Ornith-1.5-9B-MTP-BF16.gguf"
IMAT="$FORGE/imatrix/ornith-1.5-9b.imatrix"
THREADS=${THREADS:-16}

for RUNG in IQ4_XS IQ3_M IQ2_M; do
  DST="$FORGE/out/Ornith-1.5-9B-MTP-${RUNG}.gguf"
  if [[ -f "$DST" ]]; then echo "skip $RUNG (exists)"; continue; fi
  echo "=== $RUNG ==="
  "$BIN" \
    --imatrix "$IMAT" \
    --tensor-type 'blk\.32\.=q8_0' \
    "$SRC" "$DST" "$RUNG" "$THREADS"
  ls -l "$DST"
done
