#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGED_DIR="${SCRIPT_DIR}/models/aura_es_merged"
OUTFILE="${SCRIPT_DIR}/models/aura_es_final.gguf"

echo "Convirtiendo modelo fusionado a formato GGUF..."
python convert.py \
  --model "${MERGED_DIR}" \
  --outfile "${OUTFILE}" \
  --format gguf \
  --outtype f16

echo "Conversión completada. Archivo guardado en ${OUTFILE}"
