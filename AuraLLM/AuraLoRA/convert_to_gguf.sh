#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGED_DIR="${SCRIPT_DIR}/models/aura_es_merged"
BASE_GGUF="${SCRIPT_DIR}/models/aura.gguf"
OUTFILE="${SCRIPT_DIR}/models/aura_final.gguf"

if [[ ! -f "${BASE_GGUF}" ]]; then
  echo "No se encontró el archivo base GGUF en ${BASE_GGUF}."
  echo "Ejecuta primero download_base_gguf.py para descargar RigoChat-7B-v2." >&2
  exit 1
fi

echo "Convirtiendo modelo fusionado a formato GGUF..."
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}";
CONVERT_LORA_SCRIPT=""

if [[ -n "${LLAMA_CPP_DIR}" ]]; then
  CONVERT_LORA_SCRIPT="${LLAMA_CPP_DIR}/convert-lora-to-gguf.py"
fi

if [[ -f "${CONVERT_LORA_SCRIPT}" ]]; then
  echo "Usando convert-lora-to-gguf.py desde ${CONVERT_LORA_SCRIPT}."
  python "${CONVERT_LORA_SCRIPT}" \
    --base "${BASE_GGUF}" \
    --lora "${MERGED_DIR}" \
    --outfile "${OUTFILE}" \
    --arch llama
else
  echo "No se encontró convert-lora-to-gguf.py. Se realizará conversión directa desde pesos fusionados."
  CONVERT_PY="convert.py"
  if [[ -n "${LLAMA_CPP_DIR}" ]] && [[ -f "${LLAMA_CPP_DIR}/convert.py" ]]; then
    CONVERT_PY="${LLAMA_CPP_DIR}/convert.py"
  fi
  python "${CONVERT_PY}" \
    --model "${MERGED_DIR}" \
    --outfile "${OUTFILE}" \
    --format gguf \
    --outtype f16
fi

echo "Conversión completada. Archivo guardado en ${OUTFILE}"
