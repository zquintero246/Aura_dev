#!/bin/bash
# Script auxiliar para lanzar entrenamiento distribuido con torchrun.
# NODE_RANK=0 actúa como maestro y debe ejecutarse en la máquina con MASTER_ADDR.
# Cada nodo debe exportar las mismas variables MASTER_ADDR/MASTER_PORT/NNODES
# y ajustar NODE_RANK y GPUS_PER_NODE según corresponda.

set -euo pipefail

: "${MASTER_ADDR:?Variable MASTER_ADDR no definida}"
: "${MASTER_PORT:?Variable MASTER_PORT no definida}"
: "${NNODES:?Variable NNODES no definida}"
: "${NODE_RANK:?Variable NODE_RANK no definida}"
: "${GPUS_PER_NODE:?Variable GPUS_PER_NODE no definida}"

exec torchrun \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train_gpt2_spanish.py
