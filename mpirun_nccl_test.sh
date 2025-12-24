#!/bin/bash
set -euo pipefail
# -----------------------------
# Usage check
# -----------------------------
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <NP> <CUDA_VISIBLE_DEVICES> <TENANT_ID>"
  exit 1
fi


NP="$1"
VISIBLE_DEVICES="$2"
TENANT_ID="$3"

echo "Running with NP=${NP}, CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}, TENANT_ID=${TENANT_ID}"

mpirun --allow-run-as-root -np ${NP} \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
    -x NCCL_PROFILER_PLUGIN=/home/stone/lxn/NCCL-v2.28.9-1/nccl-2.28.9-1/ext-profiler/inspector/libnccl-profiler-inspector.so \
    -x NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500 \
    -x NCCL_INSPECTOR_DUMP_DIR=/home/stone/lxn/nccl_tests/inspector/${TENANT_ID} \
    -x NCCL_INSPECTOR_ENABLE=1 \
    -x NCCL_TENANT_ID=${TENANT_ID} \
    /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 128M -e 1G -f 2 -g 1 -c 0 -n 50 -d int32 -w 10
