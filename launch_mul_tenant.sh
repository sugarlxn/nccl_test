#!/bin/bash
set -euo pipefail
./mpirun_nccl_test.sh 2 1,2 1001 &
./mpirun_nccl_test.sh 2 3,4 1022 &
wait
