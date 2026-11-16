#!/bin/bash
# External controller only: no GPU, no existing job cancellation or modification.
# Cron performs one bounded tick; Python's nonblocking lock prevents overlap.
set -euo pipefail
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
[[ $# == 2 ]] || exit 2
export PATH=/usr/local/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES='' USE_TF=0 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
exec /usr/bin/timeout 900 /usr/bin/taskset -c 2 /usr/bin/python3 -m scripts.liger_multiepoch_autorun \
  tick "$1" --authorization-sha256 "$2"
