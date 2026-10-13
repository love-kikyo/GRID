#!/bin/bash
#SBATCH --job-name=GRID-SIM-Probe
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
cd /mnt/beegfs/home/MMReco2021/liuyu/GRID
[[ $# == 1 && $1 =~ ^v[0-9]+$ ]] || exit 2
: "${SOURCE_GIT_COMMIT:?Missing source binding}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export GRID_EXPORT_CPU_BUDGET=4 GRID_CPU_BUDGET=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE USE_TF=0
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-${USER}-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,cores --kill-on-bad-exit=1 \
 /home/MMReco2021/.conda/envs/GRID/bin/python -m scripts.probe_sim_teacher \
 --output "logs/training_launch/sim_seed42_validation_2026-10-13/teacher_probe_$1.json"
