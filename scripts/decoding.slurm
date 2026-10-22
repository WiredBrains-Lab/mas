#!/bin/bash
# Submit from the MAS root after activating the RCC Python environment.
# Edit the account, memory, CPUs and wall time for your allocation/workload.
#SBATCH --job-name=meg-decoding
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the MAS project directory}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

# Copy configs/decoding.toml to configs/decoding.rcc.toml and edit server paths.
# Set all_runs = true in that config to include neighboring raw recordings.
# Slurm jobs never prompt; omit/disable it to use only the configured inputs.
# The Python executable comes from the environment active at submission.
python -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.rcc.toml \
  --torch-num-threads "$OMP_NUM_THREADS"

# For a parallel exhaustive RESUME, replace the Python command above with:
# python -u scripts/run_concept_decoding_cli.py --resume /path/to/batch \
#   --subset-workers 4 --subset-worker-threads 1
# Match cpus-per-task to workers * worker-threads; increase memory as needed.
