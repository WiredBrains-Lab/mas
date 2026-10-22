#!/bin/bash
# HPC paths/environment match temporal_decoding_concepts_array.slurm.
# Submit: sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.source-volume-beta.slurm
# For input validation only, append --dry-run after the script path.
# Adjust memory and wall time for your data and allocation.
# One 48-CPU node: 12 source workers x 4 threads, then 48 subset workers x 1.
# Normal nodes have 48 CPUs / 360 GB; 7G per CPU requests 336G total.
# RCC guidance: https://docs.rcc.mcw.edu/cluster/jobs/running-jobs/
# One Python coordinator spawns local processes within this task allocation.
# No GPU required. Memory must cover the peak worker working sets.
#SBATCH --job-name=meg-volume-beta
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=7G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-source-volume-beta-%j.log
#SBATCH --export=ALL

set -euo pipefail
cd /scratch/g/bgross/CIDMEG_SM/mas

# Override with sbatch --export=ALL,MAS_PYTHON=/absolute/path/to/python ...
decoding_python="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$decoding_python" ]]; then
  echo "Python executable not found: $decoding_python" >&2
  exit 2
fi
[[ -f scripts/run_concept_decoding_cli.py && -f configs/decoding.source-volume-beta.hpc.toml ]] || {
  echo 'The HPC MAS project directory must contain scripts/ and configs/.' >&2
  exit 2
}
# Override worker counts through sbatch --export=ALL,MAS_SOURCE_WORKERS=...,....
source_workers="${MAS_SOURCE_WORKERS:-12}"
source_worker_threads="${MAS_SOURCE_WORKER_THREADS:-4}"
subset_workers="${MAS_SUBSET_WORKERS:-48}"
subset_worker_threads="${MAS_SUBSET_WORKER_THREADS:-1}"
allocated_cpus="${SLURM_CPUS_PER_TASK:-0}"
for worker_setting in "$source_workers" "$source_worker_threads" "$subset_workers" "$subset_worker_threads"; do
  [[ "$worker_setting" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Worker counts and threads must be positive integers.' >&2
    exit 2
  }
done
[[ "$allocated_cpus" -ge "$((source_workers * source_worker_threads))" &&
   "$allocated_cpus" -ge "$((subset_workers * subset_worker_threads))" ]] || {
  echo 'Allocated CPUs must cover workers x threads for both extraction and subset search.' >&2
  exit 2
}

# Prevent each worker from spawning a full allocation of numerical threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

echo "Job ${SLURM_JOB_ID:-unknown}: ${allocated_cpus} CPUs; source ${source_workers} x ${source_worker_threads}, subsets ${subset_workers} x ${subset_worker_threads}."
echo "Python: $(command -v "$decoding_python")"
echo "Working directory: $PWD"

# Pin these settings to match the Slurm allocation. The TOML supplies inputs,
# the semantic ROI candidate pool, spectral features, and decoder settings.
# The TOML selects 19 left semantic ROIs: 524,287 exhaustive models.
# Append --source-hemisphere both to include both hemispheres.
# The TOML enables a persistent all-ROI feature cache shared across fresh jobs.
# The log reports the output/checkpoint directory for later CLI --resume runs.
exec "$decoding_python" -I -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.source-volume-beta.hpc.toml \
  --subset-search exhaustive \
  --source-workers "$source_workers" \
  --source-worker-threads "$source_worker_threads" \
  --subset-workers "$subset_workers" \
  --subset-worker-threads "$subset_worker_threads" \
  --torch-num-threads 1 \
  "$@"
