#!/bin/bash
# HPC paths/environment match temporal_decoding_concepts_array.slurm.
# Submit: sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.source-surface-beta-aparc.slurm
# For input validation only, append --dry-run after the script path:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.source-surface-beta-aparc.slurm --dry-run
#
# Runs SURFACE-SPACE concept decoding using the Desikan-Killiany cortical parcellation
# (aparc, 34 cortical regions per hemisphere) and beta band (13–30 Hz).
# Default left semantic filter yields 17 cortical ROIs (131,071 exhaustive models).
#
# One 48-CPU node: 12 source workers x 4 threads for inverse extraction, then 48 subset workers x 1.
# Normal nodes have 48 CPUs / 360 GB; 7G per CPU requests 336G total.
# No GPU required.
#SBATCH --job-name=meg-surf-aparc-beta
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=7G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-source-surface-beta-aparc-%j.log
#SBATCH --export=ALL

set -euo pipefail
cd /scratch/g/bgross/CIDMEG_SM/mas

decoding_python="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$decoding_python" ]]; then
  echo "Python executable not found: $decoding_python" >&2
  exit 2
fi
[[ -f scripts/run_concept_decoding_cli.py && -f configs/decoding.source-hcp-combined-beta.hpc.toml ]] || {
  echo 'The HPC MAS project directory must contain scripts/ and configs/.' >&2
  exit 2
}

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

# Uses surface inverses (*-inv.fif) in native subject space with FreeSurfer aparc.
# Default left semantic filter -> 17 cortical ROIs (131,071 exhaustive models).
exec "$decoding_python" -I -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.source-hcp-combined-beta.hpc.toml \
  --source-parcellation aparc \
  --source-common-space native \
  --subset-search exhaustive \
  --source-workers "$source_workers" \
  --source-worker-threads "$source_worker_threads" \
  --subset-workers "$subset_workers" \
  --subset-worker-threads "$subset_worker_threads" \
  --torch-num-threads 1 \
  "$@"
