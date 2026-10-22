#!/bin/bash
# Submit from the HPC MAS directory:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.zenodo-source.slurm
#
# Runs SOURCE-SPACE concept decoding on the Zenodo MEG Picture Naming Dataset.
# Right-sized for source parcel extraction: 8 CPUs, 32 GB RAM.
# (4 source workers x 2 threads; each worker utilizes ~5-6 GB for inverse solutions).
#
#SBATCH --job-name=meg-zenodo-source
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-source-%j.log
#SBATCH --export=ALL

set -euo pipefail
cd /scratch/g/bgross/CIDMEG_SM/mas

# Locate Python environment (defaults to mas-decoding-rcc)
decoding_python="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$decoding_python" ]]; then
  echo "Python executable not found: $decoding_python" >&2
  echo "Ensure mas-decoding-rcc conda environment is installed." >&2
  exit 2
fi

# Worker and threading layout
source_workers=4
source_worker_threads=2
allocated_cpus="${SLURM_CPUS_PER_TASK:-8}"

export OMP_NUM_THREADS="$source_worker_threads"
export MKL_NUM_THREADS="$source_worker_threads"
export OPENBLAS_NUM_THREADS="$source_worker_threads"
export NUMEXPR_NUM_THREADS="$source_worker_threads"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

DATA_DIR="/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming"
TRIAL_TABLE="${DATA_DIR}/trials_step1_visual.csv"

echo "============================================================"
echo "Starting SOURCE-SPACE Decoding Job"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "CPUs allocated:   ${allocated_cpus} (${source_workers} workers x ${source_worker_threads} threads)"
echo "Memory requested: 32 GB"
echo "Python:           $("$decoding_python" --version)"
echo "Working dir:      $PWD"
echo "============================================================"

# Step 1: Ensure trial table exists (extract from raw FIFs if needed)
if [[ ! -f "$TRIAL_TABLE" ]]; then
  echo ""
  echo "--> Trial table not found. Extracting event triggers and building trial tables..."
  "$decoding_python" -u scripts/prepare_zenodo_picture_naming_trials.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR"
fi

# Step 2: Ensure FreeSurfer anatomy and inverse operators exist
MANIFEST="configs/decoding.source.zenodo.hpc.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo ""
  echo "--> Source manifest/inverse operators not found."
  echo "--> Setting up FreeSurfer fsaverage template and generating inverse operators..."
  "$decoding_python" -u scripts/create_zenodo_inverse_operators.py \
    --data-dir "$DATA_DIR" \
    --subjects-dir "/scratch/g/bgross/CIDMEG_SM/data/freesurfer" \
    --output-manifest "$MANIFEST"
fi

# Step 3: Run MAS source-space concept decoding CLI
echo ""
echo "--> Running MAS Source-Space Concept Decoding..."
"$decoding_python" -I -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.zenodo-source.hpc.toml \
  --source-workers "$source_workers" \
  --source-worker-threads "$source_worker_threads" \
  --torch-num-threads "$source_worker_threads" \
  --force \
  "$@"

# Step 3: Log exact peak memory utilized by the job
echo ""
echo "============================================================"
echo "Source-space decoding completed: $(date)"
if [[ -f /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]]; then
  peak_mb=$(( $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes) / 1024 / 1024 ))
  echo "Slurm cgroup peak memory used: ${peak_mb} MB (out of 32768 MB requested)"
fi
echo "============================================================"
