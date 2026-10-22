#!/bin/bash
# Submit from the HPC MAS directory:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.zenodo-picture-naming.slurm
#
# Runs MAS concept decoding on the Zenodo MEG Picture Naming Dataset (Record 15789569).
# Automatically prepares the trial table if not already present.
#
#SBATCH --job-name=meg-zenodo-decoding
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-zenodo-%j.log
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

DATA_DIR="/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming"
TRIAL_TABLE="${DATA_DIR}/trials_all_steps.csv"

echo "============================================================"
echo "Starting MEG Zenodo Concept Decoding Job"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "CPUs allocated:   ${SLURM_CPUS_PER_TASK:-1}"
echo "Python:           $("$decoding_python" --version)"
echo "Working dir:      $PWD"
echo "Data dir:         $DATA_DIR"
echo "============================================================"

# Step 1: Ensure trial table exists (extract from raw FIFs if needed)
if [[ ! -f "$TRIAL_TABLE" ]]; then
  echo ""
  echo "--> Trial table not found. Extracting event triggers and building trial tables..."
  "$decoding_python" -u scripts/prepare_zenodo_picture_naming_trials.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR"
fi

# Step 2: Run MAS concept decoding CLI
echo ""
echo "--> Running MAS Concept Decoding..."
"$decoding_python" -I -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.zenodo-picture-naming.hpc.toml \
  --torch-num-threads "$OMP_NUM_THREADS" \
  "$@"

# Step 3: Log peak memory
echo ""
echo "============================================================"
echo "Decoding completed: $(date)"
if [[ -f /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]]; then
  peak_mb=$(( $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes) / 1024 / 1024 ))
  echo "Slurm cgroup peak memory used: ${peak_mb} MB (out of 8192 MB requested)"
fi
echo "============================================================"
