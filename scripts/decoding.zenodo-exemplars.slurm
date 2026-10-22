#!/bin/bash
# Submit from the HPC MAS directory:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.zenodo-exemplars.slurm
#
# Runs 6-CLASS EXEMPLAR decoding (sushi vs dumplings vs cookies vs comb vs knife vs pencil)
# on the Zenodo MEG Picture Naming Dataset (Step 1 visual presentation). Chance is 16.7%.
#
#SBATCH --job-name=meg-zenodo-exemplars
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-exemplars-%j.log
#SBATCH --export=ALL

set -euo pipefail
cd /scratch/g/bgross/CIDMEG_SM/mas

# Locate Python environment
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
TRIAL_TABLE="${DATA_DIR}/trials_step1_visual.csv"

echo "============================================================"
echo "Starting 6-CLASS EXEMPLAR Decoding Job"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "CPUs allocated:   ${SLURM_CPUS_PER_TASK:-4}"
echo "Memory requested: 8 GB"
echo "Classes:          6 exemplars (chance = 16.7%)"
echo "Python:           $("$decoding_python" --version)"
echo "Working dir:      $PWD"
echo "============================================================"

# Step 1: Ensure trial table exists
if [[ ! -f "$TRIAL_TABLE" ]]; then
  echo ""
  echo "--> Trial table not found. Extracting event triggers and building trial tables..."
  "$decoding_python" -u scripts/prepare_zenodo_picture_naming_trials.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR"
fi

# Step 2: Run MAS 6-class exemplar decoding CLI
echo ""
echo "--> Running MAS 6-Class Exemplar Decoding..."
"$decoding_python" -I -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.zenodo-exemplars.hpc.toml \
  --torch-num-threads "$OMP_NUM_THREADS" \
  --force \
  "$@"

# Step 3: Log peak memory
echo ""
echo "============================================================"
echo "Exemplar decoding completed: $(date)"
if [[ -f /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]]; then
  peak_mb=$(( $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes) / 1024 / 1024 ))
  echo "Slurm cgroup peak memory used: ${peak_mb} MB (out of 8192 MB requested)"
fi
echo "============================================================"
