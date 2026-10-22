#!/bin/bash
# Submit from the HPC MAS directory:
#   sbatch scripts/run_shuffled_null_diagnostic.slurm [JOB_DIR] [SUBSET] [N_SHUFFLES]
#
# Examples:
#   # Auto-detect latest source-volume-beta job and evaluate best subset:
#   sbatch scripts/run_shuffled_null_diagnostic.slurm
#
#   # Specify explicit batch folder:
#   sbatch scripts/run_shuffled_null_diagnostic.slurm /scratch/g/bgross/CIDMEG_SM/data/CIDMEG005/ICA/decoding_results/cidmeg005_s1_run1_raw_embedding_contrastive_..._selected_parcel_subsets best 50
#
#   # Evaluate all 19 candidate parcels with 100 permutations:
#   sbatch scripts/run_shuffled_null_diagnostic.slurm /path/to/batch all 100
#
# Or run interactively:
#   python scripts/run_shuffled_null_diagnostic.py --job-dir /path/to/batch
#
#SBATCH --job-name=null-diag
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/shuffled-null-diagnostic-%j.log
#SBATCH --export=ALL

set -euo pipefail

cd /scratch/g/bgross/CIDMEG_SM/mas

PYTHON_EXE="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

JOB_DIR="${1:-}"
SUBSET="${2:-best}"
N_SHUFFLES="${3:-50}"

EXTRA_ARGS=()
if [[ -n "$JOB_DIR" ]]; then
  EXTRA_ARGS+=(--job-dir "$JOB_DIR")
fi
EXTRA_ARGS+=(--subset "$SUBSET" --n-shuffles "$N_SHUFFLES")

# Allow passing additional CLI flags after positionals
if [[ $# -gt 3 ]]; then
  shift 3
  EXTRA_ARGS+=("$@")
fi

echo "============================================================"
echo "Starting Shuffled-Null Retraining Diagnostic"
echo "Job ID:        ${SLURM_JOB_ID:-manual}"
echo "Python:        ${PYTHON_EXE}"
echo "Target Job:    ${JOB_DIR:-auto-detect}"
echo "Subset:        ${SUBSET}"
echo "Permutations:  ${N_SHUFFLES}"
echo "============================================================"

exec "$PYTHON_EXE" -u scripts/run_shuffled_null_diagnostic.py "${EXTRA_ARGS[@]}"
