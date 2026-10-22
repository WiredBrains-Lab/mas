#!/bin/bash
# Submit from the HPC MAS directory:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.zenodo-temporal.slurm
#
# Runs TEMPORAL WINDOW SWEEP (sliding 50 ms windows, 25 ms step, from -200 ms to +800 ms)
# to track the millisecond-by-millisecond timecourse of food vs. tool decoding.
# Automatically aggregates results into sweep_summary.csv and generates the timecourse plot.
#
#SBATCH --job-name=meg-zenodo-temporal
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-temporal-%j.log
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
SWEEP_OUTPUT="${DATA_DIR}/temporal_sweep_category"
mkdir -p "$SWEEP_OUTPUT"

echo "============================================================"
echo "Starting TEMPORAL WINDOW SWEEP Decoding Job"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "CPUs allocated:   ${SLURM_CPUS_PER_TASK:-4}"
echo "Memory requested: 8 GB"
echo "Sweep range:      -200 ms to +800 ms (50 ms width, 25 ms step)"
echo "Output directory: $SWEEP_OUTPUT"
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

# Step 2: Run MAS temporal window sweep wrapper
echo ""
echo "--> Running MAS Temporal Window Sweep across all sliding windows..."
"$decoding_python" -I -u scripts/run_temporal_window_sweep.py \
  --sweep-output "$SWEEP_OUTPUT" \
  --sweep-start -0.2 \
  --sweep-stop 0.8 \
  --window-width-ms 50 \
  --step-ms 25 \
  -- \
  --config configs/decoding.zenodo-temporal.hpc.toml \
  --torch-num-threads "$OMP_NUM_THREADS" \
  --force \
  "$@"

# Step 3: Log peak memory
echo ""
echo "============================================================"
echo "Temporal sweep completed: $(date)"
echo "Results summary: ${SWEEP_OUTPUT}/sweep_summary.csv"
if [[ -f /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]]; then
  peak_mb=$(( $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes) / 1024 / 1024 ))
  echo "Slurm cgroup peak memory used: ${peak_mb} MB (out of 8192 MB requested)"
fi
echo "============================================================"
