#!/bin/bash
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/semantic_decision_windows/array-%A_%a.log
#SBATCH --array=1-37

set -euo pipefail
cd /scratch/g/bgross/CIDMEG_SM/mas
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

# Override this at submission time with MAS_PYTHON if the environment lives
# elsewhere, for example: sbatch --export=ALL,MAS_PYTHON=/path/to/python ...
python_bin="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$python_bin" ]]; then
echo "Python executable not found: $python_bin" >&2
exit 2
fi

SWEEP_OUTPUT="/scratch/g/bgross/CIDMEG_SM/semantic_decision_windows"
DECODING_CONFIG="configs/decoding.hpc.toml"

"$python_bin" -I -u scripts/run_temporal_window_sweep.py \
--sweep-output "$SWEEP_OUTPUT" \
--sweep-start -0.2 \
--sweep-stop 0.6 \
--window-width-ms 50 \
--step-ms 20 \
--task-index "${SLURM_ARRAY_TASK_ID}" \
-- \
--config "$DECODING_CONFIG" \
--torch-num-threads "$OMP_NUM_THREADS" \
--no-plots

# After the array completes:
# "$python_bin" -I scripts/run_temporal_window_sweep.py \
#   --sweep-output "$SWEEP_OUTPUT" --collect-only