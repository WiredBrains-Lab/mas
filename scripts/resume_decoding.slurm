#!/bin/bash
#SBATCH --job-name=meg-subsets
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/resume-decoding-%j.log

# Submit from the MAS directory. Supply your RCC account and batch folder:
# sbatch --account=YOUR_PI_NETID scripts/resume_decoding.slurm /path/to/batch
# Optional runtime arguments after the batch path override the defaults below.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the MAS project directory}"
batch_dir=${1:?Supply the exhaustive-screening batch folder as the first argument}
shift
python_bin="$HOME/.conda/envs/mas-decoding-rcc/bin/python"
[[ -x "$python_bin" ]] || { echo "Environment Python missing: $python_bin" >&2; exit 1; }
[[ -f "$batch_dir/manifest.json" ]] || { echo "Batch manifest missing: $batch_dir/manifest.json" >&2; exit 1; }
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Explicit Python path avoids the shared-base PATH issue on RCC.
# One task coordinates all worker processes on the allocated node.
exec "$python_bin" -I -u scripts/run_concept_decoding_cli.py \
  --no-config \
  --resume "$batch_dir" \
  --subset-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --subset-worker-threads 1 \
  --torch-num-threads 1 \
  "$@"
