#!/bin/bash
# Submit from the HPC MAS directory:
#   sbatch scripts/inspect_trials_per_decision.slurm
#
# Or run interactively on a compute or login node:
#   python scripts/inspect_trials_per_decision.py --config configs/decoding.source-volume-beta.hpc.toml
#
# Inspects the distribution of trials across concepts and semantic decision tasks.
# Outputs:
#   - Summary metrics to stdout / slurm log
#   - concept_by_decision_trial_counts.csv
#   - decision_distribution_summary.json
#
#SBATCH --job-name=inspect-trials
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/inspect-trials-%j.log
#SBATCH --export=ALL

set -eo pipefail

cd /scratch/g/bgross/CIDMEG_SM/mas

PYTHON_EXE="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
CONFIG="${1:-configs/decoding.source-volume-beta.hpc.toml}"
OUTPUT_DIR="${2:-/scratch/g/bgross/CIDMEG_SM/mas}"

echo "============================================================"
echo "Inspecting trial distribution across semantic decisions"
echo "Job ID:     ${SLURM_JOB_ID:-manual}"
echo "Python:     ${PYTHON_EXE}"
echo "Config:     ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================================"

"$PYTHON_EXE" -u scripts/inspect_trials_per_decision.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"
