#!/bin/bash
# Submit from the HPC MAS root or directly with sbatch:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/download_auditory_naming_openneuro.slurm
#
# Downloads the Auditory Naming iEEG dataset (OpenNeuro ds005545, version 1.0.3)
# to /scratch/g/bgross/CIDMEG_SM/data/auditory_naming_ds005545
# Dataset reference: https://openneuro.org/datasets/ds005545/versions/1.0.3
#
#SBATCH --job-name=download-auditory-naming
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/download-auditory-naming-%j.log
#SBATCH --export=ALL

set -euo pipefail

DATASET_ID="ds005545"
DATASET_VERSION="1.0.3"
S3_URI="s3://openneuro.org/${DATASET_ID}"

# Target download directory (can be overridden via TARGET_DATA_DIR env var)
TARGET_DATA_DIR="${TARGET_DATA_DIR:-/scratch/g/bgross/CIDMEG_SM/data/auditory_naming_${DATASET_ID}}"
mkdir -p "$TARGET_DATA_DIR"
cd "$TARGET_DATA_DIR"

echo "============================================================"
echo "Starting OpenNeuro Dataset Download: ${DATASET_ID} (v${DATASET_VERSION})"
echo "Task:             Auditory Descriptive Naming (iEEG / ECoG)"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "Target Directory: $(pwd)"
echo "S3 Source:        ${S3_URI}"
echo "Start Time:       $(date)"
echo "============================================================"

# Resolve AWS CLI or openneuro-py client
AWS_CMD=""
if command -v aws >/dev/null 2>&1; then
  AWS_CMD="$(command -v aws)"
elif [[ -x "$HOME/.local/bin/aws" ]]; then
  AWS_CMD="$HOME/.local/bin/aws"
elif [[ -x "$HOME/.conda/envs/mas-decoding-rcc/bin/aws" ]]; then
  AWS_CMD="$HOME/.conda/envs/mas-decoding-rcc/bin/aws"
else
  # Try loading cluster module if environment modules are available
  if command -v module >/dev/null 2>&1; then
    module load awscli 2>/dev/null || module load aws-cli 2>/dev/null || true
    if command -v aws >/dev/null 2>&1; then
      AWS_CMD="$(command -v aws)"
    fi
  fi
fi

# Locate Python environment for fallback
PYTHON_CMD="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$PYTHON_CMD" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python)"
  fi
fi

if [[ -n "$AWS_CMD" ]]; then
  echo "[$(date '+%H:%M:%S')] Using AWS CLI: $AWS_CMD"
  echo "[$(date '+%H:%M:%S')] Syncing from ${S3_URI} to ${TARGET_DATA_DIR}..."
  "$AWS_CMD" s3 sync --no-sign-request "$S3_URI" "$TARGET_DATA_DIR"
elif command -v openneuro-py >/dev/null 2>&1; then
  echo "[$(date '+%H:%M:%S')] Using openneuro-py CLI..."
  openneuro-py download --dataset="${DATASET_ID}" --tag="${DATASET_VERSION}" --target-dir="$TARGET_DATA_DIR"
elif [[ -x "$PYTHON_CMD" ]]; then
  echo "[$(date '+%H:%M:%S')] AWS CLI not found in PATH. Checking Python environment: $PYTHON_CMD"
  if "$PYTHON_CMD" -c "import openneuro" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Using Python openneuro package..."
    "$PYTHON_CMD" -c "import openneuro as on; on.download(dataset='${DATASET_ID}', tag='${DATASET_VERSION}', target_dir='${TARGET_DATA_DIR}')"
  else
    echo "[$(date '+%H:%M:%S')] Installing awscli into user environment..."
    "$PYTHON_CMD" -m pip install --quiet --user awscli
    export PATH="$HOME/.local/bin:$PATH"
    if command -v aws >/dev/null 2>&1; then
      aws s3 sync --no-sign-request "$S3_URI" "$TARGET_DATA_DIR"
    else
      echo "[ERROR] Failed to locate or install AWS CLI / openneuro-py!" >&2
      exit 1
    fi
  fi
else
  echo "[ERROR] Neither aws CLI, openneuro-py, nor a valid Python interpreter was found." >&2
  exit 1
fi

echo ""
echo "------------------------------------------------------------"
echo "[$(date '+%H:%M:%S')] Verifying downloaded dataset..."
echo "------------------------------------------------------------"

if [[ -f "dataset_description.json" ]]; then
  echo "[OK] Found dataset_description.json:"
  cat dataset_description.json
  echo ""
else
  echo "[WARNING] dataset_description.json not found!" >&2
fi

if [[ -f "CHANGES" ]]; then
  echo "[OK] Dataset version history (CHANGES):"
  head -n 10 CHANGES
  echo ""
fi

if [[ -f "participants.tsv" ]]; then
  NUM_PARTICIPANTS=$(awk 'NR>1 {count++} END {print count}' participants.tsv)
  echo "[OK] Found participants.tsv with ${NUM_PARTICIPANTS} participants listed."
else
  echo "[WARNING] participants.tsv not found!" >&2
fi

SUBJECT_DIRS_COUNT=$(find . -maxdepth 1 -type d -name "sub-*" | wc -l | tr -d ' ')
echo "[OK] Found ${SUBJECT_DIRS_COUNT} subject directories (sub-*)."

echo ""
echo "============================================================"
echo "OpenNeuro ${DATASET_ID} download and verification complete."
echo "End Time:         $(date)"
echo "Directory Size:   $(du -sh "$TARGET_DATA_DIR" | cut -f1)"
echo "============================================================"
