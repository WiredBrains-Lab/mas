#!/bin/bash
# Submit from the HPC MAS root or directly with sbatch:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/download_meg_zenodo.slurm
#
# Downloads the MEG Picture Naming Task dataset (Zenodo Record 15789569)
# to /scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming
#
#SBATCH --job-name=download-meg-zenodo
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/download-meg-zenodo-%j.log
#SBATCH --export=ALL

set -euo pipefail

# Target download directory
TARGET_DATA_DIR="${TARGET_DATA_DIR:-/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming}"
mkdir -p "$TARGET_DATA_DIR"
cd "$TARGET_DATA_DIR"

echo "============================================================"
echo "Starting MEG Zenodo Download"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "Target Directory: $(pwd)"
echo "Start Time:       $(date)"
echo "============================================================"

# Subject ZIP archives on Zenodo (Record 15789569)
# Total size: ~4.5 GB across 7 subjects
SUBJECTS=("1002" "1003" "1004" "1005" "1006" "1008" "1010")

for sub in "${SUBJECTS[@]}"; do
  zip_name="${sub}.zip"
  download_url="https://zenodo.org/api/records/15789569/files/${zip_name}/content"

  echo ""
  echo "------------------------------------------------------------"
  echo "[$(date '+%H:%M:%S')] Processing Subject ${sub}..."
  echo "URL: ${download_url}"
  echo "------------------------------------------------------------"

  # -L: follow redirects
  # -C -: resume partially downloaded file
  # --retry: retry transient network dropouts
  curl -L -C - --retry 5 --retry-delay 5 -o "$zip_name" "$download_url"

  # Verify ZIP integrity before extracting
  echo "[$(date '+%H:%M:%S')] Verifying ${zip_name} integrity..."
  if unzip -tq "$zip_name"; then
    echo "[$(date '+%H:%M:%S')] Archive ${zip_name} is valid. Extracting..."
    unzip -q -o "$zip_name"
    echo "[$(date '+%H:%M:%S')] Successfully extracted ${sub}/"
  else
    echo "[ERROR] ${zip_name} failed integrity check!" >&2
    exit 1
  fi
done

echo ""
echo "============================================================"
echo "All subjects downloaded and extracted successfully."
echo "End Time:         $(date)"
echo "Directory Size:   $(du -sh "$TARGET_DATA_DIR" | cut -f1)"
echo "============================================================"
