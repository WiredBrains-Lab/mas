#!/usr/bin/env bash
#
# Exports all subset_comparison.csv and manifest.json files from
# exhaustive screening runs, renaming each with its unique run/parameter signature.
#
# Usage:
#   bash scripts/export_screening_results.sh [DESTINATION_DIR]
#
# Default destination is $HOME/saved_screening_results (safe from scratch cleanup/quota)
#

set -euo pipefail

DEST_DIR="${1:-$HOME/saved_screening_results}"
mkdir -p "$DEST_DIR"

echo "============================================================"
echo "Exporting Exhaustive Screening Results"
echo "Destination: $DEST_DIR"
echo "Start time:  $(date)"
echo "============================================================"

# Known search locations for decoding results to avoid slow whole-scratch scans
SEARCH_ROOTS=(
  "/scratch/g/bgross/CIDMEG_SM/data/CIDMEG005/ICA/decoding_results"
  "/scratch/g/bgross/CIDMEG_SM/data/CIDMEG005/decoding_results"
  "/scratch/g/bgross/CIDMEG_SM/mas/decoding_results"
  "/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming/decoding_results_source"
  "/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming/decoding_results"
)

found_count=0

for root in "${SEARCH_ROOTS[@]}"; do
  [[ -d "$root" ]] || continue

  # -maxdepth 3 ensures we find subset_comparison.csv without recursing into
  # thousands of individual worker subfolders (*_parallel_*)
  while IFS= read -r csv; do
    [[ -n "$csv" ]] || continue

    run_dir="$(basename "$(dirname "$csv")")"
    parent_dir="$(basename "$(dirname "$(dirname "$csv")")")"

    # Construct descriptive filename prefix
    if [[ "$parent_dir" != "decoding_results" && "$parent_dir" != "CIDMEG_SM" && "$parent_dir" != "." ]]; then
      prefix="${parent_dir}__${run_dir}"
    else
      prefix="${run_dir}"
    fi

    dest_csv="${DEST_DIR}/subset_comparison__${prefix}.csv"
    echo "[FOUND] $(dirname "$csv")"
    echo "  -> Copying CSV: $(basename "$dest_csv")"
    cp "$csv" "$dest_csv"

    # Also preserve the run manifest (contains best model parameters & accuracy)
    manifest="$(dirname "$csv")/manifest.json"
    if [[ -f "$manifest" ]]; then
      dest_manifest="${DEST_DIR}/manifest__${prefix}.json"
      echo "  -> Copying Manifest: $(basename "$dest_manifest")"
      cp "$manifest" "$dest_manifest"
    fi

    # Also copy settings used if available
    settings="$(dirname "$csv")/settings_used.json"
    if [[ -f "$settings" ]]; then
      dest_settings="${DEST_DIR}/settings__${prefix}.json"
      cp "$settings" "$dest_settings"
    fi

    found_count=$((found_count + 1))
    echo ""
  done < <(find "$root" -maxdepth 3 -type f -name "subset_comparison.csv" 2>/dev/null)
done

# Fallback: if none found in standard paths, search MAS root with depth limit
if [[ "$found_count" -eq 0 ]]; then
  echo "Checking general /scratch/g/bgross/CIDMEG_SM directory..."
  while IFS= read -r csv; do
    [[ -n "$csv" ]] || continue
    run_dir="$(basename "$(dirname "$csv")")"
    parent_dir="$(basename "$(dirname "$(dirname "$csv")")")"
    prefix="${parent_dir}__${run_dir}"
    cp "$csv" "${DEST_DIR}/subset_comparison__${prefix}.csv"
    [[ -f "$(dirname "$csv")/manifest.json" ]] && cp "$(dirname "$csv")/manifest.json" "${DEST_DIR}/manifest__${prefix}.json"
    found_count=$((found_count + 1))
    echo "[FOUND] Copied $csv -> ${DEST_DIR}/subset_comparison__${prefix}.csv"
  done < <(find /scratch/g/bgross/CIDMEG_SM -maxdepth 6 -type f -name "subset_comparison.csv" 2>/dev/null)
fi

echo "============================================================"
echo "Export complete: ${found_count} runs exported to ${DEST_DIR}"
echo "============================================================"
