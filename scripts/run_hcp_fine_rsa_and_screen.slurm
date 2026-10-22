#!/bin/bash
# Submit: sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/run_hcp_fine_rsa_and_screen.slurm
#
# Pipeline:
#   Step 1: One-time extraction & cache of volumetric fine-grained HCP-MMP1 parcels
#           (HCPMMP1+aseg.mgz in subject native space, beta bandpower 13–30 Hz).
#           Reuses CIDMEG005/source_feature_cache/bandpower/ if already extracted.
#   Step 2: Fast RSA & cross-parcel redundancy prescreen (mRMR) on fine parcels.
#           Selects the top-K non-redundant fine parcels from winning macro-regions
#           (TPOJ, Medial Temporal, Lateral Temporal, Inferior Frontal, Auditory Assoc).
#   Step 3: 48-worker parallel exhaustive contrastive subset decoding (2^K - 1 models)
#           on the selected fine parcels (~2-3 minutes).
#
# Hardware: One 48-CPU node / 336 GB RAM. No GPU required.
#
#SBATCH --job-name=hcp-fine-rsa-screen
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=7G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/hcp-fine-rsa-screen-%j.log
#SBATCH --export=ALL

set -euo pipefail
cd /scratch/g/bgross/CIDMEG_SM/mas

decoding_python="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$decoding_python" ]]; then
  echo "Python executable not found: $decoding_python" >&2
  exit 2
fi

source_workers="${MAS_SOURCE_WORKERS:-12}"
source_worker_threads="${MAS_SOURCE_WORKER_THREADS:-4}"
subset_workers="${MAS_SUBSET_WORKERS:-48}"
subset_worker_threads="${MAS_SUBSET_WORKER_THREADS:-1}"
top_k_parcels="${TOP_K_PARCELS:-12}"
max_redundancy="${MAX_REDUNDANCY:-0.80}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/scratch/g/bgross/CIDMEG_SM/mas/decoding_results/hcp_fine_rsa_screen_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "HCP Fine-Grained Prescreen & BCI Target Decoding Pipeline"
echo "Job ID:            ${SLURM_JOB_ID:-manual}"
echo "CPUs allocated:    ${SLURM_CPUS_PER_TASK:-48}"
echo "Top-K Fine Targets: ${top_k_parcels} (max RDM correlation <= ${max_redundancy})"
echo "Output Directory:  ${OUTPUT_DIR}"
echo "Start Time:        $(date)"
echo "======================================================================"

# --------------------------------------------------------------------
# STEP 1: Feature Extraction / Cache Check & Fast Semantic RSA Prescreen
# --------------------------------------------------------------------
echo ""
echo "--> STEP 1: Feature Extraction / Cache Check & Semantic RSA Prescreen..."
echo "    Parcellation: HCPMMP1 (volume, beta 13-30 Hz, left hemisphere)"
echo "    Hardware:     ${source_workers} workers x ${source_worker_threads} threads"

RSA_OUT_DIR="${OUTPUT_DIR}/rsa_prescreen"
mkdir -p "$RSA_OUT_DIR"

"$decoding_python" -I -u scripts/run_parcel_rsa_prescreen.py \
  --config configs/decoding.source-volume-beta.hpc.toml \
  --source-parcellation HCPMMP1 \
  --source-workers "$source_workers" \
  --source-worker-threads "$source_worker_threads" \
  --hemisphere left \
  --top-k "$top_k_parcels" \
  --max-redundancy "$max_redundancy" \
  --max-per-macro-region 3 \
  --target-space volume \
  --output-dir "$RSA_OUT_DIR"

# --------------------------------------------------------------------
# STEP 2: Read Selected Non-Redundant Target Parcels
# --------------------------------------------------------------------
SELECTED_PARCELS=()
while IFS= read -r line || [[ -n "$line" ]]; do
  clean_p=$(echo "$line" | tr -d '\r ' | xargs)
  if [[ -n "$clean_p" && ! "$clean_p" =~ ^# ]]; then
    SELECTED_PARCELS+=("$clean_p")
  fi
done < "${RSA_OUT_DIR}/rsa_selected_parcels.txt"

if [[ "${#SELECTED_PARCELS[@]}" -eq 0 ]]; then
  echo "Error: No parcels selected by RSA prescreen." >&2
  exit 1
fi

echo ""
echo "--> Selected ${#SELECTED_PARCELS[@]} Non-Redundant Fine Parcels for Decoding:"
for p in "${SELECTED_PARCELS[@]}"; do
  echo "    - $p"
done

# --------------------------------------------------------------------
# STEP 3: 48-Worker Parallel Exhaustive Contrastive Subset Decoding
# --------------------------------------------------------------------
echo ""
TOTAL_SUBSETS=$(( (1 << ${#SELECTED_PARCELS[@]}) - 1 ))
echo "--> STEP 3: Launching 48-worker exhaustive decoding (${TOTAL_SUBSETS} models)..."

DECODING_OUT_DIR="${OUTPUT_DIR}/subset_decoding"

"$decoding_python" -I -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.source-volume-beta.hpc.toml \
  --source-parcellation HCPMMP1 \
  --source-parcels "${SELECTED_PARCELS[@]}" \
  --source-parcel-filter all \
  --subset-search exhaustive \
  --subset-workers "$subset_workers" \
  --subset-worker-threads "$subset_worker_threads" \
  --torch-num-threads 1 \
  --output-dir "$DECODING_OUT_DIR"

echo ""
echo "======================================================================"
echo "Pipeline Complete: $(date)"
echo "Summary & Selected BCI Targets: ${RSA_OUT_DIR}"
echo "Decoding Results:               ${DECODING_OUT_DIR}"
echo "======================================================================"
