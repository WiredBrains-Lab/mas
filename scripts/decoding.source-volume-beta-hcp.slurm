#!/bin/bash
# HPC paths/environment match temporal_decoding_concepts_array.slurm.
# Submit: sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.source-volume-beta-hcp.slurm
# For input validation only, append --dry-run after the script path:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/decoding.source-volume-beta-hcp.slurm --dry-run
#
# Runs VOLUME-SPACE concept decoding using the full volumetric HCP-MMP1 parcellation
# (HCPMMP1+aseg.mgz in subject native volume space, 180 cortical parcels/hemi + aseg)
# and beta band (13–30 Hz).
# Prerequisite: run scripts/create_hcp_volumetric_atlas.slurm first so that
# CIDMEG005/Anatomy/mri/HCPMMP1+aseg.mgz exists.
# Because the full HCPMMP1 semantic pool has >60 parcels per hemisphere, greedy forward
# selection (--subset-search forward) is used by default instead of 2^N exhaustive search.
#
# One 48-CPU node: 12 source workers x 4 threads for inverse extraction, then 48 subset workers x 1.
# Normal nodes have 48 CPUs / 360 GB; 7G per CPU requests 336G total.
# No GPU required.
#SBATCH --job-name=meg-vol-hcpfull-beta
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=7G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/decoding-source-volume-beta-hcp-%j.log
#SBATCH --export=ALL

set -euo pipefail
cd /scratch/g/bgross/CIDMEG_SM/mas

decoding_python="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"
if [[ ! -x "$decoding_python" ]]; then
  echo "Python executable not found: $decoding_python" >&2
  exit 2
fi
[[ -f scripts/run_concept_decoding_cli.py && -f configs/decoding.source-volume-beta.hpc.toml ]] || {
  echo 'The HPC MAS project directory must contain scripts/ and configs/.' >&2
  exit 2
}

source_workers="${MAS_SOURCE_WORKERS:-12}"
source_worker_threads="${MAS_SOURCE_WORKER_THREADS:-4}"
subset_workers="${MAS_SUBSET_WORKERS:-48}"
subset_worker_threads="${MAS_SUBSET_WORKER_THREADS:-1}"
allocated_cpus="${SLURM_CPUS_PER_TASK:-0}"
for worker_setting in "$source_workers" "$source_worker_threads" "$subset_workers" "$subset_worker_threads"; do
  [[ "$worker_setting" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Worker counts and threads must be positive integers.' >&2
    exit 2
  }
done
[[ "$allocated_cpus" -ge "$((source_workers * source_worker_threads))" &&
   "$allocated_cpus" -ge "$((subset_workers * subset_worker_threads))" ]] || {
  echo 'Allocated CPUs must cover workers x threads for both extraction and subset search.' >&2
  exit 2
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

echo "Job ${SLURM_JOB_ID:-unknown}: ${allocated_cpus} CPUs; source ${source_workers} x ${source_worker_threads}, subsets ${subset_workers} x ${subset_worker_threads}."
echo "Python: $(command -v "$decoding_python")"
echo "Working directory: $PWD"

# Uses configs/decoding.source-volume-beta.hpc.toml (12 CIDMEG005 ICA runs with volume
# inverses *-vol-inv.fif in native space) and overrides --source-parcellation HCPMMP1
# to load CIDMEG005/Anatomy/mri/HCPMMP1+aseg.mgz.
# Uses --subset-search backward by default because 2^N exhaustive search is intractable on full HCPMMP1
# unless you pass explicit --source-parcels <ROI1> <ROI2> ... --subset-search exhaustive --subset-workers 48.
exec "$decoding_python" -I -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.source-volume-beta.hpc.toml \
  --source-parcellation HCPMMP1 \
  --subset-search backward \
  --source-workers "$source_workers" \
  --source-worker-threads "$source_worker_threads" \
  --subset-workers 1 \
  --subset-worker-threads "$subset_worker_threads" \
  --torch-num-threads 1 \
  "$@"
