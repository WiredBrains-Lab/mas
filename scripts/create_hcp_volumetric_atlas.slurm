#!/bin/bash -l
# Submit from the HPC MAS directory:
#   sbatch /scratch/g/bgross/CIDMEG_SM/mas/scripts/create_hcp_volumetric_atlas.slurm
#
# Or specify a different SUBJECTS_DIR and SUBJECT:
#   sbatch scripts/create_hcp_volumetric_atlas.slurm /path/to/subjects_dir SubjectName
#
# Generates volumetric HCP-MMP segmentations (HCPMMP1_combined+aseg.mgz and HCPMMP1+aseg.mgz)
# for an individual FreeSurfer subject.
#
# Steps:
#   1. Fetches HCP-MMP annotations on fsaverage via MNE-Python (if missing).
#   2. Projects surface annotations from fsaverage to subject via mri_surf2surf.
#   3. Converts subject surface annotations to 3D volume segmentations via mri_aparc2aseg.
#   4. Verifies the volumetric atlas labels using MNE.
#
#SBATCH --job-name=create-hcp-vol
#SBATCH --account=bgross
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/g/bgross/CIDMEG_SM/mas/create-hcp-vol-%j.log
#SBATCH --error=/scratch/g/bgross/CIDMEG_SM/mas/create-hcp-vol-%j.log
#SBATCH --export=ALL

set -eo pipefail

# Project root
cd /scratch/g/bgross/CIDMEG_SM/mas

# 1. Subject and directory paths
TARGET_SUBJECTS_DIR="${1:-${SUBJECTS_DIR:-/scratch/g/bgross/CIDMEG_SM/data/CIDMEG005}}"
SUBJECT="${2:-${MAS_SUBJECT:-Anatomy}}"
decoding_python="${MAS_PYTHON:-$HOME/.conda/envs/mas-decoding-rcc/bin/python}"

# 2. Safely load FreeSurfer environment
# Sourcing SetUpFreeSurfer.sh or running module load in a non-interactive SLURM shell
# can return non-zero from internal pipeline checks; disable errexit, nounset, AND pipefail.
set +e
set +u
set +o pipefail

# Ensure the `module` command is initialized in non-interactive batch shells
if ! command -v module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /etc/profile; do
    if [[ -f "$init_script" ]]; then
      # shellcheck disable=SC1090
      source "$init_script" >/dev/null 2>&1 || true
      command -v module >/dev/null 2>&1 && break
    fi
  done
fi

export FS_FREESURFERENV_NO_OUTPUT=0
if command -v module >/dev/null 2>&1; then
  module load freesurfer || true
fi

# Fallback: if FREESURFER_HOME is still unset, search standard cluster locations
if [[ -z "${FREESURFER_HOME:-}" ]]; then
  for candidate in /apps/software/freesurfer/* /usr/local/freesurfer/* /opt/freesurfer/*; do
    if [[ -f "${candidate}/SetUpFreeSurfer.sh" ]]; then
      export FREESURFER_HOME="$candidate"
      break
    fi
  done
fi

if [[ -n "${FREESURFER_HOME:-}" && -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" || true
fi

set -eo pipefail

# Re-establish user subjects directory (SetUpFreeSurfer.sh defaults SUBJECTS_DIR to its own tree)
export SUBJECTS_DIR="$TARGET_SUBJECTS_DIR"

echo "============================================================"
echo "HCP Volumetric Atlas Generator"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "SUBJECTS_DIR:     ${SUBJECTS_DIR}"
echo "Subject:          ${SUBJECT}"
echo "FreeSurfer Home:  ${FREESURFER_HOME:-unknown}"
echo "Python:           ${decoding_python}"
echo "Working dir:      $PWD"
echo "============================================================"

# Verify FreeSurfer binaries are accessible
if ! command -v mri_surf2surf >/dev/null 2>&1 || ! command -v mri_aparc2aseg >/dev/null 2>&1; then
  echo "Error: FreeSurfer (mri_surf2surf and mri_aparc2aseg) could not be loaded." >&2
  exit 2
fi

# Verify subject directory exists
if [[ ! -d "${SUBJECTS_DIR}/${SUBJECT}/surf" || ! -d "${SUBJECTS_DIR}/${SUBJECT}/mri" ]]; then
  echo "Error: Subject '${SUBJECT}' not found in '${SUBJECTS_DIR}' or missing surf/mri directories." >&2
  exit 2
fi

# Ensure label directory exists
mkdir -p "${SUBJECTS_DIR}/${SUBJECT}/label"

# If fsaverage is not in the user's subjects directory, link or copy it from FreeSurfer
if [[ ! -d "${SUBJECTS_DIR}/fsaverage" && -d "${FREESURFER_HOME}/subjects/fsaverage" ]]; then
  echo "Linking fsaverage template from FreeSurfer installation..."
  mkdir -p "${SUBJECTS_DIR}"
  cp -r "${FREESURFER_HOME}/subjects/fsaverage" "${SUBJECTS_DIR}/fsaverage" 2>/dev/null || \
  ln -s "${FREESURFER_HOME}/subjects/fsaverage" "${SUBJECTS_DIR}/fsaverage" || true
fi

# 3. Ensure fsaverage has HCP-MMP annotations
echo ""
echo "--> Step 1: Ensuring fsaverage HCP-MMP annotations exist..."
if [[ -x "$decoding_python" ]]; then
  "$decoding_python" -u - <<EOF
import mne
from pathlib import Path

subjects_dir = Path("${SUBJECTS_DIR}").expanduser().resolve()
fsaverage_dir = subjects_dir / "fsaverage" / "label"

lh_comb = fsaverage_dir / "lh.HCPMMP1_combined.annot"
rh_comb = fsaverage_dir / "rh.HCPMMP1_combined.annot"
lh_full = fsaverage_dir / "lh.HCPMMP1.annot"
rh_full = fsaverage_dir / "rh.HCPMMP1.annot"

if not (lh_comb.exists() and rh_comb.exists()):
    print("Fetching HCPMMP1_combined annotations for fsaverage...")
    try:
        mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=str(subjects_dir), combine=True, accept=True, verbose=False)
    except Exception as exc:
        print(f"Notice: fetch_hcp_mmp_parcellation (combined) note: {exc}")

if not (lh_full.exists() and rh_full.exists()):
    print("Fetching full HCPMMP1 annotations for fsaverage...")
    try:
        mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=str(subjects_dir), combine=False, accept=True, verbose=False)
    except Exception as exc:
        print(f"Notice: fetch_hcp_mmp_parcellation (full) note: {exc}")

if (lh_comb.exists() and rh_comb.exists()) or (lh_full.exists() and rh_full.exists()):
    print("fsaverage HCP-MMP surface annotations verified.")
else:
    print("Warning: fsaverage annotations not yet populated in subjects_dir.")
EOF
else
  echo "Warning: Python executable not found at ${decoding_python}."
fi

# 4. Map surface annotations from fsaverage to subject
echo ""
echo "--> Step 2: Projecting HCP-MMP annotations from fsaverage to ${SUBJECT}..."

for annot in "HCPMMP1_combined" "HCPMMP1"; do
  for hemi in lh rh; do
    src_annot="${SUBJECTS_DIR}/fsaverage/label/${hemi}.${annot}.annot"
    trg_annot="${SUBJECTS_DIR}/${SUBJECT}/label/${hemi}.${annot}.annot"

    if [[ -f "$trg_annot" ]]; then
      echo "  [skip] Already exists: ${hemi}.${annot}.annot"
    else
      if [[ ! -f "$src_annot" ]]; then
        echo "  [warning] Source annotation missing: ${src_annot}; skipping ${hemi}.${annot}."
        continue
      fi
      echo "  [surf2surf] Mapping ${hemi}.${annot} -> ${SUBJECT}..."
      mri_surf2surf \
        --srcsubject fsaverage \
        --trgsubject "${SUBJECT}" \
        --hemi "${hemi}" \
        --sval-annot "${src_annot}" \
        --tval "${trg_annot}"
    fi
  done
done

# 5. Convert subject surface annotations into 3D volume segmentations
echo ""
echo "--> Step 3: Converting surface annotations to 3D volume segmentations via mri_aparc2aseg..."

for annot in "HCPMMP1_combined" "HCPMMP1"; do
  lh_annot="${SUBJECTS_DIR}/${SUBJECT}/label/lh.${annot}.annot"
  rh_annot="${SUBJECTS_DIR}/${SUBJECT}/label/rh.${annot}.annot"
  out_volume="${SUBJECTS_DIR}/${SUBJECT}/mri/${annot}+aseg.mgz"

  if [[ ! -f "$lh_annot" || ! -f "$rh_annot" ]]; then
    echo "  [skip] Missing subject annotations for ${annot}; skipping volume generation."
    continue
  fi

  echo "  [aparc2aseg] Generating ${out_volume}..."
  mri_aparc2aseg \
    --s "${SUBJECT}" \
    --annot "${annot}" \
    --o "${out_volume}" \
    --rip-unknown
  echo "  --> Generated: ${out_volume}"
done

# 6. Verify volumetric atlas labels using MNE
echo ""
echo "--> Step 4: Verifying volumetric atlas labels with Python / MNE..."
if [[ -x "$decoding_python" ]]; then
  "$decoding_python" -u - <<EOF
import mne
from pathlib import Path

subjects_dir = Path("${SUBJECTS_DIR}").expanduser().resolve()
subject_dir = subjects_dir / "${SUBJECT}" / "mri"

for annot in ["HCPMMP1_combined", "HCPMMP1"]:
    vol_path = subject_dir / f"{annot}+aseg.mgz"
    if not vol_path.exists():
        continue
    try:
        import nibabel.freesurfer as fs
        lut, _ = mne.read_freesurfer_lut()
        custom_lut = {str(k): int(v) for k, v in lut.items() if int(v) < 1000 or int(v) >= 3000}
        for hemi, offset in (("lh", 1000), ("rh", 2000)):
            annot_file = subjects_dir / "${SUBJECT}" / "label" / f"{hemi}.{annot}.annot"
            if annot_file.exists():
                _, _, names = fs.read_annot(str(annot_file))
                for idx, raw_name in enumerate(names):
                    name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                    key = f"ctx-{hemi}-unknown" if idx == 0 or name in ("???", "unknown") else f"ctx-{hemi}-{name}"
                    custom_lut[key] = offset + idx
        labels = mne.get_volume_labels_from_aseg(str(vol_path), atlas_ids=custom_lut)
        named = [name for name in labels if "unknown" not in str(name).lower()]
        print(f"Successfully loaded {vol_path.name}: {len(named)} named volumetric parcels found.")
        print(f"  Sample cortical parcels: {[n for n in named if n.startswith('ctx-lh-')][:8]}")
    except Exception as exc:
        print(f"Notice: mne.get_volume_labels_from_aseg note on {vol_path.name}: {exc}")

EOF
fi

echo ""
echo "============================================================"
echo "Volumetric HCP atlas generation complete: $(date)"
echo "Output files in ${SUBJECTS_DIR}/${SUBJECT}/mri/:"
ls -lh "${SUBJECTS_DIR}/${SUBJECT}/mri/"*HCPMMP* 2>/dev/null || echo "No HCPMMP volumes found."
echo "============================================================"
