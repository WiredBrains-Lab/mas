#!/usr/bin/env bash
# Run from conda base on RCC. Installs into a NEW named environment only.
# Usage: bash scripts/setup_decoding_env.sh [--name mas-decoding]
#        [--torch-build cu128|cu126|cu129|cpu] [--require-gpu]
# PyTorch wheel choices: https://pytorch.org/get-started/previous-versions/#v280
set -euo pipefail

usage() {
  cat <<'HELP'
Create a new Linux Conda environment for MAS decoding, screening resume,
and parallel CPU workers. Does not install packages into base.

  bash scripts/setup_decoding_env.sh
  bash scripts/setup_decoding_env.sh --name mas-decoding --require-gpu
  bash scripts/setup_decoding_env.sh --name mas-decoding-cpu --torch-build cpu

Options:
  --name NAME           Environment name (default: mas-decoding).
  --torch-build BUILD   PyTorch 2.8.0 wheel: cu128 (default), cu126, cu129, cpu.
  --require-gpu         Fail verification if CUDA is unavailable; use inside
                        an allocated GPU session, not a login/CPU session.
  --help                Show this help without installing anything.

Requires internet access, Conda and the MAS core/ and scripts/ directories.
Creates ~/.conda/envs/NAME. An existing path is never overwritten.
HELP
}

env_name=mas-decoding
torch_build=cu128
require_gpu=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name|--torch-build)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      if [[ "$1" == --name ]]; then env_name=$2; else torch_build=$2; fi
      shift 2 ;;
    --require-gpu) require_gpu=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$env_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$env_name" != base && "$env_name" != root ]] || {
  echo 'Choose a non-base environment name containing letters, numbers, dots, underscores or hyphens.' >&2; exit 2;
}
case "$torch_build" in cu128|cu126|cu129|cpu) ;; *) echo 'Invalid --torch-build.' >&2; exit 2 ;; esac
if [[ "$torch_build" == cpu && "$require_gpu" == 1 ]]; then
  echo '--require-gpu cannot be used with --torch-build cpu.' >&2; exit 2
fi
[[ "$(uname -s)" == Linux ]] || { echo 'This installer targets Linux/RCC.' >&2; exit 2; }
command -v conda >/dev/null || { echo 'Conda not found. Initialize/load Conda first.' >&2; exit 1; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
requirements="$script_dir/requirements-decoding-rcc.txt"
[[ -f "$requirements" && -f "$repo_dir/core/concept_decoding_subsets.py" ]] || {
  echo 'Copy scripts/requirements-decoding-rcc.txt and the MAS core/ directory alongside this script.' >&2; exit 1;
}
# Use a user-owned prefix rather than the shared Conda installation's envs.
env_prefix="$HOME/.conda/envs/$env_name"
if [[ -e "$env_prefix" ]]; then
  echo "Environment path '$env_prefix' already exists. Choose another --name; it was not modified." >&2
  exit 1
fi
mkdir -p "$HOME/.conda/envs" "$HOME/.conda/pkgs"
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
export MPLBACKEND=Agg

echo "Creating $env_prefix (Python 3.11; PyTorch 2.8.0/$torch_build)..."
conda create --yes --prefix "$env_prefix" --override-channels -c conda-forge python=3.11 pip
# Never depend on conda run/PATH to select the intended Python on shared HPC.
run_python() { "$env_prefix/bin/python" -I "$@"; }
run_python -c '
import pathlib, sys, sysconfig, pip
expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(sys.prefix).resolve()
site = pathlib.Path(sysconfig.get_path("purelib")).resolve()
if actual != expected or expected not in site.parents:
    raise SystemExit(f"Refusing install: Python prefix={actual}, site-packages={site}, expected={expected}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Refusing install: expected Python 3.11, got {sys.version}")
if expected not in pathlib.Path(pip.__file__).resolve().parents:
    raise SystemExit(f"Refusing install: pip was loaded outside the new environment: {pip.__file__}")
print("Verified environment Python:", sys.executable)
print("Verified package destination:", site)
' "$env_prefix"
run_python -m pip install --no-user -r "$requirements"
run_python -m pip install --no-user 'torch==2.8.0' --index-url "https://download.pytorch.org/whl/$torch_build"
run_python -m pip check

# This checks real imports and CPU/GPU operations, without running an analysis.
run_python -c '
import importlib, os, pathlib, subprocess, sys
root, require_gpu = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
sys.path.insert(0, str(root))
for name in ("numpy", "pandas", "scipy", "sklearn", "mne", "matplotlib", "joblib",
             "h5py", "threadpoolctl", "nibabel", "openpyxl", "xlrd", "pyarrow",
             "core.fif_processing", "core.concept_decoding_source",
             "core.concept_decoding_transformer",
             "core.concept_decoding_subsets", "core.concept_subset_parallel",
             "core.decoding_resume_paths", "scripts.run_concept_decoding_cli"):
    importlib.import_module(name)
import torch
print("Python:", sys.version.split()[0])
print("Environment:", sys.prefix)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
x = torch.ones((16, 16))
assert (x @ x).sum().item() == 4096
if torch.cuda.is_available():
    x = x.cuda()
    assert (x @ x).sum().item() == 4096
    torch.cuda.synchronize()
    print("GPU verified:", torch.cuda.get_device_name(0))
elif require_gpu:
    raise SystemExit("CUDA unavailable. Verify the GPU allocation and driver compatibility with the selected PyTorch wheel.")
else:
    print("CUDA not currently available; CPU verification passed. Recheck inside a GPU allocation before GPU training.")
# Save the actual resolved package versions inside the new environment.
record = pathlib.Path(sys.prefix) / "mas-decoding-pip-freeze.txt"
record.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
print("Installed package record:", record)
print("CLI and screening imports passed.")
' "$repo_dir" "$require_gpu"

printf '\nReady. Activate in your Code Server terminal:\n  conda activate %s\n' "$env_prefix"
printf 'Then validate your batch:\n  python scripts/run_concept_decoding_cli.py --resume /path/to/batch --dry-run\n'
echo 'Transformer models additionally need your CIDECoG_enc-main/src code and --seeg-decoder-src.'
