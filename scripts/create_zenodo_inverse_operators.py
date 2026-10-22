#!/usr/bin/env python3
"""Automatically create FreeSurfer template anatomy, forward solutions, and inverse operators.

Designed for datasets without individual structural MRI scans (like the Zenodo MEG dataset).
1. Downloads/verifies the FreeSurfer 'fsaverage' template anatomy (surfaces, BEM, aparc atlas).
2. Computes the forward model for each MEG recording using fsaverage.
3. Computes the noise covariance matrix and MNE inverse operator (*-inv.fif).
4. Generates configs/decoding.source.zenodo.hpc.json so MAS source-space decoding can run immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mne
from mne.minimum_norm import make_inverse_operator, write_inverse_operator


def find_raw_fif_files(data_dir: Path) -> list[Path]:
    """Locate all raw FIF files in the dataset directory."""
    files = sorted(data_dir.rglob("*_raw.fif"))
    if not files:
        files = sorted(data_dir.rglob("*.fif"))
    return [f for f in files if not f.name.startswith(".") and "-inv" not in f.name and "-fwd" not in f.name]


def setup_fsaverage_template(subjects_dir: Path) -> tuple[Path, Path, Path]:
    """Ensure fsaverage is downloaded and locate its source space and BEM solution."""
    subjects_dir.mkdir(parents=True, exist_ok=True)
    print(f"Setting up FreeSurfer template (fsaverage) in: {subjects_dir}")
    fs_dir = Path(mne.datasets.fetch_fsaverage(subjects_dir=str(subjects_dir), verbose=False))

    src_path = fs_dir / "bem" / "fsaverage-oct-6-src.fif"
    bem_path = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    if not src_path.is_file() or not bem_path.is_file():
        raise FileNotFoundError(f"Missing required fsaverage BEM or source space in {fs_dir / 'bem'}")

    return fs_dir, src_path, bem_path


def compute_inverse_for_fif(
    fif_path: Path,
    out_inv_path: Path,
    subjects_dir: Path,
    src: mne.SourceSpaces,
    bem: mne.bem.ConductorModel,
) -> Path:
    """Compute and save an MNE inverse operator for one raw FIF file."""
    out_inv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Reading raw info: {fif_path.name}")
    info = mne.io.read_info(fif_path, verbose="error")

    # Filter to usable MEG channels
    meg_picks = mne.pick_types(info, meg=True, ref_meg=False, exclude="bads")
    if len(meg_picks) == 0:
        raise ValueError(f"No usable MEG channels found in {fif_path.name}")

    # Use standard fsaverage template transformation
    trans = "fsaverage"

    # Compute forward solution
    print("  Computing forward solution...")
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        verbose="error",
    )

    # Compute ad-hoc noise covariance
    print("  Estimating noise covariance...")
    noise_cov = mne.make_ad_hoc_cov(info, verbose="error")

    # Compute inverse operator
    print("  Creating inverse operator...")
    inv = make_inverse_operator(
        info,
        fwd,
        noise_cov,
        loose=0.2,
        depth=0.8,
        verbose="error",
    )

    # Save inverse operator
    write_inverse_operator(out_inv_path, inv, overwrite=True, verbose="error")
    print(f"  Saved inverse operator: {out_inv_path.name}")
    return out_inv_path


def main():
    parser = argparse.ArgumentParser(description="Create FreeSurfer template and inverse operators for Zenodo MEG dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming"),
        help="Root directory where raw FIF files are located.",
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=Path("/scratch/g/bgross/CIDMEG_SM/data/freesurfer"),
        help="FreeSurfer subjects directory to store fsaverage.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("configs/decoding.source.zenodo.hpc.json"),
        help="Path where the source manifest JSON will be written.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    subjects_dir = args.subjects_dir.resolve()
    manifest_path = args.output_manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Setup fsaverage template anatomy
    fs_dir, src_path, bem_path = setup_fsaverage_template(subjects_dir)
    src = mne.read_source_spaces(src_path, verbose="error")
    bem = mne.read_bem_solution(bem_path, verbose="error")

    # 2. Find all raw FIF files
    raw_files = find_raw_fif_files(data_dir)
    if not raw_files:
        print(f"Error: No raw FIF files found in {data_dir}. Ensure download_meg_zenodo.slurm has completed.")
        sys.exit(1)

    print(f"\nFound {len(raw_files)} recording(s). Generating inverse operators...")
    inv_dir = data_dir / "inverse"

    manifest_runs = []
    for fif_path in raw_files:
        print(f"\nProcessing: {fif_path.name}")
        inv_name = fif_path.name.replace("_raw.fif", "-inv.fif").replace(".fif", "-inv.fif")
        inv_path = inv_dir / inv_name

        try:
            if not inv_path.is_file():
                compute_inverse_for_fif(fif_path, inv_path, subjects_dir, src, bem)
            else:
                print(f"  Reusing existing inverse: {inv_path.name}")

            manifest_runs.append({
                "fif": str(fif_path),
                "inverse": str(inv_path),
                "subject": "fsaverage",
                "subjects_dir": str(subjects_dir),
            })
        except Exception as exc:
            print(f"  Error creating inverse for {fif_path.name}: {exc}")

    if not manifest_runs:
        print("Error: Could not generate any inverse operators.")
        sys.exit(1)

    # 3. Write source manifest
    manifest_data = {"runs": manifest_runs}
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"\nSuccessfully generated source manifest: {manifest_path}")
    print(f"Contains {len(manifest_runs)} run mappings ready for source-space decoding.")


if __name__ == "__main__":
    main()
