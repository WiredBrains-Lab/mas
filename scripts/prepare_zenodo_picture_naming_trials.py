#!/usr/bin/env python3
"""Prepare MAS-compatible trial tables for the Zenodo MEG Picture Naming Dataset (Record 15789569).

Extracts thresholded multi-bit digital trigger events across analog channels
STIM1 (bit 0), STIM2 (bit 1), STIM3 (bit 2), and STIM4 (bit 3).

Maps the 6 picture concepts (Codes 8, 11, 12, 13, 14, 15) to:
  Food (3 exemplars): sushi (8), dumplings (11), cookies (12)
  Tool (3 exemplars): comb (13), knife (14), pencil (15)

Generates MAS-compatible trial tables for:
  - trials_all_steps.csv: All trial events across all steps.
  - trials_step1_visual.csv: Step 1 (Visual image onset, 0 ms).
  - trials_step2_action.csv: Step 2 (Action imagination '*' cue, +2000 ms).
  - trials_step3_covert.csv: Step 3 (Covert/silent naming '**' cue, +4000 ms).
  - trials_step4_overt.csv: Step 4 (Overt spoken pronunciation '***' cue, +6000 ms).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Stimulus mapping for the 6 picture items in Patashov et al.
# Codes 8, 11, 12, 13, 14, 15
STIM_CODE_MAP = {
    8:  ("sushi", "food"),
    11: ("dumplings", "food"),
    12: ("cookies", "food"),
    13: ("comb", "tool"),
    14: ("knife", "tool"),
    15: ("pencil", "tool"),
}

VALID_STIM_CODES = set(STIM_CODE_MAP.keys())


def find_dataset_fif_files(data_dir: Path) -> list[Path]:
    """Find all raw FIF files in the dataset folder, sorted by subject and run."""
    fif_files = sorted(data_dir.rglob("*_raw.fif"))
    if not fif_files:
        fif_files = sorted(data_dir.rglob("*.fif"))
    return [
        f for f in fif_files
        if not f.name.startswith(".") and "-inv" not in f.name and "-fwd" not in f.name
    ]


def extract_events_from_raw(raw: mne.io.Raw) -> np.ndarray:
    """Extract thresholded multi-bit digital trigger events across STIM1-STIM4."""
    stim_channels = [ch for ch in ["STIM1", "STIM2", "STIM3", "STIM4"] if ch in raw.ch_names]

    if len(stim_channels) == 4:
        # Load the 4 stim channels
        stim_data = raw.get_data(picks=stim_channels)
        # Threshold analog voltages (> 1.5 V = Digital HIGH)
        b0 = (stim_data[0] > 1.5).astype(int)
        b1 = (stim_data[1] > 1.5).astype(int)
        b2 = (stim_data[2] > 1.5).astype(int)
        b3 = (stim_data[3] > 1.5).astype(int)

        trigger_word = b0 + 2 * b1 + 4 * b2 + 8 * b3
        diff = np.diff(trigger_word, prepend=0)
        rising_onsets = np.where(diff > 0)[0]
        codes = trigger_word[rising_onsets]

        events = np.column_stack([rising_onsets, np.zeros_like(rising_onsets), codes])
        return events

    # Fallback to standard MNE find_events if 4 stim channels are not present
    return mne.find_events(raw, min_duration=0.002, verbose="error")


def build_trial_table_for_subject(fif_path: Path, raw: mne.io.Raw, events: np.ndarray) -> pd.DataFrame:
    """Map raw events into structured MAS trial rows for all 4 experimental steps."""
    if len(events) == 0:
        return pd.DataFrame()

    sfreq = float(raw.info["sfreq"])
    step2_offset_samples = int(round(2.0 * sfreq))  # +2000 ms
    step3_offset_samples = int(round(4.0 * sfreq))  # +4000 ms
    step4_offset_samples = int(round(6.0 * sfreq))  # +6000 ms

    # Extract subject identifier
    fif_stem = fif_path.name
    subject_id = fif_path.parent.name
    if not subject_id.isdigit():
        for part in fif_stem.split("_"):
            if "sub" in part.lower():
                subject_id = part.lower().replace("sub", "")
                break

    rows = []
    trial_counter = 0

    for sample, _, event_code in events:
        code = int(event_code)
        if code not in VALID_STIM_CODES:
            continue

        trial_counter += 1
        stimulus, category = STIM_CODE_MAP[code]

        # Step 1: Visual stimulus presentation (0 ms)
        rows.append({
            "source_fif_path": str(fif_path.resolve()),
            "source_fif": fif_path.name,
            "subject": str(subject_id),
            "onset_sample": int(sample),
            "event_code": code,
            "stimulus": stimulus,
            "concept_label": stimulus,
            "stimulus_group": category,
            "step": "step1_visual",
            "trial_index": trial_counter,
        })

        # Step 2: Action imagination cue (+2000 ms)
        rows.append({
            "source_fif_path": str(fif_path.resolve()),
            "source_fif": fif_path.name,
            "subject": str(subject_id),
            "onset_sample": int(sample + step2_offset_samples),
            "event_code": code,
            "stimulus": stimulus,
            "concept_label": stimulus,
            "stimulus_group": category,
            "step": "step2_action",
            "trial_index": trial_counter,
        })

        # Step 3: Covert/silent naming cue (+4000 ms)
        rows.append({
            "source_fif_path": str(fif_path.resolve()),
            "source_fif": fif_path.name,
            "subject": str(subject_id),
            "onset_sample": int(sample + step3_offset_samples),
            "event_code": code,
            "stimulus": stimulus,
            "concept_label": stimulus,
            "stimulus_group": category,
            "step": "step3_covert",
            "trial_index": trial_counter,
        })

        # Step 4: Overt pronunciation cue (+6000 ms)
        rows.append({
            "source_fif_path": str(fif_path.resolve()),
            "source_fif": fif_path.name,
            "subject": str(subject_id),
            "onset_sample": int(sample + step4_offset_samples),
            "event_code": code,
            "stimulus": stimulus,
            "concept_label": stimulus,
            "stimulus_group": category,
            "step": "step4_overt",
            "trial_index": trial_counter,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare MAS trial tables for Zenodo picture naming dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming"),
        help="Root directory where raw FIF files are located.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output trial CSV tables (defaults to --data-dir).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for FIF files in: {data_dir}")
    fif_files = find_dataset_fif_files(data_dir)
    if not fif_files:
        print(f"No .fif files found in {data_dir}. Ensure download_meg_zenodo.slurm has completed.")
        sys.exit(1)

    print(f"Found {len(fif_files)} recording(s):")
    for f in fif_files:
        print(f"  - {f.relative_to(data_dir)}")

    all_tables = []
    for fif_path in fif_files:
        print(f"\nProcessing {fif_path.name}...")
        try:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="error")
            events = extract_events_from_raw(raw)
            print(f"  Detected {len(events)} total trigger rising edges.")

            table = build_trial_table_for_subject(fif_path, raw, events)
            if not table.empty:
                all_tables.append(table)
                n_trials = len(table[table["step"] == "step1_visual"])
                print(f"  Mapped {n_trials} picture trials ({len(table)} step events across 4 steps).")
            else:
                print(f"  Warning: No valid stimulus events found in {fif_path.name}")
        except Exception as exc:
            print(f"  Error reading {fif_path.name}: {exc}")

    if not all_tables:
        print("Error: Could not extract trial events from any FIF files.")
        sys.exit(1)

    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_df["combined_trial_index"] = np.arange(1, len(combined_df) + 1)

    # Save comprehensive table
    all_path = output_dir / "trials_all_steps.csv"
    combined_df.to_csv(all_path, index=False)
    print(f"\nSaved combined trial table: {all_path} ({len(combined_df)} total events)")

    # Save step-specific tables for targeted decoding
    for step_name in ["step1_visual", "step2_action", "step3_covert", "step4_overt"]:
        sub_df = combined_df[combined_df["step"] == step_name].copy()
        if not sub_df.empty:
            out_file = output_dir / f"trials_{step_name}.csv"
            sub_df.to_csv(out_file, index=False)
            # Print class distribution for verification
            counts = sub_df["stimulus"].value_counts().to_dict()
            cat_counts = sub_df["stimulus_group"].value_counts().to_dict()
            print(f"Saved {step_name}: {out_file.name} ({len(sub_df)} trials) | Categories: {cat_counts}")

    print("\nTrial table preparation complete.")


if __name__ == "__main__":
    main()
