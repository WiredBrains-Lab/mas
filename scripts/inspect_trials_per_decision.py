#!/usr/bin/env python3
"""Inspect trial distribution across semantic decisions and concepts in CIDMEG datasets.

Usage:
  python scripts/inspect_trials_per_decision.py --config configs/decoding.source-volume-beta.hpc.toml
  python scripts/inspect_trials_per_decision.py --trial-table /path/to/table.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.concept_decoding_signal import (
    concept_label_from_trial_row,
    semantic_decision_label_from_trial_row,
)
from scripts.run_concept_decoding_cli import parse_args, load_and_prepare_trial_table


def inspect_dataset(df: pd.DataFrame, output_dir: Path | None = None) -> dict:
    print(f"\nLoaded dataset with {len(df):,} total raw trials.")
    print(f"Columns: {list(df.columns)}")

    # Extract concept and decision labels
    concepts = [concept_label_from_trial_row(None, row, group_mode="canonical") for _, row in df.iterrows()]
    decisions = [semantic_decision_label_from_trial_row(row) for _, row in df.iterrows()]

    df_labeled = df.copy()
    df_labeled["canonical_concept"] = concepts
    df_labeled["semantic_decision"] = decisions

    # Filter out unlabeled trials
    valid_mask = (df_labeled["canonical_concept"].str.strip() != "") & (df_labeled["semantic_decision"].str.strip() != "")
    df_valid = df_labeled[valid_mask].copy()

    unlabeled_concept_count = (df_labeled["canonical_concept"].str.strip() == "").sum()
    unlabeled_decision_count = (df_labeled["semantic_decision"].str.strip() == "").sum()

    print(f"Valid labeled trials: {len(df_valid):,} / {len(df):,}")
    if unlabeled_concept_count > 0:
        print(f"  - Missing canonical concept: {unlabeled_concept_count:,} trials")
    if unlabeled_decision_count > 0:
        print(f"  - Missing semantic decision: {unlabeled_decision_count:,} trials")

    unique_concepts = sorted(df_valid["canonical_concept"].unique())
    unique_decisions = sorted(df_valid["semantic_decision"].unique())

    print(f"\nIdentified {len(unique_concepts)} distinct concepts and {len(unique_decisions)} semantic decision tasks.")
    print("Semantic decision tasks found:")
    decision_totals = df_valid["semantic_decision"].value_counts()
    for idx, (dec, count) in enumerate(decision_totals.items(), 1):
        print(f"  {idx}. {dec!r}: {count:,} trials ({count / len(df_valid) * 100:.1f}%)")

    # Contingency matrix: Concepts x Decisions
    ct = pd.crosstab(df_valid["canonical_concept"], df_valid["semantic_decision"])

    # 5-tuple balance metrics per concept
    min_per_concept = ct.min(axis=1)
    max_per_concept = ct.max(axis=1)
    mean_per_concept = ct.mean(axis=1)
    total_per_concept = ct.sum(axis=1)

    ct["Balanced_5tuples"] = min_per_concept
    ct["Total_Trials"] = total_per_concept
    ct["Surplus_Trials"] = total_per_concept - (min_per_concept * len(unique_decisions))

    total_possible_5tuples = int(min_per_concept.sum())
    total_retained_in_5tuples = total_possible_5tuples * len(unique_decisions)
    total_surplus = int(ct["Surplus_Trials"].sum())

    print("\n" + "=" * 70)
    print("SUMMARY: 5-WAY TASK-BALANCED PSEUDO-EPOCH FEASIBILITY")
    print("=" * 70)
    print(f"Total concepts analyzed:                 {len(unique_concepts)}")
    print(f"Total decisions per concept:             {len(unique_decisions)}")
    print(f"Minimum trials per decision (per class): min={min_per_concept.min()}, median={min_per_concept.median():.1f}, max={min_per_concept.max()}")
    print(f"Maximum trials per decision (per class): min={max_per_concept.min()}, median={max_per_concept.median():.1f}, max={max_per_concept.max()}")
    print("-" * 70)
    print(f"Strict Balanced 5-Tuples achievable:    {total_possible_5tuples:,} pseudo-epochs across all concepts")
    print(f"  -> Trials utilized in pure 5-tuples:   {total_retained_in_5tuples:,} / {len(df_valid):,} ({total_retained_in_5tuples / len(df_valid) * 100:.1f}%)")
    print(f"  -> Surplus / unused trials:            {total_surplus:,} / {len(df_valid):,} ({total_surplus / len(df_valid) * 100:.1f}%)")
    print(f"Average balanced 5-tuples per concept:   {min_per_concept.mean():.2f} (approx. {min_per_concept.min()} to {min_per_concept.max()} per class)")
    print("=" * 70)

    # Check for concepts missing any decision
    missing_any = ct[ct["Balanced_5tuples"] == 0]
    if not missing_any.empty:
        print(f"\nWARNING: {len(missing_any)} concept(s) have 0 trials for at least one decision task:")
        for concept in missing_any.index:
            zero_cols = [col for col in unique_decisions if ct.loc[concept, col] == 0]
            print(f"  - {concept}: 0 trials for {zero_cols}")
    else:
        print("\nAll concepts have at least ONE trial in EVERY semantic decision task! (No missing cells)")

    # Print first 15 concepts sample table
    print("\nSample concept breakdown (first 15 concepts):")
    cols_to_show = list(unique_decisions) + ["Balanced_5tuples", "Total_Trials"]
    print(ct[cols_to_show].head(15).to_string())

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "concept_by_decision_trial_counts.csv"
        ct.to_csv(csv_path)
        print(f"\nSaved full concept-by-decision crosstab to: {csv_path}")

        summary = {
            "n_raw_trials": int(len(df)),
            "n_valid_trials": int(len(df_valid)),
            "n_concepts": int(len(unique_concepts)),
            "n_decisions": int(len(unique_decisions)),
            "decisions": unique_decisions,
            "decision_trial_totals": decision_totals.to_dict(),
            "total_balanced_5tuples": total_possible_5tuples,
            "trials_in_balanced_5tuples": total_retained_in_5tuples,
            "surplus_trials": total_surplus,
            "min_5tuples_per_concept": int(min_per_concept.min()),
            "median_5tuples_per_concept": float(min_per_concept.median()),
            "max_5tuples_per_concept": int(min_per_concept.max()),
            "concepts_missing_any_decision": list(missing_any.index),
        }
        json_path = output_dir / "decision_distribution_summary.json"
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary JSON to: {json_path}")

    return {
        "ct": ct,
        "valid_df": df_valid,
        "decisions": unique_decisions,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect distribution of trials per semantic decision and concept.")
    parser.add_argument("--config", help="TOML config path (e.g. configs/decoding.source-volume-beta.hpc.toml)")
    parser.add_argument("--trial-table", help="Direct path to trial table CSV")
    parser.add_argument("--output-dir", default=".", help="Directory to save summary CSV and JSON")
    cli_args, remaining = parser.parse_known_args()

    if cli_args.trial_table:
        print(f"Loading trial table from: {cli_args.trial_table}")
        df = pd.read_csv(cli_args.trial_table)
    elif cli_args.config or remaining:
        args_to_parse = []
        if cli_args.config:
            args_to_parse.extend(["--config", cli_args.config])
        args_to_parse.extend(remaining)
        args = parse_args(args_to_parse)
        df = load_and_prepare_trial_table(args)
    else:
        # Default fallback to source-volume-beta config
        default_config = ROOT / "configs" / "decoding.source-volume-beta.hpc.toml"
        if default_config.exists():
            print(f"No arguments provided; using default config: {default_config}")
            args = parse_args(["--config", str(default_config)])
            df = load_and_prepare_trial_table(args)
        else:
            parser.error("Must supply --config or --trial-table")

    inspect_dataset(df, Path(cli_args.output_dir))


if __name__ == "__main__":
    main()
