#!/usr/bin/env bash
""":"
# Self-executing bash wrapper to run with python3 or MAS_PYTHON:
python_bin="${MAS_PYTHON:-$(command -v python3 || command -v python)}"
exec "$python_bin" "$0" "$@"
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

DEFAULT_SEARCH_ROOTS = [
    "/scratch/g/bgross/CIDMEG_SM/data/CIDMEG005/ICA/decoding_results",
    "/scratch/g/bgross/CIDMEG_SM/data/CIDMEG005/decoding_results",
    "/scratch/g/bgross/CIDMEG_SM/mas/decoding_results",
    "/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming/decoding_results_source",
    "/scratch/g/bgross/CIDMEG_SM/data/meg_picture_naming/decoding_results",
]


def is_subdir(child: Path, parent: Path) -> bool:
    """Safely check that child is strictly inside parent (not parent itself)."""
    try:
        rel = child.resolve().relative_to(parent.resolve())
        return len(rel.parts) > 0
    except ValueError:
        return False


def get_dir_size(path: Path) -> int:
    """Calculate directory size in bytes without failing on missing files."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(Path(entry.path))
    except Exception:
        pass
    return total


def format_bytes(size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def find_screening_batches(search_roots: list[str]) -> list[Path]:
    """Find all screening folders containing subset_comparison.csv."""
    batches = []
    for root_str in search_roots:
        root = Path(root_str)
        if not root.is_dir():
            continue
        try:
            # Depth limit to avoid descending into worker folders
            for p in root.glob("*/*_selected_parcel_subsets"):
                if (p / "subset_comparison.csv").is_file():
                    batches.append(p)
            for p in root.glob("*_selected_parcel_subsets"):
                if (p / "subset_comparison.csv").is_file():
                    batches.append(p)
            for p in root.glob("*/*_backward_elimination"):
                if (p / "subset_comparison.csv").is_file():
                    batches.append(p)
        except Exception:
            pass
    # Deduplicate while preserving order
    unique = []
    seen = set()
    for b in batches:
        resolved = str(b.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(b)
    return unique


def prune_batch(batch_dir: Path, threshold: float, apply: bool) -> tuple[int, int, int]:
    """
    Prune subset subfolders whose MPR < threshold.
    Crucially: subset_comparison.csv, manifest.json, settings_used.json,
    and source_features.joblib are NEVER deleted, preserving 100% resume capability.
    """
    csv_path = batch_dir / "subset_comparison.csv"
    if not csv_path.is_file():
        return 0, 0, 0

    deleted_count = 0
    preserved_count = 0
    bytes_freed = 0

    try:
        content = csv_path.read_text(encoding="utf-8", errors="replace")
        rows = list(csv.DictReader(content.splitlines()))
    except Exception as exc:
        print(f"  [WARNING] Could not read {csv_path}: {exc}", file=sys.stderr)
        return 0, 0, 0

    for row in rows:
            status = row.get("status", "")
            raw_mpr = row.get("mpr")

            try:
                mpr = float(raw_mpr) if raw_mpr not in (None, "") else None
            except (ValueError, TypeError):
                mpr = None

            out_dir_str = row.get("output_dir", "")
            if not out_dir_str:
                continue

            target_path = Path(out_dir_str)
            if not target_path.exists():
                # Check if relative to batch directory
                fallback = batch_dir / target_path.name
                if fallback.exists():
                    target_path = fallback

            # Only consider existing directories that are strictly inside this batch
            if not target_path.is_dir() or not is_subdir(target_path, batch_dir):
                continue

            # Only prune completed models whose MPR is strictly less than threshold
            if status == "complete" and mpr is not None and mpr < threshold:
                # Sample size for the first few directories to avoid slow metadata crawls
                if deleted_count < 5:
                    sample_size = get_dir_size(target_path)
                    if sample_size > 0:
                        bytes_freed += sample_size
                elif deleted_count == 5 and bytes_freed > 0:
                    avg_size = bytes_freed // 5
                    bytes_freed += avg_size
                elif deleted_count > 5 and bytes_freed > 0:
                    bytes_freed += avg_size

                deleted_count += 1
                if apply:
                    try:
                        shutil.rmtree(target_path)
                    except Exception as exc:
                        print(f"  [ERROR] Failed to delete {target_path.name}: {exc}", file=sys.stderr)

                if deleted_count % 1000 == 0:
                    action_word = "Deleted" if apply else "Found"
                    print(f"    ... {action_word} {deleted_count} low-MPR model folders so far")
            else:
                preserved_count += 1

    return deleted_count, preserved_count, bytes_freed


def main():
    parser = argparse.ArgumentParser(
        description="Safely delete low-performing subset model folders (MPR < threshold) to free scratch space while preserving full resume capability."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.53,
        help="MPR cutoff threshold (default: 0.53). Models with MPR < threshold are pruned.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete folders. If not specified, runs in dry-run mode.",
    )
    parser.add_argument(
        "--path",
        nargs="*",
        default=[],
        help="Specific screening batch directory or search root to scan.",
    )

    args = parser.parse_args()

    search_roots = args.path if args.path else DEFAULT_SEARCH_ROOTS

    print("============================================================")
    print("Screening Subset Pruner (Scratch Space Saver)")
    print(f"Mode:             {'*** LIVE DELETION (--apply) ***' if args.apply else 'DRY RUN (preview only; use --apply to execute)'}")
    print(f"MPR Threshold:    < {args.threshold:.4f}")
    print("Resume Safety:    subset_comparison.csv, manifest.json,")
    print("                  source_features.joblib, & settings are PRESERVED.")
    print("============================================================")

    # If the user passed a specific batch directory directly
    batches = []
    for p_str in search_roots:
        p = Path(p_str)
        if (p / "subset_comparison.csv").is_file():
            batches.append(p)

    if not batches:
        batches = find_screening_batches(search_roots)

    if not batches:
        print("No screening batch directories found in search roots.")
        return 0

    print(f"Found {len(batches)} screening batch directory/directories:\n")

    grand_deleted = 0
    grand_preserved = 0
    grand_bytes = 0

    for batch in batches:
        print(f"Checking: {batch}")
        deleted, preserved, bytes_freed = prune_batch(batch, args.threshold, args.apply)
        grand_deleted += deleted
        grand_preserved += preserved
        grand_bytes += bytes_freed

        action_label = "Deleted" if args.apply else "Would delete"
        print(f"  -> {action_label}: {deleted} models (MPR < {args.threshold:.2f})")
        print(f"  -> Preserved:    {preserved} models (MPR >= {args.threshold:.2f} or active)")
        print(f"  -> Space:        {format_bytes(bytes_freed)} {'reclaimed' if args.apply else 'reclaimable'}\n")

    print("============================================================")
    action_label = "Total Deleted" if args.apply else "Total Reclaimable"
    print(f"{action_label}:  {grand_deleted} subset model folders ({format_bytes(grand_bytes)})")
    print(f"Total Preserved: {grand_preserved} high-performing subset models")
    if not args.apply and grand_deleted > 0:
        print("\nTo execute actual deletion and reclaim space, re-run with: --apply")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
