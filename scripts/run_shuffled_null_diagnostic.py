#!/usr/bin/env python3
"""Standalone Shuffled-Null Diagnostic for Concept Decoding Models.

Implements the "Always compare model performance to a shuffled null" diagnostic.
Evaluates whether a trained neural decoding model is genuinely decoding stimulus-related
semantic information or if it is "cheating" by fitting linear boundaries on high-dimensional
noise, data leakage, pseudo-epoch chunking, or class imbalance.

Supports completed batch jobs from `decoding.source-volume-beta.slurm` by directly
reusing pre-extracted `source_features.joblib` and `settings_used.json` without
re-extracting raw FIF files.

Usage:
  # Evaluate the best subset from the completed job:
  python scripts/run_shuffled_null_diagnostic.py --job-dir /path/to/batch_folder

  # Auto-detect latest source-volume-beta run:
  python scripts/run_shuffled_null_diagnostic.py

  # Test full parcel pool with 100 permutations:
  python scripts/run_shuffled_null_diagnostic.py --job-dir /path/to/batch_folder --subset all --n-shuffles 100
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.concept_decoding_subsets import subset_payload
from core.concept_decoding_transformer import compute_transformer_decoding_from_payload
from core.concept_decoding_signal import zero_shot_concept_train_test_split


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def find_default_job_dir() -> Path | None:
    """Search common HPC and local paths for completed decoding runs."""
    search_roots = [
        Path("/scratch/g/bgross/CIDMEG_SM/data/CIDMEG005/ICA/decoding_results"),
        Path("/scratch/g/bgross/CIDMEG_SM/data/CIDMEG005/ICA"),
        Path("/scratch/g/bgross/CIDMEG_SM/mas"),
        Path.cwd() / "decoding_results",
        Path.cwd(),
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        # Look for folders containing source_features.joblib or manifest.json
        for path in root.glob("*/*source_features.joblib"):
            candidates.append(path.parent)
        for path in root.glob("*source_features.joblib"):
            candidates.append(path.parent)
        for path in root.glob("*/*selected_parcel_subsets*"):
            if path.is_dir():
                candidates.append(path)

    unique_candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return unique_candidates[0] if unique_candidates else None


def load_job_artifacts(job_dir: Path) -> dict[str, Any]:
    """Load settings, source features, and screening manifests from a job folder."""
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory does not exist: {job_dir}")

    # If pointed at a specific subset subfolder, check parent for batch files
    if (job_dir / "result_summary.json").is_file() and not (job_dir / "source_features.joblib").is_file():
        parent = job_dir.parent
        if (parent / "source_features.joblib").is_file():
            batch_dir = parent
            specific_subset_dir = job_dir
        else:
            batch_dir = job_dir
            specific_subset_dir = job_dir
    else:
        batch_dir = job_dir
        specific_subset_dir = None

    features_path = batch_dir / "source_features.joblib"
    if not features_path.is_file():
        # Check subdirectories
        matches = list(batch_dir.glob("**/source_features.joblib"))
        if matches:
            features_path = matches[0]
            batch_dir = features_path.parent
        else:
            raise FileNotFoundError(
                f"Could not find source_features.joblib in {job_dir}. "
                "Ensure the job directory contains the pre-extracted source features."
            )

    settings_path = batch_dir / "settings_used.json"
    if not settings_path.is_file() and specific_subset_dir:
        settings_path = specific_subset_dir / "settings_used.json"
    if not settings_path.is_file():
        raise FileNotFoundError(f"Could not find settings_used.json in {batch_dir}")

    manifest_path = batch_dir / "manifest.json"
    comparison_path = batch_dir / "subset_comparison.csv"

    _log(f"Loading pre-extracted features from {features_path}...")
    payload = joblib.load(features_path)
    options = json.loads(settings_path.read_text(encoding="utf-8"))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    comparison_df = pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame()

    return {
        "batch_dir": batch_dir,
        "specific_subset_dir": specific_subset_dir,
        "payload": payload,
        "options": options,
        "manifest": manifest,
        "comparison_df": comparison_df,
    }


def resolve_target_parcels(
    artifacts: dict[str, Any],
    subset_request: str,
) -> tuple[list[str], dict[str, Any]]:
    """Determine the parcels and baseline metrics for the requested subset."""
    manifest = artifacts["manifest"]
    comp_df = artifacts["comparison_df"]
    payload = artifacts["payload"]
    specific_dir = artifacts.get("specific_subset_dir")

    # If pointed at a specific subset subfolder, load its result_summary.json directly
    if specific_dir and (specific_dir / "result_summary.json").is_file():
        summary = json.loads((specific_dir / "result_summary.json").read_text(encoding="utf-8"))
        parcels = summary.get("selected_parcellations", [])
        if not parcels and "roi_names" in summary:
            parcels = summary["roi_names"]
        subset_info = {
            "source": f"directory: {specific_dir.name}",
            "subset_id": summary.get("subset_id", specific_dir.name),
            "n_parcels": len(parcels),
            "parcels": parcels,
            "saved_summary": summary,
        }
        return parcels, subset_info

    all_parcels = payload.get("roi_names")
    if not all_parcels and "feature_meta_df" in payload:
        all_parcels = sorted(payload["feature_meta_df"]["roi"].astype(str).unique())

    req = subset_request.strip().lower()
    if req == "all":
        return list(all_parcels), {
            "source": "all_candidate_parcels",
            "subset_id": "all",
            "n_parcels": len(all_parcels),
            "parcels": list(all_parcels),
            "saved_summary": None,
        }

    if req == "best":
        # 1. Try manifest.json best_parcels
        best_parcels = manifest.get("best_parcels")
        best_id = manifest.get("best_subset_id")
        if best_parcels and isinstance(best_parcels, list):
            saved = None
            if not comp_df.empty and "subset_id" in comp_df.columns and best_id is not None:
                match = comp_df.loc[comp_df["subset_id"].astype(str) == str(best_id)]
                if not match.empty:
                    saved = match.iloc[0].to_dict()
            return best_parcels, {
                "source": "manifest_best",
                "subset_id": best_id or "best",
                "n_parcels": len(best_parcels),
                "parcels": best_parcels,
                "saved_summary": saved,
            }

        # 2. Try comparison_df top row by top1_accuracy or mpr
        if not comp_df.empty and "status" in comp_df.columns:
            complete = comp_df.loc[comp_df["status"] == "complete"].copy()
            if not complete.empty:
                sort_col = "top1_accuracy" if "top1_accuracy" in complete.columns else "mpr"
                best_row = complete.sort_values(sort_col, ascending=False).iloc[0]
                parcels = str(best_row["parcels"]).split(";")
                return parcels, {
                    "source": f"comparison_df_best ({sort_col})",
                    "subset_id": best_row.get("subset_id", "best"),
                    "n_parcels": len(parcels),
                    "parcels": parcels,
                    "saved_summary": best_row.to_dict(),
                }

        # Fallback to all parcels if no manifest / comparison table
        _log("No subset screening ranking found; defaulting to all candidate parcels.")
        return list(all_parcels), {
            "source": "fallback_all",
            "subset_id": "all",
            "n_parcels": len(all_parcels),
            "parcels": list(all_parcels),
            "saved_summary": None,
        }

    # Custom subset ID or parcel string
    if req.isdigit() and not comp_df.empty and "subset_id" in comp_df.columns:
        match = comp_df.loc[comp_df["subset_id"].astype(str) == req]
        if not match.empty:
            row = match.iloc[0]
            parcels = str(row["parcels"]).split(";")
            return parcels, {
                "source": f"subset_id_{req}",
                "subset_id": req,
                "n_parcels": len(parcels),
                "parcels": parcels,
                "saved_summary": row.to_dict(),
            }

    # Delimited parcel list
    delimiters = [";", ","]
    for d in delimiters:
        if d in subset_request:
            parcels = [p.strip() for p in subset_request.split(d) if p.strip()]
            return parcels, {
                "source": "custom_parcel_list",
                "subset_id": "custom",
                "n_parcels": len(parcels),
                "parcels": parcels,
                "saved_summary": None,
            }

    # Single parcel
    if subset_request in all_parcels:
        return [subset_request], {
            "source": "single_parcel",
            "subset_id": subset_request,
            "n_parcels": 1,
            "parcels": [subset_request],
            "saved_summary": None,
        }

    raise ValueError(f"Could not resolve requested subset: {subset_request}")


def create_shuffled_null_payload(
    payload: dict[str, Any],
    rng: np.random.Generator,
    mode: str = "concept_mapping",
    zero_shot: bool = True,
    zero_shot_random_state: int = 42,
    zero_shot_unseen_fraction: float = 0.2,
    explicit_split: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Create a null payload with permuted labels/concept mappings.

    In zero-shot mode:
    - Permutes concept-to-target assignments strictly within the training & validation trials.
    - Test concepts remain completely unseen and test trials remain untouched.
    - Preserves trial timing, covariance, and within-concept trial structure.
    """
    null_payload = dict(payload)
    y = np.asarray(payload["y"], dtype=str).copy()
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32).copy() if "embeddings" in payload else None
    trial_df = payload["trial_df"].copy() if "trial_df" in payload else pd.DataFrame()

    if explicit_split is not None:
        available = list(explicit_split["train"]) + list(explicit_split["validation"])
    elif zero_shot:
        train_idx, _, _ = zero_shot_concept_train_test_split(
            y,
            unseen_fraction=zero_shot_unseen_fraction,
            random_state=zero_shot_random_state,
        )
        available = list(train_idx)
    else:
        available = list(range(len(y)))

    train_labels = sorted(set(y[available]))

    if mode == "concept_mapping":
        # Permute concept identities (preserves trial groupings within each concept)
        shuffled_labels = rng.permutation(train_labels)
        mapping = dict(zip(train_labels, shuffled_labels))
        target_map = {l: embeddings[np.flatnonzero(y == l)[0]].copy() for l in train_labels} if embeddings is not None else {}

        for i in available:
            old_label = y[i]
            new_label = mapping[old_label]
            y[i] = new_label
            if embeddings is not None:
                embeddings[i] = target_map[new_label]
            for col in ("concept_label", "canonical_stimulus", "embedding_key"):
                if col in trial_df.columns:
                    trial_df.loc[i, col] = new_label
    else:  # trial_labels
        # Scramble labels across trials independently
        perm_indices = rng.permutation(available)
        y[available] = y[perm_indices]
        if embeddings is not None:
            embeddings[available] = embeddings[perm_indices]
        for col in ("concept_label", "canonical_stimulus", "embedding_key"):
            if col in trial_df.columns:
                trial_df.loc[available, col] = trial_df.loc[perm_indices, col].to_numpy()

    null_payload["y"] = y
    if embeddings is not None:
        null_payload["embeddings"] = embeddings
    null_payload["trial_df"] = trial_df
    return null_payload


def compute_diagnostics(
    real_metrics: dict[str, float],
    null_records: list[dict[str, Any]],
    n_classes: int,
) -> dict[str, Any]:
    """Calculate empirical p-values, z-scores, and quality checks against the null distribution."""
    df_null = pd.DataFrame(null_records)
    summary: dict[str, Any] = {}

    metrics = [
        ("mpr", 0.50),
        ("top1_accuracy", 1.0 / max(1, n_classes)),
        ("top5_accuracy", min(1.0, 5.0 / max(1, n_classes))),
        ("balanced_accuracy", 1.0 / max(1, n_classes)),
    ]

    for key, chance in metrics:
        if key not in real_metrics or key not in df_null.columns:
            continue
        real_val = float(real_metrics[key])
        null_vals = pd.to_numeric(df_null[key], errors="coerce").dropna().to_numpy()
        if not null_vals.size:
            continue

        n_completed = len(null_vals)
        n_greater_equal = int(np.sum(null_vals >= real_val))
        empirical_p = float((1 + n_greater_equal) / (n_completed + 1))

        mean = float(np.mean(null_vals))
        std = float(np.std(null_vals, ddof=1)) if n_completed > 1 else 0.0
        z_score = float((real_val - mean) / std) if std > 1e-8 else (0.0 if np.isclose(real_val, mean) else np.nan)

        summary[key] = {
            "real": real_val,
            "theoretical_chance": float(chance),
            "null_mean": mean,
            "null_std": std,
            "null_min": float(np.min(null_vals)),
            "null_max": float(np.max(null_vals)),
            "null_p05": float(np.percentile(null_vals, 5)),
            "null_p95": float(np.percentile(null_vals, 95)),
            "z_score": z_score,
            "empirical_p_value": empirical_p,
            "n_permutations": n_completed,
            "null_greater_or_equal_count": n_greater_equal,
        }

    # Methodological Quality Checks
    mpr_stats = summary.get("mpr", {})
    null_mpr_mean = mpr_stats.get("null_mean", np.nan)
    null_mpr_max = mpr_stats.get("null_max", np.nan)
    p_mpr = mpr_stats.get("empirical_p_value", 1.0)
    z_mpr = mpr_stats.get("z_score", 0.0)

    checks: dict[str, Any] = {}
    # Check 1: Null centered near theoretical chance (0.500)
    chance_drift = abs(null_mpr_mean - 0.500) if np.isfinite(null_mpr_mean) else np.nan
    checks["null_centered_at_chance"] = bool(chance_drift <= 0.03)
    checks["null_mpr_chance_delta"] = float(chance_drift) if np.isfinite(chance_drift) else None

    # Check 2: Model cheating / spurious boundary risk
    # If null models achieve high MPR (> 0.55) without real labels, pipeline might be leaking
    checks["spurious_boundary_risk"] = "high" if null_mpr_max > 0.55 else ("moderate" if null_mpr_max > 0.53 else "low")

    # Check 3: Statistical significance
    checks["statistically_significant"] = bool(p_mpr < 0.05 and z_mpr >= 2.0)

    # Overall Verdict
    if checks["statistically_significant"] and checks["null_centered_at_chance"] and checks["spurious_boundary_risk"] == "low":
        verdict = "PASS"
        verdict_message = (
            "Model performance reliably exceeds the retrained shuffled null. "
            "Null distribution collapsed cleanly to theoretical chance with low leakage risk."
        )
    elif checks["spurious_boundary_risk"] == "high":
        verdict = "WARNING"
        verdict_message = (
            "Null models achieved above-chance decoding without labels (max null MPR > 0.55). "
            "High risk of feature leakage, temporal autocorrelation, or boundary overfitting."
        )
    elif not checks["null_centered_at_chance"]:
        verdict = "WARNING"
        verdict_message = (
            f"Null distribution is shifted from theoretical chance (|mean - 0.5| = {chance_drift:.3f}). "
            "Examine class balance or split stratification."
        )
    elif not checks["statistically_significant"]:
        verdict = "FAIL"
        verdict_message = (
            f"Real model does not significantly outperform the shuffled null (p={p_mpr:.4f}, z={z_mpr:.2f}). "
            "Model may not be extracting genuine concept signal beyond chance."
        )
    else:
        verdict = "INCONCLUSIVE"
        verdict_message = "Review detailed metrics below."

    return {
        "verdict": verdict,
        "verdict_message": verdict_message,
        "metrics": summary,
        "checks": checks,
    }


def plot_diagnostics(
    summary: dict[str, Any],
    df_null: pd.DataFrame,
    output_png: Path,
) -> None:
    """Render publication-quality 2-panel null distribution vs real model plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    metrics_info = [
        ("mpr", "Mean Percentile Rank (MPR)", axes[0]),
        ("top1_accuracy", "Top-1 Accuracy", axes[1]),
    ]

    for key, label, ax in metrics_info:
        stats = summary.get("metrics", {}).get(key)
        if not stats or key not in df_null.columns:
            ax.set_title(label)
            continue

        null_vals = pd.to_numeric(df_null[key], errors="coerce").dropna().to_numpy()
        real_val = stats["real"]
        chance_val = stats["theoretical_chance"]
        p_val = stats["empirical_p_value"]
        z_val = stats["z_score"]

        # Histogram
        counts, bins, _ = ax.hist(
            null_vals,
            bins=min(15, max(6, len(null_vals) // 3)),
            color="#3274ad",
            edgecolor="white",
            alpha=0.75,
            density=True,
            label=f"Shuffled Null (N={len(null_vals)})",
        )

        # Theoretical chance line
        ax.axvline(
            chance_val,
            color="#6c757d",
            linestyle="--",
            linewidth=1.8,
            label=f"Chance ({chance_val:.3f})",
        )

        # Real model line
        ax.axvline(
            real_val,
            color="#d9534f",
            linestyle="-",
            linewidth=2.2,
            label=f"Real Model ({real_val:.3f})",
        )

        # Text annotation
        z_str = f"z = {z_val:+.2f} SD" if np.isfinite(z_val) else ""
        ax.text(
            0.05,
            0.92,
            f"Empirical p = {p_val:.4f}\n{z_str}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ced4da", alpha=0.9),
        )

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{label}: Real vs. Null", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    verdict = summary.get("verdict", "DIAGNOSTIC")
    fig.suptitle(f"Shuffled-Null Retraining Diagnostic [{verdict}]", fontsize=13, fontweight="bold")
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    _log(f"Saved diagnostic figure: {output_png}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone Shuffled-Null Retraining Diagnostic for Concept Decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help="Path to the completed job directory (e.g. *_selected_parcel_subsets).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="best",
        help="Which parcel subset to evaluate: 'best' (from manifest), 'all' (candidate pool), or specific ID/parcel names.",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=50,
        help="Number of retrained null models to fit.",
    )
    parser.add_argument(
        "--shuffle-mode",
        choices=["concept_mapping", "trial_labels"],
        default="concept_mapping",
        help="Shuffling mechanism: 'concept_mapping' (preserves trial groups within concepts) or 'trial_labels' (trial-wise).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save diagnostics (defaults to <job_dir>/shuffled_null_diagnostics).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=20261019,
        help="Base random seed for generating label permutations.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating diagnostic plots.",
    )
    parser.add_argument(
        "--force-refit-real",
        action="store_true",
        help="Always refit the real model from scratch even if result_summary.json exists.",
    )

    args = parser.parse_args(argv)

    # 1. Locate Job Directory
    job_dir = args.job_dir
    if job_dir is None:
        job_dir = find_default_job_dir()
        if job_dir is None:
            print(
                "Error: No job directory specified and could not auto-detect a completed run.\n"
                "Please provide --job-dir /path/to/decoding_results/your_job_folder",
                file=sys.stderr,
            )
            return 2
        _log(f"Auto-detected job directory: {job_dir}")
    else:
        job_dir = job_dir.resolve()

    # 2. Load Artifacts
    artifacts = load_job_artifacts(job_dir)
    batch_dir = artifacts["batch_dir"]
    payload = artifacts["payload"]
    options = artifacts["options"]

    # Ensure required metadata exists in payload
    payload.setdefault("window_start", options.get("window_start", 0.0))
    payload.setdefault("window_end", options.get("window_end", 0.5))
    payload.setdefault("channel_mode", options.get("channel_mode", "meg"))
    payload.setdefault("group_mode", options.get("group_mode", "canonical"))

    out_dir = args.output_dir or (batch_dir / "shuffled_null_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Resolve Target Parcels
    parcels, subset_info = resolve_target_parcels(artifacts, args.subset)
    _log(
        f"Evaluating subset: {subset_info['subset_id']} ({len(parcels)} parcels from {subset_info['source']})\n"
        f"Parcels: {', '.join(parcels)}"
    )

    # Prepare payload subset
    selected_payload = subset_payload(payload, parcels)
    run_options = dict(
        options,
        selected_parcellations=list(parcels),
        n_permutations=0,  # Disable internal post-hoc shuffle; we are doing retrained nulls
        progress_callback=None,
    )

    # 4. Determine Real Model Baseline
    saved_summary = subset_info.get("saved_summary")
    real_metrics: dict[str, float] = {}

    if not args.force_refit_real and saved_summary and "mpr" in saved_summary:
        _log(f"Reusing archived real model metrics from {subset_info['source']}")
        real_metrics = {
            "mpr": float(saved_summary.get("mpr", np.nan)),
            "top1_accuracy": float(saved_summary.get("top1_accuracy", np.nan)),
            "top5_accuracy": float(saved_summary.get("top5_accuracy", np.nan)),
            "balanced_accuracy": float(saved_summary.get("balanced_accuracy", saved_summary.get("top1_accuracy", np.nan))),
        }
    else:
        _log("Fitting real baseline model...")
        real_result, real_msg = compute_transformer_decoding_from_payload(None, run_options, selected_payload)
        if real_result is None:
            print(f"Error fitting real baseline model: {real_msg}", file=sys.stderr)
            return 1
        real_metrics = {
            "mpr": float(real_result.get("mpr", np.nan)),
            "top1_accuracy": float(real_result.get("top1_accuracy", np.nan)),
            "top5_accuracy": float(real_result.get("top5_accuracy", np.nan)),
            "balanced_accuracy": float(real_result.get("balanced_accuracy", real_result.get("top1_accuracy", np.nan))),
        }

    _log(
        f"Real Model Baseline: MPR = {real_metrics['mpr']:.4f}, "
        f"Top-1 = {real_metrics['top1_accuracy'] * 100:.2f}%, "
        f"Top-5 = {real_metrics['top5_accuracy'] * 100:.2f}%"
    )

    # Check classes
    n_classes = len(set(payload.get("y", [])))
    is_zero_shot = str(options.get("split_mode", "zero_shot")).lower() == "zero_shot"

    # 5. Run Retrained Shuffled Null Permutations
    _log(f"Starting {args.n_shuffles} retrained shuffled-null models (mode: {args.shuffle_mode})...")
    null_records: list[dict[str, Any]] = []
    start_time = time.time()

    for idx in range(1, args.n_shuffles + 1):
        perm_rng = np.random.default_rng(args.shuffle_seed + idx)
        null_payload = create_shuffled_null_payload(
            selected_payload,
            rng=perm_rng,
            mode=args.shuffle_mode,
            zero_shot=is_zero_shot,
            zero_shot_random_state=int(options.get("zero_shot_random_state", 42)),
            zero_shot_unseen_fraction=float(options.get("zero_shot_unseen_fraction", 0.2)),
            explicit_split=options.get("explicit_split"),
        )

        null_result, err = compute_transformer_decoding_from_payload(None, run_options, null_payload)
        if null_result is None:
            _log(f"Warning: Permutation {idx}/{args.n_shuffles} failed: {err}")
            continue

        record = {
            "permutation": idx,
            "mpr": float(null_result.get("mpr", np.nan)),
            "top1_accuracy": float(null_result.get("top1_accuracy", np.nan)),
            "top5_accuracy": float(null_result.get("top5_accuracy", np.nan)),
            "balanced_accuracy": float(null_result.get("balanced_accuracy", null_result.get("top1_accuracy", np.nan))),
            "validation_mpr": float(null_result.get("validation_mpr", np.nan)),
            "validation_top1": float(null_result.get("validation_top1_accuracy", np.nan)),
        }
        null_records.append(record)

        if idx % 10 == 0 or idx == args.n_shuffles:
            elapsed = time.time() - start_time
            rate = idx / max(1e-9, elapsed)
            remaining = (args.n_shuffles - idx) / max(1e-9, rate)
            _log(
                f"Completed {idx}/{args.n_shuffles} null models "
                f"({rate:.1f} models/sec; ~{remaining:.1f}s remaining); "
                f"latest null MPR = {record['mpr']:.4f}, Top-1 = {record['top1_accuracy'] * 100:.2f}%"
            )

    df_null = pd.DataFrame(null_records)
    if df_null.empty:
        print("Error: All null models failed to train.", file=sys.stderr)
        return 1

    # 6. Compute Diagnostics & Quality Checks
    diagnostics = compute_diagnostics(real_metrics, null_records, n_classes)
    verdict = diagnostics["verdict"]

    output_summary = {
        "verdict": verdict,
        "verdict_message": diagnostics["verdict_message"],
        "target_job": str(batch_dir),
        "evaluated_subset": subset_info,
        "shuffle_mode": args.shuffle_mode,
        "n_permutations_requested": args.n_shuffles,
        "n_permutations_completed": len(df_null),
        "real_metrics": real_metrics,
        "null_diagnostics": diagnostics["metrics"],
        "quality_checks": diagnostics["checks"],
        "timestamp": _timestamp(),
    }

    # 7. Save Files
    summary_path = out_dir / "shuffled_null_summary.json"
    iterations_path = out_dir / "shuffled_null_iterations.csv"
    plot_path = out_dir / "shuffled_null_diagnostic.png"

    summary_path.write_text(json.dumps(output_summary, indent=2), encoding="utf-8")
    df_null.to_csv(iterations_path, index=False)
    _log(f"Wrote summary JSON: {summary_path}")
    _log(f"Wrote iterations CSV: {iterations_path}")

    if not args.no_plots:
        try:
            plot_diagnostics(output_summary, df_null, plot_path)
        except Exception as exc:
            _log(f"Warning: Plot generation failed: {exc}")

    # 8. Terminal Report
    mpr_info = diagnostics["metrics"].get("mpr", {})
    top1_info = diagnostics["metrics"].get("top1_accuracy", {})

    print("\n" + "=" * 76)
    print(f"SHUFFLED-NULL DIAGNOSTIC VERDICT: {verdict}")
    print("=" * 76)
    print(f"Evaluated Subset: {subset_info['subset_id']} ({len(parcels)} parcels)")
    print(f"Parcels:          {', '.join(parcels)}")
    print(f"Shuffle Mode:      {args.shuffle_mode} (N = {len(df_null)} retrained models)")
    print("-" * 76)
    print(f"Real Model MPR:    {real_metrics['mpr']:.4f}")
    print(
        f"Shuffled Null MPR: {mpr_info.get('null_mean', np.nan):.4f} ± {mpr_info.get('null_std', np.nan):.4f} "
        f"[95% range: {mpr_info.get('null_p05', np.nan):.4f} - {mpr_info.get('null_p95', np.nan):.4f}]"
    )
    print(f"Empirical p-value: {mpr_info.get('empirical_p_value', np.nan):.4f} ({mpr_info.get('null_greater_or_equal_count', 0)}/{len(df_null)} nulls ≥ real)")
    print(f"Z-Score:           {mpr_info.get('z_score', np.nan):+.2f} SD above null mean")
    print(f"Chance Baseline:   0.5000 (null deviation from chance: {diagnostics['checks'].get('null_mpr_chance_delta', 0.0):.4f})")
    print("-" * 76)
    print(f"Real Top-1 Acc:    {real_metrics['top1_accuracy'] * 100:.2f}%")
    print(
        f"Null Top-1 Acc:    {top1_info.get('null_mean', np.nan) * 100:.2f}% ± {top1_info.get('null_std', np.nan) * 100:.2f}% "
        f"(Empirical p = {top1_info.get('empirical_p_value', np.nan):.4f})"
    )
    print("-" * 76)
    print(f"Assessment: {diagnostics['verdict_message']}")
    print("=" * 76 + "\n")

    return 0 if verdict == "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
