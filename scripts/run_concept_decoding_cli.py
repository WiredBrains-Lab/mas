#!/usr/bin/env python3
"""Run MEG concept decoding from the command line.

Quick start (Python 3.11+)
-------------------------
Edit configs/decoding.toml to set your default FIF path and decoder, then run:
    python scripts/run_concept_decoding_cli.py

Override the defaults:
    python scripts/run_concept_decoding_cli.py --fif /data/run.fif --decoder logistic
    python scripts/run_concept_decoding_cli.py --fif /data/run.fif --prepare-only
    python scripts/run_concept_decoding_cli.py --fif /data/run.fif --all-runs
    python scripts/run_concept_decoding_cli.py --config configs/decoding.rcc.toml

Decoders without semantic dictionaries: logistic, template, mlp, pairwise_svm.
Semantic decoders (require --embedding-dict): embedding_ridge,
embedding_contrastive, embedding_mlp, transformer.
See --help for all options and configs/decoding.toml for annotated defaults.
FIF inputs automatically create/reuse trial tables beside each analysis FIF.
The older --trial-table input remains available for exported/combined tables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "decoding.toml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.concept_decoding_signal import (  # noqa: E402
    concept_label_from_trial_row,
    compute_signal_space_concept_decoding,
)


DECODER_CHOICES = (
    "pairwise_svm",
    "logistic",
    "template",
    "mlp",
    "embedding_ridge",
    "embedding_contrastive",
    "embedding_mlp",
    "transformer",
)
TABLE_SUFFIXES = {".csv", ".tsv", ".tab", ".xlsx", ".xls", ".parquet", ".pq", ".json", ".jsonl"}
KNOWN_OUTPUT_FILES = (
    "timecourse.csv", "pairwise_accuracy.csv", "pairwise_accuracy.npz",
    "pairwise_trial_index.csv", "pairwise_timecourse.png",
    "run_options.json",
    "input_trial_table.csv",
    "summary.csv",
    "summary.json",
    "metrics.csv",
    "predictions.csv",
    "feature_importance.csv",
    "model_architecture.csv",
    "split.csv",
    "qc_summary.csv",
    "trial_qc.csv",
    "per_class_accuracy.csv",
    "permutation_baseline.csv",
    "template_top_features.csv",
    "template_feature_meta.csv",
    "template_matrix.npy",
    "transformer_train_loss.csv",
    "confusion_matrix.csv",
    "class_names.txt",
    "input_concept_counts.csv",
    "confusion_matrix.png",
    "per_class_accuracy.png",
    "transformer_train_loss.png",
    "permutation_baseline.png",
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _progress(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


class _LogEmitter:
    def emit(self, message: str) -> None:
        _progress(str(message))


def _canonicalize_meg_stimulus_label(stimulus_name: Any) -> str:
    if stimulus_name is None:
        return ""
    stim = os.path.basename(str(stimulus_name).strip())
    stem, _ = os.path.splitext(stim)
    m = re.match(r"^(.*?)(?:_\d+_\d+)$", stem)
    if m and m.group(1):
        return str(m.group(1))
    m2 = re.match(r"^(.*?)(?:_\d+)$", stem)
    if m2 and m2.group(1):
        return str(m2.group(1))
    return str(stem)


def _clean_text(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in TABLE_SUFFIXES:
        raise ValueError(f"Unsupported table format {suffix!r}. Supported: {sorted(TABLE_SUFFIXES)}")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def _write_dataframe(df: Any, path: Path, *, index: bool = False) -> bool:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return True


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(val) for val in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(val) for val in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _parse_path_maps(raw_maps: list[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for raw in raw_maps:
        if "=" not in str(raw):
            raise ValueError(f"Invalid --path-map {raw!r}; expected OLD=NEW.")
        old, new = str(raw).split("=", 1)
        old = os.path.expandvars(os.path.expanduser(old.strip()))
        new = os.path.expandvars(os.path.expanduser(new.strip()))
        if not old or not new:
            raise ValueError(f"Invalid --path-map {raw!r}; OLD and NEW must be non-empty.")
        mappings.append((old.rstrip(os.sep), new.rstrip(os.sep)))
    return mappings


def _remap_path(path_value: Any, path_maps: list[tuple[str, str]], base_dir: Path) -> str:
    text = _clean_text(path_value)
    if not text:
        return ""
    text = os.path.expandvars(os.path.expanduser(text))
    for old, new in path_maps:
        if text == old:
            text = new
            break
        prefix = old + os.sep
        if text.startswith(prefix):
            text = new + text[len(old) :]
            break
    if not os.path.isabs(text):
        text = str((base_dir / text).resolve(strict=False))
    else:
        text = os.path.abspath(text)
    return text


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in df.columns}


def _load_stimulus_group_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    df = _read_table(path)
    if df.empty:
        return {}
    cols = _column_lookup(df)
    stim_col = (
        cols.get("stimulus")
        or cols.get("canonical_stimulus")
        or cols.get("concept")
        or cols.get("concept_label")
        or cols.get("name")
    )
    group_col = (
        cols.get("stimulus_group")
        or cols.get("category")
        or cols.get("classification")
        or cols.get("group")
    )
    if stim_col is None or group_col is None:
        available = ", ".join(str(col) for col in df.columns)
        raise ValueError(
            "Stimulus group map needs stimulus/canonical_stimulus plus stimulus_group/category columns. "
            f"Available columns: {available}"
        )
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        stim = _clean_text(row.get(stim_col))
        group = _clean_text(row.get(group_col))
        canonical = _canonicalize_meg_stimulus_label(stim)
        if canonical and group:
            mapping[canonical] = group
    return mapping


def _read_auto_group_map_for_fif(fif_filename: Any) -> dict[str, str]:
    fif_path = os.path.abspath(str(fif_filename or "").strip())
    if not fif_path:
        return {}
    csv_path = Path(fif_path).parent.parent / "stimulus_with_groups.csv"
    if not csv_path.is_file():
        return {}
    return _load_stimulus_group_map(csv_path)


class HeadlessConceptDecodingContext:
    """Small context shim exposing the hooks expected by the GUI decoder backend."""

    def __init__(
        self,
        trial_df: pd.DataFrame,
        *,
        first_fif_path: str,
        stimulus_group_mapping: dict[str, str] | None = None,
        auto_stimulus_group_map: bool = True,
        override_stimulus_groups: bool = False,
    ) -> None:
        self.b_fif_loaded = True
        self._fif_loaded = True
        self._fif_filename = str(first_fif_path or "")
        self._fif_raw = None
        self._fif_trial_response_table = trial_df.copy()
        self._explicit_stimulus_group_mapping = dict(stimulus_group_mapping or {})
        self._auto_stimulus_group_map = bool(auto_stimulus_group_map)
        self._override_stimulus_groups = bool(override_stimulus_groups)
        self._meg_stimulus_group_cache: dict[str, dict[str, str]] = {}
        self.c = SimpleNamespace(log=_LogEmitter())

    @staticmethod
    def _canonicalize_meg_stimulus_label(stimulus_name: Any) -> str:
        return _canonicalize_meg_stimulus_label(stimulus_name)

    def _stimulus_group_mapping_for_fif(self, fif_filename: Any) -> dict[str, str]:
        if self._explicit_stimulus_group_mapping:
            return dict(self._explicit_stimulus_group_mapping)
        if not self._auto_stimulus_group_map:
            return {}
        fif_path = os.path.abspath(str(fif_filename or "").strip())
        if not fif_path:
            return {}
        if fif_path not in self._meg_stimulus_group_cache:
            self._meg_stimulus_group_cache[fif_path] = _read_auto_group_map_for_fif(fif_path)
        return dict(self._meg_stimulus_group_cache.get(fif_path, {}))

    def _stimulus_group_mapping_for_trials(self, trial_df: pd.DataFrame, fallback_fif: Any) -> dict[str, str]:
        if self._explicit_stimulus_group_mapping:
            return dict(self._explicit_stimulus_group_mapping)
        mapping = self._stimulus_group_mapping_for_fif(fallback_fif)
        if isinstance(trial_df, pd.DataFrame) and "source_fif_path" in trial_df.columns:
            for fif_path in pd.Index(trial_df["source_fif_path"].astype(str)).drop_duplicates().tolist():
                mapping.update(self._stimulus_group_mapping_for_fif(fif_path))
        return mapping

    def _annotate_trial_df_with_stimulus_groups(self, trial_df: pd.DataFrame, fif_filename: Any) -> pd.DataFrame:
        if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
            return trial_df
        df = trial_df.copy()
        if "stimulus" in df.columns:
            if "canonical_stimulus" not in df.columns:
                df["canonical_stimulus"] = df["stimulus"].map(self._canonicalize_meg_stimulus_label)
            else:
                fallback = df["stimulus"].map(self._canonicalize_meg_stimulus_label)
                df["canonical_stimulus"] = df["canonical_stimulus"].where(
                    df["canonical_stimulus"].astype(str).str.strip().astype(bool),
                    fallback,
                )
        elif "canonical_stimulus" not in df.columns and "concept" in df.columns:
            df["canonical_stimulus"] = df["concept"].map(self._canonicalize_meg_stimulus_label)

        mapping = self._stimulus_group_mapping_for_trials(df, fif_filename)
        if "stimulus_group" not in df.columns:
            df["stimulus_group"] = ""
        if mapping and "canonical_stimulus" in df.columns:
            mapped = df["canonical_stimulus"].map(lambda x: mapping.get(str(x).strip(), ""))
            if self._override_stimulus_groups:
                df["stimulus_group"] = mapped
            else:
                existing = df["stimulus_group"].astype(str).str.strip()
                df["stimulus_group"] = existing.where(existing.astype(bool), mapped)
        return df


def select_fif_runs(args: argparse.Namespace) -> list[str]:
    """Resolve optional sibling runs and avoid counting split parts twice."""
    import mne

    folder = getattr(args, "fif_folder", None)
    if folder and not Path(folder).is_dir():
        raise NotADirectoryError(f"FIF folder does not exist or is not a directory: {folder}")
    explicit = list(dict.fromkeys(args.fif or []))
    interactive = sys.stdin.isatty() and not os.environ.get("SLURM_JOB_ID")
    discover = args.all_runs is True or (args.all_runs is None and interactive and len(explicit) == 1)
    candidates = list(explicit)
    if discover or folder:
        folders = [Path(folder)] if folder else dict.fromkeys(Path(p).parent for p in explicit)
        for directory in folders:
            candidates.extend(str(p.resolve()) for p in sorted(directory.iterdir())
                              if p.is_file() and p.name.lower().endswith((".fif", ".fif.gz")))
    recordings = {}
    for path in dict.fromkeys(candidates):
        try:
            with mne.io.read_raw_fif(path, preload=False, verbose="error") as raw:
                # Header inspection excludes inverse/ICA/epochs FIFs, and the
                # recording's actual split-file links identify continuation parts.
                recordings[path] = [str(Path(p).resolve()) for p in raw.filenames if p]
        except Exception as exc:
            if path in explicit:
                raise ValueError(f"Cannot read selected raw FIF {path}: {exc}") from exc
            _progress(f"Skipping non-raw or unreadable neighboring FIF {path}: {exc}")
    continuation = {part for parts in recordings.values() for part in parts[1:]}
    roots = [path for path in recordings if path not in continuation]
    selected = [path for path in roots if path in explicit or any(p in explicit for p in recordings[path])]
    neighbors = [path for path in roots if path not in selected]
    include = bool(folder) or args.all_runs is True
    if neighbors and discover and args.all_runs is None and not folder:
        print("Other raw FIF recordings in the same folder:", flush=True)
        for path in neighbors:
            print(f"  {Path(path).name}", flush=True)
        try:
            include = input(f"Include these {len(neighbors)} additional recording(s)? [y/N] ").strip().lower() in {"y", "yes"}
        except EOFError:
            include = False
    if include:
        selected.extend(neighbors)
    if not selected:
        raise ValueError(f"No readable raw FIF recordings found in {folder or explicit}.")
    _progress(f"Selected {len(selected)} recording(s): " + ", ".join(Path(p).name for p in selected))
    return selected


def save_combined_trial_table(table: pd.DataFrame, args: argparse.Namespace) -> None:
    """Save a fresh selection snapshot; never append to a previous selection."""
    import hashlib
    from core.fif_processing import FIFProcessingMixin

    paths = list(dict.fromkeys(table["source_fif_path"].astype(str)))
    if len(paths) < 2:
        return
    selection_id = hashlib.sha256(json.dumps(sorted(paths)).encode()).hexdigest()[:12]
    destination = Path(paths[0] + f".combined-{selection_id}.trial_table.csv")
    stored = table.copy()
    if "channel_delta_pairs" in stored:
        stored["channel_delta_pairs"] = stored["channel_delta_pairs"].map(json.dumps)
    FIFProcessingMixin._atomic_write_text(destination, stored.to_csv(index=False))
    args.combined_trial_table = str(destination)
    _progress(f"Combined trial table: {destination} ({len(table)} trials from {len(paths)} recordings)")


def load_and_prepare_trial_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.fif or args.fif_folder:
        from core.fif_processing import HeadlessFIFProcessor

        args.requested_fif = list(args.fif or [])
        args.fif = select_fif_runs(args)
        processor = HeadlessFIFProcessor(_progress, rebuild_trial_table=args.rebuild_trial_table, ica=args.ica)
        tables = []
        args.prepared_runs = []
        prepared_paths = set()
        for path in dict.fromkeys(args.fif):
            table, metadata = processor.prepare(path)
            analysis_path = str(table["source_fif_path"].iloc[0])
            if analysis_path in prepared_paths:
                _progress(f"Skipping duplicate analysis recording: {analysis_path}")
                continue
            prepared_paths.add(analysis_path)
            tables.append(table)
            args.prepared_runs.append(metadata)
        combined = pd.concat(tables, ignore_index=True)
        # Keep the existing run-local trial_index for backend compatibility.
        combined.insert(0, "combined_trial_index", np.arange(1, len(combined) + 1))
        save_combined_trial_table(combined, args)
        return combined
    table_path = Path(args.trial_table).expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"Trial table not found: {table_path}")
    df = _read_table(table_path)
    if df.empty:
        raise ValueError("Trial table is empty.")

    fif_col = str(args.fif_path_column or "source_fif_path")
    if "source_fif_path" not in df.columns and fif_col in df.columns:
        df["source_fif_path"] = df[fif_col]
    elif "source_fif_path" not in df.columns:
        raise ValueError(f"Trial table needs a source FIF path column. Pass --fif-path-column if it is not source_fif_path.")

    if "onset_sample" not in df.columns:
        raise ValueError("Trial table needs an onset_sample column with event onset samples relative to each FIF file.")

    label_cols = {"stimulus", "canonical_stimulus", "stimulus_group", "concept", "concept_label"}
    if not any(col in df.columns for col in label_cols):
        raise ValueError(
            "Trial table needs at least one label column: stimulus, canonical_stimulus, stimulus_group, concept, or concept_label."
        )

    path_maps = _parse_path_maps(args.path_map or [])
    df["source_fif_path"] = [
        _remap_path(value, path_maps, table_path.parent) for value in df["source_fif_path"].tolist()
    ]
    if "source_fif" not in df.columns:
        df["source_fif"] = [Path(path).name for path in df["source_fif_path"].astype(str).tolist()]
    if "trial_index" not in df.columns:
        df["trial_index"] = np.arange(len(df), dtype=int)
    return df


def _first_nonempty_fif_path(df: pd.DataFrame) -> str:
    for value in df.get("source_fif_path", pd.Series(dtype=str)).astype(str).tolist():
        if value.strip():
            return value
    return ""


def source_run_mappings(args: argparse.Namespace, trial_df: pd.DataFrame) -> list[dict[str, str]]:
    """Resolve explicit analysis-FIF → inverse/anatomy associations."""
    paths = list(dict.fromkeys(os.path.abspath(str(path)) for path in trial_df["source_fif_path"]))
    if args.allow_missing_fif:
        paths = [path for path in paths if Path(path).is_file()]
    if not paths:
        raise ValueError("No analysis FIFs remain for source decoding.")
    path_maps = _parse_path_maps(args.path_map)
    if args.source_manifest:
        manifest_path = Path(args.source_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("runs") if isinstance(manifest, dict) else manifest
        if not isinstance(rows, list) or not rows:
            raise ValueError("Source manifest must contain a nonempty 'runs' list.")
        base = manifest_path.parent
    else:
        if len(paths) != 1:
            raise ValueError("--source-inverse supports one analysis FIF; use --source-manifest for multiple runs.")
        rows = [{"fif": paths[0], "inverse": args.source_inverse}]
        base = Path.cwd()
    mapping = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("fif") or not row.get("inverse"):
            raise ValueError("Every source manifest run requires 'fif' and 'inverse'.")
        resolved = {
            key: _remap_path(row[key], path_maps, base) for key in ("fif", "inverse")
        }
        if resolved["fif"] in mapping:
            raise ValueError(f"Duplicate source manifest FIF: {resolved['fif']}")
        subject = str(row.get("subject") or args.source_subject or "").strip()
        subjects_dir = row.get("subjects_dir") or args.subjects_dir
        if not subject or not subjects_dir:
            raise ValueError("Source runs require subject and subjects_dir, via flags or manifest entries.")
        if Path(subject).name != subject or subject in {".", ".."}:
            raise ValueError("Source subject must be a FreeSurfer subject name, not a path.")
        resolved.update(subject=subject, subjects_dir=_remap_path(subjects_dir, path_maps, base))
        mapping[resolved["fif"]] = resolved
    missing = set(paths) - set(mapping)
    if missing:
        raise ValueError(f"Source manifest has no entry for analysis FIFs: {sorted(missing)}")
    return [mapping[path] for path in paths]


def build_context(args: argparse.Namespace, trial_df: pd.DataFrame) -> HeadlessConceptDecodingContext:
    group_map_path = Path(args.stimulus_group_map).expanduser().resolve() if args.stimulus_group_map else None
    group_map = _load_stimulus_group_map(group_map_path)
    context = HeadlessConceptDecodingContext(
        trial_df,
        first_fif_path=_first_nonempty_fif_path(trial_df),
        stimulus_group_mapping=group_map,
        auto_stimulus_group_map=not bool(args.no_auto_stimulus_group_map),
        override_stimulus_groups=bool(args.override_stimulus_groups),
    )
    if args.feature_space == "source":
        from core.concept_decoding_headless_source import HeadlessSourceContext
        source_context = HeadlessSourceContext(source_run_mappings(args, trial_df), args.source_space_kind)
        source_context.attach(context)
    return context


def resolve_source_parcels(args, context):
    """Resolve presets before preflight, model counting, and saved run options."""
    if not args.source_parcel_filter and args.source_hemisphere == "both":
        return
    from core.source_parcellation import filter_parcels
    from core.concept_decoding_source import concept_source_labels_for_subject, concept_volume_labels_for_subject

    viewer = context.meg_3d
    available = None
    checked = set()
    for row in viewer.runs.values():
        key = (row["subject"], row["subjects_dir"])
        if key in checked:
            continue
        checked.add(key)
        if args.source_space_kind == "volume":
            names = concept_volume_labels_for_subject(*key, parcellation=args.source_parcellation)[4]
        else:
            names = concept_source_labels_for_subject(viewer, *key, parcellation=args.source_parcellation)[2]
        available = list(names) if available is None else [name for name in available if name in set(names)]
    if args.source_parcels:
        missing = set(args.source_parcels) - set(available or [])
        if missing:
            raise ValueError(f"Requested source parcels are absent from an atlas: {sorted(missing)}")
        available = list(dict.fromkeys(args.source_parcels))
    if args.source_space_kind == "volume":
        parcellation = args.source_parcellation if args.source_parcellation in ("HCPMMP1", "HCPMMP1_combined") else "aparc+aseg"
    else:
        parcellation = args.source_parcellation
    selected = filter_parcels(available or [], args.source_parcel_filter or "all",
                              parcellation, args.source_hemisphere)
    if not selected:
        raise ValueError("Source parcel filters matched no ROIs; change the filter, hemisphere, or explicit parcel list.")
    args.source_parcels = selected
    _progress(f"Source parcel selection: {len(selected)} ROIs "
              f"(filter={args.source_parcel_filter or 'all'}, hemisphere={args.source_hemisphere}): "
              + ", ".join(selected))


def build_decoder_options(args: argparse.Namespace) -> dict[str, Any]:
    shared_epochs = int(getattr(args, "epochs", None) or getattr(args, "transformer_epochs", None) or 30)
    options: dict[str, Any] = {
        "decoder_type": args.decoder,
        "pairwise_supertrial_size": args.pairwise_supertrial_size,
        "pairwise_repeats": args.pairwise_repeats,
        "pairwise_random_state": args.pairwise_random_state,
        "pairwise_svm_c": args.pairwise_svm_c,
        "group_mode": args.group_mode,
        "window_start": args.window_start,
        "window_end": args.window_end,
        "bin_size_ms": args.bin_size_ms,
        "feature_binning_enabled": not bool(args.no_feature_binning),
        "feature_mode": args.feature_mode,
        "feature_space": args.feature_space,
        "mne_resample_enabled": bool(args.downsample),
        "resample_sfreq": args.downsample_sfreq,
        "channel_mode": args.channel_mode,
        "max_channels": args.max_channels,
        "bandpass_low_hz": args.bandpass_low_hz,
        "bandpass_high_hz": args.bandpass_high_hz,
        "baseline_correct": not bool(args.no_baseline),
        "mne_epoch_reject_enabled": not bool(args.no_mne_epoch_reject),
        "reject_mag_ft": args.reject_mag_ft,
        "reject_grad_ft_cm": args.reject_grad_ft_cm,
        "exclude_trial_outliers": not bool(args.include_trial_outliers),
        "trial_outlier_z": args.trial_outlier_z,
        "min_trials_for_outlier_qc": args.min_trials_for_outlier_qc,
        "min_trials_per_class": args.min_trials_per_class,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "mlp_validation_fraction": args.validation_fraction,
        "mlp_dropout": args.mlp_dropout,
        "embedding_ridge_alpha": getattr(args, "embedding_ridge_alpha", 1.0),
        "embedding_contrastive_temperature": args.contrastive_temperature,
        "embedding_contrastive_weight_decay": args.contrastive_weight_decay,
        "embedding_contrastive_learning_rate": args.contrastive_learning_rate,
        "embedding_input_pca_components": args.embedding_input_pca_components,
        "embedding_target_svd_components": args.embedding_target_svd_components,
        "embedding_early_stopping_patience": args.embedding_early_stopping_patience,
        "embedding_early_stopping_min_delta": args.embedding_early_stopping_min_delta,
        "embedding_dynamic_pseudo_epochs": not bool(args.no_dynamic_pseudo_epochs),
        "balance_training_classes": bool(args.balance_classes),
        "embedding_validation_metric": args.embedding_validation_metric,
        "split_mode": args.split_mode,
        "zero_shot": bool(getattr(args, "zero_shot", False) or str(args.decoder).startswith("zero_shot")),
        "zero_shot_unseen_concepts": getattr(args, "zero_shot_unseen_concepts", None),
        "zero_shot_unseen_fraction": getattr(args, "zero_shot_unseen_fraction", 0.2),
        "zero_shot_random_state": getattr(args, "zero_shot_random_state", 42),
        "n_permutations": args.n_permutations,
        "avg_trials": getattr(args, "avg_trials", 5),
        "trials_per_average": getattr(args, "avg_trials", 5),
        "pseudo_epoch_trials": getattr(args, "avg_trials", 5),
        "transformer_trials_per_average": getattr(args, "avg_trials", 5),
        "task_stratified_pseudo_epochs": bool(getattr(args, "task_stratified_pseudo_epochs", True)),
        "epochs": shared_epochs,
        "mlp_epochs": shared_epochs,
        "transformer_epochs": shared_epochs,
        "batch_size": getattr(args, "batch_size", 16),
        "transformer_batch_size": getattr(args, "batch_size", 16),
        "transformer_model_type": args.transformer_model_type,
        "progress_callback": _progress,
    }
    if args.feature_space == "source":
        options.update(
            source_space_kind=args.source_space_kind,
            source_parcellation=args.source_parcellation,
            source_common_space=args.source_common_space,
            selected_parcellations=args.source_parcels,
            method=args.source_method,
            lambda2=1.0 / args.source_snr ** 2,
            max_rois=args.max_rois,
            source_batch_size=args.source_batch_size,
            source_workers=args.source_workers,
            source_worker_threads=args.source_worker_threads,
            source_power_bands=args.source_power_bands,
            source_power_baseline=args.source_power_baseline,
            source_feature_cache_dir=args.source_feature_cache_dir,
            source_feature_cache=args.source_feature_cache,
            rebuild_source_features=args.rebuild_source_features,
            parcel_baseline_start=args.parcel_baseline_start,
            parcel_baseline_end=args.parcel_baseline_end,
        )
        if args.source_power_padding_s is not None:
            options["source_power_padding_s"] = args.source_power_padding_s
    if args.epoch_cache_dir:
        options["signal_epoch_cache_dir"] = str(Path(args.epoch_cache_dir).expanduser().resolve())
    if args.embedding_dict:
        options["embedding_dict_path"] = str(Path(args.embedding_dict).expanduser().resolve())
    if args.seeg_decoder_src:
        options["seeg_decoder_src"] = str(Path(args.seeg_decoder_src).expanduser().resolve())
    return options


def concept_counts_for_input(context: HeadlessConceptDecodingContext, args: argparse.Namespace) -> pd.DataFrame:
    df = context._annotate_trial_df_with_stimulus_groups(
        context._fif_trial_response_table,
        context._fif_filename,
    )
    labels = [
        concept_label_from_trial_row(context, row, group_mode=args.group_mode)
        for _, row in df.iterrows()
    ]
    label_df = pd.DataFrame({"concept": [str(label).strip() for label in labels]})
    valid = label_df["concept"].astype(bool) & (label_df["concept"].str.lower() != "unknown")
    counts = (
        label_df.loc[valid]
        .groupby("concept", sort=False)
        .size()
        .reset_index(name="n_trials")
        .sort_values("n_trials", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    counts["retained_by_min_trials"] = counts["n_trials"] >= int(args.min_trials_per_class)
    return counts


def preflight_inputs(args: argparse.Namespace, trial_df: pd.DataFrame, context: HeadlessConceptDecodingContext) -> None:
    paths = [path for path in pd.Index(trial_df["source_fif_path"].astype(str)).drop_duplicates().tolist() if path]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing and not args.allow_missing_fif:
        sample = "\n  ".join(missing[:8])
        more = f"\n  ... {len(missing) - 8} more" if len(missing) > 8 else ""
        raise FileNotFoundError(
            "Some FIF paths do not exist after applying --path-map. "
            "Fix the trial table or pass --allow-missing-fif to let the decoder skip them.\n"
            f"  {sample}{more}"
        )

    if args.feature_space == "source":
        context.meg_3d.validate(build_decoder_options(args))
        _progress(f"Validated saved inverses and anatomy for {len(context.meg_3d.runs)} source runs.")

    counts = concept_counts_for_input(context, args)
    retained = counts.loc[counts["retained_by_min_trials"].astype(bool)]
    min_required = 3 if (bool(getattr(args, "zero_shot", False)) or str(args.decoder).startswith("zero_shot")) else 2
    if retained.shape[0] < min_required:
        raise ValueError(
            f"Need at least {min_required} concepts/categories with >= {int(args.min_trials_per_class)} trials for {'zero-shot ' if min_required == 3 else ''}decoding. "
            f"Found {retained.shape[0]}."
        )


def write_options(out_dir: Path, args: argparse.Namespace, options: dict[str, Any]) -> None:
    serializable_options = {key: value for key, value in options.items() if key != "progress_callback"}
    payload = {
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "trial_table": str(Path(args.trial_table).expanduser().resolve()) if args.trial_table else None,
        "fif": args.fif,
        "config": args.config,
        "ica": args.ica,
        "prepared_runs": getattr(args, "prepared_runs", []),
        "arguments": vars(args),
        "options": serializable_options,
    }
    _safe_write_json(out_dir / "run_options.json", payload)


def write_outputs(result: dict[str, Any], out_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}

    def record(name: str, path: Path, wrote: bool) -> None:
        if wrote:
            outputs[name] = str(path)

    for name in ("timecourse", "pairwise_accuracy", "pairwise_trial_index"):
        path = out_dir / f"{name}.csv"
        record(f"{name}_csv", path, _write_dataframe(result.get(f"{name}_df"), path))
    if "pairwise_accuracy_matrix" in result:
        path = out_dir / "pairwise_accuracy.npz"
        np.savez_compressed(path, accuracy=result["pairwise_accuracy_matrix"],
                            times_s=result["pairwise_times_s"], class_names=result["class_names"])
        outputs["pairwise_accuracy_npz"] = str(path)

    record("summary_csv", out_dir / "summary.csv", _write_dataframe(result.get("summary_df"), out_dir / "summary.csv"))
    record("metrics_csv", out_dir / "metrics.csv", _write_dataframe(result.get("metrics_df"), out_dir / "metrics.csv"))
    record(
        "predictions_csv",
        out_dir / "predictions.csv",
        _write_dataframe(result.get("predictions_df"), out_dir / "predictions.csv"),
    )
    record(
        "feature_importance_csv",
        out_dir / "feature_importance.csv",
        _write_dataframe(result.get("feature_df"), out_dir / "feature_importance.csv"),
    )
    record(
        "model_architecture_csv",
        out_dir / "model_architecture.csv",
        _write_dataframe(result.get("model_architecture_df"), out_dir / "model_architecture.csv"),
    )
    record("split_csv", out_dir / "split.csv", _write_dataframe(result.get("split_df"), out_dir / "split.csv"))
    record("qc_summary_csv", out_dir / "qc_summary.csv", _write_dataframe(result.get("qc_summary_df"), out_dir / "qc_summary.csv"))
    record("trial_qc_csv", out_dir / "trial_qc.csv", _write_dataframe(result.get("trial_qc_df"), out_dir / "trial_qc.csv"))
    record(
        "per_class_accuracy_csv",
        out_dir / "per_class_accuracy.csv",
        _write_dataframe(result.get("per_class_accuracy_df"), out_dir / "per_class_accuracy.csv"),
    )
    record(
        "permutation_baseline_csv",
        out_dir / "permutation_baseline.csv",
        _write_dataframe(result.get("permutation_df"), out_dir / "permutation_baseline.csv"),
    )
    record(
        "template_top_features_csv",
        out_dir / "template_top_features.csv",
        _write_dataframe(result.get("template_df"), out_dir / "template_top_features.csv"),
    )
    record(
        "template_feature_meta_csv",
        out_dir / "template_feature_meta.csv",
        _write_dataframe(result.get("template_feature_meta_df"), out_dir / "template_feature_meta.csv"),
    )
    record(
        "transformer_train_loss_csv",
        out_dir / "transformer_train_loss.csv",
        _write_dataframe(result.get("transformer_train_loss_df"), out_dir / "transformer_train_loss.csv"),
    )

    class_names = [str(value) for value in result.get("class_names", [])]
    if class_names:
        class_path = out_dir / "class_names.txt"
        class_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
        outputs["class_names_txt"] = str(class_path)

    cm = np.asarray(result.get("confusion_matrix", np.empty((0, 0))), dtype=float)
    if cm.size > 0 and class_names:
        cm_df = pd.DataFrame(cm.astype(int), index=class_names, columns=class_names)
        cm_df.index.name = "true_concept"
        cm_path = out_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        outputs["confusion_matrix_csv"] = str(cm_path)

    template_matrix = result.get("template_matrix")
    if isinstance(template_matrix, np.ndarray) and template_matrix.size > 0:
        path = out_dir / "template_matrix.npy"
        np.save(path, template_matrix)
        outputs["template_matrix_npy"] = str(path)

    summary_skip_types = (pd.DataFrame, np.ndarray)
    skip_keys = {
        "summary_df",
        "metrics_df",
        "predictions_df",
        "feature_df",
        "model_architecture_df",
        "split_df",
        "qc_summary_df",
        "trial_qc_df",
        "per_class_accuracy_df",
        "permutation_df",
        "template_df",
        "template_feature_meta_df",
        "transformer_train_loss_df",
        "template_matrix",
        "confusion_matrix",
    }
    summary = {
        key: value
        for key, value in result.items()
        if key not in skip_keys and not isinstance(value, summary_skip_types)
    }
    summary["output_files"] = outputs
    summary_path = out_dir / "summary.json"
    _safe_write_json(summary_path, summary)
    outputs["summary_json"] = str(summary_path)
    return outputs


def write_plots(result: dict[str, Any], out_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _progress(f"plotting skipped: matplotlib unavailable ({exc})")
        return {}

    outputs: dict[str, str] = {}
    if result.get("decoder_type") == "pairwise_svm":
        from core.concept_decoding_pairwise import plot_pairwise_timecourse
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_pairwise_timecourse(ax, result)
        fig.tight_layout()
        path = out_dir / "pairwise_timecourse.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs["pairwise_timecourse_png"] = str(path)

    class_names = [str(value) for value in result.get("class_names", [])]
    cm = np.asarray(result.get("confusion_matrix", np.empty((0, 0))), dtype=float)
    if cm.size > 0 and class_names:
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(class_names)), max(6, 0.5 * len(class_names))))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Held-out confusion matrix")
        ax.set_xlabel("Predicted concept")
        ax.set_ylabel("True concept")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=55, ha="right")
        ax.set_yticklabels(class_names)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                value = int(cm[row, col])
                ax.text(col, row, str(value), ha="center", va="center", color="white" if value > np.nanmax(cm) * 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = out_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs["confusion_matrix_png"] = str(path)

    per_class = result.get("per_class_accuracy_df")
    if isinstance(per_class, pd.DataFrame) and not per_class.empty and {"concept", "test_accuracy"}.issubset(per_class.columns):
        metric = str(result.get("primary_metric", result.get("metric", "accuracy")) or "accuracy").strip()
        is_mpr = metric.upper() == "MPR"
        metric_label = "MPR" if is_mpr else "accuracy"
        plot_df = per_class.sort_values("test_accuracy", ascending=True, kind="stable")
        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(plot_df))))
        ax.barh(plot_df["concept"].astype(str), pd.to_numeric(plot_df["test_accuracy"], errors="coerce"))
        chance = float(result.get("chance_accuracy", np.nan))
        if np.isfinite(chance):
            ax.axvline(chance, color="red", linestyle="--", linewidth=1.5, label=f"chance={chance:.3f}")
            ax.legend(loc="lower right")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(f"Held-out {metric_label}")
        ax.set_title(f"Per-concept held-out {metric_label}")
        fig.tight_layout()
        path = out_dir / "per_class_accuracy.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs["per_class_accuracy_png"] = str(path)

    train_loss = result.get("transformer_train_loss_df")
    if isinstance(train_loss, pd.DataFrame) and not train_loss.empty and {"epoch", "train_loss"}.issubset(train_loss.columns):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pd.to_numeric(train_loss["epoch"], errors="coerce"), pd.to_numeric(train_loss["train_loss"], errors="coerce"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine loss")
        ax.set_title("Transformer training loss")
        fig.tight_layout()
        path = out_dir / "transformer_train_loss.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs["transformer_train_loss_png"] = str(path)

    permutation = result.get("permutation_df")
    if isinstance(permutation, pd.DataFrame) and not permutation.empty and "balanced_accuracy" in permutation.columns:
        metric = str(result.get("primary_metric", result.get("metric", "accuracy")) or "accuracy").strip()
        is_mpr = metric.upper() == "MPR"
        metric_label = "MPR" if is_mpr else "accuracy"
        fig, ax = plt.subplots(figsize=(7, 4))
        values = pd.to_numeric(permutation["balanced_accuracy"], errors="coerce").dropna()
        if not values.empty:
            ax.hist(values, bins=min(30, max(8, int(np.sqrt(len(values))))), color="#5176d7", alpha=0.8)
            observed = float(result.get("balanced_accuracy", np.nan))
            if np.isfinite(observed):
                ax.axvline(observed, color="black", linewidth=2.0, label=f"observed={observed:.3f}")
                ax.legend()
            ax.set_xlabel(f"Balanced {metric_label} under permutation")
            ax.set_ylabel("Count")
            ax.set_title("Permutation baseline")
            fig.tight_layout()
            path = out_dir / "permutation_baseline.png"
            fig.savefig(path, dpi=180)
            outputs["permutation_baseline_png"] = str(path)
        plt.close(fig)
    return outputs


def prepare_output_dir(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in KNOWN_OUTPUT_FILES if (out_dir / name).exists()]
    if existing and not args.force:
        shown = ", ".join(existing[:8])
        raise FileExistsError(
            f"Output directory already contains decoder outputs ({shown}). Pass --force or choose a new --output-dir."
        )
    return out_dir


def configure_runtime(args: argparse.Namespace) -> None:
    if args.seeg_decoder_src:
        os.environ["CIDECOG_SEEG_DECODER_SRC"] = str(Path(args.seeg_decoder_src).expanduser().resolve())
    if args.torch_num_threads and int(args.torch_num_threads) > 0:
        n_threads = str(int(args.torch_num_threads))
        os.environ.setdefault("OMP_NUM_THREADS", n_threads)
        os.environ.setdefault("MKL_NUM_THREADS", n_threads)
        try:
            import torch

            torch.set_num_threads(int(n_threads))
        except Exception:
            pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run signal-space MEG concept decoding without the GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Config: configs/decoding.toml (TOML comments describe decoders). "
               "CLI flags override config. FIF mode automatically creates/reuses "
               "<analysis.fif>.trial_table.csv beside each analysis FIF. "
               "Example: python scripts/run_concept_decoding_cli.py --fif /data/run.fif --decoder logistic",

    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Enable zero-shot concept decoding on held-out unseen concepts.",
    )
    parser.add_argument(
        "--zero-shot-unseen-concepts",
        help="Comma-separated list of concept names to hold out as unseen test concepts.",
    )
    parser.add_argument(
        "--zero-shot-unseen-fraction",
        type=float,
        default=0.2,
        help="Fraction of concepts to hold out as unseen test concepts (default: 0.2).",
    )
    parser.add_argument(
        "--zero-shot-random-state",
        type=int,
        default=42,
        help="Random seed for zero-shot concept holdout split (default: 42).",
    )
    parser.add_argument("--pairwise-supertrial-size", type=int, default=6)
    parser.add_argument("--pairwise-repeats", type=int, default=100)
    parser.add_argument("--pairwise-random-state", type=int, default=42)
    parser.add_argument("--pairwise-svm-c", type=float, default=1.0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Commented TOML configuration file; paths inside it are relative to that file.")
    parser.add_argument("--no-config", action="store_true", help="Ignore the default configuration.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--fif", nargs="+", help="One or more FIF paths. Trial tables are created/reused automatically.")
    inputs.add_argument("--fif-folder", "--fif_folder", dest="fif_folder", help="Process all raw FIF recordings directly in this folder, without prompting (nonrecursive).")
    inputs.add_argument("--resume", help="Resume an exhaustive parcel-screening batch directory using its saved settings/features.")
    parser.add_argument("--subset-search", choices=["exhaustive", "backward"], help="Start source parcel screening over --source-parcels or a --source-parcel-filter (no parcel-count cap).")
    parser.add_argument("--subset-workers", type=int, default=1, help="Concurrent CPU subset fits for exhaustive screening or --resume; default 1 retains sequential behavior.")
    parser.add_argument("--subset-worker-threads", type=int, default=1, help="CPU threads per parallel subset worker (BLAS and Torch).")
    parser.add_argument("--resume-search-root", action="append", default=[], help="Search this relocated data directory for missing checkpoint files; repeatable.")
    inputs.add_argument("--trial-table", help="Optional legacy input: an already exported per-trial table.")
    parser.add_argument("--all-runs", action=argparse.BooleanOptionalAction, default=None, help="Include other raw FIF recordings in each input folder without prompting. --no-all-runs uses only explicit inputs. Default: prompt for one input in an interactive terminal; no prompt in batch jobs.")
    parser.add_argument("--ica", choices=("auto", "off"), default="auto", help="Auto creates/reuses ICA derivatives for tSSS/SSS folders, matching the GUI. Off uses the supplied FIF as-is.")
    parser.add_argument("--rebuild-trial-table", action="store_true", help="Rebuild automatic trial-table caches even when current.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare FIFs and save trial tables without running decoding.")
    parser.add_argument("--output-dir", help="Results directory; default is a new timestamped folder beside the input.")
    parser.add_argument(
        "--decoder",
        choices=DECODER_CHOICES,
        default="transformer",
        help="Decoder model to train/test.",
    )
    parser.add_argument(
        "--group-mode",
        choices=("category", "canonical", "individual", "semantic_decision"),
        default="category",
        help=("Labels to decode: stimulus_group/category, canonical stimulus name, individual stimulus file, "
              "or line_2 semantic decision task from block-instructions.csv matched by block_id."),
    )
    parser.add_argument(
        "--fif-path-column",
        default="source_fif_path",
        help="Trial table column containing FIF paths if not already named source_fif_path.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Remap stored FIF paths for a server mount. Can be repeated. Format: OLD=NEW.",
    )
    parser.add_argument("--stimulus-group-map", help="Optional table mapping stimulus/canonical_stimulus to stimulus_group/category.")
    parser.add_argument(
        "--no-auto-stimulus-group-map",
        action="store_true",
        help="Disable automatic lookup of ../stimulus_with_groups.csv next to each FIF run.",
    )
    parser.add_argument(
        "--override-stimulus-groups",
        action="store_true",
        help="Let stimulus group maps replace existing stimulus_group values instead of filling blanks only.",
    )
    parser.add_argument("--window-start", type=float, default=0.15, help="Epoch feature window start in seconds.")
    parser.add_argument("--window-end", type=float, default=0.45, help="Epoch feature window end in seconds.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Validation fraction taken from the training split for trainable neural and contrastive decoders.",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.35,
        help="Hidden-layer dropout probability for the feed-forward neural net.",
    )
    parser.add_argument(
        "--bin-size-ms",
        type=float,
        default=0.0,
        help="Legacy feature bin size in milliseconds. Ignored when --no-feature-binning is set.",
    )
    parser.add_argument(
        "--no-feature-binning",
        action="store_true",
        help="Use raw resampled epoch samples instead of time-bin or whole-window summary features.",
    )
    parser.add_argument("--feature-binning", dest="no_feature_binning", action="store_false", default=argparse.SUPPRESS, help="Override no_feature_binning=true in config.")
    parser.add_argument(
        "--feature-mode",
        choices=("mean_rms", "mean", "rms", "raw", "bandpower"),
        default="raw",
        help="Feature representation. bandpower requires --feature-space source; --no-feature-binning then uses whole-window power.",
    )
    source = parser.add_argument_group("Source-space decoding (saved inverses and FreeSurfer anatomy)")
    source.add_argument("--feature-space", choices=("signal", "source"), default="signal")
    source.add_argument("--source-space-kind", choices=("surface", "volume"), default="surface")
    inverse_inputs = source.add_mutually_exclusive_group()
    inverse_inputs.add_argument("--source-inverse", help="Existing inverse FIF for one analysis FIF.")
    inverse_inputs.add_argument("--source-manifest", help="JSON runs list mapping analysis FIFs to inverse, subject, and subjects_dir.")
    source.add_argument("--source-subject", help="FreeSurfer subject name; manifest entries can override it.")
    source.add_argument("--subjects-dir", help="FreeSurfer SUBJECTS_DIR; manifest entries can override it.")
    source.add_argument("--source-parcellation", choices=("aparc", "HCPMMP1", "HCPMMP1_combined"), default="aparc", help="Surface atlas; volume always uses aparc+aseg.")
    source.add_argument("--source-common-space", choices=("auto", "native", "fsaverage"), default="auto")
    source.add_argument("--source-parcel-filter", choices=["all", "visual", "semantic", "visual_semantic"],
                        help="GUI parcel preset; filters explicit parcels when supplied, otherwise atlas labels shared across runs.")
    source.add_argument("--source-hemisphere", choices=["both", "left", "right"], default="both",
                        help="Restrict source parcels to a hemisphere. Both includes midline labels.")
    source.add_argument("--source-parcels", nargs="+", default=[], help="Atlas ROI names; aparc uses ctx-lh-/ctx-rh- names. Default: all atlas regions.")
    source.add_argument("--source-method", choices=("MNE", "dSPM", "sLORETA", "eLORETA"), default="MNE")
    source.add_argument("--source-snr", type=float, default=3.0, help="Inverse regularization SNR (lambda2=1/SNR^2).")
    source.add_argument("--max-rois", type=int, default=0, help="Maximum selected source ROIs; 0 means no limit.")
    source.add_argument("--source-batch-size", type=int, default=4)
    source.add_argument("--source-power-bands", nargs="+", default=["all"], help="all, or space-separated delta theta alpha beta low_gamma high_gamma.")
    source.add_argument("--source-power-baseline", action=argparse.BooleanOptionalAction, default=True, help="Baseline dB for source band power; --no-source-power-baseline selects log10 power.")
    source.add_argument("--source-power-padding-s", type=float, help="Override padding, at least max(0.5, 3/lowest band edge) seconds.")
    source.add_argument("--parcel-baseline-start", type=float, default=-0.2)
    source.add_argument("--parcel-baseline-end", type=float, default=0.0)
    parser.add_argument("--downsample", action=argparse.BooleanOptionalAction, default=False, help="Use MNE Epochs.resample before feature extraction.")
    parser.add_argument("--downsample-sfreq", type=float, default=400.0, help="Target sampling rate in Hz when --downsample is set.")
    parser.add_argument("--channel-mode", choices=("all", "mag", "grad"), default="all", help="MEG channel subset.")
    parser.add_argument("--max-channels", type=int, default=0, help="Limit common channels after bad-channel exclusion. 0 uses all.")
    parser.add_argument("--bandpass-low-hz", type=float, default=0.1, help="Low cutoff before feature extraction. Use 0 to disable.")
    parser.add_argument("--bandpass-high-hz", type=float, default=150.0, help="High cutoff before feature extraction. Use 0 to disable.")
    parser.add_argument("--no-baseline", action="store_true", help="Disable baseline correction using -0.2 to 0.0 s.")
    parser.add_argument("--no-mne-epoch-reject", action="store_true", help="Disable MNE peak-to-peak epoch rejection.")
    parser.add_argument("--reject-mag-ft", type=float, default=4000.0, help="MNE magnetometer epoch rejection threshold in fT. Use 0 to disable.")
    parser.add_argument("--reject-grad-ft-cm", type=float, default=4000.0, help="MNE gradiometer epoch rejection threshold in fT/cm. Use 0 to disable.")
    parser.add_argument(
        "--include-trial-outliers",
        action="store_true",
        help="Keep trials flagged by robust signal QC instead of dropping them.",
    )
    parser.add_argument("--trial-outlier-z", type=float, default=6.0, help="Robust z threshold for trial outlier QC.")
    parser.add_argument("--min-trials-for-outlier-qc", type=int, default=8, help="Minimum trials per run for outlier QC.")
    parser.add_argument("--min-trials-per-class", type=int, default=2, help="Minimum usable trials per decoded label.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Held-out fraction per concept/category.")
    parser.add_argument(
        "--split-mode",
        choices=("random", "zero_shot", "leave_one_run", "leave_one_session"),
        default="random",
        help="Held-out split mode: 'random' (per-concept random), 'zero_shot' (completely unseen concepts), or grouped ('leave_one_run', 'leave_one_session').",
    )
    parser.add_argument("--n-permutations", type=int, default=99, help="Permutation baseline runs for chance testing.")
    parser.add_argument(
        "--embedding-dict",
        help="JSON/HDF5/NPZ/CSV embedding dictionary required for semantic embedding decoders.",
    )
    parser.add_argument(
        "--embedding-ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization alpha for linear ridge embedding decoder.",
    )
    parser.add_argument(
        "--contrastive-temperature",
        type=float,
        default=0.07,
        help="Cosine-logit temperature for the class-aware contrastive linear decoder.",
    )
    parser.add_argument(
        "--contrastive-weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay for the class-aware contrastive linear decoder.",
    )
    parser.add_argument(
        "--contrastive-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the class-aware contrastive linear decoder.",
    )
    parser.add_argument(
        "--embedding-input-pca-components",
        type=int,
        default=256,
        help="Train-only PCA dimensions for flat semantic decoders; 0 disables PCA.",
    )
    parser.add_argument(
        "--embedding-target-svd-components",
        type=int,
        default=40,
        help="Train-only centered semantic SVD dimensions; 0 retains the full embedding.",
    )
    parser.add_argument(
        "--embedding-early-stopping-patience",
        type=int,
        default=4,
        help="Validation epochs without improvement before stopping; 0 disables early stopping.",
    )
    parser.add_argument(
        "--embedding-early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement used for semantic early stopping.",
    )
    parser.add_argument(
        "--no-dynamic-pseudo-epochs",
        action="store_true",
        help="Reuse fixed training trial averages instead of re-averaging each semantic training epoch.",
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Equalize semantic-decoder training pseudo-epochs across labels. Extra trials from larger "
              "classes rotate across dynamic training epochs; validation and test sets remain intact."),
    )
    parser.add_argument(
        "--embedding-validation-metric",
        choices=("top1", "balanced_top1", "mpr", "balanced_mpr"),
        default="top1",
        help="Metric used to select the best contrastive/MLP validation epoch.",
    )
    parser.add_argument(
        "--seeg-decoder-src",
        help="Path to CIDECoG_enc-main/src. Also accepted via CIDECOG_SEEG_DECODER_SRC.",
    )
    parser.add_argument(
        "--avg-trials",
        "--transformer-avg-trials",
        dest="avg_trials",
        type=int,
        default=5,
        help="Number of source trials per averaged pseudo-epoch. If 5, strictly balances the 5 semantic decisions (1 each). If 6, proportionally balances decisions (2 Animal + 1 each of Food, Vehicle, Instrument, Body Part; 100%% data retention).",
    )
    parser.add_argument(
        "--task-stratified-pseudo-epochs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Task-stratified pseudo-epoching for N=5/6 (enabled by default; set false or --no-task-stratified-pseudo-epochs to use standard random chunking).",
    )
    parser.add_argument(
        "--epochs",
        "--transformer-epochs",
        dest="epochs",
        type=int,
        default=30,
        help="Training epochs for feed-forward, contrastive linear, semantic MLP, and neural decoders.",
    )
    parser.add_argument(
        "--batch-size",
        "--transformer-batch-size",
        dest="batch_size",
        type=int,
        default=16,
        help="Mini-batch size for iterative neural/embedding decoders.",
    )
    parser.add_argument("--transformer-model-type", default="axial_transformer", help="Transformer model type.")
    parser.add_argument(
        "--epoch-cache-dir",
        help="Optional persistent cache directory for bandpass-filtered epochs across repeated CLI runs.",
    )
    parser.add_argument("--source-workers", type=int, default=1, help="Parallel source extraction processes; capped at the number of runs. Each process loads its own inverse and recording.")
    parser.add_argument("--source-worker-threads", type=int, default=1, help="Numerical threads per source extraction process.")
    parser.add_argument("--source-feature-cache-dir", help="Override the ROI cache directory; default DATA/source_feature_cache/FEATURE_MODE (e.g. CIDMEG005/source_feature_cache/bandpower).")
    parser.add_argument("--source-feature-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable persistent ROI cache reads and writes (enabled by default beside the data).")
    parser.add_argument("--rebuild-source-features", action="store_true", help="Re-extract ROI features and replace the matching cache entry.")
    parser.add_argument("--torch-num-threads", type=int, default=0, help="Limit Torch/OpenMP CPU threads. 0 leaves defaults.")
    parser.add_argument(
        "--allow-missing-fif",
        action="store_true",
        help="Do not fail preflight if some FIF paths are missing; backend will skip unavailable runs.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot creation.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare FIFs if needed, validate inputs and write counts, but do not train. May run ICA.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files in --output-dir.")
    # Parse configuration as ordinary arguments so argparse validates choices,
    # types and unknown keys exactly as it does for command-line arguments.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG))
    bootstrap.add_argument("--no-config", action="store_true")
    early, _ = bootstrap.parse_known_args(argv)
    config_tokens = []
    path_keys = {"resume", "resume_search_root", "fif", "fif_folder", "trial_table", "output_dir", "stimulus_group_map",
                 "embedding_dict", "seeg_decoder_src", "epoch_cache_dir", "source_inverse", "source_manifest", "subjects_dir", "source_feature_cache_dir"}
    if not early.no_config and not any(flag in argv for flag in ("--help", "-h")):
        config_path = Path(os.path.expandvars(early.config)).expanduser().resolve()
        try:
            settings = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read config {config_path}: {exc}")
        actions = {}
        for action in parser._actions:
            actions.setdefault(action.dest, action)
        explicit_input = any(token.split("=", 1)[0] in {"--resume", "--fif", "--fif-folder", "--fif_folder", "--trial-table"} for token in argv)
        for raw_key, value in settings.items():
            key = raw_key
            if key == "transformer_avg_trials":
                key = "avg_trials"
            elif key == "transformer_batch_size":
                key = "batch_size"
            elif key == "transformer_epochs":
                key = "epochs"
            action = actions.get(key)
            if action is None or key in {"help", "config", "no_config"}:
                parser.error(f"Unknown or unsupported config key: {raw_key}")
            if explicit_input and key in {"resume", "fif", "fif_folder", "trial_table"}:
                continue
            if value == "" or value == []:
                continue
            flag = action.option_strings[0]
            # Any explicit occurrence replaces the config value (also for lists).
            related_flags = [flag for candidate in parser._actions if candidate.dest == key for flag in candidate.option_strings]
            if any(token.split("=", 1)[0] in related_flags for token in argv):
                continue
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    parser.error(f"Config {key} must be true or false")
                config_tokens.append(action.option_strings[0 if value else 1])
                continue
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f"Config {key} must be true or false")
                if value:
                    config_tokens.append(flag)
                continue
            values = value if isinstance(value, list) else [value]
            if key in path_keys:
                values = [str((config_path.parent / Path(os.path.expandvars(str(v))).expanduser()).resolve()) for v in values]
            if isinstance(action, argparse._AppendAction):
                for v in values:
                    config_tokens.extend([flag, str(v)])
            else:
                config_tokens.extend([flag, *map(str, values)])
    args = parser.parse_args(config_tokens + argv)
    for key in path_keys:
        value = getattr(args, key)
        if value:
            resolve = lambda v: str(Path(os.path.expandvars(v)).expanduser().resolve())
            setattr(args, key, [resolve(v) for v in value] if isinstance(value, list) else resolve(value))
    if (args.source_parcel_filter or args.source_hemisphere != "both") and args.feature_space != "source":
        parser.error("Source parcel filters require --feature-space source")
    if args.source_workers < 1 or args.source_worker_threads < 1:
        parser.error("Source workers and threads must be positive")
    if args.subset_workers < 1 or args.subset_worker_threads < 1:
        parser.error("Subset workers and threads must be positive")
    if args.subset_workers > 1 and not (args.resume or args.subset_search == "exhaustive"):
        parser.error("Multiple --subset-workers requires --subset-search exhaustive or --resume")
    if args.resume and args.subset_search:
        parser.error("Use --resume alone to continue a saved search")
    if args.subset_search:
        if args.feature_space != "source" or args.decoder not in {"embedding_ridge", "embedding_contrastive"}:
            parser.error("Subset screening requires source features and embedding_ridge or embedding_contrastive")
        if not args.source_parcels and not args.source_parcel_filter and args.source_hemisphere == "both":
            parser.error("Subset screening requires --source-parcels, --source-parcel-filter, or --source-hemisphere")
        if args.split_mode not in {"random", "zero_shot"}:
            parser.error("Subset screening requires random or zero_shot splitting")
        if args.subset_search == "backward" and args.validation_fraction <= 0:
            parser.error("Backward elimination requires positive --validation-fraction")
        if args.max_rois:
            parser.error("Subset screening requires --max-rois 0 to preserve the selected pool")
    if args.resume:
        if args.prepare_only:
            parser.error("--prepare-only cannot be used with --resume; use --dry-run to validate.")
        return args
    if not args.fif and not args.fif_folder and not args.trial_table:
        parser.error("Set fif or fif_folder in configs/decoding.toml or supply --fif, --fif_folder, or --trial-table.")
    if args.all_runs is not None and not (args.fif or args.fif_folder):
        parser.error("--all-runs/--no-all-runs requires FIF input")
    if args.prepare_only and not (args.fif or args.fif_folder):
        parser.error("--prepare-only requires FIF input")
    if args.feature_mode == "bandpower" and args.feature_space != "source":
        parser.error("--feature-mode bandpower requires --feature-space source")
    if args.feature_space == "source":
        if not args.source_inverse and not args.source_manifest:
            parser.error("Source decoding requires --source-inverse or --source-manifest")
        if args.source_inverse and (not args.source_subject or not args.subjects_dir):
            parser.error("--source-inverse requires --source-subject and --subjects-dir")
        if args.decoder not in {"logistic", "template", "mlp", "embedding_ridge", "embedding_contrastive"}:
            parser.error(f"Decoder {args.decoder!r} is not supported by source-space decoding")
        if not np.isfinite(args.source_snr) or args.source_snr <= 0 or args.source_batch_size < 1:
            parser.error("Source SNR and batch size must be positive")
        if args.source_space_kind == "volume" and args.source_common_space == "fsaverage":
            parser.error("Volume source decoding uses native subject space")
        if args.source_space_kind == "volume" and args.source_parcellation not in {"aparc", "HCPMMP1", "HCPMMP1_combined"}:
            parser.error("Volume source decoding uses aparc+aseg or volumetric HCP-MMP (HCPMMP1 / HCPMMP1_combined)")
        if args.feature_mode == "bandpower" and args.source_power_baseline:
            if not (np.isfinite(args.parcel_baseline_start) and np.isfinite(args.parcel_baseline_end)
                    and args.parcel_baseline_start < args.parcel_baseline_end <= 0):
                parser.error("Source power baseline must be finite with start < end <= 0")
    elif args.source_inverse or args.source_manifest:
        parser.error("Source inverse/manifest requires --feature-space source")
    if not args.output_dir:
        source = Path(args.fif_folder or (args.fif[0] if args.fif else args.trial_table))
        output_base = source if args.fif_folder else source.parent
        args.output_dir = str(output_base / "decoding_results" / f"{source.stem}_{args.decoder}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        _progress("CLI started; configuring runtime...")
        configure_runtime(args)
        if args.resume:
            from core.concept_decoding_subsets import run_semantic_subset_batch
            from core.decoding_resume_paths import ResumePathResolver

            folder = Path(args.resume)
            _progress(f"Opening resume checkpoint: {folder}")
            manifest = json.loads((folder / "manifest.json").read_text())
            resolver = ResumePathResolver(folder, manifest.get("output_dir"),
                                          args.resume_search_root, _parse_path_maps(args.path_map), _progress)
            _, message = run_semantic_subset_batch(SimpleNamespace(), {
                "semantic_subset_resume_dir": str(folder),
                "resume_path_resolver": resolver,
                "resume_dry_run": args.dry_run,
                "subset_workers": args.subset_workers,
                "subset_worker_threads": args.subset_worker_threads,
                "progress_callback": _progress, "log_callback": _progress,
            })
            _progress(message)
            if args.dry_run:
                return 0
            status = json.loads((folder / "manifest.json").read_text()).get("status")
            return 0 if status == "complete" else 1
        # Refuse output overwrites before starting expensive preprocessing.
        out_dir = None if args.prepare_only else prepare_output_dir(args)
        trial_df = load_and_prepare_trial_table(args)
        if args.prepare_only:
            _progress(f"Preparation complete: {len(trial_df)} trials; tables saved beside analysis FIFs.")
            return 0
        _write_dataframe(trial_df, out_dir / "input_trial_table.csv")
        context = build_context(args, trial_df)
        resolve_source_parcels(args, context)
        if args.subset_search:
            from core.concept_decoding_subsets import exhaustive_fit_count, backward_fit_count
            counter = exhaustive_fit_count if args.subset_search == "exhaustive" else backward_fit_count
            _progress(f"WARNING: {counter(args.source_parcels):,} subset models will be run "
                      f"({args.subset_search}). Large searches can take substantial time and disk space.")
        preflight_inputs(args, trial_df, context)
        counts = concept_counts_for_input(context, args)
        _write_dataframe(counts, out_dir / "input_concept_counts.csv")

        options = build_decoder_options(args)
        from core.concept_decoding_run_summary import log_decoding_run_summary
        if not args.subset_search or args.dry_run:
            log_decoding_run_summary(options, _progress)
        if args.subset_search:
            options.update(semantic_subset_batch=True, subset_search_strategy=args.subset_search,
                           semantic_subset_output_dir=str(out_dir), subset_workers=args.subset_workers,
                           subset_worker_threads=args.subset_worker_threads)
        if args.feature_space == "source":
            options["source_runs"] = list(context.meg_3d.runs.values())
        write_options(out_dir, args, options)

        retained = counts.loc[counts["retained_by_min_trials"].astype(bool)]
        _progress(
            f"input ready: rows={len(trial_df)}, runs={trial_df['source_fif_path'].astype(str).nunique()}, "
            f"retained_labels={len(retained)}, decoder={args.decoder}"
        )
        if args.dry_run:
            _progress(f"dry run complete; wrote {out_dir / 'input_concept_counts.csv'}")
            return 0

        semantic_decoders = {
            "embedding_ridge",
            "embedding_contrastive",
            "embedding_mlp",
            "transformer",
        }
        if args.decoder in semantic_decoders and not args.embedding_dict:
            raise ValueError(
                "--embedding-dict is required when using a semantic embedding decoder."
            )
        if (args.split_mode == "zero_shot" or args.zero_shot) and args.decoder not in semantic_decoders:
            raise ValueError(
                f"Zero-shot decoding requires a semantic embedding decoder (got {args.decoder!r}). "
                f"Supported zero-shot decoders: {sorted(semantic_decoders)}"
            )

        start = time.time()
        if args.subset_search:
            from core.concept_decoding_subsets import run_semantic_subset_batch
            result, message = run_semantic_subset_batch(context, options)
            _progress(message)
            # Each fit and the shared comparison/checkpoint are saved by the batch runner.
            if result is None:
                return 1
            folder = Path(result["semantic_subset_output_dir"])
            status = json.loads((folder / "manifest.json").read_text())["status"]
            _progress(f"Checkpoint: {folder}")
            return 0 if status == "complete" else 1
        if args.feature_space == "source":
            from core.concept_decoding_source import compute_source_space_concept_decoding
            result, message = compute_source_space_concept_decoding(context, options)
        else:
            result, message = compute_signal_space_concept_decoding(context, options)
        if result is None:
            print(f"Concept decoding failed: {message}", file=sys.stderr)
            return 1

        outputs = write_outputs(result, out_dir)
        if not args.no_plots:
            plot_outputs = write_plots(result, out_dir)
            if plot_outputs:
                outputs.update(plot_outputs)
                summary_path = out_dir / "summary.json"
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary.setdefault("output_files", {}).update(plot_outputs)
                    _safe_write_json(summary_path, summary)
                except Exception:
                    pass
        elapsed = time.time() - start
        _progress(f"finished in {elapsed / 60.0:.1f} min: {message}")
        _progress(f"summary: {out_dir / 'summary.json'}")
        if "confusion_matrix_png" in outputs:
            _progress(f"confusion matrix plot: {outputs['confusion_matrix_png']}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
