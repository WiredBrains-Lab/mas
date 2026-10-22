"""Persistent, data-local ROI features shared by GUI and CLI source decoding.

Loaded GUI raw state is fingerprinted as well as files to track unsaved edits.
Cache entries are local trusted joblib files, never model fits.
"""
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
import scipy


# Keep extraction settings separate from train/test splits, embedding alignment,
# averaging, PCA, and model hyperparameters, which are applied after this cache.
FEATURE_OPTIONS = """
group_mode window_start window_end bin_size_ms feature_binning_enabled feature_mode
min_trials_per_class source_space_kind
source_parcellation source_common_space source_batch_size max_source_features
method lambda2 source_power_bands source_power_baseline source_power_padding_s
source_parcel_baseline_zscore parcel_baseline_start parcel_baseline_end baseline_correct
parcel_frequency_band bandpass_low_hz bandpass_high_hz mne_resample_enabled
resample_sfreq downsample_enabled downsample_sfreq mne_epoch_reject_enabled
epoch_reject_enabled reject_mag_ft reject_grad_ft_cm mne_reject_mag_ft mne_reject_grad_ft_cm
exclude_trial_outliers trial_outlier_z min_trials_for_outlier_qc
preprocess_continuous_recording source_fif_paths selected_concept_label max_presentations
_use_stimulus_ui_labels _limit_presentations_after_qc _presentation_candidate_limit
_allow_single_class
""".split()


def source_cache_runs(context, options):
    """Resolve explicit CLI mappings or the GUI's run-specific inverse/anatomy."""
    if options.get("_source_feature_cache_runs"):
        return options["_source_feature_cache_runs"]
    from core.concept_decoding_headless_source import HeadlessSourceContext
    from core.concept_decoding_source import (
        _meg_3d_viewer, configure_source_subject_for_fif,
        _find_or_create_source_inverse, restore_meg_sensor_fif,
    )
    viewer = _meg_3d_viewer(context)
    if isinstance(viewer, HeadlessSourceContext):
        return list(viewer.runs.values())
    table = getattr(context, "_fif_trial_response_table", None)
    if viewer is None or not isinstance(table, pd.DataFrame) or "source_fif_path" not in table:
        raise ValueError("source run mappings are unavailable")
    paths = list(dict.fromkeys(str(path) for path in table["source_fif_path"] if pd.notna(path) and str(path)))
    selected = options.get("source_fif_paths", [])
    if len(selected):
        selected = {str(Path(path).resolve()) for path in selected}
        paths = [path for path in paths if str(Path(path).resolve()) in selected]
    runs = []
    restore = getattr(context, "_fif_filename", None)
    try:
        for path in paths:
            subject, error = configure_source_subject_for_fif(context, path, options)
            if subject is None:
                raise ValueError(error)
            inverse, _, error = _find_or_create_source_inverse(
                context, path, options.get("source_space_kind", "surface"))
            if inverse is None:
                raise ValueError(error)
            runs.append(dict(fif=str(Path(path).resolve()), inverse=str(Path(inverse).resolve()),
                             subject=subject[0], subjects_dir=subject[1]))
    finally:
        restore_meg_sensor_fif(context, restore)
    if not runs:
        raise ValueError("no source runs to cache")
    return runs


def source_feature_cache_directory(context, options):
    """Default to DATA/source_feature_cache/FEATURE_MODE; an override is exact."""
    override = options.get("source_feature_cache_dir")
    if override:
        return Path(override).expanduser().resolve()
    runs = source_cache_runs(context, options)
    parents = [Path(row["fif"]).expanduser().resolve().parent for row in runs]
    root = Path(os.path.commonpath(parents))
    # In the subject-local FreeSurfer layout, subjects_dir is CIDMEG005 and
    # its subject is Anatomy. Do not use a shared, unrelated FS subjects folder.
    while root.name.lower() in {"ica", "raw", "fif", "meg", "processed", "preprocessed"}:
        root = root.parent
    anatomy_roots = {Path(row["subjects_dir"]).expanduser().resolve() / row["subject"] for row in runs}
    candidates = set(anatomy_roots)
    candidates.update(path.parent for path in anatomy_roots if path.name.lower() == "anatomy")
    ancestors = [path for path in candidates if all(parent.is_relative_to(path) for parent in parents)]
    if ancestors:
        root = max(ancestors, key=lambda path: len(path.parts))
    if root == Path(root.anchor):
        raise ValueError("inputs have no common data folder; specify source_feature_cache_dir")
    mode = str(options.get("feature_mode", "mean_rms") or "mean_rms").lower()
    if mode not in {"bandpower", "raw", "mean", "rms", "mean_rms"}:
        raise ValueError(f"Unsupported source cache feature mode: {mode}")
    return root / "source_feature_cache" / mode


def _needs_common(options, subject_roots):
    common = str(options.get("source_common_space", "auto"))
    return options.get("source_space_kind", "surface") == "surface" and (
        common in {"fsaverage", "common", "common_space", "common-space", "true", "1", "yes"}
        or options.get("source_parcellation", "aparc") != "aparc"
        or (len(subject_roots) > 1 and common not in {"native", "subject", "subject_space", "subject-space", "false", "0", "no"}))


def all_roi_extraction_options(context, options):
    """Find atlas ROIs with source support in every run, without decoding trials.

    Ones probe the same spatial averaging operator used for actual features.
    Empty labels yield zero; retain only labels supported by all source grids.
    """
    from core.concept_decoding_source import (
        concept_source_labels_for_subject, concept_volume_labels_for_subject,
        fsaverage_source_space,
    )
    runs = source_cache_runs(context, options)
    subjects = {(row["subject"], row["subjects_dir"]) for row in runs}
    common = _needs_common(options, subjects)
    volume = options.get("source_space_kind", "surface") == "volume"
    subject = "fsaverage" if common else runs[0]["subject"]
    subjects_dir = runs[0]["subjects_dir"]
    if volume:
        from core.source_parcellation import volume_atlas_ids_for_aseg
        atlas, _, _, names, _ = concept_volume_labels_for_subject(
            subject, subjects_dir, parcellation=options.get("source_parcellation", "aparc"))
        vol_atlas_ids = volume_atlas_ids_for_aseg(atlas, parcellation=options.get("source_parcellation", "aparc"))
        vol_labels_arg = (atlas, {name: vol_atlas_ids[name] for name in names}) if vol_atlas_ids else (atlas, names)
    else:
        labels, names, _, _ = concept_source_labels_for_subject(
            context, subject, subjects_dir, parcellation=options.get("source_parcellation", "aparc"))
    supported = np.ones(len(names), dtype=bool)
    if common:
        common_src, _ = fsaverage_source_space(subjects_dir)
        source_spaces = [common_src]
    else:
        # Only needed on a cold cache. Release each inverse before loading the next.
        source_spaces = (mne.minimum_norm.read_inverse_operator(row["inverse"], verbose="error")["src"]
                         for row in runs)
    for src in source_spaces:
        vertices = [part["vertno"] for part in src]
        values = np.ones((sum(len(v) for v in vertices), 1))
        cls = mne.VolSourceEstimate if volume else mne.SourceEstimate
        probe = cls(values, vertices, tmin=0, tstep=1, subject=subject)
        if volume:
            means = probe.extract_label_time_course(vol_labels_arg, src=src,
                mode="mean", allow_empty=True, verbose="error")
        else:
            means = mne.extract_label_time_course(probe, labels, src,
                mode="mean", allow_empty=True, verbose="error")
        supported &= np.asarray(means).reshape(len(names), -1).max(axis=1) > 0
    usable = [name for name, keep in zip(names, supported) if keep]
    excluded = [name for name, keep in zip(names, supported) if not keep]
    if not usable:
        raise ValueError("No atlas ROIs have source support in every input run.")
    return dict(options, selected_parcellations=usable, max_rois=0), excluded


def _select_rois(payload, options):
    from core.concept_decoding_subsets import subset_payload
    requested = set(options.get("selected_parcellations") or [])
    available = list(payload["roi_names"])
    missing = requested - set(available)
    if missing:
        return None, ("Requested ROIs have no usable cached source features (absent from the atlas "
                      f"or unsupported by one or more source grids): {sorted(missing)}")
    selected = [name for name in available if not requested or name in requested]
    limit = int(options.get("max_rois", 0))
    if limit > 0:
        selected = selected[:limit]
    return subset_payload(payload, selected), None


def _portable_feature_options(options, runs):
    """Canonicalize GUI/CLI spellings and omitted backend defaults."""
    from core.concept_decoding_signal import CONCEPT_PREPROCESSING_DEFAULTS
    defaults = dict(CONCEPT_PREPROCESSING_DEFAULTS, group_mode="canonical",
        window_start=.15, window_end=.45, bin_size_ms=50., feature_binning_enabled=True,
        feature_mode="mean_rms", min_trials_per_class=2, source_space_kind="surface",
        source_parcellation="aparc", source_common_space="auto", max_source_features=75000,
        method="MNE", lambda2=1. / 9., source_power_bands=["all"], source_power_baseline=True,
        source_power_padding_s=None, source_parcel_baseline_zscore=False,
        parcel_baseline_start=-.2, parcel_baseline_end=0., baseline_correct=True,
        parcel_frequency_band="broadband", min_trials_for_outlier_qc=8,
        preprocess_continuous_recording=True, source_fif_paths=[], selected_concept_label="",
        max_presentations=0, _use_stimulus_ui_labels=False, _limit_presentations_after_qc=False,
        _presentation_candidate_limit=0, _allow_single_class=False)
    aliases = dict(downsample_enabled="mne_resample_enabled", downsample_sfreq="resample_sfreq",
                   epoch_reject_enabled="mne_epoch_reject_enabled", mne_reject_mag_ft="reject_mag_ft",
                   mne_reject_grad_ft_cm="reject_grad_ft_cm")
    values = dict(defaults)
    for alias, canonical in aliases.items():
        if alias in options and canonical not in options:
            values[canonical] = options[alias]
    values.update({key: options[key] for key in FEATURE_OPTIONS if key in options and key not in aliases})
    power = str(values["feature_mode"]).lower() == "bandpower"
    values.setdefault("source_batch_size", 4 if power else (8 if values["feature_mode"] == "raw" else 16))
    if power:
        values["baseline_correct"] = False
        values["source_parcel_baseline_zscore"] = False
    elif values["source_parcel_baseline_zscore"]:
        values["baseline_correct"] = False
    bands = values["source_power_bands"]
    if isinstance(bands, str):
        bands = bands.split(",")
    values["source_power_bands"] = sorted(set(str(band).strip().lower() for band in bands))
    subjects = {(row["subject"], row["subjects_dir"]) for row in runs}
    values["source_common_space"] = "fsaverage" if _needs_common(values, subjects) else "native"
    # JSON numbers are type-sensitive; GUI spin boxes often supply floats for
    # integer-valued defaults. Use the backend's effective numeric types.
    for key, default in defaults.items():
        if isinstance(default, bool):
            values[key] = bool(values[key])
        elif isinstance(default, float):
            values[key] = float(values[key])
        elif isinstance(default, int):
            values[key] = int(values[key])
    values["source_batch_size"] = int(values["source_batch_size"])
    return values


def feature_cache_identity(context, options):
    """Content-based portable identity, with host paths kept only as provenance."""
    from core.concept_decoding_signal import concept_label_from_trial_row
    from core.concept_decoding_cache_portability import (
        file_content_identity, loaded_raw_identity, translate_paths,
    )
    runs = source_cache_runs(context, options)
    bindings, files, portable_runs, subject_roots = {}, [], [], []
    # Exact file roles take precedence over enclosing directories on translation.
    for index, run in enumerate(runs):
        role = f"run/{index}"
        for field in ("fif", "inverse"):
            bindings[f"{role}/{field}"] = str(Path(run[field]).resolve())
        root = Path(run["subjects_dir"]).resolve() / run["subject"]
        if root not in subject_roots:
            subject_roots.append(root)
        portable_runs.append(dict(fif=f"mas-cache://{role}/fif", inverse=f"mas-cache://{role}/inverse",
                                  subject=run["subject"], anatomy=subject_roots.index(root)))
        files.append(dict(role=f"{role}/inverse", **file_content_identity(run["inverse"])))
        with mne.io.read_raw_fif(run["fif"], preload=False, verbose="error") as raw:
            for part_index, path in enumerate(raw.filenames):
                if path:
                    part_role = f"{role}/fif_part/{part_index}"
                    bindings[part_role] = str(Path(path).resolve())
                    files.append(dict(role=part_role, **file_content_identity(path)))
    if _needs_common(options, subject_roots):
        fsaverage = Path(runs[0]["subjects_dir"]).resolve() / "fsaverage"
        if not fsaverage.is_dir():
            raise ValueError("fsaverage must exist in the first run's subjects_dir for common-space caching")
        if fsaverage not in subject_roots:
            subject_roots.append(fsaverage)
    for index, root in enumerate(subject_roots):
        if not root.is_dir():
            raise ValueError(f"Missing anatomy directory: {root}")
        role = f"anatomy/{index}"
        for name in ("mri", "surf", "label", "bem"):
            for path in sorted((root / name).rglob("*")):
                if path.is_file() and not any(part.startswith('.') for part in path.relative_to(root).parts):
                    file_role = f"{role}/{path.relative_to(root).as_posix()}"
                    bindings[file_role] = str(path.resolve())
                    files.append(dict(role=file_role, **file_content_identity(path)))
        bindings[role] = str(root)
        bindings[f"subjects_dir/{index}"] = str(root.parent)
    # Other path-valued trial metadata beneath the data root can also relocate.
    data_root = source_feature_cache_directory(context, dict(options, source_feature_cache_dir=None)).parent.parent
    bindings["data"] = str(data_root)
    table = context._annotate_trial_df_with_stimulus_groups(
        context._fif_trial_response_table, context._fif_filename).copy()
    table["_cache_concept_label"] = [concept_label_from_trial_row(
        context, row, options.get("group_mode", "canonical")) for _, row in table.iterrows()]
    table = translate_paths(table, bindings)
    table_digest = hashlib.sha256(table.to_json(
        orient="split", date_format="iso", double_precision=15, default_handler=str).encode()).hexdigest()

    core = Path(__file__).parent
    implementation = {}
    for name in ("concept_decoding_feature_cache.py", "concept_decoding_cache_portability.py",
                 "concept_decoding_source.py", "concept_decoding_source_parallel.py", "concept_decoding_spectral.py", "concept_decoding_signal.py",
                 "source_parcellation.py", "meg_channels.py", "fif_processing.py",
                 "concept_decoding_headless_source.py"):
        implementation[name] = hashlib.sha256((core / name).read_bytes()).hexdigest()
    implementation["cli"] = hashlib.sha256(
        (core.parent / "scripts/run_concept_decoding_cli.py").read_bytes()).hexdigest()
    raw = getattr(context, "_fif_raw", None)
    current = str(Path(context._fif_filename).resolve())
    raw_digest = loaded_raw_identity(raw, current) if raw is not None and any(
        str(Path(row["fif"]).resolve()) == current for row in runs) else None
    identity = dict(version=4, scope="all_supported_rois",
                    options=translate_paths(_portable_feature_options(options, runs), bindings),
                    runs=portable_runs, trial_table_digest=table_digest, loaded_raw_digest=raw_digest,
                    files=files, implementation=implementation)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value))
    metadata = dict(identity=json.loads(encoded), path_bindings=bindings, current_runs=runs,
                    producer_libraries=dict(python=platform.python_version(), mne=mne.__version__,
                                            numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    return hashlib.sha256(encoded.encode()).hexdigest(), metadata


def _restore_portable_payload(payload, metadata):
    from core.concept_decoding_cache_portability import translate_paths
    result = translate_paths(payload, metadata["path_bindings"], restore=True)
    # Resolve subject directory aliases explicitly when anatomy and data move
    # into separate destination folders rather than retaining the old layout.
    runs = {str(Path(row["fif"]).resolve()): row for row in metadata["current_runs"]}
    for field in ("inverse_paths", "source_subject_by_path"):
        if field in result:
            for path in result[field]:
                if path not in runs:
                    raise ValueError(f"Cached source run is absent from current inputs: {path}")
                row = runs[path]
                result[field][path] = (row["inverse"] if field == "inverse_paths" else
                                      dict(subject=row["subject"], subjects_dir=row["subjects_dir"]))
    if "subjects_dir" in result:
        matching = [row for row in runs.values() if row["subject"] == result.get("subject")]
        result["subjects_dir"] = (matching or list(runs.values()))[0]["subjects_dir"]
    return result


def cached_source_features(context, options, extract):
    """Load or build one entry; coordinate concurrent jobs and publish atomically."""
    from core.concept_decoding_signal import emit_concept_progress
    if getattr(context, "_fif_raw", None) is not None:
        # The recording/epoch disk cache identifies files, not arbitrary unsaved
        # GUI signal edits. On a source-cache miss, preprocess the live raw object.
        options = dict(options, enable_signal_epoch_cache=False)
    def report(message):
        emit_concept_progress(options, message)

    try:
        import fcntl
        options = dict(options, _source_feature_cache_runs=source_cache_runs(context, options))
        report("Verifying ROI cache input contents (FIF, inverse, and anatomy files)...")
        key, metadata = feature_cache_identity(context, options)
        directory = source_feature_cache_directory(context, options)
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / f"{key}.lock").open("a")
    except Exception as exc:
        report(f"ROI feature cache unavailable; extracting normally: {exc}")
        return extract(context, options)

    with lock:
        report(f"Checking ROI feature cache (waiting if another job is extracting): {key[:12]}")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as exc:
            report(f"ROI cache locking unavailable; extracting normally: {exc}")
            return extract(context, options)
        path = directory / f"{key}.joblib"
        # Inputs may have changed while waiting for another job's extraction.
        if feature_cache_identity(context, options)[0] != key:
            report("ROI inputs changed while waiting; extracting without caching this run.")
            return extract(context, options)
        if path.is_file() and not options.get("rebuild_source_features", False):
            try:
                saved = joblib.load(path)
                payload = saved["payload"]
                if saved["key"] != key or saved["metadata"].get("identity") != metadata["identity"] or not isinstance(payload, dict):
                    raise ValueError("cache metadata mismatch")
                if np.asarray(payload["X"]).shape != (len(payload["trial_df"]), len(payload["feature_meta_df"])):
                    raise ValueError("cache feature shape mismatch")
                payload = _restore_portable_payload(payload, metadata)
                payload["source_feature_cache_producer_libraries"] = saved["metadata"]["producer_libraries"]
                if saved["metadata"]["producer_libraries"] != metadata["producer_libraries"]:
                    report("Reusing computed ROI arrays from a different software environment; producer versions are retained in cache metadata.")
                report(f"Reusing saved ROI features: {path}")
                return _select_rois(dict(payload, source_feature_cache_hit=True, source_feature_cache_path=str(path)), options)
            except Exception as exc:
                report(f"ROI cache entry unreadable; rebuilding: {exc}")
        report("Preparing all supported atlas ROIs for the reusable feature cache.")
        extraction_options, excluded = all_roi_extraction_options(context, options)
        report(f"Extracting {len(extraction_options['selected_parcellations'])} ROIs; "
               f"{len(excluded)} atlas ROIs lack support in at least one source grid.")
        if excluded:
            report(f"Unsupported atlas ROIs: {', '.join(excluded)}")
        extraction_options = dict(extraction_options,
            _source_run_checkpoint_dir=str(directory / "runs" / key))
        payload, error = extract(context, extraction_options)
        if payload is None:
            return payload, error
        payload = dict(payload, source_feature_cache_excluded_rois=excluded,
                       source_feature_cache_scope="all_supported_rois")
        # Never preserve a partial run that skipped input files.
        if payload.get("skipped_paths"):
            report("ROI features include skipped runs; not storing a reusable cache entry.")
            return _select_rois(payload, options)
        temporary = None
        try:
            if feature_cache_identity(context, options)[0] != key:
                report("ROI input dependencies changed during extraction; not caching this run.")
                return _select_rois(payload, options)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f"{key}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
            from core.concept_decoding_cache_portability import translate_paths
            portable_payload = translate_paths(payload, metadata["path_bindings"])
            joblib.dump(dict(key=key, metadata=metadata, payload=portable_payload), temporary, compress=3)
            os.replace(temporary, path)
            report(f"Saved reusable ROI features: {path}")
            return _select_rois(dict(payload, source_feature_cache_hit=False, source_feature_cache_path=str(path)), options)
        except Exception as exc:
            report(f"Could not save ROI feature cache; continuing with extracted features: {exc}")
            return _select_rois(payload, options)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
