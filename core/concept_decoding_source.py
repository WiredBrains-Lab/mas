"""Source-space concept decoding feature extraction."""

import gc
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from core.source_parcellation import (
    HCP_MMP_PARCS,
    get_volume_labels_for_aseg,
    normalize_parcellation_key,
    read_surface_labels,
    subject_aseg_path,
    volume_atlas_ids_for_aseg,
)
from core.concept_decoding_signal import (
    CONCEPT_PREPROCESSING_DEFAULTS,
    annotate_signal_trial_outliers,
    apply_concept_bandpass_to_raw,
    concept_preprocessed_recording,
    _events_for_preprocessed_recording,
    compute_concept_decoding_from_payload,
    concept_bandpass_from_options,
    concept_epoch_reject_from_options,
    concept_epoch_reject_summary,
    concept_feature_binning_enabled,
    concept_label_from_trial_row,
    parcel_trial_frequency_band_settings,
    concept_raw_for_path,
    concept_resample_from_options,
    concept_stat_names,
    emit_concept_progress,
    feature_edges_for_concept_decoding,
    filter_epoch_reject_for_picks,
    signal_trial_metric_rows,
    validate_concept_resample_settings,
    validate_grouped_split,
)
from core.meg_channels import emit_bad_meg_channel_log, good_meg_picks
from core.concept_decoding_spectral import (
    resolve_source_power_bands,
    source_bandpower_trial,
    source_power_feature_metadata,
)


SOURCE_CONCEPT_PARCS = {"aparc", *HCP_MMP_PARCS}
PARCEL_TIMECOURSE_CACHE_VERSION = "parcel_timecourses_v4"


def baseline_zscore_parcel_time_courses(
    label_time_courses,
    times,
    baseline_start=-0.2,
    baseline_end=0.0,
):
    """Z-score every trial/parcel using its own prestimulus baseline."""
    values = np.asarray(label_time_courses, dtype=float)
    sample_times = np.asarray(times, dtype=float)
    if values.ndim != 3 or sample_times.ndim != 1 or values.shape[2] != sample_times.size:
        raise ValueError("Parcel time courses and time axis are inconsistent.")
    try:
        baseline_start = float(baseline_start)
        baseline_end = float(baseline_end)
    except Exception as exc:
        raise ValueError("Parcel baseline limits must be valid times in seconds.") from exc
    if not np.isfinite(baseline_start) or not np.isfinite(baseline_end) or baseline_end <= baseline_start:
        raise ValueError("Parcel baseline end must be greater than baseline start.")

    # Exclude time zero so stimulus-onset activity never enters the baseline.
    tolerance = max(1e-12, np.finfo(float).eps * max(1.0, abs(baseline_end)))
    baseline_mask = (
        (sample_times >= baseline_start - tolerance)
        & (sample_times < baseline_end - tolerance)
    )
    if int(np.sum(baseline_mask)) < 2:
        raise ValueError(
            "Parcel baseline z-scoring requires at least two prestimulus source samples."
        )

    baseline = values[:, :, baseline_mask]
    baseline_mean = np.nanmean(baseline, axis=2, keepdims=True)
    baseline_std = np.nanstd(baseline, axis=2, keepdims=True)
    finite_positive = np.isfinite(baseline_std) & (baseline_std > 0.0)
    valid_scales = baseline_std[finite_positive]
    if valid_scales.size == 0:
        raise ValueError("All parcel prestimulus baselines have zero or invalid variance.")
    scale_floor = max(float(np.nanmedian(valid_scales)) * 1e-6, np.finfo(float).tiny)
    valid = finite_positive & (baseline_std >= scale_floor)
    safe_std = np.where(valid, baseline_std, 1.0)
    z_values = (values - baseline_mean) / safe_std
    z_values = np.where(valid, z_values, 0.0)
    z_values = np.nan_to_num(z_values, nan=0.0, posinf=0.0, neginf=0.0)
    return z_values, int(np.sum(baseline_mask)), int(np.sum(~valid))


def baseline_db_hilbert_parcel_power(
    label_time_courses,
    times,
    baseline_start=-0.2,
    baseline_end=0.0,
    display_start=0.0,
    display_end=0.4,
):
    """Return narrow-band analytic power in dB relative to prestimulus power.

    Hilbert power is computed before cropping so callers can supply padding on
    both sides of the displayed interval. Each trial/parcel uses its own mean
    linear baseline power as the reference; time zero is excluded from that
    reference.
    """
    values = np.asarray(label_time_courses, dtype=float)
    sample_times = np.asarray(times, dtype=float)
    if values.ndim != 3 or sample_times.ndim != 1 or values.shape[2] != sample_times.size:
        raise ValueError("Parcel time courses and time axis are inconsistent.")
    try:
        baseline_start = float(baseline_start)
        baseline_end = float(baseline_end)
        display_start = float(display_start)
        display_end = float(display_end)
    except Exception as exc:
        raise ValueError("Parcel power baseline and display limits must be valid times.") from exc
    limits = (baseline_start, baseline_end, display_start, display_end)
    if not all(np.isfinite(value) for value in limits):
        raise ValueError("Parcel power baseline and display limits must be finite.")
    if baseline_end <= baseline_start:
        raise ValueError("Parcel power baseline end must be greater than baseline start.")
    if baseline_end > 0.0:
        raise ValueError(
            "Parcel band-power baseline must end at or before stimulus onset (0 ms)."
        )
    if display_end <= display_start:
        raise ValueError("Parcel power display end must be greater than display start.")

    tolerance = max(1e-12, np.finfo(float).eps * max(1.0, abs(baseline_end)))
    baseline_mask = (
        (sample_times >= baseline_start - tolerance)
        & (sample_times < baseline_end - tolerance)
    )
    if int(np.sum(baseline_mask)) < 2:
        raise ValueError(
            "Parcel band power requires at least two prestimulus source samples."
        )
    display_mask = (
        (sample_times >= display_start - tolerance)
        & (sample_times <= display_end + tolerance)
    )
    if not np.any(display_mask):
        raise ValueError("Parcel band power display window contains no source samples.")

    analytic = hilbert(values, axis=2)
    power = np.square(np.abs(analytic))
    baseline_power = np.nanmean(power[:, :, baseline_mask], axis=2, keepdims=True)
    finite_positive = np.isfinite(baseline_power) & (baseline_power > 0.0)
    valid_references = baseline_power[finite_positive]
    if valid_references.size == 0:
        raise ValueError("All parcel prestimulus power references are zero or invalid.")
    reference_floor = max(
        float(np.nanmedian(valid_references)) * 1e-12,
        np.finfo(float).tiny,
    )
    valid = finite_positive & (baseline_power >= reference_floor)
    safe_reference = np.where(valid, baseline_power, 1.0)
    safe_power = np.maximum(power, safe_reference * 1e-12)
    power_db = 10.0 * np.log10(safe_power / safe_reference)
    power_db = np.where(valid, power_db, 0.0)
    power_db = np.nan_to_num(power_db, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        power_db[:, :, display_mask],
        sample_times[display_mask],
        int(np.sum(baseline_mask)),
        int(np.sum(~valid)),
    )


def parcel_power_hilbert_padding_s(low_hz):
    """Choose bounded edge padding based on two cycles of the band low edge."""
    try:
        low_hz = float(low_hz)
    except Exception as exc:
        raise ValueError("Band power requires a finite positive band low cutoff.") from exc
    if not np.isfinite(low_hz) or low_hz <= 0.0:
        raise ValueError("Band power requires a finite positive band low cutoff.")
    return float(np.clip(2.0 / low_hz, 0.25, 1.0))


def source_trial_feature_values(label_time_course, stat_names, bin_masks):
    """Return signed temporal features plus polarity-invariant RMS features."""
    tc = np.asarray(label_time_course, dtype=float)
    if tc.ndim != 2:
        return []
    values = []
    for mask in bin_masks:
        window_tc = tc[:, mask]
        if window_tc.ndim != 2 or window_tc.shape[1] == 0:
            return []
        if "raw" in stat_names:
            values.extend(window_tc.reshape(-1).tolist())
            continue
        if "mean" in stat_names:
            values.extend(np.nanmean(window_tc, axis=1).tolist())
        if "rms" in stat_names:
            values.extend(np.sqrt(np.nanmean(np.square(window_tc), axis=1)).tolist())
    return values


def _meg_3d_viewer(context):
    """Return the active MEG 3D viewer."""
    return getattr(context, "meg_3d", None)


def _context_log(context, message):
    try:
        context.c.log.emit(str(message))
    except Exception:
        pass


def _source_concept_parcellation(parcellation):
    parc = normalize_parcellation_key(parcellation or "aparc")
    if parc == "aparc+aseg":
        return "aparc"
    if parc not in SOURCE_CONCEPT_PARCS:
        return "aparc"
    return parc


def _source_space_kind_from_options(options):
    kind = str((options or {}).get("source_space_kind", "surface") or "surface").strip().lower()
    if kind in {"volume", "vol"}:
        return "volume"
    return "surface"


def _set_context_source_space_kind(context, source_space_kind):
    viewer = _meg_3d_viewer(context)
    if viewer is None:
        return
    source_space_kind = "volume" if str(source_space_kind).lower() == "volume" else "surface"
    previous = str(getattr(viewer, "_meg_source_space_kind", "surface") or "surface")
    if previous != source_space_kind:
        for attr in (
            "_meg_inverse_key",
            "_meg_inverse_operator",
            "_meg_source_subject_key",
            "_meg_source_src",
            "_meg_source_bem",
        ):
            try:
                setattr(viewer, attr, None)
            except Exception:
                pass
    try:
        viewer._meg_source_space_kind = source_space_kind
    except Exception:
        pass
    combo = getattr(viewer, "meg_source_space_combo", None)
    if combo is not None:
        try:
            combo.blockSignals(True)
            idx = combo.findData(source_space_kind)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            try:
                combo.blockSignals(False)
            except Exception:
                pass


def _src_source_space_kind(src):
    try:
        kinds = {str(part.get("type", "")).strip().lower() for part in list(src or [])}
    except Exception:
        kinds = set()
    if any(kind in {"vol", "discrete"} or "vol" in kind for kind in kinds):
        return "volume"
    return "surface"


def _inverse_matches_source_space_kind(inv, source_space_kind):
    return _src_source_space_kind((inv or {}).get("src")) == ("volume" if source_space_kind == "volume" else "surface")


def _source_inverse_output_path(context, fif_path, source_space_kind):
    try:
        output_path = context._default_inverse_operator_output_path(fif_path)
    except Exception:
        output_path = None
    if not output_path:
        return None
    if source_space_kind != "volume":
        return str(output_path)
    root, ext = os.path.splitext(str(output_path))
    if root.endswith("-inv"):
        root = root[:-4]
    elif root.endswith("_inv"):
        root = root[:-4]
    return f"{root}-vol-inv{ext or '.fif'}"


def _find_or_create_source_inverse(context, fif_path, source_space_kind):
    _set_context_source_space_kind(context, source_space_kind)
    candidates = []
    expected_path = _source_inverse_output_path(context, fif_path, source_space_kind)
    if expected_path and os.path.isfile(expected_path):
        candidates.append(expected_path)
    try:
        found_path = context._find_inverse_operator_for_fif(fif_path)
    except Exception:
        found_path = None
    if found_path and found_path not in candidates:
        candidates.append(found_path)

    for candidate in candidates:
        try:
            if not context._inverse_operator_matches_fif(candidate, fif_path):
                continue
            inv = mne.minimum_norm.read_inverse_operator(candidate, verbose="error")
            if _inverse_matches_source_space_kind(inv, source_space_kind):
                return candidate, inv, None
        except Exception:
            continue

    create_path = expected_path
    try:
        created_path, create_msg = context._create_inverse_operator_for_fif(
            fif_path,
            output_path=create_path,
        )
        if create_msg:
            _context_log(context, create_msg)
    except Exception as exc:
        return None, None, str(exc)
    if not created_path or not os.path.isfile(created_path):
        return None, None, "missing run-specific inverse operator"
    try:
        inv = mne.minimum_norm.read_inverse_operator(created_path, verbose="error")
    except Exception as exc:
        return None, None, f"could not read inverse operator: {exc}"
    if not _inverse_matches_source_space_kind(inv, source_space_kind):
        return None, None, f"inverse operator is not a {source_space_kind} source-space inverse"
    return created_path, inv, None


def concept_source_labels_for_subject(context, subject, subjects_dir, requested_parcellations=None, parcellation="aparc"):
    """Return surface labels/ROI names for source-space concept decoding."""
    if not subject or not subjects_dir:
        raise RuntimeError("No FreeSurfer subject is available for source-space decoding.")
    parcellation = _source_concept_parcellation(parcellation)
    requested = []
    if isinstance(requested_parcellations, (list, tuple, np.ndarray)):
        requested = [str(v) for v in requested_parcellations if str(v).strip()]

    if parcellation == "aparc":
        available_parcellations = context._get_available_semantic_parcellations(
            subject=subject,
            subjects_dir=subjects_dir,
        )
        if requested:
            available_set = set(available_parcellations)
            roi_names = [name for name in requested if name in available_set]
        else:
            roi_names = list(available_parcellations)
        if not roi_names:
            raise RuntimeError("No source ROI definitions are available.")
        annot_labels = mne.read_labels_from_annot(
            subject=str(subject),
            parc="aparc",
            subjects_dir=str(subjects_dir),
            verbose="error",
        )
    else:
        annot_labels = read_surface_labels("fsaverage", subjects_dir, parcellation)
        available_parcellations = sorted(
            {
                str(getattr(lbl, "name", "") or "")
                for lbl in annot_labels
                if str(getattr(lbl, "name", "") or "").strip()
                and "unknown" not in str(getattr(lbl, "name", "") or "").lower()
            }
        )
        requested_set = set(requested)
        roi_names = [name for name in available_parcellations if name in requested_set] if requested_set else list(available_parcellations)
        if not roi_names:
            roi_names = list(available_parcellations)
        if not roi_names:
            raise RuntimeError(f"No {parcellation} source ROI definitions are available.")

    annot_map = {str(lbl.name): lbl for lbl in annot_labels}
    selected_labels = []
    selected_roi_names = []
    for roi_name in roi_names:
        annot_name = str(context._ctx_volume_name_to_annot_name(roi_name)) if parcellation == "aparc" else str(roi_name)
        lbl = annot_map.get(annot_name)
        if lbl is not None:
            selected_labels.append(lbl)
            selected_roi_names.append(str(roi_name))
    if not selected_labels:
        raise RuntimeError(f"None of the requested source ROI labels were found in the {parcellation} annotation.")
    return selected_labels, selected_roi_names, available_parcellations, parcellation


def concept_volume_labels_for_subject(subject, subjects_dir, requested_labels=None, parcellation=None):
    """Return aparc+aseg or HCP+aseg volume label indices and names for source-space decoding."""
    aseg_path = subject_aseg_path(subject, subjects_dir, parcellation=parcellation)
    if not aseg_path:
        raise RuntimeError("No aparc+aseg volume atlas is available for volume source decoding.")
    label_names, _ = get_volume_labels_for_aseg(str(aseg_path), parcellation=parcellation)
    available = [
        name
        for name in label_names
        if str(name).strip() and "unknown" not in str(name).lower()
    ]
    requested = []
    if isinstance(requested_labels, (list, tuple, np.ndarray)):
        requested = [str(v) for v in requested_labels if str(v).strip()]
    requested_set = set(requested)
    selected_names = [name for name in available if name in requested_set] if requested_set else list(available)
    if not selected_names:
        selected_names = list(available)
    if not selected_names:
        raise RuntimeError("No volume ROI definitions are available in aparc+aseg.")
    selected_set = set(selected_names)
    selected_indices = [idx for idx, name in enumerate(label_names) if name in selected_set]
    if not selected_indices:
        raise RuntimeError("Requested volume labels were not found in aparc+aseg.")
    return str(aseg_path), label_names, selected_indices, selected_names, available


def ensure_fsaverage_in_subjects_dir(subjects_dir):
    """Return a subjects_dir containing fsaverage, linking MNE's copy if needed."""
    if not subjects_dir:
        return None
    try:
        subjects_dir = Path(subjects_dir).expanduser().resolve()
        dest = subjects_dir / "fsaverage"
        if dest.exists():
            return str(subjects_dir)

        sample_subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
        mne.datasets.fetch_fsaverage(subjects_dir=sample_subjects_dir, verbose=False)
        src = sample_subjects_dir / "fsaverage"
        if not src.exists():
            return None

        subjects_dir.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(src, target_is_directory=True)
        except FileExistsError:
            pass
        return str(subjects_dir) if dest.exists() else None
    except Exception:
        return None


def fsaverage_source_space(subjects_dir):
    """Return the fsaverage ico-5 surface source space used after morphing."""
    common_subjects_dir = ensure_fsaverage_in_subjects_dir(subjects_dir)
    if not common_subjects_dir:
        sample_subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
        mne.datasets.fetch_fsaverage(subjects_dir=sample_subjects_dir, verbose=False)
        common_subjects_dir = str(sample_subjects_dir)

    src_path = Path(common_subjects_dir) / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
    if src_path.exists():
        return mne.read_source_spaces(str(src_path), verbose="error"), str(common_subjects_dir)
    src = mne.setup_source_space(
        "fsaverage",
        spacing="ico5",
        subjects_dir=str(common_subjects_dir),
        add_dist=False,
        n_jobs=1,
        verbose="error",
    )
    return src, str(common_subjects_dir)


def source_space_vertices_signature(src):
    """Return a compact vertex signature for a surface source space."""
    try:
        return tuple(tuple(np.asarray(part.get("vertno", []), dtype=int).tolist()) for part in list(src)[:2])
    except Exception:
        return tuple()


def stc_vertices_signature(stc):
    try:
        return tuple(tuple(np.asarray(v, dtype=int).tolist()) for v in getattr(stc, "vertices", []))
    except Exception:
        return tuple()


def configure_source_subject_for_fif(context, fif_path, options=None):
    """Resolve the FreeSurfer subject that belongs to a FIF file."""
    cfg_ok, cfg_msg = context._auto_configure_freesurfer_subject_for_fif(
        fif_path,
        require_bem=False,
    )
    if cfg_msg:
        _context_log(context, f"MEG source concept config: {cfg_msg}")
    if not cfg_ok:
        return None, cfg_msg or "FreeSurfer subject configuration failed"
    viewer = _meg_3d_viewer(context)
    if viewer is None:
        return None, "3D viewer is required for source-space concept decoding."
    try:
        subject, subjects_dir = viewer._get_brain_subject_info()
    except Exception as exc:
        return None, f"FreeSurfer subject lookup failed: {exc}"
    if not subject or not subjects_dir:
        return None, "No FreeSurfer subject is available for this FIF file"
    return (str(subject), os.path.abspath(str(subjects_dir))), None


def morph_stcs_to_common_space(stcs, subject_from, subjects_dir_from, common_src_sig, morph_cache):
    """Morph source estimates to fsaverage ico-5 when common-space decoding is needed."""
    if not stcs:
        return stcs
    first_sig = stc_vertices_signature(stcs[0])
    subject_from = str(subject_from or "")
    if subject_from.lower() == "fsaverage" and first_sig == common_src_sig:
        return stcs

    morph_subjects_dir = ensure_fsaverage_in_subjects_dir(subjects_dir_from)
    if not morph_subjects_dir:
        raise RuntimeError(
            f"Cannot morph {subject_from or 'unknown subject'} to fsaverage: "
            "fsaverage is not available in the source SUBJECTS_DIR."
        )

    morph_key = (
        subject_from,
        "fsaverage",
        str(Path(morph_subjects_dir).expanduser().resolve()),
        first_sig,
    )
    morph = morph_cache.get(morph_key)
    if morph is None:
        morph = mne.compute_source_morph(
            stcs[0],
            subject_from=subject_from,
            subject_to="fsaverage",
            subjects_dir=morph_subjects_dir,
            spacing=5,
            warn=False,
            verbose="error",
        )
        morph_cache[morph_key] = morph
    return [morph.apply(stc) for stc in stcs]


def source_bin_masks(times, bin_edges, include_last_endpoint=True):
    times = np.asarray(times, dtype=float)
    # Repeated decimal edge construction (for example, 5 ms bins) can place a
    # nominal boundary a few ulps above or below the corresponding MNE sample.
    # Use one small boundary tolerance and assign values within that tolerance
    # to the bin on the right so bins equal to the sample interval stay disjoint
    # and contain exactly one sample.
    finite_times = times[np.isfinite(times)]
    scale = float(np.nanmax(np.abs(finite_times))) if finite_times.size else 1.0
    tolerance = max(1e-12, np.finfo(float).eps * max(1.0, scale) * 16.0)
    masks = []
    for idx, (bin_start, bin_end) in enumerate(bin_edges):
        if idx == len(bin_edges) - 1 and include_last_endpoint:
            mask = (times >= float(bin_start) - tolerance) & (
                times <= float(bin_end) + tolerance
            )
        else:
            mask = (times >= float(bin_start) - tolerance) & (
                times < float(bin_end) - tolerance
            )
        masks.append(mask)
    return masks


def source_feature_names_and_meta(roi_names, stat_names, bin_edges, bin_masks, times):
    feature_names = []
    feature_meta = []
    raw_sample_counts = []
    for bin_idx, (bin_start, bin_end) in enumerate(bin_edges):
        mask = np.asarray(bin_masks[bin_idx], dtype=bool)
        bin_times = np.asarray(times, dtype=float)[mask]
        if "raw" in stat_names:
            raw_sample_counts.append(int(bin_times.size))
            for roi_name in roi_names:
                for sample_idx, rel_time_s in enumerate(bin_times.tolist()):
                    feature_names.append(
                        f"{roi_name}|raw|{bin_start:.3f}-{bin_end:.3f}s|sample_{sample_idx}"
                    )
                    feature_meta.append(
                        {
                            "feature": feature_names[-1],
                            "roi": roi_name,
                            "stat": "raw",
                            "bin_start_s": float(bin_start),
                            "bin_end_s": float(bin_end),
                            "sample_index": int(sample_idx),
                            "relative_time_s": float(rel_time_s),
                        }
                    )
            continue

        for stat_name in stat_names:
            for roi_name in roi_names:
                feature_names.append(f"{roi_name}|{stat_name}|{bin_start:.3f}-{bin_end:.3f}s")
                feature_meta.append(
                    {
                        "feature": feature_names[-1],
                        "roi": roi_name,
                        "stat": stat_name,
                        "bin_start_s": float(bin_start),
                        "bin_end_s": float(bin_end),
                    }
                )
    return feature_names, feature_meta, raw_sample_counts


def restore_meg_sensor_fif(context, fif_filename):
    """Restore the 3D MEG sensor context after multi-run source work."""
    viewer = _meg_3d_viewer(context)
    if not fif_filename or viewer is None:
        return
    try:
        if hasattr(viewer, "_fif_file_identity"):
            target_identity = viewer._fif_file_identity(fif_filename)
            current_identity = getattr(viewer, "_meg_current_fif_identity", None)
            if target_identity == current_identity:
                return
        else:
            current_path = os.path.abspath(str(getattr(viewer, "_meg_current_fif_path", "") or ""))
            if current_path == os.path.abspath(str(fif_filename)):
                return
        viewer.load_fif_sensors(fif_filename)
    except Exception:
        pass


def build_source_space_concept_features(context, options=None):
    """Extract ROI features, reusing the data-local persistent cache by default."""
    options = options or {}
    if options.get("source_feature_cache", True):
        from core.concept_decoding_feature_cache import cached_source_features
        return cached_source_features(context, options, _extract_source_space_concept_features)
    return _extract_source_space_concept_features(context, options)


def _extract_source_run(context, options, raw, path_abs, file_df, *,
    band_defs,
    bandpass_high_hz,
    bandpass_low_hz,
    baseline,
    bin_edges,
    common_src_sig,
    epoch_reject,
    epoch_tmax,
    epoch_tmin,
    exclude_qc_outliers,
    feature_binning,
    lambda2,
    method,
    min_outlier_trials,
    ordered_subject_keys,
    parcel_baseline_end,
    parcel_baseline_start,
    power_baseline,
    resample_enabled,
    resample_sfreq,
    selected_roi_names,
    source_batch_size,
    source_label_src,
    source_labels,
    source_parcel_baseline_zscore,
    source_power,
    source_space_kind,
    stat_names,
    trial_outlier_z,
    use_common_space,
    volume_aseg_path,
    volume_label_indices,
    window_end,
    window_start,
):
    """Extract one run with private accumulators; shared by serial and process paths."""
    feature_rows = []
    kept_rows = []
    trial_qc_rows = []
    dropped_trials = 0
    feature_names = None
    feature_meta = None
    expected_raw_sample_counts = None
    inverse_paths = {}
    morph_cache = {}
    parcel_baseline_sample_count = 0
    invalid_parcel_baseline_count = 0

    skipped_paths = []
    source_subject_by_path = {}
    raw_by_path = {path_abs: raw}
    grouped_files = [(path_abs, file_df)]
    for file_idx, (path_abs, file_df) in enumerate(grouped_files, start=1):
        emit_concept_progress(
            options,
            f"extracting source features run {file_idx}/{len(grouped_files)}: {os.path.basename(path_abs)}",
        )
        raw = raw_by_path.get(path_abs)
        if raw is None:
            continue
        path_qc_rows = []
        path_feature_start = len(feature_rows)
        path_kept_start = len(kept_rows)
        try:
            current_subject_key, key_err = configure_source_subject_for_fif(context, path_abs, options=options)
            if current_subject_key is None:
                skipped_paths.append((path_abs, key_err or "FreeSurfer subject configuration failed"))
                continue
            source_subject_by_path[path_abs] = current_subject_key
            if not use_common_space and current_subject_key != ordered_subject_keys[0]:
                skipped_paths.append((path_abs, "different FreeSurfer subject than native decoding subject"))
                continue

            inv_path, inv, inv_err = _find_or_create_source_inverse(
                context,
                path_abs,
                source_space_kind,
            )
            if inv is None or not inv_path:
                skipped_paths.append((path_abs, inv_err or f"missing run-specific {source_space_kind} inverse operator"))
                continue
            inverse_paths[path_abs] = inv_path

            picks, bad_names = good_meg_picks(raw.info, meg=True, ref_meg=False)
            if bad_names:
                emit_bad_meg_channel_log(
                    getattr(getattr(context, "c", None), "log", None),
                    "Source-space decoding",
                    fif_path=path_abs,
                    bad_names=bad_names,
                )
            if picks.size == 0:
                skipped_paths.append((path_abs, "no MEG channels"))
                continue
            path_epoch_reject = filter_epoch_reject_for_picks(raw.info, picks, epoch_reject)
            sfreq = float(raw.info["sfreq"])
            first_samp = int(getattr(raw, "first_samp", 0))
            n_times = int(raw.n_times)
            events = []
            input_rows = []
            for row_idx, row in file_df.iterrows():
                try:
                    onset_sample = int(round(float(row.get("onset_sample"))))
                except Exception:
                    dropped_trials += 1
                    continue
                s0 = int(round(onset_sample + epoch_tmin * sfreq))
                s1 = int(round(onset_sample + epoch_tmax * sfreq))
                if s0 < 0 or s1 >= n_times or s1 <= s0:
                    dropped_trials += 1
                    continue
                events.append([int(onset_sample + first_samp), 0, 1])
                kept = row.to_dict()
                kept["_source_table_index"] = int(row_idx)
                input_rows.append(kept)
            if not events:
                continue

            input_df = pd.DataFrame(input_rows).reset_index(drop=True)
            events_arr = np.asarray(events, dtype=np.int64)
            if bool(options.get("preprocess_continuous_recording", True)):
                prepared_raw = concept_preprocessed_recording(
                    raw, path_abs, options, bandpass_low_hz, bandpass_high_hz,
                    resample_sfreq if resample_enabled else None,
                )
                events_arr = _events_for_preprocessed_recording(events_arr, raw, prepared_raw)
            else:
                prepared_raw = apply_concept_bandpass_to_raw(raw, bandpass_low_hz, bandpass_high_hz)
            path_failed = False
            for batch_start in range(0, len(events_arr), source_batch_size):
                batch_stop = min(len(events_arr), batch_start + source_batch_size)
                emit_concept_progress(options, (
                    f"source batch {batch_start // source_batch_size + 1}/"
                    f"{(len(events_arr) + source_batch_size - 1) // source_batch_size}: "
                    f"{os.path.basename(path_abs)}"
                ))
                batch_events = events_arr[batch_start:batch_stop]
                batch_input_df = input_df.iloc[batch_start:batch_stop].reset_index(drop=True)
                epochs = None
                stcs = None
                stcs_for_labels = None
                label_tc = None
                try:
                    epochs = mne.Epochs(
                        prepared_raw,
                        batch_events,
                        event_id={"trial": 1},
                        tmin=epoch_tmin,
                        tmax=epoch_tmax,
                        baseline=baseline,
                        reject=path_epoch_reject,
                        reject_tmin=(min(window_start, parcel_baseline_start) if power_baseline else window_start) if source_power else None,
                        reject_tmax=(max(window_end, parcel_baseline_end) if power_baseline else window_end) if source_power else None,
                        picks=picks,
                        preload=True,
                        reject_by_annotation=True,
                        event_repeated="drop",
                        verbose="error",
                    )
                    if len(epochs) == 0:
                        dropped_trials += int(len(batch_events))
                        continue
                    if resample_enabled and resample_sfreq is not None:
                        current_sfreq = float(epochs.info["sfreq"])
                        target_sfreq = float(resample_sfreq)
                        if target_sfreq > current_sfreq + 1e-6:
                            raise RuntimeError(
                                f"downsample target ({target_sfreq:g} Hz) is above this run's sampling rate "
                                f"({current_sfreq:g} Hz)"
                            )
                        if target_sfreq < current_sfreq - 1e-6:
                            emit_concept_progress(
                                options,
                                f"resampling source epochs: {current_sfreq:g} Hz -> {target_sfreq:g} Hz",
                            )
                            epochs.resample(target_sfreq, npad="auto", n_jobs=1, verbose="error")
                    selected_idx = np.asarray(
                        getattr(epochs, "selection", np.arange(len(epochs))),
                        dtype=int,
                    )
                    batch_input_df = batch_input_df.iloc[selected_idx].reset_index(drop=True)

                    inverse_orientation = None
                    if source_power:
                        from mne.minimum_norm.inverse import is_fixed_orient
                        inverse_orientation = None if is_fixed_orient(inv) else "vector"
                    stcs = mne.minimum_norm.apply_inverse_epochs(
                        epochs,
                        inv,
                        lambda2=lambda2,
                        method=method,
                        pick_ori=inverse_orientation,
                        return_generator=False,
                        verbose="error",
                    )
                    if not stcs:
                        dropped_trials += len(batch_input_df)
                        continue
                    stcs_for_labels = stcs
                    src_for_labels = inv["src"]
                    volume_atlas_ids = volume_atlas_ids_for_aseg(volume_aseg_path) if source_space_kind == "volume" else None
                    if source_power:
                        times = np.asarray(stcs[0].times, dtype=float)
                        power_masks = source_bin_masks(times, bin_edges, include_last_endpoint=not feature_binning)
                        baseline_mask = source_bin_masks(
                            times, [(parcel_baseline_start, parcel_baseline_end)],
                            include_last_endpoint=False,
                        )[0] if power_baseline else None
                        parcel_baseline_sample_count = int(np.sum(baseline_mask)) if power_baseline else 0
                        vol_power_labels = (
                            (volume_aseg_path, {name: volume_atlas_ids[name] for name in selected_roi_names})
                            if volume_atlas_ids else (volume_aseg_path, selected_roi_names)
                        ) if source_space_kind == "volume" else None

                        def aggregate_power(power_stc):
                            if source_space_kind == "volume":
                                return power_stc.extract_label_time_course(
                                    vol_power_labels, src=inv["src"],
                                    mode="mean", allow_empty=False, verbose="error",
                                )
                            if use_common_space:
                                power_stc = morph_stcs_to_common_space(
                                    [power_stc], current_subject_key[0], current_subject_key[1],
                                    common_src_sig, morph_cache,
                                )[0]
                            return mne.extract_label_time_course(
                                power_stc, source_labels,
                                source_label_src if use_common_space else inv["src"],
                                mode="mean", allow_empty=False, verbose="error",
                            )

                        spectral_features = np.stack([
                            source_bandpower_trial(stc, band_defs, power_masks, baseline_mask, aggregate_power)
                            for stc in stcs
                        ])
                        # The ordinary waveform branch uses this only for trial count below.
                        label_tc = spectral_features.reshape(len(stcs), len(selected_roi_names), -1)
                    else:
                        if source_space_kind == "volume":
                            vol_all_labels = (
                                (volume_aseg_path, {name: volume_atlas_ids[name] for name in volume_label_names})
                                if volume_atlas_ids else volume_aseg_path
                            )
                            label_tc_rows = []
                            for stc_one in stcs_for_labels:
                                tc = np.asarray(
                                    stc_one.extract_label_time_course(
                                        vol_all_labels,
                                        src=src_for_labels,
                                        mode="mean",
                                        allow_empty=True,
                                        verbose="error",
                                    ),
                                    dtype=float,
                                )
                                if tc.ndim != 2:
                                    label_tc_rows = []
                                    break
                                label_tc_rows.append(tc[np.asarray(volume_label_indices, dtype=int), :])
                            label_tc = np.asarray(label_tc_rows, dtype=float)
                        elif use_common_space:
                            stcs_for_labels = morph_stcs_to_common_space(
                                stcs,
                                current_subject_key[0],
                                current_subject_key[1],
                                common_src_sig,
                                morph_cache,
                            )
                            src_for_labels = source_label_src
                            label_tc = mne.extract_label_time_course(
                                stcs_for_labels,
                                source_labels,
                                src_for_labels,
                                mode="mean_flip",
                                return_generator=False,
                                allow_empty=True,
                                verbose="error",
                            )
                            label_tc = np.asarray(label_tc, dtype=float)
                        else:
                            label_tc = mne.extract_label_time_course(
                                stcs_for_labels,
                                source_labels,
                                src_for_labels,
                                mode="mean_flip",
                                return_generator=False,
                                allow_empty=True,
                                verbose="error",
                            )
                            label_tc = np.asarray(label_tc, dtype=float)
                    if label_tc.ndim != 3 or label_tc.shape[0] != len(batch_input_df):
                        dropped_trials += len(batch_input_df)
                        continue
                    times = np.asarray(stcs_for_labels[0].times, dtype=float)
                    if source_parcel_baseline_zscore:
                        label_tc, batch_baseline_samples, batch_invalid_baselines = (
                            baseline_zscore_parcel_time_courses(
                                label_tc,
                                times,
                                baseline_start=parcel_baseline_start,
                                baseline_end=parcel_baseline_end,
                            )
                        )
                        if (
                            parcel_baseline_sample_count
                            and batch_baseline_samples != parcel_baseline_sample_count
                        ):
                            raise RuntimeError(
                                "Source runs produced different parcel-baseline sample counts."
                            )
                        parcel_baseline_sample_count = int(batch_baseline_samples)
                        invalid_parcel_baseline_count += int(batch_invalid_baselines)
                    try:
                        sensor_epoch_data = epochs.get_data(copy=False)
                    except TypeError:
                        sensor_epoch_data = epochs.get_data()
                    batch_qc_df = signal_trial_metric_rows(
                        sensor_epoch_data,
                        np.asarray(epochs.times, dtype=float),
                        bin_edges,
                        batch_input_df,
                        path_abs,
                    )
                    bin_masks = source_bin_masks(
                        times,
                        bin_edges,
                        include_last_endpoint=not feature_binning,
                    )
                    if any(not np.any(mask) for mask in bin_masks):
                        skipped_paths.append((path_abs, "source epoch time grid does not cover every feature bin"))
                        path_failed = True
                        break

                    if source_power and feature_names is None:
                        feature_meta = source_power_feature_metadata(selected_roi_names, band_defs, bin_edges, power_baseline)
                        feature_names = [row["feature"] for row in feature_meta]
                    elif feature_names is None:
                        feature_names, feature_meta, expected_raw_sample_counts = source_feature_names_and_meta(
                            selected_roi_names,
                            stat_names,
                            bin_edges,
                            bin_masks,
                            times,
                        )
                    elif "raw" in stat_names:
                        raw_counts = [int(np.sum(mask)) for mask in bin_masks]
                        if raw_counts != list(expected_raw_sample_counts):
                            skipped_paths.append((path_abs, "source raw sample bins differ from first decoded run"))
                            path_failed = True
                            break

                    for trial_idx in range(label_tc.shape[0]):
                        tc = np.asarray(label_tc[trial_idx], dtype=float)
                        values = (
                            spectral_features[trial_idx].ravel()
                            if source_power else source_trial_feature_values(tc, stat_names, bin_masks)
                        )

                        if feature_names is None or len(values) != len(feature_names):
                            dropped_trials += 1
                            continue
                        feature_rows.append(
                            np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                        )
                        kept_rows.append(batch_input_df.iloc[trial_idx].to_dict())
                        qc_row = (
                            batch_qc_df.iloc[trial_idx].to_dict()
                            if trial_idx < len(batch_qc_df)
                            else batch_input_df.iloc[trial_idx].to_dict()
                        )
                        path_qc_rows.append(qc_row)
                finally:
                    del epochs, stcs, stcs_for_labels, label_tc
                    gc.collect()
            if path_failed:
                del feature_rows[path_feature_start:]
                del kept_rows[path_kept_start:]
                continue

            path_qc_df = annotate_signal_trial_outliers(
                pd.DataFrame(path_qc_rows),
                threshold=trial_outlier_z,
                min_trials=min_outlier_trials,
            )
            path_feature_count = len(feature_rows) - path_feature_start
            if len(path_qc_df) != path_feature_count:
                del feature_rows[path_feature_start:]
                del kept_rows[path_kept_start:]
                skipped_paths.append((path_abs, "source trial QC alignment failed"))
                continue
            keep_path = np.ones(path_feature_count, dtype=bool)
            for local_idx in range(path_feature_count):
                qc_row = path_qc_df.iloc[local_idx].to_dict()
                is_outlier = bool(qc_row.get("qc_outlier", False))
                if is_outlier and exclude_qc_outliers:
                    keep_path[local_idx] = False
                    qc_row["qc_status"] = "dropped"
                    qc_row["qc_reason"] = "qc_outlier"
                    dropped_trials += 1
                else:
                    qc_row["qc_status"] = "kept"
                    qc_row["qc_reason"] = "qc_outlier_not_excluded" if is_outlier else ""
                trial_qc_rows.append(qc_row)
            if not np.all(keep_path):
                path_features = feature_rows[path_feature_start:]
                path_kept = kept_rows[path_kept_start:]
                del feature_rows[path_feature_start:]
                del kept_rows[path_kept_start:]
                feature_rows.extend([row for row, keep in zip(path_features, keep_path) if keep])
                kept_rows.extend([row for row, keep in zip(path_kept, keep_path) if keep])
        except Exception as exc:
            del feature_rows[path_feature_start:]
            del kept_rows[path_kept_start:]
            skipped_paths.append((path_abs, str(exc)))
            continue

    return {
        "feature_rows": feature_rows,
        "kept_rows": kept_rows,
        "trial_qc_rows": trial_qc_rows,
        "dropped_trials": dropped_trials,
        "feature_names": feature_names,
        "feature_meta": feature_meta,
        "expected_raw_sample_counts": expected_raw_sample_counts,
        "inverse_paths": inverse_paths,
        "parcel_baseline_sample_count": parcel_baseline_sample_count,
        "invalid_parcel_baseline_count": invalid_parcel_baseline_count,
        "skipped_paths": skipped_paths,
    }


def _extract_source_space_concept_features(context, options=None):
    """Extract parcellated source-space MEG features across mapped FIF runs."""
    if options is None:
        options = {}
    emit_concept_progress(options, "building source-space ROI features...")
    trial_df = getattr(context, "_fif_trial_response_table", pd.DataFrame())
    if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
        return None, "No mapped trial table is available. Load FIF runs with events_log mapping first."
    if "source_fif_path" not in trial_df.columns:
        return None, "Mapped trial table does not include source FIF paths."
    if _meg_3d_viewer(context) is None:
        return None, "3D viewer is required for source-space concept decoding."

    group_mode = str(options.get("group_mode", "canonical") or "canonical").strip().lower()
    if group_mode not in {"canonical", "category", "individual", "semantic_decision"}:
        group_mode = "canonical"
    try:
        window_start = float(options.get("window_start", 0.15))
        window_end = float(options.get("window_end", 0.45))
    except Exception:
        return None, "Invalid concept decoding window."
    if not np.isfinite(window_start) or not np.isfinite(window_end) or window_end <= window_start:
        return None, "Concept decoding requires a valid feature window."
    try:
        shared_bandpass_low_hz, shared_bandpass_high_hz = concept_bandpass_from_options(
            options
        )
    except ValueError as exc:
        return None, str(exc)
    parcel_frequency_band = str(
        options.get("parcel_frequency_band", "broadband") or "broadband"
    )
    try:
        bandpass_low_hz, bandpass_high_hz, _frequency_band_label = (
            parcel_trial_frequency_band_settings(
                parcel_frequency_band,
                shared_bandpass_low_hz,
                shared_bandpass_high_hz,
            )
        )
    except ValueError as exc:
        return None, str(exc)
    try:
        min_trials = max(2, int(options.get("min_trials_per_class", 2)))
    except Exception:
        min_trials = 2
    try:
        max_rois = int(options.get("max_rois", 0))
    except Exception:
        max_rois = 0

    feature_binning = concept_feature_binning_enabled(options)
    if feature_binning:
        bin_edges = feature_edges_for_concept_decoding(
            window_start,
            window_end,
            options.get("bin_size_ms", 50),
        )
    else:
        bin_edges = [(float(window_start), float(window_end))]
    if not bin_edges:
        return None, "Concept decoding feature bins could not be constructed."
    source_power = str(options.get("feature_mode", "")).strip().lower() == "bandpower"
    stat_names = ["bandpower"] if source_power else concept_stat_names(options.get("feature_mode", "mean_rms"))
    power_baseline = bool(options.get("source_power_baseline", True)) if source_power else False
    band_defs = []
    power_padding = 0.0
    source_parcel_baseline_zscore = bool(
        options.get("source_parcel_baseline_zscore", False)
    ) and not source_power
    try:
        parcel_baseline_start = float(options.get("parcel_baseline_start", -0.2))
        parcel_baseline_end = float(options.get("parcel_baseline_end", 0.0))
    except Exception:
        return None, "Invalid source-parcel prestimulus baseline window."
    if (source_parcel_baseline_zscore or power_baseline) and (
        not np.isfinite(parcel_baseline_start)
        or not np.isfinite(parcel_baseline_end)
        or parcel_baseline_end <= parcel_baseline_start
    ):
        return None, "Source-parcel baseline end must be greater than baseline start."
    if power_baseline and parcel_baseline_end > 0.0:
        return None, "Source power baseline must end at or before stimulus onset."
    # Parcel z-scoring removes the prestimulus mean and also scales by that
    # trial/parcel's baseline deviation. Avoid redundant sensor-level mean
    # subtraction when this stronger source normalization is enabled.
    baseline_correct = bool(options.get("baseline_correct", True)) and not source_parcel_baseline_zscore and not source_power
    try:
        resample_enabled, resample_sfreq = concept_resample_from_options(options)
    except ValueError as exc:
        return None, str(exc)
    requested_parcellations = options.get("selected_parcellations", [])
    source_space_kind = _source_space_kind_from_options(options)
    requested_parc = _source_concept_parcellation(options.get("source_parcellation", "aparc"))
    source_parcellation = (
        requested_parc
        if (source_space_kind != "volume" or requested_parc in HCP_MMP_PARCS)
        else "aparc+aseg"
    )
    try:
        source_batch_size = int(options.get("source_batch_size", 4 if source_power else (8 if "raw" in stat_names else 16)))
    except Exception:
        source_batch_size = 4 if source_power else (8 if "raw" in stat_names else 16)
    source_batch_size = max(1, min(64, source_batch_size))
    try:
        max_source_features = int(options.get("max_source_features", 75000))
    except Exception:
        max_source_features = 75000
    max_source_features = max(1000, max_source_features)
    method = str(options.get("method", "MNE") or "MNE").strip()
    if method.upper() == "LCMV":
        return None, "Source-space concept decoding currently supports MNE/dSPM/sLORETA/eLORETA, not LCMV."
    try:
        lambda2 = float(options.get("lambda2", 1.0 / 9.0))
    except Exception:
        lambda2 = 1.0 / 9.0
    lambda2 = max(lambda2, 1e-12)

    df = context._annotate_trial_df_with_stimulus_groups(
        trial_df.copy(),
        getattr(context, "_fif_filename", None),
    )
    use_stimulus_ui_labels = bool(options.get("_use_stimulus_ui_labels", False))

    def _stimulus_ui_label(row):
        if group_mode == "individual":
            stimulus = row.get("stimulus")
            if pd.notna(stimulus) and str(stimulus).strip():
                return os.path.basename(str(stimulus).strip())
        if group_mode == "category":
            stimulus_group = row.get("stimulus_group", "")
            if pd.notna(stimulus_group) and str(stimulus_group).strip():
                return str(stimulus_group).strip()
        canonical = row.get("canonical_stimulus")
        if pd.notna(canonical) and str(canonical).strip():
            return str(canonical).strip()
        return context._canonicalize_meg_stimulus_label(row.get("stimulus"))

    if use_stimulus_ui_labels:
        df["concept_label"] = [_stimulus_ui_label(row) for _, row in df.iterrows()]
    else:
        df["concept_label"] = [
            concept_label_from_trial_row(context, row, group_mode=group_mode)
            for _, row in df.iterrows()
        ]
    df["concept_label"] = df["concept_label"].astype(str).str.strip()
    valid = (
        df["concept_label"].astype(bool)
        & (df["concept_label"].str.lower() != "unknown")
        & df["source_fif_path"].astype(str).str.len().gt(0)
    )
    df = df.loc[valid].copy().reset_index(drop=True)
    if df.empty:
        return None, "No valid concept-labeled trials are available for source-space decoding."

    selected_source_paths = options.get("source_fif_paths", [])
    if isinstance(selected_source_paths, (list, tuple, np.ndarray)) and len(selected_source_paths) > 0:
        selected_source_paths = {
            os.path.abspath(str(path))
            for path in selected_source_paths
            if str(path or "").strip()
        }
        df["source_fif_path"] = df["source_fif_path"].map(lambda path: os.path.abspath(str(path)))
        df = df[df["source_fif_path"].isin(selected_source_paths)].copy().reset_index(drop=True)
        if df.empty:
            return None, "No mapped trials belong to the selected source run scope."

    selected_concept_label = str(options.get("selected_concept_label", "") or "").strip()
    if selected_concept_label and selected_concept_label != "All stimuli":
        df = df[df["concept_label"] == selected_concept_label].copy().reset_index(drop=True)
        if df.empty:
            return None, f"No trials matched stimulus group '{selected_concept_label}'."

    try:
        max_presentations = max(0, int(options.get("max_presentations", 0)))
    except Exception:
        max_presentations = 0
    limit_presentations_after_qc = bool(
        options.get("_limit_presentations_after_qc", False)
    )
    try:
        presentation_candidate_limit = max(
            0, int(options.get("_presentation_candidate_limit", 0))
        )
    except Exception:
        presentation_candidate_limit = 0
    matched_presentations_before_candidate_limit = int(len(df))
    if (
        limit_presentations_after_qc
        and presentation_candidate_limit > 0
        and len(df) > presentation_candidate_limit
    ):
        # Parcel visualization only needs enough candidates to fill the requested
        # panels after rejection/QC. Reconstructing every occurrence before
        # truncating made category/all-run views unnecessarily expensive.
        candidate_rank = df.groupby("concept_label", sort=False).cumcount()
        df = df.loc[candidate_rank < presentation_candidate_limit].copy().reset_index(drop=True)
        emit_concept_progress(
            options,
            (
                f"source reconstruction candidate pool: {len(df)} of "
                f"{matched_presentations_before_candidate_limit} matched presentations"
            ),
        )
    if (
        max_presentations > 0
        and len(df) > max_presentations
        and not limit_presentations_after_qc
    ):
        df = df.iloc[:max_presentations].copy().reset_index(drop=True)

    allow_single_class = bool(options.get("_allow_single_class", False))
    required_class_count = 1 if allow_single_class else 2
    counts = df["concept_label"].value_counts()
    keep_concepts = counts[counts >= min_trials].index.tolist()
    if len(keep_concepts) < required_class_count:
        if allow_single_class:
            return None, "Need at least one stimulus group with enough trials for source-space visualization."
        return None, "Need at least two concepts with enough trials for source-space decoding."
    df = df[df["concept_label"].isin(keep_concepts)].copy().reset_index(drop=True)

    restore_fif_path = getattr(context, "_fif_filename", None)
    raw_cache = {}
    raw_by_path = {}
    skipped_paths = []
    for path in pd.Index(df["source_fif_path"].astype(str)).drop_duplicates().tolist():
        path_abs = os.path.abspath(str(path))
        try:
            emit_concept_progress(options, f"loading source MEG run {len(raw_by_path) + 1}: {os.path.basename(path_abs)}")
            raw = concept_raw_for_path(context, path_abs, raw_cache)
            raw_by_path[path_abs] = raw
        except Exception as exc:
            skipped_paths.append((path_abs, str(exc)))
    if not raw_by_path:
        detail = f" First error: {skipped_paths[0][1]}" if skipped_paths else ""
        return None, f"Could not load/filter any FIF files for source-space decoding.{detail}"
    try:
        validate_concept_resample_settings(raw_by_path, resample_enabled, resample_sfreq, h_freq=bandpass_high_hz)
    except ValueError as exc:
        return None, str(exc)

    available_paths = set(raw_by_path.keys())
    df["source_fif_path"] = df["source_fif_path"].map(lambda p: os.path.abspath(str(p)))
    df = df[df["source_fif_path"].isin(available_paths)].copy().reset_index(drop=True)
    if df.empty:
        return None, "No trials remained after loading available FIF files."
    df["_source_qc_uid"] = np.arange(len(df), dtype=int)

    path_order = pd.Index(df["source_fif_path"].astype(str)).drop_duplicates().tolist()
    source_subject_by_path = {}
    for path_abs in path_order:
        key, key_err = configure_source_subject_for_fif(context, path_abs, options=options)
        if key is None:
            skipped_paths.append((path_abs, key_err or "FreeSurfer subject configuration failed"))
            continue
        source_subject_by_path[path_abs] = key

    ordered_subject_keys = []
    for path_abs in path_order:
        key = source_subject_by_path.get(path_abs)
        if key is not None and key not in ordered_subject_keys:
            ordered_subject_keys.append(key)
    if not ordered_subject_keys:
        detail = f" First error: {skipped_paths[-1][1]}" if skipped_paths else ""
        restore_meg_sensor_fif(context, restore_fif_path)
        return None, f"No FIF run could be matched to a FreeSurfer subject for source-space decoding.{detail}"

    space_option = str(options.get("source_common_space", "auto") or "auto").strip().lower()
    if source_space_kind == "volume" and len(ordered_subject_keys) > 1:
        restore_meg_sensor_fif(context, restore_fif_path)
        return None, (
            "Volume source-space concept decoding currently supports one FreeSurfer subject. "
            "Use surface source space for multi-subject/common-space decoding."
        )
    force_common_space = space_option in {"fsaverage", "common", "common_space", "common-space", "true", "1", "yes"}
    force_common_space = bool(source_space_kind == "surface" and (force_common_space or source_parcellation in HCP_MMP_PARCS))
    native_only = space_option in {"native", "subject", "subject_space", "subject-space", "false", "0", "no"}
    use_common_space = bool(source_space_kind == "surface" and (force_common_space or (len(ordered_subject_keys) > 1 and not native_only)))
    native_subjects = list(dict.fromkeys([key[0] for key in ordered_subject_keys]))

    if use_common_space:
        common_subject = "fsaverage"
        common_subjects_dir = ensure_fsaverage_in_subjects_dir(ordered_subject_keys[0][1])
        if not common_subjects_dir:
            try:
                sample_subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
                mne.datasets.fetch_fsaverage(subjects_dir=sample_subjects_dir, verbose=False)
                common_subjects_dir = str(sample_subjects_dir)
            except Exception:
                common_subjects_dir = None
        if not common_subjects_dir:
            restore_meg_sensor_fif(context, restore_fif_path)
            reason = (
                f"{source_parcellation} source decoding"
                if source_parcellation in HCP_MMP_PARCS
                else "Multi-subject source decoding"
            )
            return None, f"{reason} needs fsaverage, but fsaverage could not be made available."
        common_reason = (
            f"{source_parcellation} atlas"
            if source_parcellation in HCP_MMP_PARCS
            else f"{len(ordered_subject_keys)} FreeSurfer subjects"
        )
        emit_concept_progress(
            options,
            f"using fsaverage common space for source decoding ({common_reason})...",
        )
        source_label_src, common_subjects_dir = fsaverage_source_space(common_subjects_dir)
        common_src_sig = source_space_vertices_signature(source_label_src)
    else:
        common_subject, common_subjects_dir = ordered_subject_keys[0]
        source_label_src = None
        common_src_sig = tuple()

    source_labels = []
    volume_aseg_path = ""
    volume_label_indices = []
    volume_label_names = []
    if source_space_kind == "volume":
        volume_aseg_path, volume_label_names, volume_label_indices, selected_roi_names, available_parcellations = concept_volume_labels_for_subject(
            common_subject,
            common_subjects_dir,
            requested_labels=requested_parcellations,
            parcellation=source_parcellation,
        )
    else:
        source_labels, selected_roi_names, available_parcellations, source_parcellation = concept_source_labels_for_subject(
            context,
            common_subject,
            common_subjects_dir,
            requested_parcellations=requested_parcellations,
            parcellation=source_parcellation,
        )
    if max_rois > 0:
        if source_space_kind == "surface":
            source_labels = source_labels[:max_rois]
        else:
            keep_names = set(selected_roi_names[:max_rois])
            volume_label_indices = [
                idx for idx in volume_label_indices
                if volume_label_names[idx] in keep_names
            ]
        selected_roi_names = selected_roi_names[:max_rois]
    first_raw = next(iter(raw_by_path.values()))
    estimate_sfreq = float(resample_sfreq) if resample_enabled and resample_sfreq is not None else float(first_raw.info["sfreq"])
    if source_power:
        try:
            # Use bands supported by every run; do not silently change columns per run.
            effective_sfreq = min(
                float(resample_sfreq) if resample_enabled and resample_sfreq is not None else float(raw.info["sfreq"])
                for raw in raw_by_path.values()
            )
            band_defs = resolve_source_power_bands(
                options.get("source_power_bands", "all"), effective_sfreq,
                bandpass_low_hz, bandpass_high_hz,
            )
            minimum_padding = max(0.5, 3.0 / min(band[1] for band in band_defs))
            power_padding = float(options.get("source_power_padding_s", minimum_padding))
            if not np.isfinite(power_padding) or power_padding < minimum_padding:
                raise ValueError(f"Source power padding must be at least {minimum_padding:g} seconds for these bands.")
        except (ValueError, TypeError) as exc:
            restore_meg_sensor_fif(context, restore_fif_path)
            return None, str(exc)
    estimated_bin_samples = [
        max(1, int(np.ceil((float(bin_end) - float(bin_start)) * estimate_sfreq)))
        for bin_start, bin_end in bin_edges
    ]
    if "raw" in stat_names:
        estimated_features = int(len(selected_roi_names) * sum(estimated_bin_samples))
    else:
        estimated_features = int(len(selected_roi_names) * len(bin_edges) * (len(band_defs) if source_power else len(stat_names)))
    if estimated_features > max_source_features:
        restore_meg_sensor_fif(context, restore_fif_path)
        return None, (
            "Source-space decoding would create too many features "
            f"({estimated_features:,}). Enable downsampling, shorten the window, or limit Max channels/ROIs."
        )

    feature_rows = []
    kept_rows = []
    trial_qc_rows = []
    dropped_trials = 0
    feature_names = None
    feature_meta = None
    expected_raw_sample_counts = None
    inverse_paths = {}
    parcel_baseline_sample_count = 0
    invalid_parcel_baseline_count = 0

    if source_power:
        epoch_tmin = min(window_start, parcel_baseline_start) if power_baseline else window_start
        epoch_tmax = max(window_end, parcel_baseline_end) if power_baseline else window_end
        epoch_tmin -= power_padding
        epoch_tmax += power_padding
    elif source_parcel_baseline_zscore:
        epoch_tmin = min(float(window_start), parcel_baseline_start)
        epoch_tmax = max(float(window_end), parcel_baseline_end)
    else:
        epoch_tmin = min(float(window_start), -0.1 if baseline_correct else float(window_start))
        epoch_tmax = float(window_end)
    baseline = (-0.1, 0.0) if baseline_correct and epoch_tmin <= -0.1 else None
    epoch_reject = concept_epoch_reject_from_options(options)
    reject_summary = concept_epoch_reject_summary(epoch_reject)
    exclude_qc_outliers = bool(
        options.get(
            "exclude_trial_outliers",
            CONCEPT_PREPROCESSING_DEFAULTS["exclude_trial_outliers"],
        )
    )
    try:
        trial_outlier_z = float(
            options.get(
                "trial_outlier_z",
                CONCEPT_PREPROCESSING_DEFAULTS["trial_outlier_z"],
            )
        )
    except Exception:
        trial_outlier_z = CONCEPT_PREPROCESSING_DEFAULTS["trial_outlier_z"]
    try:
        min_outlier_trials = max(4, int(options.get("min_trials_for_outlier_qc", 8)))
    except Exception:
        min_outlier_trials = 8

    grouped_files = list(df.groupby("source_fif_path", sort=False))
    run_settings = {
        "band_defs": band_defs,
        "bandpass_high_hz": bandpass_high_hz,
        "bandpass_low_hz": bandpass_low_hz,
        "baseline": baseline,
        "bin_edges": bin_edges,
        "common_src_sig": common_src_sig,
        "epoch_reject": epoch_reject,
        "epoch_tmax": epoch_tmax,
        "epoch_tmin": epoch_tmin,
        "exclude_qc_outliers": exclude_qc_outliers,
        "feature_binning": feature_binning,
        "lambda2": lambda2,
        "method": method,
        "min_outlier_trials": min_outlier_trials,
        "ordered_subject_keys": ordered_subject_keys,
        "parcel_baseline_end": parcel_baseline_end,
        "parcel_baseline_start": parcel_baseline_start,
        "power_baseline": power_baseline,
        "resample_enabled": resample_enabled,
        "resample_sfreq": resample_sfreq,
        "selected_roi_names": selected_roi_names,
        "source_batch_size": source_batch_size,
        "source_label_src": source_label_src,
        "source_labels": source_labels,
        "source_parcel_baseline_zscore": source_parcel_baseline_zscore,
        "source_power": source_power,
        "source_space_kind": source_space_kind,
        "stat_names": stat_names,
        "trial_outlier_z": trial_outlier_z,
        "use_common_space": use_common_space,
        "volume_aseg_path": volume_aseg_path,
        "volume_label_indices": volume_label_indices,
        "window_end": window_end,
        "window_start": window_start,
    }
    from core.concept_decoding_source_parallel import extract_source_runs
    run_results = extract_source_runs(context, options, raw_by_path, grouped_files, run_settings)
    for path_abs, result in run_results:
        skipped_paths.extend(result["skipped_paths"])
        inverse_paths.update(result["inverse_paths"])
        dropped_trials += result["dropped_trials"]
        if not result["feature_rows"]:
            continue
        if feature_names is not None and result["feature_names"] != feature_names:
            skipped_paths.append((path_abs, "source feature columns differ from first decoded run"))
            continue
        if (parcel_baseline_sample_count and result["parcel_baseline_sample_count"]
                and parcel_baseline_sample_count != result["parcel_baseline_sample_count"]):
            skipped_paths.append((path_abs, "Source runs produced different parcel-baseline sample counts."))
            continue
        if feature_names is None:
            feature_names = result["feature_names"]
            feature_meta = result["feature_meta"]
            expected_raw_sample_counts = result["expected_raw_sample_counts"]
        feature_rows.extend(result["feature_rows"])
        kept_rows.extend(result["kept_rows"])
        trial_qc_rows.extend(result["trial_qc_rows"])
        parcel_baseline_sample_count = result["parcel_baseline_sample_count"]
        invalid_parcel_baseline_count += result["invalid_parcel_baseline_count"]

    restore_meg_sensor_fif(context, restore_fif_path)

    if not feature_rows:
        detail = ""
        if skipped_paths:
            first_path, first_error = skipped_paths[0]
            detail = (
                f" First run error ({os.path.basename(str(first_path))}): "
                f"{first_error}"
            )
        return None, (
            "No usable trial features were extracted from source-space data."
            f"{detail}"
        )

    X = np.vstack(feature_rows).astype(np.float32, copy=False)
    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    trial_qc_df = pd.DataFrame(trial_qc_rows).reset_index(drop=True)
    if not trial_qc_df.empty:
        sort_cols = [
            col
            for col in ("source_fif_path", "trial_index_in_file", "trial_index", "onset_sample")
            if col in trial_qc_df.columns
        ]
        if sort_cols:
            trial_qc_df = trial_qc_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    y = kept_df["concept_label"].astype(str).to_numpy()
    post_counts = pd.Series(y).value_counts()
    keep_after_extract = post_counts[post_counts >= min_trials].index.tolist()
    if len(keep_after_extract) < required_class_count:
        if allow_single_class:
            return None, "No stimulus group retained enough valid source-space trials."
        return None, "Need at least two concepts with enough valid source-space trials."
    keep_mask = np.isin(y, keep_after_extract)
    if "_source_qc_uid" in kept_df.columns and "_source_qc_uid" in trial_qc_df.columns:
        low_count_uids = set(kept_df.loc[~keep_mask, "_source_qc_uid"].astype(int).tolist())
        if low_count_uids:
            low_mask = trial_qc_df["_source_qc_uid"].isin(low_count_uids)
            trial_qc_df.loc[low_mask, "qc_status"] = "dropped"
            trial_qc_df.loc[low_mask, "qc_reason"] = "too_few_trials_after_qc"
            dropped_trials += int(np.sum(~keep_mask))
    X = X[keep_mask]
    y = y[keep_mask]
    kept_df = kept_df.loc[keep_mask].reset_index(drop=True)
    if "raw" in stat_names:
        n_time_points = int(sum(expected_raw_sample_counts or []))
    else:
        n_time_points = int(len(bin_edges))
    used_paths = set(pd.Index(kept_df["source_fif_path"].astype(str)).drop_duplicates().tolist())
    used_subject_keys = []
    for path_abs in path_order:
        key = source_subject_by_path.get(path_abs)
        if path_abs in used_paths and key is not None and key not in used_subject_keys:
            used_subject_keys.append(key)
    used_native_subjects = list(dict.fromkeys([key[0] for key in used_subject_keys])) or native_subjects

    payload = {
        "X": X,
        "y": y,
        "trial_df": kept_df,
        "trial_qc_df": trial_qc_df,
        "feature_names": feature_names or [],
        "feature_meta_df": pd.DataFrame(feature_meta or []),
        "channel_names": [],
        "roi_names": selected_roi_names or [],
        "available_parcellations": available_parcellations,
        "inverse_paths": inverse_paths,
        "bin_edges": bin_edges,
        "stat_names": stat_names,
        "feature_mode": "bandpower" if source_power else str(options.get("feature_mode", "mean_rms")),
        "band_defs": band_defs,
        "n_bands": len(band_defs),
        "source_power_baseline": power_baseline,
        "source_power_padding_s": power_padding,
        "source_power_aggregation": "vertex_power_then_roi_mean" if source_power else "",
        "source_power_orientation": "component_power_sum_or_fixed_signed" if source_power else "",
        "epoch_tmin": float(epoch_tmin),
        "epoch_tmax": float(epoch_tmax),
        "n_time_points": int(n_time_points),
        "feature_binning_enabled": bool(feature_binning),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "bandpass_low_hz": bandpass_low_hz,
        "bandpass_high_hz": bandpass_high_hz,
        "shared_bandpass_low_hz": shared_bandpass_low_hz,
        "shared_bandpass_high_hz": shared_bandpass_high_hz,
        "resample_enabled": bool(resample_enabled),
        "resample_sfreq": float(resample_sfreq) if resample_enabled and resample_sfreq is not None else np.nan,
        "group_mode": group_mode,
        "channel_mode": "source",
        "feature_space": "source",
        "baseline_correct": baseline_correct,
        "source_parcel_baseline_zscore": bool(source_parcel_baseline_zscore),
        "parcel_baseline_start": float(parcel_baseline_start),
        "parcel_baseline_end": float(parcel_baseline_end),
        "parcel_baseline_samples": int(parcel_baseline_sample_count),
        "invalid_parcel_baselines": int(invalid_parcel_baseline_count),
        "mne_epoch_reject_enabled": bool(reject_summary.get("enabled", False)),
        "mne_reject_mag_ft": reject_summary.get("mag_ft", np.nan),
        "mne_reject_grad_ft_cm": reject_summary.get("grad_ft_cm", np.nan),
        "trial_outlier_z": float(trial_outlier_z),
        "exclude_trial_outliers": bool(exclude_qc_outliers),
        "method": method,
        "lambda2": float(lambda2),
        "source_space_kind": str(source_space_kind),
        "source_parcellation": str(source_parcellation),
        "volume_aseg_path": str(volume_aseg_path),
        "dropped_trials": int(dropped_trials),
        "skipped_paths": skipped_paths,
        "n_runs": int(len(pd.Index(kept_df["source_fif_path"]).drop_duplicates())),
        "subject": str(common_subject or ""),
        "subjects_dir": str(common_subjects_dir or ""),
        "source_common_space": "fsaverage" if use_common_space else "native",
        "source_subject_count": int(len(used_subject_keys) or len(ordered_subject_keys)),
        "native_subjects": used_native_subjects,
        "source_subject_by_path": {
            str(path): {"subject": key[0], "subjects_dir": key[1]}
            for path, key in source_subject_by_path.items()
        },
        "source_batch_size": int(source_batch_size),
        "matched_presentations": int(matched_presentations_before_candidate_limit),
        "presentation_candidate_limit": int(presentation_candidate_limit),
    }
    return payload, None


def _parcel_cache_float(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    return round(value, 9) if np.isfinite(value) else None


def _parcel_cache_file_identity(path):
    path_abs = os.path.abspath(str(path or ""))
    try:
        stat = os.stat(path_abs)
        return path_abs, int(stat.st_size), int(stat.st_mtime_ns)
    except Exception:
        return path_abs, None, None


def _parcel_cache_dataframe_digest(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ""
    columns = [
        col
        for col in (
            "source_fif_path",
            "onset_sample",
            "trial_index",
            "stimulus",
            "canonical_stimulus",
            "stimulus_group",
            "concept",
        )
        if col in df.columns
    ]
    if not columns:
        return ""
    normalized = df[columns].copy()
    for col in columns:
        normalized[col] = normalized[col].fillna("").astype(str)
    hashed = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:24]


def parcel_trial_timecourse_cache_key(context, options):
    """Return a stable key for an extracted parcel-time-course result."""
    options = dict(options or {})
    paths = options.get("source_fif_paths", [])
    if not isinstance(paths, (list, tuple, np.ndarray)) or not len(paths):
        current_path = str(getattr(context, "_fif_filename", "") or "").strip()
        paths = [current_path] if current_path else []
    file_identities = tuple(_parcel_cache_file_identity(path) for path in paths)

    source_space_kind = str(options.get("source_space_kind", "surface") or "surface")
    inverse_identities = []
    for path in paths:
        inverse_path = _source_inverse_output_path(context, path, source_space_kind)
        try:
            found_path = context._find_inverse_operator_for_fif(path)
        except Exception:
            found_path = None
        resolved_path = (
            inverse_path
            if inverse_path and os.path.isfile(inverse_path)
            else found_path or inverse_path or ""
        )
        inverse_identities.append(_parcel_cache_file_identity(resolved_path))

    trial_digest = _parcel_cache_dataframe_digest(
        getattr(context, "_fif_trial_response_table", pd.DataFrame())
    )
    stimulus_digest = _parcel_cache_dataframe_digest(
        getattr(context, "_fif_stimulus_summary_table", pd.DataFrame())
    )
    selected_parcellations = options.get("selected_parcellations", [])
    if not isinstance(selected_parcellations, (list, tuple, np.ndarray)):
        selected_parcellations = []

    option_fields = (
        str(options.get("selected_concept_label", "") or ""),
        str(options.get("group_mode", "canonical") or "canonical"),
        _parcel_cache_float(options.get("window_start", 0.0)),
        _parcel_cache_float(options.get("window_end", 0.4)),
        int(options.get("max_presentations", 0) or 0),
        int(options.get("_presentation_candidate_limit", 0) or 0),
        bool(options.get("normalize_parcel_baseline", True)),
        bool(options.get("source_parcel_baseline_zscore", False)),
        str(options.get("parcel_representation", "activity") or "activity"),
        _parcel_cache_float(options.get("parcel_power_padding_s", np.nan)),
        _parcel_cache_float(options.get("parcel_baseline_start", -0.2)),
        _parcel_cache_float(options.get("parcel_baseline_end", 0.0)),
        _parcel_cache_float(options.get("bandpass_low_hz", 1.0)),
        _parcel_cache_float(options.get("bandpass_high_hz", 40.0)),
        str(options.get("parcel_frequency_band", "broadband") or "broadband"),
        bool(options.get("mne_resample_enabled", True)),
        _parcel_cache_float(options.get("resample_sfreq", 200.0)),
        bool(options.get("mne_epoch_reject_enabled", True)),
        _parcel_cache_float(options.get("reject_mag_ft", 4000.0)),
        _parcel_cache_float(options.get("reject_grad_ft_cm", 4000.0)),
        bool(options.get("exclude_trial_outliers", True)),
        _parcel_cache_float(options.get("trial_outlier_z", 6.0)),
        str(options.get("method", "MNE") or "MNE"),
        _parcel_cache_float(options.get("lambda2", 1.0 / 9.0)),
        source_space_kind,
        str(options.get("source_parcellation", "aparc") or "aparc"),
        str(options.get("source_common_space", "auto") or "auto"),
        tuple(str(value) for value in selected_parcellations),
        int(options.get("max_rois", 0) or 0),
    )
    return (
        PARCEL_TIMECOURSE_CACHE_VERSION,
        file_identities,
        tuple(inverse_identities),
        trial_digest,
        stimulus_digest,
        option_fields,
    )


def _parcel_timecourse_cache(context):
    cache = getattr(context, "_parcel_timecourse_cache", None)
    if isinstance(cache, OrderedDict):
        return cache
    cache = OrderedDict()
    try:
        setattr(context, "_parcel_timecourse_cache", cache)
    except Exception:
        return None
    return cache


def clear_parcel_timecourse_cache(context):
    cache = getattr(context, "_parcel_timecourse_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _parcel_timecourse_result_bytes(result):
    total = 0
    seen = set()
    if not isinstance(result, dict):
        return 0
    for value in result.values():
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
        elif isinstance(value, pd.DataFrame):
            try:
                total += int(value.memory_usage(index=True, deep=True).sum())
            except Exception:
                pass
    return int(total)


def _parcel_timecourse_cache_lookup(context, key):
    cache = _parcel_timecourse_cache(context)
    if cache is None or key not in cache:
        return None
    entry = cache.pop(key)
    cache[key] = entry
    return entry


def _parcel_timecourse_cache_store(context, key, result, message, options):
    cache = _parcel_timecourse_cache(context)
    if cache is None:
        return 0
    try:
        max_entries = max(1, int((options or {}).get("parcel_timecourse_cache_max_entries", 8)))
    except Exception:
        max_entries = 8
    try:
        max_bytes = max(
            1,
            int(float((options or {}).get("parcel_timecourse_cache_max_mb", 512.0)) * 1024 * 1024),
        )
    except Exception:
        max_bytes = 512 * 1024 * 1024
    entry_bytes = _parcel_timecourse_result_bytes(result)
    if entry_bytes > max_bytes:
        return 0
    if key in cache:
        cache.pop(key)
    cache[key] = {
        "result": result,
        "message": str(message),
        "bytes": int(entry_bytes),
    }
    total_bytes = sum(int(entry.get("bytes", 0) or 0) for entry in cache.values())
    while len(cache) > max_entries or total_bytes > max_bytes:
        _old_key, old_entry = cache.popitem(last=False)
        total_bytes -= int(old_entry.get("bytes", 0) or 0)
    return int(entry_bytes)


def parcel_trial_candidate_pool_size(max_presentations):
    """Bound source work while retaining spare candidates for rejection/QC."""
    try:
        requested = max(0, int(max_presentations))
    except Exception:
        requested = 0
    if requested <= 0:
        return 0
    rejection_reserve = max(2, int(np.ceil(requested * 0.25)))
    return int(requested + rejection_reserve)


def build_source_parcel_trial_time_courses(context, options=None):
    """Return trial-by-parcel time courses for one selected stimulus group.

    This reuses the source-decoding extraction path so inverse operators,
    run-specific anatomy, baseline handling, optional common-space morphing,
    and trial-to-events-log alignment stay identical across the UI.
    """
    options = dict(options or {})
    # These are also the UI defaults for source-parcel concept decoding. Keep
    # direct/non-UI callers on the same preprocessing profile.
    for key, value in CONCEPT_PREPROCESSING_DEFAULTS.items():
        options.setdefault(key, value)
    representation = str(
        options.get("parcel_representation", "activity") or "activity"
    ).strip().lower()
    if representation not in {"activity", "band_power_db"}:
        return None, f"Unknown Parcel trials representation: {representation}."
    normalize_parcel_baseline = bool(options.get("normalize_parcel_baseline", True))
    try:
        parcel_baseline_start = float(options.get("parcel_baseline_start", -0.2))
        parcel_baseline_end = float(options.get("parcel_baseline_end", 0.0))
    except Exception:
        return None, "Invalid parcel prestimulus baseline window."
    try:
        display_window_start = float(options.get("window_start", 0.0))
        display_window_end = float(options.get("window_end", 0.4))
    except Exception:
        return None, "Invalid parcel display window."
    if (
        not np.isfinite(display_window_start)
        or not np.isfinite(display_window_end)
        or display_window_end <= display_window_start
    ):
        return None, "Parcel display window end must be greater than its start."

    # Activity z-scoring happens inside the shared extraction path. Band power
    # instead needs unnormalized waveforms plus baseline and edge padding; it
    # is converted to dB after source reconstruction.
    options["baseline_correct"] = False
    options["source_parcel_baseline_zscore"] = (
        normalize_parcel_baseline and representation == "activity"
    )
    if representation == "band_power_db":
        if parcel_baseline_end > 0.0:
            return None, (
                "Parcel band-power baseline must end at or before stimulus onset (0 ms)."
            )
        band_key = str(
            options.get("parcel_frequency_band", "broadband") or "broadband"
        ).strip().lower()
        if band_key == "broadband":
            return None, (
                "Hilbert band power requires a named frequency band. Select delta, "
                "theta, alpha, beta, low gamma, or high gamma."
            )
        try:
            shared_low, shared_high = concept_bandpass_from_options(options)
            resample_enabled, resample_sfreq = concept_resample_from_options(options)
            band_low, _band_high, resolved_band_label = (
                parcel_trial_frequency_band_settings(
                    band_key,
                    shared_low,
                    shared_high,
                    resample_enabled=resample_enabled,
                    resample_sfreq=resample_sfreq,
                )
            )
            power_padding_s = float(
                options.get(
                    "parcel_power_padding_s",
                    parcel_power_hilbert_padding_s(band_low),
                )
            )
        except ValueError as exc:
            return None, str(exc)
        if not np.isfinite(power_padding_s) or power_padding_s < 0.0:
            return None, "Parcel power padding must be a finite non-negative duration."
        options["parcel_frequency_band_label"] = str(
            options.get("parcel_frequency_band_label", "") or resolved_band_label
        )
        options["parcel_power_padding_s"] = power_padding_s
        options["window_start"] = min(
            display_window_start, parcel_baseline_start
        ) - power_padding_s
        options["window_end"] = max(
            display_window_end, parcel_baseline_end
        ) + power_padding_s
        options["max_source_features"] = max(
            500000, int(options.get("max_source_features", 500000) or 500000)
        )
    # Do not select the first N candidates before rejection/QC: a rejected
    # candidate would otherwise leave fewer than N displayed presentations
    # even when later usable presentations exist.
    options["_limit_presentations_after_qc"] = True
    try:
        requested_presentations = max(0, int(options.get("max_presentations", 0)))
    except Exception:
        requested_presentations = 0
    if requested_presentations > 0:
        options["_presentation_candidate_limit"] = parcel_trial_candidate_pool_size(
            requested_presentations
        )
    options.update(
        {
            "feature_mode": "raw",
            "feature_binning_enabled": False,
            "min_trials_per_class": 1,
            "_allow_single_class": True,
            "_use_stimulus_ui_labels": True,
        }
    )
    cache_enabled = bool(options.get("enable_parcel_timecourse_cache", True))
    cache_key = parcel_trial_timecourse_cache_key(context, options)
    if cache_enabled:
        cached_entry = _parcel_timecourse_cache_lookup(context, cache_key)
        if isinstance(cached_entry, dict) and isinstance(cached_entry.get("result"), dict):
            cached_result = dict(cached_entry["result"])
            cached_result["parcel_cache_hit"] = True
            cached_result["parcel_cache_entry_bytes"] = int(
                cached_entry.get("bytes", 0) or 0
            )
            emit_concept_progress(
                options,
                "using cached preprocessed parcel time courses",
            )
            cached_message = str(cached_entry.get("message", "") or "")
            return cached_result, (
                "Reused cached preprocessed parcel time courses. " + cached_message
            ).strip()

    payload, err_msg = build_source_space_concept_features(context, options)
    if payload is None:
        return None, err_msg or "Could not extract parcel time courses."

    X = np.asarray(payload.get("X", []), dtype=np.float32)
    roi_names = [str(v) for v in payload.get("roi_names", [])]
    feature_meta = payload.get("feature_meta_df", pd.DataFrame())
    if X.ndim != 2 or X.shape[0] == 0 or not roi_names:
        return None, "Source extraction returned no trial-by-parcel samples."
    if not isinstance(feature_meta, pd.DataFrame) or feature_meta.empty:
        return None, "Source extraction did not return a parcel time axis."

    if "roi" not in feature_meta.columns:
        return None, "Source extraction did not return parcel identities for raw samples."
    first_roi = roi_names[0]
    first_roi_meta = feature_meta[feature_meta["roi"].astype(str) == first_roi].copy()
    if first_roi_meta.empty or "relative_time_s" not in first_roi_meta.columns:
        return None, "Source extraction did not return raw parcel sample times."
    times = pd.to_numeric(first_roi_meta["relative_time_s"], errors="coerce").to_numpy(dtype=float)
    times = times[np.isfinite(times)]
    n_times = int(times.size)
    expected_features = int(len(roi_names) * n_times)
    if n_times == 0 or X.shape[1] != expected_features:
        return None, (
            "Parcel time-course shape is inconsistent: "
            f"features={X.shape[1]}, parcels={len(roi_names)}, samples={n_times}."
        )

    available_presentations = int(X.shape[0])
    try:
        max_presentations = max(0, int(options.get("max_presentations", 0)))
    except Exception:
        max_presentations = 0
    if max_presentations > 0 and X.shape[0] > max_presentations:
        X = X[:max_presentations]
    data = X.reshape(X.shape[0], len(roi_names), n_times)
    baseline_sample_count = int(payload.get("parcel_baseline_samples", 0) or 0)
    invalid_baseline_count = int(payload.get("invalid_parcel_baselines", 0) or 0)
    if representation == "band_power_db":
        try:
            data, times, baseline_sample_count, invalid_baseline_count = (
                baseline_db_hilbert_parcel_power(
                    data,
                    times,
                    baseline_start=parcel_baseline_start,
                    baseline_end=parcel_baseline_end,
                    display_start=display_window_start,
                    display_end=display_window_end,
                )
            )
        except ValueError as exc:
            return None, str(exc)
    trial_df = payload.get("trial_df", pd.DataFrame())
    if not isinstance(trial_df, pd.DataFrame):
        trial_df = pd.DataFrame()
    if len(trial_df) > data.shape[0]:
        trial_df = trial_df.iloc[: data.shape[0]].copy().reset_index(drop=True)
    displayed_run_count = int(payload.get("n_runs", 0) or 0)
    if not trial_df.empty and "source_fif_path" in trial_df.columns:
        displayed_run_count = int(
            trial_df["source_fif_path"].astype(str).replace("", np.nan).dropna().nunique()
        )
    result = {
        "data": data,
        "times": times,
        "roi_names": roi_names,
        "trial_df": trial_df.copy().reset_index(drop=True),
        "stimulus": str(options.get("selected_concept_label", "All stimuli") or "All stimuli"),
        "group_mode": str(payload.get("group_mode", options.get("group_mode", "canonical"))),
        "source_space_kind": str(payload.get("source_space_kind", "surface")),
        "source_parcellation": str(payload.get("source_parcellation", options.get("source_parcellation", "aparc"))),
        "source_common_space": str(payload.get("source_common_space", "native")),
        "method": str(payload.get("method", options.get("method", "MNE"))),
        "normalization": (
            "parcel_power_db"
            if representation == "band_power_db"
            else "parcel_baseline_zscore"
            if normalize_parcel_baseline
            else "none"
        ),
        "parcel_representation": representation,
        "parcel_power_padding_s": float(options.get("parcel_power_padding_s", 0.0) or 0.0),
        "window_start": display_window_start,
        "window_end": display_window_end,
        "parcel_baseline_start": parcel_baseline_start,
        "parcel_baseline_end": parcel_baseline_end,
        "parcel_baseline_samples": int(baseline_sample_count),
        "invalid_parcel_baselines": int(invalid_baseline_count),
        "bandpass_low_hz": payload.get("bandpass_low_hz", np.nan),
        "bandpass_high_hz": payload.get("bandpass_high_hz", np.nan),
        "shared_bandpass_low_hz": payload.get(
            "shared_bandpass_low_hz", payload.get("bandpass_low_hz", np.nan)
        ),
        "shared_bandpass_high_hz": payload.get(
            "shared_bandpass_high_hz", payload.get("bandpass_high_hz", np.nan)
        ),
        "frequency_band": str(
            options.get("parcel_frequency_band", "broadband") or "broadband"
        ),
        "frequency_band_label": str(
            options.get("parcel_frequency_band_label", "") or ""
        ),
        "resample_enabled": bool(payload.get("resample_enabled", False)),
        "resample_sfreq": payload.get("resample_sfreq", np.nan),
        "mne_epoch_reject_enabled": bool(payload.get("mne_epoch_reject_enabled", False)),
        "mne_reject_mag_ft": payload.get("mne_reject_mag_ft", np.nan),
        "mne_reject_grad_ft_cm": payload.get("mne_reject_grad_ft_cm", np.nan),
        "exclude_trial_outliers": bool(payload.get("exclude_trial_outliers", False)),
        "trial_outlier_z": payload.get("trial_outlier_z", np.nan),
        "n_runs": displayed_run_count,
        "available_presentations": available_presentations,
        "matched_presentations": int(
            payload.get("matched_presentations", available_presentations) or 0
        ),
        "presentation_candidate_limit": int(
            payload.get("presentation_candidate_limit", 0) or 0
        ),
        "max_presentations": max_presentations,
        "dropped_trials": int(payload.get("dropped_trials", 0) or 0),
        "skipped_paths": list(payload.get("skipped_paths", []) or []),
        "parcel_cache_hit": False,
        "parcel_cache_entry_bytes": 0,
    }
    message = (
        f"Extracted {data.shape[0]} presentation(s), {data.shape[1]} parcels, "
        f"and {data.shape[2]} samples from {result['n_runs']} run(s)."
    )
    if result["dropped_trials"]:
        message += f" Dropped {result['dropped_trials']} invalid/rejected presentation(s)."
    if max_presentations > 0 and available_presentations > data.shape[0]:
        message += (
            f" Showing {data.shape[0]} of {available_presentations} usable presentation(s)."
        )
    elif max_presentations > 0 and data.shape[0] < max_presentations:
        message += (
            f" Only {data.shape[0]} usable presentation(s) were available out of the "
            f"requested maximum of {max_presentations}."
        )
    candidate_limit = int(result.get("presentation_candidate_limit", 0) or 0)
    matched_presentations = int(result.get("matched_presentations", 0) or 0)
    if candidate_limit > 0 and matched_presentations > candidate_limit:
        message += (
            f" QC-screened the first {candidate_limit} of {matched_presentations} matched "
            "presentations to avoid unnecessary source reconstruction."
        )
    low = result["bandpass_low_hz"]
    high = result["bandpass_high_hz"]
    try:
        low_value = float(low)
    except (TypeError, ValueError):
        low_value = np.nan
    try:
        high_value = float(high)
    except (TypeError, ValueError):
        high_value = np.nan
    low_label = "open" if not np.isfinite(low_value) else f"{low_value:g}"
    high_label = "open" if not np.isfinite(high_value) else f"{high_value:g}"
    band_label = str(result.get("frequency_band_label", "") or "").strip()
    if band_label:
        message += f" Frequency band: {band_label}."
    else:
        message += f" Bandpass {low_label}-{high_label} Hz."
    try:
        resample_value = float(result["resample_sfreq"])
    except (TypeError, ValueError):
        resample_value = np.nan
    if result["resample_enabled"] and np.isfinite(resample_value):
        message += f" Resampled to {resample_value:g} Hz."
    if result["mne_epoch_reject_enabled"]:
        message += " MNE epoch rejection enabled."
    if result["exclude_trial_outliers"]:
        try:
            outlier_z = float(result["trial_outlier_z"])
        except (TypeError, ValueError):
            outlier_z = np.nan
        if np.isfinite(outlier_z):
            message += f" Robust trial QC enabled (z>{outlier_z:g})."
        else:
            message += " Robust trial QC enabled."
    if representation == "band_power_db":
        message += (
            f" Hilbert power expressed in dB relative to each parcel's "
            f"{parcel_baseline_start * 1000.0:g}–{parcel_baseline_end * 1000.0:g} ms "
            f"baseline ({baseline_sample_count} samples)."
        )
        if invalid_baseline_count:
            message += f" Zeroed {invalid_baseline_count} zero/invalid power references."
    elif normalize_parcel_baseline:
        message += (
            f" Parcel-wise z-scored to {parcel_baseline_start * 1000.0:g}–"
            f"{parcel_baseline_end * 1000.0:g} ms baseline "
            f"({baseline_sample_count} samples)."
        )
        if invalid_baseline_count:
            message += f" Zeroed {invalid_baseline_count} zero/invalid parcel baselines."
    if cache_enabled:
        # An inverse operator may have been created during the first extraction,
        # so recompute the key before storing to include its final file identity.
        cache_key = parcel_trial_timecourse_cache_key(context, options)
        result["parcel_cache_entry_bytes"] = _parcel_timecourse_cache_store(
            context,
            cache_key,
            result,
            message,
            options,
        )
    return result, message


def compute_source_space_concept_decoding(context, options=None):
    """Train/test concept decoder in source ROI space."""
    if options is None:
        options = {}
    if not getattr(context, "b_fif_loaded", False):
        return None, "No FIF loaded. Please load a FIF file before running concept decoding."
    if options.get("semantic_subset_batch", False):
        from core.concept_decoding_subsets import run_semantic_subset_batch
        return run_semantic_subset_batch(context, options)
    from core.concept_decoding_transformer import (
        compute_transformer_decoding_from_payload,
        is_semantic_embedding_decoder,
        is_source_space_semantic_decoder,
        prepare_semantic_embedding_feature_payload,
    )

    decoder_type = str(options.get("decoder_type", "logistic") or "logistic")
    is_zs = (
        str(decoder_type).strip().lower() in {
            "zero_shot",
            "zero_shot_decoding",
            "zero_shot_embedding",
            "zero_shot_ridge",
            "zero_shot_linear",
            "zero_shot_contrastive",
            "zero_shot_mlp",
            "zeroshot",
        }
        or bool(options.get("zero_shot", False))
        or str(options.get("split_mode", "")).strip().lower() in {"zero_shot", "unseen_concepts"}
    )
    semantic_decoder = is_semantic_embedding_decoder(decoder_type) or is_zs
    source_semantic_decoder = is_source_space_semantic_decoder(decoder_type) or is_zs
    if semantic_decoder and not source_semantic_decoder:
        return None, "The selected semantic decoder currently requires signal-space features."
    requested_split = str(options.get("split_mode", "random") or "random").strip().lower()
    if semantic_decoder and not is_zs and requested_split != "random":
        return None, "Semantic embedding decoders currently support 'random' and 'zero_shot' splits."
    if not semantic_decoder:
        split_err = validate_grouped_split(context, options)
        if split_err:
            return None, split_err
    feature_payload, err_msg = build_source_space_concept_features(context, options)
    if feature_payload is None:
        return None, err_msg or "Could not build source-space concept features."
    if decoder_type in {"clip_generation", "clip_generation_hybrid"}:
        from core.concept_image_generation import compute_generation_decoding
        return compute_generation_decoding(feature_payload, options)
    if source_semantic_decoder:
        feature_payload, err_msg = prepare_semantic_embedding_feature_payload(
            feature_payload, options
        )
        if feature_payload is None:
            return None, err_msg or "Could not align source-space semantic embeddings."
        emit_concept_progress(
            options, "training/testing source-space semantic embedding decoder..."
        )
        return compute_transformer_decoding_from_payload(
            context, options, feature_payload
        )
    emit_concept_progress(options, "training/testing source-space decoder...")
    return compute_concept_decoding_from_payload(context, options, feature_payload, "source")
