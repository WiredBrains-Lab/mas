"""Source-estimate parcellation helpers.

These utilities keep atlas loading and label-time-course extraction out of the
Qt widgets. They intentionally avoid requiring a surface source space for
surface labels by intersecting label vertices directly with STC vertices.
"""

from pathlib import Path
import re

import mne
import numpy as np


HCP_MMP_PARCS = {"HCPMMP1", "HCPMMP1_combined"}
SURFACE_PARCS = {"aparc", *HCP_MMP_PARCS}
VOLUME_PARCS = {"aparc+aseg"}


_PARCEL_REGION_ORDER = (
    "Occipital / visual",
    "Parietal",
    "Temporal / auditory",
    "Sensorimotor",
    "Frontal / insular",
    "Cingulate / medial",
    "Subcortical",
    "Cerebellum",
    "White matter",
    "Ventricular / CSF",
    "Other",
)

_APARC_REGIONS = {
    "Occipital / visual": {
        "cuneus",
        "lateraloccipital",
        "lingual",
        "pericalcarine",
    },
    "Parietal": {
        "inferiorparietal",
        "precuneus",
        "superiorparietal",
        "supramarginal",
    },
    "Temporal / auditory": {
        "bankssts",
        "entorhinal",
        "fusiform",
        "inferiortemporal",
        "middletemporal",
        "parahippocampal",
        "superiortemporal",
        "temporalpole",
        "transversetemporal",
    },
    "Sensorimotor": {"paracentral", "postcentral", "precentral"},
    "Frontal / insular": {
        "caudalmiddlefrontal",
        "frontalpole",
        "insula",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "parsopercularis",
        "parsorbitalis",
        "parstriangularis",
        "rostralmiddlefrontal",
        "superiorfrontal",
    },
    "Cingulate / medial": {
        "caudalanteriorcingulate",
        "isthmuscingulate",
        "posteriorcingulate",
        "rostralanteriorcingulate",
    },
}

_APARC_VISUAL_PARCELS = {
    "cuneus",
    "lateraloccipital",
    "lingual",
    "pericalcarine",
    "fusiform",
    "inferiortemporal",
}

_APARC_SEMANTIC_PARCELS = {
    "temporalpole",
    "inferiortemporal",
    "middletemporal",
    "superiortemporal",
    "bankssts",
    "fusiform",
    "parahippocampal",
    "entorhinal",
    "inferiorparietal",
    "supramarginal",
    "precuneus",
    "parstriangularis",
    "parsorbitalis",
    "parsopercularis",
    "rostralmiddlefrontal",
    "posteriorcingulate",
    "isthmuscingulate",
}

_VOLUME_SEMANTIC_PARCELS = _APARC_SEMANTIC_PARCELS | {
    "hippocampus",
    "amygdala",
}


def _parcel_hemisphere(parcel_name):
    """Return a stable anatomical hemisphere label from common atlas names."""
    text = str(parcel_name or "").strip()
    lower = text.lower()
    if (
        lower.startswith(("ctx-lh-", "wm-lh-", "lh-", "lh_", "l_", "left-"))
        or lower.endswith(("-lh", "_lh"))
    ):
        return "Left"
    if (
        lower.startswith(("ctx-rh-", "wm-rh-", "rh-", "rh_", "r_", "right-"))
        or lower.endswith(("-rh", "_rh"))
    ):
        return "Right"
    if any(token in lower for token in ("corpus-callosum", "brain-stem", "3rd-ventricle", "4th-ventricle", "csf")):
        return "Bilateral"
    return "Other"


def _parcel_base_name(parcel_name):
    text = str(parcel_name or "").strip()
    lower = text.lower()
    for prefix in ("ctx-lh-", "ctx-rh-", "wm-lh-", "wm-rh-", "left-", "right-", "lh-", "rh-"):
        if lower.startswith(prefix):
            text = text[len(prefix):]
            lower = text.lower()
            break
    text = re.sub(r"(?:-lh|-rh|_lh|_rh)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[LR]_", "", text)
    text = re.sub(r"_ROI$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _aparc_region(parcel_name):
    base = re.sub(r"[^a-z0-9]+", "", _parcel_base_name(parcel_name).lower())
    for region, names in _APARC_REGIONS.items():
        if base in names:
            return region
    return "Other"


def _hcp_region(parcel_name):
    """Map HCP-MMP individual or combined labels to broad cortical systems."""
    base = _parcel_base_name(parcel_name).lower()
    compact = re.sub(r"[^a-z0-9]+", "", base)

    if "visual" in base or re.match(
        r"^(v[1-8]|vmv|vvc|pit|ffc|v3cd|lo[123]|v4t|fst|mt|mst|ph$|ips1)",
        compact,
    ):
        return "Occipital / visual"
    if any(token in base for token in ("auditory", "temporal")) or re.match(
        r"^(a[145]|[lmp]belt|ri$|sts|stg|ta2|te[12]|tg[vd]|tf$|h$|pres|ec$|peec|pha|pht)",
        compact,
    ):
        return "Temporal / auditory"
    if "parietal" in base or "temporo-parieto" in base or re.match(
        r"^(lip|vip|aip|mip|7|pg|pf|ip[0-2]|tpoj|stv|psl)", compact
    ):
        return "Parietal"
    if any(token in base for token in ("somatosensory", "motor", "premotor", "paracentral", "posterior opercular")) or re.match(
        r"^(4$|3[ab]$|1$|2$|5|6|scef|fef|pef|55b|43$|op|fop1$|pfcm$)",
        compact,
    ):
        return "Sensorimotor"
    if "cingulate" in base or "medial prefrontal" in base or re.match(
        r"^(2[34]|3[123]|rsc|pos|dvt|pros|pcv|8bm|9m|10[vr]|25$)", compact
    ):
        return "Cingulate / medial"
    if compact in {"", "unknown"} or "???" in base:
        return "Other"
    return "Frontal / insular"


def _volume_region(parcel_name):
    lower = str(parcel_name or "").lower()
    if lower.startswith("ctx-"):
        return _aparc_region(parcel_name)
    if any(token in lower for token in ("ventricle", "choroid", "csf")):
        return "Ventricular / CSF"
    if "white-matter" in lower or lower.startswith("wm-") or "corpus-callosum" in lower:
        return "White matter"
    if "cerebell" in lower:
        return "Cerebellum"
    if any(
        token in lower
        for token in (
            "thalam",
            "caudate",
            "putamen",
            "pallid",
            "hippocamp",
            "amygdala",
            "accumbens",
            "ventraldc",
            "ventral-dc",
        )
    ):
        return "Subcortical"
    if "cortex" in lower:
        return "Other"
    return "Other"


def is_visual_parcel(parcel_name, parcellation="aparc"):
    """Return True if parcel is part of visual or ventral visual cortex."""
    base = re.sub(r"[^a-z0-9]+", "", _parcel_base_name(parcel_name).lower())
    parc = normalize_parcellation_key(parcellation)
    if parc in {"aparc", "aparc+aseg", *VOLUME_PARCS}:
        return base in _APARC_VISUAL_PARCELS
    if parc in HCP_MMP_PARCS:
        if _hcp_region(parcel_name) == "Occipital / visual":
            return True
        return bool(re.match(r"^(v[1-8]|vmv|vvc|pit|ffc|v3cd|lo[123]|v4t|fst|mt|mst|ph$|ips1)", base))
    return base in _APARC_VISUAL_PARCELS or "visual" in base or "occipital" in base


def is_semantic_parcel(parcel_name, parcellation="aparc"):
    """Return True if parcel is part of the core semantic network."""
    base = re.sub(r"[^a-z0-9]+", "", _parcel_base_name(parcel_name).lower())
    parc = normalize_parcellation_key(parcellation)
    if parc == "aparc":
        return base in _APARC_SEMANTIC_PARCELS
    if parc in {"aparc+aseg", *VOLUME_PARCS}:
        return base in _VOLUME_SEMANTIC_PARCELS
    if parc in HCP_MMP_PARCS:
        if base in {"hippocampus", "amygdala", "inferiorfrontalcortex", "posteriorcingulatecortex"}:
            return True
        region = _hcp_region(parcel_name)
        if region in {"Temporal / auditory", "Parietal"}:
            return True
        if any(token in base for token in ("44", "45", "47", "ifj", "ifs", "sts", "te1", "te2", "tg", "pgi", "pgs", "rsc", "pos1", "pos2", "23", "31", "7m", "dvt", "pcv", "pros")):
            return True
        return False
    return base in _APARC_SEMANTIC_PARCELS or any(token in base for token in ("temporal", "parietal", "frontal"))


def anatomical_parcel_row_order(roi_names, parcellation="aparc"):
    """Order parcels by hemisphere then broad anatomical/functional region.

    Returns
    -------
    order : ndarray of int
        Indices that reorder the input parcel axis.
    groups : list of dict
        Contiguous row groups with ``start`` (inclusive), ``stop`` (exclusive),
        ``hemisphere``, ``region``, and a compact display ``label``.
    """
    names = [str(value) for value in roi_names]
    parc = normalize_parcellation_key(parcellation)
    hemisphere_rank = {"Left": 0, "Right": 1, "Bilateral": 2, "Other": 3}
    region_rank = {name: index for index, name in enumerate(_PARCEL_REGION_ORDER)}

    rows = []
    for index, name in enumerate(names):
        hemisphere = _parcel_hemisphere(name)
        if parc == "aparc":
            region = _aparc_region(name)
        elif parc in HCP_MMP_PARCS:
            region = _hcp_region(name)
        elif parc == "aparc+aseg":
            region = _volume_region(name)
        else:
            region = "Other"
        rows.append((index, hemisphere, region, name))

    rows.sort(
        key=lambda row: (
            hemisphere_rank.get(row[1], 99),
            region_rank.get(row[2], 99),
            row[3].lower(),
            row[0],
        )
    )
    order = np.asarray([row[0] for row in rows], dtype=int)
    groups = []
    start = 0
    while start < len(rows):
        hemisphere, region = rows[start][1], rows[start][2]
        stop = start + 1
        while stop < len(rows) and rows[stop][1:3] == (hemisphere, region):
            stop += 1
        hemi_label = {
            "Left": "LH",
            "Right": "RH",
            "Bilateral": "Bilateral",
            "Other": "Other",
        }.get(hemisphere, hemisphere)
        groups.append(
            {
                "start": int(start),
                "stop": int(stop),
                "hemisphere": hemisphere,
                "region": region,
                "label": f"{hemi_label} › {region} ({stop - start})",
            }
        )
        start = stop
    return order, groups


def temporal_bin_parcel_time_courses(data, times, bin_width_ms):
    """Average trial-by-parcel samples into wider display-only time bins.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_parcels, n_times)
        Parcel time courses.
    times : ndarray, shape (n_times,)
        Sample times in seconds.
    bin_width_ms : float
        Requested display-bin width in milliseconds.

    Returns
    -------
    binned_data, bin_centers, extent_ms, effective_bin_ms
        The original samples are returned unchanged when the requested width
        is no wider than their native interval.
    """
    values = np.asarray(data, dtype=float)
    sample_times = np.asarray(times, dtype=float)
    if values.ndim != 3 or sample_times.ndim != 1 or values.shape[2] != sample_times.size:
        raise ValueError("Parcel data and time-axis dimensions are inconsistent.")
    if sample_times.size == 0 or not np.all(np.isfinite(sample_times)):
        raise ValueError("Parcel time axis is empty or invalid.")
    if sample_times.size == 1:
        requested = max(0.0, float(bin_width_ms))
        time_ms = float(sample_times[0] * 1000.0)
        return values, sample_times, (time_ms, time_ms), requested

    intervals = np.diff(sample_times)
    if np.any(~np.isfinite(intervals)) or np.any(intervals <= 0.0):
        raise ValueError("Parcel time axis must be finite and strictly increasing.")
    native_bin_ms = float(np.nanmedian(intervals) * 1000.0)
    try:
        requested_bin_ms = float(bin_width_ms)
    except Exception as exc:
        raise ValueError("Display bin must be a valid number of milliseconds.") from exc
    if not np.isfinite(requested_bin_ms) or requested_bin_ms <= 0.0:
        raise ValueError("Display bin must be greater than zero milliseconds.")

    start = float(sample_times[0])
    stop = float(sample_times[-1])
    extent_ms = (start * 1000.0, stop * 1000.0)
    if requested_bin_ms <= native_bin_ms * 1.01:
        return values, sample_times, extent_ms, native_bin_ms

    bin_width_s = requested_bin_ms / 1000.0
    duration = stop - start
    n_bins = max(1, int(np.ceil(duration / bin_width_s - 1e-9)))
    binned = []
    centers = []
    for bin_index in range(n_bins):
        lower = start + bin_index * bin_width_s
        upper = min(stop, lower + bin_width_s)
        mask = (sample_times >= lower) & (sample_times < upper)
        if not np.any(mask):
            continue
        binned.append(np.nanmean(values[:, :, mask], axis=2))
        centers.append((lower + upper) / 2.0)
    if not binned:
        raise ValueError("Display bins did not contain any parcel samples.")
    return (
        np.stack(binned, axis=2),
        np.asarray(centers, dtype=float),
        extent_ms,
        requested_bin_ms,
    )


def parcel_trial_summary_maps(data):
    """Return cross-presentation signed mean and dominant-sign consistency.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_parcels, n_times)
        Signed parcel time courses, normally after display-only time binning.

    Returns
    -------
    signed_mean, sign_consistency : ndarray, shape (n_parcels, n_times)
        ``sign_consistency`` is the fraction of non-zero finite presentations
        sharing the dominant sign. Cells that are zero in every presentation
        are returned as NaN because they have no response sign.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("Parcel summary data must contain one or more presentations.")

    finite = np.isfinite(values)
    finite_count = finite.sum(axis=0)
    signed_sum = np.where(finite, values, 0.0).sum(axis=0)
    signed_mean = np.full(values.shape[1:], np.nan, dtype=float)
    np.divide(signed_sum, finite_count, out=signed_mean, where=finite_count > 0)

    positive_count = (finite & (values > 0.0)).sum(axis=0)
    negative_count = (finite & (values < 0.0)).sum(axis=0)
    signed_count = positive_count + negative_count
    dominant_count = np.maximum(positive_count, negative_count)
    sign_consistency = np.full(values.shape[1:], np.nan, dtype=float)
    has_sign = signed_count > 0
    np.divide(
        dominant_count,
        signed_count,
        out=sign_consistency,
        where=has_sign,
    )
    return signed_mean, sign_consistency


def parcel_trial_contrast_map(primary_data, comparison_data):
    """Return the signed primary-minus-comparison cross-presentation mean."""
    primary = np.asarray(primary_data, dtype=float)
    comparison = np.asarray(comparison_data, dtype=float)
    if primary.ndim != 3 or comparison.ndim != 3:
        raise ValueError("Parcel contrast inputs must be trial-by-parcel-by-time arrays.")
    if primary.shape[0] == 0 or comparison.shape[0] == 0:
        raise ValueError("Parcel contrast inputs must contain presentations from both groups.")
    if primary.shape[1:] != comparison.shape[1:]:
        raise ValueError("Parcel contrast inputs must have matching parcel and time dimensions.")
    return np.nanmean(primary, axis=0) - np.nanmean(comparison, axis=0)


def _dominant_sign_agreement(values):
    """Return dominant non-zero sign count, eligible count, and fraction."""
    samples = np.asarray(values, dtype=float)
    positive = int(np.sum(np.isfinite(samples) & (samples > 0.0)))
    negative = int(np.sum(np.isfinite(samples) & (samples < 0.0)))
    total = positive + negative
    if total == 0:
        return 0, 0, np.nan
    count = max(positive, negative)
    return count, total, float(count / total)


def _nested_loo_single_parcel_balanced_accuracy(primary, comparison):
    """Score one parcel while selecting its time bin inside every LOO fold."""
    group_a = np.asarray(primary, dtype=float)
    group_b = np.asarray(comparison, dtype=float)
    if (
        group_a.ndim != 2
        or group_b.ndim != 2
        or group_a.shape[1] != group_b.shape[1]
        or group_a.shape[0] < 2
        or group_b.shape[0] < 2
    ):
        return np.nan

    recalls = []
    for held_group, held_values in ((0, group_a), (1, group_b)):
        correct = 0
        evaluated = 0
        for held_index in range(held_values.shape[0]):
            train_a = (
                np.delete(group_a, held_index, axis=0)
                if held_group == 0
                else group_a
            )
            train_b = (
                np.delete(group_b, held_index, axis=0)
                if held_group == 1
                else group_b
            )
            with np.errstate(invalid="ignore"):
                centroid_a_by_time = np.nanmean(train_a, axis=0)
                centroid_b_by_time = np.nanmean(train_b, axis=0)
            separation = np.abs(centroid_a_by_time - centroid_b_by_time)
            valid_bins = np.isfinite(separation)
            if not np.any(valid_bins):
                continue
            candidate_bins = np.flatnonzero(valid_bins)
            selected_bin = int(candidate_bins[np.argmax(separation[valid_bins])])
            held_value = float(held_values[held_index, selected_bin])
            centroid_a = float(centroid_a_by_time[selected_bin])
            centroid_b = float(centroid_b_by_time[selected_bin])
            if not np.all(np.isfinite([held_value, centroid_a, centroid_b])):
                continue
            distance_a = abs(held_value - centroid_a)
            distance_b = abs(held_value - centroid_b)
            predicted_group = 0 if distance_a <= distance_b else 1
            correct += int(predicted_group == held_group)
            evaluated += 1
        if evaluated == 0:
            return np.nan
        recalls.append(correct / float(evaluated))
    return float(np.mean(recalls))


def rank_parcel_contrast_features(
    primary_data,
    comparison_data,
    roi_names,
    bin_centers,
    bin_width_ms,
    extent_ms=None,
    primary_label="Primary",
    comparison_label="Comparison",
):
    """Rank parcels by leakage-safe single-parcel decoding performance.

    The displayed peak interval and contrast are descriptive full-data
    summaries. The balanced accuracy is nested leave-one-presentation-out:
    each fold selects its strongest time bin using training presentations only.
    """
    primary = np.asarray(primary_data, dtype=float)
    comparison = np.asarray(comparison_data, dtype=float)
    names = [str(value) for value in roi_names]
    centers = np.asarray(bin_centers, dtype=float)
    if primary.ndim != 3 or comparison.ndim != 3:
        raise ValueError("Parcel ranking inputs must be trial-by-parcel-by-time arrays.")
    if primary.shape[0] == 0 or comparison.shape[0] == 0:
        raise ValueError("Parcel ranking requires presentations from both groups.")
    if primary.shape[1:] != comparison.shape[1:]:
        raise ValueError("Parcel ranking groups must use matching parcel and time dimensions.")
    if len(names) != primary.shape[1] or centers.size != primary.shape[2]:
        raise ValueError("Parcel ranking labels or time centers do not match the data.")
    width_ms = float(bin_width_ms)
    if not np.isfinite(width_ms) or width_ms <= 0.0:
        raise ValueError("Parcel ranking bin width must be positive.")
    if extent_ms is None:
        lower_extent = -np.inf
        upper_extent = np.inf
    else:
        lower_extent, upper_extent = [float(value) for value in extent_ms]

    contrast = parcel_trial_contrast_map(primary, comparison)
    rows = []
    for parcel_index, parcel_name in enumerate(names):
        parcel_contrast = contrast[parcel_index]
        finite_bins = np.isfinite(parcel_contrast)
        if np.any(finite_bins):
            candidates = np.flatnonzero(finite_bins)
            peak_index = int(
                candidates[np.argmax(np.abs(parcel_contrast[finite_bins]))]
            )
            mean_primary = float(np.nanmean(primary[:, parcel_index, peak_index]))
            mean_comparison = float(
                np.nanmean(comparison[:, parcel_index, peak_index])
            )
            mean_contrast = float(parcel_contrast[peak_index])
            if np.isfinite(lower_extent):
                peak_start_ms = lower_extent + peak_index * width_ms
                peak_end_ms = min(upper_extent, peak_start_ms + width_ms)
            else:
                center_ms = float(centers[peak_index] * 1000.0)
                peak_start_ms = center_ms - width_ms / 2.0
                peak_end_ms = center_ms + width_ms / 2.0
            primary_count, primary_total, primary_consistency = (
                _dominant_sign_agreement(primary[:, parcel_index, peak_index])
            )
            comparison_count, comparison_total, comparison_consistency = (
                _dominant_sign_agreement(comparison[:, parcel_index, peak_index])
            )
        else:
            peak_index = -1
            mean_primary = np.nan
            mean_comparison = np.nan
            mean_contrast = np.nan
            peak_start_ms = np.nan
            peak_end_ms = np.nan
            primary_count = primary_total = comparison_count = comparison_total = 0
            primary_consistency = comparison_consistency = np.nan
        rows.append(
            {
                "parcel": parcel_name,
                "peak_bin_index": peak_index,
                "peak_start_ms": peak_start_ms,
                "peak_end_ms": peak_end_ms,
                "direction": (
                    str(primary_label)
                    if mean_contrast > 0.0
                    else str(comparison_label)
                    if mean_contrast < 0.0
                    else "Equal"
                ),
                "primary_mean": mean_primary,
                "comparison_mean": mean_comparison,
                "mean_contrast": mean_contrast,
                "absolute_contrast": abs(mean_contrast),
                "primary_sign_count": primary_count,
                "primary_sign_total": primary_total,
                "primary_sign_consistency": primary_consistency,
                "comparison_sign_count": comparison_count,
                "comparison_sign_total": comparison_total,
                "comparison_sign_consistency": comparison_consistency,
                "nested_loo_balanced_accuracy": (
                    _nested_loo_single_parcel_balanced_accuracy(
                        primary[:, parcel_index, :], comparison[:, parcel_index, :]
                    )
                ),
            }
        )

    def sort_key(row):
        score = row["nested_loo_balanced_accuracy"]
        contrast_size = row["absolute_contrast"]
        return (
            -float(score) if np.isfinite(score) else np.inf,
            -float(contrast_size) if np.isfinite(contrast_size) else np.inf,
            str(row["parcel"]),
        )

    rows.sort(key=sort_key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def normalize_parcellation_key(parcellation):
    """Return a canonical atlas key used by the viewer controls."""
    text = str(parcellation or "").strip()
    lower = text.lower().replace("_", "-").replace(" ", "-")
    if lower in {"hcp-mmp", "hcpmmp", "hcp-mmp1", "hcpmmp1", "hcp"}:
        return "HCPMMP1"
    if lower in {"hcp-mmp-combined", "hcpmmp1-combined", "hcp-combined"}:
        return "HCPMMP1_combined"
    if lower in {"aparc+aseg", "volume", "vol"}:
        return "aparc+aseg"
    if lower in {"aparc", "surface-aparc", "freesurfer-aparc"}:
        return "aparc"
    return text or "aparc+aseg"


def is_surface_stc(stc):
    """Return True for two-hemisphere surface SourceEstimate-like objects."""
    vertices = getattr(stc, "vertices", None)
    return isinstance(vertices, list) and len(vertices) == 2


def is_volume_stc(stc):
    """Return True for volume SourceEstimate-like objects."""
    if stc is None:
        return False
    class_name = stc.__class__.__name__.lower()
    if class_name.startswith("vol"):
        return True
    vertices = getattr(stc, "vertices", None)
    return not is_surface_stc(stc) and vertices is not None


def ensure_hcp_mmp_parcellation(subjects_dir):
    """Fetch HCP-MMP annotations into a subjects_dir and return that path."""
    subjects_dir = Path(subjects_dir).expanduser().resolve()
    target_lh = subjects_dir / "fsaverage" / "label" / "lh.HCPMMP1_combined.annot"
    target_rh = subjects_dir / "fsaverage" / "label" / "rh.HCPMMP1_combined.annot"
    if target_lh.exists() and target_rh.exists():
        return str(subjects_dir)

    for candidate in [
        Path.home() / "mne_data" / "MNE-fsaverage-data",
        Path.home() / "mne_data" / "MNE-sample-data" / "subjects",
    ]:
        if (candidate / "fsaverage" / "label" / "lh.HCPMMP1_combined.annot").exists() and (
            candidate / "fsaverage" / "label" / "rh.HCPMMP1_combined.annot"
        ).exists():
            return str(candidate)

    subjects_dir.mkdir(parents=True, exist_ok=True)
    try:
        mne.datasets.fetch_fsaverage(subjects_dir=str(subjects_dir), verbose=False)
    except Exception as exc:
        print(f"fetch_fsaverage notice: {exc}")
    try:
        mne.datasets.fetch_hcp_mmp_parcellation(
            subjects_dir=str(subjects_dir),
            combine=True,
            accept=True,
            verbose=False,
        )
    except Exception as exc:
        print(f"fetch_hcp_mmp_parcellation notice: {exc}")
    return str(subjects_dir)


def read_surface_labels(subject, subjects_dir, parcellation):
    """Read labels for a surface parcellation."""
    parc = normalize_parcellation_key(parcellation)
    if parc in HCP_MMP_PARCS:
        try:
            subjects_dir = ensure_hcp_mmp_parcellation(subjects_dir)
        except Exception as exc:
            print(f"ensure_hcp_mmp_parcellation error: {exc}")
        subject = "fsaverage"
    try:
        return mne.read_labels_from_annot(
            subject=str(subject),
            parc=parc,
            subjects_dir=str(subjects_dir),
            verbose="error",
        )
    except Exception as exc:
        print(f"read_labels_from_annot error for {subject} {parc}: {exc}")
        for candidate in [
            Path.home() / "mne_data" / "MNE-fsaverage-data",
            Path.home() / "mne_data" / "MNE-sample-data" / "subjects",
        ]:
            if candidate.exists() and str(candidate) != str(subjects_dir):
                try:
                    return mne.read_labels_from_annot(
                        subject="fsaverage",
                        parc=parc,
                        subjects_dir=str(candidate),
                        verbose="error",
                    )
                except Exception:
                    pass
        raise


def subject_aseg_path(subject, subjects_dir, parcellation=None):
    """Return the first supported aseg-like atlas path for a subject."""
    subject_dir = Path(subjects_dir).expanduser().resolve() / str(subject)
    if parcellation:
        parc = normalize_parcellation_key(parcellation)
        if parc in HCP_MMP_PARCS:
            for name in (f"{parc}+aseg.mgz", f"{parc}.mgz"):
                path = subject_dir / "mri" / name
                if path.exists():
                    return str(path)
            raise FileNotFoundError(
                f"Volumetric HCP atlas '{parc}+aseg.mgz' not found in {subject_dir / 'mri'}. "
                f"Run scripts/create_hcp_volumetric_atlas.slurm first."
            )
    for name in ("aparc+aseg.mgz", "aparc.DKTatlas+aseg.deep.mgz"):
        path = subject_dir / "mri" / name
        if path.exists():
            return str(path)
    return ""


def aseg_name_from_path(aseg_path):
    """Return the MNE aseg name corresponding to an atlas file path."""
    name = Path(str(aseg_path)).name
    if name.endswith(".mgz"):
        name = name[:-4]
    return name or "aparc+aseg"


def surface_label_time_courses(stc, labels, *, use_abs=True):
    """Extract mean label time courses by intersecting label and STC vertices."""
    if not is_surface_stc(stc):
        raise RuntimeError("Surface label extraction requires a surface SourceEstimate.")

    data = np.asarray(stc.data, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0:
        raise RuntimeError("Source estimate has invalid data shape.")
    if use_abs:
        data = np.abs(data)

    vertices = [np.asarray(v, dtype=int) for v in stc.vertices]
    hemi_offsets = {"lh": 0, "rh": len(vertices[0])}
    vertex_maps = [
        {int(vertex): idx for idx, vertex in enumerate(vertices[0])},
        {int(vertex): idx for idx, vertex in enumerate(vertices[1])},
    ]

    label_names = []
    label_objects = []
    label_tc = []
    for label in labels or []:
        hemi = str(getattr(label, "hemi", "") or "").lower()
        if hemi not in {"lh", "rh"}:
            continue
        hemi_idx = 0 if hemi == "lh" else 1
        rows = []
        for vertex in np.asarray(getattr(label, "vertices", []), dtype=int):
            row = vertex_maps[hemi_idx].get(int(vertex))
            if row is not None:
                rows.append(hemi_offsets[hemi] + row)
        if not rows:
            continue
        label_names.append(str(getattr(label, "name", "") or "label"))
        label_objects.append(label)
        label_tc.append(np.nanmean(data[np.asarray(rows, dtype=int), :], axis=0))

    if not label_tc:
        raise RuntimeError("No atlas labels intersect the source estimate vertices.")
    return np.asarray(label_tc, dtype=float), label_names, label_objects


def volume_atlas_ids_for_aseg(aseg_path, parcellation=None):
    """Build an MNE-compatible {label_name: int_id} lookup dict for custom HCP+aseg volumes.

    mri_aparc2aseg assigns left cortical annotation indices to 1000 + annot_idx
    and right cortical annotation indices to 2000 + annot_idx, while keeping
    subcortical aseg IDs (<1000). Without a custom LUT, MNE's default
    FreeSurferColorLUT.txt misinterprets 1001..1035 as Desikan-Killiany aparc
    names (ctx-lh-bankssts, etc.) instead of the HCP-MMP annotation names.
    """
    path = Path(str(aseg_path)).expanduser().resolve()
    parc = normalize_parcellation_key(parcellation) if parcellation else ""
    if parc not in HCP_MMP_PARCS:
        for candidate in ("HCPMMP1_combined", "HCPMMP1"):
            if path.name.startswith(candidate):
                parc = candidate
                break
    if parc not in HCP_MMP_PARCS:
        return None

    import nibabel.freesurfer as fs
    subject_dir = path.parent.parent
    subjects_dir = subject_dir.parent
    lut, _ = mne.read_freesurfer_lut()
    # Retain subcortical (<1000) and non-cortical (>=3000) FreeSurfer LUT entries
    custom_lut = {str(k): int(v) for k, v in lut.items() if int(v) < 1000 or int(v) >= 3000}

    for hemi, offset in (("lh", 1000), ("rh", 2000)):
        annot_candidates = [
            subject_dir / "label" / f"{hemi}.{parc}.annot",
            subjects_dir / "fsaverage" / "label" / f"{hemi}.{parc}.annot",
            Path.home() / "mne_data" / "MNE-fsaverage-data" / "fsaverage" / "label" / f"{hemi}.{parc}.annot",
            Path.home() / "mne_data" / "MNE-sample-data" / "subjects" / "fsaverage" / "label" / f"{hemi}.{parc}.annot",
        ]
        annot_path = next((p for p in annot_candidates if p.is_file()), None)
        if annot_path is None:
            try:
                resolved_dir = Path(ensure_hcp_mmp_parcellation(subjects_dir))
                cand = resolved_dir / "fsaverage" / "label" / f"{hemi}.{parc}.annot"
                if cand.is_file():
                    annot_path = cand
            except Exception:
                pass
        if annot_path is None:
            raise FileNotFoundError(
                f"Cannot resolve cortical label names for {path.name}: missing {hemi}.{parc}.annot in {subject_dir / 'label'}"
            )
        _, _, names = fs.read_annot(str(annot_path))
        for idx, raw_name in enumerate(names):
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            name = name.strip()
            if idx == 0 or not name or name in {"???", "unknown", "Unknown"}:
                label_key = f"ctx-{hemi}-unknown"
            else:
                label_key = f"ctx-{hemi}-{name}"
            custom_lut[label_key] = offset + idx
    return custom_lut


def get_volume_labels_for_aseg(aseg_path, parcellation=None):
    """Return (label_names, atlas_ids_or_None) for an aseg or HCP+aseg volume."""
    atlas_ids = volume_atlas_ids_for_aseg(aseg_path, parcellation=parcellation)
    label_names = [
        str(v)
        for v in mne.get_volume_labels_from_aseg(str(aseg_path), atlas_ids=atlas_ids)
    ]
    return label_names, atlas_ids


def volume_label_time_courses(stc, src, aseg_path, *, use_abs=True, parcellation=None):
    """Extract volume-label time courses from an aseg-like atlas."""
    if src is None:
        raise RuntimeError("Volume label extraction requires the source space used to create the STC.")
    if not aseg_path or not Path(str(aseg_path)).exists():
        raise RuntimeError("A valid aparc+aseg atlas is required for volume label extraction.")
    label_names, atlas_ids = get_volume_labels_for_aseg(aseg_path, parcellation=parcellation)
    labels_arg = (str(aseg_path), {name: atlas_ids[name] for name in label_names}) if atlas_ids else str(aseg_path)
    label_tc = np.asarray(
        stc.extract_label_time_course(
            labels_arg,
            src=src,
            mode="mean",
            allow_empty=True,
            verbose="error",
        ),
        dtype=float,
    )
    if use_abs:
        label_tc = np.abs(label_tc)
    if label_tc.ndim != 2:
        raise RuntimeError("Volume label extraction returned invalid data.")
    if len(label_names) != label_tc.shape[0]:
        label_names = [f"Label {idx + 1}" for idx in range(label_tc.shape[0])]
    return label_tc, label_names


def top_label_indices(label_tc, label_names, top_n=7):
    """Return label indices ordered by descending peak activation."""
    data = np.asarray(label_tc, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0:
        return []
    try:
        top_n = max(1, int(top_n))
    except Exception:
        top_n = 7
    peaks = np.nanmax(np.abs(data), axis=1)
    ordered = np.argsort(peaks)[::-1]
    selected = []
    for idx in ordered:
        name = str(label_names[int(idx)] if int(idx) < len(label_names) else "")
        if not name or "unknown" in name.lower():
            continue
        if not np.isfinite(peaks[int(idx)]):
            continue
        selected.append(int(idx))
        if len(selected) >= top_n:
            break
    return selected


def filter_parcels(parcel_names, filter_name="all", parcellation="aparc", hemisphere="both"):
    """Apply the GUI's anatomical presets, optionally restricted to one hemisphere.

    Preserve input order. ``both`` includes midline/unlateralized labels.
    """
    if filter_name == "both":
        filter_name = "visual_semantic"
    if filter_name not in {"all", "clear", "visual", "semantic", "visual_semantic"}:
        raise ValueError(f"Unknown parcel filter: {filter_name}")
    if hemisphere not in {"both", "left", "right"}:
        raise ValueError(f"Unknown parcel hemisphere: {hemisphere}")
    selected = []
    for name in parcel_names:
        if hemisphere != "both" and _parcel_hemisphere(name).lower() != hemisphere:
            continue
        visual = filter_name in {"visual", "visual_semantic"} and is_visual_parcel(name, parcellation)
        semantic = filter_name in {"semantic", "visual_semantic"} and is_semantic_parcel(name, parcellation)
        if filter_name == "all" or visual or semantic:
            selected.append(name)
    return selected
