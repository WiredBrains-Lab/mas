"""Orientation-preserving, single-trial source band-power features."""

import mne
import numpy as np
from scipy.signal import hilbert


# Match the signal-space semantic decoder's bands.
SOURCE_POWER_BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 70.0),
    ("high_gamma", 70.0, 150.0),
)


def resolve_source_power_bands(selection, sfreq, low_hz, high_hz):
    """All selects surviving bands; explicit selections must survive in full."""
    if isinstance(selection, str):
        selection = [part.strip().lower() for part in selection.split(",") if part.strip()]
    else:
        selection = [str(part).strip().lower() for part in (selection or ["all"])]
    if not selection:
        raise ValueError("Select at least one source power band.")
    known = {name for name, _, _ in SOURCE_POWER_BANDS}
    if set(selection) - known - {"all"}:
        raise ValueError(f"Unknown source power bands: {selection}.")
    if "all" in selection and len(selection) != 1:
        raise ValueError("Select 'all' or individual source power bands, not both.")
    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Source band power requires a positive sampling rate.")
    available = [
        band for band in SOURCE_POWER_BANDS
        if (not low_hz or band[1] >= low_hz)
        and (not high_hz or band[2] <= high_hz)
        and band[2] < sfreq / 2.0
    ]
    if selection != ["all"]:
        unavailable = set(selection) - {name for name, _, _ in available}
        if unavailable:
            raise ValueError(
                f"Source power bands {sorted(unavailable)} are unavailable after "
                f"the {low_hz:g}–{high_hz:g} Hz passband / {sfreq:g} Hz sampling."
            )
        available = [band for band in available if band[0] in selection]
    if not available:
        raise ValueError("No complete source power bands survive preprocessing and Nyquist.")
    return available


def source_power_feature_metadata(roi_names, band_defs, bin_edges, baseline_db):
    """Metadata in ROI, band, bin order, matching a flattened trial tensor."""
    rows = []
    stat = "band_power_db" if baseline_db else "log10_band_power"
    for roi in roi_names:
        for band, low, high in band_defs:
            for start, end in bin_edges:
                rows.append({
                    "feature": f"{roi}|{stat}|{band}|{start:.6f}-{end:.6f}s",
                    "roi": roi, "stat": stat, "band": band,
                    "band_low_hz": float(low), "band_high_hz": float(high),
                    "bin_start_s": float(start), "bin_end_s": float(end),
                    "relative_time_s": float((start + end) / 2),
                    "feature_index": len(rows),
                })
    return rows


def source_bandpower_trial(stc, band_defs, bin_masks, baseline_mask, aggregate,
                          vertex_batch_size=256):
    """Return [ROI, band, bin] log power from signed or vector source currents.

    Spectral transforms precede both orientation pooling and spatial averaging.
    ``aggregate`` receives a scalar surface/volume STC whose columns contain
    linear-power window means (plus baseline, if requested). Its time axis is
    an index axis solely for MNE morphing/atlas extraction, not physical time.
    This avoids materializing a vertex × band × sample power tensor.
    """
    data = np.asarray(stc.data)
    if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[1] != 3):
        raise ValueError("Expected signed [vertex, time] or vector [vertex, 3, time] currents.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Source currents contain non-finite samples.")
    masks = [np.asarray(mask, dtype=bool) for mask in bin_masks]
    if baseline_mask is not None:
        baseline_mask = np.asarray(baseline_mask, dtype=bool)
        if np.count_nonzero(baseline_mask) < 2:
            raise ValueError("Source power baseline requires at least two samples.")
        masks.append(baseline_mask)
    if not masks or any(mask.shape != (data.shape[-1],) or not np.any(mask) for mask in masks):
        raise ValueError("Source power windows are not covered by the padded epoch.")
    sfreq = 1.0 / float(stc.tstep)
    batch_size = max(1, int(vertex_batch_size))
    results = []
    for _name, low, high in band_defs:
        summaries = np.empty((data.shape[0], len(masks)), dtype=float)
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            filtered = mne.filter.filter_data(
                np.asarray(data[start:stop], dtype=np.float64), sfreq,
                l_freq=float(low), h_freq=float(high), method="iir",
                iir_params={"order": 4, "ftype": "butter"},
                n_jobs=1, verbose="error",
            )
            power = np.abs(hilbert(filtered, axis=-1)) ** 2
            if power.ndim == 3:
                power = power.sum(axis=1)
            for index, mask in enumerate(masks):
                summaries[start:stop, index] = power[:, mask].mean(axis=-1)
        scalar_class = (
            mne.VolSourceEstimate
            if isinstance(stc, (mne.VolSourceEstimate, mne.VolVectorSourceEstimate))
            else mne.SourceEstimate
        )
        power_stc = scalar_class(
            summaries, vertices=stc.vertices, tmin=0.0, tstep=1.0, subject=stc.subject,
        )
        regional = np.asarray(aggregate(power_stc), dtype=float)
        if regional.ndim != 2 or regional.shape[1] != len(masks):
            raise ValueError("Source power atlas extraction returned inconsistent dimensions.")
        if not np.all(np.isfinite(regional)) or np.any(regional < 0):
            raise ValueError("Source power atlas extraction returned invalid power.")
        if np.any(np.max(regional, axis=1) <= 0):
            raise ValueError("A selected source power ROI is empty or has zero signal.")
        tiny = np.finfo(float).tiny
        if baseline_mask is None:
            values = np.log10(np.maximum(regional, tiny))
        else:
            baseline = regional[:, -1:]
            if np.any(baseline <= tiny):
                raise ValueError("A selected source power ROI has zero baseline power.")
            values = 10.0 * np.log10(np.maximum(regional[:, :-1], baseline * 1e-12) / baseline)
        results.append(values)
    return np.stack(results, axis=1).astype(np.float32)
