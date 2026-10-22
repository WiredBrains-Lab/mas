"""MEG adapter for the CIDECoG transformer decoder."""

from __future__ import annotations

import json
import os
import re
import sys
import types
import importlib.util
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from core.instruction_embeddings import load_instruction_targets, score_instructions
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from core.concept_decoding_signal import (
    CONCEPT_PREPROCESSING_DEFAULTS,
    annotate_signal_trial_outliers,
    common_concept_channels,
    concept_bandpass_from_options,
    concept_cached_epochs_for_file,
    concept_epoch_reject_from_options,
    concept_epoch_reject_summary,
    concept_feature_binning_enabled,
    concept_label_from_trial_row,
    concept_model_architecture_table,
    concept_qc_summary_table,
    concept_raw_for_path,
    concept_resample_from_options,
    feature_edges_for_concept_decoding,
    per_concept_train_test_split,
    zero_shot_concept_train_test_split,
    signal_epoch_bin_masks,
    signal_trial_metric_rows,
    validate_concept_resample_settings,
)


DEFAULT_SEEG_DECODER_SRC = os.environ.get("CIDECOG_DECODER_SRC", "")
SEEG_DECODER_SRC = os.environ.get("CIDECOG_SEEG_DECODER_SRC", DEFAULT_SEEG_DECODER_SRC)

TRANSFORMER_DECODER_ALIASES = {
    "transformer",
    "axial_transformer",
    "registered_transformer",
}

RIDGE_EMBEDDING_DECODER_ALIASES = {
    "embedding_ridge",
    "ridge_embedding",
    "word2vec_ridge",
    "linear_embedding",
}

CONTRASTIVE_LINEAR_EMBEDDING_DECODER_ALIASES = {
    "embedding_contrastive",
    "contrastive_embedding",
    "contrastive_linear",
    "linear_contrastive",
    "class_aware_contrastive",
}

MLP_EMBEDDING_DECODER_ALIASES = {
    "embedding_mlp",
    "mlp_embedding",
    "word2vec_mlp",
    "neural_embedding",
}

ZERO_SHOT_DECODER_ALIASES = {
    "zero_shot",
    "zero_shot_decoding",
    "zero_shot_embedding",
    "zero_shot_ridge",
    "zero_shot_linear",
    "zero_shot_contrastive",
    "zero_shot_mlp",
    "zeroshot",
}

SEMANTIC_EMBEDDING_DECODER_ALIASES = (
    TRANSFORMER_DECODER_ALIASES
    | RIDGE_EMBEDDING_DECODER_ALIASES
    | CONTRASTIVE_LINEAR_EMBEDDING_DECODER_ALIASES
    | MLP_EMBEDDING_DECODER_ALIASES
    | ZERO_SHOT_DECODER_ALIASES
)

SOURCE_SPACE_SEMANTIC_DECODER_ALIASES = (
    RIDGE_EMBEDDING_DECODER_ALIASES
    | CONTRASTIVE_LINEAR_EMBEDDING_DECODER_ALIASES
    | ZERO_SHOT_DECODER_ALIASES
)


def is_zero_shot_decoder(decoder_type: str | None) -> bool:
    return str(decoder_type or "").strip().lower() in ZERO_SHOT_DECODER_ALIASES


def is_transformer_decoder(decoder_type: str | None) -> bool:
    return str(decoder_type or "").strip().lower() in TRANSFORMER_DECODER_ALIASES


def is_semantic_embedding_decoder(decoder_type: str | None) -> bool:
    return str(decoder_type or "").strip().lower() in SEMANTIC_EMBEDDING_DECODER_ALIASES


def is_source_space_semantic_decoder(decoder_type: str | None) -> bool:
    return str(decoder_type or "").strip().lower() in SOURCE_SPACE_SEMANTIC_DECODER_ALIASES


def semantic_embedding_decoder_kind(decoder_type: str | None) -> str:
    key = str(decoder_type or "transformer").strip().lower()
    if key in ZERO_SHOT_DECODER_ALIASES:
        if "contrastive" in key:
            return "zero_shot_contrastive"
        if "mlp" in key:
            return "zero_shot_mlp"
        return "zero_shot"
    if key in RIDGE_EMBEDDING_DECODER_ALIASES:
        return "embedding_ridge"
    if key in CONTRASTIVE_LINEAR_EMBEDDING_DECODER_ALIASES:
        return "embedding_contrastive"
    if key in MLP_EMBEDDING_DECODER_ALIASES:
        return "embedding_mlp"
    return "transformer"


def _emit_progress(options_or_callback: Any, message: str) -> None:
    callback = options_or_callback
    if isinstance(options_or_callback, dict):
        callback = options_or_callback.get("progress_callback")
    if callable(callback):
        try:
            callback(str(message))
        except Exception:
            pass


def _emit_training_metric(options: dict[str, Any] | None, metric: dict[str, Any]) -> None:
    callback = options.get("training_metric_callback") if isinstance(options, dict) else None
    if callable(callback):
        try:
            callback(dict(metric or {}))
        except Exception:
            pass


def _decode_h5_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def _canonicalize_embedding_key(value: Any) -> str:
    text = _decode_h5_str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _as_2d_float32(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {out.shape}.")
    return out


def _load_json_embedding_map(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("JSON embedding file must contain an object.")
    if "names" in payload and "embeddings" in payload:
        names = [_decode_h5_str(v) for v in payload["names"]]
        emb = _as_2d_float32(payload["embeddings"])
        if emb.shape[0] != len(names):
            raise ValueError("JSON names/embeddings length mismatch.")
        return {str(name): emb[idx] for idx, name in enumerate(names)}

    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        vec = np.asarray(value, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Embedding for {key!r} must be 1D, got shape {vec.shape}.")
        out[str(key)] = vec
    return out


def _read_h5_strings(h5: h5py.File, key: str) -> list[str]:
    raw = h5[key][:]
    return [_decode_h5_str(v) for v in raw.tolist()]


def _load_h5_embedding_map(path: str) -> dict[str, np.ndarray]:
    name_keys = ("names", "all_concepts", "stimuli/all_concepts", "correct_names", "stimuli/correct_names")
    embedding_keys = ("embeddings", "semantic_embeddings", "stimuli/semantic_embeddings")
    with h5py.File(path, "r") as h5:
        name_key = next((key for key in name_keys if key in h5 and isinstance(h5[key], h5py.Dataset)), None)
        emb_key = next((key for key in embedding_keys if key in h5 and isinstance(h5[key], h5py.Dataset)), None)
        if name_key is not None and emb_key is not None:
            names = _read_h5_strings(h5, name_key)
            emb = _as_2d_float32(h5[emb_key][:])
            if emb.shape[0] != len(names):
                raise ValueError("HDF5 names/embeddings length mismatch.")
            return {str(name): emb[idx] for idx, name in enumerate(names)}

        emb_group_key = next((key for key in embedding_keys if key in h5 and isinstance(h5[key], h5py.Group)), None)
        if emb_group_key is not None:
            out: dict[str, np.ndarray] = {}

            def collect(group: h5py.Group, prefix: str = "") -> None:
                for name, obj in group.items():
                    full_name = f"{prefix}/{name}" if prefix else str(name)
                    if isinstance(obj, h5py.Dataset):
                        vec = np.asarray(obj[:], dtype=np.float32)
                        if vec.ndim == 1:
                            out[full_name] = vec
                    elif isinstance(obj, h5py.Group):
                        collect(obj, prefix=full_name)

            collect(h5[emb_group_key])
            if out:
                return out
    raise ValueError("Could not find a compatible HDF5 embedding dictionary.")


def _load_npz_embedding_map(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        if len(keys) == 1:
            obj = data[keys[0]]
            if obj.shape == () and isinstance(obj.item(), dict):
                raw = obj.item()
                return {str(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}

        name_key = next((key for key in ("names", "all_concepts", "correct_names", "stimuli") if key in data), None)
        emb_key = next((key for key in ("embeddings", "semantic_embeddings", "vectors") if key in data), None)
        if name_key is None or emb_key is None:
            raise ValueError("NPZ embedding file needs names/all_concepts plus embeddings/semantic_embeddings arrays.")
        names = [_decode_h5_str(v) for v in np.asarray(data[name_key]).tolist()]
        emb = _as_2d_float32(data[emb_key])
        if emb.shape[0] != len(names):
            raise ValueError("NPZ names/embeddings length mismatch.")
        return {str(name): emb[idx] for idx, name in enumerate(names)}


def _load_table_embedding_map(path: str) -> dict[str, np.ndarray]:
    sep = "\t" if str(path).lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty or df.shape[1] < 2:
        raise ValueError("Embedding table must have one name column plus numeric embedding columns.")
    preferred = [col for col in ("name", "concept", "stimulus", "canonical_stimulus", "label") if col in df.columns]
    name_col = preferred[0] if preferred else df.columns[0]
    value_cols = [col for col in df.columns if col != name_col]
    values = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Embedding table did not contain numeric embedding columns.")
    names = df[name_col].astype(str).tolist()
    return {str(name): values[idx] for idx, name in enumerate(names)}


def load_embedding_map(path: str) -> dict[str, np.ndarray]:
    clean_path = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not clean_path or not os.path.exists(clean_path):
        raise ValueError("Semantic embedding decoding requires an embedding dictionary file path.")
    suffix = Path(clean_path).suffix.lower()
    if suffix == ".json":
        return _load_json_embedding_map(clean_path)
    if suffix in {".h5", ".hdf5"}:
        return _load_h5_embedding_map(clean_path)
    if suffix == ".npz":
        return _load_npz_embedding_map(clean_path)
    if suffix in {".csv", ".tsv", ".tab"}:
        return _load_table_embedding_map(clean_path)
    raise ValueError(f"Unsupported embedding dictionary format: {suffix}.")


def _canonical_embedding_lookup(embedding_map: dict[str, np.ndarray]) -> dict[str, tuple[str, np.ndarray]]:
    lookup: dict[str, tuple[str, np.ndarray]] = {}
    for key, value in embedding_map.items():
        canonical = _canonicalize_embedding_key(key)
        if canonical and canonical not in lookup:
            lookup[canonical] = (str(key), np.asarray(value, dtype=np.float32).reshape(-1))
    return lookup


def _trial_embedding_candidates(row: pd.Series | dict[str, Any]) -> list[str]:
    values: list[str] = []
    for col in ("canonical_stimulus", "concept", "stimulus", "concept_label", "stimulus_group"):
        value = row.get(col, "")
        if pd.notna(value) and str(value).strip():
            values.append(str(value).strip())
            base = os.path.basename(str(value).strip())
            if base and base != str(value).strip():
                values.append(base)
            stem = os.path.splitext(base)[0] if base else ""
            if stem:
                values.append(stem)
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def resolve_trial_embeddings(trial_df: pd.DataFrame, embedding_map: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
        raise ValueError("No trials are available for embedding alignment.")
    direct = {str(key): np.asarray(value, dtype=np.float32).reshape(-1) for key, value in embedding_map.items()}
    canonical_lookup = _canonical_embedding_lookup(direct)
    vectors: list[np.ndarray] = []
    matched_keys: list[str] = []
    missing: list[str] = []
    expected_dim: int | None = None
    for _, row in trial_df.iterrows():
        matched_key = ""
        vector = None
        for candidate in _trial_embedding_candidates(row):
            if candidate in direct:
                matched_key = candidate
                vector = direct[candidate]
                break
            canonical = _canonicalize_embedding_key(candidate)
            if canonical in canonical_lookup:
                matched_key, vector = canonical_lookup[canonical]
                break
        if vector is None:
            missing.append(str(row.get("canonical_stimulus", row.get("stimulus", row.get("concept_label", "")))))
            continue
        if expected_dim is None:
            expected_dim = int(vector.shape[0])
        if int(vector.shape[0]) != int(expected_dim):
            raise ValueError("Embedding dictionary contains inconsistent vector dimensions.")
        vectors.append(np.asarray(vector, dtype=np.float32))
        matched_keys.append(str(matched_key))
    if missing:
        sample = ", ".join([str(v) for v in missing[:8]])
        raise ValueError(f"Missing embeddings for {len(missing)} trial stimuli/concepts. First missing: {sample}")
    if not vectors:
        raise ValueError("No trial embeddings could be resolved.")
    return np.vstack(vectors).astype(np.float32), matched_keys


def prepare_semantic_embedding_feature_payload(
    feature_payload: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Attach dictionary embeddings and a generic decoder input to a feature payload."""
    options = dict(options or {})
    payload = dict(feature_payload or {})
    trial_df = payload.get("trial_df", pd.DataFrame())
    if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
        return None, "Semantic embedding decoding requires a non-empty trial table."

    X_values = payload.get("X_tensor")
    if X_values is None:
        X_values = payload.get("X")
    if X_values is None:
        return None, "Semantic embedding decoding requires a trial feature matrix."
    X_values = np.asarray(X_values, dtype=np.float32)
    if X_values.ndim < 2 or X_values.shape[0] != len(trial_df):
        return None, "Semantic embedding features and trial rows are inconsistent."

    y = np.asarray(payload.get("y", []), dtype=str).reshape(-1)
    if y.size != len(trial_df):
        return None, "Semantic embedding labels and trial rows are inconsistent."

    embedding_path = str(options.get("embedding_dict_path", "") or "").strip()
    if not embedding_path:
        embedding_path = str(payload.get("embedding_dict_path", "") or "").strip()
    if not embedding_path:
        return None, "Semantic embedding decoding requires a concept embedding dictionary path."
    try:
        embedding_map = load_embedding_map(embedding_path)
        embeddings, matched_embedding_keys = resolve_trial_embeddings(
            trial_df, embedding_map
        )
    except Exception as exc:
        return None, f"Could not align trial embeddings: {exc}"

    prepared_trial_df = trial_df.copy().reset_index(drop=True)
    prepared_trial_df["embedding_key"] = matched_embedding_keys
    payload.update(
        {
            "X_tensor": X_values,
            "embeddings": embeddings,
            "trial_df": prepared_trial_df,
            "y": y,
            "embedding_dict_path": os.path.abspath(
                os.path.expanduser(embedding_path)
            ),
        }
    )
    return payload, None


def _default_band_defs(
    sfreq: float,
    available_low: float | None = None,
    available_high: float | None = None,
) -> list[tuple[str, float, float]]:
    """Return only canonical bands fully preserved by preprocessing/Nyquist."""
    nyq = max(0.0, float(sfreq) / 2.0)
    raw = [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("low_gamma", 30.0, 70.0),
        ("high_gamma", 70.0, 150.0),
    ]
    try:
        passband_low = float(available_low)
    except Exception:
        passband_low = 0.0
    try:
        passband_high = float(available_high)
    except Exception:
        passband_high = 0.0
    passband_low = passband_low if np.isfinite(passband_low) and passband_low > 0.0 else 0.0
    passband_high = passband_high if np.isfinite(passband_high) and passband_high > 0.0 else np.inf
    usable_high = min(passband_high, nyq * 0.98)
    tolerance = 1e-9
    bands: list[tuple[str, float, float]] = []
    for name, low, high in raw:
        if float(low) + tolerance < passband_low:
            continue
        if float(high) > usable_high + tolerance:
            continue
        bands.append((name, float(low), float(high)))
    return bands


def _compute_log_bandpower_tensor(
    epoch_data: np.ndarray,
    times: np.ndarray,
    bin_edges: list[tuple[float, float]],
    sfreq: float,
    band_defs: list[tuple[str, float, float]],
    baseline: tuple[float, float] | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute temporally binned Hilbert power, optionally as baseline dB.

    Power is estimated from the full padded epoch and then averaged inside the
    requested bins. This preserves low-frequency content; running an FFT on an
    isolated 5--20 ms bin cannot resolve delta, theta, alpha, or beta power.
    """
    import mne
    from scipy.signal import hilbert

    data = np.asarray(epoch_data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Epoch data must have shape [trials, channels, samples].")
    masks = signal_epoch_bin_masks(times, bin_edges)
    out = np.empty((data.shape[0], data.shape[1], len(band_defs), len(masks)), dtype=np.float32)
    for mask in masks:
        if int(np.sum(mask)) < 1:
            raise ValueError("Band-power bins require at least one sample.")
    baseline_mask = None
    if baseline is not None:
        baseline_start, baseline_end = [float(value) for value in baseline]
        baseline_mask = (
            (np.asarray(times, dtype=float) >= baseline_start)
            & (np.asarray(times, dtype=float) < baseline_end)
        )
        if int(np.sum(baseline_mask)) < 2:
            raise ValueError("Band-power baseline requires at least two samples.")

    batch_size = max(1, int(batch_size))
    eps = np.float64(1e-30)
    for band_idx, (_name, low, high) in enumerate(band_defs):
        for start in range(0, int(data.shape[0]), batch_size):
            stop = min(int(data.shape[0]), start + batch_size)
            batch = np.asarray(data[start:stop], dtype=np.float64)
            filtered = mne.filter.filter_data(
                batch,
                sfreq=float(sfreq),
                l_freq=float(low),
                h_freq=float(high),
                method="iir",
                iir_params={"order": 4, "ftype": "butter"},
                n_jobs=1,
                verbose="error",
            )
            power = np.square(np.abs(hilbert(filtered, axis=-1)))
            baseline_power = None
            if baseline_mask is not None:
                baseline_power = np.nanmean(
                    power[:, :, baseline_mask], axis=-1, keepdims=False
                )
                finite_positive = baseline_power[
                    np.isfinite(baseline_power) & (baseline_power > 0.0)
                ]
                floor = (
                    max(float(np.nanmedian(finite_positive)) * 1e-12, eps)
                    if finite_positive.size
                    else eps
                )
                baseline_power = np.maximum(baseline_power, floor)
            for bin_idx, mask in enumerate(masks):
                bin_power = np.nanmean(power[:, :, mask], axis=-1)
                if baseline_power is None:
                    values = np.log10(np.maximum(bin_power, eps))
                else:
                    values = 10.0 * np.log10(
                        np.maximum(bin_power, baseline_power * 1e-12)
                        / baseline_power
                    )
                out[start:stop, :, band_idx, bin_idx] = np.nan_to_num(
                    values, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
    return out


def _compute_log_bandpower_timeseries(
    epoch_data: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    band_defs: list[tuple[str, float, float]],
    display_start: float,
    display_end: float,
    baseline: tuple[float, float] | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute cropped log band-power envelopes at each displayed sample."""
    import mne
    from scipy.signal import hilbert

    data = np.asarray(epoch_data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Epoch data must have shape [trials, channels, samples].")
    times = np.asarray(times, dtype=float)
    display_mask = (times >= float(display_start)) & (times <= float(display_end))
    if not np.any(display_mask):
        raise ValueError("Band-power display window contains no samples.")
    baseline_mask = None
    if baseline is not None:
        baseline_mask = (times >= float(baseline[0])) & (times < float(baseline[1]))
        if int(np.sum(baseline_mask)) < 2:
            raise ValueError("Band-power baseline requires at least two samples.")
    out = np.empty(
        (data.shape[0], data.shape[1], len(band_defs), int(np.sum(display_mask))),
        dtype=np.float32,
    )
    eps = np.float64(1e-30)
    batch_size = max(1, int(batch_size))
    for band_idx, (_name, low, high) in enumerate(band_defs):
        for start in range(0, int(data.shape[0]), batch_size):
            stop = min(int(data.shape[0]), start + batch_size)
            batch = np.asarray(data[start:stop], dtype=np.float64)
            filtered = mne.filter.filter_data(
                batch,
                sfreq=float(sfreq),
                l_freq=float(low),
                h_freq=float(high),
                method="iir",
                iir_params={"order": 4, "ftype": "butter"},
                n_jobs=1,
                verbose="error",
            )
            power = np.square(np.abs(hilbert(filtered, axis=-1)))
            if baseline_mask is None:
                values = np.log10(np.maximum(power[:, :, display_mask], eps))
            else:
                baseline_power = np.nanmean(
                    power[:, :, baseline_mask], axis=-1, keepdims=True
                )
                finite_positive = baseline_power[
                    np.isfinite(baseline_power) & (baseline_power > 0.0)
                ]
                floor = (
                    max(float(np.nanmedian(finite_positive)) * 1e-12, eps)
                    if finite_positive.size
                    else eps
                )
                baseline_power = np.maximum(baseline_power, floor)
                values = 10.0 * np.log10(
                    np.maximum(
                        power[:, :, display_mask], baseline_power * 1e-12
                    )
                    / baseline_power
                )
            out[start:stop, :, band_idx, :] = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
    return out


def build_signal_bandpower_transformer_payload(context: Any, options: dict[str, Any] | None = None):
    if options is None:
        options = {}
    _emit_progress(options, "preparing signal-space band-power features...")
    trial_df = getattr(context, "_fif_trial_response_table", pd.DataFrame())
    if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
        return None, "No mapped trial table is available. Load FIF runs with events_log mapping first."
    if "source_fif_path" not in trial_df.columns:
        return None, "Mapped trial table does not include source FIF paths."

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
        bandpass_low_hz, bandpass_high_hz = concept_bandpass_from_options(options)
    except ValueError as exc:
        return None, str(exc)
    try:
        min_trials = max(2, int(options.get("min_trials_per_class", 2)))
    except Exception:
        min_trials = 2
    channel_mode = str(options.get("channel_mode", "all") or "all").strip().lower()
    if channel_mode not in {"all", "mag", "grad"}:
        channel_mode = "all"
    try:
        max_channels = int(options.get("max_channels", 0))
    except Exception:
        max_channels = 0

    feature_binning = concept_feature_binning_enabled(options)
    if feature_binning:
        bin_edges = feature_edges_for_concept_decoding(
            window_start,
            window_end,
            options.get("bin_size_ms", 0),
        )
    else:
        bin_edges = [(float(window_start), float(window_end))]
    if not bin_edges:
        return None, "Concept decoding feature bins could not be constructed."
    baseline_correct = bool(options.get("baseline_correct", True))
    try:
        resample_enabled, resample_sfreq = concept_resample_from_options(options)
    except ValueError as exc:
        return None, str(exc)

    df = context._annotate_trial_df_with_stimulus_groups(
        trial_df.copy(),
        getattr(context, "_fif_filename", None),
    )
    df["concept_label"] = [concept_label_from_trial_row(context, row, group_mode=group_mode) for _, row in df.iterrows()]
    df["concept_label"] = df["concept_label"].astype(str).str.strip()
    valid = (
        df["concept_label"].astype(bool)
        & (df["concept_label"].str.lower() != "unknown")
        & df["source_fif_path"].astype(str).str.len().gt(0)
    )
    df = df.loc[valid].copy().reset_index(drop=True)
    if df.empty:
        return None, "No valid concept-labeled trials are available for semantic embedding decoding."

    counts = df["concept_label"].value_counts()
    keep_concepts = counts[counts >= min_trials].index.tolist()
    if len(keep_concepts) < 2:
        return None, "Need at least two concepts with enough trials for semantic embedding decoding."
    df = df[df["concept_label"].isin(keep_concepts)].copy().reset_index(drop=True)

    embedding_path = str(options.get("embedding_dict_path", "") or "").strip()
    if not embedding_path:
        return None, "Semantic embedding decoding requires a concept embedding dictionary path."
    try:
        embedding_map = load_embedding_map(embedding_path)
    except Exception as exc:
        return None, f"Could not load embedding dictionary: {exc}"

    raw_cache: dict[str, Any] = {}
    raw_by_path: dict[str, Any] = {}
    skipped_paths: list[tuple[str, str]] = []
    for path in pd.Index(df["source_fif_path"].astype(str)).drop_duplicates().tolist():
        path_abs = os.path.abspath(str(path))
        try:
            _emit_progress(options, f"loading MEG run {len(raw_by_path) + 1}: {os.path.basename(path_abs)}")
            raw = concept_raw_for_path(context, path_abs, raw_cache)
            raw_by_path[path_abs] = raw
        except Exception as exc:
            skipped_paths.append((path_abs, str(exc)))
    if not raw_by_path:
        detail = f" First error: {skipped_paths[0][1]}" if skipped_paths else ""
        return None, f"Could not load/filter any FIF files for transformer decoding.{detail}"
    try:
        validate_concept_resample_settings(raw_by_path, resample_enabled, resample_sfreq, h_freq=bandpass_high_hz)
    except ValueError as exc:
        return None, str(exc)

    available_paths = set(raw_by_path.keys())
    df["source_fif_path"] = df["source_fif_path"].map(lambda p: os.path.abspath(str(p)))
    df = df[df["source_fif_path"].isin(available_paths)].copy().reset_index(drop=True)
    if df.empty:
        return None, "No trials remained after loading available FIF files."

    channel_names = common_concept_channels(raw_by_path, channel_mode, max_channels=max_channels)
    if not channel_names:
        return None, f"No common {channel_mode} MEG channels are available across decoded runs."

    sfreqs = []
    for raw in raw_by_path.values():
        try:
            sfreqs.append(float(raw.info["sfreq"]))
        except Exception:
            pass
    if not sfreqs:
        return None, "Transformer band-power extraction requires valid FIF sampling rates."
    if resample_enabled:
        sfreq = float(resample_sfreq)
    else:
        if max(sfreqs) - min(sfreqs) > 1e-6:
            return None, "Transformer band-power extraction requires the same sampling rate across decoded runs unless downsampling is enabled."
        sfreq = float(sfreqs[0])
    band_defs = _default_band_defs(
        sfreq,
        available_low=bandpass_low_hz,
        available_high=bandpass_high_hz,
    )
    if not band_defs:
        return None, (
            "No complete canonical frequency band survives the shared bandpass "
            "and downsampled Nyquist limit."
        )

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
        power_padding_s = max(
            0.0, float(options.get("semantic_power_padding_s", 0.5))
        )
    except Exception:
        power_padding_s = 0.5
    power_baseline = (-0.2, 0.0) if baseline_correct else None
    epoch_tmin = min(
        float(window_start), power_baseline[0] if power_baseline is not None else float(window_start)
    ) - power_padding_s
    epoch_tmax = float(window_end) + power_padding_s
    # Band-power normalization is performed in linear power after filtering;
    # waveform mean subtraction is not an equivalent baseline correction.
    epoch_baseline = None
    epoch_reject = concept_epoch_reject_from_options(options)
    reject_summary = concept_epoch_reject_summary(epoch_reject)

    tensor_rows: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []
    trial_qc_rows: list[dict[str, Any]] = []
    feature_times: np.ndarray | None = None
    dropped_trials = 0
    epoch_cache_hits = 0
    epoch_cache_misses = 0
    df["_signal_qc_uid"] = np.arange(len(df), dtype=int)

    grouped_files = list(df.groupby("source_fif_path", sort=False))
    for file_idx, (path_abs, file_df) in enumerate(grouped_files, start=1):
        _emit_progress(
            options,
            f"extracting band-power run {file_idx}/{len(grouped_files)}: {os.path.basename(path_abs)}",
        )
        raw = raw_by_path.get(path_abs)
        if raw is None:
            continue
        first_samp = int(getattr(raw, "first_samp", 0))

        events = []
        input_rows = []
        for row_idx, row in file_df.iterrows():
            try:
                onset_sample = int(round(float(row.get("onset_sample"))))
            except Exception:
                dropped_trials += 1
                dropped = row.to_dict()
                dropped.update(
                    {
                        "_source_table_index": int(row_idx),
                        "source_fif_path": str(path_abs),
                        "qc_status": "dropped",
                        "qc_reason": "invalid_onset_sample",
                        "qc_outlier": False,
                        "qc_outlier_score": np.nan,
                    }
                )
                trial_qc_rows.append(dropped)
                continue
            events.append([int(onset_sample + first_samp), 0, 1])
            kept = row.to_dict()
            kept["_source_table_index"] = int(row_idx)
            input_rows.append(kept)

        if not events:
            continue
        input_df = pd.DataFrame(input_rows).reset_index(drop=True)
        events_arr = np.asarray(events, dtype=np.int64)
        epoch_payload, epoch_err, cache_hit = concept_cached_epochs_for_file(
            context,
            options,
            raw,
            path_abs,
            events_arr,
            channel_names,
            bandpass_low_hz,
            bandpass_high_hz,
            epoch_tmin,
            epoch_tmax,
            epoch_baseline,
            epoch_reject=epoch_reject,
            resample_sfreq=resample_sfreq if resample_enabled else None,
            label="transformer band-power epochs",
        )
        if cache_hit:
            epoch_cache_hits += 1
        else:
            epoch_cache_misses += 1
        if epoch_payload is None:
            skipped_paths.append((path_abs, epoch_err or "epoching failed"))
            dropped_trials += len(input_df)
            continue

        selected_idx = np.asarray(epoch_payload.get("selected_idx", []), dtype=int)
        selected_set = set(selected_idx.tolist())
        drop_log = list(epoch_payload.get("drop_log", []))
        for original_idx, row in input_df.iterrows():
            if int(original_idx) in selected_set:
                continue
            reason_items = []
            if original_idx < len(drop_log):
                try:
                    reason_items = [str(v) for v in drop_log[int(original_idx)] if str(v)]
                except Exception:
                    reason_items = []
            reason = "; ".join(reason_items) if reason_items else "mne_epoch_rejected"
            dropped = row.to_dict()
            dropped.update(
                {
                    "source_fif_path": str(path_abs),
                    "qc_status": "dropped",
                    "qc_reason": reason,
                    "qc_outlier": False,
                    "qc_outlier_score": np.nan,
                }
            )
            trial_qc_rows.append(dropped)
            dropped_trials += 1
        epoch_data = np.asarray(epoch_payload.get("epoch_data", np.empty((0, 0, 0))), dtype=np.float32)
        if epoch_data.shape[0] == 0:
            continue

        epoch_input_df = input_df.iloc[selected_idx].reset_index(drop=True)
        times = np.asarray(epoch_payload.get("times", []), dtype=float)
        qc_df = signal_trial_metric_rows(epoch_data, times, bin_edges, epoch_input_df, path_abs)
        qc_df = annotate_signal_trial_outliers(qc_df, threshold=trial_outlier_z, min_trials=8)
        try:
            path_sfreq = float(epoch_payload.get("sfreq", sfreq))
            if abs(path_sfreq - float(sfreq)) > 1e-6:
                raise ValueError(
                    f"resampled sampling rate mismatch: expected {float(sfreq):g} Hz, got {path_sfreq:g} Hz"
                )
            if feature_binning:
                bandpower = _compute_log_bandpower_tensor(
                    epoch_data,
                    times,
                    bin_edges,
                    sfreq,
                    band_defs,
                    baseline=power_baseline,
                    batch_size=int(options.get("semantic_power_batch_size", 32) or 32),
                )
            else:
                bandpower = _compute_log_bandpower_timeseries(
                    epoch_data,
                    times,
                    sfreq,
                    band_defs,
                    display_start=window_start,
                    display_end=window_end,
                    baseline=power_baseline,
                    batch_size=int(options.get("semantic_power_batch_size", 32) or 32),
                )
                path_feature_times = np.asarray(times, dtype=float)
                path_feature_times = path_feature_times[
                    (path_feature_times >= float(window_start))
                    & (path_feature_times <= float(window_end))
                ]
                if feature_times is None:
                    feature_times = path_feature_times
                elif feature_times.shape != path_feature_times.shape or not np.allclose(feature_times, path_feature_times, atol=1e-9):
                    raise ValueError("resampled transformer time grid differs from first decoded run")
        except Exception as exc:
            skipped_paths.append((path_abs, f"band-power extraction failed: {exc}"))
            dropped_trials += len(epoch_input_df)
            continue

        for epoch_idx in range(bandpower.shape[0]):
            qc_row = qc_df.iloc[epoch_idx].to_dict() if epoch_idx < len(qc_df) else epoch_input_df.iloc[epoch_idx].to_dict()
            is_outlier = bool(qc_row.get("qc_outlier", False))
            if is_outlier and exclude_qc_outliers:
                qc_row["qc_status"] = "dropped"
                qc_row["qc_reason"] = "qc_outlier"
                trial_qc_rows.append(qc_row)
                dropped_trials += 1
                continue
            qc_row["qc_status"] = "kept"
            qc_row["qc_reason"] = "qc_outlier_not_excluded" if is_outlier else ""
            trial_qc_rows.append(qc_row)
            tensor_rows.append(np.nan_to_num(bandpower[epoch_idx], nan=0.0, posinf=0.0, neginf=0.0))
            kept = epoch_input_df.iloc[epoch_idx].to_dict()
            for key in (
                "qc_finite_fraction",
                "qc_max_abs",
                "qc_ptp_max",
                "qc_ptp_median",
                "qc_rms_median",
                "qc_rms_max",
                "qc_std_median",
                "qc_baseline_rms_median",
                "qc_outlier",
                "qc_outlier_score",
            ):
                kept[key] = qc_row.get(key, np.nan)
            kept_rows.append(kept)

    if not tensor_rows:
        return None, "No usable band-power trials were extracted for transformer decoding."

    X_tensor = np.stack(tensor_rows, axis=0).astype(np.float32)
    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    trial_qc_df = pd.DataFrame(trial_qc_rows).reset_index(drop=True)
    y = kept_df["concept_label"].astype(str).to_numpy()
    post_counts = pd.Series(y).value_counts()
    keep_after_extract = post_counts[post_counts >= min_trials].index.tolist()
    if len(keep_after_extract) < 2:
        return None, "Need at least two concepts with enough valid extracted trials."
    keep_mask = np.isin(y, keep_after_extract)
    X_tensor = X_tensor[keep_mask]
    y = y[keep_mask]
    kept_df = kept_df.loc[keep_mask].reset_index(drop=True)
    try:
        embeddings, matched_embedding_keys = resolve_trial_embeddings(kept_df, embedding_map)
    except Exception as exc:
        return None, f"Could not align trial embeddings: {exc}"
    kept_df["embedding_key"] = matched_embedding_keys

    feature_meta_rows = []
    feature_index = 0
    if feature_times is None:
        feature_times = np.asarray([], dtype=float)
    for ch_name in channel_names:
        for band_name, band_low, band_high in band_defs:
            time_iter = list(enumerate(bin_edges)) if feature_binning else list(enumerate(feature_times.tolist()))
            for time_idx, time_value in time_iter:
                if feature_binning:
                    bin_start, bin_end = time_value
                    feature = f"{ch_name}|{band_name}|{bin_start:.3f}-{bin_end:.3f}s"
                    row_extra = {
                        "bin_index": int(time_idx),
                        "bin_start_s": float(bin_start),
                        "bin_end_s": float(bin_end),
                    }
                else:
                    rel_time_s = float(time_value)
                    feature = f"{ch_name}|{band_name}|sample_{time_idx}@{rel_time_s:.4f}s"
                    row_extra = {
                        "time_index": int(time_idx),
                        "relative_time_s": rel_time_s,
                    }
                feature_meta_rows.append(
                    {
                        "feature_index": int(feature_index),
                        "feature": feature,
                        "channel": str(ch_name),
                        "band": str(band_name),
                        "band_low_hz": float(band_low),
                        "band_high_hz": float(band_high),
                        **row_extra,
                    }
                )
                feature_index += 1

    payload = {
        "X_tensor": X_tensor,
        "embeddings": embeddings.astype(np.float32),
        "y": y,
        "trial_df": kept_df,
        "trial_qc_df": trial_qc_df,
        "feature_meta_df": pd.DataFrame(feature_meta_rows),
        "channel_names": channel_names,
        "band_defs": band_defs,
        "bin_edges": bin_edges,
        "n_time_points": int(X_tensor.shape[3]),
        "feature_binning_enabled": bool(feature_binning),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "bandpass_low_hz": bandpass_low_hz,
        "bandpass_high_hz": bandpass_high_hz,
        "resample_enabled": bool(resample_enabled),
        "resample_sfreq": float(resample_sfreq) if resample_enabled and resample_sfreq is not None else np.nan,
        "group_mode": group_mode,
        "channel_mode": channel_mode,
        "feature_space": "signal",
        "baseline_correct": baseline_correct,
        "mne_epoch_reject_enabled": bool(reject_summary.get("enabled", False)),
        "mne_reject_mag_ft": reject_summary.get("mag_ft", np.nan),
        "mne_reject_grad_ft_cm": reject_summary.get("grad_ft_cm", np.nan),
        "epoch_tmin": float(epoch_tmin),
        "epoch_tmax": float(epoch_tmax),
        "semantic_power_padding_s": float(power_padding_s),
        "bandpower_baseline": power_baseline,
        "trial_outlier_z": float(trial_outlier_z),
        "exclude_trial_outliers": bool(exclude_qc_outliers),
        "dropped_trials": int(dropped_trials),
        "epoch_cache_hits": int(epoch_cache_hits),
        "epoch_cache_misses": int(epoch_cache_misses),
        "skipped_paths": skipped_paths,
        "n_runs": int(len(pd.Index(kept_df["source_fif_path"]).drop_duplicates())),
        "embedding_dict_path": os.path.abspath(os.path.expanduser(embedding_path)),
        "sfreq": float(sfreq),
    }
    return payload, None


def _l2_normalize(arr: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    norms = np.where(norms > float(eps), norms, 1.0)
    return values / norms


def _center_embedding_targets(
    embeddings: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Remove the shared semantic direction and L2-normalize residual targets."""
    normalized = _l2_normalize(embeddings)
    center_row = np.asarray(center, dtype=np.float32).reshape(1, -1)
    if normalized.shape[1] != center_row.shape[1]:
        raise ValueError("Embedding target center has the wrong dimension.")
    return _l2_normalize(normalized - center_row)


def _make_pseudo_epochs(
    X_tensor: np.ndarray,
    embeddings: np.ndarray,
    y: np.ndarray,
    trial_df: pd.DataFrame,
    indices: np.ndarray,
    *,
    trials_per_average: int,
    seed: int,
    subset: str,
    balance_classes: bool = False,
    task_stratified: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    indices = np.asarray(indices, dtype=int).reshape(-1)
    labels = np.asarray(y, dtype=str)
    pseudo_X: list[np.ndarray] = []
    pseudo_E: list[np.ndarray] = []
    pseudo_y: list[str] = []
    rows: list[dict[str, Any]] = []
    n_avg = max(1, int(trials_per_average))
    unique_labels = pd.Index(labels[indices]).drop_duplicates().tolist()

    decisions = None
    if task_stratified and n_avg in (5, 6) and isinstance(trial_df, pd.DataFrame) and not trial_df.empty:
        if "semantic_decision" in trial_df.columns:
            dec_series = trial_df["semantic_decision"].astype(str).str.strip()
            if (dec_series != "").any():
                decisions = dec_series.to_numpy()
        if decisions is None:
            try:
                from core.concept_decoding_signal import semantic_decision_label_from_trial_row
                dec_list = [semantic_decision_label_from_trial_row(row) for _, row in trial_df.iterrows()]
                dec_arr = np.asarray([str(d).strip() for d in dec_list], dtype=object)
                if np.any(dec_arr != ""):
                    decisions = dec_arr
            except Exception:
                decisions = None

    class_chunks: dict[str, list[np.ndarray]] = {}
    class_stratified: dict[str, bool] = {}
    class_trial_limit = None
    if balance_classes and unique_labels:
        class_trial_limit = min(
            int(np.sum(labels[indices] == str(label))) for label in unique_labels
        )

    for label in unique_labels:
        label_indices = indices[labels[indices] == str(label)].copy()
        if label_indices.size == 0:
            continue

        stratified_done = False
        if decisions is not None:
            lbl_decisions = decisions[label_indices]
            valid_mask = lbl_decisions != ""
            valid_indices = label_indices[valid_mask]
            valid_tasks = lbl_decisions[valid_mask]
            unique_tasks = sorted(list(dict.fromkeys(valid_tasks)))

            if len(unique_tasks) == 5:
                task_indices = {t: valid_indices[valid_tasks == t].copy() for t in unique_tasks}
                for t in unique_tasks:
                    rng.shuffle(task_indices[t])

                if n_avg == 5:
                    # Option A: Strict 5-way balanced pseudo-epochs (1 trial from each semantic decision)
                    n_tuples = min(len(task_indices[t]) for t in unique_tasks)
                    if n_tuples > 0:
                        class_chunks[label] = [
                            np.asarray([task_indices[t][j] for t in unique_tasks], dtype=int)
                            for j in range(n_tuples)
                        ]
                        class_stratified[label] = True
                        stratified_done = True
                elif n_avg == 6:
                    # Option B: Proportional 6-way balanced pseudo-epochs (2 from dominant task, 1 from each other)
                    counts = {t: len(task_indices[t]) for t in unique_tasks}
                    dominant_task = max(unique_tasks, key=lambda t: counts[t])
                    other_tasks = [t for t in unique_tasks if t != dominant_task]

                    if counts[dominant_task] >= 2 * min(counts[ot] for ot in other_tasks) - 2:
                        n_tuples = min(counts[dominant_task] // 2, min(counts[ot] for ot in other_tasks))
                        if n_tuples > 0:
                            chunks = []
                            for j in range(n_tuples):
                                chunk = [
                                    task_indices[dominant_task][2 * j],
                                    task_indices[dominant_task][2 * j + 1],
                                ] + [task_indices[ot][j] for ot in other_tasks]
                                chunks.append(np.asarray(chunk, dtype=int))
                            class_chunks[label] = chunks
                            class_stratified[label] = True
                            stratified_done = True
                    else:
                        # Fallback for 6 when no single dominant 2x task: 1 of each + round-robin 6th
                        n_tuples = min(counts[t] for t in unique_tasks)
                        if n_tuples > 0:
                            chunks = []
                            for j in range(n_tuples):
                                extra_t = unique_tasks[j % len(unique_tasks)]
                                extra_trial = task_indices[extra_t][(n_tuples + j) % len(task_indices[extra_t])]
                                chunk = [task_indices[t][j] for t in unique_tasks] + [extra_trial]
                                chunks.append(np.asarray(chunk, dtype=int))
                            class_chunks[label] = chunks
                            class_stratified[label] = True
                            stratified_done = True

        if not stratified_done:
            # Standard unstratified random chunking
            rng.shuffle(label_indices)
            if class_trial_limit is not None:
                label_indices = label_indices[:class_trial_limit]
            chunks = []
            for start in range(0, int(label_indices.size), n_avg):
                chunk = label_indices[start : start + n_avg]
                if chunk.size > 0:
                    chunks.append(chunk)
            class_chunks[label] = chunks
            class_stratified[label] = False

    # Balance number of pseudo-epochs across classes if requested
    if balance_classes and class_chunks:
        min_chunks = min(len(class_chunks[lbl]) for lbl in unique_labels if lbl in class_chunks)
        for lbl in unique_labels:
            if lbl in class_chunks:
                class_chunks[lbl] = class_chunks[lbl][:min_chunks]

    total_stratified = 0
    total_pseudo = 0
    for label in unique_labels:
        if label not in class_chunks:
            continue
        chunks = class_chunks[label]
        is_strat = class_stratified.get(label, False)
        for chunk in chunks:
            if chunk.size == 0:
                continue
            total_pseudo += 1
            if is_strat:
                total_stratified += 1
            pseudo_X.append(np.nanmean(X_tensor[chunk], axis=0))
            pseudo_E.append(np.nanmean(embeddings[chunk], axis=0))
            pseudo_y.append(str(label))
            row0 = trial_df.iloc[int(chunk[0])].to_dict()
            row_update = {
                "pseudo_epoch_subset": str(subset),
                "pseudo_epoch_label": str(label),
                "n_source_trials": int(chunk.size),
                "source_trial_indices": ",".join(str(int(v)) for v in chunk.tolist()),
                "source_trial_table_indices": ",".join(
                    str(int(v)) for v in trial_df.index.to_numpy(dtype=int)[chunk].tolist()
                ),
            }
            if decisions is not None:
                row_update["source_semantic_decisions"] = ",".join(
                    str(decisions[int(v)]) for v in chunk.tolist()
                )
            row0.update(row_update)
            rows.append(row0)

    is_overall_stratified = total_pseudo > 0 and total_stratified == total_pseudo
    if not pseudo_X:
        return {
            "X": np.empty((0,) + tuple(X_tensor.shape[1:]), dtype=np.float32),
            "embeddings": np.empty((0, embeddings.shape[1]), dtype=np.float32),
            "y": np.asarray([], dtype=str),
            "df": pd.DataFrame(),
            "class_balanced": bool(balance_classes),
            "source_trials_per_class_limit": class_trial_limit,
            "task_stratified": False,
            "trials_per_average": int(n_avg),
        }
    return {
        "X": np.stack(pseudo_X, axis=0).astype(np.float32),
        "embeddings": np.vstack(pseudo_E).astype(np.float32),
        "y": np.asarray(pseudo_y, dtype=str),
        "df": pd.DataFrame(rows).reset_index(drop=True),
        "class_balanced": bool(balance_classes),
        "source_trials_per_class_limit": class_trial_limit,
        "task_stratified": bool(is_overall_stratified),
        "trials_per_average": int(n_avg),
    }



def _import_seeg_decoder_model(seeg_decoder_src: str | None = None):
    src_root = os.path.abspath(
        os.path.expanduser(str(seeg_decoder_src or os.environ.get("CIDECOG_SEEG_DECODER_SRC") or SEEG_DECODER_SRC))
    )
    package_name = "cidecog_seeg_decoder"
    package_dir = os.path.join(src_root, "seeg_decoder")
    model_path = os.path.join(package_dir, "model.py")
    if not os.path.isfile(model_path):
        raise RuntimeError(
            "Could not find the SEEG decoder source. Set --seeg-decoder-src or "
            f"CIDECOG_SEEG_DECODER_SRC to the CIDECoG_enc-main/src directory. Checked: {src_root}"
        )
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [package_dir]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package

    for module_basename in ("semantic_codec", "model"):
        module_name = f"{package_name}.{module_basename}"
        if module_name in sys.modules:
            continue
        module_path = os.path.join(package_dir, f"{module_basename}.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load SEEG decoder module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    model_module = sys.modules[f"{package_name}.model"]
    DecoderConfig = getattr(model_module, "DecoderConfig")
    build_decoder_model = getattr(model_module, "build_decoder_model")

    return DecoderConfig, build_decoder_model


def _select_torch_device():
    import torch

    if os.environ.get("MAS_SUBSET_WORKER_CPU") == "1":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_safe_transformer_device(device, num_time, time_bins):
    """Use CPU when MPS cannot run AdaptiveAvgPool1d for this time grid."""
    if str(device).lower() != "mps":
        return device, ""
    try:
        num_time = int(num_time)
        time_bins = int(time_bins)
    except Exception:
        return device, ""
    if num_time > 0 and time_bins > 0 and num_time % time_bins == 0:
        return device, ""

    import torch

    reason = (
        "MPS AdaptiveAvgPool1d fallback: input time bins "
        f"({num_time}) are not divisible by transformer pooled time bins ({time_bins}); "
        "using CPU for the transformer."
    )
    return torch.device("cpu"), reason


def _train_transformer_embedding_model(
    train_X: np.ndarray,
    train_embeddings: np.ndarray,
    *,
    channel_names: list[str],
    model_type: str,
    max_epochs: int,
    batch_size: int,
    random_state: int,
    progress_callback: Any = None,
    seeg_decoder_src: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    DecoderConfig, build_decoder_model = _import_seeg_decoder_model(seeg_decoder_src)
    device = _select_torch_device()
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    train_X = np.asarray(train_X, dtype=np.float32)
    normalized_targets = _l2_normalize(train_embeddings)
    target_center = np.nanmean(normalized_targets, axis=0, keepdims=True).astype(
        np.float32
    )
    targets = _center_embedding_targets(normalized_targets, target_center)
    mean = np.nanmean(train_X, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(train_X, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    train_Xz = np.nan_to_num((train_X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    num_time = int(train_Xz.shape[3])
    time_bins = max(1, min(5, num_time))
    device, device_note = _mps_safe_transformer_device(device, num_time, time_bins)
    model_dim = 64
    num_heads = 4
    if model_dim % num_heads != 0:
        num_heads = 1
    metadata = {
        "names": [str(v) for v in channel_names],
        "codes": ["MEG" for _ in channel_names],
        "hemispheres": ["UNK" for _ in channel_names],
        "numbers": [idx + 1 for idx in range(len(channel_names))],
    }
    cfg = DecoderConfig(
        output_dim=int(targets.shape[1]),
        num_electrodes=int(train_Xz.shape[1]),
        num_bands=int(train_Xz.shape[2]),
        num_time=num_time,
        channels=[16, 16],
        model_type=str(model_type or "axial_transformer"),
        subject_ids=["meg"],
        channel_metadata_by_subject={"meg": metadata},
        time_bins=time_bins,
        model_dim=model_dim,
        num_transformer_layers=2,
        num_heads=num_heads,
        num_readout_tokens=2,
        dropout=0.1,
        attention_dropout=0.1,
    )
    model = build_decoder_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(train_Xz), torch.from_numpy(targets.astype(np.float32)))
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=True)
    losses: list[float] = []
    epochs = max(1, int(max_epochs))
    model.train()
    _emit_progress(
        progress_callback,
        f"training transformer on {train_Xz.shape[0]} averaged epochs for {epochs} epochs...",
    )
    for epoch_idx in range(epochs):
        running = []
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, subject_id="meg")
            loss = 1.0 - F.cosine_similarity(pred, yb, dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running.append(float(loss.detach().cpu().item()))
        epoch_loss = float(np.nanmean(running)) if running else np.nan
        losses.append(epoch_loss)
        _emit_progress(
            progress_callback,
            f"training transformer epoch {epoch_idx + 1}/{epochs}: loss={epoch_loss:.4f}",
        )

    meta = {
        "device": str(device),
        "train_losses": losses,
        "final_train_loss": float(losses[-1]) if losses else np.nan,
        "input_mean": mean,
        "input_std": std,
        "model_type": str(getattr(model, "model_type", model_type)),
        "torch_epochs": int(epochs),
        "torch_batch_size": int(max(1, int(batch_size))),
        "target_normalized": True,
        "embedding_target_centered": True,
        "embedding_target_center": target_center,
        "time_bins": int(time_bins),
        "device_note": str(device_note),
        "seeg_decoder_src": str(os.path.abspath(os.path.expanduser(seeg_decoder_src or os.environ.get("CIDECOG_SEEG_DECODER_SRC") or SEEG_DECODER_SRC))),
    }
    return model, meta


def _predict_embeddings(model: Any, X: np.ndarray, train_meta: dict[str, Any]) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    mean = np.asarray(train_meta["input_mean"], dtype=np.float32)
    std = np.asarray(train_meta["input_std"], dtype=np.float32)
    Xz = np.nan_to_num((np.asarray(X, dtype=np.float32) - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, int(Xz.shape[0]), 64):
            xb = torch.from_numpy(Xz[start : start + 64]).to(device=device, dtype=torch.float32)
            pred = model(xb, subject_id="meg")
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
    if not preds:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(preds).astype(np.float32)


def _flatten_embedding_features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError(f"Embedding decoder needs at least a 2D input array, got {values.shape}.")
    return np.nan_to_num(
        values.reshape(values.shape[0], -1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _standardize_embedding_features(
    train_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = _flatten_embedding_features(train_X)
    mean = np.nanmean(flat, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(flat, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    standardized = np.nan_to_num(
        (flat - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    return standardized, mean, std


def _transform_flat_embedding_features(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    flat = _flatten_embedding_features(X)
    return np.nan_to_num(
        (flat - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _fit_embedding_input_transform(
    train_X: np.ndarray,
    *,
    pca_components: int = 0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, int]:
    """Standardize flattened features and optionally fit train-only PCA."""
    standardized, mean, std = _standardize_embedding_features(train_X)
    try:
        requested = max(0, int(pca_components))
    except Exception:
        requested = 0
    max_components = max(
        0,
        min(int(standardized.shape[1]), int(standardized.shape[0]) - 1),
    )
    effective = min(requested, max_components)
    pca = None
    transformed = standardized
    if effective > 0 and effective < int(standardized.shape[1]):
        from sklearn.decomposition import PCA

        pca = PCA(
            n_components=int(effective),
            svd_solver="randomized",
            random_state=int(random_state),
        )
        transformed = np.asarray(pca.fit_transform(standardized), dtype=np.float32)
    else:
        effective = int(standardized.shape[1])
    return transformed, mean, std, pca, int(effective)


def _apply_embedding_input_transform(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pca: Any = None,
) -> np.ndarray:
    standardized = _transform_flat_embedding_features(X, mean, std)
    if pca is None:
        return standardized
    return np.asarray(pca.transform(standardized), dtype=np.float32)


def _fit_embedding_target_transform(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    class_names: list[str],
    *,
    svd_components: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int, float]:
    """Center semantic targets and optionally retain their train-only SVD span."""
    labels = np.asarray(train_labels, dtype=str)
    normalized_targets = _l2_normalize(train_embeddings)
    raw_prototypes = _class_prototypes(labels, normalized_targets, class_names)
    target_center = np.nanmean(raw_prototypes, axis=0, keepdims=True).astype(
        np.float32
    )
    centered_targets = _center_embedding_targets(normalized_targets, target_center)
    centered_prototypes = _class_prototypes(
        labels, centered_targets, class_names
    ).astype(np.float32)
    singular_values = np.linalg.svd(
        centered_prototypes, full_matrices=False, compute_uv=False
    )
    if singular_values.size:
        tolerance = float(
            max(centered_prototypes.shape)
            * np.finfo(np.float32).eps
            * singular_values[0]
        )
        semantic_rank = int(np.sum(singular_values > tolerance))
    else:
        semantic_rank = 0
    try:
        requested = max(0, int(svd_components))
    except Exception:
        requested = 0
    projection = None
    effective = int(centered_targets.shape[1])
    explained_variance = 1.0
    if requested > 0 and semantic_rank > 0:
        effective = min(requested, semantic_rank)
        _u, singular_values_full, vt = np.linalg.svd(
            centered_prototypes, full_matrices=False
        )
        projection = np.asarray(vt[:effective].T, dtype=np.float32)
        centered_targets = _l2_normalize(centered_targets @ projection)
        total_variance = float(np.sum(np.square(singular_values_full)))
        explained_variance = (
            float(np.sum(np.square(singular_values_full[:effective])) / total_variance)
            if total_variance > 0.0
            else 1.0
        )
    return (
        centered_targets.astype(np.float32),
        target_center,
        projection,
        int(effective),
        int(semantic_rank),
        float(explained_variance),
    )


def _transform_embedding_targets(
    embeddings: np.ndarray,
    target_center: np.ndarray,
    target_projection: np.ndarray | None = None,
) -> np.ndarray:
    targets = _center_embedding_targets(embeddings, target_center)
    if target_projection is not None:
        targets = _l2_normalize(
            targets @ np.asarray(target_projection, dtype=np.float32)
        )
    return targets.astype(np.float32)


def _embedding_feature_importance(
    weights: np.ndarray,
    *,
    original_feature_count: int,
    input_pca: Any = None,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        return np.full(int(original_feature_count), np.nan, dtype=np.float32)
    if input_pca is not None:
        components = np.asarray(getattr(input_pca, "components_", []), dtype=np.float32)
        if components.ndim != 2 or values.shape[1] != components.shape[0]:
            return np.full(int(original_feature_count), np.nan, dtype=np.float32)
        values = values @ components
    if values.shape[1] != int(original_feature_count):
        return np.full(int(original_feature_count), np.nan, dtype=np.float32)
    return np.nanmean(np.abs(values), axis=0).astype(np.float32)


def _train_ridge_embedding_model(
    train_X: np.ndarray,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    *,
    class_names: list[str],
    alpha: float,
    input_pca_components: int = 0,
    target_svd_components: int = 0,
    random_state: int = 42,
    progress_callback: Any = None,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.linear_model import Ridge

    original_feature_count = int(_flatten_embedding_features(train_X).shape[1])
    Xz, mean, std, input_pca, effective_input_dim = _fit_embedding_input_transform(
        train_X,
        pca_components=input_pca_components,
        random_state=random_state,
    )
    (
        targets,
        target_center,
        target_projection,
        effective_target_dim,
        semantic_rank,
        semantic_explained_variance,
    ) = _fit_embedding_target_transform(
        train_embeddings,
        train_labels,
        class_names,
        svd_components=target_svd_components,
    )
    alpha = max(0.0, float(alpha))
    _emit_progress(
        progress_callback,
        (
            f"training linear ridge embedding decoder on {Xz.shape[0]} averaged epochs "
            f"({original_feature_count}->{effective_input_dim} neural features, "
            f"{train_embeddings.shape[1]}->{effective_target_dim} semantic dimensions)..."
        ),
    )
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(Xz, targets)
    train_pred = np.asarray(model.predict(Xz), dtype=np.float32)
    train_loss = float(
        np.nanmean(1.0 - np.sum(_l2_normalize(train_pred) * targets, axis=1))
    )
    coef = np.asarray(getattr(model, "coef_", []), dtype=np.float32)
    intercept = np.asarray(getattr(model, "intercept_", []), dtype=np.float32)
    feature_importance = _embedding_feature_importance(
        coef,
        original_feature_count=original_feature_count,
        input_pca=input_pca,
    )
    return model, {
        "input_mean": mean,
        "input_std": std,
        "input_pca": input_pca,
        "input_pca_requested_components": int(max(0, input_pca_components)),
        "input_pca_components": int(effective_input_dim),
        "input_pca_explained_variance": float(
            np.sum(getattr(input_pca, "explained_variance_ratio_", [1.0]))
        ),
        "input_original_dim": int(original_feature_count),
        "ridge_alpha": float(alpha),
        "final_train_loss": train_loss,
        "feature_importance": feature_importance,
        "train_losses": [train_loss],
        "training_history": [],
        "device": "cpu",
        "target_normalized": True,
        "embedding_target_centered": True,
        "embedding_target_center": target_center,
        "embedding_target_projection": target_projection,
        "embedding_target_svd_requested_components": int(max(0, target_svd_components)),
        "embedding_target_svd_components": int(effective_target_dim),
        "embedding_target_rank": int(semantic_rank),
        "embedding_target_svd_explained_variance": float(semantic_explained_variance),
        "embedding_dim": int(effective_target_dim),
        "model_parameter_count": int(coef.size + intercept.size),
    }


def _predict_flat_embedding_model(model: Any, X: np.ndarray, train_meta: dict[str, Any]) -> np.ndarray:
    Xz = _apply_embedding_input_transform(
        X,
        train_meta["input_mean"],
        train_meta["input_std"],
        train_meta.get("input_pca"),
    )
    if hasattr(model, "predict"):
        return np.asarray(model.predict(Xz), dtype=np.float32)

    import torch

    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(Xz.shape[0]), 128):
            xb = torch.from_numpy(Xz[start : start + 128]).to(device=device, dtype=torch.float32)
            rows.append(model(xb).detach().cpu().numpy().astype(np.float32))
    if not rows:
        output_dim = int(train_meta.get("embedding_dim", 0) or 0)
        return np.empty((0, output_dim), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)


def _embedding_targets_for_scoring(
    embeddings: np.ndarray,
    train_meta: dict[str, Any],
) -> np.ndarray:
    """Apply the target-space transform learned by an embedding decoder."""
    if bool(train_meta.get("embedding_target_centered", False)):
        center = np.asarray(
            train_meta.get("embedding_target_center", []), dtype=np.float32
        )
        if center.size:
            projection = train_meta.get("embedding_target_projection")
            if projection is not None:
                projection = np.asarray(projection, dtype=np.float32)
            return _transform_embedding_targets(embeddings, center, projection)
    return _l2_normalize(embeddings)


def _embedding_retrieval_metrics(
    predicted_embeddings: np.ndarray,
    y_true: np.ndarray,
    prototypes: np.ndarray,
    class_names: list[str],
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=str)
    if y_true.size == 0:
        return np.nan, np.nan
    scores = _l2_normalize(predicted_embeddings) @ _l2_normalize(prototypes).T
    y_pred = np.asarray([class_names[int(idx)] for idx in np.argmax(scores, axis=1)], dtype=str)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
    )


def _train_torch_embedding_model(
    train_X: np.ndarray,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    *,
    validation_X: np.ndarray | None,
    validation_embeddings: np.ndarray | None,
    validation_labels: np.ndarray | None,
    class_names: list[str],
    max_epochs: int,
    batch_size: int,
    hidden_layer_sizes: tuple[int, ...],
    dropout: float,
    weight_decay: float,
    random_state: int,
    options: dict[str, Any] | None = None,
    learning_rate: float = 1e-3,
    contrastive_weight_override: float | None = None,
    contrastive_temperature_override: float | None = None,
    model_key: str = "embedding_mlp",
    model_display_name: str = "MLP embedding",
    restore_best_option: str = "mlp_restore_best_validation",
    dynamic_epoch_data_factory: Any = None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))
    device = _select_torch_device()
    labels = np.asarray(train_labels, dtype=str)
    try:
        input_pca_requested = max(
            0, int((options or {}).get("embedding_input_pca_components", 0))
        )
    except Exception:
        input_pca_requested = 0
    try:
        target_svd_requested = max(
            0, int((options or {}).get("embedding_target_svd_components", 0))
        )
    except Exception:
        target_svd_requested = 0
    original_feature_count = int(_flatten_embedding_features(train_X).shape[1])
    Xz, mean, std, input_pca, effective_input_dim = _fit_embedding_input_transform(
        train_X,
        pca_components=input_pca_requested,
        random_state=random_state,
    )
    (
        targets,
        target_center,
        target_projection,
        effective_target_dim,
        semantic_rank,
        semantic_explained_variance,
    ) = _fit_embedding_target_transform(
        train_embeddings,
        labels,
        class_names,
        svd_components=target_svd_requested,
    )
    class_to_index = {str(label): idx for idx, label in enumerate(class_names)}
    target_class_indices = np.asarray(
        [class_to_index[str(label)] for label in labels], dtype=np.int64
    )
    val_Xz = None
    val_targets = None
    val_labels = np.asarray([], dtype=str)
    val_class_indices = np.asarray([], dtype=np.int64)
    if validation_X is not None and np.asarray(validation_X).shape[0] > 0:
        val_Xz = _apply_embedding_input_transform(
            validation_X, mean, std, input_pca
        )
        val_targets = _transform_embedding_targets(
            np.asarray(validation_embeddings, dtype=np.float32),
            target_center,
            target_projection,
        )
        val_labels = np.asarray(validation_labels, dtype=str)
        val_class_indices = np.asarray(
            [class_to_index[str(label)] for label in val_labels], dtype=np.int64
        )

    hidden = tuple(max(1, int(value)) for value in hidden_layer_sizes if int(value) > 0)
    dims = (int(Xz.shape[1]),) + hidden + (int(targets.shape[1]),)
    layers: list[Any] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(p=min(0.9, max(0.0, float(dropout)))))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(1e-8, float(learning_rate)),
        weight_decay=max(0.0, float(weight_decay)),
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xz),
            torch.from_numpy(targets.astype(np.float32)),
            torch.from_numpy(target_class_indices),
        ),
        batch_size=max(1, min(int(batch_size), int(Xz.shape[0]))),
        shuffle=True,
    )
    prototypes = _class_prototypes(labels, targets, class_names)
    prototype_tensor = torch.from_numpy(prototypes).to(
        device=device, dtype=torch.float32
    )
    try:
        contrastive_weight = float(
            contrastive_weight_override
            if contrastive_weight_override is not None
            else (options or {}).get("embedding_mlp_contrastive_weight", 0.75)
        )
    except Exception:
        contrastive_weight = 0.75
    contrastive_weight = min(1.0, max(0.0, contrastive_weight))
    cosine_weight = 1.0 - contrastive_weight
    try:
        contrastive_temperature = float(
            contrastive_temperature_override
            if contrastive_temperature_override is not None
            else (options or {}).get("embedding_mlp_temperature", 0.07)
        )
    except Exception:
        contrastive_temperature = 0.07
    contrastive_temperature = min(1.0, max(0.01, contrastive_temperature))
    epochs = max(1, int(max_epochs))
    try:
        early_stopping_patience = max(
            0, int((options or {}).get("embedding_early_stopping_patience", 0))
        )
    except Exception:
        early_stopping_patience = 0
    try:
        early_stopping_min_delta = max(
            0.0,
            float((options or {}).get("embedding_early_stopping_min_delta", 1e-4)),
        )
    except Exception:
        early_stopping_min_delta = 1e-4
    history: list[dict[str, Any]] = []
    best_state = None
    best_validation_accuracy = -np.inf
    best_validation_selection = -np.inf
    best_validation_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    dynamic_training_enabled = callable(dynamic_epoch_data_factory)
    _emit_progress(
        options,
        (
            f"training {model_display_name} decoder on {Xz.shape[0]} averaged epochs "
            f"for up to {epochs} epochs ({original_feature_count}->{effective_input_dim} "
            f"neural features, {train_embeddings.shape[1]}->{effective_target_dim} "
            f"semantic dimensions, early-stop patience={early_stopping_patience or 'off'}, "
            f"dynamic averaging={'on' if dynamic_training_enabled else 'off'})..."
        ),
    )

    for epoch_idx in range(epochs):
        epoch_loader = loader
        if dynamic_training_enabled and epoch_idx > 0:
            epoch_payload = dynamic_epoch_data_factory(int(epoch_idx + 1))
            if isinstance(epoch_payload, dict):
                epoch_X = np.asarray(epoch_payload.get("X", []), dtype=np.float32)
                epoch_embeddings = np.asarray(
                    epoch_payload.get("embeddings", []), dtype=np.float32
                )
                epoch_labels = np.asarray(epoch_payload.get("y", []), dtype=str)
                if (
                    epoch_X.shape[0] > 0
                    and epoch_embeddings.shape[0] == epoch_X.shape[0]
                    and epoch_labels.size == epoch_X.shape[0]
                ):
                    epoch_Xz = _apply_embedding_input_transform(
                        epoch_X, mean, std, input_pca
                    )
                    epoch_targets = _transform_embedding_targets(
                        epoch_embeddings, target_center, target_projection
                    )
                    epoch_class_indices = np.asarray(
                        [class_to_index[str(label)] for label in epoch_labels],
                        dtype=np.int64,
                    )
                    epoch_loader = DataLoader(
                        TensorDataset(
                            torch.from_numpy(epoch_Xz),
                            torch.from_numpy(epoch_targets),
                            torch.from_numpy(epoch_class_indices),
                        ),
                        batch_size=max(
                            1, min(int(batch_size), int(epoch_Xz.shape[0]))
                        ),
                        shuffle=True,
                    )
        model.train()
        batch_losses = []
        for xb, yb, class_idx in epoch_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            class_idx = class_idx.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(xb)
            cosine_loss = 1.0 - F.cosine_similarity(predicted, yb, dim=1).mean()
            logits = F.normalize(predicted, dim=1) @ prototype_tensor.T
            contrastive_loss = F.cross_entropy(
                logits / contrastive_temperature, class_idx
            )
            loss = (
                cosine_weight * cosine_loss
                + contrastive_weight * contrastive_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        improved_validation = False
        model.eval()
        with torch.no_grad():
            train_pred_tensor = model(
                torch.from_numpy(Xz).to(device=device, dtype=torch.float32)
            )
            train_target_tensor = torch.from_numpy(targets).to(
                device=device, dtype=torch.float32
            )
            train_class_tensor = torch.from_numpy(target_class_indices).to(
                device=device, dtype=torch.long
            )
            train_cosine_loss = float(
                (
                    1.0
                    - F.cosine_similarity(
                        train_pred_tensor, train_target_tensor, dim=1
                    ).mean()
                )
                .cpu()
                .item()
            )
            train_contrastive_loss = float(
                F.cross_entropy(
                    (F.normalize(train_pred_tensor, dim=1) @ prototype_tensor.T)
                    / contrastive_temperature,
                    train_class_tensor,
                )
                .cpu()
                .item()
            )
            train_loss = float(
                cosine_weight * train_cosine_loss
                + contrastive_weight * train_contrastive_loss
            )
            train_pred = train_pred_tensor.cpu().numpy()
            train_acc, train_bal = _embedding_retrieval_metrics(
                train_pred, labels, prototypes, class_names
            )
            validation_loss = np.nan
            validation_cosine_loss = np.nan
            validation_contrastive_loss = np.nan
            validation_acc = np.nan
            validation_bal = np.nan
            if val_Xz is not None and val_targets is not None and val_labels.size:
                val_pred_tensor = model(
                    torch.from_numpy(val_Xz).to(device=device, dtype=torch.float32)
                )
                val_target_tensor = torch.from_numpy(val_targets).to(
                    device=device, dtype=torch.float32
                )
                val_class_tensor = torch.from_numpy(val_class_indices).to(
                    device=device, dtype=torch.long
                )
                validation_cosine_loss = float(
                    (
                        1.0
                        - F.cosine_similarity(
                            val_pred_tensor, val_target_tensor, dim=1
                        ).mean()
                    )
                    .cpu()
                    .item()
                )
                validation_contrastive_loss = float(
                    F.cross_entropy(
                        (F.normalize(val_pred_tensor, dim=1) @ prototype_tensor.T)
                        / contrastive_temperature,
                        val_class_tensor,
                    )
                    .cpu()
                    .item()
                )
                validation_loss = float(
                    cosine_weight * validation_cosine_loss
                    + contrastive_weight * validation_contrastive_loss
                )
                val_pred = val_pred_tensor.cpu().numpy()
                validation_acc, validation_bal = _embedding_retrieval_metrics(
                    val_pred, val_labels, prototypes, class_names
                )
                validation_metric = str(
                    (options or {}).get("embedding_validation_metric", "top1") or "top1"
                ).strip().lower()
                selection_score = validation_acc
                if validation_metric == "balanced_top1":
                    selection_score = validation_bal
                elif validation_metric in {"mpr", "balanced_mpr"}:
                    validation_mpr, validation_percentile_ranks, _ = _mean_percentile_rank(
                        _l2_normalize(val_pred) @ _l2_normalize(prototypes).T,
                        val_labels, class_names,
                    )
                    selection_score = (
                        _balanced_percentile_rank(
                            validation_percentile_ranks, val_labels, class_names
                        )
                        if validation_metric == "balanced_mpr"
                        else validation_mpr
                    )
                improved_accuracy = (
                    np.isfinite(selection_score)
                    and selection_score > best_validation_selection
                )
                tied_accuracy_lower_loss = (
                    np.isfinite(selection_score)
                    and np.isclose(selection_score, best_validation_selection)
                    and np.isfinite(validation_loss)
                    and validation_loss
                    < best_validation_loss - early_stopping_min_delta
                )
                if improved_accuracy or tied_accuracy_lower_loss:
                    improved_validation = True
                    best_validation_accuracy = float(validation_acc)
                    best_validation_selection = float(selection_score)
                    best_validation_loss = float(validation_loss)
                    best_epoch = int(epoch_idx + 1)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

        metric = {
            "model": str(model_key),
            "series": 1,
            "series_label": f"{model_display_name} fit",
            "epoch": int(epoch_idx + 1),
            "epochs": int(epochs),
            "train_accuracy": train_acc,
            "train_balanced_accuracy": train_bal,
            "validation_accuracy": validation_acc,
            "validation_balanced_accuracy": validation_bal,
            "train_loss": train_loss,
            "train_cosine_loss": train_cosine_loss,
            "train_contrastive_loss": train_contrastive_loss,
            "validation_loss": validation_loss,
            "validation_cosine_loss": validation_cosine_loss,
            "validation_contrastive_loss": validation_contrastive_loss,
            "chance_accuracy": float(1.0 / max(1, len(class_names))),
            "n_candidates": int(len(class_names)),
            "n_train": int(Xz.shape[0]),
            "n_validation": int(val_labels.size),
        }
        history.append(metric)
        _emit_training_metric(options, metric)
        _emit_progress(
            options,
            (
                f"training {model_display_name} epoch {epoch_idx + 1}/{epochs}: "
                f"loss={train_loss:.4f}, retrieval_accuracy={train_acc:.3f}"
                + (
                    f", val_accuracy={validation_acc:.3f}"
                    if np.isfinite(validation_acc)
                    else ""
                )
            ),
        )

        if early_stopping_patience > 0 and val_Xz is not None and val_labels.size:
            if improved_validation:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                early_stopped = True
                _emit_progress(
                    options,
                    (
                        f"early stopping {model_display_name} after {epoch_idx + 1} epochs; "
                        f"best validation epoch={best_epoch}."
                    ),
                )
                break

    restored_best = bool(
        best_state is not None
        and bool((options or {}).get(restore_best_option, True))
    )
    if restored_best:
        model.load_state_dict(best_state)
    first_linear = next((module for module in model if isinstance(module, nn.Linear)), None)
    feature_importance = _embedding_feature_importance(
        (
            first_linear.weight.detach().cpu().numpy()
            if first_linear is not None
            else np.asarray([])
        ),
        original_feature_count=original_feature_count,
        input_pca=input_pca,
    )
    last_epoch_train_loss = float(history[-1]["train_loss"]) if history else np.nan
    selected_epoch = int(best_epoch if restored_best and best_epoch > 0 else len(history))
    final_train_loss = (
        float(history[selected_epoch - 1]["train_loss"])
        if selected_epoch > 0 and selected_epoch <= len(history)
        else last_epoch_train_loss
    )
    return model, {
        "input_mean": mean,
        "input_std": std,
        "input_pca": input_pca,
        "input_pca_requested_components": int(input_pca_requested),
        "input_pca_components": int(effective_input_dim),
        "input_pca_explained_variance": float(
            np.sum(getattr(input_pca, "explained_variance_ratio_", [1.0]))
        ),
        "input_original_dim": int(original_feature_count),
        "embedding_dim": int(targets.shape[1]),
        "device": str(device),
        "torch_epochs": int(len(history)),
        "torch_epochs_requested": int(epochs),
        "torch_batch_size": int(max(1, min(int(batch_size), int(Xz.shape[0])))),
        "hidden_layer_sizes": hidden,
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "learning_rate": float(learning_rate),
        "train_losses": [float(row["train_loss"]) for row in history],
        "training_history": history,
        "final_train_loss": final_train_loss,
        "last_epoch_train_loss": last_epoch_train_loss,
        "selected_epoch": selected_epoch,
        "best_validation_epoch": int(best_epoch),
        "best_validation_accuracy": (
            float(best_validation_accuracy) if np.isfinite(best_validation_accuracy) else np.nan
        ),
        "best_validation_loss": (
            float(best_validation_loss) if np.isfinite(best_validation_loss) else np.nan
        ),
        "n_validation": int(val_labels.size),
        "feature_importance": feature_importance,
        "target_normalized": True,
        "embedding_target_centered": True,
        "embedding_target_center": target_center,
        "embedding_target_projection": target_projection,
        "embedding_target_svd_requested_components": int(target_svd_requested),
        "embedding_target_svd_components": int(effective_target_dim),
        "embedding_target_rank": int(semantic_rank),
        "embedding_target_svd_explained_variance": float(semantic_explained_variance),
        "contrastive_weight": float(contrastive_weight),
        "cosine_weight": float(cosine_weight),
        "contrastive_temperature": float(contrastive_temperature),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopped": bool(early_stopped),
        "dynamic_pseudo_epochs": bool(dynamic_training_enabled),
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }


def _class_prototypes(labels: np.ndarray, embeddings: np.ndarray, class_names: list[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=str)
    embeddings = _l2_normalize(embeddings)
    prototypes = []
    for label in class_names:
        mask = labels == str(label)
        if not np.any(mask):
            prototypes.append(np.zeros(embeddings.shape[1], dtype=np.float32))
            continue
        prototypes.append(np.nanmean(embeddings[mask], axis=0))
    return _l2_normalize(np.vstack(prototypes).astype(np.float32))


def _mean_percentile_rank(
    scores: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return CIDECoG-style MPR where 1.0 is closest and 0.0 is farthest."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=str)
    class_to_idx = {str(label): idx for idx, label in enumerate(class_names)}
    n_trials = int(y_true.size)
    percentile_ranks = np.full(n_trials, np.nan, dtype=float)
    rank_positions = np.full(n_trials, np.nan, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != n_trials or scores.shape[1] == 0:
        return np.nan, percentile_ranks, rank_positions

    for row_idx, label in enumerate(y_true):
        true_idx = class_to_idx.get(str(label))
        if true_idx is None or true_idx >= scores.shape[1]:
            continue
        row = np.asarray(scores[row_idx], dtype=float)
        true_score = float(row[true_idx])
        if not np.isfinite(true_score):
            continue
        finite = np.isfinite(row)
        n_candidates = int(np.sum(finite))
        if n_candidates <= 1:
            percentile_ranks[row_idx] = 1.0
            rank_positions[row_idx] = 1.0
            continue
        finite_scores = row[finite]
        greater = int(np.sum(finite_scores > true_score))
        tied = int(np.sum(finite_scores == true_score))
        avg_rank_zero_based = float(greater + max(0, tied - 1) / 2.0)
        rank_positions[row_idx] = avg_rank_zero_based + 1.0
        percentile_ranks[row_idx] = 1.0 - (avg_rank_zero_based / float(n_candidates - 1))

    return (
        float(np.nanmean(percentile_ranks)) if np.any(np.isfinite(percentile_ranks)) else np.nan,
        percentile_ranks,
        rank_positions,
    )


def _balanced_percentile_rank(percentile_ranks: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> float:
    percentile_ranks = np.asarray(percentile_ranks, dtype=float)
    y_true = np.asarray(y_true, dtype=str)
    per_class = []
    for label in class_names:
        mask = y_true == str(label)
        if np.any(mask):
            values = percentile_ranks[mask]
            values = values[np.isfinite(values)]
            if values.size:
                per_class.append(float(np.mean(values)))
    return float(np.nanmean(per_class)) if per_class else np.nan


def _label_permutation_baseline(
    scores: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    observed_accuracy: float,
    observed_balanced_accuracy: float,
    n_permutations: int,
) -> dict[str, Any]:
    try:
        n_permutations = int(n_permutations)
    except Exception:
        n_permutations = 0
    if n_permutations <= 0:
        return {
            "permutation_df": pd.DataFrame(),
            "permutation_n_completed": 0,
            "permutation_mean_accuracy": np.nan,
            "permutation_std_accuracy": np.nan,
            "permutation_max_accuracy": np.nan,
            "permutation_mean_balanced_accuracy": np.nan,
            "permutation_std_balanced_accuracy": np.nan,
            "permutation_max_balanced_accuracy": np.nan,
            "permutation_p_value_accuracy": np.nan,
            "permutation_p_value_balanced_accuracy": np.nan,
        }
    rng = np.random.default_rng(991)
    y_test = np.asarray(y_test, dtype=str)
    rows = []
    for perm_idx in range(int(n_permutations)):
        shuffled = y_test.copy()
        rng.shuffle(shuffled)
        perm_mpr, perm_percentile_ranks, _ = _mean_percentile_rank(scores, shuffled, class_names)
        perm_balanced_mpr = _balanced_percentile_rank(perm_percentile_ranks, shuffled, class_names)
        rows.append(
            {
                "permutation": int(perm_idx + 1),
                "decoder": "Transformer label-shuffle baseline",
                "metric": "MPR",
                "accuracy": float(perm_mpr),
                "balanced_accuracy": float(perm_balanced_mpr),
                "mpr": float(perm_mpr),
                "balanced_mpr": float(perm_balanced_mpr),
                "n_test": int(y_test.size),
            }
        )
    df = pd.DataFrame(rows)
    acc = pd.to_numeric(df["accuracy"], errors="coerce").to_numpy(dtype=float)
    bal = pd.to_numeric(df["balanced_accuracy"], errors="coerce").to_numpy(dtype=float)
    return {
        "permutation_df": df,
        "permutation_n_completed": int(len(df)),
        "permutation_mean_accuracy": float(np.nanmean(acc)) if acc.size else np.nan,
        "permutation_std_accuracy": float(np.nanstd(acc)) if acc.size else np.nan,
        "permutation_max_accuracy": float(np.nanmax(acc)) if acc.size else np.nan,
        "permutation_mean_balanced_accuracy": float(np.nanmean(bal)) if bal.size else np.nan,
        "permutation_std_balanced_accuracy": float(np.nanstd(bal)) if bal.size else np.nan,
        "permutation_max_balanced_accuracy": float(np.nanmax(bal)) if bal.size else np.nan,
        "permutation_p_value_accuracy": (
            float((1 + np.sum(acc >= float(observed_accuracy))) / (acc.size + 1)) if acc.size else np.nan
        ),
        "permutation_p_value_balanced_accuracy": (
            float((1 + np.sum(bal >= float(observed_balanced_accuracy))) / (bal.size + 1)) if bal.size else np.nan
        ),
    }


def compute_transformer_decoding_from_payload(context: Any, options: dict[str, Any], feature_payload: dict[str, Any]):
    del context
    training_seed = int(options.get("training_seed", 42))
    decoder_key = semantic_embedding_decoder_kind(options.get("decoder_type", "transformer"))
    model_training_key = {
        "zero_shot": "embedding_ridge",
        "zero_shot_contrastive": "embedding_contrastive",
        "zero_shot_mlp": "embedding_mlp",
    }.get(decoder_key, decoder_key)
    feature_space = str(feature_payload.get("feature_space", "signal") or "signal").strip().lower()
    feature_space = "source" if feature_space in {"source", "source_space", "source-space", "roi", "parcels"} else "signal"
    if decoder_key == "transformer" and feature_space != "signal":
        return None, "Transformer embedding decoding requires signal-space band-power features."
    space_label = "source-space" if feature_space == "source" else "signal-space"
    decoder_label_map = {
        "transformer": f"{space_label} transformer embedding",
        "embedding_ridge": f"{space_label} linear ridge embedding",
        "embedding_contrastive": f"{space_label} class-aware contrastive linear embedding",
        "embedding_mlp": f"{space_label} MLP embedding",
        "zero_shot": f"{space_label} zero-shot linear ridge embedding",
        "zero_shot_contrastive": f"{space_label} zero-shot class-aware contrastive linear embedding",
        "zero_shot_mlp": f"{space_label} zero-shot MLP embedding",
    }
    decoder_label = decoder_label_map.get(decoder_key, f"{space_label} {decoder_key}")
    X_values = feature_payload.get("X_tensor")
    if X_values is None:
        X_values = feature_payload.get("X")
    if X_values is None:
        return None, "Semantic embedding decoding requires a trial feature matrix."
    X_tensor = np.asarray(X_values, dtype=np.float32)
    embeddings = np.asarray(feature_payload["embeddings"], dtype=np.float32)
    y = np.asarray(feature_payload["y"], dtype=str)
    trial_df = feature_payload["trial_df"].copy()
    try:
        instruction_targets = load_instruction_targets(
            str(feature_payload.get("embedding_dict_path", "") or options.get("embedding_dict_path", "")),
            embeddings.shape[1],
        )
    except Exception as exc:
        return None, f"Could not load instruction targets: {exc}"
    if X_tensor.ndim < 2:
        return None, f"Semantic embedding features must be at least 2D, got {X_tensor.shape}."
    if not (X_tensor.shape[0] == embeddings.shape[0] == y.size == len(trial_df)):
        return None, "Semantic embedding features, targets, labels, and trial rows are inconsistent."
    trial_qc_df = feature_payload.get("trial_qc_df", pd.DataFrame())
    if not isinstance(trial_qc_df, pd.DataFrame):
        trial_qc_df = pd.DataFrame()

    try:
        test_fraction = float(options.get("test_fraction", 0.2))
    except Exception:
        test_fraction = 0.2
    test_fraction = min(0.5, max(0.05, test_fraction))
    try:
        n_permutations = max(0, int(options.get("n_permutations", 10)))
    except Exception:
        n_permutations = 10
    try:
        trials_per_average = max(1, int(
            options.get(
                "avg_trials",
                options.get(
                    "trials_per_average",
                    options.get(
                        "pseudo_epoch_trials",
                        options.get("transformer_trials_per_average", 5),
                    ),
                ),
            )
        ))
    except Exception:
        trials_per_average = 5
    try:
        max_epochs = max(1, int(
            options.get("epochs", options.get("mlp_epochs", options.get("transformer_epochs", 50)))
        ))
    except Exception:
        max_epochs = 50
    try:
        batch_size = max(1, int(
            options.get("batch_size", options.get("transformer_batch_size", 16))
        ))
    except Exception:
        batch_size = 16
    model_type = str(options.get("model_type", options.get("transformer_model_type", "axial_transformer")) or "axial_transformer")
    progress_callback = options.get("progress_callback")
    seeg_decoder_src = str(options.get("seeg_decoder_src", "") or "").strip() or None

    try:
        validation_fraction = float(
            options.get("mlp_validation_fraction", options.get("validation_fraction", 0.2))
        )
    except Exception:
        validation_fraction = 0.2
    validation_fraction = min(0.5, max(0.0, validation_fraction))
    try:
        ridge_alpha = max(0.0, float(options.get("embedding_ridge_alpha", 1.0)))
    except Exception:
        ridge_alpha = 1.0
    try:
        mlp_hidden_1 = max(1, int(options.get("mlp_hidden_1", 64)))
    except Exception:
        mlp_hidden_1 = 64
    try:
        mlp_hidden_2 = max(0, int(options.get("mlp_hidden_2", 32)))
    except Exception:
        mlp_hidden_2 = 32
    try:
        mlp_dropout = min(0.9, max(0.0, float(options.get("mlp_dropout", 0.35))))
    except Exception:
        mlp_dropout = 0.35
    try:
        mlp_weight_decay = max(0.0, float(options.get("mlp_weight_decay", 1e-3)))
    except Exception:
        mlp_weight_decay = 1e-3
    try:
        contrastive_temperature = min(
            1.0,
            max(0.01, float(options.get("embedding_contrastive_temperature", 0.07))),
        )
    except Exception:
        contrastive_temperature = 0.07
    try:
        contrastive_weight_decay = max(
            0.0, float(options.get("embedding_contrastive_weight_decay", 1e-2))
        )
    except Exception:
        contrastive_weight_decay = 1e-2
    try:
        contrastive_learning_rate = max(
            1e-8, float(options.get("embedding_contrastive_learning_rate", 1e-3))
        )
    except Exception:
        contrastive_learning_rate = 1e-3
    try:
        input_pca_components = max(
            0, int(options.get("embedding_input_pca_components", 0))
        )
    except Exception:
        input_pca_components = 0
    try:
        target_svd_components = max(
            0, int(options.get("embedding_target_svd_components", 0))
        )
    except Exception:
        target_svd_components = 0
    dynamic_pseudo_epochs = bool(
        options.get("embedding_dynamic_pseudo_epochs", False)
    )
    balance_training_classes = bool(options.get("balance_training_classes", False))

    requested_split_mode = str(options.get("split_mode", "random") or "random").strip().lower()
    is_zero_shot = (
        decoder_key in {"zero_shot", "zero_shot_contrastive", "zero_shot_mlp"}
        or requested_split_mode in {"zero_shot", "unseen_concepts", "held_out_concepts"}
        or bool(options.get("zero_shot", False))
    )
    task_stratified_option = bool(options.get("task_stratified_pseudo_epochs", True))
    stratify_groups = None
    if task_stratified_option and isinstance(trial_df, pd.DataFrame) and not trial_df.empty:
        if "semantic_decision" in trial_df.columns:
            dec_series = trial_df["semantic_decision"].astype(str).str.strip()
            if (dec_series != "").any():
                stratify_groups = dec_series.to_numpy()
        if stratify_groups is None:
            try:
                from core.concept_decoding_signal import semantic_decision_label_from_trial_row
                dec_list = [semantic_decision_label_from_trial_row(row) for _, row in trial_df.iterrows()]
                dec_arr = np.asarray([str(d).strip() for d in dec_list], dtype=object)
                if np.any(dec_arr != ""):
                    stratify_groups = dec_arr
            except Exception:
                stratify_groups = None

    explicit_split = options.get("explicit_split")
    if explicit_split is not None:
        from core.concept_benchmark_splits import validate_indices
        try:
            train_idx, validation_idx, test_idx = validate_indices(explicit_split, y, zero_shot=is_zero_shot)
        except (ValueError, KeyError) as exc:
            return None, f"Invalid explicit split: {exc}"
        train_classes = sorted(set(y[train_idx]))
        test_classes = sorted(set(y[test_idx]))
        class_names = train_classes
        split_df = pd.DataFrame({"concept": sorted(set(y))})
    elif is_zero_shot:
        unseen_concepts = options.get("zero_shot_unseen_concepts", options.get("unseen_concepts"))
        try:
            unseen_fraction = float(
                options.get(
                    "zero_shot_unseen_fraction",
                    options.get("unseen_fraction", test_fraction),
                )
            )
        except Exception:
            unseen_fraction = 0.2
        try:
            split_seed = int(options.get("zero_shot_random_state", 42))
        except Exception:
            split_seed = 42
        try:
            train_idx, test_idx, split_df = zero_shot_concept_train_test_split(
                y,
                unseen_concepts=unseen_concepts,
                unseen_fraction=unseen_fraction,
                random_state=split_seed,
                min_train_concepts=2,
                min_test_concepts=1,
                concept_categories=(
                    trial_df["stimulus_group"].to_numpy()
                    if "stimulus_group" in trial_df.columns else None
                ),
                balance_categories=bool(options.get("zero_shot_balance_categories", False)),
            )
        except ValueError as exc:
            return None, str(exc)
        if train_idx.size == 0 or test_idx.size == 0:
            return None, "Zero-shot split could not be created."
        train_classes = sorted(list(set(y[train_idx].tolist())))
        test_classes = sorted(list(set(y[test_idx].tolist())))
        class_names = train_classes
    else:
        if requested_split_mode != "random":
            return None, "Semantic embedding decoders currently support 'random' and 'zero_shot' splits."

        train_idx, test_idx, split_df = per_concept_train_test_split(
            y,
            test_fraction=test_fraction,
            random_state=42,
            stratify_groups=stratify_groups,
        )
        if train_idx.size == 0 or test_idx.size == 0:
            return None, "Per-concept 80/20 split could not be created."
        train_classes = set(y[train_idx].tolist())
        test_classes = set(y[test_idx].tolist())
        class_names = [str(v) for v in pd.Index(y).drop_duplicates().tolist() if str(v) in train_classes and str(v) in test_classes]
        if len(class_names) < 2:
            return None, "Need at least two concepts represented in both train and test splits."
        valid_train = np.asarray([str(y[i]) in class_names for i in train_idx], dtype=bool)
        valid_test = np.asarray([str(y[i]) in class_names for i in test_idx], dtype=bool)
        train_idx = train_idx[valid_train]
        test_idx = test_idx[valid_test]

    outer_train_idx = train_idx.copy()
    if explicit_split is None:
        validation_idx = np.asarray([], dtype=int)
    require_validation = bool(options.get("require_subset_validation", False))
    if explicit_split is None and (model_training_key in {"embedding_mlp", "embedding_contrastive"} or require_validation) and validation_fraction > 0.0:
        fit_rel, validation_rel, _validation_split_df = per_concept_train_test_split(
            y[outer_train_idx],
            test_fraction=validation_fraction,
            random_state=43,
            stratify_groups=np.asarray(stratify_groups)[outer_train_idx] if stratify_groups is not None else None,
        )
        if fit_rel.size > 0 and validation_rel.size > 0:
            candidate_train_idx = outer_train_idx[fit_rel]
            candidate_validation_idx = outer_train_idx[validation_rel]
            fit_classes = set(y[candidate_train_idx].tolist())
            validation_classes = set(y[candidate_validation_idx].tolist())
            if all(label in fit_classes and label in validation_classes for label in class_names):
                train_idx = candidate_train_idx
                validation_idx = candidate_validation_idx

    if require_validation and validation_idx.size == 0:
        return None, "Backward elimination requires a nonempty validation split containing every class. Increase trials or validation fraction."

    train_pseudo = _make_pseudo_epochs(
        X_tensor,
        embeddings,
        y,
        trial_df,
        train_idx,
        trials_per_average=trials_per_average,
        seed=420,
        subset="train",
        balance_classes=balance_training_classes,
        task_stratified=task_stratified_option,
    )
    test_pseudo = _make_pseudo_epochs(
        X_tensor,
        embeddings,
        y,
        trial_df,
        test_idx,
        trials_per_average=trials_per_average,
        seed=421,
        subset="test",
        task_stratified=task_stratified_option,
    )
    validation_pseudo = _make_pseudo_epochs(
        X_tensor,
        embeddings,
        y,
        trial_df,
        validation_idx,
        trials_per_average=trials_per_average,
        seed=422,
        subset="validation",
        task_stratified=task_stratified_option,
    )
    if train_pseudo["X"].shape[0] == 0 or test_pseudo["X"].shape[0] == 0:
        return None, "Averaged pseudo-epoch split could not be created."
    if train_pseudo.get("task_stratified"):
        if trials_per_average == 5:
            _emit_progress(
                progress_callback,
                "Pseudo-epochs: Task-stratified 5-way balancing active (1 trial each from all 5 semantic decisions per pseudo-epoch)."
            )
        elif trials_per_average == 6:
            _emit_progress(
                progress_callback,
                "Pseudo-epochs: Proportional 6-way balancing active (2 Animal + 1 each from Food, Vehicle, Instrument, Body Part; 100% data retention)."
            )
    train_pseudo_classes = set(np.asarray(train_pseudo["y"], dtype=str).tolist())
    test_pseudo_classes = set(np.asarray(test_pseudo["y"], dtype=str).tolist())
    if is_zero_shot:
        train_class_names = [label for label in train_classes if label in train_pseudo_classes]
        test_class_names = [label for label in test_classes if label in test_pseudo_classes]
        if len(train_class_names) < 2:
            return None, "Averaging left fewer than two concepts in train pseudo-epochs."
        if len(test_class_names) < 1:
            return None, "Averaging left no concepts in held-out unseen pseudo-epochs."
        train_keep = np.asarray([label in train_class_names for label in np.asarray(train_pseudo["y"], dtype=str)], dtype=bool)
        test_keep = np.asarray([label in test_class_names for label in np.asarray(test_pseudo["y"], dtype=str)], dtype=bool)
        for key in ("X", "embeddings", "y"):
            train_pseudo[key] = train_pseudo[key][train_keep]
            test_pseudo[key] = test_pseudo[key][test_keep]
        train_pseudo["df"] = train_pseudo["df"].loc[train_keep].reset_index(drop=True)
        test_pseudo["df"] = test_pseudo["df"].loc[test_keep].reset_index(drop=True)
        validation_keep = np.asarray(
            [label in train_class_names for label in np.asarray(validation_pseudo["y"], dtype=str)],
            dtype=bool,
        )
        for key in ("X", "embeddings", "y"):
            validation_pseudo[key] = validation_pseudo[key][validation_keep]
        if not validation_pseudo["df"].empty:
            validation_pseudo["df"] = validation_pseudo["df"].loc[validation_keep].reset_index(drop=True)
        class_names = train_class_names
        candidate_classes = test_class_names
    else:
        class_names = [label for label in class_names if label in train_pseudo_classes and label in test_pseudo_classes]
        if len(class_names) < 2:
            return None, "Averaging left fewer than two concepts in both train and held-out pseudo-epochs."
        train_keep = np.asarray([label in class_names for label in np.asarray(train_pseudo["y"], dtype=str)], dtype=bool)
        test_keep = np.asarray([label in class_names for label in np.asarray(test_pseudo["y"], dtype=str)], dtype=bool)
        for key in ("X", "embeddings", "y"):
            train_pseudo[key] = train_pseudo[key][train_keep]
            test_pseudo[key] = test_pseudo[key][test_keep]
        train_pseudo["df"] = train_pseudo["df"].loc[train_keep].reset_index(drop=True)
        test_pseudo["df"] = test_pseudo["df"].loc[test_keep].reset_index(drop=True)
        validation_keep = np.asarray(
            [label in class_names for label in np.asarray(validation_pseudo["y"], dtype=str)],
            dtype=bool,
        )
        for key in ("X", "embeddings", "y"):
            validation_pseudo[key] = validation_pseudo[key][validation_keep]
        if not validation_pseudo["df"].empty:
            validation_pseudo["df"] = validation_pseudo["df"].loc[validation_keep].reset_index(drop=True)
        candidate_classes = class_names
        train_class_names = class_names
        test_class_names = class_names

    dynamic_epoch_data_factory = None
    if dynamic_pseudo_epochs and model_training_key in {"embedding_mlp", "embedding_contrastive"}:
        def _dynamic_epoch_data(epoch_number: int) -> dict[str, Any]:
            epoch_pseudo = _make_pseudo_epochs(
                X_tensor,
                embeddings,
                y,
                trial_df,
                train_idx,
                trials_per_average=trials_per_average,
                seed=420 + int(epoch_number),
                subset="train",
                balance_classes=balance_training_classes,
                task_stratified=task_stratified_option,
            )
            epoch_keep = np.asarray(
                [
                    label in class_names
                    for label in np.asarray(epoch_pseudo["y"], dtype=str)
                ],
                dtype=bool,
            )
            for key in ("X", "embeddings", "y"):
                epoch_pseudo[key] = epoch_pseudo[key][epoch_keep]
            return epoch_pseudo

        dynamic_epoch_data_factory = _dynamic_epoch_data


    try:
        _emit_progress(
            progress_callback,
            (
                f"created averaged epochs: train={train_pseudo['X'].shape[0]}, "
                f"validation={validation_pseudo['X'].shape[0]}, test={test_pseudo['X'].shape[0]}"
                + (
                    f", balanced training classes at {train_pseudo.get('source_trials_per_class_limit')} source trials/class"
                    if balance_training_classes
                    else ""
                )
            ),
        )
        if model_training_key == "embedding_ridge":
            model, train_meta = _train_ridge_embedding_model(
                train_pseudo["X"],
                train_pseudo["embeddings"],
                train_pseudo["y"],
                class_names=class_names,
                alpha=ridge_alpha,
                input_pca_components=input_pca_components,
                target_svd_components=target_svd_components,
                random_state=training_seed,
                progress_callback=progress_callback,
            )
        elif model_training_key == "embedding_contrastive":
            model, train_meta = _train_torch_embedding_model(
                train_pseudo["X"],
                train_pseudo["embeddings"],
                train_pseudo["y"],
                validation_X=(
                    validation_pseudo["X"]
                    if validation_pseudo["X"].shape[0] > 0
                    else None
                ),
                validation_embeddings=(
                    validation_pseudo["embeddings"]
                    if validation_pseudo["X"].shape[0] > 0
                    else None
                ),
                validation_labels=(
                    validation_pseudo["y"]
                    if validation_pseudo["X"].shape[0] > 0
                    else None
                ),
                class_names=class_names,
                max_epochs=max_epochs,
                batch_size=max(
                    1,
                    int(
                        options.get(
                            "embedding_contrastive_batch_size", batch_size
                        )
                    ),
                ),
                hidden_layer_sizes=(),
                dropout=0.0,
                weight_decay=contrastive_weight_decay,
                random_state=training_seed,
                options=options,
                learning_rate=contrastive_learning_rate,
                contrastive_weight_override=1.0,
                contrastive_temperature_override=contrastive_temperature,
                model_key="embedding_contrastive",
                model_display_name="linear contrastive semantic-embedding",
                restore_best_option="embedding_contrastive_restore_best_validation",
                dynamic_epoch_data_factory=dynamic_epoch_data_factory,
            )
        elif model_training_key == "embedding_mlp":
            hidden_layers = (
                (mlp_hidden_1, mlp_hidden_2)
                if mlp_hidden_2 > 0
                else (mlp_hidden_1,)
            )
            model, train_meta = _train_torch_embedding_model(
                train_pseudo["X"],
                train_pseudo["embeddings"],
                train_pseudo["y"],
                validation_X=(
                    validation_pseudo["X"]
                    if validation_pseudo["X"].shape[0] > 0
                    else None
                ),
                validation_embeddings=(
                    validation_pseudo["embeddings"]
                    if validation_pseudo["X"].shape[0] > 0
                    else None
                ),
                validation_labels=(
                    validation_pseudo["y"]
                    if validation_pseudo["X"].shape[0] > 0
                    else None
                ),
                class_names=class_names,
                max_epochs=max_epochs,
                batch_size=max(1, int(options.get("mlp_batch_size", 32))),
                hidden_layer_sizes=hidden_layers,
                dropout=mlp_dropout,
                weight_decay=mlp_weight_decay,
                random_state=training_seed,
                options=options,
                dynamic_epoch_data_factory=dynamic_epoch_data_factory,
            )
        else:
            model, train_meta = _train_transformer_embedding_model(
                train_pseudo["X"],
                train_pseudo["embeddings"],
                channel_names=list(feature_payload.get("channel_names", [])),
                model_type=model_type,
                max_epochs=max_epochs,
                batch_size=batch_size,
                random_state=training_seed,
                progress_callback=progress_callback,
                seeg_decoder_src=seeg_decoder_src,
            )
    except Exception as exc:
        return None, f"{decoder_label} training failed: {exc}"

    _emit_progress(progress_callback, "predicting held-out embeddings and scoring concepts...")
    if model_training_key == "transformer":
        pred_embeddings = _predict_embeddings(model, test_pseudo["X"], train_meta)
    else:
        pred_embeddings = _predict_flat_embedding_model(model, test_pseudo["X"], train_meta)
    pred_norm = _l2_normalize(pred_embeddings)
    scoring_train_targets = _embedding_targets_for_scoring(
        train_pseudo["embeddings"], train_meta
    )
    scoring_test_targets = _embedding_targets_for_scoring(
        test_pseudo["embeddings"], train_meta
    )
    if is_zero_shot:
        prototypes = _class_prototypes(
            test_pseudo["y"], scoring_test_targets, candidate_classes
        )
    else:
        prototypes = _class_prototypes(
            train_pseudo["y"], scoring_train_targets, candidate_classes
        )
    scores = pred_norm @ prototypes.T
    pred_idx = np.argmax(scores, axis=1)
    y_pred = np.asarray([candidate_classes[int(idx)] for idx in pred_idx], dtype=str)
    y_test = np.asarray(test_pseudo["y"], dtype=str)
    predicted_counts = pd.Series(y_pred).value_counts()
    predicted_unique_count = int(len(predicted_counts))
    most_common_prediction = (
        str(predicted_counts.index[0]) if not predicted_counts.empty else ""
    )
    most_common_prediction_fraction = (
        float(predicted_counts.iloc[0] / max(1, len(y_pred)))
        if not predicted_counts.empty
        else np.nan
    )
    collapse_unique_limit = max(1, int(np.ceil(0.05 * len(candidate_classes))))
    prediction_collapse_detected = bool(
        predicted_unique_count <= collapse_unique_limit
        or (
            np.isfinite(most_common_prediction_fraction)
            and most_common_prediction_fraction >= 0.8
        )
    )
    confidence_temperature = (
        contrastive_temperature if model_training_key == "embedding_contrastive" else 1.0
    )
    confidence_logits = scores / max(1e-8, float(confidence_temperature))
    confidence = np.exp(
        confidence_logits - np.nanmax(confidence_logits, axis=1, keepdims=True)
    )
    confidence = confidence / np.maximum(np.sum(confidence, axis=1, keepdims=True), 1e-12)
    confidence_max = np.nanmax(confidence, axis=1)
    target_similarity = np.sum(
        pred_norm * _l2_normalize(scoring_test_targets), axis=1
    )

    top1_acc = float(accuracy_score(y_test, y_pred))
    top1_bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    mpr, percentile_ranks, true_rank_positions = _mean_percentile_rank(scores, y_test, candidate_classes)
    balanced_mpr = _balanced_percentile_rank(percentile_ranks, y_test, candidate_classes)
    reciprocal_ranks = np.where(
        np.isfinite(true_rank_positions) & (true_rank_positions > 0),
        1.0 / true_rank_positions,
        np.nan,
    )
    mean_reciprocal_rank = (
        float(np.nanmean(reciprocal_ranks))
        if np.any(np.isfinite(reciprocal_ranks))
        else np.nan
    )
    top3_accuracy = float(np.nanmean(true_rank_positions <= min(3, len(candidate_classes))))
    top5_accuracy = float(np.nanmean(true_rank_positions <= min(5, len(candidate_classes))))
    acc = float(mpr)
    bal_acc = float(balanced_mpr)
    chance_accuracy = float(0.5 if len(candidate_classes) > 1 else 1.0)
    top1_chance_accuracy = float(1.0 / max(1, len(candidate_classes)))
    cm = confusion_matrix(y_test, y_pred, labels=candidate_classes)
    report = classification_report(y_test, y_pred, labels=candidate_classes, output_dict=True, zero_division=0)
    metrics_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    permutation_stats = _label_permutation_baseline(
        scores,
        y_test,
        observed_accuracy=acc,
        observed_balanced_accuracy=bal_acc,
        class_names=candidate_classes,
        n_permutations=n_permutations,
    )

    pairwise_2afc_trials = []
    for row_idx in range(len(y_test)):
        true_c = str(y_test[row_idx])
        if true_c not in candidate_classes:
            continue
        c_col = candidate_classes.index(true_c)
        true_score = scores[row_idx, c_col]
        distractor_cols = [col for col in range(len(candidate_classes)) if col != c_col]
        if not distractor_cols:
            continue
        distractor_scores = scores[row_idx, distractor_cols]
        wins = np.sum(true_score > distractor_scores) + 0.5 * np.sum(true_score == distractor_scores)
        pairwise_2afc_trials.append(float(wins / len(distractor_cols)))
    pairwise_2afc_acc = (
        float(np.mean(pairwise_2afc_trials))
        if pairwise_2afc_trials
        else (1.0 if top1_acc == 1.0 else 0.5)
    )

    predictions_df = test_pseudo["df"].copy().reset_index(drop=True)
    predictions_df.insert(0, "true_concept", y_test)
    predictions_df.insert(1, "predicted_concept", y_pred)
    predictions_df.insert(2, "correct", y_test == y_pred)
    predictions_df.insert(3, "confidence", confidence_max)
    predictions_df.insert(4, "predicted_target_cosine", target_similarity)
    predictions_df.insert(5, "mpr_percentile_rank", percentile_ranks)
    predictions_df.insert(6, "true_concept_rank", true_rank_positions)
    predictions_df.insert(7, "reciprocal_rank", reciprocal_ranks)
    if is_zero_shot and len(pairwise_2afc_trials) == len(predictions_df):
        predictions_df.insert(8, "pairwise_2afc_accuracy", pairwise_2afc_trials)
    keep_cols = [
        "true_concept",
        "predicted_concept",
        "correct",
        "confidence",
        "predicted_target_cosine",
        "mpr_percentile_rank",
        "true_concept_rank",
        "reciprocal_rank",
        "pairwise_2afc_accuracy",
        "n_source_trials",
        "source_trial_indices",
        "trial_index",
        "stimulus",
        "canonical_stimulus",
        "stimulus_group",
        "embedding_key",
        "source_fif",
        "source_fif_path",
        "session_id",
        "run_id",
        "block_id",
        "onset_meg_s",
        "qc_outlier",
        "qc_outlier_score",
    ]
    predictions_df = predictions_df[[col for col in keep_cols if col in predictions_df.columns]]
    instruction_summary = {"instruction_evaluation_status": "no_compatible_instruction_targets"}
    if instruction_targets:
        try:
            instruction_summary = score_instructions(
                predictions_df, pred_norm, instruction_targets, trial_df,
                lambda vectors: _embedding_targets_for_scoring(vectors, train_meta),
            )
            instruction_summary["instruction_evaluation_status"] = "evaluated"
        except Exception as exc:
            return None, f"Instruction evaluation failed: {exc}"

    per_class_dict = {
        "concept": y_test,
        "correct": y_test == y_pred,
        "mpr_percentile_rank": percentile_ranks,
        "true_concept_rank": true_rank_positions,
        "reciprocal_rank": reciprocal_ranks,
    }
    if is_zero_shot and len(pairwise_2afc_trials) == len(y_test):
        per_class_dict["pairwise_2afc_accuracy"] = pairwise_2afc_trials
    agg_spec = {
        "test_accuracy": ("mpr_percentile_rank", "mean"),
        "test_mpr": ("mpr_percentile_rank", "mean"),
        "top1_accuracy": ("correct", "mean"),
        "mean_true_rank": ("true_concept_rank", "mean"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
    }
    if is_zero_shot and "pairwise_2afc_accuracy" in per_class_dict:
        agg_spec["pairwise_2afc_accuracy"] = ("pairwise_2afc_accuracy", "mean")
    agg_spec["n_test"] = ("correct", "size")

    per_class_accuracy_df = (
        pd.DataFrame(per_class_dict)
        .groupby("concept", sort=False)
        .agg(**agg_spec)
        .reset_index()
    )
    if isinstance(split_df, pd.DataFrame) and "concept" in split_df.columns:
        split_df = split_df.merge(per_class_accuracy_df, on="concept", how="left")

    feature_meta_df = feature_payload.get("feature_meta_df", pd.DataFrame())
    if not isinstance(feature_meta_df, pd.DataFrame):
        feature_meta_df = pd.DataFrame()
    else:
        feature_meta_df = feature_meta_df.copy()
    flat_train = np.asarray(train_pseudo["X"], dtype=np.float32).reshape(train_pseudo["X"].shape[0], -1)
    if len(feature_meta_df) != flat_train.shape[1]:
        feature_names = [str(value) for value in feature_payload.get("feature_names", [])]
        if len(feature_names) != flat_train.shape[1]:
            feature_names = [f"feature_{idx}" for idx in range(flat_train.shape[1])]
        feature_meta_df = pd.DataFrame(
            {
                "feature_index": np.arange(flat_train.shape[1], dtype=int),
                "feature": feature_names,
            }
        )
    feature_importance = np.asarray(train_meta.get("feature_importance", []), dtype=float).reshape(-1)
    if feature_importance.size == len(feature_meta_df):
        feature_meta_df["importance"] = feature_importance
    else:
        feature_meta_df["importance"] = np.nan
    feature_meta_df["mean_abs_train_signal"] = np.nanmean(np.abs(flat_train), axis=0)
    feature_meta_df["std_train_signal"] = np.nanstd(flat_train, axis=0)
    feature_df = (
        feature_meta_df.sort_values("importance", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    if not options.get("save_full_feature_table", False):
        feature_df = feature_df.head(250)

    n_bands = (
        int(train_pseudo["X"].shape[2])
        if feature_space == "signal" and train_pseudo["X"].ndim >= 4
        else int(feature_payload.get("n_bands", 0) or 0)
    )
    n_time_bins = (
        int(train_pseudo["X"].shape[3])
        if feature_space == "signal" and train_pseudo["X"].ndim >= 4
        else int(
            feature_payload.get(
                "n_time_points", len(feature_payload.get("bin_edges", []))
            )
            or 0
        )
    )
    input_feature_note = (
        "Source-parcel feature pseudo-epochs"
        if feature_space == "source"
        else "Band-power pseudo-epochs"
    )
    primary_metric_name = "2AFC" if is_zero_shot and len(candidate_classes) >= 2 else "MPR"
    metric_value = pairwise_2afc_acc if primary_metric_name == "2AFC" else acc
    unseen_category_counts = {}
    if is_zero_shot and "stimulus_category" in split_df.columns:
        unseen_categories = split_df.loc[split_df["is_unseen"], "stimulus_category"]
        unseen_categories = unseen_categories.astype(str).str.strip()
        unseen_category_counts = {
            str(category): int(count)
            for category, count in unseen_categories[unseen_categories.astype(bool)].value_counts().sort_index().items()
        }
    summary = {
        "decoder": decoder_label,
        "decoder_type": decoder_key,
        "is_zero_shot": bool(is_zero_shot),
        "zero_shot": bool(is_zero_shot),
        "zero_shot_category_balanced": bool(
            is_zero_shot and options.get("zero_shot_balance_categories", False)
        ),
        "unseen_category_counts": unseen_category_counts,
        "base_decoder": model_training_key,
        "primary_metric": primary_metric_name,
        "metric": primary_metric_name,
        "split": (
            f"explicit {requested_split_mode}: {len(train_idx)} train, {len(validation_idx)} validation, {len(test_idx)} test raw trials"
            if explicit_split is not None
            else f"zero-shot: {len(train_class_names)} train concepts, {len(test_class_names)} unseen test concepts"
            if is_zero_shot
            else f"{int(round((1.0 - test_fraction) * 100))}/{int(round(test_fraction * 100))} per concept, averaged after split"
        ),
        "split_mode": requested_split_mode if explicit_split is not None else ("zero_shot" if is_zero_shot else "random"),
        "training_seed": training_seed,
        "explicit_split": explicit_split is not None,
        "embedding_validation_metric": options.get("embedding_validation_metric", "top1"),
        "balance_training_classes": bool(balance_training_classes),
        "training_source_trials_per_class_limit": (
            int(train_pseudo["source_trials_per_class_limit"])
            if train_pseudo.get("source_trials_per_class_limit") is not None
            else np.nan
        ),
        "training_pseudo_epochs_per_class": {
            str(label): int(count)
            for label, count in pd.Series(train_pseudo["y"], dtype=str).value_counts().sort_index().items()
        },
        "pairwise_2afc_accuracy": pairwise_2afc_acc if is_zero_shot else np.nan,
        "pairwise_2afc_chance_accuracy": 0.5 if is_zero_shot else np.nan,
        "above_chance_pairwise_2afc": bool(pairwise_2afc_acc > 0.5) if (is_zero_shot and np.isfinite(pairwise_2afc_acc)) else False,
        "accuracy": metric_value,
        "balanced_accuracy": bal_acc,
        "mpr": acc,
        "balanced_mpr": bal_acc,
        "top1_accuracy": top1_acc,
        "top1_balanced_accuracy": top1_bal_acc,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "mean_reciprocal_rank": mean_reciprocal_rank,
        "chance_accuracy": chance_accuracy,
        "top1_chance_accuracy": top1_chance_accuracy,
        "above_chance_accuracy": bool(acc > chance_accuracy),
        "above_chance_balanced_accuracy": bool(bal_acc > chance_accuracy),
        "prediction_collapse_detected": prediction_collapse_detected,
        "predicted_unique_concepts": predicted_unique_count,
        "most_common_prediction": most_common_prediction,
        "most_common_prediction_fraction": most_common_prediction_fraction,
        "prediction_confidence_temperature": float(confidence_temperature),
        "n_trials": int(train_pseudo["X"].shape[0] + test_pseudo["X"].shape[0]),
        "n_source_trials": int(len(y)),
        "n_train": int(train_pseudo["X"].shape[0]),
        "n_validation": int(validation_pseudo["X"].shape[0]),
        "n_test": int(test_pseudo["X"].shape[0]),
        "n_source_train_trials": int(train_idx.size),
        "n_source_validation_trials": int(validation_idx.size),
        "n_source_test_trials": int(test_idx.size),
        "n_concepts": int(len(candidate_classes)),
        "n_train_concepts": int(len(train_class_names)) if is_zero_shot else int(len(class_names)),
        "n_unseen_concepts": int(len(test_class_names)) if is_zero_shot else 0,
        "train_concepts": train_class_names if is_zero_shot else class_names,
        "unseen_concepts": test_class_names if is_zero_shot else [],
        "candidate_concepts": candidate_classes,
        "n_runs": int(feature_payload.get("n_runs", 0)),
        "n_channels": int(len(feature_payload.get("channel_names", []))),
        "n_rois": int(len(feature_payload.get("roi_names", []))),
        "n_features": int(np.prod(train_pseudo["X"].shape[1:])),
        "n_bands": n_bands,
        "n_time_bins": n_time_bins,
        "trials_per_average": int(trials_per_average),
        "embedding_dim": int(train_meta.get("embedding_dim", embeddings.shape[1])),
        "embedding_original_dim": int(embeddings.shape[1]),
        "embedding_input_pca_requested_components": int(
            train_meta.get("input_pca_requested_components", 0)
        ),
        "embedding_input_pca_components": int(
            train_meta.get("input_pca_components", np.prod(train_pseudo["X"].shape[1:]))
        ),
        "embedding_input_pca_applied": bool(train_meta.get("input_pca") is not None),
        "embedding_input_pca_explained_variance": float(
            train_meta.get("input_pca_explained_variance", 1.0)
        ),
        "embedding_target_svd_requested_components": int(
            train_meta.get("embedding_target_svd_requested_components", 0)
        ),
        "embedding_target_svd_components": int(
            train_meta.get("embedding_target_svd_components", embeddings.shape[1])
        ),
        "embedding_target_rank": int(
            train_meta.get("embedding_target_rank", embeddings.shape[1])
        ),
        "embedding_target_svd_explained_variance": float(
            train_meta.get("embedding_target_svd_explained_variance", 1.0)
        ),
        "embedding_model_parameter_count": int(
            train_meta.get("model_parameter_count", 0) or 0
        ),
        "embedding_dict_path": str(feature_payload.get("embedding_dict_path", "")),
        "trials_per_average": int(trials_per_average),
        "avg_trials": int(trials_per_average),
        "pseudo_epoch_trials": int(trials_per_average),
        "model_type": str(train_meta.get("model_type", model_type)),
        "device": str(train_meta.get("device", "")),
        "epochs": int(train_meta.get("torch_epochs", max_epochs)),
        "epochs_requested": int(train_meta.get("torch_epochs_requested", max_epochs)),
        "batch_size": int(train_meta.get("torch_batch_size", batch_size)),
        "final_train_loss": float(train_meta.get("final_train_loss", np.nan)),
        "last_epoch_train_loss": float(
            train_meta.get("last_epoch_train_loss", train_meta.get("final_train_loss", np.nan))
        ),
        # Backward-compatible transformer_ aliases:
        "transformer_trials_per_average": int(trials_per_average),
        "transformer_model_type": str(train_meta.get("model_type", model_type)),
        "transformer_device": str(train_meta.get("device", "")),
        "transformer_pool_time_bins": int(train_meta.get("time_bins", 0) or 0),
        "transformer_epochs": int(train_meta.get("torch_epochs", max_epochs)),
        "transformer_epochs_requested": int(
            train_meta.get("torch_epochs_requested", max_epochs)
        ),
        "transformer_batch_size": int(train_meta.get("torch_batch_size", batch_size)),
        "transformer_final_train_loss": float(train_meta.get("final_train_loss", np.nan)),
        "transformer_last_epoch_train_loss": float(
            train_meta.get("last_epoch_train_loss", train_meta.get("final_train_loss", np.nan))
        ),
        "embedding_selected_epoch": int(train_meta.get("selected_epoch", 0) or 0),
        "embedding_early_stopping_patience": int(
            train_meta.get("early_stopping_patience", 0) or 0
        ),
        "embedding_early_stopped": bool(train_meta.get("early_stopped", False)),
        "embedding_dynamic_pseudo_epochs": bool(
            train_meta.get("dynamic_pseudo_epochs", False)
        ),
        "seeg_decoder_src": str(train_meta.get("seeg_decoder_src", seeg_decoder_src or "")),
        "embedding_ridge_alpha": float(train_meta.get("ridge_alpha", ridge_alpha)),
        "embedding_contrastive_temperature": float(
            train_meta.get("contrastive_temperature", contrastive_temperature)
        ),
        "embedding_contrastive_loss_weight": float(
            train_meta.get("contrastive_weight", np.nan)
        ),
        "embedding_contrastive_weight_decay": float(
            train_meta.get("weight_decay", contrastive_weight_decay)
        ),
        "embedding_contrastive_learning_rate": float(
            train_meta.get("learning_rate", contrastive_learning_rate)
        ),
        "embedding_contrastive_best_validation_epoch": int(
            train_meta.get("best_validation_epoch", 0) or 0
        ),
        "embedding_contrastive_best_validation_accuracy": train_meta.get(
            "best_validation_accuracy", np.nan
        ),
        "embedding_mlp_hidden_layers": "->".join(
            str(value) for value in train_meta.get("hidden_layer_sizes", ())
        ),
        "embedding_mlp_dropout": float(train_meta.get("dropout", mlp_dropout)),
        "embedding_mlp_weight_decay": float(train_meta.get("weight_decay", mlp_weight_decay)),
        "embedding_mlp_best_validation_epoch": int(
            train_meta.get("best_validation_epoch", 0) or 0
        ),
        "embedding_mlp_best_validation_accuracy": train_meta.get(
            "best_validation_accuracy", np.nan
        ),
        "embedding_target_centered": bool(
            train_meta.get("embedding_target_centered", False)
        ),
        "embedding_mlp_contrastive_weight": train_meta.get(
            "contrastive_weight", np.nan
        ),
        "embedding_mlp_temperature": train_meta.get(
            "contrastive_temperature", np.nan
        ),
        "window_start_s": float(feature_payload.get("window_start")),
        "window_end_s": float(feature_payload.get("window_end")),
        "bandpass_low_hz": feature_payload.get("bandpass_low_hz", np.nan),
        "bandpass_high_hz": feature_payload.get("bandpass_high_hz", np.nan),
        "feature_binning_enabled": bool(feature_payload.get("feature_binning_enabled", True)),
        "resample_enabled": bool(feature_payload.get("resample_enabled", False)),
        "resample_sfreq": feature_payload.get("resample_sfreq", np.nan),
        "feature_space": feature_space,
        "channel_mode": str(feature_payload.get("channel_mode")),
        "group_mode": str(feature_payload.get("group_mode")),
        "baseline_correct": bool(feature_payload.get("baseline_correct")),
        "feature_mode": str(feature_payload.get("feature_mode", "")),
        "band_defs": feature_payload.get("band_defs", []),
        "source_power_baseline": bool(feature_payload.get("source_power_baseline", False)),
        "source_power_padding_s": feature_payload.get("source_power_padding_s", 0.0),
        "source_power_aggregation": feature_payload.get("source_power_aggregation", ""),
        "source_power_orientation": feature_payload.get("source_power_orientation", ""),
        "source_parcel_baseline_zscore": bool(
            feature_payload.get("source_parcel_baseline_zscore", False)
        ),
        "parcel_baseline_start_s": feature_payload.get(
            "parcel_baseline_start", np.nan
        ),
        "parcel_baseline_end_s": feature_payload.get("parcel_baseline_end", np.nan),
        "parcel_baseline_samples": int(
            feature_payload.get("parcel_baseline_samples", 0) or 0
        ),
        "invalid_parcel_baselines": int(
            feature_payload.get("invalid_parcel_baselines", 0) or 0
        ),
        "source_method": str(feature_payload.get("method", "")),
        "source_lambda2": feature_payload.get("lambda2", np.nan),
        "source_batch_size": int(feature_payload.get("source_batch_size", 0) or 0),
        "source_space_kind": str(feature_payload.get("source_space_kind", "")),
        "source_parcellation": str(feature_payload.get("source_parcellation", "")),
        "source_common_space": str(feature_payload.get("source_common_space", "")),
        "source_subject": str(feature_payload.get("subject", "")),
        "source_subjects_dir": str(feature_payload.get("subjects_dir", "")),
        "source_subject_count": int(
            feature_payload.get("source_subject_count", 0) or 0
        ),
        "source_subjects": ", ".join(
            str(value)
            for value in feature_payload.get("native_subjects", [])
            if str(value).strip()
        ),
        "mne_epoch_reject_enabled": bool(feature_payload.get("mne_epoch_reject_enabled", False)),
        "mne_reject_mag_ft": feature_payload.get("mne_reject_mag_ft", np.nan),
        "mne_reject_grad_ft_cm": feature_payload.get("mne_reject_grad_ft_cm", np.nan),
        "epoch_tmin_s": feature_payload.get("epoch_tmin", np.nan),
        "epoch_tmax_s": feature_payload.get("epoch_tmax", np.nan),
        "exclude_trial_outliers": bool(feature_payload.get("exclude_trial_outliers", False)),
        "trial_outlier_z": feature_payload.get("trial_outlier_z", np.nan),
        "decoder_note": (
            f"{input_feature_note} are averaged within train/test splits separately; "
            "no held-out trials are used in training averages. Primary held-out metric is mean percentile rank "
            "(MPR) against candidate concept embeddings; confusion matrix and top1_accuracy use cosine candidate "
            "ranking. Embedding targets are centered before training to remove the trivial semantic-centroid "
            "solution. Any neural PCA and semantic SVD transforms are fitted on training data only. "
            f"Model={decoder_key}. {str(train_meta.get('device_note', '')).strip()}"
        ).strip(),
        "dropped_trials": int(feature_payload.get("dropped_trials", 0)),
        "qc_outlier_trials": (
            int(pd.Series(trial_qc_df.get("qc_outlier", pd.Series(dtype=bool))).fillna(False).astype(bool).sum())
            if not trial_qc_df.empty
            else 0
        ),
        "qc_dropped_trials": (
            int((trial_qc_df.get("qc_status", pd.Series(dtype=str)).astype(str) == "dropped").sum())
            if not trial_qc_df.empty
            else int(feature_payload.get("dropped_trials", 0))
        ),
        "epoch_cache_hits": int(feature_payload.get("epoch_cache_hits", 0) or 0),
        "epoch_cache_misses": int(feature_payload.get("epoch_cache_misses", 0) or 0),
        "n_permutations_requested": int(n_permutations),
        "n_permutations_completed": int(permutation_stats.get("permutation_n_completed", 0)),
        "permutation_mean_accuracy": permutation_stats.get("permutation_mean_accuracy", np.nan),
        "permutation_std_accuracy": permutation_stats.get("permutation_std_accuracy", np.nan),
        "permutation_max_accuracy": permutation_stats.get("permutation_max_accuracy", np.nan),
        "permutation_p_value_accuracy": permutation_stats.get("permutation_p_value_accuracy", np.nan),
        "permutation_mean_balanced_accuracy": permutation_stats.get("permutation_mean_balanced_accuracy", np.nan),
        "permutation_std_balanced_accuracy": permutation_stats.get("permutation_std_balanced_accuracy", np.nan),
        "permutation_max_balanced_accuracy": permutation_stats.get("permutation_max_balanced_accuracy", np.nan),
        "permutation_p_value_balanced_accuracy": permutation_stats.get("permutation_p_value_balanced_accuracy", np.nan),
    }
    summary.update(instruction_summary)
    qc_summary_df = concept_qc_summary_table(trial_qc_df, feature_payload=feature_payload)
    architecture_df = concept_model_architecture_table(summary, feature_payload, class_names)
    summary_df = pd.DataFrame([summary])
    result = {
        **summary,
        "class_names": candidate_classes,
        "train_class_names": train_class_names if is_zero_shot else class_names,
        "test_class_names": test_class_names if is_zero_shot else class_names,
        "unseen_class_names": test_class_names if is_zero_shot else [],
        "confusion_matrix": cm,
        "summary_df": summary_df,
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
        "feature_df": feature_df,
        "split_df": split_df,
        "trial_qc_df": trial_qc_df,
        "qc_summary_df": qc_summary_df,
        "per_class_accuracy_df": per_class_accuracy_df,
        "permutation_df": permutation_stats.get("permutation_df", pd.DataFrame()),
        "model_architecture_df": architecture_df,
        "template_df": pd.DataFrame(),
        "template_matrix": np.empty((0, 0), dtype=np.float32),
        "template_classes": [],
        "template_feature_meta_df": pd.DataFrame(),
        "training_history_df": pd.DataFrame(train_meta.get("training_history", [])),
        "train_loss_df": pd.DataFrame(
            {
                "epoch": np.arange(1, len(train_meta.get("train_losses", [])) + 1, dtype=int),
                "train_loss": train_meta.get("train_losses", []),
            }
        ),
        "transformer_train_loss_df": pd.DataFrame(
            {
                "epoch": np.arange(1, len(train_meta.get("train_losses", [])) + 1, dtype=int),
                "train_loss": train_meta.get("train_losses", []),
            }
        ),
    }

    if require_validation:
        validation_predictions = _predict_flat_embedding_model(model, validation_pseudo["X"], train_meta)
        validation_prototypes = _class_prototypes(train_pseudo["y"], scoring_train_targets, class_names)
        validation_scores = _l2_normalize(validation_predictions) @ validation_prototypes.T
        validation_top1, _ = _embedding_retrieval_metrics(
            validation_predictions, validation_pseudo["y"], validation_prototypes, class_names)
        validation_mpr, _, _ = _mean_percentile_rank(validation_scores, validation_pseudo["y"], class_names)
        result.update(validation_top1_accuracy=validation_top1, validation_mpr=validation_mpr)
        result["summary_df"]["validation_top1_accuracy"] = validation_top1
        result["summary_df"]["validation_mpr"] = validation_mpr

    artifact_callback = options.get("inference_artifact_callback")
    if callable(artifact_callback):
        artifact_callback({
            "instruction_targets": instruction_targets,
            "model": model, "train_meta": train_meta, "decoder_type": decoder_key,
            "test_X": test_pseudo["X"], "test_y": y_test,
            "prototypes": prototypes, "class_names": class_names, "scores": scores,
            "feature_meta_df": feature_meta_df,
            "train_indices": train_idx, "validation_indices": validation_idx,
            "test_indices": test_idx,
            **({"validation_X": validation_pseudo["X"],
                "validation_y": validation_pseudo["y"],
                "validation_scores": validation_scores} if require_validation else {}),
        })

    skipped_paths = feature_payload.get("skipped_paths", [])
    skip_msg = f", skipped_runs={len(skipped_paths)}" if skipped_paths else ""
    cache_hits = int(summary.get("epoch_cache_hits", 0) or 0)
    cache_misses = int(summary.get("epoch_cache_misses", 0) or 0)
    cache_msg = f", epoch_cache={cache_hits} hit/{cache_misses} miss" if (cache_hits or cache_misses) else ""
    p_bal = permutation_stats.get("permutation_p_value_balanced_accuracy", np.nan)
    p_msg = f", p_bal={float(p_bal):.3f}" if np.isfinite(float(p_bal)) else ""
    feature_shape_msg = (
        f"rois={summary['n_rois']}, features={summary['n_features']}"
        if feature_space == "source"
        else (
            f"channels={summary['n_channels']}, bands={summary['n_bands']}, "
            f"bins={summary['n_time_bins']}"
        )
    )
    if is_zero_shot:
        metric_msg = f"2AFC={pairwise_2afc_acc:.3f}, top1_accuracy={top1_acc:.3f} (chance={top1_chance_accuracy:.3f}), MPR={acc:.3f}"
        concept_msg = f"train_concepts={len(train_class_names)}, unseen_concepts={len(test_class_names)}"
    else:
        metric_msg = f"MPR={acc:.3f}, balanced_MPR={bal_acc:.3f}, top1_accuracy={top1_acc:.3f} (chance={top1_chance_accuracy:.3f})"
        concept_msg = f"concepts={len(class_names)}"
    msg = (
        f"{decoder_label}: {metric_msg}, {concept_msg}, "
        f"pseudo train/test={summary['n_train']}/{summary['n_test']}, "
        f"source train/test={summary['n_source_train_trials']}/{summary['n_source_test_trials']}, "
        f"avg_N={trials_per_average}, {feature_shape_msg}, MPR_chance={chance_accuracy:.3f}{p_msg}"
        f"{cache_msg}"
        f"{skip_msg}"
    )
    if prediction_collapse_detected:
        msg += (
            f" WARNING: prediction collapse detected ({predicted_unique_count} unique "
            f"prediction(s); {most_common_prediction}="
            f"{most_common_prediction_fraction:.1%})."
        )
    return result, msg


def compute_signal_space_semantic_embedding_concept_decoding(
    context: Any,
    options: dict[str, Any] | None = None,
):
    if options is None:
        options = {}
    if not getattr(context, "b_fif_loaded", False):
        return None, "No FIF loaded."
    payload, err_msg = build_signal_bandpower_transformer_payload(context, options)
    if payload is None:
        return None, err_msg or "Could not build semantic band-power features."
    return compute_transformer_decoding_from_payload(context, options, payload)


def compute_signal_space_transformer_concept_decoding(
    context: Any,
    options: dict[str, Any] | None = None,
):
    """Backward-compatible entry point for the original Transformer decoder."""
    return compute_signal_space_semantic_embedding_concept_decoding(context, options)
