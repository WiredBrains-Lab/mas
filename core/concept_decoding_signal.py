"""Signal-space concept decoding and shared decoder utilities."""

import os
import warnings
import hashlib
import json
import sys
import tempfile
import shutil
from functools import lru_cache
from pathlib import Path
from collections import OrderedDict

import mne
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from core.meg_channels import emit_bad_meg_channel_log, good_meg_channel_names
from core.instruction_embeddings import block_key, find_instruction_csv, read_instruction_csv


GROUPED_SPLIT_MODES = {
    "leave_one_run", "run", "run_out", "file",
    "leave_one_session", "session", "session_out",
}

# One default preprocessing profile is shared by signal decoding, source-parcel
# decoding, and parcel-trial visualization. UI values may override any field.
CONCEPT_PREPROCESSING_DEFAULTS = {
    "bandpass_low_hz": 0.1,
    "bandpass_high_hz": 150.0,
    "mne_resample_enabled": True,
    "resample_sfreq": 400.0,
    "mne_epoch_reject_enabled": True,
    "reject_mag_ft": 4000.0,
    "reject_grad_ft_cm": 4000.0,
    "exclude_trial_outliers": True,
    "trial_outlier_z": 6.0,
}

# Real data on either side moves resampling edge effects outside the returned
# window. This is separate from FFT npad and from downstream Hilbert padding.
# MNE recommends longer epochs when resampling after epoching:
# https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
CONCEPT_RESAMPLE_PADDING_S = 0.5

# Parcel-trial views can select a narrower band inside the shared preprocessing
# passband. Keep the ranges explicit so UI availability checks, cache keys, and
# extraction options describe the same operation.
PARCEL_TRIAL_FREQUENCY_BANDS = (
    ("broadband", "Broadband (shared bandpass)", None, None),
    ("delta", "Delta (1–4 Hz)", 1.0, 4.0),
    ("theta", "Theta (4–8 Hz)", 4.0, 8.0),
    ("alpha", "Alpha (8–13 Hz)", 8.0, 13.0),
    ("beta", "Beta (13–30 Hz)", 13.0, 30.0),
    ("low_gamma", "Low gamma (30–45 Hz)", 30.0, 45.0),
    ("high_gamma", "High gamma (70–150 Hz)", 70.0, 150.0),
)


def parcel_trial_frequency_band_settings(
    selection,
    shared_low,
    shared_high,
    *,
    resample_enabled=False,
    resample_sfreq=None,
):
    """Resolve a Parcel trials band selector to filter limits and a label.

    Broadband inherits the shared cutoffs. A named band is available only when
    its full range survives the shared passband and optional downsampling.
    """
    selection = str(selection or "broadband").strip().lower()
    by_key = {row[0]: row for row in PARCEL_TRIAL_FREQUENCY_BANDS}
    if selection not in by_key:
        raise ValueError(f"Unknown Parcel trials frequency band: {selection}.")

    try:
        low = float(shared_low)
    except Exception:
        low = 0.0
    try:
        high = float(shared_high)
    except Exception:
        high = 0.0
    low = low if np.isfinite(low) and low > 0.0 else 0.0
    high = high if np.isfinite(high) and high > 0.0 else 0.0
    if low > 0.0 and high > 0.0 and high <= low:
        raise ValueError("Shared bandpass high cutoff must be greater than low cutoff.")
    low_label = f"{low:g}" if low > 0.0 else "open"
    high_label = f"{high:g}" if high > 0.0 else "open"
    shared_label = f"{low_label}–{high_label} Hz"

    _key, display_label, band_low, band_high = by_key[selection]
    if selection == "broadband":
        effective_low, effective_high = low, high
        resolved_label = f"Broadband ({shared_label})"
    else:
        band_low = float(band_low)
        band_high = float(band_high)
        if (low > 0.0 and low > band_low) or (high > 0.0 and high < band_high):
            raise ValueError(
                f"{display_label} is unavailable after the shared {shared_label} "
                f"bandpass; its full {band_low:g}–{band_high:g} Hz range must pass."
            )
        effective_low, effective_high = band_low, band_high
        resolved_label = str(display_label)

    if bool(resample_enabled) and effective_high > 0.0:
        try:
            target_sfreq = float(resample_sfreq)
        except Exception:
            target_sfreq = np.nan
        if (
            np.isfinite(target_sfreq)
            and target_sfreq > 0.0
            and effective_high >= target_sfreq / 2.0
        ):
            raise ValueError(
                f"{resolved_label} is unavailable at the {target_sfreq:g} Hz shared "
                f"downsample rate because its {effective_high:g} Hz upper edge must be "
                f"below the {target_sfreq / 2.0:g} Hz Nyquist frequency."
            )
    return effective_low, effective_high, resolved_label


def emit_concept_progress(options, message):
    callback = None
    if isinstance(options, dict):
        callback = options.get("progress_callback")
    if callable(callback):
        try:
            callback(str(message))
        except Exception:
            pass


def emit_concept_training_metric(options, metric):
    """Emit one structured neural-network training update, if requested."""
    callback = None
    if isinstance(options, dict):
        callback = options.get("training_metric_callback")
    if callable(callback):
        try:
            callback(dict(metric or {}))
        except Exception:
            pass


def concept_trial_set_id(trial_df):
    """Return a stable short identifier for retained labeled trials."""
    if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
        return ""
    columns = [
        col
        for col in ("source_fif_path", "onset_sample", "concept_label")
        if col in trial_df.columns
    ]
    if not columns:
        return ""
    normalized = trial_df[columns].copy()
    for col in columns:
        normalized[col] = normalized[col].astype(str)
    normalized = normalized.sort_values(columns, kind="stable").reset_index(drop=True)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class PrototypeCorrelationClassifier(ClassifierMixin, BaseEstimator):
    """Nearest-template classifier using centered waveform-shape correlation."""

    def __init__(self, similarity_scale=8.0, eps=1e-12):
        self.similarity_scale = similarity_scale
        self.eps = eps

    def _normalize_rows(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Template matching expects a 2D feature matrix.")
        centered = X - np.nanmean(X, axis=1, keepdims=True)
        centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms > float(self.eps), norms, 1.0)
        return centered / norms

    def fit(self, X, y, sample_weight=None):
        Xn = self._normalize_rows(X)
        y = np.asarray(y, dtype=str)
        if Xn.shape[0] != y.size:
            raise ValueError("X and y have inconsistent sample counts.")
        classes = []
        for label in y.tolist():
            if str(label) not in classes:
                classes.append(str(label))
        if len(classes) < 2:
            raise ValueError("Template matching requires at least two classes.")

        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape[0] != y.size:
                weights = None

        templates = []
        for label in classes:
            mask = y == str(label)
            class_rows = Xn[mask]
            if class_rows.size == 0:
                continue
            if weights is not None:
                w = weights[mask]
                finite = np.isfinite(w) & (w > 0)
                if np.any(finite):
                    template = np.average(class_rows[finite], axis=0, weights=w[finite])
                else:
                    template = np.nanmean(class_rows, axis=0)
            else:
                template = np.nanmean(class_rows, axis=0)
            template = np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0)
            norm = float(np.linalg.norm(template))
            if norm > float(self.eps):
                template = template / norm
            templates.append(template)

        if len(templates) < 2:
            raise ValueError("Template matching could not learn at least two class templates.")
        self.classes_ = np.asarray(classes[: len(templates)], dtype=str)
        self.templates_ = np.vstack(templates)
        if self.templates_.shape[0] > 1:
            self.feature_importances_ = np.nanstd(self.templates_, axis=0)
        else:
            self.feature_importances_ = np.abs(self.templates_[0])
        return self

    def decision_function(self, X):
        if not hasattr(self, "templates_"):
            raise RuntimeError("Template matcher has not been fit.")
        Xn = self._normalize_rows(X)
        return np.dot(Xn, self.templates_.T)

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        scores = np.asarray(self.decision_function(X), dtype=float)
        scaled = scores * float(self.similarity_scale)
        scaled = scaled - np.nanmax(scaled, axis=1, keepdims=True)
        exp_scores = np.exp(np.nan_to_num(scaled, nan=-100.0, posinf=0.0, neginf=-100.0))
        denom = np.sum(exp_scores, axis=1, keepdims=True)
        denom = np.where(denom > float(self.eps), denom, 1.0)
        return exp_scores / denom


class DropoutMLPClassifier(ClassifierMixin, BaseEstimator):
    """Small sklearn-compatible PyTorch MLP with hidden-layer dropout."""

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        dropout=0.35,
        alpha=1e-3,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=100,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _torch_modules():
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("Feed-forward decoding with dropout requires PyTorch.") from exc
        return torch, nn

    def _initialize(self, n_features, classes):
        torch, nn = self._torch_modules()
        torch.manual_seed(int(self.random_state))
        self.classes_ = np.asarray(classes, dtype=str)
        if self.classes_.size < 2:
            raise ValueError("Feed-forward decoding requires at least two classes.")
        self.n_features_in_ = int(n_features)
        hidden = tuple(max(1, int(v)) for v in self.hidden_layer_sizes if int(v) > 0)
        dims = (self.n_features_in_,) + hidden + (int(self.classes_.size),)
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
                if float(self.dropout) > 0:
                    layers.append(nn.Dropout(p=min(0.9, max(0.0, float(self.dropout)))))
        self.model_ = nn.Sequential(*layers).to("cpu")
        self.optimizer_ = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate_init),
            weight_decay=max(0.0, float(self.alpha)),
        )
        self.loss_fn_ = nn.CrossEntropyLoss(reduction="none")
        self._class_to_index_ = {str(label): idx for idx, label in enumerate(self.classes_.tolist())}
        self._rng_ = np.random.default_rng(int(self.random_state))
        self.n_iter_ = 0
        self.loss_ = np.nan
        self._sync_sklearn_weights()

    def _sync_sklearn_weights(self):
        if not hasattr(self, "model_"):
            return
        torch, nn = self._torch_modules()
        linear_layers = [module for module in self.model_ if isinstance(module, nn.Linear)]
        self.coefs_ = [module.weight.detach().cpu().numpy().T.copy() for module in linear_layers]
        self.intercepts_ = [module.bias.detach().cpu().numpy().copy() for module in linear_layers]

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        torch, _nn = self._torch_modules()
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y, dtype=str)
        if X.ndim != 2 or X.shape[0] != y.size:
            raise ValueError("Feed-forward decoder received inconsistent training data.")
        if not hasattr(self, "model_"):
            class_values = np.asarray(classes if classes is not None else pd.Index(y).drop_duplicates(), dtype=str)
            self._initialize(X.shape[1], class_values)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError("Feed-forward decoder feature count changed between epochs.")

        try:
            y_idx = np.asarray([self._class_to_index_[str(label)] for label in y], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Unknown feed-forward decoder class: {exc}") from exc
        weights = np.ones(y.size, dtype=np.float32)
        if sample_weight is not None:
            candidate = np.asarray(sample_weight, dtype=np.float32)
            if candidate.shape == weights.shape:
                weights = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 0.0)

        self.model_.train()
        order = self._rng_.permutation(y.size)
        batch_size = max(1, min(int(self.batch_size), y.size))
        losses = []
        for start in range(0, y.size, batch_size):
            idx = order[start:start + batch_size]
            xb = torch.from_numpy(X[idx])
            yb = torch.from_numpy(y_idx[idx])
            wb = torch.from_numpy(weights[idx])
            self.optimizer_.zero_grad(set_to_none=True)
            logits = self.model_(xb)
            per_sample = self.loss_fn_(logits, yb)
            denom = torch.clamp(wb.sum(), min=1e-12)
            loss = torch.sum(per_sample * wb) / denom
            loss.backward()
            self.optimizer_.step()
            losses.append(float(loss.detach().cpu()))
        self.n_iter_ += 1
        self.loss_ = float(np.mean(losses)) if losses else np.nan
        self._sync_sklearn_weights()
        return self

    def fit(self, X, y, sample_weight=None):
        if hasattr(self, "model_"):
            del self.model_
        classes = pd.Index(np.asarray(y, dtype=str)).drop_duplicates().to_numpy(dtype=str)
        for _ in range(max(1, int(self.max_iter))):
            self.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
            classes = None
        return self

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Feed-forward decoder has not been fit.")
        torch, _nn = self._torch_modules()
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_training_state(self):
        if not hasattr(self, "model_"):
            return None
        return {
            "model": {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()},
            "loss": float(self.loss_),
            "n_iter": int(self.n_iter_),
        }

    def set_training_state(self, state):
        if not state or not hasattr(self, "model_"):
            return
        self.model_.load_state_dict(state["model"])
        self.loss_ = float(state.get("loss", np.nan))
        self.n_iter_ = int(state.get("n_iter", self.n_iter_))
        self._sync_sklearn_weights()


@lru_cache(maxsize=128)
def _semantic_decision_mapping_for_location(location):
    """Return block_id -> line_2 for the instructions nearest a MEG run."""
    try:
        path = find_instruction_csv(location)
        if not path:
            return {}
        return read_instruction_csv(path)["block_mapping"]
    except (OSError, ValueError):
        # Missing/malformed instructions must never be replaced with a guessed
        # task label. The normal unlabeled-trial filtering handles this case.
        return {}


def semantic_decision_label_from_trial_row(row):
    """Look up a trial's semantic decision task from its block instructions."""
    location = row.get("source_fif_path", "")
    block_id = row.get("block_id", "")
    if pd.isna(location) or not str(location).strip() or pd.isna(block_id):
        return ""
    mapping = _semantic_decision_mapping_for_location(str(location).strip())
    return str(mapping.get(block_key(block_id), "")).strip()


def concept_label_from_trial_row(context, row, group_mode="canonical"):
    """Return the concept label used by concept decoding."""
    mode = str(group_mode or "canonical").strip().lower()
    if mode == "semantic_decision":
        return semantic_decision_label_from_trial_row(row)
    if mode == "individual":
        stim = row.get("stimulus")
        if pd.notna(stim) and str(stim).strip():
            return os.path.basename(str(stim).strip())
    if mode == "category":
        group = row.get("stimulus_group", "")
        if pd.notna(group) and str(group).strip():
            return str(group).strip()
    concept = row.get("concept")
    if pd.notna(concept) and str(concept).strip():
        return str(concept).strip()
    canonical = row.get("canonical_stimulus")
    if pd.notna(canonical) and str(canonical).strip():
        return str(canonical).strip()
    return context._canonicalize_meg_stimulus_label(row.get("stimulus"))


def feature_edges_for_concept_decoding(window_start, window_end, bin_size_ms):
    duration = float(window_end) - float(window_start)
    if duration <= 0:
        return []
    try:
        bin_size_s = float(bin_size_ms) / 1000.0
    except Exception:
        bin_size_s = 0.0
    if bin_size_s <= 0 or bin_size_s >= duration:
        return [(float(window_start), float(window_end))]
    edges = [float(window_start)]
    cursor = float(window_start)
    while cursor + bin_size_s < float(window_end) - 1e-12:
        cursor += bin_size_s
        edges.append(float(cursor))
    edges.append(float(window_end))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def concept_stat_names(feature_mode):
    mode = str(feature_mode or "mean_rms").strip().lower()
    if mode == "mean":
        return ["mean"]
    if mode == "rms":
        return ["rms"]
    if mode == "raw":
        return ["raw"]
    return ["mean", "rms"]


def concept_feature_binning_enabled(options):
    """Whether to aggregate time samples into feature bins."""
    if options is None:
        return True
    return bool(options.get("feature_binning_enabled", True))


def concept_resample_from_options(options):
    """Return optional MNE resampling settings for concept decoding epochs."""
    if options is None:
        options = {}
    enabled = bool(
        options.get(
            "mne_resample_enabled",
            options.get(
                "downsample_enabled",
                CONCEPT_PREPROCESSING_DEFAULTS["mne_resample_enabled"],
            ),
        )
    )
    if not enabled:
        return False, None
    raw_value = options.get(
        "resample_sfreq",
        options.get(
            "downsample_sfreq",
            CONCEPT_PREPROCESSING_DEFAULTS["resample_sfreq"],
        ),
    )
    try:
        sfreq = float(raw_value)
    except Exception as exc:
        raise ValueError("Downsample rate must be a valid number in Hz.") from exc
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Downsample rate must be greater than 0 Hz.")
    return True, float(sfreq)


def validate_concept_resample_settings(raw_by_path, enabled, target_sfreq, h_freq=None):
    """Validate that requested downsampling is possible for all loaded runs."""
    if not enabled or target_sfreq is None:
        return
    sfreqs = []
    for raw in (raw_by_path or {}).values():
        try:
            sfreqs.append(float(raw.info["sfreq"]))
        except Exception:
            pass
    if not sfreqs:
        raise ValueError("Downsampling requires valid FIF sampling rates.")
    min_sfreq = float(np.nanmin(sfreqs))
    max_sfreq = float(np.nanmax(sfreqs))
    target_sfreq = float(target_sfreq)
    if target_sfreq > min_sfreq + 1e-6:
        raise ValueError(
            f"Downsample rate ({target_sfreq:g} Hz) cannot exceed the lowest run sampling rate "
            f"({min_sfreq:g} Hz)."
        )
    if target_sfreq >= max_sfreq - 1e-6 and max_sfreq - min_sfreq > 1e-6:
        raise ValueError(
            "Downsampling is enabled, but the target rate would not align all runs. "
            "Choose a rate below every run's native sampling rate."
        )
    if h_freq is not None and np.isfinite(float(h_freq)) and float(h_freq) >= target_sfreq / 2.0:
        raise ValueError(
            f"Bandpass high cutoff ({float(h_freq):g} Hz) must be below the downsampled Nyquist "
            f"({target_sfreq / 2.0:g} Hz)."
        )


def concept_raw_for_path(context, fif_path, raw_cache):
    path = os.path.abspath(str(fif_path))
    if path in raw_cache:
        return raw_cache[path]
    current = os.path.abspath(str(getattr(context, "_fif_filename", "") or ""))
    if current and current == path and getattr(context, "_fif_raw", None) is not None:
        raw = context._fif_raw
    else:
        raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
    raw_cache[path] = raw
    return raw


def concept_bandpass_from_options(options):
    """Return validated shared analysis bandpass cutoffs from UI/options."""
    if options is None:
        options = {}
    try:
        l_freq = float(
            options.get(
                "bandpass_low_hz",
                CONCEPT_PREPROCESSING_DEFAULTS["bandpass_low_hz"],
            )
        )
    except Exception:
        l_freq = CONCEPT_PREPROCESSING_DEFAULTS["bandpass_low_hz"]
    try:
        h_freq = float(
            options.get(
                "bandpass_high_hz",
                CONCEPT_PREPROCESSING_DEFAULTS["bandpass_high_hz"],
            )
        )
    except Exception:
        h_freq = CONCEPT_PREPROCESSING_DEFAULTS["bandpass_high_hz"]

    l_freq = None if not np.isfinite(l_freq) or l_freq <= 0.0 else float(l_freq)
    h_freq = None if not np.isfinite(h_freq) or h_freq <= 0.0 else float(h_freq)
    if l_freq is not None and h_freq is not None and h_freq <= l_freq:
        raise ValueError("Bandpass high cutoff must be greater than low cutoff.")
    return l_freq, h_freq


def apply_concept_bandpass_to_raw(raw, l_freq, h_freq):
    """Return a filtered Raw copy for concept decoding without mutating shared state."""
    if raw is None or (l_freq is None and h_freq is None):
        return raw
    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    if h_freq is not None and h_freq >= nyq:
        raise ValueError(
            f"Bandpass high cutoff ({h_freq:g} Hz) must be below Nyquist "
            f"({nyq:g} Hz for sfreq={sfreq:g} Hz)."
        )
    filtered = raw.copy().load_data(verbose="error")
    filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="meg",
        n_jobs=1,
        verbose="error",
    )
    return filtered


def _float_cache_token(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return None
    return round(float(value), 9)


def _file_identity_for_epoch_cache(path):
    try:
        stat = os.stat(path)
        return int(stat.st_size), int(stat.st_mtime_ns)
    except Exception:
        return None


def _epoch_baseline_cache_token(epoch_baseline):
    if epoch_baseline is None:
        return None
    try:
        return tuple(_float_cache_token(v) for v in epoch_baseline)
    except Exception:
        return str(epoch_baseline)


def concept_epoch_reject_from_options(options=None):
    """Return an MNE Epochs reject dict using user-facing MEG units."""
    options = options or {}
    enabled = bool(
        options.get(
            "mne_epoch_reject_enabled",
            options.get(
                "epoch_reject_enabled",
                CONCEPT_PREPROCESSING_DEFAULTS["mne_epoch_reject_enabled"],
            ),
        )
    )
    if not enabled:
        return None

    reject = {}
    try:
        mag_ft = float(
            options.get(
                "reject_mag_ft",
                options.get(
                    "mne_reject_mag_ft",
                    CONCEPT_PREPROCESSING_DEFAULTS["reject_mag_ft"],
                ),
            )
        )
    except Exception:
        mag_ft = CONCEPT_PREPROCESSING_DEFAULTS["reject_mag_ft"]
    try:
        grad_ft_cm = float(
            options.get(
                "reject_grad_ft_cm",
                options.get(
                    "mne_reject_grad_ft_cm",
                    CONCEPT_PREPROCESSING_DEFAULTS["reject_grad_ft_cm"],
                ),
            )
        )
    except Exception:
        grad_ft_cm = CONCEPT_PREPROCESSING_DEFAULTS["reject_grad_ft_cm"]

    if np.isfinite(mag_ft) and mag_ft > 0:
        reject["mag"] = float(mag_ft) * 1e-15
    if np.isfinite(grad_ft_cm) and grad_ft_cm > 0:
        reject["grad"] = float(grad_ft_cm) * 1e-13
    return reject or None


def concept_epoch_reject_summary(epoch_reject):
    """Return user-facing reject thresholds for result tables."""
    if not epoch_reject:
        return {"enabled": False, "mag_ft": np.nan, "grad_ft_cm": np.nan}
    return {
        "enabled": True,
        "mag_ft": round(float(epoch_reject.get("mag", np.nan)) / 1e-15, 6) if "mag" in epoch_reject else np.nan,
        "grad_ft_cm": round(float(epoch_reject.get("grad", np.nan)) / 1e-13, 6) if "grad" in epoch_reject else np.nan,
    }


def filter_epoch_reject_for_picks(info, picks, epoch_reject):
    """Keep only reject thresholds for channel types present in selected picks."""
    if not epoch_reject:
        return None
    present_types = set()
    for pick in np.asarray(picks, dtype=int).tolist():
        try:
            present_types.add(str(mne.channel_type(info, int(pick))))
        except Exception:
            pass
    filtered = {
        str(key): float(value)
        for key, value in dict(epoch_reject).items()
        if str(key) in present_types and np.isfinite(float(value)) and float(value) > 0
    }
    return filtered or None


def _epoch_reject_cache_token(epoch_reject):
    if not epoch_reject:
        return None
    try:
        return tuple(sorted((str(key), _float_cache_token(value)) for key, value in dict(epoch_reject).items()))
    except Exception:
        return str(epoch_reject)


def _concept_epoch_cache(context):
    cache = getattr(context, "_concept_signal_epoch_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        try:
            setattr(context, "_concept_signal_epoch_cache", cache)
        except Exception:
            pass
    return cache


def _concept_epoch_cache_key(
    path_abs,
    raw,
    channel_names,
    bandpass_low_hz,
    bandpass_high_hz,
    epoch_tmin,
    epoch_tmax,
    epoch_baseline,
    events_arr,
    epoch_reject=None,
    resample_sfreq=None,
):
    try:
        sfreq = float(raw.info["sfreq"])
    except Exception:
        sfreq = np.nan
    try:
        first_samp = int(getattr(raw, "first_samp", 0))
    except Exception:
        first_samp = 0
    event_samples = tuple(int(v) for v in np.asarray(events_arr, dtype=np.int64)[:, 0].tolist())
    return (
        "concept_signal_epochs_v5",
        mne.__version__,
        CONCEPT_RESAMPLE_PADDING_S,
        os.path.abspath(str(path_abs)),
        _file_identity_for_epoch_cache(path_abs),
        tuple((str(p), _file_identity_for_epoch_cache(p)) for p in (getattr(raw, "filenames", ()) or ()) if p),
        tuple(str(v) for v in raw.info.get("bads", [])),
        (_float_cache_token(raw.info.get("highpass")), _float_cache_token(raw.info.get("lowpass"))),
        hashlib.sha256(repr([
            (p.get("active"), p.get("desc"), p.get("data", {}).get("col_names"),
             np.asarray(p.get("data", {}).get("data", [])).tobytes())
            for p in raw.info.get("projs", [])
        ]).encode()).hexdigest(),
        hashlib.sha256(repr([
            (float(a["onset"]), float(a["duration"]), str(a["description"]), tuple(a.get("ch_names", ())))
            for a in raw.annotations
        ] + [str(raw.annotations.orig_time)]).encode()).hexdigest(),
        _float_cache_token(sfreq),
        int(first_samp),
        int(raw.n_times),
        tuple(str(ch) for ch in channel_names),
        _float_cache_token(bandpass_low_hz),
        _float_cache_token(bandpass_high_hz),
        _float_cache_token(epoch_tmin),
        _float_cache_token(epoch_tmax),
        _epoch_baseline_cache_token(epoch_baseline),
        _epoch_reject_cache_token(epoch_reject),
        _float_cache_token(resample_sfreq),
        event_samples,
    )


def concept_epoch_cache_directory():
    """Local application cache, outside synced recording/project folders."""
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return str(root / "MAS" / "concept_epochs")


def _concept_epoch_disk_cache_path(options, cache_key):
    if not (options or {}).get("enable_signal_epoch_disk_cache", True):
        return None
    cache_dir = str((options or {}).get("signal_epoch_cache_dir", "") or "").strip()
    if not cache_dir:
        cache_dir = concept_epoch_cache_directory()
    try:
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
        return os.path.join(cache_dir, f"concept_epochs_{digest}.npz")
    except Exception:
        return None


def _load_concept_epoch_disk_cache(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            drop_log_json = str(data["drop_log_json"].tolist())
            return {
                "epoch_data": np.asarray(data["epoch_data"], dtype=np.float32),
                "times": np.asarray(data["times"], dtype=float),
                "selected_idx": np.asarray(data["selected_idx"], dtype=int),
                "drop_log": [tuple(str(v) for v in row) for row in json.loads(drop_log_json)],
                "sfreq": float(np.asarray(data["sfreq"]).reshape(-1)[0]),
            }
    except Exception:
        return None


def _save_concept_epoch_disk_cache(path, payload):
    temporary_path = None
    try:
        drop_log_json = json.dumps([list(row) for row in payload.get("drop_log", [])])
        handle, temporary_path = tempfile.mkstemp(prefix=".concept_epochs_", suffix=".npz", dir=os.path.dirname(path))
        os.close(handle)
        np.savez(
            temporary_path,
            epoch_data=np.asarray(payload.get("epoch_data"), dtype=np.float32),
            times=np.asarray(payload.get("times"), dtype=float),
            selected_idx=np.asarray(payload.get("selected_idx"), dtype=int),
            drop_log_json=np.asarray(drop_log_json),
            sfreq=np.asarray([float(payload.get("sfreq", np.nan))], dtype=float),
        )
        os.replace(temporary_path, path)
        return True
    except Exception:
        return False
    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _prune_concept_epoch_disk_cache(directory, max_bytes=10 * 1024**3, keep=None):
    """Bound this cache to 10 GiB, removing least recently used entries first."""
    try:
        entries = [(p.stat().st_mtime_ns, p.stat().st_size, p)
                   for p in Path(directory).glob("concept_epochs_*.npz")]
        entries.extend(
            (p.stat().st_mtime_ns, sum(f.stat().st_size for f in p.iterdir() if f.is_file()), p)
            for p in Path(directory).glob("recording_*") if p.is_dir()
        )
        entries.sort(key=lambda item: item[0])
        total = sum(size for _, size, _ in entries)
        for _, size, path in entries:
            if total <= max_bytes:
                break
            if keep is not None and path == Path(keep):
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            total -= size
    except OSError:
        pass  # Cache maintenance must not prevent decoding.


def concept_preprocessed_recording(raw, path_abs, options, low, high, target_sfreq):
    """Cache continuous preprocessing independently of events and epoch settings.

    The original remains untouched. A directory holds the FIF (including any
    split parts) and is published only after saving completes. Reads are lazy.
    """
    options = options or {}
    native_sfreq = float(raw.info["sfreq"])
    target = native_sfreq if target_sfreq is None else float(target_sfreq)
    if not np.isfinite(target) or target <= 0 or target > native_sfreq + 1e-6:
        raise ValueError("Recording cache target rate must be positive and no higher than the native rate.")
    identity = _concept_epoch_cache_key(
        path_abs, raw, raw.ch_names, low, high, 0.0, 0.0, None,
        np.empty((0, 3), dtype=np.int64), resample_sfreq=target,
    )
    key = ("concept_continuous_v1", identity)
    cache_path = None
    if options.get("enable_signal_epoch_cache", True) and os.path.isfile(path_abs):
        placeholder = _concept_epoch_disk_cache_path(options, key)
        if placeholder:
            digest = hashlib.sha256(repr(key).encode()).hexdigest()
            cache_path = Path(placeholder).parent / f"recording_{digest}"
    if cache_path is not None and cache_path.is_dir():
        try:
            cached = mne.io.read_raw_fif(cache_path / "prepared_raw.fif", preload=False, verbose="error")
            if not np.isclose(cached.info["sfreq"], target, rtol=0, atol=1e-6):
                raise ValueError("Cached recording sampling rate mismatch")
            if cached.ch_names != raw.ch_names:
                raise ValueError("Cached recording channel mismatch")
            try:
                os.utime(cache_path, None)
            except OSError:
                pass
            emit_concept_progress(options, f"reusing filtered/downsampled recording: {os.path.basename(path_abs)}")
            return cached
        except Exception:
            # Only remove our own incomplete/corrupt entry, then rebuild it.
            shutil.rmtree(cache_path, ignore_errors=True)

    emit_concept_progress(options, f"preparing continuous recording: {os.path.basename(path_abs)}")
    prepared = apply_concept_bandpass_to_raw(raw, low, high)
    if prepared is raw:
        prepared = raw.copy()
    prepared.load_data(verbose="error")
    if target < native_sfreq - 1e-6:
        emit_concept_progress(options, f"resampling recording once: {native_sfreq:g} Hz -> {target:g} Hz")
        # Events are mapped independently below using MNE's same sample rule,
        # so new event selections do not require resampling the recording again.
        prepared, _ = prepared.resample(
            target, events=np.empty((0, 3), dtype=np.int64), npad="auto", n_jobs=1, verbose="error",
        )
    if cache_path is not None:
        temporary_dir = None
        try:
            # Avoid saving an entry larger than the entire shared cache budget.
            if prepared.n_times * len(prepared.ch_names) * 8 > 10 * 1024**3:
                emit_concept_progress(options, "Prepared recording exceeds the 10 GiB cache budget; using it without saving.")
                return prepared
            temporary_dir = tempfile.mkdtemp(prefix=".recording_", dir=cache_path.parent)
            prepared.save(Path(temporary_dir) / "prepared_raw.fif", fmt="double", overwrite=True, verbose="error")
            if cache_path.exists():
                # Another worker may have completed the same entry meanwhile.
                shutil.rmtree(temporary_dir)
            else:
                os.rename(temporary_dir, cache_path)
            temporary_dir = None
            # Parallel source readers reopen this lazy FIF for each batch.
            # Defer eviction until all workers finish so their inputs survive.
            if not options.get("_defer_recording_cache_prune", False):
                _prune_concept_epoch_disk_cache(cache_path.parent, keep=cache_path)
            emit_concept_progress(options, f"saved filtered/downsampled recording: {cache_path}")
            # Use the same serialized representation on the first and later
            # runs (FIF metadata has finite precision); read only needed epochs.
            return mne.io.read_raw_fif(cache_path / "prepared_raw.fif", preload=False, verbose="error")
        except Exception as exc:
            emit_concept_progress(options, f"Could not save recording cache; continuing in memory: {exc}")
        finally:
            if temporary_dir:
                shutil.rmtree(temporary_dir, ignore_errors=True)
    return prepared


def _events_for_preprocessed_recording(events, original_raw, prepared_raw):
    """Match Raw.resample(events=...) rounding, including first_samp offsets."""
    mapped = np.asarray(events, dtype=np.int64).copy()
    ratio = float(prepared_raw.info["sfreq"]) / float(original_raw.info["sfreq"])
    if not np.isclose(ratio, 1.0, rtol=0, atol=1e-12):
        mapped[:, 0] = np.minimum(
            np.round(mapped[:, 0] * ratio).astype(np.int64),
            prepared_raw.n_times + prepared_raw.first_samp - 1,
        )
    return mapped


def concept_cached_epochs_for_file(
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
    epoch_reject=None,
    resample_sfreq=None,
    *,
    label="signal epochs",
):
    """Return epochs from a reusable filtered/downsampled recording by default.

    Events are remapped from their original sample indices. Baseline correction
    and rejection use the requested window on the prepared recording. The legacy
    preprocess_continuous_recording=False path pads epochs before resampling and
    crops afterward, retaining native-grid event alignment and rejection.
    """
    if raw is None:
        return None, "Raw data is unavailable.", False
    path_abs = os.path.abspath(str(path_abs))
    events_arr = np.asarray(events_arr, dtype=np.int64)
    if events_arr.ndim != 2 or events_arr.shape[0] == 0 or events_arr.shape[1] < 3:
        return None, "No valid events were provided for epoching.", False
    picks = mne.pick_channels(raw.ch_names, include=list(channel_names), ordered=True)
    if len(picks) != len(channel_names):
        return None, "common good MEG channel set is incomplete", False
    epoch_reject = filter_epoch_reject_for_picks(raw.info, picks, epoch_reject)

    cache_enabled = bool((options or {}).get("enable_signal_epoch_cache", True))
    try:
        max_entries = max(1, int((options or {}).get("signal_epoch_cache_max_entries", 24)))
    except Exception:
        max_entries = 24
    cache_key = _concept_epoch_cache_key(
        path_abs,
        raw,
        channel_names,
        bandpass_low_hz,
        bandpass_high_hz,
        epoch_tmin,
        epoch_tmax,
        epoch_baseline,
        events_arr,
        epoch_reject=epoch_reject,
        resample_sfreq=resample_sfreq,
    )
    continuous = bool((options or {}).get("preprocess_continuous_recording", True))
    cache_key = ("continuous" if continuous else "epochwise", cache_key)
    cache = _concept_epoch_cache(context) if cache_enabled else OrderedDict()
    if cache_enabled and cache_key in cache:
        cached = cache.pop(cache_key)
        cache[cache_key] = cached
        emit_concept_progress(options, f"using cached bandpass epochs for {os.path.basename(path_abs)}")
        return cached, None, True

    # Persistent entries need identifiable source files; never reuse an entry
    # for an unsaved/synthetic Raw object based on a nonexistent path.
    disk_cache_path = (
        _concept_epoch_disk_cache_path(options, cache_key)
        if cache_enabled and os.path.isfile(path_abs) else None
    )
    if disk_cache_path and os.path.isfile(disk_cache_path):
        cached = _load_concept_epoch_disk_cache(disk_cache_path)
        if cached is not None:
            try:
                os.utime(disk_cache_path, None)
            except OSError:
                pass
            cache[cache_key] = cached
            while len(cache) > max_entries:
                try:
                    cache.popitem(last=False)
                except Exception:
                    break
            emit_concept_progress(options, f"using disk-cached bandpass epochs for {os.path.basename(path_abs)}")
            return cached, None, True

    emit_concept_progress(options, f"filtering/epoching {label}: {os.path.basename(path_abs)}")
    try:
        if continuous:
            epoch_raw = concept_preprocessed_recording(
                raw, path_abs, options, bandpass_low_hz, bandpass_high_hz, resample_sfreq,
            )
            epoch_events = _events_for_preprocessed_recording(events_arr, raw, epoch_raw)
        else:
            epoch_raw = apply_concept_bandpass_to_raw(raw, bandpass_low_hz, bandpass_high_hz)
            epoch_events = events_arr
        current_sfreq = float(epoch_raw.info["sfreq"])
        target_sfreq = float(resample_sfreq) if resample_sfreq is not None else current_sfreq
        if not np.isfinite(target_sfreq) or target_sfreq <= 0:
            return None, "Downsample rate must be finite and greater than 0 Hz.", False
        if target_sfreq > current_sfreq + 1e-6:
            return None, (
                f"downsample target ({target_sfreq:g} Hz) is above this run's sampling rate "
                f"({current_sfreq:g} Hz)"
            ), False
        downsampling = target_sfreq < current_sfreq - 1e-6
        decim = int(round(current_sfreq / target_sfreq))
        # MNE recommends continuous low-pass filtering followed by epoch
        # decimation when the rate ratio is integral. Require an explicit
        # filter applied above, not merely lowpass metadata on the input file.
        # Also account for MNE's default minimum 2 Hz transition bandwidth.
        high = float(bandpass_high_hz) if bandpass_high_hz is not None else np.nan
        fast_decimation = (
            downsampling
            and decim >= 2
            and np.isclose(current_sfreq / decim, target_sfreq, rtol=0, atol=1e-9)
            and np.isfinite(high)
            and 0 < high <= target_sfreq / 3.0
            and high + max(0.25 * high, 2.0) < target_sfreq / 2.0
            and all(mne.channel_type(epoch_raw.info, int(p)) in ("mag", "grad") for p in picks)
        )
        # Round up to a whole target sample so padding does not shift the
        # retained time grid for integer-ratio sampling rates.
        padding_s = (
            np.ceil(CONCEPT_RESAMPLE_PADDING_S * target_sfreq) / target_sfreq
            if downsampling else 0.0
        )
        # Resolve open-ended baseline intervals before extending the epoch.
        baseline = epoch_baseline
        if baseline is not None:
            baseline = (
                float(epoch_tmin) if baseline[0] is None else baseline[0],
                float(epoch_tmax) if baseline[1] is None else baseline[1],
            )
        epochs = mne.Epochs(
            epoch_raw,
            epoch_events,
            event_id={"trial": 1},
            tmin=float(epoch_tmin) - padding_s,
            tmax=float(epoch_tmax) + padding_s,
            baseline=baseline,
            reject=None,
            picks=picks,
            preload=True,
            reject_by_annotation=True,
            event_repeated="drop",
            verbose="error",
        )
        # Loading checks annotations across the full padded interval. MNE also
        # uses reject_tmin/tmax for annotation checks, so limit the amplitude
        # rejection window only after loading has completed.
        if epoch_reject:
            epochs.reject_tmin = float(epoch_tmin)
            epochs.reject_tmax = float(epoch_tmax)
            epochs.drop_bad(reject=epoch_reject, verbose="error")
        if downsampling and len(epochs):
            method_label = f"decimating by {decim}" if fast_decimation else "FFT resampling"
            emit_concept_progress(
                options,
                f"{method_label} {label}: {current_sfreq:g} Hz -> {target_sfreq:g} Hz "
                f"with {padding_s:g} s context per side; cropping to requested window",
            )
            if fast_decimation:
                epochs.decimate(decim, verbose="error")
            else:
                epochs.resample(target_sfreq, npad="auto", n_jobs=1, verbose="error")
            epochs.crop(tmin=float(epoch_tmin), tmax=float(epoch_tmax), verbose="error")
    except Exception as exc:
        return None, f"epoching failed: {exc}", False

    if len(epochs):
        epoch_data = np.asarray(epochs.get_data(copy=False), dtype=np.float32)
        times = np.asarray(epochs.times, dtype=float)
    else:
        # MNE cannot resample an empty Epochs object. Preserve per-trial drop
        # reasons even when every trial lacks valid padding/context.
        times = np.arange(
            int(round(float(epoch_tmin) * target_sfreq)),
            int(round(float(epoch_tmax) * target_sfreq)) + 1,
        ) / target_sfreq
        epoch_data = np.empty((0, len(picks), len(times)), dtype=np.float32)
    payload = {
        "epoch_data": epoch_data,
        "times": times,
        "selected_idx": np.asarray(getattr(epochs, "selection", np.arange(len(epochs))), dtype=int),
        "drop_log": [
            tuple(str(v) for v in entry if str(v))
            for entry in list(getattr(epochs, "drop_log", []))
        ],
        "sfreq": target_sfreq,
    }
    if cache_enabled:
        cache[cache_key] = payload
        while len(cache) > max_entries:
            try:
                cache.popitem(last=False)
            except Exception:
                break
        if disk_cache_path:
            if _save_concept_epoch_disk_cache(disk_cache_path, payload):
                _prune_concept_epoch_disk_cache(os.path.dirname(disk_cache_path))
                emit_concept_progress(options, f"saved preprocessed decoding inputs: {disk_cache_path}")
            else:
                emit_concept_progress(options, "Could not save decoding inputs to disk; keeping the in-memory cache.")
    return payload, None, False


def concept_channel_names_for_raw(raw, channel_mode):
    mode = str(channel_mode or "all").strip().lower()
    meg_arg = True
    if mode == "mag":
        meg_arg = "mag"
    elif mode == "grad":
        meg_arg = "grad"
    names, _bad_names = good_meg_channel_names(raw.info, meg=meg_arg, ref_meg=False)
    return names


def common_concept_channels(raw_by_path, channel_mode, max_channels=0):
    common = None
    for raw in raw_by_path.values():
        names = concept_channel_names_for_raw(raw, channel_mode)
        if not names:
            continue
        if common is None:
            common = list(names)
        else:
            valid = set(names)
            common = [name for name in common if name in valid]
    if not common:
        return []
    try:
        max_channels = int(max_channels)
    except Exception:
        max_channels = 0
    if max_channels > 0:
        common = common[:max_channels]
    return common


def signal_epoch_bin_masks(times, bin_edges):
    """Return boolean time masks for feature bins on an Epochs time grid."""
    times = np.asarray(times, dtype=float)
    masks = []
    for idx, (bin_start, bin_end) in enumerate(bin_edges):
        if idx == len(bin_edges) - 1:
            mask = (times >= float(bin_start)) & (times <= float(bin_end))
        else:
            mask = (times >= float(bin_start)) & (times < float(bin_end))
        masks.append(mask)
    return masks


def robust_zscore(values):
    """Median/MAD robust z-score."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if np.sum(finite) < 4:
        return out
    vals = arr[finite]
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    if not np.isfinite(mad) or mad <= 1e-18:
        std = float(np.nanstd(vals))
        if not np.isfinite(std) or std <= 1e-18:
            return out
        out[finite] = (vals - med) / std
        return out
    out[finite] = 0.67448975 * (vals - med) / mad
    return out


def signal_feature_names_and_meta(channel_names, stat_names, bin_edges, times=None):
    """Build signal-space feature names and metadata from an Epochs time grid."""
    feature_names = []
    feature_meta = []
    raw_sample_counts = []
    bin_masks = None
    if "raw" in stat_names:
        if times is None:
            raise RuntimeError("Raw signal features require an Epochs time grid.")
        times = np.asarray(times, dtype=float)
        bin_masks = signal_epoch_bin_masks(times, bin_edges)

    for bin_idx, (bin_start, bin_end) in enumerate(bin_edges):
        if "raw" in stat_names:
            mask = np.asarray(bin_masks[bin_idx], dtype=bool)
            bin_times = np.asarray(times, dtype=float)[mask]
            raw_sample_counts.append(int(bin_times.size))
            if bin_times.size < 2:
                raise RuntimeError("Raw sample feature bins must contain at least two samples.")
            for ch_name in channel_names:
                for sample_idx, rel_time_s in enumerate(bin_times.tolist()):
                    feature_names.append(
                        f"{ch_name}|raw|{bin_start:.3f}-{bin_end:.3f}s|sample_{sample_idx}"
                    )
                    feature_meta.append(
                        {
                            "feature": feature_names[-1],
                            "channel": ch_name,
                            "stat": "raw",
                            "bin_start_s": float(bin_start),
                            "bin_end_s": float(bin_end),
                            "sample_index": int(sample_idx),
                            "relative_time_s": float(rel_time_s),
                        }
                    )
            continue

        for stat_name in stat_names:
            for ch_name in channel_names:
                feature_names.append(f"{ch_name}|{stat_name}|{bin_start:.3f}-{bin_end:.3f}s")
                feature_meta.append(
                    {
                        "feature": feature_names[-1],
                        "channel": ch_name,
                        "stat": stat_name,
                        "bin_start_s": float(bin_start),
                        "bin_end_s": float(bin_end),
                    }
                )
    return feature_names, feature_meta, raw_sample_counts


def signal_trial_metric_rows(epoch_data, times, bin_edges, input_df, path_abs):
    """Compute compact per-trial QC metrics from epoch data."""
    data = np.asarray(epoch_data, dtype=float)
    times = np.asarray(times, dtype=float)
    if data.ndim != 3:
        return pd.DataFrame()
    n_epochs = int(data.shape[0])
    rows = []
    response_masks = signal_epoch_bin_masks(times, bin_edges)
    response_mask = np.any(np.vstack(response_masks), axis=0) if response_masks else np.ones(times.shape, dtype=bool)
    baseline_mask = (times >= -0.1) & (times <= 0.0)
    if not np.any(response_mask):
        response_mask = np.ones(times.shape, dtype=bool)

    for epoch_idx in range(n_epochs):
        trial = np.asarray(data[epoch_idx], dtype=float)
        finite = np.isfinite(trial)
        finite_fraction = float(np.mean(finite)) if trial.size else 0.0
        trial_filled = np.nan_to_num(trial, nan=0.0, posinf=0.0, neginf=0.0)
        resp = trial_filled[:, response_mask]
        base = trial_filled[:, baseline_mask] if np.any(baseline_mask) else np.empty((trial.shape[0], 0))
        ch_rms = np.sqrt(np.nanmean(np.square(resp), axis=1)) if resp.size else np.full(trial.shape[0], np.nan)
        ch_std = np.nanstd(resp, axis=1) if resp.size else np.full(trial.shape[0], np.nan)
        ch_ptp = np.ptp(resp, axis=1) if resp.size else np.full(trial.shape[0], np.nan)
        base_rms = (
            np.sqrt(np.nanmean(np.square(base), axis=1))
            if base.size
            else np.full(trial.shape[0], np.nan)
        )
        row = input_df.iloc[epoch_idx].to_dict() if epoch_idx < len(input_df) else {}
        row.update(
            {
                "source_fif_path": str(path_abs),
                "qc_status": "candidate",
                "qc_reason": "",
                "qc_epoch_index": int(epoch_idx),
                "qc_finite_fraction": finite_fraction,
                "qc_max_abs": float(np.nanmax(np.abs(resp))) if resp.size else np.nan,
                "qc_ptp_max": float(np.nanmax(ch_ptp)) if np.any(np.isfinite(ch_ptp)) else np.nan,
                "qc_ptp_median": float(np.nanmedian(ch_ptp)) if np.any(np.isfinite(ch_ptp)) else np.nan,
                "qc_rms_median": float(np.nanmedian(ch_rms)) if np.any(np.isfinite(ch_rms)) else np.nan,
                "qc_rms_max": float(np.nanmax(ch_rms)) if np.any(np.isfinite(ch_rms)) else np.nan,
                "qc_std_median": float(np.nanmedian(ch_std)) if np.any(np.isfinite(ch_std)) else np.nan,
                "qc_baseline_rms_median": (
                    float(np.nanmedian(base_rms)) if np.any(np.isfinite(base_rms)) else np.nan
                ),
                "qc_outlier": False,
                "qc_outlier_score": np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_signal_trial_outliers(qc_df, threshold=6.0, min_trials=8):
    """Mark robust outliers in a trial QC dataframe."""
    if not isinstance(qc_df, pd.DataFrame) or qc_df.empty:
        return qc_df
    out = qc_df.copy()
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 6.0
    try:
        min_trials = max(4, int(min_trials))
    except Exception:
        min_trials = 8
    if not np.isfinite(threshold) or threshold <= 0:
        out["qc_outlier"] = False
        out["qc_outlier_score"] = np.nan
        return out

    metric_cols = ["qc_max_abs", "qc_ptp_max", "qc_rms_median", "qc_std_median"]
    out["qc_outlier"] = False
    out["qc_outlier_score"] = np.nan
    for _path, idx in out.groupby("source_fif_path", sort=False).groups.items():
        idx = list(idx)
        if len(idx) < min_trials:
            continue
        scores = []
        for col in metric_cols:
            vals = pd.to_numeric(out.loc[idx, col], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.log10(np.maximum(np.abs(vals), 1e-30))
            z = np.abs(robust_zscore(vals))
            scores.append(z)
        if not scores:
            continue
        score = np.nanmax(np.vstack(scores), axis=0)
        out.loc[idx, "qc_outlier_score"] = score
        out.loc[idx, "qc_outlier"] = np.isfinite(score) & (score > threshold)
    return out


def build_signal_space_concept_features(context, options=None):
    """Extract trial-level signal-space MEG features across mapped FIF runs."""
    if options is None:
        options = {}
    emit_concept_progress(options, "building signal-space trial features...")
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
            options.get("bin_size_ms", 50),
        )
    else:
        bin_edges = [(float(window_start), float(window_end))]
    if not bin_edges:
        return None, "Concept decoding feature bins could not be constructed."
    stat_names = concept_stat_names(options.get("feature_mode", "mean_rms"))
    if not feature_binning:
        stat_names = ["raw"]
    baseline_correct = bool(options.get("baseline_correct", True))
    epoch_reject = concept_epoch_reject_from_options(options)
    reject_summary = concept_epoch_reject_summary(epoch_reject)
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
        return None, "No valid concept-labeled trials are available for decoding."

    counts = df["concept_label"].value_counts()
    keep_concepts = counts[counts >= min_trials].index.tolist()
    if len(keep_concepts) < 2:
        return None, "Need at least two concepts with enough trials for signal-space decoding."
    df = df[df["concept_label"].isin(keep_concepts)].copy().reset_index(drop=True)

    raw_cache = {}
    raw_by_path = {}
    skipped_paths = []
    for path in pd.Index(df["source_fif_path"].astype(str)).drop_duplicates().tolist():
        path_abs = os.path.abspath(str(path))
        try:
            emit_concept_progress(options, f"loading MEG run {len(raw_by_path) + 1}: {os.path.basename(path_abs)}")
            raw = concept_raw_for_path(context, path_abs, raw_cache)
            raw_by_path[path_abs] = raw
        except Exception as exc:
            skipped_paths.append((path_abs, str(exc)))
    if not raw_by_path:
        detail = f" First error: {skipped_paths[0][1]}" if skipped_paths else ""
        return None, f"Could not load/filter any FIF files for signal-space decoding.{detail}"
    try:
        validate_concept_resample_settings(raw_by_path, resample_enabled, resample_sfreq, h_freq=bandpass_high_hz)
    except ValueError as exc:
        return None, str(exc)

    meg_arg_for_log = True
    if channel_mode == "mag":
        meg_arg_for_log = "mag"
    elif channel_mode == "grad":
        meg_arg_for_log = "grad"
    log_seen = set()
    logger = getattr(getattr(context, "c", None), "log", None)
    for path_abs, raw in raw_by_path.items():
        _names, bad_names = good_meg_channel_names(raw.info, meg=meg_arg_for_log, ref_meg=False)
        key = (path_abs, tuple(bad_names))
        if bad_names and key not in log_seen:
            emit_bad_meg_channel_log(
                logger,
                "Signal-space decoding",
                fif_path=path_abs,
                bad_names=bad_names,
            )
            log_seen.add(key)

    available_paths = set(raw_by_path.keys())
    df["source_fif_path"] = df["source_fif_path"].map(lambda p: os.path.abspath(str(p)))
    df = df[df["source_fif_path"].isin(available_paths)].copy().reset_index(drop=True)
    if df.empty:
        return None, "No trials remained after loading available FIF files."

    channel_names = common_concept_channels(raw_by_path, channel_mode, max_channels=max_channels)
    if not channel_names:
        return None, f"No common {channel_mode} MEG channels are available across decoded runs."

    if "raw" in stat_names and not resample_enabled:
        sfreqs = []
        for raw in raw_by_path.values():
            try:
                sfreqs.append(float(raw.info["sfreq"]))
            except Exception:
                pass
        if not sfreqs:
            return None, "Raw sample features require valid FIF sampling rates."
        if max(sfreqs) - min(sfreqs) > 1e-6:
            return None, "Raw sample features require the same sampling rate across all decoded runs."

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

    epoch_tmin = min(float(window_start), -0.1 if baseline_correct else float(window_start))
    epoch_tmax = float(window_end)
    epoch_baseline = (-0.1, 0.0) if baseline_correct and epoch_tmin <= -0.1 and epoch_tmax >= 0.0 else None

    feature_names = None
    feature_meta = None
    expected_raw_sample_counts = None

    feature_rows = []
    kept_rows = []
    trial_qc_rows = []
    dropped_trials = 0
    epoch_cache_hits = 0
    epoch_cache_misses = 0
    df["_signal_qc_uid"] = np.arange(len(df), dtype=int)
    grouped_files = list(df.groupby("source_fif_path", sort=False))
    for file_idx, (path_abs, file_df) in enumerate(grouped_files, start=1):
        emit_concept_progress(
            options,
            f"extracting signal features run {file_idx}/{len(grouped_files)}: {os.path.basename(path_abs)}",
        )
        raw = raw_by_path.get(path_abs)
        if raw is None:
            continue
        sfreq = float(raw.info["sfreq"])
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
            label="signal feature epochs",
        )
        if cache_hit:
            epoch_cache_hits += 1
        else:
            epoch_cache_misses += 1
        if epoch_payload is None:
            skipped_paths.append((path_abs, epoch_err or "epoching failed"))
            dropped_trials += len(input_df)
            for row_idx, row in input_df.iterrows():
                dropped = row.to_dict()
                dropped.update(
                    {
                        "source_fif_path": str(path_abs),
                        "qc_status": "dropped",
                        "qc_reason": str(epoch_err or "epoching_failed"),
                        "qc_outlier": False,
                        "qc_outlier_score": np.nan,
                    }
                )
                trial_qc_rows.append(dropped)
            continue

        selected_idx = np.asarray(epoch_payload.get("selected_idx", []), dtype=int)
        selected_set = set(selected_idx.tolist())
        drop_log = list(epoch_payload.get("drop_log", []))
        mne_dropped_count = 0
        for original_idx, row in input_df.iterrows():
            if int(original_idx) in selected_set:
                continue
            mne_dropped_count += 1
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
        epoch_data = np.asarray(epoch_payload.get("epoch_data", np.empty((0, 0, 0))), dtype=np.float32)
        if epoch_data.shape[0] == 0:
            dropped_trials += int(mne_dropped_count)
            continue
        dropped_trials += int(mne_dropped_count)

        epoch_input_df = input_df.iloc[selected_idx].reset_index(drop=True)
        times = np.asarray(epoch_payload.get("times", []), dtype=float)
        try:
            current_feature_names, current_feature_meta, current_raw_counts = signal_feature_names_and_meta(
                channel_names,
                stat_names,
                bin_edges,
                times=times if "raw" in stat_names else None,
            )
        except Exception as exc:
            skipped_paths.append((path_abs, f"feature grid failed: {exc}"))
            dropped_trials += len(epoch_input_df)
            for _, row in epoch_input_df.iterrows():
                dropped = row.to_dict()
                dropped.update(
                    {
                        "source_fif_path": str(path_abs),
                        "qc_status": "dropped",
                        "qc_reason": f"feature_grid_failed: {exc}",
                        "qc_outlier": False,
                        "qc_outlier_score": np.nan,
                    }
                )
                trial_qc_rows.append(dropped)
            continue
        if feature_names is None:
            feature_names = current_feature_names
            feature_meta = current_feature_meta
            expected_raw_sample_counts = list(current_raw_counts)
        elif current_feature_names != feature_names:
            skipped_paths.append((path_abs, "signal epoch feature grid differs from first decoded run"))
            dropped_trials += len(epoch_input_df)
            for _, row in epoch_input_df.iterrows():
                dropped = row.to_dict()
                dropped.update(
                    {
                        "source_fif_path": str(path_abs),
                        "qc_status": "dropped",
                        "qc_reason": "feature_grid_mismatch",
                        "qc_outlier": False,
                        "qc_outlier_score": np.nan,
                    }
                )
                trial_qc_rows.append(dropped)
            continue
        if "raw" in stat_names and list(current_raw_counts) != list(expected_raw_sample_counts or []):
            skipped_paths.append((path_abs, "raw sample bins differ from first decoded run"))
            dropped_trials += len(epoch_input_df)
            continue

        qc_df = signal_trial_metric_rows(epoch_data, times, bin_edges, epoch_input_df, path_abs)
        qc_df = annotate_signal_trial_outliers(
            qc_df,
            threshold=trial_outlier_z,
            min_trials=min_outlier_trials,
        )
        bin_masks = signal_epoch_bin_masks(times, bin_edges)
        for epoch_idx in range(epoch_data.shape[0]):
            qc_row = qc_df.iloc[epoch_idx].to_dict() if epoch_idx < len(qc_df) else epoch_input_df.iloc[epoch_idx].to_dict()
            is_outlier = bool(qc_row.get("qc_outlier", False))
            if is_outlier and exclude_qc_outliers:
                qc_row["qc_status"] = "dropped"
                qc_row["qc_reason"] = "qc_outlier"
                trial_qc_rows.append(qc_row)
                dropped_trials += 1
                continue

            trial = np.asarray(epoch_data[epoch_idx], dtype=float)
            if trial.ndim != 2 or trial.shape[0] != len(channel_names):
                qc_row["qc_status"] = "dropped"
                qc_row["qc_reason"] = "invalid_epoch_shape"
                trial_qc_rows.append(qc_row)
                dropped_trials += 1
                continue
            values = []
            for bin_idx, mask in enumerate(bin_masks):
                segment = trial[:, mask]
                if segment.ndim != 2 or segment.shape[1] < 2:
                    values = []
                    break
                if "raw" in stat_names:
                    values.extend(segment.reshape(-1).tolist())
                    continue
                if "mean" in stat_names:
                    values.extend(np.nanmean(segment, axis=1).tolist())
                if "rms" in stat_names:
                    values.extend(np.sqrt(np.nanmean(np.square(segment), axis=1)).tolist())
            if not values or len(values) != len(feature_names):
                qc_row["qc_status"] = "dropped"
                qc_row["qc_reason"] = "feature_extraction_failed"
                trial_qc_rows.append(qc_row)
                dropped_trials += 1
                continue

            qc_row["qc_status"] = "kept"
            qc_row["qc_reason"] = "qc_outlier_not_excluded" if is_outlier else ""
            trial_qc_rows.append(qc_row)
            feature_rows.append(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
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

    if not feature_rows:
        return None, "No usable trial features were extracted from signal-space data."

    X = np.vstack(feature_rows)
    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    trial_qc_df = pd.DataFrame(trial_qc_rows).reset_index(drop=True)
    if not trial_qc_df.empty:
        sort_cols = [col for col in ("source_fif_path", "trial_index_in_file", "trial_index", "onset_sample") if col in trial_qc_df.columns]
        if sort_cols:
            trial_qc_df = trial_qc_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    y = kept_df["concept_label"].astype(str).to_numpy()
    post_counts = pd.Series(y).value_counts()
    keep_after_extract = post_counts[post_counts >= min_trials].index.tolist()
    if len(keep_after_extract) < 2:
        return None, "Need at least two concepts with enough valid extracted trials."
    keep_mask = np.isin(y, keep_after_extract)
    if isinstance(trial_qc_df, pd.DataFrame) and "_signal_qc_uid" in kept_df.columns and "_signal_qc_uid" in trial_qc_df.columns:
        low_count_uids = set(kept_df.loc[~keep_mask, "_signal_qc_uid"].astype(int).tolist())
        if low_count_uids:
            low_mask = trial_qc_df["_signal_qc_uid"].isin(low_count_uids)
            trial_qc_df.loc[low_mask, "qc_status"] = "dropped"
            trial_qc_df.loc[low_mask, "qc_reason"] = "too_few_trials_after_qc"
            dropped_trials += int(np.sum(~keep_mask))
    X = X[keep_mask]
    y = y[keep_mask]
    kept_df = kept_df.loc[keep_mask].reset_index(drop=True)
    if feature_names is None:
        return None, "No signal-space feature grid could be constructed."
    if feature_meta is None:
        feature_meta = []
    if "raw" in stat_names:
        n_time_points = int(sum(expected_raw_sample_counts or []))
    else:
        n_time_points = int(len(bin_edges))

    payload = {
        "X": X,
        "y": y,
        "trial_df": kept_df,
        "trial_qc_df": trial_qc_df,
        "feature_names": feature_names,
        "feature_meta_df": pd.DataFrame(feature_meta),
        "channel_names": channel_names,
        "bin_edges": bin_edges,
        "stat_names": stat_names,
        "n_time_points": int(n_time_points),
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
        "trial_outlier_z": float(trial_outlier_z),
        "exclude_trial_outliers": bool(exclude_qc_outliers),
        "dropped_trials": int(dropped_trials),
        "epoch_cache_hits": int(epoch_cache_hits),
        "epoch_cache_misses": int(epoch_cache_misses),
        "skipped_paths": skipped_paths,
        "n_runs": int(len(pd.Index(kept_df["source_fif_path"]).drop_duplicates())),
    }
    return payload, None


def per_concept_train_test_split(y, test_fraction=0.2, random_state=42, stratify_groups=None):
    """Split each concept independently so every retained concept has held-out trials."""
    rng = np.random.default_rng(int(random_state))
    labels = pd.Index(y).astype(str)
    train_idx = []
    test_idx = []
    split_rows = []
    strat_arr = None
    if stratify_groups is not None:
        try:
            strat_arr = np.asarray(stratify_groups)
            if strat_arr.shape[0] != len(labels):
                strat_arr = None
        except Exception:
            strat_arr = None

    for label in labels.drop_duplicates().tolist():
        idx = np.where(labels.to_numpy() == str(label))[0]
        if idx.size < 2:
            continue
        if strat_arr is not None:
            concept_strat = strat_arr[idx]
            subgroups = [sg for sg in pd.Index(concept_strat).drop_duplicates().tolist() if str(sg).strip() != ""]
            concept_train = []
            concept_test = []
            if len(subgroups) > 1:
                for sg in subgroups:
                    sg_idx = idx[concept_strat == sg].copy()
                    if sg_idx.size == 0:
                        continue
                    rng.shuffle(sg_idx)
                    n_sg_test = int(round(float(test_fraction) * sg_idx.size))
                    n_sg_test = max(0, min(n_sg_test, sg_idx.size))
                    if sg_idx.size > 1 and n_sg_test == 0 and float(test_fraction) > 0:
                        n_sg_test = 1
                    concept_test.extend(sg_idx[:n_sg_test].tolist())
                    concept_train.extend(sg_idx[n_sg_test:].tolist())
            if not concept_train or not concept_test:
                shuffled = np.asarray(idx, dtype=int).copy()
                rng.shuffle(shuffled)
                n_test = int(np.ceil(float(test_fraction) * shuffled.size))
                n_test = max(1, min(n_test, shuffled.size - 1))
                concept_test = shuffled[:n_test].tolist()
                concept_train = shuffled[n_test:].tolist()
        else:
            shuffled = np.asarray(idx, dtype=int).copy()
            rng.shuffle(shuffled)
            n_test = int(np.ceil(float(test_fraction) * shuffled.size))
            n_test = max(1, min(n_test, shuffled.size - 1))
            concept_test = shuffled[:n_test].tolist()
            concept_train = shuffled[n_test:].tolist()
        if len(concept_train) == 0 or len(concept_test) == 0:
            continue
        train_idx.extend(concept_train)
        test_idx.extend(concept_test)
        split_rows.append(
            {
                "concept": str(label),
                "n_trials": int(len(idx)),
                "n_train": int(len(concept_train)),
                "n_test": int(len(concept_test)),
                "test_fraction_actual": float(len(concept_test) / len(idx)),
            }
        )
    train_idx = np.asarray(sorted(train_idx), dtype=int)
    test_idx = np.asarray(sorted(test_idx), dtype=int)
    return train_idx, test_idx, pd.DataFrame(split_rows)



def zero_shot_concept_train_test_split(
    y,
    *,
    unseen_concepts=None,
    unseen_fraction=0.2,
    random_state=42,
    min_train_concepts=2,
    min_test_concepts=1,
    concept_categories=None,
    balance_categories=False,
):
    """Split trials by concept identity so unseen test concepts are completely held out from training.

    Parameters
    ----------
    y : array-like of shape (n_trials,)
        Target concept labels for each trial.
    unseen_concepts : list or set or str, optional
        Explicit concept names to designate as unseen test concepts. If a comma-separated
        string is passed, it will be split.
    unseen_fraction : float, default=0.2
        Fraction of unique concepts to hold out as unseen test concepts when
        unseen_concepts is not explicitly provided.
    random_state : int, default=42
        Random seed for reproducible concept assignment.
    min_train_concepts : int, default=2
        Minimum number of training concepts required.
    min_test_concepts : int, default=1
        Minimum number of unseen test concepts required.
    concept_categories : array-like of shape (n_trials,), optional
        Stimulus-category label for each trial. Every concept must map to exactly
        one nonempty category when balance_categories is enabled.
    balance_categories : bool, default=False
        Select an equal number of randomly held-out concepts from every category.
        The requested number of unseen concepts must be divisible by the number
        of represented categories.

    Returns
    -------
    train_idx : np.ndarray of shape (n_train_trials,)
        Trial indices belonging to training concepts.
    test_idx : np.ndarray of shape (n_test_trials,)
        Trial indices belonging to unseen test concepts.
    split_df : pd.DataFrame
        Concept-level split metadata table.
    """
    labels = pd.Index(y).astype(str)
    unique_concepts = [str(c) for c in labels.drop_duplicates().tolist() if str(c).strip()]
    req_total = int(min_train_concepts + min_test_concepts)
    if len(unique_concepts) < req_total:
        raise ValueError(
            f"Zero-shot decoding requires at least {req_total} concepts, "
            f"but only found {len(unique_concepts)}: {unique_concepts}"
        )

    category_by_concept = {}
    if concept_categories is not None:
        categories = np.asarray(concept_categories, dtype=object).reshape(-1)
        if categories.size != labels.size:
            raise ValueError(
                "Zero-shot concept categories must contain one value per trial "
                f"({categories.size} categories for {labels.size} trials)."
            )
        labels_arr = labels.to_numpy()
        for concept in unique_concepts:
            values = {
                str(value).strip()
                for value in categories[labels_arr == concept]
                if value is not None
                and str(value).strip()
                and str(value).strip().lower() not in {"nan", "none", "unknown"}
            }
            if len(values) == 1:
                category_by_concept[concept] = next(iter(values))
            elif balance_categories:
                detail = "no category" if not values else f"multiple categories {sorted(values)}"
                raise ValueError(f"Concept '{concept}' has {detail}; category-balanced zero-shot splitting requires exactly one.")
    if balance_categories and len(category_by_concept) != len(unique_concepts):
        missing = sorted(set(unique_concepts) - set(category_by_concept))
        raise ValueError(
            "Category-balanced zero-shot splitting requires a stimulus category for every concept. "
            f"Missing: {missing}"
        )

    if unseen_concepts is not None:
        if isinstance(unseen_concepts, str):
            unseen_candidates = [c.strip() for c in unseen_concepts.split(",") if c.strip()]
        else:
            unseen_candidates = [str(c).strip() for c in unseen_concepts if str(c).strip()]
        test_concepts = set()
        for cand in unseen_candidates:
            if cand in unique_concepts:
                test_concepts.add(cand)
            else:
                for u in unique_concepts:
                    if u.lower() == cand.lower():
                        test_concepts.add(u)
        if not test_concepts:
            raise ValueError(
                f"None of the specified unseen concepts {unseen_candidates} were found in available concepts: {unique_concepts}"
            )
        if balance_categories:
            all_categories = sorted(set(category_by_concept.values()), key=str.lower)
            test_counts = {
                category: sum(category_by_concept[concept] == category for concept in test_concepts)
                for category in all_categories
            }
            if not test_counts or len(set(test_counts.values())) != 1 or min(test_counts.values()) < 1:
                raise ValueError(
                    "Explicit unseen concepts are not equally represented across all stimulus categories: "
                    f"{test_counts}"
                )
    else:
        rng = np.random.default_rng(int(random_state))
        n_unseen = int(round(float(unseen_fraction) * len(unique_concepts)))
        max_unseen = max(1, len(unique_concepts) - int(min_train_concepts))
        n_unseen = max(int(min_test_concepts), min(n_unseen, max_unseen))
        if balance_categories:
            all_categories = sorted(set(category_by_concept.values()), key=str.lower)
            if n_unseen % len(all_categories):
                raise ValueError(
                    f"Holding out {n_unseen} of {len(unique_concepts)} concepts cannot be divided equally "
                    f"across {len(all_categories)} stimulus categories. Choose a test percentage that "
                    "holds out a whole number of concepts per category."
                )
            per_category = n_unseen // len(all_categories)
            if per_category < 1:
                raise ValueError(
                    f"Category-balanced zero-shot splitting needs at least {len(all_categories)} unseen concepts "
                    f"(one from each category), but the requested split selects {n_unseen}."
                )
            test_concepts = set()
            for category in all_categories:
                candidates = sorted(
                    concept for concept in unique_concepts
                    if category_by_concept[concept] == category
                )
                if len(candidates) <= per_category:
                    raise ValueError(
                        f"Stimulus category '{category}' has {len(candidates)} concepts; cannot hold out "
                        f"{per_category} while retaining at least one for training."
                    )
                rng.shuffle(candidates)
                test_concepts.update(candidates[:per_category])
        else:
            shuffled = sorted(unique_concepts)
            rng.shuffle(shuffled)
            test_concepts = set(shuffled[:n_unseen])

    train_concepts = set(c for c in unique_concepts if c not in test_concepts)
    if len(train_concepts) < min_train_concepts:
        raise ValueError(
            f"Zero-shot split left fewer than {min_train_concepts} training concepts ({sorted(train_concepts)})."
        )
    if len(test_concepts) < min_test_concepts:
        raise ValueError(
            f"Zero-shot split left fewer than {min_test_concepts} unseen test concepts ({sorted(test_concepts)})."
        )

    labels_arr = labels.to_numpy()
    train_idx = np.where(np.isin(labels_arr, list(train_concepts)))[0]
    test_idx = np.where(np.isin(labels_arr, list(test_concepts)))[0]

    split_rows = []
    for concept in unique_concepts:
        n_c = int(np.sum(labels_arr == concept))
        is_unseen = concept in test_concepts
        split_rows.append(
            {
                "concept": str(concept),
                "stimulus_category": category_by_concept.get(concept, ""),
                "split": "unseen_test" if is_unseen else "train",
                "is_unseen": bool(is_unseen),
                "n_trials": n_c,
                "n_train": 0 if is_unseen else n_c,
                "n_test": n_c if is_unseen else 0,
                "test_fraction_actual": 1.0 if is_unseen else 0.0,
            }
        )
    train_idx = np.asarray(sorted(train_idx), dtype=int)
    test_idx = np.asarray(sorted(test_idx), dtype=int)
    return train_idx, test_idx, pd.DataFrame(split_rows)


def concept_group_column_for_split(trial_df, split_mode):
    mode = str(split_mode or "random").strip().lower()
    if mode in {"leave_one_session", "session", "session_out"}:
        for col in ("session_id", "session", "block_id"):
            if col in trial_df.columns and trial_df[col].astype(str).str.strip().nunique() > 1:
                return col, "leave-one-session-out"
        return None, "leave-one-session-out"
    if mode in {"leave_one_run", "run", "run_out", "file"}:
        for col in ("source_fif_path", "run_id", "source_fif", "block_id"):
            if col in trial_df.columns and trial_df[col].astype(str).str.strip().nunique() > 1:
                return col, "leave-one-run-out"
        return None, "leave-one-run-out"
    return None, "per-concept random"


def validate_grouped_split(context, options=None):
    if options is None:
        options = {}
    requested_split_mode = str(options.get("split_mode", "random") or "random").strip().lower()
    if requested_split_mode not in GROUPED_SPLIT_MODES:
        return None
    trial_df = getattr(context, "_fif_trial_response_table", pd.DataFrame())
    if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
        return "Grouped concept split requires a mapped trial table."
    group_col, split_label = concept_group_column_for_split(trial_df, requested_split_mode)
    if group_col is None:
        return f"{split_label} requires at least two runs/sessions in the mapped trial table."
    return None


def concept_qc_summary_table(trial_qc_df, feature_payload=None):
    """Summarize QC outcomes; percentages use all QC-recorded trials in each scope."""
    rows = []
    feature_payload = feature_payload if isinstance(feature_payload, dict) else {}
    reject_enabled = bool(feature_payload.get("mne_epoch_reject_enabled", False))
    rows.append(
        {
            "scope": "settings",
            "run": "all",
            "qc_status": "mne_epoch_reject",
            "qc_reason": "enabled" if reject_enabled else "disabled",
            "n_trials": np.nan,
            "qc_outlier_trials": np.nan,
            "mne_reject_mag_ft": feature_payload.get("mne_reject_mag_ft", np.nan),
            "mne_reject_grad_ft_cm": feature_payload.get("mne_reject_grad_ft_cm", np.nan),
        }
    )

    if not isinstance(trial_qc_df, pd.DataFrame) or trial_qc_df.empty:
        rows.append(
            {
                "scope": "overall",
                "run": "all",
                "qc_status": "unavailable",
                "qc_reason": "no_trial_qc_rows",
                "n_trials": 0,
                "qc_outlier_trials": 0,
                "mne_reject_mag_ft": feature_payload.get("mne_reject_mag_ft", np.nan),
                "mne_reject_grad_ft_cm": feature_payload.get("mne_reject_grad_ft_cm", np.nan),
            }
        )
        return pd.DataFrame(rows)

    df = trial_qc_df.copy()
    status = df.get("qc_status", pd.Series(["unknown"] * len(df), index=df.index)).astype(str).str.strip()
    reason = df.get("qc_reason", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()
    df["_qc_status_summary"] = status.mask(status.eq(""), "unknown")
    df["_qc_reason_summary"] = reason.mask(reason.eq(""), "kept")
    outliers = pd.Series(df.get("qc_outlier", False), index=df.index).fillna(False).astype(bool)
    df["_qc_outlier_summary"] = outliers
    if "source_fif" in df.columns:
        run = df["source_fif"].astype(str).str.strip()
    elif "source_fif_path" in df.columns:
        run = df["source_fif_path"].map(lambda p: os.path.basename(str(p))).astype(str).str.strip()
    else:
        run = pd.Series(["unknown"] * len(df), index=df.index)
    df["_qc_run_summary"] = run.mask(run.eq(""), "unknown")

    def totals_row(scope, run_name, group):
        total = int(len(group))
        discarded = int(group["_qc_status_summary"].eq("dropped").sum())
        return {
            "scope": scope,
            "run": run_name,
            "qc_status": "total",
            "qc_reason": "all_qc_recorded_trials",
            "n_trials": total,
            "discarded_trials": discarded,
            "discarded_pct": 100.0 * discarded / total,
            "kept_trials": int(group["_qc_status_summary"].eq("kept").sum()),
            "qc_outlier_trials": int(group["_qc_outlier_summary"].sum()),
        }

    # Put totals first so the discarded count and denominator are immediately visible.
    rows.insert(0, totals_row("overall", "all", df))
    for run_name, group in df.groupby("_qc_run_summary", sort=False):
        rows.append(totals_row("run", str(run_name), group))

    for (qc_status, qc_reason), group in df.groupby(["_qc_status_summary", "_qc_reason_summary"], sort=False):
        rows.append(
            {
                "scope": "overall",
                "run": "all",
                "qc_status": str(qc_status),
                "qc_reason": str(qc_reason),
                "n_trials": int(len(group)),
                "qc_outlier_trials": int(group["_qc_outlier_summary"].sum()),
                "mne_reject_mag_ft": feature_payload.get("mne_reject_mag_ft", np.nan),
                "mne_reject_grad_ft_cm": feature_payload.get("mne_reject_grad_ft_cm", np.nan),
            }
        )

    for (run_name, qc_status, qc_reason), group in df.groupby(
        ["_qc_run_summary", "_qc_status_summary", "_qc_reason_summary"],
        sort=False,
    ):
        rows.append(
            {
                "scope": "run",
                "run": str(run_name),
                "qc_status": str(qc_status),
                "qc_reason": str(qc_reason),
                "n_trials": int(len(group)),
                "qc_outlier_trials": int(group["_qc_outlier_summary"].sum()),
                "mne_reject_mag_ft": feature_payload.get("mne_reject_mag_ft", np.nan),
                "mne_reject_grad_ft_cm": feature_payload.get("mne_reject_grad_ft_cm", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _option_int(options, key, default, lower=None, upper=None):
    try:
        value = int((options or {}).get(key, default))
    except Exception:
        value = int(default)
    if lower is not None:
        value = max(int(lower), int(value))
    if upper is not None:
        value = min(int(upper), int(value))
    return int(value)


def _option_float(options, key, default, lower=None, upper=None):
    try:
        value = float((options or {}).get(key, default))
    except Exception:
        value = float(default)
    if lower is not None:
        value = max(float(lower), float(value))
    if upper is not None:
        value = min(float(upper), float(value))
    return float(value)


def _mlp_feature_select_k(options, n_features, n_classes):
    try:
        n_features = int(n_features)
    except Exception:
        n_features = 0
    try:
        n_classes = max(2, int(n_classes))
    except Exception:
        n_classes = 2
    if n_features <= 0:
        return 0
    requested = _option_int(options, "mlp_feature_select_k", 0, lower=0)
    if requested > 0:
        return min(int(requested), int(n_features))
    # Keep the nonlinear model intentionally compact for the usual p >> n
    # neuroimaging setting. The former 100 * classes rule retained 3,700
    # inputs for 37 classes and created a roughly two-million-parameter net.
    auto_k = max(64, min(256, 4 * n_classes))
    return min(int(auto_k), int(n_features))


def _mlp_validation_fraction(options):
    value = (options or {}).get("mlp_validation_fraction", (options or {}).get("validation_fraction", 0.2))
    return _option_float({"value": value}, "value", 0.2, lower=0.0, upper=0.5)


def _mlp_train_validation_indices(y_train, validation_fraction, random_state=42):
    y_train = np.asarray(y_train, dtype=str)
    all_idx = np.arange(y_train.size, dtype=int)
    if y_train.size < 4 or float(validation_fraction) <= 0:
        return all_idx, np.asarray([], dtype=int), "disabled"

    counts = pd.Series(y_train).value_counts()
    if counts.empty or int(counts.min()) < 2:
        return all_idx, np.asarray([], dtype=int), "disabled: at least one class has fewer than two training samples"

    n_classes = int(len(counts))
    n_val = int(round(float(y_train.size) * float(validation_fraction)))
    n_val = max(n_classes, n_val)
    n_val = min(y_train.size - n_classes, n_val)
    if n_val < n_classes or (y_train.size - n_val) < n_classes:
        return all_idx, np.asarray([], dtype=int), "disabled: not enough samples for stratified validation"

    try:
        fit_idx, val_idx = train_test_split(
            all_idx,
            test_size=int(n_val),
            random_state=int(random_state),
            stratify=y_train,
        )
    except Exception as exc:
        return all_idx, np.asarray([], dtype=int), f"disabled: {exc}"
    return np.asarray(fit_idx, dtype=int), np.asarray(val_idx, dtype=int), ""


def concept_decoder_spec(decoder_type="logistic", random_state=42, options=None, n_features=None, n_classes=None):
    """Return an sklearn pipeline plus display metadata for concept decoding."""
    decoder_key = str(decoder_type or "logistic").strip().lower()
    if decoder_key == "svm":
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual=False,
                      max_iter=10000, random_state=int(random_state)),
        )
        return clf, "signal-space linear SVM", "svm"
    if decoder_key in {"template", "prototype", "prototype_correlation", "template_correlation", "matched_template"}:
        clf = make_pipeline(
            StandardScaler(),
            PrototypeCorrelationClassifier(similarity_scale=8.0),
        )
        return clf, "signal-space template correlation", "template"

    if decoder_key in {"mlp", "neural_net", "neural-network", "feedforward", "feed-forward"}:
        try:
            n_features_int = int(n_features) if n_features is not None else 0
        except Exception:
            n_features_int = 0
        try:
            n_classes_int = int(n_classes) if n_classes is not None else 0
        except Exception:
            n_classes_int = 0
        max_iter = _option_int(options, "mlp_epochs", (options or {}).get("epochs", 100), lower=1, upper=5000)
        feature_select_k = _mlp_feature_select_k(options, n_features_int, n_classes_int)
        hidden_1 = _option_int(options, "mlp_hidden_1", 64, lower=16, upper=4096)
        hidden_2 = _option_int(options, "mlp_hidden_2", 32, lower=0, upper=4096)
        hidden_layers = (hidden_1, hidden_2) if hidden_2 > 0 else (hidden_1,)
        batch_size = _option_int(options, "mlp_batch_size", 32, lower=4, upper=4096)
        dropout = _option_float(options, "mlp_dropout", 0.35, lower=0.0, upper=0.9)
        weight_decay = _option_float(options, "mlp_weight_decay", 1e-3, lower=0.0, upper=1.0)
        steps = [
            StandardScaler(),
        ]
        if n_features_int > 0 and feature_select_k > 0 and feature_select_k < n_features_int:
            steps.append(SelectKBest(score_func=f_classif, k=int(feature_select_k)))
        steps.append(
            DropoutMLPClassifier(
                hidden_layer_sizes=hidden_layers,
                dropout=dropout,
                alpha=weight_decay,
                batch_size=batch_size,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                random_state=int(random_state),
            ),
        )
        # Preserve the historical final-step name used by fitting/report code.
        pipeline_steps = [("standardscaler", steps[0])]
        if len(steps) == 3:
            pipeline_steps.append(("selectkbest", steps[1]))
        pipeline_steps.append(("mlpclassifier", steps[-1]))
        clf = Pipeline(pipeline_steps)
        return clf, "signal-space feed-forward neural network", "mlp"

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", random_state=int(random_state)),
    )
    return clf, "signal-space logistic regression", "logistic"


def _architecture_row(stage, component, purpose, configuration="", shape=""):
    return {
        "stage": str(stage),
        "component": str(component),
        "purpose": str(purpose),
        "configuration": str(configuration),
        "shape": str(shape),
    }


def concept_model_architecture_table(summary, feature_payload=None, class_names=None):
    """Describe the trained concept-decoding model in a compact table."""
    summary = dict(summary or {})
    feature_payload = dict(feature_payload or {})
    class_names = [str(v) for v in (class_names or [])]
    decoder_key = str(summary.get("decoder_type", "") or "").strip().lower()
    feature_space = str(summary.get("feature_space", feature_payload.get("feature_space", "signal")) or "signal")
    n_classes = int(summary.get("n_concepts", len(class_names) or 0) or 0)
    n_features = int(summary.get("n_features", 0) or 0)
    n_channels = int(summary.get("n_channels", 0) or 0)
    n_rois = int(summary.get("n_rois", 0) or 0)
    n_bins = int(summary.get("n_time_bins", len(feature_payload.get("bin_edges", []))) or 0)
    window = f"{summary.get('window_start_s', feature_payload.get('window_start', np.nan))}-{summary.get('window_end_s', feature_payload.get('window_end', np.nan))} s"
    bandpass = f"{summary.get('bandpass_low_hz', feature_payload.get('bandpass_low_hz', np.nan))}-{summary.get('bandpass_high_hz', feature_payload.get('bandpass_high_hz', np.nan))} Hz"
    stat_names = ", ".join(str(v) for v in feature_payload.get("stat_names", []) if str(v).strip())
    feature_binning = bool(summary.get("feature_binning_enabled", feature_payload.get("feature_binning_enabled", True)))
    resample_enabled = bool(summary.get("resample_enabled", feature_payload.get("resample_enabled", False)))
    resample_sfreq = summary.get("resample_sfreq", feature_payload.get("resample_sfreq", np.nan))
    source_common_space = str(summary.get("source_common_space", feature_payload.get("source_common_space", "")) or "")
    source_space_kind = str(summary.get("source_space_kind", feature_payload.get("source_space_kind", "")) or "")
    source_subject_count = int(summary.get("source_subject_count", feature_payload.get("source_subject_count", 0)) or 0)
    source_parcel_baseline_zscore = bool(
        summary.get(
            "source_parcel_baseline_zscore",
            feature_payload.get("source_parcel_baseline_zscore", False),
        )
    )
    source_config = ""
    if feature_space == "source":
        source_parcellation = str(summary.get("source_parcellation", feature_payload.get("source_parcellation", "")) or "")
        source_config = (
            f"; atlas={source_parcellation or 'aparc'}; "
            f"source_kind={source_space_kind or 'surface'}; "
            f"source_space={source_common_space or 'native'}; source_subjects={source_subject_count or 1}"
        )
    rows = [
        _architecture_row(
            "Input",
            "Trial-locked MEG epochs",
            "Build one sample per valid stimulus trial before train/test splitting.",
            f"feature_space={feature_space}; group_mode={summary.get('group_mode', feature_payload.get('group_mode', ''))}; window={window}; bandpass={bandpass}; baseline_correct={summary.get('baseline_correct', feature_payload.get('baseline_correct', ''))}; parcel_baseline_zscore={source_parcel_baseline_zscore}",
            f"trials={summary.get('n_source_trials', summary.get('n_trials', ''))}; channels={n_channels}; rois={n_rois}; time_bins={n_bins}",
        ),
    ]
    rows.append(
        _architecture_row(
            "Preprocessing",
            "MNE resampling" if resample_enabled else "Native sampling rate",
            (
                "Resample with MNE before feature extraction; continuous recording preprocessing is the default."
                if resample_enabled
                else "Keep epoch samples at the native FIF sampling rate."
            ),
            f"enabled={resample_enabled}; target_sfreq={resample_sfreq}",
            "",
        )
    )
    if feature_space == "source" and feature_payload.get("feature_mode") == "bandpower":
        rows.append(_architecture_row(
            "Features", "Source band power",
            "Filter signed/vector currents per trial; sum component powers, then average linear power within ROIs before log normalization.",
            f"bands={feature_payload.get('band_defs', [])}; baseline_dB={feature_payload.get('source_power_baseline', False)}; padding_s={feature_payload.get('source_power_padding_s', '')}",
            f"rois={n_rois}; time_bins={n_bins}",
        ))
    if feature_space == "source" and source_parcel_baseline_zscore:
        baseline_start = summary.get(
            "parcel_baseline_start_s",
            feature_payload.get("parcel_baseline_start", np.nan),
        )
        baseline_end = summary.get(
            "parcel_baseline_end_s",
            feature_payload.get("parcel_baseline_end", np.nan),
        )
        baseline_samples = int(
            summary.get(
                "parcel_baseline_samples",
                feature_payload.get("parcel_baseline_samples", 0),
            )
            or 0
        )
        rows.append(
            _architecture_row(
                "Preprocessing",
                "Per-trial parcel baseline z-score",
                "Normalize every parcel using only its own prestimulus samples before temporal feature extraction.",
                f"baseline={baseline_start}-{baseline_end} s; samples={baseline_samples}",
                f"rois={n_rois}",
            )
        )
    if not feature_binning and feature_payload.get("feature_mode") != "bandpower":
        rows.append(
            _architecture_row(
                "Features",
                "No feature binning",
                "Use resampled epoch samples directly instead of time-bin or whole-window summary features.",
                "feature_binning_enabled=False",
                "",
            )
        )
    cache_hits = int(summary.get("epoch_cache_hits", feature_payload.get("epoch_cache_hits", 0)) or 0)
    cache_misses = int(summary.get("epoch_cache_misses", feature_payload.get("epoch_cache_misses", 0)) or 0)
    if cache_hits or cache_misses:
        rows.append(
            _architecture_row(
                "Preprocessing",
                "Bandpass epoch cache",
                "Reuse filtered, baseline-corrected epoch arrays across decoder runs when preprocessing settings match.",
                f"hits={cache_hits}; misses={cache_misses}; key=FIF identity + channels + bandpass + baseline + epoch window + resample + event samples",
                "",
            )
        )

    embedding_decoder_keys = {
        "transformer", "embedding_ridge",
        "embedding_contrastive", "embedding_mlp",
    }
    if decoder_key in embedding_decoder_keys:
        if feature_space == "source":
            rows.append(
                _architecture_row(
                    "Features",
                    "Source-parcel feature matrix",
                    "Use the selected source atlas, parcel feature representation, and time bins before within-split trial averaging.",
                    f"stats={stat_names or 'source features'}; time_points={n_bins}; feature_binning={feature_binning}; averaged_after_split=True{source_config}",
                    f"[pseudo_epochs, features={n_features}]; rois={n_rois}",
                )
            )
            embedding_input_name = "source-parcel"
        else:
            rows.append(
                _architecture_row(
                    "Features",
                    "Baseline-relative band-power tensor",
                    "Convert padded signal epochs into Hilbert-envelope power bins; retain only complete bands inside the shared passband.",
                    f"bands={int(summary.get('n_bands', 0) or 0)}; time_points={n_bins}; feature_binning={feature_binning}; averaged_after_split=True",
                    f"[pseudo_epochs, channels={n_channels}, bands={int(summary.get('n_bands', 0) or 0)}, time_points={n_bins}]",
                )
            )
            embedding_input_name = "MEG band-power"
        model_input_dim = int(
            summary.get("embedding_input_pca_components", n_features) or n_features
        )
        original_embedding_dim = int(
            summary.get("embedding_original_dim", summary.get("embedding_dim", 0)) or 0
        )
        model_embedding_dim = int(summary.get("embedding_dim", original_embedding_dim) or 0)
        if bool(summary.get("embedding_input_pca_applied", False)):
            rows.append(
                _architecture_row(
                    "Dimensionality reduction",
                    "Train-only neural PCA",
                    "Fit PCA using training pseudo-epochs only, then reuse that transform for validation and held-out test epochs.",
                    f"requested_components={summary.get('embedding_input_pca_requested_components', '')}; explained_variance={summary.get('embedding_input_pca_explained_variance', '')}; fitted_on=train_only",
                    f"{n_features}->{model_input_dim}",
                )
            )
        if model_embedding_dim < original_embedding_dim:
            rows.append(
                _architecture_row(
                    "Dimensionality reduction",
                    "Train-only semantic SVD",
                    "Project centered candidate embeddings into the compact semantic span learned from training concepts only.",
                    f"requested_components={summary.get('embedding_target_svd_requested_components', '')}; rank={summary.get('embedding_target_rank', '')}; explained_variance={summary.get('embedding_target_svd_explained_variance', '')}",
                    f"{original_embedding_dim}->{model_embedding_dim}",
                )
            )
        if decoder_key == "embedding_ridge":
            rows.append(
                _architecture_row(
                    "Model",
                    "Linear ridge embedding regression",
                    f"Flatten standardized {embedding_input_name} features and predict semantic embedding vectors with one linear map.",
                    f"alpha={summary.get('embedding_ridge_alpha', '')}; solver=lsqr",
                    f"input={model_input_dim}; embedding_dim={model_embedding_dim}; parameters={summary.get('embedding_model_parameter_count', '')}",
                )
            )
            loss_configuration = (
                f"objective=squared error with L2 coefficient regularization; "
                f"final_cosine_loss={summary.get('transformer_final_train_loss', '')}"
            )
        elif decoder_key == "embedding_contrastive":
            rows.append(
                _architecture_row(
                    "Model",
                    "Linear class-aware contrastive embedding",
                    f"Map flattened standardized {embedding_input_name} features to semantic vectors with one linear layer.",
                    f"temperature={summary.get('embedding_contrastive_temperature', '')}; contrastive_weight={summary.get('embedding_contrastive_loss_weight', '')}; weight_decay={summary.get('embedding_contrastive_weight_decay', '')}; learning_rate={summary.get('embedding_contrastive_learning_rate', '')}; epochs={summary.get('transformer_epochs', '')}/{summary.get('transformer_epochs_requested', '')}; early_stop_patience={summary.get('embedding_early_stopping_patience', '')}; best_validation_epoch={summary.get('embedding_contrastive_best_validation_epoch', '')}; dynamic_pseudo_epochs={summary.get('embedding_dynamic_pseudo_epochs', '')}",
                    f"input={model_input_dim}; embedding_dim={model_embedding_dim}; parameters={summary.get('embedding_model_parameter_count', '')}",
                )
            )
            loss_configuration = (
                "loss=class-aware cross-entropy over cosine similarities to all unique concept candidates; "
                f"temperature={summary.get('embedding_contrastive_temperature', '')}; optimizer=AdamW; "
                f"final_train_loss={summary.get('transformer_final_train_loss', '')}"
            )
        elif decoder_key == "embedding_mlp":
            rows.append(
                _architecture_row(
                    "Model",
                    "MLP embedding regression",
                    f"Flatten standardized {embedding_input_name} features and predict semantic embedding vectors with nonlinear hidden layers.",
                    f"hidden={summary.get('embedding_mlp_hidden_layers', '')}; dropout={summary.get('embedding_mlp_dropout', '')}; epochs={summary.get('transformer_epochs', '')}; best_validation_epoch={summary.get('embedding_mlp_best_validation_epoch', '')}",
                    f"input={model_input_dim}; embedding_dim={model_embedding_dim}; parameters={summary.get('embedding_model_parameter_count', '')}",
                )
            )
            loss_configuration = (
                f"loss=centered cosine + prototype contrastive cross-entropy; "
                f"contrastive_weight={summary.get('embedding_mlp_contrastive_weight', '')}; "
                f"temperature={summary.get('embedding_mlp_temperature', '')}; optimizer=AdamW; "
                f"final_train_loss={summary.get('transformer_final_train_loss', '')}"
            )
        else:
            rows.append(
                _architecture_row(
                    "Model",
                    "Transformer",
                    "Predict semantic embedding vectors from structured MEG band-power pseudo-epochs.",
                    f"model_type={summary.get('transformer_model_type', '')}; pooled_time_bins={summary.get('transformer_pool_time_bins', '')}; device={summary.get('transformer_device', '')}; epochs={summary.get('transformer_epochs', '')}; batch_size={summary.get('transformer_batch_size', '')}",
                    f"embedding_dim={summary.get('embedding_dim', '')}",
                )
            )
            loss_configuration = (
                f"loss=1 - cosine_similarity; optimizer=AdamW; "
                f"final_train_loss={summary.get('transformer_final_train_loss', '')}"
            )
        rows.extend(
            [
                _architecture_row(
                    "Loss/optimizer",
                    "Semantic embedding alignment",
                    "Align predictions with centered concept/Word2Vec vectors while avoiding the shared semantic centroid as a constant solution.",
                    loss_configuration,
                    "",
                ),
                _architecture_row(
                    "Classifier",
                    "Cosine semantic rank scoring",
                    "Score held-out predicted embeddings against candidate concept vectors and report MPR plus nearest-concept top-1 accuracy.",
                    f"classes={n_classes}; trials_per_average={summary.get('trials_per_average', '')}; chance_MPR={summary.get('chance_accuracy', '')}; chance_top1={summary.get('top1_chance_accuracy', '')}",
                    "",
                ),
            ]
        )
    else:
        rows.append(
            _architecture_row(
                "Features",
                "Flattened signal/source feature matrix",
                "Compute per-trial feature vectors from cached epochs.",
                f"stats={stat_names or 'source/raw features'}; bins={n_bins}; feature_binning={feature_binning}; channel_mode={summary.get('channel_mode', feature_payload.get('channel_mode', ''))}{source_config}",
                f"[trials={summary.get('n_trials', '')}, features={n_features}]",
            )
        )
        rows.append(
            _architecture_row(
                "Preprocessing",
                "StandardScaler",
                "MNE epoch rejection/QC before feature extraction, then z-score each feature using training data only.",
                (
                    "epoch_reject="
                    f"{'on' if summary.get('mne_epoch_reject_enabled', False) else 'off'}; "
                    f"mag={summary.get('mne_reject_mag_ft', '')} fT; "
                    f"grad={summary.get('mne_reject_grad_ft_cm', '')} fT/cm; "
                    "mean/std learned on train fold"
                ),
                f"features={n_features}",
            )
        )
        if decoder_key == "template":
            rows.append(
                _architecture_row(
                    "Model",
                    "PrototypeCorrelationClassifier",
                    "Build one centered, L2-normalized prototype template per concept and predict the highest correlation.",
                    "template = weighted mean of normalized train trials; similarity_scale=8.0",
                    f"templates=[classes={n_classes}, features={n_features}]",
                )
            )
        elif decoder_key == "svm":
            rows.append(
                _architecture_row(
                    "Model", "LinearSVC",
                    "One-versus-rest linear support-vector classifier over training-standardized features.",
                    "C=1.0; class_weight=balanced; loss=squared_hinge; dual=False; max_iter=10000; random_state=42; probabilities unavailable",
                    f"input={n_features}; output={n_classes}",
                )
            )
        elif decoder_key == "mlp":
            mlp_hidden = str(summary.get("mlp_hidden_layers", "64->32") or "64->32")
            mlp_selected = summary.get("mlp_feature_select_k", "")
            rows.append(
                _architecture_row(
                    "Model",
                    "Regularized feed-forward neural network",
                    "Nonlinear classifier over a supervised subset of standardized trial features.",
                    f"SelectKBest(k={mlp_selected}) -> DropoutMLP(hidden_layer_sizes={mlp_hidden}, activation=relu, dropout={summary.get('mlp_dropout', '')}, optimizer=AdamW, weight_decay={summary.get('mlp_weight_decay', '')}, learning_rate_init=1e-3, batch_size={summary.get('mlp_batch_size', '')}, epochs={summary.get('mlp_epochs', '')}, validation_fraction={summary.get('mlp_validation_fraction', '')}, restore_best_validation_epoch=True)",
                    f"input={n_features}; selected={mlp_selected}; hidden={mlp_hidden}; output={n_classes}",
                )
            )
        else:
            rows.append(
                _architecture_row(
                    "Model",
                    "LogisticRegression",
                    "Linear class-balanced multiclass decoder over standardized trial features.",
                    "class_weight=balanced; max_iter=4000; random_state=42",
                    f"input={n_features}; output={n_classes}",
                )
            )
        rows.append(
            _architecture_row(
                "Classifier output",
                "Held-out concept probabilities/scores",
                "Predict the concept label for trials not used during training.",
                f"split={summary.get('split', '')}; classes={n_classes}; chance={summary.get('chance_accuracy', '')}",
                "",
            )
        )
    return pd.DataFrame(rows)


def _fit_mlp_epochwise(clf, X_train, y_train, options=None, sample_weight=None, eval_X=None, eval_y=None):
    """Fit a Pipeline ending in the dropout MLP one epoch at a time."""
    if not hasattr(clf, "steps") or not getattr(clf, "steps", None):
        raise RuntimeError("Feed-forward decoder must be an sklearn Pipeline.")
    named_steps = getattr(clf, "named_steps", {})
    mlp = named_steps.get("mlpclassifier")
    if mlp is None:
        raise RuntimeError("Feed-forward decoder pipeline is missing its MLP classifier.")

    X_all = np.asarray(X_train, dtype=float)
    y_all = np.asarray(y_train, dtype=str)
    if X_all.ndim != 2 or X_all.shape[0] != y_all.shape[0]:
        raise ValueError("Feed-forward decoder received inconsistent training data.")

    random_state = getattr(mlp, "random_state", 42)
    try:
        random_state = int(random_state)
    except Exception:
        random_state = 42
    validation_fraction = _mlp_validation_fraction(options)
    fit_idx, val_idx, validation_note = _mlp_train_validation_indices(
        y_all,
        validation_fraction,
        random_state=random_state,
    )
    X_fit = X_all[fit_idx]
    y_fit = y_all[fit_idx]
    X_val = X_all[val_idx] if val_idx.size > 0 else None
    y_val = y_all[val_idx] if val_idx.size > 0 else None
    X_test = None
    y_test = None
    if eval_X is not None and eval_y is not None:
        try:
            X_test = np.asarray(eval_X, dtype=float)
            y_test = np.asarray(eval_y, dtype=str)
            if X_test.ndim != 2 or X_test.shape[0] != y_test.shape[0] or X_test.shape[0] == 0:
                X_test = None
                y_test = None
        except Exception:
            X_test = None
            y_test = None
    for _name, step in clf.steps[:-1]:
        if hasattr(step, "fit_transform"):
            X_fit = step.fit_transform(X_fit, y_fit)
        else:
            step.fit(X_fit, y_fit)
            X_fit = step.transform(X_fit)
        X_fit = np.asarray(X_fit, dtype=float)
        if X_val is not None:
            try:
                X_val = np.asarray(step.transform(X_val), dtype=float)
            except Exception:
                X_val = None
                y_val = None
        if X_test is not None:
            try:
                X_test = np.asarray(step.transform(X_test), dtype=float)
            except Exception:
                X_test = None
                y_test = None

    try:
        epochs = max(1, int(getattr(mlp, "max_iter", 100) or 100))
    except Exception:
        epochs = 100
    classes = np.asarray(pd.Index(y_all).drop_duplicates().astype(str).tolist(), dtype=str)
    emit_progress = bool((options or {}).get("emit_mlp_epoch_progress", True))
    emit_training_metrics = bool(
        (options or {}).get("emit_mlp_training_metrics", True)
        and callable((options or {}).get("training_metric_callback"))
    )
    try:
        training_series = int((options or {}).get("_mlp_training_series_counter", 0)) + 1
    except Exception:
        training_series = 1
    if isinstance(options, dict):
        options["_mlp_training_series_counter"] = int(training_series)
    training_series_label = str(
        (options or {}).get("_mlp_training_series_label", "") or f"Training fit {training_series}"
    )
    training_history = []
    sample_weight_arr = None
    if sample_weight is not None:
        try:
            sample_weight_arr = np.asarray(sample_weight, dtype=float)[fit_idx]
            if sample_weight_arr.shape[0] != y_fit.shape[0]:
                sample_weight_arr = None
        except Exception:
            sample_weight_arr = None

    mlp.validation_fraction_used_ = float(validation_fraction) if val_idx.size > 0 else 0.0
    mlp.training_sample_count_ = int(y_fit.size)
    mlp.validation_sample_count_ = int(val_idx.size)
    mlp.validation_note_ = str(validation_note or "")
    mlp.best_validation_epoch_ = 0
    mlp.best_validation_accuracy_ = np.nan
    mlp.best_validation_balanced_accuracy_ = np.nan
    best_state = None
    best_val_bal = -np.inf
    best_val_acc = np.nan
    best_epoch = 0
    restore_best = bool((options or {}).get("mlp_restore_best_validation", True))
    emit_test_each_epoch = bool((options or {}).get("emit_mlp_test_epoch_progress", False))
    if emit_progress and val_idx.size == 0 and validation_note:
        emit_concept_progress(options, f"feed-forward neural net validation split {validation_note}.")

    for epoch_idx in range(epochs):
        kwargs = {}
        if sample_weight_arr is not None:
            kwargs["sample_weight"] = sample_weight_arr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if epoch_idx == 0:
                    mlp.partial_fit(X_fit, y_fit, classes=classes, **kwargs)
                else:
                    mlp.partial_fit(X_fit, y_fit, **kwargs)
            except TypeError:
                if epoch_idx == 0:
                    mlp.partial_fit(X_fit, y_fit, classes=classes)
                else:
                    mlp.partial_fit(X_fit, y_fit)

        if emit_progress or emit_training_metrics:
            try:
                y_train_pred = np.asarray(mlp.predict(X_fit), dtype=str)
                train_acc = float(accuracy_score(y_fit, y_train_pred))
                train_bal = float(balanced_accuracy_score(y_fit, y_train_pred))
            except Exception:
                train_acc = np.nan
                train_bal = np.nan
            val_msg = ""
            val_acc = np.nan
            val_bal = np.nan
            if X_val is not None and y_val is not None:
                try:
                    y_val_pred = np.asarray(mlp.predict(X_val), dtype=str)
                    val_acc = float(accuracy_score(y_val, y_val_pred))
                    val_bal = float(balanced_accuracy_score(y_val, y_val_pred))
                    val_msg = f", val_accuracy={val_acc:.3f}, val_balanced={val_bal:.3f}"
                    if np.isfinite(val_bal) and (val_bal > best_val_bal or best_state is None):
                        best_val_bal = float(val_bal)
                        best_val_acc = float(val_acc)
                        best_epoch = int(epoch_idx + 1)
                        state_getter = getattr(mlp, "get_training_state", None)
                        best_state = state_getter() if callable(state_getter) else {
                            "coefs": [coef.copy() for coef in getattr(mlp, "coefs_", [])],
                            "intercepts": [intercept.copy() for intercept in getattr(mlp, "intercepts_", [])],
                            "loss": float(getattr(mlp, "loss_", np.nan)),
                        }
                except Exception:
                    val_msg = ", val_accuracy=n/a"
            test_msg = ""
            test_acc = np.nan
            test_bal = np.nan
            if emit_test_each_epoch and X_test is not None and y_test is not None:
                try:
                    y_test_pred = np.asarray(mlp.predict(X_test), dtype=str)
                    test_acc = float(accuracy_score(y_test, y_test_pred))
                    test_bal = float(balanced_accuracy_score(y_test, y_test_pred))
                    test_msg = f", test_accuracy={test_acc:.3f}, test_balanced={test_bal:.3f}"
                except Exception:
                    test_msg = ", test_accuracy=n/a"
            try:
                loss = float(getattr(mlp, "loss_", np.nan))
            except Exception:
                loss = np.nan
            loss_msg = f", loss={loss:.4f}" if np.isfinite(loss) else ""
            metric = {
                "model": "mlp",
                "series": int(training_series),
                "series_label": training_series_label,
                "epoch": int(epoch_idx + 1),
                "epochs": int(epochs),
                "train_accuracy": float(train_acc),
                "train_balanced_accuracy": float(train_bal),
                "validation_accuracy": float(val_acc),
                "validation_balanced_accuracy": float(val_bal),
                "loss": float(loss),
                "n_train": int(y_fit.size),
                "n_validation": int(0 if y_val is None else y_val.size),
                "chance_accuracy": float(1.0 / max(1, len(classes))),
            }
            training_history.append(metric)
            mlp.training_history_ = list(training_history)
            mlp.training_series_ = int(training_series)
            mlp.training_series_label_ = training_series_label
            if emit_training_metrics:
                emit_concept_training_metric(options, metric)
            if emit_progress:
                emit_concept_progress(
                    options,
                    (
                        f"training feed-forward neural net epoch {epoch_idx + 1}/{epochs}: "
                        f"train_accuracy={train_acc:.3f}, train_balanced={train_bal:.3f}"
                        f"{val_msg}{test_msg}{loss_msg}"
                    ),
                )
        elif X_val is not None and y_val is not None:
            try:
                y_val_pred = np.asarray(mlp.predict(X_val), dtype=str)
                val_acc = float(accuracy_score(y_val, y_val_pred))
                val_bal = float(balanced_accuracy_score(y_val, y_val_pred))
                if np.isfinite(val_bal) and (val_bal > best_val_bal or best_state is None):
                    best_val_bal = float(val_bal)
                    best_val_acc = float(val_acc)
                    best_epoch = int(epoch_idx + 1)
                    state_getter = getattr(mlp, "get_training_state", None)
                    best_state = state_getter() if callable(state_getter) else {
                        "coefs": [coef.copy() for coef in getattr(mlp, "coefs_", [])],
                        "intercepts": [intercept.copy() for intercept in getattr(mlp, "intercepts_", [])],
                        "loss": float(getattr(mlp, "loss_", np.nan)),
                    }
            except Exception:
                pass

    if not hasattr(mlp, "training_history_"):
        mlp.training_history_ = list(training_history)
        mlp.training_series_ = int(training_series)
        mlp.training_series_label_ = training_series_label
    if best_state is not None:
        mlp.best_validation_epoch_ = int(best_epoch)
        mlp.best_validation_accuracy_ = float(best_val_acc)
        mlp.best_validation_balanced_accuracy_ = float(best_val_bal)
        if restore_best:
            try:
                state_setter = getattr(mlp, "set_training_state", None)
                if callable(state_setter) and "model" in best_state:
                    state_setter(best_state)
                else:
                    mlp.coefs_ = [coef.copy() for coef in best_state.get("coefs", [])]
                    mlp.intercepts_ = [intercept.copy() for intercept in best_state.get("intercepts", [])]
                    if np.isfinite(best_state.get("loss", np.nan)):
                        mlp.loss_ = float(best_state.get("loss"))
                if emit_progress:
                    emit_concept_progress(
                        options,
                        (
                            f"selected feed-forward neural net epoch {best_epoch}/{epochs} "
                            f"by validation balanced accuracy={best_val_bal:.3f} "
                            f"(validation accuracy={best_val_acc:.3f})."
                        ),
                    )
            except Exception:
                pass
    return clf


def fit_concept_decoder(clf, X_train, y_train, options=None, eval_X=None, eval_y=None):
    """Fit a decoder pipeline with class-balanced sample weights when supported."""
    if clf is None:
        raise RuntimeError("Decoder is unavailable.")
    y_train = np.asarray(y_train, dtype=str)
    fit_kwargs = {}
    try:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        if hasattr(clf, "steps") and clf.steps:
            final_step_name = str(clf.steps[-1][0])
            fit_kwargs[f"{final_step_name}__sample_weight"] = sample_weight
        else:
            fit_kwargs["sample_weight"] = sample_weight
    except Exception:
        fit_kwargs = {}
    # LinearSVC already applies balanced class weights; do not apply them twice.
    if "linearsvc" in getattr(clf, "named_steps", {}):
        fit_kwargs.pop("linearsvc__sample_weight", None)
    mlp = getattr(clf, "named_steps", {}).get("mlpclassifier") if hasattr(clf, "named_steps") else None
    if mlp is not None:
        return _fit_mlp_epochwise(
            clf,
            X_train,
            y_train,
            options=options,
            sample_weight=fit_kwargs.get("mlpclassifier__sample_weight"),
            eval_X=eval_X,
            eval_y=eval_y,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            clf.fit(X_train, y_train)
    return clf


def grouped_concept_predictions(X, y, trial_df, group_col, decoder_type="logistic", options=None):
    """Predict held-out groups, fitting a fresh decoder for each run/session group."""
    if group_col not in trial_df.columns:
        return None, f"Grouped concept split requires `{group_col}`."
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=str)
    groups = trial_df[group_col].astype(str).fillna("").str.strip().to_numpy()
    valid_groups = [g for g in pd.Index(groups).drop_duplicates().tolist() if str(g)]
    if len(valid_groups) < 2:
        return None, "Grouped concept split requires at least two non-empty groups."

    rows = []
    pred_records = []
    for group_idx, heldout_group in enumerate(valid_groups):
        test_idx = np.where(groups == heldout_group)[0]
        train_idx = np.where(groups != heldout_group)[0]
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        train_classes = set(y[train_idx].tolist())
        test_classes = set(y[test_idx].tolist())
        common_classes = [str(v) for v in pd.Index(y).drop_duplicates().tolist() if str(v) in train_classes and str(v) in test_classes]
        if len(common_classes) < 2:
            rows.append(
                {
                    "heldout_group": str(heldout_group),
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "n_common_concepts": int(len(common_classes)),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "skipped": True,
                    "reason": "fewer than two concepts in both train and held-out data",
                }
            )
            continue
        valid_train = np.asarray([str(y[i]) in common_classes for i in train_idx], dtype=bool)
        valid_test = np.asarray([str(y[i]) in common_classes for i in test_idx], dtype=bool)
        train_idx = train_idx[valid_train]
        test_idx = test_idx[valid_test]
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        clf, decoder_label, decoder_key = concept_decoder_spec(
            decoder_type=decoder_type,
            random_state=42 + int(group_idx),
            options=options,
            n_features=X.shape[1],
            n_classes=len(common_classes),
        )
        try:
            previous_series_label = (options or {}).get("_mlp_training_series_label")
            if isinstance(options, dict):
                options["_mlp_training_series_label"] = f"Held-out group: {heldout_group}"
            fit_concept_decoder(
                clf,
                X[train_idx],
                y[train_idx],
                options=options,
                eval_X=X[test_idx],
                eval_y=y[test_idx],
            )
            if isinstance(options, dict):
                if previous_series_label is None:
                    options.pop("_mlp_training_series_label", None)
                else:
                    options["_mlp_training_series_label"] = previous_series_label
            y_group_pred = np.asarray(clf.predict(X[test_idx]), dtype=str)
        except Exception as exc:
            if isinstance(options, dict):
                if previous_series_label is None:
                    options.pop("_mlp_training_series_label", None)
                else:
                    options["_mlp_training_series_label"] = previous_series_label
            rows.append(
                {
                    "heldout_group": str(heldout_group),
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "n_common_concepts": int(len(common_classes)),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "skipped": True,
                    "reason": str(exc),
                }
            )
            continue

        confidence = np.full(test_idx.size, np.nan, dtype=float)
        try:
            prob = clf.predict_proba(X[test_idx])
            confidence = np.nanmax(np.asarray(prob, dtype=float), axis=1)
        except Exception:
            pass

        template_similarity = np.full(test_idx.size, np.nan, dtype=float)
        template_margin = np.full(test_idx.size, np.nan, dtype=float)
        if decoder_key == "template":
            try:
                scores = np.asarray(clf.decision_function(X[test_idx]), dtype=float)
                if scores.ndim == 2 and scores.shape[0] == test_idx.size:
                    sorted_scores = np.sort(scores, axis=1)
                    template_similarity = sorted_scores[:, -1]
                    if scores.shape[1] > 1:
                        template_margin = sorted_scores[:, -1] - sorted_scores[:, -2]
            except Exception:
                pass

        for local_idx, row_idx in enumerate(test_idx.tolist()):
            pred_records.append(
                {
                    "index": int(row_idx),
                    "y_test": str(y[int(row_idx)]),
                    "y_pred": str(y_group_pred[int(local_idx)]),
                    "confidence": float(confidence[int(local_idx)]) if np.isfinite(confidence[int(local_idx)]) else np.nan,
                    "template_similarity": (
                        float(template_similarity[int(local_idx)])
                        if np.isfinite(template_similarity[int(local_idx)])
                        else np.nan
                    ),
                    "template_margin": (
                        float(template_margin[int(local_idx)])
                        if np.isfinite(template_margin[int(local_idx)])
                        else np.nan
                    ),
                    "heldout_group": str(heldout_group),
                }
            )
        rows.append(
            {
                "heldout_group": str(heldout_group),
                "decoder": str(decoder_label),
                "decoder_type": str(decoder_key),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "n_common_concepts": int(len(common_classes)),
                "accuracy": float(accuracy_score(y[test_idx], y_group_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], y_group_pred)),
                "skipped": False,
                "reason": "",
            }
        )

    pred_df = pd.DataFrame(pred_records)
    if pred_df.empty:
        return None, "No grouped held-out predictions could be generated."
    pred_df = pred_df.sort_values("index").reset_index(drop=True)
    return {
        "test_idx": pred_df["index"].to_numpy(dtype=int),
        "y_test": pred_df["y_test"].astype(str).to_numpy(),
        "y_pred": pred_df["y_pred"].astype(str).to_numpy(),
        "confidence": pd.to_numeric(pred_df["confidence"], errors="coerce").to_numpy(dtype=float),
        "template_similarity": pd.to_numeric(pred_df["template_similarity"], errors="coerce").to_numpy(dtype=float),
        "template_margin": pd.to_numeric(pred_df["template_margin"], errors="coerce").to_numpy(dtype=float),
        "prediction_groups": pred_df["heldout_group"].astype(str).to_numpy(),
        "split_df": pd.DataFrame(rows),
    }, None


def concept_feature_importance_from_model(clf, n_features):
    """Compute a simple model-specific feature-importance vector."""
    if clf is None:
        return np.full(int(n_features), np.nan, dtype=float)

    lr = getattr(clf, "named_steps", {}).get("logisticregression") if hasattr(clf, "named_steps") else None
    if lr is None:
        lr = getattr(clf, "named_steps", {}).get("linearsvc")
    coef = getattr(lr, "coef_", None)
    if coef is not None:
        importance = np.mean(np.abs(np.asarray(coef, dtype=float)), axis=0)
        if importance.size == int(n_features):
            return importance

    mlp = getattr(clf, "named_steps", {}).get("mlpclassifier") if hasattr(clf, "named_steps") else None
    coefs = getattr(mlp, "coefs_", None)
    if coefs is not None and len(coefs) > 0:
        first_layer = np.asarray(coefs[0], dtype=float)
        if first_layer.ndim == 2 and first_layer.shape[0] == int(n_features):
            return np.mean(np.abs(first_layer), axis=1)
        selector = getattr(clf, "named_steps", {}).get("selectkbest") if hasattr(clf, "named_steps") else None
        support = getattr(selector, "get_support", None)
        if first_layer.ndim == 2 and callable(support):
            try:
                mask = np.asarray(support(), dtype=bool)
                if mask.size == int(n_features) and int(np.sum(mask)) == first_layer.shape[0]:
                    importance = np.full(int(n_features), np.nan, dtype=float)
                    importance[mask] = np.mean(np.abs(first_layer), axis=1)
                    return importance
            except Exception:
                pass

    template = getattr(clf, "named_steps", {}).get("prototypecorrelationclassifier") if hasattr(clf, "named_steps") else None
    tmpl_importance = getattr(template, "feature_importances_", None)
    if tmpl_importance is not None:
        importance = np.asarray(tmpl_importance, dtype=float)
        if importance.size == int(n_features):
            return importance

    return np.full(int(n_features), np.nan, dtype=float)


def concept_template_payload_from_model(clf, feature_meta_df, max_rows_per_concept=100):
    """Return learned prototype templates and a compact inspection table."""
    empty = {
        "template_matrix": np.empty((0, 0), dtype=np.float32),
        "template_classes": [],
        "template_feature_meta_df": pd.DataFrame(),
        "template_df": pd.DataFrame(),
    }
    if clf is None or not hasattr(clf, "named_steps"):
        return empty

    template_step = clf.named_steps.get("prototypecorrelationclassifier")
    templates = getattr(template_step, "templates_", None)
    classes = getattr(template_step, "classes_", None)
    if templates is None or classes is None:
        return empty

    try:
        template_matrix = np.asarray(templates, dtype=np.float32)
    except Exception:
        return empty
    if template_matrix.ndim != 2 or template_matrix.shape[0] == 0 or template_matrix.shape[1] == 0:
        return empty
    template_classes = [str(v) for v in np.asarray(classes).tolist()]
    if len(template_classes) != template_matrix.shape[0]:
        template_classes = [f"class_{i}" for i in range(template_matrix.shape[0])]

    n_features = int(template_matrix.shape[1])
    if isinstance(feature_meta_df, pd.DataFrame) and len(feature_meta_df) == n_features:
        meta_df = feature_meta_df.copy().reset_index(drop=True)
    else:
        meta_df = pd.DataFrame({"feature": [f"feature_{i}" for i in range(n_features)]})
    if "feature_index" not in meta_df.columns:
        meta_df.insert(0, "feature_index", np.arange(n_features, dtype=int))

    max_rows_per_concept = max(1, int(max_rows_per_concept))
    rows = []
    for class_idx, concept in enumerate(template_classes):
        values = template_matrix[class_idx]
        finite_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        top_idx = np.argsort(np.abs(finite_values))[::-1][:max_rows_per_concept]
        compact = meta_df.iloc[top_idx].copy().reset_index(drop=True)
        compact.insert(0, "concept", concept)
        compact.insert(1, "template_value", values[top_idx])
        compact.insert(2, "abs_template_value", np.abs(values[top_idx]))
        rows.append(compact)
    template_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return {
        "template_matrix": template_matrix,
        "template_classes": template_classes,
        "template_feature_meta_df": meta_df,
        "template_df": template_df,
    }


def concept_permutation_baseline(
    X,
    y,
    test_fraction,
    observed_accuracy,
    observed_balanced_accuracy,
    n_permutations=10,
    decoder_type="logistic",
    options=None,
):
    """Run label-shuffle baselines for the per-concept random split."""
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

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=str)
    rng = np.random.default_rng(777)
    rows = []
    decoder_key = str(decoder_type or "").strip().lower()
    progress_every = 1 if decoder_key in {"mlp", "neural_net", "neural-network", "feedforward", "feed-forward"} else max(1, int(n_permutations) // 20)
    emit_concept_progress(options, f"running permutation baseline: {n_permutations} shuffled-label fit(s)...")
    for perm_idx in range(n_permutations):
        if perm_idx == 0 or (perm_idx + 1) % progress_every == 0:
            emit_concept_progress(
                options,
                f"permutation baseline {perm_idx + 1}/{n_permutations}: fitting shuffled labels...",
            )
        y_perm = y.copy()
        rng.shuffle(y_perm)
        train_idx, test_idx, _split_df = per_concept_train_test_split(
            y_perm,
            test_fraction=test_fraction,
            random_state=2000 + int(perm_idx),
        )
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        train_classes = set(y_perm[train_idx].tolist())
        test_classes = set(y_perm[test_idx].tolist())
        class_names = [str(v) for v in pd.Index(y_perm).drop_duplicates().tolist() if str(v) in train_classes and str(v) in test_classes]
        if len(class_names) < 2:
            continue
        valid_train = np.asarray([str(y_perm[i]) in class_names for i in train_idx], dtype=bool)
        valid_test = np.asarray([str(y_perm[i]) in class_names for i in test_idx], dtype=bool)
        train_idx = train_idx[valid_train]
        test_idx = test_idx[valid_test]
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        clf, decoder_label, _decoder_key = concept_decoder_spec(
            decoder_type=decoder_type,
            random_state=1000 + int(perm_idx),
            options=options,
            n_features=X.shape[1],
            n_classes=len(class_names),
        )
        try:
            fit_options = dict(options or {})
            fit_options["emit_mlp_epoch_progress"] = False
            fit_options["emit_mlp_training_metrics"] = False
            fit_options.pop("training_metric_callback", None)
            fit_concept_decoder(clf, X[train_idx], y_perm[train_idx], options=fit_options)
            y_pred = clf.predict(X[test_idx])
            y_test = y_perm[test_idx]
            rows.append(
                {
                    "permutation": int(perm_idx + 1),
                    "decoder": str(decoder_label),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                }
            )
        except Exception:
            continue
        if rows and ((perm_idx + 1) % progress_every == 0 or (perm_idx + 1) == n_permutations):
            latest = rows[-1]
            emit_concept_progress(
                options,
                (
                    f"permutation baseline {perm_idx + 1}/{n_permutations}: "
                    f"completed={len(rows)}, accuracy={latest.get('accuracy', np.nan):.3f}, "
                    f"balanced={latest.get('balanced_accuracy', np.nan):.3f}"
                ),
            )

    permutation_df = pd.DataFrame(rows)
    if permutation_df.empty:
        return {
            "permutation_df": permutation_df,
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

    perm_acc = pd.to_numeric(permutation_df["accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
    perm_bal = pd.to_numeric(permutation_df["balanced_accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
    p_acc = np.nan
    p_bal = np.nan
    if perm_acc.size > 0 and np.isfinite(observed_accuracy):
        p_acc = float((1 + np.sum(perm_acc >= float(observed_accuracy))) / (perm_acc.size + 1))
    if perm_bal.size > 0 and np.isfinite(observed_balanced_accuracy):
        p_bal = float((1 + np.sum(perm_bal >= float(observed_balanced_accuracy))) / (perm_bal.size + 1))

    return {
        "permutation_df": permutation_df,
        "permutation_n_completed": int(len(permutation_df)),
        "permutation_mean_accuracy": float(np.nanmean(perm_acc)) if perm_acc.size > 0 else np.nan,
        "permutation_std_accuracy": float(np.nanstd(perm_acc)) if perm_acc.size > 0 else np.nan,
        "permutation_max_accuracy": float(np.nanmax(perm_acc)) if perm_acc.size > 0 else np.nan,
        "permutation_mean_balanced_accuracy": float(np.nanmean(perm_bal)) if perm_bal.size > 0 else np.nan,
        "permutation_std_balanced_accuracy": float(np.nanstd(perm_bal)) if perm_bal.size > 0 else np.nan,
        "permutation_max_balanced_accuracy": float(np.nanmax(perm_bal)) if perm_bal.size > 0 else np.nan,
        "permutation_p_value_accuracy": p_acc,
        "permutation_p_value_balanced_accuracy": p_bal,
    }


def compute_concept_decoding_from_payload(context, options, feature_payload, feature_space):
    """Train/test a concept decoder from an already-built feature payload."""
    if options is None:
        options = {}
    if feature_payload is None:
        return None, "Could not build concept decoding features."

    if options.get("decoder_type") == "pairwise_svm":
        from core.concept_decoding_pairwise import compute_pairwise_decoding
        return compute_pairwise_decoding({**feature_payload, "feature_space": feature_space}, options)

    X = np.asarray(feature_payload["X"], dtype=float)
    y = np.asarray(feature_payload["y"], dtype=str)
    trial_df = feature_payload["trial_df"].copy()
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

    decoder_type = str(options.get("decoder_type", "logistic") or "logistic").strip().lower()
    decoder_note = ""

    requested_split_mode = str(options.get("split_mode", "random") or "random").strip().lower()
    split_mode = requested_split_mode
    clf, decoder_label, decoder_key = concept_decoder_spec(
        decoder_type=decoder_type,
        random_state=42,
        options=options,
        n_features=X.shape[1],
        n_classes=len(pd.Index(y).drop_duplicates()),
    )
    if feature_space == "source":
        decoder_label = decoder_label.replace("signal-space", "source-space")

    split_label = "per-concept random"
    prediction_groups = None
    template_similarity = None
    template_margin = None
    if split_mode in GROUPED_SPLIT_MODES:
        emit_concept_progress(options, f"training grouped {decoder_label} models...")
        group_col, split_label = concept_group_column_for_split(trial_df, split_mode)
        if group_col is None:
            return None, f"{split_label} requires at least two runs/sessions in the mapped trial table."
        grouped_result, grouped_err = grouped_concept_predictions(
            X,
            y,
            trial_df,
            group_col=group_col,
            decoder_type=decoder_key,
            options=options,
        )
        if grouped_result is None:
            return None, grouped_err or f"{split_label} failed."
        test_idx = np.asarray(grouped_result["test_idx"], dtype=int)
        y_test = np.asarray(grouped_result["y_test"], dtype=str)
        y_pred = np.asarray(grouped_result["y_pred"], dtype=str)
        confidence = np.asarray(grouped_result["confidence"], dtype=float)
        template_similarity = np.asarray(grouped_result["template_similarity"], dtype=float)
        template_margin = np.asarray(grouped_result["template_margin"], dtype=float)
        prediction_groups = np.asarray(grouped_result["prediction_groups"], dtype=str)
        split_df = grouped_result["split_df"]
        class_names = [
            str(v)
            for v in pd.Index(y).drop_duplicates().tolist()
            if str(v) in set(y_test.tolist()) and str(v) in set(y_pred.tolist()).union(set(y_test.tolist()))
        ]
        if len(class_names) < 2:
            return None, f"{split_label} needs at least two concepts represented in held-out predictions."
        train_idx = np.arange(len(y), dtype=int)
        try:
            emit_concept_progress(options, f"fitting final {decoder_label} on all valid trials...")
            previous_series_label = options.get("_mlp_training_series_label")
            options["_mlp_training_series_label"] = "Final fit on all valid trials"
            fit_concept_decoder(
                clf,
                X[train_idx],
                y[train_idx],
                options=options,
            )
            if previous_series_label is None:
                options.pop("_mlp_training_series_label", None)
            else:
                options["_mlp_training_series_label"] = previous_series_label
        except Exception as exc:
            if previous_series_label is None:
                options.pop("_mlp_training_series_label", None)
            else:
                options["_mlp_training_series_label"] = previous_series_label
            return None, f"{decoder_label} final training failed: {exc}"
        n_permutations = 0
    else:
        split_mode = "random"
        train_idx, test_idx, split_df = per_concept_train_test_split(
            y,
            test_fraction=test_fraction,
            random_state=42,
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
        if train_idx.size == 0 or test_idx.size == 0:
            return None, "No trials remained after aligning train/test concept classes."

        try:
            emit_concept_progress(
                options,
                f"training {decoder_label}: train={train_idx.size}, test={test_idx.size}, concepts={len(class_names)}",
            )
            previous_series_label = options.get("_mlp_training_series_label")
            options["_mlp_training_series_label"] = "Train/validation fit"
            fit_concept_decoder(
                clf,
                X[train_idx],
                y[train_idx],
                options=options,
                eval_X=X[test_idx],
                eval_y=y[test_idx],
            )
            if previous_series_label is None:
                options.pop("_mlp_training_series_label", None)
            else:
                options["_mlp_training_series_label"] = previous_series_label
            emit_concept_progress(options, "predicting held-out trials...")
            y_pred = clf.predict(X[test_idx])
            emit_concept_progress(options, "held-out prediction complete; estimating confidence...")
        except Exception as exc:
            if previous_series_label is None:
                options.pop("_mlp_training_series_label", None)
            else:
                options["_mlp_training_series_label"] = previous_series_label
            return None, f"{decoder_label} training failed: {exc}"
        y_test = y[test_idx]
        confidence = np.full(test_idx.size, np.nan, dtype=float)
        try:
            prob = clf.predict_proba(X[test_idx])
            confidence = np.nanmax(np.asarray(prob, dtype=float), axis=1)
        except Exception:
            pass
        if decoder_key == "template":
            template_similarity = np.full(test_idx.size, np.nan, dtype=float)
            template_margin = np.full(test_idx.size, np.nan, dtype=float)
            try:
                scores = np.asarray(clf.decision_function(X[test_idx]), dtype=float)
                if scores.ndim == 2 and scores.shape[0] == test_idx.size:
                    sorted_scores = np.sort(scores, axis=1)
                    template_similarity = sorted_scores[:, -1]
                    if scores.shape[1] > 1:
                        template_margin = sorted_scores[:, -1] - sorted_scores[:, -2]
            except Exception:
                pass

    emit_concept_progress(options, "computing held-out metrics...")
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    chance_accuracy = float(1.0 / max(1, len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    report = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    if n_permutations > 0:
        emit_concept_progress(options, f"starting permutation baseline ({n_permutations} fit(s))...")
    permutation_stats = concept_permutation_baseline(
        X,
        y,
        test_fraction=test_fraction,
        observed_accuracy=acc,
        observed_balanced_accuracy=bal_acc,
        n_permutations=n_permutations,
        decoder_type=decoder_key,
        options=options,
    )
    if int(permutation_stats.get("permutation_n_completed", 0) or 0) > 0:
        emit_concept_progress(
            options,
            f"completed {int(permutation_stats.get('permutation_n_completed', 0))} permutation baseline runs...",
        )

    emit_concept_progress(options, "building held-out prediction tables...")
    predictions_df = trial_df.iloc[test_idx].copy().reset_index(drop=True)
    predictions_df.insert(0, "true_concept", y_test)
    predictions_df.insert(1, "predicted_concept", y_pred)
    predictions_df.insert(2, "correct", y_test == y_pred)
    predictions_df.insert(3, "confidence", confidence)
    insert_at = 4
    if template_similarity is not None:
        predictions_df.insert(insert_at, "template_similarity", template_similarity)
        insert_at += 1
    if template_margin is not None:
        predictions_df.insert(insert_at, "template_margin", template_margin)
        insert_at += 1
    if prediction_groups is not None:
        predictions_df.insert(insert_at, "heldout_group", prediction_groups)
    keep_cols = [
        "true_concept",
        "predicted_concept",
        "correct",
        "confidence",
        "template_similarity",
        "template_margin",
        "heldout_group",
        "trial_index",
        "stimulus",
        "canonical_stimulus",
        "stimulus_group",
        "source_fif",
        "source_fif_path",
        "session_id",
        "run_id",
        "onset_meg_s",
        "qc_outlier",
        "qc_outlier_score",
        "qc_max_abs",
        "qc_ptp_max",
        "qc_rms_median",
        "qc_baseline_rms_median",
    ]
    predictions_df = predictions_df[[col for col in keep_cols if col in predictions_df.columns]]

    per_class_accuracy_df = (
        pd.DataFrame({"concept": y_test, "correct": y_test == y_pred})
        .groupby("concept", sort=False)
        .agg(test_accuracy=("correct", "mean"), n_test=("correct", "size"))
        .reset_index()
    )
    if isinstance(split_df, pd.DataFrame) and "concept" in split_df.columns:
        split_df = split_df.merge(per_class_accuracy_df, on="concept", how="left")

    emit_concept_progress(options, "computing model feature summaries...")
    importance = concept_feature_importance_from_model(clf, X.shape[1])
    feature_meta_df = feature_payload["feature_meta_df"].copy()
    feature_meta_df["importance"] = importance
    feature_meta_df["mean_abs_train_signal"] = np.nanmean(np.abs(X[train_idx]), axis=0)
    feature_meta_df["std_train_signal"] = np.nanstd(X[train_idx], axis=0)
    feature_df = (
        feature_meta_df.sort_values("importance", ascending=False, na_position="last")
        .head(250)
        .reset_index(drop=True)
    )
    template_payload = concept_template_payload_from_model(
        clf,
        feature_meta_df,
        max_rows_per_concept=100,
    ) if decoder_key == "template" else {
        "template_matrix": np.empty((0, 0), dtype=np.float32),
        "template_classes": [],
        "template_feature_meta_df": pd.DataFrame(),
        "template_df": pd.DataFrame(),
    }
    if decoder_key == "mlp":
        mlp_epochs = _option_int(options, "mlp_epochs", (options or {}).get("epochs", 100), lower=1, upper=5000)
        mlp_feature_select_k = _mlp_feature_select_k(options, X.shape[1], len(class_names))
        mlp_hidden_1 = _option_int(options, "mlp_hidden_1", 64, lower=16, upper=4096)
        mlp_hidden_2 = _option_int(options, "mlp_hidden_2", 32, lower=0, upper=4096)
        mlp_hidden_layers = f"{mlp_hidden_1}->{mlp_hidden_2}" if mlp_hidden_2 > 0 else str(mlp_hidden_1)
        mlp_batch_size = _option_int(options, "mlp_batch_size", 32, lower=4, upper=4096)
        mlp_dropout = _option_float(options, "mlp_dropout", 0.35, lower=0.0, upper=0.9)
        mlp_weight_decay = _option_float(options, "mlp_weight_decay", 1e-3, lower=0.0, upper=1.0)
        mlp_validation_fraction = _mlp_validation_fraction(options)
        mlp_step = getattr(clf, "named_steps", {}).get("mlpclassifier") if hasattr(clf, "named_steps") else None
        mlp_n_fit = int(getattr(mlp_step, "training_sample_count_", 0) or 0) if mlp_step is not None else 0
        mlp_n_validation = int(getattr(mlp_step, "validation_sample_count_", 0) or 0) if mlp_step is not None else 0
        mlp_best_epoch = int(getattr(mlp_step, "best_validation_epoch_", 0) or 0) if mlp_step is not None else 0
        mlp_best_validation_accuracy = (
            float(getattr(mlp_step, "best_validation_accuracy_", np.nan)) if mlp_step is not None else np.nan
        )
        mlp_best_validation_balanced_accuracy = (
            float(getattr(mlp_step, "best_validation_balanced_accuracy_", np.nan)) if mlp_step is not None else np.nan
        )
        mlp_training_history_df = pd.DataFrame(
            list(getattr(mlp_step, "training_history_", []) or [])
        )
    else:
        mlp_epochs = 0
        mlp_feature_select_k = 0
        mlp_hidden_layers = ""
        mlp_batch_size = 0
        mlp_dropout = 0.0
        mlp_weight_decay = 0.0
        mlp_validation_fraction = 0.0
        mlp_n_fit = 0
        mlp_n_validation = 0
        mlp_best_epoch = 0
        mlp_best_validation_accuracy = np.nan
        mlp_best_validation_balanced_accuracy = np.nan
        mlp_training_history_df = pd.DataFrame()

    summary = {
        "decoder": decoder_label,
        "decoder_type": decoder_key,
        "split": (
            f"{int(round((1.0 - test_fraction) * 100))}/{int(round(test_fraction * 100))} per concept"
            if split_mode == "random"
            else split_label
        ),
        "split_mode": split_mode,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "chance_accuracy": chance_accuracy,
        "above_chance_accuracy": bool(acc > chance_accuracy),
        "above_chance_balanced_accuracy": bool(bal_acc > chance_accuracy),
        "n_trials": int(len(y)),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_concepts": int(len(class_names)),
        "trial_set_id": concept_trial_set_id(trial_df),
        "n_runs": int(feature_payload.get("n_runs", 0)),
        "n_channels": int(len(feature_payload.get("channel_names", []))),
        "n_rois": int(len(feature_payload.get("roi_names", []))),
        "n_features": int(X.shape[1]),
        "window_start_s": float(feature_payload.get("window_start")),
        "window_end_s": float(feature_payload.get("window_end")),
        "bandpass_low_hz": feature_payload.get("bandpass_low_hz", np.nan),
        "bandpass_high_hz": feature_payload.get("bandpass_high_hz", np.nan),
        "n_time_bins": int(feature_payload.get("n_time_points", len(feature_payload.get("bin_edges", []))) or 0),
        "feature_binning_enabled": bool(feature_payload.get("feature_binning_enabled", True)),
        "resample_enabled": bool(feature_payload.get("resample_enabled", False)),
        "resample_sfreq": feature_payload.get("resample_sfreq", np.nan),
        "feature_space": str(feature_payload.get("feature_space", feature_space)),
        "channel_mode": str(feature_payload.get("channel_mode")),
        "group_mode": str(feature_payload.get("group_mode")),
        "baseline_correct": bool(feature_payload.get("baseline_correct")),
        "feature_mode": str(feature_payload.get("feature_mode", "")),
        "band_defs": feature_payload.get("band_defs", []),
        "n_bands": int(feature_payload.get("n_bands", 0)),
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
        "mne_epoch_reject_enabled": bool(feature_payload.get("mne_epoch_reject_enabled", False)),
        "mne_reject_mag_ft": feature_payload.get("mne_reject_mag_ft", np.nan),
        "mne_reject_grad_ft_cm": feature_payload.get("mne_reject_grad_ft_cm", np.nan),
        "epoch_tmin_s": feature_payload.get("epoch_tmin", np.nan),
        "epoch_tmax_s": feature_payload.get("epoch_tmax", np.nan),
        "exclude_trial_outliers": bool(feature_payload.get("exclude_trial_outliers", False)),
        "trial_outlier_z": feature_payload.get("trial_outlier_z", np.nan),
        "source_method": str(feature_payload.get("method", "")),
        "source_lambda2": feature_payload.get("lambda2", np.nan),
        "source_batch_size": int(feature_payload.get("source_batch_size", 0) or 0),
        "source_space_kind": str(feature_payload.get("source_space_kind", "")),
        "source_parcellation": str(feature_payload.get("source_parcellation", "")),
        "source_common_space": str(feature_payload.get("source_common_space", "")),
        "source_subject": str(feature_payload.get("subject", "")),
        "source_subjects_dir": str(feature_payload.get("subjects_dir", "")),
        "source_subject_count": int(feature_payload.get("source_subject_count", 0) or 0),
        "source_subjects": ", ".join(str(v) for v in feature_payload.get("native_subjects", []) if str(v).strip()),
        "mlp_epochs": int(mlp_epochs),
        "mlp_feature_select_k": int(mlp_feature_select_k),
        "mlp_hidden_layers": str(mlp_hidden_layers),
        "mlp_batch_size": int(mlp_batch_size),
        "mlp_dropout": float(mlp_dropout),
        "mlp_weight_decay": float(mlp_weight_decay),
        "mlp_validation_fraction": float(mlp_validation_fraction),
        "mlp_n_fit": int(mlp_n_fit),
        "mlp_n_validation": int(mlp_n_validation),
        "mlp_best_epoch": int(mlp_best_epoch),
        "mlp_best_validation_accuracy": mlp_best_validation_accuracy,
        "mlp_best_validation_balanced_accuracy": mlp_best_validation_balanced_accuracy,
        "decoder_note": decoder_note.lstrip(", "),
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
    qc_summary_df = concept_qc_summary_table(trial_qc_df, feature_payload=feature_payload)
    architecture_df = concept_model_architecture_table(summary, feature_payload, class_names)
    summary_df = pd.DataFrame([summary])

    result = {
        **summary,
        "class_names": class_names,
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
        "training_history_df": mlp_training_history_df,
        "template_df": template_payload.get("template_df", pd.DataFrame()),
        "template_matrix": template_payload.get("template_matrix", np.empty((0, 0), dtype=np.float32)),
        "template_classes": template_payload.get("template_classes", []),
        "template_feature_meta_df": template_payload.get("template_feature_meta_df", pd.DataFrame()),
    }
    skipped_paths = feature_payload.get("skipped_paths", [])
    skip_msg = f", skipped_runs={len(skipped_paths)}" if skipped_paths else ""
    perm_completed = int(permutation_stats.get("permutation_n_completed", 0) or 0)
    if perm_completed > 0:
        p_bal = permutation_stats.get("permutation_p_value_balanced_accuracy", np.nan)
        perm_mean = permutation_stats.get("permutation_mean_balanced_accuracy", np.nan)
        try:
            perm_msg = f", chance={chance_accuracy:.3f}, perm_bal_mean={float(perm_mean):.3f}, p_bal={float(p_bal):.3f}"
        except Exception:
            perm_msg = f", chance={chance_accuracy:.3f}, permutations={perm_completed}"
    else:
        perm_msg = f", chance={chance_accuracy:.3f}, permutations=off"
    feature_count_label = "rois" if feature_space == "source" else "channels"
    feature_count = summary["n_rois"] if feature_space == "source" else summary["n_channels"]
    cache_hits = int(summary.get("epoch_cache_hits", 0) or 0)
    cache_misses = int(summary.get("epoch_cache_misses", 0) or 0)
    cache_msg = f", epoch_cache={cache_hits} hit/{cache_misses} miss" if (cache_hits or cache_misses) else ""
    source_space_msg = ""
    if feature_space == "source":
        source_space = str(summary.get("source_common_space", "") or "").strip()
        source_kind = str(summary.get("source_space_kind", "") or "").strip()
        source_parcellation = str(summary.get("source_parcellation", "") or "").strip()
        subject_count = int(summary.get("source_subject_count", 0) or 0)
        if source_kind:
            source_space_msg = f", source_kind={source_kind}"
        if source_parcellation:
            source_space_msg += f", atlas={source_parcellation}"
        if source_space:
            source_space_msg += f", source_space={source_space}"
            if subject_count > 1:
                source_space_msg += f"({subject_count} subjects)"
    bandpass_low = summary.get("bandpass_low_hz", np.nan)
    bandpass_high = summary.get("bandpass_high_hz", np.nan)
    try:
        bandpass_msg = f", bandpass={float(bandpass_low):g}-{float(bandpass_high):g}Hz"
    except Exception:
        bandpass_msg = ""
    msg = (
        f"accuracy={acc:.3f}, balanced_accuracy={bal_acc:.3f}, "
        f"concepts={len(class_names)}, train/test={train_idx.size}/{test_idx.size}, "
        f"runs={summary['n_runs']}, {feature_count_label}={feature_count}, features={summary['n_features']}"
        f", dropped_trials={summary['dropped_trials']}, qc_outliers={summary['qc_outlier_trials']}"
        f"{cache_msg}"
        f"{source_space_msg}"
        f"{bandpass_msg}"
        f"{decoder_note}{perm_msg}{skip_msg}"
    )
    return result, msg


def compute_signal_space_concept_decoding(context, options=None):
    """Train/test concept decoder in signal space."""
    if options is None:
        options = {}
    if not getattr(context, "b_fif_loaded", False):
        return None, "No FIF loaded. Please load a FIF file before running concept decoding."
    decoder_type = str(options.get("decoder_type", "logistic") or "logistic").strip().lower()
    if decoder_type in {
        "transformer",
        "axial_transformer",
        "registered_transformer",
        "embedding_ridge",
        "ridge_embedding",
        "word2vec_ridge",
        "linear_embedding",
        "embedding_contrastive",
        "contrastive_embedding",
        "contrastive_linear",
        "linear_contrastive",
        "class_aware_contrastive",
        "embedding_mlp",
        "mlp_embedding",
        "word2vec_mlp",
        "neural_embedding",
        "zero_shot",
        "zero_shot_decoding",
        "zero_shot_embedding",
        "zero_shot_ridge",
        "zero_shot_linear",
        "zero_shot_contrastive",
        "zero_shot_mlp",
        "zeroshot",
    } or bool(options.get("zero_shot", False)) or str(options.get("split_mode", "")).strip().lower() in {"zero_shot", "unseen_concepts"}:
        from core.concept_decoding_transformer import (
            compute_signal_space_semantic_embedding_concept_decoding,
        )

        return compute_signal_space_semantic_embedding_concept_decoding(context, options)
    if decoder_type == "pairwise_svm":
        options = {**options, "feature_binning_enabled": False, "feature_mode": "raw"}
    split_err = validate_grouped_split(context, options)
    if split_err:
        return None, split_err
    feature_payload, err_msg = build_signal_space_concept_features(context, options)
    if feature_payload is None:
        return None, err_msg or "Could not build signal-space concept features."
    emit_concept_progress(options, "training/testing signal-space decoder...")
    return compute_concept_decoding_from_payload(context, options, feature_payload, "signal")
