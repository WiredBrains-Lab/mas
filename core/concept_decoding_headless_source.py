"""Anatomy/inverse context for CLI source decoding, without a Qt viewer."""

import os
from pathlib import Path

import mne
import numpy as np


class HeadlessSourceContext:
    """Supply the source backend's context hooks using explicit run mappings.

    Inverse operators must already exist. This adapter never constructs a
    forward model, guesses a run's inverse, or modifies anatomical files.
    """

    def __init__(self, runs, source_space_kind):
        self.runs = {os.path.abspath(row["fif"]): dict(row) for row in runs}
        self._meg_source_space_kind = source_space_kind
        self._meg_current_fif_path = next(iter(self.runs))
        self._validated = set()
        self._atlas_cache = {}

    def attach(self, context):
        context.meg_3d = self
        for name in (
            "_auto_configure_freesurfer_subject_for_fif",
            "_find_inverse_operator_for_fif", "_inverse_operator_matches_fif",
            "_create_inverse_operator_for_fif", "_get_available_semantic_parcellations",
            "_ctx_volume_name_to_annot_name",
        ):
            setattr(context, name, getattr(self, name))

    def validate(self, options):
        """Check every run before extraction, including atlas names and bands."""
        from core.concept_decoding_source import (
            concept_source_labels_for_subject, concept_volume_labels_for_subject,
        )
        from core.concept_decoding_spectral import resolve_source_power_bands

        subjects = {(row["subject"], row["subjects_dir"]) for row in self.runs.values()}
        if self._meg_source_space_kind == "volume" and len(subjects) != 1:
            raise ValueError("Volume source decoding supports one anatomical subject per run batch.")
        atlas_checked = set()
        for fif, row in self.runs.items():
            for key in ("fif", "inverse"):
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(f"Source {key} does not exist: {row[key]}")
            subject_dir = Path(row["subjects_dir"]) / row["subject"]
            if not subject_dir.is_dir():
                raise FileNotFoundError(f"FreeSurfer subject directory does not exist: {subject_dir}")
            inv = mne.minimum_norm.read_inverse_operator(row["inverse"], verbose="error")
            kinds = {part["type"] for part in inv["src"]}
            expected = {"surf"} if self._meg_source_space_kind == "surface" else {"vol", "discrete"}
            if not kinds or not kinds.issubset(expected):
                raise ValueError(f"Inverse {row['inverse']} does not match {self._meg_source_space_kind} source space.")
            inverse_subjects = {part.get("subject_his_id") for part in inv["src"]} - {None, ""}
            if inverse_subjects and inverse_subjects != {row["subject"]}:
                raise ValueError(f"Inverse subject {sorted(inverse_subjects)} does not match {row['subject']}.")
            info = mne.io.read_info(fif, verbose="error")
            required = set(inv["noise_cov"]["names"]) - set(inv["info"]["bads"])
            available = set(info["ch_names"]) - set(info["bads"])
            if required - available:
                raise ValueError(f"Inverse channels missing or marked bad in {fif}: {sorted(required - available)}")
            transform_keys = [key for key in ("dev_head_t", "ctf_head_t") if inv["info"].get(key) is not None]
            if not transform_keys and any(info.get(key) is not None for key in ("dev_head_t", "ctf_head_t")):
                raise ValueError(f"Inverse lacks a device-to-head transform for {fif}")
            for key in transform_keys:
                inverse_head, raw_head = inv["info"][key], info.get(key)
                if raw_head is None or not np.allclose(inverse_head["trans"], raw_head["trans"], atol=1e-6, rtol=1e-6):
                    raise ValueError(f"Inverse and FIF have different {key} transforms: {fif}")
            if options.get("feature_mode") == "bandpower":
                sfreq = float(info["sfreq"])
                if options.get("mne_resample_enabled"):
                    target = float(options["resample_sfreq"])
                    if target > sfreq:
                        raise ValueError("Source downsampling target is above the recording sampling rate.")
                    sfreq = target
                bands = resolve_source_power_bands(
                    options.get("source_power_bands", "all"), sfreq,
                    options["bandpass_low_hz"], options["bandpass_high_hz"],
                )
                padding = options.get("source_power_padding_s")
                minimum_padding = max(0.5, 3.0 / min(band[1] for band in bands))
                if padding is not None and (not np.isfinite(padding) or padding < minimum_padding):
                    raise ValueError(f"Source power padding must be at least {minimum_padding:g} seconds.")
            atlas_key = (row["subject"], row["subjects_dir"])
            if atlas_key not in atlas_checked:
                if self._meg_source_space_kind == "volume":
                    _path, _names, _indices, _selected, available_names = concept_volume_labels_for_subject(
                        *atlas_key, parcellation=options.get("source_parcellation", "aparc"),
                    )
                else:
                    _labels, _selected, available_names, _parc = concept_source_labels_for_subject(
                        self, *atlas_key, parcellation=options["source_parcellation"],
                    )
                missing = set(options.get("selected_parcellations", [])) - set(available_names)
                if missing:
                    raise ValueError(f"Requested source parcels are absent from the atlas: {sorted(missing)}")
                atlas_checked.add(atlas_key)
            self._validated.add((os.path.abspath(row["inverse"]), fif))

    def _auto_configure_freesurfer_subject_for_fif(self, fif_path, require_bem=False):
        path = os.path.abspath(fif_path)
        if path not in self.runs:
            return False, f"No source inverse/anatomy mapping for {path}"
        self._meg_current_fif_path = path
        return True, ""

    def _get_brain_subject_info(self):
        row = self.runs[self._meg_current_fif_path]
        return row["subject"], row["subjects_dir"]

    def load_fif_sensors(self, path):
        self._auto_configure_freesurfer_subject_for_fif(path)

    def _find_inverse_operator_for_fif(self, fif_path):
        return self.runs[os.path.abspath(fif_path)]["inverse"]

    def _inverse_operator_matches_fif(self, inverse_path, fif_path):
        return (os.path.abspath(inverse_path), os.path.abspath(fif_path)) in self._validated

    def _create_inverse_operator_for_fif(self, fif_path, output_path=None):
        raise ValueError(f"No validated saved inverse for {fif_path}; CLI source decoding requires an existing matching inverse.")

    @staticmethod
    def _ctx_volume_name_to_annot_name(label_name):
        text = str(label_name)
        if text.startswith("ctx-lh-"):
            return f"{text[7:]}-lh"
        if text.startswith("ctx-rh-"):
            return f"{text[7:]}-rh"
        return text

    def _get_available_semantic_parcellations(self, subject, subjects_dir):
        key = (subject, subjects_dir)
        if key not in self._atlas_cache:
            labels = mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=subjects_dir, verbose="error")
            self._atlas_cache[key] = [
                f"ctx-{label.name[-2:]}-{label.name[:-3]}" for label in labels
                if label.name.endswith(("-lh", "-rh")) and "unknown" not in label.name.lower()
            ]
        return self._atlas_cache[key]
