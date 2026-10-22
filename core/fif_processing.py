"""Shared, GUI-independent FIF preprocessing and persistent trial tables.

Both Orchestrator and the decoding CLI inherit this implementation. Keep the
trigger matching and ICA policy here so GUI and batch analyses stay identical.
"""
import glob
import hashlib
import json
import os
import re
import tempfile
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd

from core.meg_channels import emit_bad_meg_channel_log, good_meg_picks


class FIFProcessingMixin:
    def _infer_run_id_from_fif_filename(self, fif_filename):
        name = os.path.basename(str(fif_filename))
        m = re.search(r"run(\d+)", name, flags=re.IGNORECASE)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None


    def _infer_session_id_from_fif_filename(self, fif_filename):
        """Return a session number from names such as ``*_s2_run3_*``."""
        name = os.path.basename(str(fif_filename))
        m = re.search(r"(?:^|[_-])s(?:ession)?(\d+)(?=[_-]|$)", name, flags=re.IGNORECASE)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None


    @staticmethod
    def _event_log_timing_match_score(onset_rows, onset_events, sfreq):
        """Score a log against FIF triggers using constant-offset-invariant ISIs."""
        if onset_rows is None or onset_rows.empty or onset_events is None:
            return np.nan
        try:
            log_onsets = pd.to_numeric(onset_rows["onset_time"], errors="coerce").to_numpy(dtype=float)
            fif_onsets = np.asarray(onset_events, dtype=np.int64)[:, 0].astype(float) / float(sfreq)
        except Exception:
            return np.nan
        n = min(int(log_onsets.size), int(fif_onsets.size))
        if n < 3:
            return np.nan
        log_isi = np.diff(log_onsets[:n])
        fif_isi = np.diff(fif_onsets[:n])
        finite = np.isfinite(log_isi) & np.isfinite(fif_isi)
        if int(np.sum(finite)) < 2:
            return np.nan
        # Mean absolute ISI error distinguishes repeated acquisitions even when
        # their trial counts and nominal presentation durations are identical.
        return float(np.mean(np.abs(log_isi[finite] - fif_isi[finite])))


    def _candidate_events_log_dirs(self, fif_filename):
        """Return preferred events_log directory candidates for a FIF file."""
        fif_dir = os.path.dirname(os.path.abspath(str(fif_filename)))
        parent_dir = os.path.dirname(fif_dir)
        candidates = [
            os.path.join(parent_dir, "events_log"),
            os.path.join(fif_dir, "events_log"),
        ]
        # Keep order and drop duplicates.
        unique_candidates = []
        for path in candidates:
            if path not in unique_candidates:
                unique_candidates.append(path)
        return unique_candidates


    def _load_event_log_onsets_for_fif(self, fif_filename, onset_events=None, raw=None):
        """Load onset rows from events_log CSV matching current FIF run."""
        events_dir = None
        for candidate in self._candidate_events_log_dirs(fif_filename):
            if os.path.isdir(candidate):
                events_dir = candidate
                break
        if events_dir is None:
            return pd.DataFrame(), {"events_dir_candidates": self._candidate_events_log_dirs(fif_filename)}

        csv_paths = sorted(glob.glob(os.path.join(events_dir, "*.csv")))
        if not csv_paths:
            return pd.DataFrame(), {"events_dir": events_dir}

        run_id = self._infer_run_id_from_fif_filename(fif_filename)
        session_id = self._infer_session_id_from_fif_filename(fif_filename)
        candidates = []

        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "event_code" not in df.columns:
                continue
            req_cols = {"trial", "stimulus", "event_code", "onset_time"}
            if not req_cols.issubset(set(df.columns)):
                continue
            onset_rows = df[df["event_code"] == 2].copy()
            onset_rows = onset_rows.dropna(subset=["onset_time", "stimulus"])
            onset_rows["onset_time"] = pd.to_numeric(onset_rows["onset_time"], errors="coerce")
            onset_rows = onset_rows.dropna(subset=["onset_time"])
            onset_rows["trial"] = pd.to_numeric(onset_rows["trial"], errors="coerce")
            onset_rows = onset_rows.sort_values(by=["trial", "onset_time"]).reset_index(drop=True)
            if onset_rows.empty:
                continue

            block_id_val = None
            if "block_id" in onset_rows.columns:
                values = pd.to_numeric(onset_rows["block_id"], errors="coerce").dropna().unique()
                if len(values) == 1:
                    block_id_val = int(values[0])
            if block_id_val is None:
                match = re.search(r"(?:^|_)b(\d+)(?:_|$)", os.path.basename(path), flags=re.IGNORECASE)
                block_id_val = int(match.group(1)) if match is not None else None

            session_id_val = None
            if "session_id" in onset_rows.columns:
                values = pd.to_numeric(onset_rows["session_id"], errors="coerce").dropna().unique()
                if len(values) == 1:
                    session_id_val = int(values[0])
            if session_id_val is None:
                match = re.search(r"(?:^|_)s(\d+)(?:_|$)", os.path.basename(path), flags=re.IGNORECASE)
                session_id_val = int(match.group(1)) if match is not None else None

            candidates.append(
                {
                    "path": path,
                    "df": df,
                    "onset_rows": onset_rows,
                    "block_id": block_id_val,
                    "session_id": session_id_val,
                }
            )

        meta_base = {
            "events_dir": events_dir,
            "run_id": run_id,
            "session_id_from_fif": session_id,
        }
        if run_id is not None:
            candidates = [item for item in candidates if item["block_id"] == run_id]
        if session_id is not None:
            candidates = [item for item in candidates if item["session_id"] == session_id]
        if not candidates:
            meta_base["event_log_match_error"] = "No event log matches the FIF session and run."
            return pd.DataFrame(), meta_base

        timing_scores = {}
        if len(candidates) > 1 and onset_events is not None and raw is not None:
            sfreq = float(raw.info["sfreq"])
            for item in candidates:
                score = self._event_log_timing_match_score(item["onset_rows"], onset_events, sfreq)
                item["timing_score"] = score
                timing_scores[os.path.basename(item["path"])] = score
            finite_candidates = [item for item in candidates if np.isfinite(item.get("timing_score", np.nan))]
            if finite_candidates:
                finite_candidates.sort(key=lambda item: (float(item["timing_score"]), str(item["path"])))
                if len(finite_candidates) == 1:
                    candidates = finite_candidates
                else:
                    best = float(finite_candidates[0]["timing_score"])
                    second = float(finite_candidates[1]["timing_score"])
                    # Require a meaningful separation; otherwise retain the
                    # ambiguity and refuse to attach labels silently.
                    if second - best > max(1e-6, 0.05 * max(best, 1e-12)):
                        candidates = finite_candidates[:1]
                    else:
                        candidates = finite_candidates

        if len(candidates) > 1:
            meta_base["event_log_candidates"] = [item["path"] for item in candidates]
            meta_base["event_log_timing_scores"] = timing_scores
            meta_base["event_log_match_error"] = "Multiple event logs match the FIF session and run."
            return pd.DataFrame(), meta_base

        selected = candidates[0]
        selected_path = selected["path"]
        df = selected["df"].copy()
        onset_rows = selected["onset_rows"].copy()
        selected_score = selected.get("timing_score", np.nan)
        if onset_events is not None and raw is not None:
            if len(onset_rows) != len(onset_events):
                meta_base.update(
                    {
                        "event_log_csv": selected_path,
                        "csv_onsets": int(len(onset_rows)),
                        "fif_onsets": int(len(onset_events)),
                        "event_log_match_error": "Event-log and FIF onset counts differ; labels were not mapped.",
                    }
                )
                return pd.DataFrame(), meta_base
            if not np.isfinite(selected_score):
                selected_score = self._event_log_timing_match_score(
                    onset_rows,
                    onset_events,
                    float(raw.info["sfreq"]),
                )
            # Correct WBL runs are consistently below 0.4 ms mean ISI error.
            # Twenty milliseconds is deliberately conservative while still
            # detecting shifted or unrelated trigger sequences.
            if np.isfinite(selected_score) and float(selected_score) > 0.02:
                meta_base.update(
                    {
                        "event_log_csv": selected_path,
                        "event_log_timing_score": float(selected_score),
                        "event_log_match_error": "Event-log timing does not match FIF triggers; labels were not mapped.",
                    }
                )
                return pd.DataFrame(), meta_base

        if "offset_time" in df.columns:
            off = df[df["event_code"] == 3].copy()
            off["trial"] = pd.to_numeric(off["trial"], errors="coerce")
            off["offset_time"] = pd.to_numeric(off["offset_time"], errors="coerce")
            off = off.dropna(subset=["trial", "offset_time"])
            off_first = off.groupby("trial", as_index=False)["offset_time"].first()
            onset_rows = onset_rows.merge(off_first, on="trial", how="left", suffixes=("", "_off"))
        else:
            onset_rows["offset_time"] = np.nan

        block_id = None
        if "block_id" in onset_rows.columns:
            try:
                bvals = pd.to_numeric(onset_rows["block_id"], errors="coerce").dropna().unique()
                if len(bvals) > 0:
                    block_id = int(bvals[0])
            except Exception:
                block_id = None

        meta = {
            "events_dir": events_dir,
            "run_id": run_id,
            "session_id_from_fif": session_id,
            "event_log_csv": selected_path,
            "block_id": block_id,
            "subject_id": onset_rows["subject_id"].iloc[0] if "subject_id" in onset_rows.columns and len(onset_rows) > 0 else "",
            "session_id": onset_rows["session_id"].iloc[0] if "session_id" in onset_rows.columns and len(onset_rows) > 0 else "",
            "stimlist_version": onset_rows["stimlist_version"].iloc[0] if "stimlist_version" in onset_rows.columns and len(onset_rows) > 0 else "",
            "event_log_timing_score": selected_score,
        }
        return onset_rows, meta


    def _find_fif_stim_events(self, raw=None):
        """Find onset/offset trigger events directly from FIF stim channels."""
        if raw is None:
            raw = getattr(self, "_fif_raw", None)
        if raw is None:
            return np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.int64)
        stim_picks = mne.pick_types(raw.info, stim=True, exclude=[])
        stim_names = [raw.ch_names[int(i)] for i in stim_picks]
        if not stim_names:
            return np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.int64)

        events = mne.find_events(
            raw,
            stim_channel=stim_names,
            shortest_event=1,
            min_duration=0.0,
            verbose='error',
        )
        if events is None or len(events) == 0:
            return np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.int64)

        events = np.asarray(events, dtype=np.int64)
        onset_events = events[events[:, 2] == 2]
        offset_events = events[events[:, 2] == 4]
        return onset_events, offset_events


    def _build_trial_response_table_from_event_logs(
        self,
        onset_rows,
        onset_events,
        offset_events,
        max_report_channels=None,
        raw=None,
        report_picks=None,
    ):
        """Map events_log order to FIF triggers and compute trial response score."""
        if onset_rows is None or onset_rows.empty:
            return pd.DataFrame()
        if onset_events is None or len(onset_events) == 0:
            return pd.DataFrame()
        if raw is None:
            raw = getattr(self, "_fif_raw", None)
        if raw is None:
            return pd.DataFrame()

        sfreq = float(raw.info["sfreq"])
        first_samp = int(getattr(raw, "first_samp", 0))
        n_times = int(raw.n_times)

        n_map = min(int(len(onset_rows)), int(len(onset_events)))
        if n_map <= 0:
            return pd.DataFrame()

        mapped = onset_rows.iloc[:n_map].copy().reset_index(drop=True)
        mapped["trial_index"] = np.arange(1, n_map + 1, dtype=int)
        mapped["onset_sample"] = np.asarray(onset_events[:n_map, 0], dtype=np.int64) - first_samp
        mapped["onset_meg_s"] = np.asarray(mapped["onset_sample"], dtype=float) / sfreq
        mapped["timing_delta_s"] = np.asarray(mapped["onset_meg_s"], dtype=float) - np.asarray(mapped["onset_time"], dtype=float)

        if offset_events is not None and len(offset_events) >= n_map:
            mapped["offset_sample"] = np.asarray(offset_events[:n_map, 0], dtype=np.int64) - first_samp
            mapped["offset_meg_s"] = np.asarray(mapped["offset_sample"], dtype=float) / sfreq
        else:
            mapped["offset_sample"] = np.nan
            mapped["offset_meg_s"] = np.nan
        mapped["duration"] = (
            pd.to_numeric(mapped["offset_meg_s"], errors="coerce")
            - pd.to_numeric(mapped["onset_meg_s"], errors="coerce")
        )

        if report_picks is None:
            report_picks = list(getattr(self, "_fif_picks", []))
        else:
            report_picks = [int(p) for p in report_picks]
        if max_report_channels is not None and int(max_report_channels) > 0 and len(report_picks) > int(max_report_channels):
            report_picks = report_picks[: int(max_report_channels)]
        if not report_picks:
            mapped["trial_score"] = np.nan
            return mapped
        report_channel_names = [raw.ch_names[int(i)] for i in report_picks]

        baseline_start_s = -0.2
        baseline_end_s = 0.0
        response_start_s = 0.05
        response_end_s = 0.35
        eps = 1e-20
        scores = np.full(n_map, np.nan, dtype=float)
        baseline_rms_med = np.full(n_map, np.nan, dtype=float)
        response_rms_med = np.full(n_map, np.nan, dtype=float)
        trial_channel_deltas = [[] for _ in range(n_map)]
        top_response_channel = ["" for _ in range(n_map)]
        top_response_score = [np.nan for _ in range(n_map)]

        for i, onset_rel in enumerate(np.asarray(mapped["onset_sample"], dtype=int)):
            b0 = int(round(onset_rel + baseline_start_s * sfreq))
            b1 = int(round(onset_rel + baseline_end_s * sfreq))
            r0 = int(round(onset_rel + response_start_s * sfreq))
            r1 = int(round(onset_rel + response_end_s * sfreq))
            b0 = max(0, b0)
            b1 = max(0, b1)
            r0 = max(0, r0)
            r1 = max(0, r1)
            b1 = min(n_times, b1)
            r1 = min(n_times, r1)
            if b1 - b0 < 8 or r1 - r0 < 8:
                continue
            try:
                baseline = raw.get_data(
                    picks=report_picks,
                    start=b0,
                    stop=b1,
                    reject_by_annotation='omit',
                )
                response = raw.get_data(
                    picks=report_picks,
                    start=r0,
                    stop=r1,
                    reject_by_annotation='omit',
                )
            except Exception:
                continue
            if baseline is None or response is None:
                continue
            baseline = np.asarray(baseline, dtype=float)
            response = np.asarray(response, dtype=float)
            if baseline.ndim != 2 or response.ndim != 2:
                continue
            if baseline.shape[1] < 8 or response.shape[1] < 8:
                continue

            b_rms = np.sqrt(np.mean(np.square(baseline), axis=1))
            r_rms = np.sqrt(np.mean(np.square(response), axis=1))
            with np.errstate(divide='ignore', invalid='ignore'):
                delta = (r_rms - b_rms) / np.maximum(np.abs(b_rms), eps)
            if delta.ndim != 1 or delta.size == 0:
                continue
            finite_mask = np.isfinite(delta)
            if not np.any(finite_mask):
                continue
            delta_fin = delta[finite_mask]
            ch_fin = [report_channel_names[k] for k, ok in enumerate(finite_mask) if ok]
            pairs = [(str(ch), float(val)) for ch, val in zip(ch_fin, delta_fin)]
            trial_channel_deltas[i] = pairs

            max_idx = int(np.argmax(delta_fin))
            scores[i] = float(delta_fin[max_idx])  # any-channel criterion
            top_response_channel[i] = str(ch_fin[max_idx])
            top_response_score[i] = float(delta_fin[max_idx])
            baseline_rms_med[i] = float(np.median(b_rms[np.isfinite(b_rms)])) if np.any(np.isfinite(b_rms)) else np.nan
            response_rms_med[i] = float(np.median(r_rms[np.isfinite(r_rms)])) if np.any(np.isfinite(r_rms)) else np.nan

        mapped["baseline_rms_median"] = baseline_rms_med
        mapped["response_rms_median"] = response_rms_med
        mapped["trial_score"] = scores
        mapped["top_response_channel"] = top_response_channel
        mapped["top_response_score"] = top_response_score
        mapped["channel_delta_pairs"] = trial_channel_deltas
        mapped["trial_score"] = pd.to_numeric(mapped["trial_score"], errors="coerce")
        return mapped


    def _compute_trial_response_table_uncached(self, fif_filename, max_report_channels=None):
        """Compute one FIF trial table mapped by its matching events_log CSV."""
        fif_path = os.path.abspath(str(fif_filename))
        primary_path = (
            os.path.abspath(str(self._fif_filename))
            if getattr(self, "_fif_filename", None) is not None
            else None
        )
        use_loaded_raw = (
            getattr(self, "_fif_raw", None) is not None
            and primary_path is not None
            and primary_path == fif_path
        )
        if use_loaded_raw:
            raw = self._fif_raw
            report_picks = list(getattr(self, "_fif_picks", []))
        else:
            raw = mne.io.read_raw_fif(fif_path, preload=False, verbose='error')
            report_picks, bad_meg_names = good_meg_picks(raw.info, meg='mag', ref_meg=False)
            if bad_meg_names:
                emit_bad_meg_channel_log(
                    self.c.log,
                    "Stimulus report",
                    fif_path=fif_path,
                    bad_names=bad_meg_names,
                )
            report_picks = report_picks.tolist()
            if len(report_picks) == 0:
                report_picks, bad_meg_names = good_meg_picks(raw.info, meg='grad', ref_meg=False)
                if bad_meg_names:
                    emit_bad_meg_channel_log(
                        self.c.log,
                        "Stimulus report",
                        fif_path=fif_path,
                        bad_names=bad_meg_names,
                    )
                report_picks = report_picks.tolist()

        try:
            if len(report_picks) == 0:
                return pd.DataFrame(), {"source_fif_path": fif_path}, "No MEG channels found."
    
            onset_events, offset_events = self._find_fif_stim_events(raw=raw)
            if len(onset_events) == 0:
                return pd.DataFrame(), {"source_fif_path": fif_path}, "No stim onset events found in FIF."
    
            onset_rows, meta = self._load_event_log_onsets_for_fif(
                fif_path,
                onset_events=onset_events,
                raw=raw,
            )
            if onset_rows.empty:
                reason = str(meta.get("event_log_match_error", "") or "No events_log onset rows found.")
                return pd.DataFrame(), meta, reason
    
            trial_df = self._build_trial_response_table_from_event_logs(
                onset_rows=onset_rows,
                onset_events=onset_events,
                offset_events=offset_events,
                max_report_channels=max_report_channels,
                raw=raw,
                report_picks=report_picks,
            )
            if trial_df.empty:
                return pd.DataFrame(), meta, "Could not map events_log onsets to FIF events."
    
            trial_df = self._annotate_trial_df_with_stimulus_groups(trial_df, fif_path)
            trial_df = trial_df.copy()
            ica_meta = self._read_ica_metadata(fif_path)
            original_fif_path = (
                ica_meta.get("source_fif_path")
                if isinstance(ica_meta, dict) and ica_meta.get("source_fif_path")
                else fif_path
            )
            trial_df["source_fif"] = os.path.basename(fif_path)
            trial_df["source_fif_path"] = fif_path
            trial_df["source_original_fif_path"] = original_fif_path
            trial_df["source_ica_applied"] = bool(ica_meta.get("applied")) if isinstance(ica_meta, dict) else False
            trial_df["run_id"] = meta.get("run_id")
            if "block_id" not in trial_df.columns and meta.get("block_id") is not None:
                trial_df["block_id"] = meta.get("block_id")
    
            meta = dict(meta)
            meta["source_fif"] = os.path.basename(fif_path)
            meta["source_fif_path"] = fif_path
            meta["source_original_fif_path"] = original_fif_path
            meta["source_ica_applied"] = bool(ica_meta.get("applied")) if isinstance(ica_meta, dict) else False
            if isinstance(ica_meta, dict) and ica_meta:
                meta["ica_status"] = str(ica_meta.get("status", ""))
                meta["ica_excluded_components"] = list(ica_meta.get("excluded_components", []))
            meta["mapped_trials"] = int(len(trial_df))
            meta["csv_onsets"] = int(len(onset_rows))
            meta["fif_onsets"] = int(len(onset_events))
            return trial_df, meta, None
        finally:
            if not use_loaded_raw:
                raw.close()


    @staticmethod
    def _file_identity_for_metadata(path):
        if not path:
            return None
        path_abs = os.path.abspath(str(path))
        try:
            stat = os.stat(path_abs)
            return {
                "path": path_abs,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        except Exception:
            return {"path": path_abs, "size": None, "mtime_ns": None}


    @staticmethod
    def _canonicalize_meg_stimulus_label(stimulus_name):
        if stimulus_name is None:
            return ""
        stim = os.path.basename(str(stimulus_name).strip())
        stem, _ = os.path.splitext(stim)
        m = re.match(r"^(.*?)(?:_\d+_\d+)$", stem)
        if m and m.group(1):
            return m.group(1)
        m2 = re.match(r"^(.*?)(?:_\d+)$", stem)
        if m2 and m2.group(1):
            return m2.group(1)
        return stem


    def _stimulus_group_mapping_for_fif(self, fif_filename):
        """Load stimulus->category mapping from `../stimulus_with_groups.csv`."""
        fif_path = os.path.abspath(str(fif_filename or getattr(self, "_fif_filename", "") or ""))
        if not fif_path:
            return {}
        cache = getattr(self, "_meg_stimulus_group_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._meg_stimulus_group_cache = cache
        csv_path = os.path.join(os.path.dirname(os.path.dirname(fif_path)), "stimulus_with_groups.csv")
        cache_key = os.path.abspath(csv_path)
        if cache_key in cache:
            return dict(cache[cache_key])
        if not os.path.isfile(csv_path):
            cache[cache_key] = {}
            return {}
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            cache[cache_key] = {}
            return {}
        cols = {str(c).strip().lower(): c for c in df.columns}
        stim_col = cols.get("stimulus")
        group_col = cols.get("stimulus_group")
        if stim_col is None or group_col is None:
            cache[cache_key] = {}
            return {}
        mapping = {}
        for _, row in df.iterrows():
            stim = row.get(stim_col)
            group = row.get(group_col)
            if pd.isna(stim) or pd.isna(group):
                continue
            canonical = self._canonicalize_meg_stimulus_label(stim)
            group_name = str(group).strip()
            if canonical and group_name:
                mapping[canonical] = group_name
        cache[cache_key] = dict(mapping)
        return mapping


    def _annotate_trial_df_with_stimulus_groups(self, trial_df, fif_filename):
        """Add canonical stimulus and category columns to trial data if possible."""
        if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
            return trial_df
        df = trial_df.copy()
        if "stimulus" in df.columns:
            if "canonical_stimulus" not in df.columns:
                df["canonical_stimulus"] = df["stimulus"].map(self._canonicalize_meg_stimulus_label)
            else:
                df["canonical_stimulus"] = df["canonical_stimulus"].fillna(
                    df["stimulus"].map(self._canonicalize_meg_stimulus_label)
                )
        mapping = self._stimulus_group_mapping_for_fif(fif_filename)
        if mapping and "canonical_stimulus" in df.columns:
            df["stimulus_group"] = df["canonical_stimulus"].map(
                lambda x: mapping.get(str(x).strip(), "unmapped") if str(x).strip() else "unmapped"
            ).fillna("unmapped")
        elif "stimulus_group" not in df.columns:
            df["stimulus_group"] = ""
        return df


    @staticmethod
    def _is_tsss_fif_path(fif_path):
        """Return True for FIFs loaded from tSSS/SSS derivative folders."""
        try:
            parent = os.path.basename(os.path.dirname(os.path.abspath(str(fif_path)))).lower()
        except Exception:
            return False
        return parent in {"tsss", "sss", "maxfilter", "maxfiltered"} or "tsss" in parent


    def _ica_output_path_for_tsss_fif(self, fif_path):
        """Return the cached ICA-cleaned FIF path for a tSSS FIF."""
        if not fif_path:
            return None
        fif_abs = os.path.abspath(str(fif_path))
        fif_dir = os.path.dirname(fif_abs)
        parent_name = os.path.basename(fif_dir).lower()
        if parent_name == "ica" or not self._is_tsss_fif_path(fif_abs):
            return None
        experiment_root = os.path.dirname(fif_dir)
        if not experiment_root:
            return None
        return os.path.join(experiment_root, "ica", os.path.basename(fif_abs))


    @staticmethod
    def _ica_metadata_path(cleaned_fif_path):
        if not cleaned_fif_path:
            return None
        return f"{os.path.abspath(str(cleaned_fif_path))}.ica_meta.json"


    def _read_ica_metadata(self, cleaned_fif_path):
        meta_path = self._ica_metadata_path(cleaned_fif_path)
        if not meta_path or not os.path.isfile(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta if isinstance(meta, dict) else {}
        except Exception:
            return {}


    def _ica_derivative_is_current(self, source_fif_path, cleaned_fif_path):
        """Return True when an ICA-cleaned FIF is reusable for the source FIF."""
        if not cleaned_fif_path or not os.path.isfile(cleaned_fif_path):
            return False
        source_identity = self._file_identity_for_metadata(source_fif_path)
        meta = self._read_ica_metadata(cleaned_fif_path)
        if meta.get("input_identity") == source_identity:
            return True
        try:
            return os.path.getmtime(cleaned_fif_path) >= os.path.getmtime(source_fif_path)
        except Exception:
            return False


    @staticmethod
    def _first_ica_aux_channel_name(raw, kind):
        """Find a likely EOG/ECG channel name for automatic ICA scoring."""
        try:
            if kind == "eog":
                picks = mne.pick_types(raw.info, meg=False, eog=True, exclude=[])
                patterns = ("eog", "veog", "heog", "eeg061", "eeg062")
            else:
                picks = mne.pick_types(raw.info, meg=False, ecg=True, exclude=[])
                patterns = ("ecg", "ekg")
            if len(picks) > 0:
                return raw.ch_names[int(picks[0])]
        except Exception:
            patterns = ("eog", "veog", "heog", "eeg061", "eeg062") if kind == "eog" else ("ecg", "ekg")

        for name in getattr(raw, "ch_names", []):
            low = str(name).lower()
            if any(pattern in low for pattern in patterns):
                return str(name)
        return None


    @staticmethod
    def _limited_ica_component_list(components, limit=3):
        out = []
        try:
            iterable = [] if components is None else list(components)
        except Exception:
            iterable = []
        for value in iterable:
            try:
                idx = int(value)
            except Exception:
                continue
            if idx not in out:
                out.append(idx)
            if len(out) >= int(limit):
                break
        return out


    def _write_ica_metadata(self, cleaned_fif_path, meta):
        meta_path = self._ica_metadata_path(cleaned_fif_path)
        if not meta_path:
            return
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except Exception:
            pass


    def _run_ica_cleaning_for_tsss_fif(self, source_fif_path, cleaned_fif_path):
        """Fit/apply ICA for one tSSS FIF and save a cached cleaned derivative."""
        source_fif_path = os.path.abspath(str(source_fif_path))
        cleaned_fif_path = os.path.abspath(str(cleaned_fif_path))
        os.makedirs(os.path.dirname(cleaned_fif_path), exist_ok=True)

        self.c.log.emit(f"MEG preprocessing: fitting ICA for {os.path.basename(source_fif_path)} ...")
        raw = mne.io.read_raw_fif(source_fif_path, preload=True, verbose="error")
        meg_picks, bad_meg_names = good_meg_picks(raw.info, meg=True, ref_meg=False)
        if bad_meg_names:
            emit_bad_meg_channel_log(
                self.c.log,
                "MEG ICA preprocessing",
                fif_path=source_fif_path,
                bad_names=bad_meg_names,
            )
        n_meg = int(len(meg_picks))

        rank_info = {}
        n_components = 0
        excluded = []
        eog_components = []
        ecg_components = []
        eog_channel = None
        ecg_channel = None
        status = "cleaned"
        ica_solution_path = None

        if n_meg >= 2:
            raw_for_ica = raw.copy()
            raw_for_ica.filter(
                l_freq=1.0,
                h_freq=None,
                picks=meg_picks,
                n_jobs=1,
                verbose="error",
            )
            try:
                rank_info = mne.compute_rank(raw_for_ica, rank="info", verbose="error")
            except Exception:
                rank_info = {}

            meg_rank = rank_info.get("meg")
            if meg_rank is None:
                rank_vals = []
                for key in ("mag", "grad"):
                    try:
                        rank_vals.append(int(rank_info[key]))
                    except Exception:
                        pass
                meg_rank = int(sum(rank_vals)) if rank_vals else None

            max_rank_components = n_meg - 1
            try:
                if meg_rank is not None:
                    max_rank_components = min(max_rank_components, max(1, int(meg_rank) - 1))
            except Exception:
                pass
            n_components = int(min(40, max_rank_components))

            if n_components >= 2:
                ica = mne.preprocessing.ICA(
                    n_components=n_components,
                    method="fastica",
                    random_state=97,
                    max_iter="auto",
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    ica.fit(
                        raw_for_ica,
                        picks=meg_picks,
                        decim=3,
                        reject_by_annotation=True,
                        verbose="error",
                    )

                eog_channel = self._first_ica_aux_channel_name(raw, "eog")
                if eog_channel:
                    try:
                        eog_found, _eog_scores = ica.find_bads_eog(
                            raw_for_ica,
                            ch_name=eog_channel,
                            verbose="error",
                        )
                        eog_components = self._limited_ica_component_list(eog_found, limit=3)
                    except Exception:
                        eog_components = []

                ecg_channel = self._first_ica_aux_channel_name(raw, "ecg")
                try:
                    ecg_kwargs = {"ch_name": ecg_channel} if ecg_channel else {"ch_name": None}
                    ecg_found, _ecg_scores = ica.find_bads_ecg(
                        raw_for_ica,
                        method="correlation",
                        verbose="error",
                        **ecg_kwargs,
                    )
                    ecg_components = self._limited_ica_component_list(ecg_found, limit=3)
                except Exception:
                    ecg_components = []

                for idx in eog_components + ecg_components:
                    if idx not in excluded:
                        excluded.append(int(idx))
                    if len(excluded) >= 6:
                        break

                if excluded:
                    ica.apply(raw, exclude=excluded, verbose="error")
                else:
                    status = "no_artifact_components_found"

                ica_solution_path = f"{os.path.splitext(cleaned_fif_path)[0]}-ica.fif"
                try:
                    ica.save(ica_solution_path, overwrite=True, verbose="error")
                except Exception:
                    ica_solution_path = None
            else:
                status = "skipped_low_rank"
        else:
            status = "skipped_no_meg_channels"

        raw.save(cleaned_fif_path, overwrite=True, verbose="error")
        meta = {
            "kind": "mas_meg_ica_cleaned_fif",
            "version": 1,
            "source_fif_path": source_fif_path,
            "cleaned_fif_path": cleaned_fif_path,
            "input_identity": self._file_identity_for_metadata(source_fif_path),
            "output_identity": self._file_identity_for_metadata(cleaned_fif_path),
            "ica_solution_path": ica_solution_path or "",
            "n_meg_channels": n_meg,
            "excluded_bad_meg_channels": list(bad_meg_names),
            "rank_info": {str(k): int(v) for k, v in rank_info.items()} if isinstance(rank_info, dict) else {},
            "n_components": int(n_components),
            "eog_channel": eog_channel or "",
            "ecg_channel": ecg_channel or "",
            "eog_components": [int(i) for i in eog_components],
            "ecg_components": [int(i) for i in ecg_components],
            "excluded_components": [int(i) for i in excluded],
            "applied": bool(excluded),
            "status": status,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_ica_metadata(cleaned_fif_path, meta)
        self.c.log.emit(
            "MEG preprocessing: ICA "
            f"{status.replace('_', ' ')} for {os.path.basename(source_fif_path)} "
            f"(excluded={len(excluded)}, output={cleaned_fif_path})."
        )
        return cleaned_fif_path


    def _resolve_ica_cleaned_fif(self, fif_path):
        """Return the analysis FIF path, creating/reusing ICA output for tSSS inputs."""
        if not fif_path:
            return fif_path
        source_fif_path = os.path.abspath(str(fif_path))
        cleaned_fif_path = self._ica_output_path_for_tsss_fif(source_fif_path)
        if not cleaned_fif_path:
            return source_fif_path
        if self._ica_derivative_is_current(source_fif_path, cleaned_fif_path):
            self.c.log.emit(
                "MEG preprocessing: using cached ICA-cleaned FIF "
                f"{os.path.basename(cleaned_fif_path)}."
            )
            return cleaned_fif_path
        try:
            return self._run_ica_cleaning_for_tsss_fif(source_fif_path, cleaned_fif_path)
        except Exception as exc:
            self.c.log.emit(
                f"MEG preprocessing: ICA failed for {os.path.basename(source_fif_path)}; "
                f"using tSSS file directly ({exc})."
            )
            print(traceback.format_exc())
            return source_fif_path


    def _resolve_ica_cleaned_fif_files(self, fif_files):
        """Resolve a list of FIF paths through the tSSS->ICA cache."""
        resolved = []
        for path in fif_files or []:
            if not path:
                continue
            resolved_path = self._resolve_ica_cleaned_fif(path)
            resolved_abs = os.path.abspath(str(resolved_path))
            if resolved_abs not in resolved:
                resolved.append(resolved_abs)
        return resolved



    @staticmethod
    def trial_table_path(fif_filename):
        """A unique cache beside each FIF, including ICA derivatives."""
        return Path(str(Path(fif_filename).resolve()) + '.trial_table.csv')

    def _trial_cache_signature(self, fif_filename, max_report_channels):
        path = Path(fif_filename).resolve()
        # Include all event-log candidates: adding another matching log must
        # invalidate the cache and run ambiguity detection again.
        inputs = {}
        for directory in self._candidate_events_log_dirs(path):
            folder = Path(directory)
            if folder.is_dir():
                for item in sorted(folder.glob('*.csv')):
                    inputs[str(item.relative_to(path.parent.parent))] = hashlib.sha256(item.read_bytes()).hexdigest()
                break
        group_map = path.parent.parent / 'stimulus_with_groups.csv'
        if group_map.is_file():
            inputs['stimulus_with_groups.csv'] = hashlib.sha256(group_map.read_bytes()).hexdigest()
        # Split FIF parts must also invalidate derived trial response scores.
        with mne.io.read_raw_fif(path, preload=False, verbose='error') as raw:
            files = [Path(p) for p in raw.filenames if p is not None]
            fif_parts = [{"name": p.name, "size": p.stat().st_size,
                          "mtime_ns": p.stat().st_mtime_ns} for p in files]
        return {'version': 1, 'fif_parts': fif_parts, 'inputs': inputs,
                'max_report_channels': max_report_channels}

    @staticmethod
    def _atomic_write_text(path, text):
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=path.parent, delete=False) as stream:
                temporary = stream.name
                stream.write(text)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _compute_trial_response_table_for_fif(self, fif_filename, max_report_channels=None):
        """Reuse a validated on-disk table, otherwise build and save it.

        CSV nested channel scores use JSON, never pickle. Metadata and a CSV
        checksum detect interrupted writes or mismatched concurrent writers.
        """
        fif_path = str(Path(fif_filename).resolve())
        cache_path = self.trial_table_path(fif_path)
        meta_path = Path(str(cache_path) + '.meta.json')
        signature = self._trial_cache_signature(fif_path, max_report_channels)
        if not getattr(self, 'rebuild_trial_table', False) and cache_path.is_file() and meta_path.is_file():
            try:
                saved = json.loads(meta_path.read_text(encoding='utf-8'))
                csv_bytes = cache_path.read_bytes()
                if saved['signature'] == signature and saved['csv_sha256'] == hashlib.sha256(csv_bytes).hexdigest():
                    from io import BytesIO
                    df = pd.read_csv(BytesIO(csv_bytes))
                    if df.empty or not {'onset_sample', 'stimulus', 'source_fif_path'}.issubset(df.columns):
                        raise ValueError('Incomplete trial cache')
                    if 'channel_delta_pairs' in df:
                        df['channel_delta_pairs'] = df['channel_delta_pairs'].map(json.loads)
                    # Rebase the single-run paths after moving the folder to RCC.
                    df['source_fif_path'] = fif_path
                    df['source_fif'] = Path(fif_path).name
                    ica_meta = self._read_ica_metadata(fif_path)
                    original = ica_meta.get('source_fif_path') or fif_path
                    df['source_original_fif_path'] = original
                    meta = saved['metadata']
                    meta.update(source_fif_path=fif_path, source_fif=Path(fif_path).name,
                                source_original_fif_path=original, trial_table_path=str(cache_path))
                    self.c.log.emit(f'MEG trial table: reusing {cache_path}')
                    return df, meta, None
            except (ValueError, KeyError, TypeError, OSError):
                pass  # Rebuild unreadable, stale, or incomplete caches.
        # A long-lived GUI can retain an obsolete category mapping in memory.
        self._meg_stimulus_group_cache = {}
        df, meta, error = self._compute_trial_response_table_uncached(fif_path, max_report_channels)
        if error or df.empty:
            return df, meta, error
        stored = df.copy()
        if 'channel_delta_pairs' in stored:
            stored['channel_delta_pairs'] = stored['channel_delta_pairs'].map(json.dumps)
        csv_text = stored.to_csv(index=False)
        metadata = {'signature': signature, 'metadata': meta,
                    'csv_sha256': hashlib.sha256(csv_text.encode('utf-8')).hexdigest()}
        # Convert numpy scalars in event-log metadata to JSON primitives.
        json_text = json.dumps(metadata, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
        try:
            self._atomic_write_text(cache_path, csv_text)
            self._atomic_write_text(meta_path, json_text)
        except OSError as exc:
            raise OSError(f'Cannot save trial table beside FIF: {cache_path}. Ensure this data directory is writable.') from exc
        meta['trial_table_path'] = str(cache_path)
        self.c.log.emit(f'MEG trial table: created {cache_path}')
        return df, meta, None


class HeadlessFIFProcessor(FIFProcessingMixin):
    """Preprocess FIFs and build trial tables without importing Qt or viewers."""

    def __init__(self, log=print, *, rebuild_trial_table=False, ica='auto'):
        self.c = SimpleNamespace(log=SimpleNamespace(emit=log))
        self.rebuild_trial_table = rebuild_trial_table
        self.ica = ica

    def prepare(self, fif_filename):
        path = Path(fif_filename).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'FIF not found: {path}')
        analysis_path = str(path)
        if self.ica == 'auto':
            target = self._ica_output_path_for_tsss_fif(path)
            if target:
                # Batch jobs must report preprocessing failures, not silently
                # continue with uncleaned data after a failed ICA fit.
                if self._ica_derivative_is_current(str(path), target):
                    analysis_path = target
                    self.c.log.emit(f'MEG preprocessing: reusing ICA-cleaned FIF {target}')
                else:
                    analysis_path = self._run_ica_cleaning_for_tsss_fif(str(path), target)
        df, meta, error = self._compute_trial_response_table_for_fif(analysis_path)
        if error:
            raise ValueError(f'{analysis_path}: {error}')
        return df, meta
