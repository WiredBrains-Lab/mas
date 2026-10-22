"""MEG analysis panel with time-, frequency-, sensor-, and source-domain views."""

import json
import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets, QtGui

from core.concept_decoding_signal import (
    CONCEPT_PREPROCESSING_DEFAULTS,
    concept_epoch_cache_directory,
    concept_qc_summary_table,
    PARCEL_TRIAL_FREQUENCY_BANDS,
    parcel_trial_frequency_band_settings,
)
from core.concept_decoding_subsets import exhaustive_fit_count, backward_fit_count
from core.source_parcellation import (
    HCP_MMP_PARCS,
    anatomical_parcel_row_order,
    aseg_name_from_path,
    is_surface_stc,
    is_volume_stc,
    normalize_parcellation_key,
    parcel_trial_contrast_map,
    parcel_trial_summary_maps,
    rank_parcel_contrast_features,
    read_surface_labels,
    subject_aseg_path,
    surface_label_time_courses,
    temporal_bin_parcel_time_courses,
    top_label_indices,
    volume_label_time_courses,
    filter_parcels,
)
from core.stimulus_response_report import StimulusResponseReport
from usercontrols.checkable_treecombobox import CheckableTreeComboBox
from usercontrols.pandasmodel import PandasModel


MEG_EVENTS_CHANNEL_LABEL = "events"


class _LayoutHost:
    """Adapter so classes expecting `addWidget` can target a layout."""

    def __init__(self, layout):
        self._layout = layout

    def addWidget(self, widget):
        self._layout.addWidget(widget)


class _NullSignal:
    def emit(self, *_args, **_kwargs):
        return None


class _NullContext:
    def __init__(self):
        self.sig_view_acq_channels_selection = _NullSignal()


class _ConceptDecodeWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    log = QtCore.Signal(str)
    training_metric = QtCore.Signal(object)
    finished = QtCore.Signal(object, str)
    failed = QtCore.Signal(str)

    def __init__(self, callback, options):
        super().__init__()
        self._callback = callback
        self._options = dict(options or {})

    def run(self):
        try:
            options = dict(self._options)
            options["progress_callback"] = lambda message: self.progress.emit(str(message))
            options["log_callback"] = lambda message: self.log.emit(str(message))
            options["training_metric_callback"] = lambda metric: self.training_metric.emit(
                dict(metric or {})
            )
            options["cancel_callback"] = lambda: QtCore.QThread.currentThread().isInterruptionRequested()
            self.progress.emit("extracting MEG features...")
            result = self._callback(options)
            if isinstance(result, tuple) and len(result) >= 2:
                self.finished.emit(result[0], str(result[1]))
            else:
                self.failed.emit("Invalid concept decoding result.")
        except Exception as exc:
            self.failed.emit(str(exc))


class _ParcelHeatmapWorker(QtCore.QObject):
    """Run trial-level source extraction without blocking the Qt event loop."""

    progress = QtCore.Signal(str)
    finished = QtCore.Signal(object, str)
    failed = QtCore.Signal(str)

    def __init__(self, callback, options):
        super().__init__()
        self._callback = callback
        self._options = dict(options or {})

    def run(self):
        try:
            options = dict(self._options)
            comparison_stimulus = str(
                options.pop("comparison_concept_label", "") or ""
            ).strip()
            options["progress_callback"] = lambda message: self.progress.emit(str(message))
            self.progress.emit("extracting trial-level source parcels...")
            result = self._callback(options)
            if not isinstance(result, tuple) or len(result) < 2:
                self.failed.emit("Invalid parcel heatmap result.")
                return
            primary_result, primary_message = result[0], str(result[1])
            if not isinstance(primary_result, dict) or not comparison_stimulus:
                self.finished.emit(primary_result, primary_message)
                return

            primary_stimulus = str(options.get("selected_concept_label", "stimulus") or "stimulus")
            comparison_options = dict(options)
            comparison_options["selected_concept_label"] = comparison_stimulus
            comparison_options["progress_callback"] = lambda message: self.progress.emit(
                f"{comparison_stimulus}: {message}"
            )
            self.progress.emit(
                f"extracting comparison stimulus '{comparison_stimulus}'..."
            )
            comparison = self._callback(comparison_options)
            if not isinstance(comparison, tuple) or len(comparison) < 2:
                self.failed.emit("Invalid comparison parcel heatmap result.")
                return
            comparison_result, comparison_message = comparison[0], str(comparison[1])
            if not isinstance(comparison_result, dict):
                self.failed.emit(
                    comparison_message
                    or f"Could not extract comparison stimulus '{comparison_stimulus}'."
                )
                return
            primary_result = dict(primary_result)
            primary_result["comparison_result"] = comparison_result
            primary_result["comparison_stimulus"] = comparison_stimulus
            comparison_count = int(
                np.asarray(comparison_result.get("data", [])).shape[0]
            )
            self.finished.emit(
                primary_result,
                f"{primary_message} Contrast: {primary_stimulus} minus "
                f"{comparison_stimulus} ({comparison_count} "
                "comparison presentation(s)).",
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class MEGAnalysisPanel:
    """Container widget for MEG post-load analysis views."""

    def __init__(self, report_view, c=None):
        self.report_view = report_view
        self.c = c
        self._trial_df = pd.DataFrame()
        self._metadata = {}
        self._raw = None
        self._epochs = None
        self._topomap_epochs_by_path = {}
        self._topomap_run_meta_by_path = {}
        self._topomap_run_order = []
        self._evokeds = {}
        self._primary_event = None
        self._primary_evoked = None
        self._fif_filename = None
        self._source_stc = None
        self._source_src = None
        self._source_status_text = ""
        self._source_compute_callback = None
        self._source_display_callback = None
        self._source_brain_view_callback = None
        self._source_subjects_dir = ""
        self._source_subject = ""
        self._source_time_index = None
        self._source_time_marker_line = None
        self._source_time_marker_canvas = None
        self._source_plot_has_brain_image = False
        self._source_plot_morph_cache = {}
        self._semantic_preview_callback = None
        self._semantic_decode_callback = None
        self._concept_decode_callback = None
        self._memory_telemetry_provider = None
        self._semantic_preview_result = None
        self._semantic_train_result = None
        self._concept_decode_result = None
        self._parcel_heatmap_result = None
        self._concept_decode_thread = None
        self._concept_decode_worker = None
        self._parcel_heatmap_callback = None
        self._parcel_heatmap_thread = None
        self._parcel_heatmap_worker = None
        self._concept_decode_last_options = {}
        self._concept_training_history = []
        self._concept_training_series = None
        self._concept_training_figure = None
        self._concept_training_axis = None
        self._concept_training_lines = {}
        self._concept_outputs_base_dir = ""
        self._semantic_preview_status_text = ""
        self._semantic_train_status_text = ""
        self._concept_decode_status_text = ""
        self._concept_available_parcels = []
        self.semantic_metrics_model = None
        self.semantic_features_model = None
        self.semantic_trials_model = None
        self.concept_summary_model = None
        self.concept_metrics_model = None
        self.concept_predictions_model = None
        self.concept_features_model = None
        self.concept_split_model = None
        self.concept_permutation_model = None
        self.concept_templates_model = None
        self.concept_architecture_model = None
        self.concept_trial_qc_model = None
        self.concept_qc_summary_model = None
        self.parcel_ranking_model = None
        self._semantic_parcellation_context = _NullContext()
        self._source_run_context = _NullContext()
        self._plot_state = {}
        self._init_layout()

    def _init_layout(self):
        self.widget = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.widget.setLayout(root)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        self.status_label = QtWidgets.QLabel("No MEG analysis loaded.")
        status_row.addWidget(self.status_label, 1)
        self.lbl_memory_telemetry = QtWidgets.QLabel(
            "RAM: -- | epochs: 0 B / 0 runs | decoder cache: 0 B / 0 | parcel cache: 0 B / 0"
        )
        self.lbl_memory_telemetry.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_memory_telemetry.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_memory_telemetry.setToolTip(
            "Live process resident memory and estimated MEG epoch/decoder-cache allocations. "
            "Shared epoch objects are counted once."
        )
        status_row.addWidget(self.lbl_memory_telemetry)
        root.addLayout(status_row)

        self._init_shared_preprocessing_panel(root)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._on_analysis_tab_changed)
        root.addWidget(self.tabs)

        self._init_stimulus_report_tab()
        self._init_epoch_image_tab()
        self._init_evoked_tab()
        self._init_topomap_tab()
        self._init_frequency_domain_tab()
        self._init_concept_decoding_tab()
        self._init_source_tc_tab()
        self._init_parcel_trial_heatmap_tab()

        self._memory_telemetry_timer = QtCore.QTimer(self.widget)
        self._memory_telemetry_timer.setInterval(2000)
        self._memory_telemetry_timer.timeout.connect(self._refresh_memory_telemetry)
        self._memory_telemetry_timer.start()
        self._refresh_memory_telemetry()

        if self.report_view is not None:
            self.report_view.addWidget(self.widget)

    def _init_shared_preprocessing_panel(self, root_layout):
        """Create the persistent preprocessing controls used by trial analyses."""
        self.shared_preprocessing_frame = QtWidgets.QFrame()
        self.shared_preprocessing_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.shared_preprocessing_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Maximum,
        )
        frame_layout = QtWidgets.QVBoxLayout()
        frame_layout.setContentsMargins(6, 4, 6, 6)
        frame_layout.setSpacing(4)
        self.shared_preprocessing_frame.setLayout(frame_layout)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        title = QtWidgets.QLabel("Shared preprocessing")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        self.lbl_shared_preprocessing_summary = QtWidgets.QLabel()
        self.lbl_shared_preprocessing_summary.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.lbl_shared_preprocessing_summary.setToolTip(
            "Current preprocessing profile used by trial-derived analyses."
        )
        header_layout.addWidget(self.lbl_shared_preprocessing_summary, 1)
        self.btn_shared_preprocessing_toggle = QtWidgets.QPushButton("Collapse")
        self.btn_shared_preprocessing_toggle.setCheckable(True)
        self.btn_shared_preprocessing_toggle.setChecked(True)
        self.btn_shared_preprocessing_toggle.setToolTip(
            "Collapse the shared controls while keeping their active settings visible."
        )
        self.btn_shared_preprocessing_toggle.toggled.connect(
            self._set_shared_preprocessing_expanded
        )
        header_layout.addWidget(self.btn_shared_preprocessing_toggle)
        frame_layout.addLayout(header_layout)

        self.shared_preprocessing_body = QtWidgets.QWidget()
        body_layout = QtWidgets.QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        self.shared_preprocessing_body.setLayout(body_layout)

        filter_group = QtWidgets.QGroupBox("Filtering and sampling")
        filter_layout = QtWidgets.QGridLayout()
        filter_layout.setContentsMargins(6, 6, 6, 6)
        filter_layout.setHorizontalSpacing(6)
        filter_layout.setVerticalSpacing(4)
        filter_group.setLayout(filter_layout)

        filter_layout.addWidget(QtWidgets.QLabel("Bandpass low"), 0, 0)
        self.spb_shared_bandpass_low = QtWidgets.QDoubleSpinBox()
        self.spb_shared_bandpass_low.setDecimals(2)
        self.spb_shared_bandpass_low.setRange(0.0, 2000.0)
        self.spb_shared_bandpass_low.setSingleStep(0.1)
        self.spb_shared_bandpass_low.setSuffix(" Hz")
        self.spb_shared_bandpass_low.setValue(
            CONCEPT_PREPROCESSING_DEFAULTS["bandpass_low_hz"]
        )
        self.spb_shared_bandpass_low.setToolTip(
            "Shared high-pass cutoff used by supported analysis views. Use 0 to disable "
            "the high-pass cutoff."
        )
        filter_layout.addWidget(self.spb_shared_bandpass_low, 0, 1)

        filter_layout.addWidget(QtWidgets.QLabel("Bandpass high"), 0, 2)
        self.spb_shared_bandpass_high = QtWidgets.QDoubleSpinBox()
        self.spb_shared_bandpass_high.setDecimals(1)
        self.spb_shared_bandpass_high.setRange(0.0, 5000.0)
        self.spb_shared_bandpass_high.setSingleStep(1.0)
        self.spb_shared_bandpass_high.setSuffix(" Hz")
        self.spb_shared_bandpass_high.setValue(
            CONCEPT_PREPROCESSING_DEFAULTS["bandpass_high_hz"]
        )
        self.spb_shared_bandpass_high.setToolTip(
            "Shared low-pass cutoff used by supported analysis views. It must be below "
            "the effective Nyquist frequency."
        )
        filter_layout.addWidget(self.spb_shared_bandpass_high, 0, 3)

        self.chk_shared_downsample = QtWidgets.QCheckBox("Downsample data")
        self.chk_shared_downsample.setChecked(
            CONCEPT_PREPROCESSING_DEFAULTS["mne_resample_enabled"]
        )
        self.chk_shared_downsample.setToolTip(
            "Resample each filtered recording before epoching and feature or source extraction."
        )
        filter_layout.addWidget(self.chk_shared_downsample, 1, 0, 1, 2)
        filter_layout.addWidget(QtWidgets.QLabel("Target rate"), 1, 2)
        self.spb_shared_downsample_sfreq = QtWidgets.QDoubleSpinBox()
        self.spb_shared_downsample_sfreq.setDecimals(1)
        self.spb_shared_downsample_sfreq.setRange(1.0, 5000.0)
        self.spb_shared_downsample_sfreq.setSingleStep(10.0)
        self.spb_shared_downsample_sfreq.setSuffix(" Hz")
        self.spb_shared_downsample_sfreq.setValue(
            CONCEPT_PREPROCESSING_DEFAULTS["resample_sfreq"]
        )
        self.spb_shared_downsample_sfreq.setToolTip(
            "Target sampling rate. It cannot exceed any selected run's native rate."
        )
        filter_layout.addWidget(self.spb_shared_downsample_sfreq, 1, 3)
        self.chk_shared_disk_cache = QtWidgets.QCheckBox("Reuse saved decoding inputs")
        self.chk_shared_disk_cache.setChecked(True)
        self.chk_shared_disk_cache.setToolTip(
            "Save each filtered/downsampled recording for reuse across trials, windows and app restarts. "
            "Changed filters, sampling rates or source files create a new recording entry. "
            "Prepared epochs are cached separately. Events are mapped to the new sample grid. "
            "Old entries are removed automatically above 10 GiB.\n"
            + concept_epoch_cache_directory()
        )
        filter_layout.addWidget(self.chk_shared_disk_cache, 2, 0, 1, 4)
        body_layout.addWidget(filter_group, 1)

        rejection_group = QtWidgets.QGroupBox("Trial rejection")
        rejection_layout = QtWidgets.QGridLayout()
        rejection_layout.setContentsMargins(6, 6, 6, 6)
        rejection_layout.setHorizontalSpacing(6)
        rejection_layout.setVerticalSpacing(4)
        rejection_group.setLayout(rejection_layout)

        self.chk_shared_mne_reject = QtWidgets.QCheckBox("MNE epoch reject")
        self.chk_shared_mne_reject.setChecked(
            CONCEPT_PREPROCESSING_DEFAULTS["mne_epoch_reject_enabled"]
        )
        self.chk_shared_mne_reject.setToolTip(
            "Reject high-amplitude epochs before feature or source extraction."
        )
        rejection_layout.addWidget(self.chk_shared_mne_reject, 0, 0, 1, 2)
        rejection_layout.addWidget(QtWidgets.QLabel("Mag"), 0, 2)
        self.spb_shared_reject_mag = QtWidgets.QDoubleSpinBox()
        self.spb_shared_reject_mag.setDecimals(0)
        self.spb_shared_reject_mag.setRange(0.0, 1000000.0)
        self.spb_shared_reject_mag.setSingleStep(500.0)
        self.spb_shared_reject_mag.setSuffix(" fT")
        self.spb_shared_reject_mag.setValue(
            CONCEPT_PREPROCESSING_DEFAULTS["reject_mag_ft"]
        )
        self.spb_shared_reject_mag.setToolTip(
            "Magnetometer peak-to-peak rejection threshold. Use 0 to disable."
        )
        rejection_layout.addWidget(self.spb_shared_reject_mag, 0, 3)

        self.chk_shared_exclude_outliers = QtWidgets.QCheckBox("Exclude QC outliers")
        self.chk_shared_exclude_outliers.setChecked(
            CONCEPT_PREPROCESSING_DEFAULTS["exclude_trial_outliers"]
        )
        self.chk_shared_exclude_outliers.setToolTip(
            "Exclude robust trial-level signal outliers after epoch rejection."
        )
        rejection_layout.addWidget(self.chk_shared_exclude_outliers, 1, 0, 1, 2)
        rejection_layout.addWidget(QtWidgets.QLabel("Grad"), 1, 2)
        self.spb_shared_reject_grad = QtWidgets.QDoubleSpinBox()
        self.spb_shared_reject_grad.setDecimals(0)
        self.spb_shared_reject_grad.setRange(0.0, 1000000.0)
        self.spb_shared_reject_grad.setSingleStep(500.0)
        self.spb_shared_reject_grad.setSuffix(" fT/cm")
        self.spb_shared_reject_grad.setValue(
            CONCEPT_PREPROCESSING_DEFAULTS["reject_grad_ft_cm"]
        )
        self.spb_shared_reject_grad.setToolTip(
            "Gradiometer peak-to-peak rejection threshold. Use 0 to disable."
        )
        rejection_layout.addWidget(self.spb_shared_reject_grad, 1, 3)
        rejection_layout.addWidget(QtWidgets.QLabel("QC z"), 2, 2)
        self.spb_shared_outlier_z = QtWidgets.QDoubleSpinBox()
        self.spb_shared_outlier_z.setDecimals(1)
        self.spb_shared_outlier_z.setRange(3.0, 20.0)
        self.spb_shared_outlier_z.setSingleStep(0.5)
        self.spb_shared_outlier_z.setValue(
            CONCEPT_PREPROCESSING_DEFAULTS["trial_outlier_z"]
        )
        self.spb_shared_outlier_z.setToolTip(
            "Robust z-score threshold for trial outlier exclusion."
        )
        rejection_layout.addWidget(self.spb_shared_outlier_z, 2, 3)
        body_layout.addWidget(rejection_group, 1)

        normalization_group = QtWidgets.QGroupBox("Source-trial normalization")
        normalization_layout = QtWidgets.QGridLayout()
        normalization_layout.setContentsMargins(6, 6, 6, 6)
        normalization_layout.setHorizontalSpacing(6)
        normalization_layout.setVerticalSpacing(4)
        normalization_group.setLayout(normalization_layout)

        self.chk_shared_parcel_baseline = QtWidgets.QCheckBox("Parcel z-score")
        self.chk_shared_parcel_baseline.setChecked(True)
        self.chk_shared_parcel_baseline.setToolTip(
            "For every presentation and source parcel, subtract its prestimulus mean and "
            "divide by its prestimulus standard deviation for activity views and source "
            "decoding. Hilbert power always uses the baseline window as its dB reference."
        )
        normalization_layout.addWidget(self.chk_shared_parcel_baseline, 0, 0, 1, 2)
        normalization_layout.addWidget(QtWidgets.QLabel("Baseline start"), 1, 0)
        self.spb_shared_baseline_start_ms = QtWidgets.QDoubleSpinBox()
        self.spb_shared_baseline_start_ms.setDecimals(1)
        self.spb_shared_baseline_start_ms.setRange(-2000.0, -0.1)
        self.spb_shared_baseline_start_ms.setSingleStep(5.0)
        self.spb_shared_baseline_start_ms.setSuffix(" ms")
        self.spb_shared_baseline_start_ms.setValue(-200.0)
        normalization_layout.addWidget(self.spb_shared_baseline_start_ms, 1, 1)
        normalization_layout.addWidget(QtWidgets.QLabel("Baseline end"), 1, 2)
        self.spb_shared_baseline_end_ms = QtWidgets.QDoubleSpinBox()
        self.spb_shared_baseline_end_ms.setDecimals(1)
        self.spb_shared_baseline_end_ms.setRange(-1999.9, 1000.0)
        self.spb_shared_baseline_end_ms.setSingleStep(5.0)
        self.spb_shared_baseline_end_ms.setSuffix(" ms")
        self.spb_shared_baseline_end_ms.setValue(0.0)
        normalization_layout.addWidget(self.spb_shared_baseline_end_ms, 1, 3)
        normalization_layout.addWidget(QtWidgets.QLabel("Temporal bin"), 2, 0)
        self.spb_shared_temporal_bin_ms = QtWidgets.QDoubleSpinBox()
        self.spb_shared_temporal_bin_ms.setDecimals(1)
        self.spb_shared_temporal_bin_ms.setRange(5.0, 100.0)
        self.spb_shared_temporal_bin_ms.setSingleStep(5.0)
        self.spb_shared_temporal_bin_ms.setSuffix(" ms")
        self.spb_shared_temporal_bin_ms.setValue(5.0)
        self.spb_shared_temporal_bin_ms.setToolTip(
            "Display averaging width in Parcel trials and feature-bin width for binned "
            "source-parcel Concept decoding."
        )
        normalization_layout.addWidget(self.spb_shared_temporal_bin_ms, 2, 1)
        body_layout.addWidget(normalization_group, 1)

        frame_layout.addWidget(self.shared_preprocessing_body)
        root_layout.addWidget(self.shared_preprocessing_frame)

        # Compatibility aliases keep every analysis path on these exact widget
        # instances while the duplicated tab controls are removed.
        self.spb_concept_bandpass_low = self.spb_shared_bandpass_low
        self.spb_parcel_heatmap_bandpass_low = self.spb_shared_bandpass_low
        self.spb_topomap_bandpass_low = self.spb_shared_bandpass_low
        self.spb_source_bandpass_low = self.spb_shared_bandpass_low
        self.spb_concept_bandpass_high = self.spb_shared_bandpass_high
        self.spb_parcel_heatmap_bandpass_high = self.spb_shared_bandpass_high
        self.spb_topomap_bandpass_high = self.spb_shared_bandpass_high
        self.spb_source_bandpass_high = self.spb_shared_bandpass_high
        self.chk_concept_downsample = self.chk_shared_downsample
        self.chk_parcel_heatmap_downsample = self.chk_shared_downsample
        self.spb_concept_downsample_sfreq = self.spb_shared_downsample_sfreq
        self.spb_parcel_heatmap_downsample_sfreq = self.spb_shared_downsample_sfreq
        self.chk_concept_mne_reject = self.chk_shared_mne_reject
        self.chk_parcel_heatmap_mne_reject = self.chk_shared_mne_reject
        self.spb_concept_reject_mag = self.spb_shared_reject_mag
        self.spb_parcel_heatmap_reject_mag = self.spb_shared_reject_mag
        self.spb_concept_reject_grad = self.spb_shared_reject_grad
        self.spb_parcel_heatmap_reject_grad = self.spb_shared_reject_grad
        self.chk_concept_exclude_outliers = self.chk_shared_exclude_outliers
        self.chk_parcel_heatmap_exclude_outliers = self.chk_shared_exclude_outliers
        self.spb_concept_outlier_z = self.spb_shared_outlier_z
        self.spb_parcel_heatmap_outlier_z = self.spb_shared_outlier_z
        self.chk_concept_parcel_baseline_zscore = self.chk_shared_parcel_baseline
        self.chk_parcel_heatmap_baseline = self.chk_shared_parcel_baseline
        self.spb_concept_feature_bin_ms = self.spb_shared_temporal_bin_ms
        self.spb_parcel_heatmap_display_bin_ms = self.spb_shared_temporal_bin_ms

        value_controls = (
            self.spb_shared_bandpass_low,
            self.spb_shared_bandpass_high,
            self.spb_shared_downsample_sfreq,
            self.spb_shared_reject_mag,
            self.spb_shared_reject_grad,
            self.spb_shared_outlier_z,
            self.spb_shared_baseline_start_ms,
            self.spb_shared_baseline_end_ms,
            self.spb_shared_temporal_bin_ms,
        )
        checked_controls = (
            self.chk_shared_downsample,
            self.chk_shared_mne_reject,
            self.chk_shared_exclude_outliers,
            self.chk_shared_parcel_baseline,
        )
        for control in value_controls:
            control.valueChanged.connect(self._update_shared_preprocessing_summary)
        for control in checked_controls:
            control.toggled.connect(self._update_shared_preprocessing_summary)
        self.chk_shared_downsample.toggled.connect(
            self._update_shared_source_preprocessing_controls
        )
        self.chk_shared_mne_reject.toggled.connect(
            self._update_shared_source_preprocessing_controls
        )
        self.chk_shared_exclude_outliers.toggled.connect(
            self._update_shared_source_preprocessing_controls
        )
        self.chk_shared_parcel_baseline.toggled.connect(
            self._update_shared_source_preprocessing_controls
        )
        self.spb_shared_temporal_bin_ms.valueChanged.connect(
            self._render_parcel_trial_heatmaps
        )
        self.chk_shared_parcel_baseline.toggled.connect(
            self._update_concept_source_preprocessing_controls
        )
        self._update_shared_source_preprocessing_controls()
        self._update_shared_preprocessing_summary()

    def _set_shared_preprocessing_expanded(self, expanded):
        expanded = bool(expanded)
        self.shared_preprocessing_body.setVisible(expanded)
        self.btn_shared_preprocessing_toggle.setText(
            "Collapse" if expanded else "Expand"
        )
        self.btn_shared_preprocessing_toggle.setToolTip(
            "Collapse the shared controls while keeping their active settings visible."
            if expanded
            else "Expand the shared preprocessing controls."
        )

    def _update_shared_preprocessing_summary(self, *_args):
        low = float(self.spb_shared_bandpass_low.value())
        high = float(self.spb_shared_bandpass_high.value())
        low_label = f"{low:g}" if low > 0.0 else "open"
        high_label = f"{high:g}" if high > 0.0 else "open"
        parts = [f"{low_label}–{high_label} Hz"]
        if self.chk_shared_downsample.isChecked():
            parts.append(f"{self.spb_shared_downsample_sfreq.value():g} Hz")
        else:
            parts.append("native rate")
        parts.append(
            "epoch reject" if self.chk_shared_mne_reject.isChecked() else "epoch reject off"
        )
        if self.chk_shared_exclude_outliers.isChecked():
            parts.append(f"QC z>{self.spb_shared_outlier_z.value():g}")
        else:
            parts.append("QC exclusion off")
        if self.chk_shared_parcel_baseline.isChecked():
            parts.append("parcel z-score")
        else:
            parts.append("parcel z-score off")
        parts.append(f"{self.spb_shared_temporal_bin_ms.value():g} ms bins")
        self.lbl_shared_preprocessing_summary.setText(" · ".join(parts))

    @staticmethod
    def _format_memory_bytes(n_bytes):
        try:
            value = max(0.0, float(n_bytes))
        except Exception:
            return "--"
        units = ("B", "KB", "MB", "GB", "TB")
        unit_idx = 0
        while value >= 1024.0 and unit_idx < len(units) - 1:
            value /= 1024.0
            unit_idx += 1
        if unit_idx == 0:
            return f"{int(value)} {units[unit_idx]}"
        return f"{value:.1f} {units[unit_idx]}"

    @staticmethod
    def _process_resident_bytes():
        try:
            import psutil

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return None

    @staticmethod
    def _mne_data_nbytes(instance):
        data = getattr(instance, "_data", None)
        return int(getattr(data, "nbytes", 0) or 0)

    def set_memory_telemetry_provider(self, callback):
        """Set a callback returning additional cache telemetry fields."""
        self._memory_telemetry_provider = callback if callable(callback) else None
        self._refresh_memory_telemetry()

    def _refresh_memory_telemetry(self):
        label = getattr(self, "lbl_memory_telemetry", None)
        if label is None:
            return

        epoch_bytes = 0
        epoch_count = 0
        seen_epochs = set()
        epoch_objects = list(getattr(self, "_topomap_epochs_by_path", {}).values())
        current_epochs = getattr(self, "_epochs", None)
        if current_epochs is not None:
            epoch_objects.append(current_epochs)
        for epochs in epoch_objects:
            object_id = id(epochs)
            if object_id in seen_epochs:
                continue
            seen_epochs.add(object_id)
            epoch_count += 1
            epoch_bytes += self._mne_data_nbytes(epochs)

        extra = {}
        provider = getattr(self, "_memory_telemetry_provider", None)
        if callable(provider):
            try:
                value = provider()
                if isinstance(value, dict):
                    extra = value
            except Exception:
                extra = {}
        decoder_bytes = int(extra.get("decoder_cache_bytes", 0) or 0)
        decoder_entries = int(extra.get("decoder_cache_entries", 0) or 0)
        parcel_bytes = int(extra.get("parcel_cache_bytes", 0) or 0)
        parcel_entries = int(extra.get("parcel_cache_entries", 0) or 0)
        rss_bytes = self._process_resident_bytes()
        rss_text = self._format_memory_bytes(rss_bytes) if rss_bytes is not None else "--"
        label.setText(
            f"RAM: {rss_text} | epochs: {self._format_memory_bytes(epoch_bytes)} / "
            f"{epoch_count} run{'s' if epoch_count != 1 else ''} | "
            f"decoder cache: {self._format_memory_bytes(decoder_bytes)} / {decoder_entries} | "
            f"parcel cache: {self._format_memory_bytes(parcel_bytes)} / {parcel_entries}"
        )
        details = [
            f"Process resident memory: {rss_text}",
            f"Preloaded MEG epochs: {self._format_memory_bytes(epoch_bytes)} across {epoch_count} run(s)",
            f"Signal-decoder epoch cache: {self._format_memory_bytes(decoder_bytes)} across {decoder_entries} entry/entries",
            f"Parcel time-course cache: {self._format_memory_bytes(parcel_bytes)} across {parcel_entries} entry/entries",
            "Epoch estimates count shared MNE objects once and exclude short-lived plotting/source copies.",
        ]
        label.setToolTip("\n".join(details))

    def _init_stimulus_report_tab(self):
        stim_tab = QtWidgets.QWidget()
        stim_layout = QtWidgets.QVBoxLayout()
        stim_layout.setContentsMargins(0, 0, 0, 0)
        stim_layout.setSpacing(0)
        stim_tab.setLayout(stim_layout)
        self.stimulus_report = StimulusResponseReport(report_view=_LayoutHost(stim_layout), c=self.c)
        if hasattr(self.stimulus_report, "btn_recompute"):
            self.stimulus_report.btn_recompute.clicked.connect(self._on_stimulus_report_recomputed)
        self.tabs.addTab(stim_tab, "Stimulus report")

    @staticmethod
    def _new_channel_tree_widget():
        """Create a channel picker matching the signal viewer control."""
        combo = CheckableTreeComboBox(c=_NullContext())
        combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        combo.setToolTip(
            "Open the tree to select MEG channels by sensor type, hemisphere, or coarse helmet region. "
            "Sensor regions describe helmet location, not cortical source localization."
        )
        return combo

    def _init_epoch_image_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        tab.setLayout(layout)

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        controls.addWidget(QtWidgets.QLabel("Stimulus"), 0, 0)
        self.cmb_epoch_stimulus = QtWidgets.QComboBox()
        self.cmb_epoch_stimulus.currentIndexChanged.connect(self._on_epoch_stimulus_changed)
        controls.addWidget(self.cmb_epoch_stimulus, 0, 1)

        controls.addWidget(QtWidgets.QLabel("Channels"), 1, 0)
        self.lst_epoch_channels = self._new_channel_tree_widget()
        controls.addWidget(self.lst_epoch_channels, 1, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_epoch_select_all = QtWidgets.QPushButton("Select all")
        self.btn_epoch_select_all.clicked.connect(
            lambda: self._set_all_channel_selection(self.lst_epoch_channels, True)
        )
        self.btn_epoch_clear_sel = QtWidgets.QPushButton("Clear")
        self.btn_epoch_clear_sel.clicked.connect(
            lambda: self._set_all_channel_selection(self.lst_epoch_channels, False)
        )
        self.chk_epoch_combine = QtWidgets.QCheckBox("Combine channels (MNE mean)")
        self.chk_epoch_combine.setChecked(False)
        self.btn_plot_epoch = QtWidgets.QPushButton("Plot epoch image")
        self.btn_plot_epoch.clicked.connect(self._plot_epoch_image)
        btn_row.addWidget(self.btn_epoch_select_all)
        btn_row.addWidget(self.btn_epoch_clear_sel)
        btn_row.addWidget(self.chk_epoch_combine)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_plot_epoch)
        controls.addLayout(btn_row, 2, 1)

        layout.addLayout(controls)
        self.epoch_plot_container = QtWidgets.QWidget()
        self.epoch_plot_layout = QtWidgets.QVBoxLayout()
        self.epoch_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.epoch_plot_layout.setSpacing(0)
        self.epoch_plot_container.setLayout(self.epoch_plot_layout)
        self.lbl_epoch_placeholder = QtWidgets.QLabel("Load a .fif file to display epoch images.")
        self.lbl_epoch_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.epoch_plot_layout.addWidget(self.lbl_epoch_placeholder)
        layout.addWidget(self.epoch_plot_container)

        self.tabs.addTab(tab, "Epoch images")

    def _init_evoked_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        tab.setLayout(layout)

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        controls.addWidget(QtWidgets.QLabel("Stimulus"), 0, 0)
        self.cmb_evoked_stimulus = QtWidgets.QComboBox()
        self.cmb_evoked_stimulus.currentIndexChanged.connect(self._on_evoked_stimulus_changed)
        controls.addWidget(self.cmb_evoked_stimulus, 0, 1)
        controls.addWidget(QtWidgets.QLabel("Grouping"), 0, 2)
        self.cmb_evoked_grouping = QtWidgets.QComboBox()
        self.cmb_evoked_grouping.addItems(["Grouped (canonical)", "Grouped (category)", "Individual stimulus"])
        self.cmb_evoked_grouping.currentIndexChanged.connect(self._on_evoked_grouping_changed)
        controls.addWidget(self.cmb_evoked_grouping, 0, 3)

        controls.addWidget(QtWidgets.QLabel("Channels"), 1, 0)
        self.lst_evoked_channels = self._new_channel_tree_widget()
        controls.addWidget(self.lst_evoked_channels, 1, 1, 1, 3)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_evoked_select_all = QtWidgets.QPushButton("Select all")
        self.btn_evoked_select_all.clicked.connect(
            lambda: self._set_all_channel_selection(self.lst_evoked_channels, True)
        )
        self.btn_evoked_clear_sel = QtWidgets.QPushButton("Clear")
        self.btn_evoked_clear_sel.clicked.connect(
            lambda: self._set_all_channel_selection(self.lst_evoked_channels, False)
        )
        self.btn_plot_evoked = QtWidgets.QPushButton("Plot evoked response")
        self.btn_plot_evoked.clicked.connect(self._plot_evoked_response)
        btn_row.addWidget(self.btn_evoked_select_all)
        btn_row.addWidget(self.btn_evoked_clear_sel)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_plot_evoked)
        controls.addLayout(btn_row, 2, 1)

        layout.addLayout(controls)
        self.evoked_plot_container = QtWidgets.QWidget()
        self.evoked_plot_layout = QtWidgets.QVBoxLayout()
        self.evoked_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.evoked_plot_layout.setSpacing(0)
        self.evoked_plot_container.setLayout(self.evoked_plot_layout)
        self.lbl_evoked_placeholder = QtWidgets.QLabel("Load a .fif file to display evoked responses.")
        self.lbl_evoked_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.evoked_plot_layout.addWidget(self.lbl_evoked_placeholder)
        layout.addWidget(self.evoked_plot_container)
        self.lbl_evoked_channels_info = QtWidgets.QLabel("Channels: none")
        self.lbl_evoked_channels_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_evoked_channels_info)

        self.tabs.addTab(tab, "Evoked responses")

    def _init_topomap_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        tab.setLayout(layout)

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        controls.addWidget(QtWidgets.QLabel("Stimulus"), 0, 0)
        self.cmb_topomap_stimulus = QtWidgets.QComboBox()
        self.cmb_topomap_stimulus.currentIndexChanged.connect(self._on_topomap_stimulus_changed)
        controls.addWidget(self.cmb_topomap_stimulus, 0, 1)

        controls.addWidget(QtWidgets.QLabel("Grouping"), 0, 2)
        self.cmb_topomap_grouping = QtWidgets.QComboBox()
        self.cmb_topomap_grouping.addItems(["Grouped (canonical)", "Grouped (category)", "Individual stimulus"])
        self.cmb_topomap_grouping.currentIndexChanged.connect(self._on_topomap_grouping_changed)
        controls.addWidget(self.cmb_topomap_grouping, 0, 3)

        controls.addWidget(QtWidgets.QLabel("Run"), 0, 4)
        self.cmb_topomap_run = QtWidgets.QComboBox()
        self.cmb_topomap_run.currentIndexChanged.connect(self._on_topomap_run_changed)
        controls.addWidget(self.cmb_topomap_run, 0, 5)

        controls.addWidget(QtWidgets.QLabel("Times (s)"), 1, 0)
        self.le_topomap_times = QtWidgets.QLineEdit("0.00, 0.08, 0.10, 0.12, 0.20, 0.30, 0.40")
        self.le_topomap_times.setToolTip("Comma-separated times in seconds.")
        controls.addWidget(self.le_topomap_times, 1, 1, 1, 3)

        controls.addWidget(QtWidgets.QLabel("View mode"), 1, 4)
        self.cmb_topomap_view_mode = QtWidgets.QComboBox()
        self.cmb_topomap_view_mode.addItems(["Times view", "Slider view"])
        self.cmb_topomap_view_mode.currentIndexChanged.connect(self._on_topomap_view_mode_changed)
        controls.addWidget(self.cmb_topomap_view_mode, 1, 5)

        lbl_topomap_preprocessing = QtWidgets.QLabel(
            "Filtering uses Shared preprocessing above"
        )
        lbl_topomap_preprocessing.setToolTip(
            "Topomap filtering uses the shared bandpass low and high cutoffs."
        )
        controls.addWidget(lbl_topomap_preprocessing, 2, 0, 1, 3)
        self.spb_shared_bandpass_low.valueChanged.connect(
            self._on_topomap_bandpass_changed
        )
        self.spb_shared_bandpass_high.valueChanged.connect(
            self._on_topomap_bandpass_changed
        )

        self.topomap_slider_widget = QtWidgets.QWidget()
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(6)
        slider_layout.addWidget(QtWidgets.QLabel("Time slider"))
        self.sld_topomap_time = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_topomap_time.setRange(0, 400)
        self.sld_topomap_time.setSingleStep(10)
        self.sld_topomap_time.setPageStep(20)
        self.sld_topomap_time.setValue(100)
        self.sld_topomap_time.valueChanged.connect(self._on_topomap_slider_changed)
        slider_layout.addWidget(self.sld_topomap_time, 1)
        self.lbl_topomap_slider_time = QtWidgets.QLabel("0.100 s")
        self.lbl_topomap_slider_time.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        slider_layout.addWidget(self.lbl_topomap_slider_time)
        self.topomap_slider_widget.setLayout(slider_layout)
        controls.addWidget(self.topomap_slider_widget, 3, 0, 1, 6)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_plot_topomap = QtWidgets.QPushButton("Plot topomap")
        self.btn_plot_topomap.clicked.connect(self._plot_topomap_only)
        self.btn_plot_joint = QtWidgets.QPushButton("Plot joint")
        self.btn_plot_joint.clicked.connect(self._plot_topomap_joint)
        btn_row.addWidget(self.btn_plot_topomap)
        btn_row.addWidget(self.btn_plot_joint)
        btn_row.addStretch()
        controls.addLayout(btn_row, 2, 3, 1, 3)

        layout.addLayout(controls)
        self.topomap_plot_container = QtWidgets.QWidget()
        self.topomap_plot_layout = QtWidgets.QVBoxLayout()
        self.topomap_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.topomap_plot_layout.setSpacing(0)
        self.topomap_plot_container.setLayout(self.topomap_plot_layout)
        self.lbl_topomap_placeholder = QtWidgets.QLabel("Load a .fif file to display topomaps.")
        self.lbl_topomap_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.topomap_plot_layout.addWidget(self.lbl_topomap_placeholder)
        layout.addWidget(self.topomap_plot_container)

        self.lbl_topomap_info = QtWidgets.QLabel("")
        self.lbl_topomap_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_topomap_info)

        self.tabs.addTab(tab, "Topomap")
        self._on_topomap_view_mode_changed()

    def _init_frequency_domain_tab(self):
        """Build MNE-powered spectral QC and time-frequency controls."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        controls.addWidget(QtWidgets.QLabel("PSD source"), 0, 0)
        self.cmb_frequency_source = QtWidgets.QComboBox()
        self.cmb_frequency_source.addItem("Continuous raw (data quality)", "raw")
        self.cmb_frequency_source.addItem("Epoched trials (condition comparison)", "epochs")
        self.cmb_frequency_source.currentIndexChanged.connect(self._on_frequency_source_changed)
        controls.addWidget(self.cmb_frequency_source, 0, 1)

        controls.addWidget(QtWidgets.QLabel("Grouping"), 0, 2)
        self.cmb_frequency_grouping = QtWidgets.QComboBox()
        self.cmb_frequency_grouping.addItems(
            ["Grouped (canonical)", "Grouped (category)", "Individual stimulus"]
        )
        self.cmb_frequency_grouping.currentIndexChanged.connect(
            self._on_frequency_grouping_changed
        )
        controls.addWidget(self.cmb_frequency_grouping, 0, 3)

        controls.addWidget(QtWidgets.QLabel("Stimulus"), 0, 4)
        self.cmb_frequency_stimulus = QtWidgets.QComboBox()
        controls.addWidget(self.cmb_frequency_stimulus, 0, 5)

        controls.addWidget(QtWidgets.QLabel("Channels"), 1, 0)
        self.lst_frequency_channels = self._new_channel_tree_widget()
        self.lst_frequency_channels.setToolTip(
            "Channels used for PSD and time-frequency power. TFR is limited to 24 channels."
        )
        controls.addWidget(self.lst_frequency_channels, 1, 1, 1, 5)

        self.btn_frequency_select_all = QtWidgets.QPushButton("Select all")
        self.btn_frequency_select_all.clicked.connect(
            lambda: self._set_all_channel_selection(self.lst_frequency_channels, True)
        )
        self.btn_frequency_clear = QtWidgets.QPushButton("Clear")
        self.btn_frequency_clear.clicked.connect(
            lambda: self._set_all_channel_selection(self.lst_frequency_channels, False)
        )
        controls.addWidget(self.btn_frequency_select_all, 2, 0)
        controls.addWidget(self.btn_frequency_clear, 2, 1)

        controls.addWidget(QtWidgets.QLabel("Frequency"), 2, 2)
        self.spb_frequency_fmin = QtWidgets.QDoubleSpinBox()
        self.spb_frequency_fmin.setRange(0.1, 1000.0)
        self.spb_frequency_fmin.setDecimals(1)
        self.spb_frequency_fmin.setValue(1.0)
        self.spb_frequency_fmin.setSuffix(" Hz")
        controls.addWidget(self.spb_frequency_fmin, 2, 3)
        self.spb_frequency_fmax = QtWidgets.QDoubleSpinBox()
        self.spb_frequency_fmax.setRange(0.2, 2000.0)
        self.spb_frequency_fmax.setDecimals(1)
        self.spb_frequency_fmax.setValue(120.0)
        self.spb_frequency_fmax.setSuffix(" Hz")
        controls.addWidget(self.spb_frequency_fmax, 2, 4)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_plot_frequency_psd = QtWidgets.QPushButton("Plot MNE PSD")
        self.btn_plot_frequency_psd.setToolTip(
            "Welch PSD for broadband shape, line noise, and condition-level spectral differences."
        )
        self.btn_plot_frequency_psd.clicked.connect(self._plot_frequency_psd)
        self.btn_plot_frequency_tfr = QtWidgets.QPushButton("Plot Morlet time-frequency")
        self.btn_plot_frequency_tfr.setToolTip(
            "MNE Morlet power across trials; pre-stimulus baseline is used when available."
        )
        self.btn_plot_frequency_tfr.clicked.connect(self._plot_frequency_tfr)
        button_row.addStretch()
        button_row.addWidget(self.btn_plot_frequency_psd)
        button_row.addWidget(self.btn_plot_frequency_tfr)
        controls.addLayout(button_row, 2, 5)
        layout.addLayout(controls)

        self.frequency_plot_container = QtWidgets.QWidget()
        self.frequency_plot_layout = QtWidgets.QVBoxLayout(self.frequency_plot_container)
        self.frequency_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.frequency_plot_layout.setSpacing(0)
        self.lbl_frequency_placeholder = QtWidgets.QLabel(
            "Load a .fif file, then plot spectral QC or time-frequency power."
        )
        self.lbl_frequency_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_frequency_placeholder.setWordWrap(True)
        self.frequency_plot_layout.addWidget(self.lbl_frequency_placeholder)
        layout.addWidget(self.frequency_plot_container, 1)

        self.lbl_frequency_info = QtWidgets.QLabel(
            "PSD highlights broadband shape and 60 Hz contamination; condition spectra and TFR can reveal candidate decoding features."
        )
        self.lbl_frequency_info.setWordWrap(True)
        self.lbl_frequency_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_frequency_info.setToolTip(
            "Condition eta-squared is a descriptive univariate screen, not cross-validated "
            "decoding accuracy. Confirm viability in the Concept decoding tab."
        )
        layout.addWidget(self.lbl_frequency_info)

        self.tabs.addTab(tab, "Frequency domain")
        self._on_frequency_source_changed()

    def _init_source_tc_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        tab.setLayout(layout)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(6)
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        self.cmb_source_mode = QtWidgets.QComboBox()
        self.cmb_source_mode.addItems(["Particular time", "Window"])
        self.cmb_source_mode.currentIndexChanged.connect(self._on_source_mode_changed)
        mode_row.addWidget(self.cmb_source_mode)
        mode_row.addWidget(QtWidgets.QLabel("Brain"))
        self.cmb_source_brain_view = QtWidgets.QComboBox()
        self.cmb_source_brain_view.addItem("Patient MRI", "subject")
        self.cmb_source_brain_view.addItem("fsaverage", "fsaverage")
        self.cmb_source_brain_view.setToolTip(
            "Display source estimates on the patient FreeSurfer brain or morphed to fsaverage."
        )
        self.cmb_source_brain_view.currentIndexChanged.connect(self._on_source_brain_view_changed)
        mode_row.addWidget(self.cmb_source_brain_view)
        mode_row.addWidget(QtWidgets.QLabel("Source space"))
        self.cmb_source_space_kind = QtWidgets.QComboBox()
        self.cmb_source_space_kind.addItem("Surface", "surface")
        self.cmb_source_space_kind.addItem("Volume", "volume")
        self.cmb_source_space_kind.setToolTip("Choose the source space used when an inverse must be generated.")
        self.cmb_source_space_kind.currentIndexChanged.connect(self._on_source_space_kind_changed)
        mode_row.addWidget(self.cmb_source_space_kind)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        self.source_point_widget = QtWidgets.QWidget()
        point_layout = QtWidgets.QVBoxLayout()
        point_layout.setContentsMargins(0, 0, 0, 0)
        point_layout.setSpacing(6)
        self.source_point_widget.setLayout(point_layout)
        layout.addWidget(self.source_point_widget)

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        controls.addWidget(QtWidgets.QLabel("Stimulus"), 0, 0)
        self.cmb_source_evoked = QtWidgets.QComboBox()
        self.cmb_source_evoked.currentIndexChanged.connect(self._on_source_stimulus_changed)
        controls.addWidget(self.cmb_source_evoked, 0, 1)

        controls.addWidget(QtWidgets.QLabel("Run"), 0, 2)
        self.cmb_source_run = QtWidgets.QComboBox()
        self.cmb_source_run.setToolTip("Choose a FIF run, or average source estimates across all runs.")
        self.cmb_source_run.currentIndexChanged.connect(self._on_source_run_changed)
        controls.addWidget(self.cmb_source_run, 0, 3)

        controls.addWidget(QtWidgets.QLabel("Grouping"), 0, 4)
        self.cmb_source_grouping = QtWidgets.QComboBox()
        self.cmb_source_grouping.addItems(["Grouped (canonical)", "Grouped (category)", "Individual stimulus"])
        self.cmb_source_grouping.currentIndexChanged.connect(self._on_source_grouping_changed)
        controls.addWidget(self.cmb_source_grouping, 0, 5)

        controls.addWidget(QtWidgets.QLabel("Method"), 1, 0)
        self.cmb_source_method = QtWidgets.QComboBox()
        self.cmb_source_method.addItems(["MNE", "dSPM", "sLORETA", "eLORETA", "LCMV"])
        self.cmb_source_method.setCurrentText("MNE")
        controls.addWidget(self.cmb_source_method, 1, 1)

        controls.addWidget(QtWidgets.QLabel("Initial time (s)"), 1, 2)
        self.spb_source_initial_time = QtWidgets.QDoubleSpinBox()
        self.spb_source_initial_time.setDecimals(3)
        self.spb_source_initial_time.setRange(-1.0, 5.0)
        self.spb_source_initial_time.setSingleStep(0.01)
        self.spb_source_initial_time.setValue(0.10)
        self.spb_source_initial_time.editingFinished.connect(self._on_source_time_spinbox_committed)
        controls.addWidget(self.spb_source_initial_time, 1, 3)

        controls.addWidget(QtWidgets.QLabel("Views"), 2, 2)
        self.le_source_views = QtWidgets.QLineEdit("lat, med, dor, ven")
        self.le_source_views.setToolTip("Comma-separated cortical views, e.g. lat, med, dor, ven, ros, cau")
        controls.addWidget(self.le_source_views, 2, 3)

        self.chk_source_brain_screenshot = QtWidgets.QCheckBox("Brain screenshots")
        self.chk_source_brain_screenshot.setChecked(False)
        self.chk_source_brain_screenshot.setToolTip(
            "Render PyVista brain screenshots in this panel. Leave off if the 3D viewer is already open."
        )
        controls.addWidget(self.chk_source_brain_screenshot, 2, 4, 1, 2)

        controls.addWidget(QtWidgets.QLabel("SNR"), 1, 4)
        self.spb_source_snr = QtWidgets.QDoubleSpinBox()
        self.spb_source_snr.setDecimals(2)
        self.spb_source_snr.setRange(0.01, 100.0)
        self.spb_source_snr.setSingleStep(0.25)
        self.spb_source_snr.setValue(3.0)
        self.spb_source_snr.valueChanged.connect(self._on_source_snr_changed)
        controls.addWidget(self.spb_source_snr, 1, 5)

        controls.addWidget(QtWidgets.QLabel("lambda2"), 2, 0)
        self.lbl_source_lambda2 = QtWidgets.QLabel("0.111111")
        controls.addWidget(self.lbl_source_lambda2, 2, 1)

        controls.addWidget(QtWidgets.QLabel("Inverse operator"), 3, 0)
        self.le_source_inverse = QtWidgets.QLineEdit()
        self.le_source_inverse.setPlaceholderText("Optional: select *-inv.fif")
        controls.addWidget(self.le_source_inverse, 3, 1, 1, 4)
        self.btn_source_inverse_browse = QtWidgets.QPushButton("Browse")
        self.btn_source_inverse_browse.clicked.connect(self._browse_inverse_operator_file)
        controls.addWidget(self.btn_source_inverse_browse, 3, 5)

        controls.addWidget(QtWidgets.QLabel("subjects_dir"), 4, 0)
        self.lbl_source_subjects_dir = QtWidgets.QLabel("")
        self.lbl_source_subjects_dir.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        controls.addWidget(self.lbl_source_subjects_dir, 4, 1, 1, 2)

        controls.addWidget(QtWidgets.QLabel("subject"), 4, 3)
        self.lbl_source_subject = QtWidgets.QLabel("")
        self.lbl_source_subject.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        controls.addWidget(self.lbl_source_subject, 4, 4, 1, 2)

        lbl_source_preprocessing = QtWidgets.QLabel(
            "Filtering uses Shared preprocessing above"
        )
        lbl_source_preprocessing.setToolTip(
            "Source time-course filtering uses the shared bandpass low and high cutoffs."
        )
        controls.addWidget(lbl_source_preprocessing, 5, 0, 1, 5)

        self.btn_source_compute = QtWidgets.QPushButton("Compute/refresh source time course")
        self.btn_source_compute.clicked.connect(self._plot_source_time_course)
        controls.addWidget(self.btn_source_compute, 5, 5)

        step_row = QtWidgets.QHBoxLayout()
        step_row.setContentsMargins(0, 0, 0, 0)
        step_row.setSpacing(6)
        self.btn_source_time_back = QtWidgets.QPushButton("Back")
        self.btn_source_time_back.setToolTip("Move backward by the configured source time step.")
        self.btn_source_time_back.clicked.connect(self._step_source_time_backward)
        self.btn_source_time_forward = QtWidgets.QPushButton("Forward")
        self.btn_source_time_forward.setToolTip("Move forward by the configured source time step.")
        self.btn_source_time_forward.clicked.connect(self._step_source_time_forward)
        self.spb_source_step_ms = QtWidgets.QDoubleSpinBox()
        self.spb_source_step_ms.setDecimals(1)
        self.spb_source_step_ms.setRange(0.1, 5000.0)
        self.spb_source_step_ms.setSingleStep(1.0)
        self.spb_source_step_ms.setValue(5.0)
        self.spb_source_step_ms.setSuffix(" ms")
        self.spb_source_step_ms.setToolTip("Milliseconds to move when pressing Back or Forward.")
        self.lbl_source_time_sample = QtWidgets.QLabel("sample --/--")
        self.lbl_source_time_sample.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        step_row.addWidget(self.btn_source_time_back)
        step_row.addWidget(self.btn_source_time_forward)
        step_row.addWidget(QtWidgets.QLabel("Step"))
        step_row.addWidget(self.spb_source_step_ms)
        step_row.addWidget(self.lbl_source_time_sample)
        step_row.addStretch()
        controls.addLayout(step_row, 6, 0, 1, 6)

        movie_row = QtWidgets.QHBoxLayout()
        movie_row.setContentsMargins(0, 0, 0, 0)
        movie_row.setSpacing(6)
        self.spb_source_movie_tmin = QtWidgets.QDoubleSpinBox()
        self.spb_source_movie_tmin.setDecimals(3)
        self.spb_source_movie_tmin.setRange(-10.0, 10.0)
        self.spb_source_movie_tmin.setSingleStep(0.005)
        self.spb_source_movie_tmin.setValue(0.050)
        self.spb_source_movie_tmin.setSuffix(" s")
        self.spb_source_movie_tmin.setToolTip("Movie start time passed to brain.save_movie(tmin=...).")
        self.spb_source_movie_tmax = QtWidgets.QDoubleSpinBox()
        self.spb_source_movie_tmax.setDecimals(3)
        self.spb_source_movie_tmax.setRange(-10.0, 10.0)
        self.spb_source_movie_tmax.setSingleStep(0.005)
        self.spb_source_movie_tmax.setValue(0.160)
        self.spb_source_movie_tmax.setSuffix(" s")
        self.spb_source_movie_tmax.setToolTip("Movie end time passed to brain.save_movie(tmax=...).")
        self.btn_source_movie_full_range = QtWidgets.QPushButton("Full range")
        self.btn_source_movie_full_range.setToolTip("Set movie tmin/tmax to the current source estimate time range.")
        self.btn_source_movie_full_range.clicked.connect(self._set_source_movie_full_range)
        self.btn_source_save_movie = QtWidgets.QPushButton("Save movie")
        self.btn_source_save_movie.setToolTip(
            "Save the current source estimate as an MNE Brain movie with time_dilation=20, "
            "interpolation='linear', and framerate=10."
        )
        self.btn_source_save_movie.clicked.connect(self._save_source_movie)
        movie_row.addWidget(QtWidgets.QLabel("Movie tmin"))
        movie_row.addWidget(self.spb_source_movie_tmin)
        movie_row.addWidget(QtWidgets.QLabel("tmax"))
        movie_row.addWidget(self.spb_source_movie_tmax)
        movie_row.addWidget(self.btn_source_movie_full_range)
        movie_row.addWidget(self.btn_source_save_movie)
        movie_row.addStretch()
        controls.addLayout(movie_row, 7, 0, 1, 6)

        point_layout.addLayout(controls)
        self.lbl_source_status = QtWidgets.QLabel("Source status: no source estimate available.")
        point_layout.addWidget(self.lbl_source_status)

        self.source_plot_container = QtWidgets.QWidget()
        self.source_plot_layout = QtWidgets.QVBoxLayout()
        self.source_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.source_plot_layout.setSpacing(0)
        self.source_plot_container.setLayout(self.source_plot_layout)
        self.lbl_source_placeholder = QtWidgets.QLabel("Compute inverse source estimate to view source time course.")
        self.lbl_source_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.source_plot_layout.addWidget(self.lbl_source_placeholder)
        point_layout.addWidget(self.source_plot_container)

        metric_row = QtWidgets.QHBoxLayout()
        metric_row.setContentsMargins(0, 0, 0, 0)
        metric_row.setSpacing(6)
        metric_row.addWidget(QtWidgets.QLabel("Display metric"))
        self.cmb_source_display_metric = QtWidgets.QComboBox()
        self.cmb_source_display_metric.addItems(["mean |stc|", "max |stc|", "LH mean |stc|", "RH mean |stc|"])
        self.cmb_source_display_metric.currentIndexChanged.connect(lambda *_args: self._render_source_time_course(render_brain_screenshots=False))
        metric_row.addWidget(self.cmb_source_display_metric)
        metric_row.addWidget(QtWidgets.QLabel("Top ROI atlas"))
        self.cmb_source_roi_parcellation = QtWidgets.QComboBox()
        self.cmb_source_roi_parcellation.addItem("Auto", "auto")
        self.cmb_source_roi_parcellation.addItem("aparc", "aparc")
        self.cmb_source_roi_parcellation.addItem("aparc+aseg", "aparc+aseg")
        self.cmb_source_roi_parcellation.addItem("HCP-MMP", "HCPMMP1")
        self.cmb_source_roi_parcellation.addItem("HCP-MMP combined", "HCPMMP1_combined")
        self.cmb_source_roi_parcellation.setToolTip("Atlas used for top source-label time courses.")
        self.cmb_source_roi_parcellation.currentIndexChanged.connect(lambda *_args: self._render_source_time_course(render_brain_screenshots=False))
        metric_row.addWidget(self.cmb_source_roi_parcellation)
        metric_row.addWidget(QtWidgets.QLabel("N"))
        self.spb_source_top_roi_count = QtWidgets.QSpinBox()
        self.spb_source_top_roi_count.setRange(0, 24)
        self.spb_source_top_roi_count.setValue(7)
        self.spb_source_top_roi_count.setToolTip("Number of strongest atlas labels to plot. Set 0 to hide ROI traces.")
        self.spb_source_top_roi_count.valueChanged.connect(lambda *_args: self._render_source_time_course(render_brain_screenshots=False))
        metric_row.addWidget(self.spb_source_top_roi_count)
        metric_row.addStretch()
        point_layout.addLayout(metric_row)

        self.source_window_widget = QtWidgets.QWidget()
        window_layout = QtWidgets.QVBoxLayout()
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(6)
        self.source_window_widget.setLayout(window_layout)
        layout.addWidget(self.source_window_widget)
        self._init_semantic_decoding_section(window_layout)

        self.tabs.addTab(tab, "Source analysis")
        self._on_source_snr_changed()
        self._on_source_mode_changed()
        self._update_source_time_controls()

    def _init_parcel_trial_heatmap_tab(self):
        """Create the trial-by-parcel heatmap view for repeated stimuli."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        tab.setLayout(layout)

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        controls.addWidget(QtWidgets.QLabel("Stimulus"), 0, 0)
        self.cmb_parcel_heatmap_stimulus = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_stimulus.setToolTip(
            "Choose the repeated stimulus or category whose presentations should be compared."
        )
        self.cmb_parcel_heatmap_stimulus.currentIndexChanged.connect(
            self._refresh_parcel_heatmap_comparison_selector
        )
        controls.addWidget(self.cmb_parcel_heatmap_stimulus, 0, 1)

        controls.addWidget(QtWidgets.QLabel("Grouping"), 0, 2)
        self.cmb_parcel_heatmap_grouping = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_grouping.addItems(
            ["Grouped (canonical)", "Grouped (category)", "Individual stimulus"]
        )
        self.cmb_parcel_heatmap_grouping.currentIndexChanged.connect(
            self._on_parcel_heatmap_grouping_changed
        )
        controls.addWidget(self.cmb_parcel_heatmap_grouping, 0, 3)

        controls.addWidget(QtWidgets.QLabel("Run"), 0, 4)
        self.cmb_parcel_heatmap_run = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_run.setToolTip(
            "Choose a FIF run or combine matching presentations across all loaded runs."
        )
        self.cmb_parcel_heatmap_run.currentIndexChanged.connect(
            self._on_parcel_heatmap_run_changed
        )
        controls.addWidget(self.cmb_parcel_heatmap_run, 0, 5)

        controls.addWidget(QtWidgets.QLabel("Source space"), 1, 0)
        self.cmb_parcel_heatmap_source_space = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_source_space.addItem("Surface", "surface")
        self.cmb_parcel_heatmap_source_space.addItem("Volume", "volume")
        self.cmb_parcel_heatmap_source_space.currentIndexChanged.connect(
            self._on_parcel_heatmap_source_space_changed
        )
        controls.addWidget(self.cmb_parcel_heatmap_source_space, 1, 1)

        controls.addWidget(QtWidgets.QLabel("Atlas"), 1, 2)
        self.cmb_parcel_heatmap_parcellation = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_parcellation.addItem("aparc", "aparc")
        self.cmb_parcel_heatmap_parcellation.addItem("HCP-MMP", "HCPMMP1")
        self.cmb_parcel_heatmap_parcellation.addItem("HCP-MMP combined", "HCPMMP1_combined")
        self.cmb_parcel_heatmap_parcellation.setToolTip(
            "Surface source spaces use a cortical atlas; volume source spaces use aparc+aseg."
        )
        controls.addWidget(self.cmb_parcel_heatmap_parcellation, 1, 3)

        controls.addWidget(QtWidgets.QLabel("Method"), 1, 4)
        self.cmb_parcel_heatmap_method = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_method.addItems(["MNE", "dSPM", "sLORETA", "eLORETA"])
        controls.addWidget(self.cmb_parcel_heatmap_method, 1, 5)

        controls.addWidget(QtWidgets.QLabel("Window start"), 2, 0)
        self.spb_parcel_heatmap_tmin_ms = QtWidgets.QDoubleSpinBox()
        self.spb_parcel_heatmap_tmin_ms.setDecimals(1)
        self.spb_parcel_heatmap_tmin_ms.setRange(-1000.0, 5000.0)
        self.spb_parcel_heatmap_tmin_ms.setValue(0.0)
        self.spb_parcel_heatmap_tmin_ms.setSuffix(" ms")
        controls.addWidget(self.spb_parcel_heatmap_tmin_ms, 2, 1)

        controls.addWidget(QtWidgets.QLabel("Window end"), 2, 2)
        self.spb_parcel_heatmap_tmax_ms = QtWidgets.QDoubleSpinBox()
        self.spb_parcel_heatmap_tmax_ms.setDecimals(1)
        self.spb_parcel_heatmap_tmax_ms.setRange(-999.0, 10000.0)
        self.spb_parcel_heatmap_tmax_ms.setValue(400.0)
        self.spb_parcel_heatmap_tmax_ms.setSuffix(" ms")
        controls.addWidget(self.spb_parcel_heatmap_tmax_ms, 2, 3)

        controls.addWidget(QtWidgets.QLabel("SNR"), 2, 4)
        self.spb_parcel_heatmap_snr = QtWidgets.QDoubleSpinBox()
        self.spb_parcel_heatmap_snr.setDecimals(2)
        self.spb_parcel_heatmap_snr.setRange(0.01, 100.0)
        self.spb_parcel_heatmap_snr.setValue(3.0)
        controls.addWidget(self.spb_parcel_heatmap_snr, 2, 5)

        controls.addWidget(QtWidgets.QLabel("Frequency band"), 3, 0)
        self.cmb_parcel_heatmap_frequency_band = QtWidgets.QComboBox()
        for band_key, band_label, _low, _high in PARCEL_TRIAL_FREQUENCY_BANDS:
            self.cmb_parcel_heatmap_frequency_band.addItem(band_label, band_key)
        self.cmb_parcel_heatmap_frequency_band.setToolTip(
            "Broadband uses the shared low/high cutoffs. A named band must fit entirely "
            "inside that shared passband and below the downsampled Nyquist frequency. "
            "Hilbert power requires one of the named bands."
        )
        self.cmb_parcel_heatmap_frequency_band.currentIndexChanged.connect(
            self._on_parcel_heatmap_frequency_band_changed
        )
        self.spb_shared_bandpass_low.valueChanged.connect(
            self._on_parcel_heatmap_frequency_band_changed
        )
        self.spb_shared_bandpass_high.valueChanged.connect(
            self._on_parcel_heatmap_frequency_band_changed
        )
        self.chk_shared_downsample.toggled.connect(
            self._on_parcel_heatmap_frequency_band_changed
        )
        self.spb_shared_downsample_sfreq.valueChanged.connect(
            self._on_parcel_heatmap_frequency_band_changed
        )
        controls.addWidget(self.cmb_parcel_heatmap_frequency_band, 3, 1)

        controls.addWidget(QtWidgets.QLabel("Representation"), 3, 2)
        self.cmb_parcel_heatmap_representation = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_representation.addItem(
            "Source activity", "activity"
        )
        self.cmb_parcel_heatmap_representation.addItem(
            "Hilbert band power (dB)", "band_power_db"
        )
        self.cmb_parcel_heatmap_representation.setToolTip(
            "Source activity shows the existing band-limited waveform. Hilbert band "
            "power shows each trial/parcel's instantaneous power in dB relative to "
            "its own prestimulus baseline."
        )
        self.cmb_parcel_heatmap_representation.currentIndexChanged.connect(
            self._on_parcel_heatmap_representation_changed
        )
        controls.addWidget(self.cmb_parcel_heatmap_representation, 3, 3)

        self.chk_parcel_heatmap_absolute = QtWidgets.QCheckBox("Absolute magnitude")
        self.chk_parcel_heatmap_absolute.setChecked(True)
        self.chk_parcel_heatmap_absolute.setToolTip(
            "Show mean rectified band-limited parcel activity. Uncheck to display the "
            "signed mean-flip waveform. Neither mode is a PSD power estimate."
        )
        self.chk_parcel_heatmap_absolute.toggled.connect(self._render_parcel_trial_heatmaps)
        controls.addWidget(self.chk_parcel_heatmap_absolute, 4, 0, 1, 2)

        controls.addWidget(QtWidgets.QLabel("Max presentations"), 3, 4)
        self.spb_parcel_heatmap_max_presentations = QtWidgets.QSpinBox()
        self.spb_parcel_heatmap_max_presentations.setRange(1, 36)
        self.spb_parcel_heatmap_max_presentations.setValue(12)
        self.spb_parcel_heatmap_max_presentations.setToolTip(
            "Maximum usable presentations to display. Extraction scans later matches so "
            "trials removed by rejection or QC can be backfilled."
        )
        controls.addWidget(self.spb_parcel_heatmap_max_presentations, 3, 5)

        self.btn_parcel_heatmap_compute = QtWidgets.QPushButton("Compute parcel heatmaps")
        self.btn_parcel_heatmap_compute.clicked.connect(self._compute_parcel_trial_heatmaps)
        controls.addWidget(self.btn_parcel_heatmap_compute, 4, 4, 1, 2)

        controls.addWidget(QtWidgets.QLabel("Compare with"), 4, 2)
        self.cmb_parcel_heatmap_comparison = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_comparison.setToolTip(
            "Optionally extract a second stimulus with identical settings and plot the signed "
            "selected-minus-comparison mean contrast."
        )
        controls.addWidget(self.cmb_parcel_heatmap_comparison, 4, 3)
        controls.addWidget(QtWidgets.QLabel("Parcel order"), 5, 0)
        self.cmb_parcel_heatmap_row_order = QtWidgets.QComboBox()
        self.cmb_parcel_heatmap_row_order.addItem(
            "Hemisphere → region", "anatomical"
        )
        self.cmb_parcel_heatmap_row_order.addItem("Atlas order", "atlas")
        self.cmb_parcel_heatmap_row_order.setToolTip(
            "Group heatmap rows first by hemisphere and then by a broad anatomical or "
            "functional region. Atlas order preserves the source atlas sequence."
        )
        self.cmb_parcel_heatmap_row_order.currentIndexChanged.connect(
            self._render_parcel_trial_heatmaps
        )
        controls.addWidget(self.cmb_parcel_heatmap_row_order, 5, 1)
        contrast_note = QtWidgets.QLabel(
            "Optional signed contrast; both groups use identical preprocessing"
        )
        contrast_note.setWordWrap(True)
        controls.addWidget(contrast_note, 5, 2, 1, 4)

        layout.addLayout(controls)
        self.lbl_parcel_heatmap_status = QtWidgets.QLabel(
            "Select a repeated stimulus, then compute its 0–400 ms parcel patterns."
        )
        self.lbl_parcel_heatmap_status.setWordWrap(True)
        layout.addWidget(self.lbl_parcel_heatmap_status)

        self.parcel_heatmap_dashboard_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.parcel_heatmap_dashboard_splitter.setChildrenCollapsible(False)
        self.parcel_heatmap_top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.parcel_heatmap_top_splitter.setChildrenCollapsible(False)
        self.parcel_heatmap_bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.parcel_heatmap_bottom_splitter.setChildrenCollapsible(False)

        self.parcel_heatmap_plot_container = QtWidgets.QWidget()
        self.parcel_heatmap_plot_layout = QtWidgets.QVBoxLayout()
        self.parcel_heatmap_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.parcel_heatmap_plot_layout.setSpacing(0)
        self.parcel_heatmap_plot_container.setLayout(self.parcel_heatmap_plot_layout)
        self.lbl_parcel_heatmap_placeholder = QtWidgets.QLabel(
            "No parcel heatmaps computed. The shared color scale will be identical across presentations."
        )
        self.lbl_parcel_heatmap_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.parcel_heatmap_plot_layout.addWidget(self.lbl_parcel_heatmap_placeholder)

        self.parcel_summary_plot_container = QtWidgets.QWidget()
        self.parcel_summary_plot_layout = QtWidgets.QVBoxLayout()
        self.parcel_summary_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.parcel_summary_plot_layout.setSpacing(0)
        self.parcel_summary_plot_container.setLayout(self.parcel_summary_plot_layout)
        self.lbl_parcel_summary_placeholder = QtWidgets.QLabel(
            "Cross-presentation mean and sign-consistency maps appear here."
        )
        self.lbl_parcel_summary_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_parcel_summary_placeholder.setWordWrap(True)
        self.parcel_summary_plot_layout.addWidget(self.lbl_parcel_summary_placeholder)

        self.parcel_contrast_plot_container = QtWidgets.QWidget()
        self.parcel_contrast_plot_layout = QtWidgets.QVBoxLayout()
        self.parcel_contrast_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.parcel_contrast_plot_layout.setSpacing(0)
        self.parcel_contrast_plot_container.setLayout(self.parcel_contrast_plot_layout)
        self.lbl_parcel_contrast_placeholder = QtWidgets.QLabel(
            "Select Compare with and compute to show a contrast map."
        )
        self.lbl_parcel_contrast_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_parcel_contrast_placeholder.setWordWrap(True)
        self.parcel_contrast_plot_layout.addWidget(self.lbl_parcel_contrast_placeholder)

        self.grp_parcel_ranking = QtWidgets.QGroupBox("Parcel decoding candidates")
        ranking_layout = QtWidgets.QVBoxLayout()
        ranking_layout.setContentsMargins(6, 6, 6, 6)
        ranking_layout.setSpacing(4)
        self.grp_parcel_ranking.setLayout(ranking_layout)
        self.lbl_parcel_ranking_note = QtWidgets.QLabel(
            "Peak interval and contrast use all presentations. Nested LOO balanced accuracy "
            "selects the strongest time bin inside each training fold."
        )
        self.lbl_parcel_ranking_note.setWordWrap(True)
        ranking_layout.addWidget(self.lbl_parcel_ranking_note)
        self.tbl_parcel_ranking = QtWidgets.QTableView()
        self.tbl_parcel_ranking.setAlternatingRowColors(True)
        self.tbl_parcel_ranking.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_parcel_ranking.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_parcel_ranking.setMinimumHeight(120)
        self.tbl_parcel_ranking.verticalHeader().setVisible(False)
        ranking_layout.addWidget(self.tbl_parcel_ranking)
        self.grp_parcel_ranking.setVisible(False)

        self.parcel_heatmap_top_splitter.addWidget(self.parcel_heatmap_plot_container)
        self.parcel_heatmap_top_splitter.addWidget(self.parcel_summary_plot_container)
        self.parcel_heatmap_top_splitter.setStretchFactor(0, 3)
        self.parcel_heatmap_top_splitter.setStretchFactor(1, 2)
        self.parcel_heatmap_top_splitter.setSizes([720, 480])
        self.parcel_heatmap_bottom_splitter.addWidget(self.parcel_contrast_plot_container)
        self.parcel_heatmap_bottom_splitter.addWidget(self.grp_parcel_ranking)
        self.parcel_heatmap_bottom_splitter.setStretchFactor(0, 3)
        self.parcel_heatmap_bottom_splitter.setStretchFactor(1, 2)
        self.parcel_heatmap_bottom_splitter.setSizes([720, 480])
        self.parcel_heatmap_bottom_splitter.setVisible(False)
        self.parcel_heatmap_dashboard_splitter.addWidget(self.parcel_heatmap_top_splitter)
        self.parcel_heatmap_dashboard_splitter.addWidget(self.parcel_heatmap_bottom_splitter)
        self.parcel_heatmap_dashboard_splitter.setStretchFactor(0, 2)
        self.parcel_heatmap_dashboard_splitter.setStretchFactor(1, 1)
        self.parcel_heatmap_dashboard_splitter.setSizes([620, 300])
        layout.addWidget(self.parcel_heatmap_dashboard_splitter, 1)

        self.tabs.addTab(tab, "Parcel trials")

    def _init_semantic_decoding_section(self, layout):
        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        controls.addWidget(QtWidgets.QLabel("Stimulus"), 0, 0)
        self.cmb_semantic_stimulus = QtWidgets.QComboBox()
        self.cmb_semantic_stimulus.currentIndexChanged.connect(self._on_semantic_preview_stimulus_changed)
        controls.addWidget(self.cmb_semantic_stimulus, 0, 1)

        controls.addWidget(QtWidgets.QLabel("Grouping"), 0, 2)
        self.cmb_semantic_grouping = QtWidgets.QComboBox()
        self.cmb_semantic_grouping.addItems(["Grouped (canonical)", "Grouped (category)", "Individual stimulus"])
        self.cmb_semantic_grouping.currentIndexChanged.connect(self._on_semantic_grouping_changed)
        controls.addWidget(self.cmb_semantic_grouping, 0, 3)

        controls.addWidget(QtWidgets.QLabel("Trial"), 0, 4)
        self.cmb_semantic_trial = QtWidgets.QComboBox()
        self.cmb_semantic_trial.currentIndexChanged.connect(self._on_semantic_preview_trial_changed)
        controls.addWidget(self.cmb_semantic_trial, 0, 5)

        controls.addWidget(QtWidgets.QLabel("Window start (s)"), 1, 0)
        self.spb_semantic_tmin = QtWidgets.QDoubleSpinBox()
        self.spb_semantic_tmin.setDecimals(3)
        self.spb_semantic_tmin.setRange(-1.0, 5.0)
        self.spb_semantic_tmin.setSingleStep(0.01)
        self.spb_semantic_tmin.setValue(0.10)
        controls.addWidget(self.spb_semantic_tmin, 1, 1)

        controls.addWidget(QtWidgets.QLabel("Window end (s)"), 1, 2)
        self.spb_semantic_tmax = QtWidgets.QDoubleSpinBox()
        self.spb_semantic_tmax.setDecimals(3)
        self.spb_semantic_tmax.setRange(-1.0, 5.0)
        self.spb_semantic_tmax.setSingleStep(0.01)
        self.spb_semantic_tmax.setValue(0.30)
        controls.addWidget(self.spb_semantic_tmax, 1, 3)

        controls.addWidget(QtWidgets.QLabel("Min trials/class"), 1, 4)
        self.spb_semantic_min_trials = QtWidgets.QSpinBox()
        self.spb_semantic_min_trials.setRange(2, 1000)
        self.spb_semantic_min_trials.setValue(2)
        controls.addWidget(self.spb_semantic_min_trials, 1, 5)

        controls.addWidget(QtWidgets.QLabel("Parcellations"), 2, 0)
        self.cmb_semantic_parcellations = CheckableTreeComboBox(self._semantic_parcellation_context)
        self.cmb_semantic_parcellations.setEditable(True)
        self.cmb_semantic_parcellations.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        controls.addWidget(self.cmb_semantic_parcellations, 2, 1, 1, 4)

        self.lbl_semantic_source_settings = QtWidgets.QLabel(
            "Choose parcellations, then run decoding. Uses Source time course settings: inverse operator, method, and SNR."
        )
        self.lbl_semantic_source_settings.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        controls.addWidget(self.lbl_semantic_source_settings, 3, 0, 1, 5)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self.btn_semantic_compute = QtWidgets.QPushButton("Compute / Refresh")
        self.btn_semantic_compute.clicked.connect(self._compute_semantic_preview)
        btn_row.addWidget(self.btn_semantic_compute)
        self.btn_semantic_decode = QtWidgets.QPushButton("Train")
        self.btn_semantic_decode.clicked.connect(self._run_semantic_decoding)
        btn_row.addWidget(self.btn_semantic_decode)
        btn_row.addStretch()
        controls.addLayout(btn_row, 3, 5)

        layout.addLayout(controls)

        self.lbl_semantic_status = QtWidgets.QLabel("Semantic decoding: no preview computed. Model not trained.")
        self.lbl_semantic_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_semantic_status)

        self.semantic_plot_container = QtWidgets.QWidget()
        self.semantic_plot_layout = QtWidgets.QVBoxLayout()
        self.semantic_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.semantic_plot_layout.setSpacing(0)
        self.semantic_plot_container.setLayout(self.semantic_plot_layout)
        self.lbl_semantic_placeholder = QtWidgets.QLabel(
            "Compute / Refresh to preview window-averaged source activation for the selected stimulus group."
        )
        self.lbl_semantic_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.semantic_plot_layout.addWidget(self.lbl_semantic_placeholder)
        layout.addWidget(self.semantic_plot_container)

        self.semantic_table_tabs = QtWidgets.QTabWidget()

        self.tbl_semantic_trials = QtWidgets.QTableView()
        self.tbl_semantic_trials.setSortingEnabled(True)
        self.tbl_semantic_trials.setAlternatingRowColors(True)
        self.semantic_table_tabs.addTab(self.tbl_semantic_trials, "Training data")

        self.tbl_semantic_metrics = QtWidgets.QTableView()
        self.tbl_semantic_metrics.setSortingEnabled(True)
        self.tbl_semantic_metrics.setAlternatingRowColors(True)
        self.semantic_table_tabs.addTab(self.tbl_semantic_metrics, "Metrics")

        self.tbl_semantic_features = QtWidgets.QTableView()
        self.tbl_semantic_features.setSortingEnabled(True)
        self.tbl_semantic_features.setAlternatingRowColors(True)
        self.semantic_table_tabs.addTab(self.tbl_semantic_features, "ROI features")
        layout.addWidget(self.semantic_table_tabs)

    def _toggle_concept_config_panel(self, checked):
        """Show or collapse the detailed concept decoding settings panel."""
        if hasattr(self, "concept_config_container"):
            self.concept_config_container.setVisible(checked)
        if hasattr(self, "btn_concept_config_toggle"):
            self.btn_concept_config_toggle.setText("Settings ▴" if checked else "Settings ▾")
            self.btn_concept_config_toggle.setToolTip(
                "Collapse settings to expand results view" if checked else "Expand settings configuration"
            )

    def _update_concept_summary_badge(self, *_args):
        """Update the compact settings summary text in the header bar."""
        if not hasattr(self, "lbl_concept_config_summary"):
            return
        try:
            grouping = getattr(self, "cmb_concept_grouping", None)
            grouping_txt = grouping.currentText() if grouping is not None else "Canonical concept"
            feat_space = getattr(self, "cmb_concept_feature_space", None)
            is_source = feat_space is not None and str(feat_space.currentData() or "") == "source"
            feat_txt = "Source parcels" if is_source else "Signal space"
            decoder = getattr(self, "cmb_concept_decoder", None)
            decoder_txt = decoder.currentText().split("(")[0].strip() if decoder is not None else "Logistic regression"
            tmin = self.spb_concept_tmin.value() if hasattr(self, "spb_concept_tmin") else 0.150
            tmax = self.spb_concept_tmax.value() if hasattr(self, "spb_concept_tmax") else 0.450
            split_pct = self.spb_concept_test_pct.value() if hasattr(self, "spb_concept_test_pct") else 20
            summary = f"{grouping_txt}  •  {feat_txt}  •  {decoder_txt}  •  {tmin:.3f}–{tmax:.3f} s  •  {split_pct}% test"
            self.lbl_concept_config_summary.setText(summary)
            self.lbl_concept_config_summary.setToolTip(
                f"Active Configuration:\n"
                f"• Concept labels: {grouping_txt}\n"
                f"• Feature space: {feat_txt}\n"
                f"• Decoder: {decoder_txt}\n"
                f"• Window: {tmin:.3f} to {tmax:.3f} s\n"
                f"• Test split: {split_pct}%"
            )
        except Exception:
            pass

    def _init_concept_decoding_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        tab.setLayout(layout)

        # ---------------------------------------------------------
        # Top Header & Action Toolbar
        # ---------------------------------------------------------
        header_card = QtWidgets.QFrame()
        header_card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        header_card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        header_layout = QtWidgets.QHBoxLayout(header_card)
        header_layout.setContentsMargins(6, 4, 6, 4)
        header_layout.setSpacing(8)

        title_label = QtWidgets.QLabel("Concept Decoding")
        title_font = title_label.font()
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setToolTip(
            "Extracts trial-locked MEG features across mapped FIF runs in signal space "
            "or source ROI space, then evaluates held-out concept decoding."
        )
        header_layout.addWidget(title_label)

        self.lbl_concept_status = QtWidgets.QLabel("Concept decoding: model not trained.")
        self.lbl_concept_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_concept_status.setStyleSheet(
            "QLabel { "
            "  background: rgba(120, 140, 160, 0.12); "
            "  border: 1px solid rgba(120, 140, 160, 0.28); "
            "  border-radius: 4px; "
            "  padding: 2px 8px; "
            "  font-size: 8.5pt; "
            "}"
        )
        header_layout.addWidget(self.lbl_concept_status)

        self.lbl_concept_config_summary = QtWidgets.QLabel()
        self.lbl_concept_config_summary.setStyleSheet("color: #666; font-size: 8pt;")
        self.lbl_concept_config_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        header_layout.addWidget(self.lbl_concept_config_summary, 1)

        self.btn_concept_decode = QtWidgets.QPushButton("Train / test decoder")
        self.btn_concept_decode.setToolTip("Train and test decoder on trial-locked MEG features.")
        self.btn_concept_decode.setStyleSheet("font-weight: 600; padding: 3px 12px;")
        self.btn_concept_decode.clicked.connect(self._run_concept_decoding)
        header_layout.addWidget(self.btn_concept_decode)

        self.btn_concept_save_outputs = QtWidgets.QPushButton("Save outputs")
        self.btn_concept_save_outputs.setEnabled(False)
        self.btn_concept_save_outputs.setToolTip(
            "Save the completed Concept decoding run settings, result tables, plots, and accuracy tracker."
        )
        self.btn_concept_save_outputs.clicked.connect(self._save_concept_outputs)
        header_layout.addWidget(self.btn_concept_save_outputs)

        self.btn_concept_config_toggle = QtWidgets.QPushButton("Settings ▴")
        self.btn_concept_config_toggle.setCheckable(True)
        self.btn_concept_config_toggle.setChecked(True)
        self.btn_concept_config_toggle.setToolTip("Collapse settings to expand results view")
        self.btn_concept_config_toggle.toggled.connect(self._toggle_concept_config_panel)
        header_layout.addWidget(self.btn_concept_config_toggle)

        layout.addWidget(header_card)

        # ---------------------------------------------------------
        # Collapsible Detailed Configuration Panel
        # ---------------------------------------------------------
        self.concept_config_container = QtWidgets.QFrame()
        self.concept_config_container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.concept_config_container.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        config_container_layout = QtWidgets.QVBoxLayout(self.concept_config_container)
        config_container_layout.setContentsMargins(4, 4, 4, 4)
        config_container_layout.setSpacing(4)

        self.concept_config_tabs = QtWidgets.QTabWidget()
        self.concept_config_tabs.setDocumentMode(True)
        self.concept_config_tabs.setStyleSheet(
            "QTabBar::tab { padding: 4px 12px; font-size: 8.5pt; }"
            "QTabBar::tab:selected { font-weight: 600; }"
        )

        # ---------------------------------------------------------
        # Tab 1: Setup & Models
        # ---------------------------------------------------------
        self.tab_config_setup = QtWidgets.QWidget()
        setup_grid = QtWidgets.QGridLayout(self.tab_config_setup)
        setup_grid.setContentsMargins(6, 6, 6, 6)
        setup_grid.setHorizontalSpacing(10)
        setup_grid.setVerticalSpacing(4)

        # Row 0: Labels & Feature Space
        setup_grid.addWidget(QtWidgets.QLabel("Concept labels"), 0, 0)
        self.cmb_concept_grouping = QtWidgets.QComboBox()
        self.cmb_concept_grouping.addItems(
            ["Canonical concept", "Stimulus category", "Individual stimulus", "Semantic decision task"]
        )
        self.cmb_concept_grouping.setItemData(
            3,
            "Uses line_2 from the nearest block-instructions.csv, matched to each trial by block_id.",
            QtCore.Qt.ToolTipRole,
        )
        setup_grid.addWidget(self.cmb_concept_grouping, 0, 1)

        setup_grid.addWidget(QtWidgets.QLabel("Feature space"), 0, 2)
        self.cmb_concept_feature_space = QtWidgets.QComboBox()
        self.cmb_concept_feature_space.addItem("Signal space", "signal")
        self.cmb_concept_feature_space.addItem("Source parcels", "source")
        self.cmb_concept_feature_space.setToolTip(
            "Source parcels use the Source atlas selector and the Source time course inverse settings."
        )
        setup_grid.addWidget(self.cmb_concept_feature_space, 0, 3)

        # Row 1: Decoder & Channels
        setup_grid.addWidget(QtWidgets.QLabel("Decoder"), 1, 0)
        self.cmb_concept_decoder = QtWidgets.QComboBox()
        self.cmb_concept_decoder.addItem("Logistic regression", "logistic")
        self.cmb_concept_decoder.addItem("Linear SVM", "svm")
        self.cmb_concept_decoder.addItem("Time-resolved pairwise SVM", "pairwise_svm")
        self.cmb_concept_decoder.addItem("CLIP image generation (image-specific ridge, experimental)", "clip_generation")
        self.cmb_concept_decoder.addItem("CLIP image generation (two-stage hybrid, experimental)", "clip_generation_hybrid")
        self.cmb_concept_decoder.addItem("Template correlation", "template")
        self.cmb_concept_decoder.addItem("Feed-forward neural net", "mlp")
        self.cmb_concept_decoder.addItem("Linear semantic embedding (ridge)", "embedding_ridge")
        self.cmb_concept_decoder.addItem(
            "Linear semantic embedding (contrastive)", "embedding_contrastive"
        )
        self.cmb_concept_decoder.addItem("Neural semantic embedding (MLP)", "embedding_mlp")
        self.cmb_concept_decoder.addItem("Transformer embedding", "transformer")
        self.cmb_concept_decoder.setToolTip(
            "Model used for held-out concept decoding. Semantic embedding models predict vectors "
            "such as Word2Vec and rank candidate concepts by cosine similarity."
        )

        decoder_box = QtWidgets.QWidget()
        decoder_layout = QtWidgets.QVBoxLayout(decoder_box)
        decoder_layout.setContentsMargins(0, 0, 0, 0)
        decoder_layout.setSpacing(2)
        decoder_layout.addWidget(self.cmb_concept_decoder)

        self.pairwise_controls = QtWidgets.QWidget()
        pairwise_form = QtWidgets.QFormLayout(self.pairwise_controls)
        pairwise_form.setContentsMargins(0, 0, 0, 0)
        self.spb_pairwise_size = QtWidgets.QSpinBox()
        self.spb_pairwise_size.setRange(1, 1000)
        self.spb_pairwise_size.setValue(6)
        self.spb_pairwise_repeats = QtWidgets.QSpinBox()
        self.spb_pairwise_repeats.setRange(1, 10000)
        self.spb_pairwise_repeats.setValue(100)
        pairwise_form.addRow("Trials / supertrial", self.spb_pairwise_size)
        pairwise_form.addRow("Random repeats", self.spb_pairwise_repeats)
        self.pairwise_controls.setToolTip(
            "Uses raw samples and leave-one-supertrial-out validation; requires twice the supertrial size per concept. Chance is 50%."
        )
        self.pairwise_controls.setVisible(False)
        decoder_layout.addWidget(self.pairwise_controls)
        setup_grid.addWidget(decoder_box, 1, 1)

        setup_grid.addWidget(QtWidgets.QLabel("Channels"), 1, 2)
        ch_row = QtWidgets.QHBoxLayout()
        ch_row.setContentsMargins(0, 0, 0, 0)
        ch_row.setSpacing(4)
        self.cmb_concept_channels = QtWidgets.QComboBox()
        self.cmb_concept_channels.addItems(["All MEG channels", "Magnetometers", "Gradiometers"])
        ch_row.addWidget(self.cmb_concept_channels, 2)
        ch_max_lbl = QtWidgets.QLabel("Max:")
        ch_row.addWidget(ch_max_lbl)
        self.spb_concept_max_channels = QtWidgets.QSpinBox()
        self.spb_concept_max_channels.setRange(0, 2000)
        self.spb_concept_max_channels.setSpecialValueText("All")
        self.spb_concept_max_channels.setValue(0)
        self.spb_concept_max_channels.setToolTip("Use 0 to keep all common MEG channels or all source ROIs.")
        ch_row.addWidget(self.spb_concept_max_channels, 1)
        setup_grid.addLayout(ch_row, 1, 3)

        # Row 2: Time Window & Split
        setup_grid.addWidget(QtWidgets.QLabel("Time window (s)"), 2, 0)
        window_row = QtWidgets.QHBoxLayout()
        window_row.setContentsMargins(0, 0, 0, 0)
        window_row.setSpacing(4)
        self.spb_concept_tmin = QtWidgets.QDoubleSpinBox()
        self.spb_concept_tmin.setDecimals(3)
        self.spb_concept_tmin.setRange(-1.0, 5.0)
        self.spb_concept_tmin.setSingleStep(0.01)
        self.spb_concept_tmin.setValue(0.150)
        self.spb_concept_tmin.setToolTip("Window start (seconds relative to trial onset)")
        window_row.addWidget(self.spb_concept_tmin, 1)
        window_sep = QtWidgets.QLabel("to")
        window_row.addWidget(window_sep)
        self.spb_concept_tmax = QtWidgets.QDoubleSpinBox()
        self.spb_concept_tmax.setDecimals(3)
        self.spb_concept_tmax.setRange(-1.0, 5.0)
        self.spb_concept_tmax.setSingleStep(0.01)
        self.spb_concept_tmax.setValue(0.450)
        self.spb_concept_tmax.setToolTip("Window end (seconds relative to trial onset)")
        window_row.addWidget(self.spb_concept_tmax, 1)
        setup_grid.addLayout(window_row, 2, 1)

        setup_grid.addWidget(QtWidgets.QLabel("Split & Test %"), 2, 2)
        split_row = QtWidgets.QHBoxLayout()
        split_row.setContentsMargins(0, 0, 0, 0)
        split_row.setSpacing(4)
        self.cmb_concept_split = QtWidgets.QComboBox()
        self.cmb_concept_split.addItem("Per-concept random", "random")
        self.cmb_concept_split.addItem("Zero-shot unseen concepts", "zero_shot")
        self.cmb_concept_split.addItem("Leave-one-run-out", "leave_one_run")
        self.cmb_concept_split.addItem("Leave-one-session-out", "leave_one_session")
        self.cmb_concept_split.setToolTip(
            "Split strategy: 'Per-concept random' partitions trials within each concept. "
            "'Zero-shot unseen concepts' completely holds out concepts to test zero-shot generalization. "
            "'Leave-one-run/session' tests generalization across recording runs or sessions."
        )
        split_row.addWidget(self.cmb_concept_split, 2)
        self.spb_concept_test_pct = QtWidgets.QSpinBox()
        self.spb_concept_test_pct.setRange(5, 50)
        self.spb_concept_test_pct.setSuffix("%")
        self.spb_concept_test_pct.setValue(20)
        self.spb_concept_test_pct.setToolTip("Percentage of trials per concept held out for testing.")
        split_row.addWidget(self.spb_concept_test_pct, 1)
        setup_grid.addLayout(split_row, 2, 3)

        # Row 3: Min trials & Permutations / Baseline
        setup_grid.addWidget(QtWidgets.QLabel("Min trials / concept"), 3, 0)
        self.spb_concept_min_trials = QtWidgets.QSpinBox()
        self.spb_concept_min_trials.setRange(2, 1000)
        self.spb_concept_min_trials.setValue(2)
        self.spb_concept_min_trials.setToolTip("Minimum number of presentations required to include a concept.")
        setup_grid.addWidget(self.spb_concept_min_trials, 3, 1)

        setup_grid.addWidget(QtWidgets.QLabel("Baseline & Perms"), 3, 2)
        base_perm_row = QtWidgets.QHBoxLayout()
        base_perm_row.setContentsMargins(0, 0, 0, 0)
        base_perm_row.setSpacing(8)
        self.chk_concept_baseline = QtWidgets.QCheckBox("Baseline (-100 to 0 ms)")
        self.chk_concept_baseline.setChecked(True)
        self.chk_concept_baseline.setToolTip("Sensor/epoch mean subtraction over -100 to 0 ms.")
        base_perm_row.addWidget(self.chk_concept_baseline)
        perms_lbl = QtWidgets.QLabel("Perms:")
        base_perm_row.addWidget(perms_lbl)
        self.spb_concept_permutations = QtWidgets.QSpinBox()
        self.spb_concept_permutations.setRange(0, 1000)
        self.spb_concept_permutations.setValue(99)
        self.spb_concept_permutations.setToolTip("Label-shuffle runs used to estimate empirical chance and p-values.")
        base_perm_row.addWidget(self.spb_concept_permutations)
        setup_grid.addLayout(base_perm_row, 3, 3)

        self.concept_config_tabs.addTab(self.tab_config_setup, "Setup & Models")

        # ---------------------------------------------------------
        # Tab 2: Source & Parcels
        # ---------------------------------------------------------
        self.tab_config_source = QtWidgets.QWidget()
        source_grid = QtWidgets.QGridLayout(self.tab_config_source)
        source_grid.setContentsMargins(6, 6, 6, 6)
        source_grid.setHorizontalSpacing(10)
        source_grid.setVerticalSpacing(4)

        # Row 0: Source space & Atlas
        source_grid.addWidget(QtWidgets.QLabel("Source space"), 0, 0)
        self.cmb_concept_source_space_kind = QtWidgets.QComboBox()
        self.cmb_concept_source_space_kind.addItem("Surface", "surface")
        self.cmb_concept_source_space_kind.addItem("Volume", "volume")
        self.cmb_concept_source_space_kind.setToolTip(
            "Source space used when Concept decoding Feature space is Source parcels."
        )
        self.cmb_concept_source_space_kind.currentIndexChanged.connect(self._on_concept_source_space_kind_changed)
        source_grid.addWidget(self.cmb_concept_source_space_kind, 0, 1)

        source_grid.addWidget(QtWidgets.QLabel("Source atlas"), 0, 2)
        self.cmb_concept_source_parcellation = QtWidgets.QComboBox()
        self.cmb_concept_source_parcellation.addItem("aparc", "aparc")
        self.cmb_concept_source_parcellation.addItem("aparc+aseg", "aparc+aseg")
        self.cmb_concept_source_parcellation.addItem("HCP-MMP", "HCPMMP1")
        self.cmb_concept_source_parcellation.addItem("HCP-MMP combined", "HCPMMP1_combined")
        self.cmb_concept_source_parcellation.setToolTip(
            "Atlas used when concept decoding Feature space is Source parcels. "
            "Volume source space uses aparc+aseg; HCP-MMP decodes in fsaverage surface common space."
        )
        self.cmb_concept_source_parcellation.currentIndexChanged.connect(self._on_concept_source_parcellation_changed)
        source_grid.addWidget(self.cmb_concept_source_parcellation, 0, 3)

        # Row 1: Parcels & Features
        source_grid.addWidget(QtWidgets.QLabel("Parcels"), 1, 0)
        parcel_row = QtWidgets.QHBoxLayout()
        parcel_row.setContentsMargins(0, 0, 0, 0)
        parcel_row.setSpacing(4)
        self.cmb_concept_parcels = CheckableTreeComboBox(self._semantic_parcellation_context)
        self.cmb_concept_parcels.setEditable(True)
        self.cmb_concept_parcels.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.cmb_concept_parcels.setToolTip(
            "Checked parcels limit source-space concept decoding. Leave all unchecked to use every parcel."
        )
        self._parcel_selection_in_progress = False
        self.cmb_concept_parcels.model.itemChanged.connect(
            self._on_concept_parcel_item_changed
        )
        parcel_row.addWidget(self.cmb_concept_parcels, 1)

        self.btn_concept_parcel_filter = QtWidgets.QToolButton()
        self.btn_concept_parcel_filter.setText("Filter ▾")
        self.btn_concept_parcel_filter.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_concept_parcel_filter.setToolTip(
            "Filter and select parcels by functional network (Visual, Semantic, Visual+Semantic, All, Clear)."
        )
        parcel_filter_menu = QtWidgets.QMenu(self.btn_concept_parcel_filter)
        act_vis = parcel_filter_menu.addAction("Select Visual parcels (Occipital & ventral stream)")
        act_vis.triggered.connect(lambda: self._select_concept_parcel_filter("visual"))
        act_sem = parcel_filter_menu.addAction("Select Semantic parcels (Temporal, parietal, IFG)")
        act_sem.triggered.connect(lambda: self._select_concept_parcel_filter("semantic"))
        act_both = parcel_filter_menu.addAction("Select Visual + Semantic parcels")
        act_both.triggered.connect(lambda: self._select_concept_parcel_filter("visual_semantic"))
        parcel_filter_menu.addSeparator()
        act_all = parcel_filter_menu.addAction("Select All parcels")
        act_all.triggered.connect(lambda: self._select_concept_parcel_filter("all"))
        act_clr = parcel_filter_menu.addAction("Clear selection")
        act_clr.triggered.connect(lambda: self._select_concept_parcel_filter("clear"))
        self.btn_concept_parcel_filter.setMenu(parcel_filter_menu)
        parcel_row.addWidget(self.btn_concept_parcel_filter)
        source_grid.addLayout(parcel_row, 1, 1)

        source_grid.addWidget(QtWidgets.QLabel("Features"), 1, 2)
        feature_layout = QtWidgets.QHBoxLayout()
        feature_layout.setContentsMargins(0, 0, 0, 0)
        feature_layout.setSpacing(4)
        self.cmb_concept_feature_mode = QtWidgets.QComboBox()
        self.cmb_concept_feature_mode.setToolTip(
            "For Source parcels, binned features average each parcel within the selected temporal "
            "bin width configured in Shared preprocessing. Raw samples keep every parcel time point."
        )
        self.cmb_concept_feature_mode.currentIndexChanged.connect(
            self._update_concept_source_preprocessing_controls
        )
        self.cmb_concept_power_band = QtWidgets.QComboBox()
        self.cmb_concept_power_band.addItem("All frequency bands", "all")
        from core.concept_decoding_spectral import SOURCE_POWER_BANDS
        for name, low, high in SOURCE_POWER_BANDS:
            self.cmb_concept_power_band.addItem(f"{name.replace('_', ' ').title()} ({low:g}–{high:g} Hz)", name)
        self.cmb_concept_power_band.setToolTip(
            "Source band-power decoding: select one band or all bands fully preserved by shared filtering and sampling."
        )
        feature_layout.addWidget(self.cmb_concept_feature_mode, 1)
        feature_layout.addWidget(self.cmb_concept_power_band, 1)
        source_grid.addLayout(feature_layout, 1, 3)

        # Row 2: Screening & Batch Actions
        source_grid.addWidget(QtWidgets.QLabel("Parcel screening"), 2, 0)
        screening_row = QtWidgets.QHBoxLayout()
        screening_row.setContentsMargins(0, 0, 0, 0)
        screening_row.setSpacing(8)
        self.chk_concept_semantic_subsets = QtWidgets.QCheckBox("Exhaustive subsets + auto-save")
        self.chk_concept_semantic_subsets.setToolTip(
            "Train all nonempty subsets of the explicitly checked parcels, using the selected atlas. "
            "Extract once and automatically save each fit and comparison table. Requires a source-space "
            "linear embedding decoder."
        )
        screening_row.addWidget(self.chk_concept_semantic_subsets)
        self.chk_concept_backward_elimination = QtWidgets.QCheckBox("Backward elimination + auto-save")
        self.chk_concept_backward_elimination.setToolTip(
            "Start with all checked parcels; retrain each one-parcel removal and retain the best "
            "validation score. Continue down to one parcel."
        )
        screening_row.addWidget(self.chk_concept_backward_elimination)
        self.chk_concept_backward_elimination.toggled.connect(
            lambda checked: self.chk_concept_semantic_subsets.setChecked(False) if checked else None)
        self.chk_concept_semantic_subsets.toggled.connect(
            lambda checked: self.chk_concept_backward_elimination.setChecked(False) if checked else None)
        source_grid.addLayout(screening_row, 2, 1)

        source_grid.addWidget(QtWidgets.QLabel("Batch execution"), 2, 2)
        batch_btn_row = QtWidgets.QHBoxLayout()
        batch_btn_row.setContentsMargins(0, 0, 0, 0)
        batch_btn_row.setSpacing(6)
        self.btn_concept_stop_batch = QtWidgets.QPushButton("Stop batch")
        self.btn_concept_stop_batch.setEnabled(False)
        self.btn_concept_stop_batch.setToolTip("Stop scheduling new subsets, finish active fits, and save completed results.")
        self.btn_concept_stop_batch.clicked.connect(self._stop_concept_subset_batch)
        batch_btn_row.addWidget(self.btn_concept_stop_batch)
        self.btn_concept_resume_batch = QtWidgets.QPushButton("Resume…")
        self.btn_concept_resume_batch.setToolTip(
            "Select a stopped exhaustive-screening output folder and continue from its saved checkpoint."
        )
        self.btn_concept_resume_batch.clicked.connect(self._resume_concept_subset_batch)
        batch_btn_row.addWidget(self.btn_concept_resume_batch)
        source_grid.addLayout(batch_btn_row, 2, 3)

        # Row 3: Parallel Workers
        source_grid.addWidget(QtWidgets.QLabel("Parallel screening"), 3, 0)
        worker_row = QtWidgets.QHBoxLayout()
        worker_row.setContentsMargins(0, 0, 0, 0)
        worker_row.setSpacing(6)
        worker_row.addWidget(QtWidgets.QLabel("Workers:"))
        self.spb_concept_subset_workers = QtWidgets.QSpinBox()
        self.spb_concept_subset_workers.setRange(1, max(1, os.cpu_count() or 1))
        self.spb_concept_subset_workers.setValue(1)
        self.spb_concept_subset_workers.setToolTip(
            "For exhaustive screening and Resume: 1 runs sequentially. "
            "More than 1 runs independent CPU fits in parallel."
        )
        worker_row.addWidget(self.spb_concept_subset_workers)
        worker_row.addWidget(QtWidgets.QLabel("Threads/worker:"))
        self.spb_concept_subset_worker_threads = QtWidgets.QSpinBox()
        self.spb_concept_subset_worker_threads.setRange(1, max(1, os.cpu_count() or 1))
        self.spb_concept_subset_worker_threads.setValue(1)
        self.spb_concept_subset_worker_threads.setToolTip(
            "CPU threads per parallel worker (BLAS and Torch). Applies when workers > 1."
        )
        worker_row.addWidget(self.spb_concept_subset_worker_threads)
        worker_note = QtWidgets.QLabel("(Workers > 1: independent CPU fits)")
        worker_note.setStyleSheet("color: #777; font-size: 8pt;")
        worker_row.addWidget(worker_note)
        source_grid.addLayout(worker_row, 3, 1, 1, 3)

        self.concept_config_tabs.addTab(self.tab_config_source, "Source & Parcels")

        # ---------------------------------------------------------
        # Tab 3: Embeddings & Neural
        # ---------------------------------------------------------
        self.tab_config_embeddings = QtWidgets.QWidget()
        embed_grid = QtWidgets.QGridLayout(self.tab_config_embeddings)
        embed_grid.setContentsMargins(6, 6, 6, 6)
        embed_grid.setHorizontalSpacing(10)
        embed_grid.setVerticalSpacing(4)

        # Row 0: Embeddings Path
        embed_grid.addWidget(QtWidgets.QLabel("Embeddings path"), 0, 0)
        emb_path_row = QtWidgets.QHBoxLayout()
        emb_path_row.setContentsMargins(0, 0, 0, 0)
        emb_path_row.setSpacing(4)
        self.le_concept_embedding_path = QtWidgets.QLineEdit()
        self.le_concept_embedding_path.setPlaceholderText(
            "Required for semantic decoders: Word2Vec/image/concept embedding dictionary (.json, .h5, .npz, .csv)"
        )
        self.le_concept_embedding_path.setToolTip(
            "Embedding dictionary keyed by stimulus/concept names, including exported Word2Vec maps."
        )
        emb_path_row.addWidget(self.le_concept_embedding_path, 1)
        self.btn_concept_embedding_browse = QtWidgets.QPushButton("Browse…")
        self.btn_concept_embedding_browse.clicked.connect(self._browse_concept_embedding_file)
        emb_path_row.addWidget(self.btn_concept_embedding_browse)
        embed_grid.addLayout(emb_path_row, 0, 1, 1, 3)

        # Row 1: Averaging & Epochs / Validation & Dropout
        embed_grid.addWidget(QtWidgets.QLabel("Averaging & Epochs"), 1, 0)
        avg_ep_row = QtWidgets.QHBoxLayout()
        avg_ep_row.setContentsMargins(0, 0, 0, 0)
        avg_ep_row.setSpacing(4)
        avg_ep_row.addWidget(QtWidgets.QLabel("Avg trials:"))
        self.spb_concept_avg_trials = QtWidgets.QSpinBox()
        self.spb_concept_avg_trials.setRange(1, 1000)
        self.spb_concept_avg_trials.setValue(5)
        self.spb_concept_avg_trials.setToolTip(
            "Semantic embedding decoders: average this many same-label trials after train/test splitting.\n"
            "• N=5: Strictly balances the 5 semantic decisions (1 trial from each task per pseudo-epoch).\n"
            "• N=6: Proportionally balances decisions (2 Animal + 1 each from Food, Vehicle, Instrument, Body Part; 100% trials retained).\n"
            "• Other N: Standard random within-class averaging."
        )
        self.spb_concept_transformer_avg_n = self.spb_concept_avg_trials  # backward-compatible alias
        avg_ep_row.addWidget(self.spb_concept_avg_trials)
        avg_ep_row.addWidget(QtWidgets.QLabel("Epochs:"))
        self.spb_concept_epochs = QtWidgets.QSpinBox()
        self.spb_concept_epochs.setRange(1, 1000)
        self.spb_concept_epochs.setValue(30)
        self.spb_concept_epochs.setToolTip(
            "Training epochs for contrastive linear, semantic MLP, and neural decoders."
        )
        self.spb_concept_transformer_epochs = self.spb_concept_epochs  # backward-compatible alias
        avg_ep_row.addWidget(self.spb_concept_epochs)
        embed_grid.addLayout(avg_ep_row, 1, 1)

        embed_grid.addWidget(QtWidgets.QLabel("Validation & Dropout"), 1, 2)
        val_drop_row = QtWidgets.QHBoxLayout()
        val_drop_row.setContentsMargins(0, 0, 0, 0)
        val_drop_row.setSpacing(4)
        val_drop_row.addWidget(QtWidgets.QLabel("Val split:"))
        self.spb_concept_validation_pct = QtWidgets.QSpinBox()
        self.spb_concept_validation_pct.setRange(0, 50)
        self.spb_concept_validation_pct.setSuffix("%")
        self.spb_concept_validation_pct.setValue(20)
        self.spb_concept_validation_pct.setToolTip(
            "Validation fraction taken from the training split for early stopping and epoch selection."
        )
        val_drop_row.addWidget(self.spb_concept_validation_pct)
        val_drop_row.addWidget(QtWidgets.QLabel("Dropout:"))
        self.spb_concept_mlp_dropout = QtWidgets.QDoubleSpinBox()
        self.spb_concept_mlp_dropout.setDecimals(2)
        self.spb_concept_mlp_dropout.setRange(0.0, 0.9)
        self.spb_concept_mlp_dropout.setSingleStep(0.05)
        self.spb_concept_mlp_dropout.setValue(0.35)
        self.spb_concept_mlp_dropout.setToolTip("Hidden-layer dropout probability used during training.")
        val_drop_row.addWidget(self.spb_concept_mlp_dropout)
        embed_grid.addLayout(val_drop_row, 1, 3)

        # Row 2: Input PCA & Semantic SVD Dims
        embed_grid.addWidget(QtWidgets.QLabel("Input PCA"), 2, 0)
        self.spb_concept_embedding_input_pca = QtWidgets.QSpinBox()
        self.spb_concept_embedding_input_pca.setRange(0, 4096)
        self.spb_concept_embedding_input_pca.setSpecialValueText("Off")
        self.spb_concept_embedding_input_pca.setValue(256)
        self.spb_concept_embedding_input_pca.setToolTip(
            "Semantic ridge, contrastive, and MLP decoders: fit this many PCA components using training pseudo-epochs."
        )
        embed_grid.addWidget(self.spb_concept_embedding_input_pca, 2, 1)

        embed_grid.addWidget(QtWidgets.QLabel("Semantic SVD dims"), 2, 2)
        self.spb_concept_embedding_target_svd = QtWidgets.QSpinBox()
        self.spb_concept_embedding_target_svd.setRange(0, 3072)
        self.spb_concept_embedding_target_svd.setSpecialValueText("Full")
        self.spb_concept_embedding_target_svd.setValue(40)
        self.spb_concept_embedding_target_svd.setToolTip(
            "Project centered candidate embeddings into a train-only SVD space capped by independent concept directions."
        )
        embed_grid.addWidget(self.spb_concept_embedding_target_svd, 2, 3)

        # Row 3: Early stopping & Dynamic pseudo-epochs
        embed_grid.addWidget(QtWidgets.QLabel("Early stopping patience"), 3, 0)
        pat_row = QtWidgets.QHBoxLayout()
        pat_row.setContentsMargins(0, 0, 0, 0)
        pat_row.setSpacing(4)
        self.spb_concept_embedding_patience = QtWidgets.QSpinBox()
        self.spb_concept_embedding_patience.setRange(0, 100)
        self.spb_concept_embedding_patience.setSpecialValueText("Off")
        self.spb_concept_embedding_patience.setValue(6)
        self.spb_concept_embedding_patience.setToolTip(
            "Stop contrastive or semantic MLP training after this many validation epochs without improvement."
        )
        pat_row.addWidget(self.spb_concept_embedding_patience)
        pat_row.addWidget(QtWidgets.QLabel("epochs"))
        embed_grid.addLayout(pat_row, 3, 1)

        embed_grid.addWidget(QtWidgets.QLabel("Training pseudo-epochs"), 3, 2)
        self.chk_concept_dynamic_pseudo_epochs = QtWidgets.QCheckBox(
            "Re-average training trials each epoch"
        )
        self.chk_concept_dynamic_pseudo_epochs.setChecked(True)
        self.chk_concept_dynamic_pseudo_epochs.setToolTip(
            "Create new within-class trial averages on every contrastive or semantic MLP epoch."
        )
        embed_grid.addWidget(self.chk_concept_dynamic_pseudo_epochs, 3, 3)

        # Row 4: Task-stratified pseudo-epochs
        embed_grid.addWidget(QtWidgets.QLabel("Task Stratification"), 4, 0)
        self.chk_concept_task_stratified_avg = QtWidgets.QCheckBox(
            "Stratify semantic decision tasks (N=5 or N=6)"
        )
        self.chk_concept_task_stratified_avg.setChecked(True)
        self.chk_concept_task_stratified_avg.setToolTip(
            "Cancel semantic decision task confounds when pseudo-epoch averaging.\n"
            "• N=5: Strict 5-way balance (1 trial each from all 5 semantic decision tasks).\n"
            "• N=6: Proportional 6-way balance (2 Animal + 1 each of Food, Vehicle, Instrument, Body Part; 100% data retention).\n"
            "• Other N (or when unchecked): Standard random within-class averaging."
        )
        embed_grid.addWidget(self.chk_concept_task_stratified_avg, 4, 1, 1, 3)

        self.concept_config_tabs.addTab(self.tab_config_embeddings, "Embeddings & Neural")

        # ---------------------------------------------------------
        # Tab 4: Image Generation (CLIP Experimental)
        # ---------------------------------------------------------
        self.tab_config_images = QtWidgets.QWidget()
        image_grid = QtWidgets.QGridLayout(self.tab_config_images)
        image_grid.setContentsMargins(6, 6, 6, 6)
        image_grid.setHorizontalSpacing(8)
        image_grid.setVerticalSpacing(6)

        clip_note = QtWidgets.QLabel(
            "CLIP Image Synthesis (Experimental): Synthesizes held-out stimulus images from "
            "predicted 768-dimensional visual vectors."
        )
        clip_note.setStyleSheet("color: #666; font-size: 8.5pt;")
        image_grid.addWidget(clip_note, 0, 0, 1, 4)

        img_btn_row = QtWidgets.QHBoxLayout()
        img_btn_row.setContentsMargins(0, 0, 0, 0)
        img_btn_row.setSpacing(6)

        self.btn_concept_generate_image = QtWidgets.QPushButton("Generate image…")
        self.btn_concept_generate_image.setToolTip(
            "Choose a held-out average from a compatible 768-dimensional CLIP model. "
            "Saves MEG and reference-vector images."
        )
        self.btn_concept_generate_image.clicked.connect(self._generate_concept_image)
        img_btn_row.addWidget(self.btn_concept_generate_image)

        self.btn_concept_open_images = QtWidgets.QPushButton("Open images folder")
        self.btn_concept_open_images.setEnabled(False)
        self.btn_concept_open_images.clicked.connect(self._open_concept_images_folder)
        img_btn_row.addWidget(self.btn_concept_open_images)

        self.btn_concept_generate_all = QtWidgets.QPushButton("Generate all / resume…")
        self.btn_concept_generate_all.setToolTip(
            "Save one comparison per prediction, including a third image generated after subtracting "
            "the contributing block concept vector."
        )
        self.btn_concept_generate_all.clicked.connect(lambda: self._generate_concept_image(batch=True))
        img_btn_row.addWidget(self.btn_concept_generate_all)

        self.btn_concept_stop_images = QtWidgets.QPushButton("Stop images")
        self.btn_concept_stop_images.setEnabled(False)
        self.btn_concept_stop_images.clicked.connect(self._stop_concept_images)
        img_btn_row.addWidget(self.btn_concept_stop_images)
        img_btn_row.addStretch()
        image_grid.addLayout(img_btn_row, 1, 0, 1, 4)

        prog_row = QtWidgets.QHBoxLayout()
        prog_row.setContentsMargins(0, 0, 0, 0)
        prog_row.setSpacing(6)
        prog_row.addWidget(QtWidgets.QLabel("Progress:"))
        self.concept_image_progress = QtWidgets.QProgressBar()
        self.concept_image_progress.setValue(0)
        prog_row.addWidget(self.concept_image_progress, 1)
        image_grid.addLayout(prog_row, 2, 0, 1, 4)

        self.concept_config_tabs.addTab(self.tab_config_images, "Image Generation")

        config_container_layout.addWidget(self.concept_config_tabs)
        layout.addWidget(self.concept_config_container)

        # Wire settings change events
        self.cmb_concept_feature_space.currentIndexChanged.connect(self._on_concept_feature_space_changed)
        self.cmb_concept_decoder.currentIndexChanged.connect(self._on_concept_decoder_changed)
        self.cmb_concept_decoder.currentIndexChanged.connect(
            lambda: self.pairwise_controls.setVisible(self.cmb_concept_decoder.currentData() == "pairwise_svm")
        )
        self.cmb_concept_grouping.currentIndexChanged.connect(self._update_concept_summary_badge)
        self.cmb_concept_split.currentIndexChanged.connect(self._update_concept_summary_badge)
        self.spb_concept_tmin.valueChanged.connect(self._update_concept_summary_badge)
        self.spb_concept_tmax.valueChanged.connect(self._update_concept_summary_badge)
        self.spb_concept_test_pct.valueChanged.connect(self._update_concept_summary_badge)

        self._on_concept_feature_space_changed()
        self._update_concept_source_preprocessing_controls()
        self._on_concept_decoder_changed()
        self._update_concept_summary_badge()

        self.concept_result_tabs = QtWidgets.QTabWidget()
        self.concept_result_tabs.setDocumentMode(True)
        self.concept_table_tabs = self.concept_result_tabs

        self.tbl_concept_summary = QtWidgets.QTableView()
        self.tbl_concept_summary.setSortingEnabled(True)
        self.tbl_concept_summary.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_summary, "Summary")

        self.tbl_concept_elimination_path = QtWidgets.QTableView()
        self.tbl_concept_elimination_path.setSortingEnabled(True)
        self.tbl_concept_elimination_path.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_elimination_path, "Elimination path")

        self.tbl_concept_metrics = QtWidgets.QTableView()
        self.tbl_concept_metrics.setSortingEnabled(True)
        self.tbl_concept_metrics.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_metrics, "Metrics")

        self.concept_training_tab = QtWidgets.QWidget()
        self.concept_training_layout = QtWidgets.QVBoxLayout()
        self.concept_training_layout.setContentsMargins(0, 0, 0, 0)
        self.concept_training_layout.setSpacing(0)
        self.concept_training_tab.setLayout(self.concept_training_layout)
        self.lbl_concept_training_placeholder = QtWidgets.QLabel(
            "Select Feed-forward neural net and start training to show live accuracy curves."
        )
        self.lbl_concept_training_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_concept_training_placeholder.setWordWrap(True)
        self.concept_training_layout.addWidget(self.lbl_concept_training_placeholder)
        self.concept_result_tabs.addTab(self.concept_training_tab, "Training curve")

        self.concept_confusion_tab = QtWidgets.QWidget()
        self.concept_confusion_layout = QtWidgets.QVBoxLayout()
        self.concept_confusion_layout.setContentsMargins(0, 0, 0, 0)
        self.concept_confusion_layout.setSpacing(0)
        self.concept_confusion_tab.setLayout(self.concept_confusion_layout)
        self.lbl_concept_confusion_placeholder = QtWidgets.QLabel(
            "Train / test the decoder to show the held-out confusion matrix."
        )
        self.lbl_concept_confusion_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.concept_confusion_layout.addWidget(self.lbl_concept_confusion_placeholder)
        self.concept_result_tabs.addTab(self.concept_confusion_tab, "Confusion matrix")

        self.concept_accuracy_tab = QtWidgets.QWidget()
        self.concept_accuracy_layout = QtWidgets.QVBoxLayout()
        self.concept_accuracy_layout.setContentsMargins(0, 0, 0, 0)
        self.concept_accuracy_layout.setSpacing(0)
        self.concept_accuracy_tab.setLayout(self.concept_accuracy_layout)
        self.lbl_concept_accuracy_placeholder = QtWidgets.QLabel(
            "Train / test the decoder to show per-concept held-out accuracy."
        )
        self.lbl_concept_accuracy_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.concept_accuracy_layout.addWidget(self.lbl_concept_accuracy_placeholder)
        self.concept_result_tabs.addTab(self.concept_accuracy_tab, "Per-concept accuracy")

        self.tbl_concept_predictions = QtWidgets.QTableView()
        self.tbl_concept_predictions.setSortingEnabled(True)
        self.tbl_concept_predictions.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_predictions, "Test predictions")

        self.concept_trial_qc_tab = QtWidgets.QWidget()
        self.concept_trial_qc_layout = QtWidgets.QVBoxLayout()
        self.concept_trial_qc_layout.setContentsMargins(0, 0, 0, 0)
        self.concept_trial_qc_layout.setSpacing(4)
        self.concept_trial_qc_tab.setLayout(self.concept_trial_qc_layout)
        self.lbl_concept_trial_qc_summary = QtWidgets.QLabel("Trial QC counts unavailable.")
        self.lbl_concept_trial_qc_summary.setWordWrap(True)
        self.concept_trial_qc_layout.addWidget(self.lbl_concept_trial_qc_summary)
        self.concept_trial_qc_plot_container = QtWidgets.QWidget()
        self.concept_trial_qc_plot_layout = QtWidgets.QVBoxLayout()
        self.concept_trial_qc_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.concept_trial_qc_plot_layout.setSpacing(0)
        self.concept_trial_qc_plot_container.setLayout(self.concept_trial_qc_plot_layout)
        self.lbl_concept_trial_qc_placeholder = QtWidgets.QLabel(
            "Train / test the decoder to show trial quality metrics and outliers."
        )
        self.lbl_concept_trial_qc_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.concept_trial_qc_plot_layout.addWidget(self.lbl_concept_trial_qc_placeholder)
        self.concept_trial_qc_layout.addWidget(self.concept_trial_qc_plot_container, 2)
        self.tbl_concept_trial_qc = QtWidgets.QTableView()
        self.tbl_concept_trial_qc.setSortingEnabled(True)
        self.tbl_concept_trial_qc.setAlternatingRowColors(True)
        self.concept_trial_qc_layout.addWidget(self.tbl_concept_trial_qc, 1)
        self.concept_result_tabs.addTab(self.concept_trial_qc_tab, "Trial QC")

        self.tbl_concept_qc_summary = QtWidgets.QTableView()
        self.tbl_concept_qc_summary.setSortingEnabled(True)
        self.tbl_concept_qc_summary.setAlternatingRowColors(True)
        self.concept_qc_summary_tab = QtWidgets.QWidget()
        qc_summary_layout = QtWidgets.QVBoxLayout(self.concept_qc_summary_tab)
        qc_summary_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_concept_qc_summary = QtWidgets.QLabel("Trial QC counts unavailable.")
        self.lbl_concept_qc_summary.setWordWrap(True)
        qc_summary_layout.addWidget(self.lbl_concept_qc_summary)
        qc_summary_layout.addWidget(self.tbl_concept_qc_summary)
        self.concept_result_tabs.addTab(self.concept_qc_summary_tab, "QC summary")

        self.tbl_concept_features = QtWidgets.QTableView()
        self.tbl_concept_features.setSortingEnabled(True)
        self.tbl_concept_features.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_features, "Feature importance")

        self.tbl_concept_split = QtWidgets.QTableView()
        self.tbl_concept_split.setSortingEnabled(True)
        self.tbl_concept_split.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_split, "Split by concept")

        self.tbl_concept_permutation = QtWidgets.QTableView()
        self.tbl_concept_permutation.setSortingEnabled(True)
        self.tbl_concept_permutation.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_permutation, "Permutation baseline")

        self.concept_templates_tab = QtWidgets.QWidget()
        self.concept_templates_layout = QtWidgets.QVBoxLayout()
        self.concept_templates_layout.setContentsMargins(0, 0, 0, 0)
        self.concept_templates_layout.setSpacing(4)
        self.concept_templates_tab.setLayout(self.concept_templates_layout)
        self.concept_templates_plot_container = QtWidgets.QWidget()
        self.concept_templates_plot_layout = QtWidgets.QVBoxLayout()
        self.concept_templates_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.concept_templates_plot_layout.setSpacing(0)
        self.concept_templates_plot_container.setLayout(self.concept_templates_plot_layout)
        self.lbl_concept_templates_placeholder = QtWidgets.QLabel(
            "Use Template correlation to show learned concept templates."
        )
        self.lbl_concept_templates_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.concept_templates_plot_layout.addWidget(self.lbl_concept_templates_placeholder)
        self.concept_templates_layout.addWidget(self.concept_templates_plot_container, 2)
        self.tbl_concept_templates = QtWidgets.QTableView()
        self.tbl_concept_templates.setSortingEnabled(True)
        self.tbl_concept_templates.setAlternatingRowColors(True)
        self.concept_templates_layout.addWidget(self.tbl_concept_templates, 1)
        self.concept_result_tabs.addTab(self.concept_templates_tab, "Templates")

        self.tbl_concept_architecture = QtWidgets.QTableView()
        self.tbl_concept_architecture.setSortingEnabled(True)
        self.tbl_concept_architecture.setAlternatingRowColors(True)
        self.concept_result_tabs.addTab(self.tbl_concept_architecture, "Model architecture")

        # Polish tables for compact professional look
        for tbl in (
            self.tbl_concept_summary,
            self.tbl_concept_elimination_path,
            self.tbl_concept_metrics,
            self.tbl_concept_predictions,
            self.tbl_concept_trial_qc,
            self.tbl_concept_qc_summary,
            self.tbl_concept_features,
            self.tbl_concept_split,
            self.tbl_concept_permutation,
            self.tbl_concept_templates,
            self.tbl_concept_architecture,
        ):
            tbl.verticalHeader().setDefaultSectionSize(22)
            tbl.verticalHeader().setVisible(False)
            tbl.horizontalHeader().setHighlightSections(False)
            tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        # Polish empty state placeholders
        placeholder_style = "QLabel { color: #888; font-style: italic; padding: 20px; font-size: 9pt; }"
        self.lbl_concept_training_placeholder.setStyleSheet(placeholder_style)
        self.lbl_concept_confusion_placeholder.setStyleSheet(placeholder_style)
        self.lbl_concept_accuracy_placeholder.setStyleSheet(placeholder_style)
        self.lbl_concept_trial_qc_placeholder.setStyleSheet(placeholder_style)
        self.lbl_concept_templates_placeholder.setStyleSheet(placeholder_style)

        layout.addWidget(self.concept_result_tabs, 1)
        self.tabs.addTab(tab, "Concept decoding")

    @staticmethod
    def _set_all_channel_selection(channel_control, selected):
        if channel_control is None:
            return
        model = getattr(channel_control, "model", None)
        if model is not None:
            state = QtCore.Qt.Checked if selected else QtCore.Qt.Unchecked
            root = model.invisibleRootItem()
            for row in range(root.rowCount()):
                item = root.child(row)
                if item is not None:
                    item.setCheckState(state)
            channel_control.updateText()
            return
        for i in range(channel_control.count()):
            item = channel_control.item(i)
            if item is not None:
                item.setSelected(bool(selected))

    @staticmethod
    def _selected_channels(channel_control):
        if channel_control is None:
            return []
        if hasattr(channel_control, "getCheckedItems"):
            return list(channel_control.getCheckedItems())
        return [item.text() for item in channel_control.selectedItems() if item is not None]

    @classmethod
    def _selected_data_channels(cls, channel_control):
        """Return selected sensor channels, excluding the virtual event marker."""
        return [
            channel
            for channel in cls._selected_channels(channel_control)
            if channel != MEG_EVENTS_CHANNEL_LABEL
        ]

    @classmethod
    def _events_selected(cls, channel_control):
        """Return whether event-onset markers are enabled in a channel picker."""
        return MEG_EVENTS_CHANNEL_LABEL in cls._selected_channels(channel_control)

    @staticmethod
    def _set_checked_channels(channel_control, channel_names):
        """Check exactly the requested leaf channels in a tree picker."""
        if channel_control is None:
            return
        targets = {str(name) for name in channel_names}
        model = getattr(channel_control, "model", None)
        if model is None:
            for index in range(channel_control.count()):
                item = channel_control.item(index)
                if item is not None:
                    item.setSelected(item.text() in targets)
            return

        def update_item(item):
            if item.hasChildren():
                for row in range(item.rowCount()):
                    update_item(item.child(row))
            else:
                item.setCheckState(
                    QtCore.Qt.Checked if item.text() in targets else QtCore.Qt.Unchecked
                )

        root = model.invisibleRootItem()
        for row in range(root.rowCount()):
            update_item(root.child(row))
        channel_control.updateText()

    def _remove_plot(self, key):
        state = self._plot_state.get(key, {})
        toolbars = state.get("toolbars", [])
        canvases = state.get("canvases", [])
        figs = state.get("figs", [])

        # Backward compatibility with old single-figure state keys.
        if not toolbars and state.get("toolbar") is not None:
            toolbars = [state.get("toolbar")]
        if not canvases and state.get("canvas") is not None:
            canvases = [state.get("canvas")]
        if not figs and state.get("fig") is not None:
            figs = [state.get("fig")]

        for toolbar in toolbars:
            if toolbar is None:
                continue
            toolbar.setParent(None)
            toolbar.deleteLater()
        for canvas in canvases:
            if canvas is None:
                continue
            canvas.setParent(None)
            canvas.deleteLater()
        for fig in figs:
            if fig is None:
                continue
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        self._plot_state[key] = {}

    def _set_plot(self, key, container_layout, placeholder_label, fig):
        self._remove_plot(key)
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
                NavigationToolbar2QT as NavigationToolbar,
            )
        except Exception:
            from matplotlib.backends.backend_qt5agg import (  # type: ignore
                FigureCanvasQTAgg as FigureCanvas,
                NavigationToolbar2QT as NavigationToolbar,
            )

        figures = list(fig) if isinstance(fig, (list, tuple)) else [fig]
        figures = [f for f in figures if f is not None]
        if not figures:
            return

        if placeholder_label is not None:
            placeholder_label.setVisible(False)
        canvases = []
        toolbars = []
        for one_fig in figures:
            canvas = FigureCanvas(one_fig)
            toolbar = NavigationToolbar(canvas, container_layout.parentWidget())
            container_layout.addWidget(toolbar)
            container_layout.addWidget(canvas)
            canvas.draw_idle()
            canvases.append(canvas)
            toolbars.append(toolbar)
        self._plot_state[key] = {"canvases": canvases, "toolbars": toolbars, "figs": figures}

    def _reset_plot_placeholder(self, key, placeholder_label, text):
        self._remove_plot(key)
        if placeholder_label is not None:
            placeholder_label.setText(str(text))
            placeholder_label.setVisible(True)

    def _reset_parcel_heatmap_dashboard(self, pattern_text):
        """Clear every pane in the Parcel trials dashboard."""
        self._reset_plot_placeholder(
            "parcel_heatmaps", self.lbl_parcel_heatmap_placeholder, pattern_text
        )
        self._reset_plot_placeholder(
            "parcel_heatmap_summaries",
            self.lbl_parcel_summary_placeholder,
            "Cross-presentation mean and sign-consistency maps appear here.",
        )
        self._reset_plot_placeholder(
            "parcel_heatmap_contrast",
            self.lbl_parcel_contrast_placeholder,
            "Select Compare with and compute to show a contrast map.",
        )
        self._clear_parcel_ranking_table()
        self.parcel_heatmap_bottom_splitter.setVisible(False)

    def clear(self, message="No MEG analysis loaded."):
        self._trial_df = pd.DataFrame()
        self._metadata = {}
        self._raw = None
        self._epochs = None
        self._topomap_epochs_by_path = {}
        self._topomap_run_meta_by_path = {}
        self._topomap_run_order = []
        self._evokeds = {}
        self._primary_event = None
        self._primary_evoked = None
        self._fif_filename = None
        self._source_stc = None
        self._source_src = None
        self._source_status_text = ""
        self._source_subjects_dir = ""
        self._source_subject = ""
        self._source_time_index = None
        self._source_time_marker_line = None
        self._source_time_marker_canvas = None
        self._source_plot_has_brain_image = False
        self._semantic_preview_result = None
        self._semantic_train_result = None
        self._concept_decode_result = None
        self._parcel_heatmap_result = None
        self._concept_decode_last_options = {}
        self._semantic_preview_status_text = ""
        self._semantic_train_status_text = ""
        self._concept_decode_status_text = ""
        self.stimulus_report.clear(message)
        self.status_label.setText(str(message))
        self._refresh_memory_telemetry()
        self.lbl_source_status.setText("Source status: no source estimate available.")
        self.lbl_semantic_status.setText("Semantic decoding: no preview computed. Model not trained.")
        self.lbl_concept_status.setText("Concept decoding: model not trained.")
        if getattr(self, "btn_concept_save_outputs", None) is not None:
            self.btn_concept_save_outputs.setEnabled(False)
        self.cmb_semantic_stimulus.clear()
        self.cmb_semantic_trial.clear()
        if getattr(self, "cmb_semantic_parcellations", None) is not None:
            self.cmb_semantic_parcellations.model.clear()
            self.cmb_semantic_parcellations.channel_dict = {}
            self.cmb_semantic_parcellations.setEditText("")
        if getattr(self, "cmb_concept_parcels", None) is not None:
            self.cmb_concept_parcels.model.clear()
            self.cmb_concept_parcels.channel_dict = {}
            self.cmb_concept_parcels.setEditText("")
        self._concept_available_parcels = []
        self.cmb_epoch_stimulus.clear()
        self.cmb_evoked_stimulus.clear()
        self.cmb_topomap_stimulus.clear()
        self.cmb_frequency_stimulus.clear()
        if getattr(self, "cmb_topomap_run", None) is not None:
            self.cmb_topomap_run.clear()
        if getattr(self, "cmb_source_run", None) is not None:
            self.cmb_source_run.clear()
        if getattr(self, "cmb_parcel_heatmap_run", None) is not None:
            self.cmb_parcel_heatmap_run.clear()
        if getattr(self, "cmb_parcel_heatmap_stimulus", None) is not None:
            self.cmb_parcel_heatmap_stimulus.clear()
        self.cmb_source_evoked.clear()
        self.lst_epoch_channels.clear()
        self.lst_evoked_channels.clear()
        self.lst_frequency_channels.clear()
        self.le_source_inverse.setText("")
        self.lbl_source_subjects_dir.setText("")
        self.lbl_source_subject.setText("")
        self._update_source_time_controls()
        self.lbl_topomap_info.setText("")
        self._reset_plot_placeholder("epoch", self.lbl_epoch_placeholder, "Load a .fif file to display epoch images.")
        self._reset_plot_placeholder("evoked", self.lbl_evoked_placeholder, "Load a .fif file to display evoked responses.")
        self._reset_plot_placeholder("topomap", self.lbl_topomap_placeholder, "Load a .fif file to display topomaps.")
        self.lbl_frequency_info.setText(
            "PSD highlights broadband shape and 60 Hz contamination; condition spectra and TFR can reveal candidate decoding features."
        )
        self._reset_plot_placeholder(
            "frequency",
            self.lbl_frequency_placeholder,
            "Load a .fif file, then plot spectral QC or time-frequency power.",
        )
        self._reset_plot_placeholder("source", self.lbl_source_placeholder, "Compute inverse source estimate to view source time course.")
        self._reset_parcel_heatmap_dashboard(
            "No parcel heatmaps computed. Load a FIF run and select a repeated stimulus.",
        )
        self.lbl_parcel_heatmap_status.setText(
            "Select a repeated stimulus, then compute its 0–400 ms parcel patterns."
        )
        self._reset_plot_placeholder(
            "semantic",
            self.lbl_semantic_placeholder,
            "Compute / Refresh to preview window-averaged source activation for the selected stimulus group.",
        )
        self._reset_plot_placeholder(
            "concept_confusion",
            self.lbl_concept_confusion_placeholder,
            "Train / test the decoder to show the held-out confusion matrix.",
        )
        self._reset_plot_placeholder(
            "concept_accuracy",
            self.lbl_concept_accuracy_placeholder,
            "Train / test the decoder to show per-concept held-out accuracy.",
        )
        self._reset_concept_training_curve(
            "Select Feed-forward neural net and start training to show live accuracy curves."
        )
        self._reset_plot_placeholder(
            "concept_templates",
            self.lbl_concept_templates_placeholder,
            "Use Template correlation to show learned concept templates.",
        )
        self._reset_plot_placeholder(
            "concept_trial_qc",
            self.lbl_concept_trial_qc_placeholder,
            "Train / test the decoder to show trial quality metrics and outliers.",
        )
        self.tbl_semantic_trials.setModel(None)
        self.tbl_semantic_metrics.setModel(None)
        self.tbl_semantic_features.setModel(None)
        self.tbl_concept_summary.setModel(None)
        self.tbl_concept_elimination_path.setModel(None)
        self.tbl_concept_metrics.setModel(None)
        self.tbl_concept_predictions.setModel(None)
        self.tbl_concept_features.setModel(None)
        self.tbl_concept_split.setModel(None)
        self.tbl_concept_permutation.setModel(None)
        self.tbl_concept_templates.setModel(None)
        self.tbl_concept_trial_qc.setModel(None)
        self.tbl_concept_qc_summary.setModel(None)
        self.lbl_concept_trial_qc_summary.setText("Trial QC counts unavailable.")
        self.lbl_concept_qc_summary.setText("Trial QC counts unavailable.")
        self.parcel_ranking_model = None
        self.tbl_parcel_ranking.setModel(None)
        self.grp_parcel_ranking.setVisible(False)

    def set_trial_data(self, trial_df, metadata=None):
        if metadata is None:
            metadata = {}
        self._metadata = dict(metadata)
        if isinstance(trial_df, pd.DataFrame):
            self._trial_df = trial_df.copy()
        else:
            self._trial_df = pd.DataFrame()
        self.stimulus_report.set_trial_data(self._trial_df, metadata=self._metadata)
        self._refresh_stimulus_selectors()
        if getattr(self, "cmb_source_run", None) is not None:
            self._refresh_source_run_selector()
            self._refresh_source_evoked_selector()
        if getattr(self, "cmb_parcel_heatmap_run", None) is not None:
            self._refresh_parcel_heatmap_run_selector()
            self._refresh_parcel_heatmap_stimulus_selector()

    def get_summary_df(self):
        return self.stimulus_report.get_summary_df()

    def _analysis_trial_df(self):
        """Return the best available trial table for analysis controls."""
        if isinstance(self._trial_df, pd.DataFrame) and not self._trial_df.empty:
            return self._trial_df
        report = getattr(self, "stimulus_report", None)
        for attr in ("_trial_df_raw", "_trial_df_current"):
            df = getattr(report, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return pd.DataFrame()

    def _analysis_metadata(self):
        """Return merged metadata from this panel and the embedded stimulus report."""
        meta = {}
        report = getattr(self, "stimulus_report", None)
        report_meta = getattr(report, "_metadata", None)
        if isinstance(report_meta, dict):
            meta.update(report_meta)
        if isinstance(self._metadata, dict):
            meta.update(self._metadata)
        return meta

    def set_mne_context(
        self,
        raw=None,
        epochs=None,
        evokeds=None,
        primary_event=None,
        primary_evoked=None,
        fif_filename=None,
        topomap_epochs_by_path=None,
        topomap_run_meta_by_path=None,
    ):
        self._raw = raw
        self._epochs = epochs
        self._evokeds = dict(evokeds) if isinstance(evokeds, dict) else {}
        self._primary_event = primary_event
        self._primary_evoked = primary_evoked
        self._fif_filename = str(fif_filename) if fif_filename else None
        self._set_topomap_epoch_runs(
            epochs=epochs,
            fif_filename=fif_filename,
            topomap_epochs_by_path=topomap_epochs_by_path,
            topomap_run_meta_by_path=topomap_run_meta_by_path,
        )
        self._refresh_channel_selectors()
        self._refresh_stimulus_selectors()
        self._refresh_topomap_run_selector()
        self._refresh_topomap_stimulus_selector()
        self._on_topomap_stimulus_changed()
        self._update_topomap_slider_range()
        self._refresh_source_run_selector()
        self._refresh_source_evoked_selector()
        self._refresh_parcel_heatmap_run_selector()
        self._refresh_parcel_heatmap_stimulus_selector()
        n_epochs = int(len(epochs)) if epochs is not None else 0
        n_evoked = int(len(self._evokeds))
        self.status_label.setText(
            f"MEG analysis ready: epochs={n_epochs}, conditions={n_evoked}, file={os.path.basename(str(self._fif_filename or 'n/a'))}"
        )
        self._refresh_memory_telemetry()

    def set_source_compute_callback(self, callback):
        """Set callback returning `(stc, message[, metadata])`."""
        self._source_compute_callback = callback

    def set_parcel_heatmap_callback(self, callback):
        """Set callback returning trial-by-parcel source time courses."""
        self._parcel_heatmap_callback = callback

    def set_source_display_callback(self, callback):
        """Set callback receiving `(stc, time_sec)` for 3D source display updates."""
        self._source_display_callback = callback

    def set_source_brain_view_callback(self, callback):
        """Set callback receiving `subject` or `fsaverage` for source display brain mode."""
        self._source_brain_view_callback = callback

    def set_semantic_preview_callback(self, callback):
        """Set callback returning `(result_dict, message)` for semantic preview."""
        self._semantic_preview_callback = callback

    def set_semantic_decode_callback(self, callback):
        """Set callback returning `(result_dict, message)` for semantic decoding."""
        self._semantic_decode_callback = callback

    def set_concept_decode_callback(self, callback):
        """Set callback returning `(result_dict, message)` for signal-space concept decoding."""
        self._concept_decode_callback = callback

    def set_semantic_parcellations(self, all_parcellations, selected=None):
        if getattr(self, "cmb_semantic_parcellations", None) is None:
            return
        if selected is None:
            selected = []
        selected_set = {str(v) for v in selected}
        tree = {"Left hemisphere": (False, {}), "Right hemisphere": (False, {}), "Other": (False, {})}
        for name in [str(v) for v in all_parcellations or []]:
            bucket = "Other"
            lname = name.lower()
            if lname.startswith("ctx-lh-"):
                bucket = "Left hemisphere"
            elif lname.startswith("ctx-rh-"):
                bucket = "Right hemisphere"
            tree[bucket][1][name] = (name in selected_set, {})
        combo = self.cmb_semantic_parcellations
        combo.model.clear()
        combo.channel_dict = {}
        combo.addItems(tree)
        combo.updateText()

    def set_source_stc(self, stc, status_text=None, source_src=None):
        self._source_stc = stc
        self._source_src = source_src
        self._source_time_index = None
        if status_text is not None:
            self._source_status_text = str(status_text)
            self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
        self._update_source_time_controls()

    def refresh_source_plot(self):
        """Public wrapper to refresh source time-course figure."""
        self._render_source_time_course()

    def set_source_defaults(self, inverse_operator_path=None, subjects_dir=None, subject=None):
        """Set default source/inverse context values in UI."""
        if inverse_operator_path:
            self.le_source_inverse.setText(str(inverse_operator_path))
        else:
            self.le_source_inverse.setText("")
        if subjects_dir:
            self._source_subjects_dir = str(subjects_dir)
            self.lbl_source_subjects_dir.setText(str(subjects_dir))
        if subject:
            self._source_subject = str(subject)
            self.lbl_source_subject.setText(str(subject))
        self._refresh_concept_parcel_selector()

    def set_source_run_options(
        self,
        run_paths=None,
        fif_filename=None,
        trial_df=None,
        run_meta_by_path=None,
    ):
        """Set Source-tab run choices without requiring cached epochs."""
        if fif_filename:
            self._fif_filename = str(fif_filename)
        if isinstance(trial_df, pd.DataFrame):
            self._trial_df = trial_df.copy()
        if isinstance(run_meta_by_path, dict):
            for path, meta in run_meta_by_path.items():
                norm = self._normalize_fif_path(path)
                if norm:
                    self._topomap_run_meta_by_path[norm] = dict(meta) if isinstance(meta, dict) else {}

        ordered = []
        current_path = self._normalize_fif_path(fif_filename or self._fif_filename)
        if current_path:
            ordered.append(current_path)
        for path in list(run_paths or []):
            norm = self._normalize_fif_path(path)
            if norm and norm not in ordered:
                ordered.append(norm)
        if isinstance(self._trial_df, pd.DataFrame) and "source_fif_path" in self._trial_df.columns:
            for value in self._trial_df["source_fif_path"].dropna().tolist():
                norm = self._normalize_fif_path(value)
                if norm and norm not in ordered:
                    ordered.append(norm)
        for path in ordered:
            if path not in self._topomap_run_order:
                self._topomap_run_order.append(path)
        self._refresh_source_run_selector()
        self._refresh_source_evoked_selector()
        self._refresh_parcel_heatmap_run_selector()
        self._refresh_parcel_heatmap_stimulus_selector()

    @staticmethod
    def _normalize_fif_path(path):
        if not path:
            return None
        try:
            return os.path.abspath(str(path))
        except Exception:
            return str(path)

    @staticmethod
    def _topomap_run_display_label(path, meta=None):
        path_str = str(path or "")
        base = os.path.basename(path_str) if path_str else "current FIF"
        run_id = None
        if isinstance(meta, dict):
            run_id = meta.get("run_id")
        if run_id is None:
            m = re.search(r"run[_-]?(\d+)|run(\d+)", base, flags=re.IGNORECASE)
            if m is not None:
                run_id = m.group(1) or m.group(2)
        if run_id not in (None, ""):
            return f"Run {run_id} ({base})"
        return base

    def _set_topomap_epoch_runs(
        self,
        epochs=None,
        fif_filename=None,
        topomap_epochs_by_path=None,
        topomap_run_meta_by_path=None,
    ):
        runs = {}
        order = []
        meta_by_path = {}

        if isinstance(topomap_run_meta_by_path, dict):
            for path, meta in topomap_run_meta_by_path.items():
                norm = self._normalize_fif_path(path)
                if norm:
                    meta_by_path[norm] = dict(meta) if isinstance(meta, dict) else {}

        if isinstance(topomap_epochs_by_path, dict):
            for path, run_epochs in topomap_epochs_by_path.items():
                norm = self._normalize_fif_path(path)
                if norm and run_epochs is not None:
                    runs[norm] = run_epochs
                    if norm not in order:
                        order.append(norm)

        current_path = self._normalize_fif_path(fif_filename or self._fif_filename)
        if current_path and epochs is not None:
            runs[current_path] = epochs
            if current_path not in order:
                order.insert(0, current_path)
        elif epochs is not None and not runs:
            current_path = "__current__"
            runs[current_path] = epochs
            order.append(current_path)

        self._topomap_epochs_by_path = runs
        self._topomap_run_order = [path for path in order if path in runs]
        for path in runs:
            if path not in self._topomap_run_order:
                self._topomap_run_order.append(path)
        self._topomap_run_meta_by_path = meta_by_path

    def _refresh_topomap_run_selector(self):
        combo = getattr(self, "cmb_topomap_run", None)
        if combo is None:
            return
        prev = combo.currentData() if combo.count() > 0 else None
        paths = [path for path in self._topomap_run_order if path in self._topomap_epochs_by_path]
        if not paths and self._epochs is not None:
            current_path = self._normalize_fif_path(self._fif_filename) or "__current__"
            paths = [current_path]
            self._topomap_epochs_by_path[current_path] = self._epochs
            self._topomap_run_order = paths

        combo.blockSignals(True)
        combo.clear()
        if len(paths) > 1:
            combo.addItem(f"All runs ({len(paths)})", "__all__")
        for path in paths:
            label = self._topomap_run_display_label(path, self._topomap_run_meta_by_path.get(path, {}))
            combo.addItem(label, path)

        target = prev
        if target is None:
            current_path = self._normalize_fif_path(self._fif_filename)
            target = current_path if current_path in paths else ("__all__" if len(paths) > 1 else None)
        idx = 0
        if target is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == target:
                    idx = i
                    break
        combo.setCurrentIndex(idx if combo.count() > 0 else -1)
        combo.blockSignals(False)

    def _selected_topomap_run_entries(self):
        paths = [path for path in self._topomap_run_order if path in self._topomap_epochs_by_path]
        if not paths and self._epochs is not None:
            current_path = self._normalize_fif_path(self._fif_filename) or "__current__"
            self._topomap_epochs_by_path[current_path] = self._epochs
            self._topomap_run_order = [current_path]
            paths = [current_path]

        combo = getattr(self, "cmb_topomap_run", None)
        selected = combo.currentData() if combo is not None and combo.count() > 0 else None
        if selected == "__all__":
            selected_paths = paths
        elif selected in self._topomap_epochs_by_path:
            selected_paths = [selected]
        else:
            current_path = self._normalize_fif_path(self._fif_filename)
            selected_paths = [current_path] if current_path in self._topomap_epochs_by_path else paths[:1]

        entries = []
        for path in selected_paths:
            epochs = self._topomap_epochs_by_path.get(path)
            if epochs is not None:
                entries.append((path, epochs))
        return entries

    def _topomap_run_scope_label(self):
        combo = getattr(self, "cmb_topomap_run", None)
        if combo is None or combo.count() == 0:
            return "current run"
        return str(combo.currentText() or "current run")

    def _refresh_source_run_selector(self):
        combo = getattr(self, "cmb_source_run", None)
        if combo is None:
            return
        prev_paths = self._selected_source_run_paths()
        paths = self._source_run_paths()
        if not paths and self._epochs is not None:
            current_path = self._normalize_fif_path(self._fif_filename) or "__current__"
            paths = [current_path]
            self._topomap_epochs_by_path[current_path] = self._epochs
            self._topomap_run_order = paths

        target = None
        if len(prev_paths) == len(paths) and len(paths) > 1:
            target = "__all__"
        elif len(prev_paths) == 1 and prev_paths[0] in paths:
            target = prev_paths[0]
        if target is None:
            current_path = self._normalize_fif_path(self._fif_filename)
            if len(paths) > 1:
                target = "__all__"
            elif current_path in paths:
                target = current_path
            elif paths:
                target = paths[0]

        combo.blockSignals(True)
        combo.clear()
        if len(paths) > 1:
            combo.addItem(f"All runs ({len(paths)})", "__all__")
        for path in paths:
            label = self._topomap_run_display_label(path, self._topomap_run_meta_by_path.get(path, {}))
            combo.addItem(label, path)
        idx = 0
        if target is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == target:
                    idx = i
                    break
        combo.setCurrentIndex(idx if combo.count() > 0 else -1)
        combo.blockSignals(False)

    def _source_run_paths(self):
        paths = []
        for path in self._topomap_run_order:
            norm = self._normalize_fif_path(path)
            if norm and norm not in paths:
                paths.append(norm)

        current_path = self._normalize_fif_path(self._fif_filename)
        if not current_path:
            meta = self._analysis_metadata()
            current_path = self._normalize_fif_path(meta.get("source_fif_path"))
        if current_path and current_path not in paths:
            paths.insert(0, current_path)
            if self._epochs is not None:
                self._topomap_epochs_by_path.setdefault(current_path, self._epochs)

        trial_df = self._analysis_trial_df()
        if isinstance(trial_df, pd.DataFrame) and "source_fif_path" in trial_df.columns:
            for value in trial_df["source_fif_path"].dropna().tolist():
                norm = self._normalize_fif_path(value)
                if norm and norm not in paths:
                    paths.append(norm)

        if not paths and self._epochs is not None:
            current_path = current_path or "__current__"
            self._topomap_epochs_by_path[current_path] = self._epochs
            self._topomap_run_order = [current_path]
            paths = [current_path]
        return paths

    def _checked_source_run_paths(self):
        combo = getattr(self, "cmb_source_run", None)
        paths = self._source_run_paths()
        if combo is None:
            return []
        if combo.count() > 0:
            selected = combo.currentData()
            if selected == "__all__":
                return list(paths)
            if selected in paths:
                return [selected]
        return []

    def _selected_source_run_paths(self):
        paths = self._source_run_paths()
        combo = getattr(self, "cmb_source_run", None)
        if combo is not None and combo.count() > 0:
            selected = combo.currentData()
            if selected == "__all__":
                return list(paths)
            if selected in paths:
                return [selected]
        current_path = self._normalize_fif_path(self._fif_filename)
        if current_path in self._topomap_epochs_by_path:
            return [current_path]
        return paths[:1]

    def _selected_source_run_entries(self):
        entries = []
        for path in self._selected_source_run_paths():
            epochs = self._topomap_epochs_by_path.get(path)
            if epochs is not None:
                entries.append((path, epochs))
        return entries

    def _selected_source_run_entry(self):
        entries = self._selected_source_run_entries()
        if not entries:
            return None, None
        return entries[0]

    def _source_run_scope_label(self):
        paths = self._selected_source_run_paths()
        all_paths = self._source_run_paths()
        if not paths:
            return "no runs selected"
        if len(paths) == len(all_paths) and len(all_paths) > 1:
            return f"all runs ({len(paths)})"
        if len(paths) == 1:
            return self._topomap_run_display_label(paths[0], self._topomap_run_meta_by_path.get(paths[0], {}))
        return f"{len(paths)} selected runs"

    def _parcel_heatmap_group_mode(self):
        combo = getattr(self, "cmb_parcel_heatmap_grouping", None)
        return self._group_mode_from_index(combo.currentIndex() if combo is not None else 0)

    def _selected_parcel_heatmap_run_paths(self):
        paths = self._source_run_paths()
        combo = getattr(self, "cmb_parcel_heatmap_run", None)
        if combo is not None and combo.count() > 0:
            selected = combo.currentData()
            if selected == "__all__":
                return list(paths)
            if selected in paths:
                return [selected]
        return paths[:1]

    def _parcel_heatmap_run_scope_label(self):
        selected = self._selected_parcel_heatmap_run_paths()
        available = self._source_run_paths()
        if not selected:
            return "no runs selected"
        if len(selected) == len(available) and len(available) > 1:
            return f"all runs ({len(selected)})"
        if len(selected) == 1:
            return self._topomap_run_display_label(
                selected[0], self._topomap_run_meta_by_path.get(selected[0], {})
            )
        return f"{len(selected)} selected runs"

    def _refresh_parcel_heatmap_run_selector(self):
        combo = getattr(self, "cmb_parcel_heatmap_run", None)
        if combo is None:
            return
        paths = self._source_run_paths()
        previous = combo.currentData() if combo.count() > 0 else None
        combo.blockSignals(True)
        combo.clear()
        if len(paths) > 1:
            combo.addItem(f"All runs ({len(paths)})", "__all__")
        for path in paths:
            combo.addItem(
                self._topomap_run_display_label(path, self._topomap_run_meta_by_path.get(path, {})),
                path,
            )
        target = previous
        if target not in paths and target != "__all__":
            target = "__all__" if len(paths) > 1 else (paths[0] if paths else None)
        index = combo.findData(target) if target is not None else -1
        combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
        combo.blockSignals(False)

    def _parcel_heatmap_indices_by_stimulus(self):
        mapping = {}
        mode = self._parcel_heatmap_group_mode()
        for path in self._selected_parcel_heatmap_run_paths():
            trial_df = self._trial_df_for_topomap_path(path)
            for index in range(len(trial_df)):
                label = self._stimulus_label(trial_df.iloc[index], mode=mode)
                mapping.setdefault(label, []).append((path, index))
        return mapping

    def _refresh_parcel_heatmap_stimulus_selector(self):
        combo = getattr(self, "cmb_parcel_heatmap_stimulus", None)
        if combo is None:
            return
        previous = str(combo.currentData() or "") if combo.count() > 0 else ""
        mapping = self._parcel_heatmap_indices_by_stimulus()
        combo.blockSignals(True)
        combo.clear()
        for label, presentations in mapping.items():
            combo.addItem(f"{label} (n={len(presentations)})", str(label))
        restore_index = combo.findData(previous)
        combo.setCurrentIndex(restore_index if restore_index >= 0 else (0 if combo.count() else -1))
        combo.blockSignals(False)
        self._refresh_parcel_heatmap_comparison_selector()

    def _refresh_parcel_heatmap_comparison_selector(self, *_args):
        combo = getattr(self, "cmb_parcel_heatmap_comparison", None)
        stimulus_combo = getattr(self, "cmb_parcel_heatmap_stimulus", None)
        if combo is None or stimulus_combo is None:
            return
        previous = str(combo.currentData() or "") if combo.count() > 0 else ""
        primary = str(stimulus_combo.currentData() or "")
        mapping = self._parcel_heatmap_indices_by_stimulus()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None", "")
        for label, presentations in mapping.items():
            label = str(label)
            if label == primary:
                continue
            combo.addItem(f"{label} (n={len(presentations)})", label)
        restore_index = combo.findData(previous)
        combo.setCurrentIndex(restore_index if restore_index >= 0 else 0)
        combo.blockSignals(False)

    def _trial_df_for_topomap_path(self, path):
        trial_df = self._analysis_trial_df()
        if trial_df is None or trial_df.empty:
            return pd.DataFrame()
        df = trial_df.copy()
        if "source_fif_path" not in df.columns:
            current = self._normalize_fif_path(self._fif_filename)
            norm = self._normalize_fif_path(path)
            return df.reset_index(drop=True) if current is None or norm == current else pd.DataFrame()
        norm = self._normalize_fif_path(path)
        if not norm:
            return pd.DataFrame()
        try:
            source_paths = df["source_fif_path"].astype(str).map(lambda value: os.path.abspath(str(value)))
        except Exception:
            source_paths = df["source_fif_path"].astype(str)
        mask = source_paths == norm
        if np.any(mask):
            return df[mask].copy().reset_index(drop=True)
        return pd.DataFrame()

    def _trial_df_for_current_file(self):
        trial_df = self._analysis_trial_df()
        if trial_df is None or trial_df.empty:
            return pd.DataFrame()
        df = trial_df.copy()
        if "source_fif_path" in df.columns and self._fif_filename:
            fname = os.path.abspath(str(self._fif_filename))
            mask = df["source_fif_path"].astype(str) == fname
            if np.any(mask):
                df = df[mask].copy()
        return df.reset_index(drop=True)

    @staticmethod
    def _stimulus_label(row, mode="canonical"):
        if str(mode) == "individual":
            if "stimulus" in row and pd.notna(row["stimulus"]):
                return os.path.basename(str(row["stimulus"]))
        if str(mode) == "category":
            if "stimulus_group" in row and pd.notna(row["stimulus_group"]):
                group = str(row["stimulus_group"]).strip()
                if group:
                    return group
        if "canonical_stimulus" in row and pd.notna(row["canonical_stimulus"]):
            return str(row["canonical_stimulus"])
        if "stimulus" in row and pd.notna(row["stimulus"]):
            return StimulusResponseReport._canonicalize_stimulus_label(str(row["stimulus"]))
        return "unknown"

    @staticmethod
    def _group_mode_from_index(idx):
        try:
            idx = int(idx)
        except Exception:
            idx = 0
        if idx == 1:
            return "category"
        if idx == 2:
            return "individual"
        return "canonical"

    @staticmethod
    def _group_mode_label(mode):
        mode = str(mode or "canonical").strip().lower()
        if mode == "category":
            return "stimulus categories"
        if mode == "individual":
            return "individual stimuli"
        return "canonical groups"

    @staticmethod
    def _format_channel_list(channels, max_show=12):
        if not channels:
            return "none"
        channels = [str(ch) for ch in channels]
        if len(channels) <= int(max_show):
            return ", ".join(channels)
        shown = channels[: int(max_show)]
        return f"{', '.join(shown)} ... (+{len(channels) - int(max_show)} more)"

    def _stimulus_labels_for_epochs(self):
        if self._epochs is None or len(self._epochs) == 0:
            return []
        trial_df = self._trial_df_for_current_file()
        if trial_df.empty:
            return []
        n = min(len(self._epochs), len(trial_df))
        labels = []
        seen = set()
        for i in range(n):
            lbl = self._stimulus_label(trial_df.iloc[i], mode="canonical")
            if lbl not in seen:
                labels.append(lbl)
                seen.add(lbl)
        return labels

    def _epoch_indices_by_stimulus(self):
        mapping = {}
        if self._epochs is None or len(self._epochs) == 0:
            return mapping
        trial_df = self._trial_df_for_current_file()
        if trial_df.empty:
            return mapping
        n = min(len(self._epochs), len(trial_df))
        for i in range(n):
            lbl = self._stimulus_label(trial_df.iloc[i], mode="canonical")
            mapping.setdefault(lbl, []).append(i)
        return mapping

    def _evoked_group_mode(self):
        return self._group_mode_from_index(self.cmb_evoked_grouping.currentIndex())

    def _evoked_indices_by_stimulus(self):
        mapping = {}
        if self._epochs is None or len(self._epochs) == 0:
            return mapping
        trial_df = self._trial_df_for_current_file()
        if trial_df.empty:
            return mapping
        mode = self._evoked_group_mode()
        n = min(len(self._epochs), len(trial_df))
        for i in range(n):
            lbl = self._stimulus_label(trial_df.iloc[i], mode=mode)
            mapping.setdefault(lbl, []).append(i)
        return mapping

    def _frequency_group_mode(self):
        return self._group_mode_from_index(self.cmb_frequency_grouping.currentIndex())

    def _frequency_indices_by_stimulus(self):
        mapping = {}
        if self._epochs is None or len(self._epochs) == 0:
            return mapping
        trial_df = self._trial_df_for_current_file()
        mode = self._frequency_group_mode()
        if not trial_df.empty:
            n = min(len(self._epochs), len(trial_df))
            for i in range(n):
                label = self._stimulus_label(trial_df.iloc[i], mode=mode)
                mapping.setdefault(label, []).append(i)
        if mapping:
            return mapping

        # Keep the tab useful for ordinary MNE Epochs that have no stimulus table.
        events = np.asarray(getattr(self._epochs, "events", []))
        for label, code in getattr(self._epochs, "event_id", {}).items():
            if events.ndim == 2 and events.shape[1] >= 3:
                idx = np.flatnonzero(events[:, 2] == int(code)).astype(int).tolist()
                if idx:
                    mapping[str(label)] = idx
        return mapping

    def _topomap_group_mode(self):
        return self._group_mode_from_index(self.cmb_topomap_grouping.currentIndex())

    def _source_group_mode(self):
        return self._group_mode_from_index(self.cmb_source_grouping.currentIndex())

    def _topomap_indices_by_stimulus(self):
        mapping = {}
        mode = self._topomap_group_mode()
        for path, epochs in self._selected_topomap_run_entries():
            if epochs is None or len(epochs) == 0:
                continue
            trial_df = self._trial_df_for_topomap_path(path)
            if trial_df.empty:
                continue
            n = min(len(epochs), len(trial_df))
            for i in range(n):
                lbl = self._stimulus_label(trial_df.iloc[i], mode=mode)
                mapping.setdefault(lbl, []).append((path, i))
        return mapping

    def _source_indices_by_stimulus(self):
        mapping = {}
        mode = self._source_group_mode()
        for path in self._selected_source_run_paths():
            trial_df = self._trial_df_for_topomap_path(path)
            if trial_df.empty:
                continue
            epochs = self._topomap_epochs_by_path.get(path)
            n = len(trial_df)
            if epochs is not None:
                n = min(int(len(epochs)), n)
            for i in range(n):
                lbl = self._stimulus_label(trial_df.iloc[i], mode=mode)
                mapping.setdefault(lbl, []).append((path, i))
        return mapping

    def _semantic_indices_by_stimulus(self):
        mapping = {}
        if self._epochs is None or len(self._epochs) == 0:
            return mapping
        trial_df = self._trial_df_for_current_file()
        if trial_df.empty:
            return mapping
        mode = self._semantic_group_mode()
        n = min(len(self._epochs), len(trial_df))
        for i in range(n):
            lbl = self._stimulus_label(trial_df.iloc[i], mode=mode)
            mapping.setdefault(lbl, []).append(i)
        return mapping

    def _refresh_stimulus_selectors(self):
        all_label = "All stimuli"
        labels = self._stimulus_labels_for_epochs()
        self.cmb_epoch_stimulus.blockSignals(True)
        self.cmb_epoch_stimulus.clear()
        self.cmb_epoch_stimulus.addItem(all_label)
        for lbl in labels:
            self.cmb_epoch_stimulus.addItem(lbl)
        self.cmb_epoch_stimulus.blockSignals(False)
        self._refresh_evoked_stimulus_selector()
        self._refresh_frequency_stimulus_selector()
        self._refresh_topomap_stimulus_selector()
        self._refresh_semantic_stimulus_selector()
        self._on_epoch_stimulus_changed()
        self._on_evoked_stimulus_changed()
        self._on_topomap_stimulus_changed()

    def _refresh_frequency_stimulus_selector(self):
        combo = getattr(self, "cmb_frequency_stimulus", None)
        if combo is None:
            return
        previous = str(combo.currentText() or "All stimuli")
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All stimuli")
        for label in self._frequency_indices_by_stimulus():
            combo.addItem(str(label))
        restore = combo.findText(previous)
        combo.setCurrentIndex(restore if restore >= 0 else 0)
        combo.blockSignals(False)

    def _refresh_evoked_stimulus_selector(self):
        all_label = "All stimuli"
        mapping = self._evoked_indices_by_stimulus()
        labels = list(mapping.keys())
        prev = str(self.cmb_evoked_stimulus.currentText()) if self.cmb_evoked_stimulus.count() > 0 else all_label
        self.cmb_evoked_stimulus.blockSignals(True)
        self.cmb_evoked_stimulus.clear()
        self.cmb_evoked_stimulus.addItem(all_label)
        for lbl in labels:
            self.cmb_evoked_stimulus.addItem(lbl)
        idx = self.cmb_evoked_stimulus.findText(prev)
        if idx < 0:
            idx = 0
        self.cmb_evoked_stimulus.setCurrentIndex(idx)
        self.cmb_evoked_stimulus.blockSignals(False)

    def _refresh_topomap_stimulus_selector(self):
        all_label = "All stimuli"
        mapping = self._topomap_indices_by_stimulus()
        labels = list(mapping.keys())
        prev = str(self.cmb_topomap_stimulus.currentText()) if self.cmb_topomap_stimulus.count() > 0 else all_label
        self.cmb_topomap_stimulus.blockSignals(True)
        self.cmb_topomap_stimulus.clear()
        self.cmb_topomap_stimulus.addItem(all_label)
        for lbl in labels:
            self.cmb_topomap_stimulus.addItem(lbl)
        idx = self.cmb_topomap_stimulus.findText(prev)
        if idx < 0:
            idx = 0
        self.cmb_topomap_stimulus.setCurrentIndex(idx)
        self.cmb_topomap_stimulus.blockSignals(False)

    def _refresh_source_evoked_selector(self):
        all_label = "All stimuli"
        mapping = self._source_indices_by_stimulus()
        labels = list(mapping.keys())
        prev = str(self.cmb_source_evoked.currentText()) if self.cmb_source_evoked.count() > 0 else all_label
        self.cmb_source_evoked.blockSignals(True)
        self.cmb_source_evoked.clear()
        self.cmb_source_evoked.addItem(all_label)
        for lbl in labels:
            self.cmb_source_evoked.addItem(lbl)
        idx = self.cmb_source_evoked.findText(prev)
        if idx < 0:
            idx = 0
        self.cmb_source_evoked.setCurrentIndex(idx)
        self.cmb_source_evoked.blockSignals(False)

    def _refresh_semantic_stimulus_selector(self, allowed_labels=None):
        all_label = "All stimuli"
        mapping = self._semantic_indices_by_stimulus()
        labels = list(mapping.keys())
        if allowed_labels is not None:
            allowed = {str(v) for v in allowed_labels}
            labels = [lbl for lbl in labels if lbl in allowed]
        prev = str(self.cmb_semantic_stimulus.currentText()) if self.cmb_semantic_stimulus.count() > 0 else all_label
        self.cmb_semantic_stimulus.blockSignals(True)
        self.cmb_semantic_stimulus.clear()
        self.cmb_semantic_stimulus.addItem(all_label)
        for lbl in labels:
            self.cmb_semantic_stimulus.addItem(lbl)
        idx = self.cmb_semantic_stimulus.findText(prev)
        if idx < 0:
            idx = 0
        self.cmb_semantic_stimulus.setCurrentIndex(idx)
        self.cmb_semantic_stimulus.blockSignals(False)
        self._refresh_semantic_trial_selector()

    def _refresh_semantic_trial_selector(self):
        combo = getattr(self, "cmb_semantic_trial", None)
        if combo is None:
            return
        selected_group = str(self.cmb_semantic_stimulus.currentText() or "All stimuli")
        mapping = self._semantic_indices_by_stimulus()
        idxs = list(mapping.get(selected_group, [])) if selected_group != "All stimuli" else []
        prev_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Average across matched trials", None)
        trial_df = self._trial_df_for_current_file()
        n = min(len(self._epochs) if self._epochs is not None else 0, len(trial_df))
        for idx in idxs:
            if idx >= n:
                continue
            row = trial_df.iloc[int(idx)]
            trial_num = row.get("trial_index", row.get("trial", int(idx) + 1))
            stim_name = row.get("stimulus", "")
            label = f"Trial {trial_num}"
            if pd.notna(stim_name) and str(stim_name).strip():
                label = f"{label}: {os.path.basename(str(stim_name))}"
            combo.addItem(label, int(idx))
        if prev_data is not None:
            restore_idx = combo.findData(prev_data)
            if restore_idx >= 0:
                combo.setCurrentIndex(restore_idx)
        combo.blockSignals(False)

    @staticmethod
    def _sensor_region(position):
        """Return a coarse helmet region for an MNE head-coordinate position."""
        if position is None:
            return "Unlocalized"
        pos = np.asarray(position, dtype=float).reshape(-1)
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])) or np.linalg.norm(pos[:3]) < 1e-9:
            return "Unlocalized"
        x, y, z = pos[:3]
        # MNE head coordinates: +x right, +y anterior, +z superior. These
        # boundaries describe sensor locations on the helmet, not cortical ROIs.
        if abs(x) >= 0.065 and z < 0.045 and -0.055 < y < 0.045:
            return "Temporal"
        if y >= 0.035:
            return "Frontal"
        if y <= -0.055:
            return "Occipital"
        if y <= -0.005:
            return "Parietal"
        return "Central"

    @staticmethod
    def _channel_positions_in_head(instance, channel_names):
        """Return channel positions transformed to MNE head coordinates when possible."""
        if instance is None:
            return {}
        info = getattr(instance, "info", None)
        if info is None:
            return {}
        info_names = list(info.get("ch_names", []))
        channels = list(info.get("chs", []))
        index_by_name = {str(name): index for index, name in enumerate(info_names)}
        transform = None
        try:
            dev_head_t = info.get("dev_head_t")
            if dev_head_t is not None:
                transform = np.asarray(dev_head_t["trans"], dtype=float)
                if transform.shape != (4, 4):
                    transform = None
        except Exception:
            transform = None

        positions = {}
        for name in channel_names:
            index = index_by_name.get(str(name))
            if index is None or index >= len(channels):
                positions[str(name)] = None
                continue
            channel = channels[index]
            try:
                position = np.asarray(channel.get("loc", [])[:3], dtype=float)
                # FIFF coordinate frame 1 is device; frame 4 is already head.
                if int(channel.get("coord_frame", 0)) == 1 and transform is not None:
                    position = transform[:3, :3] @ position + transform[:3, 3]
                positions[str(name)] = position
            except Exception:
                positions[str(name)] = None
        return positions

    @classmethod
    def _group_sensor_channels(cls, channel_names, positions):
        """Group sensor names by hemisphere and coarse helmet region."""
        grouped = {}
        for channel_name in channel_names:
            position = positions.get(str(channel_name))
            region = cls._sensor_region(position)
            if region == "Unlocalized":
                hemisphere = "Unlocalized"
            else:
                hemisphere = "Left" if float(np.asarray(position)[0]) < 0.0 else "Right"
            grouped.setdefault(hemisphere, {}).setdefault(region, []).append(str(channel_name))
        return grouped

    def _refresh_channel_selectors(self):
        ch_names = []
        if self._epochs is not None and len(getattr(self._epochs, "ch_names", [])) > 0:
            ch_names = list(self._epochs.ch_names)
        elif self._raw is not None and len(getattr(self._raw, "ch_names", [])) > 0:
            ch_names = list(self._raw.ch_names)
        self.lst_epoch_channels.clear()
        self.lst_evoked_channels.clear()
        self.lst_frequency_channels.clear()
        channel_types = []
        instance = self._epochs if self._epochs is not None else self._raw
        if instance is not None:
            try:
                channel_types = list(instance.get_channel_types())
            except Exception:
                channel_types = []

        grouped_names = {"mag": [], "grad": [], "other": []}
        for index, channel_name in enumerate(ch_names):
            channel_type = channel_types[index] if index < len(channel_types) else "other"
            group = channel_type if channel_type in ("mag", "grad") else "other"
            grouped_names[group].append(str(channel_name))

        positions = self._channel_positions_in_head(instance, ch_names)

        # Match the Channels Viewer: a virtual event-marker entry followed by
        # independently checkable magnetometer and gradiometer branches.
        analysis_tree = {MEG_EVENTS_CHANNEL_LABEL: (True, {})}
        hemisphere_order = ("Left", "Right", "Unlocalized")
        region_order = ("Frontal", "Central", "Temporal", "Parietal", "Occipital", "Unlocalized")
        type_order = (
            ("Magnetometers", "mag"),
            ("Gradiometers", "grad"),
            ("Other channels", "other"),
        )
        for type_label, key in type_order:
            grouped = self._group_sensor_channels(grouped_names[key], positions)
            hemisphere_children = {}
            for hemisphere in hemisphere_order:
                region_groups = grouped.get(hemisphere, {})
                if not region_groups:
                    continue
                region_children = {}
                for region in region_order:
                    region_names = region_groups.get(region, [])
                    if region_names:
                        region_children[region] = (
                            False,
                            {name: (False, {}) for name in region_names},
                        )
                hemisphere_children[hemisphere] = (False, region_children)
            if hemisphere_children:
                analysis_tree[type_label] = (False, hemisphere_children)

        self.lst_epoch_channels.addItems(analysis_tree)
        self.lst_evoked_channels.addItems(analysis_tree)
        frequency_tree = {
            "MEG": (False, {str(ch): (False, {}) for ch in ch_names})
        }
        self.lst_frequency_channels.addItems(frequency_tree)
        self._set_checked_channels(self.lst_frequency_channels, ch_names[: min(24, len(ch_names))])
        self._on_epoch_stimulus_changed()
        self._on_evoked_stimulus_changed()

    def _responsive_score_threshold(self):
        """Get current responsiveness threshold from stimulus report UI."""
        try:
            return float(self.stimulus_report.spb_score_thr.value())
        except Exception:
            return 0.15

    def _responsive_channels_for_stimulus(self, stimulus_label, label_mode="canonical"):
        """Return ordered responsive channels for the selected stimulus label."""
        df = self._trial_df_for_current_file()
        if df.empty:
            return []

        selected_label = str(stimulus_label or "All stimuli")
        if selected_label != "All stimuli":
            labels = [self._stimulus_label(df.iloc[i], mode=label_mode) for i in range(len(df))]
            mask = np.asarray([lbl == selected_label for lbl in labels], dtype=bool)
            if np.any(mask):
                df = df.loc[mask].copy().reset_index(drop=True)
            else:
                return []

        score_thr = self._responsive_score_threshold()
        channels = {}

        if "responsive_channels_list" in df.columns:
            for val in df["responsive_channels_list"].tolist():
                if isinstance(val, (list, tuple, set)):
                    for ch in val:
                        chs = str(ch).strip()
                        if chs:
                            channels[chs] = True
        if not channels and "responsive_channels" in df.columns:
            for val in df["responsive_channels"].tolist():
                if val is None:
                    continue
                parts = [p.strip() for p in str(val).split(",")]
                for chs in parts:
                    if chs:
                        channels[chs] = True
        if not channels and "channel_delta_pairs" in df.columns:
            for pairs in df["channel_delta_pairs"].tolist():
                if not isinstance(pairs, (list, tuple)):
                    continue
                for item in pairs:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    chs = str(item[0]).strip()
                    try:
                        score = float(item[1])
                    except Exception:
                        continue
                    if chs and np.isfinite(score) and score >= score_thr:
                        channels[chs] = True
        if not channels and "top_response_channel" in df.columns:
            for ch in df["top_response_channel"].tolist():
                chs = str(ch).strip()
                if chs:
                    channels[chs] = True

        if not channels:
            return []

        ordered = []
        ch_order = []
        if self._epochs is not None and len(getattr(self._epochs, "ch_names", [])) > 0:
            ch_order = list(self._epochs.ch_names)
        elif self._raw is not None and len(getattr(self._raw, "ch_names", [])) > 0:
            ch_order = list(self._raw.ch_names)
        ch_set = set(channels.keys())
        for ch in ch_order:
            if ch in ch_set:
                ordered.append(ch)
        for ch in channels.keys():
            if ch not in ordered:
                ordered.append(ch)
        return ordered

    def _apply_default_channel_selection(self, channel_control, stimulus_label, label_mode="canonical"):
        """Select responsive channels for stimulus with fallback to first channels."""
        if channel_control is None:
            return
        preserved = []
        if self._events_selected(channel_control):
            preserved.append(MEG_EVENTS_CHANNEL_LABEL)
        responsive = self._responsive_channels_for_stimulus(stimulus_label, label_mode=label_mode)
        if responsive:
            self._set_checked_channels(channel_control, preserved + responsive)
            return
        ch_names = []
        if self._epochs is not None:
            ch_names = list(getattr(self._epochs, "ch_names", []) or [])
        elif self._raw is not None:
            ch_names = list(getattr(self._raw, "ch_names", []) or [])
        self._set_checked_channels(channel_control, preserved + ch_names[:4])

    def _on_epoch_stimulus_changed(self, *_args):
        stim = str(self.cmb_epoch_stimulus.currentText()) if self.cmb_epoch_stimulus is not None else "All stimuli"
        self._apply_default_channel_selection(self.lst_epoch_channels, stim, label_mode="canonical")

    def _on_evoked_stimulus_changed(self, *_args):
        stim = str(self.cmb_evoked_stimulus.currentText()) if self.cmb_evoked_stimulus is not None else "All stimuli"
        self._apply_default_channel_selection(
            self.lst_evoked_channels,
            stim,
            label_mode=self._evoked_group_mode(),
        )

    def _on_evoked_grouping_changed(self, *_args):
        self._refresh_evoked_stimulus_selector()
        self._on_evoked_stimulus_changed()

    def _on_topomap_grouping_changed(self, *_args):
        self._refresh_topomap_stimulus_selector()
        self._on_topomap_stimulus_changed()

    def _on_topomap_run_changed(self, *_args):
        self._refresh_topomap_stimulus_selector()
        self._on_topomap_stimulus_changed()
        if not self._selected_topomap_run_entries():
            return
        self._plot_topomap_joint()

    def _on_topomap_stimulus_changed(self, *_args):
        selected = str(self.cmb_topomap_stimulus.currentText()) if self.cmb_topomap_stimulus is not None else "All stimuli"
        mode = self._group_mode_label(self._topomap_group_mode())
        self.lbl_topomap_info.setText(
            f"Topomap selection: {selected} ({mode}, run={self._topomap_run_scope_label()})"
        )

    def _on_source_grouping_changed(self, *_args):
        self._refresh_source_evoked_selector()
        self._source_stc = None
        self._source_src = None

    def _on_source_space_kind_changed(self, *_args):
        self._source_stc = None
        self._source_src = None
        self._source_status_text = "source space changed; recompute source estimate."
        if getattr(self, "lbl_source_status", None) is not None:
            self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
        self._render_source_time_course()

    def _on_concept_source_space_kind_changed(self, *_args):
        combo = getattr(self, "cmb_concept_source_space_kind", None)
        atlas_combo = getattr(self, "cmb_concept_source_parcellation", None)
        if combo is None or atlas_combo is None:
            return
        is_volume = str(combo.currentData() or "surface") == "volume"
        atlas_combo.setEnabled(not is_volume)
        target = "aparc+aseg" if is_volume else "aparc"
        if is_volume or str(atlas_combo.currentData() or "") == "aparc+aseg":
            idx = atlas_combo.findData(target)
            if idx >= 0:
                atlas_combo.setCurrentIndex(idx)
        self._refresh_concept_parcel_selector()

    def _on_concept_source_parcellation_changed(self, *_args):
        self._refresh_concept_parcel_selector()

    def _on_concept_feature_space_changed(self, *_args):
        feature_space_combo = getattr(self, "cmb_concept_feature_space", None)
        feature_mode_combo = getattr(self, "cmb_concept_feature_mode", None)
        if feature_space_combo is None or feature_mode_combo is None:
            return
        previous = feature_mode_combo.currentData()
        feature_space = str(feature_space_combo.currentData() or "signal")

        feature_mode_combo.blockSignals(True)
        feature_mode_combo.clear()
        if feature_space == "source":
            feature_mode_combo.addItem("Binned signed mean", "mean_binned")
            feature_mode_combo.addItem("Binned mean + RMS", "mean_rms_binned")
            feature_mode_combo.addItem("Window mean + RMS", "mean_rms_window")
            feature_mode_combo.addItem("Window signed mean", "mean_window")
            feature_mode_combo.addItem("Window RMS", "rms_window")
            feature_mode_combo.addItem("Raw parcel samples", "raw")
            feature_mode_combo.addItem("Binned band power (baseline dB)", "bandpower_binned_db")
            feature_mode_combo.addItem("Window band power (baseline dB)", "bandpower_window_db")
            feature_mode_combo.addItem("Binned band power (log power)", "bandpower_binned_log")
            feature_mode_combo.addItem("Window band power (log power)", "bandpower_window_log")
            feature_mode_combo.setToolTip(
                "Binned signed mean matches the temporal patterns displayed in Parcel trials. "
                "Mean + RMS adds polarity-invariant magnitude, while raw samples preserve every time point. "
                "Band power sums vector-component powers before averaging within each surface or volume ROI. "
                "Baseline dB uses the shared prestimulus interval; log power has no baseline normalization."
            )
            compatible = {
                "bandpower_binned_db", "bandpower_window_db",
                "bandpower_binned_log", "bandpower_window_log",
                "mean_binned",
                "mean_rms_binned",
                "mean_rms_binned_50",
                "mean_rms_window",
                "mean_window",
                "rms_window",
            }
            target = previous if previous in compatible else "mean_binned"
            if target == "mean_rms_binned_50":
                target = "mean_rms_binned"
        else:
            feature_mode_combo.addItem("Raw signal samples", "raw")
            feature_mode_combo.setToolTip(
                "Signal-space decoding currently uses raw channel time samples after optional downsampling."
            )
            target = "raw"
        idx = feature_mode_combo.findData(target)
        if idx < 0:
            idx = 0
        feature_mode_combo.setCurrentIndex(idx)
        feature_mode_combo.blockSignals(False)
        self._update_concept_source_preprocessing_controls()
        self._update_concept_summary_badge()

    def _on_concept_decoder_changed(self, *_args):
        """Keep decoder choices aligned with their supported feature spaces."""
        decoder_combo = getattr(self, "cmb_concept_decoder", None)
        feature_space_combo = getattr(self, "cmb_concept_feature_space", None)
        split_combo = getattr(self, "cmb_concept_split", None)
        if decoder_combo is None or feature_space_combo is None:
            return
        decoder_key = str(decoder_combo.currentData() or "logistic")
        pairwise_mode = decoder_key == "pairwise_svm"
        if pairwise_mode:
            feature_space_combo.setCurrentIndex(feature_space_combo.findData("signal"))
            if split_combo is not None:
                split_combo.setCurrentIndex(split_combo.findData("random"))
        generation_mode = decoder_key in {"clip_generation", "clip_generation_hybrid"}
        if generation_mode:
            feature_space_combo.setCurrentIndex(feature_space_combo.findData("source"))
            if split_combo is not None:
                split_combo.setCurrentIndex(split_combo.findData("random"))
        source_capable_embedding_decoder = decoder_key in {
            "embedding_ridge",
            "embedding_contrastive",
        }
        signal_only_embedding_decoder = decoder_key in {
            "embedding_mlp",
            "transformer",
        }
        semantic_embedding_decoder = (
            source_capable_embedding_decoder or signal_only_embedding_decoder
        )
        if signal_only_embedding_decoder:
            signal_idx = feature_space_combo.findData("signal")
            if signal_idx >= 0 and feature_space_combo.currentIndex() != signal_idx:
                feature_space_combo.setCurrentIndex(signal_idx)
            feature_space_combo.setToolTip(
                "Semantic embedding decoders use baseline-relative Hilbert band-power "
                "pseudo-epochs, restricted to bands preserved by the shared passband. "
                "Feature space is fixed to Signal space."
            )
        elif source_capable_embedding_decoder:
            feature_space_combo.setToolTip(
                "Linear semantic embedding decoders support either signal-space "
                "band-power features or the selected source-parcel features."
            )
        else:
            feature_space_combo.setToolTip(
                "Source parcels use the Source atlas selector and the Source time course inverse settings."
            )
        if split_combo is not None:
            current_split = str(split_combo.currentData() or "random")
            if semantic_embedding_decoder:
                # Semantic embedding decoders support 'random' and 'zero_shot'
                split_combo.setEnabled(not generation_mode and not pairwise_mode)
                if current_split in {"leave_one_run", "leave_one_session"}:
                    split_combo.setCurrentIndex(split_combo.findData("random"))
            else:
                # Standard discrete decoders support 'random', 'leave_one_run', 'leave_one_session'
                split_combo.setEnabled(not generation_mode and not pairwise_mode)
                if current_split == "zero_shot":
                    split_combo.setCurrentIndex(split_combo.findData("random"))
        feature_space_combo.setEnabled(not signal_only_embedding_decoder and not generation_mode and not pairwise_mode)
        compact_controls_enabled = decoder_key in {
            "embedding_ridge",
            "embedding_contrastive",
            "embedding_mlp",
            "clip_generation",
            "clip_generation_hybrid",
        }
        for name in (
            "spb_concept_embedding_input_pca",
            "spb_concept_embedding_target_svd",
        ):
            control = getattr(self, name, None)
            if control is not None:
                control.setEnabled(compact_controls_enabled)
        trainable_embedding = decoder_key in {
            "embedding_contrastive",
            "embedding_mlp",
        }
        for name in (
            "spb_concept_embedding_patience",
            "chk_concept_dynamic_pseudo_epochs",
        ):
            control = getattr(self, name, None)
            if control is not None:
                control.setEnabled(trainable_embedding)
        for name in ("spb_concept_transformer_epochs", "le_concept_embedding_path", "btn_concept_embedding_browse"):
            control = getattr(self, name, None)
            if control is not None:
                control.setEnabled(not generation_mode)
        if generation_mode and hasattr(self, "concept_config_tabs") and hasattr(self, "tab_config_images"):
            self.concept_config_tabs.setCurrentWidget(self.tab_config_images)
        self._update_concept_summary_badge()

    def _update_concept_source_preprocessing_controls(self, *_args):
        """Enable parcel-only controls when source-parcel features use them."""
        feature_space_combo = getattr(self, "cmb_concept_feature_space", None)
        is_source = bool(
            feature_space_combo is not None
            and str(feature_space_combo.currentData() or "signal") == "source"
        )
        mode_combo = getattr(self, "cmb_concept_feature_mode", None)
        is_power = is_source and mode_combo is not None and str(mode_combo.currentData() or "").startswith("bandpower_")
        power_band = getattr(self, "cmb_concept_power_band", None)
        if power_band is not None:
            power_band.setVisible(is_power)
        if hasattr(self, "chk_shared_parcel_baseline"):
            self._update_shared_source_preprocessing_controls()
        parcel_baseline = getattr(self, "chk_concept_parcel_baseline_zscore", None)
        mean_baseline = getattr(self, "chk_concept_baseline", None)
        if mean_baseline is not None:
            mean_baseline.setEnabled(
                not is_power and (
                    not is_source or parcel_baseline is None or not parcel_baseline.isChecked()
                )
            )
            mean_baseline.setToolTip(
                "Sensor/epoch mean subtraction over -100 to 0 ms. It is bypassed when "
                "source-parcel baseline z-scoring or source band-power features are enabled."
            )

    def _update_shared_source_preprocessing_controls(self, *_args):
        """Enable shared thresholds only when their preprocessing stage is active."""
        self.spb_shared_downsample_sfreq.setEnabled(
            self.chk_shared_downsample.isChecked()
        )
        reject_enabled = self.chk_shared_mne_reject.isChecked()
        self.spb_shared_reject_mag.setEnabled(reject_enabled)
        self.spb_shared_reject_grad.setEnabled(reject_enabled)
        self.spb_shared_outlier_z.setEnabled(
            self.chk_shared_exclude_outliers.isChecked()
        )
        representation_combo = getattr(
            self, "cmb_parcel_heatmap_representation", None
        )
        parcel_power_selected = bool(
            representation_combo is not None
            and str(representation_combo.currentData() or "activity")
            == "band_power_db"
        )
        feature_combo = getattr(self, "cmb_concept_feature_mode", None)
        decoding_power_baseline = bool(
            feature_combo is not None
            and str(feature_combo.currentData() or "") in {"bandpower_binned_db", "bandpower_window_db"}
        )
        baseline_enabled = (
            self.chk_shared_parcel_baseline.isChecked() or parcel_power_selected or decoding_power_baseline
        )
        self.spb_shared_baseline_start_ms.setEnabled(baseline_enabled)
        self.spb_shared_baseline_end_ms.setEnabled(baseline_enabled)

    def _source_parcel_preprocessing_options(self):
        """Return the one shared source-parcel preprocessing profile."""
        return {
            "mne_resample_enabled": bool(self.chk_shared_downsample.isChecked()),
            "enable_signal_epoch_disk_cache": bool(self.chk_shared_disk_cache.isChecked()),
            "resample_sfreq": float(self.spb_shared_downsample_sfreq.value()),
            "bandpass_low_hz": float(self.spb_shared_bandpass_low.value()),
            "bandpass_high_hz": float(self.spb_shared_bandpass_high.value()),
            "mne_epoch_reject_enabled": bool(self.chk_shared_mne_reject.isChecked()),
            "reject_mag_ft": float(self.spb_shared_reject_mag.value()),
            "reject_grad_ft_cm": float(self.spb_shared_reject_grad.value()),
            "exclude_trial_outliers": bool(self.chk_shared_exclude_outliers.isChecked()),
            "trial_outlier_z": float(self.spb_shared_outlier_z.value()),
            "parcel_baseline_start": float(
                self.spb_shared_baseline_start_ms.value()
            )
            / 1000.0,
            "parcel_baseline_end": float(self.spb_shared_baseline_end_ms.value())
            / 1000.0,
        }

    def _on_source_run_changed(self, *_args):
        self._refresh_source_evoked_selector()
        self._source_stc = None
        self._source_src = None
        self._source_status_text = f"selected source run: {self._source_run_scope_label()}"
        if getattr(self, "lbl_source_status", None) is not None:
            self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
        if getattr(self, "le_source_inverse", None) is not None:
            self.le_source_inverse.setText("")
        self._update_source_time_controls()
        self._render_source_time_course()

    def _on_source_stimulus_changed(self, *_args):
        self._source_stc = None
        self._source_src = None

    def _on_parcel_heatmap_grouping_changed(self, *_args):
        self._parcel_heatmap_result = None
        self._refresh_parcel_heatmap_stimulus_selector()
        self._reset_parcel_heatmap_dashboard(
            "Grouping changed. Select a stimulus and compute parcel heatmaps.",
        )

    def _on_parcel_heatmap_run_changed(self, *_args):
        self._parcel_heatmap_result = None
        self._refresh_parcel_heatmap_stimulus_selector()
        self._reset_parcel_heatmap_dashboard(
            "Run selection changed. Select a stimulus and compute parcel heatmaps."
        )
        self.lbl_parcel_heatmap_status.setText(
            f"Run scope: {self._parcel_heatmap_run_scope_label()}. Select a stimulus and compute."
        )

    def _on_parcel_heatmap_source_space_changed(self, *_args):
        combo = getattr(self, "cmb_parcel_heatmap_source_space", None)
        atlas = getattr(self, "cmb_parcel_heatmap_parcellation", None)
        if combo is None or atlas is None:
            return
        is_volume = str(combo.currentData() or "surface") == "volume"
        atlas.setEnabled(not is_volume)
        self._parcel_heatmap_result = None
        self._reset_parcel_heatmap_dashboard(
            "Source space changed. Compute parcel heatmaps with the new selection."
        )

    def _on_parcel_heatmap_frequency_band_changed(self, *_args):
        """Invalidate plots whose source time courses used another filter band."""
        self._parcel_heatmap_result = None
        combo = getattr(self, "cmb_parcel_heatmap_frequency_band", None)
        label = str(combo.currentText() or "frequency band") if combo is not None else "frequency band"
        if getattr(self, "lbl_parcel_heatmap_status", None) is not None:
            try:
                self._validate_parcel_heatmap_representation()
                message = f"Frequency band: {label}. Compute parcel heatmaps to apply it."
            except ValueError as exc:
                message = f"Frequency band unavailable: {exc}"
            self.lbl_parcel_heatmap_status.setText(message)
        if getattr(self, "parcel_heatmap_dashboard_splitter", None) is not None:
            self._reset_parcel_heatmap_dashboard(
                "Frequency band changed. Compute parcel heatmaps with the new selection."
            )

    def _parcel_heatmap_representation_key(self):
        combo = getattr(self, "cmb_parcel_heatmap_representation", None)
        return str(combo.currentData() or "activity") if combo is not None else "activity"

    def _validate_parcel_heatmap_representation(self):
        band_settings = self._parcel_heatmap_frequency_band_settings()
        band_combo = getattr(self, "cmb_parcel_heatmap_frequency_band", None)
        band_key = str(band_combo.currentData() or "broadband") if band_combo is not None else "broadband"
        if (
            self._parcel_heatmap_representation_key() == "band_power_db"
            and band_key == "broadband"
        ):
            raise ValueError(
                "Hilbert band power requires a named band such as alpha or beta."
            )
        if (
            self._parcel_heatmap_representation_key() == "band_power_db"
            and float(self.spb_shared_baseline_end_ms.value()) > 0.0
        ):
            raise ValueError(
                "Hilbert band-power baseline must end at or before stimulus onset (0 ms)."
            )
        return band_settings

    def _on_parcel_heatmap_representation_changed(self, *_args):
        """Update controls and invalidate plots when activity/power changes."""
        self._parcel_heatmap_result = None
        is_power = self._parcel_heatmap_representation_key() == "band_power_db"
        absolute = getattr(self, "chk_parcel_heatmap_absolute", None)
        if absolute is not None:
            absolute.setEnabled(not is_power)
            absolute.setToolTip(
                "Hilbert power is already a magnitude and is shown as signed dB change "
                "from prestimulus baseline."
                if is_power
                else "Show mean rectified band-limited parcel activity. Uncheck to display "
                "the signed mean-flip waveform. Neither mode is a PSD power estimate."
            )
        if hasattr(self, "_update_shared_source_preprocessing_controls"):
            self._update_shared_source_preprocessing_controls()
        representation_label = "Hilbert band power" if is_power else "source activity"
        try:
            self._validate_parcel_heatmap_representation()
            message = (
                f"Representation: {representation_label}. Compute parcel heatmaps to apply it."
            )
        except ValueError as exc:
            message = f"Representation unavailable: {exc}"
        if getattr(self, "lbl_parcel_heatmap_status", None) is not None:
            self.lbl_parcel_heatmap_status.setText(message)
        if getattr(self, "parcel_heatmap_dashboard_splitter", None) is not None:
            self._reset_parcel_heatmap_dashboard(
                "Representation changed. Compute parcel heatmaps with the new selection."
            )

    def _parcel_heatmap_frequency_band_settings(self):
        combo = getattr(self, "cmb_parcel_heatmap_frequency_band", None)
        band_key = str(combo.currentData() or "broadband") if combo is not None else "broadband"
        return parcel_trial_frequency_band_settings(
            band_key,
            float(self.spb_shared_bandpass_low.value()),
            float(self.spb_shared_bandpass_high.value()),
            resample_enabled=bool(self.chk_shared_downsample.isChecked()),
            resample_sfreq=float(self.spb_shared_downsample_sfreq.value()),
        )

    def _on_analysis_tab_changed(self, index):
        try:
            tab_text = str(self.tabs.tabText(int(index)))
        except Exception:
            tab_text = ""
        if tab_text == "Source analysis":
            self._refresh_source_run_selector()
            self._refresh_source_evoked_selector()
            if getattr(self, "cmb_source_run", None) is not None and self.cmb_source_run.count() == 0:
                self._source_status_text = "No source runs available. Load or reload a FIF run first."
                self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
        elif tab_text == "Parcel trials":
            self._refresh_parcel_heatmap_run_selector()
            self._refresh_parcel_heatmap_stimulus_selector()
            if self.cmb_parcel_heatmap_run.count() == 0:
                self.lbl_parcel_heatmap_status.setText(
                    "No source runs available. Load or reload a FIF run first."
                )

    def _on_frequency_source_changed(self, *_args):
        is_epochs = str(self.cmb_frequency_source.currentData() or "raw") == "epochs"
        note = (
            "Grouping and stimulus select the condition spectrum."
            if is_epochs else
            "Grouping and stimulus are ignored by raw PSD but still select the Morlet TFR trials."
        )
        self.cmb_frequency_stimulus.setToolTip(note)
        self.cmb_frequency_grouping.setToolTip(note)

    def _on_frequency_grouping_changed(self, *_args):
        self._refresh_frequency_stimulus_selector()

    def _on_source_mode_changed(self, *_args):
        try:
            mode_idx = int(self.cmb_source_mode.currentIndex())
        except Exception:
            mode_idx = 0
        if getattr(self, "source_point_widget", None) is not None:
            self.source_point_widget.setVisible(mode_idx == 0)
        if getattr(self, "source_window_widget", None) is not None:
            self.source_window_widget.setVisible(mode_idx == 1)

    def _source_brain_view_mode(self):
        combo = getattr(self, "cmb_source_brain_view", None)
        if combo is None:
            return "subject"
        mode = str(combo.currentData() or "subject").strip().lower()
        return "fsaverage" if mode == "fsaverage" else "subject"

    def _on_source_brain_view_changed(self, *_args):
        mode = self._source_brain_view_mode()
        if callable(getattr(self, "_source_brain_view_callback", None)):
            try:
                self._source_brain_view_callback(mode)
            except Exception as exc:
                self._source_status_text = f"brain view switch failed: {exc}"
                self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
                return
        if self._source_stc is not None:
            idx = self._source_time_index
            if idx is None:
                idx = self._source_time_index_for_value()
            if idx is not None:
                self._set_source_time_index(int(idx), update_3d=True, render="marker")

    def _on_stimulus_report_recomputed(self):
        """Refresh default channel selections after threshold/rate recompute."""
        QtCore.QTimer.singleShot(0, self._on_epoch_stimulus_changed)
        QtCore.QTimer.singleShot(0, self._on_evoked_stimulus_changed)
        QtCore.QTimer.singleShot(0, self._on_topomap_stimulus_changed)
        QtCore.QTimer.singleShot(0, self._refresh_source_run_selector)
        QtCore.QTimer.singleShot(0, self._refresh_source_evoked_selector)
        QtCore.QTimer.singleShot(0, self._refresh_parcel_heatmap_run_selector)
        QtCore.QTimer.singleShot(0, self._refresh_parcel_heatmap_stimulus_selector)
        QtCore.QTimer.singleShot(0, self._refresh_semantic_stimulus_selector)

    def _on_source_snr_changed(self, *_args):
        snr = max(1e-6, float(self.spb_source_snr.value()))
        lambda2 = 1.0 / (snr ** 2)
        self.lbl_source_lambda2.setText(f"{lambda2:.6f}")

    def _semantic_group_mode(self):
        return self._group_mode_from_index(self.cmb_semantic_grouping.currentIndex())

    def _concept_group_mode(self):
        if self.cmb_concept_grouping.currentIndex() == 3:
            return "semantic_decision"
        return self._group_mode_from_index(self.cmb_concept_grouping.currentIndex())

    def _on_semantic_grouping_changed(self, *_args):
        self._refresh_semantic_stimulus_selector()
        if self._semantic_preview_result is not None:
            self._render_semantic_decoding()

    def _on_semantic_preview_trial_changed(self, *_args):
        if self._semantic_preview_result:
            self._render_semantic_decoding()

    def _semantic_selected_parcellations(self):
        combo = getattr(self, "cmb_semantic_parcellations", None)
        if combo is None:
            return []
        checked = combo.getCheckedItems()
        return [str(v) for v in checked if str(v) and "hemisphere" not in str(v).lower()]

    @staticmethod
    def _annot_name_to_ctx_volume_name(label_name):
        text = str(label_name or "").strip()
        if text.endswith("-lh"):
            return f"ctx-lh-{text[:-3]}"
        if text.endswith("-rh"):
            return f"ctx-rh-{text[:-3]}"
        return text

    @staticmethod
    def _concept_parcel_bucket(name, source_space_kind="surface"):
        text = str(name or "").strip()
        lower = text.lower()
        if str(source_space_kind or "").lower() == "volume":
            if (
                lower.startswith("left")
                or lower.startswith("ctx-lh")
                or lower.startswith("wm-lh")
                or lower.endswith(("-lh", "_lh"))
                or lower.startswith(("lh.", "lh_", "lh-", "l_", "left_"))
                or lower.startswith("left-")
            ):
                return "Left volume"
            if (
                lower.startswith("right")
                or lower.startswith("ctx-rh")
                or lower.startswith("wm-rh")
                or lower.endswith(("-rh", "_rh"))
                or lower.startswith(("rh.", "rh_", "rh-", "r_", "right_"))
                or lower.startswith("right-")
            ):
                return "Right volume"
        if (
            lower.startswith("ctx-lh")
            or lower.endswith("-lh")
            or lower.startswith(("lh.", "lh_", "lh-", "l_", "left_"))
            or lower.startswith("left-")
        ):
            return "Left hemisphere"
        if (
            lower.startswith("ctx-rh")
            or lower.endswith("-rh")
            or lower.startswith(("rh.", "rh_", "rh-", "r_", "right_"))
            or lower.startswith("right-")
        ):
            return "Right hemisphere"
        if lower.startswith("cc"):
            return "Corpus callosum"
        return "Other"

    def _concept_source_space_kind(self):
        combo = getattr(self, "cmb_concept_source_space_kind", None)
        text = str(combo.currentData() if combo is not None else "surface").strip().lower()
        return "volume" if text == "volume" else "surface"

    def _concept_source_parcellation(self):
        if self._concept_source_space_kind() == "volume":
            return "aparc+aseg"
        combo = getattr(self, "cmb_concept_source_parcellation", None)
        return normalize_parcellation_key(combo.currentData() if combo is not None else "aparc")

    def _semantic_all_parcellations(self):
        combo = getattr(self, "cmb_semantic_parcellations", None)
        if combo is None:
            return []
        names = []

        def visit(parent):
            for row in range(parent.rowCount()):
                item = parent.child(row)
                if item is None:
                    continue
                if item.hasChildren():
                    visit(item)
                else:
                    text = str(item.text() or "").strip()
                    if text:
                        names.append(text)

        visit(combo.model.invisibleRootItem())
        return sorted(set(names))

    def _available_concept_parcels(self):
        source_space_kind = self._concept_source_space_kind()
        parcellation = self._concept_source_parcellation()
        subject = str(getattr(self, "_source_subject", "") or "").strip()
        subjects_dir = str(getattr(self, "_source_subjects_dir", "") or os.environ.get("SUBJECTS_DIR", "")).strip()

        if source_space_kind == "volume":
            if subject and subjects_dir:
                try:
                    from core.source_parcellation import get_volume_labels_for_aseg
                    aseg_path = subject_aseg_path(subject, subjects_dir, parcellation=parcellation)
                    if aseg_path:
                        labels, _ = get_volume_labels_for_aseg(str(aseg_path), parcellation=parcellation)
                        names = [
                            str(name)
                            for name in labels
                            if str(name).strip() and "unknown" not in str(name).lower()
                        ]
                        if names:
                            return sorted(set(names))
                except Exception:
                    pass
            base_names = self._semantic_all_parcellations()
            volume_subcortical = [
                "Left-Hippocampus",
                "Right-Hippocampus",
                "Left-Amygdala",
                "Right-Amygdala",
                "Left-Thalamus-Proper",
                "Right-Thalamus-Proper",
                "Left-Caudate",
                "Right-Caudate",
                "Left-Putamen",
                "Right-Putamen",
                "Left-Pallidum",
                "Right-Pallidum",
                "Left-Accumbens-area",
                "Right-Accumbens-area",
                "Left-VentralDC",
                "Right-VentralDC",
                "Brain-Stem",
                "Left-Cerebellum-Cortex",
                "Right-Cerebellum-Cortex",
            ]
            if base_names:
                return sorted(set(base_names + volume_subcortical))
            return volume_subcortical

        if parcellation in HCP_MMP_PARCS:
            if not subjects_dir:
                return []
            try:
                labels = read_surface_labels("fsaverage", subjects_dir, parcellation)
                return sorted(
                    {
                        str(getattr(label, "name", "") or "")
                        for label in labels
                        if str(getattr(label, "name", "") or "").strip()
                        and "unknown" not in str(getattr(label, "name", "") or "").lower()
                    }
                )
            except Exception:
                return []

        if subject and subjects_dir:
            try:
                labels = read_surface_labels(subject, subjects_dir, "aparc")
                names = []
                for label in labels:
                    ctx_name = self._annot_name_to_ctx_volume_name(getattr(label, "name", ""))
                    lname = ctx_name.lower()
                    if not lname.startswith("ctx-") or lname.endswith("unknown"):
                        continue
                    names.append(ctx_name)
                if names:
                    return sorted(set(names))
            except Exception:
                pass
        return self._semantic_all_parcellations()

    def _refresh_concept_parcel_selector(self, selected=None):
        combo = getattr(self, "cmb_concept_parcels", None)
        if combo is None:
            return
        if selected is None:
            selected = self._concept_selected_parcels()
        selected_set = {str(v) for v in selected or []}
        names = [str(v) for v in self._available_concept_parcels() if str(v).strip()]
        names = sorted(set(names))
        self._concept_available_parcels = names

        combo.model.clear()
        combo.channel_dict = {}
        combo.setEditText("")
        if not names:
            return

        source_space_kind = self._concept_source_space_kind()
        bucket_order = (
            ["Left volume", "Right volume", "Corpus callosum", "Other"]
            if source_space_kind == "volume"
            else ["Left hemisphere", "Right hemisphere", "Corpus callosum", "Other"]
        )
        tree = {bucket: (False, {}) for bucket in bucket_order}
        for name in names:
            bucket = self._concept_parcel_bucket(name, source_space_kind=source_space_kind)
            if bucket not in tree:
                tree[bucket] = (False, {})
            tree[bucket][1][name] = (name in selected_set, {})
        tree = {bucket: value for bucket, value in tree.items() if value[1]}
        quick_filters = {
            "⚡ Select Visual parcels (Occipital & ventral)": (False, {}),
            "⚡ Select Semantic parcels (Temporal, parietal, IFG)": (False, {}),
            "⚡ Select Visual + Semantic parcels": (False, {}),
            "⚡ Clear selection": (False, {}),
        }
        tree = {"⚡ Preset Filters": (False, quick_filters), **tree}
        combo.addItems(tree)
        combo.updateText()

    def _select_concept_parcel_filter(self, filter_name):
        combo = getattr(self, "cmb_concept_parcels", None)
        if combo is None:
            return
        available = getattr(self, "_concept_available_parcels", []) or []
        if not available:
            return
        parcellation = self._concept_source_parcellation()

        if filter_name not in {"visual", "semantic", "visual_semantic", "both", "all", "clear"}:
            return
        selected = filter_parcels(available, filter_name, parcellation)

        self._refresh_concept_parcel_selector(selected=selected)

    def _on_concept_parcel_item_changed(self, item):
        if getattr(self, "_parcel_selection_in_progress", False):
            return
        text = str(item.text() if item is not None else "").strip()
        if "Visual parcels" in text and item.checkState() == QtCore.Qt.Checked:
            self._parcel_selection_in_progress = True
            try:
                self._select_concept_parcel_filter("visual")
            finally:
                self._parcel_selection_in_progress = False
        elif "Semantic parcels" in text and item.checkState() == QtCore.Qt.Checked:
            self._parcel_selection_in_progress = True
            try:
                self._select_concept_parcel_filter("semantic")
            finally:
                self._parcel_selection_in_progress = False
        elif "Visual + Semantic" in text and item.checkState() == QtCore.Qt.Checked:
            self._parcel_selection_in_progress = True
            try:
                self._select_concept_parcel_filter("visual_semantic")
            finally:
                self._parcel_selection_in_progress = False
        elif "Clear selection" in text and item.checkState() == QtCore.Qt.Checked:
            self._parcel_selection_in_progress = True
            try:
                self._select_concept_parcel_filter("clear")
            finally:
                self._parcel_selection_in_progress = False

    def _concept_selected_parcels(self):
        combo = getattr(self, "cmb_concept_parcels", None)
        if combo is None:
            return []
        available = set(getattr(self, "_concept_available_parcels", []) or [])
        checked = combo.getCheckedItems()
        return [str(v) for v in checked if str(v) in available]

    def _browse_inverse_operator_file(self):
        start_dir = os.path.dirname(str(self._fif_filename)) if self._fif_filename else os.getcwd()
        path = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select inverse operator FIF",
            start_dir,
            "FIF files (*.fif);;All files (*.*)",
        )[0]
        if path:
            self.le_source_inverse.setText(str(path))

    def _browse_concept_embedding_file(self):
        start_dir = os.path.dirname(str(self._fif_filename)) if self._fif_filename else os.getcwd()
        path = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select concept/image embedding dictionary",
            start_dir,
            "Embedding files (*.json *.h5 *.hdf5 *.npz *.csv *.tsv);;All files (*.*)",
        )[0]
        if path:
            self.le_concept_embedding_path.setText(str(path))

    def _resolve_selected_epoch_indices(self, combo):
        if self._epochs is None:
            return []
        mapping = self._epoch_indices_by_stimulus()
        selected = str(combo.currentText()) if combo is not None else "All stimuli"
        if selected == "All stimuli":
            all_idx = []
            for idxs in mapping.values():
                all_idx.extend(list(idxs))
            if all_idx:
                return sorted(set(all_idx))
            return list(range(len(self._epochs)))
        return list(mapping.get(selected, []))

    def _resolve_selected_topomap_indices(self):
        if self._epochs is None and not self._topomap_epochs_by_path:
            return []
        mapping = self._topomap_indices_by_stimulus()
        selected = str(self.cmb_topomap_stimulus.currentText()) if self.cmb_topomap_stimulus is not None else "All stimuli"
        if selected == "All stimuli":
            all_idx = []
            for idxs in mapping.values():
                all_idx.extend(list(idxs))
            if all_idx:
                return sorted(set(all_idx))
            fallback = []
            for path, epochs in self._selected_topomap_run_entries():
                fallback.extend((path, i) for i in range(len(epochs)))
            return fallback
        return list(mapping.get(selected, []))

    def _resolve_selected_source_indices(self):
        selected_paths = self._selected_source_run_paths()
        if not selected_paths:
            return []
        mapping = self._source_indices_by_stimulus()
        selected = str(self.cmb_source_evoked.currentText()) if self.cmb_source_evoked is not None else "All stimuli"
        if selected == "All stimuli":
            all_idx = []
            for idxs in mapping.values():
                all_idx.extend(list(idxs))
            if all_idx:
                return sorted(set(all_idx), key=lambda item: (str(item[0]), int(item[1])))
            fallback = []
            for path in selected_paths:
                epochs = self._topomap_epochs_by_path.get(path)
                if epochs is not None:
                    fallback.extend((path, i) for i in range(len(epochs)))
                    continue
                trial_df = self._trial_df_for_topomap_path(path)
                if not trial_df.empty:
                    fallback.extend((path, i) for i in range(len(trial_df)))
            return fallback
        return list(mapping.get(selected, []))

    def _source_epoch_indices_by_run(self, source_indices, include_empty=False):
        by_run = {path: [] for path, _epochs in self._selected_source_run_entries()} if include_empty else {}
        for item in list(source_indices or []):
            try:
                path, idx = item
            except Exception:
                continue
            path = self._normalize_fif_path(path)
            if not path:
                continue
            by_run.setdefault(path, []).append(int(idx))
        return {
            path: sorted(set(int(i) for i in idxs))
            for path, idxs in by_run.items()
        }

    def _parse_topomap_times(self):
        text = str(self.le_topomap_times.text() or "").strip()
        if not text:
            return [0.0, 0.08, 0.10, 0.12, 0.20, 0.30, 0.40]
        values = []
        for chunk in text.replace(";", ",").split(","):
            token = chunk.strip()
            if not token:
                continue
            try:
                val = float(token)
            except Exception:
                continue
            if np.isfinite(val):
                values.append(float(val))
        if not values:
            return [0.0, 0.08, 0.10, 0.12, 0.20, 0.30, 0.40]
        return values

    def _parse_source_views(self):
        text = str(getattr(self, "le_source_views", None).text() if getattr(self, "le_source_views", None) is not None else "").strip()
        if not text:
            return ["lat", "med", "dor", "ven"]
        values = []
        valid = {"lat", "med", "dor", "ven", "ros", "cau", "fro", "par"}
        for chunk in text.replace(";", ",").split(","):
            token = str(chunk).strip().lower()
            if token in valid and token not in values:
                values.append(token)
        if not values:
            return ["lat", "med", "dor", "ven"]
        return values

    @staticmethod
    def _split_source_views(source_views):
        primary = []
        secondary = []
        for view in [str(v).strip().lower() for v in source_views or []]:
            if view in {"dor", "ven"}:
                if view not in secondary:
                    secondary.append(view)
            elif view and view not in primary:
                primary.append(view)
        if not primary:
            primary = ["lat", "med"]
        return primary, secondary

    @staticmethod
    def _ensure_fsaverage_in_subjects_dir(subjects_dir):
        if not subjects_dir:
            return None
        try:
            import mne
            from pathlib import Path

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

    def _stc_for_source_brain_view_plot(self, stc):
        """Return `(stc, subject, subjects_dir)` for the Source tab screenshot target."""
        if stc is None:
            return None, None, None
        source_subject = str(getattr(stc, "subject", "") or "")
        if is_volume_stc(stc):
            return stc, source_subject or str(self._source_subject or ""), str(self._source_subjects_dir or "")
        mode = self._source_brain_view_mode()
        source_subjects_dir = str(self._source_subjects_dir or "")
        if mode == "fsaverage":
            target_subject = "fsaverage"
            target_subjects_dir = self._ensure_fsaverage_in_subjects_dir(source_subjects_dir)
            if not target_subjects_dir:
                try:
                    import mne
                    target_subjects_dir = str(mne.datasets.sample.data_path() / "subjects")
                except Exception:
                    target_subjects_dir = source_subjects_dir
        else:
            target_subject = str(self._source_subject or source_subject or "")
            target_subjects_dir = source_subjects_dir
        if not target_subject or not target_subjects_dir:
            return stc, target_subject, target_subjects_dir
        if not source_subject or source_subject == target_subject:
            return stc, target_subject, target_subjects_dir
        try:
            import mne
            vertices_sig = tuple(tuple(np.asarray(v, dtype=int).tolist()) for v in stc.vertices)
            morph_key = (source_subject, target_subject, str(target_subjects_dir), vertices_sig)
            morph = self._source_plot_morph_cache.get(morph_key)
            if morph is None:
                morph = mne.compute_source_morph(
                    stc,
                    subject_from=source_subject,
                    subject_to=target_subject,
                    subjects_dir=target_subjects_dir,
                    spacing=5,
                    warn=False,
                    verbose="error",
                )
                self._source_plot_morph_cache[morph_key] = morph
            return morph.apply(stc), target_subject, target_subjects_dir
        except Exception:
            return stc, source_subject, source_subjects_dir

    def _render_stc_brain_screenshots(self, stc, initial_time, subject_for_plot, subjects_dir_for_plot):
        """Render source screenshots, splitting dorsal/ventral into a side panel."""
        if stc is None or not subjects_dir_for_plot:
            return None, None, [], []
        if is_volume_stc(stc):
            source_views = self._parse_source_views()
            primary_views, secondary_views = self._split_source_views(source_views)
            return None, None, primary_views, secondary_views
        screenshot_enabled = bool(
            getattr(getattr(self, "chk_source_brain_screenshot", None), "isChecked", lambda: False)()
        )
        if not screenshot_enabled:
            source_views = self._parse_source_views()
            primary_views, secondary_views = self._split_source_views(source_views)
            return None, None, primary_views, secondary_views

        def _screenshot_for_views(views):
            if not views:
                return None
            brain = None
            try:
                brain = stc.plot(
                    initial_time=float(initial_time),
                    hemi="split",
                    views=list(views),
                    subjects_dir=subjects_dir_for_plot,
                    subject=subject_for_plot,
                    time_viewer=False,
                    show_traces=False,
                    brain_kwargs={"show": False},
                )
                if brain is not None:
                    try:
                        return brain.screenshot(mode="rgb")
                    except TypeError:
                        return brain.screenshot()
            except Exception:
                return None
            finally:
                if brain is not None:
                    try:
                        brain.close()
                    except Exception:
                        pass
            return None

        source_views = self._parse_source_views()
        primary_views, secondary_views = self._split_source_views(source_views)
        primary_image = _screenshot_for_views(primary_views)
        secondary_image = _screenshot_for_views(secondary_views) if secondary_views else None
        return primary_image, secondary_image, primary_views, secondary_views

    def _surface_stc_for_subject(self, stc, target_subject, target_subjects_dir):
        """Morph a surface STC to target subject when needed."""
        if stc is None or not is_surface_stc(stc):
            return stc, "native"
        source_subject = str(getattr(stc, "subject", "") or "")
        target_subject = str(target_subject or "")
        target_subjects_dir = str(target_subjects_dir or "")
        if not source_subject or not target_subject or source_subject == target_subject:
            return stc, "native"
        try:
            import mne
            vertices_sig = tuple(tuple(np.asarray(v, dtype=int).tolist()) for v in stc.vertices)
            morph_key = (source_subject, target_subject, target_subjects_dir, vertices_sig)
            morph = self._source_plot_morph_cache.get(morph_key)
            if morph is None:
                morph = mne.compute_source_morph(
                    stc,
                    subject_from=source_subject,
                    subject_to=target_subject,
                    subjects_dir=target_subjects_dir,
                    spacing=5,
                    warn=False,
                    verbose="error",
                )
                self._source_plot_morph_cache[morph_key] = morph
            return morph.apply(stc), f"morphed {source_subject}->{target_subject}"
        except Exception:
            return stc, "native"

    def _source_top_roi_count(self):
        spin = getattr(self, "spb_source_top_roi_count", None)
        if spin is None:
            return 0
        try:
            return max(0, int(spin.value()))
        except Exception:
            return 0

    def _source_roi_parcellation(self, stc):
        combo = getattr(self, "cmb_source_roi_parcellation", None)
        parcellation = "auto"
        if combo is not None:
            parcellation = str(combo.currentData() or "auto")
        if parcellation == "auto":
            return "aparc+aseg" if is_volume_stc(stc) else "aparc"
        return normalize_parcellation_key(parcellation)

    def _source_top_label_time_courses(self, stc, top_n):
        """Return `(times, label_tc, label_names, top_idxs, title_suffix)` for ROI traces."""
        if stc is None or top_n <= 0:
            return None
        parcellation = self._source_roi_parcellation(stc)
        times = np.asarray(getattr(stc, "times", []), dtype=float)
        if times.size == 0:
            return None
        if is_volume_stc(stc):
            subject = str(getattr(stc, "subject", "") or self._source_subject or "")
            subjects_dir = str(self._source_subjects_dir or "")
            aseg_path = subject_aseg_path(subject, subjects_dir)
            label_tc, label_names = volume_label_time_courses(
                stc,
                self._source_src,
                aseg_path,
                use_abs=True,
            )
            top_idxs = top_label_indices(label_tc, label_names, top_n=top_n)
            return times, label_tc, label_names, top_idxs, aseg_name_from_path(aseg_path)

        if parcellation == "aparc+aseg":
            parcellation = "aparc"
        if parcellation in HCP_MMP_PARCS:
            target_subject = "fsaverage"
            target_subjects_dir = self._ensure_fsaverage_in_subjects_dir(self._source_subjects_dir)
            if not target_subjects_dir:
                try:
                    import mne
                    target_subjects_dir = str(mne.datasets.sample.data_path() / "subjects")
                    mne.datasets.fetch_fsaverage(subjects_dir=target_subjects_dir, verbose=False)
                except Exception:
                    target_subjects_dir = str(self._source_subjects_dir or "")
            stc_for_labels, _space = self._surface_stc_for_subject(
                stc,
                target_subject,
                target_subjects_dir,
            )
            labels = read_surface_labels(target_subject, target_subjects_dir, parcellation)
        else:
            stc_for_labels, target_subject, target_subjects_dir = self._stc_for_source_brain_view_plot(stc)
            labels = read_surface_labels(target_subject, target_subjects_dir, "aparc")

        label_tc, label_names, _label_objects = surface_label_time_courses(
            stc_for_labels,
            labels,
            use_abs=True,
        )
        top_idxs = top_label_indices(label_tc, label_names, top_n=top_n)
        label_times = np.asarray(getattr(stc_for_labels, "times", times), dtype=float)
        return label_times, label_tc, label_names, top_idxs, parcellation

    def _topomap_view_mode(self):
        try:
            idx = int(self.cmb_topomap_view_mode.currentIndex())
        except Exception:
            idx = 0
        return "slider" if idx == 1 else "times"

    def _topomap_slider_time_value(self):
        slider = getattr(self, "sld_topomap_time", None)
        if slider is None:
            return 0.10
        try:
            return float(slider.value()) / 1000.0
        except Exception:
            return 0.10

    def _current_topomap_times(self):
        if self._topomap_view_mode() == "slider":
            return [self._topomap_slider_time_value()]
        return self._parse_topomap_times()

    def _topomap_bandpass_settings(self):
        """Return `(l_freq, h_freq, label)` for topomap display filtering."""
        try:
            l_freq = float(self.spb_topomap_bandpass_low.value())
        except Exception:
            l_freq = CONCEPT_PREPROCESSING_DEFAULTS["bandpass_low_hz"]
        try:
            h_freq = float(self.spb_topomap_bandpass_high.value())
        except Exception:
            h_freq = CONCEPT_PREPROCESSING_DEFAULTS["bandpass_high_hz"]
        l_freq = None if not np.isfinite(l_freq) or l_freq <= 0.0 else float(l_freq)
        h_freq = None if not np.isfinite(h_freq) or h_freq <= 0.0 else float(h_freq)
        if l_freq is None and h_freq is None:
            return None, None, "unfiltered"
        if l_freq is not None and h_freq is not None and h_freq <= l_freq:
            raise ValueError("Topomap bandpass high cutoff must be greater than low cutoff.")
        epoch_sources = [epochs for _path, epochs in self._selected_topomap_run_entries()]
        if not epoch_sources and getattr(self, "_epochs", None) is not None:
            epoch_sources = [self._epochs]
        nyquist_values = []
        for epochs in epoch_sources:
            try:
                nyquist_values.append(float(epochs.info["sfreq"]) / 2.0)
            except Exception:
                continue
        if nyquist_values:
            nyq = min(nyquist_values)
            if h_freq is not None and h_freq >= nyq:
                raise ValueError(
                    f"Topomap bandpass high cutoff ({h_freq:g} Hz) must be below Nyquist "
                    f"({nyq:g} Hz for selected run scope)."
                )
        low_label = "0" if l_freq is None else f"{l_freq:g}"
        high_label = "Nyq" if h_freq is None else f"{h_freq:g}"
        return l_freq, h_freq, f"bandpass={low_label}-{high_label} Hz"

    def _source_bandpass_settings(self):
        """Return `(enabled, l_freq, h_freq, label)` for source-estimate filtering."""
        try:
            l_freq = float(self.spb_source_bandpass_low.value())
        except Exception:
            l_freq = CONCEPT_PREPROCESSING_DEFAULTS["bandpass_low_hz"]
        try:
            h_freq = float(self.spb_source_bandpass_high.value())
        except Exception:
            h_freq = CONCEPT_PREPROCESSING_DEFAULTS["bandpass_high_hz"]
        l_freq = None if not np.isfinite(l_freq) or l_freq <= 0.0 else float(l_freq)
        h_freq = None if not np.isfinite(h_freq) or h_freq <= 0.0 else float(h_freq)
        if l_freq is None and h_freq is None:
            return False, None, None, "unfiltered"
        if l_freq is not None and h_freq is not None and h_freq <= l_freq:
            raise ValueError("Source bandpass high cutoff must be greater than low cutoff.")
        low_label = "0" if l_freq is None else f"{l_freq:g}"
        high_label = "Nyq" if h_freq is None else f"{h_freq:g}"
        return True, l_freq, h_freq, f"bandpass={low_label}-{high_label} Hz"

    def _filtered_topomap_epochs(self, epochs, idx):
        """Return selected epochs, optionally filtered on a copy for topomap display."""
        if epochs is None:
            return None, "No epochs available for topomap.", "unfiltered"
        l_freq, h_freq, filter_label = self._topomap_bandpass_settings()
        epochs_sel = epochs[idx]
        if l_freq is None and h_freq is None:
            return epochs_sel, None, filter_label
        try:
            epochs_sel = epochs_sel.copy()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="filter_length.*is longer than the signal.*",
                    category=RuntimeWarning,
                )
                epochs_sel.filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    picks="meg",
                    n_jobs=1,
                    verbose="error",
                )
            return epochs_sel, None, filter_label
        except Exception as exc:
            return None, f"Could not apply topomap bandpass filter: {exc}", filter_label

    def _combine_topomap_evokeds(self, evokeds):
        if not evokeds:
            return None, "No evoked responses available to combine."
        if len(evokeds) == 1:
            return evokeds[0], None
        try:
            import mne

            common_chs = list(evokeds[0].ch_names)
            for evk in evokeds[1:]:
                available = set(evk.ch_names)
                common_chs = [ch for ch in common_chs if ch in available]
            if not common_chs:
                return None, "Runs do not share any channels for this topomap channel type."

            tmin = max(float(evk.times[0]) for evk in evokeds)
            tmax = min(float(evk.times[-1]) for evk in evokeds)
            if tmax <= tmin:
                return None, "Runs do not share an overlapping epoch time window."

            aligned = []
            for evk in evokeds:
                one = evk.copy().pick(common_chs)
                one.crop(tmin=tmin, tmax=tmax)
                aligned.append(one)
            combined = mne.combine_evoked(aligned, weights="nave")
            combined.comment = "All runs"
            return combined, None
        except Exception as exc:
            return None, f"Could not combine evoked responses across runs: {exc}"

    def _set_topomap_slider_label(self, value=None):
        if value is None:
            value = self._topomap_slider_time_value()
        if getattr(self, "lbl_topomap_slider_time", None) is not None:
            self.lbl_topomap_slider_time.setText(f"{float(value):.3f} s")

    def _update_topomap_slider_range(self):
        slider = getattr(self, "sld_topomap_time", None)
        if slider is None:
            return
        tmin = 0.0
        tmax = 0.4
        epochs = getattr(self, "_epochs", None)
        if epochs is not None and len(epochs) > 0:
            try:
                tmin = float(epochs.times[0])
                tmax = float(epochs.times[-1])
            except Exception:
                tmin = 0.0
                tmax = 0.4
        min_ms = int(np.floor(tmin * 1000.0))
        max_ms = int(np.ceil(tmax * 1000.0))
        if max_ms <= min_ms:
            max_ms = min_ms + 1
        slider.blockSignals(True)
        slider.setRange(min_ms, max_ms)
        slider.setSingleStep(max(1, int(round((max_ms - min_ms) / 100.0))))
        slider.setPageStep(max(1, int(round((max_ms - min_ms) / 20.0))))
        curr = slider.value()
        if curr < min_ms or curr > max_ms:
            default_ms = int(round(min(0.10, tmax) * 1000.0))
            curr = min(max(default_ms, min_ms), max_ms)
            slider.setValue(curr)
        slider.blockSignals(False)
        self._set_topomap_slider_label(float(curr) / 1000.0)

    def _on_topomap_view_mode_changed(self, *_args):
        use_slider = self._topomap_view_mode() == "slider"
        if getattr(self, "le_topomap_times", None) is not None:
            self.le_topomap_times.setVisible(not use_slider)
        if getattr(self, "topomap_slider_widget", None) is not None:
            self.topomap_slider_widget.setVisible(use_slider)
        if getattr(self, "btn_plot_topomap", None) is not None:
            self.btn_plot_topomap.setVisible(not use_slider)
        if getattr(self, "btn_plot_joint", None) is not None:
            self.btn_plot_joint.setVisible(not use_slider)
        self._update_topomap_slider_range()
        if use_slider and self._selected_topomap_run_entries():
            self._plot_topomap_only()

    def _on_topomap_slider_changed(self, value):
        self._set_topomap_slider_label(float(value) / 1000.0)
        if self._topomap_view_mode() == "slider":
            self._plot_topomap_only()

    def _on_topomap_bandpass_changed(self, *_args):
        if not self._selected_topomap_run_entries():
            return
        state = self._plot_state.get("topomap", {}) if isinstance(getattr(self, "_plot_state", None), dict) else {}
        has_topomap_plot = bool(state.get("figs"))
        if self._topomap_view_mode() == "slider" or has_topomap_plot:
            self._plot_topomap_only()

    def _plot_epoch_image(self):
        if self._epochs is None or len(self._epochs) == 0:
            self._reset_plot_placeholder("epoch", self.lbl_epoch_placeholder, "No epochs available to plot.")
            return
        selected_channels = self._selected_data_channels(self.lst_epoch_channels)
        show_events = self._events_selected(self.lst_epoch_channels)
        if not selected_channels and len(self._epochs.ch_names) > 0:
            selected_channels = [self._epochs.ch_names[0]]
        idx = self._resolve_selected_epoch_indices(self.cmb_epoch_stimulus)
        if not idx:
            self._reset_plot_placeholder("epoch", self.lbl_epoch_placeholder, "No trials matched the selected stimulus.")
            return
        try:
            epochs_sel = self._epochs[idx]
            combine_mode = "mean" if self.chk_epoch_combine.isChecked() else None
            fig = epochs_sel.plot_image(
                picks=selected_channels,
                combine=combine_mode,
                colorbar=True,
                ts_args={"vlines": [0.0] if show_events else []},
                show=False,
            )
            if fig is None:
                raise RuntimeError("MNE did not return a figure.")
            self._set_plot("epoch", self.epoch_plot_layout, self.lbl_epoch_placeholder, fig)
        except Exception as exc:
            self._reset_plot_placeholder("epoch", self.lbl_epoch_placeholder, f"Could not render epoch image: {exc}")

    def _plot_evoked_response(self):
        if self._epochs is None or len(self._epochs) == 0:
            self._reset_plot_placeholder("evoked", self.lbl_evoked_placeholder, "No epochs available to plot.")
            return
        selected_channels = self._selected_data_channels(self.lst_evoked_channels)
        show_events = self._events_selected(self.lst_evoked_channels)
        if not selected_channels and len(self._epochs.ch_names) > 0:
            selected_channels = [self._epochs.ch_names[0]]
        self.lbl_evoked_channels_info.setText(
            f"Channels ({len(selected_channels)}): {self._format_channel_list(selected_channels)}"
        )

        mapping = self._evoked_indices_by_stimulus()
        if not mapping:
            self._reset_plot_placeholder("evoked", self.lbl_evoked_placeholder, "No trials matched the selected stimulus.")
            return
        selected_stim = str(self.cmb_evoked_stimulus.currentText())
        if selected_stim == "All stimuli":
            stim_items = [(k, v) for k, v in mapping.items() if len(v) > 0]
        else:
            stim_items = [(selected_stim, list(mapping.get(selected_stim, [])))]
        if not stim_items:
            self._reset_plot_placeholder("evoked", self.lbl_evoked_placeholder, "No stimulus groups available for evoked plot.")
            return

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for stim_label, idx in stim_items:
                if not idx:
                    continue
                evk = self._epochs[idx].average()
                data = evk.copy().pick(selected_channels).data
                if data.ndim != 2 or data.shape[1] == 0:
                    continue
                trace = np.mean(data, axis=0)
                ax.plot(evk.times, trace, linewidth=1.6, label=f"{stim_label} (n={len(idx)})")
            if show_events:
                ax.axvline(0.0, color="k", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean amplitude")
            group_label = self._group_mode_label(self._evoked_group_mode())
            ax.set_title(
                f"Evoked responses by {group_label} (channels={len(selected_channels)})"
            )
            if len(stim_items) <= 12:
                ax.legend(loc="best", fontsize=8)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                fig.tight_layout()
            self._set_plot("evoked", self.evoked_plot_layout, self.lbl_evoked_placeholder, fig)
        except Exception as exc:
            self._reset_plot_placeholder("evoked", self.lbl_evoked_placeholder, f"Could not render evoked plot: {exc}")

    def _frequency_settings(self):
        """Return validated frequency limits and selected channels."""
        instance = self._epochs if self._epochs is not None else self._raw
        if instance is None:
            raise RuntimeError("No raw or epoched MEG data are loaded.")
        sfreq = float(instance.info["sfreq"])
        nyquist = sfreq / 2.0
        fmin = max(0.1, float(self.spb_frequency_fmin.value()))
        fmax = min(float(self.spb_frequency_fmax.value()), nyquist * 0.99)
        if fmax <= fmin:
            raise ValueError(f"Frequency maximum must exceed minimum and remain below {nyquist:.1f} Hz.")
        available = set(getattr(instance, "ch_names", []))
        channels = [
            name for name in self._selected_channels(self.lst_frequency_channels)
            if name in available
        ]
        if not channels:
            channels = list(getattr(instance, "ch_names", []))[:24]
        if not channels:
            raise RuntimeError("No MEG channels are available.")
        return fmin, fmax, channels, sfreq

    @staticmethod
    def _spectrum_arrays(spectrum):
        try:
            data, freqs = spectrum.get_data(return_freqs=True)
        except TypeError:
            data = spectrum.get_data()
            freqs = getattr(spectrum, "freqs", [])
        data = np.asarray(data, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        if data.ndim not in (2, 3) or freqs.ndim != 1 or data.shape[-1] != freqs.size:
            raise RuntimeError("MNE returned an unexpected spectrum shape.")
        return data, freqs

    @staticmethod
    def _frequency_band_summary(power, freqs):
        """Return relative band power for a linear-power spectrum."""
        power = np.asarray(power, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        integrate = getattr(np, "trapezoid", np.trapz)
        finite = np.isfinite(power) & np.isfinite(freqs)
        if np.sum(finite) < 2:
            return []
        total = float(integrate(power[finite], freqs[finite]))
        if not np.isfinite(total) or total <= 0:
            return []
        bands = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
                 ("beta", 13, 30), ("gamma", 30, 80))
        result = []
        for name, low, high in bands:
            mask = finite & (freqs >= low) & (freqs < high)
            if np.sum(mask) >= 2:
                value = float(integrate(power[mask], freqs[mask])) / total * 100.0
                result.append((name, value))
        return result

    @staticmethod
    def _line_noise_excess_db(power, freqs, line_freq=60.0):
        power = np.asarray(power, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        signal_mask = np.abs(freqs - float(line_freq)) <= 1.0
        reference_mask = ((freqs >= line_freq - 6.0) & (freqs <= line_freq - 2.0)) | (
            (freqs >= line_freq + 2.0) & (freqs <= line_freq + 6.0)
        )
        if not np.any(signal_mask) or not np.any(reference_mask):
            return None
        signal_power = float(np.nanmean(power[signal_mask]))
        reference_power = float(np.nanmedian(power[reference_mask]))
        if signal_power <= 0 or reference_power <= 0:
            return None
        return 10.0 * np.log10(signal_power / reference_power)

    @staticmethod
    def _band_separation_scores(epoch_power, freqs, groups):
        """Estimate univariate condition separation (eta squared) by frequency band."""
        values = np.asarray(epoch_power, dtype=float)
        if values.ndim != 3 or len(groups) != values.shape[0]:
            return []
        # One log-power feature per trial and band, averaged across selected sensors.
        values = np.nanmean(values, axis=1)
        labels = np.asarray(groups, dtype=object)
        unique = [label for label in dict.fromkeys(labels.tolist()) if np.sum(labels == label) >= 2]
        if len(unique) < 2:
            return []
        results = []
        for name, low, high in (("theta", 4, 8), ("alpha", 8, 13),
                                ("beta", 13, 30), ("gamma", 30, 80)):
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                continue
            feature = 10.0 * np.log10(
                np.maximum(np.nanmean(values[:, mask], axis=1), np.finfo(float).tiny)
            )
            valid = np.isfinite(feature) & np.isin(labels, unique)
            feature = feature[valid]
            used_labels = labels[valid]
            if feature.size < 4:
                continue
            grand = float(np.mean(feature))
            total_ss = float(np.sum((feature - grand) ** 2))
            if total_ss <= 0:
                continue
            between_ss = 0.0
            for label in unique:
                group_values = feature[used_labels == label]
                if group_values.size:
                    between_ss += group_values.size * float((np.mean(group_values) - grand) ** 2)
            results.append((name, between_ss / total_ss))
        return results

    def _plot_frequency_psd(self):
        try:
            fmin, fmax, channels, sfreq = self._frequency_settings()
            source = str(self.cmb_frequency_source.currentData() or "raw")
            if source == "raw":
                if self._raw is None:
                    raise RuntimeError("Continuous raw data are not available for this FIF load.")
                data = self._raw.copy().pick(channels)
                if data.n_times > int(round(60.0 * sfreq)):
                    data.crop(tmin=0.0, tmax=60.0, include_tmax=False)
                n_fft = min(4096, int(data.n_times))
                if n_fft < 16:
                    raise RuntimeError("The raw recording is too short for a stable PSD.")
                spectrum = data.compute_psd(
                    method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft,
                    n_per_seg=n_fft, n_overlap=n_fft // 2, window="hann", verbose="error"
                )
                psd, freqs = self._spectrum_arrays(spectrum)
                # Channel normalization makes magnetometer and gradiometer spectral
                # shapes comparable despite their different physical units.
                channel_reference = np.nanmedian(psd, axis=-1, keepdims=True)
                psd_relative = psd / np.maximum(channel_reference, np.finfo(float).tiny)
                mean_linear = np.nanmean(psd_relative, axis=0)
                channel_db = 10.0 * np.log10(
                    np.maximum(psd_relative, np.finfo(float).tiny)
                )

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9.5, 5.2))
                for trace in channel_db[:80]:
                    ax.plot(freqs, trace, color="#94a3b8", linewidth=0.6, alpha=0.28)
                ax.plot(
                    freqs, 10.0 * np.log10(np.maximum(mean_linear, np.finfo(float).tiny)),
                    color="#1d4ed8", linewidth=2.0, label="channel mean"
                )
                for line in (60.0, 120.0):
                    if fmin <= line <= fmax:
                        ax.axvline(line, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.75)
                ax.set_title("Continuous-data spectral QC (MNE Welch PSD, first 60 s)")
                ax.set_ylabel("Relative PSD (dB re channel median)")
                details = [f"raw PSD: {len(channels)} channels, {data.times[-1]:.1f} s"]
                line_excess = self._line_noise_excess_db(mean_linear, freqs)
                if line_excess is not None:
                    details.append(f"60 Hz excess={line_excess:+.1f} dB vs neighbors")
                bad_selected = sorted(set(channels) & set(self._raw.info.get("bads", [])))
                if bad_selected:
                    details.append(f"marked bad={len(bad_selected)}")
                band_summary = self._frequency_band_summary(mean_linear, freqs)
                if band_summary:
                    details.append("relative bands: " + ", ".join(f"{n} {v:.1f}%" for n, v in band_summary))
            else:
                if self._epochs is None or len(self._epochs) == 0:
                    raise RuntimeError("No epochs are available for condition spectra.")
                epochs = self._epochs.copy().pick(channels)
                n_fft = min(2048, int(epochs.get_data(copy=False).shape[-1]))
                if n_fft < 16:
                    raise RuntimeError("Epochs are too short for a stable PSD.")
                spectrum = epochs.compute_psd(
                    method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft,
                    n_per_seg=n_fft, n_overlap=n_fft // 2, window="hann", verbose="error"
                )
                psd, freqs = self._spectrum_arrays(spectrum)
                channel_reference = np.nanmedian(psd, axis=-1, keepdims=True)
                psd_relative = psd / np.maximum(channel_reference, np.finfo(float).tiny)
                mapping = self._frequency_indices_by_stimulus()
                selected = str(self.cmb_frequency_stimulus.currentText() or "All stimuli")
                if selected == "All stimuli":
                    items = [(label, idx) for label, idx in mapping.items() if idx]
                    if not items:
                        items = [("all epochs", list(range(len(epochs))))]
                else:
                    items = [(selected, mapping.get(selected, []))]
                items = items[:12]

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9.5, 5.2))
                plotted = 0
                for label, indices in items:
                    indices = [int(i) for i in indices if 0 <= int(i) < psd.shape[0]]
                    if not indices:
                        continue
                    trial_power = np.nanmean(psd_relative[indices], axis=1)
                    trial_db = 10.0 * np.log10(np.maximum(trial_power, np.finfo(float).tiny))
                    mean_db = np.nanmean(trial_db, axis=0)
                    sem_db = np.nanstd(trial_db, axis=0) / max(1.0, np.sqrt(len(indices)))
                    line = ax.plot(freqs, mean_db, linewidth=1.7, label=f"{label} (n={len(indices)})")[0]
                    ax.fill_between(
                        freqs, mean_db - sem_db, mean_db + sem_db,
                        color=line.get_color(), alpha=0.12, linewidth=0
                    )
                    plotted += 1
                if not plotted:
                    raise RuntimeError("No trials matched the selected stimulus grouping.")
                ax.set_title(f"Epoched condition spectra ({self._group_mode_label(self._frequency_group_mode())})")
                ax.set_ylabel("Relative PSD (dB re channel median)")
                ax.legend(loc="best", fontsize=8, ncol=2)
                details = [f"epoch PSD: {len(epochs)} trials, {len(channels)} channels, {plotted} group(s)"]
                labels_by_epoch = np.full(len(epochs), "", dtype=object)
                for label, indices in mapping.items():
                    for idx in indices:
                        if 0 <= int(idx) < len(labels_by_epoch):
                            labels_by_epoch[int(idx)] = str(label)
                separation = self._band_separation_scores(
                    psd_relative, freqs, labels_by_epoch.tolist()
                )
                if separation:
                    best_name, best_eta = max(separation, key=lambda item: item[1])
                    details.append(
                        f"best univariate condition separation: {best_name} eta-squared={best_eta:.3f}"
                    )

            ax.set_xlabel("Frequency (Hz)")
            ax.set_xlim(fmin, fmax)
            ax.grid(True, alpha=0.22)
            fig.tight_layout()
            self._set_plot("frequency", self.frequency_plot_layout, self.lbl_frequency_placeholder, fig)
            self.lbl_frequency_info.setText(" | ".join(details))
        except Exception as exc:
            self._reset_plot_placeholder(
                "frequency", self.lbl_frequency_placeholder, f"Could not render frequency PSD: {exc}"
            )
            self.lbl_frequency_info.setText(f"Frequency-domain error: {exc}")

    def _plot_frequency_tfr(self):
        try:
            if self._epochs is None or len(self._epochs) == 0:
                raise RuntimeError("Time-frequency analysis requires epoched data.")
            fmin, fmax, channels, sfreq = self._frequency_settings()
            if len(channels) > 24:
                channels = channels[:24]
            mapping = self._frequency_indices_by_stimulus()
            selected = str(self.cmb_frequency_stimulus.currentText() or "All stimuli")
            if selected == "All stimuli":
                indices = sorted({int(i) for values in mapping.values() for i in values})
                if not indices:
                    indices = list(range(len(self._epochs)))
            else:
                indices = list(mapping.get(selected, []))
            if not indices:
                raise RuntimeError("No epochs matched the selected stimulus.")

            epochs = self._epochs[indices].copy().pick(channels)
            duration = float(epochs.times[-1] - epochs.times[0])
            effective_fmin = max(fmin, 2.0 / max(duration, 0.05))
            effective_fmax = min(fmax, sfreq * 0.45)
            if effective_fmax <= effective_fmin:
                raise RuntimeError("The epoch duration and sampling rate do not support this frequency range.")
            freqs = np.geomspace(effective_fmin, effective_fmax, num=24)
            wavelet_window = min(0.25, max(0.08, duration / 4.0))
            n_cycles = np.maximum(1.0, freqs * wavelet_window)
            decim = max(1, int(round(sfreq / 200.0)))
            power = epochs.compute_tfr(
                method="morlet", freqs=freqs, n_cycles=n_cycles, use_fft=True,
                output="power", average=True, return_itc=False, decim=decim,
                verbose="error"
            )
            baseline = None
            if float(power.times[0]) < 0.0 and np.sum(np.asarray(power.times) <= 0.0) >= 2:
                baseline = (float(power.times[0]), 0.0)
                power.apply_baseline(baseline=baseline, mode="logratio", verbose="error")
                image_data = 10.0 * np.nanmean(np.asarray(power.data, dtype=float), axis=0)
                color_label = "Power change (dB vs baseline)"
            else:
                power_data = np.asarray(power.data, dtype=float)
                channel_reference = np.nanmedian(power_data, axis=-1, keepdims=True)
                relative = power_data / np.maximum(channel_reference, np.finfo(float).tiny)
                image_data = 10.0 * np.nanmean(
                    np.log10(np.maximum(relative, np.finfo(float).tiny)), axis=0
                )
                color_label = "Power (dB vs within-epoch median)"

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9.5, 5.2))
            mesh = ax.pcolormesh(power.times, power.freqs, image_data, shading="auto", cmap="RdBu_r")
            if power.times[0] <= 0.0 <= power.times[-1]:
                ax.axvline(0.0, color="k", linewidth=1.0, alpha=0.65)
            ax.set_yscale("log")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            label = selected if selected != "All stimuli" else "all stimuli"
            ax.set_title(f"Morlet time-frequency power: {label} (n={len(indices)})")
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label(color_label)
            fig.tight_layout()
            self._set_plot("frequency", self.frequency_plot_layout, self.lbl_frequency_placeholder, fig)

            post_mask = np.asarray(power.times) >= 0.0
            summary = (
                f"MNE Morlet TFR: {len(indices)} trials, {len(channels)} channels, "
                f"{effective_fmin:.1f}-{effective_fmax:.1f} Hz"
            )
            if baseline is not None and np.any(post_mask):
                post_data = image_data[:, post_mask]
                peak = np.unravel_index(int(np.nanargmax(np.abs(post_data))), post_data.shape)
                peak_time = np.asarray(power.times)[post_mask][peak[1]]
                summary += (
                    f" | strongest post-onset modulation={post_data[peak]:+.2f} dB "
                    f"at {power.freqs[peak[0]]:.1f} Hz, {peak_time:.3f} s"
                )
            elif baseline is None:
                summary += " | no pre-stimulus baseline was available; interpret absolute power cautiously"
            self.lbl_frequency_info.setText(summary)
        except Exception as exc:
            self._reset_plot_placeholder(
                "frequency", self.lbl_frequency_placeholder, f"Could not render time-frequency power: {exc}"
            )
            self.lbl_frequency_info.setText(f"Frequency-domain error: {exc}")

    def _build_topomap_evokeds(self):
        if (self._epochs is None or len(self._epochs) == 0) and not self._topomap_epochs_by_path:
            return [], "No epochs available for topomap.", "unfiltered"
        refs = self._resolve_selected_topomap_indices()
        if not refs:
            return [], "No trials matched the selected stimulus.", "unfiltered"

        refs_by_path = {}
        for path, idx in refs:
            refs_by_path.setdefault(path, []).append(int(idx))

        _, _, filter_label = self._topomap_bandpass_settings()
        by_type = {"mag": [], "grad": []}
        for path in self._topomap_run_order:
            if path not in refs_by_path:
                continue
            epochs = self._topomap_epochs_by_path.get(path)
            if epochs is None:
                continue
            idx = sorted(set(refs_by_path[path]))
            if not idx:
                continue
            epochs_sel, err, filter_label = self._filtered_topomap_epochs(epochs, idx)
            if epochs_sel is None:
                return [], err or "Could not prepare topomap epochs.", filter_label
            evk = epochs_sel.average()
            try:
                available_types = list(evk.get_channel_types(unique=True))
            except Exception:
                available_types = []
            for ch_type in ["mag", "grad"]:
                if ch_type not in available_types:
                    continue
                try:
                    one = evk.copy().pick(ch_type)
                except Exception:
                    continue
                if len(getattr(one, "ch_names", [])) > 0:
                    by_type.setdefault(ch_type, []).append(one)

        evokeds = []
        for ch_type in ["mag", "grad"]:
            combined, combine_err = self._combine_topomap_evokeds(by_type.get(ch_type, []))
            if combine_err and by_type.get(ch_type):
                return [], combine_err, filter_label
            if combined is not None and len(getattr(combined, "ch_names", [])) > 0:
                evokeds.append((ch_type, combined))
        if not evokeds:
            return [], "No magnetometer or gradiometer channels available for topomap.", filter_label
        return evokeds, None, filter_label

    def _plot_topomap_only(self):
        if self._topomap_view_mode() == "slider":
            self._plot_topomap_joint()
            return
        evoked_items, err, filter_label = self._build_topomap_evokeds()
        if not evoked_items:
            self._reset_plot_placeholder("topomap", self.lbl_topomap_placeholder, err or "Could not prepare topomap evoked.")
            return
        times = self._current_topomap_times()
        try:
            figs = []
            info_parts = []
            for ch_type, evk in evoked_items:
                fig = evk.plot_topomap(
                    times=times,
                    ch_type=ch_type,
                    colorbar=True,
                    show=False,
                )
                figs.append(fig)
                info_parts.append(f"{ch_type}: channels={len(evk.ch_names)}")
            self._set_plot("topomap", self.topomap_plot_layout, self.lbl_topomap_placeholder, figs)
            self.lbl_topomap_info.setText(
                f"Topomap ({self._topomap_view_mode()}): {' | '.join(info_parts)}, "
                f"nave={int(getattr(evoked_items[0][1], 'nave', 0))}, times={times}, "
                f"run={self._topomap_run_scope_label()}, {filter_label}"
            )
        except Exception as exc:
            self._reset_plot_placeholder("topomap", self.lbl_topomap_placeholder, f"Could not render topomap: {exc}")

    def _plot_topomap_joint(self):
        evoked_items, err, filter_label = self._build_topomap_evokeds()
        if not evoked_items:
            self._reset_plot_placeholder("topomap", self.lbl_topomap_placeholder, err or "Could not prepare joint plot evoked.")
            return
        times = self._current_topomap_times()
        try:
            figs = []
            info_parts = []
            for ch_type, evk in evoked_items:
                fig = evk.plot_joint(
                    times=times,
                    picks="all",
                    show=False,
                )
                figs.append(fig)
                info_parts.append(f"{ch_type}: channels={len(evk.ch_names)}")
            self._set_plot("topomap", self.topomap_plot_layout, self.lbl_topomap_placeholder, figs)
            self.lbl_topomap_info.setText(
                f"Joint plot ({self._topomap_view_mode()}): {' | '.join(info_parts)}, "
                f"nave={int(getattr(evoked_items[0][1], 'nave', 0))}, times={times}, "
                f"run={self._topomap_run_scope_label()}, {filter_label}"
            )
        except Exception as exc:
            self._reset_plot_placeholder("topomap", self.lbl_topomap_placeholder, f"Could not render joint plot: {exc}")

    def _source_times(self):
        stc = getattr(self, "_source_stc", None)
        if stc is None:
            return np.asarray([], dtype=float)
        try:
            times = np.asarray(stc.times, dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
        if times.ndim != 1:
            return np.asarray([], dtype=float)
        if not np.any(np.isfinite(times)):
            return np.asarray([], dtype=float)
        return times

    def _source_time_index_for_value(self, time_sec=None):
        times = self._source_times()
        if times.size == 0:
            return None
        if time_sec is None:
            try:
                time_sec = float(self.spb_source_initial_time.value())
            except Exception:
                time_sec = float(times[0])
        try:
            idx = int(np.argmin(np.abs(times - float(time_sec))))
        except Exception:
            idx = 0
        return max(0, min(idx, int(times.size) - 1))

    def _source_step_ms_value(self):
        spin = getattr(self, "spb_source_step_ms", None)
        if spin is None:
            return 5.0
        try:
            value = float(spin.value())
        except Exception:
            value = 5.0
        if not np.isfinite(value) or value <= 0:
            return 5.0
        return float(value)

    def _source_time_index_for_step(self, direction):
        times = self._source_times()
        if times.size == 0:
            return None
        idx = self._source_time_index
        if idx is None:
            idx = self._source_time_index_for_value()
        if idx is None:
            return None
        idx = max(0, min(int(idx), int(times.size) - 1))
        if direction < 0 and idx <= 0:
            return idx
        if direction > 0 and idx >= int(times.size) - 1:
            return idx

        step_s = self._source_step_ms_value() / 1000.0
        target_time = float(times[idx]) + (float(direction) * step_s)
        try:
            next_idx = int(np.argmin(np.abs(times - target_time)))
        except Exception:
            next_idx = idx + (1 if direction > 0 else -1)
        next_idx = max(0, min(next_idx, int(times.size) - 1))

        if direction > 0 and next_idx <= idx:
            next_idx = min(idx + 1, int(times.size) - 1)
        elif direction < 0 and next_idx >= idx:
            next_idx = max(idx - 1, 0)
        return next_idx

    def _source_movie_tmin_tmax(self):
        tmin_spin = getattr(self, "spb_source_movie_tmin", None)
        tmax_spin = getattr(self, "spb_source_movie_tmax", None)
        try:
            tmin = float(tmin_spin.value()) if tmin_spin is not None else 0.05
            tmax = float(tmax_spin.value()) if tmax_spin is not None else 0.16
        except Exception:
            return None, None, "Invalid movie time range."
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            return None, None, "Movie tmax must be greater than movie tmin."
        times = self._source_times()
        if times.size > 0:
            lo = float(times[0])
            hi = float(times[-1])
            if tmax < lo or tmin > hi:
                return None, None, f"Movie range {tmin:.3f}-{tmax:.3f}s is outside the STC time range {lo:.3f}-{hi:.3f}s."
            tmin = max(tmin, lo)
            tmax = min(tmax, hi)
        return float(tmin), float(tmax), None

    def _set_source_movie_full_range(self):
        times = self._source_times()
        if times.size == 0:
            self.lbl_source_status.setText("Source status: no source estimate available for movie range.")
            return
        tmin_spin = getattr(self, "spb_source_movie_tmin", None)
        tmax_spin = getattr(self, "spb_source_movie_tmax", None)
        if tmin_spin is not None:
            tmin_spin.setValue(float(times[0]))
        if tmax_spin is not None:
            tmax_spin.setValue(float(times[-1]))

    def _default_source_movie_path(self, tmin, tmax):
        start_dir = os.path.dirname(str(self._fif_filename)) if self._fif_filename else os.getcwd()
        try:
            stimulus = str(self.cmb_source_evoked.currentText() or "source")
        except Exception:
            stimulus = "source"
        try:
            run_label = str(self._source_run_scope_label() or "runs")
        except Exception:
            run_label = "runs"
        stem = f"source_movie_{stimulus}_{run_label}_{float(tmin):.3f}-{float(tmax):.3f}s"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "source_movie"
        return os.path.join(start_dir, f"{stem}.mp4")

    def _save_source_movie(self):
        stc = getattr(self, "_source_stc", None)
        if stc is None:
            self.lbl_source_status.setText("Source status: compute a source estimate before saving a movie.")
            return
        tmin, tmax, time_err = self._source_movie_tmin_tmax()
        if time_err:
            self.lbl_source_status.setText(f"Source status: {time_err}")
            return

        default_path = self._default_source_movie_path(tmin, tmax)
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save source movie",
            default_path,
            "MP4 movie (*.mp4);;GIF movie (*.gif);;MOV movie (*.mov);;All files (*.*)",
        )
        if not path:
            return
        root, ext = os.path.splitext(str(path))
        if not ext:
            if "GIF" in str(selected_filter):
                path = f"{path}.gif"
            elif "MOV" in str(selected_filter):
                path = f"{path}.mov"
            else:
                path = f"{path}.mp4"

        brain = None
        btn = getattr(self, "btn_source_save_movie", None)
        try:
            if btn is not None:
                btn.setEnabled(False)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.lbl_source_status.setText(
                f"Source status: saving source movie {float(tmin):.3f}-{float(tmax):.3f}s..."
            )
            QtWidgets.QApplication.processEvents()

            plot_stc, subject_for_plot, subjects_dir_for_plot = self._stc_for_source_brain_view_plot(stc)
            if plot_stc is None or not subject_for_plot or not subjects_dir_for_plot:
                raise RuntimeError("Source estimate, subject, or subjects_dir is unavailable for movie export.")
            source_views = self._parse_source_views()
            if not source_views:
                source_views = ["lat", "med"]

            brain = plot_stc.plot(
                initial_time=float(tmin),
                hemi="split",
                views=list(source_views),
                subjects_dir=subjects_dir_for_plot,
                subject=subject_for_plot,
                time_viewer=False,
                show_traces=False,
                brain_kwargs={"show": False},
            )
            brain.save_movie(
                filename=str(path),
                time_dilation=20,
                tmin=float(tmin),
                tmax=float(tmax),
                interpolation="linear",
                framerate=10,
            )
            self._source_status_text = (
                f"saved source movie: {os.path.basename(str(path))} "
                f"({float(tmin):.3f}-{float(tmax):.3f}s)"
            )
            self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
        except Exception as exc:
            self._source_status_text = f"source movie save failed: {exc}"
            self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
        finally:
            if brain is not None:
                try:
                    brain.close()
                except Exception:
                    pass
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass
            self._update_source_time_controls()

    def _update_source_time_controls(self, time_index=None):
        times = self._source_times()
        has_times = bool(times.size > 0)
        if time_index is None:
            time_index = self._source_time_index
        if time_index is None and has_times:
            time_index = self._source_time_index_for_value()
        if time_index is not None:
            time_index = max(0, min(int(time_index), int(times.size) - 1)) if has_times else None

        btn_back = getattr(self, "btn_source_time_back", None)
        btn_forward = getattr(self, "btn_source_time_forward", None)
        if btn_back is not None:
            btn_back.setEnabled(has_times and time_index is not None and time_index > 0)
        if btn_forward is not None:
            btn_forward.setEnabled(has_times and time_index is not None and time_index < int(times.size) - 1)

        label = getattr(self, "lbl_source_time_sample", None)
        if label is not None:
            if has_times and time_index is not None:
                label.setText(f"sample {int(time_index) + 1}/{int(times.size)}  time={float(times[time_index]):.3f} s")
            else:
                label.setText("sample --/--")

        spin = getattr(self, "spb_source_initial_time", None)
        if spin is not None and has_times:
            lo = min(-1.0, float(times[0]))
            hi = max(5.0, float(times[-1]))
            spin.setRange(lo, hi)
        for movie_spin in (
            getattr(self, "spb_source_movie_tmin", None),
            getattr(self, "spb_source_movie_tmax", None),
        ):
            if movie_spin is not None and has_times:
                movie_spin.setRange(float(times[0]), float(times[-1]))
        movie_full = getattr(self, "btn_source_movie_full_range", None)
        if movie_full is not None:
            movie_full.setEnabled(has_times)
        movie_save = getattr(self, "btn_source_save_movie", None)
        if movie_save is not None:
            movie_save.setEnabled(has_times and getattr(self, "_source_stc", None) is not None)

    def _update_source_time_marker_only(self):
        line = getattr(self, "_source_time_marker_line", None)
        canvas = getattr(self, "_source_time_marker_canvas", None)
        times = self._source_times()
        idx = getattr(self, "_source_time_index", None)
        if line is None or canvas is None or times.size == 0 or idx is None:
            return False
        try:
            time_sec = float(times[int(idx)])
            line.set_xdata([time_sec, time_sec])
            canvas.draw_idle()
            return True
        except Exception:
            return False

    def _set_source_time_index(self, time_index, update_3d=True, render=True):
        times = self._source_times()
        if times.size == 0:
            self._source_time_index = None
            self._update_source_time_controls()
            return
        idx = max(0, min(int(time_index), int(times.size) - 1))
        self._source_time_index = idx
        time_sec = float(times[idx])
        spin = getattr(self, "spb_source_initial_time", None)
        if spin is not None:
            spin.blockSignals(True)
            spin.setValue(time_sec)
            spin.blockSignals(False)
        self._update_source_time_controls(idx)
        if update_3d and callable(getattr(self, "_source_display_callback", None)):
            try:
                self._source_display_callback(self._source_stc, time_sec)
            except Exception as exc:
                self._source_status_text = f"3D source display failed: {exc}"
                self.lbl_source_status.setText(f"Source status: {self._source_status_text}")
        if render == "marker":
            screenshot_enabled = bool(
                getattr(getattr(self, "chk_source_brain_screenshot", None), "isChecked", lambda: False)()
            )
            if screenshot_enabled:
                self._render_source_time_course(render_brain_screenshots=True)
            elif getattr(self, "_source_plot_has_brain_image", False):
                self._render_source_time_course(render_brain_screenshots=False)
            elif not self._update_source_time_marker_only():
                self._render_source_time_course(render_brain_screenshots=False)
        elif render:
            self._render_source_time_course()

    def _step_source_time_backward(self):
        idx = self._source_time_index_for_step(-1)
        if idx is None:
            return
        self._set_source_time_index(int(idx), update_3d=True, render="marker")

    def _step_source_time_forward(self):
        idx = self._source_time_index_for_step(1)
        if idx is None:
            return
        self._set_source_time_index(int(idx), update_3d=True, render="marker")

    def _on_source_time_spinbox_committed(self):
        idx = self._source_time_index_for_value()
        if idx is None:
            self._update_source_time_controls()
            return
        self._set_source_time_index(idx, update_3d=True, render="marker")

    def _compute_parcel_trial_heatmaps(self):
        if not callable(self._parcel_heatmap_callback):
            self.lbl_parcel_heatmap_status.setText(
                "Parcel heatmaps are unavailable because the source extraction callback is not configured."
            )
            return
        if self._parcel_heatmap_thread is not None:
            self.lbl_parcel_heatmap_status.setText("Parcel extraction is already running.")
            return

        stimulus = self.cmb_parcel_heatmap_stimulus.currentData()
        stimulus = str(stimulus or "").strip()
        comparison_stimulus = str(
            self.cmb_parcel_heatmap_comparison.currentData() or ""
        ).strip()
        run_paths = self._selected_parcel_heatmap_run_paths()
        tmin = float(self.spb_parcel_heatmap_tmin_ms.value()) / 1000.0
        tmax = float(self.spb_parcel_heatmap_tmax_ms.value()) / 1000.0
        if not stimulus:
            self.lbl_parcel_heatmap_status.setText("Select a repeated stimulus before computing.")
            return
        if not run_paths:
            self.lbl_parcel_heatmap_status.setText("No source run is selected.")
            return
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            self.lbl_parcel_heatmap_status.setText("Window end must be greater than window start.")
            return

        source_space_kind = str(self.cmb_parcel_heatmap_source_space.currentData() or "surface")
        parcellation = (
            "aparc+aseg"
            if source_space_kind == "volume"
            else str(self.cmb_parcel_heatmap_parcellation.currentData() or "aparc")
        )
        snr = max(1e-6, float(self.spb_parcel_heatmap_snr.value()))
        max_presentations = int(self.spb_parcel_heatmap_max_presentations.value())
        candidate_pool = max_presentations + max(
            2, int(np.ceil(max_presentations * 0.25))
        )
        options = {
            "selected_concept_label": stimulus,
            "comparison_concept_label": comparison_stimulus,
            "group_mode": self._parcel_heatmap_group_mode(),
            "source_fif_paths": list(run_paths),
            "window_start": tmin,
            "window_end": tmax,
            "normalize_parcel_baseline": bool(self.chk_parcel_heatmap_baseline.isChecked()),
            "method": str(self.cmb_parcel_heatmap_method.currentText() or "MNE"),
            "lambda2": 1.0 / (snr ** 2),
            "source_space_kind": source_space_kind,
            "source_parcellation": parcellation,
            "max_presentations": max_presentations,
            # One inverse call for the usual 12-presentation view, while
            # retaining a conservative cap for larger requests.
            "source_batch_size": max(8, min(16, candidate_pool)),
        }
        options.update(self._source_parcel_preprocessing_options())
        band_key = str(
            self.cmb_parcel_heatmap_frequency_band.currentData() or "broadband"
        )
        representation = self._parcel_heatmap_representation_key()
        try:
            band_low, band_high, band_label = (
                self._validate_parcel_heatmap_representation()
            )
        except ValueError as exc:
            self.lbl_parcel_heatmap_status.setText(
                f"Parcel heatmaps: selection unavailable: {exc}"
            )
            return
        options.update(
            {
                "parcel_frequency_band": band_key,
                "parcel_frequency_band_label": band_label,
                "parcel_representation": representation,
            }
        )

        self._parcel_heatmap_result = None
        self._reset_parcel_heatmap_dashboard(
            f"Computing {stimulus}; presentation patterns will appear here."
        )
        self.btn_parcel_heatmap_compute.setEnabled(False)
        self.lbl_parcel_heatmap_status.setText(
            f"Computing {stimulus}: up to {options['max_presentations']} presentations, "
            f"{tmin * 1000.0:g}–{tmax * 1000.0:g} ms, {band_label}, "
            f"{'Hilbert power (dB vs baseline)' if representation == 'band_power_db' else 'source activity'}"
            + (f"; contrast versus {comparison_stimulus}..." if comparison_stimulus else "...")
        )
        thread = QtCore.QThread(self.widget)
        worker = _ParcelHeatmapWorker(self._parcel_heatmap_callback, options)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_parcel_heatmap_progress)
        worker.finished.connect(self._on_parcel_heatmap_finished)
        worker.failed.connect(self._on_parcel_heatmap_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_parcel_heatmap_thread_finished)
        self._parcel_heatmap_thread = thread
        self._parcel_heatmap_worker = worker
        thread.start()

    def _on_parcel_heatmap_progress(self, message):
        self.lbl_parcel_heatmap_status.setText(f"Parcel heatmaps: {message}")

    def _on_parcel_heatmap_finished(self, result, message):
        if not isinstance(result, dict):
            self._parcel_heatmap_result = None
            self.lbl_parcel_heatmap_status.setText(
                f"Parcel heatmaps: {message or 'no usable presentations were extracted.'}"
            )
            self._reset_parcel_heatmap_dashboard(
                "No parcel heatmaps could be computed for this selection.",
            )
            return
        self._parcel_heatmap_result = result
        self.lbl_parcel_heatmap_status.setText(f"Parcel heatmaps: {message}")
        self._render_parcel_trial_heatmaps()

    def _on_parcel_heatmap_failed(self, message):
        self._parcel_heatmap_result = None
        self.lbl_parcel_heatmap_status.setText(f"Parcel heatmaps failed: {message}")
        self._reset_parcel_heatmap_dashboard(
            "Parcel heatmap extraction failed. See the status message above.",
        )

    def _on_parcel_heatmap_thread_finished(self):
        self._parcel_heatmap_thread = None
        self._parcel_heatmap_worker = None
        self.btn_parcel_heatmap_compute.setEnabled(True)

    def _clear_parcel_ranking_table(self):
        self.parcel_ranking_model = None
        table = getattr(self, "tbl_parcel_ranking", None)
        if table is not None:
            table.setModel(None)
        group = getattr(self, "grp_parcel_ranking", None)
        if group is not None:
            group.setVisible(False)

    def _set_parcel_ranking_table(
        self,
        rows,
        primary_label,
        comparison_label,
        primary_count,
        comparison_count,
        frequency_band_label="",
        representation="activity",
    ):
        if not rows:
            self._clear_parcel_ranking_table()
            return
        is_power = str(representation or "activity") == "band_power_db"
        table_rows = []
        for row in rows:
            start_ms = float(row.get("peak_start_ms", np.nan))
            end_ms = float(row.get("peak_end_ms", np.nan))
            interval = (
                f"{start_ms:g}–{end_ms:g}"
                if np.isfinite(start_ms) and np.isfinite(end_ms)
                else "—"
            )
            primary_agreement = (
                f"{int(row.get('primary_sign_count', 0))}/"
                f"{int(row.get('primary_sign_total', 0))}"
            )
            comparison_agreement = (
                f"{int(row.get('comparison_sign_count', 0))}/"
                f"{int(row.get('comparison_sign_total', 0))}"
            )
            table_rows.append(
                {
                    "Rank": int(row.get("rank", 0)),
                    "Parcel": str(row.get("parcel", "")),
                    "Peak interval (ms)": interval,
                    "Direction": str(row.get("direction", "")),
                    "Mean contrast (dB)" if is_power else "Mean contrast": round(
                        float(row.get("mean_contrast", np.nan)), 3
                    ),
                    f"{primary_label} ERD/ERS"
                    if is_power
                    else f"{primary_label} sign": primary_agreement,
                    f"{comparison_label} ERD/ERS"
                    if is_power
                    else f"{comparison_label} sign": comparison_agreement,
                    "Nested LOO balanced accuracy": round(
                        float(row.get("nested_loo_balanced_accuracy", np.nan)), 3
                    ),
                }
            )
        ranking_df = pd.DataFrame(table_rows)
        self.parcel_ranking_model = PandasModel(ranking_df)
        self.tbl_parcel_ranking.setModel(self.parcel_ranking_model)
        header = self.tbl_parcel_ranking.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        parcel_column = ranking_df.columns.get_loc("Parcel")
        header.setSectionResizeMode(parcel_column, QtWidgets.QHeaderView.Stretch)
        self.lbl_parcel_ranking_note.setText(
            (f"Frequency band: {frequency_band_label}. " if frequency_band_label else "")
            + f"{primary_label} (n={primary_count}) versus {comparison_label} "
            + f"(n={comparison_count}). Peak interval and contrast use all presentations; "
            + (
                "positive contrast means greater baseline-relative band power for the first group; "
                if is_power
                else ""
            )
            + "nested LOO balanced accuracy selects the strongest time bin inside each "
            "training fold. Chance = 0.5; no permutation p-value is included."
        )
        self.grp_parcel_ranking.setVisible(True)

    @staticmethod
    def _parcel_heatmap_trial_title(row, presentation_index):
        if isinstance(row, pd.Series):
            trial = row.get("trial_index", row.get("trial", presentation_index + 1))
            try:
                trial = int(trial)
            except Exception:
                trial = presentation_index + 1
            run_path = str(row.get("source_fif_path", "") or "")
            run_label = os.path.basename(run_path)
            if len(run_label) > 28:
                run_label = f"{run_label[:25]}..."
            return f"Presentation {presentation_index + 1} · trial {trial}" + (
                f"\n{run_label}" if run_label else ""
            )
        return f"Presentation {presentation_index + 1}"

    @staticmethod
    def _add_parcel_group_separators(ax, parcel_groups):
        """Draw broad-region and hemisphere boundaries on a parcel heatmap."""
        previous_hemisphere = None
        for group in parcel_groups or []:
            start = int(group.get("start", 0))
            hemisphere = str(group.get("hemisphere", "") or "")
            if start > 0:
                hemisphere_boundary = hemisphere != previous_hemisphere
                ax.axhline(
                    start - 0.5,
                    color="#f8fafc",
                    linewidth=1.35 if hemisphere_boundary else 0.55,
                    alpha=0.95 if hemisphere_boundary else 0.72,
                )
            previous_hemisphere = hemisphere

    def _render_parcel_trial_heatmaps(self, *_args):
        result = self._parcel_heatmap_result
        if not isinstance(result, dict):
            return
        try:
            from matplotlib.figure import Figure

            data = np.asarray(result.get("data", []), dtype=float)
            times = np.asarray(result.get("times", []), dtype=float)
            roi_names = [str(v) for v in result.get("roi_names", [])]
            original_roi_names = list(roi_names)
            trial_df = result.get("trial_df", pd.DataFrame())
            if data.ndim != 3 or data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
                raise RuntimeError("No trial-by-parcel data were returned.")
            if times.size != data.shape[2] or len(roi_names) != data.shape[1]:
                raise RuntimeError("Parcel data dimensions do not match the labels or time axis.")

            parcellation = str(
                result.get("source_parcellation", "parcels") or "parcels"
            )
            parcel_groups = []
            row_order_mode = str(
                getattr(self, "cmb_parcel_heatmap_row_order", None).currentData()
                if getattr(self, "cmb_parcel_heatmap_row_order", None) is not None
                else "anatomical"
            )
            if row_order_mode == "anatomical":
                row_order, parcel_groups = anatomical_parcel_row_order(
                    roi_names, parcellation
                )
                if row_order.size != len(roi_names):
                    raise RuntimeError("Parcel anatomical ordering is incomplete.")
                data = data[:, row_order, :]
                roi_names = [roi_names[int(index)] for index in row_order]

            normalization = str(result.get("normalization", "none") or "none")
            is_zscore = normalization == "parcel_baseline_zscore"
            representation = str(
                result.get("parcel_representation", "activity") or "activity"
            )
            is_power = representation == "band_power_db" or normalization == "parcel_power_db"
            show_absolute = (
                not is_power and bool(self.chk_parcel_heatmap_absolute.isChecked())
            )
            requested_bin_ms = float(self.spb_parcel_heatmap_display_bin_ms.value())
            signed_plot_data, display_times, display_extent_ms, effective_bin_ms = (
                temporal_bin_parcel_time_courses(data, times, requested_bin_ms)
            )
            if show_absolute:
                plot_data, _, _, _ = temporal_bin_parcel_time_courses(
                    np.abs(data), times, requested_bin_ms
                )
            else:
                plot_data = signed_plot_data
            signed_mean, sign_consistency = parcel_trial_summary_maps(signed_plot_data)
            comparison_result = result.get("comparison_result")
            contrast_data = None
            comparison_stimulus = ""
            comparison_count = 0
            if isinstance(comparison_result, dict):
                comparison_data = np.asarray(
                    comparison_result.get("data", []), dtype=float
                )
                comparison_times = np.asarray(
                    comparison_result.get("times", []), dtype=float
                )
                comparison_rois = [
                    str(v) for v in comparison_result.get("roi_names", [])
                ]
                comparison_normalization = str(
                    comparison_result.get("normalization", "none") or "none"
                )
                comparison_representation = str(
                    comparison_result.get("parcel_representation", "activity")
                    or "activity"
                )
                if comparison_data.ndim != 3 or comparison_data.shape[0] == 0:
                    raise RuntimeError("Comparison stimulus returned no parcel data.")
                if comparison_rois != original_roi_names:
                    raise RuntimeError(
                        "Contrast groups have different parcel identities or ordering."
                    )
                if (
                    comparison_times.shape != times.shape
                    or not np.allclose(comparison_times, times, rtol=0.0, atol=1e-9)
                ):
                    raise RuntimeError("Contrast groups have different source time grids.")
                if comparison_normalization != normalization:
                    raise RuntimeError("Contrast groups use different normalization settings.")
                if comparison_representation != representation:
                    raise RuntimeError("Contrast groups use different parcel representations.")
                if row_order_mode == "anatomical":
                    comparison_data = comparison_data[:, row_order, :]
                comparison_plot_data, comparison_display_times, _, comparison_bin_ms = (
                    temporal_bin_parcel_time_courses(
                        comparison_data, comparison_times, requested_bin_ms
                    )
                )
                if (
                    comparison_display_times.shape != display_times.shape
                    or not np.allclose(
                        comparison_display_times, display_times, rtol=0.0, atol=1e-9
                    )
                    or not np.isclose(comparison_bin_ms, effective_bin_ms)
                ):
                    raise RuntimeError("Contrast groups produced different display bins.")
                contrast_data = parcel_trial_contrast_map(
                    signed_plot_data, comparison_plot_data
                )
                comparison_stimulus = str(
                    result.get(
                        "comparison_stimulus",
                        comparison_result.get("stimulus", "comparison"),
                    )
                    or "comparison"
                )
                comparison_count = int(comparison_data.shape[0])
            finite = plot_data[np.isfinite(plot_data)]
            if finite.size == 0:
                raise RuntimeError("Parcel data contain no finite values.")
            if show_absolute:
                vmin = 0.0
                vmax = float(np.nanpercentile(finite, 99.0))
                cmap = "magma"
                color_label = "Mean |baseline z|" if is_zscore else "Mean |source activity| (a.u.)"
            else:
                vmax = float(np.nanpercentile(np.abs(finite), 99.0))
                vmin = -vmax
                cmap = "RdBu_r"
                color_label = (
                    "Band power change (dB vs prestimulus)"
                    if is_power
                    else "Parcel baseline z-score"
                    if is_zscore
                    else "Signed mean source activity (a.u.)"
                )
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
                if not show_absolute:
                    vmin = -1.0

            n_presentations = int(data.shape[0])
            n_cols = min(3, n_presentations)
            n_rows = int(np.ceil(n_presentations / float(n_cols)))
            top_figure_height = max(4.8, 1.65 * n_rows + 0.8)
            fig = Figure(
                figsize=(7.2, top_figure_height),
                constrained_layout=True,
            )
            axes = fig.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey=True)
            axes_flat = list(axes.flat)
            extent = [
                float(display_extent_ms[0]),
                float(display_extent_ms[1]),
                data.shape[1] - 0.5,
                -0.5,
            ]
            if parcel_groups:
                y_ticks = np.asarray(
                    [
                        (float(group["start"]) + float(group["stop"]) - 1.0) / 2.0
                        for group in parcel_groups
                    ],
                    dtype=float,
                )
                y_tick_labels = [str(group["label"]) for group in parcel_groups]
                y_axis_label = "Hemisphere › region"
            else:
                y_tick_count = min(8, len(roi_names))
                y_ticks = np.unique(
                    np.linspace(0, len(roi_names) - 1, y_tick_count, dtype=int)
                )
                y_tick_labels = [roi_names[int(tick)] for tick in y_ticks]
                y_axis_label = "Source parcel"
            image = None
            for index, ax in enumerate(axes_flat):
                if index >= n_presentations:
                    ax.set_axis_off()
                    continue
                image = ax.imshow(
                    plot_data[index],
                    aspect="auto",
                    interpolation="nearest",
                    origin="upper",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                row = trial_df.iloc[index] if isinstance(trial_df, pd.DataFrame) and index < len(trial_df) else None
                ax.set_title(self._parcel_heatmap_trial_title(row, index), fontsize=8)
                ax.axvline(0.0, color="0.65", linewidth=0.7)
                self._add_parcel_group_separators(ax, parcel_groups)
                if index % n_cols == 0:
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_tick_labels, fontsize=6)
                    ax.set_ylabel(y_axis_label, fontsize=8)
                else:
                    ax.tick_params(axis="y", labelleft=False)
                if index // n_cols == n_rows - 1:
                    ax.set_xlabel("Time after stimulus (ms)", fontsize=8)
                ax.tick_params(axis="x", labelsize=7)

            if image is not None:
                colorbar = fig.colorbar(image, ax=axes_flat[:n_presentations], shrink=0.78, pad=0.015)
                colorbar.set_label(color_label, fontsize=8)
                colorbar.ax.tick_params(labelsize=7)

            summary_fig = Figure(
                figsize=(5.3, top_figure_height), constrained_layout=True
            )
            summary_axes = summary_fig.subplots(
                2, 1, squeeze=False, sharex=True, sharey=True
            )
            mean_ax, consistency_ax = summary_axes[:, 0]
            mean_finite = signed_mean[np.isfinite(signed_mean)]
            mean_limit = (
                float(np.nanpercentile(np.abs(mean_finite), 99.0))
                if mean_finite.size
                else 1.0
            )
            if not np.isfinite(mean_limit) or mean_limit <= 0.0:
                mean_limit = 1.0
            mean_image = mean_ax.imshow(
                signed_mean,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                extent=extent,
                cmap="RdBu_r",
                vmin=-mean_limit,
                vmax=mean_limit,
            )
            mean_title = (
                "Mean band-power change across presentations"
                if is_power
                else "Mean signed z-score across presentations"
                if is_zscore
                else "Mean signed source activity across presentations"
            )
            mean_ax.set_title(mean_title, fontsize=9)
            mean_ax.set_yticks(y_ticks)
            mean_ax.set_yticklabels(y_tick_labels, fontsize=6)
            mean_ax.set_ylabel(y_axis_label, fontsize=8)
            mean_ax.set_xlabel("Time after stimulus (ms)", fontsize=8)
            mean_ax.tick_params(axis="x", labelsize=7)
            mean_ax.axvline(0.0, color="0.65", linewidth=0.7)
            self._add_parcel_group_separators(mean_ax, parcel_groups)
            mean_colorbar = summary_fig.colorbar(
                mean_image, ax=mean_ax, shrink=0.82, pad=0.015
            )
            mean_colorbar.set_label(
                "Mean power change (dB vs prestimulus)"
                if is_power
                else "Mean parcel baseline z-score"
                if is_zscore
                else "Mean signed activity (a.u.)",
                fontsize=8,
            )
            mean_colorbar.ax.tick_params(labelsize=7)

            consistency_image = consistency_ax.imshow(
                sign_consistency,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                extent=extent,
                cmap="viridis",
                vmin=0.5,
                vmax=1.0,
            )
            consistency_ax.set_title(
                (
                    "ERD/ERS direction consistency (0.5 = split, 1 = unanimous)"
                    if is_power
                    else "Dominant-sign consistency (0.5 = split, 1 = unanimous)"
                ),
                fontsize=8,
            )
            consistency_ax.set_yticks(y_ticks)
            consistency_ax.set_yticklabels(y_tick_labels, fontsize=6)
            consistency_ax.set_ylabel(y_axis_label, fontsize=8)
            consistency_ax.set_xlabel("Time after stimulus (ms)", fontsize=8)
            consistency_ax.tick_params(axis="x", labelsize=7)
            consistency_ax.axvline(0.0, color="0.65", linewidth=0.7)
            self._add_parcel_group_separators(consistency_ax, parcel_groups)
            consistency_colorbar = summary_fig.colorbar(
                consistency_image, ax=consistency_ax, shrink=0.82, pad=0.015
            )
            consistency_colorbar.set_label(
                "Fraction sharing ERD/ERS direction"
                if is_power
                else "Fraction sharing dominant sign",
                fontsize=8,
            )
            consistency_colorbar.ax.tick_params(labelsize=7)

            stimulus = str(result.get("stimulus", "stimulus") or "stimulus")
            frequency_band_label = str(
                result.get("frequency_band_label", "") or ""
            ).strip()
            if not frequency_band_label:
                try:
                    _low, _high, frequency_band_label = parcel_trial_frequency_band_settings(
                        "broadband",
                        result.get("bandpass_low_hz", 0.0),
                        result.get("bandpass_high_hz", 0.0),
                    )
                except ValueError:
                    frequency_band_label = "Band-limited"
            normalization_label = (
                "Hilbert power · dB vs prestimulus"
                if is_power
                else "parcel baseline z-score"
                if normalization == "parcel_baseline_zscore"
                else "native source units"
            )
            fig.suptitle(
                f"{stimulus}: parcel patterns across {n_presentations} presentation(s) · "
                f"{display_extent_ms[0]:g}–{display_extent_ms[1]:g} ms · "
                f"{effective_bin_ms:g} ms display bins · {parcellation} · "
                f"{normalization_label} · {frequency_band_label}",
                fontsize=11,
            )
            summary_fig.suptitle(
                f"{stimulus}: cross-presentation summaries across {n_presentations} presentation(s) · "
                f"{effective_bin_ms:g} ms display bins · {frequency_band_label}",
                fontsize=11,
            )
            if contrast_data is not None:
                ranking_rows = rank_parcel_contrast_features(
                    signed_plot_data,
                    comparison_plot_data,
                    roi_names,
                    display_times,
                    effective_bin_ms,
                    extent_ms=display_extent_ms,
                    primary_label=stimulus,
                    comparison_label=comparison_stimulus,
                )
                contrast_fig = Figure(figsize=(7.2, 3.8), constrained_layout=True)
                contrast_ax = contrast_fig.subplots(1, 1)
                contrast_finite = contrast_data[np.isfinite(contrast_data)]
                contrast_limit = (
                    float(np.nanpercentile(np.abs(contrast_finite), 99.0))
                    if contrast_finite.size
                    else 1.0
                )
                if not np.isfinite(contrast_limit) or contrast_limit <= 0.0:
                    contrast_limit = 1.0
                contrast_image = contrast_ax.imshow(
                    contrast_data,
                    aspect="auto",
                    interpolation="nearest",
                    origin="upper",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-contrast_limit,
                    vmax=contrast_limit,
                )
                contrast_ax.set_title(
                    f"Signed mean contrast: {stimulus} − {comparison_stimulus}", fontsize=10
                )
                contrast_ax.set_yticks(y_ticks)
                contrast_ax.set_yticklabels(y_tick_labels, fontsize=7)
                contrast_ax.set_ylabel(y_axis_label, fontsize=8)
                contrast_ax.set_xlabel("Time after stimulus (ms)", fontsize=8)
                contrast_ax.tick_params(axis="x", labelsize=7)
                contrast_ax.axvline(0.0, color="0.65", linewidth=0.7)
                self._add_parcel_group_separators(contrast_ax, parcel_groups)
                contrast_colorbar = contrast_fig.colorbar(
                    contrast_image, ax=contrast_ax, shrink=0.82, pad=0.015
                )
                contrast_colorbar.set_label(
                    "Difference in mean band-power change (dB)"
                    if is_power
                    else "Difference in mean parcel baseline z-score"
                    if is_zscore
                    else "Difference in signed mean activity (a.u.)",
                    fontsize=8,
                )
                contrast_colorbar.ax.tick_params(labelsize=7)
                contrast_fig.suptitle(
                    f"{stimulus} (n={n_presentations}) − {comparison_stimulus} "
                    f"(n={comparison_count}) · {effective_bin_ms:g} ms display bins · "
                    f"{parcellation} · {frequency_band_label}",
                    fontsize=11,
                )
                self._set_plot(
                    "parcel_heatmap_contrast",
                    self.parcel_contrast_plot_layout,
                    self.lbl_parcel_contrast_placeholder,
                    contrast_fig,
                )
                self._set_parcel_ranking_table(
                    ranking_rows,
                    stimulus,
                    comparison_stimulus,
                    n_presentations,
                    comparison_count,
                    frequency_band_label,
                    representation,
                )
                self.parcel_heatmap_bottom_splitter.setVisible(True)
                self.parcel_heatmap_bottom_splitter.setSizes([720, 480])
                self.parcel_heatmap_dashboard_splitter.setSizes([620, 300])
            else:
                self._reset_plot_placeholder(
                    "parcel_heatmap_contrast",
                    self.lbl_parcel_contrast_placeholder,
                    "Select Compare with and compute to show a contrast map.",
                )
                self._clear_parcel_ranking_table()
                self.parcel_heatmap_bottom_splitter.setVisible(False)
            self._set_plot(
                "parcel_heatmaps",
                self.parcel_heatmap_plot_layout,
                self.lbl_parcel_heatmap_placeholder,
                fig,
            )
            self._set_plot(
                "parcel_heatmap_summaries",
                self.parcel_summary_plot_layout,
                self.lbl_parcel_summary_placeholder,
                summary_fig,
            )
        except Exception as exc:
            self._reset_parcel_heatmap_dashboard(
                f"Could not render parcel heatmaps: {exc}",
            )

    def _plot_source_time_course(self):
        if callable(self._source_compute_callback):
            try:
                self._source_src = None
                method = str(self.cmb_source_method.currentText() or "MNE")
                snr = max(1e-6, float(self.spb_source_snr.value()))
                lambda2 = 1.0 / (snr ** 2)
                bandpass_enabled, bandpass_low, bandpass_high, bandpass_label = self._source_bandpass_settings()
                inv_path = str(self.le_source_inverse.text() or "").strip()
                evoked_name = str(self.cmb_source_evoked.currentText() or "").strip()
                source_indices = self._resolve_selected_source_indices()
                source_run_paths = self._selected_source_run_paths()
                source_selection_explicit = evoked_name != "All stimuli"
                epoch_indices_by_run = self._source_epoch_indices_by_run(
                    source_indices,
                    include_empty=source_selection_explicit,
                )
                source_run_path = source_run_paths[0] if len(source_run_paths) == 1 else ""
                single_epoch_indices = (
                    epoch_indices_by_run.get(source_run_path, [])
                    if source_run_path
                    else []
                )
                options = {
                    "method": method,
                    "lambda2": lambda2,
                    "snr": snr,
                    "initial_time": float(self.spb_source_initial_time.value()),
                    "inverse_operator_path": inv_path,
                    "evoked_name": evoked_name,
                    "source_fif_path": source_run_path,
                    "source_fif_paths": list(source_run_paths),
                    "source_run_label": self._source_run_scope_label(),
                    "source_space_kind": self.cmb_source_space_kind.currentData() or "surface",
                    "epoch_indices": list(single_epoch_indices),
                    "epoch_indices_by_run": epoch_indices_by_run,
                    "source_epoch_indices_explicit": bool(source_selection_explicit),
                    "group_mode": self._source_group_mode(),
                    "source_brain_view": self._source_brain_view_mode(),
                    "source_bandpass_enabled": bool(bandpass_enabled),
                    "bandpass_low_hz": bandpass_low,
                    "bandpass_high_hz": bandpass_high,
                    "source_bandpass_label": bandpass_label,
                }
                result = self._source_compute_callback(options)
                if isinstance(result, tuple) and len(result) >= 3:
                    stc, msg, meta = result[0], result[1], result[2]
                    if isinstance(meta, dict):
                        subjects_dir = meta.get("subjects_dir")
                        if subjects_dir:
                            self._source_subjects_dir = str(subjects_dir)
                            self.lbl_source_subjects_dir.setText(str(subjects_dir))
                        subject = meta.get("subject")
                        if subject:
                            self._source_subject = str(subject)
                            self.lbl_source_subject.setText(str(subject))
                        inverse_operator_path = meta.get("inverse_operator_path")
                        if inverse_operator_path:
                            self.le_source_inverse.setText(str(inverse_operator_path))
                        if meta.get("source_src") is not None:
                            self._source_src = meta.get("source_src")
                        cov_label = str(meta.get("noise_cov_selected_method") or meta.get("noise_cov_method") or "").strip()
                        if cov_label and msg and "noise_cov=" not in str(msg):
                            msg = f"{msg} | noise_cov={cov_label}"
                        if meta.get("inverse_operator_regenerated") and msg:
                            msg = f"{msg} | regenerated inverse for current transform"
                elif isinstance(result, tuple) and len(result) >= 2:
                    stc, msg = result[0], result[1]
                else:
                    stc, msg = None, "Invalid source callback result."
                if stc is not None:
                    self._source_stc = stc
                    if callable(getattr(self, "_source_brain_view_callback", None)):
                        try:
                            self._source_brain_view_callback(self._source_brain_view_mode())
                        except Exception as exc:
                            self._source_status_text = f"brain view switch failed: {exc}"
                    idx = self._source_time_index_for_value()
                    if idx is not None:
                        self._set_source_time_index(int(idx), update_3d=True, render=False)
                if msg:
                    self._source_status_text = str(msg)
            except Exception as exc:
                self._source_status_text = f"source computation failed: {exc}"
                self._source_stc = None
                self._source_src = None
        self._render_source_time_course()

    def _render_source_time_course(self, render_brain_screenshots=None):
        if self._source_stc is None:
            msg = self._source_status_text or "no source estimate available."
            self.lbl_source_status.setText(f"Source status: {msg}")
            self._source_time_index = None
            self._source_time_marker_line = None
            self._source_time_marker_canvas = None
            self._source_plot_has_brain_image = False
            self._update_source_time_controls()
            self._reset_plot_placeholder("source", self.lbl_source_placeholder, "Source estimate not available.")
            return

        stc = self._source_stc
        try:
            from matplotlib.figure import Figure
            data_abs = np.abs(np.asarray(stc.data, dtype=float))
            if data_abs.ndim != 2 or data_abs.shape[1] == 0:
                raise RuntimeError("Invalid source estimate data shape.")
            times = np.asarray(stc.times, dtype=float)
            if times.size == 0:
                raise RuntimeError("Source estimate has no time axis.")

            initial_time = float(self.spb_source_initial_time.value())
            if initial_time < float(times[0]):
                initial_time = float(times[0])
            if initial_time > float(times[-1]):
                initial_time = float(times[-1])
            time_idx = int(np.argmin(np.abs(times - float(initial_time))))
            time_idx = max(0, min(time_idx, int(times.size) - 1))
            initial_time = float(times[time_idx])
            self._source_time_index = int(time_idx)
            self.spb_source_initial_time.blockSignals(True)
            self.spb_source_initial_time.setValue(initial_time)
            self.spb_source_initial_time.blockSignals(False)
            self._update_source_time_controls(time_idx)

            n_lh = int(len(stc.vertices[0])) if isinstance(stc.vertices, list) and len(stc.vertices) > 0 else 0
            if n_lh <= 0 or n_lh >= data_abs.shape[0]:
                n_lh = data_abs.shape[0] // 2
            lh_data = data_abs[:n_lh, :]
            rh_data = data_abs[n_lh:, :]
            rms_trace = np.sqrt(np.nanmean(np.square(data_abs), axis=0))
            lh_trace = np.nanmax(lh_data, axis=0) if lh_data.size > 0 else np.zeros_like(times)
            rh_trace = np.nanmax(rh_data, axis=0) if rh_data.size > 0 else np.zeros_like(times)
            metric = str(self.cmb_source_display_metric.currentText())
            if metric == "max |stc|":
                main_trace = np.nanmax(data_abs, axis=0)
                main_label = "Global max"
            elif metric == "LH mean |stc|" and is_surface_stc(stc):
                main_trace = np.nanmean(lh_data, axis=0) if lh_data.size > 0 else np.nanmean(data_abs, axis=0)
                main_label = "LH mean"
            elif metric == "RH mean |stc|" and is_surface_stc(stc):
                main_trace = np.nanmean(rh_data, axis=0) if rh_data.size > 0 else np.nanmean(data_abs, axis=0)
                main_label = "RH mean"
            else:
                main_trace = np.nanmean(data_abs, axis=0)
                main_label = "Mean |stc|"

            top_roi_result = None
            top_roi_error = ""
            top_n = self._source_top_roi_count()
            if top_n > 0:
                try:
                    top_roi_result = self._source_top_label_time_courses(stc, top_n)
                except Exception as exc:
                    top_roi_error = str(exc)

            plot_stc, subject_for_plot, subjects_dir_for_plot = self._stc_for_source_brain_view_plot(stc)
            if render_brain_screenshots is False:
                source_views = self._parse_source_views()
                primary_views, secondary_views = self._split_source_views(source_views)
                brain_image, brain_image_side = None, None
            else:
                brain_image, brain_image_side, primary_views, secondary_views = self._render_stc_brain_screenshots(
                    plot_stc,
                    initial_time,
                    subject_for_plot,
                    subjects_dir_for_plot,
                )

            fig = Figure(figsize=(10.5, 8.0))
            marker_line = None
            if brain_image is not None:
                if brain_image_side is not None:
                    grid = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.0], width_ratios=[3.4, 1.4], hspace=0.14, wspace=0.06)
                    ax_img = fig.add_subplot(grid[0, 0])
                    ax_img_side = fig.add_subplot(grid[0, 1])
                    ax_img_side.imshow(brain_image_side)
                    ax_img_side.set_axis_off()
                    if secondary_views:
                        ax_img_side.set_title(
                            f"Supplemental views: {secondary_views}",
                            fontsize=10,
                        )
                    ax_tc = fig.add_subplot(grid[1, :])
                else:
                    grid = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.14)
                    ax_img = fig.add_subplot(grid[0])
                    ax_tc = fig.add_subplot(grid[1])
                ax_img.imshow(brain_image)
                ax_img.set_axis_off()
                ax_img.set_title(
                    f"Source estimate ({self._source_run_scope_label()}, hemi='split', views={primary_views})  time={initial_time:.3f}s",
                    fontsize=11,
                )
            else:
                ax_tc = fig.add_subplot(1, 1, 1)

            ax_tc.plot(times, main_trace, linewidth=2.0, label=main_label)
            ax_tc.plot(times, rms_trace, linewidth=1.6, label="RMS")
            if is_surface_stc(stc):
                ax_tc.plot(times, lh_trace, linewidth=1.3, alpha=0.9, label="LH max")
                ax_tc.plot(times, rh_trace, linewidth=1.3, alpha=0.9, label="RH max")
            if top_roi_result is not None:
                roi_times, roi_tc, roi_names, roi_idxs, roi_atlas = top_roi_result
                for idx in roi_idxs:
                    label = str(roi_names[idx])
                    if len(label) > 42:
                        label = label[:39] + "..."
                    ax_tc.plot(
                        roi_times,
                        roi_tc[idx],
                        linewidth=1.15,
                        linestyle="--",
                        alpha=0.85,
                        label=label,
                    )
                if roi_idxs:
                    ax_tc.set_title(f"Top {len(roi_idxs)} ROI traces: {roi_atlas}", fontsize=10)
            marker_line = ax_tc.axvline(initial_time, color="k", linewidth=1.2, alpha=0.85)
            ax_tc.set_xlabel("Time (s)")
            ax_tc.set_ylabel("Activation (a.u.)")
            ax_tc.grid(alpha=0.25)
            legend_count = 4 + (len(top_roi_result[3]) if top_roi_result is not None else 0)
            ax_tc.legend(loc="upper right", fontsize=7 if legend_count > 8 else 8, ncol=2 if legend_count > 8 else 1)
            if brain_image is None:
                title = f"Source time course ({self._source_run_scope_label()})  time={initial_time:.3f}s"
                if top_roi_result is not None and top_roi_result[3]:
                    title += f" | top ROI: {top_roi_result[4]}"
                ax_tc.set_title(
                    title
                )
            fig.tight_layout()
            self._set_plot("source", self.source_plot_layout, self.lbl_source_placeholder, fig)
            self._source_time_marker_line = marker_line
            state = self._plot_state.get("source", {})
            canvases = state.get("canvases", [])
            self._source_time_marker_canvas = canvases[0] if canvases else None
            self._source_plot_has_brain_image = bool(brain_image is not None)
            if self._source_status_text:
                suffix = f" | ROI labels unavailable: {top_roi_error}" if top_roi_error else ""
                self.lbl_source_status.setText(f"Source status: {self._source_status_text}{suffix}")
            else:
                suffix = f", ROI labels unavailable: {top_roi_error}" if top_roi_error else ""
                self.lbl_source_status.setText(
                    f"Source status: vertices={data_abs.shape[0]}, samples={data_abs.shape[1]}{suffix}"
                )
        except Exception as exc:
            self.lbl_source_status.setText(f"Source status: source plotting failed - {exc}")
            self._reset_plot_placeholder("source", self.lbl_source_placeholder, f"Could not render source time course: {exc}")

    def _concept_subset_runtime_options(self, subset_batch, backward_batch):
        # Apply after saved resume settings so users can change worker counts
        # without changing the saved scientific settings.
        return {
            "subset_workers": self.spb_concept_subset_workers.value() if subset_batch and not backward_batch else 1,
            "subset_worker_threads": self.spb_concept_subset_worker_threads.value(),
        }

    def _stop_concept_subset_batch(self):
        if self._concept_decode_thread is not None:
            self._concept_decode_thread.requestInterruption()
            self.btn_concept_stop_batch.setEnabled(False)
            self.lbl_concept_status.setText("Concept decoding: stopping new subsets; finishing active extraction/fits and saving results.")

    def _resume_concept_subset_batch(self):
        if self._concept_decode_thread is not None:
            self.lbl_concept_status.setText("Concept decoding: a decoder is already running.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self.widget,
            "Select exhaustive-screening output folder to resume",
            self._concept_output_default_base_dir(),
        )
        if folder:
            self._run_concept_decoding(resume_dir=folder)

    def _run_concept_decoding(self, resume_dir=None):
        if getattr(self, "_concept_image_process", None) is not None:
            QtWidgets.QMessageBox.information(self.widget, "CLIP generation", "Wait for image generation to finish before starting another decoder.")
            return
        if not callable(self._concept_decode_callback):
            self._concept_decode_status_text = "No FIF recording loaded. Load a FIF file (File > Open FIF) before running concept decoding."
            self._concept_decode_result = None
            self._render_concept_decoding()
            return

        resume_settings = {}
        resume_manifest = {}
        if resume_dir:
            try:
                with open(os.path.join(resume_dir, "settings_used.json"), "r", encoding="utf-8") as handle:
                    resume_settings = json.load(handle)
                with open(os.path.join(resume_dir, "manifest.json"), "r", encoding="utf-8") as handle:
                    resume_manifest = json.load(handle)
                if resume_manifest.get("search_strategy") != "exhaustive":
                    raise ValueError("the selected folder is not an exhaustive-screening checkpoint")
            except Exception as exc:
                self.lbl_concept_status.setText(f"Concept decoding: cannot resume checkpoint: {exc}")
                return

        tmin = float(resume_settings.get("window_start", self.spb_concept_tmin.value()))
        tmax = float(resume_settings.get("window_end", self.spb_concept_tmax.value()))
        if tmax <= tmin:
            self._concept_decode_status_text = "Concept decoding requires window end > window start."
            self._concept_decode_result = None
            self._render_concept_decoding()
            return

        channel_idx = int(self.cmb_concept_channels.currentIndex())
        channel_mode = "all"
        if channel_idx == 1:
            channel_mode = "mag"
        elif channel_idx == 2:
            channel_mode = "grad"

        feature_space = resume_settings.get("feature_space", self.cmb_concept_feature_space.currentData() or "signal")
        decoder_type = str(resume_settings.get("decoder_type", self.cmb_concept_decoder.currentData() or "logistic"))
        generation_root = None
        if decoder_type in {"clip_generation", "clip_generation_hybrid"}:
            if self.spb_concept_validation_pct.value() <= 0:
                QtWidgets.QMessageBox.warning(self.widget, "CLIP generation", "A positive validation split is required.")
                return
            generation_root = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "Select folder containing stimulus images")
            if not generation_root:
                return
        semantic_embedding_decoder = decoder_type in {
            "embedding_ridge",
            "embedding_contrastive",
            "embedding_mlp",
            "transformer",
        }
        source_semantic_decoder = bool(
            feature_space == "source"
            and decoder_type in {"embedding_ridge", "embedding_contrastive"}
        )
        split_mode = str(resume_settings.get("split_mode", self.cmb_concept_split.currentData() or "random"))
        if split_mode == "zero_shot" and not semantic_embedding_decoder:
            self.lbl_concept_status.setText(
                "Concept decoding: Zero-shot split requires a semantic embedding decoder "
                "(e.g. Linear ridge, Contrastive, Neural MLP, or Transformer)."
            )
            return
        signal_semantic_decoder = bool(
            semantic_embedding_decoder and not source_semantic_decoder
        )
        feature_mode_choice = str(self.cmb_concept_feature_mode.currentData() or "raw")
        feature_binning_enabled = False
        feature_mode = "raw"
        source_power = feature_space == "source" and feature_mode_choice.startswith("bandpower_")
        if source_power:
            feature_mode = "bandpower"
            feature_binning_enabled = "_binned_" in feature_mode_choice
        elif feature_mode_choice == "mean_binned":
            feature_mode = "mean"
            feature_binning_enabled = True
        elif feature_mode_choice in {"mean_rms_binned", "mean_rms_binned_50"}:
            feature_mode = "mean_rms"
            feature_binning_enabled = True
        elif feature_mode_choice in {"mean_rms_window", "mean_rms"}:
            feature_mode = "mean_rms"
        elif feature_mode_choice in {"mean_window", "mean"}:
            feature_mode = "mean"
        elif feature_mode_choice in {"rms_window", "rms"}:
            feature_mode = "rms"
        elif feature_mode_choice == "raw":
            feature_mode = "raw"
        else:
            feature_text = str(self.cmb_concept_feature_mode.currentText() or "Raw signal samples").lower()
            feature_mode = "mean_rms"
            if feature_text.startswith("binned signed"):
                feature_mode = "mean"
                feature_binning_enabled = True
            elif feature_text.startswith("binned") or feature_text.startswith("50 ms"):
                feature_mode = "mean_rms"
                feature_binning_enabled = True
            elif feature_text.startswith("window mean abs") or feature_text.startswith("mean amplitude"):
                feature_mode = "mean"
            elif feature_text.startswith("window rms") or feature_text.startswith("rms"):
                feature_mode = "rms"
            elif feature_text.startswith("raw"):
                feature_mode = "raw"
        if feature_space != "source":
            feature_mode = "raw"
            feature_binning_enabled = False
        bin_size_ms = float(self.spb_concept_feature_bin_ms.value())
        if signal_semantic_decoder:
            # Compact envelope bins retain band-power meaning while avoiding a
            # 200k-feature flattened input. Very short FFT bins are not used.
            feature_space = "signal"
            feature_mode = "band_power_db"
            feature_binning_enabled = True
            bin_size_ms = max(20.0, bin_size_ms)
        source_parcel_baseline_zscore = bool(
            feature_space == "source"
            and not source_power
            and self.chk_concept_parcel_baseline_zscore.isChecked()
        )

        if self._concept_decode_thread is not None:
            self._concept_decode_status_text = "decoder is already running."
            self.lbl_concept_status.setText(f"Concept decoding: {self._concept_decode_status_text}")
            return

        backward_batch = False if resume_dir else self.chk_concept_backward_elimination.isChecked()
        subset_batch = bool(resume_dir) or self.chk_concept_semantic_subsets.isChecked() or backward_batch
        if subset_batch and not source_semantic_decoder:
            self.lbl_concept_status.setText(
                "Concept decoding: subset batch requires Source parcels "
                "and a linear ridge or contrastive embedding decoder."
            )
            return
        if subset_batch and split_mode not in {"random", "zero_shot"}:
            self.lbl_concept_status.setText(
                "Concept decoding: selected-parcel screening supports Per-concept random "
                "and Zero-shot unseen concepts. Leave-one-run/session is not supported "
                "by the source-space semantic decoder."
            )
            return
        if subset_batch:
            try:
                batch_parcels = (
                    list(resume_manifest.get("candidate_pool", []))
                    if resume_dir else self._concept_selected_parcels()
                )
                batch_total = (
                    int(resume_manifest.get("total_subsets", 0))
                    if resume_dir else
                    backward_fit_count(batch_parcels) if backward_batch else exhaustive_fit_count(batch_parcels)
                )
                if not batch_parcels or batch_total <= 0:
                    raise ValueError("resume checkpoint does not contain a valid parcel pool or subset total")
                if backward_batch and self.spb_concept_validation_pct.value() <= 0:
                    raise ValueError("Backward elimination requires validation percentage greater than zero.")
            except ValueError as exc:
                self.lbl_concept_status.setText(f"Concept decoding: {exc}")
                return

        if subset_batch:
            remaining = batch_total - int(resume_manifest.get("attempted", 0)) if resume_dir else batch_total
            QtWidgets.QMessageBox.warning(
                self, "Parcel screening workload",
                f"{len(batch_parcels)} selected parcels produce {batch_total:,} subset models.\n"
                f"{remaining:,} models remain to run.\n"
                "Large searches can take substantial time and disk space."
            )

        try:
            snr = max(1e-6, float(self.spb_source_snr.value()))
            options = {
                "group_mode": self._concept_group_mode(),
                "window_start": tmin,
                "window_end": tmax,
                "test_fraction": float(self.spb_concept_test_pct.value()) / 100.0,
                "split_mode": self.cmb_concept_split.currentData() or "random",
                "zero_shot_balance_categories": bool(
                    split_mode == "zero_shot" and self._concept_group_mode() == "canonical"
                ),
                "min_trials_per_class": int(self.spb_concept_min_trials.value()),
                "feature_binning_enabled": bool(feature_binning_enabled),
                "bin_size_ms": float(bin_size_ms),
                "channel_mode": channel_mode,
                "max_channels": int(self.spb_concept_max_channels.value()),
                "max_rois": int(self.spb_concept_max_channels.value()),
                "feature_space": feature_space,
                "source_space_kind": self.cmb_concept_source_space_kind.currentData() or "surface",
                "source_parcellation": self.cmb_concept_source_parcellation.currentData() or "aparc",
                "selected_parcellations": self._concept_selected_parcels(),
                "method": str(self.cmb_source_method.currentText() or "MNE"),
                "snr": snr,
                "lambda2": 1.0 / (snr ** 2),
                "baseline_correct": bool(
                    self.chk_concept_baseline.isChecked()
                    and not source_parcel_baseline_zscore
                    and not source_power
                ),
                "source_parcel_baseline_zscore": source_parcel_baseline_zscore,
                "feature_mode": feature_mode,
                "source_power_bands": self.cmb_concept_power_band.currentData() or "all",
                "source_power_baseline": bool(source_power and feature_mode_choice.endswith("_db")),
                "n_permutations": int(self.spb_concept_permutations.value()),
                "decoder_type": decoder_type,
                "pairwise_supertrial_size": self.spb_pairwise_size.value(),
                "pairwise_repeats": self.spb_pairwise_repeats.value(),
                "embedding_dict_path": str(self.le_concept_embedding_path.text() or "").strip(),
                "generation_image_root": generation_root,
                "generation_output_base": self._concept_output_default_base_dir(),
                "epochs": int(self.spb_concept_epochs.value()),
                "mlp_epochs": int(self.spb_concept_epochs.value()),
                "validation_fraction": float(self.spb_concept_validation_pct.value()) / 100.0,
                "mlp_validation_fraction": float(self.spb_concept_validation_pct.value()) / 100.0,
                "mlp_dropout": float(self.spb_concept_mlp_dropout.value()),
                "avg_trials": int(self.spb_concept_avg_trials.value()),
                "trials_per_average": int(self.spb_concept_avg_trials.value()),
                "pseudo_epoch_trials": int(self.spb_concept_avg_trials.value()),
                "task_stratified_pseudo_epochs": bool(
                    getattr(self, "chk_concept_task_stratified_avg", None) is not None
                    and self.chk_concept_task_stratified_avg.isChecked()
                ),
                # Legacy transformer_ aliases:
                "transformer_trials_per_average": int(self.spb_concept_avg_trials.value()),
                "transformer_epochs": int(self.spb_concept_epochs.value()),
                "transformer_model_type": "axial_transformer",
                "semantic_power_padding_s": 0.5,
                "semantic_auto_compact_features": bool(signal_semantic_decoder),
                "embedding_mlp_contrastive_weight": 0.75,
                "embedding_mlp_temperature": 0.07,
                "embedding_contrastive_temperature": 0.07,
                "embedding_contrastive_weight_decay": 1e-2,
                "embedding_contrastive_learning_rate": 1e-3,
                "embedding_input_pca_components": int(
                    self.spb_concept_embedding_input_pca.value()
                ),
                "embedding_target_svd_components": int(
                    self.spb_concept_embedding_target_svd.value()
                ),
                "embedding_early_stopping_patience": int(
                    self.spb_concept_embedding_patience.value()
                ),
                "embedding_early_stopping_min_delta": 1e-4,
                "embedding_dynamic_pseudo_epochs": bool(
                    self.chk_concept_dynamic_pseudo_epochs.isChecked()
                ),
                "source_batch_size": 4 if source_power else (8 if feature_mode == "raw" else 16),
                "max_source_features": 75000,
            }
            options.update(self._source_parcel_preprocessing_options())
            if resume_dir:
                options.update(resume_settings)
                options.update(
                    semantic_subset_batch=True,
                    semantic_subset_resume_dir=str(resume_dir),
                    subset_search_strategy="exhaustive",
                    feature_space="source",
                    selected_parcellations=batch_parcels,
                    max_rois=0,
                )
            elif subset_batch:
                options.update(semantic_subset_batch=True,
                               subset_search_strategy="backward" if backward_batch else "exhaustive",
                               semantic_subset_output_dir=self._concept_output_default_base_dir(),
                               selected_parcellations=batch_parcels, max_rois=0)
            options.update(self._concept_subset_runtime_options(subset_batch, backward_batch))
        except Exception as exc:
            self._concept_decode_result = None
            self._concept_decode_status_text = f"Concept decoding failed: {exc}"
            self._render_concept_decoding()
            return

        btn = getattr(self, "btn_concept_decode", None)
        if btn is not None:
            btn.setEnabled(False)
        save_btn = getattr(self, "btn_concept_save_outputs", None)
        if save_btn is not None:
            save_btn.setEnabled(False)
        self._concept_decode_result = None
        self._concept_decode_last_options = dict(options)
        self.chk_concept_semantic_subsets.setEnabled(False)
        self.chk_concept_backward_elimination.setEnabled(False)
        self.btn_concept_resume_batch.setEnabled(False)
        self.spb_concept_subset_workers.setEnabled(False)
        self.spb_concept_subset_worker_threads.setEnabled(False)
        self.btn_concept_stop_batch.setEnabled(subset_batch)
        self._concept_decode_status_text = (
            f"starting {batch_total} selected-parcel subset fits with automatic saving..."
            if subset_batch else "starting background decoder worker..."
        )
        self._render_concept_decoding()
        if subset_batch and options["subset_workers"] > 1:
            self._reset_concept_training_curve(
                "Parallel screening reports completed subsets and rate in the status line. "
                "Per-fit training metrics are saved with each subset's results."
            )
        elif str(options.get("decoder_type", "")) in {
            "mlp",
            "embedding_contrastive",
            "embedding_mlp",
        }:
            self._start_concept_training_curve()
        else:
            self._reset_concept_training_curve(
                "Live training and validation accuracy are available for the "
                "classification MLP and trainable semantic-embedding decoders."
            )

        thread = QtCore.QThread()
        worker = _ConceptDecodeWorker(self._concept_decode_callback, options)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_concept_decode_progress)
        worker.log.connect(self._on_concept_decode_log)
        worker.training_metric.connect(self._on_concept_training_metric)
        worker.finished.connect(self._on_concept_decode_finished)
        worker.failed.connect(self._on_concept_decode_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(lambda *_args: worker.deleteLater())
        worker.failed.connect(lambda *_args: worker.deleteLater())
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_concept_decode_worker_refs)
        self._concept_decode_thread = thread
        self._concept_decode_worker = worker
        thread.start()

    def _clear_concept_decode_worker_refs(self):
        self._concept_decode_thread = None
        self._concept_decode_worker = None
        self.chk_concept_semantic_subsets.setEnabled(True)
        self.chk_concept_backward_elimination.setEnabled(True)
        self.btn_concept_resume_batch.setEnabled(True)
        self.spb_concept_subset_workers.setEnabled(True)
        self.spb_concept_subset_worker_threads.setEnabled(True)
        self.btn_concept_stop_batch.setEnabled(False)

    def _on_concept_decode_progress(self, message):
        self._concept_decode_status_text = str(message)
        self.lbl_concept_status.setText(f"Concept decoding: {self._concept_decode_status_text}")

    def _on_concept_decode_log(self, message):
        try:
            self.c.log.emit(str(message))
        except Exception:
            pass

    def _on_concept_decode_finished(self, result, message):
        self._concept_decode_result = result if isinstance(result, dict) else None
        self._concept_decode_status_text = str(message)
        btn = getattr(self, "btn_concept_decode", None)
        if btn is not None:
            btn.setEnabled(True)
        save_btn = getattr(self, "btn_concept_save_outputs", None)
        if save_btn is not None:
            save_btn.setEnabled(bool(self._concept_decode_result))
        self._render_concept_decoding()

    def _on_concept_decode_failed(self, message):
        self._concept_decode_result = None
        self._concept_decode_status_text = f"Concept decoding failed: {message}"
        btn = getattr(self, "btn_concept_decode", None)
        if btn is not None:
            btn.setEnabled(True)
        save_btn = getattr(self, "btn_concept_save_outputs", None)
        if save_btn is not None:
            save_btn.setEnabled(False)
        self._render_concept_decoding()

    @staticmethod
    def _safe_filename_token(text, fallback="run"):
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip()).strip("._-")
        return token or str(fallback)

    @classmethod
    def _safe_excel_sheet_name(cls, text, used):
        base = re.sub(r"[\[\]:*?/\\]+", "_", str(text or "sheet")).strip() or "sheet"
        base = base[:31]
        name = base
        suffix = 1
        while name in used:
            tail = f"_{suffix}"
            name = f"{base[:31 - len(tail)]}{tail}"
            suffix += 1
        used.add(name)
        return name

    @staticmethod
    def _json_safe_value(value):
        if isinstance(value, (str, bool, int)) or value is None:
            return value
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            val = float(value)
            return val if np.isfinite(val) else None
        if isinstance(value, np.ndarray):
            if value.size <= 1000:
                return MEGAnalysisPanel._json_safe_value(value.tolist())
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, pd.DataFrame):
            return {"type": "dataframe", "rows": int(len(value)), "columns": [str(v) for v in value.columns]}
        if isinstance(value, dict):
            return {str(k): MEGAnalysisPanel._json_safe_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [MEGAnalysisPanel._json_safe_value(v) for v in value]
        return str(value)

    def _concept_output_default_base_dir(self):
        if self._concept_outputs_base_dir:
            return self._concept_outputs_base_dir
        start_dir = os.path.dirname(str(self._fif_filename)) if self._fif_filename else os.getcwd()
        return os.path.join(start_dir, "concept_decoding_outputs")

    def _concept_output_folder_name(self, result):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        decoder = self._safe_filename_token(result.get("decoder_type", result.get("decoder", "decoder")), "decoder")
        feature_space = self._safe_filename_token(result.get("feature_space", "space"), "space")
        group_mode = self._safe_filename_token(result.get("group_mode", "concepts"), "concepts")
        metric = str(
            result.get("primary_metric", result.get("metric", "accuracy"))
            or "accuracy"
        ).strip().upper()
        try:
            acc = float(result.get("accuracy", np.nan))
        except Exception:
            acc = np.nan
        if metric == "MPR":
            try:
                top1 = float(result.get("top1_accuracy", np.nan))
            except Exception:
                top1 = np.nan
            mpr_token = f"mpr{acc:.3f}" if np.isfinite(acc) else "mprNA"
            top1_token = f"top1{top1:.3f}" if np.isfinite(top1) else "top1NA"
            acc_token = f"{mpr_token}_{top1_token}"
        else:
            acc_token = f"acc{acc:.3f}" if np.isfinite(acc) else "accNA"
        return self._safe_filename_token(f"{timestamp}_{group_mode}_{feature_space}_{decoder}_{acc_token}", "concept_run")

    def _concept_result_tables(self, result):
        table_keys = [
            ("timecourse", "timecourse_df"),
            ("pairwise_accuracy", "pairwise_accuracy_df"),
            ("pairwise_trial_index", "pairwise_trial_index_df"),
            ("summary", "summary_df"),
            ("metrics", "metrics_df"),
            ("training_history", "training_history_df"),
            ("per_class_accuracy", "per_class_accuracy_df"),
            ("predictions", "predictions_df"),
            ("split", "split_df"),
            ("features", "feature_df"),
            ("permutations", "permutation_df"),
            ("templates", "template_df"),
            ("template_feature_meta", "template_feature_meta_df"),
            ("model_architecture", "model_architecture_df"),
            ("trial_qc", "trial_qc_df"),
            ("qc_summary", "qc_summary_df"),
        ]
        tables = []
        for label, key in table_keys:
            df = result.get(key, pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
            if isinstance(df, pd.DataFrame) and not df.empty:
                tables.append((label, df.copy()))
        return tables

    def _write_concept_tables(self, run_dir, result):
        tables = self._concept_result_tables(result)
        tables_dir = os.path.join(run_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        for label, df in tables:
            df.to_csv(os.path.join(tables_dir, f"{label}.csv"), index=False)

        workbook_path = os.path.join(run_dir, "concept_decoding_outputs.xlsx")
        if tables:
            used = set()
            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                for label, df in tables:
                    sheet_name = self._safe_excel_sheet_name(label, used)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        return workbook_path if tables else ""

    def _write_concept_arrays(self, run_dir, result):
        if "pairwise_accuracy_matrix" in result:
            np.savez_compressed(os.path.join(run_dir, "pairwise_accuracy.npz"),
                                accuracy=result["pairwise_accuracy_matrix"],
                                times_s=result["pairwise_times_s"], class_names=result["class_names"])
        cm = np.asarray(result.get("confusion_matrix", []), dtype=float)
        if cm.ndim == 2 and cm.size > 0:
            class_names = [str(v) for v in result.get("class_names", [])]
            cm_df = pd.DataFrame(cm)
            if len(class_names) == cm.shape[0]:
                cm_df.index = class_names
            if len(class_names) == cm.shape[1]:
                cm_df.columns = class_names
            cm_df.to_csv(os.path.join(run_dir, "confusion_matrix.csv"))

        template_matrix = np.asarray(result.get("template_matrix", []), dtype=float)
        if template_matrix.ndim == 2 and template_matrix.size > 0:
            template_classes = [str(v) for v in result.get("template_classes", [])]
            feature_names = [str(v) for v in result.get("feature_names", [])]
            matrix_df = pd.DataFrame(template_matrix)
            if len(template_classes) == template_matrix.shape[0]:
                matrix_df.index = template_classes
            if len(feature_names) == template_matrix.shape[1]:
                matrix_df.columns = feature_names
            matrix_df.to_csv(os.path.join(run_dir, "template_matrix.csv"))

        class_names = [str(v) for v in result.get("class_names", [])]
        if class_names:
            pd.DataFrame({"class_name": class_names}).to_csv(os.path.join(run_dir, "class_names.csv"), index=False)

    def _write_concept_figures(self, run_dir):
        figures_dir = os.path.join(run_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        saved = []
        for key in (
            "concept_training",
            "concept_confusion",
            "concept_accuracy",
            "concept_templates",
            "concept_trial_qc",
        ):
            state = self._plot_state.get(key, {}) if isinstance(getattr(self, "_plot_state", None), dict) else {}
            figs = list(state.get("figs", []) or [])
            for idx, fig in enumerate(figs, start=1):
                if fig is None:
                    continue
                name = f"{key}.png" if len(figs) == 1 else f"{key}_{idx}.png"
                path = os.path.join(figures_dir, name)
                fig.savefig(path, dpi=180, bbox_inches="tight")
                saved.append(path)
        return saved

    def _concept_accuracy_tracker_rows(self, run_dir, folder_name, result, options):
        per_class_df = result.get("per_class_accuracy_df", pd.DataFrame())
        if isinstance(per_class_df, pd.DataFrame) and not per_class_df.empty:
            rows = per_class_df.copy()
        else:
            rows = pd.DataFrame([{"concept": "__overall__", "test_accuracy": result.get("accuracy", np.nan), "n_test": result.get("n_test", np.nan)}])
        rows.insert(0, "folder_name", str(folder_name))
        rows.insert(1, "run_folder", str(run_dir))
        rows.insert(2, "export_timestamp", datetime.now().isoformat(timespec="seconds"))

        extra = {
            "overall_accuracy": result.get("accuracy", np.nan),
            "balanced_accuracy": result.get("balanced_accuracy", np.nan),
            "chance_accuracy": result.get("chance_accuracy", np.nan),
            "top1_accuracy": result.get("top1_accuracy", result.get("accuracy", np.nan)),
            "top1_balanced_accuracy": result.get(
                "top1_balanced_accuracy", result.get("balanced_accuracy", np.nan)
            ),
            "top1_chance_accuracy": result.get(
                "top1_chance_accuracy", result.get("chance_accuracy", np.nan)
            ),
            "mean_reciprocal_rank": result.get("mean_reciprocal_rank", np.nan),
            "decoder": result.get("decoder", ""),
            "decoder_type": result.get("decoder_type", ""),
            "feature_space": result.get("feature_space", options.get("feature_space", "")),
            "feature_mode": options.get("feature_mode", ""),
            "source_power_bands": options.get("source_power_bands", ""),
            "source_power_baseline": options.get("source_power_baseline", False),
            "band_defs": result.get("band_defs", []),
            "feature_binning_enabled": options.get("feature_binning_enabled", ""),
            "bin_size_ms": options.get("bin_size_ms", ""),
            "source_parcel_baseline_zscore": options.get(
                "source_parcel_baseline_zscore", False
            ),
            "parcel_baseline_start": options.get("parcel_baseline_start", np.nan),
            "parcel_baseline_end": options.get("parcel_baseline_end", np.nan),
            "source_space_kind": result.get("source_space_kind", options.get("source_space_kind", "")),
            "source_parcellation": result.get("source_parcellation", options.get("source_parcellation", "")),
            "n_rois": result.get("n_rois", np.nan),
            "n_channels": result.get("n_channels", np.nan),
            "n_features": result.get("n_features", np.nan),
            "window_start": options.get("window_start", result.get("window_start_s", np.nan)),
            "window_end": options.get("window_end", result.get("window_end_s", np.nan)),
            "split_mode": options.get("split_mode", result.get("split", "")),
            "group_mode": options.get("group_mode", result.get("group_mode", "")),
            "resample_enabled": options.get("mne_resample_enabled", result.get("resample_enabled", "")),
            "resample_sfreq": options.get("resample_sfreq", result.get("resample_sfreq", np.nan)),
            "selected_parcellations": ";".join(str(v) for v in options.get("selected_parcellations", []) or []),
        }
        for key, value in extra.items():
            rows[key] = value
        return rows

    def _append_concept_accuracy_tracker(self, base_dir, run_dir, folder_name, result, options):
        tracker_path = os.path.join(base_dir, "concept_decoding_accuracy_log.xlsx")
        new_rows = self._concept_accuracy_tracker_rows(run_dir, folder_name, result, options)
        if os.path.isfile(tracker_path):
            try:
                old_rows = pd.read_excel(tracker_path)
            except Exception:
                old_rows = pd.DataFrame()
        else:
            old_rows = pd.DataFrame()
        combined = pd.concat([old_rows, new_rows], ignore_index=True, sort=False) if not old_rows.empty else new_rows
        combined.to_excel(tracker_path, index=False)
        combined.to_csv(os.path.join(base_dir, "concept_decoding_accuracy_log.csv"), index=False)
        return tracker_path

    def _open_concept_images_folder(self):
        folder = getattr(self, "_concept_images_folder", None)
        if folder and os.path.isdir(folder):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def _stop_concept_images(self):
        from pathlib import Path
        folder = getattr(self, "_concept_image_batch_folder", None)
        if folder and getattr(self, "_concept_image_process", None) is not None:
            Path(folder, "STOP").touch()
            self.btn_concept_stop_images.setEnabled(False)
            self.lbl_concept_status.setText("Stopping after the current image; completed images will be kept.")

    def _generate_concept_image(self, batch=False):
        from pathlib import Path
        from core.concept_image_generation import worker_command, validate_bundle
        import joblib
        if getattr(self, "_concept_image_process", None) is not None:
            QtWidgets.QMessageBox.information(self.widget, "Image generation", "An image is already generating.")
            return
        if getattr(self, "_concept_decode_thread", None) is not None:
            QtWidgets.QMessageBox.information(self.widget, "Image generation", "Wait for training to finish first.")
            return
        result = self._concept_decode_result or {}
        bundle_path = result.get("generation_bundle_path")
        if not bundle_path:
            bundle_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.widget, "Select a trusted compatible saved inference bundle", self._concept_output_default_base_dir(), "Model bundle (inference_bundle.joblib)")
            if not bundle_path:
                return
        try:
            bundle = joblib.load(bundle_path)
            validate_bundle(bundle)
            if batch:
                index = 0
                out = Path(bundle_path).parent / "images_all_comparisons"
                answer = QtWidgets.QMessageBox.question(
                    self.widget,
                    "Generate all test predictions",
                    f"Generate/resume {len(bundle['predictions'])} comparisons? Each comparison shows the input stimuli, "
                    "the MEG prediction, and a third image generated from MEG prediction minus the block concept vector. "
                    "Block vectors are means of image_1..image_3 in block-instructions.csv. Completed matching "
                    f"comparisons are skipped. This may take hours.\n\n{out}",
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    return
                out.mkdir(parents=True, exist_ok=True)
                (out / "STOP").unlink(missing_ok=True)
                command = worker_command("batch", bundle_path, index, out)
            else:
                index, ok = QtWidgets.QInputDialog.getInt(self.widget, "Held-out prediction", "Test average index (matches Predictions table; zero-based)", 0, 0, len(bundle["predictions"])-1)
                if not ok:
                    return
                out = Path(bundle_path).parent / ("images_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                command = worker_command("generate", bundle_path, index, out)
            process = QtCore.QProcess(self.widget)
            env = QtCore.QProcessEnvironment.systemEnvironment()
            env.insert("HF_HUB_OFFLINE", "1")
            process.setProcessEnvironment(env)
            process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self._concept_image_process = process
            self._concept_image_log = ""
            self._concept_image_batch_folder = str(out) if batch else None
            self.btn_concept_generate_all.setEnabled(False)
            self.btn_concept_stop_images.setEnabled(batch)
            self.concept_image_progress.setRange(0, 0)
            self.btn_concept_generate_image.setEnabled(False)
            self.btn_concept_generate_image.setText("Generating…")
            self.lbl_concept_status.setText("Starting batch image generation…" if batch else f"Starting local image generation for test average {index}…")
            def read_output():
                self._concept_image_log += bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
                self.lbl_concept_status.setText("Experimental image generation is running locally…")
                matches = re.findall(r"BATCH_PROGRESS (\d+) (\d+)", self._concept_image_log)
                if matches:
                    done, total = map(int, matches[-1])
                    self.concept_image_progress.setRange(0, total)
                    self.concept_image_progress.setValue(done)
                    self.lbl_concept_status.setText(f"Image batch: {done}/{total} comparisons saved.")
            def finish(code, *_args):
                if self._concept_image_process is not process:
                    return
                read_output()
                self._concept_image_process = None
                self.btn_concept_generate_image.setEnabled(True)
                self.btn_concept_generate_image.setText("Generate image…")
                self.btn_concept_generate_all.setEnabled(True)
                self.btn_concept_stop_images.setEnabled(False)
                if self.concept_image_progress.maximum() == 0:
                    self.concept_image_progress.setRange(0, 1)
                    self.concept_image_progress.setValue(int(code == 0))
                if out.is_dir():
                    self._concept_images_folder = str(out)
                    self.btn_concept_open_images.setEnabled(True)
                    self.btn_concept_open_images.setToolTip(str(out))
                if code == 0 and batch and (out / "batch_manifest.json").exists():
                    state = json.loads((out / "batch_manifest.json").read_text())
                    self.lbl_concept_status.setText(f"Image batch {state['status']}. Saved to {out}. Use Generate all / resume to continue.")
                elif code == 0 and (out / "comparison.png").exists():
                    dialog = QtWidgets.QDialog(self.widget)
                    dialog.setWindowTitle(
                        "Experimental: stimulus / ideal image / MEG prediction / prediction minus block concept"
                        if bundle["generation_profile"]["target"] == "individual_stimulus_image"
                        else "Experimental: stimulus / ideal concept / MEG prediction / prediction minus block concept"
                    )
                    layout = QtWidgets.QVBoxLayout(dialog)
                    layout.addWidget(QtWidgets.QLabel("Generated appearance is not evidence of accurate decoding. No text prompts used."))
                    label = QtWidgets.QLabel()
                    label.setPixmap(QtGui.QPixmap(str(out / "comparison.png")).scaledToWidth(1100, QtCore.Qt.SmoothTransformation))
                    layout.addWidget(label)
                    location = QtWidgets.QLabel(str(out)); location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                    layout.addWidget(location)
                    dialog.show()
                    self._concept_image_dialog = dialog
                    self.lbl_concept_status.setText(f"Generated images saved to {out}")
                else:
                    self.lbl_concept_status.setText("Image generation failed; see error details.")
                    QtWidgets.QMessageBox.warning(self.widget, "Image generation failed", self._concept_image_log[-2000:] or "Could not start image runtime.")
                process.deleteLater()
            process.readyReadStandardOutput.connect(read_output)
            process.finished.connect(finish)
            process.errorOccurred.connect(lambda error: finish(-1) if error == QtCore.QProcess.FailedToStart else None)
            process.start(command[0], command[1:])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.widget, "Image generation", str(exc))

    def _save_concept_outputs(self):
        result = self._concept_decode_result if isinstance(self._concept_decode_result, dict) else None
        if not result:
            self.lbl_concept_status.setText("Concept decoding: train/test decoder before saving outputs.")
            return

        default_base_dir = self._concept_output_default_base_dir()
        try:
            os.makedirs(default_base_dir, exist_ok=True)
        except Exception:
            default_base_dir = os.getcwd()
        base_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select Concept decoding output folder",
            default_base_dir,
        )
        if not base_dir:
            return
        base_dir = os.path.abspath(str(base_dir))
        os.makedirs(base_dir, exist_ok=True)
        self._concept_outputs_base_dir = base_dir

        folder_name = self._concept_output_folder_name(result)
        run_dir = os.path.join(base_dir, folder_name)
        counter = 2
        while os.path.exists(run_dir):
            run_dir = os.path.join(base_dir, f"{folder_name}_{counter}")
            counter += 1
        folder_name = os.path.basename(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        options = dict(getattr(self, "_concept_decode_last_options", {}) or {})
        save_btn = getattr(self, "btn_concept_save_outputs", None)
        try:
            if save_btn is not None:
                save_btn.setEnabled(False)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.lbl_concept_status.setText(f"Concept decoding: saving outputs to {folder_name}...")
            QtWidgets.QApplication.processEvents()

            with open(os.path.join(run_dir, "settings_used.json"), "w", encoding="utf-8") as f:
                json.dump(self._json_safe_value(options), f, indent=2, sort_keys=True)
            with open(os.path.join(run_dir, "result_summary.json"), "w", encoding="utf-8") as f:
                json.dump(self._json_safe_value(result), f, indent=2, sort_keys=True)

            workbook_path = self._write_concept_tables(run_dir, result)
            self._write_concept_arrays(run_dir, result)
            saved_figs = self._write_concept_figures(run_dir)
            tracker_path = self._append_concept_accuracy_tracker(base_dir, run_dir, folder_name, result, options)

            manifest = {
                "folder_name": folder_name,
                "run_folder": run_dir,
                "settings_file": os.path.join(run_dir, "settings_used.json"),
                "result_summary_file": os.path.join(run_dir, "result_summary.json"),
                "result_workbook": workbook_path,
                "accuracy_tracker": tracker_path,
                "figures": saved_figs,
            }
            with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(self._json_safe_value(manifest), f, indent=2, sort_keys=True)

            self._concept_decode_status_text = (
                f"saved outputs to {run_dir}; updated {os.path.basename(tracker_path)}"
            )
            self.lbl_concept_status.setText(f"Concept decoding: {self._concept_decode_status_text}")
        except Exception as exc:
            self._concept_decode_status_text = f"save outputs failed: {exc}"
            self.lbl_concept_status.setText(f"Concept decoding: {self._concept_decode_status_text}")
        finally:
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass
            if save_btn is not None:
                save_btn.setEnabled(bool(self._concept_decode_result))

    def _set_dataframe_table(self, table, model_attr, df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            model = PandasModel(df.copy())
            setattr(self, model_attr, model)
            table.setModel(model)
            table.resizeColumnsToContents()
            return
        table.setModel(None)
        setattr(self, model_attr, None)

    def _reset_concept_training_curve(self, message):
        """Clear the live MLP curve and restore its placeholder."""
        self._remove_plot("concept_training")
        self._concept_training_history = []
        self._concept_training_series = None
        self._concept_training_figure = None
        self._concept_training_axis = None
        self._concept_training_lines = {}
        placeholder = getattr(self, "lbl_concept_training_placeholder", None)
        if placeholder is not None:
            placeholder.setText(str(message))
            placeholder.setVisible(True)

    def _configure_concept_training_axis(self, series_label="", total_epochs=1):
        ax = self._concept_training_axis
        if ax is None:
            return
        ax.clear()
        train_line, = ax.plot(
            [], [], color="#2563eb", linewidth=2.2, marker="o", markersize=3,
            label="Training accuracy",
        )
        validation_line, = ax.plot(
            [], [], color="#dc2626", linewidth=2.2, marker="o", markersize=3,
            label="Validation accuracy",
        )
        chance_line = ax.axhline(
            0.0, color="#64748b", linewidth=1.0, linestyle=":", label="Chance"
        )
        best_line = ax.axvline(
            1.0, color="#16a34a", linewidth=1.2, linestyle="--", label="Peak validation accuracy"
        )
        best_line.set_visible(False)
        ax.set_xlim(1, max(2, int(total_epochs)))
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(str(series_label or "Waiting for first neural-network epoch..."))
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", fontsize=8)
        self._concept_training_lines = {
            "train": train_line,
            "validation": validation_line,
            "chance": chance_line,
            "best": best_line,
        }

    def _start_concept_training_curve(self):
        """Create the canvas before the worker starts emitting neural-network epochs."""
        from matplotlib.figure import Figure

        self._reset_concept_training_curve(
            "Waiting for the first trainable-decoder epoch..."
        )
        total_epochs = int(self.spb_concept_transformer_epochs.value())
        figure = Figure(figsize=(8.2, 5.0), constrained_layout=True)
        self._concept_training_figure = figure
        self._concept_training_axis = figure.subplots(1, 1)
        self._configure_concept_training_axis(
            "Preparing decoder training...", total_epochs
        )
        self._set_plot(
            "concept_training",
            self.concept_training_layout,
            self.lbl_concept_training_placeholder,
            figure,
        )
        self.concept_result_tabs.setCurrentWidget(self.concept_training_tab)

    def _on_concept_training_metric(self, metric):
        """Update the live train/validation accuracy graph on the Qt thread."""
        if not isinstance(metric, dict) or str(metric.get("model", "")) not in {
            "mlp",
            "embedding_contrastive",
            "embedding_mlp",
        }:
            return
        try:
            series = int(metric.get("series", 1))
            epoch = int(metric.get("epoch", 0))
            total_epochs = max(epoch, int(metric.get("epochs", epoch)))
        except Exception:
            return
        if epoch <= 0:
            return
        if self._concept_training_figure is None or self._concept_training_axis is None:
            self._start_concept_training_curve()
        if self._concept_training_series != series:
            self._concept_training_series = series
            self._concept_training_history = []
            self._configure_concept_training_axis(
                str(metric.get("series_label", "") or f"Training fit {series}"),
                total_epochs,
            )
            if str(metric.get("model", "")) in {
                "embedding_contrastive",
                "embedding_mlp",
            }:
                self._concept_training_lines["train"].set_label("Training retrieval accuracy")
                self._concept_training_lines["validation"].set_label("Validation retrieval accuracy")
                self._concept_training_axis.set_ylabel("Nearest-concept accuracy")
                self._concept_training_axis.legend(loc="lower right", fontsize=8)

        self._concept_training_history.append(dict(metric))
        epochs = np.asarray(
            [int(row.get("epoch", 0)) for row in self._concept_training_history],
            dtype=int,
        )
        train_accuracy = np.asarray(
            [float(row.get("train_accuracy", np.nan)) for row in self._concept_training_history],
            dtype=float,
        )
        validation_accuracy = np.asarray(
            [float(row.get("validation_accuracy", np.nan)) for row in self._concept_training_history],
            dtype=float,
        )
        self._concept_training_lines["train"].set_data(epochs, train_accuracy)
        self._concept_training_lines["validation"].set_data(
            epochs, validation_accuracy
        )
        chance = float(metric.get("chance_accuracy", np.nan))
        if np.isfinite(chance):
            self._concept_training_lines["chance"].set_ydata([chance, chance])
        finite_validation = np.flatnonzero(np.isfinite(validation_accuracy))
        best_line = self._concept_training_lines["best"]
        if finite_validation.size:
            best_local = int(
                finite_validation[np.argmax(validation_accuracy[finite_validation])]
            )
            best_epoch = int(epochs[best_local])
            best_line.set_xdata([best_epoch, best_epoch])
            best_line.set_visible(True)
        else:
            best_line.set_visible(False)

        series_label = str(metric.get("series_label", "") or f"Training fit {series}")
        n_train = int(metric.get("n_train", 0) or 0)
        n_validation = int(metric.get("n_validation", 0) or 0)
        validation_note = (
            f"validation n={n_validation}; validation is drawn only from the training split"
            if n_validation > 0
            else "no validation subset available; increase the validation split or provide more trials"
        )
        self._concept_training_axis.set_title(
            f"{series_label} · epoch {epoch}/{total_epochs} · train n={n_train}\n"
            f"{validation_note}"
        )
        self._concept_training_axis.set_xlim(1, max(2, total_epochs))
        self._concept_training_axis.set_ylim(0.0, 1.05)
        state = self._plot_state.get("concept_training", {})
        canvases = state.get("canvases", [])
        if canvases:
            canvases[0].draw_idle()

    @staticmethod
    def _concept_result_title(result):
        acc = float(result.get("accuracy", np.nan))
        bal_acc = float(result.get("balanced_accuracy", np.nan))
        n_train = int(result.get("n_train", 0))
        n_test = int(result.get("n_test", 0))
        n_runs = int(result.get("n_runs", 0))
        n_channels = int(result.get("n_channels", 0))
        n_rois = int(result.get("n_rois", 0))
        decoder = str(result.get("decoder", "decoder"))
        metric = str(result.get("primary_metric", result.get("metric", "accuracy")) or "accuracy").strip()
        chance = float(result.get("chance_accuracy", np.nan))
        p_bal = result.get("permutation_p_value_balanced_accuracy", np.nan)
        feature_space = str(result.get("feature_space", "signal"))
        location_text = f"rois={n_rois}" if feature_space == "source" else f"channels={n_channels}"
        if feature_space == "source":
            source_kind = str(result.get("source_space_kind", "") or "").strip()
            source_space = str(result.get("source_common_space", "") or "").strip()
            if source_kind:
                location_text = f"{location_text}, kind={source_kind}"
            if source_space:
                location_text = f"{location_text}, space={source_space}"
        p_text = f", chance={chance:.3f}" if np.isfinite(chance) else ""
        if metric.upper() == "MPR":
            top1 = result.get("top1_accuracy", np.nan)
            top1_chance = result.get("top1_chance_accuracy", np.nan)
            try:
                top1_text = (
                    f", top-1={float(top1):.3f} (chance={float(top1_chance):.3f})"
                    if np.isfinite(float(top1)) and np.isfinite(float(top1_chance))
                    else ""
                )
            except Exception:
                top1_text = ""
            try:
                p_bal_text = f", perm p(bal MPR)={float(p_bal):.3f}" if np.isfinite(float(p_bal)) else ""
            except Exception:
                p_bal_text = ""
            collapse_text = ""
            if bool(result.get("prediction_collapse_detected", False)):
                collapse_text = (
                    f", COLLAPSE: {result.get('predicted_unique_concepts', 0)} unique; "
                    f"{result.get('most_common_prediction', '')}="
                    f"{float(result.get('most_common_prediction_fraction', 0.0)):.0%}"
                )
            return (
                f"{decoder}: MPR={acc:.3f}, balanced MPR={bal_acc:.3f}, "
                f"train/test={n_train}/{n_test}{top1_text}, runs={n_runs}, "
                f"{location_text}{p_text}{p_bal_text}{collapse_text}"
            )
        try:
            p_bal_text = f", perm p(bal)={float(p_bal):.3f}" if np.isfinite(float(p_bal)) else ""
        except Exception:
            p_bal_text = ""
        return (
            f"{decoder}: accuracy={acc:.3f}, balanced={bal_acc:.3f}, "
            f"train/test={n_train}/{n_test}, runs={n_runs}, {location_text}{p_text}{p_bal_text}"
        )

    def _render_concept_confusion_plot(self, result, class_names):
        try:
            import matplotlib.pyplot as plt

            cm = np.asarray(result.get("confusion_matrix", []), dtype=float)
            fig = plt.figure(figsize=(8.4, 6.4))
            ax = fig.add_subplot(1, 1, 1)
            if cm.ndim == 2 and cm.shape[0] > 0 and cm.shape[1] > 0:
                im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title("Held-out confusion matrix")
                ax.set_xlabel("Predicted concept")
                ax.set_ylabel("True concept")
                if class_names and len(class_names) <= 24:
                    ax.set_xticks(np.arange(len(class_names)))
                    ax.set_yticks(np.arange(len(class_names)))
                    ax.set_xticklabels(class_names, rotation=55, ha="right", fontsize=7)
                    ax.set_yticklabels(class_names, fontsize=7)
                else:
                    ax.set_xlabel("Predicted concept index")
                    ax.set_ylabel("True concept index")
                if cm.shape[0] <= 10 and cm.shape[1] <= 10:
                    threshold = float(np.nanmax(cm)) / 2.0 if np.isfinite(np.nanmax(cm)) else 0.0
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(
                                j,
                                i,
                                str(int(cm[i, j])),
                                ha="center",
                                va="center",
                                color="white" if cm[i, j] > threshold else "black",
                                fontsize=8,
                            )
            else:
                ax.text(0.5, 0.5, "No confusion matrix available", ha="center", va="center")
                ax.set_axis_off()
            fig.suptitle(self._concept_result_title(result), fontsize=10)
            fig.tight_layout()
            self._set_plot(
                "concept_confusion",
                self.concept_confusion_layout,
                self.lbl_concept_confusion_placeholder,
                fig,
            )
        except Exception as exc:
            self._reset_plot_placeholder(
                "concept_confusion",
                self.lbl_concept_confusion_placeholder,
                f"Could not render confusion matrix: {exc}",
            )

    def _render_concept_accuracy_plot(self, result, per_class_df):
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8.4, 6.4))
            ax = fig.add_subplot(1, 1, 1)
            metric = str(result.get("primary_metric", result.get("metric", "accuracy")) or "accuracy").strip()
            is_mpr = metric.upper() == "MPR"
            metric_label = "MPR" if is_mpr else "accuracy"
            if result.get("decoder_type") == "pairwise_svm":
                from core.concept_decoding_pairwise import plot_pairwise_timecourse
                plot_pairwise_timecourse(ax, result)
            elif isinstance(per_class_df, pd.DataFrame) and not per_class_df.empty:
                plot_df = per_class_df.copy()
                plot_df["test_accuracy"] = pd.to_numeric(plot_df.get("test_accuracy", np.nan), errors="coerce")
                plot_df = plot_df.sort_values(["test_accuracy", "concept"], ascending=[True, True]).head(40)
                y_pos = np.arange(len(plot_df))
                ax.barh(y_pos, plot_df["test_accuracy"].fillna(0.0), color="#2563eb", alpha=0.82)
                chance = float(result.get("chance_accuracy", np.nan))
                if np.isfinite(chance):
                    ax.axvline(chance, color="#dc2626", linewidth=1.4, linestyle="--", label=f"chance={chance:.3f}")
                    ax.legend(loc="lower right", fontsize=8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(plot_df["concept"].astype(str).tolist(), fontsize=7)
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel(f"Test {metric_label}")
                ax.set_title(f"Per-concept held-out {metric_label}")
                ax.grid(axis="x", alpha=0.25)
            else:
                ax.text(0.5, 0.5, f"No per-concept {metric_label} available", ha="center", va="center")
                ax.set_axis_off()
            fig.suptitle(self._concept_result_title(result), fontsize=10)
            fig.tight_layout()
            self._set_plot(
                "concept_accuracy",
                self.concept_accuracy_layout,
                self.lbl_concept_accuracy_placeholder,
                fig,
            )
        except Exception as exc:
            self._reset_plot_placeholder(
                "concept_accuracy",
                self.lbl_concept_accuracy_placeholder,
                f"Could not render per-concept accuracy: {exc}",
            )

    @staticmethod
    def _template_location_column(meta_df):
        if isinstance(meta_df, pd.DataFrame):
            if "roi" in meta_df.columns:
                return "roi"
            if "channel" in meta_df.columns:
                return "channel"
        return None

    def _render_concept_templates_plot(self, result):
        decoder_type = str(result.get("decoder_type", "")).strip().lower()
        if decoder_type != "template":
            self._reset_plot_placeholder(
                "concept_templates",
                self.lbl_concept_templates_placeholder,
                "Template plots are available when Decoder is set to Template correlation.",
            )
            return

        try:
            import matplotlib.pyplot as plt

            matrix = np.asarray(result.get("template_matrix", []), dtype=float)
            classes = [str(v) for v in result.get("template_classes", [])]
            meta_df = result.get("template_feature_meta_df", pd.DataFrame())
            if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
                raise ValueError("no learned template matrix available")
            if len(classes) != matrix.shape[0]:
                classes = [f"class_{i}" for i in range(matrix.shape[0])]
            if not isinstance(meta_df, pd.DataFrame) or len(meta_df) != matrix.shape[1]:
                meta_df = pd.DataFrame({"feature_index": np.arange(matrix.shape[1], dtype=int)})
            if "feature_index" not in meta_df.columns:
                meta_df = meta_df.copy().reset_index(drop=True)
                meta_df.insert(0, "feature_index", np.arange(len(meta_df), dtype=int))

            loc_col = self._template_location_column(meta_df)
            has_raw_time = "relative_time_s" in meta_df.columns and loc_col is not None
            if has_raw_time:
                abs_template = np.nanmean(np.abs(matrix), axis=0)
                loc_scores = []
                for loc, loc_df in meta_df.groupby(loc_col, sort=False):
                    idx = loc_df["feature_index"].astype(int).to_numpy()
                    idx = idx[(idx >= 0) & (idx < matrix.shape[1])]
                    if idx.size:
                        loc_scores.append((str(loc), float(np.nanmean(abs_template[idx]))))
                loc_scores = sorted(loc_scores, key=lambda item: item[1], reverse=True)[:3]
                if not loc_scores:
                    has_raw_time = False

            if has_raw_time:
                fig, axes = plt.subplots(
                    len(loc_scores),
                    1,
                    figsize=(9.0, max(3.2, 2.6 * len(loc_scores))),
                    squeeze=False,
                )
                vmax = float(np.nanpercentile(np.abs(matrix), 99)) if np.any(np.isfinite(matrix)) else 1.0
                vmax = max(vmax, 1e-12)
                for ax, (loc, _score) in zip(axes[:, 0], loc_scores):
                    loc_df = meta_df[meta_df[loc_col].astype(str) == str(loc)].copy()
                    loc_df["relative_time_s"] = pd.to_numeric(loc_df["relative_time_s"], errors="coerce")
                    loc_df = loc_df.dropna(subset=["relative_time_s"]).sort_values(["relative_time_s", "feature_index"])
                    idx = loc_df["feature_index"].astype(int).to_numpy()
                    idx = idx[(idx >= 0) & (idx < matrix.shape[1])]
                    if idx.size == 0:
                        ax.text(0.5, 0.5, f"No template samples for {loc}", ha="center", va="center")
                        ax.set_axis_off()
                        continue
                    times = loc_df["relative_time_s"].to_numpy(dtype=float)[: idx.size]
                    im = ax.imshow(
                        matrix[:, idx],
                        aspect="auto",
                        interpolation="nearest",
                        cmap="coolwarm",
                        vmin=-vmax,
                        vmax=vmax,
                    )
                    if times.size > 1:
                        tick_idx = np.linspace(0, times.size - 1, num=min(6, times.size), dtype=int)
                        ax.set_xticks(tick_idx)
                        ax.set_xticklabels([f"{times[i]:.3f}" for i in tick_idx], fontsize=8)
                    ax.set_title(f"Template waveforms: {loc}")
                    ax.set_xlabel("Time from stimulus onset (s)")
                    ax.set_ylabel("Concept")
                    if len(classes) <= 30:
                        ax.set_yticks(np.arange(len(classes)))
                        ax.set_yticklabels(classes, fontsize=7)
                    else:
                        ax.set_yticks([])
                    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            else:
                top_n = min(80, matrix.shape[1])
                score = np.nanmean(np.abs(matrix), axis=0)
                idx = np.argsort(np.nan_to_num(score, nan=0.0))[::-1][:top_n]
                fig = plt.figure(figsize=(9.0, 6.4))
                ax = fig.add_subplot(1, 1, 1)
                vmax = float(np.nanpercentile(np.abs(matrix[:, idx]), 99)) if idx.size else 1.0
                vmax = max(vmax, 1e-12)
                im = ax.imshow(
                    matrix[:, idx],
                    aspect="auto",
                    interpolation="nearest",
                    cmap="coolwarm",
                    vmin=-vmax,
                    vmax=vmax,
                )
                if len(classes) <= 30:
                    ax.set_yticks(np.arange(len(classes)))
                    ax.set_yticklabels(classes, fontsize=7)
                else:
                    ax.set_yticks([])
                if idx.size <= 30 and "feature" in meta_df.columns:
                    labels = meta_df.iloc[idx]["feature"].astype(str).tolist()
                    ax.set_xticks(np.arange(idx.size))
                    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
                else:
                    ax.set_xlabel("Top template feature index")
                ax.set_title("Learned concept templates")
                ax.set_ylabel("Concept")
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

            fig.suptitle("Template correlation prototypes in standardized feature space", fontsize=10)
            fig.tight_layout()
            self._set_plot(
                "concept_templates",
                self.concept_templates_plot_layout,
                self.lbl_concept_templates_placeholder,
                fig,
            )
        except Exception as exc:
            self._reset_plot_placeholder(
                "concept_templates",
                self.lbl_concept_templates_placeholder,
                f"Could not render concept templates: {exc}",
            )

    def _render_concept_trial_qc_plot(self, result):
        try:
            import matplotlib.pyplot as plt

            qc_df = result.get("trial_qc_df", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
            if not isinstance(qc_df, pd.DataFrame) or qc_df.empty:
                self._reset_plot_placeholder(
                    "concept_trial_qc",
                    self.lbl_concept_trial_qc_placeholder,
                    "No trial QC metrics available for this decoder run.",
                )
                return

            plot_df = qc_df.copy().reset_index(drop=True)
            metric_col = "qc_rms_median" if "qc_rms_median" in plot_df.columns else "qc_max_abs"
            plot_df[metric_col] = pd.to_numeric(plot_df.get(metric_col, np.nan), errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                plot_df["_qc_metric_log10"] = np.log10(np.maximum(np.abs(plot_df[metric_col].to_numpy(dtype=float)), 1e-30))
            plot_df["qc_status"] = plot_df.get("qc_status", "").astype(str)
            plot_df["qc_reason"] = plot_df.get("qc_reason", "").astype(str)
            plot_df["qc_outlier"] = plot_df.get("qc_outlier", False).fillna(False).astype(bool)

            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), gridspec_kw={"width_ratios": [2.1, 1.0]})
            ax = axes[0]
            kept = plot_df["qc_status"].eq("kept") & ~plot_df["qc_outlier"]
            outliers = plot_df["qc_outlier"]
            dropped = plot_df["qc_status"].eq("dropped") & ~outliers
            x = np.arange(len(plot_df), dtype=int)
            if np.any(kept):
                ax.scatter(x[kept], plot_df.loc[kept, "_qc_metric_log10"], s=18, c="#2563eb", alpha=0.78, label="kept")
            if np.any(dropped):
                ax.scatter(x[dropped], plot_df.loc[dropped, "_qc_metric_log10"], s=20, c="#6b7280", alpha=0.62, label="dropped")
            if np.any(outliers):
                ax.scatter(
                    x[outliers],
                    plot_df.loc[outliers, "_qc_metric_log10"],
                    s=32,
                    c="#dc2626",
                    alpha=0.9,
                    label="outlier",
                )
            ax.set_title("Trial QC metric by trial")
            ax.set_xlabel("Decoded trial order")
            ax.set_ylabel(f"log10(abs({metric_col}))")
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize=8)

            reason_counts = (
                plot_df.assign(
                    reason=plot_df["qc_reason"].replace("", "kept"),
                )
                .groupby("reason", sort=False)
                .size()
                .sort_values(ascending=True)
                .tail(12)
            )
            axes[1].barh(np.arange(len(reason_counts)), reason_counts.to_numpy(), color="#475569", alpha=0.82)
            axes[1].set_yticks(np.arange(len(reason_counts)))
            axes[1].set_yticklabels(reason_counts.index.astype(str).tolist(), fontsize=8)
            axes[1].set_xlabel("Trials")
            axes[1].set_title("QC outcomes")
            axes[1].grid(axis="x", alpha=0.2)

            fig.suptitle(self._concept_result_title(result), fontsize=10)
            fig.tight_layout()
            self._set_plot(
                "concept_trial_qc",
                self.concept_trial_qc_plot_layout,
                self.lbl_concept_trial_qc_placeholder,
                fig,
            )
        except Exception as exc:
            self._reset_plot_placeholder(
                "concept_trial_qc",
                self.lbl_concept_trial_qc_placeholder,
                f"Could not render trial QC plot: {exc}",
            )

    def _render_concept_decoding(self):
        result = self._concept_decode_result if isinstance(self._concept_decode_result, dict) else None
        if not result:
            if getattr(self, "btn_concept_save_outputs", None) is not None:
                self.btn_concept_save_outputs.setEnabled(False)
            status = self._concept_decode_status_text or "model not trained."
            self.lbl_concept_status.setText(f"Concept decoding: {status}")
            self._reset_plot_placeholder(
                "concept_confusion",
                self.lbl_concept_confusion_placeholder,
                "Concept decoding results not available. Press Train / test decoder.",
            )
            self._reset_plot_placeholder(
                "concept_accuracy",
                self.lbl_concept_accuracy_placeholder,
                "Concept decoding results not available. Press Train / test decoder.",
            )
            self._reset_plot_placeholder(
                "concept_templates",
                self.lbl_concept_templates_placeholder,
                "Use Template correlation to show learned concept templates.",
            )
            self._reset_plot_placeholder(
                "concept_trial_qc",
                self.lbl_concept_trial_qc_placeholder,
                "Concept decoding results not available. Press Train / test decoder.",
            )
            self._set_dataframe_table(self.tbl_concept_summary, "concept_summary_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_elimination_path, "concept_elimination_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_metrics, "concept_metrics_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_predictions, "concept_predictions_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_trial_qc, "concept_trial_qc_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_features, "concept_features_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_split, "concept_split_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_permutation, "concept_permutation_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_templates, "concept_templates_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_architecture, "concept_architecture_model", pd.DataFrame())
            self._set_dataframe_table(self.tbl_concept_qc_summary, "concept_qc_summary_model", pd.DataFrame())
            self.lbl_concept_trial_qc_summary.setText("Trial QC counts unavailable.")
            self.lbl_concept_qc_summary.setText("Trial QC counts unavailable.")
            return

        if getattr(self, "btn_concept_save_outputs", None) is not None:
            self.btn_concept_save_outputs.setEnabled(not bool(result.get("semantic_subset_output_dir")))
        summary_df = result.get("subset_comparison_df", result.get("summary_df", pd.DataFrame()))
        metrics_df = result.get("metrics_df", pd.DataFrame())
        predictions_df = result.get("predictions_df", pd.DataFrame())
        feature_df = result.get("feature_df", pd.DataFrame())
        split_df = result.get("split_df", pd.DataFrame())
        permutation_df = result.get("permutation_df", pd.DataFrame())
        architecture_df = result.get("model_architecture_df", pd.DataFrame())
        per_class_df = result.get("per_class_accuracy_df", pd.DataFrame())
        template_df = result.get("template_df", pd.DataFrame())
        trial_qc_df = result.get("trial_qc_df", pd.DataFrame())
        qc_summary_df = result.get("qc_summary_df", pd.DataFrame())
        if isinstance(trial_qc_df, pd.DataFrame) and not trial_qc_df.empty:
            # Rebuild for saved results that predate the totals columns.
            current_summary = concept_qc_summary_table(trial_qc_df, feature_payload=result)
            totals = current_summary.loc[current_summary["qc_status"].eq("total")]
            if isinstance(qc_summary_df, pd.DataFrame) and not qc_summary_df.empty and "qc_status" in qc_summary_df:
                details = qc_summary_df.loc[~qc_summary_df["qc_status"].eq("total")]
                qc_summary_df = pd.concat([totals, details], ignore_index=True)
            else:
                qc_summary_df = current_summary
            total_row = totals.iloc[0]
            qc_text = (
                f"Discarded {int(total_row['discarded_trials']):,} of "
                f"{int(total_row['n_trials']):,} trials ({total_row['discarded_pct']:.1f}%). "
                f"Kept: {int(total_row['kept_trials']):,}. "
                "Percentage is based on all trials recorded in QC."
            )
        else:
            qc_text = "Trial QC counts unavailable."
        self.lbl_concept_trial_qc_summary.setText(qc_text)
        self.lbl_concept_qc_summary.setText(qc_text)
        class_names = [str(v) for v in result.get("class_names", [])]

        self._render_concept_confusion_plot(result, class_names)
        self._render_concept_accuracy_plot(result, per_class_df)
        self._render_concept_templates_plot(result)
        self._render_concept_trial_qc_plot(result)

        status = self._concept_decode_status_text or "results ready"
        self.lbl_concept_status.setText(f"Concept decoding: {status}")
        self._set_dataframe_table(self.tbl_concept_summary, "concept_summary_model", summary_df)
        self._set_dataframe_table(self.tbl_concept_elimination_path, "concept_elimination_model", result.get("elimination_path_df", pd.DataFrame()))
        self._set_dataframe_table(self.tbl_concept_metrics, "concept_metrics_model", metrics_df)
        self._set_dataframe_table(self.tbl_concept_predictions, "concept_predictions_model", predictions_df)
        self._set_dataframe_table(self.tbl_concept_trial_qc, "concept_trial_qc_model", trial_qc_df)
        self._set_dataframe_table(self.tbl_concept_features, "concept_features_model", feature_df)
        self._set_dataframe_table(self.tbl_concept_split, "concept_split_model", split_df)
        self._set_dataframe_table(self.tbl_concept_permutation, "concept_permutation_model", permutation_df)
        self._set_dataframe_table(self.tbl_concept_templates, "concept_templates_model", template_df)
        self._set_dataframe_table(self.tbl_concept_architecture, "concept_architecture_model", architecture_df)
        self._set_dataframe_table(self.tbl_concept_qc_summary, "concept_qc_summary_model", qc_summary_df)

    def _compute_semantic_preview(self):
        if not callable(self._semantic_preview_callback):
            self._semantic_preview_status_text = "Semantic preview callback unavailable."
            self._semantic_preview_result = None
            self._render_semantic_decoding()
            return
        tmin = float(self.spb_semantic_tmin.value())
        tmax = float(self.spb_semantic_tmax.value())
        if tmax <= tmin:
            self._semantic_preview_status_text = "Semantic preview requires window end > window start."
            self._semantic_preview_result = None
            self._render_semantic_decoding()
            return
        if subset_batch:
            remaining = batch_total - int(resume_manifest.get("attempted", 0)) if resume_dir else batch_total
            QtWidgets.QMessageBox.warning(
                self, "Parcel screening workload",
                f"{len(batch_parcels)} selected parcels produce {batch_total:,} subset models.\n"
                f"{remaining:,} models remain to run.\n"
                "Large searches can take substantial time and disk space."
            )

        try:
            snr = max(1e-6, float(self.spb_source_snr.value()))
            options = {
                "window_start": tmin,
                "window_end": tmax,
                "group_mode": self._semantic_group_mode(),
                "method": str(self.cmb_source_method.currentText() or "MNE"),
                "snr": snr,
                "lambda2": 1.0 / (snr ** 2),
                "inverse_operator_path": str(self.le_source_inverse.text() or "").strip(),
                "selected_parcellations": self._semantic_selected_parcellations(),
                "stimulus_name": str(self.cmb_semantic_stimulus.currentText() or "All stimuli"),
                "trial_epoch_index": self.cmb_semantic_trial.currentData(),
            }
            result = self._semantic_preview_callback(options)
            if isinstance(result, tuple) and len(result) >= 2:
                self._semantic_preview_result, self._semantic_preview_status_text = result[0], str(result[1])
            else:
                self._semantic_preview_result, self._semantic_preview_status_text = None, "Invalid semantic preview result."
        except Exception as exc:
            self._semantic_preview_result = None
            self._semantic_preview_status_text = f"Semantic preview failed: {exc}"
        self._render_semantic_decoding()

    def _run_semantic_decoding(self):
        if not callable(self._semantic_decode_callback):
            self._semantic_train_status_text = "Semantic training callback unavailable."
            self._semantic_train_result = None
            self._render_semantic_decoding()
            return
        tmin = float(self.spb_semantic_tmin.value())
        tmax = float(self.spb_semantic_tmax.value())
        if tmax <= tmin:
            self._semantic_train_status_text = "Semantic decoding requires window end > window start."
            self._semantic_train_result = None
            self._render_semantic_decoding()
            return
        if subset_batch:
            remaining = batch_total - int(resume_manifest.get("attempted", 0)) if resume_dir else batch_total
            QtWidgets.QMessageBox.warning(
                self, "Parcel screening workload",
                f"{len(batch_parcels)} selected parcels produce {batch_total:,} subset models.\n"
                f"{remaining:,} models remain to run.\n"
                "Large searches can take substantial time and disk space."
            )

        try:
            snr = max(1e-6, float(self.spb_source_snr.value()))
            options = {
                "window_start": tmin,
                "window_end": tmax,
                "group_mode": self._semantic_group_mode(),
                "min_trials_per_class": int(self.spb_semantic_min_trials.value()),
                "method": str(self.cmb_source_method.currentText() or "MNE"),
                "snr": snr,
                "lambda2": 1.0 / (snr ** 2),
                "inverse_operator_path": str(self.le_source_inverse.text() or "").strip(),
                "selected_parcellations": self._semantic_selected_parcellations(),
            }
            result = self._semantic_decode_callback(options)
            if isinstance(result, tuple) and len(result) >= 2:
                self._semantic_train_result, self._semantic_train_status_text = result[0], str(result[1])
            else:
                self._semantic_train_result, self._semantic_train_status_text = None, "Invalid semantic decoding result."
        except Exception as exc:
            self._semantic_train_result = None
            self._semantic_train_status_text = f"Semantic decoding failed: {exc}"
        self._render_semantic_decoding()

    def _on_semantic_preview_stimulus_changed(self, *_args):
        self._refresh_semantic_trial_selector()
        if self._semantic_preview_result:
            self._render_semantic_decoding()

    def _render_semantic_decoding(self):
        preview = self._semantic_preview_result if isinstance(self._semantic_preview_result, dict) else None
        train_result = self._semantic_train_result if isinstance(self._semantic_train_result, dict) else None
        if not preview:
            preview_status = self._semantic_preview_status_text or "no preview computed."
            train_status = self._semantic_train_status_text or "model not trained."
            self.lbl_semantic_status.setText(f"Semantic decoding: preview={preview_status} | train={train_status}")
            self._reset_plot_placeholder(
                "semantic",
                self.lbl_semantic_placeholder,
                "Semantic preview not available. Press Compute / Refresh.",
            )
        else:
            allowed_labels = [str(v) for v in preview.get("available_stimulus_groups", [])]
            self._refresh_semantic_stimulus_selector(allowed_labels=allowed_labels)
            selected_group = str(self.cmb_semantic_stimulus.currentText() or "")
            if selected_group == "All stimuli" and allowed_labels:
                selected_group = allowed_labels[0]

            try:
                import matplotlib.pyplot as plt

                times = np.asarray(preview.get("times", []), dtype=float)
                selected_timecourse = np.asarray(preview.get("selected_timecourse", []), dtype=float)
                whole_brain_timecourse = np.asarray(preview.get("whole_brain_timecourse", []), dtype=float)
                window_stc = preview.get("window_stc")
                window_start = float(preview.get("window_start", self.spb_semantic_tmin.value()))
                window_end = float(preview.get("window_end", self.spb_semantic_tmax.value()))
                selected_parcellations = [str(v) for v in preview.get("selected_parcellations", [])]
                preview_label = str(preview.get("stimulus_name", selected_group) or selected_group)

                fig = plt.figure(figsize=(10.8, 8.0))
                subjects_dir_for_plot = self._source_subjects_dir or preview.get("subjects_dir") or None
                subject_for_plot = self._source_subject or preview.get("subject") or None
                brain_image = None
                brain_image_side = None
                primary_views = []
                secondary_views = []
                if window_stc is not None and subjects_dir_for_plot:
                    initial_time = float(window_stc.times[0]) if getattr(window_stc, "times", None) is not None else 0.0
                    brain_image, brain_image_side, primary_views, secondary_views = self._render_stc_brain_screenshots(
                        window_stc,
                        initial_time,
                        subject_for_plot,
                        subjects_dir_for_plot,
                    )

                if brain_image is not None:
                    if brain_image_side is not None:
                        grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.2], width_ratios=[3.4, 1.4], hspace=0.15, wspace=0.06)
                        ax_img = fig.add_subplot(grid[0, 0])
                        ax_img_side = fig.add_subplot(grid[0, 1])
                        ax_img_side.imshow(brain_image_side)
                        ax_img_side.set_axis_off()
                        if secondary_views:
                            ax_img_side.set_title(
                                f"Supplemental views: {secondary_views}",
                                fontsize=10,
                            )
                        ax_tc = fig.add_subplot(grid[1, :])
                    else:
                        grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.15)
                        ax_img = fig.add_subplot(grid[0])
                        ax_tc = fig.add_subplot(grid[1])
                    ax_img.imshow(brain_image)
                    ax_img.set_axis_off()
                    ax_img.set_title(
                        f"{preview_label}: window-averaged source activity ({window_start:.3f}-{window_end:.3f}s, views={primary_views})",
                        fontsize=11,
                    )
                else:
                    ax_tc = fig.add_subplot(1, 1, 1)

                if (
                    (selected_timecourse.ndim == 2 and selected_timecourse.size > 0 and times.size == selected_timecourse.shape[1])
                    or (whole_brain_timecourse.ndim == 1 and whole_brain_timecourse.size == times.size)
                ):
                    ax_tc.axvspan(window_start, window_end, color="#d95f02", alpha=0.18, label="Window")
                    if whole_brain_timecourse.ndim == 1 and whole_brain_timecourse.size == times.size:
                        ax_tc.plot(times, whole_brain_timecourse, linewidth=2.4, color="#111111", label="Whole brain mean |source|")
                    if selected_timecourse.ndim == 2 and selected_timecourse.size > 0 and times.size == selected_timecourse.shape[1]:
                        overall_tc = np.nanmean(selected_timecourse, axis=0)
                        for roi_idx in range(selected_timecourse.shape[0]):
                            ax_tc.plot(times, selected_timecourse[roi_idx], linewidth=1.0, alpha=0.28, color="#6baed6")
                        ax_tc.plot(
                            times,
                            overall_tc,
                            linewidth=2.1,
                            color="#08519c",
                            label=f"Selected parcellations mean (n={selected_timecourse.shape[0]})",
                        )
                    ax_tc.axvline(0.0, color="k", linewidth=0.9, alpha=0.6)
                    ax_tc.set_xlabel("Time (s)")
                    ax_tc.set_ylabel("Mean |source|")
                    if brain_image is None:
                        ax_tc.set_title(
                            f"{preview_label}: window-averaged source activity ({window_start:.3f}-{window_end:.3f}s)"
                        )
                    ax_tc.grid(alpha=0.25)
                    ax_tc.legend(loc="best", fontsize=8)
                else:
                    ax_tc.text(0.5, 0.5, "No preview time course available", ha="center", va="center")
                    ax_tc.set_axis_off()

                fig.tight_layout()
                self._set_plot("semantic", self.semantic_plot_layout, self.lbl_semantic_placeholder, fig)
            except Exception as exc:
                self._reset_plot_placeholder(
                    "semantic",
                    self.lbl_semantic_placeholder,
                    f"Could not render semantic preview: {exc}",
                )

            preview_status = self._semantic_preview_status_text or "preview ready"
            train_status = self._semantic_train_status_text or "model not trained."
            parcel_text = len(selected_parcellations) if selected_parcellations else "all"
            self.lbl_semantic_status.setText(
                f"Semantic decoding: preview={preview_status} | train={train_status} | preview={preview_label}, parcels={parcel_text}"
            )

        trial_feature_df = train_result.get("trial_feature_df", pd.DataFrame()) if train_result else pd.DataFrame()
        metrics_df = train_result.get("metrics_df", pd.DataFrame()) if train_result else pd.DataFrame()
        feature_df = train_result.get("feature_df", pd.DataFrame()) if train_result else pd.DataFrame()
        if isinstance(trial_feature_df, pd.DataFrame) and not trial_feature_df.empty:
            self.semantic_trials_model = PandasModel(trial_feature_df.copy())
            self.tbl_semantic_trials.setModel(self.semantic_trials_model)
            self.tbl_semantic_trials.resizeColumnsToContents()
        else:
            self.tbl_semantic_trials.setModel(None)

        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            self.semantic_metrics_model = PandasModel(metrics_df.copy())
            self.tbl_semantic_metrics.setModel(self.semantic_metrics_model)
            self.tbl_semantic_metrics.resizeColumnsToContents()
        else:
            self.tbl_semantic_metrics.setModel(None)

        if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
            self.semantic_features_model = PandasModel(feature_df.copy())
            self.tbl_semantic_features.setModel(self.semantic_features_model)
            self.tbl_semantic_features.resizeColumnsToContents()
        else:
            self.tbl_semantic_features.setModel(None)
