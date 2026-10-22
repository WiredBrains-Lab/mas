"""MEG/FIF integration helpers for the 3D viewer.

This module isolates MEG sensor visualization, legend UI, and source
projection logic from the main MEG3DViewer class to keep meg_3d.py manageable.
"""

from pathlib import Path

import mne
import numpy as np
from matplotlib import colormaps
from mne.minimum_norm import apply_inverse, make_inverse_operator
from qtpy import QtCore, QtGui, QtWidgets

from core.meg_channels import format_bad_meg_channel_message, good_meg_picks
from core.source_parcellation import (
    HCP_MMP_PARCS,
    aseg_name_from_path,
    is_surface_stc,
    is_volume_stc,
    normalize_parcellation_key,
    read_surface_labels,
    subject_aseg_path,
    surface_label_time_courses,
    top_label_indices,
    volume_label_time_courses,
)


class MEGIntegrationMixin:
    """Mixin that adds MEG/FIF visualization and source projection support."""

    def _init_meg_state(self):
        """Initialize MEG/FIF-related state variables."""
        self._meg_sensor_info = None
        self._meg_sensor_trans = None
        self._meg_trans = None
        self._meg_current_fif_path = None
        self._meg_current_fif_identity = None
        self._meg_loaded_fif_paths = []
        self._meg_loaded_trans_path = None
        self._meg_loaded_trans_file_identity = None
        self._meg_trans_fif_identity = None
        self._meg_auto_coreg_attempt_key = None
        self._meg_auto_coreg_error = None
        self._meg_activity_scores = {}
        self._meg_show_helmet = True
        self._meg_helmet_actor = None
        self._meg_activity_unit = "a.u."
        self._meg_activity_value_range = None
        self.meg_activity_legend_widget = None
        self.meg_activity_legend_title_label = None
        self.meg_activity_legend_bar_label = None
        self.meg_activity_legend_min_label = None
        self.meg_activity_legend_max_label = None
        self.meg_source_project_btn = None
        self.meg_source_auto_checkbox = None
        self.meg_source_space_combo = None
        self.meg_source_clear_btn = None
        self.meg_show_helmet_checkbox = None
        self.meg_show_alignment_checkbox = None
        self.meg_source_time_label = None
        self.meg_source_top_label_btn = None
        self.meg_source_top_label_count_spin = None
        self.meg_source_top_label_parc_combo = None
        self.meg_volume_plot_mode_combo = None
        self._meg_source_auto = False
        self._meg_source_space_kind = "surface"
        self._meg_volume_source_pos_mm = 7.0
        self._meg_volume_stc_mode = "stat_map"
        self._meg_source_window_data_si = None
        self._meg_source_window_ch_names = []
        self._meg_source_window_sfreq = None
        self._meg_source_window_start_sample = None
        self._meg_source_window_tmin = 0.0
        self._meg_source_subject_key = None
        self._meg_source_src = None
        self._meg_source_bem = None
        self._meg_noise_cov = None
        self._meg_noise_cov_metadata = {}
        self._meg_noise_cov_version = 0
        self._meg_inverse_key = None
        self._meg_inverse_operator = None
        self._meg_last_stc = None
        self._meg_last_stc_src = None
        self._meg_last_stc_subject = None
        self._meg_last_stc_subjects_dir = None
        self._meg_last_stc_time_sec = None
        self._meg_volume_stc_plot = None
        self._meg_top_label_figures = []
        self._meg_stc_morph_cache = {}
        self._meg_estimated_trans = None
        self._meg_estimated_trans_key = None
        self._meg_alignment_fig = None
        self._meg_alignment_widget = None
        self._meg_alignment_plotter = None
        self._meg_alignment_active = False
        self.meg_coreg_fif_combo = None

    def _register_loaded_meg_fif(self, fif_path):
        """Track loaded FIFs for MEG coregistration review."""
        if not fif_path:
            return
        try:
            path = str(Path(fif_path).expanduser().resolve())
        except Exception:
            path = str(fif_path)
        if not path:
            return
        loaded = list(getattr(self, "_meg_loaded_fif_paths", []) or [])
        if path not in loaded:
            loaded.append(path)
        self._meg_loaded_fif_paths = loaded
        if hasattr(self, "_refresh_meg_coreg_fif_combo"):
            try:
                self._refresh_meg_coreg_fif_combo(selected_path=path)
            except Exception:
                pass

    @staticmethod
    def _fif_file_identity(fif_path):
        """Return a stable identity tuple for a FIF path."""
        path_str = str(fif_path or "").strip()
        if not path_str:
            return None
        path = Path(path_str).expanduser().resolve()
        try:
            stat = path.stat()
            return (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        except Exception:
            return (str(path), None, None)

    @staticmethod
    def _matrix_signature(matrix, decimals=12):
        """Return a hashable numeric signature for an affine-like matrix."""
        try:
            arr = np.asarray(matrix, dtype=float)
        except Exception:
            return None
        if arr.size == 0:
            return None
        return tuple(np.round(arr.reshape(-1), int(decimals)).tolist())

    def _transform_signature(self, trans):
        """Return a hashable signature for an MNE transform or transform token."""
        if trans is None:
            return None
        if isinstance(trans, str):
            return ("token", trans)
        try:
            matrix = trans["trans"]
        except Exception:
            matrix = getattr(trans, "trans", None)
        return self._matrix_signature(matrix)

    def _info_dev_head_signature(self, info):
        """Return a signature for the run-specific device->head transform."""
        if info is None:
            return None
        try:
            dev_head_t = info.get("dev_head_t")
        except Exception:
            dev_head_t = None
        return self._transform_signature(dev_head_t)

    def _get_meg_experiment_root(self, fif_path=None):
        """Return the experiment root one level above the FIF folder."""
        path_str = str(fif_path or getattr(self, "_meg_current_fif_path", "") or "").strip()
        if not path_str:
            return None
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            return None
        try:
            return path.parent.parent
        except Exception:
            return None

    def _default_meg_trans_output_path(self, fif_path=None, subject_id=None):
        """Return the preferred `-trans.fif` path under `../trans`."""
        root = self._get_meg_experiment_root(fif_path=fif_path)
        path_str = str(fif_path or getattr(self, "_meg_current_fif_path", "") or "").strip()
        if root is None or not path_str:
            return None
        if subject_id is None:
            try:
                subject_id, _ = self._get_brain_subject_info()
            except Exception:
                subject_id = None
        stem = Path(path_str).stem
        stem = stem.replace("_raw", "").replace("-raw", "")
        if stem.endswith("-trans"):
            out_name = f"{stem}.fif"
        elif subject_id and str(subject_id).strip() and str(subject_id).lower() != "fsaverage":
            out_name = f"{stem}-{subject_id}-trans.fif"
        else:
            out_name = f"{stem}-trans.fif"
        return root / "trans" / out_name

    def _find_existing_meg_trans_file(self, fif_path=None, subject_id=None):
        """Find a nearby `-trans.fif` for the current FIF and subject."""
        path_str = str(fif_path or getattr(self, "_meg_current_fif_path", "") or "").strip()
        if not path_str:
            return None
        fif_path_obj = Path(path_str).expanduser().resolve()
        if not fif_path_obj.exists():
            return None

        if subject_id is None:
            try:
                subject_id, _ = self._get_brain_subject_info()
            except Exception:
                subject_id = None

        parent_dir = fif_path_obj.parent.parent
        trans_dir = parent_dir / "trans"
        anatomy_dir = parent_dir / "Anatomy"
        default_path = self._default_meg_trans_output_path(fif_path=fif_path_obj, subject_id=subject_id)

        candidates = []
        if default_path is not None and default_path.exists():
            candidates.append(default_path)

        fif_stem = fif_path_obj.stem.lower()
        if fif_stem.endswith("_raw"):
            fif_stem = fif_stem[:-4]
        if fif_stem.endswith("-raw"):
            fif_stem = fif_stem[:-4]

        search_dirs = [trans_dir, fif_path_obj.parent, parent_dir, anatomy_dir]
        seen = set()
        for directory in search_dirs:
            if directory is None or not directory.exists() or not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*-trans.fif")):
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(resolved)

        def _is_exact_default(path_obj):
            return default_path is not None and path_obj == default_path

        def _name_matches_run_stem(name):
            base = name
            if base.endswith("-trans.fif"):
                base = base[: -len("-trans.fif")]
            if base == fif_stem:
                return True
            for left in ("-", "_"):
                if base.startswith(f"{fif_stem}{left}") or base.endswith(f"{left}{fif_stem}"):
                    return True
                for right in ("-", "_"):
                    if f"{left}{fif_stem}{right}" in base:
                        return True
            return False

        def _is_run_specific(path_obj):
            name = path_obj.name.lower()
            if not fif_stem:
                return False
            return _name_matches_run_stem(name)

        run_candidates = [
            path_obj for path_obj in candidates
            if _is_exact_default(path_obj) or _is_run_specific(path_obj)
        ]
        if not run_candidates:
            return None

        def _rank(path_obj):
            name = path_obj.name.lower()
            exact = 0 if default_path is not None and path_obj == default_path else 1
            stem_match = 0 if fif_stem and _name_matches_run_stem(name) else 1
            subject_match = 0 if subject_id and str(subject_id).lower() in name else 1
            trans_dir_match = 0 if path_obj.parent == trans_dir else 1
            return (exact, stem_match, subject_match, trans_dir_match, name)

        return sorted(run_candidates, key=_rank)[0]

    def _load_meg_trans_file(self, trans_path, fif_path=None):
        """Load and cache a MEG transform file."""
        trans = mne.read_trans(str(trans_path))
        self._meg_trans = trans
        self._meg_loaded_trans_path = str(Path(trans_path).expanduser().resolve())
        self._meg_loaded_trans_file_identity = self._fif_file_identity(self._meg_loaded_trans_path)
        self._meg_trans_fif_identity = (
            self._fif_file_identity(
                fif_path or getattr(self, "_meg_current_fif_path", None)
            )
            or getattr(self, "_meg_current_fif_identity", None)
        )
        self._meg_inverse_key = None
        self._meg_inverse_operator = None
        return trans

    def _cache_meg_trans(self, trans, trans_path=None, fif_path=None):
        """Cache a resolved MEG transform in memory."""
        self._meg_trans = trans
        if trans_path:
            self._meg_loaded_trans_path = str(Path(trans_path).expanduser().resolve())
            self._meg_loaded_trans_file_identity = self._fif_file_identity(self._meg_loaded_trans_path)
        self._meg_trans_fif_identity = (
            self._fif_file_identity(
                fif_path or getattr(self, "_meg_current_fif_path", None)
            )
            or getattr(self, "_meg_current_fif_identity", None)
        )
        self._meg_inverse_key = None
        self._meg_inverse_operator = None
        return trans

    def _run_automated_meg_coregistration(self, info=None, fif_path=None):
        """Run MNE's scripted Coregistration pipeline and save `-trans.fif`."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        if str(subject_id).lower() == "fsaverage":
            raise RuntimeError("Automated coregistration requires a patient subject, not fsaverage.")

        if info is None:
            info = self._meg_sensor_info
        if info is None:
            raise RuntimeError("No MEG sensor info is available for automated coregistration.")
        if not info.get("dig"):
            raise RuntimeError("No head-shape digitization points are available in the FIF file.")

        coreg = mne.coreg.Coregistration(
            info,
            subject_id,
            subjects_dir,
            fiducials="estimated",
            on_defects="warn",
        )
        coreg.fit_fiducials(verbose="error")
        coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose="error")
        try:
            coreg.omit_head_shape_points(distance=5.0 / 1000.0)
        except Exception:
            pass
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose="error")
        dists_mm = np.asarray(coreg.compute_dig_mri_distances(), dtype=float) * 1e3

        save_path = self._default_meg_trans_output_path(fif_path=fif_path, subject_id=subject_id)
        if save_path is None:
            raise RuntimeError("Could not resolve output path for automated coregistration.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mne.write_trans(str(save_path), coreg.trans, overwrite=True)

        dist_msg = "distance unavailable"
        finite_dists = dists_mm[np.isfinite(dists_mm)]
        if finite_dists.size > 0:
            dist_msg = (
                f"HSP->MRI mean/min/max="
                f"{np.mean(finite_dists):.2f}/{np.min(finite_dists):.2f}/{np.max(finite_dists):.2f} mm"
            )

        self._cache_meg_trans(coreg.trans, trans_path=save_path, fif_path=fif_path)
        self._meg_auto_coreg_error = None
        return coreg.trans, save_path, dist_msg

    def _build_meg_legend_pixmap(self, width=180, height=12):
        """Create a horizontal viridis gradient pixmap for MEG activity legend."""
        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        cmap = colormaps.get_cmap("viridis")
        denom = max(1, width - 1)
        for x in range(width):
            rgba = cmap(x / denom)
            color = QtGui.QColor(
                int(max(0, min(255, round(rgba[0] * 255)))),
                int(max(0, min(255, round(rgba[1] * 255)))),
                int(max(0, min(255, round(rgba[2] * 255)))),
                255,
            )
            painter.setPen(color)
            painter.drawLine(x, 0, x, height)
        painter.end()
        return pixmap

    def _init_meg_activity_legend(self, parent_layout):
        """Attach MEG activity legend controls to the viewer toolbar area."""
        if self.meg_activity_legend_widget is not None:
            return

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)

        title = QtWidgets.QLabel("MEG Activity (RMS)")
        title.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        container_layout.addWidget(title)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        min_label = QtWidgets.QLabel("--")
        min_label.setStyleSheet("QLabel { font-size: 10px; color: #555555; min-width: 56px; }")
        row.addWidget(min_label)

        bar_label = QtWidgets.QLabel()
        bar_label.setFixedSize(180, 12)
        bar_label.setPixmap(self._build_meg_legend_pixmap(width=180, height=12))
        bar_label.setToolTip("Sensor activity color scale (low to high)")
        row.addWidget(bar_label)

        max_label = QtWidgets.QLabel("--")
        max_label.setStyleSheet("QLabel { font-size: 10px; color: #555555; min-width: 56px; }")
        row.addWidget(max_label)

        row.addStretch()
        container_layout.addLayout(row)
        container.setLayout(container_layout)

        self.meg_activity_legend_widget = container
        self.meg_activity_legend_title_label = title
        self.meg_activity_legend_bar_label = bar_label
        self.meg_activity_legend_min_label = min_label
        self.meg_activity_legend_max_label = max_label
        self.meg_activity_legend_widget.setVisible(False)

        parent_layout.addWidget(container)

    def _format_meg_activity_value(self, value):
        """Format MEG activity values compactly for legend labels."""
        try:
            value = float(value)
        except Exception:
            return "--"
        abs_val = abs(value)
        if abs_val >= 1000 or (0 < abs_val < 0.01):
            return f"{value:.2e}"
        return f"{value:.3f}"

    def _update_meg_activity_legend(self, activity_scores=None, unit_label=None):
        """Update legend range labels to match current MEG sensor coloring."""
        if isinstance(unit_label, str) and unit_label.strip():
            self._meg_activity_unit = unit_label.strip()

        if self.meg_activity_legend_widget is None:
            return

        if not isinstance(activity_scores, dict) or not activity_scores:
            self._meg_activity_value_range = None
            self.meg_activity_legend_min_label.setText("--")
            self.meg_activity_legend_max_label.setText("--")
            self.meg_activity_legend_widget.setVisible(False)
            return

        vals = []
        for val in activity_scores.values():
            try:
                fval = float(val)
            except Exception:
                continue
            if np.isfinite(fval):
                vals.append(fval)

        if not vals:
            self._meg_activity_value_range = None
            self.meg_activity_legend_min_label.setText("--")
            self.meg_activity_legend_max_label.setText("--")
            self.meg_activity_legend_widget.setVisible(False)
            return

        vmin = min(vals)
        vmax = max(vals)
        self._meg_activity_value_range = (vmin, vmax)
        self.meg_activity_legend_title_label.setText(
            f"MEG Activity (RMS, {self._meg_activity_unit})"
        )
        self.meg_activity_legend_min_label.setText(self._format_meg_activity_value(vmin))
        self.meg_activity_legend_max_label.setText(self._format_meg_activity_value(vmax))
        self.meg_activity_legend_widget.setVisible(True)

    def _init_meg_source_controls(self, parent_layout):
        """Add source-projection controls (manual/auto/clear)."""
        controls = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        btn_project = QtWidgets.QPushButton("Project Source (MNE)")
        btn_project.setToolTip("Run MNE inverse modeling for the current FIF window")
        btn_project.clicked.connect(self.project_cached_meg_window_to_source)
        row.addWidget(btn_project)

        source_space_combo = QtWidgets.QComboBox()
        source_space_combo.addItem("Surface", "surface")
        source_space_combo.addItem("Volume", "volume")
        source_space_combo.setToolTip("Choose cortical surface or volume source space for newly computed inverse estimates")
        source_space_combo.currentIndexChanged.connect(self._on_meg_source_space_changed)
        row.addWidget(source_space_combo)

        auto_chk = QtWidgets.QCheckBox("Auto")
        auto_chk.setToolTip("Automatically update source projection when window changes")
        auto_chk.stateChanged.connect(self._on_meg_source_auto_toggled)
        row.addWidget(auto_chk)

        helmet_chk = QtWidgets.QCheckBox("Helmet")
        helmet_chk.setChecked(bool(self._meg_show_helmet))
        helmet_chk.setToolTip("Show or hide the MEG helmet shell")
        helmet_chk.stateChanged.connect(self._on_meg_show_helmet_toggled)
        row.addWidget(helmet_chk)

        alignment_chk = QtWidgets.QCheckBox("Alignment")
        alignment_chk.setChecked(bool(getattr(self, "_meg_alignment_active", False)))
        alignment_chk.setToolTip("Show or hide the MEG alignment visualization")
        alignment_chk.stateChanged.connect(self._on_meg_show_alignment_toggled)
        row.addWidget(alignment_chk)

        btn_clear = QtWidgets.QPushButton("Clear Source")
        btn_clear.setToolTip("Remove projected cortical source map")
        btn_clear.clicked.connect(self.clear_meg_source_projection)
        row.addWidget(btn_clear)

        row.addStretch()
        container_layout.addLayout(row)

        label_row = QtWidgets.QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(6)

        label_row.addWidget(QtWidgets.QLabel("Top labels:"))
        top_count = QtWidgets.QSpinBox()
        top_count.setRange(1, 24)
        top_count.setValue(7)
        top_count.setToolTip("Number of highest-activation labels to plot")
        label_row.addWidget(top_count)

        top_parc = QtWidgets.QComboBox()
        top_parc.addItem("Auto atlas", "auto")
        top_parc.addItem("aparc", "aparc")
        top_parc.addItem("aparc+aseg", "aparc+aseg")
        top_parc.addItem("HCP-MMP", "HCPMMP1")
        top_parc.addItem("HCP-MMP combined", "HCPMMP1_combined")
        top_parc.setToolTip("Atlas used for top source-label time courses")
        label_row.addWidget(top_parc)

        volume_mode = QtWidgets.QComboBox()
        volume_mode.addItem("Volume slices", "stat_map")
        volume_mode.addItem("Glass brain", "glass_brain")
        volume_mode.setToolTip("Display mode for volume source estimates")
        volume_mode.currentIndexChanged.connect(self._on_meg_volume_plot_mode_changed)
        label_row.addWidget(volume_mode)

        btn_top = QtWidgets.QPushButton("Plot")
        btn_top.setToolTip("Plot the labels with the strongest source activation time courses")
        btn_top.clicked.connect(self.plot_top_meg_source_labels)
        label_row.addWidget(btn_top)
        label_row.addStretch()
        container_layout.addLayout(label_row)

        time_label = QtWidgets.QLabel("Projected timepoint: --")
        time_label.setStyleSheet("QLabel { font-size: 10px; color: #555555; }")
        time_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        container_layout.addWidget(time_label)

        controls.setLayout(container_layout)
        parent_layout.addWidget(controls)

        self.meg_source_project_btn = btn_project
        self.meg_source_auto_checkbox = auto_chk
        self.meg_source_space_combo = source_space_combo
        self.meg_show_helmet_checkbox = helmet_chk
        self.meg_show_alignment_checkbox = alignment_chk
        self.meg_source_clear_btn = btn_clear
        self.meg_source_time_label = time_label
        self.meg_source_top_label_btn = btn_top
        self.meg_source_top_label_count_spin = top_count
        self.meg_source_top_label_parc_combo = top_parc
        self.meg_volume_plot_mode_combo = volume_mode
        self._set_meg_source_time_label(None)
        self._update_meg_source_controls_enabled()

    def _set_meg_source_time_label(self, rel_sec=None, abs_sec=None, sample_idx=None):
        """Update label describing current projected timepoint."""
        label = getattr(self, "meg_source_time_label", None)
        if label is None:
            return
        if rel_sec is None:
            label.setText("Projected timepoint: --")
            return

        try:
            rel_sec = float(rel_sec)
        except Exception:
            label.setText("Projected timepoint: --")
            return
        parts = [f"t={rel_sec:.3f}s"]
        if abs_sec is not None:
            try:
                abs_sec = float(abs_sec)
            except Exception:
                abs_sec = None
            if abs_sec is not None and np.isfinite(abs_sec):
                parts.append(f"abs={abs_sec:.3f}s")
        if sample_idx is not None:
            try:
                parts.append(f"sample={int(sample_idx)}")
            except Exception:
                pass
        label.setText("Projected timepoint: " + " | ".join(parts))

    def _update_meg_source_controls_enabled(self):
        """Enable source controls only when MEG sensors are loaded."""
        enabled = self._meg_sensor_info is not None
        if self.meg_source_project_btn is not None:
            self.meg_source_project_btn.setEnabled(enabled)
        if self.meg_source_auto_checkbox is not None:
            self.meg_source_auto_checkbox.setEnabled(enabled)
        if self.meg_source_space_combo is not None:
            self.meg_source_space_combo.setEnabled(enabled)
        if self.meg_source_clear_btn is not None:
            self.meg_source_clear_btn.setEnabled(True)
        if self.meg_source_top_label_btn is not None:
            self.meg_source_top_label_btn.setEnabled(getattr(self, "_meg_last_stc", None) is not None)
        if self.meg_source_top_label_count_spin is not None:
            self.meg_source_top_label_count_spin.setEnabled(True)
        if self.meg_source_top_label_parc_combo is not None:
            self.meg_source_top_label_parc_combo.setEnabled(True)
        if self.meg_volume_plot_mode_combo is not None:
            self.meg_volume_plot_mode_combo.setEnabled(True)
        self._sync_meg_alignment_checkbox()

    def _on_meg_source_auto_toggled(self, state):
        """Toggle auto source projection mode."""
        self._meg_source_auto = bool(state == QtCore.Qt.Checked)
        if self._meg_source_auto:
            self.project_cached_meg_window_to_source()

    def _on_meg_source_space_changed(self, _index):
        """Switch between surface and volume source spaces for new inverses."""
        combo = getattr(self, "meg_source_space_combo", None)
        source_space = "surface"
        if combo is not None:
            source_space = str(combo.currentData() or "surface")
        if source_space not in {"surface", "volume"}:
            source_space = "surface"
        if source_space == getattr(self, "_meg_source_space_kind", "surface"):
            return
        self._meg_source_space_kind = source_space
        self._meg_inverse_key = None
        self._meg_inverse_operator = None
        self._meg_source_subject_key = None
        self._meg_source_src = None
        self._meg_source_bem = None
        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(f"Status: Source space set to {source_space}")
        if self._meg_source_auto and self._meg_source_window_data_si is not None:
            self.project_cached_meg_window_to_source()

    def _on_meg_volume_plot_mode_changed(self, _index):
        """Update volume STC display mode."""
        combo = getattr(self, "meg_volume_plot_mode_combo", None)
        mode = "stat_map"
        if combo is not None:
            mode = str(combo.currentData() or "stat_map")
        if mode not in {"stat_map", "glass_brain"}:
            mode = "stat_map"
        self._meg_volume_stc_mode = mode

    def _sync_meg_alignment_checkbox(self):
        """Sync the MEG alignment checkbox with the embedded alignment view."""
        chk = getattr(self, "meg_show_alignment_checkbox", None)
        if chk is None:
            return
        has_fif = bool(
            getattr(self, "_meg_current_fif_path", None)
            or (getattr(self, "_meg_loaded_fif_paths", None) or [])
        )
        chk.blockSignals(True)
        chk.setEnabled(has_fif)
        chk.setChecked(bool(getattr(self, "_meg_alignment_active", False)))
        chk.blockSignals(False)

    def _selected_meg_coreg_fif_path(self):
        """Return the selected/current FIF path for MEG alignment and coreg UI."""
        combo = getattr(self, "meg_coreg_fif_combo", None)
        if combo is not None:
            try:
                path = combo.currentData()
                if path:
                    return path
            except Exception:
                pass
        current = getattr(self, "_meg_current_fif_path", None)
        if current:
            return current
        loaded = list(getattr(self, "_meg_loaded_fif_paths", []) or [])
        return loaded[0] if loaded else None

    def _on_meg_show_alignment_toggled(self, state):
        """Toggle the embedded MEG alignment visualization."""
        show_alignment = bool(state == QtCore.Qt.Checked)
        if show_alignment:
            fif_path = self._selected_meg_coreg_fif_path()
            if fif_path:
                self.show_meg_coreg_alignment_for_fif(fif_path)
            else:
                self._show_meg_alignment_message("Status: Load a MEG FIF run before showing alignment.")
            return
        if hasattr(self, "_hide_meg_coregistration_review"):
            self._hide_meg_coregistration_review()
            return
        self._meg_alignment_active = False
        self._close_meg_alignment_figure()
        self._sync_meg_alignment_checkbox()

    def _on_meg_show_helmet_toggled(self, state):
        """Toggle MEG helmet shell visibility."""
        self._meg_show_helmet = bool(state == QtCore.Qt.Checked)
        if getattr(self, "_meg_alignment_active", False):
            current_fif = None
            combo = getattr(self, "meg_coreg_fif_combo", None)
            if combo is not None:
                current_fif = combo.currentData()
            if not current_fif:
                current_fif = getattr(self, "_meg_current_fif_path", None)
            if current_fif:
                self.show_meg_coreg_alignment_for_fif(current_fif)
            return
        helmet_actor = getattr(self, "_meg_helmet_actor", None)
        if helmet_actor is not None:
            try:
                helmet_actor.SetVisibility(bool(self._meg_show_helmet))
                self._render_plotter(self.brain._renderer.plotter)
                return
            except Exception:
                self._meg_helmet_actor = None
        self._render_meg_sensors(activity_scores=self._meg_activity_scores)

    def set_meg_source_auto(self, enabled=True, project_now=False):
        """Programmatically enable/disable automatic source updates."""
        enabled = bool(enabled)
        chk = getattr(self, "meg_source_auto_checkbox", None)
        if chk is not None:
            chk.blockSignals(True)
            chk.setChecked(enabled)
            chk.blockSignals(False)
        self._meg_source_auto = enabled
        if enabled and project_now:
            self.project_cached_meg_window_to_source()

    def set_meg_noise_cov(self, noise_cov, metadata=None):
        """Provide optional externally computed noise covariance for inverse modeling."""
        if noise_cov is None:
            self._meg_noise_cov = None
            self._meg_noise_cov_metadata = {}
        else:
            try:
                self._meg_noise_cov = noise_cov.copy()
            except Exception:
                self._meg_noise_cov = noise_cov
            self._meg_noise_cov_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        self._meg_noise_cov_version = int(getattr(self, "_meg_noise_cov_version", 0)) + 1
        self._meg_inverse_key = None
        self._meg_inverse_operator = None

    def set_meg_source_estimate_context(self, stc=None, src=None, subject=None, subjects_dir=None):
        """Cache the source-space context for the current source estimate."""
        if stc is not None:
            try:
                self._meg_last_stc = stc.copy()
            except Exception:
                self._meg_last_stc = stc
        if src is not None:
            self._meg_last_stc_src = src
        if subject is not None:
            self._meg_last_stc_subject = str(subject)
        elif stc is not None and getattr(stc, "subject", None):
            self._meg_last_stc_subject = str(getattr(stc, "subject"))
        if subjects_dir is not None:
            self._meg_last_stc_subjects_dir = str(subjects_dir)
        elif stc is not None:
            try:
                _subject_id, current_subjects_dir = self._get_brain_subject_info()
                self._meg_last_stc_subjects_dir = str(current_subjects_dir)
            except Exception:
                pass
        self._update_meg_source_controls_enabled()

    def _estimate_meg_subject_trans(self):
        """Best-effort estimate of head->MRI transform for a loaded subject."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        key = (str(subjects_dir), str(subject_id))
        if key == self._meg_estimated_trans_key and self._meg_estimated_trans is not None:
            return self._meg_estimated_trans
        trans = mne.coreg.estimate_head_mri_t(
            subject=subject_id,
            subjects_dir=subjects_dir,
            verbose="error",
        )
        self._meg_estimated_trans_key = key
        self._meg_estimated_trans = trans
        return trans

    @staticmethod
    def _find_existing_source_space_file(bem_dir, subject_id):
        """Return first existing source-space file path for subject."""
        candidates = [
            bem_dir / f"{subject_id}-ico-5-src.fif",
            bem_dir / f"{subject_id}-oct-6-src.fif",
            bem_dir / f"{subject_id}-src.fif",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @staticmethod
    def _find_existing_volume_source_space_file(bem_dir, subject_id):
        """Return first existing volume source-space file path for subject."""
        candidates = [
            bem_dir / f"{subject_id}-vol-7-src.fif",
            bem_dir / f"{subject_id}-vol-src.fif",
        ]
        for path in candidates:
            if path.exists():
                return path
        for path in sorted(bem_dir.glob("*vol*src.fif")):
            if path.exists():
                return path
        return None

    @staticmethod
    def _find_existing_bem_solution_file(bem_dir, subject_id):
        """Return first existing BEM solution FIF path for subject."""
        candidates = [
            bem_dir / f"{subject_id}-5120-5120-5120-bem-sol.fif",
            bem_dir / f"{subject_id}-bem-sol.fif",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @staticmethod
    def _find_existing_head_model_file(subject_id, subjects_dir):
        """Return a scalp/head model usable by MNE coregistration, if present."""
        subject_root = Path(subjects_dir).expanduser().resolve() / str(subject_id)
        candidates = [
            subject_root / "bem" / f"{subject_id}-head-dense.fif",
            subject_root / "surf" / "lh.seghead",
            subject_root / "surf" / "lh.smseghead",
            subject_root / "bem" / "outer_skin.surf",
            subject_root / "bem" / f"{subject_id}-head-sparse.fif",
            subject_root / "bem" / f"{subject_id}-head.fif",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @staticmethod
    def _has_bem_surfaces(bem_dir, subject_id):
        """Check for inner-skull surfaces needed to build BEM models."""
        candidates = [
            bem_dir / f"{subject_id}-inner_skull.surf",
            bem_dir / "inner_skull.surf",
            bem_dir / "watershed" / f"{subject_id}_inner_skull_surface",
        ]
        return any(path.exists() for path in candidates)

    def _create_watershed_bem_surfaces(self, subject_id, subjects_dir):
        """Create watershed BEM surfaces, including outer_skin for coregistration."""
        import mne.bem

        mne.bem.make_watershed_bem(
            subject=subject_id,
            subjects_dir=str(subjects_dir),
            overwrite=False,
            show=False,
            copy=True,
            verbose="error",
        )
        print(f"Created watershed BEM surfaces for subject {subject_id}")

    def _create_scalp_surface_files(self, subject_id, subjects_dir):
        """Create MNE scalp/head surface FIF files for coregistration."""
        import mne.bem

        mne.bem.make_scalp_surfaces(
            subject=subject_id,
            subjects_dir=str(subjects_dir),
            force=True,
            overwrite=False,
            verbose="error",
        )
        print(f"Created scalp/head surfaces for subject {subject_id}")

    def _create_source_space_file(self, subject_id, subjects_dir, bem_dir):
        """Create and save a cortical source-space file when missing."""
        src = mne.setup_source_space(
            subject=subject_id,
            spacing="ico5",
            surface="white",
            subjects_dir=str(subjects_dir),
            add_dist=False,
            n_jobs=1,
            verbose="error",
        )
        src_path = bem_dir / f"{subject_id}-ico-5-src.fif"
        mne.write_source_spaces(str(src_path), src, overwrite=True, verbose="error")
        print(f"Created source space: {src_path}")
        return src_path

    def _create_volume_source_space_file(self, subject_id, subjects_dir, bem_dir, bem_path=None):
        """Create and save a volume source-space file when missing."""
        if bem_path is None:
            bem_path = self._find_existing_bem_solution_file(bem_dir, subject_id)
        if bem_path is None:
            bem_path = self._create_bem_solution_file(subject_id, subjects_dir, bem_dir)
        pos_mm = float(getattr(self, "_meg_volume_source_pos_mm", 7.0) or 7.0)
        src = mne.setup_volume_source_space(
            subject=subject_id,
            pos=pos_mm,
            bem=str(bem_path),
            subjects_dir=str(subjects_dir),
            add_interpolator=True,
            verbose="error",
        )
        src_path = bem_dir / f"{subject_id}-vol-{int(round(pos_mm))}-src.fif"
        mne.write_source_spaces(str(src_path), src, overwrite=True, verbose="error")
        print(f"Created volume source space: {src_path}")
        return src_path

    def _create_bem_solution_file(self, subject_id, subjects_dir, bem_dir):
        """Create and save a BEM solution FIF when missing."""
        if not self._has_bem_surfaces(bem_dir, subject_id):
            try:
                self._create_watershed_bem_surfaces(subject_id, subjects_dir)
            except Exception as exc:
                raise RuntimeError(
                    "Could not create BEM surfaces automatically. "
                    "Run FreeSurfer/MNE watershed BEM generation and retry. "
                    f"Details: {exc}"
                ) from exc

        try:
            model = mne.make_bem_model(
                subject=subject_id,
                ico=4,
                conductivity=(0.3,),
                subjects_dir=str(subjects_dir),
                verbose="error",
            )
        except Exception:
            model = mne.make_bem_model(
                subject=subject_id,
                ico=4,
                conductivity=(0.3, 0.006, 0.3),
                subjects_dir=str(subjects_dir),
                verbose="error",
            )
        bem = mne.make_bem_solution(model, verbose="error")
        bem_path = bem_dir / f"{subject_id}-bem-sol.fif"
        mne.write_bem_solution(str(bem_path), bem, overwrite=True, verbose="error")
        print(f"Created BEM solution: {bem_path}")
        return bem_path

    def _ensure_meg_anatomy_assets(self, subject_id=None, subjects_dir=None, ensure_source_space=False):
        """Generate missing MNE subject assets needed for coregistration/forward modeling."""
        if subject_id is None or subjects_dir is None:
            subject_id, subjects_dir = self._get_brain_subject_info()
        subjects_dir = Path(subjects_dir).expanduser().resolve()
        subject_root = subjects_dir / str(subject_id)
        if not subject_root.exists():
            raise RuntimeError(f"FreeSurfer subject folder does not exist: {subject_root}")
        if not (subject_root / "mri").is_dir() or not (subject_root / "surf").is_dir():
            raise RuntimeError(
                f"FreeSurfer subject {subject_id} must contain mri/ and surf/ folders under {subjects_dir}."
            )

        bem_dir = subject_root / "bem"
        bem_dir.mkdir(parents=True, exist_ok=True)
        created = []

        if self._find_existing_head_model_file(subject_id, subjects_dir) is None:
            scalp_error = None
            try:
                self._create_scalp_surface_files(subject_id, subjects_dir)
                created.append("scalp/head surfaces")
            except Exception as exc:
                scalp_error = exc

            if self._find_existing_head_model_file(subject_id, subjects_dir) is None:
                try:
                    self._create_watershed_bem_surfaces(subject_id, subjects_dir)
                    created.append("watershed BEM surfaces")
                except Exception as exc:
                    details = f"scalp surfaces: {scalp_error}; watershed BEM: {exc}"
                    raise RuntimeError(
                        "Could not create a standard head model for MNE coregistration. "
                        "Run `mne make_scalp_surfaces` or `mne watershed_bem --copy` manually and retry. "
                        f"Details: {details}"
                    ) from exc

            if self._find_existing_head_model_file(subject_id, subjects_dir) is None:
                raise RuntimeError(
                    f"No standard head model was found or generated for subject {subject_id} in {subjects_dir}."
                )

        if not self._has_bem_surfaces(bem_dir, subject_id):
            try:
                self._create_watershed_bem_surfaces(subject_id, subjects_dir)
            except Exception as exc:
                raise RuntimeError(
                    "Could not create BEM surfaces automatically. "
                    "Run FreeSurfer/MNE watershed BEM generation and retry. "
                    f"Details: {exc}"
                ) from exc
            created.append("watershed BEM surfaces")

        if self._find_existing_bem_solution_file(bem_dir, subject_id) is None:
            self._create_bem_solution_file(subject_id, subjects_dir, bem_dir)
            created.append("BEM solution")

        if ensure_source_space and self._find_existing_source_space_file(bem_dir, subject_id) is None:
            self._create_source_space_file(subject_id, subjects_dir, bem_dir)
            created.append("source space")

        return created

    def _get_source_space_and_bem(self, source_space_kind=None):
        """Load and cache subject source space and BEM solution."""
        source_space_kind = str(
            source_space_kind or getattr(self, "_meg_source_space_kind", "surface") or "surface"
        ).lower()
        if source_space_kind not in {"surface", "volume"}:
            source_space_kind = "surface"
        subject_id, subjects_dir = self._get_brain_subject_info()
        key = (subject_id, str(subjects_dir), source_space_kind)
        if self._meg_source_subject_key == key and self._meg_source_src is not None and self._meg_source_bem is not None:
            return self._meg_source_src, self._meg_source_bem

        bem_dir = Path(subjects_dir) / subject_id / "bem"
        bem_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_meg_anatomy_assets(
            subject_id=subject_id,
            subjects_dir=subjects_dir,
            ensure_source_space=(source_space_kind == "surface"),
        )

        bem_path = self._find_existing_bem_solution_file(bem_dir, subject_id)
        if bem_path is None:
            bem_path = self._create_bem_solution_file(subject_id, subjects_dir, bem_dir)

        if source_space_kind == "volume":
            src_path = self._find_existing_volume_source_space_file(bem_dir, subject_id)
            if src_path is None:
                src_path = self._create_volume_source_space_file(
                    subject_id,
                    subjects_dir,
                    bem_dir,
                    bem_path=bem_path,
                )
        else:
            src_path = self._find_existing_source_space_file(bem_dir, subject_id)
            if src_path is None:
                src_path = self._create_source_space_file(subject_id, subjects_dir, bem_dir)

        src = mne.read_source_spaces(str(src_path), verbose="error")
        bem = mne.read_bem_solution(str(bem_path), verbose="error")
        self._meg_source_subject_key = key
        self._meg_source_src = src
        self._meg_source_bem = bem
        return src, bem

    def _get_inverse_for_meg_channels(self, channel_names):
        """Compute/cache inverse operator for a MEG channel subset."""
        if self._meg_sensor_info is None:
            raise RuntimeError("MEG sensor info is not available.")
        if not channel_names:
            raise RuntimeError("No channels selected for source projection.")

        info_ch_names = list(self._meg_sensor_info["ch_names"])
        picks = [info_ch_names.index(ch) for ch in channel_names if ch in info_ch_names]
        if not picks:
            raise RuntimeError("Selected channels are not present in loaded MEG sensor info.")
        info_subset = mne.pick_info(self._meg_sensor_info.copy(), picks, copy=True)

        subject_id, subjects_dir = self._get_brain_subject_info()
        trans, trans_name = self._resolve_meg_sensor_trans()
        if (
            str(trans_name).startswith("estimated head->mri transform")
            or trans_name in {"identity transform", "legacy loaded transform"}
        ):
            raise RuntimeError(
                "No FIF-specific MEG head->MRI transform is available for source projection. "
                "Create or load a matching -trans.fif for this FIF file first."
            )
        fif_identity = getattr(self, "_meg_current_fif_identity", None)
        if fif_identity is None:
            fif_identity = self._fif_file_identity(getattr(self, "_meg_current_fif_path", None))
        cov_meta = getattr(self, "_meg_noise_cov_metadata", {}) or {}
        cov_key = (
            str(cov_meta.get("method", "")) if isinstance(cov_meta, dict) else "",
            str(cov_meta.get("selected_method", "")) if isinstance(cov_meta, dict) else "",
            str(cov_meta.get("kind", "")) if isinstance(cov_meta, dict) else "",
            str(cov_meta.get("created_at", "")) if isinstance(cov_meta, dict) else "",
        )
        source_space_kind = str(getattr(self, "_meg_source_space_kind", "surface") or "surface")
        inv_key = (
            subject_id,
            str(subjects_dir),
            source_space_kind,
            fif_identity,
            tuple(info_subset["ch_names"]),
            self._info_dev_head_signature(info_subset),
            self._transform_signature(trans),
            int(getattr(self, "_meg_noise_cov_version", 0)),
            cov_key,
        )
        if self._meg_inverse_key == inv_key and self._meg_inverse_operator is not None:
            return self._meg_inverse_operator, info_subset

        src, bem = self._get_source_space_and_bem(source_space_kind=source_space_kind)
        try:
            fwd = mne.make_forward_solution(
                info_subset,
                trans=trans,
                src=src,
                bem=bem,
                meg=True,
                eeg=False,
                ignore_ref=True,
                n_jobs=1,
                verbose="error",
            )
        except RuntimeError as exc:
            if "inside the scalp surface" not in str(exc):
                raise
            fwd = mne.make_forward_solution(
                info_subset,
                trans=trans,
                src=src,
                bem=bem,
                meg=True,
                eeg=False,
                ignore_ref=True,
                on_inside="warn",
                n_jobs=1,
                verbose="error",
            )
            print("Forward model retry: proceeding with on_inside='warn' due to sensor/scalp mismatch.")

        noise_cov = None
        if self._meg_noise_cov is not None:
            try:
                noise_cov = self._meg_noise_cov.copy().pick_channels(
                    list(info_subset["ch_names"]),
                    ordered=True,
                )
            except Exception:
                try:
                    noise_cov = mne.pick_channels_cov(
                        self._meg_noise_cov,
                        include=list(info_subset["ch_names"]),
                        exclude=[],
                        ordered=True,
                        copy=True,
                    )
                except Exception:
                    noise_cov = None

        if noise_cov is None:
            if self._meg_noise_cov is not None or bool(getattr(self, "_meg_noise_cov_metadata", {}) or {}):
                raise RuntimeError(
                    "Run-specific MEG noise covariance could not be aligned to the selected channels."
                )
            noise_cov = mne.make_ad_hoc_cov(info_subset, verbose="error")

        inverse_loose = "auto" if source_space_kind == "volume" else 0.2
        inv = make_inverse_operator(
            info_subset,
            fwd,
            noise_cov,
            loose=inverse_loose,
            depth=0.8,
            verbose="error",
        )
        self._meg_inverse_key = inv_key
        self._meg_inverse_operator = inv
        return inv, info_subset

    def set_meg_projection_window(
        self,
        window_data_si,
        channel_names,
        sfreq,
        start_sample=None,
        tmin_sec=0.0,
    ):
        """Cache current MEG window data for source projection."""
        if window_data_si is None or channel_names is None:
            return
        arr = np.asarray(window_data_si, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return
        self._meg_source_window_data_si = arr
        self._meg_source_window_ch_names = list(channel_names)
        self._meg_source_window_sfreq = float(sfreq)
        self._meg_source_window_start_sample = (
            int(start_sample) if start_sample is not None else None
        )
        try:
            self._meg_source_window_tmin = float(tmin_sec)
        except Exception:
            self._meg_source_window_tmin = 0.0
        if self._meg_source_auto:
            self.project_cached_meg_window_to_source()

    def _project_meg_data_to_source(self, data_si, channel_names, method="MNE", lambda2=1.0 / 9.0):
        """Project MEG sensor data matrix to cortical source space and render it."""
        inv, info_subset = self._get_inverse_for_meg_channels(channel_names)
        data = np.asarray(data_si, dtype=float)
        if data.ndim != 2:
            raise RuntimeError("MEG source projection requires 2D data [channels x time].")
        if data.shape[0] != len(info_subset["ch_names"]):
            ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}
            rows = []
            for ch in info_subset["ch_names"]:
                if ch not in ch_to_idx:
                    raise RuntimeError(f"Missing channel data for {ch}")
                rows.append(data[ch_to_idx[ch], :])
            data = np.vstack(rows)

        evoked = mne.EvokedArray(data, info_subset, tmin=0.0, verbose="error")
        stc = apply_inverse(
            evoked,
            inv,
            lambda2=float(lambda2),
            method=str(method),
            pick_ori=None,
            verbose="error",
        )
        self.set_meg_source_estimate_context(
            stc,
            src=inv.get("src"),
            subject=getattr(stc, "subject", None),
        )

        self.display_meg_stc_on_brain(
            stc,
            time_sec=None,
            rel_tmin=float(getattr(self, "_meg_source_window_tmin", 0.0) or 0.0),
            sfreq=float(getattr(self, "_meg_source_window_sfreq", 0.0) or 0.0),
            start_sample=getattr(self, "_meg_source_window_start_sample", None),
        )
        return int(len(info_subset["ch_names"]))

    def _ensure_fsaverage_in_subjects_dir(self, subjects_dir):
        """Make fsaverage available in a patient subjects_dir for surface morphing."""
        if not subjects_dir:
            return None
        subjects_dir = Path(subjects_dir).expanduser().resolve()
        dest = subjects_dir / "fsaverage"
        if dest.exists():
            return str(subjects_dir)

        sample_subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
        mne.datasets.fetch_fsaverage(subjects_dir=sample_subjects_dir, verbose=False)
        src = sample_subjects_dir / "fsaverage"
        if not src.exists():
            return None

        try:
            subjects_dir.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(src, target_is_directory=True)
            return str(subjects_dir)
        except FileExistsError:
            return str(subjects_dir) if dest.exists() else None
        except Exception as exc:
            print(f"MEG source morph warning: could not link fsaverage into {subjects_dir}: {exc}")
            return None

    def _subjects_dir_for_stc_morph(self, subject_from, subject_to, target_subjects_dir):
        """Choose a subjects_dir containing source and target FreeSurfer subjects."""
        subject_from = str(subject_from or "")
        subject_to = str(subject_to or "")
        custom_subject = str(getattr(self, "custom_subject", "") or "")
        custom_subjects_dir = getattr(self, "custom_subjects_dir", None)

        if custom_subject and custom_subjects_dir and custom_subject in {subject_from, subject_to}:
            if "fsaverage" in {subject_from.lower(), subject_to.lower()}:
                return self._ensure_fsaverage_in_subjects_dir(custom_subjects_dir)
            return str(Path(custom_subjects_dir).expanduser().resolve())

        if target_subjects_dir:
            if "fsaverage" in {subject_from.lower(), subject_to.lower()}:
                return self._ensure_fsaverage_in_subjects_dir(target_subjects_dir)
            return str(Path(target_subjects_dir).expanduser().resolve())
        return None

    def _stc_for_current_brain_subject(self, stc):
        """Return STC data in the currently displayed brain subject space."""
        if stc is None:
            return None, "missing"
        if not is_surface_stc(stc):
            return stc, "native"
        target_subject, target_subjects_dir = self._get_brain_subject_info()
        target_subject = str(target_subject)
        source_subject = str(getattr(stc, "subject", "") or "")
        if not source_subject or source_subject == target_subject:
            return stc, "native"

        morph_subjects_dir = self._subjects_dir_for_stc_morph(
            source_subject,
            target_subject,
            target_subjects_dir,
        )
        if not morph_subjects_dir:
            raise RuntimeError(
                f"Cannot morph source estimate from {source_subject} to {target_subject}: "
                "both FreeSurfer subjects must be available in one subjects_dir."
            )

        vertices_sig = tuple(tuple(np.asarray(v, dtype=int).tolist()) for v in stc.vertices)
        morph_key = (
            source_subject,
            target_subject,
            str(Path(morph_subjects_dir).expanduser().resolve()),
            vertices_sig,
        )
        morph = self._meg_stc_morph_cache.get(morph_key)
        if morph is None:
            morph = mne.compute_source_morph(
                stc,
                subject_from=source_subject,
                subject_to=target_subject,
                subjects_dir=morph_subjects_dir,
                spacing=5,
                warn=False,
                verbose="error",
            )
            self._meg_stc_morph_cache[morph_key] = morph
        return morph.apply(stc), f"morphed {source_subject}->{target_subject}"

    def _stc_for_subject(self, stc, target_subject, target_subjects_dir):
        """Morph a surface STC to a requested target subject when needed."""
        if stc is None or not is_surface_stc(stc):
            return stc, "native"
        target_subject = str(target_subject or "")
        source_subject = str(getattr(stc, "subject", "") or "")
        if not target_subject or not source_subject or source_subject == target_subject:
            return stc, "native"

        morph_subjects_dir = self._subjects_dir_for_stc_morph(
            source_subject,
            target_subject,
            target_subjects_dir,
        )
        if not morph_subjects_dir:
            raise RuntimeError(
                f"Cannot morph source estimate from {source_subject} to {target_subject}: "
                "both FreeSurfer subjects must be available in one subjects_dir."
            )
        vertices_sig = tuple(tuple(np.asarray(v, dtype=int).tolist()) for v in stc.vertices)
        morph_key = (
            source_subject,
            target_subject,
            str(Path(morph_subjects_dir).expanduser().resolve()),
            vertices_sig,
        )
        morph = self._meg_stc_morph_cache.get(morph_key)
        if morph is None:
            morph = mne.compute_source_morph(
                stc,
                subject_from=source_subject,
                subject_to=target_subject,
                subjects_dir=morph_subjects_dir,
                spacing=5,
                warn=False,
                verbose="error",
            )
            self._meg_stc_morph_cache[morph_key] = morph
        return morph.apply(stc), f"morphed {source_subject}->{target_subject}"

    def _source_context_for_stc(self, stc=None):
        """Return `(src, subject, subjects_dir)` for a source estimate."""
        src = getattr(self, "_meg_last_stc_src", None)
        subject = str(getattr(stc, "subject", "") or getattr(self, "_meg_last_stc_subject", "") or "")
        subjects_dir = str(getattr(self, "_meg_last_stc_subjects_dir", "") or "")
        if not subject or not subjects_dir:
            try:
                current_subject, current_subjects_dir = self._get_brain_subject_info()
                subject = subject or str(current_subject)
                subjects_dir = subjects_dir or str(current_subjects_dir)
            except Exception:
                pass
        return src, subject, subjects_dir

    def display_meg_volume_stc(self, stc, src=None, time_sec=None):
        """Display a volume source estimate using MNE's nilearn-backed viewer."""
        if stc is None:
            return False
        src_ctx, subject, subjects_dir = self._source_context_for_stc(stc)
        if src is None:
            src = src_ctx
        if src is None:
            raise RuntimeError("Volume source estimate display requires the source space used to create it.")

        times = np.asarray(getattr(stc, "times", []), dtype=float)
        initial_time = None
        time_idx = None
        if times.size > 0:
            if time_sec is None:
                time_idx = int(times.size // 2)
            else:
                time_idx = int(np.argmin(np.abs(times - float(time_sec))))
            time_idx = max(0, min(time_idx, int(times.size) - 1))
            initial_time = float(times[time_idx])

        mode = str(getattr(self, "_meg_volume_stc_mode", "stat_map") or "stat_map")
        if mode not in {"stat_map", "glass_brain"}:
            mode = "stat_map"
        plot = stc.plot(
            src,
            subject=subject or getattr(stc, "subject", None),
            subjects_dir=subjects_dir or None,
            mode=mode,
            initial_time=initial_time,
            show=True,
        )
        self._meg_volume_stc_plot = plot
        self.set_meg_source_estimate_context(stc, src=src, subject=subject, subjects_dir=subjects_dir)
        if initial_time is not None:
            self._meg_last_stc_time_sec = initial_time
            self._set_meg_source_time_label(rel_sec=initial_time)
        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(
                f"Status: Volume source estimate displayed ({mode}, subject={subject or 'unknown'})"
            )
        return True

    def display_meg_stc_on_brain(self, stc, time_sec=None, rel_tmin=None, sfreq=None, start_sample=None):
        """Display an arbitrary STC on the persistent 3D viewer brain."""
        if self.brain is None or stc is None:
            return False
        if is_volume_stc(stc):
            return self.display_meg_volume_stc(stc, time_sec=time_sec)
        original_stc = stc
        stc, display_space = self._stc_for_current_brain_subject(stc)
        data_abs = np.abs(np.asarray(stc.data, dtype=float))
        if data_abs.ndim != 2 or data_abs.shape[1] == 0:
            raise RuntimeError("Projected source estimate has invalid shape.")

        times = np.asarray(getattr(stc, "times", []), dtype=float)
        if time_sec is None or times.size == 0:
            time_idx = int(data_abs.shape[1] // 2)
        else:
            try:
                time_idx = int(np.argmin(np.abs(times - float(time_sec))))
            except Exception:
                time_idx = int(data_abs.shape[1] // 2)
        time_idx = max(0, min(int(time_idx), data_abs.shape[1] - 1))
        src_vals = data_abs[:, time_idx]

        rel_sec = None
        if times.size > 0 and np.isfinite(times[time_idx]):
            rel_sec = float(times[time_idx])
        elif rel_tmin is not None and sfreq:
            rel_sec = float(rel_tmin) + (float(time_idx) / float(sfreq))

        abs_sec = None
        sample_idx = None
        if start_sample is not None and sfreq:
            sample_idx = int(start_sample) + int(round(float(rel_sec or 0.0) * float(sfreq)))
            abs_sec = float(sample_idx) / float(sfreq)
        self._set_meg_source_time_label(rel_sec=rel_sec, abs_sec=abs_sec, sample_idx=sample_idx)

        n_lh = len(stc.vertices[0])
        lh_vals = src_vals[:n_lh]
        rh_vals = src_vals[n_lh:]
        finite_vals = src_vals[np.isfinite(src_vals)]
        if finite_vals.size == 0:
            raise RuntimeError("Projected source estimate contains no finite values.")

        fmin = float(np.percentile(finite_vals, 60))
        fmid = float(np.percentile(finite_vals, 80))
        fmax = float(np.percentile(finite_vals, 98))
        if not (fmax > fmid > fmin):
            fmin = float(np.min(finite_vals))
            fmax = float(np.max(finite_vals))
            fmid = fmin + (fmax - fmin) * 0.5
            if fmax <= fmin:
                fmax = fmin + 1.0
                fmid = fmin + 0.5

        try:
            self.brain.remove_data()
        except Exception:
            pass

        self.brain.add_data(
            lh_vals,
            vertices=stc.vertices[0],
            hemi="lh",
            colormap="hot",
            fmin=fmin,
            fmid=fmid,
            fmax=fmax,
            smoothing_steps=5,
            colorbar=False,
        )
        self.brain.add_data(
            rh_vals,
            vertices=stc.vertices[1],
            hemi="rh",
            colormap="hot",
            fmin=fmin,
            fmid=fmid,
            fmax=fmax,
            smoothing_steps=5,
            colorbar=False,
        )
        self.set_meg_source_estimate_context(original_stc)
        self._meg_last_stc_time_sec = float(rel_sec) if rel_sec is not None else None
        if hasattr(self, "coreg_status_label") and display_space != "native":
            self.coreg_status_label.setText(f"Status: Source projection displayed ({display_space})")
        self._render_plotter(self.brain._renderer.plotter)
        return True

    def project_meg_evoked_to_source(self, evoked, method="MNE", lambda2=1.0 / 9.0):
        """Project an MNE Evoked (or Evoked-like) object to source space and render."""
        if self.brain is None or evoked is None:
            return False
        if self._meg_sensor_info is None:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Load MEG FIF sensors before source projection")
            return False

        try:
            evoked_meg = evoked.copy().pick("meg", exclude="bads")
            data_si = np.asarray(evoked_meg.data, dtype=float)
            ch_names = list(evoked_meg.ch_names)
            if data_si.ndim != 2 or data_si.shape[0] == 0 or data_si.shape[1] == 0:
                raise RuntimeError("Evoked data is empty for MEG source projection.")

            self._meg_source_window_data_si = data_si
            self._meg_source_window_ch_names = list(ch_names)
            self._meg_source_window_sfreq = float(evoked_meg.info.get("sfreq", 1.0))
            self._meg_source_window_start_sample = None
            try:
                self._meg_source_window_tmin = float(evoked_meg.times[0])
            except Exception:
                self._meg_source_window_tmin = 0.0

            n_channels = self._project_meg_data_to_source(
                data_si,
                ch_names,
                method=method,
                lambda2=lambda2,
            )
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(
                    f"Status: Source projection updated from evoked ({method}) | channels={n_channels}"
                )
            return True
        except Exception as exc:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Source projection error - {exc}")
            print(f"Source projection error: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def project_cached_meg_window_to_source(self):
        """Run MNE inverse on current cached window and paint cortex activity."""
        if self.brain is None:
            return False
        if self._meg_sensor_info is None:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Load MEG FIF sensors before source projection")
            return False
        if self._meg_source_window_data_si is None or not self._meg_source_window_ch_names:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: No MEG signal window available for projection")
            return False

        try:
            n_channels = self._project_meg_data_to_source(
                data_si=self._meg_source_window_data_si,
                channel_names=self._meg_source_window_ch_names,
                method="MNE",
                lambda2=1.0 / 9.0,
            )
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(
                    f"Status: Source projection updated (MNE) | channels={n_channels}"
                )
            return True
        except Exception as exc:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Source projection error - {exc}")
            print(f"Source projection error: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def clear_meg_source_projection(self):
        """Remove projected MEG source data from cortex."""
        if self.brain is None:
            return
        try:
            self.brain.remove_data()
            self._meg_last_stc = None
            self._meg_last_stc_src = None
            self._meg_last_stc_subject = None
            self._meg_last_stc_subjects_dir = None
            self._meg_last_stc_time_sec = None
            volume_plot = getattr(self, "_meg_volume_stc_plot", None)
            if volume_plot is not None:
                try:
                    volume_plot.close()
                except Exception:
                    pass
            self._meg_volume_stc_plot = None
            self._set_meg_source_time_label(None)
            self._render_plotter(self.brain._renderer.plotter)
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Source projection cleared")
            self._update_meg_source_controls_enabled()
        except Exception:
            pass

    def _selected_meg_top_label_count(self):
        spin = getattr(self, "meg_source_top_label_count_spin", None)
        if spin is None:
            return 7
        try:
            return int(spin.value())
        except Exception:
            return 7

    def _selected_meg_top_label_parcellation(self, stc=None):
        combo = getattr(self, "meg_source_top_label_parc_combo", None)
        parcellation = "auto"
        if combo is not None:
            parcellation = str(combo.currentData() or "auto")
        if parcellation == "auto":
            return "aparc+aseg" if is_volume_stc(stc) else "aparc"
        return normalize_parcellation_key(parcellation)

    def _plot_label_time_courses(self, times, label_tc, label_names, top_idxs, title):
        """Open a Matplotlib plot for top label time courses."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
        for idx in top_idxs:
            ax.plot(times, label_tc[idx], linewidth=1.8, label=str(label_names[idx]))
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean |activation|")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        fig.show()
        figures = list(getattr(self, "_meg_top_label_figures", []) or [])
        figures.append(fig)
        self._meg_top_label_figures = figures[-6:]
        return fig

    def _show_surface_top_labels_on_brain(self, label_objects, parcellation):
        """Overlay selected surface labels on the persistent Brain scene."""
        if self.brain is None:
            return
        try:
            self.brain.remove_labels()
        except Exception:
            pass
        parc = normalize_parcellation_key(parcellation)
        try:
            if parc in HCP_MMP_PARCS:
                self.brain.add_annotation(parc, borders=2, alpha=0.35, remove_existing=True)
            elif parc == "aparc":
                self.brain.add_annotation("aparc", borders=2, alpha=0.25, remove_existing=True)
        except Exception:
            pass
        for label in label_objects:
            try:
                self.brain.add_label(label, alpha=0.75, borders=False)
            except Exception:
                pass
        try:
            self._render_plotter(self.brain._renderer.plotter)
        except Exception:
            pass

    def _show_volume_top_labels_on_brain(self, label_names, aseg_path):
        """Overlay selected volume labels on the persistent Brain scene."""
        if self.brain is None or not label_names:
            return
        try:
            self.brain.remove_volume_labels()
        except Exception:
            pass
        try:
            self.brain.add_volume_labels(
                aseg=aseg_name_from_path(aseg_path),
                labels=list(label_names),
                legend=dict(bcolor=None),
            )
            self._render_plotter(self.brain._renderer.plotter)
        except Exception as exc:
            print(f"Could not overlay volume source labels: {exc}")

    def plot_top_meg_source_labels(self):
        """Plot the labels with strongest source activation in the current STC."""
        stc = getattr(self, "_meg_last_stc", None)
        if stc is None:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: No source estimate is available for top-label plotting")
            return False

        top_n = self._selected_meg_top_label_count()
        parcellation = self._selected_meg_top_label_parcellation(stc)
        times = np.asarray(getattr(stc, "times", []), dtype=float)
        if times.size == 0:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Source estimate has no time axis")
            return False

        try:
            if is_volume_stc(stc):
                src, subject, subjects_dir = self._source_context_for_stc(stc)
                aseg_path = subject_aseg_path(subject or getattr(stc, "subject", ""), subjects_dir)
                label_tc, label_names = volume_label_time_courses(stc, src, aseg_path, use_abs=True)
                top_idxs = top_label_indices(label_tc, label_names, top_n=top_n)
                if not top_idxs:
                    raise RuntimeError("No non-empty volume labels found.")
                self._plot_label_time_courses(
                    times,
                    label_tc,
                    label_names,
                    top_idxs,
                    f"Top {len(top_idxs)} volume labels ({aseg_name_from_path(aseg_path)})",
                )
                self._show_volume_top_labels_on_brain([label_names[idx] for idx in top_idxs], aseg_path)
                if hasattr(self, "coreg_status_label"):
                    self.coreg_status_label.setText(
                        f"Status: Plotted top {len(top_idxs)} volume source labels"
                    )
                return True

            if parcellation == "aparc+aseg":
                parcellation = "aparc"
            src, subject, subjects_dir = self._source_context_for_stc(stc)
            if parcellation in HCP_MMP_PARCS:
                target_subject = "fsaverage"
                target_subjects_dir = self._ensure_fsaverage_in_subjects_dir(subjects_dir)
                if not target_subjects_dir:
                    target_subjects_dir = str(mne.datasets.sample.data_path() / "subjects")
                    mne.datasets.fetch_fsaverage(subjects_dir=target_subjects_dir, verbose=False)
                stc_for_labels, display_space = self._stc_for_subject(
                    stc,
                    target_subject,
                    target_subjects_dir,
                )
                labels = read_surface_labels(target_subject, target_subjects_dir, parcellation)
                current_subject, _current_subjects_dir = self._get_brain_subject_info()
                if str(current_subject) != "fsaverage" and hasattr(self, "set_subject_view_mode"):
                    self.set_subject_view_mode("fsaverage")
            else:
                target_subject = subject or str(getattr(stc, "subject", "") or "")
                target_subjects_dir = subjects_dir
                stc_for_labels, display_space = self._stc_for_subject(
                    stc,
                    target_subject,
                    target_subjects_dir,
                )
                labels = read_surface_labels(target_subject, target_subjects_dir, "aparc")

            label_tc, label_names, label_objects = surface_label_time_courses(
                stc_for_labels,
                labels,
                use_abs=True,
            )
            top_idxs = top_label_indices(label_tc, label_names, top_n=top_n)
            if not top_idxs:
                raise RuntimeError("No non-empty surface labels found.")
            top_labels = [label_objects[idx] for idx in top_idxs]
            self._plot_label_time_courses(
                np.asarray(getattr(stc_for_labels, "times", times), dtype=float),
                label_tc,
                label_names,
                top_idxs,
                f"Top {len(top_idxs)} surface labels ({parcellation}, {display_space})",
            )
            self._show_surface_top_labels_on_brain(top_labels, parcellation)
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(
                    f"Status: Plotted top {len(top_idxs)} source labels ({parcellation})"
                )
            return True
        except Exception as exc:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Top-label plotting error - {exc}")
            print(f"Top-label source plotting error: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def get_last_meg_source_estimate(self, copy=True):
        """Return last inverse source estimate computed in the 3D MEG pipeline."""
        stc = getattr(self, "_meg_last_stc", None)
        if stc is None:
            return None
        if copy:
            try:
                return stc.copy()
            except Exception:
                return stc
        return stc

    def _read_fif_info_for_sensor_plot(self, fif_path):
        """Read MNE Info from a FIF file (raw/epochs/evoked)."""
        load_errors = []

        try:
            raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="error")
            return raw.info, "raw"
        except Exception as exc:
            load_errors.append(f"raw: {exc}")

        try:
            epochs = mne.read_epochs(fif_path, preload=False, verbose="error")
            return epochs.info, "epochs"
        except Exception as exc:
            load_errors.append(f"epochs: {exc}")

        try:
            evokeds = mne.read_evokeds(
                fif_path,
                condition=None,
                baseline=None,
                proj=True,
                verbose="error",
            )
            evoked = evokeds[0] if isinstance(evokeds, list) else evokeds
            return evoked.info, "evoked"
        except Exception as exc:
            load_errors.append(f"evoked: {exc}")

        raise RuntimeError(
            "Could not read FIF as raw, epochs, or evoked.\n"
            + "\n".join(load_errors)
        )

    @staticmethod
    def _preferred_meg_alignment_surface(subject, subjects_dir):
        """Choose the best available scalp surface for MNE plot_alignment."""
        if not subject or not subjects_dir:
            return "head", None
        subject_root = Path(subjects_dir).expanduser().resolve() / str(subject)
        dense_candidates = [
            subject_root / "bem" / f"{subject}-head-dense.fif",
            subject_root / "surf" / "lh.seghead",
            subject_root / "surf" / "lh.smseghead",
        ]
        for path in dense_candidates:
            if path.exists():
                return "head-dense", path

        head_candidates = [
            subject_root / "bem" / "outer_skin.surf",
            subject_root / "bem" / f"{subject}-head-sparse.fif",
            subject_root / "bem" / f"{subject}-head.fif",
            subject_root / "bem" / "watershed" / f"{subject}_outer_skin_surface",
        ]
        for path in head_candidates:
            if path.exists():
                return "head", path
        return "head", None

    def _close_meg_alignment_figure(self):
        """Close the current MEG plot_alignment figure/widget if present."""
        widget = getattr(self, "_meg_alignment_widget", None)
        if widget is not None:
            try:
                widget.setParent(None)
            except Exception:
                pass
            try:
                widget.hide()
            except Exception:
                pass
            try:
                QtCore.QTimer.singleShot(0, widget.deleteLater)
            except Exception:
                pass

        plotter = getattr(self, "_meg_alignment_plotter", None)
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass

        fig = getattr(self, "_meg_alignment_fig", None)
        if fig is not None:
            try:
                fig.close()
            except Exception:
                pass

        self._meg_alignment_fig = None
        self._meg_alignment_widget = None
        self._meg_alignment_plotter = None

    def _meg_alignment_widget_from_figure(self, fig):
        renderer = getattr(fig, "_renderer", None)
        plotter = getattr(renderer, "plotter", None) if renderer is not None else None
        if plotter is None:
            plotter = getattr(fig, "plotter", None)
        if plotter is None:
            raise RuntimeError("plot_alignment did not return a PyVista plotter.")
        widget = getattr(plotter, "interactor", None)
        if widget is None:
            widget = getattr(plotter, "app_window", None)
        if widget is None:
            raise RuntimeError("plot_alignment plotter does not expose a Qt widget.")
        return widget, plotter

    def _show_meg_alignment_message(self, message):
        """Show a MEG alignment status message in the right-side viewer."""
        if getattr(self, "_viewer_right_layout", None) is None:
            return
        self._close_meg_alignment_figure()
        self._meg_alignment_active = True
        preserve = [getattr(self, "_plot_widget", None)]
        self._clear_viewer_right_layout(preserve_widgets=preserve)
        label = QtWidgets.QLabel(str(message))
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("QLabel { color: #444444; padding: 18px; font-size: 12px; }")
        self._viewer_right_layout.addWidget(label, 1)
        self._viewer_right_layout.addWidget(self.get_label_chart())
        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(str(message))
        if hasattr(self, "_sync_coreg_toggle_button"):
            self._sync_coreg_toggle_button()
        self._sync_meg_alignment_checkbox()

    def show_meg_coreg_alignment_for_fif(self, fif_path):
        """Render MNE plot_alignment for one loaded MEG FIF."""
        if not fif_path:
            self._show_meg_alignment_message("Status: Select a MEG FIF run for coregistration review.")
            return False

        fif_path = str(Path(fif_path).expanduser().resolve())
        self._register_loaded_meg_fif(fif_path)
        try:
            info, info_kind = self._read_fif_info_for_sensor_plot(fif_path)
            meg_picks, bad_meg_names = good_meg_picks(info, meg=True, ref_meg=False)
            if bad_meg_names:
                message = format_bad_meg_channel_message(
                    "MEG coregistration review",
                    fif_path=fif_path,
                    bad_names=bad_meg_names,
                )
                if message:
                    print(message)
            if meg_picks.size == 0:
                raise RuntimeError("No usable MEG channels found in selected FIF after excluding marked bad channels.")
            meg_info = mne.pick_info(info, meg_picks, copy=True)

            subject, subjects_dir = self._get_brain_subject_info()
            if not subject or not subjects_dir:
                self._show_meg_alignment_message("Status: No FreeSurfer subject is available for MEG alignment review.")
                return False

            trans_path = self._find_existing_meg_trans_file(fif_path=fif_path, subject_id=subject)
            if trans_path is None or not Path(trans_path).exists():
                self._show_meg_alignment_message(
                    f"Status: No run-specific MEG transform found for {Path(fif_path).name}."
                )
                return False

            trans = mne.read_trans(str(trans_path))
            surface, _surface_path = self._preferred_meg_alignment_surface(subject, subjects_dir)
            meg_layers = ["sensors"]
            if bool(getattr(self, "_meg_show_helmet", True)):
                meg_layers.insert(0, "helmet")

            self._close_meg_alignment_figure()
            fig = mne.viz.plot_alignment(
                meg_info,
                trans=trans,
                subject=str(subject),
                subjects_dir=str(subjects_dir),
                surfaces=surface,
                dig=True,
                eeg=[],
                meg=meg_layers,
                show_axes=True,
                coord_frame="meg",
                verbose="error",
            )
            try:
                mne.viz.set_3d_view(
                    fig,
                    azimuth=45,
                    elevation=90,
                    distance=0.6,
                    focalpoint=(0.0, 0.0, 0.0),
                )
            except Exception:
                pass
            widget, plotter = self._meg_alignment_widget_from_figure(fig)
            widget.setParent(None)

            self._meg_alignment_fig = fig
            self._meg_alignment_widget = widget
            self._meg_alignment_plotter = plotter
            self._meg_alignment_active = True

            if getattr(self, "_viewer_right_layout", None) is not None:
                preserve = [getattr(self, "_plot_widget", None), widget]
                self._clear_viewer_right_layout(preserve_widgets=preserve)
                self._viewer_right_layout.addWidget(widget, 1)
                self._viewer_right_layout.addWidget(self.get_label_chart())
            try:
                plotter.render()
            except Exception:
                pass

            status = (
                f"Status: MEG coregistration review - {Path(fif_path).name} "
                f"({info_kind}) | transform {Path(trans_path).name} | surface={surface}"
            )
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(status)
            print(status)
            if hasattr(self, "_sync_coreg_toggle_button"):
                self._sync_coreg_toggle_button()
            self._sync_meg_alignment_checkbox()
            return True
        except Exception as exc:
            self._show_meg_alignment_message(f"Status: MEG coregistration review error - {exc}")
            import traceback
            traceback.print_exc()
            return False

    def _resolve_meg_sensor_trans(self, info=None, fif_path=None):
        """Resolve which transform to use for MEG sensor plotting."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        requested_fif_identity = (
            self._fif_file_identity(
                fif_path or getattr(self, "_meg_current_fif_path", None)
            )
            or getattr(self, "_meg_current_fif_identity", None)
        )

        if self._meg_trans is not None:
            trans_fif_identity = getattr(self, "_meg_trans_fif_identity", None)
            if requested_fif_identity is None or trans_fif_identity == requested_fif_identity:
                if self._meg_loaded_trans_path:
                    current_trans_identity = self._fif_file_identity(self._meg_loaded_trans_path)
                    loaded_trans_identity = getattr(self, "_meg_loaded_trans_file_identity", None)
                    if loaded_trans_identity is not None and current_trans_identity != loaded_trans_identity:
                        trans = self._load_meg_trans_file(
                            self._meg_loaded_trans_path,
                            fif_path=fif_path,
                        )
                        return trans, f"transform {Path(self._meg_loaded_trans_path).name}"
                    return self._meg_trans, f"transform {Path(self._meg_loaded_trans_path).name}"
                return self._meg_trans, "loaded MEG transform"

        trans_path = self._find_existing_meg_trans_file(fif_path=fif_path, subject_id=subject_id)
        if trans_path is not None:
            try:
                trans = self._load_meg_trans_file(trans_path, fif_path=fif_path)
                return trans, f"transform {Path(trans_path).name}"
            except Exception:
                pass

        if subject_id == "fsaverage":
            return "fsaverage", "fsaverage transform"

        coreg_key = (
            str(Path(str(fif_path or getattr(self, "_meg_current_fif_path", "") or "")).expanduser().resolve())
            if str(fif_path or getattr(self, "_meg_current_fif_path", "") or "").strip()
            else "",
            str(subject_id),
            str(subjects_dir),
        )
        if self._meg_auto_coreg_attempt_key != coreg_key:
            self._meg_auto_coreg_attempt_key = coreg_key
            try:
                trans, trans_path, dist_msg = self._run_automated_meg_coregistration(
                    info=info,
                    fif_path=fif_path,
                )
                status = (
                    f"Status: Automated MEG coreg saved {Path(trans_path).name} | {dist_msg}"
                )
                if hasattr(self, "coreg_status_label"):
                    self.coreg_status_label.setText(status)
                print(status)
                return trans, f"automated coreg ({Path(trans_path).name})"
            except Exception as exc:
                self._meg_auto_coreg_error = str(exc)

        if requested_fif_identity is None and self.trans is not None:
            return self.trans, "legacy loaded transform"
        try:
            trans = self._estimate_meg_subject_trans()
            trans_name = "estimated head->mri transform"
            if self._meg_auto_coreg_error:
                trans_name += f" (auto coreg failed: {self._meg_auto_coreg_error})"
            return trans, trans_name
        except Exception:
            pass
        return mne.transforms.Transform(fro="head", to="mri", trans=np.eye(4)), "identity transform"

    def _build_meg_sensor_colors(self, activity_scores=None):
        """Build RGBA colors for MEG sensors from channel activity scores."""
        if self._meg_sensor_info is None:
            return None
        ch_names = list(self._meg_sensor_info.get("ch_names", []))
        if not ch_names:
            return None

        colors = np.tile(np.array([0.70, 0.70, 0.70, 0.18], dtype=float), (len(ch_names), 1))
        if not isinstance(activity_scores, dict) or not activity_scores:
            return colors

        valid_vals = []
        for ch in ch_names:
            val = activity_scores.get(ch)
            if val is None:
                continue
            try:
                fval = float(val)
            except Exception:
                continue
            if np.isfinite(fval):
                valid_vals.append(fval)

        if not valid_vals:
            return colors

        vmin = min(valid_vals)
        vmax = max(valid_vals)
        denom = (vmax - vmin) if vmax > vmin else 1.0
        cmap = colormaps.get_cmap("viridis")
        for idx, ch in enumerate(ch_names):
            val = activity_scores.get(ch)
            if val is None:
                continue
            try:
                fval = float(val)
            except Exception:
                continue
            if not np.isfinite(fval):
                continue
            norm = (fval - vmin) / denom
            rgba = np.array(cmap(norm), dtype=float)
            rgba[3] = 0.85
            colors[idx] = rgba

        return colors

    def _render_meg_sensors(self, activity_scores=None):
        """Render MEG helmet/sensors, optionally color-coded by activity."""
        if self.brain is None or self._meg_sensor_info is None:
            return

        self._meg_helmet_actor = None
        try:
            self.brain.remove_sensors(kind="meg")
        except Exception:
            pass
        try:
            self.brain.remove_sensors(kind="helmet")
        except Exception:
            pass

        trans, _ = self._resolve_meg_sensor_trans()
        self._meg_sensor_trans = trans

        if self._meg_show_helmet:
            try:
                sensor_colors = self._build_meg_sensor_colors(activity_scores=activity_scores)
                helmet_result = self.brain.add_sensors(
                    self._meg_sensor_info,
                    trans=trans,
                    meg="sensors",
                    eeg=False,
                    fnirs=False,
                    ecog=False,
                    seeg=False,
                    dbs=False,
                    sensor_colors=sensor_colors,
                )
                self._meg_helmet_actor = helmet_result
            except Exception:
                pass

        
        self._render_plotter(self.brain._renderer.plotter)

    def set_meg_sensor_activity(self, activity_scores, unit_label=None):
        """Update MEG sensor coloring from `{channel_name: activity}`."""
        self._meg_activity_scores = dict(activity_scores) if isinstance(activity_scores, dict) else {}
        self._update_meg_activity_legend(
            activity_scores=self._meg_activity_scores,
            unit_label=unit_label,
        )
        self._render_meg_sensors(activity_scores=self._meg_activity_scores)

    def clear_meg_sensor_activity(self):
        """Reset MEG sensor coloring to default gray."""
        self._meg_activity_scores = {}
        self._update_meg_activity_legend(activity_scores=None)
        self._render_meg_sensors(activity_scores=None)

    def load_fif_sensors(self, fif_path):
        """Load a FIF file and render MEG sensors/helmet in the 3D viewer."""
        if not fif_path:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Error - FIF file not found")
            return False

        if self.brain is None:
            self.init_3d_layout()
        if self.brain is None:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Error - 3D viewer is not initialized")
            return False

        try:
            info, info_kind = self._read_fif_info_for_sensor_plot(fif_path)
            meg_picks, bad_meg_names = good_meg_picks(info, meg=True, ref_meg=False)
            if bad_meg_names:
                message = format_bad_meg_channel_message(
                    "MEG 3D viewer",
                    fif_path=fif_path,
                    bad_names=bad_meg_names,
                )
                if message:
                    print(message)
            if meg_picks.size == 0:
                raise ValueError("No usable MEG channels found in FIF file after excluding marked bad channels.")
            meg_info = mne.pick_info(info, meg_picks, copy=True)

            self._meg_current_fif_path = str(Path(fif_path).expanduser().resolve())
            self._register_loaded_meg_fif(self._meg_current_fif_path)
            self._meg_current_fif_identity = self._fif_file_identity(fif_path)
            self._meg_trans = None
            self._meg_loaded_trans_path = None
            self._meg_loaded_trans_file_identity = None
            self._meg_trans_fif_identity = None
            self._meg_auto_coreg_attempt_key = None
            self._meg_auto_coreg_error = None
            subject_id = None
            try:
                subject_id, _subjects_dir = self._get_brain_subject_info()
            except Exception:
                subject_id = None
            trans_name = "no run-specific transform found"
            trans = None
            trans_path = self._find_existing_meg_trans_file(fif_path=fif_path, subject_id=subject_id)
            if trans_path is not None:
                try:
                    trans = self._load_meg_trans_file(trans_path, fif_path=fif_path)
                    trans_name = f"transform {Path(trans_path).name}"
                except Exception:
                    trans = None
            self._meg_sensor_info = meg_info
            self._meg_sensor_trans = trans
            self._meg_activity_scores = {}
            self._meg_source_window_data_si = None
            self._meg_source_window_ch_names = []
            self._meg_source_window_sfreq = None
            self._meg_source_window_start_sample = None
            self._meg_source_window_tmin = 0.0
            self._meg_noise_cov = None
            self._meg_noise_cov_metadata = {}
            self._meg_noise_cov_version = int(getattr(self, "_meg_noise_cov_version", 0)) + 1
            self._meg_inverse_key = None
            self._meg_inverse_operator = None
            self._meg_estimated_trans = None
            self._meg_estimated_trans_key = None
            self._meg_source_subject_key = None
            self._meg_source_src = None
            self._meg_source_bem = None
            self._meg_last_stc = None
            self._meg_last_stc_src = None
            self._meg_last_stc_subject = None
            self._meg_last_stc_subjects_dir = None
            self._meg_last_stc_time_sec = None
            self._meg_volume_stc_plot = None
            try:
                self.brain.remove_data()
            except Exception:
                pass
            try:
                self.brain.remove_sensors(kind="meg")
            except Exception:
                pass
            try:
                self.brain.remove_sensors(kind="helmet")
            except Exception:
                pass
            self._meg_helmet_actor = None
            self._set_meg_source_time_label(None)
            self._update_meg_activity_legend(activity_scores=None)
            self._update_meg_source_controls_enabled()
            if self.meg_source_space_combo is not None:
                self.meg_source_space_combo.blockSignals(True)
                idx = self.meg_source_space_combo.findData(getattr(self, "_meg_source_space_kind", "surface"))
                if idx >= 0:
                    self.meg_source_space_combo.setCurrentIndex(idx)
                self.meg_source_space_combo.blockSignals(False)
            if self.meg_show_helmet_checkbox is not None:
                self.meg_show_helmet_checkbox.blockSignals(True)
                self.meg_show_helmet_checkbox.setChecked(bool(self._meg_show_helmet))
                self.meg_show_helmet_checkbox.blockSignals(False)
            self._sync_meg_alignment_checkbox()
            status = (
                f"Status: Loaded MEG FIF ({info_kind}) - {Path(fif_path).name} | "
                f"channels={len(meg_info['ch_names'])} | {trans_name}"
            )
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(status)
            print(status)
            if getattr(self, "_meg_alignment_active", False):
                self.show_meg_coreg_alignment_for_fif(fif_path)
            elif hasattr(self, "_sync_coreg_toggle_button"):
                self._sync_coreg_toggle_button()
            return True
        except Exception as exc:
            err = f"Status: Error loading FIF - {exc}"
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(err)
            print(err)
            import traceback
            traceback.print_exc()
            return False
