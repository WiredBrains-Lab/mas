"""
Author: Sunil Mathew
Date: June 14, 2026

This class displays MEG sensors and source estimates on MNE/FreeSurfer brain
models. It also manages MEG-to-MRI transforms and subject anatomy.

Coregistration Workflow:
------------------------
1. Patient MRI and postoperative CT are loaded through the GUI file browser
2. Automatic registration tries multiple methods in order of preference:
   a) FreeSurfer's mri_robust_register (most robust, requires FreeSurfer installation)
   b) SimpleITK registration (pure Python, good quality)
   c) DIPY registration (pure Python, fallback option)
3. The chosen method produces: registered CT volume + transformation matrix
4. Transform is converted to MNE Transform format for compatibility
5. FreeSurfer fsaverage serves as the standard template brain for visualization
6. MNE coregistration GUI can be used for manual refinement (uses fsaverage head model)
7. Final transforms enable accurate electrode localization in patient MRI space

Registration Methods:
---------------------
- FreeSurfer (preferred): Uses Normalized Mutual Information (NMI), robust to intensity differences
- SimpleITK: Multi-resolution registration with Mattes Mutual Information metric
- DIPY: Affine registration with mutual information, pure Python fallback

Requirements:
- MNE-Python with fsaverage template data (required)
- Patient preoperative MRI (T1-weighted preferred) (required)
- Patient postoperative CT with visible electrode contacts (required)
- At least ONE of the following registration tools:
  * FreeSurfer installation with FREESURFER_HOME environment variable set
  * SimpleITK: pip install SimpleITK
  * DIPY: pip install dipy

"""

import os
import sys
import re
import time
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mne.datasets import fetch_fsaverage
from qtpy import QtCore, QtWidgets, QtGui
from pathlib import Path

import atexit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from usercontrols.checkable_combobox import CheckableComboBox

PROFILE_MERGE = bool(int(os.getenv("MAS_PROFILE_MERGE", "1")))
from viewer3d.meg_integration import MEGIntegrationMixin
from core.source_parcellation import (
    HCP_MMP_PARCS,
    aseg_name_from_path,
    normalize_parcellation_key,
    read_surface_labels,
    is_visual_parcel,
    is_semantic_parcel,
)

GRAY_MATTER_LABELS = {
    3, 42, 8, 47, 10, 11, 12, 13, 16, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60
}
WHITE_MATTER_LABELS = {2, 41, 7, 46}

class MEG3DViewer(MEGIntegrationMixin):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MEG3DViewer, cls).__new__(cls)
        return cls.instance

    def __init__(self, c=None, patient_reg_folder=None, chs_3d_view=None):
        self.c = c
        self.patient_reg_folder = patient_reg_folder
        self.chs_3d_view = chs_3d_view
        self.label_names = []
        self.active_labels = []
        self.label_parcellation = "aparc+aseg"
        self.label_parcellation_combo = None
        self._aseg_name = "aparc+aseg"
        self._surface_label_parc = None
        self._surface_labels_by_name = {}
        self._surface_label_subject = None
        self._surface_label_subjects_dir = None
        self._plot_widget = None
        self._cleanup_registered = False
        self.mode = 'meg'
        self.mri_path = None
        self.ct_path = None
        self.trans = None  # transformation matrix for coregistration
        self.lta_transform_path = None  # FreeSurfer LTA transform file
        self.ct_registered_path = None  # Registered CT output file
        self.custom_subject = None  # Custom FreeSurfer subject ID
        self.custom_subjects_dir = None  # Custom FreeSurfer subjects directory
        self.force_fsaverage = False
        self._subject_to_fsaverage_trans = None
        self._subject_to_fsaverage_key = None
        self._subject_to_fsaverage_warned = False
        self._loaded_mri_to_subject_mri_trans = None
        self._loaded_mri_to_subject_mri_key = None
        self._loaded_mri_to_subject_mri_warned = False
        self.detected_electrodes = []
        self.detected_electrode_space = None
        self.detected_electrode_native_space = None
        self.detected_selected_shaft = None
        self._detected_electrode_actors = []
        self._detected_shaft_actors = []
        self._detected_shaft_label_actors = []
        self._channel_to_shaft_cache = {}
        self._playback_activity_scores = {}
        self._playback_activity_shafts = set()
        self.brain_surface = "pial"
        self.brain_cortex_style = "classic"
        self.brain_opacity = 0.85
        self._brain_surface_actors = []
        self.show_ecog_grids = True
        self.ecog_grid_checkbox = None
        self._ecog_sensor_info = None
        self._ecog_sensor_trans = None
        self._ecog_sensor_colors = None
        self._ecog_sensor_scales = None
        self._ecog_sensor_subject_key = None
        self._surface_vertex_cache = {}
        self._surface_kdtree_cache = {}
        self._aseg_path = None
        self._aseg_data = None
        self._aseg_affine = None
        self.coreg_panel = None
        self._mne_coreg_windows = []
        self._viewer_splitter = None
        self._viewer_right_container = None
        self._viewer_right_layout = None
        self._orientation_widget = None
        self._orientation_widget_plotter = None
        self.subject_view_combo = None
        self.brain_transparency_slider = None
        self.brain_transparency_value_label = None
        self._init_meg_state()

        if self.chs_3d_view is not None:
            self._register_cleanup_hooks()
            self.init_3d_layout()
            self.setup_freesurfer_environment()

    def __del__(self):
        self._teardown_renderer()

    #region freesurfer setup
    def setup_freesurfer_environment(self):
        """Configure FreeSurfer environment variables if not already set."""
        
        # Check if FREESURFER_HOME is already set
        if os.environ.get('FREESURFER_HOME'):
            print(f"✓ FREESURFER_HOME already set: {os.environ['FREESURFER_HOME']}")
            return True
        
        print("Setting up FreeSurfer environment...")
        
        # Common FreeSurfer installation locations on macOS
        possible_locations = [
            '/Applications/freesurfer',
            '/usr/local/freesurfer',
            os.path.expanduser('~/freesurfer'),
        ]
        
        freesurfer_home = None
        for location in possible_locations:
            if os.path.isdir(location):
                print(f"  Found FreeSurfer directory: {location}")
                # Check if this is a versioned installation (e.g., /Applications/freesurfer/8.1.0)
                subdirs = [d for d in Path(location).iterdir() if d.is_dir() and d.name[0].isdigit()]
                if subdirs:
                    # Use the latest versioned subdirectory
                    freesurfer_home = str(sorted(subdirs)[-1])
                    print(f"  Using versioned installation: {freesurfer_home}")
                else:
                    freesurfer_home = location
                break
        
        if not freesurfer_home:
            print("✗ FreeSurfer installation not found in common locations")
            return False
        
        # Set FreeSurfer environment variables
        os.environ['FREESURFER_HOME'] = freesurfer_home
        print(f"✓ Set FREESURFER_HOME: {freesurfer_home}")
        
        # Add FreeSurfer bin to PATH
        bin_path = os.path.join(freesurfer_home, 'bin')
        if os.path.isdir(bin_path):
            current_path = os.environ.get('PATH', '')
            if bin_path not in current_path:
                os.environ['PATH'] = f"{bin_path}:{current_path}"
            print(f"✓ Added to PATH: {bin_path}")
        
        # Set other FreeSurfer environment variables
        subjects_dir = os.path.join(freesurfer_home, 'subjects')
        if os.path.isdir(subjects_dir):
            os.environ['SUBJECTS_DIR'] = subjects_dir
            print(f"✓ Set SUBJECTS_DIR: {subjects_dir}")
        
        # Source FreeSurfer setup if needed (for library paths on macOS)
        if sys.platform == 'darwin':  # macOS
            lib_path = os.path.join(freesurfer_home, 'lib')
            if os.path.isdir(lib_path):
                dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
                if lib_path not in dyld_path:
                    os.environ['DYLD_LIBRARY_PATH'] = f"{lib_path}:{dyld_path}"
                print(f"✓ Set DYLD_LIBRARY_PATH: {lib_path}")
        
        return True
    
    def load_freesurfer_subject(self, subject_id, subjects_dir):
        """
        Load a FreeSurfer subject for visualization instead of using fsaverage.
        
        This method allows you to visualize recon-all results from any FreeSurfer
        subjects directory, including results processed on a different computer.
        
        Parameters
        ----------
        subject_id : str
            The subject ID (must match the folder name in subjects_dir)
        subjects_dir : str or Path
            Path to the FreeSurfer subjects directory containing the subject folder
            
        Returns
        -------
        bool
            True if the subject was loaded successfully, False otherwise
            
        Examples
        --------
        Load a subject processed on another computer:
        >>> meg_3d = MEG3DViewer(chs_3d_view=my_3d_view)
        >>> meg_3d.load_freesurfer_subject(
        ...     subject_id='patient01',
        ...     subjects_dir='/path/to/transferred/subjects'
        ... )
        
        Notes
        -----
        The subject folder must contain the complete FreeSurfer reconstruction
        output, including at minimum:
        - surf/ directory with pial and white surfaces
        - mri/ directory with brain volumes
        - label/ directory with parcellations (optional but recommended)
        
        After loading, you can visualize the patient-specific brain by calling
        any of the visualization methods. The Brain object will use the custom
        subject instead of fsaverage.
        """
        subjects_dir = Path(subjects_dir)
        subject_path = subjects_dir / subject_id
        
        # Validate the subject directory exists
        if not subject_path.exists():
            print(f"✗ Subject directory not found: {subject_path}")
            return False
        
        # Check for required subdirectories
        required_dirs = ['surf', 'mri']
        missing_dirs = [d for d in required_dirs if not (subject_path / d).exists()]
        
        if missing_dirs:
            print(f"✗ Subject directory is incomplete. Missing: {', '.join(missing_dirs)}")
            print(f"  Required directories: {', '.join(required_dirs)}")
            return False
        
        # Check for surface files
        surf_dir = subject_path / 'surf'
        required_surfaces = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
        missing_surfaces = [s for s in required_surfaces if not (surf_dir / s).exists()]
        
        if missing_surfaces:
            print(f"✗ Missing surface files: {', '.join(missing_surfaces)}")
            return False
        
        # Store custom subject information
        self.custom_subject = subject_id
        self.custom_subjects_dir = str(subjects_dir)
        if normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg")) in HCP_MMP_PARCS:
            self.label_parcellation = "aparc+aseg"
            self.active_labels = []
        
        print(f"✓ Loaded FreeSurfer subject: {subject_id}")
        print(f"  Location: {subject_path}")
        print(f"  Surfaces: {', '.join(required_surfaces)}")
        
        # Check for optional but useful files
        mri_dir = subject_path / 'mri'
        if (mri_dir / 'aparc+aseg.mgz').exists():
            print(f"  ✓ Segmentation available (aparc+aseg.mgz)")
        
        label_dir = subject_path / 'label'
        if label_dir.exists() and (label_dir / 'lh.aparc.annot').exists():
            print(f"  ✓ Cortical parcellation available")
        
        return True
    
    def _get_brain_subject_info(self):
        """
        Get the subject ID and subjects_dir to use for Brain visualization.
        
        Returns
        -------
        tuple
            (subject_id, subjects_dir) - uses custom subject if loaded, otherwise fsaverage
        """
        if not self.force_fsaverage and self.custom_subject and self.custom_subjects_dir:
            return self.custom_subject, self.custom_subjects_dir
        else:
            # Use fsaverage as default
            if self.custom_subjects_dir and (Path(self.custom_subjects_dir) / "fsaverage").exists():
                return "fsaverage", self.custom_subjects_dir
            for candidate in [
                Path.home() / "mne_data" / "MNE-fsaverage-data",
                Path.home() / "mne_data" / "MNE-sample-data" / "subjects",
            ]:
                if (candidate / "fsaverage").exists():
                    return "fsaverage", str(candidate)
            try:
                fsaverage_dir = Path(fetch_fsaverage(verbose=False))
                return "fsaverage", str(fsaverage_dir.parent)
            except Exception:
                pass
            if self.custom_subjects_dir:
                return "fsaverage", self.custom_subjects_dir
            return "fsaverage", str(Path.home() / "mne_data" / "MNE-fsaverage-data")

    #endregion freesurfer setup

    #region 3D viewer

    def _check_render_ready(self):
        """Best-effort test that a GUI-backed OpenGL surface is available."""
        if self.chs_3d_view is None:
            return False, "3D dock is not available"

        # Wayland/X11 availability check (Linux headless environments)
        if sys.platform.startswith("linux"):
            if not any(os.environ.get(var) for var in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")):
                return False, "No DISPLAY/WAYLAND target detected for Qt"

        qt_platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
        if qt_platform in {"offscreen", "minimal", "minimalistic", "headless"}:
            return False, f"Qt platform '{qt_platform}' does not expose windows"

        try:
            app = QtWidgets.QApplication.instance()
        except RuntimeError:
            app = None

        if app is None:
            return False, "QApplication instance not initialized"

        screen = None
        try:
            screen = app.primaryScreen()
        except RuntimeError:
            screen = None

        if screen is None and hasattr(QtGui, "QGuiApplication"):
            try:
                screen = QtGui.QGuiApplication.primaryScreen()
            except RuntimeError:
                screen = None

        if screen is None:
            return False, "Qt reports no available screens (likely headless)"

        gl_ready, gl_reason = self._probe_opengl()
        if not gl_ready:
            return False, gl_reason

        return True, ""

    def _probe_opengl(self):
        """Create a throwaway OpenGL context to ensure VTK can render."""
        required_version = (2, 1)

        if not hasattr(QtGui, "QSurfaceFormat"):
            return False, "Qt build lacks QSurfaceFormat for OpenGL probing"

        version = (0, 0)

        try:
            surface_format = QtGui.QSurfaceFormat()
            surface_format.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
            surface_format.setMajorVersion(required_version[0])
            surface_format.setMinorVersion(required_version[1])

            gl_context = QtGui.QOpenGLContext()
            gl_context.setFormat(surface_format)
            if not gl_context.create():
                return False, "Unable to create an OpenGL context (needed by PyVista)"

            offscreen_surface = QtGui.QOffscreenSurface()
            offscreen_surface.setFormat(surface_format)
            offscreen_surface.create()
            if not offscreen_surface.isValid():
                return False, "Failed to create an offscreen surface for OpenGL"

            if not gl_context.makeCurrent(offscreen_surface):
                return False, "Cannot activate OpenGL context; GPU/driver missing"

            fmt = gl_context.format()
            version = (fmt.majorVersion(), fmt.minorVersion())
            gl_context.doneCurrent()

        except Exception as exc:  # pragma: no cover - GUI/driver specific
            return False, f"Qt OpenGL probe failed: {exc}"

        if version < required_version:
            return False, f"OpenGL {required_version[0]}.{required_version[1]}+ required, but got {version[0]}.{version[1]}"

        return True, ""

    def _show_disabled_message(self, reason):
        message = (
            "3D viewer disabled: "
            + reason
            + ". This prevents PyVista/VTK from creating a window."
        )
        print(f"[WARN] {message}")

        if self.chs_3d_view is None:
            return

        placeholder = QtWidgets.QLabel(message)
        placeholder.setWordWrap(True)
        placeholder.setAlignment(QtCore.Qt.AlignTop)
        placeholder.setStyleSheet(
            "QLabel { color: #aa0000; padding: 12px; font-size: 12px; }"
        )
        self.chs_3d_view.addWidget(placeholder)


    def init_seg_labels(self):
        # Add segmentation labels to the brain
        if self.brain is None:
            print("No Brain scene is available yet; skipping segmentation labels.")
            return
        subjects_dir = mne.datasets.sample.data_path() / "subjects"
        fname_aseg = subjects_dir / "sample" / "mri" / "aparc+aseg.mgz"
        self._aseg_path = str(fname_aseg)
        self._aseg_name = "aparc+aseg"
        if not getattr(self, "label_parcellation", None):
            self.label_parcellation = "aparc+aseg"
        self._aseg_data = None
        self._aseg_affine = None
        self.label_names = mne.get_volume_labels_from_aseg(fname_aseg)
        if self.mode == 'ecog':
            # use language areas for ECoG
            lang_areas = ['ctx-lh-bankssts', 'ctx-lh-inferiortemporal', 'ctx-lh-superiortemporal', 
                                  'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 'ctx-lh-insula', 'ctx-lh-parahippocampal']
            self.active_labels = [label for label in self.label_names if label in lang_areas]
        else:
            temporal_areas = ['ctx-lh-superiortemporal', 'ctx-lh-inferiortemporal', 'ctx-lh-middletemporal',
                              'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 'ctx-lh-parahippocampal']
            self.active_labels = [label for label in self.label_names if label in temporal_areas]

        
        # self.active_labels = self.active_labels[1:5]
        legend_kwargs = dict(bcolor=None)
        self.brain.add_volume_labels(aseg="aparc+aseg", labels=self.active_labels, legend=legend_kwargs)

    def _init_empty_brain_scene(self):
        """Create a bare brain scene when no demo dataset is loaded."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        self.brain = mne.viz.Brain(
            subject=subject_id,
            subjects_dir=subjects_dir,
            cortex=self.brain_cortex_style,
            alpha=self.brain_opacity,
            background="white",
            surf=self.brain_surface,
            silhouette=False,
            show=False,
        )
        self._finalize_brain_scene(reset_camera=True)
    
    def _init_seg_labels_for_subject(self, subject_id, subjects_dir):
        """Initialize segmentation labels for a specific FreeSurfer/FastSurfer subject."""
        try:
            # Path to the subject's segmentation file
            fname_aseg = Path(subjects_dir) / subject_id / "mri" / "aparc+aseg.mgz"
            
            if not fname_aseg.exists():
                print(f"  Warning: Segmentation file not found: {fname_aseg}")
                # Try alternative location (FastSurfer sometimes uses different structure)
                fname_aseg_alt = Path(subjects_dir) / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
                if fname_aseg_alt.exists():
                    fname_aseg = fname_aseg_alt
                    print(f"  Using FastSurfer segmentation: {fname_aseg}")
                else:
                    print("  No segmentation file found - labels will not be available")
                    self.label_names = []
                    self.active_labels = []
                    return

            self._aseg_path = str(fname_aseg)
            self._aseg_name = aseg_name_from_path(fname_aseg)
            if not getattr(self, "label_parcellation", None):
                self.label_parcellation = "aparc+aseg"
            self._aseg_data = None
            self._aseg_affine = None
            
            # Get all available labels from the segmentation
            self.label_names = mne.get_volume_labels_from_aseg(str(fname_aseg))
            print(f"  Found {len(self.label_names)} segmentation labels")
            
            # Select default labels based on mode
            if self.mode == 'ecog':
                # use language areas for ECoG
                lang_areas = ['ctx-lh-bankssts', 'ctx-lh-inferiortemporal', 'ctx-lh-superiortemporal', 
                              'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 'ctx-lh-insula', 
                              'ctx-lh-parahippocampal']
                self.active_labels = [label for label in self.label_names if label in lang_areas]
            else:
                # default temporal areas
                temporal_areas = ['ctx-lh-superiortemporal', 'ctx-lh-inferiortemporal', 'ctx-lh-middletemporal',
                                  'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 'ctx-lh-parahippocampal']
                self.active_labels = [label for label in self.label_names if label in temporal_areas]
            
            # Add volume labels to the brain visualization
            legend_kwargs = dict(bcolor=None)
            aseg_name = "aparc+aseg" if "aparc+aseg.mgz" in str(fname_aseg) else "aparc.DKTatlas+aseg.deep"
            
            if self.active_labels:
                self.brain.add_volume_labels(aseg=aseg_name, labels=self.active_labels, legend=legend_kwargs)
                print(f"  Added {len(self.active_labels)} default segmentation labels to brain")
            else:
                print("  No default labels found for current mode")
                
        except Exception as e:
            print(f"  Error loading segmentation labels: {e}")
            import traceback
            traceback.print_exc()
            self.label_names = []
            self.active_labels = []

    def _load_aseg_volume(self):
        """Load aseg volume data/affine if available."""
        if not self._aseg_path or not os.path.exists(self._aseg_path):
            return None, None
        if self._aseg_data is not None and self._aseg_affine is not None:
            return self._aseg_data, self._aseg_affine
        try:
            import nibabel as nib
            aseg_img = nib.load(self._aseg_path)
            self._aseg_data = np.asarray(aseg_img.get_fdata())
            self._aseg_affine = aseg_img.affine
            return self._aseg_data, self._aseg_affine
        except Exception as exc:
            print(f"Error loading aseg volume: {exc}")
            self._aseg_data = None
            self._aseg_affine = None
            return None, None

    def _classify_detected_electrodes_tissue(self):
        """Classify detected electrodes as gray/white based on FreeSurfer aseg."""
        if not self.detected_electrodes:
            return
        aseg_data, aseg_affine = self._load_aseg_volume()
        if aseg_data is None or aseg_affine is None:
            print("No aseg volume available for tissue classification.")
            return
        if not self.mri_path or not os.path.exists(self.mri_path):
            print("No MRI path available for tissue classification.")
            return
        try:
            import nibabel as nib
            mri_affine = nib.load(self.mri_path).affine
            inv_aseg_affine = np.linalg.inv(aseg_affine)
        except Exception as exc:
            print(f"Error preparing tissue classification: {exc}")
            return

        total = 0
        gray = 0
        white = 0
        for elec in self.detected_electrodes:
            vox = np.array([elec.x, elec.y, elec.z, 1.0], dtype=float)
            world = mri_affine @ vox
            aseg_vox = inv_aseg_affine @ world
            ix, iy, iz = np.round(aseg_vox[:3]).astype(int)
            if (
                ix < 0
                or iy < 0
                or iz < 0
                or ix >= aseg_data.shape[0]
                or iy >= aseg_data.shape[1]
                or iz >= aseg_data.shape[2]
            ):
                elec.tissue = "unknown"
                total += 1
                continue

            label = int(round(aseg_data[ix, iy, iz]))
            if label in GRAY_MATTER_LABELS:
                elec.tissue = "gray"
                gray += 1
            elif label in WHITE_MATTER_LABELS:
                elec.tissue = "white"
                white += 1
            else:
                elec.tissue = "unknown"
            total += 1

        print(
            f"Tissue classification complete: gray={gray}, white={white}, unknown={total - gray - white}"
        )

    def _update_classify_button_state(self):
        """Enable tissue classification button only when electrodes exist."""
        classify_btn = None
        if self.coreg_panel is not None and hasattr(self.coreg_panel, "classify_tissue_btn"):
            classify_btn = self.coreg_panel.classify_tissue_btn
        elif hasattr(self, "classify_tissue_btn"):
            classify_btn = self.classify_tissue_btn

        if classify_btn is None:
            return

        has_electrodes = bool(self.detected_electrodes)
        has_subject = bool(self.custom_subjects_dir and self.custom_subject)
        has_aseg = bool(self._aseg_path and os.path.exists(self._aseg_path))
        classify_btn.setEnabled(has_electrodes and has_subject and has_aseg)

    def _on_classify_tissue_clicked(self):
        """Handle manual tissue classification trigger."""
        if not self.detected_electrodes:
            QtWidgets.QMessageBox.information(
                None, "No Electrodes", "Detect electrodes first before classification."
            )
            return
        if not self._aseg_path or not os.path.exists(self._aseg_path):
            QtWidgets.QMessageBox.information(
                None,
                "No FreeSurfer Aseg",
                "Load a FreeSurfer/FastSurfer subject first to classify gray/white matter."
            )
            return
        self._classify_detected_electrodes_tissue()
        QtWidgets.QMessageBox.information(
            None, "Classification Complete", "Electrode tissue classification updated."
        )
    
    def init_ecog_dummy_data(self):
        return
        # Load ECoG data and display in 3D
        # paths to mne datasets - sample ECoG and FreeSurfer subject
        bids_root = mne.datasets.epilepsy_ecog.data_path()
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = sample_path / "subjects"

        # first define the bids path
        bids_path = BIDSPath(
            root=bids_root,
            subject="pt1",
            session="presurgery",
            task="ictal",
            datatype="ieeg",
            extension=".vhdr",
        )

        # Then we'll use it to load in the sample dataset. This function changes the
        # units of some channels, so we suppress a related warning here by using
        # verbose='error'.
        raw = read_raw_bids(bids_path=bids_path, verbose="error")

        # Pick only the ECoG channels, removing the EKG channels
        raw.pick(picks="ecog")

        # Load the data
        raw.load_data()

        # Then we remove line frequency interference
        raw.notch_filter([60], trans_bandwidth=3)

        # drop bad channels
        raw.drop_channels(raw.info["bads"])

        # the coordinate frame of the montage
        montage = raw.get_montage()
        print(montage.get_positions()["coord_frame"])

        # add fiducials to montage
        montage.add_mni_fiducials(subjects_dir)

        # now with fiducials assigned, the montage will be properly converted
        # to "head" which is what MNE requires internally (this is the coordinate
        # system with the origin between LPA and RPA whereas MNI has the origin
        # at the posterior commissure)
        raw.set_montage(montage)

        # Make a 25 second epoch that spans before and after the seizure onset
        epoch_length = 25  # seconds
        epochs = mne.Epochs(
            raw,
            event_id="onset",
            tmin=13,
            tmax=13 + epoch_length,
            baseline=None,
        )
        # Make evoked from the one epoch and resample
        evoked = epochs.average().resample(200)
        del epochs

        # xy, im = snapshot_brain_montage(self.brain, raw.info)

        sfreq = 10
        gamma_power_t = (
            evoked.copy().filter(30, 90).apply_hilbert(envelope=True).resample(sfreq)
        )
        gamma_info = gamma_power_t.info

        gamma_power_at_15s = gamma_power_t.to_data_frame(index="time").loc[15]
        # scale values to be between 0 and 1, then map to colors
        gamma_power_at_15s -= gamma_power_at_15s.min()
        gamma_power_at_15s /= gamma_power_at_15s.max()
        rgba = colormaps.get_cmap("viridis")
        sensor_colors = np.array(gamma_power_at_15s.map(rgba).tolist(), float)
        sensor_colors[:, 3] = 0.5

        subject_id, subjects_dir = self._get_brain_subject_info()
        self.brain = mne.viz.Brain(
            subject=subject_id,
            subjects_dir=subjects_dir,
            cortex=self.brain_cortex_style,
            alpha=self.brain_opacity,
            background="white",
            surf=self.brain_surface,
            silhouette=False,
            show=False,
        )
        self._finalize_brain_scene(reset_camera=True)
        # Add sensors with gamma power colors
        trans = subject_id  # use subject_id for transform
        self.brain.add_sensors(raw.info, trans=trans, sensor_colors=sensor_colors)
        self._store_ecog_sensor_state(
            raw.info,
            trans=trans,
            sensor_colors=sensor_colors,
            subject_key=(subject_id, str(subjects_dir)),
        )
        # self.brain.add_sensors(raw.info, trans=trans, sensor_colors=(1.0, 1.0, 1.0, 0.5))

        # mne.viz.set_3d_view(self.brain, azimuth=0, elevation=70, focalpoint="auto", distance="auto")
                
    def plot_gamma_power(self, raw, gamma_power_t):
        # plot the gamma power over time at each electrode position
        xy, im = snapshot_brain_montage(self.brain, raw.info)

        # convert from a dictionary to array to plot
        xy_pts = np.vstack([xy[ch] for ch in raw.info["ch_names"]])

        # get a colormap to color nearby points similar colors
        cmap = plt.colormaps["viridis"]

        # create the figure of the brain with the electrode positions
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Gamma power over time", size="large")
        ax.imshow(im)
        ax.set_axis_off()

        # normalize gamma power for plotting
        gamma_power = -100 * gamma_power_t.data / gamma_power_t.data.max()
        # add the time course overlaid on the positions
        x_line = np.linspace(
            -0.025 * im.shape[0], 0.025 * im.shape[0], gamma_power_t.data.shape[1]
        )
        for i, pos in enumerate(xy_pts):
            x, y = pos
            color = cmap(i / xy_pts.shape[0])
            ax.plot(x_line + x, gamma_power[i] + y, linewidth=0.5, color=color)

    def _register_cleanup_hooks(self):
        """Ensure OpenGL widgets are torn down before Qt exits (prevents BadWindow)."""
        if self._cleanup_registered:
            return

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._teardown_renderer)

        if hasattr(self.chs_3d_view, "sigClosed"):
            try:
                self.chs_3d_view.sigClosed.connect(self._on_dock_closed)
            except Exception:
                pass

        atexit.register(self._teardown_renderer)

        self._cleanup_registered = True

    def _on_dock_closed(self, *args, **kwargs):
        self._teardown_renderer()

    def _teardown_renderer(self):
        """Release PyVista/VTK resources so X11 does not complain about dead windows."""
        if hasattr(self, "_close_meg_alignment_figure"):
            try:
                self._close_meg_alignment_figure()
            except Exception:
                pass

        if getattr(self, "_plot_widget", None) is not None:
            try:
                self._plot_widget.setParent(None)
            except Exception:
                pass
            try:
                self._plot_widget.hide()
            except Exception:
                pass
            try:
                QtCore.QTimer.singleShot(0, self._plot_widget.deleteLater)
            except Exception:
                pass
            finally:
                self._plot_widget = None

        self._orientation_widget = None
        self._orientation_widget_plotter = None
        self._brain_surface_actors = []
        if hasattr(self, "_meg_helmet_actor"):
            self._meg_helmet_actor = None

        brain = getattr(self, "brain", None)
        self.brain = None
        if brain is not None:
            renderer = getattr(brain, "_renderer", None)
            if renderer is not None:
                plotter = getattr(renderer, "plotter", None)
                if plotter is not None:
                    app_window = getattr(plotter, "app_window", None)
                    if app_window is not None:
                        try:
                            app_window.hide()
                        except Exception:
                            pass
                        try:
                            QtCore.QTimer.singleShot(0, app_window.deleteLater)
                        except Exception:
                            pass

                    interactor = getattr(plotter, "interactor", None)
                    if interactor is not None:
                        try:
                            interactor.hide()
                        except Exception:
                            pass
                        try:
                            interactor.setParent(None)
                        except Exception:
                            pass
                        try:
                            QtCore.QTimer.singleShot(0, interactor.deleteLater)
                        except Exception:
                            pass

                    for attr in ("app_window", "interactor"):
                        widget = getattr(plotter, attr, None)
                        if widget is not None:
                            try:
                                widget.hide()
                            except Exception:
                                pass

    def init_3d_layout(self):
        self._teardown_renderer()
        can_render, reason = self._check_render_ready()
        if not can_render:
            self._show_disabled_message(reason)
            return

        try:
            self._init_empty_brain_scene()

            self.init_seg_labels()

            self._plot_widget = self.brain._renderer.plotter.interactor
            self._plot_widget.setParent(None)
            self._ensure_coreg_panel()
            self._populate_viewer_right()
            
        except Exception as e:
            print(f"ERROR in init_3d_layout: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Add error message to the view instead of crashing
            error_label = QtWidgets.QLabel(f"3D Viewer Error: {str(e)}\nCheck console for details.")
            error_label.setStyleSheet("QLabel { color: red; padding: 20px; font-size: 12px; }")
            self.chs_3d_view.addWidget(error_label)

    def _ensure_coreg_panel(self):
        """Create the MEG viewer container without CT/electrode controls."""
        if self.chs_3d_view is None:
            return

        if self._viewer_right_container is None:
            self._viewer_right_container = QtWidgets.QWidget()
            self._viewer_right_layout = QtWidgets.QVBoxLayout()
            self._viewer_right_layout.setContentsMargins(0, 0, 0, 0)
            self._viewer_right_layout.setSpacing(6)
            self._viewer_right_container.setLayout(self._viewer_right_layout)
            self.chs_3d_view.addWidget(self._viewer_right_container)

    def _clear_viewer_right_layout(self, preserve_widgets=None):
        """Remove widgets from the right-side viewer layout."""
        layout = self._viewer_right_layout
        if layout is None:
            return
        preserve = {id(widget) for widget in (preserve_widgets or []) if widget is not None}
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                if id(widget) not in preserve:
                    widget.deleteLater()

    def _populate_viewer_right(self):
        """Populate the right-side viewer with plot and label controls."""
        if self._viewer_right_layout is None:
            return
        if getattr(self, "_meg_alignment_active", False) and hasattr(self, "_close_meg_alignment_figure"):
            try:
                self._close_meg_alignment_figure()
            except Exception:
                pass
            self._meg_alignment_active = False
        preserve = [getattr(self, "_plot_widget", None)]
        if getattr(self, "labels_widget", None) is not None:
            preserve.append(self.labels_widget)
        self._clear_viewer_right_layout(preserve_widgets=preserve)
        if self._plot_widget is not None:
            self._plot_widget.setParent(None)
            self._viewer_right_layout.addWidget(self._plot_widget, 1)
        if getattr(self, "labels_widget", None) is None:
            self._viewer_right_layout.addWidget(self.get_label_chart())
        else:
            self.labels_widget.setParent(None)
            self._viewer_right_layout.addWidget(self.labels_widget)
        self._ensure_orientation_widget()
        self._update_classify_button_state()
        self._sync_coreg_toggle_button()

    def _sync_coreg_toggle_button(self):
        """Update the coregistration toggle button text to match panel visibility."""
        if not hasattr(self, "coreg_toggle_btn"):
            return
        if self._has_loaded_meg_fifs():
            if self.coreg_panel is not None:
                self.coreg_panel.setVisible(False)
            self.coreg_toggle_btn.setText("Show Coregistration UI")
            self.coreg_toggle_btn.setToolTip("Open MNE's interactive MEG coregistration GUI")
            self._refresh_meg_coreg_fif_combo()
            if hasattr(self, "_sync_meg_alignment_checkbox"):
                self._sync_meg_alignment_checkbox()
            self._sync_subject_view_combo()
            return
        is_visible = bool(self.coreg_panel and self.coreg_panel.isVisible())
        self.coreg_toggle_btn.setText(
            "Hide Coregistration UI" if is_visible else "Show Coregistration UI"
        )
        self.coreg_toggle_btn.setToolTip("Show or hide the coregistration panel (MRI/CT browse + registration)")
        if hasattr(self, "_sync_meg_alignment_checkbox"):
            self._sync_meg_alignment_checkbox()
        self._sync_subject_view_combo()

    def create_file_browser_interface(self):
        """Compact coregistration control for the 3D view.

        The full MRI/CT browse UI lives in the coregistration panel. Here we
        present a single button that shows or hides that panel.
        """
        file_browser_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)

        btn = QtWidgets.QPushButton("Show Coregistration UI")
        btn.setToolTip("Show or hide the coregistration panel (MRI/CT browse + registration)")
        btn.clicked.connect(self.toggle_coregistration_panel)
        self.coreg_toggle_btn = btn
        self.coregister_btn = btn
        row_layout.addWidget(btn)

        self.meg_coreg_fif_combo = QtWidgets.QComboBox()
        self.meg_coreg_fif_combo.setToolTip("Select the loaded FIF run to inspect with MNE plot_alignment")
        self.meg_coreg_fif_combo.setVisible(False)
        self.meg_coreg_fif_combo.currentIndexChanged.connect(self._on_meg_coreg_fif_combo_changed)
        row_layout.addWidget(self.meg_coreg_fif_combo)
        self._refresh_meg_coreg_fif_combo()

        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItem("View")
        self.view_combo.addItems([
            "Superior",
            "Inferior",
            "Anterior",
            "Posterior",
            "Left",
            "Right",
        ])
        self.view_combo.setToolTip("Orient the 3D camera to a standard view")
        self.view_combo.currentIndexChanged.connect(self._on_view_combo_changed)
        row_layout.addWidget(self.view_combo)

        self.subject_view_combo = QtWidgets.QComboBox()
        self.subject_view_combo.addItem("fsaverage", "fsaverage")
        self.subject_view_combo.addItem("Patient", "subject")
        self.subject_view_combo.setToolTip("Switch between fsaverage and patient surfaces")
        self.subject_view_combo.currentIndexChanged.connect(self._on_subject_view_combo_changed)
        row_layout.addWidget(self.subject_view_combo)

        row_layout.addStretch()

        layout.addLayout(row_layout)

        surface_layout = QtWidgets.QHBoxLayout()
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_label = QtWidgets.QLabel("Surface:")
        surface_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        surface_layout.addWidget(surface_label)
        self.inflated_surface_checkbox = QtWidgets.QCheckBox("Inflated")
        self.inflated_surface_checkbox.setChecked(self.brain_surface == "inflated")
        self.inflated_surface_checkbox.stateChanged.connect(self._toggle_inflated_surface)
        surface_layout.addWidget(self.inflated_surface_checkbox)
        self.ecog_grid_checkbox = QtWidgets.QCheckBox("ECoG Grids")
        self.ecog_grid_checkbox.setChecked(self.show_ecog_grids)
        self.ecog_grid_checkbox.setToolTip("Show or hide ECoG grid contacts in the 3D viewer")
        self.ecog_grid_checkbox.stateChanged.connect(self._toggle_ecog_grids)
        surface_layout.addWidget(self.ecog_grid_checkbox)
        self._sync_ecog_grid_checkbox()
        surface_layout.addSpacing(10)
        transparency_label = QtWidgets.QLabel("Transparency:")
        transparency_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        surface_layout.addWidget(transparency_label)

        self.brain_transparency_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brain_transparency_slider.setRange(0, 100)
        self.brain_transparency_slider.setSingleStep(1)
        self.brain_transparency_slider.setFixedWidth(140)
        transparency_pct = int(round((1.0 - float(self.brain_opacity)) * 100))
        self.brain_transparency_slider.setValue(transparency_pct)
        self.brain_transparency_slider.setToolTip("Adjust cortical surface transparency")
        self.brain_transparency_slider.valueChanged.connect(self._on_brain_transparency_changed)
        surface_layout.addWidget(self.brain_transparency_slider)

        self.brain_transparency_value_label = QtWidgets.QLabel(f"{transparency_pct}%")
        self.brain_transparency_value_label.setStyleSheet("QLabel { font-size: 11px; color: #555555; min-width: 36px; }")
        surface_layout.addWidget(self.brain_transparency_value_label)

        surface_layout.addStretch()
        layout.addLayout(surface_layout)

        status_label = QtWidgets.QLabel("Status: Ready")
        status_label.setStyleSheet("QLabel { font-size: 11px; color: #555555; }")
        self.coreg_status_label = status_label
        layout.addWidget(status_label)
        self._init_meg_activity_legend(layout)
        self._init_meg_source_controls(layout)

        file_browser_widget.setLayout(layout)
        return file_browser_widget

    def _loaded_meg_fif_paths(self):
        paths = []
        for path in list(getattr(self, "_meg_loaded_fif_paths", []) or []):
            if not path:
                continue
            try:
                path_abs = str(Path(path).expanduser().resolve())
            except Exception:
                path_abs = str(path)
            if path_abs not in paths:
                paths.append(path_abs)
        current = getattr(self, "_meg_current_fif_path", None)
        if current:
            try:
                current_abs = str(Path(current).expanduser().resolve())
            except Exception:
                current_abs = str(current)
            if current_abs not in paths:
                paths.append(current_abs)
        return paths

    def _has_loaded_meg_fifs(self):
        return bool(self._loaded_meg_fif_paths())

    def _refresh_meg_coreg_fif_combo(self, selected_path=None):
        combo = getattr(self, "meg_coreg_fif_combo", None)
        if combo is None:
            return
        paths = self._loaded_meg_fif_paths()
        if selected_path:
            try:
                selected_path = str(Path(selected_path).expanduser().resolve())
            except Exception:
                selected_path = str(selected_path)
        else:
            selected_path = str(getattr(self, "_meg_current_fif_path", "") or "")

        combo.blockSignals(True)
        combo.clear()
        if not paths:
            combo.addItem("No FIF loaded", "")
        else:
            for path in paths:
                path_obj = Path(path)
                parent = path_obj.parent.name
                label = f"{path_obj.name} ({parent})" if parent else path_obj.name
                combo.addItem(label, path)
            idx = combo.findData(selected_path)
            if idx < 0 and selected_path:
                selected_name = Path(selected_path).name
                for i in range(combo.count()):
                    data = combo.itemData(i)
                    if data and Path(str(data)).name == selected_name:
                        idx = i
                        break
            if idx < 0:
                idx = 0
            combo.setCurrentIndex(idx)
        combo.setEnabled(bool(paths))
        combo.setVisible(bool(paths) and bool(getattr(self, "_meg_alignment_active", False)))
        combo.blockSignals(False)

    def _on_meg_coreg_fif_combo_changed(self, _index):
        combo = getattr(self, "meg_coreg_fif_combo", None)
        if combo is None or not combo.isVisible():
            return
        path = combo.currentData()
        if not path:
            return
        if hasattr(self, "show_meg_coreg_alignment_for_fif"):
            self.show_meg_coreg_alignment_for_fif(path)

    def _hide_meg_coregistration_review(self):
        self._meg_alignment_active = False
        combo = getattr(self, "meg_coreg_fif_combo", None)
        if combo is not None:
            combo.setVisible(False)
        if hasattr(self, "_close_meg_alignment_figure"):
            self._close_meg_alignment_figure()
        if self._plot_widget is not None and self._viewer_right_layout is not None:
            self._populate_viewer_right()
        if hasattr(self, "_sync_meg_alignment_checkbox"):
            self._sync_meg_alignment_checkbox()
        self._sync_coreg_toggle_button()
    
    def toggle_coregistration_panel(self):
        """Toggle visibility of coregistration controls."""
        if self._has_loaded_meg_fifs():
            if self.coreg_panel is not None:
                self.coreg_panel.setVisible(False)
            selected = None
            if hasattr(self, "_selected_meg_coreg_fif_path"):
                selected = self._selected_meg_coreg_fif_path()
            if selected:
                self._meg_current_fif_path = selected
            self.launch_mne_coreg_gui(fif_path=selected)
            self._sync_coreg_toggle_button()
            return

        if not self.coreg_panel:
            return
        showing = not self.coreg_panel.isVisible()
        self.coreg_panel.setVisible(showing)
        if showing and self._viewer_splitter is not None:
            total = self._viewer_splitter.size().width()
            if total <= 0:
                total = 1000
            left = int(total * 0.6)
            right = total - left
            self._viewer_splitter.setSizes([left, right])
        self._sync_coreg_toggle_button()

    def _on_view_combo_changed(self, index):
        """Handle selection from the 3D view dropdown."""
        if index <= 0:
            return
        if not hasattr(self, "view_combo"):
            return
        view_label = self.view_combo.currentText()
        self._set_brain_view(view_label)

    def _sync_subject_view_combo(self):
        combo = getattr(self, "subject_view_combo", None)
        if combo is None:
            return
        has_subject = bool(self.custom_subject and self.custom_subjects_dir)
        combo.blockSignals(True)
        combo.setEnabled(has_subject)
        mode = "fsaverage"
        if has_subject and not self.force_fsaverage:
            mode = "subject"
        idx = combo.findData(mode)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _on_subject_view_combo_changed(self, _index):
        combo = getattr(self, "subject_view_combo", None)
        if combo is None:
            return
        mode = combo.currentData() or "fsaverage"
        self.set_subject_view_mode(mode)
        if self.coreg_panel is not None and hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
            self.coreg_panel._refresh_subject_view_combo()

    def _set_brain_view(self, view_label):
        """Orient the 3D viewer camera to a standard view."""
        if self.brain is None:
            return
        view_map = {
            "Superior": ("dorsal", None),
            "Inferior": ("ventral", None),
            "Anterior": ("rostral", None),
            "Posterior": ("caudal", None),
            "Left": ("lateral", "lh"),
            "Right": ("lateral", "rh"),
        }
        view, hemi = view_map.get(view_label, (None, None))
        if view is None:
            return
        try:
            if hemi is None:
                self.brain.show_view(view)
            else:
                self.brain.show_view(view, hemi=hemi)
        except Exception:
            try:
                self.brain.show_view(view)
            except Exception:
                pass

    def _cache_brain_surface_actors(self):
        """Capture current cortical surface actors so opacity changes only affect the brain mesh."""
        self._brain_surface_actors = []
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return

        plotter = getattr(self.brain._renderer, "plotter", None)
        if plotter is None:
            return

        actor_candidates = []
        actor_sources = [
            getattr(plotter, "actors", None),
            getattr(getattr(plotter, "renderer", None), "actors", None),
        ]
        for source in actor_sources:
            if source is None:
                continue
            if hasattr(source, "values"):
                try:
                    actor_candidates.extend(list(source.values()))
                    continue
                except Exception:
                    pass
            try:
                actor_candidates.extend(list(source))
            except Exception:
                continue

        seen = set()
        for actor in actor_candidates:
            if actor is None:
                continue
            key = id(actor)
            if key in seen:
                continue
            seen.add(key)
            if hasattr(actor, "GetProperty"):
                self._brain_surface_actors.append(actor)

    def _apply_brain_opacity(self, opacity, render=True):
        """Apply cortical surface opacity to cached brain mesh actors."""
        try:
            opacity = float(opacity)
        except Exception:
            return
        opacity = max(0.0, min(1.0, opacity))
        self.brain_opacity = opacity

        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return
        if not self._brain_surface_actors:
            self._cache_brain_surface_actors()

        changed = False
        for actor in list(self._brain_surface_actors):
            try:
                prop = actor.GetProperty()
                if prop is None:
                    continue
                prop.SetOpacity(opacity)
                changed = True
            except Exception:
                continue

        if changed and render:
            try:
                self.brain._renderer.plotter.render()
            except Exception:
                pass

    def _finalize_brain_scene(self, reset_camera=True):
        """Apply consistent mesh visibility and camera settings after Brain creation."""
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return
        self._cache_brain_surface_actors()
        self._apply_brain_opacity(self.brain_opacity, render=False)

        view_label = "Left"
        combo = getattr(self, "view_combo", None)
        if combo is not None:
            try:
                current = str(combo.currentText() or "").strip()
            except Exception:
                current = ""
            if current and current != "View":
                view_label = current
        self._set_brain_view(view_label)

        plotter = getattr(self.brain._renderer, "plotter", None)
        if plotter is None:
            return
        if reset_camera:
            try:
                plotter.reset_camera()
            except Exception:
                pass
            try:
                plotter.reset_camera_clipping_range()
            except Exception:
                pass
        self._render_plotter(plotter)

    def _on_brain_transparency_changed(self, value):
        """Handle UI changes for cortical transparency."""
        try:
            value = int(value)
        except Exception:
            return
        value = max(0, min(100, value))
        if self.brain_transparency_value_label is not None:
            self.brain_transparency_value_label.setText(f"{value}%")
        opacity = (100 - value) / 100.0
        self._apply_brain_opacity(opacity, render=True)

    def _surface_files_available(self, surface_name):
        """Check if the requested FreeSurfer surface files are available."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        if not subject_id or not subjects_dir:
            return False
        surf_dir = Path(subjects_dir) / subject_id / "surf"
        return (surf_dir / f"lh.{surface_name}").exists() and (surf_dir / f"rh.{surface_name}").exists()

    def _get_surface_vertices(self, surface_name):
        """Load and cache FreeSurfer surface vertices for both hemispheres."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        if not subject_id or not subjects_dir:
            return None
        key = (str(subjects_dir), subject_id, surface_name)
        cache = self._surface_vertex_cache
        if key in cache:
            return cache[key]
        surf_dir = Path(subjects_dir) / subject_id / "surf"
        lh_path = surf_dir / f"lh.{surface_name}"
        rh_path = surf_dir / f"rh.{surface_name}"
        if not (lh_path.exists() and rh_path.exists()):
            return None
        try:
            rr_lh, _ = mne.surface.read_surface(str(lh_path))
            rr_rh, _ = mne.surface.read_surface(str(rh_path))
        except Exception:
            return None
        rr = np.vstack([rr_lh, rr_rh])
        cache[key] = rr
        return rr

    def _get_surface_kdtree(self, surface_name):
        """Return a cached KDTree for a surface."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        if not subject_id or not subjects_dir:
            return None
        key = (str(subjects_dir), subject_id, surface_name)
        cache = self._surface_kdtree_cache
        if key in cache:
            return cache[key]
        rr = self._get_surface_vertices(surface_name)
        if rr is None:
            return None
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return None
        tree = cKDTree(rr)
        cache[key] = tree
        return tree

    def _map_points_to_inflated(self, points):
        """Map MRI-space points to inflated surface using vertex correspondence."""
        white = self._get_surface_vertices("white")
        inflated = self._get_surface_vertices("inflated")
        if white is None or inflated is None or white.shape != inflated.shape:
            return self._project_points_to_surface(points, "inflated")
        tree = self._get_surface_kdtree("white")
        if tree is None:
            return self._project_points_to_surface(points, "inflated")
        _, idx = tree.query(points)
        return inflated[idx]

    def _project_points_to_surface(self, points, surface_name):
        """Project points to the nearest vertex on the given surface."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points
        rr = self._get_surface_vertices(surface_name)
        if rr is None:
            return points
        tree = self._get_surface_kdtree(surface_name)
        if tree is not None:
            _, idx = tree.query(points)
            return rr[idx]
        diff = points[:, np.newaxis, :] - rr[np.newaxis, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        return rr[idx]

    def _sync_surface_toggle(self):
        """Sync the surface checkbox with the current surface selection."""
        if not hasattr(self, "inflated_surface_checkbox"):
            return
        self.inflated_surface_checkbox.blockSignals(True)
        self.inflated_surface_checkbox.setChecked(self.brain_surface == "inflated")
        self.inflated_surface_checkbox.blockSignals(False)

    def _store_ecog_sensor_state(
        self,
        info,
        trans,
        sensor_colors=None,
        sensor_scales=None,
        subject_key=None,
    ):
        """Cache ECoG sensor state so the viewer can toggle it without rebuilding."""
        self._ecog_sensor_info = info.copy() if hasattr(info, "copy") else info
        self._ecog_sensor_trans = trans.copy() if hasattr(trans, "copy") else trans
        self._ecog_sensor_colors = (
            None if sensor_colors is None else np.array(sensor_colors, copy=True)
        )
        self._ecog_sensor_scales = (
            None if sensor_scales is None else np.array(sensor_scales, copy=True)
        )
        self._ecog_sensor_subject_key = subject_key
        self._sync_ecog_grid_checkbox()

    def _current_brain_subject_key(self):
        """Return a stable identifier for the currently displayed subject."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        return subject_id, str(subjects_dir)

    def _ecog_grids_available(self):
        """Whether the cached ECoG grid state matches the active brain view."""
        if self.mode != "ecog" or self._ecog_sensor_info is None:
            return False
        if self._ecog_sensor_subject_key is None:
            return True
        try:
            return self._ecog_sensor_subject_key == self._current_brain_subject_key()
        except Exception:
            return False

    def _sync_ecog_grid_checkbox(self):
        """Sync the ECoG grid checkbox with availability and preference state."""
        chk = getattr(self, "ecog_grid_checkbox", None)
        if chk is None:
            return

        available = self._ecog_grids_available()
        chk.blockSignals(True)
        chk.setChecked(bool(self.show_ecog_grids))
        chk.setEnabled(available)
        if self.mode != "ecog":
            chk.setToolTip("ECoG grid controls are only available in ECoG mode")
        elif available:
            chk.setToolTip("Show or hide ECoG grid contacts in the 3D viewer")
        else:
            chk.setToolTip("ECoG grids are only available for the matching fsaverage view")
        chk.blockSignals(False)

    def _render_ecog_grids(self):
        """Show or hide cached ECoG grid actors in the current scene."""
        if self.brain is None:
            self._sync_ecog_grid_checkbox()
            return

        try:
            self.brain.remove_sensors(kind="ecog")
        except Exception:
            pass

        if self.show_ecog_grids and self._ecog_grids_available():
            try:
                self.brain.add_sensors(
                    self._ecog_sensor_info,
                    trans=self._ecog_sensor_trans,
                    meg=False,
                    eeg=False,
                    fnirs=False,
                    ecog=True,
                    seeg=False,
                    dbs=False,
                    sensor_colors=self._ecog_sensor_colors,
                    sensor_scales=self._ecog_sensor_scales,
                )
            except Exception as exc:
                print(f"Could not render ECoG grids: {exc}")

        self._sync_ecog_grid_checkbox()
        self._render_plotter(self.brain._renderer.plotter)

    def _toggle_ecog_grids(self, state):
        """Handle UI toggles for ECoG grid visibility."""
        self.show_ecog_grids = bool(state)
        self._render_ecog_grids()

    def _toggle_inflated_surface(self, state):
        """Toggle between inflated and default surface in the 3D viewer."""
        target_surface = "inflated" if state == QtCore.Qt.Checked else "pial"
        if target_surface == self.brain_surface:
            return
        if not self._surface_files_available(target_surface):
            QtWidgets.QMessageBox.warning(
                None,
                "Surface Not Found",
                f"Surface files for '{target_surface}' were not found for the current subject."
            )
            self._sync_surface_toggle()
            return
        self._reload_brain_with_subject(surface=target_surface)
    
    def _get_current_view(self):
        """Get the current view label from the view combo."""
        combo = getattr(self, "view_combo", None)
        if combo is not None:
            try:
                current = str(combo.currentText() or "").strip()
                if current and current != "View":
                    return current
            except Exception:
                pass
        return "Left"  # Default view
    
    
    def browse_mri_file(self):
        """Open file browser to select MRI file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select MRI File",
            "",
            "NIfTI Files (*.nii *.nii.gz);;MGZ Files (*.mgz);;All Files (*.*)"
        )
        if file_path:
            self.mri_path = file_path
            if self.coreg_panel is not None:
                self.coreg_panel.mri_path = file_path
                if hasattr(self.coreg_panel, "mri_path_display"):
                    self.coreg_panel.mri_path_display.setText(file_path)
                if hasattr(self.coreg_panel, "_sync_output_dir"):
                    self.coreg_panel._sync_output_dir()
            self.update_button_states()
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: MRI loaded - {os.path.basename(file_path)}")
            
            # Display MRI in 3D viewer
            # self._display_patient_volumes()
    
    def browse_ct_file(self):
        """Open file browser to select postoperative CT file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Postoperative CT File",
            "",
            "NIfTI Files (*.nii *.nii.gz);;DICOM Files (*.dcm);;All Files (*.*)"
        )
        if file_path:
            self.ct_path = file_path
            if self.coreg_panel is not None:
                self.coreg_panel.ct_path = file_path
                if hasattr(self.coreg_panel, "ct_path_display"):
                    self.coreg_panel.ct_path_display.setText(file_path)
            self.update_button_states()
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: CT loaded - {os.path.basename(file_path)}")
            
            # Display CT in 3D viewer (unregistered)
            # self._display_patient_volumes()
    
    def _prompt_fastsurfer_reconstruction(self):
        """Ask user if they want to run FastSurfer for surface reconstruction."""
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Surface Reconstruction Options")
        msg_box.setIcon(QtWidgets.QMessageBox.Question)
        
        msg_text = (
            "Would you like to use FreeSurfer/FastSurfer surface reconstruction?\n\n"
            "FastSurfer is a deep learning-based alternative to FreeSurfer that:\n"
            "• Creates high-quality cortical surface meshes from MRI\n"
            "• Performs brain segmentation and parcellation\n"
            "• Runs 10-100x faster than FreeSurfer (minutes vs hours)\n"
            "• Produces FreeSurfer-compatible outputs\n\n"
            "Options:\n"
            "• Run FastSurfer: Start new surface reconstruction (5-20 min)\n"
            "• Browse Existing: Use previously generated FreeSurfer/FastSurfer outputs\n"
            "• Maybe Later: Skip for now\n\n"
            "Requirements for FastSurfer:\n"
            "• FastSurfer installation (Docker or local)\n"
            "• GPU recommended but not required"
        )
        msg_box.setText(msg_text)
        
        run_btn = msg_box.addButton("Run FastSurfer", QtWidgets.QMessageBox.AcceptRole)
        browse_btn = msg_box.addButton("Browse Existing", QtWidgets.QMessageBox.ActionRole)
        info_btn = msg_box.addButton("More Info", QtWidgets.QMessageBox.HelpRole)
        later_btn = msg_box.addButton("Maybe Later", QtWidgets.QMessageBox.RejectRole)
        
        msg_box.setDefaultButton(later_btn)
        msg_box.exec_()
        
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == run_btn:
            self._run_fastsurfer()
        elif clicked_button == browse_btn:
            self._browse_existing_surfaces()
        elif clicked_button == info_btn:
            self._show_fastsurfer_info()
    
    def _browse_existing_surfaces(self):
        """Browse for existing FreeSurfer/FastSurfer reconstruction outputs."""
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Browse FreeSurfer/FastSurfer Outputs")
        msg_box.setIcon(QtWidgets.QMessageBox.Question)
        
        msg_text = (
            "Select what type of FreeSurfer/FastSurfer data you want to load:\n\n"
            "• Subjects Directory: Browse to a FreeSurfer/FastSurfer subjects directory\n"
            "  (contains multiple subject folders with recon-all outputs)\n\n"
            "• Single Subject: Browse to a single subject's reconstruction folder\n"
            "  (e.g., /path/to/subjects_dir/subject_name)\n\n"
            "FreeSurfer/FastSurfer directory structure should contain:\n"
            "  - surf/ folder (surface meshes)\n"
            "  - mri/ folder (volumes)\n"
            "  - label/ folder (parcellations)"
        )
        msg_box.setText(msg_text)
        
        subjects_dir_btn = msg_box.addButton("Browse Subjects Directory", QtWidgets.QMessageBox.ActionRole)
        single_subject_btn = msg_box.addButton("Browse Single Subject", QtWidgets.QMessageBox.ActionRole)
        cancel_btn = msg_box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        
        msg_box.setDefaultButton(cancel_btn)
        msg_box.exec_()
        
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == subjects_dir_btn:
            self._browse_subjects_directory()
        elif clicked_button == single_subject_btn:
            self._browse_single_subject()
    
    def _browse_subjects_directory(self, start_dir=None):
        """Browse for FreeSurfer/FastSurfer subjects directory."""
        subjects_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select FreeSurfer/FastSurfer Subjects Directory",
            start_dir or os.path.expanduser("~"),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if not subjects_dir:
            return
        
        # List available subjects in the directory
        try:
            potential_subjects = []
            for item in os.listdir(subjects_dir):
                subject_path = os.path.join(subjects_dir, item)
                if os.path.isdir(subject_path):
                    # Check if it looks like a FreeSurfer subject
                    surf_dir = os.path.join(subject_path, 'surf')
                    mri_dir = os.path.join(subject_path, 'mri')
                    if os.path.exists(surf_dir) and os.path.exists(mri_dir):
                        potential_subjects.append(item)
            
            if not potential_subjects:
                QtWidgets.QMessageBox.warning(
                    None,
                    "No Subjects Found",
                    f"No valid FreeSurfer/FastSurfer subjects found in:\n{subjects_dir}\n\n"
                    "Make sure the directory contains subject folders with 'surf' and 'mri' subdirectories."
                )
                return
            
            # Let user select a subject
            subject_id, ok = QtWidgets.QInputDialog.getItem(
                None,
                "Select Subject",
                f"Found {len(potential_subjects)} subject(s). Select one:",
                potential_subjects,
                0,
                False
            )
            
            if ok and subject_id:
                self._load_freesurfer_subject(subjects_dir, subject_id)
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                "Error",
                f"Error reading subjects directory:\n{str(e)}"
            )
    
    def _browse_single_subject(self):
        """Browse for a single FreeSurfer/FastSurfer subject folder."""
        subject_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select FreeSurfer/FastSurfer Subject Folder",
            os.path.expanduser("~"),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if not subject_dir:
            return
        
        # Validate it's a FreeSurfer subject
        surf_dir = os.path.join(subject_dir, 'surf')
        mri_dir = os.path.join(subject_dir, 'mri')
        
        if not (os.path.exists(surf_dir) and os.path.exists(mri_dir)):
            QtWidgets.QMessageBox.warning(
                None,
                "Invalid Subject Folder",
                f"The selected folder does not appear to be a valid FreeSurfer/FastSurfer subject.\n\n"
                f"Selected: {subject_dir}\n\n"
                "A valid subject folder should contain:\n"
                "  - surf/ subdirectory (surface meshes)\n"
                "  - mri/ subdirectory (volumes)"
            )
            return
        
        # Extract subject ID and subjects directory
        subject_id = os.path.basename(subject_dir)
        subjects_dir = os.path.dirname(subject_dir)
        
        self._load_freesurfer_subject(subjects_dir, subject_id)
    
    def _load_freesurfer_subject(self, subjects_dir, subject_id):
        """Load a FreeSurfer/FastSurfer subject for use in visualization."""
        try:
            # Verify the subject has required files
            subject_path = os.path.join(subjects_dir, subject_id)
            surf_dir = os.path.join(subject_path, 'surf')
            
            # Check for key surface files
            required_surfaces = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
            missing_surfaces = []
            for surf in required_surfaces:
                if not os.path.exists(os.path.join(surf_dir, surf)):
                    missing_surfaces.append(surf)
            
            if missing_surfaces:
                reply = QtWidgets.QMessageBox.question(
                    None,
                    "Missing Surface Files",
                    f"Some expected surface files are missing:\n{', '.join(missing_surfaces)}\n\n"
                    "Do you want to continue anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No:
                    return
            
            # Store the subject information
            self.custom_subjects_dir = subjects_dir
            self.custom_subject = subject_id
            self.force_fsaverage = False
            if normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg")) in HCP_MMP_PARCS:
                self.label_parcellation = "aparc+aseg"
                self.active_labels = []
            self._subject_to_fsaverage_trans = None
            self._subject_to_fsaverage_key = None
            self._subject_to_fsaverage_warned = False
            self._sync_subject_view_combo()
            if self.coreg_panel is not None:
                if hasattr(self.coreg_panel, "fs_subject_path_display"):
                    subject_path = os.path.join(subjects_dir, subject_id)
                    self.coreg_panel.fs_subject_path_display.setText(subject_path)
                    self.coreg_panel.fs_subject_path = subject_path
                if hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
                    self.coreg_panel._refresh_subject_view_combo()
                if hasattr(self.coreg_panel, "_update_mni_button_state"):
                    self.coreg_panel._update_mni_button_state()
            
            # Update UI
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(
                    f"Status: FreeSurfer/FastSurfer subject loaded - {subject_id}"
                )
            
            print(f"\n✓ Loaded FreeSurfer/FastSurfer subject:")
            print(f"  Subject ID: {subject_id}")
            print(f"  Subjects Dir: {subjects_dir}")
            print(f"  Subject Path: {subject_path}")
            
            # Reload the 3D brain viewer with the new subject
            try:
                print("  Reloading 3D viewer with custom subject...")
                self._reload_brain_with_subject()
                if not self.detected_electrodes and self.coreg_panel is not None:
                    self.detected_electrodes = list(getattr(self.coreg_panel, "before_electrodes", []))
                    native_space = getattr(self.coreg_panel, "electrode_space", None)
                    display_space = getattr(self.coreg_panel, "electrode_display_space", None)
                    if native_space:
                        self.detected_electrode_native_space = native_space
                    if display_space:
                        self.detected_electrode_space = display_space
                if self.detected_electrodes:
                    self._display_detected_electrodes(
                        self.detected_electrodes, self.detected_electrode_space
                    )
                if self.coreg_panel is not None and hasattr(self.coreg_panel, "_update_mni_button_state"):
                    self.coreg_panel._update_mni_button_state()
                
                QtWidgets.QMessageBox.information(
                    None,
                    "Subject Loaded",
                    f"FreeSurfer/FastSurfer subject loaded successfully!\n\n"
                    f"Subject: {subject_id}\n"
                    f"Location: {subjects_dir}\n\n"
                    "The 3D viewer has been updated with the custom brain surfaces."
                )
            except Exception as reload_error:
                print(f"  Warning: Could not reload 3D viewer: {reload_error}")
                QtWidgets.QMessageBox.warning(
                    None,
                    "Subject Loaded (Viewer Not Updated)",
                    f"FreeSurfer/FastSurfer subject loaded successfully!\n\n"
                    f"Subject: {subject_id}\n"
                    f"Location: {subjects_dir}\n\n"
                    "Note: The 3D viewer could not be automatically updated.\n"
                    f"Reason: {str(reload_error)}\n\n"
                    "The subject is loaded and will be used for future visualizations."
                )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                "Error Loading Subject",
                f"Error loading FreeSurfer/FastSurfer subject:\n{str(e)}"
            )
    
    def _find_freesurfer_subjects_dir(self):
        """Return a likely FreeSurfer SUBJECTS_DIR if available."""
        subjects_dir = os.environ.get('SUBJECTS_DIR')
        if subjects_dir and os.path.isdir(subjects_dir):
            return subjects_dir
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if freesurfer_home:
            candidate = os.path.join(freesurfer_home, 'subjects')
            if os.path.isdir(candidate):
                return candidate
        candidate = os.path.expanduser('~/freesurfer/subjects')
        if os.path.isdir(candidate):
            return candidate
        return None

    def _prompt_freesurfer_browse_if_available(self):
        """Prompt user to browse FreeSurfer subjects if a directory is available."""
        subjects_dir = self._find_freesurfer_subjects_dir()
        if not subjects_dir:
            return False
        reply = QtWidgets.QMessageBox.question(
            None,
            "FreeSurfer Subjects Found",
            f"FreeSurfer subjects directory found:\n{subjects_dir}\n\n"
            "Would you like to browse and load a subject now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._browse_subjects_directory(start_dir=subjects_dir)
            return True
        return False

    def _reload_brain_with_subject(self, surface=None):
        """Reload the 3D brain viewer with the current custom subject."""
        if surface:
            self.brain_surface = surface
        if self.brain is None:
            print("  No brain viewer to reload - initializing from scratch")
            # If no brain exists yet, just initialize the 3D layout
            self.init_3d_layout()
            return
        
        # Teardown existing brain viewer
        print("  Cleaning up existing brain viewer...")
        self._teardown_renderer()
        self.brain = None
        
        # Recreate the brain with new subject
        print("  Creating new brain viewer with custom subject...")
        subject_id, subjects_dir = self._get_brain_subject_info()
        
        self.brain = mne.viz.Brain(
            subject=subject_id,
            subjects_dir=subjects_dir,
            cortex=self.brain_cortex_style,
            alpha=self.brain_opacity,
            background="white",
            surf=self.brain_surface,
            silhouette=False,
            show=False,
        )
        self._finalize_brain_scene(reset_camera=True)
        
        # Reinitialize segmentation labels with custom subject
        parc = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        if parc not in HCP_MMP_PARCS:
            print("  Loading segmentation labels for custom subject...")
            self._init_seg_labels_for_subject(subject_id, subjects_dir)
        else:
            try:
                self._prepare_label_names_for_current_parcellation()
                self._populate_label_selection_combos()
                self._render_active_labels()
            except Exception as exc:
                print(f"  Warning: Could not load surface labels for {parc}: {exc}")
        self._classify_detected_electrodes_tissue()
        self._update_classify_button_state()
        
        # Re-embed in Qt widget (similar to init_3d_layout)
        if self.chs_3d_view is not None:
            print("  Embedding brain viewer in Qt widget...")
            self._plot_widget = self.brain._renderer.plotter.interactor
            self._plot_widget.setParent(None)
            self._ensure_coreg_panel()
            self._populate_viewer_right()

            if self.coreg_panel is not None:
                self.coreg_panel.mri_path = self.mri_path
                self.coreg_panel.ct_path = self.ct_path
                if self.mri_path and hasattr(self.coreg_panel, "mri_path_display"):
                    self.coreg_panel.mri_path_display.setText(self.mri_path)
                if self.ct_path and hasattr(self.coreg_panel, "ct_path_display"):
                    self.coreg_panel.ct_path_display.setText(self.ct_path)

            if self.detected_electrodes:
                self._display_detected_electrodes(
                    self.detected_electrodes, self.detected_electrode_space
                )
            self._render_ecog_grids()
            if self._meg_sensor_info is not None:
                self._update_meg_activity_legend(activity_scores=self._meg_activity_scores)
                self._render_meg_sensors(activity_scores=self._meg_activity_scores)
                self._update_meg_source_controls_enabled()
                last_stc = getattr(self, "_meg_last_stc", None)
                if last_stc is not None and hasattr(self, "display_meg_stc_on_brain"):
                    try:
                        self.display_meg_stc_on_brain(
                            last_stc,
                            time_sec=getattr(self, "_meg_last_stc_time_sec", None),
                        )
                    except Exception as exc:
                        print(f"MEG source redisplay warning: {exc}")
                if self._meg_source_auto and self._meg_source_window_data_si is not None:
                    self.project_cached_meg_window_to_source()
            if hasattr(self, "inflated_surface_checkbox"):
                self._sync_surface_toggle()
            self._sync_ecog_grid_checkbox()
            if self.brain_transparency_slider is not None:
                self.brain_transparency_slider.blockSignals(True)
                transparency_pct = int(round((1.0 - float(self.brain_opacity)) * 100))
                self.brain_transparency_slider.setValue(transparency_pct)
                self.brain_transparency_slider.blockSignals(False)
            if self.brain_transparency_value_label is not None:
                transparency_pct = int(round((1.0 - float(self.brain_opacity)) * 100))
                self.brain_transparency_value_label.setText(f"{transparency_pct}%")
        
        print(f"  ✓ Brain viewer reloaded with subject: {subject_id}")

    def set_subject_view_mode(self, mode):
        """Switch between custom subject and fsaverage for visualization."""
        use_fsaverage = (mode == "fsaverage")
        if use_fsaverage == self.force_fsaverage:
            return
        if not use_fsaverage and not (self.custom_subject and self.custom_subjects_dir):
            return
        if not use_fsaverage and normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg")) in HCP_MMP_PARCS:
            self.label_parcellation = "aparc+aseg"
            self.active_labels = []
        self.force_fsaverage = use_fsaverage
        try:
            self._reload_brain_with_subject()
            if hasattr(self, "coreg_status_label"):
                label = "fsaverage" if use_fsaverage else self.custom_subject
                self.coreg_status_label.setText(f"Status: Using {label} for visualization")
            self._sync_subject_view_combo()
            if self.coreg_panel is not None and hasattr(self.coreg_panel, "_refresh_subject_view_combo"):
                self.coreg_panel._refresh_subject_view_combo()
        except Exception as exc:
            print(f"Failed to switch brain subject: {exc}")
    
    def _run_fastsurfer(self):
        """Launch FastSurfer surface reconstruction process."""
        # Check if FastSurfer is installed
        import subprocess
        import shutil
        
        # Check for Docker installation (recommended method)
        has_docker = shutil.which('docker') is not None
        
        # Check for local FastSurfer installation
        fastsurfer_home = os.environ.get('FASTSURFER_HOME')
        has_local_fastsurfer = fastsurfer_home and os.path.exists(os.path.join(fastsurfer_home, 'run_fastsurfer.sh'))
        
        if not has_docker and not has_local_fastsurfer:
            QtWidgets.QMessageBox.warning(
                None,
                "FastSurfer Not Found",
                "FastSurfer installation not detected.\n\n"
                "Please install FastSurfer using one of these methods:\n\n"
                "1. Docker (Recommended):\n"
                "   docker pull deepmi/fastsurfer:latest\n\n"
                "2. Local Installation:\n"
                "   git clone https://github.com/Deep-MI/FastSurfer.git\n"
                "   Set FASTSURFER_HOME environment variable\n\n"
                "See: https://github.com/Deep-MI/FastSurfer"
            )
            return
        
        # Prompt for output directory
        default_output = str(Path(self.mri_path).parent / "fastsurfer_output")
        output_dir, ok = QtWidgets.QInputDialog.getText(
            None,
            "FastSurfer Output Directory",
            "Enter output directory path:",
            QtWidgets.QLineEdit.Normal,
            default_output
        )
        
        if not ok or not output_dir:
            return
        
        # Prompt for subject ID
        default_subject_id = Path(self.mri_path).stem.replace('.nii', '')
        subject_id, ok = QtWidgets.QInputDialog.getText(
            None,
            "Subject ID",
            "Enter subject ID:",
            QtWidgets.QLineEdit.Normal,
            default_subject_id
        )
        
        if not ok or not subject_id:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Show progress dialog
        progress_dialog = QtWidgets.QProgressDialog(
            "Running FastSurfer surface reconstruction...\n"
            "This may take 5-20 minutes depending on your hardware.\n\n"
            "Check terminal for detailed progress.",
            "Cancel",
            0, 0,
            None
        )
        progress_dialog.setWindowTitle("FastSurfer Processing")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.show()
        
        # Construct FastSurfer command
        if has_docker:
            # Docker command
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{os.path.abspath(os.path.dirname(self.mri_path))}:/input',
                '-v', f'{os.path.abspath(output_dir)}:/output',
                '--user', f'{os.getuid()}:{os.getgid()}',
                'deepmi/fastsurfer:latest',
                '--t1', f'/input/{os.path.basename(self.mri_path)}',
                '--sd', '/output',
                '--sid', subject_id
            ]
            
            # Add GPU flag if available
            try:
                subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 'hello-world'], 
                             capture_output=True, timeout=5)
                cmd.insert(3, '--gpus')
                cmd.insert(4, 'all')
                print("GPU support detected - using GPU acceleration")
            except:
                print("Running on CPU (slower)")
        else:
            # Local installation command
            cmd = [
                os.path.join(fastsurfer_home, 'run_fastsurfer.sh'),
                '--t1', self.mri_path,
                '--sd', output_dir,
                '--sid', subject_id
            ]
        
        print(f"\nRunning FastSurfer command:")
        print(" ".join(cmd))
        print(f"\nOutput directory: {output_dir}")
        print(f"Subject ID: {subject_id}\n")
        
        # Run FastSurfer in background
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream output to console
            for line in process.stdout:
                print(line, end='')
                QtWidgets.QApplication.processEvents()
                
                if progress_dialog.wasCanceled():
                    process.terminate()
                    QtWidgets.QMessageBox.information(
                        None, "Cancelled", "FastSurfer processing was cancelled."
                    )
                    return
            
            process.wait()
            progress_dialog.close()
            
            if process.returncode == 0:
                # Success
                QtWidgets.QMessageBox.information(
                    None,
                    "FastSurfer Complete",
                    f"Surface reconstruction completed successfully!\n\n"
                    f"Output directory: {output_dir}\n"
                    f"Subject: {subject_id}\n\n"
                    "You can now use these surfaces for visualization."
                )
                
                # Update custom subject settings
                self.custom_subjects_dir = output_dir
                self.custom_subject = subject_id
                self.coreg_status_label.setText(f"Status: FastSurfer completed - {subject_id}")
            else:
                QtWidgets.QMessageBox.warning(
                    None,
                    "FastSurfer Error",
                    f"FastSurfer exited with error code {process.returncode}.\n"
                    "Check terminal output for details."
                )
        except Exception as e:
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(
                None,
                "FastSurfer Error",
                f"Error running FastSurfer:\n{str(e)}"
            )
    
    def _show_fastsurfer_info(self):
        """Show detailed information about FastSurfer."""
        info_dialog = QtWidgets.QMessageBox()
        info_dialog.setWindowTitle("About FastSurfer")
        info_dialog.setIcon(QtWidgets.QMessageBox.Information)
        
        info_text = (
            "<h3>FastSurfer - Deep Learning Brain Surface Reconstruction</h3>"
            "<p><b>What is FastSurfer?</b><br>"
            "FastSurfer is a state-of-the-art deep learning tool for automated MRI brain "
            "surface reconstruction and segmentation, developed by MIT and MGH.</p>"
            
            "<p><b>Key Features:</b></p>"
            "<ul>"
            "<li>10-100x faster than FreeSurfer (5-20 minutes vs 6-12 hours)</li>"
            "<li>High quality cortical surface meshes</li>"
            "<li>Automated brain segmentation and parcellation</li>"
            "<li>FreeSurfer-compatible output format</li>"
            "<li>Robust to different MRI qualities and contrasts</li>"
            "<li>Works with or without GPU</li>"
            "</ul>"
            
            "<p><b>Installation Options:</b></p>"
            "<ol>"
            "<li><b>Docker (Recommended):</b><br>"
            "<code>docker pull deepmi/fastsurfer:latest</code></li>"
            "<li><b>Local Installation:</b><br>"
            "Clone from: <a href='https://github.com/Deep-MI/FastSurfer'>"
            "https://github.com/Deep-MI/FastSurfer</a></li>"
            "</ol>"
            
            "<p><b>Requirements:</b></p>"
            "<ul>"
            "<li>T1-weighted MRI in NIfTI format (.nii or .nii.gz)</li>"
            "<li>~8GB RAM minimum</li>"
            "<li>GPU recommended (NVIDIA with CUDA) but not required</li>"
            "<li>5-20 minutes processing time (with GPU)</li>"
            "<li>30-60 minutes processing time (CPU only)</li>"
            "</ul>"
            
            "<p><b>Learn More:</b><br>"
            "GitHub: <a href='https://github.com/Deep-MI/FastSurfer'>Deep-MI/FastSurfer</a><br>"
            "Paper: Henschel et al., NeuroImage 2020</p>"
        )
        
        info_dialog.setText(info_text)
        info_dialog.setTextFormat(QtCore.Qt.RichText)
        info_dialog.exec_()
        
        # Ask again if they want to run it
        reply = QtWidgets.QMessageBox.question(
            None,
            "Run FastSurfer?",
            "Would you like to run FastSurfer now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self._run_fastsurfer()
    
    def update_button_states(self):
        """Enable/disable buttons based on file selection."""
        has_mri = self.mri_path is not None
        has_ct = self.ct_path is not None

        if hasattr(self, "coregister_btn"):
            self.coregister_btn.setEnabled(True)
        if hasattr(self, "launch_coreg_gui_btn"):
            self.launch_coreg_gui_btn.setEnabled(has_mri)
        if hasattr(self, "save_trans_btn"):
            self.save_trans_btn.setEnabled(self.trans is not None)
    
    def _display_patient_volumes(self):
        """Display MRI and/or CT volumes in the 3D viewer."""
        if self.brain is None:
            print("3D viewer not initialized")
            return
        
        try:
            import nibabel as nib
            import pyvista as pv
            
            # Remove any existing overlays
            plotter = self.brain._renderer.plotter
            
            # Remove old MRI slice actors
            for attr_name in ['_mri_axial_id', '_mri_coronal_id', '_mri_sagittal_id', '_mri_overlay_id']:
                if hasattr(self, attr_name):
                    try:
                        actor_id = getattr(self, attr_name)
                        if actor_id is not None:
                            plotter.remove_actor(actor_id)
                    except:
                        pass
                    setattr(self, attr_name, None)
                
            # Remove old CT overlay
            if hasattr(self, '_ct_overlay_id'):
                try:
                    if self._ct_overlay_id is not None:
                        plotter.remove_actor(self._ct_overlay_id)
                except:
                    pass
                self._ct_overlay_id = None
            
            # Display MRI if available
            if self.mri_path:
                print(f"Loading MRI for display: {self.mri_path}")
                try:
                    mri_img = nib.load(self.mri_path)
                    mri_data = mri_img.get_fdata()
                    
                    print(f"  MRI shape: {mri_img.shape}")
                    print(f"  MRI data range: [{mri_data.min():.1f}, {mri_data.max():.1f}]")
                    
                    # Create volume visualization for MRI (grayscale)
                    # Use middle slices for visualization
                    self._add_mri_slices(mri_data, mri_img.affine)
                    
                except Exception as e:
                    print(f"Error loading MRI: {e}")
            
            # Display CT if available  
            if self.ct_path:
                # Use registered CT if available, otherwise use original
                ct_to_display = self.ct_registered_path if hasattr(self, 'ct_registered_path') and self.ct_registered_path else self.ct_path
                
                print(f"Loading CT for display: {ct_to_display}")
                try:
                    ct_img = nib.load(str(ct_to_display))
                    ct_data = ct_img.get_fdata()
                    
                    print(f"  CT shape: {ct_img.shape}")
                    print(f"  CT data range: [{ct_data.min():.1f}, {ct_data.max():.1f}]")
                    
                    # Threshold CT to show only high-intensity regions (electrodes/bone)
                    threshold = ct_data.mean() + 2 * ct_data.std()
                    ct_mask = ct_data > threshold
                    
                    print(f"  CT threshold: {threshold:.1f}")
                    print(f"  Voxels above threshold: {ct_mask.sum()}")
                    
                    # Visualize high-intensity CT points (electrodes)
                    self._add_ct_points(ct_data, ct_img.affine, threshold)
                    
                    if ct_to_display == self.ct_registered_path:
                        status_msg = "registered CT displayed (yellow points)"
                    else:
                        status_msg = "unregistered CT displayed (may not align)"
                    
                    current_status = self.coreg_status_label.text()
                    if "complete" in current_status.lower() or "loaded" in current_status.lower():
                        self.coreg_status_label.setText(f"{current_status} - {status_msg}")
                    
                except Exception as e:
                    print(f"Error loading CT: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Force refresh of the 3D viewer
            if hasattr(self.brain, '_renderer'):
                try:
                    self.brain._renderer.plotter.render()
                except:
                    pass
                    
        except Exception as e:
            print(f"Error displaying patient volumes: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_mri_slices(self, mri_data, affine):
        """Add MRI volume visualization to the 3D viewer using PyVista.
        
        This adds orthogonal slices (axial, coronal, sagittal) to the 3D viewer
        without requiring FreeSurfer's recon-all processing.
        """
        try:
            import pyvista as pv
            
            print("  Adding MRI volume visualization...")
            
            # Normalize MRI data to 0-255 for better visualization
            mri_normalized = mri_data.copy()
            mri_min, mri_max = np.percentile(mri_data, [2, 98])  # Robust normalization
            mri_normalized = np.clip((mri_data - mri_min) / (mri_max - mri_min) * 255, 0, 255)
            
            # Create PyVista ImageData (uniform grid) from the MRI
            # Note: PyVista expects data in (nx, ny, nz) order
            grid = pv.ImageData()
            grid.dimensions = np.array(mri_data.shape) + 1  # cell data requires n+1 points
            
            # Extract spacing and origin from affine matrix
            # Affine format: [[sx*r11, sy*r12, sz*r13, tx],
            #                 [sx*r21, sy*r22, sz*r23, ty],
            #                 [sx*r31, sy*r32, sz*r33, tz],
            #                 [0,      0,      0,      1 ]]
            spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
            origin = affine[:3, 3]
            
            grid.spacing = spacing
            grid.origin = origin
            
            # Add MRI data as cell data
            grid.cell_data["values"] = mri_normalized.flatten(order="F")  # Fortran order for correct orientation
            
            # Get the plotter
            plotter = self.brain._renderer.plotter
            
            # Add orthogonal slices at the center of the volume
            center_idx = np.array(mri_data.shape) // 2
            
            # Axial slice (horizontal, through Z)
            axial_slice = grid.slice(normal='z', origin=grid.center)
            self._mri_axial_id = plotter.add_mesh(
                axial_slice,
                cmap='gray',
                opacity=0.8,
                name='mri_axial',
                show_scalar_bar=False
            )
            
            # Coronal slice (front-back, through Y)  
            coronal_slice = grid.slice(normal='y', origin=grid.center)
            self._mri_coronal_id = plotter.add_mesh(
                coronal_slice,
                cmap='gray',
                opacity=0.7,
                name='mri_coronal',
                show_scalar_bar=False
            )
            
            # Sagittal slice (left-right, through X)
            sagittal_slice = grid.slice(normal='x', origin=grid.center)
            self._mri_sagittal_id = plotter.add_mesh(
                sagittal_slice,
                cmap='gray',
                opacity=0.7,
                name='mri_sagittal',
                show_scalar_bar=False
            )
            
            print(f"  ✓ Added MRI orthogonal slices (center: {center_idx})")
            print(f"    - Axial (Z), Coronal (Y), Sagittal (X) slices")
            print(f"    - Volume shape: {mri_data.shape}, Spacing: {spacing}, Origin: {origin}")
            
        except Exception as e:
            print(f"Error adding MRI slices: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_ct_points(self, ct_data, affine, threshold):
        """Add CT high-intensity points (electrodes) to the 3D viewer."""
        try:
            import pyvista as pv
            
            # Get coordinates of high-intensity voxels
            coords = np.argwhere(ct_data > threshold)
            
            if len(coords) == 0:
                print("  No high-intensity points found in CT")
                return
            
            # Limit to a reasonable number of points for visualization
            if len(coords) > 10000:
                print(f"  Downsampling from {len(coords)} to 10000 points")
                indices = np.random.choice(len(coords), 10000, replace=False)
                coords = coords[indices]
            
            # Convert voxel coordinates to world coordinates
            coords_homogeneous = np.concatenate([coords, np.ones((len(coords), 1))], axis=1)
            world_coords = coords_homogeneous @ affine.T
            world_coords = world_coords[:, :3]
            
            if self.brain_surface == "inflated":
                world_coords = self._map_points_to_inflated(world_coords)

            # Create PyVista point cloud
            point_cloud = pv.PolyData(world_coords)
            point_cloud['intensity'] = ct_data[coords[:, 0], coords[:, 1], coords[:, 2]]
            
            # Add to the renderer
            plotter = self.brain._renderer.plotter
            self._ct_overlay_id = plotter.add_points(
                point_cloud,
                color='yellow',
                point_size=3,
                opacity=0.6,
                render_points_as_spheres=True
            )
            
            print(f"  Added {len(coords)} CT points to visualization")
            
        except Exception as e:
            print(f"Error adding CT points: {e}")
            import traceback
            traceback.print_exc()

    def _display_detected_electrodes(self, electrodes, electrode_space):
        """Display detected electrode points in the 3D viewer."""
        if not electrodes:
            print("No detected electrodes to display.")
            return
        self._channel_to_shaft_cache = {}
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            print("3D viewer is not initialized; cannot display electrodes.")
            return

        grouped = any(
            e.shaft not in ("Detected", "Unassigned") for e in electrodes
        )
        if grouped:
            self._display_detected_shafts(electrodes, electrode_space)
        else:
            self._display_detected_electrode_points(electrodes, electrode_space)

    @staticmethod
    def _display_space_uses_mni_world(electrode_space):
        """Whether the 3D viewer should use stored MNI-space mm coordinates."""
        space = str(electrode_space or "").strip().lower()
        return space in {"mni", "mni_tal", "mni tal", "mni_talairach"}

    def _prepare_electrode_world_groups(self, electrodes, electrode_space=None):
        """Convert detected electrodes into world-space groups keyed by shaft."""
        use_mni_world = self._display_space_uses_mni_world(electrode_space)
        voxel_to_world = None if use_mni_world else self._get_electrode_voxel_to_world_trans(
            prefer_subject_mri=bool(self.custom_subject and self.custom_subjects_dir) or self.force_fsaverage
        )
        if not use_mni_world and voxel_to_world is None:
            print("No MRI/CT affine available for electrode display.")
            return None

        groups = {}
        missing_mni = 0
        voxel_to_world = None if voxel_to_world is None else np.asarray(voxel_to_world, dtype=float)
        for elec in electrodes:
            visible = elec.get("visible", True) if hasattr(elec, "get") else getattr(elec, "visible", True)
            shaft_visible = (
                elec.get("shaft_visible", True)
                if hasattr(elec, "get")
                else getattr(elec, "shaft_visible", True)
            )
            if not visible:
                continue
            if not shaft_visible:
                continue

            shaft_name = (
                elec.get("shaft", None)
                if hasattr(elec, "get")
                else getattr(elec, "shaft", None)
            ) or "Unassigned"
            if use_mni_world:
                coords = [elec.get("mni_x"), elec.get("mni_y"), elec.get("mni_z")]
                if any(val is None for val in coords):
                    missing_mni += 1
                    continue
                world = np.asarray(coords, dtype=float)
            else:
                x = elec.get("x")
                y = elec.get("y")
                z = elec.get("z")
                if x is None or y is None or z is None:
                    continue
                vox = np.array([float(x), float(y), float(z), 1.0], dtype=float)
                world = (vox @ voxel_to_world.T)[:3]
                if self.force_fsaverage:
                    world = self._transform_world_for_fsaverage(world)[0]

            groups.setdefault(shaft_name, []).append(world)

        if use_mni_world and missing_mni:
            print(f"Skipped {missing_mni} electrodes without stored MNI coordinates.")
        return {
            shaft_name: np.asarray(points, dtype=float)
            for shaft_name, points in groups.items()
            if points
        }

    def transform_detected_electrodes_to_mni(self, electrodes=None, update_display=True):
        """Compute and store MNI Talairach coordinates for detected electrodes."""
        target_electrodes = list(electrodes if electrodes is not None else self.detected_electrodes)
        if not target_electrodes:
            raise RuntimeError("No electrodes are available to transform.")

        voxel_to_subject_mri = self._get_loaded_mri_voxel_to_subject_mri_trans()
        if voxel_to_subject_mri is None:
            raise RuntimeError(
                "Could not derive the loaded MRI voxel -> FreeSurfer MRI transform. "
                "Ensure the loaded MRI matches the FreeSurfer subject."
            )

        talxfm = self._get_subject_to_fsaverage_trans()
        if talxfm is None:
            raise RuntimeError(
                "Load a FreeSurfer/FastSurfer subject with talairach.xfm before transforming to MNI."
            )

        voxel_to_subject_mri = np.asarray(voxel_to_subject_mri, dtype=float)
        count = 0
        for elec in target_electrodes:
            x = elec.get("x")
            y = elec.get("y")
            z = elec.get("z")
            if x is None or y is None or z is None:
                continue

            vox = np.array([float(x), float(y), float(z), 1.0], dtype=float)
            mri_world = (vox @ voxel_to_subject_mri.T)[:3]
            mni_world = mne.transforms.apply_trans(talxfm, mri_world / 1000.0) * 1000.0

            elec["mri_world_x"] = float(mri_world[0])
            elec["mri_world_y"] = float(mri_world[1])
            elec["mri_world_z"] = float(mri_world[2])
            elec["mni_x"] = float(mni_world[0])
            elec["mni_y"] = float(mni_world[1])
            elec["mni_z"] = float(mni_world[2])
            elec["mni_space"] = "mni_tal"
            count += 1

        if count == 0:
            raise RuntimeError("No valid electrode coordinates were available for MRI->MNI conversion.")

        self.detected_electrodes = target_electrodes
        if self.detected_electrode_native_space is None:
            self.detected_electrode_native_space = "registered"
        self.detected_electrode_space = "mni_tal"
        if update_display:
            if self.custom_subject and self.custom_subjects_dir and not self.force_fsaverage:
                self.set_subject_view_mode("fsaverage")
            self._display_detected_electrodes(self.detected_electrodes, self.detected_electrode_space)
        return count

    def _sorted_shaft_names_for_colors(self, electrodes):
        """Return shaft names ordered by numeric suffix (S_#) then alpha."""
        names = sorted({e.shaft for e in electrodes if e.shaft})
        names = [name for name in names if name != "Unassigned"]
        trajectory_orders = {}
        for elec in electrodes:
            shaft_name = getattr(elec, "shaft", None)
            if not shaft_name or shaft_name == "Unassigned":
                continue
            order = elec.get("trajectory_order") if hasattr(elec, "get") else None
            if order is None:
                continue
            current = trajectory_orders.get(shaft_name)
            if current is None or order < current:
                trajectory_orders[shaft_name] = order
        planned = []
        numeric = []
        alpha = []
        for name in names:
            if name in trajectory_orders:
                planned.append((trajectory_orders[name], name))
                continue
            match = re.match(r"^[Ss][-_]?(\d+)$", name)
            if match:
                numeric.append((int(match.group(1)), name))
            else:
                alpha.append(name)
        planned.sort(key=lambda x: (x[0], x[1]))
        numeric.sort(key=lambda x: x[0])
        alpha.sort()
        return [name for _, name in planned] + [name for _, name in numeric] + alpha

    def _shaft_trajectory_label_map(self, electrodes):
        """Return shaft -> worksheet letter label for uniquely labeled shafts."""
        labels_by_shaft = {}
        for elec in electrodes or []:
            shaft_name = getattr(elec, "shaft", None)
            if not shaft_name or shaft_name == "Unassigned":
                continue
            label = elec.get("trajectory_label") if hasattr(elec, "get") else None
            label = str(label or "").strip()
            if not label:
                continue
            labels_by_shaft.setdefault(shaft_name, [])
            if label not in labels_by_shaft[shaft_name]:
                labels_by_shaft[shaft_name].append(label)
        return {
            shaft_name: labels[0]
            for shaft_name, labels in labels_by_shaft.items()
            if len(labels) == 1
        }

    def _shaft_display_label(self, shaft_name, trajectory_labels=None):
        """Return shaft label text with worksheet letter label appended."""
        if shaft_name == "Unassigned":
            return shaft_name
        trajectory_labels = trajectory_labels or {}
        label = str(trajectory_labels.get(shaft_name) or "").strip()
        if not label:
            return shaft_name
        if self._normalize_activity_key(label) in self._normalize_activity_key(shaft_name):
            return shaft_name
        return f"{shaft_name} ({label})"

    def _display_detected_electrode_points(self, electrodes, electrode_space=None):
        """Display detected electrode points grouped by shaft color."""
        try:
            import pyvista as pv
            t0 = time.perf_counter() if PROFILE_MERGE else None

            plotter = self.brain._renderer.plotter
            self._clear_detected_electrode_actors(plotter)
            t_clear = time.perf_counter() if PROFILE_MERGE else None
            self._detected_shaft_cache = None

            shaft_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            shaft_color_map = {"Unassigned": "#777777"}
            colored_names = self._sorted_shaft_names_for_colors(electrodes)
            for idx, name in enumerate(colored_names):
                shaft_color_map[name] = shaft_colors[idx % len(shaft_colors)]

            selected_shaft = getattr(self, "detected_selected_shaft", None)
            coords_by_shaft = self._prepare_electrode_world_groups(electrodes, electrode_space)
            if coords_by_shaft is None:
                return
            t_build = time.perf_counter() if PROFILE_MERGE else None

            total_points = 0
            for shaft_name, world in coords_by_shaft.items():
                if world.size == 0:
                    continue
                if self.brain_surface == "inflated":
                    world = self._map_points_to_inflated(world)
                point_cloud = pv.PolyData(world)
                color = shaft_color_map.get(shaft_name, "#333333")
                if selected_shaft:
                    point_size = 9 if shaft_name == selected_shaft else 5
                    opacity = 1.0 if shaft_name == selected_shaft else 0.35
                else:
                    point_size = 7
                    opacity = 0.9
                actor = plotter.add_points(
                    point_cloud,
                    color=color,
                    point_size=point_size,
                    opacity=opacity,
                    render_points_as_spheres=True
                )
                self._detected_electrode_actors.append(actor)
                total_points += len(world)
            t_mesh = time.perf_counter() if PROFILE_MERGE else None

            self._render_plotter(plotter)
            space_msg = f" ({electrode_space})" if electrode_space else ""
            print(f"Displayed {total_points} detected electrodes{space_msg} in 3D viewer.")
            if PROFILE_MERGE and t0 is not None:
                parts = []
                if t_clear is not None:
                    parts.append(f"clear={t_clear - t0:.3f}s")
                if t_build is not None:
                    parts.append(f"build={t_build - (t_clear or t0):.3f}s")
                if t_mesh is not None:
                    parts.append(f"mesh={t_mesh - (t_build or t0):.3f}s")
                parts.append(f"total={time.perf_counter() - t0:.3f}s")
                print(f"[profile] display_points {', '.join(parts)}")
        except Exception as e:
            print(f"Error displaying detected electrodes: {e}")
            import traceback
            traceback.print_exc()

    def _detected_shafts_signature(self, coords_by_shaft, affine, use_inflated, trajectory_labels=None):
        """Create a stable signature for the detected shaft geometry and labels."""
        sig_parts = [str(self.brain_surface), bool(use_inflated), bool(self.force_fsaverage)]
        trajectory_labels = trajectory_labels or {}
        if isinstance(affine, str):
            sig_parts.append(affine)
        if self.force_fsaverage:
            trans = self._get_subject_to_fsaverage_trans()
            if trans is not None:
                mat = trans["trans"] if isinstance(trans, dict) and "trans" in trans else getattr(trans, "trans", None)
                if mat is not None:
                    sig_parts.extend(np.round(np.asarray(mat).reshape(-1), 6).tolist())
        if affine is not None and not isinstance(affine, str):
            sig_parts.extend(np.round(np.asarray(affine).reshape(-1), 6).tolist())
        for name in sorted(coords_by_shaft.keys()):
            sig_parts.append(name)
            sig_parts.append(trajectory_labels.get(name, ""))
            coords = np.asarray(coords_by_shaft.get(name, []), dtype=float)
            sig_parts.append(len(coords))
            for x, y, z in coords:
                sig_parts.append(round(float(x), 3))
                sig_parts.append(round(float(y), 3))
                sig_parts.append(round(float(z), 3))
        return tuple(sig_parts)

    def _update_detected_shaft_highlight(self, cache, selected_shaft):
        """Update shaft highlight without rebuilding geometry."""
        actors_by_shaft = cache.get("actors_by_shaft", {})
        for shaft_name, bundle in actors_by_shaft.items():
            is_selected = bool(selected_shaft and shaft_name == selected_shaft)
            dim_others = bool(selected_shaft and shaft_name != selected_shaft)
            tube_opacity = 0.7 if is_selected else (0.2 if dim_others else 0.5)
            contact_opacity = 1.0 if is_selected else (0.35 if dim_others else 0.9)
            label_opacity = 1.0 if (not selected_shaft or is_selected) else 0.35

            tube_actor = bundle.get("tube")
            if tube_actor is not None:
                try:
                    tube_actor.GetProperty().SetOpacity(tube_opacity)
                except Exception:
                    pass
            for contact_actor in bundle.get("contacts", []):
                try:
                    contact_actor.GetProperty().SetOpacity(contact_opacity)
                except Exception:
                    pass
            label_actor = bundle.get("label")
            if label_actor is not None:
                try:
                    label_actor.GetProperty().SetOpacity(label_opacity)
                except Exception:
                    pass

        unassigned = cache.get("unassigned")
        if unassigned:
            if selected_shaft:
                opacity = 0.9 if selected_shaft == "Unassigned" else 0.25
                label_opacity = 1.0 if selected_shaft == "Unassigned" else 0.35
            else:
                opacity = 0.8
                label_opacity = 1.0
            point_actor = unassigned.get("points")
            if point_actor is not None:
                try:
                    point_actor.GetProperty().SetOpacity(opacity)
                except Exception:
                    pass
            label_actor = unassigned.get("label")
            if label_actor is not None:
                try:
                    label_actor.GetProperty().SetOpacity(label_opacity)
                except Exception:
                    pass

    @staticmethod
    def _normalize_activity_key(label):
        """Normalize channel/shaft labels for fuzzy matching."""
        if label is None:
            return ""
        text = str(label).strip().lower()
        text = text.replace(" ", "")
        text = text.replace("-", "")
        text = text.replace("_", "")
        text = re.sub(r"[^a-z0-9]", "", text)
        return text

    def _channel_label_candidates(self, channel_label):
        """Create normalized candidate keys from a channel label."""
        raw = str(channel_label or "").strip()
        if not raw:
            return set()

        first_token = re.split(r"[\s_]+", raw)[0]
        base = first_token
        base = re.sub(r"(raw|hires|hi\-res|ref)$", "", base, flags=re.IGNORECASE)

        variants = {raw, first_token, base}
        if base.lower().startswith("m") and len(base) > 1:
            variants.add(base[1:])
        variants.add(re.sub(r"\d+$", "", base))
        variants.add(re.sub(r"\d+$", "", first_token))

        candidates = set()
        for variant in variants:
            norm = self._normalize_activity_key(variant)
            if norm:
                candidates.add(norm)
        return candidates

    def _resolve_shaft_from_channel_label(self, channel_label):
        """Best-effort mapping from channel label to detected shaft name."""
        if not self.detected_electrodes:
            return None

        cache_key = str(channel_label)
        cached = self._channel_to_shaft_cache.get(cache_key)
        if cached:
            for elec in self.detected_electrodes:
                if getattr(elec, "shaft", None) == cached:
                    return cached
            self._channel_to_shaft_cache.pop(cache_key, None)

        shafts = {}
        for elec in self.detected_electrodes:
            shaft = getattr(elec, "shaft", None)
            if not shaft or shaft in ("Detected",):
                continue
            shafts[shaft] = True
        shaft_names = list(shafts.keys())
        if not shaft_names:
            return None

        candidates = self._channel_label_candidates(channel_label)
        if not candidates:
            return None

        normalized_shaft_map = {}
        for shaft_name in shaft_names:
            norm = self._normalize_activity_key(shaft_name)
            if not norm:
                continue
            normalized_shaft_map.setdefault(norm, []).append(shaft_name)

        # Exact normalized match first.
        for candidate in candidates:
            if candidate in normalized_shaft_map:
                match = sorted(normalized_shaft_map[candidate], key=len)[0]
                self._channel_to_shaft_cache[cache_key] = match
                return match

        # Match worksheet labels such as A/B/F1 when shaft names are full trajectory names.
        for elec in self.detected_electrodes:
            shaft = getattr(elec, "shaft", None)
            if not shaft:
                continue
            map_values = []
            if hasattr(elec, "get"):
                map_values = [elec.get("map_shaft"), elec.get("map_contact")]
            for value in map_values:
                value_norm = self._normalize_activity_key(value)
                if not value_norm:
                    continue
                for candidate in candidates:
                    if candidate == value_norm or re.match(rf"^{re.escape(value_norm)}\d+$", candidate):
                        self._channel_to_shaft_cache[cache_key] = shaft
                        return shaft
            trajectory_label = elec.get("trajectory_label") if hasattr(elec, "get") else None
            label_norm = self._normalize_activity_key(trajectory_label)
            if not label_norm:
                continue
            for candidate in candidates:
                if candidate == label_norm or re.match(rf"^{re.escape(label_norm)}\d+$", candidate):
                    self._channel_to_shaft_cache[cache_key] = shaft
                    return shaft

        # Substring fallback for names like "mLFUS07 raw" -> "LFUS".
        for candidate in sorted(candidates, key=len, reverse=True):
            if len(candidate) < 3:
                continue
            for norm_shaft, names in normalized_shaft_map.items():
                if candidate in norm_shaft or norm_shaft in candidate:
                    match = sorted(names, key=len)[0]
                    self._channel_to_shaft_cache[cache_key] = match
                    return match

        # Try matching against manually-assigned electrode names.
        for elec in self.detected_electrodes:
            shaft = getattr(elec, "shaft", None)
            elec_name = getattr(elec, "name", None)
            if not shaft or not elec_name:
                continue
            elec_norm = self._normalize_activity_key(elec_name)
            if not elec_norm:
                continue
            for candidate in candidates:
                if candidate == elec_norm or candidate in elec_norm or elec_norm in candidate:
                    self._channel_to_shaft_cache[cache_key] = shaft
                    return shaft

        return None

    def _apply_playback_shaft_activity(self, cache, normalized_scores, active_shafts):
        """Apply playback activity emphasis to shaft actors in-place."""
        actors_by_shaft = cache.get("actors_by_shaft", {})
        has_active = bool(active_shafts)

        for shaft_name, bundle in actors_by_shaft.items():
            score = float(normalized_scores.get(shaft_name, 0.0))
            is_active = shaft_name in active_shafts

            if has_active:
                tube_opacity = 0.08 + (0.62 * score if is_active else 0.0)
                contact_opacity = 0.12 + (0.88 * score if is_active else 0.0)
                label_opacity = 1.0 if is_active else 0.30
            else:
                tube_opacity = 0.5
                contact_opacity = 0.9
                label_opacity = 1.0

            tube_actor = bundle.get("tube")
            if tube_actor is not None:
                try:
                    tube_actor.GetProperty().SetOpacity(tube_opacity)
                except Exception:
                    pass
            for contact_actor in bundle.get("contacts", []):
                try:
                    contact_actor.GetProperty().SetOpacity(contact_opacity)
                except Exception:
                    pass
            label_actor = bundle.get("label")
            if label_actor is not None:
                try:
                    label_actor.GetProperty().SetOpacity(label_opacity)
                except Exception:
                    pass

        unassigned = cache.get("unassigned")
        if unassigned:
            point_actor = unassigned.get("points")
            label_actor = unassigned.get("label")
            if has_active:
                ua_active = "Unassigned" in active_shafts
                point_opacity = 0.85 if ua_active else 0.12
                label_opacity = 1.0 if ua_active else 0.30
            else:
                point_opacity = 0.8
                label_opacity = 1.0
            if point_actor is not None:
                try:
                    point_actor.GetProperty().SetOpacity(point_opacity)
                except Exception:
                    pass
            if label_actor is not None:
                try:
                    label_actor.GetProperty().SetOpacity(label_opacity)
                except Exception:
                    pass

    def set_playback_contact_activity(self, channel_activity, active_threshold=0.55):
        """Highlight detected shafts based on channel activity during playback.

        Parameters
        ----------
        channel_activity : dict
            Mapping `{channel_label: activity_value}` for currently active channels.
        active_threshold : float
            Normalized threshold in [0, 1] used to mark shafts as active.
        """
        cache = getattr(self, "_detected_shaft_cache", None)
        if not cache or not hasattr(self, "brain") or self.brain is None:
            return
        if not isinstance(channel_activity, dict) or not channel_activity:
            self.clear_playback_contact_activity()
            return

        shaft_scores = {}
        for channel_label, value in channel_activity.items():
            if value is None:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score):
                continue
            shaft_name = self._resolve_shaft_from_channel_label(channel_label)
            if not shaft_name:
                continue
            if shaft_name not in shaft_scores or score > shaft_scores[shaft_name]:
                shaft_scores[shaft_name] = score

        if not shaft_scores:
            self.clear_playback_contact_activity()
            return

        values = np.array(list(shaft_scores.values()), dtype=float)
        if len(values) == 1:
            only_name = next(iter(shaft_scores.keys()))
            normalized_scores = {only_name: 1.0 if values[0] > 0 else 0.0}
        else:
            lo = np.percentile(values, 20)
            hi = np.percentile(values, 90)
            span = max(hi - lo, 1e-6)
            normalized_scores = {
                name: float(np.clip((val - lo) / span, 0.0, 1.0))
                for name, val in shaft_scores.items()
            }

        active_shafts = {
            name for name, norm_val in normalized_scores.items()
            if norm_val >= float(active_threshold)
        }
        if not active_shafts and normalized_scores:
            active_shafts = {max(normalized_scores, key=normalized_scores.get)}

        self._playback_activity_scores = normalized_scores
        self._playback_activity_shafts = set(active_shafts)
        self._apply_playback_shaft_activity(cache, normalized_scores, self._playback_activity_shafts)
        self._render_plotter(self.brain._renderer.plotter)

    def clear_playback_contact_activity(self):
        """Clear playback-driven contact highlighting and restore normal styling."""
        self._playback_activity_scores = {}
        self._playback_activity_shafts = set()

        cache = getattr(self, "_detected_shaft_cache", None)
        if not cache or not hasattr(self, "brain") or self.brain is None:
            return
        self._update_detected_shaft_highlight(cache, getattr(self, "detected_selected_shaft", None))
        self._render_plotter(self.brain._renderer.plotter)

    def _display_detected_shafts(self, electrodes, electrode_space=None):
        """Display detected shafts as tubes with cylindrical contacts."""
        try:
            import pyvista as pv
            from matplotlib import colors as mcolors
            t0 = time.perf_counter() if PROFILE_MERGE else None

            coords_by_shaft = self._prepare_electrode_world_groups(electrodes, electrode_space)
            if coords_by_shaft is None:
                return
            affine = None if self._display_space_uses_mni_world(electrode_space) else self._get_affine_for_electrodes()

            plotter = self.brain._renderer.plotter

            shaft_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            shaft_color_map = {"Unassigned": "#777777"}
            colored_names = self._sorted_shaft_names_for_colors(electrodes)
            for idx, name in enumerate(colored_names):
                shaft_color_map[name] = shaft_colors[idx % len(shaft_colors)]

            selected_shaft = getattr(self, "detected_selected_shaft", None)
            trajectory_labels = self._shaft_trajectory_label_map(electrodes)

            def _mute_color(color, mix=0.6):
                rgb = np.array(mcolors.to_rgb(color))
                muted = rgb * (1.0 - mix) + np.array([0.6, 0.6, 0.6]) * mix
                return tuple(muted.tolist())
            t_build = time.perf_counter() if PROFILE_MERGE else None

            total_contacts = 0
            shaft_radius = 0.4
            contact_radius = 0.8
            default_contact_height = 2.0

            use_inflated = self.brain_surface == "inflated"
            signature_affine = "mni_world" if self._display_space_uses_mni_world(electrode_space) else affine
            signature = self._detected_shafts_signature(
                coords_by_shaft,
                signature_affine,
                use_inflated,
                trajectory_labels,
            )
            cache = getattr(self, "_detected_shaft_cache", None)
            if cache and cache.get("signature") == signature:
                if self._playback_activity_shafts or self._playback_activity_scores:
                    self._apply_playback_shaft_activity(
                        cache,
                        self._playback_activity_scores,
                        self._playback_activity_shafts,
                    )
                else:
                    self._update_detected_shaft_highlight(cache, selected_shaft)
                self._render_plotter(plotter)
                if PROFILE_MERGE and t0 is not None:
                    parts = []
                    if t_build is not None:
                        parts.append(f"build={t_build - t0:.3f}s")
                    parts.append(f"total={time.perf_counter() - t0:.3f}s")
                    print(f"[profile] display_shafts reuse {', '.join(parts)}")
                return

            self._clear_detected_electrode_actors(plotter)
            t_clear = time.perf_counter() if PROFILE_MERGE else None

            world_center = self._get_reference_world_center()
            if world_center is not None:
                world_center = self._transform_world_for_fsaverage(world_center)[0]
            if use_inflated and world_center is not None:
                try:
                    world_center = self._map_points_to_inflated(np.asarray([world_center]))[0]
                except Exception:
                    pass
            actors_by_shaft = {}
            for shaft_name, coords in coords_by_shaft.items():
                if coords is None or len(coords) == 0:
                    continue
                if shaft_name == "Unassigned":
                    continue
                try:
                    is_selected = bool(selected_shaft and shaft_name == selected_shaft)
                    dim_others = bool(selected_shaft and shaft_name != selected_shaft)
                    shaft_radius_use = shaft_radius * (1.35 if is_selected else 1.0)
                    contact_radius_use = contact_radius * (1.35 if is_selected else (0.7 if dim_others else 1.0))
                    tube_opacity = 0.7 if is_selected else (0.2 if dim_others else 0.5)
                    contact_opacity = 1.0 if is_selected else (0.35 if dim_others else 0.9)
                    world = np.asarray(coords, dtype=float)
                    if use_inflated:
                        world = self._map_points_to_inflated(world)

                    center = np.mean(world, axis=0)
                    if len(world) > 1:
                        _, _, vh = np.linalg.svd(world - center, full_matrices=False)
                        axis = vh[0]
                        if np.linalg.norm(axis) == 0:
                            axis = np.array([0.0, 0.0, 1.0])
                    else:
                        axis = np.array([0.0, 0.0, 1.0])
                    axis = axis / max(np.linalg.norm(axis), 1e-6)

                    proj = (world - center) @ axis
                    order = np.argsort(proj)
                    proj_sorted = proj[order]
                    world_sorted = world[order]

                    spacing = default_contact_height
                    if len(proj_sorted) > 1:
                        diffs = np.diff(proj_sorted)
                        diffs = diffs[diffs > 1e-3]
                        if len(diffs) > 0:
                            spacing = float(np.median(diffs))
                    contact_height = max(1.0, min(3.0, spacing * 0.6))

                    line_start = center + axis * np.min(proj_sorted)
                    line_end = center + axis * np.max(proj_sorted)
                    if not np.allclose(line_start, line_end):
                        line = pv.Line(line_start, line_end)
                        if getattr(line, "n_points", 0) > 0:
                            muted_color = _mute_color(shaft_color_map.get(shaft_name, "#777777"))
                            tube = line.tube(radius=shaft_radius_use)
                            if getattr(tube, "n_points", 0) > 0:
                                shaft_actor = plotter.add_mesh(
                                    tube,
                                    color=muted_color,
                                    opacity=tube_opacity
                                )
                                self._detected_shaft_actors.append(shaft_actor)
                            else:
                                shaft_actor = None
                        else:
                            shaft_actor = None
                    else:
                        shaft_actor = None

                    contact_actors = []

                    contact_color = shaft_color_map.get(shaft_name, "#333333")
                    for point in world_sorted:
                        cyl = pv.Cylinder(
                            center=point,
                            direction=axis,
                            radius=contact_radius_use,
                            height=contact_height
                        )
                        contact_actor = plotter.add_mesh(
                            cyl,
                            color=contact_color,
                            opacity=contact_opacity
                        )
                        self._detected_electrode_actors.append(contact_actor)
                        contact_actors.append(contact_actor)
                        total_contacts += 1

                    label_actor = None
                    try:
                        label_offset = max(2.0, contact_height * 1.5)
                        if world_center is not None:
                            dist_start = np.linalg.norm(line_start - world_center)
                            dist_end = np.linalg.norm(line_end - world_center)
                            if dist_start > dist_end:
                                label_pos = line_start - axis * label_offset
                            else:
                                label_pos = line_end + axis * label_offset
                        else:
                            label_pos = line_end + axis * label_offset
                        label_text = self._shaft_display_label(shaft_name, trajectory_labels)
                        label_actor = plotter.add_point_labels(
                            [label_pos],
                            [label_text],
                            text_color=contact_color,
                            font_size=12,
                            point_size=0
                        )
                        self._detected_shaft_label_actors.append(label_actor)
                    except Exception:
                        label_actor = None

                    actors_by_shaft[shaft_name] = {
                        "tube": shaft_actor,
                        "contacts": contact_actors,
                        "label": label_actor,
                    }
                except Exception as exc:
                    print(f"Skipping shaft {shaft_name}: {exc}")
                    continue
            t_mesh = time.perf_counter() if PROFILE_MERGE else None

            # Render any unassigned electrodes as points
            unassigned_bundle = None
            if "Unassigned" in coords_by_shaft:
                world = np.asarray(coords_by_shaft.get("Unassigned", []), dtype=float)
                if world.size > 0:
                    if self.brain_surface == "inflated":
                        world = self._map_points_to_inflated(world)
                    point_cloud = pv.PolyData(world)
                    if selected_shaft:
                        point_size = 8 if selected_shaft == "Unassigned" else 4
                        opacity = 0.9 if selected_shaft == "Unassigned" else 0.25
                    else:
                        point_size = 6
                        opacity = 0.8
                    actor = plotter.add_points(
                        point_cloud,
                        color="#777777",
                        point_size=point_size,
                        opacity=opacity,
                        render_points_as_spheres=True
                    )
                    self._detected_electrode_actors.append(actor)
                    total_contacts += len(world)
                    unassigned_bundle = {"points": actor, "label": None}
                    try:
                        centroid = np.mean(world, axis=0)
                        label_text = f"Unassigned ({len(world)})"
                        label_actor = plotter.add_point_labels(
                            [centroid],
                            [label_text],
                            text_color="#777777",
                            font_size=12,
                            point_size=0
                        )
                        self._detected_shaft_label_actors.append(label_actor)
                        unassigned_bundle["label"] = label_actor
                    except Exception:
                        pass
            t_unassigned = time.perf_counter() if PROFILE_MERGE else None

            self._detected_shaft_cache = {
                "signature": signature,
                "actors_by_shaft": actors_by_shaft,
                "unassigned": unassigned_bundle,
            }
            if self._playback_activity_shafts or self._playback_activity_scores:
                self._apply_playback_shaft_activity(
                    self._detected_shaft_cache,
                    self._playback_activity_scores,
                    self._playback_activity_shafts,
                )
            else:
                self._update_detected_shaft_highlight(
                    self._detected_shaft_cache,
                    getattr(self, "detected_selected_shaft", None),
                )

            self._render_plotter(plotter)
            space_msg = f" ({electrode_space})" if electrode_space else ""
            print(f"Displayed {total_contacts} detected contacts{space_msg} in 3D viewer.")
            if PROFILE_MERGE and t0 is not None:
                parts = []
                if t_clear is not None:
                    parts.append(f"clear={t_clear - t0:.3f}s")
                if t_build is not None:
                    parts.append(f"build={t_build - t0:.3f}s")
                if t_mesh is not None:
                    parts.append(f"mesh={t_mesh - (t_build or t0):.3f}s")
                if t_unassigned is not None:
                    parts.append(f"unassigned={t_unassigned - (t_mesh or t0):.3f}s")
                parts.append(f"total={time.perf_counter() - t0:.3f}s")
                print(f"[profile] display_shafts {', '.join(parts)}")
        except Exception as e:
            print(f"Error displaying detected shafts: {e}")
            import traceback
            traceback.print_exc()

    def _get_affine_for_electrodes(self):
        """Return an affine for mapping voxel electrodes to world coordinates."""
        try:
            import nibabel as nib
            if self.mri_path and os.path.exists(self.mri_path):
                return nib.load(self.mri_path).affine
            if self.ct_registered_path and os.path.exists(self.ct_registered_path):
                return nib.load(str(self.ct_registered_path)).affine
            if self.ct_path and os.path.exists(self.ct_path):
                return nib.load(self.ct_path).affine
        except Exception:
            return None
        return None

    def _get_subject_mri_volume_path(self):
        """Return the FreeSurfer volume used to define subject MRI coordinates."""
        if not (self.custom_subject and self.custom_subjects_dir):
            return None
        subject_mri_dir = Path(self.custom_subjects_dir) / self.custom_subject / "mri"
        for name in ("orig.mgz", "T1.mgz"):
            path = subject_mri_dir / name
            if path.exists():
                return path
        return None

    def _get_loaded_mri_voxel_to_subject_mri_trans(self):
        """Map loaded MRI voxel indices to FreeSurfer subject MRI surface-RAS (mm)."""
        if not (self.mri_path and os.path.exists(self.mri_path)):
            self._loaded_mri_to_subject_mri_trans = None
            self._loaded_mri_to_subject_mri_key = None
            return None
        if not (self.custom_subject and self.custom_subjects_dir):
            self._loaded_mri_to_subject_mri_trans = None
            self._loaded_mri_to_subject_mri_key = None
            return None

        key = (os.path.abspath(self.mri_path), self.custom_subjects_dir, self.custom_subject)
        if key == self._loaded_mri_to_subject_mri_key and self._loaded_mri_to_subject_mri_trans is not None:
            return self._loaded_mri_to_subject_mri_trans

        fs_mri_path = self._get_subject_mri_volume_path()
        self._loaded_mri_to_subject_mri_key = key
        if fs_mri_path is None:
            self._loaded_mri_to_subject_mri_trans = None
            if not self._loaded_mri_to_subject_mri_warned:
                print("No FreeSurfer MRI volume (orig.mgz/T1.mgz) found; cannot align detected voxels to subject MRI coordinates.")
                self._loaded_mri_to_subject_mri_warned = True
            return None

        try:
            import nibabel as nib

            loaded_img = nib.load(self.mri_path)
            fs_img = nib.load(str(fs_mri_path))
            fs_header = fs_img.header

            if not hasattr(fs_header, "get_vox2ras") or not hasattr(fs_header, "get_vox2ras_tkr"):
                raise RuntimeError(f"{fs_mri_path} does not expose FreeSurfer vox2ras transforms.")

            loaded_vox_to_scanner = np.asarray(loaded_img.affine, dtype=float)
            fs_vox_to_scanner = np.asarray(fs_header.get_vox2ras(), dtype=float)
            fs_vox_to_mri = np.asarray(fs_header.get_vox2ras_tkr(), dtype=float)

            self._loaded_mri_to_subject_mri_trans = (
                fs_vox_to_mri @ np.linalg.inv(fs_vox_to_scanner) @ loaded_vox_to_scanner
            )
            return self._loaded_mri_to_subject_mri_trans
        except Exception as exc:
            self._loaded_mri_to_subject_mri_trans = None
            print(f"Failed to derive loaded MRI voxel -> subject MRI transform: {exc}")
            return None

    def _get_electrode_voxel_to_world_trans(self, prefer_subject_mri=False):
        """Return the best available voxel->world transform for detected electrodes."""
        if prefer_subject_mri:
            subject_trans = self._get_loaded_mri_voxel_to_subject_mri_trans()
            if subject_trans is not None:
                return subject_trans
        return self._get_affine_for_electrodes()

    def _get_reference_world_center(self):
        """Return an approximate brain center in world coordinates."""
        try:
            import nibabel as nib
        except Exception:
            return None
        if self.mri_path and os.path.exists(self.mri_path):
            voxel_to_subject_mri = self._get_loaded_mri_voxel_to_subject_mri_trans()
            if voxel_to_subject_mri is not None:
                try:
                    img = nib.load(str(self.mri_path))
                    shape = np.array(img.shape[:3], dtype=float)
                    if shape.size >= 3:
                        center_vox = (shape - 1.0) / 2.0
                        center_world = nib.affines.apply_affine(voxel_to_subject_mri, center_vox)
                        return np.asarray(center_world, dtype=float)
                except Exception:
                    pass
        for path in (self.mri_path, self.ct_registered_path, self.ct_path):
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                img = nib.load(str(path))
                shape = np.array(img.shape[:3], dtype=float)
                if shape.size < 3:
                    continue
                center_vox = (shape - 1.0) / 2.0
                center_world = nib.affines.apply_affine(img.affine, center_vox)
                return np.asarray(center_world, dtype=float)
            except Exception:
                continue
        return None

    def _get_subject_to_fsaverage_trans(self):
        """Return mri->mni_tal transform for the loaded subject if available."""
        if not (self.custom_subject and self.custom_subjects_dir):
            self._subject_to_fsaverage_trans = None
            self._subject_to_fsaverage_key = None
            return None
        key = (self.custom_subjects_dir, self.custom_subject)
        if key == self._subject_to_fsaverage_key and self._subject_to_fsaverage_trans is not None:
            return self._subject_to_fsaverage_trans

        xfm_path = (
            Path(self.custom_subjects_dir)
            / self.custom_subject
            / "mri"
            / "transforms"
            / "talairach.xfm"
        )
        self._subject_to_fsaverage_key = key
        if not xfm_path.exists():
            self._subject_to_fsaverage_trans = None
            if not self._subject_to_fsaverage_warned:
                print(f"No talairach.xfm found at {xfm_path}; cannot map to fsaverage.")
                self._subject_to_fsaverage_warned = True
            return None
        try:
            trans = mne.read_talxfm(self.custom_subject, self.custom_subjects_dir)
        except Exception as exc:
            self._subject_to_fsaverage_trans = None
            print(f"Failed to read talairach.xfm: {exc}")
            return None
        self._subject_to_fsaverage_trans = trans
        return trans

    def _transform_world_for_fsaverage(self, world):
        """Transform world coordinates from subject MRI -> fsaverage (mni_tal)."""
        if not self.force_fsaverage:
            return world
        trans = self._get_subject_to_fsaverage_trans()
        if trans is None:
            return world
        world = np.asarray(world, dtype=float)
        if world.ndim == 1:
            world = world[None, :]
        return mne.transforms.apply_trans(trans, world / 1000.0) * 1000.0

    def _clear_detected_electrode_actors(self, plotter):
        """Remove any previously drawn detected electrode actors."""
        for actor in getattr(self, "_detected_electrode_actors", []):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        for actor in getattr(self, "_detected_shaft_actors", []):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        for actor in getattr(self, "_detected_shaft_label_actors", []):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        self._detected_electrode_actors = []
        self._detected_shaft_actors = []
        self._detected_shaft_label_actors = []
        self._detected_shaft_cache = None
        self._playback_activity_scores = {}
        self._playback_activity_shafts = set()

    def clear_detected_electrodes(self):
        """Clear detected electrodes from the 3D viewer."""
        self.detected_electrodes = []
        self.detected_electrode_space = None
        self.detected_electrode_native_space = None
        self.detected_selected_shaft = None
        self._channel_to_shaft_cache = {}
        self._playback_activity_scores = {}
        self._playback_activity_shafts = set()
        if self.brain is None or not hasattr(self.brain, "_renderer"):
            return
        plotter = self.brain._renderer.plotter
        self._clear_detected_electrode_actors(plotter)
        self._render_plotter(plotter)

    def _render_plotter(self, plotter):
        """Safely render the plotter if possible."""
        try:
            plotter.render()
        except Exception:
            pass

    def _ensure_orientation_widget(self):
        """Ensure the 3D plotter shows an orientation widget."""
        brain = getattr(self, "brain", None)
        renderer = getattr(brain, "_renderer", None) if brain else None
        plotter = getattr(renderer, "plotter", None) if renderer else None
        if plotter is None:
            return
        if (
            getattr(self, "_orientation_widget_plotter", None) is plotter
            and getattr(self, "_orientation_widget", None) is not None
        ):
            return
        self._orientation_widget = None
        self._orientation_widget_plotter = plotter
        try:
            if hasattr(plotter, "add_orientation_widget"):
                actor = None
                try:
                    import pyvista as pv
                    for cls_name in ("AxesActor", "Axes", "AxesAssembly"):
                        cls = getattr(pv, cls_name, None)
                        if cls is not None:
                            actor = cls()
                            break
                except Exception:
                    actor = None
                if actor is not None:
                    self._apply_orientation_labels(actor)
                    try:
                        self._orientation_widget = plotter.add_orientation_widget(
                            actor, viewport=(0.82, 0.02, 0.98, 0.18)
                        )
                    except TypeError:
                        self._orientation_widget = plotter.add_orientation_widget(actor)
                else:
                    try:
                        self._orientation_widget = plotter.add_orientation_widget(
                            viewport=(0.82, 0.02, 0.98, 0.18)
                        )
                    except TypeError:
                        self._orientation_widget = plotter.add_orientation_widget()
            elif hasattr(plotter, "add_axes"):
                try:
                    self._orientation_widget = plotter.add_axes(
                        interactive=False,
                        viewport=(0.82, 0.02, 0.98, 0.18),
                    )
                except TypeError:
                    self._orientation_widget = plotter.add_axes(interactive=False)
        except Exception as exc:
            print(f"Warning: could not add orientation widget: {exc}")
            self._orientation_widget = None

    def _apply_orientation_labels(self, actor):
        """Set RAS labels on an orientation axes actor when supported."""
        label_map = {"X": "R", "Y": "A", "Z": "S"}
        for axis, label in label_map.items():
            applied = False
            for method in (
                f"Set{axis}AxisLabelText",
                f"Set{axis}AxisLabel",
                f"Set{axis}LabelText",
                f"Set{axis}Label",
            ):
                setter = getattr(actor, method, None)
                if callable(setter):
                    try:
                        setter(label)
                        applied = True
                        break
                    except Exception:
                        pass
            if applied:
                continue
            for attr in (
                f"{axis.lower()}_label",
                f"{axis.lower()}_axis_label",
                f"{axis.lower()}_axis_label_text",
            ):
                if hasattr(actor, attr):
                    try:
                        setattr(actor, attr, label)
                        applied = True
                        break
                    except Exception:
                        pass
    
    def perform_coregistration(self):
        """Launch registration dialog with before/after visualization and progress tracking."""
        if not self.coreg_panel:
            return
        self.coreg_panel.setVisible(True)
        self._sync_coreg_toggle_button()
        if not self.mri_path or not self.ct_path:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText("Status: Load MRI and CT in Coregistration UI")

    def _apply_registration_results(self, dialog):
        """Apply registration dialog outputs and prompt for FreeSurfer subject."""
        results = dialog.get_results()
        self.trans = results['trans']
        self.ct_registered_path = results['ct_registered_path']
        self.detected_electrodes = results.get('electrodes', [])
        self.detected_electrode_space = results.get('electrode_space')
        self.detected_electrode_native_space = results.get('electrode_native_space')
        if self.detected_electrode_space is None:
            self.detected_electrode_space = self.detected_electrode_native_space
        self._update_classify_button_state()

        trans_mat = None
        if isinstance(self.trans, dict) and 'trans' in self.trans:
            trans_mat = np.array(self.trans['trans'])
        elif self.trans is not None:
            trans_mat = np.array(self.trans)
        else:
            trans_mat = np.eye(4)
            self.trans = {'trans': trans_mat}

        if not self.ct_registered_path:
            self.ct_registered_path = self.ct_path

        ct_label = None
        if self.ct_registered_path:
            ct_label = os.path.basename(self.ct_registered_path)
        elif self.ct_path:
            ct_label = os.path.basename(self.ct_path)
        else:
            ct_label = "CT"

        # Check if registration was skipped (identity transform)
        is_identity = np.allclose(trans_mat, np.eye(4))

        # Update UI
        if is_identity:
            status_msg = f"Status: CT ready (no registration) - {ct_label}"
            print("✓ Using CT as-is (already aligned)")
        else:
            status_msg = f"Status: Registration complete! Registered CT: {ct_label}"
            print("✓ Registration successful!")

        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(status_msg)
        self.update_button_states()

        print(f"  - CT file: {self.ct_registered_path}")
        print("  - Transform saved to memory")

        self._browse_subjects_directory()
        self._display_detected_electrodes(self.detected_electrodes, self.detected_electrode_space)
    
    def _try_freesurfer_registration(self):
        """Attempt coregistration using FreeSurfer mri_robust_register."""
        try:
            import subprocess
            from pathlib import Path
            
            # Check if FreeSurfer is available
            try:
                result = subprocess.run(
                    ["mri_robust_register", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    raise FileNotFoundError("FreeSurfer not found")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print("FreeSurfer not found in PATH")
                return False
            
            self.coreg_status_label.setText("Status: Running FreeSurfer mri_robust_register (this may take a few minutes)...")
            QtWidgets.QApplication.processEvents()
            
            # Create output directory for coregistration results
            if self.patient_reg_folder:
                output_dir = Path(self.patient_reg_folder)
            else:
                output_dir = Path(self.mri_path).parent / "coregistration"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Output files
            ct_registered_path = output_dir / "ct_registered_to_mri.nii.gz"
            lta_transform_path = output_dir / "ct_to_mri.lta"
            
            # Run FreeSurfer's mri_robust_register for CT-to-MRI registration
            cmd = [
                "mri_robust_register",
                "--mov", str(self.ct_path),
                "--dst", str(self.mri_path),
                "--lta", str(lta_transform_path),
                "--mapmov", str(ct_registered_path),
                "--satit",
                "--iscale",
                "--cost", "nmi"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"FreeSurfer registration failed:\n{result.stderr}")
                return False
            
            print("FreeSurfer registration output:")
            print(result.stdout)
            
            # Parse the LTA file
            self.lta_transform_path = lta_transform_path
            self.ct_registered_path = ct_registered_path
            
            affine_matrix = self._parse_lta_transform(lta_transform_path)
            
            if affine_matrix is not None:
                self.trans = mne.transforms.Transform(fro="head", to="mri", trans=affine_matrix)
                self.coreg_status_label.setText(
                    f"Status: FreeSurfer registration complete! CT saved to {ct_registered_path.name}"
                )
                self.update_button_states()
                print(f"  - Registered CT: {ct_registered_path}")
                print(f"  - Transform: {lta_transform_path}")
                
                # Display registered CT in 3D viewer
                self._display_patient_volumes()
                
                return True
            
            return False
            
        except subprocess.TimeoutExpired:
            print("FreeSurfer registration timed out")
            return False
        except Exception as e:
            print(f"FreeSurfer registration error: {e}")
            return False
    
    def _try_simpleitk_registration(self):
        """Attempt coregistration using SimpleITK (pure Python)."""
        try:
            import SimpleITK as sitk
            import nibabel as nib
            from pathlib import Path
            
            print("Attempting registration with SimpleITK...")
            self.coreg_status_label.setText("Status: Running SimpleITK registration (this may take a few minutes)...")
            QtWidgets.QApplication.processEvents()
            
            # Load images
            fixed_image = sitk.ReadImage(str(self.mri_path))
            moving_image = sitk.ReadImage(str(self.ct_path))
            
            # Initialize registration
            registration_method = sitk.ImageRegistrationMethod()
            
            # Similarity metric - Mutual Information for CT-MRI
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            
            # Optimizer
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=100,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Setup for multi-resolution framework
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # Initial transform - rigid + scale (affine would be even better)
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            
            # Interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Execute registration
            print("  SimpleITK: Starting registration...")
            final_transform = registration_method.Execute(fixed_image, moving_image)
            
            print(f"  SimpleITK: Registration complete!")
            print(f"  Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
            print(f"  Final metric value: {registration_method.GetMetricValue()}")
            
            # Apply transform to moving image
            resampled_image = sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID()
            )
            
            # Save registered CT
            if self.patient_reg_folder:
                output_dir = Path(self.patient_reg_folder)
            else:
                output_dir = Path(self.mri_path).parent / "coregistration"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            ct_registered_path = output_dir / "ct_registered_to_mri_sitk.nii.gz"
            sitk.WriteImage(resampled_image, str(ct_registered_path))
            
            # Convert SimpleITK transform to numpy affine matrix
            affine_matrix = self._simpleitk_to_affine(final_transform, fixed_image, moving_image)
            
            if affine_matrix is not None:
                self.trans = mne.transforms.Transform(fro="head", to="mri", trans=affine_matrix)
                self.ct_registered_path = ct_registered_path
                
                self.coreg_status_label.setText(
                    f"Status: SimpleITK registration complete! CT saved to {ct_registered_path.name}"
                )
                self.update_button_states()
                print(f"  - Registered CT: {ct_registered_path}")
                
                # Display registered CT in 3D viewer
                self._display_patient_volumes()
                
                return True
            
            return False
            
        except ImportError:
            print("SimpleITK not available. Install with: pip install SimpleITK")
            return False
        except Exception as e:
            print(f"SimpleITK registration error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _try_dipy_registration(self):
        """Attempt coregistration using DIPY (pure Python fallback)."""
        try:
            from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, AffineMap
            from dipy.align.transforms import RigidTransform3D, AffineTransform3D
            import nibabel as nib
            from pathlib import Path
            
            print("Attempting registration with DIPY...")
            self.coreg_status_label.setText("Status: Running DIPY registration (this may take a few minutes)...")
            QtWidgets.QApplication.processEvents()
            
            # Load images
            mri_img = nib.load(self.mri_path)
            ct_img = nib.load(self.ct_path)
            
            mri_data = mri_img.get_fdata()
            ct_data = ct_img.get_fdata()
            
            mri_affine = mri_img.affine
            ct_affine = ct_img.affine
            
            # Setup registration
            metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
            
            # Multi-resolution registration
            affreg = AffineRegistration(
                metric=metric,
                level_iters=[10000, 1000, 100],
                sigmas=[3.0, 1.0, 0.0],
                factors=[4, 2, 1]
            )
            
            print("  DIPY: Starting rigid registration...")
            # Start with rigid registration
            rigid = RigidTransform3D()
            rigid_map = affreg.optimize(
                mri_data, ct_data,
                rigid, None,
                mri_affine, ct_affine
            )
            
            print("  DIPY: Starting affine registration...")
            # Refine with affine
            affine = AffineTransform3D()
            affine_map = affreg.optimize(
                mri_data, ct_data,
                affine, None,
                mri_affine, ct_affine,
                starting_affine=rigid_map.affine
            )
            
            print("  DIPY: Registration complete!")
            
            # Apply transform
            transformed_ct = affine_map.transform(ct_data)
            
            # Save registered CT
            if self.patient_reg_folder:
                output_dir = Path(self.patient_reg_folder)
            else:
                output_dir = Path(self.mri_path).parent / "coregistration"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            ct_registered_path = output_dir / "ct_registered_to_mri_dipy.nii.gz"
            
            # Create new NIfTI with registered data
            registered_img = nib.Nifti1Image(transformed_ct, mri_affine)
            nib.save(registered_img, str(ct_registered_path))
            
            # Get affine matrix
            affine_matrix = affine_map.affine
            
            # Store transform
            self.trans = mne.transforms.Transform(fro="head", to="mri", trans=affine_matrix)
            self.ct_registered_path = ct_registered_path
            
            self.coreg_status_label.setText(
                f"Status: DIPY registration complete! CT saved to {ct_registered_path.name}"
            )
            self.update_button_states()
            print(f"  - Registered CT: {ct_registered_path}")
            
            # Display registered CT in 3D viewer
            self._display_patient_volumes()
            
            return True
            
        except ImportError:
            print("DIPY not available. Install with: pip install dipy")
            return False
        except Exception as e:
            print(f"DIPY registration error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _simpleitk_to_affine(self, sitk_transform, fixed_image, moving_image):
        """Convert SimpleITK transform to 4x4 affine matrix."""
        try:
            import SimpleITK as sitk
            import numpy as np
            
            # Get the transform parameters
            if isinstance(sitk_transform, sitk.Euler3DTransform):
                # Extract rotation and translation
                matrix = np.array(sitk_transform.GetMatrix()).reshape(3, 3)
                translation = np.array(sitk_transform.GetTranslation())
                
                # Build 4x4 affine
                affine = np.eye(4)
                affine[:3, :3] = matrix
                affine[:3, 3] = translation
                
                print(f"  SimpleITK transform matrix:\n{affine}")
                return affine
            else:
                # For other transform types, try to get affine directly
                print("  Warning: Non-Euler transform, attempting conversion")
                # This is a simplified conversion
                return np.eye(4)
                
        except Exception as e:
            print(f"Error converting SimpleITK transform: {e}")
            return None
    
    def _parse_lta_transform(self, lta_path):
        """Parse FreeSurfer LTA file and extract affine transformation matrix."""
        try:
            with open(lta_path, 'r') as f:
                lines = f.readlines()
            
            # Find the transformation matrix in the LTA file
            # LTA format has the matrix after "1 4 4" line
            matrix = []
            in_matrix = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line == "1 4 4":  # Start of matrix section
                    in_matrix = True
                    continue
                
                if in_matrix and len(matrix) < 4:
                    # Parse matrix rows
                    values = line.split()
                    if len(values) >= 4:
                        matrix.append([float(v) for v in values[:4]])
                    
                    if len(matrix) == 4:
                        break
            
            if len(matrix) == 4:
                affine = np.array(matrix)
                print(f"Parsed LTA transform:\n{affine}")
                return affine
            else:
                print(f"Error: Could not parse transformation matrix from {lta_path}")
                return None
                
        except Exception as e:
            print(f"Error parsing LTA file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def launch_mne_coreg_gui(self, fif_path=None):
        """Launch MNE's interactive coregistration GUI."""
        try:
            subject_id, subjects_dir = self._get_brain_subject_info()
            if not fif_path:
                if hasattr(self, "_selected_meg_coreg_fif_path"):
                    fif_path = self._selected_meg_coreg_fif_path()
                else:
                    fif_path = getattr(self, "_meg_current_fif_path", None)
            trans_path = None
            if hasattr(self, "_find_existing_meg_trans_file"):
                try:
                    existing_trans = self._find_existing_meg_trans_file(
                        fif_path=fif_path,
                        subject_id=subject_id,
                    )
                    if existing_trans is not None:
                        trans_path = str(existing_trans)
                except Exception:
                    trans_path = None

            self.coreg_status_label.setText("Status: Checking MNE anatomy assets...")
            QtWidgets.QApplication.processEvents()
            if hasattr(self, "_ensure_meg_anatomy_assets"):
                created_assets = self._ensure_meg_anatomy_assets(
                    subject_id=subject_id,
                    subjects_dir=subjects_dir,
                    ensure_source_space=False,
                )
                if created_assets:
                    self.coreg_status_label.setText(
                        "Status: Generated " + ", ".join(created_assets) + "; launching MNE Coregistration GUI..."
                    )
                    QtWidgets.QApplication.processEvents()

            self.coreg_status_label.setText("Status: Launching MNE Coregistration GUI...")
            QtWidgets.QApplication.processEvents()

            window = mne.gui.coregistration(
                inst=fif_path,
                subject=subject_id,
                subjects_dir=subjects_dir,
                trans=trans_path,
                block=False,
            )
            if window is not None:
                windows = getattr(self, "_mne_coreg_windows", None)
                if windows is None:
                    windows = []
                    self._mne_coreg_windows = windows
                windows.append(window)
                for method_name in ("show", "raise_", "activateWindow"):
                    method = getattr(window, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass

            status = f"Status: MNE Coreg GUI launched for {subject_id}"
            if fif_path:
                status += f" | inst={os.path.basename(fif_path)}"
            self.coreg_status_label.setText(status)

        except Exception as e:
            self.coreg_status_label.setText(f"Status: Error launching GUI - {str(e)}")
            print(f"GUI launch error: {e}")
            import traceback
            traceback.print_exc()
    
    def save_transformation(self):
        """Save the computed transformation matrix."""
        trans_to_save = getattr(self, "_meg_trans", None)
        if trans_to_save is None:
            trans_to_save = self.trans
        if trans_to_save is None:
            self.coreg_status_label.setText("Status: Error - No transformation to save")
            return

        try:
            default_path = None
            if getattr(self, "_meg_current_fif_path", None) and hasattr(self, "_default_meg_trans_output_path"):
                try:
                    default_candidate = self._default_meg_trans_output_path()
                    if default_candidate is not None:
                        default_path = str(default_candidate)
                except Exception:
                    default_path = None
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                None,
                "Save Transformation Matrix",
                default_path or "trans.fif",
                "FIF Files (*.fif);;All Files (*.*)"
            )

            if save_path:
                mne.write_trans(save_path, trans_to_save)
                if getattr(self, "_meg_current_fif_path", None):
                    self._meg_trans = trans_to_save
                    self._meg_loaded_trans_path = os.path.abspath(save_path)
                    if hasattr(self, "_fif_file_identity"):
                        self._meg_loaded_trans_file_identity = self._fif_file_identity(
                            self._meg_loaded_trans_path
                        )
                    if hasattr(self, "_fif_file_identity"):
                        fif_identity = self._fif_file_identity(
                            self._meg_current_fif_path
                        )
                        self._meg_current_fif_identity = fif_identity
                        self._meg_trans_fif_identity = fif_identity
                    self._meg_inverse_key = None
                    self._meg_inverse_operator = None
                self.coreg_status_label.setText(f"Status: Transformation saved to {os.path.basename(save_path)}")

        except Exception as e:
            self.coreg_status_label.setText(f"Status: Error saving - {str(e)}")
            print(f"Save error: {e}")
            import traceback
            traceback.print_exc()
    
    def load_transformation(self, trans_path):
        """Load a previously saved transformation matrix."""
        try:
            self.trans = mne.read_trans(trans_path)
            if getattr(self, "_meg_current_fif_path", None):
                self._meg_trans = self.trans
                self._meg_loaded_trans_path = os.path.abspath(trans_path)
                if hasattr(self, "_fif_file_identity"):
                    self._meg_loaded_trans_file_identity = self._fif_file_identity(
                        self._meg_loaded_trans_path
                    )
                if hasattr(self, "_fif_file_identity"):
                    fif_identity = self._fif_file_identity(
                        self._meg_current_fif_path
                    )
                    self._meg_current_fif_identity = fif_identity
                    self._meg_trans_fif_identity = fif_identity
                self._meg_inverse_key = None
                self._meg_inverse_operator = None
            self.coreg_status_label.setText(f"Status: Transformation loaded from {os.path.basename(trans_path)}")
            self.update_button_states()
            if getattr(self, "_meg_sensor_info", None) is not None:
                try:
                    self._render_meg_sensors(activity_scores=self._meg_activity_scores)
                except Exception:
                    pass
            return True
        except Exception as e:
            self.coreg_status_label.setText(f"Status: Error loading transformation - {str(e)}")
            print(f"Load transformation error: {e}")
            return False

    def _label_parcellation_is_surface(self, parcellation=None):
        parcellation = normalize_parcellation_key(parcellation or self.label_parcellation)
        return parcellation in {"aparc", *HCP_MMP_PARCS}

    def _clear_label_overlays(self, clear_annotations=True):
        """Remove existing anatomical label overlays from the Brain scene."""
        if not hasattr(self, "brain") or self.brain is None:
            return
        try:
            self.brain.remove_volume_labels()
        except Exception:
            pass
        try:
            self.brain.remove_labels()
        except Exception:
            pass
        if clear_annotations:
            try:
                self.brain.remove_annotations()
            except Exception:
                pass

    def _ensure_label_parcellation_subject(self, parcellation):
        """Ensure the active Brain subject can display the requested atlas."""
        parcellation = normalize_parcellation_key(parcellation)
        if parcellation not in HCP_MMP_PARCS:
            return True
        subject_id, _subjects_dir = self._get_brain_subject_info()
        if str(subject_id) == "fsaverage":
            return True
        self.force_fsaverage = True
        self._sync_subject_view_combo()
        try:
            self._reload_brain_with_subject()
        except Exception as exc:
            print(f"Error switching to fsaverage for {parcellation}: {exc}")
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Error switching to fsaverage: {exc}")
            return False
        if hasattr(self, "coreg_status_label"):
            self.coreg_status_label.setText(
                "Status: HCP-MMP is an fsaverage annotation; switched viewer to fsaverage"
            )
        return True

    def _load_volume_label_names_for_current_subject(self):
        """Load available aparc+aseg labels for the active subject."""
        subject_id, subjects_dir = self._get_brain_subject_info()
        fname_aseg = Path(subjects_dir) / subject_id / "mri" / "aparc+aseg.mgz"
        if not fname_aseg.exists():
            alt = Path(subjects_dir) / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
            fname_aseg = alt if alt.exists() else fname_aseg
        if not fname_aseg.exists():
            self.label_names = []
            self.active_labels = []
            return False
        self._aseg_path = str(fname_aseg)
        self._aseg_name = aseg_name_from_path(fname_aseg)
        self.label_names = mne.get_volume_labels_from_aseg(str(fname_aseg))
        self._surface_labels_by_name = {}
        self._surface_label_parc = None
        return True

    def _load_surface_label_names_for_current_subject(self, parcellation):
        """Load surface label names for aparc or HCP-MMP annotations."""
        parcellation = normalize_parcellation_key(parcellation)
        subject_id, subjects_dir = self._get_brain_subject_info()
        if parcellation in HCP_MMP_PARCS:
            subject_id = "fsaverage"
        try:
            labels = read_surface_labels(subject_id, subjects_dir, parcellation)
        except Exception as exc:
            print(f"Could not read surface labels for {parcellation}: {exc}")
            labels = []
        labels_by_name = {
            str(getattr(label, "name", "") or ""): label
            for label in labels
            if str(getattr(label, "name", "") or "")
        }
        self.label_names = sorted(
            name for name in labels_by_name
            if "unknown" not in name.lower()
        )
        self.active_labels = [label for label in self.active_labels if label in labels_by_name]
        self._surface_labels_by_name = labels_by_name
        self._surface_label_parc = parcellation
        self._surface_label_subject = subject_id
        self._surface_label_subjects_dir = str(subjects_dir)
        return bool(labels)

    def _prepare_label_names_for_current_parcellation(self):
        """Refresh label names for the active label atlas."""
        parcellation = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        self.label_parcellation = parcellation
        if self._label_parcellation_is_surface(parcellation):
            return self._load_surface_label_names_for_current_subject(parcellation)
        return self._load_volume_label_names_for_current_subject()

    def _label_combo_for_name(self, label):
        """Choose the label combo bucket for a label name."""
        text = str(label or "")
        lower = text.lower()
        if lower.startswith("left"):
            return self.cmbLeft
        if lower.startswith("right"):
            return self.cmbRight
        if (
            lower.startswith("ctx-lh")
            or lower.endswith("-lh")
            or lower.startswith(("lh.", "lh_", "lh-", "l_", "left_", "left-"))
        ):
            return self.cmbCortexLH
        if (
            lower.startswith("ctx-rh")
            or lower.endswith("-rh")
            or lower.startswith(("rh.", "rh_", "rh-", "r_", "right_", "right-"))
        ):
            return self.cmbCortexRH
        if lower.startswith("cc"):
            return self.cmbCC
        return self.cmbOthers

    def _populate_label_selection_combos(self):
        """Populate label selection combos from `self.label_names`."""
        combos = [self.cmbLeft, self.cmbRight, self.cmbCortexLH, self.cmbCortexRH, self.cmbCC, self.cmbOthers]
        self._syncing_labels = True
        try:
            for cmb in combos:
                cmb.clear()
            for label in self.label_names:
                if "unknown" in str(label).lower():
                    continue
                cmb = self._label_combo_for_name(label)
                cmb.addItem(label)
                item = cmb.model().item(cmb.model().rowCount() - 1)
                if item is not None:
                    item.setCheckState(QtCore.Qt.Checked if label in self.active_labels else QtCore.Qt.Unchecked)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        finally:
            self._syncing_labels = False

    def _render_active_labels(self):
        """Render active labels for the current atlas."""
        if not hasattr(self, "brain") or self.brain is None:
            return
        parcellation = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        if self._label_parcellation_is_surface(parcellation):
            self._clear_label_overlays(clear_annotations=False)
            subject_id, _ = self._get_brain_subject_info()
            if parcellation not in HCP_MMP_PARCS or str(subject_id) == "fsaverage":
                try:
                    self.brain.add_annotation(
                        parcellation,
                        borders=2,
                        alpha=0.35,
                        remove_existing=True,
                    )
                except Exception as exc:
                    print(f"Could not add surface annotation {parcellation}: {exc}")
            labels_by_name = getattr(self, "_surface_labels_by_name", {}) or {}
            for label_name in self.active_labels:
                label = labels_by_name.get(label_name)
                if label is None:
                    continue
                try:
                    self.brain.add_label(label, alpha=0.75, borders=False)
                except Exception as exc:
                    print(f"Could not add surface label {label_name}: {exc}")
        else:
            self._clear_label_overlays(clear_annotations=True)
            if self.active_labels:
                try:
                    self.brain.add_volume_labels(
                        aseg=getattr(self, "_aseg_name", "aparc+aseg"),
                        labels=self.active_labels,
                        legend=dict(bcolor=None),
                    )
                except Exception as exc:
                    print(f"Could not add volume labels: {exc}")
        try:
            self.brain._renderer.plotter.update()
        except Exception:
            pass

    def _set_label_parcellation(self, parcellation):
        """Switch anatomical label atlas in the right-side label panel."""
        parcellation = normalize_parcellation_key(parcellation)
        self.label_parcellation = parcellation
        self.active_labels = []
        if not self._ensure_label_parcellation_subject(parcellation):
            return
        try:
            self._prepare_label_names_for_current_parcellation()
            self._clear_label_overlays(clear_annotations=True)
            if self._label_parcellation_is_surface(parcellation):
                subject_id, _ = self._get_brain_subject_info()
                if parcellation not in HCP_MMP_PARCS or str(subject_id) == "fsaverage":
                    try:
                        self.brain.add_annotation(parcellation, borders=2, alpha=0.35, remove_existing=True)
                    except Exception as exc:
                        print(f"Could not add surface annotation {parcellation}: {exc}")
            self._populate_label_selection_combos()
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Label atlas set to {parcellation}")
        except Exception as exc:
            if hasattr(self, "coreg_status_label"):
                self.coreg_status_label.setText(f"Status: Label atlas error - {exc}")
            print(f"Label parcellation error: {exc}")

    def _on_label_parcellation_changed(self, _index):
        combo = getattr(self, "label_parcellation_combo", None)
        if combo is None:
            return
        parc = combo.currentData() or "aparc+aseg"
        QtCore.QTimer.singleShot(0, lambda: self._set_label_parcellation(parc))

    def get_label_chart(self):
        # Add a checkbox for each label in vertical layout and create a widget
        self.labels_widget = QtWidgets.QWidget()
        vLyt = QtWidgets.QVBoxLayout()
        self.labels_widget.setLayout(vLyt)

        # Recreate legend controls whenever the right-side panel is rebuilt.
        self.meg_activity_legend_widget = None
        self.meg_activity_legend_title_label = None
        self.meg_activity_legend_bar_label = None
        self.meg_activity_legend_min_label = None
        self.meg_activity_legend_max_label = None
        
        # Add file browser interface at the top
        file_browser = self.create_file_browser_interface()
        vLyt.addWidget(file_browser)
        
        # Add a separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        vLyt.addWidget(separator)

        parc_row = QtWidgets.QHBoxLayout()
        parc_row.setContentsMargins(0, 0, 0, 0)
        parc_row.setSpacing(6)
        parc_label = QtWidgets.QLabel("Parcellation:")
        parc_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; color: #555555; }")
        parc_row.addWidget(parc_label)
        self.label_parcellation_combo = QtWidgets.QComboBox()
        self.label_parcellation_combo.addItem("aparc+aseg", "aparc+aseg")
        self.label_parcellation_combo.addItem("aparc", "aparc")
        self.label_parcellation_combo.addItem("HCP-MMP", "HCPMMP1")
        self.label_parcellation_combo.addItem("HCP-MMP combined", "HCPMMP1_combined")
        current_parc = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        idx = self.label_parcellation_combo.findData(current_parc)
        if idx >= 0:
            self.label_parcellation_combo.setCurrentIndex(idx)
        self.label_parcellation_combo.setToolTip("Switch anatomical label atlas")
        self.label_parcellation_combo.currentIndexChanged.connect(self._on_label_parcellation_changed)
        parc_row.addWidget(self.label_parcellation_combo)
        parc_row.addStretch()
        vLyt.addLayout(parc_row)
        
        # Label selection title
        label_title = QtWidgets.QLabel("Brain Region Labels")
        label_title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; padding: 5px; }")
        vLyt.addWidget(label_title)
        
        # Add combo boxes for volume, cortex, corpus-callosum, and fallback labels.
        combo_grid = QtWidgets.QGridLayout()
        combo_grid.setContentsMargins(0, 0, 0, 0)
        combo_grid.setHorizontalSpacing(4)
        combo_grid.setVerticalSpacing(2)
        self.cmbLeft     = CheckableComboBox()
        self.cmbRight    = CheckableComboBox()
        self.cmbCortexLH = CheckableComboBox()
        self.cmbCortexRH = CheckableComboBox()
        self.cmbCC       = CheckableComboBox()
        self.cmbOthers   = CheckableComboBox()
        label_combo_specs = [
            ("Left vol", self.cmbLeft, "Left aparc+aseg volume labels"),
            ("Right vol", self.cmbRight, "Right aparc+aseg volume labels"),
            ("LH cortex", self.cmbCortexLH, "Left-hemisphere surface labels, including HCP-MMP"),
            ("RH cortex", self.cmbCortexRH, "Right-hemisphere surface labels, including HCP-MMP"),
            ("CC", self.cmbCC, "Corpus-callosum labels"),
            ("Other", self.cmbOthers, "Labels that do not match the standard atlas naming patterns"),
        ]
        for col, (title, combo, tooltip) in enumerate(label_combo_specs):
            combo_label = QtWidgets.QLabel(title)
            combo_label.setStyleSheet("QLabel { font-size: 10px; color: #555555; }")
            combo_label.setAlignment(QtCore.Qt.AlignCenter)
            combo.setToolTip(tooltip)
            combo_grid.addWidget(combo_label, 0, col)
            combo_grid.addWidget(combo, 1, col)

        # filter button (replaces standalone clear button)
        filter_col = len(label_combo_specs)
        filter_header = QtWidgets.QLabel("Filter")
        filter_header.setStyleSheet("QLabel { font-size: 10px; color: #555555; }")
        filter_header.setAlignment(QtCore.Qt.AlignCenter)
        combo_grid.addWidget(filter_header, 0, filter_col)

        self.filter_button = QtWidgets.QToolButton()
        self.filter_button.setText("Filter ▾")
        self.filter_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.filter_button.setToolTip(
            "Filter brain labels by functional network (Visual, Semantic, Visual+Semantic, All, Clear)"
        )
        self.clear_button = self.filter_button

        filter_menu = QtWidgets.QMenu(self.filter_button)
        act_vis = filter_menu.addAction("Select Visual labels (Occipital & ventral stream)")
        act_vis.triggered.connect(lambda: self.select_label_filter("visual"))
        act_sem = filter_menu.addAction("Select Semantic labels (Temporal, parietal, IFG)")
        act_sem.triggered.connect(lambda: self.select_label_filter("semantic"))
        act_both = filter_menu.addAction("Select Visual + Semantic labels")
        act_both.triggered.connect(lambda: self.select_label_filter("visual_semantic"))
        filter_menu.addSeparator()
        act_all = filter_menu.addAction("Select All labels")
        act_all.triggered.connect(lambda: self.select_label_filter("all"))
        act_clr = filter_menu.addAction("Clear selection")
        act_clr.triggered.connect(self.clear_labels)
        self.filter_button.setMenu(filter_menu)
        combo_grid.addWidget(self.filter_button, 1, filter_col)
        vLyt.addLayout(combo_grid)

        for cmb in [self.cmbLeft, self.cmbRight, self.cmbCortexLH, self.cmbCortexRH, self.cmbCC, self.cmbOthers]:
            cmb.model().itemChanged.connect(self._on_label_item_changed)

        self.label_checkboxes = {}
        try:
            self._prepare_label_names_for_current_parcellation()
        except Exception as exc:
            print(f"Could not prepare labels for {getattr(self, 'label_parcellation', 'aparc+aseg')}: {exc}")
        self._populate_label_selection_combos()
        self._render_active_labels()

        return self.labels_widget

    def _on_label_item_changed(self, item):
        """Synchronize self.active_labels with item checkState changes."""
        if getattr(self, "_syncing_labels", False):
            return
        label = str(item.text()) if item is not None else ""
        if not label or "unknown" in label.lower():
            return
        if item.checkState() == QtCore.Qt.Checked:
            if label not in self.active_labels:
                self.active_labels.append(label)
        else:
            if label in self.active_labels:
                self.active_labels = [l for l in self.active_labels if l != label]
        if hasattr(self, "brain") and self.brain is not None:
            self._render_active_labels()

    def select_label_filter(self, filter_name):
        """Filter and select labels matching a functional network."""
        if filter_name == "clear":
            self.clear_labels()
            return

        parcellation = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
        if filter_name == "visual":
            target_set = {
                str(label) for label in self.label_names
                if is_visual_parcel(label, parcellation=parcellation)
            }
        elif filter_name == "semantic":
            target_set = {
                str(label) for label in self.label_names
                if is_semantic_parcel(label, parcellation=parcellation)
            }
        elif filter_name in {"visual_semantic", "both"}:
            target_set = {
                str(label) for label in self.label_names
                if is_visual_parcel(label, parcellation=parcellation) or is_semantic_parcel(label, parcellation=parcellation)
            }
        elif filter_name == "all":
            target_set = {str(label) for label in self.label_names if "unknown" not in str(label).lower()}
        else:
            return

        self._syncing_labels = True
        try:
            combos = [self.cmbLeft, self.cmbRight, self.cmbCortexLH, self.cmbCortexRH, self.cmbCC, self.cmbOthers]
            for cmb in combos:
                model = cmb.model()
                if model is None:
                    continue
                for i in range(model.rowCount()):
                    item = model.item(i)
                    if item is not None:
                        item.setCheckState(
                            QtCore.Qt.Checked if str(item.text()) in target_set else QtCore.Qt.Unchecked
                        )

            self.active_labels = [str(label) for label in self.label_names if str(label) in target_set]
        finally:
            self._syncing_labels = False

        if hasattr(self, "brain") and self.brain is not None:
            self._render_active_labels()

    def clear_labels(self):
        self._syncing_labels = True
        try:
            for cmb in [self.cmbLeft, self.cmbRight, self.cmbCortexLH, self.cmbCortexRH, self.cmbCC, self.cmbOthers]:
                model = cmb.model()
                if model is None:
                    continue
                for i in range(model.rowCount()):
                    item = model.item(i)
                    if item is not None:
                        item.setCheckState(QtCore.Qt.Unchecked)
            self.active_labels = []
        finally:
            self._syncing_labels = False

        if hasattr(self, 'brain'):
            parcellation = normalize_parcellation_key(getattr(self, "label_parcellation", "aparc+aseg"))
            if self._label_parcellation_is_surface(parcellation):
                try:
                    self.brain.remove_labels()
                except Exception:
                    pass
                try:
                    self.brain.add_annotation(parcellation, borders=2, alpha=0.35, remove_existing=True)
                except Exception:
                    pass
            else:
                self._clear_label_overlays(clear_annotations=True)
            try:
                self.brain._renderer.plotter.update()
            except Exception:
                pass
    
    def show_hide_labels(self, index):
        cmb = index.model().parent()
        item = cmb.model().itemFromIndex(index)
        if item is None:
            return

        label = str(item.text())
        if item.checkState() == QtCore.Qt.Checked:
            if label not in self.active_labels:
                self.active_labels.append(label)
        else:
            if label in self.active_labels:
                self.active_labels = [l for l in self.active_labels if l != label]

        if hasattr(self, 'brain') and self.brain is not None:
            self._render_active_labels()

    def show_bundle_vol(self, bundle):
        legend_kwargs = dict(bcolor=None)
        if not hasattr(self, 'brain'):
            return

        # Remove any existing labels
        self._clear_label_overlays(clear_annotations=not self._label_parcellation_is_surface())
        if bundle is not None:
            if self._label_parcellation_is_surface():
                label = (getattr(self, "_surface_labels_by_name", {}) or {}).get(bundle)
                if label is not None:
                    self.brain.add_label(label, alpha=0.75, borders=False)
            else:
                self.brain.add_volume_labels(
                    aseg=getattr(self, "_aseg_name", "aparc+aseg"),
                    labels=[bundle],
                    legend=legend_kwargs,
                )
            try:
                self.brain._renderer.plotter.update()
            except Exception:
                pass

    def update_patient_mri_sensors(self, mri_path):
        pass
        
    #endregion 3D viewer
