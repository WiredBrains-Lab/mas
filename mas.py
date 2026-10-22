"""
MEG Analysis Suite (MAS)
===========================

Main Qt application entry for the MEG analysis suite.
Provides UI scaffolding and orchestrates core processing components.

Notes
-----
- Build docs: `sphinx-build -M html docs/source/ docs/build/`.
- Docstrings follow NumPy style for Sphinx Napoleon.
"""
# Author: Sunil Mathew
# Date: June 14, 2026
# MEG Analysis Suite (MAS)
# remember to add docstrings to all functions and classes for auto documentation
import multiprocessing as mp
import psutil
import pyqtgraph as pg
# pg.setConfigOptions(antialias=True, useOpenGL=True)
pg.setConfigOption('imageAxisOrder', 'row-major')

# Configure PyQtGraph to handle large data ranges better
pg.setConfigOption('crashWarning', True)
pg.setConfigOption('enableExperimental', False)

# Import warnings module to suppress overflow warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered in cast.*')
import pyvista as pv

import traceback
import os
import sys
import subprocess
import re

app_path = os.path.join(os.path.dirname(__file__))
sys.path.append(app_path)

# Import and apply macOS Qt compatibility fixes early
from macos_qt_fixes import setup_macos_compatibility
setup_macos_compatibility()

# os.environ["QT_API"] = "pyqt5"
os.environ["QT_QPA_PLATFORM"] = "windows" if sys.platform.startswith("win") else "cocoa" if sys.platform == "darwin" else "xcb"
os.environ["MKL_NUM_THREADS"] = '6'
os.environ["OMP_NUM_THREADS"] = '6'
os.environ["USE_SYSTEM_VTK"] = 'OFF'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_VERBOSITY"] = "ERROR"


if sys.platform == 'darwin':
    # Suppress Qt graphics warnings on macOS
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.cocoa.drawing.debug=false"
    os.environ["QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM"] = "1"
from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtWidgets import QApplication, QWidget
from qtpy.QtGui import QPalette
from pyvistaqt import MainWindow
from core.orchestrator import Orchestrator

class MAS(MainWindow):

    #region Initialization

    def __init__(self):
        """
         Initialize the window. This is called by __init__ () and can be overridden
        """
        super().__init__()
        ui_file = os.path.join(app_path, 'mas.ui')
        uic.loadUi(ui_file, self)
        
        # Ensure window has standard title bar buttons (minimize, maximize, close)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint | 
                           QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
        
        self.init_ui()
        self.init_orchestrator()
        # self.load_mat_data()

    def resizeEvent(self, event):
        """
         Called when the window is resized. This is the default implementation of QWidget. resizeEvent
         
         Args:
         	 event: The resize event to
        """
        try:
            QtWidgets.QMainWindow.resizeEvent(self, event)

        except:
            print("resizeEvent")
            print(traceback.format_exc())

    def init_ui(self):
        """
         Initialize the UI. This is called by __init__ () and can be overridden
        """
        widget = QWidget()
        widget.setLayout(self.vLytMain)
        self.setCentralWidget(widget)
        
        self.tabData.setLayout(self.vLytData)

        self.init_file_menu()
        self.init_3d_viewer()
        self.init_dock_area()
        
        self.tabTasks.setLayout(self.vLytTasksMain)

        self.statusBar().showMessage('Ready')
        self.pbTask = QtWidgets.QProgressBar()
        self.statusBar().addPermanentWidget(self.pbTask)

    def init_dock_area(self):
        """
            Initialize signal viewer. This is called by init_ui
        """
        palette = QApplication.palette()
        # Change background color signal viewer based on whether dark mode is on or off.
        if palette.color(QPalette.Window).value() < 128:
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')
        else:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
        self.dock_area = pg.dockarea.DockArea()
        self.vLytTasksMain.addWidget(self.dock_area)

    def restore_ui(self):
        self.orchestrator.restore_ui()

    #endregion Initialization

    #region File menu

    def init_file_menu(self):
        """
         Initialize menu for file dialog and add actions to menubar. This is called by init_ui
        """
        # Add File menu
        self.mnuFile = self.menuBar().addMenu('File')

        # Open MEG FIF file action
        open_fif_action = self.mnuFile.addAction('Open .fif file', self.open_fif_file_dialog)
        open_fif_action.setShortcut('Ctrl+O')
        
        # Open CT/MRI action
        open_ct_mri_action = self.mnuFile.addAction('Open CT/MRI', self.open_ct_mri_dialog)
        open_ct_mri_action.setShortcut('Ctrl+Shift+M')
        
        # Exit action
        exit_action = self.mnuFile.addAction('Exit', self.close)
        exit_action.setShortcut('Ctrl+Q')

        self.edit_menu = self.menuBar().addMenu('Edit')
        restore_ui_action = self.edit_menu.addAction('Restore UI', self.restore_ui)
        restore_ui_action.setShortcut('Ctrl+R')

        self.init_help_menu()

    def init_help_menu(self):
        """
         Initialize help menu and add actions to menubar. This is called by init_file_menu
        """
        self.help_menu = self.menuBar().addMenu('Help')
        self.help_menu.addAction('About', self.show_about_dialog)
        self.help_menu.addAction('Documentation', self.open_documentation)

    def show_about_dialog(self):
        """
         Show about dialog. This is called by init_help_menu
        """
        QtWidgets.QMessageBox.about(self, "About MEG Analysis Suite",
                                    "<b>MEG Analysis Suite</b> v1.0<br>"
                                    "Date: June 14, 2026<br>"
                                    "Created by: Sunil Mathew, PhD<br>"                                    
                                    "Postdoctoral Fellow<br>"
                                    "Wired Brains Lab, Medical College of Wisconsin")
    def open_documentation(self):
        """Build (if possible) and open the local Sphinx documentation.

        Attempts to rebuild docs so recent changes (e.g., index.rst) appear
        without requiring a manual command. Falls back to opening the last
        built HTML if Sphinx is unavailable.
        """
        import webbrowser
        src_dir = os.path.abspath(os.path.join(app_path, 'docs', 'source'))
        build_dir = os.path.abspath(os.path.join(app_path, 'docs', 'build'))
        doc_path = os.path.join(build_dir, 'html', 'index.html')

        # Try to rebuild docs using python -m sphinx to avoid PATH issues
        try:
            cmd = [sys.executable, '-m', 'sphinx', '-M', 'html', src_dir, build_dir]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass  # Ignore build errors; try to open whatever exists

        if os.path.exists(doc_path):
            webbrowser.open('file://' + doc_path)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Documentation not found",
                "Documentation not found. Please build it via Tools -> Documentation or run:\n"
                "python -m sphinx -M html docs/source docs/build",
            )

    def open_ct_mri_dialog(self):
        """
        Open CT/MRI file dialog
        """
        # Select folder with CT/MRI files
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder with CT/MRI files', os.path.dirname(__file__))
        # Load data from file_path if file_path is not empty
        if folder_path:
            self.load_img_data(folder_path=folder_path)

    def open_fif_file_dialog(self):
        """Open FIF file dialog and load MEG data into the 3D viewer."""
        file_path = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Open FIF file',
            os.path.dirname(__file__),
            'FIF files (*.fif);;All files (*.*)'
        )[0]
        if file_path:
            self.load_fif_file(file_path=file_path)

    #endregion File menus

    def load_fif_file(self, file_path=None):
        """Load a FIF file and visualize MEG sensors/signals."""
        if not file_path:
            return

        try:
            report_fif_files = []
            sibling_fif_files = self._find_related_fif_files(file_path)
            if sibling_fif_files:
                msg = QtWidgets.QMessageBox(self)
                msg.setIcon(QtWidgets.QMessageBox.Question)
                msg.setWindowTitle('Additional FIF blocks found')
                msg.setText(f'Found {len(sibling_fif_files)} other .fif file(s) in this folder.')
                preview = ", ".join([os.path.basename(p) for p in sibling_fif_files[:5]])
                if len(sibling_fif_files) > 5:
                    preview += ", ..."
                msg.setInformativeText(
                    "Process these files too for combined stimulus summaries and signal-space concept decoding across runs?\n"
                    f"Files: {preview}\n"
                    "The selected FIF remains the active file in 3D and signal viewers."
                )
                msg.setStandardButtons(
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
                )
                msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
                choice = msg.exec()
                if choice == QtWidgets.QMessageBox.Cancel:
                    self.orchestrator.c.log.emit('FIF load cancelled by user.')
                    return
                if choice == QtWidgets.QMessageBox.Yes:
                    report_fif_files = sibling_fif_files

            other_patient_fif_files = self._find_other_patient_tsss_fif_files(
                file_path,
                exclude_paths=[file_path] + report_fif_files,
            )
            if other_patient_fif_files:
                msg = QtWidgets.QMessageBox(self)
                msg.setIcon(QtWidgets.QMessageBox.Question)
                msg.setWindowTitle('Other patient tSSS FIF files found')
                msg.setText(
                    f'Found {len(other_patient_fif_files)} .fif file(s) in tSSS folders for other patient folders.'
                )
                preview = self._preview_fif_file_groups(other_patient_fif_files)
                msg.setInformativeText(
                    "Process these files too for combined stimulus summaries and signal-space concept decoding?\n"
                    f"{preview}\n"
                    "The originally selected FIF remains the active file in 3D and signal viewers."
                )
                msg.setStandardButtons(
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
                )
                msg.setDefaultButton(QtWidgets.QMessageBox.No)
                choice = msg.exec()
                if choice == QtWidgets.QMessageBox.Cancel:
                    self.orchestrator.c.log.emit('FIF load cancelled by user.')
                    return
                if choice == QtWidgets.QMessageBox.Yes:
                    report_fif_files.extend(other_patient_fif_files)

            loaded = self.orchestrator.load_fif_file_for_3d(
                file_path,
                report_fif_files=report_fif_files,
            )
            if loaded:
                if report_fif_files:
                    self.orchestrator.c.log.emit(
                        f'Loaded FIF file and aggregated report across {1 + len(report_fif_files)} blocks: {file_path}'
                    )
                else:
                    self.orchestrator.c.log.emit(f'Loaded FIF file in 3D and signal viewers: {file_path}')
            else:
                self.orchestrator.c.log.emit(f'Could not load FIF file in MAS viewers: {file_path}')
        except Exception:
            print(traceback.format_exc())
            self.orchestrator.c.log.emit(f'Error loading FIF file: {file_path}')

    @staticmethod
    def _fif_sort_key(path):
        """Sort FIF files by run number when present, then by name."""
        name = os.path.basename(str(path))
        m = re.search(r"run[_-]?(\d+)|run(\d+)", name, flags=re.IGNORECASE)
        if m is not None:
            try:
                return (0, int(m.group(1) or m.group(2)), name.lower())
            except Exception:
                pass
        return (1, 0, name.lower())

    @staticmethod
    def _is_analysis_fif_file(path):
        """Return True for data FIFs, excluding event/inverse/ICA sidecar files."""
        name = os.path.basename(str(path))
        lname = name.lower()
        if not lname.endswith('.fif'):
            return False
        excluded_suffixes = (
            '-eve.fif',
            '_eve.fif',
            '-inv.fif',
            '_inv.fif',
            '-ica.fif',
            '_ica.fif',
        )
        return not lname.endswith(excluded_suffixes)

    @staticmethod
    def _is_tsss_folder(path):
        """Return True for folders that look like tSSS/SSS/MaxFilter outputs."""
        name = os.path.basename(os.path.abspath(str(path))).lower()
        return name in {"tsss"} #or "tsss" in name

    def _nearest_tsss_folder(self, file_path):
        """Find the nearest tSSS-like ancestor for a FIF path."""
        try:
            folder = os.path.dirname(os.path.abspath(str(file_path)))
        except Exception:
            return None
        previous = None
        while folder and folder != previous:
            if self._is_tsss_folder(folder):
                return folder
            previous = folder
            folder = os.path.dirname(folder)
        return None

    def _iter_tsss_folders_under(self, root, max_depth=3, max_dirs=250):
        """Yield tSSS-like folders under root without recursively scanning too broadly."""
        if not root or not os.path.isdir(root):
            return
        root = os.path.abspath(str(root))
        seen = 0
        for current, dirnames, _filenames in os.walk(root):
            seen += 1
            if seen > int(max_dirs):
                dirnames[:] = []
                return
            rel = os.path.relpath(current, root)
            depth = 0 if rel == "." else len(rel.split(os.sep))
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d not in {"__pycache__", "node_modules"}
            ]
            if depth >= int(max_depth):
                dirnames[:] = []
            if self._is_tsss_folder(current):
                yield current

    def _fif_files_in_folder(self, folder):
        """Return data FIF files directly inside a folder."""
        if not folder or not os.path.isdir(folder):
            return []
        candidates = []
        try:
            names = os.listdir(folder)
        except Exception:
            return []
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path) and self._is_analysis_fif_file(path):
                candidates.append(path)
        return sorted(candidates, key=self._fif_sort_key)

    def _find_related_fif_files(self, file_path):
        """Return sibling FIF files likely belonging to additional blocks."""
        folder = os.path.dirname(os.path.abspath(str(file_path)))
        selected = os.path.abspath(str(file_path))
        if not os.path.isdir(folder):
            return []

        candidates = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if os.path.abspath(path) == selected:
                continue
            if not self._is_analysis_fif_file(path):
                continue
            candidates.append(path)
        return sorted(candidates, key=self._fif_sort_key)

    def _candidate_other_patient_roots(self, tsss_folder):
        """Return possible patient-root ancestors above a selected tSSS folder."""
        roots = []
        current = os.path.dirname(os.path.abspath(str(tsss_folder)))
        for _ in range(4):
            parent = os.path.dirname(current)
            if not current or current == parent:
                break
            # Skip broad user/storage roots; scanning their siblings would be too noisy.
            basename = os.path.basename(current).lower()
            if basename not in {"", "users", "volumes", "library", "cloudstorage", "onedrive-mcw.edu"}:
                roots.append(current)
            current = parent
        return roots

    def _find_other_patient_tsss_fif_files(self, file_path, exclude_paths=None):
        """Return FIF files from tSSS folders under sibling patient folders."""
        selected_abs = os.path.abspath(str(file_path))
        selected_tsss_folder = self._nearest_tsss_folder(selected_abs)
        if not selected_tsss_folder:
            return []

        excluded = {selected_abs}
        for path in exclude_paths or []:
            if path:
                excluded.add(os.path.abspath(str(path)))

        candidates = []
        seen_paths = set(excluded)
        searched_tsss_folders = set()
        for patient_root in self._candidate_other_patient_roots(selected_tsss_folder):
            parent = os.path.dirname(patient_root)
            if not parent or not os.path.isdir(parent):
                continue
            try:
                sibling_names = os.listdir(parent)
            except Exception:
                continue
            sibling_dirs = [
                os.path.join(parent, name)
                for name in sibling_names
                if os.path.isdir(os.path.join(parent, name))
            ]
            if len(sibling_dirs) > 500:
                continue

            try:
                relative_tsss = os.path.relpath(selected_tsss_folder, patient_root)
            except Exception:
                relative_tsss = ""

            for sibling_root in sorted(sibling_dirs, key=lambda p: os.path.basename(p).lower()):
                sibling_root_abs = os.path.abspath(sibling_root)
                if sibling_root_abs == os.path.abspath(patient_root):
                    continue

                tsss_folders = []
                if relative_tsss and not relative_tsss.startswith(".."):
                    matching_folder = os.path.join(sibling_root_abs, relative_tsss)
                    if os.path.isdir(matching_folder) and self._is_tsss_folder(matching_folder):
                        tsss_folders.append(matching_folder)

                if not tsss_folders:
                    tsss_folders.extend(
                        list(self._iter_tsss_folders_under(sibling_root_abs, max_depth=3, max_dirs=250))
                    )

                for tsss_folder in tsss_folders:
                    tsss_abs = os.path.abspath(tsss_folder)
                    if tsss_abs == os.path.abspath(selected_tsss_folder) or tsss_abs in searched_tsss_folders:
                        continue
                    searched_tsss_folders.add(tsss_abs)
                    for candidate in self._fif_files_in_folder(tsss_abs):
                        candidate_abs = os.path.abspath(candidate)
                        if candidate_abs in seen_paths:
                            continue
                        seen_paths.add(candidate_abs)
                        candidates.append(candidate_abs)
        return sorted(candidates, key=lambda p: (os.path.dirname(p).lower(), self._fif_sort_key(p)))

    def _preview_fif_file_groups(self, fif_files, max_groups=4, max_files_per_group=3):
        """Build a short dialog preview grouped by parent folder."""
        groups = {}
        for path in fif_files or []:
            folder = os.path.dirname(os.path.abspath(str(path)))
            groups.setdefault(folder, []).append(path)
        lines = []
        for idx, folder in enumerate(sorted(groups.keys(), key=lambda p: p.lower())):
            if idx >= int(max_groups):
                lines.append(f"... plus {len(groups) - int(max_groups)} more folder(s).")
                break
            files = sorted(groups[folder], key=self._fif_sort_key)
            names = ", ".join(os.path.basename(p) for p in files[: int(max_files_per_group)])
            if len(files) > int(max_files_per_group):
                names += ", ..."
            lines.append(f"{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}: {names}")
        return "\n".join(lines) if lines else "No preview available."
    
    #region Orchestration

    def init_orchestrator(self):
        """
         Initialize DSP and add it to the plot widget.
        """
        try:
            self.orchestrator = Orchestrator(sig_view=self.dock_area)
            self.orchestrator.c.progress.connect(self.update_progress)
        except:
            print(traceback.format_exc())

    #endregion Orchestration

    # region Main window events

    def update_progress(self, value=None, msg='', append=False):
        try:
            if value is None:
                return

            self.pbTask.setValue(value)
            
            if append:
                curr_text = self.statusBar().currentMessage()
                if curr_text.lower().startswith("subscr") and not curr_text.endswith(' |'):
                    self.statusBar().showMessage(curr_text + ' | ' + msg + ' |')
                elif curr_text.lower().startswith("subscr") and curr_text.endswith(' |'):
                    # remove text after second to last |
                    parts = curr_text.split(' | ')
                    if len(parts) > 2:
                        new_text = ' | '.join(parts[:-2]) + ' | ' + msg + ' |'
                    else:
                        new_text = msg + ' |'
                    self.statusBar().showMessage(new_text)
                else:
                    self.statusBar().showMessage(msg)
            else:
                self.statusBar().showMessage(msg)
        except:
            print(traceback.format_exc())

    def keyPressEvent (self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == QtCore.Qt.Key_F1:
            print('Help')
        elif key == QtCore.Qt.Key_F5:
            print('Refresh')
        elif key == QtCore.Qt.Key_Left:
            self.orchestrator.navigate_fif(direction='back')
        elif key == QtCore.Qt.Key_Up:
            print('Up')
        elif key == QtCore.Qt.Key_T:
            self.orchestrator.navigate_fif(direction='forward')
        elif key == QtCore.Qt.Key_Right:
            self.orchestrator.navigate_fif(direction='forward')
        elif key == QtCore.Qt.Key_Down:
            print('Down')

    def closeEvent(self, event):
        """
         Close the window and close pyvista plotter
         
         Args:
         	 event: 
        """
        try:
            if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, 'meg_3d'):
                self.orchestrator.meg_3d._teardown_renderer()
            if hasattr(self, 'plotter_3d'):
                self.plotter_3d.close()
        except:
            print(traceback.format_exc())
        event.accept()

    #endregion Main window events

    #region 3D Image data

    def init_3d_viewer(self):
        self.tabImgData.setLayout(self.vLytImgData3D)

        # # Add a pyvista widget to display 3D image data
        # self.plotter_3d = QtInteractor()
        # self.vLytImgData3D.addWidget(self.plotter_3d.interactor)


    def load_img_data(self, folder_path=None):
        if folder_path is None:
            return
        
        # use pyvista to load CT series data
        self.img_data = pv.DICOMReader(folder_path).read()

        self.img_data_mesh = self.img_data.contour(method='marching_cubes')

        self.plotter_3d.add_mesh(self.img_data_mesh, color='tan')

    #endregion 3D Image data

if __name__ == "__main__":
    # Configure multiprocessing for macOS to prevent dock icon duplication
    if sys.platform == "darwin":
        mp.set_start_method('spawn', force=True)
    
    # Import macOS compatibility functions
    from macos_qt_fixes import configure_app_for_macos, set_macos_app_identity, protect_app_name, patch_mne_qtapp
    
    # Set macOS bundle identity early (before QApplication creation)
    if sys.platform == "darwin":
        set_macos_app_identity("MAS")
    protect_app_name("MAS")
    patch_mne_qtapp("MAS")
    
    # Set process priority to HIGH for better timing accuracy
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == "win32":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            # On Unix-like systems, use negative nice values for higher priority
            # Range: -20 (highest) to 19 (lowest), default is 0
            # Using -10 for high priority (requires appropriate permissions)
            p.nice(-10)
        print("Process priority set to HIGH.")
    except Exception as e:
        print(f"Could not set process priority: {e}")
    
    # Set Qt application attributes before creating QApplication (macOS)
    if sys.platform == "darwin":        
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    
    app = QApplication([])
    
    # Configure application properties
    app.setOrganizationName("Wired Brains Lab")
    app.setOrganizationDomain("mcw.edu")
    app.setApplicationName("MAS")
    app.setApplicationDisplayName("MAS")
    app.setApplicationVersion("1.0")

    # Apply macOS-specific configuration after app creation
    if sys.platform == "darwin":
        configure_app_for_macos(app)
    
    window = MAS()
    window.show()
    window.setWindowState(window.windowState() | QtCore.Qt.WindowMaximized)
    logo_path = os.path.abspath(os.path.join(app_path, 'wbl.png'))
    if os.path.exists(logo_path):
        app.setWindowIcon(QtGui.QIcon(logo_path))
    window.setWindowTitle('MEG Analysis Suite')
    app.exec_()



