"""
Power spectrum and notch detection
=================================

Compute power spectral density (PSD) and detect narrowband interference
("notches") across channels, with multiprocessing support.
"""
# Author: Sunil Mathew
# Date: June 14, 2026
# Calculate power spectral density of the neural data
# based on codes_emu/cx_lfp.m

import time
import traceback
import numpy as np
import os
import multiprocessing as mp
import pickle
from scipy import signal, ndimage
from scipy.ndimage import gaussian_filter1d

from qtpy import QtWidgets, QtCore
import pyqtgraph as pg

from core.config import config
from core.shared_buffer import buffer_manager
from dsp.psd_proc import get_processor, start_processor, stop_processor, is_processor_running

# Multiprocessing start method is configured in psd_proc.py


def process_single_channel_notches(args):
    """Compute PSD and detect notches for a single channel.

    Designed to run in a worker process.

    Parameters
    ----------
    args : tuple
        `(ch_label, data, processing_params)` where `processing_params`
        contains keys: `fs`, `barthannwin`, `freq_min`, `freq_max`,
        `psd_1_x`, `span_smooth`, `db_thr`, and `filtering_params`.

    Returns
    -------
    tuple
        `(ch_label, psd_data, notches_data, filter_info)` where `psd_data`
        has `freq`, `psd_raw`, `psd_filtered`, and `notches_data` has
        `notch_freqs` and detailed entries.
    """
    try:
        ch_label, data, params = args
        
        # Unpack parameters
        fs = params['fs']
        barthannwin = params['barthannwin']
        freq_min = params['freq_min']
        freq_max = params['freq_max']
        psd_1_x = params['psd_1_x']
        span_smooth = params['span_smooth']
        db_thr = params['db_thr']
        filtering_params = params['filtering_params']
        
        # Import filtering here to avoid pickling issues
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from dsp.filtering import Filtering
        
        # Create filtering instance
        filtering = Filtering(**filtering_params)
        
        # Apply filtering to get filtered data
        filtered_data = filtering.custom_bp_filter(data=data, fs=fs, channel=ch_label)
        
        # Calculate PSD for raw data
        f_raw, Pxx_den_raw = signal.welch(x=data, fs=fs, window=barthannwin, nfft=len(data))
        pxx_db_raw = 10*np.log10(Pxx_den_raw + 1e-12)  # Add small value to avoid log(0)
        
        # Calculate PSD for filtered data
        f, Pxx_den = signal.welch(x=filtered_data, fs=fs, window=barthannwin, nfft=len(filtered_data))
        pxx_db_filtered = 10*np.log10(Pxx_den + 1e-12)
        
        # Store PSD data for this channel (limit to display range)
        idx_display = np.where(f <= psd_1_x)
        psd_data = {
            'freq': f[idx_display],
            'psd_raw': pxx_db_raw[idx_display],
            'psd_filtered': pxx_db_filtered[idx_display]
        }
        
        # Calculate notches using frequency range of interest
        idx_notch = np.where((f >= freq_min) & (f <= freq_max))
        f_filtered = f[idx_notch]
        pxx_db_filtered_range = pxx_db_filtered[idx_notch]
        
        # Calculate notches for this channel
        detected_notches = _calculate_notches_for_single_channel(
            pxx_db_filtered_range, f_filtered, ch_label, span_smooth, db_thr
        )
        
        # Extract frequencies for backward compatibility
        notch_freqs = [notch['freq'] for notch in detected_notches]
        
        # Store detailed notch data for this channel
        notches_data = {
            'notch_freqs': notch_freqs,
            'notch_details': detected_notches,
            'notch_lines': []
        }
        
        # Create notch filter coefficients for this channel
        filter_info = None
        if len(detected_notches) > 0:
            filter_info = _create_notch_filter_for_channel(detected_notches, fs, ch_label)
        
        return ch_label, psd_data, notches_data, filter_info
        
    except Exception as e:
        print(f"Error processing channel {ch_label}: {e}")
        import traceback
        traceback.print_exc()
        return ch_label, None, None, None


def _calculate_notches_for_single_channel(pxx_db, frequency_array, ch_label, span_smooth, db_thr):
    """Detect narrowband notches above a smoothed PSD baseline.

    Parameters
    ----------
    pxx_db : ndarray
        PSD values in dB for the frequency range of interest.
    frequency_array : ndarray
        Corresponding frequency bins (Hz).
    ch_label : str
        Channel label, used for logging only.
    span_smooth : int
        Window size for median smoothing of the baseline.
    db_thr : float
        dB amount above the smoothed baseline to flag as a notch.

    Returns
    -------
    list[dict]
        Each dict contains `freq`, `bandwidth`, `amplitude`, and metadata.
    """
    try:
        # Create smoothed threshold line using moving median + threshold
        smoothed_threshold_db = ndimage.median_filter(pxx_db, size=span_smooth) + db_thr

        # Handle frequencies above 2950 Hz - keep threshold constant (like MATLAB)
        high_freq_indices = np.where(frequency_array >= 2950)[0]
        if len(high_freq_indices) > 0:
            smoothed_threshold_db[high_freq_indices[0]:] = smoothed_threshold_db[high_freq_indices[0]]

        detected_notches = []
        
        # Find all frequency bins where power exceeds threshold
        above_threshold_indices = np.where(pxx_db > smoothed_threshold_db)[0]
        
        if len(above_threshold_indices) == 0:
            return []

        # Calculate maximum amplitude for normalization
        amplitude_differences = pxx_db - smoothed_threshold_db
        max_amplitude_above_threshold = np.max(amplitude_differences)
        
        # Find breaks in continuous regions
        region_break_indices = np.where(np.diff(above_threshold_indices) > 1)[0]
        
        # Determine start indices and lengths of each continuous region
        if len(region_break_indices) == 0:
            region_start_indices = [above_threshold_indices[0]]
            region_lengths = [len(above_threshold_indices)]
        else:
            region_start_indices = [above_threshold_indices[0]]
            region_start_indices.extend([above_threshold_indices[idx + 1] for idx in region_break_indices])
            
            region_lengths = []
            region_lengths.append(region_break_indices[0] + 1)
            for i in range(1, len(region_break_indices)):
                region_lengths.append(region_break_indices[i] - region_break_indices[i-1])
            last_region_start_pos = np.where(above_threshold_indices == region_start_indices[-1])[0][0]
            region_lengths.append(len(above_threshold_indices) - last_region_start_pos)

        # Process each significant region
        significant_region_indices = [i for i, length in enumerate(region_lengths) if length > 1]
        
        for region_idx in significant_region_indices:
            region_start = region_start_indices[region_idx]
            region_length = region_lengths[region_idx]
            
            region_end = region_start + region_length - 1
            current_region_indices = above_threshold_indices[
                (above_threshold_indices >= region_start) & (above_threshold_indices <= region_end)
            ]
            
            if len(current_region_indices) == 0:
                continue
            
            # Find the peak within this region
            region_power_values = pxx_db[current_region_indices]
            peak_idx_within_region = np.argmax(region_power_values)
            peak_frequency_index = current_region_indices[peak_idx_within_region]
            
            # Calculate center frequency index
            center_frequency_index = np.mean(current_region_indices)
            
            if center_frequency_index % 1 == 0.5:
                floor_idx = int(np.floor(center_frequency_index))
                ceil_idx = int(np.ceil(center_frequency_index))
                if (floor_idx < len(pxx_db) and ceil_idx < len(pxx_db) and 
                    pxx_db[floor_idx] > pxx_db[ceil_idx]):
                    center_frequency_index = floor_idx
                else:
                    center_frequency_index = ceil_idx
            else:
                center_frequency_index = int(np.ceil(center_frequency_index))
            
            center_frequency_index = max(0, min(center_frequency_index, len(frequency_array) - 1))
            
            # Calculate amplitude and bandwidth
            peak_amplitude_above_threshold = pxx_db[peak_frequency_index] - smoothed_threshold_db[peak_frequency_index]
            
            frequency_resolution = frequency_array[1] - frequency_array[0] if len(frequency_array) > 1 else 1.0
            base_bandwidth = frequency_resolution * region_length * 2 * peak_amplitude_above_threshold / max_amplitude_above_threshold
            
            # Aggressive bandwidth scaling
            amplitude_ratio = peak_amplitude_above_threshold / max_amplitude_above_threshold
            if amplitude_ratio > 0.8:
                bandwidth_multiplier = 3.0
            elif amplitude_ratio > 0.5:
                bandwidth_multiplier = 2.0
            elif amplitude_ratio > 0.3:
                bandwidth_multiplier = 1.5
            else:
                bandwidth_multiplier = 1.0
            
            aggressive_bandwidth = base_bandwidth * bandwidth_multiplier
            
            # Apply frequency-dependent bandwidth limits
            center_frequency = frequency_array[center_frequency_index]
            if center_frequency < 295:
                max_bandwidth = 3.0
            else:
                max_bandwidth = 5.0
            
            final_bandwidth = max(1.0, min(aggressive_bandwidth, max_bandwidth))
            
            # Store notch information
            notch_info = {
                'freq': center_frequency,
                'bandwidth': final_bandwidth,
                'amplitude': peak_amplitude_above_threshold,
                'peak_power_db': pxx_db[peak_frequency_index],
                'amplitude_ratio': amplitude_ratio
            }
            detected_notches.append(notch_info)

        return detected_notches
        
    except Exception as e:
        print(f'Error calculating notches for channel {ch_label}: {e}')
        return []


def _create_notch_filter_for_channel(detected_notches, fs, ch_label):
    """Create combined IIR notch filter for a channel.

    Parameters
    ----------
    detected_notches : list[dict]
        Notches with center `freq` and `bandwidth` in Hz.
    fs : float
        Sampling frequency (Hz).
    ch_label : str
        Channel identifier.

    Returns
    -------
    dict or None
        Filter coefficients info suitable for later application, or None
        if no notches were provided.
    """
    try:
        # Extract channel ID from label
        ch_id_match = ch_label.split('_')[-1]
        try:
            ch_id = int(ch_id_match)
        except ValueError:
            ch_id = hash(ch_label) % 1000
        
        # Create combined notch filter using individual bandwidths
        all_zeros = np.array([])
        all_poles = np.array([])
        combined_gain = 1.0
        
        for notch in detected_notches:
            freq = notch['freq']
            bandwidth = notch['bandwidth']
            
            if freq > 0 and freq < fs/2:
                Q_factor = freq / bandwidth
                b_notch, a_notch = signal.iirnotch(w0=freq, Q=Q_factor, fs=fs)
                zeros, poles, gain = signal.tf2zpk(b_notch, a_notch)
                
                all_zeros = np.append(all_zeros, zeros)
                all_poles = np.append(all_poles, poles)
                combined_gain *= gain
        
        if len(all_zeros) > 0:
            sos_filter = signal.zpk2sos(all_zeros, all_poles, combined_gain)
            
            return {
                'chID': ch_id,
                'ch_label': ch_label,
                'notch_details': detected_notches,
                'SOS': sos_filter,
                'G': combined_gain,
                'total_notches': len(detected_notches)
            }
        else:
            return None
            
    except Exception as e:
        print(f'Error creating notch filter for channel {ch_label}: {e}')
        return None

class PowerSpectrum():

    def __init__(self, psd_view, c, filtering) -> None:
        self.psd_view = psd_view
        self.c = c
        self.filtering = filtering

        # Power spectrum parameters
        self.fs = 30000
        self.freq_min = 300 # Hz
        self.freq_max = 3000 # Hz
        self.psd_1_x = 10000
        self.psd_2_x = 300
        self.init_fft_params()

        # Notch filter parameters
        # self.basic_notches = [60, 120]
        # self.notch_width = 1
        self.span_smooth = 21
        self.db_thr = 5 # dB
        
        # PSD shared buffer and auto-refresh
        self.shared_psd_buffer = None
        self.auto_refresh_enabled = True
        self.apply_notches = True
        self.view_notches = True
        self.show_raw_psd = True
        self.show_filtered_psd = True
        self.show_before_after_notches = False  # Show PSD before and after notch filtering
        self.show_running_average = True
        self.selected_channel_idx = 0
        self.current_channel_name = ""
        
        # Storage for PSD running averages and notches data
        self.running_average_psd = {}   # {channel_name: {'freq': freq_array, 'psd_avg': running_avg_array, 'count': int}}
        self.current_psd = {}          # {channel_name: {'freq': freq_array, 'psd_raw': current_raw, 'psd_filtered': current_filtered}}
        self.all_channel_notches = {}   # {channel_name: {'notch_freqs': [freq1, freq2, ...], 'notch_lines': []}}
        self.psd_calculation_count = 0  # Track number of PSD calculations for notch button enabling
        self.k_bartlett_target = 50     # Target number of PSD calculations before enabling notch calculation

        # Progress dialog for multiprocessing calculations
        self.progress_dialog = None
        
        # Persistent PSD processor
        self.psd_result_check_timer = QtCore.QTimer()
        self.psd_result_check_timer.timeout.connect(self._check_psd_results)
        self.pending_psd_tasks = {}  # {task_id: metadata}

        self.init_power_spectrum_layout()
    
    @staticmethod
    def get_system_info():
        """Get information about the system's parallel processing capabilities"""
        total_cores = mp.cpu_count()
        available_cores = max(1, total_cores - 1)  # Leave one core free
        
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_info = f", {memory_gb:.1f}GB RAM"
        except ImportError:
            memory_info = ""
        
        return {
            'total_cores': total_cores,
            'available_cores': available_cores,
            'memory_info': memory_info
        }
    
    def show_parallel_processing_info(self):
        """Show information about parallel processing capabilities"""
        info = self.get_system_info()
        
        message = (
            f"Multiprocessing Information:\n\n"
            f"• Total CPU cores: {info['total_cores']}\n"
            f"• Cores available for processing: {info['available_cores']}\n"
            f"• Expected speedup: {info['available_cores']:.1f}x\n"
            f"• System memory{info['memory_info']}\n\n"
            f"Mode: Multiprocessing\n"
            f"Uses separate processes for true parallel processing. "
            f"Provides better performance for CPU-intensive tasks.\n\n"
            f"The notch calculation will use multiple CPU cores to process "
            f"channels simultaneously, significantly reducing computation time."
        )
        
        QtWidgets.QMessageBox.information(
            self.psd_widget,
            "Parallel Processing Info",
            message
        )
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_resources()
    
    def cleanup_resources(self):
        """Cleanup resources"""
        try:
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
                
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def init_fft_params(self):
        self.fft_length = 2**np.ceil(np.log2(self.fs*2))/self.fs
        self.nfft = int(self.fft_length * self.fs)
        self.n_blocks = 5
        self.n_samples = self.n_blocks * self.fft_length * self.fs
        self.barthannwin = signal.windows.barthann(self.nfft)

    def init_power_spectrum_layout(self):
        """
        Initializes the power spectrum plots.
        """
        self.psd_widget = QtWidgets.QWidget()
        self.psd_layout = QtWidgets.QVBoxLayout()
        self.psd_widget.setLayout(self.psd_layout)

        # Controls
        self.psd_params_layout = QtWidgets.QHBoxLayout()
        self.psd_layout.addLayout(self.psd_params_layout)
        # Add a spacer
        self.psd_params_layout.addStretch()

        for param in config['psd']['cmb_params']:
            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(QtWidgets.QLabel(param))
            cmb = QtWidgets.QComboBox(self.psd_widget)
            cmb.setObjectName(f'cmb{param}')
            cmb.addItems(config['psd']['cmb_params'][param])
            if param == 'channel':
                self.cmb_channel = cmb
                cmb.currentIndexChanged.connect(self.on_channel_selected)
            else:
                cmb.currentIndexChanged.connect(self.update_psd_params)
            vLyt.addWidget(cmb)
            self.psd_params_layout.addLayout(vLyt)
        
        self.init_psd_view_spb()

        self.hLyt_psd_params = QtWidgets.QHBoxLayout()
        self.psd_params_layout.addLayout(self.hLyt_psd_params)
        self.init_psd_chks()

        # Acq status label
        self.lbl_acq_status = QtWidgets.QLabel('k_bartlett: 0/50 (2.2s/k)')
        self.hLyt_psd_params.addWidget(self.lbl_acq_status)

        # Calculate notches button
        self.btn_calculate_notches = QtWidgets.QPushButton('Calculate Notches')
        self.btn_calculate_notches.clicked.connect(self.start_all_channel_notch_calculation)
        self.btn_calculate_notches.setEnabled(False)
        self.btn_calculate_notches.setToolTip('Calculate notches using 50 averaged PSDs. Enabled after 50 PSD calculations.')
        self.hLyt_psd_params.addWidget(self.btn_calculate_notches)
        
        # System info button
        # self.btn_system_info = QtWidgets.QPushButton('System Info')
        # self.btn_system_info.clicked.connect(self.show_parallel_processing_info)
        # self.btn_system_info.setMaximumWidth(80)
        # self.psd_params_layout.addWidget(self.btn_system_info)

        # Acquire button
        # self.btn_acquire = QtWidgets.QPushButton('Acquire')
        # self.btn_acquire.clicked.connect(self.start_acquisition)
        # self.psd_params_layout.addWidget(self.btn_acquire)
        
        # Save PSD Data button
        # self.btn_save_psd = QtWidgets.QPushButton('Save PSD Data')
        # self.btn_save_psd.clicked.connect(self.save_all_psd_data)
        # self.psd_params_layout.addWidget(self.btn_save_psd)
        
        # Clear Data button
        self.btn_clear_data = QtWidgets.QPushButton('Clear Data')
        self.btn_clear_data.clicked.connect(self.clear_stored_data)
        self.btn_clear_data.setEnabled(False)
        self.hLyt_psd_params.addWidget(self.btn_clear_data)

        # Status label
        self.status_label = QtWidgets.QLabel('')
        self.psd_layout.addWidget(self.status_label)

        self.psd_plot_tabs = QtWidgets.QTabWidget(self.psd_widget)
        self.psd_layout.addWidget(self.psd_plot_tabs)

        # MAS native PSD plots (pyqtgraph)
        self.psd_tab_native = QtWidgets.QWidget(self.psd_widget)
        self.psd_tab_native_layout = QtWidgets.QVBoxLayout()
        self.psd_tab_native_layout.setContentsMargins(0, 0, 0, 0)
        self.psd_tab_native.setLayout(self.psd_tab_native_layout)
        self.glw_psd = pg.GraphicsLayoutWidget()
        self.psd_tab_native_layout.addWidget(self.glw_psd)
        self.psd_plot_tabs.addTab(self.psd_tab_native, "MAS PSD")

        # MNE tutorial-style PSD figure (matplotlib)
        self.psd_tab_mne = QtWidgets.QWidget(self.psd_widget)
        self.psd_tab_mne_layout = QtWidgets.QVBoxLayout()
        self.psd_tab_mne_layout.setContentsMargins(0, 0, 0, 0)
        self.psd_tab_mne.setLayout(self.psd_tab_mne_layout)
        self.lbl_mne_psd_placeholder = QtWidgets.QLabel(
            "Load a .fif file to display MNE-style PSD plots."
        )
        self.lbl_mne_psd_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.psd_tab_mne_layout.addWidget(self.lbl_mne_psd_placeholder)
        self._mne_psd_canvas = None
        self._mne_psd_toolbar = None
        self.psd_plot_tabs.addTab(self.psd_tab_mne, "MNE PSD")

        self.init_psd_plots()

    def init_psd_chks(self):
        """Initializes the PSD checkboxes."""

        # Acquire checkbox
        self.chk_acquire = QtWidgets.QCheckBox('acquire', self.psd_widget)
        self.chk_acquire.setChecked(False)
        self.chk_acquire.setToolTip('Enable/Disable PSD Acquisition. Requires clicking "Start Acq" button on the signal viewer.')
        self.chk_acquire.stateChanged.connect(self.enable_disable_psd_acquisition)
        self.psd_params_layout.addWidget(self.chk_acquire)

        chk_params = config['psd']['chk_params']
        params = list(chk_params.keys())
        # Group two checkboxes per vertical column to save space
        for i in range(0, len(params), 2):
            vcol = QtWidgets.QVBoxLayout()
            vcol.setContentsMargins(0, 0, 0, 0)
            vcol.setSpacing(0)
            for j in range(2):
                idx = i + j
                if idx >= len(params):
                    break
                param = params[idx]
                chk = QtWidgets.QCheckBox(param, self.psd_widget)
                chk.setChecked(chk_params[param])
                chk.stateChanged.connect(self.update_psd_view_chk_params)
                vcol.addWidget(chk)
            self.hLyt_psd_params.addLayout(vcol)

    def update_psd_view_chk_params(self, state):
        """
         Update psd chk params. This is called when the user changes the value of a checkbox
         
         Args:
         	 state: The state of the checkbox
        """
        try:
            chk = self.psd_widget.sender()
            param = chk.text()
            if param == 'raw':
                self.enable_disable_raw(state)
            elif param == 'filtered':
                self.enable_disable_filtered(state)
            elif param == 'show_avg':
                self.enable_disable_show_avg(state)
            elif param == 'auto_refresh':
                self.enable_disable_auto_refresh(state)
            elif param == 'apply_notches':
                self.enable_disable_apply_notches(state)
            elif param == 'view_notches':
                self.enable_disable_view_notches(state)
            elif param == 'matlab_notches':
                try:
                    use_mat = (state == QtCore.Qt.Checked)
                    if hasattr(self.filtering, 'set_use_matlab_notches'):
                        self.filtering.set_use_matlab_notches(use_mat)
                    else:
                        # Fallback: set attribute directly if method not available
                        self.filtering.use_matlab_notches = use_mat
                except Exception as e:
                    print(f"Error toggling MATLAB notches: {e}")
                
        except Exception as e:
            print(f"Error updating PSD checkbox params: {e}")
        
    def init_psd_plots(self):
        """
        Initialize the plots
        """
        # 0-10kHz
        self.psd_graph_1 = self.glw_psd.addPlot()
        self.psd_graph_1.showGrid(x=True, y=True, alpha=0.5)
        self.psd_graph_1.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_graph_1.setLabel('left', 'Power Spectral Density', units='dB/Hz')
        self.psd_graph_1.setTitle("Power Spectral Density", color='k', size='10pt')
        
        # Add legend to first plot (must be before adding plot items) - positioned at top right
        self.psd_graph_1.addLegend(offset=(-1, 1))

        self.psd_curve_1 = self.psd_graph_1.plot(name='Raw', pen=pg.mkPen(color='b', width=1.5)) # psd
        self.psd_curve_2 = self.psd_graph_1.plot(name='Filtered', pen=pg.mkPen(color='r', width=1.5)) # filtered psd

        self.glw_psd.nextRow()

        # 0-300 Hz
        self.psd_graph_2 = self.glw_psd.addPlot()
        self.psd_graph_2.showGrid(x=True, y=True, alpha=0.5)
        self.psd_graph_2.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_graph_2.setLabel('left', 'Power Spectral Density', units='dB/Hz')
        self.psd_graph_2.setTitle("Power Spectral Density", color='k', size='10pt')
        
        # Add legend to second plot (must be before adding plot items) - positioned at top right
        self.psd_graph_2.addLegend(offset=(-1, 1))

        self.psd_curve_300_1 = self.psd_graph_2.plot(name='Raw', pen=pg.mkPen(color='b', width=1.5)) # psd
        self.psd_curve_300_2 = self.psd_graph_2.plot(name='Filtered', pen=pg.mkPen(color='r', width=1.5)) # filtered psd

        # db_thr line
        self.psd_smooth_curve = self.psd_graph_1.plot()
        self.psd_smooth_curve.setPen(color='g', width=1.5)

        self.notch_lines = []

        # glw_layout = self.glw.ci.layout
        # glw_layout.setColumnStretchFactor(0, 2)
        # glw_layout.setColumnStretchFactor(1, 1)

        self.psd_view.addWidget(self.psd_widget)

    def update_psd_progress(self, mode: str, k: int, target: int, nfft: int = None):
        """Update PSD acquisition progress.

        Args:
            mode: 'all' for all-channel acquisition
            k: number of Bartlet windows accumulated
            target: target number of windows for this data batch
            nfft: window length in samples (optional; used to render seconds)
        """
        try:
            # Update status label
            secs = 0.0
            if nfft and self.fs:
                secs = float(nfft) / float(self.fs)
            self.lbl_acq_status.setText(f'k_bartlett: {self.psd_calculation_count * 5} ({secs:.1f}s/k)')

            # Enable the Calculate Notches button when we have enough averaged PSDs
            self.btn_calculate_notches.setEnabled(self.psd_calculation_count * 5 >= self.k_bartlett_target)
            
        except Exception as e:
            print(f"Error updating PSD progress UI: {e}")

    def init_psd_view_spb(self):
        """
            Initialize spinboxes for the signal viewer. This is called by init_data
        """
        spb_params = config['psd']['spb_params']
        for param in spb_params:
            # Spinbox for clustering parameters, use the config values to figure out if doublespinbox or spinbox
            if isinstance(spb_params[param][0], float):                       
                spb = QtWidgets.QDoubleSpinBox(self.psd_widget)               
            else:
                spb = QtWidgets.QSpinBox(self.psd_widget)

            spb.setObjectName(f'spb_psd_{param}')
            spb.setRange(spb_params[param][1], spb_params[param][2])
            spb.setValue(spb_params[param][0])
            spb.setSingleStep(spb_params[param][3])
            spb.valueChanged.connect(self.update_psd_view_spb_params)
            lbl = QtWidgets.QLabel(param)

            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(lbl)
            vLyt.addWidget(spb)
            self.psd_params_layout.addLayout(vLyt)

    def update_psd_view_spb_params(self, value=None):
        """
         Update psd filt order params. This is called when the user changes the value of a spinbox
         
         Args:
         	 value: The value of the spinbox
        """
        try:
            spb = self.psd_widget.sender()
            param = spb.objectName().split('spb_psd_')[1]
            value = spb.value()
            setattr(self,param,value)
            self.c.sig_view_filt_param.emit(param, value)
        except:
            print(traceback.format_exc())

    def update_psd_params(self, idx):
        """
        Updates the power spectrum parameters based on user input.
        """
        cmb = self.psd_widget.sender()
        param = cmb.objectName().replace('cmb', '')
        value = cmb.currentText().split(' ')[0]
        if cmb.currentText().split(' ')[1] == 'Hz':
            value = int(value)
        elif cmb.currentText().split(' ')[1] == 'kHz':
            value = int(value) * 1000
        setattr(self, param, int(value))

        if param == 'psd_1':
            self.psd_graph_1.setXRange(0, self.psd_1_x)
        elif param == 'psd_2':
            self.psd_graph_2.setXRange(0, self.psd_2_x)

    def enable_disable_apply_notches(self, state):
        """
        Enables or disables applying calculated notches to filtering and PSD display.
        """
        if state == QtCore.Qt.Checked:
            self.apply_notches = True
            if hasattr(self.filtering, 'set_apply_notches'):
                self.filtering.set_apply_notches(True)
        else:
            self.apply_notches = False
            if hasattr(self.filtering, 'set_apply_notches'):
                self.filtering.set_apply_notches(False)
        
        # Update current PSD data with/without notches applied
        self._refresh_current_psd_with_notches()
        
        # Refresh current plot to show/hide notch effects
        self._refresh_current_plot()
        
        # Print status for user feedback
        if self.apply_notches:
            total_channels_with_notches = sum(1 for ch_data in self.all_channel_notches.values() 
                                            if len(ch_data.get('notch_freqs', [])) > 0)
            print(f"Notch filtering ENABLED - applying calculated notches to {total_channels_with_notches} channels")
        else:
            print("Notch filtering DISABLED - showing original filtered data")

    def enable_disable_view_notches(self, state):
        """
        Enables or disables viewing calculated notches on the PSD plots.
        """
        if state == QtCore.Qt.Checked:
            self.view_notches = True
            # Show notch lines for the selected channel
            current_channel = self.cmb_channel.currentText()
            self._update_notch_lines_for_channel(current_channel)
        else:
            self.view_notches = False
            # Remove all notch lines from the plots
            for line in self.notch_lines:
                self.psd_graph_1.removeItem(line)
            self.notch_lines = []

    def enable_disable_raw(self, state):
        """
        Enables or disables displaying raw PSD.
        """
        if state == QtCore.Qt.Checked:
            self.show_raw_psd = True
        else:
            self.show_raw_psd = False
        
        # Refresh the current plot
        self._refresh_current_plot()

    def enable_disable_filtered(self, state):
        """
        Enables or disables displaying filtered PSD.
        """
        if state == QtCore.Qt.Checked:
            self.show_filtered_psd = True
        else:
            self.show_filtered_psd = False
        
        # Refresh the current plot
        self._refresh_current_plot()

    def enable_disable_show_avg(self, state):
        """
        Enables or disables displaying running average PSD.
        """
        if state == QtCore.Qt.Checked:
            self.show_running_average = True
        else:
            self.show_running_average = False
        
        # Refresh the current plot
        self._refresh_current_plot()

    def enable_disable_auto_refresh(self, state):
        """
        Enables or disables auto refresh of the power spectrum plots.
        """
        if state == QtCore.Qt.Checked:
            self.auto_refresh_enabled = True
        else:
            self.auto_refresh_enabled = False

    def enable_disable_psd_acquisition(self, state):
        """
        Enables or disables PSD data acquisition with persistent processor.
        """
        if state == QtCore.Qt.Checked:
            # Start persistent PSD processor
            if not is_processor_running():
                if start_processor():
                    print("Started persistent PSD processor")
                else:
                    print("Failed to start PSD processor")
                    QtWidgets.QMessageBox.warning(
                        self.psd_widget,
                        "PSD Processor Error",
                        "Failed to start the PSD processor. PSD calculations may not work properly."
                    )
            
            # Start result checking timer
            self.psd_result_check_timer.start(100)  # Check every 100ms
            
            self.c.start_all_ch_psd_acq.emit()
            # enable all widgets/layouts inside hLyt_psd_params
            self._set_layout_enabled(self.hLyt_psd_params, True)
        else:
            # Stop result checking timer
            self.psd_result_check_timer.stop()
            
            # Stop PSD processor when acquisition is disabled
            if is_processor_running():
                stop_processor()
                print("Stopped persistent PSD processor")
            
            self.c.stop_all_ch_psd_acq.emit()
            # disable all widgets/layouts inside hLyt_psd_params
            self._set_layout_enabled(self.hLyt_psd_params, False)

    def _set_layout_enabled(self, layout, enabled):
        """Recursively enable/disable all widgets within a layout."""
        try:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setEnabled(enabled)
                else:
                    child_layout = item.layout()
                    if child_layout is not None:
                        self._set_layout_enabled(child_layout, enabled)
        except Exception as e:
            print(f"Error setting layout enabled state: {e}")
    
    def _check_psd_results(self):
        """Check for completed PSD calculation results from the processor"""
        if not is_processor_running():
            return
            
        processor = get_processor()
        results = processor.get_all_results(timeout=0.01)  # Non-blocking check
        
        for result in results:
            if result.task_id in self.pending_psd_tasks:
                task_info = self.pending_psd_tasks[result.task_id]
                
                if result.success:
                    # Process successful PSD calculation
                    self._process_psd_result(result, task_info)
                else:
                    print(f"PSD calculation failed: {result.error_message}")
                
                # Remove completed task
                del self.pending_psd_tasks[result.task_id]
    
    def _process_psd_result(self, result, task_info):
        """Process a successful PSD calculation result"""
        try:
            psd_results = result.data.get('psd_results', {})
            
            # Update both current and running averages for each channel
            for ch_label, psd_data in psd_results.items():
                freq = psd_data['freq']
                psd_raw = psd_data['psd_raw']
                psd_filtered = psd_data['psd_filtered']
                
                # Apply calculated notches to current filtered PSD if available
                psd_filtered_with_notches = psd_filtered.copy()
                if (self.apply_notches and 
                    ch_label in self.all_channel_notches and 
                    len(self.all_channel_notches[ch_label].get('notch_freqs', [])) > 0):
                    
                    psd_filtered_with_notches = self._apply_notches_to_psd(
                        psd_filtered, freq, ch_label
                    )
                
                # Store current PSD data (with notches applied if enabled)
                self.current_psd[ch_label] = {
                    'freq': freq,
                    'psd_raw': psd_raw.copy(),
                    'psd_filtered': psd_filtered_with_notches.copy(),
                    'psd_filtered_original': psd_filtered.copy()  # Keep original for comparison
                }
                
                # Update running average
                if ch_label not in self.running_average_psd:
                    # Initialize for this channel
                    self.running_average_psd[ch_label] = {
                        'freq': freq,
                        'psd_raw_avg': psd_raw.copy(),
                        'psd_filtered_avg': psd_filtered.copy(),
                        'count': 1
                    }
                else:
                    # Update running average
                    current_data = self.running_average_psd[ch_label]
                    current_count = current_data['count']
                    
                    # Running average formula: new_avg = (old_avg * n + new_value) / (n + 1)
                    current_data['psd_raw_avg'] = (current_data['psd_raw_avg'] * current_count + psd_raw) / (current_count + 1)
                    current_data['psd_filtered_avg'] = (current_data['psd_filtered_avg'] * current_count + psd_filtered) / (current_count + 1)
                    current_data['count'] += 1
            
            # Increment global PSD calculation counter
            self.psd_calculation_count += 1
            
            # Update UI
            self.update_psd_progress('all', self.nfft, self.nfft, self.nfft)
            
            # Update channel selector if needed
            self._update_channel_selector()
            
            # Update display if auto-refresh is enabled and a channel is selected
            if self.auto_refresh_enabled:
                current_channel = self.cmb_channel.currentText()
                if current_channel and current_channel in self.running_average_psd:
                    self._display_running_average_psd(current_channel)
            
            print(f"PSD calculation {self.psd_calculation_count} complete. "
                  f"Running averages updated for {len(psd_results)} channels in {result.processing_time:.2f}s.")
            
        except Exception as e:
            print(f"Error processing PSD result: {e}")
            traceback.print_exc()
    
    def _apply_notches_to_psd(self, psd_filtered, freq_array, ch_label):
        """
        Apply calculated notches to PSD data for visualization.
        This simulates what the PSD would look like with notch filters applied.
        
        Args:
            psd_filtered: Original filtered PSD data
            freq_array: Frequency array
            ch_label: Channel label
            
        Returns:
            PSD data with notches applied (simulated)
        """
        try:
            if ch_label not in self.all_channel_notches:
                return psd_filtered
            
            notch_data = self.all_channel_notches[ch_label]
            notch_details = notch_data.get('notch_details', [])
            
            if not notch_details:
                return psd_filtered
            
            # Create a copy to modify
            psd_with_notches = psd_filtered.copy()
            
            # Apply each notch as an attenuation
            for notch in notch_details:
                notch_freq = notch['freq']
                bandwidth = notch.get('bandwidth', 2.0)  # Default 2 Hz bandwidth
                amplitude_reduction = notch.get('amplitude', 10.0)  # Default 10 dB reduction
                
                # Create a notch attenuation profile (Gaussian-like)
                # Find frequency indices within the notch bandwidth
                freq_diff = np.abs(freq_array - notch_freq)
                
                # Apply Gaussian attenuation centered at notch frequency
                sigma = bandwidth / 4.0  # Standard deviation for Gaussian
                attenuation = amplitude_reduction * np.exp(-0.5 * (freq_diff / sigma) ** 2)
                
                # Convert to linear scale, apply attenuation, convert back to dB
                psd_linear = 10 ** (psd_with_notches / 10.0)
                attenuation_linear = 10 ** (-attenuation / 10.0)
                psd_linear *= attenuation_linear
                psd_with_notches = 10 * np.log10(psd_linear + 1e-12)
            
            return psd_with_notches
            
        except Exception as e:
            print(f"Error applying notches to PSD for {ch_label}: {e}")
            return psd_filtered
    
    def _refresh_current_psd_with_notches(self):
        """Refresh current PSD data to apply/remove notches based on apply_notches flag."""
        try:
            for ch_label, psd_data in self.current_psd.items():
                if 'psd_filtered_original' in psd_data:
                    # Use original filtered data and apply notches if enabled
                    psd_filtered_original = psd_data['psd_filtered_original']
                    freq_array = psd_data['freq']
                    
                    if (self.apply_notches and 
                        ch_label in self.all_channel_notches and 
                        len(self.all_channel_notches[ch_label].get('notch_freqs', [])) > 0):
                        
                        # Apply notches
                        psd_data['psd_filtered'] = self._apply_notches_to_psd(
                            psd_filtered_original, freq_array, ch_label
                        )
                    else:
                        # Use original without notches
                        psd_data['psd_filtered'] = psd_filtered_original.copy()
                        
        except Exception as e:
            print(f"Error refreshing current PSD with notches: {e}")
        
    def update_psd_from_shared_buffer(self):
        """Update PSD plots using data from shared buffer"""
        try:
            if not self.shared_psd_buffer or not self.auto_refresh_enabled:
                return
            
            # If we have stored data and user has selected a specific channel from dropdown,
            # prefer showing the stored data unless auto-refresh is specifically for live monitoring
            current_dropdown_selection = self.cmb_channel.currentText()
            if (current_dropdown_selection and 
                current_dropdown_selection in self.all_channel_psd_data and 
                current_dropdown_selection != self.current_channel_name):
                # User has selected a different channel from dropdown, show stored data
                return
                
            # Get PSD data for the currently selected channel
            freq_data, psd_data = self.shared_psd_buffer.get_psd_data(self.selected_channel_idx)
            
            if freq_data is not None and psd_data is not None:
                # Convert to dB
                psd_db = 10 * np.log10(psd_data + 1e-12)  # Add small value to avoid log(0)
                
                # Calculate and update noise level
                noise_level_db, category = self.calculate_noise_level(psd_db, freq_data)
                self.update_noise_status(self.current_channel_name, noise_level_db, category)
                
                # Update plots - use setData with empty arrays instead of clear() to preserve legend
                if self.show_raw_psd:
                    self.psd_curve_1.setData(x=freq_data, y=psd_db)
                else:
                    self.psd_curve_1.setData(x=[], y=[])
                
                # Only show raw data for auto-refresh, no filtered curve
                self.psd_curve_2.setData(x=[], y=[])
                
                self.psd_graph_1.setXRange(0, self.psd_1_x)
                self.psd_graph_1.setYRange(-50, 50)
                self.psd_graph_1.setTitle(f'Power Spectral Density {self.current_channel_name} (0-{self.psd_1_x} Hz) - Live Auto Refresh', color='k', size='10pt')
                
                # Filter for 0-300 Hz range
                idx = np.where(freq_data <= self.psd_2_x)
                if len(idx[0]) > 0:
                    freq_filtered = freq_data[idx]
                    psd_filtered = psd_db[idx]
                    
                    if self.show_raw_psd:
                        self.psd_curve_300_1.setData(x=freq_filtered, y=psd_filtered)
                    else:
                        self.psd_curve_300_1.setData(x=[], y=[])
                    
                    self.psd_curve_300_2.setData(x=[], y=[])
                    
                    self.psd_graph_2.setXRange(0, self.psd_2_x)
                    self.psd_graph_2.setYRange(-30, 50)
                    self.psd_graph_2.setTitle(f'Power Spectral Density {self.current_channel_name} (0-{self.psd_2_x} Hz) - Live Auto Refresh', color='k', size='10pt')
                    
                    # Calculate notches for the filtered data
                    freq_notch_range = np.where(freq_filtered >= self.freq_min)
                    if len(freq_notch_range[0]) > 0:
                        self.calculate_notches(psd_filtered[freq_notch_range], freq_filtered[freq_notch_range])
                        
        except Exception as e:
            print(f"Error updating PSD from shared buffer: {e}")
    
    def set_selected_channel(self, channel_idx, channel_name=""):
        """Set the currently selected channel for auto-refresh"""
        self.selected_channel_idx = channel_idx
        self.current_channel_name = channel_name
        
        # Update shared buffer metadata
        if self.shared_psd_buffer:
            self.shared_psd_buffer.set_auto_refresh(
                self.auto_refresh_enabled, 
                channel_idx, 
                channel_name
            )
    
    def calculate_noise_level(self, psd_db, freq_data):
        """
        Calculate noise level from PSD data in a specific frequency range.
        
        Args:
            psd_db: Power spectral density in dB
            freq_data: Frequency array
            
        Returns:
            tuple: (noise_level_db, category) where category is 'Low', 'Medium', or 'High'
        """
        try:
            # Calculate noise level in the 1-100 Hz range (typical noise band for neural signals)
            idx = np.where((freq_data >= 1) & (freq_data <= 100))
            if len(idx[0]) == 0:
                return None, "Unknown"
            
            # Use median to avoid outliers from line noise
            noise_level_db = np.median(psd_db[idx])
            
            # Categorize based on thresholds (these can be adjusted based on your data)
            # Typical neural signal noise levels in dB/Hz:
            # Low: < -10 dB, Medium: -10 to 5 dB, High: > 5 dB
            if noise_level_db < -10:
                category = "Low"
            elif noise_level_db < 5:
                category = "Medium"
            else:
                category = "High"
            
            return noise_level_db, category
            
        except Exception as e:
            print(f"Error calculating noise level: {e}")
            return None, "Unknown"
    
    def update_noise_status(self, channel_name, noise_level_db, category):
        """Update the status label with noise information and notch effectiveness."""
        if noise_level_db is not None:
            status_text = f'Channel: {channel_name} | Noise Level: {noise_level_db:.1f} dB ({category})'
            
            # Add notch effectiveness if available
            if channel_name in self.all_channel_notches:
                notch_count = len(self.all_channel_notches[channel_name].get('notch_freqs', []))
                if notch_count > 0:
                    status_text += f' | {notch_count} active notches'
            
            self.status_label.setText(status_text)
        else:
            self.status_label.setText(f'Channel: {channel_name} | Noise Level: N/A')

    def update_hg_status(self, channel_name, hg_level_db):
        """Update the status label with high gamma activity information."""
        if hg_level_db is not None:
            current_text = self.status_label.text()
            hg_level_cat = "Low"
            if hg_level_db > -10:
                hg_level_cat = "Medium"
            elif hg_level_db > 5:
                hg_level_cat = "High"
            hg_text = f' | High Gamma Level (70-200 Hz): {hg_level_db:.1f} dB ({hg_level_cat})'
            if hg_text not in current_text:
                self.status_label.setText(current_text + hg_text)

    def plot_power_spectrum(self, ch_label, data, fs, filtered_data=None):
        """
        Plots the power spectrum of a selected channel. 
        There are two plots: one with most of the frequency range(10kHz) and one with 0 to 300 Hz.
        """
        
        if fs != self.fs:
            self.fs = fs
            self.init_fft_params()

        if filtered_data is None:
            filtered_data = self.filtering.custom_bp_filter(data=data, fs=fs, channel=ch_label)

        f1, Pxx_den_1 = signal.welch(x=data, fs=fs, window=self.barthannwin, nfft=self.nfft)
        f2, Pxx_den_2 = signal.welch(x=filtered_data, fs=fs, window=self.barthannwin, nfft=self.nfft)

        pxx_db = 10*np.log10(Pxx_den_1) # Converted to dB/Hz from Watts/Hz
        pxx_db_filtered = 10*np.log10(Pxx_den_2)
        
        # Calculate and update noise level
        noise_level_db, category = self.calculate_noise_level(pxx_db, f1)
        self.update_noise_status(ch_label, noise_level_db, category)

        if self.fs == 2000:
            # Get high gamma activity levels between 70-200 Hz
            idx_hg = np.where((f1 >= 70) & (f1 <= 200))
            if len(idx_hg[0]) > 0:
                hg_level = np.median(pxx_db_filtered[idx_hg])
                self.update_hg_status(ch_label, hg_level)
        
        idx = np.where((f1 <= self.psd_1_x))
        f1_10k = f1[idx]
        f2_10k = f2[idx]
        pxx_db_10k = pxx_db[idx]
        pxx_db_filtered_10k = pxx_db_filtered[idx]

        if self.show_raw_psd:
            self.psd_curve_1.setData(x=f1_10k, y=pxx_db_10k)
        else:
            self.psd_curve_1.clear()
        
        if self.show_filtered_psd:
            self.psd_curve_2.setData(x=f2_10k, y=pxx_db_filtered_10k)
        else:
            self.psd_curve_2.clear()
        
        self.psd_graph_1.setXRange(0, self.psd_1_x)
        self.psd_graph_1.setYRange(-50, 50)
        self.psd_graph_1.setTitle(f'Power Spectral Density {ch_label} (0-{self.psd_1_x} kHz)', color='k', size='10pt')        

        # Get index of 300 - 3000 Hz
        idx = np.where((f1 >= self.freq_min) & (f1 <= self.freq_max))
        idx_300 = np.where((f1 <= self.psd_2_x))
        f1_300_3k = f1[idx]
        f2_300_3k = f2[idx]
        pxx_db_300_3k = pxx_db[idx]
        pxx_db_filtered_300_3k = pxx_db_filtered[idx]

        idx_300 = np.where((f1 <= self.psd_2_x))
        f1_300 = f1[idx_300]
        f2_300 = f2[idx_300]
        pxx_db_300 = pxx_db[idx_300]
        pxx_db_filtered_300 = pxx_db_filtered[idx_300]

        if self.show_raw_psd:
            self.psd_curve_300_1.setData(x=f1_300, y=pxx_db_300)
        else:
            self.psd_curve_300_1.clear()
        
        if self.show_filtered_psd:
            self.psd_curve_300_2.setData(x=f2_300, y=pxx_db_filtered_300)
        else:
            self.psd_curve_300_2.clear()
        
        self.psd_graph_2.setXRange(0, self.psd_2_x)
        self.psd_graph_2.setYRange(-30, 50)
        self.psd_graph_2.setTitle(f'Power Spectral Density {ch_label} (0-{self.psd_2_x} Hz)', color='k', size='10pt')

        self.calculate_notches(pxx_db_filtered_300_3k, f2_300_3k)

    def calculate_notches(self, pxx_db=None, f=None):
        """
        Calculates notches for every channel.
        """
        try:
            if pxx_db is None or f is None:
                pxx_db = self.psd_curve_300_2.yData
                f = self.psd_curve_300_2.xData
            # Smooth the spectrum
            # pxx_db_smooth = signal.medfilt(pxx_db, self.span_smooth)
            # pxx_db_smooth = gaussian_filter1d(pxx_db, sigma=50)
            pxx_db_smooth = ndimage.median_filter(pxx_db, size=self.span_smooth)

            # Add self.db_thr to smooth spectrum
            pxx_db_smooth += self.db_thr

            # Plot the smoothed spectrum
            
            self.psd_smooth_curve.setData(x=f, y=pxx_db_smooth)

            # Check if pxx_db is above pxx_db_smooth to detect notches
            notches = np.where(pxx_db > pxx_db_smooth, 1, 0)

            if not np.any(notches):
                print('No notches detected.')
                return
            else:
                print(f'Detected {np.sum(notches)} notches.')

            # Find amplitude of the largest notch
            notch_indices = np.where(notches == 1)[0]  # Get original indices where notches occur
            notches_amp = pxx_db[notches == 1] - pxx_db_smooth[notches == 1]
            max_notch_idx = np.argmax(notches_amp)
            max_notch_amp = notches_amp[max_notch_idx]

            print(f'Max notch: {max_notch_amp:.2f} dB at {f[notch_indices[max_notch_idx]]:.2f} Hz.')

            # Find the indices of the notches that are greater than half of the largest notch
            significant_notches = notches_amp > (max_notch_amp * 0.5)
            significant_notch_indices = notch_indices[significant_notches]

            # remove notch lines from before
            for line in self.notch_lines:
                self.psd_graph_1.removeItem(line)

            # Plot vertical dotted lines at the location of the notches
            for notch_idx in significant_notch_indices:
                line = self.psd_graph_1.addLine(x=f[notch_idx], pen=pg.mkPen('k', width=1, style=pg.QtCore.Qt.DashLine))
                self.notch_lines.append(line)
        except:
            print(traceback.print_exc())

    def start_acquisition(self):
        """
        Start data acquisition for power spectrum calculation.
        """
        self.btn_acquire.setText('Acquiring Data...')
        self.btn_acquire.setEnabled(False)
        
        # Send message to start PSD acquisition
        self.c.start_all_ch_psd_acq.emit()
    
    def on_channel_selected(self, channel_idx=None):
        """
        Handle channel selection from dropdown.
        Display running average PSD and notches data for the selected channel.
        """
        try:
            if channel_idx is None:
                channel_name = self.cmb_channel.currentText()
            else:
                sender = self.psd_widget.sender()
                channel_name = sender.currentText()
                
            if not channel_name:
                return
                
            # Check if we have running average data for this channel
            if channel_name in self.running_average_psd:
                print(f"Displaying running average PSD data for channel: {channel_name}")
                self._display_running_average_psd(channel_name)
            else:
                print(f"No running average PSD data available for channel: {channel_name}")
                return
            
            # Display stored notches for this channel if available
            self.display_stored_notches(channel_name)
            
            # Update selected channel for monitoring
            self.set_selected_channel_for_monitoring(channel_name)
            
        except Exception as e:
            print(f"Error displaying channel data: {e}")
            print(traceback.format_exc())
    
    def display_stored_notches(self, channel_name):
        """
        Display the stored notches for a specific channel with enhanced visualization.
        """
        try:
            # Remove existing notch lines
            for line in self.notch_lines:
                self.psd_graph_1.removeItem(line)
            self.notch_lines.clear()
            
            # Add stored notch lines if they exist
            if channel_name in self.all_channel_notches:
                notch_data = self.all_channel_notches[channel_name]
                notch_freqs = notch_data.get('notch_freqs', [])
                notch_details = notch_data.get('notch_details', [])
                
                # Plot vertical lines at notch locations with enhanced visualization
                for i, freq in enumerate(notch_freqs):
                    # Use different line styles based on amplitude if we have details
                    if i < len(notch_details):
                        detail = notch_details[i]
                        amplitude_ratio = detail.get('amplitude_ratio', 0)
                        bandwidth = detail.get('bandwidth', 1)
                        
                        # Color and style based on amplitude (effectiveness)
                        if amplitude_ratio > 0.8:
                            pen = pg.mkPen('red', width=3, style=pg.QtCore.Qt.DashLine)
                            tooltip = f"Critical notch: {freq:.1f}Hz (BW:{bandwidth:.1f}Hz)"
                        elif amplitude_ratio > 0.5:
                            pen = pg.mkPen('orange', width=2, style=pg.QtCore.Qt.DashLine)
                            tooltip = f"Strong notch: {freq:.1f}Hz (BW:{bandwidth:.1f}Hz)"
                        elif amplitude_ratio > 0.3:
                            pen = pg.mkPen('yellow', width=2, style=pg.QtCore.Qt.DotLine)
                            tooltip = f"Moderate notch: {freq:.1f}Hz (BW:{bandwidth:.1f}Hz)"
                        else:
                            pen = pg.mkPen('gray', width=1, style=pg.QtCore.Qt.DotLine)
                            tooltip = f"Weak notch: {freq:.1f}Hz (BW:{bandwidth:.1f}Hz)"
                        
                        # Add bandwidth visualization as a shaded region
                        if bandwidth > 1:
                            freq_min = max(0, freq - bandwidth/2)
                            freq_max = freq + bandwidth/2
                            # Create a fill region to show the notch bandwidth
                            fill = pg.FillBetweenItem(
                                pg.PlotCurveItem([freq_min, freq_max], [-50, -50]),
                                pg.PlotCurveItem([freq_min, freq_max], [50, 50]),
                                brush=pg.mkBrush(color=pen.color(), alpha=30)
                            )
                            self.psd_graph_1.addItem(fill)
                            self.notch_lines.append(fill)
                    else:
                        pen = pg.mkPen('black', width=1, style=pg.QtCore.Qt.DashLine)
                        tooltip = f"Notch: {freq:.1f}Hz"
                    
                    line = self.psd_graph_1.addLine(x=freq, pen=pen)
                    line.setToolTip(tooltip)  # Add hover tooltip
                    self.notch_lines.append(line)
                
                print(f"Displayed {len(notch_freqs)} notches for channel {channel_name}")
                if notch_details:
                    print("Notch effectiveness (color-coded):")
                    for detail in notch_details:
                        effectiveness = "Critical" if detail['amplitude_ratio'] > 0.8 else \
                                      "Strong" if detail['amplitude_ratio'] > 0.5 else \
                                      "Moderate" if detail['amplitude_ratio'] > 0.3 else "Weak"
                        print(f"  {detail['freq']:.1f} Hz: {effectiveness} (BW={detail['bandwidth']:.1f}Hz, "
                              f"Reduction={detail['amplitude']:.1f}dB)")
            
            # Update the smooth curve if we have frequency range data
            if channel_name in self.all_channel_psd_data:
                psd_data = self.all_channel_psd_data[channel_name]
                freq_data = psd_data['freq']
                psd_filtered = psd_data['psd_filtered']
                
                # Filter to frequency range of interest for notch detection
                idx = np.where((freq_data >= self.freq_min) & (freq_data <= self.freq_max))
                if len(idx[0]) > 0:
                    f_filtered = freq_data[idx]
                    psd_filtered_range = psd_filtered[idx]
                    
                    # Calculate smooth curve for display
                    psd_smooth = gaussian_filter1d(psd_filtered_range, sigma=50) + self.db_thr
                    self.psd_smooth_curve.setData(x=f_filtered, y=psd_smooth)
                    
        except Exception as e:
            print(f"Error displaying stored notches: {e}")

    def set_selected_channel_for_monitoring(self, channel_name, channel_idx=None):
        """
        Set the selected channel for PSD monitoring and auto-refresh.
        This should be called when user selects a channel in the signal viewer.
        """
        try:
            if channel_idx is None:
                # Try to determine channel index from name
                # This would need to be coordinated with the acquisition system
                # For now, use a simple heuristic
                if channel_name.startswith('ch_'):
                    try:
                        channel_idx = int(channel_name.split('_')[1])
                    except:
                        channel_idx = 0
                else:
                    channel_idx = 0
            
            self.set_selected_channel(channel_idx, channel_name)
            print(f"Set PSD monitoring channel to: {channel_name} (index: {channel_idx})")
            
        except Exception as e:
            print(f"Error setting selected channel: {e}")
    
    def clear_stored_data(self):
        """Clear all running average PSD, current PSD, and notches data"""
        self.running_average_psd.clear()
        self.current_psd.clear()
        self.all_channel_notches.clear()
        self.psd_calculation_count = 0
        
        # Clear channel selector
        self.cmb_channel.blockSignals(True)
        self.cmb_channel.clear()
        self.cmb_channel.blockSignals(False)
        
        # Clear plots
        self.psd_curve_1.clear()
        self.psd_curve_2.clear()
        self.psd_curve_300_1.clear()
        self.psd_curve_300_2.clear()
        self.psd_smooth_curve.clear()
        
        # Remove notch lines
        for line in self.notch_lines:
            self.psd_graph_1.removeItem(line)
        self.notch_lines.clear()
        
        # Update status and buttons
        self.update_psd_progress('all', 0, self.k_bartlett_target, self.nfft)
        
        print("Cleared all running average PSD and notches data")

    def clear_mne_psd_plot(self, message="Load a .fif file to display MNE-style PSD plots."):
        """Clear the embedded MNE-style PSD matplotlib figure."""
        try:
            if self._mne_psd_toolbar is not None:
                self._mne_psd_toolbar.setParent(None)
                self._mne_psd_toolbar.deleteLater()
                self._mne_psd_toolbar = None
            if self._mne_psd_canvas is not None:
                fig = getattr(self._mne_psd_canvas, "figure", None)
                self._mne_psd_canvas.setParent(None)
                self._mne_psd_canvas.deleteLater()
                self._mne_psd_canvas = None
                if fig is not None:
                    try:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    except Exception:
                        pass
            if self.lbl_mne_psd_placeholder is not None:
                self.lbl_mne_psd_placeholder.setText(str(message))
                self.lbl_mne_psd_placeholder.setVisible(True)
        except Exception as e:
            print(f"Error clearing MNE PSD plot: {e}")

    def plot_mne_spectrum(self, spectrum, source_label="FIF"):
        """Render tutorial-style MNE PSD plot into the PSD tab.

        Parameters
        ----------
        spectrum : mne.time_frequency.Spectrum
            Precomputed spectrum object returned by `raw.compute_psd(...)`.
        source_label : str
            Optional source text shown in the figure title.
        """
        try:
            if spectrum is None:
                self.clear_mne_psd_plot(message="No MNE Spectrum available for plotting.")
                return False

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

            try:
                data, freqs = spectrum.get_data(return_freqs=True)
            except TypeError:
                data = spectrum.get_data()
                freqs = getattr(spectrum, "freqs", None)
            data = np.asarray(data, dtype=float)
            freqs = np.asarray(freqs if freqs is not None else [], dtype=float)
            if data.ndim == 3:
                data = np.nanmean(data, axis=0)
            if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
                self.clear_mne_psd_plot(message="MNE Spectrum did not contain 2D PSD data.")
                return False
            if freqs.ndim != 1 or freqs.size != data.shape[1]:
                freqs = np.arange(data.shape[1], dtype=float)

            from matplotlib.figure import Figure

            self.clear_mne_psd_plot(message="")
            fig = Figure(figsize=(8.5, 4.8), tight_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            psd_db = 10.0 * np.log10(np.maximum(data, np.finfo(float).tiny))
            finite = np.isfinite(psd_db)
            max_lines = min(psd_db.shape[0], 80)
            line_indices = np.linspace(0, psd_db.shape[0] - 1, max_lines, dtype=int)
            for idx in line_indices:
                y = psd_db[int(idx)]
                if np.any(np.isfinite(y)):
                    ax.plot(freqs, y, color="#94a3b8", linewidth=0.65, alpha=0.34)
            mean_psd = np.nanmean(np.where(finite, psd_db, np.nan), axis=0)
            ax.plot(freqs, mean_psd, color="#1d4ed8", linewidth=1.8, label="channel mean")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power spectral density (dB/Hz)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

            try:
                ax.set_title(f"MNE PSD ({source_label})", fontsize=10)
            except Exception:
                pass

            self._mne_psd_canvas = FigureCanvas(fig)
            self._mne_psd_toolbar = NavigationToolbar(self._mne_psd_canvas, self.psd_tab_mne)
            self.psd_tab_mne_layout.addWidget(self._mne_psd_toolbar)
            self.psd_tab_mne_layout.addWidget(self._mne_psd_canvas)
            self.lbl_mne_psd_placeholder.setVisible(False)
            self._mne_psd_canvas.draw_idle()
            return True
        except Exception as e:
            self.clear_mne_psd_plot(
                message=f"Could not render MNE-style PSD plot: {e}"
            )
            print(f"Error plotting MNE spectrum: {e}")
            traceback.print_exc()
            return False

    def load_precomputed_psd_data(self, psd_by_channel, fs=None, source_label="FIF", clear_notches=True):
        """Load externally computed PSD data into the PSD tab.

        Parameters
        ----------
        psd_by_channel : dict
            Mapping `{channel_name: {'freq': ..., 'psd_raw': ..., 'psd_filtered': ...}}`.
            Arrays are expected in dB/Hz.
        fs : float | None
            Sampling rate. When provided, updates internal FFT params.
        source_label : str
            Text shown in status/title to indicate PSD source.
        clear_notches : bool
            Whether to clear previously stored notch overlays from unrelated data.
        """
        try:
            if fs is not None and np.isfinite(float(fs)) and float(fs) > 0:
                fs_val = int(round(float(fs)))
                if fs_val != int(self.fs):
                    self.fs = fs_val
                    self.init_fft_params()

            if not isinstance(psd_by_channel, dict) or not psd_by_channel:
                self.status_label.setText(f"No precomputed PSD data to load ({source_label})")
                return False

            self.running_average_psd.clear()
            self.current_psd.clear()
            if clear_notches:
                self.all_channel_notches.clear()
                for line in self.notch_lines:
                    try:
                        self.psd_graph_1.removeItem(line)
                    except Exception:
                        pass
                self.notch_lines.clear()

            inserted = 0
            for ch_label, payload in psd_by_channel.items():
                if not isinstance(payload, dict):
                    continue
                freq = np.asarray(payload.get('freq', []), dtype=float).ravel()
                psd_raw = np.asarray(payload.get('psd_raw', []), dtype=float).ravel()
                psd_filtered = np.asarray(payload.get('psd_filtered', []), dtype=float).ravel()
                if freq.size == 0 or psd_raw.size == 0 or psd_filtered.size == 0:
                    continue
                if not (freq.size == psd_raw.size == psd_filtered.size):
                    continue
                if not np.any(np.isfinite(freq)):
                    continue
                if not np.any(np.isfinite(psd_raw)) or not np.any(np.isfinite(psd_filtered)):
                    continue

                ch_name = str(ch_label)
                self.current_psd[ch_name] = {
                    'freq': freq.copy(),
                    'psd_raw': psd_raw.copy(),
                    'psd_filtered': psd_filtered.copy(),
                    'psd_filtered_original': psd_filtered.copy(),
                }
                self.running_average_psd[ch_name] = {
                    'freq': freq.copy(),
                    'psd_raw_avg': psd_raw.copy(),
                    'psd_filtered_avg': psd_filtered.copy(),
                    'count': 1,
                }
                inserted += 1

            if inserted == 0:
                self.status_label.setText(f"No valid precomputed PSD arrays ({source_label})")
                return False

            self.psd_calculation_count = 1
            self._refresh_current_psd_with_notches()
            self._update_channel_selector()

            if self.cmb_channel.count() > 0:
                first_channel = self.cmb_channel.itemText(0)
                self.cmb_channel.setCurrentText(first_channel)
                self._display_running_average_psd(first_channel)

            self.btn_clear_data.setEnabled(True)
            self.update_psd_progress('all', 0, self.k_bartlett_target, self.nfft)
            self.status_label.setText(
                f"Loaded {inserted} precomputed PSD channels from {source_label}"
            )
            return True
        except Exception as e:
            print(f"Error loading precomputed PSD data: {e}")
            traceback.print_exc()
            return False
    
    def _refresh_current_plot(self):
        """Refresh the currently displayed plot based on visibility flags."""
        try:
            channel_name = self.cmb_channel.currentText()
            if channel_name and channel_name in self.running_average_psd:
                self._display_running_average_psd(channel_name)
        except Exception as e:
            print(f"Error refreshing plot: {e}")
    
    def get_stored_channel_names(self):
        """Get list of channel names with running average PSD data"""
        return list(self.running_average_psd.keys())
    
    def has_stored_data(self):
        """Check if there is any running average PSD data"""
        return len(self.running_average_psd) > 0
    
    def save_all_psd_data(self, filename=None):
        """
        Save all running average PSD and notches data to a file
        
        Args:
            filename: Optional filename, if None will use default timestamp-based name
        """
        try:
            if not self.has_stored_data():
                print("No running average PSD data to save")
                return
                
            if filename is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"psd_running_avg_{timestamp}.npy"
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'psd_data')
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, filename)
            
            # Combine all data into one structure
            save_data = {
                'running_average_psd': self.running_average_psd,
                'notches_data': self.all_channel_notches,
                'psd_calculation_count': self.psd_calculation_count,
                'parameters': {
                    'fs': self.fs,
                    'freq_min': self.freq_min,
                    'freq_max': self.freq_max,
                    'db_thr': self.db_thr,
                    'psd_1_x': self.psd_1_x,
                    'psd_2_x': self.psd_2_x,
                    'k_bartlett_target': self.k_bartlett_target
                }
            }
            
            np.save(filepath, save_data)
            print(f"Saved running average PSD data for {len(self.running_average_psd)} channels to: {filepath}")
            
        except Exception as e:
            print(f"Error saving running average PSD data: {e}")
    
    def load_psd_data(self, filepath):
        """
        Load previously saved running average PSD and notches data
        
        Args:
            filepath: Path to the saved PSD data file
        """
        try:
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                return False
                
            data = np.load(filepath, allow_pickle=True).item()
            
            self.running_average_psd = data.get('running_average_psd', {})
            self.all_channel_notches = data.get('notches_data', {})
            self.psd_calculation_count = data.get('psd_calculation_count', 0)
            
            # Update parameters if available
            params = data.get('parameters', {})
            if params:
                for param_name, param_value in params.items():
                    if hasattr(self, param_name):
                        setattr(self, param_name, param_value)
            
            # Update channel selector
            self.cmb_channel.blockSignals(True)
            self.cmb_channel.clear()
            if self.running_average_psd:
                self.cmb_channel.addItems(sorted(self.running_average_psd.keys()))
                # Display the first channel
                first_channel = self.cmb_channel.itemText(0)
                self.cmb_channel.setCurrentText(first_channel)
                self.on_channel_selected()
            self.cmb_channel.blockSignals(False)
            
            # Update UI
            self.update_psd_progress('all', 0, self.k_bartlett_target, self.nfft)
            
            print(f"Loaded running average PSD data for {len(self.running_average_psd)} channels")
            return True
            
        except Exception as e:
            print(f"Error loading running average PSD data: {e}")
            return False

    def start_all_channel_notch_calculation(self):
        """
        Start the process of calculating notches using running averaged PSDs.
        Only available after k_bartlett_target PSD calculations.
        """
        try:            
            # Use the running averaged PSDs for notch calculation
            self.calculate_all_channel_notches()
            
        except Exception as e:
            print(f"Error in notch calculation: {e}")
            self._reset_calculate_button()
    
    def _start_multiprocessing_notch_calculation(self, psd_data_dict):
        """
        Start multiprocessing-based notch calculation using averaged PSD data.
        
        Args:
            psd_data_dict: Dictionary with channel names as keys and PSD arrays as values
        """
        try:
            # Disable the button and show progress
            self.btn_calculate_notches.setText('Calculating...')
            self.btn_calculate_notches.setEnabled(False)
            
            # Create progress dialog
            self.progress_dialog = QtWidgets.QProgressDialog(
                f"Calculating notches for {len(psd_data_dict)} channels...", 
                "Cancel", 
                0, 
                len(psd_data_dict), 
                self.psd_widget
            )
            self.progress_dialog.setWindowTitle("Notch Calculation")
            self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            self.progress_dialog.show()
            
            # Process each channel for notch detection
            all_notches = {}
            notch_filters = []
            
            for i, (ch_label, psd_filtered_avg) in enumerate(psd_data_dict.items()):
                if self.progress_dialog.wasCanceled():
                    break
                    
                # Update progress
                self.progress_dialog.setValue(i)
                self.progress_dialog.setLabelText(f"Processing: {ch_label} ({i+1}/{len(psd_data_dict)})")
                QtWidgets.QApplication.processEvents()
                
                # Get frequency array from running average data
                if ch_label in self.running_average_psd:
                    freq_data = self.running_average_psd[ch_label]['freq']
                    
                    # Filter to notch detection frequency range
                    idx_notch = np.where((freq_data >= self.freq_min) & (freq_data <= self.freq_max))
                    if len(idx_notch[0]) > 0:
                        f_notch = freq_data[idx_notch]
                        psd_notch = psd_filtered_avg[idx_notch]
                        
                        # Calculate notches for this channel
                        detected_notches = self._calculate_notches_for_channel(psd_notch, f_notch, ch_label)
                        
                        if detected_notches:
                            all_notches[ch_label] = detected_notches
                            
                            # Create notch filter
                            filter_info = self._create_notch_filter(detected_notches, self.fs, ch_label)
                            if filter_info:
                                notch_filters.append(filter_info)
                        
                        # Store notch data for display
                        notch_freqs = [notch['freq'] for notch in detected_notches]
                        self.all_channel_notches[ch_label] = {
                            'notch_freqs': notch_freqs,
                            'notch_details': detected_notches,
                            'notch_lines': []
                        }
            
            # Update UI after calculation
            if not self.progress_dialog.wasCanceled():
                # Update the display if a channel is selected
                current_channel = self.cmb_channel.currentText()
                if current_channel and current_channel in self.all_channel_notches:
                    self.display_stored_notches(current_channel)
                
                # Save results
                self._save_notches_to_file(all_notches, notch_filters, self.fs)
                
                print(f"Notch calculation complete for {len(notch_filters)} channels with notches")
            
        except Exception as e:
            print(f"Error in notch calculation: {e}")
            print(traceback.format_exc())
        finally:
            self._reset_calculate_button()
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
    
    def _update_ui_after_calculation(self, psd_data, notches_data, notch_filters):
        """Update UI elements after calculation completion"""
        try:
            # Update channel combobox
            self.cmb_channel.blockSignals(True)
            self.cmb_channel.clear()
            if psd_data:
                self.cmb_channel.addItems(sorted(psd_data.keys()))
                # Display the first channel
                first_channel = self.cmb_channel.itemText(0)
                self.cmb_channel.setCurrentText(first_channel)
                # Manually trigger channel selection since signals are blocked
                self.cmb_channel.blockSignals(False)
                self.on_channel_selected()
            else:
                self.cmb_channel.blockSignals(False)
                
        except Exception as e:
            print(f"Error updating UI: {e}") 
    
    def _reset_calculate_button(self):
        """Reset the calculate button to its original state."""
        self.btn_calculate_notches.setText('Calculate Notches')
        self.btn_calculate_notches.setEnabled(self.psd_calculation_count >= self.k_bartlett_target)
    
    def _update_channel_selector(self):
        """Update the channel selector dropdown with available channels."""
        try:
            # Block signals to prevent triggering selection events
            self.cmb_channel.blockSignals(True)
            
            current_selection = self.cmb_channel.currentText()
            self.cmb_channel.clear()
            
            if self.running_average_psd:
                channel_names = sorted(self.running_average_psd.keys())
                self.cmb_channel.addItems(channel_names)
                
                # Restore previous selection if still available
                if current_selection in channel_names:
                    self.cmb_channel.setCurrentText(current_selection)
                elif len(channel_names) > 0:
                    self.cmb_channel.setCurrentIndex(0)
            
            self.cmb_channel.blockSignals(False)
            
        except Exception as e:
            print(f"Error updating channel selector: {e}")

    def _update_notch_lines_for_channel(self, channel_name):
        """Update the notch lines displayed for a specific channel."""
        try:
            # Remove existing notch lines
            for line in self.notch_lines:
                self.psd_graph_1.removeItem(line)
            self.notch_lines.clear()
            
            # Add stored notch lines if they exist
            if channel_name in self.all_channel_notches:
                notch_data = self.all_channel_notches[channel_name]
                notch_freqs = notch_data.get('notch_freqs', [])
                
                # Plot vertical dotted lines at the location of the notches
                for freq in notch_freqs:
                    line = self.psd_graph_1.addLine(x=freq, pen=pg.mkPen('k', width=1, style=pg.QtCore.Qt.DashLine))
                    self.notch_lines.append(line)
                    
        except Exception as e:
            print(f"Error updating notch lines for channel {channel_name}: {e}")
    
    def _display_running_average_psd(self, channel_name):
        """Display PSD for the specified channel - either running average or current based on show_running_average flag."""
        try:
            if channel_name not in self.running_average_psd:
                return
                
            psd_data = self.running_average_psd[channel_name]
            freq_data = psd_data['freq']
            count = psd_data['count']
            
            # Choose data based on show_running_average flag
            if self.show_running_average:
                # Show running averages
                psd_raw_display_data = psd_data['psd_raw_avg']
                psd_filtered_display_data = psd_data['psd_filtered_avg']
                title_suffix = f"{count} averages"
            else:
                # Show most recent/current PSD
                if channel_name in self.current_psd:
                    current_data = self.current_psd[channel_name]
                    psd_raw_display_data = current_data['psd_raw']
                    psd_filtered_display_data = current_data['psd_filtered']
                    
                    # Add notch status to title
                    notch_status = ""
                    if (channel_name in self.all_channel_notches and 
                        len(self.all_channel_notches[channel_name].get('notch_freqs', [])) > 0):
                        notch_count = len(self.all_channel_notches[channel_name]['notch_freqs'])
                        if self.apply_notches:
                            notch_status = f" (notches active: {notch_count})"
                        else:
                            notch_status = f" (notches available: {notch_count})"
                    
                    title_suffix = f"current{notch_status}"
                else:
                    # Fallback to averages if current data not available
                    psd_raw_display_data = psd_data['psd_raw_avg']
                    psd_filtered_display_data = psd_data['psd_filtered_avg']
                    title_suffix = f"{count} averages (no current data)"
            
            # Calculate and update noise level using raw PSD
            noise_level_db, category = self.calculate_noise_level(psd_raw_display_data, freq_data)
            self.update_noise_status(channel_name, noise_level_db, category)
            
            # Limit to display range for upper plot
            idx_display = np.where(freq_data <= self.psd_1_x)
            freq_display = freq_data[idx_display]
            psd_raw_display = psd_raw_display_data[idx_display]
            psd_filtered_display = psd_filtered_display_data[idx_display]
            
            # Update upper plot (0-10kHz)
            if self.show_raw_psd:
                self.psd_curve_1.setData(x=freq_display, y=psd_raw_display)
            else:
                self.psd_curve_1.clear()
            
            if self.show_filtered_psd:
                self.psd_curve_2.setData(x=freq_display, y=psd_filtered_display)
            else:
                self.psd_curve_2.clear()
            
            self.psd_graph_1.setXRange(0, self.psd_1_x)
            self.psd_graph_1.setYRange(-50, 50)
            self.psd_graph_1.setTitle(f'Power Spectrum {channel_name} (0-{self.psd_1_x} Hz) - {title_suffix}', color='k', size='10pt')
            
            # Filter for lower plot (0-300 Hz)
            idx_300 = np.where(freq_data <= self.psd_2_x)
            if len(idx_300[0]) > 0:
                freq_300 = freq_data[idx_300]
                psd_raw_300 = psd_raw_display_data[idx_300]
                psd_filtered_300 = psd_filtered_display_data[idx_300]
                
                if self.show_raw_psd:
                    self.psd_curve_300_1.setData(x=freq_300, y=psd_raw_300)
                else:
                    self.psd_curve_300_1.clear()
                
                if self.show_filtered_psd:
                    self.psd_curve_300_2.setData(x=freq_300, y=psd_filtered_300)
                else:
                    self.psd_curve_300_2.clear()
                
                self.psd_graph_2.setXRange(0, self.psd_2_x)
                self.psd_graph_2.setYRange(-30, 50)
                self.psd_graph_2.setTitle(f'Power Spectrum {channel_name} (0-{self.psd_2_x} Hz) - {title_suffix}', color='k', size='10pt')
            
            # Update smooth curve for notch detection visualization
            idx_notch = np.where((freq_data >= self.freq_min) & (freq_data <= self.freq_max))
            if len(idx_notch[0]) > 0:
                f_notch = freq_data[idx_notch]
                psd_notch = psd_filtered_display_data[idx_notch]
                
                # Calculate smooth curve
                psd_smooth = ndimage.median_filter(psd_notch, size=self.span_smooth) + self.db_thr
                self.psd_smooth_curve.setData(x=f_notch, y=psd_smooth)
            
        except Exception as e:
            print(f"Error displaying PSD: {e}")

    def calculate_all_channel_psd(self, ch_data, fs):
        """
        Calculate PSD for all channels using persistent multiprocessing (non-blocking).
        
        Args:
            ch_data: Dictionary with channel labels as keys and data arrays as values
            fs: Sampling frequency
        """
        if not ch_data:
            print("No channel data provided for PSD calculation")
            return
            
        # Update fs if needed
        if fs is not None and fs != self.fs:
            self.fs = fs
            self.init_fft_params()
        
        # Check if processor is running
        if not is_processor_running():
            print("PSD processor not running. Please enable acquisition first.")
            return
        
        try:
            # Update filter parameters in processor
            processor = get_processor()
            processor.update_filter_params(
                lowcut=getattr(self.filtering, 'lowcut', 300.0),
                highcut=getattr(self.filtering, 'highcut', 3000.0),
                filt_order=getattr(self.filtering, 'filt_order', 4),
                fs=self.fs,
                notch_freqs=getattr(self.filtering, 'basic_notches', [60.0, 120.0, 180.0]),
                apply_notches=getattr(self.filtering, 'apply_notches', True),
                use_matlab_notches=getattr(self.filtering, 'use_matlab_notches', False),
                span_smooth=self.span_smooth,
                db_thr=self.db_thr,
                freq_min=self.freq_min,
                freq_max=self.freq_max
            )
            
            # Submit PSD calculation task (non-blocking)
            task_id = processor.calculate_psd(ch_data)
            
            # Store task metadata for processing results
            self.pending_psd_tasks[task_id] = {
                'timestamp': time.time(),
                'channel_count': len(ch_data),
                'fs': fs
            }
            
            print(f"Submitted PSD calculation task for {len(ch_data)} channels (Task ID: {task_id})")
            
        except Exception as e:
            print(f"Error submitting PSD calculation: {e}")
            traceback.print_exc()

    def calculate_all_channel_notches(self):
        """
        Calculate notches using the running averaged PSDs.
        """
        
        # Check if we have enough PSD calculations for notch calculation
        if self.psd_calculation_count * 5 < self.k_bartlett_target:
            print(f"Not enough PSD calculations. Need {self.k_bartlett_target}, have {self.psd_calculation_count * 5}")
            return
            
        if not self.running_average_psd:
            print("No running average PSD data available for notch calculation")
            return
            
        print(f"Starting notch calculation using averaged PSDs for {len(self.running_average_psd)} channels...")
        
        # Convert running averages to the format expected by the notch calculation
        ch_data_for_notches = {}
        for ch_label, psd_data in self.running_average_psd.items():
            # Use the filtered PSD for notch detection
            ch_data_for_notches[ch_label] = psd_data['psd_filtered_avg']
        
        # Use multiprocessing for notch calculation
        self._start_multiprocessing_notch_calculation(ch_data_for_notches)

    def _calculate_notches_for_channel(self, pxx_db, frequency_array, ch_label):
        """
        Calculate notches for a single channel based on MATLAB implementation.
        Uses bandwidth calculation for aggressive notching of larger amplitude interference.
        
        Args:
            pxx_db: Power spectral density in dB
            frequency_array: Frequency array corresponding to pxx_db
            ch_label: Channel label for logging
        
        Returns:
            List of dictionaries with notch information: [{'freq': freq, 'bandwidth': bw, 'amplitude': amp}, ...]
        """
        try:
            # Create smoothed threshold line using moving median + threshold
            smoothed_threshold_db = ndimage.median_filter(pxx_db, size=self.span_smooth) + self.db_thr

            # Handle frequencies above 2950 Hz - keep threshold constant (like MATLAB)
            high_freq_indices = np.where(frequency_array >= 2950)[0]
            if len(high_freq_indices) > 0:
                smoothed_threshold_db[high_freq_indices[0]:] = smoothed_threshold_db[high_freq_indices[0]]

            detected_notches = []
            
            # Find all frequency bins where power exceeds threshold
            above_threshold_indices = np.where(pxx_db > smoothed_threshold_db)[0]
            
            if len(above_threshold_indices) == 0:
                print(f'No notches detected for channel {ch_label}')
                return []

            # Calculate maximum amplitude for normalization
            amplitude_differences = pxx_db - smoothed_threshold_db
            max_amplitude_above_threshold = np.max(amplitude_differences)
            
            # Find breaks in continuous regions (where difference between consecutive indices > 1)
            region_break_indices = np.where(np.diff(above_threshold_indices) > 1)[0]
            
            # Determine start indices and lengths of each continuous region
            if len(region_break_indices) == 0:
                # Single continuous region
                region_start_indices = [above_threshold_indices[0]]
                region_lengths = [len(above_threshold_indices)]
            else:
                # Multiple regions
                region_start_indices = [above_threshold_indices[0]]
                region_start_indices.extend([above_threshold_indices[idx + 1] for idx in region_break_indices])
                
                region_lengths = []
                region_lengths.append(region_break_indices[0] + 1)  # First region
                # Middle regions
                for i in range(1, len(region_break_indices)):
                    region_lengths.append(region_break_indices[i] - region_break_indices[i-1])
                # Last region
                last_region_start_pos = np.where(above_threshold_indices == region_start_indices[-1])[0][0]
                region_lengths.append(len(above_threshold_indices) - last_region_start_pos)

            # Process each region that has more than 1 sample (significant regions only)
            significant_region_indices = [i for i, length in enumerate(region_lengths) if length > 1]
            
            for region_idx in significant_region_indices:
                region_start = region_start_indices[region_idx]
                region_length = region_lengths[region_idx]
                
                # Find all indices belonging to this region
                region_end = region_start + region_length - 1
                current_region_indices = above_threshold_indices[
                    (above_threshold_indices >= region_start) & (above_threshold_indices <= region_end)
                ]
                
                if len(current_region_indices) == 0:
                    continue
                
                # Find the peak (maximum power) within this region
                region_power_values = pxx_db[current_region_indices]
                peak_idx_within_region = np.argmax(region_power_values)
                peak_frequency_index = current_region_indices[peak_idx_within_region]
                
                # Calculate center frequency index (geometric center of the region)
                center_frequency_index = np.mean(current_region_indices)
                
                # Handle case where center is exactly between two samples
                if center_frequency_index % 1 == 0.5:
                    floor_idx = int(np.floor(center_frequency_index))
                    ceil_idx = int(np.ceil(center_frequency_index))
                    # Choose the sample with higher power
                    if (floor_idx < len(pxx_db) and ceil_idx < len(pxx_db) and 
                        pxx_db[floor_idx] > pxx_db[ceil_idx]):
                        center_frequency_index = floor_idx
                    else:
                        center_frequency_index = ceil_idx
                else:
                    center_frequency_index = int(np.ceil(center_frequency_index))
                
                # Ensure center index is within bounds
                center_frequency_index = max(0, min(center_frequency_index, len(frequency_array) - 1))
                
                # Calculate amplitude above threshold at peak
                peak_amplitude_above_threshold = pxx_db[peak_frequency_index] - smoothed_threshold_db[peak_frequency_index]
                
                # Calculate bandwidth based on region width and amplitude
                frequency_resolution = frequency_array[1] - frequency_array[0] if len(frequency_array) > 1 else 1.0
                base_bandwidth = frequency_resolution * region_length * 2 * peak_amplitude_above_threshold / max_amplitude_above_threshold
                
                # Aggressive bandwidth scaling based on amplitude
                # Larger amplitudes get wider notches for more aggressive suppression
                amplitude_ratio = peak_amplitude_above_threshold / max_amplitude_above_threshold
                if amplitude_ratio > 0.8:  # Very strong interference
                    bandwidth_multiplier = 3.0
                elif amplitude_ratio > 0.5:  # Strong interference
                    bandwidth_multiplier = 2.0
                elif amplitude_ratio > 0.3:  # Moderate interference
                    bandwidth_multiplier = 1.5
                else:  # Weak interference
                    bandwidth_multiplier = 1.0
                
                aggressive_bandwidth = base_bandwidth * bandwidth_multiplier
                
                # Apply frequency-dependent bandwidth limits
                center_frequency = frequency_array[center_frequency_index]
                if center_frequency < 295:
                    max_bandwidth = 3.0
                else:
                    max_bandwidth = 5.0
                
                # Ensure bandwidth is within reasonable limits
                final_bandwidth = max(1.0, min(aggressive_bandwidth, max_bandwidth))
                
                # Store notch information
                notch_info = {
                    'freq': center_frequency,
                    'bandwidth': final_bandwidth,
                    'amplitude': peak_amplitude_above_threshold,
                    'peak_power_db': pxx_db[peak_frequency_index],
                    'amplitude_ratio': amplitude_ratio
                }
                detected_notches.append(notch_info)

            print(f'Channel {ch_label}: Found {len(detected_notches)} notches:')
            for notch in detected_notches:
                print(f'  {notch["freq"]:.1f} Hz, BW: {notch["bandwidth"]:.1f} Hz, '
                      f'Amp: {notch["amplitude"]:.1f} dB (ratio: {notch["amplitude_ratio"]:.2f})')
            
            return detected_notches
            
        except Exception as e:
            print(f'Error calculating notches for channel {ch_label}: {e}')
            print(traceback.format_exc())
            return []

    def _create_notch_filter(self, detected_notches, fs, ch_label):
        """
        Create notch filter coefficients for given notches with their specific bandwidths.
        
        Args:
            detected_notches: List of dictionaries with notch information
            fs: Sampling frequency
            ch_label: Channel label
        
        Returns:
            Dictionary with filter information
        """
        try:
            # Extract channel ID from label (assuming format like 'ch_001' or similar)
            ch_id_match = ch_label.split('_')[-1]
            try:
                ch_id = int(ch_id_match)
            except ValueError:
                # If we can't extract a number, use a hash of the channel label
                ch_id = hash(ch_label) % 1000
            
            # Create combined notch filter using individual bandwidths
            all_zeros = np.array([])
            all_poles = np.array([])
            combined_gain = 1.0
            
            for notch in detected_notches:
                freq = notch['freq']
                bandwidth = notch['bandwidth']
                
                if freq > 0 and freq < fs/2:  # Ensure frequency is valid
                    # Calculate Q factor from bandwidth: Q = center_freq / bandwidth
                    Q_factor = freq / bandwidth
                    
                    # Create notch filter with specific bandwidth
                    b_notch, a_notch = signal.iirnotch(w0=freq, Q=Q_factor, fs=fs)
                    zeros, poles, gain = signal.tf2zpk(b_notch, a_notch)
                    
                    all_zeros = np.append(all_zeros, zeros)
                    all_poles = np.append(all_poles, poles)
                    combined_gain *= gain
            
            if len(all_zeros) > 0:
                # Convert to second-order sections for numerical stability
                sos_filter = signal.zpk2sos(all_zeros, all_poles, combined_gain)
                
                return {
                    'chID': ch_id,
                    'ch_label': ch_label,
                    'notch_details': detected_notches,
                    'SOS': sos_filter,
                    'G': combined_gain,
                    'total_notches': len(detected_notches)
                }
            else:
                return None
                
        except Exception as e:
            print(f'Error creating notch filter for channel {ch_label}: {e}')
            return None

    def _save_notches_to_file(self, all_notches, notch_filters, fs):
        """
        Save notches and filter coefficients to files.
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'notch_filters')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save human-readable notches summary
            notches_file = os.path.join(output_dir, 'detected_notches.npy')
            np.save(notches_file, all_notches)
            print(f"Saved notches summary to: {notches_file}")
            
            # Save filter coefficients for use in filtering.py
            filters_file = os.path.join(output_dir, 'notch_filters.npy')
            np.save(filters_file, notch_filters)
            print(f"Saved notch filters to: {filters_file}")
            
            # Also save a human-readable summary
            summary_file = os.path.join(output_dir, 'notches_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Aggressive Notch Detection Summary\n")
                f.write(f"==================================\n")
                f.write(f"Sampling Frequency: {fs} Hz\n")
                f.write(f"Detection Threshold: {self.db_thr} dB\n")
                f.write(f"Frequency Range: {self.freq_min}-{self.freq_max} Hz\n")
                f.write(f"Bandwidth Scaling: Aggressive based on amplitude\n\n")
                
                for ch_label, notch_list in all_notches.items():
                    f.write(f"Channel {ch_label}: {len(notch_list)} notches detected\n")
                    for i, notch in enumerate(notch_list, 1):
                        f.write(f"  {i}. {notch['freq']:.1f} Hz, "
                               f"BW: {notch['bandwidth']:.1f} Hz, "
                               f"Amp: {notch['amplitude']:.1f} dB "
                               f"(ratio: {notch['amplitude_ratio']:.2f})\n")
                    f.write("\n")
                    
            print(f"Saved summary to: {summary_file}")
            
            # Reload notch filters in the filtering object so they're immediately available
            if hasattr(self.filtering, 'reload_notch_filters'):
                self.filtering.reload_notch_filters()
                print("Reloaded notch filters in filtering module")
            
        except Exception as e:
            print(f"Error saving notches to file: {e}")
            print(traceback.format_exc())
