"""
Signal filtering utilities
==========================

Filtering helpers for neural signals, including custom bandpass filter
construction, optional notch filter composition, and spike-detection
filters with initial-condition handling.
"""
# Author: Sunil Mathew
# Date: June 14, 2026
# Filtering class for signal processing
import os
import traceback
import threading
import numpy as np
import scipy.io
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import pyqtgraph as pg


class Filtering():
    """Encapsulates filter configuration and operations for signals.

    Notes
    -----
    - Defaults target 30 kHz sampling (`fs=30000`).
    - Notch filters can be loaded from precomputed numpy or MATLAB files.
    """
    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, 'instance'):
    #         cls.instance = super(Filtering, cls).__new__(cls)
    #     return cls.instance
    
    def __init__(self, **kwargs) -> None:
        # Bandpass filter parameters (allow optional overrides)
        self.lowcut = kwargs.pop('lowcut', kwargs.pop('low_cutoff', 300.0))
        self.highcut = kwargs.pop('highcut', kwargs.pop('high_cutoff', 3000.0))
        self.filt_order = kwargs.pop('filt_order', kwargs.pop('filter_order', 4))
        self.rp = kwargs.pop('rp', kwargs.pop('passband_ripple', 0.1))
        self.rs = kwargs.pop('rs', kwargs.pop('stopband_attenuation', 40.0))
        self.fs = kwargs.pop('fs', kwargs.pop('sampling_rate', 30000))
        self.waveform_length = kwargs.pop('waveform_length', 40)
        self.apply_notches = bool(kwargs.pop('apply_notches', True))
        self.use_matlab_notches = bool(kwargs.pop('use_matlab_notches', False))

        # Notch filter parameters
        basic_notches = kwargs.pop('basic_notches', kwargs.pop('notch_freqs', [60, 120]))
        if basic_notches is None:
            basic_notches = []
        elif isinstance(basic_notches, (int, float)):
            basic_notches = [basic_notches]
        self.basic_notches = list(basic_notches)
        self.notch_width = kwargs.pop('notch_width', kwargs.pop('notch_bandwidth', 1))
        self.span_smooth = kwargs.pop('span_smooth', 21)
        self.db_thr = kwargs.pop('db_thr', 10)

        # Drop keys we deliberately ignore to avoid noisy warnings
        kwargs.pop('filter_type', None)

        self.filt_lock = threading.Lock()
        self.task_folder = kwargs.pop('task_folder', None)
        self.base_sos = None

        # Sources for per-channel notch filters
        self.notches_info = None            # Active source (selected below)
        self.notches_info_npy = None        # Loaded from notch_filters.npy
        self.notches_info_mat = None        # Loaded from pre_processing_info.mat

        # Initialize default bandpass + baseline notches
        self.init_bp_custom_filter()

        # Load pre-calculated notches if available (from both sources)
        self.load_notch_filters()

        # Surface unused parameters to aid debugging
        if kwargs:
            unused_keys = ', '.join(sorted(kwargs.keys()))
            print(f"Filtering: unused configuration parameters ignored: {unused_keys}")

    #region Signal filtering (bandpass, notch, power spectrum)

    def init_bp_custom_filter(self):
        """Initialize composite notch + bandpass filter in SOS form.

        Returns
        -------
        bool
            True if filter coefficients were initialized.
        """
        bOK = False
        self.k = 1  # Gain
        self.z = np.array([], dtype=np.complex128)  # Zeros
        self.p = np.array([], dtype=np.complex128)  # Poles

        if self.apply_notches:
            for freq in self.basic_notches:
                try:
                    b, a = signal.iirnotch(w0=freq, Q=max(freq / self.notch_width, 1e-6), fs=self.fs)
                except ValueError:
                    continue
                z, p, k = signal.tf2zpk(b, a)
                self.z = np.append(self.z, z)
                self.p = np.append(self.p, p)
                self.k *= k

        if self.lowcut > self.highcut:
            print('Lowcut frequency is greater than highcut frequency. Check the filter parameters.')
            return bOK
        if self.lowcut > self.fs or self.highcut > self.fs:
            print(f'Cutoff frequency {self.lowcut}-{self.highcut} is greater than sampling frequency of {self.fs}. Check the filter parameters.')
            return bOK

        Wn = [self.lowcut * 2 / self.fs, self.highcut * 2 / self.fs]
        z, p, k = signal.ellip(N=self.filt_order, rp=self.rp, rs=self.rs,
                               Wn=Wn, btype='bandpass', analog=False, output='zpk')
        self.z = np.append(self.z, z)
        self.p = np.append(self.p, p)
        self.k *= k

        self.base_sos = signal.zpk2sos(z=self.z, p=self.p, k=self.k)
        self.sos = self.base_sos

        bOK = True
        return bOK
    
    def init_spk_detection_filter(self, filt_order=4, notches_info=None):
        """Initialize spike-detection bandpass, optionally with per-channel notches.

        Parameters
        ----------
        filt_order : int, default 4
            IIR elliptic filter order for the bandpass section.
        notches_info : dict or None
            When provided, prepends SOS for channel-specific notches.
        """
        self.k = 1 # Gain
        self.z = np.array([]) # Zeros
        self.p = np.array([]) # Poles
        self.rp_det = 0.1 # 0.07 Maximum ripple in passband in dB
        self.rs_det = 40.0 # 20 Minimum attenuation in stopband in dB
        
        # TBD: Compute order for a ellip bandpass filter based on the lowcut and highcut frequencies
        if self.lowcut > self.highcut:
            print('Lowcut frequency is greater than highcut frequency. Check the filter parameters.')
            return
        Wn = [self.lowcut * 2 / self.fs, self.highcut * 2 / self.fs]
        self.sos = signal.ellip(N=filt_order, rp=self.rp_det, rs=self.rs_det, 
                                              Wn=Wn, btype='bandpass', analog=False, output='sos') # Bandpass filter
        # First value of first row of self.SOS is the gain
        self.k = self.sos[0, 0]
        # Divide the first three values of first row by the gain
        self.sos[0, 0:3] /= self.k
        if notches_info is not None:
            self.k *= notches_info['G']
            self.sos = np.vstack((notches_info['SOS'], self.sos))
            self.sos[0, 0:3] = self.sos[0, 0:3] * self.k
            self.spk_b, self.spk_a, self.zi, self.n_fact = self.compute_initial_conditions(self.sos)
        else:
            self.spk_b, self.spk_a = signal.ellip(N=filt_order, rp=self.rp_det, rs=self.rs_det,
                                                    Wn=Wn, btype='bandpass', analog=False, output='ba')

    def compute_initial_conditions(self, sos):
        """Compute zi and helpers for zero-phase filtering with DC handling.

        Parameters
        ----------
        sos : ndarray
            Second-order sections for the composite filter.

        Returns
        -------
        b1, a1 : ndarray
            Per-section numerator and denominator coefficients (3 x L).
        zi : ndarray
            Initial state matrix (2 x L) for each SOS.
        n_fact : int
            Padding factor used in boundary handling.
        """
        b1, a1 = signal.sos2tf(sos)
        # Normalize b1
        max_b1 = np.max(np.abs(b1))
        # print(max_b1)
        b1 = b1 / max_b1
        # Last nonzero coefficient of b1
        b1_last_nonzero = np.max(np.nonzero(b1))
        # print(b1_last_nonzero)

        # Normalize a1
        max_a1 = np.max(np.abs(a1))
        # print(max_a1)
        a1 = a1 / max_a1
        # Last nonzero coefficient of a1
        a1_last_nonzero = np.max(np.nonzero(a1))

        # filter order n is maximum of the last nonzero coefficient of a1 and b1
        ord = max(b1_last_nonzero, a1_last_nonzero)
        # print(ord)
        n_fact = max(1, 3 * ord)
        # print(n_fact)

        a1 = sos[:, 3:6].T
        # print(b1)
        b1 = sos[:, 0:3].T
        # print(a1)

        # Compute initial conditions to remove DC offset at beginning and end of
        # filtered sequence.  Use sparse matrix to solve linear system for initial
        # conditions zi, which is the vector of states for the filter b(z)/a(z) in
        # the state-space formulation of the filter.
        L = b1.shape[1]  # Number of sections
        zi = np.zeros((2, L))  # Initialize zi with zeros

        for ii in range(L):
            rhs = (b1[1:3, ii] - b1[0, ii] * a1[1:3, ii])
            zi[:, ii] = np.linalg.solve(np.eye(2) - np.vstack((-a1[1:3, ii], [1, 0])).T, rhs)
        # for value in zi[0, :]:
        #     print(f"{value:.6f}", end=' ')
        return b1, a1, zi, n_fact
        
    def FilterX_fc(self, b, a, x, zi, direction='forward'):
        if direction == 'forward':
            y, zf = signal.lfilter(b, a, x, zi=zi)
        else:
            y, zf = signal.lfilter(b, a, x[::-1], zi=zi)
            y = y[::-1]
        return y, zf

    def ff_one_chan(self, x):
        """Zero-phase like filtering using custom initial conditions.

        Applies forward/backward lfilter with padding to reduce edge transients.
        """
        L = self.spk_b.shape[1]
        for ii in range(L):
            b = self.spk_b[:, ii]
            a = self.spk_a[:, ii]
            zi = self.zi[:, ii]
            # Calculate xt
            xt = -x[self.n_fact:0:-1] + 2 * x[0]
            
            # Perform the first filter operation
            _, zo = self.FilterX_fc(b=b, a=a, x=xt, zi=zi * xt[0])
            yc2, zo = self.FilterX_fc(b=b, a=a, x=x, zi=zo)
            
            xt = -x[-2:-self.n_fact-2:-1] + 2 * x[-1]
            yc3, zo = self.FilterX_fc(b, a, xt, zo)
            
            # Reverse the sequence
            _, zo = self.FilterX_fc(b=b, a=a, x=yc3, zi=zi * yc3[-1], direction='reverse')
            x = self.FilterX_fc(b=b, a=a, x=yc2, zi=zo, direction='reverse')[0]
        
        return x

    def bandpass_filter(self, data):
        """Simple Butterworth bandpass filter (reference implementation)."""
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data
    
    def notch_filter(self, data):
        """Apply composite notch filter defined by `self.sos`."""
        if not self.apply_notches:
            return data
        filtered_data = signal.sosfilt(self.sos, data)
        return filtered_data
    
    def custom_bp_filter(self, data, fs=30000, channel=None):
        """Apply preconfigured notch+bandpass filter with optional custom notches.

        Parameters
        ----------
        data : ndarray
            Input signal.
        fs : float, default 30000
            Sampling rate (Hz); updates internal `fs` for coefficient calc.
        channel : Any, optional
            Channel identifier (label, numeric ID, or object with `label`/`id`).
        """
        self.fs = fs
        bOK = self.init_bp_custom_filter()
        if not bOK or self.base_sos is None:
            return None

        sos = self.base_sos
        if self.apply_notches:
            channel_sos = self._get_channel_notch_sos(channel)
            if channel_sos is not None:
                sos = np.vstack((channel_sos, sos))

        filtered_data = signal.sosfiltfilt(sos, data)
        return filtered_data

    def set_apply_notches(self, enabled: bool):
        """Enable or disable applying notch sections during filtering."""
        with self.filt_lock:
            self.apply_notches = bool(enabled)
            self.init_bp_custom_filter()

    def are_notches_enabled(self) -> bool:
        """Return True when notch filters are currently applied."""
        return bool(self.apply_notches)

    def _get_channel_notch_sos(self, channel):
        """Return SOS coefficients for a specific channel if available from active source."""
        info = self._get_channel_notch_info(channel)
        if info is None:
            return None
        return self._coerce_sos(info.get('SOS'))

    def _get_channel_notch_info(self, channel):
        """Return notch info dict for a specific channel from the active source."""
        if channel is None:
            return None
        active = self._get_active_notches_list()
        if not active:
            return None

        ch_label, ch_id = self._normalize_channel_identifier(channel)

        for entry in active:
            info = entry
            if isinstance(info, (list, tuple)) and len(info) == 1 and isinstance(info[0], dict):
                info = info[0]
            if not isinstance(info, dict) and hasattr(info, 'item'):
                try:
                    candidate = info.item()
                    if isinstance(candidate, dict):
                        info = candidate
                except Exception:
                    info = None
            if not isinstance(info, dict):
                continue

            if ch_label and info.get('ch_label') == ch_label:
                return info
            if ch_id is not None and info.get('chID') == ch_id:
                return info

        return None

    def _get_active_notches_list(self):
        """Return the currently active list of per-channel notches based on toggle and availability."""
        if self.use_matlab_notches and self.notches_info_mat:
            return self.notches_info_mat
        if self.notches_info_npy:
            return self.notches_info_npy
        # As last fallback if legacy code assigned directly
        return self.notches_info

    def _normalize_channel_identifier(self, channel):
        """Extract channel label and numeric ID from diverse inputs."""
        ch_label = None
        ch_id = None

        if isinstance(channel, dict):
            ch_label = channel.get('label') or channel.get('ch_label')
            ch_id = channel.get('id') or channel.get('chID')
        elif isinstance(channel, (tuple, list)):
            if len(channel) > 0 and isinstance(channel[0], str):
                ch_label = channel[0]
            if len(channel) > 1:
                try:
                    ch_id = int(channel[1])
                except (TypeError, ValueError):
                    ch_id = None
        elif isinstance(channel, str):
            ch_label = channel
            ch_id = self._parse_channel_id_from_string(channel)
        elif isinstance(channel, (int, np.integer)):
            ch_id = int(channel)
        else:
            if hasattr(channel, 'label'):
                ch_label = getattr(channel, 'label')
            elif hasattr(channel, 'name'):
                ch_label = getattr(channel, 'name')
            if hasattr(channel, 'id'):
                ch_id = getattr(channel, 'id')
            elif hasattr(channel, 'ID'):
                ch_id = getattr(channel, 'ID')

        if ch_id is None and ch_label:
            parsed = self._parse_channel_id_from_string(ch_label)
            if parsed is not None:
                ch_id = parsed

        return ch_label, ch_id

    def _parse_channel_id_from_string(self, text):
        if not isinstance(text, str):
            return None
        tail = text.split('_')[-1]
        digits = ''.join(ch for ch in tail if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    def _coerce_sos(self, sos):
        if sos is None:
            return None
        sos_array = np.asarray(sos, dtype=float)
        if sos_array.size == 0:
            return None
        if sos_array.ndim == 1:
            sos_array = sos_array.reshape(1, -1)
        if sos_array.shape[1] != 6:
            return None
        return sos_array
    
    def spk_detection_filter(self, data, channel, filt_order=4):
        """Filter a single channel for spike detection with optional notches.

        Parameters
        ----------
        data : ndarray
            1D signal.
        channel : str
            Channel label used to look up notch coefficients if available.
        filt_order : int, default 4
            Bandpass order.
        """
        try:
            self.filt_lock.acquire()
            channel_info = None
            if self.apply_notches:
                channel_info = self._get_channel_notch_info(channel)

            self.init_spk_detection_filter(notches_info=channel_info, filt_order=filt_order)
            if channel_info is not None:
                notch_count = len(channel_info.get('notch_freqs', [])) if isinstance(channel_info, dict) else 0
                source = 'MATLAB' if (self.use_matlab_notches and self.notches_info_mat) else 'Numpy'
                print(f"Applying {source} custom notch filter for channel {channel} with {notch_count} notches")
                filtered_data = self.ff_one_chan(data)
            else:
                filtered_data = signal.filtfilt(self.spk_b, self.spk_a, data, 
                                                padtype = 'odd', padlen=3*(max(len(self.spk_b),len(self.spk_a))-1))
            return filtered_data
        except:
            print(f'Filtering failed for {channel}')
            print(traceback.print_exc())
        finally:
            self.filt_lock.release()
 
    #endregion Signal filtering (bandpass, notch, power spectrum)

    def remove_packet_loss_data(self, data):
        """Replace sentinel min-int16 values (packet loss) with zeros."""
        pak_lost = data == np.iinfo(np.int16).min
        data[pak_lost] = 0

        return data

    def load_notch_filters(self):
        """Load precomputed notch filters from both numpy and MATLAB files, if available."""
        try:
            assets_folder = os.path.join(os.path.dirname(__file__), '..', 'assets', 'notch_filters')

            # Load NumPy notch filters
            npy_file = os.path.join(assets_folder, 'notch_filters.npy')
            self.notches_info_npy = None
            if os.path.exists(npy_file):
                loaded = np.load(npy_file, allow_pickle=True)
                if isinstance(loaded, np.ndarray):
                    self.notches_info_npy = [entry for entry in loaded.tolist() if entry is not None]
                elif isinstance(loaded, list):
                    self.notches_info_npy = [entry for entry in loaded if entry is not None]
                else:
                    self.notches_info_npy = loaded
                print(f"Loaded notch filters (NumPy) for {len(self.notches_info_npy or [])} channels from {npy_file}")

            # Load MATLAB notch filters
            self._load_notch_filters_from_mat()  # sets self.notches_info_mat when possible

            # self.compare_notch_files(assets_folder=assets_folder, tolerance=1.0, verbose=True)

            # Decide active source
            if self.use_matlab_notches and self.notches_info_mat:
                self.notches_info = self.notches_info_mat
            elif self.notches_info_npy:
                self.notches_info = self.notches_info_npy
            else:
                self.notches_info = None

            # Optional: load human-readable summary
            try:
                summary_file = os.path.join(assets_folder, 'detected_notches.npy')
                if os.path.exists(summary_file):
                    notches_summary = np.load(summary_file, allow_pickle=True)
                    # Only print brief info to avoid noise
                    if isinstance(notches_summary, dict) or hasattr(notches_summary, 'item'):
                        obj = notches_summary if isinstance(notches_summary, dict) else notches_summary.item()
                        print(f"Available detected notch channels: {len(list(obj.keys()))}")
            except Exception:
                pass

        except Exception as e:
            print(f"Error loading notch filters: {e}")
            # Still attempt MATLAB after error
            try:
                self._load_notch_filters_from_mat()
                self.notches_info = self.notches_info_mat
            except Exception:
                self.notches_info = None

    def _load_notch_filters_from_mat(self):
        """Loader for MATLAB `.mat` notch filter files; populates self.notches_info_mat if found."""
        self.notches_info_mat = None
        try:
            candidates = []
            # 1) Task folder, if provided
            if self.task_folder:
                candidates.append(os.path.join(self.task_folder, 'pre_processing_info.mat'))
            # 2) Project assets folder
            assets_mat = os.path.join(os.path.dirname(__file__), '..', 'assets', 'notch_filters', 'pre_processing_info.mat')
            candidates.append(assets_mat)

            for mat_path in candidates:
                if mat_path and os.path.exists(mat_path):
                    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
                    # Expect a 'process_info' key containing list-like per-channel entries
                    if 'process_info' in mat:
                        # Convert MATLAB 1-based indexing to Python 0-based indexing
                        self.notches_info_mat = []
                        for entry in mat['process_info']:
                            if isinstance(entry, dict) and 'chID' in entry:
                                # Create a copy and adjust chID for 0-based indexing
                                converted_entry = entry.copy()
                                converted_entry['chID'] = entry['chID'] - 1
                                self.notches_info_mat.append(converted_entry)
                            else:
                                self.notches_info_mat.append(entry)
                        print(f"Loaded notch filters (MATLAB) from: {mat_path} (converted to 0-based indexing)")
                        break
            if self.notches_info_mat is None:
                print("No MATLAB notch filter file found (pre_processing_info.mat)")
        except Exception as e:
            print(f"Error loading MATLAB notch filters: {e}")
            self.notches_info_mat = None

    def _convert_mat_to_notches_info(self, mat_data):
        """Convert MATLAB notch data to the expected Python dict structure.

        Parameters
        ----------
        mat_data : dict
            Data loaded via `scipy.io.loadmat`.

        Returns
        -------
        list[dict] or None
            List of notch info dicts or None when unparseable.
        """
        try:
            # This conversion depends on the structure of your MATLAB file
            # Adjust the field names and structure according to your specific .mat file format
            
            notches_info = []
            
            # Example conversion - modify based on your actual MATLAB file structure
            if 'notches' in mat_data:
                mat_notches = mat_data['notches']
                
                # If mat_notches is a structured array
                if hasattr(mat_notches, 'dtype') and mat_notches.dtype.names:
                    for i in range(mat_notches.size):
                        notch_entry = {}
                        
                        # Extract channel ID
                        if 'chID' in mat_notches.dtype.names:
                            notch_entry['chID'] = int(mat_notches['chID'][i][0][0])
                        elif 'channel_id' in mat_notches.dtype.names:
                            notch_entry['chID'] = int(mat_notches['channel_id'][i][0][0])
                        
                        # Extract channel label if available
                        if 'ch_label' in mat_notches.dtype.names:
                            notch_entry['ch_label'] = str(mat_notches['ch_label'][i][0])
                        
                        # Extract notch frequencies
                        if 'notch_freqs' in mat_notches.dtype.names:
                            notch_entry['notch_freqs'] = mat_notches['notch_freqs'][i][0].flatten().tolist()
                        
                        # Extract filter coefficients (SOS format)
                        if 'SOS' in mat_notches.dtype.names:
                            notch_entry['SOS'] = mat_notches['SOS'][i][0]
                        
                        # Extract gain
                        if 'G' in mat_notches.dtype.names:
                            notch_entry['G'] = float(mat_notches['G'][i][0][0])
                        
                        notches_info.append(notch_entry)
                
                # Alternative: if it's a simple dictionary structure
                elif isinstance(mat_notches, dict):
                    for ch_id, ch_data in mat_notches.items():
                        if isinstance(ch_id, str) and ch_id.startswith('ch_'):
                            notch_entry = {
                                'chID': int(ch_id.split('_')[-1]),
                                'ch_label': ch_id,
                                'notch_freqs': ch_data.get('freqs', []).flatten().tolist() if hasattr(ch_data.get('freqs', []), 'flatten') else ch_data.get('freqs', []),
                                'SOS': ch_data.get('SOS', np.array([])),
                                'G': float(ch_data.get('G', 1.0))
                            }
                            notches_info.append(notch_entry)
            
            return notches_info if notches_info else None
            
        except Exception as e:
            print(f"Error converting MATLAB notch filter data: {e}")
            print("You may need to adjust the _convert_mat_to_notches_info method for your specific MATLAB file format")
            return None

    def reload_notch_filters(self):
        """Reload notch filters after recomputation on disk."""
        self.load_notch_filters()

    def set_use_matlab_notches(self, enabled: bool):
        """Toggle between MATLAB and NumPy notch sources when both are available."""
        self.use_matlab_notches = bool(enabled)
        # Switch active list without reloading from disk
        prev = 'MATLAB' if (self.notches_info is self.notches_info_mat) else 'NumPy'
        if self.use_matlab_notches and self.notches_info_mat:
            self.notches_info = self.notches_info_mat
            now = 'MATLAB'
        elif self.notches_info_npy:
            self.notches_info = self.notches_info_npy
            now = 'NumPy'
        else:
            self.notches_info = None
            now = 'None'
        if prev != now:
            print(f"Notch source switched: {prev} -> {now}")

    def set_task_folder(self, task_folder: str):
        """Update task folder location and reload MATLAB notches if available."""
        self.task_folder = task_folder
        self.reload_notch_filters()

    def compare_notch_files(self, assets_folder=None, tolerance=1.0, verbose=True):
        """Compare notches stored in pre_processing_info.mat and notch_filters.npy.

        Parameters
        ----------
        assets_folder : str, optional
            Path to assets/notch_filters folder. If None, uses default location.
        tolerance : float, default 1.0
            Frequency tolerance in Hz for matching notches between files.
        verbose : bool, default True
            Whether to print detailed comparison results.

        Returns
        -------
        dict
            Comparison results containing matched channels, frequency differences,
            and summary statistics.
        """
        if assets_folder is None:
            assets_folder = os.path.join(os.path.dirname(__file__), '..', 'assets', 'notch_filters')

        npy_file = os.path.join(assets_folder, 'notch_filters.npy')
        mat_file = os.path.join(assets_folder, 'pre_processing_info.mat')

        if not os.path.exists(npy_file):
            if verbose:
                print(f"notch_filters.npy not found at: {npy_file}")
            return {'error': 'npy_file_not_found'}

        if not os.path.exists(mat_file):
            if verbose:
                print(f"pre_processing_info.mat not found at: {mat_file}")
            return {'error': 'mat_file_not_found'}

        try:
            # Load numpy data
            npy_data = np.load(npy_file, allow_pickle=True).tolist()
            if verbose:
                print(f"Loaded {len(npy_data)} channels from notch_filters.npy")

            # Load MATLAB data
            mat_data = scipy.io.loadmat(mat_file, simplify_cells=True)
            mat_process_info = mat_data['process_info']
            if verbose:
                print(f"Loaded {len(mat_process_info)} channels from pre_processing_info.mat")

            # Create lookup dictionaries for both datasets
            npy_by_chid = {}
            for entry in npy_data:
                if isinstance(entry, dict) and 'chID' in entry:
                    npy_by_chid[entry['chID']] = entry

            mat_by_chid = {}
            for entry in mat_process_info:
                if isinstance(entry, dict) and 'chID' in entry:
                    mat_by_chid[entry['chID']] = entry

            # Find common channels
            npy_chids = set(npy_by_chid.keys())
            mat_chids = set(mat_by_chid.keys())
            common_chids = npy_chids.intersection(mat_chids)

            if verbose:
                print(f"\nChannel comparison:")
                print(f"  NPY file channels: {len(npy_chids)} (IDs: {sorted(npy_chids)})")
                print(f"  MAT file channels: {len(mat_chids)} (IDs: {sorted(mat_chids)})")
                print(f"  Common channels: {len(common_chids)} (IDs: {sorted(common_chids)})")
                print(f"  Only in NPY: {sorted(npy_chids - mat_chids)}")
                print(f"  Only in MAT: {sorted(mat_chids - npy_chids)}")

            # Compare notches for common channels
            results = {
                'common_channels': len(common_chids),
                'npy_only_channels': len(npy_chids - mat_chids),
                'mat_only_channels': len(mat_chids - npy_chids),
                'channel_comparisons': {},
                'summary': {
                    'total_matched': 0,
                    'total_npy_notches': 0,
                    'total_mat_notches': 0,
                    'avg_frequency_diff': 0.0,
                    'max_frequency_diff': 0.0
                }
            }

            total_matched = 0
            total_npy_notches = 0
            total_mat_notches = 0
            all_freq_diffs = []

            for chid in sorted(common_chids):
                npy_entry = npy_by_chid[chid]
                mat_entry = mat_by_chid[chid]

                # Extract notch frequencies from NPY data
                npy_freqs = []
                if 'notch_details' in npy_entry:
                    npy_freqs = [detail['freq'] for detail in npy_entry['notch_details']]

                # Extract notch frequencies from MAT data
                mat_freqs = []
                if 'notches' in mat_entry and 'freq' in mat_entry['notches']:
                    mat_freqs = list(mat_entry['notches']['freq'])

                # Match frequencies within tolerance
                matched_pairs = []
                npy_matched = set()
                mat_matched = set()

                for i, npy_freq in enumerate(npy_freqs):
                    for j, mat_freq in enumerate(mat_freqs):
                        if j not in mat_matched:
                            freq_diff = abs(npy_freq - mat_freq)
                            if freq_diff <= tolerance:
                                matched_pairs.append((i, j, npy_freq, mat_freq, freq_diff))
                                npy_matched.add(i)
                                mat_matched.add(j)
                                all_freq_diffs.append(freq_diff)
                                break

                # Store channel comparison results
                channel_result = {
                    'ch_label': npy_entry.get('ch_label', f'ch_{chid}'),
                    'npy_notches': len(npy_freqs),
                    'mat_notches': len(mat_freqs),
                    'matched_notches': len(matched_pairs),
                    'npy_unmatched': len(npy_freqs) - len(npy_matched),
                    'mat_unmatched': len(mat_freqs) - len(mat_matched),
                    'npy_frequencies': npy_freqs,
                    'mat_frequencies': mat_freqs,
                    'matched_pairs': matched_pairs,
                    'avg_freq_diff': np.mean([pair[4] for pair in matched_pairs]) if matched_pairs else 0.0
                }

                results['channel_comparisons'][chid] = channel_result

                # Update totals
                total_matched += len(matched_pairs)
                total_npy_notches += len(npy_freqs)
                total_mat_notches += len(mat_freqs)

                if verbose:
                    print(f"\nChannel {chid} ({channel_result['ch_label']}):")
                    print(f"  NPY notches: {len(npy_freqs)}")
                    print(f"  MAT notches: {len(mat_freqs)}")
                    print(f"  Matched: {len(matched_pairs)}")
                    print(f"  NPY unmatched: {channel_result['npy_unmatched']}")
                    print(f"  MAT unmatched: {channel_result['mat_unmatched']}")
                    if matched_pairs:
                        print(f"  Avg frequency diff: {channel_result['avg_freq_diff']:.3f} Hz")

            # Calculate summary statistics
            results['summary'] = {
                'total_matched': total_matched,
                'total_npy_notches': total_npy_notches,
                'total_mat_notches': total_mat_notches,
                'match_rate_npy': total_matched / total_npy_notches if total_npy_notches > 0 else 0.0,
                'match_rate_mat': total_matched / total_mat_notches if total_mat_notches > 0 else 0.0,
                'avg_frequency_diff': np.mean(all_freq_diffs) if all_freq_diffs else 0.0,
                'max_frequency_diff': np.max(all_freq_diffs) if all_freq_diffs else 0.0,
                'std_frequency_diff': np.std(all_freq_diffs) if all_freq_diffs else 0.0
            }

            if verbose:
                print(f"\n=== SUMMARY ===")
                print(f"Total notches in NPY: {total_npy_notches}")
                print(f"Total notches in MAT: {total_mat_notches}")
                print(f"Total matched notches: {total_matched}")
                print(f"NPY match rate: {results['summary']['match_rate_npy']:.1%}")
                print(f"MAT match rate: {results['summary']['match_rate_mat']:.1%}")
                if all_freq_diffs:
                    print(f"Average frequency difference: {results['summary']['avg_frequency_diff']:.3f} Hz")
                    print(f"Max frequency difference: {results['summary']['max_frequency_diff']:.3f} Hz")
                    print(f"Std frequency difference: {results['summary']['std_frequency_diff']:.3f} Hz")

                # Overall similarity assessment
                avg_match_rate = (results['summary']['match_rate_npy'] + results['summary']['match_rate_mat']) / 2
                if avg_match_rate > 0.9:
                    similarity = "Very Similar"
                elif avg_match_rate > 0.7:
                    similarity = "Mostly Similar"
                elif avg_match_rate > 0.5:
                    similarity = "Moderately Similar"
                else:
                    similarity = "Different"

                print(f"\nOverall Assessment: {similarity} (avg match rate: {avg_match_rate:.1%})")

            return results

        except Exception as e:
            if verbose:
                print(f"Error comparing notch files: {e}")
                import traceback
                traceback.print_exc()
            return {'error': str(e)}
    

if __name__ == '__main__':
    import os

    # Example usage
    filtering = Filtering()
    
    # Compare notch files if both are available
    comparison_result = filtering.compare_notch_files()
    print("Notch comparison completed!")
