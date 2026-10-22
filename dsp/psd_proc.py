"""
PSD Processing with Persistent Multiprocessing - Windows Compatible
==================================================================

A persistent multiprocessing system for PSD calculation that works reliably
on Windows by avoiding complex object pickling issues.
"""
# Author: Sunil Mathew
# Date: June 14, 2026

import time
import traceback
import numpy as np
import os
import sys
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from scipy import signal

# Add the project root to path for importing filtering module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set multiprocessing start method for Windows compatibility
if __name__ != '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


@dataclass
class PSDTask:
    """Simple task structure with basic types only"""
    task_id: str
    ch_data: Dict[str, np.ndarray] 
    fs: float
    # Filtering parameters
    lowcut: float
    highcut: float
    filt_order: int
    notch_freqs: List[float]
    apply_notches: bool
    use_matlab_notches: bool
    # Notch detection parameters
    span_smooth: int = 21
    db_thr: float = 5.0
    freq_min: float = 300.0
    freq_max: float = 3000.0


@dataclass
class PSDResult:
    """Simple result structure with basic types only"""
    task_id: str
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0


def simple_psd_worker(task_queue, result_queue, worker_id):
    """Simple worker function that recreates all objects locally"""
    # Configure for macOS - use Accessory policy to allow dialogs but minimize dock presence
    if sys.platform == "darwin":
        try:
            import objc
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            app = NSApplication.sharedApplication()
            # Use Accessory policy: can show windows but no dock icon
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except:
            pass
    
    print(f"PSD Worker {worker_id} started")
    
    while True:
        try:
            # Get task from queue
            task = task_queue.get(timeout=1.0)
            
            if task is None:  # Shutdown signal
                print(f"PSD Worker {worker_id} shutting down")
                break
                
            # Process the task
            start_time = time.time()
            result = process_psd_task_simple(task)
            result.processing_time = time.time() - start_time
            
            # Send result back
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"PSD Worker {worker_id} error: {e}")
            traceback.print_exc()


def process_psd_task_simple(task: PSDTask) -> PSDResult:
    """Process PSD task with all objects created locally using the filtering module"""
    try:
        # Import filtering module locally in worker
        try:
            from dsp.filtering import Filtering
            print(f"PSD Worker: Using filtering.py module for task {task.task_id}")
        except ImportError as e:
            print(f"PSD Worker: Failed to import filtering module: {e}, using fallback")
            # Fallback to simple filtering if filtering module not available
            return process_psd_task_fallback(task)
        
        # Create filtering object locally in worker
        try:
            filtering = Filtering(
                lowcut=task.lowcut,
                highcut=task.highcut,
                filt_order=task.filt_order,
                fs=task.fs,
                basic_notches=task.notch_freqs,
                apply_notches=task.apply_notches,
                use_matlab_notches=task.use_matlab_notches
            )
            print(f"PSD Worker: Created filtering object - notches enabled: {task.apply_notches}, "
                  f"notch freqs: {task.notch_freqs}")
        except Exception as e:
            print(f"PSD Worker: Failed to create filtering object: {e}, using fallback")
            return process_psd_task_fallback(task)
        
        # Create FFT parameters locally
        fft_length = 2**np.ceil(np.log2(task.fs*2))/task.fs
        nfft = int(fft_length * task.fs)
        window = signal.windows.barthann(nfft)
        
        psd_results = {}
        failed_channels = []
        
        for ch_label, data in task.ch_data.items():
            try:
                # Calculate raw PSD
                f, Pxx_den = signal.welch(x=data, fs=task.fs, window=window, nfft=nfft)
                psd_raw = 10*np.log10(Pxx_den + 1e-12)
                
                # Apply proper filtering using filtering module
                try:
                    filtered_data = filtering.custom_bp_filter(data=data, fs=task.fs, channel=ch_label)
                    if filtered_data is not None:
                        # Calculate filtered PSD
                        f_filt, Pxx_den_filt = signal.welch(x=filtered_data, fs=task.fs, 
                                                           window=window, nfft=nfft)
                        psd_filtered = 10*np.log10(Pxx_den_filt + 1e-12)
                        
                        # Debug: Check if notches were applied
                        if task.apply_notches and len(task.notch_freqs) > 0:
                            # Check for significant differences between raw and filtered PSD at notch frequencies
                            notch_effects = []
                            for notch_freq in task.notch_freqs:
                                # Find closest frequency bin
                                freq_idx = np.argmin(np.abs(f - notch_freq))
                                if freq_idx < len(psd_raw) and freq_idx < len(psd_filtered):
                                    reduction = psd_raw[freq_idx] - psd_filtered[freq_idx]
                                    notch_effects.append(f"{notch_freq}Hz: {reduction:.1f}dB")
                            
                            if len(notch_effects) > 0:
                                print(f"PSD Worker {ch_label}: Notch effects - {', '.join(notch_effects)}")
                    else:
                        # Fallback if filtering fails
                        psd_filtered = psd_raw.copy()
                        print(f"Warning: Filtering failed for {ch_label}, using raw PSD")
                except Exception as filter_error:
                    print(f"Filtering error for {ch_label}: {filter_error}")
                    psd_filtered = psd_raw.copy()
                
                psd_results[ch_label] = {
                    'freq': f,
                    'psd_raw': psd_raw,
                    'psd_filtered': psd_filtered
                }
                
            except Exception as e:
                print(f"Error processing channel {ch_label}: {e}")
                failed_channels.append(ch_label)
                continue
        
        return PSDResult(
            task_id=task.task_id,
            success=True,
            data={
                'psd_results': psd_results,
                'failed_channels': failed_channels,
                'total_channels': len(task.ch_data),
                'successful_channels': len(psd_results)
            },
            error_message=None
        )
        
    except Exception as e:
        return PSDResult(
            task_id=task.task_id,
            success=False,
            data={},
            error_message=str(e)
        )


def process_psd_task_fallback(task: PSDTask) -> PSDResult:
    """Fallback PSD processing with simple scipy filtering"""
    try:
        print(f"PSD Worker: Using fallback filtering for task {task.task_id}")
        # Create FFT parameters locally
        fft_length = 2**np.ceil(np.log2(task.fs*2))/task.fs
        nfft = int(fft_length * task.fs)
        window = signal.windows.barthann(nfft)
        
        psd_results = {}
        failed_channels = []
        
        for ch_label, data in task.ch_data.items():
            try:
                # Calculate raw PSD
                f, Pxx_den = signal.welch(x=data, fs=task.fs, window=window, nfft=nfft)
                psd_raw = 10*np.log10(Pxx_den + 1e-12)
                
                # Apply simple bandpass filter (fallback)
                if task.lowcut < task.highcut < task.fs/2:
                    sos = signal.butter(task.filt_order, [task.lowcut, task.highcut], 
                                      btype='band', fs=task.fs, output='sos')
                    filtered_data = signal.sosfilt(sos, data)
                    
                    # Calculate filtered PSD
                    f_filt, Pxx_den_filt = signal.welch(x=filtered_data, fs=task.fs, 
                                                       window=window, nfft=nfft)
                    psd_filtered = 10*np.log10(Pxx_den_filt + 1e-12)
                else:
                    psd_filtered = psd_raw.copy()
                
                psd_results[ch_label] = {
                    'freq': f,
                    'psd_raw': psd_raw,
                    'psd_filtered': psd_filtered
                }
                
            except Exception as e:
                print(f"Error processing channel {ch_label}: {e}")
                failed_channels.append(ch_label)
                continue
        
        return PSDResult(
            task_id=task.task_id,
            success=True,
            data={
                'psd_results': psd_results,
                'failed_channels': failed_channels,
                'total_channels': len(task.ch_data),
                'successful_channels': len(psd_results)
            },
            error_message=None
        )
        
    except Exception as e:
        return PSDResult(
            task_id=task.task_id,
            success=False,
            data={},
            error_message=str(e)
        )


class PSDProcessor:
    """Simplified PSD processor that avoids pickling issues"""
    
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        self.is_running = False
        
        # Default filter parameters (matching filtering.py defaults)
        self.fs = 30000.0
        self.lowcut = 300.0
        self.highcut = 3000.0
        self.filt_order = 4
        self.notch_freqs = [60.0, 120.0, 180.0]
        self.apply_notches = True
        self.use_matlab_notches = False
        # PSD calculation parameters
        self.span_smooth = 21
        self.db_thr = 5.0
        self.freq_min = 300.0
        self.freq_max = 3000.0
        
    def start(self) -> bool:
        """Start the processor"""
        if self.is_running:
            print("PSD Processor already running")
            return True
            
        try:
            print(f"Starting PSD Processor with {self.num_workers} workers...")
            
            for i in range(self.num_workers):
                try:
                    worker = mp.Process(target=simple_psd_worker, 
                                      args=(self.task_queue, self.result_queue, i))
                    worker.daemon = True
                    worker.start()
                    self.workers.append(worker)
                    print(f"PSD Worker {i} started")
                except Exception as e:
                    print(f"Error starting worker {i}: {e}")
                    continue
            
            if len(self.workers) == 0:
                print("Failed to start any workers")
                return False
            
            self.is_running = True
            print(f"PSD Processor started successfully with {len(self.workers)} workers")
            return True
            
        except Exception as e:
            print(f"Error starting PSD Processor: {e}")
            traceback.print_exc()
            self.stop()
            return False
    
    def stop(self):
        """Stop the processor"""
        if not self.is_running:
            return
            
        print("Stopping PSD Processor...")
        
        # Send shutdown signals
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        self.workers.clear()
        self.is_running = False
        print("PSD Processor stopped")
    
    def update_filter_params(self, lowcut=None, highcut=None, fs=None, 
                           filt_order=None, notch_freqs=None, apply_notches=None,
                           use_matlab_notches=None, span_smooth=None, db_thr=None,
                           freq_min=None, freq_max=None):
        """Update filter parameters"""
        if lowcut is not None:
            self.lowcut = lowcut
        if highcut is not None:
            self.highcut = highcut
        if fs is not None:
            self.fs = fs
        if filt_order is not None:
            self.filt_order = filt_order
        if notch_freqs is not None:
            self.notch_freqs = notch_freqs
        if apply_notches is not None:
            self.apply_notches = apply_notches
        if use_matlab_notches is not None:
            self.use_matlab_notches = use_matlab_notches
        if span_smooth is not None:
            self.span_smooth = span_smooth
        if db_thr is not None:
            self.db_thr = db_thr
        if freq_min is not None:
            self.freq_min = freq_min
        if freq_max is not None:
            self.freq_max = freq_max
    
    def calculate_psd(self, ch_data: Dict[str, np.ndarray], task_id: str = None) -> str:
        """Submit PSD calculation task"""
        if not self.is_running:
            raise RuntimeError("PSD Processor not running. Call start() first.")
            
        if task_id is None:
            task_id = f"psd_{time.time()}"
            
        task = PSDTask(
            task_id=task_id,
            ch_data=ch_data,
            fs=self.fs,
            lowcut=self.lowcut,
            highcut=self.highcut,
            filt_order=self.filt_order,
            notch_freqs=self.notch_freqs,
            apply_notches=self.apply_notches,
            use_matlab_notches=self.use_matlab_notches,
            span_smooth=self.span_smooth,
            db_thr=self.db_thr,
            freq_min=self.freq_min,
            freq_max=self.freq_max
        )
        
        self.task_queue.put(task)
        return task_id
    
    def get_result(self, timeout: float = None) -> Optional[PSDResult]:
        """Get result from queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_all_results(self, timeout: float = 0.0) -> List[PSDResult]:
        """Retrieve all available results, optionally waiting for the first one."""
        results: List[PSDResult] = []

        # Attempt to fetch the first result with the provided timeout
        try:
            if timeout and timeout > 0.0:
                first_result = self.result_queue.get(timeout=timeout)
            else:
                first_result = self.result_queue.get_nowait()
            results.append(first_result)
        except queue.Empty:
            return results

        # Drain any additional results without blocking
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break

        return results


# Global processor instance
_processor = None


def get_processor() -> PSDProcessor:
    """Get the global PSD processor instance"""
    global _processor
    if _processor is None:
        _processor = PSDProcessor()
    return _processor


def start_processor() -> bool:
    """Start the global PSD processor"""
    processor = get_processor()
    return processor.start()


def stop_processor():
    """Stop the global PSD processor"""
    global _processor
    if _processor is not None:
        _processor.stop()
        _processor = None


def is_processor_running() -> bool:
    """Check if the global PSD processor is running"""
    global _processor
    return _processor is not None and _processor.is_running


# Context manager for automatic processor management
class PSDProcessorContext:
    """Context manager for PSD processor lifecycle management"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers
        self.processor = None
        
    def __enter__(self) -> PSDProcessor:
        self.processor = PSDProcessor(self.num_workers)
        if not self.processor.start():
            raise RuntimeError("Failed to start PSD processor")
        return self.processor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.processor:
            self.processor.stop()


if __name__ == "__main__":
    # Test the PSD processor
    print("Testing PSD Processor...")
    
    # Create some test data
    fs = 30000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(fs * duration), False)
    
    # Create test signals with known frequencies
    test_data = {
        'ch_001': np.sin(2*np.pi*60*t) + 0.5*np.sin(2*np.pi*500*t) + 0.1*np.random.randn(len(t)),
        'ch_002': np.sin(2*np.pi*120*t) + 0.3*np.sin(2*np.pi*800*t) + 0.1*np.random.randn(len(t)),
        'ch_003': 0.2*np.random.randn(len(t))  # Noise only
    }
    
    # Test with context manager
    try:
        with PSDProcessorContext(num_workers=2) as processor:
            print("Processor started successfully")
            
            # Calculate PSDs
            task_id = processor.calculate_psd(test_data)
            print(f"Submitted PSD task: {task_id}")
            
            # Wait for results
            start_time = time.time()
            psd_result = None
            while time.time() - start_time < 10.0:  # 10 second timeout
                result = processor.get_result(timeout=0.5)
                if result and result.task_id == task_id:
                    psd_result = result
                    break
            
            if psd_result and psd_result.success:
                print(f"PSD calculation completed in {psd_result.processing_time:.2f}s")
                psd_data = psd_result.data['psd_results']
                print(f"Successfully processed {len(psd_data)} channels")
            else:
                print("PSD calculation failed or timed out")
                
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()