# Author: Sunil Mathew
# Date: June 14, 2026
# Shared memory buffer for high-performance data transfer between acquisition and UI processes

import multiprocessing.shared_memory as sm
import numpy as np
import threading
import time
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class BufferMetadata:
    """Metadata for the shared visualization buffer"""
    write_position: int = 0
    read_position: int = 0
    timestamp: float = 0.0
    sample_rate: float = 30000.0
    num_channels: int = 0
    buffer_size: int = 0
    events_count: int = 0
    is_valid: bool = False
    sequence_number: int = 0


@dataclass
class PSDMetadata:
    """Metadata for PSD accumulation buffer"""
    selected_channel: int = 0
    channel_name: str = ""
    sample_rate: float = 30000.0
    fft_length: float = 2.0  # seconds
    nfft: int = 65536
    accumulated_samples: int = 0
    n_averages: int = 0
    auto_refresh_enabled: bool = False
    last_update: float = 0.0
    is_valid: bool = False


class SharedVisualizationBuffer:
    """
    High-performance shared memory buffer for visualization data transfer.
    Eliminates serialization overhead between acquisition and UI processes.
    """
    
    METADATA_SIZE = 1024  # Size for metadata structure
    EVENT_ENTRY_SIZE = 16  # 8 bytes timestamp + 4 bytes data + 4 bytes padding
    MAX_EVENTS = 1000     # Maximum number of events to store
    
    def __init__(self, length_seconds: float = 5.0, fs: float = 30000.0, 
                 num_channels: int = 32, create: bool = True, name_prefix: str = "meg_vis"):
        """
        Initialize shared visualization buffer.
        
        Args:
            length_seconds: Buffer length in seconds
            fs: Sampling frequency
            num_channels: Number of channels
            create: True to create new buffer, False to attach to existing
            name_prefix: Prefix for shared memory names
        """
        self.length_seconds = length_seconds
        self.fs = fs
        self.num_channels = num_channels
        self.buffer_size = int(length_seconds * fs)
        self.name_prefix = name_prefix
        
        # Calculate sizes
        self.data_size = self.buffer_size * num_channels * 4  # float32
        self.events_size = self.MAX_EVENTS * self.EVENT_ENTRY_SIZE
        
        # Shared memory names
        self.data_name = f"{name_prefix}_data"
        self.meta_name = f"{name_prefix}_meta"
        self.events_name = f"{name_prefix}_events"
        
        # Thread safety
        self._lock = threading.RLock()
        self._sequence_number = 0
        
        if create:
            self._create_shared_memory()
        else:
            self._attach_shared_memory()
            
        self._setup_numpy_views()
        
        if create:
            self._initialize_metadata()
    
    def _create_shared_memory(self):
        """Create new shared memory segments"""
        try:
            self.shm_data = sm.SharedMemory(create=True, size=self.data_size, name=self.data_name)
            self.shm_meta = sm.SharedMemory(create=True, size=self.METADATA_SIZE, name=self.meta_name)
            self.shm_events = sm.SharedMemory(create=True, size=self.events_size, name=self.events_name)
        except FileExistsError:
            # If already exists, clean up and recreate
            self._cleanup_existing()
            self.shm_data = sm.SharedMemory(create=True, size=self.data_size, name=self.data_name)
            self.shm_meta = sm.SharedMemory(create=True, size=self.METADATA_SIZE, name=self.meta_name)
            self.shm_events = sm.SharedMemory(create=True, size=self.events_size, name=self.events_name)
    
    def _attach_shared_memory(self):
        """Attach to existing shared memory segments"""
        try:
            self.shm_data = sm.SharedMemory(name=self.data_name)
            self.shm_meta = sm.SharedMemory(name=self.meta_name)
            self.shm_events = sm.SharedMemory(name=self.events_name)
        except FileNotFoundError as e:
            raise RuntimeError(f"Shared memory buffer not found: {e}")
    
    def _setup_numpy_views(self):
        """Create numpy array views of shared memory"""
        # Data buffer: [channels, samples]
        self.data = np.ndarray((self.num_channels, self.buffer_size), 
                              dtype=np.float32, buffer=self.shm_data.buf)
        
        # Metadata buffer
        self.meta_buffer = np.ndarray(self.METADATA_SIZE // 4, 
                                     dtype=np.float32, buffer=self.shm_meta.buf)
        
        # Events buffer: structured array
        self.events_buffer = np.ndarray(self.MAX_EVENTS, 
                                       dtype=[('timestamp', np.float64), ('data', np.int32)],
                                       buffer=self.shm_events.buf)
    
    def _initialize_metadata(self):
        """Initialize metadata structure"""
        with self._lock:
            # Pack metadata into buffer
            metadata = BufferMetadata(
                write_position=0,
                read_position=0,
                timestamp=time.time(),
                sample_rate=self.fs,
                num_channels=self.num_channels,
                buffer_size=self.buffer_size,
                events_count=0,
                is_valid=True,
                sequence_number=0
            )
            self._write_metadata(metadata)
    
    def _write_metadata(self, metadata: BufferMetadata):
        """Write metadata to shared memory"""
        # Pack metadata into first part of buffer
        packed_data = struct.pack('iiddiiibi', 
                                 metadata.write_position,
                                 metadata.read_position,
                                 metadata.timestamp,
                                 metadata.sample_rate,
                                 metadata.num_channels,
                                 metadata.buffer_size,
                                 metadata.events_count,
                                 metadata.is_valid,
                                 metadata.sequence_number)
        
        # Write to beginning of meta buffer
        np.frombuffer(packed_data, dtype=np.uint8).astype(np.float32)[:len(packed_data)//4]
        self.meta_buffer[:len(packed_data)//4] = np.frombuffer(packed_data, dtype=np.uint8).view(np.float32)[:len(packed_data)//4]
    
    def _read_metadata(self) -> BufferMetadata:
        """Read metadata from shared memory"""
        # Read packed data
        packed_size = struct.calcsize('iiddiiibi')
        raw_bytes = self.meta_buffer[:packed_size//4].view(np.uint8).tobytes()[:packed_size]
        
        unpacked = struct.unpack('iiddiiibi', raw_bytes)
        
        return BufferMetadata(
            write_position=unpacked[0],
            read_position=unpacked[1],
            timestamp=unpacked[2],
            sample_rate=unpacked[3],
            num_channels=unpacked[4],
            buffer_size=unpacked[5],
            events_count=unpacked[6],
            is_valid=bool(unpacked[7]),
            sequence_number=unpacked[8]
        )
    
    def update_buffer(self, new_data: np.ndarray, timestamp: float, 
                     events: Optional[Dict] = None) -> bool:
        """
        Update buffer with new data using circular buffer approach.
        
        Args:
            new_data: New data array [channels, samples]
            timestamp: Timestamp of first sample
            events: Optional events dictionary with 'data' and 'timestamp' keys
            
        Returns:
            True if update successful, False otherwise
        """
        if new_data.shape[0] != self.num_channels:
            return False
            
        samples_to_write = new_data.shape[1]
        if samples_to_write > self.buffer_size:
            return False
            
        with self._lock:
            try:
                # Read current metadata
                metadata = self._read_metadata()
                
                # Update circular buffer
                write_pos = metadata.write_position
                if write_pos + samples_to_write <= self.buffer_size:
                    # Simple case: write fits without wrapping
                    self.data[:, write_pos:write_pos + samples_to_write] = new_data
                else:
                    # Need to wrap around
                    first_chunk = self.buffer_size - write_pos
                    second_chunk = samples_to_write - first_chunk
                    
                    self.data[:, write_pos:] = new_data[:, :first_chunk]
                    self.data[:, :second_chunk] = new_data[:, first_chunk:]
                
                # Update events if provided
                events_count = 0
                if events and 'data' in events and 'timestamp' in events:
                    events_count = min(len(events['data']), self.MAX_EVENTS)
                    for i in range(events_count):
                        self.events_buffer[i]['timestamp'] = events['timestamp'][i]
                        self.events_buffer[i]['data'] = events['data'][i]
                
                # Update metadata
                self._sequence_number += 1
                new_write_pos = (write_pos + samples_to_write) % self.buffer_size
                
                updated_metadata = BufferMetadata(
                    write_position=new_write_pos,
                    read_position=metadata.read_position,
                    timestamp=timestamp,
                    sample_rate=self.fs,
                    num_channels=self.num_channels,
                    buffer_size=self.buffer_size,
                    events_count=events_count,
                    is_valid=True,
                    sequence_number=self._sequence_number
                )
                
                self._write_metadata(updated_metadata)
                return True
                
            except Exception as e:
                print(f"Error updating buffer: {e}")
                return False
    
    def read_buffer(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                  Optional[Dict], BufferMetadata]:
        """
        Read current buffer data.
        
        Returns:
            Tuple of (time_vector, data_array, events_dict, metadata)
        """
        with self._lock:
            try:
                metadata = self._read_metadata()
                
                if not metadata.is_valid:
                    return None, None, None, metadata
                
                # Create time vector
                time_vector = np.arange(self.buffer_size, dtype=np.float32) / self.fs
                
                # Copy data (to avoid race conditions)
                data_copy = self.data.copy()
                
                # Read events
                events_dict = None
                if metadata.events_count > 0:
                    events_dict = {
                        'data': [int(self.events_buffer[i]['data']) for i in range(metadata.events_count)],
                        'timestamp': [float(self.events_buffer[i]['timestamp']) for i in range(metadata.events_count)]
                    }
                
                return time_vector, data_copy, events_dict, metadata
                
            except Exception as e:
                print(f"Error reading buffer: {e}")
                return None, None, None, BufferMetadata(is_valid=False)
    
    def get_latest_data(self, length_samples: Optional[int] = None) -> Tuple[Optional[np.ndarray], 
                                                                            Optional[np.ndarray]]:
        """
        Get the most recent data from the circular buffer.
        
        Args:
            length_samples: Number of samples to retrieve (default: full buffer)
            
        Returns:
            Tuple of (time_vector, data_array) for the latest data
        """
        if length_samples is None:
            length_samples = self.buffer_size
            
        length_samples = min(length_samples, self.buffer_size)
        
        with self._lock:
            try:
                metadata = self._read_metadata()
                
                if not metadata.is_valid:
                    return None, None
                
                # Calculate read position for latest data
                end_pos = metadata.write_position
                start_pos = (end_pos - length_samples) % self.buffer_size
                
                # Extract data
                if start_pos < end_pos:
                    # No wrap-around
                    data_slice = self.data[:, start_pos:end_pos].copy()
                else:
                    # Wrap-around case
                    first_part = self.data[:, start_pos:].copy()
                    second_part = self.data[:, :end_pos].copy()
                    data_slice = np.concatenate([first_part, second_part], axis=1)
                
                # Create corresponding time vector
                time_vector = np.arange(length_samples, dtype=np.float32) / self.fs
                
                return time_vector, data_slice
                
            except Exception as e:
                print(f"Error getting latest data: {e}")
                return None, None
    
    def is_valid(self) -> bool:
        """Check if buffer is valid and accessible"""
        try:
            metadata = self._read_metadata()
            return metadata.is_valid
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get buffer information"""
        metadata = self._read_metadata()
        return {
            'length_seconds': self.length_seconds,
            'sample_rate': metadata.sample_rate,
            'num_channels': metadata.num_channels,
            'buffer_size': metadata.buffer_size,
            'data_size_mb': self.data_size / (1024 * 1024),
            'is_valid': metadata.is_valid,
            'sequence_number': metadata.sequence_number,
            'last_update': metadata.timestamp
        }
    
    def _cleanup_existing(self):
        """Clean up existing shared memory segments"""
        for name in [self.data_name, self.meta_name, self.events_name]:
            try:
                shm = sm.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
    
    def cleanup(self):
        """Clean up shared memory resources"""
        try:
            if hasattr(self, 'shm_data'):
                self.shm_data.close()
                self.shm_data.unlink()
            if hasattr(self, 'shm_meta'):
                self.shm_meta.close()
                self.shm_meta.unlink()
            if hasattr(self, 'shm_events'):
                self.shm_events.close()
                self.shm_events.unlink()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't cleanup automatically - let explicit cleanup handle it
        pass


class SharedPSDBuffer:
    """
    Shared memory buffer for PSD data accumulation and running averages.
    Supports real-time PSD calculation with running average functionality.
    """
    
    METADATA_SIZE = 1024  # Size for metadata structure
    MAX_CHANNELS = 256    # Maximum number of channels supported
    
    def __init__(self, fs: float = 30000.0, max_channels: int = 256, 
                 fft_length: float = 2.0, create: bool = True, name_prefix: str = "meg_psd"):
        """
        Initialize shared PSD buffer.
        
        Args:
            fs: Sampling frequency
            max_channels: Maximum number of channels
            fft_length: FFT window length in seconds
            create: True to create new buffer, False to attach to existing
            name_prefix: Prefix for shared memory names
        """
        self.fs = fs
        self.max_channels = max_channels
        self.fft_length = fft_length
        self.nfft = int(2**np.ceil(np.log2(fs * fft_length)))
        self.n_freq_bins = self.nfft // 2 + 1  # One-sided spectrum
        self.name_prefix = name_prefix
        
        # Calculate buffer sizes
        # Channel data: [channels, accumulation_samples] 
        self.accumulation_samples = int(50 * self.nfft)  # ~2 minutes of data
        self.channel_data_size = max_channels * self.accumulation_samples * 4  # float32
        
        # PSD data: [channels, freq_bins] for running averages
        self.psd_data_size = max_channels * self.n_freq_bins * 4  # float32
        
        # Frequency vector: [freq_bins]
        self.freq_data_size = self.n_freq_bins * 4  # float32
        
        # Sample counts per channel: [channels]
        self.counts_data_size = max_channels * 4  # int32
        
        # Shared memory names
        self.channel_data_name = f"{name_prefix}_channel_data"
        self.psd_data_name = f"{name_prefix}_psd_data"
        self.freq_data_name = f"{name_prefix}_freq_data"
        self.counts_data_name = f"{name_prefix}_counts_data"
        self.meta_name = f"{name_prefix}_meta"
        
        # Thread safety
        self._lock = threading.RLock()
        
        if create:
            self._create_shared_memory()
        else:
            self._attach_shared_memory()
            
        self._setup_numpy_views()
        
        if create:
            self._initialize_metadata()
            self._initialize_frequency_vector()
    
    def _create_shared_memory(self):
        """Create new shared memory segments"""
        try:
            self.shm_channel_data = sm.SharedMemory(create=True, size=self.channel_data_size, name=self.channel_data_name)
            self.shm_psd_data = sm.SharedMemory(create=True, size=self.psd_data_size, name=self.psd_data_name)
            self.shm_freq_data = sm.SharedMemory(create=True, size=self.freq_data_size, name=self.freq_data_name)
            self.shm_counts_data = sm.SharedMemory(create=True, size=self.counts_data_size, name=self.counts_data_name)
            self.shm_meta = sm.SharedMemory(create=True, size=self.METADATA_SIZE, name=self.meta_name)
        except FileExistsError:
            # If already exists, clean up and recreate
            self._cleanup_existing()
            self.shm_channel_data = sm.SharedMemory(create=True, size=self.channel_data_size, name=self.channel_data_name)
            self.shm_psd_data = sm.SharedMemory(create=True, size=self.psd_data_size, name=self.psd_data_name)
            self.shm_freq_data = sm.SharedMemory(create=True, size=self.freq_data_size, name=self.freq_data_name)
            self.shm_counts_data = sm.SharedMemory(create=True, size=self.counts_data_size, name=self.counts_data_name)
            self.shm_meta = sm.SharedMemory(create=True, size=self.METADATA_SIZE, name=self.meta_name)
    
    def _attach_shared_memory(self):
        """Attach to existing shared memory segments"""
        try:
            self.shm_channel_data = sm.SharedMemory(name=self.channel_data_name)
            self.shm_psd_data = sm.SharedMemory(name=self.psd_data_name)
            self.shm_freq_data = sm.SharedMemory(name=self.freq_data_name)
            self.shm_counts_data = sm.SharedMemory(name=self.counts_data_name)
            self.shm_meta = sm.SharedMemory(name=self.meta_name)
        except FileNotFoundError as e:
            raise RuntimeError(f"PSD shared memory buffer not found: {e}")
    
    def _setup_numpy_views(self):
        """Create numpy array views of shared memory"""
        # Channel data buffer: [channels, samples]
        self.channel_data = np.ndarray((self.max_channels, self.accumulation_samples), 
                                      dtype=np.float32, buffer=self.shm_channel_data.buf)
        
        # PSD buffer: [channels, freq_bins] - running averages
        self.psd_data = np.ndarray((self.max_channels, self.n_freq_bins), 
                                  dtype=np.float32, buffer=self.shm_psd_data.buf)
        
        # Frequency vector: [freq_bins]
        self.freq_data = np.ndarray(self.n_freq_bins, dtype=np.float32, buffer=self.shm_freq_data.buf)
        
        # Sample counts per channel: [channels]
        self.counts_data = np.ndarray(self.max_channels, dtype=np.int32, buffer=self.shm_counts_data.buf)
        
        # Metadata buffer
        self.meta_buffer = np.ndarray(self.METADATA_SIZE // 4, 
                                     dtype=np.float32, buffer=self.shm_meta.buf)
    
    def _initialize_metadata(self):
        """Initialize metadata structure"""
        with self._lock:
            metadata = PSDMetadata(
                selected_channel=0,
                channel_name="",
                sample_rate=self.fs,
                fft_length=self.fft_length,
                nfft=self.nfft,
                accumulated_samples=0,
                n_averages=0,
                auto_refresh_enabled=False,
                last_update=time.time(),
                is_valid=True
            )
            self._write_metadata(metadata)
    
    def _initialize_frequency_vector(self):
        """Initialize frequency vector"""
        with self._lock:
            self.freq_data[:] = np.linspace(0, self.fs/2, self.n_freq_bins)
    
    def _write_metadata(self, metadata: PSDMetadata):
        """Write metadata to shared memory"""
        # Pack metadata into buffer
        channel_name_bytes = metadata.channel_name.encode('utf-8')[:32]  # Limit to 32 chars
        channel_name_padded = channel_name_bytes + b'\x00' * (32 - len(channel_name_bytes))
        
        packed_data = struct.pack('i32sddiiibdb', 
                                 metadata.selected_channel,
                                 channel_name_padded,
                                 metadata.sample_rate,
                                 metadata.fft_length,
                                 metadata.nfft,
                                 metadata.accumulated_samples,
                                 metadata.n_averages,
                                 metadata.auto_refresh_enabled,
                                 metadata.last_update,
                                 metadata.is_valid)
        
        # Write to beginning of meta buffer
        packed_size = len(packed_data)
        self.meta_buffer[:packed_size//4] = np.frombuffer(packed_data, dtype=np.uint8).view(np.float32)[:packed_size//4]
    
    def _read_metadata(self) -> PSDMetadata:
        """Read metadata from shared memory"""
        # Read packed data
        packed_size = struct.calcsize('i32sddiiibdb')
        raw_bytes = self.meta_buffer[:packed_size//4].view(np.uint8).tobytes()[:packed_size]
        
        unpacked = struct.unpack('i32sddiiibdb', raw_bytes)
        
        channel_name = unpacked[1].decode('utf-8').rstrip('\x00')
        
        return PSDMetadata(
            selected_channel=unpacked[0],
            channel_name=channel_name,
            sample_rate=unpacked[2],
            fft_length=unpacked[3],
            nfft=unpacked[4],
            accumulated_samples=unpacked[5],
            n_averages=unpacked[6],
            auto_refresh_enabled=bool(unpacked[7]),
            last_update=unpacked[8],
            is_valid=bool(unpacked[9])
        )
    
    def add_channel_data(self, channel_idx: int, new_data: np.ndarray, channel_name: str = "") -> bool:
        """
        Add new data for a specific channel and update running PSD average.
        
        Args:
            channel_idx: Channel index (0-based)
            new_data: New data samples to add
            channel_name: Optional channel name
            
        Returns:
            True if successful, False otherwise
        """
        if channel_idx >= self.max_channels or len(new_data) == 0:
            return False
            
        with self._lock:
            try:
                metadata = self._read_metadata()
                
                # Get current sample count for this channel
                current_samples = self.counts_data[channel_idx]
                samples_to_add = len(new_data)
                
                # Calculate how much we can add without exceeding buffer
                max_add = min(samples_to_add, self.accumulation_samples - current_samples)
                
                if max_add > 0:
                    # Add new data to circular buffer
                    end_idx = current_samples + max_add
                    if end_idx <= self.accumulation_samples:
                        # Simple case: no wrap around
                        self.channel_data[channel_idx, current_samples:end_idx] = new_data[:max_add]
                    else:
                        # Wrap around case - shift existing data and add new
                        # For simplicity, just overwrite oldest data
                        shift_amount = end_idx - self.accumulation_samples
                        self.channel_data[channel_idx, :-shift_amount] = self.channel_data[channel_idx, shift_amount:]
                        self.channel_data[channel_idx, -shift_amount:] = new_data[:max_add]
                        current_samples = self.accumulation_samples - shift_amount
                        end_idx = self.accumulation_samples
                    
                    # Update sample count
                    self.counts_data[channel_idx] = end_idx
                    
                    # Calculate PSD if we have enough data (at least one FFT window)
                    if end_idx >= self.nfft:
                        self._update_psd_running_average(channel_idx)
                    
                    # Update metadata if this is the selected channel
                    if channel_idx == metadata.selected_channel and channel_name:
                        metadata.channel_name = channel_name
                        metadata.accumulated_samples = end_idx
                        metadata.last_update = time.time()
                        self._write_metadata(metadata)
                
                return True
                
            except Exception as e:
                print(f"Error adding channel data: {e}")
                return False
    
    def _update_psd_running_average(self, channel_idx: int):
        """Update running PSD average for a specific channel"""
        try:
            # Get the accumulated data for this channel
            n_samples = self.counts_data[channel_idx]
            if n_samples < self.nfft:
                return
            
            # Use a segment of data for PSD calculation (last nfft samples)
            segment_start = max(0, n_samples - self.nfft)
            data_segment = self.channel_data[channel_idx, segment_start:segment_start + self.nfft]
            
            # Calculate periodogram (equivalent to MATLAB's periodogram function)
            from scipy import signal
            window = signal.windows.barthann(self.nfft)
            
            # Calculate power spectral density
            f, psd = signal.periodogram(data_segment, fs=self.fs, window=window, 
                                       nfft=self.nfft, return_onesided=True)
            
            # Update running average using the formula from your MATLAB code:
            # psd = (psd + ch_data(i).psd*(N-1))*Nover_1
            # where Nover_1 = 1/N and N is the number of averages
            
            # Get current average count (stored in a way we can track per channel)
            # For simplicity, we'll use a heuristic: n_averages = accumulated_samples / nfft
            n_averages = max(1, n_samples // self.nfft)
            n_over_1 = 1.0 / n_averages
            
            # Apply running average formula
            current_psd = self.psd_data[channel_idx, :]
            if np.sum(current_psd) == 0:  # First calculation
                self.psd_data[channel_idx, :] = psd
            else:
                # Running average: new_avg = (new_val + old_avg*(N-1)) / N
                self.psd_data[channel_idx, :] = (psd + current_psd * (n_averages - 1)) * n_over_1
            
        except Exception as e:
            print(f"Error updating PSD running average: {e}")
    
    def get_psd_data(self, channel_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get PSD data for a specific channel.
        
        Args:
            channel_idx: Channel index
            
        Returns:
            Tuple of (frequency_vector, psd_values) or (None, None) if no data
        """
        if channel_idx >= self.max_channels:
            return None, None
            
        with self._lock:
            try:
                if self.counts_data[channel_idx] < self.nfft:
                    return None, None
                
                freq_copy = self.freq_data.copy()
                psd_copy = self.psd_data[channel_idx, :].copy()
                
                # Return only valid data (non-zero)
                if np.sum(psd_copy) > 0:
                    return freq_copy, psd_copy
                else:
                    return None, None
                    
            except Exception as e:
                print(f"Error getting PSD data: {e}")
                return None, None
    
    def set_auto_refresh(self, enabled: bool, selected_channel: int = 0, channel_name: str = ""):
        """Set auto-refresh state and selected channel"""
        with self._lock:
            metadata = self._read_metadata()
            metadata.auto_refresh_enabled = enabled
            metadata.selected_channel = selected_channel
            if channel_name:
                metadata.channel_name = channel_name
            metadata.last_update = time.time()
            self._write_metadata(metadata)
    
    def get_auto_refresh_state(self) -> Tuple[bool, int, str]:
        """Get auto-refresh state and selected channel info"""
        metadata = self._read_metadata()
        return metadata.auto_refresh_enabled, metadata.selected_channel, metadata.channel_name
    
    def has_enough_data_for_notches(self, min_samples: Optional[int] = None) -> bool:
        """Check if we have enough accumulated data to calculate notches"""
        if min_samples is None:
            min_samples = self.accumulation_samples // 2  # At least half the buffer
            
        metadata = self._read_metadata()
        return metadata.accumulated_samples >= min_samples
    
    def get_channel_data_for_notches(self, channel_idx: int) -> Optional[np.ndarray]:
        """Get accumulated channel data for notch calculation"""
        if channel_idx >= self.max_channels:
            return None
            
        with self._lock:
            n_samples = self.counts_data[channel_idx]
            if n_samples < self.nfft:
                return None
            
            # Return all accumulated data for this channel
            return self.channel_data[channel_idx, :n_samples].copy()
    
    def get_all_channels_data_for_notches(self) -> Dict[str, np.ndarray]:
        """Get all accumulated channel data for notch calculation"""
        result = {}
        with self._lock:
            for ch_idx in range(self.max_channels):
                n_samples = self.counts_data[ch_idx]
                if n_samples >= self.nfft:
                    # Use channel index as key if no name is available
                    channel_name = f"ch_{ch_idx:03d}"
                    result[channel_name] = self.channel_data[ch_idx, :n_samples].copy()
        return result
    
    def clear_all_data(self):
        """Clear all accumulated data"""
        with self._lock:
            self.channel_data[:] = 0
            self.psd_data[:] = 0
            self.counts_data[:] = 0
            
            metadata = self._read_metadata()
            metadata.accumulated_samples = 0
            metadata.n_averages = 0
            metadata.last_update = time.time()
            self._write_metadata(metadata)
    
    def is_valid(self) -> bool:
        """Check if buffer is valid and accessible"""
        try:
            metadata = self._read_metadata()
            return metadata.is_valid
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get buffer information"""
        metadata = self._read_metadata()
        return {
            'sample_rate': metadata.sample_rate,
            'fft_length': metadata.fft_length,
            'nfft': metadata.nfft,
            'max_channels': self.max_channels,
            'accumulation_samples': self.accumulation_samples,
            'selected_channel': metadata.selected_channel,
            'channel_name': metadata.channel_name,
            'accumulated_samples': metadata.accumulated_samples,
            'n_averages': metadata.n_averages,
            'auto_refresh_enabled': metadata.auto_refresh_enabled,
            'is_valid': metadata.is_valid,
            'last_update': metadata.last_update
        }
    
    def _cleanup_existing(self):
        """Clean up existing shared memory segments"""
        for name in [self.channel_data_name, self.psd_data_name, self.freq_data_name, 
                     self.counts_data_name, self.meta_name]:
            try:
                shm = sm.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
    
    def cleanup(self):
        """Clean up shared memory resources"""
        try:
            if hasattr(self, 'shm_channel_data'):
                self.shm_channel_data.close()
                self.shm_channel_data.unlink()
            if hasattr(self, 'shm_psd_data'):
                self.shm_psd_data.close()
                self.shm_psd_data.unlink()
            if hasattr(self, 'shm_freq_data'):
                self.shm_freq_data.close()
                self.shm_freq_data.unlink()
            if hasattr(self, 'shm_counts_data'):
                self.shm_counts_data.close()
                self.shm_counts_data.unlink()
            if hasattr(self, 'shm_meta'):
                self.shm_meta.close()
                self.shm_meta.unlink()
        except Exception as e:
            print(f"Error during PSD buffer cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't cleanup automatically - let explicit cleanup handle it
        pass


class SharedBufferManager:
    """Manager for shared buffers across multiple processes"""
    
    def __init__(self):
        self.buffers = {}
        self.psd_buffers = {}
    
    def create_buffer(self, name: str, length_seconds: float = 5.0, 
                     fs: float = 30000.0, num_channels: int = 32) -> SharedVisualizationBuffer:
        """Create a new shared buffer"""
        if name in self.buffers:
            self.buffers[name].cleanup()
        
        buffer = SharedVisualizationBuffer(
            length_seconds=length_seconds,
            fs=fs,
            num_channels=num_channels,
            create=True,
            name_prefix=name
        )
        
        self.buffers[name] = buffer
        return buffer
    
    def create_psd_buffer(self, name: str, fs: float = 30000.0, max_channels: int = 256,
                         fft_length: float = 2.0) -> SharedPSDBuffer:
        """Create a new PSD shared buffer"""
        if name in self.psd_buffers:
            self.psd_buffers[name].cleanup()
        
        buffer = SharedPSDBuffer(
            fs=fs,
            max_channels=max_channels,
            fft_length=fft_length,
            create=True,
            name_prefix=name
        )
        
        self.psd_buffers[name] = buffer
        return buffer
    
    def attach_buffer(self, name: str) -> Optional[SharedVisualizationBuffer]:
        """Attach to an existing shared buffer"""
        try:
            buffer = SharedVisualizationBuffer(create=False, name_prefix=name)
            self.buffers[name] = buffer
            return buffer
        except RuntimeError:
            return None
    
    def attach_psd_buffer(self, name: str) -> Optional[SharedPSDBuffer]:
        """Attach to an existing PSD shared buffer"""
        try:
            buffer = SharedPSDBuffer(create=False, name_prefix=name)
            self.psd_buffers[name] = buffer
            return buffer
        except RuntimeError:
            return None
    
    def get_buffer(self, name: str) -> Optional[SharedVisualizationBuffer]:
        """Get buffer by name"""
        return self.buffers.get(name)
    
    def get_psd_buffer(self, name: str) -> Optional[SharedPSDBuffer]:
        """Get PSD buffer by name"""
        return self.psd_buffers.get(name)
    
    def cleanup_all(self):
        """Clean up all managed buffers"""
        for buffer in self.buffers.values():
            buffer.cleanup()
        self.buffers.clear()
        
        for buffer in self.psd_buffers.values():
            buffer.cleanup()
        self.psd_buffers.clear()


# Global buffer manager instance
buffer_manager = SharedBufferManager()