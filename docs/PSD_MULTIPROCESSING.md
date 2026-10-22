# PSD Multiprocessing Implementation

This document describes the new multiprocessing implementation for PSD (Power Spectral Density) calculation and notch detection in the MEG Analysis Suite (MAS).

## Overview

The `psd_proc.py` module provides a multiprocessing-based alternative to the original multithreading approach for calculating PSD and detecting notches across multiple channels simultaneously. This implementation offers better performance for CPU-intensive tasks by utilizing true parallel processing across multiple CPU cores.

## Key Features

### 1. True Parallel Processing
- Uses separate processes instead of threads
- Bypasses Python's Global Interpreter Lock (GIL)
- Better CPU utilization for compute-intensive tasks
- Scales linearly with the number of CPU cores

### 2. Automatic Load Balancing
- Uses `ProcessPoolExecutor` for efficient task distribution
- Automatically handles process creation and cleanup
- Dynamic progress monitoring and reporting

### 3. Memory Efficiency
- Each process operates independently with its own memory space
- Automatic cleanup of completed processes
- Configurable number of worker processes

### 4. Error Handling
- Robust error handling for individual channel processing
- Failed channels don't affect other channels
- Detailed error reporting and logging

## Architecture

### Main Components

1. **MultiprocessingPSDCalculator**: Main coordinator class
2. **process_single_channel_psd_and_notches()**: Worker function for individual channels
3. **Helper functions**: Notch detection and filter creation functions

### Processing Pipeline

```
Input Data (ch_data) 
    ↓
Parameter Setup
    ↓
ProcessPoolExecutor with N workers
    ↓
Parallel Processing:
├── Worker 1: Channel 1 → PSD + Notches
├── Worker 2: Channel 2 → PSD + Notches  
├── Worker 3: Channel 3 → PSD + Notches
└── Worker N: Channel N → PSD + Notches
    ↓
Results Aggregation
    ↓
File Output (PSD data, notches, filters)
```

## Usage

### From Power Spectrum Interface

1. **Select Processing Mode**: Choose "Multiprocessing" from the Mode dropdown
2. **Click Calculate Notches**: The system will automatically use multiprocessing
3. **Monitor Progress**: A progress dialog shows real-time processing status
4. **View Results**: Results are displayed in the UI and saved to files

### Programmatic Usage

```python
from dsp.psd_proc import MultiprocessingPSDCalculator

# Create calculator with 4 worker processes
calculator = MultiprocessingPSDCalculator(max_workers=4)

# Setup processing parameters
processing_params = {
    'fs': 30000,
    'barthannwin': signal.windows.barthann(nfft),
    'freq_min': 300,
    'freq_max': 3000,
    'psd_1_x': 10000,
    'span_smooth': 21,
    'db_thr': 5,
    'filtering_params': {
        'lowcut': 1.0,
        'highcut': 14999,
        'order': 4
    }
}

# Process all channels
psd_data, notches_data, filters = calculator.calculate_all_channel_psd_and_notches(
    ch_data, processing_params, progress_callback
)

# Save results
calculator.save_results_to_files(psd_data, notches_data, filters, fs)
```

## Performance Comparison

### Multiprocessing vs Multithreading

| Aspect | Multiprocessing | Multithreading |
|--------|-----------------|----------------|
| **CPU Utilization** | True parallel, uses all cores | Limited by GIL, ~1.2x speedup |
| **Memory Usage** | Higher (separate processes) | Lower (shared memory) |
| **Scalability** | Linear with CPU cores | Limited by GIL |
| **Stability** | Isolated processes | Shared memory risks |
| **Overhead** | Higher startup cost | Lower overhead |

### Expected Performance Gains

- **2-4 cores**: 1.5-3x speedup
- **4-8 cores**: 2-6x speedup  
- **8+ cores**: 4-12x speedup

*Actual performance depends on data size, system configuration, and available memory.*

## Configuration

### Optimal Worker Count

```python
import multiprocessing as mp

# Conservative approach (recommended)
optimal_workers = max(1, mp.cpu_count() - 1)

# Aggressive approach (for dedicated processing)
optimal_workers = mp.cpu_count()

# Memory-constrained systems
optimal_workers = min(4, max(1, mp.cpu_count() - 1))
```

### Memory Considerations

- Each worker process requires ~200-500 MB RAM
- For 32 channels @ 30kHz, ~2 seconds data: ~8-16 MB per channel
- Recommended: 2+ GB available RAM for 4-8 workers

### Windows Compatibility

The implementation uses `spawn` start method on Windows for compatibility:
```python
if sys.platform == 'win32':
    mp.set_start_method('spawn', force=True)
```

## File Output

### Generated Files

1. **PSD Data**: `psd_data_mp_TIMESTAMP.npy`
   - Contains PSD data for all channels
   - Includes processing parameters and metadata

2. **Notch Filters**: `notch_filters_mp_TIMESTAMP.npy`  
   - Filter coefficients for detected notches
   - Ready for use in filtering operations

3. **Summary Report**: `notches_summary_mp_TIMESTAMP.txt`
   - Human-readable summary of detected notches
   - Processing statistics and parameters

### File Locations

```
mas/
├── assets/
│   ├── psd_data/           # PSD calculation results
│   └── notch_filters/      # Filter coefficients and summaries
```

## Testing

### Run Performance Test

```bash
cd "mas"
python test_psd_multiprocessing.py
```

This will:
1. Generate synthetic test data
2. Compare multiprocessing vs single-threading
3. Validate notch detection accuracy
4. Report performance metrics

### Expected Test Output

```
Multiprocessing completed in 2.34 seconds
Single-threading completed in 8.76 seconds  
Speedup factor: 3.74x
✓ Multiprocessing shows significant improvement!
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `psd_proc.py` is in the `dsp/` directory
2. **Slow Performance**: Check available CPU cores and memory
3. **Memory Errors**: Reduce number of workers or data size
4. **Windows Spawn Issues**: Restart Python session if changing start method

### Debug Mode

Enable debug output in `psd_proc.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor CPU and memory usage during processing:
```python
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## Migration Guide

### From Multithreading

1. **No Code Changes Required**: Existing interface remains the same
2. **Select Mode**: Choose "Multiprocessing" in the UI dropdown
3. **Verify Performance**: Monitor processing time improvements
4. **Adjust Workers**: Tune worker count based on system resources

### Parameter Compatibility

All existing parameters are fully supported:
- `fs`: Sampling frequency
- `freq_min/freq_max`: Notch detection range  
- `db_thr`: Detection threshold
- `span_smooth`: Smoothing window size
- All filtering parameters

## Future Enhancements

### Planned Features

1. **Adaptive Worker Count**: Automatically adjust based on system load
2. **Memory Monitoring**: Dynamic worker reduction if memory is low
3. **GPU Acceleration**: CUDA support for compatible systems
4. **Distributed Computing**: Support for cluster-based processing

### Optimization Opportunities

1. **Data Chunking**: Process very large datasets in chunks
2. **Caching**: Cache FFT results for repeated calculations
3. **Compression**: Compress intermediate results to save memory
4. **Streaming**: Process data as it's acquired

## References

- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [NumPy Performance Guidelines](https://numpy.org/doc/stable/user/performance.html)