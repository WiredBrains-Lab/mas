"""Content verification and relocation of trusted source-feature cache payloads."""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd


def _stat_key(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=8192)
def _file_sha256(path, stat_key):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if _stat_key(path) != stat_key:
        raise ValueError(f'Input changed while verifying its contents: {path}')
    return digest.hexdigest()


def file_content_identity(path):
    path = str(Path(path).expanduser().resolve())
    stat = _stat_key(path)
    # Stat information only avoids repeated reads within this process. It is
    # never part of the portable identity; new jobs verify actual file bytes.
    return {'size': stat[2], 'sha256': _file_sha256(path, stat)}


def translate_paths(value, bindings, *, restore=False):
    """Translate known absolute paths to stable role tokens, or back to this host."""
    prefix = 'mas-cache://'
    pairs = [(prefix + role, path) if restore else (str(Path(path).resolve()), prefix + role)
             for role, path in bindings.items()]
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    def string(text):
        if not restore and os.path.isabs(text):
            text = str(Path(text).resolve())
        for old, new in pairs:
            if text == old or text.startswith(old + '/'):
                return new + text[len(old):]
        if restore and text.startswith(prefix):
            raise ValueError(f'Cache path has no current input binding: {text}')
        return text

    def walk(item):
        if isinstance(item, str):
            return string(item)
        if isinstance(item, pd.DataFrame):
            result = item.copy()
            for column in result.columns:
                if pd.api.types.is_object_dtype(result[column].dtype) or isinstance(result[column].dtype, pd.StringDtype):
                    result[column] = result[column].map(walk)
            result.attrs = walk(result.attrs)
            return result
        if isinstance(item, pd.Series):
            return item.map(walk)
        if isinstance(item, dict):
            return {string(key) if isinstance(key, str) else key: walk(val) for key, val in item.items()}
        if isinstance(item, list):
            return [walk(val) for val in item]
        if isinstance(item, tuple):
            return tuple(walk(val) for val in item)
        if isinstance(item, np.ndarray) and item.dtype.kind in {'O', 'U', 'S'}:
            return np.array([walk(val) for val in item.flat], dtype=object).reshape(item.shape)
        return item

    return walk(value)


def loaded_raw_identity(raw, fif):
    """Pristine GUI raw matches the disk/CLI identity; retain unsaved edits."""
    if raw is None:
        return None
    with mne.io.read_raw_fif(fif, preload=False, verbose='error') as disk:
        state = (raw.info, raw.annotations, raw.first_samp, raw.n_times)
        pristine = joblib.hash(state) == joblib.hash((disk.info, disk.annotations, disk.first_samp, disk.n_times))
        digest = hashlib.sha256()
        # Bound peak RAM when comparing a preloaded recording with its saved FIF.
        step = max(1, (16 * 1024 * 1024) // (8 * max(1, len(raw.ch_names))))
        for start in range(0, raw.n_times, step):
            stop = min(start + step, raw.n_times)
            data = raw.get_data(start=start, stop=stop)
            digest.update(np.asarray(data, dtype='<f8').tobytes())
            if pristine:
                pristine = np.array_equal(data, disk.get_data(start=start, stop=stop), equal_nan=True)
        if pristine:
            return None
        # Edited GUI data cannot match an unedited CLI run. No Raw object or
        # filename is serialized into this identity.
        return joblib.hash((state, digest.hexdigest()), hash_name='sha1')
