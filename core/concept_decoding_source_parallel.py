"""Isolated per-run source extraction and atomic restart checkpoints."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import joblib
import mne
from threadpoolctl import threadpool_limits

from core.concept_decoding_signal import emit_concept_progress


def _worker_progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _source_run_worker(run, validated, file_df, options, settings):
    """Only file mappings and trial metadata cross the process boundary."""
    from core.concept_decoding_headless_source import HeadlessSourceContext
    from core.concept_decoding_source import _extract_source_run

    context = SimpleNamespace(_fif_filename=run["fif"])
    viewer = HeadlessSourceContext([run], settings["source_space_kind"])
    viewer._validated = set(validated)
    viewer.attach(context)
    options = dict(options, progress_callback=_worker_progress)
    threads = int(options.get("source_worker_threads", 1))
    with threadpool_limits(limits=threads):
        with mne.io.read_raw_fif(run["fif"], preload=False, verbose="error") as raw:
            return _extract_source_run(context, options, raw, run["fif"], file_df, **settings)


def _checkpoint_path(options, path, file_df):
    directory = options.get("_source_run_checkpoint_dir")
    if not directory:
        return None
    # The enclosing directory identifies all input contents, extraction options,
    # and implementation. Include absolute paths/global row IDs: intermediate
    # checkpoints are host-local, unlike the final portable feature cache.
    digest = hashlib.sha256((path + file_df.to_json(
        orient="split", date_format="iso", double_precision=15,
        default_handler=str)).encode()).hexdigest()
    return Path(directory) / f"{digest}.joblib"


def _load_checkpoint(path, options):
    if path is None or options.get("rebuild_source_features", False) or not path.is_file():
        return None
    try:
        result = joblib.load(path)
        if (not result["skipped_paths"] and result["feature_rows"]
                and len(result["feature_rows"]) == len(result["kept_rows"])
                and all(len(row) == len(result["feature_names"]) for row in result["feature_rows"])):
            return result
    except Exception as exc:
        emit_concept_progress(options, f"Source run checkpoint unreadable; re-extracting: {exc}")
    return None


def _save_checkpoint(path, result, options):
    if path is None or result["skipped_paths"] or not result["feature_rows"]:
        return
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        joblib.dump(result, temporary, compress=3)
        os.replace(temporary, path)
    except Exception as exc:
        emit_concept_progress(options, f"Could not save source run checkpoint: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_source_runs(context, options, raw_by_path, grouped_files, settings):
    """Yield results in input order, checkpointing each as soon as it completes.

    Global class/ROI selection and sampling validation happen in the caller.
    Workers use private headless contexts; GUI live raw state stays serial.
    """
    from core.concept_decoding_headless_source import HeadlessSourceContext
    from core.concept_decoding_source import _extract_source_run, _meg_3d_viewer

    workers = min(int(options.get("source_workers", 1)), len(grouped_files))
    threads = int(options.get("source_worker_threads", 1))
    if workers < 1 or threads < 1:
        raise ValueError("Source workers and worker threads must be positive.")
    viewer = _meg_3d_viewer(context)
    if workers > 1 and (not isinstance(viewer, HeadlessSourceContext)
                        or getattr(context, "_fif_raw", None) is not None):
        raise ValueError("Parallel source extraction requires a headless CLI context with saved FIF inputs.")
    results, pending = {}, []
    for index, (path, file_df) in enumerate(grouped_files):
        checkpoint = _checkpoint_path(options, path, file_df)
        result = _load_checkpoint(checkpoint, options)
        if result is not None:
            emit_concept_progress(options, f"Reusing source run checkpoint {index + 1}/{len(grouped_files)}: {os.path.basename(path)}")
            results[path] = result
        else:
            pending.append((path, file_df, checkpoint))
    if workers == 1:
        for path, file_df, checkpoint in pending:
            started = time.monotonic()
            result = _extract_source_run(context, options, raw_by_path[path], path, file_df, **settings)
            _save_checkpoint(checkpoint, result, options)
            results[path] = result
            emit_concept_progress(options, f"Completed source run: {os.path.basename(path)} ({time.monotonic() - started:.1f}s)")
    elif pending:
        workers = min(workers, len(pending))
        emit_concept_progress(options, f"Extracting {len(pending)} source runs with {workers} processes, {threads} numerical threads each")
        # CLI callbacks are not necessarily picklable; workers emit flushed logs.
        worker_options = {key: value for key, value in options.items() if not callable(value)}
        worker_options["_defer_recording_cache_prune"] = True
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {}
            for path, file_df, checkpoint in pending:
                future = executor.submit(_source_run_worker, viewer.runs[path], viewer._validated,
                                         file_df, worker_options, settings)
                futures[future] = (path, checkpoint, time.monotonic())
            for future in as_completed(futures):
                path, checkpoint, started = futures[future]
                result = future.result()
                _save_checkpoint(checkpoint, result, options)
                results[path] = result
                emit_concept_progress(options, f"Completed source run {len(results)}/{len(grouped_files)}: {os.path.basename(path)} ({time.monotonic() - started:.1f}s since submission)")
        if options.get("enable_signal_epoch_cache", True):
            from core.concept_decoding_signal import (
                _concept_epoch_disk_cache_path, _prune_concept_epoch_disk_cache,
            )
            cache_path = _concept_epoch_disk_cache_path(options, ("source_worker_cleanup",))
            if cache_path:
                _prune_concept_epoch_disk_cache(Path(cache_path).parent)
    for path, _ in grouped_files:
        yield path, results[path]
