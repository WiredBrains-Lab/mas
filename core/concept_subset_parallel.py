"""Bounded process workers for exhaustive screens; only the parent checkpoints."""
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_PAYLOAD = None
_OPTIONS = None
_LIMITER = None


def _initialize(payload, options, threads):
    global _PAYLOAD, _OPTIONS, _LIMITER
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Coordinator handles Ctrl-C.
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(threads)
    os.environ['MAS_SUBSET_WORKER_CPU'] = '1'
    from threadpoolctl import threadpool_limits
    _LIMITER = threadpool_limits(limits=threads)
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass
    _PAYLOAD, _OPTIONS = payload, options


def _fit(index, parcels, destination, note):
    from core.concept_decoding_subsets import subset_payload, _save_result, _save_inference, _write_json
    from core.concept_decoding_transformer import compute_transformer_decoding_from_payload
    folder = Path(destination)
    opts = dict(_OPTIONS, selected_parcellations=list(parcels))
    opts['inference_artifact_callback'] = lambda bundle: _save_inference(folder, bundle)
    row = dict(subset_id=index, n_parcels=len(parcels), parcels=';'.join(parcels),
               output_dir=str(folder), worker_pid=os.getpid())
    try:
        result, message = compute_transformer_decoding_from_payload(
            SimpleNamespace(), opts, subset_payload(_PAYLOAD, parcels))
        if not isinstance(result, dict):
            raise ValueError(message or 'Decoder returned no result.')
        result.update(subset_screening_note=note, selected_parcellations=list(parcels))
        _save_result(folder, result, opts)
        row.update(status='complete', **{key: result.get(key) for key in (
            'mpr', 'top1_accuracy', 'top5_accuracy', 'n_train', 'n_validation',
            'n_test', 'n_features', 'n_source_trials', 'validation_top1_accuracy', 'validation_mpr')})
    except Exception as exc:
        row.update(status='failed', error=str(exc))
        _write_json(folder / 'error.json', row)
    return row


def parallel_rows(jobs, folder, payload, options, workers, threads, note, cancelled, progress):
    """Yield finished rows immediately, keeping at most workers jobs in flight.

    Cancellation drains submitted work; hard termination leaves unrecorded fits
    pending for the next resume. No worker writes the shared checkpoint.
    """
    jobs = iter(jobs)
    clean = {key: value for key, value in options.items() if not callable(value)}
    stopped = False
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize, initargs=(payload, clean, threads)) as executor:
        active = {}
        exhausted = False
        while active or not exhausted:
            try:
                if cancelled():
                    stopped = True
                while not stopped and not exhausted and len(active) < workers:
                    try:
                        index, parcels = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    # Subset ID is stable across worker counts and completion order.
                    destination = Path(folder) / f'{index:03d}_n{len(parcels)}_parallel_{datetime.now():%Y%m%d_%H%M%S_%f}'
                    destination.mkdir()
                    future = executor.submit(_fit, index, parcels, str(destination), note)
                    active[future] = index
                    progress(f'Subset {index}: started ({len(parcels)} parcels; {len(active)}/{workers} workers active)')
                if not active:
                    break
                finished, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = active.pop(future)
                    # A crashed worker is not a scientifically failed fit. Stop
                    # and leave that subset pending rather than recording failure.
                    row = future.result()
                    yield row
                if stopped and not active:
                    break
            except KeyboardInterrupt:
                if stopped:
                    raise
                stopped = True
                progress('Stopping submission; saving active subsets as they finish. Interrupt again to exit.')
