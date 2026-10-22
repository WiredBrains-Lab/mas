"""Exploratory, matched-data selected-parcel screening."""

import copy
import json
import os
import hashlib
import re
import time
from collections import deque
from datetime import datetime
from itertools import combinations
from math import comb
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SEMANTIC_APARC_POOL = tuple(f"ctx-lh-{name}" for name in (
    "temporalpole", "middletemporal", "inferiorparietal",
    "parstriangularis", "parsorbitalis", "bankssts",
))
SCREENING_NOTE = (
    "Exploratory subset screen: the same deterministic train/test split is reused "
    "across every parcel subset. "
    "Selecting the best test score across subsets is optimistically biased; "
    "confirm on independent, group-aware/nested evaluation. Parcel exclusion "
    "alone does not establish semantic rather than visual decoding."
)
BACKWARD_NOTE = (
    "Greedy backward elimination: removals maximize validation top-1 accuracy, "
    "then validation MPR, with deterministic ties. Test scores never choose removals. "
    "Validation is reused for early stopping and parcel selection, so these scores "
    "are selection-biased. This is not nested/group-aware evaluation and does not "
    "guarantee an optimal subset. Do not select footprint size by test performance."
)


def _format_duration(seconds):
    """Format an ETA compactly without relying on wall-clock timestamps."""
    total_seconds = max(0, int(round(float(seconds))))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def backward_fit_count(parcels):
    n = len(selected_pool(parcels))
    return n * (n + 1) // 2


def choose_backward_candidate(rows):
    """Use validation only; incomplete stages must not silently choose a winner."""
    if not rows or any(r.get("status") != "complete" for r in rows):
        raise ValueError("Backward elimination stopped: a candidate fit failed; cannot compare the complete removal stage.")
    for row in rows:
        if not all(np.isfinite(float(row.get(k, np.nan))) for k in ("validation_top1_accuracy", "validation_mpr")):
            raise ValueError("Backward elimination requires finite validation scores for every candidate.")
    return sorted(rows, key=lambda r: (-r["validation_top1_accuracy"], -r["validation_mpr"], r["parcels"]))[0]


def selected_pool(parcels):
    pool = tuple(dict.fromkeys(str(p).strip() for p in (parcels or []) if str(p).strip()))
    if not pool:
        raise ValueError("Select at least one parcel explicitly for subset screening.")
    return pool


def exhaustive_fit_count(parcels):
    return (1 << len(selected_pool(parcels))) - 1


class ExhaustiveSubsets:
    """Lazy stable ordering, with direct indexing for checkpoint validation.

    Use total for counts exceeding Python's platform-sized len() limit.
    """
    def __init__(self, parcels):
        self.pool = selected_pool(parcels)
        self.total = exhaustive_fit_count(self.pool)

    def __len__(self):
        return self.total

    def __iter__(self):
        yield self.pool
        for size in range(1, len(self.pool)):
            yield from combinations(self.pool, size)

    def __getitem__(self, index):
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        if index == 0:
            return self.pool
        rank = index - 1
        n = len(self.pool)
        for size in range(1, n):
            count = comb(n, size)
            if rank < count:
                break
            rank -= count
        chosen, start = [], 0
        for remaining in range(size - 1, -1, -1):
            for candidate in range(start, n - remaining):
                count = comb(n - candidate - 1, remaining)
                if rank < count:
                    chosen.append(self.pool[candidate])
                    start = candidate + 1
                    break
                rank -= count
        return tuple(chosen)


def exhaustive_subsets(parcels):
    return ExhaustiveSubsets(parcels)


def semantic_subsets():
    """Compatibility helper for the original six-parcel experiment."""
    return exhaustive_subsets(SEMANTIC_APARC_POOL)


def subset_payload(payload, parcels):
    """Slice columns by metadata, never assume ROI-major feature ordering."""
    meta = payload["feature_meta_df"]
    X = np.asarray(payload["X"])
    if X.ndim != 2 or len(meta) != X.shape[1] or "roi" not in meta:
        raise ValueError("Source feature matrix and ROI metadata do not align.")
    missing = set(parcels) - set(meta["roi"].astype(str))
    if missing:
        raise ValueError(f"Missing selected parcels: {sorted(missing)}")
    mask = meta["roi"].astype(str).isin(parcels).to_numpy()
    selected = meta.loc[mask].copy().reset_index(drop=True)
    selected["feature_index"] = np.arange(len(selected))
    result = dict(payload)
    result.update(X=X[:, mask], X_tensor=X[:, mask], feature_meta_df=selected,
                  feature_names=selected["feature"].astype(str).tolist(),
                  roi_names=list(parcels))
    return result


def _json_value(value):
    if isinstance(value, pd.DataFrame):
        return {"type": "dataframe", "rows": len(value), "columns": list(value.columns)}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _save_result(folder, result, options):
    tables = folder / "tables"
    tables.mkdir(exist_ok=True)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            name = "features" if key == "feature_df" else key.removesuffix("_df")
            value.to_csv(tables / f"{name}.csv", index=False)
        elif isinstance(value, np.ndarray):
            np.save(folder / f"{key}.npy", value, allow_pickle=False)
    _write_json(folder / "settings_used.json", options)
    _write_json(folder / "result_summary.json", result)
    joblib.dump(result, folder / "result.joblib", compress=3)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    cm = np.asarray(result.get("confusion_matrix", []))
    if cm.ndim == 2 and cm.size:
        fig = Figure(figsize=(7, 6), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        fig.colorbar(ax.imshow(cm, cmap="Blues", aspect="auto"), ax=ax)
        names = result.get("class_names", [])
        if len(names) == cm.shape[0]:
            ax.set_yticks(range(len(names)), names, fontsize=5)
            ax.set_xticks(range(len(names)), names, rotation=90, fontsize=5)
        ax.set(xlabel="Predicted concept", ylabel="True concept", title="Held-out confusion matrix")
        fig.savefig(folder / "confusion_matrix.png", dpi=150)


def _save_inference(folder, bundle):
    saved = dict(bundle)
    model = copy.deepcopy(saved["model"])
    if hasattr(model, "cpu"):
        model = model.cpu()
    saved["model"] = model
    saved["train_meta"] = dict(saved["train_meta"], device="cpu")
    joblib.dump(saved, folder / "inference_bundle.joblib", compress=3)


def _comparison_plot(folder, rows, pool_size):
    # No pyplot/Qt state: safe to render in the decoder worker.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    frame = pd.DataFrame([r for r in rows if r["status"] == "complete"])
    if frame.empty:
        return
    fig = Figure(figsize=(8, 4), layout="constrained")
    FigureCanvasAgg(fig)
    for ax, metric, label in zip(fig.subplots(1, 2),
                                  ("top1_accuracy", "mpr"), ("Top-1 accuracy", "MPR")):
        ax.scatter(frame["n_parcels"], frame[metric], alpha=0.6)
        reference = frame.loc[frame["n_parcels"] == pool_size, metric]
        if len(reference):
            ax.axhline(reference.iloc[0], color="gray", linestyle="--", label="Full pool")
            ax.legend()
        ax.set(xlabel="Number of parcels", ylabel=label)
        if pool_size <= 10:
            ax.set_xticks(range(1, pool_size + 1))
    fig.suptitle("Exploratory subset screen — shared test split, no uncertainty estimate")
    fig.savefig(folder / "performance_vs_parcel_count.png", dpi=150)


def _elimination_plot(folder, rows):
    if not rows:
        return
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    frame = pd.DataFrame(rows)
    fig = Figure(figsize=(8, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(frame.n_parcels, frame.validation_top1_accuracy, "o-", label="Validation (used for selection)")
    ax.plot(frame.n_parcels, frame.top1_accuracy, "o--", label="Test (not used for selection)")
    ax.set(xlabel="Number of retained parcels", ylabel="Top-1 accuracy",
           title="Greedy elimination path — shared split, no uncertainty estimate")
    ax.legend()
    fig.savefig(folder / "elimination_path.png", dpi=150)


def _log_screening_trial_counts(payload, progress, *, saved=False):
    """Report retained individual trials, before splitting or pseudo-epoch averaging."""
    trials = payload["trial_df"]
    count = len(trials)
    concepts = len(pd.unique(np.asarray(payload["y"])))
    runs = trials["source_fif_path"].nunique() if "source_fif_path" in trials else None
    run_label = f", {runs:,} runs" if runs is not None else ""
    origin = "loaded from saved checkpoint" if saved else "features ready"
    progress(f"Parcel batch trial summary ({origin}): {count:,} individual trials retained "
             f"after QC and eligibility filtering; {concepts:,} concepts{run_label}. "
             "Shared across all ROI subsets, before train/validation/test splitting and trial averaging.")


class _ScreeningBests:
    """Independent finite metric leaders, rebuilt from rows when resuming."""

    def __init__(self, rows=()):
        self.leaders = {}
        for row in rows:
            self.update(row)

    def update(self, row):
        if row.get("status") != "complete":
            return
        for metric in ("top1_accuracy", "mpr"):
            try:
                value = float(row.get(metric))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            previous = self.leaders.get(metric)
            subset_id = int(row["subset_id"])
            if previous is None or value > previous["value"] or (
                value == previous["value"] and subset_id < previous["subset_id"]
            ):
                self.leaders[metric] = dict(value=value, subset_id=subset_id,
                                           n_parcels=int(row["n_parcels"]))

    def label(self):
        labels = []
        for metric, name in (("top1_accuracy", "top-1"), ("mpr", "MPR")):
            leader = self.leaders.get(metric)
            if leader is None:
                labels.append(f"best {name} unavailable")
                continue
            value = leader["value"]
            score = f"{100 * value:.2f}%" if metric == "top1_accuracy" else f"{value:.4f}"
            labels.append(f"best {name} {score} [subset {leader['subset_id']}, "
                          f"{leader['n_parcels']} parcels]")
        return "; ".join(labels)


def _run_semantic_subset_batch(context, options):
    from core.concept_decoding_source import build_source_space_concept_features
    from core.concept_decoding_transformer import (
        prepare_semantic_embedding_feature_payload,
        compute_transformer_decoding_from_payload,
    )

    progress = options.get("progress_callback", lambda _message: None)
    log = options.get("log_callback", lambda _message: None)
    cancelled = options.get("cancel_callback", lambda: False)
    resume_value = str(options.get("semantic_subset_resume_dir", "") or "").strip()
    resuming = bool(resume_value)

    if resuming:
        folder = Path(resume_value).expanduser().resolve()
        progress(f"Validating screening checkpoint: {folder}")
        manifest_path = folder / "manifest.json"
        settings_path = folder / "settings_used.json"
        feature_path = folder / "source_features.joblib"
        comparison_path = folder / "subset_comparison.csv"
        missing = [str(path.name) for path in (manifest_path, settings_path, feature_path) if not path.is_file()]
        if missing:
            raise ValueError(f"Cannot resume exhaustive screening; missing checkpoint file(s): {', '.join(missing)}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("search_strategy") != "exhaustive":
            raise ValueError("Only exhaustive subset screening can be resumed with this command.")
        opts = json.loads(settings_path.read_text(encoding="utf-8"))
        pool = selected_pool(manifest.get("candidate_pool", opts.get("selected_parcellations")))
        all_subsets = exhaustive_subsets(pool)
        total = all_subsets.total
        if int(manifest.get("total_subsets", total)) != total:
            raise ValueError("Resume checkpoint total does not match the saved candidate parcel pool.")
        if not comparison_path.is_file() and int(manifest.get("attempted", 0)) > 0:
            raise ValueError("Resume checkpoint is missing subset_comparison.csv despite recorded progress; copy the complete batch folder.")
        rows = pd.read_csv(comparison_path, dtype={"subset_id": str}).to_dict("records") if comparison_path.is_file() else []
        if len(rows) > total:
            raise ValueError("Resume checkpoint contains more rows than the exhaustive subset search.")
        attempted_ids = set()
        for row in rows:
            raw_index = row.get("subset_id", -1)
            expected_index = int(raw_index)
            if str(expected_index) != str(raw_index) or not 1 <= expected_index <= total or expected_index in attempted_ids:
                raise ValueError(f"Invalid or duplicate checkpoint subset ID: {raw_index}")
            attempted_ids.add(expected_index)
            expected = all_subsets[expected_index - 1]
            observed = tuple(str(row.get("parcels", "")).split(";"))
            if int(row.get("subset_id", -1)) != expected_index or observed != expected:
                raise ValueError(
                    f"Resume checkpoint does not match the exhaustive search at subset {expected_index}."
                )
            if row.get("status") not in {"complete", "failed"}:
                raise ValueError(f"Resume checkpoint has invalid status at subset {expected_index}: {row.get('status')}")
        rows.sort(key=lambda row: int(row["subset_id"]))
        pending = ((index, parcels) for index, parcels in enumerate(all_subsets, 1) if index not in attempted_ids)
        opts.update(feature_space="source", selected_parcellations=list(pool), max_rois=0,
                    semantic_subset_batch=False, save_full_feature_table=True)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["output_dir"] = str(folder)
    else:
        opts = dict(options)
        pool = selected_pool(options.get("selected_parcellations"))
        strategy = options.get("subset_search_strategy", "exhaustive")
        if strategy == "backward" and float(
            options.get("mlp_validation_fraction", options.get("validation_fraction", 0.2))
        ) <= 0:
            raise ValueError("Backward elimination requires validation fraction greater than zero.")
        pending = enumerate(exhaustive_subsets(pool), 1) if strategy == "exhaustive" else deque([pool])
        total = backward_fit_count(pool) if strategy == "backward" else exhaustive_fit_count(pool)
        base = Path(options["semantic_subset_output_dir"]).expanduser().resolve()
        suffix = "_backward_elimination" if strategy == "backward" else "_selected_parcel_subsets"
        folder = base / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix)
        folder.mkdir(parents=True, exist_ok=False)
        rows = []

    if opts.get("decoder_type") not in {"embedding_ridge", "embedding_contrastive"}:
        raise ValueError("Semantic subset batch requires a linear ridge or contrastive embedding decoder.")
    split_mode = str(opts.get("split_mode", "random") or "random").strip().lower()
    if split_mode not in {"random", "zero_shot"}:
        raise ValueError(
            "Semantic subset screening supports per-concept random and zero-shot "
            "unseen-concept splits; leave-one-run/session splits are not supported."
        )
    backward = opts.get("subset_search_strategy", "exhaustive") == "backward"
    if opts.get("subset_search_strategy", "exhaustive") not in {"exhaustive", "backward"}:
        raise ValueError("Unknown parcel subset search strategy.")
    if resuming and backward:
        raise ValueError("Backward elimination checkpoints cannot be resumed yet.")
    if backward and float(opts.get("mlp_validation_fraction", opts.get("validation_fraction", 0.2))) <= 0:
        raise ValueError("Backward elimination requires validation fraction greater than zero.")
    from core.concept_decoding_run_summary import log_decoding_run_summary
    log_decoding_run_summary(opts, progress, saved=resuming)
    workers = int(options.get("subset_workers", 1))
    worker_threads = int(options.get("subset_worker_threads", 1))
    if workers < 1 or worker_threads < 1:
        raise ValueError("Subset worker and thread counts must be positive.")
    if backward and workers > 1:
        raise ValueError("Parallel workers currently support exhaustive screening only.")
    progress(f"WARNING: {len(pool)} selected parcels; {total:,} subset models in total; "
             f"{total - len(rows):,} remaining. Large searches can take substantial time and disk space.")
    note = BACKWARD_NOTE if backward else SCREENING_NOTE
    opts.update(feature_space="source", selected_parcellations=list(pool),
                max_rois=0, semantic_subset_batch=False, save_full_feature_table=True)
    if backward:
        opts["require_subset_validation"] = True
    last_result = None
    path_rows, stage_rows = [], []
    current_pool = pool
    complete_rows = [row for row in rows if row.get("status") == "complete"]
    best_top1 = max(
        (float(row.get("top1_accuracy", np.nan)) for row in complete_rows
         if np.isfinite(float(row.get("top1_accuracy", np.nan)))),
        default=-np.inf,
    )
    if not resuming:
        manifest = {"status": "extracting", "candidate_pool": pool,
                    "total_subsets": total, "output_dir": str(folder), "note": note,
                    "search_strategy": "backward" if backward else "exhaustive",
                    "split_mode": split_mode,
                    "selection_metric": "validation_top1_accuracy_then_validation_mpr" if backward else None}

    metric_bests = _ScreeningBests(complete_rows)

    def checkpoint(status):
        manifest["metric_bests"] = metric_bests.leaders
        manifest.update(status=status, attempted=len(rows),
                        completed=sum(r["status"] == "complete" for r in rows),
                        failed=sum(r["status"] == "failed" for r in rows))
        rows.sort(key=lambda row: int(row["subset_id"]))
        if rows:
            temporary = folder / "subset_comparison.csv.tmp"
            pd.DataFrame(rows).to_csv(temporary, index=False)
            os.replace(temporary, folder / "subset_comparison.csv")
        _write_json(folder / "manifest.json", manifest)
        if path_rows:
            temporary = folder / "elimination_path.csv.tmp"
            pd.DataFrame(path_rows).to_csv(temporary, index=False)
            os.replace(temporary, folder / "elimination_path.csv")

    resume_payload = None
    if resuming:
        # Resolve external dependencies before writing progress or fitting any
        # subsets. Keep archived settings/features unchanged for provenance.
        progress("Loading source_features.joblib...")
        resume_payload = joblib.load(folder / "source_features.joblib")
        progress("Saved source features loaded.")
        resolver = options.get("resume_path_resolver")
        if resolver is not None:
            progress("Checking and remapping saved dependency paths...")
            opts = resolver.transform(opts)
            resume_payload = resolver.transform(resume_payload)
            rows = resolver.transform(rows)
            complete_rows = [row for row in rows if row.get("status") == "complete"]
        progress("Checking saved parcel features...")
        subset_payload(resume_payload, pool)
        _log_screening_trial_counts(resume_payload, progress, saved=True)
        progress("Resume preflight complete.")
        if options.get("resume_dry_run"):
            return None, f"Resume validated: {len(rows)}/{total} attempted; {total - len(rows)} remaining. Outputs: {folder}"

    checkpoint("resuming" if resuming else "extracting")
    if not resuming:
        _write_json(folder / "settings_used.json", opts)
        (folder / "README.txt").write_text(note + "\nStop takes effect between subset fits.\n", encoding="utf-8")
    try:
        if resuming:
            progress(f"Parcel batch: resuming {len(rows)}/{total} attempted fits from {folder}")
            payload = resume_payload
            subset_payload(payload, pool)
            for row in reversed(complete_rows):
                result_path = Path(str(row.get("output_dir", ""))) / "result.joblib"
                if result_path.is_file():
                    last_result = joblib.load(result_path)
                    break
        else:
            progress(f"Parcel batch: extracting {len(pool)} selected parcels once; saving to {folder}")
            payload, error = build_source_space_concept_features(context, opts)
            if payload is None:
                raise ValueError(error or "Source extraction failed.")
            # Fail rather than silently run a smaller pool if labels are unavailable.
            subset_payload(payload, pool)
            payload, error = prepare_semantic_embedding_feature_payload(payload, opts)
            if payload is None:
                raise ValueError(error or "Embedding alignment failed.")
            _log_screening_trial_counts(payload, progress)
            joblib.dump(payload, folder / "source_features.joblib", compress=3)
        fit_batch_started = time.monotonic()
        session_start_attempted = len(rows)
        session_start_completed = sum(row["status"] == "complete" for row in rows)

        def completion_rate_label():
            # Session-only throughput across all workers. Historical completed
            # fits and failed fits must not inflate the reported rate.
            elapsed = max(time.monotonic() - fit_batch_started, 1e-9)
            completed = manifest["completed"] - session_start_completed
            return f"{60.0 * completed / elapsed:.2f} completed subsets/min (session avg)"

        if workers > 1:
            from core.concept_subset_parallel import parallel_rows
            manifest.update(subset_workers=workers, subset_worker_threads=worker_threads,
                            checkpoint_format_version=2)
            progress(f"Running up to {workers} CPU subset workers, {worker_threads} threads each.")
            jobs = pending
            for row in parallel_rows(jobs, folder, payload, opts, workers, worker_threads, note, cancelled, progress):
                rows.append(row)
                metric_bests.update(row)
                top1_value = row.get("top1_accuracy")
                top1 = float(top1_value) if row["status"] == "complete" and top1_value is not None else -np.inf
                if np.isfinite(top1) and top1 > best_top1:
                    best_top1 = top1
                    manifest.update(best_top1_accuracy=top1, best_top5_accuracy=row.get("top5_accuracy"),
                                    best_mpr=row.get("mpr"), best_subset_id=row["subset_id"],
                                    best_parcels=row["parcels"].split(";"))
                    log(f"New exhaustive-screening best top-1: {100 * top1:.2f}%; subset {row['subset_id']}")
                checkpoint("running")
                session_attempted = len(rows) - session_start_attempted
                elapsed = time.monotonic() - fit_batch_started
                eta = elapsed / max(1, session_attempted) * (total - len(rows))
                err_suffix = f" ({row['error']})" if row.get("status") == "failed" and row.get("error") else ""
                progress(
                    f"Subset {row['subset_id']} {row['status']}{err_suffix}; {len(rows)}/{total} fits recorded; "
                    f"{completion_rate_label()}; fit ETA ~{_format_duration(eta)}; "
                    f"{metric_bests.label()}."
                )
            pending = iter(())
            successful = [row for row in rows if row["status"] == "complete"]
            if successful:
                result_file = Path(successful[-1]["output_dir"]) / "result.joblib"
                if result_file.is_file():
                    last_result = joblib.load(result_file)
        while pending:
            if cancelled():
                break
            if backward:
                parcels = pending.popleft()
                index = len(rows) + 1
            else:
                try:
                    index, parcels = next(pending)
                except StopIteration:
                    break
            token = re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(parcels))[:90]
            digest = hashlib.sha256("\n".join(parcels).encode()).hexdigest()[:10]
            name = f"{index:03d}_n{len(parcels)}_{token}_{digest}"
            run_folder = folder / name
            if run_folder.exists():
                run_folder = folder / f"{name}_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            run_folder.mkdir()
            session_attempted = len(rows) - session_start_attempted
            if session_attempted:
                elapsed = time.monotonic() - fit_batch_started
                eta = elapsed / session_attempted * (total - len(rows))
                eta_label = f"fit ETA ~{_format_duration(eta)}"
            else:
                eta_label = "fit ETA calculating"
            prefix = f"Subset {index}/{total} ({len(parcels)} parcels; {eta_label})"
            progress(f"{prefix}: {', '.join(parcels)}")
            run_opts = dict(opts, selected_parcellations=list(parcels))
            run_opts["progress_callback"] = lambda message, p=prefix: progress(f"{p}: {message}")
            metric_callback = options.get("training_metric_callback")
            if callable(metric_callback):
                run_opts["training_metric_callback"] = lambda metric, i=index, p=prefix: metric_callback(
                    dict(metric, series=i, series_label=p))
            run_opts["inference_artifact_callback"] = lambda bundle, dest=run_folder: _save_inference(dest, bundle)
            row = {"subset_id": index, "n_parcels": len(parcels),
                   "parcels": ";".join(parcels), "output_dir": str(run_folder)}
            try:
                result, message = compute_transformer_decoding_from_payload(
                    context, run_opts, subset_payload(payload, parcels))
                if not isinstance(result, dict):
                    raise ValueError(message or "Decoder returned no result.")
                result["subset_screening_note"] = note
                result["selected_parcellations"] = list(parcels)
                _save_result(run_folder, result, run_opts)
                row.update(status="complete", **{key: result.get(key) for key in (
                    "mpr", "top1_accuracy", "top5_accuracy", "n_train", "n_validation",
                    "n_test", "n_features", "n_source_trials", "validation_top1_accuracy", "validation_mpr")})
                last_result = result
                top1 = float(result.get("top1_accuracy", np.nan))
                if not backward and np.isfinite(top1) and top1 > best_top1:
                    best_top1 = top1
                    mpr = float(result.get("mpr", np.nan))
                    top5 = float(result.get("top5_accuracy", np.nan))
                    mpr_label = f"{mpr:.4f}" if np.isfinite(mpr) else "unavailable"
                    top5_label = f"{100.0 * top5:.2f}%" if np.isfinite(top5) else "unavailable"
                    manifest.update(
                        best_top1_accuracy=top1,
                        best_top5_accuracy=top5 if np.isfinite(top5) else None,
                        best_mpr=mpr if np.isfinite(mpr) else None,
                        best_subset_id=index,
                        best_parcels=list(parcels),
                    )
                    log(
                        f"New exhaustive-screening best top-1: {100.0 * top1:.2f}% "
                        f"(top-5 {top5_label}, MPR {mpr_label}); subset {index}/{total}, "
                        f"{len(parcels)} parcel(s): {', '.join(parcels)}"
                    )
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                _write_json(run_folder / "error.json", row)
                progress(f"{prefix} failed: {exc}; continuing.")
            rows.append(row)
            metric_bests.update(row)
            if backward:
                stage_rows.append(row)
            checkpoint("running")
            elapsed = time.monotonic() - fit_batch_started
            remaining = total - len(rows)
            session_attempted = len(rows) - session_start_attempted
            eta = elapsed / max(1, session_attempted) * remaining
            progress(
                f"Parcel batch: {len(rows)}/{total} fits attempted; {completion_rate_label()}; "
                f"fit ETA ~{_format_duration(eta)}"
                + (f"; {metric_bests.label()}." if not backward else ".")
            )
            if backward and not pending:
                winner = choose_backward_candidate(stage_rows)
                chosen_pool = tuple(winner["parcels"].split(";"))
                removed = sorted(set(current_pool) - set(chosen_pool))
                path_rows.append(dict(winner, removed_parcel=";".join(removed), step=len(path_rows)))
                current_pool = chosen_pool
                stage_rows = []
                pending = deque(
                    [current_pool[:i] + current_pool[i+1:] for i in range(len(current_pool))]
                    if len(current_pool) > 1 else []
                )
                progress(f"Backward path: retained {len(current_pool)} parcels; validation top-1={winner['validation_top1_accuracy']:.4f}")
                checkpoint("running")
        status = "cancelled" if len(rows) < total else ("complete_with_errors" if manifest["failed"] else "complete")
        checkpoint(status)
        _comparison_plot(folder, rows, len(pool))
        _elimination_plot(folder, path_rows)
    except Exception as exc:
        manifest["error"] = str(exc)
        checkpoint("failed")
        raise RuntimeError(f"Semantic batch failed: {exc}. Saved progress: {folder}") from exc
    message = f"Parcel batch {status}: {manifest['completed']}/{total} saved, {manifest['failed']} failed. Outputs: {folder}"
    if last_result is not None:
        message += " (Panel displays the last successful subset, not a selected winner.)"
        last_result = dict(last_result, subset_comparison_df=pd.DataFrame(rows),
                           elimination_path_df=pd.DataFrame(path_rows),
                           semantic_subset_output_dir=str(folder))
    return last_result, message


def run_semantic_subset_batch(context, options):
    """Allow only one active coordinator per resumed batch (macOS/Linux)."""
    resume = options.get("semantic_subset_resume_dir")
    if not resume or options.get("resume_dry_run"):
        return _run_semantic_subset_batch(context, options)
    import fcntl
    folder = Path(resume).expanduser().resolve()
    with (folder / ".subset_batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another screening coordinator is already using {folder}") from exc
        try:
            return _run_semantic_subset_batch(context, options)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
