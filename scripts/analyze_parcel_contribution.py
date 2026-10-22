#!/usr/bin/env python3
"""Reproduce a saved source decoder and measure grouped parcel permutation importance.

Uses existing inverse operators only, extracts one run at a time to bound RAM,
and keeps all analysis artifacts separate from the original saved run.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import mne
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_concept_decoding_cli import HeadlessConceptDecodingContext
from core import concept_decoding_source as source
from core import concept_decoding_transformer as semantic


def log(message):
    if "resampling source epochs" not in str(message):
        print(message, flush=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


class SourceContext(HeadlessConceptDecodingContext):
    def __init__(self, trials, subject_root):
        super().__init__(trials, first_fif_path=str(trials.iloc[0].source_fif_path))
        self.subject_root = Path(subject_root)
        self.meg_3d = SimpleNamespace(
            _meg_source_space_kind="surface",
            _get_brain_subject_info=lambda: ("Anatomy", str(self.subject_root)),
        )

    def _default_inverse_operator_output_path(self, fif_path):
        stem = Path(fif_path).stem.replace("_raw", "").replace("-raw", "")
        return str(self.subject_root / "inverse" / f"{stem}-inv.fif")

    _find_inverse_operator_for_fif = _default_inverse_operator_output_path

    def _inverse_operator_matches_fif(self, inverse_path, fif_path):
        metadata = json.loads(Path(str(inverse_path) + ".source_meta.json").read_text())
        st = Path(fif_path).stat()
        identity = {"path": str(Path(fif_path).absolute()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        if metadata.get("fif_identity") != identity:
            raise RuntimeError(f"Inverse/raw provenance mismatch: {inverse_path}")
        trans = metadata.get("trans_identity", {})
        if trans:
            ts = Path(trans["path"]).stat()
            if ts.st_size != trans["size"] or ts.st_mtime_ns != trans["mtime_ns"]:
                raise RuntimeError(f"Inverse/transform provenance mismatch: {inverse_path}")
        return True

    def _create_inverse_operator_for_fif(self, *args, **kwargs):
        raise RuntimeError("Analysis requires a matching existing inverse; generation is disabled")

    def _auto_configure_freesurfer_subject_for_fif(self, fif_path, **kwargs):
        meta = json.loads(Path(self._find_inverse_operator_for_fif(fif_path) + ".source_meta.json").read_text())
        if meta["subject"] != "Anatomy" or Path(meta["subjects_dir"]).resolve() != self.subject_root.resolve():
            raise RuntimeError("Unexpected source subject")
        return True, ""

    def _get_available_semantic_parcellations(self, subject, subjects_dir):
        labels = mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=subjects_dir, verbose="error")
        return sorted(f"ctx-{label.hemi}-{label.name.rsplit('-', 1)[0]}" for label in labels if "unknown" not in label.name)

    def _ctx_volume_name_to_annot_name(self, name):
        _, hemi, region = name.split("-", 2)
        return f"{region}-{hemi}"


def get_payload(run, out, options, summary, feature_cache_dir=None):
    trials = pd.read_csv(run / "tables/trial_qc.csv")
    # QC columns are recalculated by the same backend; retain original event mapping.
    trials = trials.drop(columns=[c for c in trials if c.startswith("qc_") or c.startswith("_source_")])
    paths = trials.source_fif_path.drop_duplicates().tolist()
    parts = []
    for index, path in enumerate(paths):
        cache = (feature_cache_dir or out) / f"features_run_{index + 1:02d}.joblib"
        if cache.exists():
            part = joblib.load(cache)
            log(f"Reusing analysis feature cache {index + 1}/{len(paths)}")
        else:
            frame = trials.loc[trials.source_fif_path == path].copy().reset_index(drop=True)
            context = SourceContext(frame, summary["source_subjects_dir"])
            log(f"Reconstructing source features {index + 1}/{len(paths)}: {Path(path).name}")
            part, error = source.build_source_space_concept_features(context, {**options, "progress_callback": log})
            if part is None or part.get("skipped_paths"):
                raise RuntimeError(error or str(part.get("skipped_paths")))
            joblib.dump(part, cache, compress=3)
            del context
            gc.collect()
        parts.append(part)
        log(f"Retained {len(part['y'])} trials; features {part['X'].shape[1]}")
    payload = dict(parts[0])
    for part in parts:
        assert part["feature_names"] == payload["feature_names"]
    payload["X"] = np.concatenate([p["X"] for p in parts])
    payload["y"] = np.concatenate([p["y"] for p in parts])
    for key in ("trial_df", "trial_qc_df"):
        payload[key] = pd.concat([p[key] for p in parts], ignore_index=True)
        payload[key]["_source_qc_uid"] = np.arange(len(payload[key]))
    payload["n_runs"] = len(paths)
    payload["dropped_trials"] = sum(p["dropped_trials"] for p in parts)
    payload["inverse_paths"] = {k: v for p in parts for k, v in p["inverse_paths"].items()}
    saved_kept = pd.read_csv(run / "tables/trial_qc.csv")
    saved_kept = saved_kept.loc[saved_kept.qc_status == "kept"]
    identity = ["source_fif_path", "onset_sample", "canonical_stimulus"]
    current_ids = payload["trial_df"][identity].astype(str).to_numpy().tolist()
    original_ids = saved_kept[identity].astype(str).to_numpy().tolist()
    write_json(out / "reconstruction_check.json", {"saved_retained": len(original_ids), "recomputed_retained": len(current_ids), "same_ordered_trial_ids": current_ids == original_ids})
    if current_ids != original_ids:
        raise RuntimeError("Reconstructed trials differ from saved run; see reconstruction_check.json")
    payload, error = semantic.prepare_semantic_embedding_feature_payload(payload, options)
    if payload is None:
        raise RuntimeError(error)
    return payload


def metric_rows(scores, y, names):
    mpr, ranks, _ = semantic._mean_percentile_rank(scores, y, names)
    truth = np.asarray([names.index(label) for label in y])
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0] == truth
    top5 = np.any(order[:, :5] == truth[:, None], axis=1)
    return np.column_stack([ranks, top1, top5])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--feature-cache-dir", type=Path)
    args = parser.parse_args()
    run, out = args.saved_run.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    options = json.loads((run / "settings_used.json").read_text())
    summary = json.loads((run / "result_summary.json").read_text())
    if options["decoder_type"] != "embedding_contrastive" or options["feature_space"] != "source":
        raise ValueError("This analysis is scoped to source contrastive linear runs")
    write_json(out / "analysis_settings.json", {"saved_run": str(run), "repeats": args.repeats, "shuffle_seed": 20260908, "training_seed": args.training_seed, "options": options})
    features = pd.read_csv(run / "tables/features.csv")
    if len(features) == summary["n_features"]:
        weights = features.groupby("roi").importance.agg(["mean", "max", "count"]).sort_values("mean", ascending=False)
        write_json(out / "saved_coefficient_ranking.json", weights.reset_index().to_dict("records"))
        log("Complete saved coefficient screen:\n" + weights.to_string())
    else:
        log("Saved coefficients are truncated to top 250; they cannot give an unbiased parcel ranking")
    payload = get_payload(run, out, options, summary, args.feature_cache_dir)
    captures = {}
    original_train = semantic._train_torch_embedding_model
    original_pseudo = semantic._make_pseudo_epochs

    def capture_train(*a, **kw):
        kw["random_state"] = args.training_seed
        model, metadata = original_train(*a, **kw)
        captures.update(model=model, metadata=metadata, names=kw["class_names"])
        return model, metadata

    def capture_pseudo(*a, **kw):
        data = original_pseudo(*a, **kw)
        captures.setdefault(kw["subset"], data)
        return data

    semantic._train_torch_embedding_model = capture_train
    semantic._make_pseudo_epochs = capture_pseudo
    try:
        result, error = semantic.compute_transformer_decoding_from_payload(None, {**options, "n_permutations": 0, "progress_callback": log}, payload)
    finally:
        semantic._train_torch_embedding_model = original_train
        semantic._make_pseudo_epochs = original_pseudo
    if result is None:
        raise RuntimeError(error)
    log(error)
    model, meta, names = captures["model"], captures["metadata"], captures["names"]
    test = captures["test"]
    train = captures["train"]
    prototypes = semantic._class_prototypes(train["y"], semantic._embedding_targets_for_scoring(train["embeddings"], meta), names)
    X = np.asarray(test["X"]).reshape(len(test["y"]), -1)

    def score(values):
        predictions = semantic._predict_flat_embedding_model(model, values, meta)
        return semantic._l2_normalize(predictions) @ prototypes.T

    baseline_scores = score(X)
    baseline_rows = metric_rows(baseline_scores, test["y"], names)
    baseline = baseline_rows.mean(axis=0)
    keys = ["mpr", "top1_accuracy", "top5_accuracy"]
    reproduction = {"saved": {k: summary[k] for k in keys}, "recomputed": dict(zip(keys, baseline)), "saved_epoch": summary["embedding_selected_epoch"], "recomputed_epoch": result["embedding_selected_epoch"], "same_trial_ids": True}
    write_json(out / "model_reproduction.json", reproduction)
    if max(abs(baseline[i] - summary[k]) for i, k in enumerate(keys)) > 1e-5:
        log("NOTE: baseline differs from saved model; importance will describe this reproduced fit")
    feature_meta = payload["feature_meta_df"].reset_index(drop=True)
    assert len(feature_meta) == X.shape[1]
    # Retain everything needed to repeat perturbations without source reconstruction.
    joblib.dump({"model": model, "train_meta": meta, "test_X": X, "test_y": test["y"], "prototypes": prototypes, "class_names": names, "feature_meta": feature_meta}, out / "inference_bundle.joblib", compress=3)
    rng = np.random.default_rng(20260908)
    permutations = [rng.permutation(len(X)) for _ in range(args.repeats)]
    rankings = []
    shuffle_rows = []
    groups = [(str(roi), np.flatnonzero(feature_meta.roi.to_numpy() == roi)) for roi in feature_meta.roi.drop_duplicates()]
    groups += [("ALL_PARCELS_CONTROL", np.arange(X.shape[1]))]
    for roi, columns in groups:
        drops = []
        for repeat, order in enumerate(permutations):
            shuffled = X.copy()
            shuffled[:, columns] = X[order][:, columns]
            perturbed = metric_rows(score(shuffled), test["y"], names).mean(axis=0)
            delta = baseline - perturbed
            drops.append(delta)
            shuffle_rows.append({"roi": roi, "repeat": repeat, **dict(zip(keys, perturbed)), **{f"drop_{k}": delta[i] for i, k in enumerate(keys)}})
        drops = np.asarray(drops)
        row = {"roi": roi, "n_features": len(columns)}
        for i, key in enumerate(keys):
            row.update({f"drop_{key}": float(drops[:, i].mean()), f"shuffle_sd_{key}": float(drops[:, i].std(ddof=1)), f"after_shuffle_{key}": float(baseline[i] - drops[:, i].mean())})
        rankings.append(row)
        log(f"{roi}: MPR drop={row['drop_mpr']:.4f}; top1 drop={100*row['drop_top1_accuracy']:.2f} percentage points")
    rankings.sort(key=lambda row: row["drop_mpr"], reverse=True)
    write_json(out / "parcel_permutation_ranking.json", {"baseline": dict(zip(keys, baseline)), "reproduction": reproduction, "repeats": args.repeats, "rows": rankings, "interpretation": "Held-out, post-hoc model reliance. Shuffle SD measures perturbation randomness, not population uncertainty or causal importance. Correlated parcels may share information. All time bins of each parcel are shuffled together across examples. No retraining or test-driven model selection."})
    write_json(out / "shuffle_results.json", shuffle_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    real = [row for row in rankings if row["roi"] != "ALL_PARCELS_CONTROL"]
    fig, ax = plt.subplots(figsize=(11, max(4, len(real) * .34 + 1.8)))
    ax.barh([r["roi"] for r in real][::-1], [r["drop_mpr"] for r in real][::-1], xerr=[r["shuffle_sd_mpr"] for r in real][::-1], color="#3274ad", capsize=3)
    ax.axvline(0, color="black", linewidth=.7)
    ax.set(xlabel="Decrease in held-out MPR when parcel is shuffled (mean ± shuffle SD)", title=f"Parcel contribution to concept decoding\nBaseline MPR {baseline[0]:.3f}; {args.repeats} shuffles per parcel")
    fig.tight_layout()
    fig.savefig(out / "parcel_contribution.png", dpi=180)
    plt.close(fig)
    lines = ["# Parcel contribution analysis", "", f"Source run: `{run.name}`", "", "Grouped permutation importance: shuffle all time bins of one parcel together across held-out pseudoepochs. Larger MPR drops indicate stronger model reliance.", "", f"Recomputed baseline: MPR {baseline[0]:.4f}; top-1 {baseline[1]:.1%}; top-5 {baseline[2]:.1%}.", "", "| Parcel | MPR drop | Top-1 drop (percentage points) |", "|---|---:|---:|"]
    lines += [f"| {r['roi']} | {r['drop_mpr']:.4f} | {100*r['drop_top1_accuracy']:.2f} |" for r in real]
    lines += ["", "This is an exploratory ranking within the selected model and its random test split. It does not establish unique anatomical or causal contribution; correlations between parcel features can redistribute importance. Error bars show variability across shuffles, not a confidence interval across subjects or new datasets.", "", f"Training seed: {args.training_seed}. Archived model: MPR {summary['mpr']:.4f}, top-1 {summary['top1_accuracy']:.1%}. Trial identities match, but reconstruction/refitting may differ from the archived fit; see model_reproduction.json.", "", "Method: https://scikit-learn.org/stable/modules/permutation_importance.html"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    log(f"Analysis complete: {out}")


if __name__ == "__main__":
    main()
