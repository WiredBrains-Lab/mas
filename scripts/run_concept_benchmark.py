#!/usr/bin/env python3
"""Prepare, run/resume and collect a frozen source-space decoding benchmark.

Start with `preflight`, then `prepare`, `run`, and `collect`.
The costly refitted null stage is separate: `run --stage null`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.concept_benchmark_splits import make_splits

HCP = ["Inferior Parietal Cortex-lh", "Lateral Temporal Cortex-lh",
       "MT+ Complex and Neighboring Visual Areas-lh", "Medial Temporal Cortex-lh",
       "Superior Parietal Cortex-lh"]
SEMANTIC = [f"ctx-lh-{r}" for r in ("temporalpole", "middletemporal", "inferiorparietal",
                                  "parstriangularis", "parsorbitalis", "bankssts")]
MODELS = {
    "hcp5": ("hcp", HCP),
    "semantic6": ("aparc", SEMANTIC),
    "semantic3": ("aparc", [f"ctx-lh-{r}" for r in ("middletemporal", "inferiorparietal", "parsorbitalis")]),
    "semantic4": ("aparc", [f"ctx-lh-{r}" for r in ("temporalpole", "inferiorparietal", "parstriangularis", "bankssts")]),
    "aparc34": ("aparc", None),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    temporary.replace(path)


def identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def trial_metadata(frame, y=None):
    required = ["source_fif_path", "onset_sample", "stimulus", "session_id"]
    if frame[required].isna().any().any():
        raise ValueError("Missing trial identity, image or session metadata")
    result = pd.DataFrame({
        "trial_id": frame.source_fif_path.astype(str) + "::" + frame.onset_sample.astype(int).astype(str),
        "concept": np.asarray(y if y is not None else frame.canonical_stimulus, dtype=str),
        "image": frame.stimulus.astype(str).map(lambda s: Path(s.replace('\\', '/')).name),
        "session": frame.session_id.astype(str),
    })
    if result.trial_id.duplicated().any():
        raise ValueError("Duplicate raw-trial IDs")
    return result


def load_setup(config_path):
    cfg = json.loads(config_path.read_text())
    for key in ("reference_run", "output_dir"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((config_path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    run = Path(cfg["reference_run"])
    options = json.loads((run / "settings_used.json").read_text())
    summary = json.loads((run / "result_summary.json").read_text())
    options.update(cfg["options"])
    options.update(feature_space="source", save_full_feature_table=False,
                   require_subset_validation=True, n_permutations=0)
    trials = pd.read_csv(run / "tables/trial_qc.csv")
    trials = trials.drop(columns=[c for c in trials if c.startswith("qc_") or c.startswith("_source_")])
    frame = trial_metadata(trials)
    if frame.concept.nunique() != cfg["expected_concepts"]:
        raise ValueError("Unexpected candidate count in input metadata")
    paths = [run / "settings_used.json", run / "tables/trial_qc.csv", Path(options["embedding_dict_path"])]
    paths += [Path(p) for p in trials.source_fif_path.unique()]
    inverse_dir = Path(summary["source_subjects_dir"]) / "inverse"
    paths += sorted(inverse_dir.glob("*-inv.fif*"))
    for meta_path in inverse_dir.glob("*-inv.fif.source_meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("trans_identity", {}).get("path"):
            paths.append(Path(meta["trans_identity"]["path"]))
    for subject in ("Anatomy", "fsaverage"):
        subject_dir = Path(summary["source_subjects_dir"]) / subject
        paths += sorted((subject_dir / "label").glob("*.annot"))
        paths += sorted((subject_dir / "surf").glob("*sphere*"))
        paths += sorted((subject_dir / "surf").glob("*.white"))
    code = [Path(__file__), ROOT / "core/concept_benchmark_splits.py",
            ROOT / "core/concept_decoding_transformer.py", ROOT / "core/concept_decoding_source.py",
            ROOT / "core/concept_decoding_signal.py", ROOT / "scripts/analyze_parcel_contribution.py"]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "mne", "torch", "joblib")}
    provenance = dict(config=cfg, options=options, inputs=[identity(p) for p in sorted(set(paths))],
                      code={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}, versions=versions)
    return cfg, options, summary, trials, frame, provenance


def make_jobs(cfg, splits):
    jobs = []
    for split in splits:
        track = split["track"]
        models = {"image": ["hcp5", "semantic6", "semantic3", "semantic4"],
                  "zero_shot": ["aparc34", "semantic6"], "session": ["semantic6"]}[track]
        for model in models:
            for decoder in ("embedding_ridge", "embedding_contrastive"):
                seeds = cfg["training_seeds"] if decoder == "embedding_contrastive" else cfg["training_seeds"][:1]
                for seed in seeds:
                    jobs.append(dict(stage="main", track=track, fold=split["fold"], model=model, decoder=decoder, seed=seed, permutation=0))
        if track != "session":
            for perm in range(1, cfg["null_permutations"] + 1):
                jobs.append(dict(stage="null", track=track, fold=split["fold"], model="semantic6",
                                 decoder="embedding_contrastive", seed=cfg["training_seeds"][0], permutation=perm))
    for job in jobs:
        job["id"] = "__".join(str(job[k]) for k in ("stage", "track", "fold", "model", "decoder", "seed", "permutation"))
    return jobs


def preflight(cfg, options, summary, trials, frame, provenance):
    from scripts.analyze_parcel_contribution import SourceContext
    # Verify raw/inverse matching before any expensive reconstruction.
    context = SourceContext(trials, summary["source_subjects_dir"])
    for p in trials.source_fif_path.unique():
        context._inverse_operator_matches_fif(context._find_inverse_operator_for_fif(p), p)
    splits = make_splits(frame, cfg["folds"], cfg["split_seed"])
    jobs = make_jobs(cfg, splits)
    print(json.dumps(dict(raw_trials=len(frame), concepts=frame.concept.nunique(),
                         images=frame.image.nunique(), sessions=sorted(frame.session.unique()),
                         minimum_images_per_concept=int(frame.groupby("concept").image.nunique().min()),
                         main_fits=sum(j["stage"] == "main" for j in jobs),
                         null_fits=sum(j["stage"] == "null" for j in jobs),
                         fingerprint=digest(provenance), output_dir=cfg["output_dir"]), indent=2))


def prepare(cfg, options, summary, trials, frame, provenance):
    import gc
    import joblib
    from scripts.analyze_parcel_contribution import SourceContext
    from core.concept_decoding_source import build_source_space_concept_features
    from core.concept_decoding_transformer import prepare_semantic_embedding_feature_payload
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(provenance)
    frozen = out / "provenance.json"
    if frozen.exists() and digest(json.loads(frozen.read_text())) != fingerprint:
        raise ValueError("Inputs, code, environment or settings changed. Use a new output_dir.")
    atomic_json(frozen, provenance)
    if (out / "plan.json").exists():
        print(f"Already prepared: {out}. Use run to resume fitting.")
        return
    atlas_frames = {}
    for atlas in ("aparc", "hcp"):
        atlas_options = dict(options, source_parcellation="aparc" if atlas == "aparc" else "HCPMMP1_combined",
                             selected_parcellations=options["selected_parcellations"] if atlas == "aparc" else HCP)
        parts = []
        for index, (_, group) in enumerate(trials.groupby("source_fif_path", sort=True)):
            cache = out / "features" / f"{atlas}_run{index:02d}.joblib"
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                part = joblib.load(cache)
            else:
                context = SourceContext(group.reset_index(drop=True), summary["source_subjects_dir"])
                part, error = build_source_space_concept_features(context, dict(atlas_options, progress_callback=lambda s: print(s, flush=True)))
                if part is None or part.get("skipped_paths"):
                    raise RuntimeError(error or str(part.get("skipped_paths")))
                temporary = cache.with_suffix(".tmp")
                joblib.dump(part, temporary, compress=3)
                temporary.replace(cache)
                del context
            parts.append(part)
        for part in parts[1:]:
            if part["feature_names"] != parts[0]["feature_names"]:
                raise ValueError("Feature ordering differs across runs")
        payload = dict(parts[0])
        payload.update(X=np.concatenate([p["X"] for p in parts]), y=np.concatenate([p["y"] for p in parts]),
                       trial_df=pd.concat([p["trial_df"] for p in parts], ignore_index=True),
                       trial_qc_df=pd.concat([p["trial_qc_df"] for p in parts], ignore_index=True), n_runs=len(parts))
        payload, error = prepare_semantic_embedding_feature_payload(payload, atlas_options)
        if payload is None:
            raise RuntimeError(error)
        if set(payload["roi_names"]) != set(atlas_options["selected_parcellations"]):
            raise ValueError(f"Requested parcels missing or substituted in {atlas}")
        # Text targets must be one fixed embedding per concept for these benchmarks.
        for label in set(payload["y"]):
            targets = payload["embeddings"][payload["y"] == label]
            if not np.allclose(targets, targets[0], atol=1e-6):
                raise ValueError("Benchmark requires fixed concept text embeddings")
        atlas_frames[atlas] = trial_metadata(payload["trial_df"], payload["y"])
        joblib.dump(payload, out / "features" / f"{atlas}.joblib", compress=3)
        del payload, parts
        gc.collect()
    common = set(atlas_frames["hcp"].trial_id) & set(atlas_frames["aparc"].trial_id)
    matched = atlas_frames["aparc"].loc[lambda x: x.trial_id.isin(common)].reset_index(drop=True)
    hcp_matched = atlas_frames["hcp"].set_index("trial_id").loc[matched.trial_id].reset_index()
    if not matched.equals(hcp_matched[matched.columns]):
        raise ValueError("Atlas payload labels or metadata disagree")
    if matched.concept.nunique() != cfg["expected_concepts"]:
        raise ValueError("QC removed a concept")
    splits = make_splits(matched, cfg["folds"], cfg["split_seed"])
    matched.to_csv(out / "matched_trials.csv", index=False)
    jobs = make_jobs(cfg, splits)
    atomic_json(out / "plan.json", dict(fingerprint=fingerprint, splits=splits, jobs=jobs,
                                       matched_digest=digest(matched.to_dict("records")),
                                       feature_files={a: identity(out / "features" / f"{a}.joblib") for a in atlas_frames},
                                       qc=dict(input=len(frame), matched=len(matched), retained_by_atlas={a: len(f) for a, f in atlas_frames.items()})))
    print(f"Prepared {len(matched)} matched trials and {len(jobs)} jobs in {out}")


def run_jobs(cfg, options, provenance, args):
    import joblib
    from core.concept_decoding_subsets import subset_payload, _save_result
    from core.concept_decoding_transformer import compute_transformer_decoding_from_payload
    out = Path(cfg["output_dir"])
    plan = json.loads((out / "plan.json").read_text())
    if digest(provenance) != plan["fingerprint"]:
        raise ValueError("Frozen benchmark provenance changed. Use a new output_dir and prepare again.")
    matched = pd.read_csv(out / "matched_trials.csv", dtype=str)
    if digest(matched.to_dict("records")) != plan["matched_digest"]:
        raise ValueError("Matched trial manifest changed")
    for atlas, file_identity in plan["feature_files"].items():
        if identity(file_identity["path"]) != file_identity:
            raise ValueError(f"Frozen {atlas} features changed")
    active_atlas, payload = None, None
    completed = 0
    for job in plan["jobs"]:
        if job["stage"] != args.stage or (args.track and job["track"] != args.track) or (args.decoder and job["decoder"] != args.decoder):
            continue
        dest = out / "runs" / job["id"]
        marker = dest / "complete.json"
        if marker.exists():
            if json.loads(marker.read_text())["fingerprint"] != plan["fingerprint"]:
                raise ValueError("Completed job belongs to a different benchmark")
            continue
        if args.limit is not None and completed >= args.limit:
            break
        dest.mkdir(parents=True, exist_ok=True)
        lock = dest / "running.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"Run locked: {lock}. Check the process before removing a stale lock.")
        os.close(fd)
        try:
            atlas, parcels = MODELS[job["model"]]
            if atlas != active_atlas:
                payload = joblib.load(out / "features" / f"{atlas}.joblib")
                ids = trial_metadata(payload["trial_df"], payload["y"]).set_index("trial_id")
                lookup = {key: i for i, key in enumerate(ids.index)}
                indices = [lookup[key] for key in matched.trial_id]
                for key in ("X", "X_tensor", "y", "embeddings"):
                    if key in payload:
                        payload[key] = np.asarray(payload[key])[indices]
                payload["trial_df"] = payload["trial_df"].iloc[indices].reset_index(drop=True)
                active_atlas = atlas
            selected = subset_payload(payload, parcels) if parcels is not None else dict(payload)
            split = next(s for s in plan["splits"] if s["track"] == job["track"] and s["fold"] == job["fold"])
            if job["permutation"]:
                # Permute concept-to-target assignments consistently across train/validation.
                # Test labels and targets remain untouched. Refit the entire decoder.
                selected = dict(selected, y=selected["y"].copy(), embeddings=selected["embeddings"].copy(), trial_df=selected["trial_df"].copy())
                available = split["train"] + split["validation"]
                labels = sorted(set(selected["y"][available]))
                shuffled = np.random.default_rng(cfg["split_seed"] + job["permutation"]).permutation(labels)
                mapping = dict(zip(labels, shuffled))
                targets = {label: selected["embeddings"][np.flatnonzero(selected["y"] == label)[0]].copy() for label in labels}
                for i in available:
                    label = mapping[selected["y"][i]]
                    selected["y"][i] = label
                    selected["embeddings"][i] = targets[label]
                atomic_json(dest / "null_mapping.json", mapping)
            run_options = dict(options, decoder_type=job["decoder"], training_seed=job["seed"],
                               source_parcellation="aparc" if atlas == "aparc" else "HCPMMP1_combined",
                               selected_parcellations=parcels or options["selected_parcellations"],
                               split_mode="zero_shot" if job["track"] == "zero_shot" else job["track"],
                               explicit_split_indices={k: split[k] for k in ("train", "validation", "test")})
            print(f"Running {job['id']}", flush=True)
            result, error = compute_transformer_decoding_from_payload(None, run_options, selected)
            if result is None:
                raise RuntimeError(error)
            _save_result(dest, result, run_options)
            atomic_json(marker, dict(job=job, fingerprint=plan["fingerprint"]))
            completed += 1
        except Exception:
            (dest / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            lock.unlink(missing_ok=True)
    collect(cfg)


def collect(cfg):
    out = Path(cfg["output_dir"])
    plan = json.loads((out / "plan.json").read_text())
    rows, concepts = [], []
    metrics = ["mpr", "top1_accuracy", "top5_accuracy"]
    for job in plan["jobs"]:
        dest = out / "runs" / job["id"]
        if not (dest / "complete.json").exists():
            continue
        if json.loads((dest / "complete.json").read_text())["fingerprint"] != plan["fingerprint"]:
            raise ValueError("Cannot collect results from a different benchmark fingerprint")
        result = json.loads((dest / "result_summary.json").read_text())
        rows.append(dict(job, **{k: result.get(k) for k in metrics + ["n_test", "n_source_test_trials", "n_concepts", "validation_mpr"]}))
        if job["stage"] == "main":
            per = pd.read_csv(dest / "tables/per_class_accuracy.csv")
            for key in ("track", "fold", "model", "decoder", "seed"):
                per[key] = job[key]
            concepts.append(per)
    report = out / "report"
    report.mkdir(exist_ok=True)
    pd.DataFrame([dict(j, status="complete" if (out / "runs" / j["id"] / "complete.json").exists() else "pending") for j in plan["jobs"]]).to_csv(report / "run_status.csv", index=False)
    if not rows:
        print("No completed results yet")
        return
    results = pd.DataFrame(rows)
    results.to_csv(report / "all_runs.csv", index=False)
    main = results.loc[results.stage == "main"]
    if main.empty:
        return
    keys = ["track", "model", "decoder"]
    # Seeds first, then folds: do not count seeds as independent held-out datasets.
    folds = main.groupby(keys + ["fold"])[metrics].mean().reset_index()
    folds.to_csv(report / "fold_metrics.csv", index=False)
    main.groupby(keys + ["fold"])[metrics].std().to_csv(report / "seed_variation.csv")
    folds.groupby(keys)[metrics].agg(["mean", "std", "count"]).to_csv(report / "summary.csv")
    expected = pd.DataFrame([j for j in plan["jobs"] if j["stage"] == "main"]).groupby(keys).size().rename("expected_fits")
    coverage = expected.to_frame().join(main.groupby(keys).size().rename("completed_fits")).fillna(0)
    coverage["complete"] = coverage.expected_fits == coverage.completed_fits
    coverage.to_csv(report / "comparison_coverage.csv")
    reference = main.loc[main.model == "semantic6", ["track", "fold", "decoder", "seed"] + metrics]
    paired = main.merge(reference, on=["track", "fold", "decoder", "seed"], suffixes=("", "_reference"))
    for metric in metrics:
        paired[metric + "_difference"] = paired[metric] - paired[metric + "_reference"]
    paired.to_csv(report / "paired_differences.csv", index=False)
    per = pd.concat(concepts, ignore_index=True)
    per.to_csv(report / "per_concept_all_runs.csv", index=False)
    if (per.groupby(keys + ["fold", "concept"]).n_test.nunique() > 1).any():
        raise ValueError("Test averaging/counts differ across training seeds")
    per_fold = per.groupby(keys + ["fold", "concept"]).agg(mpr=("test_mpr", "mean"), top1=("top1_accuracy", "mean"), n_test=("n_test", "first")).reset_index()
    for metric in ("mpr", "top1"):
        per_fold[metric + "_sum"] = per_fold[metric] * per_fold.n_test
    pooled = per_fold.groupby(keys + ["concept"])[["mpr_sum", "top1_sum", "n_test"]].sum()
    pooled["mpr"] = pooled.mpr_sum / pooled.n_test
    pooled["top1_accuracy"] = pooled.top1_sum / pooled.n_test
    pooled.to_csv(report / "per_concept_pooled.csv")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for (track, model, decoder), group in pooled.reset_index().groupby(keys):
        group = group.sort_values("concept")
        fig = Figure(figsize=(12, max(4, len(group) * 0.23)), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        candidate_count = int(main.loc[(main.track == track) & (main.model == model) & (main.decoder == decoder), "n_concepts"].iloc[0])
        for ax, metric, chance in zip(axes, ("mpr", "top1_accuracy"), (0.5, 1 / candidate_count)):
            ax.barh(group.concept, group[metric], color="#3D8DFF")
            ax.axvline(chance, color="black", linestyle="--", label="Chance")
            ax.set(xlim=(0, 1), xlabel=metric)
            ax.invert_yaxis()
            ax.legend()
        fig.suptitle(f"{track}: {model}, {decoder} (check comparison_coverage.csv)")
        fig.savefig(report / f"concepts_{track}_{model}_{decoder}.png", dpi=160)
    text = ["# Concept benchmark", "", f"Completed {len(rows)} / {len(plan['jobs'])} planned fits.", "",
            "MPR chance is 0.5. Candidate sets: image/session 50, zero-shot 10 (default five folds).",
            "Summary averages seeds within each fold, then reports mean and SD across folds.",
            "Session directions appear individually in fold_metrics.csv. SD is descriptive, not a confidence interval.",
            "Consult run_status.csv: incomplete folds or seeds are provisional and must not support a final comparison.",
            "Per-concept counts count test averages once per fold, not once per training seed.", "",
            "Session transfer may reuse image identities across sessions. It answers session transfer only.",
            "Historical selected subsets remain exploratory. No new parcel search uses these test sets.", "",
            "Null controls refit shuffled concept-to-target mappings on training/validation only.",
            "Their exchangeability assumption is across concepts. They do not establish cross-patient significance."]
    null = results.loc[results.stage == "null"]
    null_rows = []
    for track, group in null.groupby("track"):
        expected = {s["fold"] for s in plan["splits"] if s["track"] == track}
        reference = main.loc[(main.track == track) & (main.model == "semantic6") & (main.decoder == "embedding_contrastive") & (main.seed == cfg["training_seeds"][0])]
        complete_perms = [g.mpr.mean() for _, g in group.groupby("permutation") if set(g.fold) == expected and len(g) == len(expected)]
        if len(reference) == len(expected) and set(reference.fold) == expected and len(complete_perms) == cfg["null_permutations"]:
            observed = float(reference.mpr.mean())
            null_rows.append(dict(track=track, observed_mpr=observed, permutations=len(complete_perms),
                                  p_value=(1 + np.sum(np.asarray(complete_perms) >= observed)) / (1 + len(complete_perms))))
    pd.DataFrame(null_rows, columns=["track", "observed_mpr", "permutations", "p_value"]).to_csv(report / "null_tests.csv", index=False)
    (report / "README.md").write_text("\n".join(text) + "\n")
    print(f"Collected results into {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "prepare", "run", "collect", "all"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/concept_benchmark.json")
    parser.add_argument("--stage", choices=["main", "null"], default="main")
    parser.add_argument("--track", choices=["image", "zero_shot", "session"])
    parser.add_argument("--decoder", choices=["embedding_ridge", "embedding_contrastive"])
    parser.add_argument("--limit", type=int, help="Maximum new fits; completed fits are skipped")
    parser.add_argument("--include-nulls", action="store_true", help="With all, also run the costly refitted null stage")
    args = parser.parse_args()
    cfg, options, summary, trials, frame, provenance = load_setup(args.config.resolve())
    if args.command == "all":
        if args.limit is not None or args.track or args.decoder or args.stage != "main":
            parser.error("Use run for filtered or limited execution")
        preflight(cfg, options, summary, trials, frame, provenance)
        prepare(cfg, options, summary, trials, frame, provenance)
        args.decoder = "embedding_ridge"
        run_jobs(cfg, options, provenance, args)
        args.decoder = None
        run_jobs(cfg, options, provenance, args)
        if args.include_nulls:
            args.stage = "null"
            run_jobs(cfg, options, provenance, args)
    elif args.command == "preflight":
        preflight(cfg, options, summary, trials, frame, provenance)
    elif args.command == "prepare":
        preflight(cfg, options, summary, trials, frame, provenance)
        prepare(cfg, options, summary, trials, frame, provenance)
    elif args.command == "run":
        run_jobs(cfg, options, provenance, args)
    else:
        collect(cfg)


if __name__ == "__main__":
    main()
