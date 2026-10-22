"""Time-resolved pairwise SVM decoding following Bankson et al. (2018).

Uses existing MAS epoch preprocessing, with train-fold channel standardization;
this does not reproduce the paper's global PCA/baseline scaling/smoothing.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def supertrial_indices(indices, size, count, rng):
    """Disjoint random groups; unused trials are redrawn on the next repeat."""
    return rng.permutation(indices)[:size * count].reshape(count, size)


def compute_pairwise_decoding(feature_payload, options=None):
    options = dict(options or {})
    if str(options.get("split_mode", "random")) != "random":
        return None, "Time-resolved pairwise decoding requires the random split setting; it uses leave-one-supertrial-out validation."
    if feature_payload.get("feature_space", "signal") != "signal":
        return None, "Time-resolved pairwise decoding currently requires signal-space features."
    try:
        size = int(options.get("pairwise_supertrial_size", 6))
        repeats = int(options.get("pairwise_repeats", 100))
        seed = int(options.get("pairwise_random_state", 42))
        c = float(options.get("pairwise_svm_c", 1.0))
        if size < 1 or repeats < 1 or not np.isfinite(c) or c <= 0:
            raise ValueError("Supertrial size, repeats and SVM C must be positive.")
        X = np.asarray(feature_payload["X"], dtype=float)
        y = np.asarray(feature_payload["y"], dtype=str)
        meta = feature_payload["feature_meta_df"].reset_index(drop=True)
        if X.ndim != 2 or len(y) != len(X) or len(meta) != X.shape[1] or not np.isfinite(X).all():
            raise ValueError("Invalid trial features or feature metadata.")
        if "relative_time_s" not in meta or not meta["stat"].eq("raw").all():
            raise ValueError("Time-resolved pairwise decoding requires raw sample features.")
        times = np.sort(meta.relative_time_s.unique().astype(float))
        if not len(times) or not np.isfinite(times).all():
            raise ValueError("Invalid sample times.")
        columns = [np.flatnonzero(meta.relative_time_s.to_numpy() == t) for t in times]
        channels = [meta.iloc[idx].channel.tolist() for idx in columns]
        if any(ch != channels[0] or len(set(ch)) != len(ch) for ch in channels):
            raise ValueError("Each time point must have the same unique channels.")
        classes = sorted(np.unique(y).tolist())
        if len(classes) < 2:
            raise ValueError("At least two concepts are required.")
        indices = {label: np.flatnonzero(y == label) for label in classes}
        short = [label for label in classes if len(indices[label]) < 2 * size]
        if short:
            raise ValueError(f"Need at least {2 * size} trials per concept for two supertrials; insufficient: {', '.join(short)}. Reduce pairwise_supertrial_size or adjust labels/trial selection.")
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc)

    rng = np.random.default_rng(seed)
    pairs = list(combinations(range(len(classes)), 2))
    matrices = np.full((len(times), len(classes), len(classes)), np.nan)
    rows, split_rows = [], []
    callback = options.get("progress_callback")
    for pair_number, (a, b) in enumerate(pairs, 1):
        left, right = classes[a], classes[b]
        count = min(len(indices[left]), len(indices[right])) // size
        scores = np.zeros(len(times))
        for repeat in range(repeats):
            groups = [supertrial_indices(indices[label], size, count, rng) for label in (left, right)]
            averages = np.concatenate([X[g].mean(axis=1) for g in groups])
            labels = np.repeat([0, 1], count)
            for label, group in zip((left, right), groups):
                for fold, members in enumerate(group):
                    split_rows.append({"concept_a": left, "concept_b": right, "repeat": repeat,
                                       "concept": label, "held_out_fold": fold,
                                       "trial_indices": ";".join(map(str, members))})
            for fold in range(count):
                test = np.array([fold, count + fold])
                train = np.setdiff1d(np.arange(2 * count), test)
                for t, cols in enumerate(columns):
                    model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=c))
                    model.fit(averages[np.ix_(train, cols)], labels[train])
                    scores[t] += np.mean(model.predict(averages[np.ix_(test, cols)]) == labels[test])
            if callable(callback):
                callback(f"Pairwise SVM: pair {pair_number}/{len(pairs)}, repeat {repeat + 1}/{repeats}")
        scores /= repeats * count
        matrices[:, a, b] = matrices[:, b, a] = scores
        for t, score in zip(times, scores):
            rows.append({"time_s": t, "concept_a": left, "concept_b": right,
                         "accuracy": score, "chance_accuracy": .5,
                         "supertrials_per_concept": count, "repeats": repeats,
                         "trials_used_per_concept_per_repeat": count * size})
    pair_df = pd.DataFrame(rows)
    curve = pair_df.groupby("time_s", as_index=False).accuracy.mean()
    curve["chance_accuracy"] = .5
    curve["n_pairs"] = len(pairs)
    summary = dict(decoder="Time-resolved pairwise linear SVM", decoder_type="pairwise_svm",
                   feature_space="signal", accuracy=float(curve.accuracy.mean()),
                   balanced_accuracy=float(curve.accuracy.mean()), chance_accuracy=.5,
                   accuracy_aggregation="mean across concept pairs and time points",
                   n_classes=len(classes), n_concepts=len(classes), n_trials=len(y), n_channels=len(channels[0]),
                   n_features=len(channels[0]), n_time_points=len(times), n_pairs=len(pairs),
                   n_runs=feature_payload.get("n_runs", 0), group_mode=feature_payload.get("group_mode", "canonical"),
                   split="leave-one-supertrial-per-concept-out", split_mode="pairwise_loso",
                   pairwise_supertrial_size=size, pairwise_repeats=repeats,
                   pairwise_random_state=seed, pairwise_svm_c=c,
                   n_permutations_completed=0,
                   decoder_note="Existing MAS preprocessing; train-fold channel scaling; no paper PCA or Gaussian smoothing. Group sign-permutation inference is not performed.")
    for key in ("window_start", "window_end", "bandpass_low_hz", "bandpass_high_hz",
                "baseline_correct", "resample_enabled", "resample_sfreq", "dropped_trials",
                "epoch_cache_hits", "epoch_cache_misses", "channel_mode"):
        summary[key] = feature_payload.get(key)
    summary["feature_binning_enabled"] = False
    result = {**summary, "summary_df": pd.DataFrame([summary]), "metrics_df": curve,
              "timecourse_df": curve, "pairwise_accuracy_df": pair_df,
              "pairwise_accuracy_matrix": matrices, "pairwise_times_s": times,
              "class_names": classes, "split_df": pd.DataFrame(split_rows),
              "trial_qc_df": feature_payload.get("trial_qc_df", pd.DataFrame()),
              "pairwise_trial_index_df": feature_payload["trial_df"].reset_index(drop=True).rename_axis("pairwise_payload_index").reset_index()}
    return result, f"Pairwise SVM: {len(pairs)} pairs, {len(times)} time points, {repeats} repeats; chance=0.5."


def plot_pairwise_timecourse(ax, result):
    curve = result["timecourse_df"]
    ax.plot(curve.time_s * 1000, curve.accuracy, color="tab:blue")
    ax.axhline(.5, color="gray", linestyle="--", label="Chance (50%)")
    ax.axvline(0, color="gray", linewidth=.7)
    ax.set(xlabel="Time from stimulus onset (ms)", ylabel="Mean pairwise accuracy",
           title="Time-resolved pairwise decoding", ylim=(0, 1))
    ax.legend()
