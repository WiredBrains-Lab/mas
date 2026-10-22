# Time-resolved pairwise decoding

Select **Time-resolved pairwise SVM** in the Concept decoding panel. The
controls below the decoder select the trials per supertrial (default 6) and
random repeats (default 100). Select the time window, channel set and existing
preprocessing controls as usual. The decoder uses raw samples regardless of
feature binning settings, and automatically selects signal space and the random
split setting. Its actual validation is leave-one-supertrial-per-concept-out;
the usual train/test percentage does not apply.

The Accuracy plot shows mean pairwise accuracy versus time. Metrics contains
that time course. Saved tables include each pair's accuracy at every time point,
the supertrial membership, and a trial-index lookup. The NPZ file contains
`accuracy[time, concept, concept]`, `times_s`, and ordered `class_names`. Matrices
are symmetric with undefined (NaN) diagonals. Chance accuracy is 0.5 regardless
of the total number of concepts. The summary accuracy averages over both time
and pairs; use the time course to inspect temporal effects.

## CLI

```sh
python scripts/run_concept_decoding_cli.py \
  --trial-table /path/to/trials.csv --output-dir /path/to/results \
  --decoder pairwise_svm --group-mode canonical \
  --window-start -0.1 --window-end 1.0 \
  --pairwise-supertrial-size 6 --pairwise-repeats 100
```

Optional `--pairwise-random-state` (default 42) and `--pairwise-svm-c` (default 1)
control reproducibility and linear SVM regularization. Use canonical object
labels to decode concepts, or individual labels to decode particular stimuli.
Category labels instead answer a broader category discrimination question.
Run each participant separately; this is not a cross-participant validation.

## Method and scope

Based on Bankson, Hebart, Groen & Baker (2018), *The temporal evolution of
conceptual object representations revealed through models of behavior,
semantics and deep neural networks*, NeuroImage 178, 172–182, section
“Multivariate MEG decoding” (attached paper, page 174).

For each concept pair and repetition, shuffle trials without replacement and
average disjoint groups of six. Use the same number K of supertrials for both
concepts, determined by the smaller trial count. Each of K folds holds out one
supertrial from each concept and trains on the remaining K−1 from each.
Fit an independent linear SVM at every time point, using channels as features.
Average accuracy over folds and 100 random regroupings, then average all unique
concept pairs for the time course. Regroupings and folds are shared across time
points so time comparisons use the same trials. Surplus trials are randomly
omitted per repetition. At least two supertrials per concept are required
(12 usable trials with the defaults); insufficient counts produce an error.
The split table's zero-based trial indices refer to the `pairwise_payload_index` column in `pairwise_trial_index.csv`.

This implements the paper's decoding design, with these preprocessing differences:
MAS retains its selected filtering, baseline correction, artifact rejection,
and optional epoch resampling. Channel standardization is fitted on training
supertrials only for each fold and time point. The paper instead used global
PCA, component baseline normalization, Gaussian temporal smoothing with 15 ms
half-duration at half-maximum, and downsampling to 120 Hz; this decoder does
not add those steps. Consequently this is not an exact preprocessing replication.

The usual permutation-count setting is not applied. The paper's significance
analysis used participant-level sign permutations and FDR correction; this
single-participant decoder does not compute significance or confidence bands.
Repeated regroupings are not independent participants. Temporal generalization
(training at one time and testing at another) is also outside this decoder.

Runtime scales with pairs × repeats × folds × time points. Use fewer repeats
for a smoke test, then restore 100 for the paper-style analysis. Downsampling
can reduce runtime, but changes temporal resolution.
