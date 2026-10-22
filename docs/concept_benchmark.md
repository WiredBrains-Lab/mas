# Matched concept-decoding benchmark

The configuration is `configs/concept_benchmark.json`. It already points to the
local CIDMEG005 data, saved trial metadata, and text-embedding dictionary. No
manual trial export is needed. The input table contains 5,037 source trials,
50 concepts, 1,788 stimulus filenames, and two sessions. Extraction applies fixed
epoch rejection and retains the intersection of usable trials across atlases.
The final retained count therefore comes from `plan.json`, not the historical
7,133-trial result. This is a new matched benchmark, not a reproduction of those
historical headline values.

## Run locally

The verified local environment uses Python 3.11 with the MAS requirements.
From the MAS directory, one command prepares features, runs the linear baselines
first, runs the contrastive comparisons, and collects results:

```bash
python scripts/run_concept_benchmark.py all
```

Run this in a terminal that will remain open, with sleep disabled for long jobs.
Re-running the command resumes completed extraction and fits. It does not
restart completed models. Failed or interrupted fits restart from the beginning
of that fit. A hard-killed process can leave a `running.lock`; check that its
process is gone before removing that lock.

To include the optional 990 refitted null fits after the 128 main fits:

```bash
python scripts/run_concept_benchmark.py all --include-nulls
```

Individual stages and a one-fit real-data pilot:

```bash
python scripts/run_concept_benchmark.py preflight
python scripts/run_concept_benchmark.py prepare
python scripts/run_concept_benchmark.py run --decoder embedding_ridge --limit 1
python scripts/run_concept_benchmark.py run
python scripts/run_concept_benchmark.py run --stage null
python scripts/run_concept_benchmark.py collect
```

Use the verified interpreter above in place of `python` if your shell selects
another environment. `--track image`, `--track zero_shot`, or `--track session`
restricts `run`. Every stage accepts `--config /path/to/config.json`. Relative
configuration paths resolve against the config directory.

## Prespecified comparisons

| Track | Anatomical configurations | Outer folds | Contrastive seeds | Ridge fits | Main fits |
|---|---|---|---|---|---|
| Unseen images | HCP five, semantic six, semantic three, semantic four | 5 | 3 | 20 | 80 |
| Unseen concepts | Left aparc 34, semantic six | 5, ten test concepts each | 3 | 10 | 40 |
| Session transfer | Semantic six | Both directions | 3 | 2 | 8 |

The six-parcel configuration is the prespecified reference. Historical three-
and four-parcel winners remain exploratory comparisons because prior results
informed their selection. There is no new test-driven parcel search.

All configurations use 100-400 ms, 5 ms mean features, three trials per average,
the same text embeddings and matched raw-trial membership. All 34 aparc parcels
come from the reference run's saved settings; the script rejects missing parcels.
The HCP configuration contains MT+ complex and neighboring visual areas,
superior parietal, lateral temporal, inferior parietal and medial temporal cortex.

Fixed sensor epoch-rejection thresholds and trial-local baseline normalization
remain enabled. Adaptive across-trial outlier exclusion is disabled so test
trials do not set a data-dependent rejection threshold. ICA/inverse operators
are existing preprocessing artifacts; this benchmark does not refit them within
folds. Keep that limitation when interpreting strict prospective session transfer.

Image identity is the exact stimulus filename, globally across runs/sessions.
If distinct filenames encode the same underlying image, canonicalize that identity
before interpreting this as unseen-image evaluation. The script does not infer
visual duplicates from filename suffixes. Validation holds out image identities
within the outer training data. Zero-shot test concepts never enter fitting or
validation. Session transfer can reuse image identities across sessions and
must be described as session transfer, not simultaneously unseen-image transfer.

Partitions are frozen on raw trials before averaging. PCA, scaling, target
centering and SVD fit on training data within the existing decoder. Early
stopping selects validation MPR, with validation loss breaking ties. Ridge alpha
is prespecified at 1.0. Candidate sets are fixed within each track: 50 for
image/session, 10 for zero-shot. No language model or generated images enter
these comparisons.

## Saved evidence

Results live in `outputs/concept_benchmark_2026-10-01/`:

- `provenance.json`: resolved settings, input file identities, package versions
  and source-code hashes. Changed inputs/code/settings require a new output folder.
- `matched_trials.csv` and `plan.json`: exact trial IDs, image/session labels,
  raw indices for train/validation/test, QC counts, and complete job manifest.
- `features/`: per-run extraction checkpoints and combined features for each atlas.
- `runs/`: settings, predictions, confusion matrices, per-concept metrics,
  training histories, and completion markers for every fit.
- `report/run_status.csv` and `comparison_coverage.csv`: completion and missing fits.
- `report/all_runs.csv`, `fold_metrics.csv`, `summary.csv`, `seed_variation.csv`:
  all metrics, seed-averaged folds, fold mean/SD, and within-fold seed variation.
- `report/paired_differences.csv`: differences from the semantic-six reference
  using the same track, fold, decoder and seed.
- `report/per_concept_pooled.csv` and `concepts_*.png`: pooled per-concept MPR and
  top-1 plots for each configuration. Counts do not multiply by the seed count.
- `report/null_tests.csv`: null p-values only after all prescribed permutations
  and corresponding reference folds finish.

The collector intentionally permits partial reports for progress monitoring.
Use `comparison_coverage.csv` before presenting a completed comparison. Fold SD
is descriptive; folds share training data, and seeds are not independent subjects.
Session directions are available separately in `fold_metrics.csv`.

## Refitted null stage

For the semantic-six contrastive reference at seed 42, each of 99 permutations
shuffles the concept-to-text-target mapping consistently in train/validation,
leaves test targets unchanged, and refits the complete training pipeline.
Both image and zero-shot tracks use all five folds: 990 extra fits.
The collector compares the observed fold-mean MPR with the 99 complete null
fold means using `(1 + exceedances) / (1 + permutations)`.
This null assumes exchangeability across concept identities. It is distinct
from the old fixed-prediction test-label shuffle, which is disabled here.
It tests the chosen reference only, not every historical model or participant.

## Remaining work

The real-data preflight and synthetic fitting/resume tests are short. Source
reconstruction and the full benchmark are long-running and must finish before
we know the matched scores or runtime. No new acquisition is needed for these
offline comparisons. New cue-free recordings or new participants are separate
experiments. Image-generation controls remain a separate pipeline and are not
included in the 128-fit decoding benchmark.
