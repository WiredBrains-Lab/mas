# Temporal decoding on an HPC

MAS exposes two complementary command-line analyses.

## 1. Time-resolved pairwise decoding

This fits an independent linear SVM at every sample and writes
`timecourse.csv`, pair-level accuracies, an NPZ matrix, and a plot. One job
covers the complete time range:

```bash
python -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.hpc.toml \
  --decoder pairwise_svm \
  --group-mode semantic_decision \
  --window-start -0.2 --window-end 0.6 \
  --pairwise-supertrial-size 6 \
  --pairwise-repeats 100 \
  --output-dir /scratch/$USER/semantic_decision_pairwise
```

Use `--group-mode canonical` for the 50 concepts. Fifty concepts require 1,225
pairs and can be much more expensive than the five semantic-decision labels;
use a small repeat count for a smoke test before the final run.

## 2. Fixed-width sliding-window sweep

`run_temporal_window_sweep.py` runs the normal decoder in overlapping windows.
Arguments after `--` go to `run_concept_decoding_cli.py`. The wrapper controls
the window bounds and output directory.

First create the manifest:

```bash
python scripts/run_temporal_window_sweep.py \
  --sweep-output /scratch/$USER/decision_windows \
  --sweep-start -0.2 --sweep-stop 0.6 \
  --window-width-ms 50 --step-ms 20 \
  --manifest-only \
  -- --config configs/decoding.hpc.toml
```

The command prints the required zero-based Slurm array range. The defaults
above produce 38 tasks (`0-37`). Copy and edit
`scripts/temporal_decoding_array.slurm`, especially the account, array range,
output path, resources, and decoding config, then submit it:

```bash
sbatch scripts/temporal_decoding_array.slurm
```

Each task has a deterministic directory containing the ordinary decoder
outputs, its exact `command.txt`, and `temporal_task_status.json`. Tasks reuse
the persistent `epoch_cache_dir` configured in `decoding.hpc.toml`; put that
cache on fast scratch storage shared by the compute nodes.

After the array completes, collect all metrics into one table:

```bash
python scripts/run_temporal_window_sweep.py \
  --sweep-output /scratch/$USER/decision_windows \
  --collect-only
```

Collection reads the saved manifest, so the sweep bounds do not need to be
repeated.

The combined result is `temporal_window_summary.csv`. Pending and failed tasks
remain explicit in its `status` column, so a failed array element can be
resubmitted by its index.

For a local sequential smoke test, omit `--task-index` and
`SLURM_ARRAY_TASK_ID`; the wrapper runs every window in order. Add
`--continue-on-error` if desired.

## Recommended configurations

For semantic-decision labels, compare the poststimulus curve with the
prestimulus windows. The label is a block instruction, so prestimulus decoding
measures persistent task/block state rather than stimulus-evoked processing.
The current semantic embedding decoder supports random and zero-shot splits;
the supplied HPC config therefore uses `split_mode = "random"`. Session-held-out
embedding validation is not implemented yet.

Because ANIMAL or INSECT occurs in two of the six blocks, the supplied HPC
config enables `balance_classes = true`. Each training epoch uses the same
number of pseudo-epochs per label and dynamically rotates through surplus
trials from the larger class. Validation and test observations are not
discarded. `embedding_validation_metric = "balanced_mpr"` selects the best
training epoch using a macro-average across labels.

For canonical concepts, use the same decoder, input trials, split mode, seed,
and preprocessing for every window. A grouped split is preferable when the
scientific question requires generalization across runs, sessions, or repeated
exemplars.

Window sweeps quantify when a decoder trained on a particular interval works.
They do not yet perform temporal generalization, held-out time-bin occlusion,
or multiple-comparison-corrected inference.
