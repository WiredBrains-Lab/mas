# FIF-to-decoding CLI

Use Python 3.11 or newer, with the analysis environment you already use for MAS.
Commands below assume your terminal is in the MAS project directory. No Qt,
PyVista, window system, or GUI launch is needed by the CLI.

## First local run

Edit `configs/decoding.toml` and set:

```toml
fif = "/absolute/path/to/subject_s1_run1_raw.fif"
decoder = "logistic"
group_mode = "canonical"
```

Then run:

```bash
python scripts/run_concept_decoding_cli.py --prepare-only
python scripts/run_concept_decoding_cli.py --dry-run
python scripts/run_concept_decoding_cli.py
```

`--prepare-only` does ICA where applicable and creates/reuses trial tables. It
creates no decoder result directory. `--dry-run` also checks label counts and
writes run settings without training. It is not a read-only operation: missing
ICA derivatives and trial tables are prepared first.

The config lists decoder choices with comments. `--help` lists every CLI option.
To change inputs or models without editing config:

```bash
python scripts/run_concept_decoding_cli.py --fif "/data/run.fif" --decoder template
python scripts/run_concept_decoding_cli.py --fif "/data/run1.fif" "/data/run2.fif" --decoder logistic
python scripts/run_concept_decoding_cli.py --decoder embedding_ridge --embedding-dict "/data/embeddings.json"
```

## Process a folder

```bash
python scripts/run_concept_decoding_cli.py --fif_folder "/data/subject/tsss" --decoder logistic
python scripts/run_concept_decoding_cli.py --fif_folder "/data/subject/tsss" --prepare-only
```

`--fif-folder` is an equivalent spelling. In config, set
`fif_folder = "/data/subject/tsss"` and leave `fif` empty. Folder input replaces
configured FIF/table input and cannot be combined with explicit `--fif` or
`--trial-table`. It processes all readable raw `.fif` and `.fif.gz` recordings
in that folder, without prompting, regardless of `all_runs`. Subfolders are not
scanned. Non-raw/unreadable files are skipped with a message; an empty folder
or one containing no readable recordings fails clearly. Split parts are counted
once. Per-recording caches and the combined trial table work as described below.
Default decoder outputs go inside the folder's `decoding_results` directory.

## Include neighboring runs

With one FIF input in an interactive terminal, the CLI lists other raw FIF
recordings in that folder and asks whether to include them (`[y/N]`). This also
works when the single input comes from config. Enter `y` to include the list;
Enter, `n`, or end-of-input keeps only the selected recording. An explicit list
of multiple FIFs does not prompt.

For predictable unattended/RCC execution:

```bash
# Include raw recordings from the same folder, without a prompt.
python scripts/run_concept_decoding_cli.py --fif "/data/run1.fif" --all-runs

# Use exactly the explicit recording(s), without a prompt.
python scripts/run_concept_decoding_cli.py --fif "/data/run1.fif" --no-all-runs
```

`all_runs = true` or `all_runs = false` can also be set in TOML. If unset,
noninteractive input and Slurm jobs use only the explicit files and never wait
for a prompt. With multiple explicit inputs, `--all-runs` scans each input folder
(nonrecursively). Non-raw/unreadable neighboring FIF files are skipped with a
message; invalid explicitly selected files fail. FIF header links identify split
continuation files so they are not counted as extra runs.

Each selected analysis FIF keeps its own reusable trial-table cache. For multiple
recordings, the CLI also saves a combined snapshot beside the first analysis FIF:
`<first.fif>.combined-<selection-id>.trial_table.csv`. This is saved even with
`--prepare-only`. It contains every selected run's rows, preserving
`source_fif_path`, run/session labels and run-local `trial_index`; the additional
`combined_trial_index` uniquely numbers the combined rows. Each execution
rebuilds this snapshot from current per-run tables instead of appending to an
older snapshot, avoiding duplicate trials. Different selections have different
snapshot names. Decoder outputs also contain `input_trial_table.csv` with the
same combined rows and `run_options.json` records the requested/selected FIFs.

The default config is located relative to the script, independent of your working
directory. Paths inside a config are relative to that config file. Paths supplied
on the command line are relative to the terminal's current directory. Both accept
`~`; config paths also expand environment variables. Use absolute paths for batch
jobs. CLI flags replace corresponding config values, including input lists.
`--no-downsample` and `--feature-binning` disable the enabled config defaults.
For other enabled one-way boolean flags, set the value false in your config or
use `--no-config` to run solely from built-in defaults and explicit CLI arguments.

Without `--output-dir`, each decoding run gets a timestamped subfolder inside
`decoding_results` beside the selected input. It contains the combined input trial
table, resolved run options/provenance, and the selected decoder's CSV/JSON/plot
outputs. Existing named output directories are protected unless `--force` is set.

## Automatic trial tables in GUI and CLI

Both interfaces use `core/fif_processing.py` for event matching, ICA, trial scores,
and caching. The files are named:

```text
subject_s1_run1_raw.fif
subject_s1_run1_raw.fif.trial_table.csv
subject_s1_run1_raw.fif.trial_table.csv.meta.json
```

No trial-table path is required. Caches belong to the **analysis FIF**. With
`ica = "auto"`, a file in a `tsss`, `sss`, `maxfilter`, or `maxfiltered` folder
uses a cached ICA derivative in the sibling `ica` folder, matching GUI behavior.
Its trial table therefore lives in the `ica` folder beside that cleaned FIF.
`--ica off` uses the selected FIF directly. This does not perform tSSS/MaxFilter.

The GUI automatically creates/reuses the same table when building its stimulus
report. No manual export is necessary. Old exported tables without matching
metadata are not automatically trusted as caches; tables are rebuilt from FIF
triggers and matching event logs.

The metadata validates FIF size/modification time (including split FIF parts),
event-log contents, category-map contents, and the response-channel limit. A
checksum detects a damaged or incomplete CSV. Changed inputs invalidate the cache;
`--rebuild-trial-table` forces regeneration. Channel response scores survive the
CSV round trip for the GUI report. Cache files are atomically replaced, and the
data directory must be writable.

Keep this existing data layout, for example:

```text
subject/
  events_log/                  # acquisition event CSVs
  stimulus_with_groups.csv     # needed for category labels
  tsss/
    subject_s1_run1_raw.fif
  ica/                         # generated/reused automatically
```

Event logs may alternatively be in `events_log/` inside the FIF's folder. Matching
uses session/run, trigger counts, and timing; ambiguous or inconsistent logs are
rejected. Copy all split FIF parts. Preserve modification times when moving data
if you want immediate cache reuse; otherwise safe regeneration occurs. Automatic
FIF input rebases cached analysis paths to the new machine.

For legacy exported tables, `--trial-table` is still supported and bypasses FIF
preparation. Stored paths can be remapped with repeated `--path-map OLD=NEW` flags.

## Models and scope

- `logistic`, `template`, `mlp`, `pairwise_svm`: no semantic dictionary required.
- `embedding_ridge`, `embedding_contrastive`, `embedding_mlp`, `transformer`:
  require an embedding dictionary.

The CLI runs signal-space decoding and source-parcel decoding with saved inverse
operators and FreeSurfer anatomy. Source mode supports both surface and volume
band power. It does not build inverse operators or run FreeSurfer reconstruction.
Automatic ICA failures stop CLI jobs with an error instead of silently continuing
with uncleaned data.

For surface and volume band-power features in the GUI/source backend, see
[source frequency decoding](source_frequency_decoding.md).

## RCC

Copy the code, FIF files, event logs, category mapping and required embedding
files. Use a Linux Python environment on RCC; do not copy a macOS environment.
The lightweight baseline requirements are in `scripts/requirements-decoding.txt`.
Install PyTorch and external transformer code as needed for your chosen model and
RCC's CPU/GPU environment. Export package versions from the tested environment for
reproducibility.

Copy `configs/decoding.toml` to `configs/decoding.rcc.toml`, set RCC paths and use:

```bash
python scripts/run_concept_decoding_cli.py --config configs/decoding.rcc.toml --prepare-only
python scripts/run_concept_decoding_cli.py --config configs/decoding.rcc.toml --dry-run
```

For batch runs, edit the account/resources in `scripts/decoding.slurm`, activate
your Python environment, and submit from the MAS directory:

```bash
sbatch scripts/decoding.slurm
```

The job template uses Slurm CPUs to limit BLAS/OpenMP and Torch threads. Its
resource requests are examples, not measured requirements for your recordings.
Use Code Server for editing and interactive checks within its allocation; submit
long runs as batch jobs. RCC's current job instructions are at
https://docs.rcc.mcw.edu/cluster/quickstart/ .

## Reuse source ROI features between jobs

GUI and CLI source decoding enable a persistent all-ROI feature cache beside the
data by default, for example `CIDMEG005/source_feature_cache/bandpower`. The HPC
TOML no longer specifies a machine-specific cache directory. A fresh job reuses matching
features and selects the requested ROI columns, even when the screening pool or
training settings change. Changed preprocessing, time windows, bands, baselines,
anatomy, or inverse file contents require a new entry. Portable cache entries can
be copied with the data from laptop to HPC: paths are remapped and file contents
are verified automatically. Producer runtime versions are recorded as provenance. See
[cache settings and behavior](source_frequency_decoding.md#persistent-all-roi-cache-on-laptop-and-hpc).

## Start a parcel-screening batch from the CLI

Both surface and volume source features support fresh screening:

```bash
python -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.source-volume.hpc.toml \
  --subset-search exhaustive --subset-workers 4 --subset-worker-threads 1 \
  --dry-run
```

Remove `--dry-run` to run the fits. The volume TOML uses the GUI's semantic
preset: 38 ROIs for CIDMEG005, or 274,877,906,943 exhaustive models. This remains
a very large search. Add `--source-hemisphere left` for 19 ROIs / 524,287 models:

```bash
python -u scripts/run_concept_decoding_cli.py \
  --config configs/decoding.source-volume.hpc.toml \
  --source-parcel-filter semantic --source-hemisphere left --dry-run
```

`--source-parcel-filter` accepts `all`, `visual`, `semantic`, or `visual_semantic`,
using exactly the GUI's definitions for surface and volume atlases.
`--source-hemisphere` accepts `both` (default, including midline), `left`, or `right`.
Filters select from atlas labels shared across runs. If `--source-parcels NAME ...`
is also supplied, filters narrow that explicit list; names must be exact labels.
A filter matching no parcels fails rather than falling back to all ROIs.
The selected names and model count are printed before screening, and resolved
names are saved in run options. To restore all atlas labels, use
`--source-parcel-filter all --source-hemisphere both` without an explicit list.
Screening needs an explicit list or filter/hemisphere selection; `max_rois` must be zero.
Resume uses the saved candidate pool.

Equivalent TOML options are:

```toml
source_parcel_filter = "semantic"
source_hemisphere = "left"
subset_search = "exhaustive"
subset_workers = 4
subset_worker_threads = 1
```

The GUI and CLI have no parcel-count cap. They warn with the number of models
before running. Subsets are generated lazily, with at most the requested worker
count in flight; the entire search is never materialized in memory. Storage and
checkpoint bookkeeping still grow with completed fits. More workers do not
remove the exponential compute cost. Allocate at least workers × worker threads
CPUs in your HPC job, and allow memory for each worker's feature data/model.

Screening requires `feature_space = "source"`, `embedding_ridge` or
`embedding_contrastive`, and random or zero-shot splitting. Saved inverse files
are required for fresh CLI source extraction. Band-power and waveform features
use the same screening path. `--subset-search backward` also works, with one
worker and a positive validation fraction; backward checkpoints cannot resume.

A timestamped batch directory is created below the output directory. It contains
`manifest.json`, `settings_used.json`, `source_features.joblib`,
`subset_comparison.csv`, and per-subset results. Use that batch directory with
`--no-config --resume PATH` after an interruption. Existing checkpoint IDs and
subset ordering are preserved. The dry run validates inputs and prints the model
count without extracting source features or fitting models.

## Resume an exhaustive parcel-screening batch

```bash
python scripts/run_concept_decoding_cli.py --resume "/data/batch_folder" --dry-run
python scripts/run_concept_decoding_cli.py --resume "/data/batch_folder"
```

Copy the entire batch folder, including `manifest.json`, `settings_used.json`,
`source_features.joblib`, `subset_comparison.csv`, and completed subset folders.
Copy its original embedding dictionary too. Use checkpoints you created/trust:
joblib files contain serialized Python objects. Stop the original batch before
copying; run only one writer against a batch directory.

Resume uses saved screening settings and source features, not the current config's
decoder/preprocessing values. No FIF input, source reconstruction, or new output
directory is required. Outputs continue inside the resumed folder. Completed and
failed recorded subsets are skipped; an interrupted, unrecorded fit restarts.
Only exhaustive ridge/contrastive screening is supported, not backward elimination.

Existing dependency paths are retained. Explicit `--path-map` mappings are tried
first for missing paths. Only required dependencies (the embedding dictionary)
trigger automatic searches; missing historical FIF/anatomy paths never trigger
a scan. Required files are searched by exact filename in the batch and up to four nearby ancestor directories (excluding the
filesystem root and home). Batch-owned result paths are rebased to the resumed
folder. To search a different data location or restrict ambiguous matches:

```bash
python scripts/run_concept_decoding_cli.py --resume "/data/batch_folder" \
  --resume-search-root "/data/WBL" --dry-run
# Repeat --resume-search-root for multiple locations.
# Or specify a prefix mapping for missing paths:
python scripts/run_concept_decoding_cli.py --resume "/data/batch_folder" \
  --path-map "/old/Mac/WBL=/data/WBL"
```

A missing or ambiguous required embedding file stops before progress is changed.
Missing historical FIF/anatomy references may remain unchanged: cached-feature
resume does not need those files. Remapping happens in memory; archived settings
and source_features.joblib are preserved. The console reports recovered paths.
`--dry-run` validates checkpoint subset IDs, saved features and dependency recovery
without fitting or writing progress. It does not guarantee identical numerics
across different hardware or library versions; retain the tested environment
versions. `--torch-num-threads` can limit compute threads on RCC.

## Parallel exhaustive subset screening

```bash
python scripts/run_concept_decoding_cli.py \
  --resume "/data/batch_folder" \
  --subset-workers 4 \
  --subset-worker-threads 1
```

Default `--subset-workers 1` preserves sequential fitting and its existing device
selection. With more than one worker, exhaustive subsets run in separate **CPU**
processes; each worker limits BLAS and Torch to `--subset-worker-threads` (default
1). This mode does not launch competing jobs on the same GPU. These runtime
settings override saved batch runtime settings without changing model/split
settings. `--torch-num-threads` controls the main process, not parallel workers.

For RCC request at least workers × worker-threads CPUs in the Slurm job (the
example needs four CPUs), and enough RAM for each worker's feature payload and
model. The payload is copied into each process once; it is not shared memory.
Parallel execution is local to one allocated node, not a multi-node job array.
Backward elimination is not supported in parallel mode.

Workers write distinct numbered result folders. Only the coordinator writes the
comparison CSV and manifest. Each finished subset is checkpointed immediately;
completion order can differ from subset order. Checkpoints now support missing
IDs between completed subsets, so resuming (with any worker count) runs only
unrecorded IDs. Older sequential checkpoints still work. Earlier versions of
MAS that require a contiguous prefix cannot resume a partially finished parallel
checkpoint; copy the updated code to RCC.

A lock prevents two coordinators from writing the same resumed batch. Ctrl-C
stops scheduling new subsets and allows active fits to finish and be saved. A
hard stop can leave unrecorded fits, which are rerun on resume. As before, failed
fits recorded in the comparison CSV are skipped. Resume still needs only the four
checkpoint files and embedding dictionary; old numbered result folders are
optional.

Progress after each subset includes `completed subsets/min (session avg)` in
both sequential and parallel mode. This is successful fits completed during the
current invocation divided by elapsed fitting-phase time, summed across workers.
It excludes prior checkpoint results and failed fits. The clock starts after
feature preparation, includes worker startup and output saving, and resets on
resume. Allow several completions before comparing CPU/GPU throughput.

## Create the RCC Conda environment

Copy `scripts/setup_decoding_env.sh` and
`scripts/requirements-decoding-rcc.txt` with the updated MAS code, then run from
Code Server with Conda available:

```bash
bash scripts/setup_decoding_env.sh
conda activate mas-decoding
python scripts/run_concept_decoding_cli.py --resume "/data/batch_folder" --dry-run
```

This creates a new Python 3.11 environment, installs pinned analysis dependencies
and PyTorch 2.8.0 from the official CUDA 12.8 wheel index, checks imports and Torch
operations, and records resolved packages in
`$CONDA_PREFIX/mas-decoding-pip-freeze.txt`. Core package versions match the tested
local environment; transitive dependencies are resolved during installation.
Base and existing named environments are not modified. If a partial installation
needs retrying, choose a new `--name` or deliberately remove the incomplete
environment yourself first.

Inside a GPU allocation, add `--require-gpu` to require a successful GPU operation.
Without it, installation on a login/CPU node is allowed and reports that CUDA is
not currently available. GPU availability and driver compatibility are verified
on the allocated node; this script does not install drivers. Alternative official
Torch 2.8 wheel builds are `--torch-build cu126`, `cu129`, or `cpu`. For example:

```bash
bash scripts/setup_decoding_env.sh --name mas-decoding-cpu --torch-build cpu
```

The selected CUDA builds follow the [official PyTorch 2.8 installation commands](https://pytorch.org/get-started/previous-versions/#v280).
The transformer model still needs the external CIDECoG source tree discussed above;
the script installs its Python dependencies, not that separately maintained code.
In Code Server, select the `mas-decoding` Python interpreter if using the Run button.
Terminal commands and batch jobs should activate the same environment.

If an older installer attempts to write under `/hpc/apps/miniforge`, copy the
updated `scripts/setup_decoding_env.sh` and run it with a fresh name, for example
`--name mas-decoding-rcc`. It creates `~/.conda/envs/mas-decoding-rcc` and uses
that environment's absolute Python path for every install. It checks Python,
pip and site-packages locations before installing and disables pip destination
overrides. Activate with `conda activate "$HOME/.conda/envs/mas-decoding-rcc"`.
Do not use `--user` to work around a base-environment permission error.

## Parallel screening in the GUI

In the Concept decoder panel, set **Subset workers** and **Threads / worker**.
Enable **Selected parcel subsets + auto-save** and start decoding, or select
**Resume…** to continue an existing exhaustive batch. The current worker controls
override saved runtime settings without changing the batch's model/split settings.
One worker retains normal CPU/GPU selection; more than one runs CPU fits in
parallel. Backward elimination and ordinary single-decoder runs remain sequential.
Controls are locked during a run and restored afterward. **Stop batch** stops new
submissions and drains active fits, saving their results. The status line shows
completed subsets per minute. Live per-epoch plots are unavailable for parallel
fits; per-subset training metrics are saved in each result folder.

Parallel completion updates also show `fit ETA` and `best top-1`. ETA uses the
current session's aggregate attempted-fit throughput across workers and the
number of unrecorded subsets remaining. It includes failed attempts (which are
not retried), while completed subsets/min counts successes only. Best top-1
includes successful fits saved before resuming, even if their result folders
have been deleted. ETA is approximate because fit time varies by subset size.

### Parallel source extraction on HPC

`--source-workers N` extracts independent runs in separate processes; the default
is 1 and the effective count is capped at the number of selected runs.
`--source-worker-threads T` limits each process's numerical threads (default 1).
Trial/class and ROI selection remain global, rejection/outlier QC remains per run,
and feature rows are combined in original run/trial order. Parallel extraction
uses saved FIF inputs and a headless CLI context.

The source-volume Slurm script requests one 48-CPU node and 7 GB RAM per CPU
(336 GB total on the 360 GB nodes). It uses 12 source
processes with 4 numerical threads each, followed by 48 subset processes with 1
thread each. These stages run sequentially. The source-band filtering itself
still uses one job; extra numerical threads can accelerate inverse matrix work.
Set the Slurm memory request to cover the peak worker working sets; CPU counts
alone do not establish a safe memory limit. Override the request with
`sbatch --mem-per-cpu=... scripts/decoding.source-volume.slurm` after checking node RAM.
Worker defaults can be overridden with `MAS_SOURCE_WORKERS`,
`MAS_SOURCE_WORKER_THREADS`, `MAS_SUBSET_WORKERS`, and
`MAS_SUBSET_WORKER_THREADS` via `sbatch --export=ALL,...`.

When the persistent source-feature cache is enabled, each successful run is saved
atomically under `source_feature_cache/<mode>/runs/<analysis-key>/`. Restart the
same analysis to reuse completed runs, even if the final all-run cache was never
written. These intermediate checkpoints are host-local; the final feature cache
remains portable. `--rebuild-source-features` bypasses both caches. Worker counts
and thread counts do not change the cache identity. Batch and run completion
messages show progress during extraction.
