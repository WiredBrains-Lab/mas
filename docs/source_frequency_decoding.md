# Source-space frequency decoding

Concept decoding supports band-power features for both **Surface** and **Volume** source spaces. This is single-trial Hilbert band-power decoding; it does not add a Morlet transform or phase/connectivity features.

## Using the interface

1. In **Concept decoding**, select **Source parcels**, then **Surface** or **Volume** and the atlas/regions.
2. Under **Features**, choose **Binned band power (baseline dB)** or **Window band power (baseline dB)**. The log-power variants omit baseline normalization.
3. Select one frequency band or **All available frequency bands** in the selector below Features.
4. For binned features, set the temporal bin width in Shared preprocessing. Set the prestimulus baseline there for the dB variants.
5. Run a decoder supported by source space, including ordinary classifiers and the supported semantic embedding decoders. Parcel subset screening uses the same frequency features and retains their metadata.

The bands match signal-space semantic decoding: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, low gamma 30–70, and high gamma 70–150 Hz. These differ from the parcel-trial viewer's low-gamma display band (30–45 Hz). “All available” includes only complete bands that survive shared filtering and are below the sampling Nyquist frequency. An explicitly selected unavailable band raises an error.

## Computation

For each trial, the pipeline reconstructs signed currents for fixed-orientation inverses, or all three current components for loose/free-orientation inverses. Each source location is bandpass-filtered with a fourth-order Butterworth IIR filter using MNE's default zero-phase application, followed by the Hilbert transform. The squared analytic-envelope magnitudes are summed across vector components **after** the spectral transform.

Linear power is averaged over each requested time bin, then spatially averaged within the selected ROI. Surface power maps are morphed to common space when the existing source configuration requires it, using mean aggregation without sign flips. Volume power maps use the subject's volume atlas. This measures mean regional power rather than the power of a spatially averaged current waveform.

The dB representation is `10 * log10(bin_ROI_power / baseline_ROI_power)`, with a separate baseline for each trial, ROI, and band. Baseline samples precede the selected end time, which must be at or before zero. The log-power representation is `log10(bin_ROI_power)`. Sensor baseline subtraction and parcel waveform z-scoring are bypassed for these features. Power is computed before the decoder's within-split pseudo-trial averaging.

Output columns follow **ROI → band → time bin** order. Saved feature metadata includes ROI, band name, frequency limits, time boundaries, and normalization statistic. Decoder summaries record actual bands, padding, orientation handling, and aggregation method.

## Backend options

```python
options = {
    "feature_space": "source",
    "source_space_kind": "surface",  # or "volume"
    "feature_mode": "bandpower",
    "source_power_bands": ["alpha", "beta"],  # or "all" / "alpha,beta"
    "source_power_baseline": True,
    "parcel_baseline_start": -0.2,
    "parcel_baseline_end": 0.0,
    "feature_binning_enabled": True,  # False: one whole-window power value
    "bin_size_ms": 50,
    "window_start": 0.15,
    "window_end": 0.45,
}
```

These options go through the existing source-decoding backend/context. The CLI can now reconstruct trial sources using **saved inverse operators**, then extract the same features and run the decoder. It does not create new inverse operators or perform FreeSurfer reconstruction.

## Command-line runs

For one already-cleaned analysis FIF and its matching surface inverse:

```bash
python scripts/run_concept_decoding_cli.py --no-config \
  --fif /data/run_raw.fif --no-all-runs --ica off \
  --feature-space source --source-space-kind surface \
  --source-inverse /data/run-inv.fif \
  --source-subject subject01 --subjects-dir /data/subjects \
  --source-parcels ctx-lh-inferiortemporal ctx-lh-middletemporal \
  --feature-mode bandpower --source-power-bands alpha beta \
  --feature-binning --bin-size-ms 50 \
  --window-start 0.15 --window-end 0.45 \
  --bandpass-low-hz 0.1 --bandpass-high-hz 150 \
  --downsample --downsample-sfreq 400 \
  --decoder logistic --group-mode canonical --n-permutations 0 \
  --output-dir /data/results/source_alpha_beta --dry-run
```

Remove `--dry-run` to reconstruct and train/test. Use a fresh output directory,
or add `--force` to reuse the directory containing only your preflight outputs.
A dry run validates mappings,
inverse geometry/subject, channel availability, head transforms, atlas parcel
names, and frequency-band availability, and saves input counts/options. It does
not check every trial's source reconstruction. The inverse must be appropriate
for the supplied analysis FIF; these checks cannot establish its noise-covariance
provenance. Use cleaned FIFs matching the saved inverses; automatic ICA may change
the analysis FIF path and therefore its manifest entry.

For volume decoding, change `--source-space-kind` to `volume`, supply a volume
inverse, and select volume atlas names, for example `--source-parcels
Left-Hippocampus Right-Hippocampus`. Select regions covered by the inverse grid.
Volume mode uses aparc+aseg and retains the single-subject restriction.

- `--source-power-bands all` selects every fully supported band.
- `--no-source-power-baseline` selects log10 power; the default is baseline dB.
- `--parcel-baseline-start -0.2 --parcel-baseline-end 0` controls the dB baseline.
- `--no-feature-binning` produces whole-window power.
- For semantic decoding use `--decoder embedding_ridge` or
  `--decoder embedding_contrastive`, with `--embedding-dict /data/embeddings.json`.
- TOML users can copy [decoding.source-bandpower.toml](../configs/decoding.source-bandpower.toml)
  and fill in their paths; every option also works as a config key with underscores.
- Source mode additionally needs `h5py` for the shared semantic-decoder imports;
  volume atlas IO needs `nibabel`. Neither Qt nor the GUI is needed.

For multiple analysis FIFs, use `--source-manifest /data/source-runs.json` instead
of `--source-inverse`. The manifest explicitly pairs each run with its inverse
and anatomy:

```json
{
  "runs": [
    {"fif": "ICA/run1.fif", "inverse": "inverse/run1-inv.fif", "subject": "subject01", "subjects_dir": "subjects"},
    {"fif": "ICA/run2.fif", "inverse": "inverse/run2-inv.fif", "subject": "subject01", "subjects_dir": "subjects"}
  ]
}
```

Manifest paths are relative to the manifest file; `--path-map OLD=NEW` also
applies to them. Supply the corresponding FIF list, `--fif-folder`, or trial
table as usual. Every included analysis FIF requires an entry. `--source-subject`
and `--subjects-dir` provide defaults for entries that omit those fields.
Resolved mappings are saved in `run_options.json`; normal summary, predictions,
feature-importance, and model-architecture outputs are retained.

## Interpretation and limits

- Padding on each side is at least `max(0.5, 3 / lowest_band_edge)` seconds. An optional `source_power_padding_s` may increase it. Trials too close to recording boundaries are omitted. Epoch amplitude rejection uses the analysis/baseline interval rather than the additional padding.
- Time bins describe output sampling, not independent temporal resolution. Low-frequency filtering spreads information across time, including around stimulus onset and into nearby trials. Compare representations on consistent trial sets/splits and interpret latency accordingly.
- Power extraction is processed in vertex batches, but source reconstruction still consumes memory. The default source batch size is four trials for band power.
- Empty/zero-power selected ROIs and invalid baselines fail explicitly.
- Volume decoding retains the existing single-anatomical-subject restriction; surface decoding retains its common-space support.
- The existing transformer remains restricted to signal space. The frequency feature choice does not change decoder eligibility.

## Persistent all-ROI cache on laptop and HPC

Caching is enabled by default for GUI and CLI source decoding. No machine-specific
cache path is needed. For CIDMEG005 band-power features the default is:

```text
CIDMEG005/source_feature_cache/bandpower/<settings-and-input-hash>.joblib
```

Other feature modes use sibling folders such as `raw`, `mean`, `rms`, and `mean_rms`.
The data root is inferred from the FIF/anatomy layout; run subfolders such as `ICA`
are kept below the subject root. Multi-subject batches use their common data folder.
The HPC TOML leaves the directory unset:

```toml
source_feature_cache = true
rebuild_source_features = false
```

On the first matching run, the source backend extracts features for **all supported atlas
ROIs**, saves a compressed joblib entry on disk, then selects the requested
ROI columns (`source_parcels` in the CLI). Later jobs can select different ROIs or a different
screening pool without repeating source reconstruction. `max_rois` also applies
after loading. The screening pool is controlled by the ROI selection and optional GUI-equivalent
`--source-parcel-filter {all,visual,semantic,visual_semantic}` and
`--source-hemisphere {both,left,right}`. Explicit lists are narrowed by these filters.
The HPC TOML uses the semantic preset: 38 volume ROIs / 274,877,906,943 exhaustive
models. Add `--source-hemisphere left` for 19 ROIs / 524,287 models. These remain
large searches; use `--dry-run` to inspect the resolved pool and count. Filtering
does not restrict the all-ROI cache, which can serve later selections.

The cache persists across Python restarts and separate Slurm submissions. The
Slurm script uses the TOML automatically. Scratch retention/purge policies still
apply; move the cache directory to persistent project storage if needed. Entries
are not automatically pruned. They are trusted local joblib files.

Reuse requires matching FIF inputs (including split parts), inverse files,
anatomical files, trial tables/labels, atlas/source space, and extraction settings.
These include bandpass, downsampling, source frequency bands, time window, binning,
baseline, inverse method/SNR, and trial QC. SHA-256 hashes of input file contents
identify dependencies, including split FIF parts and anatomical files. Extraction
code and the cache format must match. Changed data or settings create a separate
entry; absolute paths, file timestamps, and producer library versions do not enter
the portable key. This cache contains final
ROI features for a particular window/binning, not arbitrary full-recording source
time courses.

Changing ROI selection, decoder, embedding dictionary, training epochs, model
hyperparameters, train/test split, PCA, averaging, or worker count reuses the
features. Embedding alignment and all train-dependent operations run again after
loading, using the current settings. The cache is saved before these operations.

ROIs without source support in any input run are excluded and recorded in the
cache payload and log. Explicitly requesting an unsupported ROI fails rather than
silently dropping it. Concurrent jobs with identical cache keys coordinate through
a file lock; completed entries are published atomically. Failed/partially skipped
extractions are not cached. A corrupt entry is rebuilt.

Use `--rebuild-source-features` to replace a matching entry, or
`--no-source-feature-cache` to bypass cache reads and writes. `--dry-run` validates
inputs without creating ROI features. Existing screening `--resume` continues to
use its batch-local `source_features.joblib`; it does not change extraction settings.
Both GUI and CLI runs support surface and volume caches. Common-space surface
caching requires fsaverage under the first run's subjects directory; otherwise
extraction proceeds without this cache. GUI loaded raw state is fingerprinted,
including signal data, bad channels and annotations, so unsaved changes invalidate
an entry. On a GUI source-cache miss, the underlying disk recording/epoch cache is
bypassed to ensure extraction uses those edits.

An explicit `source_feature_cache_dir` / `--source-feature-cache-dir` remains an
exact directory override. Existing entries in the former shared HPC cache are
left in place; new runs use the automatic data-local location.

### Copying a laptop cache to HPC

1. Use the updated MAS code on both machines. Generate a cache with the portable
   format (version 4) on the laptop.
2. Copy `CIDMEG005/source_feature_cache/` to the same location within CIDMEG005 on
   HPC, together with the matching FIF files (including split parts), inverses,
   and FreeSurfer anatomy. Preserve the data files themselves; their timestamps
   and parent directories may change.
3. Point the HPC TOML/manifest and trial table at the HPC inputs and use the same
   extraction settings. Submit the existing Slurm script normally.

The log reports `Reusing saved ROI features` when verification succeeds. Saved
paths in trial/QC tables, inverse mappings, and anatomy metadata are remapped to
the current inputs. ROI selection and training settings can differ as usual.
Equivalent GUI/CLI band selections and backend defaults are normalized.

Input verification reads and hashes files on each new process; this is I/O work,
but it does not rerun filtering, inverse reconstruction, or feature extraction.
Repeated checks within one process reuse hashes while file stat information stays
unchanged. Hidden OS metadata files such as `.DS_Store` are ignored in anatomy.

An unchanged GUI-loaded recording shares the disk/CLI identity. Unsaved signal,
bad-channel, annotation, or other raw-state edits remain distinct: a CLI job
reading the original unedited FIF cannot reuse features from those edits.

Producer Python/MNE/NumPy/pandas/SciPy versions are retained as provenance. A
version difference alone does not discard already computed arrays; the cache
must still deserialize and pass schema/shape checks. Incompatible or corrupt
entries are rebuilt. Extraction code changes also require rebuilding.

Caches written before portable format version 4 lack content fingerprints and
need one regeneration with the updated code before transfer. They are not
silently promoted based on matching filenames or timestamps.
