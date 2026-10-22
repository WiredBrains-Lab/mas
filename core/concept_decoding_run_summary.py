"""Shared, human-readable decoding settings for fresh and resumed runs."""


def log_decoding_run_summary(options, progress, *, saved=False):
    space = options.get("feature_space", "signal")
    mode = options.get("feature_mode", "raw")
    geometry = options.get("source_space_kind", "surface")
    description = f"source space ({geometry})" if space == "source" else f"{space} space"
    split = "zero_shot" if options.get("zero_shot") else options.get("split_mode", "random")
    origin = "saved checkpoint settings" if saved else "configured settings"
    progress(f"Decoding run summary ({origin}): {description}; feature mode={mode}; "
             f"decoder={options.get('decoder_type', 'unspecified')}; split={split}.")
    start, end = options.get("window_start", .15), options.get("window_end", .45)
    bins = (f"{options.get('bin_size_ms', 50):g} ms bins"
            if options.get("feature_binning_enabled", False) else "binning disabled")
    progress(f"Decoding window: {1000 * float(start):g} to {1000 * float(end):g} ms; {bins}.")
    if space == "source":
        parc = options.get("source_parcellation", "aparc")
        atlas = (f"{parc}+aseg" if parc in {"HCPMMP1", "HCPMMP1_combined"} else "aparc+aseg") if geometry == "volume" else parc
        parcels = options.get("selected_parcellations") or []
        pool = f"{len(parcels)} selected parcels" if parcels else "all available parcels"
        progress(f"Source settings: atlas={atlas}; {pool}; inverse method={options.get('method', 'MNE')}.")
        if mode == "bandpower":
            bands = options.get("source_power_bands", "all")
            if not isinstance(bands, str):
                bands = ", ".join(map(str, bands))
            normalization = "baseline dB" if options.get("source_power_baseline", True) else "log10 power"
            progress(f"Source band power: requested bands={bands}; normalization={normalization}; "
                     "selected bands combined within each model.")
    else:
        progress(f"Signal channels: {options.get('channel_mode', 'all')}.")
