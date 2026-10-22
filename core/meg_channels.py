"""Shared MEG channel selection helpers."""

from pathlib import Path

import mne
import numpy as np


def good_meg_picks(info, meg=True, ref_meg=False):
    """Return MEG picks excluding channels marked in ``info['bads']``.

    Returns
    -------
    picks : ndarray
        MNE channel indices for usable MEG channels.
    excluded_bad_names : list of str
        Marked-bad MEG channel names excluded for this ``meg`` selection.
    """
    if info is None:
        return np.array([], dtype=int), []
    all_picks = mne.pick_types(info, meg=meg, ref_meg=ref_meg, exclude=[])
    picks = mne.pick_types(info, meg=meg, ref_meg=ref_meg, exclude="bads")
    try:
        bad_names = set(str(name) for name in info.get("bads", []) if str(name).strip())
        ch_names = list(info["ch_names"])
    except Exception:
        bad_names = set()
        ch_names = []
    good_set = set(int(idx) for idx in np.asarray(picks, dtype=int).tolist())
    excluded = []
    for idx in np.asarray(all_picks, dtype=int).tolist():
        try:
            name = str(ch_names[int(idx)])
        except Exception:
            continue
        if int(idx) not in good_set and name in bad_names:
            excluded.append(name)
    return np.asarray(picks, dtype=int), excluded


def good_meg_channel_names(info, meg=True, ref_meg=False):
    """Return usable MEG channel names and marked-bad MEG names."""
    picks, excluded = good_meg_picks(info, meg=meg, ref_meg=ref_meg)
    try:
        ch_names = list(info["ch_names"])
    except Exception:
        ch_names = []
    names = []
    for idx in np.asarray(picks, dtype=int).tolist():
        try:
            names.append(str(ch_names[int(idx)]))
        except Exception:
            continue
    return names, excluded


def format_bad_meg_channel_message(context, fif_path=None, bad_names=None, max_names=12):
    """Build a concise log message for marked bad MEG channels."""
    bad_names = [str(name) for name in (bad_names or []) if str(name).strip()]
    if not bad_names:
        return ""
    label = str(context or "MEG")
    if fif_path:
        try:
            label = f"{label} {Path(fif_path).name}"
        except Exception:
            label = f"{label} {fif_path}"
    shown = ", ".join(bad_names[: int(max_names)])
    if len(bad_names) > int(max_names):
        shown = f"{shown}, ..."
    return f"{label}: excluding {len(bad_names)} marked bad MEG channel(s): {shown}"


def emit_bad_meg_channel_log(logger, context, fif_path=None, bad_names=None):
    """Emit a marked-bad-channel log message if a logger is available."""
    message = format_bad_meg_channel_message(context, fif_path=fif_path, bad_names=bad_names)
    if not message:
        return
    try:
        logger.emit(message)
    except Exception:
        try:
            logger(message)
        except Exception:
            print(message)
