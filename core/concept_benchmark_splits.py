"""Frozen raw-trial partitions for matched concept-decoding benchmarks."""
import numpy as np
import pandas as pd


def validate_indices(split, y, *, zero_shot=False):
    """Reject overlap, missing rows, invalid indices and candidate loss."""
    y = np.asarray(y, dtype=str)
    parts = []
    for name in ("train", "validation", "test"):
        raw = np.asarray(split[name])
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or not raw.size:
            raise ValueError(f"{name} requires a nonempty integer index vector")
        parts.append(raw.astype(int))
    joined = np.concatenate(parts)
    if not np.array_equal(np.sort(joined), np.arange(len(y))):
        raise ValueError("Split must assign every retained trial exactly once")
    train, validation, test = parts
    train_labels, val_labels, test_labels = (set(y[p]) for p in parts)
    if train_labels != val_labels:
        raise ValueError("Training and validation must contain identical concepts")
    if zero_shot:
        if train_labels & test_labels or len(test_labels) < 2:
            raise ValueError("Zero-shot requires at least two disjoint test concepts")
    elif train_labels != test_labels:
        raise ValueError("Closed-set partitions must contain identical concepts")
    return train, validation, test


def _validation_by_image(frame, available, rng):
    train, validation = [], []
    for _, group in frame.loc[available].groupby("concept", sort=True):
        images = np.array(sorted(group.image.unique()))
        if len(images) < 2:
            raise ValueError("At least two training image identities per concept are required")
        rng.shuffle(images)
        selected = set(images[:max(1, int(round(len(images) * 0.2)))])
        mask = group.image.isin(selected)
        validation.extend(group.index[mask])
        train.extend(group.index[~mask])
    return sorted(train), sorted(validation)


def make_splits(frame, folds=5, seed=20261001):
    """Image CV, exhaustive disjoint concept folds and both session directions."""
    frame = frame.reset_index(drop=True)
    for name in ("concept", "image", "session"):
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {name} metadata")
    if (frame.groupby("image").concept.nunique() > 1).any():
        raise ValueError("An image identity maps to multiple concepts")
    rng = np.random.default_rng(seed)
    image_fold = {}
    for _, group in frame.groupby("concept", sort=True):
        images = np.array(sorted(group.image.unique()))
        if len(images) < folds:
            raise ValueError(f"Need >= {folds} distinct images per concept")
        rng.shuffle(images)
        image_fold.update({image: i % folds for i, image in enumerate(images)})
    concepts = np.array(sorted(frame.concept.unique()))
    if len(concepts) % folds:
        raise ValueError("Concept count must divide evenly into folds")
    rng.shuffle(concepts)
    splits = []

    def append(track, fold, test):
        available = frame.index.difference(test)
        train, validation = _validation_by_image(frame, available, np.random.default_rng(seed + len(splits)))
        item = dict(track=track, fold=str(fold), train=train, validation=validation, test=sorted(test))
        validate_indices(item, frame.concept, zero_shot=track == "zero_shot")
        for a, b in ((train, validation), (train, test), (validation, test)):
            if track != "session" and set(frame.loc[a, "image"]) & set(frame.loc[b, "image"]):
                raise ValueError("Image leakage across partitions")
        splits.append(item)

    for fold in range(folds):
        append("image", fold, frame.index[frame.image.map(image_fold).eq(fold)])
    for fold, unseen in enumerate(np.array_split(concepts, folds)):
        append("zero_shot", fold, frame.index[frame.concept.isin(unseen)])
    sessions = sorted(frame.session.unique())
    if len(sessions) != 2:
        raise ValueError(f"Two-session transfer expects exactly two sessions, found {sessions}")
    for session in sessions:
        append("session", f"test-{session}", frame.index[frame.session.eq(session)])
    return splits
