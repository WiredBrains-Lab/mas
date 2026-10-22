"""Target Embedding Residualization for concept decoding in MAS.

Removes task-demand, superordinate category, or semantic decision confounds
from target concept embeddings so that neural decoders predict concept-unique
features (e.g., airplane semantics) rather than the decision variable
(e.g., 'not an animal/insect' or vehicle centroid).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32)


def _build_confound_design_matrix(
    confounds: Sequence[Any] | np.ndarray,
    n_samples: int,
    known_categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Convert discrete labels or continuous vectors into a design matrix [N, K]."""
    if isinstance(confounds, np.ndarray) and confounds.ndim == 2:
        if confounds.shape[0] != n_samples:
            raise ValueError(
                f"Confound matrix shape {confounds.shape} does not match "
                f"sample count {n_samples}."
            )
        return confounds.astype(np.float64), None

    labels = [str(c).strip() for c in confounds]
    if len(labels) != n_samples:
        raise ValueError(
            f"Confound labels length ({len(labels)}) does not match "
            f"sample count ({n_samples})."
        )

    if known_categories is None:
        categories = sorted(list(dict.fromkeys(labels)))
    else:
        categories = list(known_categories)

    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    design = np.zeros((n_samples, len(categories)), dtype=np.float64)

    for i, label in enumerate(labels):
        if label in cat_to_idx:
            design[i, cat_to_idx[label]] = 1.0

    return design, categories


class TargetEmbeddingResidualizer:
    """Decomposes target embeddings into confound-explained and residual variance.

    Fits an Ordinary Least Squares (OLS) projection:
        Y = C * B + E
        Y_res = Y - C * B
    where C is the confound matrix (e.g., task decision, category one-hot,
    or instruction embedding) with an intercept term.
    """

    def __init__(self, include_intercept: bool = True, l2_normalize_output: bool = True):
        self.include_intercept = include_intercept
        self.l2_normalize_output = l2_normalize_output
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.categories_: list[str] | None = None
        self.variance_explained_: float = 0.0
        self.is_fitted_: bool = False

    def fit(
        self,
        embeddings: np.ndarray,
        confounds: Sequence[Any] | np.ndarray,
    ) -> TargetEmbeddingResidualizer:
        """Fit the confound projection matrix on training embeddings and confounds."""
        Y = np.asarray(embeddings, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {Y.shape}.")
        n_samples, n_dim = Y.shape

        C_raw, categories = _build_confound_design_matrix(confounds, n_samples)
        self.categories_ = categories

        if self.include_intercept:
            row_sums = np.sum(C_raw, axis=1)
            is_full_partition = np.allclose(row_sums, 1.0) and categories is not None
            if is_full_partition:
                C = C_raw
            else:
                C = np.hstack([np.ones((n_samples, 1), dtype=np.float64), C_raw])
        else:
            C = C_raw

        # OLS estimation via pseudo-inverse: B = (C^T C)^+ C^T Y
        C_pinv = np.linalg.pinv(C)
        B = C_pinv @ Y  # [K, D]
        self.coef_ = B.astype(np.float32)

        # Variance explained calculation
        Y_pred = C @ B
        total_var = float(np.sum(np.square(Y - np.mean(Y, axis=0, keepdims=True))))
        residual_var = float(np.sum(np.square(Y - Y_pred)))
        if total_var > 1e-12:
            self.variance_explained_ = max(0.0, float(1.0 - (residual_var / total_var)))
        else:
            self.variance_explained_ = 0.0

        self.is_fitted_ = True
        return self

    def transform(
        self,
        embeddings: np.ndarray,
        confounds: Sequence[Any] | np.ndarray,
    ) -> np.ndarray:
        """Project out the fitted confound from embeddings."""
        if not self.is_fitted_ or self.coef_ is None:
            raise RuntimeError("Residualizer is not fitted yet. Call fit() first.")

        Y = np.asarray(embeddings, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {Y.shape}.")
        n_samples, n_dim = Y.shape

        C_raw, _ = _build_confound_design_matrix(
            confounds, n_samples, known_categories=self.categories_
        )

        if self.include_intercept:
            row_sums = np.sum(C_raw, axis=1)
            is_full_partition = (
                np.allclose(row_sums, 1.0) and self.categories_ is not None
            )
            if is_full_partition:
                C = C_raw
            else:
                C = np.hstack([np.ones((n_samples, 1), dtype=np.float64), C_raw])
        else:
            C = C_raw

        if C.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Confound design matrix columns ({C.shape[1]}) does not match "
                f"fitted coefficient rows ({self.coef_.shape[0]})."
            )

        Y_pred = C @ self.coef_.astype(np.float64)
        Y_res = (Y - Y_pred).astype(np.float32)

        if self.l2_normalize_output:
            Y_res = _l2_normalize(Y_res)

        return Y_res

    def fit_transform(
        self,
        embeddings: np.ndarray,
        confounds: Sequence[Any] | np.ndarray,
    ) -> np.ndarray:
        """Fit and transform embeddings in one step."""
        return self.fit(embeddings, confounds).transform(embeddings, confounds)


def resolve_concept_confounds(
    names: Sequence[str],
    *,
    trial_df: Any = None,
    confound_source: str = "category",
    category_csv: str | Path | None = None,
    instruction_path: str | Path | None = None,
) -> dict[str, str]:
    """Resolve category or task-instruction confounds for a list of concept names.

    Parameters
    ----------
    names : Sequence[str]
        List of unique concept names to be embedded.
    trial_df : pd.DataFrame | None
        Current trial table containing stimulus and category/block columns.
    confound_source : str
        'category', 'instruction', or 'custom_csv'.
    category_csv : str | Path | None
        Optional path to concept-to-category mapping file (CSV/TSV).
    instruction_path : str | Path | None
        Optional path to block-instructions.csv.
    """
    mapping: dict[str, str] = {}
    source = str(confound_source or "category").strip().lower()

    # 1. Custom CSV mapping if provided
    csv_file = str(category_csv or "").strip()
    if csv_file and os.path.isfile(os.path.abspath(os.path.expanduser(csv_file))):
        clean_path = os.path.abspath(os.path.expanduser(csv_file))
        delimiter = "\t" if clean_path.endswith((".tsv", ".tab")) else ","
        with open(clean_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            if header:
                key_idx = 0
                val_idx = 1 if len(header) > 1 else 0
                header_lower = [h.strip().lower() for h in header]
                for idx, col in enumerate(header_lower):
                    if col in ("concept", "stimulus", "name", "canonical_stimulus", "label"):
                        key_idx = idx
                    elif col in ("category", "stimulus_group", "group", "class", "confound", "task"):
                        val_idx = idx

                for row in reader:
                    if len(row) > max(key_idx, val_idx):
                        k = str(row[key_idx]).strip()
                        v = str(row[val_idx]).strip()
                        if k and v:
                            mapping[k] = v
                            mapping[k.lower()] = v

    # 2. Block instructions mapping if requested
    if source == "instruction":
        from core.instruction_embeddings import read_instruction_csv, block_key
        inst_p = str(instruction_path or "").strip()
        if not inst_p or not os.path.isfile(os.path.abspath(os.path.expanduser(inst_p))):
            raise ValueError("Confound source 'instruction' requires a valid block-instructions.csv path.")
        inst_info = read_instruction_csv(inst_p)
        block_mapping = inst_info.get("block_mapping", {})

        if trial_df is not None and not getattr(trial_df, "empty", True):
            stim_col = next(
                (c for c in ("canonical_stimulus", "concept", "stimulus", "concept_label") if c in trial_df.columns),
                None,
            )
            if stim_col and "block_id" in trial_df.columns:
                for _, row in trial_df.iterrows():
                    stim = str(row.get(stim_col, "")).strip()
                    b_id = block_key(row.get("block_id", ""))
                    if stim and b_id in block_mapping:
                        mapping[stim] = block_mapping[b_id]
                        mapping[stim.lower()] = block_mapping[b_id]

    # 3. Stimulus categories from trial_df if source is category
    elif source == "category" and trial_df is not None and not getattr(trial_df, "empty", True):
        stim_col = next(
            (c for c in ("canonical_stimulus", "concept", "stimulus", "concept_label") if c in trial_df.columns),
            None,
        )
        cat_col = next(
            (c for c in ("stimulus_group", "category", "category_label", "group") if c in trial_df.columns),
            None,
        )
        if stim_col and cat_col:
            for _, row in trial_df.iterrows():
                stim = str(row.get(stim_col, "")).strip()
                cat = str(row.get(cat_col, "")).strip()
                if stim and cat and stim not in mapping:
                    mapping[stim] = cat
                    mapping[stim.lower()] = cat

    # Construct final mapping for all requested names
    out: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        str_name = str(name).strip()
        if str_name in mapping:
            out[str_name] = mapping[str_name]
        elif str_name.lower() in mapping:
            out[str_name] = mapping[str_name.lower()]
        else:
            missing.append(str_name)

    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"Could not resolve confound label for {len(missing)} concepts "
            f"(e.g., {sample}). Provide a Category CSV or select a trial summary with category columns."
        )

    distinct = set(out.values())
    if len(distinct) < 2:
        raise ValueError(
            f"Residualization requires at least 2 distinct confound classes, "
            f"but found only {len(distinct)}: {list(distinct)}."
        )

    return out


def residualize_embedding_dictionary(
    embedding_dict_path: str | Path,
    concept_to_confound: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a concept embedding JSON dictionary, remove confounds, and save/return it."""
    dict_path = Path(embedding_dict_path).expanduser().resolve()
    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names = data.get("names", [])
    raw_embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
    if raw_embeddings.ndim != 2 or raw_embeddings.shape[0] != len(names):
        raise ValueError("Invalid embedding dictionary format (names/embeddings mismatch).")

    confounds = []
    missing = []
    for name in names:
        if name in concept_to_confound:
            confounds.append(concept_to_confound[name])
        elif name.lower() in concept_to_confound:
            confounds.append(concept_to_confound[name.lower()])
        else:
            missing.append(name)

    if missing:
        raise ValueError(
            f"Missing confound mapping for {len(missing)} concepts. "
            f"First missing: {missing[:5]}"
        )

    residualizer = TargetEmbeddingResidualizer(include_intercept=True, l2_normalize_output=True)
    res_embeddings = residualizer.fit_transform(raw_embeddings, confounds)

    metadata = dict(data.get("metadata", {}))
    metadata["residualized"] = True
    metadata["residualization_variance_explained"] = residualizer.variance_explained_
    metadata["confound_categories"] = residualizer.categories_

    output_payload = {
        "names": names,
        "embeddings": res_embeddings.tolist(),
        "metadata": metadata,
    }

    if output_path:
        out_p = Path(output_path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

    return output_payload
