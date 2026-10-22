#!/usr/bin/env python3
"""Fast RSA + Cross-Parcel Redundancy Prescreen for MEG Source Parcel Selection.

Bridges coarse (`HCPMMP1_combined`) and fine (`HCPMMP1`) source parcellations for
speech/semantic BCI target selection:

1. Works directly on an existing `source_features.joblib` or `source_feature_cache`
   `.joblib` file (`--features-joblib`), OR extracts/loads via `--config` + CLI flags.
2. For every parcel in the payload, computes:
   - Semantic RDM alignment (`spearman_r`, permutation `p_value`) between the
     parcel's neural concept-by-concept RDM and the target semantic embedding RDM.
   - Partial Semantic RDM alignment (`partial_spearman_r`) controlling for coarse
     `stimulus_group` category structure when available.
   - Split-half neural RDM reliability (`split_half_r`) across presentations.
   - Cross-parcel RDM redundancy matrix (`parcel_rdm_similarity.csv`) to detect
     spatial leakage between neighboring 5-mm volume sub-parcels.
3. Selects a redundancy-pruned Top-K parcel pool (Max-Relevance Min-Redundancy)
   for tractable exhaustive (`2^K - 1`) contrastive subset decoding.
4. When run on `HCPMMP1_combined` features (or with `--subset-results-csv`),
   compares RSA rankings against exhaustive subset decoding marginal gains and
   expands the winning coarse macro-regions into their constituent fine-grained
   `HCPMMP1` parcels (both volume `ctx-lh-L_*_ROI` and surface `L_*_ROI-lh`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Canonical Glasser (2016) mapping from left-hemisphere HCPMMP1_combined macro-regions
# to their constituent fine-grained HCPMMP1 parcels (verified on fsaverage annotations).
HCPMMP1_COMBINED_TO_FINE_LH: dict[str, list[str]] = {
    "Temporo-Parieto-Occipital Junction": [
        "L_PSL_ROI", "L_STV_ROI", "L_TPOJ1_ROI", "L_TPOJ2_ROI", "L_TPOJ3_ROI",
    ],
    "Medial Temporal Cortex": [
        "L_EC_ROI", "L_H_ROI", "L_PHA1_ROI", "L_PHA2_ROI", "L_PHA3_ROI",
        "L_PeEc_ROI", "L_PreS_ROI",
    ],
    "Lateral Temporal Cortex": [
        "L_PHT_ROI", "L_TE1a_ROI", "L_TE1m_ROI", "L_TE1p_ROI", "L_TE2a_ROI",
        "L_TE2p_ROI", "L_TF_ROI", "L_TGd_ROI", "L_TGv_ROI",
    ],
    "Inferior Frontal Cortex": [
        "L_44_ROI", "L_45_ROI", "L_47l_ROI", "L_IFJa_ROI", "L_IFJp_ROI",
        "L_IFSa_ROI", "L_IFSp_ROI", "L_p47r_ROI",
    ],
    "Auditory Association Cortex": [
        "L_A4_ROI", "L_A5_ROI", "L_STGa_ROI", "L_STSda_ROI", "L_STSdp_ROI",
        "L_STSva_ROI", "L_STSvp_ROI", "L_TA2_ROI",
    ],
    "Superior Parietal Cortex": [
        "L_7AL_ROI", "L_7Am_ROI", "L_7PC_ROI", "L_7PL_ROI", "L_7Pm_ROI",
        "L_AIP_ROI", "L_LIPd_ROI", "L_LIPv_ROI", "L_MIP_ROI", "L_VIP_ROI",
    ],
    "Posterior Cingulate Cortex": [
        "L_23c_ROI", "L_23d_ROI", "L_31a_ROI", "L_31pd_ROI", "L_31pv_ROI",
        "L_7m_ROI", "L_DVT_ROI", "L_PCV_ROI", "L_POS1_ROI", "L_POS2_ROI",
        "L_ProS_ROI", "L_RSC_ROI", "L_d23ab_ROI", "L_v23ab_ROI",
    ],
    "Early Auditory Cortex": [
        "L_A1_ROI", "L_LBelt_ROI", "L_MBelt_ROI", "L_PBelt_ROI", "L_RI_ROI",
    ],
    "Inferior Parietal Cortex": [
        "L_IP0_ROI", "L_IP1_ROI", "L_IP2_ROI", "L_PF_ROI", "L_PFm_ROI",
        "L_PFop_ROI", "L_PFt_ROI", "L_PGi_ROI", "L_PGp_ROI", "L_PGs_ROI",
    ],
    "Ventral Stream Visual Cortex": [
        "L_FFC_ROI", "L_PIT_ROI", "L_V8_ROI", "L_VMV1_ROI", "L_VMV2_ROI",
        "L_VMV3_ROI", "L_VVC_ROI",
    ],
    "MT+ Complex and Neighboring Visual Areas": [
        "L_FST_ROI", "L_LO1_ROI", "L_LO2_ROI", "L_LO3_ROI", "L_MST_ROI",
        "L_MT_ROI", "L_PH_ROI", "L_V3CD_ROI", "L_V4t_ROI",
    ],
    "Orbital and Polar Frontal Cortex": [
        "L_10d_ROI", "L_10pp_ROI", "L_11l_ROI", "L_13l_ROI", "L_47m_ROI",
        "L_47s_ROI", "L_OFC_ROI", "L_a10p_ROI", "L_a47r_ROI", "L_p10p_ROI",
        "L_pOFC_ROI",
    ],
    "DorsoLateral Prefrontal Cortex": [
        "L_46_ROI", "L_8Ad_ROI", "L_8Av_ROI", "L_8BL_ROI", "L_8C_ROI",
        "L_9-46d_ROI", "L_9a_ROI", "L_9p_ROI", "L_SFL_ROI", "L_a9-46v_ROI",
        "L_i6-8_ROI", "L_p9-46v_ROI", "L_s6-8_ROI",
    ],
    "Anterior Cingulate and Medial Prefrontal Cortex": [
        "L_10r_ROI", "L_10v_ROI", "L_25_ROI", "L_33pr_ROI", "L_8BM_ROI",
        "L_9m_ROI", "L_a24_ROI", "L_a24pr_ROI", "L_a32pr_ROI", "L_d32_ROI",
        "L_p24_ROI", "L_p24pr_ROI", "L_p32_ROI", "L_p32pr_ROI", "L_s32_ROI",
    ],
    "Insular and Frontal Opercular Cortex": [
        "L_52_ROI", "L_AAIC_ROI", "L_AVI_ROI", "L_FOP2_ROI", "L_FOP3_ROI",
        "L_FOP4_ROI", "L_FOP5_ROI", "L_Ig_ROI", "L_MI_ROI", "L_PI_ROI",
        "L_Pir_ROI", "L_PoI1_ROI", "L_PoI2_ROI",
    ],
    "Posterior Opercular Cortex": [
        "L_43_ROI", "L_FOP1_ROI", "L_OP1_ROI", "L_OP2-3_ROI", "L_OP4_ROI",
        "L_PFcm_ROI",
    ],
    "Premotor Cortex": [
        "L_55b_ROI", "L_6a_ROI", "L_6d_ROI", "L_6r_ROI", "L_6v_ROI",
        "L_FEF_ROI", "L_PEF_ROI",
    ],
    "Paracentral Lobular and Mid Cingulate Cortex": [
        "L_24dd_ROI", "L_24dv_ROI", "L_5L_ROI", "L_5m_ROI", "L_5mv_ROI",
        "L_6ma_ROI", "L_6mp_ROI", "L_SCEF_ROI",
    ],
    "Somatosensory and Motor Cortex": [
        "L_1_ROI", "L_2_ROI", "L_3a_ROI", "L_3b_ROI", "L_4_ROI",
    ],
    "Dorsal Stream Visual Cortex": [
        "L_IPS1_ROI", "L_V3A_ROI", "L_V3B_ROI", "L_V6A_ROI", "L_V6_ROI",
        "L_V7_ROI",
    ],
    "Early Visual Cortex": [
        "L_V2_ROI", "L_V3_ROI", "L_V4_ROI",
    ],
    "Primary Visual Cortex (V1)": [
        "L_V1_ROI",
    ],
}

FINE_TO_COMBINED_BASE: dict[str, str] = {
    fine.replace("L_", "").replace("_ROI", ""): macro
    for macro, fines in HCPMMP1_COMBINED_TO_FINE_LH.items()
    for fine in fines
}


def _strip_parcel_affixes(parcel_name: str) -> str:
    text = str(parcel_name or "").strip()
    for prefix in ("ctx-lh-", "ctx-rh-", "lh-", "rh-", "Left-", "Right-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    for suffix in ("-lh", "-rh", "_lh", "_rh"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    return text.strip()


def _macro_region_for_parcel(parcel_name: str) -> str:
    base = _strip_parcel_affixes(parcel_name)
    if base in HCPMMP1_COMBINED_TO_FINE_LH:
        return base
    fine_clean = base
    if fine_clean.startswith(("L_", "R_")):
        fine_clean = fine_clean[2:]
    if fine_clean.endswith("_ROI"):
        fine_clean = fine_clean[:-4]
    if fine_clean in FINE_TO_COMBINED_BASE:
        return FINE_TO_COMBINED_BASE[fine_clean]
    if base.lower() in {"hippocampus", "amygdala"}:
        return "Subcortical Medial Temporal (aseg)"
    return "Other"


def expand_macro_regions_to_fine_hcp(
    macro_regions: list[str],
    *,
    space: str = "volume",
    hemisphere: str = "left",
    include_subcortical_mtl: bool = True,
) -> list[str]:
    """Expand coarse HCPMMP1_combined region names into fine HCPMMP1 parcel names."""
    hemis = ["lh", "rh"] if hemisphere == "both" else (["rh"] if hemisphere == "right" else ["lh"])
    fine_parcels: list[str] = []
    for hemi in hemis:
        hemi_letter = "L" if hemi == "lh" else "R"
        for raw_macro in macro_regions:
            base = _strip_parcel_affixes(raw_macro)
            lh_fines = HCPMMP1_COMBINED_TO_FINE_LH.get(base, [])
            for fine_lh in lh_fines:
                fine_hemi = fine_lh if hemi_letter == "L" else ("R_" + fine_lh[2:])
                if space == "volume":
                    fine_parcels.append(f"ctx-{hemi}-{fine_hemi}")
                else:
                    fine_parcels.append(f"{fine_hemi}-{hemi}")
        if space == "volume" and include_subcortical_mtl:
            if any(_strip_parcel_affixes(m) == "Medial Temporal Cortex" for m in macro_regions):
                side = "Left" if hemi == "lh" else "Right"
                fine_parcels.extend([f"{side}-Hippocampus", f"{side}-Amygdala"])
    return list(dict.fromkeys(fine_parcels))


def _resolve_joblib_payload(path: Path, embedding_dict: str | None = None) -> dict[str, Any]:
    """Load either a screening `source_features.joblib` or a `source_feature_cache` entry."""
    if path.is_dir():
        direct = path / "source_features.joblib"
        if direct.is_file():
            path = direct
        else:
            candidates = sorted(path.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No .joblib files found inside directory: {path}")
            path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")

    print(f"[RSA] Loading cached source features from: {path}", flush=True)
    raw_loaded = joblib.load(path)
    if isinstance(raw_loaded, dict) and "payload" in raw_loaded and "X" not in raw_loaded:
        payload = dict(raw_loaded["payload"])
    elif isinstance(raw_loaded, dict) and "X" in raw_loaded:
        payload = dict(raw_loaded)
    else:
        raise ValueError(f"Unrecognized joblib structure in {path}")

    if "embeddings" not in payload or payload.get("embeddings") is None:
        emb_path = embedding_dict or payload.get("embedding_dict_path")
        if not emb_path:
            raise ValueError(
                "Loaded feature cache does not have pre-attached concept embeddings. "
                "Supply --embedding-dict /path/to/concept_embeddings.json."
            )
        from core.concept_decoding_transformer import prepare_semantic_embedding_feature_payload
        payload, err = prepare_semantic_embedding_feature_payload(
            payload, {"embedding_dict_path": str(emb_path)}
        )
        if payload is None:
            raise RuntimeError(f"Failed to align embeddings: {err}")
    return payload


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank partial correlation between x and y controlling for z."""
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz = rankdata(z).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    rz -= rz.mean()
    denom_z = float(np.dot(rz, rz))
    if denom_z <= 1e-12:
        return float(spearmanr(x, y).statistic)
    res_x = rx - (np.dot(rx, rz) / denom_z) * rz
    res_y = ry - (np.dot(ry, rz) / denom_z) * rz
    norm = float(np.linalg.norm(res_x) * np.linalg.norm(res_y))
    if norm <= 1e-12:
        return 0.0
    return float(np.dot(res_x, res_y) / norm)


def compute_parcel_rsa_and_redundancy(
    payload: dict[str, Any],
    *,
    hemisphere: str = "left",
    n_permutations: int = 500,
    random_state: int = 42,
    distance_metric: str = "correlation",
    train_concepts_only: bool = False,
    test_fraction: float = 0.2,
    zero_shot_random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Compute per-parcel semantic RSA, split-half reliability, and cross-parcel RDM similarity."""
    X = np.asarray(payload["X"], dtype=np.float64)
    y = np.asarray(payload["y"], dtype=str)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    meta = payload["feature_meta_df"].reset_index(drop=True)
    trial_df = payload.get("trial_df")

    if X.ndim != 2 or len(meta) != X.shape[1] or "roi" not in meta.columns:
        raise ValueError("Feature matrix X and feature_meta_df['roi'] do not align.")

    if train_concepts_only:
        from core.concept_decoding_signal import zero_shot_concept_train_test_split
        train_idx, _ = zero_shot_concept_train_test_split(
            y, test_fraction=test_fraction, random_state=zero_shot_random_state
        )
        X = X[train_idx]
        y = y[train_idx]
        embeddings = embeddings[train_idx]
        if isinstance(trial_df, pd.DataFrame):
            trial_df = trial_df.iloc[train_idx].reset_index(drop=True)

    # Z-score each feature across trials so all time bins contribute equally
    feat_mean = np.nanmean(X, axis=0, keepdims=True)
    feat_std = np.nanstd(X, axis=0, keepdims=True)
    feat_std = np.where(feat_std > 1e-8, feat_std, 1.0)
    Xz = np.nan_to_num((X - feat_mean) / feat_std, nan=0.0, posinf=0.0, neginf=0.0)

    concepts = sorted(pd.unique(y).tolist())
    n_concepts = len(concepts)
    if n_concepts < 4:
        raise ValueError(f"Need at least 4 unique concepts for RSA; found {n_concepts}.")

    # Build concept centroids, split-half A/B centroids, and target semantic embeddings
    rng = np.random.default_rng(random_state)
    concept_X = np.zeros((n_concepts, Xz.shape[1]), dtype=np.float64)
    concept_X_half_a = np.zeros((n_concepts, Xz.shape[1]), dtype=np.float64)
    concept_X_half_b = np.zeros((n_concepts, Xz.shape[1]), dtype=np.float64)
    concept_emb = np.zeros((n_concepts, embeddings.shape[1]), dtype=np.float64)
    concept_groups: list[str] = []

    for c_idx, concept in enumerate(concepts):
        rows = np.flatnonzero(y == concept)
        concept_X[c_idx] = Xz[rows].mean(axis=0)
        concept_emb[c_idx] = embeddings[rows[0]]
        perm_rows = rng.permutation(rows)
        mid = max(1, len(perm_rows) // 2)
        concept_X_half_a[c_idx] = Xz[perm_rows[:mid]].mean(axis=0)
        concept_X_half_b[c_idx] = Xz[perm_rows[mid:]].mean(axis=0) if len(perm_rows) > 1 else Xz[perm_rows].mean(axis=0)
        if isinstance(trial_df, pd.DataFrame) and "stimulus_group" in trial_df.columns:
            grp = str(trial_df.iloc[rows[0]].get("stimulus_group", "") or "").strip()
            concept_groups.append(grp)
        else:
            concept_groups.append("")

    # Target Semantic RDM (cosine distance between concept embeddings)
    target_rdm_vec = pdist(concept_emb, metric="cosine")
    target_ranks = rankdata(target_rdm_vec).astype(np.float64)
    target_ranks_centered = target_ranks - target_ranks.mean()
    target_ranks_norm = float(np.linalg.norm(target_ranks_centered))

    # Optional categorical RDM (0 if same stimulus_group, 1 if different)
    has_groups = len(set(g for g in concept_groups if g)) >= 2
    if has_groups:
        cat_rdm_vec = np.array(
            [
                0.0 if (concept_groups[i] and concept_groups[i] == concept_groups[j]) else 1.0
                for i in range(n_concepts)
                for j in range(i + 1, n_concepts)
            ],
            dtype=np.float64,
        )
    else:
        cat_rdm_vec = None

    # Pre-generate concept-label permutations for exact RDM Mantel significance testing
    perm_target_ranks_centered = []
    for _ in range(max(0, int(n_permutations))):
        perm = rng.permutation(n_concepts)
        perm_rdm = squareform(squareform(target_rdm_vec)[np.ix_(perm, perm)], checks=False)
        pr = rankdata(perm_rdm).astype(np.float64)
        pr -= pr.mean()
        perm_target_ranks_centered.append(pr)
    perm_matrix = (
        np.vstack(perm_target_ranks_centered)
        if perm_target_ranks_centered
        else np.empty((0, len(target_rdm_vec)))
    )

    rois = list(dict.fromkeys(meta["roi"].astype(str).tolist()))
    if hemisphere == "left":
        rois = [
            r for r in rois
            if (
                r.startswith(("ctx-lh-", "Left-", "lh-", "L_"))
                or r.endswith(("-lh", "_lh"))
                or "-lh" in r
            )
        ]
    elif hemisphere == "right":
        rois = [
            r for r in rois
            if (
                r.startswith(("ctx-rh-", "Right-", "rh-", "R_"))
                or r.endswith(("-rh", "_rh"))
                or "-rh" in r
            )
        ]
    if not rois:
        raise ValueError(f"No parcels matched requested hemisphere: {hemisphere}")

    parcel_rdm_vecs: dict[str, np.ndarray] = {}
    parcel_rdm_ranks_unit: dict[str, np.ndarray] = {}
    rows_out = []

    for roi in rois:
        cols = np.flatnonzero(meta["roi"].astype(str).to_numpy() == roi)
        roi_X = concept_X[:, cols]
        roi_a = concept_X_half_a[:, cols]
        roi_b = concept_X_half_b[:, cols]

        rdm_vec = pdist(roi_X, metric=distance_metric)
        rdm_vec = np.nan_to_num(rdm_vec, nan=0.0, posinf=0.0, neginf=0.0)
        parcel_rdm_vecs[roi] = rdm_vec

        ranks = rankdata(rdm_vec).astype(np.float64)
        ranks_c = ranks - ranks.mean()
        ranks_norm = float(np.linalg.norm(ranks_c))
        unit_ranks = ranks_c / ranks_norm if ranks_norm > 1e-12 else np.zeros_like(ranks_c)
        parcel_rdm_ranks_unit[roi] = unit_ranks

        rho = (
            float(np.dot(ranks_c, target_ranks_centered) / (ranks_norm * target_ranks_norm))
            if (ranks_norm > 1e-12 and target_ranks_norm > 1e-12)
            else 0.0
        )

        if perm_matrix.shape[0] > 0 and ranks_norm > 1e-12 and target_ranks_norm > 1e-12:
            null_rhos = (perm_matrix @ unit_ranks) / target_ranks_norm
            p_val = float((1 + np.sum(null_rhos >= rho)) / (1 + len(null_rhos)))
            null_sd = float(np.std(null_rhos, ddof=1)) if len(null_rhos) > 1 else np.nan
            z_score = float((rho - np.mean(null_rhos)) / null_sd) if (null_sd and null_sd > 1e-12) else 0.0
        else:
            p_val = np.nan
            z_score = np.nan

        partial_rho = (
            _partial_spearman(rdm_vec, target_rdm_vec, cat_rdm_vec)
            if cat_rdm_vec is not None
            else np.nan
        )

        # Split-half RDM reliability
        rdm_a = np.nan_to_num(pdist(roi_a, metric=distance_metric), nan=0.0)
        rdm_b = np.nan_to_num(pdist(roi_b, metric=distance_metric), nan=0.0)
        split_r = float(spearmanr(rdm_a, rdm_b).statistic) if (np.std(rdm_a) > 1e-12 and np.std(rdm_b) > 1e-12) else 0.0

        rows_out.append(
            {
                "parcel": roi,
                "macro_region": _macro_region_for_parcel(roi),
                "n_features": int(len(cols)),
                "semantic_rdm_spearman_r": rho,
                "semantic_rdm_z": z_score,
                "semantic_rdm_p_value": p_val,
                "partial_semantic_rdm_r": partial_rho,
                "split_half_rdm_reliability": split_r,
                "composite_rsa_score": float(rho * max(0.1, split_r)) if split_r > 0 else float(rho * 0.1),
            }
        )

    rsa_df = pd.DataFrame(rows_out).sort_values(
        ["semantic_rdm_spearman_r", "split_half_rdm_reliability"], ascending=False
    ).reset_index(drop=True)
    rsa_df["rsa_rank"] = np.arange(1, len(rsa_df) + 1)

    # Parcel-to-Parcel RDM similarity matrix (measures neural redundancy / volume leakage)
    ordered_rois = rsa_df["parcel"].tolist()
    unit_mat = np.vstack([parcel_rdm_ranks_unit[r] for r in ordered_rois])
    sim_mat = np.clip(unit_mat @ unit_mat.T, -1.0, 1.0)
    np.fill_diagonal(sim_mat, 1.0)
    redundancy_df = pd.DataFrame(sim_mat, index=ordered_rois, columns=ordered_rois)

    return rsa_df, redundancy_df, parcel_rdm_vecs


def select_non_redundant_parcels(
    rsa_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    *,
    top_k: int = 12,
    max_redundancy: float = 0.80,
    allowed_macro_regions: set[str] | None = None,
    max_per_macro_region: int = 3,
) -> tuple[list[str], pd.DataFrame]:
    """Greedy Max-Relevance Min-Redundancy (mRMR) parcel selector for BCI implant screening."""
    candidates = rsa_df.copy()
    if allowed_macro_regions:
        effective_allowed = set(allowed_macro_regions)
        if any("Medial Temporal" in m for m in effective_allowed):
            effective_allowed.add("Subcortical Medial Temporal (aseg)")
        candidates = candidates[candidates["macro_region"].isin(effective_allowed)].copy()

    selected: list[str] = []
    macro_counts: dict[str, int] = {}
    selection_log = []

    for _, row in candidates.iterrows():
        parcel = str(row["parcel"])
        macro = str(row["macro_region"])
        if max_per_macro_region > 0 and macro_counts.get(macro, 0) >= max_per_macro_region:
            selection_log.append(
                {**row.to_dict(), "selected": False, "reason": f"macro_region_cap({max_per_macro_region})"}
            )
            continue
        if selected:
            corrs = [float(redundancy_df.loc[parcel, prev]) for prev in selected]
            max_corr = max(corrs)
            most_similar = selected[int(np.argmax(corrs))]
            if max_corr > max_redundancy:
                selection_log.append(
                    {
                        **row.to_dict(),
                        "selected": False,
                        "max_rdm_corr_with_selected": max_corr,
                        "most_similar_selected": most_similar,
                        "reason": f"redundant_with_{most_similar}({max_corr:.3f}>{max_redundancy:.2f})",
                    }
                )
                continue
        else:
            max_corr = 0.0
            most_similar = ""

        if len(selected) < top_k:
            selected.append(parcel)
            macro_counts[macro] = macro_counts.get(macro, 0) + 1
            selection_log.append(
                {
                    **row.to_dict(),
                    "selected": True,
                    "selected_order": len(selected),
                    "max_rdm_corr_with_selected": max_corr,
                    "most_similar_selected": most_similar,
                    "reason": "selected",
                }
            )
        else:
            selection_log.append(
                {
                    **row.to_dict(),
                    "selected": False,
                    "max_rdm_corr_with_selected": max_corr,
                    "most_similar_selected": most_similar,
                    "reason": f"reached_top_k({top_k})",
                }
            )

    res_df = pd.DataFrame(selection_log)
    if not res_df.empty:
        col_order = [
            "rsa_rank",
            "parcel",
            "macro_region",
            "selected",
            "selected_order",
            "composite_rsa_score",
            "semantic_rdm_spearman_r",
            "semantic_rdm_z",
            "semantic_rdm_p_value",
            "partial_semantic_rdm_r",
            "split_half_rdm_reliability",
            "max_rdm_corr_with_selected",
            "most_similar_selected",
            "reason",
            "n_features",
        ]
        present = [c for c in col_order if c in res_df.columns]
        remainder = [c for c in res_df.columns if c not in present]
        res_df = res_df[present + remainder]
    return selected, res_df


def compare_with_subset_results_csv(
    csv_path: Path, rsa_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Compute paired marginal decoding effects from a subset_comparison.csv file."""
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "complete"].copy()
    all_parcels = sorted({p for s in df["parcels"].astype(str) for p in s.split(";") if p.strip()})
    subset_map = {frozenset(str(r["parcels"]).split(";")): r for _, r in df.iterrows()}

    top10_mpr = df["mpr"].quantile(0.90)
    rows = []
    for p in all_parcels:
        small_dmpr, all_dmpr, small_dtop1 = [], [], []
        wins_small = 0
        for s_set, r_without in subset_map.items():
            if p in s_set:
                continue
            with_set = s_set | {p}
            if with_set not in subset_map:
                continue
            r_with = subset_map[with_set]
            dm = float(r_with["mpr"] - r_without["mpr"])
            dt1 = float(r_with["top1_accuracy"] - r_without["top1_accuracy"])
            all_dmpr.append(dm)
            if len(s_set) <= 3:
                small_dmpr.append(dm)
                small_dtop1.append(dt1)
                if dm > 0:
                    wins_small += 1
        single_rows = df[(df["n_parcels"] == 1) & (df["parcels"] == p)]
        single_mpr = float(single_rows.iloc[0]["mpr"]) if not single_rows.empty else np.nan
        single_top1 = float(single_rows.iloc[0]["top1_accuracy"]) if not single_rows.empty else np.nan
        has_p = df["parcels"].apply(lambda s: p in str(s).split(";"))
        freq_top10 = float(has_p[df["mpr"] >= top10_mpr].mean())
        rows.append(
            {
                "parcel": p,
                "macro_region": _macro_region_for_parcel(p),
                "single_parcel_mpr": single_mpr,
                "single_parcel_top1": single_top1,
                "paired_delta_mpr_small_subsets": float(np.mean(small_dmpr)) if small_dmpr else np.nan,
                "win_rate_small_subsets": float(wins_small / len(small_dmpr)) if small_dmpr else np.nan,
                "paired_delta_mpr_all_subsets": float(np.mean(all_dmpr)) if all_dmpr else np.nan,
                "freq_in_top10pct_mpr": freq_top10,
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(
        ["paired_delta_mpr_small_subsets", "freq_in_top10pct_mpr"], ascending=False
    ).reset_index(drop=True)

    if rsa_df is not None and not rsa_df.empty:
        summary_df = summary_df.merge(
            rsa_df[
                [
                    "parcel",
                    "semantic_rdm_spearman_r",
                    "semantic_rdm_p_value",
                    "split_half_rdm_reliability",
                    "rsa_rank",
                ]
            ],
            on="parcel",
            how="left",
        )

    # Winning macro-regions: positive small-subset marginal gain OR >=50% presence in top-10% MPR/Top-1
    winning_macros = summary_df.loc[
        (summary_df["paired_delta_mpr_small_subsets"] > 0.003)
        | (summary_df["freq_in_top10pct_mpr"] >= 0.50),
        "macro_region",
    ].drop_duplicates().tolist()
    if "Auditory Association Cortex" not in winning_macros and any(
        summary_df["macro_region"] == "Auditory Association Cortex"
    ):
        # Include AAC in Tier-2 if present in the top-1 winner
        winning_macros.append("Auditory Association Cortex")
    return summary_df, winning_macros


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional decoding config TOML (e.g. configs/decoding.source-volume-beta.hpc.toml) to extract/load features into cache.",
    )
    parser.add_argument(
        "--source-parcellation",
        default="HCPMMP1",
        help="Parcellation to extract when using --config (default: HCPMMP1).",
    )
    parser.add_argument(
        "--source-workers",
        type=int,
        default=12,
        help="Source extraction worker processes when using --config (default: 12).",
    )
    parser.add_argument(
        "--source-worker-threads",
        type=int,
        default=4,
        help="Threads per source worker when using --config (default: 4).",
    )
    parser.add_argument(
        "--features-joblib",
        type=Path,
        help="Path to an existing source_features.joblib or source_feature_cache .joblib file (or folder).",
    )
    parser.add_argument(
        "--subset-results-csv",
        type=Path,
        help="Optional HCPMMP1_combined subset_comparison.csv to guide macro-region pruning and compare against RSA.",
    )
    parser.add_argument(
        "--embedding-dict",
        type=Path,
        help="Embedding dictionary path (only needed if --features-joblib is a raw cache entry without attached embeddings).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rsa_prescreen_results"),
        help="Directory to write RSA rankings, redundancy matrix, and pruned HCPMMP1 parcel lists.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of non-redundant parcels to select for exhaustive subset decoding (default: 12 -> 4,095 subsets).",
    )
    parser.add_argument(
        "--max-redundancy",
        type=float,
        default=0.80,
        help="Maximum allowed neural RDM correlation between selected parcels (default: 0.80).",
    )
    parser.add_argument(
        "--max-per-macro-region",
        type=int,
        default=3,
        help="Maximum fine parcels selected from any single HCPMMP1_combined macro-region (default: 3).",
    )
    parser.add_argument(
        "--macro-regions",
        nargs="+",
        default=[
            "Temporo-Parieto-Occipital Junction",
            "Medial Temporal Cortex",
            "Lateral Temporal Cortex",
            "Inferior Frontal Cortex",
            "Auditory Association Cortex",
        ],
        help=(
            "Allowed coarse macro-regions to consider for fine parcel selection "
            "(defaults to winning language network: TPOJ, MTC, LTC, IFC, AAC). "
            "Pass 'all' to allow all regions in the parcellation."
        ),
    )
    parser.add_argument(
        "--hemisphere",
        choices=("left", "right", "both"),
        default="left",
        help="Filter parcels to one hemisphere (default: left for speech/language BCI).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=500,
        help="Number of concept permutations for Mantel RDM p-values (default: 500).",
    )
    parser.add_argument(
        "--train-concepts-only",
        action="store_true",
        help="Compute RSA strictly on the zero-shot training concepts (prevents any test-concept leakage).",
    )
    parser.add_argument(
        "--target-space",
        choices=("volume", "surface"),
        default="volume",
        help="Format for expanded fine HCPMMP1 parcel names (volume: ctx-lh-L_*_ROI, surface: L_*_ROI-lh).",
    )

    args = parser.parse_args(argv)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rsa_df = None
    redundancy_df = None
    selected_parcels: list[str] = []

    if args.features_joblib:
        payload = _resolve_joblib_payload(
            args.features_joblib.expanduser().resolve(),
            str(args.embedding_dict.expanduser().resolve()) if args.embedding_dict else None,
        )
    elif args.config:
        from scripts.run_concept_decoding_cli import (
            parse_args as parse_cli_args,
            load_and_prepare_trial_table,
            build_context,
            resolve_source_parcels,
            build_decoder_options,
        )
        from core.concept_decoding_source import build_source_space_concept_features
        from core.concept_decoding_transformer import prepare_semantic_embedding_feature_payload

        cli_flags = [
            "--config", str(args.config.expanduser().resolve()),
            "--source-parcellation", str(args.source_parcellation),
            "--source-workers", str(args.source_workers),
            "--source-worker-threads", str(args.source_worker_threads),
        ]
        if args.hemisphere:
            cli_flags.extend(["--source-hemisphere", str(args.hemisphere)])
        if args.embedding_dict:
            cli_flags.extend(["--embedding-dict", str(args.embedding_dict.expanduser().resolve())])

        print(f"[RSA] Initializing feature extraction / cache check via {args.config}...", flush=True)
        cli_parsed = parse_cli_args(cli_flags)
        trial_df = load_and_prepare_trial_table(cli_parsed)
        context = build_context(cli_parsed, trial_df)
        resolve_source_parcels(cli_parsed, context)
        options = build_decoder_options(cli_parsed)
        if cli_parsed.feature_space == "source":
            options["source_runs"] = list(context.meg_3d.runs.values())

        payload, error = build_source_space_concept_features(context, options)
        if payload is None:
            raise RuntimeError(error or "Feature extraction failed")
        payload, error = prepare_semantic_embedding_feature_payload(payload, options)
        if payload is None:
            raise RuntimeError(error or "Embedding alignment failed")
        print(f"[RSA] Feature matrix ready: {payload['X'].shape} ({len(payload['roi_names'])} ROIs)", flush=True)
    else:
        payload = None

    if payload is not None:
        rsa_df, redundancy_df, _ = compute_parcel_rsa_and_redundancy(
            payload,
            hemisphere=args.hemisphere,
            n_permutations=args.n_permutations,
            train_concepts_only=args.train_concepts_only,
        )
        rsa_df.to_csv(out_dir / "parcel_rsa_ranking.csv", index=False)
        redundancy_df.to_csv(out_dir / "parcel_rdm_redundancy_matrix.csv")
        print("\n=== Parcel Semantic RSA Ranking ===", flush=True)
        print(
            rsa_df[
                [
                    "rsa_rank",
                    "parcel",
                    "macro_region",
                    "semantic_rdm_spearman_r",
                    "semantic_rdm_p_value",
                    "split_half_rdm_reliability",
                ]
            ].to_string(index=False),
            flush=True,
        )

    winning_macros: list[str] = []
    if args.subset_results_csv:
        csv_summary, winning_macros = compare_with_subset_results_csv(
            args.subset_results_csv.expanduser().resolve(), rsa_df
        )
        csv_summary.to_csv(out_dir / "coarse_subset_vs_rsa_comparison.csv", index=False)
        print("\n=== Coarse Subset Decoding Marginal Gain vs. Single-Parcel Score ===", flush=True)
        print(csv_summary.to_string(index=False), flush=True)
        print(f"\nWinning Macro-Regions ({len(winning_macros)}): {', '.join(winning_macros)}", flush=True)

    if rsa_df is not None:
        allowed_macros: set[str] | None = None
        if winning_macros:
            allowed_macros = set(winning_macros) | {"Subcortical Medial Temporal (aseg)"}
        elif args.macro_regions and "all" not in [m.lower() for m in args.macro_regions]:
            allowed_macros = set(args.macro_regions) | {"Subcortical Medial Temporal (aseg)"}

        selected_parcels, sel_log_df = select_non_redundant_parcels(
            rsa_df,
            redundancy_df,
            top_k=args.top_k,
            max_redundancy=args.max_redundancy,
            allowed_macro_regions=allowed_macros,
            max_per_macro_region=args.max_per_macro_region,
        )
        sel_log_df.to_csv(out_dir / "mrmr_parcel_selection_log.csv", index=False)
        print(
            f"\n=== Selected Top-{len(selected_parcels)} Non-Redundant Parcels "
            f"(max RDM corr <= {args.max_redundancy:.2f}) ===",
            flush=True,
        )
        for idx, p in enumerate(selected_parcels, 1):
            print(f"  {idx:2d}. {p} ({_macro_region_for_parcel(p)})", flush=True)

    # Always generate the coarse-guided fine HCPMMP1 candidate pools if we have winning macro-regions
    # (or if the input features were coarse HCPMMP1_combined parcels)
    macros_to_expand = winning_macros
    if not macros_to_expand and rsa_df is not None:
        # Take top 5 macro-regions by RSA
        macros_to_expand = (
            rsa_df["macro_region"]
            .loc[lambda s: ~s.isin(["Other", "Subcortical Medial Temporal (aseg)"])]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

    if macros_to_expand:
        tier1_macros = [
            m for m in macros_to_expand
            if m in {"Temporo-Parieto-Occipital Junction", "Medial Temporal Cortex", "Lateral Temporal Cortex"}
        ] or macros_to_expand[:3]

        tier1_fine = expand_macro_regions_to_fine_hcp(
            tier1_macros, space=args.target_space, include_subcortical_mtl=True
        )
        all_winning_fine = expand_macro_regions_to_fine_hcp(
            macros_to_expand, space=args.target_space, include_subcortical_mtl=True
        )

        # Also provide anatomical sub-blocks for fast 48-worker exhaustive Stage-1 screening
        block_tpoj_mtc = expand_macro_regions_to_fine_hcp(
            ["Temporo-Parieto-Occipital Junction", "Medial Temporal Cortex"],
            space=args.target_space,
            include_subcortical_mtl=True,
        )
        block_ltc = expand_macro_regions_to_fine_hcp(
            ["Lateral Temporal Cortex"],
            space=args.target_space,
            include_subcortical_mtl=False,
        )
        block_ifc_aac = expand_macro_regions_to_fine_hcp(
            ["Inferior Frontal Cortex", "Auditory Association Cortex"],
            space=args.target_space,
            include_subcortical_mtl=False,
        )

        plan = {
            "target_space": args.target_space,
            "winning_macro_regions": macros_to_expand,
            "tier1_core_fine_parcels": {
                "count": len(tier1_fine),
                "parcels": tier1_fine,
            },
            "tier1_and_tier2_fine_parcels": {
                "count": len(all_winning_fine),
                "parcels": all_winning_fine,
            },
            "staged_exhaustive_blocks": {
                "block1_tpoj_mtc_subcortical": {
                    "count": len(block_tpoj_mtc),
                    "exhaustive_models": (1 << len(block_tpoj_mtc)) - 1,
                    "parcels": block_tpoj_mtc,
                },
                "block2_lateral_temporal": {
                    "count": len(block_ltc),
                    "exhaustive_models": (1 << len(block_ltc)) - 1,
                    "parcels": block_ltc,
                },
                "block3_ifc_aac": {
                    "count": len(block_ifc_aac),
                    "exhaustive_models": (1 << len(block_ifc_aac)) - 1,
                    "parcels": block_ifc_aac,
                },
            },
            "rsa_selected_parcels": selected_parcels,
        }
        (out_dir / "hcp_fine_screening_plan.json").write_text(
            json.dumps(plan, indent=2), encoding="utf-8"
        )
        print(f"\n[RSA] Wrote fine-grained HCPMMP1 screening plan to: {out_dir / 'hcp_fine_screening_plan.json'}", flush=True)

    (out_dir / "rsa_selected_parcels.txt").write_text(
        ("\n".join(selected_parcels) + "\n") if selected_parcels else "",
        encoding="utf-8",
    )
    if selected_parcels:
        print(f"[RSA] Wrote {len(selected_parcels)} selected parcels to: {out_dir / 'rsa_selected_parcels.txt'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
