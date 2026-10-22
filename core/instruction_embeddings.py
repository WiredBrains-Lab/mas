"""Block instruction targets and held-out retrieval reporting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def block_key(value):
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def find_instruction_csv(location):
    location = Path(location).expanduser().resolve()
    if location.is_file():
        location = location.parent
    for directory in [location, *location.parents]:
        candidate = directory / "block-instructions.csv"
        if candidate.is_file():
            return str(candidate)
    return ""


def read_instruction_csv(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"block_id", "line_2"}.issubset(table.columns):
        raise ValueError("Block instructions CSV requires block_id and line_2 columns.")
    mapping = {}
    for row in table.to_dict("records"):
        key = block_key(row["block_id"])
        value = " ".join(row["line_2"].split())
        if not key:
            continue
        if key in mapping and mapping[key] != value:
            raise ValueError(f"Conflicting line_2 instructions for block {key}.")
        mapping[key] = value
    names = list(dict.fromkeys(value for value in mapping.values() if value))
    if not names:
        raise ValueError("Block instructions CSV has no nonempty line_2 values.")
    return {"source_path": str(Path(path).resolve()), "block_mapping": mapping, "names": names}


def load_instruction_targets(path, dimension):
    if not path or Path(path).suffix.lower() != ".json":
        return None
    with open(path, encoding="utf-8") as stream:
        targets = json.load(stream).get("metadata", {}).get("instruction_targets")
    if not targets or targets.get("status") != "embedded":
        return None
    vectors = np.asarray(targets["embeddings"], dtype=np.float32)
    if (vectors.shape != (len(targets["names"]), dimension)
            or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)):
        raise ValueError("Instruction targets have invalid vectors or incompatible dimensions.")
    return targets


def score_instructions(predictions, predicted_vectors, targets, trial_df, transform):
    """Compare instructions separately; mixed or unknown expectations are not scored."""
    names = targets["names"]
    vectors = transform(np.asarray(targets["embeddings"], dtype=np.float32))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("An instruction target has zero length after the scoring transform.")
    scores = predicted_vectors @ (vectors / norms).T
    mapping = targets["block_mapping"]
    rows = []
    for index, row in predictions.iterrows():
        members = [int(v) for v in str(row["source_trial_indices"]).split(",")]
        expected = [mapping.get(block_key(trial_df.iloc[v].get("block_id", "")), "") for v in members]
        unique = list(dict.fromkeys(expected))
        status = "scored"
        if "" in unique:
            status = "missing_block_instruction"
        elif len(unique) != 1:
            status = "mixed_block_instructions"
        elif unique[0] not in names:
            status = "instruction_not_embedded"
        best = int(np.argmax(scores[index]))
        tied = int(np.sum(np.isclose(scores[index], scores[index, best], atol=1e-7, rtol=0))) > 1
        valid = status == "scored"
        rows.append({
            "expected_instruction": " | ".join(v for v in unique if v),
            "predicted_instruction": names[best],
            "instruction_cosine": float(scores[index, best]),
            "expected_instruction_cosine": float(scores[index, names.index(unique[0])]) if valid else np.nan,
            "instruction_match": bool(names[best] == unique[0] and not tied) if valid else None,
            "instruction_status": "tied_prediction" if valid and tied else status,
        })
    result = pd.DataFrame(rows, index=predictions.index)
    for column in result:
        predictions[column] = result[column]
    eligible = result["instruction_match"].dropna()
    return {"instruction_candidate_count": len(names),
            "instruction_scored_predictions": len(eligible),
            "instruction_unscored_predictions": len(result) - len(eligible),
            "instruction_top1_accuracy": float(eligible.astype(float).mean()) if len(eligible) else np.nan}
