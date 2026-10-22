#!/usr/bin/env python3
"""Create a concept embedding dictionary for MEG concept decoding.

The output JSON has the shape expected by the Concept decoding tab:

{
  "names": ["animal", "vehicle_1"],
  "embeddings": [[... 3072 floats ...], [... 3072 floats ...]],
  "metadata": {"provider": "openai", "model": "text-embedding-3-large", ...}
}

OpenAI text embeddings are the default. Pretrained Word2Vec/KeyedVectors,
CLIP text/image, and DINOv2 image targets can also be created.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/embeddings"
DEFAULT_PROVIDER = "openai"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
AUTO_COLUMNS = (
    "concept_label",
    "stimulus_group",
    "canonical_stimulus",
    "concept",
    "Full name",
    "Name",
    "Classification",
    "Folder",
    "stimulus",
)


def _clean_name(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = _clean_name(value)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _read_text_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = _clean_name(line)
        if clean:
            names.append(clean)
    return names


def _read_json_names(path: Path, column: str | None) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            key = column or next((candidate for candidate in AUTO_COLUMNS if candidate in payload[0]), None)
            if key is None:
                raise ValueError("JSON rows are objects; pass --column to choose the concept-name field.")
            return [_clean_name(row.get(key, "")) for row in payload]
        return [_clean_name(value) for value in payload]
    if isinstance(payload, dict):
        if "names" in payload:
            return [_clean_name(value) for value in payload["names"]]
        key = column or next((candidate for candidate in AUTO_COLUMNS if candidate in payload), None)
        if key is not None:
            raw = payload[key]
            if isinstance(raw, list):
                return [_clean_name(value) for value in raw]
        return [_clean_name(key) for key in payload.keys()]
    raise ValueError("Unsupported JSON concept-name structure.")


def _read_table_names(path: Path, column: str | None, sheet: str | int | None) -> list[str]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Reading tables requires pandas.") from exc

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        sheet_name: str | int = 0
        if sheet is not None:
            sheet_text = str(sheet).strip()
            sheet_name = int(sheet_text) if sheet_text.isdigit() else sheet_text
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        sep = "\t" if suffix in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep)
    if df.empty:
        return []
    selected = column
    if not selected:
        selected = next((candidate for candidate in AUTO_COLUMNS if candidate in df.columns), None)
    if not selected:
        columns = ", ".join(str(col) for col in df.columns)
        raise ValueError(f"Could not auto-select concept column. Pass --column. Available columns: {columns}")
    if selected not in df.columns:
        raise ValueError(f"Column {selected!r} not found. Available columns: {list(df.columns)}")
    return [_clean_name(value) for value in df[selected].tolist()]


def load_concept_names(args: argparse.Namespace) -> list[str]:
    names = [_clean_name(value) for value in args.name]
    if args.input:
        path = Path(args.input).expanduser().resolve()
        suffix = path.suffix.lower()
        if suffix in {".txt", ".list"}:
            names.extend(_read_text_names(path))
        elif suffix == ".json":
            names.extend(_read_json_names(path, args.column))
        elif suffix in {".csv", ".tsv", ".tab", ".xlsx", ".xls"}:
            names.extend(_read_table_names(path, args.column, args.sheet))
        else:
            raise ValueError(f"Unsupported input file type: {suffix}")
    names = _dedupe_preserve_order(names)
    if not names:
        raise ValueError("No concept names found. Use --name or --input.")
    return names


def _post_embeddings_request(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    inputs: list[str],
    dimensions: int | None,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": inputs,
        "encoding_format": "float",
    }
    if dimensions is not None:
        payload["dimensions"] = int(dimensions)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout)) as response:
        return json.loads(response.read().decode("utf-8"))


def create_embeddings(
    names: list[str],
    *,
    api_key: str,
    endpoint: str,
    model: str,
    dimensions: int | None,
    batch_size: int,
    timeout: float,
    max_retries: int,
) -> list[list[float]]:
    embeddings: list[list[float] | None] = [None] * len(names)
    batch_size = max(1, int(batch_size))
    for start in range(0, len(names), batch_size):
        batch = names[start : start + batch_size]
        for attempt in range(max(1, int(max_retries))):
            try:
                payload = _post_embeddings_request(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    inputs=batch,
                    dimensions=dimensions,
                    timeout=timeout,
                )
                data = sorted(payload.get("data", []), key=lambda row: int(row.get("index", 0)))
                if len(data) != len(batch):
                    raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(data)}.")
                for local_idx, row in enumerate(data):
                    vector = row.get("embedding")
                    if not isinstance(vector, list) or not vector:
                        raise RuntimeError("Embedding response did not contain a numeric vector.")
                    embeddings[start + local_idx] = [float(value) for value in vector]
                print(f"Embedded {min(start + batch_size, len(names))}/{len(names)} concepts", file=sys.stderr)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt + 1 >= max_retries:
                    raise RuntimeError(f"OpenAI embeddings request failed: HTTP {exc.code}: {detail}") from exc
                time.sleep(min(2.0 ** attempt, 30.0))
            except Exception:
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(min(2.0 ** attempt, 30.0))
    return [vector for vector in embeddings if vector is not None]


def _load_word2vec_format_subset(model_path: str, required_keys: set[str]):
    """Read only requested vectors from a standard binary or text Word2Vec file."""
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Word2Vec embedding creation requires numpy.") from exc

    lower_path = model_path.lower()
    compressed = lower_path.endswith(".gz")
    binary = _word2vec_file_is_binary(model_path)
    opener = gzip.open if compressed else open
    found: dict[str, Any] = {}

    if binary:
        with opener(model_path, "rb") as stream:
            header = stream.readline().split()
            if len(header) != 2:
                raise ValueError("Word2Vec binary file is missing its vocabulary/dimension header.")
            try:
                vocabulary_size, dimension = (int(value) for value in header)
            except Exception as exc:
                raise ValueError("Word2Vec binary header is invalid.") from exc
            vector_bytes = 4 * dimension
            for _ in range(vocabulary_size):
                word_buffer = bytearray()
                while True:
                    character = stream.read(1)
                    if not character:
                        raise ValueError("Word2Vec binary file ended before its declared vocabulary.")
                    if character == b" ":
                        break
                    if character != b"\n":
                        word_buffer.extend(character)
                raw_vector = stream.read(vector_bytes)
                if len(raw_vector) != vector_bytes:
                    raise ValueError("Word2Vec binary file ended inside a vector.")
                word = word_buffer.decode("utf-8", errors="replace")
                if word in required_keys:
                    found[word] = np.frombuffer(raw_vector, dtype="<f4").astype(
                        np.float32, copy=True
                    )
        return found

    expected_dimension = None

    def consider_line(line: str):
        nonlocal expected_dimension
        parts = line.rstrip().split()
        if len(parts) < 2:
            return
        word = parts[0]
        if word not in required_keys:
            return
        try:
            vector = np.asarray(parts[1:], dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Invalid numeric Word2Vec vector for {word!r}.") from exc
        if expected_dimension is None:
            expected_dimension = int(vector.size)
        if int(vector.size) != int(expected_dimension):
            raise ValueError(
                f"Word2Vec vector for {word!r} has {vector.size} values; "
                f"expected {expected_dimension}."
            )
        found[word] = vector

    with opener(model_path, "rt", encoding="utf-8", errors="replace") as stream:
        first_line = stream.readline()
        first_parts = first_line.rstrip().split()
        has_header = (
            len(first_parts) == 2
            and all(part.lstrip("+-").isdigit() for part in first_parts)
        )
        if has_header:
            expected_dimension = int(first_parts[1])
        else:
            consider_line(first_line)
        for line in stream:
            consider_line(line)
    return found


def _word2vec_file_is_binary(model_path: str) -> bool:
    """Infer standard Word2Vec format, including Gensim's extensionless .gz name."""
    lower_path = str(model_path).lower()
    if lower_path.endswith((".bin", ".bin.gz")):
        return True
    if lower_path.endswith((".vec", ".txt", ".vec.gz", ".txt.gz")):
        return False
    if not lower_path.endswith(".gz"):
        return False

    # Gensim-data calls the binary Google News file
    # ``word2vec-google-news-300.gz``. Inspect bytes after its text header so
    # callers do not need to rename or decompress the downloaded model.
    try:
        with gzip.open(model_path, "rb") as stream:
            stream.readline()
            sample = stream.read(4096)
    except Exception:
        return False
    text_bytes = set(range(32, 127)) | {9, 10, 13}
    return any(value not in text_bytes for value in sample)


def _load_word2vec_model(path: str, required_keys: set[str] | None = None):
    """Load only needed standard vectors, reserving gensim for native model files."""
    model_path = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("Select an existing Word2Vec/KeyedVectors model file.")

    lower_path = model_path.lower()
    if not lower_path.endswith((".kv", ".kvs", ".model")):
        return _load_word2vec_format_subset(model_path, required_keys or set())

    try:
        from gensim.models import KeyedVectors, Word2Vec
    except Exception as import_exc:
        raise RuntimeError(
            "Native KeyedVectors/Word2Vec model files require gensim. Install it in "
            "the MAS environment with `python -m pip install gensim`, or choose a "
            "standard .bin, .vec, or .txt Word2Vec export."
        ) from import_exc

    try:
        if lower_path.endswith((".kv", ".kvs")):
            return KeyedVectors.load(model_path, mmap="r")
        loaded = Word2Vec.load(model_path)
        return loaded.wv
    except Exception as exc:
        raise RuntimeError(f"Could not load Word2Vec model {model_path}: {exc}") from exc


def _word2vec_key_candidates(value: str) -> list[str]:
    text = _clean_name(value)
    title = text.title()
    candidates = [
        text,
        text.lower(),
        title,
        text.replace(" ", "_"),
        text.lower().replace(" ", "_"),
        title.replace(" ", "_"),
    ]
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _word2vec_lookup(model: Any, value: str):
    """Return a model vector plus the matched key, trying common case variants."""
    for candidate in _word2vec_key_candidates(value):
        try:
            if candidate in model:
                return model[candidate], candidate
        except Exception:
            try:
                return model[candidate], candidate
            except (KeyError, IndexError):
                pass
    return None, ""


def create_word2vec_embeddings(
    names: list[str],
    *,
    model_path: str,
    aggregation: str = "exact_or_mean",
    oov_policy: str = "error",
    keyed_vectors: Any = None,
) -> tuple[list[str], list[list[float]], dict[str, Any]]:
    """Create one vector per label from a pretrained Word2Vec-style model."""
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Word2Vec embedding creation requires numpy.") from exc

    aggregation = str(aggregation or "exact_or_mean").strip().lower()
    if aggregation not in {"exact_or_mean", "mean"}:
        raise ValueError(f"Unsupported Word2Vec aggregation: {aggregation!r}.")
    oov_policy = str(oov_policy or "error").strip().lower()
    if oov_policy not in {"error", "skip"}:
        raise ValueError(f"Unsupported Word2Vec OOV policy: {oov_policy!r}.")

    required_keys: set[str] = set()
    for raw_name in names:
        name = _clean_name(raw_name)
        required_keys.update(_word2vec_key_candidates(name))
        for token in re.findall(r"[A-Za-z0-9]+", name.replace("_", " ")):
            required_keys.update(_word2vec_key_candidates(token))
    model = (
        keyed_vectors
        if keyed_vectors is not None
        else _load_word2vec_model(model_path, required_keys=required_keys)
    )
    output_names: list[str] = []
    output_vectors: list[list[float]] = []
    missing_labels: list[str] = []
    partial_labels: dict[str, list[str]] = {}
    matched_keys: dict[str, list[str]] = {}

    for raw_name in names:
        name = _clean_name(raw_name)
        vector = None
        keys: list[str] = []
        missing_tokens: list[str] = []

        if aggregation == "exact_or_mean":
            exact, key = _word2vec_lookup(model, name)
            if exact is not None:
                vector = np.asarray(exact, dtype=np.float32).reshape(-1)
                keys = [key]

        if vector is None:
            tokens = re.findall(r"[A-Za-z0-9]+", name.replace("_", " "))
            token_vectors = []
            for token in tokens:
                token_vector, key = _word2vec_lookup(model, token)
                if token_vector is None:
                    missing_tokens.append(token)
                    continue
                token_vectors.append(np.asarray(token_vector, dtype=np.float32).reshape(-1))
                keys.append(key)
            if token_vectors:
                token_dims = {int(item.size) for item in token_vectors}
                if len(token_dims) != 1:
                    raise ValueError(f"Word2Vec token dimensions differ for label {name!r}.")
                vector = np.mean(np.vstack(token_vectors), axis=0, dtype=np.float32)

        if vector is None or vector.size == 0 or not np.all(np.isfinite(vector)):
            missing_labels.append(name)
            continue
        output_names.append(name)
        output_vectors.append([float(value) for value in vector.tolist()])
        matched_keys[name] = keys
        if missing_tokens:
            partial_labels[name] = missing_tokens

    if missing_labels and oov_policy == "error":
        preview = ", ".join(missing_labels[:10])
        suffix = "" if len(missing_labels) <= 10 else f" (+{len(missing_labels) - 10} more)"
        raise ValueError(
            f"Word2Vec has no usable vector for {len(missing_labels)} label(s): {preview}{suffix}. "
            "Choose 'Skip labels with no usable words' in the creation dialog to continue."
        )
    if not output_vectors:
        raise ValueError("The Word2Vec model did not contain a usable vector for any stimulus label.")

    dims = {len(vector) for vector in output_vectors}
    if len(dims) != 1:
        raise ValueError(f"Word2Vec output dimensions are inconsistent: {sorted(dims)}")
    metadata = {
        "provider": "word2vec",
        "model": os.path.basename(os.path.expanduser(str(model_path or ""))) or "in-memory Word2Vec",
        "model_path": os.path.abspath(os.path.expanduser(str(model_path))) if model_path else "",
        "aggregation": aggregation,
        "oov_policy": oov_policy,
        "requested_labels": int(len(names)),
        "embedded_labels": int(len(output_names)),
        "missing_labels": missing_labels,
        "partial_labels": partial_labels,
        "matched_keys": matched_keys,
    }
    return output_names, output_vectors, metadata


def _normalized_rows(values: Any):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Embedding creation requires numpy.") from exc

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Embedding features must be a 2D array, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding features contain non-finite values.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding features contain a zero-length vector.")
    return array / norms


def resolve_stimulus_image_paths(
    names: list[str],
    image_references_by_label: dict[str, list[str]],
    image_root: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Resolve event-log stimulus references against one selected image tree."""
    root = Path(str(image_root or "")).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Select an existing stimulus image folder.")

    by_basename: dict[str, list[Path]] = {}
    by_relative_path: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = candidate.resolve()
        by_basename.setdefault(candidate.name.lower(), []).append(resolved)
        try:
            relative = candidate.relative_to(root).as_posix().lower()
            by_relative_path[relative] = resolved
        except Exception:
            pass

    resolved_by_label: dict[str, list[str]] = {}
    missing_references: dict[str, list[str]] = {}
    ambiguous_references: dict[str, dict[str, list[str]]] = {}
    for raw_name in names:
        name = _clean_name(raw_name)
        references = list(image_references_by_label.get(name, []))
        label_paths: list[str] = []
        missing: list[str] = []
        ambiguous: dict[str, list[str]] = {}
        for raw_reference in references:
            reference = str(raw_reference or "").strip()
            if not reference:
                continue
            direct = Path(reference).expanduser()
            if direct.is_file() and direct.suffix.lower() in IMAGE_EXTENSIONS:
                label_paths.append(str(direct.resolve()))
                continue
            relative_key = reference.replace("\\", "/").lstrip("./").lower()
            relative_match = by_relative_path.get(relative_key)
            if relative_match is not None:
                label_paths.append(str(relative_match))
                continue
            matches = by_basename.get(os.path.basename(reference).lower(), [])
            if len(matches) == 1:
                label_paths.append(str(matches[0]))
            elif len(matches) > 1:
                ambiguous[reference] = [str(path) for path in matches]
            else:
                missing.append(reference)
        unique_paths = list(dict.fromkeys(label_paths))
        if unique_paths:
            resolved_by_label[name] = unique_paths
        if missing:
            missing_references[name] = missing
        if ambiguous:
            ambiguous_references[name] = ambiguous

    missing_labels = [name for name in names if _clean_name(name) not in resolved_by_label]
    metadata = {
        "image_root": str(root),
        "resolved_images_by_label": resolved_by_label,
        "missing_labels": missing_labels,
        "missing_image_references": missing_references,
        "ambiguous_image_references": ambiguous_references,
    }
    return resolved_by_label, metadata


def _torch_device(requested: str, torch_module: Any) -> str:
    requested = str(requested or "auto").strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = mps is not None and bool(mps.is_available())
    if requested == "cuda":
        if not cuda_available:
            raise ValueError("CUDA was selected, but PyTorch cannot access a CUDA device.")
        return "cuda"
    if requested == "mps":
        if not mps_available:
            raise ValueError("Apple Metal (MPS) was selected, but it is not available.")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if requested != "auto":
        raise ValueError(f"Unsupported Torch device: {requested!r}.")
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _vision_dependencies():
    try:
        import torch
        from PIL import Image
        import transformers
    except Exception as exc:
        raise RuntimeError(
            "CLIP/DINOv2 embedding creation requires torch, transformers, and Pillow. "
            f"The vision stack could not be imported: {exc}"
        ) from exc

    # Transformers' PIL/numpy image processors do not require torchvision, but
    # Transformers detects an installed copy by package metadata. If that copy
    # is binary-incompatible with torch, mark the optional backend unavailable
    # before importing model classes so the supported slow processors are used.
    try:
        import torchvision  # noqa: F401
    except Exception:
        try:
            transformers.utils.import_utils._torchvision_available = False
            for module_name in list(sys.modules):
                if module_name == "torchvision" or module_name.startswith("torchvision."):
                    sys.modules.pop(module_name, None)
        except Exception:
            pass

    try:
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
    except Exception as exc:
        raise RuntimeError(
            "CLIP/DINOv2 model classes could not be imported from transformers: "
            f"{exc}"
        ) from exc
    return torch, Image, AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


def _clip_text_features(prompts: list[str], model_name: str, device: str, batch_size: int):
    torch, _, _, _, AutoProcessor, CLIPModel = _vision_dependencies()
    resolved_device = _torch_device(device, torch)
    try:
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = CLIPModel.from_pretrained(model_name).to(resolved_device).eval()
    except Exception as exc:
        raise RuntimeError(f"Could not load CLIP model {model_name!r}: {exc}") from exc
    rows = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
        rows.append(features.detach().float().cpu().numpy())
    import numpy as np

    return np.vstack(rows), resolved_device


def create_clip_text_embeddings(
    names: list[str],
    *,
    model_name: str = DEFAULT_CLIP_MODEL,
    prompt_template: str = "a photo of a {label}",
    device: str = "auto",
    batch_size: int = 32,
    feature_extractor: Any = None,
) -> tuple[list[str], list[list[float]], dict[str, Any]]:
    """Create normalized CLIP text targets in CLIP's joint image/text space."""
    if "{label}" not in str(prompt_template):
        raise ValueError("The CLIP text prompt must contain the {label} placeholder.")
    output_names = [_clean_name(name) for name in names]
    prompts = [str(prompt_template).format(label=name) for name in output_names]
    if feature_extractor is None:
        raw_features, resolved_device = _clip_text_features(
            prompts, model_name, device, batch_size
        )
    else:
        raw_features = feature_extractor(prompts)
        resolved_device = "test"
    features = _normalized_rows(raw_features)
    if features.shape[0] != len(output_names):
        raise ValueError("CLIP text feature count does not match the requested labels.")
    metadata = {
        "provider": "clip_text",
        "model": str(model_name),
        "source": "prompted concept label text",
        "prompt_template": str(prompt_template),
        "device": resolved_device,
        "normalized": True,
        "requested_labels": int(len(names)),
        "embedded_labels": int(len(output_names)),
    }
    return output_names, features.astype("float32").tolist(), metadata


def _transformer_image_features(
    image_paths: list[str],
    provider: str,
    model_name: str,
    device: str,
    batch_size: int,
):
    torch, Image, AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel = _vision_dependencies()
    resolved_device = _torch_device(device, torch)
    provider = str(provider).strip().lower()
    try:
        if provider == "clip_image":
            processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
            model = CLIPModel.from_pretrained(model_name).to(resolved_device).eval()
        elif provider == "dinov2_image":
            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
            model = AutoModel.from_pretrained(model_name).to(resolved_device).eval()
        else:
            raise ValueError(f"Unsupported image embedding provider: {provider!r}.")
    except Exception as exc:
        raise RuntimeError(f"Could not load {provider} model {model_name!r}: {exc}") from exc

    rows = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
        with torch.inference_mode():
            if provider == "clip_image":
                features = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                features = getattr(outputs, "pooler_output", None)
                if features is None:
                    features = outputs.last_hidden_state[:, 0]
        rows.append(features.detach().float().cpu().numpy())
    import numpy as np

    return np.vstack(rows), resolved_device


def create_image_embeddings(
    names: list[str],
    *,
    image_references_by_label: dict[str, list[str]],
    image_root: str,
    provider: str,
    model_name: str,
    missing_policy: str = "error",
    device: str = "auto",
    batch_size: int = 16,
    feature_extractor: Any = None,
) -> tuple[list[str], list[list[float]], dict[str, Any]]:
    """Create normalized image embeddings and average variants into label prototypes."""
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Image embedding creation requires numpy.") from exc

    provider = str(provider or "").strip().lower()
    if provider not in {"clip_image", "dinov2_image"}:
        raise ValueError(f"Unsupported image embedding provider: {provider!r}.")
    missing_policy = str(missing_policy or "error").strip().lower()
    if missing_policy not in {"error", "skip"}:
        raise ValueError(f"Unsupported missing-image policy: {missing_policy!r}.")

    resolved_by_label, resolution_metadata = resolve_stimulus_image_paths(
        names, image_references_by_label, image_root
    )
    missing_labels = list(resolution_metadata["missing_labels"])
    ambiguous = dict(resolution_metadata["ambiguous_image_references"])
    if missing_policy == "error" and (missing_labels or ambiguous):
        problems = []
        if missing_labels:
            preview = ", ".join(missing_labels[:8])
            problems.append(f"no image for {len(missing_labels)} label(s): {preview}")
        if ambiguous:
            preview = ", ".join(list(ambiguous)[:8])
            problems.append(f"ambiguous filenames for {len(ambiguous)} label(s): {preview}")
        raise ValueError(
            "Image resolution failed (" + "; ".join(problems) + "). Select the exact "
            "stimulus folder or choose 'Skip labels without an unambiguous image'."
        )

    output_names = [_clean_name(name) for name in names if _clean_name(name) in resolved_by_label]
    if not output_names:
        raise ValueError("No stimulus images could be resolved for the current summary labels.")
    unique_image_paths = list(
        dict.fromkeys(
            path for name in output_names for path in resolved_by_label.get(name, [])
        )
    )
    if feature_extractor is None:
        raw_features, resolved_device = _transformer_image_features(
            unique_image_paths, provider, model_name, device, batch_size
        )
    else:
        raw_features = feature_extractor(unique_image_paths)
        resolved_device = "test"
    image_features = _normalized_rows(raw_features)
    if image_features.shape[0] != len(unique_image_paths):
        raise ValueError("Image feature count does not match the resolved stimulus images.")
    features_by_path = {
        path: image_features[index] for index, path in enumerate(unique_image_paths)
    }

    prototypes = []
    for name in output_names:
        rows = np.vstack([features_by_path[path] for path in resolved_by_label[name]])
        prototype = np.mean(rows, axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Image prototype for {name!r} has zero or invalid length.")
        prototypes.append(prototype / norm)

    metadata = {
        "provider": provider,
        "model": str(model_name),
        "source": "presented stimulus images",
        "device": resolved_device,
        "normalized": True,
        "prototype_aggregation": "mean normalized image vectors, then L2 normalize",
        "missing_policy": missing_policy,
        "requested_labels": int(len(names)),
        "embedded_labels": int(len(output_names)),
        "resolved_image_count": int(len(unique_image_paths)),
    }
    metadata.update(resolution_metadata)
    return output_names, np.asarray(prototypes, dtype=np.float32).tolist(), metadata


def write_output(
    path: Path,
    names: list[str],
    embeddings: list[list[float]],
    args: argparse.Namespace,
    metadata: dict[str, Any] | None = None,
) -> None:
    if len(names) != len(embeddings):
        raise ValueError(f"Name/vector count mismatch: {len(names)} names, {len(embeddings)} embeddings.")
    dims = {len(vector) for vector in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Embedding dimensions are inconsistent: {sorted(dims)}")
    output_metadata = {
        "provider": getattr(args, "provider", DEFAULT_PROVIDER),
        "model": args.model,
        "dimension": int(next(iter(dims))),
        "source": "concept name strings",
        "input": args.input,
        "column": args.column,
    }
    if metadata:
        output_metadata.update(metadata)
    payload = {
        "names": names,
        "embeddings": embeddings,
        "metadata": output_metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a JSON embedding dictionary using OpenAI, Word2Vec, CLIP, or DINOv2."
        )
    )
    parser.add_argument("--input", help="Concept list/table: .txt, .json, .csv, .tsv, .xlsx.")
    parser.add_argument("--column", help="Column/key containing concept names for table or row-object JSON inputs.")
    parser.add_argument("--sheet", help="Excel sheet name/index. Defaults to first sheet.")
    parser.add_argument("--name", action="append", default=[], help="Add one concept name directly. Can be repeated.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--provider",
        choices=("openai", "word2vec", "clip_text", "clip_image", "dinov2_image"),
        default=DEFAULT_PROVIDER,
        help=f"Embedding generator. Default: {DEFAULT_PROVIDER}.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model. Default: {DEFAULT_MODEL}.")
    parser.add_argument(
        "--dimensions",
        type=int,
        help="Optional reduced dimension for supported OpenAI models. Omit for the model's native size.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Concept names per API request.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retry count for API/rate-limit failures.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Embeddings API endpoint.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--word2vec-model", help="Pretrained .bin/.vec/.txt/.kv/.model Word2Vec model file.")
    parser.add_argument(
        "--word2vec-aggregation",
        choices=("exact_or_mean", "mean"),
        default="exact_or_mean",
        help="How multiword labels are converted to Word2Vec vectors.",
    )
    parser.add_argument(
        "--word2vec-oov-policy",
        choices=("error", "skip"),
        default="error",
        help="What to do with labels that have no usable Word2Vec words.",
    )
    parser.add_argument(
        "--transformer-model",
        help="CLIP/DINOv2 Hugging Face model name or local model directory.",
    )
    parser.add_argument(
        "--prompt-template",
        default="a photo of a {label}",
        help="CLIP text template; must contain {label}.",
    )
    parser.add_argument(
        "--image-root",
        help="Stimulus image folder for CLIP/DINOv2 image providers.",
    )
    parser.add_argument(
        "--image-missing-policy",
        choices=("error", "skip"),
        default="error",
        help="What image providers do when a label cannot be resolved to one image.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Torch inference device for CLIP/DINOv2.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved concept names; do not call the API.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"Output already exists; pass --force to overwrite: {output_path}", file=sys.stderr)
        return 2
    names = load_concept_names(args)
    if args.dry_run:
        for name in names:
            print(name)
        print(f"{len(names)} unique concept names", file=sys.stderr)
        return 0
    metadata: dict[str, Any] = {"provider": args.provider}
    if args.provider == "word2vec":
        if not args.word2vec_model:
            print("--word2vec-model is required for the Word2Vec provider.", file=sys.stderr)
            return 2
        names, embeddings, word2vec_metadata = create_word2vec_embeddings(
            names,
            model_path=args.word2vec_model,
            aggregation=args.word2vec_aggregation,
            oov_policy=args.word2vec_oov_policy,
        )
        args.model = word2vec_metadata["model"]
        metadata.update(word2vec_metadata)
    elif args.provider == "clip_text":
        args.model = args.transformer_model or DEFAULT_CLIP_MODEL
        names, embeddings, generated_metadata = create_clip_text_embeddings(
            names,
            model_name=args.model,
            prompt_template=args.prompt_template,
            device=args.device,
        )
        metadata.update(generated_metadata)
    elif args.provider in {"clip_image", "dinov2_image"}:
        if not args.image_root:
            print("--image-root is required for image embedding providers.", file=sys.stderr)
            return 2
        args.model = args.transformer_model or (
            DEFAULT_CLIP_MODEL if args.provider == "clip_image" else DEFAULT_DINOV2_MODEL
        )
        names, embeddings, generated_metadata = create_image_embeddings(
            names,
            image_references_by_label={name: [name] for name in names},
            image_root=args.image_root,
            provider=args.provider,
            model_name=args.model,
            missing_policy=args.image_missing_policy,
            device=args.device,
        )
        metadata.update(generated_metadata)
    else:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        if not api_key:
            print(f"Missing API key. Set {args.api_key_env}.", file=sys.stderr)
            return 2
        embeddings = create_embeddings(
            names,
            api_key=api_key,
            endpoint=args.endpoint,
            model=args.model,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    write_output(output_path, names, embeddings, args, metadata=metadata)
    print(f"Wrote {len(names)} {args.provider} embeddings to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
