"""Experimental full-vector CLIP decoding; heavy dependencies stay in a subprocess."""
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

MODEL = 'lambda/sd-image-variations-diffusers'
REVISION = 'a2a13984e57db80adcc9e3f85d568dcccb9b29fc'
LEGACY_PROFILE = dict(model=MODEL, revision=REVISION, dimension=768,
               representation='raw_projected_clip', target='training_unique_image_concept_mean',
               resize='224_bicubic_no_antialias', schema_version=1)

PROFILE = dict(LEGACY_PROFILE, target='individual_stimulus_image', schema_version=2)


def runtime_python():
    path = Path(os.environ.get('CLIP_GENERATION_PYTHON', str(Path.home()/'.cache/clip-image-test-venv/bin/python')))
    if not path.is_file():
        raise RuntimeError('Image-generation environment unavailable. Set CLIP_GENERATION_PYTHON to the prepared Python interpreter.')
    return str(path)


def worker_command(*args):
    return [runtime_python(), str(Path(__file__).resolve()), *map(str,args)]


def validate_bundle(bundle):
    if bundle.get('generation_profile') not in (PROFILE, LEGACY_PROFILE):
        raise ValueError('Incompatible image-generation profile: retrain with CLIP image generation (ridge).')
    for key in ['predictions','prototypes']:
        values=np.asarray(bundle[key])
        if values.ndim!=2 or values.shape[1]!=768 or not np.isfinite(values).all():
            raise ValueError(f'Invalid generator vectors: {key}')
    if bundle['generation_profile'] == PROFILE:
        targets = np.asarray(bundle.get('test_target_vectors'))
        if targets.shape != np.asarray(bundle['predictions']).shape or not np.isfinite(targets).all():
            raise ValueError('Invalid image-specific test target vectors.')


def average_trials(X,y,indices,n,seed,image_ids=None):
    rng=np.random.default_rng(seed); rows=[]; labels=[]; members=[]
    groups = y if image_ids is None else np.asarray(image_ids)
    for key in sorted(set(groups[indices])):
        group=indices[groups[indices]==key].copy();rng.shuffle(group)
        label = y[group[0]]
        if len(set(y[group])) != 1: raise ValueError('An image has conflicting concept labels.')
        for start in range(0,len(group),n):
            chunk=group[start:start+n]
            rows.append(X[chunk].mean(0));labels.append(label);members.append(chunk.tolist())
    return np.asarray(rows),np.asarray(labels),members


def batch_fingerprint(bundle):
    import hashlib
    identity = hashlib.sha256()
    for key in ['predictions', 'prototypes']:
        a = np.ascontiguousarray(bundle[key])
        identity.update(str((a.shape, a.dtype)).encode())
        identity.update(a.tobytes())
    identity.update(json.dumps([bundle['generation_profile'], list(map(str, bundle['test_labels'])),
                                bundle['class_names'], bundle['test_members']], sort_keys=True).encode())
    return identity.hexdigest()


def load_block_types(bundle, location):
    """Use the nearest ancestor block-instructions.csv; join by block_id, not row order."""
    location = Path(location).resolve()
    for directory in [location, *location.parents]:
        path = directory / 'block-instructions.csv'
        if not path.is_file():
            continue
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        image_columns = [f'image_{n}' for n in range(1, 4)]
        required = {'block_id', 'line_2', *image_columns}
        if not required.issubset(table.columns):
            raise ValueError(f'Block instructions require block_id, line_2, and image_1..image_3: {path}')
        mapping = {}
        exemplars = {}
        for row in table.to_dict('records'):
            key = str(int(row['block_id']))
            value = row['line_2'].strip()
            if key in mapping and mapping[key] != value:
                raise ValueError(f'Conflicting instructions for block {key}: {path}')
            mapping[key] = value
            exemplars[key] = [str(row[column]).strip() for column in image_columns]
            if not all(exemplars[key]):
                raise ValueError(f'Block {key} has a missing exemplar image: {path}')
        bundle['block_types'] = mapping
        bundle['block_exemplars'] = exemplars
        bundle['block_instructions_path'] = str(path)
        return


def ensure_block_concept_vectors(bundle, encode_image=None):
    """Attach one image-space concept vector per block from its three exemplars."""
    import hashlib
    existing = bundle.get('block_concept_vectors')
    if isinstance(existing, dict) and existing:
        for block, vector in existing.items():
            values = np.asarray(vector)
            if values.shape != (768,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid block concept vector for block {block}.')
        return
    exemplars = bundle.get('block_exemplars')
    if not isinstance(exemplars, dict) or not exemplars:
        raise ValueError('Block concept vectors require block-instructions.csv with exemplar images.')
    root = Path(bundle['image_root'])
    cache = Path.home() / '.cache/meg_clip_generation_vectors'
    cache.mkdir(parents=True, exist_ok=True)
    vectors = {}
    hashes = {}
    for block, names in exemplars.items():
        block_vectors = []
        hashes[str(block)] = {}
        for name in names:
            path = root / Path(str(name).replace('\\', '/')).name
            if not path.is_file():
                raise FileNotFoundError(f'Block {block} exemplar image is missing: {path}')
            blob = path.read_bytes()
            image_hash = hashlib.sha256(blob).hexdigest()
            key = hashlib.sha256(blob + json.dumps(LEGACY_PROFILE, sort_keys=True).encode()).hexdigest()
            entry = cache / (key + '.npy')
            if entry.is_file():
                vector = np.load(entry, allow_pickle=False)
            elif encode_image is not None:
                vector = np.asarray(encode_image(path))
                np.save(entry, vector)
            else:
                raise RuntimeError(
                    f'CLIP embedding is not cached for block exemplar {path.name}; retrain or generate with the image encoder available.'
                )
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f'Invalid CLIP embedding for block exemplar: {path}')
            block_vectors.append(vector)
            hashes[str(block)][path.name] = image_hash
        vectors[str(block)] = np.mean(block_vectors, axis=0)
    bundle['block_concept_vectors'] = vectors
    bundle['block_concept_exemplar_sha256'] = hashes
    bundle['block_concept_definition'] = 'mean raw projected CLIP image embedding of image_1..image_3'


def block_concept_for_index(bundle, index):
    """Average trial-level block vectors for one test average and return audit metadata."""
    rows = comparison_inputs(bundle, index)
    vectors = bundle.get('block_concept_vectors') or {}
    used = []
    for row in rows:
        block = row.get('block_id')
        if block is None or pd.isna(block):
            raise ValueError(f'Test average {index} has no block_id for a contributing trial.')
        key = str(int(block))
        if key not in vectors:
            raise ValueError(f'No block concept vector is available for block {key}.')
        used.append(key)
    vector = np.mean([np.asarray(vectors[key]) for key in used], axis=0)
    if vector.shape != (768,) or not np.isfinite(vector).all():
        raise ValueError(f'Invalid averaged block concept vector for test average {index}.')
    labels = [f"block {key} ({bundle.get('block_types', {}).get(key, 'unknown')})" for key in sorted(set(used), key=int)]
    return vector, dict(block_ids=used, block_labels=labels)


def load_trial_metadata(bundle, bundle_path):
    """Read a checked sidecar for older bundles and the recording's block instructions."""
    if 'trial_metadata' not in bundle:
        path = Path(bundle_path).parent / 'trial_metadata.csv'
        if path.is_file():
            table = pd.read_csv(path)
            if table['stimulus'].astype(str).tolist() != bundle['stimuli']:
                raise ValueError('Trial metadata does not match the saved stimulus sequence.')
            bundle['trial_metadata'] = json.loads(table.to_json(orient='records'))
    load_block_types(bundle, Path(bundle_path).parent)


def comparison_inputs(bundle, index):
    """Keep metadata and source images aligned with the exact averaged members."""
    metadata = bundle.get('trial_metadata')
    if metadata is not None and len(metadata) != len(bundle['stimuli']):
        raise ValueError('Trial metadata length does not match stimuli.')
    rows = []
    for member in bundle['test_members'][index]:
        row = dict(metadata[member]) if metadata is not None else {}
        stimulus = bundle['stimuli'][member]
        if row.get('stimulus', stimulus) != stimulus:
            raise ValueError('Trial metadata does not match the saved stimulus sequence.')
        block = row.get('block_id')
        if block is not None and not pd.isna(block) and 'block_types' in bundle:
            row['block_type'] = bundle['block_types'].get(str(int(block)), '')
        row.update(generator_trial_row_index=int(member), stimulus=stimulus)
        row['image_path'] = str(Path(bundle['image_root']) / Path(stimulus.replace('\\', '/')).name)
        rows.append(row)
    return rows


def trial_caption(row, include_block_type=True):
    def display(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return str(int(value)) if isinstance(value, (int, float)) else str(value)
    trial = display(row.get('trial_index_in_file'))
    session = display(row.get('session_id'))
    block = display(row.get('block_id'))
    scope = []
    if session: scope.append('S' + session)
    if block:
        label = 'block ' + block
        if include_block_type and row.get('block_type'):
            label += ' (' + str(row['block_type']) + ')'
        scope.append(label)
    if not scope and row.get('source_fif'):
        scope.append(str(row['source_fif']))
    if not scope and row.get('source_fif_path'):
        scope.append(Path(str(row['source_fif_path'])).name)
    return (' / '.join(scope) + ': ' if scope else '') + (trial or 'unavailable')


def make_input_comparison(bundle, index, prediction, predicted, block_subtracted, block_metadata):
    """Compose inputs, the MEG prediction, and its block-concept-subtracted result."""
    import math
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    rows = comparison_inputs(bundle, index)
    if not rows:
        raise ValueError('Prediction has no contributing trials.')
    def font(size):
        for name in ['DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf']:
            try: return ImageFont.truetype(name, size)
            except OSError: pass
        return ImageFont.load_default()
    title_font, small = font(22), font(18)
    width = max(512, prediction.width, block_subtracted.width)
    gap, margin = 12, 18
    canvas_width = 3 * width + 2 * gap + 2 * margin
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    title = 'Trial index in file: ' + ' | '.join(trial_caption(row) for row in rows)
    lines, line = [], ''
    for word in title.split():
        candidate = (line + ' ' + word).strip()
        if line and measure.textlength(candidate, font=title_font) > canvas_width - 2 * margin:
            lines.append(line); line = word
        else: line = candidate
    lines.append(line)
    header = 90 + 28 * len(lines)
    height = max(512, prediction.height)
    panel = Image.new('RGB', (canvas_width, header + height + margin), 'white')
    draw = ImageDraw.Draw(panel)
    for n, line in enumerate(lines):
        draw.text((margin, 12 + 28 * n), line, font=title_font, fill='#171717')
    draw.text((margin, header - 62), f'Test average: {index} | Source-file trial numbering', font=small, fill='#444444')
    draw.text((margin, header - 32), f"Input stimuli: {bundle['test_labels'][index]} ({len(rows)} trials)", font=small, fill='#171717')
    right = margin + width + gap
    draw.text((right, header - 32), f'MEG prediction (retrieved: {predicted})', font=small, fill='#171717')
    residual_left = margin + 2 * (width + gap)
    block_label = ' + '.join(block_metadata['block_labels'])
    residual_title = f'MEG prediction - block concept: {block_label}'
    residual_font = small
    if draw.textlength(residual_title, font=residual_font) > width:
        residual_font = font(max(8, int(18 * width / draw.textlength(residual_title, font=small))))
    draw.text((residual_left, header - 32), residual_title, font=residual_font, fill='#171717')
    cols = math.ceil(math.sqrt(len(rows)))
    nrows = math.ceil(len(rows) / cols)
    cell_w, cell_h = width // cols, height // nrows
    for n, row in enumerate(rows):
        x, y = margin + (n % cols) * cell_w, header + (n // cols) * cell_h
        with Image.open(row['image_path']) as source:
            thumbnail = ImageOps.contain(source.convert('RGB'), (cell_w - 8, cell_h - 32))
        panel.paste(thumbnail, (x + (cell_w - thumbnail.width) // 2, y + (cell_h - 32 - thumbnail.height) // 2))
        caption = trial_caption(row, include_block_type=False)
        caption_font = small
        if draw.textlength(caption, font=caption_font) > cell_w - 8:
            caption_font = font(max(8, int(18 * (cell_w - 8) / draw.textlength(caption, font=small))))
        draw.text((x + 4, y + cell_h - 26), caption, font=caption_font, fill='#171717')
    panel.paste(prediction.convert('RGB'), (right + (width - prediction.width) // 2,
                                           header + (height - prediction.height) // 2))
    panel.paste(block_subtracted.convert('RGB'),
                (residual_left + (width - block_subtracted.width) // 2,
                 header + (height - block_subtracted.height) // 2))
    return panel


def generate_batch(bundle, out, render, cancelled=lambda: False, progress=print):
    """Save input/prediction/block-subtracted comparisons with resumable rendering."""
    import hashlib
    from PIL import Image
    validate_bundle(bundle)
    ensure_block_concept_vectors(bundle)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = [comparison_inputs(bundle, i) for i in range(len(bundle['predictions']))]
    # Check inputs before any expensive generation, and fingerprint their content.
    image_hashes = {}
    for rows in inputs:
        for row in rows:
            path = row['image_path']
            if path not in image_hashes:
                with Image.open(path) as image: image.verify()
                image_hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # Block-type text can be refreshed without invalidating saved prediction pixels.
    identity_inputs = [[{k: v for k, v in row.items() if k != 'block_type'} for row in rows] for rows in inputs]
    fingerprint = batch_fingerprint(bundle)
    block_identity = hashlib.sha256()
    block_identity.update(json.dumps(bundle.get('block_exemplars', {}), sort_keys=True).encode())
    block_identity.update(json.dumps(bundle.get('block_concept_exemplar_sha256', {}), sort_keys=True).encode())
    for block in sorted(bundle['block_concept_vectors'], key=int):
        block_identity.update(block.encode())
        block_identity.update(np.ascontiguousarray(bundle['block_concept_vectors'][block]).tobytes())
    config = dict(fingerprint=fingerprint, layout='input_prediction_block_subtracted_v2',
                  input_fingerprint=hashlib.sha256(json.dumps([identity_inputs, image_hashes], sort_keys=True).encode()).hexdigest(),
                  block_concept_fingerprint=block_identity.hexdigest(),
                  block_label_fingerprint=hashlib.sha256(json.dumps(bundle.get('block_types', {}), sort_keys=True).encode()).hexdigest(),
                  seed=42, steps=30, guidance=3., profile=bundle['generation_profile'])
    path = out / 'batch_manifest.json'
    state = json.loads(path.read_text()) if path.exists() else dict(config=config, records={}, status='running')
    if state['config'] != config:
        previous = state.get('config', {})
        shared_keys = ['fingerprint', 'input_fingerprint', 'seed', 'steps', 'guidance', 'profile']
        same_inputs = all(previous.get(key) == config.get(key) for key in shared_keys)
        migratable = (
            previous.get('layout') == 'input_prediction_v1' and same_inputs
        ) or (
            previous.get('layout') == config['layout'] and same_inputs
            and previous.get('block_concept_fingerprint') == config['block_concept_fingerprint']
        )
        if migratable:
            state['config'] = config
        else:
            raise ValueError('Batch folder belongs to different predictions/settings, block concepts, or input images/metadata. Choose a new folder.')
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)); temp.replace(path)
        csv = out / 'predictions.tmp'
        pd.DataFrame(state['records'].values()).to_csv(csv, index=False)
        csv.replace(out / 'predictions.csv')
    def valid(record, folder):
        if not record: return False
        file = folder / record['file']
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != record['sha256']:
            return False
        try:
            with Image.open(file) as image: image.verify()
        except Exception: return False
        return True
    legacy = []
    for folder in ['images_all_with_ideals', 'images_all_meg']:
        directory = out.parent / folder
        manifest = directory / 'batch_manifest.json'
        if not manifest.exists(): continue
        old = json.loads(manifest.read_text())
        c = old.get('config', {})
        if all(c.get(k) == config[k] for k in ['fingerprint', 'seed', 'steps', 'guidance', 'profile']) and 'layout' not in c:
            legacy.append((directory, old.get('records', {})))
    total = len(bundle['predictions'])
    def current(i):
        record = state['records'].get(f'meg_{i}')
        return (valid(record, out) and record.get('input_trials') == inputs[i]
                and record.get('block_subtracted_box') is not None
                and record.get('block_concept_fingerprint') == config['block_concept_fingerprint'])
    done = sum(current(i) for i in range(total))
    state['status'] = 'running'; state.pop('error', None); save()
    progress(f'BATCH_PROGRESS {done} {total}')
    try:
        for i, vector in enumerate(bundle['predictions']):
            if cancelled():
                state['status'] = 'cancelled'; save(); return state
            key = f'meg_{i}'
            if current(i): continue
            image, flags = None, None
            previous = state['records'].get(key)
            if valid(previous, out):
                with Image.open(out / previous['file']) as source:
                    box = previous.get('prediction_box')
                    # v1 app comparisons used 512x512 predictions with an 18px margin.
                    if box is None and source.width == 1072 and source.height >= 648:
                        box = [542, source.height - 530, 1054, source.height - 18]
                    if box is not None:
                        image = source.crop(box).convert('RGB')
                        flags = previous.get('nsfw')
            for directory, records in legacy:
                if image is not None: break
                record = records.get(key)
                if valid(record, directory):
                    with Image.open(directory / record['file']) as source: image = source.convert('RGB')
                    flags = record.get('nsfw'); break
            if image is None: image, flags = render(vector)
            block_vector, block_metadata = block_concept_for_index(bundle, i)
            residual_vector = np.asarray(vector) - block_vector
            if residual_vector.shape != (768,) or not np.isfinite(residual_vector).all():
                raise ValueError(f'Invalid block-subtracted vector for test average {i}.')
            block_image, block_flags = render(residual_vector)
            scores = np.einsum('ij,j->i', bundle['prototypes'] / np.maximum(np.linalg.norm(bundle['prototypes'], axis=1, keepdims=True), 1e-12), vector / max(np.linalg.norm(vector), 1e-12))
            predicted = str(bundle['class_names'][int(scores.argmax())])
            panel = make_input_comparison(bundle, i, image, predicted, block_image, block_metadata)
            filename = f'test_{i:05d}.png'
            temporary = out / (filename + '.tmp')
            panel.save(temporary, format='PNG'); temporary.replace(out / filename)
            state['records'][key] = dict(kind='meg', test_average=i, actual=str(bundle['test_labels'][i]),
                predicted=predicted, file=filename, sha256=hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                nsfw=flags, block_subtracted_nsfw=block_flags,
                trial_indices=bundle['test_members'][i], input_trials=inputs[i],
                block_ids=block_metadata['block_ids'], block_labels=block_metadata['block_labels'],
                block_concept_definition=bundle['block_concept_definition'],
                block_concept_norm=float(np.linalg.norm(block_vector)),
                predicted_vector_norm=float(np.linalg.norm(vector)),
                block_subtracted_vector_norm=float(np.linalg.norm(residual_vector)),
                block_subtracted_vector_sha256=hashlib.sha256(np.ascontiguousarray(residual_vector).tobytes()).hexdigest(),
                block_concept_fingerprint=config['block_concept_fingerprint'],
                prediction_box=[18 + max(512, image.width) + 12 + (max(512, image.width) - image.width) // 2,
                                panel.height - 18 - max(512, image.height) + (max(512, image.height) - image.height) // 2,
                                18 + max(512, image.width) + 12 + (max(512, image.width) - image.width) // 2 + image.width,
                                panel.height - 18 - max(512, image.height) + (max(512, image.height) - image.height) // 2 + image.height],
                block_subtracted_box=[18 + 2 * (max(512, image.width, block_image.width) + 12)
                                      + (max(512, image.width, block_image.width) - block_image.width) // 2,
                                      panel.height - 18 - max(512, image.height, block_image.height)
                                      + (max(512, image.height, block_image.height) - block_image.height) // 2,
                                      18 + 2 * (max(512, image.width, block_image.width) + 12)
                                      + (max(512, image.width, block_image.width) - block_image.width) // 2 + block_image.width,
                                      panel.height - 18 - max(512, image.height, block_image.height)
                                      + (max(512, image.height, block_image.height) - block_image.height) // 2 + block_image.height])
            done += 1; save(); progress(f'BATCH_PROGRESS {done} {total}')
        state['status'] = 'complete'; save(); return state
    except Exception as exc:
        state['status'] = 'failed'; state['error'] = str(exc); save(); raise


def image_group_split(y, image_ids, test_fraction=.2, validation_fraction=.2):
    """Split image identities within each concept before trial averaging."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError('Test and validation fractions must be between zero and one.')
    y, image_ids = np.asarray(y).astype(str), np.asarray(image_ids).astype(str)
    if len(y) != len(image_ids): raise ValueError('Image identities and trials must align.')
    for identity in set(image_ids):
        if len(set(y[image_ids == identity])) != 1:
            raise ValueError('An image has conflicting concept labels.')
    rng = np.random.default_rng(42)
    partitions = [[], [], []]
    for label in sorted(set(y)):
        identities = np.array(sorted(set(image_ids[y == label])))
        if len(identities) < 3:
            raise ValueError(f'Concept {label!r} needs at least 3 distinct images for train/validation/test.')
        rng.shuffle(identities)
        nt = min(len(identities) - 2, max(1, int(np.ceil(len(identities) * test_fraction))))
        nv = min(len(identities) - nt - 1, max(1, int(np.ceil((len(identities) - nt) * validation_fraction))))
        for dest, group in zip(partitions, [identities[nt + nv:], identities[nt:nt + nv], identities[:nt]]):
            dest.extend(np.flatnonzero(np.isin(image_ids, group)).tolist())
    return tuple(np.asarray(sorted(part), dtype=int) for part in partitions)


def image_retrieval_metrics(predictions, targets, test_ids, labels):
    """Evaluate against unique held-out images, including alternatives of the same concept."""
    norm = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    ids, first = np.unique(test_ids, return_index=True)
    scores = np.einsum('ik,jk->ij', norm(predictions), norm(targets[first]))
    truth = np.searchsorted(ids, test_ids)
    def ranks_and_credit(row, target):
        value = row[target]
        return float((row > value).sum() + ((row == value).sum() - 1) / 2 + 1), float(value == row.max()) / int((row == row.max()).sum())
    ranks, credit, within_ranks, within_credit, counts = [], [], [], [], []
    gallery_labels = np.asarray(labels)[first]
    for i, row in enumerate(scores):
        rank, hit = ranks_and_credit(row, truth[i]);ranks.append(rank);credit.append(hit)
        candidates = np.flatnonzero(gallery_labels == labels[i]);counts.append(len(candidates))
        rank, hit = ranks_and_credit(row[candidates], int(np.flatnonzero(candidates == truth[i])[0]))
        within_ranks.append(rank if len(candidates) > 1 else np.nan)
        within_credit.append(hit if len(candidates) > 1 else np.nan)
    eligible = np.asarray(counts) > 1
    return dict(image_top1_accuracy=float(np.mean(credit)), image_candidate_count=len(ids),
                within_concept_image_top1_accuracy=float(np.mean(np.asarray(within_credit)[eligible])) if eligible.any() else None,
                within_concept_image_chance=float(np.mean(1 / np.asarray(counts)[eligible])) if eligible.any() else None,
                within_concept_image_evaluable_predictions=int(eligible.sum())), ranks, within_ranks, counts


def fit_vectors(payload, options, vectors, paths, image_hashes=None):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, RidgeClassifier
    from sklearn.metrics import confusion_matrix, balanced_accuracy_score
    X=np.asarray(payload['X'],dtype=np.float64);y=np.asarray(payload['y']).astype(str)
    if not np.isfinite(X).all():raise ValueError('Nonfinite neural features')
    stim=payload['trial_df']['stimulus'].astype(str).tolist()
    resolve=lambda s:str(Path(options['generation_image_root'])/Path(s.replace('\\','/')).name)
    if len(stim) != len(y) or X.ndim != 2 or len(X) != len(y):
        raise ValueError('Neural features, concepts and trial metadata must align.')
    resolved=[resolve(name) for name in stim]
    image_ids=np.asarray([image_hashes[path] if image_hashes is not None else path for path in resolved])
    train,val,test=image_group_split(y,image_ids,float(options.get('test_fraction',.2)),float(options.get('validation_fraction',.2)))
    classes=sorted(set(y))
    if len(classes)<2:raise ValueError('At least two concepts are required.')
    vectors=np.asarray(vectors)
    if vectors.shape != (len(paths), 768) or not np.isfinite(vectors).all():
        raise ValueError('Expected one finite 768-dimensional vector per image.')
    lookup=dict(zip(paths,vectors))
    if any(path not in lookup for path in resolved):raise ValueError('Missing stimulus image embedding.')
    # These training-only concept means are retrieval references, never regression targets.
    prototypes = []
    for concept in classes:
        members = train[y[train] == concept]
        _, unique = np.unique(image_ids[members], return_index=True)
        prototypes.append(np.mean([lookup[resolved[i]] for i in members[unique]], axis=0))
    prototypes = np.stack(prototypes)
    n=int(options.get('transformer_trials_per_average',3))
    if n<1:raise ValueError('Averaging must be positive')
    tr,va,te=[average_trials(X,y,idx,n,seed,image_ids=image_ids) for idx,seed in zip([train,val,test],[420,422,421])]
    train_y,val_y,test_y=[np.stack([lookup[resolved[members[0]]] for members in partition[2]]) for partition in [tr,va,te]]

    target_svd_requested = max(0, int(options.get('clip_target_svd_components', options.get('embedding_target_svd_components', 0))))
    decoder_type_str = str(options.get('decoder_type', 'clip_generation')).strip().lower()
    is_hybrid = bool(options.get('clip_hybrid_decoding', False) or options.get('clip_two_stage', False) or decoder_type_str.endswith('hybrid'))
    hybrid_weight_opt = options.get('clip_hybrid_weight', options.get('clip_residual_weight', None))
    fixed_beta = float(hybrid_weight_opt) if hybrid_weight_opt is not None else None

    class_to_idx = {c: i for i, c in enumerate(classes)}

    if is_hybrid:
        stage1_model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=100.0, class_weight='balanced'))
        stage1_model.fit(tr[0], tr[1])
        stage1_val_pred = stage1_model.predict(va[0])
        stage1_val_acc = float(np.mean(stage1_val_pred == va[1]))
        stage1_test_pred = stage1_model.predict(te[0])
        stage1_test_acc = float(np.mean(stage1_test_pred == te[1]))

        train_anchor = np.stack([prototypes[class_to_idx[str(c)]] for c in tr[1]])
        val_anchor = np.stack([prototypes[class_to_idx[str(c)]] for c in stage1_val_pred])
        test_anchor = np.stack([prototypes[class_to_idx[str(c)]] for c in stage1_test_pred])

        T_train = train_y - train_anchor
    else:
        stage1_model = None
        stage1_val_acc = None
        stage1_test_acc = None
        train_anchor = None
        val_anchor = None
        test_anchor = None
        T_train = train_y

    target_center = T_train.mean(0)
    centered_T_train = T_train - target_center
    if target_svd_requested > 0:
        _u, S, Vt = np.linalg.svd(centered_T_train, full_matrices=False)
        tol = float(max(centered_T_train.shape) * np.finfo(np.float64).eps * (S[0] if len(S) else 1.0))
        semantic_rank = int(np.sum(S > tol))
        k = min(target_svd_requested, semantic_rank, centered_T_train.shape[0] - 1)
        if k > 0:
            V_k = Vt[:k].T
            target_train_proj = centered_T_train @ V_k
            total_var = float(np.sum(S ** 2))
            svd_explained_variance = float(np.sum(S[:k] ** 2) / total_var) if total_var > 0 else 1.0
        else:
            V_k = None
            target_train_proj = T_train
            svd_explained_variance = 1.0
            k = 0
    else:
        V_k = None
        target_train_proj = T_train
        svd_explained_variance = 1.0
        k = 0

    candidates = []; best = None
    alpha_list = [1., 10., 100., 1000., 10000.]
    if is_hybrid:
        beta_candidates = [fixed_beta] if fixed_beta is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        beta_candidates = [1.0]

    for alpha in alpha_list:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd')).fit(tr[0], target_train_proj)
        pred_val_raw = model.predict(va[0])
        if V_k is not None:
            reconstructed_val_target = target_center + pred_val_raw @ V_k.T
        else:
            reconstructed_val_target = pred_val_raw

        for beta in beta_candidates:
            if is_hybrid:
                val_pred_full = val_anchor + beta * reconstructed_val_target
            else:
                val_pred_full = reconstructed_val_target

            mse = float(np.mean((val_pred_full - val_y) ** 2))
            if not np.isfinite(mse):
                raise ValueError('Nonfinite validation loss')
            candidate_record = dict(alpha=alpha, validation_mse=mse)
            if is_hybrid:
                candidate_record['beta'] = beta
            candidates.append(candidate_record)
            if best is None or mse < best[0]:
                best = (mse, model, alpha, beta)

    best_model = best[1]
    best_alpha = best[2]
    best_beta = best[3]

    pred_test_raw = best_model.predict(te[0])
    if V_k is not None:
        reconstructed_test_target = target_center + pred_test_raw @ V_k.T
    else:
        reconstructed_test_target = pred_test_raw

    if is_hybrid:
        predictions = test_anchor + best_beta * reconstructed_test_target
    else:
        predictions = reconstructed_test_target

    norm = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    scores = np.einsum('ik,jk->ij', norm(predictions), norm(prototypes))
    truth = np.array([classes.index(c) for c in te[1]]); pred = np.asarray(classes)[scores.argmax(1)]
    # Average zero-based ranks for ties, matching the existing MPR definition.
    correct = scores[np.arange(len(truth)), truth, None]
    rank = (scores > correct).sum(1) + ((scores == correct).sum(1) - 1) / 2
    top5 = np.mean([truth[i] in np.argsort(-s)[:min(5, len(classes))] for i, s in enumerate(scores)])

    decoder_label = 'Image-specific CLIP ridge (experimental)'
    if is_hybrid and k > 0:
        decoder_label = f'Image-specific CLIP hybrid ridge (SVD={k})'
    elif is_hybrid:
        decoder_label = 'Image-specific CLIP hybrid ridge (experimental)'
    elif k > 0:
        decoder_label = f'Image-specific CLIP ridge (SVD={k})'

    summary = dict(
        decoder=decoder_label,
        decoder_type=str(options.get('decoder_type', 'clip_generation')),
        feature_space='source',
        accuracy=float(np.mean(pred == te[1])),
        top1_accuracy=float(np.mean(pred == te[1])),
        top5_accuracy=float(top5),
        mpr=float(np.mean(1 - rank / (len(classes) - 1))),
        balanced_accuracy=float(balanced_accuracy_score(te[1], pred)),
        chance_accuracy=1 / len(classes),
        n_concepts=len(classes),
        n_train=len(tr[0]),
        n_validation=len(va[0]),
        n_test=len(te[0]),
        n_features=X.shape[1],
        ridge_alpha=best_alpha,
        trials_per_average=n,
        split='per-concept image-group split; no repeated image across partitions',
        target_dimensions=768,
        target_svd_requested=target_svd_requested,
        target_svd_components=k,
        target_svd_explained_variance=float(svd_explained_variance),
        hybrid_decoding=is_hybrid,
        hybrid_residual_weight=float(best_beta),
    )
    if is_hybrid:
        summary['stage1_validation_accuracy'] = float(stage1_val_acc)
        summary['stage1_test_accuracy'] = float(stage1_test_acc)

    test_ids = image_ids[[members[0] for members in te[2]]]
    image_metrics, image_ranks, within_ranks, image_counts = image_retrieval_metrics(predictions, test_y, test_ids, te[1])
    target_mean = train_y.mean(0)
    summary.update(
        image_metrics,
        target='individual_stimulus_image',
        test_image_embedding_mse=float(np.mean((predictions - test_y) ** 2)),
        constant_training_mean_mse=float(np.mean((target_mean - test_y) ** 2)),
        n_train_images=len(set(image_ids[train])),
        n_validation_images=len(set(image_ids[val])),
        n_test_images=len(set(image_ids[test])),
    )
    bundle = dict(
        generation_profile=PROFILE,
        model=best_model,
        predictions=predictions,
        prototypes=prototypes,
        class_names=classes,
        test_labels=te[1],
        test_members=te[2],
        stimuli=stim,
        image_root=options['generation_image_root'],
        train_indices=train,
        validation_indices=val,
        test_indices=test,
        validation_candidates=candidates,
        summary=summary,
        test_target_vectors=test_y,
        test_image_ids=test_ids.tolist(),
        image_ids=image_ids.tolist(),
        train_members=tr[2],
        validation_members=va[2],
        training_target_mean=target_mean,
        target_svd_components=k,
        target_svd_projection=V_k if k > 0 else None,
        target_svd_center=target_center if k > 0 else None,
        hybrid_decoding=is_hybrid,
        stage1_model=stage1_model if is_hybrid else None,
    )
    columns = [c for c in ['stimulus', 'trial_index_in_file', 'session_id', 'block_id', 'source_fif', 'source_fif_path'] if c in payload['trial_df'].columns]
    bundle['trial_metadata'] = json.loads(payload['trial_df'][columns].to_json(orient='records'))
    if options.get('generation_output_base'):
        load_block_types(bundle, options['generation_output_base'])
    validate_bundle(bundle)
    bundle['feature_meta_df']=payload.get('feature_meta_df',pd.DataFrame())
    bundle['roi_names']=payload.get('roi_names',[])
    bundle['preprocessing_options']={k:v for k,v in options.items() if not callable(v)}
    table=pd.DataFrame(dict(test_average=np.arange(len(pred)),actual=te[1],predicted=pred,rank=rank+1,trials=[len(m) for m in te[2]]))
    table['stimulus']=[stim[m[0]] for m in te[2]]
    table['image_rank']=image_ranks
    table['within_concept_image_rank']=within_ranks
    table['within_concept_image_candidates']=image_counts
    result=dict(**summary,summary_df=pd.DataFrame([summary]),predictions_df=table,class_names=classes,confusion_matrix=confusion_matrix(te[1],pred,labels=classes),generation_profile=PROFILE)
    return bundle,result


def compute_generation_decoding(payload,options):
    if options.get('split_mode','random')!='random' or float(options.get('validation_fraction',0))<=0:
        raise ValueError('Generation mode requires a random split and positive validation fraction.')
    runtime_python()
    output=Path(options['generation_output_base'])/('clip_generation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    safe={k:v for k,v in options.items() if not callable(v)}
    with tempfile.TemporaryDirectory(prefix='meg_clip_') as temp:
        request=Path(temp)/'request.joblib';joblib.dump(dict(payload=payload,options=safe),request)
        env=dict(os.environ,HF_HUB_OFFLINE='1')
        process=subprocess.Popen(worker_command('train',request,output),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
        log=[]
        for line in process.stdout:
            log.append(line)
            callback=options.get('progress_callback')
            if callable(callback):callback(line.strip())
        if process.wait()!=0:raise RuntimeError('CLIP training failed: '+''.join(log)[-2000:])
    result=joblib.load(output/'result.joblib');result['generation_bundle_path']=str(output/'inference_bundle.joblib')
    joblib.dump(result,output/'result.joblib')
    return result,f"Image-specific CLIP model: image top-1={result['image_top1_accuracy']:.3f}, concept top-1={result['top1_accuracy']:.3f}; saved to {output}"


def main():
    # Invoked directly in the isolated runtime, not imported into the GUI process.
    import torch
    from PIL import Image,ImageDraw
    from torchvision import transforms
    from huggingface_hub import snapshot_download
    import hashlib
    cached=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    for f in cached.rglob('*.bin'):
        blob=f.resolve()
        with blob.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=blob.name:raise ValueError('Corrupt model checkpoint: '+str(f))
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    if sys.argv[1]=='train':
        from transformers import CLIPVisionModelWithProjection
        request=joblib.load(sys.argv[2]);p=request['payload'];o=request['options'];out=Path(sys.argv[3])
        # Frozen encoding of validation/test images supplies evaluation targets, not fitted parameters.
        paths=sorted({str(Path(o['generation_image_root'])/Path(str(name).replace('\\','/')).name) for name in p['trial_df']['stimulus']})
        if not all(Path(f).is_file() for f in paths):raise ValueError('Stimulus image missing from selected folder.')
        encoder=CLIPVisionModelWithProjection.from_pretrained(str(cached/'image_encoder')).eval().to(device)
        tf=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC,antialias=False),transforms.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])])
        cache=Path.home()/'.cache/meg_clip_generation_vectors';cache.mkdir(parents=True,exist_ok=True)
        vectors=[];image_hashes={}
        for j,f in enumerate(paths):
            key=hashlib.sha256(Path(f).read_bytes()+json.dumps(LEGACY_PROFILE,sort_keys=True).encode()).hexdigest();entry=cache/(key+'.npy')
            image_hashes[f]=hashlib.sha256(Path(f).read_bytes()).hexdigest()
            if entry.exists():v=np.load(entry,allow_pickle=False)
            else:
                with Image.open(f) as im:t=tf(im.convert('RGB')).unsqueeze(0).to(device)
                with torch.inference_mode():v=encoder(t).image_embeds.cpu().numpy()[0]
                np.save(entry,v)
            if v.shape!=(768,) or not np.isfinite(v).all():raise ValueError('Invalid cached image vector')
            vectors.append(v)
            if j%32==0:print(f'CLIP targets: {j+1}/{len(paths)}',flush=True)
        bundle,result=fit_vectors(p,o,np.asarray(vectors),paths,image_hashes=image_hashes)
        bundle['image_sha256']=image_hashes
        training_paths={str(Path(o['generation_image_root'])/Path(bundle['stimuli'][i].replace('\\','/')).name) for i in bundle['train_indices']}
        bundle['training_image_sha256']={path:image_hashes[path] for path in training_paths}
        load_block_types(bundle, out)
        ensure_block_concept_vectors(bundle)
        joblib.dump(bundle,out/'inference_bundle.joblib');joblib.dump(result,out/'result.joblib')
        result['predictions_df'].to_csv(out/'predictions.csv',index=False)
        (out/'settings.json').write_text(json.dumps(o,indent=2,default=str));(out/'profile.json').write_text(json.dumps(PROFILE,indent=2))
        print('Training complete',flush=True)
    elif sys.argv[1] in {'generate','batch'}:
        from diffusers import StableDiffusionImageVariationPipeline
        from types import MethodType
        bundle=joblib.load(sys.argv[2]);validate_bundle(bundle);index=int(sys.argv[3]);out=Path(sys.argv[4])
        load_trial_metadata(bundle, sys.argv[2])
        if sys.argv[1]=='generate':out.mkdir(parents=True,exist_ok=False)
        if sys.argv[1]=='generate' and not 0<=index<len(bundle['predictions']):raise ValueError('Invalid held-out average index')
        pipe = None
        def get_pipe():
            nonlocal pipe
            if pipe is None:
                pipe = StableDiffusionImageVariationPipeline.from_pretrained(str(cached),torch_dtype=torch.float32,use_safetensors=False).to(device)
                pipe.enable_attention_slicing()
                pipe._encode_image = MethodType(encode, pipe)
            return pipe
        def encode(self,image,device,num_images_per_prompt,do_classifier_free_guidance):
            v=self.saved_vector.to(device=device,dtype=self.unet.dtype).unsqueeze(1).repeat_interleave(num_images_per_prompt,dim=0)
            return torch.cat([torch.zeros_like(v),v]) if do_classifier_free_guidance else v
        def encode_block_exemplar(path):
            transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC,antialias=False),transforms.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])])
            with Image.open(path) as image:
                tensor=transform(image.convert('RGB')).unsqueeze(0).to(device)
            with torch.inference_mode():
                return get_pipe().image_encoder(tensor).image_embeds.cpu().numpy()[0]
        ensure_block_concept_vectors(bundle, encode_image=encode_block_exemplar)
        if sys.argv[1]=='batch':
            def render(vector):
                pipe = get_pipe()
                pipe.saved_vector=torch.tensor(vector[None],dtype=torch.float32)
                result=pipe(image=torch.zeros(1,3,224,224),num_inference_steps=30,guidance_scale=3.,generator=torch.Generator('cpu').manual_seed(42),height=512,width=512)
                return result.images[0],result.nsfw_content_detected
            state=generate_batch(bundle,out,render,cancelled=lambda:(out/'STOP').exists(),progress=lambda s:print(s,flush=True))
            print('BATCH_STATUS '+state['status'],flush=True)
            return
        pipe = get_pipe()
        label=str(bundle['test_labels'][index]);members=bundle['test_members'][index]
        panel=Image.new('RGB',(2048,550),'white');draw=ImageDraw.Draw(panel)
        source=Path(bundle['image_root'])/Path(bundle['stimuli'][members[0]].replace('\\','/')).name
        if source.is_file():
            with Image.open(source) as im:panel.paste(im.convert('RGB').resize((512,512)),(0,30))
        draw.text((8,8),f'One input stimulus ({len(members)} trials averaged)',fill='black')
        manifest=dict(profile=bundle['generation_profile'],index=index,actual=label,members=members,seed=42,steps=30,guidance=3,text_prompt=None,outputs={},warning='Experimental visualization, not evidence of accurate brain reconstruction.')
        manifest['bundle_path']=str(Path(sys.argv[2]).resolve())
        manifest['input_stimuli']=[bundle['stimuli'][i] for i in members]
        ideal = ('ideal_image', bundle['test_target_vectors'][index]) if bundle['generation_profile'] == PROFILE else ('ideal_concept',bundle['prototypes'][bundle['class_names'].index(label)])
        block_vector, block_metadata = block_concept_for_index(bundle, index)
        block_subtracted_vector = np.asarray(bundle['predictions'][index]) - block_vector
        manifest['block_concept'] = dict(
            **block_metadata,
            definition=bundle['block_concept_definition'],
            norm=float(np.linalg.norm(block_vector)),
            block_subtracted_norm=float(np.linalg.norm(block_subtracted_vector)),
        )
        outputs=[ideal,('meg_prediction',bundle['predictions'][index]),('meg_prediction_minus_block_concept',block_subtracted_vector)]
        for col,(kind,vector) in enumerate(outputs,1):
            pipe.saved_vector=torch.tensor(vector[None],dtype=torch.float32);np.save(out/(kind+'.npy'),vector)
            print('Generating '+kind,flush=True)
            r=pipe(image=torch.zeros(1,3,224,224),num_inference_steps=30,guidance_scale=3.,generator=torch.Generator('cpu').manual_seed(42),height=512,width=512)
            r.images[0].save(out/(kind+'.png'));panel.paste(r.images[0],(col*512,30));draw.text((col*512+8,8),kind.replace('_',' '),fill='black')
            manifest['outputs'][kind]={'nsfw':r.nsfw_content_detected,'norm':float(np.linalg.norm(vector))}
        panel.save(out/'comparison.png');(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Generation complete',flush=True)
    else:raise ValueError('Unknown operation')


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    main()
