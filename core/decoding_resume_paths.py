"""Relocate checkpoint paths in memory without altering archived inputs."""
import os
from pathlib import Path

import pandas as pd


class ResumePathResolver:
    def __init__(self, folder, old_folder, search_roots=(), path_maps=(), log=print):
        self.folder = Path(folder).resolve()
        self.old_folder = str(old_folder or '').rstrip('/')
        self.maps = list(path_maps)
        self.log = log
        self.cache = {}
        self.index = {}
        roots = [self.folder, *search_roots]
        if not search_roots:
            roots.extend(p for p in list(self.folder.parents)[:4]
                         if len(p.parts) >= 3 and p != Path.home())
        self.roots = list(dict.fromkeys(Path(p).resolve() for p in roots))
        for root in self.roots:
            if not root.is_dir():
                raise ValueError(f'Resume search directory not found: {root}')

    def resolve(self, value, required=False):
        if not isinstance(value, str) or not value:
            return value
        if not os.path.isabs(value):
            if required:
                return self.resolve(str(self.folder / value), required=True)
            return value
        # Batch-owned result paths must point to the resumed copy even when
        # the original batch still exists on the same machine.
        if self.old_folder and (value == self.old_folder or value.startswith(self.old_folder + '/')):
            relocated = self.folder / os.path.relpath(value, self.old_folder)
            if not required or relocated.is_file():
                return str(relocated)
        if Path(value).exists():
            return value
        cache_key = (value, required)
        if cache_key in self.cache:
            result = self.cache[cache_key]
        else:
            matches = set()
            for old, new in self.maps:
                if value == old or value.startswith(old + '/'):
                    candidate = Path(new + value[len(old):])
                    if candidate.exists():
                        matches.add(str(candidate.resolve()))
            # Historical FIF/anatomy paths are provenance, not dependencies.
            # Rebase/map them cheaply, but never scan the filesystem for them.
            if not matches and not required:
                return value
            # Explicit prefix mappings take precedence over automatic search.
            if not matches:
                name = Path(value).name
                if name not in self.index:
                    found = set()
                    # Walk overlapping roots only once; do not follow symlink directories.
                    roots = [r for r in self.roots if not any(other != r and other in r.parents for other in self.roots)]
                    visited = 0
                    for root in roots:
                        self.log(f"Searching for required checkpoint file {name} under {root}...")
                        for directory, dirs, files in os.walk(root):
                            visited += 1
                            if visited % 1000 == 0:
                                self.log(f"Path search: scanned {visited} directories; currently {directory}")
                            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv'}]
                            if name in files:
                                found.add(str((Path(directory) / name).resolve()))
                    self.index[name] = found
                    self.log(f"Path search complete: scanned {visited} directories.")
                matches = self.index[name]
            if len(matches) > 1 and not required:
                self.log(f"Leaving ambiguous historical path unchanged: {value}")
                return value
            if len(matches) > 1:
                raise ValueError(f'Ambiguous relocated path {value}: {sorted(matches)}. Use --path-map OLD=NEW or a narrower --resume-search-root.')
            result = next(iter(matches), None)
            self.cache[cache_key] = result
            if result:
                self.log(f'Remapped checkpoint path: {value} -> {result}')
        if result:
            return result
        if required:
            raise FileNotFoundError(f'Cannot locate checkpoint dependency: {value}. Copy it and use --resume-search-root or --path-map OLD=NEW.')
        return value  # Unneeded FIF/anatomy provenance can remain historical.

    def transform(self, value):
        if isinstance(value, dict):
            return {k: self.resolve(v, required=True) if k == 'embedding_dict_path' and isinstance(v, str)
                    else self.transform(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.transform(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.transform(v) for v in value)
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
            for column in frame:
                if 'path' in str(column) or str(column) == 'output_dir':
                    frame[column] = frame[column].map(self.resolve)
            return frame
        return self.resolve(value)
