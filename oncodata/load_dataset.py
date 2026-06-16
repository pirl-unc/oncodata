# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loader for bundled + downloaded data files.

Two data roots are checked in order:

  1. ``_BUNDLED_DATA_DIR`` — files shipped in the wheel (the small curated
     tables) AND files present in a git checkout's ``oncodata/data/``.
  2. the cache populated by :mod:`oncodata.data_bundle` (the large per-cohort
     expression summaries fetched from the GitHub Release).

Any file present in (1) wins over (2). When a callable requests one of the
:data:`oncodata.data_bundle.DOWNLOADABLE_PATHS` items and it's missing from
both, :func:`oncodata.data_bundle.ensure_local` triggers a one-time download.
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path

import pandas as pd

from . import data_bundle

_BUNDLED_DATA_DIR = Path(__file__).parent / "data"
_DATASET_PATHS = None
_CACHED_DATAFRAMES: dict = {}

# Back-compat alias.
_DATA_DIR = _BUNDLED_DATA_DIR

#: Clear callbacks for caches DERIVED from get_data results (e.g. processed
#: lookups in incidence / cta). Each cache-holder registers its ``cache_clear`` via
#: :func:`_register_derived_cache`; :func:`_clear_cache` drives them so swapping a
#: bundled fixture invalidates the derived views too — no module needs to know
#: about another's caches.
_DERIVED_CACHE_CLEARERS: list = []


def _register_derived_cache(clear_fn) -> None:
    """Register a ``cache_clear``-style callback to run on :func:`_clear_cache`."""
    _DERIVED_CACHE_CLEARERS.append(clear_fn)


def _clear_cache() -> None:
    """Drop cached frames + dataset-path map + every derived cache. Test hook for
    swapping fixtures."""
    _CACHED_DATAFRAMES.clear()
    _invalidate_dataset_paths()
    for clear_fn in _DERIVED_CACHE_CLEARERS:
        clear_fn()


def _data_roots() -> list[Path]:
    """Roots checked when resolving a data file, in priority order."""
    return [_BUNDLED_DATA_DIR, data_bundle.cache_dir()]


def _ensure_downloadable(name: str) -> None:
    """Fetch the bundle if ``name`` maps to a downloadable item missing from
    BOTH the bundled checkout AND the cache. No-op when present in either."""
    stem_with_csv = name if name.endswith(".csv") else f"{name}.csv"
    stem = name.removesuffix(".csv").removesuffix(".gz")
    candidates = {name, stem, stem_with_csv, stem_with_csv.removesuffix(".csv")}
    for cand in candidates:
        if not data_bundle.is_downloadable(cand):
            continue
        if (_BUNDLED_DATA_DIR / cand).exists():
            return
        if data_bundle.find(cand) is not None:
            return
        data_bundle.ensure_local()
        return


def _shard_directories() -> list[Path]:
    """Subdirectories holding sharded CSV datasets, gathered from both roots.

    A shard directory ``<root>/<name>/`` containing one or more ``*.csv.gz``
    files acts as a single logical dataset addressable as ``<name>`` via
    :func:`get_data`; its shards are concatenated transparently.
    """
    seen: dict[str, Path] = {}
    for root in _data_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if any(child.glob("*.csv")) or any(child.glob("*.csv.gz")):
                seen.setdefault(child.name, child)
    return [seen[name] for name in sorted(seen)]


def _shard_paths(shard_dir: Path) -> list[Path]:
    return sorted(list(shard_dir.glob("*.csv")) + list(shard_dir.glob("*.csv.gz")))


def get_all_csv_paths() -> list:
    """Paths to every top-level CSV file across both data roots (bundled wins)."""
    seen: dict[str, Path] = {}
    for root in _data_roots():
        if not root.exists():
            continue
        for p in sorted(list(root.glob("*.csv")) + list(root.glob("*.csv.gz"))):
            seen.setdefault(p.name, p)
    return list(seen.values())


def _load_shard_directory(shard_dir: Path) -> pd.DataFrame:
    """Concatenate every ``*.csv[.gz]`` shard in a sharded dataset directory.

    Keeps a best-effort parquet cache of the concatenated frame in
    ``~/.cache/oncodata/shard_cache/``, keyed on a signature of the shard
    files (count + total size + newest mtime), auto-invalidating on change.
    """
    paths = _shard_paths(shard_dir)
    if not paths:
        raise FileNotFoundError(f"no CSV shards found under {shard_dir}")
    sig = repr(
        (len(paths), sum(p.stat().st_size for p in paths), max(p.stat().st_mtime_ns for p in paths))
    )
    cache_dir = Path.home() / ".cache" / "oncodata" / "shard_cache"
    cache_file = cache_dir / f"{shard_dir.name}.parquet"
    sig_file = cache_dir / f"{shard_dir.name}.sig"
    try:
        if cache_file.exists() and sig_file.exists() and sig_file.read_text() == sig:
            return pd.read_parquet(cache_file)
    except Exception as e:
        # Corrupt/unreadable cache: self-heal by removing it (so it doesn't fail
        # every run) and surface a warning, then rebuild from the authoritative
        # CSVs below.
        warnings.warn(
            f"oncodata: rebuilding unreadable shard cache {cache_file.name}: {e}",
            stacklevel=2,
        )
        for stale in (cache_file, sig_file):
            with contextlib.suppress(OSError):
                stale.unlink()
    df = pd.concat([pd.read_csv(str(p), low_memory=False) for p in paths], ignore_index=True)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        sig_file.write_text(sig)
    except Exception as e:
        # Caching is best-effort — a write failure (disk full, read-only FS)
        # must never fail the load, but make it visible.
        warnings.warn(
            f"oncodata: could not write shard cache {cache_file.name}: {e}",
            stacklevel=2,
        )
    return df


def _invalidate_dataset_paths() -> None:
    global _DATASET_PATHS
    _DATASET_PATHS = None


def _dataset_paths():
    """Map accepted dataset names to their on-disk CSV path or shard dir."""
    global _DATASET_PATHS
    if _DATASET_PATHS is not None:
        return _DATASET_PATHS

    paths: dict[str, Path] = {}
    for csv_path in get_all_csv_paths():
        csv_key = csv_path.name.removesuffix(".gz")
        stem_key = csv_key.removesuffix(".csv")
        for key in {csv_key, csv_key.lower(), stem_key, stem_key.lower()}:
            paths[key] = csv_path
    for shard_dir in _shard_directories():
        stem_key = shard_dir.name
        csv_key = stem_key + ".csv"
        for key in {csv_key, csv_key.lower(), stem_key, stem_key.lower()}:
            paths[key] = shard_dir
    _DATASET_PATHS = paths
    return paths


def get_data(name, _dataframes_dict=None, *, copy=True):
    """Load a packaged dataset as a DataFrame.

    By default returns a defensive ``.copy()`` so callers that mutate in place
    can't corrupt the shared cache. Pass ``copy=False`` for read-only callers
    that slice/copy before mutating.
    """
    candidates = [name, name.lower()]
    for candidate in list(candidates):
        candidates.append(candidate + ".csv")

    if _dataframes_dict is not None:
        for candidate in candidates:
            if candidate in _dataframes_dict:
                return _dataframes_dict[candidate].copy() if copy else _dataframes_dict[candidate]
        raise ValueError(f"Dataset {name} not found")

    # Trigger download for downloadable items before resolving paths.
    _ensure_downloadable(name)
    paths = _dataset_paths()

    miss = not any(c in paths for c in candidates)
    if miss and data_bundle.is_downloadable(name):
        data_bundle.ensure_local()
        _invalidate_dataset_paths()
        paths = _dataset_paths()

    for candidate in candidates:
        if candidate in paths:
            resolved = paths[candidate]
            if resolved.is_dir():
                cache_key = resolved.name + ".csv"
                if cache_key not in _CACHED_DATAFRAMES:
                    _CACHED_DATAFRAMES[cache_key] = _load_shard_directory(resolved)
            else:
                cache_key = resolved.name.removesuffix(".gz")
                if cache_key not in _CACHED_DATAFRAMES:
                    _CACHED_DATAFRAMES[cache_key] = pd.read_csv(str(resolved), low_memory=False)
            cached = _CACHED_DATAFRAMES[cache_key]
            return cached.copy() if copy else cached
    raise ValueError(f"Dataset {name} not found")
