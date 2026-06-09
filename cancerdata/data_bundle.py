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

"""Lazy download of the large per-cohort expression bundle from a GitHub Release.

The wheel ships small curated tables directly (cancer-type registry, TMB,
incidence/mortality — ~60 KB). The much larger per-cohort expression summaries
and percentile vectors (~340 MB) download on first access from the version-pinned
GitHub Release.

Cache layout (version-pinned so upgrades trigger a re-fetch):

  ~/.cache/pirlygenes/bundled_data/v<DATA_VERSION>/
    cancer-reference-expression/...
    cancer-reference-expression-percentiles/...
    pan-cancer-expression.csv
    hpa-cell-type-expression.csv

The historical ``~/.cache/pirlygenes`` root and ``PIRLYGENES_BUNDLED_DATA`` env
var are preserved (this data used to be fetched by pirlygenes) so existing caches
are reused as-is; ``CANCERDATA_BUNDLED_DATA`` takes precedence when set. The
bundle is still hosted on the pirlygenes releases until its ownership migrates.

Public API:
  cache_dir()      → version-pinned cache Path
  is_local()       → bool: every downloadable path present?
  fetch()          → download + extract from the GitHub Release
  ensure_local()   → fetch if missing; safe to call on every access
  find(path)       → cached path or None
  status()         → dict summarizing local state
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .version import DATA_VERSION

# The bundle is hosted on the pirlygenes releases for now (the data historically
# lived there); ownership migrates to a cancerdata release in a later milestone.
GITHUB_REPO = "pirl-unc/pirlygenes"
TARBALL_FILENAME = f"pirlygenes-data-v{DATA_VERSION}.tar.gz"
RELEASE_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/download/v{DATA_VERSION}/{TARBALL_FILENAME}"
)

#: Env var that overrides the cache (points at the version-pinned dir).
CACHE_DIR_ENV_VAR = "CANCERDATA_BUNDLED_DATA"
#: Back-compat env var honored when this package's own override is unset.
LEGACY_CACHE_DIR_ENV_VAR = "PIRLYGENES_BUNDLED_DATA"

# Names that live in the downloadable tarball (relative to the cache root) and
# are NOT bundled in the wheel. load_dataset checks here after the wheel data dir.
DOWNLOADABLE_PATHS: tuple[str, ...] = (
    "cancer-reference-expression",  # directory of per-source shards
    "cancer-reference-expression-representatives",  # per-cohort medoid parquets
    "cancer-reference-expression-percentiles",  # per-gene percentile vectors
    "pan-cancer-expression.csv",
    "hpa-cell-type-expression.csv",
)


def _cache_override() -> str | None:
    return os.environ.get(CACHE_DIR_ENV_VAR) or os.environ.get(LEGACY_CACHE_DIR_ENV_VAR)


def cache_root() -> Path:
    """Parent of all version-pinned cache dirs (``v<version>/`` lives inside)."""
    override = _cache_override()
    if override:
        # Override points at the version-pinned dir; its parent is the root.
        return Path(override).expanduser().parent
    return Path.home() / ".cache" / "pirlygenes" / "bundled_data"


def cache_dir() -> Path:
    """Where the downloaded bundle lives on disk for this version."""
    override = _cache_override()
    if override:
        return Path(override).expanduser()
    return cache_root() / f"v{DATA_VERSION}"


def is_local() -> bool:
    """Every downloadable path exists in the cache for this version."""
    root = cache_dir()
    return all((root / p).exists() for p in DOWNLOADABLE_PATHS)


def find(relative_path: str) -> Path | None:
    """Resolve a downloadable file to its on-disk cached location, or None."""
    candidate = cache_dir() / relative_path
    return candidate if candidate.exists() else None


def fetch(*, verbose: bool = True) -> Path:
    """Download + extract the bundle for this version into the cache.

    Always overwrites — safe to call to repair a corrupt cache. Returns the
    cache directory.
    """
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    if verbose:
        sys.stderr.write(
            f"cancerdata: downloading data bundle for v{DATA_VERSION} "
            "(~350 MB, one-time)\n"
            f"  from {RELEASE_URL}\n"
            f"  to   {root}\n"
        )
        sys.stderr.flush()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(RELEASE_URL) as resp, tmp_path.open("wb") as h:
            shutil.copyfileobj(resp, h, length=1024 * 1024)
        if verbose:
            sys.stderr.write("cancerdata: extracting...\n")
            sys.stderr.flush()
        with tarfile.open(tmp_path) as tf:
            # filter=data is Python 3.12+; fall back to the older API.
            try:
                tf.extractall(root, filter="data")
            except TypeError:
                tf.extractall(root)
    finally:
        tmp_path.unlink(missing_ok=True)
    if verbose:
        sys.stderr.write(f"cancerdata: data bundle ready at {root}\n")
        sys.stderr.flush()
    return root


def ensure_local(*, auto_fetch: bool = True, verbose: bool = True) -> Path:
    """Make sure the bundle is present locally; download if not.

    With ``auto_fetch=False``, raises ``FileNotFoundError`` instead of
    triggering a network call — for read-only inspection paths that shouldn't
    surprise users with a 340 MB download.
    """
    if is_local():
        return cache_dir()
    if not auto_fetch:
        raise FileNotFoundError(
            f"cancerdata data bundle not found at {cache_dir()}. "
            "Run `cancerdata fetch` to download it."
        )
    return fetch(verbose=verbose)


def status() -> dict:
    """Snapshot of cache state — used by ``cancerdata status``."""
    root = cache_dir()
    items: dict[str, dict] = {}
    for p in DOWNLOADABLE_PATHS:
        path = root / p
        size_bytes = 0
        if path.exists():
            if path.is_dir():
                size_bytes = sum(
                    (f.stat().st_size for f in path.rglob("*") if f.is_file()),
                    start=0,
                )
            else:
                size_bytes = path.stat().st_size
        items[p] = {
            "present": path.exists(),
            "path": str(path),
            "size_bytes": size_bytes,
        }
    return {
        "data_version": DATA_VERSION,
        "cache_dir": str(root),
        "release_url": RELEASE_URL,
        "items": items,
        "all_local": is_local(),
    }


def is_downloadable(relative_path: str) -> bool:
    """True if ``relative_path`` falls under one of the downloadable roots."""
    parts = Path(relative_path).parts
    if not parts:
        return False
    first = parts[0]
    return first in DOWNLOADABLE_PATHS or relative_path in DOWNLOADABLE_PATHS


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def list_cache_versions() -> list[dict]:
    """Enumerate every version-pinned cache dir under :func:`cache_root`.

    Returns ``{"version", "path", "size_bytes", "is_current"}`` dicts sorted by
    version label. Used by ``cancerdata prune`` to find upgrade leftovers.
    """
    root = cache_root()
    if not root.exists():
        return []
    current = cache_dir()
    out: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("v"):
            continue
        out.append(
            {
                "version": child.name,
                "path": child,
                "size_bytes": _dir_size_bytes(child),
                "is_current": child.resolve() == current.resolve(),
            }
        )
    return out


def prune_cache(*, keep_current: bool = True, dry_run: bool = False) -> list[dict]:
    """Delete every version-pinned cache dir EXCEPT the installed version (when
    ``keep_current=True``). With ``dry_run=True`` returns the candidate list
    without touching disk. Returns the list of dirs deleted (or planned)."""
    candidates = []
    for entry in list_cache_versions():
        if keep_current and entry["is_current"]:
            continue
        candidates.append(entry)
    if dry_run:
        return candidates
    for entry in candidates:
        path = entry["path"]
        for child in sorted(path.rglob("*"), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    child.rmdir()
            except OSError:
                pass
        with contextlib.suppress(OSError):
            path.rmdir()
    return candidates


__all__ = [
    "DOWNLOADABLE_PATHS",
    "GITHUB_REPO",
    "RELEASE_URL",
    "TARBALL_FILENAME",
    "cache_dir",
    "cache_root",
    "ensure_local",
    "fetch",
    "find",
    "is_downloadable",
    "is_local",
    "list_cache_versions",
    "prune_cache",
    "status",
]
