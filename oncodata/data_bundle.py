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

  ~/.cache/oncodata/bundled_data/v<DATA_VERSION>/
    cancer-reference-expression/...
    cancer-reference-expression-percentiles/...
    pan-cancer-expression.csv
    hpa-cell-type-expression.csv

oncodata now owns the bundle: new downloads land under ``~/.cache/oncodata``
and prefer the ``pirl-unc/oncodata`` release. To avoid a forced re-download
during the migration, an already-populated legacy ``~/.cache/pirlygenes`` cache
for the current version is reused as-is, and the fetch falls back to the
``pirl-unc/pirlygenes`` release if oncodata hasn't published this version's
tarball yet. The ``PIRLYGENES_BUNDLED_DATA`` env var is still honored;
``CANCERDATA_BUNDLED_DATA`` takes precedence when set.

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
import urllib.error
import urllib.request
from pathlib import Path

from .version import DATA_VERSION


def _release_url(repo: str, filename: str) -> str:
    return f"https://github.com/{repo}/releases/download/v{DATA_VERSION}/{filename}"


# oncodata owns the bundle: its own release is tried first. The historical
# pirlygenes release is kept as a fallback for one migration release so a version
# whose oncodata tarball isn't uploaded yet still fetches (a 404 from the
# primary falls through to the fallback rather than hanging the user).
GITHUB_REPO = "pirl-unc/oncodata"
TARBALL_FILENAME = f"oncodata-data-v{DATA_VERSION}.tar.gz"
RELEASE_URL = _release_url(GITHUB_REPO, TARBALL_FILENAME)

FALLBACK_GITHUB_REPO = "pirl-unc/pirlygenes"
FALLBACK_TARBALL_FILENAME = f"pirlygenes-data-v{DATA_VERSION}.tar.gz"
FALLBACK_RELEASE_URL = _release_url(FALLBACK_GITHUB_REPO, FALLBACK_TARBALL_FILENAME)

#: Release URLs tried in order until one downloads.
RELEASE_URLS: tuple[str, ...] = (RELEASE_URL, FALLBACK_RELEASE_URL)

#: Env var that overrides the cache (points at the version-pinned dir).
CACHE_DIR_ENV_VAR = "CANCERDATA_BUNDLED_DATA"
#: Back-compat env var honored when this package's own override is unset.
LEGACY_CACHE_DIR_ENV_VAR = "PIRLYGENES_BUNDLED_DATA"

#: Default cache parents. New downloads go under the oncodata root; an existing
#: pirlygenes cache for the current version is reused to avoid a re-download.
_DEFAULT_CACHE_PARENT = Path.home() / ".cache" / "oncodata" / "bundled_data"
_LEGACY_CACHE_PARENT = Path.home() / ".cache" / "pirlygenes" / "bundled_data"

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
    """Parent of all version-pinned cache dirs (``v<version>/`` lives inside).

    Defaults to the oncodata cache root, but if *this* version was already
    downloaded under the legacy pirlygenes root (and not yet under the new one),
    that legacy root is returned so the migration doesn't force a re-download.
    """
    override = _cache_override()
    if override:
        # Override points at the version-pinned dir; its parent is the root.
        return Path(override).expanduser().parent
    version_dir = f"v{DATA_VERSION}"
    if (
        not (_DEFAULT_CACHE_PARENT / version_dir).exists()
        and (_LEGACY_CACHE_PARENT / version_dir).exists()
    ):
        return _LEGACY_CACHE_PARENT
    return _DEFAULT_CACHE_PARENT


def cache_dir() -> Path:
    """Where the downloaded bundle lives on disk for this version."""
    override = _cache_override()
    if override:
        return Path(override).expanduser()
    return cache_root() / f"v{DATA_VERSION}"


def _path_complete(path: Path) -> bool:
    """A downloadable entry counts as present only if it actually holds data.

    A bare file must exist; a directory must contain at least one file. An
    interrupted extract that created the shard directories but no shards (or a
    legacy/empty cache dir) would otherwise read as "local" forever and never
    re-fetch — later reads then fail with confusing empties (issue #21)."""
    if not path.exists():
        return False
    if path.is_dir():
        return any(f.is_file() for f in path.rglob("*"))
    return True


def is_local() -> bool:
    """Every downloadable path is present AND non-empty in the cache."""
    root = cache_dir()
    return all(_path_complete(root / p) for p in DOWNLOADABLE_PATHS)


def find(relative_path: str) -> Path | None:
    """Resolve a downloadable file to its on-disk cached location, or None."""
    candidate = cache_dir() / relative_path
    return candidate if candidate.exists() else None


def _download_and_extract(url: str, root: Path, *, verbose: bool) -> None:
    """Download a tarball from ``url`` and extract it into ``root``.

    Extraction goes into a staging dir first and is promoted into ``root`` only
    after it completes, so a failed/corrupt download never leaves a half-populated
    cache that a later run would read as valid. Raises ``urllib.error.URLError``
    (unreachable, e.g. 404) or ``tarfile.TarError`` (corrupt/non-tar body) — the
    caller falls back to the next source on either.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as h:
            shutil.copyfileobj(resp, h, length=1024 * 1024)
        if verbose:
            sys.stderr.write("oncodata: extracting...\n")
            sys.stderr.flush()
        with tarfile.open(tmp_path) as tf:
            # filter=data is Python 3.12+; fall back to the older API.
            try:
                tf.extractall(staging, filter="data")
            except TypeError:
                tf.extractall(staging)
        # Promote staged entries into the cache, replacing any prior copy.
        for entry in staging.iterdir():
            dest = root / entry.name
            if dest.is_dir():
                shutil.rmtree(dest)
            elif dest.exists():
                dest.unlink()
            shutil.move(str(entry), str(dest))
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def fetch(*, verbose: bool = True) -> Path:
    """Download + extract the bundle for this version into the cache.

    Tries each URL in :data:`RELEASE_URLS` (oncodata first, pirlygenes fallback)
    until one succeeds, so a version not yet published on the oncodata release
    transparently falls back. Always overwrites — safe to call to repair a corrupt
    cache. Returns the cache directory.
    """
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in RELEASE_URLS:
        if verbose:
            sys.stderr.write(
                f"oncodata: downloading data bundle for v{DATA_VERSION} "
                "(~350 MB, one-time)\n"
                f"  from {url}\n"
                f"  to   {root}\n"
            )
            sys.stderr.flush()
        try:
            _download_and_extract(url, root, verbose=verbose)
        except (urllib.error.URLError, tarfile.TarError) as e:
            # URLError covers HTTPError (404); TarError covers a corrupt body or
            # an HTML error page served with 200 — fall back to the next source.
            errors.append(f"{url}: {e}")
            if verbose:
                sys.stderr.write(f"oncodata: {url} unavailable ({e}); trying next source\n")
                sys.stderr.flush()
            continue
        if verbose:
            sys.stderr.write(f"oncodata: data bundle ready at {root}\n")
            sys.stderr.flush()
        return root
    raise RuntimeError(
        "oncodata: could not download the data bundle from any release source:\n  "
        + "\n  ".join(errors)
    )


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
            f"oncodata data bundle not found at {cache_dir()}. Run `oncodata fetch` to download it."
        )
    return fetch(verbose=verbose)


def status() -> dict:
    """Snapshot of cache state — used by ``oncodata status``."""
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
        "release_urls": list(RELEASE_URLS),
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
    version label. Used by ``oncodata prune`` to find upgrade leftovers.
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
    "FALLBACK_GITHUB_REPO",
    "FALLBACK_RELEASE_URL",
    "FALLBACK_TARBALL_FILENAME",
    "GITHUB_REPO",
    "RELEASE_URL",
    "RELEASE_URLS",
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
