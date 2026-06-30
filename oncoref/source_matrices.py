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

"""Raw per-sample TPM matrices, fetched **per cohort**.

These are the rawest cohort expression — the build inputs every derived artifact
(percentiles, exemplars, within-sample, proteoform sums, CTA regen) is computed
from. They total ~21 GB across 118 cohorts, so each cohort's matrix is an
individually fetchable release asset: pull only the cohorts you need rather than
one monolithic blob.

    from oncoref import source_matrices as sm
    sm.available_cohorts()            # the 118 cohorts with a per-sample matrix
    sm.ensure("LUAD")                 # download LUAD's matrix -> Path
    pd.read_parquet(sm.ensure("LUAD"))

A shipped registry (``source-matrices.csv``: cancer_code, source_cohort,
n_samples) lists what's available without any download. Cache layout:

    ~/.cache/oncoref/source-matrices/v<SOURCE_MATRIX_VERSION>/<CODE>.parquet
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .cancer_types import resolve_cancer_type
from .load_dataset import get_data
from .version import SOURCE_MATRIX_VERSION

#: Per-cohort matrices are release assets on a dedicated tag of oncoref's repo. Pinned to
#: SOURCE_MATRIX_VERSION (the raw-input version), NOT the derived-bundle DATA_VERSION — a
#: canonical-space bundle re-release must not repoint or orphan these unchanged raw matrices.
GITHUB_REPO = "pirl-unc/oncoref"
RELEASE_TAG = f"source-v{SOURCE_MATRIX_VERSION}"

#: Env var overriding the per-cohort cache root.
CACHE_DIR_ENV_VAR = "CANCERDATA_SOURCE_MATRICES"


class SourceMatrixError(RuntimeError):
    """Unknown cohort or per-cohort download failure."""


@lru_cache(maxsize=1)
def registry() -> pd.DataFrame:
    """The per-cohort registry (``cancer_code``, ``source_cohort``, ``n_samples``)
    — every cohort with a raw per-sample matrix. Defensive copy."""
    return get_data("source-matrices").copy()


def available_cohorts() -> list[str]:
    """Cancer codes that have a per-sample matrix (sorted)."""
    return sorted(registry()["cancer_code"].astype(str))


def _registry_index() -> dict[str, dict]:
    return {str(r["cancer_code"]): dict(r) for _, r in registry().iterrows()}


def _resolve(code: str) -> str:
    resolved = resolve_cancer_type(code, strict=False) or code
    if resolved not in _registry_index():
        raise SourceMatrixError(
            f"no per-sample matrix for {code!r}; see source_matrices.available_cohorts()"
        )
    return resolved


def cohort_info(code: str) -> dict:
    """Registry row for a cohort (``source_cohort``, ``n_samples``)."""
    return _registry_index()[_resolve(code)]


def cache_dir() -> Path:
    """Per-cohort cache directory for this data version (created on demand)."""
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    base = (
        Path(override).expanduser()
        if override
        else (Path.home() / ".cache" / "oncoref" / "source-matrices")
    )
    out = base / f"v{SOURCE_MATRIX_VERSION}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def local_path(code: str) -> Path:
    """Expected cache path for a cohort's matrix (may not exist yet)."""
    return cache_dir() / f"{_resolve(code)}.parquet"


def is_cached(code: str) -> bool:
    return local_path(code).exists()


def release_url(code: str) -> str:
    resolved = _resolve(code)
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download/"
        f"{RELEASE_TAG}/{resolved}_per_sample_tpm.parquet"
    )


def fetch(code: str, *, force: bool = False, verbose: bool = True) -> Path:
    """Download one cohort's per-sample matrix into the cache. Returns the path."""
    dest = local_path(code)
    if dest.exists() and not force:
        return dest
    url = release_url(code)
    tmp = dest.with_suffix(".parquet.part")
    if verbose:
        sys.stderr.write(f"oncoref: downloading per-sample matrix {dest.stem} ({url})\n")
        sys.stderr.flush()
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as h:
            shutil.copyfileobj(resp, h, length=1024 * 1024)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise SourceMatrixError(
            f"failed to download per-sample matrix for {code!r} ({url}): {e}"
        ) from e
    return dest


def ensure(code: str) -> Path:
    """Local path to a cohort's per-sample matrix, downloading if absent."""
    dest = local_path(code)
    return dest if dest.exists() else fetch(code)


def sample_qc(
    code: str,
    *,
    auto_fetch: bool = True,
    min_detected_genes: int | None = None,
    min_housekeeping_detected: int | None = None,
    max_top_gene_fraction: float | None = None,
    max_top10_gene_fraction: float | None = None,
) -> pd.DataFrame:
    """Compute source-matrix sample QC for one cohort.

    This is the ``source_matrices``-side entry point for the shared QC policy used
    by expression reads and expression-artifact rebuilds. It delegates to
    :func:`oncoref.expression.sample_expression_qc`, preserving the same columns:
    detected-gene counts, source-scale class, top-gene concentration,
    housekeeping-panel detection, ``sample_qc_status``, and reasons.

    Optional threshold arguments default to the expression module's policy values.
    """
    from . import expression

    kwargs = {
        "auto_fetch": auto_fetch,
    }
    if min_detected_genes is not None:
        kwargs["min_detected_genes"] = min_detected_genes
    if min_housekeeping_detected is not None:
        kwargs["min_housekeeping_detected"] = min_housekeeping_detected
    if max_top_gene_fraction is not None:
        kwargs["max_top_gene_fraction"] = max_top_gene_fraction
    if max_top10_gene_fraction is not None:
        kwargs["max_top10_gene_fraction"] = max_top10_gene_fraction
    return expression.sample_expression_qc(code, **kwargs)


def sample_qc_manifest(
    cancer_type=None,
    *,
    sample_qc: str = "all",
    auto_fetch: bool = True,
    on_missing: str = "empty",
) -> pd.DataFrame:
    """Read the generated source-matrix sample-QC manifest from the data bundle.

    This is a semantic alias for
    :func:`oncoref.expression.source_matrix_sample_qc_manifest`. It represents the
    QC rows recorded during expression-artifact generation, not a live recompute
    from a raw per-sample matrix. Use :func:`sample_qc` for live per-cohort QC.
    """
    from . import expression

    return expression.source_matrix_sample_qc_manifest(
        cancer_type,
        sample_qc=sample_qc,
        auto_fetch=auto_fetch,
        on_missing=on_missing,
    )
