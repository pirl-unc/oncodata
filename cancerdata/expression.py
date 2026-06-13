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

"""Read-only accessors over the per-cohort expression bundle.

These read pre-computed parquet artifacts from the downloadable bundle — the
heavy normalization that produced them runs offline in the build scripts, so the
read path here is light (resolve a cancer code, read a parquet, maybe ``expm1``).

  - ``cohort_gene_percentiles`` — per-gene percentile vector for a cohort
    (within-cohort, across samples): the "how high does this gene get in the
    cohort tail" signal.
  - ``representative_cohort_samples`` — bounded medoid per-sample vectors.

The richer analysis accessors (``cancer_reference_expression`` etc.) live in
pirlygenes; this module owns only the data-layer read surface that downstream
target-selection consumes.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_bundle, source_matrices
from ._build import WITHIN_SAMPLE_THRESHOLDS as _WITHIN_SAMPLE_THRESHOLD_COLS
from .cancer_types import cohort_aggregates, resolve_cancer_type
from .expression_engine import ID_COLUMNS
from .load_dataset import _BUNDLED_DATA_DIR, _register_derived_cache
from .normalization import clean_tpm

_REPRESENTATIVES_DIR = "cancer-reference-expression-representatives"
_PERCENTILES_DIR = "cancer-reference-expression-percentiles"
_PERCENTILES_PROTEOFORM_DIR = "cancer-reference-expression-percentiles-proteoform"
_WITHIN_SAMPLE_DIR = "cancer-reference-expression-within-sample-top5"
_WITHIN_SAMPLE_PROTEOFORM_DIR = "cancer-reference-expression-within-sample-top5-proteoform"


def _bundle_subdir(name: str, *, auto_fetch: bool = True) -> Path:
    """Locate a bundle shard directory: an in-repo checkout (``cancerdata/data/…``)
    wins, else the downloaded bundle cache; the bundle is fetched if absent.

    ``auto_fetch=False`` skips the (potentially 340 MB) download — used for
    artifacts not yet shipped in any released bundle, where a fetch couldn't
    provide them anyway; the returned path simply won't exist."""
    in_repo = Path(_BUNDLED_DATA_DIR) / name
    if in_repo.exists():
        return in_repo
    cached = data_bundle.find(name)
    if cached is not None:
        return cached
    if auto_fetch:
        data_bundle.ensure_local()
    return data_bundle.cache_dir() / name


def _available_shard_codes(root: Path) -> list[str]:
    """Sorted cohort codes that ship a parquet shard under ``root``."""
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.parquet"))


def _resolve_cancer_types(
    cancer_types: str | Iterable[str] | None,
    *,
    expand_aggregates: bool = False,
) -> list[str] | None:
    """Resolve a code / alias / iterable to canonical codes. With
    ``expand_aggregates``, a computed-aggregate code (``SARC`` and the
    ``SARC_RMS`` / ``SARC_LPS`` rollups) expands to its member subtypes."""
    if cancer_types is None:
        return None
    if isinstance(cancer_types, str):
        requested = [cancer_types]
    else:
        requested = list(cancer_types)
    if not expand_aggregates:
        return [resolve_cancer_type(code) for code in requested]

    aggregates = cohort_aggregates()
    out: list[str] = []
    for code in requested:
        members = aggregates.get(str(code))
        if members is None:
            resolved = resolve_cancer_type(code)
            members = aggregates.get(resolved)
            if members is None:
                out.append(resolved)
                continue
        out.extend(members)
    return list(dict.fromkeys(out))


def _representatives_root() -> Path:
    return _bundle_subdir(_REPRESENTATIVES_DIR)


def _percentiles_root(*, proteoform: bool = False) -> Path:
    if proteoform:
        # The proteoform-summed variant isn't in a released bundle yet, so never
        # trigger a fetch for it.
        return _bundle_subdir(_PERCENTILES_PROTEOFORM_DIR, auto_fetch=False)
    return _bundle_subdir(_PERCENTILES_DIR)


def available_representative_cohorts() -> list[str]:
    """Registry codes that ship a representative-samples shard (sorted)."""
    return _available_shard_codes(_representatives_root())


def available_percentile_cohorts(*, proteoform: bool = False) -> list[str]:
    """Cohort codes that ship a percentile-vector shard (sorted). With
    ``proteoform=True``, the proteoform-summed variant (one vector per proteoform
    key, identical-protein members collapsed before ranking)."""
    return _available_shard_codes(_percentiles_root(proteoform=proteoform))


_PER_SAMPLE_NORMALIZE = ("tpm_raw", "tpm_clean", "tpm_clean_log1p")


def per_sample_expression(
    cancer_type, *, normalize: str = "tpm_clean", auto_fetch: bool = True
) -> pd.DataFrame:
    """Full per-sample expression matrix (genes x **every** sample) for a cohort.

    The packaged references are summaries — per-gene percentile vectors, bounded
    medoid :func:`representative_cohort_samples`, within-sample top fractions. This
    returns the raw material behind them: one column per individual sample, so a
    consumer can ask per-patient questions a summary can't answer ("in what
    fraction of patients is this gene expressed", greedy antigen co-occurrence
    coverage, …). It fetches the cohort's per-sample matrix via
    :mod:`cancerdata.source_matrices` (a per-cohort release asset, tens of MB) and
    normalizes it.

    ``normalize``:
      - ``"tpm_clean"`` (default) — two-compartment clean TPM (the comparable
        biological view the summaries are built on);
      - ``"tpm_clean_log1p"`` — clean TPM, ``log1p``-transformed;
      - ``"tpm_raw"`` — the matrix as shipped (raw TPM), no normalization.

    ``auto_fetch=False`` raises instead of downloading if the matrix isn't cached.
    Returns ``Ensembl_Gene_ID``, ``Symbol`` and one column per sample.
    """
    if normalize not in _PER_SAMPLE_NORMALIZE:
        raise ValueError(f"normalize must be one of {_PER_SAMPLE_NORMALIZE}")
    code = resolve_cancer_type(cancer_type, strict=False) or cancer_type
    if auto_fetch:
        path = source_matrices.ensure(code)
    else:
        path = source_matrices.local_path(code)
        if not path.exists():
            raise FileNotFoundError(
                f"per-sample matrix for {code!r} not cached at {path}. "
                f"Run source_matrices.fetch({code!r}) to download it."
            )
    # The read + clean_tpm of a tens-of-MB matrix is the dominant cost on every
    # coverage / 9-mer call; memoize it (keyed on the matrix path + its mtime +
    # normalize) and hand callers a fresh copy so the shared frame can't be mutated.
    # The mtime in the key self-invalidates the cache if the matrix is re-fetched.
    return _load_per_sample_matrix(str(path), os.path.getmtime(path), normalize).copy()


@lru_cache(maxsize=2)
def _load_per_sample_matrix(path: str, mtime: float, normalize: str) -> pd.DataFrame:
    """Read + normalize one cohort's per-sample matrix (the cached canonical frame).
    ``path``/``mtime`` identify the on-disk parquet (mtime keys cache invalidation);
    the matrix must already be present."""
    raw = pd.read_parquet(path)
    base = ["Ensembl_Gene_ID", "Symbol"]
    samples = [c for c in raw.columns if c not in base]
    if normalize == "tpm_raw":
        return raw
    clean = clean_tpm(raw[samples], gene_table=raw[base])
    out = pd.concat([raw[base].reset_index(drop=True), clean.reset_index(drop=True)], axis=1)
    if normalize == "tpm_clean_log1p":
        out[samples] = np.log1p(out[samples].to_numpy(dtype=float))
    return out


_register_derived_cache(_load_per_sample_matrix.cache_clear)


def cohort_mean_expression(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    statistic: str = "mean",
    auto_fetch: bool = True,
    proteoform: bool = False,
    scope: str = "cta",
) -> pd.DataFrame:
    """Per-gene **across-patient summary** of a cohort's expression (one value per
    gene, collapsed over all patients).

    The continuous cohort-level expression that downstream mechanism/correlation
    analyses need (e.g. cohort-mean TGF-β-signature expression vs aPD1 ORR) — which
    the percentile vectors and n=5 medoids don't give directly. Reduces
    :func:`per_sample_expression` over the sample axis with ``statistic`` (``"mean"``
    / ``"median"``). ``normalize`` is passed through (clean TPM by default; use
    ``"tpm_clean_log1p"`` to average in log space). A **gene-level** frame.

    With ``proteoform=True``, identical-protein paralogs are summed per sample
    (:func:`cancerdata.proteoforms.collapse_to_proteoforms`) **before** the
    across-patient reduction, so the summary is over the reduced proteoform key space
    (rows carry ``proteoform_key`` — a **proteoform-level** frame, see
    :func:`cancerdata.proteoforms.expression_level`). ``scope`` selects the gene
    universe to collapse: ``"cta"`` (focused) or ``"genome"`` (every protein-coding
    gene). Returns ``Ensembl_Gene_ID``, ``Symbol`` (plus the proteoform identity
    columns when collapsed) and one ``expression`` column."""
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")
    df = per_sample_expression(cancer_type, normalize=normalize, auto_fetch=auto_fetch)
    if proteoform:
        from .proteoforms import collapse_to_proteoforms

        df = collapse_to_proteoforms(df, scope=scope)
    id_cols = [c for c in ID_COLUMNS if c in df.columns]
    samples = [c for c in df.columns if c not in id_cols]
    reducer = df[samples].mean(axis=1) if statistic == "mean" else df[samples].median(axis=1)
    out = df[id_cols].copy()
    out["expression"] = reducer.to_numpy()
    return out


def representative_cohort_samples(
    cancer_types: str | Iterable[str] | None = None,
    *,
    k: int | None = None,
    normalize: str = "tpm_clean",
    format: str = "wide",
    include_provenance: bool = False,
) -> pd.DataFrame:
    """Representative real per-sample expression vectors per cohort.

    The packaged cohort references are per-cohort aggregates; this returns a
    bounded set of real joint per-sample vectors per cohort — medoids spanning
    the within-cohort variation — in the same ``clean TPM`` basis.

    ``cancer_types`` accepts a code, alias, or iterable; a computed-aggregate
    code expands to its member subtypes; ``None`` returns every cohort that ships
    representatives. ``k`` keeps at most the first ``k`` reps per cohort.
    ``format`` is ``"wide"`` (genes × reps) or ``"long"``.
    """
    if normalize not in ("tpm_clean", "tpm_clean_log1p"):
        raise ValueError(
            "representative_cohort_samples normalize must be 'tpm_clean' or "
            "'tpm_clean_log1p' (the artifact ships only in clean TPM)"
        )
    if format not in ("wide", "long"):
        raise ValueError("format must be 'wide' or 'long'")
    if include_provenance and format != "long":
        # Provenance is per-representative (one row each); it only attaches to the
        # long form. Fail loudly rather than silently dropping the request.
        raise ValueError("include_provenance=True requires format='long'")

    root = _representatives_root()
    available = set(available_representative_cohorts())
    if cancer_types is None:
        codes = sorted(available)
    else:
        requested = _resolve_cancer_types(cancer_types, expand_aggregates=True)
        codes = [c for c in dict.fromkeys(requested) if c in available]

    base = ["Ensembl_Gene_ID", "Symbol"]
    wide = None
    long_parts = []
    for code in codes:
        shard = pd.read_parquet(root / f"{code}.parquet")
        rep_cols = [c for c in shard.columns if c not in base]
        if k is not None:
            rep_cols = rep_cols[:k]
        if normalize == "tpm_clean_log1p":
            shard[rep_cols] = np.log1p(shard[rep_cols].to_numpy(dtype=float))
        if format == "wide":
            part = shard[base + rep_cols]
            wide = part if wide is None else wide.merge(part, on=base, how="outer")
        else:
            melted = shard[base + rep_cols].melt(
                id_vars=base, var_name="representative_id", value_name="expression"
            )
            melted.insert(2, "cancer_code", code)
            long_parts.append(melted)

    if format == "wide":
        if wide is None:
            return pd.DataFrame(columns=base)
        return wide

    if not long_parts:
        cols = [*base, "cancer_code", "representative_id", "expression"]
        return pd.DataFrame(columns=cols)
    long = pd.concat(long_parts, ignore_index=True)
    if include_provenance:
        prov_path = root / "_provenance.csv"
        if prov_path.exists():
            prov = pd.read_csv(prov_path)
            keep = ["representative_id", "source_cohort", "source_project", "n_cohort_samples"]
            long = long.merge(
                prov[[c for c in keep if c in prov.columns]],
                on="representative_id",
                how="left",
            )
    return long


def cohort_gene_percentiles(
    cancer_type, *, as_tpm: bool = True, proteoform: bool = False
) -> pd.DataFrame:
    """Tail-weighted per-gene percentile vector for one cohort.

    Returns one row per gene (``Ensembl_Gene_ID`` + ``Symbol``) with 26
    breakpoint columns — ``p0, p1, p5, p10 … p90, p95, p96, p97, p98, p99,
    p100`` — dense in the actionable upper tail. Lets a consumer place a sample's
    gene as a **percentile rank within the cohort** instead of an absolute TPM.

    Computed on the biological clean TPM view (technical genes dropped).
    Stored compactly as ``log1p`` + float16; ``as_tpm=True`` (default)
    ``expm1``-restores clean-TPM values, ``as_tpm=False`` returns the stored
    log1p values. Raises if the cohort has no per-sample data (summary-only
    cohorts have no vector — see :func:`available_percentile_cohorts`).

    With ``proteoform=True``, reads the proteoform-summed variant: identical-protein
    members were summed per sample **before** the percentiles were computed, so the
    vector is one row per proteoform key (``proteoform_key``/``Symbol`` carry the
    collapsed identity); build it with ``generate_cohort_percentiles.py --proteoform``.
    """
    code = resolve_cancer_type(cancer_type)
    shard = _percentiles_root(proteoform=proteoform) / f"{code}.parquet"
    if not shard.exists():
        variant = "proteoform-summed " if proteoform else ""
        raise ValueError(
            f"no {variant}percentile vector for {code!r} — only cohorts with per-sample "
            f"data ship one; see available_percentile_cohorts(proteoform={proteoform})."
        )
    df = pd.read_parquet(shard)
    bp_cols = [c for c in df.columns if c not in ID_COLUMNS]
    df[bp_cols] = df[bp_cols].astype("float32")
    if as_tpm:
        df[bp_cols] = np.expm1(df[bp_cols])
    return df


# ---------- within-sample percentile prevalence (signal a) ----------

#: within-sample percentile-rank threshold -> output column: ``_WITHIN_SAMPLE_THRESHOLD_COLS``
#: (imported above from :data:`cancerdata._build.WITHIN_SAMPLE_THRESHOLDS`) is the single
#: source of truth shared with the generator, so the read side and write side can't drift.


def _within_sample_root(*, proteoform: bool = False) -> Path:
    # Not (yet) part of a released bundle, so never trigger a 340 MB fetch.
    name = _WITHIN_SAMPLE_PROTEOFORM_DIR if proteoform else _WITHIN_SAMPLE_DIR
    return _bundle_subdir(name, auto_fetch=False)


def available_within_sample_cohorts(*, proteoform: bool = False) -> list[str]:
    """Cohort codes that ship a within-sample top-fraction shard (sorted).

    With ``proteoform=True``, the proteoform-summed variant (identical-protein
    members collapsed before ranking — see :func:`within_sample_top_fraction`)."""
    return _available_shard_codes(_within_sample_root(proteoform=proteoform))


def within_sample_top_fraction(
    cancer_type, *, threshold: float = 0.95, proteoform: bool = False
) -> pd.DataFrame:
    """Per-gene fraction of a cohort's samples in which the gene is highly
    expressed *within that sample* — the "top ~5% expressed gene in this tumor"
    prevalence (signal a, the producer side of the within-sample signal).

    Returns one row per gene (``Ensembl_Gene_ID`` + ``Symbol``) with the
    ``frac_samples_top{1,5,10}pct`` column for the requested ``threshold``
    (0.99 / 0.95 / 0.90) plus ``n_samples``. Raises if the cohort has no
    within-sample shard — that table is built offline from per-sample matrices
    (see ``scripts/generate_within_sample_top5.py``) and shipped in the bundle.

    With ``proteoform=True``, reads the proteoform-summed variant: identical-
    protein paralogs (CTAG1A+CTAG1B, the CT47A family, …) are summed per sample
    *before* the within-sample ranking, so a duplicated antigen is ranked as one
    proteoform rather than several individually-diluted genes. Rows for those
    members are replaced by a single proteoform-labelled row; build it with
    ``generate_within_sample_top5.py --proteoform``. Note that collapsing members
    shrinks the gene axis the percentiles are computed over, so an ungrouped
    gene's fraction can shift slightly between the two variants — they are not
    row-for-row comparable for genes outside any proteoform group.
    """
    col = _WITHIN_SAMPLE_THRESHOLD_COLS.get(threshold)
    if col is None:
        raise ValueError(f"threshold must be one of {sorted(_WITHIN_SAMPLE_THRESHOLD_COLS)}")
    code = resolve_cancer_type(cancer_type)
    shard = _within_sample_root(proteoform=proteoform) / f"{code}.parquet"
    if not shard.exists():
        variant = "proteoform-summed " if proteoform else ""
        raise ValueError(
            f"no {variant}within-sample top-fraction vector for {code!r} — only "
            f"cohorts with per-sample data ship one; see "
            f"available_within_sample_cohorts(proteoform={proteoform})."
        )
    df = pd.read_parquet(shard)
    keep = ["Ensembl_Gene_ID", "Symbol", col]
    if "n_samples" in df.columns:
        keep.append("n_samples")
    return df[keep]


# ---------- proteoform-level summation (identical-protein paralogs) ----------


def proteoform_representative_samples(
    cancer_types: str | Iterable[str] | None = None,
    *,
    k: int | None = None,
) -> pd.DataFrame:
    """Representative per-sample vectors with identical-protein genes summed to
    proteoform level.

    Same per-sample medoid vectors as :func:`representative_cohort_samples`
    (``format="wide"``), but the member genes of each proteoform group
    (CTAG1A+CTAG1B → ``CTAG1A/CTAG1B``, SSX4+SSX4B → ``SSX4/SSX4B``, the 12-member
    CT47A family, …) are *summed* per sample — the multi-mapping-correct unit
    when reads can't be uniquely assigned between identical-protein loci. Genes in
    no group pass through unchanged.

    Always operates on linear ``clean TPM`` (summing log1p values would be
    wrong); ``log1p`` afterward if you need it. This is the runtime,
    every-cohort proteoform view over the shipped medoid samples; the same
    :func:`cancerdata._build.sum_proteoform_tpm` core can run inside the offline
    percentile/within-sample generators to ship proteoform-summed artifacts.
    """
    from .proteoforms import collapse_to_proteoforms

    wide = representative_cohort_samples(cancer_types, k=k, normalize="tpm_clean", format="wide")
    sample_cols = [c for c in wide.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    if wide.empty or not sample_cols:
        return wide
    return collapse_to_proteoforms(wide, sample_cols=sample_cols)
