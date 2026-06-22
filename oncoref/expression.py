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

"""Read accessors over the per-cohort expression data.

This module is a small, regular surface over one underlying object — a cohort's
**per-sample expression matrix** (genes × samples) — plus the pre-computed summary
artifacts derived from it. The design is two orthogonal axes; everything here is a
point on that grid, which is why the accessors look repetitive by construction
rather than by accident.

**Axis 1 — expression level** (``proteoforms.py``). Every value is available at
*gene* level (one row per Ensembl gene) or *proteoform* level (identical-protein
paralogs summed per sample — CTAG1A+CTAG1B → NY-ESO-1 — keyed by ``proteoform_key``;
see :mod:`oncoref.proteoforms`). The base functions take ``proteoform=``/``scope=``
flags; each also has an explicit ``gene_*`` / ``proteoform_*`` wrapper so the level is
legible at the call site and discoverable by name. The wrappers are deliberately
thin — the collapse logic lives in exactly one place (the base function).

**Axis 2 — the dataset / reduction**:
  - ``per_sample_expression`` — the raw matrix itself (one column per sample).
  - ``cohort_mean_expression`` — one across-patient statistic (mean/median).
  - ``cohort_stats`` / ``pooled_cohort_stats`` — the full per-gene statistic suite,
    for one cohort or a heterogeneity-safe pool of several.
  - ``cohort_gene_percentiles`` — the within-cohort percentile vector.
  - ``within_sample_top_fraction`` — within-sample top-expression prevalence.
  - ``representative_cohort_samples`` — bounded medoid per-sample vectors.
  - ``pan_cancer_expression`` — the wide tumor+normal HPA/TCGA reference (a distinct
    data product, not the per-sample-matrix family).

**Normalize spaces** (``per_sample_expression``'s ``normalize=``): raw TPM, clean
two-compartment TPM (the default biological view), its ``log1p``, and the
housekeeping ratio (``tpm_clean_hk``). Every summary computed live inherits the space.

**Where the numbers come from.** The light artifacts (percentiles, within-sample,
representatives) are pre-computed offline by the ``scripts/generate_*`` build cores
and shipped as per-cohort parquet *shards*; the read path just resolves a code and
reads a parquet. When a shard isn't shipped (every proteoform variant, today), the
summary is **recomputed on the fly** from the per-sample matrix via the *same* build
core, so shipped and on-the-fly values agree. Which artifact ships which variant, and
whether a missing shard may be fetched, is recorded once in the :class:`ShardDataset`
registry (:data:`SHARD_DATASETS`) rather than restated per accessor.

Higher-level consumer accessors such as ``cancer_reference_expression`` are thin
views over these same shipped artifacts rather than separate data products.
"""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_bundle, source_matrices
from .cancer_types import cohort_aggregates, resolve_cancer_type
from .expression_builders import WITHIN_SAMPLE_THRESHOLDS as _WITHIN_SAMPLE_THRESHOLD_COLS
from .expression_engine import id_columns, sample_columns
from .gene_ids import ensembl_id_alias_symbols, resolve_ensembl_id, unversioned
from .load_dataset import _BUNDLED_DATA_DIR, _register_derived_cache, get_data
from .normalization import clean_tpm, percentile_rank, tpm_to_housekeeping_normalized


@dataclass(frozen=True)
class ShardDataset:
    """A per-cohort summary artifact: one parquet shard per cohort code, available at
    gene level and (optionally) proteoform level. The single declarative record of
    everything the read path needs for one artifact, so the accessors stay generic over
    it rather than special-casing each.

    Fields:
      - ``noun`` — human label used in error messages.
      - ``gene_dir`` — bundle subdirectory of the gene-level shards.
      - ``gene_fetches`` — whether the gene-level shards ship in a released bundle (so a
        missing one may be auto-fetched) vs. must be recomputed on the fly.
      - ``proteoform_stem`` — the *stem* of the proteoform shard directory; ``None`` if
        the artifact has no proteoform variant. The on-disk directory is **scope-suffixed**
        (``f"{proteoform_stem}-{scope}"``) because identical-protein members group
        differently under ``"cta"`` vs ``"genome"``, so each scope is its own shard set.
      - ``proteoform_fetches`` — whether proteoform shards ship (none do yet).
      - ``build_attr`` — name of the :mod:`oncoref.expression_builders` core that regenerates a
        missing shard from the per-sample matrix (the same core that produced the shipped
        shards, so on-the-fly and shipped values agree).

    The registry of the concrete artifacts is :data:`SHARD_DATASETS`.
    """

    noun: str
    gene_dir: str
    gene_fetches: bool
    proteoform_stem: str | None = None
    proteoform_fetches: bool = False
    build_attr: str | None = None

    def subdir(self, *, proteoform: bool, scope: str = "cta") -> str:
        """Bundle subdirectory holding this artifact's shards at the requested level.
        Gene-level is scope-independent; the proteoform variant is **scope-specific**
        (``f"{proteoform_stem}-{scope}"``)."""
        if not proteoform:
            return self.gene_dir
        if self.proteoform_stem is None:
            raise ValueError(f"{self.noun} has no proteoform variant")
        return f"{self.proteoform_stem}-{scope}"

    def fetches(self, *, proteoform: bool) -> bool:
        """Whether the requested level's shards ship in a released bundle (so a missing
        shard may be auto-fetched rather than only recomputed)."""
        return self.proteoform_fetches if proteoform else self.gene_fetches


#: The expression summary artifacts, keyed by short name. Proteoform shards aren't shipped
#: in any bundle yet (``proteoform_fetches=False``), so the proteoform variant is recomputed
#: on the fly from the per-sample matrix (see ``_read_shard_or_recompute``) until they ship;
#: when they do, each scope is a distinct ``{stem}-{scope}`` directory (see
#: :meth:`ShardDataset.subdir`).
SHARD_DATASETS: dict[str, ShardDataset] = {
    "representatives": ShardDataset(
        noun="representative-samples shard",
        gene_dir="cancer-reference-expression-representatives",
        gene_fetches=True,
    ),
    "percentiles": ShardDataset(
        noun="percentile vector",
        gene_dir="cancer-reference-expression-percentiles",
        gene_fetches=True,
        proteoform_stem="cancer-reference-expression-percentiles-proteoform",
        build_attr="cohort_percentile_vectors",
    ),
    "within_sample": ShardDataset(
        noun="within-sample top-fraction vector",
        gene_dir="cancer-reference-expression-within-sample-top5",
        gene_fetches=False,  # not part of a released bundle yet -> never trigger a fetch
        proteoform_stem="cancer-reference-expression-within-sample-top5-proteoform",
        build_attr="within_sample_top_fractions",
    ),
}
_REPRESENTATIVES = SHARD_DATASETS["representatives"]
_PERCENTILES = SHARD_DATASETS["percentiles"]
_WITHIN_SAMPLE = SHARD_DATASETS["within_sample"]


# How many cleaned per-sample matrices to keep in the in-process LRU. Each frame is
# a full gene x sample matrix (~100MB+), so the default is intentionally small. A
# workflow that pools the same N>2 cohorts repeatedly (e.g. a gene then a proteoform
# pool over one cohort set) can raise CANCERDATA_PER_SAMPLE_CACHE to keep them all
# warm and skip the re-read, trading memory for latency.
def _per_sample_cache_size(default: int = 2) -> int:
    """Parse the cache-size env knob, falling back to ``default`` on any malformed
    value (empty string, non-integer) — a tuning knob must never break ``import``."""
    try:
        return max(1, int(os.environ.get("CANCERDATA_PER_SAMPLE_CACHE", str(default))))
    except (TypeError, ValueError):
        return default


_PER_SAMPLE_CACHE_SIZE = _per_sample_cache_size()


def _bundle_subdir(name: str, *, auto_fetch: bool = True) -> Path:
    """Locate a bundle shard directory: an in-repo checkout (``oncoref/data/…``)
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


def _shard_dir(dataset: ShardDataset, *, proteoform: bool = False, scope: str = "cta") -> Path:
    """The bundle directory holding ``dataset``'s per-cohort shards at the requested
    level and ``scope`` (the proteoform variant is scope-specific — see
    :meth:`ShardDataset.subdir`). The per-variant auto-fetch policy lives on the dataset
    record, so it's decided in one place rather than re-stated at each call site."""
    return _bundle_subdir(
        dataset.subdir(proteoform=proteoform, scope=scope),
        auto_fetch=dataset.fetches(proteoform=proteoform),
    )


def _available_cohorts(
    dataset: ShardDataset, *, proteoform: bool = False, scope: str = "cta"
) -> list[str]:
    """Sorted cohort codes with a shipped shard for ``dataset`` (gene, or the
    scope-specific proteoform variant)."""
    return _available_shard_codes(_shard_dir(dataset, proteoform=proteoform, scope=scope))


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


def _gene_filter_mask(df: pd.DataFrame, genes: str | Iterable[str] | None) -> pd.Series:
    if genes is None:
        return pd.Series(True, index=df.index)
    wanted = {genes} if isinstance(genes, str) else set(genes)
    wanted = {str(g) for g in wanted}
    wanted_unversioned = {unversioned(g) for g in wanted}
    ids = df["Ensembl_Gene_ID"].astype(str)
    return (
        ids.isin(wanted)
        | ids.map(unversioned).isin(wanted_unversioned)
        | df["Symbol"].astype(str).isin(wanted)
    )


def available_representative_cohorts() -> list[str]:
    """Registry codes that ship a representative-samples shard (sorted)."""
    return _available_cohorts(_REPRESENTATIVES)


def available_percentile_cohorts(*, proteoform: bool = False, scope: str = "cta") -> list[str]:
    """Cohort codes that ship a percentile-vector shard (sorted). With
    ``proteoform=True``, the proteoform-summed variant (one vector per proteoform
    key, identical-protein members collapsed before ranking, in ``scope``)."""
    return _available_cohorts(_PERCENTILES, proteoform=proteoform, scope=scope)


_PER_SAMPLE_NORMALIZE = ("tpm_raw", "tpm_clean", "tpm_clean_log1p", "tpm_clean_hk")


def per_sample_expression(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    proteoform: bool = False,
    scope: str = "cta",
) -> pd.DataFrame:
    """Full per-sample expression matrix (genes x **every** sample) for a cohort —
    the raw **TPM values** at gene level (default) or proteoform level.

    The packaged references are summaries — per-gene percentile vectors, bounded
    medoid :func:`representative_cohort_samples`, within-sample top fractions. This
    returns the raw material behind them: one column per individual sample, so a
    consumer can ask per-patient questions a summary can't answer ("in what
    fraction of patients is this gene expressed", greedy antigen co-occurrence
    coverage, …). It fetches the cohort's per-sample matrix via
    :mod:`oncoref.source_matrices` (a per-cohort release asset, tens of MB) and
    normalizes it.

    ``normalize``:
      - ``"tpm_clean"`` (default) — two-compartment clean TPM (the comparable
        biological view the summaries are built on);
      - ``"tpm_clean_log1p"`` — clean TPM, ``log1p``-transformed;
      - ``"tpm_clean_hk"`` — clean TPM divided per sample by the housekeeping-panel
        geometric mean (unit-free ratio-to-baseline, robust to library-depth drift);
      - ``"tpm_raw"`` — the matrix as shipped (raw TPM), no normalization.

    With ``proteoform=True``, identical-protein paralogs are **summed per sample** to
    proteoform level (:func:`oncoref.proteoforms.collapse_to_proteoforms`, ``scope``
    = ``"cta"``/``"genome"``) — a **proteoform-level** frame carrying ``proteoform_key``
    (see :func:`oncoref.proteoforms.expression_level`). The sum is always taken in
    **linear** TPM and the ``log1p`` transform (if any) applied *after*, so the
    proteoform value is ``log1p(Σ member TPM)``, not the meaningless ``Σ log1p``.
    (``scope`` is ignored when ``proteoform=False``.)

    ``auto_fetch=False`` raises instead of downloading if the matrix isn't cached.
    Returns ``Ensembl_Gene_ID``, ``Symbol`` and one column per sample (plus the
    proteoform identity columns when collapsed).
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
    mtime = os.path.getmtime(path)
    if not proteoform:
        return _load_per_sample_matrix(str(path), mtime, normalize).copy()
    # Proteoform level: sum members in LINEAR TPM, then apply the requested transform.
    from .proteoforms import collapse_to_proteoforms

    linear = "tpm_raw" if normalize == "tpm_raw" else "tpm_clean"
    out = collapse_to_proteoforms(_load_per_sample_matrix(str(path), mtime, linear), scope=scope)
    samples = sample_columns(out)
    if normalize == "tpm_clean_log1p":
        out[samples] = np.log1p(out[samples].to_numpy(dtype=float))
    elif normalize == "tpm_clean_hk":
        out = _housekeeping_normalize(out, samples)
    return out


@lru_cache(maxsize=_PER_SAMPLE_CACHE_SIZE)
def _load_per_sample_matrix(path: str, mtime: float, normalize: str) -> pd.DataFrame:
    """Read + normalize one cohort's per-sample matrix (the cached canonical frame).
    ``path``/``mtime`` identify the on-disk parquet (mtime keys cache invalidation);
    the matrix must already be present."""
    raw = pd.read_parquet(path)
    # Make the matrix dense in the CANONICAL gene-id space before anything else: sum
    # alt-haplotype/patch copies into their primary gene (in LINEAR raw TPM) and relabel
    # retired ids to their successor (resolve_ensembl_id). Doing it pre-clean_tpm means
    # the alt copy inherits the primary's compartment and column totals are unchanged, so
    # clean_tpm's renormalization of every other gene is untouched; every downstream
    # accessor (cohort_stats, coverage, pooled, percentile/within-sample recompute) then
    # shares one canonical key space. (pirlygenes#465 / oncoref#135 item 6.)
    raw = _canonicalize_gene_rows(raw, sample_cols=sample_columns(raw))
    base = id_columns(raw)
    samples = sample_columns(raw)
    if normalize == "tpm_raw":
        return raw
    clean = clean_tpm(raw[samples], gene_table=raw[base])
    out = pd.concat([raw[base].reset_index(drop=True), clean.reset_index(drop=True)], axis=1)
    if normalize == "tpm_clean_log1p":
        out[samples] = np.log1p(out[samples].to_numpy(dtype=float))
    elif normalize == "tpm_clean_hk":
        out = _housekeeping_normalize(out, samples)
    return out


def _housekeeping_normalize(df: pd.DataFrame, sample_cols) -> pd.DataFrame:
    """Divide each sample column by its housekeeping-panel geometric mean (a per-sample
    rescale to a unit-free ratio-to-baseline scale). Commutes with the proteoform sum
    (the denominator is per-column), so it can be applied before or after collapse."""
    from .normalization import tpm_to_housekeeping_normalized

    out, _ = tpm_to_housekeeping_normalized(df, value_cols=list(sample_cols))
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
    (:func:`oncoref.proteoforms.collapse_to_proteoforms`) **before** the
    across-patient reduction, so the summary is over the reduced proteoform key space
    (rows carry ``proteoform_key`` — a **proteoform-level** frame, see
    :func:`oncoref.proteoforms.expression_level`). ``scope`` selects the gene
    universe to collapse: ``"cta"`` (focused) or ``"genome"`` (every protein-coding
    gene). Returns ``Ensembl_Gene_ID``, ``Symbol`` (plus the proteoform identity
    columns when collapsed) and one ``expression`` column."""
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")
    df = per_sample_expression(
        cancer_type,
        normalize=normalize,
        auto_fetch=auto_fetch,
        proteoform=proteoform,
        scope=scope,
    )
    id_cols = id_columns(df)
    samples = sample_columns(df)
    reducer = df[samples].mean(axis=1) if statistic == "mean" else df[samples].median(axis=1)
    out = df[id_cols].copy()
    out["expression"] = reducer.to_numpy()
    return out


def cancer_reference_expression(
    cancer_types: str | Iterable[str] | None = None,
    genes: str | Iterable[str] | None = None,
    normalize: str | Iterable[str] = "tpm_clean",
    *,
    format: str = "long",
    include_provenance: bool = True,
    auto_fetch: bool = False,
) -> pd.DataFrame:
    """Observed tumor expression references as cohort-level clean TPM summaries.

    The default long form mirrors the downstream reference contract:
    ``Ensembl_Gene_ID``, ``Symbol``, ``cancer_code``, ``source_cohort``,
    ``normalization``, ``expression`` (median clean TPM), ``q1`` and ``q3``.
    Wide form returns one row per gene with ``<CODE>_TPM_clean`` columns.
    """
    modes = _reference_normalize_modes(normalize)
    if format not in ("long", "wide"):
        raise ValueError("format must be 'long' or 'wide'")
    if modes != {"tpm_clean"}:
        raise ValueError(
            "cancer_reference_expression currently supports normalize='tpm_clean' "
            "(alias 'clean_tpm')"
        )
    available = set(available_percentile_cohorts())
    if cancer_types is None:
        codes = sorted(available)
    else:
        requested = _resolve_cancer_types(cancer_types, expand_aggregates=True) or []
        codes = [code for code in dict.fromkeys(requested) if code in available]

    long_parts: list[pd.DataFrame] = []
    wide_parts: list[pd.DataFrame] = []
    for code in codes:
        pct = cohort_gene_percentiles(code, as_tpm=True, auto_fetch=auto_fetch)
        pct = pct[_gene_filter_mask(pct, genes)].reset_index(drop=True)
        if format == "long":
            part = pct[["Ensembl_Gene_ID", "Symbol", "p25", "p50", "p75"]].copy()
            part.insert(2, "cancer_code", code)
            if include_provenance:
                part.insert(3, "source_cohort", code)
            part["normalization"] = "tpm_clean"
            part = part.rename(columns={"p50": "expression", "p25": "q1", "p75": "q3"})
            cols = ["Ensembl_Gene_ID", "Symbol", "cancer_code"]
            if include_provenance:
                cols.append("source_cohort")
            cols += ["normalization", "expression", "q1", "q3"]
            long_parts.append(part[cols])
        else:
            part = pct[["Ensembl_Gene_ID", "Symbol", "p50"]].rename(
                columns={"p50": f"{code}_TPM_clean"}
            )
            wide_parts.append(part)

    if format == "long":
        cols = ["Ensembl_Gene_ID", "Symbol", "cancer_code"]
        if include_provenance:
            cols.append("source_cohort")
        cols += ["normalization", "expression", "q1", "q3"]
        if not long_parts:
            return pd.DataFrame(columns=cols)
        return pd.concat(long_parts, ignore_index=True)

    if not wide_parts:
        return pd.DataFrame(columns=["Ensembl_Gene_ID", "Symbol"])
    out = wide_parts[0]
    for part in wide_parts[1:]:
        out = out.merge(part, on=["Ensembl_Gene_ID", "Symbol"], how="outer")
    return out.sort_values("Ensembl_Gene_ID").reset_index(drop=True)


def _reference_normalize_modes(normalize: str | Iterable[str]) -> set[str]:
    if isinstance(normalize, str):
        modes = {normalize.lower()}
    else:
        modes = {str(mode).lower() for mode in normalize}
    return {"tpm_clean" if mode == "clean_tpm" else mode for mode in modes}


#: Per-gene cohort summary statistic -> output column. The percentiles are taken across
#: the cohort's samples (the same axis as :func:`cohort_gene_percentiles`).
_COHORT_STAT_PERCENTILES = {
    0: "min",
    1: "p1",
    5: "p5",
    10: "p10",
    15: "p15",
    20: "p20",
    25: "p25",
    50: "p50",
    75: "p75",
    80: "p80",
    85: "p85",
    90: "p90",
    95: "p95",
    99: "p99",
    100: "max",
}


def _write_cohort_stat_columns(out: pd.DataFrame, mat: np.ndarray) -> None:
    """Write the ``mean``/``std`` + percentile-ladder columns onto ``out`` from a
    ``(n_genes, n_samples)`` value matrix — **availability-aware** (``NaN`` cells are
    skipped, so each gene reduces only over the samples that measured it). ``std`` is
    ``NaN`` for a gene measured by fewer than two samples (a lone observation has no
    spread — the same ``n >= 2`` rule used for ``std_between``). Shared by
    :func:`cohort_stats` and :func:`pooled_cohort_stats` so the suite is defined once."""
    pcts = list(_COHORT_STAT_PERCENTILES)
    n_obs = np.sum(~np.isnan(mat), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN gene rows -> NaN
        out["mean"] = np.nanmean(mat, axis=1)
        out["std"] = np.where(n_obs >= 2, np.nanstd(mat, axis=1), np.nan)
        q = np.nanpercentile(mat, pcts, axis=1)  # (len(pcts), n_genes)
    for i, p in enumerate(pcts):
        out[_COHORT_STAT_PERCENTILES[p]] = q[i]


def cohort_stats(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    proteoform: bool = False,
    scope: str = "cta",
) -> pd.DataFrame:
    """Per-gene **summary statistics** across a cohort's samples, in one pass —
    ``mean``, ``std`` and a uniform percentile ladder ``min, p1, p5, p10, p15, p20,
    p25, p50, p75, p80, p85, p90, p95, p99, max`` (``p25``/``p50``/``p75`` are
    q1/median/q3).

    The richer companion to :func:`cohort_mean_expression` (a single statistic) and
    :func:`cohort_gene_percentiles` (the dense percentile vector): everything a consumer
    needs to describe a gene's distribution over the cohort in one frame. Computed on
    the per-sample matrix in the ``normalize`` space (linear clean TPM by default; pass
    ``"tpm_clean_log1p"`` for log-space stats).

    ``proteoform=True`` summarizes the reduced proteoform key space (members summed per
    sample first, ``scope`` ``"cta"``/``"genome"``) — a proteoform-level frame carrying
    ``proteoform_key`` (see :func:`oncoref.proteoforms.expression_level`). Returns the
    id columns plus one column per statistic."""
    df = per_sample_expression(
        cancer_type, normalize=normalize, auto_fetch=auto_fetch, proteoform=proteoform, scope=scope
    )
    id_cols = id_columns(df)
    samples = sample_columns(df)
    if not samples:
        raise ValueError(f"no per-sample columns to summarize for {cancer_type!r}")
    out = df[id_cols].copy()
    _write_cohort_stat_columns(out, df[samples].to_numpy(dtype=float))
    return out


def _canonicalize_gene_rows(df: pd.DataFrame, *, sample_cols=None) -> pd.DataFrame:
    """Re-key one cohort's rows onto the **canonical** gene id so a locus can't be
    fragmented across cohorts (the pirlygenes#465-class bug).

    Each ``Ensembl_Gene_ID`` is unversioned *and* migration-resolved through the shipped
    ensembl-id-aliases map (:func:`resolve_ensembl_id`), so an alt-haplotype / patch /
    retired id collapses onto its canonical primary-assembly id. When a cohort carries
    BOTH an alias id and its canonical sibling (a full-assembly quantification annotates
    the gene on the primary contig *and* its alt-haplotype copy), their per-sample TPMs
    are **summed** under the canonical id: RNA-seq reads multi-map between the copies, so
    each row individually under-counts the gene — the same rationale as proteoform
    summation, one level up (gene rather than protein). A cross-release retired id and its
    successor never co-occur in one sample, so summing them degenerates to the lone value
    (a relabel). All-``NaN`` cells stay ``NaN`` (``min_count=1``) so an unmeasured gene is
    never turned into a measured zero — the canonical symbol is taken from the
    primary-contig row (sorted first) deterministically. The fast path (no collisions)
    only rewrites the id column.

    ``sample_cols`` names the value columns to **sum** (the rest are kept via ``first``);
    summing must be done in **linear** TPM, so the caller transforms (log1p/hk) only
    afterwards. When ``None`` the value columns are inferred as the numeric dtypes — fine
    for the gene×sample frames here, but pass them explicitly at the per-sample chokepoint
    so a stray numeric id column can never be summed."""
    canon = df["Ensembl_Gene_ID"].astype(str).map(resolve_ensembl_id)
    if not canon.duplicated().any():
        return df.assign(Ensembl_Gene_ID=canon.to_numpy())
    orig = df["Ensembl_Gene_ID"].astype(str).map(unversioned)
    is_primary = orig.to_numpy() == canon.to_numpy()
    df = df.assign(Ensembl_Gene_ID=canon.to_numpy(), _primary=is_primary)
    df = df.sort_values("_primary", ascending=False, kind="stable").drop(columns="_primary")
    if sample_cols is None:
        sum_cols = df.select_dtypes("number").columns.tolist()
    else:
        sum_cols = [c for c in sample_cols if c in df.columns]
    keep_cols = [c for c in df.columns if c != "Ensembl_Gene_ID" and c not in sum_cols]
    grouped = df.groupby("Ensembl_Gene_ID", sort=False)
    parts = []
    if keep_cols:  # canonical symbol / id columns: keep the primary-contig row's value
        parts.append(grouped[keep_cols].first())
    if sum_cols:  # per-sample TPM: SUM alt-haplotype reads into the canonical gene
        parts.append(grouped[sum_cols].sum(min_count=1))
    out = pd.concat(parts, axis=1).reset_index()
    return out[list(df.columns)]


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

    root = _shard_dir(_REPRESENTATIVES)
    available = set(available_representative_cohorts())
    if cancer_types is None:
        codes = sorted(available)
    else:
        requested = _resolve_cancer_types(cancer_types, expand_aggregates=True)
        codes = [c for c in dict.fromkeys(requested) if c in available]

    base = ["Ensembl_Gene_ID", "Symbol"]
    # Combine cohorts on a CANONICAL gene id only — never the (Ensembl_Gene_ID, Symbol)
    # pair. Cohorts were quantified against different Ensembl releases, so the same locus
    # carries a different release-alias symbol per cohort; merging on the pair fragments one
    # gene into many mutually-disjoint sparse rows (the pirlygenes#465-class bug). We key
    # each cohort by the canonical gene id — unversioned AND migration-resolved through the
    # shipped ensembl-id-aliases map (resolve_ensembl_id), so an alt-haplotype/archived id
    # collapses onto its primary-contig id instead of standing as a separate row — then
    # resolve one canonical symbol per gene afterwards.
    wide_parts = []
    long_parts = []
    symbols: dict[str, Counter] = defaultdict(Counter)  # canonical id -> symbol counts
    for code in codes:
        shard = pd.read_parquet(root / f"{code}.parquet")
        rep_cols = sample_columns(shard)
        if k is not None:
            rep_cols = rep_cols[:k]
        # Key on the canonical gene id (unversion + migration-resolve), collapsing an
        # alt-haplotype row onto its primary-contig sibling within the cohort first. The
        # shard ships in LINEAR clean TPM, so the alt-copy sum happens in linear space;
        # the log1p transform (if any) is applied AFTER — never sum log-space values
        # (log1p(a)+log1p(b) != log1p(a+b)), mirroring the proteoform linear-then-log1p
        # contract in per_sample_expression.
        shard = _canonicalize_gene_rows(shard)
        if normalize == "tpm_clean_log1p":
            shard[rep_cols] = np.log1p(shard[rep_cols].to_numpy(dtype=float))
        gid = shard["Ensembl_Gene_ID"].astype(str)
        for g, s in zip(gid, shard["Symbol"].astype(str)):
            symbols[g][s] += 1
        mat = shard[rep_cols].set_axis(gid.to_numpy(), axis=0)
        if format == "wide":
            wide_parts.append(mat)
        else:
            melted = mat.reset_index(names="Ensembl_Gene_ID").melt(
                id_vars="Ensembl_Gene_ID", var_name="representative_id", value_name="expression"
            )
            melted.insert(1, "cancer_code", code)
            long_parts.append(melted)

    alias_symbols = ensembl_id_alias_symbols()

    def _canonical_symbol(gid: str) -> str:
        # The curated symbol for a migrated locus is authoritative and wins outright;
        # otherwise prefer a real name over the raw-ENSG backfill that release-unaware
        # cohorts carry (the most common alias across cohorts, deterministic), else the id.
        auth = alias_symbols.get(gid)
        if auth:
            return auth
        named = Counter({s: c for s, c in symbols.get(gid, {}).items() if s and s != gid})
        return named.most_common(1)[0][0] if named else gid

    if format == "wide":
        if not wide_parts:
            return pd.DataFrame(columns=base)
        combined = pd.concat(wide_parts, axis=1, join="outer").sort_index()
        if combined.index.has_duplicates:  # belt-and-suspenders: one row per gene id
            combined = combined.groupby(level=0).first()
        out = combined.reset_index(names="Ensembl_Gene_ID")
        out.insert(1, "Symbol", out["Ensembl_Gene_ID"].map(_canonical_symbol))
        return out

    if not long_parts:
        cols = [*base, "cancer_code", "representative_id", "expression"]
        return pd.DataFrame(columns=cols)
    long = pd.concat(long_parts, ignore_index=True)
    long.insert(1, "Symbol", long["Ensembl_Gene_ID"].map(_canonical_symbol))
    if include_provenance:
        prov_path = root / "_provenance.csv"
        if prov_path.exists():
            prov = pd.read_csv(prov_path)
            keep = [
                "representative_id",
                "source_cohort",
                "source_version",
                "source_project",
                "n_cohort_samples",
            ]
            long = long.merge(
                prov[[c for c in keep if c in prov.columns]],
                on="representative_id",
                how="left",
            )
    return long


def _biological_per_sample(
    code, *, proteoform: bool, auto_fetch: bool, scope: str = "cta"
) -> pd.DataFrame:
    """Clean-TPM per-sample matrix with technical/censored genes dropped — the
    biological view the summary artifacts are built on — collapsed to proteoform level
    (in ``scope``) when requested. The runtime input to the percentile / within-sample
    build cores, so a summary can be recomputed on the fly from the per-sample matrix
    (no shard). ``scope`` is ignored when ``proteoform`` is False."""
    from .gene_families import clean_tpm_censored_gene_ids

    clean = per_sample_expression(code, normalize="tpm_clean", auto_fetch=auto_fetch)
    censored = clean_tpm_censored_gene_ids()
    unversioned = clean["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    bio = clean[~unversioned.isin(censored)].reset_index(drop=True)
    if proteoform:
        from .proteoforms import collapse_to_proteoforms

        bio = collapse_to_proteoforms(bio, scope=scope, sample_cols=sample_columns(bio))
    return bio


def _read_shard_or_recompute(
    dataset: ShardDataset, code: str, *, proteoform: bool, auto_fetch: bool, scope: str = "cta"
) -> pd.DataFrame:
    """Read ``code``'s shard for ``dataset`` (the ``scope``-specific one at proteoform
    level); if no shard is present, recompute it on the fly from the per-sample matrix
    via the dataset's ``expression_builders`` core (the same core that produced the shipped shards —
    so the on-the-fly and shipped values agree).

    The single home of the shard-or-recompute fallback shared by the percentile and
    within-sample readers. Raises a clear :class:`ValueError` — not a bare
    ``FileNotFoundError`` — when neither the shard nor the per-sample matrix is available
    (the proteoform variant has no shipped shard yet, so it always takes this path)."""
    shard = _shard_dir(dataset, proteoform=proteoform, scope=scope) / f"{code}.parquet"
    if shard.exists():
        return pd.read_parquet(shard)
    try:
        bio = _biological_per_sample(
            code, proteoform=proteoform, auto_fetch=auto_fetch, scope=scope
        )
    except FileNotFoundError as e:
        variant = "proteoform-summed " if proteoform else ""
        raise ValueError(
            f"no {variant}{dataset.noun} for {code!r} and its per-sample matrix isn't "
            f"cached — fetch it (source_matrices.fetch / auto_fetch=True)."
        ) from e
    from importlib import import_module

    build_core = getattr(import_module("oncoref.expression_builders"), dataset.build_attr)
    return build_core(bio, sample_columns(bio))


def cohort_gene_percentiles(
    cancer_type,
    *,
    as_tpm: bool = True,
    proteoform: bool = False,
    scope: str = "cta",
    auto_fetch: bool = False,
) -> pd.DataFrame:
    """Tail-weighted per-gene percentile vector for one cohort.

    Returns one row per gene (``Ensembl_Gene_ID`` + ``Symbol``) with 26
    breakpoint columns — ``p0, p1, p5, p10 … p90, p95, p96, p97, p98, p99,
    p100`` — dense in the actionable upper tail. Lets a consumer place a sample's
    gene as a **percentile rank within the cohort** instead of an absolute TPM.

    Computed on the biological clean TPM view (technical genes dropped).
    Stored compactly as ``log1p`` + float16; ``as_tpm=True`` (default)
    ``expm1``-restores clean-TPM values, ``as_tpm=False`` returns the stored
    log1p values.

    With ``proteoform=True``, the vector is one row per proteoform key
    (``proteoform_key``/``Symbol`` carry the collapsed identity), identical-protein
    members summed **before** the percentiles are computed (``scope`` ``"cta"``/``"genome"``,
    ignored when ``proteoform`` is False).

    The shipped percentile **shard** can't be converted to the proteoform view (you
    can't sum already-computed percentiles), so when no shard is present the vector is
    **recomputed on the fly** from the per-sample matrix via the same build core — the
    live path for the proteoform variant until its shard ships. That needs the cohort's
    per-sample matrix cached (pass ``auto_fetch=True`` to download it); otherwise a
    clear error.
    """
    code = resolve_cancer_type(cancer_type)
    df = _read_shard_or_recompute(
        _PERCENTILES, code, proteoform=proteoform, auto_fetch=auto_fetch, scope=scope
    )
    bp_cols = sample_columns(df)
    df[bp_cols] = df[bp_cols].astype("float32")
    if as_tpm:
        df[bp_cols] = np.expm1(df[bp_cols])
    return df


# ---------- within-sample percentile prevalence (signal a) ----------

#: within-sample percentile-rank threshold -> output column: ``_WITHIN_SAMPLE_THRESHOLD_COLS``
#: (imported above from :data:`oncoref.expression_builders.WITHIN_SAMPLE_THRESHOLDS`) is the single
#: source of truth shared with the generator, so the read side and write side can't drift.


def available_within_sample_cohorts(*, proteoform: bool = False, scope: str = "cta") -> list[str]:
    """Cohort codes that ship a within-sample top-fraction shard (sorted).

    With ``proteoform=True``, the proteoform-summed variant (identical-protein
    members collapsed before ranking, in ``scope`` — see :func:`within_sample_top_fraction`)."""
    return _available_cohorts(_WITHIN_SAMPLE, proteoform=proteoform, scope=scope)


def within_sample_top_fraction(
    cancer_type,
    *,
    threshold: float = 0.95,
    proteoform: bool = False,
    scope: str = "cta",
    auto_fetch: bool = False,
) -> pd.DataFrame:
    """Per-gene fraction of a cohort's samples in which the gene is highly
    expressed *within that sample* — the "top ~5% expressed gene in this tumor"
    prevalence (signal a, the producer side of the within-sample signal).

    Returns one row per gene (``Ensembl_Gene_ID`` + ``Symbol``) with the
    ``frac_samples_top{1,5,10}pct`` column for the requested ``threshold``
    (0.99 / 0.95 / 0.90) plus ``n_samples``.

    With ``proteoform=True``, identical-protein paralogs (CTAG1A+CTAG1B, the CT47A
    family, …) are summed per sample *before* the within-sample ranking, so a
    duplicated antigen is ranked as one proteoform rather than several individually-
    diluted genes (``proteoform_key``/``Symbol`` carry the collapsed identity;
    ``scope`` ``"cta"``/``"genome"``, ignored when ``proteoform`` is False). Note
    collapsing members shrinks the gene axis the within-sample rank is computed over,
    so an ungrouped gene's fraction can shift slightly vs the gene variant.

    Reads the shipped shard when present, else **recomputes on the fly** from the
    per-sample matrix via the same build core (the live path for the proteoform
    variant until its shard ships) — needs the cohort's per-sample matrix cached (pass
    ``auto_fetch=True`` to download it), else a clear error.
    """
    col = _WITHIN_SAMPLE_THRESHOLD_COLS.get(threshold)
    if col is None:
        raise ValueError(f"threshold must be one of {sorted(_WITHIN_SAMPLE_THRESHOLD_COLS)}")
    code = resolve_cancer_type(cancer_type)
    df = _read_shard_or_recompute(
        _WITHIN_SAMPLE, code, proteoform=proteoform, auto_fetch=auto_fetch, scope=scope
    )
    keep = [*id_columns(df), col]
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
    :func:`oncoref.expression_builders.sum_proteoform_tpm` core can run inside the offline
    percentile/within-sample generators to ship proteoform-summed artifacts.
    """
    from .proteoforms import collapse_to_proteoforms

    wide = representative_cohort_samples(cancer_types, k=k, normalize="tpm_clean", format="wide")
    sample_cols = [c for c in wide.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    if wide.empty or not sample_cols:
        return wide
    return collapse_to_proteoforms(wide, sample_cols=sample_cols)


# ----- Symmetric gene-level / proteoform-level expression accessors -----
#
# Every expression dataset is exposed as a matched pair whose name carries the level —
# no boolean flag to read. ``gene_*`` is one row per Ensembl gene; ``proteoform_*``
# sums identical-protein paralogs to one row per proteoform key (see
# :func:`oncoref.proteoforms.expression_level`). Both are thin wrappers over the one
# base accessor, so the collapse logic lives in exactly one place; the unprefixed base
# names (``per_sample_expression`` etc.) remain the gene-level implementation.
#
#   dataset          gene_*                            proteoform_*
#   per-sample TPM   gene_per_sample_expression        proteoform_per_sample_expression
#   cohort-mean TPM  gene_cohort_mean_expression       proteoform_cohort_mean_expression
#   percentiles      gene_cohort_percentiles           proteoform_cohort_percentiles
#   within-sample    gene_within_sample_top_fraction   proteoform_within_sample_top_fraction
#   representatives  gene_representative_samples       proteoform_representative_samples


def gene_per_sample_expression(
    cancer_type, *, normalize: str = "tpm_clean", auto_fetch: bool = True
) -> pd.DataFrame:
    """Gene-level per-sample **TPM values** (one row per Ensembl gene). Proteoform
    counterpart: :func:`proteoform_per_sample_expression`."""
    return per_sample_expression(cancer_type, normalize=normalize, auto_fetch=auto_fetch)


def proteoform_per_sample_expression(
    cancer_type, *, normalize: str = "tpm_clean", auto_fetch: bool = True, scope: str = "cta"
) -> pd.DataFrame:
    """Proteoform-level per-sample **TPM values** — identical-protein paralogs summed
    per sample. Gene-level counterpart: :func:`gene_per_sample_expression`."""
    return per_sample_expression(
        cancer_type, normalize=normalize, auto_fetch=auto_fetch, proteoform=True, scope=scope
    )


def gene_cohort_mean_expression(
    cancer_type, *, normalize: str = "tpm_clean", statistic: str = "mean", auto_fetch: bool = True
) -> pd.DataFrame:
    """Gene-level across-patient **TPM** summary. Proteoform counterpart:
    :func:`proteoform_cohort_mean_expression`."""
    return cohort_mean_expression(
        cancer_type, normalize=normalize, statistic=statistic, auto_fetch=auto_fetch
    )


def proteoform_cohort_mean_expression(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    statistic: str = "mean",
    auto_fetch: bool = True,
    scope: str = "cta",
) -> pd.DataFrame:
    """Proteoform-level across-patient **TPM** summary. Gene-level counterpart:
    :func:`gene_cohort_mean_expression`."""
    return cohort_mean_expression(
        cancer_type,
        normalize=normalize,
        statistic=statistic,
        auto_fetch=auto_fetch,
        proteoform=True,
        scope=scope,
    )


def gene_cohort_stats(
    cancer_type, *, normalize: str = "tpm_clean", auto_fetch: bool = True
) -> pd.DataFrame:
    """Gene-level per-gene cohort **summary statistics** (mean/std + the percentile ladder
    min/p1/p5/p10/p15/p20/p25/p50/p75/p80/p85/p90/p95/p99/max). Proteoform counterpart:
    :func:`proteoform_cohort_stats`."""
    return cohort_stats(cancer_type, normalize=normalize, auto_fetch=auto_fetch)


def proteoform_cohort_stats(
    cancer_type, *, normalize: str = "tpm_clean", auto_fetch: bool = True, scope: str = "cta"
) -> pd.DataFrame:
    """Proteoform-level per-gene cohort **summary statistics**. Gene-level counterpart:
    :func:`gene_cohort_stats`."""
    return cohort_stats(
        cancer_type, normalize=normalize, auto_fetch=auto_fetch, proteoform=True, scope=scope
    )


def gene_pooled_cohort_stats(
    cancer_types, *, normalize: str = "tpm_clean", auto_fetch: bool = True, min_cohorts: int = 1
) -> pd.DataFrame:
    """Gene-level heterogeneity-safe cross-cohort pool. Proteoform counterpart:
    :func:`proteoform_pooled_cohort_stats`. (Alias of :func:`pooled_cohort_stats`.)"""
    return pooled_cohort_stats(
        cancer_types, normalize=normalize, auto_fetch=auto_fetch, min_cohorts=min_cohorts
    )


def proteoform_pooled_cohort_stats(
    cancer_types,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    scope: str = "cta",
    min_cohorts: int = 1,
) -> pd.DataFrame:
    """Proteoform-level heterogeneity-safe cross-cohort pool. Gene-level counterpart:
    :func:`gene_pooled_cohort_stats`."""
    return pooled_cohort_stats(
        cancer_types,
        normalize=normalize,
        auto_fetch=auto_fetch,
        proteoform=True,
        scope=scope,
        min_cohorts=min_cohorts,
    )


def gene_cohort_percentiles(
    cancer_type, *, as_tpm: bool = True, auto_fetch: bool = False
) -> pd.DataFrame:
    """Gene-level per-cohort **percentile vectors**. Proteoform counterpart:
    :func:`proteoform_cohort_percentiles`. (Alias of :func:`cohort_gene_percentiles`.)"""
    return cohort_gene_percentiles(cancer_type, as_tpm=as_tpm, auto_fetch=auto_fetch)


def proteoform_cohort_percentiles(
    cancer_type, *, as_tpm: bool = True, scope: str = "cta", auto_fetch: bool = True
) -> pd.DataFrame:
    """Proteoform-level per-cohort **percentile vectors** (members summed before
    ranking, ``scope`` ``"cta"``/``"genome"``). Gene-level counterpart:
    :func:`gene_cohort_percentiles`. No proteoform shard ships yet, so this **always**
    recomputes from the per-sample matrix — hence ``auto_fetch`` defaults to ``True``
    here (unlike the gene variant, which reads a shipped shard); pass ``False`` to
    require the matrix already be cached."""
    return cohort_gene_percentiles(
        cancer_type, as_tpm=as_tpm, proteoform=True, scope=scope, auto_fetch=auto_fetch
    )


def gene_within_sample_top_fraction(
    cancer_type, *, threshold: float = 0.95, auto_fetch: bool = False
) -> pd.DataFrame:
    """Gene-level within-sample top-fraction prevalence. Proteoform counterpart:
    :func:`proteoform_within_sample_top_fraction`."""
    return within_sample_top_fraction(cancer_type, threshold=threshold, auto_fetch=auto_fetch)


def proteoform_within_sample_top_fraction(
    cancer_type, *, threshold: float = 0.95, scope: str = "cta", auto_fetch: bool = True
) -> pd.DataFrame:
    """Proteoform-level within-sample top-fraction prevalence (``scope``
    ``"cta"``/``"genome"``). Gene-level counterpart: :func:`gene_within_sample_top_fraction`.
    No proteoform shard ships yet, so this **always** recomputes from the per-sample
    matrix — hence ``auto_fetch`` defaults to ``True`` here (unlike the gene variant,
    which reads a shipped shard); pass ``False`` to require the matrix already be cached."""
    return within_sample_top_fraction(
        cancer_type, threshold=threshold, proteoform=True, scope=scope, auto_fetch=auto_fetch
    )


def gene_representative_samples(
    cancer_types: str | Iterable[str] | None = None,
    *,
    k: int | None = None,
    normalize: str = "tpm_clean",
    format: str = "wide",
    include_provenance: bool = False,
) -> pd.DataFrame:
    """Gene-level representative per-sample vectors. Proteoform counterpart:
    :func:`proteoform_representative_samples`. (Alias of
    :func:`representative_cohort_samples`.)"""
    return representative_cohort_samples(
        cancer_types,
        k=k,
        normalize=normalize,
        format=format,
        include_provenance=include_provenance,
    )


def pan_cancer_expression(
    genes: str | Iterable[str] | None = None,
    *,
    normalize: str | Iterable[str] | None = "tpm_clean",
) -> pd.DataFrame:
    """Wide pan-cancer reference: each gene's expression across **50 HPA normal
    tissues** and **33 TCGA tumor cohorts**, tumor and normal side by side in one
    frame — the combined companion to the per-cohort accessors above.

    Columns are entity-first: ``<tissue>_nTPM_raw`` for HPA normal tissues and
    ``<CODE>_TPM_raw`` for TCGA tumor cohorts. Source ``FPKM_<CODE>`` columns are
    deterministically converted FPKM→TPM before being exposed as ``*_TPM_raw``.
    The default ``normalize="tpm_clean"`` (alias ``"clean_tpm"``) appends
    clean TPM companions named ``<entity>_<measure>_clean``.
    ``normalize="housekeeping"`` / ``"hk"`` appends ``*_hk`` columns, and
    ``"percentile"`` appends ``*_percentile`` columns. ``"tpm_log1p"`` and
    ``"tpm_clean_log1p"`` append natural-log companions over the raw and clean
    TPM/nTPM values, respectively. Pass ``normalize=None`` or ``"tpm"`` for the
    raw TPM/nTPM companions only.

    ``genes`` filters to the given Ensembl gene ids (version-insensitive) or
    symbols; ``None`` returns the full matrix. The FPKM→TPM conversion runs over
    **all** genes before any filtering, so a filtered slice still carries the
    cohort-wide TPM scaling."""
    df = get_data("pan-cancer-expression")
    # Dense canonical space: sum alt-haplotype/patch copies into their primary gene and
    # relabel retired ids (oncoref#135 item 6). The cohort/tissue columns are per-gene
    # abundances (nTPM / FPKM) — linear-additive — so summing the rows of a fragmented
    # gene is exact, and it precedes the FPKM->TPM rescale (whose per-cohort 1e6 total is
    # conserved under row-summing, so every other gene's conversion is unchanged).
    id_cols = ["Ensembl_Gene_ID", "Symbol"]
    value_cols = [c for c in df.columns if c not in id_cols]
    df = _canonicalize_gene_rows(df, sample_cols=value_cols)
    out = df[id_cols].copy()

    raw_cols: list[str] = []
    ntpm_cols = [c for c in df.columns if c.startswith("nTPM_")]
    for col in ntpm_cols:
        entity = col[len("nTPM_") :]
        target = f"{entity}_nTPM_raw"
        out[target] = pd.to_numeric(df[col], errors="coerce")
        raw_cols.append(target)

    fpkm_cols = [c for c in df.columns if c.startswith("FPKM_")]
    if fpkm_cols:
        from .normalization import fpkm_to_tpm

        converted, _ = fpkm_to_tpm(df[id_cols + fpkm_cols], value_cols=fpkm_cols)
        for col in fpkm_cols:
            entity = col[len("FPKM_") :]
            target = f"{entity}_TPM_raw"
            out[target] = converted[col]
            raw_cols.append(target)

    tpm_cols = [c for c in df.columns if c.startswith("TPM_")]
    for col in tpm_cols:
        entity = col[len("TPM_") :]
        target = f"{entity}_TPM_raw"
        out[target] = pd.to_numeric(df[col], errors="coerce")
        raw_cols.append(target)

    modes = _pan_cancer_normalize_modes(normalize)
    clean_cols: list[str] = []
    if modes & {"tpm_clean", "housekeeping", "hk", "percentile", "tpm_clean_log1p"}:
        clean = clean_tpm(out[raw_cols], gene_table=out[id_cols])
        for col in raw_cols:
            target = col[: -len("_raw")] + "_clean"
            out[target] = clean[col]
            clean_cols.append(target)

    if modes & {"housekeeping", "hk"}:
        hk_input_cols = clean_cols or raw_cols
        hk_input = out[id_cols + hk_input_cols].copy()
        hk, _ = tpm_to_housekeeping_normalized(hk_input, value_cols=hk_input_cols)
        for col in hk_input_cols:
            target = col.rsplit("_", 1)[0] + "_hk"
            out[target] = hk[col]

    if "percentile" in modes:
        pct_input_cols = clean_cols or raw_cols
        pct = percentile_rank(out[id_cols + pct_input_cols], value_cols=pct_input_cols)
        for col in pct_input_cols:
            target = col.rsplit("_", 1)[0] + "_percentile"
            out[target] = pct[col]

    if "tpm_log1p" in modes:
        for col in raw_cols:
            target = col + "_log1p"
            out[target] = np.log1p(out[col].to_numpy(dtype=float))

    if "tpm_clean_log1p" in modes:
        for col in clean_cols:
            target = col + "_log1p"
            out[target] = np.log1p(out[col].to_numpy(dtype=float))

    if genes is not None:
        wanted = {genes} if isinstance(genes, str) else set(genes)
        wanted = {str(g) for g in wanted}
        wanted_unversioned = {unversioned(g) for g in wanted}
        ids = out["Ensembl_Gene_ID"].astype(str)
        mask = (
            ids.isin(wanted)
            | ids.map(unversioned).isin(wanted_unversioned)
            | out["Symbol"].astype(str).isin(wanted)
        )
        out = out[mask].reset_index(drop=True)
    return out


def _pan_cancer_normalize_modes(normalize: str | Iterable[str] | None) -> set[str]:
    if normalize is None:
        return set()
    if isinstance(normalize, str):
        modes = {normalize.lower()}
    else:
        modes = {str(mode).lower() for mode in normalize}
    aliases = {"clean_tpm": "tpm_clean", "housekeeping": "housekeeping", "hk": "hk"}
    modes = {aliases.get(mode, mode) for mode in modes}
    allowed = {
        "tpm",
        "raw",
        "tpm_log1p",
        "tpm_clean",
        "housekeeping",
        "hk",
        "percentile",
        "tpm_clean_log1p",
    }
    unknown = modes - allowed
    if unknown:
        raise ValueError(
            "normalize must be None, 'tpm', 'tpm_clean'/'clean_tpm', "
            "'housekeeping', 'hk', 'percentile', 'tpm_log1p', "
            "or 'tpm_clean_log1p'"
        )
    return modes - {"tpm", "raw"}


def pooled_cohort_stats(
    cancer_types: str | Iterable[str],
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    proteoform: bool = False,
    scope: str = "cta",
    min_cohorts: int = 1,
) -> pd.DataFrame:
    """**Heterogeneity-safe** per-gene summary pooled across several cohorts.

    The cross-cohort companion to the single-cohort :func:`cohort_stats`. Pools the
    requested cohorts' per-sample matrices at the **sample** level on the shared key
    (``Ensembl_Gene_ID``, or ``proteoform_key`` when ``proteoform=True``) and
    summarizes each gene over the union of samples — **availability-aware**: a cell
    a cohort never measured is ``NaN`` and never treated as a zero, so the per-gene
    denominator ``n_available`` (not the constant ``n_samples``) is the honest one.

    Returns an id-keyed frame with the same statistic suite as :func:`cohort_stats`
    (``mean, std, min, p1, p5, p10, p15, p20, p25, p50, p75, p80, p85, p90, p95, p99, max`` over the
    pooled samples) plus the pooling columns:

    - ``balanced_mean`` — the mean of each cohort's per-gene mean, **equal weight
      per cohort**. The heterogeneity-safe central value: a large cohort can't
      dominate it the way it dominates the sample-pooled ``mean``. The gap between
      ``mean`` and ``balanced_mean`` is itself the cross-cohort imbalance signal.
    - ``std_between`` — std *across* the per-cohort means (between-cohort spread:
      how differently the cancer types express the gene), ``NaN`` for a single cohort.
    - ``n_samples`` — total pooled sample columns (constant across genes).
    - ``n_available`` — per-gene count of samples that **measured** it (non-``NaN``).
    - ``n_detected`` — measured **and** ``> 0``.
    - ``n_cohorts`` — how many of the pooled cohorts measured the gene.

    ``min_cohorts`` drops genes measured by fewer than that many cohorts (default
    ``1`` keeps everything). ``normalize`` selects the pooling space (linear clean
    TPM by default; see :func:`per_sample_expression`). ``proteoform=True`` pools
    the reduced proteoform key space (``scope`` ``"cta"``/``"genome"``). An aggregate
    code (e.g. ``"SARC"``) expands to its member subtypes — pooling them is exactly
    what a rollup cohort means."""
    codes = _resolve_cancer_types(cancer_types, expand_aggregates=True)
    codes = list(dict.fromkeys(codes or []))
    if not codes:
        raise ValueError("pooled_cohort_stats needs at least one cancer type")

    key = "proteoform_key" if proteoform else "Ensembl_Gene_ID"
    sample_frames: list[pd.DataFrame] = []  # per cohort: key-indexed sample matrix
    cohort_means: list[pd.Series] = []  # per cohort: key -> per-gene mean
    id_rows: list[pd.DataFrame] = []  # per cohort: id columns, key-indexed
    # per_sample_expression already returns the dense CANONICAL space (alt-haplotype copies
    # summed in linear TPM, retired ids relabeled, transform applied after) so an alias id
    # can't stand as a separate sparse row here — no extra canonicalization needed.
    for code in codes:
        df = per_sample_expression(
            code, normalize=normalize, auto_fetch=auto_fetch, proteoform=proteoform, scope=scope
        )
        id_cols = id_columns(df)
        samples = sample_columns(df)
        if not samples:
            continue
        indexed = df.set_index(key)
        mat = indexed[samples]
        # Sample labels can collide across cohorts ("s1"); namespace them so the
        # outer concat keeps every sample as its own column.
        mat = mat.add_prefix(f"{code}::")
        sample_frames.append(mat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN gene -> NaN mean
            cohort_means.append(indexed[samples].mean(axis=1, skipna=True).rename(code))
        id_rows.append(df.set_index(key)[[c for c in id_cols if c != key]])
    if not sample_frames:
        raise ValueError(f"no per-sample columns to pool for {cancer_types!r}")

    pooled = pd.concat(sample_frames, axis=1, join="outer").sort_index()
    per_cohort_mean = pd.concat(cohort_means, axis=1).reindex(pooled.index)
    ids = pd.concat(id_rows).groupby(level=0).first().reindex(pooled.index)

    measured = pooled.notna()
    mat = pooled.to_numpy(dtype=float)
    out = ids.reset_index()
    _write_cohort_stat_columns(out, mat)
    cohort_mean_mat = per_cohort_mean.to_numpy(dtype=float)
    n_cohorts = per_cohort_mean.notna().to_numpy().sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["balanced_mean"] = np.nanmean(cohort_mean_mat, axis=1)
        # Between-cohort spread is undefined for a single measuring cohort (nanstd
        # would report a misleading 0) -> NaN there, mirroring std's >=2 rule.
        out["std_between"] = np.where(n_cohorts >= 2, np.nanstd(cohort_mean_mat, axis=1), np.nan)
    out["n_samples"] = pooled.shape[1]
    out["n_available"] = measured.to_numpy().sum(axis=1)
    out["n_detected"] = ((mat > 0) & measured.to_numpy()).sum(axis=1)
    out["n_cohorts"] = n_cohorts
    if min_cohorts > 1:
        out = out[out["n_cohorts"] >= min_cohorts].reset_index(drop=True)
    return out
