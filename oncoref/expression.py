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

import json
import os
import re
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
from .expression_builders import (
    PERCENTILE_BREAKPOINTS as _PERCENTILE_BREAKPOINTS,
)
from .expression_builders import (
    WITHIN_SAMPLE_THRESHOLDS as _WITHIN_SAMPLE_THRESHOLD_COLS,
)
from .expression_engine import id_columns, sample_columns
from .gene_ids import ensembl_id_alias_symbols, resolve_ensembl_id, unversioned
from .load_dataset import _BUNDLED_DATA_DIR, _register_derived_cache, get_data
from .normalization import clean_tpm, percentile_rank, tpm_to_housekeeping_normalized
from .version import DATA_VERSION, SOURCE_MATRIX_VERSION


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
      - ``proteoform_fetch_scopes`` — proteoform scopes whose shards ship in the active
        bundle. Other scopes still recompute on demand from the per-sample matrix
        because proteoform groups are scope-specific.
      - ``build_attr`` — name of the :mod:`oncoref.expression_builders` core that regenerates a
        missing shard from the per-sample matrix (the same core that produced the shipped
        shards, so on-the-fly and shipped values agree).

    The registry of the concrete artifacts is :data:`SHARD_DATASETS`.
    """

    noun: str
    gene_dir: str
    gene_fetches: bool
    proteoform_stem: str | None = None
    proteoform_fetch_scopes: tuple[str, ...] = ()
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

    def fetches(self, *, proteoform: bool, scope: str = "cta") -> bool:
        """Whether the requested level's shards ship in a released bundle (so a missing
        shard may be auto-fetched rather than only recomputed)."""
        if not proteoform:
            return self.gene_fetches
        return scope in self.proteoform_fetch_scopes


#: The expression summary artifacts, keyed by short name. The QC-policy data bundle ships
#: gene-level percentiles, representatives, within-sample prevalence, and CTA-scope
#: proteoform percentile/within-sample shards. Other proteoform scopes still recompute
#: on demand from the source matrix.
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
        proteoform_fetch_scopes=("cta",),
        build_attr="cohort_percentile_vectors",
    ),
    "within_sample": ShardDataset(
        noun="within-sample top-fraction vector",
        gene_dir="cancer-reference-expression-within-sample-top5",
        gene_fetches=True,
        proteoform_stem="cancer-reference-expression-within-sample-top5-proteoform",
        proteoform_fetch_scopes=("cta",),
        build_attr="within_sample_top_fractions",
    ),
}
_REPRESENTATIVES = SHARD_DATASETS["representatives"]
_PERCENTILES = SHARD_DATASETS["percentiles"]
_WITHIN_SAMPLE = SHARD_DATASETS["within_sample"]

REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION = "representative_expression_v1"
PERCENTILE_ARTIFACT_SCHEMA_VERSION = "cohort_percentile_expression_v1"
REFERENCE_EXPRESSION_SCHEMA_VERSION = "cancer_reference_expression_v2"
REPRESENTATIVE_SELECTION_METHOD = "central_medoid_then_farthest_first"
REPRESENTATIVE_SELECTION_BASIS = "biological_clean_tpm_log1p_distance"


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

    ``auto_fetch=False`` skips the potentially large bundle download; the
    returned path simply won't exist when the shard is absent locally."""
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
    return sorted(p.stem for p in root.glob("*.parquet") if not p.name.startswith("._"))


def _shard_dir(dataset: ShardDataset, *, proteoform: bool = False, scope: str = "cta") -> Path:
    """The bundle directory holding ``dataset``'s per-cohort shards at the requested
    level and ``scope`` (the proteoform variant is scope-specific — see
    :meth:`ShardDataset.subdir`). The per-variant auto-fetch policy lives on the dataset
    record, so it's decided in one place rather than re-stated at each call site."""
    return _bundle_subdir(
        dataset.subdir(proteoform=proteoform, scope=scope),
        auto_fetch=dataset.fetches(proteoform=proteoform, scope=scope),
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


_REPRESENTATIVE_PROVENANCE_COLUMNS = [
    "source_cohort",
    "source_version",
    "source_project",
    "source_sample",
    "n_cohort_samples",
    "selection_rank",
    "selection_method",
    "selection_basis",
    "artifact_schema_version",
    "data_version",
    "source_matrix_version",
]


_INTERNAL_REPRESENTATIVE_ID_RE = re.compile(r"^(?P<code>.+)__rep(?P<rank>\d+)$")
_PIRLYGENES_REPRESENTATIVE_ID_RE = re.compile(r"^(?P<code>.+)_rep(?P<rank>\d+)$")


def _representative_rank(representative_id: object) -> int | None:
    text = str(representative_id)
    for pattern in (_INTERNAL_REPRESENTATIVE_ID_RE, _PIRLYGENES_REPRESENTATIVE_ID_RE):
        match = pattern.match(text)
        if match:
            return int(match.group("rank"))
    return None


def _representative_id_for_style(representative_id: object, *, style: str) -> str:
    _validate_representative_id_style(style)
    text = str(representative_id)
    if style == "internal":
        return text
    match = _INTERNAL_REPRESENTATIVE_ID_RE.match(text)
    if not match:
        return text
    return f"{match.group('code')}_rep{int(match.group('rank')):02d}"


def _validate_representative_id_style(style: str) -> None:
    if style not in {"pirlygenes", "internal"}:
        raise ValueError("representative_id_style must be 'pirlygenes' or 'internal'")


def _representative_attrs(
    *,
    codes: Iterable[str],
    normalize: str,
    format: str,
    k: int | None,
    representative_id_style: str,
) -> dict[str, object]:
    return {
        "artifact": "representative_samples",
        "schema_version": REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION,
        "data_version": DATA_VERSION,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "cancer_codes": list(codes),
        "normalize": normalize,
        "format": format,
        "k": k,
        "representative_id_style": representative_id_style,
        "selection_method": REPRESENTATIVE_SELECTION_METHOD,
        "selection_basis": REPRESENTATIVE_SELECTION_BASIS,
    }


def _representative_empty_frame(*, include_provenance: bool) -> pd.DataFrame:
    cols = ["Ensembl_Gene_ID", "Symbol", "cancer_code", "representative_id", "expression"]
    if include_provenance:
        cols.extend(_REPRESENTATIVE_PROVENANCE_COLUMNS)
    return pd.DataFrame(columns=cols)


def _attach_representative_provenance(long: pd.DataFrame, root: Path) -> pd.DataFrame:
    prov_path = root / "_provenance.csv"
    if prov_path.exists():
        prov = pd.read_csv(prov_path)
        keep = ["representative_id", *_REPRESENTATIVE_PROVENANCE_COLUMNS]
        long = long.merge(
            prov[[c for c in keep if c in prov.columns]],
            on="representative_id",
            how="left",
        )
    for col in ("source_cohort", "source_version", "source_project", "source_sample"):
        if col not in long.columns:
            long[col] = pd.NA
    for col in ("n_cohort_samples", "selection_rank"):
        if col not in long.columns:
            long[col] = pd.NA
    if "selection_rank" in long.columns:
        ranks = pd.Series(long["representative_id"].map(_representative_rank), index=long.index)
        long["selection_rank"] = (
            long["selection_rank"].astype("Int64").fillna(ranks).astype("Int64")
        )
    long["selection_method"] = REPRESENTATIVE_SELECTION_METHOD
    long["selection_basis"] = REPRESENTATIVE_SELECTION_BASIS
    long["artifact_schema_version"] = REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION
    long["data_version"] = DATA_VERSION
    long["source_matrix_version"] = SOURCE_MATRIX_VERSION
    return long


def _percentile_cols() -> list[str]:
    return [f"p{bp}" for bp in _PERCENTILE_BREAKPOINTS]


def _percentile_identity_cols(*, proteoform: bool) -> list[str]:
    if proteoform:
        return ["proteoform_key", "Ensembl_Gene_ID", "Symbol", "proteoform_members"]
    return ["Ensembl_Gene_ID", "Symbol"]


def _percentile_provenance_columns() -> list[str]:
    return [
        "cancer_code",
        "normalization",
        "expression_unit",
        "percentile_basis",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
    ]


def _percentile_attrs(
    *,
    code: str,
    as_tpm: bool,
    proteoform: bool,
    scope: str,
    source: str,
    missing_reason: str | None = None,
) -> dict[str, object]:
    attrs: dict[str, object] = {
        "artifact": "cohort_gene_percentiles",
        "schema_version": PERCENTILE_ARTIFACT_SCHEMA_VERSION,
        "data_version": DATA_VERSION,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "cancer_code": code,
        "as_tpm": as_tpm,
        "proteoform": proteoform,
        "scope": scope,
        "source": source,
        "normalization": "tpm_clean" if as_tpm else "tpm_clean_log1p",
        "expression_unit": "tpm_clean" if as_tpm else "log1p_tpm_clean",
        "percentile_basis": "biological_clean_tpm_across_samples",
    }
    if missing_reason is not None:
        attrs["missing_reason"] = missing_reason
    return attrs


def _empty_percentile_frame(
    *,
    code: str,
    as_tpm: bool,
    proteoform: bool,
    scope: str,
    include_provenance: bool,
    missing_reason: str,
) -> pd.DataFrame:
    cols = [*_percentile_identity_cols(proteoform=proteoform), *_percentile_cols()]
    if include_provenance:
        cols.extend(_percentile_provenance_columns())
    out = pd.DataFrame(columns=cols)
    out.attrs.update(
        _percentile_attrs(
            code=code,
            as_tpm=as_tpm,
            proteoform=proteoform,
            scope=scope,
            source="missing",
            missing_reason=missing_reason,
        )
    )
    return out


def _attach_percentile_provenance(df: pd.DataFrame, *, code: str, as_tpm: bool) -> pd.DataFrame:
    out = df.copy()
    out["cancer_code"] = code
    out["normalization"] = "tpm_clean" if as_tpm else "tpm_clean_log1p"
    out["expression_unit"] = "tpm_clean" if as_tpm else "log1p_tpm_clean"
    out["percentile_basis"] = "biological_clean_tpm_across_samples"
    out["artifact_schema_version"] = PERCENTILE_ARTIFACT_SCHEMA_VERSION
    out["data_version"] = DATA_VERSION
    out["source_matrix_version"] = SOURCE_MATRIX_VERSION
    return out


def available_representative_cohorts() -> list[str]:
    """Registry codes that ship a representative-samples shard (sorted)."""
    return _available_cohorts(_REPRESENTATIVES)


def available_percentile_cohorts(*, proteoform: bool = False, scope: str = "cta") -> list[str]:
    """Cohort codes that ship a percentile-vector shard (sorted). With
    ``proteoform=True``, the proteoform-summed variant (one vector per proteoform
    key, identical-protein members collapsed before ranking, in ``scope``)."""
    return _available_cohorts(_PERCENTILES, proteoform=proteoform, scope=scope)


def expression_artifact_gene_universe_deltas(
    *,
    product: str | None = None,
    cancer_type: str | None = None,
    delta_kind: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """Known row-universe deltas between pirlygenes and oncoref expression artifacts.

    This is a provenance/audit table, not a value-transforming compatibility shim. It
    records the known remapped, missing, and extra rows from the pirlygenes 5.23.2 vs
    oncoref 5.23.3 parity run tracked in #191/#193 so downstream migration code can
    distinguish intentional canonicalization from unresolved artifact differences.

    Optional filters are exact for ``product``, ``delta_kind``, and ``status``.
    ``cancer_type`` resolves aliases and matches semicolon-separated cohort lists in the
    table (for deltas shared by PRAD/COAD_MSI/READ_MSI, for example).
    """
    df = get_data("expression-artifact-gene-universe-deltas")
    if product is not None:
        df = df[df["product"].astype(str) == str(product)]
    if delta_kind is not None:
        df = df[df["delta_kind"].astype(str) == str(delta_kind)]
    if status is not None:
        df = df[df["status"].astype(str) == str(status)]
    if cancer_type is not None:
        code = resolve_cancer_type(cancer_type)

        def _matches(cell: object) -> bool:
            codes = [c for c in str(cell or "").split(";") if c]
            return code in codes

        df = df[df["cancer_code"].map(_matches)]
    df = df.reset_index(drop=True)
    df.attrs["comparison"] = "pirlygenes_5.23.2_vs_oncoref_5.23.3"
    df.attrs["issues"] = ["#191", "#193"]
    return df


def expression_artifact_gene_universe_delta_summary() -> pd.DataFrame:
    """Counts of known expression-artifact row-universe deltas by product/status."""
    df = expression_artifact_gene_universe_deltas()
    if df.empty:
        return pd.DataFrame(columns=["product", "cancer_code", "delta_kind", "status", "n"])
    return (
        df.groupby(["product", "cancer_code", "delta_kind", "status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )


_PER_SAMPLE_NORMALIZE = ("tpm_raw", "tpm_clean", "tpm_clean_log1p", "tpm_clean_hk")
_SAMPLE_QC_MODES = ("all", "pass", "pass_or_warn")
SAMPLE_EXPRESSION_QC_POLICY_VERSION = "sample_expression_qc_v1"
SOURCE_MATRIX_SAMPLE_QC_MANIFEST_SCHEMA_VERSION = "source_matrix_sample_qc_manifest_v1"
EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION = "expression_artifact_build_metadata_v1"
SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH = "source-matrix-sample-qc.csv"
EXPRESSION_ARTIFACT_BUILD_METADATA_PATH = "expression-artifact-build-metadata.csv"
EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH = "expression-artifact-build-metadata.json"

_SOURCE_MATRIX_SAMPLE_QC_MANIFEST_COLUMNS = [
    "cancer_code",
    "sample_qc_policy_version",
    "source_cohort",
    "source_type",
    "unit",
    "source_scale_class",
    "linear_tpm_comparable",
    "sample_id",
    "n_measured_genes",
    "n_detected_genes",
    "n_detected_raw",
    "n_detected_clean",
    "n_detected_clean_biological",
    "detected_gene_fraction",
    "zero_fraction_raw",
    "parse_missing_fraction",
    "total_tpm",
    "top_gene_id",
    "top_gene_symbol",
    "top_gene_tpm",
    "top_gene_fraction",
    "top1_fraction_raw",
    "top10_tpm",
    "top10_fraction",
    "top10_fraction_raw",
    "top1_fraction_clean",
    "top10_fraction_clean",
    "housekeeping_genes_present",
    "housekeeping_genes_detected",
    "housekeeping_genes_above_30",
    "housekeeping_zero_fraction",
    "tpm_proxy",
    "qc_flags",
    "qc_status",
    "qc_reasons",
    "sample_qc_status",
    "sample_qc_reasons",
    "passes_expression_qc",
    "recommended_for_absolute_tpm_floor",
    "source_matrix_path",
]

_EXPRESSION_ARTIFACT_BUILD_METADATA_COLUMNS = [
    "cancer_code",
    "source_cohort",
    "source_version",
    "source_matrix_path",
    "sample_qc",
    "sample_qc_policy_version",
    "n_source_samples",
    "n_cohort_samples",
    "n_qc_pass",
    "n_qc_warn",
    "n_qc_fail",
]
DEFAULT_MIN_DETECTED_GENES_FOR_QC = 5000
DEFAULT_MIN_HOUSEKEEPING_GENES_FOR_QC = 10
DEFAULT_MAX_TOP_GENE_FRACTION_FOR_QC = 0.20
DEFAULT_MAX_TOP10_GENE_FRACTION_FOR_QC = 0.50
_BLOCKING_SAMPLE_QC_REASONS = frozenset(
    {
        "low_detected_genes",
        "low_housekeeping_detection",
        "high_top_gene_fraction",
        "high_top10_gene_fraction",
    }
)


def _clean_tpm_housekeeping_panel_ids() -> frozenset[str]:
    from .gene_families import clean_tpm_biological_housekeeping_gene_ids

    return clean_tpm_biological_housekeeping_gene_ids()


def _min_housekeeping_detected(panel_ids) -> int | None:
    n = len(panel_ids)
    return min(DEFAULT_MIN_HOUSEKEEPING_GENES_FOR_QC, n) if n else None


def _selected_expression_source_metadata(code: str) -> dict[str, str | bool | None]:
    source_type = unit = None
    try:
        info = source_matrices.cohort_info(code)
        source_cohort = str(info.get("source_cohort") or "")
    except source_matrices.SourceMatrixError:
        source_cohort = ""

    from .expression_registry import sources_for_cancer_code

    sources = sources_for_cancer_code(code)
    selected = next(
        (s for s in sources if source_cohort and s.source_cohort == source_cohort),
        sources[0] if sources else None,
    )
    if selected is not None:
        source_type = selected.source_type
        unit = selected.unit
        if not source_cohort:
            source_cohort = selected.source_cohort or ""
        special = selected.special_handling or ""
    else:
        special = ""
    text = " ".join(str(x or "") for x in (source_type, unit, special)).lower()
    tpm_proxy = "microarray" in text or "tpm-proxy" in text or "tpm proxy" in text
    if tpm_proxy:
        source_scale_class = "microarray_tpm_proxy"
        linear_tpm_comparable = False
    elif selected is not None:
        source_scale_class = "linear_rnaseq_tpm"
        linear_tpm_comparable = True
    else:
        source_scale_class = "unknown"
        linear_tpm_comparable = False
    return {
        "source_cohort": source_cohort or None,
        "source_type": source_type,
        "unit": unit,
        "source_scale_class": source_scale_class,
        "linear_tpm_comparable": linear_tpm_comparable,
        "tpm_proxy": tpm_proxy,
    }


def _sample_qc_status(reasons: list[str]) -> str:
    if set(reasons) & _BLOCKING_SAMPLE_QC_REASONS:
        return "fail"
    return "warn" if reasons else "pass"


def _validate_sample_qc(sample_qc: str) -> str:
    mode = str(sample_qc).lower()
    if mode not in _SAMPLE_QC_MODES:
        raise ValueError(f"sample_qc must be one of {_SAMPLE_QC_MODES}")
    return mode


def _apply_sample_qc_filter(
    df: pd.DataFrame, code: str, *, sample_qc: str, auto_fetch: bool
) -> pd.DataFrame:
    mode = _validate_sample_qc(sample_qc)
    if mode == "all":
        return df
    samples = sample_columns(df)
    if not samples:
        return df
    qc = sample_expression_qc(code, auto_fetch=auto_fetch)
    if qc.empty:
        return df[id_columns(df)].copy()
    if mode == "pass":
        allowed = set(qc.loc[qc["sample_qc_status"] == "pass", "sample_id"].astype(str))
    else:
        allowed = set(
            qc.loc[qc["sample_qc_status"].isin(["pass", "warn"]), "sample_id"].astype(str)
        )
    keep_samples = [s for s in samples if s in allowed]
    return df[[*id_columns(df), *keep_samples]].copy()


def sample_expression_qc_from_matrix(
    raw: pd.DataFrame,
    *,
    cancer_type=None,
    source_metadata: dict[str, str | bool | None] | None = None,
    min_detected_genes: int = DEFAULT_MIN_DETECTED_GENES_FOR_QC,
    min_housekeeping_detected: int | None = None,
    max_top_gene_fraction: float = DEFAULT_MAX_TOP_GENE_FRACTION_FOR_QC,
    max_top10_gene_fraction: float = DEFAULT_MAX_TOP10_GENE_FRACTION_FOR_QC,
) -> pd.DataFrame:
    """Per-sample QC metrics for an already-loaded raw expression matrix.

    This is the shared policy core behind the public read-path QC and the offline
    artifact rebuild. It canonicalizes gene rows before measuring sparsity so the QC
    contract is evaluated in the same gene-ID space used by expression accessors.
    """
    code = (
        resolve_cancer_type(cancer_type, strict=False) or cancer_type
        if cancer_type is not None
        else None
    )
    raw = _canonicalize_gene_rows(raw, sample_cols=sample_columns(raw)).reset_index(drop=True)
    samples = sample_columns(raw)
    if not samples:
        return pd.DataFrame()

    id_cols = id_columns(raw)
    ids = raw["Ensembl_Gene_ID"].astype(str).map(unversioned)
    panel_ids = _clean_tpm_housekeeping_panel_ids()
    panel = {str(g).split(".")[0] for g in panel_ids}
    hk_rows = ids.isin(panel)
    from .gene_families import clean_tpm_censored_gene_ids

    biological_rows = ~ids.isin(clean_tpm_censored_gene_ids())
    clean = clean_tpm(raw[samples], gene_table=raw[id_cols])
    min_hk = (
        min_housekeeping_detected
        if min_housekeeping_detected is not None
        else _min_housekeeping_detected(panel_ids)
    )
    if source_metadata is None:
        meta = (
            _selected_expression_source_metadata(str(code))
            if code is not None
            else {
                "source_cohort": None,
                "source_type": None,
                "unit": None,
                "source_scale_class": "unknown",
                "linear_tpm_comparable": False,
                "tpm_proxy": False,
            }
        )
    else:
        meta = {
            "source_cohort": None,
            "source_type": None,
            "unit": None,
            "source_scale_class": "unknown",
            "linear_tpm_comparable": False,
            "tpm_proxy": False,
            **source_metadata,
        }

    rows: list[dict] = []
    for sample in samples:
        vals = pd.to_numeric(raw[sample], errors="coerce")
        clean_vals = pd.to_numeric(clean[sample], errors="coerce")
        measured = vals.notna()
        positive = vals > 0
        n_measured = int(measured.sum())
        n_detected = int(positive.sum())
        n_zero = int((vals == 0).sum())
        n_missing = int(vals.isna().sum())
        n_detected_clean = int((clean_vals > 0).sum())
        n_detected_clean_biological = int((clean_vals[biological_rows] > 0).sum())
        total = float(vals.sum(skipna=True))
        top_idx = vals.idxmax(skipna=True) if n_measured else None
        top_tpm = float(vals.loc[top_idx]) if top_idx is not None and pd.notna(top_idx) else 0.0
        top10_tpm = float(vals.nlargest(min(10, n_measured)).sum()) if n_measured else 0.0
        clean_total = float(clean_vals.sum(skipna=True))
        clean_top_tpm = float(clean_vals.max(skipna=True)) if n_measured else 0.0
        clean_top10_tpm = (
            float(clean_vals.nlargest(min(10, n_measured)).sum()) if n_measured else 0.0
        )
        hk_vals = vals[hk_rows]
        hk_measured = int(hk_vals.notna().sum())
        hk_detected = int((hk_vals > 0).sum())
        hk_zero = int((hk_vals == 0).sum())
        hk_floor_count = int((hk_vals >= 30).sum())
        top_fraction = top_tpm / total if total > 0 else np.nan
        top10_fraction = top10_tpm / total if total > 0 else np.nan
        clean_top_fraction = clean_top_tpm / clean_total if clean_total > 0 else np.nan
        clean_top10_fraction = clean_top10_tpm / clean_total if clean_total > 0 else np.nan
        detected_fraction = n_detected / n_measured if n_measured else np.nan
        zero_fraction = n_zero / n_measured if n_measured else np.nan
        parse_missing_fraction = n_missing / len(vals) if len(vals) else np.nan
        hk_zero_fraction = hk_zero / hk_measured if hk_measured else np.nan

        flags: list[str] = []
        if n_detected < min_detected_genes:
            flags.append("low_detected_genes")
        if min_hk is not None and hk_detected < min_hk:
            flags.append("low_housekeeping_detection")
        if pd.notna(top_fraction) and top_fraction > max_top_gene_fraction:
            flags.append("high_top_gene_fraction")
        if (
            n_measured > 10
            and pd.notna(top10_fraction)
            and top10_fraction > max_top10_gene_fraction
        ):
            flags.append("high_top10_gene_fraction")
        if meta["tpm_proxy"]:
            flags.append("tpm_proxy_scale")

        status = _sample_qc_status(flags)
        rows.append(
            {
                "cancer_code": code,
                "sample_qc_policy_version": SAMPLE_EXPRESSION_QC_POLICY_VERSION,
                "source_cohort": meta["source_cohort"],
                "source_type": meta["source_type"],
                "unit": meta["unit"],
                "source_scale_class": meta["source_scale_class"],
                "linear_tpm_comparable": bool(meta["linear_tpm_comparable"]),
                "sample_id": sample,
                "n_measured_genes": n_measured,
                "n_detected_genes": n_detected,
                "n_detected_raw": n_detected,
                "n_detected_clean": n_detected_clean,
                "n_detected_clean_biological": n_detected_clean_biological,
                "detected_gene_fraction": detected_fraction,
                "zero_fraction_raw": zero_fraction,
                "parse_missing_fraction": parse_missing_fraction,
                "total_tpm": total,
                "top_gene_id": raw.loc[top_idx, "Ensembl_Gene_ID"] if top_idx is not None else None,
                "top_gene_symbol": raw.loc[top_idx, "Symbol"] if top_idx is not None else None,
                "top_gene_tpm": top_tpm,
                "top_gene_fraction": top_fraction,
                "top1_fraction_raw": top_fraction,
                "top10_tpm": top10_tpm,
                "top10_fraction": top10_fraction,
                "top10_fraction_raw": top10_fraction,
                "top1_fraction_clean": clean_top_fraction,
                "top10_fraction_clean": clean_top10_fraction,
                "housekeeping_genes_present": hk_measured,
                "housekeeping_genes_detected": hk_detected,
                "housekeeping_genes_above_30": hk_floor_count,
                "housekeeping_zero_fraction": hk_zero_fraction,
                "tpm_proxy": bool(meta["tpm_proxy"]),
                "qc_flags": ";".join(flags),
                "qc_status": status,
                "qc_reasons": ";".join(flags),
                "sample_qc_status": status,
                "sample_qc_reasons": ";".join(flags),
                "passes_expression_qc": status != "fail",
                "recommended_for_absolute_tpm_floor": status != "fail"
                and bool(meta["linear_tpm_comparable"]),
            }
        )
    return pd.DataFrame(rows)


def sample_expression_qc(
    cancer_type,
    *,
    auto_fetch: bool = True,
    min_detected_genes: int = DEFAULT_MIN_DETECTED_GENES_FOR_QC,
    min_housekeeping_detected: int | None = None,
    max_top_gene_fraction: float = DEFAULT_MAX_TOP_GENE_FRACTION_FOR_QC,
    max_top10_gene_fraction: float = DEFAULT_MAX_TOP10_GENE_FRACTION_FOR_QC,
) -> pd.DataFrame:
    """Per-sample QC metrics for a cohort's raw expression matrix.

    This is an audit surface over the raw per-sample matrix before clean-TPM
    normalization. It is designed to catch source/sample artifacts such as literal-zero
    sparsity in otherwise universal genes, while still making source-type caveats
    explicit (for example microarray TPM-proxy sources). It does not exclude samples by
    itself; downstream code can use ``passes_expression_qc`` or inspect ``qc_flags``.
    """
    code = resolve_cancer_type(cancer_type, strict=False) or cancer_type
    raw = per_sample_expression(code, normalize="tpm_raw", auto_fetch=auto_fetch, sample_qc="all")
    return sample_expression_qc_from_matrix(
        raw,
        cancer_type=code,
        min_detected_genes=min_detected_genes,
        min_housekeeping_detected=min_housekeeping_detected,
        max_top_gene_fraction=max_top_gene_fraction,
        max_top10_gene_fraction=max_top10_gene_fraction,
    )


def _validate_metadata_on_missing(on_missing: str) -> str:
    mode = str(on_missing).lower()
    if mode not in {"empty", "raise"}:
        raise ValueError("on_missing must be 'empty' or 'raise'")
    return mode


def _empty_metadata_frame(columns: list[str], *, schema_version: str, missing_reason: str):
    out = pd.DataFrame(columns=columns)
    out.attrs["schema_version"] = schema_version
    out.attrs["data_version"] = DATA_VERSION
    out.attrs["source_matrix_version"] = SOURCE_MATRIX_VERSION
    out.attrs["missing_reason"] = missing_reason
    return out


def _optional_bundle_metadata_path(relative_path: str, *, auto_fetch: bool) -> Path | None:
    path = data_bundle.find(relative_path)
    if path is not None:
        return path
    if auto_fetch:
        data_bundle.ensure_local(auto_fetch=True, verbose=False)
        path = data_bundle.find(relative_path)
    return path


def _requested_cancer_codes(cancer_type) -> set[str] | None:
    if cancer_type is None:
        return None
    if isinstance(cancer_type, str):
        values = [cancer_type]
    else:
        values = list(cancer_type)
    return {resolve_cancer_type(value, strict=False) or str(value) for value in values}


def source_matrix_sample_qc_manifest(
    cancer_type=None,
    *,
    sample_qc: str = "all",
    auto_fetch: bool = True,
    on_missing: str = "empty",
) -> pd.DataFrame:
    """Read the generated source-matrix sample-QC manifest from the data bundle.

    This is the bundle-level companion to :func:`sample_expression_qc`, which computes
    QC live from a single source matrix. The rebuild script emits
    ``source-matrix-sample-qc.csv`` for the exact samples that fed generated
    expression artifacts. Current released bundles may not yet contain it; by default
    this returns an empty schema-stable frame rather than synthesizing rows.
    """
    mode = _validate_metadata_on_missing(on_missing)
    path = _optional_bundle_metadata_path(
        SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH, auto_fetch=auto_fetch
    )
    if path is None:
        if mode == "raise":
            raise FileNotFoundError(
                f"{SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH} is not present in the "
                f"oncoref data bundle at {data_bundle.cache_dir()}"
            )
        return _empty_metadata_frame(
            _SOURCE_MATRIX_SAMPLE_QC_MANIFEST_COLUMNS,
            schema_version=SOURCE_MATRIX_SAMPLE_QC_MANIFEST_SCHEMA_VERSION,
            missing_reason="source-matrix sample QC manifest not present in bundle",
        )

    out = pd.read_csv(path)
    for col in _SOURCE_MATRIX_SAMPLE_QC_MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    extra_cols = [
        col for col in out.columns if col not in _SOURCE_MATRIX_SAMPLE_QC_MANIFEST_COLUMNS
    ]
    out = out[[*_SOURCE_MATRIX_SAMPLE_QC_MANIFEST_COLUMNS, *extra_cols]]

    codes = _requested_cancer_codes(cancer_type)
    if codes is not None:
        if not codes:
            out = out.iloc[0:0].copy()
        else:
            out = out[out["cancer_code"].astype(str).isin(codes)].copy()
    qc_mode = _validate_sample_qc(sample_qc)
    if qc_mode == "pass":
        out = out[out["sample_qc_status"].astype(str) == "pass"].copy()
    elif qc_mode == "pass_or_warn":
        out = out[out["sample_qc_status"].astype(str).isin(["pass", "warn"])].copy()

    out.attrs["schema_version"] = SOURCE_MATRIX_SAMPLE_QC_MANIFEST_SCHEMA_VERSION
    out.attrs["data_version"] = DATA_VERSION
    out.attrs["source_matrix_version"] = SOURCE_MATRIX_VERSION
    out.attrs["path"] = str(path)
    return out.reset_index(drop=True)


def expression_artifact_build_metadata(
    cancer_type=None,
    *,
    auto_fetch: bool = True,
    on_missing: str = "empty",
) -> pd.DataFrame:
    """Read per-cohort build metadata for generated expression artifacts.

    The rows are emitted by ``scripts/rebuild_expression_artifacts.py`` and record the
    selected source matrix, QC policy, selected sample count, and QC pass/warn/fail
    counts for each cohort. Missing current-bundle metadata returns a schema-stable
    empty frame by default.
    """
    mode = _validate_metadata_on_missing(on_missing)
    path = _optional_bundle_metadata_path(
        EXPRESSION_ARTIFACT_BUILD_METADATA_PATH, auto_fetch=auto_fetch
    )
    if path is None:
        if mode == "raise":
            raise FileNotFoundError(
                f"{EXPRESSION_ARTIFACT_BUILD_METADATA_PATH} is not present in the "
                f"oncoref data bundle at {data_bundle.cache_dir()}"
            )
        return _empty_metadata_frame(
            _EXPRESSION_ARTIFACT_BUILD_METADATA_COLUMNS,
            schema_version=EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION,
            missing_reason="expression artifact build metadata not present in bundle",
        )

    out = pd.read_csv(path)
    for col in _EXPRESSION_ARTIFACT_BUILD_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    extra_cols = [
        col for col in out.columns if col not in _EXPRESSION_ARTIFACT_BUILD_METADATA_COLUMNS
    ]
    out = out[[*_EXPRESSION_ARTIFACT_BUILD_METADATA_COLUMNS, *extra_cols]]
    codes = _requested_cancer_codes(cancer_type)
    if codes is not None:
        if not codes:
            out = out.iloc[0:0].copy()
        else:
            out = out[out["cancer_code"].astype(str).isin(codes)].copy()
    out.attrs["schema_version"] = EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    out.attrs["data_version"] = DATA_VERSION
    out.attrs["source_matrix_version"] = SOURCE_MATRIX_VERSION
    out.attrs["path"] = str(path)
    return out.reset_index(drop=True)


def expression_artifact_build_summary(
    *,
    auto_fetch: bool = True,
    on_missing: str = "empty",
) -> dict:
    """Read bundle-level expression artifact build metadata.

    Returns the JSON summary emitted beside the per-cohort metadata CSV. The summary
    remains optional until a regenerated data bundle ships it; callers that require it
    should pass ``on_missing="raise"``.
    """
    mode = _validate_metadata_on_missing(on_missing)
    path = _optional_bundle_metadata_path(
        EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH, auto_fetch=auto_fetch
    )
    if path is None:
        if mode == "raise":
            raise FileNotFoundError(
                f"{EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH} is not present in the "
                f"oncoref data bundle at {data_bundle.cache_dir()}"
            )
        return {
            "schema_version": EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION,
            "data_version": DATA_VERSION,
            "source_matrix_version": SOURCE_MATRIX_VERSION,
            "missing_reason": "expression artifact build summary not present in bundle",
        }
    with path.open() as handle:
        out = json.load(handle)
    out.setdefault("schema_version", EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION)
    out.setdefault("data_version", DATA_VERSION)
    out.setdefault("source_matrix_version", SOURCE_MATRIX_VERSION)
    out["path"] = str(path)
    return out


def per_sample_expression(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    proteoform: bool = False,
    scope: str = "cta",
    sample_qc: str = "all",
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
      - ``"tpm_clean_hk"`` — clean TPM divided per sample by the biological
        clean-TPM housekeeping-panel geometric mean (unit-free ratio-to-baseline,
        robust to library-depth drift);
      - ``"tpm_raw"`` — the matrix as shipped (raw TPM), no normalization.

    With ``proteoform=True``, identical-protein paralogs are **summed per sample** to
    proteoform level (:func:`oncoref.proteoforms.collapse_to_proteoforms`, ``scope``
    = ``"cta"``/``"genome"``) — a **proteoform-level** frame carrying ``proteoform_key``
    (see :func:`oncoref.proteoforms.expression_level`). The sum is always taken in
    **linear** TPM and the ``log1p`` transform (if any) applied *after*, so the
    proteoform value is ``log1p(Σ member TPM)``, not the meaningless ``Σ log1p``.
    (``scope`` is ignored when ``proteoform=False``.)

    ``auto_fetch=False`` raises instead of downloading if the matrix isn't cached.
    ``sample_qc`` can be ``"all"`` (default), ``"pass"``, or ``"pass_or_warn"``; the
    latter two use :func:`sample_expression_qc` to drop failing sample columns while
    retaining the gene rows.
    Returns ``Ensembl_Gene_ID``, ``Symbol`` and one column per sample (plus the
    proteoform identity columns when collapsed).
    """
    if normalize not in _PER_SAMPLE_NORMALIZE:
        raise ValueError(f"normalize must be one of {_PER_SAMPLE_NORMALIZE}")
    sample_qc = _validate_sample_qc(sample_qc)
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
        out = _load_per_sample_matrix(str(path), mtime, normalize).copy()
        return _apply_sample_qc_filter(out, str(code), sample_qc=sample_qc, auto_fetch=auto_fetch)
    # Proteoform level: sum members in LINEAR TPM, then apply the requested transform.
    from .proteoforms import collapse_to_proteoforms

    linear = "tpm_raw" if normalize == "tpm_raw" else "tpm_clean"
    out = collapse_to_proteoforms(_load_per_sample_matrix(str(path), mtime, linear), scope=scope)
    samples = sample_columns(out)
    if normalize == "tpm_clean_log1p":
        out[samples] = np.log1p(out[samples].to_numpy(dtype=float))
    elif normalize == "tpm_clean_hk":
        out = _housekeeping_normalize(out, samples)
    return _apply_sample_qc_filter(out, str(code), sample_qc=sample_qc, auto_fetch=auto_fetch)


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
    """Divide each clean-TPM sample column by the biological housekeeping-panel geometric
    mean. Commutes with the proteoform sum (the denominator is per-column), so it can
    be applied before or after collapse."""
    from .normalization import tpm_to_housekeeping_normalized

    panel_ids = _clean_tpm_housekeeping_panel_ids()
    out, _ = tpm_to_housekeeping_normalized(
        df,
        value_cols=list(sample_cols),
        panel_ids=panel_ids,
        panel_name="clean_tpm_biological_housekeeping",
        min_panel_detected=_min_housekeeping_detected(panel_ids),
        drop_zero_panel_values=True,
        warn_on_unreliable=True,
    )
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
    sample_qc: str = "pass",
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
    :func:`oncoref.proteoforms.expression_level`). ``sample_qc`` defaults to
    ``"pass"`` so live summaries exclude source/sample QC failures; pass ``"all"`` for
    forensic parity with the current source matrix. ``scope`` selects the gene
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
        sample_qc=sample_qc,
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
    include_request_metadata: bool = False,
    on_missing: str = "omit",
    auto_fetch: bool = False,
    sample_qc: str = "pass",
) -> pd.DataFrame:
    """Observed tumor expression references as cohort-level clean TPM summaries.

    The default long form mirrors the downstream reference contract:
    ``Ensembl_Gene_ID``, ``Symbol``, ``cancer_code``, ``source_cohort``,
    ``normalization``, ``expression`` (median clean TPM), ``q1`` and ``q3``.
    Wide form returns one row per gene with ``<CODE>_<normalization>`` columns.

    ``normalize`` accepts one mode or an iterable of modes:
      - ``"tpm_clean"`` / ``"clean_tpm"``: shipped biological clean-TPM percentiles;
      - ``"tpm_clean_biological"``: explicit alias for that biological-only artifact;
      - ``"tpm_clean_log1p"``: stored log1p biological clean-TPM percentiles;
      - ``"tpm_raw"`` / ``"tpm"`` / ``"raw_tpm"``: raw-TPM cohort stats recomputed from
        the source matrix.

    Raw-TPM mode needs the per-sample source matrix available; pass
    ``auto_fetch=True`` to download it. Raw-TPM summaries default to
    ``sample_qc="pass"`` so sparse/source-QC-failed samples do not shape new
    derived summaries; use ``"pass_or_warn"`` or ``"all"`` for audit/parity views.
    Missing requested cohorts are omitted by default to preserve the historical
    behavior; pass ``on_missing="empty"`` to preserve a schema-stable empty result
    with missing-request metadata in ``df.attrs["missing_requests"]`` or
    ``on_missing="raise"`` to fail when any requested cohort/mode is unavailable.
    """
    modes = _reference_normalize_modes(normalize)
    sample_qc = _validate_sample_qc(sample_qc)
    if format not in ("long", "wide"):
        raise ValueError("format must be 'long' or 'wide'")
    if on_missing not in ("omit", "empty", "raise"):
        raise ValueError("on_missing must be 'omit', 'empty', or 'raise'")
    if include_request_metadata and format != "long":
        raise ValueError("include_request_metadata=True requires format='long'")

    requests = _reference_expression_requests(cancer_types, modes)
    availability = _reference_expression_availability_for_requests(
        requests, modes, sample_qc=sample_qc
    )
    missing = availability.loc[~availability["available"]].reset_index(drop=True)
    if on_missing == "raise" and not missing.empty:
        detail = ", ".join(
            f"{r.cancer_code}/{r.normalization}: {r.missing_reason}"
            for r in missing.itertuples(index=False)
        )
        raise ValueError(f"missing cancer reference expression artifact(s): {detail}")
    available_keys = {
        (str(r.requested_code), str(r.cancer_code), str(r.normalization))
        for r in availability.loc[availability["available"]].itertuples(index=False)
    }
    request_lookup = {
        (str(r.requested_code), str(r.cancer_code), str(r.normalization)): r
        for r in availability.loc[availability["available"]].itertuples(index=False)
    }

    long_parts: list[pd.DataFrame] = []
    wide_parts: list[pd.DataFrame] = []
    for request in requests:
        code = request["cancer_code"]
        for mode in modes:
            request_key = (request["requested_code"], code, mode)
            if request_key not in available_keys:
                continue
            ref, method = _reference_expression_frame(
                code, mode, auto_fetch=auto_fetch, sample_qc=sample_qc
            )
            ref = ref[_gene_filter_mask(ref, genes)].reset_index(drop=True)
            label = _REFERENCE_NORMALIZE_LABELS[mode]
            if format == "long":
                part = ref[["Ensembl_Gene_ID", "Symbol", "p25", "p50", "p75"]].copy()
                part.insert(2, "cancer_code", code)
                part["normalization"] = label
                if include_request_metadata:
                    request_row = request_lookup[request_key]
                    part["requested_code"] = request_row.requested_code
                    part["request_kind"] = request_row.request_kind
                    part["available"] = bool(request_row.available)
                    part["missing_reason"] = str(request_row.missing_reason)
                part = part.rename(columns={"p50": "expression", "p25": "q1", "p75": "q3"})
                if include_provenance:
                    provenance = _reference_expression_provenance(
                        code, mode, method, sample_qc=sample_qc
                    )
                    for col, value in provenance.items():
                        part[col] = value
                long_parts.append(
                    part[_reference_long_columns(include_provenance, include_request_metadata)]
                )
            else:
                suffix = _REFERENCE_WIDE_SUFFIXES[mode]
                part = ref[["Ensembl_Gene_ID", "Symbol", "p50"]].rename(
                    columns={"p50": f"{code}_{suffix}"}
                )
                wide_parts.append(part)

    if format == "long":
        cols = _reference_long_columns(include_provenance, include_request_metadata)
        if not long_parts:
            out = pd.DataFrame(columns=cols)
        else:
            out = pd.concat(long_parts, ignore_index=True)
        _attach_reference_expression_attrs(out, availability if on_missing != "omit" else None)
        return out

    if not wide_parts:
        out = pd.DataFrame(columns=["Ensembl_Gene_ID", "Symbol"])
    else:
        out = _merge_cancer_reference_wide_parts(wide_parts)
    _attach_reference_expression_attrs(out, availability if on_missing != "omit" else None)
    return out


def cancer_reference_expression_availability(
    cancer_types: str | Iterable[str] | None = None,
    normalize: str | Iterable[str] = "tpm_clean",
    *,
    sample_qc: str = "pass",
) -> pd.DataFrame:
    """Availability/provenance table for :func:`cancer_reference_expression`.

    The result is one row per requested cancer code (expanding computed aggregate
    requests to member cohorts) and normalization mode. It is intentionally
    gene-independent: use it to decide whether an empty expression frame means an
    empty gene filter or an unavailable upstream artifact.
    """
    modes = _reference_normalize_modes(normalize)
    sample_qc = _validate_sample_qc(sample_qc)
    requests = _reference_expression_requests(cancer_types, modes)
    return _reference_expression_availability_for_requests(requests, modes, sample_qc=sample_qc)


_REFERENCE_NORMALIZE_ALIASES = {
    "clean_tpm": "tpm_clean",
    "tpm_clean": "tpm_clean",
    "clean_tpm_biological": "tpm_clean_biological",
    "tpm_clean_biological": "tpm_clean_biological",
    "biological_clean_tpm": "tpm_clean_biological",
    "tpm_clean_log1p": "tpm_clean_log1p",
    "clean_tpm_log1p": "tpm_clean_log1p",
    "tpm": "tpm_raw",
    "raw_tpm": "tpm_raw",
    "tpm_raw": "tpm_raw",
}

_REFERENCE_NORMALIZE_LABELS = {
    "tpm_clean": "tpm_clean",
    "tpm_clean_biological": "tpm_clean_biological",
    "tpm_clean_log1p": "tpm_clean_log1p",
    "tpm_raw": "tpm_raw",
}

_REFERENCE_WIDE_SUFFIXES = {
    "tpm_clean": "TPM_clean",
    "tpm_clean_biological": "TPM_clean_biological",
    "tpm_clean_log1p": "TPM_clean_log1p",
    "tpm_raw": "TPM_raw",
}

_REFERENCE_PROVENANCE_COLUMNS = [
    "source_cohort",
    "source_type",
    "source_unit",
    "source_scale_class",
    "linear_tpm_comparable",
    "reference_method",
    "sample_qc",
    "data_version",
    "source_matrix_version",
]

_REFERENCE_REQUEST_COLUMNS = [
    "requested_code",
    "request_kind",
    "available",
    "missing_reason",
]


def _reference_expression_requests(
    cancer_types: str | Iterable[str] | None, modes: list[str]
) -> list[dict[str, str]]:
    """Resolved request rows while preserving aggregate-vs-direct intent."""
    if cancer_types is None:
        return [
            {"requested_code": code, "cancer_code": code, "request_kind": "default_available"}
            for code in _reference_available_codes_for_modes(modes)
        ]
    raw_values = [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
    aggregates = cohort_aggregates()
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_values:
        resolved = resolve_cancer_type(raw)
        members = aggregates.get(str(raw)) or aggregates.get(resolved)
        if members:
            expanded = [(member, "aggregate_member") for member in members]
        else:
            expanded = [(resolved, "direct")]
        for code, kind in expanded:
            key = (resolved, code)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"requested_code": resolved, "cancer_code": code, "request_kind": kind})
    return rows


def _reference_available_codes_for_modes(modes: list[str]) -> list[str]:
    out: set[str] = set()
    if any(mode in {"tpm_clean", "tpm_clean_biological", "tpm_clean_log1p"} for mode in modes):
        out.update(available_percentile_cohorts())
    if "tpm_raw" in modes:
        out.update(source_matrices.available_cohorts())
    return sorted(out)


def _reference_expression_availability_for_requests(
    requests: list[dict[str, str]], modes: list[str], *, sample_qc: str
) -> pd.DataFrame:
    percentile_available = set(available_percentile_cohorts())
    source_matrix_available = set(source_matrices.available_cohorts())
    rows: list[dict] = []
    for request in requests:
        code = request["cancer_code"]
        for mode in modes:
            available, missing_reason = _reference_mode_availability(
                code, mode, percentile_available, source_matrix_available
            )
            method = _reference_expected_method(mode)
            row = {
                "requested_code": request["requested_code"],
                "cancer_code": code,
                "request_kind": request["request_kind"],
                "normalization": _REFERENCE_NORMALIZE_LABELS[mode],
                "available": bool(available),
                "missing_reason": "" if available else missing_reason,
                "reference_method": method,
                "artifact_schema_version": REFERENCE_EXPRESSION_SCHEMA_VERSION,
                "data_version": DATA_VERSION,
                "source_matrix_version": SOURCE_MATRIX_VERSION,
            }
            row.update(_reference_expression_provenance(code, mode, method, sample_qc=sample_qc))
            rows.append(row)
    columns = [
        "requested_code",
        "cancer_code",
        "request_kind",
        "normalization",
        "available",
        "missing_reason",
        "source_cohort",
        "source_type",
        "source_unit",
        "source_scale_class",
        "linear_tpm_comparable",
        "reference_method",
        "sample_qc",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out.attrs["artifact_schema_version"] = REFERENCE_EXPRESSION_SCHEMA_VERSION
    out.attrs["data_version"] = DATA_VERSION
    out.attrs["source_matrix_version"] = SOURCE_MATRIX_VERSION
    out.attrs["issues"] = ["#207"]
    return out


def _reference_mode_availability(
    code: str,
    mode: str,
    percentile_available: set[str],
    source_matrix_available: set[str],
) -> tuple[bool, str]:
    if mode in {"tpm_clean", "tpm_clean_biological", "tpm_clean_log1p"}:
        return (True, "") if code in percentile_available else (False, "no_percentile_artifact")
    if mode == "tpm_raw":
        return (True, "") if code in source_matrix_available else (False, "no_source_matrix")
    raise AssertionError(f"unhandled reference normalize mode: {mode}")


def _reference_expected_method(mode: str) -> str:
    if mode in {"tpm_clean", "tpm_clean_biological"}:
        return "percentile_shard"
    if mode == "tpm_clean_log1p":
        return "percentile_shard_log1p"
    if mode == "tpm_raw":
        return "source_matrix_stats"
    raise AssertionError(f"unhandled reference normalize mode: {mode}")


def _attach_reference_expression_attrs(df: pd.DataFrame, availability: pd.DataFrame | None) -> None:
    df.attrs["artifact_schema_version"] = REFERENCE_EXPRESSION_SCHEMA_VERSION
    df.attrs["data_version"] = DATA_VERSION
    df.attrs["source_matrix_version"] = SOURCE_MATRIX_VERSION
    if availability is None:
        return
    missing = availability.loc[~availability["available"]]
    df.attrs["availability"] = availability.to_dict("records")
    df.attrs["missing_requests"] = missing.to_dict("records")


def _reference_normalize_modes(normalize: str | Iterable[str]) -> list[str]:
    if isinstance(normalize, str):
        requested = [normalize]
    else:
        requested = list(normalize)
    modes: list[str] = []
    bad: list[str] = []
    for mode in requested:
        key = str(mode).lower()
        resolved = _REFERENCE_NORMALIZE_ALIASES.get(key)
        if resolved is None:
            bad.append(str(mode))
        elif resolved not in modes:
            modes.append(resolved)
    if bad:
        supported = ", ".join(sorted(_REFERENCE_NORMALIZE_ALIASES))
        raise ValueError(f"unsupported reference normalize mode(s): {bad}; supported: {supported}")
    return modes


def _reference_long_columns(include_provenance: bool, include_request_metadata: bool) -> list[str]:
    cols = ["Ensembl_Gene_ID", "Symbol", "cancer_code", "normalization"]
    if include_request_metadata:
        cols += _REFERENCE_REQUEST_COLUMNS
    if include_provenance:
        cols += _REFERENCE_PROVENANCE_COLUMNS
    return [*cols, "expression", "q1", "q3"]


def _reference_expression_frame(
    code: str, mode: str, *, auto_fetch: bool, sample_qc: str
) -> tuple[pd.DataFrame, str]:
    if mode in {"tpm_clean", "tpm_clean_biological"}:
        return cohort_gene_percentiles(code, as_tpm=True, auto_fetch=auto_fetch), "percentile_shard"
    if mode == "tpm_clean_log1p":
        return (
            cohort_gene_percentiles(code, as_tpm=False, auto_fetch=auto_fetch),
            "percentile_shard_log1p",
        )
    if mode == "tpm_raw":
        return (
            cohort_stats(code, normalize="tpm_raw", auto_fetch=auto_fetch, sample_qc=sample_qc),
            "source_matrix_stats",
        )
    raise AssertionError(f"unhandled reference normalize mode: {mode}")


def _reference_sample_qc_label(mode: str, sample_qc: str) -> str:
    return sample_qc if mode == "tpm_raw" else "artifact"


def _reference_expression_provenance(code: str, mode: str, method: str, *, sample_qc: str) -> dict:
    meta = _selected_expression_source_metadata(code)
    return {
        "source_cohort": meta["source_cohort"] or code,
        "source_type": meta["source_type"],
        "source_unit": meta["unit"],
        "source_scale_class": meta["source_scale_class"],
        "linear_tpm_comparable": bool(meta["linear_tpm_comparable"]),
        "reference_method": method,
        "sample_qc": _reference_sample_qc_label(mode, sample_qc),
        "data_version": DATA_VERSION,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
    }


def _merge_cancer_reference_wide_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge cohort reference vectors by canonical ENSG only.

    Cohorts can carry different release-era symbols for the same canonical gene.
    Joining on ``(Ensembl_Gene_ID, Symbol)`` fragments those loci into sparse rows.
    """
    if not parts:
        return pd.DataFrame(columns=["Ensembl_Gene_ID", "Symbol"])

    value_cols: list[str] = []
    symbols: dict[str, Counter] = defaultdict(Counter)
    keyed_parts: list[pd.DataFrame] = []
    for part in parts:
        value_cols.extend(c for c in part.columns if c not in {"Ensembl_Gene_ID", "Symbol"})
        for gid, symbol in zip(part["Ensembl_Gene_ID"].astype(str), part["Symbol"].astype(str)):
            symbols[gid][symbol] += 1
        keyed = part.drop(columns=["Symbol"], errors="ignore")
        if keyed["Ensembl_Gene_ID"].duplicated().any():
            keyed = keyed.groupby("Ensembl_Gene_ID", sort=False).first().reset_index()
        keyed_parts.append(keyed)

    alias_symbols = ensembl_id_alias_symbols()

    def _symbol(gid: str) -> str:
        auth = alias_symbols.get(gid)
        if auth:
            return auth
        named = Counter({s: c for s, c in symbols.get(gid, {}).items() if s and s != gid})
        return named.most_common(1)[0][0] if named else gid

    out = keyed_parts[0]
    for part in keyed_parts[1:]:
        out = out.merge(part, on="Ensembl_Gene_ID", how="outer")
    out = out.sort_values("Ensembl_Gene_ID").reset_index(drop=True)
    out.insert(1, "Symbol", out["Ensembl_Gene_ID"].astype(str).map(_symbol))
    return out[["Ensembl_Gene_ID", "Symbol", *value_cols]]


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
    sample_qc: str = "pass",
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
    id columns plus one column per statistic. ``sample_qc`` defaults to ``"pass"``
    for live summaries; use ``"all"`` to include every source-matrix sample."""
    df = per_sample_expression(
        cancer_type,
        normalize=normalize,
        auto_fetch=auto_fetch,
        proteoform=proteoform,
        scope=scope,
        sample_qc=sample_qc,
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
    representative_id_style: str = "pirlygenes",
) -> pd.DataFrame:
    """Representative real per-sample expression vectors per cohort.

    The packaged cohort references are per-cohort aggregates; this returns a
    bounded set of real joint per-sample vectors per cohort — medoids spanning
    the within-cohort variation — in the same ``clean TPM`` basis.

    ``cancer_types`` accepts a code, alias, or iterable; a computed-aggregate
    code expands to its member subtypes; ``None`` returns every cohort that ships
    representatives. ``k`` keeps at most the first ``k`` reps per cohort.
    ``format`` is ``"wide"`` (genes × reps) or ``"long"``. Public representative
    IDs default to pirlygenes-compatible ``CODE_rep01`` columns/values; pass
    ``representative_id_style="internal"`` for the shard/provenance IDs
    (``CODE__rep1``). Long output can attach representative-level provenance:
    source cohort/project/sample, selection rank, selection method/basis, and
    package/data schema versions.
    """
    _validate_representative_id_style(representative_id_style)
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
        display_cols = [
            _representative_id_for_style(c, style=representative_id_style) for c in rep_cols
        ]
        mat = shard[rep_cols].set_axis(gid.to_numpy(), axis=0).set_axis(display_cols, axis=1)
        if format == "wide":
            wide_parts.append(mat)
        else:
            internal_mat = shard[rep_cols].set_axis(gid.to_numpy(), axis=0)
            melted = internal_mat.reset_index(names="Ensembl_Gene_ID").melt(
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
            out = pd.DataFrame(columns=base)
            out.attrs.update(
                _representative_attrs(
                    codes=codes,
                    normalize=normalize,
                    format=format,
                    k=k,
                    representative_id_style=representative_id_style,
                )
            )
            return out
        combined = pd.concat(wide_parts, axis=1, join="outer").sort_index()
        if combined.index.has_duplicates:  # belt-and-suspenders: one row per gene id
            combined = combined.groupby(level=0).first()
        out = combined.reset_index(names="Ensembl_Gene_ID")
        out.insert(1, "Symbol", out["Ensembl_Gene_ID"].map(_canonical_symbol))
        out.attrs.update(
            _representative_attrs(
                codes=codes,
                normalize=normalize,
                format=format,
                k=k,
                representative_id_style=representative_id_style,
            )
        )
        return out

    if not long_parts:
        out = _representative_empty_frame(include_provenance=include_provenance)
        out.attrs.update(
            _representative_attrs(
                codes=codes,
                normalize=normalize,
                format=format,
                k=k,
                representative_id_style=representative_id_style,
            )
        )
        return out
    long = pd.concat(long_parts, ignore_index=True)
    long.insert(1, "Symbol", long["Ensembl_Gene_ID"].map(_canonical_symbol))
    if include_provenance:
        long = _attach_representative_provenance(long, root)
    long["representative_id"] = long["representative_id"].map(
        lambda x: _representative_id_for_style(x, style=representative_id_style)
    )
    long.attrs.update(
        _representative_attrs(
            codes=codes,
            normalize=normalize,
            format=format,
            k=k,
            representative_id_style=representative_id_style,
        )
    )
    return long


def _biological_per_sample(
    code, *, proteoform: bool, auto_fetch: bool, scope: str = "cta", sample_qc: str = "pass"
) -> pd.DataFrame:
    """Clean-TPM per-sample matrix with technical/censored genes dropped — the
    biological view the summary artifacts are built on — collapsed to proteoform level
    (in ``scope``) when requested. The runtime input to the percentile / within-sample
    build cores, so a summary can be recomputed on the fly from the per-sample matrix
    (no shard). ``scope`` is ignored when ``proteoform`` is False."""
    from .gene_families import clean_tpm_censored_gene_ids

    clean = per_sample_expression(
        code, normalize="tpm_clean", auto_fetch=auto_fetch, sample_qc=sample_qc
    )
    censored = clean_tpm_censored_gene_ids()
    unversioned = clean["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    bio = clean[~unversioned.isin(censored)].reset_index(drop=True)
    if proteoform:
        from .proteoforms import collapse_to_proteoforms

        bio = collapse_to_proteoforms(bio, scope=scope, sample_cols=sample_columns(bio))
    return bio


def _read_shard_or_recompute(
    dataset: ShardDataset,
    code: str,
    *,
    proteoform: bool,
    auto_fetch: bool,
    scope: str = "cta",
    sample_qc: str = "pass",
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
            code,
            proteoform=proteoform,
            auto_fetch=auto_fetch,
            scope=scope,
            sample_qc=sample_qc,
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
    include_provenance: bool = False,
    on_missing: str = "raise",
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

    ``include_provenance=True`` appends stable artifact/provenance columns. Missing
    cohorts still raise by default; pass ``on_missing="empty"`` to return an
    empty schema-stable frame with ``attrs["missing_reason"]`` instead.

    With ``proteoform=True``, the vector is one row per proteoform key
    (``proteoform_key``/``Symbol`` carry the collapsed identity), identical-protein
    members summed **before** the percentiles are computed (``scope`` ``"cta"``/``"genome"``,
    ignored when ``proteoform`` is False).

    The shipped percentile **shard** can't be converted across proteoform scopes (you
    can't sum already-computed percentiles), so when no matching shard is present the
    vector is **recomputed on the fly** from the per-sample matrix via the same build
    core. That needs the cohort's per-sample matrix cached (pass ``auto_fetch=True`` to
    download it); otherwise a clear error.
    """
    if on_missing not in ("raise", "empty"):
        raise ValueError("on_missing must be 'raise' or 'empty'")

    code = resolve_cancer_type(cancer_type)
    try:
        df = _read_shard_or_recompute(
            _PERCENTILES, code, proteoform=proteoform, auto_fetch=auto_fetch, scope=scope
        )
        source = "shard_or_recomputed"
    except ValueError as e:
        if on_missing != "empty" or "per-sample matrix isn't cached" not in str(e):
            raise
        return _empty_percentile_frame(
            code=code,
            as_tpm=as_tpm,
            proteoform=proteoform,
            scope=scope,
            include_provenance=include_provenance,
            missing_reason=str(e),
        )
    bp_cols = sample_columns(df)
    df[bp_cols] = df[bp_cols].astype("float32")
    if as_tpm:
        df[bp_cols] = np.expm1(df[bp_cols])
    if include_provenance:
        df = _attach_percentile_provenance(df, code=code, as_tpm=as_tpm)
    df.attrs.update(
        _percentile_attrs(
            code=code,
            as_tpm=as_tpm,
            proteoform=proteoform,
            scope=scope,
            source=source,
        )
    )
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
    per-sample matrix via the same build core. Recompute needs the cohort's per-sample
    matrix cached (pass ``auto_fetch=True`` to download it), else a clear error.
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
    cancer_type, *, normalize: str = "tpm_clean", auto_fetch: bool = True, sample_qc: str = "all"
) -> pd.DataFrame:
    """Gene-level per-sample **TPM values** (one row per Ensembl gene). Proteoform
    counterpart: :func:`proteoform_per_sample_expression`."""
    return per_sample_expression(
        cancer_type, normalize=normalize, auto_fetch=auto_fetch, sample_qc=sample_qc
    )


def proteoform_per_sample_expression(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    scope: str = "cta",
    sample_qc: str = "all",
) -> pd.DataFrame:
    """Proteoform-level per-sample **TPM values** — identical-protein paralogs summed
    per sample. Gene-level counterpart: :func:`gene_per_sample_expression`."""
    return per_sample_expression(
        cancer_type,
        normalize=normalize,
        auto_fetch=auto_fetch,
        proteoform=True,
        scope=scope,
        sample_qc=sample_qc,
    )


def gene_cohort_mean_expression(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    statistic: str = "mean",
    auto_fetch: bool = True,
    sample_qc: str = "pass",
) -> pd.DataFrame:
    """Gene-level across-patient **TPM** summary. Proteoform counterpart:
    :func:`proteoform_cohort_mean_expression`."""
    return cohort_mean_expression(
        cancer_type,
        normalize=normalize,
        statistic=statistic,
        auto_fetch=auto_fetch,
        sample_qc=sample_qc,
    )


def proteoform_cohort_mean_expression(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    statistic: str = "mean",
    auto_fetch: bool = True,
    scope: str = "cta",
    sample_qc: str = "pass",
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
        sample_qc=sample_qc,
    )


def gene_cohort_stats(
    cancer_type, *, normalize: str = "tpm_clean", auto_fetch: bool = True, sample_qc: str = "pass"
) -> pd.DataFrame:
    """Gene-level per-gene cohort **summary statistics** (mean/std + the percentile ladder
    min/p1/p5/p10/p15/p20/p25/p50/p75/p80/p85/p90/p95/p99/max). Proteoform counterpart:
    :func:`proteoform_cohort_stats`."""
    return cohort_stats(
        cancer_type, normalize=normalize, auto_fetch=auto_fetch, sample_qc=sample_qc
    )


def proteoform_cohort_stats(
    cancer_type,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    scope: str = "cta",
    sample_qc: str = "pass",
) -> pd.DataFrame:
    """Proteoform-level per-gene cohort **summary statistics**. Gene-level counterpart:
    :func:`gene_cohort_stats`."""
    return cohort_stats(
        cancer_type,
        normalize=normalize,
        auto_fetch=auto_fetch,
        proteoform=True,
        scope=scope,
        sample_qc=sample_qc,
    )


def gene_pooled_cohort_stats(
    cancer_types,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    min_cohorts: int = 1,
    sample_qc: str = "pass",
) -> pd.DataFrame:
    """Gene-level heterogeneity-safe cross-cohort pool. Proteoform counterpart:
    :func:`proteoform_pooled_cohort_stats`. (Alias of :func:`pooled_cohort_stats`.)"""
    return pooled_cohort_stats(
        cancer_types,
        normalize=normalize,
        auto_fetch=auto_fetch,
        min_cohorts=min_cohorts,
        sample_qc=sample_qc,
    )


def proteoform_pooled_cohort_stats(
    cancer_types,
    *,
    normalize: str = "tpm_clean",
    auto_fetch: bool = True,
    scope: str = "cta",
    min_cohorts: int = 1,
    sample_qc: str = "pass",
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
        sample_qc=sample_qc,
    )


def gene_cohort_percentiles(
    cancer_type,
    *,
    as_tpm: bool = True,
    auto_fetch: bool = False,
    include_provenance: bool = False,
    on_missing: str = "raise",
) -> pd.DataFrame:
    """Gene-level per-cohort **percentile vectors**. Proteoform counterpart:
    :func:`proteoform_cohort_percentiles`. (Alias of :func:`cohort_gene_percentiles`.)"""
    return cohort_gene_percentiles(
        cancer_type,
        as_tpm=as_tpm,
        auto_fetch=auto_fetch,
        include_provenance=include_provenance,
        on_missing=on_missing,
    )


def proteoform_cohort_percentiles(
    cancer_type,
    *,
    as_tpm: bool = True,
    scope: str = "cta",
    auto_fetch: bool = True,
    include_provenance: bool = False,
    on_missing: str = "raise",
) -> pd.DataFrame:
    """Proteoform-level per-cohort **percentile vectors** (members summed before
    ranking, ``scope`` ``"cta"``/``"genome"``). Gene-level counterpart:
    :func:`gene_cohort_percentiles`. No proteoform shard ships yet, so this **always**
    recomputes from the per-sample matrix — hence ``auto_fetch`` defaults to ``True``
    here (unlike the gene variant, which reads a shipped shard); pass ``False`` to
    require the matrix already be cached."""
    return cohort_gene_percentiles(
        cancer_type,
        as_tpm=as_tpm,
        proteoform=True,
        scope=scope,
        auto_fetch=auto_fetch,
        include_provenance=include_provenance,
        on_missing=on_missing,
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
    to_tpm: bool | None = None,
    column_style: str | None = None,
) -> pd.DataFrame:
    """Wide pan-cancer reference: each gene's expression across **50 HPA normal
    tissues** and **33 TCGA tumor cohorts**, tumor and normal side by side in one
    frame — the combined companion to the per-cohort accessors above.

    Columns are entity-first: ``<tissue>_nTPM_raw`` for HPA normal tissues,
    ``<CODE>_FPKM_raw`` for TCGA source/provenance values, and ``<CODE>_TPM_raw``
    for deterministic FPKM→TPM tumor companions.
    The default ``normalize="tpm_clean"`` (alias ``"clean_tpm"``) appends
    clean TPM companions named ``<entity>_<measure>_clean``.
    ``normalize="housekeeping"`` / ``"hk"`` appends ``*_hk`` columns, and
    ``"percentile"`` appends ``*_percentile`` columns. ``"tpm_log1p"`` and
    ``"tpm_clean_log1p"`` append natural-log companions over the raw and clean
    TPM/nTPM values, respectively. Pass ``normalize=None`` or ``"tpm"`` for the
    raw TPM/nTPM companions only.

    ``column_style="pirlygenes"`` returns the same data with pirlygenes-style
    unsuffixed analysis names (for example ``LUAD_FPKM``, ``LUAD_TPM``, and
    ``liver_nTPM``). The legacy ``to_tpm`` keyword is accepted as a compatibility
    alias; when used without an explicit ``column_style`` it selects the
    pirlygenes-style output and, for the default normalization, maps to
    ``normalize="tpm"``.

    ``genes`` filters to the given Ensembl gene ids (version-insensitive) or
    symbols; ``None`` returns the full matrix. The FPKM→TPM conversion runs over
    **all** genes before any filtering, so a filtered slice still carries the
    cohort-wide TPM scaling."""
    if to_tpm is not None:
        if normalize == "tpm_clean":
            normalize = "tpm" if to_tpm else None
        if column_style is None:
            column_style = "pirlygenes"
    column_style = _pan_cancer_column_style(column_style)

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
    raw_parts: list[pd.DataFrame] = []
    ntpm_cols = [c for c in df.columns if c.startswith("nTPM_")]
    ntpm_data: dict[str, pd.Series] = {}
    for col in ntpm_cols:
        entity = col[len("nTPM_") :]
        target = f"{entity}_nTPM_raw"
        ntpm_data[target] = pd.to_numeric(df[col], errors="coerce")
        raw_cols.append(target)
    if ntpm_data:
        raw_parts.append(pd.DataFrame(ntpm_data, index=df.index))

    fpkm_cols = [c for c in df.columns if c.startswith("FPKM_")]
    if fpkm_cols:
        from .normalization import fpkm_to_tpm

        converted, _ = fpkm_to_tpm(df[id_cols + fpkm_cols], value_cols=fpkm_cols)
        fpkm_data: dict[str, pd.Series] = {}
        for col in fpkm_cols:
            entity = col[len("FPKM_") :]
            provenance_col = f"{entity}_FPKM_raw"
            fpkm_data[provenance_col] = pd.to_numeric(df[col], errors="coerce")
            target = f"{entity}_TPM_raw"
            fpkm_data[target] = converted[col]
            raw_cols.append(target)
        raw_parts.append(pd.DataFrame(fpkm_data, index=df.index))

    tpm_cols = [c for c in df.columns if c.startswith("TPM_")]
    tpm_data: dict[str, pd.Series] = {}
    for col in tpm_cols:
        entity = col[len("TPM_") :]
        target = f"{entity}_TPM_raw"
        tpm_data[target] = pd.to_numeric(df[col], errors="coerce")
        raw_cols.append(target)
    if tpm_data:
        raw_parts.append(pd.DataFrame(tpm_data, index=df.index))
    if raw_parts:
        out = pd.concat([out, *raw_parts], axis=1)

    modes = _pan_cancer_normalize_modes(normalize)
    clean_cols: list[str] = []
    if modes & {"tpm_clean", "housekeeping", "hk", "percentile", "tpm_clean_log1p"}:
        clean = clean_tpm(out[raw_cols], gene_table=out[id_cols])
        clean_data: dict[str, pd.Series] = {}
        for col in raw_cols:
            target = col[: -len("_raw")] + "_clean"
            clean_data[target] = clean[col]
            clean_cols.append(target)
        out = pd.concat([out, pd.DataFrame(clean_data, index=out.index)], axis=1)

    if modes & {"housekeeping", "hk"}:
        hk_input_cols = clean_cols or raw_cols
        hk_input = out[id_cols + hk_input_cols].copy()
        panel_ids = _clean_tpm_housekeeping_panel_ids()
        hk, _ = tpm_to_housekeeping_normalized(
            hk_input,
            value_cols=hk_input_cols,
            panel_ids=panel_ids,
            panel_name="clean_tpm_biological_housekeeping",
            min_panel_detected=_min_housekeeping_detected(panel_ids),
            drop_zero_panel_values=True,
            warn_on_unreliable=True,
        )
        hk_data: dict[str, pd.Series] = {}
        for col in hk_input_cols:
            target = col.rsplit("_", 1)[0] + "_hk"
            hk_data[target] = hk[col]
        out = pd.concat([out, pd.DataFrame(hk_data, index=out.index)], axis=1)

    if "percentile" in modes:
        pct_input_cols = clean_cols or raw_cols
        pct = percentile_rank(out[id_cols + pct_input_cols], value_cols=pct_input_cols)
        pct_data: dict[str, pd.Series] = {}
        for col in pct_input_cols:
            target = col.rsplit("_", 1)[0] + "_percentile"
            pct_data[target] = pct[col]
        out = pd.concat([out, pd.DataFrame(pct_data, index=out.index)], axis=1)

    if "tpm_log1p" in modes:
        log_data: dict[str, np.ndarray] = {}
        for col in raw_cols:
            target = col + "_log1p"
            log_data[target] = np.log1p(out[col].to_numpy(dtype=float))
        out = pd.concat([out, pd.DataFrame(log_data, index=out.index)], axis=1)

    if "tpm_clean_log1p" in modes:
        clean_log_data: dict[str, np.ndarray] = {}
        for col in clean_cols:
            target = col + "_log1p"
            clean_log_data[target] = np.log1p(out[col].to_numpy(dtype=float))
        out = pd.concat([out, pd.DataFrame(clean_log_data, index=out.index)], axis=1)

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
    out = _format_pan_cancer_columns(out, column_style=column_style)
    out.attrs["oncoref"] = {
        "dataset": "pan-cancer-expression",
        "data_version": DATA_VERSION,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "column_style": column_style,
    }
    return out


def _pan_cancer_column_style(column_style: str | None) -> str:
    if column_style is None:
        return "oncoref"
    style = str(column_style).lower()
    aliases = {"default": "oncoref", "entity_first": "oncoref", "legacy": "pirlygenes"}
    style = aliases.get(style, style)
    if style not in {"oncoref", "pirlygenes"}:
        raise ValueError("column_style must be None, 'oncoref', or 'pirlygenes'")
    return style


def _format_pan_cancer_columns(df: pd.DataFrame, *, column_style: str) -> pd.DataFrame:
    if column_style == "oncoref":
        return df
    rename: dict[str, str] = {}
    for col in df.columns:
        if col.endswith("_raw_log1p"):
            rename[col] = col.replace("_raw_log1p", "_log1p")
        elif col.endswith(("_nTPM_raw", "_FPKM_raw", "_TPM_raw")):
            rename[col] = col[: -len("_raw")]
    return df.rename(columns=rename)


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
    sample_qc: str = "pass",
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
    ``1`` keeps everything). ``sample_qc`` defaults to ``"pass"`` so QC-failed
    sample columns do not affect live pooled summaries. ``normalize`` selects the
    pooling space (linear clean TPM by default; see :func:`per_sample_expression`).
    ``proteoform=True`` pools the reduced proteoform key space (``scope`` ``"cta"``/
    ``"genome"``). An aggregate code (e.g. ``"SARC"``) expands to its member subtypes
    — pooling them is exactly what a rollup cohort means."""
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
            code,
            normalize=normalize,
            auto_fetch=auto_fetch,
            proteoform=proteoform,
            scope=scope,
            sample_qc=sample_qc,
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
