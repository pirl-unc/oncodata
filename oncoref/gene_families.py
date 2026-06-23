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

"""Gene-family reference lists for expression normalization.

The curated families behind oncoref's normalization: the technical-RNA loci
whose RNA-seq abundance is library-prep artifact rather than biology (mtDNA, NUMT
pseudogenes, rRNA, nuclear-retained lncRNAs), the ribosomal-protein / histone /
hemoglobin / small-ncRNA families, the housekeeping panel, and the censored-gene
surrogate TPMs. This is the read surface; the clean-TPM engine that consumes them
lives in :mod:`oncoref.normalization`.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .load_dataset import get_data

#: family name -> bundled dataset.
_FAMILIES = {
    "histone": "histone-genes",
    "ribosomal_protein": "ribosomal-protein-genes",
    "ribosomal_protein_pseudogene": "ribosomal-protein-pseudogenes",
    "mitochondrial": "mitochondrial-genes",
    "rrna": "rrna-and-pseudogenes",
    "numt_pseudogene": "numt-pseudogenes",
    "small_noncoding_rna": "small-noncoding-rnas",
    "nuclear_retained_lncrna": "nuclear-retained-lncrnas",
    "hemoglobin": "hemoglobin-genes",
}

#: The gene families that make up "technical RNA" — polyA-protocol artifacts whose
#: abundance reflects library prep, not biology (dropped by ``filter_technical_rna``,
#: and the bulk of the clean-TPM technical compartment). Public so a consumer can build
#: the technical-RNA id set itself without importing a ``_``-prefixed global.
TECHNICAL_RNA_FAMILIES = ("mitochondrial", "numt_pseudogene", "rrna", "nuclear_retained_lncrna")
#: Back-compat private alias (prefer :data:`TECHNICAL_RNA_FAMILIES`).
_TECHNICAL_RNA_FAMILIES = TECHNICAL_RNA_FAMILIES


def gene_families() -> tuple[str, ...]:
    """The available gene-family names."""
    return tuple(_FAMILIES)


def gene_family(name: str) -> pd.DataFrame:
    """The table for one gene family (``Symbol``, ``Ensembl_Gene_ID``, …). Copy."""
    try:
        dataset = _FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown gene family {name!r}; one of {sorted(_FAMILIES)}") from None
    return get_data(dataset).copy()


def _unversioned_ids(df: pd.DataFrame) -> frozenset[str]:
    ids = df["Ensembl_Gene_ID"].dropna().astype(str).str.split(".").str[0]
    return frozenset(ids)


@lru_cache(maxsize=len(_FAMILIES))
def gene_family_ids(name: str) -> frozenset[str]:
    """Unversioned Ensembl IDs of one gene family."""
    if name not in _FAMILIES:
        raise ValueError(f"unknown gene family {name!r}; one of {sorted(_FAMILIES)}")
    return _unversioned_ids(get_data(_FAMILIES[name], copy=False))


@lru_cache(maxsize=1)
def technical_rna_gene_ids() -> frozenset[str]:
    """Union of the technical-RNA family IDs (mtDNA / NUMT / rRNA / nuclear-retained
    lncRNA) — the set ``filter_technical_rna`` drops."""
    out: set[str] = set()
    for fam in _TECHNICAL_RNA_FAMILIES:
        out |= gene_family_ids(fam)
    return frozenset(out)


def housekeeping_genes() -> pd.DataFrame:
    """The housekeeping panel (``Symbol``, ``Ensembl_Gene_ID``, ``Category``, …)."""
    return get_data("housekeeping-genes").copy()


@lru_cache(maxsize=1)
def housekeeping_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs of the housekeeping panel."""
    return _unversioned_ids(get_data("housekeeping-genes", copy=False))


def legacy_qpcr_housekeeping_genes() -> pd.DataFrame:
    """Classic qPCR/reference-gene housekeeping panel.

    This is the historical :func:`housekeeping_genes` table. It intentionally remains
    available, but clean-TPM biological denominators should use
    :func:`clean_tpm_biological_housekeeping_genes` instead because this legacy panel
    includes ribosomal-protein genes that live in clean TPM's 16% ribosomal compartment.
    """
    return housekeeping_genes()


def legacy_qpcr_housekeeping_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs of the classic qPCR/reference-gene panel."""
    return housekeeping_gene_ids()


def _primary_panel_mask(df: pd.DataFrame) -> pd.Series:
    return df["primary_panel"].astype(str).str.lower().isin({"true", "1", "yes"})


def clean_tpm_biological_housekeeping_genes(*, primary_only: bool = True) -> pd.DataFrame:
    """HPA-stable biological housekeeping genes for clean-TPM denominators.

    The bundled table contains all 47 strict HPA whole-tissue candidates that are
    protein-coding, clean-TPM biological genes with ``min_tpm >= 100`` across 50 HPA
    tissues and low tissue coefficient of variation. By default this returns the
    curated 30-gene primary denominator panel. Pass ``primary_only=False`` for the
    full candidate/audit table including HPA provenance and exclusion notes.
    """
    df = get_data("clean-tpm-biological-housekeeping-genes")
    if primary_only:
        df = df[_primary_panel_mask(df)]
    return df.reset_index(drop=True)


@lru_cache(maxsize=2)
def clean_tpm_biological_housekeeping_gene_ids(*, primary_only: bool = True) -> frozenset[str]:
    """Unversioned Ensembl IDs for the clean-TPM biological housekeeping panel.

    Defaults to the 30-gene primary denominator panel. Pass ``primary_only=False``
    to include every strict HPA-stable candidate in the shipped audit table.
    """
    return _unversioned_ids(clean_tpm_biological_housekeeping_genes(primary_only=primary_only))


@lru_cache(maxsize=2)
def clean_tpm_censored_gene_ids(*, include_ribosomal_proteins: bool = True) -> frozenset[str]:
    """Unversioned Ensembl IDs censored by the clean-TPM transform — technical RNA
    plus, by default, ribosomal-protein genes. Pass
    ``include_ribosomal_proteins=False`` for the strict technical-only set.

    **Curated, biology-defined membership — NOT data-derived.** Membership is the technical-
    RNA families + ribosomal proteins: genes that are non-mRNA or library-prep artifacts by
    their molecular nature. Never expand this set from expression variance or abundance —
    cancer-testis antigens are high-variance *by definition* (silent in most samples, high in
    a few, which is exactly what makes them targets), so a variance-based rule would censor the
    very antigens oncoref exists to find. Data (TCGA LUAD/SKCM) is used only to *calibrate*
    the clean-TPM compartment fractions and to *validate completeness* of this list (how the
    missing rRNA genes, incl. 28S, were found)."""
    df = get_data("clean-tpm-censored-genes", copy=False)
    if not include_ribosomal_proteins:
        df = df[df["category"].astype(str) == "technical"]
    return _unversioned_ids(df)


@lru_cache(maxsize=1)
def clean_tpm_ribosomal_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs assigned to clean TPM's 16% ribosomal compartment.

    This is defined by ``clean-tpm-censored-genes.csv`` where
    ``category == "ribosomal_protein"``, not by the broader ribosomal-protein family
    table. That distinction is CTA-safe: broad ribosomal-family members that are absent
    from the censored table, such as ``RPL10L`` / ``ENSG00000165496``, remain biological.
    """
    df = get_data("clean-tpm-censored-genes", copy=False)
    df = df[df["category"].astype(str) == "ribosomal_protein"]
    return _unversioned_ids(df)


@lru_cache(maxsize=1)
def clean_tpm_other_technical_gene_ids() -> frozenset[str]:
    """Unversioned Ensembl IDs assigned to clean TPM's 9% other-technical compartment.

    This is defined by ``clean-tpm-censored-genes.csv`` where
    ``category == "technical"``.
    """
    df = get_data("clean-tpm-censored-genes", copy=False)
    df = df[df["category"].astype(str) == "technical"]
    return _unversioned_ids(df)


@lru_cache(maxsize=1)
def censored_gene_reference_tpm() -> dict[str, float]:
    """``{Symbol: reference_tpm}`` — the fixed surrogate TPM each censored gene holds
    in every cohort (median across the Treehouse PolyA compendium)."""
    df = get_data("censored-gene-reference-tpm", copy=False)
    return {str(s): float(v) for s, v in zip(df["Symbol"], df["reference_tpm"])}
