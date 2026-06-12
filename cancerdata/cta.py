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

"""Cancer-testis antigens (CTAs): the gene list + HPA tissue-restriction evidence.

A CTA is a gene whose normal expression is restricted to reproductive tissues
(testis / ovary / placenta) and which reactivates in tumors — a tissue-restriction
call over HPA normal-tissue expression, i.e. cancer **reference data**. cancerdata
owns the definition: the bundled ``cancer-testis-antigens.csv`` carries the
candidate list (from 5 source databases) plus the HPA-derived per-tissue
restriction columns and filter flags.

The MS-evidence restriction tiers and peptide/MHC presentation that build on top
of this list are the target-selection layer's domain and are intentionally NOT
here. ``restriction`` and ``restriction_confidence`` in the bundled table are the
**HPA-only** synthesis (protein + RNA modalities; see :func:`synthesize_restriction`)
— no MS contribution — so the values match the data cancerdata owns.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .cta_tissues import HPA_EXPRESSION_FLOOR_NTPM
from .load_dataset import get_data

#: Borderline-but-real CTAs kept in the expressed set despite an HPA
#: ``never_expressed`` flag (low but corroborated testis signal). Unversioned ENSG.
MANUALLY_EXPRESSED_CTA: frozenset[str] = frozenset({"ENSG00000171405"})  # XAGE5

#: Genes present in a source database but excluded from the CTA universe — core
#: histones, placental hCG-beta, alpha-tubulins (not tumor-restricted antigens).
NON_CTA_EXCLUDED_GENE_IDS: frozenset[str] = frozenset(
    {
        "ENSG00000274618",  # H4C6
        "ENSG00000146047",  # H2BC1
        "ENSG00000276410",  # H2BC3
        "ENSG00000124610",  # H1-1
        "ENSG00000213030",  # CGB8 (placental hCG-beta)
        "ENSG00000198033",  # TUBA3C
        "ENSG00000152086",  # TUBA3E
    }
)

_PASSES_FILTERS_COLUMN = "passes_filters"
_LEGACY_FILTERED_COLUMN = "filtered"


def synthesize_restriction(row) -> tuple[str, str]:
    """HPA-only tissue restriction + confidence for a CTA row.

    Best tissue modality (protein > RNA); ``restriction_confidence`` is HIGH /
    MODERATE / LOW / NO_DATA from per-modality agreement and HPA reliability.
    This is the **HPA-only** synthesis — the MS-evidence contribution that the
    target-selection layer adds is intentionally excluded, so the value's
    provenance matches the data cancerdata owns.
    """
    protein_r = str(row.get("protein_restriction", "") or "")
    protein_rel = str(row.get("protein_reliability", "") or "")
    rna_r = str(row.get("rna_restriction", "") or "")
    rna_level = str(row.get("rna_restriction_level", "") or "")

    protein_has_data = bool(protein_r) and protein_r != "NO_DATA"
    rna_has_data = bool(rna_r) and rna_r != "NO_DATA"
    if protein_has_data:
        tissue = protein_r
    elif rna_has_data:
        tissue = rna_r
    else:
        tissue = "NO_DATA"

    score = 0.0
    sources = 0
    if protein_has_data:
        sources += 1
        score += 1.0
        if protein_rel in ("Enhanced", "Supported"):
            score += 0.5
    if rna_has_data:
        sources += 1
        rna_agrees = rna_r == tissue or (rna_r == "REPRODUCTIVE" and protein_has_data)
        if rna_agrees:
            score += 1.0
            if rna_level == "STRICT":
                score += 0.5

    if sources == 0:
        confidence = "NO_DATA"
    elif score / sources >= 1.2:
        confidence = "HIGH"
    elif score / sources >= 0.8:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # Cap HIGH when the only evidence is RNA below the expression floor: the scorer
    # credits any STRICT RNA equally, so a gene at ~1-2 nTPM (never_expressed, no
    # protein) would otherwise earn HIGH from near-noise RNA. Genes with protein
    # evidence are untouched. (HPA-only port of tsarina#114; no MS term here.)
    if confidence == "HIGH" and not protein_has_data:
        rna_max = pd.to_numeric(row.get("rna_max_ntpm"), errors="coerce")
        if pd.notna(rna_max) and rna_max < HPA_EXPRESSION_FLOOR_NTPM:
            confidence = "MODERATE"
    return tissue, confidence


@lru_cache(maxsize=1)
def _cta_frame() -> pd.DataFrame:
    """Cached CTA table with the non-CTA excluded genes dropped. Internal,
    read-only — do not mutate; public callers get a copy via cta_dataframe()."""
    df = get_data("cancer-testis-antigens", copy=False)
    if "Ensembl_Gene_ID" in df.columns and NON_CTA_EXCLUDED_GENE_IDS:
        unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
        df = df[~unversioned.isin(NON_CTA_EXCLUDED_GENE_IDS)].reset_index(drop=True)
    return df


def cta_dataframe() -> pd.DataFrame:
    """Full CTA evidence table (one row per candidate), with the non-CTA
    excluded genes (histones, etc.) dropped. Returns a defensive copy."""
    return _cta_frame().copy()


#: Alias matching the target-selection layer's public name.
def CTA_evidence() -> pd.DataFrame:
    """The CTA evidence DataFrame (alias of :func:`cta_dataframe`)."""
    return cta_dataframe()


def passes_filters_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for rows passing the HPA curation filter (reproductive
    restriction). Accepts the legacy ``filtered`` column too."""
    if _PASSES_FILTERS_COLUMN in df.columns:
        values = df[_PASSES_FILTERS_COLUMN]
    elif _LEGACY_FILTERED_COLUMN in df.columns:
        values = df[_LEGACY_FILTERED_COLUMN]
    else:
        return pd.Series(True, index=df.index)
    return values.astype(str).str.lower() == "true"


def _cta_by_column(
    column: str,
    *,
    filtered_only: bool = False,
    exclude_never_expressed: bool = False,
) -> set[str]:
    df = _cta_frame()
    mask = pd.Series(True, index=df.index)
    if filtered_only:
        mask = passes_filters_mask(df)
    if exclude_never_expressed and "never_expressed" in df.columns:
        never = df["never_expressed"].astype(str).str.lower() == "true"
        if "Ensembl_Gene_ID" in df.columns:
            rescued = (
                df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0].isin(MANUALLY_EXPRESSED_CTA)
            )
            never = never & ~rescued
        mask = mask & ~never
    subset = df[mask]
    result: set[str] = set()
    if column in subset.columns:
        for x in subset[column]:
            if isinstance(x, str):
                result.update(xi.strip() for xi in x.split(";"))
    return result


def _all_by_column(column: str) -> set[str]:
    df = _cta_frame()
    result: set[str] = set()
    if column in df.columns:
        for x in df[column]:
            if isinstance(x, str):
                result.update(xi.strip() for xi in x.split(";"))
    return result


def CTA_gene_names() -> set[str]:
    """CTA gene symbols that pass the HPA filter AND are expressed (>= 2 nTPM
    somewhere) — the recommended default set."""
    return _cta_by_column("Symbol", filtered_only=True, exclude_never_expressed=True)


def CTA_gene_ids() -> set[str]:
    """CTA Ensembl gene IDs that pass the HPA filter AND are expressed."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True, exclude_never_expressed=True)


def CTA_filtered_gene_names() -> set[str]:
    """All CTA symbols passing the HPA filter (including never-expressed)."""
    return _cta_by_column("Symbol", filtered_only=True)


def CTA_filtered_gene_ids() -> set[str]:
    """All CTA Ensembl gene IDs passing the HPA filter (including never-expressed)."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True)


def CTA_never_expressed_gene_names() -> set[str]:
    """Filter-passing CTAs with no meaningful HPA expression (no protein, max RNA < 2)."""
    return CTA_filtered_gene_names() - CTA_gene_names()


def CTA_unfiltered_gene_names() -> set[str]:
    """Every candidate CTA symbol across all source databases (the full universe)."""
    return _all_by_column("Symbol")


def CTA_unfiltered_gene_ids() -> set[str]:
    """Every candidate CTA Ensembl gene ID across all source databases."""
    return _all_by_column("Ensembl_Gene_ID")


def CTA_excluded_gene_names() -> set[str]:
    """Candidate CTAs that FAIL the reproductive-restriction filter (somatic leakage)."""
    return CTA_unfiltered_gene_names() - CTA_filtered_gene_names()


def CTA_gene_id_to_name() -> dict[str, str]:
    """``{Ensembl_Gene_ID (unversioned): Symbol}`` over the expressed CTA set."""
    df = _cta_frame()
    ids = CTA_gene_ids()
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        gid = str(row.get("Ensembl_Gene_ID", "")).split(".")[0]
        if gid in ids:
            out[gid] = str(row.get("Symbol", gid))
    return out
