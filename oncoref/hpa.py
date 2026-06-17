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

"""Loaders over the Human Protein Atlas normal-tissue reference data.

These fetch (once, cached) and parse the HPA tables registered in
:mod:`oncoref.reference_data`:

- ``hpa_rna_consensus`` — per-tissue RNA nTPM (``Gene``, ``Gene name``,
  ``Tissue``, ``nTPM``).
- ``hpa_normal_tissue`` — IHC protein detection (``Gene``, ``Gene name``,
  ``Tissue``, ``Cell type``, ``Level``, ``Reliability``).
- ``hpa_single_cell`` — single-cell-type RNA nTPM (``Gene``, ``Gene name``,
  ``Cell type``, ``nTPM``).

This is the normal-tissue expression evidence behind the cancer-testis-antigen
tissue-restriction definition and protein-level / single-cell comparisons.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache

import pandas as pd

from . import reference_data


def _read_hpa(name: str) -> pd.DataFrame:
    """Load an HPA table, caching a columnar **parquet** copy next to the raw TSV
    so repeated/cold reads are fast and compact (HPA TSVs are large — single-cell
    is tens of MB; parquet is a few-fold smaller and skips re-parsing). The parquet
    cache is best-effort and regenerated if the TSV is re-downloaded."""
    tsv = reference_data.ensure(name)
    parquet = tsv.with_suffix(".parquet")
    if parquet.exists() and parquet.stat().st_mtime >= tsv.stat().st_mtime:
        return pd.read_parquet(parquet)
    df = pd.read_csv(tsv, sep="\t")
    # parquet cache is an optimization, not required — never fail a read over it.
    with contextlib.suppress(Exception):
        df.to_parquet(parquet, index=False)
    return df


@lru_cache(maxsize=1)
def hpa_rna_consensus() -> pd.DataFrame:
    """HPA RNA consensus per-tissue nTPM (downloads HPA v23 on first use)."""
    return _read_hpa("hpa_rna_consensus")


@lru_cache(maxsize=1)
def hpa_normal_tissue() -> pd.DataFrame:
    """HPA IHC protein detection per tissue/cell type (downloads on first use)."""
    return _read_hpa("hpa_normal_tissue")


@lru_cache(maxsize=1)
def hpa_single_cell() -> pd.DataFrame:
    """HPA single-cell-type RNA nTPM (downloads on first use)."""
    return _read_hpa("hpa_single_cell")


def _strip_version(gene_id: str) -> str:
    return str(gene_id).split(".")[0]


def gene_tissue_ntpm(gene_id: str) -> dict[str, float]:
    """``{tissue (lowercased): nTPM}`` of normal RNA expression for one gene
    (unversioned Ensembl ID)."""
    gid = _strip_version(gene_id)
    df = hpa_rna_consensus()
    sub = df[df["Gene"].astype(str).map(_strip_version) == gid]
    return {str(t).strip().lower(): float(v) for t, v in zip(sub["Tissue"], sub["nTPM"])}


def gene_cell_type_ntpm(gene_id: str) -> dict[str, float]:
    """``{cell_type (lowercased): nTPM}`` of single-cell RNA for one gene."""
    gid = _strip_version(gene_id)
    df = hpa_single_cell()
    sub = df[df["Gene"].astype(str).map(_strip_version) == gid]
    return {str(t).strip().lower(): float(v) for t, v in zip(sub["Cell type"], sub["nTPM"])}


def gene_protein_tissues(gene_id: str, *, levels=("Low", "Medium", "High")) -> set[str]:
    """Tissues (lowercased) where the gene has detected IHC protein at one of
    ``levels`` (default Low/Medium/High)."""
    gid = _strip_version(gene_id)
    df = hpa_normal_tissue()
    sub = df[df["Gene"].astype(str).map(_strip_version) == gid]
    sub = sub[sub["Level"].astype(str).isin(levels)]
    return {str(t).strip().lower() for t in sub["Tissue"]}
