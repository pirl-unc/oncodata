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

"""Expression normalization — putting per-cohort RNA-seq in a comparable space.

The headline transform is **clean TPM** (:func:`clean_tpm`): a two-compartment
renormalization that forces the technical/QC compartment (mtDNA, rRNA, the
polyA-bias lncRNAs MALAT1/NEAT1, and — by default — ribosomal proteins) to a fixed
fraction of the per-sample budget and the biological compartment to the rest,
renormalizing within each. The variable, pipeline-driven technical fraction no
longer inflates real genes, so biological clean TPM is directly comparable across
samples and sources. Plus the supporting helpers: drop technical genes for a
biology-only view, housekeeping/log/rank transforms.

Censored-gene and gene-family lists come from :mod:`cancerdata.gene_families`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import gene_families

_ID_COLUMNS = ("Ensembl_Gene_ID", "Symbol")


def _value_cols(df: pd.DataFrame, value_cols=None) -> list[str]:
    if value_cols is not None:
        return list(value_cols)
    return [c for c in df.columns if c not in _ID_COLUMNS]


def _unversioned(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(".").str[0]


def _censored_mask(gene_table: pd.DataFrame, *, exclude_ribosomal_proteins: bool) -> pd.Series:
    """Boolean (row-aligned to ``gene_table``) for the clean-TPM censored genes."""
    if "Ensembl_Gene_ID" not in gene_table.columns:
        raise ValueError("clean TPM needs an 'Ensembl_Gene_ID' column (censoring is ENSG-keyed)")
    ids = gene_families.clean_tpm_censored_gene_ids(
        include_ribosomal_proteins=exclude_ribosomal_proteins
    )
    return _unversioned(gene_table["Ensembl_Gene_ID"]).isin(ids)


def clean_tpm(
    values: pd.DataFrame,
    gene_table: pd.DataFrame | None = None,
    *,
    removable: pd.Series | None = None,
    exclude_ribosomal_proteins: bool = True,
    technical_fraction: float = 0.25,
) -> pd.DataFrame:
    """Two-compartment **clean TPM** on a gene×sample matrix.

    ``values`` is genes (rows) × samples (cols). Provide a ``gene_table``
    (``Symbol`` + ``Ensembl_Gene_ID``, row-aligned to ``values``) so the censored
    (technical/ribosomal) rows can be identified, or an explicit boolean
    ``removable`` mask **positionally aligned to ``values``'s rows** (it is applied
    by position, not by index label).

    The technical compartment is forced to ``technical_fraction`` of the 1e6
    budget (default 25%) and the biological compartment to the remaining
    ``1 - technical_fraction`` (75%), each renormalized internally. So every sample
    is the same fraction technical, the biological compartment lands on a constant
    750k budget, and biological clean TPM is cross-sample / cross-source comparable
    (a sample with no technical mass keeps technical at 0; biology still fills 75%).
    """
    if not 0.0 < technical_fraction < 1.0:
        raise ValueError("technical_fraction must be in (0, 1)")
    if removable is None:
        if gene_table is None:
            raise ValueError("clean_tpm needs either gene_table or removable")
        removable = _censored_mask(
            gene_table, exclude_ribosomal_proteins=exclude_ribosomal_proteins
        )
    rem = removable.to_numpy()

    tech_budget = technical_fraction * 1_000_000.0
    bio_budget = (1.0 - technical_fraction) * 1_000_000.0
    tech_sum = values.loc[rem].sum(axis=0)
    bio_sum = values.loc[~rem].sum(axis=0)

    tscale = pd.Series(0.0, index=values.columns, dtype=float)
    bscale = pd.Series(0.0, index=values.columns, dtype=float)
    tscale.loc[tech_sum > 0] = tech_budget / tech_sum.loc[tech_sum > 0]
    bscale.loc[bio_sum > 0] = bio_budget / bio_sum.loc[bio_sum > 0]

    clean = values.astype(float).copy()
    clean.loc[rem] = values.loc[rem].mul(tscale, axis=1)
    clean.loc[~rem] = values.loc[~rem].mul(bscale, axis=1)
    return clean.fillna(0.0)


def drop_technical_genes(
    df: pd.DataFrame, *, exclude_ribosomal_proteins: bool = True
) -> pd.DataFrame:
    """Biology-only view: drop the clean-TPM censored rows (technical RNA, and by
    default ribosomal proteins). Row order preserved."""
    mask = _censored_mask(df, exclude_ribosomal_proteins=exclude_ribosomal_proteins)
    return df.loc[~mask].reset_index(drop=True)


def filter_technical_rna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the strict technical-RNA loci (mtDNA / NUMT / rRNA / nuclear-retained
    lncRNA) — a lighter filter than :func:`drop_technical_genes` (keeps ribosomal
    proteins). Row order preserved."""
    mask = _unversioned(df["Ensembl_Gene_ID"]).isin(gene_families.technical_rna_gene_ids())
    return df.loc[~mask].reset_index(drop=True)


def normalize_to_housekeeping(df: pd.DataFrame, value_cols=None) -> pd.DataFrame:
    """Rescale each value column by its housekeeping-gene median (unitless: 1.0 =
    baseline). NaN/non-positive housekeeping median -> that column becomes NaN."""
    cols = _value_cols(df, value_cols)
    hk = gene_families.housekeeping_gene_ids()
    hk_rows = _unversioned(df["Ensembl_Gene_ID"]).isin(hk)
    out = df.copy()
    for c in cols:
        med = df.loc[hk_rows, c].median()
        out[c] = df[c] / med if med and med > 0 else np.nan
    return out


def log2_transform(df: pd.DataFrame, value_cols=None, *, pseudocount: float = 1.0) -> pd.DataFrame:
    """``log2(x + pseudocount)`` over the value columns."""
    cols = _value_cols(df, value_cols)
    out = df.copy()
    out[cols] = np.log2(df[cols].to_numpy(dtype=float) + pseudocount)
    return out


def log1p_transform(df: pd.DataFrame, value_cols=None) -> pd.DataFrame:
    """``log1p(x)`` over the value columns."""
    cols = _value_cols(df, value_cols)
    out = df.copy()
    out[cols] = np.log1p(df[cols].to_numpy(dtype=float))
    return out


def percentile_rank(df: pd.DataFrame, value_cols=None) -> pd.DataFrame:
    """Within-sample percentile rank (0–100) per gene — each value column ranked
    independently. Pipeline-robust (rank is preserved across quantifiers where
    absolute TPM is not)."""
    cols = _value_cols(df, value_cols)
    out = df.copy()
    out[cols] = df[cols].rank(axis=0, pct=True) * 100.0
    return out
