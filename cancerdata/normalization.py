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
from .expression_engine import ID_COLUMNS


def _value_cols(df: pd.DataFrame, value_cols=None) -> list[str]:
    if value_cols is not None:
        return list(value_cols)
    return [c for c in df.columns if c not in ID_COLUMNS]


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


# ---------- TPM-scale rescaling (FPKM->TPM, renormalize to 1e6) ----------

#: Column-name conventions for "an expression value column" — TPM/nTPM/FPKM named
#: (mirrors the pirlygenes expression schema), excluding the ``_raw`` provenance
#: copies that must never be rescaled.
_VALUE_COL_PREFIXES = ("TPM", "nTPM_", "FPKM_")
_VALUE_COL_SUFFIXES = (
    "_TPM",
    "_nTPM",
    "_FPKM",
    "_TPM_log1p",
    "_nTPM_log1p",
    "_TPM_clean",
    "_nTPM_clean",
    "_TPM_clean_log1p",
    "_nTPM_clean_log1p",
    "_TPM_hk",
    "_nTPM_hk",
    "_TPM_percentile",
    "_nTPM_percentile",
)
_RAW_VALUE_COL_PREFIXES = ("TPM_raw_", "nTPM_raw_")
_RAW_VALUE_COL_SUFFIXES = ("_TPM_raw", "_nTPM_raw")


def is_expression_value_col(col: object) -> bool:
    """True if ``col`` names an expression value column (TPM/nTPM/FPKM-named),
    excluding the ``_raw`` provenance copies."""
    name = str(col)
    return (name.startswith(_VALUE_COL_PREFIXES) or name.endswith(_VALUE_COL_SUFFIXES)) and not (
        name.startswith(_RAW_VALUE_COL_PREFIXES) or name.endswith(_RAW_VALUE_COL_SUFFIXES)
    )


def renormalize_to_million(df: pd.DataFrame, *, value_cols=None) -> tuple[pd.DataFrame, dict]:
    """Rescale each expression column so its finite sum is exactly 1e6 (the TPM
    convention). Drops nothing — a bare utility, e.g. after :func:`clean_tpm` /
    technical-gene removal if you also want the post-filter total pinned at 1e6.

    Returns ``(rescaled_df, stats)`` where ``stats`` records, per column, the input
    sum and applied scale (a column whose sum is ≤ 0 is left untouched, scale 1.0).
    ``value_cols`` defaults to the TPM/nTPM/FPKM-named columns
    (:func:`is_expression_value_col`)."""
    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    columns: dict[str, dict] = {}
    any_applied = False
    for col in value_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        col_sum = float(vals.sum())
        columns[col] = {"input_sum": col_sum}
        if col_sum <= 0:
            columns[col]["scale"] = 1.0
            continue
        scale = 1e6 / col_sum
        out[col] = vals * scale
        columns[col]["scale"] = scale
        columns[col]["output_sum"] = 1e6
        any_applied = True
    stats = {"applied": any_applied, "columns": columns, "value_cols": value_cols}
    return out, stats


def fpkm_to_tpm(df: pd.DataFrame, *, value_cols=None) -> tuple[pd.DataFrame, dict]:
    """Convert FPKM-scale expression columns to TPM by per-column rescaling:
    ``TPM_i = FPKM_i / sum(FPKM_j) * 1e6`` over finite rows. Mathematically the same
    as :func:`renormalize_to_million` (FPKM and TPM share the same per-sample
    normalization), but a self-documenting entry point for the quantifier→TPM step.
    Returns ``(df, stats)``."""
    return renormalize_to_million(df, value_cols=value_cols)


# ---------- technical-RNA normalization (the comparable biology view) ----------

#: QC groups removed by default in the legacy zero-and-renormalize path — the
#: technical-RNA set (mtDNA / NUMT-like / rRNA-like / polyA-bias lncRNA).
_TECHNICAL_RNA_GROUPS = frozenset(
    {"mt_dna", "mt_like_pseudogene", "rrna_like", "polyadenylation_bias_lncrna"}
)


def _technical_mask(df, *, label_col, id_col, remove_groups) -> pd.Series:
    """Boolean (row-aligned) for rows whose QC group is in ``remove_groups``,
    classified ENSG-first via :func:`cancerdata.gene_qc.classify_gene_qc`."""
    from .gene_qc import classify_gene_qc

    labels = df[label_col].fillna("").astype(str).str.strip()
    if id_col and id_col in df.columns:
        ids = df[id_col].fillna("").astype(str).str.split(".").str[0].str.strip()
        groups = [classify_gene_qc(s, ensembl_id=e).group for s, e in zip(labels, ids)]
    else:
        groups = [classify_gene_qc(s).group for s in labels]
    return pd.Series([g in remove_groups for g in groups], index=df.index, dtype=bool)


def _zero_and_renormalize(out, idx, value_cols, removable, records) -> bool:
    """Zero the removable rows in ``value_cols`` over rows ``idx`` and rescale the
    kept rows so each column's total mass is preserved. Mutates ``out`` in place;
    records per-column stats; returns whether anything was applied."""
    idx = list(idx)
    idx_set = set(idx)
    grp_removable = removable.loc[idx]
    applied = False
    for col in value_cols:
        vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
        valid = vals.notna()
        removable_valid = grp_removable & valid
        raw_sum = float(vals.sum())
        removed = float(vals[removable_valid].sum())
        remaining = raw_sum - removed
        records[col] = {
            "input_sum": raw_sum,
            "removed_tpm": removed,
            "removed_fraction": removed / raw_sum if raw_sum > 0 else 0.0,
            "removed_gene_count": int(grp_removable.sum()),
            "renormalization_factor": (
                raw_sum / remaining if raw_sum > 0 and remaining > 0 else 1.0
            ),
        }
        if raw_sum <= 0 or removed <= 0:
            continue
        remove_idx = [i for i in removable_valid[removable_valid].index if i in idx_set]
        out.loc[remove_idx, col] = 0.0
        if remaining <= 0:
            applied = True
            continue
        keep_valid = (~grp_removable) & valid
        keep_idx = [i for i in keep_valid[keep_valid].index if i in idx_set]
        out.loc[keep_idx, col] = vals.loc[keep_idx] * (raw_sum / remaining)
        applied = True
    return applied


def normalize_expression(
    df,
    *,
    label_col: str = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    value_cols=None,
    group_cols=None,
    censored_fill: str = "zero",
    technical_fraction: float = 0.25,
    exclude_ribosomal_proteins: bool = True,
    remove_groups=_TECHNICAL_RNA_GROUPS,
) -> tuple[pd.DataFrame, dict]:
    """Censor technical-RNA features and rescale each column's mass — the comparable
    biology view used after QC. Returns ``(normalized_df, stats)``.

    ``censored_fill``:
      - ``"zero"`` (default, legacy) — zero the technical-RNA rows
        (:func:`cancerdata.gene_qc.classify_gene_qc` group in ``remove_groups``:
        mtDNA / NUMT-like / rRNA-like / polyA-bias lncRNA) and rescale the kept rows
        so each column's **original total** is preserved. With ``group_cols``,
        normalizes within each group independently (e.g. per cohort in a long table).
      - any other value — apply the two-compartment reference :func:`clean_tpm`
        (clean_tpm_v4) over the value columns instead (the basis the packaged
        references ship on); ``technical_fraction`` / ``exclude_ribosomal_proteins``
        tune it.

    Classification is ENSG-first when ``id_col`` is present, else symbol-only.
    """
    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    if label_col not in df.columns:
        return df.copy(), {"applied": False, "reason": f"label column {label_col!r} not present"}
    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {"applied": False, "reason": "no expression value columns", "columns": {}}

    if censored_fill != "zero":
        gene_table = pd.DataFrame(
            {
                "Ensembl_Gene_ID": out[id_col] if id_col and id_col in out.columns else "",
                "Symbol": out[label_col],
            }
        )
        clean = clean_tpm(
            out[value_cols],
            gene_table=gene_table,
            exclude_ribosomal_proteins=exclude_ribosomal_proteins,
            technical_fraction=technical_fraction,
        )
        out[value_cols] = clean
        return out, {
            "applied": True,
            "reason": "clean_tpm (two-compartment)",
            "mode": censored_fill,
        }

    removable = _technical_mask(
        out, label_col=label_col, id_col=id_col, remove_groups={str(g) for g in remove_groups}
    )
    column_records: dict = {}
    group_records: dict = {}
    if group_cols is None:
        applied = _zero_and_renormalize(out, out.index, value_cols, removable, column_records)
    else:
        group_cols = [str(c) for c in group_cols if str(c) in out.columns]
        if not group_cols:
            return out, {"applied": False, "reason": "missing grouping columns", "columns": {}}
        applied = False
        for key, idx in out.groupby(group_cols, dropna=False).groups.items():
            key_label = "|".join(str(p) for p in (key if isinstance(key, tuple) else (key,)))
            group_records[key_label] = {}
            applied = (
                _zero_and_renormalize(out, idx, value_cols, removable, group_records[key_label])
                or applied
            )
    return out, {
        "applied": applied,
        "reason": "technical RNA zeroed and remaining expression renormalized"
        if applied
        else "no removable technical burden",
        "columns": column_records,
        "groups": group_records,
        "value_cols": value_cols,
        "removed_technical_gene_count": int(removable.sum()),
    }


def normalize_technical_rna_columns(
    df,
    *,
    label_col: str = "Symbol",
    value_cols=None,
    censored_fill: str = "zero",
    technical_fraction: float = 0.25,
) -> tuple[pd.DataFrame, dict]:
    """Technical-RNA normalization over a wide gene×sample frame — thin wrapper on
    :func:`normalize_expression` (no grouping, no noncoding removal)."""
    return normalize_expression(
        df,
        label_col=label_col,
        value_cols=value_cols,
        censored_fill=censored_fill,
        technical_fraction=technical_fraction,
    )


def normalize_technical_rna_long_table(
    df,
    *,
    label_col: str = "symbol",
    group_cols=("cancer_code", "subtype"),
    value_cols=("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"),
    censored_fill: str = "zero",
    technical_fraction: float = 0.25,
) -> tuple[pd.DataFrame, dict]:
    """Technical-RNA normalization applied **within each long-table cohort group** —
    thin wrapper on :func:`normalize_expression` with ``group_cols``."""
    return normalize_expression(
        df,
        label_col=label_col,
        group_cols=group_cols,
        value_cols=value_cols,
        censored_fill=censored_fill,
        technical_fraction=technical_fraction,
    )


def tpm_to_housekeeping_normalized(
    df,
    *,
    label_col: str = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    value_cols=None,
    panel_ids=None,
    pseudocount: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """Divide each expression column by the **geometric mean** of a housekeeping
    panel, putting expression on a unit-free ratio-to-baseline scale that survives
    library-prep depth drift in a way TPM doesn't.

    The panel defaults to cancerdata's housekeeping gene set
    (:func:`cancerdata.gene_families.housekeeping_gene_ids`), matched by Ensembl id.
    A small ``pseudocount`` makes the geometric mean robust to zeros. Returns
    ``(normalized_df, stats)`` with per-column denominator + panel coverage."""
    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {"applied": False, "reason": "no expression value columns", "columns": {}}

    panel = set(panel_ids) if panel_ids is not None else set(gene_families.housekeeping_gene_ids())
    if not (id_col and id_col in out.columns):
        return out, {
            "applied": False,
            "reason": "no id column for housekeeping panel",
            "columns": {},
        }
    panel_rows = _unversioned(out[id_col]).isin({str(g).split(".")[0] for g in panel})
    n_panel = int(panel_rows.sum())
    if n_panel == 0:
        return out, {
            "applied": False,
            "reason": "no housekeeping panel genes present",
            "columns": {},
        }

    columns: dict = {}
    applied = False
    for col in value_cols:
        vals = pd.to_numeric(out.loc[panel_rows, col], errors="coerce").dropna()
        denom = float(np.exp(np.log(vals.to_numpy() + pseudocount).mean())) if len(vals) else 0.0
        columns[col] = {"denominator": denom, "panel_genes_present": n_panel}
        if denom > 0:
            out[col] = pd.to_numeric(out[col], errors="coerce") / denom
            applied = True
    return out, {
        "applied": applied,
        "reason": "divided by housekeeping geometric mean" if applied else "panel denominator <= 0",
        "columns": columns,
        "value_cols": value_cols,
        "panel_genes_present": n_panel,
    }
