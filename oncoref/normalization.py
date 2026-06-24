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

The headline transform is **clean TPM** (:func:`clean_tpm`): a **multi-compartment**
renormalization. Each non-biological compartment is pinned, per sample, to a fixed
fraction of the 1e6 budget — **ribosomal proteins → 16%** (:data:`RIBOSOMAL_PROTEIN_FRACTION`),
**other-technical → 9%** (:data:`OTHER_TECHNICAL_FRACTION`; mtDNA, NUMT, rRNA + pseudogenes,
polyA-bias lncRNA), **biology → 75%**
(:data:`BIOLOGICAL_FRACTION`). The variable, pipeline-driven other-technical/ribosomal
fractions no longer inflate real genes, so biological clean TPM is directly comparable
across samples and sources.

**Calibration.** The fractions are the *fresh-frozen polyA* median, measured on TCGA
LUAD/SKCM raw TPM (ribosomal ~16%, technical ~9%); pinning each compartment to its
clean-prep typical nudges a degraded / different-prep / different-depletion sample back
toward that reference (and giving ribosomal proteins their own budget means high rRNA in
one sample can't squeeze them). **Biology-neutrality:** the split is biology-neutral up to
a constant — biology excludes both censored compartments and is pinned to a fixed fraction
regardless of how the censored budget is internally divided — so it does not change
biological comparability; it keeps the censored compartments themselves comparable and the
budget empirically interpretable. (Verified: LUAD clean TPM lands at exactly 16/9/75 per
sample.)

**Curated membership.** Which genes are technical/ribosomal is a *curated, biology-defined*
list, with compartment assignment taken from ``clean-tpm-censored-genes.csv``:
``category == "ribosomal_protein"`` gets the 16% budget and ``category == "technical"``
gets the 9% budget (see :mod:`oncoref.gene_families`). Never define censoring from
expression variance or abundance: cancer-testis antigens are high-variance *by definition*
(that's what makes them targets), so a variance-based rule would censor the very antigens
this library exists to find. Use data only to *calibrate* the fractions and *validate
completeness* of the curated list.

**Single definition.** :func:`clean_tpm` is the one and only clean-TPM implementation; every
consumer routes through it (the per-sample matrix loader, :func:`normalize_expression`'s
reference path, and ``scripts/rebuild_expression_artifacts``). Never inline a compartment
split anywhere; read the budgets from the public fraction constants, not magic numbers.

Plus the supporting helpers: drop technical genes for a biology-only view,
housekeeping/log/rank transforms. Censored-gene and gene-family lists come from
:mod:`oncoref.gene_families`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import gene_families
from .expression_engine import sample_columns


def _value_cols(df: pd.DataFrame, value_cols=None) -> list[str]:
    if value_cols is not None:
        return list(value_cols)
    return sample_columns(df)


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


# clean-TPM compartment budgets — the fraction of the 1e6 per-sample budget each
# non-biological compartment is pinned to. CALIBRATED to the fresh-frozen-polyA median
# (measured on TCGA LUAD/SKCM: ribosomal-protein category ~16%, other technical
# genes ~9%); biology gets the remainder. Public so a consumer reads the applied value
# instead of re-hardcoding the magic number, and the value survives future re-calibration.
#: Clean-TPM budget for the ribosomal-protein compartment
#: (``clean-tpm-censored-genes.csv:category == "ribosomal_protein"``).
RIBOSOMAL_PROTEIN_FRACTION = 0.16
#: Clean-TPM budget for the **other-technical** compartment (mtDNA, NUMT, rRNA +
#: pseudogenes, polyA-bias lncRNA;
#: ``clean-tpm-censored-genes.csv:category == "technical"``).
OTHER_TECHNICAL_FRACTION = 0.09
#: Combined censored budget (ribosomal + other-technical = 25%). The two compartments are
#: pinned separately, but this is the total non-biological fraction — matches pirlygenes'
#: ``TECHNICAL_FRACTION`` semantics (the constant is the *combined* 25%; the per-compartment
#: splits are RIBOSOMAL_PROTEIN_FRACTION / OTHER_TECHNICAL_FRACTION).
TECHNICAL_FRACTION = round(RIBOSOMAL_PROTEIN_FRACTION + OTHER_TECHNICAL_FRACTION, 10)  # 0.25
#: Clean-TPM budget for the biological compartment (everything else) — the remainder.
BIOLOGICAL_FRACTION = round(1.0 - TECHNICAL_FRACTION, 10)  # 0.75


def _compartment_masks(
    gene_table: pd.DataFrame, *, exclude_ribosomal_proteins: bool
) -> tuple[pd.Series, pd.Series]:
    """``(ribosomal_mask, technical_mask)`` row-aligned to ``gene_table`` — the two
    non-biological clean-TPM compartments (biology is everything in neither).

    ``ribosomal`` is every row in ``clean-tpm-censored-genes.csv`` with
    ``category == "ribosomal_protein"``; ``technical`` is every row with
    ``category == "technical"``. The category-specific censored table is the contract,
    not the broad ribosomal-family table: a ribosomal-family CTA deliberately absent from
    the censored table (RPL10L / ENSG00000165496) stays biology. With
    ``exclude_ribosomal_proteins=False`` the ribosomal proteins join biology (empty
    ribosomal mask) and only the technical-RNA set is censored (the legacy view)."""
    if "Ensembl_Gene_ID" not in gene_table.columns:
        raise ValueError("clean TPM needs an 'Ensembl_Gene_ID' column (censoring is ENSG-keyed)")
    ids = _unversioned(gene_table["Ensembl_Gene_ID"])
    if exclude_ribosomal_proteins:
        ribosomal = ids.isin(gene_families.clean_tpm_ribosomal_gene_ids())
        technical = ids.isin(gene_families.clean_tpm_other_technical_gene_ids())
    else:
        ribosomal = pd.Series(False, index=gene_table.index)
        technical = ids.isin(
            gene_families.clean_tpm_censored_gene_ids(include_ribosomal_proteins=False)
        )
    return ribosomal, technical


def clean_tpm(
    values: pd.DataFrame,
    gene_table: pd.DataFrame | None = None,
    *,
    exclude_ribosomal_proteins: bool = True,
    ribosomal_protein_fraction: float = RIBOSOMAL_PROTEIN_FRACTION,
    other_technical_fraction: float = OTHER_TECHNICAL_FRACTION,
) -> pd.DataFrame:
    """Multi-compartment **clean TPM** on a gene×sample matrix.

    ``values`` is genes (rows) × samples (cols); ``gene_table`` (``Ensembl_Gene_ID``
    row-aligned to ``values``) assigns each gene to a compartment. Each non-biological
    compartment is rescaled, **per sample**, to a fixed fraction of the 1e6 budget:

      - **ribosomal proteins** (``category == "ribosomal_protein"`` in the censored list)
        → ``ribosomal_protein_fraction`` (16%);
      - **other-technical** (``category == "technical"`` in the censored list)
        → ``other_technical_fraction`` (9%);
      - **biology** (everything else) → the remainder (75%).

    **Why two censored compartments, not one.** The fractions are calibrated to the
    *fresh-frozen polyA* median (TCGA LUAD/SKCM: ribosomal ~16%, technical ~9%). Pinning
    each compartment to its clean-prep typical nudges a degraded / different-prep /
    different-depletion sample — whose rRNA or ribosomal fraction is inflated — *back
    toward that reference*, so biological clean TPM is comparable across preps. Giving
    ribosomal proteins their **own** budget (rather than lumping them with contamination)
    means high rRNA in one sample can't squeeze them, and vice-versa — each is pinned
    independently. (The split is biology-neutral up to a constant — biology excludes both
    compartments and is pinned to a fixed fraction regardless — but it keeps the censored
    compartments themselves comparable and the budget empirically interpretable.)

    An empty/zero compartment simply contributes 0 (the others still fill their share).
    The public clean-TPM contract is deliberately singular: 16% ribosomal proteins,
    9% other technical RNA, and 75% biological genes. Use separately named helpers
    such as :func:`drop_technical_rna` / :func:`filter_technical_rna` for biology-only
    or technical-drop views, not alternate clean-TPM definitions."""
    if not exclude_ribosomal_proteins:
        raise ValueError(
            "clean_tpm has one canonical 16/9/75 contract and always censors "
            "ribosomal proteins; use drop_technical_rna(..., "
            "exclude_ribosomal_proteins=False) or filter_technical_rna for "
            "non-clean-TPM biology-only/drop views"
        )
    if (
        ribosomal_protein_fraction != RIBOSOMAL_PROTEIN_FRACTION
        or other_technical_fraction != OTHER_TECHNICAL_FRACTION
    ):
        raise ValueError(
            "clean_tpm fraction knobs are deprecated: clean_tpm always uses the "
            "canonical 16% ribosomal / 9% other-technical / 75% biological budget"
        )
    for name, frac in (
        ("ribosomal_protein_fraction", ribosomal_protein_fraction),
        ("other_technical_fraction", other_technical_fraction),
    ):
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if ribosomal_protein_fraction + other_technical_fraction >= 1.0:
        raise ValueError(
            "ribosomal_protein_fraction + other_technical_fraction must be < 1 (biology needs a budget)"
        )
    if gene_table is None:
        raise ValueError(
            "clean_tpm needs a gene_table with 'Ensembl_Gene_ID' to assign compartments"
        )

    ribosomal, technical = _compartment_masks(
        gene_table, exclude_ribosomal_proteins=exclude_ribosomal_proteins
    )
    rm, tm = ribosomal.to_numpy(), technical.to_numpy()
    bm = ~(rm | tm)
    bio_fraction = 1.0 - ribosomal_protein_fraction - other_technical_fraction

    clean = values.astype(float).copy()
    for mask, fraction in (
        (rm, ribosomal_protein_fraction),
        (tm, other_technical_fraction),
        (bm, bio_fraction),
    ):
        if not mask.any():
            continue
        comp_sum = values.loc[mask].sum(axis=0)
        scale = pd.Series(0.0, index=values.columns, dtype=float)
        scale.loc[comp_sum > 0] = fraction * 1_000_000.0 / comp_sum.loc[comp_sum > 0]
        clean.loc[mask] = values.loc[mask].mul(scale, axis=1)
    return clean.fillna(0.0)


def drop_technical_rna(
    df: pd.DataFrame, *, exclude_ribosomal_proteins: bool = True
) -> pd.DataFrame:
    """Biology-only view: drop the clean-TPM censored rows (technical RNA, and by
    default ribosomal proteins). Row order preserved."""
    mask = _censored_mask(df, exclude_ribosomal_proteins=exclude_ribosomal_proteins)
    return df.loc[~mask].reset_index(drop=True)


def filter_technical_rna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the strict technical-RNA loci (mtDNA / NUMT / rRNA / nuclear-retained
    lncRNA) — a lighter filter than :func:`drop_technical_rna` (keeps ribosomal
    proteins). Row order preserved."""
    mask = _unversioned(df["Ensembl_Gene_ID"]).isin(gene_families.technical_rna_gene_ids())
    return df.loc[~mask].reset_index(drop=True)


def normalize_to_housekeeping(df: pd.DataFrame, value_cols=None) -> pd.DataFrame:
    """Rescale each value column to its housekeeping-panel baseline (unitless: 1.0 =
    panel level). The df-only convenience over :func:`tpm_to_housekeeping_normalized`
    (the canonical normalizer, which also returns per-column diagnostics) — both use the
    **geometric mean** of the housekeeping panel, the single housekeeping method (see
    that function for why geomean over median). A column with no measurable panel gene
    becomes NaN rather than silently staying on the input scale.

    ``value_cols`` defaults to every per-sample value column (not just named-TPM
    columns), so a plain ``genes × samples`` frame normalizes as expected. A frame
    without an ``Ensembl_Gene_ID`` column (no way to locate the panel) passes through
    **unchanged** — call :func:`tpm_to_housekeeping_normalized` directly for the stats
    dict that reports whether normalization was applied."""
    out, _ = tpm_to_housekeeping_normalized(df, value_cols=_value_cols(df, value_cols))
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
#: technical-RNA set. Single source of truth lives in :mod:`oncoref.gene_qc`.
from .gene_qc import TECHNICAL_RNA_GROUPS as _TECHNICAL_RNA_GROUPS  # noqa: E402


def _technical_mask(df, *, label_col, id_col, remove_groups) -> pd.Series:
    """Boolean (row-aligned) for rows whose QC group is in ``remove_groups``,
    classified ENSG-first via :func:`oncoref.gene_qc.classify_gene_qc`."""
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
    ribosomal_protein_fraction: float = RIBOSOMAL_PROTEIN_FRACTION,
    other_technical_fraction: float = OTHER_TECHNICAL_FRACTION,
    exclude_ribosomal_proteins: bool = True,
    remove_groups=_TECHNICAL_RNA_GROUPS,
) -> tuple[pd.DataFrame, dict]:
    """Censor technical-RNA features and rescale each column's mass — the comparable
    biology view used after QC. Returns ``(normalized_df, stats)``.

    ``censored_fill``:
      - ``"zero"`` (default, legacy) — zero the technical-RNA rows
        (:func:`oncoref.gene_qc.classify_gene_qc` group in ``remove_groups``:
        mtDNA / NUMT-like / rRNA-like / polyA-bias lncRNA) and rescale the kept rows
        so each column's **original total** is preserved. With ``group_cols``,
        normalizes within each group independently (e.g. per cohort in a long table).
      - any other value — apply the canonical :func:`clean_tpm` 16/9/75 transform
        over the value columns instead (the basis the packaged references ship on).
        Historical budget knobs remain in the signature only to reject noncanonical
        transition uses; they do not define alternate clean-TPM modes.

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
            ribosomal_protein_fraction=ribosomal_protein_fraction,
            other_technical_fraction=other_technical_fraction,
        )
        out[value_cols] = clean
        # Stamp the *applied* compartment budgets into the metadata (#446-analog) so a
        # consumer reads the values actually used, surviving future re-calibration.
        return out, {
            "applied": True,
            "reason": "clean_tpm (multi-compartment)",
            "mode": censored_fill,
            "ribosomal_protein_fraction": ribosomal_protein_fraction,
            "other_technical_fraction": other_technical_fraction,
            "biological_fraction": 1.0 - ribosomal_protein_fraction - other_technical_fraction,
            "exclude_ribosomal_proteins": bool(exclude_ribosomal_proteins),
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
    other_technical_fraction: float = OTHER_TECHNICAL_FRACTION,
) -> tuple[pd.DataFrame, dict]:
    """Technical-RNA normalization over a wide gene×sample frame — thin wrapper on
    :func:`normalize_expression` (no grouping, no noncoding removal)."""
    return normalize_expression(
        df,
        label_col=label_col,
        value_cols=value_cols,
        censored_fill=censored_fill,
        other_technical_fraction=other_technical_fraction,
    )


def normalize_technical_rna_long_table(
    df,
    *,
    label_col: str = "symbol",
    group_cols=("cancer_code", "subtype"),
    value_cols=("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"),
    censored_fill: str = "zero",
    other_technical_fraction: float = OTHER_TECHNICAL_FRACTION,
) -> tuple[pd.DataFrame, dict]:
    """Technical-RNA normalization applied **within each long-table cohort group** —
    thin wrapper on :func:`normalize_expression` with ``group_cols``."""
    return normalize_expression(
        df,
        label_col=label_col,
        group_cols=group_cols,
        value_cols=value_cols,
        censored_fill=censored_fill,
        other_technical_fraction=other_technical_fraction,
    )


def tpm_to_housekeeping_normalized(
    df,
    *,
    label_col: str = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    value_cols=None,
    panel_ids=None,
    panel_name: str | None = None,
    pseudocount: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """Divide each expression column by the **geometric mean** of a housekeeping
    panel, putting expression on a unit-free ratio-to-baseline scale that survives
    library-prep depth drift in a way TPM doesn't.

    The single, canonical housekeeping normalizer — the method behind
    ``per_sample_expression(normalize="tpm_clean_hk")`` and the df-only convenience
    :func:`normalize_to_housekeeping`. The geometric mean (the geNorm convention) is
    used over the median because it is the natural centre for multiplicative
    log-normal expression and is dominated by no single deviating reference gene; a
    small ``pseudocount`` keeps it finite through zeros.

    The panel defaults to oncoref's legacy qPCR/reference housekeeping gene set
    (:func:`oncoref.gene_families.housekeeping_gene_ids`), matched by Ensembl id.
    Clean-TPM callers should pass
    :func:`oncoref.gene_families.clean_tpm_biological_housekeeping_gene_ids`
    explicitly, because ribosomal legacy references live in clean TPM's non-biological
    ribosomal compartment.
    Because HK-normalized values are ratios to this panel, numerical thresholds are
    panel-dependent; prefer clean TPM, log1p(clean TPM), or percentile-rank clean TPM
    when the analysis needs an absolute, compressed, or rank-only expression space.
    ``value_cols`` defaults to the **named-TPM** columns (:func:`is_expression_value_col`)
    — for a plain ``genes × samples`` frame whose columns aren't ``*_TPM``, pass them
    explicitly or use :func:`normalize_to_housekeeping` (which defaults to all sample
    columns). Returns ``(normalized_df, stats)`` with per-column denominator + panel
    coverage; a column with no measurable panel gene is blanked to NaN (never left on
    the input scale beside normalized siblings)."""
    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {"applied": False, "reason": "no expression value columns", "columns": {}}

    if panel_ids is None:
        panel = set(gene_families.housekeeping_gene_ids())
        panel_name = panel_name or "legacy_qpcr_housekeeping"
    else:
        panel = set(panel_ids)
        panel_name = panel_name or "custom"
    if not (id_col and id_col in out.columns):
        return out, {
            "applied": False,
            "reason": "no id column for housekeeping panel",
            "panel": panel_name,
            "columns": {},
        }
    panel_rows = _unversioned(out[id_col]).isin({str(g).split(".")[0] for g in panel})
    n_panel = int(panel_rows.sum())
    if n_panel == 0:
        return out, {
            "applied": False,
            "reason": "no housekeeping panel genes present",
            "panel": panel_name,
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
        else:
            # No measurable panel genes in this column -> it can't be put on the
            # ratio-to-baseline scale. Blank it to NaN rather than silently leaving
            # it on the raw-TPM scale alongside normalized siblings (the scale-mixing
            # trap).
            out[col] = np.nan
    return out, {
        "applied": applied,
        "reason": "divided by housekeeping geometric mean" if applied else "panel denominator <= 0",
        "panel": panel_name,
        "columns": columns,
        "value_cols": value_cols,
        "panel_genes_present": n_panel,
    }
