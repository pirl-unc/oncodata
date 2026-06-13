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

"""Pure build-time transforms for precomputed expression artifacts.

Kept dependency-light (pandas/numpy only) and separate from the read accessors so
the heavy artifact builders in ``scripts/`` can be unit-tested without any data
bundle. The maintainer runs the ``scripts/generate_*`` wrappers; this module is
the testable core.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .expression_engine import ID_COLUMNS as _ID_COLS

#: Per-gene cohort percentile breakpoints — dense in the actionable upper tail so
#: a consumer can place a sample's gene as a percentile rank within the cohort
#: (matches the shipped ``cancer-reference-expression-percentiles`` schema: 26
#: ``p{n}`` columns from p0 to p100). Read back by ``expression.cohort_gene_percentiles``.
PERCENTILE_BREAKPOINTS = (
    0,
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    96,
    97,
    98,
    99,
    100,
)

#: threshold (within-sample percentile rank) -> output column name.
#: 0.95 = "in the top 5% of expressed genes in that sample".
WITHIN_SAMPLE_THRESHOLDS = {
    0.99: "frac_samples_top1pct",
    0.95: "frac_samples_top5pct",
    0.90: "frac_samples_top10pct",
}


def sample_columns(df: pd.DataFrame) -> list[str]:
    """Per-sample value columns (everything that isn't a gene-id column)."""
    return [c for c in df.columns if c not in _ID_COLS]


def sum_proteoform_tpm(
    df: pd.DataFrame,
    group_map: Mapping[str, Iterable[str]],
    sample_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Collapse genes that encode an identical protein into one summed row.

    For each proteoform group in ``group_map`` (``{label: member gene IDs}``), the
    per-sample TPM of the member genes is *summed* into a single row labelled by
    the group — the biologically correct unit when reads multi-map between
    identical-protein loci (CTAG1A/CTAG1B, SSX4/SSX4B, the CT47A family, …). Genes
    in no group pass through unchanged.

    This is the pure core behind the runtime ``proteoform_representative_samples``
    accessor; the offline percentile/within-sample generators can apply it first
    so a proteoform-summed artifact and a per-gene one share the same arithmetic.

    ``df`` has ``Ensembl_Gene_ID``, ``Symbol`` and one column per sample. Gene IDs
    are matched version-insensitively. The output carries a stable proteoform
    identity **without overloading the Ensembl ID column** (so it stays a real,
    joinable ENSG):

      - ``Ensembl_Gene_ID`` — the group's **canonical member** ENSG (the
        lexicographically-smallest member id) for a collapsed group; the gene's own
        ENSG for a singleton. Always a genuine Ensembl id.
      - ``proteoform_id`` — the equivalence-class identity: the slash-joined label
        for a group (``CTAG1A/CTAG1B``), the gene's own ``Symbol`` for a singleton.
        Total over every row — the join key for proteoform-level analyses.
      - ``Symbol`` — the slash-label for a group / the gene's symbol for a singleton
        (display).

    First-appearance row order is preserved.

    Summation uses ``min_count=1`` so a missing measurement stays missing: a cell
    that is all-NaN (e.g. a gene absent from a cohort after an outer-merge across
    cohorts) remains NaN rather than collapsing to 0.0 — "not measured" must not
    silently become "measured zero". Within a group, a present member still sums
    even when a sibling is NaN.

    Note: matching is best-effort by gene ID. If the frame's IDs are on a
    different Ensembl basis than the registry, unmatched members simply pass
    through ungrouped (no error) — the build-script anchor check guards the
    registry side; align the bases if a known group fails to collapse.
    """
    cols = list(sample_cols) if sample_cols is not None else sample_columns(df)
    work = df.copy()
    unversioned = work["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    gene_to_label: dict[str, str] = {}
    gene_to_canonical: dict[str, str] = {}
    for raw_label, members in group_map.items():
        member_ids = [str(g).split(".")[0] for g in members]
        canonical = min(member_ids)  # stable, collision-free representative ENSG
        for mid in member_ids:
            gene_to_label[mid] = str(raw_label)
            gene_to_canonical[mid] = canonical

    label = unversioned.map(gene_to_label)
    in_group = label.notna()
    canonical = unversioned.map(gene_to_canonical)

    work["_key"] = label.where(in_group, unversioned)
    # ENSG stays a real Ensembl id (canonical member / own id), never the label.
    work["_out_id"] = canonical.where(in_group, work["Ensembl_Gene_ID"].astype(str))
    work["_out_symbol"] = label.where(in_group, work["Symbol"].astype(str))
    # proteoform_id is total: the class label for groups, the symbol for singletons.
    work["_out_pfid"] = label.where(in_group, work["Symbol"].astype(str))

    grouped = work.groupby("_key", sort=False)
    ids = grouped[["_out_id", "_out_symbol", "_out_pfid"]].first()
    sums = grouped[cols].sum(min_count=1)
    agg = ids.join(sums).rename(
        columns={
            "_out_id": "Ensembl_Gene_ID",
            "_out_symbol": "Symbol",
            "_out_pfid": "proteoform_id",
        }
    )
    return agg.reset_index(drop=True)[["Ensembl_Gene_ID", "Symbol", "proteoform_id", *cols]]


def within_sample_top_fractions(
    df: pd.DataFrame,
    sample_cols: Iterable[str] | None = None,
    *,
    thresholds: Iterable[float] = tuple(WITHIN_SAMPLE_THRESHOLDS),
) -> pd.DataFrame:
    """Per-gene fraction of a cohort's samples in which the gene is highly
    expressed *within that sample* (signal a).

    For each sample (column), genes are ranked across the whole gene axis into a
    within-sample percentile (``rank(pct=True)``). A gene clears threshold ``t``
    in a sample when its within-sample percentile ≥ ``t``. The per-gene output is
    the fraction of the cohort's samples where it clears ``t`` — i.e. "in what
    fraction of these tumors is this gene among the top ``(1-t)`` of expressed
    genes."

    Returns one row per gene with ``Ensembl_Gene_ID``, ``Symbol``,
    ``frac_samples_top{1,5,10}pct`` (per threshold), and ``n_samples``.
    """
    cols = list(sample_cols) if sample_cols is not None else sample_columns(df)
    if not cols:
        raise ValueError("no per-sample columns to rank")

    # Rank genes WITHIN each sample (axis=0 = down the gene rows, per column).
    ranks = df[cols].rank(axis=0, pct=True)

    out = pd.DataFrame(
        {
            "Ensembl_Gene_ID": df["Ensembl_Gene_ID"].astype(str).to_numpy(),
            "Symbol": df["Symbol"].astype(str).to_numpy(),
        }
    )
    if "proteoform_id" in df.columns:  # carry the proteoform identity through (collapsed input)
        out["proteoform_id"] = df["proteoform_id"].astype(str).to_numpy()
    for t in thresholds:
        pct = round((1.0 - t) * 100)
        out[f"frac_samples_top{pct}pct"] = (ranks >= t).mean(axis=1).to_numpy()
    out["n_samples"] = len(cols)
    return out


def cohort_percentile_vectors(
    df: pd.DataFrame,
    sample_cols: Iterable[str] | None = None,
    *,
    breakpoints: Iterable[int] = PERCENTILE_BREAKPOINTS,
    store_log1p: bool = True,
) -> pd.DataFrame:
    """Per-gene cohort percentile vector (the ``…-percentiles`` artifact).

    For each gene, take its expression across the cohort's samples and reduce that
    distribution to the ``breakpoints`` percentiles (``p0`` = min … ``p50`` =
    median … ``p100`` = max). Unlike :func:`within_sample_top_fractions` (the
    within-sample, across-genes axis), this is the within-cohort, across-samples
    axis — it lets a consumer place a sample's gene as a percentile rank within
    the cohort rather than an absolute TPM.

    Computed on whatever rows are passed in — the generator drops technical genes
    first, so the breakpoints describe the biological clean-TPM view. ``NaN``
    cells (a gene unmeasured in some samples) are ignored per gene. Stored as
    ``log1p`` (``store_log1p=True``) in ``float16`` for compactness, exactly the
    encoding :func:`cancerdata.expression.cohort_gene_percentiles` restores with
    ``expm1``. Returns one row per gene with ``Ensembl_Gene_ID``, ``Symbol`` and a
    ``p{n}`` column per breakpoint.
    """
    cols = list(sample_cols) if sample_cols is not None else sample_columns(df)
    if not cols:
        raise ValueError("no per-sample columns to summarize")
    bps = list(breakpoints)

    mat = df[cols].to_numpy(dtype=float)
    if store_log1p:
        mat = np.log1p(mat)
    # nanpercentile down the sample axis (axis=1) -> shape (len(bps), n_genes).
    # all-NaN gene rows yield NaN (np warns); suppress since it's expected.
    with np.errstate(all="ignore"):
        q = np.nanpercentile(mat, bps, axis=1)

    out = pd.DataFrame(
        {
            "Ensembl_Gene_ID": df["Ensembl_Gene_ID"].astype(str).to_numpy(),
            "Symbol": df["Symbol"].astype(str).to_numpy(),
        }
    )
    if "proteoform_id" in df.columns:  # carry the proteoform identity through (collapsed input)
        out["proteoform_id"] = df["proteoform_id"].astype(str).to_numpy()
    for i, bp in enumerate(bps):
        out[f"p{bp}"] = q[i].astype("float16")
    return out


def cohort_medoids(
    df: pd.DataFrame,
    sample_cols: Iterable[str] | None = None,
    *,
    k: int = 5,
    distance_log1p: bool = True,
) -> pd.DataFrame:
    """Pick ``k`` representative real per-sample vectors spanning a cohort.

    The packaged cohort references are aggregates; this selects a bounded set of
    *real* per-sample columns to keep (the ``…-representatives`` artifact). The
    first pick is the cohort medoid — the sample minimizing total distance to all
    others (the most central, "typical" tumor). Each subsequent pick is the
    sample farthest (max–min Euclidean distance) from those already chosen, a
    deterministic farthest-first traversal that spreads the picks across the
    within-cohort variation rather than clustering them near the center.

    Distance is computed on ``log1p`` TPM (``distance_log1p=True``) so a few
    very-high-TPM genes don't dominate the geometry, but the returned values are
    the **original** TPM (the artifact ships clean TPM; the reader optionally
    ``log1p``-transforms). Cohorts with ``≤ k`` samples keep all of them, in
    input order. Returns ``Ensembl_Gene_ID``, ``Symbol`` and the ``k`` selected
    sample columns (original names), medoid first.
    """
    cols = list(sample_cols) if sample_cols is not None else sample_columns(df)
    if not cols:
        raise ValueError("no per-sample columns to choose from")
    base = ["Ensembl_Gene_ID", "Symbol"]

    if len(cols) <= k:
        return df[base + cols].reset_index(drop=True)

    mat = df[cols].to_numpy(dtype=float).T  # samples × genes
    if distance_log1p:
        mat = np.log1p(mat)
    mat = np.nan_to_num(mat, nan=0.0)

    # Pairwise squared Euclidean via the ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
    # identity (avoids materializing an n×n×g intermediate).
    gram = mat @ mat.T
    sq_norms = np.diag(gram)
    sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    dist = np.sqrt(np.maximum(sq, 0.0))

    selected = [int(np.argmin(dist.sum(axis=1)))]  # central medoid
    while len(selected) < k:
        min_to_selected = dist[:, selected].min(axis=1)
        min_to_selected[selected] = -np.inf  # never re-pick
        selected.append(int(np.argmax(min_to_selected)))

    keep = [cols[i] for i in selected]
    return df[base + keep].reset_index(drop=True)
