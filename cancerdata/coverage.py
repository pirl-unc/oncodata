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

"""Per-patient antigen coverage over the full per-sample matrices.

These answer questions a per-cohort summary cannot, because they need the *joint*
per-sample matrix — which patient expresses which antigen, not just per-gene
prevalence:

  - :func:`cta_patient_fractions` — per gene, the fraction of a cohort's patients
    expressing it above a TPM threshold;
  - :func:`addressable_fraction` — the fraction of patients expressing **≥1** of a
    gene panel (the union — "what share of this cohort a CTA-directed therapy could
    address"), which is NOT the per-gene fractions summed;
  - :func:`greedy_coverage` — a minimal antigen panel by greedy set cover: at each
    step add the gene covering the most still-uncovered patients.

All operate on :func:`cancerdata.expression.per_sample_expression` (clean TPM), so
they need the cohort's per-sample matrix fetched (see :mod:`cancerdata.source_matrices`).
The default gene panel is the expressed CTA set (:func:`cancerdata.cta.CTA_gene_ids`).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .cta import CTA_gene_id_to_name, CTA_gene_ids
from .expression import per_sample_expression

#: Default clean-TPM threshold for "expressed in a patient". 10 TPM is a common
#: cut for a confidently-expressed transcript.
DEFAULT_EXPRESSED_TPM: float = 10.0

_BASE = ["Ensembl_Gene_ID", "Symbol"]


def _panel_ids(gene_ids: Iterable[str] | None) -> set[str]:
    ids = set(gene_ids) if gene_ids is not None else set(CTA_gene_ids())
    return {str(g).split(".")[0] for g in ids}


#: Identity columns a (possibly proteoform-collapsed) antigen frame may carry;
#: everything else is a per-sample expression column.
_ANTIGEN_ID_COLS = ("Ensembl_Gene_ID", "Symbol", "proteoform_id")


def _hit_matrix(cancer_type, *, threshold_tpm: float, gene_ids, proteoform: bool = True):
    """``(panel rows, sample columns, gene×sample boolean hit matrix, id columns)``
    where a hit is clean TPM >= ``threshold_tpm`` for one antigen in one patient.

    With ``proteoform=True`` (default), identical-protein paralogs in the panel
    (CTAG1A+CTAG1B = NY-ESO-1, XAGE1A+XAGE1B, …) are **summed per patient** into one
    antigen row before thresholding — the biologically correct unit for antigen
    coverage: RNA-seq reads multi-map between the loci (so per-gene TPM under-counts
    the proteoform), and a TCR/vaccine targets the shared protein once. With
    ``proteoform=False`` each gene is kept separate."""
    df = per_sample_expression(cancer_type, normalize="tpm_clean")
    unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    panel = _panel_ids(gene_ids)
    sub = df[unversioned.isin(panel)].reset_index(drop=True)
    if proteoform and len(sub):
        from .proteoforms import collapse_to_proteoforms

        sub = collapse_to_proteoforms(sub).reset_index(drop=True)
    id_cols = [c for c in _ANTIGEN_ID_COLS if c in sub.columns]
    samples = [c for c in sub.columns if c not in id_cols]
    hits = sub[samples].to_numpy(dtype=float) >= float(threshold_tpm)
    return sub, samples, hits, id_cols


def cta_patient_fractions(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> pd.DataFrame:
    """Per antigen, the fraction of a cohort's patients expressing it above
    ``threshold_tpm`` (clean TPM). Returns ``Ensembl_Gene_ID`` (a real ENSG — the
    canonical member for a collapsed proteoform), ``Symbol``, ``proteoform_id`` (the
    antigen identity, present when ``proteoform=True``), ``fraction_expressing``,
    ``n_patients_expressing``, ``n_patients`` — sorted by prevalence. The default
    gene panel is the expressed CTA set; identical-protein paralogs are summed to
    one antigen (``proteoform=True``)."""
    sub, samples, hits, id_cols = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    n = len(samples)
    out = sub[id_cols].copy()
    counts = hits.sum(axis=1)
    out["n_patients_expressing"] = counts
    out["fraction_expressing"] = counts / n if n else 0.0
    out["n_patients"] = n
    return out.sort_values("fraction_expressing", ascending=False).reset_index(drop=True)


def addressable_fraction(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> float:
    """Fraction of a cohort's patients expressing **at least one** antigen in the
    panel above ``threshold_tpm`` — the faithful "addressable" share (the union across
    patients, which the per-gene fractions can't be summed into). Identical-protein
    paralogs are summed to one antigen (``proteoform=True``). 0.0 for an empty
    cohort/panel."""
    _, samples, hits, _ = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    if not samples or hits.size == 0:
        return 0.0
    return float(hits.any(axis=0).mean())


def greedy_coverage(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    max_genes: int | None = None,
    proteoform: bool = True,
) -> pd.DataFrame:
    """Greedy set-cover panel: at each step add the antigen covering the most
    patients not yet covered by the chosen panel, until every coverable patient is
    covered (or ``max_genes`` is reached). Identical-protein paralogs are summed to
    one antigen (``proteoform=True``), so e.g. CTAG1A/CTAG1B counts once.

    Returns one row per chosen antigen, in selection order: ``rank``,
    ``Ensembl_Gene_ID``, ``Symbol``, ``marginal_patients`` (newly covered),
    ``marginal_fraction``, ``cumulative_patients``, ``cumulative_fraction``. The
    cumulative fraction is the coverage curve; its last value equals
    :func:`addressable_fraction` once the panel is exhausted (unless ``max_genes``
    truncates it first). Ties are broken by total prevalence (deterministic)."""
    sub, samples, hits, _ = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    n = len(samples)
    rows: list[dict] = []
    if n == 0 or hits.size == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "Ensembl_Gene_ID",
                "Symbol",
                "marginal_patients",
                "marginal_fraction",
                "cumulative_patients",
                "cumulative_fraction",
            ]
        )

    covered = np.zeros(n, dtype=bool)
    remaining = set(range(len(sub)))
    total_prev = hits.sum(axis=1)  # tie-break: prefer the more broadly expressed gene
    limit = max_genes if max_genes is not None else len(sub)

    while remaining and len(rows) < limit:
        # Pick the gene covering the most still-uncovered patients; ties by total
        # prevalence, then by smallest row index — fully deterministic (sorted scan +
        # strict tuple comparison), independent of set-iteration order.
        best_g, best_new, best_prev = None, 0, -1
        for g in sorted(remaining):
            new = int(np.count_nonzero(hits[g] & ~covered))
            if (new, total_prev[g]) > (best_new, best_prev):
                best_g, best_new, best_prev = g, new, int(total_prev[g])
        if best_g is None or best_new == 0:
            break  # no remaining gene covers a new patient
        covered |= hits[best_g]
        remaining.discard(best_g)
        cum = int(covered.sum())
        row = {
            "rank": len(rows) + 1,
            "Ensembl_Gene_ID": str(sub.at[best_g, "Ensembl_Gene_ID"]),
            "Symbol": str(sub.at[best_g, "Symbol"]),
            "marginal_patients": best_new,
            "marginal_fraction": best_new / n,
            "cumulative_patients": cum,
            "cumulative_fraction": cum / n,
        }
        if "proteoform_id" in sub.columns:
            row["proteoform_id"] = str(sub.at[best_g, "proteoform_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def mean_antigens_per_patient(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> float:
    """Mean number of panel antigens a patient in the cohort expresses above
    ``threshold_tpm`` — the per-patient antigen *load* (how many CTAs the average
    patient presents, not just whether ≥1). Equals the sum over antigens of their
    per-patient prevalence; identical-protein paralogs count once
    (``proteoform=True``). 0.0 for an empty cohort/panel."""
    _, samples, hits, _ = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    if not samples or hits.size == 0:
        return 0.0
    return float(hits.sum(axis=0).mean())


def _scalar_by_cohort(
    scalar_fn, name, cohorts, *, threshold_tpm, gene_ids, proteoform
) -> pd.Series:
    """Map a per-cohort scalar coverage function over cohorts → ``Series``. When
    ``cohorts`` is ``None`` every cohort with a *cached* per-sample matrix is used
    (uncached ones are skipped, never fetched implicitly); an explicit ``cohorts``
    list is taken as-is."""
    from . import source_matrices

    codes = list(cohorts) if cohorts is not None else source_matrices.available_cohorts()
    out: dict[str, float] = {}
    for code in codes:
        if cohorts is None and not source_matrices.is_cached(code):
            continue
        out[str(code)] = scalar_fn(
            code, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
        )
    return pd.Series(out, name=name)


def mean_antigens_per_patient_by_cohort(
    cohorts: Iterable[str] | None = None,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> pd.Series:
    """``{cohort code -> mean antigens per patient}`` over the cohorts that have a
    cached per-sample matrix (default: all cached ones); uncached cohorts are skipped
    rather than fetched."""
    return _scalar_by_cohort(
        mean_antigens_per_patient,
        "mean_antigens_per_patient",
        cohorts,
        threshold_tpm=threshold_tpm,
        gene_ids=gene_ids,
        proteoform=proteoform,
    )


def addressable_fraction_by_cohort(
    cohorts: Iterable[str] | None = None,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> pd.Series:
    """``{cohort code -> addressable fraction}`` over the cohorts that have a cached
    per-sample matrix (default: all cached ones); uncached cohorts are skipped rather
    than fetched."""
    return _scalar_by_cohort(
        addressable_fraction,
        "addressable_fraction",
        cohorts,
        threshold_tpm=threshold_tpm,
        gene_ids=gene_ids,
        proteoform=proteoform,
    )


def cta_id_to_name() -> dict[str, str]:
    """``{unversioned CTA gene id -> symbol}`` for labelling coverage outputs."""
    return CTA_gene_id_to_name()
