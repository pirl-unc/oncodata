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


def _hit_matrix(cancer_type, *, threshold_tpm: float, gene_ids):
    """``(panel rows of the matrix, sample columns, gene×sample boolean hit matrix)``
    where a hit is clean TPM >= ``threshold_tpm`` for one gene in one patient."""
    df = per_sample_expression(cancer_type, normalize="tpm_clean")
    unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    panel = _panel_ids(gene_ids)
    sub = df[unversioned.isin(panel)].reset_index(drop=True)
    samples = [c for c in df.columns if c not in _BASE]
    hits = sub[samples].to_numpy(dtype=float) >= float(threshold_tpm)
    return sub, samples, hits


def cta_patient_fractions(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per gene, the fraction of a cohort's patients expressing it above
    ``threshold_tpm`` (clean TPM). Returns ``Ensembl_Gene_ID``, ``Symbol``,
    ``fraction_expressing``, ``n_patients_expressing``, ``n_patients`` — sorted by
    prevalence. The default gene panel is the expressed CTA set."""
    sub, samples, hits = _hit_matrix(cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids)
    n = len(samples)
    out = sub[_BASE].copy()
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
) -> float:
    """Fraction of a cohort's patients expressing **at least one** gene in the panel
    above ``threshold_tpm`` — the faithful "addressable" share (the union across
    patients, which the per-gene fractions can't be summed into). 0.0 for an empty
    cohort/panel."""
    _, samples, hits = _hit_matrix(cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids)
    if not samples or hits.size == 0:
        return 0.0
    return float(hits.any(axis=0).mean())


def greedy_coverage(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    max_genes: int | None = None,
) -> pd.DataFrame:
    """Greedy set-cover panel: at each step add the gene covering the most patients
    not yet covered by the chosen panel, until every coverable patient is covered
    (or ``max_genes`` is reached).

    Returns one row per chosen gene, in selection order: ``rank``,
    ``Ensembl_Gene_ID``, ``Symbol``, ``marginal_patients`` (newly covered),
    ``marginal_fraction``, ``cumulative_patients``, ``cumulative_fraction``. The
    cumulative fraction is the coverage curve; its last value equals
    :func:`addressable_fraction` once the panel is exhausted (unless ``max_genes``
    truncates it first). Ties are broken by total prevalence (deterministic)."""
    sub, samples, hits = _hit_matrix(cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids)
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
        best_g, best_new = None, 0
        for g in remaining:
            new = int(np.count_nonzero(hits[g] & ~covered))
            if new > best_new or (
                new == best_new and best_g is not None and total_prev[g] > total_prev[best_g]
            ):
                best_g, best_new = g, new
        if best_g is None or best_new == 0:
            break  # no remaining gene covers a new patient
        covered |= hits[best_g]
        remaining.discard(best_g)
        cum = int(covered.sum())
        rows.append(
            {
                "rank": len(rows) + 1,
                "Ensembl_Gene_ID": str(sub.at[best_g, "Ensembl_Gene_ID"]),
                "Symbol": str(sub.at[best_g, "Symbol"]),
                "marginal_patients": best_new,
                "marginal_fraction": best_new / n,
                "cumulative_patients": cum,
                "cumulative_fraction": cum / n,
            }
        )
    return pd.DataFrame(rows)


def addressable_fraction_by_cohort(
    cohorts: Iterable[str] | None = None,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
) -> pd.Series:
    """``{cohort code -> addressable fraction}`` over the cohorts that have a
    per-sample matrix (default: all of them). Skips cohorts whose matrix isn't
    available rather than fetching every one implicitly."""
    from . import source_matrices

    codes = list(cohorts) if cohorts is not None else source_matrices.available_cohorts()
    out: dict[str, float] = {}
    for code in codes:
        if cohorts is None and not source_matrices.is_cached(code):
            continue
        out[str(code)] = addressable_fraction(code, threshold_tpm=threshold_tpm, gene_ids=gene_ids)
    return pd.Series(out, name="addressable_fraction")


def cta_id_to_name() -> dict[str, str]:
    """``{unversioned CTA gene id -> symbol}`` for labelling coverage outputs."""
    return CTA_gene_id_to_name()
