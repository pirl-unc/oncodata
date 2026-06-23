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

"""Generic antigen-panel patient coverage helpers.

These wrappers make the gene-panel input explicit. For CTA defaults, use
:mod:`oncoref.cta_coverage`.
"""

from __future__ import annotations

from collections.abc import Iterable

from .coverage import (
    DEFAULT_EXPRESSED_TPM,
    DEFAULT_THRESHOLDS,
    addressable_fraction,
    greedy_coverage,
    mean_antigens_per_patient,
    patient_coverage,
    resolve_gene_set,
)


def patient_antigen_fractions(
    cancer_type,
    *,
    gene_ids: Iterable[str],
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    proteoform: bool = True,
):
    """Per antigen, fraction of patients expressing it above ``threshold_tpm``."""
    from .coverage import cta_patient_fractions

    return cta_patient_fractions(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )


def addressable_antigen_fraction(
    cancer_type,
    *,
    gene_ids: Iterable[str],
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    proteoform: bool = True,
) -> float:
    """Fraction of patients expressing at least one antigen in an explicit panel."""
    return addressable_fraction(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )


def greedy_antigen_coverage(
    cancer_type,
    *,
    gene_ids: Iterable[str],
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    max_genes: int | None = None,
    proteoform: bool = True,
):
    """Greedy set-cover order for an explicit antigen panel."""
    return greedy_coverage(
        cancer_type,
        threshold_tpm=threshold_tpm,
        gene_ids=gene_ids,
        max_genes=max_genes,
        proteoform=proteoform,
    )


def mean_panel_antigens_per_patient(
    cancer_type,
    *,
    gene_ids: Iterable[str],
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    proteoform: bool = True,
) -> float:
    """Mean number of expressed antigens per patient for an explicit panel."""
    return mean_antigens_per_patient(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )


__all__ = [
    "DEFAULT_EXPRESSED_TPM",
    "DEFAULT_THRESHOLDS",
    "addressable_antigen_fraction",
    "greedy_antigen_coverage",
    "mean_panel_antigens_per_patient",
    "patient_antigen_fractions",
    "patient_coverage",
    "resolve_gene_set",
]
