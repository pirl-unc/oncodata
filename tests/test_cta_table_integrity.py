# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integrity guards for the oncoref-owned CTA table.

oncoref is the source of truth for the cancer-testis-antigen *definition* — the
HPA tissue-restriction call over the candidate list. The mass-spec / peptide /
target-selection layer stays OUT (it lives in tsarina). These guards pin that
ownership boundary and the table's basic integrity.
"""

from pathlib import Path

import pandas as pd

from oncoref.cta import cta_unfiltered_gene_names

_CSV = Path(__file__).resolve().parents[1] / "oncoref" / "data" / "cancer-testis-antigens.csv"

# The exact shipped CTA schema (HPA-only projection, in column order). This is the
# stable contract downstream consumers read against — an exact, order-sensitive
# check so an added/dropped/reordered column is caught, not just missing required
# ones. tsarina's mass-spec columns (ms_restriction/ms_pmids/...) are deliberately
# absent: the ownership seam keeps the MS/peptide layer downstream.
CTA_COLUMNS = [
    "Symbol",
    "Aliases",
    "Full_Name",
    "Function",
    "Ensembl_Gene_ID",
    "source_databases",
    "protein_reproductive",
    "protein_thymus",
    "protein_reliability",
    "rna_reproductive",
    "rna_thymus",
    "protein_strict_expression",
    "rna_reproductive_frac",
    "rna_reproductive_and_thymus_frac",
    "rna_deflated_reproductive_frac",
    "rna_deflated_reproductive_and_thymus_frac",
    "Canonical_Transcript_ID",
    "biotype",
    "rna_80_pct_filter",
    "rna_90_pct_filter",
    "rna_95_pct_filter",
    "rna_97_pct_filter",
    "rna_98_pct_filter",
    "rna_99_pct_filter",
    "passes_filters",
    "rna_max_ntpm",
    "never_expressed",
    "rna_testis_ntpm",
    "rna_ovary_ntpm",
    "rna_placenta_ntpm",
    "rna_max_somatic_tissue",
    "rna_max_somatic_ntpm",
    "rna_somatic_detected_count",
    "rna_brain_max_ntpm",
    "rna_heart_max_ntpm",
    "rna_lung_max_ntpm",
    "rna_liver_max_ntpm",
    "rna_pancreas_max_ntpm",
    "protein_restriction",
    "protein_testis",
    "protein_ovary",
    "protein_placenta",
    "rna_restriction",
    "rna_restriction_level",
    "restriction",
    "restriction_confidence",
    "safety_flags",
]

# tsarina's mass-spec evidence columns must NOT ship in oncoref's HPA-only table
# (the ownership seam: HPA-only definition here, MS/peptide downstream).
_MASS_SPEC_COLUMNS = {"ms_restriction", "ms_pmids", "ms_healthy_somatic_tissues"}

# Paralog + placental onco-germline genes the table must carry.
_CURATED_GENES = {"SSX4B", "MAGEA2B", "GAGE10", "CT45A5", "CGB1", "PSG2", "CSH1"}


def test_schema_matches_contract():
    assert list(pd.read_csv(_CSV, nrows=0).columns) == CTA_COLUMNS


def test_no_mass_spec_columns():
    cols = set(pd.read_csv(_CSV, nrows=0).columns)
    assert cols.isdisjoint(_MASS_SPEC_COLUMNS), "MS columns belong in tsarina, not oncoref"


def test_curated_genes_present():
    missing = _CURATED_GENES - cta_unfiltered_gene_names()
    assert not missing, f"CTA table missing curated genes: {sorted(missing)}"


def test_no_duplicate_gene_ids():
    df = pd.read_csv(_CSV)
    dups = df["Ensembl_Gene_ID"].dropna()
    dups = dups[dups.duplicated()]
    assert dups.empty, f"duplicate Ensembl IDs in CTA table: {sorted(set(dups))}"
