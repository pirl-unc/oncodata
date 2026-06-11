# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integrity guards for the cancerdata-owned CTA table.

cancerdata is the source of truth for the cancer-testis-antigen *definition* — the
HPA tissue-restriction call over the candidate list. The mass-spec / peptide /
target-selection layer stays OUT (it lives in tsarina). These guards pin that
ownership boundary and the table's basic integrity.
"""

from pathlib import Path

import pandas as pd

from cancerdata.cta import CTA_unfiltered_gene_names

_CSV = Path(__file__).resolve().parents[1] / "cancerdata" / "data" / "cancer-testis-antigens.csv"

# The HPA-evidence columns that must be present for the restriction synthesis.
_REQUIRED_COLUMNS = {
    "Symbol",
    "Ensembl_Gene_ID",
    "passes_filters",
    "never_expressed",
    "protein_restriction",
    "rna_restriction",
    "restriction",
    "restriction_confidence",
}

# tsarina's mass-spec evidence columns must NOT ship in cancerdata's HPA-only table
# (the ownership seam: HPA-only definition here, MS/peptide downstream).
_MASS_SPEC_COLUMNS = {"ms_restriction", "ms_pmids", "ms_healthy_somatic_tissues"}

# Paralog + placental onco-germline genes the table must carry.
_CURATED_GENES = {"SSX4B", "MAGEA2B", "GAGE10", "CT45A5", "CGB1", "PSG2", "CSH1"}


def test_required_columns_present():
    cols = set(pd.read_csv(_CSV, nrows=0).columns)
    assert cols >= _REQUIRED_COLUMNS


def test_no_mass_spec_columns():
    cols = set(pd.read_csv(_CSV, nrows=0).columns)
    assert cols.isdisjoint(_MASS_SPEC_COLUMNS), "MS columns belong in tsarina, not cancerdata"


def test_curated_genes_present():
    missing = _CURATED_GENES - CTA_unfiltered_gene_names()
    assert not missing, f"CTA table missing curated genes: {sorted(missing)}"


def test_no_duplicate_gene_ids():
    df = pd.read_csv(_CSV)
    dups = df["Ensembl_Gene_ID"].dropna()
    dups = dups[dups.duplicated()]
    assert dups.empty, f"duplicate Ensembl IDs in CTA table: {sorted(set(dups))}"
