# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Consistency guards for the tsarina->cancerdata CTA table sync (issue #16)."""

import importlib.util
from pathlib import Path

import pandas as pd

from cancerdata.cta import CTA_unfiltered_gene_names

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "cancerdata" / "data" / "cancer-testis-antigens.csv"
_SYNC = _REPO / "scripts" / "sync_cta_table.py"

# Paralog + placental onge-germline genes that were missing before the tsarina
# re-sync; their absence was the bug in issue #16.
_NEWLY_SYNCED = {"SSX4B", "MAGEA2B", "GAGE10", "CT45A5", "CGB1", "PSG2", "CSH1"}

# tsarina's mass-spec evidence columns must NOT ship in cancerdata's HPA-only table.
_MS_COLUMNS = {"ms_restriction", "ms_pmids", "ms_healthy_somatic_tissues"}


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("_sync_cta", _SYNC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shipped_schema_matches_canonical_contract():
    cols = list(pd.read_csv(_CSV, nrows=0).columns)
    assert cols == _load_sync_module().CANONICAL_COLUMNS


def test_no_mass_spec_columns_leak():
    cols = set(pd.read_csv(_CSV, nrows=0).columns)
    assert cols.isdisjoint(_MS_COLUMNS)


def test_newly_synced_genes_present():
    names = CTA_unfiltered_gene_names()
    missing = _NEWLY_SYNCED - names
    assert not missing, f"CTA table regressed — missing re-synced genes: {sorted(missing)}"


def test_no_duplicate_gene_ids():
    df = pd.read_csv(_CSV)
    dups = df["Ensembl_Gene_ID"].dropna()
    dups = dups[dups.duplicated()]
    assert dups.empty, f"duplicate Ensembl IDs in CTA table: {sorted(set(dups))}"
