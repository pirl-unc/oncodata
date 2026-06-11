# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Consistency guards for the pirlygenes->cancerdata cancer-type registry sync (#27, O1)."""

import importlib.util
from pathlib import Path

import pandas as pd

from cancerdata import cancer_type_registry, cancer_type_subtypes_of

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "cancerdata" / "data" / "cancer-type-registry.csv"
_SYNC = _REPO / "scripts" / "sync_cancer_type_registry.py"

# Codes added in the refresh to pirlygenes' 159-code taxonomy (incl. the CRC parent
# and the MSI/POLE/sarcoma subtypes the ontology navigation relies on).
_NEW_CODES = {"CRC", "UCEC_MSI", "UCEC_POLE", "SARC_RMS", "SARC_LPS", "SARC_ESS", "MENINGIOMA"}


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("_sync_registry", _SYNC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_schema_matches_contract():
    cols = list(pd.read_csv(_CSV, nrows=0).columns)
    assert cols == _load_sync_module().REGISTRY_COLUMNS


def test_new_codes_present():
    codes = set(cancer_type_registry()["code"])
    missing = _NEW_CODES - codes
    assert not missing, f"registry regressed — missing synced codes: {sorted(missing)}"


def test_no_duplicate_codes():
    codes = pd.read_csv(_CSV, dtype=str)["code"]
    dups = sorted(codes[codes.duplicated()])
    assert not dups, f"duplicate cancer-type codes: {dups}"


def test_parent_code_referential_integrity():
    # Every parent_code must name an existing code — no orphan branches in the tree.
    df = pd.read_csv(_CSV, dtype=str, keep_default_na=False)
    codes = set(df["code"])
    orphans = sorted({p for p in df["parent_code"] if p and p not in codes})
    assert not orphans, f"parent_code(s) with no matching code: {orphans}"


def test_crc_hierarchy():
    # The headline example: CRC is the parent of COAD/READ.
    assert set(cancer_type_subtypes_of("CRC")) >= {"COAD", "READ"}
