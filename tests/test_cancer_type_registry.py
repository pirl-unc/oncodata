# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integrity guards for the oncodata-owned cancer-type registry.

oncodata owns the cancer-type ontology outright — the registry is authoritative
here, not mirrored from anywhere. These guards pin the schema and the structural
invariants the navigation/grouping/fusion accessors rely on.
"""

from pathlib import Path

import pandas as pd

from oncodata import cancer_type_registry, cancer_type_subtypes_of

_CSV = Path(__file__).resolve().parents[1] / "oncodata" / "data" / "cancer-type-registry.csv"

# The registry schema (shipped column order).
REGISTRY_COLUMNS = [
    "code",
    "name",
    "family",
    "primary_tissue",
    "primary_template",
    "parent_code",
    "subtype_key",
    "expression_source",
    "source_cohort",
    "source_pmid",
    "notes",
    "mixture_cohort",
    "pediatric",
    "differentiation",
    "viral_etiology",
    "viral_agent",
    "fusion_driven",
    "fusion_driver",
]


def test_schema_matches_contract():
    assert list(pd.read_csv(_CSV, nrows=0).columns) == REGISTRY_COLUMNS


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
    assert set(cancer_type_subtypes_of("CRC")) >= {"COAD", "READ"}


def test_registry_has_expected_scale():
    # Sanity floor so an accidental truncation is caught.
    assert len(cancer_type_registry()) >= 159
