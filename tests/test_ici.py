# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd

from oncoref import apd1, ici


def test_regimens_and_table():
    assert ici.ici_regimens() == ("PD-1", "PD-L1", "PD-1+CTLA-4")
    df = ici.cancer_ici_response_df()
    assert {"cancer_code", "regimen", "orr_pct"} <= set(df.columns)
    # all three regimens are actually present (not just PD-1)
    assert set(df["regimen"]) == {"PD-1", "PD-L1", "PD-1+CTLA-4"}
    assert (df["regimen"] == "PD-L1").sum() >= 10  # anti-PD-L1 is well-represented


def test_ici_anchor_table_exposes_evidence_schema():
    df = ici.cancer_ici_response_df()
    expected = {
        "response_metric",
        "response_unit",
        "response_ci_low",
        "response_ci_high",
        "response_numerator",
        "response_denominator",
        "source_n",
        "source_verified",
        "value_basis",
        "source_anchor",
        "endpoint_population",
        "therapy_regimen_class",
        "evidence_type",
        "histology_match",
        "is_direct_cancer_code_evidence",
        "evidence_source_code",
        "source_scope",
        "missing_reason",
    }
    assert expected <= set(df.columns)

    crc = df[(df["cancer_code"] == "CRC_MSI") & (df["regimen"] == "PD-1")].iloc[0]
    assert crc["response_metric"] == "ORR"
    assert crc["response_numerator"] == 67
    assert crc["response_denominator"] == 153
    assert crc["response_ci_low"] == 35.8
    assert crc["response_ci_high"] == 52.0
    assert crc["therapy_regimen_class"] == "anti_pd1_monotherapy"
    assert crc["evidence_type"] == "direct_reported"
    assert crc["histology_match"] == "direct"
    assert bool(crc["is_direct_cancer_code_evidence"]) is True
    assert crc["source_scope"] == "aggregate_source"
    assert crc["source_anchor"] == "PMID:33264544"

    coad = df[(df["cancer_code"] == "COAD") & (df["regimen"] == "PD-1")].iloc[0]
    assert coad["evidence_type"] == "derived_blend"
    assert coad["histology_match"] == "derived"
    assert bool(coad["is_direct_cancer_code_evidence"]) is False
    assert coad["source_scope"] == "derived_blend"
    assert pd.isna(coad["response_denominator"])
    assert pd.isna(coad["source_anchor"])


def test_apd1_anchor_table_uses_same_evidence_schema_for_fallback_targets():
    df = apd1.cancer_apd1_response_df()
    assert {
        "response_metric",
        "response_numerator",
        "response_denominator",
        "therapy_regimen_class",
        "evidence_type",
        "source_scope",
    } <= set(df.columns)

    acc = df[df["cancer_code"] == "ACC"].iloc[0]
    assert acc["drug_target"] == "PD-1+CTLA-4"
    assert acc["therapy_regimen_class"] == "anti_pd1_ctla4_combination"
    assert acc["response_numerator"] == 3
    assert acc["response_denominator"] == 21
    assert acc["evidence_type"] == "direct_reported"


def test_per_regimen_and_pin():
    # melanoma carries both anti-PD-1 mono and the ipi+nivo doublet, as distinct sources
    per = ici.cancer_ici_response("SKCM", fallback=False)
    assert per["PD-1"] > 0 and per["PD-1+CTLA-4"] > per["PD-1"]
    assert ici.cancer_ici_response("SKCM", regimen="PD-1+CTLA-4") == per["PD-1+CTLA-4"]


def test_fallback_prefers_pd1_then_pdl1():
    # SKCM has anti-PD-1 -> fallback picks PD-1, not the higher combo value.
    assert ici.cancer_ici_regimen("SKCM") == "PD-1"
    assert ici.cancer_ici_response("SKCM") == ici.cancer_ici_response("SKCM", regimen="PD-1")
    # SARC_ASPS has only anti-PD-L1 -> fallback resolves to PD-L1.
    assert ici.cancer_ici_regimen("SARC_ASPS") == "PD-L1"
    assert ici.cancer_ici_response("SARC_ASPS") == ici.cancer_ici_response(
        "SARC_ASPS", regimen="PD-L1"
    )


def test_maps_and_alias():
    assert ici.cancer_ici_response("melanoma") == ici.cancer_ici_response("SKCM")
    full = ici.cancer_ici_response()
    pdl1 = ici.cancer_ici_response(regimen="PD-L1")
    assert len(full) > len(pdl1) >= 10
    assert all(isinstance(v, float) for v in full.values())


def test_whole_table_per_regimen_mapping():
    # cancer_type=None with fallback=False -> {code: {regimen: orr}} for every cancer.
    per = ici.cancer_ici_response(fallback=False)
    assert isinstance(per["SKCM"], dict)
    assert per["SKCM"] == {"PD-1": 42.0, "PD-1+CTLA-4": 57.6}
    # single-regimen cancers carry a one-entry mapping
    assert set(per["SARC_ASPS"]) == {"PD-L1"}
    # the PD-L1 members match the pinned PD-L1 map
    assert {c for c, m in per.items() if "PD-L1" in m} == set(
        ici.cancer_ici_response(regimen="PD-L1")
    )


def test_crc_msi_ici_is_single_source_scope_row():
    full = ici.cancer_ici_response()
    per = ici.cancer_ici_response(fallback=False)
    assert full["CRC_MSI"] == 43.8
    assert per["CRC_MSI"] == {"PD-1": 43.8, "PD-1+CTLA-4": 55.0}
    assert "COAD_MSI" not in full
    assert "READ_MSI" not in full
    assert ici.cancer_ici_response("COAD_MSI") == full["CRC_MSI"]
    assert ici.cancer_ici_response("READ_MSI", regimen="PD-1") == full["CRC_MSI"]
    assert ici.cancer_ici_response("COAD_MSI", fallback=False) == per["CRC_MSI"]
    assert ici.cancer_ici_response("READ_MSI", inherit=False) is None
    assert ici.cancer_ici_regimen("READ_MSI") == "PD-1"


def test_crc_msi_ici_record_preserves_inheritance_metadata():
    record = ici.cancer_ici_response_record("COAD_MSI")
    assert record["requested_cancer_code"] == "COAD_MSI"
    assert record["resolved_cancer_code"] == "CRC_MSI"
    assert record["inheritance_kind"] == "source_scope"
    assert record["is_inherited_evidence"] is True
    assert record["regimen"] == "PD-1"
    assert record["selected_regimen"] == "PD-1"
    assert record["orr_pct"] == 43.8
    assert record["response_numerator"] == 67
    assert record["response_denominator"] == 153
    assert record["response_ci_low"] == 35.8
    assert record["response_ci_high"] == 52.0
    assert record["source_anchor"] == "PMID:33264544"
    assert record["source_scope"] == "aggregate_source"
    assert record["endpoint_population"] == (
        "first-line metastatic MSI-H/dMMR colorectal (pembrolizumab arm)"
    )

    per_regimen = ici.cancer_ici_response_record("READ_MSI", fallback=False)
    assert set(per_regimen) == {"PD-1", "PD-1+CTLA-4"}
    assert per_regimen["PD-1"]["requested_cancer_code"] == "READ_MSI"
    assert per_regimen["PD-1"]["resolved_cancer_code"] == "CRC_MSI"
    assert per_regimen["PD-1+CTLA-4"]["response_denominator"] == 119
    assert per_regimen["PD-1+CTLA-4"]["source_anchor"] == "PMID:29355075"

    assert ici.cancer_ici_response_record("READ_MSI", inherit=False) is None
    assert ici.cancer_ici_response_record("READ_MSI", fallback=False, inherit=False) == {}


def test_resolve_ici_response_source_reports_direct_proxy_and_missing():
    direct = ici.resolve_ici_response_source("SKCM")
    assert direct["requested_cancer_code"] == "SKCM"
    assert direct["resolved_cancer_code"] == "SKCM"
    assert direct["inheritance_kind"] == "direct"
    assert direct["is_inherited_evidence"] is False
    assert direct["selected_regimen"] == "PD-1"
    assert direct["available_regimens"] == ("PD-1", "PD-1+CTLA-4")
    assert direct["has_ici_response_source"] is True
    assert direct["source_anchor"] == "PMID:28889792"

    proxy = ici.resolve_ici_response_source("COAD_MSI")
    assert proxy["requested_cancer_code"] == "COAD_MSI"
    assert proxy["resolved_cancer_code"] == "CRC_MSI"
    assert proxy["inheritance_kind"] == "source_scope"
    assert proxy["is_inherited_evidence"] is True
    assert proxy["selected_regimen"] == "PD-1"
    assert proxy["available_regimens"] == ("PD-1", "PD-1+CTLA-4")
    assert proxy["source_anchor"] == "PMID:33264544"
    assert proxy["source_scope"] == "aggregate_source"

    per_regimen = ici.resolve_ici_response_source("READ_MSI", fallback=False)
    assert per_regimen["resolved_cancer_code"] == "CRC_MSI"
    assert per_regimen["selected_regimen"] is None
    assert per_regimen["available_regimens"] == ("PD-1", "PD-1+CTLA-4")

    missing = ici.resolve_ici_response_source("NBL")
    assert missing == {
        "requested_cancer_code": "NBL",
        "resolved_cancer_code": None,
        "inheritance_kind": "missing",
        "is_inherited_evidence": False,
        "selected_regimen": None,
        "available_regimens": (),
        "has_ici_response_source": False,
    }


def test_ici_response_record_whole_table_matches_value_maps():
    records = ici.cancer_ici_response_record()
    values = ici.cancer_ici_response()
    assert set(records) == set(values)
    assert {code: record["orr_pct"] for code, record in records.items()} == values
    assert "COAD_MSI" not in records
    assert records["CRC_MSI"]["inheritance_kind"] == "direct"

    pdl1_records = ici.cancer_ici_response_record(regimen="PD-L1")
    pdl1_values = ici.cancer_ici_response(regimen="PD-L1")
    assert {code: record["orr_pct"] for code, record in pdl1_records.items()} == pdl1_values


def test_parent_code_helper_treats_nan_parent_as_missing():
    registry = ici.cancer_type_registry().set_index("code")
    assert ici._parent_code("CRC", registry) is None


def test_regimen_maps_cached():
    # _regimen_maps is memoized (same object back from the cache).
    assert ici._regimen_maps() is ici._regimen_maps()
