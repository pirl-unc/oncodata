# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from oncoref import apd1, ici_response


def test_apd1_map_nonempty_floats():
    mapping = apd1.cancer_apd1_response()
    assert mapping
    assert all(isinstance(v, float) for v in mapping.values())


def test_apd1_resolves_alias():
    # melanoma (SKCM) is the canonical high-responder.
    assert apd1.cancer_apd1_response("melanoma") == apd1.cancer_apd1_response("SKCM")
    assert apd1.cancer_apd1_response("SKCM") > 0


def test_apd1_inherits_from_parent():
    mapping = apd1.cancer_apd1_response()
    reg = apd1.cancer_type_registry()
    reg = reg[reg["parent_code"].notna()]
    found = False
    for _, row in reg.iterrows():
        code, parent = str(row["code"]), str(row["parent_code"])
        if code not in mapping and parent in mapping:
            assert apd1.cancer_apd1_response(code) == mapping[parent]
            assert apd1.cancer_apd1_response(code, inherit=False) is None
            found = True
            break
    assert found, "expected at least one subtype that inherits its parent's ORR"


def test_crc_msi_apd1_is_single_source_scope_row():
    mapping = apd1.cancer_apd1_response()
    assert mapping["CRC_MSI"] == 43.8
    assert "COAD_MSI" not in mapping
    assert "READ_MSI" not in mapping
    assert apd1.cancer_apd1_response("COAD_MSI") == mapping["CRC_MSI"]
    assert apd1.cancer_apd1_response("READ_MSI") == mapping["CRC_MSI"]
    assert apd1.cancer_apd1_response("READ_MSI", inherit=False) is None

    inherited = apd1.cancer_apd1_response(include_inherited=True)
    assert inherited["COAD_MSI"] == mapping["CRC_MSI"]
    assert inherited["READ_MSI"] == mapping["CRC_MSI"]
    assert "READ_MSI" not in apd1.cancer_apd1_response(include_inherited=True, inherit=False)


def test_crc_msi_apd1_record_preserves_requested_and_source_codes():
    record = apd1.cancer_apd1_response_record("COAD_MSI")

    assert record["requested_cancer_code"] == "COAD_MSI"
    assert record["resolved_cancer_code"] == "CRC_MSI"
    assert record["inheritance_kind"] == "source_scope"
    assert record["is_inherited_evidence"] is True
    assert record["selected_regimen"] == "PD-1"
    assert record["selected_drug_target"] == "PD-1"
    assert record["apd1_orr_pct"] == 43.8
    assert record["source_anchor"] == "PMID:33264544"

    source = apd1.resolve_apd1_response_source("READ_MSI")
    assert source["requested_cancer_code"] == "READ_MSI"
    assert source["resolved_cancer_code"] == "CRC_MSI"
    assert source["has_apd1_response_source"] is True


def test_apd1_record_inherit_false_and_bulk_direct_rows_only():
    assert apd1.cancer_apd1_response_record("READ_MSI", inherit=False) is None
    missing = apd1.resolve_apd1_response_source("READ_MSI", inherit=False)
    assert missing == {
        "requested_cancer_code": "READ_MSI",
        "resolved_cancer_code": None,
        "inheritance_kind": "missing",
        "is_inherited_evidence": False,
        "selected_regimen": None,
        "selected_drug_target": None,
        "has_apd1_response_source": False,
    }

    bulk = apd1.cancer_apd1_response_record()
    assert "CRC_MSI" in bulk
    assert "COAD_MSI" not in bulk
    assert "READ_MSI" not in bulk
    assert bulk["CRC_MSI"]["inheritance_kind"] == "direct"

    inherited = apd1.cancer_apd1_response_record(include_inherited=True)
    assert inherited["COAD_MSI"]["requested_cancer_code"] == "COAD_MSI"
    assert inherited["COAD_MSI"]["resolved_cancer_code"] == "CRC_MSI"
    assert inherited["COAD_MSI"]["inheritance_kind"] == "source_scope"
    assert inherited["COAD_MSI"]["is_inherited_evidence"] is True


def test_apd1_record_helpers_are_exported():
    import oncoref

    assert oncoref.cancer_apd1_response_record is apd1.cancer_apd1_response_record
    assert oncoref.resolve_apd1_response_source is apd1.resolve_apd1_response_source
    assert ici_response.apd1_response_record("COAD_MSI")["resolved_cancer_code"] == "CRC_MSI"
    assert ici_response.apd1_response_source("COAD_MSI")["inheritance_kind"] == "source_scope"


def test_btc_apd1_is_single_pan_biliary_source_scope_row():
    mapping = apd1.cancer_apd1_response()
    assert mapping["BTC"] == 5.8
    assert "CHOL" not in mapping
    assert "GBC" not in mapping
    assert apd1.cancer_apd1_response("CHOL") == mapping["BTC"]
    assert apd1.cancer_apd1_response("GBC") == mapping["BTC"]
    assert apd1.cancer_apd1_response("GBC", inherit=False) is None


def test_sgc_apd1_is_single_pan_salivary_source_scope_row():
    mapping = apd1.cancer_apd1_response()
    assert mapping["SGC"] == 4.6
    assert "ACINIC" not in mapping
    assert apd1.cancer_apd1_response("ACINIC") == mapping["SGC"]
    assert apd1.cancer_apd1_response("ACINIC", inherit=False) is None
    # ADCC still has a direct dual-checkpoint anchor in the compact fallback table.
    assert mapping["ADCC"] == 6.0


def test_net_nonpancreatic_apd1_is_single_source_scope_row():
    mapping = apd1.cancer_apd1_response()
    assert mapping["NET_NONPANCREATIC"] == 0.0
    assert "NET_LUNG" not in mapping
    assert "NET_MIDGUT" not in mapping
    assert "NET_RECTAL" not in mapping
    assert apd1.cancer_apd1_response("NET_LUNG") == mapping["NET_NONPANCREATIC"]
    assert apd1.cancer_apd1_response("NET_MIDGUT") == mapping["NET_NONPANCREATIC"]
    assert apd1.cancer_apd1_response("NET_RECTAL") == mapping["NET_NONPANCREATIC"]
    assert apd1.cancer_apd1_response("NET_LUNG", inherit=False) is None
    assert mapping["NET_PANCREAS"] == 11.0


def test_extrapulmonary_g3_nen_apd1_is_not_lung_lcnec():
    mapping = apd1.cancer_apd1_response()
    assert mapping["NEN_G3_EXTRAPULMONARY"] == 3.4
    assert mapping["NEC_LUNG_LARGECELL"] == 29.4
    assert apd1.cancer_apd1_response("extrapulmonary G3 NEN") == 3.4
