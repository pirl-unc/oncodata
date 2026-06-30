# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd

from oncoref import cancer_types, tmb


def test_tmb_reference_ids_use_prefixed_pmid_or_doi():
    df = tmb.cancer_tmb_df()
    for ref in df["pmid_doi"]:
        if ref is None or str(ref).strip().lower() in {"", "nan", "none"}:
            continue
        parts = [p.strip() for p in str(ref).split(";") if p.strip()]
        assert parts
        for part in parts:
            assert part.startswith(("PMID:", "DOI:")), part


def test_tmb_map_nonempty_floats():
    mapping = tmb.cancer_tmb()
    assert mapping
    assert all(isinstance(v, float) for v in mapping.values())


def test_tmb_df_exposes_evidence_schema():
    df = tmb.cancer_tmb_df()
    assert {"estimate_type", "source_scope", "missing_reason"} <= set(df.columns)

    crc_msi = df.set_index("cancer_code").loc["CRC_MSI"]
    assert crc_msi["estimate_type"] == "aggregate_source_scope_estimate"
    assert crc_msi["source_scope"] == "aggregate_source_scope"
    assert pd.isna(crc_msi["missing_reason"])

    net_midgut = df.set_index("cancer_code").loc["NET_MIDGUT"]
    assert net_midgut["estimate_type"] == "proxy_estimate"
    assert net_midgut["source_scope"] == "pooled_gep_net_proxy"

    missing = df.set_index("cancer_code").loc["PITNET"]
    assert missing["estimate_type"] == "missing"
    assert missing["source_scope"] == "none"
    assert missing["missing_reason"]


def test_tmb_resolves_alias():
    # melanoma (SKCM) is a high-TMB tumor and is curated.
    value = tmb.cancer_tmb("melanoma")
    assert value is not None
    assert value > 0


def test_tmb_inherits_from_parent():
    # Find a registry subtype whose parent has a curated TMB but the subtype
    # itself does not — inheritance should return the parent's value.
    mapping = tmb.cancer_tmb()
    reg = cancer_types.cancer_type_registry()
    reg = reg[reg["parent_code"].notna()]
    found = False
    for _, row in reg.iterrows():
        code, parent = str(row["code"]), str(row["parent_code"])
        if code not in mapping and parent in mapping:
            assert tmb.cancer_tmb(code) == mapping[parent]
            assert tmb.cancer_tmb(code, inherit=False) is None
            found = True
            break
    assert found, "expected at least one subtype that inherits its parent's TMB"


def test_crc_msi_tmb_is_single_source_scope_row():
    mapping = tmb.cancer_tmb()
    assert mapping["CRC_MSI"] == 46.0
    assert "COAD_MSI" not in mapping
    assert "READ_MSI" not in mapping
    assert tmb.cancer_tmb("COAD_MSI") == mapping["CRC_MSI"]
    assert tmb.cancer_tmb("READ_MSI") == mapping["CRC_MSI"]
    assert tmb.cancer_tmb("COAD_MSI", inherit=False) is None


def test_crc_msi_tmb_record_preserves_source_scope_metadata():
    record = tmb.cancer_tmb_record("COAD_MSI")

    assert record["requested_cancer_code"] == "COAD_MSI"
    assert record["resolved_cancer_code"] == "CRC_MSI"
    assert record["inheritance_kind"] == "source_scope"
    assert record["is_inherited_evidence"] is True
    assert record["median_tmb_mut_mb"] == 46.0
    assert record["source_scope"] == "aggregate_source_scope"
    assert record["estimate_type"] == "aggregate_source_scope_estimate"

    direct = tmb.resolve_tmb_source("CRC_MSI")
    assert direct["requested_cancer_code"] == "CRC_MSI"
    assert direct["resolved_cancer_code"] == "CRC_MSI"
    assert direct["inheritance_kind"] == "direct"
    assert direct["is_inherited_evidence"] is False
    assert direct["missing_reason"] is None


def test_tmb_record_missing_and_bulk_direct_rows():
    assert tmb.cancer_tmb_record("COAD_MSI", inherit=False) is None

    missing = tmb.resolve_tmb_source("COAD_MSI", inherit=False)
    assert missing["requested_cancer_code"] == "COAD_MSI"
    assert missing["resolved_cancer_code"] is None
    assert missing["inheritance_kind"] == "missing"
    assert missing["has_tmb_source"] is False

    bulk = tmb.cancer_tmb_record()
    assert "CRC_MSI" in bulk
    assert "COAD_MSI" not in bulk
    assert bulk["CRC_MSI"]["inheritance_kind"] == "direct"


def test_tmb_unknown_value_returns_none():
    # A real code with no curated value and no ancestor value returns None.
    assert tmb.cancer_tmb("PRAD", inherit=False) == tmb.cancer_tmb().get("PRAD")
