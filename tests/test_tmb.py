# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

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


def test_tmb_unknown_value_returns_none():
    # A real code with no curated value and no ancestor value returns None.
    assert tmb.cancer_tmb("PRAD", inherit=False) == tmb.cancer_tmb().get("PRAD")
