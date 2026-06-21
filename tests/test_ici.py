# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from oncoref import ici


def test_regimens_and_table():
    assert ici.ici_regimens() == ("PD-1", "PD-L1", "PD-1+CTLA-4")
    df = ici.cancer_ici_response_df()
    assert {"cancer_code", "regimen", "orr_pct"} <= set(df.columns)
    # all three regimens are actually present (not just PD-1)
    assert set(df["regimen"]) == {"PD-1", "PD-L1", "PD-1+CTLA-4"}
    assert (df["regimen"] == "PD-L1").sum() >= 10  # anti-PD-L1 is well-represented


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


def test_regimen_maps_cached():
    # _regimen_maps is memoized (same object back from the cache).
    assert ici._regimen_maps() is ici._regimen_maps()
