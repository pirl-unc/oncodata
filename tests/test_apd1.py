# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from oncoref import apd1


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
