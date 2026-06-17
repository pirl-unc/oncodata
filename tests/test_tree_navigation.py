# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cancer-type tree navigation: ancestors / descendants / lineage / tree (#27, O3)."""

from oncoref import (
    cancer_type_ancestors,
    cancer_type_descendants,
    cancer_type_lineage,
    cancer_type_registry,
    cancer_type_subtypes_of,
    cancer_type_tree,
)


def test_ancestors_walks_up():
    assert cancer_type_ancestors("COAD_MSI") == ["COAD", "CRC"]


def test_ancestors_of_root_is_empty():
    assert cancer_type_ancestors("CRC") == []


def test_descendants_full_subtree():
    assert cancer_type_descendants("CRC") == [
        "COAD",
        "COAD_MSI",
        "COAD_MSS",
        "READ",
        "READ_MSI",
        "READ_MSS",
    ]


def test_descendants_include_self():
    assert cancer_type_descendants("CRC", include_self=True)[0] == "CRC"


def test_leaf_has_no_descendants():
    assert cancer_type_descendants("COAD_MSI") == []


def test_lineage_root_to_leaf():
    # The headline example: CRC -> COAD -> COAD_MSI.
    assert cancer_type_lineage("COAD_MSI") == ["CRC", "COAD", "COAD_MSI"]


def test_tree_subtree_shape():
    assert cancer_type_tree("CRC") == {
        "CRC": {
            "COAD": {"COAD_MSI": {}, "COAD_MSS": {}},
            "READ": {"READ_MSI": {}, "READ_MSS": {}},
        }
    }


def test_full_forest_covers_every_code_once():
    # Every registry code appears exactly once across the forest (roots + descendants).
    forest = cancer_type_tree()

    seen = []

    def _walk(node):
        for code, sub in node.items():
            seen.append(code)
            _walk(sub)

    _walk(forest)
    codes = set(cancer_type_registry()["code"])
    assert set(seen) == codes
    assert len(seen) == len(codes)  # no code reachable by two paths


def test_descendants_consistent_with_direct_subtypes():
    # Direct subtypes are exactly the depth-1 descendants.
    direct = set(cancer_type_subtypes_of("CRC"))
    desc = set(cancer_type_descendants("CRC"))
    assert direct <= desc
    assert direct == {"COAD", "READ"}
