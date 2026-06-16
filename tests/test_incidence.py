# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from oncodata import incidence
from oncodata import load_dataset as ld


def test_derived_burden_cache_invalidates_on_clear(monkeypatch):
    # The lru_cached burden maps must be cleared by load_dataset._clear_cache(),
    # so swapping a bundled fixture is reflected (test-isolation guard).
    incidence._tissue_burden_map()  # populate the cache
    real = incidence.get_data

    def fake(name, *a, **k):
        if name == "tissue-burden-map":
            return pd.DataFrame(
                {"primary_tissue": ["unobtanium"], "burden_category": ["xyzzy"], "scope": ["solid"]}
            )
        return real(name, *a, **k)

    monkeypatch.setattr(incidence, "get_data", fake)
    ld._clear_cache()
    try:
        assert incidence._tissue_burden_map() == {"unobtanium": "xyzzy"}
    finally:
        monkeypatch.undo()
        ld._clear_cache()  # restore real data for other tests


def test_burden_map_nonempty_floats():
    mapping = incidence.cancer_burden()
    assert mapping
    assert all(isinstance(v, float) for v in mapping.values())


def test_burden_metric_validation():
    with pytest.raises(ValueError):
        incidence.cancer_burden(metric="not_a_metric")


def test_burden_all_metrics_resolve():
    for metric in incidence._BURDEN_METRICS:
        mapping = incidence.cancer_burden(metric=metric)
        assert mapping


def test_burden_category_from_primary_tissue():
    assert incidence.burden_category("PRAD") == "prostate"
    assert incidence.burden_category("LUAD") == "lung"


def test_burden_category_sarcoma_bone_vs_soft():
    # Sarcoma family splits on primary_tissue: bone vs soft tissue.
    assert incidence.burden_category("SARC_OS") == "bone_and_joint"
    assert incidence.burden_category("SARC") == "soft_tissue_sarcoma"


def test_burden_category_override_table():
    # cancer-code-burden-map.csv carries the true ontology exceptions.
    overrides = incidence.cancer_code_burden_map()
    assert overrides
    for code, expected in overrides.items():
        assert incidence.burden_category(code) == expected


def test_burden_category_unknown_returns_none():
    assert incidence.burden_category("not_a_real_cancer") is None


def test_every_registry_code_resolves_to_a_burden_category():
    # Drift guard: the primary_tissue/family -> burden maps in incidence.py must
    # cover the registry's vocabulary. A new registry primary_tissue/family that
    # nothing maps would silently yield None here.
    from oncodata.cancer_types import cancer_type_registry

    registry = cancer_type_registry()
    unmapped = [
        code for code in registry["code"].astype(str) if incidence.burden_category(code) is None
    ]
    assert not unmapped, f"registry codes with no burden category: {unmapped}"


def test_family_burden_map_has_no_stale_families():
    # Drift guard for the data-driven family fallback: every family key must be a
    # current registry family. (This caught the stale 'cns' slug left after the
    # registry split cns -> cns-glial/-embryonal/... .)
    from oncodata.cancer_types import cancer_type_registry

    registry_families = set(cancer_type_registry()["family"].dropna().astype(str))
    stale = [f for f in incidence._family_burden_map() if f not in registry_families]
    assert not stale, f"family-burden-map keys not in the registry: {stale}"
