# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

import oncoref as cd
from oncoref import cancer_types


def test_resolve_common_name_alias():
    assert cancer_types.resolve_cancer_type("prostate") == "PRAD"
    assert cancer_types.resolve_cancer_type("melanoma") == "SKCM"


def test_resolve_canonical_code_passthrough():
    assert cancer_types.resolve_cancer_type("PRAD") == "PRAD"


def test_resolve_renamed_code():
    # Pre-rename codes resolve to the current canonical code.
    assert cancer_types.resolve_cancer_type("OS") == "SARC_OS"
    assert cancer_types.resolve_cancer_type("PANNET") == "NET_PANCREAS"


def test_resolve_display_name_case_insensitive():
    name = cd.CANCER_TYPE_NAMES["PRAD"]
    assert cancer_types.resolve_cancer_type(name.lower()) == "PRAD"


def test_resolve_none_passthrough():
    assert cancer_types.resolve_cancer_type(None) is None


def test_resolve_unknown_raises_strict():
    with pytest.raises(ValueError):
        cancer_types.resolve_cancer_type("not_a_real_cancer")


def test_resolve_unknown_nonstrict_returns_none():
    assert cancer_types.resolve_cancer_type("not_a_real_cancer", strict=False) is None


def test_canonical_cancer_code_is_pure():
    assert cancer_types.canonical_cancer_code("MID_NET") == "NET_MIDGUT"
    assert cancer_types.canonical_cancer_code("PRAD") == "PRAD"
    assert cancer_types.canonical_cancer_code("anything_else") == "anything_else"


def test_registry_has_core_columns():
    df = cancer_types.cancer_type_registry()
    for col in ("code", "name", "family", "primary_tissue", "parent_code"):
        assert col in df.columns
    assert "PRAD" in set(df["code"])


def test_cancer_type_info_assembles_derived_fields():
    info = cancer_types.cancer_type_info("prostate")
    assert info["code"] == "PRAD"
    assert info["name"]
    # derived fields pulled from the TMB + burden tables (cycle-safe lazy import)
    assert "burden_category" in info
    assert "tmb" in info
    assert info["burden_category"] == "prostate"


def test_cancer_type_info_none_passthrough():
    assert cancer_types.cancer_type_info(None) is None


def test_synonyms_roundtrip():
    syns = cancer_types.cancer_type_synonyms("PRAD")
    assert "prostate" in syns
    assert "PRAD" not in syns  # the code itself is excluded


def test_families_nonempty():
    fams = cancer_types.cancer_type_families()
    assert fams
    assert all(isinstance(v, str) and v for v in fams.values())


def test_cohort_aggregates_sarc_grand_union():
    members = cancer_types.cohort_aggregate_members("SARC")
    assert members
    assert "SARC" not in members  # no self-membership


def test_cohort_registry_is_validation_authority():
    ids = cancer_types.known_cohort_ids()
    assert ids
    assert isinstance(ids, frozenset)


def test_cohort_source_version_parses_ensembl_release():
    # The per-cohort source_version for auditing the canonical gene-ID space: a code
    # resolves to its source cohort, whose provenance records the harmonized Ensembl
    # release. The shipped cohorts are harmonized to Ensembl 112.
    assert cancer_types.cohort_source_version("LUAD") == "112"
    assert cancer_types.cohort_source_version("ACC") == "112"
    # an unknown code has no recorded source version (rather than raising)
    assert cancer_types.cohort_source_version("NOT_A_REAL_CODE") is None


def test_clear_caches_allows_reload(monkeypatch):
    # The names view caches the registry; clearing forces a re-read.
    assert "PRAD" in cd.CANCER_TYPE_NAMES
    cancer_types._clear_caches()
    assert "PRAD" in cd.CANCER_TYPE_NAMES


def test_every_registry_family_has_a_curated_display_name():
    # Drift guard: every registry family must have a curated label, not the
    # title-cased fallback. (This caught the stale 'cns'/'endocrine' keys left
    # after the registry split those into cns-*/endocrine-* families.)
    families = set(cancer_types.cancer_type_registry()["family"].dropna().astype(str))
    missing = sorted(f for f in families if f not in cancer_types._FAMILY_DISPLAY_NAMES)
    assert not missing, f"registry families without a curated display name: {missing}"


def test_lineage_group_resolution():
    # Coarse histogenesis rollup: family default + per-code override (inherited).
    assert cancer_types.cancer_lineage_group("LUAD") == "Epithelial"
    assert cancer_types.cancer_lineage_group("SARC_OS") == "Sarcoma"
    assert cancer_types.cancer_lineage_group("SKCM") == "Melanoma"
    assert cancer_types.cancer_lineage_group("NET_PANCREAS") == "Neuroendocrine"
    # NBL overrides its neuroendocrine family default -> Embryonal, inherited by subtypes.
    assert cancer_types.cancer_lineage_group("NBL") == "Embryonal"
    assert cancer_types.cancer_lineage_group("NBL_MYCNamp") == "Embryonal"
    assert cancer_types.cancer_lineage_group("not_a_real_cancer") is None


def test_every_registry_family_rolls_up_to_a_lineage_group():
    # Drift guard: the coarse rollup must cover every registry family, so a new
    # family can't silently yield a None lineage group.
    families = set(cancer_types.cancer_type_registry()["family"].dropna().astype(str))
    groups = cancer_types.cancer_lineage_groups()
    missing = sorted(f for f in families if f not in groups)
    assert not missing, f"registry families with no lineage group: {missing}"
