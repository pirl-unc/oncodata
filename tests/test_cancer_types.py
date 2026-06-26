# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

import oncoref as cd
from oncoref import cancer_types

HPA_RNA_V23_TISSUES = {
    "adipose tissue",
    "adrenal gland",
    "amygdala",
    "appendix",
    "basal ganglia",
    "bone marrow",
    "breast",
    "cerebellum",
    "cerebral cortex",
    "cervix",
    "choroid plexus",
    "colon",
    "duodenum",
    "endometrium",
    "epididymis",
    "esophagus",
    "fallopian tube",
    "gallbladder",
    "heart muscle",
    "hippocampal formation",
    "hypothalamus",
    "kidney",
    "liver",
    "lung",
    "lymph node",
    "midbrain",
    "ovary",
    "pancreas",
    "parathyroid gland",
    "pituitary gland",
    "placenta",
    "prostate",
    "rectum",
    "retina",
    "salivary gland",
    "seminal vesicle",
    "skeletal muscle",
    "skin",
    "small intestine",
    "smooth muscle",
    "spinal cord",
    "spleen",
    "stomach",
    "testis",
    "thymus",
    "thyroid gland",
    "tongue",
    "tonsil",
    "urinary bladder",
    "vagina",
}


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
    assert cancer_types.cancer_lineage_group("ASTB") == "CNS"
    # NBL overrides its neuroendocrine family default -> Embryonal, inherited by subtypes.
    assert cancer_types.cancer_lineage_group("NBL") == "Embryonal"
    assert cancer_types.cancer_lineage_group("NBL_MYCNamp") == "Embryonal"
    assert cancer_types.cancer_lineage_group("not_a_real_cancer") is None


def test_astb_registry_row_for_trufflepig_parity():
    assert cancer_types.resolve_cancer_type("ASTB") == "ASTB"
    assert cancer_types.resolve_cancer_type("astroblastoma (mn1-altered)") == "ASTB"

    raw = cancer_types.cancer_type_registry().set_index("code").loc["ASTB"]
    assert bool(raw["pediatric"]) is True
    assert raw["fusion_driven"] == "defining"
    assert raw["fusion_driver"] == "MN1-BEND2"

    row = cancer_types.cancer_type_records(["ASTB"]).iloc[0]
    assert row["name"] == "Astroblastoma (MN1-altered)"
    assert row["family"] == "cns-glial"
    assert row["primary_tissue"] == "cerebrum"
    assert row["source_cohort"] == "LITERATURE_CURATED"
    assert row["source_pmid"] == "PMID:36604386"
    assert row["lineage_group"] == "CNS"
    assert row["normal_tissue_code"] == "cerebral_cortex"
    assert row["hpa_tissues"] == ("cerebral cortex",)
    assert bool(row["has_expression_matrix"]) is False


def test_every_registry_family_rolls_up_to_a_lineage_group():
    # Drift guard: the coarse rollup must cover every registry family, so a new
    # family can't silently yield a None lineage group.
    families = set(cancer_types.cancer_type_registry()["family"].dropna().astype(str))
    groups = cancer_types.cancer_lineage_groups()
    missing = sorted(f for f in families if f not in groups)
    assert not missing, f"registry families with no lineage group: {missing}"


def test_cancer_type_records_query_hierarchy_and_molecular_groups():
    crc = cancer_types.cancer_type_records(under="CRC")
    assert crc["code"].tolist() == [
        "CRC",
        "COAD",
        "READ",
        "COAD_MSI",
        "COAD_MSS",
        "READ_MSI",
        "READ_MSS",
    ]
    assert cancer_types.cancer_type_codes(subtype_group="MSI", under="CRC") == [
        "COAD_MSI",
        "READ_MSI",
    ]
    epithelial_msi = cancer_types.cancer_type_records(
        subtype_group="MSI", lineage_group="Epithelial"
    )
    assert {"COAD_MSI", "READ_MSI", "UCEC_MSI"} <= set(epithelial_msi["code"])
    assert set(epithelial_msi["lineage_group"]) == {"Epithelial"}


def test_cancer_type_records_empty_selection_stays_empty():
    empty = cancer_types.cancer_type_records([])
    assert empty.empty
    assert list(empty.columns) == list(cancer_types.cancer_type_records(["PRAD"]).columns)
    assert cancer_types.cancer_type_records([], under="CRC").empty
    assert cancer_types.cancer_type_siblings("CRC").empty


def test_cancer_type_path_makes_semantic_levels_explicit():
    path = cancer_types.cancer_type_path("COAD_MSI")
    assert path[["kind", "code"]].apply(tuple, axis=1).tolist() == [
        ("lineage_group", "Epithelial"),
        ("family", "carcinoma-gi"),
        ("cancer_type", "CRC"),
        ("cancer_type", "COAD"),
        ("cancer_type", "COAD_MSI"),
    ]
    assert path.iloc[-1]["normal_tissue_code"] == "colon"


def test_cancer_type_siblings_use_parent_hierarchy():
    siblings = cancer_types.cancer_type_siblings("COAD")
    assert siblings["code"].tolist() == ["READ"]
    molecular_siblings = cancer_types.cancer_type_siblings("COAD_MSI")
    assert molecular_siblings["code"].tolist() == ["COAD_MSS"]


def test_cancer_type_records_include_evidence_expression_and_normal_tissue():
    records = cancer_types.cancer_type_records(["COAD_MSI", "CRC_MSI"]).set_index("code")
    assert records.loc["COAD_MSI", "evidence_source_code"] == "CRC_MSI"
    assert records.loc["COAD_MSI", "evidence_source_kind"] == "source_scope"
    assert records.loc["COAD_MSI", "normal_tissue_code"] == "colon"
    assert records.loc["COAD_MSI", "hpa_tissues"] == ("colon",)
    assert bool(records.loc["COAD_MSI", "has_expression_matrix"]) is True

    assert records.loc["CRC_MSI", "evidence_source_code"] == "CRC_MSI"
    assert records.loc["CRC_MSI", "evidence_source_kind"] == "direct"
    assert records.loc["CRC_MSI", "normal_tissue_code"] == "colorectum"
    assert records.loc["CRC_MSI", "hpa_tissues"] == ("colon", "rectum")
    assert bool(records.loc["CRC_MSI", "has_expression_matrix"]) is False


def test_normal_tissue_map_uses_hpa_rna_v23_tissue_names():
    tissue_map = cancer_types.cancer_normal_tissue_map().set_index("primary_tissue")
    used = {
        str(tissue).lower() for tissues in tissue_map["hpa_tissues"] for tissue in (tissues or ())
    }
    assert used <= HPA_RNA_V23_TISSUES
    assert tissue_map.loc["soft_tissue", "hpa_tissues"] == ()
    assert tissue_map.loc["soft_tissue", "match_confidence"] == "unresolved"
    assert tissue_map.loc["oral_cavity", "hpa_tissues"] == ("tongue",)
    assert tissue_map.loc["pharynx", "hpa_tissues"] == ("tonsil",)


def test_cancer_type_reference_data_joins_scalar_oncoref_metrics():
    refs = cancer_types.cancer_type_reference_data(["COAD_MSI", "READ_MSI"]).set_index("code")
    assert refs.loc["COAD_MSI", "evidence_source_code"] == "CRC_MSI"
    assert refs.loc["COAD_MSI", "ici_response_source_code"] == "CRC_MSI"
    assert refs.loc["COAD_MSI", "ici_inheritance_kind"] == "source_scope"
    assert refs.loc["COAD_MSI", "tmb"] == 46.0
    assert refs.loc["READ_MSI", "normal_tissue_code"] == "rectum"
    assert bool(refs.loc["READ_MSI", "has_expression_matrix"]) is True
    assert refs.loc["READ_MSI", "us_incidence_pct"] is not None


def test_cancer_type_reference_data_honors_inherit_false_for_tmb():
    refs = cancer_types.cancer_type_reference_data(["LUAD_KRAS"], inherit=False).set_index("code")
    assert refs.loc["LUAD_KRAS", "tmb"] is None


def test_matched_normal_tissue_expression_uses_hpa_tissue_match(monkeypatch):
    import pandas as pd

    from oncoref import hpa

    hpa_fixture = pd.DataFrame(
        {
            "Gene": ["ENSG1", "ENSG1", "ENSG2"],
            "Gene name": ["G1", "G1", "G2"],
            "Tissue": ["colon", "rectum", "colon"],
            "nTPM": [10.0, 20.0, 30.0],
        }
    )
    monkeypatch.setattr(hpa, "hpa_rna_consensus", lambda: hpa_fixture.copy())
    out = cancer_types.matched_normal_tissue_expression("COAD", genes="G1")
    assert out["cancer_code"].unique().tolist() == ["COAD"]
    assert out["normal_tissue_code"].unique().tolist() == ["colon"]
    assert out["Tissue"].tolist() == ["colon"]
    assert out["nTPM"].tolist() == [10.0]


def test_matched_normal_tissue_missing_input_is_unresolved():
    assert cancer_types.matched_normal_tissue(None) is None
    assert cancer_types.matched_normal_tissue("not_a_real_cancer") is None
    assert cancer_types.matched_normal_tissue_expression(None).empty
    assert cancer_types.matched_normal_tissue_expression("not_a_real_cancer").empty


def test_matched_normal_tissue_expression_uses_hpa_rna_consensus_names(monkeypatch):
    import pandas as pd

    from oncoref import hpa

    hpa_fixture = pd.DataFrame(
        {
            "Gene": ["ENSG1", "ENSG1", "ENSG1", "ENSG1", "ENSG1", "ENSG1"],
            "Gene name": ["G1", "G1", "G1", "G1", "G1", "G1"],
            "Tissue": ["cervix", "endometrium", "skin", "stomach", "tongue", "tonsil"],
            "nTPM": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    monkeypatch.setattr(hpa, "hpa_rna_consensus", lambda: hpa_fixture.copy())

    assert cancer_types.matched_normal_tissue_expression("CESC", genes="G1")["Tissue"].tolist() == [
        "cervix"
    ]
    assert cancer_types.matched_normal_tissue_expression("UCEC", genes="G1")["Tissue"].tolist() == [
        "endometrium"
    ]
    assert cancer_types.matched_normal_tissue_expression("SKCM", genes="G1")["Tissue"].tolist() == [
        "skin"
    ]
    assert cancer_types.matched_normal_tissue_expression("STAD", genes="G1")["Tissue"].tolist() == [
        "stomach"
    ]
    assert cancer_types.matched_normal_tissue_expression("HNSC_HPVneg", genes="G1")[
        "Tissue"
    ].tolist() == ["tongue"]
    assert cancer_types.matched_normal_tissue_expression("HNSC", genes="G1")["Tissue"].tolist() == [
        "tonsil"
    ]
    assert cancer_types.matched_normal_tissue_expression("SARC", genes="G1").empty
    assert cancer_types.matched_normal_tissue_expression("NPC", genes="G1").empty
