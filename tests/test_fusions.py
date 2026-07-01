# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Driver fusions per cancer type + reverse lookups (#27, O4)."""

import pytest

from oncoref import (
    cancer_fusions,
    cancer_fusions_df,
    cancer_type_registry,
    cancer_types_with_fusion,
    fusion_partners,
    fusions,
    protein_family,
)


def test_fusions_for_subtype():
    # The headline example: the alveolar-RMS subtype's PAX-FOXO1 fusion.
    arms = cancer_fusions("SARC_RMS_ARMS")
    pairs = set(zip(arms["gene_5prime"], arms["gene_3prime"]))
    assert ("PAX3", "FOXO1") in pairs
    assert ("PAX7", "FOXO1") in pairs


def test_astb_fusion_side_tables_match_registry_driver():
    astb = cancer_fusions("ASTB", defining_only=True)
    pairs = set(zip(astb["gene_5prime"], astb["gene_3prime"]))
    assert pairs == {("MN1", "BEND2")}
    assert cancer_types_with_fusion("MN1-BEND2", defining_only=True) == ["ASTB"]
    assert fusions.fusion_surrogate_genes_for_cancer("ASTB") == ["BEND2"]


def test_defining_only_filter():
    ews = cancer_fusions("SARC_EWS", defining_only=True)
    assert (ews["is_defining"].astype(str).str.lower() == "true").all()
    assert "EWSR1" in set(ews["gene_5prime"])


def test_reverse_lookup_by_fusion():
    assert cancer_types_with_fusion("EWSR1-FLI1") == ["SARC_EWS"]
    # `::` separator also accepted.
    assert cancer_types_with_fusion("EWSR1::FLI1") == ["SARC_EWS"]


def test_reverse_lookup_by_partner_family():
    fet = cancer_types_with_fusion(partner_family="FET")
    assert "SARC_EWS" in fet and "SARC_DSRCT" in fet


def test_reverse_lookup_requires_exactly_one_arg():
    with pytest.raises(ValueError, match="exactly one"):
        cancer_types_with_fusion("EWSR1-FLI1", partner="EWSR1")
    with pytest.raises(ValueError, match="exactly one"):
        cancer_types_with_fusion()


def test_fusion_partners_promiscuous():
    partners = fusion_partners("EWSR1")
    assert {"FLI1", "ERG"} <= partners


def test_fusion_partners_bad_side():
    with pytest.raises(ValueError, match="side must be"):
        fusion_partners("EWSR1", side="middle")


def test_protein_family():
    assert protein_family("PAX3") == "PAX"
    assert protein_family("FLI1") == "ETS"
    assert protein_family("NOT_A_GENE") is None


def test_nan_query_does_not_match_fusion_negative_rows():
    # The "(none)" fusion-negative rows have NaN gene names; a 'NAN'/'nan' query
    # must NOT match them (regression: .astype(str) turned NaN into "NAN").
    assert cancer_types_with_fusion(partner="NAN") == []
    assert cancer_types_with_fusion(partner="nan") == []
    assert cancer_types_with_fusion("NAN-NAN") == []
    assert fusion_partners("NAN") == set()
    assert protein_family("NAN") is None


def test_every_fusion_code_is_a_registry_code():
    # A fusion row must never reference a cancer code that doesn't exist.
    fusion_codes = set(cancer_fusions_df()["cancer_code"])
    codes = set(cancer_type_registry()["code"])
    missing = sorted(fusion_codes - codes)
    assert not missing, f"fusion cancer_codes not in the registry: {missing}"


def test_known_wrong_fusion_pmids_are_replaced():
    """Issue #160 examples: real PMIDs that resolved to unrelated papers."""

    fusions_df = cancer_fusions_df()
    known_bad_pmids = {
        "PMID:7951326",
        "PMID:7954420",
        "PMID:9590769",
        "PMID:21885844",
        "PMID:9537325",
        "PMID:16462738",
        "PMID:9192848",
        "PMID:11427703",
        "PMID:8316832",
        "PMID:1565502",
        "PMID:6304885",
        "PMID:3929080",
        "PMID:6262918",
        "PMID:9596662",
        "PMID:10866930",
    }
    observed_pmids = set(fusions_df["pmid"].dropna())
    assert known_bad_pmids.isdisjoint(observed_pmids)

    expected_pmids = {
        ("SARC_SYN", "SS18-SSX", "SSX1"): "PMID:7951320",
        ("SARC_SYN", "SS18-SSX", "SSX2"): "PMID:7951320",
        ("SARC_SYN", "SS18-SSX", "SSX4"): "PMID:11368913",
        ("SARC_DSRCT", "EWSR1-WT1", "WT1"): "PMID:7862627",
        ("SARC_IFS", "ETV6-NTRK3", "NTRK3"): "PMID:9462753",
        ("SARC_EHE", "WWTR1-CAMTA1", "CAMTA1"): "PMID:21885404",
        ("SARC_EMC", "EWSR1-NR4A3", "NR4A3"): "PMID:18855877",
        ("SARC_IMT", "RANBP2-ALK", "ALK"): "PMID:24034896",
        ("SARC_DFSP", "COL1A1-PDGFB", "PDGFB"): "PMID:31949795",
        ("SARC_ESS_LG", "JAZF1-SUZ12", "SUZ12"): "PMID:16049311",
        ("LAML", "CBFB-MYH11", "MYH11"): "PMID:8142642",
        ("LAML", "DEK-NUP214", "NUP214"): "PMID:32526729",
        ("CML", "BCR-ABL1", "ABL1"): "PMID:40360311",
        ("FL", "BCL2-IGH", "BCL2"): "PMID:18684042",
        ("BL", "MYC-IGH", "MYC"): "PMID:23673335",
        ("MM", "CCND1-IGH", "CCND1"): "PMID:35982976",
        ("THCA", "PAX8-PPARG", "PPARG"): "PMID:25069464",
    }
    for (cancer_code, fusion_family, gene_3prime), expected_pmid in expected_pmids.items():
        row = fusions_df[
            (fusions_df["cancer_code"] == cancer_code)
            & (fusions_df["fusion_family"] == fusion_family)
            & (fusions_df["gene_3prime"] == gene_3prime)
        ]
        assert len(row) == 1
        assert row.iloc[0]["pmid"] == expected_pmid
