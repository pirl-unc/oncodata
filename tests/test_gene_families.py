# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Normalization-reference gene families (#35, R-norm)."""

import pytest

import oncoref
from oncoref import gene_families as gf


def test_families_load_with_ids():
    for name in gf.gene_families():
        ids = gf.gene_family_ids(name)
        assert ids, f"{name} family is empty"
        assert all(i.startswith("ENSG") and "." not in i for i in ids)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown gene family"):
        gf.gene_family("not_a_family")
    with pytest.raises(ValueError, match="unknown gene family"):
        gf.gene_family_ids("not_a_family")


def test_technical_rna_is_union_of_its_families():
    union = set()
    for fam in ("mitochondrial", "numt_pseudogene", "rrna", "nuclear_retained_lncrna"):
        union |= gf.gene_family_ids(fam)
    assert gf.technical_rna_gene_ids() == union
    assert gf.technical_rna_gene_ids()  # non-empty


def test_clean_tpm_compartment_ids_follow_censored_table_categories():
    censored = gf.clean_tpm_censored_gene_ids()
    ribosomal = gf.clean_tpm_ribosomal_gene_ids()
    other_technical = gf.clean_tpm_other_technical_gene_ids()

    assert len(ribosomal) == 1764
    assert len(other_technical) == 1022
    assert ribosomal | other_technical == censored
    assert ribosomal.isdisjoint(other_technical)

    # RPL10AP1 is a ribosomal-protein pseudogene whose public censored-table category
    # is ribosomal_protein, so it belongs to the 16% budget, not the 9% remainder.
    assert "ENSG00000244691" in ribosomal
    assert "ENSG00000244691" not in other_technical


def test_clean_tpm_censoring_is_cta_safe():
    from oncoref import cta

    rpl10l = "ENSG00000165496"
    cta_universe = {str(g).split(".")[0] for g in cta.cta_unfiltered_gene_ids()}
    censored = gf.clean_tpm_censored_gene_ids()

    assert rpl10l in gf.gene_family_ids("ribosomal_protein")
    assert rpl10l in cta_universe
    assert rpl10l not in gf.clean_tpm_ribosomal_gene_ids()
    assert rpl10l not in gf.clean_tpm_other_technical_gene_ids()
    assert rpl10l not in censored
    assert cta_universe.isdisjoint(censored)


def test_housekeeping_panel():
    df = gf.housekeeping_genes()
    assert {"Symbol", "Ensembl_Gene_ID"} <= set(df.columns)
    assert "ACTB" in set(df["Symbol"])  # canonical housekeeping gene
    assert gf.housekeeping_gene_ids()


def test_clean_tpm_biological_housekeeping_panel_contract():
    from oncoref.gene_ids import is_protein_coding_gene

    full = gf.clean_tpm_biological_housekeeping_genes(primary_only=False)
    primary = gf.clean_tpm_biological_housekeeping_genes()

    assert len(full) == 47
    assert len(primary) == 30
    assert set(primary["Ensembl_Gene_ID"]) < set(full["Ensembl_Gene_ID"])
    assert {
        "Symbol",
        "Ensembl_Gene_ID",
        "hpa_tissue_count",
        "min_tpm",
        "mean_tpm",
        "cv",
        "max_tpm",
        "clean_tpm_component",
        "primary_panel",
        "selection_note",
    } <= set(full.columns)

    assert set(full["clean_tpm_component"]) == {"biological"}
    assert full["hpa_tissue_count"].eq(50).all()
    assert full["min_tpm"].ge(100).all()
    assert full["cv"].lt(0.5).all()
    assert full["Ensembl_Gene_ID"].map(is_protein_coding_gene).all()

    censored = gf.clean_tpm_censored_gene_ids()
    assert gf.clean_tpm_biological_housekeeping_gene_ids().isdisjoint(censored)
    assert gf.clean_tpm_biological_housekeeping_gene_ids(primary_only=False).isdisjoint(censored)


def test_clean_tpm_housekeeping_helpers_are_top_level_exports():
    for name in (
        "clean_tpm_biological_housekeeping_gene_ids",
        "clean_tpm_biological_housekeeping_genes",
        "clean_tpm_ribosomal_gene_ids",
        "clean_tpm_other_technical_gene_ids",
    ):
        assert hasattr(oncoref, name)
        assert name in oncoref.__all__


def test_clean_tpm_housekeeping_primary_excludes_legacy_ribosomal_references():
    primary_symbols = set(gf.clean_tpm_biological_housekeeping_genes()["Symbol"])
    full_symbols = set(gf.clean_tpm_biological_housekeeping_genes(primary_only=False)["Symbol"])

    assert {"SUMO2", "EEF1A1", "COX7A2"} <= primary_symbols
    assert "PPIA" in full_symbols and "PPIA" not in primary_symbols
    assert {
        "RPLP0",
        "RPL13A",
        "RPS18",
        "RPS13",
        "RPL19",
        "RPL27",
        "RPS27",
    }.isdisjoint(primary_symbols)


def test_legacy_qpcr_housekeeping_aliases_preserve_historical_panel():
    assert gf.legacy_qpcr_housekeeping_genes().equals(gf.housekeeping_genes())
    assert gf.legacy_qpcr_housekeeping_gene_ids() == gf.housekeeping_gene_ids()
    assert "ACTB" in set(gf.legacy_qpcr_housekeeping_genes()["Symbol"])


def test_censored_references():
    assert gf.clean_tpm_censored_gene_ids()  # the clean_tpm_v4 censored set
    ref = gf.censored_gene_reference_tpm()
    assert ref and all(isinstance(v, float) and v >= 0 for v in ref.values())
