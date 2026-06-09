# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from cancerdata import cta


def test_expressed_set_size():
    # 263 expressed CTAs (passes_filters AND not never_expressed) at HPA v23.
    # Locks the cancerdata list against the source-of-truth at port time.
    assert len(cta.CTA_gene_names()) == 263
    assert len(cta.CTA_gene_ids()) == 263


def test_unfiltered_universe_size():
    assert len(cta.CTA_unfiltered_gene_ids()) == 359


def test_set_relationships():
    expressed = cta.CTA_gene_names()
    filtered = cta.CTA_filtered_gene_names()
    unfiltered = cta.CTA_unfiltered_gene_names()
    assert expressed <= filtered <= unfiltered
    # never-expressed = filtered minus expressed
    assert cta.CTA_never_expressed_gene_names() == filtered - expressed
    # excluded = unfiltered minus filtered (fail reproductive restriction)
    assert cta.CTA_excluded_gene_names() == unfiltered - filtered


def test_canonical_ctas_present():
    expressed = cta.CTA_gene_names()
    for g in ("MAGEA4", "MAGEA1", "CTAG1B", "PRAME"):
        assert g in expressed


def test_manually_rescued_cta_kept():
    # XAGE5 is HPA never_expressed but manually rescued into the expressed set.
    assert "ENSG00000171405" in cta.CTA_gene_ids()


def test_non_cta_excluded_genes_dropped():
    # Histones / tubulins flagged out of the CTA universe entirely.
    df = cta.cta_dataframe()
    unversioned = set(df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0])
    assert unversioned.isdisjoint(cta.NON_CTA_EXCLUDED_GENE_IDS)
    assert cta.NON_CTA_EXCLUDED_GENE_IDS.isdisjoint(cta.CTA_unfiltered_gene_ids())


def test_evidence_has_no_ms_columns():
    # MS-runtime columns stay in the target-selection layer, not cancerdata.
    cols = set(cta.CTA_evidence().columns)
    assert not any(c.startswith("ms_") for c in cols)
    # but the HPA-derived restriction columns are present
    for c in ("passes_filters", "never_expressed", "protein_restriction", "rna_restriction"):
        assert c in cols


def test_gene_id_to_name():
    m = cta.CTA_gene_id_to_name()
    assert len(m) == len(cta.CTA_gene_ids())
    assert all(not k.count(".") for k in m)  # unversioned keys
