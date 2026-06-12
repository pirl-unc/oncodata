# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from cancerdata import cta


def test_set_sizes_reasonable():
    # Robust to curation refreshes: assert the sets are substantial and that
    # symbols and IDs agree in cardinality, rather than pinning exact counts
    # (which drift whenever the source databases or HPA version update).
    expressed = cta.CTA_gene_names()
    assert len(expressed) == len(cta.CTA_gene_ids())
    assert 150 < len(expressed) < len(cta.CTA_unfiltered_gene_names())
    assert len(cta.CTA_unfiltered_gene_ids()) > 250


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


def test_never_expressed_rescue_is_a_uniform_rule():
    # never_expressed CTAs with MODERATE confidence + STRICT reproductive RNA are
    # kept in the expressed set by a uniform rule (not a one-gene XAGE5 override).
    # XAGE5 is rescued, and so are its peers with the same signature.
    expressed = cta.CTA_gene_ids()
    assert "ENSG00000171405" in expressed  # XAGE5
    assert "MAGEA2B" in cta.CTA_gene_names()  # a same-signature peer, also kept
    # The rescue is exactly the rule, applied to every row.
    df = cta.cta_dataframe()
    rescued = df[cta._never_expressed_rescue_mask(df)]
    never = rescued["never_expressed"].astype(str).str.lower() == "true"
    kept = set(rescued.loc[never, "Ensembl_Gene_ID"].astype(str).str.split(".").str[0])
    assert kept and kept <= cta.CTA_gene_ids()


def test_non_cta_excluded_genes_dropped():
    # Histones / tubulins flagged out of the CTA universe entirely.
    df = cta.cta_dataframe()
    unversioned = set(df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0])
    assert unversioned.isdisjoint(cta.NON_CTA_EXCLUDED_GENE_IDS)
    assert cta.NON_CTA_EXCLUDED_GENE_IDS.isdisjoint(cta.CTA_unfiltered_gene_ids())


def test_no_histone_or_tubulin_survives_in_cta_universe():
    # The exclusion is a gene-family rule, not a hand-list: EVERY core histone
    # and alpha-tubulin candidate is dropped, so a sibling can't be left in (the
    # H1-6-vs-H2BC1 inconsistency the family rule fixed). Guards future drift.
    from cancerdata.gene_families import gene_family_ids
    from cancerdata.load_dataset import get_data

    raw = get_data("cancer-testis-antigens")
    uid = raw["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    histone_candidates = set(uid[uid.isin(gene_family_ids("histone"))])
    tubulin_candidates = set(uid[raw["Symbol"].str.match(r"^TUBA\d", na=False)])
    survivors = (histone_candidates | tubulin_candidates) & cta.CTA_unfiltered_gene_ids()
    assert not survivors, f"housekeeping structural genes left in the CTA universe: {survivors}"


def test_cgb8_not_deny_listed():
    # #20: CGB8 was hardcoded out as "placental hCG-beta" while its hCG-beta
    # siblings (CGB1/2/3/5/7) flow through the normal HPA filter. CGB8 must be
    # curated by that same filter, not a one-gene deny-list. It passes the filter
    # (protein REPRODUCTIVE), so it lands in the filtered set like CGB2.
    assert "ENSG00000213030" not in cta.NON_CTA_EXCLUDED_GENE_IDS
    assert "CGB8" in cta.CTA_unfiltered_gene_names()
    assert "CGB8" in cta.CTA_filtered_gene_names()


def test_evidence_has_no_ms_columns():
    # MS-runtime columns stay in the target-selection layer, not cancerdata.
    cols = set(cta.CTA_evidence().columns)
    assert not any(c.startswith("ms_") for c in cols)
    # but the HPA-derived restriction columns are present
    for c in ("passes_filters", "never_expressed", "protein_restriction", "rna_restriction"):
        assert c in cols


def test_shipped_restriction_is_hpa_only_synthesis():
    # Every shipped restriction/confidence pair must equal the HPA-only synthesis
    # of its own row — i.e. no MS contribution leaked into the bundled table.
    df = cta.CTA_evidence()
    for _, row in df.iterrows():
        tissue, conf = cta.synthesize_restriction(row)
        assert str(row["restriction"]) == tissue
        assert str(row["restriction_confidence"]) == conf


def test_synthesize_restriction_drops_ms():
    # A protein-SOMATIC row is SOMATIC regardless of any ms_restriction value —
    # confirms the synthesis ignores MS columns entirely.
    row = {
        "protein_restriction": "TESTIS",
        "protein_reliability": "Enhanced",
        "rna_restriction": "TESTIS",
        "rna_restriction_level": "STRICT",
        "ms_restriction": "RECURRENT_HEALTHY",  # would lower MS-aware confidence
    }
    tissue, conf = cta.synthesize_restriction(row)
    assert tissue == "TESTIS"
    assert conf == "HIGH"  # protein(1.5)+rna-agree(1.5) over 2 sources = 1.5 >= 1.2


def test_gene_id_to_name():
    m = cta.CTA_gene_id_to_name()
    assert len(m) == len(cta.CTA_gene_ids())
    assert all(not k.count(".") for k in m)  # unversioned keys
