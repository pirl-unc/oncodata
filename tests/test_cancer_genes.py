# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-type gene biology + fusion-rule tables (#35, R-onto)."""

from oncoref import cancer_genes as cg
from oncoref import fusions as f


def _split_refs(value):
    return {x.strip() for x in str(value).split(";") if x.strip()}


def test_viral_antigens():
    assert cg.cancer_viral_antigens("HPV") == ["E6", "E7"]
    assert cg.cancer_viral_antigens("not_a_virus") == []
    all_map = cg.cancer_viral_antigens()
    assert "HPV" in all_map
    # reverse lookup: a cervical/HPV cohort should surface HPV antigens
    pairs = cg.viral_antigens_for_cancer("CESC")
    assert any(v == "HPV" for v, _ in pairs)


def test_key_genes_roles():
    df = cg.cancer_key_genes_df()
    assert {"cancer_code", "symbol", "role"} <= set(df.columns)
    # PRAD biomarkers should include AR (androgen receptor)
    biomarkers = cg.cancer_biomarker_genes("PRAD")
    assert "AR" in biomarkers
    targets = cg.cancer_therapy_targets("PRAD")
    assert (targets["role"].astype(str) == "target").all()


def test_type_gene_sets():
    sets = cg.cancer_type_gene_sets("PRAD")
    assert sets  # non-empty role->{ensembl:symbol}
    assert all(isinstance(v, dict) for v in sets.values())


def test_narrative_and_rule_loaders():
    assert {"set_name", "members"} <= set(cg.narrative_gene_sets_df().columns)
    assert "rule_id" in cg.disease_state_rules_df().columns
    assert "pair_id" in cg.degenerate_subtype_pairs_df().columns
    assert {"Symbol", "Ensembl_Gene_ID"} <= set(cg.cancer_driver_genes_df().columns)


def test_fusion_rule_tables():
    assert {"cancer_code", "gene_a", "gene_b"} <= set(f.rare_cancer_fusion_rules_df().columns)
    assert {"fusion_class", "surrogate_gene"} <= set(f.fusion_surrogate_expression_df().columns)
    assert {"gene_a", "gene_b"} <= set(f.fusion_expression_effect_rules_df().columns)
    assert isinstance(f.fusion_surrogate_genes_for_cancer("SARC_EWS"), list)


def test_mcl_diagnostic_references_are_source_anchored():
    bad_pmids = {
        "PMID:9500537",
        "PMID:8049438",
        "PMID:20554603",
        "PMID:14983945",
        "PMID:12438234",
        "PMID:18832546",
    }

    key = cg.cancer_key_genes_df()
    small_b_cell_key = key[key["cancer_code"].isin(["CLL", "MCL"])]
    all_key_refs = set().union(*(_split_refs(v) for v in small_b_cell_key["source"]))
    assert bad_pmids.isdisjoint(all_key_refs)

    mcl_key = key[key["cancer_code"] == "MCL"]
    ccnd1 = mcl_key[mcl_key["symbol"] == "CCND1"]
    assert len(ccnd1) == 1
    ccnd1 = ccnd1.iloc[0]
    assert {"PMID:34114641", "PMID:40381701"} <= _split_refs(ccnd1["source"])
    assert "pathognomonic" not in str(ccnd1["rationale"]).lower()
    assert "cyclin D1 overexpression" in str(ccnd1["rationale"])

    sox11 = mcl_key[mcl_key["symbol"] == "SOX11"]
    assert len(sox11) == 1
    sox11 = sox11.iloc[0]
    assert sox11["source"] == "PMID:19801969"
    assert "~95%" in str(sox11["rationale"])

    cll_key = key[key["cancer_code"] == "CLL"]
    for symbol in ("CD5", "CD23"):
        row = cll_key[cll_key["symbol"] == symbol]
        assert len(row) == 1
        assert row.iloc[0]["source"] == "PMID:32249238"

    fusion = f.cancer_fusions("MCL")
    ccnd1_igh = fusion[fusion["fusion_family"] == "CCND1-IGH"]
    assert len(ccnd1_igh) == 1
    ccnd1_igh = ccnd1_igh.iloc[0]
    assert ccnd1_igh["pmid"] == "PMID:40381701"
    assert ccnd1_igh["frequency"] == "~95%"
    assert "unverified" not in str(ccnd1_igh["notes"]).lower()

    pair = cg.degenerate_subtype_pairs_df()
    pair = pair[pair["pair_id"] == "CLL_vs_MCL_vs_FL"]
    assert len(pair) == 1
    pair = pair.iloc[0]
    assert bad_pmids.isdisjoint(_split_refs(pair["refs"]))
    assert {"PMID:34114641", "PMID:40381701", "PMID:23897248", "PMID:32249238"} <= _split_refs(
        pair["refs"]
    )
