# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-type gene biology + fusion-rule tables (#35, R-onto)."""

from oncodata import cancer_genes as cg
from oncodata import fusions as f


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
