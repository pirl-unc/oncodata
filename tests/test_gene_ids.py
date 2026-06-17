# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Gene-id / symbol resolution references (#35, R-resolve)."""

from oncoref import gene_ids as g


def test_unversioned_normalizer():
    assert g.unversioned("ENSG00000141510.5") == "ENSG00000141510"
    assert g.unversioned("ENSG00000141510") == "ENSG00000141510"  # idempotent on bare
    assert g.unversioned("  ENSG00000141510.5  ") == "ENSG00000141510"  # strips whitespace
    assert g.unversioned("ENSG00000141510") == g.unversioned(" ENSG00000141510")  # padded == bare


def test_ensembl_id_alias_resolution():
    aliases = g.ensembl_id_aliases()
    assert aliases  # non-empty
    alt, primary = next(iter(aliases.items()))
    assert g.resolve_ensembl_id(alt) == primary
    assert g.resolve_ensembl_id(f"{alt}.7") == primary  # version-insensitive
    # a non-alias id passes through unchanged (unversioned)
    assert g.resolve_ensembl_id("ENSG99999999999.3") == "ENSG99999999999"


def test_symbol_synonym_resolution():
    syn = g.symbol_synonyms()
    assert len(syn) > 1000
    alias, official = next(iter(syn.items()))
    assert g.resolve_symbol(alias) == official
    assert g.resolve_symbol(alias.lower()) == official  # case-insensitive
    # an unknown / already-official symbol passes through
    assert g.resolve_symbol("NOT_A_REAL_GENE") == "NOT_A_REAL_GENE"


def test_loaders_have_expected_columns():
    assert {"transcript_id", "ensembl_gene_id"} <= set(g.extra_transcript_mappings().columns)
    assert {"ensembl_gene_id", "n_members"} <= set(g.cdna_identical_groups().columns)
    assert "reason" in g.proteoform_collapse_overrides().columns
