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


def test_alias_table_is_migration_aware_and_acyclic():
    # The map covers both alt-haplotype/patch copies and cross-release id turnover, and
    # must be a clean forest: no self-maps, no duplicate keys, and no chains (a target
    # that is itself an alias key) so a single lookup always lands on a canonical id.
    aliases = g.ensembl_id_aliases()
    keys, targets = set(aliases), set(aliases.values())
    assert not (keys & targets), "alias targets must not themselves be alias keys (no chains)"
    assert all(k != v for k, v in aliases.items()), "no self-maps"
    # cross-release migration: GRCh37 GGNBP2 retired -> its current primary-assembly id
    assert g.resolve_ensembl_id("ENSG00000005955") == "ENSG00000278311"
    # resolution is idempotent (targets are canonical, never themselves keys)
    for k in list(aliases)[:2000]:
        assert g.resolve_ensembl_id(g.resolve_ensembl_id(k)) == g.resolve_ensembl_id(k)


def test_every_alias_target_is_in_the_canonical_space():
    # resolve_ensembl_id must always land in the canonical gene space, so a resolved id
    # has a biotype and passes is_canonical_gene — otherwise the migration map and the
    # space artifact disagree about what "canonical" means.
    space = set(g.canonical_gene_space()["ensembl_gene_id"])
    targets = set(g.ensembl_id_aliases().values())
    assert targets <= space, sorted(targets - space)[:5]


def test_canonical_gene_id_any_identifier():
    # The unified entry point (oncoref#135 item 1): any Ensembl id -> canonical ENSG.
    assert g.canonical_gene_id("ENSG00000005955") == "ENSG00000278311"  # old GRCh37 id
    assert g.canonical_gene_id("ENSG00000005955.7") == "ENSG00000278311"  # version-insensitive
    assert g.canonical_gene_id("ENSG00000278311") == "ENSG00000278311"  # already canonical
    assert g.canonical_gene_id("") is None and g.canonical_gene_id("   ") is None
    assert g.canonical_gene_ids(["ENSG00000005955", "ENSG00000278311"]) == [
        "ENSG00000278311",
        "ENSG00000278311",
    ]


def test_symbol_synonym_resolution():
    syn = g.symbol_synonyms()
    assert len(syn) > 1000
    alias, official = next(iter(syn.items()))
    assert g.resolve_symbol(alias) == official
    assert g.resolve_symbol(alias.lower()) == official  # case-insensitive
    # an unknown / already-official symbol passes through
    assert g.resolve_symbol("NOT_A_REAL_GENE") == "NOT_A_REAL_GENE"


def test_canonical_gene_space_and_biotype():
    sp = g.canonical_gene_space()
    assert {"ensembl_gene_id", "symbol", "biotype", "seqname"} <= set(sp.columns)
    assert sp["ensembl_gene_id"].is_unique  # one canonical row per gene
    assert (sp["biotype"] == "protein_coding").sum() > 19_000  # the ~20k coding set
    assert sp["seqname"].isin({*(str(i) for i in range(1, 23)), "X", "Y", "MT"}).all()

    # biotype / coding lookups, resolving an old/alt id through the migration map first
    assert g.gene_biotype("ENSG00000141510") == "protein_coding"  # TP53
    assert g.is_protein_coding_gene("ENSG00000141510")
    assert not g.is_protein_coding_gene("ENSG00000251562")  # MALAT1 (lncRNA)
    assert g.gene_biotype("ENSG00000005955") == "protein_coding"  # old GGNBP2 id -> resolved
    # membership: a real gene is in the space; a non-Ensembl/garbage id is not
    assert g.is_canonical_gene("ENSG00000141510")
    assert not g.is_canonical_gene("ENSG99999999999")


def test_loaders_have_expected_columns():
    assert {"transcript_id", "ensembl_gene_id"} <= set(g.extra_transcript_mappings().columns)
    assert {"ensembl_gene_id", "n_members"} <= set(g.cdna_identical_groups().columns)
    assert "reason" in g.proteoform_collapse_overrides().columns
