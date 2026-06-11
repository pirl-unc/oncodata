# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Genome-wide proteoform registry + CTA-subset consistency (issue #12)."""

import pytest

from cancerdata.proteoforms import (
    proteoform_group_map,
    proteoform_groups,
    proteoform_symbol_map,
)


def test_genome_scope_is_a_superset_of_cta():
    cta = proteoform_group_map(scope="cta")
    genome = proteoform_group_map(scope="genome")
    assert len(genome) > len(cta)  # genome-wide finds many more families


def test_invalid_scope_raises():
    with pytest.raises(ValueError, match="scope must be one of"):
        proteoform_groups(scope="nonsense")


def test_genome_schema_matches_cta():
    assert list(proteoform_groups(scope="genome").columns) == list(
        proteoform_groups(scope="cta").columns
    )


def test_n_members_matches_actual_group_size():
    df = proteoform_groups(scope="genome")
    sizes = df.groupby("proteoform_id")["member_gene_id"].nunique()
    declared = df.groupby("proteoform_id")["n_members"].first()
    assert (sizes == declared).all()


def test_no_empty_or_nan_member_symbols():
    # Genome scope includes unsymbolled genes; the generator fills the gene id so
    # the symbol map never carries an empty string or a literal "nan".
    for members in proteoform_symbol_map(scope="genome").values():
        assert all(m and m != "nan" for m in members), members


def test_no_duplicate_member_gene_ids():
    # A gene can belong to at most one identical-protein group.
    df = proteoform_groups(scope="genome")
    dups = df["member_gene_id"][df["member_gene_id"].duplicated()]
    assert dups.empty, f"gene in multiple genome groups: {sorted(set(dups))}"


def test_every_cta_group_maps_into_one_genome_group():
    # The CTA registry is a strict subset: each CTA group's members must all land
    # in a single genome group, and that genome group must contain them all
    # (genome scope can MERGE in extra non-CTA paralogs, never SPLIT a CTA group).
    cta = proteoform_groups(scope="cta")
    genome = proteoform_groups(scope="genome")
    gene_to_genome = dict(zip(genome["member_gene_id"], genome["proteoform_id"]))
    genome_members = genome.groupby("proteoform_id")["member_gene_id"].apply(set).to_dict()

    for label, sub in cta.groupby("proteoform_id"):
        ids = set(sub["member_gene_id"])
        glabels = {gene_to_genome.get(g) for g in ids}
        assert None not in glabels, f"{label}: CTA member absent from genome registry"
        assert len(glabels) == 1, f"{label}: CTA group split across genome groups {glabels}"
        assert ids <= genome_members[glabels.pop()], f"{label}: genome group missing members"
