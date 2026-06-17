# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Normalization-reference gene families (#35, R-norm)."""

import pytest

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


def test_housekeeping_panel():
    df = gf.housekeeping_genes()
    assert {"Symbol", "Ensembl_Gene_ID"} <= set(df.columns)
    assert "ACTB" in set(df["Symbol"])  # canonical housekeeping gene
    assert gf.housekeeping_gene_ids()


def test_censored_references():
    assert gf.clean_tpm_censored_gene_ids()  # the clean_tpm_v4 censored set
    ref = gf.censored_gene_reference_tpm()
    assert ref and all(isinstance(v, float) and v >= 0 for v in ref.values())
