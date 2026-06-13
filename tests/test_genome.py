# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib.util

import pandas as pd
import pytest

# Gated on the optional [genome] extra + an installed human Ensembl release.
_HAS_PYENSEMBL = importlib.util.find_spec("pyensembl") is not None
if _HAS_PYENSEMBL:
    from cancerdata import genome as gx

    _HAS_RELEASE = bool(gx.genomes())
else:
    _HAS_RELEASE = False

pytestmark = pytest.mark.skipif(
    not _HAS_RELEASE, reason="pyensembl + a human Ensembl release not installed"
)


def test_symbol_to_canonical_id():
    gid, sym = gx.canonical_gene_id_and_name("TP53")
    assert gid == "ENSG00000141510"
    assert sym == "TP53"


def test_alias_resolution():
    # NY-ESO-1 is a curated display alias for CTAG1B.
    assert gx.find_gene_id_by_name("NY-ESO-1") == gx.find_gene_id_by_name("CTAG1B")


def test_transcript_id_to_gene_name():
    assert gx.find_gene_name_from_ensembl_transcript_id("ENST00000269305") == "TP53"
    assert gx.find_gene_name_from_ensembl_transcript_id("ENST_FAKE") is None


def test_full_aggregate_resolves_and_assigns_ids():
    df = pd.DataFrame(
        {
            "transcript_id": ["ENST00000269305", "ENST00000288602", "ENST_FAKE"],
            "tpm": [40.0, 60.0, 5.0],
        }
    )
    out = gx.aggregate_gene_expression(df)
    by = dict(zip(out["gene"], out["TPM"]))
    assert by["TP53"] == 40.0 and by["BRAF"] == 60.0
    assert by["unresolved"] == 5.0  # unknown transcript bucketed, not dropped
    ids = dict(zip(out["gene"], out["gene_id"]))
    assert ids["TP53"] == "ENSG00000141510"
    assert out.attrs["aggregation_stats"]["n_genes"] == 2
