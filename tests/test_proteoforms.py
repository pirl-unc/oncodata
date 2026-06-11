# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from cancerdata import (
    gene_to_proteoform,
    proteoform_for_gene,
    proteoform_group_map,
    proteoform_groups,
    proteoform_representative_samples,
    proteoform_symbol_map,
)
from cancerdata._build import sum_proteoform_tpm

# Known identical-protein groups that must be in the shipped registry.
_KNOWN = {
    "SSX4/SSX4B": {"SSX4", "SSX4B"},
    "CTAG1A/CTAG1B": {"CTAG1A", "CTAG1B"},
    "MAGEA2/MAGEA2B": {"MAGEA2", "MAGEA2B"},
    "XAGE1A/XAGE1B": {"XAGE1A", "XAGE1B"},
}


def test_registry_shape():
    df = proteoform_groups()
    assert set(df.columns) >= {
        "proteoform_id",
        "member_symbol",
        "member_gene_id",
        "protein_length",
        "n_members",
    }
    # Every group has at least two members, and n_members matches the row count.
    counts = df.groupby("proteoform_id")["member_symbol"].nunique()
    assert (counts >= 2).all()
    for _label, sub in df.groupby("proteoform_id"):
        assert (sub["n_members"] == len(sub)).all()


@pytest.mark.parametrize("label,members", _KNOWN.items())
def test_known_groups_present(label, members):
    symbol_map = proteoform_symbol_map()
    assert label in symbol_map
    assert set(symbol_map[label]) == members


def test_ct47_family_is_a_single_large_group():
    # The CT47A family duplicates one identical protein across a dozen loci —
    # exactly the multi-mapping case proteoform summation exists for.
    symbol_map = proteoform_symbol_map()
    ct47 = [label for label in symbol_map if label.startswith("CT47A")]
    assert len(ct47) == 1
    assert len(symbol_map[ct47[0]]) >= 10


def test_proteoform_for_gene_by_id_symbol_and_version():
    assert proteoform_for_gene("ENSG00000268009") == "SSX4/SSX4B"  # SSX4 gene id
    assert proteoform_for_gene("ENSG00000269791") == "SSX4/SSX4B"  # SSX4B gene id
    assert proteoform_for_gene("ENSG00000268009.3") == "SSX4/SSX4B"  # version suffix
    assert proteoform_for_gene("ssx4") == "SSX4/SSX4B"  # symbol, case-insensitive
    assert proteoform_for_gene("PRAME") is None  # not duplicated


def test_group_map_and_gene_to_proteoform_agree():
    group_map = proteoform_group_map()
    g2p = gene_to_proteoform()
    for label, gene_ids in group_map.items():
        for gene_id in gene_ids:
            assert g2p[gene_id] == label


def test_sum_proteoform_tpm_collapses_members_and_passes_through():
    # SSX4 + SSX4B sum; PRAME is untouched.
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000268009", "ENSG00000269791", "ENSG00000185686"],
            "Symbol": ["SSX4", "SSX4B", "PRAME"],
            "s1": [3.0, 5.0, 100.0],
            "s2": [1.0, 0.0, 7.0],
        }
    )
    out = sum_proteoform_tpm(df, {"SSX4/SSX4B": ("ENSG00000268009", "ENSG00000269791")})
    by_symbol = out.set_index("Symbol")
    assert by_symbol.loc["SSX4/SSX4B", "s1"] == 8.0
    assert by_symbol.loc["SSX4/SSX4B", "s2"] == 1.0
    assert by_symbol.loc["SSX4/SSX4B", "Ensembl_Gene_ID"] == "SSX4/SSX4B"
    # PRAME passes through unchanged, keeping its original id.
    assert by_symbol.loc["PRAME", "s1"] == 100.0
    assert by_symbol.loc["PRAME", "Ensembl_Gene_ID"] == "ENSG00000185686"
    # One collapsed row + one passthrough row.
    assert len(out) == 2


def test_sum_proteoform_tpm_matches_version_suffixed_ids():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000268009.4", "ENSG00000269791.2"],
            "Symbol": ["SSX4", "SSX4B"],
            "s1": [2.0, 6.0],
        }
    )
    out = sum_proteoform_tpm(df, {"SSX4/SSX4B": ("ENSG00000268009", "ENSG00000269791")})
    assert len(out) == 1
    assert out.iloc[0]["s1"] == 8.0


def test_sum_proteoform_tpm_preserves_nan_for_unmeasured():
    # The multi-cohort outer-merge case: a gene absent from a cohort is NaN, not
    # measured-zero. NaN must survive summation (min_count=1) for both ungrouped
    # pass-throughs and all-NaN groups; a present member still sums over a NaN
    # sibling.
    import numpy as np

    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000185686",  # PRAME, ungrouped
                "ENSG00000268009",  # SSX4
                "ENSG00000269791",  # SSX4B
            ],
            "Symbol": ["PRAME", "SSX4", "SSX4B"],
            "s1": [100.0, 3.0, 5.0],
            "s2": [np.nan, np.nan, 2.0],  # cohort 2: PRAME & SSX4 not measured
        }
    )
    out = sum_proteoform_tpm(df, {"SSX4/SSX4B": ("ENSG00000268009", "ENSG00000269791")}).set_index(
        "Symbol"
    )
    # Ungrouped, unmeasured -> stays NaN (not 0.0).
    assert pd.isna(out.loc["PRAME", "s2"])
    # Group with one present member -> that member's value (NaN sibling skipped).
    assert out.loc["SSX4/SSX4B", "s2"] == 2.0
    assert out.loc["SSX4/SSX4B", "s1"] == 8.0


def test_sum_proteoform_tpm_all_nan_group_stays_nan():
    import numpy as np

    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000268009", "ENSG00000269791"],
            "Symbol": ["SSX4", "SSX4B"],
            "s1": [np.nan, np.nan],
        }
    )
    out = sum_proteoform_tpm(df, {"SSX4/SSX4B": ("ENSG00000268009", "ENSG00000269791")})
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["s1"])


@pytest.fixture
def representatives_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-representatives"
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000268009", "ENSG00000269791", "ENSG00000185686"],
            "Symbol": ["SSX4", "SSX4B", "PRAME"],
            "rep_0": [4.0, 6.0, 50.0],
            "rep_1": [2.0, 2.0, 9.0],
        }
    ).to_parquet(shard_dir / "PRAD.parquet", index=False)
    return tmp_path


def test_proteoform_representative_samples_sums_members(representatives_cache):
    out = proteoform_representative_samples("PRAD")
    by_symbol = out.set_index("Symbol")
    assert by_symbol.loc["SSX4/SSX4B", "rep_0"] == 10.0
    assert by_symbol.loc["SSX4/SSX4B", "rep_1"] == 4.0
    assert by_symbol.loc["PRAME", "rep_0"] == 50.0
