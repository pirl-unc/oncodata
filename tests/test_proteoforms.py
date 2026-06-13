# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from cancerdata import (
    gene_to_proteoform,
    proteoform_aliases,
    proteoform_for_gene,
    proteoform_group_map,
    proteoform_groups,
    proteoform_representative_samples,
    proteoform_symbol,
    proteoform_symbol_map,
)
from cancerdata._build import sum_proteoform_tpm
from cancerdata.proteoforms import _contract_members

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


def test_expression_level_marker():
    from cancerdata import expression_level

    gene = pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["A"], "s1": [1.0]})
    assert expression_level(gene) == "gene"
    assert expression_level(gene.assign(proteoform_key=["E1"])) == "proteoform"


def test_genome_scope_is_superset_of_cta():
    # Every gene mapped to its best protein, genome-wide: the genome registry has many
    # more identical-protein groups (histones, tubulins, PAR X/Y, …) than the CTA subset.
    assert len(proteoform_group_map(scope="genome")) > len(proteoform_group_map(scope="cta"))


def test_contract_members_factors_common_prefix():
    assert _contract_members("XAGE1A/XAGE1B") == "XAGE1A/B"
    assert _contract_members("CGB3/CGB5/CGB8") == "CGB3/5/8"
    assert _contract_members("SSX4/SSX4B") == "SSX4/B"
    assert _contract_members("GAGE12C/GAGE12D/GAGE12E") == "GAGE12C/D/E"
    assert _contract_members("PRAME") == "PRAME"  # singleton unchanged
    assert _contract_members("AAA/BBB") == "AAA/BBB"  # no shared prefix -> full join


def test_proteoform_symbol_prefers_curated_alias():
    assert proteoform_aliases()["CTAG1A/CTAG1B"] == "NY-ESO-1"
    assert proteoform_symbol("CTAG1A/CTAG1B") == "NY-ESO-1"  # alias wins
    assert proteoform_symbol("XAGE1A/XAGE1B") == "XAGE1A/B"  # else contracted


def test_collapse_decreases_key_count_no_duplicate_keys():
    # The reduction invariant: collapsing identical-protein members yields fewer keys
    # (members merge), and the surviving Symbol/ENSG keys are unique.
    from cancerdata.proteoforms import collapse_to_proteoforms

    gmap = proteoform_group_map()
    members = next(ids for ids in gmap.values() if len(ids) > 1)
    extra = "ENSG00000185686"  # PRAME singleton
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [*members, extra],
            "Symbol": [f"m{i}" for i in range(len(members))] + ["PRAME"],
            "s1": [1.0] * (len(members) + 1),
        }
    )
    out = collapse_to_proteoforms(df, sample_cols=["s1"])
    assert len(out) < len(df)  # fewer keys after collapse
    assert len(out) == 2  # one proteoform + PRAME
    assert out["proteoform_key"].is_unique and out["Ensembl_Gene_ID"].is_unique
    # The PRAME singleton keys by its ENSG; the group keys by its proteoform symbol.
    keys = set(out["proteoform_key"])
    assert extra in keys  # 1:1 gene -> ENSG
    assert any(not k.startswith("ENSG") for k in keys)  # group -> a symbol


def test_proteoform_for_gene_by_id_symbol_and_version():
    # Returns the proteoform SYMBOL (prefix-contracted members), not the raw slash-label.
    assert proteoform_for_gene("ENSG00000268009") == "SSX4/B"  # SSX4 gene id
    assert proteoform_for_gene("ENSG00000269791") == "SSX4/B"  # SSX4B gene id
    assert proteoform_for_gene("ENSG00000268009.3") == "SSX4/B"  # version suffix
    assert proteoform_for_gene("ssx4") == "SSX4/B"  # symbol, case-insensitive
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
    # No group_symbols passed -> Symbol falls back to the raw members label; the
    # sorted members are always in proteoform_members.
    out = sum_proteoform_tpm(df, {"SSX4/SSX4B": ("ENSG00000268009", "ENSG00000269791")})
    by_pf = out.set_index("proteoform_members")
    assert by_pf.loc["SSX4/SSX4B", "s1"] == 8.0
    assert by_pf.loc["SSX4/SSX4B", "s2"] == 1.0
    # ENSG stays a REAL Ensembl id (the canonical = smallest member), not the label.
    assert by_pf.loc["SSX4/SSX4B", "Ensembl_Gene_ID"] == "ENSG00000268009"
    assert by_pf.loc["SSX4/SSX4B", "Symbol"] == "SSX4/SSX4B"
    # PRAME passes through: singleton -> proteoform_members is its own symbol, ENSG kept.
    assert by_pf.loc["PRAME", "s1"] == 100.0
    assert by_pf.loc["PRAME", "Ensembl_Gene_ID"] == "ENSG00000185686"
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
    # Symbol is now the contracted proteoform symbol; members are in proteoform_members.
    by_symbol = out.set_index("Symbol")
    assert by_symbol.loc["SSX4/B", "rep_0"] == 10.0
    assert by_symbol.loc["SSX4/B", "rep_1"] == 4.0
    assert by_symbol.loc["PRAME", "rep_0"] == 50.0
    assert out.set_index("proteoform_members").loc["SSX4/SSX4B", "rep_0"] == 10.0


def test_gene_to_proteoform_id_is_total():
    # Every gene maps to a class: grouped -> label, singleton -> symbol (or ENSG).
    from cancerdata.proteoforms import gene_to_proteoform_id

    genes = ["ENSG00000268009", "ENSG00000269791", "ENSG00000185686"]
    m = gene_to_proteoform_id(genes)
    assert set(m) == set(genes)  # total
    # The grouped SSX4 / SSX4B map to the proteoform symbol (the reduced key).
    assert m["ENSG00000268009"] == "SSX4/B"
    assert m["ENSG00000269791"] == "SSX4/B"
    # PRAME uniquely owns its protein -> its own ENSG is the key (the 1:1 case).
    assert m["ENSG00000185686"] == "ENSG00000185686"


def test_proteoform_key_ensg_for_unique_symbol_for_group():
    from cancerdata.proteoforms import proteoform_key

    # Group member -> proteoform symbol; unique gene -> its own ENSG.
    assert proteoform_key("ENSG00000184033") == "NY-ESO-1"  # CTAG1B (aliased group)
    assert proteoform_key("ENSG00000268009") == "SSX4/B"  # SSX4 (contracted group)
    assert proteoform_key("ENSG00000185686") == "ENSG00000185686"  # PRAME, unique -> ENSG
    assert proteoform_key("ENSG00000268009.4") == "SSX4/B"  # version-insensitive


def test_contract_members_dedupes_identical_symbols():
    # Genome-scope X/Y paralogs can share a symbol (AKAP17A/AKAP17A): the contraction
    # must not leave a trailing slash.
    assert _contract_members("AKAP17A/AKAP17A") == "AKAP17A"
    assert _contract_members("CD99/CD99") == "CD99"


def test_collapse_to_proteoforms_keeps_ensembl_id_real():
    # The reusable collapse entry point: ENSG column stays a real Ensembl id, the
    # sorted members are in proteoform_members, and the key count drops to 1.
    from cancerdata.proteoforms import (
        collapse_to_proteoforms,
        proteoform_group_map,
        proteoform_symbol,
    )

    gmap = proteoform_group_map()
    # pick a real CTA group to exercise the collapse
    label, members = next(iter(gmap.items()))
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": list(members),
            "Symbol": [f"m{i}" for i in range(len(members))],
            "s1": [10.0] * len(members),
        }
    )
    out = collapse_to_proteoforms(df, sample_cols=["s1"])
    assert "proteoform_members" in out.columns
    assert "proteoform_id" not in out.columns
    assert len(out) == 1  # members collapsed to one key
    assert out.iloc[0]["proteoform_members"] == label  # provenance = sorted members
    assert out.iloc[0]["Symbol"] == proteoform_symbol(label)  # alias / contracted
    assert out.iloc[0]["Ensembl_Gene_ID"] == min(members)  # canonical member
    assert out.iloc[0]["s1"] == 10.0 * len(members)  # summed
