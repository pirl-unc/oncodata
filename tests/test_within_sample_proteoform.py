# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from cancerdata.expression import (
    available_within_sample_cohorts,
    within_sample_top_fraction,
)

_GEN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_within_sample_top5.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("_gen_within_sample", _GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_proteoform_collapses_members_before_ranking(tmp_path):
    gen = _load_generator()
    input_dir = tmp_path / "per_sample"
    input_dir.mkdir()
    # 4 genes across 3 samples; SSX4 + SSX4B encode the identical protein.
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000268009",  # SSX4
                "ENSG00000269791",  # SSX4B
                "ENSG00000185686",  # PRAME (ungrouped)
                "ENSG00000005961",  # filler so ranks aren't degenerate
            ],
            "Symbol": ["SSX4", "SSX4B", "PRAME", "ITGA2B"],
            "s1": [3.0, 5.0, 100.0, 1.0],
            "s2": [4.0, 4.0, 2.0, 0.0],
            "s3": [10.0, 0.0, 0.0, 50.0],
        }
    ).to_parquet(input_dir / "PRAD.parquet", index=False)

    out_dir = tmp_path / "out"
    gen.build(input_dir, drop_genes=set(), out_dir=out_dir, proteoform=True)

    out = pd.read_parquet(out_dir / "PRAD.parquet")
    symbols = set(out["Symbol"])
    # SSX4 + SSX4B collapsed to a single proteoform row; the members are gone from the
    # key space (Symbol is the contracted proteoform symbol; provenance in members).
    assert "SSX4/B" in symbols
    assert "SSX4" not in symbols and "SSX4B" not in symbols
    assert "SSX4/SSX4B" in set(out["proteoform_members"])
    # Ungrouped genes survive untouched.
    assert {"PRAME", "ITGA2B"} <= symbols
    # n_samples is the sample count, and the threshold columns exist.
    assert (out["n_samples"] == 3).all()
    assert "frac_samples_top5pct" in out.columns


def test_generator_proteoform_rescues_diluted_members(tmp_path):
    # The motivating case: two identical-protein paralogs each rank below the
    # top-5% bar on their own, but their summed proteoform clears it. Build both
    # variants from the same input and confirm the proteoform fraction exceeds
    # either member's per-gene fraction.
    gen = _load_generator()
    input_dir = tmp_path / "per_sample"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000268009",  # SSX4
                "ENSG00000269791",  # SSX4B
                "ENSG00000185686",  # PRAME (ungrouped)
                "ENSG00000005961",  # ITGA2B (ungrouped)
            ],
            "Symbol": ["SSX4", "SSX4B", "PRAME", "ITGA2B"],
            "s1": [3.0, 5.0, 100.0, 1.0],
            "s2": [4.0, 4.0, 2.0, 0.0],
            "s3": [10.0, 0.0, 0.0, 50.0],
        }
    ).to_parquet(input_dir / "PRAD.parquet", index=False)

    per_gene_dir = tmp_path / "per_gene"
    proteoform_dir = tmp_path / "proteoform"
    gen.build(input_dir, drop_genes=set(), out_dir=per_gene_dir, proteoform=False)
    gen.build(input_dir, drop_genes=set(), out_dir=proteoform_dir, proteoform=True)

    per_gene = pd.read_parquet(per_gene_dir / "PRAD.parquet").set_index("Symbol")
    proteoform = pd.read_parquet(proteoform_dir / "PRAD.parquet").set_index("Symbol")

    member_fracs = per_gene.loc[["SSX4", "SSX4B"], "frac_samples_top5pct"]
    proteoform_frac = proteoform.loc["SSX4/B", "frac_samples_top5pct"]
    # Neither diluted member clears the bar; the summed proteoform does.
    assert (member_fracs == 0.0).all()
    assert proteoform_frac > member_fracs.max()


@pytest.fixture
def proteoform_within_sample_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-within-sample-top5-proteoform"
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["SSX4/SSX4B", "ENSG00000185686"],
            "Symbol": ["SSX4/SSX4B", "PRAME"],
            "frac_samples_top5pct": [0.42, 0.10],
            "n_samples": [200, 200],
        }
    ).to_parquet(shard_dir / "PRAD.parquet", index=False)
    return tmp_path


def test_accessor_reads_proteoform_variant(proteoform_within_sample_cache):
    assert available_within_sample_cohorts(proteoform=True) == ["PRAD"]
    # The per-gene variant has no shard, so it stays empty.
    assert available_within_sample_cohorts() == []

    out = within_sample_top_fraction("PRAD", threshold=0.95, proteoform=True)
    assert list(out["Symbol"]) == ["SSX4/SSX4B", "PRAME"]
    assert out.set_index("Symbol").loc["SSX4/SSX4B", "frac_samples_top5pct"] == 0.42


def test_accessor_proteoform_missing_shard_raises(proteoform_within_sample_cache):
    with pytest.raises(ValueError, match="proteoform-summed"):
        within_sample_top_fraction("LUAD", proteoform=True)
