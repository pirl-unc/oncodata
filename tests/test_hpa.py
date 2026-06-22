# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

from oncoref import hpa


def _seed(cache_root, name, version, filename, text):
    path = cache_root / "sources" / name / version / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def hpa_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    # clear lru_caches so each test sees its own fixture
    for fn in (
        hpa.hpa_rna_consensus,
        hpa.hpa_normal_tissue,
        hpa.hpa_single_cell,
        hpa.hpa_cell_type_expression,
    ):
        fn.cache_clear()
    _seed(
        tmp_path,
        "hpa_rna_consensus",
        "v23",
        "rna_tissue_consensus.tsv",
        "Gene\tGene name\tTissue\tnTPM\n"
        "ENSG00000001\tG1\ttestis\t120.0\n"
        "ENSG00000001\tG1\tliver\t0.2\n"
        "ENSG00000002\tG2\tliver\t50.0\n",
    )
    _seed(
        tmp_path,
        "hpa_normal_tissue",
        "v23",
        "normal_tissue.tsv",
        "Gene\tGene name\tTissue\tCell type\tLevel\tReliability\n"
        "ENSG00000001\tG1\ttestis\tgerm cells\tHigh\tEnhanced\n"
        "ENSG00000001\tG1\tliver\thepatocytes\tNot detected\tEnhanced\n",
    )
    _seed(
        tmp_path,
        "hpa_single_cell",
        "v23",
        "rna_single_cell_type.tsv",
        "Gene\tGene name\tCell type\tnTPM\n"
        "ENSG00000001\tG1\tspermatocytes\t300.0\n"
        "ENSG00000001\tG1\thepatocytes\t0.0\n",
    )
    yield tmp_path
    for fn in (
        hpa.hpa_rna_consensus,
        hpa.hpa_normal_tissue,
        hpa.hpa_single_cell,
        hpa.hpa_cell_type_expression,
    ):
        fn.cache_clear()


def test_rna_consensus_loads(hpa_cache):
    df = hpa.hpa_rna_consensus()
    assert set(df.columns) == {"Gene", "Gene name", "Tissue", "nTPM"}
    assert len(df) == 3


def test_gene_tissue_ntpm(hpa_cache):
    expr = hpa.gene_tissue_ntpm("ENSG00000001.5")  # versioned id tolerated
    assert expr == {"testis": 120.0, "liver": 0.2}


def test_gene_cell_type_ntpm(hpa_cache):
    sc = hpa.gene_cell_type_ntpm("ENSG00000001")
    assert sc["spermatocytes"] == 300.0


def test_hpa_cell_type_expression_wide(hpa_cache):
    wide = hpa.hpa_cell_type_expression()
    assert list(wide.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "hepatocytes",
        "spermatocytes",
    ]
    row = wide.iloc[0]
    assert row["Ensembl_Gene_ID"] == "ENSG00000001"
    assert row["Symbol"] == "G1"
    assert row["spermatocytes"] == 300.0


def test_gene_protein_tissues_detected_only(hpa_cache):
    # "Not detected" liver row is excluded; testis High is kept.
    assert hpa.gene_protein_tissues("ENSG00000001") == {"testis"}


def test_cli_sources_list(hpa_cache, capsys):
    from oncoref import cli

    assert cli.main(["hpa", "list"]) == 0
    out = capsys.readouterr().out
    assert "hpa_rna_consensus" in out and "hpa_single_cell" in out


def test_cli_sources_path_uses_cache(hpa_cache, capsys):
    from oncoref import cli

    # already seeded -> ensure() returns the path without a network fetch
    assert cli.main(["hpa", "path", "hpa_rna_consensus"]) == 0
    assert "rna_tissue_consensus.tsv" in capsys.readouterr().out


def test_hpa_parquet_cache(monkeypatch, tmp_path):
    # _read_hpa caches a parquet next to the TSV and reads it on the next call.
    import pandas as pd

    from oncoref import hpa, reference_data

    tsv = tmp_path / "rna_tissue_consensus.tsv"
    tsv.write_text("Gene\tTissue\tnTPM\nENSG1\tliver\t12.5\nENSG2\tlung\t3.0\n")
    monkeypatch.setattr(reference_data, "ensure", lambda name, *a, **k: tsv)

    parquet = tsv.with_suffix(".parquet")
    assert not parquet.exists()
    df1 = hpa._read_hpa("hpa_rna_consensus")
    assert parquet.exists()  # parquet cache written
    assert list(df1.columns) == ["Gene", "Tissue", "nTPM"]
    assert len(df1) == 2

    # Second call reads the parquet (delete the TSV to prove it's not re-parsed).
    tsv.unlink()
    monkeypatch.setattr(reference_data, "ensure", lambda name, *a, **k: tsv)
    df2 = pd.read_parquet(parquet)
    assert df2.equals(df1)
