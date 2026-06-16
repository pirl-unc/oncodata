# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oncodata import expression_builders

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Optional real-data parity: the per-sample matrices + pirlygenes' shipped
# percentile artifact live only on a maintainer's machine (~22 GB cache), so this
# is gated like the HPA-dependent tests and skips cleanly everywhere else.
_ACC_MATRIX = glob.glob(
    os.path.expanduser("~/.cache/pirlygenes/expression/*/derived/tcga_acc_per_sample_tpm.parquet")
)
_ACC_REF = Path(
    os.path.expanduser(
        "~/code/pirlygenes/pirlygenes/data/cancer-reference-expression-percentiles/ACC.parquet"
    )
)
_PARITY_READY = bool(_ACC_MATRIX) and _ACC_REF.exists()


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _matrix(genes, samples, values):
    """genes × samples DataFrame with id columns."""
    df = pd.DataFrame(values, columns=samples)
    df.insert(0, "Symbol", [f"G{i}" for i in range(len(genes))])
    df.insert(0, "Ensembl_Gene_ID", genes)
    return df


# ---------- cohort_percentile_vectors ----------


def test_percentile_vectors_schema_matches_reader():
    # 26 dense breakpoints p0..p100, plus the two id columns — the exact schema
    # expression.cohort_gene_percentiles reads back.
    genes = ["A", "B"]
    vals = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    out = expression_builders.cohort_percentile_vectors(_matrix(genes, list("wxyz"), vals))
    bp_cols = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert len(bp_cols) == 26
    assert bp_cols[0] == "p0" and bp_cols[-1] == "p100"
    assert "p50" in bp_cols
    assert len(out) == 2


def test_percentile_vectors_log1p_roundtrip():
    # Stored log1p; expm1 of p0/p50/p100 recovers min/median/max of the gene's
    # across-sample distribution (the reader's as_tpm=True path).
    vals = np.array([[0.0, 10.0, 100.0, 1000.0]])
    out = expression_builders.cohort_percentile_vectors(_matrix(["A"], list("wxyz"), vals))
    row = out.iloc[0]
    assert np.isclose(np.expm1(np.float32(row["p0"])), 0.0, atol=1e-1)
    assert np.isclose(np.expm1(np.float32(row["p100"])), 1000.0, rtol=2e-2)
    # median of [0,10,100,1000] in log1p space, restored
    expected_med = np.median(np.log1p(vals[0]))
    assert np.isclose(np.float32(row["p50"]), expected_med, rtol=2e-2)


def test_percentile_vectors_ignores_nan():
    # A gene unmeasured in some samples: NaN cells are dropped, not treated as 0.
    vals = np.array([[np.nan, 10.0, 10.0, 10.0]])
    out = expression_builders.cohort_percentile_vectors(_matrix(["A"], list("wxyz"), vals))
    assert np.isclose(np.expm1(np.float32(out.iloc[0]["p50"])), 10.0, rtol=2e-2)


def test_percentile_vectors_requires_samples():
    with pytest.raises(ValueError):
        expression_builders.cohort_percentile_vectors(_matrix(["A"], [], np.empty((1, 0))))


# ---------- cohort_medoids ----------


def test_medoids_returns_base_plus_k():
    rng = np.arange(50.0).reshape(5, 10)  # 5 genes × 10 samples
    out = expression_builders.cohort_medoids(
        _matrix([f"g{i}" for i in range(5)], [f"s{j}" for j in range(10)], rng), k=3
    )
    rep_cols = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert len(rep_cols) == 3
    assert list(out["Ensembl_Gene_ID"]) == [f"g{i}" for i in range(5)]


def test_medoids_small_cohort_keeps_all():
    vals = np.array([[1.0, 2.0]])
    out = expression_builders.cohort_medoids(_matrix(["A"], ["s1", "s2"], vals), k=5)
    assert [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == ["s1", "s2"]


def test_medoids_central_first_then_outlier():
    # 4 near-identical "typical" samples + 1 far outlier. The medoid (pick 1)
    # must come from the dense cluster; the farthest pick (2) must be the outlier.
    genes = [f"g{i}" for i in range(6)]
    typical = np.tile(np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]), (1, 4))
    typical = typical + np.array([[0, 0.01, -0.01, 0.0]] * 6)  # tiny jitter
    outlier = np.array([[100.0], [100.0], [100.0], [100.0], [100.0], [100.0]])
    vals = np.hstack([typical, outlier])
    samples = ["t1", "t2", "t3", "t4", "outlier"]
    out = expression_builders.cohort_medoids(_matrix(genes, samples, vals), k=2)
    picks = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert picks[0] != "outlier"  # central medoid from the dense cluster
    assert picks[1] == "outlier"  # farthest-first grabs the outlier


def test_medoids_preserve_original_tpm():
    # Distance uses log1p internally, but stored values are the original TPM.
    vals = np.array([[7.0, 8.0, 9.0]])
    out = expression_builders.cohort_medoids(_matrix(["A"], ["s1", "s2", "s3"], vals), k=3)
    kept = out[[c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]].to_numpy()
    assert set(kept.ravel()) == {7.0, 8.0, 9.0}


def test_medoids_deterministic():
    rng = (np.arange(60.0) * 1.7 % 11).reshape(6, 10)
    df = _matrix([f"g{i}" for i in range(6)], [f"s{j}" for j in range(10)], rng)
    a = expression_builders.cohort_medoids(df, k=4)
    b = expression_builders.cohort_medoids(df, k=4)
    assert list(a.columns) == list(b.columns)


# ---------- generator scripts (end-to-end on a synthetic per-sample dir) ----------


def _write_cohort(tmp_path, code, genes, samples, values):
    df = _matrix(genes, samples, values)
    path = tmp_path / f"{code}.parquet"
    df.to_parquet(path, index=False)
    return path


def test_percentiles_generator_writes_readable_shards(tmp_path):
    gen = _load_script("generate_cohort_percentiles")
    inp = tmp_path / "in"
    inp.mkdir()
    _write_cohort(
        inp,
        "COHORT_A",
        ["A", "B"],
        list("wxyz"),
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
    )
    out = tmp_path / "out"
    gen.build(inp, drop_genes=set(), out_dir=out)
    shard = pd.read_parquet(out / "COHORT_A.parquet")
    assert len(shard) == 2
    bp_cols = [c for c in shard.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert len(bp_cols) == 26
    # p100 of gene A (max 4.0), stored log1p
    a = shard[shard["Ensembl_Gene_ID"] == "A"].iloc[0]
    assert np.isclose(np.expm1(np.float32(a["p100"])), 4.0, rtol=2e-2)


def test_percentiles_generator_drops_genes(tmp_path):
    gen = _load_script("generate_cohort_percentiles")
    inp = tmp_path / "in"
    inp.mkdir()
    _write_cohort(
        inp,
        "COHORT_A",
        ["KEEP", "DROPME"],
        ["s1", "s2"],
        np.array([[1.0, 2.0], [9.0, 9.0]]),
    )
    out = tmp_path / "out"
    gen.build(inp, drop_genes={"DROPME"}, out_dir=out)
    shard = pd.read_parquet(out / "COHORT_A.parquet")
    assert list(shard["Ensembl_Gene_ID"]) == ["KEEP"]


def test_representatives_generator_writes_shards_and_provenance(tmp_path):
    gen = _load_script("generate_representatives")
    inp = tmp_path / "in"
    inp.mkdir()
    # 6 samples so k=3 actually selects a subset
    vals = np.array([[float(g * 10 + s) for s in range(6)] for g in range(4)])
    _write_cohort(inp, "COHORT_A", [f"g{i}" for i in range(4)], [f"s{j}" for j in range(6)], vals)
    out = tmp_path / "out"
    gen.build(inp, k=3, out_dir=out)

    shard = pd.read_parquet(out / "COHORT_A.parquet")
    rep_cols = [c for c in shard.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert rep_cols == ["COHORT_A__rep1", "COHORT_A__rep2", "COHORT_A__rep3"]

    prov = pd.read_csv(out / "_provenance.csv")
    assert set(prov["representative_id"]) == set(rep_cols)
    assert (prov["n_cohort_samples"] == 6).all()
    # The reader merges on these exact columns — all must be present (source_project
    # is best-effort and empty for an unregistered synthetic code, but the column
    # must exist so consumers don't KeyError).
    for col in ("representative_id", "source_cohort", "source_project", "n_cohort_samples"):
        assert col in prov.columns
    # Unregistered code -> source_cohort falls back to the code itself.
    assert (prov["source_cohort"] == "COHORT_A").all()


# ---------- real-data parity (skipped without the maintainer's matrix cache) ----------


@pytest.mark.skipif(not _PARITY_READY, reason="per-sample matrix cache / pirlygenes ref absent")
def test_percentiles_reproduce_pirlygenes_reference():
    # End-to-end on REAL data: raw per-sample matrix -> clean_tpm -> percentile
    # vectors must reproduce pirlygenes' shipped percentile artifact for the same
    # cohort. Proves the generator + oncodata's clean_tpm port are faithful.
    from oncodata import normalization as nz

    raw = pd.read_parquet(_ACC_MATRIX[0])
    samples = [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gene_table = raw[["Symbol", "Ensembl_Gene_ID"]]
    clean = nz.clean_tpm(raw[samples], gene_table=gene_table)
    clean_df = pd.concat([gene_table, clean], axis=1)

    mine = expression_builders.cohort_percentile_vectors(clean_df, samples).set_index(
        "Ensembl_Gene_ID"
    )
    ref = pd.read_parquet(_ACC_REF).set_index("Ensembl_Gene_ID")
    # Column schema is identical.
    assert [c for c in mine.columns if c != "Symbol"] == [c for c in ref.columns if c != "Symbol"]

    common = mine.index.intersection(ref.index)
    assert len(common) > 10_000
    # The deterministic mid/upper percentiles match (expm1 back to TPM); tiny tail
    # deviation at p99 is float16 rounding, so correlation must be essentially 1.
    for col in ("p50", "p95"):
        a = np.expm1(mine.loc[common, col].astype("float32"))
        b = np.expm1(ref.loc[common, col].astype("float32"))
        mask = (a > 0) | (b > 0)
        assert np.corrcoef(a[mask], b[mask])[0, 1] > 0.999
