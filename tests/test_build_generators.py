# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oncoref import expression_builders

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


def test_medoids_small_cohort_uses_stable_sample_id_order():
    vals = np.array([[2.0, 1.0]])
    out = expression_builders.cohort_medoids(_matrix(["A"], ["s2", "s1"], vals), k=5)
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


def test_medoids_can_select_on_biological_view_but_return_full_values():
    full = _matrix(
        ["BIO", "TECH"],
        ["s3", "s1", "s2"],
        np.array(
            [
                [10.0, 0.0, 5.0],
                [1000.0, 1000.0, 0.0],
            ]
        ),
    )
    biological = full[full["Ensembl_Gene_ID"] == "BIO"].reset_index(drop=True)

    out = expression_builders.cohort_medoids(
        full,
        sample_cols=["s3", "s1", "s2"],
        k=1,
        selection_df=biological,
    )

    rep_cols = [c for c in out.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert rep_cols == ["s2"]
    assert list(out["Ensembl_Gene_ID"]) == ["BIO", "TECH"]
    assert out.set_index("Ensembl_Gene_ID").loc["TECH", "s2"] == 0.0


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


def test_representatives_generator_selects_on_biological_view(tmp_path, monkeypatch):
    gen = _load_script("generate_representatives")
    inp = tmp_path / "in"
    inp.mkdir()
    _write_cohort(
        inp,
        "COHORT_A",
        ["BIO", "TECH"],
        ["sample_a", "sample_b"],
        np.array([[5.0, 10.0], [1000.0, 0.0]]),
    )
    monkeypatch.setattr(gen, "clean_tpm_censored_gene_ids", lambda: {"TECH"})
    seen = {}

    def fake_medoids(df, sample_cols, *, k, selection_df):
        seen["value_ids"] = df["Ensembl_Gene_ID"].tolist()
        seen["selection_ids"] = selection_df["Ensembl_Gene_ID"].tolist()
        seen["sample_cols"] = list(sample_cols)
        return df[["Ensembl_Gene_ID", "Symbol", sample_cols[0]]].copy()

    monkeypatch.setattr(gen, "cohort_medoids", fake_medoids)

    gen.build(inp, k=1, out_dir=tmp_path / "out")

    assert seen == {
        "value_ids": ["BIO", "TECH"],
        "selection_ids": ["BIO"],
        "sample_cols": ["sample_a", "sample_b"],
    }


def _write_rebuild_inputs(tmp_path):
    cache = tmp_path / "cache"
    ref = tmp_path / "ref"
    source_dir = cache / "TEST_SOURCE" / "derived"
    source_dir.mkdir(parents=True)
    ref.mkdir()
    _matrix(
        ["ENSG000001", "ENSG000002", "ENSG000003"],
        ["pass_sample", "warn_sample", "fail_sample"],
        np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0],
            ]
        ),
    ).to_parquet(source_dir / "X_per_sample_tpm.parquet", index=False)
    _write_cohort(ref, "X", ["ENSG000001"], ["reference"], np.array([[1.0]]))
    return cache, ref


def _patch_rebuild_registry(monkeypatch, gen):
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type, cancer_type],
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["pass", "warn", "fail"],
                "sample_qc_reasons": ["", "tpm_proxy_scale", "low_detected_genes"],
            }
        ),
    )


def test_rebuild_expression_artifacts_defaults_to_qc_passing_samples(tmp_path, monkeypatch):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    _patch_rebuild_registry(monkeypatch, gen)
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="pass")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == ["pass_sample"]

    reps = pd.read_parquet(out / "cancer-reference-expression-representatives" / "X.parquet")
    assert [c for c in reps.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == ["X__rep1"]

    prov = pd.read_csv(out / "cancer-reference-expression-representatives" / "_provenance.csv")
    assert prov.loc[0, "n_source_samples"] == 3
    assert prov.loc[0, "n_cohort_samples"] == 1
    assert prov.loc[0, "sample_qc"] == "pass"
    assert prov.loc[0, "sample_qc_policy_version"] == "sample_expression_qc_v1"
    assert prov.loc[0, "n_qc_pass"] == 1
    assert prov.loc[0, "n_qc_warn"] == 1
    assert prov.loc[0, "n_qc_fail"] == 1

    qc = pd.read_csv(out / "source-matrix-sample-qc.csv")
    assert list(qc["sample_id"]) == ["pass_sample", "warn_sample", "fail_sample"]

    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "n_source_samples"] == 3
    assert build_meta.loc[0, "n_cohort_samples"] == 1
    assert build_meta.loc[0, "sample_qc_policy_version"] == "sample_expression_qc_v1"

    metadata = json.loads((out / "expression-artifact-build-metadata.json").read_text())
    assert metadata["sample_qc"] == "pass"
    assert metadata["sample_qc_manifest"] == "source-matrix-sample-qc.csv"
    assert metadata["n_source_samples"] == 3
    assert metadata["n_cohort_samples"] == 1


def test_rebuild_expression_artifacts_keeps_warn_proxy_source_when_pass_empty(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type, cancer_type],
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["warn", "warn", "warn"],
                "sample_qc_reasons": ["tpm_proxy_scale", "tpm_proxy_scale", "tpm_proxy_scale"],
                "source_scale_class": [
                    "microarray_tpm_proxy",
                    "microarray_tpm_proxy",
                    "microarray_tpm_proxy",
                ],
            }
        ),
    )
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="pass")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == [
        "pass_sample",
        "warn_sample",
        "fail_sample",
    ]
    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "sample_qc"] == "pass"
    assert build_meta.loc[0, "sample_qc_effective"] == "pass_or_warn"
    assert build_meta.loc[0, "sample_qc_fallback_reason"] == "no_pass_samples_tpm_proxy_source"
    metadata = json.loads((out / "expression-artifact-build-metadata.json").read_text())
    assert metadata["sample_qc_fallbacks"] == 1


def test_rebuild_expression_artifacts_keeps_concentration_only_source_when_pass_empty(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type, cancer_type],
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["fail", "fail", "fail"],
                "sample_qc_reasons": [
                    "high_top10_gene_fraction",
                    "high_top_gene_fraction;high_top10_gene_fraction",
                    "high_top10_gene_fraction",
                ],
            }
        ),
    )
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="pass")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == [
        "pass_sample",
        "warn_sample",
        "fail_sample",
    ]
    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "sample_qc"] == "pass"
    assert build_meta.loc[0, "sample_qc_effective"] == "all"
    assert (
        build_meta.loc[0, "sample_qc_fallback_reason"]
        == "no_pass_samples_high_concentration_source"
    )


def test_rebuild_expression_artifacts_clips_negative_source_values():
    gen = _load_script("rebuild_expression_artifacts")
    df = _matrix(["G1", "G2"], ["s1", "s2"], np.array([[-2.0, 3.0], [4.0, -0.5]]))

    out, n_negative = gen._clip_negative_expression(df, ["s1", "s2"])

    assert n_negative == 2
    assert out[["s1", "s2"]].to_numpy().min() == 0.0
    assert df[["s1", "s2"]].to_numpy().min() < 0.0


def test_rebuild_expression_artifacts_disambiguates_source_by_registry_sample_count(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache = tmp_path / "cache"
    ref = tmp_path / "ref"
    ref.mkdir()
    _write_cohort(ref, "X", ["G1", "G2"], ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    source_a = cache / "source-a" / "derived"
    source_b = cache / "source-b" / "derived"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    _matrix(["G1", "G2"], ["s1"], np.array([[1.0], [2.0]])).to_parquet(
        source_a / "X_per_sample_tpm.parquet", index=False
    )
    _matrix(["G1", "G2"], ["s1", "s2"], np.array([[1.0, 2.0], [2.0, 3.0]])).to_parquet(
        source_b / "X_per_sample_tpm.parquet", index=False
    )
    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame(
            {"cancer_code": ["X"], "source_cohort": ["AGGREGATE_LABEL"], "n_samples": [2]}
        ),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type] * (len(raw.columns) - 2),
                "sample_id": [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")],
                "sample_qc_status": ["pass"] * (len(raw.columns) - 2),
                "sample_qc_reasons": [""] * (len(raw.columns) - 2),
            }
        ),
    )

    gen.rebuild(cache, ref, tmp_path / "out", limit=None, validate=False, sample_qc="pass")

    build_meta = pd.read_csv(tmp_path / "out" / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "n_source_samples"] == 2
    assert build_meta.loc[0, "source_matrix_path"].endswith(
        "source-b/derived/X_per_sample_tpm.parquet"
    )


def test_rebuild_expression_artifacts_prefers_aggregate_source_directory(tmp_path):
    gen = _load_script("rebuild_expression_artifacts")
    preferred = tmp_path / "geo-heme" / "derived" / "X_per_sample_tpm.parquet"
    duplicate = tmp_path / "gse100026-cml" / "derived" / "X_per_sample_tpm.parquet"
    preferred.parent.mkdir(parents=True)
    duplicate.parent.mkdir(parents=True)
    _matrix(["G1"], ["s1"], np.array([[1.0]])).to_parquet(preferred, index=False)
    _matrix(["G1"], ["s1"], np.array([[1.0]])).to_parquet(duplicate, index=False)

    selected = gen._select_source(
        "X",
        [("gse100026-cml", duplicate), ("geo-heme", preferred)],
        {"X": "GEO_HEME_2022"},
        {"X": 1},
    )

    assert selected == preferred


def test_rebuild_expression_artifacts_selects_representatives_on_biological_view(
    tmp_path, monkeypatch
):
    gen = _load_script("rebuild_expression_artifacts")
    cache = tmp_path / "cache"
    ref = tmp_path / "ref"
    source_dir = cache / "TEST_SOURCE" / "derived"
    source_dir.mkdir(parents=True)
    ref.mkdir()
    _matrix(
        ["BIO", "TECH"],
        ["sample_a", "sample_b"],
        np.array([[5.0, 10.0], [1000.0, 0.0]]),
    ).to_parquet(source_dir / "X_per_sample_tpm.parquet", index=False)
    _write_cohort(ref, "X", ["BIO"], ["reference"], np.array([[1.0]]))

    monkeypatch.setattr(
        gen,
        "source_registry",
        lambda: pd.DataFrame({"cancer_code": ["X"], "source_cohort": ["TEST_SOURCE"]}),
    )
    monkeypatch.setattr(gen, "cohort_source_version", lambda code: "test-source-version")
    monkeypatch.setattr(
        gen,
        "sample_expression_qc_from_matrix",
        lambda raw, cancer_type: pd.DataFrame(
            {
                "cancer_code": [cancer_type, cancer_type],
                "sample_id": ["sample_a", "sample_b"],
                "sample_qc_status": ["pass", "pass"],
                "sample_qc_reasons": ["", ""],
            }
        ),
    )
    monkeypatch.setattr(gen, "clean_tpm", lambda values, gene_table: values.copy())
    monkeypatch.setattr(gen, "clean_tpm_censored_gene_ids", lambda: {"TECH"})

    seen = {}

    def fake_medoids(df, sample_cols, *, k, selection_df):
        seen["value_ids"] = df["Ensembl_Gene_ID"].tolist()
        seen["selection_ids"] = selection_df["Ensembl_Gene_ID"].tolist()
        seen["sample_cols"] = list(sample_cols)
        return df[["Ensembl_Gene_ID", "Symbol", sample_cols[0]]].copy()

    monkeypatch.setattr(gen, "cohort_medoids", fake_medoids)

    gen.rebuild(cache, ref, tmp_path / "out", limit=None, validate=False, sample_qc="pass")

    assert seen == {
        "value_ids": ["BIO", "TECH"],
        "selection_ids": ["BIO"],
        "sample_cols": ["sample_a", "sample_b"],
    }


def test_rebuild_expression_artifacts_keeps_all_samples_when_requested(tmp_path, monkeypatch):
    gen = _load_script("rebuild_expression_artifacts")
    cache, ref = _write_rebuild_inputs(tmp_path)
    _patch_rebuild_registry(monkeypatch, gen)
    out = tmp_path / "out"

    gen.rebuild(cache, ref, out, limit=None, validate=False, sample_qc="all")

    clean = pd.read_parquet(out / "clean" / "X.parquet")
    assert [c for c in clean.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == [
        "pass_sample",
        "warn_sample",
        "fail_sample",
    ]
    build_meta = pd.read_csv(out / "expression-artifact-build-metadata.csv")
    assert build_meta.loc[0, "sample_qc"] == "all"
    assert build_meta.loc[0, "n_source_samples"] == 3
    assert build_meta.loc[0, "n_cohort_samples"] == 3


# ---------- real-data parity (skipped without the maintainer's matrix cache) ----------


@pytest.mark.skipif(not _PARITY_READY, reason="per-sample matrix cache / pirlygenes ref absent")
def test_percentiles_reproduce_pirlygenes_reference():
    # End-to-end on REAL data: raw per-sample matrix -> clean_tpm -> percentile
    # vectors must reproduce pirlygenes' shipped percentile artifact for the same
    # cohort. Proves the generator + oncoref's clean_tpm port are faithful.
    from oncoref import normalization as nz

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
