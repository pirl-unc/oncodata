# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pandas as pd
import pytest

from cancerdata import expression

_BREAKPOINTS = [0, 1, 5, 10, 50, 90, 95, 99, 100]


def _write_percentile_shard(cache_dir, code, tpm_values):
    """Write a synthetic percentile parquet (stored as log1p, like the real one)."""
    shard_dir = cache_dir / "cancer-reference-expression-percentiles"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cols = {"Ensembl_Gene_ID": ["ENSG00000000003", "ENSG00000000005"], "Symbol": ["TSPAN6", "TNMD"]}
    for bp in _BREAKPOINTS:
        # store log1p of a per-breakpoint tpm so as_tpm restores tpm_values[bp]
        cols[f"p{bp}"] = np.log1p([tpm_values[bp], tpm_values[bp] * 2]).astype("float16")
    pd.DataFrame(cols).to_parquet(shard_dir / f"{code}.parquet", index=False)


@pytest.fixture
def percentile_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    tpm = {bp: float(bp) for bp in _BREAKPOINTS}  # p95 -> 95 tpm, etc.
    _write_percentile_shard(tmp_path, "PRAD", tpm)
    return tmp_path


def test_available_percentile_cohorts(percentile_cache):
    assert expression.available_percentile_cohorts() == ["PRAD"]


def test_cohort_gene_percentiles_as_tpm(percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=True)
    assert list(df["Symbol"]) == ["TSPAN6", "TNMD"]
    # gene 0, p95 breakpoint should restore to ~95 tpm (stored as log1p(95)).
    assert df.loc[0, "p95"] == pytest.approx(95.0, rel=1e-2)


def test_cohort_gene_percentiles_log_space(percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=False)
    # stored log1p value, not expm1-restored
    assert df.loc[0, "p95"] == pytest.approx(np.log1p(95.0), rel=1e-2)


def test_cohort_gene_percentiles_resolves_alias(percentile_cache):
    # "prostate" resolves to PRAD via the registry.
    df = expression.cohort_gene_percentiles("prostate")
    assert len(df) == 2


def test_cohort_gene_percentiles_missing_raises(percentile_cache):
    with pytest.raises(ValueError, match="no percentile vector"):
        expression.cohort_gene_percentiles("BRCA")


@pytest.fixture
def proteoform_percentile_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-percentiles-proteoform"
    shard_dir.mkdir(parents=True)
    # A collapsed shard carries the proteoform identity columns alongside the breakpoints.
    cols = {
        "proteoform_key": ["NY-ESO-1", "ENSG00000185686"],
        "Ensembl_Gene_ID": ["ENSG00000184033", "ENSG00000185686"],
        "Symbol": ["NY-ESO-1", "PRAME"],
        "proteoform_members": ["CTAG1A/CTAG1B", "PRAME"],
    }
    for bp in _BREAKPOINTS:
        cols[f"p{bp}"] = np.log1p([float(bp), float(bp) * 2]).astype("float16")
    pd.DataFrame(cols).to_parquet(shard_dir / "PRAD.parquet", index=False)
    return tmp_path


def test_available_percentile_cohorts_proteoform(proteoform_percentile_cache):
    # The proteoform variant reads the local shard (auto_fetch=False, no bundle fetch).
    assert expression.available_percentile_cohorts(proteoform=True) == ["PRAD"]


def test_cohort_gene_percentiles_proteoform(proteoform_percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=True, proteoform=True)
    # the proteoform key space: NY-ESO-1 (collapsed) + PRAME (singleton -> ENSG key)
    assert list(df["proteoform_key"]) == ["NY-ESO-1", "ENSG00000185686"]
    # breakpoint columns are restored to TPM; the id columns are excluded from them
    assert df.loc[0, "p95"] == pytest.approx(95.0, rel=1e-2)
    assert set(df.columns) >= {"proteoform_key", "Ensembl_Gene_ID", "Symbol", "proteoform_members"}


def test_cohort_gene_percentiles_proteoform_missing_raises(proteoform_percentile_cache):
    with pytest.raises(ValueError, match="no proteoform-summed percentile vector"):
        expression.cohort_gene_percentiles("BRCA", proteoform=True)


def test_representatives_provenance_requires_long_format():
    # Provenance is per-representative; asking for it in the default wide form is
    # a no-op, so it must fail loudly rather than silently dropping the request.
    with pytest.raises(ValueError, match="include_provenance=True requires format='long'"):
        expression.representative_cohort_samples("PRAD", include_provenance=True)


def _raw_matrix(tmp_path):
    # A tiny raw-TPM per-sample matrix (genes x samples) whose columns sum near 1e6.
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
            "Symbol": ["A", "B", "C"],
            "s1": [500000.0, 300000.0, 200000.0],
            "s2": [100000.0, 600000.0, 300000.0],
        }
    )
    path = tmp_path / "PRAD.parquet"
    df.to_parquet(path, index=False)
    return path


def test_per_sample_expression_normalize_modes(tmp_path, monkeypatch):
    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)

    raw = expression.per_sample_expression("PRAD", normalize="tpm_raw")
    assert list(raw.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]
    assert raw["s1"].sum() == pytest.approx(1e6)

    clean = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    # No technical/censored genes in this fixture -> the technical compartment is
    # empty and the biological compartment fills its 750k budget (clean_tpm's
    # two-compartment contract). A real matrix with technical genes sums to ~1e6.
    assert clean["s1"].sum() == pytest.approx(750000.0, rel=1e-6)
    assert list(clean.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]

    logged = expression.per_sample_expression("PRAD", normalize="tpm_clean_log1p")
    assert np.allclose(logged["s1"].to_numpy(), np.log1p(clean["s1"].to_numpy()))


def test_per_sample_expression_memoizes_and_copies(tmp_path, monkeypatch):
    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    expression._load_per_sample_matrix.cache_clear()

    a = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    expression.per_sample_expression("PRAD", normalize="tpm_clean")  # served from cache
    info = expression._load_per_sample_matrix.cache_info()
    assert info.misses == 1 and info.hits >= 1

    # Each call returns an independent copy — mutating one must not corrupt the cache.
    col = a.columns[2]
    a.loc[0, col] = -12345.0
    c = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    assert c.loc[0, col] != -12345.0


def test_per_sample_expression_bad_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: _raw_matrix(tmp_path))
    with pytest.raises(ValueError, match="normalize must be one of"):
        expression.per_sample_expression("PRAD", normalize="zscore")


def test_per_sample_expression_no_autofetch_raises(monkeypatch, tmp_path):
    missing = tmp_path / "nope.parquet"
    monkeypatch.setattr(expression.source_matrices, "local_path", lambda code: missing)
    with pytest.raises(FileNotFoundError, match="not cached"):
        expression.per_sample_expression("PRAD", auto_fetch=False)


def test_cohort_mean_expression(monkeypatch):
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "s1": [10.0, 0.0],
            "s2": [20.0, 4.0],
            "s3": [30.0, 2.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: fixture.copy())
    mean = expression.cohort_mean_expression("X", statistic="mean")
    assert list(mean.columns) == ["Ensembl_Gene_ID", "Symbol", "expression"]
    assert dict(zip(mean["Symbol"], mean["expression"])) == {"A": 20.0, "B": 2.0}
    median = expression.cohort_mean_expression("X", statistic="median")
    assert dict(zip(median["Symbol"], median["expression"])) == {"A": 20.0, "B": 2.0}


def test_cohort_mean_expression_scope_threads_to_collapse(monkeypatch):
    # proteoform scope="genome" must reach collapse_to_proteoforms (the universal
    # all-genes key space), not just the CTA subset.
    import cancerdata.proteoforms as pmod

    seen = {}

    def spy_group_map(*, scope="cta"):
        seen["scope"] = scope
        return {}  # no groups -> every gene a singleton (proteoform_key = its ENSG)

    monkeypatch.setattr(pmod, "proteoform_group_map", spy_group_map)
    fixture = pd.DataFrame(
        {"Ensembl_Gene_ID": ["E1", "E2"], "Symbol": ["A", "B"], "s1": [1.0, 2.0], "s2": [3.0, 4.0]}
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: fixture.copy())
    out = expression.cohort_mean_expression("X", proteoform=True, scope="genome")
    assert seen["scope"] == "genome"
    assert pmod.expression_level(out) == "proteoform"


def test_cohort_mean_expression_bad_statistic(monkeypatch):
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="statistic must be"):
        expression.cohort_mean_expression("X", statistic="mode")


def test_cohort_mean_expression_proteoform_collapses_first(monkeypatch):
    # Two identical-protein paralogs are summed per sample BEFORE the across-patient
    # reduction, so the proteoform mean is over the summed values, not per-member.
    import cancerdata.proteoforms as pmod

    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_A2", "ENSG_B"],
            "Symbol": ["A1", "A2", "B"],
            "s1": [3.0, 5.0, 1.0],
            "s2": [1.0, 1.0, 9.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: fixture.copy())
    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A1/A2": ["ENSG_A1", "ENSG_A2"]}
    )
    out = expression.cohort_mean_expression("X", statistic="mean", proteoform=True)
    assert "proteoform_key" in out.columns
    by_key = dict(zip(out["proteoform_key"], out["expression"]))
    # A1+A2 summed per sample = [8, 2], mean 5.0 (NOT each member's mean of ~2/3)
    assert by_key["A1/2"] == pytest.approx(5.0)
    assert by_key["ENSG_B"] == pytest.approx(5.0)  # singleton keyed by ENSG
