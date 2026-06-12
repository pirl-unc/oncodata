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


def test_representatives_provenance_requires_long_format():
    # Provenance is per-representative; asking for it in the default wide form is
    # a no-op, so it must fail loudly rather than silently dropping the request.
    with pytest.raises(ValueError, match="include_provenance=True requires format='long'"):
        expression.representative_cohort_samples("PRAD", include_provenance=True)
