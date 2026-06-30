# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-cohort raw per-sample matrix fetch (#35, D-source)."""

import urllib.error

import pandas as pd
import pytest

from oncoref import catalog
from oncoref import source_matrices as sm


def test_registry_and_available_cohorts():
    cohorts = sm.available_cohorts()
    assert len(cohorts) >= 118
    assert "LUAD" in cohorts and "BRCA" in cohorts
    info = sm.cohort_info("LUAD")
    assert info["source_cohort"] and info["n_samples"] > 0


def test_alias_resolves():
    assert sm.local_path("lung_adeno").name == "LUAD.parquet"


def test_unknown_cohort_raises():
    with pytest.raises(sm.SourceMatrixError, match="no per-sample matrix"):
        sm.local_path("NOT_A_COHORT")


def test_release_url_is_per_cohort():
    url = sm.release_url("LUAD")
    assert url.endswith("LUAD_per_sample_tpm.parquet")
    # source matrices are pinned to SOURCE_MATRIX_VERSION (raw inputs), independent of the
    # derived-bundle DATA_VERSION, so a canonical-space bundle bump can't repoint them.
    assert f"source-v{sm.SOURCE_MATRIX_VERSION}" in url


def test_cache_and_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    assert not sm.is_cached("LUAD")

    def fake_urlopen(url):
        import io

        class _R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _R(b"PAR1-fake")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = sm.ensure("LUAD")
    assert p.exists() and p.read_bytes() == b"PAR1-fake"
    assert sm.is_cached("LUAD")
    # second ensure is a no-op (already cached)
    assert sm.ensure("LUAD") == p


def test_fetch_download_failure_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))

    def boom(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(sm.SourceMatrixError, match="failed to download"):
        sm.fetch("BRCA")
    assert not sm.is_cached("BRCA")  # no partial file left


def test_sample_qc_facade_uses_shared_expression_policy(monkeypatch):
    from oncoref import expression

    calls = {}

    def fake_sample_expression_qc(code, **kwargs):
        calls["code"] = code
        calls["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "sample_id": ["S1"],
                "sample_qc_status": ["pass"],
                "source_scale_class": ["linear_rnaseq_tpm"],
            }
        )

    monkeypatch.setattr(expression, "sample_expression_qc", fake_sample_expression_qc)

    out = sm.sample_qc("lung_adeno", auto_fetch=False, min_detected_genes=123)

    assert calls == {
        "code": "lung_adeno",
        "kwargs": {"auto_fetch": False, "min_detected_genes": 123},
    }
    assert out["sample_qc_status"].tolist() == ["pass"]


def test_sample_qc_manifest_semantic_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    pd.DataFrame(
        {
            "cancer_code": ["LUAD", "LUAD", "BRCA"],
            "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
            "sample_qc_status": ["pass", "warn", "fail"],
            "source_cohort": ["SRC", "SRC", "SRC2"],
        }
    ).to_csv(tmp_path / "source-matrix-sample-qc.csv", index=False)

    out = sm.sample_qc_manifest("lung_adeno", sample_qc="pass_or_warn", auto_fetch=False)

    assert out["sample_id"].tolist() == ["pass_sample", "warn_sample"]
    assert out.attrs["path"].endswith("source-matrix-sample-qc.csv")


# ---- catalog routing + group fetch ----


def test_catalog_per_sample_addressing(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    assert catalog.path("per-sample:LUAD") is None
    called = {}
    monkeypatch.setattr(
        sm, "ensure", lambda code: called.setdefault("code", code) or tmp_path / "x"
    )
    catalog.ensure("per-sample:LUAD")
    assert called["code"] == "LUAD"


def test_catalog_fetch_groups(monkeypatch):
    from oncoref import data_bundle, reference_data

    # hpa group -> only HPA sources downloaded
    monkeypatch.setattr(
        reference_data,
        "local_path",
        lambda n, *a, **k: type("P", (), {"exists": lambda s: False})(),
    )
    hpa_calls = []
    monkeypatch.setattr(reference_data, "download", lambda n, *a, **k: hpa_calls.append(n))
    monkeypatch.setattr(
        data_bundle, "fetch", lambda *a, **k: pytest.fail("bundle must not fetch for hpa group")
    )
    out = catalog.fetch("hpa")
    assert set(out) == set(reference_data.REFERENCE_SOURCES)
    assert set(hpa_calls) == set(reference_data.REFERENCE_SOURCES)


def test_catalog_fetch_source_group(monkeypatch):
    fetched = []
    monkeypatch.setattr(sm, "is_cached", lambda c: False)
    monkeypatch.setattr(sm, "fetch", lambda c, *, force=False: fetched.append(c))
    out = catalog.fetch("source")
    assert len(out) == len(sm.available_cohorts())
    assert out[0].startswith("per-sample:")
