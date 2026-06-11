# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unified data catalog over the bundle + HPA backends (#35, D-cat)."""

import pytest

from cancerdata import catalog, data_bundle, reference_data


def test_datasets_span_both_backends():
    ds = catalog.datasets()
    kinds = {d.kind for d in ds}
    assert kinds == {"bundle", "hpa"}
    names = {d.name for d in ds}
    assert set(data_bundle.DOWNLOADABLE_PATHS) <= names
    assert set(reference_data.REFERENCE_SOURCES) <= names


def test_dataset_lookup_and_unknown():
    assert catalog.dataset("hpa_normal_tissue").kind == "hpa"
    assert catalog.dataset("cancer-reference-expression-percentiles").kind == "bundle"
    with pytest.raises(KeyError, match="unknown dataset"):
        catalog.dataset("not_a_dataset")


def test_status_uniform_and_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "bundle" / "vX"))
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path / "ref"))
    rows = catalog.status()
    assert {r["name"] for r in rows} == {d.name for d in catalog.datasets()}
    assert all(
        set(r) == {"name", "kind", "present", "path", "size_bytes", "description"} for r in rows
    )
    assert all(r["present"] is False and r["size_bytes"] == 0 for r in rows)


def test_path_none_when_absent_then_present(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path / "ref"))
    assert catalog.path("hpa_normal_tissue") is None
    # Materialize the HPA file where reference_data expects it.
    target = reference_data.local_path("hpa_normal_tissue")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Gene\tTissue\n")
    assert catalog.path("hpa_normal_tissue") == target
    assert catalog.status("hpa_normal_tissue")[0]["present"] is True


def test_ensure_delegates_to_backends(monkeypatch, tmp_path):
    # HPA -> reference_data.ensure
    called = {}

    def fake_hpa_ensure(n, *a, **k):
        called["hpa"] = n
        return tmp_path / "h.tsv"

    monkeypatch.setattr(reference_data, "ensure", fake_hpa_ensure)
    assert catalog.ensure("hpa_single_cell") == tmp_path / "h.tsv"
    assert called["hpa"] == "hpa_single_cell"

    # bundle -> ensure_local() then find()
    def fake_ensure_local(*a, **k):
        called["bundle_fetch"] = True

    monkeypatch.setattr(data_bundle, "ensure_local", fake_ensure_local)
    monkeypatch.setattr(data_bundle, "find", lambda n: tmp_path / n)
    out = catalog.ensure("pan-cancer-expression.csv")
    assert out == tmp_path / "pan-cancer-expression.csv"
    assert called["bundle_fetch"] is True


def test_fetch_all_fetches_bundle_once_and_each_hpa(monkeypatch):
    calls = {"bundle_fetch": 0, "hpa": []}
    monkeypatch.setattr(data_bundle, "is_local", lambda: False)
    monkeypatch.setattr(
        data_bundle,
        "fetch",
        lambda *a, **k: calls.__setitem__("bundle_fetch", calls["bundle_fetch"] + 1),
    )
    monkeypatch.setattr(reference_data, "download", lambda n, *a, **k: calls["hpa"].append(n))

    fetched = catalog.fetch("all")
    # The tarball is fetched exactly once despite 5 bundle members.
    assert calls["bundle_fetch"] == 1
    assert set(calls["hpa"]) == set(reference_data.REFERENCE_SOURCES)
    assert set(fetched) == {d.name for d in catalog.datasets()}


def test_fetch_reports_only_actual_downloads(monkeypatch):
    # Everything already cached + no force -> nothing is reported as downloaded.
    monkeypatch.setattr(data_bundle, "is_local", lambda: True)
    monkeypatch.setattr(data_bundle, "fetch", lambda *a, **k: pytest.fail("should not fetch"))

    class _P:
        def exists(self):
            return True

    monkeypatch.setattr(reference_data, "local_path", lambda n, *a, **k: _P())
    monkeypatch.setattr(reference_data, "download", lambda n, *a, **k: None)
    assert catalog.fetch("all") == []


def test_datasets_disjoint_backends():
    # The catalog routes by name; the two backends must never share a name.
    bundle = set(data_bundle.DOWNLOADABLE_PATHS)
    hpa = set(reference_data.REFERENCE_SOURCES)
    assert bundle.isdisjoint(hpa)
