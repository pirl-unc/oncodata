# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from cancerdata import data_bundle


def test_is_downloadable_distinguishes_bundle_from_wheel():
    assert data_bundle.is_downloadable("cancer-reference-expression")
    assert data_bundle.is_downloadable("cancer-reference-expression-percentiles")
    assert data_bundle.is_downloadable("pan-cancer-expression.csv")
    # Small wheel-bundled tables are NOT downloadable items.
    assert not data_bundle.is_downloadable("cancer-type-registry")
    assert not data_bundle.is_downloadable("cancer-tmb")


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "vX"))
    assert data_bundle.cache_dir() == tmp_path / "vX"
    assert data_bundle.cache_root() == tmp_path


def test_legacy_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("CANCERDATA_BUNDLED_DATA", raising=False)
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(tmp_path / "vY"))
    assert data_bundle.cache_dir() == tmp_path / "vY"


def test_new_env_wins_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "new"))
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(tmp_path / "old"))
    assert data_bundle.cache_dir() == tmp_path / "new"


def test_status_reports_missing_without_download(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    snap = data_bundle.status()
    assert snap["all_local"] is False
    assert set(snap["items"]) == set(data_bundle.DOWNLOADABLE_PATHS)
    assert all(not v["present"] for v in snap["items"].values())


def test_prune_keeps_current(monkeypatch, tmp_path):
    root = tmp_path / "bundled_data"
    (root / "v1.0.0").mkdir(parents=True)
    (root / "v2.0.0").mkdir(parents=True)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root / "v2.0.0"))
    planned = data_bundle.prune_cache(dry_run=True)
    versions = {e["version"] for e in planned}
    assert versions == {"v1.0.0"}  # current (v2.0.0) is kept
