# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import tarfile
import urllib.error

import pytest

from oncodata import data_bundle
from oncodata.version import DATA_VERSION


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


def test_is_local_requires_nonempty_dirs(monkeypatch, tmp_path):
    # #21: an interrupted extract that created the shard directories but no
    # shards must NOT read as "local" (else ensure_local never re-fetches).
    root = tmp_path / f"v{DATA_VERSION}"
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root))
    # Create every downloadable path but leave the directories empty.
    for p in data_bundle.DOWNLOADABLE_PATHS:
        target = root / p
        if target.suffix:  # a file entry
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")
        else:  # a directory entry — created but empty
            target.mkdir(parents=True, exist_ok=True)
    assert data_bundle.is_local() is False
    # Now drop a shard into each directory entry -> complete.
    for p in data_bundle.DOWNLOADABLE_PATHS:
        target = root / p
        if not target.suffix:
            (target / "shard.parquet").write_text("data")
    assert data_bundle.is_local() is True


def test_prune_keeps_current(monkeypatch, tmp_path):
    root = tmp_path / "bundled_data"
    (root / "v1.0.0").mkdir(parents=True)
    (root / "v2.0.0").mkdir(parents=True)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root / "v2.0.0"))
    planned = data_bundle.prune_cache(dry_run=True)
    versions = {e["version"] for e in planned}
    assert versions == {"v1.0.0"}  # current (v2.0.0) is kept


def test_release_urls_prefer_oncodata_then_pirlygenes():
    assert data_bundle.RELEASE_URLS == (
        data_bundle.RELEASE_URL,
        data_bundle.FALLBACK_RELEASE_URL,
    )
    assert "pirl-unc/oncodata" in data_bundle.RELEASE_URL
    assert "pirl-unc/pirlygenes" in data_bundle.FALLBACK_RELEASE_URL
    assert f"v{DATA_VERSION}" in data_bundle.RELEASE_URL


def test_fetch_falls_back_when_primary_404s(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / f"v{DATA_VERSION}"))
    attempted = []

    def fake_download(url, root, *, verbose):
        attempted.append(url)
        if url == data_bundle.RELEASE_URL:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        # fallback "succeeds"

    monkeypatch.setattr(data_bundle, "_download_and_extract", fake_download)
    out = data_bundle.fetch(verbose=False)
    assert out == data_bundle.cache_dir()
    assert attempted == [data_bundle.RELEASE_URL, data_bundle.FALLBACK_RELEASE_URL]


def test_fetch_falls_back_on_corrupt_primary_tarball(monkeypatch, tmp_path):
    # A 200 response whose body isn't a valid tar (e.g. an HTML error page) must
    # fall back to the next source, not propagate.
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / f"v{DATA_VERSION}"))
    attempted = []

    def fake_download(url, root, *, verbose):
        attempted.append(url)
        if url == data_bundle.RELEASE_URL:
            raise tarfile.ReadError("not a gzip file")

    monkeypatch.setattr(data_bundle, "_download_and_extract", fake_download)
    out = data_bundle.fetch(verbose=False)
    assert out == data_bundle.cache_dir()
    assert attempted == [data_bundle.RELEASE_URL, data_bundle.FALLBACK_RELEASE_URL]


def test_fetch_raises_when_all_sources_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / f"v{DATA_VERSION}"))

    def always_404(url, root, *, verbose):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data_bundle, "_download_and_extract", always_404)
    with pytest.raises(RuntimeError, match="could not download"):
        data_bundle.fetch(verbose=False)


def test_cache_root_reuses_legacy_pirlygenes_dir(monkeypatch, tmp_path):
    # No env override; legacy cache already has THIS version, new root does not.
    monkeypatch.delenv("CANCERDATA_BUNDLED_DATA", raising=False)
    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA", raising=False)
    new_root = tmp_path / "oncodata" / "bundled_data"
    legacy_root = tmp_path / "pirlygenes" / "bundled_data"
    (legacy_root / f"v{DATA_VERSION}").mkdir(parents=True)
    monkeypatch.setattr(data_bundle, "_DEFAULT_CACHE_PARENT", new_root)
    monkeypatch.setattr(data_bundle, "_LEGACY_CACHE_PARENT", legacy_root)

    assert data_bundle.cache_root() == legacy_root  # reuse, no forced re-download
    # Once the new root has this version too, it wins.
    (new_root / f"v{DATA_VERSION}").mkdir(parents=True)
    assert data_bundle.cache_root() == new_root


def test_cache_root_defaults_to_oncodata_when_no_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("CANCERDATA_BUNDLED_DATA", raising=False)
    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA", raising=False)
    new_root = tmp_path / "oncodata" / "bundled_data"
    legacy_root = tmp_path / "pirlygenes" / "bundled_data"
    monkeypatch.setattr(data_bundle, "_DEFAULT_CACHE_PARENT", new_root)
    monkeypatch.setattr(data_bundle, "_LEGACY_CACHE_PARENT", legacy_root)
    assert data_bundle.cache_root() == new_root
