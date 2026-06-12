# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unified data catalog over the bundle + HPA backends (#35, D-cat)."""

import pytest

from cancerdata import catalog, cli, data_bundle, reference_data


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
        set(r) == {"name", "kind", "present", "path", "size_bytes", "cohorts", "description"}
        for r in rows
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


# ---- unified `cancerdata data` CLI over the catalog ----


def test_cli_data_list(capsys):
    assert cli.main(["data", "list"]) == 0
    out = capsys.readouterr().out
    for r in catalog.inventory():
        assert r["name"] in out


def test_cli_data_status_absent(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "b" / "vX"))
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path / "ref"))
    assert cli.main(["data", "status"]) == 0
    out = capsys.readouterr().out
    assert "Present" in out and "no" in out


def test_cli_data_status_unknown_errors(capsys):
    assert cli.main(["data", "status", "nope"]) == 1
    assert "unknown dataset" in capsys.readouterr().err


def test_cli_data_path_requires_name(capsys):
    assert cli.main(["data", "path"]) == 1
    assert "requires a dataset name" in capsys.readouterr().err


def test_cli_data_fetch_reports(monkeypatch, capsys):
    monkeypatch.setattr(catalog, "fetch", lambda name="all", *, force=False: ["hpa_normal_tissue"])
    assert cli.main(["data", "fetch", "hpa_normal_tissue"]) == 0
    assert "Downloaded: hpa_normal_tissue" in capsys.readouterr().out


def test_cli_data_fetch_nothing(monkeypatch, capsys):
    monkeypatch.setattr(catalog, "fetch", lambda name="all", *, force=False: [])
    assert cli.main(["data", "fetch"]) == 0
    assert "Already present." in capsys.readouterr().out


def test_cli_help_never_crashes():
    # argparse formats help strings as `%`-templates; an unescaped `%` (e.g. a
    # literal "(%)") raises ValueError at --help time. Walk every parser.
    import argparse

    parser = cli._build_parser()
    parser.format_help()  # top-level: covers every subcommand's help= string
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                subparser.format_help()  # each subcommand's own argument help


# ---- full inventory (manifest-driven) ----


def test_inventory_covers_all_held_buckets():
    from cancerdata import data_manifest

    rows = catalog.inventory()
    names = {r["name"] for r in rows}
    assert set(data_manifest.WHEEL) <= names
    assert set(data_manifest.PLANNED) <= names
    assert {"per-sample-tpm-matrices"} <= names  # the raw source matrices
    held = {r["held"] for r in rows}
    assert held == {"wheel", "bundle", "hpa", "source", "planned"}


def test_inventory_wheel_always_available():
    wheel = [r for r in catalog.inventory() if r["held"] == "wheel"]
    assert wheel and all(r["available"] for r in wheel)


def test_inventory_planned_not_available():
    planned = [r for r in catalog.inventory() if r["held"] == "planned"]
    assert planned and all(not r["available"] for r in planned)


def test_cli_data_list_shows_full_inventory(capsys):
    assert cli.main(["data", "list"]) == 0
    out = capsys.readouterr().out
    assert "housekeeping-genes" in out  # a planned dataset appears
    assert "per-sample-tpm-matrices" in out  # the raw source matrices appear
    assert "planned" in out and "wheel" in out


def test_cohort_count_surfaces_per_cohort_scale(monkeypatch, tmp_path):
    # A directory dataset reports the per-cohort file count (the cancer-type scale
    # held *inside* one catalog entry); single-file/absent datasets report None.
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "vX"))
    member = tmp_path / "vX" / "cancer-reference-expression-percentiles"
    member.mkdir(parents=True)
    for code in ("ACC", "BRCA", "LUAD"):
        (member / f"{code}.parquet").write_bytes(b"x")
    (member / "_provenance.csv").write_text("x")  # excluded (leading underscore)

    rows = {r["name"]: r for r in catalog.status()}
    assert rows["cancer-reference-expression-percentiles"]["cohorts"] == 3
    # a single-file dataset has no per-cohort breakdown
    assert rows["pan-cancer-expression.csv"]["cohorts"] is None
