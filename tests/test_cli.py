# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

from cancerdata import cli


def test_version(capsys):
    assert cli.main(["version"]) == 0
    assert "cancerdata v" in capsys.readouterr().out


def test_proteoforms_gene_lookup(capsys):
    assert cli.main(["proteoforms", "--gene", "SSX4B"]) == 0
    assert "SSX4/SSX4B" in capsys.readouterr().out


def test_proteoforms_unknown_gene_errors(capsys):
    assert cli.main(["proteoforms", "--gene", "PRAME"]) == 1
    assert "not in any proteoform group" in capsys.readouterr().err


def test_proteoforms_count(capsys):
    assert cli.main(["proteoforms", "--count"]) == 0
    # The derivation finds the four anchor pairs plus the larger families
    # (CT47A, GAGE12, CT45A, …); a healthy registry is well into double digits.
    assert int(capsys.readouterr().out.strip()) >= 10


def test_cancer_type_prints_json(capsys):
    assert cli.main(["cancer-type", "prostate"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["code"] == "PRAD"


def test_cancer_type_unknown_errors(capsys):
    assert cli.main(["cancer-type", "not_a_real_cancer"]) == 1
    assert "Error" in capsys.readouterr().err


def test_tmb_single_code(capsys):
    assert cli.main(["tmb", "melanoma"]) == 0
    out = capsys.readouterr().out.strip()
    assert float(out) > 0


def test_tmb_full_map(capsys):
    assert cli.main(["tmb"]) == 0
    assert "\t" in capsys.readouterr().out


def test_burden_full_map(capsys):
    assert cli.main(["burden", "--metric", "us_mortality_pct"]) == 0
    assert "\t" in capsys.readouterr().out


def test_cache_dir_respects_env(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    assert cli.main(["cache-dir"]) == 0
    out = capsys.readouterr().out.strip()
    assert str(tmp_path) in out


def test_status_no_download(capsys, monkeypatch, tmp_path):
    # status must never trigger a fetch, even with an empty cache dir.
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "All local:    no" in out
    assert "cancer-reference-expression" in out


def test_prune_dry_run_default(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "v0.0.0"))
    assert cli.main(["prune"]) == 0
    assert "Nothing to prune" in capsys.readouterr().out
