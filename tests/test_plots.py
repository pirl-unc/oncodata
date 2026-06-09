# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

pytest.importorskip("matplotlib")

from cancerdata import cli, plots


def test_apd1_vs_tmb_renders(tmp_path):
    out = tmp_path / "apd1_vs_tmb.png"
    fig = plots.apd1_vs_tmb(save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_apd1_orr_bars_renders(tmp_path):
    out = tmp_path / "bars.png"
    plots.apd1_orr_bars(save=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_incidence_vs_mortality_renders(tmp_path):
    out = tmp_path / "inc.png"
    plots.incidence_vs_mortality(region="world", save=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_incidence_bad_region():
    with pytest.raises(ValueError, match="region must be"):
        plots.incidence_vs_mortality(region="moon")


def test_cli_plot(tmp_path):
    out = tmp_path / "cli.png"
    assert cli.main(["plot", "apd1-vs-tmb", "--out", str(out)]) == 0
    assert out.exists()
