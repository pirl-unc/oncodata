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


# ---- CTA expression heatmap (needs the expression bundle / percentile data) ----

_HAS_PERCENTILES = bool(__import__("cancerdata").available_percentile_cohorts())
_needs_bundle = pytest.mark.skipif(
    not _HAS_PERCENTILES, reason="expression bundle (percentile artifacts) not present"
)


@_needs_bundle
def test_cta_expression_heatmap_renders(tmp_path):
    out = tmp_path / "cta.png"
    cohorts = __import__("cancerdata").available_percentile_cohorts()[:6]
    fig = plots.cta_expression_heatmap(cohorts=cohorts, n_cohorts=4, n_ctas=8, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_expression_heatmap_bad_stat():
    with pytest.raises(ValueError, match="stat must be one of"):
        plots.cta_expression_heatmap(stat="mean", cohorts=["X"])


def test_cta_expression_heatmap_proteoform_not_implemented():
    with pytest.raises(NotImplementedError, match="#13"):
        plots.cta_expression_heatmap(proteoform=True, cohorts=["X"])


def test_cta_expression_heatmap_no_cohorts():
    with pytest.raises(ValueError, match="no cohorts"):
        plots.cta_expression_heatmap(cohorts=[])


def test_cta_expression_heatmap_skips_unusable_cohorts():
    # A code with no percentile vector is skipped with a warning, not a crash; with
    # only unusable codes the matrix is empty -> clean ValueError.
    with (
        pytest.warns(UserWarning, match="without a percentile vector"),
        pytest.raises(ValueError, match="no CTA expression data"),
    ):
        plots.cta_expression_heatmap(cohorts=["NOT_A_REAL_COHORT"])
