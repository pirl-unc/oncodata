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


# ---- CTA addressable burden (P2) + 9mer scaffold (P3) ----


def test_cta_addressable_burden_renders(tmp_path, monkeypatch):
    # Hermetic: stub the within-sample prevalence so the test doesn't need the
    # bundle. Real codes map to a burden category + incidence, so bars render.
    monkeypatch.setattr(
        plots,
        "_cta_prevalence_by_cohort",
        lambda threshold: {"LUAD": 0.6, "SKCM": 0.4, "BRCA": 0.2},
    )
    out = tmp_path / "burden.png"
    fig = plots.cta_addressable_burden(n=10, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_addressable_burden_no_within_sample(monkeypatch):
    monkeypatch.setattr(plots, "_cta_prevalence_by_cohort", lambda threshold: {})
    with pytest.raises(ValueError, match="no CTA prevalence available"):
        plots.cta_addressable_burden()


def test_cta_addressable_burden_no_mapped_cohorts(monkeypatch):
    # A code that resolves to no burden category yields no bars -> clean ValueError.
    monkeypatch.setattr(plots, "_cta_prevalence_by_cohort", lambda threshold: {"NOT_A_CODE": 0.5})
    with pytest.raises(ValueError, match="no cohort mapped"):
        plots.cta_addressable_burden()


def test_cta_specific_9mer_counts_not_implemented():
    with pytest.raises(NotImplementedError, match="#15"):
        plots.cta_specific_9mer_counts()


# ---- per-patient plots (consume the per-sample matrices via coverage.py) ----


def test_cta_addressable_burden_per_sample_source(tmp_path, monkeypatch):
    # source="per_sample" pulls the faithful union from coverage; stub it hermetically.
    from cancerdata import coverage

    monkeypatch.setattr(
        coverage,
        "addressable_fraction_by_cohort",
        lambda **k: __import__("pandas").Series({"LUAD": 0.95, "SKCM": 0.99, "BRCA": 0.5}),
    )
    out = tmp_path / "burden_faithful.png"
    fig = plots.cta_addressable_burden(source="per_sample", n=10, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_addressable_burden_bad_source():
    with pytest.raises(ValueError, match="source must be"):
        plots.cta_addressable_burden(source="nonsense")


def test_cta_patient_count_heatmap_renders(tmp_path, monkeypatch):
    import pandas as pd

    from cancerdata import coverage

    def fake_fractions(code, *, threshold_tpm):
        # two cohorts, three CTAs with different per-patient prevalences
        base = {"LUAD": [0.7, 0.3, 0.1], "SKCM": [0.9, 0.2, 0.5]}[code]
        return pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["E1", "E2", "E3"],
                "Symbol": ["GA", "GB", "GC"],
                "fraction_expressing": base,
                "n_patients_expressing": [int(x * 100) for x in base],
                "n_patients": [100, 100, 100],
            }
        )

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM"])
    monkeypatch.setattr(coverage, "cta_patient_fractions", fake_fractions)
    out = tmp_path / "patient_heatmap.png"
    fig = plots.cta_patient_count_heatmap(n_ctas=3, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_patient_count_heatmap_no_cohorts(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    with pytest.raises(ValueError, match="no cohorts with a cached per-sample matrix"):
        plots.cta_patient_count_heatmap()


def test_cta_patient_count_heatmap_duplicate_symbols(tmp_path, monkeypatch):
    # Paralog CTAs sharing a Symbol must not crash the cohort×CTA frame alignment.
    import pandas as pd

    from cancerdata import coverage

    def fake_fractions(code, *, threshold_tpm):
        return pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["E1", "E2", "E3"],
                "Symbol": ["GA", "GA", "GB"],  # GA duplicated (two paralogs)
                "fraction_expressing": [0.7, 0.3, 0.1],
                "n_patients_expressing": [70, 30, 10],
                "n_patients": [100, 100, 100],
            }
        )

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM"])
    monkeypatch.setattr(coverage, "cta_patient_fractions", fake_fractions)
    out = tmp_path / "dup.png"
    fig = plots.cta_patient_count_heatmap(save=str(out))  # must not raise
    assert out.exists() and fig is not None


def test_cta_coverage_curves_renders(tmp_path, monkeypatch):
    import pandas as pd

    from cancerdata import coverage

    def fake_greedy(code, *, threshold_tpm, max_genes):
        return pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "Ensembl_Gene_ID": ["E1", "E2", "E3"],
                "Symbol": ["GA", "GB", "GC"],
                "marginal_patients": [60, 20, 10],
                "marginal_fraction": [0.6, 0.2, 0.1],
                "cumulative_patients": [60, 80, 90],
                "cumulative_fraction": [0.6, 0.8, 0.9],
            }
        )

    monkeypatch.setattr(coverage, "greedy_coverage", fake_greedy)
    out = tmp_path / "curves.png"
    fig = plots.cta_coverage_curves(["LUAD", "SKCM"], save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_coverage_curves_empty_raises(monkeypatch):
    import pandas as pd

    from cancerdata import coverage

    monkeypatch.setattr(coverage, "greedy_coverage", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="no coverage curve"):
        plots.cta_coverage_curves(["LUAD"])


def test_apd1_response_signature_scatter_renders(tmp_path, monkeypatch):
    from cancerdata import response_signatures as rs

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM", "MM"])
    monkeypatch.setattr(
        plots, "cancer_apd1_response", lambda: {"LUAD": 19.0, "SKCM": 42.0, "MM": 3.0}
    )
    monkeypatch.setattr(
        rs, "signature_score", lambda code, sig, **k: {"LUAD": 3.1, "SKCM": 2.9, "MM": 1.0}[code]
    )
    out = tmp_path / "sig.png"
    fig = plots.apd1_response_signature_scatter("t_cell_inflamed", save=str(out))
    assert out.exists() and fig is not None


def test_apd1_response_signature_scatter_no_data(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    with pytest.raises(ValueError, match="no cohort with both"):
        plots.apd1_response_signature_scatter("t_cell_inflamed")


def test_apd1_response_signature_bad_name():
    with pytest.raises(ValueError, match="unknown signature"):
        plots.apd1_response_signature_scatter("not_a_signature")
