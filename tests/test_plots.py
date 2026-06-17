# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest

pytest.importorskip("matplotlib")

from oncoref import cli, plots


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


def test_cli_plot_burden_category_bars(tmp_path):
    out = tmp_path / "cats.png"
    assert cli.main(["plot", "burden-category-bars", "--region", "world", "--out", str(out)]) == 0
    assert out.exists()


def test_cli_plot_coverage_stacked_needs_codes(capsys):
    # The coverage plots require --codes; missing it is a clean error, not a crash.
    assert cli.main(["plot", "cta-coverage-stacked", "--out", "x.png"]) == 1


# ---- CTA expression heatmap (needs the expression bundle / percentile data) ----

_HAS_PERCENTILES = bool(__import__("oncoref").available_percentile_cohorts())
_needs_bundle = pytest.mark.skipif(
    not _HAS_PERCENTILES, reason="expression bundle (percentile artifacts) not present"
)


@_needs_bundle
def test_cta_expression_heatmap_renders(tmp_path):
    out = tmp_path / "cta.png"
    cohorts = __import__("oncoref").available_percentile_cohorts()[:6]
    fig = plots.cta_expression_heatmap(cohorts=cohorts, n_cohorts=4, n_ctas=8, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_expression_heatmap_bad_stat():
    with pytest.raises(ValueError, match="stat must be one of"):
        plots.cta_expression_heatmap(stat="mean", cohorts=["X"])


def test_cta_expression_heatmap_proteoform_reads_collapsed_vectors(tmp_path, monkeypatch):
    # proteoform=True reads the proteoform-summed percentile vectors and labels CTA
    # columns by the proteoform symbol (NY-ESO-1). Stub the per-cohort vector.
    import pandas as pd

    from oncoref import cta
    from oncoref.proteoforms import gene_to_proteoform_id

    # Use two real CTAs and their real proteoform keys, so the heatmap's CTA-key filter
    # (derived from the same CTA set) keeps both rows. At least one is a collapsed group.
    grouped = next(g for g in sorted(cta.cta_gene_ids()) if "/" in gene_to_proteoform_id([g])[g])
    singleton = next(
        g for g in sorted(cta.cta_gene_ids()) if "/" not in gene_to_proteoform_id([g])[g]
    )
    keys = gene_to_proteoform_id([grouped, singleton])

    def fake_pct(code, *, as_tpm=True, proteoform=False):
        assert proteoform  # the heatmap must request the collapsed variant
        return pd.DataFrame(
            {
                "proteoform_key": [keys[grouped], keys[singleton]],
                "Ensembl_Gene_ID": [grouped, singleton],
                "Symbol": [keys[grouped], "OTHER_CTA"],
                "proteoform_members": ["CTAG1A/CTAG1B", "OTHER_CTA"],
                "p25": [5.0, 1.0],
                "p50": [40.0, 2.0],
                "p75": [80.0, 3.0],
            }
        )

    monkeypatch.setattr(plots, "cohort_gene_percentiles", fake_pct)
    monkeypatch.setattr(plots, "available_percentile_cohorts", lambda *, proteoform=False: ["LUAD"])
    out = tmp_path / "cta_pf.png"
    fig = plots.cta_expression_heatmap(proteoform=True, n_cohorts=2, n_ctas=4, save=str(out))
    assert out.exists() and fig is not None


def test_cta_expression_heatmap_proteoform_missing_bundle():
    with pytest.raises(ValueError, match="proteoform-summed percentile vector"):
        plots.cta_expression_heatmap(proteoform=True, cohorts=[])


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


def test_cta_specific_9mer_load_renders(tmp_path, monkeypatch):
    from oncoref import peptides

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM", "MM"])
    monkeypatch.setattr(plots, "cancer_tmb", lambda: {"LUAD": 6.9, "SKCM": 13.0, "MM": 1.5})
    monkeypatch.setattr(
        peptides,
        "cta_specific_9mer_load",
        lambda code, **k: {"LUAD": 120.0, "SKCM": 540.0, "MM": 60.0}[code],
    )
    out = tmp_path / "9mer.png"
    fig = plots.cta_specific_9mer_load(against="tmb", save=str(out))
    assert out.exists() and fig is not None


def test_cta_specific_9mer_load_bad_against():
    with pytest.raises(ValueError, match="against must be"):
        plots.cta_specific_9mer_load(against="nonsense")


def test_cta_specific_9mer_load_no_data(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    with pytest.raises(ValueError, match="no cohort with both"):
        plots.cta_specific_9mer_load()


# ---- per-patient plots (consume the per-sample matrices via coverage.py) ----


def test_cta_addressable_burden_per_sample_source(tmp_path, monkeypatch):
    # source="per_sample" pulls the faithful union from coverage; stub it hermetically.
    from oncoref import coverage

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

    from oncoref import coverage

    def fake_fractions(code, *, threshold_tpm):
        # two cohorts, three CTAs with different per-patient prevalences
        base = {"LUAD": [0.7, 0.3, 0.1], "SKCM": [0.9, 0.2, 0.5]}[code]
        return pd.DataFrame(
            {
                "proteoform_key": ["E1", "E2", "E3"],
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
    # Distinct proteoforms that happen to share a display Symbol must not crash the
    # cohort×CTA frame alignment — keying is on the unique proteoform_key.
    import pandas as pd

    from oncoref import coverage

    def fake_fractions(code, *, threshold_tpm):
        return pd.DataFrame(
            {
                "proteoform_key": ["K1", "K2", "K3"],  # unique keys
                "Ensembl_Gene_ID": ["E1", "E2", "E3"],
                "Symbol": ["GA", "GA", "GB"],  # GA duplicated as a display label
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

    from oncoref import coverage

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

    from oncoref import coverage

    monkeypatch.setattr(coverage, "greedy_coverage", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="no coverage curve"):
        plots.cta_coverage_curves(["LUAD"])


def test_antigen_family_grouping():
    assert plots._antigen_family("MAGEA4") == "MAGE-A"
    assert plots._antigen_family("MAGEB2") == "MAGE-B"
    assert plots._antigen_family("CTAG1B") == "CTAG/NY-ESO"
    assert plots._antigen_family("LAGE1") == "CTAG/NY-ESO"  # same family as CTAG
    assert plots._antigen_family("SSX2") == "SSX"
    assert plots._antigen_family("CT83") == "other"


def test_cta_coverage_stacked_bars_renders(tmp_path, monkeypatch):
    import pandas as pd

    from oncoref import coverage

    def fake_greedy(code, *, threshold_tpm, max_genes):
        return pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "Ensembl_Gene_ID": ["E1", "E2", "E3"],
                "Symbol": ["MAGEA4", "CTAG1B", "SSX2"],
                "marginal_patients": [60, 20, 10],
                "marginal_fraction": [0.6, 0.2, 0.1],
                "cumulative_patients": [60, 80, 90],
                "cumulative_fraction": [0.6, 0.8, 0.9],
            }
        )

    monkeypatch.setattr(coverage, "greedy_coverage", fake_greedy)
    out = tmp_path / "stacked.png"
    fig = plots.cta_coverage_stacked_bars(["LUAD", "SKCM"], save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_coverage_stacked_bars_empty_raises(monkeypatch):
    import pandas as pd

    from oncoref import coverage

    monkeypatch.setattr(coverage, "greedy_coverage", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="no coverage to plot"):
        plots.cta_coverage_stacked_bars(["LUAD"])


def test_burden_category_bars_renders(tmp_path):
    out = tmp_path / "cats.png"
    fig = plots.burden_category_bars(region="us", n=8, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_burden_category_bars_bad_region():
    with pytest.raises(ValueError, match="region must be"):
        plots.burden_category_bars(region="moon")


def test_cta_burden_vs_response_renders(tmp_path, monkeypatch):
    from oncoref import coverage

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM", "MM"])
    monkeypatch.setattr(
        plots, "cancer_apd1_response", lambda: {"LUAD": 19.0, "SKCM": 42.0, "MM": 3.0}
    )
    monkeypatch.setattr(
        coverage,
        "mean_antigens_per_patient",
        lambda code, **k: {"LUAD": 1.5, "SKCM": 4.0, "MM": 8.0}[code],
    )
    out = tmp_path / "load.png"
    fig = plots.cta_burden_vs_response(against="apd1", save=str(out))
    assert out.exists() and fig is not None


def test_cta_burden_vs_response_bad_against():
    with pytest.raises(ValueError, match="against must be"):
        plots.cta_burden_vs_response(against="nonsense")


def test_cta_burden_vs_response_no_data(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    with pytest.raises(ValueError, match="no cohort with both"):
        plots.cta_burden_vs_response()


def test_apd1_response_signature_scatter_renders(tmp_path, monkeypatch):
    from oncoref import response_signatures as rs

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


def test_regenerate_plots_runner_references_real_functions():
    # Guard: every (family, name, fn_attr, kwargs) job in the batch runner must
    # name a real callable on oncoref.plots — catches typos like a removed or
    # renamed figure before a full run does.
    import importlib.util
    from pathlib import Path

    from oncoref import plots

    runner = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_plots.py"
    spec = importlib.util.spec_from_file_location("_regen_plots", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    jobs = mod._jobs()
    assert jobs, "runner produced no jobs"
    for family, name, fn_attr, kwargs in jobs:
        assert family and name and isinstance(kwargs, dict)
        assert callable(getattr(plots, fn_attr, None)), f"{fn_attr} is not a plots function"


def test_top_cohorts_by_samples_caps_and_ranks():
    from oncoref import plots

    # top_n=None or fewer codes than the cap -> unchanged (order preserved)
    assert plots._top_cohorts_by_samples(["A", "B"], None) == ["A", "B"]
    assert plots._top_cohorts_by_samples(["A", "B"], 5) == ["A", "B"]
    # caps to the largest cohorts by sample count, deterministically
    counts = plots._cohort_sample_counts()
    big = sorted(counts, key=lambda c: counts[c], reverse=True)[:50]
    top = plots._top_cohorts_by_samples(big, 10)
    assert len(top) == 10
    assert top == sorted(big, key=lambda c: (-counts[c], c))[:10]
