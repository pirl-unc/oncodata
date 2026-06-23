import pandas as pd

import oncoref
from oncoref import antigen_coverage, apd1, cta_coverage, cta_peptides, ici, ici_response


def test_semantic_modules_are_top_level_facades():
    for name in (
        "antigen_coverage",
        "cancer_ontology",
        "cohorts",
        "cta_coverage",
        "cta_peptides",
        "ici_response",
    ):
        assert hasattr(oncoref, name)
        assert name in oncoref.__all__


def test_clearer_names_stay_in_semantic_modules_not_flat_namespace():
    for name in (
        "best_available_ici_response",
        "ici_response_by_regimen",
        "cta_specific_9mer_count_map",
        "cta_addressable_fraction",
        "addressable_antigen_fraction",
    ):
        assert not hasattr(oncoref, name)


def test_ici_response_facade_delegates_to_compatibility_api():
    assert ici_response.DEFAULT_ICI_REGIMEN_PRIORITY == ici.REGIMEN_FALLBACK
    assert ici_response.ICI_REGIMEN_LABELS == ici.REGIMEN_LABELS
    assert ici_response.RESPONSE_PROPORTION_METRICS == ici.PROPORTION_METRICS

    assert ici_response.apd1_response("SKCM") == apd1.cancer_apd1_response("SKCM")
    assert ici_response.best_available_ici_response("SKCM") == ici.cancer_ici_response("SKCM")
    assert ici_response.ici_response_by_regimen("SKCM") == ici.cancer_ici_response(
        "SKCM", fallback=False
    )
    assert ici_response.selected_ici_regimen("SKCM") == ici.cancer_ici_regimen("SKCM")
    pd.testing.assert_frame_equal(
        ici_response.ici_response_anchor_df(), ici.cancer_ici_response_df()
    )


def test_cta_peptides_clear_count_map_name(monkeypatch):
    monkeypatch.setattr(
        cta_peptides,
        "cta_specific_9mer_weights",
        lambda *, k=9, by="proteoform_key": {f"{by}:{k}": 7},
    )
    assert cta_peptides.cta_specific_9mer_count_map(k=3, by="symbol") == {"symbol:3": 7}


def test_cta_coverage_names_make_cta_default_explicit(monkeypatch):
    calls = []

    def fake_addressable(cancer_type, *, threshold_tpm, proteoform, **kwargs):
        calls.append((cancer_type, threshold_tpm, proteoform, kwargs))
        return 0.5

    monkeypatch.setattr(cta_coverage, "addressable_fraction", fake_addressable)
    assert cta_coverage.cta_addressable_fraction("LUAD", threshold_tpm=5) == 0.5
    assert calls == [("LUAD", 5, True, {})]


def test_antigen_coverage_requires_explicit_gene_panel(monkeypatch):
    calls = []

    def fake_addressable(cancer_type, *, threshold_tpm, gene_ids, proteoform):
        calls.append((cancer_type, threshold_tpm, gene_ids, proteoform))
        return 0.25

    monkeypatch.setattr(antigen_coverage, "addressable_fraction", fake_addressable)
    assert (
        antigen_coverage.addressable_antigen_fraction(
            "LUAD", gene_ids={"ENSG00000141510"}, threshold_tpm=2
        )
        == 0.25
    )
    assert calls == [("LUAD", 2, {"ENSG00000141510"}, True)]
