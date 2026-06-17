# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from oncoref import ici

# PMIDs that the reference audit found pointing to UNRELATED papers (birds, diabetic
# retinopathy, PheWAS methodology, mitochondrial stem cells, etc.). They were corrupted
# in both the ICI and apd1 tables and have been replaced with verified citations; this
# list guards against regression.
CORRUPTED_PMIDS = {
    "PMID:33052747",  # -> NSCLC ctDNA review (was HNSC/KEYNOTE-048)
    "PMID:29260193",  # -> diabetic retinopathy (was STAD/KEYNOTE-059)
    "PMID:27269732",  # -> mitochondrial stem cells (was SCLC/CheckMate-032)
    "PMID:31218020",  # -> evolution of birds (was OV/KEYNOTE-100)
    "PMID:33125908",  # -> trial-design commentary (was MESO/KEYNOTE-158)
    "PMID:34272311",  # -> PheWAS methodology (was KICH/KIRP/KEYNOTE-427)
    "PMID:33812497",  # -> CRC quality-of-life (was BCC/cemiplimab)
    "PMID:31562797",  # -> melanoma 5y (was LUAD combo/CheckMate-227)
    "PMID:32319072",  # -> boron/rat duodenum (was CHOL/KEYNOTE-158)
}


def _pmids(df):
    return {str(v) for v in df["pmid_doi"] if isinstance(v, str) and v.startswith("PMID:")}


def test_no_corrupted_citations_remain():
    ici_pmids = _pmids(ici.cancer_ici_response_df())
    from oncoref import apd1

    apd1_pmids = {
        str(v)
        for v in apd1.cancer_apd1_response_df()["pmid_doi"]
        if isinstance(v, str) and v.startswith("PMID:")
    }
    assert CORRUPTED_PMIDS.isdisjoint(ici_pmids), CORRUPTED_PMIDS & ici_pmids
    assert CORRUPTED_PMIDS.isdisjoint(apd1_pmids), CORRUPTED_PMIDS & apd1_pmids


def test_citation_format_well_formed():
    df = ici.cancer_ici_response_df()
    for v in df["pmid_doi"]:
        if v is None or (isinstance(v, float)):  # blank cells parse as NaN/float
            continue
        s = str(v).strip()
        assert s == "" or s.startswith("PMID:") or s.startswith("DOI:"), s


def test_estimates_table_shape_and_coverage():
    est = ici.cancer_ici_response_estimates_df()
    expected = {
        "cancer_code",
        "regimen",
        "role",
        "drug",
        "trial",
        "ref",
        "metric",
        "value",
        "unit",
        "ci_low",
        "ci_high",
        "metric_n",
        "responders",
        "source_verified",
    }
    assert expected <= set(est.columns)
    assert len(est) > 800
    # every (cancer, regimen) in the curated anchor table has >=1 estimate row
    anchor = ici.cancer_ici_response_df()
    anchor_cells = set(zip(anchor["cancer_code"], anchor["regimen"]))
    est_cells = set(zip(est["cancer_code"], est["regimen"]))
    assert anchor_cells <= est_cells, anchor_cells - est_cells
    # ORR is the dominant metric and present for most cells
    assert (est["metric"].str.upper() == "ORR").sum() >= 100


def test_wilson_ci_basics():
    # 0/30 -> lower bound pinned at 0, finite upper bound
    lo, hi = ici._wilson_ci(0, 30)
    assert lo == 0.0 and 0 < hi < 30
    # 50/100 -> interval straddles 50
    lo, hi = ici._wilson_ci(50, 100)
    assert lo < 50 < hi
    # empty -> (None, None)
    assert ici._wilson_ci(0, 0) == (None, None)


def test_pooled_proportion_responder_weighted():
    # NEC_MERKEL anti-PD-L1 has >=2 verified trials reporting responders + n.
    r = ici.pooled_ici_response("NEC_MERKEL", regimen="PD-L1", metric="ORR")
    assert r["poolable"] is True
    assert r["n_studies"] >= 2 and r["n_total"] > 0
    assert r["responders_total"] is not None
    # pooled point estimate must sit inside its own Wilson CI
    assert r["ci_low"] <= r["pooled_pct"] <= r["ci_high"]
    # responder-weighted: pooled == 100 * responders_total / n_total
    assert abs(r["pooled_pct"] - 100 * r["responders_total"] / r["n_total"]) < 0.1
    assert all(ref.startswith(("PMID:", "DOI:")) for ref in r["refs"])


def test_pooled_time_to_event_not_poolable():
    # OS is a median in months -> not responder-poolable.
    r = ici.pooled_ici_response("LIHC", regimen="PD-1+CTLA-4", metric="OS")
    assert r["poolable"] is False
    assert r["pooled_pct"] is None


def test_pooled_verified_only_and_alternates_switches():
    # include_alternates=False never yields more studies than the full pool
    full = ici.pooled_ici_response("BLCA", regimen="PD-1", metric="ORR")
    prim = ici.pooled_ici_response("BLCA", regimen="PD-1", metric="ORR", include_alternates=False)
    assert prim["n_studies"] <= full["n_studies"]
    # verified_only is the default; turning it off cannot drop sources
    loose = ici.pooled_ici_response("READ", regimen="PD-1", metric="ORR", verified_only=False)
    strict = ici.pooled_ici_response("READ", regimen="PD-1", metric="ORR")
    assert len(loose["sources"]) >= len(strict["sources"])


def test_pooled_result_contract():
    r = ici.pooled_ici_response("SKCM", regimen="PD-1", metric="ORR")
    for key in (
        "cancer_code",
        "regimen",
        "metric",
        "poolable",
        "pooled_pct",
        "ci_low",
        "ci_high",
        "n_total",
        "n_studies",
        "refs",
        "value_range",
        "sources",
    ):
        assert key in r
    assert r["cancer_code"] == "SKCM"
