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
        "trial_name",
        "trial_alias",
        "trial_nct",
        "ref",
        "metric",
        "value",
        "unit",
        "ci_low",
        "ci_high",
        "metric_n",
        "responders",
        "source_verified",
        "value_basis",
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


def test_derived_blends_marked_and_never_pooled():
    est = ici.cancer_ici_response_estimates_df()
    derived = est[est["value_basis"] == "derived_blend"]
    # exactly the three all-comer MMR-dependent cells are derived blends
    assert set(zip(derived["cancer_code"], derived["regimen"])) == {
        ("COAD", "PD-1"),
        ("READ", "PD-1"),
        ("UCEC", "PD-1"),
    }
    # the derived all-comer ORR is dropped from pooling even with verified_only=False
    r = ici.pooled_ici_response("COAD", regimen="PD-1", metric="ORR", verified_only=False)
    assert r["sources"] == [] and r["n_studies"] == 0 and r["pooled_pct"] is None


def test_msi_subtype_value_corrected_and_rolls_up():
    anchor = ici.cancer_ici_response_df()

    def orr(code):
        row = anchor[(anchor["cancer_code"] == code) & (anchor["regimen"] == "PD-1")]
        return float(row["orr_pct"].iloc[0])

    # COAD_MSI / READ_MSI corrected to the published KEYNOTE-177 value (43.8, not 45)
    assert abs(orr("COAD_MSI") - 43.8) < 0.01
    assert abs(orr("READ_MSI") - 43.8) < 0.01
    # MSS components present and ~0; all-comer is the (low) prevalence-weighted blend
    assert orr("COAD_MSS") == 0.0
    assert orr("COAD") < orr("COAD_MSI")  # blend is far below the MSI subtype
    # COAD all-comer ~ 43.8 * dMMR-prevalence(~0.13)
    assert 4.0 <= orr("COAD") <= 7.0
    # UCEC corrected to KEYNOTE-158 components (dMMR 48, pMMR 7); all-comer is the
    # roll-up at advanced-EC dMMR prevalence ~20% (0.20*48 + 0.80*7 = 15.2), not 8.
    assert abs(orr("UCEC_MSI") - 48.0) < 0.01
    assert orr("UCEC_CNH") == 7.0 and orr("UCEC_CNL") == 7.0
    rolled = 0.20 * orr("UCEC_MSI") + 0.80 * orr("UCEC_CNH")
    assert abs(orr("UCEC") - rolled) <= 1.0


def test_trial_columns_split_and_clean():
    import re

    df = ici.cancer_ici_response_df()
    assert {"trial_name", "trial_alias", "trial_nct"} <= set(df.columns)
    assert "trial" not in df.columns
    for _, r in df.iterrows():
        name = str(r["trial_name"]).strip()
        assert name and name.lower() != "nan", f"{r['cancer_code']} missing trial_name"
        assert "(" not in name, f"{r['cancer_code']} trial_name still has parentheses: {name}"
        nct = r["trial_nct"]
        if isinstance(nct, str) and nct.strip():
            assert re.fullmatch(r"NCT\d{8}", nct.strip()), f"{r['cancer_code']} bad NCT {nct}"
        alias = r["trial_alias"]
        if isinstance(alias, str) and alias.strip():
            assert alias.strip() != name, f"{r['cancer_code']} alias echoes trial_name"

    def row(code, regimen):
        m = df[(df["cancer_code"] == code) & (df["regimen"] == regimen)]
        return m.iloc[0]

    # the formerly acronym-less rows are now resolved
    ifct = row("SCLC", "PD-L1")  # IFCT-1603: name IS the protocol code -> no alias
    assert ifct["trial_name"] == "IFCT-1603" and ifct["trial_nct"] == "NCT03059667"
    assert not str(ifct["trial_alias"]).strip() or str(ifct["trial_alias"]) == "nan"
    # pooled/basket anchors correctly carry no NCT
    paad = row("PAAD", "PD-1")
    assert not (isinstance(paad["trial_nct"], str) and paad["trial_nct"].strip())


def test_pooled_sources_expose_trial_labels():
    r = ici.pooled_ici_response("NEC_MERKEL", regimen="PD-L1", metric="ORR")
    assert r["sources"], "expected NEC_MERKEL PD-L1 sources"
    s = r["sources"][0]
    assert {"trial_name", "trial_alias", "trial_nct"} <= set(s)


def _num(v):
    import math

    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def test_estimates_internal_consistency():
    """Machine-checkable invariants on the estimates table — catches transcription / drift
    errors without re-verifying every paper. (Per-paper correctness rests on the audit.)"""
    df = ici.cancer_ici_response_estimates_df()
    HARD_PROPORTION = {"ORR", "CRR", "DCR", "PR"}  # value MUST equal 100·responders/n
    for _, r in df.iterrows():
        m = str(r["metric"]).upper()
        tag = f"{r['cancer_code']}/{r['regimen']}/{m}/{r['role']}"
        assert r["role"] in ("primary", "alternate"), f"{tag}: bad role"
        assert r["value_basis"] in ("reported", "derived_blend"), f"{tag}: bad value_basis"
        ref = r["ref"]
        if isinstance(ref, str) and ref.strip():
            assert ref.startswith(("PMID:", "DOI:")), f"{tag}: bad ref {ref!r}"
        v, resp, n = _num(r["value"]), _num(r["responders"]), _num(r["metric_n"])
        lo, hi = _num(r["ci_low"]), _num(r["ci_high"])
        if resp is not None and n is not None:
            assert resp <= n, f"{tag}: responders {resp} > n {n}"
        if m in HARD_PROPORTION and v is not None:
            assert 0 <= v <= 100, f"{tag}: proportion value {v} out of range"
            if resp is not None and n:
                assert abs(v - 100 * resp / n) <= 2.0, f"{tag}: value {v} != {resp}/{n}"
        if lo is not None and hi is not None and v is not None:
            assert lo - 0.6 <= v <= hi + 0.6, f"{tag}: value {v} outside CI [{lo},{hi}]"


def test_anchor_orr_in_ballpark_of_estimates_primary():
    """The representative anchor (cancer-ici-response.csv) and the estimates table's primary
    ORR for a cell must be in the same ballpark — and there must be exactly ONE primary ORR
    per cell. The anchor is a deliberately *rounded representative* value (per the table
    docstring, "not an exact reproducible constant"), so small gaps vs the precise audited
    value are expected; this only catches GROSS drift (a wrong cell, a 2× transcription
    error) — i.e. the two tables falling out of sync."""
    anchor = ici.cancer_ici_response_df()
    est = ici.cancer_ici_response_estimates_df()
    prim = est[(est["role"] == "primary") & (est["metric"].str.upper() == "ORR")]
    by_cell = {}
    for _, r in prim.iterrows():
        cell = (r["cancer_code"], r["regimen"])
        assert cell not in by_cell, f"{cell}: more than one primary ORR row"
        by_cell[cell] = _num(r["value"])
    for _, a in anchor.iterrows():
        cell = (a["cancer_code"], a["regimen"])
        if cell in by_cell and by_cell[cell] is not None:
            assert abs(float(a["orr_pct"]) - by_cell[cell]) <= 5.0, (
                f"{cell}: anchor {a['orr_pct']} far from estimates primary {by_cell[cell]}"
            )


def test_audited_anchor_values_match_primary_orr():
    anchor = ici.cancer_ici_response_df()
    est = ici.cancer_ici_response_estimates_df()
    audited = {
        ("LIHC", "PD-1"): 20.0,  # CheckMate 040 dose-expansion ORR, PMID:28434648
        ("MDS", "PD-1"): 0.0,  # KEYNOTE-013: no CR/PR by IWG criteria
    }
    for cell, expected in audited.items():
        code, regimen = cell
        a = anchor[(anchor["cancer_code"] == code) & (anchor["regimen"] == regimen)]
        assert len(a) == 1
        assert abs(float(a["orr_pct"].iloc[0]) - expected) < 0.01

        p = est[
            (est["cancer_code"] == code)
            & (est["regimen"] == regimen)
            & (est["role"] == "primary")
            & (est["metric"].str.upper() == "ORR")
        ]
        assert len(p) == 1
        assert abs(float(p["value"].iloc[0]) - expected) < 0.01

    from oncoref import apd1

    apd1_anchor = apd1.cancer_apd1_response_df()
    for (code, regimen), expected in audited.items():
        row = apd1_anchor[
            (apd1_anchor["cancer_code"] == code) & (apd1_anchor["drug_target"] == regimen)
        ]
        assert len(row) == 1
        assert abs(float(row["apd1_orr_pct"].iloc[0]) - expected) < 0.01


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
        "n_pooled",
        "refs",
        "value_range",
        "sources",
    ):
        assert key in r
    assert r["cancer_code"] == "SKCM"
    # n_studies is the full evidence count; n_pooled (proportion only) is what entered the
    # responder-weighted pool, so it can only be <= n_studies.
    assert r["n_pooled"] is None or r["n_pooled"] <= r["n_studies"]
