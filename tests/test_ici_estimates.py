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
    # PAAD now uses the direct KEYNOTE-028 pancreatic cohort rather than a
    # derived no-citation near-zero anchor.
    paad = row("PAAD", "PD-1")
    assert paad["trial_name"] == "KEYNOTE-028" and paad["trial_nct"] == "NCT02054806"


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


def _truthy(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def test_estimates_internal_consistency():
    """Machine-checkable invariants on the estimates table — catches transcription / drift
    errors without re-verifying every paper. (Per-paper correctness rests on the audit.)"""
    df = ici.cancer_ici_response_estimates_df()
    HARD_PROPORTION = {"ORR", "CRR", "DCR", "PR"}  # value MUST equal 100·responders/n
    for _, r in df.iterrows():
        m = str(r["metric"]).upper()
        tag = f"{r['cancer_code']}/{r['regimen']}/{m}/{r['role']}"
        assert r["role"] in ("primary", "alternate"), f"{tag}: bad role"
        assert r["value_basis"] in (
            "reported",
            "derived_blend",
            "reported_context",
        ), f"{tag}: bad value_basis"
        ref = r["ref"]
        if isinstance(ref, str) and ref.strip():
            assert ref.startswith(("PMID:", "DOI:", "NCT")), f"{tag}: bad ref {ref!r}"
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


def test_unverified_rows_do_not_claim_source_verification():
    df = ici.cancer_ici_response_estimates_df()
    reported = df[
        (df["value_basis"].astype(str) == "reported") & (~df["source_verified"].map(_truthy))
    ]
    notes = reported["note"].fillna("").astype(str)
    claims_verified = notes.str.contains(
        r"verified|confirmed|matches full text|confirmed exactly",
        case=False,
        regex=True,
    )
    explicit_uncertainty = notes.str.contains(
        r"unverified|not independently confirmed|not confirmed|could not confirm|"
        r"pending primary confirmation|not supported|does not report|not in abstract|"
        r"not captured|not source-verified",
        case=False,
        regex=True,
    )
    bad = reported[claims_verified & ~explicit_uncertainty]
    assert bad.empty, bad[
        ["cancer_code", "regimen", "trial_name", "ref", "metric", "note"]
    ].to_dict("records")


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
        ("PAAD", "PD-1"): 0.0,  # KEYNOTE-028 pancreatic cohort: 0/24
        ("SCLC", "PD-1"): 10.0,  # CheckMate 032 nivolumab monotherapy: 10/98
        ("EPN", "PD-1"): 4.5,  # CheckMate 908 pooled EPN arms: 1/22
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


def test_paad_keynote028_source_endpoints():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[
        (est["cancer_code"] == "PAAD") & (est["regimen"] == "PD-1") & (est["role"] == "primary")
    ]
    by_metric = {str(r["metric"]): r for _, r in rows.iterrows()}
    assert {"ORR", "PFS", "OS"} <= set(by_metric)

    orr = by_metric["ORR"]
    assert orr["ref"] == "PMID:30557521"
    assert float(orr["value"]) == 0.0
    assert float(orr["ci_low"]) == 0.0 and float(orr["ci_high"]) == 14.2
    assert float(orr["metric_n"]) == 24 and float(orr["responders"]) == 0

    pfs = by_metric["PFS"]
    assert pfs["ref"] == "NCT02054806"
    assert float(pfs["value"]) == 1.7
    assert float(pfs["ci_low"]) == 1.5 and float(pfs["ci_high"]) == 1.8

    os = by_metric["OS"]
    assert os["ref"] == "NCT02054806"
    assert float(os["value"]) == 3.9
    assert float(os["ci_low"]) == 2.8 and float(os["ci_high"]) == 5.5


def test_sclc_checkmate032_source_endpoints():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[(est["cancer_code"] == "SCLC") & (est["ref"] == "PMID:27269741")]

    def row(regimen, role, drug, metric, metric_n=None):
        m = rows[
            (rows["regimen"] == regimen)
            & (rows["role"] == role)
            & (rows["drug"] == drug)
            & (rows["metric"] == metric)
        ]
        if metric_n is not None:
            m = m[m["metric_n"] == metric_n]
        assert len(m) == 1
        return m.iloc[0]

    mono = row("PD-1", "primary", "nivolumab", "ORR")
    assert mono["trial_alias"] == "CA209-032"
    assert mono["trial_nct"] == "NCT01928394"
    assert float(mono["value"]) == 10.0
    assert float(mono["ci_low"]) == 5.0 and float(mono["ci_high"]) == 18.0
    assert float(mono["metric_n"]) == 98 and float(mono["responders"]) == 10

    combo_hi_ipi = row("PD-1", "alternate", "nivolumab + ipilimumab", "ORR", 61)
    assert float(combo_hi_ipi["value"]) == 23.0
    assert float(combo_hi_ipi["ci_low"]) == 13.0 and float(combo_hi_ipi["ci_high"]) == 36.0
    assert float(combo_hi_ipi["metric_n"]) == 61 and float(combo_hi_ipi["responders"]) == 14
    assert bool(combo_hi_ipi["source_verified"]) is True

    combo_hi_nivo = row("PD-1", "alternate", "nivolumab + ipilimumab", "ORR", 54)
    assert float(combo_hi_nivo["value"]) == 19.0
    assert float(combo_hi_nivo["ci_low"]) == 9.0 and float(combo_hi_nivo["ci_high"]) == 31.0
    assert float(combo_hi_nivo["responders"]) == 10
    assert bool(combo_hi_nivo["source_verified"]) is True


def test_epn_checkmate908_pooled_orr_counts():
    est = ici.cancer_ici_response_estimates_df()
    rows = est[
        (est["cancer_code"] == "EPN")
        & (est["regimen"] == "PD-1")
        & (est["ref"] == "PMID:36808285")
        & (est["metric"] == "ORR")
    ]

    primary = rows[rows["role"] == "primary"]
    assert len(primary) == 1
    primary = primary.iloc[0]
    assert primary["drug"] == "nivolumab +/- ipilimumab"
    assert float(primary["source_n"]) == 22
    assert float(primary["metric_n"]) == 22
    assert float(primary["responders"]) == 1
    assert float(primary["value"]) == 4.5
    assert bool(primary["source_verified"]) is True
    assert primary["value_basis"] == "reported"
    assert _num(primary["ci_low"]) is None and _num(primary["ci_high"]) is None

    combo = rows[rows["role"] == "alternate"]
    assert len(combo) == 1
    combo = combo.iloc[0]
    assert float(combo["metric_n"]) == 10
    assert float(combo["responders"]) == 0
    assert combo["value_basis"] == "reported_context"

    pooled = ici.pooled_ici_response("EPN", regimen="PD-1", metric="ORR", verified_only=False)
    assert pooled["responders_total"] == 1
    assert pooled["n_total"] == 22
    assert pooled["n_pooled"] == 1
    assert pooled["n_studies"] == 1
    assert pooled["pooled_pct"] == 4.5
    assert pooled["refs"] == ["PMID:36808285"]


def test_sarc028_expansion_source_endpoints_and_pools():
    est = ici.cancer_ici_response_estimates_df()
    doi = "DOI:10.1200/JCO.2019.37.15_suppl.11015"

    def row(code, metric, role="primary"):
        m = est[
            (est["cancer_code"] == code)
            & (est["regimen"] == "PD-1")
            & (est["role"] == role)
            & (est["metric"] == metric)
            & (est["ref"] == doi)
        ]
        assert len(m) == 1
        return m.iloc[0]

    ddlps_orr = row("SARC_DDLPS", "ORR")
    assert float(ddlps_orr["source_n"]) == 40
    assert float(ddlps_orr["value"]) == 10.0
    assert float(ddlps_orr["metric_n"]) == 39 and float(ddlps_orr["responders"]) == 4
    assert bool(ddlps_orr["source_verified"]) is True

    ddlps_pfs = row("SARC_DDLPS", "PFS")
    assert float(ddlps_pfs["value"]) == 2.0
    assert float(ddlps_pfs["ci_low"]) == 2.0 and float(ddlps_pfs["ci_high"]) == 4.0
    ddlps_pfs_rate = row("SARC_DDLPS", "PFS_RATE")
    assert float(ddlps_pfs_rate["value"]) == 44.0
    assert float(ddlps_pfs_rate["ci_low"]) == 28.0 and float(ddlps_pfs_rate["ci_high"]) == 60.0
    ddlps_os = row("SARC_DDLPS", "OS")
    assert float(ddlps_os["value"]) == 13.0 and float(ddlps_os["ci_low"]) == 8.0

    ups_orr = row("SARC_UPS", "ORR")
    assert float(ups_orr["source_n"]) == 40
    assert float(ups_orr["value"]) == 23.0
    assert float(ups_orr["metric_n"]) == 40 and float(ups_orr["responders"]) == 9
    ups_crr = row("SARC_UPS", "CRR")
    assert float(ups_crr["value"]) == 5.0
    assert float(ups_crr["metric_n"]) == 40 and float(ups_crr["responders"]) == 2
    ups_pfs = row("SARC_UPS", "PFS")
    assert float(ups_pfs["value"]) == 3.0
    assert float(ups_pfs["ci_low"]) == 2.0 and float(ups_pfs["ci_high"]) == 5.0
    ups_pfs_rate = row("SARC_UPS", "PFS_RATE")
    assert float(ups_pfs_rate["value"]) == 50.0
    assert float(ups_pfs_rate["ci_low"]) == 35.0 and float(ups_pfs_rate["ci_high"]) == 65.0
    ups_os = row("SARC_UPS", "OS")
    assert float(ups_os["value"]) == 12.0
    assert float(ups_os["ci_low"]) == 7.0 and float(ups_os["ci_high"]) == 34.0

    for code, responders, n, pooled in (
        ("SARC_DDLPS", 4, 39, 10.3),
        ("SARC_UPS", 9, 40, 22.5),
    ):
        pooled_orr = ici.pooled_ici_response(
            code,
            regimen="PD-1",
            metric="ORR",
            verified_only=False,
        )
        assert pooled_orr["responders_total"] == responders
        assert pooled_orr["n_total"] == n
        assert pooled_orr["n_pooled"] == 1
        assert pooled_orr["pooled_pct"] == pooled
        assert pooled_orr["refs"] == [doi]
        assert all("context" not in str(s["setting"]).lower() for s in pooled_orr["sources"])
        assert all("comparator" not in str(s["setting"]).lower() for s in pooled_orr["sources"])
        assert all("initial" not in str(s["setting"]).lower() for s in pooled_orr["sources"])

    context = est[
        (est["cancer_code"].isin(["SARC_DDLPS", "SARC_UPS"]))
        & (est["regimen"] == "PD-1")
        & (est["role"] == "alternate")
        & (est["metric"] == "ORR")
    ]
    assert set(context["value_basis"]) == {"reported_context"}


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
