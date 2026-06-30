# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import numpy as np
import pandas as pd
import pytest

from oncoref import expression

_BREAKPOINTS = [0, 1, 5, 10, 50, 90, 95, 99, 100]


def _write_percentile_shard(cache_dir, code, tpm_values):
    """Write a synthetic percentile parquet (stored as log1p, like the real one)."""
    shard_dir = cache_dir / "cancer-reference-expression-percentiles"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cols = {"Ensembl_Gene_ID": ["ENSG00000000003", "ENSG00000000005"], "Symbol": ["TSPAN6", "TNMD"]}
    for bp in _BREAKPOINTS:
        # store log1p of a per-breakpoint tpm so as_tpm restores tpm_values[bp]
        cols[f"p{bp}"] = np.log1p([tpm_values[bp], tpm_values[bp] * 2]).astype("float16")
    pd.DataFrame(cols).to_parquet(shard_dir / f"{code}.parquet", index=False)


@pytest.fixture
def percentile_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    tpm = {bp: float(bp) for bp in _BREAKPOINTS}  # p95 -> 95 tpm, etc.
    _write_percentile_shard(tmp_path, "PRAD", tpm)
    return tmp_path


def test_available_percentile_cohorts(percentile_cache):
    sidecar = percentile_cache / "cancer-reference-expression-percentiles" / "._ACC.parquet"
    sidecar.write_bytes(b"appledouble sidecar")
    assert expression.available_percentile_cohorts() == ["PRAD"]


def test_expression_artifact_gene_universe_deltas_expose_known_remaps():
    df = expression.expression_artifact_gene_universe_deltas()

    paxx = df[
        (df["legacy_ensembl_gene_id"] == "ENSG00000148362")
        & (df["oncoref_ensembl_gene_id"] == "ENSG00000310560")
    ]
    assert not paxx.empty
    assert set(paxx["symbol"]) == {"PAXX"}
    assert df.attrs["comparison"] == "pirlygenes_5.23.2_vs_oncoref_5.23.0"


def test_expression_artifact_gene_universe_deltas_filter_by_product_and_code():
    cll = expression.expression_artifact_gene_universe_deltas(
        product="cohort_gene_percentiles",
        cancer_type="CLL",
        delta_kind="pirlygenes_only",
    )

    assert len(cll) == 11
    assert set(cll["status"]) == {"canonical_replacement_absent_from_output"}
    assert "ENSG00000225489" in set(cll["legacy_ensembl_gene_id"])


def test_expression_artifact_gene_universe_delta_summary():
    summary = expression.expression_artifact_gene_universe_delta_summary()

    hit = summary[
        (summary["product"] == "cohort_gene_percentiles")
        & (summary["cancer_code"] == "CLL")
        & (summary["delta_kind"] == "pirlygenes_only")
        & (summary["status"] == "canonical_replacement_absent_from_output")
    ]
    assert hit["n"].iloc[0] == 11


def test_cohort_gene_percentiles_as_tpm(percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=True)
    assert list(df["Symbol"]) == ["TSPAN6", "TNMD"]
    # gene 0, p95 breakpoint should restore to ~95 tpm (stored as log1p(95)).
    assert df.loc[0, "p95"] == pytest.approx(95.0, rel=1e-2)


def test_cohort_gene_percentiles_log_space(percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=False)
    # stored log1p value, not expm1-restored
    assert df.loc[0, "p95"] == pytest.approx(np.log1p(95.0), rel=1e-2)


def test_cohort_gene_percentiles_resolves_alias(percentile_cache):
    # "prostate" resolves to PRAD via the registry.
    df = expression.cohort_gene_percentiles("prostate")
    assert len(df) == 2


def _no_cached_matrix(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(expression, "per_sample_expression", _raise)


def test_cohort_gene_percentiles_missing_raises(percentile_cache, monkeypatch):
    # No shard AND no cached matrix (on-the-fly can't run) -> clear error.
    _no_cached_matrix(monkeypatch)
    with pytest.raises(ValueError, match="no percentile vector"):
        expression.cohort_gene_percentiles("BRCA")


def test_cohort_gene_percentiles_missing_empty_schema(percentile_cache, monkeypatch):
    _no_cached_matrix(monkeypatch)
    df = expression.cohort_gene_percentiles("BRCA", on_missing="empty", include_provenance=True)

    assert df.empty
    assert list(df.columns[:2]) == ["Ensembl_Gene_ID", "Symbol"]
    assert "p0" in df.columns and "p100" in df.columns
    assert "cancer_code" in df.columns
    assert df.attrs["schema_version"] == expression.PERCENTILE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["cancer_code"] == "BRCA"
    assert "missing_reason" in df.attrs


def test_cohort_gene_percentiles_provenance_columns_and_attrs(percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", include_provenance=True)

    assert set(df.columns) >= {
        "cancer_code",
        "normalization",
        "expression_unit",
        "percentile_basis",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
    }
    assert set(df["cancer_code"]) == {"PRAD"}
    assert set(df["normalization"]) == {"tpm_clean"}
    assert df.attrs["schema_version"] == expression.PERCENTILE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["cancer_code"] == "PRAD"


@pytest.fixture
def proteoform_percentile_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    shard_dir = tmp_path / "cancer-reference-expression-percentiles-proteoform-cta"
    shard_dir.mkdir(parents=True)
    # A collapsed shard carries the proteoform identity columns alongside the breakpoints.
    cols = {
        "proteoform_key": ["NY-ESO-1", "ENSG00000185686"],
        "Ensembl_Gene_ID": ["ENSG00000184033", "ENSG00000185686"],
        "Symbol": ["NY-ESO-1", "PRAME"],
        "proteoform_members": ["CTAG1A/CTAG1B", "PRAME"],
    }
    for bp in _BREAKPOINTS:
        cols[f"p{bp}"] = np.log1p([float(bp), float(bp) * 2]).astype("float16")
    pd.DataFrame(cols).to_parquet(shard_dir / "PRAD.parquet", index=False)
    return tmp_path


def test_available_percentile_cohorts_proteoform(proteoform_percentile_cache):
    # The proteoform variant reads the local shard (auto_fetch=False, no bundle fetch).
    assert expression.available_percentile_cohorts(proteoform=True) == ["PRAD"]


def test_cohort_gene_percentiles_proteoform(proteoform_percentile_cache):
    df = expression.cohort_gene_percentiles("PRAD", as_tpm=True, proteoform=True)
    # the proteoform key space: NY-ESO-1 (collapsed) + PRAME (singleton -> ENSG key)
    assert list(df["proteoform_key"]) == ["NY-ESO-1", "ENSG00000185686"]
    # breakpoint columns are restored to TPM; the id columns are excluded from them
    assert df.loc[0, "p95"] == pytest.approx(95.0, rel=1e-2)
    assert set(df.columns) >= {"proteoform_key", "Ensembl_Gene_ID", "Symbol", "proteoform_members"}


def test_cohort_gene_percentiles_proteoform_missing_raises(
    proteoform_percentile_cache, monkeypatch
):
    _no_cached_matrix(monkeypatch)
    with pytest.raises(ValueError, match="no proteoform-summed percentile vector"):
        expression.cohort_gene_percentiles("BRCA", proteoform=True)


def test_cohort_gene_percentiles_proteoform_computed_on_the_fly(
    proteoform_percentile_cache, monkeypatch
):
    # LUAD has no proteoform shard -> the percentile vector is recomputed on the fly
    # from the (stubbed) per-sample matrix: members collapse before the percentiles.
    import oncoref.proteoforms as pmod

    fake = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_A2", "ENSG_B"],
            "Symbol": ["A1", "A2", "B"],
            "s1": [3.0, 5.0, 1.0],
            "s2": [9.0, 1.0, 4.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: fake.copy())
    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A1/A2": ["ENSG_A1", "ENSG_A2"]}
    )
    out = expression.cohort_gene_percentiles("LUAD", proteoform=True)
    assert "proteoform_key" in out.columns
    assert "A1/2" in set(out["Symbol"]) and "A1" not in set(out["Symbol"])
    # A1/A2 summed per sample: s1=8, s2=10 -> p100 (max) restores to 10.0 TPM
    assert out.set_index("Symbol").loc["A1/2", "p100"] == pytest.approx(10.0, rel=1e-2)


def test_cohort_gene_percentiles_threads_scope(proteoform_percentile_cache, monkeypatch):
    # scope= reaches the on-the-fly collapse: the cta universe leaves A1/A2 separate,
    # the genome universe groups them. Previously scope was silently fixed.
    import oncoref.proteoforms as pmod

    fake = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_A2", "ENSG_B"],
            "Symbol": ["A1", "A2", "B"],
            "s1": [3.0, 5.0, 1.0],
            "s2": [9.0, 1.0, 4.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: fake.copy())
    maps = {"cta": {}, "genome": {"A1/A2": ["ENSG_A1", "ENSG_A2"]}}
    monkeypatch.setattr(pmod, "proteoform_group_map", lambda *, scope="cta": maps[scope])

    cta = expression.cohort_gene_percentiles("LUAD", proteoform=True, scope="cta", auto_fetch=False)
    assert {"A1", "A2"} <= set(cta["Symbol"])  # cta universe: ungrouped here
    genome = expression.cohort_gene_percentiles(
        "LUAD", proteoform=True, scope="genome", auto_fetch=False
    )
    assert "A1/2" in set(genome["Symbol"]) and "A1" not in set(genome["Symbol"])  # collapsed


def test_proteoform_shard_selection_is_scope_specific(proteoform_percentile_cache, monkeypatch):
    # The fixture ships a *cta-scope* proteoform shard. A cta request reads it; a genome
    # request must NOT (it resolves the genome dir, which is absent) — proving scope
    # selects the shard directory, not just the on-the-fly collapse universe.
    cta = expression.cohort_gene_percentiles("PRAD", proteoform=True, scope="cta", auto_fetch=False)
    assert list(cta["proteoform_key"]) == ["NY-ESO-1", "ENSG00000185686"]

    def _no_matrix(*a, **k):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(expression, "per_sample_expression", _no_matrix)
    with pytest.raises(ValueError, match="per-sample matrix isn't cached"):
        expression.cohort_gene_percentiles(
            "PRAD", proteoform=True, scope="genome", auto_fetch=False
        )


def test_shard_dataset_registry_is_public_and_scope_aware():
    # ShardDataset + SHARD_DATASETS are public; the proteoform shard dir is scope-specific.
    import oncoref

    assert set(oncoref.SHARD_DATASETS) == {"representatives", "percentiles", "within_sample"}
    pct = oncoref.SHARD_DATASETS["percentiles"]
    assert isinstance(pct, oncoref.ShardDataset)
    assert pct.subdir(proteoform=False) == "cancer-reference-expression-percentiles"
    assert pct.subdir(proteoform=True, scope="cta").endswith("-proteoform-cta")
    assert pct.subdir(proteoform=True, scope="genome").endswith("-proteoform-genome")
    assert pct.fetches(proteoform=False) is True
    assert pct.fetches(proteoform=True, scope="cta") is True
    assert pct.fetches(proteoform=True, scope="genome") is False
    assert oncoref.SHARD_DATASETS["within_sample"].fetches(proteoform=False) is True
    assert oncoref.SHARD_DATASETS["within_sample"].fetches(proteoform=True, scope="cta") is True
    assert oncoref.SHARD_DATASETS["within_sample"].fetches(proteoform=True, scope="genome") is False
    # an artifact with no proteoform variant rejects the request
    with pytest.raises(ValueError, match="no proteoform variant"):
        oncoref.SHARD_DATASETS["representatives"].subdir(proteoform=True)
    # the column-vocabulary helpers are now package-public too
    assert callable(oncoref.id_columns) and callable(oncoref.sample_columns)


def test_proteoform_summary_wrappers_thread_scope_and_default_autofetch_true(monkeypatch):
    # Proteoform wrappers still default auto_fetch=True and thread scope; if the
    # active bundle lacks a requested scope-specific shard, they can recompute
    # from the source matrix through the same path.
    seen = {}
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda ct, **k: seen.update(k) or pd.DataFrame()
    )
    expression.proteoform_cohort_percentiles("X", scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome" and seen["auto_fetch"] is True

    seen.clear()
    monkeypatch.setattr(
        expression, "within_sample_top_fraction", lambda ct, **k: seen.update(k) or pd.DataFrame()
    )
    expression.proteoform_within_sample_top_fraction("X", scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome" and seen["auto_fetch"] is True
    # The gene variant still defaults auto_fetch=False (reads a shipped shard).
    seen.clear()
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda ct, **k: seen.update(k) or pd.DataFrame()
    )
    expression.gene_cohort_percentiles("X")
    assert seen["auto_fetch"] is False


def test_representatives_provenance_requires_long_format():
    # Provenance is per-representative; asking for it in the default wide form is
    # a no-op, so it must fail loudly rather than silently dropping the request.
    with pytest.raises(ValueError, match="include_provenance=True requires format='long'"):
        expression.representative_cohort_samples("PRAD", include_provenance=True)


def test_representative_provenance_includes_source_sample_and_selection_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1"],
            "Symbol": ["GENE1"],
            "PRAD__rep1": [1.0],
        }
    ).to_parquet(d / "PRAD.parquet", index=False)
    pd.DataFrame(
        {
            "representative_id": ["PRAD__rep1"],
            "source_cohort": ["PRAD"],
            "source_project": ["TCGA"],
            "source_sample": ["TCGA-XX-0001"],
            "n_cohort_samples": [10],
        }
    ).to_csv(d / "_provenance.csv", index=False)

    df = expression.representative_cohort_samples("PRAD", format="long", include_provenance=True)

    assert df.loc[0, "representative_id"] == "PRAD_rep01"
    assert df.loc[0, "source_sample"] == "TCGA-XX-0001"
    assert df.loc[0, "selection_rank"] == 1
    assert df.loc[0, "selection_method"] == expression.REPRESENTATIVE_SELECTION_METHOD
    assert df.loc[0, "selection_basis"] == expression.REPRESENTATIVE_SELECTION_BASIS
    assert df.loc[0, "artifact_schema_version"] == expression.REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["schema_version"] == expression.REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["cancer_codes"] == ["PRAD"]
    assert df.attrs["representative_id_style"] == "pirlygenes"

    internal = expression.representative_cohort_samples(
        "PRAD",
        format="long",
        include_provenance=True,
        representative_id_style="internal",
    )
    assert internal.loc[0, "representative_id"] == "PRAD__rep1"


def test_representative_empty_long_schema_includes_requested_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    (tmp_path / "cancer-reference-expression-representatives").mkdir(parents=True)

    df = expression.representative_cohort_samples("PRAD", format="long", include_provenance=True)

    assert df.empty
    assert list(df.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "cancer_code",
        "representative_id",
        "expression",
        "source_cohort",
        "source_version",
        "source_project",
        "source_sample",
        "n_cohort_samples",
        "selection_rank",
        "selection_method",
        "selection_basis",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
    ]
    assert df.attrs["schema_version"] == expression.REPRESENTATIVE_ARTIFACT_SCHEMA_VERSION
    assert df.attrs["representative_id_style"] == "pirlygenes"


def test_representative_id_style_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    (tmp_path / "cancer-reference-expression-representatives").mkdir(parents=True)

    with pytest.raises(ValueError, match="representative_id_style"):
        expression.representative_cohort_samples("PRAD", representative_id_style="legacy")


def test_representative_wide_does_not_fragment_genes(monkeypatch, tmp_path):
    # The #465-class bug: cohorts quantified on different Ensembl releases carry the SAME
    # locus under different alias symbols (and one as the raw id). The wide combiner must
    # key on the canonical gene id, so the gene stays ONE row covered by all cohorts —
    # never three mutually-disjoint sparse alias rows.
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG234", "ENSG999"],
            "Symbol": ["AP000959.1", "PRAME"],
            "A__rep1": [1.0, 2.0],
        }
    ).to_parquet(d / "A.parquet", index=False)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG234", "ENSG888"],
            "Symbol": ["RP11-844P9.5", "MAGEA4"],
            "B__rep1": [3.0, 4.0],
        }
    ).to_parquet(d / "B.parquet", index=False)
    pd.DataFrame(
        {"Ensembl_Gene_ID": ["ENSG234"], "Symbol": ["ENSG234"], "C__rep1": [5.0]}  # raw-id backfill
    ).to_parquet(d / "C.parquet", index=False)

    w = expression.representative_cohort_samples(format="wide")
    assert len(w) == w["Ensembl_Gene_ID"].nunique()  # one row per gene id (no fragmentation)
    shared = w[w["Ensembl_Gene_ID"] == "ENSG234"]
    assert len(shared) == 1
    assert shared["Symbol"].iloc[0] != "ENSG234"  # canonical symbol prefers a real alias
    # all three cohorts' values land on that single row
    assert bool(shared[["A_rep01", "B_rep01", "C_rep01"]].notna().all(axis=1).iloc[0])

    internal = expression.representative_cohort_samples(
        format="wide", representative_id_style="internal"
    )
    assert {"A__rep1", "B__rep1", "C__rep1"} <= set(internal.columns)

    # the long form carries the same canonical symbol (so a downstream pivot won't fragment)
    lg = expression.representative_cohort_samples(format="long")
    assert lg.loc[lg["Ensembl_Gene_ID"] == "ENSG234", "Symbol"].nunique() == 1


def test_representative_wide_merges_alt_haplotype_aliases(monkeypatch, tmp_path):
    # An alt-haplotype/archived id in one cohort must collapse onto its primary-contig id
    # via the shipped ensembl-id-aliases migration map (not just unversioning) when another
    # cohort carries the primary — so the two cohorts land on ONE canonical row.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame({"Ensembl_Gene_ID": [alt], "Symbol": ["X"], "A__rep1": [1.0]}).to_parquet(
        d / "A.parquet", index=False
    )
    pd.DataFrame({"Ensembl_Gene_ID": [primary], "Symbol": ["X"], "B__rep1": [2.0]}).to_parquet(
        d / "B.parquet", index=False
    )
    w = expression.representative_cohort_samples(format="wide")
    assert list(w["Ensembl_Gene_ID"]) == [primary]  # one row, the canonical primary
    assert alt not in set(w["Ensembl_Gene_ID"])
    assert bool(w[["A_rep01", "B_rep01"]].notna().all(axis=1).iloc[0])  # both cohorts on it


def test_representative_wide_sums_alt_haplotype_within_cohort(monkeypatch, tmp_path):
    # When one cohort carries BOTH an alt-haplotype id and its primary (a full-assembly
    # quantification annotates the gene on both contigs), their per-sample TPMs must be
    # SUMMED under the canonical id (reads multi-map between copies), not deduped to one.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    # one cohort, one representative, the gene present under both ids
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": [primary, alt],
            "Symbol": ["G", "G-alt"],
            "A__rep1": [10.0, 3.0],
        }
    ).to_parquet(d / "A.parquet", index=False)

    w = expression.representative_cohort_samples(format="wide")
    assert list(w["Ensembl_Gene_ID"]) == [primary]  # collapsed onto the canonical id
    assert w["A_rep01"].iloc[0] == pytest.approx(13.0)  # 10 + 3 summed, not 10 or 3


def test_representative_log1p_sums_in_linear_space(monkeypatch, tmp_path):
    # The alt-haplotype sum must be taken in LINEAR TPM, then log1p applied — never
    # sum log-space values: log1p(10)+log1p(3) != log1p(10+3).
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    d = tmp_path / "cancer-reference-expression-representatives"
    d.mkdir(parents=True)
    pd.DataFrame(
        {"Ensembl_Gene_ID": [primary, alt], "Symbol": ["G", "G-alt"], "A__rep1": [10.0, 3.0]}
    ).to_parquet(d / "A.parquet", index=False)

    w = expression.representative_cohort_samples(format="wide", normalize="tpm_clean_log1p")
    assert list(w["Ensembl_Gene_ID"]) == [primary]
    assert w["A_rep01"].iloc[0] == pytest.approx(np.log1p(13.0))  # log1p(10+3)
    assert w["A_rep01"].iloc[0] != pytest.approx(np.log1p(10.0) + np.log1p(3.0))  # NOT Σlog1p


def _raw_matrix(tmp_path):
    # A tiny raw-TPM per-sample matrix (genes x samples) whose columns sum near 1e6.
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
            "Symbol": ["A", "B", "C"],
            "s1": [500000.0, 300000.0, 200000.0],
            "s2": [100000.0, 600000.0, 300000.0],
        }
    )
    path = tmp_path / "PRAD.parquet"
    df.to_parquet(path, index=False)
    return path


def test_per_sample_matrix_cache_size_is_tunable():
    # The LRU is wired to the (env-configurable) constant, not a hardcoded literal,
    # so heavy pooling workflows can keep >2 matrices warm via CANCERDATA_PER_SAMPLE_CACHE.
    assert (
        expression._load_per_sample_matrix.cache_info().maxsize == expression._PER_SAMPLE_CACHE_SIZE
    )
    assert expression._PER_SAMPLE_CACHE_SIZE >= 1


def test_per_sample_cache_size_tolerates_malformed_env(monkeypatch):
    # A tuning knob must never break `import oncoref`: a malformed/empty value
    # falls back to the default rather than raising at parse time.
    for bad in ("", "abc", "2.5"):
        monkeypatch.setenv("CANCERDATA_PER_SAMPLE_CACHE", bad)
        assert expression._per_sample_cache_size() == 2
    monkeypatch.setenv("CANCERDATA_PER_SAMPLE_CACHE", "0")  # clamps to >=1
    assert expression._per_sample_cache_size() == 1
    monkeypatch.setenv("CANCERDATA_PER_SAMPLE_CACHE", "8")  # honored
    assert expression._per_sample_cache_size() == 8


def test_per_sample_expression_normalize_modes(tmp_path, monkeypatch):
    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)

    raw = expression.per_sample_expression("PRAD", normalize="tpm_raw")
    assert list(raw.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]
    assert raw["s1"].sum() == pytest.approx(1e6)

    clean = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    # No technical/censored genes in this fixture -> the technical compartment is
    # empty and the biological compartment fills its 750k budget (clean_tpm's
    # two-compartment contract). A real matrix with technical genes sums to ~1e6.
    assert clean["s1"].sum() == pytest.approx(750000.0, rel=1e-6)
    assert list(clean.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]

    logged = expression.per_sample_expression("PRAD", normalize="tpm_clean_log1p")
    assert np.allclose(logged["s1"].to_numpy(), np.log1p(clean["s1"].to_numpy()))

    # hk mode runs (no HK panel genes in this fixture -> a no-op rescale, but valid).
    hk = expression.per_sample_expression("PRAD", normalize="tpm_clean_hk")
    assert list(hk.columns) == ["Ensembl_Gene_ID", "Symbol", "s1", "s2"]


def test_per_sample_expression_canonicalizes_alias_genes(tmp_path, monkeypatch):
    # per_sample_expression returns the DENSE canonical space (oncoref#135 item 6): an
    # alt-haplotype copy is summed into its primary in LINEAR TPM, a versioned id is
    # unversioned, a retired id is relabeled to its successor — every transform applied
    # AFTER the linear sum.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [primary, alt, "ENSG00000005955", "ENSG00000141510.5"],
            "Symbol": ["G", "G-alt", "GGNBP2-old", "TP53"],
            "s1": [100000.0, 30000.0, 1000.0, 200000.0],
            "s2": [50000.0, 10000.0, 2000.0, 400000.0],
        }
    )
    path = tmp_path / "X.parquet"
    raw.to_parquet(path, index=False)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)

    rawout = expression.per_sample_expression("X", normalize="tpm_raw").set_index("Ensembl_Gene_ID")
    assert alt not in rawout.index  # alt-haplotype copy folded into its primary
    assert rawout.loc[primary, "s1"] == pytest.approx(130000.0)  # 100000 + 30000, linear
    assert "ENSG00000141510" in rawout.index  # versioned id -> unversioned canonical
    assert "ENSG00000278311" in rawout.index  # GGNBP2 retired id -> successor
    assert "ENSG00000005955" not in rawout.index

    # log1p is applied AFTER the linear alt-copy sum, never log1p(a)+log1p(b)
    clean = expression.per_sample_expression("X", normalize="tpm_clean").set_index(
        "Ensembl_Gene_ID"
    )
    logged = expression.per_sample_expression("X", normalize="tpm_clean_log1p").set_index(
        "Ensembl_Gene_ID"
    )
    assert np.allclose(
        logged.loc[primary, ["s1", "s2"]].to_numpy(dtype=float),
        np.log1p(clean.loc[primary, ["s1", "s2"]].to_numpy(dtype=float)),
    )


def test_pan_cancer_expression_canonicalizes_alias_genes(monkeypatch):
    # pan_cancer_expression is dense canonical too: cohort/tissue columns are linear
    # abundances (nTPM/FPKM), so summing an alt copy into its primary row is exact.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [primary, alt],
            "Symbol": ["G", "G-alt"],
            "nTPM_liver": [3.0, 7.0],
        }
    )
    monkeypatch.setattr(expression, "get_data", lambda name: fixture.copy())
    out = expression.pan_cancer_expression(normalize=None).set_index("Ensembl_Gene_ID")
    assert list(out.index) == [primary]  # alt folded into primary
    assert out.loc[primary, "liver_nTPM_raw"] == pytest.approx(10.0)  # 3 + 7 summed


def test_housekeeping_normalize_divides_by_panel_geomean(monkeypatch):
    import oncoref.gene_families as gf

    monkeypatch.setattr(
        gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: frozenset({"ENSG_HK"})
    )
    df = pd.DataFrame(
        {"Ensembl_Gene_ID": ["ENSG_HK", "ENSG_X"], "Symbol": ["HK", "X"], "s1": [100.0, 50.0]}
    )
    out = expression._housekeeping_normalize(df, ["s1"])
    # Each column divided by its housekeeping geomean -> the gene/HK *ratio* is preserved.
    assert out.loc[1, "s1"] / out.loc[0, "s1"] == pytest.approx(0.5)


def test_housekeeping_normalize_blanks_sparse_housekeeping_denominator(monkeypatch):
    import oncoref.gene_families as gf

    panel = frozenset({"HK1", "HK2", "HK3"})
    monkeypatch.setattr(gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: panel)
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["HK1", "HK2", "HK3", "ENSG_X"],
            "Symbol": ["HK1", "HK2", "HK3", "X"],
            "good": [100.0, 120.0, 80.0, 50.0],
            "sparse": [0.0, 0.0, 100.0, 50.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="housekeeping normalization skipped"):
        out = expression._housekeeping_normalize(df, ["good", "sparse"])

    assert out.loc[out["Symbol"] == "X", "good"].iloc[0] > 0
    assert out["sparse"].isna().all()


def test_sample_expression_qc_flags_sparse_samples(tmp_path, monkeypatch):
    import oncoref.gene_families as gf
    from oncoref import expression_registry

    panel = frozenset({"HK1", "HK2", "HK3"})
    monkeypatch.setattr(gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: panel)
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["HK1", "HK2", "HK3", "ENSG_A", "ENSG_B", "ENSG_C"],
            "Symbol": ["HK1", "HK2", "HK3", "A", "B", "C"],
            "good": [100.0, 120.0, 80.0, 100.0, 100.0, 100.0],
            "sparse": [0.0, 0.0, 20.0, 1000.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "X.parquet"
    raw.to_parquet(path, index=False)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        expression.source_matrices,
        "cohort_info",
        lambda code: {"source_cohort": "TEST_SOURCE", "n_samples": 2},
    )
    monkeypatch.setattr(
        expression_registry,
        "sources_for_cancer_code",
        lambda code: [
            expression_registry.ExpressionSource(
                id="test-source",
                category="expression",
                cancer_codes=("X",),
                source_type="gdc",
                unit="TPM",
                source_cohort="TEST_SOURCE",
            )
        ],
    )

    qc = expression.sample_expression_qc(
        "X",
        min_detected_genes=3,
        min_housekeeping_detected=2,
        max_top_gene_fraction=0.8,
    ).set_index("sample_id")

    assert bool(qc.loc["good", "passes_expression_qc"]) is True
    assert qc.loc["good", "sample_qc_status"] == "pass"
    assert qc.loc["good", "source_scale_class"] == "linear_rnaseq_tpm"
    assert bool(qc.loc["good", "linear_tpm_comparable"]) is True
    assert qc.loc["good", "housekeeping_genes_detected"] == 3
    assert bool(qc.loc["sparse", "passes_expression_qc"]) is False
    assert qc.loc["sparse", "sample_qc_status"] == "fail"
    assert qc.loc["sparse", "n_detected_raw"] == 2
    assert qc.loc["sparse", "n_detected_clean_biological"] == 2
    assert qc.loc["sparse", "n_detected_genes"] == 2
    assert qc.loc["sparse", "housekeeping_genes_detected"] == 1
    assert "low_housekeeping_detection" in qc.loc["sparse", "qc_flags"]
    assert "low_housekeeping_detection" in qc.loc["sparse", "sample_qc_reasons"]
    assert "high_top_gene_fraction" in qc.loc["sparse", "qc_flags"]


def test_per_sample_expression_filters_by_sample_qc(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2"],
            "Symbol": ["A", "B"],
            "pass_sample": [10.0, 20.0],
            "warn_sample": [30.0, 40.0],
            "fail_sample": [50.0, 60.0],
        }
    )
    path = tmp_path / "X.parquet"
    raw.to_parquet(path, index=False)
    expression._load_per_sample_matrix.cache_clear()
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        expression,
        "sample_expression_qc",
        lambda code, **k: pd.DataFrame(
            {
                "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
                "sample_qc_status": ["pass", "warn", "fail"],
            }
        ),
    )

    passed = expression.per_sample_expression("X", normalize="tpm_raw", sample_qc="pass")
    assert expression.sample_columns(passed) == ["pass_sample"]

    pass_or_warn = expression.per_sample_expression(
        "X", normalize="tpm_raw", sample_qc="pass_or_warn"
    )
    assert expression.sample_columns(pass_or_warn) == ["pass_sample", "warn_sample"]

    all_samples = expression.per_sample_expression("X", normalize="tpm_raw", sample_qc="all")
    assert expression.sample_columns(all_samples) == ["pass_sample", "warn_sample", "fail_sample"]


def test_source_matrix_sample_qc_manifest_missing_is_schema_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))

    out = expression.source_matrix_sample_qc_manifest(auto_fetch=False)

    assert out.empty
    assert "sample_qc_status" in out.columns
    assert out.attrs["schema_version"] == expression.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_SCHEMA_VERSION
    assert out.attrs["data_version"] == expression.DATA_VERSION
    assert "missing_reason" in out.attrs
    with pytest.raises(FileNotFoundError, match="source-matrix-sample-qc"):
        expression.source_matrix_sample_qc_manifest(auto_fetch=False, on_missing="raise")


def test_source_matrix_sample_qc_manifest_reads_filters_and_exports(tmp_path, monkeypatch):
    import oncoref

    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    pd.DataFrame(
        {
            "cancer_code": ["PRAD", "PRAD", "BRCA"],
            "sample_id": ["pass_sample", "warn_sample", "fail_sample"],
            "sample_qc_status": ["pass", "warn", "fail"],
            "source_cohort": ["SRC", "SRC", "SRC2"],
            "custom_future_column": [1, 2, 3],
        }
    ).to_csv(tmp_path / expression.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH, index=False)

    out = expression.source_matrix_sample_qc_manifest(
        "prostate", sample_qc="pass_or_warn", auto_fetch=False
    )

    assert out["sample_id"].tolist() == ["pass_sample", "warn_sample"]
    assert out["custom_future_column"].tolist() == [1, 2]
    assert out.attrs["path"].endswith(expression.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_PATH)
    assert oncoref.source_matrix_sample_qc_manifest is expression.source_matrix_sample_qc_manifest
    assert (
        oncoref.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_SCHEMA_VERSION
        == expression.SOURCE_MATRIX_SAMPLE_QC_MANIFEST_SCHEMA_VERSION
    )


def test_expression_artifact_build_metadata_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    pd.DataFrame(
        {
            "cancer_code": ["PRAD", "BRCA"],
            "source_cohort": ["SRC_PRAD", "SRC_BRCA"],
            "sample_qc": ["pass", "all"],
            "n_source_samples": [5, 7],
            "n_cohort_samples": [4, 7],
            "extra_future_column": ["x", "y"],
        }
    ).to_csv(tmp_path / expression.EXPRESSION_ARTIFACT_BUILD_METADATA_PATH, index=False)
    (tmp_path / expression.EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH).write_text(
        json.dumps(
            {
                "artifact": "expression-derived-shards",
                "sample_qc": "pass",
                "n_cohorts": 2,
            }
        )
        + "\n"
    )

    meta = expression.expression_artifact_build_metadata("PRAD", auto_fetch=False)
    summary = expression.expression_artifact_build_summary(auto_fetch=False)

    assert meta["source_cohort"].tolist() == ["SRC_PRAD"]
    assert meta["extra_future_column"].tolist() == ["x"]
    assert "n_qc_pass" in meta.columns
    assert (
        meta.attrs["schema_version"] == expression.EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    )
    assert summary["artifact"] == "expression-derived-shards"
    assert summary["schema_version"] == expression.EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    assert summary["path"].endswith(expression.EXPRESSION_ARTIFACT_BUILD_METADATA_JSON_PATH)


def test_expression_artifact_build_metadata_missing_can_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))

    out = expression.expression_artifact_build_metadata(auto_fetch=False)
    summary = expression.expression_artifact_build_summary(auto_fetch=False)

    assert out.empty
    assert (
        out.attrs["schema_version"] == expression.EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION
    )
    assert summary["missing_reason"]
    with pytest.raises(FileNotFoundError, match=r"expression-artifact-build-metadata\.csv"):
        expression.expression_artifact_build_metadata(auto_fetch=False, on_missing="raise")
    with pytest.raises(FileNotFoundError, match=r"expression-artifact-build-metadata\.json"):
        expression.expression_artifact_build_summary(auto_fetch=False, on_missing="raise")


def test_per_sample_expression_gene_and_proteoform_levels(tmp_path, monkeypatch):
    # ENSG1+ENSG2 are an identical-protein group; per_sample_expression(proteoform=True)
    # sums them per sample. Gene-level and proteoform-level are both available.
    import oncoref.proteoforms as pmod

    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A/B": ("ENSG1", "ENSG2")}
    )

    gene = expression.per_sample_expression("PRAD", normalize="tpm_raw")
    assert pmod.expression_level(gene) == "gene" and len(gene) == 3

    pf = expression.per_sample_expression("PRAD", normalize="tpm_raw", proteoform=True)
    assert pmod.expression_level(pf) == "proteoform" and len(pf) == 2  # group merged
    # A/B summed per sample: s1 = 500000 + 300000 = 800000
    a_row = pf[pf["proteoform_key"] == "A/B"]
    assert a_row["s1"].iloc[0] == pytest.approx(800000.0)


def test_per_sample_expression_proteoform_log_sums_in_linear_space(tmp_path, monkeypatch):
    # log1p + proteoform must sum members in LINEAR TPM then log1p — NOT sum the logs.
    import oncoref.proteoforms as pmod

    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    monkeypatch.setattr(
        pmod, "proteoform_group_map", lambda *, scope="cta": {"A/B": ("ENSG1", "ENSG2")}
    )
    lin = expression.per_sample_expression("PRAD", normalize="tpm_clean", proteoform=True)
    log = expression.per_sample_expression("PRAD", normalize="tpm_clean_log1p", proteoform=True)
    assert np.allclose(log["s1"].to_numpy(), np.log1p(lin["s1"].to_numpy(dtype=float)))


def test_per_sample_expression_memoizes_and_copies(tmp_path, monkeypatch):
    path = _raw_matrix(tmp_path)
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: path)
    expression._load_per_sample_matrix.cache_clear()

    a = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    expression.per_sample_expression("PRAD", normalize="tpm_clean")  # served from cache
    info = expression._load_per_sample_matrix.cache_info()
    assert info.misses == 1 and info.hits >= 1

    # Each call returns an independent copy — mutating one must not corrupt the cache.
    col = a.columns[2]
    a.loc[0, col] = -12345.0
    c = expression.per_sample_expression("PRAD", normalize="tpm_clean")
    assert c.loc[0, col] != -12345.0


def test_per_sample_expression_bad_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: _raw_matrix(tmp_path))
    with pytest.raises(ValueError, match="normalize must be one of"):
        expression.per_sample_expression("PRAD", normalize="zscore")


def test_per_sample_expression_no_autofetch_raises(monkeypatch, tmp_path):
    missing = tmp_path / "nope.parquet"
    monkeypatch.setattr(expression.source_matrices, "local_path", lambda code: missing)
    with pytest.raises(FileNotFoundError, match="not cached"):
        expression.per_sample_expression("PRAD", auto_fetch=False)


def test_cohort_mean_expression(monkeypatch):
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "s1": [10.0, 0.0],
            "s2": [20.0, 4.0],
            "s3": [30.0, 2.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: fixture.copy())
    mean = expression.cohort_mean_expression("X", statistic="mean")
    assert list(mean.columns) == ["Ensembl_Gene_ID", "Symbol", "expression"]
    assert dict(zip(mean["Symbol"], mean["expression"])) == {"A": 20.0, "B": 2.0}
    median = expression.cohort_mean_expression("X", statistic="median")
    assert dict(zip(median["Symbol"], median["expression"])) == {"A": 20.0, "B": 2.0}


def test_cohort_mean_expression_threads_proteoform_and_scope(monkeypatch):
    # cohort_mean delegates the collapse to per_sample_expression — it must pass
    # proteoform= and scope= through.
    from oncoref import expression_level

    seen = {}

    def fake_per_sample(code, *, normalize, auto_fetch, proteoform, scope, sample_qc):
        seen.update(proteoform=proteoform, scope=scope, sample_qc=sample_qc)
        return pd.DataFrame(
            {
                "proteoform_key": ["E1"],
                "Ensembl_Gene_ID": ["E1"],
                "Symbol": ["A"],
                "proteoform_members": ["A"],
                "s1": [1.0],
                "s2": [3.0],
            }
        )

    monkeypatch.setattr(expression, "per_sample_expression", fake_per_sample)
    out = expression.cohort_mean_expression("X", proteoform=True, scope="genome")
    assert seen == {"proteoform": True, "scope": "genome", "sample_qc": "pass"}
    assert expression_level(out) == "proteoform"  # proteoform_key carried through


def test_cancer_reference_expression_long_and_wide(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "p25": [1.0, 2.0],
            "p50": [3.0, 4.0],
            "p75": [5.0, 6.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X", "Y"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    long = expression.cancer_reference_expression("x", genes=["E1"])
    assert list(long.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "cancer_code",
        "normalization",
        "source_cohort",
        "source_type",
        "source_unit",
        "source_scale_class",
        "linear_tpm_comparable",
        "reference_method",
        "sample_qc",
        "data_version",
        "source_matrix_version",
        "expression",
        "q1",
        "q3",
    ]
    assert long["cancer_code"].tolist() == ["X"]
    assert long["normalization"].tolist() == ["tpm_clean"]
    assert long["reference_method"].tolist() == ["percentile_shard"]
    assert long["sample_qc"].tolist() == ["artifact"]
    assert long["expression"].tolist() == [3.0]
    assert long["q1"].tolist() == [1.0] and long["q3"].tolist() == [5.0]
    assert expression.cancer_reference_expression("x", genes=["E1"], normalize="clean_tpm").equals(
        long
    )

    wide = expression.cancer_reference_expression(["x", "y"], format="wide")
    assert {"X_TPM_clean", "Y_TPM_clean"} <= set(wide.columns)
    assert wide.loc[wide["Symbol"] == "A", "X_TPM_clean"].iloc[0] == 3.0


def test_cancer_reference_expression_multiple_normalizations(monkeypatch):
    pct_tpm = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [3.0],
            "p75": [5.0],
        }
    )
    pct_log = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [np.log1p(1.0)],
            "p50": [np.log1p(3.0)],
            "p75": [np.log1p(5.0)],
        }
    )

    def fake_pct(code, *, as_tpm=True, **kwargs):
        return pct_tpm.copy() if as_tpm else pct_log.copy()

    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", fake_pct)

    long = expression.cancer_reference_expression("x", normalize=["clean_tpm", "tpm_clean_log1p"])
    assert long["normalization"].tolist() == ["tpm_clean", "tpm_clean_log1p"]
    assert long["expression"].tolist() == [3.0, pytest.approx(np.log1p(3.0))]

    wide = expression.cancer_reference_expression(
        "x", normalize=["tpm_clean", "tpm_clean_log1p"], format="wide"
    )
    assert wide.loc[0, "X_TPM_clean"] == 3.0
    assert wide.loc[0, "X_TPM_clean_log1p"] == pytest.approx(np.log1p(3.0))


def test_cancer_reference_expression_availability_reports_missing(monkeypatch):
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["Y"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())

    out = expression.cancer_reference_expression_availability(
        ["x", "z"], normalize=["tpm_clean", "tpm_raw"]
    )

    assert list(out.columns) == [
        "requested_code",
        "cancer_code",
        "request_kind",
        "normalization",
        "available",
        "missing_reason",
        "source_cohort",
        "source_type",
        "source_unit",
        "source_scale_class",
        "linear_tpm_comparable",
        "reference_method",
        "sample_qc",
        "artifact_schema_version",
        "data_version",
        "source_matrix_version",
    ]
    assert out.attrs["artifact_schema_version"] == expression.REFERENCE_EXPRESSION_SCHEMA_VERSION
    keyed = out.set_index(["cancer_code", "normalization"])
    assert bool(keyed.loc[("X", "tpm_clean"), "available"]) is True
    assert keyed.loc[("X", "tpm_clean"), "sample_qc"] == "artifact"
    assert keyed.loc[("X", "tpm_raw"), "sample_qc"] == "pass"
    assert keyed.loc[("X", "tpm_raw"), "missing_reason"] == "no_source_matrix"
    assert keyed.loc[("Z", "tpm_clean"), "missing_reason"] == "no_percentile_artifact"


def test_cancer_reference_expression_missing_empty_and_raise(monkeypatch):
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: [])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())

    empty = expression.cancer_reference_expression("z", on_missing="empty")
    assert empty.empty
    assert list(empty.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "cancer_code",
        "normalization",
        "source_cohort",
        "source_type",
        "source_unit",
        "source_scale_class",
        "linear_tpm_comparable",
        "reference_method",
        "sample_qc",
        "data_version",
        "source_matrix_version",
        "expression",
        "q1",
        "q3",
    ]
    assert empty.attrs["missing_requests"][0]["cancer_code"] == "Z"
    assert empty.attrs["missing_requests"][0]["missing_reason"] == "no_percentile_artifact"
    with pytest.raises(ValueError, match="Z/tpm_clean: no_percentile_artifact"):
        expression.cancer_reference_expression("z", on_missing="raise")


def test_cancer_reference_expression_request_metadata_for_aggregate(monkeypatch):
    pct = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [1.0],
            "p50": [3.0],
            "p75": [5.0],
        }
    )
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X", "Y"])
    monkeypatch.setattr(expression, "cohort_aggregates", lambda: {"AGG": ["X", "Y"]})
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(expression, "cohort_gene_percentiles", lambda *a, **k: pct.copy())

    out = expression.cancer_reference_expression("agg", include_request_metadata=True)

    assert out["requested_code"].tolist() == ["AGG", "AGG"]
    assert out["cancer_code"].tolist() == ["X", "Y"]
    assert out["request_kind"].tolist() == ["aggregate_member", "aggregate_member"]
    assert out["available"].tolist() == [True, True]
    assert out["missing_reason"].tolist() == ["", ""]


def test_cancer_reference_expression_raw_tpm_uses_source_stats(monkeypatch):
    stats = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "p25": [10.0],
            "p50": [20.0],
            "p75": [30.0],
        }
    )
    seen = {}
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression.source_matrices, "available_cohorts", lambda: ["X"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(
        expression,
        "cohort_stats",
        lambda code, **k: seen.update(k) or stats.copy(),
    )
    monkeypatch.setattr(
        expression,
        "_selected_expression_source_metadata",
        lambda code: {
            "source_cohort": "SRC_X",
            "source_type": "gdc",
            "unit": "TPM",
            "source_scale_class": "linear_rnaseq_tpm",
            "linear_tpm_comparable": True,
            "tpm_proxy": False,
        },
    )

    long = expression.cancer_reference_expression("x", normalize="tpm", auto_fetch=True)
    assert seen == {"normalize": "tpm_raw", "auto_fetch": True, "sample_qc": "pass"}
    assert long["normalization"].tolist() == ["tpm_raw"]
    assert long["sample_qc"].tolist() == ["pass"]

    expression.cancer_reference_expression("x", normalize="tpm", auto_fetch=True, sample_qc="all")
    assert seen == {"normalize": "tpm_raw", "auto_fetch": True, "sample_qc": "all"}
    assert long["source_cohort"].tolist() == ["SRC_X"]
    assert long["source_type"].tolist() == ["gdc"]
    assert long["reference_method"].tolist() == ["source_matrix_stats"]
    assert long["expression"].tolist() == [20.0]


def test_cancer_reference_expression_wide_merges_by_gene_id_not_symbol(monkeypatch):
    by_code = {
        "X": pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["E1", "E2"],
                "Symbol": ["A", "B"],
                "p25": [1.0, 2.0],
                "p50": [3.0, 4.0],
                "p75": [5.0, 6.0],
            }
        ),
        "Y": pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["E1", "E3"],
                "Symbol": ["A_OLD", "C"],
                "p25": [5.0, 8.0],
                "p50": [7.0, 9.0],
                "p75": [9.0, 10.0],
            }
        ),
    }
    monkeypatch.setattr(expression, "available_percentile_cohorts", lambda: ["X", "Y"])
    monkeypatch.setattr(expression, "resolve_cancer_type", lambda code: str(code).upper())
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda code, *a, **k: by_code[code].copy()
    )

    wide = expression.cancer_reference_expression(["x", "y"], format="wide")
    e1 = wide[wide["Ensembl_Gene_ID"] == "E1"]
    assert len(e1) == 1
    assert e1["Symbol"].iloc[0] == "A"
    assert e1["X_TPM_clean"].iloc[0] == pytest.approx(3.0)
    assert e1["Y_TPM_clean"].iloc[0] == pytest.approx(7.0)


def test_proteoform_named_accessors_delegate_with_proteoform_true(monkeypatch):
    # The proteoform_* named accessors are thin wrappers that set proteoform=True on
    # their gene-level base, threading scope/statistic/etc. through.
    seen = {}
    empty = pd.DataFrame()

    monkeypatch.setattr(
        expression, "per_sample_expression", lambda c, **k: seen.update(ps=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_mean_expression", lambda c, **k: seen.update(mean=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda c, **k: seen.update(pct=k) or empty
    )
    monkeypatch.setattr(
        expression, "within_sample_top_fraction", lambda c, **k: seen.update(ws=k) or empty
    )

    expression.proteoform_per_sample_expression("X", scope="genome")
    expression.proteoform_cohort_mean_expression("X", statistic="median", scope="genome")
    expression.proteoform_cohort_percentiles("X", as_tpm=False)
    expression.proteoform_within_sample_top_fraction("X", threshold=0.9)

    assert seen["ps"]["proteoform"] is True and seen["ps"]["scope"] == "genome"
    assert seen["mean"]["proteoform"] is True
    assert seen["mean"]["statistic"] == "median" and seen["mean"]["scope"] == "genome"
    assert seen["pct"]["proteoform"] is True and seen["pct"]["as_tpm"] is False
    assert seen["ws"]["proteoform"] is True and seen["ws"]["threshold"] == 0.9


def test_gene_named_accessors_are_gene_level(monkeypatch):
    # The symmetric gene_* accessors delegate to the gene-level base (proteoform not set,
    # or False), exposing only gene-relevant params.
    seen = {}
    empty = pd.DataFrame()

    monkeypatch.setattr(
        expression, "per_sample_expression", lambda c, **k: seen.update(ps=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_mean_expression", lambda c, **k: seen.update(mean=k) or empty
    )
    monkeypatch.setattr(
        expression, "cohort_gene_percentiles", lambda c, **k: seen.update(pct=k) or empty
    )
    monkeypatch.setattr(
        expression, "within_sample_top_fraction", lambda c, **k: seen.update(ws=k) or empty
    )

    expression.gene_per_sample_expression("X")
    expression.gene_cohort_mean_expression("X", statistic="median")
    expression.gene_cohort_percentiles("X", as_tpm=False)
    expression.gene_within_sample_top_fraction("X", threshold=0.9)

    assert seen["ps"].get("proteoform", False) is False  # gene level
    assert seen["mean"].get("proteoform", False) is False and seen["mean"]["statistic"] == "median"
    assert seen["pct"].get("proteoform", False) is False and seen["pct"]["as_tpm"] is False
    assert seen["ws"].get("proteoform", False) is False and seen["ws"]["threshold"] == 0.9


def test_cohort_stats_full_suite(monkeypatch):
    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "s1": [10.0],
            "s2": [20.0],
            "s3": [30.0],
            "s4": [40.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: fixture.copy())
    s = expression.cohort_stats("X")
    assert set(s.columns) >= {
        "mean",
        "std",
        "min",
        "p1",
        "p5",
        "p10",
        "p15",
        "p20",
        "p25",
        "p50",
        "p75",
        "p80",
        "p85",
        "p90",
        "p95",
        "p99",
        "max",
    }
    row = s.iloc[0]
    assert row["mean"] == pytest.approx(25.0)
    assert row["std"] == pytest.approx(np.std([10, 20, 30, 40]))
    assert row["min"] == 10.0 and row["max"] == 40.0
    assert row["p50"] == pytest.approx(25.0)
    assert row["p25"] == pytest.approx(17.5) and row["p75"] == pytest.approx(32.5)
    assert row["p90"] == pytest.approx(37.0)


def test_cohort_stats_named_accessors_delegate(monkeypatch):
    seen = {}
    fixture = pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["A"], "s1": [1.0], "s2": [2.0]})
    monkeypatch.setattr(
        expression, "per_sample_expression", lambda c, **k: seen.update(k) or fixture.copy()
    )
    expression.gene_cohort_stats("X")
    assert seen.get("proteoform", False) is False
    expression.proteoform_cohort_stats("X", scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome"


def _pan_cancer_fixture():
    return pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000001.5", "ENSG00000002"],
            "Symbol": ["GENE1", "GENE2"],
            "nTPM_liver": [3.0, 7.0],
            "FPKM_LUAD": [2.0, 8.0],
            "FPKM_BLCA": [5.0, 5.0],
        }
    )


def test_pan_cancer_expression_converts_fpkm_to_tpm(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    out = expression.pan_cancer_expression()
    # FPKM_<CODE> tumor columns become entity-first <CODE>_TPM_raw;
    # HPA nTPM columns become <tissue>_nTPM_raw. Clean companions are default.
    assert "LUAD_FPKM_raw" in out.columns and "FPKM_LUAD" not in out.columns
    assert "LUAD_TPM_raw" in out.columns
    assert "liver_nTPM_raw" in out.columns
    assert "LUAD_TPM_clean" in out.columns and "liver_nTPM_clean" in out.columns
    assert "TPM_LUAD" not in out.columns and "nTPM_liver" not in out.columns
    assert out["LUAD_FPKM_raw"].tolist() == pytest.approx([2.0, 8.0])
    # Each TCGA column is rescaled to sum 1e6: FPKM_LUAD [2,8] -> [200000, 800000].
    assert out["LUAD_TPM_raw"].tolist() == pytest.approx([200000.0, 800000.0])
    assert out["BLCA_TPM_raw"].sum() == pytest.approx(1e6)


def test_pan_cancer_expression_raw_only(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    out = expression.pan_cancer_expression(normalize=None)
    assert "LUAD_FPKM_raw" in out.columns
    assert "LUAD_TPM_raw" in out.columns
    assert "LUAD_TPM_clean" not in out.columns
    assert out["LUAD_FPKM_raw"].tolist() == pytest.approx([2.0, 8.0])
    assert out["LUAD_TPM_raw"].tolist() == pytest.approx([200000.0, 800000.0])


def test_pan_cancer_expression_pirlygenes_column_style(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    out = expression.pan_cancer_expression(normalize="tpm", column_style="pirlygenes")

    assert {"liver_nTPM", "LUAD_FPKM", "LUAD_TPM"} <= set(out.columns)
    assert not any(c.endswith("_raw") for c in out.columns)
    assert out["LUAD_FPKM"].tolist() == pytest.approx([2.0, 8.0])
    assert out["LUAD_TPM"].tolist() == pytest.approx([200000.0, 800000.0])
    assert out.attrs["oncoref"]["column_style"] == "pirlygenes"


def test_pan_cancer_expression_to_tpm_legacy_keyword(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    out = expression.pan_cancer_expression(genes=["ENSG00000001"], to_tpm=True)

    assert out["Symbol"].tolist() == ["GENE1"]
    assert {"liver_nTPM", "LUAD_FPKM", "LUAD_TPM"} <= set(out.columns)
    assert "LUAD_TPM_clean" not in out.columns
    assert out["LUAD_TPM"].iloc[0] == pytest.approx(200000.0)


def test_pan_cancer_expression_empty_gene_filter_preserves_schema(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    empty = expression.pan_cancer_expression(genes=[], normalize="tpm", column_style="pirlygenes")
    full = expression.pan_cancer_expression(normalize="tpm", column_style="pirlygenes")

    assert empty.empty
    assert list(empty.columns) == list(full.columns)
    assert empty.attrs["oncoref"]["dataset"] == "pan-cancer-expression"


def test_pan_cancer_expression_bad_column_style(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    with pytest.raises(ValueError, match="column_style"):
        expression.pan_cancer_expression(column_style="source")


def test_pan_cancer_expression_accepts_clean_tpm_alias(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    canonical = expression.pan_cancer_expression(normalize="tpm_clean")
    alias = expression.pan_cancer_expression(normalize="clean_tpm")
    assert "LUAD_TPM_clean" in alias.columns
    assert alias["LUAD_TPM_clean"].tolist() == pytest.approx(canonical["LUAD_TPM_clean"].tolist())


def test_pan_cancer_expression_housekeeping_mode(monkeypatch):
    import oncoref.gene_families as gf

    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    monkeypatch.setattr(
        gf, "clean_tpm_biological_housekeeping_gene_ids", lambda: frozenset({"ENSG00000001"})
    )
    out = expression.pan_cancer_expression(normalize=["tpm_clean", "hk", "percentile"])
    assert "LUAD_TPM_hk" in out.columns
    assert "liver_nTPM_hk" in out.columns
    assert "LUAD_TPM_percentile" in out.columns


def test_pan_cancer_expression_log_modes(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    raw_logged = expression.pan_cancer_expression(normalize="tpm_log1p")
    clean_logged = expression.pan_cancer_expression(normalize="tpm_clean_log1p")
    assert "LUAD_TPM_raw_log1p" in raw_logged.columns
    assert "LUAD_TPM_clean_log1p" not in raw_logged.columns
    assert raw_logged["LUAD_TPM_raw_log1p"].iloc[0] == pytest.approx(
        np.log1p(raw_logged["LUAD_TPM_raw"].iloc[0])
    )
    assert "LUAD_TPM_clean_log1p" in clean_logged.columns


def test_pan_cancer_expression_gene_filter(monkeypatch):
    monkeypatch.setattr(expression, "get_data", lambda name: _pan_cancer_fixture())
    # Filter by symbol, and by unversioned Ensembl id (fixture id is versioned).
    by_symbol = expression.pan_cancer_expression(genes="GENE2")
    assert by_symbol["Symbol"].tolist() == ["GENE2"]
    by_id = expression.pan_cancer_expression(genes=["ENSG00000001"])
    assert by_id["Symbol"].tolist() == ["GENE1"]
    # Conversion still reflects the cohort-wide scaling computed before filtering.
    assert by_id["LUAD_TPM_raw"].iloc[0] == pytest.approx(200000.0)


def test_pooled_cohort_stats_availability_and_heterogeneity(monkeypatch):
    # Two cohorts with overlapping but ragged gene sets and very different sizes.
    # BIG measures E1+E2 (3 samples), SMALL measures E2+E3 (1 sample). E2 is shared.
    big = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "s1": [10.0, 0.0],
            "s2": [20.0, 0.0],
            "s3": [30.0, 6.0],
        }
    )
    small = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E2", "E3"],
            "Symbol": ["B", "C"],
            "x1": [100.0, 4.0],
        }
    )
    frames = {"BIG": big, "SMALL": small}
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: frames[code].copy())
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: list(ct))

    out = expression.pooled_cohort_stats(["BIG", "SMALL"]).set_index("Ensembl_Gene_ID")

    # n_samples is the constant pooled width; n_available is the honest per-gene
    # denominator; n_cohorts counts measuring cohorts.
    assert (out["n_samples"] == 4).all()
    assert out.loc["E1", "n_available"] == 3 and out.loc["E1", "n_cohorts"] == 1
    assert out.loc["E2", "n_available"] == 4 and out.loc["E2", "n_cohorts"] == 2
    assert out.loc["E3", "n_available"] == 1 and out.loc["E3", "n_cohorts"] == 1
    # n_detected ignores measured-but-zero: E2 is 0 in BIG's s1/s2, >0 in s3 + SMALL.
    assert out.loc["E2", "n_detected"] == 2

    # E2 sample-pooled mean is (0+0+6+100)/4 = 26.5; the balanced (equal-cohort) mean
    # is mean(BIG_mean=2.0, SMALL_mean=100.0) = 51.0 — the big cohort no longer
    # dominates. std_between captures the cross-cohort spread (>0 for E2).
    assert out.loc["E2", "mean"] == pytest.approx(26.5)
    assert out.loc["E2", "balanced_mean"] == pytest.approx(51.0)
    assert out.loc["E2", "std_between"] == pytest.approx(np.std([2.0, 100.0]))
    # A single-cohort gene has no between-cohort spread.
    assert np.isnan(out.loc["E1", "std_between"])
    # std is NaN for a gene with <2 measured samples (E3: one sample in SMALL); a
    # multi-sample gene gets a real std.
    assert np.isnan(out.loc["E3", "std"])
    assert out.loc["E1", "std"] == pytest.approx(np.std([10.0, 20.0, 30.0]))
    # Off-panel cells are never imputed to zero: E1's max is BIG's max (30), not
    # dragged down by SMALL's missing samples.
    assert out.loc["E1", "max"] == 30.0


def test_pooled_cohort_stats_merges_alt_haplotype_aliases(tmp_path, monkeypatch):
    # End-to-end #465-class fix: per_sample_expression now returns the dense canonical
    # space, so when cohort A carries an alt-haplotype id and cohort B its primary,
    # pooling sees ONE canonical gene across both cohorts — not two disjoint sparse rows.
    # Drive the REAL per_sample (normalize="tpm_raw" so values pass through clean_tpm-free
    # for an exact pooled mean); the canonicalization itself lives one layer down now.
    from oncoref.gene_ids import ensembl_id_aliases

    alt, primary = next((a, p) for a, p in ensembl_id_aliases().items() if a != p)
    pa, pb = tmp_path / "A.parquet", tmp_path / "B.parquet"
    pd.DataFrame({"Ensembl_Gene_ID": [alt], "Symbol": ["G"], "s1": [10.0]}).to_parquet(
        pa, index=False
    )
    pd.DataFrame({"Ensembl_Gene_ID": [primary], "Symbol": ["G"], "x1": [20.0]}).to_parquet(
        pb, index=False
    )
    paths = {"A": pa, "B": pb}
    expression._load_per_sample_matrix.cache_clear()
    monkeypatch.setattr(expression.source_matrices, "ensure", lambda code: paths[code])
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: list(ct))

    out = expression.pooled_cohort_stats(
        ["A", "B"], normalize="tpm_raw", sample_qc="all"
    ).set_index("Ensembl_Gene_ID")
    assert list(out.index) == [primary]  # one canonical row, the primary id
    assert alt not in set(out.index)
    # both cohorts pooled onto it (mean of the two single-sample cohorts)
    assert out.loc[primary, "n_cohorts"] == 2
    assert out.loc[primary, "n_samples"] == 2
    assert out.loc[primary, "mean"] == pytest.approx(15.0)


def test_pooled_cohort_stats_expands_aggregate_cohorts(monkeypatch):
    # An aggregate code (CRC = COAD + READ) pools its member subtypes, not the
    # rollup label (which has no per-sample matrix of its own).
    served = {
        "COAD": pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["A"], "c1": [10.0]}),
        "READ": pd.DataFrame({"Ensembl_Gene_ID": ["E1"], "Symbol": ["A"], "r1": [20.0]}),
    }
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: served[code].copy())
    out = expression.pooled_cohort_stats(["CRC"]).set_index("Ensembl_Gene_ID")
    # Both members contributed: 2 cohorts, 2 samples, pooled mean (10+20)/2 = 15.
    assert out.loc["E1", "n_cohorts"] == 2
    assert out.loc["E1", "n_samples"] == 2
    assert out.loc["E1", "mean"] == pytest.approx(15.0)


def test_pooled_cohort_stats_min_cohorts_filter(monkeypatch):
    big = pd.DataFrame({"Ensembl_Gene_ID": ["E1", "E2"], "Symbol": ["A", "B"], "s1": [1.0, 2.0]})
    small = pd.DataFrame({"Ensembl_Gene_ID": ["E2"], "Symbol": ["B"], "x1": [3.0]})
    frames = {"BIG": big, "SMALL": small}
    monkeypatch.setattr(expression, "per_sample_expression", lambda code, **k: frames[code].copy())
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: list(ct))
    out = expression.pooled_cohort_stats(["BIG", "SMALL"], min_cohorts=2)
    # Only E2 is measured by both cohorts.
    assert out["Ensembl_Gene_ID"].tolist() == ["E2"]


def test_pooled_cohort_stats_requires_cohorts(monkeypatch):
    monkeypatch.setattr(expression, "_resolve_cancer_types", lambda ct, **k: [])
    with pytest.raises(ValueError, match="at least one cancer type"):
        expression.pooled_cohort_stats([])


def test_pooled_cohort_stats_named_accessors_delegate(monkeypatch):
    seen = {}

    def _fake_pool(ct, **k):
        seen.update(k)
        seen["ct"] = ct
        return pd.DataFrame()

    monkeypatch.setattr(expression, "pooled_cohort_stats", _fake_pool)
    expression.gene_pooled_cohort_stats(["X"])
    assert seen.get("proteoform", False) is False
    expression.proteoform_pooled_cohort_stats(["X"], scope="genome")
    assert seen["proteoform"] is True and seen["scope"] == "genome"


def test_cohort_mean_expression_bad_statistic(monkeypatch):
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="statistic must be"):
        expression.cohort_mean_expression("X", statistic="mode")


def test_cohort_mean_expression_reduces_proteoform_frame(monkeypatch):
    # Given a proteoform-level per-sample frame (members already summed), cohort_mean
    # reduces over patients keeping the proteoform key space.
    collapsed = pd.DataFrame(
        {
            "proteoform_key": ["A1/2", "ENSG_B"],
            "Ensembl_Gene_ID": ["ENSG_A1", "ENSG_B"],
            "Symbol": ["A1/2", "B"],
            "proteoform_members": ["A1/A2", "B"],
            "s1": [8.0, 1.0],  # A1+A2 already summed per sample
            "s2": [2.0, 9.0],
        }
    )
    monkeypatch.setattr(expression, "per_sample_expression", lambda *a, **k: collapsed.copy())
    out = expression.cohort_mean_expression("X", statistic="mean", proteoform=True)
    assert "proteoform_key" in out.columns
    by_key = dict(zip(out["proteoform_key"], out["expression"]))
    assert by_key["A1/2"] == pytest.approx(5.0)  # (8 + 2) / 2
    assert by_key["ENSG_B"] == pytest.approx(5.0)  # (1 + 9) / 2, singleton keyed by ENSG
