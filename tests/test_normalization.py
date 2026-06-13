# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expression normalization: clean TPM + helpers (#35, Phase N)."""

import numpy as np
import pandas as pd
import pytest

from cancerdata import gene_families as gf
from cancerdata import normalization as norm


def _matrix():
    tech = list(gf.gene_family_ids("mitochondrial"))[:2]
    gt = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [*tech, "ENSG00000111111", "ENSG00000222222", "ENSG00000333333"],
            "Symbol": ["MT1", "MT2", "BIO1", "BIO2", "BIO3"],
        }
    )
    vals = pd.DataFrame(
        {"s1": [100.0, 200, 300, 400, 500], "s2": [10.0, 0, 100, 200, 300]}, index=gt.index
    )
    return gt, vals


def test_clean_tpm_two_compartment_budget():
    gt, vals = _matrix()
    clean = norm.clean_tpm(vals, gt)
    rem = norm._censored_mask(gt, exclude_ribosomal_proteins=True).to_numpy()
    # technical -> 250k, biological -> 750k, exactly, per sample.
    assert np.allclose(clean.loc[rem].sum(), 250_000.0)
    assert np.allclose(clean.loc[~rem].sum(), 750_000.0)
    # within-biology ratios preserved (300:400:500)
    bio = clean.loc[~rem, "s1"].to_numpy()
    assert np.allclose(bio / bio.min(), [1.0, 4 / 3, 5 / 3])


def test_clean_tpm_no_technical_mass():
    gt, vals = _matrix()
    rem = norm._censored_mask(gt, exclude_ribosomal_proteins=True)
    vals.loc[rem] = 0.0  # a sample/library with no technical reads
    clean = norm.clean_tpm(vals, gt)
    assert np.allclose(clean.loc[rem.to_numpy()].sum(), 0.0)  # technical stays 0
    assert np.allclose(clean.loc[~rem.to_numpy()].sum(), 750_000.0)  # biology still 75%


def test_clean_tpm_validates():
    gt, vals = _matrix()
    with pytest.raises(ValueError, match="technical_fraction"):
        norm.clean_tpm(vals, gt, technical_fraction=1.5)
    with pytest.raises(ValueError, match="gene_table or removable"):
        norm.clean_tpm(vals)


def test_value_cols_excludes_proteoform_members():
    # A proteoform-collapsed frame carries a proteoform_members provenance column; the
    # "everything-not-id" value-column rule must not treat it as a sample (it would
    # crash/poison geomean normalization). Uses the shared ID_COLUMNS constant.
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["GA"],
            "proteoform_members": ["GA/GB"],
            "s1": [3.0],
            "s2": [4.0],
        }
    )
    assert norm._value_cols(df) == ["s1", "s2"]


def test_drop_technical_vs_filter_technical_rna():
    gt, _ = _matrix()
    df = gt.assign(s1=1.0)
    # drop_technical_genes removes the censored set (incl. ribosomal); the two
    # mito genes here are technical, so both go.
    assert len(norm.drop_technical_genes(df)) == 3
    assert len(norm.filter_technical_rna(df)) == 3  # mito is technical RNA too


def test_log_and_rank_helpers():
    gt, vals = _matrix()
    df = vals.assign(Ensembl_Gene_ID=gt["Ensembl_Gene_ID"], Symbol=gt["Symbol"])
    log1p = norm.log1p_transform(df)
    assert np.allclose(log1p["s1"], np.log1p(vals["s1"]))
    rank = norm.percentile_rank(df)
    assert rank["s1"].max() <= 100.0 and rank["s1"].min() >= 0.0


def test_normalize_to_housekeeping():
    # housekeeping genes present -> column scaled by their median (1.0 = baseline)
    hk = list(gf.housekeeping_gene_ids())[:2]
    gt = pd.DataFrame({"Ensembl_Gene_ID": [*hk, "ENSG00000999999"], "Symbol": ["H1", "H2", "X"]})
    df = gt.assign(s1=[10.0, 30.0, 100.0])  # hk median = 20
    out = norm.normalize_to_housekeeping(df)
    assert np.allclose(out["s1"], [0.5, 1.5, 5.0])


# ---- FPKM->TPM / renormalize-to-million ----


def test_renormalize_to_million_rescales_each_column():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2"],
            "Symbol": ["A", "B"],
            "s1_TPM": [1.0, 3.0],  # sum 4 -> scale 250000
            "s2_TPM": [10.0, 10.0],  # sum 20 -> scale 50000
        }
    )
    out, stats = norm.renormalize_to_million(df)
    assert out["s1_TPM"].sum() == pytest.approx(1e6)
    assert out["s2_TPM"].sum() == pytest.approx(1e6)
    assert stats["applied"] is True
    assert stats["columns"]["s1_TPM"]["scale"] == pytest.approx(250000.0)
    # id columns untouched
    assert list(out["Symbol"]) == ["A", "B"]


def test_renormalize_ignores_raw_and_zero_columns():
    df = pd.DataFrame(
        {
            "s1_TPM": [1.0, 1.0],
            "s2_TPM_raw": [5.0, 5.0],  # _raw provenance -> never rescaled
            "s3_TPM": [0.0, 0.0],  # zero sum -> left untouched, scale 1.0
        }
    )
    out, stats = norm.renormalize_to_million(df)
    assert list(out["s2_TPM_raw"]) == [5.0, 5.0]  # untouched
    assert "s2_TPM_raw" not in stats["columns"]
    assert stats["columns"]["s3_TPM"]["scale"] == 1.0
    assert out["s1_TPM"].sum() == pytest.approx(1e6)


def test_fpkm_to_tpm_equals_renormalize():
    df = pd.DataFrame({"x_FPKM": [2.0, 6.0, 2.0]})
    out, _ = norm.fpkm_to_tpm(df, value_cols=["x_FPKM"])
    assert out["x_FPKM"].sum() == pytest.approx(1e6)
    assert out["x_FPKM"].to_numpy() == pytest.approx([2e5, 6e5, 2e5])


def test_is_expression_value_col():
    assert norm.is_expression_value_col("LUAD_TPM_clean")
    assert norm.is_expression_value_col("TPM")
    assert not norm.is_expression_value_col("LUAD_TPM_raw")
    assert not norm.is_expression_value_col("Ensembl_Gene_ID")


# ---- normalize_expression (technical-RNA zero+renormalize) ----


def test_normalize_expression_zeros_technical_and_preserves_total():
    df = pd.DataFrame(
        {
            "Symbol": ["PRAME", "MT-CO1", "RPL13", "MALAT1", "TP53"],
            "Ensembl_Gene_ID": ["E1", "E2", "E3", "E4", "E5"],
            "s1_TPM": [
                100.0,
                300.0,
                100.0,
                200.0,
                300.0,
            ],  # sum 1000; technical = MT-CO1+MALAT1=500
        }
    )
    out, stats = norm.normalize_expression(df, value_cols=["s1_TPM"])
    by = dict(zip(out["Symbol"], out["s1_TPM"]))
    assert by["MT-CO1"] == 0.0 and by["MALAT1"] == 0.0  # technical zeroed
    assert out["s1_TPM"].sum() == pytest.approx(1000.0)  # total preserved
    # RPL13 is kept (ribosomal protein is NOT technical) -> rescaled up
    assert by["RPL13"] > 100.0
    assert stats["applied"] is True
    assert stats["removed_technical_gene_count"] == 2


def test_normalize_expression_fixed_fraction_delegates_to_clean_tpm():
    df = pd.DataFrame(
        {
            "Symbol": ["A", "B"],
            "Ensembl_Gene_ID": ["E1", "E2"],
            "s1_TPM": [10.0, 30.0],
        }
    )
    out, stats = norm.normalize_expression(
        df, value_cols=["s1_TPM"], censored_fill="fixed_fraction"
    )
    # no technical genes -> biological compartment fills 750k
    assert out["s1_TPM"].sum() == pytest.approx(750000.0, rel=1e-6)
    assert stats["mode"] == "fixed_fraction"


def test_normalize_long_table_groups_independently():
    df = pd.DataFrame(
        {
            "symbol": ["MT-CO1", "TP53", "MT-CO1", "TP53"],
            "Ensembl_Gene_ID": ["E2", "E5", "E2", "E5"],
            "cancer_code": ["LUAD", "LUAD", "SKCM", "SKCM"],
            "tumor_tpm_median": [400.0, 600.0, 100.0, 900.0],
        }
    )
    out, _ = norm.normalize_technical_rna_long_table(
        df, group_cols=["cancer_code"], value_cols=["tumor_tpm_median"]
    )
    by = {(r.cancer_code, r.symbol): r.tumor_tpm_median for r in out.itertuples()}
    assert by[("LUAD", "MT-CO1")] == 0.0
    assert by[("LUAD", "TP53")] == pytest.approx(1000.0)  # 600 rescaled to the 1000 group total
    assert by[("SKCM", "TP53")] == pytest.approx(1000.0)


def test_tpm_to_housekeeping_normalized():
    from cancerdata import gene_families

    hk = list(gene_families.housekeeping_gene_ids())[:2]
    df = pd.DataFrame(
        {
            "Symbol": ["HK1", "HK2", "GENE"],
            "Ensembl_Gene_ID": [hk[0], hk[1], "ENSG_X"],
            "s1_TPM": [100.0, 100.0, 50.0],
        }
    )
    out, stats = norm.tpm_to_housekeeping_normalized(df, value_cols=["s1_TPM"])
    # geomean of [100,100] (+0.1) ~ 100.1 -> GENE 50 / 100.1 ~ 0.4995
    assert stats["applied"] is True
    assert out.loc[out["Symbol"] == "GENE", "s1_TPM"].iloc[0] == pytest.approx(50 / 100.1, rel=1e-3)
