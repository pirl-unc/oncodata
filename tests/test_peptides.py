# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from oncodata import peptides


class _FakeTr:
    def __init__(self, gene_id, biotype, protein_sequence):
        self.gene_id = gene_id
        self.biotype = biotype
        self.protein_sequence = protein_sequence


class _FakeGenome:
    release = 999

    def __init__(self, transcripts):
        self._trs = transcripts

    def transcripts(self):
        return self._trs


def test_kmers():
    assert peptides._kmers("AAACCC", 3) == {"AAA", "AAC", "ACC", "CCC"}
    assert peptides._kmers("AB", 3) == set()  # shorter than k


def test_longest_protein_per_gene_keeps_longest():
    genome = _FakeGenome(
        [
            _FakeTr("G1", "protein_coding", "AAAA"),
            _FakeTr("G1", "protein_coding", "AAAAAA"),  # longer -> wins
            _FakeTr("G1", "lncRNA", "ZZZZZZZZ"),  # non-coding -> ignored
            _FakeTr("G2", "protein_coding", "CC"),  # shorter than k -> dropped
            _FakeTr("G3", "protein_coding", "MKLP*"),  # stop codon stripped
        ]
    )
    longest = peptides._longest_protein_per_gene(genome, 3)
    assert longest["G1"] == "AAAAAA"
    assert "G2" not in longest
    assert longest["G3"] == "MKLP"


@pytest.fixture
def fake_proteome(monkeypatch, tmp_path):
    # CTA protein "AAACCC" (3-mers: AAA,AAC,ACC,CCC); background "CCCDDD" shares CCC.
    genome = _FakeGenome(
        [
            _FakeTr("ENSG_CTA", "protein_coding", "AAACCC"),
            _FakeTr("ENSG_BG", "protein_coding", "CCCDDD"),
            _FakeTr("ENSG_NC", "lncRNA", "EEEEEE"),
        ]
    )
    monkeypatch.setattr(peptides, "_usable_genome", lambda: genome)
    monkeypatch.setattr(peptides, "_derived_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(peptides, "CTA_gene_ids", lambda: ["ENSG_CTA"])
    monkeypatch.setattr(peptides, "CTA_unfiltered_gene_ids", lambda: ["ENSG_CTA"])
    monkeypatch.setattr(peptides, "CTA_gene_id_to_name", lambda: {"ENSG_CTA": "CTAX"})
    monkeypatch.setattr(peptides, "_MIN_PROTEOME_GENES", 1)  # tiny fake proteome
    peptides._COUNTS_CACHE.clear()
    yield
    peptides._COUNTS_CACHE.clear()


def test_specific_9mer_counts_subtracts_background(fake_proteome):
    df = peptides.cta_specific_9mer_counts(k=3)
    assert list(df["Symbol"]) == ["CTAX"]
    row = df.iloc[0]
    assert row["n_9mers"] == 4  # AAA, AAC, ACC, CCC
    assert row["n_specific_9mers"] == 3  # CCC also in background -> excluded


def test_specific_9mer_counts_caches_with_fingerprint(fake_proteome, tmp_path):
    peptides.cta_specific_9mer_counts(k=3)
    # filename is keyed by release AND a CTA-set fingerprint
    cached = list(tmp_path.glob("cta_specific_3mers_r999_*.csv"))
    assert len(cached) == 1


def test_specific_9mer_counts_returns_fresh_copy(fake_proteome):
    a = peptides.cta_specific_9mer_counts(k=3)
    a.loc[0, "n_specific_9mers"] = -999  # mutating the result must not corrupt the cache
    b = peptides.cta_specific_9mer_counts(k=3)
    assert b.loc[0, "n_specific_9mers"] == 3


def test_specific_9mer_counts_refresh_rebuilds(fake_proteome, tmp_path):
    peptides.cta_specific_9mer_counts(k=3)
    # corrupt the on-disk cache; refresh=True must drop it and rebuild correctly
    (next(tmp_path.glob("cta_specific_3mers_r999_*.csv"))).write_text("garbage\n")
    peptides._COUNTS_CACHE.clear()
    df = peptides.cta_specific_9mer_counts(k=3, refresh=True)
    assert int(df.loc[0, "n_specific_9mers"]) == 3


def test_specific_9mer_weights_keyed_by_proteoform_key_by_default(fake_proteome):
    # Default key is the proteoform_key; for the singleton ENSG_CTA that is its ENSG.
    assert peptides.cta_specific_9mer_weights(k=3) == {"ENSG_CTA": 3}
    assert peptides.cta_specific_9mer_weights(k=3, by="ensembl_gene_id") == {"ENSG_CTA": 3}
    assert peptides.cta_specific_9mer_weights(k=3, by="symbol") == {"CTAX": 3}
    with pytest.raises(ValueError, match="by must be"):
        peptides.cta_specific_9mer_weights(k=3, by="nonsense")


def test_weight_by_proteoform_key_covers_group_via_any_member(monkeypatch):
    # A real CGB3/5/8 group whose canonical min-ENSG (CGB3) is unexpressed: the counts
    # table has only the expressed member (CGB8), yet keying by proteoform_key gives the
    # group its weight — no canonical-member ambiguity.
    import pandas as pd

    fake_counts = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000213030"],  # CGB8 (expressed member of CGB3/5/8)
            "Symbol": ["CGB8"],
            "n_9mers": [40],
            "n_specific_9mers": [31],
        }
    )
    monkeypatch.setattr(
        peptides, "cta_specific_9mer_counts", lambda *, k=peptides.DEFAULT_K: fake_counts
    )
    weights = peptides.cta_specific_9mer_weights(by="proteoform_key")
    assert weights == {"CGB3/5/8": 31}


def test_specific_9mer_load_joins_on_proteoform_key(fake_proteome, monkeypatch):
    # The load joins on proteoform_key (the uniform key), NOT Symbol/ENSG: a stub whose
    # Symbol/ENSG don't match the weight key still resolves via proteoform_key.
    from oncodata import coverage

    def fake_fractions(code, *, threshold_tpm):
        return pd.DataFrame(
            {
                "proteoform_key": ["ENSG_CTA"],  # the weight key (singleton -> ENSG)
                "Ensembl_Gene_ID": ["ENSG_CTA"],
                "Symbol": ["SOMETHING_ELSE"],  # deliberately not the join key
                "fraction_expressing": [0.5],
            }
        )

    monkeypatch.setattr(coverage, "cta_patient_fractions", fake_fractions)
    # load = fraction (0.5) * n_specific_9mers (3) = 1.5
    assert peptides.cta_specific_9mer_load("X", threshold_tpm=5, k=3) == pytest.approx(1.5)


def test_specific_9mer_load_empty_cohort(fake_proteome, monkeypatch):
    from oncodata import coverage

    monkeypatch.setattr(coverage, "cta_patient_fractions", lambda code, **k: pd.DataFrame())
    assert peptides.cta_specific_9mer_load("X", k=3) == 0.0
