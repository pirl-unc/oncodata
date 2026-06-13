# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from cancerdata import expression_engine as ee


def test_aggregate_sums_transcripts_per_gene():
    df = pd.DataFrame(
        {
            "transcript_id": ["ENST1.2", "ENST2", "ENST3", "ENSTX"],
            "tpm": [10.0, 5.0, 20.0, 7.0],
        }
    )
    tx_map = {"ENST1": "GENEA", "ENST2": "GENEA", "ENST3": "GENEB"}  # ENSTX unknown
    out = ee.aggregate_transcripts_to_genes(df, tx_map)
    by = dict(zip(out["gene"], out["TPM"]))
    assert by["GENEA"] == 15.0  # ENST1(versioned) + ENST2
    assert by["GENEB"] == 20.0
    assert by["unresolved"] == 7.0  # ENSTX kept, not dropped
    stats = out.attrs["aggregation_stats"]
    assert stats["unresolved_tpm"] == 7.0
    assert stats["n_genes"] == 2
    assert stats["unresolved_fraction"] == pytest.approx(7.0 / 42.0)


def test_find_column_absorbs_naming():
    df = pd.DataFrame({"Target_ID": ["t"], "TPM": [1.0]})
    assert ee.find_column(df, ["transcript", "target_id"], "tx") == "Target_ID"
    assert ee.find_column(df, ["tpm"], "TPM") == "TPM"
    with pytest.raises(ValueError, match="no column for"):
        ee.find_column(df, ["nope"], "missing")


def test_expanded_tx_map_versionless():
    m = ee.expanded_tx_map({"ENST9.3": "G"})
    assert m["ENST9.3"] == "G"
    assert m["ENST9"] == "G"  # versionless key added


def test_default_map_is_cancerdata_extra_tx():
    # The default map comes from cancerdata's curated extra-tx-mappings.
    df = pd.DataFrame({"transcript_id": ["ENST00000264036"], "tpm": [99.0]})
    out = ee.aggregate_transcripts_to_genes(df)
    assert "MCAM" in set(out["gene"])  # ENST00000264036 -> MCAM in extra-tx-mappings
