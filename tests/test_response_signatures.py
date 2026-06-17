# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from oncoref import response_signatures as rs


def test_signature_catalog():
    names = rs.response_signature_names()
    assert {"t_cell_inflamed", "cytotoxic", "antigen_presentation", "tgfb_exclusion"} <= set(names)
    assert "IFNG" in rs.response_signature_genes("t_cell_inflamed")
    assert rs.response_signature_direction("t_cell_inflamed") == "positive"
    assert rs.response_signature_direction("tgfb_exclusion") == "negative"


def test_unknown_signature_raises():
    with pytest.raises(ValueError, match="unknown signature"):
        rs.response_signature_genes("not_a_signature")


def test_signature_score(monkeypatch):
    # Stub cohort_mean_expression so the score is hermetic.
    import oncoref.expression as ex

    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E2", "E3"],
            "Symbol": ["IFNG", "STAT1", "OTHER"],
            "expression": [2.0, 4.0, 100.0],
        }
    )
    monkeypatch.setattr(ex, "cohort_mean_expression", lambda *a, **k: fixture)
    # mean over the signature's genes present (IFNG=2, STAT1=4) -> 3.0; OTHER ignored
    score = rs.signature_score("X", "t_cell_inflamed")
    assert score == 3.0


def test_signature_score_nan_when_no_genes(monkeypatch):
    import oncoref.expression as ex

    empty = pd.DataFrame({"Ensembl_Gene_ID": [], "Symbol": [], "expression": []})
    monkeypatch.setattr(ex, "cohort_mean_expression", lambda *a, **k: empty)
    assert rs.signature_score("X", "cytotoxic") != rs.signature_score("X", "cytotoxic")  # NaN
