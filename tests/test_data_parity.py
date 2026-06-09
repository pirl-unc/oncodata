# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Locks for curated values that drifted in upstream pirlygenes after the
initial copy (Literature-audit corrections). These guard against silently
shipping stale reference data.
"""

from cancerdata import cancer_burden, cancer_tmb, cancer_tmb_df


def test_tmb_has_n_samples_column():
    # Added by the upstream literature audit alongside per-row sample counts.
    assert "n_samples" in cancer_tmb_df().columns


def test_tmb_filled_gaps():
    # MTC and CRANIO had no curated median in the initial copy; the audit added
    # cited values. Inheritance is off here — these are direct curated values.
    assert cancer_tmb("MTC", inherit=False) == 1.0
    assert cancer_tmb("CRANIO", inherit=False) == 0.9


def test_tmb_new_entities_present():
    codes = set(cancer_tmb_df()["cancer_code"].astype(str))
    for code in ("HCL", "ACINIC", "ALCL", "URETH"):
        assert code in codes


def test_incidence_corrections_applied():
    # GLOBOCAN2022 world-incidence corrections from the audit.
    assert cancer_burden("liver", metric="world_incidence_pct") == 4.3
    assert cancer_burden("thyroid", metric="world_incidence_pct") == 4.1
