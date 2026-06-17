# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Locks for curated values that drifted in upstream pirlygenes after the
initial copy (Literature-audit corrections). These guard against silently
shipping stale reference data.
"""

from oncoref import cancer_burden, cancer_tmb, cancer_tmb_df


def test_tmb_has_n_samples_column():
    # Added by the upstream literature audit alongside per-row sample counts.
    assert "n_samples" in cancer_tmb_df().columns


def test_tmb_filled_gaps():
    # MTC and CRANIO had no curated median in the initial copy; the audit added
    # cited values. Assert the gaps are *filled* (a positive curated value), not
    # the exact numbers — those can be re-curated.
    for code in ("MTC", "CRANIO"):
        value = cancer_tmb(code, inherit=False)
        assert value is not None and value > 0


def test_tmb_new_entities_present():
    codes = set(cancer_tmb_df()["cancer_code"].astype(str))
    for code in ("HCL", "ACINIC", "ALCL", "URETH"):
        assert code in codes


def test_incidence_corrections_applied():
    # GLOBOCAN2022 world-incidence corrections: liver was a stale 6.0, thyroid a
    # stale 2.5 before the audit. Assert they're no longer the stale values and
    # remain plausible shares — robust to a future re-curation of the exact %.
    liver = cancer_burden("liver", metric="world_incidence_pct")
    thyroid = cancer_burden("thyroid", metric="world_incidence_pct")
    assert liver != 6.0 and 0 < liver < 10
    assert thyroid != 2.5 and 0 < thyroid < 10
