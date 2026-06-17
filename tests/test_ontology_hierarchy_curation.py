# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""NET umbrella + SARC_RMS sub-parenting — the curated hierarchy now that
oncoref owns the registry (#27, O1.5)."""

from oncoref import (
    cancer_fusions,
    cancer_type_descendants,
    cancer_type_lineage,
    cancer_type_subtypes_of,
    cohort_aggregate_members,
)


def test_net_umbrella_navigation():
    # The headline example: NET -> NET_PANCREAS ("NET_PAN").
    assert cancer_type_lineage("NET_PANCREAS") == ["NET", "NET_PANCREAS"]
    assert set(cancer_type_descendants("NET")) == {
        "NET_PANCREAS",
        "NET_MIDGUT",
        "NET_RECTAL",
        "NET_LUNG",
    }


def test_net_excludes_carcinomas_and_pituitary():
    # Poorly-differentiated NEC_* and pituitary PITNET are distinct, not children.
    desc = set(cancer_type_descendants("NET"))
    assert "NEC_MERKEL" not in desc
    assert "NEC_LUNG_LARGECELL" not in desc
    assert "PITNET" not in desc


def test_net_cohort_aggregate():
    assert set(cohort_aggregate_members("NET")) == {
        "NET_PANCREAS",
        "NET_MIDGUT",
        "NET_RECTAL",
        "NET_LUNG",
    }


def test_rms_subtypes_reparented_under_sarc_rms():
    assert set(cancer_type_subtypes_of("SARC_RMS")) == {
        "SARC_RMS_ERMS",
        "SARC_RMS_ARMS",
        "SARC_RMS_PRMS",
        "SARC_RMS_SSRMS",
    }


def test_sarc_rms_fusion_rollup():
    # "Characteristic driver fusions of all SARC_RMS subtypes" now rolls up via the tree.
    roll = cancer_fusions("SARC_RMS", include_subtypes=True)
    pairs = {(a, b) for a, b in zip(roll["gene_5prime"], roll["gene_3prime"])}
    assert ("PAX3", "FOXO1") in pairs


def test_pan_sarcoma_still_pools_rms_atoms():
    # Re-parenting must not change the pan-sarcoma grand union (computed by family).
    sarc = cohort_aggregate_members("SARC")
    assert "SARC_RMS_ARMS" in sarc and "SARC_RMS_ERMS" in sarc
