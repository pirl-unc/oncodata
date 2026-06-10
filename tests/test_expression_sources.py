# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from cancerdata import expression_registry as es
from cancerdata import samples


def test_registry_loads_all_sources():
    srcs = es.expression_sources()
    assert len(srcs) > 50
    # every source has an id, a type, and at least one cancer code
    assert all(s.id and s.source_type and s.cancer_codes for s in srcs)


def test_source_types_cover_the_major_providers():
    types = {s.source_type for s in es.expression_sources()}
    for t in ("gdc", "treehouse-compendium", "recount3", "broad-cllmap"):
        assert t in types


def test_lookup_by_id_and_code():
    src = es.expression_source("mmrf-commpass")
    assert src is not None
    assert "MM" in src.cancer_codes
    assert src.source_type == "gdc"
    assert es.expression_source("not-a-source") is None
    assert any(s.id == "mmrf-commpass" for s in es.sources_for_cancer_code("MM"))


def test_registry_dataframe_shape():
    df = es.registry_dataframe()
    assert {"id", "source_type", "cancer_codes", "citation"} <= set(df.columns)
    assert len(df) == len(es.expression_sources())


def test_sample_manifest_loads():
    df = samples.sample_manifest()
    assert len(df) > 1000
    for col in ("cancer_code", "source_cohort", "sample_id", "included"):
        assert col in df.columns


def test_samples_for_cancer_code_included_only():
    inc = samples.samples_for_cancer_code("BL", included_only=True)
    allrows = samples.samples_for_cancer_code("BL", included_only=False)
    assert len(inc) <= len(allrows)
    assert (inc["included"].astype(str).str.lower() == "true").all()


def test_sample_counts_sum():
    counts = samples.sample_counts_by_cancer_code()
    assert counts.sum() > 1000
    assert (counts > 0).all()
