# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-sample curation manifest — incl. alias resolution (#22)."""

from oncodata import sample_manifest, samples_for_cancer_code


def _a_present_code_with_alias():
    # LAML is in the manifest and has the alias 'aml'.
    codes = set(sample_manifest()["cancer_code"].astype(str))
    assert "LAML" in codes
    return "LAML", "aml"


def test_samples_for_cancer_code_resolves_aliases():
    code, alias = _a_present_code_with_alias()
    by_code = samples_for_cancer_code(code)
    by_alias = samples_for_cancer_code(alias)
    assert len(by_code) > 0
    assert by_alias.equals(by_code)  # alias resolves to the same canonical rows


def test_unknown_code_returns_empty_not_error():
    assert samples_for_cancer_code("NOT_A_REAL_CODE").empty


def test_included_only_filters():
    code, _ = _a_present_code_with_alias()
    incl = samples_for_cancer_code(code, included_only=True)
    allrows = samples_for_cancer_code(code, included_only=False)
    assert len(incl) <= len(allrows)
