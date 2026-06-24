# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anti-PD-1 monotherapy response (objective response rate) by cancer type."""

from __future__ import annotations

from .cancer_types import cancer_evidence_source_code, cancer_type_registry, resolve_cancer_type
from .ici import response_anchor_evidence_df
from .load_dataset import get_data


def cancer_apd1_response_df():
    """Return the curated ``cancer-apd1-response.csv`` reference: representative
    objective response rate (ORR, %) to anti-PD-1 **monotherapy**
    (pembrolizumab / nivolumab) per cancer-type code, with the drug, pivotal
    trial, treatment setting, a published source PMID/DOI, and a confidence flag.

    Intended as a per-cancer-type plotting axis (e.g. TMB vs aPD1 ORR, CTA burden
    vs aPD1 ORR). Values are representative anchors, not exact reproducible
    constants — they shift with data cutoff, line of therapy, and biomarker
    selection (PD-L1 / MSI / MMR); the ``setting`` and ``notes`` columns record
    that context. Evidence/provenance fields are joined from the audited ICI estimates
    table, keyed by ``drug_target`` (``PD-1`` / ``PD-L1`` / ``PD-1+CTLA-4``), so
    non-monotherapy fallback anchors remain explicit."""
    return response_anchor_evidence_df(
        get_data("cancer-apd1-response"),
        value_col="apd1_orr_pct",
        regimen_col="drug_target",
    )


def cancer_apd1_response(cancer_type=None, *, inherit=True):
    """Anti-PD-1 monotherapy ORR (%) for one cancer type, or the whole
    ``{code: orr_pct}`` map. ``cancer_type`` is resolved through
    :func:`resolve_cancer_type`; with ``inherit`` (default) a code with no
    curated row of its own inherits its nearest ancestor's value via the registry
    ``parent_code`` chain. Returns ``None`` if neither the code nor any ancestor
    has a value. Mirrors :func:`oncoref.cancer_tmb`."""
    df = cancer_apd1_response_df()
    vals = df.dropna(subset=["apd1_orr_pct"])
    mapping = dict(zip(vals["cancer_code"].astype(str), vals["apd1_orr_pct"].astype(float)))
    if cancer_type is None:
        return mapping
    code = resolve_cancer_type(cancer_type)
    if code in mapping or not inherit:
        return mapping.get(code)
    source_code = cancer_evidence_source_code(code)
    if source_code != code and source_code in mapping:
        return mapping[source_code]
    reg = cancer_type_registry().set_index("code")
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in mapping:
            return mapping[cur]
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
    return None
