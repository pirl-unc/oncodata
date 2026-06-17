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

"""Tumor mutational burden (TMB) reference data, keyed by cancer-type code."""

from __future__ import annotations

from .cancer_types import cancer_type_registry, resolve_cancer_type
from .load_dataset import get_data


def cancer_tmb_df():
    """Return the curated ``cancer-tmb.csv`` reference: median tumor mutational
    burden (mut/Mb) per cancer-type code, with a per-row published source/PMID
    and a confidence flag.

    Cohorts with no defensible published per-Mb median are present with a blank
    ``median_tmb_mut_mb`` (and a ``confidence`` of ``none``) so the gap is
    explicit rather than silently absent. Values mix WES-anchored medians
    (Lawrence 2013) with panel-based medians (Chalmers 2017) and disease-specific
    studies; see the ``source``/``notes`` columns — panel and WES TMB are not
    strictly comparable in the low-TMB range."""
    return get_data("cancer-tmb")


def cancer_tmb(cancer_type=None, *, inherit=True):
    """Median TMB (mut/Mb) for one cancer type, or the whole
    ``{code: median_tmb}`` map (codes with no published value omitted).

    ``cancer_type`` is resolved through :func:`resolve_cancer_type`, so aliases
    and display names work. When ``inherit`` (default), a code with no curated
    value of its own inherits its nearest ancestor's TMB by walking the registry
    ``parent_code`` chain — so molecular / histology subtypes (``LUAD_EGFR`` ->
    ``LUAD``, ``SCLC_ASCL1`` -> ``SCLC``, rare ``SARC_*`` -> ``SARC``) resolve
    without a curated row each. Returns ``None`` if neither the code nor any
    ancestor has a value."""
    df = cancer_tmb_df()
    vals = df.dropna(subset=["median_tmb_mut_mb"])
    mapping = dict(zip(vals["cancer_code"].astype(str), vals["median_tmb_mut_mb"].astype(float)))
    if cancer_type is None:
        return mapping
    code = resolve_cancer_type(cancer_type)
    if code in mapping or not inherit:
        return mapping.get(code)
    # walk the registry parent chain to inherit an ancestor's value
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


def cancer_frameshift_burden_df():
    """Return the curated ``cancer-frameshift-burden.csv`` reference: per-type
    frameshift-indel burden (``cancer_code``, ``indel_class``, ``indel_score``,
    ``basis``, ``pmid_doi``, ``confidence``, ``notes``). A complement to TMB —
    frameshift indels yield disproportionately many high-affinity neoantigens."""
    return get_data("cancer-frameshift-burden")


def cancer_frameshift_burden(cancer_type=None, *, inherit=True):
    """Frameshift-indel ``indel_score`` for a cancer type (alias-resolved), or the
    full ``{code: score}`` map when ``cancer_type`` is None. Subtypes inherit a
    parent's score if unmapped (mirrors :func:`cancer_tmb`)."""
    df = cancer_frameshift_burden_df()
    scores = {
        str(c): int(s)
        for c, s in zip(df["cancer_code"], df["indel_score"])
        if str(s).strip() and str(s).strip().lower() != "nan"
    }
    if cancer_type is None:
        return scores
    code = resolve_cancer_type(cancer_type)
    if code in scores:
        return scores[code]
    if inherit:
        parent_of = cancer_type_registry().set_index("code")["parent_code"].to_dict()
        cur = code
        seen = set()
        while cur in parent_of and isinstance(parent_of[cur], str) and parent_of[cur]:
            cur = parent_of[cur]
            if cur in seen:
                break
            seen.add(cur)
            if cur in scores:
                return scores[cur]
    return None
