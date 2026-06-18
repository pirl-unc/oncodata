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

"""Immune-checkpoint-inhibitor (ICI) response (ORR) by cancer type and regimen.

Generalizes the anti-PD-1 layer (:mod:`oncoref.apd1`) to **all three checkpoint
regimens, each kept as a distinct source of response data**:

- ``"PD-1"``        — anti-PD-1 monotherapy (pembrolizumab / nivolumab / cemiplimab)
- ``"PD-L1"``       — anti-PD-L1 monotherapy (atezolizumab / durvalumab / avelumab)
- ``"PD-1+CTLA-4"`` — anti-PD-1 + anti-CTLA-4 combination (nivolumab + ipilimumab)

Unlike the representative one-row-per-cancer ``cancer-apd1-response.csv``, the ICI
table (``cancer-ici-response.csv``) is a **long table**: a cancer type can carry a
value for more than one regimen (e.g. melanoma under both anti-PD-1 mono and the
ipi+nivo doublet), so regimens can be compared within a cancer. Every value is a
representative ORR anchor from a pivotal trial, with a citation — not an exact
reproducible constant (it shifts with data cutoff, line of therapy, and PD-L1 / MSI
selection; the ``setting`` and ``notes`` columns record that context).

:func:`cancer_ici_response` exposes the **fallback resolution** the analysis layer
usually wants — prefer anti-PD-1 monotherapy, fall back to anti-PD-L1 where that is
missing, then to the combination — via the default :data:`REGIMEN_FALLBACK` order;
pass ``regimen=`` to pin a single regimen instead, or ``fallback=False`` to get the
full per-regimen mapping.

Curation & pooling criteria
---------------------------
The wider evidence base (``cancer-ici-response-estimates.csv``, exposed by
:func:`cancer_ici_response_estimates_df` and pooled by :func:`pooled_ici_response`)
follows three rules that matter when curating new trials or interpreting a pooled value:

1. **Reported vs. derived values.** Most estimate rows are ORR/CRR/DCR/… *directly
   reported* in the cited paper or trial-results record. A handful of anchors in ``cancer-ici-response.csv`` are
   instead **curator-derived blends** — no single trial reports them. The clearest case is
   the "all-comer" ORR for MSI/MMR-dependent cancers: ``READ`` 5%, ``COAD`` 5%, ``UCEC``
   8% are *prevalence-weighted blends* of the MSI-H/dMMR responders (~45–50%) and the
   MSS/pMMR non-responders (~0%), because the pivotal trials (KEYNOTE-177/-164/-158) enroll
   ONLY the biomarker-selected subtype. Since no paper reports the blend, the reference
   audit marks these ``citation_matches=False`` → ``source_verified=False`` (flags
   ``orr-is-derived-estimate`` / ``orr-is-derived-blend`` / ``all-comer-figure-not-in-cited-paper``).
   In the estimates table these carry ``value_basis="derived_blend"`` (vs ``"reported"``),
   and :func:`pooled_ici_response` drops them unconditionally — a derived blend must never
   be pooled as if it were trial data. Rows with ``value_basis="reported_context"`` are
   source-reported values from overlapping or comparator cohorts kept for audit context,
   and are also excluded from subtype-specific pooling. The blend is reconstructable from
   its components:
   ``all_comer ≈ ORR_MSI · p_dMMR + ORR_MSS · (1 − p_dMMR)`` (COAD: 43.8·0.13 ≈ 5.7%;
   READ: 43.8·0.07 ≈ 3.1%; UCEC: 48·0.20 + 7·0.80 ≈ 15%, using the KEYNOTE-158 dMMR/pMMR
   cohorts at an advanced-EC dMMR prevalence ~20%). When adding such an anchor,
   keep the reported subtype values in the ``<code>_MSI`` / ``<code>_MSS`` rows and record
   the prevalence weighting in ``notes`` — never cite a paper that does not contain the
   blended number.

2. **Never double-count patients.** A single trial routinely reports an all-comer cohort
   AND its own biomarker subgroup (e.g. ``BLCA`` KEYNOTE-052 all-comers + the CPS≥10 subset;
   ``LUAD`` all-comers + PD-L1≥50%). Those share patients, so summing their denominators
   inflates ``n``. Each estimate row therefore carries a ``role``: ``"primary"`` (the one
   representative cited setting) or ``"alternate"`` (other trials/subgroups). Pool only
   rows describing the **same population and line of therapy**, and never an all-comer
   cohort together with a subgroup drawn from it. ``pooled_ici_response`` does *not*
   auto-dedupe overlapping subgroups — it returns the full ``sources`` list and
   ``value_range`` so the overlap stays visible; ``include_alternates=False`` restricts
   the pool to ``primary`` rows (one per cancer+regimen), which can never overlap.

3. **Comparability.** ORR shifts with line of therapy, PD-L1/MSI selection, and data
   cutoff; medians (PFS/OS/DOR) cannot be pooled at all without patient-level data. Treat
   ``value_range`` as a heterogeneity check before trusting a single pooled number.
"""

from __future__ import annotations

import math
from functools import lru_cache

from .cancer_types import cancer_type_registry, resolve_cancer_type
from .load_dataset import _register_derived_cache, get_data

#: Response-proportion endpoints that can be responder-weighted-pooled (each needs a
#: responder count and a denominator n). Time-to-event medians and landmark rates are
#: deliberately excluded — see :func:`pooled_ici_response`.
PROPORTION_METRICS: tuple[str, ...] = ("ORR", "CRR", "DCR", "PR")

#: Values that are useful audit context but should not enter pooled estimates.
NON_POOLABLE_VALUE_BASIS: frozenset[str] = frozenset({"derived_blend", "reported_context"})

#: Regimen tags in preference order — the default fallback when no regimen is pinned:
#: anti-PD-1 monotherapy first, then anti-PD-L1, then the anti-PD-1+anti-CTLA-4 doublet.
REGIMEN_FALLBACK: tuple[str, ...] = ("PD-1", "PD-L1", "PD-1+CTLA-4")

#: Human-readable label for each regimen tag.
REGIMEN_LABELS = {
    "PD-1": "anti-PD-1 monotherapy",
    "PD-L1": "anti-PD-L1 monotherapy",
    "PD-1+CTLA-4": "anti-PD-1 + anti-CTLA-4",
}


def cancer_ici_response_df():
    """The curated ``cancer-ici-response.csv`` long table: one row per
    (``cancer_code``, ``regimen``) with the representative ORR (%), drug, pivotal trial
    (split into ``trial_name`` / ``trial_alias`` / ``trial_nct`` — the acronym, the
    distinct protocol/sponsor code if any, and the ClinicalTrials.gov id), setting,
    source PMID/DOI, and confidence. A cancer type may appear under several regimens."""
    return get_data("cancer-ici-response")


def ici_regimens() -> tuple[str, ...]:
    """The regimen tags, in fallback-preference order."""
    return REGIMEN_FALLBACK


@lru_cache(maxsize=1)
def _regimen_maps() -> dict[str, dict[str, float]]:
    """``{regimen: {cancer_code: orr_pct}}`` from the curated table. Cached; callers
    must treat the result as read-only (copy before mutating)."""
    df = cancer_ici_response_df().dropna(subset=["orr_pct"])
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMEN_FALLBACK}
    for code, regimen, orr in zip(df["cancer_code"], df["regimen"], df["orr_pct"]):
        out.setdefault(str(regimen), {})[str(code)] = float(orr)
    return out


_register_derived_cache(_regimen_maps.cache_clear)


def _resolve_with_fallback(code: str, maps: dict[str, dict[str, float]], order):
    for regimen in order:
        if code in maps.get(regimen, {}):
            return maps[regimen][code], regimen
    return None, None


def cancer_ici_response(cancer_type=None, *, regimen=None, fallback=True, inherit=True):
    """ICI objective response rate (%) for a cancer type.

    ``regimen`` pins one of :data:`REGIMEN_FALLBACK` (``"PD-1"`` / ``"PD-L1"`` /
    ``"PD-1+CTLA-4"``); leave it ``None`` to resolve across regimens.

    With ``regimen=None`` and ``fallback=True`` (default), the value is taken from the
    first regimen present in :data:`REGIMEN_FALLBACK` order (anti-PD-1 → anti-PD-L1 →
    combination) — the "best-available" anchor. With ``fallback=False`` the per-regimen
    mapping ``{regimen: orr}`` is returned instead.

    ``cancer_type`` is resolved via :func:`resolve_cancer_type`. With ``inherit``
    (default) a code with no row of its own inherits its nearest ancestor's value via
    the registry ``parent_code`` chain. Returns ``None`` (or ``{}``) when nothing is
    found.

    With ``cancer_type=None`` returns the whole ``{code: orr}`` map under the same
    resolution (a single regimen if pinned, else the fallback pick) — ready as a
    per-cancer plotting axis.
    """
    maps = _regimen_maps()
    order = (regimen,) if regimen is not None else REGIMEN_FALLBACK

    if cancer_type is None:
        if regimen is not None:
            return dict(maps.get(regimen, {}))
        codes = {c for m in maps.values() for c in m}
        if not fallback:
            # Per-regimen mapping for every covered cancer: {code: {regimen: orr}}.
            return {c: {r: maps[r][c] for r in REGIMEN_FALLBACK if c in maps[r]} for c in codes}
        # Fallback pick per cancer across the union of covered codes.
        out = {}
        for c in codes:
            val, _ = _resolve_with_fallback(c, maps, REGIMEN_FALLBACK)
            if val is not None:
                out[c] = val
        return out

    code = resolve_cancer_type(cancer_type)

    if regimen is None and not fallback:
        per = {r: maps[r][code] for r in REGIMEN_FALLBACK if code in maps.get(r, {})}
        if per or not inherit:
            return per
        # walk ancestors for a per-regimen mapping
        reg = cancer_type_registry().set_index("code")
        cur, seen = code, set()
        while cur and cur not in seen:
            seen.add(cur)
            hit = {r: maps[r][cur] for r in REGIMEN_FALLBACK if cur in maps.get(r, {})}
            if hit:
                return hit
            if cur not in reg.index:
                break
            cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
        return {}

    val, _ = _resolve_with_fallback(code, maps, order)
    if val is not None or not inherit:
        return val
    reg = cancer_type_registry().set_index("code")
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        val, _ = _resolve_with_fallback(cur, maps, order)
        if val is not None:
            return val
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
    return None


def cancer_ici_regimen(cancer_type):
    """The regimen tag (``"PD-1"`` / ``"PD-L1"`` / ``"PD-1+CTLA-4"``) the fallback
    resolution selects for a cancer type — i.e. *which source* its
    :func:`cancer_ici_response` value comes from. ``None`` if no row (no inheritance)."""
    code = resolve_cancer_type(cancer_type)
    _, regimen = _resolve_with_fallback(code, _regimen_maps(), REGIMEN_FALLBACK)
    return regimen


# --------------------------------------------------------------------------------------
# Multi-endpoint estimates + pooling
#
# ``cancer-ici-response.csv`` carries ONE representative ORR anchor per (cancer, regimen).
# ``cancer-ici-response-estimates.csv`` is the wider evidence base behind it: every
# endpoint (ORR/CRR/DCR/DOR/PFS/OS + landmark rates) from every trial-source for that
# cell, with CIs and n, produced by the reference audit. The pooling helper combines
# those sources into a single responder-weighted estimate with a Wilson CI.
# --------------------------------------------------------------------------------------


def cancer_ici_response_estimates_df():
    """The verified ``cancer-ici-response-estimates.csv`` long table: one row per
    (``cancer_code``, ``regimen``, trial-source, ``metric``).

    Generalizes the one-anchor-per-cell :func:`cancer_ici_response_df` to *all* extracted
    endpoints — ORR, CRR, DCR, DOR, PFS, OS and landmark PFS/OS rates — each with
    ``value``, ``unit`` (``percent`` / ``months`` / ``rate_percent``), 95% CI
    (``ci_low`` / ``ci_high``), ``timepoint``, sample size (``metric_n`` / ``source_n``)
    and ``responders``. ``role`` is ``"primary"`` (the cited representative setting) or
    ``"alternate"`` (other trials / subgroups for the same cancer + regimen).
    ``source_verified`` marks rows whose citation was confirmed against PubMed/Crossref or
    a ClinicalTrials.gov results record in the reference audit. ``value_basis`` is
    ``"reported"`` (value reported in the cited trial source), ``"reported_context"``
    (source-reported overlapping or comparator cohort kept for audit context but excluded
    from pooling), or ``"derived_blend"`` (a curator-computed prevalence-weighted blend,
    e.g. the all-comer MMR-dependent ORRs — :func:`pooled_ici_response` never pools
    these)."""
    return get_data("cancer-ici-response-estimates")


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _wilson_ci(k: float, n: float, z: float = 1.96):
    """95% Wilson score interval (returned as percentages) for ``k`` responders of ``n``."""
    if not n:
        return (None, None)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * (center - half), 1), round(100 * (center + half), 1))


def pooled_ici_response(
    cancer_type,
    *,
    regimen=None,
    metric="ORR",
    verified_only=True,
    include_alternates=True,
):
    """Pool every audited estimate for one cancer + regimen + endpoint.

    Returns a dict::

        {cancer_code, regimen, metric, poolable, pooled_pct, ci_low, ci_high,
         responders_total, n_total, n_studies, n_pooled, refs, value_range, sources}

    For **proportion endpoints** (:data:`PROPORTION_METRICS` — ORR / CRR / DCR / PR) the
    pool is *responder-weighted*: ``pooled_pct = 100 · Σresponders / Σn`` over the sources
    that report both, with a 95% Wilson score CI. ``n_total`` is the summed sample size and
    ``responders_total`` the summed responders. ``n_studies`` is the number of trial-sources
    found for the cell (the full evidence count); ``n_pooled`` is how many of them actually
    entered the responder-weighted pool (``None`` for non-proportion endpoints). ``refs``
    lists the citations behind the reported estimate.

    For **time-to-event endpoints** (median PFS/OS/DOR in months) and **landmark rates**,
    medians/rates cannot be validly pooled without patient-level data — ``poolable`` is
    ``False``, ``pooled_pct`` is ``None``, and only the per-trial ``sources`` and their
    ``value_range`` are returned.

    Setting heterogeneity is real (all-comer vs PD-L1/MSI-selected vs different lines).
    By default the pool includes the cited primary setting *and* the ``alternate`` rows;
    pass ``include_alternates=False`` to pool only the representative primary setting, or
    inspect each source's ``setting`` in ``sources`` to judge comparability. The
    per-source breakdown and ``value_range`` are always returned so heterogeneity (and
    any overlapping subgroups) stays visible. ``verified_only`` (default) keeps only
    audit-confirmed citations.
    """
    code = resolve_cancer_type(cancer_type)
    metric = str(metric).upper()
    df = cancer_ici_response_estimates_df()
    sub = df[(df["cancer_code"] == code) & (df["metric"].astype(str).str.upper() == metric)]
    if regimen is not None:
        sub = sub[sub["regimen"] == regimen]
    # Derived blends and contextual comparator/overlapping rows are audit evidence, not
    # subtype-specific pool inputs.
    if "value_basis" in sub.columns:
        basis = sub["value_basis"].astype(str)
        sub = sub[~basis.isin(NON_POOLABLE_VALUE_BASIS)]
    if verified_only:
        sub = sub[sub["source_verified"].map(_truthy)]
    if not include_alternates:
        sub = sub[sub["role"] == "primary"]

    def _num(v):
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    sources, seen = [], set()
    for _, r in sub.iterrows():
        ref = None if r.get("ref") is None else str(r.get("ref"))
        dedupe = (ref, str(r.get("trial_name")), str(r.get("setting")), _num(r.get("value")))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        sources.append(
            {
                "role": r.get("role"),
                "drug": r.get("drug"),
                "trial_name": r.get("trial_name"),
                "trial_alias": r.get("trial_alias"),
                "trial_nct": r.get("trial_nct"),
                "ref": ref,
                "setting": r.get("setting"),
                "value": _num(r.get("value")),
                "unit": r.get("unit"),
                "ci_low": _num(r.get("ci_low")),
                "ci_high": _num(r.get("ci_high")),
                "n": _num(r.get("metric_n")),
                "responders": _num(r.get("responders")),
            }
        )

    values = [s["value"] for s in sources if s["value"] is not None]
    # Sources that actually enter the responder-weighted pool (need responders + n);
    # empty for non-proportion endpoints, which are not pooled.
    contrib = (
        [s for s in sources if s["responders"] is not None and s["n"]]
        if metric in PROPORTION_METRICS
        else []
    )
    # `refs` = the citations behind the *reported* estimate: the pooled sources when a
    # pool is produced, else every source feeding the value_range.
    ref_sources = contrib if contrib else sources
    result = {
        "cancer_code": code,
        "regimen": regimen,
        "metric": metric,
        "poolable": metric in PROPORTION_METRICS,
        "pooled_pct": None,
        "ci_low": None,
        "ci_high": None,
        "responders_total": None,
        "n_total": None,
        "n_studies": len(sources),  # trial-sources found for this cell + metric
        "n_pooled": len(contrib) if metric in PROPORTION_METRICS else None,  # entered the pool
        "refs": sorted({s["ref"] for s in ref_sources if s["ref"]}),
        "value_range": (min(values), max(values)) if values else None,
        "sources": sources,
    }

    if contrib:
        k = sum(s["responders"] for s in contrib)
        n = sum(s["n"] for s in contrib)
        if n:
            lo, hi = _wilson_ci(k, n)
            result.update(
                pooled_pct=round(100 * k / n, 1),
                ci_low=lo,
                ci_high=hi,
                responders_total=int(k),
                n_total=int(n),
            )
    return result
