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

Generalizes the anti-PD-1 layer (:mod:`cancerdata.apd1`) to **all three checkpoint
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
"""

from __future__ import annotations

from .cancer_types import cancer_type_registry, resolve_cancer_type
from .load_dataset import get_data

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
    (``cancer_code``, ``regimen``) with the representative ORR (%), drug, pivotal
    trial, setting, source PMID/DOI, and confidence. A cancer type may appear under
    several regimens."""
    return get_data("cancer-ici-response")


def ici_regimens() -> tuple[str, ...]:
    """The regimen tags, in fallback-preference order."""
    return REGIMEN_FALLBACK


def _regimen_maps() -> dict[str, dict[str, float]]:
    """``{regimen: {cancer_code: orr_pct}}`` from the curated table."""
    df = cancer_ici_response_df().dropna(subset=["orr_pct"])
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMEN_FALLBACK}
    for code, regimen, orr in zip(df["cancer_code"], df["regimen"], df["orr_pct"]):
        out.setdefault(str(regimen), {})[str(code)] = float(orr)
    return out


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
        # Fallback pick per cancer across the union of covered codes.
        codes = {c for m in maps.values() for c in m}
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
