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

"""Characteristic gene fusions / oncogenic translocations per cancer type.

The curated ``cancer-fusions.csv`` reference: one row per concrete 5'/3' gene
pairing, with the partner protein families annotated (FET, ETS, PAX, FOX, MiT/TFE,
RTK, Ig locus, …), an ``is_defining`` flag for the characteristic lesion of the
entity, and the stronger ``pathognomonic`` flag for pairs that map to a single
entity. This module is the read + query surface: per-type lookup, reverse lookup
(which types carry a fusion / partner / partner family), and partner sets.
"""

from __future__ import annotations

from .cancer_types import cancer_type_descendants, resolve_cancer_type
from .load_dataset import get_data


def cancer_fusions_df():
    """The full curated fusion table (defensive copy).

    Columns: ``cancer_code``, ``fusion_family``, ``gene_5prime``,
    ``gene_5prime_family``, ``gene_3prime``, ``gene_3prime_family``, ``frequency``,
    ``is_defining``, ``pathognomonic``, ``rnaseq_detectable``, ``mechanism``,
    ``confidence``, ``pmid``, ``notes``. Fusion-negative entities carry a single
    ``fusion_family="(none)"`` row naming the real driver.
    """
    return get_data("cancer-fusions").copy()


def _truthy(series):
    # is_defining / pathognomonic load as bool, but compare via lowercased string
    # so the filter is robust whether the column is bool or str.
    return series.astype(str).str.lower() == "true"


def _upper(series):
    """Uppercased view of a gene/family column with NaN preserved as non-matching.

    The fusion-negative ``"(none)"`` rows carry NaN gene names; ``.str.upper()``
    keeps those NaN (so ``NaN == "X"`` is False) instead of ``.astype(str)``
    turning them into the literal string ``"NAN"`` that a query could match.
    """
    return series.str.upper()


def cancer_fusions(
    cancer_type=None, *, defining_only=False, pathognomonic_only=False, include_subtypes=False
):
    """Fusion rows for one cancer type (alias-resolved), or the whole table when
    ``cancer_type`` is None.

    ``defining_only`` / ``pathognomonic_only`` filter to the characteristic /
    diagnostic rows. ``include_subtypes`` also pulls the fusions of every code in
    the cancer type's subtree (via :func:`cancer_type_descendants`) — e.g. the
    union over an ``SARC_RMS`` parent once its subtypes are parented under it.
    """
    df = cancer_fusions_df()
    if cancer_type is not None:
        code = resolve_cancer_type(cancer_type, strict=False) or cancer_type
        codes = {code}
        if include_subtypes:
            codes |= set(cancer_type_descendants(code))
        df = df[df["cancer_code"].astype(str).isin(codes)]
    if defining_only:
        df = df[_truthy(df["is_defining"])]
    if pathognomonic_only:
        df = df[_truthy(df["pathognomonic"])]
    return df.reset_index(drop=True)


def fusion_partners(gene, *, side=None):
    """The set of fusion partners of ``gene`` in the table.

    ``side=None`` returns partners on either end; ``side="5prime"`` the observed
    3' partners when ``gene`` is 5'; ``side="3prime"`` the observed 5' partners.
    """
    if side not in (None, "5prime", "3prime"):
        raise ValueError("side must be None, '5prime', or '3prime'")
    g = str(gene).strip().upper()
    df = cancer_fusions_df()
    out = set()
    if side in (None, "5prime"):
        out |= set(df.loc[_upper(df["gene_5prime"]) == g, "gene_3prime"])
    if side in (None, "3prime"):
        out |= set(df.loc[_upper(df["gene_3prime"]) == g, "gene_5prime"])
    return {p for p in out if isinstance(p, str) and p.strip()}


def cancer_types_with_fusion(
    fusion=None, *, partner=None, partner_family=None, defining_only=False, as_rows=False
):
    """Reverse fusion lookup — the inverse of :func:`cancer_fusions`.

    Pass exactly one of:
    - ``fusion="EWSR1-FLI1"`` — a directional ``5'-3'`` string (``::`` also accepted);
    - ``partner="EWSR1"`` — a partner gene on either end;
    - ``partner_family="FET"`` — a partner-family tag.

    ``defining_only`` restricts to is_defining rows. Returns sorted cancer codes,
    or the matching rows when ``as_rows=True``.
    """
    given = [x for x in (fusion, partner, partner_family) if x is not None]
    if len(given) != 1:
        raise ValueError("pass exactly one of fusion=, partner=, or partner_family=")
    df = cancer_fusions(defining_only=defining_only)
    g5 = _upper(df["gene_5prime"])
    g3 = _upper(df["gene_3prime"])
    if fusion is not None:
        parts = str(fusion).upper().replace("::", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"fusion must look like '5GENE-3GENE'; got {fusion!r}")
        a, b = (p.strip() for p in parts)
        mask = (g5 == a) & (g3 == b)
    elif partner is not None:
        p = str(partner).strip().upper()
        mask = (g5 == p) | (g3 == p)
    else:
        fam = str(partner_family).strip().upper()
        f5 = _upper(df["gene_5prime_family"])
        f3 = _upper(df["gene_3prime_family"])
        mask = (f5 == fam) | (f3 == fam)
    mask = mask.fillna(False)
    hits = df[mask]
    if as_rows:
        return hits.reset_index(drop=True)
    return sorted({str(c) for c in hits["cancer_code"] if str(c).strip()})


def protein_family(gene):
    """Protein/gene family of a fusion partner (EWSR1→FET, FLI1→ETS, PAX3→PAX,
    FOXO1→FOX, ALK→RTK), or ``None`` if the gene has no family annotation."""
    g = str(gene).strip().upper()
    df = cancer_fusions_df()
    for col, fam in (("gene_5prime", "gene_5prime_family"), ("gene_3prime", "gene_3prime_family")):
        hit = df.loc[_upper(df[col]) == g, fam]
        for v in hit:
            if isinstance(v, str) and v.strip():
                return v
    return None
