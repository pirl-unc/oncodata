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

"""Proteoform groups: sets of CGA genes that encode an identical protein.

Some cancer-germline antigens are duplicated to distinct genomic loci that encode
a *byte-identical* protein (CTAG1A/CTAG1B = NY-ESO-1, XAGE1A/XAGE1B, SSX4/SSX4B,
MAGEA2/MAGEA2B, the 12-member CT47A family, …). RNA-seq reads multi-map between
such loci, so each gene's individual TPM under-counts the protein; the
biologically meaningful unit is the *proteoform* — the sum of TPM across member
genes.

This module is the read surface over the curated registry
(``proteoform-groups.csv``, derived offline by
``scripts/generate_proteoform_groups.py`` from pyensembl protein sequences). It
owns *which genes sum together*; the per-sample TPM summation itself lives in
:func:`cancerdata.expression.proteoform_representative_samples` (runtime, over the
shipped medoid samples) and :func:`cancerdata._build.sum_proteoform_tpm` (the
pure build-time core, ready for the offline percentile/within-sample generators
to apply before ranking when proteoform-summed artifacts are added to the
bundle).

Group label convention matches the downstream target-selection layer: the
slash-joined, sorted member symbols (``"SSX4/SSX4B"``).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .load_dataset import get_data

_LABEL_COLUMN = "proteoform_id"
_SYMBOL_COLUMN = "member_symbol"
_GENE_ID_COLUMN = "member_gene_id"

#: Registry scopes -> bundled dataset name. ``cta`` is the shipped default (the
#: focused CGA universe); ``genome`` is the opt-in genome-wide identical-protein
#: grouping (issue #12) — a strict superset whose summation shifts many more
#: genes' expression, so it is offered, not defaulted.
_DATASET_BY_SCOPE = {
    "cta": "proteoform-groups",
    "genome": "proteoform-groups-genome",
}


def _dataset_for_scope(scope: str) -> str:
    try:
        return _DATASET_BY_SCOPE[scope]
    except KeyError:
        raise ValueError(
            f"scope must be one of {sorted(_DATASET_BY_SCOPE)}, got {scope!r}"
        ) from None


@lru_cache(maxsize=len(_DATASET_BY_SCOPE))
def _proteoform_frame(scope: str) -> pd.DataFrame:
    """Cached registry frame for a scope. Internal, read-only — public callers
    get a copy. ``scope`` is required (no default) so the cache has exactly one
    key per scope; callers that want the default pass ``"cta"`` explicitly."""
    # get_data() returns its own copy (copy defaults True); normalize the gene-id
    # column on that copy — never mutate the shared get_data cache in place.
    df = get_data(_dataset_for_scope(scope))
    df[_GENE_ID_COLUMN] = df[_GENE_ID_COLUMN].astype(str).str.split(".").str[0]
    return df


def proteoform_groups(*, scope: str = "cta") -> pd.DataFrame:
    """The proteoform registry: one row per member gene. Defensive copy.

    Columns: ``proteoform_id`` (slash-joined sorted member symbols),
    ``member_symbol``, ``member_gene_id`` (unversioned Ensembl), ``protein_length``,
    ``n_members``.

    ``scope="genome"`` returns the genome-wide identical-protein grouping (every
    protein-coding family, not just CGAs). The default ``"cta"`` registry is a
    *refinement* of it: every CTA group's member genes fall within a single genome
    group, but the genome group may merge in additional non-CTA paralogs and so
    carry a larger label (e.g. CTA ``CT45A5/CT45A7`` ⊆ genome ``CT45A5/CT45A6/CT45A7``).
    Do not assume a gene keeps the same label across scopes.
    """
    return _proteoform_frame(scope).copy()


@lru_cache(maxsize=len(_DATASET_BY_SCOPE))
def proteoform_group_map(*, scope: str = "cta") -> dict[str, tuple[str, ...]]:
    """``{proteoform label: (member gene IDs, …)}`` for every group (see
    :func:`proteoform_groups` for ``scope``)."""
    df = _proteoform_frame(scope)
    out: dict[str, tuple[str, ...]] = {}
    for label, sub in df.groupby(_LABEL_COLUMN):
        out[str(label)] = tuple(sub[_GENE_ID_COLUMN].astype(str))
    return out


@lru_cache(maxsize=len(_DATASET_BY_SCOPE))
def proteoform_symbol_map(*, scope: str = "cta") -> dict[str, tuple[str, ...]]:
    """``{proteoform label: (member symbols, …)}`` for every group (see
    :func:`proteoform_groups` for ``scope``)."""
    df = _proteoform_frame(scope)
    out: dict[str, tuple[str, ...]] = {}
    for label, sub in df.groupby(_LABEL_COLUMN):
        out[str(label)] = tuple(sub[_SYMBOL_COLUMN].astype(str))
    return out


@lru_cache(maxsize=len(_DATASET_BY_SCOPE))
def _member_to_label(scope: str) -> dict[str, str]:
    """Lookup keyed by BOTH the unversioned gene ID and the uppercased symbol ->
    proteoform label, for one scope. The two key spaces don't collide (``ENSG…`` vs
    symbols), so one flat dict serves both ``proteoform_for_gene`` lookup paths."""
    df = _proteoform_frame(scope)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        label = str(row[_LABEL_COLUMN])
        out[str(row[_GENE_ID_COLUMN])] = label
        out[str(row[_SYMBOL_COLUMN]).upper()] = label
    return out


def proteoform_for_gene(gene: str, *, scope: str = "cta") -> str | None:
    """Proteoform label for a gene given by Ensembl ID (version-insensitive) or
    symbol (case-insensitive), within ``scope`` (see :func:`proteoform_groups`).
    ``None`` if the gene isn't in any group in that scope."""
    mapping = _member_to_label(scope)
    return mapping.get(str(gene).split(".")[0]) or mapping.get(str(gene).upper())


def gene_to_proteoform() -> dict[str, str]:
    """``{member gene ID: proteoform label}`` (Ensembl IDs only) — multi-member
    groups only. For a **total** map (every gene → its class, singletons included)
    use :func:`gene_to_proteoform_id`."""
    df = _proteoform_frame("cta")
    return dict(zip(df[_GENE_ID_COLUMN].astype(str), df[_LABEL_COLUMN].astype(str)))


def gene_to_proteoform_id(genes, *, symbols=None, scope: str = "cta") -> dict[str, str]:
    """Total ``{gene → proteoform_id}`` over the given genes: every gene maps to
    exactly one proteoform equivalence class.

    A gene in a multi-member group maps to the group's ``proteoform_id`` (the
    slash-joined label); a gene in no group is its **own** singleton class, mapping
    to its symbol (if ``symbols`` is given, aligned to ``genes``) else its
    unversioned Ensembl id. This is the join key for proteoform-level analyses —
    unlike :func:`gene_to_proteoform`, it is defined for **every** gene, so callers
    never need an ad-hoc "is it grouped?" branch. ``genes``/``symbols`` are parallel
    iterables of Ensembl ids / symbols."""
    pf = _proteoform_frame(scope)
    member_map = dict(zip(pf[_GENE_ID_COLUMN].astype(str), pf[_LABEL_COLUMN].astype(str)))
    genes = [str(g).split(".")[0] for g in genes]
    if symbols is None:
        syms: list = [None] * len(genes)
    else:
        syms = list(symbols)
        if len(syms) != len(genes):
            # zip() would silently truncate to the shorter and drop genes from the
            # output entirely — fail loudly instead.
            raise ValueError(
                f"symbols has length {len(syms)} but genes has length {len(genes)}; "
                "they must be parallel"
            )
    out: dict[str, str] = {}
    for gid, sym in zip(genes, syms):
        out[gid] = member_map.get(gid) or (str(sym) if sym is not None else gid)
    return out


def collapse_to_proteoforms(
    df: pd.DataFrame, *, scope: str = "cta", sample_cols=None
) -> pd.DataFrame:
    """Collapse a genes×samples expression frame to proteoform level — the single
    reusable entry point for proteoform summation.

    Identical-protein members in ``scope`` are summed per sample into one antigen
    row (see :func:`cancerdata._build.sum_proteoform_tpm`). The output keeps
    ``Ensembl_Gene_ID`` as a real **canonical-member** ENSG (never the label) and
    adds a total ``proteoform_id`` column (the class identity). Use this instead of
    calling ``sum_proteoform_tpm`` + ``proteoform_group_map`` directly, so coverage,
    the medoid/within-sample generators, and any future consumer share one
    collapse + one identity scheme."""
    from ._build import sum_proteoform_tpm

    return sum_proteoform_tpm(df, proteoform_group_map(scope=scope), sample_cols)
