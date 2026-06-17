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

THE PROTEOFORM KEY SPACE
========================
After collapse (:func:`collapse_to_proteoforms`), expression is keyed by **one**
identity per protein — :func:`proteoform_key` — built by a single rule used
consistently everywhere a frame is compared / quantified / plotted:

  - a gene that **uniquely** encodes its protein keys to its own **unversioned
    Ensembl gene id** (``ENSG00000185686``) — the clean 1:1 gene→protein case;
  - an identical-protein **group** keys to a **proteoform symbol** — the curated
    alias (``NY-ESO-1``) or, failing that, the prefix-contracted member symbols
    (``XAGE1A/XAGE1B`` → ``XAGE1A/B``, ``CGB3/CGB5/CGB8`` → ``CGB3/5/8``). The member
    ENSGs are summed away and **never appear as keys**.

The two alphabets are disjoint (``ENSG…`` vs symbols), so the key space is
unambiguous, and the collapse strictly **reduces** the key count (members merge).
A collapsed frame carries: ``proteoform_key`` (the identity above), ``Symbol``
(display — the proteoform symbol / the gene symbol), ``Ensembl_Gene_ID`` (the
canonical-member ENSG, a real id for joining to per-gene Ensembl references), and
``proteoform_members`` (the sorted slash-joined member symbols, provenance only).

How the groups are defined: every gene is reduced to its **best (canonical, longest)
protein sequence**, and genes whose best protein is byte-identical form one group —
so the key space is "one entry per distinct protein," and a group's proteoform-level
expression is the **sum of its member genes' TPMs** (RNA-seq reads multi-map between
identical-protein loci, so per-gene TPM under-counts the protein). ``scope`` selects
the gene universe: ``"cta"`` (the focused cancer-germline set, the shipped default)
or ``"genome"`` (all protein-coding genes — histones, tubulins, the PAR X/Y pairs,
…). Both registries ship; the genome one is the universal "do this for every gene".

GENE-LEVEL vs PROTEOFORM-LEVEL: an expression frame is one or the other, told apart
by :func:`expression_level` (the ``proteoform_key`` column is present iff collapsed).
Keep the two straight — don't mix a gene-level dict/stat with a proteoform-level one;
convert with :func:`collapse_to_proteoforms`.

This module is the read surface over the curated registry
(``proteoform-groups.csv``, derived offline by
``scripts/generate_proteoform_groups.py`` from pyensembl protein sequences). It
owns *which genes sum together*; the per-sample TPM summation itself lives in
:func:`oncoref.expression.proteoform_representative_samples` (runtime, over the
shipped medoid samples) and :func:`oncoref.expression_builders.sum_proteoform_tpm` (the
pure build-time core, ready for the offline percentile/within-sample generators
to apply before ranking when proteoform-summed artifacts are added to the bundle).

The registry's ``proteoform_id`` column is the sorted slash-joined member symbols
(``"SSX4/SSX4B"``) — what surfaces as ``proteoform_members`` after a collapse.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

from .gene_ids import unversioned
from .load_dataset import get_data

_LABEL_COLUMN = "proteoform_id"  # registry column: the sorted slash-joined members
_SYMBOL_COLUMN = "member_symbol"
_GENE_ID_COLUMN = "member_gene_id"

#: Curated preferred names for specific proteoforms, keyed by the **sorted
#: slash-joined member symbols** (the canonical members label). A group with an alias
#: collapses to the alias instead of the contracted member symbols. Small and
#: hand-maintained — extend as needed.
_PROTEOFORM_ALIASES = {
    "CTAG1A/CTAG1B": "NY-ESO-1",
}


def expression_level(df) -> str:
    """Which space an expression frame is in: ``"proteoform"`` if it carries a
    ``proteoform_key`` column (the collapsed key space, where identical-protein genes
    are summed into one row), else ``"gene"`` (one row per Ensembl gene).

    The single, uniform way for any code path or caller to be **cognizant** of the
    level it's holding — so a gene-level dict/stat is never silently mixed with a
    proteoform-level one. Pair with :func:`collapse_to_proteoforms` to convert
    gene-level → proteoform-level."""
    return "proteoform" if "proteoform_key" in getattr(df, "columns", ()) else "gene"


def _contract_members(members_label: str) -> str:
    """Compact a sorted slash-joined members label by factoring out the common symbol
    prefix: ``XAGE1A/XAGE1B`` → ``XAGE1A/B``, ``CGB3/CGB5/CGB8`` → ``CGB3/5/8``,
    ``SSX4/SSX4B`` → ``SSX4/B``. No shared prefix → unchanged. Duplicate member
    symbols (genome-scope X/Y paralogs sharing a symbol, e.g. ``AKAP17A/AKAP17A``) are
    de-duplicated, so the result never carries a trailing or empty ``/`` segment."""
    parts = list(dict.fromkeys(members_label.split("/")))  # de-dup, order-preserving
    if len(parts) < 2:
        return parts[0] if parts else members_label
    prefix = os.path.commonprefix(parts)
    suffixes = [p[len(prefix) :] for p in parts[1:]]
    return parts[0] + "".join("/" + s for s in suffixes if s)


def proteoform_aliases() -> dict[str, str]:
    """Curated ``{sorted members label → preferred name}`` (e.g. ``CTAG1A/CTAG1B →
    NY-ESO-1``). Copy."""
    return dict(_PROTEOFORM_ALIASES)


def proteoform_symbol(members_label: str) -> str:
    """The single proteoform symbol that survives a collapse, from a group's sorted
    members label: the curated alias if one exists, else the prefix-contracted member
    symbols (``XAGE1A/XAGE1B`` → ``XAGE1A/B``). The reduced key — the individual
    members no longer appear."""
    return _PROTEOFORM_ALIASES.get(members_label) or _contract_members(members_label)


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
    """Proteoform **symbol** for a gene given by Ensembl ID (version-insensitive) or
    symbol (case-insensitive), within ``scope`` (see :func:`proteoform_groups`): the
    curated alias (``NY-ESO-1``) or the prefix-contracted members (``XAGE1A/B``).
    ``None`` if the gene isn't in any multi-member group in that scope."""
    mapping = _member_to_label(scope)
    label = mapping.get(unversioned(gene)) or mapping.get(str(gene).upper())
    return proteoform_symbol(label) if label is not None else None


def proteoform_members_for_gene(gene: str, *, scope: str = "cta") -> tuple[str, ...] | None:
    """The member symbols of the proteoform a gene belongs to (the provenance behind
    :func:`proteoform_for_gene`), or ``None`` if it's in no multi-member group."""
    mapping = _member_to_label(scope)
    label = mapping.get(unversioned(gene)) or mapping.get(str(gene).upper())
    return proteoform_symbol_map(scope=scope).get(label) if label is not None else None


def proteoform_key(gene: str, *, scope: str = "cta") -> str:
    """The **proteoform key** for a gene — the one identity used to compare, quantify
    and plot across the codebase:

      - a gene that **uniquely** owns its protein (no identical-protein paralog) keys
        to its own **unversioned Ensembl gene id** (``ENSG…``);
      - a gene in an identical-protein **group** keys to the group's **proteoform
        symbol** (``NY-ESO-1`` / ``XAGE1A/B``) — the member ENSGs are summed away and
        never appear as keys.

    The two key alphabets are disjoint (``ENSG…`` vs symbols), so the key space is
    unambiguous. Pass an Ensembl gene id for the ENSG-keyed singleton result; a symbol
    input keys to itself when ungrouped."""
    sym = proteoform_for_gene(gene, scope=scope)
    return sym if sym is not None else unversioned(gene)


def gene_to_proteoform() -> dict[str, str]:
    """``{member gene ID: proteoform label}`` (Ensembl IDs only) — multi-member
    groups only. For a **total** map (every gene → its class, singletons included)
    use :func:`gene_to_proteoform_id`."""
    df = _proteoform_frame("cta")
    return dict(zip(df[_GENE_ID_COLUMN].astype(str), df[_LABEL_COLUMN].astype(str)))


def gene_to_proteoform_id(genes, *, scope: str = "cta") -> dict[str, str]:
    """Total ``{Ensembl gene id → proteoform key}`` over the given genes — the batch
    form of :func:`proteoform_key`. Every gene maps to exactly one key: its own
    **unversioned ENSG** if it uniquely owns its protein, the group's **proteoform
    symbol** (``NY-ESO-1`` / ``XAGE1A/B``) if its ENSG was summed into a group. Total,
    so callers never need an ad-hoc "is it grouped?" branch."""
    pf = _proteoform_frame(scope)
    member_map = dict(zip(pf[_GENE_ID_COLUMN].astype(str), pf[_LABEL_COLUMN].astype(str)))
    out: dict[str, str] = {}
    for g in genes:
        gid = unversioned(g)
        label = member_map.get(gid)
        out[gid] = proteoform_symbol(label) if label is not None else gid
    return out


def collapse_to_proteoforms(
    df: pd.DataFrame, *, scope: str = "cta", sample_cols=None
) -> pd.DataFrame:
    """Collapse a genes×samples expression frame to proteoform level — the single
    reusable entry point for proteoform summation, and the "second step" applied
    before comparing/quantifying/plotting so everything shares one reduced key space.

    Identical-protein members in ``scope`` are summed per sample into one row, so the
    key count **decreases** (the members disappear). The output (see the module
    docstring's "key space"):

      - ``proteoform_key`` — THE identity: the gene's own ENSG when it uniquely owns
        its protein, the proteoform symbol (``NY-ESO-1`` / ``XAGE1A/B``) for a group;
      - ``Ensembl_Gene_ID`` — the group's canonical-member ENSG (``min``), a real
        joinable Ensembl id; the gene's own ENSG for a singleton (the 1:1 case);
      - ``Symbol`` — display: the proteoform symbol for a group, the gene's symbol for
        a singleton;
      - ``proteoform_members`` — the sorted slash-joined member symbols
        (``CTAG1A/CTAG1B``), provenance only; the gene's symbol for a singleton.

    Use this instead of calling ``sum_proteoform_tpm`` + ``proteoform_group_map``
    directly, so coverage, the medoid/within-sample generators, and any future
    consumer share one collapse + one identity scheme."""
    from .expression_builders import sum_proteoform_tpm

    gmap = proteoform_group_map(scope=scope)
    group_symbols = {label: proteoform_symbol(label) for label in gmap}
    return sum_proteoform_tpm(df, gmap, sample_cols, group_symbols=group_symbols)
