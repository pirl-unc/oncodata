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


@lru_cache(maxsize=1)
def _member_to_label() -> dict[str, str]:
    """Lookup keyed by BOTH the unversioned gene ID and the uppercased symbol ->
    proteoform label. The two key spaces don't collide (``ENSG…`` vs symbols), so
    one flat dict serves both ``proteoform_for_gene`` lookup paths."""
    df = _proteoform_frame("cta")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        label = str(row[_LABEL_COLUMN])
        out[str(row[_GENE_ID_COLUMN])] = label
        out[str(row[_SYMBOL_COLUMN]).upper()] = label
    return out


def proteoform_for_gene(gene: str) -> str | None:
    """Proteoform label for a gene given by Ensembl ID (version-insensitive) or
    symbol (case-insensitive). ``None`` if the gene isn't in any group."""
    key = str(gene).split(".")[0]
    mapping = _member_to_label()
    return mapping.get(key) or mapping.get(str(gene).upper())


def gene_to_proteoform() -> dict[str, str]:
    """``{member gene ID: proteoform label}`` (Ensembl IDs only)."""
    df = _proteoform_frame("cta")
    return dict(zip(df[_GENE_ID_COLUMN].astype(str), df[_LABEL_COLUMN].astype(str)))
