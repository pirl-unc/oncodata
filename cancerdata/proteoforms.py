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


@lru_cache(maxsize=1)
def _proteoform_frame() -> pd.DataFrame:
    """Cached registry frame. Internal, read-only — public callers get a copy."""
    df = get_data("proteoform-groups", copy=False)
    df[_GENE_ID_COLUMN] = df[_GENE_ID_COLUMN].astype(str).str.split(".").str[0]
    return df


def proteoform_groups() -> pd.DataFrame:
    """The proteoform registry: one row per member gene. Defensive copy.

    Columns: ``proteoform_id`` (slash-joined sorted member symbols),
    ``member_symbol``, ``member_gene_id`` (unversioned Ensembl), ``protein_length``,
    ``n_members``.
    """
    return _proteoform_frame().copy()


@lru_cache(maxsize=1)
def proteoform_group_map() -> dict[str, tuple[str, ...]]:
    """``{proteoform label: (member gene IDs, …)}`` for every group."""
    df = _proteoform_frame()
    out: dict[str, tuple[str, ...]] = {}
    for label, sub in df.groupby(_LABEL_COLUMN):
        out[str(label)] = tuple(sub[_GENE_ID_COLUMN].astype(str))
    return out


@lru_cache(maxsize=1)
def proteoform_symbol_map() -> dict[str, tuple[str, ...]]:
    """``{proteoform label: (member symbols, …)}`` for every group."""
    df = _proteoform_frame()
    out: dict[str, tuple[str, ...]] = {}
    for label, sub in df.groupby(_LABEL_COLUMN):
        out[str(label)] = tuple(sub[_SYMBOL_COLUMN].astype(str))
    return out


@lru_cache(maxsize=1)
def _member_to_label() -> dict[str, str]:
    """Lookup keyed by BOTH the unversioned gene ID and the uppercased symbol ->
    proteoform label. The two key spaces don't collide (``ENSG…`` vs symbols), so
    one flat dict serves both ``proteoform_for_gene`` lookup paths."""
    df = _proteoform_frame()
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
    df = _proteoform_frame()
    return dict(zip(df[_GENE_ID_COLUMN].astype(str), df[_LABEL_COLUMN].astype(str)))
