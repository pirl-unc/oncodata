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

"""Gene-id and symbol resolution references.

oncoref data comes from many pipelines on different Ensembl/HGNC bases. These
curated maps reconcile them: alt-haplotype Ensembl gene ids → their primary-contig
id, NCBI symbol synonyms → the official symbol, plus supplemental transcript→gene
mappings and the cDNA-identical / proteoform-collapse-override grouping refinements.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .load_dataset import get_data


def unversioned(gene_id: str) -> str:
    """Canonical Ensembl-id normalizer: ``ENSG00000251562.5`` → ``ENSG00000251562``,
    stripping a single version suffix *and* surrounding whitespace (idempotent on bare
    ids). The one definition every layer shares, so a padded/versioned id can't match
    in one place and miss in another."""
    return str(gene_id).split(".", 1)[0].strip()


#: Back-compat internal alias.
_unversioned = unversioned


@lru_cache(maxsize=1)
def ensembl_id_aliases() -> dict[str, str]:
    """``{alt_haplotype_gene_id: primary_contig_gene_id}`` (unversioned) — alt-locus
    Ensembl gene ids mapped to their primary-assembly id."""
    df = get_data("ensembl-id-aliases", copy=False)
    return {
        _unversioned(a): _unversioned(p)
        for a, p in zip(df["alt_haplotype_id"], df["primary_contig_id"])
    }


def resolve_ensembl_id(gene_id: str) -> str:
    """Map an alt-haplotype Ensembl gene id to its primary-contig id (unversioned);
    returns the id unchanged if it isn't an alias."""
    key = _unversioned(gene_id)
    return ensembl_id_aliases().get(key, key)


@lru_cache(maxsize=1)
def symbol_synonyms() -> dict[str, str]:
    """``{ALIAS (uppercased): official_symbol}`` from NCBI gene synonyms."""
    df = get_data("ncbi-symbol-synonyms", copy=False)
    return {str(a).upper(): str(o) for a, o in zip(df["alias"], df["official_symbol"])}


def resolve_symbol(symbol: str) -> str:
    """Map a gene-symbol synonym to its official symbol (case-insensitive); returns
    the input unchanged if it's already official / unknown."""
    return symbol_synonyms().get(str(symbol).upper(), str(symbol))


def extra_transcript_mappings() -> pd.DataFrame:
    """Supplemental transcript→gene mappings (``transcript_id``, ``gene_symbol``,
    ``ensembl_gene_id``, ``biotype``, …). Defensive copy."""
    return get_data("extra-tx-mappings").copy()


def cdna_identical_groups() -> pd.DataFrame:
    """cDNA-identical gene groups (one row per member: ``group_canonical_*``,
    ``ensembl_gene_id``, ``symbol``, ``cds_nt``, ``n_members``). Defensive copy."""
    return get_data("cdna-identical-gene-groups").copy()


def proteoform_collapse_overrides() -> pd.DataFrame:
    """Manual proteoform-collapse overrides (``group_canonical_ensembl_gene_id``,
    ``group_symbol``, ``reason``). Defensive copy."""
    return get_data("proteoform-collapse-overrides").copy()
