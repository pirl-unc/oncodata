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
    """Map an Ensembl gene id to its canonical primary-assembly id (unversioned),
    resolving alt-haplotype / patch copies and cross-release ID turnover through the
    shipped ensembl-id-aliases map; returns the id unchanged if it isn't an alias."""
    key = _unversioned(gene_id)
    return ensembl_id_aliases().get(key, key)


def canonical_gene_id(identifier: str, *, source_version: str | None = None) -> str | None:
    """Map **any** gene identifier into the harmonized canonical ENSG space.

    The single entry point a consumer reaches for when harmonizing arbitrary inputs
    (oncoref#135): accepts an Ensembl gene id from *any* release/assembly — versioned or
    not, alt-haplotype, patch, or a retired/old-release id — or an HGNC/NCBI gene
    **symbol**, and returns the canonical (newest primary-assembly) ENSG, or ``None`` if
    it can't be resolved.

    - An ``ENSG…`` id is unversioned and run through the alias / cross-release migration
      map (:func:`resolve_ensembl_id`) — so e.g. GRCh37 ``GGNBP2`` ``ENSG00000005955`` →
      ``ENSG00000278311``. The id is returned as-is when it's already canonical / unknown.
    - Anything else is treated as a symbol and resolved via the installed Ensembl
      releases + NCBI synonyms (lazy import of the genome layer; needs the ``genome``
      extra). Symbol resolution already returns a canonical primary-assembly id.

    ``source_version`` (e.g. ``"GRCh37"`` / ``"75"``) is accepted for forward
    compatibility and caller intent; the migration map is release-agnostic today."""
    s = str(identifier).strip()
    if not s:
        return None
    if s.upper().startswith("ENSG"):
        return resolve_ensembl_id(s)
    from .genome import canonical_gene_id_and_name  # lazy: avoids genome/pyensembl dep here

    gid, _ = canonical_gene_id_and_name(s)
    return gid


def canonical_gene_ids(
    identifiers: list[str], *, source_version: str | None = None
) -> list[str | None]:
    """Batch :func:`canonical_gene_id` → one canonical ENSG (or ``None``) per input."""
    return [canonical_gene_id(x, source_version=source_version) for x in identifiers]


def canonical_gene_space() -> pd.DataFrame:
    """The authoritative **canonical gene-ID space** (oncoref#135 item 4): one row per
    primary-assembly Ensembl gene — ``ensembl_gene_id``, ``symbol``, ``biotype``,
    ``seqname``, ``ensembl_release`` — that the alias/migration map resolves *into*.
    Filter on ``biotype == "protein_coding"`` to drop the large ncRNA/pseudogene tail.
    Defensive copy."""
    return get_data("canonical-gene-space").copy()


@lru_cache(maxsize=1)
def _canonical_gene_index() -> dict[str, tuple[str, str]]:
    """``{ensembl_gene_id (unversioned): (symbol, biotype)}`` over the canonical space."""
    df = get_data("canonical-gene-space", copy=False)
    return {
        _unversioned(str(g)): (str(s), str(b))
        for g, s, b in zip(df["ensembl_gene_id"], df["symbol"], df["biotype"])
    }


def is_canonical_gene(gene_id: str) -> bool:
    """Is ``gene_id`` (after alias/migration resolution) a member of the canonical
    primary-assembly gene space? ``False`` for RefSeq-lift / non-Ensembl / retired-without-
    successor ids that have no canonical entry."""
    return resolve_ensembl_id(gene_id) in _canonical_gene_index()


def gene_biotype(gene_id: str) -> str | None:
    """Ensembl biotype (``"protein_coding"``, ``"lncRNA"``, …) of the canonical gene
    ``gene_id`` resolves to, or ``None`` if it isn't in the canonical space."""
    hit = _canonical_gene_index().get(resolve_ensembl_id(gene_id))
    return hit[1] if hit else None


def is_protein_coding_gene(gene_id: str) -> bool:
    """``True`` iff ``gene_id`` resolves to a ``protein_coding`` canonical gene."""
    return gene_biotype(gene_id) == "protein_coding"


@lru_cache(maxsize=1)
def ensembl_id_alias_symbols() -> dict[str, str]:
    """``{primary_contig_gene_id (unversioned): gene_symbol}`` — the curated canonical
    symbol for a migrated/consolidated locus, from the alias table (rows with no symbol,
    e.g. archive replacements, are skipped)."""
    df = get_data("ensembl-id-aliases", copy=False)
    out: dict[str, str] = {}
    for primary, symbol in zip(df["primary_contig_id"], df["symbol"]):
        sym = str(symbol).strip()
        if sym and sym.lower() != "nan":
            out[_unversioned(str(primary))] = sym
    return out


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
