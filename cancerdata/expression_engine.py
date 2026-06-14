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

"""Transcript→gene expression aggregation (the pandas-only grouping core).

Sums transcript-level TPM to gene level given a transcript→gene mapping. This is
the part of the operation that is cancerdata's domain — the deterministic grouping
and TPM summation — independent of which transcript reference produced the quant.

**Dependency boundary.** Resolving an *arbitrary, unknown* transcript ID to a gene
(and a gene symbol to its Ensembl ID) is a reference-genome operation that needs
``pyensembl`` — out of cancerdata's pandas-only base layer. So this function maps
transcripts via the supplied ``tx_to_gene_name`` dict (default: cancerdata's curated
``extra-tx-mappings`` back-compat set) and reports the unresolved TPM fraction;
transcripts not in the map are summed into an ``unresolved`` bucket rather than
silently dropped. A consumer that needs full Ensembl-reference resolution passes a
complete ``tx_to_gene_name`` (e.g. built from pyensembl on its side).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

#: Canonical identity columns of a cancerdata expression frame. Everything else is a
#: per-sample / per-representative value column. One definition shared by every
#: "value columns = all columns except the id columns" consumer (the build
#: generators, the normalization helpers, the coverage hit-matrix) so a
#: proteoform-collapsed frame's identity columns (``proteoform_key`` / ``Symbol`` /
#: ``proteoform_members``) are never mistaken for samples. (The *named*-TPM rule for
#: curated frames is :func:`is_expression_value_col`, a deliberately distinct concept.)
ID_COLUMNS = ("proteoform_key", "Ensembl_Gene_ID", "Symbol", "proteoform_members")


def id_columns(df: pd.DataFrame) -> list[str]:
    """The identity columns of an expression frame, in canonical :data:`ID_COLUMNS`
    order, restricted to those actually present (a gene-level frame lacks the
    ``proteoform_*`` columns). The single definition of "which columns are identity",
    so no consumer has to re-list them."""
    return [c for c in ID_COLUMNS if c in df.columns]


def sample_columns(df: pd.DataFrame) -> list[str]:
    """The per-sample / per-representative **value** columns of an expression frame:
    every column that is not one of the :data:`ID_COLUMNS`. The single definition of
    "which columns hold expression values" — the partner of :func:`id_columns`, and
    the one place the value/identity boundary is decided. (For the distinct *named*-TPM
    rule on curated long tables, see :func:`is_expression_value_col`.)"""
    return [c for c in df.columns if c not in ID_COLUMNS]


_DEFAULT_TX_COLUMN_CANDIDATES = (
    "transcript",
    "transcript_id",
    "transcriptid",
    "target",
    "target_id",
    "targetid",
    "name",
)
_DEFAULT_TPM_COLUMN_CANDIDATES = ("tpm",)


def find_column(df: pd.DataFrame, candidates, column_name: str) -> str:
    """The column of ``df`` matching the highest-priority ``candidates`` entry —
    absorbs naming variation across upstream quantifiers (``transcript_id`` vs ``tx``
    …). ``candidates`` is consulted **in order**, so a frame carrying both
    ``transcript_id`` and ``name`` resolves by candidate priority, not by column
    order. Matching is case-insensitive. Raises ``ValueError`` listing the available
    columns if nothing matches."""
    by_lower = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        col = by_lower.get(cand.lower())
        if col is not None:
            return col
    raise ValueError(
        f"no column for {column_name} in expression data; available: {list(df.columns)}"
    )


def expanded_tx_map(tx_to_gene_name: dict) -> dict:
    """Expand a ``{transcript_id: gene}`` map to also key the versionless id
    (``ENST….5`` → ``ENST…``), first-seen value winning. So a versioned input id
    matches a versionless map entry and vice versa."""
    out: dict = {}
    for k, v in tx_to_gene_name.items():
        out.setdefault(str(k), v)
        out.setdefault(str(k).split(".", 1)[0], v)
    return out


@lru_cache(maxsize=1)
def _default_tx_to_gene() -> dict:
    """cancerdata's curated ``extra-tx-mappings`` (transcript_id → gene_symbol) —
    the back-compat known set; not a full Ensembl reference."""
    from .gene_ids import extra_transcript_mappings

    df = extra_transcript_mappings()
    return dict(zip(df["transcript_id"].astype(str), df["gene_symbol"].astype(str)))


def aggregate_transcripts_to_genes(
    df: pd.DataFrame,
    tx_to_gene_name: dict | None = None,
    *,
    transcript_id_column_candidates=_DEFAULT_TX_COLUMN_CANDIDATES,
    tpm_column_candidates=_DEFAULT_TPM_COLUMN_CANDIDATES,
    unresolved_label: str = "unresolved",
) -> pd.DataFrame:
    """Aggregate transcript-level TPM to gene level.

    Finds the transcript-id and TPM columns (by the candidate name lists), maps each
    transcript to a gene via ``tx_to_gene_name`` (default :func:`_default_tx_to_gene`,
    matched version-insensitively), and sums TPM per gene. Transcripts not in the map
    are summed into one ``unresolved_label`` row (never dropped — a gene whose every
    transcript is unknown must stay accounted for, not vanish from the quant).

    Returns a DataFrame with ``gene`` and ``TPM`` (one row per gene, plus the
    ``unresolved`` row if any), sorted by TPM. ``df.attrs["aggregation_stats"]``
    carries the known/unresolved TPM split so a caller can gate on resolution
    quality. See the module docstring for the pyensembl boundary."""
    tx_col = find_column(df, transcript_id_column_candidates, "transcript ID")
    tpm_col = find_column(df, tpm_column_candidates, "TPM")

    tx0 = df[tx_col].astype(str).str.split(".", n=1).str[0]
    tpm = pd.to_numeric(df[tpm_col], errors="coerce").fillna(0.0)
    tx_map = expanded_tx_map(
        tx_to_gene_name if tx_to_gene_name is not None else _default_tx_to_gene()
    )
    gene = tx0.map(tx_map)

    unknown = gene.isna()
    unknown_tpm = float(tpm[unknown].sum())
    gene = gene.where(~unknown, unresolved_label)

    agg = (
        pd.DataFrame({"gene": gene.astype(str), "TPM": tpm.to_numpy()})
        .groupby("gene", as_index=False, sort=False)["TPM"]
        .sum()
        .sort_values("TPM")
        .reset_index(drop=True)
    )
    total = float(tpm.sum())
    agg.attrs["aggregation_stats"] = {
        "total_tpm": total,
        "unresolved_tpm": unknown_tpm,
        "unresolved_fraction": (unknown_tpm / total) if total > 0 else 0.0,
        "unresolved_transcript_count": int(unknown.sum()),
        "n_genes": int((agg["gene"] != unresolved_label).sum()),
    }
    return agg


__all__ = [
    "aggregate_transcripts_to_genes",
    "expanded_tx_map",
    "find_column",
]
