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

"""CTA-specific 9-mer (peptide) enumeration from the reference proteome.

A 9-mer is **CTA-specific** when it appears in a CTA protein but in *no* non-CTA
protein anywhere in the proteome — a measure of how much truly tumor-restricted
neoepitope source a CTA provides (a shared 9-mer would risk central tolerance /
off-target reactivity). This mirrors pirlygenes' definition exactly:

  - protein sequences come from the installed Ensembl release (pyensembl), the
    **longest** protein-coding transcript per gene (stop codon stripped);
  - 9-mers are the distinct sliding-window (stride 1) substrings of a protein;
  - the **background** is the union of 9-mers over every non-CTA protein (the full
    CTA *universe* — :func:`cancerdata.cta.CTA_unfiltered_gene_ids` — is excluded so
    borderline CTAs don't poison the background);
  - a CTA's ``n_specific_9mers`` is how many of its 9-mers miss that background.

Building the background scans the whole proteome (~20k genes, a minute or two), so
:func:`cta_specific_9mer_counts` caches its per-CTA table to a CSV under the
cancerdata cache, keyed by Ensembl release. Needs a downloaded human release with
protein sequences (``pyensembl install --release N --species homo_sapiens``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .cta import CTA_gene_id_to_name, CTA_gene_ids, CTA_unfiltered_gene_ids
from .genome import find_gene_id_by_name, genomes, strip_version

#: 9 = the canonical MHC-I epitope length.
DEFAULT_K: int = 9

#: A complete human proteome resolves ~19–20k protein-coding genes. Far fewer means
#: the cDNA/protein FASTA is missing or partial — refuse to build (and cache) a
#: degenerate background that would inflate every CTA's "specific" count.
_MIN_PROTEOME_GENES: int = 10_000

#: In-process memo of the per-CTA counts table, keyed by (k, release, CTA-set
#: fingerprint). Holds the canonical frame; public accessors return a copy.
_COUNTS_CACHE: dict[tuple[int, int, str], pd.DataFrame] = {}


def _derived_cache_dir() -> Path:
    from .data_bundle import cache_root

    d = cache_root() / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _usable_genome():
    """Newest installed release whose protein sequences are available (GTF + protein
    FASTA), probed cheaply via TP53. ``None`` if none is usable."""
    probe = strip_version(find_gene_id_by_name("TP53") or "")
    for g in genomes():
        try:
            gene = g.gene_by_id(probe)
            if any(t.biotype == "protein_coding" and t.protein_sequence for t in gene.transcripts):
                return g
        except Exception:
            continue
    return None


def _kmers(seq: str, k: int) -> set[str]:
    """Distinct length-``k`` sliding-window substrings of ``seq``."""
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _longest_protein_per_gene(genome, k: int) -> dict[str, str]:
    """``{unversioned gene id -> longest protein-coding protein sequence}`` (≥ ``k``
    aa, trailing stop codon stripped) for one Ensembl release."""
    longest: dict[str, str] = {}
    for tr in genome.transcripts():
        if tr.biotype != "protein_coding":
            continue
        try:
            seq = tr.protein_sequence
        except Exception:
            seq = None
        if not seq or len(seq) < k:
            continue
        seq = seq.rstrip("*")
        if len(seq) < k:
            continue
        gid = strip_version(tr.gene_id)
        if gid not in longest or len(seq) > len(longest[gid]):
            longest[gid] = seq
    return longest


def _cta_set_fingerprint() -> str:
    """Short hash of the filtered + unfiltered CTA id sets — the inputs that define
    the output rows and the background. Embedded in the cache key so a curation edit
    (a CTA added/removed) invalidates a stale cache *within* the same Ensembl release,
    not only across releases."""
    payload = "F:" + ",".join(sorted(strip_version(g) for g in CTA_gene_ids()))
    payload += "|U:" + ",".join(sorted(strip_version(g) for g in CTA_unfiltered_gene_ids()))
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _build_counts(genome, k: int) -> pd.DataFrame:
    """Compute the per-CTA specific-9-mer table from one genome (no caching)."""
    longest = _longest_protein_per_gene(genome, k)
    if len(longest) < _MIN_PROTEOME_GENES:
        raise RuntimeError(
            f"only {len(longest)} protein sequences resolved from Ensembl release "
            f"{genome.release} — the protein FASTA looks incomplete, so the CTA-specific "
            "background would be wrong. Reinstall with "
            f"`pyensembl install --release {genome.release} --species homo_sapiens`."
        )
    cta_filtered = {strip_version(g) for g in CTA_gene_ids()}
    cta_universe = {strip_version(g) for g in CTA_unfiltered_gene_ids()}

    background: set[str] = set()
    for gid, seq in longest.items():
        if gid in cta_universe:
            continue  # exclude the whole CTA universe so borderline CTAs don't mask specificity
        background |= _kmers(seq, k)

    id2name = CTA_gene_id_to_name()
    rows = []
    for gid in sorted(cta_filtered):
        seq = longest.get(gid)
        km = _kmers(seq, k) if seq else set()
        rows.append(
            {
                "Ensembl_Gene_ID": gid,
                "Symbol": id2name.get(gid, gid),
                "n_9mers": len(km),
                "n_specific_9mers": sum(1 for x in km if x not in background),
            }
        )
    return pd.DataFrame(rows, columns=["Ensembl_Gene_ID", "Symbol", "n_9mers", "n_specific_9mers"])


def cta_specific_9mer_counts(*, k: int = DEFAULT_K, refresh: bool = False) -> pd.DataFrame:
    """Per filtered CTA, its distinct-9-mer count and how many are CTA-specific.

    Columns: ``Ensembl_Gene_ID`` (unversioned), ``Symbol``, ``n_9mers`` (distinct
    9-mers in its longest protein), ``n_specific_9mers`` (those absent from every
    non-CTA protein). Cached to a CSV keyed by Ensembl release **and** a fingerprint
    of the CTA gene set, then memoized in-process; each call returns a fresh copy.
    Pass ``refresh=True`` to drop both caches and rebuild. Raises if no usable Ensembl
    release is installed or its proteome looks incomplete."""
    genome = _usable_genome()
    if genome is None:
        raise RuntimeError(
            "no usable human Ensembl release with protein sequences installed — run "
            "`pyensembl install --release 111 --species homo_sapiens`"
        )
    fp = _cta_set_fingerprint()
    key = (k, genome.release, fp)
    cache = _derived_cache_dir() / f"cta_specific_{k}mers_r{genome.release}_{fp}.csv"
    if refresh:
        _COUNTS_CACHE.pop(key, None)
        cache.unlink(missing_ok=True)
    if key not in _COUNTS_CACHE:
        if cache.exists():
            _COUNTS_CACHE[key] = pd.read_csv(cache)
        else:
            df = _build_counts(genome, k)
            df.to_csv(cache, index=False)
            _COUNTS_CACHE[key] = df
    return _COUNTS_CACHE[key].copy()


def cta_specific_9mer_weights(*, k: int = DEFAULT_K, by: str = "ensembl_gene_id") -> dict[str, int]:
    """``{key -> n_specific_9mers}`` from :func:`cta_specific_9mer_counts`.

    ``by="ensembl_gene_id"`` (default) keys by unversioned Ensembl gene id — the safe
    join key, since a proteoform-collapsed row's ``Symbol`` is a slash-joined label
    that no per-gene symbol key matches (see :func:`cta_specific_9mer_load`).
    ``by="symbol"`` keys by gene symbol for display. Identical-protein paralogs share
    a sequence, hence one count, so a canonical member's id carries the group's weight."""
    df = cta_specific_9mer_counts(k=k)
    if by == "ensembl_gene_id":
        keys = df["Ensembl_Gene_ID"]
    elif by == "symbol":
        keys = df["Symbol"]
    else:
        raise ValueError("by must be 'ensembl_gene_id' or 'symbol'")
    return {str(key): int(n) for key, n in zip(keys, df["n_specific_9mers"])}


def cta_specific_9mer_load(
    cancer_type, *, threshold_tpm: float = 10.0, k: int = DEFAULT_K
) -> float:
    """Mean per-patient CTA-specific 9-mer **load** for a cohort: for the average
    patient, the total CTA-specific 9-mers summed over the CTAs they express above
    ``threshold_tpm``.

    By linearity this equals ``Σ over antigens (fraction_expressing ×
    n_specific_9mers)``, so it is built directly on
    :func:`cancerdata.coverage.cta_patient_fractions` (proteoform-summed) — no second
    pass over the per-sample matrix. Needs the cohort's per-sample matrix cached.

    The join is on the **canonical-member Ensembl gene id**, not ``Symbol``: a
    collapsed proteoform's ``Symbol`` is the slash-joined label (``CTAG1A/CTAG1B``),
    which would never match the per-gene weight table — but identical-protein members
    share a sequence (hence one specific-9-mer count), so the canonical member's id
    carries the proteoform's weight."""
    from .coverage import cta_patient_fractions

    pf = cta_patient_fractions(cancer_type, threshold_tpm=threshold_tpm)
    if pf.empty:
        return 0.0
    weight_by_id = cta_specific_9mer_weights(k=k, by="ensembl_gene_id")
    w = pf["Ensembl_Gene_ID"].astype(str).map(lambda g: weight_by_id.get(strip_version(g), 0))
    return float((pf["fraction_expressing"] * w).sum())


__all__ = [
    "DEFAULT_K",
    "cta_specific_9mer_counts",
    "cta_specific_9mer_load",
    "cta_specific_9mer_weights",
]
