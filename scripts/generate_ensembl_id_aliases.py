#!/usr/bin/env python
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

"""Derive the comprehensive Ensembl gene-ID alias / migration map.

oncoref's cohorts were quantified against many Ensembl releases (GRCh37/75 through
GRCh38/115) and assemblies, so one biological gene fragments into several stable IDs:

1. **Alt-haplotype / patch / scaffold copies** — a gene annotated on a non-primary
   contig (``CHR_HSCHR6_MHC_*``, ``HG..._PATCH``, unplaced scaffolds) carries a
   *different* stable ID from its primary-assembly copy (e.g. LHX1, ZFP57, HLA-Z,
   TRIM26). RNA-seq reads multi-map between the copies, so the downstream read layer
   **sums** the alt copy's TPM into the primary gene. Tagged ``ensembl_alt_contig``.

2. **Cross-release ID turnover** — a stable ID retired in a newer release with a
   same-named successor on the same primary chromosome (e.g. GRCh37 ``GGNBP2``
   ``ENSG00000005955`` → ``ENSG00000278311``; ``CDR1`` ``ENSG00000184258`` →
   ``ENSG00000288642``, biotype reclassified). Tagged ``ensembl_id_history_name``.
   These are the *same locus across releases* — the read layer relabels them to the
   newest ID (they don't co-occur per-sample, so summing degenerates to the value).

The canonical target is the gene's id on a **primary chromosome** (``1``–``22``,
``X``, ``Y``, ``MT``) in the newest installed release, keyed by its **unique** gene
name there (names shared by >1 primary gene are skipped — we never guess across a
copy-number family). Pure locus-overlap "merged-into" matches are deliberately
EXCLUDED: a short gene nested in a host gene's intron (a snoRNA, an antisense lncRNA)
overlaps it ~100% yet is a *distinct* gene — empirically those "successors" co-occur
with the old gene in ~all modern cohorts, proving they are not the same gene.

pyensembl is a BUILD-TIME tool only (not a runtime dependency of oncoref, cf. the
percentile/proteoform generators); it reads each release's GTF, which this script
queries directly for speed. Run:

    python scripts/generate_ensembl_id_aliases.py

Output: ``oncoref/data/ensembl-id-aliases.csv`` with columns
``alt_haplotype_id, primary_contig_id, symbol, source`` (the column names are
historical — ``alt_haplotype_id`` is now any non-canonical/old id, ``primary_contig_id``
its canonical successor). Existing curated rows whose alias id this build does not
reproduce are preserved (e.g. the 3 ``ensembl_archive_replacement`` rows).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "oncoref" / "data" / "ensembl-id-aliases.csv"

# Installed human releases to scan for old/alt ids, newest first. 115 (GRCh38) is the
# canonical target; 75 is GRCh37 — the only place GRCh37-retired ids (GGNBP2) live.
_DEFAULT_RELEASES = (
    115, 114, 112, 111, 110, 109, 108, 107, 105, 103,
    97, 93, 90, 87, 83, 81, 80, 79, 78, 77, 75,
)  # fmt: skip
_TARGET_RELEASE = 115

#: Canonical primary chromosomes; everything else (CHR_*, HG*_PATCH, KI*, GL*) is a
#: non-primary contig whose gene is an alias of its primary-assembly copy.
_PRIMARY_CONTIGS = frozenset([*(str(i) for i in range(1, 23)), "X", "Y", "MT"])


def _unversioned(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0].strip()


def _gene_table(release: int) -> pd.DataFrame:
    """The release's ``gene`` table as a DataFrame (gene_id, gene_name, seqname, …)."""
    import pyensembl

    er = pyensembl.EnsemblRelease(release, species="human")
    df = pd.read_sql(
        "SELECT gene_id, gene_name, seqname, start, end, strand FROM gene", er.db.connection
    )
    df["gid"] = df["gene_id"].map(_unversioned)
    df["seqname"] = df["seqname"].astype(str)
    return df


def build_alias_map(releases=_DEFAULT_RELEASES, target=_TARGET_RELEASE) -> pd.DataFrame:
    g_target = _gene_table(target)
    target_ids = set(g_target["gid"])
    target_seq = dict(zip(g_target["gid"], g_target["seqname"]))

    # name -> the UNIQUE primary-assembly gene id in the target release
    primary = g_target[g_target["seqname"].isin(_PRIMARY_CONTIGS)]
    name_counts = primary.groupby("gene_name")["gid"].nunique()
    unique_names = set(name_counts[name_counts == 1].index)
    name_to_primary = {
        nm: gid
        for nm, gid in zip(primary["gene_name"], primary["gid"])
        if isinstance(nm, str) and nm in unique_names
    }

    # newest-seen (name, seqname) for every human gene id across the scanned releases
    seen: dict[str, tuple[str, str]] = {}
    for rel in releases:
        g = _gene_table(rel)
        for gid, nm, sq in zip(g["gid"], g["gene_name"], g["seqname"]):
            if gid not in seen:
                seen[gid] = (nm, sq)

    rows: list[dict[str, str]] = []
    for gid, (nm, sq) in seen.items():
        # already a canonical primary-assembly gene in the target release -> not an alias
        if gid in target_ids and target_seq[gid] in _PRIMARY_CONTIGS:
            continue
        if not (isinstance(nm, str) and nm in name_to_primary):
            continue
        primary_id = name_to_primary[nm]
        if primary_id == gid:
            continue
        if sq not in _PRIMARY_CONTIGS:
            # non-primary contig copy (alt-haplotype / patch / scaffold) -> SUM class
            source = "ensembl_alt_contig"
        elif gid not in target_ids and target_seq.get(primary_id) == sq:
            # retired primary-chr id, same-named successor on the same chromosome
            source = "ensembl_id_history_name"
        else:
            continue
        rows.append(
            {
                "alt_haplotype_id": gid,
                "primary_contig_id": primary_id,
                "symbol": nm,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-release", type=int, default=_TARGET_RELEASE)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    generated = build_alias_map(target=args.target_release)

    # Preserve any curated rows this build does not reproduce (e.g. the 3
    # ensembl_archive_replacement rows, plus any rest-lookup entry not regenerated).
    if args.output.exists():
        existing = pd.read_csv(args.output, dtype=str).fillna("")
        keep = existing[~existing["alt_haplotype_id"].isin(set(generated["alt_haplotype_id"]))]
        out = pd.concat([generated, keep], ignore_index=True)
    else:
        out = generated
    out = out.drop_duplicates("alt_haplotype_id").sort_values("alt_haplotype_id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote {len(out):,} alias rows to {args.output}")
    print(out["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
