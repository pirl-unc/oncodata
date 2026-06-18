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

"""Derive the **canonical gene-ID space** — the authoritative target set the
alias/migration map (``ensembl-id-aliases.csv``) resolves *into* (oncoref#135 item 4).

One row per Ensembl gene on a **primary chromosome** (``1``–``22``, ``X``, ``Y``,
``MT``) in the target release, with its single canonical symbol and biotype, so a
consumer can (a) check membership in the minimal harmonized space and (b) filter to
the ~20k protein-coding genes, dropping the large mostly-zero ncRNA / pseudogene tail
(pirlygenes#465). Alt-haplotype/patch contigs are excluded — those collapse onto a
primary gene via the alias map, so they are never their own canonical entry.

Versioned by the Ensembl release it derives from (``--target-release``, default 115);
the release is recorded in the ``ensembl_release`` column for auditability.

pyensembl is a BUILD-TIME tool only (cf. the alias/proteoform generators). Run:

    python scripts/generate_canonical_gene_space.py

Output: ``oncoref/data/canonical-gene-space.csv.gz`` with columns
``ensembl_gene_id, symbol, biotype, seqname, ensembl_release``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "oncoref" / "data" / "canonical-gene-space.csv.gz"
_TARGET_RELEASE = 115
_PRIMARY_CONTIGS = frozenset([*(str(i) for i in range(1, 23)), "X", "Y", "MT"])


def build_canonical_gene_space(target: int = _TARGET_RELEASE) -> pd.DataFrame:
    import pyensembl

    er = pyensembl.EnsemblRelease(target, species="human")
    df = pd.read_sql("SELECT gene_id, gene_name, seqname, gene_biotype FROM gene", er.db.connection)
    df["ensembl_gene_id"] = df["gene_id"].astype(str).str.split(".", n=1).str[0].str.strip()
    df["seqname"] = df["seqname"].astype(str)
    df = df[df["seqname"].isin(_PRIMARY_CONTIGS)].copy()
    out = pd.DataFrame(
        {
            "ensembl_gene_id": df["ensembl_gene_id"],
            "symbol": df["gene_name"].fillna("").astype(str),
            "biotype": df["gene_biotype"].fillna("").astype(str),
            "seqname": df["seqname"],
            "ensembl_release": target,
        }
    )
    # gene_id is a primary key in the GTF, but be defensive about any duplication.
    out = out.drop_duplicates("ensembl_gene_id").sort_values("ensembl_gene_id")
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-release", type=int, default=_TARGET_RELEASE)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    space = build_canonical_gene_space(args.target_release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    space.to_csv(args.output, index=False, compression="gzip")

    print(f"Wrote {len(space):,} canonical genes (Ensembl {args.target_release}) to {args.output}")
    print(space["biotype"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
