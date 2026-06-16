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

"""Regenerate the HPA-derived columns of the bundled ``cancer-testis-antigens.csv``.

Re-derives every RNA / protein / restriction / filter column for every existing
row from the current pinned HPA release (v23: ``rna_tissue_consensus`` for RNA,
``normal_tissue`` for IHC), using :func:`oncodata.cta_regen.regenerate_cta_columns`.
The HPA tables are downloaded + cached on first use via the oncodata accessors.

The gene list and identity/annotation columns (Symbol, Ensembl_Gene_ID, Aliases,
Full_Name, Function, source_databases, Canonical_Transcript_ID, biotype) are
preserved verbatim -- this is HPA-only (no MS, no pyensembl).

Safe by default: writes a side-by-side ``*.regen.csv`` sidecar and prints a
per-column delta report against the shipped table, but does NOT overwrite the
bundled CSV unless ``--apply`` is passed.

Usage::

    python scripts/regenerate_cta_table.py            # dry run: sidecar + delta report
    python scripts/regenerate_cta_table.py --apply     # overwrite the bundled CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from oncodata.cta_regen import (  # noqa: E402
    PRESERVED_COLUMNS,
    RECOMPUTED_COLUMNS,
    regenerate_cta_columns,
)

CSV_PATH = _REPO_ROOT / "oncodata" / "data" / "cancer-testis-antigens.csv"


def _report(old: pd.DataFrame, new: pd.DataFrame, columns: list[str]) -> None:
    def boolcol(df: pd.DataFrame, c: str) -> pd.Series:
        return df.set_index("Symbol")[c].astype(str).str.lower().eq("true")

    print("=" * 78)
    print("CTA REGENERATION DELTA REPORT  (current HPA vs shipped table)")
    print("=" * 78)
    print(f"rows: {len(old)}")

    for col in ["passes_filters", "never_expressed"]:
        if col not in old.columns or col not in new.columns:
            continue
        o, n = boolcol(old, col), boolcol(new, col)
        common = o.index.intersection(n.index)
        flips = common[o[common] != n[common]]
        gained = sorted(flips[(~o[flips]) & n[flips]])
        lost = sorted(flips[o[flips] & (~n[flips])])
        print(f"\n{col}: {len(flips)} flips   +{len(gained)} / -{len(lost)}")
        if gained:
            print(f"  now TRUE : {', '.join(gained)}")
        if lost:
            print(f"  now FALSE: {', '.join(lost)}")

    print("\nper-column value changes (cells differing):")
    o_idx, n_idx = old.set_index("Symbol"), new.set_index("Symbol")
    common = o_idx.index.intersection(n_idx.index)
    any_change = False
    for col in columns:
        if col == "Symbol" or col not in n_idx.columns or col not in o_idx.columns:
            continue
        oc = o_idx.loc[common, col].astype(str)
        nc = n_idx.loc[common, col].astype(str)
        diff = int((oc != nc).sum())
        if diff:
            any_change = True
            marker = " (preserved!)" if col in PRESERVED_COLUMNS else ""
            print(f"  {col:42s} {diff:4d} changed{marker}")
    if not any_change:
        print("  (no cell differences -- regeneration reproduces the shipped table)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Overwrite the bundled CSV in place.")
    args = parser.parse_args()

    old = pd.read_csv(CSV_PATH)
    columns = list(old.columns)

    print(
        f"Recomputing {len(RECOMPUTED_COLUMNS)} HPA columns; preserving {len(PRESERVED_COLUMNS)}."
    )
    new = regenerate_cta_columns(old)
    new = new[columns]

    # Write first, then diff against the round-tripped CSV so the report reflects
    # what actually lands on disk (avoids spurious in-memory dtype "changes",
    # e.g. bool True vs str "True", NaN vs "").
    dest = CSV_PATH if args.apply else CSV_PATH.with_suffix(".regen.csv")
    new.to_csv(dest, index=False)
    _report(old, pd.read_csv(dest), columns)

    if args.apply:
        print(f"\n--apply: wrote {len(new)} rows to {CSV_PATH}")
    else:
        print(f"\n(dry run) wrote regenerated table to {dest}; shipped CSV untouched.")
        print("Re-run with --apply to overwrite the bundled table.")


if __name__ == "__main__":
    main()
