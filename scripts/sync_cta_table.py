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

"""Re-sync the bundled CTA table from tsarina's authoritative HPA-derived table.

tsarina owns the cancer-testis-antigen regeneration (``scripts/add_cta_gene.py``,
HPA-derived; it weighs paralogs and normal-tissue expression). cancerdata ships a
**projection** of that table: the HPA-only evidence columns, *without* tsarina's
mass-spec columns (``ms_restriction``/``ms_pmids``/``ms_healthy_somatic_tissues``),
which belong to the target-selection layer, not the reference-data layer.

This keeps cancerdata's copy current without duplicating the HPA pipeline. Until
the regeneration itself moves here (the larger half of issue #16), run this after
a tsarina CTA update::

    python scripts/sync_cta_table.py --source ../tsarina/tsarina/data/cancer-testis-antigens.csv

The committed ``cancer-testis-antigens.csv`` is what ships; this script only
regenerates it. ``CANONICAL_COLUMNS`` is the contract — every column cancerdata's
``cta`` module reads must appear here and in the source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEST = _REPO_ROOT / "cancerdata" / "data" / "cancer-testis-antigens.csv"
_DEFAULT_SOURCE = _REPO_ROOT.parent / "tsarina" / "tsarina" / "data" / "cancer-testis-antigens.csv"

# The HPA-only projection cancerdata ships, in shipped column order. This is the
# schema contract: tsarina's mass-spec columns are intentionally excluded.
CANONICAL_COLUMNS = [
    "Symbol",
    "Aliases",
    "Full_Name",
    "Function",
    "Ensembl_Gene_ID",
    "source_databases",
    "protein_reproductive",
    "protein_thymus",
    "protein_reliability",
    "rna_reproductive",
    "rna_thymus",
    "protein_strict_expression",
    "rna_reproductive_frac",
    "rna_reproductive_and_thymus_frac",
    "rna_deflated_reproductive_frac",
    "rna_deflated_reproductive_and_thymus_frac",
    "Canonical_Transcript_ID",
    "biotype",
    "rna_80_pct_filter",
    "rna_90_pct_filter",
    "rna_95_pct_filter",
    "rna_97_pct_filter",
    "rna_98_pct_filter",
    "rna_99_pct_filter",
    "passes_filters",
    "rna_max_ntpm",
    "never_expressed",
    "rna_testis_ntpm",
    "rna_ovary_ntpm",
    "rna_placenta_ntpm",
    "rna_max_somatic_tissue",
    "rna_max_somatic_ntpm",
    "rna_somatic_detected_count",
    "rna_brain_max_ntpm",
    "rna_heart_max_ntpm",
    "rna_lung_max_ntpm",
    "rna_liver_max_ntpm",
    "rna_pancreas_max_ntpm",
    "protein_restriction",
    "protein_testis",
    "protein_ovary",
    "protein_placenta",
    "rna_restriction",
    "rna_restriction_level",
    "restriction",
    "restriction_confidence",
    "safety_flags",
]


def sync(source: Path, dest: Path = _DEST) -> pd.DataFrame:
    """Project tsarina's CTA table onto cancerdata's canonical columns, write it."""
    if not source.exists():
        raise SystemExit(
            f"tsarina CTA table not found at {source} — pass --source <path to a "
            f"tsarina checkout's tsarina/data/cancer-testis-antigens.csv>."
        )
    src = pd.read_csv(source)
    missing = [c for c in CANONICAL_COLUMNS if c not in src.columns]
    if missing:
        raise SystemExit(
            f"tsarina table is missing {len(missing)} canonical column(s): {missing}. "
            f"The upstream schema changed — reconcile CANONICAL_COLUMNS before syncing."
        )
    out = src[CANONICAL_COLUMNS].copy()

    # tsarina computes restriction/confidence WITH its mass-spec evidence; cancerdata
    # ships the HPA-only synthesis (no MS leakage — see cta.synthesize_restriction and
    # test_shipped_restriction_is_hpa_only_synthesis). Re-derive both from HPA columns
    # so the shipped values match the data cancerdata actually owns.
    from cancerdata.cta import synthesize_restriction

    derived = out.apply(synthesize_restriction, axis=1, result_type="expand")
    n_changed = int((out["restriction_confidence"].astype(str) != derived[1]).sum())
    out["restriction"], out["restriction_confidence"] = derived[0], derived[1]
    out.to_csv(dest, index=False)
    print(
        f"wrote {len(out)} rows × {len(CANONICAL_COLUMNS)} cols -> {dest} "
        f"(re-derived HPA-only confidence on {n_changed} rows that carried MS-aware values)",
        flush=True,
    )
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE,
        help="Path to tsarina's authoritative cancer-testis-antigens.csv",
    )
    args = p.parse_args(argv)
    sync(args.source)


if __name__ == "__main__":
    sys.exit(main())
