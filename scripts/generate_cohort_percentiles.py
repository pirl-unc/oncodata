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

"""Build the per-gene × cohort *percentile-vector* artifact (within-cohort axis).

For each cohort with a per-sample expression matrix, reduce every gene's
distribution across that cohort's samples to a dense set of percentile
breakpoints (p0…p100, dense in the upper tail). This is the within-cohort,
across-samples axis — "where does this gene sit in the cohort's distribution" —
the complement of the within-sample artifact. A consumer can then place a new
sample's gene as a percentile rank within the cohort instead of an absolute TPM.

It is computed from the full per-sample matrices, which are never shipped (see
``source_matrices`` for the per-cohort fetch). Pass ``--drop-genes`` with the
clean-TPM censored-gene list so the breakpoints describe the biological view the
reader expects (``oncoref.gene_families.clean_tpm_censored_gene_ids``).

Input
-----
A directory of per-cohort parquet files, one per cohort code
(``<INPUT>/<CODE>.parquet``), each with ``Ensembl_Gene_ID`` + ``Symbol`` columns
and one column per sample (clean TPM).

Output
------
``oncoref/data/cancer-reference-expression-percentiles/<CODE>.parquet`` with
``Ensembl_Gene_ID, Symbol`` and 26 ``p{n}`` columns, stored log1p + float16 — the
exact encoding ``expression.cohort_gene_percentiles`` restores with ``expm1``.

Run:
    python scripts/generate_cohort_percentiles.py --input <per-sample-dir>

After building, ensure ``cancer-reference-expression-percentiles`` is in
``data_bundle.DOWNLOADABLE_PATHS``, rebuild + upload the data tarball, and bump
``DATA_VERSION`` (never bump it before the tarball is uploaded — a 404 hangs fetch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression import SHARD_DATASETS
from oncoref.expression_builders import cohort_percentile_vectors, sample_columns

_DATASET = SHARD_DATASETS["percentiles"]  # the reader's record — derive dirs from it so
_DATA_DIR = Path(__file__).resolve().parents[1] / "oncoref" / "data"  # producer/reader can't drift
OUT_DIR = _DATA_DIR / _DATASET.gene_dir


def _proteoform_out_dir(scope: str) -> Path:
    """The scope-specific proteoform shard directory the reader expects."""
    return _DATA_DIR / _DATASET.subdir(proteoform=True, scope=scope)


def _load_drop_genes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build(
    input_dir: Path,
    *,
    drop_genes: set[str],
    out_dir: Path | None = None,
    proteoform: bool = False,
    scope: str = "cta",
) -> None:
    """Build the percentile-vector artifact for each cohort under ``input_dir``.

    With ``proteoform=True``, each cohort's per-sample matrix is collapsed to the
    proteoform key space (identical-protein members summed, in ``scope``) *before* the
    percentiles are computed, so the vector is one row per proteoform key. Output lands
    in the scope-specific ``…-percentiles-proteoform-<scope>`` directory the reader
    expects (see :meth:`oncoref.expression.ShardDataset.subdir`).
    """
    if out_dir is None:
        out_dir = _proteoform_out_dir(scope) if proteoform else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(input_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no per-sample parquet files under {input_dir}")
    n = 0
    for shard in shards:
        code = shard.stem
        df = pd.read_parquet(shard)
        if drop_genes:
            df = df[~df["Ensembl_Gene_ID"].astype(str).isin(drop_genes)].reset_index(drop=True)
        cols = sample_columns(df)
        if not cols:
            print(f"  {code}: no sample columns, skipped", flush=True)
            continue
        if proteoform:
            # Collapse identical-protein members per sample first, then rank on the
            # reduced proteoform key space (one reusable collapse + proteoform_key).
            from oncoref.proteoforms import collapse_to_proteoforms

            df = collapse_to_proteoforms(df, scope=scope, sample_cols=cols)
            cols = sample_columns(df)
        out = cohort_percentile_vectors(df, cols)
        out.to_parquet(out_dir / f"{code}.parquet", index=False, compression="zstd")
        n += 1
        print(f"  {code}: {len(out)} genes (n={len(cols)})", flush=True)
    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.parquet")) / 1e6
    print(f"\ndone: {n} cohorts, {total_mb:.1f} MB -> {out_dir}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", required=True, type=Path, help="Dir of per-cohort per-sample parquet files"
    )
    p.add_argument(
        "--drop-genes",
        type=Path,
        default=None,
        help="Optional newline-delimited Ensembl IDs of technical genes to drop",
    )
    p.add_argument(
        "--proteoform",
        action="store_true",
        help="Collapse identical-protein members before ranking (proteoform key space)",
    )
    p.add_argument(
        "--scope",
        default="cta",
        choices=("cta", "genome"),
        help="Proteoform-collapse scope (only with --proteoform); writes a scope-specific dir",
    )
    args = p.parse_args(argv)
    build(
        args.input,
        drop_genes=_load_drop_genes(args.drop_genes),
        proteoform=args.proteoform,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
