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

"""End-to-end rebuild of the per-cohort expression artifacts from per-sample matrices.

Ties the pieces together so cancerdata can **regenerate** (not just hold) its
expression bundle:

    raw per-sample TPM matrices  (one+ per cancer code, possibly multi-source)
        -> pool samples per code  (outer-join on gene; not-measured stays NaN)
        -> clean_tpm               (two-compartment biological view)
        -> percentile vectors / n=5 representatives / within-sample top-fractions

Input matrices are discovered under ``--cache`` as
``<cohort>/derived/<NAME>_per_sample_tpm.parquet``. The derived ``<NAME>`` is mapped
to a cancer code case-insensitively (``tcga_acc`` -> ``ACC``, ``LAML_ELNadv`` kept),
matched against the reference codes in ``--ref`` (a dir of ``<CODE>.parquet`` whose
names define the canonical casing). Several matrices mapping to one code are pooled.

Outputs land under ``--out`` (a staging dir, NOT ``cancerdata/data`` — the artifacts
are large and ship via the release tarball, so they're never committed):

    <out>/clean/<CODE>.parquet                                 (pooled clean-TPM matrix)
    <out>/cancer-reference-expression-percentiles/<CODE>.parquet
    <out>/cancer-reference-expression-representatives/<CODE>.parquet + _provenance.csv
    <out>/cancer-reference-expression-within-sample-top5/<CODE>.parquet

``--validate`` additionally correlates each rebuilt percentile vector against the
reference artifact in ``--ref`` and prints the per-code agreement.

Run:
    python scripts/rebuild_expression_artifacts.py \
        --cache ~/.cache/pirlygenes/expression \
        --ref   ~/code/pirlygenes/pirlygenes/data/cancer-reference-expression-percentiles \
        --out   ~/.cache/cancerdata/rebuild-staging [--limit N] [--validate]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancerdata._build import cohort_medoids, cohort_percentile_vectors, within_sample_top_fractions
from cancerdata.normalization import clean_tpm

_BASE = ["Ensembl_Gene_ID", "Symbol"]


def _code_key(name: str) -> str:
    """Normalize a matrix/reference name to a case-insensitive join key."""
    n = name[5:] if name.lower().startswith("tcga_") else name
    return n.replace("_", "").lower()


def discover(cache: Path, ref: Path) -> dict[str, list[Path]]:
    """Map each reference cancer code to its one-or-more per-sample matrices."""
    ref_by_key = {_code_key(p.stem): p.stem for p in ref.glob("*.parquet")}
    by_code: dict[str, list[Path]] = defaultdict(list)
    unmatched = []
    for m in cache.glob("*/derived/*_per_sample_tpm.parquet"):
        stem = m.name.replace("_per_sample_tpm.parquet", "")
        code = ref_by_key.get(_code_key(stem))
        if code is None:
            unmatched.append(stem)
            continue
        by_code[code].append(m)
    if unmatched:
        print(f"  note: {len(unmatched)} matrices matched no reference code: {unmatched}")
    return dict(by_code)


def pool_matrices(paths: list[Path]) -> pd.DataFrame:
    """Pool one+ per-sample matrices for a code: outer-join on gene id, concat
    sample columns. A gene absent in one source stays NaN there (not-measured is
    not zero). Sample columns are uniquified by source stem to avoid collisions."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        samples = [c for c in df.columns if c not in _BASE]
        if len(paths) > 1:
            tag = p.name.replace("_per_sample_tpm.parquet", "")
            df = df.rename(columns={s: f"{tag}:{s}" for s in samples})
        frames.append(df)
    if len(frames) == 1:
        return frames[0]
    merged = frames[0]
    for nxt in frames[1:]:
        merged = merged.merge(nxt, on="Ensembl_Gene_ID", how="outer", suffixes=("", "_dup"))
        # collapse duplicated Symbol column from the join
        if "Symbol_dup" in merged.columns:
            merged["Symbol"] = merged["Symbol"].fillna(merged["Symbol_dup"])
            merged = merged.drop(columns=["Symbol_dup"])
    return merged


def build_clean(paths: list[Path]) -> pd.DataFrame:
    """Pooled, clean-TPM matrix for a code (genes x samples + id cols)."""
    pooled = pool_matrices(paths)
    samples = [c for c in pooled.columns if c not in _BASE]
    gene_table = pooled[_BASE]
    clean = clean_tpm(pooled[samples], gene_table=gene_table)
    return pd.concat([gene_table.reset_index(drop=True), clean.reset_index(drop=True)], axis=1)


def rebuild(cache: Path, ref: Path, out: Path, *, limit: int | None, validate: bool) -> None:
    by_code = discover(cache, ref)
    codes = sorted(by_code)
    if limit:
        codes = codes[:limit]
    print(f"rebuilding {len(codes)} cohorts -> {out}", flush=True)

    clean_dir = out / "clean"
    pct_dir = out / "cancer-reference-expression-percentiles"
    rep_dir = out / "cancer-reference-expression-representatives"
    ws_dir = out / "cancer-reference-expression-within-sample-top5"
    for d in (clean_dir, pct_dir, rep_dir, ws_dir):
        d.mkdir(parents=True, exist_ok=True)

    provenance: list[dict] = []
    corrs: list[float] = []
    for code in codes:
        clean_df = build_clean(by_code[code])
        samples = [c for c in clean_df.columns if c not in _BASE]
        clean_df.to_parquet(clean_dir / f"{code}.parquet", index=False, compression="zstd")

        pct = cohort_percentile_vectors(clean_df, samples)
        pct.to_parquet(pct_dir / f"{code}.parquet", index=False, compression="zstd")

        reps = cohort_medoids(clean_df, k=5)
        rep_cols = [c for c in reps.columns if c not in _BASE]
        rep_ids = [f"{code}__rep{i}" for i in range(1, len(rep_cols) + 1)]
        reps = reps.rename(columns=dict(zip(rep_cols, rep_ids)))
        reps.to_parquet(rep_dir / f"{code}.parquet", index=False, compression="zstd")
        for rep_id in rep_ids:
            provenance.append(
                {
                    "representative_id": rep_id,
                    "source_cohort": code,
                    "n_cohort_samples": len(samples),
                }
            )

        ws = within_sample_top_fractions(clean_df, samples)
        ws.to_parquet(ws_dir / f"{code}.parquet", index=False, compression="zstd")

        msg = f"  {code}: {len(samples)} samples, {len(pct)} genes"
        if validate:
            corr = _validate_one(pct, ref / f"{code}.parquet")
            if corr is not None:
                corrs.append(corr)
                msg += f"  p95-corr={corr:.4f}"
        print(msg, flush=True)

    pd.DataFrame(provenance).to_csv(rep_dir / "_provenance.csv", index=False)
    if validate and corrs:
        # nan-robust: a cohort whose reference vector is constant/empty yields a
        # nan correlation that must not poison the summary.
        arr = np.array(corrs)
        finite = arr[~np.isnan(arr)]
        nan_n = len(arr) - len(finite)
        ge99 = int((finite >= 0.99).sum())
        worst = sorted(zip(codes, corrs), key=lambda t: (np.isnan(t[1]), t[1]))[:8]
        print(
            f"\nvalidation: {len(finite)} cohorts vs reference ({nan_n} undefined)  "
            f"p95-corr median={np.median(finite):.4f} mean={finite.mean():.4f} "
            f"min={finite.min():.4f}  (>={0.99}: {ge99}/{len(finite)})",
            flush=True,
        )
        print("  lowest agreement: " + ", ".join(f"{c}={v:.3f}" for c, v in worst), flush=True)
    print(f"\ndone -> {out}", flush=True)


def _validate_one(pct: pd.DataFrame, ref_path: Path) -> float | None:
    """Pearson correlation of rebuilt vs reference p95 (in TPM space)."""
    if not ref_path.exists():
        return None
    ref = pd.read_parquet(ref_path).set_index("Ensembl_Gene_ID")
    mine = pct.set_index("Ensembl_Gene_ID")
    common = mine.index.intersection(ref.index)
    if len(common) < 100 or "p95" not in ref.columns:
        return None
    a = np.expm1(mine.loc[common, "p95"].astype("float32").to_numpy())
    b = np.expm1(ref.loc[common, "p95"].astype("float32").to_numpy())
    mask = (a > 0) | (b > 0)
    if mask.sum() < 100:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path, help="Per-sample matrix cache root")
    p.add_argument(
        "--ref", required=True, type=Path, help="Reference percentile dir (defines codes)"
    )
    p.add_argument(
        "--out", required=True, type=Path, help="Staging output dir (not cancerdata/data)"
    )
    p.add_argument("--limit", type=int, default=None, help="Only the first N codes (a test run)")
    p.add_argument("--validate", action="store_true", help="Correlate vs the reference artifacts")
    args = p.parse_args(argv)
    rebuild(
        args.cache.expanduser(),
        args.ref.expanduser(),
        args.out.expanduser(),
        limit=args.limit,
        validate=args.validate,
    )


if __name__ == "__main__":
    main()
