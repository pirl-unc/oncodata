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

    raw per-sample TPM matrices  (candidate source cohorts per cancer code)
        -> select ONE source per code  (the source-matrices.csv choice; never pool)
        -> clean_tpm                   (two-compartment biological view)
        -> drop technical genes        (biology-only, matching the shipped artifact)
        -> percentile vectors / n=5 representatives / within-sample top-fractions

Input matrices are discovered under ``--cache`` as
``<cohort>/derived/<NAME>_per_sample_tpm.parquet``. The derived ``<NAME>`` is mapped
to a cancer code case-insensitively (``tcga_acc`` -> ``ACC``, ``LAML_ELNadv`` kept),
matched against the reference codes in ``--ref`` (a dir of ``<CODE>.parquet`` whose
names define the canonical casing). A code with several candidate source cohorts is
resolved to the single one recorded in ``source-matrices.csv`` (pirlygenes selects
one source per code; it never pools) — so the artifacts match the shipped reference.

Outputs land under ``--out`` (a staging dir, NOT ``cancerdata/data`` — the artifacts
are large and ship via the release tarball, so they're never committed):

    <out>/clean/<CODE>.parquet                                 (clean-TPM matrix, full)
    <out>/cancer-reference-expression-percentiles/<CODE>.parquet      (biology-only)
    <out>/cancer-reference-expression-representatives/<CODE>.parquet + _provenance.csv
    <out>/cancer-reference-expression-within-sample-top5/<CODE>.parquet (biology-only)

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
from cancerdata.gene_families import clean_tpm_censored_gene_ids
from cancerdata.normalization import clean_tpm
from cancerdata.source_matrices import registry as source_registry

_BASE = ["Ensembl_Gene_ID", "Symbol"]


def _code_key(name: str) -> str:
    """Normalize a matrix/reference name to a case-insensitive join key."""
    n = name[5:] if name.lower().startswith("tcga_") else name
    return n.replace("_", "").lower()


def _source_key(name: str) -> str:
    """Normalize a cohort directory or registry source-cohort id to a join key —
    by GSE accession when present, else alphanumeric-only (so ``treehouse-polya-25-01``
    and ``TREEHOUSE_POLYA_25_01`` match, and ``gse75885-sarc`` matches its registry
    id ``GSE75885_DELESPAUL_2017`` on the shared GSE accession)."""
    import re

    m = re.search(r"GSE\d+", name.upper())
    return m.group() if m else re.sub(r"[^A-Z0-9]", "", name.upper())


def discover(cache: Path, ref: Path) -> dict[str, list[tuple[str, Path]]]:
    """Map each reference cancer code to its candidate ``(cohort_dir, matrix path)``."""
    ref_by_key = {_code_key(p.stem): p.stem for p in ref.glob("*.parquet")}
    by_code: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    unmatched = []
    for m in cache.glob("*/derived/*_per_sample_tpm.parquet"):
        cohort_dir = m.parent.parent.name
        stem = m.name.replace("_per_sample_tpm.parquet", "")
        code = ref_by_key.get(_code_key(stem))
        if code is None:
            unmatched.append(stem)
            continue
        by_code[code].append((cohort_dir, m))
    if unmatched:
        print(f"  note: {len(unmatched)} matrices matched no reference code: {unmatched}")
    return dict(by_code)


def _select_source(code: str, candidates: list[tuple[str, Path]], code_to_source: dict) -> Path:
    """Pick the single source matrix for a code — never pool.

    pirlygenes selects exactly one source cohort per code (RNA-seq over microarray
    proxy, then a primary-tumor source, then most samples); cancerdata's shipped
    ``source-matrices.csv`` already records that choice as ``code -> source_cohort``.
    So with a single candidate we use it; with several we keep the one whose cohort
    directory matches the registry's source_cohort. This replaces the old concat-pool
    (which over-counted multi-source codes) — single-source codes were always a no-op."""
    if len(candidates) == 1:
        return candidates[0][1]
    src = code_to_source.get(code)
    if src is not None:
        want = _source_key(src)
        hits = [p for d, p in candidates if _source_key(d) == want]
        if len(hits) == 1:
            return hits[0]
    # Fall back to the most-sampled source so a registry miss still picks one source,
    # never a pool. (Not expected for the shipped registry.)
    print(
        f"  warn: {code} has {len(candidates)} sources, no unique registry match; "
        f"using the largest",
        flush=True,
    )
    return max(candidates, key=lambda c: pd.read_parquet(c[1]).shape[1])[1]


def build_clean(path: Path) -> pd.DataFrame:
    """Clean-TPM matrix for one source's per-sample matrix (genes x samples + ids)."""
    raw = pd.read_parquet(path)
    samples = [c for c in raw.columns if c not in _BASE]
    gene_table = raw[_BASE]
    clean = clean_tpm(raw[samples], gene_table=gene_table)
    return pd.concat([gene_table.reset_index(drop=True), clean.reset_index(drop=True)], axis=1)


def _drop_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the clean-TPM censored (technical + ribosomal) genes, so the percentile /
    within-sample artifacts describe the biological view pirlygenes ships. clean_tpm
    has already deflated these into the technical compartment; dropping the rows
    doesn't change any biological gene's percentile (they're per-row independent)."""
    censored = clean_tpm_censored_gene_ids()
    unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    return df[~unversioned.isin(censored)].reset_index(drop=True)


def rebuild(cache: Path, ref: Path, out: Path, *, limit: int | None, validate: bool) -> None:
    by_code = discover(cache, ref)
    reg = source_registry()
    code_to_source = dict(zip(reg["cancer_code"].astype(str), reg["source_cohort"].astype(str)))
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
        source_path = _select_source(code, by_code[code], code_to_source)
        clean_df = build_clean(source_path)
        samples = [c for c in clean_df.columns if c not in _BASE]
        clean_df.to_parquet(clean_dir / f"{code}.parquet", index=False, compression="zstd")

        # Biological view (technical genes dropped) for the percentile + within-sample
        # artifacts, matching pirlygenes' shipped biology-only artifacts.
        bio_df = _drop_technical(clean_df)
        pct = cohort_percentile_vectors(bio_df, samples)
        pct.to_parquet(pct_dir / f"{code}.parquet", index=False, compression="zstd")

        # Representatives keep the full gene set (real per-sample vectors).
        reps = cohort_medoids(clean_df, k=5)
        rep_cols = [c for c in reps.columns if c not in _BASE]
        rep_ids = [f"{code}__rep{i}" for i in range(1, len(rep_cols) + 1)]
        reps = reps.rename(columns=dict(zip(rep_cols, rep_ids)))
        reps.to_parquet(rep_dir / f"{code}.parquet", index=False, compression="zstd")
        for rep_id in rep_ids:
            provenance.append(
                {
                    "representative_id": rep_id,
                    "source_cohort": code_to_source.get(code, code),
                    "n_cohort_samples": len(samples),
                }
            )

        ws = within_sample_top_fractions(bio_df, samples)
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
