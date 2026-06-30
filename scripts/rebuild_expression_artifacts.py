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

Ties the pieces together so oncoref can **regenerate** (not just hold) its
expression bundle:

    raw per-sample TPM matrices  (candidate source cohorts per cancer code)
        -> select ONE source per code  (the source-matrices.csv choice; never pool)
        -> clean_tpm                   (two-compartment biological view)
        -> sample QC filter            (pass by default; opt out with --sample-qc all)
        -> drop technical genes        (biology-only, matching the shipped artifact)
        -> percentile vectors / n=5 representatives / within-sample top-fractions

Input matrices are discovered under ``--cache`` as
``<cohort>/derived/<NAME>_per_sample_tpm.parquet``. The derived ``<NAME>`` is mapped
to a cancer code case-insensitively (``tcga_acc`` -> ``ACC``, ``LAML_ELNadv`` kept),
matched against the reference codes in ``--ref`` (a dir of ``<CODE>.parquet`` whose
names define the canonical casing). A code with several candidate source cohorts is
resolved to the single one recorded in ``source-matrices.csv`` (pirlygenes selects
one source per code; it never pools) — so the artifacts match the shipped reference.

Outputs land under ``--out`` (a staging dir, NOT ``oncoref/data`` — the artifacts
are large and ship via the release tarball, so they're never committed):

    <out>/clean/<CODE>.parquet                                 (QC-filtered clean TPM)
    <out>/cancer-reference-expression-percentiles/<CODE>.parquet      (biology-only)
    <out>/cancer-reference-expression-representatives/<CODE>.parquet + _provenance.csv
    <out>/cancer-reference-expression-within-sample-top5/<CODE>.parquet (biology-only)
    <out>/source-matrix-sample-qc.csv                          (per-sample QC manifest)
    <out>/expression-artifact-build-metadata.json              (QC/build policy metadata)

``--validate`` additionally correlates each rebuilt percentile vector against the
reference artifact in ``--ref`` and prints the per-code agreement.

Run:
    python scripts/rebuild_expression_artifacts.py \
        --cache ~/.cache/pirlygenes/expression \
        --ref   ~/code/pirlygenes/pirlygenes/data/cancer-reference-expression-percentiles \
        --out   ~/.cache/oncoref/rebuild-staging [--limit N] [--validate]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.cancer_types import cohort_source_version
from oncoref.expression import (
    SAMPLE_EXPRESSION_QC_POLICY_VERSION,
    SHARD_DATASETS,
    _canonicalize_gene_rows,
    _validate_sample_qc,
    sample_columns,
    sample_expression_qc_from_matrix,
)
from oncoref.expression_builders import (
    cohort_medoids,
    cohort_percentile_vectors,
    within_sample_top_fractions,
)
from oncoref.gene_families import clean_tpm_censored_gene_ids
from oncoref.normalization import clean_tpm
from oncoref.source_matrices import registry as source_registry

# Rebuilt artifacts must land in the exact directories the reader resolves; derive every
# name from the shared registry so producer and reader can't drift. Proteoform shards are
# built at "cta" scope (the only scope generated today).
_PCT_DS = SHARD_DATASETS["percentiles"]
_WS_DS = SHARD_DATASETS["within_sample"]
_PROTEOFORM_SCOPE = "cta"

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


def _parquet_sample_count(path: Path) -> int:
    """Fast sample-column count from parquet metadata."""
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    return len([c for c in names if c not in _BASE])


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


def _select_source(
    code: str,
    candidates: list[tuple[str, Path]],
    code_to_source: dict,
    code_to_n_samples: dict | None = None,
) -> Path:
    """Pick the single source matrix for a code — never pool.

    pirlygenes selects exactly one source cohort per code (RNA-seq over microarray
    proxy, then a primary-tumor source, then most samples); oncoref's shipped
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
        # Some source registry rows name an aggregate release with a year suffix
        # (e.g. GEO_HEME_2022), while the staged cache directory uses the family
        # label (geo-heme). Prefer that explicit aggregate directory over
        # duplicate per-study cache aliases when it is unique.
        want_without_trailing_digits = want.rstrip("0123456789")
        if want_without_trailing_digits and want_without_trailing_digits != want:
            hits = [
                p for d, p in candidates if _source_key(d) == want_without_trailing_digits
            ]
            if len(hits) == 1:
                return hits[0]
    if code_to_n_samples:
        want_n = code_to_n_samples.get(code)
        if want_n is not None:
            count_hits = []
            for _d, p in candidates:
                if _parquet_sample_count(p) == want_n:
                    count_hits.append(p)
            if len(count_hits) == 1:
                return count_hits[0]
    # Fall back to the most-sampled source so a registry miss still picks one source,
    # never a pool. (Not expected for the shipped registry.)
    print(
        f"  warn: {code} has {len(candidates)} sources, no unique registry match; "
        f"using the largest",
        flush=True,
    )
    return max(candidates, key=lambda c: pd.read_parquet(c[1]).shape[1])[1]


def read_raw(path: Path) -> pd.DataFrame:
    """Canonical raw per-sample matrix for one source (genes x samples + ids).

    Canonicalize the raw matrix first (sum alt-haplotype/patch copies in LINEAR TPM,
    relabel retired ids), exactly as the runtime `_load_per_sample_matrix` does, so the
    shipped percentile/representative shards are natively dense in the canonical gene-ID
    space and match the on-the-fly recompute path (oncoref#135 item 6)."""
    raw = pd.read_parquet(path)
    return _canonicalize_gene_rows(raw, sample_cols=sample_columns(raw)).reset_index(drop=True)


def _clip_negative_expression(df: pd.DataFrame, sample_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """TPM-like source matrices should be nonnegative; clip invalid negatives to zero."""
    if not sample_cols:
        return df, 0
    values = df[sample_cols]
    negative = values < 0
    n_negative = int(negative.to_numpy().sum())
    if not n_negative:
        return df, 0
    out = df.copy()
    out.loc[:, sample_cols] = values.clip(lower=0)
    return out, n_negative


def build_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean-TPM matrix for one source's per-sample matrix (genes x samples + ids)."""
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


def _sample_selection_for_qc(
    samples: list[str], qc: pd.DataFrame, sample_qc: str
) -> tuple[list[str], str, str]:
    mode = _validate_sample_qc(sample_qc)
    if mode == "all":
        return samples, "all", ""
    if qc.empty:
        return [], mode, ""
    status = dict(zip(qc["sample_id"].astype(str), qc["sample_qc_status"].astype(str)))
    if mode == "pass":
        allowed = {"pass"}
    else:
        allowed = {"pass", "warn"}
    selected = [s for s in samples if status.get(str(s)) in allowed]
    if selected or mode != "pass":
        return selected, mode, ""

    # Source-aware escape hatch: microarray/TPM-proxy sources are flagged warn
    # because their absolute scale is not linear RNA-seq TPM, but dropping the
    # entire cohort would erase a curated reference. Keep warn samples only when
    # every source sample is warn for the explicit proxy-scale reason.
    reasons = set()
    if "sample_qc_reasons" in qc.columns:
        for value in qc["sample_qc_reasons"].fillna("").astype(str):
            reasons.update(part for part in value.split(";") if part)
    scale_classes = set(qc.get("source_scale_class", pd.Series(dtype=str)).fillna("").astype(str))
    statuses = set(qc["sample_qc_status"].astype(str))
    if statuses <= {"warn"} and (
        "tpm_proxy_scale" in reasons or "microarray_tpm_proxy" in scale_classes
    ):
        warn_samples = [s for s in samples if status.get(str(s)) == "warn"]
        return warn_samples, "pass_or_warn", "no_pass_samples_tpm_proxy_source"
    concentration_only = {"high_top_gene_fraction", "high_top10_gene_fraction"}
    if statuses <= {"fail"} and reasons and reasons <= concentration_only:
        fail_samples = [s for s in samples if status.get(str(s)) == "fail"]
        return fail_samples, "all", "no_pass_samples_high_concentration_source"
    return [], mode, ""


def _qc_counts(qc: pd.DataFrame) -> dict[str, int]:
    counts = qc["sample_qc_status"].value_counts().to_dict() if not qc.empty else {}
    return {f"n_qc_{name}": int(counts.get(name, 0)) for name in ("pass", "warn", "fail")}


def rebuild(
    cache: Path,
    ref: Path,
    out: Path,
    *,
    limit: int | None,
    validate: bool,
    sample_qc: str = "pass",
) -> None:
    sample_qc = _validate_sample_qc(sample_qc)
    by_code = discover(cache, ref)
    reg = source_registry()
    code_to_source = dict(zip(reg["cancer_code"].astype(str), reg["source_cohort"].astype(str)))
    code_to_n_samples = (
        dict(zip(reg["cancer_code"].astype(str), reg["n_samples"].astype(int)))
        if "n_samples" in reg.columns
        else {}
    )
    codes = sorted(by_code)
    if limit:
        codes = codes[:limit]
    print(f"rebuilding {len(codes)} cohorts -> {out}", flush=True)

    clean_dir = out / "clean"
    pct_dir = out / _PCT_DS.gene_dir
    pct_pf_dir = out / _PCT_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE)
    rep_dir = out / SHARD_DATASETS["representatives"].gene_dir
    ws_dir = out / _WS_DS.gene_dir
    ws_pf_dir = out / _WS_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE)
    for d in (clean_dir, pct_dir, pct_pf_dir, rep_dir, ws_dir, ws_pf_dir):
        d.mkdir(parents=True, exist_ok=True)

    provenance: list[dict] = []
    qc_manifest: list[pd.DataFrame] = []
    build_rows: list[dict] = []
    corrs: list[float] = []
    for code in codes:
        source_path = _select_source(code, by_code[code], code_to_source, code_to_n_samples)
        raw_df = read_raw(source_path)
        source_samples = [c for c in raw_df.columns if c not in _BASE]
        raw_df, n_negative_values_clipped = _clip_negative_expression(raw_df, source_samples)
        qc = sample_expression_qc_from_matrix(raw_df, cancer_type=code)
        if not qc.empty:
            qc = qc.assign(source_matrix_path=str(source_path))
            qc_manifest.append(qc)
        samples, effective_sample_qc, sample_qc_fallback_reason = _sample_selection_for_qc(
            source_samples, qc, sample_qc
        )
        if not samples:
            raise ValueError(f"{code}: sample_qc={sample_qc!r} leaves no source samples")

        clean_df = build_clean(raw_df)
        clean_df = clean_df[[*_BASE, *samples]].copy()
        clean_df.to_parquet(clean_dir / f"{code}.parquet", index=False, compression="zstd")

        # Biological view (technical genes dropped) for the percentile, within-sample,
        # and representative-selection geometry, matching the shared contract:
        # choose representatives on biological signal but store full clean TPM vectors.
        bio_df = _drop_technical(clean_df)
        pct = cohort_percentile_vectors(bio_df, samples)
        pct.to_parquet(pct_dir / f"{code}.parquet", index=False, compression="zstd")

        # Representatives keep the full gene set (real per-sample vectors), while
        # the medoid/farthest-first distances are computed on biological genes only.
        reps = cohort_medoids(clean_df, sample_cols=samples, k=5, selection_df=bio_df)
        rep_cols = [c for c in reps.columns if c not in _BASE]
        rep_ids = [f"{code}__rep{i}" for i in range(1, len(rep_cols) + 1)]
        reps = reps.rename(columns=dict(zip(rep_cols, rep_ids)))
        reps.to_parquet(rep_dir / f"{code}.parquet", index=False, compression="zstd")
        source_version = cohort_source_version(code)
        qc_counts = _qc_counts(qc)
        build_row = {
            "cancer_code": code,
            "source_cohort": code_to_source.get(code, code),
            "source_version": source_version,
            "source_matrix_path": str(source_path),
            "sample_qc": sample_qc,
            "sample_qc_effective": effective_sample_qc,
            "sample_qc_fallback_reason": sample_qc_fallback_reason,
            "sample_qc_policy_version": SAMPLE_EXPRESSION_QC_POLICY_VERSION,
            "n_source_samples": len(source_samples),
            "n_cohort_samples": len(samples),
            "n_negative_values_clipped": n_negative_values_clipped,
            **qc_counts,
        }
        build_rows.append(build_row)
        for rep_id in rep_ids:
            provenance.append(
                {
                    "representative_id": rep_id,
                    "source_cohort": code_to_source.get(code, code),
                    "source_version": source_version,  # harmonized Ensembl release
                    "source_matrix_path": str(source_path),
                    "sample_qc": sample_qc,
                    "sample_qc_effective": effective_sample_qc,
                    "sample_qc_fallback_reason": sample_qc_fallback_reason,
                    "sample_qc_policy_version": SAMPLE_EXPRESSION_QC_POLICY_VERSION,
                    "n_source_samples": len(source_samples),
                    "n_cohort_samples": len(samples),
                    "n_negative_values_clipped": n_negative_values_clipped,
                    **qc_counts,
                }
            )

        ws = within_sample_top_fractions(bio_df, samples)
        ws.to_parquet(ws_dir / f"{code}.parquet", index=False, compression="zstd")

        # Proteoform key space: collapse identical-protein members per sample, then
        # build the same percentile + within-sample summaries on the reduced space so
        # every downstream read can compare/quantify/plot on one collapsed key space.
        from oncoref.proteoforms import collapse_to_proteoforms

        bio_pf = collapse_to_proteoforms(bio_df, scope=_PROTEOFORM_SCOPE, sample_cols=samples)
        pf_samples = sample_columns(bio_pf)
        cohort_percentile_vectors(bio_pf, pf_samples).to_parquet(
            pct_pf_dir / f"{code}.parquet", index=False, compression="zstd"
        )
        within_sample_top_fractions(bio_pf, pf_samples).to_parquet(
            ws_pf_dir / f"{code}.parquet", index=False, compression="zstd"
        )

        msg = f"  {code}: {len(samples)} samples, {len(pct)} genes"
        if validate:
            corr = _validate_one(pct, ref / f"{code}.parquet")
            if corr is not None:
                corrs.append(corr)
                msg += f"  p95-corr={corr:.4f}"
        if len(samples) != len(source_samples) or sample_qc_fallback_reason:
            msg += f"  sample_qc={effective_sample_qc}:{len(samples)}/{len(source_samples)}"
            if sample_qc_fallback_reason:
                msg += f" ({sample_qc_fallback_reason})"
        if n_negative_values_clipped:
            msg += f"  clipped_negative_values={n_negative_values_clipped}"
        print(msg, flush=True)

    pd.DataFrame(provenance).to_csv(rep_dir / "_provenance.csv", index=False)
    if qc_manifest:
        pd.concat(qc_manifest, ignore_index=True).to_csv(
            out / "source-matrix-sample-qc.csv", index=False
        )
    pd.DataFrame(build_rows).to_csv(out / "expression-artifact-build-metadata.csv", index=False)
    metadata = {
        "artifact": "expression-derived-shards",
        "sample_qc": sample_qc,
        "sample_qc_policy_version": SAMPLE_EXPRESSION_QC_POLICY_VERSION,
        "sample_qc_fallbacks": int(
            sum(1 for row in build_rows if row.get("sample_qc_fallback_reason"))
        ),
        "sample_qc_manifest": "source-matrix-sample-qc.csv",
        "cohort_metadata": "expression-artifact-build-metadata.csv",
        "n_cohorts": len(build_rows),
        "n_source_samples": int(sum(row["n_source_samples"] for row in build_rows)),
        "n_cohort_samples": int(sum(row["n_cohort_samples"] for row in build_rows)),
        "n_negative_values_clipped": int(
            sum(row["n_negative_values_clipped"] for row in build_rows)
        ),
        "derived_artifacts": [
            "clean",
            _PCT_DS.gene_dir,
            _PCT_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE),
            SHARD_DATASETS["representatives"].gene_dir,
            _WS_DS.gene_dir,
            _WS_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE),
        ],
    }
    (out / "expression-artifact-build-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
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
    p.add_argument("--out", required=True, type=Path, help="Staging output dir (not oncoref/data)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N codes (a test run)")
    p.add_argument("--validate", action="store_true", help="Correlate vs the reference artifacts")
    p.add_argument(
        "--sample-qc",
        choices=("pass", "pass_or_warn", "all"),
        default="pass",
        help="Which source-matrix samples feed rebuilt artifacts (default: pass)",
    )
    args = p.parse_args(argv)
    rebuild(
        args.cache.expanduser(),
        args.ref.expanduser(),
        args.out.expanduser(),
        limit=args.limit,
        validate=args.validate,
        sample_qc=args.sample_qc,
    )


if __name__ == "__main__":
    main()
