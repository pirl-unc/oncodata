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

"""Regenerate **all** oncoref figures into one timestamped run directory.

Mirrors the pirlygenes ``analyses/regenerate_plots.py`` convention: every run
writes into a fresh ``outputs/run_<YYYYMMDD-HHMMSS>/`` snapshot (gitignored),
organised by plot family in subfolders, so a new run never overwrites an older
one. A ``latest`` symlink points at the most recent run; an ``index.md`` lists
what was produced (and what was skipped).

Each figure is independent: one that can't be drawn (missing per-sample matrix,
empty data) is reported and skipped, never aborting the batch. Expression-backed
figures need the relevant shards/matrices cached locally; the coverage/response
panels in particular need per-sample matrices (see
``plots._cached_per_sample_cohorts``), so they cover only the cached cohorts.

Usage::

    python scripts/regenerate_plots.py                 # all figures -> new run dir
    python scripts/regenerate_plots.py --out-dir DIR   # base dir override
    python scripts/regenerate_plots.py --no-timestamp  # write straight into base
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from oncoref import cta_curation_plots, plots  # noqa: E402
from oncoref.plots import _cached_per_sample_cohorts  # noqa: E402

CTA_TPM_THRESHOLDS = (25.0, 50.0)
WITHIN_SAMPLE_THRESHOLDS = (0.90, 0.95)
BURDEN_AXES = ("us_incidence", "us_mortality", "world_incidence", "world_mortality")
REFERENCE_AXES = ("tmb", "apd1", "ici", *BURDEN_AXES)


def _jobs() -> list[tuple[str, str, str, dict]]:
    """``(family, name, fn_attr, kwargs)`` for every figure, with sensible
    defaults. Coverage/response panels are restricted to the cohorts that
    actually have a cached per-sample matrix."""
    cached = sorted(_cached_per_sample_cohorts())
    jobs: list[tuple[str, str, str, dict]] = [
        # aPD-1 / ICI response. Match pirlygenes' two response scopes:
        # broad ICI anchors (PD-1 + proxies/fallbacks) and strict PD-1 monotherapy.
        ("apd1", "apd1_vs_tmb_ici", "apd1_vs_tmb", {"strict_pd1": False}),
        ("apd1", "apd1_vs_tmb_strict_pd1", "apd1_vs_tmb", {"strict_pd1": True}),
        ("apd1", "apd1_orr_bars_ici", "apd1_orr_bars", {"strict_pd1": False}),
        ("apd1", "apd1_orr_bars_strict_pd1", "apd1_orr_bars", {"strict_pd1": True}),
        (
            "apd1",
            "apd1_response_signature_antigen_presentation",
            "apd1_response_signature_scatter",
            {"signature": "antigen_presentation"},
        ),
        (
            "apd1",
            "apd1_response_signature_cytotoxic",
            "apd1_response_signature_scatter",
            {"signature": "cytotoxic"},
        ),
        (
            "apd1",
            "apd1_response_signature_t_cell_inflamed",
            "apd1_response_signature_scatter",
            {"signature": "t_cell_inflamed"},
        ),
        (
            "apd1",
            "apd1_response_signature_tgfb_exclusion",
            "apd1_response_signature_scatter",
            {"signature": "tgfb_exclusion"},
        ),
        # ICI regimens as distinct response sources and pooled estimate forest plots.
        ("ici", "ici_response_by_regimen_multi", "ici_response_by_regimen", {}),
        (
            "ici",
            "ici_response_by_regimen_all",
            "ici_response_by_regimen",
            {"only_multi": False},
        ),
        ("ici", "ici_regimen_comparison_all", "ici_regimen_comparison", {}),
        (
            "ici",
            "ici_regimen_comparison_multi",
            "ici_regimen_comparison",
            {"min_regimens": 2},
        ),
        ("ici", "ici_orr_pooled_forest_fallback", "ici_orr_pooled_forest", {}),
        ("ici", "ici_orr_pooled_forest_pd1", "ici_orr_pooled_forest", {"regimen": "PD-1"}),
        ("ici", "ici_orr_pooled_forest_pdl1", "ici_orr_pooled_forest", {"regimen": "PD-L1"}),
        (
            "ici",
            "ici_orr_pooled_forest_pd1_ctla4",
            "ici_orr_pooled_forest",
            {"regimen": "PD-1+CTLA-4"},
        ),
        # incidence / burden
        ("burden", "incidence_vs_mortality_us", "incidence_vs_mortality", {"region": "us"}),
        (
            "burden",
            "incidence_vs_mortality_world",
            "incidence_vs_mortality",
            {"region": "world"},
        ),
        ("burden", "burden_category_bars_us", "burden_category_bars", {"region": "us"}),
        (
            "burden",
            "burden_category_bars_world",
            "burden_category_bars",
            {"region": "world"},
        ),
        # CTA expression
        (
            "cta_expression",
            "cta_expression_heatmap_q1",
            "cta_expression_heatmap",
            {"stat": "q1"},
        ),
        (
            "cta_expression",
            "cta_expression_heatmap_median",
            "cta_expression_heatmap",
            {"stat": "median"},
        ),
        (
            "cta_expression",
            "cta_expression_heatmap_median_gene",
            "cta_expression_heatmap",
            {"stat": "median", "proteoform": False},
        ),
        (
            "cta_expression",
            "cta_expression_heatmap_q3",
            "cta_expression_heatmap",
            {"stat": "q3"},
        ),
        # CTA addressable burden + per-patient prevalence. Uses the faithful
        # per-sample source (cached cohorts); the portable within_sample source
        # needs the within-sample bundle, which isn't always cached locally.
    ]
    for axis in BURDEN_AXES:
        metric = f"{axis}_pct"
        jobs.extend(
            (
                "cta_addressable",
                f"cta_addressable_burden_within_sample_p{int(threshold * 100)}_{axis}",
                "cta_addressable_burden",
                {"source": "within_sample", "threshold": threshold, "metric": metric},
            )
            for threshold in WITHIN_SAMPLE_THRESHOLDS
        )
        jobs.extend(
            (
                "cta_addressable",
                f"cta_addressable_burden_per_sample_t{threshold:g}_{axis}",
                "cta_addressable_burden",
                {"source": "per_sample", "threshold_tpm": threshold, "metric": metric},
            )
            for threshold in CTA_TPM_THRESHOLDS
        )
    jobs.extend(
        (
            "cta_patient",
            f"cta_patient_count_heatmap_p{int(threshold * 100)}",
            "cta_patient_count_heatmap",
            {"threshold": threshold},
        )
        for threshold in WITHIN_SAMPLE_THRESHOLDS
    )
    jobs.extend(
        (
            "cta_patient",
            f"cta_patient_count_heatmap_t{threshold:g}",
            "cta_patient_count_heatmap",
            {"threshold_tpm": threshold},
        )
        for threshold in CTA_TPM_THRESHOLDS
    )
    for threshold in CTA_TPM_THRESHOLDS:
        for axis in REFERENCE_AXES:
            jobs.append(
                (
                    "cta_response",
                    f"cta_burden_vs_{axis}_t{threshold:g}",
                    "cta_burden_vs_response",
                    {"against": axis, "threshold_tpm": threshold},
                )
            )
        for axis in REFERENCE_AXES:
            jobs.append(
                (
                    "cta_response",
                    f"cta_specific_9mer_load_vs_{axis}_t{threshold:g}",
                    "cta_specific_9mer_load",
                    {"against": axis, "threshold_tpm": threshold},
                )
            )
    # Coverage panels need explicit cohort codes with cached per-sample matrices.
    if cached:
        for threshold in CTA_TPM_THRESHOLDS:
            jobs += [
                (
                    "cta_coverage",
                    f"cta_coverage_curves_t{threshold:g}",
                    "cta_coverage_curves",
                    {"cancer_types": cached, "threshold_tpm": threshold},
                ),
                (
                    "cta_coverage",
                    f"cta_coverage_stacked_bars_t{threshold:g}",
                    "cta_coverage_stacked_bars",
                    {"cancer_types": cached, "threshold_tpm": threshold},
                ),
            ]
    return jobs


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    base = args.out_dir.resolve() if args.out_dir else _REPO_ROOT / "outputs"
    base.mkdir(parents=True, exist_ok=True)
    if args.no_timestamp:
        return base
    run = args.run_name or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = base / run
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _update_latest(run_dir: Path) -> None:
    """Best-effort ``latest`` symlink next to the run dir."""
    link = run_dir.parent / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name)
    except OSError:
        pass  # symlinks may be unavailable (e.g. some Windows setups)


def _write_all_figures_pdf(run_dir: Path, generated: list[str]) -> Path | None:
    pngs = [run_dir / rel for rel in generated if rel.endswith(".png")]
    if not pngs:
        return None

    pdf = run_dir / "all-figures.pdf"
    with PdfPages(pdf) as pages:
        for png in pngs:
            image = mpimg.imread(png)
            height, width = image.shape[:2]
            aspect = width / height if height else 1
            fig_width = 11
            fig_height = max(4, min(11, fig_width / aspect + 0.6))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.imshow(image)
            ax.set_axis_off()
            fig.suptitle(png.relative_to(run_dir).as_posix(), fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pages.savefig(fig)
            plt.close(fig)
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="base output dir (default: ./outputs)"
    )
    ap.add_argument(
        "--run-name", default=None, help="run subfolder name (default: run_<timestamp>)"
    )
    ap.add_argument("--no-timestamp", action="store_true", help="write straight into the base dir")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args)
    jobs = _jobs()
    print(f"Regenerating {len(jobs)} figures into {run_dir}")

    done: list[str] = []
    skipped: list[tuple[str, str]] = []
    for family, name, fn_attr, kwargs in jobs:
        fam_dir = run_dir / family
        fam_dir.mkdir(parents=True, exist_ok=True)
        out = fam_dir / f"{name}.png"
        try:
            getattr(plots, fn_attr)(save=out, **kwargs)
            done.append(f"{family}/{name}.png")
            print(f"  ok    {family}/{name}.png")
        except Exception as e:
            skipped.append((f"{family}/{name}", f"{type(e).__name__}: {e}"))
            print(f"  SKIP  {family}/{name}  ({type(e).__name__}: {e})", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    curation_dir = run_dir / "cta_curation"
    try:
        result = cta_curation_plots.render(out_dir=curation_dir)
        for path in result["paths"].values():
            done.append(f"cta_curation/{path.name}")
            print(f"  ok    cta_curation/{path.name}")
    except Exception as e:
        skipped.append(("cta_curation", f"{type(e).__name__}: {e}"))
        print(f"  SKIP  cta_curation  ({type(e).__name__}: {e})", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    pdf = _write_all_figures_pdf(run_dir, done)
    if pdf is not None:
        print(f"  ok    {pdf.name}")

    index = run_dir / "index.md"
    lines = [
        f"# oncoref figures — {run_dir.name}",
        "",
        f"{len(done)} generated, {len(skipped)} skipped.",
        "",
        f"Combined PDF: `{pdf.name}`" if pdf is not None else "Combined PDF: not generated.",
        "",
        "## Generated",
        *(f"- `{p}`" for p in done),
    ]
    if skipped:
        lines += ["", "## Skipped", *(f"- `{n}` — {why}" for n, why in skipped)]
    index.write_text("\n".join(lines) + "\n")

    _update_latest(run_dir)
    print(f"\n{len(done)} figures written, {len(skipped)} skipped. Index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
